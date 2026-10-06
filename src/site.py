import json

import geopandas
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from shapely.geometry import MultiPolygon

from .economy import marketplaces
from .network import CITIES, EARTH_RADIUS_KM, load_data, snapshot

MAP_CRS = "+proj=aea +lat_1=52 +lat_2=64 +lat_0=0 +lon_0=100 +datum=WGS84 +units=m"
TRAITS = {
    "share_food": ("еда", "доля еды в тратах {:.0%}"),
    "share_marketplaces": ("маркетплейсы", "доля маркетплейсов в тратах {:.0%}"),
    "share_cafe": ("общепит", "доля общепита в тратах {:.1%}"),
    "share_transport": ("транспорт", "доля транспорта в тратах {:.1%}"),
    "share_health": ("здоровье", "доля здоровья в тратах {:.1%}"),
    "spend_real": ("траты", "траты на жителя {:.2f} медианы"),
    "wage_real": ("зарплата", "зарплата {:.2f} медианы"),
    "emp_rate": ("занятые на жителя", "работников на жителя {:.2f}"),
    "emp_agri": ("агро", "в сельском хозяйстве {:.0%} занятых"),
    "emp_manufacturing": ("обработка", "в обрабатывающей промышленности {:.0%} занятых"),
    "emp_public": ("бюджет", "в бюджетной сфере {:.0%} занятых"),
    "emp_services": ("услуги", "в услугах {:.0%} занятых"),
    "urban_share": ("горожане", "горожан {:.0%}"),
    "log_density": ("плотность", "плотность {:.3g} чел. на км2"),
    "share_young": ("моложе трудоспособного", "моложе трудоспособного возраста {:.0%}"),
    "share_old": ("старше трудоспособного", "старше трудоспособного возраста {:.0%}"),
}
METHODS = {
    "kmeans": "k-means", "ward": "Ward", "gmm": "GMM", "spectral": "spectral", "fused_spectral": "fused spectral",
    "kefrin_euclidean": "KEFRiN euclidean", "kefrin_cosine": "KEFRiN cosine", "refined_spectral w0.7": "refined spectral",
}


def encode(geometry, scale):
    polygons = geometry.geoms if isinstance(geometry, MultiPolygon) else [geometry]
    parts, box = [], [np.inf, np.inf, -np.inf, -np.inf]
    for polygon in polygons:
        for ring in [polygon.exterior, *polygon.interiors]:
            points = np.round(np.asarray(ring.coords) * [1, -1] * scale / 1000).astype(int)
            keep = np.r_[True, np.any(np.diff(points, axis=0) != 0, axis=1)]
            points = points[keep]
            if len(points) > 1 and (points[0] == points[-1]).all():
                points = points[:-1]
            if len(points) < 3:
                continue
            box = [min(box[0], points[:, 0].min()), min(box[1], points[:, 1].min()), max(box[2], points[:, 0].max()), max(box[3], points[:, 1].max())]
            steps = np.diff(points, axis=0).ravel()
            parts.append(f"M{points[0, 0]} {points[0, 1]}l" + " ".join(map(str, steps)) + "z")
    return "".join(parts), [int(v) for v in box]


def build_shapes(config, territories, ids):
    settings = config["site"]
    shapes = geopandas.read_parquet(config["paths"]["prepared"] / "shapes.parquet").to_crs(MAP_CRS)
    shapes = shapes.drop_duplicates("territory_id")
    districts = territories[territories["city_district"]]
    shapes["territory_id"] = [CITIES.get(districts["region"].get(t), t) if t in districts.index else t for t in shapes["territory_id"]]
    shapes = shapes.dissolve("territory_id").reset_index()
    shapes["geometry"] = shapes["geometry"].simplify(settings["tolerance_km"] * 1000)
    shapes = shapes[~shapes["geometry"].is_empty]
    paths, boxes, anchors = {}, {}, {}
    for t, geometry in zip(shapes["territory_id"], shapes["geometry"]):
        d, box = encode(geometry, settings["scale"])
        if d:
            paths[int(t)], boxes[int(t)] = d, box
            point = geometry.representative_point()
            anchors[int(t)] = [round(point.x * settings["scale"] / 1000), round(-point.y * settings["scale"] / 1000)]
    x0, y0 = min(b[0] for b in boxes.values()), min(b[1] for b in boxes.values())
    x1, y1 = max(b[2] for b in boxes.values()), max(b[3] for b in boxes.values())
    missing = set(ids) - set(paths)
    print(f"{len(paths)} shapes, {len(missing)} typed municipalities without shape")
    return {"box": [x0, y0, x1, y1], "paths": paths, "boxes": boxes, "anchors": anchors}


def build_traits(summary, profiles, top):
    result = {}
    for t in summary.index:
        z = profiles.loc[t, list(TRAITS)]
        rows = z.abs().sort_values(ascending=False).index[:top]
        result[int(t)] = [{"text": TRAITS[c][1].format(summary.loc[t, TRAITS[c][0]]), "up": bool(z[c] > 0)} for c in rows]
    return result


def great_circle(a, b):
    lat1, lon1, lat2, lon2 = map(np.radians, (a[0], a[1], b[0], b[1]))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(h)))


def build_network(config, data, assigned, period, anchors, network_stats, neighbors=10):
    shot = snapshot(config, data, period)
    ids = [int(t) for t in shot["ids"]]
    matrix = shot["graph"].tocsr()
    upper = sparse.triu(matrix, k=1).tocoo()
    graph = nx.Graph()
    graph.add_nodes_from(range(len(ids)))
    graph.add_weighted_edges_from(zip(upper.row.tolist(), upper.col.tolist(), upper.data.tolist()))
    position = nx.spring_layout(graph, k=0.6 / np.sqrt(len(ids)), iterations=400, weight="weight", seed=0)
    xy = np.array([position[i] for i in range(len(ids))])
    xy = (xy - xy.min(axis=0)) / (xy.max(axis=0) - xy.min(axis=0))
    territories = data["territories"]
    where = {t: (territories.at[t, "lat"], territories.at[t, "lon"]) for t in ids if t in territories.index and pd.notna(territories.at[t, "lat"])}
    lists, distances, own = [], [], 0
    for i, t in enumerate(ids):
        row = matrix.getrow(i)
        top = row.indices[np.argsort(-row.data)][:neighbors]
        pairs = []
        for j in top:
            u = ids[j]
            km = round(great_circle(where[t], where[u])) if t in where and u in where else -1
            pairs.append([int(j), km])
            if km >= 0:
                distances.append(km)
            own += int(territories.at[u, "region"] == territories.at[t, "region"])
        lists.append(pairs)
    stats = {
        "nodes": len(ids), "edges": graph.number_of_edges(), "neighbors": neighbors,
        "degree_min": int(network_stats["degree_min"]), "degree_max": int(network_stats["degree_max"]),
        "clustering": round(float(network_stats["clustering"]), 2), "clustering_random": round(float(network_stats["clustering_rewired"]), 3),
        "median_km": int(round(float(np.median(distances)), -1)), "own_region": round(own / (neighbors * len(ids)), 2),
    }
    print(f"network: median distance to the {neighbors} nearest by spending {np.median(distances):.0f} km, in the own region {own / (neighbors * len(ids)):.3f}")
    return {
        "period": period, "nodes": [[t, round(float(x), 4), round(float(y), 4), int(assigned.get(t, -1))] for t, (x, y) in zip(ids, xy)],
        "edges": [[int(i), int(j)] for i, j in graph.edges()], "neighbors": lists, "anchors": anchors, "stats": stats,
    }


def build_extra(data, assigned, period, gap, changes):
    features = data["features"]
    rows = features[features["period"] == period].set_index("territory_id")
    rows = rows.loc[rows.index.intersection(assigned.index)]
    passport = {}
    for t, r in rows.iterrows():
        passport[int(t)] = [
            int(round(float(r["population"]))), round(float(np.exp(r["spend_real"])), 2),
            round(float(np.exp(r["wage_real"])), 2) if pd.notna(r["wage_real"]) else None,
            round(float(r["share_marketplaces"]), 3), round(float(gap[t]), 3) if t in gap.index else None,
        ]
    typical = rows.groupby(assigned.loc[rows.index]).agg(spend=("spend_real", "median"), wage=("wage_real", "median"), market=("share_marketplaces", "median"))
    changes = changes.drop(columns="название")
    return {
        "passport": passport,
        "typical": {int(k): [round(float(np.exp(v["spend"])), 2), round(float(np.exp(v["wage"])), 2), round(float(v["market"]), 3)] for k, v in typical.iterrows()},
        "changes": {"columns": list(changes.columns), "rows": {str(int(k)): [round(float(x), 2) for x in v] for k, v in changes.iterrows()}},
    }


def site(config):
    out, target = config["paths"]["results"], config["paths"]["results"].parent / "site" / "data"
    target.mkdir(parents=True, exist_ok=True)
    settings, describe = config["site"], config["describe"]
    types = pd.read_csv(out / "types.csv")
    confidence = pd.read_csv(out / "confidence.csv")
    robust = pd.read_csv(out / "robust_transitions.csv")
    summary = pd.read_csv(out / "summary.csv", index_col="type")
    profiles = pd.read_csv(out / "profiles.csv", index_col="type")
    external = pd.read_csv(out / "validate_external_medians.csv", index_col=0)
    transitions = pd.read_csv(out / "transitions.csv", index_col=0)
    methods = pd.read_csv(out / "methods.csv")
    ranking = pd.read_csv(out / "validate_ranking.csv", index_col=0)
    truths = pd.read_csv(out / "synthetic_truths_ari.csv", index_col="method")
    territories = pd.read_csv(config["paths"]["processed"] / "territories.csv", index_col=0)
    periods = sorted(types["period"].unique())
    first, last = periods[0], periods[-1]
    order = describe["order"]
    names = {int(k): v for k, v in describe["names"].items()}
    short = {int(k): v for k, v in describe["short_names"].items()}
    colors = dict(zip(order, describe["colors"]))

    wide = types.pivot(index="territory_id", columns="period", values="type")
    certain = confidence.pivot(index="territory_id", columns="period", values="confidence")
    flags = robust.set_index("territory_id")["доля прогонов со сменой типа"]
    info = types.drop_duplicates("territory_id", keep="last").set_index("territory_id")
    records = []
    for t in wide.index:
        kinds = [-1 if pd.isna(wide.loc[t, p]) else int(wide.loc[t, p]) for p in periods]
        certainty = [-1 if pd.isna(certain.loc[t].get(p)) else round(float(certain.loc[t, p]), 2) for p in periods]
        records.append([int(t), info.loc[t, "name"], info.loc[t, "region"], kinds, certainty, round(float(flags[t]), 2) if t in flags.index else 0])

    shapes = build_shapes(config, territories, wide.index)
    cards = []
    for t in order:
        row = summary.loc[t]
        cards.append({
            "id": t, "name": names[t], "short": short[t], "color": colors[t], "count": int(row["МО"]), "share": float(row["доля населения"]),
            "population": float(row["население, млн"]), "traits": build_traits(summary, profiles, settings["traits"])[t],
            "external": {c: float(external.loc[short[t], c]) for c in ["retail", "housing", "investment"]},
        })

    monthly = pd.read_parquet(config["paths"]["processed"] / "spending_monthly.parquet")
    shares, _, count = marketplaces(monthly, types, last)
    market = {"dates": [str(d)[:7] for d in shares.index], "count": int(count), "series": {int(t): [round(float(v), 4) for v in shares[t]] for t in order}}

    six = methods[methods["k"] == 6]
    points = [{"method": r.method, "alpha": None if pd.isna(r.alpha) else float(r.alpha), "sw": round(float(r.SW), 4), "mq": round(float(r.MQ), 4)} for r in six.itertuples()]
    worst = truths[[c for c in truths.columns if c.endswith("calibrated")]].min(axis=1)
    bars = [{"name": METHODS[m], "ari": round(float(worst[m]), 3), "model": m == "refined_spectral w0.7"} for m in METHODS]

    matrix = transitions.to_numpy()
    total, changed = int(matrix.sum()), int(matrix.sum() - np.trace(matrix))
    ids = {v: k for k, v in names.items()}
    solid = np.zeros_like(matrix)
    for a, b in zip(robust["из типа"].map(ids), robust["в тип"].map(ids)):
        solid[a, b] += 1
    moves = [[int(r["territory_id"]), r["МО"], r["регион"], ids[r["из типа"]], ids[r["в тип"]], round(float(r["доля прогонов со сменой типа"]), 2)] for _, r in robust.sort_values("доля прогонов со сменой типа", ascending=False).iterrows()]
    model = ranking[(ranking["method"] == "refined_spectral") & (ranking["alpha"] == config["model"]["alpha"])].iloc[0]
    data = {
        "periods": periods, "first": first, "last": last, "order": order, "colors": colors, "names": names, "short": short, "cards": cards,
        "mo": records, "matrix": matrix.tolist(), "solid": solid.tolist(), "moves": moves, "total": total, "changed": changed, "robust": int(len(robust)),
        "market": market, "methods": points, "ari": bars,
        "tiles": {"types": len(order), "worst": int(model["worst"]), "ari_model": round(float(worst["refined_spectral w0.7"]), 2), "ari_kmeans": round(float(worst["kmeans"]), 2), "configs": int(len(six)), "robust": int(len(robust))},
    }
    dump(target / "atlas.json", data)
    dump(target / "shapes.json", {"box": shapes["box"], "paths": shapes["paths"], "boxes": shapes["boxes"]})
    data = load_data(config)
    assigned = types[types["period"] == last].set_index("territory_id")["type"]
    network_stats = pd.read_csv(out / "validate_network.csv", index_col=0)["value"]
    dump(target / "network.json", build_network(config, data, assigned, last, shapes["anchors"], network_stats))
    gap = pd.read_csv(out / "spending_gap.csv", index_col="territory_id")["gap"]
    dump(target / "extra.json", build_extra(data, assigned, last, gap, pd.read_csv(out / "changes.csv", index_col="type")))
    for path in sorted(target.glob("*.json")):
        print(path.name, round(path.stat().st_size / 1024), "KB")


def dump(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))