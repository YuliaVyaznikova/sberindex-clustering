import numpy as np
import pandas as pd
from scipy import sparse

from .features import read_package

SPEND_SHARES = ["share_food", "share_marketplaces", "share_transport", "share_health", "share_cafe", "share_other"]
EMP_SHARES = ["emp_agri", "emp_mining", "emp_manufacturing", "emp_utilities", "emp_construction", "emp_trade", "emp_transport", "emp_services", "emp_public"]
PLAIN = ["spend_real", "wage_real", "log_population", "urban_share", "log_density", "share_young", "share_old", "migration"]
BLOCKS = {
    "spending": SPEND_SHARES + ["spend_real"],
    "labor": EMP_SHARES + ["wage_real", "emp_rate"],
    "place": ["log_population", "urban_share", "log_density", "share_young", "share_old", "migration", "market_access"],
}
CITIES = {"Москва": 9001, "Санкт-Петербург": 9002}
QUARTER_END = {"Q1": "03", "Q2": "06", "Q3": "09", "Q4": "12"}
EARTH_RADIUS_KM = 6371


def load_data(config):
    processed = config["paths"]["processed"]
    features = pd.read_parquet(processed / config["network"]["features"])
    territories = pd.read_csv(processed / "territories.csv", index_col=0)
    monthly = pd.read_parquet(processed / "spending_monthly.parquet")
    settings = config["network"]
    if settings["cities"] == "merge":
        features, territories, monthly = merge_cities(features, territories, monthly, settings["city_coverage"])
    elif settings["cities"] == "drop":
        keep = ~territories["city_district"]
        features = features[features["territory_id"].map(keep)]
        territories = territories[keep]
    if settings["complete"]:
        features = features[features["complete"]]
    return {"features": features, "territories": territories, "monthly": monthly}


def merge_cities(features, territories, monthly, coverage):
    districts = territories[territories["city_district"]]
    city_of = districts["region"].map(CITIES)
    part = features[features["territory_id"].isin(districts.index)].assign(city=lambda t: t["territory_id"].map(city_of))
    columns = [c for c in features.columns if c not in ("territory_id", "period", "months", "complete")]
    rows = []
    for (city, period), group in part.groupby(["city", "period"]):
        full = group[group["complete"]]
        if full.empty:
            continue
        row = {c: np.average(full[c], weights=full["population"]) for c in columns}
        covered = full["population"].sum() / group["population"].sum()
        row.update(territory_id=city, period=period, months=full["months"].min(), complete=bool(covered >= coverage))
        row["population"] = group["population"].sum()
        row["emp_total"] = group["emp_total"].sum()
        row["emp_rate"] = row["emp_total"] / row["population"]
        row["log_population"] = np.log(row["population"])
        rows.append(row)
    features = pd.concat([features[~features["territory_id"].isin(districts.index)], pd.DataFrame(rows)], ignore_index=True)

    population = part.groupby("territory_id")["population"].mean()
    totals = monthly.pivot(index="date", columns="territory_id", values="total")
    extra = []
    for region, city in CITIES.items():
        members = [d for d in districts.index[districts["region"] == region] if d in totals.columns]
        weight = population.reindex(members)
        values = totals[members]
        series = (values * weight).sum(axis=1) / (values.notna() * weight).sum(axis=1)
        extra.append(pd.DataFrame({"territory_id": city, "date": series.index, "total": series.to_numpy()}))
    monthly = pd.concat([monthly[~monthly["territory_id"].isin(districts.index)], *extra], ignore_index=True)

    nodes = []
    for region, city in CITIES.items():
        own = districts[districts["region"] == region]
        nodes.append(pd.Series({
            "name": region, "type": "город федерального значения", "region": region,
            "region_code": own["region_code"].iloc[0], "lat": own["lat"].mean(), "lon": own["lon"].mean(),
            "area_km2": own["area_km2"].sum(), "capital": True, "city_district": False,
            "full_series": own["full_series"].all(), "members": ",".join(map(str, own.index)),
        }, name=city))
    territories = pd.concat([territories[~territories["city_district"]], pd.DataFrame(nodes)])
    return features, territories, monthly


def transform(features):
    result = pd.DataFrame(index=features.index)
    logs = np.log(features[SPEND_SHARES].clip(lower=1e-4))
    result[SPEND_SHARES] = logs.sub(logs.mean(axis=1), axis=0)
    result[EMP_SHARES] = np.sqrt(features[EMP_SHARES].clip(lower=0))
    result["emp_rate"] = np.log(features["emp_rate"])
    result["market_access"] = np.log(features["market_access"])
    result[PLAIN] = features[PLAIN]
    result = result.clip(result.quantile(0.01), result.quantile(0.99), axis=1)
    return (result - result.mean()) / result.std()


def attributes(data, blocks, period):
    features = data["features"]
    scaled = transform(features)
    rows = features["period"] == period
    parts = [scaled.loc[rows, BLOCKS[name]].to_numpy() / np.sqrt(len(BLOCKS[name])) for name in blocks]
    return features.loc[rows, "territory_id"].to_numpy(), np.hstack(parts)


def cosine(x):
    unit = x / np.linalg.norm(x, axis=1, keepdims=True)
    return unit @ unit.T


def symmetric(rows, cols, values, n):
    w = sparse.coo_matrix((values, (rows, cols)), shape=(n, n)).tocsr()
    w = w.maximum(w.T)
    w.setdiag(0)
    w.eliminate_zeros()
    return w


def knn(similarity, k, mutual=False):
    n = similarity.shape[0]
    similarity = similarity.copy()
    np.fill_diagonal(similarity, -np.inf)
    rows = np.repeat(np.arange(n), k)
    cols = np.argpartition(-similarity, k, axis=1)[:, :k].ravel()
    values = np.clip(similarity[rows, cols], 1e-6, None)
    if not mutual:
        return symmetric(rows, cols, values, n)
    w = sparse.coo_matrix((values, (rows, cols)), shape=(n, n)).tocsr()
    w = w.minimum(w.T)
    w.eliminate_zeros()
    return w


def threshold(similarity, mean_degree):
    n = similarity.shape[0]
    eps = np.sort(similarity[np.triu_indices(n, 1)])[-int(mean_degree * n / 2)]
    rows, cols = np.nonzero(np.triu(similarity >= eps, 1))
    return symmetric(rows, cols, similarity[rows, cols], n)


def growth_correlation(data, ids, period, months=12):
    end = period[:4] + "-" + QUARTER_END[period[4:]]
    totals = data["monthly"].pivot(index="date", columns="territory_id", values="total").sort_index()
    totals = totals.loc[:end].tail(months + 1)[ids]
    growth = np.log(totals).diff().iloc[1:]
    residual = growth.sub(growth.median(axis=1), axis=0).fillna(0)
    return np.corrcoef(residual.to_numpy().T)


def great_circle(territories, ids):
    lat = np.radians(territories.loc[ids, "lat"].to_numpy(dtype=float))
    lon = np.radians(territories.loc[ids, "lon"].to_numpy(dtype=float))
    a = np.sin((lat[:, None] - lat[None, :]) / 2) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin((lon[:, None] - lon[None, :]) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def road_distances(config, data, ids):
    territories = data["territories"]
    roads = read_package(config, "connection")
    roads = roads[roads["type"] == "highway"]
    node_of = {tid: tid for tid in ids}
    if "members" in territories.columns:
        for city, members in territories["members"].dropna().items():
            node_of.update({int(m): city for m in members.split(",")})
    position = {tid: i for i, tid in enumerate(ids)}
    pairs = pd.DataFrame({
        "x": roads["territory_id_x"].map(node_of).map(position),
        "y": roads["territory_id_y"].map(node_of).map(position),
        "d": roads["distance"],
    }).dropna()
    pairs = pairs[pairs["x"] != pairs["y"]].astype({"x": int, "y": int}).groupby(["x", "y"], as_index=False)["d"].min()
    straight = great_circle(territories, ids)
    road = np.full_like(straight, np.nan)
    road[pairs["x"], pairs["y"]] = pairs["d"]
    road = np.fmin(road, road.T)
    known = ~np.isnan(road) & (straight > 1)
    detour = float(np.median(road[known] / straight[known]))
    distance = np.where(np.isnan(road), straight * detour, road)
    np.fill_diagonal(distance, 0)
    return distance, detour


def proximity(distance, scale):
    return np.exp(-distance / scale)


def snapshot(config, data, period):
    settings = config["network"]
    ids, x = attributes(data, settings["attribute_blocks"], period)
    _, edge_x = attributes(data, settings["edge_blocks"], period)
    graph = knn(cosine(edge_x), settings["k"])
    attribute_graph = knn(cosine(x), settings["k"])
    return {"ids": ids, "x": x, "graph": graph, "attribute_graph": attribute_graph, "edge_similarity": cosine(edge_x)}


def edge_variants(config, data, period):
    settings = config["network"]
    base = snapshot(config, data, period)
    similarity, k = base["edge_similarity"], settings["k"]
    distance, detour = road_distances(config, data, base["ids"])
    nearest = float(np.median(np.sort(distance, axis=1)[:, k]))
    variants = {
        f"spend_knn{k // 2}": knn(similarity, k // 2),
        f"spend_knn{k}": base["graph"],
        f"spend_knn{2 * k}": knn(similarity, 2 * k),
        f"spend_mutual_knn{k}": knn(similarity, k, mutual=True),
        "spend_threshold": threshold(similarity, base["graph"].getnnz() / len(base["ids"])),
        f"full_knn{k}": base["attribute_graph"],
        f"geo_knn{k}": knn(proximity(distance, nearest), k),
        f"hybrid_knn{k}": knn(np.clip(similarity, 0, None) * proximity(distance, settings["hybrid_scale_km"]), k),
    }
    if period >= data["monthly"]["date"].min()[:4] + "Q4":
        variants[f"dynamics_knn{k}"] = knn(growth_correlation(data, list(base["ids"]), period), k)
    return base, variants, distance, {"detour": detour, "geo_scale_km": nearest}