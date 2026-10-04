import copy
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.spatial.distance import squareform
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from sklearn.model_selection import cross_val_score

from . import methods
from .compare import agreement, subsamples
from .describe import INK, MUTED, PROFILE, style
from .features import load_territories, match, read_prepared
from .network import BLOCKS, load_data, snapshot, transform

LABELS = {
    **PROFILE, "share_other": "доля прочего", "emp_utilities": "занятость: энергетика и ЖКХ",
    "emp_construction": "занятость: стройка", "emp_trade": "занятость: торговля", "emp_transport": "занятость: транспорт",
}
BLOCK_COLORS = {"spending": "#2a78d6", "labor": "#eb6834", "place": "#1baf7a"}
ABLATION = {
    "все блоки": (["spending", "labor", "place"], ["spending"]),
    "без потребления в атрибутах": (["labor", "place"], ["spending"]),
    "без труда": (["spending", "place"], ["spending"]),
    "без места": (["spending", "labor"], ["spending"]),
    "только потребление": (["spending"], ["spending"]),
    "рёбра по всем признакам": (["spending", "labor", "place"], ["spending", "labor", "place"]),
}
INDICES = {"SW": 1, "CH": 1, "S_Dbw": -1, "AVI": 1, "AVU": -1, "MQ": 1}


def window_types(types, period, ids):
    return types[types["period"] == period].set_index("territory_id")["type"].reindex(ids)


def single_window(config, data, period, attribute_blocks, edge_blocks):
    local = copy.deepcopy(config)
    local["network"]["attribute_blocks"], local["network"]["edge_blocks"] = attribute_blocks, edge_blocks
    shot = snapshot(local, data, period)
    labels = methods.fused_spectral(shot["graph"], shot["attribute_graph"], config["model"]["k"], config["model"]["alpha"])
    return pd.Series(labels, index=shot["ids"])


def check_blocks(config, data, period):
    variants = {name: single_window(config, data, period, *blocks) for name, blocks in ABLATION.items()}
    base = variants["все блоки"]
    return pd.Series({name: adjusted_rand_score(base, labels.reindex(base.index)) for name, labels in variants.items()}, name="ari_with_all_blocks")


def check_importance(features, final, period):
    columns = [c for block in BLOCKS.values() for c in block]
    rows = features[features["period"] == period].set_index("territory_id").loc[final.index, columns]
    forest = RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1)
    accuracy = cross_val_score(forest, rows, final, cv=5).mean()
    forest.fit(rows, final)
    result = permutation_importance(forest, rows, final, n_repeats=10, random_state=0, n_jobs=-1)
    table = pd.DataFrame({
        "feature": columns, "block": [b for b, block in BLOCKS.items() for _ in block],
        "importance": result.importances_mean, "std": result.importances_std,
    })
    return table.sort_values("importance", ascending=False), accuracy


def membership(codes, k):
    return sparse.csr_matrix((np.ones(len(codes)), (np.arange(len(codes)), codes)), shape=(len(codes), k))


def mean_path(a, rng, sources=400):
    sample = rng.choice(a.shape[0], min(sources, a.shape[0]), replace=False)
    distances = shortest_path(a, unweighted=True, directed=False, indices=sample)
    return distances[np.isfinite(distances) & (distances > 0)].mean()


def network_summary(w, rng, rewirings=5):
    a = (w > 0).astype(float).tocsr()
    graph = nx.from_scipy_sparse_array(a)
    degree = np.asarray(a.sum(axis=1)).ravel()
    clustering, path = nx.average_clustering(graph), mean_path(a, rng)
    null_clustering, null_path = [], []
    for seed in range(rewirings):
        rewired = graph.copy()
        nx.double_edge_swap(rewired, nswap=10 * rewired.number_of_edges(), max_tries=10 ** 7, seed=seed)
        null_clustering.append(nx.average_clustering(rewired))
        null_path.append(mean_path(nx.to_scipy_sparse_array(rewired, format="csr"), rng))
    return {
        "nodes": a.shape[0], "edges": a.nnz // 2, "degree_min": degree.min(), "degree_median": np.median(degree),
        "degree_max": degree.max(), "components": connected_components(a, directed=False)[0],
        "clustering": clustering, "clustering_rewired": np.mean(null_clustering),
        "mean_path": path, "mean_path_rewired": np.mean(null_path),
        "small_world_sigma": (clustering / np.mean(null_clustering)) / (path / np.mean(null_path)),
        "degree_assortativity": nx.degree_assortativity_coefficient(graph),
    }


def mixing(w, codes, k):
    member = membership(codes, k)
    between = (member.T @ w @ member).toarray()
    total = between.sum()
    return between, between.trace() / total, ((between.sum(axis=1) / total) ** 2).sum()


def moran(w, values, rng, permutations=499):
    rows = sparse.diags(1 / np.asarray(w.sum(axis=1)).ravel()) @ w

    def statistic(z):
        return len(z) / rows.sum() * (z @ (rows @ z)) / (z @ z)

    z = values - values.mean()
    observed = statistic(z)
    null = np.array([statistic(rng.permutation(z)) for _ in range(permutations)])
    return observed, (np.sum(null >= observed) + 1) / (permutations + 1)


def participation(w, codes, k):
    by_type = (w @ membership(codes, k)).toarray()
    share = by_type / by_type.sum(axis=1, keepdims=True)
    own = share[np.arange(len(codes)), codes]
    outside = share.copy()
    outside[np.arange(len(codes)), codes] = -1
    return 1 - (share ** 2).sum(axis=1), own, outside.argmax(axis=1)


def check_network(config, data, shot, final, confidence, period, rng):
    k = config["model"]["k"]
    keep = final.notna().to_numpy()
    ids, w, codes = shot["ids"][keep], shot["graph"][keep][:, keep].tocsr(), final[keep].astype(int).to_numpy()
    territories = data["territories"]
    summary = network_summary(w, rng)
    between, within, expected = mixing(w, codes, k)
    regions, _ = pd.factorize(territories.loc[ids, "region"])
    _, region_within, region_expected = mixing(w, regions, regions.max() + 1)
    summary.update({"type_within_weight": within, "type_within_expected": expected, "region_within_weight": region_within, "region_within_expected": region_expected})

    features = data["features"]
    rows = features["period"] == period
    scaled = transform(features).loc[rows].set_axis(features.loc[rows, "territory_id"].to_numpy()).loc[ids]
    table = []
    for block, columns in BLOCKS.items():
        for column in columns:
            value, p_value = moran(w, scaled[column].to_numpy(), rng)
            table.append({"feature": column, "block": block, "moran_i": value, "p": p_value})
    morans = pd.DataFrame(table).sort_values("moran_i", ascending=False)

    p, own, other = participation(w, codes, k)
    bridges = pd.DataFrame({
        "name": territories.loc[ids, "name"].to_numpy(), "region": territories.loc[ids, "region"].to_numpy(),
        "type": codes, "participation": p, "own_type_share": own, "main_other_type": other,
        "confidence": confidence.reindex(ids).to_numpy(),
    }, index=pd.Index(ids, name="territory_id"))
    return pd.Series(summary), between / between.sum(axis=1, keepdims=True), morans, bridges


def check_changes(config, data, types, confidence, first, last):
    k = config["model"]["k"]
    shot = snapshot(config, data, first)
    early = window_types(types, first, shot["ids"])
    keep = early.notna().to_numpy()
    ids, w = shot["ids"][keep], shot["graph"][keep][:, keep].tocsr()
    p, own, _ = participation(w, early[keep].astype(int).to_numpy(), k)
    frame = pd.DataFrame({"participation": p, "own_type_share": own}, index=ids)
    frame["confidence"] = confidence[confidence["period"] == first].set_index("territory_id")["confidence"].reindex(ids)
    late = types[types["period"] == last].set_index("territory_id")["type"]
    frame = frame.loc[frame.index.intersection(late.index)].dropna()
    moved = (early[frame.index] != late[frame.index]).astype(int).to_numpy()
    rows = []
    for name, score in [("1 - confidence", 1 - frame["confidence"]), ("participation", frame["participation"]), ("1 - own_type_share", 1 - frame["own_type_share"])]:
        rows.append({"score": name, "auc": roc_auc_score(moved, score), "p": mannwhitneyu(score[moved == 1], score[moved == 0]).pvalue})
    bands = pd.cut(frame["confidence"], [0, 0.7, 0.9, 1.0], include_lowest=True)
    shares = pd.Series(moved, index=frame.index).groupby(bands, observed=True).agg(["size", "mean"])
    return pd.DataFrame(rows), shares.rename(columns={"size": "municipalities", "mean": "changed_share"}), int(moved.sum()), len(moved)


def with_k(method, w, k, steps):
    for resolution in np.geomspace(0.05, 3.0, steps):
        labels = method(w, resolution)
        if labels.max() + 1 == k:
            return labels, resolution
    return None, None


def partitions(config, x, w, wa):
    k = config["model"]["k"]
    result = {name: method(x, k) for name, method in methods.ATTRIBUTE_METHODS.items()}
    result["spectral"] = methods.spectral(w, k)
    for name, method in methods.GRAPH_METHODS.items():
        labels, _ = with_k(method, w, k, config["compare"]["resolutions"])
        if labels is not None:
            result[name] = labels
    for name in methods.JOINT_METHODS:
        for alpha in config["validate"]["consensus_alphas"]:
            result[f"{name} {alpha}"] = methods.joint(name, x, w, wa, k, alpha)
    return result


def coassociation(parts):
    n = len(parts[0])
    total = np.zeros((n, n), dtype=np.float32)
    for labels in parts:
        member = membership(labels, labels.max() + 1).astype(np.float32)
        total += (member @ member.T).toarray()
    return total / len(parts)


def check_consensus(parts, k):
    names = list(parts)
    ari = pd.DataFrame([[adjusted_rand_score(parts[a], parts[b]) for b in names] for a in names], index=names, columns=names)
    together = coassociation([labels for name, labels in parts.items() if name != "multilayer"])
    agreed = methods.spectral(sparse.csr_matrix(together), k)
    summary = pd.DataFrame({
        "mean_ari_with_others": (ari.sum(axis=1) - 1) / (len(names) - 1),
        "ari_with_consensus": pd.Series({name: adjusted_rand_score(agreed, labels) for name, labels in parts.items()}),
    }).sort_values("mean_ari_with_others", ascending=False)
    final = parts["multilayer"]
    same = final[:, None] == final[None, :]
    np.fill_diagonal(same, False)
    support = (together * same).sum(axis=1) / np.maximum(same.sum(axis=1), 1)
    return ari, summary, support


def external_values(config, data, period):
    features = data["features"]
    population = features[features["period"] == period].set_index("territory_id")["population"]
    _, oktmo = load_territories(config, population.index)
    result = {}
    for name, spec in config["external"]["indicators"].items():
        table = match(read_prepared(config, f"external_{name}.csv.gz"), oktmo, [])
        latest = table.sort_values("year").drop_duplicates("territory_id", keep="last").set_index("territory_id")["value"]
        values = latest * spec["scale"] / population.reindex(latest.index)
        result[name] = values.replace([np.inf, -np.inf], np.nan).dropna()
    return result


def eta_squared_h(values, labels):
    groups = [values[labels == g] for g in np.unique(labels)]
    return (kruskal(*groups).statistic - len(groups) + 1) / (len(values) - len(groups))


def check_external(values, parts, ids, regions):
    final = pd.Series(parts["multilayer"], index=ids)
    scores, medians = [], {}
    for name, series in values.items():
        series = series.reindex(ids).dropna()
        logged = np.log1p(series.clip(lower=0)).to_numpy()
        labels = final[series.index].to_numpy()
        statistic, p_value = kruskal(*[logged[labels == t] for t in np.unique(labels)])
        row = {"indicator": name, "municipalities": len(series), "kruskal_h": statistic, "p": p_value}
        row.update({method: eta_squared_h(logged, pd.Series(part, index=ids)[series.index].to_numpy()) for method, part in parts.items()})
        row["region"] = eta_squared_h(logged, regions.reindex(series.index).to_numpy())
        scores.append(row)
        medians[name] = series.groupby(final[series.index]).median()
    return pd.DataFrame(scores).set_index("indicator"), pd.DataFrame(medians)


def four_russias(config, data, final, period):
    settings = config["four_russias"]
    territories = data["territories"].loc[final.index]
    features = data["features"]
    population = features[features["period"] == period].set_index("territory_id")["population"].reindex(final.index)
    city = territories["type"].isin(settings["city_types"])
    group = pd.Series(3, index=final.index, name="russia")
    group[city & (population >= settings["middle_city"])] = 2
    group[city & (population >= settings["big_city"])] = 1
    group[territories["region_code"].isin(settings["fourth_region_codes"])] = 4
    counts = pd.crosstab(final.rename("type"), group)
    people = pd.crosstab(final.rename("type"), group, values=population / 1e6, aggfunc="sum").fillna(0)
    chi2 = chi2_contingency(counts.to_numpy())[0]
    return counts, people, np.sqrt(chi2 / (counts.to_numpy().sum() * (min(counts.shape) - 1))), group


def runner(config, name, alpha, x, w, wa, k):
    if name in methods.ATTRIBUTE_METHODS:
        return lambda keep, seed: methods.ATTRIBUTE_METHODS[name](x[keep], k, seed)
    if name == "spectral":
        return lambda keep, seed: methods.spectral(w[keep][:, keep], k, seed)
    if name in methods.GRAPH_METHODS:
        _, resolution = with_k(methods.GRAPH_METHODS[name], w, k, config["compare"]["resolutions"])
        return lambda keep, seed: methods.GRAPH_METHODS[name](w[keep][:, keep], resolution, seed)
    return lambda keep, seed: methods.joint(name, x[keep], w[keep][:, keep], wa[keep][:, keep], k, alpha, seed)


def check_stability(config, x, w, wa, best):
    settings, model = config["validate"], config["model"]
    samples = subsamples(len(x), {"seed": settings["seed"], "share": settings["share"], "subsamples": settings["stability_runs"]})
    rows = []

    def record(part, name, alpha, k):
        run = runner(config, name, alpha, x, w, wa, k)
        scores = agreement(run, run(np.arange(len(x)), 0), samples)
        low, high = np.percentile(scores, [5, 95])
        rows.append({"part": part, "config": label(name, alpha), "k": k, "mean": scores.mean(), "std": scores.std(), "p5": low, "p95": high, "share_above_085": np.mean(scores >= 0.85)})

    for k in settings["stability_ks"]:
        record("curve", "kmeans", np.nan, k)
        record("curve", "fused_spectral", model["alpha"], k)
    for name, alpha in best[["method", "alpha"]].itertuples(index=False):
        if name != "kmeans" and not (name == "fused_spectral" and alpha == model["alpha"]):
            record("best", name, alpha, model["k"])
    return pd.DataFrame(rows)


def label(name, alpha):
    return name if np.isnan(alpha) else f"{name} {alpha:.1f}"


def thirds(values):
    low, high = np.percentile(values, [100 / 3, 200 / 3])
    return pd.Series(np.where(values >= high, 1, np.where(values >= low, 2, 3)), index=values.index)


def rank_methods(table, k):
    part = table[table["k"] == k].copy()
    part.index = [label(name, alpha) for name, alpha in zip(part["method"], part["alpha"])]
    scores = pd.DataFrame({name: part[name] * direction for name, direction in INDICES.items()})
    grades = scores.apply(thirds)
    result = grades.assign(
        method=part["method"], alpha=part["alpha"], worst=(grades == 3).sum(axis=1), middle=(grades == 2).sum(axis=1), best=(grades == 1).sum(axis=1),
        borda=scores.rank().sum(axis=1), within=part["within"], stability=part["stability"],
    )
    result["place"] = (10 * result["worst"] + result["middle"]).rank(method="min").astype(int)
    return result.sort_values(["place", "borda"], ascending=[True, False])


def plot_network(mix, morans, names, order, period, n, path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    shown = mix[np.ix_(order, order)]
    ax.imshow(shown, cmap="Blues", vmin=0, vmax=1)
    labels = [names[t] for t in order]
    ax.set_xticks(range(len(order)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(order)), labels)
    for i in range(len(order)):
        for j in range(len(order)):
            ax.text(j, i, f"{shown[i, j]:.0%}", ha="center", va="center", fontsize=9, color="white" if shown[i, j] > 0.5 else INK)
    ax.set_title("Куда ведут рёбра сети потребления: доля веса рёбер типа", loc="left", fontsize=11)
    ax = axes[1]
    rows = morans.sort_values("moran_i")
    ax.barh(range(len(rows)), rows["moran_i"], color=[BLOCK_COLORS[b] for b in rows["block"]])
    ax.set_yticks(range(len(rows)), [LABELS[f] for f in rows["feature"]], fontsize=8.5)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel("I Морана на графе потребления (0: нет сетевой автокорреляции)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in BLOCK_COLORS.values()]
    ax.legend(handles, ["потребление", "труд", "место"], frameon=False, loc="lower right")
    ax.set_title("Какие признаки согласованы с сетью потребления", loc="left", fontsize=11)
    fig.text(0.01, 0.01, f"Окно {period}, {n} МО, граф 10 ближайших по структуре трат. Признаки потребления согласованы с сетью по построению; "
             "важно, что признаки места и зарплата тоже согласованы, а отраслевая занятость слабее.", color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_consensus(ari, k, period, path):
    order = leaves_list(linkage(squareform(1 - ari.to_numpy(), checks=False), "average"))
    shown = ari.iloc[order, order]
    fig, ax = plt.subplots(figsize=(12, 10.5))
    ax.imshow(shown.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(shown)), shown.columns, rotation=60, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(shown)), shown.index, fontsize=8.5)
    for i in range(len(shown)):
        for j in range(len(shown)):
            v = shown.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.6 else INK)
    ax.set_title(f"Насколько похожи разбиения разных методов: ARI, k = {k}, окно {period}", loc="left")
    fig.text(0.01, 0.01, "Порядок по иерархической кластеризации методов (средняя связь по 1 - ARI). Число после названия: вес графа alpha. "
             "multilayer: итоговые типы.", color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_stability(table, alpha, runs, path):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for offset, (name, label, color) in zip([-0.08, 0.08], [("kmeans", "k-means", "#52514e"), ("fused_spectral", f"совместная спектральная, alpha {alpha}", "#1baf7a")]):
        part = table[(table["part"] == "curve") & table["config"].str.startswith(name)]
        ax.errorbar(part["k"] + offset, part["mean"], yerr=[part["mean"] - part["p5"], part["p95"] - part["mean"]], fmt="o-", color=color, capsize=3, label=label, markersize=5)
    ax.axhline(0.85, color=MUTED, linestyle="--", linewidth=0.9)
    ax.text(table["k"].max() + 0.3, 0.85, "0.85", color=MUTED, va="center", fontsize=9)
    ax.set_xlabel("число типов k")
    ax.set_ylabel("ARI с разбиением всех МО")
    ax.set_title(f"Устойчивость к составу МО: {runs} подвыборок по 90%, точки: среднее, усы: 5-95%", loc="left", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def validate(config):
    out = config["paths"]["results"]
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    style()
    settings = config["validate"]
    rng = np.random.default_rng(settings["seed"])
    data = load_data(config)
    period, k = config["compare"]["period"], config["model"]["k"]
    names = {int(t): v for t, v in config["describe"]["short_names"].items()}
    order = config["describe"]["order"]
    types = pd.read_csv(out / "types.csv")
    confidence = pd.read_csv(out / "confidence.csv")
    periods = sorted(types["period"].unique())
    shot = snapshot(config, data, period)
    x, w, wa = shot["x"], shot["graph"], shot["attribute_graph"]
    final = window_types(types, period, shot["ids"])
    start = time.time()

    blocks = check_blocks(config, data, period)
    blocks.round(4).to_csv(out / "validate_blocks.csv")
    importance, accuracy = check_importance(data["features"], final.dropna().astype(int), period)
    importance.round(4).to_csv(out / "validate_importance.csv", index=False)
    print(blocks.round(3).to_string())
    print(f"random forest reproduces the types with 5-fold accuracy {accuracy:.3f}; importance by block:", importance.groupby("block")["importance"].sum().round(3).to_dict(), flush=True)

    window_confidence = confidence[confidence["period"] == period].set_index("territory_id")["confidence"]
    summary, mix, morans, bridges = check_network(config, data, shot, final, window_confidence, period, rng)
    summary.to_csv(out / "validate_network.csv", header=["value"])
    pd.DataFrame(mix, index=range(k), columns=range(k)).rename(index=names, columns=names).round(4).to_csv(out / "validate_mixing.csv")
    morans.round(4).to_csv(out / "validate_moran.csv", index=False)
    bridges.assign(type=bridges["type"].map(names), main_other_type=bridges["main_other_type"].map(names)).round(4).to_csv(out / "validate_bridges.csv")
    plot_network(mix, morans, names, order, period, len(bridges), figures / "network_structure.png")
    print(summary.round(4).to_string())
    print("Moran I by block (median):", morans.groupby("block")["moran_i"].median().round(3).to_dict())
    rho = spearmanr(bridges["participation"], bridges["confidence"], nan_policy="omit").statistic
    print(f"participation vs bootstrap confidence: Spearman {rho:.3f}; connectors with participation above {settings['bridge_participation']}: {(bridges['participation'] > settings['bridge_participation']).sum()}", flush=True)

    scores, shares, moved, total = check_changes(config, data, types, confidence, periods[0], periods[-1])
    scores.to_csv(out / "validate_changes.csv", index=False)
    shares.round(4).to_csv(out / "validate_changes_by_confidence.csv")
    print(f"type change {periods[0]} -> {periods[-1]}: {moved} of {total} municipalities")
    print(scores.round(4).to_string(index=False))
    print(shares.round(3).to_string(), flush=True)

    keep = final.notna().to_numpy()
    parts = {name: labels[keep] for name, labels in partitions(config, x, w, wa).items()}
    parts["multilayer"] = final[keep].astype(int).to_numpy()
    ari, consensus, support = check_consensus(parts, k)
    ari.round(4).to_csv(out / "validate_methods_ari.csv")
    consensus.round(4).to_csv(out / "validate_consensus.csv")
    pd.DataFrame({"type": final[keep].astype(int).map(names), "method_support": support, "confidence": window_confidence.reindex(final.index[keep])}).round(4).to_csv(out / "validate_support.csv")
    plot_consensus(ari, k, period, figures / "methods_ari.png")
    print(consensus.round(3).to_string())
    print(f"method support vs bootstrap confidence: Spearman {spearmanr(support, window_confidence.reindex(final.index[keep]), nan_policy='omit').statistic:.3f}", flush=True)

    values = external_values(config, data, period)
    external, medians = check_external(values, parts, final.index[keep], data["territories"]["region"])
    external.round(4).to_csv(out / "validate_external.csv")
    medians.rename(index=names).round(2).to_csv(out / "validate_external_medians.csv")
    print(external[["municipalities", "kruskal_h", "p"]].to_string())
    print("eta squared H, mean over indicators:", external.drop(columns=["municipalities", "kruskal_h", "p"]).mean().round(3).sort_values(ascending=False).to_dict())
    counts, people, cramer, groups = four_russias(config, data, final[keep].astype(int), period)
    counts.rename(index=names).to_csv(out / "validate_four_russias.csv")
    people.rename(index=names).round(3).to_csv(out / "validate_four_russias_population.csv")
    print(counts.rename(index=names).to_string())
    print(f"four Russias: Cramer V {cramer:.3f}, ARI {adjusted_rand_score(groups, final[keep].astype(int)):.3f}", flush=True)

    ranking = rank_methods(pd.read_csv(out / "methods.csv"), k)
    ranking.round(4).to_csv(out / "validate_ranking.csv")
    best = ranking.groupby("method", sort=False).head(1)
    print(f"{len(ranking)} configurations with k = {k}", flush=True)

    stability = check_stability(config, x, w, wa, best)
    stability.round(4).to_csv(out / "validate_stability.csv", index=False)
    plot_stability(stability, config["model"]["alpha"], settings["stability_runs"], figures / "stability_curve.png")
    print(stability.round(3).to_string(index=False))
    best = best.assign(stability_runs=best.index.map(stability[stability["k"] == k].set_index("config")["mean"]))
    best.round(4).to_csv(out / "validate_ranking_methods.csv")
    print("best configuration of each method by the threshold rule, with stability on the validate subsamples:")
    print(best.round(3).to_string(), flush=True)
    print(f"validate {time.time() - start:.0f}s")