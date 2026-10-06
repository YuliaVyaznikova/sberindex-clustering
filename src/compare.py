import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from . import methods
from .kefrin import default_alpha
from .metrics import panel
from .network import edge_variants, load_data, snapshot


def edge_set(w):
    rows, cols = w.nonzero()
    return set(zip(rows[rows < cols], cols[rows < cols]))


def jaccard(a, b):
    a, b = edge_set(a), edge_set(b)
    return len(a & b) / len(a | b)


def describe_graph(w, labels, regions, types, distance):
    n = w.shape[0]
    degree = np.diff(w.indptr)
    _, component = connected_components(w, directed=False)
    upper = w.tocoo()
    keep = upper.row < upper.col
    rows, cols, weights = upper.row[keep], upper.col[keep], upper.data[keep]
    graph = nx.from_scipy_sparse_array(w)
    parts = [set(np.flatnonzero(labels == c)) for c in np.unique(labels)]
    return {
        "edges": len(rows),
        "mean_degree": 2 * len(rows) / n,
        "max_degree": int(degree.max()),
        "isolated": int((degree == 0).sum()),
        "components": int(component.max() + 1),
        "giant_share": np.bincount(component).max() / n,
        "clustering": nx.average_clustering(graph),
        "same_region": np.average(regions[rows] == regions[cols], weights=weights),
        "same_type": np.average(types[rows] == types[cols], weights=weights),
        "median_edge_km": float(np.median(distance[rows, cols])),
        "louvain_communities": len(parts),
        "modularity": nx.community.modularity(graph, parts, weight="weight"),
        "nmi_region": normalized_mutual_info_score(regions, labels),
    }


def compare_edges(config, data, out):
    period = config["compare"]["period"]
    base, variants, distance, info = edge_variants(config, data, period)
    territories = data["territories"]
    regions = territories.loc[base["ids"], "region"].to_numpy()
    types = territories.loc[base["ids"], "type"].to_numpy()
    rows, labels = [], {}
    for name, w in variants.items():
        labels[name] = methods.louvain(w, 1.0)
        rows.append({"graph": name, **describe_graph(w, labels[name], regions, types, distance)})
    table = pd.DataFrame(rows).set_index("graph")
    table.round(3).to_csv(out / "edges.csv")
    names = list(variants)
    pd.DataFrame([[jaccard(variants[a], variants[b]) for b in names] for a in names], index=names, columns=names).round(3).to_csv(out / "edges_jaccard.csv")
    pd.DataFrame([[adjusted_rand_score(labels[a], labels[b]) for b in names] for a in names], index=names, columns=names).round(3).to_csv(out / "edges_partition_ari.csv")
    print(f"edges: {len(base['ids'])} municipalities in {period}, detour {info['detour']:.2f}, geo scale {info['geo_scale_km']:.0f} km")
    print(table.round(3).to_string())
    compare_edges_over_time(config, data, out)


def compare_edges_over_time(config, data, out):
    periods = sorted(data["features"]["period"].unique())
    previous, rows = None, []
    for period in periods:
        base, variants, _, _ = edge_variants(config, data, period)
        current = {name: (base["ids"], w, methods.louvain(w, 1.0)) for name, w in variants.items()}
        if previous:
            for name, (ids, w, labels) in current.items():
                if name not in previous[1]:
                    continue
                old_ids, old_w, old_labels = previous[1][name]
                common = np.intersect1d(ids, old_ids)
                now, before = pd.Index(ids).get_indexer(common), pd.Index(old_ids).get_indexer(common)
                rows.append({
                    "graph": name, "from": previous[0], "to": period,
                    "edge_jaccard": jaccard(old_w[before][:, before], w[now][:, now]),
                    "louvain_ari": adjusted_rand_score(old_labels[before], labels[now]),
                })
        previous = (period, current)
    table = pd.DataFrame(rows)
    table.round(3).to_csv(out / "edges_over_time.csv", index=False)
    print(table.groupby("graph")[["edge_jaccard", "louvain_ari"]].mean().round(3).to_string())


def subsamples(n, settings):
    rng = np.random.default_rng(settings["seed"])
    return [(np.sort(rng.choice(n, int(settings["share"] * n), replace=False)), int(rng.integers(1_000_000))) for _ in range(settings["subsamples"])]


def agreement(run, labels, samples):
    return np.array([adjusted_rand_score(labels[keep], run(keep, seed)) for keep, seed in samples])


def stability(run, labels, samples):
    return float(agreement(run, labels, samples).mean())


def compare_methods(config, data, out):
    settings = config["compare"]
    rng = np.random.default_rng(settings["seed"])
    shot = snapshot(config, data, settings["period"])
    x, w, wa = shot["x"], shot["graph"], shot["attribute_graph"]
    samples = subsamples(len(x), settings)
    defaults = {metric: default_alpha(x, w, metric) for metric in ["euclidean", "cosine"]}
    pd.Series(defaults, name="alpha").to_csv(out / "kefrin_default_alpha.csv")
    print("KEFRiN with rho = xi = 1 corresponds to alpha", {m: round(a, 3) for m, a in defaults.items()}, flush=True)
    rows = []
    weight = config["model"]["refine_weight"]

    def add(method, source, alpha, labels, stable):
        rows.append({"method": method, "source": source, "alpha": alpha, "stability": stable, **panel(x, w, labels, rng)})

    start = time.time()
    for name, method in methods.ATTRIBUTE_METHODS.items():
        for k in settings["ks"]:
            labels = method(x, k)
            add(name, "attributes", np.nan, labels, stability(lambda keep, seed: method(x[keep], k, seed), labels, samples))
    for k in settings["ks"]:
        labels = methods.spectral(w, k)
        add("spectral", "graph", np.nan, labels, stability(lambda keep, seed: methods.spectral(w[keep][:, keep], k, seed), labels, samples))
    for name, method in methods.GRAPH_METHODS.items():
        found = {}
        for resolution in np.geomspace(0.05, 3.0, settings["resolutions"]):
            labels = method(w, resolution)
            k = labels.max() + 1
            if k in settings["ks"] and k not in found:
                found[k] = (resolution, labels)
        for k, (resolution, labels) in sorted(found.items()):
            add(name, "graph", np.nan, labels, stability(lambda keep, seed: method(w[keep][:, keep], resolution, seed), labels, samples))
    print(f"baselines {time.time() - start:.0f}s", flush=True)

    for name in methods.JOINT_METHODS:
        start = time.time()
        for k in settings["joint_ks"]:
            for alpha in settings["alphas"]:
                labels = methods.joint(name, x, w, wa, k, alpha, weight)
                stable = np.nan
                if alpha in settings["stable_alphas"]:
                    stable = stability(lambda keep, seed: methods.joint(name, x[keep], w[keep][:, keep], wa[keep][:, keep], k, alpha, weight, seed), labels, samples)
                add(name, "joint", alpha, labels, stable)
            pd.DataFrame(rows).to_csv(out / "methods.csv", index=False)
        print(f"{name} {time.time() - start:.0f}s", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(out / "methods.csv", index=False)
    plot_methods(table, config["model"], out / "figures")
    return table


def plot_methods(table, model, folder):
    folder.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    colors = {"kefrin_euclidean": "tab:red", "kefrin_cosine": "tab:purple", "canus_euclidean": "tab:orange", "canus_cosine": "tab:brown", "dmon": "tab:pink", "fused_spectral": "tab:olive", "refined_spectral": "tab:green"}
    markers = {"kmeans": "s", "ward": "D", "gmm": "v", "spectral": "^", "louvain": "o", "leiden": "P"}
    for ax, k in zip(axes, [4, 6, 8]):
        part = table[table["k"] == k]
        for name, color in colors.items():
            line = part[part["method"] == name].sort_values("alpha")
            ax.plot(line["SW"], line["MQ"], marker="o", markersize=3.5, linewidth=1.2, color=color, label="SPECTRA" if name == "refined_spectral" else name)
        for name, marker in markers.items():
            point = part[part["method"] == name]
            ax.scatter(point["SW"], point["MQ"], marker=marker, s=60, color="black", label=name, zorder=5)
        chosen = part[(part["method"] == "refined_spectral") & (part["alpha"] == model["alpha"])]
        ax.scatter(chosen["SW"], chosen["MQ"], s=220, facecolors="none", edgecolors="tab:green", linewidths=2, zorder=6)
        ax.set_title(f"k = {k}")
        ax.set_xlabel("SW, пространство признаков")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("MQ, граф потребления")
    axes[0].legend(fontsize=7.5, loc="lower left", ncol=2)
    fig.suptitle(f"Качество в пространстве признаков и на графе. Линии показывают путь совместных методов по весу графа alpha от 0 до 1, кружком отмечена SPECTRA (alpha {model['alpha']})")
    fig.tight_layout()
    fig.savefig(folder / "methods_tradeoff.png", dpi=130)
    plt.close(fig)


def compare(config):
    out = config["paths"]["results"]
    out.mkdir(parents=True, exist_ok=True)
    data = load_data(config)
    compare_edges(config, data, out)
    table = compare_methods(config, data, out)
    shown = table[table["alpha"].isna() | (table["alpha"] == 0.3)]
    print(shown.pivot_table(index="method", columns="k", values=["SW", "MQ", "stability"]).round(3).to_string())