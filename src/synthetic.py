import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

from . import methods
from .describe import GRID, INK, MUTED, style
from .metrics import panel
from .network import BLOCKS, cosine, knn, load_data, snapshot
from .validate import with_k

SPEND = slice(0, len(BLOCKS["spending"]))
REST = slice(len(BLOCKS["spending"]), None)
SCENARIOS = {
    "calibrated": "как реальные данные",
    "pairs_by_spending": "три пары типов различаются\nтолько тратами",
    "spending_only": "типы различаются только тратами",
    "noisier": "шум в 1.5 раза сильнее",
    "independent_graph": "граф со своим сигналом,\nпризнаки шумные",
    "no_spending": "траты не связаны с типом:\nграф потребления бесполезен",
}
SHOWN = [
    ("kmeans", "k-means", "#52514e", "s"),
    ("kefrin_euclidean", "KEFRiN, евклид", "#eb6834", "D"),
    ("spectral", "спектральная по графу", "#2a78d6", "^"),
    ("dmon", "DMoN", "#e87ba4", "v"),
    ("fused_spectral", "совместная спектральная без уточнения", "#eda100", "P"),
    ("refined_spectral", "совместная спектральная с уточнением", "#1baf7a", "o"),
]
INDICES = ["SW", "CH", "S_Dbw", "DB", "AVI", "AVU", "MQ", "within"]


def window(config):
    period = config["compare"]["period"]
    shot = snapshot(config, load_data(config), period)
    types = pd.read_csv(config["paths"]["results"] / "types.csv")
    labels = types[types["period"] == period].set_index("territory_id")["type"].reindex(shot["ids"])
    keep = labels.notna().to_numpy()
    return shot, keep, labels[keep].astype(int).to_numpy()


def type_centers(x, y):
    return np.array([x[y == c].mean(axis=0) for c in range(y.max() + 1)])


def real_types(config):
    shot, keep, y = window(config)
    x = shot["x"][keep]
    return x, y, type_centers(x, y)


def truth_types(config, source):
    if source == "model":
        return real_types(config)
    shot, keep, y = window(config)
    x, k = shot["x"][keep], y.max() + 1
    y = methods.kmeans(x, k) if source == "kmeans" else methods.spectral(shot["graph"][keep][:, keep], k)
    return x, y, type_centers(x, y)


def scenario_centers(centers, grand, layout):
    result = centers.copy()
    if layout == "spending_only":
        result[:, REST] = grand[REST]
    elif layout == "no_spending":
        result[:, SPEND] = grand[SPEND]
    elif layout == "pairs_by_spending":
        for a in range(0, len(result) - 1, 2):
            result[a + 1, REST] = result[a, REST]
    return result


def planted(truth, degree, inside, rng):
    n = len(truth)
    members = [np.flatnonzero(truth == c) for c in range(truth.max() + 1)]
    others = [np.flatnonzero(truth != c) for c in range(truth.max() + 1)]
    sources = rng.integers(n, size=n * degree // 2)
    same = rng.random(len(sources)) < inside
    targets = np.array([rng.choice(members[truth[u]] if s else others[truth[u]]) for u, s in zip(sources, same)])
    w = sparse.coo_matrix((np.ones(len(sources)), (sources, targets)), shape=(n, n)).tocsr()
    w = (w + w.T).tocsr()
    w.data[:] = 1.0
    w.setdiag(0)
    w.eliminate_zeros()
    return w


def dataset(config, name, centers, residuals, sizes, rng):
    settings, k = config["synthetic"], config["network"]["k"]
    noisy = name in ("noisier", "independent_graph")
    truth = rng.permutation(np.repeat(np.arange(len(sizes)), sizes))
    grand = sizes @ centers / sizes.sum()
    x = scenario_centers(centers, grand, "calibrated" if noisy else name)[truth]
    x = x + (settings["noise"] if noisy else 1.0) * residuals[rng.integers(len(residuals), size=len(truth))]
    w = knn(cosine(x[:, SPEND]), k)
    if name == "independent_graph":
        w = planted(truth, int(np.median(np.diff(w.indptr))), settings["inside_share"], rng)
    return x, w, knn(cosine(x), k), truth


def partitions(config, x, w, wa, k):
    alpha, weight = config["model"]["alpha"], config["model"]["refine_weight"]
    result = {name: method(x, k) for name, method in methods.ATTRIBUTE_METHODS.items()}
    result["spectral"] = methods.spectral(w, k)
    for name, method in methods.GRAPH_METHODS.items():
        labels, _ = with_k(method, w, k, config["compare"]["resolutions"])
        if labels is not None:
            result[name] = labels
    for name in methods.JOINT_METHODS:
        if name != "refined_spectral":
            result[f"{name} {alpha}"] = methods.joint(name, x, w, wa, k, alpha, weight)
    for value in config["synthetic"]["alphas"]:
        result[f"refined_spectral {value}"] = methods.joint("refined_spectral", x, w, wa, k, value, weight)
    return result


def plot(table, alpha, n, path):
    stats = table.groupby(["scenario", "method"])["ari"].agg(["mean", "min", "max"])
    fig, ax = plt.subplots(figsize=(13, 6.6))
    offsets = np.linspace(-0.28, 0.28, len(SHOWN))
    for row, scenario in enumerate(SCENARIOS):
        if row % 2 == 0:
            ax.axhspan(row - 0.45, row + 0.45, color=GRID, alpha=0.35, linewidth=0)
        for offset, (method, label, color, marker) in zip(offsets, SHOWN):
            key = method if (scenario, method) in stats.index else f"{method} {alpha}"
            mean, low, high = stats.loc[(scenario, key)]
            y = row + offset
            ax.plot([low, high], [y, y], color=color, linewidth=2, solid_capstyle="round")
            name = f"{label}, alpha {alpha}" if method in methods.JOINT_METHODS else label
            ax.scatter([mean], [y], s=64, marker=marker, color=color, edgecolor="white", linewidth=1.2, zorder=3, label=name if row == 0 else None)
            if method in ("kmeans", "refined_spectral"):
                ax.text(high + 0.012, y, f"{mean:.2f}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(SCENARIOS)), list(SCENARIOS.values()))
    ax.set_ylim(len(SCENARIOS) - 0.5, -0.5)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("ARI с истинными типами (точка: среднее по наборам, отрезок: от худшего до лучшего)")
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.01, 0.925), ncol=3)
    fig.suptitle("Синтетические данные с известными типами: ARI разных методов в шести сценариях", x=0.01, ha="left", fontsize=12)
    fig.text(0.01, 0.012, f"{n} МО; центры типов, их размеры и разброс внутри типов взяты из окна модели. Граф строится как в модели (ближайшие соседи по тратам),\n"
             "кроме сценария со своим графом (planted partition). Подписи: среднее у k-means и у модели (совместная спектральная с уточнением).", color=MUTED, fontsize=8.5)
    fig.subplots_adjust(left=0.215, right=0.99, top=0.83, bottom=0.15)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def synthetic(config):
    out = config["paths"]["results"]
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    style()
    settings = config["synthetic"]
    x, y, centers = real_types(config)
    residuals = x - centers[y]
    sizes = np.bincount(y)
    k = len(sizes)
    rng_metrics = np.random.default_rng(settings["seed"])
    rows = []
    for number, name in enumerate(SCENARIOS):
        for repeat in range(settings["repeats"]):
            start = time.time()
            rng = np.random.default_rng([settings["seed"], number, repeat])
            xs, w, wa, truth = dataset(config, name, centers, residuals, sizes, rng)
            rows.append({"scenario": name, "repeat": repeat, "method": "truth", "ari": 1.0, **panel(xs, w, truth, rng_metrics)})
            for method, labels in partitions(config, xs, w, wa, k).items():
                rows.append({"scenario": name, "repeat": repeat, "method": method, "ari": adjusted_rand_score(truth, labels), **panel(xs, w, labels, rng_metrics)})
            pd.DataFrame(rows).to_csv(out / "synthetic.csv", index=False)
            print(time.strftime("%H:%M:%S"), f"{name} {repeat}: {time.time() - start:.0f}s", flush=True)
    table = pd.DataFrame(rows)
    found = table[table["method"] != "truth"]
    ari = found.pivot_table(index="method", columns="scenario", values="ari", aggfunc="mean")[list(SCENARIOS)]
    ari.round(4).to_csv(out / "synthetic_ari.csv")
    correlation = pd.DataFrame([
        {"scenario": s, "repeat": r, "index": index, "spearman": spearmanr(part[index], part["ari"]).statistic}
        for (s, r), part in found.groupby(["scenario", "repeat"]) for index in INDICES
    ]).pivot_table(index="index", columns="scenario", values="spearman", aggfunc="mean")[list(SCENARIOS)]
    correlation.round(4).to_csv(out / "synthetic_indices.csv")
    plot(found, config["model"]["alpha"], len(y), figures / "synthetic_ari.png")
    truth = table[table["method"] == "truth"].groupby("scenario")[["SW", "MQ"]].mean().reindex(list(SCENARIOS))
    with pd.option_context("display.width", 250):
        print("indices of the true partition:")
        print(truth.round(3).to_string())
        print("ARI with the true types, mean over repeats:")
        print(ari.round(3).to_string())
        print("Spearman correlation of each index with ARI across methods:")
        print(correlation.round(2).to_string())
    check_truths(config)


def truth_partitions(config, x, w, wa, k):
    alpha, weight = config["model"]["alpha"], config["model"]["refine_weight"]
    result = {name: method(x, k) for name, method in methods.ATTRIBUTE_METHODS.items()}
    result["spectral"] = methods.spectral(w, k)
    for name in ("kefrin_euclidean", "kefrin_cosine", "fused_spectral"):
        result[name] = methods.joint(name, x, w, wa, k, alpha, weight)
    for value in config["validate"]["refine_weights"]:
        result[f"refined_spectral w{value}"] = methods.refined_spectral(x, w, wa, k, alpha, value)
    return result


def check_truths(config):
    out = config["paths"]["results"]
    settings = config["synthetic"]
    reference = f"refined_spectral w{config['model']['refine_weight']}"
    scenarios = list(SCENARIOS)
    rows = []
    for position, source in enumerate(settings["truths"]):
        x, y, centers = truth_types(config, source)
        residuals = x - centers[y]
        sizes = np.bincount(y)
        for name in settings["truth_scenarios"]:
            number = scenarios.index(name)
            for repeat in range(settings["repeats"]):
                start = time.time()
                seed = [settings["seed"], number, repeat] + ([] if source == "model" else [position])
                xs, w, wa, truth = dataset(config, name, centers, residuals, sizes, np.random.default_rng(seed))
                for method, labels in truth_partitions(config, xs, w, wa, len(sizes)).items():
                    rows.append({"truth": source, "scenario": name, "repeat": repeat, "method": method, "ari": adjusted_rand_score(truth, labels)})
                pd.DataFrame(rows).to_csv(out / "synthetic_truths.csv", index=False)
                print(time.strftime("%H:%M:%S"), f"truth {source}, {name} {repeat}: {time.time() - start:.0f}s", flush=True)
    table = pd.DataFrame(rows)
    ari = table.pivot_table(index="method", columns=["truth", "scenario"], values="ari", aggfunc="mean")
    ari = ari[[(t, s) for t in settings["truths"] for s in settings["truth_scenarios"]]]
    ari.columns = [f"{t} {s}" for t, s in ari.columns]
    ari.round(4).to_csv(out / "synthetic_truths_ari.csv")
    with pd.option_context("display.width", 250):
        print("ARI with the true types from other methods, mean over repeats (columns: truth source and scenario):")
        print(ari.round(3).to_string())
        print(f"repeats in which {reference} beats the method:")
        for (source, name), part in table.groupby(["truth", "scenario"], sort=False):
            wide = part.pivot(index="repeat", columns="method", values="ari")
            wins = {m: int((wide[reference] > wide[m]).sum()) for m in wide.columns if m != reference}
            print(f"{source} {name} (of {len(wide)}):", wins)