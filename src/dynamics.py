import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, silhouette_score

from .methods import fuse, refine, refine_costs, relabel, spectral
from .metrics import graph_scores
from .network import load_data, snapshot


def snapshots(config, data):
    periods = sorted(data["features"]["period"].unique())
    alpha = config["model"]["alpha"]
    shots = {}
    for period in periods:
        shot = snapshot(config, data, period)
        shot["affinity"] = fuse(shot["graph"], shot["attribute_graph"], alpha)
        shots[period] = shot
    return periods, shots


def on_panel(shots, periods):
    ids = np.array(sorted(set.intersection(*[set(shots[p]["ids"]) for p in periods])))
    parts = []
    for period in periods:
        index = pd.Index(shots[period]["ids"]).get_indexer(ids)
        shot = shots[period]
        parts.append((shot["x"][index], shot["affinity"][index][:, index].tocsr(), shot["graph"][index][:, index].tocsr()))
    return ids, [p[0] for p in parts], [p[1] for p in parts], [p[2] for p in parts]


def align(previous, current, k):
    overlap = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            union = np.sum((previous == i) | (current == j))
            overlap[i, j] = np.sum((previous == i) & (current == j)) / union if union else 0.0
    rows, cols = linear_sum_assignment(-overlap)
    mapping = np.empty(k, dtype=int)
    mapping[cols] = rows
    return mapping[current]


def chain(labels, k):
    aligned = [labels[0]]
    for current in labels[1:]:
        aligned.append(align(aligned[-1], current, k))
    return aligned


def block_stats(w, labels, k):
    member = sparse.csr_matrix((np.ones(len(labels)), (np.arange(len(labels)), labels)), shape=(len(labels), k))
    sizes = np.bincount(labels, minlength=k).astype(float)
    counts = np.outer(sizes, sizes) - np.diag(sizes)
    sums = np.asarray((member.T @ w @ member).todense())
    squares = np.asarray((member.T @ w.multiply(w) @ member).todense())
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    variances = np.divide(squares - counts * means ** 2, counts - 1, out=np.zeros_like(sums), where=counts > 1)
    return member, counts, means, np.maximum(variances, 0)


def affect_weight(smoothed, current, labels, k):
    member, counts, means, variances = block_stats(current, labels, k)
    noise = float((counts * variances).sum())
    cross = np.asarray((member.T @ smoothed @ member).todense())
    bias = float(smoothed.multiply(smoothed).sum() - 2 * (means * cross).sum() + (counts * means ** 2).sum())
    return noise / (bias + noise) if bias + noise > 0 else 0.0


def evolutionary(affinities, k, past=None, iterations=3):
    smoothed, labels, weights = affinities[0], [spectral(affinities[0], k)], []
    for w in affinities[1:]:
        current = labels[-1]
        for _ in range(iterations if past is None else 1):
            weight = affect_weight(smoothed, w, current, k) if past is None else past
            candidate = (weight * smoothed + (1 - weight) * w).tocsr()
            current = spectral(candidate, k)
        smoothed = candidate
        weights.append(weight)
        labels.append(current)
    return chain(labels, k), weights


def supra(affinities, ids, k, coupling):
    sizes = [len(i) for i in ids]
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    strength = coupling / np.mean(sizes)
    rows, cols = [], []
    for t in range(len(ids) - 1):
        common = np.intersect1d(ids[t], ids[t + 1])
        rows.append(offsets[t] + pd.Index(ids[t]).get_indexer(common))
        cols.append(offsets[t + 1] + pd.Index(ids[t + 1]).get_indexer(common))
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    links = sparse.coo_matrix((np.full(len(rows), strength), (rows, cols)), shape=(offsets[-1], offsets[-1]))
    labels = relabel(spectral((sparse.block_diag(affinities) + links + links.T).tocsr(), k))
    return [labels[offsets[t]:offsets[t + 1]] for t in range(len(ids))]


def refine_over_time(xs, graphs, ids, labels, k, weight, smoothing):
    costs = [pd.DataFrame(refine_costs(x, g, l, k, weight), index=i) for x, g, i, l in zip(xs, graphs, ids, labels)]
    result = []
    for t, (own, current) in enumerate(zip(ids, labels)):
        total, norm = np.zeros((len(own), k)), np.zeros((len(own), 1))
        for s, cost in enumerate(costs):
            factor = smoothing ** abs(t - s)
            if factor > 0:
                part = cost.reindex(own)
                total += factor * part.fillna(0).to_numpy()
                norm += factor * part.notna().all(axis=1).to_numpy()[:, None]
        result.append(refine(total / norm, current))
    return result


def fixed_typology(xs, affinities, k):
    reference = spectral((sum(affinities) / len(affinities)).tocsr(), k)
    centers = np.mean([[x[reference == c].mean(axis=0) for c in range(k)] for x in xs], axis=0)
    return [((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).argmin(axis=1) for x in xs]


def evaluate(name, labels, xs, graphs):
    changed = [float(np.mean(a != b)) for a, b in zip(labels, labels[1:])]
    agreement = [adjusted_rand_score(a, b) for a, b in zip(labels, labels[1:])]
    k = max(l.max() for l in labels) + 1
    return {
        "approach": name,
        "SW": float(np.mean([silhouette_score(x, l) for x, l in zip(xs, labels)])),
        "MQ": float(np.mean([graph_scores(g, l)["MQ"] for g, l in zip(graphs, labels)])),
        "changed_per_step": float(np.mean(changed)),
        "ari_per_step": float(np.mean(agreement)),
        "changed_first_last": float(np.mean(labels[0] != labels[-1])),
        "ari_first_last": adjusted_rand_score(labels[0], labels[-1]),
        "min_share": min(np.bincount(l, minlength=k).min() / len(l) for l in labels),
    }


def compare_dynamics(config, shots, periods, out):
    settings, k = config["dynamics"], config["model"]["k"]
    ids, xs, affinities, graphs = on_panel(shots, periods)
    rows = [evaluate("independent", chain([spectral(a, k) for a in affinities], k), xs, graphs)]
    for past in settings["past_weights"]:
        rows.append(evaluate(f"evolutionary {past}", evolutionary(affinities, k, past)[0], xs, graphs))
    labels, weights = evolutionary(affinities, k, iterations=settings["affect_iterations"])
    rows.append(evaluate("affect", labels, xs, graphs) | {"past_weights": ";".join(f"{w:.3f}" for w in weights)})
    for coupling in settings["couplings"]:
        rows.append(evaluate(f"supra {coupling}", supra(affinities, [ids] * len(periods), k, coupling), xs, graphs))
    labels = supra(affinities, [ids] * len(periods), k, settings["coupling"])
    refined = refine_over_time(xs, graphs, [ids] * len(periods), labels, k, config["model"]["refine_weight"], settings["smoothing"])
    rows.append(evaluate(f"supra {settings['coupling']} + refinement", refined, xs, graphs))
    rows.append(evaluate("fixed typology", fixed_typology(xs, affinities, k), xs, graphs))
    table = pd.DataFrame(rows)
    table.to_csv(out / "dynamics_approaches.csv", index=False)
    print(f"dynamics on {len(ids)} municipalities present in all windows {periods[0]}..{periods[-1]}")
    print(table.drop(columns="past_weights", errors="ignore").round(3).to_string(index=False))


def track(config):
    out = config["paths"]["results"]
    out.mkdir(parents=True, exist_ok=True)
    data = load_data(config)
    periods, shots = snapshots(config, data)
    compare_dynamics(config, shots, periods, out)
    ids = [shots[p]["ids"] for p in periods]
    k = config["model"]["k"]
    labels = supra([shots[p]["affinity"] for p in periods], ids, k, config["dynamics"]["coupling"])
    xs, graphs = [shots[p]["x"] for p in periods], [shots[p]["graph"] for p in periods]
    labels = refine_over_time(xs, graphs, ids, labels, k, config["model"]["refine_weight"], config["dynamics"]["smoothing"])
    types = pd.concat([pd.DataFrame({"territory_id": i, "period": p, "type": l}) for p, i, l in zip(periods, ids, labels)], ignore_index=True)
    types = types.join(data["territories"][["name", "region"]], on="territory_id")
    types.to_csv(out / "types.csv", index=False)
    wide = types.pivot(index="territory_id", columns="period", values="type").dropna()
    transitions = pd.crosstab(wide[periods[0]].astype(int), wide[periods[-1]].astype(int))
    transitions.to_csv(out / "transitions.csv")
    changed = (wide[periods[0]] != wide[periods[-1]]).mean()
    print(f"types for {types['territory_id'].nunique()} municipalities in {len(periods)} windows, {changed:.1%} changed type from {periods[0]} to {periods[-1]}")
    print(transitions.to_string())