import numpy as np
from scipy import sparse
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


def s_dbw(x, labels):
    clusters = np.unique(labels)
    k = len(clusters)
    centers = np.array([x[labels == c].mean(axis=0) for c in clusters])
    sigma = np.array([np.linalg.norm(x[labels == c].var(axis=0)) for c in clusters])
    scat = sigma.mean() / np.linalg.norm(x.var(axis=0))
    stdev = np.sqrt(sigma.sum()) / k

    def density(point, members):
        return np.sum(np.linalg.norm(members - point, axis=1) <= stdev)

    dens = 0.0
    for a in range(k):
        for b in range(k):
            if a != b:
                members = x[(labels == clusters[a]) | (labels == clusters[b])]
                top = max(density(centers[a], members), density(centers[b], members))
                if top > 0:
                    dens += density((centers[a] + centers[b]) / 2, members) / top
    return scat + dens / (k * (k - 1))


def block_sums(w, labels):
    clusters, codes = np.unique(labels, return_inverse=True)
    member = sparse.csr_matrix((np.ones(len(codes)), (np.arange(len(codes)), codes)), shape=(len(codes), len(clusters)))
    return np.asarray((member.T @ w @ member).todense()), np.bincount(codes)


def graph_scores(w, labels):
    s, sizes = block_sums(w, labels)
    k = len(sizes)
    inside = np.diag(s)
    total = s.sum(axis=1)
    outside = total - inside
    avi = float(np.mean(np.divide(inside, total, out=np.zeros(k), where=total > 0)))
    pairs = [s[i, j] / (outside[i] + outside[j] - s[i, j]) if outside[i] + outside[j] - s[i, j] > 0 else 0.0
             for i in range(k) for j in range(k) if i != j]
    avu = float(np.mean(pairs))
    two_m = s.sum()
    internal = inside / 2
    return {
        "AVI": avi,
        "AVU": avu,
        "ANUI": avi / (1 + avi * avu),
        "MQ": float(np.sum(internal / (two_m / 2) - (total / two_m) ** 2)),
        "DM": float(np.sum((internal - total ** 2 / (2 * two_m)) / sizes)),
        "within": float(inside.sum() / two_m),
    }


def feature_scores(x, labels):
    return {
        "SW": silhouette_score(x, labels),
        "CH": calinski_harabasz_score(x, labels),
        "DB": davies_bouldin_score(x, labels),
        "S_Dbw": s_dbw(x, labels),
    }


def random_level(w, labels, rng, repeats=10):
    scores = [graph_scores(w, rng.permutation(labels)) for _ in range(repeats)]
    return {name: float(np.mean([s[name] for s in scores])) for name in ["AVI", "AVU", "MQ"]}


def panel(x, w, labels, rng):
    sizes = np.bincount(np.unique(labels, return_inverse=True)[1])
    result = {"k": len(sizes), "min_share": sizes.min() / len(labels), "max_share": sizes.max() / len(labels)}
    result.update(feature_scores(x, labels))
    result.update(graph_scores(w, labels))
    level = random_level(w, labels, rng)
    result["AVI_lift"] = result["AVI"] - level["AVI"]
    result["AVU_gain"] = level["AVU"] - result["AVU"]
    result["MQ_lift"] = result["MQ"] - level["MQ"]
    return result