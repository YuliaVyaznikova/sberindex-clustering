import numpy as np
from scipy import sparse


def standardize_links(w):
    p = w.toarray() if sparse.issparse(w) else np.asarray(w, dtype=float)
    return p - p.sum(axis=1, keepdims=True) @ p.sum(axis=0, keepdims=True) / p.sum()


def unit_rows(z):
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    return z / np.where(norms > 0, norms, 1)


def scatter(z):
    return float(((z - z.mean(axis=0)) ** 2).sum())


def prepare(x, w, metric):
    y = np.asarray(x, dtype=np.float32)
    p = standardize_links(w).astype(np.float32)
    if metric == "cosine":
        y, p = unit_rows(y), unit_rows(p)
    return y, p


def squared_distances(z, centers):
    d = (z ** 2).sum(axis=1)[:, None] - 2 * z @ centers.T + (centers ** 2).sum(axis=1)[None, :]
    return np.maximum(d, 0)


def joint_distances(y, p, cy, cp, rho, xi):
    return rho * squared_distances(y, cy) + xi * squared_distances(p, cp)


def seeds(y, p, k, rho, xi, rng):
    chosen = [int(rng.integers(len(y)))]
    nearest = np.full(len(y), np.inf)
    for _ in range(k - 1):
        last = chosen[-1]
        nearest = np.minimum(nearest, joint_distances(y, p, y[last:last + 1], p[last:last + 1], rho, xi)[:, 0])
        chosen.append(int(rng.choice(len(y), p=nearest / nearest.sum())))
    return y[chosen].copy(), p[chosen].copy()


def centers(z, labels, k, metric):
    result = np.array([z[labels == c].mean(axis=0) for c in range(k)])
    return unit_rows(result) if metric == "cosine" else result


def run_once(y, p, k, rho, xi, metric, rng, max_iter):
    cy, cp = seeds(y, p, k, rho, xi, rng)
    labels = None
    for _ in range(max_iter):
        new = joint_distances(y, p, cy, cp, rho, xi).argmin(axis=1)
        if len(np.unique(new)) < k or (labels is not None and np.array_equal(new, labels)):
            break
        labels = new
        cy, cp = centers(y, labels, k, metric), centers(p, labels, k, metric)
    if labels is None or len(np.unique(labels)) < k:
        return None, np.inf
    return labels, joint_distances(y, p, cy, cp, rho, xi)[np.arange(len(y)), labels].sum()


def fit(y, p, k, alpha, metric, n_init=5, seed=0, max_iter=60):
    rho, xi = (1 - alpha) / scatter(y), alpha / scatter(p)
    rng = np.random.default_rng(seed)
    best, best_criterion = None, np.inf
    for _ in range(n_init):
        labels, criterion = run_once(y, p, k, rho, xi, metric, rng, max_iter)
        if criterion < best_criterion:
            best, best_criterion = labels, criterion
    return best


def kefrin(x, w, k, alpha=0.5, metric="cosine", **options):
    y, p = prepare(x, w, metric)
    return fit(y, p, k, alpha, metric, **options)


def default_alpha(x, w, metric):
    y, p = prepare(x, w, metric)
    return scatter(p) / (scatter(y) + scatter(p))