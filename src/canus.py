import numpy as np

from .kefrin import prepare, scatter, squared_distances


def scale(y, p, metric):
    if metric == "cosine":
        return y, p
    return y / np.sqrt(scatter(y) / len(y)), p / np.sqrt(scatter(p) / len(p))


class Centers:
    def __init__(self, y, p, start, rho, xi, step, metric):
        self.y, self.p, self.rho, self.xi, self.step, self.metric = y, p, rho, xi, step, metric
        self.ny, self.np = (y ** 2).sum(axis=1), (p ** 2).sum(axis=1)
        self.cy, self.cp = y[start].copy(), p[start].copy()
        self.ncy, self.ncp = (self.cy ** 2).sum(axis=1), (self.cp ** 2).sum(axis=1)

    def visit(self, i, band=None):
        dy = np.maximum(self.ny[i] - 2 * self.cy @ self.y[i] + self.ncy, 0)
        dp = np.maximum(self.np[i] - 2 * self.cp @ self.p[i] + self.ncp, 0)
        c = int((self.rho * dy + self.xi * dp).argmin())
        size = 2 * self.rho * np.sqrt(dy[c]) + 2 * self.xi * np.sqrt(dp[c])
        if band is None or band[0] <= size <= band[1]:
            self.move(self.cy, self.ncy, c, self.y[i], 2 * self.step * self.rho)
            self.move(self.cp, self.ncp, c, self.p[i], 2 * self.step * self.xi)
        return size

    def move(self, centers, norms, c, point, rate):
        centers[c] *= 1 - rate
        centers[c] += rate * point
        if self.metric == "cosine":
            centers[c] /= max(np.sqrt(centers[c] @ centers[c]), 1e-12)
        norms[c] = centers[c] @ centers[c]


def run_once(y, p, k, rho, xi, metric, rng, step, tau, warmup, passes):
    centers = Centers(y, p, rng.choice(len(y), k, replace=False), rho, xi, step, metric)
    sizes = [centers.visit(i) for _ in range(warmup) for i in rng.integers(len(y), size=len(y))]
    mean, spread = np.mean(sizes), np.std(sizes)
    band = (mean - tau * spread, mean + tau * spread)
    for _ in range(passes):
        for i in rng.permutation(len(y)):
            centers.visit(i, band)
    distances = rho * squared_distances(y, centers.cy) + xi * squared_distances(p, centers.cp)
    labels = distances.argmin(axis=1)
    if len(np.unique(labels)) < k:
        return None, np.inf
    return labels, distances[np.arange(len(y)), labels].sum()


def canus(x, w, k, alpha=0.5, metric="euclidean", n_init=1, seed=0, step=0.1, tau=0.3, warmup=100, passes=10):
    y, p = scale(*prepare(x, w, metric), metric)
    y, p = np.ascontiguousarray(y, dtype=np.float64), np.ascontiguousarray(p, dtype=np.float64)
    rng = np.random.default_rng(seed)
    best, best_criterion = None, np.inf
    for _ in range(n_init):
        labels, criterion = run_once(y, p, k, 1 - alpha, alpha, metric, rng, step, tau, warmup, passes)
        if criterion < best_criterion:
            best, best_criterion = labels, criterion
    return best