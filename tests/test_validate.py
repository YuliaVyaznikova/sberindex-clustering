import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.validate import eta_squared_h, moran, participation, rank_methods


def path_graph(n):
    w = sparse.diags([np.ones(n - 1), np.ones(n - 1)], [-1, 1], shape=(n, n))
    return sparse.csr_matrix(w)


def test_participation_by_hand():
    edges = [(0, 1), (0, 2), (3, 4), (3, 1)]
    rows, cols = zip(*edges)
    w = sparse.coo_matrix((np.ones(len(edges)), (rows, cols)), shape=(5, 5))
    w = (w + w.T).tocsr()
    codes = np.array([0, 0, 0, 1, 1])
    p, own, other = participation(w, codes, 2)
    assert p[0] == pytest.approx(0.0)
    assert p[3] == pytest.approx(0.5)
    assert own[3] == pytest.approx(0.5)
    assert other[3] == 0


def test_moran_detects_smooth_values():
    rng = np.random.default_rng(0)
    w = path_graph(200)
    smooth, p_smooth = moran(w, np.sin(np.linspace(0, 6, 200)), rng)
    noise, p_noise = moran(w, rng.normal(size=200), rng)
    assert smooth > 0.9 and p_smooth < 0.01
    assert abs(noise) < 0.2 and p_noise > 0.01


def test_eta_squared_h():
    rng = np.random.default_rng(1)
    labels = np.repeat([0, 1, 2], 50)
    separated = labels * 10 + rng.normal(size=150)
    mixed = rng.normal(size=150)
    n, k = 150, 3
    h = 12 / (n * (n + 1)) * (50 * np.array([25.5, 75.5, 125.5]) ** 2).sum() - 3 * (n + 1)
    assert eta_squared_h(separated, labels) == pytest.approx((h - k + 1) / (n - k))
    assert abs(eta_squared_h(mixed, labels)) < 0.05


def test_threshold_ranking_puts_dominant_configuration_first():
    rows = [
        {"method": "a", "alpha": np.nan, "k": 3, "SW": 0.3, "CH": 300, "S_Dbw": 0.5, "AVI": 0.9, "AVU": 0.1, "MQ": 0.6},
        {"method": "b", "alpha": 0.5, "k": 3, "SW": 0.2, "CH": 200, "S_Dbw": 0.7, "AVI": 0.8, "AVU": 0.2, "MQ": 0.5},
        {"method": "c", "alpha": np.nan, "k": 3, "SW": 0.1, "CH": 100, "S_Dbw": 0.9, "AVI": 0.7, "AVU": 0.3, "MQ": 0.4},
        {"method": "d", "alpha": np.nan, "k": 4, "SW": 0.9, "CH": 900, "S_Dbw": 0.1, "AVI": 0.99, "AVU": 0.01, "MQ": 0.9},
    ]
    ranking = rank_methods(pd.DataFrame(rows).assign(within=0.5, stability=0.9), 3)
    assert list(ranking.index) == ["a", "b 0.5", "c"]
    assert ranking.loc["a", "worst"] == 0
    assert ranking.loc["c", "worst"] == 6