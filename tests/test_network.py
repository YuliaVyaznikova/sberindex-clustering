import numpy as np
import pytest
from scipy import sparse

from src.compare import moran_index
from src.network import dtw_distances, lagged_correlation


def test_lagged_correlation_finds_known_shift():
    rng = np.random.default_rng(0)
    leader = rng.normal(size=60)
    follower = np.roll(leader, 2) + rng.normal(scale=0.1, size=60)
    series = np.column_stack([leader, follower, rng.normal(size=60)])[2:]
    best, lags = lagged_correlation(series, 3)
    assert best[1, 0] > 0.9
    assert lags[1, 0] == 2 and lags[0, 1] == -2
    assert abs(best[2, 0]) < 0.5


def test_dtw_is_zero_for_identical_series_and_absorbs_small_shifts():
    rng = np.random.default_rng(1)
    base = rng.normal(size=12)
    series = np.column_stack([base, base, np.roll(base, 1), rng.normal(size=12)])
    d = dtw_distances(series, 2)
    assert d[0, 1] == pytest.approx(0)
    assert np.allclose(d, d.T)
    assert d[0, 2] < d[0, 3]


def test_moran_index_ignores_isolated_nodes():
    w = sparse.csr_matrix(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=float))
    assert np.isfinite(moran_index(w, np.array([1.0, 2.0, 5.0])))