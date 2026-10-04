import numpy as np
import pytest
from scipy import sparse
from sklearn.metrics import adjusted_rand_score

from src import methods
from src.dynamics import refine_over_time, supra
from src.network import cosine, knn


def blobs(n=40, k=3, dimensions=6, spread=0.6, seed=0):
    rng = np.random.default_rng(seed)
    truth = np.repeat(np.arange(k), n)
    centers = rng.normal(scale=3, size=(k, dimensions))
    return centers[truth] + rng.normal(scale=spread, size=(len(truth), dimensions)), truth


def graphs(x):
    return knn(cosine(x[:, :3]), 8), knn(cosine(x), 8)


def test_relabel_orders_clusters_by_size():
    assert methods.relabel(np.array([5, 5, 2, 7, 7, 7])).tolist() == [1, 1, 2, 0, 0, 0]


def test_knn_graph_is_symmetric_without_loops():
    x, _ = blobs()
    w = knn(cosine(x), 5)
    assert abs(w - w.T).max() == 0
    assert w.diagonal().sum() == 0
    assert (np.diff(w.indptr) >= 5).all()


def test_fuse_mixes_normalized_graphs():
    x, _ = blobs()
    w, wa = graphs(x)
    fused = methods.fuse(w, wa, 0.3)
    assert fused.sum() == pytest.approx(1.0)
    assert abs(methods.fuse(w, wa, 1.0) - w / w.sum()).max() == pytest.approx(0)


@pytest.mark.parametrize("name", methods.JOINT_METHODS)
def test_joint_methods_are_reproducible_and_find_clear_clusters(name):
    x, truth = blobs()
    w, wa = graphs(x)
    first = methods.joint(name, x, w, wa, 3, 0.3, 0.7, seed=1)
    second = methods.joint(name, x, w, wa, 3, 0.3, 0.7, seed=1)
    assert (first == second).all()
    assert len(np.unique(first)) == 3
    assert adjusted_rand_score(truth, first) > 0.8


def test_attribute_and_graph_methods_recover_clear_clusters():
    x, truth = blobs()
    w, _ = graphs(x)
    for method in methods.ATTRIBUTE_METHODS.values():
        assert adjusted_rand_score(truth, method(x, 3)) > 0.9
    assert adjusted_rand_score(truth, methods.spectral(w, 3)) > 0.9
    for method in methods.GRAPH_METHODS.values():
        labels = method(w, 1.0)
        purity = sum(np.bincount(truth[labels == c]).max() for c in np.unique(labels)) / len(truth)
        assert purity > 0.95


def test_supra_keeps_types_across_identical_windows():
    x, truth = blobs()
    w, wa = graphs(x)
    affinity = methods.fuse(w, wa, 0.3)
    ids = np.arange(len(x))
    first, second = supra([affinity, affinity], [ids, ids], 3, 1.0)
    assert (first == second).all()
    assert adjusted_rand_score(truth, first) > 0.9


def test_refinement_moves_a_wrong_label_back():
    x, truth = blobs()
    w, _ = graphs(x)
    labels = truth.copy()
    labels[0] = 1
    assert (methods.refine(methods.refine_costs(x, w, labels, 3, 0.0), labels) == truth).all()


def test_refinement_keeps_clear_clusters():
    x, truth = blobs()
    w, wa = graphs(x)
    assert adjusted_rand_score(truth, methods.refined_spectral(x, w, wa, 3, 0.3, 0.7)) == pytest.approx(1.0)


def test_full_smoothing_gives_one_type_per_municipality():
    x, truth = blobs()
    w, _ = graphs(x)
    noisy = x + np.random.default_rng(1).normal(scale=1.5, size=x.shape)
    ids = np.arange(len(x))
    first, second = refine_over_time([x, noisy], [w, knn(cosine(noisy[:, :3]), 8)], [ids, ids], [truth, truth], 3, 0.3, 1.0)
    assert (first == second).all()