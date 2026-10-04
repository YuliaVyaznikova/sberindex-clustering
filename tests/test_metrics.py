import networkx as nx
import numpy as np
import pytest
from scipy import sparse

from src.metrics import graph_scores, panel


def random_graph(n=60, p=0.1, seed=0):
    rng = np.random.default_rng(seed)
    upper = sparse.triu(sparse.random(n, n, density=p, random_state=seed, data_rvs=lambda size: rng.uniform(0.1, 1, size)), 1)
    return (upper + upper.T).tocsr()


def two_triangles():
    edges = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)]
    rows, cols = zip(*edges)
    w = sparse.coo_matrix((np.ones(len(edges)), (rows, cols)), shape=(6, 6))
    return (w + w.T).tocsr(), np.array([0, 0, 0, 1, 1, 1])


def test_modularity_matches_networkx():
    w = random_graph()
    labels = np.random.default_rng(1).integers(4, size=w.shape[0])
    graph = nx.from_scipy_sparse_array(w)
    parts = [set(np.flatnonzero(labels == c)) for c in np.unique(labels)]
    assert graph_scores(w, labels)["MQ"] == pytest.approx(nx.community.modularity(graph, parts, weight="weight"))


def test_two_triangles_by_hand():
    w, labels = two_triangles()
    scores = graph_scores(w, labels)
    assert scores["AVI"] == pytest.approx(6 / 7)
    assert scores["AVU"] == pytest.approx(1.0)
    assert scores["within"] == pytest.approx(12 / 14)
    assert scores["MQ"] == pytest.approx(12 / 14 - 2 * (7 / 14) ** 2)


def test_avu_is_one_third_for_any_partition_into_three():
    w = random_graph(seed=2)
    rng = np.random.default_rng(3)
    for _ in range(5):
        assert graph_scores(w, rng.integers(3, size=w.shape[0]))["AVU"] == pytest.approx(1 / 3)


def test_panel_on_separated_clusters():
    rng = np.random.default_rng(4)
    labels = np.repeat([0, 1, 2], 30)
    x = np.eye(3)[labels] * 10 + rng.normal(size=(90, 3))
    w = sparse.csr_matrix((labels[:, None] == labels[None, :]).astype(float) - np.eye(90))
    result = panel(x, w, labels, rng)
    assert result["k"] == 3
    assert result["SW"] > 0.8
    assert result["AVI"] == pytest.approx(1.0)
    assert result["MQ_lift"] > 0.5