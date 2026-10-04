import networkx as nx
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture

from .canus import canus
from .dmon import dmon
from .kefrin import joint_distances, kefrin, prepare, scatter


def relabel(labels):
    _, codes = np.unique(labels, return_inverse=True)
    order = np.argsort(-np.bincount(codes))
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return rank[codes]


def kmeans(x, k, seed=0):
    return relabel(KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(x))


def ward(x, k, seed=0):
    return relabel(AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(x))


def gmm(x, k, seed=0):
    return relabel(GaussianMixture(n_components=k, covariance_type="diag", n_init=3, random_state=seed).fit_predict(x))


def spectral(w, k, seed=0):
    model = SpectralClustering(n_clusters=k, affinity="precomputed", assign_labels="cluster_qr", random_state=seed)
    return relabel(model.fit_predict(w))


def from_communities(communities, n):
    labels = np.empty(n, dtype=int)
    for c, nodes in enumerate(communities):
        labels[list(nodes)] = c
    return relabel(labels)


def louvain(w, resolution, seed=0):
    graph = nx.from_scipy_sparse_array(w)
    return from_communities(nx.community.louvain_communities(graph, weight="weight", resolution=resolution, seed=seed), w.shape[0])


def leiden(w, resolution, seed=0):
    graph = nx.from_scipy_sparse_array(w)
    return from_communities(nx.community.leiden_communities(graph, weight="weight", resolution=resolution, metric="modularity", seed=seed), w.shape[0])


def fuse(graph, attribute_graph, alpha):
    return (alpha * graph / graph.sum() + (1 - alpha) * attribute_graph / attribute_graph.sum()).tocsr()


def fused_spectral(graph, attribute_graph, k, alpha, seed=0):
    return spectral(fuse(graph, attribute_graph, alpha), k, seed)


def refine_costs(x, graph, labels, k, weight):
    y, p = prepare(x, graph, "euclidean")
    rho, xi = (1 - weight) / scatter(y), weight / scatter(p)
    found = np.unique(labels)
    cy = np.array([y[labels == c].mean(axis=0) for c in found])
    cp = np.array([p[labels == c].mean(axis=0) for c in found])
    costs = np.full((len(x), k), np.inf)
    costs[:, found] = len(x) * joint_distances(y, p, cy, cp, rho, xi)
    return costs


def refine(costs, labels):
    moved = costs.argmin(axis=1)
    return moved if len(np.unique(moved)) == len(np.unique(labels)) else labels


def refined_spectral(x, graph, attribute_graph, k, alpha, weight, seed=0):
    labels = fused_spectral(graph, attribute_graph, k, alpha, seed)
    return relabel(refine(refine_costs(x, graph, labels, k, weight), labels))


def joint(name, x, graph, attribute_graph, k, alpha, weight, seed=0):
    if name == "fused_spectral":
        return fused_spectral(graph, attribute_graph, k, alpha, seed)
    if name == "refined_spectral":
        return refined_spectral(x, graph, attribute_graph, k, alpha, weight, seed)
    if name == "dmon":
        return relabel(dmon(x, fuse(graph, attribute_graph, alpha), k, seed=seed))
    family, metric = name.split("_")
    run = kefrin if family == "kefrin" else canus
    return relabel(run(x, graph, k, alpha=alpha, metric=metric, seed=seed))


ATTRIBUTE_METHODS = {"kmeans": kmeans, "ward": ward, "gmm": gmm}
GRAPH_METHODS = {"louvain": louvain, "leiden": leiden}
JOINT_METHODS = ["kefrin_euclidean", "kefrin_cosine", "canus_euclidean", "canus_cosine", "dmon", "fused_spectral", "refined_spectral"]