import networkx as nx
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture

from .kefrin import kefrin


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


def louvain(w, resolution, seed=0):
    graph = nx.from_scipy_sparse_array(w)
    labels = np.empty(w.shape[0], dtype=int)
    for c, nodes in enumerate(nx.community.louvain_communities(graph, weight="weight", resolution=resolution, seed=seed)):
        labels[list(nodes)] = c
    return relabel(labels)


def fuse(graph, attribute_graph, alpha):
    return (alpha * graph / graph.sum() + (1 - alpha) * attribute_graph / attribute_graph.sum()).tocsr()


def fused_spectral(graph, attribute_graph, k, alpha, seed=0):
    return spectral(fuse(graph, attribute_graph, alpha), k, seed)


def joint(name, x, graph, attribute_graph, k, alpha, seed=0):
    if name == "fused_spectral":
        return fused_spectral(graph, attribute_graph, k, alpha, seed)
    metric = name.removeprefix("kefrin_")
    return relabel(kefrin(x, graph, k, alpha=alpha, metric=metric, seed=seed))


ATTRIBUTE_METHODS = {"kmeans": kmeans, "ward": ward, "gmm": gmm}
JOINT_METHODS = ["kefrin_euclidean", "kefrin_cosine", "fused_spectral"]