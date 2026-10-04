import numpy as np
import torch
from scipy import sparse


def clean(w):
    w = sparse.csr_matrix(w, dtype=np.float64)
    w = w.maximum(w.T)
    w.setdiag(0)
    w.eliminate_zeros()
    return w


def normalized(w):
    degree = np.asarray(w.sum(axis=1)).ravel()
    scale = sparse.diags(1 / np.sqrt(np.where(degree > 0, degree, 1)))
    return scale @ w @ scale


def to_torch(w):
    w = w.tocoo()
    index = torch.tensor(np.vstack([w.row, w.col]), dtype=torch.long)
    return torch.sparse_coo_tensor(index, torch.tensor(w.data, dtype=torch.float32), w.shape, check_invariants=False).coalesce()


class Network(torch.nn.Module):
    def __init__(self, features, hidden, k, dropout):
        super().__init__()
        self.convolution = torch.nn.Linear(features, hidden, bias=False)
        self.skip = torch.nn.Linear(features, hidden, bias=False)
        self.assign = torch.nn.Linear(hidden, k)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, smoothed, x):
        hidden = torch.nn.functional.selu(self.convolution(smoothed) + self.skip(x))
        return torch.softmax(self.assign(self.dropout(hidden)), dim=1)


def objective(c, a, degree, edges, k, collapse):
    within = torch.sum(c * torch.sparse.mm(a, c))
    expected = torch.sum((degree @ c) ** 2) / (2 * edges)
    modularity = (within - expected) / (2 * edges)
    balance = torch.linalg.norm(c.sum(dim=0)) / c.shape[0] * np.sqrt(k) - 1
    return -modularity + collapse * balance


def train(smoothed, x, a, degree, edges, k, hidden, dropout, collapse, epochs, rate):
    network = Network(x.shape[1], hidden, k, dropout)
    optimizer = torch.optim.Adam(network.parameters(), lr=rate)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = objective(network(smoothed, x), a, degree, edges, k, collapse)
        loss.backward()
        optimizer.step()
    network.eval()
    with torch.no_grad():
        c = network(smoothed, x)
        return c.argmax(dim=1).numpy(), float(objective(c, a, degree, edges, k, collapse))


def dmon(x, w, k, seed=0, n_init=1, hidden=64, dropout=0.5, collapse=1.0, epochs=1000, rate=0.001, threads=2):
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    w = clean(w)
    a = to_torch(w)
    smoothed = torch.tensor(normalized(w) @ np.asarray(x, dtype=np.float64), dtype=torch.float32)
    features = torch.tensor(np.asarray(x), dtype=torch.float32)
    degree = torch.tensor(np.asarray(w.sum(axis=1)).ravel(), dtype=torch.float32)
    edges = degree.sum() / 2
    best, best_loss = None, np.inf
    for _ in range(n_init):
        labels, loss = train(smoothed, features, a, degree, edges, k, hidden, dropout, collapse, epochs, rate)
        if loss < best_loss:
            best, best_loss = labels, loss
    return best