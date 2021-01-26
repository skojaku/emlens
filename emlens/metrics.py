import faiss
import numpy as np
from joblib import Parallel, delayed
from scipy import sparse, stats
from sklearn import utils
from sklearn.metrics import average_precision_score, roc_curve
from tqdm import tqdm

from .semaxis import calcSemAxis


def make_knn_graph(X, k=5):
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X.astype(np.float32))
    distances, indices = index.search(X.astype(np.float32), k=k)
    r = np.outer(np.arange(indices.shape[0]), np.ones((1, k))).astype(int)
    c = indices.astype(int).reshape(-1)
    r = r.reshape(-1)
    N = X.shape[0]
    A = sparse.csr_matrix((np.ones_like(r), (r, c)), shape=(N, N))
    A = A + A.T
    A.data = np.ones_like(A.data)
    return A


def calc_assortativity(emb, y, label_type="cont", A=None, k=5):
    """Calculate the assortativity for a KNN graph constructed based on the
    embedding.

    Params
    ------
    emb: numpy.ndarray (num_entities, dim)
    y: labels for each entity
    label_type: type of the label. label_type="cont" if y is continuous. label = "disc" if y is discrete (i.e., groups labels).
    A: scipy.sparse matrix (Optional; Default None). If given, use this graph instead of constructing the KNN from the scratch.
    k: # of neighbors

    Returns
    -------
    corr: Assortativity of the KNN graph

    Example
    -------
    >>> indeg = np.array(net.sum(axis = 0)).reshape(-1)
    >>> calc_assortativity(emb, indeg)
    """
    if A is None:
        A = make_knn_graph(emb, k=k)
    r, c, v = sparse.find(A)

    if label_type == "cont":
        return stats.pearsonr(y[r], y[c])[0]
    elif label_type == "disc":
        deg = np.array(A.sum(axis=0))
        labels, yids = np.unique(y, return_inverse=True)
        U = sparse.csr_matrix(
            (np.ones_like(yids), (np.arange(yids.size), yids)),
            shape=(yids.size, len(labels)),
        )
        D = np.array(deg.reshape(1, -1) @ U).reshape(-1)
        Q = np.trace((U.T @ A @ U) - np.outer(D, D) / np.sum(D)) / np.sum(D)
        return Q


def calc_pairwise_dot_sim(X, y):
    labels, yids = np.unique(y, return_inverse=True)

    U = sparse.csr_matrix(
        (np.ones_like(yids), (np.arange(yids.size), yids)),
        shape=(X.shape[0], len(labels)),
    )
    U = U @ sparse.diags(1 / np.maximum(1, np.array(U.sum(axis=0)).reshape(-1)))

    UX = U.T @ X
    S = UX @ UX.T
    return S, labels


def calc_cluster_separativity(emb):
    ypred = emb.reshape(-1)
    y = np.eye(emb.shape[0]).reshape(-1)
    return average_precision_score(y, ypred)


def calc_pairwise_distance(X, y):
    labels, yids = np.unique(y, return_inverse=True)
    U = sparse.csr_matrix(
        (np.ones_like(yids), (np.arange(yids.size), yids)),
        shape=(X.shape[0], len(labels)),
    )
    U = U @ sparse.diags(1 / np.maximum(1, np.array(U.sum(axis=0)).reshape(-1)))
    UX = U.T @ X
    K = UX.shape[0]
    D = np.zeros((K, K))
    for k in range(K):
        for ll in range(K):
            if k < ll:
                d = np.sqrt(np.sum((UX[ll, :] - UX[k, :]) ** 2))
                D[k, ll] = d
                D[ll, k] = d
    return D, labels
