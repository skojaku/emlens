import faiss
import numpy as np
from joblib import Parallel, delayed
from scipy import sparse, stats
from sklearn import utils
from sklearn.metrics import average_precision_score, roc_curve
from tqdm import tqdm

from .semaxis import calcSemAxis


def make_knn_graph(X, k=5):
    """Construct the kNN graph.

    :param X: embedding vectors
    :type X: numpy.ndarray (num_entities, dim)
    :param k: Number of neighbors, defaults to 5
    :type k: int, optional
    :return: KNN graph
    :rtype: sparse.csr_matrix
    """
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
    """Calculate the assortativity for a KNN graph constructed based on the.

    :param emb: embedding vectors
    :type emb: numpy.ndarray (num_entities, dim)
    :param y: labels for entities
    :type y: nuimpy.ndarray or list
    :param label_type: type of labels, defaults to "cont"
    :type label_type: str, optional
    :param A: precomputed graph, defaults to None
    :type A: scipy.csr_matrix, optional
    :param k: Number of neighbors, defaults to 5
    :type k: int, optional
    :return: assortativity index
    :rtype: sparse.csr_matrix
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
    """Pairwise dot similarity between groups. The dot similarity between two
    groups is computed by averaging over the dot similarity between entities in
    the groups.

    :param X: embedding
    :type X: numpy.ndarray
    :param y: group index
    :type y: numpy.ndarray or list
    :return: Similarity matrix S, and labels for groups
    :rtype: numpy.ndarray, numpy.ndarray
    """
    labels, yids = np.unique(y, return_inverse=True)

    U = sparse.csr_matrix(
        (np.ones_like(yids), (np.arange(yids.size), yids)),
        shape=(X.shape[0], len(labels)),
    )
    U = U @ sparse.diags(1 / np.maximum(1, np.array(U.sum(axis=0)).reshape(-1)))

    UX = U.T @ X
    S = UX @ UX.T
    return S, labels


def calc_pairwise_distance(X, y):
    """Pairwise distance between groups. The distance between two groups is the
    distance between the centroid of the groups.

    :param X: embedding
    :type X: numpy.ndarray
    :param y: group index
    :type y: numpy.ndarray or list
    :return: Distance matrix D, and labels for groups
    :rtype: numpy.ndarray, numpy.ndarray
    """
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
