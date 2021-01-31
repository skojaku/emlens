import faiss
import numpy as np
from scipy import sparse, stats


def make_knn_graph(emb, k=5):
    """Construct the kNN graph.

    :param emb: embedding vectors
    :type emb: numpy.ndarray (num_entities, dim)
    :param k: Number of neighbors, defaults to 5
    :type k: int, optional
    :return: KNN graph
    :rtype: sparse.csr_matrix

    .. highlight:: python
    .. code-block:: python

        >>> import emlens
        >>> import numpy as np
        >>> emb = np.random.randn(100, 20)
        >>> A = emlens.make_knn_graph(emb, k = 10)
    """
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(emb.astype(np.float32))
    distances, indices = index.search(emb.astype(np.float32), k=k)
    r = np.outer(np.arange(indices.shape[0]), np.ones((1, k))).astype(int)
    c = indices.astype(int).reshape(-1)
    r = r.reshape(-1)
    N = emb.shape[0]
    A = sparse.csr_matrix((np.ones_like(r), (r, c)), shape=(N, N))
    A = A + A.T
    A.data = np.ones_like(A.data)
    return A


def assortativity(emb, y, A=None, k=5):
    """Calculate the assortativity for a KNN graph constructed based on the.

    :param emb: embedding vectors
    :type emb: numpy.ndarray (num_entities, dim)
    :param y: labels for entities
    :type y: nuimpy.ndarray or list
    :param A: precomputed graph, defaults to None
    :type A: scipy.csr_matrix, optional
    :param k: Number of neighbors, defaults to 5
    :type k: int, optional
    :return: assortativity index
    :rtype: sparse.csr_matrix

    .. highlight:: python
    .. code-block:: python

        >>> import emlens
        >>> import numpy as np
        >>> emb = np.random.randn(100, 20)
        >>> y = np.random.randn(emb.shape[0])
        >>> rho = emlens.assortativity(emb, y)
    """

    if A is None:
        A = make_knn_graph(emb, k=k)
    r, c, v = sparse.find(A)

    return stats.pearsonr(y[r], y[c])[0]


def modularity(emb, y, A=None, k=5):
    """Calculate the assortativity for a KNN graph constructed based on the.

    :param emb: embedding vectors
    :type emb: numpy.ndarray (num_entities, dim)
    :param y: labels for entities
    :type y: nuimpy.ndarray or list
    :param A: precomputed graph, defaults to None
    :type A: scipy.csr_matrix, optional
    :param k: Number of neighbors, defaults to 5
    :type k: int, optional
    :return: assortativity index
    :rtype: sparse.csr_matrix

    .. highlight:: python
    .. code-block:: python

        >>> import emlens
        >>> import numpy as np
        >>> emb = np.random.randn(100, 20)
        >>> y = np.random.choice(10, 100)
        >>> rho = emlens.modularity(emb, y)
    """

    if A is None:
        A = make_knn_graph(emb, k=k)
    r, c, v = sparse.find(A)

    deg = np.array(A.sum(axis=0))
    labels, yids = np.unique(y, return_inverse=True)
    U = sparse.csr_matrix(
        (np.ones_like(yids), (np.arange(yids.size), yids)),
        shape=(yids.size, len(labels)),
    )
    D = np.array(deg.reshape(1, -1) @ U).reshape(-1)
    Q = np.trace((U.T @ A @ U) - np.outer(D, D) / np.sum(D)) / np.sum(D)
    return Q


def pairwise_dot_sim(emb, y):
    """Pairwise dot similarity between groups. The dot similarity between two
    groups is computed by averaging over the dot similarity between entities in
    the groups.

    :param emb: embedding
    :type emb: numpy.ndarray
    :param y: group index
    :type y: numpy.ndarray or list
    :return: Similarity matrix S, and labels for groups
    :rtype: numpy.ndarray, numpy.ndarray

    .. highlight:: python
    .. code-block:: python

        >>> import emlens
        >>> import numpy as np
        >>> import seaborn as sns
        >>> emb = np.random.randn(100, 20)
        >>> y = np.random.choice(10, 100)
        >>> S, labels = emlens.pairwise_dot_sim(emb, y)
        >>> sns.heatmap(pd.DataFrame(S, index = labels, columns = labels))
    """
    labels, yids = np.unique(y, return_inverse=True)

    U = sparse.csr_matrix(
        (np.ones_like(yids), (np.arange(yids.size), yids)),
        shape=(emb.shape[0], len(labels)),
    )
    U = U @ sparse.diags(1 / np.maximum(1, np.array(U.sum(axis=0)).reshape(-1)))

    Uemb = U.T @ emb
    S = Uemb @ Uemb.T
    return S, labels


def pairwise_distance(emb, y):
    """Pairwise distance between groups. The distance between two groups is the
    distance between the centroid of the groups.

    :param emb: embedding
    :type emb: numpy.ndarray
    :param y: group index
    :type y: numpy.ndarray or list
    :return: Distance matrix D, and labels for groups
    :rtype: numpy.ndarray, numpy.ndarray

    .. highlight:: python
    .. code-block:: python

        >>> import emlens
        >>> import numpy as np
        >>> import seaborn as sns
        >>> emb = np.random.randn(100, 20)
        >>> y = np.random.choice(10, 100)
        >>> S, labels = emlens.pairwise_distance(emb, y)
        >>> sns.heatmap(pd.DataFrame(S, index = labels, columns = labels))
    """
    labels, yids = np.unique(y, return_inverse=True)
    U = sparse.csr_matrix(
        (np.ones_like(yids), (np.arange(yids.size), yids)),
        shape=(emb.shape[0], len(labels)),
    )
    U = U @ sparse.diags(1 / np.maximum(1, np.array(U.sum(axis=0)).reshape(-1)))
    Uemb = U.T @ emb
    K = Uemb.shape[0]
    D = np.zeros((K, K))
    for k in range(K):
        for ll in range(K):
            if k < ll:
                d = np.sqrt(np.sum((Uemb[ll, :] - Uemb[k, :]) ** 2))
                D[k, ll] = d
                D[ll, k] = d
    return D, labels


def radius_of_gyration(emb, distance_function="euc"):
    """Calculate the radius of gyration -- atypicalness for sets of embedding
    vectors For the detail, please read
    https://en.wikipedia.org/wiki/Radius_of_gyration.

    :param emb: embedding vector (num_entities, dim)
    :type emb: numpy.ndarray
    :param distance_function: only cosine distance ('cos') and eucliduan distance ('euc') are supported.
    :type distance_function: str
    :return: ROG value
    :rtype: float

    .. highlight:: python
    .. code-block:: python

        >>> import emlens
        >>> import numpy as np
        >>> emb = np.random.randn(100, 20)
        >>> rog = emlens.pairwise_distance(emb, 'cos')
    """

    mean_vec = emb.mean(axis=0)
    if distance_function == "euc":
        diff_emb = emb - mean_vec
        d = np.sum(np.power(diff_emb, 2), axis=1)
        rog = np.sqrt(np.mean(d))
    elif distance_function == "cos":
        mean_vec = mean_vec / np.linalg.norm(mean_vec)
        emb = (emb.T / np.array(np.linalg.norm(emb, axis=1)).reshape(-1)).T
        d = 1 - emb @ mean_vec
        rog = np.sqrt(np.mean(d ** 2))
    else:
        raise NotImplementedError(
            "radious of gyration function does not support distance_function: {}".format(
                distance_function
            )
        )

    return rog
