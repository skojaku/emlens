import faiss
import numpy as np
from scipy import sparse, stats


def make_knn_graph(emb, k=5):
    """Construct the k-nearest neighbor graph from embedding vectors.

    :param emb: embedding vectors
    :type emb: numpy.ndarray (num_entities, dim)
    :param k: Number of nearest neighbors, defaults to 5
    :type k: int, optional
    :return: The adjacency matrix of the k-nearest neighbor graph
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
    """Calculate the assortativity of y for close entities in the embedding
    space. A positive/negative assortativity indicates that the close entities
    tend to have a similar/dissimilar y. Zero assortativity means y is
    independent of the embedding.

    :param emb: embedding vectors
    :type emb: numpy.ndarray (num_entities, dim)
    :param y: feature of y
    :type y: numpy.ndarray
    :param A: precomputed adjacency matrix of the graph. If None, a k-nearest neighbor graph will be constructed., defaults to None
    :type A: scipy.csr_matrix, optional
    :param k: Number of the nearest neighbors, defaults to 5
    :type k: int, optional
    :return: assortativity
    :rtype: float

    .. highlight:: python
    .. code-block:: python

        >>> import emlens
        >>> import numpy as np
        >>> emb = np.random.randn(100, 20)
        >>> y = np.random.randn(emb.shape[0])
        >>> rho = emlens.assortativity(emb, y)

    .. note::
        To calculate the assortativity, a k-nearest neighbor graph is constructed.
        Then, the assortativity is calculated as the Pearson correlation of y between the adjacent nodes.
    """

    if A is None:
        A = make_knn_graph(emb, k=k)
    r, c, v = sparse.find(A)

    return stats.pearsonr(y[r], y[c])[0]


def modularity(emb, g, A=None, k=5):
    """Calculate the modularity of entities with group membership g. The
    modularity ranges between [-1,1], where a positive modularity indicates
    that nodes with the same group membership tend to be close each other. Zero
    modularity means that membership g is independent of the embedding.

    :param emb: embedding vectors
    :type emb: numpy.ndarray (num_entities, dim)
    :param g: group membership for entities
    :type g: numpy.ndarray
    :param A: precomputed adjacency matrix of the graph. If None, a k-nearest neighbor graph will be constructed., defaults to None
    :type A: scipy.csr_matrix, optional
    :param k: Number of the nearest neighbors, defaults to 5
    :type k: int, optional
    :return: modularity
    :rtype: float

    .. highlight:: python
    .. code-block:: python

        >>> import emlens
        >>> import numpy as np
        >>> emb = np.random.randn(100, 20)
        >>> g = np.random.choice(10, 100)
        >>> rho = emlens.modularity(emb, g)
    """

    if A is None:
        A = make_knn_graph(emb, k=k)
    r, c, v = sparse.find(A)

    deg = np.array(A.sum(axis=0))
    labels, gids = np.unique(g, return_inverse=True)
    U = sparse.csr_matrix(
        (np.ones_like(gids), (np.arange(gids.size), gids)),
        shape=(gids.size, len(labels)),
    )
    D = np.array(deg.reshape(1, -1) @ U).reshape(-1)
    Q = np.trace((U.T @ A @ U) - np.outer(D, D) / np.sum(D)) / np.sum(D)
    return Q


def pairwise_dot_sim(emb, g):
    """Average dot similarity between entities in groups.

    :param emb: embedding vectors
    :type emb: numpy.ndarray (num_entities, dim)
    :param g: group membership
    :type g: numpy.ndarray (num_entities)
    :return: S, groups
    :rtype: numpy.ndarray, numpy.ndarray

        * **D**: Distance matrix for groups.

        * **groups**:  group[i] is the group for the ith row/column of D.

    .. highlight:: python
    .. code-block:: python

        >>> import emlens
        >>> import numpy as np
        >>> import seaborn as sns
        >>> emb = np.random.randn(100, 20)
        >>> g = np.random.choice(10, 100)
        >>> S, groups = emlens.pairwise_dot_sim(emb, g)
        >>> sns.heatmap(pd.DataFrame(S, index = groups, columns = groups))
    """
    groups, gids = np.unique(g, return_inverse=True)

    U = sparse.csr_matrix(
        (np.ones_like(gids), (np.arange(gids.size), gids)),
        shape=(emb.shape[0], len(groups)),
    )
    U = U @ sparse.diags(1 / np.maximum(1, np.array(U.sum(axis=0)).reshape(-1)))

    Uemb = U.T @ emb
    S = Uemb @ Uemb.T
    return S, groups


def pairwise_distance(emb, g):
    """Pairwise distance between groups. The distance between two groups is the
    distance between the centroid of the groups.

    :param emb: embedding
    :type emb: numpy.ndarray
    :param g: group membership
    :type g: numpy.ndarray (num_entities)
    :return: D, groups
    :rtype: numpy.ndarray, numpy.ndarray

        * **D**: Distance matrix for groups.

        * **groups**:  group[i] is the group for the ith row/column of D.

    .. highlight:: python
    .. code-block:: python

        >>> import emlens
        >>> import numpy as np
        >>> import seaborn as sns
        >>> emb = np.random.randn(100, 20)
        >>> g = np.random.choice(10, 100)
        >>> S, groups = emlens.pairwise_distance(emb, g)
        >>> sns.heatmap(pd.DataFrame(S, index = groups, columns = groups))
    """
    groups, gids = np.unique(g, return_inverse=True)
    U = sparse.csr_matrix(
        (np.ones_like(gids), (np.arange(gids.size), gids)),
        shape=(emb.shape[0], len(groups)),
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
    return D, groups


def rog(emb, center=None, metric="euc"):
    """Calculate the radius of gyration (ROG) for the embedding vectors. The
    ROG is a standard deviation of distance of points from a center point. See
    https://en.wikipedia.org/wiki/Radius_of_gyration.

    :param emb: embedding vector (num_entities, dim)
    :type emb: numpy.ndarray
    :param metric: The metric for the distance between points. The available metrics are cosine ('cos') euclidean ('euc') distances.
    :type metric: str
    :param center: The embedding vector for the center location. If None, the centroid of the given embedding vectors is used as the center., defaults to None
    :type center: numpy.ndarray (num_entities, 1)
    :return: ROG value
    :rtype: float

    .. highlight:: python
    .. code-block:: python

        >>> import emlens
        >>> import numpy as np
        >>> emb = np.random.randn(100, 20)
        >>> rog = emlens.pairwise_metric(emb, 'cos')
    """

    if center is None:
        center = emb.mean(axis=0)

    if metric == "euc":
        d = np.sum(np.power(emb - center, 2), axis=1)
        rog = np.sqrt(np.mean(d))
    elif metric == "cos":
        center = center / np.linalg.norm(center)
        emb = (emb.T / np.array(np.linalg.norm(emb, axis=1)).reshape(-1)).T
        d = 1 - emb @ center
        rog = np.sqrt(np.mean(d ** 2))
    else:
        raise NotImplementedError("rog does not support metric: {}".format(metric))
    return rog
