import faiss
import numpy as np
from scipy import sparse, stats


def make_knn_graph(emb, k=5):
    """Construct the k-nearest neighbor graph from the embedding vectors.

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
    """Calculate the assortativity of `y` for close entities in the embedding
    space. A positive/negative assortativity indicates that the close entities
    tend to have a similar/dissimilar `y`. Zero assortativity means `y` is
    independent of the embedding.

    :param emb: embedding vectors
    :type emb: numpy.ndarray (num_entities, dim)
    :param y: feature of y
    :type y: numpy.ndarray (num_entities,)
    :param A: precomputed adjacency matrix of the graph. If None, a k-nearest neighbor graph will be constructed, defaults to None
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


def modularity(emb, group_ids, A=None, k=10):
    """Calculate the modularity of entities with group membership. The
    modularity ranges between [-1,1], where a positive modularity indicates
    that nodes with the same group membership tend to be close each other. Zero
    modularity means that membership `group_ids` is independent of the
    embedding.

    :param emb: embedding vectors
    :type emb: numpy.ndarray (num_entities, dim)
    :param group_ids: group membership for entities
    :type group_ids: numpy.ndarray (num_entities, )
    :param A: precomputed adjacency matrix of the graph. If None, a k-nearest neighbor graph will be constructed, defaults to None
    :type A: scipy.csr_matrix, optional
    :param k: Number of the nearest neighbors, defaults to 10
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
    labels, gids = np.unique(group_ids, return_inverse=True)
    U = sparse.csr_matrix(
        (np.ones_like(gids), (np.arange(gids.size), gids)),
        shape=(gids.size, len(labels)),
    )
    D = np.array(deg.reshape(1, -1) @ U).reshape(-1)
    Q = np.trace((U.T @ A @ U) - np.outer(D, D) / np.sum(D)) / np.sum(D)
    return Q


def nmi(emb, group_ids, A=None, k=10):
    """Calculate the Normalized Mutual Information for the entities with group
    membership. The NMI stands for the Normalized Mutual Information and takes
    a value between [0,1]. A larger NMI indicates that nodes with the same
    group membership tend to be close each other. Zero NMI means that
    membership `group_ids` is independent of the embedding.

    NMI is calculated as follows.
    1. Construct a k-nearest neighbor graph.
    2. Calculate the joint distribution of the group memberships of nodes connected by edges
    3. Calculate the normalized mutual information for the joint distribution.

    :param emb: embedding vectors
    :type emb: numpy.ndarray (num_entities, dim)
    :param group_ids: group membership for entities
    :type group_ids: numpy.ndarray (num_entities, )
    :param A: precomputed adjacency matrix of the graph. If None, a k-nearest neighbor graph will be constructed, defaults to None
    :type A: scipy.csr_matrix, optional
    :param k: Number of the nearest neighbors, defaults to 10
    :type k: int, optional
    :return: normalized mutual information
    :rtype: float

    .. highlight:: python
    .. code-block:: python

        >>> import emlens
        >>> import numpy as np
        >>> emb = np.random.randn(100, 20)
        >>> g = np.random.choice(10, 100)
        >>> rho = emlens.nmi(emb, g)
    """
    if A is None:
        A = make_knn_graph(emb, k=k)

    # Assign integers to group ids
    _, cids = np.unique(group_ids, return_inverse=True)

    # Get size
    K = max(cids) + 1
    N = cids.size

    # Calculate the joint distribution
    U = sparse.csr_matrix(
        (np.ones_like(cids), (np.arange(cids.size), cids)), shape=(N, K)
    )
    prc = np.array((U.T @ A @ U).toarray())
    prc = prc / np.sum(prc)
    pr = np.array(np.sum(prc, axis=1)).reshape(-1)
    pc = np.array(np.sum(prc, axis=0)).reshape(-1)

    # Calculate the mutual information
    Irc = stats.entropy(prc.reshape(-1), np.outer(pr, pc).reshape(-1))

    # Normalize MI
    Q = 2 * Irc / (stats.entropy(pr) + stats.entropy(pc))
    return Q


def element_sim(emb, group_ids, A=None, k=10):
    """Calculate the Element Centric Clustering Similarity for the entities
    with group membership.

    Gates, A. J., Wood, I. B., Hetrick, W. P., & Ahn, Y. Y. (2019).
    Element-centric clustering comparison unifies overlaps and hierarchy.
    Scientific Reports, 9(1), 1–13. https://doi.org/10.1038/s41598-019-44892-y

    This similarity takes a value between [0,1]. A larger value indicates that nodes with the same group
    membership tend to be close each other. Zero value means that membership `group_ids` is independent of
    the embedding.

    The Element Centric Clustering Similarity is calculated as follows.
    1. Construct a k-nearest neighbor graph.
    2. For each edge connecting nodes i and j (i<j), find the groups g_i and g_j to which the nodes belong.
    4. Make a list, L, of g_i's for nodes at the one end of the edges. Then, make another list, L', of nodes
       at the other end of the edges.
    5. Calculate the difference between L and L' using the Element Centric Clustering Similarity.

    :param emb: embedding vectors
    :type emb: numpy.ndarray (num_entities, dim)
    :param group_ids: group membership for entities
    :type group_ids: numpy.ndarray (num_entities, )
    :param A: precomputed adjacency matrix of the graph. If None, a k-nearest neighbor graph will be constructed, defaults to None
    :type A: scipy.csr_matrix, optional
    :param k: Number of the nearest neighbors, defaults to 10
    :type k: int, optional
    :return: element centric similarity
    :rtype: float

    .. highlight:: python
    .. code-block:: python

        >>> import emlens
        >>> import numpy as np
        >>> emb = np.random.randn(100, 20)
        >>> g = np.random.choice(10, 100)
        >>> rho = emlens.element_sim(emb, g)
    """
    if A is None:
        A = make_knn_graph(emb, k=k)

    # Assign integers to group ids
    _, cids = np.unique(group_ids, return_inverse=True)

    # Get size
    K = max(cids) + 1
    M = len(A.data)

    # Make a list of group memebrships
    r, c, _ = sparse.find(A)
    gA, gB = cids[r], cids[c]

    # Calculate the element centric similarity
    # A naive calculation is to compute individual p_ij and then sum them up to calculate S_i.
    # However, this requires O(N^2) memory and computation time. To compute it efficiently, we
    # rewrite \sum_{j} |p^A _{ij} - p^B _{ij}| in Eq. 5 of the original paper as
    #   \sum_{j} |p^A _{ij} - p^B _{ij}| + \sum_{j} |p^A _{ij}| + \sum_{j} |p^B _{ij}|
    #       + \sum_{g^A _i = g^A _j and g^B _i = g^B _j} ( |p^A _{ij} - p^B _{ij}| - |p^A _{ij}| - |p^B _{ij}|)
    #   = 2 - \sum_{g^A _i = g^A _j and g^B _i = g^B _j} |p^A _{ij}| + |p^B _{ij}| -  |p^A _{ij} - p^B _{ij}|.
    # where g^A _i is the membership of i in partition A.
    # Denote by n^A _r the number of elements that belong to group r in partition A (and we analagously define n^B _c).
    # In ddition, denote by n_{rc} the number of elements that belong to group r in partition A and group c in partition B.
    # By substituting p_ij = alpha / n_A + (1-alpha) * (i == j), we have
    #   S_i = 0.5 * n_{rc} * ( 1/n^A _{g^A _i} +  1/n^B _{g^B _i} - |1/n^A _{g^A _i} - 1/n^B _{g^B _i}|).
    # Computing this for N nodes requires memory and computation time in order O(NK), where K is the number of groups.
    # This order can be substantially lower than O(N^2) if K<<N.
    UA = sparse.csr_matrix((np.ones_like(gA), (np.arange(gA.size), gA)), shape=(M, K))
    UB = sparse.csr_matrix((np.ones_like(gB), (np.arange(gB.size), gB)), shape=(M, K))

    fA = np.array(UA.sum(axis=0)).reshape(-1)
    fB = np.array(UB.sum(axis=0)).reshape(-1)
    fAB = (UA.T @ UB).toarray()

    Si = (
        0.5
        * fAB[(gA, gB)]
        * (1.0 / fA[gA] + 1.0 / fB[gB] - np.abs(1.0 / fA[gA] - 1.0 / fB[gB]))
    )
    S = np.mean(Si)
    return S


def pairwise_dot_sim(emb, group_ids):
    """Pairwise distance between groups. The dot similarity between two groups
    i and j is calculated by averaging the dot similarity of entities in group
    i and those in group j.

    :param emb: embedding vectors
    :type emb: numpy.ndarray (num_entities, dim)
    :param group_ids: group membership
    :type group_ids: numpy.ndarray (num_entities)
    :return: S, groups
    :rtype: numpy.ndarray, numpy.ndarray

        * **S** (numpy.ndarray (num_groups, num_groups)): Similarity matrix for groups.

        * **groups** (numpy.ndarray (num_groups])):  group[i] is the group for the ith row/column of S.

    .. highlight:: python
    .. code-block:: python

        >>> import emlens
        >>> import numpy as np
        >>> import seaborn as sns
        >>> emb = np.random.randn(100, 20)
        >>> group_ids = np.random.choice(10, 100)
        >>> S, groups = emlens.pairwise_dot_sim(emb, group_ids)
        >>> sns.heatmap(pd.DataFrame(S, index = groups, columns = groups))
    """
    groups, gids = np.unique(group_ids, return_inverse=True)

    U = sparse.csr_matrix(
        (np.ones_like(gids), (np.arange(gids.size), gids)),
        shape=(emb.shape[0], len(groups)),
    )
    U = U @ sparse.diags(1 / np.maximum(1, np.array(U.sum(axis=0)).reshape(-1)))

    Uemb = U.T @ emb
    S = Uemb @ Uemb.T
    return S, groups


def pairwise_distance(emb, group_ids):
    """Pairwise distance between the centroid of groups. The centroid of a
    group is the average embedding vectors of the entities in the group.

    :param emb: embedding
    :type emb: numpy.ndarray (num_entities, dim)
    :param group_ids: group membership
    :type group_ids: numpy.ndarray (num_entities)
    :return: D, groups
    :rtype: numpy.ndarray, numpy.ndarray

        * **D** (numpy.ndarray (num_groups, num_groups)): Distance matrix for groups.

        * **groups** (numpy.ndarray (num_groups])): group[i] is the group for the ith row/column of D.

    .. highlight:: python
    .. code-block:: python

        >>> import emlens
        >>> import numpy as np
        >>> import seaborn as sns
        >>> emb = np.random.randn(100, 20)
        >>> group_ids = np.random.choice(10, 100)
        >>> D, groups = emlens.pairwise_distance(emb, group_ids)
        >>> sns.heatmap(pd.DataFrame(D, index = groups, columns = groups))
    """
    groups, gids = np.unique(group_ids, return_inverse=True)
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
