import numbers
from functools import partial

import faiss
import numpy as np
from scipy import linalg, sparse, stats
from sklearn import metrics as skmetrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def make_knn_graph(
    emb, k=5, binarize=True, metric="euclidean", mutual=True, gpu_id=None
):
    """Construct the k-nearest neighbor graph from the embedding vectors.

    :param emb: embedding vectors
    :type emb: numpy.ndarray (num_entities, dim)
    :param k: Number of nearest neighbors. If list or array is given, then construct a k-nearest neighbor graph for each k, defaults to 5
    :type k: int or iterable, optional
    :param binarize: `binarize=False` will set the weight of the between nodes i and j  by exp(-d_{ij]}). `binarize=True` will set to one., defaults to True
    :paramm metric: Distance metric for finding nearest neighbors. Available metric `metric="euclidean"`, `metric="cosine"` , `metric="dotsim"`
    :type metric: str
    :return: The adjacency matrix of the k-nearest neighbor graph
    :rtype: sparse.csr_matrix

    .. highlight:: python
    .. code-block:: python

        >>> import emlens
        >>> emb = np.random.randn(100, 20)
        >>> A = emlens.make_knn_graph(emb, k = 10)
    """
    if isinstance(k, numbers.Number):
        k = [k]

    kmax = int(np.max(k))
    num_points = emb.shape[0]

    if kmax >= num_points:
        raise ValueError("Number of neighbors is larger than the number of data points")

    nodes, neighbors, distances = find_nearest_neighbors(
        emb, emb, k=kmax, gpu_id=gpu_id
    )

    retval = []
    for _k in k:
        c = neighbors[:, :_k].copy().reshape(-1)
        r = nodes[:, :_k].copy().reshape(-1)
        sim = distances[:, :_k].copy().reshape(-1)

        if mutual:
            r, c, sim = find_mutual_edges(r, c, sim)

        A = sparse.csr_matrix((sim, (r, c)), shape=(num_points, num_points))

        if binarize:
            A.data = np.ones_like(A.data)
        retval += [{"A": A, "k": _k}]
    return retval


def find_mutual_edges(r, c, v=None):
    N = int(np.maximum(np.max(r), np.max(c)) + 1)
    A = sparse.csr_matrix((np.ones_like(r), (r, c)), shape=(N, N))
    A = A + A.T
    A.data[A.data < 2] = 0
    A.eliminate_zeros()
    if v is None:
        r, c, _ = sparse.find(sparse.triu(A, 1))
        return r, c
    elif v is not None:
        br, bc, _ = sparse.find(sparse.triu(A, 1))
        W = sparse.csr_matrix((v, (r, c)), shape=(N, N))
        W = (W + W.T) / 2
        bv = np.array(W[(br, bc)]).reshape(-1)
        return br, bc, bv


def find_nearest_neighbors(
    target, emb, k=5, metric="euclidean", gpu_id=None, exact=True
):
    """Find the nearest neighbors for each point.

    :param emb: vectors for the points for which we find the nearest neighbors
    :type emb: numpy.ndarray (num_entities, dim)
    :param emb: vectors for the points from which we find the nearest neighbors.
    :type emb: numpy.ndarray (num_entities, dim)
    :param k: Number of nearest neighbors, defaults to 5
    :type k: int, optional
    :paramm metric: Distance metric for finding nearest neighbors. Available metric `metric="euclidean"`, `metric="cosine"` , `metric="dotsim"`
    :type metric: str
    :return: IDs of emb (indices), and similarity (distances)
    :rtype: indices (numpy.ndarray), distances (numpy.ndarray)

    .. highlight:: python
    .. code-block:: python

        >>> import emlens
        >>> import numpy as np
        >>> emb = np.random.randn(100, 20)
        >>> target = np.random.randn(10, 20)
        >>> A = emlens.find_nearest_neighbors(target, emb, k = 10)
    """
    if emb.flags["C_CONTIGUOUS"]:
        emb = emb.copy(order="C")
    if target.flags["C_CONTIGUOUS"]:
        target = target.copy(order="C")
    emb = emb.astype(np.float32)
    target = target.astype(np.float32)
    # Find the nearest neighbors
    if metric == "euclidean":
        if exact:
            index = faiss.IndexFlatL2(emb.shape[1])
        else:
            quantiser = faiss.IndexFlatL2(emb.shape[1])
            nlist = int(np.ceil(10 * np.sqrt(emb.shape[0])))
            index = faiss.IndexIVFFlat(quantiser, emb.shape[1], nlist, faiss.METRIC_L2)
            index.train(emb)
    elif metric == "cosine":
        denom = np.array(np.linalg.norm(emb, axis=1)).reshape(-1)
        denom[np.isclose(denom, 0)] = 1
        emb = np.einsum("i,ij->ij", 1 / denom, emb)

        denom = np.array(np.linalg.norm(target, axis=1)).reshape(-1)
        denom[np.isclose(denom, 0)] = 1
        target = np.einsum("i,ij->ij", 1 / denom, target)

        if exact:
            index = faiss.IndexFlatIP(emb.shape[1])
        else:
            quantiser = faiss.IndexFlatIP(emb.shape[1])
            nlist = int(np.ceil(10 * np.sqrt(emb.shape[0])))
            index = faiss.IndexIVFFlat(
                quantiser, emb.shape[1], nlist, faiss.METRIC_INNER_PRODUCT
            )
            index.train(emb)
    elif metric == "dotsim":
        if exact:
            index = faiss.IndexFlatIP(emb.shape[1])
        else:
            quantiser = faiss.IndexFlatIP(emb.shape[1])
            nlist = int(np.ceil(10 * np.sqrt(emb.shape[0])))
            index = faiss.IndexIVFFlat(
                quantiser, emb.shape[1], nlist, faiss.METRIC_INNER_PRODUCT
            )
            index.train(emb)
    else:
        raise NotImplementedError("does not support metric: {}".format(metric))

    if gpu_id is None:
        gpu_id = 0

    if k >= 2048:  # if k is larger than that supported by GPU
        index.add(emb)
    else:
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, gpu_id, index)
            index.add(emb)
        except (RuntimeError, AttributeError):
            index.add(emb)
    distances, neighbors = index.search(target, k=k)

    assert distances.dtype == "float32"
    assert neighbors.dtype == "int64"

    nodes = (np.arange(target.shape[0]).reshape((-1, 1)) @ np.ones((1, k))).astype(int)
    neighbors = neighbors.astype(int)
    return nodes, neighbors, distances


def assortativity(emb, y, k=5, metric="euclidean", gpu_id=None):
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
    :param k: Number of the nearest neighbors. If a list is given, the assortativity for each k in the list will be calculated (in the same order of the list), defaults to 5
    :type k: int or list, optional
    :paramm metric: Distance metric for finding nearest neighbors. Available metric `metric="euclidean"`, `metric="cosine"` , `metric="dotsim"`
    :type metric: str
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

    if isinstance(k, numbers.Number):
        k = [k]

    scores = []
    for _k in k:
        nodes, neighbors, _ = find_nearest_neighbors(
            emb, emb, k=_k, metric=metric, gpu_id=gpu_id
        )
        score = stats.pearsonr(
            y[nodes[:, :_k]].reshape(-1), y[neighbors[:, :_k]].reshape(-1)
        )[0]
        scores.append(score)

    if len(scores) == 1:
        return scores[0]

    return scores


def modularity(emb, group_ids, k=10, metric="euclidean", gpu_id=None):
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
    :paramm metric: Distance metric for finding nearest neighbors. Available metric `metric="euclidean"`, `metric="cosine"` , `metric="dotsim"`
    :type metric: str
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
    if isinstance(k, numbers.Number):
        k = [k]

    labels, gids = np.unique(group_ids, return_inverse=True)
    U = sparse.csr_matrix(
        (np.ones_like(gids), (np.arange(gids.size), gids)),
        shape=(gids.size, len(labels)),
    )

    Alist = make_knn_graph(
        emb, k=k, binarize=True, metric=metric, mutual=True, gpu_id=gpu_id
    )
    scores = []
    for row in Alist:
        A = row["A"]
        deg = np.array(A.sum(axis=0))
        D = np.array(deg.reshape(1, -1) @ U).reshape(-1)
        score = np.trace((U.T @ A @ U) - np.outer(D, D) / np.sum(D)) / np.sum(D)
        scores.append(score)

    if len(scores) == 1:
        return scores[0]

    return scores


def nmi(emb, group_ids, A=None, k=10, metric="euclidean", gpu_id=None):
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
    :paramm metric: Distance metric for finding nearest neighbors. Available metric `metric="euclidean"`, `metric="cosine"` , `metric="dotsim"`
    :type metric: str
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
    if isinstance(k, numbers.Number):
        k = [k]

    # Assign integers to group ids
    _, cids = np.unique(group_ids, return_inverse=True)

    # Get size
    K = max(cids) + 1
    N = cids.size

    # Calculate the joint distribution
    U = sparse.csr_matrix(
        (np.ones_like(cids), (np.arange(cids.size), cids)), shape=(N, K)
    )

    Alist = make_knn_graph(
        emb, k=k, binarize=True, metric=metric, mutual=True, gpu_id=gpu_id
    )
    scores = []
    for row in Alist:
        A = row["A"]
        prc = np.array((U.T @ A @ U).toarray())
        prc = prc / np.sum(prc)
        pr = np.array(np.sum(prc, axis=1)).reshape(-1)
        pc = np.array(np.sum(prc, axis=0)).reshape(-1)

        # Calculate the mutual information
        Irc = stats.entropy(prc.reshape(-1), np.outer(pr, pc).reshape(-1))

        # Normalize MI
        score = 2 * Irc / (stats.entropy(pr) + stats.entropy(pc))
        scores.append(score)

    if len(scores) == 1:
        return scores[0]
    return scores


def element_sim(emb, group_ids, A=None, k=10, metric="euclidean", gpu_id=None):
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
    if isinstance(k, numbers.Number):
        k = [k]

    # Assign integers to group ids
    _, cids = np.unique(group_ids, return_inverse=True)

    # Get size
    K = max(cids) + 1

    # Calculate the joint distribution
    Alist = make_knn_graph(
        emb, k=k, binarize=True, metric=metric, mutual=True, gpu_id=gpu_id
    )
    scores = []
    for row in Alist:
        A = row["A"]

        # Make a list of group memebrships
        r, c, _ = sparse.find(A)
        M = len(A.data)
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
        # Computing this for N nodes requires memory and computation time in order of O(NK), where K is the number of groups.
        # This order can be substantially lower than O(N^2) if K<<N.
        UA = sparse.csr_matrix(
            (np.ones_like(gA), (np.arange(gA.size), gA)), shape=(M, K)
        )
        UB = sparse.csr_matrix(
            (np.ones_like(gB), (np.arange(gB.size), gB)), shape=(M, K)
        )

        fA = np.array(UA.sum(axis=0)).reshape(-1)
        fB = np.array(UB.sum(axis=0)).reshape(-1)
        fAB = (UA.T @ UB).toarray()

        Si = (
            0.5
            * fAB[(gA, gB)]
            * (1.0 / fA[gA] + 1.0 / fB[gB] - np.abs(1.0 / fA[gA] - 1.0 / fB[gB]))
        )
        score = np.mean(Si)
        scores.append(score)
    if len(scores) == 1:
        return scores[0]
    return score


def f1_score(emb, target, agg="mode", **params):
    """Measuring the prediction performance based on the K-Nearest Neighbor
    Graph. Equivalent to knn_pred_score(emb, target, target_type = "disc").

    This function measures how well the embedding space can predict the metadata of entities using the k-nearest neighbor algorithm.
    To this end, the following K-folds cross-validation is performed:
    0. Split all entities into K groups.
    1. Take one group as a test set and the other groups as a training set
    2. Using the training set, predict the `target` variable for the entities in the training set. The prediction is made by the most frequent target variables of the nearest neighbors.
    3. Calculate the prediction accuracy by the f1-score
    4. Repeat Steps 1-3 such that each group is used as the test set once.
    5. Compute the average of the prediction accuracy computed in Step 3.

    :param emb: embedding vectors
    :type emb: numpy.ndarray (num_entities, dim)
    :param target: target variable to predict
    :type target: numpy.ndarray (num_target,)
    :paramm metric: Distance metric for finding nearest neighbors. Available metric `metric="euclidean"`, `metric="cosine"` , `metric="dotsim"`
    :type metric: str
    :paramm agg: How to aggregate the neighbors' variables. Setting `aggregation='mode'` uses the most frequent label, `='mean'` uses the mean as the predicted variable.
    :type agg: str
    :param k: Number of nearest neighbors, defaults to 10
    :type k: int or list, optional
    :param n_splits: Number of folds for the cross-validation, defaults to 10
    :type n_splits: int, optional
    :param iteration: Number of rounds of the cross validation. If iteration>1, the average of the cross validation score will be returned. If `return_score_all=True`, all scores will be returned, defaults to 1.
    :type iteration: int
    :param return_all_scores: Set `True` to return all scores for the cross-vaidations. If set `False`, the mean of the score is returned
    :type return_all_scores: bool
    :param gpu_id: ID of the GPU device.
    :type gpu_id: string or int
    :param knn_exact: Set `True` to use the exact nearest neighbors for prediction. If set `False`, hueristics are used to find "probably" the nearest neighbors for the sake of substantial computation speed up.
    :type knn_exact: string or int
    :return: dict object {"k", "score"}, where k is the number of neighbors, and score is the prediction score.
    :rtype: dict
    """
    scoring_func = partial(skmetrics.f1_score, average="micro")
    _, _target = np.unique(target, return_inverse=True)

    return knn_pred_score(emb, _target, scoring_func=scoring_func, agg=agg, **params)


def r2_score(emb, target, model="linear", test=True, **params):
    """Measuring the prediction performance based on the K-Nearest Neighbor
    Graph or Linear Regression.

    If model == "knn", this is quivalent to knn_pred_score(emb, target, target_type = "cont").

    This function measures how well the embedding space can predict the metadata of entities using the k-nearest neighbor algorithm.
    To this end, the following K-folds cross-validation is performed:
    0. Split all entities into K groups.
    1. Take one group as a test set and the other groups as a training set
    2. Using the training set, predict the `target` variable for the entities in the training set. The prediction is made by the average target variables of the nearest neighbors.
    3. Calculate the prediction accuracy by the R^2 score
    4. Repeat Steps 1-3 such that each group is used as the test set once.
    5. Compute the average of the prediction accuracy computed in Step 3.

    :param emb: embedding vectors
    :type emb: numpy.ndarray (num_entities, dim)
    :param target: target variable to predict
    :type target: numpy.ndarray (num_target,)
    :param model: model to predict node attributes. With model="linear", the prediction is based on a linear regression model that predicts targets from the given embedding. With model="knn"i, the prediction is based on k-nearest neighbor graphs.
    :type model:str
    :return: performance score
    :rtype: float
    """
    if model == "knn":
        scoring_func = skmetrics.r2_score
        return knn_pred_score(
            emb, target, scoring_func=scoring_func, agg="mean", **params
        )
    elif model == "linear":
        return linear_pred_score(emb, target, **params)


def linear_pred_score(
    emb, target, n_splits=10, iteration=1, return_all_scores=False,
):
    """Measuring the prediction performance based on a linear regression model.

    This function measures how well the embedding space can predict the metadata of entities using the linear regression model.
    To this end, the following K-folds cross-validation is performed:
    0. Split all entities into K groups.
    1. Take one group as a test set and the other groups as a training set
    2. Using the training set, predict the `target` variable for the entities in the training set using a linear regression model.
    3. Calculate the prediction accuracy
    4. Repeat Steps 1-3 such that each group is used as the test set once.
    5. Compute the average of the prediction accuracy computed in Step 3.

    The performance score is measured based on the R^2 score.

    :param emb: embedding vectors
    :type emb: numpy.ndarray (num_entities, dim)
    :param target: target variable to predict
    :type target: numpy.ndarray (num_target,)
    :param n_splits: Number of folds, defaults to 10
    :type n_splits: int, optional
    :param iteration: Number of rounds of the cross validation. If iteration>1, the average of the cross validation score will be returned., defaults to 1.
    :type iteration: int
    :param return_all_scores: "return_all_scores=True" or "=False" to return all scores in the cross validations or not, respectively.
    :type  return_all_scores: bool
    :return: performance score
    :rtype: float
    """
    scores = []
    all_score = []
    for _i in range(iteration):
        kf = KFold(n_splits=n_splits, shuffle=True)
        _scores = []
        for train_index, test_index in kf.split(target):
            x_train = emb[train_index, :]
            x_test = emb[test_index, :]
            y_train = target[train_index]
            y_test = target[test_index]

            # Train
            reg = LinearRegression().fit(x_train, y_train)
            _score = reg.score(x_test, y_test)

            if np.isnan(_score):
                continue
            _scores += [_score]
            all_score += [_score]

        scores += [np.mean(_scores)]
    score = np.mean(scores)
    if return_all_scores:
        return all_score
    else:
        return score


def knn_pred_score(
    emb,
    target,
    scoring_func,
    metric="euclidean",
    agg="mode",
    k=10,
    n_splits=10,
    iteration=1,
    return_all_scores=False,
    gpu_id=None,
    knn_exact=True,
):
    """Prediction based on k-Nearest neighbor graph.

    :param emb: embedding vectors
    :type emb: numpy.ndarray (num_entities, dim)
    :param target: target variable to predict
    :type target: numpy.ndarray (num_target,)
    :param scoring_func: scoring function. This function will take a target variable `y` as the first argumebt and predicted variable `ypred` as the second argumebt, and ouputs the prediction score `score`, i.e., score=scoring_func(y, ypred).
    :type scoring_func: numpy func
    :paramm metric: Distance metric for finding nearest neighbors. Available metric `metric="euclidean"`, `metric="cosine"` , `metric="dotsim"`
    :type metric: str
    :paramm agg: How to aggregate the neighbors' variables. Setting `aggregation='mode'` uses the most frequent label, `='mean'` uses the mean as the predicted variable.
                 If there are more than k neighbors, aggregate the k neighbors connected by the edges with the largest weights, defaults to 'mode'
    :type agg: str
    :param k: Number of nearest neighbors, defaults to 10
    :type k: int or list, optional
    :param n_splits: Number of folds, defaults to 10
    :type n_splits: int, optional
    :param iteration: Number of rounds of the cross validation. If iteration>1, the average of the cross validation score will be returned., defaults to 1.
    :type iteration: int
    :param return_all_scores: Set `True` to return all scores for the cross-vaidations. If set `False`, the mean of the score is returned
    :type return_all_scores: bool
    :param gpu_id: ID of the GPU device.
    :type gpu_id: string or int
    :param knn_exact: Set `True` to use the exact nearest neighbors for prediction. If set `False`, hueristics are used to find "probably" the nearest neighbors for the sake of substantial computation speed up.
    :type knn_exact: string or int
    :return: dict object {"k", "score"}, where k is the number of neighbors, and score is the prediction score.
    :rtype: dict
    """

    scores = []
    pbar = tqdm(total=iteration * n_splits)
    for _ in range(iteration):
        kf = KFold(n_splits=n_splits, shuffle=True)
        for _, (train_index, test_index) in enumerate(kf.split(target)):
            train_emb = emb[train_index, :]
            test_emb = emb[test_index, :]
            train_labels = target[train_index]
            test_labels = target[test_index]

            pred_k = _make_knn_pred(
                test_emb,
                train_emb,
                train_labels=train_labels,
                klist=k,
                agg=agg,
                metric=metric,
                gpu_id=gpu_id,
                knn_exact=knn_exact,
            )

            for _k, pred in pred_k.items():
                score = scoring_func(test_labels, pred)
                scores.append({"score": score, "k": _k})
            pbar.update(1)
    return scores


def _make_knn_pred(
    target,
    emb,
    train_labels,
    klist,
    agg="mode",
    metric="euclidean",
    gpu_id=None,
    knn_exact=True,
):
    """Inner function for make_knn_pred_score."""

    if isinstance(klist, numbers.Number):
        klist = [klist]

    # Train
    kmax = int(np.max(klist))
    _, indices, distances = find_nearest_neighbors(
        target, emb, k=kmax, metric=metric, gpu_id=gpu_id, exact=knn_exact
    )

    # Agrgegation
    pred_k = {}
    if agg == "mode":
        train_label_names, train_label_ids = np.unique(
            train_labels, return_inverse=True
        )
        X = train_label_ids[indices]
        for k in klist:
            y_pred = stats.mode(X[:, :k], axis=1)[0].reshape(-1)
            pred_k[k] = train_label_names[y_pred]
    else:
        X = train_labels[indices]
        for k in klist:
            y_pred = np.array(X[:, :k].mean(axis=1)).reshape(-1)
            pred_k[k] = y_pred
    return pred_k


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


def rog(emb, metric="euc", center=None):
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
        >>> rog = emlens.rog(emb, 'cos')
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


def effective_dimension(emb, q=1, normalize=False, is_cov=False):
    """Effective dimensionality of a set of points in space.

    Effection dimensionality is the number of orthogonal dimensions needed to capture the overall correlational structure of data.
    See Del Giudice, M. (2020). Effective Dimensionality: A Tutorial. _Multivariate Behavioral Research, 0(0), 1–16. https://doi.org/10.1080/00273171.2020.1743631.

    :param emb: embedding vectors
    :type emb: numpy.ndarray (num_entities, dim)
    :param q: Parameter for the Renyi entropy function, defaults to 1
    :type q: int, optional
    :param normalize: Set True to center data. For spherical or quasi-spherical data (such as the embedding by word2vec), normalize=False is recommended, defaults to False
    :type normalize: bool, optional
    :param is_cov: Set True if `emb` is the covariance matrix, defaults to False
    :type is_cov: bool, optional
    :return: effective dimensionality
    :rtype: float


    .. highlight:: python
    .. code-block:: python

        >>> import emlens
        >>> import numpy as np
        >>> emb = np.random.randn(100, 20)
        >>> ed = emlens.effective_dimension(emb)
    """
    if is_cov:
        Cov = emb
    else:
        if normalize:
            emb = StandardScaler().fit_transform(emb)
        Cov = (emb.T @ emb) / emb.shape[0]
    lam = linalg.eigvalsh(Cov)
    lam = np.real(lam)
    lam = np.maximum(lam, 1e-10)
    p = lam / np.sum(lam)
    p = p[p > 0]
    if q == 1:
        return np.exp(stats.entropy(p))
    elif np.isinf(q):
        return -np.log(np.max(p))
    else:
        return np.log(np.sum(np.power(p, q))) / (1 - q)


def effective_dimension_vector(emb, normalize=False, is_cov=False):
    """Effective dimensionality of a set of points in space.

    Effection dimensionality is the number of orthogonal dimensions needed to capture the overall correlational structure of data.
    See Del Giudice, M. (2020). Effective Dimensionality: A Tutorial. _Multivariate Behavioral Research, 0(0), 1–16. https://doi.org/10.1080/00273171.2020.1743631.

    :param emb: embedding vectors
    :type emb: numpy.ndarray (num_entities, dim)
    :param q: Parameter for the Renyi entropy function, defaults to 1
    :type q: int, optional
    :param normalize: Set True to center data. For spherical or quasi-spherical data (such as the embedding by word2vec), normalize=False is recommended, defaults to False
    :type normalize: bool, optional
    :param is_cov: Set True if `emb` is the covariance matrix, defaults to False
    :type is_cov: bool, optional
    :return: effective dimensionality
    :rtype: float


    .. highlight:: python
    .. code-block:: python

        >>> import emlens
        >>> import numpy as np
        >>> emb = np.random.randn(100, 20)
        >>> ed = emlens.effective_dimension(emb)
    """
    if is_cov:
        Cov = emb
    else:
        if normalize:
            emb = StandardScaler().fit_transform(emb)
        Cov = (emb.T @ emb) / emb.shape[0]
    lam, v = linalg.eig(Cov)
    order = np.argsort(lam)[::-1]
    lam = lam[order]
    v = v[:, order]

    lam = np.real(lam)
    lam = np.maximum(lam, 1e-10)
    p = lam / np.sum(lam)
    p = p[p > 0]
    return v, p
