import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import roc_curve
from sklearn import utils
from tqdm import tqdm
from .semaxis import calcSemAxis


def youden_separation_score(emb_a, emb_b):
    """
    Calculate the level of separation between two sets of vectors using the Youden index.
    
    Given points of two classes (a, b), i.e., emb_a, emb_b, we project points onto a line
    using the LDA; LDA does so by maximizing the separation between two classes on the line.
    Then, we measure the level of separation of the two classes on the line.
    We consider class "a" as a positive class and the other class "b" as a negative class.
    Then, for the points on the line, we compute the Youden index.
    By running a threshold from one end of the line to the other end, we compute the true positive rate and false positive rate.
    The Youden index is given by maximum absolute difference between the true positive rate - false positive rate.
   
    Parameters
    ----------
    emb_a : numpy.array, shape=(num_point, dim)
        Embedding vectors for class a.
    emb_b : numpy.array, shape=(num_point, dim)
        Embedding vectors for class b.
   
    Returns
    -------
    score : float
        Youden index
    """

    # Data label for LDA
    y = np.concatenate([np.ones(emb_a.shape[0]), np.zeros(emb_b.shape[0])])

    # Input data for LDA
    emb_ab = np.vstack([emb_a, emb_b])

    # Shuffle the data. This is to prevent overflow in sklearn's LDA which
    # sometimes happen when training and test data are the same
    emb_ab, y = utils.shuffle(emb_ab, y)

    # Embed poins onto a line
    x = calcSemAxis(emb_ab, emb_ab, labels=y, dim=1, mode="lda",)
    x = x.reshape(-1)

    # Calculate the ROC curve
    fpr, tpr, _ = roc_curve(y, x, 1)

    # Calculate the separation level
    score = np.max(np.abs(tpr - fpr))
    return score


def calc_pairwise_separation_score(
    emb,
    groups,
    separation_score_func=youden_separation_score,
    is_symmetric=True,
    n_jobs=20,
):
    """
    Calculate the separation score for each pair of groups of vectors.
   
    Parameters
    ----------
    emb : numpy.array, shape=(num_point, dim)
        embedding vector
    groups : numpy.array, shape=(num_point,)
        group membership for points. Can be integer or string
        calc_separation_score
    separation_score_func : function
        Function to compute the separation score. This function takes two arguments, emb_a and emb_b, and
        compute the separation level for the two vectors.
    is_symmetric : bool (Optional; Default True)
        Set True if the separation score is symmetric with respect to the groups, i.e., the score does not change
        when we swap the class labels.
    n_jobs : int (Optional; Default 20)
        Number of cores
       
    Returns
    -------
    separation_matrix : numpy.array, shape=(num_groups, num_groups)
        separation_matrix[i,j] indicates the sepration level of group_labels[i] and group_labels[j].
    group_labels : numpy.array
        List of groups.
    """
    group_labels = np.unique(groups)
    num_groups = len(group_labels)

    wrap_func = lambda _i, _j, emb_a, emb_b: (
        _i,
        _j,
        separation_score_func(emb_a, emb_b),
    )

    tasks = []
    for i, group_i in enumerate(group_labels):
        for j, group_j in enumerate(group_labels):
            if is_symmetric and j <= i:
                continue
            tasks += [
                delayed(wrap_func)(
                    i, j, emb[groups == group_i, :], emb[groups == group_j, :]
                )
            ]

    results = Parallel(n_jobs=n_jobs)(tasks)
    separation_matrix = np.zeros((num_groups, num_groups))
    for result in results:
        i, j, score = result
        separation_matrix[i, j] = score

    if is_symmetric:
        separation_matrix += separation_matrix.T
    return separation_matrix, group_labels
