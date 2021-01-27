import json
import os
import shutil

import numpy as np
import scipy
from scipy import sparse
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def SemAxis(
    vec,
    class_vec,
    labels,
    label_order=None,
    dim=1,
    mode="fda",
    centering=True,
    return_class_vec=False,
    **params
):
    """SemAxis.

    :param vec: embedding vectors
    :type vec: numpy.ndarray (num_entities, dim)
    :param class_vec: labeled vectors
    :type class_vec: numpy.ndarray (num_labeled_entities, dim)
    :param labels: labels
    :type labels: numpy.ndarray(num_labeled_entities, dim)
    :param label_order: label order, defaults to None
    :type label_order: list, optional
    :param dim: reduced dimension, defaults to 1
    :type dim: int, optional
    :param mode: projection method. mode="semaxis", "lda", "fda" are supported, defaults to "fda"
    :type mode: str, optional
    :param centering: center the projected space, defaults to True
    :type centering: bool, optional
    :param return_class_vec: whether to return the projected labeled vector, defaults to False
    :type return_class_vec: bool, optional
    :return: tuple containing

        retvec : np.array, shape=(num_data, dim)

            The vectors for the data points projected onto semaxis (or semspace).

        class_vec : np.array, shape=(num_data_with_labels, dim) (Optional)

            The projection of vectors used to construct the axis (or space).

            This variable is returned only when return_class_vec=True.

        labels : np.array, shape=(num_data_with_labels,) (Optional)

            The class labels for vectors used to construct the axis (or space).

            This variable is returned only when return_class_vec=True.

    :rtype: tuple

    .. highlight:: python
    .. code-block:: python

        >>> import emlens
        >>> import numpy as np
        >>> emb = np.random.randn(100, 20)
        >>> labels = np.random.choice(100, 10)
        >>> xy = emlens.SemAxis(vec = emb, class_vec = emb, labels = labels, dim=2, mode = "fda")
    """

    def calc_mean_vec(vec, s=None):
        if s is None:
            return np.mean(vec, axis=0) if vec.shape[1] > 1 else np.mean(vec)
        else:
            return np.mean(vec[s, :], axis=0) if vec.shape[1] > 1 else np.mean(vec[s])

    if label_order is None:
        class_labels, class_ids = np.unique(labels, return_inverse=True)
        n_class = len(class_labels)
    else:
        label2cids = {ll: i for i, ll in enumerate(label_order)}
        class_ids = np.array([label2cids[ll] for ll in labels])
        n_class = len(label2cids)

    if mode == "semaxis":
        left_center = np.mean(class_vec[class_ids == 0, :], axis=0)
        right_center = np.mean(class_vec[class_ids == 1, :], axis=0)
        vr = right_center - left_center
        denom = np.linalg.norm(vec, axis=1) * np.linalg.norm(vr)
        denom = 1 / np.maximum(denom, 1e-20)
        ret_vec = sparse.diags(denom) @ (vec @ vr.T)

        denom = np.linalg.norm(class_vec, axis=1) * np.linalg.norm(vr)
        denom = 1 / np.maximum(denom, 1e-20)
        prj_class_vec = sparse.diags(denom) @ (class_vec @ vr.T)
    elif mode == "lda":
        lda = LinearDiscriminantAnalysis(n_components=dim, **params)
        lda.fit(class_vec, class_ids)
        ret_vec = lda.transform(vec)
        prj_class_vec = lda.transform(class_vec)
    elif mode == "fda":
        ret_vec, prj_class_vec = fisher_linear_discriminant(
            vec, class_vec, class_ids, dim, return_class_vec=True, **params
        )

    if centering:
        class_centers = np.zeros((n_class, dim))
        for cid in range(n_class):
            class_centers[cid, :] = calc_mean_vec(prj_class_vec, class_ids == cid)
        ret_vec -= calc_mean_vec(class_centers)
        prj_class_vec -= calc_mean_vec(class_centers)

    if return_class_vec:
        return ret_vec, prj_class_vec, labels
    else:
        return ret_vec


def fisher_linear_discriminant(
    vec,
    class_vec,
    class_labels,
    dim,
    priors="data",
    shrinkage=0,
    return_class_vec=False,
):
    """Fisher's linear discriminant analysis.

    :param vec: embedding vectors
    :type vec: numpy.ndarray (num_entities, dim)
    :param class_vec: labeled vectors
    :type class_vec: numpy.ndarray (num_labeled_entities, dim)
    :param class_labels: labels
    :type class_labels: numpy.ndarray(num_labeled_entities, dim)
    :param dim: dimension
    :type dim: int
    :param priors: prior distribution for data, defaults to "data"
    :type priors: str, optional
    :param shrinkage: shrinkage strength, defaults to 0
    :type shrinkage: int, optional
    :param return_class_vec: whether to return the projected labeled entities, defaults to False
    :type return_class_vec: bool, optional
    :return: tuple containing
    :rtype: tuple

        retvec : np.array, shape=(num_data, dim)

            The vectors for the data points projected onto semaxis (or semspace).

        class_vec : np.array, shape=(num_data_with_labels, dim) (Optional)

            The projection of vectors used to construct the axis (or space).
            This variable is returned only when return_class_vec=True.

        labels : np.array, shape=(num_data_with_labels,) (Optional)

            The class labels for vectors used to construct the axis (or space).
            This variable is returned only when return_class_vec=True.
    """

    labels, group_ids = np.unique(class_labels, return_inverse=True)

    K = len(labels)
    DIM = vec.shape[1]

    class_weight = np.zeros(K)
    if isinstance(priors, dict):
        for k, lab in enumerate(labels):
            class_weight[k] = priors[lab]
    elif priors == "data":
        for k, _ in enumerate(labels):
            class_weight[k] = sum(group_ids == k)
    elif priors == "uniform":
        for k, _ in enumerate(labels):
            class_weight[k] = 1

    class_weight = class_weight / np.sum(class_weight)

    Cw = np.zeros((DIM, DIM))  # Within-class variance
    MU = np.zeros((K, DIM))  # Mean for each class
    for k in range(K):
        v_k = class_vec[group_ids == k, :]
        mu_k = np.mean(v_k, axis=0)
        MU[k, :] = mu_k

        if v_k.shape[0] == 1:
            continue

        Cw += class_weight[k] * np.cov(v_k.T)

    # Shrinkage
    if (shrinkage < 1) and (shrinkage > 0):
        Cw = Cw + shrinkage / (1 - shrinkage) * np.eye(DIM)
    elif shrinkage == 1:
        Cw = np.eye(DIM)

    # between class
    if K == 2:
        u = np.linalg.inv(Cw) @ (MU[1, :] - MU[0, :]).reshape((DIM, 1))
        u = u / np.linalg.norm(u)
    else:
        Cb = np.zeros((DIM, DIM))  # Between-class
        mu = np.mean(MU, axis=0)
        for k in range(K):
            Cb += class_weight[k] * np.outer(MU[k, :] - mu, MU[k, :] - mu)

        if (shrinkage < 1) and (shrinkage > 0):
            Cw = Cw + shrinkage / (1 - shrinkage) * np.eye(DIM)
        elif shrinkage == 1:
            Cw = np.eye(DIM)

        # Solve generalized eigenvalue problem
        s, u, _ = scipy.linalg.eig(Cb, Cw, left=True)
        ids = np.argsort(-np.array(s).reshape(-1))[:dim]
        u = u[:, ids]  # @ np.diag(np.sqrt(np.real(s[ids])))

    # Projection
    prj_vec = vec @ u
    if return_class_vec:
        prj_class_vec = class_vec @ u
        return prj_vec, prj_class_vec
    else:
        return prj_vec


def saveSemAxis(filename, class_vec, labels, **kwargs):
    """Save SemAxis into a file.

    :param filename: name of file
    :type filename: str
    :param class_vec: embedding vectors for labeled entities
    :type class_vec: numpy.ndarray
    :param labels: labels
    :type labels: numpy.ndarray

    .. highlight:: python
    .. code-block:: python

        >>> import emlens
        >>> import numpy as np
        >>> emb = np.random.randn(100, 20)
        >>> labels = np.random.choice(100, 10)
        >>> emlens.saveSemAxis('semspace', class_vec = emb, labels = labels, dim=2, mode = "fda")
    """
    if os.path.exists(filename):
        shutil.rmtree(filename)
    os.mkdir(filename)
    emb_filename = "{dir_name}/vec.npz".format(dir_name=filename)
    param_filename = "{dir_name}/param.json".format(dir_name=filename)

    np.savez(emb_filename, class_vec=class_vec, labels=labels)
    with open(param_filename, "w") as f:
        json.dump(kwargs, f)


def SemAxis_from_file(filename, vec, **params):
    """Calculate SemAxis from file.

    :param filename: filename
    :type filename: str
    :param vec: Embedding vectors
    :type vec: numpy.ndarray
    :return: tuple containing the returns from SemAxis
    :rtype: tuple

    .. highlight:: python
    .. code-block:: python

        >>> import emlens
        >>> xy = emlens.SemAxis_from_file('semspace', emb)
    """
    emb_filename = "{dir_name}/vec.npz".format(dir_name=filename)
    param_filename = "{dir_name}/param.json".format(dir_name=filename)

    data = np.load(emb_filename, allow_pickle=True)
    class_vec = data["class_vec"]
    labels = data["labels"]

    with open(param_filename, "r") as f:
        saved_params = json.load(f)

    if isinstance(params, dict):
        for k, v in params.items():
            saved_params[k] = v
    return SemAxis(vec, class_vec, labels, **saved_params)
