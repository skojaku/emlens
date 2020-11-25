import numpy as np
import scipy
from scipy import sparse
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os
import json
import shutil


def calcSemAxis(
    vec,
    class_vec,
    labels,
    label_order=None,
    dim=1,
    mode="semaxis",
    centering=True,
    return_class_vec=False,
    **params
):
    """
    Find SemAxis

    Parameters
    ----------
    vec : np.array, shape=(num_data, num_dim)
        Embedding vector for data points.
    class_vec : np.array, shape=(num_data, num_dim)
        Embedding vector for data with class labels.
    labels : list, np.array
        class labels associated with class_vec. labels[i] indicates the label for a point embedded at class_vec[i,:]
    label_order : list (Optional;Default None)
        Order of labels. Only affect the rotation of the space.
    dim : int (Optional;Default 1)
        The dimension of space onto which the data points are projected.
        For example, set dim=1 to get an axis. Setting dim=2 will generate 2d space.
        If mode='semaxis', dim is automatically set to 1.
    mode : 'semaxis', 'lda', or 'fda' (Optional; Default 'semaxis')
        If mode='semaxis', an axis is computed based on [1]. If mode='lda', data are projected using lda. If mode='fda', Fisher discriminant analysis. 
        fda is often a better choice over 'lda' since lda assumes that each class has the same covariance while fda does not.
    centering : bool (Optiona; Default True)
        If True, centerize the projected data points.
    return_class_vec : bool (Optional; Default False)
        If return_class_vec=True, return the projection of the embedding vector with class labels.
    **params : parameters
        Parameters for sklean.discriminant_analysis.LinearDiscriminantAnalysis


    Returns
    -------
    retvec : np.array, shape=(num_data, dim)
        The vectors for the data points projected onto semaxis (or semspace).
    class_vec : np.array, shape=(num_data_with_labels, dim) (Optional)
        The projection of vectors used to construct the axis (or space).
        This variable is returned only when return_class_vec=True.
    labels : np.array, shape=(num_data_with_labels,) (Optional)
        The class labels for vectors used to construct the axis (or space).
        This variable is returned only when return_class_vec=True.
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
        label2cids = {l: i for i, l in enumerate(label_order)}
        class_ids = np.array([label2cids[l] for l in labels])
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
    """
    Fisher's linear discriminant analysis. 

    Parameters
    ----------
    vec : np.array, shape=(num_data, num_dim)
        Embedding vector for data points.
    class_vec : np.array, shape=(num_data, num_dim)
        Embedding vector for data with class labels.
    class_labels : list, np.array
        class labels associated with class_vec. labels[i] indicates the label for a point embedded at class_vec[i,:]
    dim : int (Optional; Default 1)
        The dimension of space onto which the data points are projected.
        For example, set dim=1 to get an axis. Setting dim=2 will generate 2d space.
        If mode='semaxis', dim is automatically set to 1.
    shrinkage : float (Optional; Default 0)
        A shrinkage parameter that controls the level of fitting of model. Useful when dimension is close to the number of data points. 
    priors : str (Optional; 'data', 'uniform', dict)
        A prior distribution for the classes. priors='data' weight each class based on the number of data points. with priors='uniform', each 
        class will be treated equally. One can manually set a prior by passing a dict object taking keys as the class label and values as the (positive) weight.
    return_class_vec : bool (Optional; Default False)
        If return_class_vec=True, return the projection of the embedding vector with class labels.

    Returns
    -------
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
        for k, lab in enumerate(labels):
            class_weight[k] = sum(group_ids == k)
    elif priors == "uniform":
        for k, lab in enumerate(labels):
            class_weight[k] = 1

    class_weight = class_weight / np.sum(class_weight)

    Cw = np.zeros((DIM, DIM))  # Within-class variance
    MU = np.zeros((K, DIM))  # Between-class
    for k in range(K):
        v_k = class_vec[group_ids == k, :]
        mu_k = np.mean(v_k, axis=0)
        MU[k, :] = mu_k

        Cw += class_weight[k] * np.cov(v_k.T)

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
    u = u[:, np.argsort(-np.array(s).reshape(-1))[:dim]]

    # Projection
    prj_vec = vec @ u
    if return_class_vec:
        prj_class_vec = class_vec @ u
        return prj_vec, prj_class_vec
    else:
        return prj_vec


def saveSemAxis(filename, class_vec, labels, **kwargs):
    """
    Save SemAxis into a file

    Parameters
    ----------
    filename : str
        File name
    class_vec : (num_data, num_dim)
        Embedding vectors with class labels
    labels : np.array
        Labels for the class_vec.
    """
    if os.path.exists(filename):
        shutil.rmtree(filename)
    os.mkdir(filename)
    emb_filename = "{dir_name}/vec.npz".format(dir_name = filename)
    param_filename = "{dir_name}/param.json".format(dir_name = filename)
    
    np.savez(emb_filename, class_vec=class_vec, labels=labels)
    with open(param_filename, "w") as f:
        json.dump(kwargs, f)


def calcSemAxis_from_file(filename, vec, **params):
    """
    Calculate SemAxis from file

    Parameters
    ----------
    filename : str
        File name
    vec : (num_data, num_dim)
        Embedding vectors
    **params : all parameters will be passed to calcSemAxis

    Returns
    -------
    return : return from calcSemAxis
    """
    emb_filename = "{dir_name}/vec.npz".format(dir_name = filename)
    param_filename = "{dir_name}/param.json".format(dir_name = filename)

    data = np.load(emb_filename, allow_pickle=True)
    class_vec = data["class_vec"]
    labels = data["labels"]
    
    with open(param_filename, "r") as f:
        saved_params = json.load(f)
   
    if isinstance(params, dict):
        for k, v in params.items():
            saved_params[k] = v
    return calcSemAxis(vec, class_vec, labels, **saved_params)
