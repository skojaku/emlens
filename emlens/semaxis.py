import numpy as np
from scipy import sparse
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


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

    Params
    ------
    vec: np.array, shape=(num_data, num_dim)
        Embedding vector for data points.
    class_vec: np.array, shape=(num_data, num_dim)
        Embedding vector for data with class labels.
    labels: list, np.array
        class labels associated with class_vec. labels[i] indicates the label for a point embedded at class_vec[i,:]
    label_order: list (Optional;Default None)
        Order of labels. Only affect the rotation of the space.
    dim: int (Optional;Default 1)
        The dimension of space onto which the data points are projected.
        For example, set dim=1 to get an axis. Setting dim=2 will generate 2d space.
        If mode='semaxis', dim is automatically set to 1.
    mode: 'semaxis' or 'lda' (Optional; Default 'semaxis')
        If mode='semaxis', an axis is computed based on [1]. If mode='lda', data are projected using lda.
    centering: bool (Optiona; Default True)
        If True, centerize the projected data points.
    return_class_vec: bool (Optional; Default False)
        If return_class_vec=True, return the projection of the embedding vector with class labels.
    **params: parameters
        Parameters for sklean.discriminant_analysis.LinearDiscriminantAnalysis


    Returns
    -------
    retvec: np.array, shape=(num_data, dim)
        The vectors for the data points projected onto semaxis (or semspace).
    class_vec: np.array, shape=(num_data_with_labels, dim) (Optional)
        The projection of vectors used to construct the axis (or space).
        This variable is returned only when return_class_vec=True.
    labels: np.array, shape=(num_data_with_labels,) (Optional)
        The class labels for vectors used to construct the axis (or space).
        This variable is returned only when return_class_vec=True.
    """

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
        projected_class_vec = sparse.diags(denom) @ (class_vec @ vr.T)
    elif mode == "lda":
        lda = LinearDiscriminantAnalysis(n_components=dim, **params)
        lda.fit(class_vec, class_ids)
        ret_vec = lda.transform(vec)
        projected_class_vec = lda.transform(class_vec)
        print(projected_class_vec, class_vec)

    if centering:
        if dim == 1 and mode == "lda":
            mu0 = np.mean(projected_class_vec[class_ids == 0])
            mu1 = np.mean(projected_class_vec[class_ids == 1])

            s = np.sum(
                np.power(
                    np.array(projected_class_vec).reshape(-1)
                    - np.array([mu0, mu1])[class_ids],
                    2,
                )
            ) / (len(projected_class_vec) - 2)
            p0 = np.sum(class_ids == 0) / class_ids.size
            p1 = np.sum(class_ids == 1) / class_ids.size

            x = (mu0 * mu0 / (2 * s) - mu1 * mu1 / (2 * s) + np.log(p1 / p0)) / (
                mu0 / s - mu1 / s
            )
            ret_vec -= x
            projected_class_vec -= x

        else:
            class_centers = np.zeros((n_class, dim))
            for cid in range(n_class):
                class_centers[cid, :] = (
                    np.mean(projected_class_vec[class_ids == cid, :], axis=0)
                    if dim > 1
                    else np.mean(projected_class_vec[class_ids == cid, :])
                )
            ret_vec -= (
                np.mean(class_centers, axis=0) if dim > 1 else np.mean(class_centers)
            )
            projected_class_vec -= (
                np.mean(class_centers, axis=0) if dim > 1 else np.mean(class_centers)
            )

    if return_class_vec:
        return ret_vec, projected_class_vec, labels
    else:
        return ret_vec


def saveSemAxis(filename, class_vec, labels):
    """
    Save SemAxis into a file

    Params
    ------
    filename: str
        File name
    class_vec: (num_data, num_dim)
        Embedding vectors with class labels
    labels: np.array
        Labels for the class_vec.
    """
    np.savez(filename, class_vec=class_vec, labels=labels)


def calcSemAxis_from_file(filename, vec, **params):
    """
    Calculate SemAxis from file

    Params
    ------
    filename: str
        File name
    vec: (num_data, num_dim)
        Embedding vectors
    dim: int (Optional;Default 1)
        The dimension of space onto which the data points are projected.
        For example, set dim=1 to get an axis. Setting dim=2 will generate 2d space.
        If mode='semaxis', dim is automatically set to 1.
    mode: 'semaxis' or 'lda' (Optional; Default 'semaxis')
        If mode='semaxis', an axis is computed based on [1]. If mode='lda', data are projected using lda.
    return_class_vec: bool (Optional; Default False)
        If return_class_vec=True, return the projection of the embedding vector with class labels.
    **params: all parameters will be passed to calcSemAxis

    Return
    ------
    retvec: np.array, shape=(num_data, dim)
        The vectors for the data points projected onto semaxis (or semspace).
    class_vec: np.array, shape=(num_data_with_labels, dim) (Optional)
        The projection of vectors used to construct the axis (or space).
        This variable is returned only when return_class_vec=True.
    labels: np.array, shape=(num_data_with_labels,) (Optional)
        The class labels for vectors used to construct the axis (or space).
        This variable is returned only when return_class_vec=True.
    """
    data = np.load(filename, allow_pickle=True)
    class_vec = data["class_vec"]
    labels = data["labels"]
    return calcSemAxis(vec, class_vec, labels, **params)
