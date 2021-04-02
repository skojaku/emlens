import faiss
import numpy as np
from scipy import sparse, special


def estimate_pdf(target, emb, C=0.1):
    """Estimate the density of entities at the given target locations in the
    embedding space using the density estimator based on the k-nearest
    neighbors.

    :param target: Target location at which the density is calculated.
    :type target: numpy.array, shape=(num_target, dim)
    :param emb: Embedding vectors for the entities
    :type emb: numpy.ndarray, (num_entities, dim)
    :param C: Bandwidth for kernels. Ranges between (0,1]. Roughly C * num_entities nearest neighbors will be used for estimating the density at a single target location.
    :type C: float, optional
    :return: Log-density of points at the target locations.
    :rtype: numpy.ndarray (num_target,)

    Reference
    https://faculty.washington.edu/yenchic/18W_425/Lec7_knn_basis.pdf

    .. highlight:: python
    .. code-block:: python

        >>> import emlens
        >>> import numpy as np
        >>> emb = np.random.randn(100, 20)
        >>> target = np.random.randn(10, 20)
        >>> density = emlens.estimate_pdf(target=target, emb = emb)
    """
    if len(emb.shape) != 2:
        raise TypeError(
            "emb should be 2D numpy array of size (number of points, dimensions)"
        )

    if len(target.shape) != 2:
        raise TypeError(
            "target should be 2D numpy array of size (number of points, dimensions)"
        )

    n = emb.shape[0]
    dim = emb.shape[1]

    k = np.maximum(1, np.round(C * np.power(n, 4 / 5)))
    k = int(k)

    # Construct the knn graph
    index = faiss.IndexFlatL2(dim)
    index.add(emb.astype(np.float32))
    distances, indices = index.search(target.astype(np.float32), k=k)

    #
    # KNN density estimator
    # https://faculty.washington.edu/yenchic/18W_425/Lec7_knn_basis.pdf
    #
    logVd = np.log(np.pi) * (dim / 2.0) - special.loggamma(dim / 2.0 + 1)
    Rk = np.max(distances, axis=1)
    density = np.log(k) - np.log(n) - dim * np.log(Rk) - logVd
    return density
