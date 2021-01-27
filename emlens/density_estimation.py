import faiss
import numpy as np
from scipy import sparse, special


def estimate_pdf(locations, emb, C0=0.1):
    """Estimate the density of points at given locations in embedding space
    using the KNN desity estimator.

    Reference
    https://faculty.washington.edu/yenchic/18W_425/Lec7_knn_basis.pdf

    :params: locations: Location at which the density is calculated.
    :type locations: numpy.array, shape=(num_locations, dim)
    :params emb: Embedding vectors of points
    :type emb: numpy.ndarray, (num_point, dim)
    :params C: Parameter for band width. Roughly C * emb.shape[0] nearest neighbors will be used for density estimation.
    :type C: str, optional
    :return: Density of points given by `emb` at `locations`.
    :rtype: numpy.ndarray (num_locations,)
    
    .. highlight:: python
    .. code-block:: python 

        >>> import emlens
        >>> import numpy as np
        >>> emb = np.random.randn(100, 20)
        >>> density = emlens.make_knn_graph(locations=emb, emb = emb)
    """
    n = emb.shape[0]
    dim = emb.shape[1]
    k = int(np.round(C0 * np.power(n, 4 / 5)))

    # Construct the knn graph
    index = faiss.IndexFlatL2(dim)
    index.add(emb.astype(np.float32))
    distances, indices = index.search(locations.astype(np.float32), k=k)

    #
    # KNN density estimator
    # https://faculty.washington.edu/yenchic/18W_425/Lec7_knn_basis.pdf
    #
    logVd = np.log(np.pi) * (dim / 2.0) - special.loggamma(dim / 2.0 + 1)
    Rk = np.max(distances, axis=1)
    density = np.log(k) - np.log(n) - dim * np.log(Rk) - logVd
    return density
