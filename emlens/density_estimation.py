import numpy as np
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.neighbors import KernelDensity


def estimate_pdf(locations, emb, kernel="epanechnikov", dh=None, n_jobs=30):
    """Estimate the density of points at given locations in embedding space. A
    kernel density estimation is used to compute the density.

    :params: locations: Location at which the density is calculated.
    :type locations: numpy.array, shape=(num_locations, dim)
    :params emb: Embedding vectors of points
    :type emb: numpy.ndarray, (num_point, dim)
    :params kernel: type of kernel, See https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html#sklearn.neighbors.KernelDensity., defaults to "epanechnikov"
    :type kernel: str, optional
    :params n_jobs: Number of cores., defaults to 30
    :type n_jobs: int, optional
    :return: Density of points given by `emb` at `locations`.
    :rtype: numpy.ndarray (num_locations,)
    """
    if dh is None:
        dh = np.power(
            emb.shape[0], -1.0 / (emb.shape[1] + 4)
        )  # Silverman's rule-of-thumb rule
    kde = KernelDensity(kernel=kernel, bandwidth=dh).fit(emb)

    def est(kde, locations, ids):
        p = kde.score_samples(locations[ids, :])
        return p, ids

    chunks = np.array_split(np.arange(locations.shape[0]), n_jobs)
    results = Parallel(n_jobs=n_jobs)(
        delayed(est)(kde, locations, ids) for ids in chunks
    )

    ids = np.concatenate([res[1] for res in results])
    prob_density = np.concatenate([res[0] for res in results])
    order = np.argsort(ids)
    ids = ids[order]
    prob_density = prob_density[order]
    return prob_density
