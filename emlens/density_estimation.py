import numpy as np
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.neighbors import KernelDensity


def estimate_pdf(locations, emb, kernel="epanechnikov", dh=None, n_jobs=30):
    """Estimate the density of points at given locations in embedding space. A
    kernel density estimation is used to compute the density.

    Parameters
    ----------
    locations : numpy.array, shape=(num_locations, dim)
        Location at which the density is calculated.
    emb : numpy.array, shape=(num_point, dim)
        Embedding vectors of points.
    kernel : str (Optional; Default "epanechnikov")
        Type of kernel function used to estimate density.
        See https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html#sklearn.neighbors.KernelDensity.
    n_jobs : int (Optional; Default 30)
        Number of cores.

    Returns
    -------
    prob_density : numpy.array (num_locations,)
        Density of points given by `emb` at `locations`.
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
