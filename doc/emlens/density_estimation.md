Module emlens.density_estimation
================================

Functions
---------

    
`estimate_pdf(locations, emb, kernel='epanechnikov', n_jobs=30)`
:   Estimate the density of points at given locations in embedding space.
    A kernel density estimation is used to compute the density.
    
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