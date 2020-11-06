Module emlens
=============

Sub-modules
-----------
* emlens.density_estimation
* emlens.metrics
* emlens.semaxis
* emlens.vis


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


Module emlens.metrics
=====================

Functions
---------

    
`calc_pairwise_separation_score(emb, groups, separation_score_func=<function youden_separation_score>, is_symmetric=True, n_jobs=20)`
:   Calculate the separation score for each pair of groups of vectors.
    
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

    
`youden_separation_score(emb_a, emb_b)`
:   Calculate the level of separation between two sets of vectors using the Youden index.
    
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


Module emlens.semaxis
=====================

Functions
---------

    
`calcSemAxis(vec, class_vec, labels, label_order=None, dim=1, mode='semaxis', centering=True, return_class_vec=False, **params)`
:   Find SemAxis
    
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
    mode : 'semaxis' or 'lda' (Optional; Default 'semaxis')
        If mode='semaxis', an axis is computed based on [1]. If mode='lda', data are projected using lda.
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

    
`calcSemAxis_from_file(filename, vec, **params)`
:   Calculate SemAxis from file
    
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

    
`saveSemAxis(filename, class_vec, labels)`
:   Save SemAxis into a file
    
    Parameters
    ----------
    filename : str
        File name
    class_vec : (num_data, num_dim)
        Embedding vectors with class labels
    labels : np.array
        Labels for the class_vec.


Module emlens.vis
=================

Functions
---------

    
`repel_labels(ax, x, y, labels, color='#4d4d4d', label_width=30, arrow_shrink=5, text_params={}, adjust_text_params={})`
:   Add text labels to the points. The position of text will be automatically adjusted to avoid overlap.