Module emlens.semaxis
=====================

Functions
---------

    
`calcSemAxis(vec, class_vec, labels, label_order=None, dim=1, mode='semaxis', centering=True, return_class_vec=False, **params)`
:   Find SemAxis
    
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

    
`calcSemAxis_from_file(filename, vec, **params)`
:   Calculate SemAxis from file
    
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

    
`saveSemAxis(filename, class_vec, labels)`
:   Save SemAxis into a file
    
    Params
    ------
    filename: str
        File name
    class_vec: (num_data, num_dim)
        Embedding vectors with class labels
    labels: np.array
        Labels for the class_vec.