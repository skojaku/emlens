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