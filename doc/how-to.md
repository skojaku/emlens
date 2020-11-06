# Table of Contents

* [emlens](#emlens)
* [emlens.metrics](#emlens.metrics)
  * [youden\_separation\_score](#emlens.metrics.youden_separation_score)
  * [calc\_pairwise\_separation\_score](#emlens.metrics.calc_pairwise_separation_score)
* [emlens.density\_estimation](#emlens.density_estimation)
* [emlens.vis](#emlens.vis)
  * [repel\_labels](#emlens.vis.repel_labels)
* [emlens.semaxis](#emlens.semaxis)
  * [calcSemAxis](#emlens.semaxis.calcSemAxis)
  * [saveSemAxis](#emlens.semaxis.saveSemAxis)
  * [calcSemAxis\_from\_file](#emlens.semaxis.calcSemAxis_from_file)

<a name="emlens"></a>
# emlens

<a name="emlens.metrics"></a>
# emlens.metrics

<a name="emlens.metrics.youden_separation_score"></a>
#### youden\_separation\_score

```python
youden_separation_score(emb_a, emb_b)
```

Calculate the level of separation between two sets of vectors using the Youden index.

Given points of two classes (a, b), i.e., emb_a, emb_b, we project points onto a line
using the LDA; LDA does so by maximizing the separation between two classes on the line.
Then, we measure the level of separation of the two classes on the line.
We consider class "a" as a positive class and the other class "b" as a negative class.
Then, for the points on the line, we compute the Youden index.
By running a threshold from one end of the line to the other end, we compute the true positive rate and false positive rate.
The Youden index is given by maximum absolute difference between the true positive rate - false positive rate.

Params
------
emb_a: numpy.array, shape=(num_point, dim)
    Embedding vectors for class a.
emb_a: numpy.array, shape=(num_point, dim)
    Embedding vectors for class b.

Return
------
score: float
    Youden index

<a name="emlens.metrics.calc_pairwise_separation_score"></a>
#### calc\_pairwise\_separation\_score

```python
calc_pairwise_separation_score(emb, groups, separation_score_func=youden_separation_score, is_symmetric=True, n_jobs=20)
```

Calculate the separation score for each pair of groups of vectors.

Params
------
emb: numpy.array, shape=(num_point, dim)
    embedding vector
groups: numpy.array, shape=(num_point,)
    group membership for points. Can be integer or string
    calc_separation_score
separation_score_func: function
    Function to compute the separation score. This function takes two arguments, emb_a and emb_b, and
    compute the separation level for the two vectors.
is_symmetric: bool (Optional; Default True)
    Set True if the separation score is symmetric with respect to the groups, i.e., the score does not change
    when we swap the class labels.
n_jobs: int (Optional; Default 20)
    Number of cores

Returns
-------
separation_matrix: numpy.array, shape=(num_groups, num_groups)
    separation_matrix[i,j] indicates the sepration level of group_labels[i] and group_labels[j].
group_labels: numpy.array
    List of groups.

<a name="emlens.density_estimation"></a>
# emlens.density\_estimation

<a name="emlens.vis"></a>
# emlens.vis

<a name="emlens.vis.repel_labels"></a>
#### repel\_labels

```python
repel_labels(ax, x, y, labels, color="#4d4d4d", label_width=30, arrow_shrink=5, text_params={}, adjust_text_params={})
```

Add text labels to the points. The position of text will be automatically adjusted to avoid overlap.

<a name="emlens.semaxis"></a>
# emlens.semaxis

<a name="emlens.semaxis.calcSemAxis"></a>
#### calcSemAxis

```python
calcSemAxis(vec, class_vec, labels, label_order=None, dim=1, mode="semaxis", centering=True, return_class_vec=False, **params)
```

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

<a name="emlens.semaxis.saveSemAxis"></a>
#### saveSemAxis

```python
saveSemAxis(filename, class_vec, labels)
```

Save SemAxis into a file

Params
------
filename: str
    File name
class_vec: (num_data, num_dim)
    Embedding vectors with class labels
labels: np.array
    Labels for the class_vec.

<a name="emlens.semaxis.calcSemAxis_from_file"></a>
#### calcSemAxis\_from\_file

```python
calcSemAxis_from_file(filename, vec, **params)
```

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

