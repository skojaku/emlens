import json
import os
import shutil

import numpy as np
from scipy import sparse
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class SemAxis:
    """SemAxis.

    A SemAxis is an axis going through two entity groups in the embedding space [1].

    Reference:

        * [1] An, J., Kwak, H., & Ahn, Y.-Y. (2018). SemAxis: A Lightweight Framework to Characterize Domain-Specific Word Semantics Beyond Sentiment. Proc. the 56th Annual Meeting of the Association for Computational Linguistics, 1, 2450–2461.

    .. highlight:: python
    .. code-block:: python

        >>> import emlens
        >>> import numpy as np
        >>> emb = np.random.randn(100, 20) # Embedding vectors to ground the SemAxis
        >>> group_ids = np.random.choice(2, 100) # Membership of entities
        >>> target = np.random.randn(10, 20) # Vectors we will project onto the SemAxis
        >>> model = emlens.SemAxis() # load SemAxis Object
        >>> model.fit(emb, group_ids) # Fit the SemAxis
        >>> model.transform(target) # Project `target` to the axis
        >>> model.save("random-semaxis.sm")
    """

    def __init__(self):
        self.emb = None
        self.prj_emb = None
        self.semaxis = None

        # private
        self._group_ids = None  # int ids of group membership

    def fit(self, emb, group_ids, group_order=None):
        """Finding the SemAxis from embedding vectors.

        :param emb: embedding vectors for locating the SemAxis
        :type emb: numpy.ndarray (num_entities, dim)
        :param group_ids: group_ids.
        :type group_ids: numpy.ndarray (num_entities, dim), defaults to None.
        :type group_order: list, optional
        """
        if group_order is None:
            self.group_order, self._group_ids = np.unique(
                group_ids, return_inverse=True
            )
            self.group_order = self.group_order.tolist()
            self.n_group = len(self.group_order)
        else:
            self.group_order = group_order
            group2cids = {ll: i for i, ll in enumerate(self.group_order)}
            self._group_ids = np.array([group2cids[ll] for ll in group_ids])
            self.n_group = len(group2cids)
        self.emb = emb
        return self

    def transform(self, target):
        """Project the target vectors onto SemAxis.

        :param target: target embedding vectors to project onto the SemAxis.
        :type target: numpy.ndarray (num_target, dim)
        :return: Projected embedding vectors.
        :rtype: numpy.ndarray (num_data,)
        """
        left_center = np.mean(self.emb[self._group_ids == 0, :], axis=0)
        right_center = np.mean(self.emb[self._group_ids == 1, :], axis=0)
        semaxis = right_center - left_center

        denom = np.linalg.norm(target, axis=1) * np.linalg.norm(semaxis)
        denom = 1 / np.maximum(denom, 1e-20)
        prj_target = sparse.diags(denom) @ (target @ semaxis.T)

        denom = np.linalg.norm(self.emb, axis=1) * np.linalg.norm(semaxis)
        denom = 1 / np.maximum(denom, 1e-20)
        self.prj_emb = sparse.diags(denom) @ (self.emb @ semaxis.T)

        self.semaxis = semaxis

        return prj_target

    def save(self, filename):
        """Save the fitted axis.

        :param filename: name of file
        :type filename: str

        .. highlight:: python
        .. code-block:: python

            >>> import emlens
            >>> import numpy as np
            >>> emb = np.random.randn(100, 20)
            >>> group_ids = np.random.choice(100, 10)
            >>> emlens.saveSemAxis('semspace.sm')
        """
        if os.path.exists(filename):
            shutil.rmtree(filename)
        os.mkdir(filename)

        emb_filename = "{dir_name}/emb.npz".format(dir_name=filename)
        param_filename = "{dir_name}/param.json".format(dir_name=filename)

        np.savez(
            emb_filename, emb=self.emb, group_ids=self._group_ids, semaxis=self.semaxis
        )

        params = {"n_group": self.n_group, "group_order": self.group_order}
        with open(param_filename, "w") as f:
            json.dump(params, f)

    def load(self, filename):
        """Load a saved SemAxis file.

        :param filename: filename
        :type filename: str

        .. highlight:: python
        .. code-block:: python

            >>> import emlens
            >>> xy = emlens.SemAxis_from_file('semspace', emb)
        """
        emb_filename = "{dir_name}/emb.npz".format(dir_name=filename)
        param_filename = "{dir_name}/param.json".format(dir_name=filename)

        data = np.load(emb_filename, allow_pickle=True)
        self.emb = data["emb"]
        self.semaxis = data["semaxis"]
        self._group_ids = data["_group_ids"]

        with open(param_filename, "r") as f:
            params = json.load(f)

        for k, v in params.items():
            setattr(self, k, v)


class LDASemAxis(SemAxis):
    """SemAxis based on Linear Discriminant Analysis.

    A variant of SemAxis that finds the axis based on the Linear Discriminat Analysis (LDA). This LDA-based SemAxis separates the two groups more than the original SemAxis approach. Furtheremore, the variant can find a "space" that best separates the groups.
    See https://en.wikipedia.org/wiki/Linear_discriminant_analysis.


    .. highlight:: python
    .. code-block:: python

        >>> import emlens
        >>> import numpy as np
        >>> emb = np.random.randn(100, 20) # Embedding vectors to ground the SemAxis
        >>> group_ids = np.random.choice(2, 100) # Membership of entities
        >>> target = np.random.randn(10, 20) # Vectors we will project onto the SemAxis
        >>> model = emlens.LDASemAxis() # load SemAxis Object
        >>> model.fit(emb, group_ids) # Fit the SemAxis
        >>> model.transform(target, dim = 1) # Project `target` to the axis
        >>> model.transform(target, dim = 2) # Project `target` to a 2D space
        >>> model.save("random-semaxis.sm")
    """

    def __init__(self, **params):
        """Initialize the instnace of SemAxis based on Linear Discriminant
        Analysis.

        :param mode: type of algorithm, defaults to "fda"
        :type mode: str, optional
        """
        SemAxis.__init__(self, **params)

    def transform(self, target, dim=1, **params):
        """Project the target vectors onto SemAxis.

        :param dim: dimension for the projected space, defaults to 1
        :type dim: int, optional
        :return: Projected embedding vectors.
        :rtype: numpy.ndarray (num_data,)
        """
        lda = LinearDiscriminantAnalysis(n_components=dim, **params)
        lda.fit(self.emb, self._group_ids)
        prj_target = lda.transform(target)
        self.prj_emb = lda.transform(self.emb)
        return prj_target
