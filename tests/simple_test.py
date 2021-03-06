import shutil
import unittest

import numpy as np
from scipy import sparse

import emlens


class TestCalc(unittest.TestCase):
    def setUp(self):
        self.emb = np.random.randn(300, 30)
        self.deg = np.random.randn(self.emb.shape[0])
        self.membership = np.random.randint(10, size=self.emb.shape[0])
        self.K = len(set(self.membership))

    def test_semaxis(self):
        model = emlens.SemAxis()
        model.fit(self.emb, self.membership)
        model.transform(self.emb)
        model.save("random-semaxis.sm")
        model = emlens.LDASemAxis().load("random-semaxis.sm")
        model.transform(self.emb)
        shutil.rmtree("random-semaxis.sm")

    def test_lda_semaxis(self):
        model = emlens.LDASemAxis()
        model.fit(self.emb, self.membership)
        model.transform(self.emb)
        model.save("random-semaxis.sm")
        shutil.rmtree("random-semaxis.sm")

    def test_make_knn_graph(self):
        emlens.make_knn_graph(self.emb)
        emlens.make_knn_graph(self.emb, k=40, binarize=False)
        emlens.make_knn_graph(self.emb, k=[3, 10, 20, 100], binarize=True)

    def test_assortativity(self):
        for metric in ["euclidean", "cosine", "dotsim"]:
            emlens.assortativity(self.emb, self.membership, metric=metric)

    def test_modularity(self):
        for metric in ["euclidean", "cosine", "dotsim"]:
            emlens.modularity(self.emb, self.membership, metric=metric)

    def test_nmi(self):
        for metric in ["euclidean", "cosine", "dotsim"]:
            emlens.nmi(self.emb, self.membership, metric=metric)

    def test_knn_pred_score(self):
        for metric in ["euclidean", "cosine", "dotsim"]:
            emlens.f1_score(self.emb, self.membership, metric=metric)

    def test_knn_pred_score_from_net(self):
        A = emlens.make_knn_graph(self.emb, k=40, binarize=False)
        for model in ["knn"]:
            for metric in ["euclidean", "cosine", "dotsim"]:
                emlens.r2_score(self.emb, self.membership, model=model, metric=metric)
        emlens.r2_score(self.emb, self.membership, model="linear")

    def test_r2score_from_net(self):
        emlens.r2_score(self.emb, self.membership, model="linear")

    def test_element_sim(self):
        for metric in ["euclidean", "cosine", "dotsim"]:
            emlens.element_sim(self.emb, self.membership, metric=metric)

    def test_pairwise_dot_similarity(self):
        S, _ = emlens.pairwise_dot_sim(self.emb, self.membership)
        self.assertEqual(S.shape[1], self.K)

    def test_pairwise_distance(self):
        S, _ = emlens.pairwise_distance(self.emb, self.membership)
        self.assertEqual(S.shape[1], self.K)

    def test_estimate_pdf(self):
        emlens.estimate_pdf(target=self.emb, emb=self.emb, C=0.1)
        a = np.array([1, 2, 3])
        b = np.array([[1, 2, 3], [1, 2, 3]])
        emlens.estimate_pdf(target=a.reshape((1, -1)), emb=b, C=0.1)

    def test_calculate_rog(self):
        emlens.rog(self.emb, "euc")
        emlens.rog(self.emb, "cos")

    def test_effective_dimensionality(self):
        emlens.effective_dimension(self.emb, q=1)


if __name__ == "__main__":
    unittest.main()
