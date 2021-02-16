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

    def test_assortativity(self):
        emlens.assortativity(self.emb, self.deg)

    def test_modularity(self):
        emlens.modularity(self.emb, self.membership)

    def test_nmi(self):
        emlens.nmi(self.emb, self.membership)

    def test_element_sim(self):
        emlens.element_sim(self.emb, self.membership)

    def test_pairwise_dot_similarity(self):
        S, _ = emlens.pairwise_dot_sim(self.emb, self.membership)
        self.assertEqual(S.shape[1], self.K)

    def test_pairwise_distance(self):
        S, _ = emlens.pairwise_distance(self.emb, self.membership)
        self.assertEqual(S.shape[1], self.K)

    def test_estimate_pdf(self):
        emlens.estimate_pdf(target=self.emb, emb=self.emb, C=0.1)

    def calculate_rog(self):
        emlens.radius_of_gyration(self.emb, "euc")
        emlens.radius_of_gyration(self.emb, "cos")


if __name__ == "__main__":
    unittest.main()
