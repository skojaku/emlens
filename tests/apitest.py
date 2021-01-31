import unittest

import numpy as np
from scipy import sparse

import emlens


class TestCalc(unittest.TestCase):
    def setUp(self):
        emb_url = "https://raw.githubusercontent.com/skojaku/emlens/main/data/airportnet/emb.txt"
        self.emb = np.loadtxt(emb_url)
        self.deg = np.random.randn(self.emb.shape[0])
        self.membership = np.random.randint(10, size=self.emb.shape[0])
        self.K = len(set(self.membership))

    def test_semaxis(self):
        for mode in ["fda", "lda"]:
            xy = emlens.SemAxis(
                vec=self.emb,
                class_vec=self.emb,
                labels=self.membership,
                dim=2,
                mode=mode,
            )
            self.assertEqual(xy.shape[0], self.emb.shape[0])
            self.assertEqual(xy.shape[1], 2)

    def test_assortativity(self):
        emlens.assortativity(self.emb, self.deg)

    def test_modularity(self):
        emlens.modularity(self.emb, self.membership)

    def test_pairwise_dot_similarity(self):
        S, _ = emlens.pairwise_dot_sim(self.emb, self.membership)
        self.assertEqual(S.shape[1], self.K)

    def test_pairwise_distance(self):
        S, _ = emlens.pairwise_distance(self.emb, self.membership)
        self.assertEqual(S.shape[1], self.K)

    def test_estimate_pdf(self):
        emlens.estimate_pdf(locations=self.emb, emb=self.emb)

    def calculate_rog(self):
        emlens.radius_of_gyration(self.emb, "euc")
        emlens.radius_of_gyration(self.emb, "cos")


if __name__ == "__main__":
    unittest.main()
