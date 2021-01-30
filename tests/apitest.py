import unittest

import numpy as np
import pandas as pd
from scipy import sparse

import emlens


class TestCalc(unittest.TestCase):
    def setUp(self):
        emb_url = "https://raw.githubusercontent.com/skojaku/emlens/main/data/airportnet/emb.txt"
        node_url = "https://raw.githubusercontent.com/skojaku/emlens/main/data/airportnet/nodes.csv"
        edge_url = "https://raw.githubusercontent.com/skojaku/emlens/main/data/airportnet/edges.csv"
        self.emb = np.loadtxt(emb_url)
        self.node_table = pd.read_csv(node_url)
        edge_table = pd.read_csv(edge_url)

        N = self.node_table.shape[0]
        self.net = sparse.csr_matrix(
            (edge_table.weight, (edge_table.source, edge_table.target)), shape=(N, N)
        )
        self.deg = np.array(self.net.sum(axis=0)).reshape(-1)  # calculate the degree

    def test_semaxis(self):
        for mode in ["fda", "lda"]:
            xy = emlens.SemAxis(
                vec=self.emb,
                class_vec=self.emb,
                labels=self.node_table["region"].values,
                dim=2,
                mode=mode,
            )
            self.assertEqual(xy.shape[0], self.emb.shape[0])
            self.assertEqual(xy.shape[1], 2)

    def test_assortativity(self):
        rho = emlens.assortativity(self.emb, self.deg)

    def test_modularity(self):
        rho = emlens.modularity(self.emb, self.node_table["region"])

    def test_pairwise_dot_similarity(self):
        S, labels = emlens.pairwise_dot_sim(self.emb, self.node_table["region"])
        self.assertEqual(S.shape[1], 6)

    def test_pairwise_distance(self):
        S, labels = emlens.pairwise_distance(self.emb, self.node_table["region"])
        self.assertEqual(S.shape[1], 6)

    def test_estimate_pdf(self):
        density = emlens.estimate_pdf(locations=self.emb, emb=self.emb)
        
    def calculate_rog(self):
        emb_by_region = emb[node_table["region"] == "Asia"]
        rog = emlens.radius_of_gyration(emb_by_region, 'euc')


if __name__ == "__main__":
    unittest.main()
