import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from clustering.base import Partition
from metrics.neighbors import build_delaunay_adjacency


def _partition(labels_points: dict[int, list[int]], n_rows: int) -> Partition:
    regions = [
        {"points": points, "cluster_label": label}
        for label, points in labels_points.items()
    ]
    return Partition(method="fake", params={}, labels=np.zeros(n_rows, dtype=int), regions=regions)


class NeighborsTests(unittest.TestCase):
    def test_three_clusters_form_one_triangle_all_adjacent(self):
        # Centroids at (0,0), (0,4), (3,2): non-collinear => single Delaunay
        # triangle => every cluster is adjacent to the other two.
        df = pd.DataFrame({"lat": [0.0, 0.0, 3.0], "lon": [0.0, 4.0, 2.0]})
        partition = _partition({0: [0], 1: [1], 2: [2]}, n_rows=3)

        adjacency = build_delaunay_adjacency(partition, df)

        self.assertEqual(adjacency, {0: [1, 2], 1: [0, 2], 2: [0, 1]})

    def test_two_clusters_are_mutual_peers(self):
        df = pd.DataFrame({"lat": [0.0, 5.0], "lon": [0.0, 5.0]})
        partition = _partition({0: [0], 1: [1]}, n_rows=2)

        self.assertEqual(build_delaunay_adjacency(partition, df), {0: [1], 1: [0]})

    def test_single_cluster_has_no_peers(self):
        df = pd.DataFrame({"lat": [0.0], "lon": [0.0]})
        partition = _partition({0: [0]}, n_rows=1)

        self.assertEqual(build_delaunay_adjacency(partition, df), {0: []})


if __name__ == "__main__":
    unittest.main()
