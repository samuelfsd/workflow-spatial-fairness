import sys
import unittest
from pathlib import Path

import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from clustering.hdbscan import fit_hdbscan_partition


class HDBSCANTests(unittest.TestCase):
    def test_noise_is_reported_outside_cluster_regions(self):
        df = pd.DataFrame(
            {
                "lat": [0.0, 0.0001, 0.0002, 1.0, 1.0001, 1.0002, 20.0],
                "lon": [0.0, 0.0001, 0.0002, 1.0, 1.0001, 1.0002, 20.0],
            }
        )

        partition = fit_hdbscan_partition(df, min_cluster_frac=0.4, min_cluster_size_min=2)
        clustered_points = {point for region in partition.regions for point in region["points"]}

        self.assertIsInstance(partition.noise_points, list)
        self.assertTrue(clustered_points.isdisjoint(partition.noise_points))
        self.assertEqual(len(clustered_points) + partition.noise_n, len(df))


if __name__ == "__main__":
    unittest.main()
