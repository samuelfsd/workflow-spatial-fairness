import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from data_loading import DATASET_SPECS
from regions import create_grid_from_dataset, create_random_partitionings, create_regions, create_rtree, create_seeds


class RegionTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "lat": np.linspace(0.0, 1.0, 100),
                "lon": np.linspace(0.0, 1.0, 100),
                "outcome": [0, 1] * 50,
            }
        )
        self.rtree = create_rtree(self.df)

    def test_fixed_grid_region_count(self):
        _, _, regions = create_grid_from_dataset(self.df, self.rtree, lon_n=4, lat_n=3)

        self.assertEqual(len(regions), 12)

    def test_random_partitionings_respect_range(self):
        partitionings = create_random_partitionings(self.df, self.rtree, n_partitionings=5, seed=7)

        self.assertEqual(len(partitionings), 5)
        for grid_info, _, _ in partitionings:
            self.assertGreaterEqual(grid_info["lon_n"], 10)
            self.assertLessEqual(grid_info["lon_n"], 40)
            self.assertGreaterEqual(grid_info["lat_n"], 10)
            self.assertLessEqual(grid_info["lat_n"], 40)

    def test_kmeans_lar_defaults_generate_two_thousand_regions(self):
        radii = np.round(
            np.arange(
                DATASET_SPECS["lar"].radii_start,
                DATASET_SPECS["lar"].radii_stop,
                DATASET_SPECS["lar"].radii_step,
            ),
            10,
        )
        seeds = create_seeds(self.df, self.rtree, n_seeds=100, random_state=1)
        regions = create_regions(self.df, self.rtree, seeds, radii)

        self.assertEqual(len(seeds), 100)
        self.assertEqual(len(regions), 2000)


if __name__ == "__main__":
    unittest.main()
