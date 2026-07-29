import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from clustering.base import Partition
from clustering.capped import (
    density_subclusters,
    recursive_density_split,
    run_capped_hdbscan_sweep,
)
from clustering.registry import get_partitioner, partitioner_names


def _two_blobs(n_per: int = 30, sep: float = 10.0) -> pd.DataFrame:
    a_lat = np.linspace(0.0, 0.01, n_per)
    b_lat = sep + np.linspace(0.0, 0.01, n_per)
    lat = np.concatenate([a_lat, b_lat])
    lon = np.concatenate([np.zeros(n_per), np.full(n_per, sep)])
    return pd.DataFrame({"lat": lat, "lon": lon})


def _one_blob(n: int = 60) -> pd.DataFrame:
    return pd.DataFrame({"lat": np.linspace(0.0, 0.01, n), "lon": np.zeros(n)})


class CappedSplitTests(unittest.TestCase):
    def test_recursive_split_separates_two_dense_groups(self):
        df = _two_blobs(30)
        groups = recursive_density_split(df, list(range(60)), max_size=40)
        self.assertEqual(len(groups), 2)
        self.assertTrue(all(len(g) <= 40 for g in groups))
        self.assertEqual(sorted(p for g in groups for p in g), list(range(60)))

    def test_recursive_split_leaves_homogeneous_blob_intact(self):
        # A single dense blob does not subdivide by density => returned as-is
        # (ADR-0001: an unsplittable blob is a finding, not a forced cut).
        df = _one_blob(60)
        groups = recursive_density_split(df, list(range(60)), max_size=40)
        self.assertEqual(len(groups), 1)
        self.assertEqual(sorted(groups[0]), list(range(60)))

    def test_density_subclusters_covers_all_points(self):
        df = _two_blobs(30)
        subs = density_subclusters(df, list(range(60)))
        self.assertEqual(sorted(p for s in subs for p in s), list(range(60)))

    def test_capped_partitioner_is_registered(self):
        self.assertIn("capped_hdbscan", partitioner_names())

    def test_capped_partitioner_splits_oversized_clusters(self):
        df = _two_blobs(30)
        partitions = get_partitioner("capped_hdbscan")(
            df, (0.05,), max_cluster_size=40, min_samples=5
        )
        self.assertTrue(partitions)
        for partition in partitions:
            self.assertIsInstance(partition, Partition)
            for region in partition.regions:
                self.assertLessEqual(len(region["points"]), 40)


if __name__ == "__main__":
    unittest.main()
