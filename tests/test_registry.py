import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from clustering.base import Partition
from clustering.registry import PARTITIONERS, get_partitioner, partitioner_names


class RegistryTests(unittest.TestCase):
    def test_hdbscan_is_registered(self):
        self.assertIn("hdbscan", partitioner_names())

    def test_unknown_partitioner_raises(self):
        with self.assertRaises(ValueError):
            get_partitioner("nope")

    def test_registered_hdbscan_returns_partitions(self):
        df = pd.DataFrame(
            {
                "lat": [0.0, 0.0001, 0.0002, 1.0, 1.0001, 1.0002, 20.0],
                "lon": [0.0, 0.0001, 0.0002, 1.0, 1.0001, 1.0002, 20.0],
            }
        )

        partitions = get_partitioner("hdbscan")(df, (0.4,), 2)

        self.assertTrue(partitions)
        for partition in partitions:
            self.assertIsInstance(partition, Partition)
            self.assertEqual(partition.method, "hdbscan")
            self.assertIn("min_cluster_size", partition.params)
            for region in partition.regions:
                self.assertIn("points", region)

    def test_plugging_a_new_partitioner_works(self):
        def fake_partitioner(df, **_):
            return [
                Partition(
                    method="fake",
                    params={},
                    labels=np.zeros(len(df), dtype=int),
                    regions=[{"points": list(range(len(df)))}],
                )
            ]

        PARTITIONERS["fake"] = fake_partitioner
        try:
            partitions = get_partitioner("fake")(pd.DataFrame({"lat": [0.0, 1.0], "lon": [0.0, 1.0]}))
            self.assertEqual(partitions[0].method, "fake")
            self.assertEqual(partitions[0].regions[0]["points"], [0, 1])
            self.assertEqual(partitions[0].noise_n, 0)
        finally:
            PARTITIONERS.pop("fake")


if __name__ == "__main__":
    unittest.main()
