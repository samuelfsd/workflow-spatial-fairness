import sys
import unittest
from pathlib import Path

import pandas as pd
import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from clustering.base import Partition
from clustering.registry import get_partitioner, partitioner_names
from clustering.rescue import statistical_cap_directive


def _dense_blobs_with_sparse_background(seed: int = 18) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "lat": rng.normal(0.0, 0.005, 100),
                    "lon": rng.normal(0.0, 0.005, 100),
                }
            ),
            pd.DataFrame(
                {
                    "lat": rng.normal(1.0, 0.005, 100),
                    "lon": rng.normal(1.0, 0.005, 100),
                }
            ),
            pd.DataFrame(
                {
                    "lat": rng.uniform(-3.0, 4.0, 200),
                    "lon": rng.uniform(-3.0, 4.0, 200),
                }
            ),
        ],
        ignore_index=True,
    )


def _one_large_and_three_regular_clusters(seed: int = 22) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    specs = ((0.0, 0.0, 200), (3.0, 3.0, 50), (6.0, 0.0, 50), (0.0, 6.0, 50))
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "lat": rng.normal(lat, 0.01, size),
                    "lon": rng.normal(lon, 0.01, size),
                }
            )
            for lat, lon, size in specs
        ],
        ignore_index=True,
    )


class RescuePartitionerTests(unittest.TestCase):
    def test_rescue_partitioner_is_registered(self):
        self.assertIn("hdbscan_rescue", partitioner_names())

        df = pd.DataFrame(
            {
                "lat": [0.0, 0.0001, 0.0002, 1.0, 1.0001, 1.0002, 20.0],
                "lon": [0.0, 0.0001, 0.0002, 1.0, 1.0001, 1.0002, 20.0],
            }
        )
        partitions = get_partitioner("hdbscan_rescue")(
            df,
            (0.4,),
            rescue_min_samples=(2,),
            min_samples=2,
            min_cluster_size_min=2,
            stat_cap=False,
        )

        self.assertEqual(len(partitions), 1)
        self.assertIsInstance(partitions[0], Partition)
        self.assertEqual(partitions[0].method, "hdbscan_rescue")

    def test_second_pass_recovers_only_previously_unassigned_points(self):
        df = _dense_blobs_with_sparse_background()
        partition = get_partitioner("hdbscan_rescue")(
            df,
            (0.1,),  # 40 points: the same absolute floor in both passes.
            rescue_min_samples=(5,),
            min_samples=40,
            min_cluster_size_min=25,
            stat_cap=False,
        )[0]

        organic = [region for region in partition.regions if region["origin"] == "organic"]
        rescued = [region for region in partition.regions if region["origin"] == "rescue"]
        organic_points = {point for region in organic for point in region["points"]}
        rescued_points = {point for region in rescued for point in region["points"]}

        self.assertGreaterEqual(len(organic), 2)
        self.assertGreaterEqual(len(rescued), 1)
        self.assertTrue(organic_points.isdisjoint(rescued_points))
        self.assertTrue(rescued_points.isdisjoint(partition.noise_points))
        self.assertEqual(len(organic_points | rescued_points) + partition.noise_n, len(df))
        self.assertTrue(all(region["min_cluster_size"] == 40 for region in rescued))


class StatisticalCapTests(unittest.TestCase):
    def test_only_a_cluster_beyond_the_leave_one_out_mean_plus_sigma_is_targeted(self):
        # For 40, the other sizes are 10/10/10: mean=10, sample sigma=0,
        # so 40 triggers and its density-split target is the mean, 10.
        # Each 10 sees 10/10/40: mean=20 and sample sigma≈17.32, so it stays.
        self.assertEqual(statistical_cap_directive([10, 10, 10, 40]), {3: 10.0})

    def test_equal_cluster_sizes_do_not_trigger_the_statistical_cap(self):
        self.assertEqual(statistical_cap_directive([25, 25, 25, 25]), {})

    def test_partitioner_applies_one_statistical_density_split_and_records_refusals(self):
        df = _one_large_and_three_regular_clusters()
        partition = get_partitioner("hdbscan_rescue")(
            df,
            (25 / len(df),),
            rescue_min_samples=(5,),
            min_samples=5,
            min_cluster_size_min=25,
            stat_cap=True,
        )[0]

        targeted = [region for region in partition.regions if "stat_cap_target" in region]
        self.assertEqual(partition.params["stat_cap_targets"], 1)
        self.assertGreaterEqual(len(targeted), 1)
        self.assertTrue(all(region["stat_cap_target"] == 50 for region in targeted))
        self.assertTrue(all(region["origin"] == "organic" for region in targeted))
        self.assertTrue(any(region["forced_uncapped"] for region in targeted))

    def test_statistical_cap_can_be_disabled_to_isolate_the_rescue_effect(self):
        df = _one_large_and_three_regular_clusters()
        partition = get_partitioner("hdbscan_rescue")(
            df,
            (25 / len(df),),
            rescue_min_samples=(5,),
            min_samples=5,
            min_cluster_size_min=25,
            stat_cap=False,
        )[0]

        self.assertEqual(
            sorted(len(region["points"]) for region in partition.regions),
            [50, 50, 50, 200],
        )
        self.assertEqual(partition.params["stat_cap_targets"], 0)
        self.assertFalse(any("stat_cap_target" in region for region in partition.regions))


if __name__ == "__main__":
    unittest.main()
