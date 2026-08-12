import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from clustering.registry import get_partitioner, partitioner_names
from clustering.stat_cap import statistical_cap_directive


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


def _homogeneous_large_and_three_regular_clusters(seed: int = 22) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = [pd.DataFrame({"lat": np.zeros(200), "lon": np.zeros(200)})]
    for lat, lon in ((3.0, 3.0), (6.0, 0.0), (0.0, 6.0)):
        frames.append(
            pd.DataFrame(
                {
                    "lat": rng.normal(lat, 0.01, 50),
                    "lon": rng.normal(lon, 0.01, 50),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


class StatisticalCapDirectiveTests(unittest.TestCase):
    def test_one_sigma_leave_one_out_targets_the_size_tail(self):
        # For 40, the other sizes are 10/10/10: mean=10 and sample sigma=0.
        # Each 10 sees 10/10/40: mean=20 and sample sigma≈17.32, so it stays.
        self.assertEqual(
            statistical_cap_directive([10, 10, 10, 40], sigma_multiplier=1.0),
            {3: 10.0},
        )

    def test_lar_one_sigma_targets_five_clusters_including_the_two_largest(self):
        # Sizes from the serious organic LAR run, ordered by original label.
        sizes = [
            3435, 6430, 2404, 2064, 5526, 1059, 2865, 1878, 20409, 1456,
            1105, 2063, 17633, 1629, 2320, 4526, 2067, 1624, 1432, 7503,
            4077, 7350, 1726, 1517, 1632, 1891, 4253, 3615, 1076, 1257,
            1379, 1075, 4444, 1182, 10727, 9290, 1442, 2940, 1263, 2346,
            8334,
        ]

        directives = statistical_cap_directive(sizes, sigma_multiplier=1.0)

        self.assertEqual(set(directives), {8, 12, 34, 35, 40})
        self.assertEqual(directives[8], 3545.875)


class OrganicStatisticalCapPartitionerTests(unittest.TestCase):
    def test_registered_partitioner_targets_the_tail_and_preserves_organic_coverage(self):
        self.assertIn("hdbscan_stat_cap", partitioner_names())
        df = _one_large_and_three_regular_clusters()
        frac = 25 / len(df)

        organic = get_partitioner("hdbscan")(
            df,
            (frac,),
            min_samples=5,
            min_cluster_size_min=25,
        )[0]
        refined = get_partitioner("hdbscan_stat_cap")(
            df,
            (frac,),
            min_samples=5,
            min_cluster_size_min=25,
        )[0]

        self.assertEqual(refined.method, "hdbscan_stat_cap")
        self.assertEqual(refined.noise_points, organic.noise_points)
        self.assertEqual(refined.params["stat_cap_sigma"], 1.0)
        self.assertEqual(refined.params["stat_cap_targets"], 1)
        large_parent = next(
            region["cluster_label"]
            for region in organic.regions
            if len(region["points"]) == 200
        )
        self.assertEqual(refined.params["stat_cap_directives"], {large_parent: 50})
        targeted = [r for r in refined.regions if "stat_cap_target" in r]
        self.assertTrue(targeted)
        self.assertTrue(all(r["stat_cap_target"] == 50 for r in targeted))
        self.assertTrue(all(r["origin"] == "organic" for r in refined.regions))

    def test_leaf_refinement_keeps_only_density_leaves_without_nearest_assignment(self):
        self.assertIn("hdbscan_stat_leaf", partitioner_names())
        df = _one_large_and_three_regular_clusters()
        frac = 25 / len(df)

        organic = get_partitioner("hdbscan")(
            df,
            (frac,),
            min_samples=5,
            min_cluster_size_min=25,
        )[0]
        refined = get_partitioner("hdbscan_stat_leaf")(
            df,
            (frac,),
            min_samples=5,
            min_cluster_size_min=25,
        )[0]

        self.assertEqual(refined.method, "hdbscan_stat_leaf")
        self.assertEqual(refined.params["stat_cap_targets"], 1)
        self.assertEqual(refined.params["stat_leaf_split_parents"], 1)
        self.assertEqual(refined.params["stat_leaf_refusals"], 0)
        self.assertEqual(refined.params["refinement_cluster_selection_method"], "leaf")
        self.assertGreater(len(refined.noise_points), len(organic.noise_points))
        self.assertEqual(
            refined.params["stat_leaf_noise_n"],
            len(refined.noise_points) - len(organic.noise_points),
        )
        targeted = [r for r in refined.regions if "stat_cap_target" in r]
        self.assertEqual(sorted(len(r["points"]) for r in targeted), [28, 81])
        self.assertTrue(all(r["split_mode"] == "density_leaf" for r in targeted))
        assigned = {point for region in refined.regions for point in region["points"]}
        self.assertTrue(assigned.isdisjoint(refined.noise_points))
        self.assertEqual(len(assigned) + len(refined.noise_points), len(df))

    def test_leaf_refinement_preserves_parent_when_density_has_fewer_than_two_leaves(self):
        df = _homogeneous_large_and_three_regular_clusters()
        frac = 25 / len(df)

        organic = get_partitioner("hdbscan")(
            df,
            (frac,),
            min_samples=5,
            min_cluster_size_min=25,
        )[0]
        refined = get_partitioner("hdbscan_stat_leaf")(
            df,
            (frac,),
            min_samples=5,
            min_cluster_size_min=25,
        )[0]

        self.assertEqual(refined.noise_points, organic.noise_points)
        self.assertEqual(refined.params["stat_leaf_split_parents"], 0)
        self.assertEqual(refined.params["stat_leaf_refusals"], 1)
        targeted = [r for r in refined.regions if "stat_cap_target" in r]
        self.assertEqual(len(targeted), 1)
        self.assertEqual(len(targeted[0]["points"]), 200)
        self.assertTrue(targeted[0]["stat_leaf_refused"])


if __name__ == "__main__":
    unittest.main()
