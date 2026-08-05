import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from clustering.base import Partition
from descriptives import (
    cluster_card_data,
    cluster_frame,
    compare_configs,
    dataset_balance,
    dispersion_summary,
    expected_sigma_ratio,
    organic_local_z_deltas,
    partition_profile,
    peer_rate,
    standardized_residuals,
    subcluster_frame,
)
from metrics.group_fairness import calculate_meanvar


def _df(coords: list[tuple[float, float]]) -> pd.DataFrame:
    return pd.DataFrame({"lat": [lat for lat, _ in coords], "lon": [lon for _, lon in coords]})


def _partition(regions: list[list[int]], n_points: int, noise: list[int] | None = None) -> Partition:
    labels = np.full(n_points, -1, dtype=int)
    for label, points in enumerate(regions):
        labels[points] = label
    return Partition(
        method="fake",
        params={"min_cluster_frac": 0.01},
        labels=labels,
        regions=[
            {"points": points, "cluster_label": label} for label, points in enumerate(regions)
        ],
        noise_points=noise or [],
    )


class ClusterFrameTests(unittest.TestCase):
    def test_counts_positives_and_negatives_per_cluster(self):
        # Cluster 0: 3 points, 2 positive => 1 negative, rho 2/3.
        # Cluster 1: 2 points, 0 positive => 2 negative, rho 0.
        types = np.array([1, 1, 0, 0, 0])
        df = _df([(0.0, 0.0)] * 5)
        frame = cluster_frame(df, _partition([[0, 1, 2], [3, 4]], 5), types)

        self.assertEqual(list(frame["cluster_label"]), [0, 1])
        self.assertEqual(list(frame["n"]), [3, 2])
        self.assertEqual(list(frame["p"]), [2, 0])
        self.assertEqual(list(frame["n_neg"]), [1, 2])
        np.testing.assert_allclose(frame["rho"], [2 / 3, 0.0])

    def test_negatives_and_positives_always_sum_to_n(self):
        types = np.array([1, 0, 1, 1, 0, 0])
        df = _df([(0.0, 0.0)] * 6)
        frame = cluster_frame(df, _partition([[0, 1, 2], [3, 4, 5]], 6), types)
        np.testing.assert_array_equal(frame["p"] + frame["n_neg"], frame["n"])

    def test_compactness_is_haversine_distance_to_centroid_in_km(self):
        # Two points 0.1 degrees of latitude apart: the centroid sits halfway,
        # so every radius is 0.05 deg = 0.05 * 111.195 km (mean == p95 here).
        types = np.array([1, 0])
        df = _df([(0.0, 0.0), (0.1, 0.0)])
        frame = cluster_frame(df, _partition([[0, 1]], 2), types)
        expected_km = math.radians(0.05) * 6371.0088
        self.assertAlmostEqual(frame["raio_medio_km"].iloc[0], expected_km, places=3)
        self.assertAlmostEqual(frame["raio_p95_km"].iloc[0], expected_km, places=3)

    def test_compactness_of_a_single_point_cluster_is_zero(self):
        types = np.array([1])
        frame = cluster_frame(_df([(10.0, 20.0)]), _partition([[0]], 1), types)
        self.assertEqual(frame["raio_medio_km"].iloc[0], 0.0)

    def test_longitude_spread_shrinks_with_latitude(self):
        # The same 1-degree longitude gap is a shorter real distance far from
        # the equator: this is why compactness is km, never degrees.
        types = np.array([1, 0])
        equator = cluster_frame(_df([(0.0, 0.0), (0.0, 1.0)]), _partition([[0, 1]], 2), types)
        north = cluster_frame(_df([(60.0, 0.0), (60.0, 1.0)]), _partition([[0, 1]], 2), types)
        self.assertLess(north["raio_medio_km"].iloc[0], equator["raio_medio_km"].iloc[0])

    def test_cluster_origin_is_preserved_for_coverage_and_compactness_reports(self):
        types = np.array([1, 0, 1, 0])
        df = _df([(0.0, 0.0), (0.1, 0.0), (1.0, 1.0), (1.1, 1.0)])
        partition = _partition([[0, 1], [2, 3]], 4)
        partition.regions[0]["origin"] = "organic"
        partition.regions[1]["origin"] = "rescue"

        frame = cluster_frame(df, partition, types)

        self.assertEqual(list(frame["origin"]), ["organic", "rescue"])
        self.assertTrue(frame["raio_medio_km"].notna().all())


class DispersionTests(unittest.TestCase):
    def _frame(self) -> pd.DataFrame:
        types = np.array([1, 1, 0, 0, 1, 0, 1, 1, 1, 0])
        df = _df([(0.0, 0.0)] * 10)
        return cluster_frame(df, _partition([[0, 1, 2, 3], [4, 5], [6, 7, 8, 9]], 10), types)

    def test_summary_reports_mean_std_var_cv_min_max_per_variable(self):
        frame = self._frame()
        summary = dispersion_summary(frame)

        # n = [4, 2, 4]: mean 10/3, sample std of [4,2,4] = 1.1547.
        self.assertAlmostEqual(summary.loc["n", "mean"], 10 / 3)
        self.assertAlmostEqual(summary.loc["n", "std"], float(np.std([4, 2, 4], ddof=1)))
        self.assertAlmostEqual(summary.loc["n", "var"], summary.loc["n", "std"] ** 2)
        self.assertAlmostEqual(summary.loc["n", "cv"], summary.loc["n", "std"] / (10 / 3))
        self.assertEqual(summary.loc["n", "min"], 2)
        self.assertEqual(summary.loc["n", "max"], 4)

    def test_summary_covers_counts_rates_and_compactness(self):
        summary = dispersion_summary(self._frame())
        for variable in ("n", "p", "n_neg", "rho", "raio_medio_km"):
            self.assertIn(variable, summary.index)

    def test_rate_dispersion_is_the_square_root_of_meanvar(self):
        # sigma(rho) and MeanVar are the same number at two names; they differ
        # only by ddof (pandas uses n-1, MeanVar the population form).
        frame = self._frame()
        summary = dispersion_summary(frame)
        n_clusters = len(frame)
        population_sigma = summary.loc["rho", "std"] * math.sqrt((n_clusters - 1) / n_clusters)
        self.assertAlmostEqual(population_sigma, math.sqrt(calculate_meanvar(frame["rho"])))

    def test_compare_configs_puts_one_column_per_configuration(self):
        frame = self._frame()
        table = compare_configs({"hdbscan": frame, "capped": frame})
        self.assertEqual(list(table.columns), ["hdbscan", "capped"])
        self.assertIn(("n", "cv"), table.index)
        self.assertAlmostEqual(table.loc[("n", "mean"), "hdbscan"], 10 / 3)


class ReadingHelpersTests(unittest.TestCase):
    def test_dataset_balance_splits_positives_and_negatives(self):
        balance = dataset_balance(np.array([1, 1, 1, 0]))
        self.assertEqual(balance["N"], 4)
        self.assertEqual(balance["P"], 3)
        self.assertEqual(balance["n_neg"], 1)
        self.assertAlmostEqual(balance["global_rate"], 0.75)

    def test_expected_sigma_ratio_is_the_odds_of_the_global_rate(self):
        # Under a single shared rate, sigma(p)/sigma(n_neg) is forced to
        # rho/(1-rho) -- so the raw comparison is arithmetic, not a finding.
        self.assertAlmostEqual(expected_sigma_ratio(0.5), 1.0)
        self.assertAlmostEqual(expected_sigma_ratio(0.8), 4.0)
        self.assertTrue(math.isnan(expected_sigma_ratio(1.0)))

    def test_standardized_residual_is_zero_when_the_cluster_matches_global(self):
        types = np.array([1, 0, 1, 0])
        df = _df([(0.0, 0.0)] * 4)
        frame = cluster_frame(df, _partition([[0, 1], [2, 3]], 4), types)
        residuals = standardized_residuals(frame, global_rate=0.5)
        np.testing.assert_allclose(residuals, [0.0, 0.0])

    def test_standardized_residual_signs_follow_the_deviation(self):
        # Cluster 0 all-positive, cluster 1 all-negative, global rate 0.5.
        types = np.array([1, 1, 0, 0])
        df = _df([(0.0, 0.0)] * 4)
        frame = cluster_frame(df, _partition([[0, 1], [2, 3]], 4), types)
        residuals = standardized_residuals(frame, global_rate=0.5)
        self.assertGreater(residuals[0], 0)
        self.assertLess(residuals[1], 0)
        # (2 - 0.5*2) / sqrt(2*0.25) = 1 / 0.7071
        self.assertAlmostEqual(residuals[0], 1.0 / math.sqrt(0.5))

    def test_partition_profile_reports_regions_and_unassigned_share(self):
        partition = _partition([[0, 1], [2, 3]], 6, noise=[4, 5])
        profile = partition_profile(partition, n_total=6)
        self.assertEqual(profile["n_regions"], 2)
        self.assertEqual(profile["noise_n"], 2)
        self.assertAlmostEqual(profile["noise_rate"], 2 / 6)
        self.assertEqual(profile["forced_uncapped"], 0)
        self.assertEqual(profile["over_cap"], 0)

    def test_partition_profile_decomposes_organic_rescue_and_unassigned_coverage(self):
        partition = _partition([[0, 1], [2, 3, 4]], 6, noise=[5])
        partition.regions[0]["origin"] = "organic"
        partition.regions[1]["origin"] = "rescue"

        profile = partition_profile(partition, n_total=6)

        self.assertEqual(profile["organic_n"], 2)
        self.assertEqual(profile["rescue_n"], 3)
        self.assertEqual(profile["noise_n"], 1)
        self.assertAlmostEqual(
            profile["organic_rate"] + profile["rescue_rate"] + profile["noise_rate"],
            1.0,
        )

    def test_partition_profile_counts_forced_splits(self):
        partition = _partition([[0, 1], [2, 3]], 4)
        partition.regions[0]["forced_uncapped"] = True
        self.assertEqual(partition_profile(partition, n_total=4)["forced_uncapped"], 1)

    def test_partition_profile_counts_oversized_pieces_separately(self):
        # A recursion can split a cluster and still leave one piece over the cap:
        # that piece is not a "forced" split, but it must still be reported.
        partition = _partition([[0, 1], [2, 3]], 4)
        partition.regions[1]["over_cap"] = True
        profile = partition_profile(partition, n_total=4)
        self.assertEqual(profile["over_cap"], 1)
        self.assertEqual(profile["forced_uncapped"], 0)

    def test_organic_local_z_delta_is_zero_when_no_rescue_cluster_was_added(self):
        coords = [
            (0.0, 0.0), (0.01, 0.0),
            (0.0, 1.0), (0.01, 1.0),
            (1.0, 0.0), (1.01, 0.0),
            (1.0, 1.0), (1.01, 1.0),
        ]
        types = np.array([1, 0] * 4)
        partition = _partition([[0, 1], [2, 3], [4, 5], [6, 7]], 8)
        for region in partition.regions:
            region["origin"] = "organic"
            region["origin_cluster_label"] = region["cluster_label"]

        deltas = organic_local_z_deltas(
            partition,
            _df(coords),
            types,
            n_total=8,
            p_total=4,
        )

        self.assertEqual(len(deltas), 4)
        np.testing.assert_allclose(deltas["local_z_delta"], 0.0)


class ClusterCardTests(unittest.TestCase):
    def test_subcluster_frame_reports_one_row_per_subcluster(self):
        types = np.array([1, 1, 0, 0])
        frame = subcluster_frame([0, 1, 2, 3], types, lambda points: [[0, 1], [2, 3]])
        self.assertEqual(list(frame["n"]), [2, 2])
        np.testing.assert_allclose(frame["rho"], [1.0, 0.0])

    def test_peer_rate_pools_neighbours_by_size(self):
        # Peers of cluster 0 are clusters 1 and 2: pooled (1+2)/(2+4) = 0.5.
        types = np.array([1, 1, 1, 0, 1, 1, 0, 0])
        df = _df([(0.0, 0.0)] * 8)
        frame = cluster_frame(df, _partition([[0, 1], [2, 3], [4, 5, 6, 7]], 8), types)
        self.assertAlmostEqual(peer_rate(frame, {0: [1, 2]}, 0), 0.5)

    def test_peer_rate_is_nan_without_enough_peers(self):
        types = np.array([1, 0])
        df = _df([(0.0, 0.0)] * 2)
        frame = cluster_frame(df, _partition([[0, 1]], 2), types)
        self.assertTrue(math.isnan(peer_rate(frame, {0: []}, 0)))

    def test_card_carries_the_three_reference_rates_and_the_subclusters(self):
        types = np.array([1, 1, 0, 0, 1, 1, 1, 1])
        df = _df([(0.0, 0.0)] * 8)
        partition = _partition([[0, 1, 2, 3], [4, 5], [6, 7]], 8)
        card = cluster_card_data(
            df,
            partition,
            types,
            cluster_label=0,
            splitter=lambda points: [points[:2], points[2:]],
            adjacency={0: [1, 2]},
            global_rate=0.75,
        )

        self.assertEqual(card["cluster_label"], 0)
        self.assertEqual(card["n"], 4)
        self.assertAlmostEqual(card["rho_in"], 0.5)
        self.assertAlmostEqual(card["rho_peer"], 1.0)   # clusters 1 and 2 are all-positive
        self.assertAlmostEqual(card["rho_global"], 0.75)
        self.assertAlmostEqual(card["gini_subcluster"], 0.5)  # rates 1.0 and 0.0
        self.assertEqual(list(card["subclusters"]["rho"]), [1.0, 0.0])
        self.assertFalse(card["homogeneous"])

    def test_card_flags_a_cluster_that_does_not_subdivide(self):
        # An unsplittable cluster is a finding ("homogeneous inside"), not a
        # broken chart: one subcluster, Gini 0, flagged.
        types = np.array([1, 0, 1, 0])
        df = _df([(0.0, 0.0)] * 4)
        partition = _partition([[0, 1], [2, 3]], 4)
        card = cluster_card_data(
            df,
            partition,
            types,
            cluster_label=0,
            splitter=lambda points: [list(points)],
            adjacency={0: [1]},
            global_rate=0.5,
        )
        self.assertTrue(card["homogeneous"])
        self.assertEqual(card["gini_subcluster"], 0.0)
        self.assertEqual(len(card["subclusters"]), 1)

    def test_card_rejects_an_unknown_cluster_label(self):
        types = np.array([1, 0])
        df = _df([(0.0, 0.0)] * 2)
        with self.assertRaises(ValueError):
            cluster_card_data(
                df,
                _partition([[0, 1]], 2),
                types,
                cluster_label=99,
                splitter=lambda points: [list(points)],
                adjacency={},
                global_rate=0.5,
            )


if __name__ == "__main__":
    unittest.main()
