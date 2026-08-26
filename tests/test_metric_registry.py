import sys
import unittest
from pathlib import Path

import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from clustering.base import Partition
from clustering.internal import InternalSubdivision
from metrics.base import MetricContext, MetricResult
from metrics.registry import (
    METRICS,
    candidate_metric_names,
    evaluate_primary,
    get_metric,
    get_metric_definition,
    get_primary_capabilities,
    metric_names,
    primary_metric_names,
)


def _tiny_partition() -> Partition:
    return Partition(
        method="fake",
        params={},
        labels=np.array([0, 0, 0, 1, 1]),
        regions=[
            {"points": [0, 1, 2], "cluster_label": 0},
            {"points": [3, 4], "cluster_label": 1},
        ],
    )


def _context(types: np.ndarray) -> MetricContext:
    return MetricContext(
        n_total=len(types),
        p_total=int(types.sum()),
        adjacency={0: [1], 1: [0]},
        rng=np.random.default_rng(0),
        internal_subdivider=lambda points: InternalSubdivision(
            [list(points)], [], len(points), len(points)
        ),
    )


class MetricRegistryTests(unittest.TestCase):
    def test_candidate_capabilities_have_one_registry_source(self):
        self.assertEqual(
            set(candidate_metric_names()),
            {"peer_rate_difference", "peer_log_rate_ratio", "peer_gini_gap"},
        )
        rate = get_metric_definition("peer_rate_difference")
        self.assertEqual(rate.needs, frozenset({"neighbors"}))
        self.assertTrue(rate.outcome_direction)
        self.assertTrue(rate.confirmatory_candidate)
        gini = get_metric_definition("peer_gini_gap")
        self.assertEqual(gini.needs, frozenset({"neighbors", "subclusters"}))
        self.assertFalse(gini.outcome_direction)
        self.assertFalse(gini.confirmatory_candidate)

    def test_registry_needs_match_every_metric_result_contract(self):
        partition = _tiny_partition()
        types = np.array([1, 0, 1, 0, 1])
        ctx = _context(types)
        for name in metric_names():
            with self.subTest(metric=name):
                result = get_metric(name)(partition, types, ctx)
                self.assertEqual(get_metric_definition(name).needs, result.needs)

    def test_unknown_metric_raises(self):
        with self.assertRaises(ValueError):
            get_metric("nope")

    def test_sul_metric_matches_hand_computed_value_and_flags(self):
        # 10 points: region A all-positive, region B all-negative, global rate 0.5.
        # Perfectly separated => SUL = 10*ln(2) for each region (hand-computed).
        types = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        partition = Partition(
            method="fake",
            params={},
            labels=np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
            regions=[
                {"points": [0, 1, 2, 3, 4], "cluster_label": 0},
                {"points": [5, 6, 7, 8, 9], "cluster_label": 1},
            ],
        )
        result = get_metric("sul")(partition, types, _context(types))
        np.testing.assert_allclose(result.per_cluster, [10 * np.log(2), 10 * np.log(2)])
        self.assertIsNone(result.partition_scalar)
        self.assertTrue(result.supports_mc)
        self.assertFalse(result.signed)

    def test_gini_metric_exposes_scalar_and_contributions(self):
        # Region A all-positive (rho 1.0), region B all-negative (rho 0.0).
        # Gini([0,1]) = 0.5 and contributions = [0.5, 0.5] (from test_metrics).
        types = np.array([1, 1, 0, 0])
        partition = Partition(
            method="fake",
            params={},
            labels=np.array([0, 0, 1, 1]),
            regions=[
                {"points": [0, 1], "cluster_label": 0},
                {"points": [2, 3], "cluster_label": 1},
            ],
        )
        result = get_metric("gini")(partition, types, _context(types))
        self.assertAlmostEqual(result.partition_scalar, 0.5)
        np.testing.assert_allclose(result.per_cluster, [0.5, 0.5])
        self.assertFalse(result.supports_mc)
        self.assertTrue(result.signed)

    def test_local_z_matches_hand_computed_two_proportion_z(self):
        # 3 clusters of n=100, all mutually adjacent (triangle).
        # c0: 30/100 (rho .30); peers c1,c2 pool to 100/200 (rho .50) => z=-3.2956.
        # c1: 50/100; peers c0,c2 pool to 80/200 (rho .40) => z=+1.6477. Same for c2.
        # se = sqrt(.4333*.5667*(1/100+1/200)) = .0606905; z0 = -.2/se, z1 = .1/se.
        types = np.zeros(300, dtype=int)
        types[0:30] = 1     # cluster 0: 30 positives
        types[100:150] = 1  # cluster 1: 50 positives
        types[200:250] = 1  # cluster 2: 50 positives
        partition = Partition(
            method="fake",
            params={},
            labels=np.zeros(300, dtype=int),
            regions=[
                {"points": list(range(0, 100)), "cluster_label": 0},
                {"points": list(range(100, 200)), "cluster_label": 1},
                {"points": list(range(200, 300)), "cluster_label": 2},
            ],
        )
        ctx = MetricContext(
            n_total=300,
            p_total=130,
            adjacency={0: [1, 2], 1: [0, 2], 2: [0, 1]},
            rng=np.random.default_rng(0),
            internal_subdivider=lambda points: InternalSubdivision(
                [list(points)], [], len(points), len(points)
            ),
        )
        result = get_metric("local_z")(partition, types, ctx)
        np.testing.assert_allclose(
            result.per_cluster, [-3.295410, 1.647705, 1.647705], atol=1e-4
        )
        self.assertTrue(result.signed)
        self.assertTrue(result.supports_mc)
        self.assertIsNone(result.partition_scalar)
        self.assertEqual(result.needs, frozenset({"neighbors"}))

    def test_local_z_is_nan_for_clusters_with_fewer_than_two_peers(self):
        types = np.array([1, 0, 1, 0])
        partition = Partition(
            method="fake",
            params={},
            labels=np.array([0, 0, 1, 1]),
            regions=[
                {"points": [0, 1], "cluster_label": 0},
                {"points": [2, 3], "cluster_label": 1},
            ],
        )
        # Two clusters => each has a single peer (< 2) => NaN.
        ctx = MetricContext(
            n_total=4, p_total=2, adjacency={0: [1], 1: [0]},
            rng=np.random.default_rng(0),
            internal_subdivider=lambda points: InternalSubdivision(
                [list(points)], [], len(points), len(points)
            ),
        )
        result = get_metric("local_z")(partition, types, ctx)
        self.assertTrue(np.all(np.isnan(result.per_cluster)))

    def test_peer_rate_candidates_keep_effect_size_separate_from_standardized_evidence(self):
        # Same worked example as local-z: cluster rates .30, .50 and .50.
        # The difference candidate reports percentage-rate effect in native units;
        # the ratio candidate uses a continuity-corrected log ratio centred at 0.
        types = np.zeros(300, dtype=int)
        types[0:30] = 1
        types[100:150] = 1
        types[200:250] = 1
        partition = Partition(
            method="fake",
            params={},
            labels=np.repeat(np.arange(3), 100),
            regions=[
                {"points": list(range(0, 100)), "cluster_label": 0},
                {"points": list(range(100, 200)), "cluster_label": 1},
                {"points": list(range(200, 300)), "cluster_label": 2},
            ],
        )
        ctx = MetricContext(
            n_total=300,
            p_total=130,
            adjacency={0: [1, 2], 1: [0, 2], 2: [0, 1]},
        )

        difference = get_metric("peer_rate_difference")(partition, types, ctx)
        np.testing.assert_allclose(difference.per_cluster, [-0.20, 0.10, 0.10])
        self.assertTrue(difference.signed)
        self.assertTrue(difference.supports_mc)
        self.assertFalse(difference.standardized)

        ratio = get_metric("peer_log_rate_ratio")(partition, types, ctx)
        expected = [
            np.log((30.5 / 101) / (100.5 / 201)),
            np.log((50.5 / 101) / (80.5 / 201)),
            np.log((50.5 / 101) / (80.5 / 201)),
        ]
        np.testing.assert_allclose(ratio.per_cluster, expected)
        self.assertTrue(ratio.signed)
        self.assertTrue(ratio.supports_mc)
        self.assertFalse(ratio.standardized)

    def test_peer_gini_gap_compares_internal_gini_with_weighted_peer_ginis(self):
        # c0 has internal subcluster rates [0, 1] => Gini .5.
        # c1 and c2 have [0, 0] and [1, 1] => Gini 0.
        # In a triangle of equally sized clusters: gaps [.5, -.25, -.25].
        types = np.array([
            0, 0, 1, 1,
            0, 0, 0, 0,
            1, 1, 1, 1,
        ])
        partition = Partition(
            method="fake",
            params={},
            labels=np.repeat(np.arange(3), 4),
            regions=[
                {"points": [0, 1, 2, 3], "cluster_label": 0},
                {"points": [4, 5, 6, 7], "cluster_label": 1},
                {"points": [8, 9, 10, 11], "cluster_label": 2},
            ],
        )
        ctx = MetricContext(
            n_total=12,
            p_total=6,
            adjacency={0: [1, 2], 1: [0, 2], 2: [0, 1]},
            internal_subdivider=lambda points: InternalSubdivision(
                [points[:2], points[2:]], [], 2, len(points)
            ),
        )

        result = get_metric("peer_gini_gap")(partition, types, ctx)
        np.testing.assert_allclose(result.per_cluster, [0.5, -0.25, -0.25])
        np.testing.assert_allclose(
            result.per_cluster_metadata["internal_gini"], [0.5, 0.0, 0.0]
        )
        np.testing.assert_allclose(
            result.per_cluster_metadata["peer_gini"], [0.0, 0.25, 0.25]
        )
        self.assertTrue(result.signed)
        self.assertTrue(result.supports_mc)
        self.assertFalse(result.standardized)
        self.assertEqual(result.needs, frozenset({"neighbors", "subclusters"}))

    def test_local_z_is_the_only_standardized_metric(self):
        # The analytic (Sidak) threshold only applies to metrics expressed in
        # standard-error units, so exactly one metric may claim it.
        types = np.array([1, 0, 1, 0, 1])
        partition = _tiny_partition()
        standardized = {
            name
            for name in metric_names()
            if get_metric(name)(partition, types, _context(types)).standardized
        }
        self.assertEqual(standardized, {"local_z"})

    def test_only_local_z_and_sul_satisfy_the_primary_contract(self):
        self.assertEqual(set(primary_metric_names()), {"local_z", "sul"})
        self.assertEqual(get_primary_capabilities("local_z").rate_reference, "peers")
        self.assertEqual(get_primary_capabilities("sul").rate_reference, "outside")
        for name in (
            "gini", "gini_subcluster", "meanvar", "dp_difference", "dp_ratio",
            "peer_rate_difference", "peer_log_rate_ratio", "peer_gini_gap",
        ):
            with self.subTest(metric=name), self.assertRaises(ValueError):
                get_primary_capabilities(name)

    def test_primary_direction_uses_the_declared_statistical_contrast(self):
        local = evaluate_primary(
            "local_z", score=-3.0, threshold=2.0, rho_in=0.4, rho_reference=0.5
        )
        sul = evaluate_primary(
            "sul", score=9.0, threshold=5.0, rho_in=0.4, rho_reference=0.6
        )
        self.assertEqual(local.direction, "negative")
        self.assertEqual(sul.direction, "negative")
        self.assertEqual(local.detection_class, "negative")
        self.assertEqual(sul.detection_class, "negative")

    def test_non_evaluated_is_distinct_from_nothing_detected(self):
        no_score = evaluate_primary(
            "local_z", score=float("nan"), threshold=2.0, rho_in=0.5, rho_reference=0.5
        )
        no_threshold = evaluate_primary(
            "local_z", score=1.0, threshold=None, rho_in=0.5, rho_reference=0.5
        )
        nothing = evaluate_primary(
            "local_z", score=1.0, threshold=2.0, rho_in=0.5, rho_reference=0.4
        )
        self.assertEqual(no_score.evaluation_status, "não avaliado")
        self.assertEqual(no_score.evaluation_reason, "score_nao_finito")
        self.assertIsNone(no_score.detection_class)
        self.assertEqual(no_threshold.evaluation_reason, "limiar_ausente_ou_invalido")
        self.assertEqual(nothing.evaluation_status, "avaliado")
        self.assertEqual(nothing.detection_class, "neutral")

    def test_gini_subcluster_is_gini_across_subcluster_rates(self):
        # One cluster split into two subclusters with rates 0.0 and 1.0 => Gini 0.5.
        types = np.array([0, 0, 1, 1])
        partition = Partition(
            method="fake",
            params={},
            labels=np.zeros(4, dtype=int),
            regions=[{"points": [0, 1, 2, 3], "cluster_label": 0}],
        )
        ctx = MetricContext(
            n_total=4,
            p_total=2,
            adjacency={0: []},
            rng=np.random.default_rng(0),
            internal_subdivider=lambda points: InternalSubdivision(
                [[0, 1], [2, 3]], [], 2, 4
            ),
        )
        result = get_metric("gini_subcluster")(partition, types, ctx)
        np.testing.assert_allclose(result.per_cluster, [0.5])
        self.assertFalse(result.supports_mc)
        self.assertFalse(result.signed)
        self.assertEqual(result.needs, frozenset({"subclusters"}))

    def test_dp_difference_is_max_minus_min_selection_rate(self):
        # Three clusters with rates 0.50, 0.25, 0.75 => difference 0.50,
        # ratio 0.25/0.75 = 1/3 (verified against fairlearn directly).
        types = np.array([1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0])
        partition = Partition(
            method="fake",
            params={},
            labels=np.array([0] * 4 + [1] * 4 + [2] * 4),
            regions=[
                {"points": [0, 1, 2, 3], "cluster_label": 0},
                {"points": [4, 5, 6, 7], "cluster_label": 1},
                {"points": [8, 9, 10, 11], "cluster_label": 2},
            ],
        )
        ctx = _context(types)

        difference = get_metric("dp_difference")(partition, types, ctx)
        self.assertAlmostEqual(difference.partition_scalar, 0.5)
        np.testing.assert_allclose(difference.per_cluster, [0.5, 0.25, 0.75])
        self.assertFalse(difference.supports_mc)
        self.assertFalse(difference.standardized)

        ratio = get_metric("dp_ratio")(partition, types, ctx)
        self.assertAlmostEqual(ratio.partition_scalar, 1 / 3)

    def test_dp_ignores_points_outside_every_cluster(self):
        # Points 4..7 belong to no region (label -1). Including them as a group
        # would change the parity spread; they must be invisible to the metric.
        types = np.array([1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0])
        assigned = Partition(
            method="fake",
            params={},
            labels=np.array([0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 1, 1]),
            regions=[
                {"points": [0, 1, 2, 3], "cluster_label": 0},
                {"points": [8, 9, 10, 11], "cluster_label": 1},
            ],
        )
        result = get_metric("dp_difference")(assigned, types, _context(types))
        # rates 0.5 and 0.25 => difference 0.25; the all-positive unassigned
        # block (rate 1.0) would have pushed it to 0.75.
        np.testing.assert_allclose(result.per_cluster, [0.5, 0.25])
        self.assertAlmostEqual(result.partition_scalar, 0.25)

    def test_dp_is_nan_for_a_single_cluster(self):
        types = np.array([1, 0])
        partition = Partition(
            method="fake",
            params={},
            labels=np.zeros(2, dtype=int),
            regions=[{"points": [0, 1], "cluster_label": 0}],
        )
        result = get_metric("dp_difference")(partition, types, _context(types))
        self.assertTrue(np.isnan(result.partition_scalar))

    def test_plugging_a_new_metric_works(self):
        def fake_metric(partition, types, ctx) -> MetricResult:
            per_cluster = np.array([len(r["points"]) for r in partition.regions], dtype=float)
            return MetricResult(
                per_cluster=per_cluster,
                partition_scalar=float(per_cluster.sum()),
                signed=False,
                supports_mc=False,
                needs=frozenset(),
            )

        METRICS["fake_size"] = fake_metric
        try:
            self.assertIn("fake_size", metric_names())
            partition = _tiny_partition()
            types = np.array([1, 0, 1, 0, 1])
            result = get_metric("fake_size")(partition, types, _context(types))
            np.testing.assert_array_equal(result.per_cluster, [3.0, 2.0])
            self.assertEqual(result.partition_scalar, 5.0)
            self.assertEqual(len(result.per_cluster), len(partition.regions))
        finally:
            METRICS.pop("fake_size")


if __name__ == "__main__":
    unittest.main()
