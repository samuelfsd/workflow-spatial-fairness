import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from repeated_generator import generate_geography, generate_outcomes
from repeated_plan import RepeatedPlan, expand_plan
from repeated_statistics import block_fwer_gate, paired_recovery_contrast
from spatial_recovery import compare_method_detection_sets, evaluate_spatial_recovery


class RepeatedPlanTests(unittest.TestCase):
    def test_plan_requires_n_and_grid(self):
        with self.assertRaisesRegex(ValueError, "N"):
            RepeatedPlan(n_points=None, reference_grid=(10, 10)).validate()
        with self.assertRaisesRegex(ValueError, "grade"):
            RepeatedPlan(n_points=200, reference_grid=None).validate()
        with self.assertRaisesRegex(ValueError, "fonte"):
            RepeatedPlan(
                n_points=200,
                reference_grid=(10, 10),
                coordinate_source_dataset="",
            ).validate()

    def test_expansion_is_predeclared_not_cartesian(self):
        plan = RepeatedPlan(n_points=500, reference_grid=(10, 10))
        rows = expand_plan(plan)
        core = rows[rows["layer"] == "core"]
        self.assertEqual(len(core), 21)
        self.assertEqual(core["family"].nunique(), 3)
        self.assertEqual(core["condition"].nunique(), 7)
        sensitivity = rows[rows["layer"] == "sensitivity"]
        for _, row in sensitivity.iterrows():
            changes = sum([
                row.effect_pp != 10.0, row.support_frac != .02,
                row.target_shape != "circle", row.hdbscan_frac != .005,
            ])
            self.assertEqual(changes, 1)
        self.assertFalse(rows["scenario_id"].duplicated().any())


class RepeatedGeneratorTests(unittest.TestCase):
    def test_all_families_and_shapes_are_deterministic_and_roles_partition_points(self):
        source = pd.DataFrame({"lat": np.linspace(-2, 2, 100), "lon": np.linspace(-3, 3, 100)})
        for family in ("uniform", "clustered", "realistic_irregular"):
            for shape in ("circle", "rotated_ellipse", "irregular_nonconvex"):
                first = generate_geography(200, family, shape, .05, 17, coordinate_source=source)
                second = generate_geography(200, family, shape, .05, 17, coordinate_source=source)
                pd.testing.assert_frame_equal(first.points, second.points)
                self.assertEqual(len(first.points), 200)
                self.assertEqual(first.points["point_id"].tolist(), list(range(200)))
                self.assertEqual(set(first.points["role"]), {"focal_target", "manipulated_context", "compensation", "null_background"})
                self.assertEqual(first.points["role"].notna().sum(), 200)
                self.assertEqual((first.points["role"] == "focal_target").sum(), 10)

    def test_irregular_geography_requires_coordinate_source(self):
        with self.assertRaisesRegex(ValueError, "fonte"):
            generate_geography(100, "realistic_irregular", "circle", .02, 1)

    def test_mirrored_outcomes_preserve_total_and_invert_target_direction(self):
        geography = generate_geography(500, "uniform", "circle", .1, 4)
        positive = generate_outcomes(geography.points, "local_positive", 10.0, .5, 8)
        negative = generate_outcomes(geography.points, "local_negative", 10.0, .5, 8)
        target = geography.points["role"].eq("focal_target").to_numpy()
        self.assertEqual(positive.sum(), 250)
        self.assertEqual(negative.sum(), 250)
        self.assertGreater(positive[target].mean(), positive[~target].mean())
        self.assertLess(negative[target].mean(), negative[~target].mean())

    def test_simultaneous_opposite_keeps_null_background_unmanipulated(self):
        geography = generate_geography(1000, "uniform", "circle", .02, 4)
        outcome = generate_outcomes(
            geography.points, "simultaneous_opposite", 20.0, .5, 8
        )
        rates = pd.DataFrame({"role": geography.points["role"], "outcome": outcome}).groupby("role")["outcome"].mean()
        self.assertGreater(rates["focal_target"], .5)
        self.assertLess(rates["manipulated_context"], .5)
        self.assertAlmostEqual(rates["null_background"], .5, delta=.002)


class SpatialRecoveryTests(unittest.TestCase):
    def setUp(self):
        roles = ["focal_target"] * 4 + ["manipulated_context"] * 2 + ["compensation"] * 2 + ["null_background"] * 2
        self.truth = pd.DataFrame({"point_id": range(10), "role": roles})

    def test_union_deduplicates_and_half_threshold_passes(self):
        regions = [
            {"point_ids": [0, 1, 4], "direction": "positive", "significant": True, "consolidated": True},
            {"point_ids": [1, 2, 5], "direction": "positive", "significant": True, "consolidated": True},
        ]
        result = evaluate_spatial_recovery(
            self.truth,
            regions,
            expected_direction="positive",
            fair=False,
            evaluated_point_ids=[0, 1, 2, 4, 5, 6, 7, 8, 9],
        )
        self.assertEqual(result["recovered_n"], 5)
        self.assertEqual(result["true_positive_n"], 3)
        self.assertEqual(result["false_positive_n"], 2)
        self.assertEqual(result["false_negative_n"], 1)
        self.assertEqual(result["true_negative_n"], 4)
        self.assertAlmostEqual(result["precision"], 3 / 5)
        self.assertAlmostEqual(result["recall"], 3 / 4)
        self.assertAlmostEqual(result["f1"], 2 / 3)
        self.assertEqual(result["target_covered_n"], 3)
        self.assertAlmostEqual(result["target_coverage"], 3 / 4)
        self.assertEqual(result["unassigned_target_n"], 1)
        self.assertEqual(result["detected_point_ids"], [0, 1, 2, 4, 5])
        self.assertTrue(result["correct_recovery"])
        self.assertEqual(result["role_manipulated_context_n"], 2)

    def test_wrong_direction_fails_and_fair_counts_once(self):
        regions = [{"point_ids": [0, 1, 2, 3], "direction": "negative", "significant": True, "consolidated": True}]
        unfair = evaluate_spatial_recovery(self.truth, regions, expected_direction="positive", fair=False)
        fair = evaluate_spatial_recovery(self.truth, regions * 3, expected_direction=None, fair=True)
        self.assertFalse(unfair["correct_recovery"])
        self.assertEqual(unfair["true_positive_n"], 4)
        self.assertEqual(unfair["directional_recovered_n"], 0)
        self.assertEqual(unfair["directional_recall"], 0.0)
        self.assertTrue(fair["familywise_false_alarm"])
        self.assertEqual(fair["scenario_detected"], True)

    def test_candidate_without_rate_direction_can_use_unsigned_recovery(self):
        regions = [
            {"point_ids": [0, 1, 2, 4], "direction": "positive", "significant": True, "consolidated": True}
        ]
        result = evaluate_spatial_recovery(
            self.truth,
            regions,
            expected_direction=None,
            fair=False,
            direction_required=False,
        )
        self.assertFalse(result["direction_required"])
        self.assertEqual(result["true_positive_n"], 3)
        self.assertTrue(result["correct_recovery"])

    def test_exploratory_candidate_reports_overlap_without_primary_recovery(self):
        regions = [
            {"point_ids": [0, 1, 2, 4], "direction": None, "significant": True, "consolidated": True}
        ]
        result = evaluate_spatial_recovery(
            self.truth,
            regions,
            expected_direction=None,
            fair=False,
            direction_required=False,
            recovery_eligible=False,
        )
        self.assertEqual(result["true_positive_n"], 3)
        self.assertAlmostEqual(result["f1"], .75)
        self.assertIsNone(result["correct_recovery"])

    def test_pairwise_method_agreement_counts_point_intersections(self):
        agreement = compare_method_detection_sets({
            "grid_sul": [0, 1, 2, 4, 5],
            "hdbscan_local_z": [1, 2, 3, 6],
        })
        row = agreement.iloc[0]
        self.assertEqual(row["intersection_n"], 2)
        self.assertEqual(row["first_only_n"], 3)
        self.assertEqual(row["second_only_n"], 2)
        self.assertEqual(row["union_n"], 7)
        self.assertAlmostEqual(row["point_jaccard"], 2 / 7)


class RepeatedStatisticsTests(unittest.TestCase):
    def test_fwer_bootstrap_blocks_geometries_and_gate_boundary(self):
        rows = pd.DataFrame({
            "family": ["uniform"] * 6,
            "method_id": ["grid_sul"] * 6,
            "geometry_seed": [1, 1, 1, 2, 2, 2],
            "familywise_false_alarm": [False, False, False, False, False, False],
        })
        gate = block_fwer_gate(rows, n_bootstrap=100, seed=3)
        self.assertEqual(gate.iloc[0]["fwer"], 0.0)
        self.assertLessEqual(gate.iloc[0]["upper_one_sided_95"], .01)
        self.assertTrue(gate.iloc[0]["gate_passed"])

    def test_paired_contrast_preserves_sign_and_reports_inconclusive(self):
        rows = []
        for geometry in range(4):
            for outcome in range(3):
                rows.extend([
                    {"family": "uniform", "scenario_id": "x", "geometry_seed": geometry, "outcome_seed": outcome, "method_id": "a", "correct_recovery": True},
                    {"family": "uniform", "scenario_id": "x", "geometry_seed": geometry, "outcome_seed": outcome, "method_id": "b", "correct_recovery": geometry % 2 == 0},
                ])
        result = paired_recovery_contrast(pd.DataFrame(rows), "a", "b", n_bootstrap=200, seed=9)
        self.assertGreater(result.iloc[0]["difference_pp"], 0)
        self.assertIn(result.iloc[0]["conclusion"], {"inconclusivo", "favorece_primeiro"})


if __name__ == "__main__":
    unittest.main()
