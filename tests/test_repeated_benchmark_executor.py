import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from repeated_benchmark import RepeatedBenchmarkRunner, benchmark_method_spec
from repeated_plan import RepeatedPlan, expand_plan


class RepeatedBenchmarkExecutorTests(unittest.TestCase):
    def test_candidate_method_ids_declare_system_metric_and_direction_contract(self):
        difference = benchmark_method_spec("hdbscan_peer_rate_difference")
        ratio = benchmark_method_spec("hdbscan_peer_log_rate_ratio")
        gini = benchmark_method_spec("hdbscan_peer_gini_gap")

        self.assertEqual((difference.system, difference.metric), ("hdbscan", "peer_rate_difference"))
        self.assertEqual((ratio.system, ratio.metric), ("hdbscan", "peer_log_rate_ratio"))
        self.assertTrue(difference.direction_required)
        self.assertFalse(gini.direction_required)

    def test_minimal_plan_runs_all_requested_methods_and_resumes(self):
        plan = RepeatedPlan(
            n_points=120, reference_grid=(4, 4), geometry_seeds=(2,),
            fair_outcomes_per_geometry=1, unfair_outcomes_per_geometry=1,
            null_worlds=3, kmeans_seeds=3, scan_radii=(.15,),
            methods=(
                "hdbscan_local_z", "hdbscan_peer_rate_difference",
                "hdbscan_peer_log_rate_ratio", "hdbscan_peer_gini_gap",
                "hdbscan_sul", "grid_sul", "scan_sul",
            ),
            bootstrap_repetitions=20,
        )
        expanded = expand_plan(plan)
        selected = [
            expanded[(expanded.layer == "core") & (expanded.family == "uniform") & (expanded.condition == condition)].iloc[0].scenario_id
            for condition in ("fair", "local_positive")
        ]
        with tempfile.TemporaryDirectory() as tmp:
            runner = RepeatedBenchmarkRunner(plan, Path(tmp))
            first = runner.run(scenario_ids=selected)
            self.assertEqual(len(first), 14)
            self.assertEqual(set(first["method_id"]), set(plan.methods))
            self.assertTrue({
                "geometry_seed", "outcome_seed", "null_seed", "threshold",
                "scenario_detected", "target_n", "true_positive_n",
                "false_positive_n", "false_negative_n", "true_negative_n",
                "f1", "target_coverage", "detected_point_ids",
                "directional_detected_point_ids", "evaluation_coverage",
                "detected_coverage", "role_focal_target_rate",
            }.issubset(first.columns))
            scan = first[first["method_id"].eq("scan_sul")]
            self.assertTrue((scan["coverage"] == scan["evaluation_coverage"]).all())
            self.assertTrue(scan["evaluation_coverage"].between(0, 1).all())
            gini = first[first["method_id"].eq("hdbscan_peer_gini_gap")]
            self.assertTrue(gini["correct_recovery"].isna().all())
            manifests = list((Path(tmp) / "results").rglob("manifest.json"))
            truth_file = next((Path(tmp) / "truth").rglob("results.csv"))
            truth = pd.read_csv(truth_file)
            self.assertEqual(len(truth), plan.n_points)
            self.assertTrue({
                "point_id", "lat", "lon", "role", "outcome",
                "geometry_seed", "outcome_seed", "null_seed",
            }.issubset(truth.columns))
            detection_file = next((Path(tmp) / "detections").rglob("results.csv"))
            detections = pd.read_csv(detection_file)
            self.assertTrue({
                "candidate", "significant", "consolidated", "point_ids",
            }.issubset(detections.columns))
            scan_partition = next((Path(tmp) / "partitions").rglob("scan/results.csv"))
            self.assertTrue(__import__("pandas").read_csv(scan_partition)["point_ids"].isna().all())
            mtimes = {path: path.stat().st_mtime_ns for path in manifests}
            second = RepeatedBenchmarkRunner(plan, Path(tmp)).run(scenario_ids=selected)
            self.assertEqual(len(second), 14)
            self.assertEqual(mtimes, {path: path.stat().st_mtime_ns for path in manifests})

    def test_partition_and_threshold_cache_keys_include_rate(self):
        plan = RepeatedPlan(n_points=100, reference_grid=(4, 4), geometry_seeds=(1,), null_worlds=2)
        runner = RepeatedBenchmarkRunner(plan, Path("unused"))
        first = runner.calibration_key("g", "grid_sul", "sul", 50)
        changed = runner.calibration_key("g", "grid_sul", "sul", 49)
        self.assertNotEqual(first, changed)
        changed_partition = runner.calibration_key(
            "g", "grid_sul", "sul", 50, "different-partition"
        )
        self.assertNotEqual(first, changed_partition)

    def test_coordinate_source_must_match_the_frozen_plan(self):
        plan = RepeatedPlan(
            n_points=100,
            reference_grid=(4, 4),
            coordinate_source_dataset="lar",
        )
        source = pd.DataFrame({"lat": [0.0], "lon": [0.0]})
        source.attrs["dataset_name"] = "crime"
        with self.assertRaisesRegex(ValueError, "não corresponde ao plano"):
            RepeatedBenchmarkRunner(plan, Path("unused"), coordinate_source=source)


if __name__ == "__main__":
    unittest.main()
