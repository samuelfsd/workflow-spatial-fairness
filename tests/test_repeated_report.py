import sys
import tempfile
import unittest
import json
from pathlib import Path

import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from repeated_report import _figures, build_repeated_tables, publish_repeated_report


class RepeatedReportTests(unittest.TestCase):
    def setUp(self):
        rows = []
        for method in (
            "hdbscan_local_z", "hdbscan_peer_rate_difference",
            "hdbscan_peer_log_rate_ratio", "hdbscan_peer_gini_gap",
            "hdbscan_sul", "grid_sul", "scan_sul",
        ):
            for fair, condition in ((True, "fair"), (False, "local_positive")):
                detected = [] if fair else ([0, 1, 2] if method == "grid_sul" else [1, 2, 3])
                all_detected = [] if fair else (
                    [0, 1, 2, 4] if method == "grid_sul" else [1, 2, 3]
                )
                rows.append({
                    "family": "uniform", "layer": "core", "scenario_id": condition,
                    "condition": condition, "method_id": method,
                    "metric": (
                        "local_z" if method.endswith("local_z")
                        else "peer_rate_difference" if method.endswith("rate_difference")
                        else "peer_log_rate_ratio" if method.endswith("rate_ratio")
                        else "peer_gini_gap" if method.endswith("gini_gap")
                        else "sul"
                    ),
                    "expected_direction": None if fair else "positive",
                    "geometry_seed": 1, "outcome_seed": 1,
                    "familywise_false_alarm": False, "scenario_detected": not fair,
                    "correct_recovery": not fair, "precision": .7 if not fair else 0,
                    "recall": .8 if not fair else 0, "f1": .7467 if not fair else 0,
                    "iou": .6 if not fair else 0,
                    "directional_precision": .7 if not fair else 0,
                    "directional_recall": .8 if not fair else 0,
                    "directional_f1": .7467 if not fair else 0,
                    "directional_iou": .6 if not fair else 0,
                    "true_positive_n": 2 if not fair else 0,
                    "false_positive_n": 1 if not fair else 0,
                    "false_negative_n": 1 if not fair else 0,
                    "true_negative_n": 6 if not fair else 10,
                    "target_coverage": .9 if not fair else 1.0,
                    "unassigned_target_n": 1 if not fair else 0,
                    "role_focal_target_n": 2 if not fair else 0,
                    "role_focal_target_total_n": 3,
                    "role_focal_target_rate": 2 / 3 if not fair else 0,
                    "role_manipulated_context_n": 1 if not fair else 0,
                    "role_manipulated_context_total_n": 2,
                    "role_manipulated_context_rate": .5 if not fair else 0,
                    "role_compensation_n": 0,
                    "role_compensation_total_n": 2,
                    "role_compensation_rate": 0,
                    "role_null_background_n": 0,
                    "role_null_background_total_n": 3,
                    "role_null_background_rate": 0,
                    "detected_point_ids": json.dumps(detected),
                    "all_detected_point_ids": json.dumps(all_detected),
                    "spatial_false_alarm": False, "coverage": .9,
                    "evaluation_coverage": .9,
                    "detected_coverage": .3 if not fair else 0,
                    "candidate_regions": 10, "consolidated_regions": 1,
                    "raw_significant_regions": 2,
                })
        base = pd.DataFrame(rows)
        self.results = pd.concat([
            base,
            base.assign(family="clustered", geometry_seed=2),
            base.assign(family="irregular", geometry_seed=3),
        ], ignore_index=True)

    def test_tables_follow_gate_recovery_location_operational_hierarchy(self):
        tables = build_repeated_tables(self.results, n_bootstrap=20, seed=7)
        self.assertEqual(list(tables)[:8], [
            "validity_gate", "recovery", "point_confusion",
            "method_agreement", "role_reach", "exploratory_candidates",
            "location", "operational",
        ])
        self.assertEqual(len(tables["contrasts"]["contrast"].drop_duplicates()), 7)
        self.assertNotIn("score_total", tables["canonical"].columns)
        self.assertIn("expected_direction", tables["canonical"].columns)
        self.assertIn("scan_redundancy", tables["operational"].columns)
        self.assertIn("evaluation_coverage", tables["operational"].columns)
        self.assertIn("detected_coverage", tables["operational"].columns)
        confusion = tables["point_confusion"]
        self.assertTrue({
            "true_positive_n", "false_positive_n", "false_negative_n",
            "true_negative_n", "precision", "recall", "f1", "iou",
            "target_coverage",
        }.issubset(confusion.columns))
        agreement = tables["method_agreement"]
        pair = agreement[
            agreement["first_method"].eq("grid_sul")
            & agreement["second_method"].eq("hdbscan_local_z")
            & agreement["condition"].eq("local_positive")
        ].iloc[0]
        self.assertEqual(pair["mean_intersection_n"], 2)
        self.assertAlmostEqual(pair["mean_point_jaccard"], .4)
        self.assertEqual(pair["agreement_basis"], "all_significant_consolidated")
        self.assertNotIn(
            "hdbscan_peer_gini_gap", set(tables["recovery"]["method_id"])
        )
        self.assertEqual(
            set(tables["exploratory_candidates"]["method_id"]),
            {"hdbscan_peer_gini_gap"},
        )
        focal = tables["role_reach"][
            tables["role_reach"]["truth_role"].eq("focal_target")
            & tables["role_reach"]["condition"].eq("local_positive")
        ]
        self.assertTrue((focal["mean_detected_n"] == 2).all())

    def test_transactional_publication_writes_manifest_and_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = publish_repeated_report(
                self.results, Path(tmp) / "report", n_bootstrap=20, seed=7,
                plan_metadata={"plan_id": "tiny"},
            )
            self.assertTrue((destination / "manifest.json").exists())
            self.assertTrue((destination / "tables" / "validity_gate.csv").exists())
            self.assertTrue((destination / "tables" / "point_confusion.csv").exists())
            self.assertTrue((destination / "tables" / "method_agreement.csv").exists())
            self.assertTrue((destination / "tables" / "role_reach.csv").exists())
            self.assertTrue((destination / "tables" / "exploratory_candidates.csv").exists())
            self.assertTrue((destination / "figures" / "repeated_benchmark.pdf").exists())
            manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["hierarchy"][:4], [
                "validity_gate", "recovery", "point_confusion", "method_agreement",
            ])

    def test_figures_use_bounded_method_by_family_axes(self):
        tables = build_repeated_tables(self.results, n_bootstrap=20, seed=7)
        figures = _figures(tables)
        try:
            for _, figure in figures:
                for axis in figure.axes:
                    visible_x = [tick.get_text() for tick in axis.get_xticklabels() if tick.get_visible() and tick.get_text()]
                    visible_y = [tick.get_text() for tick in axis.get_yticklabels() if tick.get_visible() and tick.get_text()]
                    self.assertLessEqual(len(visible_x), 3)
                    self.assertLessEqual(len(visible_y), 7)
                    self.assertFalse(any("hdbscan_" in label for label in visible_x + visible_y))
        finally:
            import matplotlib.pyplot as plt
            for _, figure in figures:
                plt.close(figure)


if __name__ == "__main__":
    unittest.main()
