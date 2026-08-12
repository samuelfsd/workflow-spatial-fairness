import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.text import Text


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from data_loading import DatasetSpec, LoadedDataset, file_sha256
from exploration import build_exploration_tables, robust_heatmap_frame
from exploration_figures import core_figures
from figures import close
from run_snapshot import RunSnapshot


def exploration_fixture(root: Path) -> tuple[LoadedDataset, RunSnapshot]:
    source = root / "fixture.csv"
    source.write_text("fixture\n", encoding="utf-8")
    spec = DatasetSpec(
        "fixture", "fixture.csv", "label", 0.1, 0.2, 0.1, ((2, 2),),
        "evento positivo", "evento negativo", "não declarada", directory="fixture",
    )
    df = pd.DataFrame(
        {
            "lat": [0.0, 0.1, 0.0, 0.1, 1.0, 1.1, 1.0, 1.1],
            "lon": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            "label": [1, 0, 1, 1, 0, 0, 1, 0],
            "outcome": [1, 0, 1, 1, 0, 0, 1, 0],
        }
    )
    types = df["outcome"].to_numpy(dtype=int)
    dataset = LoadedDataset(
        "fixture", df, types, 8, 4, np.array([0.1]), ((2, 2),), spec,
        source, file_sha256(source), 8,
    )
    assignments = pd.DataFrame(
        {
            "point_id": range(8),
            "cluster_label": pd.array([0, 0, 1, 1, 2, 2, 3, pd.NA], dtype="Int64"),
            "assignment_status": ["assigned"] * 7 + ["unassigned"],
            "origin": ["organic"] * 6 + ["rescue", None],
            "origin_cluster_label": pd.array([0, 0, 1, 1, 2, 2, 8, pd.NA], dtype="Int64"),
        }
    )
    scores = pd.DataFrame(
        {
            "cluster_label": [0, 1, 2, 3] * 2,
            "metric": ["local_z"] * 4 + ["sul"] * 4,
            "score": [-3.0, 3.0, 1.0, 0.5, 4.0, 5.0, 1.0, 0.5],
        }
    )
    thresholds = pd.DataFrame(
        {
            "metric": ["local_z", "sul"],
            "threshold": [2.0, 3.0],
            "n_worlds": [1000, 1000],
            "effective_seed": [42, 43],
        }
    )
    manifest = {
        "schema_version": 1,
        "dataset": {
            "name": "fixture",
            "outcome": {
                "positive_label": "evento positivo",
                "negative_label": "evento negativo",
                "desirability": "não declarada",
            },
        },
        "partition": {"method": "fake", "params": {"min_cluster_size": 2, "min_samples": 2}},
        "run": {"signif_level": 0.005},
    }
    snapshot = RunSnapshot(
        root, manifest, assignments, scores, thresholds,
        pd.DataFrame(columns=["metric", "world_idx", "max_abs_score", "effective_seed"]),
    )
    return dataset, snapshot


class ExplorationTableTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dataset, self.snapshot = exploration_fixture(Path(self.tmp.name))
        self.tables = build_exploration_tables(self.dataset, self.snapshot, "local_z")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_coverage_keeps_unassigned_points_outside_clusters(self):
        coverage = self.tables.coverage_audit.set_index("scope")
        self.assertEqual(coverage.loc["assigned", "n"], 7)
        self.assertEqual(coverage.loc["unassigned", "n"], 1)
        self.assertEqual(coverage.loc["unassigned", "n_neg"], 1)
        self.assertAlmostEqual(coverage.loc["assigned", "pct_dataset"], 7 / 8)
        self.assertEqual(coverage.loc["assigned_organic", "n"], 6)
        self.assertEqual(coverage.loc["assigned_rescue", "n"], 1)
        self.assertEqual(len(self.tables.cluster_features), 4)

    def test_canonical_balance_and_reference_formulas(self):
        row = self.tables.cluster_features.set_index("cluster_label").loc[0]
        self.assertEqual((row["n"], row["p"], row["n_neg"]), (2, 1, 1))
        self.assertEqual((row["pct_positive"], row["pct_negative"]), (0.5, 0.5))
        self.assertEqual(row["internal_predominance"], 0.0)
        self.assertEqual(row["global_deviation"], 0.0)
        self.assertEqual(row["primary_score"], -3.0)
        self.assertEqual(row["evaluation_status"], "avaliado")
        self.assertEqual(row["detection_class"], "negative")
        self.assertEqual(row["outcome_desirability"], "não declarada")
        self.assertIn("auto_selected", self.tables.cluster_features.columns)
        self.assertIn("selection_reasons", self.tables.cluster_features.columns)

    def test_size_summary_uses_sample_standard_deviation_and_quartiles(self):
        summary = self.tables.distribution_summary.set_index("metric").loc["n"]
        values = np.array([2, 2, 2, 1], dtype=float)
        self.assertEqual(summary["minimum"], 1)
        self.assertEqual(summary["maximum"], 2)
        self.assertEqual(summary["median"], 2)
        self.assertAlmostEqual(summary["std"], np.std(values, ddof=1))
        self.assertAlmostEqual(summary["iqr"], summary["q3"] - summary["q1"])

    def test_spatial_stats_use_haversine_and_expose_class_separation(self):
        row = self.tables.cluster_features.set_index("cluster_label").loc[0]
        expected_half_gap = math.radians(0.05) * 6371.0088
        expected_gap = math.radians(0.1) * 6371.0088
        self.assertAlmostEqual(row["distance_mean_km"], expected_half_gap, places=3)
        self.assertAlmostEqual(row["class_centroid_separation_km"], expected_gap, places=3)
        self.assertEqual(row["positive_own_dispersion_mean_km"], 0.0)
        self.assertEqual(row["negative_own_dispersion_mean_km"], 0.0)

    def test_absent_outcome_class_is_nan_with_reason(self):
        row = self.tables.cluster_features.set_index("cluster_label").loc[2]
        self.assertTrue(math.isnan(row["class_centroid_separation_km"]))
        self.assertEqual(row["positive_spatial_reason"], "outcome_ausente")

    def test_detection_summary_has_distinct_cluster_and_population_denominators(self):
        summary = self.tables.detection_summary.set_index("detection_class")
        self.assertEqual(summary["n_clusters"].sum(), 4)
        self.assertEqual(summary["n_points"].sum(), 7)
        self.assertAlmostEqual(summary["pct_clusters"].sum(), 1.0)
        self.assertAlmostEqual(summary["pct_points"].sum(), 1.0)

    def test_robust_heatmap_preserves_na_and_handles_zero_iqr(self):
        frame = pd.DataFrame({"cluster_label": [0, 1, 2], "constant": [2.0, 2.0, np.nan]})
        heatmap = robust_heatmap_frame(frame, ["constant"])
        self.assertEqual(list(heatmap["constant"][:2]), [0.0, 0.0])
        self.assertTrue(math.isnan(heatmap["constant"].iloc[2]))

    def test_core_figures_use_projection_safe_typography(self):
        figures = core_figures(self.tables, self.snapshot.manifest, "local_z")
        try:
            for _, figure in figures:
                visible_text = [
                    item for item in figure.findobj(match=Text)
                    if item.get_visible() and item.get_text().strip()
                ]
                self.assertTrue(visible_text)
                self.assertGreaterEqual(
                    min(item.get_fontsize() for item in visible_text), 12.0
                )
        finally:
            close(*(figure for _, figure in figures))

    def test_rankings_use_magnitude_for_signed_deviations(self):
        ranked = self.tables.rankings[
            self.tables.rankings["metric"] == "global_deviation"
        ]
        self.assertTrue((ranked["ranking_basis"] == "magnitude").all())
        self.assertTrue(np.allclose(ranked["rank_value"], ranked["value"].abs()))


if __name__ == "__main__":
    unittest.main()
