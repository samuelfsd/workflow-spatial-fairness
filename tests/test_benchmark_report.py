import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from benchmark_report import (
    build_canonical_tables,
    checkpoint_summaries_to_long,
    compare_compatible_results,
    load_checkpoint_results,
    publish_initial_report,
)
from benchmark_checkpoint import BenchmarkUnitSpec


class BenchmarkReportTests(unittest.TestCase):
    def test_checkpoint_discovery_requires_complete_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            complete = root / "lar" / "unit"
            complete.mkdir(parents=True)
            pd.DataFrame([{"record_type": "summary", "dataset": "lar"}]).to_csv(
                complete / "results.csv", index=False
            )
            spec = BenchmarkUnitSpec(
                dataset="lar", dataset_sha256="hash", protocol="reproduction",
                partitioning="grid", metric="sul", params={}, seed=42,
                n_alt_worlds=3, code_provenance={"commit": "x"},
            )
            (complete / "manifest.json").write_text(
                json.dumps({"schema_version": 1, "status": "complete", "fingerprint": spec.fingerprint, "unit": spec.__dict__})
            )
            incomplete = root / "crime" / "unit"
            incomplete.mkdir(parents=True)
            (incomplete / "manifest.json").write_text(json.dumps({"status": "running"}))

            frame = load_checkpoint_results(root)

            self.assertEqual(frame["dataset"].tolist(), ["lar"])
            self.assertEqual(frame["checkpoint_fingerprint"].tolist(), [spec.fingerprint])

    def test_long_form_preserves_nulls_and_units(self):
        summaries = pd.DataFrame([
            {
                "record_type": "summary", "source": "local", "dataset": "crime",
                "protocol": "reproduction", "method": "grid", "partitioning": "grid_20x20",
                "metric": "meanvar", "N": 60849, "global_rate": 0.5596,
                "significant_regions": None, "score": 0.2, "threshold": None,
            }
        ])

        long = checkpoint_summaries_to_long(summaries)

        significant = long[long["quantity"] == "significant_regions"].iloc[0]
        self.assertTrue(pd.isna(significant["value"]))
        self.assertEqual(significant["null_reason"], "não aplicável à métrica diagnóstica")
        self.assertEqual(long.loc[long["quantity"] == "global_rate", "unit"].iloc[0], "rate")

    def test_comparison_rejects_incompatible_units(self):
        reference = pd.DataFrame([{
            "dataset": "lar", "experiment": "fixed_grid", "region_system": "grid_100x50",
            "metric": "sul", "quantity": "best_region_rate", "value": 0.84, "unit": "rate",
            "source": "paper",
        }])
        local = reference.assign(source="local", value=84.0, unit="percentage_points")

        with self.assertRaisesRegex(ValueError, "unidades incompatíveis"):
            compare_compatible_results(reference, local)

    def test_panels_separate_crime_tpr_and_lar_scan(self):
        canonical = pd.DataFrame([
            {"dataset": "crime", "experiment": "fixed_grid", "source": "local", "quantity": "global_rate", "value": .56, "unit": "rate", "metric": "sul", "region_system": "grid_20x20"},
            {"dataset": "lar", "experiment": "unrestricted_scan", "source": "local", "quantity": "candidate_regions", "value": 2000, "unit": "count", "metric": "sul", "region_system": "kmeans_square_scan"},
            {"dataset": "synth_fair", "experiment": "synthetic_control", "source": "local", "quantity": "partition_score", "value": .04, "unit": "score", "metric": "meanvar", "region_system": "random_grids"},
        ])

        tables = build_canonical_tables(canonical, pd.DataFrame())

        self.assertEqual(tables["crime"]["dataset"].unique().tolist(), ["crime"])
        self.assertNotIn("accuracy", tables["crime"].get("quantity", pd.Series(dtype=str)).tolist())
        self.assertEqual(tables["lar_scan"]["experiment"].unique().tolist(), ["unrestricted_scan"])
        self.assertEqual(tables["synthetic_auxiliary"]["dataset"].tolist(), ["synth_fair"])

    def test_publication_is_structured_and_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "report"
            canonical = pd.DataFrame([{
                "source": "local", "protocol": "reproduction", "dataset": "crime",
                "experiment": "fixed_grid", "method": "grid", "region_system": "grid_20x20",
                "metric": "sul", "quantity": "global_rate", "value": .56, "unit": "rate",
            }])
            output = publish_initial_report(
                destination, canonical=canonical, comparisons=pd.DataFrame(),
                parity=pd.DataFrame([{"dataset": "crime", "local_n": 60849}]),
                render_figures=False, render_maps=False,
            )

            self.assertEqual(output, destination)
            self.assertTrue((destination / "manifest.json").exists())
            self.assertTrue((destination / "tables" / "canonical.csv").exists())
            self.assertTrue((destination / "tables" / "canonical.md").exists())


if __name__ == "__main__":
    unittest.main()
