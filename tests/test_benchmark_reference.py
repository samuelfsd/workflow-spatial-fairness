import json
import sys
import tempfile
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from benchmark_reference import (
    build_data_parity_table,
    load_reference_manifest,
    write_data_parity_report,
)


class BenchmarkReferenceTests(unittest.TestCase):
    def _manifest(self) -> dict:
        return {
            "schema_version": 1,
            "sources": {
                "paper": {"title": "Paper", "revision": "v1"},
                "repository": {"url": "https://example.test/repo", "commit": "abc123"},
            },
            "datasets": [
                {
                    "dataset": "fixture",
                    "public_filename": "fixture.csv",
                    "sha256": "abc",
                    "published_n": 5,
                    "published_global_rate": 0.6,
                    "observation": "rounded",
                }
            ],
            "results": [
                {
                    "id": "fixture.result",
                    "experiment": "grid",
                    "dataset": "fixture",
                    "region_system": "grid_2x2",
                    "metric": "sul",
                    "quantity": "significant_regions",
                    "value": 2,
                    "unit": "count",
                    "source": "paper",
                    "source_location": "section 4",
                    "precision": "exact",
                    "observation": "",
                }
            ],
        }

    def test_manifest_loader_preserves_traceable_reference_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reference.json"
            path.write_text(json.dumps(self._manifest()), encoding="utf-8")

            loaded = load_reference_manifest(path)

        self.assertEqual(loaded["sources"]["repository"]["commit"], "abc123")
        self.assertEqual(loaded["results"][0]["value"], 2)
        self.assertEqual(loaded["results"][0]["unit"], "count")

    def test_manifest_loader_rejects_duplicate_keys_and_unknown_units(self):
        manifest = self._manifest()
        manifest["results"].append(dict(manifest["results"][0]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reference.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_reference_manifest(path)

            manifest["results"] = [dict(manifest["results"][0], id="other", unit="mystery")]
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unit"):
                load_reference_manifest(path)

    def test_data_parity_distinguishes_public_artifact_from_paper_summary(self):
        manifest = self._manifest()
        observed = {
            "fixture": {
                "source_sha256": "abc",
                "n_total": 4,
                "p_total": 2,
                "global_rate": 0.5,
            }
        }

        table = build_data_parity_table(manifest, observed)

        row = table.iloc[0]
        self.assertTrue(row["public_artifact_identical"])
        self.assertFalse(row["paper_n_matches"])
        self.assertFalse(row["paper_global_rate_matches"])
        self.assertEqual(row["local_n"], 4)
        self.assertAlmostEqual(row["local_global_rate"], 0.5)

    def test_writer_publishes_csv_and_markdown_with_the_same_dataset(self):
        table = build_data_parity_table(
            self._manifest(),
            {"fixture": {"source_sha256": "abc", "n_total": 4, "p_total": 2, "global_rate": 0.5}},
        )
        with tempfile.TemporaryDirectory() as tmp:
            written = write_data_parity_report(table, Path(tmp))
            csv_text = (Path(tmp) / "dataset_parity.csv").read_text(encoding="utf-8")
            markdown = (Path(tmp) / "dataset_parity.md").read_text(encoding="utf-8")

        self.assertEqual({path.name for path in written}, {"dataset_parity.csv", "dataset_parity.md"})
        self.assertIn("fixture", csv_text)
        self.assertIn("fixture", markdown)


if __name__ == "__main__":
    unittest.main()

