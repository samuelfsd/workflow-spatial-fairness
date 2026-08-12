import sys
import tempfile
import unittest
import json
from pathlib import Path

import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from clustering.base import Partition
from exploration_report import generate_cluster_exploration
from metrics.base import MetricResult
from run_snapshot import write_run_snapshot
from tests.test_exploration import exploration_fixture


class ExplorationReportTests(unittest.TestCase):
    def test_core_profile_materializes_tables_figures_and_main_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, _ = exploration_fixture(root)
            partition = Partition(
                "fake", {"min_cluster_size": 2, "min_samples": 2},
                np.array([0, 0, 1, 1, 2, 2, 3, -1]),
                [
                    {"points": [0, 1], "cluster_label": 0},
                    {"points": [2, 3], "cluster_label": 1},
                    {"points": [4, 5], "cluster_label": 2},
                    {"points": [6], "cluster_label": 3, "origin": "rescue",
                     "origin_cluster_label": 8},
                ], [7],
            )
            run_dir = root / "run"
            write_run_snapshot(
                run_dir, dataset=dataset, partition=partition,
                metric_results={
                    "local_z": MetricResult(np.array([-3.0, 3.0, 1.0, 0.5]),
                                             signed=True, supports_mc=True),
                    "sul": MetricResult(np.array([4.0, 5.0, 1.0, 0.5]), supports_mc=True),
                },
                thresholds={"local_z": 2.0, "sul": 3.0},
                null_distributions={"local_z": np.array([1.0, 2.0]),
                                    "sul": np.array([2.0, 3.0])},
                effective_seeds={"local_z": 42, "sul": 43}, seed=40,
                primary_metric="local_z", signif_level=0.005, n_alt_worlds=2,
                command="fixture", exploration_profile="none",
            )

            report_dir = generate_cluster_exploration(
                run_dir, primary_metric="local_z", profile="core", dataset=dataset
            )

            self.assertTrue((report_dir / "cluster_exploration.pdf").exists())
            self.assertTrue((report_dir / "tables" / "cluster_features.csv").exists())
            self.assertTrue((report_dir / "tables" / "coverage_audit.csv").exists())
            self.assertTrue((report_dir / "tables" / "distribution_summary.csv").exists())
            pngs = list((report_dir / "figures" / "global").glob("*.png"))
            pdfs = list((report_dir / "figures" / "global").glob("*.pdf"))
            self.assertEqual(len(pngs), 8)
            self.assertEqual(len(pdfs), 8)

    def test_invalid_snapshot_does_not_publish_partial_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, snapshot = exploration_fixture(root)
            run_dir = root / "bad-run"
            run_dir.mkdir()
            destination = root / "published"
            destination.mkdir()
            sentinel = destination / "complete.txt"
            sentinel.write_text("previous")

            with self.assertRaises(FileNotFoundError):
                generate_cluster_exploration(
                    run_dir, primary_metric="local_z", profile="core",
                    output_dir=destination, dataset=dataset,
                )

            self.assertEqual(sentinel.read_text(), "previous")
            self.assertEqual(list(destination.iterdir()), [sentinel])

    def test_full_profile_adds_complete_details_and_supplements(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, _ = exploration_fixture(root)
            partition = Partition(
                "fake", {"min_cluster_size": 2, "min_samples": 2},
                np.array([0, 0, 1, 1, 2, 2, 3, -1]),
                [
                    {"points": [0, 1], "cluster_label": 0},
                    {"points": [2, 3], "cluster_label": 1},
                    {"points": [4, 5], "cluster_label": 2},
                    {"points": [6], "cluster_label": 3},
                ], [7],
            )
            run_dir = root / "run"
            write_run_snapshot(
                run_dir, dataset=dataset, partition=partition,
                metric_results={
                    "local_z": MetricResult(np.array([-3.0, 3.0, 1.0, 0.5]),
                                             signed=True, supports_mc=True),
                    "sul": MetricResult(np.array([4.0, 5.0, 1.0, 0.5]), supports_mc=True),
                }, thresholds={"local_z": 2.0, "sul": 3.0},
                null_distributions={"local_z": np.array([1.0, 2.0]),
                                    "sul": np.array([2.0, 3.0])},
                effective_seeds={"local_z": 42, "sul": 43}, seed=40,
                primary_metric="local_z", signif_level=0.005, n_alt_worlds=2,
                command="fixture", exploration_profile="none",
            )
            report_dir = generate_cluster_exploration(
                run_dir, primary_metric="local_z", profile="full",
                dataset=dataset, detail_selection="all",
            )

            self.assertTrue((report_dir / "analysis_summary.json").exists())
            self.assertTrue((report_dir / "cluster_details_001.pdf").exists())
            self.assertTrue((report_dir / "cluster_supplements.pdf").exists())
            self.assertTrue((report_dir / "tables" / "selected_clusters.csv").exists())
            self.assertTrue((report_dir / "tables" / "internal_subclusters.csv").exists())
            self.assertEqual(
                len(list((report_dir / "figures" / "details").glob("*.png"))), 12
            )
            self.assertEqual(
                len(list((report_dir / "figures" / "supplementary").glob("*.png"))), 6
            )

    def test_custom_profile_selects_families_but_keeps_canonical_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, _ = exploration_fixture(root)
            partition = Partition(
                "fake", {"min_cluster_size": 2, "min_samples": 2},
                np.array([0, 0, 1, 1, 2, 2, 3, -1]),
                [{"points": [0, 1], "cluster_label": 0},
                 {"points": [2, 3], "cluster_label": 1},
                 {"points": [4, 5], "cluster_label": 2},
                 {"points": [6], "cluster_label": 3}], [7],
            )
            run_dir = root / "run"
            write_run_snapshot(
                run_dir, dataset=dataset, partition=partition,
                metric_results={"local_z": MetricResult(
                    np.array([-3.0, 3.0, 1.0, 0.5]), signed=True, supports_mc=True
                )}, thresholds={"local_z": 2.0},
                null_distributions={"local_z": np.array([1.0, 2.0])},
                effective_seeds={"local_z": 42}, seed=40,
                primary_metric="local_z", signif_level=0.005, n_alt_worlds=2,
                command="fixture", exploration_profile="none",
            )
            destination = generate_cluster_exploration(
                run_dir, primary_metric="local_z", profile="custom",
                custom_families={"supplements"}, dataset=dataset,
            )
            self.assertTrue((destination / "tables" / "cluster_features.csv").exists())
            self.assertTrue((destination / "figures" / "supplementary").exists())
            self.assertFalse((destination / "figures" / "details").exists())
            report_manifest = json.loads(
                (destination / "report_manifest.json").read_text()
            )
            self.assertEqual(report_manifest["families"], ["supplements"])

    def test_ambiguous_interrupted_publication_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, _ = exploration_fixture(root)
            partition = Partition(
                "fake", {"min_cluster_size": 2, "min_samples": 2},
                np.array([0, 0, 1, 1, 2, 2, 3, -1]),
                [{"points": [0, 1], "cluster_label": 0},
                 {"points": [2, 3], "cluster_label": 1},
                 {"points": [4, 5], "cluster_label": 2},
                 {"points": [6], "cluster_label": 3}], [7],
            )
            run_dir = root / "run"
            write_run_snapshot(
                run_dir, dataset=dataset, partition=partition,
                metric_results={"local_z": MetricResult(
                    np.array([-3.0, 3.0, 1.0, 0.5]), signed=True, supports_mc=True
                )}, thresholds={"local_z": 2.0},
                null_distributions={"local_z": np.array([1.0, 2.0])},
                effective_seeds={"local_z": 42}, seed=40,
                primary_metric="local_z", signif_level=0.005, n_alt_worlds=2,
                command="fixture", exploration_profile="none",
            )
            destination = root / "published"
            (root / ".published.backup-one").mkdir()
            (root / ".published.backup-two").mkdir()
            with self.assertRaisesRegex(RuntimeError, "Ambiguous"):
                generate_cluster_exploration(
                    run_dir, primary_metric="local_z", profile="core",
                    output_dir=destination, dataset=dataset,
                )

    def test_renderer_failure_preserves_last_complete_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, _ = exploration_fixture(root)
            partition = Partition(
                "fake", {"min_cluster_size": 2, "min_samples": 2},
                np.array([0, 0, 1, 1, 2, 2, 3, -1]),
                [{"points": [0, 1], "cluster_label": 0},
                 {"points": [2, 3], "cluster_label": 1},
                 {"points": [4, 5], "cluster_label": 2},
                 {"points": [6], "cluster_label": 3}], [7],
            )
            run_dir = root / "run"
            write_run_snapshot(
                run_dir, dataset=dataset, partition=partition,
                metric_results={"local_z": MetricResult(
                    np.array([-3.0, 3.0, 1.0, 0.5]), signed=True, supports_mc=True
                )}, thresholds={"local_z": 2.0},
                null_distributions={"local_z": np.array([1.0, 2.0])},
                effective_seeds={"local_z": 42}, seed=40,
                primary_metric="local_z", signif_level=0.005, n_alt_worlds=2,
                command="fixture", exploration_profile="none",
            )
            destination = root / "published"
            destination.mkdir()
            sentinel = destination / "complete.txt"
            sentinel.write_text("last-good")

            def failing_renderer(*args):
                raise RuntimeError("renderer failed")

            with self.assertRaisesRegex(RuntimeError, "renderer failed"):
                generate_cluster_exploration(
                    run_dir, primary_metric="local_z", profile="core",
                    output_dir=destination, dataset=dataset,
                    core_figure_builder=failing_renderer,
                )

            self.assertEqual(sentinel.read_text(), "last-good")
            self.assertEqual(list(destination.iterdir()), [sentinel])


if __name__ == "__main__":
    unittest.main()
