import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from clustering.base import Partition
from experiments import ExperimentRunner
from exploration_report import generate_cluster_exploration
from metric_comparison import compare_primary_metrics, write_primary_comparison
from run_snapshot import load_run_snapshot
from tests.test_exploration import exploration_fixture


class ExplainExplorationIntegrationTests(unittest.TestCase):
    def test_small_run_covers_snapshot_profiles_alternative_primary_and_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, _ = exploration_fixture(root)
            partition = Partition(
                "hdbscan", {"min_cluster_frac": 0.25, "min_cluster_size": 2, "min_samples": 2},
                np.array([0, 0, 1, 1, 2, 2, 3, -1]),
                [
                    {"points": [0, 1], "cluster_label": 0},
                    {"points": [2, 3], "cluster_label": 1},
                    {"points": [4, 5], "cluster_label": 2},
                    {"points": [6], "cluster_label": 3},
                ],
                [7],
            )
            run_dir = root / "run"
            runner = ExperimentRunner(
                run_dir, maps=False, seed=7, clustering_method="hdbscan", verbose=False
            )
            runner.dataset_cache[dataset.name] = dataset
            runner.partition_cache[(dataset.name, "hdbscan")] = [partition]
            runner.run_explain(
                dataset_name=dataset.name,
                n_alt_worlds=3,
                metrics=("local_z", "sul", "gini_subcluster"),
                primary_metric="local_z",
                exploration_profile="none",
            )

            snapshot = load_run_snapshot(run_dir, dataset)
            scores_before = snapshot.scores.copy(deep=True)
            self.assertFalse(snapshot.manifest["completion"]["exploration_report"])

            core = generate_cluster_exploration(
                run_dir, primary_metric="local_z", profile="core", dataset=dataset
            )
            full = generate_cluster_exploration(
                run_dir, primary_metric="local_z", profile="full", dataset=dataset,
                detail_selection="all",
            )
            sul = generate_cluster_exploration(
                run_dir, primary_metric="sul", profile="core", dataset=dataset
            )
            refreshed = load_run_snapshot(run_dir, dataset)
            comparison = compare_primary_metrics(dataset, refreshed, "local_z", "sul")
            comparison_dir = write_primary_comparison(
                comparison, run_dir / "exploration" / "comparisons" / "local_z_vs_sul",
                first="local_z", second="sul",
            )

            self.assertEqual(core, full)
            self.assertTrue((full / "cluster_exploration.pdf").exists())
            self.assertTrue((sul / "cluster_exploration.pdf").exists())
            self.assertTrue((comparison_dir / "primary_comparison.csv").exists())
            pd.testing.assert_frame_equal(refreshed.scores, scores_before)


if __name__ == "__main__":
    unittest.main()
