import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from benchmark_checkpoint import load_benchmark_checkpoint
from benchmark_sacharidis import (
    run_grid_unit,
    run_partition_unit,
    run_random_grid_meanvar_unit,
    run_scan_unit,
)
from tests.test_benchmark_evaluation import fixture_dataset, fixture_partition


PROVENANCE = {"commit": "test", "dirty": False}


class SacharidisBenchmarkUnitTests(unittest.TestCase):
    def test_grid_unit_publishes_sul_and_diagnostic_meanvar(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = run_grid_unit(
                fixture_dataset(),
                Path(tmp),
                lon_n=2,
                lat_n=2,
                protocol="reproduction",
                n_alt_worlds=20,
                signif_level=0.05,
                seed=42,
                code_provenance=PROVENANCE,
            )
            loaded = load_benchmark_checkpoint(checkpoint.path, checkpoint.spec)
            summaries = loaded.results[loaded.results["record_type"] == "summary"]

        self.assertEqual(set(summaries["metric"]), {"sul", "meanvar"})
        self.assertEqual(set(summaries["partitioning"]), {"grid_2x2"})
        self.assertEqual(summaries.loc[summaries["metric"] == "meanvar", "evaluation_mode"].item(), "diagnostic")

    def test_partition_unit_keeps_sul_and_local_z_on_the_same_partition(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = run_partition_unit(
                fixture_dataset(),
                fixture_partition(),
                Path(tmp),
                protocol="standardized",
                metrics=("sul", "local_z"),
                n_alt_worlds=20,
                signif_level=0.05,
                seed=42,
                code_provenance=PROVENANCE,
            )
            summaries = checkpoint.results[checkpoint.results["record_type"] == "summary"]

        self.assertEqual(set(summaries["metric"]), {"sul", "local_z"})
        self.assertEqual(len(set(summaries["params"])), 1)

    def test_scan_unit_declares_reproduction_and_standardized_protocols_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reproduction = run_scan_unit(
                fixture_dataset(), root, protocol="reproduction", n_seeds=2,
                n_alt_worlds=20, signif_level=0.05, seed=42,
                code_provenance=PROVENANCE,
            )
            standardized = run_scan_unit(
                fixture_dataset(), root, protocol="standardized", n_seeds=2,
                n_alt_worlds=25, signif_level=0.05, seed=42,
                code_provenance=PROVENANCE,
            )

        self.assertNotEqual(reproduction.path, standardized.path)
        self.assertEqual(reproduction.summary.loc[0, "n_alt_worlds"], 20)
        self.assertEqual(standardized.summary.loc[0, "n_alt_worlds"], 25)
        self.assertEqual(reproduction.summary.loc[0, "candidate_regions"], 4)

    def test_random_grid_meanvar_unit_reports_partition_distribution_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_random_grid_meanvar_unit(
                fixture_dataset(),
                Path(tmp),
                n_partitionings=3,
                lon_n_range=(2, 2),
                lat_n_range=(2, 2),
                seed=42,
                code_provenance=PROVENANCE,
            )

        summaries = result.results[result.results["record_type"] == "summary"]
        self.assertEqual(len(summaries), 4)
        aggregate = summaries[summaries["method"] == "random_grid_summary"].iloc[0]
        self.assertEqual(aggregate["n_partitionings"], 3)
        self.assertTrue(np.isfinite(aggregate["partition_score"]))


if __name__ == "__main__":
    unittest.main()
