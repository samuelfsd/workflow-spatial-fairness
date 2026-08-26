import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from benchmark_sacharidis import SacharidisBenchmarkRunner, SacharidisProtocol
from tests.test_benchmark_evaluation import fixture_dataset


class SacharidisBenchmarkRunnerTests(unittest.TestCase):
    def setUp(self):
        self.protocol = SacharidisProtocol(
            reproduction_scan_worlds=20,
            grid_worlds=20,
            standardized_worlds=20,
            random_partitionings=2,
            kmeans_seeds=2,
            hdbscan_fracs=(0.005, 0.01),
            signif_level=0.05,
        )

    def _loader(self, name):
        return replace(fixture_dataset(name), fixed_grids=((2, 2),))

    def test_reproduction_routes_lar_scan_and_crime_grid_without_model_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = SacharidisBenchmarkRunner(
                Path(tmp), protocol=self.protocol, dataset_loader=self._loader,
                seed=42, code_provenance={"commit": "test", "dirty": False},
            )
            lar = runner.run_reproduce("lar")
            crime = runner.run_reproduce("crime")

        self.assertEqual(sum(unit.summary.iloc[0]["method"] == "kmeans_scan" for unit in lar), 3)
        self.assertEqual(sum(unit.summary.iloc[0]["method"] == "grid" for unit in lar), 1)
        self.assertEqual(len(crime), 1)
        self.assertEqual(crime[0].summary.loc[crime[0].summary["metric"] == "sul", "rate_semantics"].item(), "TPR")
        self.assertNotIn("accuracy", crime[0].summary.columns)

    def test_comparison_runs_standardized_grid_scan_and_each_hdbscan_sensitivity(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = SacharidisBenchmarkRunner(
                Path(tmp), protocol=self.protocol, dataset_loader=self._loader,
                seed=42, code_provenance={"commit": "test", "dirty": False},
            )
            units = runner.run_compare("lar")

        methods = [unit.summary.iloc[0]["method"] for unit in units]
        self.assertEqual(methods.count("grid"), 1)
        self.assertEqual(methods.count("kmeans_scan"), 1)
        self.assertEqual(methods.count("hdbscan"), 2)
        self.assertEqual({unit.summary.iloc[0]["protocol"] for unit in units}, {"standardized"})

    def test_synthetic_reproduction_keeps_random_meanvar_and_fixed_sul_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = SacharidisBenchmarkRunner(
                Path(tmp), protocol=self.protocol, dataset_loader=self._loader,
                seed=42, code_provenance={"commit": "test", "dirty": False},
            )
            units = runner.run_reproduce("semisynth")

        self.assertEqual(len(units), 2)
        self.assertEqual(
            {unit.summary.iloc[-1]["method"] for unit in units},
            {"grid", "random_grid_summary"},
        )


if __name__ == "__main__":
    unittest.main()
