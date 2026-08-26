import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from benchmark_initial import InitialBenchmarkConfig, run_initial_benchmark
from main import build_parser


class InitialBenchmarkTests(unittest.TestCase):
    def test_cli_exposes_dataset_phase_resume_and_protocol_sizes(self):
        args = build_parser().parse_args([
            "benchmark-sacharidis", "--dataset", "crime", "--phase", "report",
            "--resume", "--scan-worlds", "7", "--grid-worlds", "9",
        ])
        self.assertEqual(args.dataset, "crime")
        self.assertEqual(args.phase, "report")
        self.assertTrue(args.resume)
        self.assertEqual(args.scan_worlds, 7)
        self.assertEqual(args.grid_worlds, 9)

    def test_report_phase_never_calls_scientific_runner(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
            "benchmark_initial.publish_report_from_checkpoints"
        ) as report, patch(
            "benchmark_initial.SacharidisBenchmarkRunner.run_reproduce"
        ) as reproduce, patch(
            "benchmark_initial.SacharidisBenchmarkRunner.run_compare"
        ) as compare:
            report.return_value = Path(tmp) / "report"
            result = run_initial_benchmark(InitialBenchmarkConfig(
                output_root=Path(tmp), datasets=("crime",), phase="report", maps=False,
            ))
            self.assertEqual(result, report.return_value)
            reproduce.assert_not_called()
            compare.assert_not_called()

    def test_invalid_dataset_is_rejected_before_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "dataset"):
                run_initial_benchmark(InitialBenchmarkConfig(
                    output_root=Path(tmp), datasets=("unknown",), phase="reproduce",
                ))


if __name__ == "__main__":
    unittest.main()
