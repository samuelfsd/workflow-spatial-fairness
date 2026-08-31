import json
import sys
import tempfile
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from repeated_workflow import load_repeated_plan, run_repeated_workflow


class RepeatedWorkflowTests(unittest.TestCase):
    def test_draft_official_plan_refuses_run_without_n_and_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            path.write_text(json.dumps({"schema_version": 1, "n_points": None, "reference_grid": None}))
            with self.assertRaisesRegex(ValueError, "N"):
                load_repeated_plan(path)

    def test_official_run_requires_explicit_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            path.write_text(json.dumps({"schema_version": 1, "n_points": 100, "reference_grid": [4, 4], "geometry_seeds": [1], "null_worlds": 2}))
            with self.assertRaisesRegex(PermissionError, "confirm"):
                run_repeated_workflow(path, Path(tmp) / "out", phase="run", confirm_official=False)

    def test_report_accepts_checkpoints_written_by_trial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "plan.json"
            path.write_text(json.dumps({
                "schema_version": 1,
                "plan_id": "performance",
                "n_points": 100,
                "reference_grid": [4, 4],
                "geometry_seeds": [1],
                "null_worlds": 2,
                "bootstrap_repetitions": 2,
                "methods": ["hdbscan_local_z"],
            }))

            run_repeated_workflow(path, root / "out", phase="trial")
            report = run_repeated_workflow(path, root / "out", phase="report")

            self.assertTrue(report.exists())


if __name__ == "__main__":
    unittest.main()
