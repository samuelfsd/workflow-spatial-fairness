import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from benchmark_checkpoint import (
    BenchmarkUnitSpec,
    checkpoint_state,
    load_benchmark_checkpoint,
    publish_benchmark_checkpoint,
)


class BenchmarkCheckpointTests(unittest.TestCase):
    def _spec(self, **changes) -> BenchmarkUnitSpec:
        values = {
            "dataset": "fixture",
            "dataset_sha256": "abc",
            "protocol": "standardized",
            "partitioning": "grid_2x2",
            "metric": "sul",
            "params": {"lon_n": 2, "lat_n": 2},
            "seed": 42,
            "n_alt_worlds": 1000,
            "code_provenance": {"commit": "deadbeef", "dirty": False},
        }
        values.update(changes)
        return BenchmarkUnitSpec(**values)

    def test_complete_checkpoint_round_trips_and_is_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "unit"
            frame = pd.DataFrame([{"score": 3.5, "significant": True}])
            first = publish_benchmark_checkpoint(destination, self._spec(), frame)
            second = publish_benchmark_checkpoint(destination, self._spec(), frame)
            loaded = load_benchmark_checkpoint(destination, self._spec())

        self.assertEqual(first, second)
        self.assertEqual(loaded.manifest["status"], "complete")
        self.assertEqual(loaded.results.loc[0, "score"], 3.5)

    def test_changed_inputs_are_incompatible_instead_of_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "unit"
            publish_benchmark_checkpoint(destination, self._spec(), pd.DataFrame([{"score": 1.0}]))

            for changed in (
                self._spec(dataset_sha256="changed"),
                self._spec(seed=7),
                self._spec(n_alt_worlds=200),
                self._spec(params={"lon_n": 3, "lat_n": 2}),
            ):
                with self.subTest(fingerprint=changed.fingerprint):
                    with self.assertRaisesRegex(ValueError, "fingerprint"):
                        publish_benchmark_checkpoint(destination, changed, pd.DataFrame([{"score": 2.0}]))

    def test_incomplete_directory_is_never_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "unit"
            destination.mkdir()
            (destination / "manifest.json").write_text(
                json.dumps({"schema_version": 1, "status": "running", "fingerprint": self._spec().fingerprint}),
                encoding="utf-8",
            )

            self.assertEqual(checkpoint_state(destination, self._spec()), "incomplete")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                load_benchmark_checkpoint(destination, self._spec())

    def test_missing_and_incompatible_states_are_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "unit"
            self.assertEqual(checkpoint_state(destination, self._spec()), "missing")
            publish_benchmark_checkpoint(destination, self._spec(), pd.DataFrame([{"score": 1.0}]))
            self.assertEqual(checkpoint_state(destination, self._spec(seed=9)), "incompatible")


if __name__ == "__main__":
    unittest.main()
