import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from clustering.base import Partition
from data_loading import DatasetSpec, LoadedDataset, file_sha256
from metrics.base import MetricResult
from run_snapshot import load_run_snapshot, write_run_snapshot


class RunSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / "fixture.csv"
        self.source.write_text("row,lat,lon,label\n0,0,0,1\n1,0,1,0\n2,1,0,1\n3,1,1,0\n")
        spec = DatasetSpec(
            "fixture", "fixture.csv", "label", 0.1, 0.2, 0.1, ((2, 2),),
            "label 1", "label 0", "não declarada", directory="fixture",
        )
        df = pd.DataFrame(
            {"lat": [0.0, 0.0, 1.0, 1.0], "lon": [0.0, 1.0, 0.0, 1.0],
             "label": [1, 0, 1, 0], "outcome": [1, 0, 1, 0]}
        )
        self.dataset = LoadedDataset(
            name="fixture", df=df, types=np.array([1, 0, 1, 0]),
            n_total=4, p_total=2, radii=np.array([0.1]), fixed_grids=((2, 2),),
            spec=spec, source_path=self.source, source_sha256=file_sha256(self.source),
            rows_before_clean=4,
        )
        self.partition = Partition(
            method="fake",
            params={"min_cluster_size": 2, "min_samples": 2},
            labels=np.array([0, 0, 1, -1]),
            regions=[
                {"points": [0, 1], "cluster_label": 0, "origin": "organic"},
                {"points": [2], "cluster_label": 1, "origin": "rescue",
                 "origin_cluster_label": 7},
            ],
            noise_points=[3],
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write(self):
        return write_run_snapshot(
            self.root / "run",
            dataset=self.dataset,
            partition=self.partition,
            metric_results={
                "local_z": MetricResult(np.array([-2.0, 3.0]), signed=True, supports_mc=True),
                "sul": MetricResult(np.array([4.0, 5.0]), supports_mc=True),
                "gini_subcluster": MetricResult(
                    np.array([0.1, 0.2]),
                    per_cluster_metadata={
                        "internal_coverage_rate": np.array([0.8, 0.9]),
                        "internal_residue_n": np.array([1, 2]),
                    },
                ),
            },
            thresholds={"local_z": 1.5, "sul": 3.5},
            null_distributions={"local_z": np.array([1.0, 1.5]), "sul": np.array([2.0, 3.5])},
            effective_seeds={"local_z": 42, "sul": 43},
            seed=40,
            primary_metric="local_z",
            signif_level=0.005,
            n_alt_worlds=2,
            command="python src/main.py explain --dataset fixture",
            exploration_profile="none",
        )

    def test_round_trip_persists_all_points_metrics_thresholds_and_null_worlds(self):
        run_dir = self._write()
        snapshot = load_run_snapshot(run_dir, self.dataset)

        self.assertEqual(list(snapshot.assignments["point_id"]), [0, 1, 2, 3])
        self.assertEqual(list(snapshot.assignments["assignment_status"]),
                         ["assigned", "assigned", "assigned", "unassigned"])
        self.assertEqual(snapshot.assignments.loc[2, "origin"], "rescue")
        self.assertEqual(snapshot.assignments.loc[2, "origin_cluster_label"], 7)
        self.assertEqual(
            set(snapshot.scores["metric"]), {"local_z", "sul", "gini_subcluster"}
        )
        internal = snapshot.scores[snapshot.scores["metric"] == "gini_subcluster"]
        self.assertEqual(list(internal["internal_residue_n"]), [1.0, 2.0])
        self.assertEqual(len(snapshot.null_distributions), 4)
        self.assertEqual(
            set(snapshot.thresholds["effective_seed"].dropna().astype(int)), {42, 43}
        )
        self.assertTrue(snapshot.manifest["completion"]["experiment"])
        self.assertFalse(snapshot.manifest["completion"]["exploration_report"])

    def test_loader_rejects_incompatible_schema(self):
        run_dir = self._write()
        manifest_path = run_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["schema_version"] = 999
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "schema"):
            load_run_snapshot(run_dir, self.dataset)

    def test_loader_rejects_changed_dataset_file(self):
        run_dir = self._write()
        self.source.write_text(self.source.read_text() + "4,2,2,1\n")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            load_run_snapshot(run_dir, self.dataset)

    def test_loader_rejects_historical_run_without_snapshot(self):
        empty = self.root / "historical"
        empty.mkdir()
        with self.assertRaisesRegex(FileNotFoundError, "snapshot versionado"):
            load_run_snapshot(empty, self.dataset)


if __name__ == "__main__":
    unittest.main()
