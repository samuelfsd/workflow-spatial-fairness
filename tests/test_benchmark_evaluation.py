import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from benchmark_evaluation import evaluate_partition, evaluate_scan
from clustering.base import Partition
from data_loading import DatasetSpec, LoadedDataset


def fixture_dataset(name: str = "fixture") -> LoadedDataset:
    df = pd.DataFrame(
        {
            "lat": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            "lon": [0.0, 0.1, 1.0, 1.1, 0.0, 0.1, 1.0, 1.1],
            "label": [1, 1, 0, 0, 1, 0, 1, 0],
            "outcome": [1, 1, 0, 0, 1, 0, 1, 0],
        }
    )
    spec = DatasetSpec(
        name, "fixture.csv", "label", 0.1, 0.3, 0.1, ((2, 2),),
        "positivo", "negativo", "não declarada",
    )
    types = df["outcome"].to_numpy(dtype=int)
    return LoadedDataset(
        name=name,
        df=df,
        types=types,
        n_total=len(df),
        p_total=int(types.sum()),
        radii=np.array([0.1, 0.2]),
        fixed_grids=((2, 2),),
        spec=spec,
        source_path=Path("fixture.csv"),
        source_sha256="fixture-sha",
        rows_before_clean=len(df),
    )


def fixture_partition() -> Partition:
    regions = [
        {"points": [0, 1], "cluster_label": 0},
        {"points": [2, 3], "cluster_label": 1},
        {"points": [4, 5], "cluster_label": 2},
        {"points": [6, 7], "cluster_label": 3},
    ]
    return Partition(
        method="fixture_partition",
        params={"kind": "fixture"},
        labels=np.repeat(np.arange(4), 2),
        regions=regions,
        noise_points=[],
    )


class PartitionEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = evaluate_partition(
            fixture_dataset(),
            fixture_partition(),
            metrics=(
                "sul", "local_z", "peer_rate_difference",
                "peer_log_rate_ratio", "peer_gini_gap", "meanvar",
            ),
            protocol="standardized",
            n_alt_worlds=20,
            signif_level=0.05,
            seed=42,
        )

    def test_partition_metrics_share_one_partition_but_keep_capabilities(self):
        summary = self.bundle.summary.set_index("metric")

        self.assertEqual(set(summary.index), {
            "sul", "local_z", "peer_rate_difference",
            "peer_log_rate_ratio", "peer_gini_gap", "meanvar",
        })
        self.assertEqual(set(summary["partitioning"]), {"fixture_partition"})
        self.assertGreater(summary.loc["sul", "threshold"], 0)
        self.assertGreater(summary.loc["local_z", "threshold"], 0)
        self.assertGreater(summary.loc["peer_rate_difference", "threshold"], 0)
        self.assertGreater(summary.loc["peer_log_rate_ratio", "threshold"], 0)
        self.assertEqual(
            summary.loc["peer_rate_difference", "evaluation_mode"],
            "candidate_calibrated",
        )
        self.assertTrue(pd.isna(summary.loc["meanvar", "threshold"]))
        self.assertEqual(summary.loc["meanvar", "evaluation_mode"], "diagnostic")
        self.assertAlmostEqual(summary.loc["meanvar", "partition_score"], 0.125)

    def test_region_records_declare_reference_direction_and_point_ids(self):
        regions = self.bundle.regions
        sul = regions[regions["metric"] == "sul"]
        local = regions[regions["metric"] == "local_z"]
        meanvar = regions[regions["metric"] == "meanvar"]
        difference = regions[regions["metric"] == "peer_rate_difference"]
        gini_gap = regions[regions["metric"] == "peer_gini_gap"]

        self.assertEqual(set(sul["rate_reference"]), {"outside"})
        self.assertEqual(set(local["rate_reference"]), {"peers"})
        self.assertEqual(set(meanvar["rate_reference"]), {"partition_mean"})
        self.assertEqual(set(difference["rate_reference"]), {"peers"})
        self.assertEqual(set(difference["indicator_name"]), {"positive_rate"})
        self.assertTrue(difference["significant"].notna().all())
        self.assertTrue(difference["detection_class"].isna().all())
        self.assertEqual(set(gini_gap["indicator_name"]), {"internal_gini"})
        self.assertTrue(gini_gap["direction"].isna().all())
        self.assertEqual(sul.iloc[0]["direction"], "positive")
        self.assertEqual(sul.iloc[1]["direction"], "negative")
        self.assertEqual(sul.iloc[0]["point_ids"], "[0, 1]")

    def test_crime_uses_tpr_semantics_without_model_accuracy(self):
        bundle = evaluate_partition(
            fixture_dataset("crime"),
            fixture_partition(),
            metrics=("sul",),
            protocol="reproduction",
            n_alt_worlds=20,
            signif_level=0.05,
            seed=42,
        )

        self.assertEqual(bundle.summary.loc[0, "rate_semantics"], "TPR")
        self.assertNotIn("accuracy", bundle.summary.columns)


class ScanEvaluationTests(unittest.TestCase):
    def test_scan_keeps_candidate_significant_and_consolidated_counts(self):
        dataset = fixture_dataset()
        regions = [
            {"points": [0, 1], "center": 0, "radius": 0.2},
            {"points": [0, 1], "center": 0, "radius": 0.3},
            {"points": [2, 3], "center": 2, "radius": 0.2},
        ]

        bundle = evaluate_scan(
            dataset,
            regions,
            protocol="reproduction",
            n_alt_worlds=20,
            signif_level=0.05,
            seed=42,
        )

        row = bundle.summary.iloc[0]
        self.assertEqual(row["candidate_regions"], 3)
        self.assertEqual(row["significant_regions"], int(bundle.regions["significant"].sum()))
        self.assertEqual(row["consolidated_regions"], int(bundle.regions["consolidated"].sum()))
        self.assertLessEqual(row["consolidated_regions"], row["significant_regions"])
        self.assertEqual(row["coverage"], 0.5)
        self.assertEqual(row["noise_n"], 4)
        self.assertEqual(set(bundle.regions["metric"]), {"sul"})


if __name__ == "__main__":
    unittest.main()
