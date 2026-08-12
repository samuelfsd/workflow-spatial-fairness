import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from exploration import build_exploration_tables
from exploration_supplements import factual_summary, spearman_pairwise
from metric_comparison import compare_primary_metrics
from tests.test_exploration import exploration_fixture


class SupplementTests(unittest.TestCase):
    def test_spearman_requires_ten_pairwise_values_and_has_no_p_values(self):
        small = pd.DataFrame({"a": range(9), "b": range(9)})
        self.assertTrue(spearman_pairwise(small, ["a", "b"]).empty)
        enough = pd.DataFrame({"a": range(10), "b": range(9, -1, -1)})
        result = spearman_pairwise(enough, ["a", "b"])
        self.assertEqual(result.iloc[0]["n_pair"], 10)
        self.assertAlmostEqual(result.iloc[0]["spearman_rho"], -1.0)
        self.assertNotIn("p_value", result.columns)

    def test_summary_is_structured_as_facts_and_caveats_only(self):
        summary = factual_summary(
            pd.DataFrame({"cluster_label": [0], "n": [10], "evidence_ratio": [2.0]}),
            pd.DataFrame({"cluster_label": [0], "reason": ["detecção"]}),
        )
        self.assertEqual(set(summary), {"facts", "rankings", "selection_criteria", "caveats"})
        serialized = str(summary).lower()
        for forbidden in ("é causado por", "método superior", "foi comprovado"):
            self.assertNotIn(forbidden, serialized)


class PrimaryComparisonTests(unittest.TestCase):
    def test_alternative_primary_changes_only_derived_reading(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset, snapshot = exploration_fixture(Path(tmp))
            scores_before = snapshot.scores.copy(deep=True)
            comparison = compare_primary_metrics(
                dataset, snapshot, "local_z", "sul"
            )

            self.assertEqual(
                set(comparison.sets["set"]),
                {"ambas", "somente_local_z", "somente_sul", "nenhuma"},
            )
            self.assertIn("local_z_evidence_ratio", comparison.joint.columns)
            self.assertIn("sul_evidence_ratio", comparison.joint.columns)
            self.assertNotIn("hybrid_detection_class", comparison.joint.columns)
            pd.testing.assert_frame_equal(snapshot.scores, scores_before)


if __name__ == "__main__":
    unittest.main()
