import math
import sys
import unittest
from pathlib import Path

import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from clustering.base import Partition
from clustering.internal import InternalSubdivision, subdivision_from_labels
from metrics.base import MetricContext
from metrics.registry import get_metric


class InternalSubdivisionTests(unittest.TestCase):
    def test_preserves_condensed_subclusters_and_residue_without_reassignment(self):
        result = subdivision_from_labels(
            points=[10, 11, 12, 13, 14, 15],
            labels=np.array([0, 0, -1, 1, -1, 1]),
            min_cluster_size=2,
        )

        self.assertEqual(result.subclusters, [[10, 11], [13, 15]])
        self.assertEqual(result.residue, [12, 14])
        self.assertEqual(result.condensed_n, 4)
        self.assertAlmostEqual(result.coverage_rate, 4 / 6)
        self.assertEqual(result.status, "subdividido")
        self.assertEqual(
            sorted(
                [point for group in result.subclusters for point in group]
                + result.residue
            ),
            [10, 11, 12, 13, 14, 15],
        )

    def test_zero_condensed_subclusters_is_missing_gini(self):
        types = np.array([0, 1, 0, 1])
        subdivision = InternalSubdivision(
            subclusters=[], residue=[0, 1, 2, 3], min_cluster_size=2, parent_n=4
        )
        result = get_metric("gini_subcluster")(
            _partition(), types, _context(types, subdivision)
        )

        self.assertTrue(math.isnan(result.per_cluster[0]))
        self.assertEqual(subdivision.status, "não subdividido nesta granularidade")
        self.assertEqual(subdivision.coverage_rate, 0.0)

    def test_one_condensed_subcluster_is_zero_gini_and_not_subdivided(self):
        types = np.array([0, 1, 1, 0])
        subdivision = InternalSubdivision(
            subclusters=[[0, 1, 2]], residue=[3], min_cluster_size=2, parent_n=4
        )
        result = get_metric("gini_subcluster")(
            _partition(), types, _context(types, subdivision)
        )

        self.assertEqual(result.per_cluster[0], 0.0)
        self.assertEqual(subdivision.status, "não subdividido nesta granularidade")
        self.assertAlmostEqual(subdivision.coverage_rate, 0.75)

    def test_two_condensed_subclusters_exclude_residue_from_gini(self):
        # Condensed rates are 0 and 1 => Gini 0.5. The positive residue would
        # change that value if it were reassigned or treated as a third group.
        types = np.array([0, 0, 1, 1, 1])
        subdivision = InternalSubdivision(
            subclusters=[[0, 1], [2, 3]], residue=[4], min_cluster_size=2, parent_n=5
        )
        result = get_metric("gini_subcluster")(
            _partition(5), types, _context(types, subdivision)
        )

        self.assertAlmostEqual(result.per_cluster[0], 0.5)
        self.assertEqual(subdivision.residue, [4])
        self.assertEqual(result.per_cluster_metadata["internal_residue_n"][0], 1)
        self.assertAlmostEqual(
            result.per_cluster_metadata["internal_coverage_rate"][0], 0.8
        )
        self.assertEqual(
            result.per_cluster_metadata["internal_subdivision_status"][0],
            "subdividido",
        )


def _partition(n: int = 4) -> Partition:
    return Partition(
        method="fake",
        params={"min_cluster_size": 2},
        labels=np.zeros(n, dtype=int),
        regions=[{"points": list(range(n)), "cluster_label": 0}],
    )


def _context(types: np.ndarray, subdivision: InternalSubdivision) -> MetricContext:
    return MetricContext(
        n_total=len(types),
        p_total=int(types.sum()),
        internal_subdivider=lambda points: subdivision,
    )


if __name__ == "__main__":
    unittest.main()
