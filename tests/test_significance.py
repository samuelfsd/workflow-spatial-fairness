import sys
import unittest
from pathlib import Path

import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from clustering.base import Partition
from metrics.base import MetricContext
from metrics.builtin import sul_metric
from metrics.group_fairness import simulate_null_max_suls
from metrics.significance import (
    analytic_threshold,
    significance_threshold,
    simulate_null_metric,
)


def _partition() -> Partition:
    regions = [
        {"points": [0, 1, 2], "cluster_label": 0},
        {"points": [3, 4], "cluster_label": 1},
    ]
    return Partition(method="fake", params={}, labels=np.zeros(10, dtype=int), regions=regions)


def _ctx() -> MetricContext:
    return MetricContext(n_total=10, p_total=5, adjacency={0: [1], 1: [0]})


class SignificanceTests(unittest.TestCase):
    def test_generic_null_reproduces_sul_monte_carlo(self):
        # New metric-driven engine with the SUL metric must match the existing
        # SUL-specific Monte Carlo exactly (same seed => same alternate worlds).
        partition = _partition()
        expected = simulate_null_max_suls(20, partition.regions, 10, 5, seed=7)
        actual = simulate_null_metric(sul_metric, partition, _ctx(), 20, 10, 5, seed=7)
        np.testing.assert_allclose(actual, expected)

    def test_null_is_deterministic(self):
        partition = _partition()
        first = simulate_null_metric(sul_metric, partition, _ctx(), 5, 10, 5, seed=3)
        second = simulate_null_metric(sul_metric, partition, _ctx(), 5, 10, 5, seed=3)
        np.testing.assert_array_equal(first, second)

    def test_threshold_is_upper_quantile_of_null(self):
        scores = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        # signif_level 0.2, w=5 => index int(0.2*5)=1 into descending sort => 4.0
        self.assertEqual(significance_threshold(0.2, scores), 4.0)

    def test_threshold_empty_null_is_zero(self):
        self.assertEqual(significance_threshold(0.005, np.array([])), 0.0)

    def test_analytic_threshold_matches_hand_computed_sidak_band(self):
        # One cluster, two-sided, alpha 5% => the familiar 1.96 sigma band.
        self.assertAlmostEqual(analytic_threshold(0.05, 1), 1.959964, places=5)
        # 42 clusters at a global 0.5%: per-test alpha = 1-(1-0.005)^(1/42),
        # split across two tails => z ~ 3.847 (the LAR reference value).
        self.assertAlmostEqual(analytic_threshold(0.005, 42), 3.847, places=3)

    def test_analytic_threshold_grows_with_the_number_of_clusters(self):
        # More regions scanned => stricter bar, which is the whole point of the
        # multiple-testing correction.
        few = analytic_threshold(0.005, 5)
        many = analytic_threshold(0.005, 500)
        self.assertLess(few, many)

    def test_analytic_threshold_is_nan_without_clusters(self):
        self.assertTrue(np.isnan(analytic_threshold(0.005, 0)))


if __name__ == "__main__":
    unittest.main()
