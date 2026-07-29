import sys
import unittest
from pathlib import Path

import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from metrics.group_fairness import (
    calculate_gini,
    calculate_gini_contributions,
    calculate_meanvar,
    calculate_sul,
    classify_direction,
    get_signif_threshold,
    simulate_null_max_suls,
)


class MetricsTests(unittest.TestCase):
    def test_sul_returns_zero_for_empty_or_full_region(self):
        self.assertEqual(calculate_sul(0, 0, 10, 5), 0.0)
        self.assertEqual(calculate_sul(10, 5, 10, 5), 0.0)

    def test_sul_handles_extreme_probabilities(self):
        score = calculate_sul(3, 3, 10, 10)
        self.assertTrue(np.isfinite(score))
        self.assertEqual(score, 0.0)

    def test_meanvar_ignores_nan(self):
        score = calculate_meanvar([0.0, np.nan, 1.0])
        self.assertAlmostEqual(score, 0.25)

    def test_gini_is_zero_for_equal_rates(self):
        self.assertEqual(calculate_gini([0.5, 0.5, 0.5]), 0.0)

    def test_gini_known_value(self):
        self.assertAlmostEqual(calculate_gini([0.0, 1.0]), 0.5)

    def test_gini_ignores_nan(self):
        self.assertAlmostEqual(calculate_gini([0.2, np.nan, 0.4]), calculate_gini([0.2, 0.4]))

    def test_gini_degenerate_inputs(self):
        self.assertEqual(calculate_gini([]), 0.0)
        self.assertEqual(calculate_gini([0.0, 0.0]), 0.0)

    def test_gini_contributions_zero_for_equal_rates(self):
        contributions = calculate_gini_contributions([0.5, 0.5, 0.5])
        np.testing.assert_allclose(contributions, [0.0, 0.0, 0.0])

    def test_gini_contributions_extreme_pair(self):
        contributions = calculate_gini_contributions([0.0, 1.0])
        np.testing.assert_allclose(contributions, [0.5, 0.5])

    def test_gini_contributions_nan_stays_nan(self):
        contributions = calculate_gini_contributions([0.2, np.nan, 0.4])
        self.assertTrue(np.isnan(contributions[1]))
        self.assertTrue(np.isfinite(contributions[0]))
        self.assertTrue(np.isfinite(contributions[2]))

    def test_classify_direction(self):
        self.assertEqual(classify_direction(4, 0, 10, 5), "negative")
        self.assertEqual(classify_direction(4, 4, 10, 5), "positive")
        self.assertEqual(classify_direction(0, 0, 10, 5), "neutral")
        self.assertEqual(classify_direction(10, 5, 10, 5), "neutral")
        self.assertEqual(classify_direction(2, 1, 10, 5), "neutral")

    def test_simulate_null_max_suls_is_deterministic(self):
        regions = [{"points": [0, 1, 2]}, {"points": [3, 4]}]
        first = simulate_null_max_suls(5, regions, 10, 5, seed=7)
        second = simulate_null_max_suls(5, regions, 10, 5, seed=7)
        self.assertEqual(len(first), 5)
        np.testing.assert_array_equal(first, second)

    def test_threshold_matches_simulated_distribution(self):
        regions = [{"points": [0, 1, 2]}, {"points": [3, 4]}]
        signif_level, n_alt_worlds = 0.2, 10
        scores = simulate_null_max_suls(n_alt_worlds, regions, 10, 5, seed=7)
        ordered = np.sort(scores)[::-1]
        expected = float(ordered[min(int(signif_level * n_alt_worlds), len(ordered) - 1)])
        threshold = get_signif_threshold(signif_level, n_alt_worlds, regions, 10, 5, seed=7)
        self.assertEqual(threshold, expected)


if __name__ == "__main__":
    unittest.main()
