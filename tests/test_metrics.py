import sys
import unittest
from pathlib import Path

import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from metrics.group_fairness import calculate_meanvar, calculate_sul


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


if __name__ == "__main__":
    unittest.main()
