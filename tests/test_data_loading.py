import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from data_loading import DATASET_SPECS, load_dataset


class DataLoadingTests(unittest.TestCase):
    def test_lar_action_taken_three_becomes_zero(self):
        dataset = load_dataset("lar")

        self.assertEqual(dataset.n_total, len(dataset.df))
        self.assertTrue(set(dataset.types).issubset({0, 1}))
        self.assertGreater((dataset.df["action_taken"] == 3).sum(), 0)
        self.assertEqual(dataset.types[dataset.df["action_taken"].to_numpy() == 3].sum(), 0)

    def test_lar_default_radii_have_twenty_values(self):
        spec = DATASET_SPECS["lar"]
        dataset = load_dataset("lar")

        self.assertEqual(spec.fixed_grids, ((100, 50), (25, 12)))
        self.assertEqual(len(dataset.radii), 20)
        self.assertAlmostEqual(float(dataset.radii[0]), 0.05)
        self.assertAlmostEqual(float(dataset.radii[-1]), 1.0)


if __name__ == "__main__":
    unittest.main()
