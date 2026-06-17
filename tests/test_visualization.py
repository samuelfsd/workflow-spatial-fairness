import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from visualization import _convex_hull


class VisualizationTests(unittest.TestCase):
    def test_convex_hull_excludes_interior_points(self):
        points = [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
            (0.5, 0.5),
        ]

        hull = _convex_hull(points)

        self.assertEqual(len(hull), 4)
        self.assertNotIn((0.5, 0.5), hull)
        self.assertEqual(set(hull), {(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)})


if __name__ == "__main__":
    unittest.main()
