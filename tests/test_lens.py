import sys
import unittest
from pathlib import Path

import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from lens import GREATER_LA_BBOX, clusters_in_bbox


class LensTests(unittest.TestCase):
    def test_selects_only_clusters_centroid_inside_bbox(self):
        # Cluster 0 in greater LA (~34.05, -118.24); cluster 1 in New Mexico.
        df = pd.DataFrame(
            {
                "lat": [34.0, 34.1, 35.1, 35.1],
                "lon": [-118.3, -118.2, -106.6, -106.6],
            }
        )
        regions = [
            {"points": [0, 1], "cluster_label": 0},
            {"points": [2, 3], "cluster_label": 1},
        ]

        self.assertEqual(clusters_in_bbox(regions, df, GREATER_LA_BBOX), [0])


if __name__ == "__main__":
    unittest.main()
