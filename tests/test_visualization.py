import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from clustering.base import Partition
from visualization import (
    _convex_hull,
    save_clustering_stage_map,
    save_detection_stage_map,
)


def _tiny_dataset() -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.DataFrame(
        {
            "lat": [0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 5.0],
            "lon": [0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 5.0],
        }
    )
    types = np.array([1, 0, 1, 0, 0, 0, 1])
    return df, types


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

    def test_clustering_stage_map_writes_html(self):
        df, types = _tiny_dataset()
        partition = Partition(
            method="hdbscan",
            params={"min_cluster_size": 3},
            labels=np.array([0, 0, 0, 1, 1, 1, -1]),
            regions=[
                {"points": [0, 1, 2], "cluster_label": 0},
                {"points": [3, 4, 5], "cluster_label": 1},
            ],
            noise_points=[6],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "stage1.html"
            save_clustering_stage_map(df, types, partition, output_path)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_cluster_origin_is_declared_in_stage_tooltips(self):
        df, types = _tiny_dataset()
        partition = Partition(
            method="hdbscan_rescue",
            params={},
            labels=np.array([0, 0, 0, 1, 1, 1, -1]),
            regions=[
                {"points": [0, 1, 2], "cluster_label": 0, "origin": "organic"},
                {"points": [3, 4, 5], "cluster_label": 1, "origin": "rescue"},
            ],
            noise_points=[6],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "stage1.html"
            save_clustering_stage_map(df, types, partition, output_path)
            html = output_path.read_text(encoding="utf-8")

        self.assertIn("origin=organic", html)
        self.assertIn("origin=rescue", html)

    def test_detection_stage_map_writes_html_with_status_colors(self):
        df, types = _tiny_dataset()
        region_results = [
            {
                "region": {"points": [0, 1, 2], "cluster_label": 0},
                "n": 3, "p": 0, "rho": 0.0, "rho_out": 0.75,
                "sul": 3.0, "significant": True, "direction": "negative",
            },
            {
                "region": {"points": [3, 4, 5], "cluster_label": 1},
                "n": 3, "p": 3, "rho": 1.0, "rho_out": 0.0,
                "sul": 2.5, "significant": True, "direction": "positive",
            },
            {
                "region": {"points": [5, 6], "cluster_label": 2},
                "n": 2, "p": 1, "rho": 0.5, "rho_out": 0.4,
                "sul": 0.1, "significant": False, "direction": "positive",
            },
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "stage4.html"
            save_detection_stage_map(
                df, types, region_results, output_path, threshold=1.0, global_rate=0.43
            )
            html = output_path.read_text(encoding="utf-8")
            self.assertIn("#d03b3b", html)
            self.assertIn("#0ca30c", html)
            self.assertIn("#bdbdbd", html)


if __name__ == "__main__":
    unittest.main()
