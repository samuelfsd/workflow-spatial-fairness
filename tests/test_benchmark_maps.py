import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from benchmark_maps import render_comparative_maps
from palette import CATEGORICAL, COLOR_NEGATIVE


class BenchmarkMapTests(unittest.TestCase):
    def test_maps_share_dataset_extent_and_meanvar_is_non_directional(self):
        geometry = json.dumps([[34.0, -119.0], [34.0, -118.0], [35.0, -118.0], [35.0, -119.0]])
        rows = []
        for metric in ("sul", "meanvar", "local_z"):
            rows.append({
                "dataset": "crime", "source": "local",
                "protocol": "standardized" if metric == "local_z" else "reproduction",
                "method": "hdbscan" if metric == "local_z" else "grid",
                "partitioning": "hdbscan" if metric == "local_z" else "grid_20x20",
                "params": "{}", "metric": metric, "region_id": 1,
                "geometry": geometry, "n": 10, "rho_in": .4, "rho_reference": .56,
                "significant": True if metric != "meanvar" else pd.NA,
                "direction": "negative" if metric != "meanvar" else None,
                "detection_class": "negative" if metric != "meanvar" else None,
                "evaluation_status": "evaluated" if metric != "meanvar" else "diagnostic",
                "score": 2.0,
            })
        with tempfile.TemporaryDirectory() as tmp:
            paths = render_comparative_maps(pd.DataFrame(), pd.DataFrame(rows), Path(tmp))
            self.assertEqual(len(paths), 3)
            html = "\n".join(path.read_text(encoding="utf-8") for path in paths)
            self.assertIn(COLOR_NEGATIVE, html)
            self.assertIn(CATEGORICAL[0], html)
            self.assertIn("ranking não direcional", html)
            self.assertIn("Extensão compartilhada", html)
            self.assertIn("taxa local", html)


if __name__ == "__main__":
    unittest.main()
