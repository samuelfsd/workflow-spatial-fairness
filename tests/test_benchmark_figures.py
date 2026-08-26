import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from benchmark_figures import (
    _key_detectors,
    _summary,
    build_benchmark_figures,
    render_benchmark_figures,
)


class BenchmarkFigureTests(unittest.TestCase):
    def setUp(self):
        self.canonical = pd.DataFrame([
            {"dataset": "semisynth", "source": "local", "protocol": "reproduction", "experiment": "synthetic_control", "method": "grid", "region_system": "grid_20x20", "metric": "sul", "quantity": "unfairness_detected", "value": False, "unit": "boolean"},
            {"dataset": "synth_unfair", "source": "local", "protocol": "reproduction", "experiment": "synthetic_control", "method": "grid", "region_system": "grid_20x20", "metric": "sul", "quantity": "unfairness_detected", "value": True, "unit": "boolean"},
            {"dataset": "semisynth", "source": "local", "protocol": "reproduction", "experiment": "synthetic_control", "method": "random_grid", "region_system": "random_grids", "metric": "meanvar", "quantity": "partition_score", "value": .052, "unit": "score"},
            {"dataset": "lar", "source": "local", "protocol": "reproduction", "experiment": "fixed_grid", "method": "grid", "region_system": "grid_100x50", "metric": "sul", "quantity": "global_rate", "value": .62, "unit": "rate"},
            {"dataset": "lar", "source": "local", "protocol": "reproduction", "experiment": "fixed_grid", "method": "grid", "region_system": "grid_100x50", "metric": "sul", "quantity": "best_region_rate", "value": .84, "unit": "rate"},
            {"dataset": "lar", "source": "local", "protocol": "reproduction", "experiment": "fixed_grid", "method": "grid", "region_system": "grid_100x50", "metric": "sul", "quantity": "best_region_n", "value": 8000, "unit": "count"},
            {"dataset": "crime", "source": "local", "protocol": "reproduction", "experiment": "fixed_grid", "method": "grid", "region_system": "grid_20x20", "metric": "sul", "quantity": "global_rate", "value": .56, "unit": "rate"},
            {"dataset": "crime", "source": "local", "protocol": "reproduction", "experiment": "fixed_grid", "method": "grid", "region_system": "grid_20x20", "metric": "sul", "quantity": "best_region_rate", "value": .51, "unit": "rate"},
            {"dataset": "crime", "source": "local", "protocol": "reproduction", "experiment": "fixed_grid", "method": "grid", "region_system": "grid_20x20", "metric": "sul", "quantity": "best_region_n", "value": 3000, "unit": "count"},
            {"dataset": "lar", "source": "local", "protocol": "reproduction", "experiment": "unrestricted_scan", "method": "kmeans_scan", "region_system": "kmeans_square_scan", "metric": "sul", "quantity": "candidate_regions", "value": 2000, "unit": "count"},
            {"dataset": "lar", "source": "local", "protocol": "reproduction", "experiment": "unrestricted_scan", "method": "kmeans_scan", "region_system": "kmeans_square_scan", "metric": "sul", "quantity": "significant_regions", "value": 700, "unit": "count"},
            {"dataset": "lar", "source": "local", "protocol": "reproduction", "experiment": "unrestricted_scan", "method": "kmeans_scan", "region_system": "kmeans_square_scan", "metric": "sul", "quantity": "consolidated_regions", "value": 28, "unit": "count"},
            {"dataset": "lar", "source": "local", "protocol": "standardized", "experiment": "standardized_comparison", "method": "hdbscan", "region_system": "hdbscan_frac_0.005", "metric": "peer_gini_gap", "quantity": "best_region_rate", "value": .70, "unit": "rate"},
            {"dataset": "lar", "source": "local", "protocol": "standardized", "experiment": "standardized_comparison", "method": "hdbscan", "region_system": "hdbscan_frac_0.005", "metric": "peer_gini_gap", "quantity": "best_reference_rate", "value": .60, "unit": "rate"},
            {"dataset": "lar", "source": "local", "protocol": "standardized", "experiment": "standardized_comparison", "method": "hdbscan", "region_system": "hdbscan_frac_0.005", "metric": "peer_gini_gap", "quantity": "candidate_regions", "value": 41, "unit": "count"},
        ])

    def test_three_figures_have_pt_br_titles(self):
        figures = build_benchmark_figures({"canonical": self.canonical})
        self.assertEqual(len(figures), 3)
        titles = " ".join(axis.get_title() for _, fig in figures for axis in fig.axes)
        self.assertIn("Controles sintéticos", titles)
        self.assertIn("Taxa local", titles)
        self.assertIn("Carga", titles)

    def test_synthetic_matrix_uses_wrapped_horizontal_detector_labels(self):
        _, figure = build_benchmark_figures({"canonical": self.canonical})[0]
        ticks = figure.axes[0].get_xticklabels()
        self.assertTrue(all(tick.get_rotation() == 0 for tick in ticks))
        self.assertTrue(any("\n" in tick.get_text() for tick in ticks))

    def test_figures_aggregate_long_tables_instead_of_labeling_every_record(self):
        noisy_rows = [
            {
                "dataset": "semisynth", "source": "local",
                "protocol": "reproduction", "experiment": "synthetic_control",
                "method": "random_grid", "region_system": f"random_grid_{idx}",
                "metric": "meanvar", "quantity": "partition_score",
                "value": .04 + idx / 10000, "unit": "score",
            }
            for idx in range(100)
        ]
        canonical = pd.concat(
            [self.canonical, pd.DataFrame(noisy_rows)], ignore_index=True
        )

        figures = build_benchmark_figures({"canonical": canonical})

        for _, figure in figures:
            for axis in figure.axes:
                visible_x = [
                    label.get_text() for label in axis.get_xticklabels()
                    if label.get_visible() and label.get_text()
                ]
                visible_y = [
                    label.get_text() for label in axis.get_yticklabels()
                    if label.get_visible() and label.get_text()
                ]
                self.assertLessEqual(len(visible_x), 10)
                self.assertLessEqual(len(visible_y), 10)

    def test_workload_uses_short_method_labels_on_the_y_axis(self):
        _, workload = build_benchmark_figures({"canonical": self.canonical})[-1]
        labels = [label.get_text() for label in workload.axes[0].get_yticklabels()]
        self.assertTrue(any("HDBSCAN" in label or "Varredura" in label for label in labels))
        self.assertFalse(any("reproduction" in label for label in labels))

    def test_rate_figure_does_not_plot_gini_as_if_it_were_a_rate_reference(self):
        _, rate_figure = build_benchmark_figures({"canonical": self.canonical})[1]
        labels = [
            label.get_text()
            for axis in rate_figure.axes
            for label in axis.get_yticklabels()
        ]
        self.assertFalse(any("Gini" in label for label in labels))

    def test_presentation_uses_one_standardized_protocol_when_available(self):
        standardized = self.canonical.iloc[[3]].copy()
        standardized["protocol"] = "standardized"
        canonical = pd.concat([self.canonical, standardized], ignore_index=True)

        selected = _key_detectors(_summary(canonical), "lar")

        self.assertEqual(set(selected["protocol"]), {"standardized"})

    def test_renderer_writes_png_pdf_and_multipage(self):
        with tempfile.TemporaryDirectory() as tmp:
            figures = render_benchmark_figures({"canonical": self.canonical}, Path(tmp))
            self.assertEqual(len(figures), 3)
            self.assertEqual(len(list(Path(tmp).glob("*.png"))), 3)
            self.assertEqual(len([p for p in Path(tmp).glob("*.pdf") if p.name != "benchmark_quantitativo.pdf"]), 3)
            self.assertTrue((Path(tmp) / "benchmark_quantitativo.pdf").exists())


if __name__ == "__main__":
    unittest.main()
