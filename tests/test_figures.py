import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from figures import (
    balance_figure,
    cluster_card_figure,
    close,
    dispersion_figure,
    metric_panels_figure,
    profile_figure,
    save_figure,
    save_pdf_report,
)
from palette import COLOR_NEGATIVE, COLOR_NEUTRAL, COLOR_POSITIVE


def _panel(name: str, values: list[float], **overrides) -> dict:
    panel = {
        "name": name,
        "labels": [0, 1, 2],
        "values": values,
        "directions": ["negative", "positive", "neutral"],
        "significant": [True, True, False],
        "threshold": 1.5,
        "analytic_threshold": 2.0,
        "signed": True,
        "caption": "legenda",
    }
    panel.update(overrides)
    return panel


def _cluster_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cluster_label": [0, 1, 2],
            "n": [10, 20, 30],
            "p": [5, 15, 10],
            "n_neg": [5, 5, 20],
            "rho": [0.5, 0.75, 1 / 3],
            "raio_medio_km": [1.0, 2.0, 3.0],
            "raio_p95_km": [1.5, 2.5, 3.5],
        }
    )


class MetricPanelsTests(unittest.TestCase):
    def test_one_bar_per_cluster_in_every_panel(self):
        fig = metric_panels_figure(
            [_panel("local_z", [-3.0, 4.0, 0.5]), _panel("sul", [10.0, 20.0, 1.0], signed=False)],
            dataset="fake",
            method="hdbscan",
        )
        try:
            self.assertEqual(len(fig.axes), 2)
            for ax in fig.axes:
                self.assertEqual(len(ax.patches), 3)
        finally:
            close(fig)

    def test_bar_colors_are_the_detection_classes(self):
        fig = metric_panels_figure([_panel("local_z", [-3.0, 4.0, 0.5])], dataset="fake", method="hdbscan")
        try:
            colors = [patch.get_facecolor() for patch in fig.axes[0].patches]
            from matplotlib.colors import to_rgba

            self.assertEqual(colors[0], to_rgba(COLOR_NEGATIVE))
            self.assertEqual(colors[1], to_rgba(COLOR_POSITIVE))
            # Third cluster is not significant => neutral, whatever its direction.
            self.assertEqual(colors[2], to_rgba(COLOR_NEUTRAL))
        finally:
            close(fig)

    def test_signed_metric_draws_both_threshold_signs(self):
        signed = metric_panels_figure([_panel("local_z", [-3.0, 4.0, 0.5])], dataset="f", method="m")
        unsigned = metric_panels_figure(
            [_panel("sul", [3.0, 4.0, 0.5], signed=False)], dataset="f", method="m"
        )
        try:
            # signed: +/- Monte Carlo, +/- analytic, plus the zero baseline.
            self.assertEqual(len(signed.axes[0].lines), 5)
            # unsigned: one Monte Carlo line, one analytic line, no zero baseline.
            self.assertEqual(len(unsigned.axes[0].lines), 2)
        finally:
            close(signed, unsigned)

    def test_missing_threshold_draws_no_reference_line(self):
        fig = metric_panels_figure(
            [_panel("gini", [0.1, 0.2, 0.3], signed=False, threshold=None, analytic_threshold=None)],
            dataset="f",
            method="m",
        )
        try:
            self.assertEqual(len(fig.axes[0].lines), 0)
        finally:
            close(fig)

    def test_nan_values_are_drawn_as_no_bar(self):
        fig = metric_panels_figure([_panel("local_z", [float("nan"), 4.0, 0.5])], dataset="f", method="m")
        try:
            self.assertEqual(fig.axes[0].patches[0].get_height(), 0.0)
        finally:
            close(fig)

    def test_empty_panel_list_is_rejected(self):
        with self.assertRaises(ValueError):
            metric_panels_figure([], dataset="f", method="m")


class BalanceAndDispersionTests(unittest.TestCase):
    def test_balance_stacks_positives_and_negatives_per_cluster(self):
        fig = balance_figure(_cluster_frame(), dataset="fake", method="hdbscan")
        try:
            # Two stacked series over three clusters.
            self.assertEqual(len(fig.axes[0].patches), 6)
            heights = [patch.get_height() for patch in fig.axes[0].patches]
            self.assertEqual(heights, [5.0, 15.0, 10.0, 5.0, 5.0, 20.0])
        finally:
            close(fig)

    def test_balance_avoids_the_detection_class_colors(self):
        # Point outcome is not a verdict: red/green belong to clusters only.
        from matplotlib.colors import to_rgba

        fig = balance_figure(_cluster_frame(), dataset="fake", method="hdbscan")
        try:
            reserved = {to_rgba(COLOR_NEGATIVE), to_rgba(COLOR_POSITIVE)}
            for patch in fig.axes[0].patches:
                self.assertNotIn(patch.get_facecolor(), reserved)
        finally:
            close(fig)

    def _profile(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "config": ["hdbscan", "hdbscan cap=1000", "capped_hdbscan cap=1000"],
                "cluster_size_cv": [1.09, 0.67, 0.76],
                "noise_rate": [0.209, 0.801, 0.209],
                "rho_sigma": [0.0715, 0.0712, 0.0772],
                "raio_medio_km_mean": [25.0, 9.0, 18.0],
            }
        )

    def test_profile_draws_one_panel_per_reading_and_one_bar_per_config(self):
        fig = profile_figure(self._profile(), dataset="lar")
        try:
            visible = [ax for ax in fig.axes if ax.get_visible()]
            self.assertEqual(len(visible), 4)
            for ax in visible:
                self.assertEqual(len(ax.patches), 3)
        finally:
            close(fig)

    def test_profile_keeps_each_reading_on_its_own_axis(self):
        # A CV and a percentage share no scale: separate panels, never a second
        # y-axis on one panel.
        fig = profile_figure(self._profile(), dataset="lar")
        try:
            for ax in fig.axes:
                self.assertEqual(len(ax.get_shared_y_axes().get_siblings(ax)), 1)
        finally:
            close(fig)

    def test_profile_skips_readings_the_frame_does_not_carry(self):
        frame = self._profile()[["config", "cluster_size_cv", "noise_rate"]]
        fig = profile_figure(frame, dataset="lar")
        try:
            self.assertEqual(len([ax for ax in fig.axes if ax.get_visible()]), 2)
        finally:
            close(fig)

    def test_profile_rejects_an_empty_profile(self):
        with self.assertRaises(ValueError):
            profile_figure(pd.DataFrame({"config": []}), dataset="lar")

    def test_dispersion_draws_one_group_per_variable_and_config(self):
        from descriptives import compare_configs

        table = compare_configs({"hdbscan": _cluster_frame(), "capped": _cluster_frame()})
        fig = dispersion_figure(table, dataset="fake")
        try:
            # 5 profiled variables x 2 configurations.
            self.assertEqual(len(fig.axes[0].patches), 10)
        finally:
            close(fig)


class ClusterCardTests(unittest.TestCase):
    def _card(self, **overrides) -> dict:
        card = {
            "cluster_label": 7,
            "n": 100,
            "p": 40,
            "n_neg": 60,
            "rho_in": 0.4,
            "rho_peer": 0.65,
            "rho_global": 0.5,
            "gini_subcluster": 0.12,
            "raio_medio_km": 3.2,
            "subclusters": pd.DataFrame(
                {
                    "component": ["subcluster", "subcluster", "subcluster"],
                    "subcluster": [0, 1, 2],
                    "n": [50, 30, 20],
                    "p": [25, 12, 3],
                    "n_neg": [25, 18, 17],
                    "rho": [0.5, 0.4, 0.15],
                }
            ),
            "internal_coverage_rate": 1.0,
            "residue_n": 0,
            "homogeneous": False,
        }
        card.update(overrides)
        return card

    def test_one_bar_per_subcluster_sorted_by_rate(self):
        fig = cluster_card_figure(self._card(), dataset="fake", granularity="25")
        try:
            heights = [patch.get_height() for patch in fig.axes[0].patches]
            self.assertEqual(heights, sorted(heights))
            self.assertEqual(len(heights), 3)
        finally:
            close(fig)

    def test_three_reference_lines_when_all_rates_exist(self):
        fig = cluster_card_figure(self._card(), dataset="fake", granularity="25")
        try:
            self.assertEqual(len(fig.axes[0].lines), 3)
        finally:
            close(fig)

    def test_peer_line_is_skipped_when_the_peer_rate_is_unavailable(self):
        fig = cluster_card_figure(self._card(rho_peer=float("nan")), dataset="f", granularity="25")
        try:
            self.assertEqual(len(fig.axes[0].lines), 2)
        finally:
            close(fig)

    def test_homogeneous_cluster_is_stated_in_the_title(self):
        card = self._card(
            homogeneous=True,
            gini_subcluster=0.0,
            subclusters=pd.DataFrame(
                {
                    "component": ["subcluster"],
                    "subcluster": [0],
                    "n": [100],
                    "p": [40],
                    "n_neg": [60],
                    "rho": [0.4],
                }
            ),
        )
        fig = cluster_card_figure(card, dataset="fake", granularity="25")
        try:
            self.assertIn("homogêneo por dentro", fig.axes[0].get_title(loc="left"))
            self.assertEqual(len(fig.axes[0].patches), 1)
        finally:
            close(fig)


class OutputTests(unittest.TestCase):
    def test_each_figure_is_written_as_png_and_pdf(self):
        fig = balance_figure(_cluster_frame(), dataset="fake", method="hdbscan")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                written = save_figure(fig, Path(tmp) / "figures" / "balance")
                self.assertEqual([path.suffix for path in written], [".png", ".pdf"])
                for path in written:
                    self.assertTrue(path.exists())
                    self.assertGreater(path.stat().st_size, 0)
        finally:
            close(fig)

    def test_multipage_report_has_one_page_per_figure(self):
        first = balance_figure(_cluster_frame(), dataset="fake", method="hdbscan")
        second = metric_panels_figure([_panel("local_z", [-3.0, 4.0, 0.5])], dataset="f", method="m")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = save_pdf_report([first, second], Path(tmp) / "report.pdf")
                self.assertTrue(path.exists())
                raw = path.read_bytes()
                # The page-tree node declares how many pages it holds.
                self.assertIn(b"/Count 2", raw)
                self.assertEqual(raw.count(b"/Type /Page") - raw.count(b"/Type /Pages"), 2)
        finally:
            close(first, second)


if __name__ == "__main__":
    unittest.main()
