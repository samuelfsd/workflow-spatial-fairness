import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from clustering.internal import InternalSubdivision
from exploration import build_exploration_tables, partition_from_snapshot
from exploration_details import (
    InternalEvidence,
    analyze_cluster_internal,
    cluster_detail_figures,
    internal_evidence_figures,
    paginate_cluster_bundles,
    select_clusters,
    shared_histogram_bins,
    stratified_point_sample,
)
from tests.test_exploration import exploration_fixture


class SelectionTests(unittest.TestCase):
    def test_auto_selection_unions_detection_large_tail_and_tukey_reasons(self):
        frame = pd.DataFrame(
            {
                "cluster_label": range(10),
                "n": [10] * 9 + [100],
                "rho_in": [0.5] * 9 + [0.99],
                "internal_predominance": [0.0] * 9 + [0.98],
                "global_deviation": [0.0] * 9 + [0.49],
                "distance_mean_km": [1.0] * 9 + [10.0],
                "distance_p95_km": [2.0] * 9 + [20.0],
                "detection_class": ["negative"] + ["neutral"] * 9,
                "evaluation_status": ["avaliado"] * 10,
            }
        )
        selected = select_clusters(frame, "auto")

        self.assertEqual(set(selected["cluster_label"]), {0, 9})
        reasons = set(selected.loc[selected["cluster_label"] == 9, "reason"])
        self.assertIn("cluster_grande_leave_one_out", reasons)
        self.assertIn("tukey_high:rho_in", reasons)
        self.assertGreater(len(reasons), 2)

    def test_tukey_is_not_invented_below_eight_finite_values(self):
        frame = pd.DataFrame(
            {
                "cluster_label": range(7), "n": [10] * 6 + [100],
                "rho_in": [0.5] * 6 + [0.99], "internal_predominance": [0] * 7,
                "global_deviation": [0] * 7, "distance_mean_km": [1] * 7,
                "distance_p95_km": [2] * 7, "detection_class": ["neutral"] * 7,
                "evaluation_status": ["avaliado"] * 7,
            }
        )
        selected = select_clusters(frame, "auto")
        self.assertFalse(selected["reason"].str.startswith("tukey_").any())

    def test_explicit_and_all_selection_deduplicate_labels(self):
        frame = pd.DataFrame(
            {
                "cluster_label": [0, 1, 2], "n": [1, 1, 1], "rho_in": [0.5] * 3,
                "internal_predominance": [0] * 3, "global_deviation": [0] * 3,
                "distance_mean_km": [1] * 3, "distance_p95_km": [2] * 3,
                "detection_class": ["neutral"] * 3, "evaluation_status": ["avaliado"] * 3,
            }
        )
        explicit = select_clusters(frame, "2,2,1")
        self.assertEqual(set(explicit["cluster_label"]), {1, 2})
        self.assertEqual(set(select_clusters(frame, "all")["cluster_label"]), {0, 1, 2})


class DetailUtilityTests(unittest.TestCase):
    def test_shared_bins_and_stratified_sample_are_deterministic(self):
        positive = np.array([0.0, 1.0, 2.0, 3.0])
        negative = np.array([0.5, 1.5, 2.5, 3.5])
        bins = shared_histogram_bins(positive, negative)
        self.assertGreaterEqual(len(bins), 2)
        self.assertEqual(bins[0], 0.0)
        self.assertEqual(bins[-1], 3.5)

        points = list(range(100))
        outcomes = np.array([0] * 80 + [1] * 20)
        first = stratified_point_sample(points, outcomes, max_points=25, seed=7)
        second = stratified_point_sample(points, outcomes, max_points=25, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 25)
        self.assertIn(0, set(outcomes[first]))
        self.assertIn(1, set(outcomes[first]))

    def test_pagination_never_splits_a_cluster_bundle(self):
        index = paginate_cluster_bundles({1: 30, 2: 30, 3: 5}, target_pages=50)
        by_cluster = index.groupby("cluster_label")["volume"].nunique()
        self.assertTrue((by_cluster == 1).all())
        self.assertEqual(index.set_index("cluster_label").loc[1, "volume"], 1)
        self.assertEqual(index.set_index("cluster_label").loc[2, "volume"], 2)


class InternalEvidenceTests(unittest.TestCase):
    def test_components_close_parent_deviation_and_multiscale_overlap_is_factual(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset, snapshot = exploration_fixture(Path(tmp))
            tables = build_exploration_tables(dataset, snapshot, "local_z")
            partition = partition_from_snapshot(snapshot, dataset.n_total)
            original_labels = partition.labels.copy()

            result = analyze_cluster_internal(
                dataset, partition, tables.cluster_features, cluster_label=0,
                primary_metric="local_z",
                subdividers={
                    "g1": lambda points: InternalSubdivision([[0], [1]], [], 1, 2),
                    "g2": lambda points: InternalSubdivision([[0]], [1], 2, 2),
                },
            )

            for scale in ("g1", "g2"):
                components = result.components[result.components["scale"] == scale]
                parent = tables.cluster_features.set_index("cluster_label").loc[0]
                expected = parent["rho_in"] - parent["rho_peer"]
                self.assertAlmostEqual(components["signed_contribution"].sum(), expected)
            self.assertIn("jaccard", result.overlap.columns)
            self.assertIn("rho_change", result.overlap.columns)
            self.assertIn("residue_n_change", result.overlap.columns)
            self.assertIn("direction_changed", result.overlap.columns)
            self.assertNotIn("robust", result.overlap.columns)
            self.assertEqual(
                result.scale_summary.set_index("scale").loc[
                    "g2", "residue_n_change_from_g1"
                ],
                1,
            )
            np.testing.assert_array_equal(partition.labels, original_labels)

    def test_detail_histogram_is_percent_per_class_and_geography_declares_centroids(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset, snapshot = exploration_fixture(Path(tmp))
            tables = build_exploration_tables(dataset, snapshot, "local_z")
            partition = partition_from_snapshot(snapshot, dataset.n_total)
            figures = cluster_detail_figures(
                dataset, partition, tables.cluster_features,
                cluster_label=0, reasons=["selection_explicit"], seed=7,
            )
            try:
                distribution = dict(figures)["distribuicao"]
                self.assertEqual(
                    distribution.axes[0].get_ylabel(), "% de pontos dentro da classe"
                )
                self.assertNotIn("densidade", distribution.axes[0].get_title().lower())
                geography = dict(figures)["geografia"]
                labels = geography.axes[0].get_legend_handles_labels()[1]
                self.assertIn("centroide geral (população)", labels)
                self.assertIn("centroide positivo (população)", labels)
                self.assertIn("centroide negativo (população)", labels)
            finally:
                import matplotlib.pyplot as plt
                for _, figure in figures:
                    plt.close(figure)

    def test_empty_internal_scale_is_rendered_as_factual_status(self):
        evidence = InternalEvidence(
            components=pd.DataFrame(
                columns=["scale", "component", "rho", "is_residue", "n", "signed_contribution"]
            ),
            scale_summary=pd.DataFrame(
                [
                    {"scale": scale, "subdivision_status": "não subdividido nesta granularidade",
                     "internal_coverage_rate": 0.0, "gini_subcluster": float("nan"),
                     "rho_reference": 0.5, "reference_type": "peers"}
                    for scale in ("g1", "g2")
                ]
            ),
            overlap=pd.DataFrame(),
        )
        figures = internal_evidence_figures(evidence, cluster_label=9)
        try:
            text = " ".join(item.get_text() for item in figures[0][1].axes[0].texts)
            self.assertIn("não subdividido nesta granularidade", text)
        finally:
            import matplotlib.pyplot as plt
            for _, figure in figures:
                plt.close(figure)


if __name__ == "__main__":
    unittest.main()
