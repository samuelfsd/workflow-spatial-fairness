import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from synth_benchmark import (
    EFFECTS_PP,
    SCENARIOS,
    build_markdown_summary,
    generate_effect_case,
    run_effect_sweep,
    write_benchmark_artifacts,
)


class EffectCaseTests(unittest.TestCase):
    def test_all_declared_scenarios_keep_the_same_geometry_and_global_rate(self):
        reference = generate_effect_case("fair", effect_pp=0, seed=42)

        self.assertEqual(SCENARIOS, ("fair", "global", "local", "both"))
        self.assertEqual(EFFECTS_PP, (5, 10, 20, 30))
        self.assertEqual(len(reference), 41_600)
        self.assertAlmostEqual(reference["label"].mean(), 0.5)

        for scenario in SCENARIOS[1:]:
            for effect_pp in EFFECTS_PP:
                case = generate_effect_case(scenario, effect_pp=effect_pp, seed=42)
                np.testing.assert_array_equal(case["lat"], reference["lat"])
                np.testing.assert_array_equal(case["lon"], reference["lon"])
                self.assertAlmostEqual(case["label"].mean(), 0.5)

    def test_local_and_global_targets_encode_the_declared_rate_contrasts(self):
        local = generate_effect_case("local", effect_pp=20, seed=7)
        global_case = generate_effect_case("global", effect_pp=20, seed=7)

        local_target = local[local["target_local"]]
        local_peers = local[local["role"] == "local_peer"]
        self.assertAlmostEqual(local_target["label"].mean(), 0.5)
        self.assertAlmostEqual(local_peers["label"].mean(), 0.7)

        global_target = global_case[global_case["target_global"]]
        global_peers = global_case[global_case["role"] == "global_peer"]
        self.assertAlmostEqual(global_target["label"].mean(), 0.3)
        self.assertAlmostEqual(global_peers["label"].mean(), 0.3)
        self.assertAlmostEqual(local[local["role"] == "filler"]["label"].mean(), 0.47)
        self.assertAlmostEqual(
            global_case[global_case["role"] == "filler"]["label"].mean(),
            0.525,
        )

    def test_invalid_scenario_or_effect_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "scenario"):
            generate_effect_case("unknown", effect_pp=10)
        with self.assertRaisesRegex(ValueError, "effect_pp"):
            generate_effect_case("local", effect_pp=7)


class EffectSweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results, cls.metadata = run_effect_sweep(
            effects_pp=(20,),
            n_alt_worlds=20,
            signif_level=0.05,
            seed=42,
        )

    def _row(self, scenario: str, target: str, metric: str):
        rows = self.results[
            (self.results["scenario"] == scenario)
            & (self.results["target"] == target)
            & (self.results["metric"] == metric)
        ]
        self.assertEqual(len(rows), 1)
        return rows.iloc[0]

    def test_sweep_reuses_one_partition_and_reports_spatial_recovery(self):
        self.assertEqual(self.metadata["n_total"], 41_600)
        self.assertGreaterEqual(self.metadata["n_clusters"], 45)
        self.assertEqual(self.metadata["n_alt_worlds"], 20)
        self.assertGreater(self.metadata["thresholds"]["local_z"], 0)
        self.assertGreater(self.metadata["thresholds"]["sul"], 0)
        self.assertAlmostEqual(
            self.metadata["case_rates"]["local_20"]["filler"],
            0.47,
        )

        target = self._row("local", "local", "local_z")
        self.assertEqual(target["target_n"], 800)
        self.assertGreater(target["partition_target_recall"], 0.9)
        self.assertGreater(target["partition_target_iou"], 0.9)

    def test_detector_recovery_is_zero_when_the_target_cluster_is_not_detected(self):
        local_target = self._row("local", "local", "sul")
        self.assertEqual(local_target["partition_target_iou"], 1.0)
        self.assertFalse(local_target["detected"])
        self.assertEqual(local_target["detected_target_recall"], 0.0)
        self.assertEqual(local_target["detected_target_iou"], 0.0)
        self.assertEqual(
            local_target["off_target_detections"],
            local_target["n_detected_map"],
        )

    def test_pure_cases_separate_the_two_reference_frames(self):
        local_z_on_local = self._row("local", "local", "local_z")
        sul_on_local = self._row("local", "local", "sul")
        local_z_on_global = self._row("global", "global", "local_z")
        sul_on_global = self._row("global", "global", "sul")

        self.assertLess(local_z_on_local["score"], -5.0)
        self.assertLess(sul_on_local["score"], 5.0)
        self.assertLess(abs(local_z_on_global["score"]), 3.0)
        self.assertGreater(sul_on_global["score"], 20.0)
        self.assertTrue(local_z_on_local["detected"])
        self.assertGreater(local_z_on_local["evidence_ratio"], 1.0)
        self.assertFalse(sul_on_local["detected"])
        self.assertLess(sul_on_local["evidence_ratio"], 1.0)
        self.assertFalse(local_z_on_global["detected"])
        self.assertLess(local_z_on_global["evidence_ratio"], 1.0)
        self.assertTrue(sul_on_global["detected"])
        self.assertGreater(sul_on_global["evidence_ratio"], 1.0)

    def test_fair_case_is_reported_as_a_map_sanity_check(self):
        fair = self.results[self.results["scenario"] == "fair"]
        self.assertEqual(set(fair["target"]), {"mapa"})
        self.assertEqual(set(fair["metric"]), {"local_z", "sul"})
        self.assertTrue((fair["effect_pp"] == 0).all())

    def test_invalid_significance_configuration_is_rejected(self):
        for invalid in (0.0, 1.0):
            with self.subTest(signif_level=invalid):
                with self.assertRaisesRegex(ValueError, "signif_level"):
                    run_effect_sweep(
                        effects_pp=(20,),
                        n_alt_worlds=20,
                        signif_level=invalid,
                    )
        with self.assertRaisesRegex(ValueError, "at least"):
            run_effect_sweep(
                effects_pp=(20,),
                n_alt_worlds=20,
                signif_level=0.005,
            )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            run_effect_sweep(
                effects_pp=(20, 20),
                n_alt_worlds=20,
                signif_level=0.05,
            )

    def test_markdown_summary_names_the_sanity_check_and_all_effects(self):
        markdown = build_markdown_summary(self.results, self.metadata)
        self.assertIn("Mapa justo", markdown)
        self.assertIn("varredura determinística", markdown)
        self.assertIn("20 p.p.", markdown)
        self.assertIn("local-z", markdown)
        self.assertIn("SUL", markdown)
        self.assertIn("fora do alvo", markdown)

    def test_writer_publishes_csv_markdown_json_and_slide_figure(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            written = write_benchmark_artifacts(self.results, self.metadata, out)
            expected = {
                "benchmark_results.csv",
                "benchmark_summary.md",
                "benchmark_metadata.json",
                "benchmark_effect_sweep.png",
                "benchmark_effect_sweep.pdf",
                "comparison_slide.md",
            }
            self.assertEqual({path.name for path in written}, expected)
            for path in written:
                self.assertTrue(path.exists(), path)
                self.assertGreater(path.stat().st_size, 0)

    def test_comparison_slide_can_include_current_lar_and_authors_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lar = root / "lar"
            comparison = lar / "exploration/comparisons/sul_vs_local_z"
            comparison.mkdir(parents=True)
            pd.DataFrame(
                {
                    "set": ["ambas", "somente_sul", "somente_local_z", "nenhuma"],
                    "n_clusters": [12, 11, 5, 13],
                }
            ).to_csv(comparison / "detection_sets.csv", index=False)

            authors = root / "authors"
            authors.mkdir()
            pd.DataFrame(
                [{"n_regions": 41, "significant_regions": 23}]
            ).to_csv(authors / "hdbscan_lar_comparison.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "method": "kmeans_scan",
                        "n_regions": 2000,
                        "significant_regions": 634,
                        "non_overlapping_regions": 42,
                    }
                ]
            ).to_csv(authors / "unrestricted_lar_regions.csv", index=False)

            written = write_benchmark_artifacts(
                self.results,
                self.metadata,
                root / "out",
                lar_run_dir=lar,
                authors_run_dir=authors,
            )
            slide = next(path for path in written if path.name == "comparison_slide.md")
            text = slide.read_text(encoding="utf-8")
            self.assertIn("17", text)
            self.assertIn("23", text)
            self.assertIn("2.000", text)
            self.assertIn("42", text)
            self.assertIn("| Leitura A | Leitura B |", text)
            self.assertIn("HDBSCAN + SUL", text)


if __name__ == "__main__":
    unittest.main()
