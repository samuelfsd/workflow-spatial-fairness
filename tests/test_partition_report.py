import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from clustering.base import Partition
from clustering.hdbscan import effective_min_cluster_size
from partition_report import build_configs, config_label, config_slug, markdown_table, profile_table


def _two_blob_df(n_per_blob: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for lat, lon in ((0.0, 0.0), (5.0, 5.0)):
        frames.append(
            pd.DataFrame(
                {
                    "lat": rng.normal(lat, 0.01, n_per_blob),
                    "lon": rng.normal(lon, 0.01, n_per_blob),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


class MinClusterSizeTests(unittest.TestCase):
    def test_floor_is_the_fraction_of_the_dataset(self):
        # LAR at frac 0.005: round(0.005 * 206418) = 1032 points.
        self.assertEqual(effective_min_cluster_size(206418, 0.005), 1032)

    def test_floor_never_drops_below_the_absolute_minimum(self):
        self.assertEqual(effective_min_cluster_size(100, 0.005), 25)


class ConfigLabelTests(unittest.TestCase):
    def test_labels_name_the_mechanism_and_call_the_cap_a_ceiling(self):
        self.assertEqual(config_label("hdbscan", None), "sem cap (orgânico)")
        self.assertEqual(config_label("hdbscan", 1000), "cap nativo (teto 1000)")
        self.assertEqual(config_label("capped_hdbscan", 1000), "redivisão (teto 1000)")

    def test_slugs_stay_ascii_for_file_names(self):
        self.assertEqual(config_slug("capped_hdbscan", 2000), "capped_hdbscan_cap2000")
        self.assertEqual(config_slug("hdbscan", None), "hdbscan")


class BuildConfigsTests(unittest.TestCase):
    def test_an_impossible_ceiling_is_skipped_with_a_reason(self):
        # 120 points at frac 0.5 => floor 60; a ceiling of 60 or below asks for
        # clusters both smaller and larger than 60, which is an empty set.
        df = _two_blob_df()
        configs, skipped = build_configs(
            df,
            methods=("hdbscan",),
            min_cluster_frac=0.5,
            max_cluster_sizes=(60,),
            min_samples=5,
        )
        self.assertEqual(len(skipped), 1)
        self.assertIn("impossível", skipped[0])
        self.assertIn("60", skipped[0])
        # The uncapped configuration still ran: one bad ceiling does not abort.
        self.assertIn("sem cap (orgânico)", configs)

    def test_a_viable_ceiling_is_kept_and_carries_its_value(self):
        df = _two_blob_df()
        configs, skipped = build_configs(
            df,
            methods=("hdbscan",),
            min_cluster_frac=0.1,
            max_cluster_sizes=(100,),
            min_samples=5,
        )
        self.assertEqual(skipped, [])
        _, _, cap = configs["cap nativo (teto 100)"]
        self.assertEqual(cap, 100)

    def test_the_recursive_split_is_never_built_without_a_ceiling(self):
        df = _two_blob_df()
        configs, _ = build_configs(
            df,
            methods=("capped_hdbscan",),
            min_cluster_frac=0.1,
            max_cluster_sizes=(),
            min_samples=5,
        )
        self.assertEqual(configs, {})

    def test_rescue_configurations_are_named_by_the_density_axis(self):
        df = _two_blob_df()
        configs, skipped = build_configs(
            df,
            methods=("hdbscan_rescue",),
            min_cluster_frac=0.1,
            max_cluster_sizes=(),
            min_samples=5,
            rescue_min_samples=(5,),
            stat_cap=False,
        )

        self.assertEqual(skipped, [])
        self.assertIn("resgate min_samples=5", configs)
        self.assertEqual(configs["resgate min_samples=5"][0].method, "hdbscan_rescue")


class ProfileTableTests(unittest.TestCase):
    def _configs_and_frames(self, cap):
        partition = Partition(
            method="capped_hdbscan",
            params={},
            labels=np.array([0, 0, 0, 1, 1]),
            regions=[
                {"points": [0, 1, 2], "cluster_label": 0},
                {"points": [3, 4], "cluster_label": 1},
            ],
        )
        frame = pd.DataFrame(
            {
                "cluster_label": [0, 1],
                "n": [3, 2],
                "p": [2, 1],
                "n_neg": [1, 1],
                "rho": [2 / 3, 0.5],
                "raio_medio_km": [1.0, 2.0],
                "raio_p95_km": [1.0, 2.0],
            }
        )
        label = "redivisão (teto 2)"
        return {label: (partition, "slug", cap)}, {label: frame}, label

    def test_compliance_is_the_share_of_clusters_within_the_ceiling(self):
        configs, frames, label = self._configs_and_frames(cap=2)
        profile = profile_table(configs, frames, n_total=5, global_rate=0.6)
        # Sizes are 3 and 2 against a ceiling of 2 => one of two complies.
        self.assertAlmostEqual(profile.loc[0, "cap_compliance"], 0.5)
        self.assertEqual(profile.loc[0, "cap"], 2)

    def test_compliance_is_undefined_without_a_ceiling(self):
        configs, frames, _ = self._configs_and_frames(cap=None)
        profile = profile_table(configs, frames, n_total=5, global_rate=0.6)
        self.assertTrue(np.isnan(profile.loc[0, "cap_compliance"]))

    def test_markdown_renders_the_undefined_compliance_as_a_dash(self):
        configs, frames, _ = self._configs_and_frames(cap=None)
        profile = profile_table(configs, frames, n_total=5, global_rate=0.6)
        table = markdown_table(profile)
        self.assertIn("Teto cumprido", table)
        self.assertIn("—", table)

    def test_profile_reports_coverage_and_compactness_by_cluster_origin(self):
        configs, frames, label = self._configs_and_frames(cap=None)
        partition = configs[label][0]
        partition.regions[0]["origin"] = "organic"
        partition.regions[1]["origin"] = "rescue"
        frames[label]["origin"] = ["organic", "rescue"]

        profile = profile_table(configs, frames, n_total=5, global_rate=0.6)

        self.assertAlmostEqual(profile.loc[0, "organic_rate"], 3 / 5)
        self.assertAlmostEqual(profile.loc[0, "rescue_rate"], 2 / 5)
        self.assertAlmostEqual(profile.loc[0, "organic_raio_medio_km_mean"], 1.0)
        self.assertAlmostEqual(profile.loc[0, "rescue_raio_p95_km_mean"], 2.0)


if __name__ == "__main__":
    unittest.main()
