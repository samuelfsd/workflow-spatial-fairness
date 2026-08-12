import sys
import unittest
from pathlib import Path

import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from clustering.hdbscan import fit_hdbscan_partition
from clustering.internal import InternalSubdivision
from metrics.base import MetricContext
from metrics.neighbors import build_delaunay_adjacency
from metrics.registry import get_metric
from metrics.significance import analytic_threshold
from synth_data import LOCAL_CONTROL, generate_local_control


class GeneratorTests(unittest.TestCase):
    def test_columns_and_size_match_the_declared_ground_truth(self):
        df = generate_local_control()
        self.assertEqual(list(df.columns), ["lat", "lon", "label"])
        self.assertEqual(len(df), LOCAL_CONTROL.n_total)
        self.assertEqual(set(df["label"].unique()), {0, 1})

    def test_global_rate_is_exactly_the_declared_one(self):
        # The blob rates are calibrated so the map-wide rate is exact, which is
        # what makes "the pocket sits at the global rate" a real statement.
        df = generate_local_control()
        self.assertAlmostEqual(df["label"].mean(), LOCAL_CONTROL.global_rate, places=6)

    def test_generation_is_deterministic_for_a_seed(self):
        first = generate_local_control(seed=7)
        second = generate_local_control(seed=7)
        self.assertTrue(first.equals(second))

    def test_a_different_seed_moves_the_points_but_not_the_rates(self):
        first = generate_local_control(seed=1)
        second = generate_local_control(seed=2)
        self.assertFalse(first["lat"].equals(second["lat"]))
        self.assertAlmostEqual(first["label"].mean(), second["label"].mean())

    def test_the_local_pocket_sits_at_the_global_rate(self):
        # This is the whole point of the control: a pocket the global baseline
        # cannot see, because its rate *is* the global rate, while its
        # neighbourhood sits well above it.
        self.assertAlmostEqual(LOCAL_CONTROL.local_pocket_rate, LOCAL_CONTROL.global_rate)
        self.assertGreater(LOCAL_CONTROL.local_peer_rate, LOCAL_CONTROL.global_rate)

    def test_the_global_pocket_deviates_while_its_peers_follow_it(self):
        # The mirror case: the SUL must fire here and the local z must not.
        self.assertLess(LOCAL_CONTROL.global_pocket_rate, LOCAL_CONTROL.global_rate)
        self.assertAlmostEqual(LOCAL_CONTROL.global_pocket_rate, LOCAL_CONTROL.global_peer_rate)


class GroundTruthTests(unittest.TestCase):
    """The validation the whole spec exists for, run through the normal seams."""

    @classmethod
    def setUpClass(cls) -> None:
        df = generate_local_control()
        cls.df = df
        cls.types = df["label"].to_numpy(dtype=int)
        cls.partition = fit_hdbscan_partition(df, min_cluster_frac=0.02, min_samples=25)
        cls.ctx = MetricContext(
            n_total=len(df),
            p_total=int(cls.types.sum()),
            adjacency=build_delaunay_adjacency(cls.partition, df),
            rng=np.random.default_rng(0),
            internal_subdivider=lambda points: InternalSubdivision(
                [list(points)], [], len(points), len(points)
            ),
        )
        cls.local_z = get_metric("local_z")(cls.partition, cls.types, cls.ctx).per_cluster
        cls.sul = get_metric("sul")(cls.partition, cls.types, cls.ctx).per_cluster
        cls.rates = np.array(
            [cls.types[region["points"]].mean() for region in cls.partition.regions]
        )
        cls.sizes = np.array([len(region["points"]) for region in cls.partition.regions])

    def _pocket_cluster(self, role: str) -> int:
        """Index of the cluster that recovered a planted blob, found by geography.

        Identifying by rate would be ambiguous — the global pocket and its peers
        share a rate by design — so the pocket is located by its planted centre.
        """
        blob = next(item for item in LOCAL_CONTROL.blobs if item.role == role)
        centroids = [
            (
                self.df["lat"].to_numpy()[region["points"]].mean(),
                self.df["lon"].to_numpy()[region["points"]].mean(),
            )
            for region in self.partition.regions
        ]
        return min(
            range(len(centroids)),
            key=lambda idx: (centroids[idx][0] - blob.lat) ** 2 + (centroids[idx][1] - blob.lon) ** 2,
        )

    def test_clustering_recovers_the_planted_blobs(self):
        # Enough separated clusters for both pockets to exist as regions.
        self.assertGreaterEqual(len(self.partition.regions), 8)

    def test_each_pocket_cluster_carries_its_planted_rate(self):
        local = self._pocket_cluster("local_pocket")
        glob = self._pocket_cluster("global_pocket")
        self.assertNotEqual(local, glob)
        self.assertAlmostEqual(self.rates[local], LOCAL_CONTROL.local_pocket_rate, places=2)
        self.assertAlmostEqual(self.rates[glob], LOCAL_CONTROL.global_pocket_rate, places=2)

    def test_the_local_pocket_is_invisible_to_the_global_baseline(self):
        idx = self._pocket_cluster("local_pocket")
        # Its rate equals the global rate, so the log-likelihood ratio collapses.
        self.assertLess(self.sul[idx], 5.0)

    def test_the_local_pocket_is_caught_by_the_peer_baseline(self):
        idx = self._pocket_cluster("local_pocket")
        threshold = analytic_threshold(0.005, len(self.partition.regions))
        self.assertLess(self.local_z[idx], -threshold)

    def test_the_global_pocket_is_caught_by_the_global_baseline(self):
        idx = self._pocket_cluster("global_pocket")
        self.assertGreater(self.sul[idx], 20.0)

    def test_the_global_pocket_is_quiet_for_the_peer_baseline(self):
        # Its neighbours share its rate, so there is nothing local to detect:
        # the local z is not a more sensitive SUL, it answers another question.
        idx = self._pocket_cluster("global_pocket")
        threshold = analytic_threshold(0.005, len(self.partition.regions))
        self.assertLess(abs(self.local_z[idx]), threshold)


if __name__ == "__main__":
    unittest.main()
