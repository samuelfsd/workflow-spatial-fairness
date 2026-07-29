"""Generic Monte Carlo significance for any metric (ADR-0003).

Preserves the authors' procedure — fixed locations, labels re-drawn as
Bernoulli(rho global), the extreme statistic per world, rank-based p-value —
but recomputes *whichever* metric is passed in, taking max|.| per world so a
signed statistic (local z) behaves like the authors' non-negative tau.
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np

from clustering.base import Partition
from metrics.base import MetricContext, MetricFn
from metrics.group_fairness import get_random_types


def simulate_null_metric(
    metric_fn: MetricFn,
    partition: Partition,
    ctx: MetricContext,
    n_alt_worlds: int,
    n_total: int,
    p_total: int,
    seed: int | None = None,
) -> np.ndarray:
    """Return the max|per-cluster| of `metric_fn` per alternate (fair) world."""
    if n_alt_worlds <= 0 or not partition.regions:
        return np.asarray([], dtype=float)

    rng = np.random.default_rng(seed)
    maxima = []
    for _ in range(n_alt_worlds):
        alt_types = get_random_types(n_total, p_total, rng=rng)
        per_cluster = np.abs(np.asarray(metric_fn(partition, alt_types, ctx).per_cluster, dtype=float))
        per_cluster = per_cluster[~np.isnan(per_cluster)]
        maxima.append(float(per_cluster.max()) if len(per_cluster) else 0.0)

    return np.asarray(maxima, dtype=float)


def significance_threshold(signif_level: float, null_scores: np.ndarray) -> float:
    """Upper `signif_level` quantile of the null max distribution (authors' rule)."""
    if len(null_scores) == 0:
        return 0.0
    ordered = np.sort(null_scores)[::-1]
    idx = min(int(signif_level * len(null_scores)), len(ordered) - 1)
    return float(ordered[idx])


def analytic_threshold(signif_level: float, n_clusters: int) -> float:
    """Two-sided Sidak-corrected sigma band for a **standardized** metric.

    Cross-check for `significance_threshold`, not a replacement: it answers the
    same question by formula instead of by simulation, and needs no worlds. Only
    valid where the per-cluster value is already in standard-error units (under
    H0, local z ~ N(0,1)); for the SUL the analytic ruler would be the
    log-likelihood/chi-square scale, not a sigma band, so callers must gate on
    `MetricResult.standardized`.

    Being an independence-assuming correction, it lands slightly *above* the
    Monte Carlo threshold, which also captures the correlation between clusters.
    """
    if n_clusters <= 0 or not 0.0 < signif_level < 1.0:
        return float("nan")

    per_test = 1.0 - (1.0 - signif_level) ** (1.0 / n_clusters)
    return float(NormalDist().inv_cdf(1.0 - per_test / 2.0))
