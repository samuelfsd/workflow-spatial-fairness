"""Built-in fairness metrics wrapped in the unified metric protocol (ADR-0002).

Each function delegates to the preserved numeric core in `group_fairness.py`;
this module only adapts those results into `MetricResult` objects so they flow
through the registry, tables, maps, and the generic Monte Carlo engine.
"""

from __future__ import annotations

import math

import numpy as np

from clustering.base import Partition
from clustering.internal import InternalSubdivision
from metrics.base import MetricContext, MetricResult
from metrics.group_fairness import (
    calculate_gini,
    calculate_gini_contributions,
    calculate_meanvar,
    calculate_sul,
    get_simple_stats,
)


def _region_rates(partition: Partition, types: np.ndarray) -> np.ndarray:
    return np.array(
        [get_simple_stats(region["points"], types)[2] for region in partition.regions],
        dtype=float,
    )


def sul_metric(partition: Partition, types: np.ndarray, ctx: MetricContext) -> MetricResult:
    """Spatial Unfairness Likelihood per cluster (global baseline).

    Non-negative log-likelihood-ratio scan statistic; direction comes separately
    from `classify_direction`, so the metric itself is unsigned.
    """
    per_cluster = np.array(
        [
            calculate_sul(*get_simple_stats(region["points"], types)[:2], ctx.n_total, ctx.p_total)
            for region in partition.regions
        ],
        dtype=float,
    )
    return MetricResult(
        per_cluster=per_cluster,
        partition_scalar=None,
        signed=False,
        supports_mc=True,
        needs=frozenset(),
    )


def local_z_metric(partition: Partition, types: np.ndarray, ctx: MetricContext) -> MetricResult:
    """Local z-score: two-proportion contrast of a cluster vs its pooled peers.

    Peers are the cluster's Delaunay neighbours (`ctx.adjacency`), pooled by size
    and excluding the cluster itself. Clusters with fewer than 2 peers get NaN
    (unreliable local baseline; reported as "not evaluated by neighbourhood").
    """
    stats: dict[int, tuple[int, int]] = {}
    for region in partition.regions:
        n, p, _ = get_simple_stats(region["points"], types)
        stats[int(region["cluster_label"])] = (n, p)

    scores = np.full(len(partition.regions), np.nan)
    for idx, region in enumerate(partition.regions):
        label = int(region["cluster_label"])
        peers = ctx.adjacency.get(label, [])
        if len(peers) < 2:
            continue

        n1, p1 = stats[label]
        n2 = sum(stats[peer][0] for peer in peers)
        p2 = sum(stats[peer][1] for peer in peers)
        if n1 == 0 or n2 == 0:
            continue

        rho_in = p1 / n1
        rho_peer = p2 / n2
        p_pool = (p1 + p2) / (n1 + n2)
        se = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2))
        if se == 0.0:
            continue
        scores[idx] = (rho_in - rho_peer) / se

    return MetricResult(
        per_cluster=scores,
        partition_scalar=None,
        signed=True,
        supports_mc=True,
        standardized=True,
        needs=frozenset({"neighbors"}),
    )


def gini_subcluster_metric(
    partition: Partition, types: np.ndarray, ctx: MetricContext
) -> MetricResult:
    """Gini with an *internal* reference: across a cluster's subcluster rates.

    The statistic is the same Gini as always; only the reference changes (the
    cluster's own density subclusters instead of the map's clusters). Quantifies
    internal heterogeneity ("a big cluster hides a bad pocket"). A cluster that
    does not subdivide has a single rate => Gini 0 (homogeneous inside). NOT
    Gini over raw binary outcomes (that is degenerate).
    """
    subdivider = ctx.internal_subdivider or (
        lambda points: InternalSubdivision(
            subclusters=[list(points)],
            residue=[],
            min_cluster_size=max(2, len(points)),
            parent_n=len(points),
        )
    )
    per_cluster = []
    subdivisions = []
    for region in partition.regions:
        subdivision = subdivider(list(region["points"]))
        subdivisions.append(subdivision)
        rates = [get_simple_stats(sub, types)[2] for sub in subdivision.subclusters]
        per_cluster.append(calculate_gini(rates) if rates else float("nan"))
    return MetricResult(
        per_cluster=np.array(per_cluster, dtype=float),
        partition_scalar=None,
        signed=False,
        supports_mc=False,
        needs=frozenset({"subclusters"}),
        per_cluster_metadata={
            "internal_subdivision_status": np.array(
                [item.status for item in subdivisions], dtype=object
            ),
            "internal_coverage_rate": np.array(
                [item.coverage_rate for item in subdivisions], dtype=float
            ),
            "internal_residue_n": np.array(
                [item.residue_n for item in subdivisions], dtype=int
            ),
            "internal_n_subclusters": np.array(
                [len(item.subclusters) for item in subdivisions], dtype=int
            ),
            "internal_min_cluster_size": np.array(
                [item.min_cluster_size for item in subdivisions], dtype=int
            ),
        },
    )


def _parity_frame(partition: Partition, types: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten a partition into (outcomes, cluster label) arrays for fairlearn.

    Iterating over `partition.regions` is what keeps the **unassigned points
    (label -1) out of `sensitive_features`** (ADR-0004): they are not a region,
    so treating them as a group would measure parity against nowhere.
    """
    outcomes: list[int] = []
    groups: list[int] = []
    for region in partition.regions:
        points = list(region["points"])
        label = int(region["cluster_label"])
        outcomes.extend(int(value) for value in types[points])
        groups.extend([label] * len(points))
    return np.asarray(outcomes, dtype=int), np.asarray(groups, dtype=int)


def _selection_rates(partition: Partition, types: np.ndarray) -> np.ndarray:
    """Per-cluster selection rate from fairlearn's `MetricFrame`, region-aligned."""
    from fairlearn.metrics import MetricFrame, selection_rate

    outcomes, groups = _parity_frame(partition, types)
    if not len(outcomes):
        return np.full(len(partition.regions), np.nan)

    by_group = MetricFrame(
        metrics=selection_rate,
        y_true=outcomes,
        y_pred=outcomes,
        sensitive_features=groups,
    ).by_group
    # Reindex by region order: MetricFrame sorts its groups, regions may not be.
    return np.array(
        [float(by_group.get(int(region["cluster_label"]), np.nan)) for region in partition.regions],
        dtype=float,
    )


def _parity_result(partition: Partition, types: np.ndarray, statistic) -> MetricResult:
    """Shared shape for the parity metrics: rates per cluster + one scalar."""
    per_cluster = _selection_rates(partition, types)
    if len(partition.regions) < 2:
        # Parity between groups is undefined with a single group; fairlearn would
        # return 0 ("perfectly fair"), which reads as a verdict we cannot make.
        scalar = float("nan")
    else:
        outcomes, groups = _parity_frame(partition, types)
        scalar = float(statistic(outcomes, outcomes, sensitive_features=groups))

    return MetricResult(
        per_cluster=per_cluster,
        partition_scalar=scalar,
        signed=False,
        supports_mc=False,
        needs=frozenset(),
    )


def dp_difference_metric(
    partition: Partition, types: np.ndarray, ctx: MetricContext
) -> MetricResult:
    """Statistical parity difference across clusters (fairlearn), i.e. max - min rate.

    Names, in the language of the fairness literature, the spread this pipeline
    already reported in prose. Same input as the partition Gini (the vector of
    cluster rates), different summary: Gini is the normalized mean pairwise gap,
    this is the extreme gap — so it is fragile to small groups and must be read
    next to the smallest/largest cluster size (ADR-0004).
    """
    from fairlearn.metrics import demographic_parity_difference

    return _parity_result(partition, types, demographic_parity_difference)


def dp_ratio_metric(partition: Partition, types: np.ndarray, ctx: MetricContext) -> MetricResult:
    """Statistical parity ratio across clusters (fairlearn), i.e. min / max rate."""
    from fairlearn.metrics import demographic_parity_ratio

    return _parity_result(partition, types, demographic_parity_ratio)


def gini_metric(partition: Partition, types: np.ndarray, ctx: MetricContext) -> MetricResult:
    """Partition Gini of per-cluster rates + per-cluster leave-one-out contribution.

    The scalar is the map-wide inequality; the per-cluster array is each cluster's
    signed contribution to it (positive = pushes inequality up). Not an MC metric.
    """
    rhos = _region_rates(partition, types)
    return MetricResult(
        per_cluster=calculate_gini_contributions(rhos),
        partition_scalar=calculate_gini(rhos),
        signed=True,
        supports_mc=False,
        needs=frozenset(),
    )


def meanvar_metric(partition: Partition, types: np.ndarray, ctx: MetricContext) -> MetricResult:
    """MeanVar partition baseline: variance of per-cluster rates.

    Per-cluster array is each cluster's squared deviation from the mean rate
    (the term it contributes to MeanVar). Only meaningful on disjoint partitions.
    """
    rhos = _region_rates(partition, types)
    valid = rhos[~np.isnan(rhos)]
    mean_rho = float(np.mean(valid)) if len(valid) else 0.0
    return MetricResult(
        per_cluster=(rhos - mean_rho) ** 2,
        partition_scalar=calculate_meanvar(rhos),
        signed=False,
        supports_mc=False,
        needs=frozenset(),
    )
