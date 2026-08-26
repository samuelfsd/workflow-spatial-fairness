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
    scores = np.full(len(partition.regions), np.nan)
    for idx, counts in enumerate(_peer_rate_counts(partition, types, ctx)):
        if counts is None:
            continue
        n1, p1, n2, p2 = counts
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


def _peer_rate_counts(
    partition: Partition, types: np.ndarray, ctx: MetricContext
) -> list[tuple[int, int, int, int] | None]:
    """Return cluster/pooled-peer counts aligned with the partition."""
    stats = {
        int(region["cluster_label"]): get_simple_stats(region["points"], types)[:2]
        for region in partition.regions
    }
    counts: list[tuple[int, int, int, int] | None] = []
    for region in partition.regions:
        label = int(region["cluster_label"])
        peers = [peer for peer in ctx.adjacency.get(label, []) if peer in stats]
        if len(peers) < 2:
            counts.append(None)
            continue
        n1, p1 = stats[label]
        n2 = sum(stats[peer][0] for peer in peers)
        p2 = sum(stats[peer][1] for peer in peers)
        counts.append((n1, p1, n2, p2) if n1 and n2 else None)
    return counts


def peer_rate_difference_metric(
    partition: Partition, types: np.ndarray, ctx: MetricContext
) -> MetricResult:
    """Signed positive-rate gap between a cluster and its pooled Delaunay peers.

    This is the native effect size standardized by ``local_z_metric``. It is a
    benchmark candidate, not an alias for demographic-parity difference across
    the whole partition.
    """
    scores = np.full(len(partition.regions), np.nan)
    for idx, counts in enumerate(_peer_rate_counts(partition, types, ctx)):
        if counts is None:
            continue
        n1, p1, n2, p2 = counts
        scores[idx] = p1 / n1 - p2 / n2
    return MetricResult(
        per_cluster=scores,
        partition_scalar=None,
        signed=True,
        supports_mc=True,
        standardized=False,
        needs=frozenset({"neighbors"}),
    )


def peer_log_rate_ratio_metric(
    partition: Partition, types: np.ndarray, ctx: MetricContext
) -> MetricResult:
    """Continuity-corrected log rate ratio of a cluster versus pooled peers.

    The log scale is centred at zero and preserves direction, unlike a raw
    ``min/max`` partition ratio. A Haldane-Anscombe half-count avoids undefined
    values when one side contains no positive observations.
    """
    scores = np.full(len(partition.regions), np.nan)
    for idx, counts in enumerate(_peer_rate_counts(partition, types, ctx)):
        if counts is None:
            continue
        n1, p1, n2, p2 = counts
        rho_in = (p1 + 0.5) / (n1 + 1.0)
        rho_peer = (p2 + 0.5) / (n2 + 1.0)
        scores[idx] = math.log(rho_in / rho_peer)
    return MetricResult(
        per_cluster=scores,
        partition_scalar=None,
        signed=True,
        supports_mc=True,
        standardized=False,
        needs=frozenset({"neighbors"}),
    )


def peer_gini_gap_metric(
    partition: Partition, types: np.ndarray, ctx: MetricContext
) -> MetricResult:
    """Internal-Gini gap of a cluster versus its size-weighted Delaunay peers.

    This exploratory candidate detects relative internal heterogeneity. Its sign
    means more/less heterogeneous than peers; it does not mean favored/harmed by
    the positive outcome and therefore is not eligible as the official primary.
    """
    subdivider = ctx.internal_subdivider or (
        lambda points: InternalSubdivision(
            subclusters=[list(points)],
            residue=[],
            min_cluster_size=max(2, len(points)),
            parent_n=len(points),
        )
    )
    labels = [int(region["cluster_label"]) for region in partition.regions]
    sizes = {
        int(region["cluster_label"]): len(region["points"])
        for region in partition.regions
    }
    internal: dict[int, float] = {}
    for region in partition.regions:
        label = int(region["cluster_label"])
        subdivision = subdivider(list(region["points"]))
        rates = [get_simple_stats(group, types)[2] for group in subdivision.subclusters]
        internal[label] = calculate_gini(rates) if rates else float("nan")

    scores = np.full(len(partition.regions), np.nan)
    peer_ginis = np.full(len(partition.regions), np.nan)
    for idx, label in enumerate(labels):
        peers = [
            peer for peer in ctx.adjacency.get(label, [])
            if peer in internal and math.isfinite(internal[peer])
        ]
        if len(peers) < 2 or not math.isfinite(internal[label]):
            continue
        denominator = sum(sizes[peer] for peer in peers)
        if not denominator:
            continue
        reference = sum(sizes[peer] * internal[peer] for peer in peers) / denominator
        peer_ginis[idx] = reference
        scores[idx] = internal[label] - reference

    return MetricResult(
        per_cluster=scores,
        partition_scalar=None,
        signed=True,
        supports_mc=True,
        standardized=False,
        needs=frozenset({"neighbors", "subclusters"}),
        per_cluster_metadata={
            "internal_gini": np.array([internal[label] for label in labels], dtype=float),
            "peer_gini": peer_ginis,
        },
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
