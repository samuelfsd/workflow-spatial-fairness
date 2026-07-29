"""Shared metric protocol types for the unified metric registry (ADR-0002).

A metric is a callable ``(partition, types, ctx) -> MetricResult``. Every metric
returns a per-cluster array (aligned to ``partition.regions``) plus an optional
partition-level scalar and capability flags, so the same result flows through
tables, maps, and the generic Monte Carlo engine.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from clustering.base import Partition


@dataclass
class MetricResult:
    """Output of one metric over one partition.

    `per_cluster` is always present and index-aligned with `partition.regions`.
    `partition_scalar` is the partition-wide value (e.g. Gini/MeanVar) or None
    for purely per-cluster metrics (SUL, local z). The flags let the pipeline
    treat any metric generically: `signed` (per_cluster carries a direction),
    `supports_mc` (enters the generic Monte Carlo), `standardized` (per_cluster
    is in standard-error units, so an analytic threshold applies), `needs` (what
    it requires from the context, e.g. "neighbors" / "subclusters").
    """

    per_cluster: np.ndarray
    partition_scalar: float | None = None
    signed: bool = False
    supports_mc: bool = False
    standardized: bool = False
    needs: frozenset[str] = field(default_factory=frozenset)


@dataclass
class MetricContext:
    """Shared resources handed to every metric so it never recomputes them.

    `adjacency` maps a cluster label to its Delaunay-neighbor labels (peer
    baseline). `split_subclusters` splits a region's point ids into density
    subclusters (within-cluster inequality, and the size cap).
    """

    n_total: int
    p_total: int
    adjacency: dict[int, list[int]] = field(default_factory=dict)
    rng: np.random.Generator | None = None
    split_subclusters: Callable[[list[int]], list[list[int]]] | None = None


MetricFn = Callable[[Partition, np.ndarray, MetricContext], MetricResult]
