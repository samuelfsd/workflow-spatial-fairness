"""Registry of pluggable fairness metrics (ADR-0002).

A metric is a callable ``(partition, types, ctx) -> MetricResult``. To add one:
implement it returning a `MetricResult` (see `metrics/base.py`), then add one
entry to `METRICS` below. Tables, maps, and Monte Carlo depend only on the
`MetricResult` shape, so no other code needs to change.
"""

from __future__ import annotations

from metrics.base import MetricFn
from metrics.builtin import (
    dp_difference_metric,
    dp_ratio_metric,
    gini_metric,
    gini_subcluster_metric,
    local_z_metric,
    meanvar_metric,
    sul_metric,
)

METRICS: dict[str, MetricFn] = {
    "sul": sul_metric,
    "local_z": local_z_metric,
    "gini": gini_metric,
    "meanvar": meanvar_metric,
    "gini_subcluster": gini_subcluster_metric,
    "dp_difference": dp_difference_metric,
    "dp_ratio": dp_ratio_metric,
}


def metric_names() -> list[str]:
    return list(METRICS)


def get_metric(name: str) -> MetricFn:
    if name not in METRICS:
        raise ValueError(f"Unknown metric: {name}. Available: {metric_names()}")
    return METRICS[name]
