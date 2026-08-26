"""Registry of pluggable fairness metrics (ADR-0002).

A metric is a callable ``(partition, types, ctx) -> MetricResult``. To add one:
implement it returning a `MetricResult` (see `metrics/base.py`), then add one
entry to `METRICS` below. Tables, maps, and Monte Carlo depend only on the
`MetricResult` shape, so no other code needs to change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from metrics.base import MetricFn
from metrics.builtin import (
    dp_difference_metric,
    dp_ratio_metric,
    gini_metric,
    gini_subcluster_metric,
    local_z_metric,
    meanvar_metric,
    peer_gini_gap_metric,
    peer_log_rate_ratio_metric,
    peer_rate_difference_metric,
    sul_metric,
)

RateReference = Literal["peers", "outside"]
DirectionRule = Literal["score_sign", "rate_contrast"]


@dataclass(frozen=True)
class MetricDefinition:
    """Pre-compute capabilities kept on the same one-line registry entry."""

    compute: MetricFn
    needs: frozenset[str] = frozenset()
    benchmark_candidate: bool = False
    outcome_direction: bool | None = None
    confirmatory_candidate: bool = False
    candidate_kind: str | None = None


METRICS: dict[str, MetricDefinition | MetricFn] = {
    "sul": MetricDefinition(sul_metric),
    "local_z": MetricDefinition(
        local_z_metric, needs=frozenset({"neighbors"})
    ),
    "gini": MetricDefinition(gini_metric),
    "meanvar": MetricDefinition(meanvar_metric),
    "gini_subcluster": MetricDefinition(
        gini_subcluster_metric, needs=frozenset({"subclusters"})
    ),
    "dp_difference": MetricDefinition(dp_difference_metric),
    "dp_ratio": MetricDefinition(dp_ratio_metric),
    "peer_rate_difference": MetricDefinition(
        peer_rate_difference_metric,
        needs=frozenset({"neighbors"}),
        benchmark_candidate=True,
        outcome_direction=True,
        confirmatory_candidate=True,
        candidate_kind="rate_difference",
    ),
    "peer_log_rate_ratio": MetricDefinition(
        peer_log_rate_ratio_metric,
        needs=frozenset({"neighbors"}),
        benchmark_candidate=True,
        outcome_direction=True,
        confirmatory_candidate=True,
        candidate_kind="log_rate_ratio",
    ),
    "peer_gini_gap": MetricDefinition(
        peer_gini_gap_metric,
        needs=frozenset({"neighbors", "subclusters"}),
        benchmark_candidate=True,
        outcome_direction=False,
        confirmatory_candidate=False,
        candidate_kind="gini_gap",
    ),
}


@dataclass(frozen=True)
class PrimaryCapabilities:
    """Contract required for a metric to govern detection-class reporting."""

    rate_reference: RateReference
    direction_rule: DirectionRule
    supports_mc: bool = True


@dataclass(frozen=True)
class DetectionDecision:
    """Evaluation status precedes the three-way detection class."""

    evaluation_status: str
    evaluation_reason: str | None
    direction: str | None
    significant: bool | None
    detection_class: str | None
    evidence_ratio: float | None


PRIMARY_CAPABILITIES: dict[str, PrimaryCapabilities] = {
    "local_z": PrimaryCapabilities("peers", "score_sign"),
    "sul": PrimaryCapabilities("outside", "rate_contrast"),
}


def metric_names() -> list[str]:
    return list(METRICS)


def get_metric_definition(name: str) -> MetricDefinition:
    if name not in METRICS:
        raise ValueError(f"Unknown metric: {name}. Available: {metric_names()}")
    entry = METRICS[name]
    return entry if isinstance(entry, MetricDefinition) else MetricDefinition(entry)


def candidate_metric_names() -> list[str]:
    return [
        name for name in METRICS
        if get_metric_definition(name).benchmark_candidate
    ]


def primary_metric_names() -> list[str]:
    """Metrics that satisfy the complete primary/detection contract."""
    return [name for name in METRICS if name in PRIMARY_CAPABILITIES]


def get_primary_capabilities(name: str) -> PrimaryCapabilities:
    if name not in PRIMARY_CAPABILITIES:
        raise ValueError(
            f"Metric {name!r} is not eligible as primary. "
            f"Available primary metrics: {primary_metric_names()}"
        )
    return PRIMARY_CAPABILITIES[name]


def evaluate_primary(
    name: str,
    *,
    score: float,
    threshold: float | None,
    rho_in: float,
    rho_reference: float,
    precondition_reason: str | None = None,
) -> DetectionDecision:
    """Apply a primary metric's declared direction and evaluation rules."""
    capabilities = get_primary_capabilities(name)
    if precondition_reason:
        return DetectionDecision(
            "não avaliado", precondition_reason, None, None, None, None
        )
    if not math.isfinite(float(score)):
        return DetectionDecision(
            "não avaliado", "score_nao_finito", None, None, None, None
        )
    if threshold is None or not math.isfinite(float(threshold)) or threshold <= 0:
        return DetectionDecision(
            "não avaliado", "limiar_ausente_ou_invalido", None, None, None, None
        )

    if capabilities.direction_rule == "score_sign":
        contrast = float(score)
    else:
        if not math.isfinite(float(rho_in)) or not math.isfinite(float(rho_reference)):
            return DetectionDecision(
                "não avaliado", "referencia_de_taxa_nao_finita", None, None, None, None
            )
        contrast = float(rho_in) - float(rho_reference)

    direction = "negative" if contrast < 0 else "positive" if contrast > 0 else "neutral"
    significant = abs(float(score)) >= float(threshold)
    detection_class = direction if significant and direction != "neutral" else "neutral"
    return DetectionDecision(
        evaluation_status="avaliado",
        evaluation_reason=None,
        direction=direction,
        significant=significant,
        detection_class=detection_class,
        evidence_ratio=abs(float(score)) / float(threshold),
    )


def get_metric(name: str) -> MetricFn:
    return get_metric_definition(name).compute
