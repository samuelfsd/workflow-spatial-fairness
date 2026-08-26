"""Canonical evaluation records for benchmark partitions and overlapping scans."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

from clustering.base import Partition
from clustering.internal import InternalSubdivision, diagnostic_density_subdivision
from data_loading import LoadedDataset
from metrics.base import MetricContext
from metrics.group_fairness import (
    classify_direction,
    get_simple_stats,
    scan_regions,
    select_significant_regions,
    simulate_null_max_suls,
)
from metrics.neighbors import build_delaunay_adjacency
from metrics.registry import (
    DetectionDecision,
    evaluate_primary,
    get_metric,
    get_metric_definition,
    get_primary_capabilities,
)
from metrics.significance import significance_threshold, simulate_null_metric
from regions import filter_non_overlapping_regions


@dataclass
class EvaluationBundle:
    summary: pd.DataFrame
    regions: pd.DataFrame
    null_distributions: dict[str, np.ndarray] = field(default_factory=dict)


def rate_semantics(dataset: LoadedDataset) -> str:
    if dataset.name == "lar":
        return "taxa de aprovação"
    if dataset.name == "crime":
        return "TPR"
    return "taxa positiva"


def _stats(region: dict[str, Any], types: np.ndarray) -> tuple[int, int, float]:
    return get_simple_stats(region["points"], types)


def _point_ids(region: dict[str, Any]) -> str:
    return json.dumps([int(point) for point in region["points"]])


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    def cross(origin, a, b):
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])
    unique = sorted(set(points))
    if len(unique) <= 2:
        return unique
    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _geometry_fields(region: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    """Serialize region geometry without relying on an in-memory partition."""
    bounds = region.get("bounds")
    geometry = None
    if bounds is not None:
        xmin, ymin, xmax, ymax = (float(value) for value in bounds)
        geometry = [[ymin, xmin], [ymin, xmax], [ymax, xmax], [ymax, xmin]]
    elif region.get("center") is not None and region.get("radius") is not None:
        center = df.iloc[int(region["center"])]
        radius = float(region["radius"])
        geometry = [
            [float(center["lat"]) - radius, float(center["lon"]) - radius],
            [float(center["lat"]) - radius, float(center["lon"]) + radius],
            [float(center["lat"]) + radius, float(center["lon"]) + radius],
            [float(center["lat"]) + radius, float(center["lon"]) - radius],
        ]
    else:
        subset = df.iloc[list(region["points"])]
        hull = _convex_hull(list(zip(subset["lon"].astype(float), subset["lat"].astype(float))))
        geometry = [[lat, lon] for lon, lat in hull]
    return {
        "region_type": region.get("type"),
        "bounds": json.dumps([float(value) for value in bounds]) if bounds is not None else None,
        "center": int(region["center"]) if region.get("center") is not None else None,
        "radius": float(region["radius"]) if region.get("radius") is not None else None,
        "geometry": json.dumps(geometry) if geometry else None,
    }


def _best_index(scores: np.ndarray) -> int | None:
    finite = np.flatnonzero(np.isfinite(scores))
    if not len(finite):
        return None
    return int(finite[np.argmax(np.abs(scores[finite]))])


def _evaluate_candidate(
    score: float,
    threshold: float | None,
    *,
    precondition_reason: str | None = None,
    outcome_direction: bool = True,
) -> DetectionDecision:
    """Evaluate an experimental signed score without promoting it to primary."""
    if precondition_reason:
        return DetectionDecision(
            "não avaliado", precondition_reason, None, None, None, None
        )
    if not np.isfinite(score):
        return DetectionDecision(
            "não avaliado", "score_nao_finito", None, None, None, None
        )
    if threshold is None or not np.isfinite(threshold) or threshold < 0:
        return DetectionDecision(
            "não avaliado", "limiar_ausente_ou_invalido", None, None, None, None
        )
    direction = (
        "negative" if score < 0 else "positive" if score > 0 else "neutral"
    ) if outcome_direction else None
    significant = abs(score) >= threshold if threshold > 0 else abs(score) > 0
    return DetectionDecision(
        evaluation_status="avaliado",
        evaluation_reason=None,
        direction=direction,
        significant=bool(significant),
        # Candidate results get their own panels and never govern the canonical
        # detection-class colors before promotion to an official primary.
        detection_class=None,
        evidence_ratio=abs(score) / threshold if threshold > 0 else None,
    )


def evaluate_partition(
    dataset: LoadedDataset,
    partition: Partition,
    *,
    metrics: Iterable[str],
    protocol: str,
    n_alt_worlds: int,
    signif_level: float,
    seed: int,
) -> EvaluationBundle:
    """Evaluate calibrated primaries and diagnostic metrics on one disjoint partition."""
    metric_names = tuple(metrics)
    required_needs = set().union(*(
        get_metric_definition(name).needs for name in metric_names
    )) if metric_names else set()
    adjacency = (
        build_delaunay_adjacency(partition, dataset.df)
        if "neighbors" in required_needs
        else {}
    )
    internal_subdivider = None
    if "subclusters" in required_needs:
        min_cluster_size = int(partition.params.get("min_cluster_size", 25))
        min_samples = int(partition.params.get("min_samples", 60))
        subdivision_cache: dict[tuple[int, ...], InternalSubdivision] = {}

        def subdivide(points: list[int]) -> InternalSubdivision:
            key = tuple(int(point) for point in points)
            if key not in subdivision_cache:
                subdivision_cache[key] = diagnostic_density_subdivision(
                    dataset.df,
                    points,
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                )
            return subdivision_cache[key]

        internal_subdivider = subdivide
    ctx = MetricContext(
        n_total=dataset.n_total,
        p_total=dataset.p_total,
        adjacency=adjacency,
        rng=np.random.default_rng(seed),
        internal_subdivider=internal_subdivider,
    )
    stats_by_label = {
        int(region["cluster_label"]): _stats(region, dataset.types)
        for region in partition.regions
    }
    region_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    nulls: dict[str, np.ndarray] = {}
    partitioning = str(partition.params.get("partitioning", partition.method))

    for offset, metric_name in enumerate(metric_names):
        definition = get_metric_definition(metric_name)
        metric = get_metric(metric_name)
        result = metric(partition, dataset.types, ctx)
        scores = np.asarray(result.per_cluster, dtype=float)
        if result.supports_mc and partition.regions and n_alt_worlds > 0:
            null = simulate_null_metric(
                metric,
                partition,
                ctx,
                n_alt_worlds,
                dataset.n_total,
                dataset.p_total,
                seed=seed + offset,
            )
            threshold: float | None = significance_threshold(signif_level, null)
            nulls[metric_name] = null
        else:
            threshold = None

        if metric_name in {"sul", "local_z"}:
            capabilities = get_primary_capabilities(metric_name)
            reference_name = capabilities.rate_reference
            evaluation_mode = "calibrated"
        elif definition.benchmark_candidate:
            capabilities = None
            reference_name = "peers"
            evaluation_mode = "candidate_calibrated"
        elif metric_name == "meanvar":
            capabilities = None
            reference_name = "partition_mean"
            evaluation_mode = "diagnostic"
        else:
            capabilities = None
            reference_name = "not_applicable"
            evaluation_mode = "diagnostic"

        rates = np.array([stats_by_label[int(r["cluster_label"])][2] for r in partition.regions])
        partition_mean = float(np.nanmean(rates)) if len(rates) else float("nan")
        for idx, region in enumerate(partition.regions):
            label = int(region["cluster_label"])
            n, p, rho_in = stats_by_label[label]
            rho_out = (
                (dataset.p_total - p) / (dataset.n_total - n)
                if dataset.n_total > n
                else float("nan")
            )
            peers = adjacency.get(label, [])
            peer_n = sum(stats_by_label[peer][0] for peer in peers)
            peer_p = sum(stats_by_label[peer][1] for peer in peers)
            rho_peer = peer_p / peer_n if peer_n else float("nan")
            score = float(scores[idx])

            indicator_name = "positive_rate"
            indicator_value = rho_in
            reference_value = rho_reference = partition_mean
            effect_value = score
            effect_unit = "native_score"

            if metric_name == "sul":
                rho_reference = rho_out
                reference_value = rho_reference
                effect_value = (rho_in - rho_reference) * 100
                effect_unit = "percentage_points"
                decision = evaluate_primary(
                    metric_name,
                    score=score,
                    threshold=threshold,
                    rho_in=rho_in,
                    rho_reference=rho_reference,
                )
            elif metric_name == "local_z":
                rho_reference = rho_peer
                reference_value = rho_reference
                effect_value = (rho_in - rho_reference) * 100
                effect_unit = "percentage_points"
                decision = evaluate_primary(
                    metric_name,
                    score=score,
                    threshold=threshold,
                    rho_in=rho_in,
                    rho_reference=rho_reference,
                    precondition_reason="menos_de_dois_peers" if len(peers) < 2 else None,
                )
            elif definition.candidate_kind in {"rate_difference", "log_rate_ratio"}:
                rho_reference = rho_peer
                reference_value = rho_reference
                effect_value = score
                effect_unit = (
                    "rate_difference"
                    if definition.candidate_kind == "rate_difference"
                    else "log_rate_ratio"
                )
                decision = _evaluate_candidate(
                    score,
                    threshold,
                    precondition_reason="menos_de_dois_peers" if len(peers) < 2 else None,
                )
            elif definition.candidate_kind == "gini_gap":
                rho_reference = rho_peer
                indicator_name = "internal_gini"
                indicator_value = float(
                    result.per_cluster_metadata["internal_gini"][idx]
                )
                reference_value = float(
                    result.per_cluster_metadata["peer_gini"][idx]
                )
                effect_value = score
                effect_unit = "gini_gap"
                decision = _evaluate_candidate(
                    score,
                    threshold,
                    precondition_reason="menos_de_dois_peers" if len(peers) < 2 else None,
                    outcome_direction=bool(definition.outcome_direction),
                )
            else:
                rho_reference = partition_mean
                decision = None

            region_rows.append(
                {
                    "record_type": "region",
                    "source": "local",
                    "dataset": dataset.name,
                    "protocol": protocol,
                    "method": partition.method,
                    "partitioning": partitioning,
                    "params": json.dumps(partition.params, sort_keys=True),
                    "metric": metric_name,
                    "evaluation_mode": evaluation_mode,
                    "region_id": label,
                    "point_ids": _point_ids(region),
                    **_geometry_fields(region, dataset.df),
                    "n": n,
                    "p": p,
                    "rho_in": rho_in,
                    "rate_reference": reference_name,
                    "rho_reference": rho_reference,
                    "contrast_pp": (rho_in - rho_reference) * 100 if np.isfinite(rho_reference) else np.nan,
                    "indicator_name": indicator_name,
                    "indicator_value": indicator_value,
                    "reference_value": reference_value,
                    "effect_value": effect_value,
                    "effect_unit": effect_unit,
                    "score": score,
                    "threshold": threshold,
                    "evidence_ratio": decision.evidence_ratio if decision else np.nan,
                    "evaluation_status": decision.evaluation_status if decision else "diagnostic",
                    "evaluation_reason": decision.evaluation_reason if decision else "metric_without_detection_contract",
                    "significant": decision.significant if decision else pd.NA,
                    "direction": decision.direction if decision else None,
                    "detection_class": decision.detection_class if decision else None,
                }
            )

        metric_frame = pd.DataFrame(row for row in region_rows if row["metric"] == metric_name)
        best_idx = _best_index(scores)
        best = metric_frame.iloc[best_idx] if best_idx is not None else None
        significant_regions = (
            int((metric_frame["significant"] == True).sum())  # noqa: E712
            if evaluation_mode in {"calibrated", "candidate_calibrated"} and len(metric_frame)
            else 0
            if evaluation_mode in {"calibrated", "candidate_calibrated"}
            else None
        )
        summary_rows.append(
            {
                "record_type": "summary",
                "source": "local",
                "dataset": dataset.name,
                "protocol": protocol,
                "method": partition.method,
                "partitioning": partitioning,
                "params": json.dumps(partition.params, sort_keys=True),
                "metric": metric_name,
                "evaluation_mode": evaluation_mode,
                "rate_semantics": rate_semantics(dataset),
                "N": dataset.n_total,
                "P": dataset.p_total,
                "global_rate": dataset.global_rate,
                "coverage": (dataset.n_total - partition.noise_n) / dataset.n_total if dataset.n_total else np.nan,
                "noise_n": partition.noise_n,
                "n_regions": len(partition.regions),
                "candidate_regions": len(partition.regions),
                "significant_regions": significant_regions,
                "consolidated_regions": significant_regions,
                "partition_score": result.partition_scalar,
                "best_region_id": int(best["region_id"]) if best is not None else None,
                "best_region_n": int(best["n"]) if best is not None else 0,
                "best_region_p": int(best["p"]) if best is not None else 0,
                "best_region_rate": float(best["rho_in"]) if best is not None else np.nan,
                "best_reference_rate": float(best["rho_reference"]) if best is not None else np.nan,
                "best_contrast_pp": float(best["contrast_pp"]) if best is not None else np.nan,
                "best_direction": best["direction"] if best is not None else None,
                "best_indicator_name": best["indicator_name"] if best is not None else None,
                "best_indicator_value": float(best["indicator_value"]) if best is not None and pd.notna(best["indicator_value"]) else np.nan,
                "best_reference_value": float(best["reference_value"]) if best is not None and pd.notna(best["reference_value"]) else np.nan,
                "best_effect_value": float(best["effect_value"]) if best is not None and pd.notna(best["effect_value"]) else np.nan,
                "best_effect_unit": best["effect_unit"] if best is not None else None,
                "score": float(best["score"]) if best is not None else np.nan,
                "threshold": threshold,
                "evidence_ratio": float(best["evidence_ratio"]) if best is not None and pd.notna(best["evidence_ratio"]) else np.nan,
                "seed": seed,
                "n_alt_worlds": n_alt_worlds if result.supports_mc else 0,
                "signif_level": signif_level if result.supports_mc else np.nan,
            }
        )
    return EvaluationBundle(pd.DataFrame(summary_rows), pd.DataFrame(region_rows), nulls)


def evaluate_scan(
    dataset: LoadedDataset,
    regions: list[dict[str, Any]],
    *,
    protocol: str,
    n_alt_worlds: int,
    signif_level: float,
    seed: int,
    direction: str = "both",
) -> EvaluationBundle:
    """Evaluate the authors' overlapping SUL scan before and after consolidation."""
    best_region, max_score, scores = scan_regions(
        regions,
        dataset.types,
        dataset.n_total,
        dataset.p_total,
        direction=direction,
    )
    null = simulate_null_max_suls(
        n_alt_worlds,
        regions,
        dataset.n_total,
        dataset.p_total,
        seed=seed,
        direction=direction,
    )
    threshold = significance_threshold(signif_level, null)
    significant = select_significant_regions(regions, scores, threshold)
    consolidated = filter_non_overlapping_regions(significant, dataset.df)
    significant_ids = {id(region) for region in significant}
    consolidated_ids = {id(region) for region in consolidated}
    rows = []
    for idx, (region, score) in enumerate(zip(regions, scores, strict=True)):
        n, p, rho_in = _stats(region, dataset.types)
        rho_out = (
            (dataset.p_total - p) / (dataset.n_total - n)
            if dataset.n_total > n
            else float("nan")
        )
        rows.append(
            {
                "record_type": "region",
                "source": "local",
                "dataset": dataset.name,
                "protocol": protocol,
                "method": "kmeans_scan",
                "partitioning": f"kmeans_square_scan_{direction}",
                "params": json.dumps({"direction": direction}, sort_keys=True),
                "metric": "sul",
                "evaluation_mode": "calibrated",
                "region_id": idx,
                # Overlapping scans can contain millions of repeated point IDs.
                # Geometry is sufficient for every candidate; persist membership
                # only for the final non-overlapping regions used downstream.
                "point_ids": _point_ids(region) if id(region) in consolidated_ids else None,
                "center": region.get("center"),
                "radius": region.get("radius"),
                "region_type": region.get("type"),
                "bounds": None,
                "geometry": _geometry_fields(region, dataset.df)["geometry"],
                "n": n,
                "p": p,
                "rho_in": rho_in,
                "rate_reference": "outside",
                "rho_reference": rho_out,
                "contrast_pp": (rho_in - rho_out) * 100,
                "score": float(score),
                "threshold": threshold,
                "evidence_ratio": float(score) / threshold if threshold else np.nan,
                "evaluation_status": "evaluated",
                "evaluation_reason": None,
                "significant": id(region) in significant_ids,
                "consolidated": id(region) in consolidated_ids,
                "direction": classify_direction(n, p, dataset.n_total, dataset.p_total),
            }
        )
    frame = pd.DataFrame(rows)
    evaluated_point_ids = {
        int(point)
        for region in regions
        for point in region["points"]
    }
    coverage = len(evaluated_point_ids) / dataset.n_total if dataset.n_total else 0.0
    best_idx = int(np.argmax(scores)) if len(scores) else None
    best = frame.iloc[best_idx] if best_idx is not None else None
    summary = pd.DataFrame(
        [
            {
                "record_type": "summary",
                "source": "local",
                "dataset": dataset.name,
                "protocol": protocol,
                "method": "kmeans_scan",
                "partitioning": f"kmeans_square_scan_{direction}",
                "params": json.dumps({"direction": direction}, sort_keys=True),
                "metric": "sul",
                "evaluation_mode": "calibrated",
                "rate_semantics": rate_semantics(dataset),
                "N": dataset.n_total,
                "P": dataset.p_total,
                "global_rate": dataset.global_rate,
                "coverage": coverage,
                "noise_n": dataset.n_total - len(evaluated_point_ids),
                "n_regions": len(regions),
                "candidate_regions": len(regions),
                "significant_regions": len(significant),
                "consolidated_regions": len(consolidated),
                "partition_score": None,
                "best_region_id": int(best["region_id"]) if best is not None else None,
                "best_region_n": int(best["n"]) if best is not None else 0,
                "best_region_p": int(best["p"]) if best is not None else 0,
                "best_region_rate": float(best["rho_in"]) if best is not None else np.nan,
                "best_reference_rate": float(best["rho_reference"]) if best is not None else np.nan,
                "best_contrast_pp": float(best["contrast_pp"]) if best is not None else np.nan,
                "best_direction": best["direction"] if best is not None else None,
                "score": max_score,
                "threshold": threshold,
                "evidence_ratio": max_score / threshold if threshold else np.nan,
                "seed": seed,
                "n_alt_worlds": n_alt_worlds,
                "signif_level": signif_level,
            }
        ]
    )
    return EvaluationBundle(summary, frame, {"sul": null})
