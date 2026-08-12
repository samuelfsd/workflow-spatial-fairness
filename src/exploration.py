"""Canonical tables for exploring one validated partition and its detections.

All global figures consume these DataFrames.  This keeps units and denominators
stable and makes the CSV evidence authoritative over any visual simplification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from clustering.base import Partition
from data_loading import LoadedDataset
from metrics.neighbors import build_delaunay_adjacency
from metrics.registry import evaluate_primary, get_primary_capabilities
from run_snapshot import RunSnapshot


EARTH_RADIUS_KM = 6371.0088


@dataclass
class ExplorationTables:
    cluster_features: pd.DataFrame
    coverage_audit: pd.DataFrame
    distribution_summary: pd.DataFrame
    detection_summary: pd.DataFrame
    rankings: pd.DataFrame
    heatmap: pd.DataFrame


def partition_from_snapshot(snapshot: RunSnapshot, n_total: int) -> Partition:
    """Reconstruct the region abstraction using only persisted point assignments."""
    assignments = snapshot.assignments.copy()
    assigned = assignments[assignments["assignment_status"] == "assigned"]
    regions: list[dict] = []
    labels = np.full(n_total, -1, dtype=int)
    for raw_label, rows in assigned.groupby("cluster_label", sort=True):
        label = int(raw_label)
        points = rows["point_id"].astype(int).tolist()
        labels[points] = label
        origin_values = rows["origin"].dropna()
        parent_values = rows["origin_cluster_label"].dropna()
        regions.append(
            {
                "points": points,
                "cluster_label": label,
                "origin": str(origin_values.iloc[0]) if len(origin_values) else "organic",
                "origin_cluster_label": (
                    int(parent_values.iloc[0]) if len(parent_values) else label
                ),
            }
        )
    noise = assignments.loc[
        assignments["assignment_status"] == "unassigned", "point_id"
    ].astype(int).tolist()
    manifest_partition = snapshot.manifest["partition"]
    return Partition(
        method=str(manifest_partition["method"]),
        params=dict(manifest_partition.get("params", {})),
        labels=labels,
        regions=regions,
        noise_points=noise,
    )


def _haversine_to(
    lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float
) -> np.ndarray:
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    lat0_rad = math.radians(lat0)
    lon0_rad = math.radians(lon0)
    inner = (
        np.sin((lat_rad - lat0_rad) / 2.0) ** 2
        + np.cos(lat_rad)
        * math.cos(lat0_rad)
        * np.sin((lon_rad - lon0_rad) / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(inner, 0.0, 1.0)))


def _haversine_between(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return float(_haversine_to(np.array([lat1]), np.array([lon1]), lat2, lon2)[0])


def _distance_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {
            f"{prefix}_mean_km": float("nan"),
            f"{prefix}_median_km": float("nan"),
            f"{prefix}_q1_km": float("nan"),
            f"{prefix}_q3_km": float("nan"),
            f"{prefix}_iqr_km": float("nan"),
            f"{prefix}_p95_km": float("nan"),
        }
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    return {
        f"{prefix}_mean_km": float(np.mean(values)),
        f"{prefix}_median_km": float(median),
        f"{prefix}_q1_km": float(q1),
        f"{prefix}_q3_km": float(q3),
        f"{prefix}_iqr_km": float(q3 - q1),
        f"{prefix}_p95_km": float(np.quantile(values, 0.95)),
    }


def _class_spatial_features(
    lat: np.ndarray,
    lon: np.ndarray,
    outcomes: np.ndarray,
    cluster_lat: float,
    cluster_lon: float,
    value: int,
    name: str,
) -> dict[str, float | str | None]:
    mask = outcomes == value
    if not np.any(mask):
        result: dict[str, float | str | None] = {
            f"{name}_centroid_lat": float("nan"),
            f"{name}_centroid_lon": float("nan"),
            f"{name}_own_dispersion_mean_km": float("nan"),
            f"{name}_own_dispersion_p95_km": float("nan"),
            f"{name}_spatial_reason": "outcome_ausente",
        }
        result.update(_distance_stats(np.array([]), f"distance_{name}"))
        return result

    class_lat = lat[mask]
    class_lon = lon[mask]
    centroid_lat = float(np.mean(class_lat))
    centroid_lon = float(np.mean(class_lon))
    to_general = _haversine_to(class_lat, class_lon, cluster_lat, cluster_lon)
    to_own = _haversine_to(class_lat, class_lon, centroid_lat, centroid_lon)
    result = {
        f"{name}_centroid_lat": centroid_lat,
        f"{name}_centroid_lon": centroid_lon,
        f"{name}_own_dispersion_mean_km": float(np.mean(to_own)),
        f"{name}_own_dispersion_p95_km": float(np.quantile(to_own, 0.95)),
        f"{name}_spatial_reason": None,
    }
    result.update(_distance_stats(to_general, f"distance_{name}"))
    return result


def _spatial_features(
    dataset: LoadedDataset, points: list[int]
) -> dict[str, float | str | None]:
    subset = dataset.df.iloc[points]
    lat = subset["lat"].to_numpy(dtype=float)
    lon = subset["lon"].to_numpy(dtype=float)
    outcomes = dataset.types[points]
    centroid_lat = float(np.mean(lat))
    centroid_lon = float(np.mean(lon))
    distances = _haversine_to(lat, lon, centroid_lat, centroid_lon)
    result: dict[str, float | str | None] = {
        "centroid_lat": centroid_lat,
        "centroid_lon": centroid_lon,
    }
    result.update(_distance_stats(distances, "distance"))
    result.update(
        _class_spatial_features(
            lat, lon, outcomes, centroid_lat, centroid_lon, 1, "positive"
        )
    )
    result.update(
        _class_spatial_features(
            lat, lon, outcomes, centroid_lat, centroid_lon, 0, "negative"
        )
    )

    positive_reason = result["positive_spatial_reason"]
    negative_reason = result["negative_spatial_reason"]
    if positive_reason or negative_reason:
        separation = float("nan")
    else:
        separation = _haversine_between(
            float(result["positive_centroid_lat"]),
            float(result["positive_centroid_lon"]),
            float(result["negative_centroid_lat"]),
            float(result["negative_centroid_lon"]),
        )
    radius_p95 = float(result["distance_p95_km"])
    positive_p95 = float(result["positive_own_dispersion_p95_km"])
    negative_p95 = float(result["negative_own_dispersion_p95_km"])
    dispersion_diff = positive_p95 - negative_p95
    valid_denominator = math.isfinite(radius_p95) and radius_p95 > 0
    result.update(
        {
            "class_centroid_separation_km": separation,
            "class_centroid_separation_relative": (
                separation / radius_p95 if valid_denominator else float("nan")
            ),
            "dispersion_p95_diff_km": dispersion_diff,
            "dispersion_p95_diff_relative": (
                dispersion_diff / radius_p95 if valid_denominator else float("nan")
            ),
            "spatial_comparison_reason": (
                "outcome_ausente"
                if positive_reason or negative_reason
                else (None if valid_denominator else "raio_p95_ausente_ou_zero")
            ),
        }
    )
    return result


def _threshold_row(snapshot: RunSnapshot, primary_metric: str) -> pd.Series:
    rows = snapshot.thresholds[snapshot.thresholds["metric"] == primary_metric]
    if len(rows) != 1:
        raise ValueError(f"Snapshot has no unique threshold row for {primary_metric!r}")
    return rows.iloc[0]


def _score_pivot(snapshot: RunSnapshot) -> pd.DataFrame:
    if snapshot.scores.duplicated(["cluster_label", "metric"]).any():
        raise ValueError("Snapshot has duplicate cluster/metric scores")
    return snapshot.scores.pivot(
        index="cluster_label", columns="metric", values="score"
    )


def _coverage_audit(dataset: LoadedDataset, snapshot: RunSnapshot) -> pd.DataFrame:
    rows = []
    scopes = [
        ("assigned", snapshot.assignments["assignment_status"] == "assigned"),
        (
            "assigned_organic",
            (snapshot.assignments["assignment_status"] == "assigned")
            & (snapshot.assignments["origin"] == "organic"),
        ),
        (
            "assigned_rescue",
            (snapshot.assignments["assignment_status"] == "assigned")
            & (snapshot.assignments["origin"] == "rescue"),
        ),
        ("unassigned", snapshot.assignments["assignment_status"] == "unassigned"),
        ("total", np.ones(len(snapshot.assignments), dtype=bool)),
    ]
    for scope, mask in scopes:
        point_ids = snapshot.assignments.loc[mask, "point_id"].to_numpy(dtype=int)
        n = len(point_ids)
        p = int(dataset.types[point_ids].sum()) if n else 0
        rows.append(
            {
                "scope": scope,
                "n": n,
                "p": p,
                "n_neg": n - p,
                "rho": p / n if n else float("nan"),
                "pct_dataset": n / dataset.n_total if dataset.n_total else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _summary_row(metric: str, values: pd.Series) -> dict[str, float | str | int]:
    finite = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(finite):
        return {key: float("nan") for key in (
            "minimum", "maximum", "mean", "median", "q1", "q2", "q3", "std", "iqr"
        )} | {"metric": metric, "count": 0}
    q1, median, q3 = np.quantile(finite, [0.25, 0.5, 0.75])
    return {
        "metric": metric,
        "count": len(finite),
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "median": float(median),
        "q1": float(q1),
        "q2": float(median),
        "q3": float(q3),
        "std": float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan"),
        "iqr": float(q3 - q1),
    }


def robust_heatmap_frame(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Median/IQR scaling; IQR zero maps finite values to zero, preserving NA."""
    result = frame[["cluster_label"]].copy()
    for metric in metrics:
        values = pd.to_numeric(frame[metric], errors="coerce").astype(float)
        finite = values.dropna()
        if finite.empty:
            result[metric] = np.nan
            continue
        median = float(finite.median())
        iqr = float(finite.quantile(0.75) - finite.quantile(0.25))
        scaled = (values - median) / iqr if iqr > 0 else values.where(values.isna(), 0.0)
        result[metric] = scaled.clip(-3.0, 3.0)
    return result


def build_exploration_tables(
    dataset: LoadedDataset, snapshot: RunSnapshot, primary_metric: str
) -> ExplorationTables:
    """Build all global-report tables without clustering or simulation."""
    capabilities = get_primary_capabilities(primary_metric)
    partition = partition_from_snapshot(snapshot, dataset.n_total)
    adjacency = build_delaunay_adjacency(partition, dataset.df)
    scores = _score_pivot(snapshot)
    if primary_metric not in scores.columns:
        raise ValueError(f"Primary metric {primary_metric!r} was not persisted in the snapshot")
    threshold_row = _threshold_row(snapshot, primary_metric)
    threshold = float(threshold_row["threshold"])

    stats_by_label: dict[int, tuple[int, int]] = {}
    for region in partition.regions:
        points = list(region["points"])
        stats_by_label[int(region["cluster_label"])] = (
            len(points), int(dataset.types[points].sum())
        )

    rows: list[dict] = []
    for region in partition.regions:
        label = int(region["cluster_label"])
        points = list(region["points"])
        n, p = stats_by_label[label]
        rho = p / n if n else float("nan")
        rho_out = (
            (dataset.p_total - p) / (dataset.n_total - n)
            if dataset.n_total > n else float("nan")
        )
        peers = adjacency.get(label, [])
        peer_n = sum(stats_by_label[peer][0] for peer in peers)
        peer_p = sum(stats_by_label[peer][1] for peer in peers)
        rho_peer = peer_p / peer_n if len(peers) >= 2 and peer_n else float("nan")
        reference = rho_peer if capabilities.rate_reference == "peers" else rho_out
        decision = evaluate_primary(
            primary_metric,
            score=float(scores.loc[label, primary_metric]),
            threshold=threshold,
            rho_in=rho,
            rho_reference=reference,
            precondition_reason=(
                "menos_de_dois_peers"
                if capabilities.rate_reference == "peers" and len(peers) < 2
                else None
            ),
        )
        row = {
            "cluster_label": label,
            "origin": region.get("origin", "organic"),
            "origin_cluster_label": region.get("origin_cluster_label", label),
            "n": n,
            "p": p,
            "n_neg": n - p,
            "pct_positive": rho,
            "pct_negative": 1.0 - rho,
            "rho_in": rho,
            "internal_predominance": abs(2.0 * rho - 1.0),
            "rho_global": dataset.global_rate,
            "rho_out": rho_out,
            "rho_peer": rho_peer,
            "n_peers": len(peers),
            "global_deviation": rho - dataset.global_rate,
            "peer_deviation": rho - rho_peer,
            "primary_metric": primary_metric,
            "primary_reference": capabilities.rate_reference,
            "outcome_positive_label": dataset.spec.positive_label,
            "outcome_negative_label": dataset.spec.negative_label,
            "outcome_desirability": dataset.spec.desirability,
            "primary_score": float(scores.loc[label, primary_metric]),
            "signif_threshold": threshold,
            "mc_worlds": int(threshold_row.get("n_worlds", 0)),
            "mc_effective_seed": threshold_row.get("effective_seed"),
            "signif_level": float(snapshot.manifest["run"]["signif_level"]),
            "evidence_ratio": decision.evidence_ratio,
            "evaluation_status": decision.evaluation_status,
            "evaluation_reason": decision.evaluation_reason,
            "direction": decision.direction,
            "significant": decision.significant,
            "detection_class": decision.detection_class,
        }
        for metric in scores.columns:
            row[f"metric_{metric}"] = float(scores.loc[label, metric])
        row.update(_spatial_features(dataset, points))
        rows.append(row)

    features = pd.DataFrame(rows)
    internal_metadata_columns = [
        "internal_subdivision_status",
        "internal_coverage_rate",
        "internal_residue_n",
        "internal_n_subclusters",
        "internal_min_cluster_size",
    ]
    available_internal = [
        column for column in internal_metadata_columns if column in snapshot.scores.columns
    ]
    if available_internal:
        internal = snapshot.scores[
            snapshot.scores["metric"] == "gini_subcluster"
        ][["cluster_label"] + available_internal]
        if not internal.empty:
            features = features.merge(internal, on="cluster_label", how="left", validate="one_to_one")
    # Selection metadata belongs in the canonical table even for the core
    # profile; full merely chooses whether to render the detail bundles.
    from exploration_details import large_cluster_labels, select_clusters

    large_labels = large_cluster_labels(features)
    automatic_selection = select_clusters(features, "auto")
    reasons_by_label = (
        automatic_selection.groupby("cluster_label")["reason"].apply(list).to_dict()
        if not automatic_selection.empty else {}
    )
    features["is_large_cluster"] = features["cluster_label"].isin(large_labels)
    features["is_tukey_outlier"] = features["cluster_label"].map(
        lambda label: any(
            reason.startswith("tukey_") for reason in reasons_by_label.get(int(label), [])
        )
    )
    features["auto_selected"] = features["cluster_label"].isin(reasons_by_label)
    features["selection_reasons"] = features["cluster_label"].map(
        lambda label: ";".join(reasons_by_label.get(int(label), [])) or None
    )
    summaries = pd.DataFrame(
        [
            _summary_row(metric, features[metric])
            for metric in (
                "n", "p", "n_neg", "rho_in", "internal_predominance",
                "distance_mean_km", "distance_p95_km",
                "class_centroid_separation_km", "dispersion_p95_diff_km",
            )
        ]
    )

    evaluated = features[features["evaluation_status"] == "avaliado"]
    detection_rows = []
    for detection_class in ("negative", "positive", "neutral"):
        subset = evaluated[evaluated["detection_class"] == detection_class]
        detection_rows.append(
            {
                "detection_class": detection_class,
                "n_clusters": len(subset),
                "pct_clusters": len(subset) / len(evaluated) if len(evaluated) else float("nan"),
                "n_points": int(subset["n"].sum()),
                "pct_points": (
                    float(subset["n"].sum()) / float(evaluated["n"].sum())
                    if float(evaluated["n"].sum()) else float("nan")
                ),
                "cluster_denominator": len(evaluated),
                "point_denominator": int(evaluated["n"].sum()),
            }
        )
    detection_summary = pd.DataFrame(detection_rows)

    ranking_rows = []
    for metric in (
        "n", "pct_positive", "internal_predominance",
        "global_deviation", "peer_deviation", "distance_mean_km",
        "distance_p95_km", "class_centroid_separation_km",
        "dispersion_p95_diff_km", "evidence_ratio",
    ):
        ranked = features[["cluster_label", metric]].dropna().copy()
        use_magnitude = metric in {
            "global_deviation", "peer_deviation", "dispersion_p95_diff_km"
        }
        ranked["rank_value"] = ranked[metric].abs() if use_magnitude else ranked[metric]
        ranked = ranked.sort_values("rank_value", ascending=False, kind="stable")
        for rank, (_, item) in enumerate(ranked.iterrows(), start=1):
            ranking_rows.append(
                {
                    "metric": metric,
                    "rank": rank,
                    "cluster_label": int(item["cluster_label"]),
                    "value": float(item[metric]),
                    "rank_value": float(item["rank_value"]),
                    "ranking_basis": "magnitude" if use_magnitude else "value",
                }
            )
    rankings = pd.DataFrame(ranking_rows)

    heatmap_metrics = [
        "n", "rho_in", "internal_predominance", "distance_mean_km",
        "distance_p95_km", "class_centroid_separation_km", "primary_score",
    ]
    heatmap = robust_heatmap_frame(features, heatmap_metrics)
    return ExplorationTables(
        cluster_features=features,
        coverage_audit=_coverage_audit(dataset, snapshot),
        distribution_summary=summaries,
        detection_summary=detection_summary,
        rankings=rankings,
        heatmap=heatmap,
    )
