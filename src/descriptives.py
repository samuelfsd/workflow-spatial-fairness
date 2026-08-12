"""Descriptive characterization of a partition — never a fairness verdict.

These numbers describe how the *clustering* behaved (balance of positives and
negatives, dispersion of cluster sizes, spatial compactness), not the audited
phenomenon. They deliberately live outside `METRICS`: mixing characterization
with verdict is the confusion the metric taxonomy exists to prevent (see
`CONTEXT.md`, "Relatório de partição").

Everything here returns a pandas object, so tables, figures and the PDF report
all consume the same frames.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from clustering.base import Partition
from metrics.base import MetricContext
from metrics.group_fairness import calculate_gini, get_simple_stats
from metrics.neighbors import build_delaunay_adjacency
from metrics.registry import get_metric

EARTH_RADIUS_KM = 6371.0088

#: Variables profiled by `dispersion_summary`, in reading order.
DISPERSION_VARIABLES = ("n", "p", "n_neg", "rho", "raio_medio_km")

Splitter = Callable[[list[int]], list[list[int]]]


def _haversine_km(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float) -> np.ndarray:
    """Great-circle distance in km from every (lat, lon) to a single origin."""
    lat_rad, lon_rad = np.radians(lat), np.radians(lon)
    lat0_rad, lon0_rad = math.radians(lat0), math.radians(lon0)
    sin_dlat = np.sin((lat_rad - lat0_rad) / 2.0) ** 2
    sin_dlon = np.sin((lon_rad - lon0_rad) / 2.0) ** 2
    inner = sin_dlat + np.cos(lat_rad) * math.cos(lat0_rad) * sin_dlon
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(inner, 0.0, 1.0)))


def _compactness_km(df: pd.DataFrame, points: list[int]) -> tuple[float, float]:
    """Mean and p95 distance from the cluster's points to its centroid, in km.

    Kilometres, never degrees: one degree of longitude is 111·cos(lat) km, so
    lat/lon spreads are not on a common ruler.
    """
    if not len(points):
        return float("nan"), float("nan")

    subset = df.iloc[points]
    lat = subset["lat"].to_numpy(dtype=float)
    lon = subset["lon"].to_numpy(dtype=float)
    distances = _haversine_km(lat, lon, float(lat.mean()), float(lon.mean()))
    return float(distances.mean()), float(np.percentile(distances, 95))


def cluster_frame(df: pd.DataFrame, partition: Partition, types: np.ndarray) -> pd.DataFrame:
    """One row per cluster: balance (`n`, `p`, `n_neg`, `rho`) and compactness.

    Row order follows `partition.regions`, so the frame is index-aligned with
    every `MetricResult.per_cluster`.
    """
    rows = []
    for region in partition.regions:
        points = list(region["points"])
        n, p, rho = get_simple_stats(points, types)
        raio_medio, raio_p95 = _compactness_km(df, points)
        rows.append(
            {
                "cluster_label": region.get("cluster_label"),
                "origin": region.get("origin", "organic"),
                "origin_cluster_label": region.get(
                    "origin_cluster_label", region.get("cluster_label")
                ),
                "n": n,
                "p": p,
                "n_neg": n - p,
                "rho": rho,
                "raio_medio_km": raio_medio,
                "raio_p95_km": raio_p95,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "cluster_label",
            "origin",
            "origin_cluster_label",
            "n",
            "p",
            "n_neg",
            "rho",
            "raio_medio_km",
            "raio_p95_km",
        ],
    )


def dispersion_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Dispersion of each variable *across* clusters: mean, std, var, CV, min, max.

    Read in three layers, and the first one carries a caveat:

    1. raw `std`/`var` of counts — describes the **balance of the partition**;
       comparing sigma(p) with sigma(n_neg) is driven by the global rate (see
       `expected_sigma_ratio`), so it is arithmetic, not a fairness finding;
    2. `cv` — the scale-free version, and the only one comparable across
       variables and across configurations;
    3. dispersion of `rho` — the layer that speaks about fairness (it is the
       square root of MeanVar, up to the ddof convention).
    """
    variables = [name for name in DISPERSION_VARIABLES if name in frame.columns]
    summary = pd.DataFrame(index=pd.Index(variables, name="variable"))
    if frame.empty:
        for column in ("mean", "std", "var", "cv", "min", "max"):
            summary[column] = np.nan
        return summary

    values = frame[variables].astype(float)
    summary["mean"] = values.mean()
    summary["std"] = values.std()  # ddof=1, pandas default
    summary["var"] = values.var()
    summary["cv"] = summary["std"] / summary["mean"].replace(0.0, np.nan)
    summary["min"] = values.min()
    summary["max"] = values.max()
    return summary


def compare_configs(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack `dispersion_summary` side by side: one column per configuration.

    This is the table that instructs the size-cap decision (ADR-0001): the same
    dispersion read for `hdbscan` and for a capped partition.
    """
    columns = {}
    for name, frame in frames.items():
        summary = dispersion_summary(frame)
        columns[name] = summary.stack()

    table = pd.DataFrame(columns)
    table.index = table.index.set_names(["variable", "stat"])
    return table


def dataset_balance(types: np.ndarray) -> dict[str, Any]:
    """Dataset-level balance: total points, positives, negatives, global rate."""
    n_total = int(len(types))
    p_total = int(np.asarray(types).sum())
    return {
        "N": n_total,
        "P": p_total,
        "n_neg": n_total - p_total,
        "global_rate": (p_total / n_total) if n_total else float("nan"),
    }


def expected_sigma_ratio(global_rate: float) -> float:
    """`sigma(p) / sigma(n_neg)` expected when every cluster shares one rate.

    If all clusters had the same rate, `p = rho·n` and `n_neg = (1-rho)·n`, so
    this ratio is forced to `rho / (1 - rho)` by arithmetic alone. Comparing the
    observed ratio against this value is what turns "positives vary more" from a
    foregone conclusion into a question worth asking.
    """
    if not 0.0 <= global_rate < 1.0:
        return float("nan")
    return global_rate / (1.0 - global_rate)


def standardized_residuals(frame: pd.DataFrame, global_rate: float) -> np.ndarray:
    """`(p - rho·n) / sqrt(n·rho(1-rho))` per cluster — the count-based deviation.

    The correct way to read dispersion of counts: how many more (or fewer)
    positives a cluster holds than its size explains. It is a z-score against
    the global baseline, which is why the descriptive layer lands in the same
    family as the audit metrics.
    """
    n = frame["n"].to_numpy(dtype=float)
    p = frame["p"].to_numpy(dtype=float)
    variance = n * global_rate * (1.0 - global_rate)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(variance > 0, (p - global_rate * n) / np.sqrt(variance), np.nan)


def partition_profile(partition: Partition, n_total: int) -> dict[str, Any]:
    """Scalars describing one partition: regions, unassigned share, cap outcomes.

    Two distinct cap readings, because they are not the same thing: a cluster the
    density split refuses to divide at all (`forced_uncapped`) and any piece that
    ends up over the cap (`over_cap`) — a recursion can leave an oversized piece
    behind while the cluster as a whole did split. Both are findings under
    ADR-0001, not failures to hide.
    """
    organic_n = sum(
        len(region["points"])
        for region in partition.regions
        if region.get("origin", "organic") == "organic"
    )
    rescue_n = sum(
        len(region["points"])
        for region in partition.regions
        if region.get("origin", "organic") == "rescue"
    )
    return {
        "method": partition.method,
        "n_regions": len(partition.regions),
        "organic_n": organic_n,
        "organic_rate": (organic_n / n_total) if n_total else float("nan"),
        "rescue_n": rescue_n,
        "rescue_rate": (rescue_n / n_total) if n_total else float("nan"),
        "assigned_n": organic_n + rescue_n,
        "coverage_rate": ((organic_n + rescue_n) / n_total) if n_total else float("nan"),
        "noise_n": partition.noise_n,
        "noise_rate": (partition.noise_n / n_total) if n_total else float("nan"),
        "forced_uncapped": sum(
            1 for region in partition.regions if region.get("forced_uncapped")
        ),
        "over_cap": sum(1 for region in partition.regions if region.get("over_cap")),
        "stat_cap_targets": int(partition.params.get("stat_cap_targets", 0)),
        "stat_cap_refusals": int(partition.params.get("stat_cap_refusals", 0)),
        "stat_leaf_split_parents": int(
            partition.params.get("stat_leaf_split_parents", 0)
        ),
        "stat_leaf_refusals": int(partition.params.get("stat_leaf_refusals", 0)),
        "stat_leaf_noise_n": int(partition.params.get("stat_leaf_noise_n", 0)),
        "cluster_size_cv_before_stat_cap": partition.params.get(
            "cluster_size_cv_before_stat_cap"
        ),
        "cluster_size_cv_after_stat_cap": partition.params.get(
            "cluster_size_cv_after_stat_cap"
        ),
    }


def _partition_before_stat_cap(
    partition: Partition,
    *,
    n_total: int,
    include_rescue: bool,
) -> Partition:
    """Reassemble origin-parent clusters from a final rescue partition."""
    grouped: dict[tuple[str, int], list[int]] = {}
    for region in partition.regions:
        origin = str(region.get("origin", "organic"))
        if origin == "rescue" and not include_rescue:
            continue
        parent = int(region.get("origin_cluster_label", region.get("cluster_label", -1)))
        grouped.setdefault((origin, parent), []).extend(int(point) for point in region["points"])

    labels = np.full(n_total, -1, dtype=int)
    regions = []
    for label, ((origin, parent), points) in enumerate(grouped.items()):
        unique_points = sorted(set(points))
        labels[unique_points] = label
        regions.append(
            {
                "points": unique_points,
                "cluster_label": label,
                "origin": origin,
                "origin_cluster_label": parent,
            }
        )
    return Partition(
        method=f"{partition.method}_before_stat_cap",
        params={},
        labels=labels,
        regions=regions,
        noise_points=np.flatnonzero(labels == -1).astype(int).tolist(),
    )


def organic_local_z_deltas(
    partition: Partition,
    df: pd.DataFrame,
    types: np.ndarray,
    *,
    n_total: int,
    p_total: int,
) -> pd.DataFrame:
    """Local-z change on organic clusters caused only by adding rescue peers.

    Origin-parent labels reconstruct both comparison partitions before the
    statistical cap: organic-only and organic+rescue. This isolates the graph
    perturbation from the later density redivision.
    """
    organic = _partition_before_stat_cap(
        partition, n_total=n_total, include_rescue=False
    )
    combined = _partition_before_stat_cap(
        partition, n_total=n_total, include_rescue=True
    )

    def scores_by_parent(candidate: Partition) -> dict[tuple[str, int], float]:
        adjacency = build_delaunay_adjacency(candidate, df)
        ctx = MetricContext(
            n_total=n_total,
            p_total=p_total,
            adjacency=adjacency,
            rng=np.random.default_rng(0),
        )
        scores = get_metric("local_z")(candidate, types, ctx).per_cluster
        return {
            (
                str(region.get("origin", "organic")),
                int(region["origin_cluster_label"]),
            ): float(scores[idx])
            for idx, region in enumerate(candidate.regions)
        }

    before = scores_by_parent(organic)
    after = scores_by_parent(combined)
    rows = []
    for (origin, parent), value_before in before.items():
        value_after = after.get((origin, parent), float("nan"))
        rows.append(
            {
                "origin_cluster_label": parent,
                "local_z_before_rescue": value_before,
                "local_z_after_rescue": value_after,
                "local_z_delta": value_after - value_before,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "origin_cluster_label",
            "local_z_before_rescue",
            "local_z_after_rescue",
            "local_z_delta",
        ],
    )


def subcluster_frame(points: list[int], types: np.ndarray, splitter: Splitter) -> pd.DataFrame:
    """One row per density subcluster of a cluster: `n`, `p`, `n_neg`, `rho`."""
    rows = []
    for idx, subset in enumerate(splitter(list(points))):
        n, p, rho = get_simple_stats(subset, types)
        rows.append({"subcluster": idx, "n": n, "p": p, "n_neg": n - p, "rho": rho})
    return pd.DataFrame(rows, columns=["subcluster", "n", "p", "n_neg", "rho"])


def peer_rate(frame: pd.DataFrame, adjacency: dict[int, list[int]], cluster_label: int) -> float:
    """Pooled positive rate of a cluster's Delaunay peers (size-weighted).

    NaN when the cluster has fewer than 2 peers — the same "not evaluated by
    neighbourhood" rule the local z-score applies, so the card never shows a
    reference line the metric itself refuses to trust.
    """
    peers = adjacency.get(cluster_label, [])
    if len(peers) < 2:
        return float("nan")

    subset = frame[frame["cluster_label"].isin(peers)]
    n_peer = float(subset["n"].sum())
    if n_peer == 0:
        return float("nan")
    return float(subset["p"].sum()) / n_peer


def cluster_card_data(
    df: pd.DataFrame,
    partition: Partition,
    types: np.ndarray,
    *,
    cluster_label: int,
    splitter: Splitter,
    adjacency: dict[int, list[int]],
    global_rate: float,
    frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Everything one cluster card shows: subcluster rates + 3 reference rates.

    Intra-cluster is the subcluster frame; extra-cluster are `rho_in`,
    `rho_peer` and `rho_global`. `homogeneous` marks a cluster the density split
    refuses to divide — a finding to state in words, not an empty chart.

    Pass `frame` when the caller already built the per-cluster frame: recomputing
    it per card would redo the haversine pass over every point in the dataset.
    """
    frame = cluster_frame(df, partition, types) if frame is None else frame
    match = frame[frame["cluster_label"] == cluster_label]
    if match.empty:
        raise ValueError(f"Unknown cluster_label: {cluster_label}")

    row = match.iloc[0]
    region = next(
        region for region in partition.regions if region.get("cluster_label") == cluster_label
    )
    subclusters = subcluster_frame(list(region["points"]), types, splitter)

    return {
        "cluster_label": cluster_label,
        "n": int(row["n"]),
        "p": int(row["p"]),
        "n_neg": int(row["n_neg"]),
        "rho_in": float(row["rho"]),
        "rho_peer": peer_rate(frame, adjacency, cluster_label),
        "rho_global": float(global_rate),
        "gini_subcluster": calculate_gini(subclusters["rho"]),
        "raio_medio_km": float(row["raio_medio_km"]),
        "subclusters": subclusters,
        "homogeneous": len(subclusters) <= 1,
    }
