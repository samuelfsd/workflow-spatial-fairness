"""Folium map helpers for spatial fairness experiment outputs.

Maps only: analytical charts are matplotlib (`figures.py`, ADR-0005). Colors come
from the shared `palette` module so maps and figures speak one language.
"""

from __future__ import annotations

from pathlib import Path

import folium
import numpy as np
import pandas as pd

from palette import (
    CATEGORICAL as _CLUSTER_PALETTE,
    COLOR_NEGATIVE as _COLOR_NEGATIVE,
    COLOR_NEUTRAL as _COLOR_NEUTRAL,
    COLOR_NOISE as _COLOR_NOISE,
    COLOR_POSITIVE as _COLOR_POSITIVE,
    POINT_NEGATIVE,
    POINT_POSITIVE,
)


def _sample_points(types: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if len(types) <= max_points:
        return np.arange(len(types), dtype=int)

    rng = np.random.default_rng(seed)
    positives = np.flatnonzero(types == 1)
    negatives = np.flatnonzero(types == 0)
    pos_n = min(len(positives), max_points // 2)
    neg_n = min(len(negatives), max_points - pos_n)

    sampled = []
    if pos_n:
        sampled.append(rng.choice(positives, size=pos_n, replace=False))
    if neg_n:
        sampled.append(rng.choice(negatives, size=neg_n, replace=False))

    return np.sort(np.concatenate(sampled)).astype(int)


def _bounds_for_points(df: pd.DataFrame, points: list[int]) -> tuple[float, float, float, float] | None:
    if not points:
        return None
    subset = df.iloc[points]
    return (
        float(subset["lon"].min()),
        float(subset["lat"].min()),
        float(subset["lon"].max()),
        float(subset["lat"].max()),
    )


def _cross(origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return convex hull points as `(lon, lat)` using Andrew's monotonic chain."""
    unique_points = sorted(set(points))
    if len(unique_points) <= 1:
        return unique_points

    lower = []
    for point in unique_points:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper = []
    for point in reversed(unique_points):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def _add_region_rectangle(mapit: folium.Map, bounds: tuple[float, float, float, float], color: str, tooltip: str) -> None:
    lon_min, lat_min, lon_max, lat_max = bounds
    folium.Rectangle(
        [(lat_min, lon_min), (lat_max, lon_max)],
        color=color,
        fill=False,
        weight=3,
        tooltip=tooltip,
    ).add_to(mapit)


def _add_region_hull(
    mapit: folium.Map | folium.FeatureGroup,
    df: pd.DataFrame,
    region: dict,
    color: str,
    tooltip: str,
    fill_opacity: float = 0.12,
) -> None:
    """Draw a region as a convex hull polygon, falling back to a bounding box."""
    point_ids = region["points"]
    if not point_ids:
        return

    subset = df.iloc[point_ids]
    hull = _convex_hull(list(zip(subset["lon"].astype(float), subset["lat"].astype(float))))

    if len(hull) >= 3:
        folium.Polygon(
            [(lat, lon) for lon, lat in hull],
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=fill_opacity,
            opacity=0.9,
            weight=3,
            tooltip=tooltip,
        ).add_to(mapit)
        return

    bounds = _bounds_for_points(df, point_ids)
    if bounds:
        _add_region_rectangle(mapit, bounds, color, tooltip)


def _add_hdbscan_hull(mapit: folium.Map, df: pd.DataFrame, types: np.ndarray, region: dict, color: str) -> None:
    point_ids = region["points"]
    if not point_ids:
        return

    n = len(point_ids)
    p = int(types[point_ids].sum())
    rho = p / n if n else 0.0
    label = region.get("cluster_label", "?")
    tooltip = f"HDBSCAN cluster={label}, n={n}, p={p}, rho={rho:.3f}"
    _add_region_hull(mapit, df, region, color, tooltip)


def _base_map(df: pd.DataFrame) -> folium.Map:
    center = [float(df["lat"].mean()), float(df["lon"].mean())]
    return folium.Map(location=center, zoom_start=5, tiles="CartoDB positron")


def _finalize_map(mapit: folium.Map, df: pd.DataFrame, output_path: Path) -> None:
    mapit.fit_bounds(
        [
            [float(df["lat"].min()), float(df["lon"].min())],
            [float(df["lat"].max()), float(df["lon"].max())],
        ]
    )
    folium.LayerControl().add_to(mapit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapit.save(str(output_path))


def save_clustering_stage_map(
    df: pd.DataFrame,
    types: np.ndarray,
    partition,
    output_path: Path,
    *,
    max_points: int = 5000,
    seed: int = 42,
) -> None:
    """Stage-1 map: every cluster (single neutral color) over the raw outcome points.

    Clusters carry no judgment at this stage, so they all share one color;
    the sampled points are colored by outcome (green=1, red=0) to show the
    raw data the clustering grouped — the algorithm itself never saw outcomes.
    """
    mapit = _base_map(df)

    # One shared sample for both point layers: every drawn point carries its
    # real assignment status, so no unassigned point can look assigned.
    sampled = _sample_points(types, max_points=max_points, seed=seed)
    noise_set = set(int(point) for point in partition.noise_points)

    points_group = folium.FeatureGroup("Amostra de pontos (outcome 1=verde, 0=vermelho)", show=True)
    for point_id in sampled:
        row = df.iloc[int(point_id)]
        color = POINT_POSITIVE if types[int(point_id)] == 1 else POINT_NEGATIVE
        folium.CircleMarker(
            location=(float(row["lat"]), float(row["lon"])),
            radius=2,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.45,
            opacity=0.45,
        ).add_to(points_group)
    points_group.add_to(mapit)

    sampled_noise = [int(point) for point in sampled if int(point) in noise_set]
    noise_group = folium.FeatureGroup(
        f"Pontos não atribuídos a nenhum cluster — fora da comparação "
        f"({len(sampled_noise)} na amostra; {len(noise_set)} no total)",
        show=True,
    )
    for point_id in sampled_noise:
        row = df.iloc[point_id]
        folium.CircleMarker(
            location=(float(row["lat"]), float(row["lon"])),
            radius=1.5,
            color=_COLOR_NOISE,
            fill=True,
            fill_color=_COLOR_NOISE,
            fill_opacity=0.6,
            opacity=0.6,
        ).add_to(noise_group)
    noise_group.add_to(mapit)

    clusters_group = folium.FeatureGroup("Clusters formados (antes das métricas)", show=True)
    for region in partition.regions:
        n, label = len(region["points"]), region.get("cluster_label", "?")
        p = int(types[region["points"]].sum()) if n else 0
        rho = p / n if n else 0.0
        tooltip = f"cluster={label}, n={n}, p={p}, rho={rho:.3f}"
        _add_region_hull(clusters_group, df, region, _CLUSTER_PALETTE[0], tooltip, fill_opacity=0.08)
    clusters_group.add_to(mapit)

    _finalize_map(mapit, df, output_path)


def save_detection_stage_map(
    df: pd.DataFrame,
    types: np.ndarray,
    region_results: list[dict],
    output_path: Path,
    *,
    threshold: float,
    global_rate: float,
    max_points: int = 5000,
    seed: int = 42,
) -> None:
    """Final map: significant regions red (negative) / green (positive), rest gray.

    `region_results` items: {"region", "n", "p", "rho", "rho_out", "sul",
    "significant", "direction"} as built by ExperimentRunner.run_explain.
    """
    mapit = _base_map(df)

    groups = {
        "negative": folium.FeatureGroup("Significativas: injustiça negativa", show=True),
        "positive": folium.FeatureGroup("Significativas: injustiça positiva", show=True),
        "neutral": folium.FeatureGroup("Não significativas", show=True),
    }
    colors = {"negative": _COLOR_NEGATIVE, "positive": _COLOR_POSITIVE, "neutral": _COLOR_NEUTRAL}

    for result in region_results:
        key = result["direction"] if result["significant"] else "neutral"
        status = f"SIGNIFICANT ({result['direction']})" if result["significant"] else "not significant"
        label = result["region"].get("cluster_label", "?")
        score = result.get("score", result.get("sul"))
        score_name = result.get("score_name", "SUL")
        tooltip = (
            f"cluster={label} | {score_name}={score:.2f} (limiar={threshold:.2f}) | "
            f"n={result['n']}, p={result['p']} | "
            f"rho_in={result['rho']:.3f} vs rho_out={result['rho_out']:.3f} "
            f"(global={global_rate:.3f}) | {status}"
        )
        fill_opacity = 0.25 if result["significant"] else 0.08
        _add_region_hull(groups[key], df, result["region"], colors[key], tooltip, fill_opacity=fill_opacity)

    for group in groups.values():
        group.add_to(mapit)

    sampled = _sample_points(types, max_points=max_points, seed=seed)
    points_group = folium.FeatureGroup("Amostra de pontos (outcome 1=verde, 0=vermelho)", show=False)
    for point_id in sampled:
        row = df.iloc[int(point_id)]
        color = POINT_POSITIVE if types[int(point_id)] == 1 else POINT_NEGATIVE
        folium.CircleMarker(
            location=(float(row["lat"]), float(row["lon"])),
            radius=2,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.45,
            opacity=0.45,
        ).add_to(points_group)
    points_group.add_to(mapit)

    _finalize_map(mapit, df, output_path)


def save_experiment_map(
    df: pd.DataFrame,
    types: np.ndarray,
    output_path: Path,
    *,
    grid_regions: list[dict] | None = None,
    box_regions: list[dict] | None = None,
    hdbscan_regions: list[dict] | None = None,
    max_points: int = 5000,
    seed: int = 42,
) -> None:
    mapit = _base_map(df)

    sampled = _sample_points(types, max_points=max_points, seed=seed)
    points_group = folium.FeatureGroup("Sampled outcomes", show=True)
    for point_id in sampled:
        row = df.iloc[int(point_id)]
        color = POINT_POSITIVE if types[int(point_id)] == 1 else POINT_NEGATIVE
        folium.CircleMarker(
            location=(float(row["lat"]), float(row["lon"])),
            radius=2,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.45,
            opacity=0.45,
        ).add_to(points_group)
    points_group.add_to(mapit)

    for region in grid_regions or []:
        bounds = region.get("bounds")
        if bounds:
            _add_region_rectangle(mapit, bounds, "#3182bd", "significant grid region")

    for region in box_regions or []:
        center_id = region.get("center")
        radius = region.get("radius")
        if center_id is None or radius is None:
            continue
        center_row = df.iloc[int(center_id)]
        lat = float(center_row["lat"])
        lon = float(center_row["lon"])
        _add_region_rectangle(
            mapit,
            (lon - radius, lat - radius, lon + radius, lat + radius),
            "#756bb1",
            "non-overlapping KMeans scan region",
        )

    for region in hdbscan_regions or []:
        _add_hdbscan_hull(mapit, df, types, region, "#f16913")

    _finalize_map(mapit, df, output_path)
