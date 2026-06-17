"""Folium map helpers for spatial fairness experiment outputs."""

from __future__ import annotations

from pathlib import Path

import folium
import numpy as np
import pandas as pd


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


def _add_hdbscan_hull(mapit: folium.Map, df: pd.DataFrame, types: np.ndarray, region: dict, color: str) -> None:
    point_ids = region["points"]
    if not point_ids:
        return

    subset = df.iloc[point_ids]
    hull = _convex_hull(list(zip(subset["lon"].astype(float), subset["lat"].astype(float))))
    n = len(point_ids)
    p = int(types[point_ids].sum())
    rho = p / n if n else 0.0
    label = region.get("cluster_label", "?")
    tooltip = f"HDBSCAN cluster={label}, n={n}, p={p}, rho={rho:.3f}"

    if len(hull) >= 3:
        folium.Polygon(
            [(lat, lon) for lon, lat in hull],
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.12,
            opacity=0.9,
            weight=3,
            tooltip=tooltip,
        ).add_to(mapit)
        return

    bounds = _bounds_for_points(df, point_ids)
    if bounds:
        _add_region_rectangle(mapit, bounds, color, tooltip)


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
    center = [float(df["lat"].mean()), float(df["lon"].mean())]
    mapit = folium.Map(location=center, zoom_start=5, tiles="CartoDB positron")

    sampled = _sample_points(types, max_points=max_points, seed=seed)
    points_group = folium.FeatureGroup("Sampled outcomes", show=True)
    for point_id in sampled:
        row = df.iloc[int(point_id)]
        color = "#2ca25f" if types[int(point_id)] == 1 else "#de2d26"
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

    mapit.fit_bounds(
        [
            [float(df["lat"].min()), float(df["lon"].min())],
            [float(df["lat"].max()), float(df["lon"].max())],
        ]
    )
    folium.LayerControl().add_to(mapit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapit.save(str(output_path))
