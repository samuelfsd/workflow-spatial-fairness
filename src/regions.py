"""Spatial region generation matching the original auditing workflow."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
from rtree import index
from sklearn.cluster import KMeans


def create_rtree(df: pd.DataFrame) -> index.Index:
    rtree = index.Index()
    for idx, row in df.iterrows():
        rtree.insert(int(idx), (row["lon"], row["lat"], row["lon"], row["lat"]))
    return rtree


def id2loc(df: pd.DataFrame, point_id: int) -> tuple[float, float]:
    row = df.loc[int(point_id)]
    return float(row["lat"]), float(row["lon"])


def query_range_box(
    rtree: index.Index,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> list[int]:
    return list(rtree.intersection((xmin, ymin, xmax, ymax)))


def query_range(df: pd.DataFrame, rtree: index.Index, center: int, radius: float) -> list[int]:
    lat, lon = id2loc(df, center)
    return query_range_box(rtree, lon - radius, lon + radius, lat - radius, lat + radius)


def create_seeds(
    df: pd.DataFrame,
    rtree: index.Index,
    n_seeds: int,
    random_state: int | None = 42,
) -> list[int]:
    n_clusters = min(n_seeds, len(df))
    x = df[["lon", "lat"]].to_numpy()
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state).fit(x)

    seeds = []
    for center in kmeans.cluster_centers_:
        seeds.append(list(rtree.nearest([center[0], center[1]], 1))[0])

    return seeds


def create_regions(
    df: pd.DataFrame,
    rtree: index.Index,
    seeds: list[int],
    radii: np.ndarray,
) -> list[dict]:
    regions = []
    for seed in seeds:
        for radius in radii:
            regions.append(
                {
                    "points": query_range(df, rtree, seed, float(radius)),
                    "center": int(seed),
                    "radius": float(radius),
                    "type": "kmeans",
                }
            )
    return regions


def create_partitioning(
    df: pd.DataFrame,
    rtree: index.Index,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    lon_n: int,
    lat_n: int,
) -> tuple[dict, dict[tuple[int, int], int], list[dict]]:
    grid_info = {
        "lon_min": float(lon_min),
        "lon_max": float(lon_max),
        "lat_min": float(lat_min),
        "lat_max": float(lat_max),
        "lon_n": int(lon_n),
        "lat_n": int(lat_n),
    }
    grid_loc2_idx = {}
    partitions = []

    for i in range(int(lat_n)):
        lat_start = lat_min + (i / lat_n) * (lat_max - lat_min)
        lat_end = lat_min + ((i + 1) / lat_n) * (lat_max - lat_min)
        for j in range(int(lon_n)):
            lon_start = lon_min + (j / lon_n) * (lon_max - lon_min)
            lon_end = lon_min + ((j + 1) / lon_n) * (lon_max - lon_min)
            partition = {
                "grid_loc": (j, i),
                "points": query_range_box(rtree, lon_start, lon_end, lat_start, lat_end),
                "bounds": (lon_start, lat_start, lon_end, lat_end),
                "type": "grid",
            }
            grid_loc2_idx[(j, i)] = len(partitions)
            partitions.append(partition)

    return grid_info, grid_loc2_idx, partitions


def create_grid_from_dataset(
    df: pd.DataFrame,
    rtree: index.Index,
    lon_n: int,
    lat_n: int,
) -> tuple[dict, dict[tuple[int, int], int], list[dict]]:
    return create_partitioning(
        df,
        rtree,
        float(df["lon"].min()),
        float(df["lon"].max()),
        float(df["lat"].min()),
        float(df["lat"].max()),
        lon_n,
        lat_n,
    )


def create_random_partitionings(
    df: pd.DataFrame,
    rtree: index.Index,
    n_partitionings: int = 100,
    lon_n_range: tuple[int, int] = (10, 40),
    lat_n_range: tuple[int, int] = (10, 40),
    seed: int = 42,
) -> list[tuple[dict, dict[tuple[int, int], int], list[dict]]]:
    rng = random.Random(seed)
    partitionings = []
    for _ in range(n_partitionings):
        lon_n = rng.randint(*lon_n_range)
        lat_n = rng.randint(*lat_n_range)
        partitionings.append(create_grid_from_dataset(df, rtree, lon_n=lon_n, lat_n=lat_n))
    return partitionings


def regions_intersect(region_a: dict, region_b: dict, df: pd.DataFrame) -> bool:
    center_a = np.asarray(id2loc(df, region_a["center"]))
    center_b = np.asarray(id2loc(df, region_b["center"]))
    radius_a = region_a["radius"]
    radius_b = region_b["radius"]

    a_top_right = center_a + np.asarray([radius_a, radius_a])
    a_bottom_left = center_a - np.asarray([radius_a, radius_a])
    b_top_right = center_b + np.asarray([radius_b, radius_b])
    b_bottom_left = center_b - np.asarray([radius_b, radius_b])

    return not (
        a_top_right[0] < b_bottom_left[0]
        or a_bottom_left[0] > b_top_right[0]
        or a_top_right[1] < b_bottom_left[1]
        or a_bottom_left[1] > b_top_right[1]
    )


def filter_non_overlapping_regions(regions: list[dict], df: pd.DataFrame) -> list[dict]:
    non_overlapping = []
    centers = set()
    for region in regions:
        center = region.get("center")
        if center in centers:
            continue

        if all(not regions_intersect(region, other, df) for other in non_overlapping):
            centers.add(center)
            non_overlapping.append(region)

    return non_overlapping
