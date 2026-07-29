"""Capped, density-recursive partitioner and shared density-split helpers (ADR-0001).

The size cap is a swept *experimental* knob, not a method constant. Oversized
clusters are split by re-running density clustering on their own points
(`recursive_density_split`), never geometrically. A homogeneous blob that will
not subdivide by density is left intact and flagged (`forced_uncapped`) — a
finding, not a forced cut. `density_subclusters` (one level of the same split)
is reused by the within-cluster inequality metric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN

from clustering.base import Partition
from clustering.hdbscan import fit_hdbscan_partition


def _haversine_coords(df: pd.DataFrame, points: list[int]) -> np.ndarray:
    return np.radians(df.iloc[points][["lat", "lon"]].to_numpy(dtype=float))


def density_subclusters(
    df: pd.DataFrame,
    points: list[int],
    min_cluster_size_min: int = 25,
    min_samples: int = 60,
) -> list[list[int]]:
    """One level of density subdivision; noise reassigned to nearest subcluster.

    Returns `[points]` unchanged when the set is too small or does not subdivide
    (a single dense blob), so callers can detect "unsplittable".
    """
    points = list(points)
    if len(points) < 2 * max(min_cluster_size_min, 2):
        return [points]

    coords = _haversine_coords(df, points)
    mcs = max(min_cluster_size_min, 2)
    labels = HDBSCAN(
        min_cluster_size=mcs,
        min_samples=min(min_samples, mcs),
        metric="haversine",
        cluster_selection_method="eom",
        copy=True,
    ).fit_predict(coords)

    unique = sorted(int(label) for label in set(labels) if label >= 0)
    if len(unique) <= 1:
        return [points]

    idx_by_label = {label: [i for i in range(len(points)) if labels[i] == label] for label in unique}
    groups = {label: [points[i] for i in idx_by_label[label]] for label in unique}
    centroids = {label: coords[idx_by_label[label]].mean(axis=0) for label in unique}

    for i in range(len(points)):
        if labels[i] < 0:
            nearest = min(unique, key=lambda label: float(((coords[i] - centroids[label]) ** 2).sum()))
            groups[nearest].append(points[i])

    return [sorted(groups[label]) for label in unique]


def recursive_density_split(
    df: pd.DataFrame,
    points: list[int],
    max_size: int,
    min_cluster_size_min: int = 25,
    min_samples: int = 60,
) -> list[list[int]]:
    """Split `points` into density pieces of at most `max_size` — **coarse**.

    "Coarse" = cut only enough to fit under the cap, not all the way down to the
    natural fine structure. Each recursion sub-splits with a minimum tied to the
    cap (`max_size // 2`), so a big cluster becomes a *few* pieces near the cap
    (e.g. 5000 with cap 1000 → ~5 pieces of ~1000), not dozens of tiny ones. An
    unsplittable (homogeneous) piece over the cap is returned intact — the caller
    flags it rather than cutting it geometrically.
    """
    points = list(points)
    if len(points) <= max_size:
        return [points]

    coarse_min = max(min_cluster_size_min, max_size // 2)
    subs = density_subclusters(df, points, coarse_min, min_samples)
    if len(subs) <= 1:
        return [points]

    pieces: list[list[int]] = []
    for group in subs:
        pieces.extend(recursive_density_split(df, group, max_size, min_cluster_size_min, min_samples))
    return pieces


def run_capped_hdbscan_sweep(
    df: pd.DataFrame,
    min_cluster_fracs: tuple[float, ...] = (0.01,),
    max_cluster_size: int = 2000,
    min_samples: int = 60,
    min_cluster_size_min: int = 25,
) -> list[Partition]:
    """HDBSCAN partition with oversized clusters recursively split by density."""
    partitions = []
    for frac in min_cluster_fracs:
        base = fit_hdbscan_partition(df, frac, min_cluster_size_min, min_samples)
        regions = []
        label = 0
        for region in base.regions:
            pieces = recursive_density_split(
                df, region["points"], max_cluster_size, min_cluster_size_min, min_samples
            )
            forced = len(pieces) == 1 and len(region["points"]) > max_cluster_size
            for piece in pieces:
                regions.append(
                    {
                        "points": list(piece),
                        "cluster_label": label,
                        "type": "capped_hdbscan",
                        "min_cluster_frac": float(frac),
                        "max_cluster_size": int(max_cluster_size),
                        "over_cap": len(piece) > max_cluster_size,
                        "forced_uncapped": bool(forced and len(piece) > max_cluster_size),
                    }
                )
                label += 1

        labels = np.full(len(df), -1, dtype=int)
        for region in regions:
            labels[region["points"]] = region["cluster_label"]

        partitions.append(
            Partition(
                method="capped_hdbscan",
                params={
                    "min_cluster_frac": float(frac),
                    "max_cluster_size": int(max_cluster_size),
                    "min_samples": int(min_samples),
                },
                labels=labels,
                regions=regions,
                noise_points=base.noise_points,
            )
        )
    return partitions
