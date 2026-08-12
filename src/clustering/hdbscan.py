"""HDBSCAN-based organic spatial partitioning."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN

from clustering.base import Partition

# Backward-compat alias: earlier code imported a dedicated HDBSCAN dataclass.
HDBSCANPartition = Partition


def effective_min_cluster_size(
    n_points: int, min_cluster_frac: float, min_cluster_size_min: int = 25
) -> int:
    """Smallest cluster HDBSCAN will accept for this dataset and fraction.

    Exposed because it is also the **floor of any size cap**: asking for a cap at
    or below this value describes an empty set ("no cluster smaller than X and
    none larger than Y ≤ X"), and HDBSCAN answers it by returning no clusters at
    all. On LAR at frac 0.005 the floor is 1.032 points.
    """
    return max(min_cluster_size_min, int(round(min_cluster_frac * n_points)))


def fit_hdbscan_partition(
    df: pd.DataFrame,
    min_cluster_frac: float,
    min_cluster_size_min: int = 25,
    min_samples: int = 60,
    max_cluster_size: int | None = None,
    min_cluster_size_override: int | None = None,
    cluster_selection_method: str = "eom",
) -> Partition:
    """Fit one HDBSCAN partition using haversine distance on lat/lon.

    `min_samples` is capped at `min_cluster_size` and kept small by default:
    sklearn's fallback (min_samples = min_cluster_size) allocates an
    n_points x min_samples core-distance matrix, which OOMs on large datasets
    when min_cluster_size is a fraction of the dataset.

    `max_cluster_size` (optional) is HDBSCAN's own EOM limit: the excess-of-mass
    selection will not return a cluster larger than this, descending the density
    hierarchy to smaller sub-clusters instead — an *organic* cap (no geometric
    split). Points that fall between selected clusters become unassigned, so a
    tighter cap tends to raise the noise fraction. See ADR-0001.

    `min_cluster_size_override` preserves an absolute region-size ruler when a
    subset is fitted. The rescue pass uses it so ``frac × N`` from the full
    dataset does not silently become ``frac × noise_n``.

    `cluster_selection_method="leaf"` exposes the finest stable leaves of the
    same density hierarchy. The organic default remains ``"eom"``; leaf is
    reserved for explicit refinement experiments.
    """
    min_cluster_size = (
        int(min_cluster_size_override)
        if min_cluster_size_override is not None
        else effective_min_cluster_size(len(df), min_cluster_frac, min_cluster_size_min)
    )
    if min_cluster_size < 2:
        raise ValueError("min_cluster_size must be at least 2")
    effective_min_samples = min(min_samples, min_cluster_size)
    if len(df) < min_cluster_size:
        labels = np.full(len(df), -1, dtype=int)
        return Partition(
            method="hdbscan",
            params={
                "min_cluster_frac": float(min_cluster_frac),
                "min_cluster_size": int(min_cluster_size),
                "min_samples": int(effective_min_samples),
                "max_cluster_size": max_cluster_size,
                "metric": "haversine",
                "cluster_selection_method": cluster_selection_method,
            },
            labels=labels,
            regions=[],
            noise_points=list(range(len(df))),
        )
    coords = np.radians(df[["lat", "lon"]].to_numpy(dtype=float))

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=effective_min_samples,
        max_cluster_size=max_cluster_size,
        metric="haversine",
        cluster_selection_method=cluster_selection_method,
        copy=True,
    )
    labels = clusterer.fit_predict(coords)

    regions = []
    for label in sorted(int(value) for value in np.unique(labels) if value >= 0):
        points = np.flatnonzero(labels == label).astype(int).tolist()
        regions.append(
            {
                "points": points,
                "cluster_label": label,
                "type": "hdbscan",
                "min_cluster_frac": float(min_cluster_frac),
                "min_cluster_size": int(min_cluster_size),
                "max_cluster_size": max_cluster_size,
                "cluster_selection_method": cluster_selection_method,
            }
        )

    noise_points = np.flatnonzero(labels == -1).astype(int).tolist()
    return Partition(
        method="hdbscan",
        params={
            "min_cluster_frac": float(min_cluster_frac),
            "min_cluster_size": int(min_cluster_size),
            "min_samples": int(effective_min_samples),
            "max_cluster_size": max_cluster_size,
            "metric": "haversine",
            "cluster_selection_method": cluster_selection_method,
        },
        labels=labels,
        regions=regions,
        noise_points=noise_points,
    )


def run_hdbscan_sweep(
    df: pd.DataFrame,
    min_cluster_fracs: tuple[float, ...] = (0.005, 0.01, 0.02),
    min_cluster_size_min: int = 25,
    min_samples: int = 60,
    max_cluster_size: int | None = None,
) -> list[Partition]:
    return [
        fit_hdbscan_partition(
            df,
            min_cluster_frac=frac,
            min_cluster_size_min=min_cluster_size_min,
            min_samples=min_samples,
            max_cluster_size=max_cluster_size,
        )
        for frac in min_cluster_fracs
    ]
