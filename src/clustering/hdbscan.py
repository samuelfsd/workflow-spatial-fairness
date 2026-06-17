"""HDBSCAN-based organic spatial partitioning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN


@dataclass
class HDBSCANPartition:
    min_cluster_frac: float
    min_cluster_size: int
    labels: np.ndarray
    regions: list[dict]
    noise_points: list[int]

    @property
    def noise_n(self) -> int:
        return len(self.noise_points)


def fit_hdbscan_partition(
    df: pd.DataFrame,
    min_cluster_frac: float,
    min_cluster_size_min: int = 25,
) -> HDBSCANPartition:
    """Fit one HDBSCAN partition using haversine distance on lat/lon."""
    min_cluster_size = max(min_cluster_size_min, int(round(min_cluster_frac * len(df))))
    coords = np.radians(df[["lat", "lon"]].to_numpy(dtype=float))

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=None,
        metric="haversine",
        cluster_selection_method="eom",
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
            }
        )

    noise_points = np.flatnonzero(labels == -1).astype(int).tolist()
    return HDBSCANPartition(
        min_cluster_frac=float(min_cluster_frac),
        min_cluster_size=int(min_cluster_size),
        labels=labels,
        regions=regions,
        noise_points=noise_points,
    )


def run_hdbscan_sweep(
    df: pd.DataFrame,
    min_cluster_fracs: tuple[float, ...] = (0.005, 0.01, 0.02),
    min_cluster_size_min: int = 25,
) -> list[HDBSCANPartition]:
    return [
        fit_hdbscan_partition(
            df,
            min_cluster_frac=frac,
            min_cluster_size_min=min_cluster_size_min,
        )
        for frac in min_cluster_fracs
    ]
