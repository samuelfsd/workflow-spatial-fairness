"""Delaunay peer adjacency for the local z-score (peer baseline).

Peers of a cluster are its neighbours in the Delaunay triangulation of the
cluster centroids (parameter-free). Adjacency is planar over raw lat/lon: at
metropolitan/state scale the projection distortion is negligible for *topology*
(who borders whom), which is all this needs — not absolute distances.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, QhullError

from clustering.base import Partition


def _centroids(partition: Partition, df: pd.DataFrame) -> np.ndarray:
    lat = df["lat"].to_numpy(dtype=float)
    lon = df["lon"].to_numpy(dtype=float)
    return np.array(
        [[lat[region["points"]].mean(), lon[region["points"]].mean()] for region in partition.regions],
        dtype=float,
    )


def build_delaunay_adjacency(partition: Partition, df: pd.DataFrame) -> dict[int, list[int]]:
    """Map each cluster label to its Delaunay-neighbour labels.

    Clusters with fewer than 3 total, or degenerate (collinear) centroids, are
    handled explicitly: 2 clusters are mutual peers; 1 or 0 have none; a
    collinear set falls back to no peers (the local z-score then reports NaN).
    """
    labels = [int(region["cluster_label"]) for region in partition.regions]

    if len(labels) < 2:
        return {label: [] for label in labels}
    if len(labels) == 2:
        return {labels[0]: [labels[1]], labels[1]: [labels[0]]}

    try:
        triangulation = Delaunay(_centroids(partition, df))
    except QhullError:
        return {label: [] for label in labels}

    adjacency: dict[int, set[int]] = {label: set() for label in labels}
    for simplex in triangulation.simplices:
        for i in simplex:
            for j in simplex:
                if i != j:
                    adjacency[labels[i]].add(labels[j])

    return {label: sorted(neighbours) for label, neighbours in adjacency.items()}
