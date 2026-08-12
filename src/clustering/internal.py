"""Honest, scale-aware density subdivision for diagnostic inspection.

This module is deliberately separate from :mod:`clustering.capped`.  A capped
partitioner must preserve coverage and may therefore attach HDBSCAN noise to a
nearby condensed group.  Diagnostic inspection answers a different question:
what structure did density actually condense inside the parent cluster?  Its
noise remains an explicit residue and is never reassigned.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN


NOT_SUBDIVIDED = "não subdividido nesta granularidade"
SUBDIVIDED = "subdividido"


@dataclass
class InternalSubdivision:
    """Condensed density groups plus the points HDBSCAN left as residue."""

    subclusters: list[list[int]]
    residue: list[int]
    min_cluster_size: int
    parent_n: int

    def __post_init__(self) -> None:
        self.subclusters = [sorted(int(point) for point in group) for group in self.subclusters]
        self.residue = sorted(int(point) for point in self.residue)
        flattened = [point for group in self.subclusters for point in group] + self.residue
        if len(flattened) != self.parent_n:
            raise ValueError("Internal subdivision does not account for every parent point")
        if len(set(flattened)) != len(flattened):
            raise ValueError("Internal subdivision assigns a parent point more than once")

    @property
    def condensed_n(self) -> int:
        return sum(len(group) for group in self.subclusters)

    @property
    def residue_n(self) -> int:
        return len(self.residue)

    @property
    def coverage_rate(self) -> float:
        return self.condensed_n / self.parent_n if self.parent_n else float("nan")

    @property
    def status(self) -> str:
        return SUBDIVIDED if len(self.subclusters) >= 2 else NOT_SUBDIVIDED


def subdivision_from_labels(
    points: list[int], labels: np.ndarray, *, min_cluster_size: int
) -> InternalSubdivision:
    """Translate HDBSCAN labels without assigning ``-1`` to another group."""
    parent_points = [int(point) for point in points]
    labels = np.asarray(labels, dtype=int)
    if len(parent_points) != len(labels):
        raise ValueError("points and labels must have the same length")

    subclusters = [
        [parent_points[index] for index in np.flatnonzero(labels == label)]
        for label in sorted(int(label) for label in np.unique(labels) if label >= 0)
    ]
    residue = [parent_points[index] for index in np.flatnonzero(labels < 0)]
    return InternalSubdivision(
        subclusters=subclusters,
        residue=residue,
        min_cluster_size=int(min_cluster_size),
        parent_n=len(parent_points),
    )


def diagnostic_density_subdivision(
    df: pd.DataFrame,
    points: list[int],
    min_cluster_size: int,
    min_samples: int = 60,
) -> InternalSubdivision:
    """Run one diagnostic HDBSCAN pass and preserve its residue verbatim.

    The canonical call uses the parent partition's ``min_cluster_size`` (g1);
    callers may use ``2 * min_cluster_size`` for the declared g2 sensitivity.
    No cap, recursion, nearest assignment, or alternate hard-coded scale occurs.
    """
    parent_points = [int(point) for point in points]
    mcs = max(2, int(min_cluster_size))
    if len(parent_points) < mcs:
        return InternalSubdivision([], parent_points, mcs, len(parent_points))

    coords = np.radians(df.iloc[parent_points][["lat", "lon"]].to_numpy(dtype=float))
    labels = HDBSCAN(
        min_cluster_size=mcs,
        min_samples=min(max(1, int(min_samples)), mcs),
        metric="haversine",
        cluster_selection_method="eom",
        copy=True,
    ).fit_predict(coords)
    return subdivision_from_labels(parent_points, labels, min_cluster_size=mcs)
