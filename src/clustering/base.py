"""Shared partition result type for all clustering backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Partition:
    """Result of one spatial partitioning run, independent of the algorithm.

    `regions` follows the repo-wide region abstraction: plain dicts with a
    `points` key of positional indices into the loaded DataFrame, plus any
    method-specific keys. Metrics and visualization only rely on `points`.
    """

    method: str
    params: dict[str, Any]
    labels: np.ndarray
    regions: list[dict]
    noise_points: list[int] = field(default_factory=list)

    @property
    def noise_n(self) -> int:
        return len(self.noise_points)
