"""Registry of pluggable spatial partitioners.

A partitioner is a callable ``(df: pd.DataFrame, **params) -> list[Partition]``.
To add a new algorithm: implement it in its own module returning `Partition`
objects (see `clustering/base.py`), then add one entry to `PARTITIONERS` below.
Metrics, experiments, and visualization only depend on the `Partition` shape,
so no other code needs to change.
"""

from __future__ import annotations

from collections.abc import Callable

from clustering.base import Partition
from clustering.capped import run_capped_hdbscan_sweep
from clustering.hdbscan import run_hdbscan_sweep

PartitionerFn = Callable[..., list[Partition]]

PARTITIONERS: dict[str, PartitionerFn] = {
    "hdbscan": run_hdbscan_sweep,
    "capped_hdbscan": run_capped_hdbscan_sweep,
    # "dbscan": run_dbscan_sweep,  # future: src/clustering/dbscan.py
}


def partitioner_names() -> list[str]:
    return list(PARTITIONERS)


def get_partitioner(name: str) -> PartitionerFn:
    if name not in PARTITIONERS:
        raise ValueError(f"Unknown partitioner: {name}. Available: {partitioner_names()}")
    return PARTITIONERS[name]
