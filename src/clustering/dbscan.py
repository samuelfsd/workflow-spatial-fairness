"""DBSCAN-based spatial partitioning

This module implements spatial clustering using DBSCAN to create
adaptive spatial units for fairness analysis.

Planned functionality:
- Fit DBSCAN with spatial distance metrics
- Convert clusters to spatial partitions / polygons
- Evaluate partition quality (compactness, balance)

Contract: implement `run_dbscan_sweep(df, **params) -> list[Partition]`
(see `clustering/base.py`) and register it in `clustering/registry.py`
under `PARTITIONERS["dbscan"]`. Nothing in metrics, experiments, or
visualization needs to change.
"""

# TODO: Implement DBSCAN spatial partitioning
