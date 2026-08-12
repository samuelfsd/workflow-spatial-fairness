"""Organic → rescue → statistical-cap partitioner (ADR-0006).

The implementation remains registered so the rejected rescue experiment is
regenerable. It is deliberately not the default partitioner: the synthetic-fair
gate showed that lowering density over unassigned points invents structure.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from clustering.base import Partition
from clustering.capped import recursive_density_split
from clustering.hdbscan import effective_min_cluster_size, fit_hdbscan_partition
from clustering.stat_cap import statistical_cap_directive


def _sample_cv(sizes: Sequence[int]) -> float | None:
    values = np.asarray(sizes, dtype=float)
    if len(values) < 2 or float(values.mean()) == 0.0:
        return None
    return float(values.std(ddof=1) / values.mean())


def run_hdbscan_rescue_sweep(
    df: pd.DataFrame,
    min_cluster_fracs: tuple[float, ...] = (0.005, 0.01, 0.02),
    rescue_min_samples: tuple[int, ...] = (60, 30, 15),
    stat_cap: bool = True,
    min_samples: int = 60,
    min_cluster_size_min: int = 25,
) -> list[Partition]:
    """Return one experimental partition per frac × rescue-density setting.

    The organic fit is reused across every rescue ``min_samples`` value for the
    same fraction. Each result keeps the full-data absolute cluster-size floor,
    maps subset positions back to canonical point IDs, then applies at most one
    statistical-cap invitation per origin-parent cluster.
    """
    partitions: list[Partition] = []
    for frac in min_cluster_fracs:
        min_cluster_size = effective_min_cluster_size(
            len(df), frac, min_cluster_size_min
        )
        base = fit_hdbscan_partition(
            df,
            min_cluster_frac=frac,
            min_cluster_size_min=min_cluster_size_min,
            min_samples=min_samples,
        )
        rescue_df = df.iloc[base.noise_points].reset_index(drop=True)
        for rescue_samples in rescue_min_samples:
            regions = [
                {
                    **region,
                    "origin": "organic",
                    "origin_cluster_label": region["cluster_label"],
                }
                for region in base.regions
            ]

            rescued = fit_hdbscan_partition(
                rescue_df,
                min_cluster_frac=frac,
                min_cluster_size_min=min_cluster_size_min,
                min_samples=rescue_samples,
                min_cluster_size_override=min_cluster_size,
            )
            for region in rescued.regions:
                regions.append(
                    {
                        **region,
                        "points": [base.noise_points[point] for point in region["points"]],
                        "type": "hdbscan_rescue",
                        "origin": "rescue",
                        "origin_cluster_label": region["cluster_label"],
                        "rescue_min_samples": int(rescue_samples),
                        "rescue_effective_min_samples": int(rescued.params["min_samples"]),
                        "min_cluster_frac": float(frac),
                    }
                )

            noise_points = [base.noise_points[point] for point in rescued.noise_points]
            cv_before_stat_cap = _sample_cv(
                [len(region["points"]) for region in regions]
            )
            directives = (
                statistical_cap_directive([len(region["points"]) for region in regions])
                if stat_cap
                else {}
            )
            capped_regions: list[dict] = []
            for idx, region in enumerate(regions):
                if idx not in directives:
                    capped_regions.append(region)
                    continue

                target = max(2, int(round(directives[idx])))
                pieces = recursive_density_split(
                    df,
                    region["points"],
                    max_size=target,
                    min_cluster_size_min=min_cluster_size_min,
                    min_samples=int(region.get("rescue_min_samples", min_samples)),
                )
                for piece in pieces:
                    refused = len(piece) > target
                    capped_regions.append(
                        {
                            **region,
                            "points": list(piece),
                            "stat_cap_target": target,
                            "over_cap": refused,
                            "forced_uncapped": refused,
                        }
                    )

            regions = capped_regions
            labels = np.full(len(df), -1, dtype=int)
            for label, region in enumerate(regions):
                region["cluster_label"] = label
                labels[region["points"]] = label
            partitions.append(
                Partition(
                    method="hdbscan_rescue",
                    params={
                        **base.params,
                        "rescue_min_samples": int(rescue_samples),
                        "rescue_effective_min_samples": int(rescued.params["min_samples"]),
                        "stat_cap": bool(stat_cap),
                        "stat_cap_targets": len(directives),
                        "stat_cap_refusals": sum(
                            1 for region in regions if region.get("forced_uncapped")
                        ),
                        "cluster_size_cv_before_stat_cap": cv_before_stat_cap,
                        "cluster_size_cv_after_stat_cap": _sample_cv(
                            [len(region["points"]) for region in regions]
                        ),
                    },
                    labels=labels,
                    regions=regions,
                    noise_points=noise_points,
                )
            )
    return partitions
