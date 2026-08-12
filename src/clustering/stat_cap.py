"""Statistical-tail refinement for organic HDBSCAN partitions (ADR-0001)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from clustering.base import Partition
from clustering.capped import recursive_density_split
from clustering.hdbscan import fit_hdbscan_partition


def statistical_cap_directive(
    sizes: Sequence[int], sigma_multiplier: float = 1.0
) -> dict[int, float]:
    """Map tail-cluster indices to their leave-one-out mean split target."""
    if sigma_multiplier < 0:
        raise ValueError("sigma_multiplier must be non-negative")

    values = np.asarray(sizes, dtype=float)
    if values.ndim != 1:
        raise ValueError("sizes must be one-dimensional")
    if len(values) < 3:
        return {}

    directives: dict[int, float] = {}
    for idx, size in enumerate(values):
        others = np.delete(values, idx)
        mean = float(others.mean())
        sigma = float(others.std(ddof=1))
        if size > mean + sigma_multiplier * sigma:
            directives[idx] = mean
    return directives


def _sample_cv(sizes: Sequence[int]) -> float | None:
    values = np.asarray(sizes, dtype=float)
    if len(values) < 2 or float(values.mean()) == 0.0:
        return None
    return float(values.std(ddof=1) / values.mean())


def run_hdbscan_stat_cap_sweep(
    df: pd.DataFrame,
    min_cluster_fracs: tuple[float, ...] = (0.005, 0.01, 0.02),
    min_samples: int = 60,
    min_cluster_size_min: int = 25,
    stat_cap_sigma: float = 1.0,
) -> list[Partition]:
    """Refine only the statistical size tail of an organic HDBSCAN partition."""
    partitions: list[Partition] = []
    for frac in min_cluster_fracs:
        base = fit_hdbscan_partition(
            df,
            min_cluster_frac=frac,
            min_cluster_size_min=min_cluster_size_min,
            min_samples=min_samples,
        )
        sizes = [len(region["points"]) for region in base.regions]
        directives = statistical_cap_directive(sizes, stat_cap_sigma)
        min_piece_size = int(base.params["min_cluster_size"])
        targets = {
            idx: max(min_piece_size, int(round(mean)))
            for idx, mean in directives.items()
        }
        refined_regions: list[dict] = []

        for idx, region in enumerate(base.regions):
            organic = {
                **region,
                "type": "hdbscan_stat_cap",
                "origin": "organic",
                "origin_cluster_label": region["cluster_label"],
            }
            if idx not in directives:
                refined_regions.append(organic)
                continue

            target = targets[idx]
            pieces = recursive_density_split(
                df,
                region["points"],
                max_size=target,
                min_cluster_size_min=min_piece_size,
                min_samples=min_samples,
            )
            for piece in pieces:
                refused = len(piece) > target
                refined_regions.append(
                    {
                        **organic,
                        "points": list(piece),
                        "stat_cap_target": target,
                        "over_cap": refused,
                        "forced_uncapped": refused,
                    }
                )

        labels = np.full(len(df), -1, dtype=int)
        for label, region in enumerate(refined_regions):
            region["cluster_label"] = label
            labels[region["points"]] = label

        partitions.append(
            Partition(
                method="hdbscan_stat_cap",
                params={
                    **base.params,
                    "stat_cap": True,
                    "stat_cap_sigma": float(stat_cap_sigma),
                    "stat_cap_targets": len(directives),
                    "stat_cap_directives": {
                        int(base.regions[idx]["cluster_label"]): target
                        for idx, target in targets.items()
                    },
                    "stat_cap_refusals": sum(
                        1 for region in refined_regions if region.get("forced_uncapped")
                    ),
                    "cluster_size_cv_before_stat_cap": _sample_cv(sizes),
                    "cluster_size_cv_after_stat_cap": _sample_cv(
                        [len(region["points"]) for region in refined_regions]
                    ),
                },
                labels=labels,
                regions=refined_regions,
                noise_points=list(base.noise_points),
            )
        )
    return partitions


def run_hdbscan_stat_leaf_sweep(
    df: pd.DataFrame,
    min_cluster_fracs: tuple[float, ...] = (0.005, 0.01, 0.02),
    min_samples: int = 60,
    min_cluster_size_min: int = 25,
    stat_cap_sigma: float = 1.0,
) -> list[Partition]:
    """Refine statistical-tail parents using HDBSCAN's density leaves.

    The first pass remains the organic EOM partition. Only parents selected by
    the leave-one-out size rule are fitted again with ``leaf`` selection and the
    same absolute ``min_cluster_size`` ruler. A refinement is accepted only when
    it exposes at least two leaves. Points that HDBSCAN leaves between accepted
    leaves remain unassigned; they are deliberately not attached to a nearest
    child. If fewer than two leaves exist, the original parent is preserved.

    The statistical target is a trigger and reporting reference, not a geometric
    maximum-size guarantee. A density leaf can therefore remain above it.
    """
    partitions: list[Partition] = []
    for frac in min_cluster_fracs:
        base = fit_hdbscan_partition(
            df,
            min_cluster_frac=frac,
            min_cluster_size_min=min_cluster_size_min,
            min_samples=min_samples,
        )
        sizes = [len(region["points"]) for region in base.regions]
        directives = statistical_cap_directive(sizes, stat_cap_sigma)
        min_piece_size = int(base.params["min_cluster_size"])
        targets = {
            idx: max(min_piece_size, int(round(mean)))
            for idx, mean in directives.items()
        }
        refined_regions: list[dict] = []
        noise_points = set(int(point) for point in base.noise_points)
        split_parents = 0
        refusals = 0
        leaf_noise_n = 0

        for idx, region in enumerate(base.regions):
            parent_label = int(region["cluster_label"])
            organic = {
                **region,
                "type": "hdbscan_stat_leaf",
                "origin": "organic",
                "origin_cluster_label": parent_label,
            }
            if idx not in directives:
                refined_regions.append(organic)
                continue

            target = targets[idx]
            parent_points = [int(point) for point in region["points"]]
            subset = df.iloc[parent_points].reset_index(drop=True)
            leaves = fit_hdbscan_partition(
                subset,
                min_cluster_frac=frac,
                min_cluster_size_min=min_cluster_size_min,
                min_cluster_size_override=min_piece_size,
                min_samples=min_samples,
                cluster_selection_method="leaf",
            )
            if len(leaves.regions) < 2:
                refusals += 1
                refined_regions.append(
                    {
                        **organic,
                        "stat_cap_target": target,
                        "stat_leaf_refused": True,
                        "over_cap": len(parent_points) > target,
                    }
                )
                continue

            split_parents += 1
            parent_leaf_noise = [
                parent_points[local_idx] for local_idx in leaves.noise_points
            ]
            noise_points.update(parent_leaf_noise)
            leaf_noise_n += len(parent_leaf_noise)
            for leaf in leaves.regions:
                points = [parent_points[local_idx] for local_idx in leaf["points"]]
                refined_regions.append(
                    {
                        **organic,
                        "points": points,
                        "cluster_selection_method": "leaf",
                        "split_mode": "density_leaf",
                        "stat_cap_target": target,
                        "stat_leaf_refused": False,
                        "stat_leaf_parent_noise_n": len(parent_leaf_noise),
                        "over_cap": len(points) > target,
                    }
                )

        labels = np.full(len(df), -1, dtype=int)
        for label, region in enumerate(refined_regions):
            region["cluster_label"] = label
            labels[region["points"]] = label
        sorted_noise = sorted(noise_points)

        partitions.append(
            Partition(
                method="hdbscan_stat_leaf",
                params={
                    **base.params,
                    "stat_cap": True,
                    "stat_cap_sigma": float(stat_cap_sigma),
                    "stat_cap_targets": len(directives),
                    "stat_cap_directives": {
                        int(base.regions[idx]["cluster_label"]): target
                        for idx, target in targets.items()
                    },
                    "stat_cap_refusals": refusals,
                    "refinement_cluster_selection_method": "leaf",
                    "stat_leaf_split_parents": split_parents,
                    "stat_leaf_refusals": refusals,
                    "stat_leaf_noise_n": leaf_noise_n,
                    "cluster_size_cv_before_stat_cap": _sample_cv(sizes),
                    "cluster_size_cv_after_stat_cap": _sample_cv(
                        [len(region["points"]) for region in refined_regions]
                    ),
                },
                labels=labels,
                regions=refined_regions,
                noise_points=sorted_noise,
            )
        )
    return partitions
