"""Regenerate the stage-1 clustering map without re-running Monte Carlo.

Fits the clustering only (same selection rule as `explain`: sweep the fracs
and keep the best partition by max SUL, unless --min-cluster-frac is given)
and rewrites `maps/explain_{ds}_{method}_stage1_clusters.html`. CSVs and
thresholds on disk are left untouched.

Usage:
    uv run python src/stage1_map.py --dataset lar --out outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path

from clustering.registry import get_partitioner
from data_loading import dataset_names, load_dataset
from metrics.group_fairness import scan_regions
from visualization import save_clustering_stage_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite the stage-1 clustering map only.")
    parser.add_argument("--dataset", choices=dataset_names(), required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument("--min-cluster-frac", type=float, default=None)
    parser.add_argument("--hdbscan-min-samples", type=int, default=60)
    parser.add_argument("--max-map-points", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    fit = get_partitioner("hdbscan")
    fracs = (args.min_cluster_frac,) if args.min_cluster_frac is not None else (0.005, 0.01, 0.02)
    candidates = fit(dataset.df, fracs, min_samples=args.hdbscan_min_samples)

    def partition_max_sul(candidate) -> float:
        _, candidate_max, _ = scan_regions(
            candidate.regions, dataset.types, dataset.n_total, dataset.p_total
        )
        return candidate_max

    partition = max(candidates, key=partition_max_sul)
    print(f"Selected partition: {partition.params} ({len(partition.regions)} clusters)")

    output_path = args.out / "maps" / f"explain_{dataset.name}_{partition.method}_stage1_clusters.html"
    save_clustering_stage_map(
        dataset.df,
        dataset.types,
        partition,
        output_path,
        max_points=args.max_map_points,
        seed=args.seed,
    )
    print(f"Map written to {output_path}")


if __name__ == "__main__":
    main()
