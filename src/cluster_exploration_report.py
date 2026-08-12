"""Standalone CLI: regenerate cluster exploration from a versioned run snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path

from exploration_report import generate_cluster_exploration
from data_loading import load_dataset
from metric_comparison import compare_primary_metrics, write_primary_comparison
from metrics.registry import primary_metric_names
from run_snapshot import MANIFEST_NAME, load_run_snapshot
import json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate one cluster-exploration reading without clustering or Monte Carlo."
    )
    parser.add_argument("run_dir", type=Path, help="Directory containing run_manifest.json.")
    parser.add_argument("--primary-metric", choices=primary_metric_names(), default="local_z")
    parser.add_argument("--profile", choices=("full", "core", "custom", "none"), default="full")
    parser.add_argument("--out", type=Path, default=None, help="Optional report directory.")
    parser.add_argument(
        "--details", default="auto",
        help="Detailed clusters: auto, all, or comma-separated cluster labels.",
    )
    parser.add_argument(
        "--compare-with", choices=primary_metric_names(), default=None,
        help="Also materialize an explicit same-snapshot primary comparison.",
    )
    parser.add_argument(
        "--families", default=None,
        help="For --profile custom: comma-separated details,multiscale,supplements.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    destination = generate_cluster_exploration(
        args.run_dir,
        primary_metric=args.primary_metric,
        profile=args.profile,
        output_dir=args.out,
        detail_selection=args.details,
        custom_families=(
            {value.strip() for value in args.families.split(",") if value.strip()}
            if args.families else None
        ),
        progress=print,
    )
    print(f"Relatório exploratório: {destination}")
    if args.compare_with and args.compare_with != args.primary_metric:
        manifest = json.loads((args.run_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
        dataset = load_dataset(manifest["dataset"]["name"])
        snapshot = load_run_snapshot(args.run_dir, dataset)
        comparison = compare_primary_metrics(
            dataset, snapshot, args.primary_metric, args.compare_with
        )
        comparison_dir = (
            args.run_dir / "exploration" / "comparisons"
            / f"{args.primary_metric}_vs_{args.compare_with}"
        )
        write_primary_comparison(
            comparison, comparison_dir,
            first=args.primary_metric, second=args.compare_with,
        )
        print(f"Comparação de métricas: {comparison_dir}")


if __name__ == "__main__":
    main()
