"""Command-line entry point for spatial fairness experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from data_loading import dataset_names
from experiments import ExperimentRunner


def _parse_fracs(value: str) -> tuple[float, ...]:
    try:
        return tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated floats, e.g. 0.005,0.01,0.02") from exc


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="Output directory.")
    parser.add_argument("--maps", action="store_true", help="Generate Folium HTML maps.")
    parser.add_argument("--no-maps", action="store_false", dest="maps", help="Skip Folium HTML maps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible runs.")
    parser.add_argument("--max-map-points", type=int, default=5000, help="Maximum sampled points per map.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logs.")
    parser.add_argument(
        "--hdbscan-fracs",
        type=_parse_fracs,
        default=(0.005, 0.01, 0.02),
        help="Comma-separated HDBSCAN min_cluster_size fractions.",
    )
    parser.set_defaults(maps=False)


def _runner(args: argparse.Namespace) -> ExperimentRunner:
    return ExperimentRunner(
        out_dir=args.out,
        maps=args.maps,
        seed=args.seed,
        hdbscan_fracs=args.hdbscan_fracs,
        max_map_points=args.max_map_points,
        verbose=not args.quiet,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Spatial fairness audit experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    unrestricted = subparsers.add_parser("unrestricted", help="Run KMeans unrestricted scan plus HDBSCAN.")
    _add_common_args(unrestricted)
    unrestricted.add_argument("--dataset", choices=dataset_names(), default="lar")
    unrestricted.add_argument("--kmeans-seeds", type=int, default=100)
    unrestricted.add_argument("--n-alt-worlds", type=int, default=200)
    unrestricted.add_argument("--signif-level", type=float, default=0.005)

    one = subparsers.add_parser("one-partitioning", help="Run fixed grid scan plus HDBSCAN.")
    _add_common_args(one)
    one.add_argument("--dataset", choices=dataset_names(), default="lar")
    one.add_argument("--n-alt-worlds", type=int, default=1000)
    one.add_argument("--signif-level", type=float, default=0.005)
    one.add_argument("--notebook-grid", action="store_true", help="Use the notebook's active 20x20 grid.")

    multiple = subparsers.add_parser("multiple-partitionings", help="Run random grid MeanVar experiment.")
    _add_common_args(multiple)
    multiple.add_argument("--dataset", choices=dataset_names(), default="semisynth")
    multiple.add_argument("--n-partitionings", type=int, default=100)

    all_cmd = subparsers.add_parser("all", help="Run the default reproduction suite.")
    _add_common_args(all_cmd)
    all_cmd.add_argument("--n-alt-unrestricted", type=int, default=200)
    all_cmd.add_argument("--n-alt-one", type=int, default=1000)
    all_cmd.add_argument("--n-partitionings", type=int, default=100)
    all_cmd.add_argument("--signif-level", type=float, default=0.005)
    all_cmd.add_argument("--notebook-grid", action="store_true", help="Use 20x20 grids for one-partitioning runs.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    runner = _runner(args)

    if args.command == "unrestricted":
        runner.run_unrestricted(
            dataset_name=args.dataset,
            n_alt_worlds=args.n_alt_worlds,
            signif_level=args.signif_level,
            kmeans_seeds=args.kmeans_seeds,
        )
    elif args.command == "one-partitioning":
        runner.run_one_partitioning(
            dataset_name=args.dataset,
            n_alt_worlds=args.n_alt_worlds,
            signif_level=args.signif_level,
            notebook_grid=args.notebook_grid,
        )
    elif args.command == "multiple-partitionings":
        runner.run_multiple_partitionings(
            dataset_name=args.dataset,
            n_partitionings=args.n_partitionings,
        )
    elif args.command == "all":
        runner.run_unrestricted(
            dataset_name="lar",
            n_alt_worlds=args.n_alt_unrestricted,
            signif_level=args.signif_level,
        )
        for dataset_name in ("lar", "crime"):
            runner.run_one_partitioning(
                dataset_name=dataset_name,
                n_alt_worlds=args.n_alt_one,
                signif_level=args.signif_level,
                notebook_grid=args.notebook_grid,
            )
        for dataset_name in ("semisynth", "synth_unfair"):
            runner.run_multiple_partitionings(
                dataset_name=dataset_name,
                n_partitionings=args.n_partitionings,
            )

    runner.write_outputs()
    print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
