"""Command-line entry point for spatial fairness experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from clustering.registry import partitioner_names
from data_loading import dataset_names, load_dataset
from experiments import ExperimentRunner
from metrics.registry import primary_metric_names


SACHARIDIS_DATASETS = ("lar", "crime", "semisynth", "synth_fair", "synth_unfair")


def _parse_fracs(value: str) -> tuple[float, ...]:
    try:
        return tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated floats, e.g. 0.005,0.01,0.02") from exc


def _parse_metrics(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_ints(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers, e.g. 60,30,15") from exc


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
    parser.add_argument(
        "--clustering",
        choices=partitioner_names(),
        default="hdbscan",
        help="Spatial partitioner used for the clustering comparison. "
        "hdbscan_stat_leaf redivides only statistical-tail parents using "
        "HDBSCAN leaves and leaves peripheral points unassigned.",
    )
    parser.add_argument(
        "--hdbscan-min-samples",
        type=int,
        default=60,
        help="HDBSCAN min_samples (density smoothing); capped at min_cluster_size. "
        "Keep small on large datasets to bound memory.",
    )
    parser.add_argument(
        "--max-cluster-size",
        type=int,
        default=None,
        help="Máx. de pontos por cluster (limite nativo EOM do HDBSCAN, orgânico). "
        "Mecanismo rejeitado como método, mantido para regenerar a evidência do ADR-0001. "
        "Com --clustering capped_hdbscan usa split recursivo por densidade em vez do EOM.",
    )
    parser.add_argument(
        "--rescue-min-samples",
        type=_parse_ints,
        default=(60, 30, 15),
        help="Comma-separated min_samples sweep for the hdbscan_rescue second pass.",
    )
    parser.add_argument(
        "--stat-cap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the statistical cap after organic + rescue clustering "
        "(--no-stat-cap isolates the rescue effect).",
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
        clustering_method=args.clustering,
        hdbscan_min_samples=args.hdbscan_min_samples,
        max_cluster_size=args.max_cluster_size,
        rescue_min_samples=args.rescue_min_samples,
        stat_cap=args.stat_cap,
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

    explain = subparsers.add_parser(
        "explain",
        help="Run the pipeline once and emit stage-by-stage explainability maps and tables.",
    )
    _add_common_args(explain)
    explain.add_argument("--dataset", choices=dataset_names(), default="crime")
    explain.add_argument(
        "--min-cluster-frac",
        type=float,
        default=None,
        help="Single HDBSCAN fraction. Default: sweep --hdbscan-fracs and keep "
        "the best partition by max SUL (same clustering the unrestricted map shows).",
    )
    explain.add_argument("--n-alt-worlds", type=int, default=200)
    explain.add_argument("--signif-level", type=float, default=0.005)
    explain.add_argument(
        "--metrics",
        type=_parse_metrics,
        default=None,
        help="Comma-separated metrics to score "
        "(default: local_z,sul,gini,gini_subcluster,dp_difference; dp_ratio also available).",
    )
    explain.add_argument(
        "--primary-metric",
        choices=primary_metric_names(),
        default="local_z",
        help="Metric that drives detection colors/significance (default: local_z).",
    )
    explain.add_argument(
        "--exploration-profile",
        choices=("full", "core", "none"),
        default="full",
        help="Exploration output: full (default), core global report, or snapshot only.",
    )

    all_cmd = subparsers.add_parser("all", help="Run the default reproduction suite.")
    _add_common_args(all_cmd)
    all_cmd.add_argument("--n-alt-unrestricted", type=int, default=200)
    all_cmd.add_argument("--n-alt-one", type=int, default=1000)
    all_cmd.add_argument("--n-partitionings", type=int, default=100)
    all_cmd.add_argument("--signif-level", type=float, default=0.005)
    all_cmd.add_argument("--notebook-grid", action="store_true", help="Use 20x20 grids for one-partitioning runs.")

    benchmark = subparsers.add_parser(
        "benchmark-sacharidis",
        help="Reproduce and compare the quantitative Sacharidis benchmark with resumable checkpoints.",
    )
    benchmark.add_argument("--out", type=Path, default=Path("outputs/benchmark_sacharidis"))
    benchmark.add_argument("--dataset", choices=["all", *SACHARIDIS_DATASETS], default="all")
    benchmark.add_argument("--phase", choices=("reproduce", "compare", "report", "all"), default="all")
    benchmark.add_argument("--seed", type=int, default=42)
    benchmark.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    benchmark.add_argument("--maps", action=argparse.BooleanOptionalAction, default=False)
    benchmark.add_argument("--scan-worlds", type=int, default=200)
    benchmark.add_argument("--grid-worlds", type=int, default=1000)
    benchmark.add_argument("--standardized-worlds", type=int, default=1000)
    benchmark.add_argument("--random-partitionings", type=int, default=100)
    benchmark.add_argument("--kmeans-seeds", type=int, default=100)
    benchmark.add_argument("--signif-level", type=float, default=0.005)

    repeated = subparsers.add_parser(
        "benchmark-repeated",
        help="Run a small trial, an explicitly confirmed official battery, or a checkpoint-only report.",
    )
    repeated.add_argument("--plan", type=Path, default=Path("benchmarks/repeated/plan.draft.json"))
    repeated.add_argument("--out", type=Path, default=Path("outputs/benchmark_repeated"))
    repeated.add_argument("--phase", choices=("trial", "run", "report"), required=True)
    repeated.add_argument("--confirm-official", action="store_true")
    repeated.add_argument(
        "--coordinate-source-dataset", choices=dataset_names(), default=None,
        help="Dataset used only as a source of coordinates for the realistic irregular geography; outcomes are ignored.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "benchmark-sacharidis":
        from benchmark_initial import DEFAULT_DATASETS, InitialBenchmarkConfig, run_initial_benchmark
        from benchmark_sacharidis import SacharidisProtocol

        datasets = DEFAULT_DATASETS if args.dataset == "all" else (args.dataset,)
        protocol = SacharidisProtocol(
            reproduction_scan_worlds=args.scan_worlds,
            grid_worlds=args.grid_worlds,
            standardized_worlds=args.standardized_worlds,
            random_partitionings=args.random_partitionings,
            kmeans_seeds=args.kmeans_seeds,
            signif_level=args.signif_level,
        )
        destination = run_initial_benchmark(InitialBenchmarkConfig(
            output_root=args.out, datasets=datasets, phase=args.phase, seed=args.seed,
            resume=args.resume, maps=args.maps, protocol=protocol,
        ))
        print(f"Benchmark written to {destination}")
        return
    if args.command == "benchmark-repeated":
        from repeated_workflow import load_repeated_plan, run_repeated_workflow

        coordinate_source = None
        plan = load_repeated_plan(args.plan)
        source_name = args.coordinate_source_dataset or plan.coordinate_source_dataset
        if source_name != plan.coordinate_source_dataset:
            parser.error(
                "--coordinate-source-dataset deve coincidir com "
                f"coordinate_source_dataset={plan.coordinate_source_dataset!r} do plano"
            )
        if args.phase != "report":
            source_dataset = load_dataset(source_name)
            coordinate_source = source_dataset.df[["lat", "lon"]].copy()
            coordinate_source.attrs["dataset_name"] = source_dataset.name
            coordinate_source.attrs["source_name"] = (
                f"dataset:{source_dataset.name}@sha256:{source_dataset.source_sha256}"
            )
        destination = run_repeated_workflow(
            args.plan, args.out, phase=args.phase,
            confirm_official=args.confirm_official,
            coordinate_source=coordinate_source,
        )
        print(f"Repeated benchmark written to {destination}")
        return
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
    elif args.command == "explain":
        runner.run_explain(
            dataset_name=args.dataset,
            min_cluster_frac=args.min_cluster_frac,
            n_alt_worlds=args.n_alt_worlds,
            signif_level=args.signif_level,
            metrics=args.metrics,
            primary_metric=args.primary_metric,
            exploration_profile=args.exploration_profile,
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
