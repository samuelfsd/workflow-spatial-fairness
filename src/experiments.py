"""Experiment orchestration for the original spatial fairness audit plus HDBSCAN."""

from __future__ import annotations

import json
import math
import time
from contextlib import contextmanager
from pathlib import Path
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd

from clustering.base import Partition
from clustering.capped import density_subclusters
from clustering.registry import get_partitioner
from data_loading import LoadedDataset, load_dataset
from descriptives import (
    cluster_card_data,
    cluster_frame,
    dataset_balance,
    dispersion_summary,
    expected_sigma_ratio,
    organic_local_z_deltas,
    partition_profile,
)
from figures import (
    balance_figure,
    cluster_card_figure,
    close as close_figures,
    metric_panels_figure,
    save_figure,
    save_pdf_report,
)
from lens import GREATER_LA_BBOX, clusters_in_bbox
from metrics.base import MetricContext
from metrics.group_fairness import (
    calculate_gini,
    calculate_meanvar,
    classify_direction,
    get_signif_threshold,
    get_simple_stats,
    scan_partitioning,
    scan_regions,
    select_significant_regions,
)
from metrics.neighbors import build_delaunay_adjacency
from metrics.registry import get_metric, metric_names
from metrics.significance import (
    analytic_threshold,
    significance_threshold,
    simulate_null_metric,
)
from regions import (
    create_grid_from_dataset,
    create_random_partitionings,
    create_regions,
    create_rtree,
    create_seeds,
    filter_non_overlapping_regions,
)
from visualization import (
    save_clustering_stage_map,
    save_detection_stage_map,
    save_experiment_map,
)

_EXPLAIN_DEFAULT_METRICS = ("local_z", "sul", "gini", "gini_subcluster", "dp_difference")

# Subcluster granularities for the cluster cards: a pocket that only shows up at
# the finer split is an artifact of the parameter, not a finding.
_CARD_GRANULARITIES = (25, 100)

#: How many clusters get a card, ranked by internal inequality.
_CARD_TOP_N = 3

# Interval/reading caption shown on each metric panel so the range is visible.
_METRIC_INFO = {
    "local_z": "z de vizinhança · intervalo (−∞, +∞) · sinal = direção · |z| ≥ limiar ⇒ significativo",
    "sul": "SUL (baseline) · intervalo [0, +∞) · sempre ≥ 0 · ≥ limiar ⇒ significativo",
    "gini": "contribuição-Gini · assinada (~ ±0,1) · > 0 puxa a desigualdade do mapa p/ cima",
    "gini_subcluster": "Gini entre subclusters · intervalo [0, 1] · 0 = homogêneo por dentro",
    "meanvar": "MeanVar · desvio² da taxa · intervalo [0, +∞)",
    "dp_difference": "taxa de seleção do cluster · escalar da partição = max − min (paridade estatística)",
    "dp_ratio": "taxa de seleção do cluster · escalar da partição = min / max (paridade estatística)",
}


def _json_params(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True)


def _safe_float(value: Any) -> float | None:
    """JSON-friendly float: None/NaN become None so the summary stays valid JSON."""
    if value is None:
        return None
    number = float(value)
    return None if math.isnan(number) else number


def _region_stats(region: dict | None, types: np.ndarray) -> dict[str, Any]:
    if region is None:
        return {
            "best_region_n": 0,
            "best_region_p": 0,
            "best_region_rate": np.nan,
        }

    n, p, rho = get_simple_stats(region["points"], types)
    return {
        "best_region_n": n,
        "best_region_p": p,
        "best_region_rate": rho,
    }


def _meanvar_summary(regions: list[dict], types: np.ndarray) -> tuple[dict | None, float, float, np.ndarray]:
    best_region, max_score, scores, rhos = scan_partitioning(regions, types)
    mean_score = float(np.nanmean(scores)) if len(scores) and not np.all(np.isnan(scores)) else 0.0
    return best_region, max_score, mean_score, rhos


class ExperimentRunner:
    def __init__(
        self,
        out_dir: Path,
        maps: bool = False,
        seed: int = 42,
        hdbscan_fracs: tuple[float, ...] = (0.005, 0.01, 0.02),
        max_map_points: int = 5000,
        verbose: bool = True,
        clustering_method: str = "hdbscan",
        hdbscan_min_samples: int = 60,
        max_cluster_size: int | None = None,
        rescue_min_samples: tuple[int, ...] = (60, 30, 15),
        stat_cap: bool = True,
    ) -> None:
        self.out_dir = out_dir
        self.maps = maps
        self.seed = seed
        self.hdbscan_fracs = hdbscan_fracs
        self.hdbscan_min_samples = hdbscan_min_samples
        self.max_cluster_size = max_cluster_size
        self.rescue_min_samples = rescue_min_samples
        self.stat_cap = stat_cap
        self.max_map_points = max_map_points
        self.verbose = verbose
        self.clustering_method = clustering_method
        self.started_at = time.perf_counter()
        self.dataset_cache: dict[str, LoadedDataset] = {}
        self.partition_cache: dict[tuple[str, str], list[Partition]] = {}

        self.unrestricted_rows: list[dict[str, Any]] = []
        self.one_partitioning_rows: list[dict[str, Any]] = []
        self.multiple_partitioning_rows: list[dict[str, Any]] = []
        self.clustering_rows: list[dict[str, Any]] = []

        self.out_dir.mkdir(parents=True, exist_ok=True)
        if self.maps:
            (self.out_dir / "maps").mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        if self.verbose:
            elapsed = time.perf_counter() - self.started_at
            print(f"[{elapsed:8.1f}s] {message}", flush=True)

    @contextmanager
    def timed_step(self, label: str) -> Iterator[None]:
        self.log(f"{label}...")
        started = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - started
            self.log(f"{label} done in {duration:.1f}s")

    def load(self, dataset_name: str) -> LoadedDataset:
        if dataset_name not in self.dataset_cache:
            self.dataset_cache[dataset_name] = load_dataset(dataset_name)
        return self.dataset_cache[dataset_name]

    def partitions(self, dataset: LoadedDataset, method: str | None = None) -> list[Partition]:
        method = method or self.clustering_method
        fit = get_partitioner(method)
        key = (dataset.name, method)
        if key not in self.partition_cache:
            partitions = []
            if method in (
                "hdbscan",
                "capped_hdbscan",
                "hdbscan_rescue",
                "hdbscan_stat_cap",
                "hdbscan_stat_leaf",
            ):
                for idx, frac in enumerate(self.hdbscan_fracs, start=1):
                    extra: dict[str, Any] = {"min_samples": self.hdbscan_min_samples}
                    if method == "capped_hdbscan":
                        extra["max_cluster_size"] = self.max_cluster_size or 2000
                    elif method == "hdbscan_rescue":
                        extra["rescue_min_samples"] = self.rescue_min_samples
                        extra["stat_cap"] = self.stat_cap
                    elif self.max_cluster_size:  # native HDBSCAN EOM cap
                        extra["max_cluster_size"] = self.max_cluster_size
                    with self.timed_step(f"Clustering ({method}) {idx}/{len(self.hdbscan_fracs)} frac={frac}"):
                        partitions.extend(fit(dataset.df, (frac,), **extra))
            else:
                with self.timed_step(f"Clustering ({method})"):
                    partitions.extend(fit(dataset.df))
            self.partition_cache[key] = partitions
        return self.partition_cache[key]

    def common_row(
        self,
        *,
        experiment: str,
        dataset: LoadedDataset,
        method: str,
        params: dict[str, Any],
        n_regions: int,
        max_sul: float | None = None,
        signif_threshold: float | None = None,
        significant_regions: int | None = None,
        best_region: dict | None = None,
        meanvar: float | None = None,
        meanvar_max_score: float | None = None,
        gini: float | None = None,
        non_overlapping_regions: int | None = None,
        noise_n: int | None = None,
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "experiment": experiment,
            "dataset": dataset.name,
            "method": method,
            "params": _json_params(params),
            "N": dataset.n_total,
            "P": dataset.p_total,
            "global_rate": dataset.global_rate,
            "n_regions": n_regions,
            "max_sul": max_sul,
            "signif_threshold": signif_threshold,
            "significant_regions": significant_regions,
            "non_overlapping_regions": non_overlapping_regions,
            "meanvar": meanvar,
            "meanvar_max_score": meanvar_max_score,
            "gini": gini,
            "noise_n": noise_n,
            "noise_rate": (noise_n / dataset.n_total) if noise_n is not None and dataset.n_total else None,
        }
        row.update(_region_stats(best_region, dataset.types))
        return row

    def evaluate_partitions(
        self,
        *,
        experiment: str,
        dataset: LoadedDataset,
        n_alt_worlds: int,
        signif_level: float,
        method: str | None = None,
    ) -> list[tuple[dict[str, Any], Partition, list[dict]]]:
        results = []
        for idx, partition in enumerate(self.partitions(dataset, method=method)):
            best_region, max_sul, statistics = scan_regions(
                partition.regions,
                dataset.types,
                dataset.n_total,
                dataset.p_total,
            )

            if n_alt_worlds > 0 and partition.regions:
                threshold = get_signif_threshold(
                    signif_level,
                    n_alt_worlds,
                    partition.regions,
                    dataset.n_total,
                    dataset.p_total,
                    seed=self.seed + int(partition.params.get("min_cluster_size", idx)),
                )
                significant = select_significant_regions(partition.regions, statistics, threshold)
            else:
                threshold = None
                significant = []

            _, meanvar_max, _, rhos = _meanvar_summary(partition.regions, dataset.types)
            meanvar = calculate_meanvar(rhos)

            row = self.common_row(
                experiment=experiment,
                dataset=dataset,
                method=partition.method,
                params=partition.params,
                n_regions=len(partition.regions),
                max_sul=max_sul,
                signif_threshold=threshold,
                significant_regions=len(significant),
                best_region=best_region,
                meanvar=meanvar,
                meanvar_max_score=meanvar_max,
                gini=calculate_gini(rhos),
                noise_n=partition.noise_n,
            )
            self.clustering_rows.append(row)
            results.append((row, partition, significant))

        return results

    def run_unrestricted(
        self,
        dataset_name: str = "lar",
        n_alt_worlds: int = 200,
        signif_level: float = 0.005,
        kmeans_seeds: int = 100,
    ) -> None:
        self.log(f"Starting unrestricted experiment for dataset={dataset_name}")
        with self.timed_step("[1/6] Loading dataset"):
            dataset = self.load(dataset_name)

        with self.timed_step("[2/6] Creating R-tree"):
            rtree = create_rtree(dataset.df)

        with self.timed_step("[3/6] Running KMeans scan"):
            seeds = create_seeds(dataset.df, rtree, kmeans_seeds, random_state=self.seed)
            regions = create_regions(dataset.df, rtree, seeds, dataset.radii)
            best_region, max_sul, statistics = scan_regions(regions, dataset.types, dataset.n_total, dataset.p_total)

        with self.timed_step(f"[4/6] Monte Carlo for KMeans scan ({n_alt_worlds} worlds)"):
            threshold = get_signif_threshold(
                signif_level,
                n_alt_worlds,
                regions,
                dataset.n_total,
                dataset.p_total,
                seed=self.seed,
            )
            significant = select_significant_regions(regions, statistics, threshold)
            non_overlapping = filter_non_overlapping_regions(significant, dataset.df)

        self.unrestricted_rows.append(
            self.common_row(
                experiment="unrestricted",
                dataset=dataset,
                method="kmeans_scan",
                params={"n_seeds": kmeans_seeds, "radii": dataset.radii.tolist()},
                n_regions=len(regions),
                max_sul=max_sul,
                signif_threshold=threshold,
                significant_regions=len(significant),
                non_overlapping_regions=len(non_overlapping),
                best_region=best_region,
            )
        )

        with self.timed_step("[5/6] Running clustering + SUL comparison"):
            hdbscan_results = self.evaluate_partitions(
                experiment="unrestricted",
                dataset=dataset,
                n_alt_worlds=n_alt_worlds,
                signif_level=signif_level,
            )

        if self.maps:
            with self.timed_step("[6/6] Writing map"):
                best_hdbscan = max(hdbscan_results, key=lambda item: item[0]["max_sul"] or 0.0, default=None)
                save_experiment_map(
                    dataset.df,
                    dataset.types,
                    self.out_dir / "maps" / f"unrestricted_{dataset.name}.html",
                    box_regions=non_overlapping[:28],
                    hdbscan_regions=best_hdbscan[2] if best_hdbscan else [],
                    max_points=self.max_map_points,
                    seed=self.seed,
                )
        else:
            self.log("[6/6] Writing map skipped (--no-maps)")

    def run_one_partitioning(
        self,
        dataset_name: str,
        n_alt_worlds: int = 1000,
        signif_level: float = 0.005,
        notebook_grid: bool = False,
    ) -> None:
        self.log(f"Starting one-partitioning experiment for dataset={dataset_name}")
        with self.timed_step("[1/5] Loading dataset"):
            dataset = self.load(dataset_name)

        with self.timed_step("[2/5] Creating R-tree"):
            rtree = create_rtree(dataset.df)

        grids = ((20, 20),) if notebook_grid else dataset.fixed_grids

        for lon_n, lat_n in grids:
            with self.timed_step(f"[3/5] Grid {lon_n}x{lat_n} SUL + MeanVar"):
                _, _, regions = create_grid_from_dataset(dataset.df, rtree, lon_n=lon_n, lat_n=lat_n)
                best_region, max_sul, statistics = scan_regions(
                    regions,
                    dataset.types,
                    dataset.n_total,
                    dataset.p_total,
                )
                meanvar_region, meanvar_max, _, rhos = _meanvar_summary(regions, dataset.types)

            with self.timed_step(f"[4/5] Monte Carlo for grid {lon_n}x{lat_n} ({n_alt_worlds} worlds)"):
                threshold = get_signif_threshold(
                    signif_level,
                    n_alt_worlds,
                    regions,
                    dataset.n_total,
                    dataset.p_total,
                    seed=self.seed + lon_n + lat_n,
                )
                significant = select_significant_regions(regions, statistics, threshold)

            self.one_partitioning_rows.append(
                self.common_row(
                    experiment="one_partitioning",
                    dataset=dataset,
                    method="grid",
                    params={"lon_n": lon_n, "lat_n": lat_n},
                    n_regions=len(regions),
                    max_sul=max_sul,
                    signif_threshold=threshold,
                    significant_regions=len(significant),
                    best_region=best_region,
                    meanvar=calculate_meanvar(rhos),
                    meanvar_max_score=meanvar_max,
                    gini=calculate_gini(rhos),
                )
                | {
                    "meanvar_best_n": _region_stats(meanvar_region, dataset.types)["best_region_n"],
                    "meanvar_best_p": _region_stats(meanvar_region, dataset.types)["best_region_p"],
                    "meanvar_best_rate": _region_stats(meanvar_region, dataset.types)["best_region_rate"],
                }
            )

            if self.maps:
                save_experiment_map(
                    dataset.df,
                    dataset.types,
                    self.out_dir / "maps" / f"one_partitioning_{dataset.name}_{lon_n}x{lat_n}.html",
                    grid_regions=significant,
                    max_points=self.max_map_points,
                    seed=self.seed,
                )

        with self.timed_step("[5/5] Running clustering + SUL comparison"):
            hdbscan_results = self.evaluate_partitions(
                experiment="one_partitioning",
                dataset=dataset,
                n_alt_worlds=n_alt_worlds,
                signif_level=signif_level,
            )
        if self.maps:
            best_hdbscan = max(hdbscan_results, key=lambda item: item[0]["max_sul"] or 0.0, default=None)
            if best_hdbscan is not None:
                save_experiment_map(
                    dataset.df,
                    dataset.types,
                    self.out_dir / "maps" / f"one_partitioning_{dataset.name}_hdbscan.html",
                    hdbscan_regions=best_hdbscan[2],
                    max_points=self.max_map_points,
                    seed=self.seed,
                )

    def run_multiple_partitionings(
        self,
        dataset_name: str,
        n_partitionings: int = 100,
    ) -> None:
        self.log(f"Starting multiple-partitionings experiment for dataset={dataset_name}")
        with self.timed_step("[1/4] Loading dataset"):
            dataset = self.load(dataset_name)

        with self.timed_step("[2/4] Creating R-tree"):
            rtree = create_rtree(dataset.df)

        with self.timed_step(f"[3/4] Generating and scanning {n_partitionings} random grids"):
            partitionings = create_random_partitionings(
                dataset.df,
                rtree,
                n_partitionings=n_partitionings,
                seed=self.seed,
            )

            mean_scores = []
            max_scores = []
            gini_scores = []
            for idx, (grid_info, _, regions) in enumerate(partitionings):
                best_region, max_score, mean_score, rhos = _meanvar_summary(regions, dataset.types)
                gini_score = calculate_gini(rhos)
                mean_scores.append(mean_score)
                max_scores.append(max_score)
                gini_scores.append(gini_score)
                self.multiple_partitioning_rows.append(
                    self.common_row(
                        experiment="multiple_partitionings",
                        dataset=dataset,
                        method="random_grid",
                        params={
                            "partitioning_idx": idx,
                            "lon_n": grid_info["lon_n"],
                            "lat_n": grid_info["lat_n"],
                        },
                        n_regions=len(regions),
                        best_region=best_region,
                        meanvar=calculate_meanvar(rhos),
                        meanvar_max_score=max_score,
                        gini=gini_score,
                    )
                )

            self.multiple_partitioning_rows.append(
                {
                    "experiment": "multiple_partitionings",
                    "dataset": dataset.name,
                    "method": "random_grid_summary",
                    "params": _json_params({"n_partitionings": n_partitionings, "lat_n_range": [10, 40], "lon_n_range": [10, 40]}),
                    "N": dataset.n_total,
                    "P": dataset.p_total,
                    "global_rate": dataset.global_rate,
                    "n_regions": None,
                    "max_sul": None,
                    "signif_threshold": None,
                    "significant_regions": None,
                    "non_overlapping_regions": None,
                    "meanvar": float(np.mean(mean_scores)) if mean_scores else 0.0,
                    "meanvar_max_score": float(np.max(max_scores)) if max_scores else 0.0,
                    "gini": float(np.mean(gini_scores)) if gini_scores else 0.0,
                    "noise_n": None,
                    "noise_rate": None,
                    "best_region_n": None,
                    "best_region_p": None,
                    "best_region_rate": None,
                }
            )

        with self.timed_step("[4/4] Running clustering reference"):
            self.evaluate_partitions(
                experiment="multiple_partitionings",
                dataset=dataset,
                n_alt_worlds=0,
                signif_level=0.005,
            )

    def run_explain(
        self,
        dataset_name: str = "crime",
        min_cluster_frac: float | None = None,
        n_alt_worlds: int = 200,
        signif_level: float = 0.005,
        metrics: tuple[str, ...] | None = None,
        primary_metric: str = "local_z",
    ) -> None:
        """Run the pipeline once and emit stage-by-stage explainability outputs.

        Maps are always written, regardless of the --maps flag. When
        `min_cluster_frac` is None, sweeps the configured `hdbscan_fracs` and
        keeps the best partition by max SUL — the same selection rule the
        unrestricted comparison map uses, so both maps show the same clustering.
        """
        method = self.clustering_method
        run_started = time.perf_counter()
        self.log(f"Starting explain run for dataset={dataset_name} method={method}")
        with self.timed_step("[1/5] Loading dataset"):
            dataset = self.load(dataset_name)

        with self.timed_step(f"[2/5] Clustering ({method})"):
            if min_cluster_frac is not None and method in (
                "hdbscan",
                "capped_hdbscan",
                "hdbscan_rescue",
                "hdbscan_stat_cap",
                "hdbscan_stat_leaf",
            ):
                fit = get_partitioner(method)
                extra: dict[str, Any] = {"min_samples": self.hdbscan_min_samples}
                if method == "capped_hdbscan":
                    extra["max_cluster_size"] = self.max_cluster_size or 2000
                elif method == "hdbscan_rescue":
                    extra["rescue_min_samples"] = self.rescue_min_samples
                    extra["stat_cap"] = self.stat_cap
                elif self.max_cluster_size:
                    extra["max_cluster_size"] = self.max_cluster_size
                candidates = fit(dataset.df, (min_cluster_frac,), **extra)
            else:
                candidates = self.partitions(dataset)

            def partition_max_sul(candidate: Partition) -> float:
                _, candidate_max, _ = scan_regions(
                    candidate.regions, dataset.types, dataset.n_total, dataset.p_total
                )
                return candidate_max

            partition = max(candidates, key=partition_max_sul)
            if len(candidates) > 1:
                self.log(f"Selected best partition by max SUL: params={partition.params}")

        maps_dir = self.out_dir / "maps"
        with self.timed_step("[3/5] Writing stage-1 clustering map"):
            save_clustering_stage_map(
                dataset.df,
                dataset.types,
                partition,
                maps_dir / f"explain_{dataset.name}_{method}_stage1_clusters.html",
                max_points=self.max_map_points,
                seed=self.seed,
            )

        adjacency = build_delaunay_adjacency(partition, dataset.df)
        ctx = MetricContext(
            n_total=dataset.n_total,
            p_total=dataset.p_total,
            adjacency=adjacency,
            rng=np.random.default_rng(self.seed),
            split_subclusters=self._subcluster_splitter(dataset.df),
        )
        metric_list = list(metrics) if metrics else list(_EXPLAIN_DEFAULT_METRICS)
        if primary_metric not in metric_list and primary_metric in metric_names():
            metric_list.insert(0, primary_metric)
        metric_list = [name for name in metric_list if name in metric_names()]
        if not metric_list:
            raise ValueError(
                f"No known metric to score. Got metrics={metrics}, "
                f"primary_metric={primary_metric!r}; available: {metric_names()}"
            )

        with self.timed_step(f"[4/5] Scoring metrics ({', '.join(metric_list)})"):
            results = {name: get_metric(name)(partition, dataset.types, ctx) for name in metric_list}
            best_region, max_sul, _ = scan_regions(
                partition.regions, dataset.types, dataset.n_total, dataset.p_total
            )
            _, meanvar_max, _, rhos = _meanvar_summary(partition.regions, dataset.types)
            meanvar = calculate_meanvar(rhos)
            gini = calculate_gini(rhos)
            organic_z_deltas = (
                organic_local_z_deltas(
                    partition,
                    dataset.df,
                    dataset.types,
                    n_total=dataset.n_total,
                    p_total=dataset.p_total,
                ).to_dict("records")
                if method == "hdbscan_rescue"
                else []
            )
            delta_by_parent = {
                int(row["origin_cluster_label"]): float(row["local_z_delta"])
                for row in organic_z_deltas
            }

        with self.timed_step(f"[5/5] Monte Carlo ({n_alt_worlds} worlds) + detection outputs"):
            thresholds: dict[str, float | None] = {}
            analytic_thresholds: dict[str, float | None] = {}
            null_by_metric: dict[str, np.ndarray] = {}
            for offset, name in enumerate(metric_list):
                result = results[name]
                # The analytic band only means anything in standard-error units,
                # and only over the clusters the metric actually evaluated.
                if result.standardized:
                    evaluated = int(np.count_nonzero(~np.isnan(result.per_cluster)))
                    analytic_thresholds[name] = analytic_threshold(signif_level, evaluated)
                else:
                    analytic_thresholds[name] = None
                if result.supports_mc and n_alt_worlds > 0 and partition.regions:
                    null = simulate_null_metric(
                        get_metric(name),
                        partition,
                        ctx,
                        n_alt_worlds,
                        dataset.n_total,
                        dataset.p_total,
                        seed=self.seed + int(partition.params.get("min_cluster_size", 0)) + offset,
                    )
                    null_by_metric[name] = null
                    thresholds[name] = significance_threshold(signif_level, null)
                else:
                    thresholds[name] = None

            primary = results.get(primary_metric)
            primary_threshold = thresholds.get(primary_metric)

            region_results = []
            for idx, region in enumerate(partition.regions):
                n, p, rho = get_simple_stats(region["points"], dataset.types)
                rho_out = (
                    (dataset.p_total - p) / (dataset.n_total - n)
                    if dataset.n_total > n
                    else float("nan")
                )
                score = float(primary.per_cluster[idx]) if primary is not None else float("nan")
                significant = (
                    primary_threshold is not None
                    and not np.isnan(score)
                    and abs(score) >= primary_threshold
                )
                if primary is not None and primary.signed:
                    direction = "negative" if score < 0 else "positive" if score > 0 else "neutral"
                else:
                    direction = classify_direction(n, p, dataset.n_total, dataset.p_total)

                row = {
                    "region": region,
                    "n": n,
                    "p": p,
                    "n_neg": n - p,
                    "rho": rho,
                    "rho_out": rho_out,
                    "score": score,
                    "score_name": primary_metric,
                    "significant": significant,
                    "direction": direction,
                    "local_z_delta_after_rescue": (
                        delta_by_parent.get(int(region.get("origin_cluster_label", -1)))
                        if region.get("origin", "organic") == "organic"
                        else None
                    ),
                }
                for name in metric_list:
                    row[f"metric_{name}"] = float(results[name].per_cluster[idx])
                region_results.append(row)

            def _sort_key(item: dict) -> float:
                value = item.get(f"metric_{primary_metric}", item["score"])
                return -1.0 if np.isnan(value) else abs(value)

            region_results.sort(key=_sort_key, reverse=True)

            save_detection_stage_map(
                dataset.df,
                dataset.types,
                region_results,
                maps_dir / f"explain_{dataset.name}_{method}_stage4_detection.html",
                threshold=primary_threshold or 0.0,
                global_rate=dataset.global_rate,
                max_points=self.max_map_points,
                seed=self.seed,
            )

            labels = [item["region"].get("cluster_label", "?") for item in region_results]
            panels = [
                {
                    "name": name,
                    "labels": labels,
                    "values": [item[f"metric_{name}"] for item in region_results],
                    "directions": [item["direction"] for item in region_results],
                    "significant": [item["significant"] for item in region_results],
                    "threshold": thresholds.get(name),
                    "analytic_threshold": analytic_thresholds.get(name),
                    "signed": results[name].signed,
                    "caption": _METRIC_INFO.get(name, ""),
                }
                for name in metric_list
            ]
            figures_dir = self.out_dir / "figures"
            report_figures = []

            frame = cluster_frame(dataset.df, partition, dataset.types)
            balance = balance_figure(frame, dataset=dataset.name, method=method)
            save_figure(balance, figures_dir / f"explain_{dataset.name}_{method}_balance")
            report_figures.append(balance)

            panels_figure = metric_panels_figure(panels, dataset=dataset.name, method=method)
            save_figure(panels_figure, figures_dir / f"explain_{dataset.name}_{method}_metrics_panels")
            report_figures.append(panels_figure)

            report_figures.extend(
                self._write_cluster_cards(
                    dataset=dataset,
                    partition=partition,
                    adjacency=adjacency,
                    results=results,
                    metric_list=metric_list,
                    figures_dir=figures_dir,
                    method=method,
                    frame=frame,
                )
            )

            save_pdf_report(report_figures, figures_dir / f"explain_{dataset.name}_{method}_report.pdf")
            close_figures(*report_figures)

            dispersion = dispersion_summary(frame)
            dispersion.to_csv(self.out_dir / f"explain_{dataset.name}_dispersion.csv")

            regions_frame = pd.DataFrame(
                [
                    {
                        "cluster_label": item["region"].get("cluster_label"),
                        "origin": item["region"].get("origin", "organic"),
                        "n": item["n"],
                        "p": item["p"],
                        "n_neg": item["n_neg"],
                        "rho": item["rho"],
                        "rho_out": item["rho_out"],
                        "primary_metric": primary_metric,
                        "primary_score": item["score"],
                        "signif_threshold": primary_threshold,
                        "analytic_threshold": analytic_thresholds.get(primary_metric),
                        "significant": item["significant"],
                        "direction": item["direction"],
                        "local_z_delta_after_rescue": item[
                            "local_z_delta_after_rescue"
                        ],
                        **{name: item[f"metric_{name}"] for name in metric_list},
                    }
                    for item in region_results
                ]
            )
            regions_frame.to_csv(self.out_dir / f"explain_{dataset.name}_regions.csv", index=False)

            la_labels = clusters_in_bbox(partition.regions, dataset.df, GREATER_LA_BBOX)
            if la_labels:
                la_frame = regions_frame[regions_frame["cluster_label"].isin(la_labels)]
                la_frame.to_csv(self.out_dir / f"explain_{dataset.name}_la_lens.csv", index=False)
                self.log(f"LA lens: {len(la_labels)} cluster(s) inside greater Los Angeles")

            if primary_metric in null_by_metric:
                pd.DataFrame(
                    {
                        "world_idx": range(len(null_by_metric[primary_metric])),
                        f"max_abs_{primary_metric}": null_by_metric[primary_metric],
                    }
                ).to_csv(self.out_dir / f"explain_{dataset.name}_null_distribution.csv", index=False)

            significant = [item for item in region_results if item["significant"]]
            finite_deltas = [
                row["local_z_delta"]
                for row in organic_z_deltas
                if not math.isnan(row["local_z_delta"])
            ]
            profile_values = partition_profile(partition, dataset.n_total)
            summary = {
                "dataset": dataset.name,
                "method": method,
                "params": partition.params,
                "N": dataset.n_total,
                "P": dataset.p_total,
                "global_rate": dataset.global_rate,
                "n_regions": len(partition.regions),
                "noise_n": partition.noise_n,
                "noise_rate": partition.noise_n / dataset.n_total if dataset.n_total else None,
                "coverage": {
                    "organic_n": profile_values["organic_n"],
                    "organic_rate": _safe_float(profile_values["organic_rate"]),
                    "rescue_n": profile_values["rescue_n"],
                    "rescue_rate": _safe_float(profile_values["rescue_rate"]),
                    "noise_n": profile_values["noise_n"],
                    "noise_rate": _safe_float(profile_values["noise_rate"]),
                },
                "organic_local_z_delta": {
                    "evaluated_clusters": len(finite_deltas),
                    "mean_abs": (
                        float(np.mean(np.abs(finite_deltas))) if finite_deltas else None
                    ),
                    "max_abs": (
                        float(np.max(np.abs(finite_deltas))) if finite_deltas else None
                    ),
                    "clusters": [
                        {
                            key: (_safe_float(value) if key != "origin_cluster_label" else value)
                            for key, value in row.items()
                        }
                        for row in organic_z_deltas
                    ],
                },
                "n_alt_worlds": n_alt_worlds,
                "signif_level": signif_level,
                "elapsed_seconds": round(time.perf_counter() - run_started, 1),
                "primary_metric": primary_metric,
                "signif_threshold": primary_threshold,
                "metric_thresholds": dict(thresholds),
                "analytic_thresholds": {
                    name: _safe_float(value) for name, value in analytic_thresholds.items()
                },
                "metric_scalars": {
                    name: _safe_float(results[name].partition_scalar) for name in metric_list
                },
                "balance": dataset_balance(dataset.types),
                "partition_profile": profile_values,
                "cluster_size_cv": _safe_float(dispersion.loc["n", "cv"]),
                "rho_sigma": _safe_float(dispersion.loc["rho", "std"]),
                "sigma_ratio_positives_over_negatives": _safe_float(
                    dispersion.loc["p", "std"] / dispersion.loc["n_neg", "std"]
                    if dispersion.loc["n_neg", "std"]
                    else float("nan")
                ),
                "sigma_ratio_expected_under_one_rate": expected_sigma_ratio(dataset.global_rate),
                "observed_max_sul": max_sul,
                "significant_regions": len(significant),
                "negative_regions": sum(1 for item in significant if item["direction"] == "negative"),
                "positive_regions": sum(1 for item in significant if item["direction"] == "positive"),
                "meanvar": meanvar,
                "meanvar_max_score": meanvar_max,
                "gini": gini,
            }
            (self.out_dir / f"explain_{dataset.name}_summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )

        self.clustering_rows.append(
            self.common_row(
                experiment="explain",
                dataset=dataset,
                method=partition.method,
                params=partition.params,
                n_regions=len(partition.regions),
                max_sul=max_sul,
                signif_threshold=primary_threshold,
                significant_regions=len(significant),
                best_region=best_region,
                meanvar=meanvar,
                meanvar_max_score=meanvar_max,
                gini=gini,
                noise_n=partition.noise_n,
            )
        )

    def _subcluster_splitter(self, df: pd.DataFrame, min_cluster_size_min: int = 25):
        """Return a function splitting a region's point ids into density subclusters.

        Reuses the density split from the capped partitioner (ADR-0001) so
        `gini_subcluster` and the size cap share one definition of "subcluster".
        `min_cluster_size_min` is the granularity knob the cluster cards vary.
        """
        return lambda points: density_subclusters(
            df, points, min_cluster_size_min, min_samples=self.hdbscan_min_samples
        )

    def _card_cluster_labels(
        self,
        partition: Partition,
        results: dict[str, Any],
        metric_list: list[str],
    ) -> list[int]:
        """Clusters worth a card: the most internally unequal ones.

        Ranked by `gini_subcluster` when it ran (the card exists to show what that
        number is made of); otherwise by the primary metric's magnitude.
        """
        ranking_metric = "gini_subcluster" if "gini_subcluster" in metric_list else (
            metric_list[0] if metric_list else None
        )
        if ranking_metric is None or not partition.regions:
            return []

        scores = np.asarray(results[ranking_metric].per_cluster, dtype=float)
        order = np.argsort(np.where(np.isnan(scores), -np.inf, np.abs(scores)))[::-1]
        return [
            int(partition.regions[int(idx)]["cluster_label"])
            for idx in order[:_CARD_TOP_N]
            if not np.isnan(scores[int(idx)])
        ]

    def _write_cluster_cards(
        self,
        *,
        dataset: LoadedDataset,
        partition: Partition,
        adjacency: dict[int, list[int]],
        results: dict[str, Any],
        metric_list: list[str],
        figures_dir: Path,
        method: str,
        frame: pd.DataFrame,
    ) -> list[Any]:
        """Write one card per interesting cluster, at each subcluster granularity.

        Two granularities on purpose: a pocket that only appears at the finer
        split is an artifact of `min_cluster_size_min`, and that has to be
        visible rather than argued.
        """
        figures = []
        rows = []
        for label in self._card_cluster_labels(partition, results, metric_list):
            for granularity in _CARD_GRANULARITIES:
                card = cluster_card_data(
                    dataset.df,
                    partition,
                    dataset.types,
                    cluster_label=label,
                    splitter=self._subcluster_splitter(dataset.df, granularity),
                    adjacency=adjacency,
                    global_rate=dataset.global_rate,
                    frame=frame,
                )
                figure = cluster_card_figure(
                    card, dataset=dataset.name, granularity=str(granularity)
                )
                save_figure(
                    figure,
                    figures_dir / f"explain_{dataset.name}_{method}_card_cluster{label}_min{granularity}",
                )
                figures.append(figure)

                subclusters = card.pop("subclusters")
                for _, sub in subclusters.iterrows():
                    rows.append(
                        {
                            "cluster_label": label,
                            "granularity": granularity,
                            "gini_subcluster": card["gini_subcluster"],
                            "rho_in": card["rho_in"],
                            "rho_peer": card["rho_peer"],
                            "rho_global": card["rho_global"],
                            "homogeneous": card["homogeneous"],
                            **sub.to_dict(),
                        }
                    )

        if rows:
            pd.DataFrame(rows).to_csv(
                self.out_dir / f"explain_{dataset.name}_cluster_cards.csv", index=False
            )
        return figures

    def write_outputs(self) -> None:
        with self.timed_step("Writing CSV outputs"):
            self._write_dataset_csvs("unrestricted_{dataset}_regions.csv", self.unrestricted_rows)
            self._write_dataset_csvs("one_partitioning_{dataset}.csv", self.one_partitioning_rows)
            self._write_dataset_csvs("multiple_partitionings_{dataset}.csv", self.multiple_partitioning_rows)
            self._write_clustering_csvs("{method}_{dataset}_comparison.csv", self.clustering_rows)

    def _write_csv(self, filename: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        pd.DataFrame(rows).to_csv(self.out_dir / filename, index=False)

    def _write_dataset_csvs(self, filename_template: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return

        frame = pd.DataFrame(rows)
        if "dataset" not in frame.columns:
            self._write_csv(filename_template.format(dataset="all"), rows)
            return

        for dataset_name, dataset_rows in frame.groupby("dataset", dropna=False, sort=True):
            filename = filename_template.format(dataset=dataset_name)
            dataset_rows.to_csv(self.out_dir / filename, index=False)

    def _write_clustering_csvs(self, filename_template: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return

        frame = pd.DataFrame(rows)
        for (method, dataset_name), group_rows in frame.groupby(["method", "dataset"], dropna=False, sort=True):
            filename = filename_template.format(method=method, dataset=dataset_name)
            group_rows.to_csv(self.out_dir / filename, index=False)
