"""Experiment orchestration for the original spatial fairness audit plus HDBSCAN."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd

from clustering.hdbscan import HDBSCANPartition, run_hdbscan_sweep
from data_loading import LoadedDataset, load_dataset
from metrics.group_fairness import (
    calculate_meanvar,
    get_signif_threshold,
    get_simple_stats,
    scan_partitioning,
    scan_regions,
    select_significant_regions,
)
from regions import (
    create_grid_from_dataset,
    create_random_partitionings,
    create_regions,
    create_rtree,
    create_seeds,
    filter_non_overlapping_regions,
)
from visualization import save_experiment_map


def _json_params(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True)


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
    ) -> None:
        self.out_dir = out_dir
        self.maps = maps
        self.seed = seed
        self.hdbscan_fracs = hdbscan_fracs
        self.max_map_points = max_map_points
        self.verbose = verbose
        self.started_at = time.perf_counter()
        self.dataset_cache: dict[str, LoadedDataset] = {}
        self.hdbscan_cache: dict[str, list[HDBSCANPartition]] = {}

        self.unrestricted_rows: list[dict[str, Any]] = []
        self.one_partitioning_rows: list[dict[str, Any]] = []
        self.multiple_partitioning_rows: list[dict[str, Any]] = []
        self.hdbscan_rows: list[dict[str, Any]] = []

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

    def hdbscan_partitions(self, dataset: LoadedDataset) -> list[HDBSCANPartition]:
        if dataset.name not in self.hdbscan_cache:
            partitions = []
            for idx, frac in enumerate(self.hdbscan_fracs, start=1):
                with self.timed_step(f"HDBSCAN {idx}/{len(self.hdbscan_fracs)} frac={frac}"):
                    partitions.extend(run_hdbscan_sweep(dataset.df, (frac,)))
            self.hdbscan_cache[dataset.name] = partitions
        return self.hdbscan_cache[dataset.name]

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
            "noise_n": noise_n,
            "noise_rate": (noise_n / dataset.n_total) if noise_n is not None and dataset.n_total else None,
        }
        row.update(_region_stats(best_region, dataset.types))
        return row

    def evaluate_hdbscan(
        self,
        *,
        experiment: str,
        dataset: LoadedDataset,
        n_alt_worlds: int,
        signif_level: float,
    ) -> list[tuple[dict[str, Any], HDBSCANPartition, list[dict]]]:
        results = []
        for partition in self.hdbscan_partitions(dataset):
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
                    seed=self.seed + partition.min_cluster_size,
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
                method="hdbscan",
                params={
                    "min_cluster_frac": partition.min_cluster_frac,
                    "min_cluster_size": partition.min_cluster_size,
                    "metric": "haversine",
                },
                n_regions=len(partition.regions),
                max_sul=max_sul,
                signif_threshold=threshold,
                significant_regions=len(significant),
                best_region=best_region,
                meanvar=meanvar,
                meanvar_max_score=meanvar_max,
                noise_n=partition.noise_n,
            )
            self.hdbscan_rows.append(row)
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

        with self.timed_step("[5/6] Running HDBSCAN + SUL comparison"):
            hdbscan_results = self.evaluate_hdbscan(
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

        with self.timed_step("[5/5] Running HDBSCAN + SUL comparison"):
            hdbscan_results = self.evaluate_hdbscan(
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
            for idx, (grid_info, _, regions) in enumerate(partitionings):
                best_region, max_score, mean_score, rhos = _meanvar_summary(regions, dataset.types)
                mean_scores.append(mean_score)
                max_scores.append(max_score)
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
                    "noise_n": None,
                    "noise_rate": None,
                    "best_region_n": None,
                    "best_region_p": None,
                    "best_region_rate": None,
                }
            )

        with self.timed_step("[4/4] Running HDBSCAN reference"):
            self.evaluate_hdbscan(
                experiment="multiple_partitionings",
                dataset=dataset,
                n_alt_worlds=0,
                signif_level=0.005,
            )

    def write_outputs(self) -> None:
        with self.timed_step("Writing CSV outputs"):
            self._write_dataset_csvs("unrestricted_{dataset}_regions.csv", self.unrestricted_rows)
            self._write_dataset_csvs("one_partitioning_{dataset}.csv", self.one_partitioning_rows)
            self._write_dataset_csvs("multiple_partitionings_{dataset}.csv", self.multiple_partitioning_rows)
            self._write_dataset_csvs("hdbscan_{dataset}_comparison.csv", self.hdbscan_rows)

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
