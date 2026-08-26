"""Checkpointed execution units for the Sacharidis quantitative benchmark."""

from __future__ import annotations

import json
import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from benchmark_checkpoint import (
    BenchmarkUnitSpec,
    checkpoint_state,
    load_benchmark_checkpoint,
    publish_benchmark_checkpoint,
)
from benchmark_evaluation import EvaluationBundle, evaluate_partition, evaluate_scan
from clustering.base import Partition
from clustering.hdbscan import fit_hdbscan_partition
from data_loading import LoadedDataset, REPO_ROOT
from data_loading import load_dataset
from regions import (
    create_grid_from_dataset,
    create_random_partitionings,
    create_regions,
    create_rtree,
    create_seeds,
)


HDBSCAN_BENCHMARK_METRICS = (
    "sul",
    "local_z",
    "peer_rate_difference",
    "peer_log_rate_ratio",
    "peer_gini_gap",
    "gini_subcluster",
    "gini",
    "meanvar",
    "dp_difference",
    "dp_ratio",
)


@dataclass
class PublishedBenchmarkUnit:
    path: Path
    spec: BenchmarkUnitSpec
    results: pd.DataFrame
    summary: pd.DataFrame
    regions: pd.DataFrame
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SacharidisProtocol:
    reproduction_scan_worlds: int = 200
    grid_worlds: int = 1000
    standardized_worlds: int = 1000
    random_partitionings: int = 100
    kmeans_seeds: int = 100
    hdbscan_fracs: tuple[float, ...] = (0.005, 0.01, 0.02)
    hdbscan_min_samples: int = 60
    signif_level: float = 0.005


def code_provenance() -> dict[str, Any]:
    """Return stable git evidence for benchmark fingerprints."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1"], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True, timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}
    source_digest = hashlib.sha256()
    for path in sorted((REPO_ROOT / "src").rglob("*.py")):
        source_digest.update(str(path.relative_to(REPO_ROOT)).encode("utf-8"))
        source_digest.update(path.read_bytes())
    return {
        "commit": commit,
        "dirty": bool(status),
        "source_sha256": source_digest.hexdigest(),
    }


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "-" for char in value)


def _unit_path(root: Path, spec: BenchmarkUnitSpec) -> Path:
    metric = _safe_name(spec.metric)
    partitioning = _safe_name(spec.partitioning)
    return (
        Path(root)
        / spec.dataset
        / spec.protocol
        / f"{partitioning}__{metric}"
    )


def _bundle_frame(bundle: EvaluationBundle) -> pd.DataFrame:
    return pd.concat([bundle.summary, bundle.regions], ignore_index=True, sort=False)


def _metadata(bundle: EvaluationBundle, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    value = dict(extra or {})
    value["null_distributions"] = {
        name: np.asarray(scores, dtype=float).tolist()
        for name, scores in bundle.null_distributions.items()
    }
    return value


def _published(path: Path, spec: BenchmarkUnitSpec) -> PublishedBenchmarkUnit:
    checkpoint = load_benchmark_checkpoint(path, spec)
    frame = checkpoint.results
    return PublishedBenchmarkUnit(
        path=path,
        spec=spec,
        results=frame,
        summary=frame[frame["record_type"] == "summary"].reset_index(drop=True),
        regions=frame[frame["record_type"] == "region"].reset_index(drop=True),
        metadata=checkpoint.metadata,
    )


def _publish(
    root: Path,
    spec: BenchmarkUnitSpec,
    bundle: EvaluationBundle,
    *,
    extra_metadata: dict[str, Any] | None = None,
) -> PublishedBenchmarkUnit:
    path = _unit_path(root, spec)
    publish_benchmark_checkpoint(
        path,
        spec,
        _bundle_frame(bundle),
        metadata=_metadata(bundle, extra_metadata),
    )
    return _published(path, spec)


def _partition_from_grid(
    dataset: LoadedDataset,
    lon_n: int,
    lat_n: int,
) -> tuple[Partition, dict[str, Any]]:
    rtree = create_rtree(dataset.df)
    grid_info, _, raw_regions = create_grid_from_dataset(
        dataset.df, rtree, lon_n=lon_n, lat_n=lat_n
    )
    regions: list[dict[str, Any]] = []
    labels = np.full(dataset.n_total, -1, dtype=int)
    for label, raw in enumerate(raw_regions):
        region = dict(raw)
        region["cluster_label"] = label
        regions.append(region)
        labels[np.asarray(region["points"], dtype=int)] = label
    noise = np.flatnonzero(labels < 0).astype(int).tolist()
    partitioning = f"grid_{lon_n}x{lat_n}"
    return (
        Partition(
            method="grid",
            params={"partitioning": partitioning, "lon_n": lon_n, "lat_n": lat_n},
            labels=labels,
            regions=regions,
            noise_points=noise,
        ),
        grid_info,
    )


def run_partition_unit(
    dataset: LoadedDataset,
    partition: Partition,
    output_root: Path,
    *,
    protocol: str,
    metrics: Iterable[str],
    n_alt_worlds: int,
    signif_level: float,
    seed: int,
    code_provenance: dict[str, Any] | None = None,
) -> PublishedBenchmarkUnit:
    metric_names = tuple(metrics)
    partitioning = str(partition.params.get("partitioning", partition.method))
    spec = BenchmarkUnitSpec(
        dataset=dataset.name,
        dataset_sha256=dataset.source_sha256,
        protocol=protocol,
        partitioning=partitioning,
        metric="+".join(metric_names),
        params={**partition.params, "signif_level": signif_level},
        seed=seed,
        n_alt_worlds=n_alt_worlds,
        code_provenance=code_provenance or globals()["code_provenance"](),
    )
    path = _unit_path(output_root, spec)
    if checkpoint_state(path, spec) == "complete":
        return _published(path, spec)
    bundle = evaluate_partition(
        dataset,
        partition,
        metrics=metric_names,
        protocol=protocol,
        n_alt_worlds=n_alt_worlds,
        signif_level=signif_level,
        seed=seed,
    )
    return _publish(output_root, spec, bundle)


def run_grid_unit(
    dataset: LoadedDataset,
    output_root: Path,
    *,
    lon_n: int,
    lat_n: int,
    protocol: str,
    n_alt_worlds: int,
    signif_level: float,
    seed: int,
    code_provenance: dict[str, Any] | None = None,
) -> PublishedBenchmarkUnit:
    partition, grid_info = _partition_from_grid(dataset, lon_n, lat_n)
    unit = run_partition_unit(
        dataset,
        partition,
        output_root,
        protocol=protocol,
        metrics=("sul", "meanvar"),
        n_alt_worlds=n_alt_worlds,
        signif_level=signif_level,
        seed=seed,
        code_provenance=code_provenance,
    )
    return unit


def run_scan_unit(
    dataset: LoadedDataset,
    output_root: Path,
    *,
    protocol: str,
    n_seeds: int,
    n_alt_worlds: int,
    signif_level: float,
    seed: int,
    direction: str = "both",
    code_provenance: dict[str, Any] | None = None,
) -> PublishedBenchmarkUnit:
    spec = BenchmarkUnitSpec(
        dataset=dataset.name,
        dataset_sha256=dataset.source_sha256,
        protocol=protocol,
        partitioning=f"kmeans_square_scan_{direction}",
        metric="sul",
        params={
            "n_seeds": n_seeds, "radii": dataset.radii.tolist(),
            "direction": direction, "signif_level": signif_level,
        },
        seed=seed,
        n_alt_worlds=n_alt_worlds,
        code_provenance=code_provenance or globals()["code_provenance"](),
    )
    path = _unit_path(output_root, spec)
    if checkpoint_state(path, spec) == "complete":
        return _published(path, spec)
    rtree = create_rtree(dataset.df)
    seeds = create_seeds(dataset.df, rtree, n_seeds, random_state=seed)
    regions = create_regions(dataset.df, rtree, seeds, dataset.radii)
    bundle = evaluate_scan(
        dataset,
        regions,
        protocol=protocol,
        n_alt_worlds=n_alt_worlds,
        signif_level=signif_level,
        seed=seed,
        direction=direction,
    )
    bundle.summary["params"] = json.dumps(spec.params, sort_keys=True)
    return _publish(output_root, spec, bundle)


def run_hdbscan_unit(
    dataset: LoadedDataset,
    output_root: Path,
    *,
    min_cluster_frac: float,
    min_samples: int,
    protocol: str,
    n_alt_worlds: int,
    signif_level: float,
    seed: int,
    code_provenance: dict[str, Any] | None = None,
) -> PublishedBenchmarkUnit:
    partition = fit_hdbscan_partition(
        dataset.df,
        min_cluster_frac=min_cluster_frac,
        min_samples=min_samples,
    )
    partition.params["partitioning"] = f"hdbscan_frac_{min_cluster_frac:g}"
    return run_partition_unit(
        dataset,
        partition,
        output_root,
        protocol=protocol,
        metrics=HDBSCAN_BENCHMARK_METRICS,
        n_alt_worlds=n_alt_worlds,
        signif_level=signif_level,
        seed=seed,
        code_provenance=code_provenance,
    )


def run_random_grid_meanvar_unit(
    dataset: LoadedDataset,
    output_root: Path,
    *,
    n_partitionings: int,
    lon_n_range: tuple[int, int] = (10, 40),
    lat_n_range: tuple[int, int] = (10, 40),
    seed: int,
    code_provenance: dict[str, Any] | None = None,
) -> PublishedBenchmarkUnit:
    params = {
        "n_partitionings": n_partitionings,
        "lon_n_range": list(lon_n_range),
        "lat_n_range": list(lat_n_range),
    }
    spec = BenchmarkUnitSpec(
        dataset=dataset.name,
        dataset_sha256=dataset.source_sha256,
        protocol="reproduction",
        partitioning="random_grids",
        metric="meanvar",
        params=params,
        seed=seed,
        n_alt_worlds=0,
        code_provenance=code_provenance or globals()["code_provenance"](),
    )
    path = _unit_path(output_root, spec)
    if checkpoint_state(path, spec) == "complete":
        return _published(path, spec)

    rtree = create_rtree(dataset.df)
    generated = create_random_partitionings(
        dataset.df,
        rtree,
        n_partitionings=n_partitionings,
        lon_n_range=lon_n_range,
        lat_n_range=lat_n_range,
        seed=seed,
    )
    summaries = []
    region_frames = []
    for idx, (grid_info, _, raw_regions) in enumerate(generated):
        labels = np.full(dataset.n_total, -1, dtype=int)
        regions = []
        for label, raw in enumerate(raw_regions):
            region = dict(raw, cluster_label=label)
            regions.append(region)
            labels[np.asarray(region["points"], dtype=int)] = label
        partition = Partition(
            method="random_grid",
            params={"partitioning": f"random_grid_{idx}", **grid_info},
            labels=labels,
            regions=regions,
            noise_points=np.flatnonzero(labels < 0).astype(int).tolist(),
        )
        bundle = evaluate_partition(
            dataset,
            partition,
            metrics=("meanvar",),
            protocol="reproduction",
            n_alt_worlds=0,
            signif_level=0.005,
            seed=seed + idx,
        )
        summary = bundle.summary.copy()
        summary["partitioning_idx"] = idx
        regions_frame = bundle.regions.copy()
        regions_frame["partitioning_idx"] = idx
        summaries.append(summary)
        region_frames.append(regions_frame)

    summary_frame = pd.concat(summaries, ignore_index=True, sort=False) if summaries else pd.DataFrame()
    region_frame = pd.concat(region_frames, ignore_index=True, sort=False) if region_frames else pd.DataFrame()
    aggregate = (
        summary_frame.iloc[0].to_dict()
        if len(summary_frame)
        else {
            "record_type": "summary", "source": "local", "dataset": dataset.name,
            "protocol": "reproduction", "metric": "meanvar",
        }
    )
    aggregate.update(
        {
            "method": "random_grid_summary",
            "partitioning": "random_grids",
            "params": json.dumps(params, sort_keys=True),
            "partition_score": float(summary_frame["partition_score"].mean()) if len(summary_frame) else np.nan,
            "score": float(summary_frame["score"].max()) if len(summary_frame) else np.nan,
            "n_partitionings": n_partitionings,
            "partitioning_idx": np.nan,
        }
    )
    summary_frame = pd.concat([summary_frame, pd.DataFrame([aggregate])], ignore_index=True, sort=False)
    bundle = EvaluationBundle(summary_frame, region_frame)
    return _publish(output_root, spec, bundle)


class SacharidisBenchmarkRunner:
    """Route frozen datasets through reproduction and standardized protocols."""

    def __init__(
        self,
        output_root: Path,
        *,
        protocol: SacharidisProtocol | None = None,
        dataset_loader=load_dataset,
        seed: int = 42,
        code_provenance: dict[str, Any] | None = None,
    ) -> None:
        self.output_root = Path(output_root)
        self.protocol = protocol or SacharidisProtocol()
        self.dataset_loader = dataset_loader
        self.seed = seed
        self.code_provenance = code_provenance or globals()["code_provenance"]()

    def _grid_units(self, dataset: LoadedDataset, *, protocol: str) -> list[PublishedBenchmarkUnit]:
        worlds = (
            self.protocol.grid_worlds
            if protocol == "reproduction"
            else self.protocol.standardized_worlds
        )
        return [
            run_grid_unit(
                dataset,
                self.output_root,
                lon_n=lon_n,
                lat_n=lat_n,
                protocol=protocol,
                n_alt_worlds=worlds,
                signif_level=self.protocol.signif_level,
                seed=self.seed,
                code_provenance=self.code_provenance,
            )
            for lon_n, lat_n in dataset.fixed_grids
        ]

    def run_reproduce(self, dataset_name: str) -> list[PublishedBenchmarkUnit]:
        dataset = self.dataset_loader(dataset_name)
        units = self._grid_units(dataset, protocol="reproduction")
        if dataset_name in {"semisynth", "synth_unfair", "synth_fair"}:
            units.append(
                run_random_grid_meanvar_unit(
                    dataset,
                    self.output_root,
                    n_partitionings=self.protocol.random_partitionings,
                    seed=self.seed,
                    code_provenance=self.code_provenance,
                )
            )
        if dataset_name == "lar":
            for direction in ("both", "less_in", "less_out"):
                units.append(
                    run_scan_unit(
                        dataset,
                        self.output_root,
                        protocol="reproduction",
                        n_seeds=self.protocol.kmeans_seeds,
                        n_alt_worlds=self.protocol.reproduction_scan_worlds,
                        signif_level=self.protocol.signif_level,
                        seed=self.seed,
                        direction=direction,
                        code_provenance=self.code_provenance,
                    )
                )
        return units

    def run_compare(self, dataset_name: str) -> list[PublishedBenchmarkUnit]:
        dataset = self.dataset_loader(dataset_name)
        units = self._grid_units(dataset, protocol="standardized")
        if dataset_name == "lar":
            units.append(
                run_scan_unit(
                    dataset,
                    self.output_root,
                    protocol="standardized",
                    n_seeds=self.protocol.kmeans_seeds,
                    n_alt_worlds=self.protocol.standardized_worlds,
                    signif_level=self.protocol.signif_level,
                    seed=self.seed,
                    direction="both",
                    code_provenance=self.code_provenance,
                )
            )
        for frac in self.protocol.hdbscan_fracs:
            units.append(
                run_hdbscan_unit(
                    dataset,
                    self.output_root,
                    min_cluster_frac=frac,
                    min_samples=self.protocol.hdbscan_min_samples,
                    protocol="standardized",
                    n_alt_worlds=self.protocol.standardized_worlds,
                    signif_level=self.protocol.signif_level,
                    seed=self.seed,
                    code_provenance=self.code_provenance,
                )
            )
        return units
