"""End-to-end orchestration for the initial Sacharidis benchmark."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

from benchmark_reference import (
    build_data_parity_table,
    load_reference_manifest,
    observed_dataset_record,
)
from benchmark_report import (
    checkpoint_summaries_to_long,
    compare_compatible_results,
    load_checkpoint_results,
    publish_initial_report,
    reject_incompatible_checkpoint_merges,
    reference_results_frame,
)
from benchmark_sacharidis import SacharidisBenchmarkRunner, SacharidisProtocol
from data_loading import DATASET_SPECS, REPO_ROOT, load_dataset


Phase = Literal["reproduce", "compare", "report", "all"]
DEFAULT_DATASETS = ("semisynth", "synth_unfair", "synth_fair", "lar", "crime")


@dataclass(frozen=True)
class InitialBenchmarkConfig:
    output_root: Path
    datasets: tuple[str, ...] = DEFAULT_DATASETS
    phase: Phase = "all"
    seed: int = 42
    resume: bool = True
    maps: bool = False
    reference_path: Path = REPO_ROOT / "benchmarks" / "sacharidis" / "reference.json"
    protocol: SacharidisProtocol = field(default_factory=SacharidisProtocol)


def _checkpoint_metadata(root: Path) -> dict:
    units = []
    for path in sorted(root.rglob("manifest.json")):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("status") == "complete" and "unit" in manifest:
            units.append(manifest["unit"])
    return {"units": units}


def publish_report_from_checkpoints(config: InitialBenchmarkConfig) -> Path:
    """Regenerate presentation artifacts without clustering, KMeans or Monte Carlo."""
    manifest = load_reference_manifest(config.reference_path)
    selected = set(config.datasets)
    manifest = {
        **manifest,
        "datasets": [item for item in manifest["datasets"] if item["dataset"] in selected],
        "results": [item for item in manifest["results"] if item["dataset"] in selected],
    }
    observed = {
        item["dataset"]: observed_dataset_record(load_dataset(item["dataset"]))
        for item in manifest["datasets"]
    }
    parity = build_data_parity_table(manifest, observed)
    checkpoint_frame = load_checkpoint_results(config.output_root / "checkpoints")
    if not checkpoint_frame.empty:
        checkpoint_frame = checkpoint_frame[checkpoint_frame["dataset"].isin(selected)]
        reject_incompatible_checkpoint_merges(checkpoint_frame)
        for value in checkpoint_frame["checkpoint_unit"].dropna().unique():
            unit = json.loads(value)
            dataset_name = unit["dataset"]
            if unit["dataset_sha256"] != observed[dataset_name]["source_sha256"]:
                raise ValueError(
                    f"checkpoint de {dataset_name} usa hash de dataset incompatível"
                )
            if int(unit["seed"]) != config.seed:
                raise ValueError(
                    f"checkpoint de {dataset_name} usa seed incompatível com o relatório"
                )
            partitioning = str(unit["partitioning"])
            protocol = str(unit["protocol"])
            expected_worlds = (
                config.protocol.standardized_worlds
                if protocol == "standardized"
                else 0
                if partitioning == "random_grids"
                else config.protocol.reproduction_scan_worlds
                if partitioning.startswith("kmeans_square_scan")
                else config.protocol.grid_worlds
            )
            if int(unit["n_alt_worlds"]) != expected_worlds:
                raise ValueError(
                    f"checkpoint de {dataset_name}/{partitioning} usa mundos nulos incompatíveis"
                )
            if protocol != "reproduction" or partitioning != "random_grids":
                if float(unit.get("params", {}).get("signif_level", float("nan"))) != config.protocol.signif_level:
                    raise ValueError(
                        f"checkpoint de {dataset_name}/{partitioning} usa alfa incompatível"
                    )
    if checkpoint_frame.empty:
        summaries = pd.DataFrame()
        regions = pd.DataFrame()
    else:
        summaries = checkpoint_frame[checkpoint_frame["record_type"].eq("summary")].copy()
        regions = checkpoint_frame[checkpoint_frame["record_type"].eq("region")].copy()
    local = checkpoint_summaries_to_long(summaries)
    references = reference_results_frame(manifest)
    canonical = pd.concat([references, local], ignore_index=True, sort=False)
    canonical = canonical.sort_values(
        ["dataset", "experiment", "region_system", "metric", "quantity", "source", "protocol"],
        kind="stable", na_position="last",
    ).reset_index(drop=True)
    comparisons = compare_compatible_results(references, local)
    metadata = _checkpoint_metadata(config.output_root / "checkpoints")
    metadata.update({
        "config": {**asdict(config), "output_root": str(config.output_root), "reference_path": str(config.reference_path), "protocol": asdict(config.protocol)},
        "dataset_hashes": {name: record["source_sha256"] for name, record in observed.items()},
    })
    return publish_initial_report(
        config.output_root / "report",
        canonical=canonical,
        comparisons=comparisons,
        parity=parity,
        regions=regions,
        render_figures=True,
        render_maps=config.maps,
        run_metadata=metadata,
    )


def run_initial_benchmark(config: InitialBenchmarkConfig) -> Path:
    unknown = sorted(set(config.datasets).difference(DATASET_SPECS))
    if unknown:
        raise ValueError(f"dataset desconhecido no benchmark: {unknown}")
    if config.phase not in {"reproduce", "compare", "report", "all"}:
        raise ValueError(f"fase desconhecida: {config.phase}")
    checkpoints = config.output_root / "checkpoints"
    if not config.resume and checkpoints.exists() and any(checkpoints.rglob("manifest.json")):
        raise FileExistsError("checkpoints existentes; use --resume ou uma nova saída")
    runner = SacharidisBenchmarkRunner(
        checkpoints, protocol=config.protocol, seed=config.seed
    )
    if config.phase in {"reproduce", "all"}:
        for dataset in config.datasets:
            runner.run_reproduce(dataset)
    if config.phase in {"compare", "all"}:
        for dataset in config.datasets:
            runner.run_compare(dataset)
    if config.phase in {"report", "all"}:
        return publish_report_from_checkpoints(config)
    return checkpoints
