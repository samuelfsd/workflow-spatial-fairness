"""Versioned, regenerable evidence snapshot for one ``explain`` run."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from clustering.base import Partition
from data_loading import LoadedDataset, REPO_ROOT, file_sha256
from metrics.base import MetricResult


SCHEMA_VERSION = 1
MANIFEST_NAME = "run_manifest.json"


@dataclass
class RunSnapshot:
    run_dir: Path
    manifest: dict[str, Any]
    assignments: pd.DataFrame
    scores: pd.DataFrame
    thresholds: pd.DataFrame
    null_distributions: pd.DataFrame


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _git_provenance() -> dict[str, Any]:
    """Record commit and a digest of dirty evidence, never the diff itself."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1"], cwd=REPO_ROOT, check=True,
            capture_output=True, timeout=5,
        ).stdout
        diff = subprocess.run(
            ["git", "diff", "--no-ext-diff", "--binary", "HEAD"],
            cwd=REPO_ROOT, check=True, capture_output=True, timeout=15,
        ).stdout
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=REPO_ROOT, check=True, capture_output=True, timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {"available": False, "commit": None, "dirty": None, "diff_sha256": None}

    dirty = bool(status)
    if dirty:
        dirty_digest = hashlib.sha256(status + diff)
        for raw_name in untracked.split(b"\0"):
            if not raw_name:
                continue
            dirty_digest.update(raw_name)
            path = REPO_ROOT / raw_name.decode("utf-8", errors="surrogateescape")
            if path.is_file():
                with path.open("rb") as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        dirty_digest.update(chunk)
        digest = dirty_digest.hexdigest()
    else:
        digest = None
    return {"available": True, "commit": commit, "dirty": dirty, "diff_sha256": digest}


def _logical_dataset_path(dataset: LoadedDataset) -> str:
    try:
        return str(dataset.source_path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(dataset.source_path)


def _point_assignments(partition: Partition, n_total: int) -> pd.DataFrame:
    labels = np.full(n_total, -1, dtype=int)
    origins: list[str | None] = [None] * n_total
    parents: list[int | None] = [None] * n_total
    seen: set[int] = set()
    for region in partition.regions:
        label = int(region["cluster_label"])
        origin = str(region.get("origin", "organic"))
        parent = int(region.get("origin_cluster_label", label))
        for raw_point in region["points"]:
            point = int(raw_point)
            if point < 0 or point >= n_total:
                raise ValueError(f"Point id outside canonical dataset: {point}")
            if point in seen:
                raise ValueError(f"Point id assigned more than once: {point}")
            seen.add(point)
            labels[point] = label
            origins[point] = origin
            parents[point] = parent

    return pd.DataFrame(
        {
            "point_id": np.arange(n_total, dtype=int),
            "cluster_label": pd.array(
                [label if label >= 0 else pd.NA for label in labels], dtype="Int64"
            ),
            "assignment_status": np.where(labels >= 0, "assigned", "unassigned"),
            "origin": origins,
            "origin_cluster_label": pd.array(parents, dtype="Int64"),
        }
    )


def _score_frame(
    partition: Partition, metric_results: dict[str, MetricResult]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, result in metric_results.items():
        if len(result.per_cluster) != len(partition.regions):
            raise ValueError(f"Metric {name!r} is not aligned with partition regions")
        for metadata_name, metadata_values in result.per_cluster_metadata.items():
            if len(metadata_values) != len(partition.regions):
                raise ValueError(
                    f"Metric {name!r} metadata {metadata_name!r} is not region-aligned"
                )
        for idx, region in enumerate(partition.regions):
            row = {
                "cluster_label": int(region["cluster_label"]),
                "metric": name,
                "score": float(result.per_cluster[idx]),
                "partition_scalar": result.partition_scalar,
                "signed": result.signed,
                "supports_mc": result.supports_mc,
                "standardized": result.standardized,
            }
            for metadata_name, metadata_values in result.per_cluster_metadata.items():
                value = metadata_values[idx]
                row[metadata_name] = value.item() if isinstance(value, np.generic) else value
            rows.append(row)
    return pd.DataFrame(rows)


def write_run_snapshot(
    run_dir: Path,
    *,
    dataset: LoadedDataset,
    partition: Partition,
    metric_results: dict[str, MetricResult],
    thresholds: dict[str, float | None],
    null_distributions: dict[str, np.ndarray],
    effective_seeds: dict[str, int],
    seed: int,
    primary_metric: str,
    signif_level: float,
    n_alt_worlds: int,
    command: str,
    exploration_profile: str,
) -> Path:
    """Persist all evidence needed to render another primary without recomputation."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    assignments = _point_assignments(partition, dataset.n_total)
    scores = _score_frame(partition, metric_results)
    threshold_rows = []
    null_rows = []
    for name, result in metric_results.items():
        null = np.asarray(null_distributions.get(name, np.array([], dtype=float)), dtype=float)
        threshold_rows.append(
            {
                "metric": name,
                "threshold": thresholds.get(name),
                "n_worlds": len(null),
                "effective_seed": effective_seeds.get(name),
                "supports_mc": result.supports_mc,
            }
        )
        for world_idx, value in enumerate(null):
            null_rows.append(
                {
                    "metric": name,
                    "world_idx": world_idx,
                    "max_abs_score": float(value),
                    "effective_seed": effective_seeds.get(name),
                }
            )
    thresholds_frame = pd.DataFrame(threshold_rows)
    null_frame = pd.DataFrame(
        null_rows, columns=["metric", "world_idx", "max_abs_score", "effective_seed"]
    )

    assignments.to_csv(run_dir / "point_assignments.csv", index=False)
    scores.to_csv(run_dir / "cluster_scores.csv", index=False)
    thresholds_frame.to_csv(run_dir / "metric_thresholds.csv", index=False)
    null_frame.to_csv(run_dir / "metric_null_distributions.csv", index=False)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": dataset.name,
            "logical_path": _logical_dataset_path(dataset),
            "source_sha256": dataset.source_sha256,
            "canonical_sha256": dataset.canonical_sha256,
            "rows_before_clean": dataset.rows_before_clean,
            "rows_after_clean": dataset.n_total,
            "columns_used": ["lat", "lon", dataset.spec.label_column],
            "outcome": {
                "positive_label": dataset.spec.positive_label,
                "negative_label": dataset.spec.negative_label,
                "desirability": dataset.spec.desirability,
            },
        },
        "partition": {
            "method": partition.method,
            "params": partition.params,
            "n_regions": len(partition.regions),
            "noise_n": int((assignments["assignment_status"] == "unassigned").sum()),
        },
        "run": {
            "seed": int(seed),
            "metrics": list(metric_results),
            "primary_metric": primary_metric,
            "n_alt_worlds_requested": int(n_alt_worlds),
            "signif_level": float(signif_level),
            "command": command,
            "exploration_profile": exploration_profile,
        },
        "git": _git_provenance(),
        "completion": {
            "experiment": True,
            "exploration_report": False,
            "exploration_reports": {},
        },
    }
    (run_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return run_dir


def mark_exploration_complete(
    run_dir: Path, *, primary_metric: str, profile: str, report_dir: Path
) -> None:
    """Atomically record a successfully published derived report."""
    manifest_path = Path(run_dir) / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    completion = manifest.setdefault("completion", {})
    completion["exploration_report"] = True
    completion.setdefault("exploration_reports", {})[primary_metric] = {
        "profile": profile,
        "report_dir": str(report_dir),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)


def load_run_snapshot(run_dir: Path, dataset: LoadedDataset) -> RunSnapshot:
    """Load a snapshot only after schema and dataset identity are validated."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Run sem snapshot versionado: {manifest_path}. Regeneração aproximada é recusada."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Incompatible snapshot schema: expected {SCHEMA_VERSION}, "
            f"got {manifest.get('schema_version')}"
        )

    recorded = manifest.get("dataset", {})
    current_source_hash = file_sha256(dataset.source_path)
    if recorded.get("source_sha256") != current_source_hash:
        raise ValueError("Dataset SHA-256 differs from the snapshot")
    if recorded.get("rows_after_clean") != dataset.n_total:
        raise ValueError("Dataset final row count differs from the snapshot")
    if recorded.get("canonical_sha256") != dataset.canonical_sha256:
        raise ValueError("Dataset canonical order/content differs from the snapshot")

    required = {
        "point_assignments.csv": "assignments",
        "cluster_scores.csv": "scores",
        "metric_thresholds.csv": "thresholds",
        "metric_null_distributions.csv": "null_distributions",
    }
    frames: dict[str, pd.DataFrame] = {}
    for filename, key in required.items():
        path = run_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Incomplete versioned snapshot: missing {filename}")
        frames[key] = pd.read_csv(path)

    assignments = frames["assignments"]
    if len(assignments) != dataset.n_total or not np.array_equal(
        assignments["point_id"].to_numpy(dtype=int), np.arange(dataset.n_total)
    ):
        raise ValueError("Snapshot point order/count is incompatible with the dataset")
    return RunSnapshot(run_dir=run_dir, manifest=manifest, **frames)
