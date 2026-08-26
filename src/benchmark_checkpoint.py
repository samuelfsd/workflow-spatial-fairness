"""Transactional, fingerprinted checkpoints for benchmark execution units."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd


CHECKPOINT_SCHEMA_VERSION = 1
CheckpointState = Literal["missing", "incomplete", "incompatible", "complete"]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class BenchmarkUnitSpec:
    dataset: str
    dataset_sha256: str
    protocol: str
    partitioning: str
    metric: str
    params: dict[str, Any]
    seed: int
    n_alt_worlds: int
    code_provenance: dict[str, Any]

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical_json(asdict(self)).encode("utf-8")).hexdigest()


@dataclass
class BenchmarkCheckpoint:
    path: Path
    manifest: dict[str, Any]
    results: pd.DataFrame
    metadata: dict[str, Any]


def checkpoint_state(path: Path, expected: BenchmarkUnitSpec) -> CheckpointState:
    path = Path(path)
    if not path.exists():
        return "missing"
    manifest_path = path / "manifest.json"
    result_path = path / "results.csv"
    if not manifest_path.exists():
        return "incomplete"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return "incomplete"
    if manifest.get("status") != "complete" or not result_path.exists():
        return "incomplete"
    if manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        return "incompatible"
    if manifest.get("fingerprint") != expected.fingerprint:
        return "incompatible"
    return "complete"


def load_benchmark_checkpoint(path: Path, expected: BenchmarkUnitSpec) -> BenchmarkCheckpoint:
    """Load a complete checkpoint only when its full fingerprint matches."""
    path = Path(path)
    state = checkpoint_state(path, expected)
    if state == "missing":
        raise FileNotFoundError(f"Benchmark checkpoint does not exist: {path}")
    if state == "incomplete":
        raise ValueError(f"Benchmark checkpoint is incomplete: {path}")
    if state == "incompatible":
        raise ValueError(f"Benchmark checkpoint fingerprint is incompatible: {path}")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    metadata_path = path / "metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    return BenchmarkCheckpoint(
        path=path,
        manifest=manifest,
        results=pd.read_csv(path / "results.csv"),
        metadata=metadata,
    )


def publish_benchmark_checkpoint(
    path: Path,
    spec: BenchmarkUnitSpec,
    results: pd.DataFrame,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Atomically publish one unit, or reuse an identical complete checkpoint."""
    path = Path(path)
    state = checkpoint_state(path, spec)
    if state == "complete":
        return path
    if state == "incompatible":
        raise ValueError(f"Benchmark checkpoint fingerprint differs: {path}")
    if state == "incomplete":
        raise ValueError(f"Benchmark checkpoint is incomplete; use a clean destination: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}.", dir=path.parent))
    try:
        results.to_csv(temporary / "results.csv", index=False)
        (temporary / "metadata.json").write_text(
            json.dumps(metadata or {}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "status": "complete",
            "fingerprint": spec.fingerprint,
            "unit": asdict(spec),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return path
