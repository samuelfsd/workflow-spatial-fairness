"""Published-reference manifest and dataset-parity evidence for the benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


REFERENCE_SCHEMA_VERSION = 1
ALLOWED_UNITS = {
    "boolean",
    "count",
    "kilometers",
    "percentage_points",
    "rate",
    "score",
    "text",
    "threshold",
}
RESULT_FIELDS = {
    "id",
    "experiment",
    "dataset",
    "region_system",
    "metric",
    "quantity",
    "value",
    "unit",
    "source",
    "source_location",
    "precision",
    "observation",
}
DATASET_FIELDS = {
    "dataset",
    "public_filename",
    "sha256",
    "published_n",
    "published_global_rate",
    "observation",
}


def _require_fields(item: Mapping[str, Any], fields: set[str], *, label: str) -> None:
    missing = sorted(fields.difference(item))
    if missing:
        raise ValueError(f"{label} missing required fields: {missing}")


def load_reference_manifest(path: Path) -> dict[str, Any]:
    """Load and validate the machine-readable Sacharidis reference manifest."""
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != REFERENCE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported reference schema: expected {REFERENCE_SCHEMA_VERSION}, "
            f"got {manifest.get('schema_version')}"
        )
    sources = manifest.get("sources")
    if not isinstance(sources, dict) or "paper" not in sources or "repository" not in sources:
        raise ValueError("Reference manifest must declare paper and repository sources")
    if not sources["repository"].get("commit"):
        raise ValueError("Reference repository source must declare a commit")

    datasets = manifest.get("datasets")
    results = manifest.get("results")
    if not isinstance(datasets, list) or not isinstance(results, list):
        raise ValueError("Reference manifest datasets and results must be lists")

    dataset_names: set[str] = set()
    for idx, item in enumerate(datasets):
        if not isinstance(item, dict):
            raise ValueError(f"dataset[{idx}] must be an object")
        _require_fields(item, DATASET_FIELDS, label=f"dataset[{idx}]")
        name = str(item["dataset"])
        if name in dataset_names:
            raise ValueError(f"duplicate dataset reference: {name}")
        dataset_names.add(name)

    result_ids: set[str] = set()
    for idx, item in enumerate(results):
        if not isinstance(item, dict):
            raise ValueError(f"result[{idx}] must be an object")
        _require_fields(item, RESULT_FIELDS, label=f"result[{idx}]")
        result_id = str(item["id"])
        if result_id in result_ids:
            raise ValueError(f"duplicate result id: {result_id}")
        result_ids.add(result_id)
        if item["unit"] not in ALLOWED_UNITS:
            raise ValueError(f"Unknown reference unit: {item['unit']}")
        if item["source"] not in sources:
            raise ValueError(f"Unknown result source: {item['source']}")
    return manifest


def observed_dataset_record(dataset: Any) -> dict[str, Any]:
    """Return the parity fields needed from a ``LoadedDataset``."""
    return {
        "source_sha256": dataset.source_sha256,
        "n_total": int(dataset.n_total),
        "p_total": int(dataset.p_total),
        "global_rate": float(dataset.global_rate),
    }


def build_data_parity_table(
    manifest: Mapping[str, Any],
    observed: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    """Compare local artifacts to public hashes and separately to paper summaries."""
    rows: list[dict[str, Any]] = []
    for reference in manifest["datasets"]:
        name = str(reference["dataset"])
        if name not in observed:
            raise ValueError(f"Missing observed dataset evidence for {name}")
        local = observed[name]
        local_n = int(local["n_total"])
        local_rate = float(local["global_rate"])
        published_n = reference["published_n"]
        published_rate = reference["published_global_rate"]
        rows.append(
            {
                "dataset": name,
                "public_filename": reference["public_filename"],
                "expected_sha256": reference["sha256"],
                "local_sha256": local["source_sha256"],
                "public_artifact_identical": local["source_sha256"] == reference["sha256"],
                "published_n": published_n,
                "local_n": local_n,
                "local_positives": int(local["p_total"]),
                "paper_n_matches": published_n is not None and int(published_n) == local_n,
                "published_global_rate": published_rate,
                "local_global_rate": local_rate,
                "paper_global_rate_matches": (
                    published_rate is not None
                    and abs(float(published_rate) - local_rate) <= 1e-12
                ),
                "observation": reference["observation"],
            }
        )
    return pd.DataFrame(rows).sort_values("dataset", kind="stable").reset_index(drop=True)


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if pd.isna(value):
                values.append("—")
            elif isinstance(value, bool):
                values.append("sim" if value else "não")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def write_data_parity_report(frame: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Publish CSV and Markdown parity evidence."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "dataset_parity.csv"
    markdown_path = output_dir / "dataset_parity.md"
    frame.to_csv(csv_path, index=False)
    markdown_path.write_text(_markdown_table(frame), encoding="utf-8")
    return [csv_path, markdown_path]

