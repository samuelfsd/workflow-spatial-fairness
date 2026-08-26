"""Canonical tables and transactional publication for the initial benchmark."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from benchmark_checkpoint import BenchmarkUnitSpec, CHECKPOINT_SCHEMA_VERSION
from metrics.registry import candidate_metric_names


QUANTITIES: dict[str, str] = {
    "N": "count",
    "P": "count",
    "global_rate": "rate",
    "coverage": "rate",
    "noise_n": "count",
    "n_regions": "count",
    "candidate_regions": "count",
    "significant_regions": "count",
    "consolidated_regions": "count",
    "partition_score": "score",
    "best_region_n": "count",
    "best_region_p": "count",
    "best_region_rate": "rate",
    "best_reference_rate": "rate",
    "best_contrast_pp": "percentage_points",
    "score": "score",
    "threshold": "threshold",
    "evidence_ratio": "score",
    "best_region_id": "text",
    "best_direction": "text",
    "best_indicator_name": "text",
    "best_indicator_value": "indicator",
    "best_reference_value": "indicator",
    "best_effect_value": "effect",
    "best_effect_unit": "text",
}
KEYS = ["dataset", "experiment", "region_system", "metric", "quantity"]


def _markdown(frame: pd.DataFrame) -> str:
    if not len(frame.columns):
        return "_Sem linhas._\n"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        rendered = []
        for value in row:
            if pd.isna(value):
                rendered.append("—")
            elif isinstance(value, (bool, np.bool_)):
                rendered.append("sim" if value else "não")
            else:
                rendered.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines) + "\n"


def load_checkpoint_results(root: Path) -> pd.DataFrame:
    """Read complete checkpoint results without recomputing scientific work."""
    frames: list[pd.DataFrame] = []
    for manifest_path in sorted(Path(root).rglob("manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        result_path = manifest_path.parent / "results.csv"
        if manifest.get("status") != "complete" or not result_path.exists():
            continue
        if manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION or "unit" not in manifest:
            continue
        try:
            spec = BenchmarkUnitSpec(**manifest["unit"])
        except (TypeError, ValueError):
            continue
        if manifest.get("fingerprint") != spec.fingerprint:
            raise ValueError(f"fingerprint inválido no checkpoint: {manifest_path.parent}")
        frame = pd.read_csv(result_path)
        frame["checkpoint_fingerprint"] = manifest.get("fingerprint")
        frame["checkpoint_unit"] = json.dumps(manifest["unit"], sort_keys=True)
        frame["checkpoint_path"] = str(manifest_path.parent)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def reject_incompatible_checkpoint_merges(frame: pd.DataFrame) -> None:
    """Refuse multiple fingerprints for the same declared scientific unit."""
    if frame.empty:
        return
    keys = ["dataset", "protocol", "partitioning", "metric"]
    available = [key for key in keys if key in frame]
    if len(available) != len(keys):
        return
    summaries = frame[frame.get("record_type", "").eq("summary")]
    conflicts = summaries.groupby(keys, dropna=False)["checkpoint_fingerprint"].nunique()
    conflicts = conflicts[conflicts > 1]
    if len(conflicts):
        raise ValueError(
            "checkpoints incompatíveis para a mesma unidade; use saídas separadas: "
            f"{conflicts.index.tolist()}"
        )


def _params(row: pd.Series) -> dict[str, Any]:
    value = row.get("params")
    if not isinstance(value, str) or not value:
        return {}
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _identity(row: pd.Series) -> tuple[str, str]:
    partitioning = str(row.get("partitioning", ""))
    params = _params(row)
    if (
        str(row.get("dataset", "")).startswith("synth")
        or str(row.get("dataset", "")) == "semisynth"
    ) and row.get("protocol") == "reproduction" and partitioning.startswith("grid_"):
        return "synthetic_control", "authors_regions"
    if partitioning == "random_grids":
        return "synthetic_control", "random_grids"
    if partitioning.startswith("grid_"):
        return "fixed_grid", partitioning
    if partitioning.startswith("kmeans_square_scan"):
        direction = str(params.get("direction", "both"))
        return ("unrestricted_scan" if direction == "both" else "directional_scan"), "kmeans_square_scan"
    if str(row.get("method", "")) == "hdbscan" or partitioning.startswith("hdbscan"):
        fraction = params.get("min_cluster_frac")
        region_system = f"hdbscan_frac_{float(fraction):g}" if fraction is not None else "hdbscan"
        return "standardized_comparison", region_system
    return str(row.get("protocol", "local")), partitioning


def checkpoint_summaries_to_long(summaries: pd.DataFrame) -> pd.DataFrame:
    """Normalize local summary records to the same quantity/unit vocabulary as references."""
    rows: list[dict[str, Any]] = []
    if summaries.empty:
        return pd.DataFrame()
    source_rows = summaries[summaries.get("record_type", "summary") == "summary"]
    for _, summary in source_rows.iterrows():
        experiment, region_system = _identity(summary)
        metric = str(summary.get("metric", ""))
        params = _params(summary)
        if experiment == "directional_scan":
            direction = params.get("direction")
            metric = "sul_less_inside" if direction == "less_in" else "sul_less_outside"
        for field, unit in QUANTITIES.items():
            if field not in summary.index:
                continue
            value = summary.get(field)
            quantity = {"score": "best_region_score", "threshold": "significance_threshold"}.get(field, field)
            reason = None
            if pd.isna(value):
                if metric == "meanvar" and quantity in {
                    "significant_regions", "consolidated_regions", "significance_threshold"
                }:
                    reason = "não aplicável à métrica diagnóstica"
                else:
                    reason = "não reportado ou não aplicável"
            rows.append({
                "source": "local",
                "protocol": summary.get("protocol"),
                "dataset": summary.get("dataset"),
                "experiment": experiment,
                "method": summary.get("method"),
                "params": summary.get("params"),
                "region_system": region_system,
                "metric": metric,
                "quantity": quantity,
                "value": value,
                "unit": unit,
                "null_reason": reason,
                "rate_semantics": summary.get("rate_semantics"),
                "seed": summary.get("seed"),
                "n_alt_worlds": summary.get("n_alt_worlds"),
                "checkpoint_fingerprint": summary.get("checkpoint_fingerprint"),
            })
        calibrated_metrics = {
            "sul", "local_z", "sul_less_inside", "sul_less_outside",
            *candidate_metric_names(),
        }
        if metric in calibrated_metrics and "significant_regions" in summary.index:
            significant = summary.get("significant_regions")
            rows.append({
                "source": "local", "protocol": summary.get("protocol"),
                "dataset": summary.get("dataset"), "experiment": experiment,
                "method": summary.get("method"), "region_system": region_system,
                "metric": metric, "quantity": "unfairness_detected",
                "value": bool(significant > 0) if pd.notna(significant) else pd.NA,
                "unit": "boolean", "null_reason": None if pd.notna(significant) else "não reportado",
                "rate_semantics": summary.get("rate_semantics"), "seed": summary.get("seed"),
                "n_alt_worlds": summary.get("n_alt_worlds"),
                "checkpoint_fingerprint": summary.get("checkpoint_fingerprint"),
            })
            dataset_name = str(summary.get("dataset"))
            if dataset_name in {"semisynth", "synth_unfair", "synth_fair"}:
                expected = dataset_name == "synth_unfair"
                detected = bool(significant > 0) if pd.notna(significant) else pd.NA
                common = {
                    "source": "local", "protocol": summary.get("protocol"),
                    "dataset": dataset_name, "experiment": experiment,
                    "method": summary.get("method"), "region_system": region_system,
                    "params": summary.get("params"),
                    "metric": metric, "rate_semantics": summary.get("rate_semantics"),
                    "seed": summary.get("seed"), "n_alt_worlds": summary.get("n_alt_worlds"),
                    "checkpoint_fingerprint": summary.get("checkpoint_fingerprint"),
                }
                rows.append({**common, "quantity": "expected_unfairness", "value": expected, "unit": "boolean", "null_reason": None})
                rows.append({**common, "quantity": "verdict_correct", "value": detected == expected if pd.notna(detected) else pd.NA, "unit": "boolean", "null_reason": None if pd.notna(detected) else "sem veredito"})
    columns = ["source", "protocol", "dataset", "experiment", "method", "params", "region_system", "metric", "quantity", "value", "unit", "null_reason", "rate_semantics", "seed", "n_alt_worlds", "checkpoint_fingerprint"]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["dataset", "experiment", "region_system", "metric", "quantity", "protocol"],
        kind="stable", na_position="last",
    ).reset_index(drop=True)


def reference_results_frame(manifest: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for item in manifest["results"]:
        row = dict(item)
        row.update({"protocol": "published", "method": "reported", "null_reason": None})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(KEYS + ["source"], kind="stable").reset_index(drop=True)


def compare_compatible_results(reference: pd.DataFrame, local: pd.DataFrame) -> pd.DataFrame:
    """Join compatible quantities; reject a silent rate/percentage-point mismatch."""
    if reference.empty or local.empty:
        return pd.DataFrame()
    ref = reference.copy().rename(columns={"value": "reference_value", "unit": "reference_unit", "source": "reference_source"})
    loc = local.copy().rename(columns={"value": "local_value", "unit": "local_unit", "source": "local_source"})
    joined = ref.merge(loc, on=KEYS, how="inner", suffixes=("_reference", "_local"))
    if not joined.empty and (joined["reference_unit"] != joined["local_unit"]).any():
        bad = joined.loc[joined["reference_unit"] != joined["local_unit"], KEYS + ["reference_unit", "local_unit"]]
        raise ValueError(f"unidades incompatíveis na comparação: {bad.to_dict('records')}")
    if joined.empty:
        return joined
    numeric_ref = pd.to_numeric(joined["reference_value"], errors="coerce")
    numeric_local = pd.to_numeric(joined["local_value"], errors="coerce")
    comparable = numeric_ref.notna() & numeric_local.notna() & ~joined["reference_unit"].eq("boolean")
    joined["absolute_difference"] = np.where(comparable, numeric_local - numeric_ref, np.nan)
    joined["relative_difference"] = np.where(comparable & numeric_ref.ne(0), (numeric_local - numeric_ref) / numeric_ref.abs(), np.nan)
    boolean = joined["reference_unit"].eq("boolean")
    joined["verdict_agreement"] = pd.Series(pd.NA, index=joined.index, dtype="boolean")
    joined.loc[boolean, "verdict_agreement"] = (
        joined.loc[boolean, "reference_value"].astype(bool).to_numpy()
        == joined.loc[boolean, "local_value"].astype(bool).to_numpy()
    )
    joined["direction_agreement"] = pd.Series(pd.NA, index=joined.index, dtype="boolean")
    direction = joined["quantity"].eq("best_direction")
    joined.loc[direction, "direction_agreement"] = (
        joined.loc[direction, "reference_value"].astype(str).to_numpy()
        == joined.loc[direction, "local_value"].astype(str).to_numpy()
    )
    joined["location_agreement"] = pd.Series(pd.NA, index=joined.index, dtype="boolean")
    location = joined["quantity"].eq("best_location")
    joined.loc[location, "location_agreement"] = (
        joined.loc[location, "reference_value"].astype(str).str.casefold().to_numpy()
        == joined.loc[location, "local_value"].astype(str).str.casefold().to_numpy()
    )
    return joined.sort_values(KEYS + ["protocol_local"], kind="stable").reset_index(drop=True)


def build_canonical_tables(canonical: pd.DataFrame, comparisons: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split the canonical long table into stable, semantically distinct panels."""
    def select(mask: pd.Series) -> pd.DataFrame:
        return canonical.loc[mask].sort_values(
            [column for column in ["dataset", "source", "protocol", "region_system", "metric", "quantity"] if column in canonical],
            kind="stable",
        ).reset_index(drop=True)

    dataset = canonical.get("dataset", pd.Series(index=canonical.index, dtype=str))
    experiment = canonical.get("experiment", pd.Series(index=canonical.index, dtype=str))
    synthetic = dataset.isin(["semisynth", "synth_unfair"])
    synthetic_diagnostic = select(synthetic & canonical.get("metric", pd.Series(index=canonical.index, dtype=str)).eq("meanvar"))
    synthetic_diagnostic["ordering_contradicts_design"] = pd.NA
    if "protocol" not in synthetic_diagnostic:
        synthetic_diagnostic["protocol"] = None
    scores = synthetic_diagnostic[synthetic_diagnostic["quantity"].eq("partition_score")]
    for (source, protocol), group in scores.groupby(["source", "protocol"], dropna=False):
        by_dataset = group.groupby("dataset")["value"].first()
        if {"semisynth", "synth_unfair"}.issubset(by_dataset.index):
            contradiction = float(by_dataset["semisynth"]) > float(by_dataset["synth_unfair"])
            mask = synthetic_diagnostic["source"].eq(source) & synthetic_diagnostic["protocol"].eq(protocol)
            synthetic_diagnostic.loc[mask, "ordering_contradicts_design"] = contradiction
    return {
        "canonical": canonical.reset_index(drop=True),
        "comparisons": comparisons.reset_index(drop=True),
        "synthetic_diagnostic": synthetic_diagnostic,
        "synthetic_location": select(synthetic & ~canonical.get("metric", pd.Series(index=canonical.index, dtype=str)).eq("meanvar")),
        "synthetic_auxiliary": select(dataset.eq("synth_fair")),
        "lar_grid_hdbscan": select(dataset.eq("lar") & experiment.isin(["fixed_grid", "standardized_comparison"])),
        "lar_scan": select(dataset.eq("lar") & experiment.eq("unrestricted_scan")),
        "lar_directional": select(dataset.eq("lar") & experiment.eq("directional_scan")),
        "crime": select(dataset.eq("crime")),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish(staging: Path, destination: Path) -> None:
    backup = destination.with_name(f".{destination.name}.backup")
    if backup.exists() and not destination.exists():
        os.replace(backup, destination)
    elif backup.exists():
        shutil.rmtree(backup)
    if destination.exists():
        os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except BaseException:
        if backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def publish_initial_report(
    destination: Path,
    *,
    canonical: pd.DataFrame,
    comparisons: pd.DataFrame,
    parity: pd.DataFrame,
    regions: pd.DataFrame | None = None,
    render_figures: bool = True,
    render_maps: bool = True,
    run_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Publish all renderers atomically; no scientific calculation occurs here."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    figures = []
    try:
        tables = build_canonical_tables(canonical, comparisons)
        tables["dataset_parity"] = parity
        table_dir = staging / "tables"
        table_dir.mkdir()
        for name, frame in tables.items():
            frame.to_csv(table_dir / f"{name}.csv", index=False)
            (table_dir / f"{name}.md").write_text(_markdown(frame), encoding="utf-8")
        if regions is not None:
            regions.to_csv(table_dir / "regions.csv", index=False)
        if render_figures:
            from benchmark_figures import render_benchmark_figures
            figures = render_benchmark_figures(tables, staging / "figures")
        advisor_route = Path("docs/ROTEIRO_ORIENTADORES.md")
        if advisor_route.exists():
            shutil.copyfile(advisor_route, staging / "RESUMO_ORIENTADORES.md")
        if render_maps:
            from benchmark_maps import render_comparative_maps
            render_comparative_maps(tables["canonical"], regions if regions is not None else pd.DataFrame(), staging / "maps")
        artifacts = []
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                artifacts.append({"path": str(path.relative_to(staging)), "sha256": _file_sha256(path)})
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "datasets": sorted(canonical.get("dataset", pd.Series(dtype=str)).dropna().unique().tolist()),
            "seeds": sorted(pd.to_numeric(canonical.get("seed", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).unique().tolist()),
            "null_worlds": sorted(pd.to_numeric(canonical.get("n_alt_worlds", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).unique().tolist()),
            "checkpoint_fingerprints": sorted(canonical.get("checkpoint_fingerprint", pd.Series(dtype=str)).dropna().unique().tolist()),
            "artifacts": artifacts,
            "run": dict(run_metadata or {}),
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _publish(staging, destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    finally:
        for figure in figures:
            try:
                import matplotlib.pyplot as plt
                plt.close(figure)
            except Exception:
                pass
    return destination
