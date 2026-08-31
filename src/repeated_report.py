"""Hierarchical, transactional report for the repeated spatial benchmark."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from itertools import combinations
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from figures import close, save_figure, save_pdf_report
from palette import CATEGORICAL
from metrics.registry import get_metric_definition, primary_metric_names
from repeated_statistics import all_predeclared_contrasts, block_fwer_gate
from spatial_recovery import compare_method_detection_sets


def _markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_Sem linhas._\n"
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join("—" if pd.isna(value) else str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines) + "\n"


def _point_ids(value: Any) -> list[int]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if isinstance(value, (list, tuple, set)):
        return [int(point) for point in value]
    return []


def build_method_agreement(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare all significant consolidated point sets for paired realizations."""
    required = {
        "family", "layer", "scenario_id", "condition", "geometry_seed",
        "outcome_seed", "method_id",
    }
    if not required.issubset(results.columns):
        return pd.DataFrame(), pd.DataFrame()
    mask_column = (
        "all_detected_point_mask"
        if "all_detected_point_mask" in results.columns else None
    )
    point_column = (
        "all_detected_point_ids"
        if "all_detected_point_ids" in results.columns
        else "detected_point_ids"
        if "detected_point_ids" in results.columns
        else None
    )
    if point_column is None and mask_column is None:
        return pd.DataFrame(), pd.DataFrame()
    realization = [
        "family", "layer", "scenario_id", "condition",
        "geometry_seed", "outcome_seed",
    ]
    rows = []
    for keys, group in results.groupby(realization, sort=True, dropna=False):
        if mask_column is not None:
            method_masks = {
                str(row.method_id): int.from_bytes(
                    getattr(row, mask_column), "little"
                )
                for row in group.itertuples()
            }
            comparisons = []
            for first, second in combinations(sorted(method_masks), 2):
                first_mask, second_mask = method_masks[first], method_masks[second]
                intersection = (first_mask & second_mask).bit_count()
                first_n, second_n = first_mask.bit_count(), second_mask.bit_count()
                union = (first_mask | second_mask).bit_count()
                comparisons.append({
                    "first_method": first, "second_method": second,
                    "first_detected_n": first_n, "second_detected_n": second_n,
                    "intersection_n": intersection,
                    "first_only_n": first_n - intersection,
                    "second_only_n": second_n - intersection,
                    "union_n": union,
                    "point_jaccard": intersection / union if union else 1.0,
                })
            comparison = pd.DataFrame(comparisons)
        else:
            method_points = {
                str(row.method_id): _point_ids(getattr(row, point_column))
                for row in group.itertuples()
            }
            comparison = compare_method_detection_sets(method_points)
        if comparison.empty:
            continue
        for column, value in zip(realization, keys, strict=True):
            comparison[column] = value
        comparison["agreement_basis"] = "all_significant_consolidated"
        rows.append(comparison)
    raw = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    if raw.empty:
        return raw, pd.DataFrame()
    group = [
        "family", "layer", "scenario_id", "condition",
        "first_method", "second_method", "agreement_basis",
    ]
    aggregate = raw.groupby(group, sort=True, dropna=False).agg(
        mean_first_detected_n=("first_detected_n", "mean"),
        mean_second_detected_n=("second_detected_n", "mean"),
        mean_intersection_n=("intersection_n", "mean"),
        mean_first_only_n=("first_only_n", "mean"),
        mean_second_only_n=("second_only_n", "mean"),
        mean_union_n=("union_n", "mean"),
        mean_point_jaccard=("point_jaccard", "mean"),
        realizations=("point_jaccard", "size"),
    ).reset_index()
    return raw, aggregate


def build_repeated_tables(results: pd.DataFrame, *, n_bootstrap: int = 10000, seed: int = 43000) -> dict[str, pd.DataFrame]:
    results = results.copy()
    agreement_raw, method_agreement = build_method_agreement(results)
    results = results.drop(
        columns=["all_detected_point_mask"], errors="ignore"
    )
    if "evaluation_coverage" not in results and "coverage" in results:
        results["evaluation_coverage"] = results["coverage"]
    if "detected_coverage" not in results:
        results["detected_coverage"] = float("nan")
    if "confirmatory_method" not in results:
        primaries = set(primary_metric_names())

        def confirmatory(metric: object) -> bool:
            name = str(metric)
            try:
                definition = get_metric_definition(name)
            except ValueError:
                return True
            return name in primaries or definition.confirmatory_candidate

        results["confirmatory_method"] = results["metric"].map(confirmatory)
    fair = results[results["condition"].eq("fair")]
    gate_all = block_fwer_gate(fair, n_bootstrap=n_bootstrap, seed=seed)
    method_status = results[["method_id", "confirmatory_method"]].drop_duplicates()
    gate_all = gate_all.merge(method_status, on="method_id", how="left")
    gate = gate_all[gate_all["confirmatory_method"].eq(True)].reset_index(drop=True)
    unfair = results[~results["condition"].eq("fair")]
    group = [
        "family", "layer", "scenario_id", "condition", "expected_direction",
        "method_id", "metric", "confirmatory_method",
    ]
    aggregations: dict[str, tuple[str, Any]] = {
        "power": ("scenario_detected", "mean"),
        "correct_recovery": (
            "correct_recovery",
            lambda values: pd.to_numeric(values, errors="coerce").mean(),
        ),
        "precision": ("precision", "mean"),
        "recall": ("recall", "mean"),
        "f1": ("f1", "mean") if "f1" in unfair else ("precision", lambda values: float("nan")),
        "iou": ("iou", "mean"),
        "spatial_false_alarm": ("spatial_false_alarm", "mean"),
        "coverage": ("coverage", "mean"),
        "candidate_regions": ("candidate_regions", "mean"),
        "raw_significant_regions": ("raw_significant_regions", "mean"),
        "consolidated_regions": ("consolidated_regions", "mean"),
        "elapsed_seconds": ("elapsed_seconds", "mean") if "elapsed_seconds" in unfair else ("candidate_regions", lambda values: float("nan")),
        "evaluation_python_peak_bytes": ("evaluation_python_peak_bytes", "max") if "evaluation_python_peak_bytes" in unfair else ("candidate_regions", lambda values: float("nan")),
        "realizations": ("correct_recovery", "size"),
    }
    for column in (
        "true_positive_n", "false_positive_n", "false_negative_n",
        "true_negative_n", "target_coverage", "unassigned_target_n",
        "directional_precision", "directional_recall", "directional_f1",
        "directional_iou", "directional_recovered_n", "detected_coverage",
        "evaluation_coverage",
    ):
        if column in unfair:
            aggregations[column] = (column, "mean")
    for role in (
        "focal_target", "manipulated_context", "compensation", "null_background"
    ):
        for suffix in ("n", "total_n", "rate"):
            column = f"role_{role}_{suffix}"
            if column in unfair:
                aggregations[column] = (column, "mean")
    aggregate = unfair.groupby(group, dropna=False, sort=True).agg(**aggregations).reset_index()
    aggregate = aggregate.merge(
        gate_all[[
            "family", "method_id", "fwer", "upper_one_sided_95", "gate_passed"
        ]],
        on=["family", "method_id"], how="left",
    )
    confirmatory = aggregate[aggregate["confirmatory_method"].eq(True)]
    exploratory = aggregate[aggregate["confirmatory_method"].eq(False)]
    recovery = confirmatory[
        group + ["gate_passed", "power", "correct_recovery", "realizations"]
    ].copy()
    location = confirmatory[
        group + [
            "gate_passed", "precision", "recall", "f1", "iou",
            "directional_precision", "directional_recall", "directional_f1",
            "directional_iou", "spatial_false_alarm",
        ]
    ].copy()
    confusion_columns = [
        "true_positive_n", "false_positive_n", "false_negative_n",
        "true_negative_n", "precision", "recall", "f1", "iou",
        "directional_precision", "directional_recall", "directional_f1",
        "directional_iou", "target_coverage", "unassigned_target_n",
    ]
    point_confusion = aggregate[
        group + ["gate_passed"] + [
            column for column in confusion_columns if column in aggregate
        ]
    ].copy()
    role_frames = []
    for role in (
        "focal_target", "manipulated_context", "compensation", "null_background"
    ):
        detected = f"role_{role}_n"
        total = f"role_{role}_total_n"
        rate = f"role_{role}_rate"
        if {detected, total, rate}.issubset(aggregate.columns):
            role_frames.append(
                aggregate[group + ["gate_passed", detected, total, rate]]
                .rename(columns={
                    detected: "mean_detected_n",
                    total: "mean_role_total_n",
                    rate: "mean_detection_rate",
                })
                .assign(truth_role=role)
            )
    role_reach = (
        pd.concat(role_frames, ignore_index=True, sort=False)
        if role_frames else pd.DataFrame()
    )
    exploratory_candidates = exploratory[
        group + [
            "fwer", "upper_one_sided_95", "gate_passed", "power",
            "precision", "recall", "f1", "iou", "spatial_false_alarm",
            "realizations",
        ]
    ].copy()
    aggregate["scan_redundancy"] = aggregate["raw_significant_regions"] - aggregate["consolidated_regions"]
    operational = aggregate[group + [
        "coverage", "evaluation_coverage", "detected_coverage",
        "candidate_regions", "raw_significant_regions",
        "consolidated_regions", "scan_redundancy", "elapsed_seconds",
        "evaluation_python_peak_bytes",
    ]].copy()
    core = unfair[unfair["layer"].eq("core")]
    contrasts = all_predeclared_contrasts(core, gates=gate_all, n_bootstrap=n_bootstrap, seed=seed + n_bootstrap)
    gate_canonical = gate_all.assign(record_family="validity_gate")
    aggregate_canonical = aggregate.assign(record_family="performance")
    canonical = pd.concat([gate_canonical, aggregate_canonical], ignore_index=True, sort=False)
    sensitivities = aggregate[aggregate["layer"].eq("sensitivity")].reset_index(drop=True)
    stresses = aggregate[aggregate["layer"].eq("stress")].reset_index(drop=True)
    return {
        "validity_gate": gate,
        "recovery": recovery,
        "point_confusion": point_confusion,
        "method_agreement": method_agreement,
        "role_reach": role_reach,
        "exploratory_candidates": exploratory_candidates,
        "location": location,
        "operational": operational,
        "contrasts": contrasts,
        "sensitivities": sensitivities,
        "stresses": stresses,
        "canonical": canonical,
        "raw_results": results.sort_values(["family", "layer", "scenario_id", "geometry_seed", "outcome_seed", "method_id"], kind="stable").reset_index(drop=True),
        "method_agreement_raw": agreement_raw,
    }


_METHOD_LABELS = {
    "hdbscan_local_z": "HDBSCAN + local-z",
    "hdbscan_peer_rate_difference": "HDBSCAN + Δ taxa peers",
    "hdbscan_peer_log_rate_ratio": "HDBSCAN + log razão peers",
    "hdbscan_peer_gini_gap": "HDBSCAN + gap Gini peers",
    "hdbscan_sul": "HDBSCAN + SUL",
    "grid_sul": "Grade + SUL",
    "scan_sul": "Varredura + SUL",
}
_METHOD_ORDER = tuple(_METHOD_LABELS)
_FAMILY_LABELS = {
    "uniform": "Uniforme",
    "clustered": "Clusterizada",
    "realistic_irregular": "Irregular realista",
}


def _matrix(frame: pd.DataFrame, value: str) -> tuple[np.ndarray, list[str], list[str]]:
    methods = [method for method in _METHOD_ORDER if method in set(frame["method_id"])]
    methods.extend(sorted(set(frame["method_id"]) - set(methods)))
    families = [family for family in _FAMILY_LABELS if family in set(frame["family"])]
    families.extend(sorted(set(frame["family"]) - set(families)))
    grouped = frame.groupby(["method_id", "family"], dropna=False, sort=False)[value].mean()
    pivot = grouped.unstack("family").reindex(index=methods, columns=families)
    return (
        pivot.to_numpy(dtype=float),
        [_METHOD_LABELS.get(method, str(method)) for method in methods],
        [_FAMILY_LABELS.get(family, str(family)) for family in families],
    )


def _draw_matrix(
    ax,
    values: np.ndarray,
    methods: list[str],
    families: list[str],
    *,
    title: str,
    annotations: np.ndarray | None = None,
    categorical: bool = False,
    percent: bool = False,
) -> None:
    shown = values
    if categorical:
        shown = np.where(np.isnan(values), np.nan, values <= .01).astype(float)
        cmap = ListedColormap([
            CATEGORICAL[2], CATEGORICAL[0]
        ]).with_extremes(bad="#E5E7EB")
    else:
        cmap = plt.colormaps["Blues"].with_extremes(bad="#E5E7EB")
    ax.imshow(shown, aspect="auto", vmin=0, vmax=1, cmap=cmap)
    ax.set_xticks(range(len(families)), families)
    ax.set_yticks(range(len(methods)), methods)
    ax.set_title(title)
    labels = values if annotations is None else annotations
    for row in range(labels.shape[0]):
        for column in range(labels.shape[1]):
            value = labels[row, column]
            if not np.isnan(value):
                label = (
                    f"{'✓' if value <= .01 else '×'} {value:.3f}"
                    if categorical else f"{value * 100:.2f}%" if percent
                    else f"{value:.3f}"
                )
                color = "white" if shown[row, column] >= .55 else "#111827"
                ax.text(column, row, label, ha="center", va="center", fontsize=9, color=color)


def _figures(tables: dict[str, pd.DataFrame]):
    figures = []
    gate = tables["validity_gate"]
    gate_values, methods, families = _matrix(gate, "upper_one_sided_95")
    fig, ax = plt.subplots(figsize=(10, 5.625))
    _draw_matrix(
        ax, gate_values, methods, families,
        title="1. Gate de FWER — ✓ quando o limite superior é ≤ 0,01",
        categorical=True,
    )
    fig.tight_layout()
    figures.append(("01_gate_validade", fig))

    recovery = tables["recovery"]
    core = recovery[recovery["layer"].eq("core")]
    recovery_values, methods, families = _matrix(core, "correct_recovery")
    fig, ax = plt.subplots(figsize=(10, 5.625))
    _draw_matrix(
        ax, recovery_values, methods, families,
        title="2. Recuperação correta média — cenários injustos do núcleo",
        percent=True,
    )
    fig.tight_layout()
    figures.append(("02_recuperacao", fig))

    location = tables["location"]
    core = location[location["layer"].eq("core")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 7.875), sharey=True)
    for axis, metric, title in zip(
        axes,
        ("precision", "recall", "iou"),
        ("Precisão", "Recall", "IoU"),
        strict=True,
    ):
        values, methods, families = _matrix(core, metric)
        _draw_matrix(axis, values, methods, families, title=title, percent=True)
    fig.suptitle("3. Localização ponto a ponto — média do núcleo", y=.98)
    fig.tight_layout(rect=(0, 0, 1, .95))
    figures.append(("03_localizacao", fig))

    exploratory = tables["exploratory_candidates"]
    if not exploratory.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6.75), sharey=True)
        for axis, metric, title in zip(
            axes,
            ("power", "iou"),
            ("Detecção do cenário", "IoU com o alvo"),
            strict=True,
        ):
            values, methods, families = _matrix(exploratory, metric)
            _draw_matrix(axis, values, methods, families, title=title, percent=True)
        fig.suptitle(
            "4. Braço exploratório — heterogeneidade interna relativa",
            y=.98,
        )
        fig.tight_layout(rect=(0, 0, 1, .94))
        figures.append(("04_exploratorio_gini", fig))

    sensitivities = tables.get("sensitivities", pd.DataFrame())
    if not sensitivities.empty:
        effect = sensitivities[
            sensitivities["scenario_id"].astype(str).str.contains(
                r"sensitivity__uniform__local_positive__e(?:5|10|20|30)__s0\.02__circle__h0\.005",
                regex=True,
            )
            & sensitivities["confirmatory_method"].eq(True)
        ].copy()
        if not effect.empty:
            effect["effect_pp"] = effect["scenario_id"].map(
                lambda value: float(re.search(r"__e([0-9.]+)__", str(value)).group(1))
            )
            fig, axes = plt.subplots(1, 2, figsize=(12, 7.2), sharex=True)
            for method in _METHOD_ORDER:
                rows = effect[effect["method_id"].eq(method)].sort_values("effect_pp")
                if rows.empty:
                    continue
                label = _METHOD_LABELS.get(method, method)
                axes[0].plot(rows["effect_pp"], rows["f1"] * 100, marker="o", label=label)
                axes[1].plot(
                    rows["effect_pp"], rows["correct_recovery"] * 100,
                    marker="o", label=label,
                )
            for axis, title, ylabel in (
                (axes[0], "F1 ponto a ponto", "F1 médio (%)"),
                (axes[1], "Recuperação correta", "Realizações aprovadas (%)"),
            ):
                axis.set_title(title)
                axis.set_xlabel("Efeito plantado (p.p.)")
                axis.set_ylabel(ylabel)
                axis.set_xticks([5, 10, 20, 30])
                axis.grid(alpha=.3)
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(
                handles, labels, loc="upper center", bbox_to_anchor=(.5, .91),
                ncol=3, frameon=False,
            )
            fig.suptitle(
                "5. Sensibilidade ao efeito — alvo circular de 2% na geografia uniforme",
                y=.985,
            )
            fig.tight_layout(rect=(0, 0, 1, .76))
            figures.append(("05_sensibilidade_efeito", fig))

    confusion = tables.get("point_confusion", pd.DataFrame())
    strong_id = "sensitivity__uniform__local_positive__e30__s0.02__circle__h0.005"
    strong = confusion[confusion.get("scenario_id", pd.Series(dtype=str)).eq(strong_id)].copy()
    if not strong.empty:
        strong["method_order"] = strong["method_id"].map(
            {method: index for index, method in enumerate(_METHOD_ORDER)}
        ).fillna(99)
        strong = strong.sort_values("method_order")
        labels = [_METHOD_LABELS.get(method, method) for method in strong["method_id"]]
        y = np.arange(len(strong))
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.6))
        tp = strong["true_positive_n"].to_numpy(float)
        fn = strong["false_negative_n"].to_numpy(float)
        axes[0].barh(y, tp, color=CATEGORICAL[0], label="Alvo recuperado")
        axes[0].barh(y, fn, left=tp, color="#D1D5DB", label="Alvo não recuperado")
        for row, (correct, missed) in enumerate(zip(tp, fn, strict=True)):
            axes[0].text(correct / 2 if correct > 8 else correct + 2, row, f"{correct:.1f}", va="center", ha="center" if correct > 8 else "left", fontsize=9)
            axes[0].text(correct + missed - 2, row, f"{missed:.1f}", va="center", ha="right", fontsize=9)
        axes[0].set_yticks(y, labels)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("Média de pontos do alvo (total = 200)")
        axes[0].set_title("Quais pontos do alvo foram recuperados?")
        axes[0].legend(
            frameon=False, loc="upper center", bbox_to_anchor=(.5, -.12), ncol=2,
        )

        width = .24
        for offset, (column, label, color) in enumerate((
            ("precision", "Precisão", CATEGORICAL[0]),
            ("recall", "Parcela do alvo recuperada", CATEGORICAL[1]),
            ("f1", "Equilíbrio entre as duas", CATEGORICAL[2]),
        )):
            values = strong[column].to_numpy(float) * 100
            positions = y + (offset - 1) * width
            axes[1].barh(positions, values, height=width, color=color, label=label)
            for position, value in zip(positions, values, strict=True):
                axes[1].text(value + .7, position, f"{value:.1f}%", va="center", fontsize=8)
        axes[1].set_yticks(y, labels)
        axes[1].invert_yaxis()
        axes[1].set_xlabel("Percentual")
        axes[1].set_title("Qualidade da localização")
        axes[1].legend(
            frameon=False, loc="upper center", bbox_to_anchor=(.5, -.12), ncol=3,
        )
        axes[1].grid(axis="x", alpha=.3)
        fig.suptitle(
            "6. Cenário forte: taxa positiva de 50% para 80% dentro do alvo\n"
            "Alvo = região circular com 200 dos 10.000 pontos (2%)",
            y=.985,
        )
        fig.tight_layout(rect=(0, .08, 1, .94))
        figures.append(("06_gabarito_ponto_a_ponto", fig))
    return figures


def publish_repeated_report(
    results: pd.DataFrame,
    destination: Path,
    *,
    n_bootstrap: int = 10000,
    seed: int = 43000,
    plan_metadata: Mapping[str, Any] | None = None,
) -> Path:
    destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    figures = []
    try:
        tables = build_repeated_tables(results, n_bootstrap=n_bootstrap, seed=seed)
        table_dir = staging / "tables"; table_dir.mkdir()
        for name, frame in tables.items():
            frame.to_csv(table_dir / f"{name}.csv", index=False)
            (table_dir / f"{name}.md").write_text(_markdown(frame), encoding="utf-8")
        figure_dir = staging / "figures"; figure_dir.mkdir()
        pairs = _figures(tables); figures = [figure for _, figure in pairs]
        for name, figure in pairs:
            save_figure(figure, figure_dir / name)
        save_pdf_report(figures, figure_dir / "repeated_benchmark.pdf")
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "plan": dict(plan_metadata or {}),
            "hierarchy": [
                "validity_gate", "recovery", "point_confusion",
                "method_agreement", "role_reach", "exploratory_candidates",
                "location", "operational",
            ],
            "weighted_score": None,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        backup = destination.with_name(f".{destination.name}.backup")
        if backup.exists() and not destination.exists(): os.replace(backup, destination)
        elif backup.exists(): shutil.rmtree(backup)
        if destination.exists(): os.replace(destination, backup)
        try: os.replace(staging, destination)
        except BaseException:
            if backup.exists() and not destination.exists(): os.replace(backup, destination)
            raise
        if backup.exists(): shutil.rmtree(backup)
    except BaseException:
        if staging.exists(): shutil.rmtree(staging)
        raise
    finally:
        close(*figures)
    return destination
