"""Controlled synthetic effect sweep for local-z and SUL.

This standalone benchmark keeps one geography and one global positive rate while
changing which reference frame carries a planted contrast.  It is deliberately
small enough for an advisor presentation; it is a deterministic sensitivity
sweep, not a repeated power study.

Usage:
    uv run python src/synth_benchmark.py --out outputs/synth_benchmark_initial
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Final, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clustering.base import Partition
from clustering.hdbscan import fit_hdbscan_partition
from metrics.base import MetricContext
from metrics.neighbors import build_delaunay_adjacency
from metrics.registry import get_metric
from metrics.significance import significance_threshold, simulate_null_metric
from palette import CATEGORICAL, GRID, INK_PRIMARY, INK_SECONDARY, SURFACE
from synth_data import Blob, LOCAL_CONTROL


SCENARIOS: Final = ("fair", "global", "local", "both")
EFFECTS_PP: Final = (5, 10, 20, 30)
GLOBAL_RATE: Final = 0.5
BENCHMARK_FILLER_COUNT: Final = 40
BENCHMARK_BLOBS: Final = tuple(
    blob for blob in LOCAL_CONTROL.blobs if blob.role != "filler"
) + tuple(
    Blob(
        lat=40.0 + 1.2 * row,
        lon=-112.0 + 2.0 * column,
        n=LOCAL_CONTROL.blob_n,
        rate=GLOBAL_RATE,
        role="filler",
    )
    for row in range(5)
    for column in range(8)
)


def _role_rates(scenario: str, effect_pp: int) -> dict[str, float]:
    delta = effect_pp / 100.0
    rates = {
        "local_pocket": GLOBAL_RATE,
        "local_peer": GLOBAL_RATE,
        "global_pocket": GLOBAL_RATE,
        "global_peer": GLOBAL_RATE,
        "filler": GLOBAL_RATE,
    }
    if scenario in {"local", "both"}:
        rates["local_peer"] += delta
    if scenario in {"global", "both"}:
        rates["global_pocket"] -= delta
        rates["global_peer"] -= delta

    # Six local peers produce +6d blobs; the global pocket and four peers
    # produce -5d blobs. Many small filler blobs spread the exact balancing
    # adjustment so no single background cluster becomes a compensation extreme.
    excess_blobs = (6 if scenario in {"local", "both"} else 0) * delta
    deficit_blobs = (5 if scenario in {"global", "both"} else 0) * delta
    rates["filler"] -= (excess_blobs - deficit_blobs) / BENCHMARK_FILLER_COUNT
    return rates


def generate_effect_case(
    scenario: str,
    effect_pp: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Return one deterministic synthetic case with declared target masks.

    Coordinates depend only on ``seed`` and are therefore identical across the
    sweep. Outcomes use exact counts inside each blob, then are independently
    permuted. This makes the planted rates and the map-wide rate exact.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}, got {scenario!r}")
    allowed_effects = (0,) if scenario == "fair" else EFFECTS_PP
    if effect_pp not in allowed_effects:
        raise ValueError(
            f"effect_pp must be {allowed_effects} for scenario={scenario!r}, got {effect_pp}"
        )

    rates = _role_rates(scenario, effect_pp)
    coordinate_rng = np.random.default_rng(seed)
    outcome_offset = SCENARIOS.index(scenario) * 10_000 + effect_pp * 101
    outcome_rng = np.random.default_rng(seed + outcome_offset)
    frames: list[pd.DataFrame] = []

    for blob_id, blob in enumerate(BENCHMARK_BLOBS):
        lat = coordinate_rng.normal(blob.lat, LOCAL_CONTROL.blob_spread_deg, blob.n)
        lon = coordinate_rng.normal(blob.lon, LOCAL_CONTROL.blob_spread_deg, blob.n)
        positives = round(blob.n * rates[blob.role])
        labels = np.zeros(blob.n, dtype=int)
        labels[:positives] = 1
        frames.append(
            pd.DataFrame(
                {
                    "lat": lat,
                    "lon": lon,
                    "label": outcome_rng.permutation(labels),
                    "role": blob.role,
                    "blob_id": blob_id,
                    "target_local": blob.role == "local_pocket",
                    "target_global": blob.role == "global_pocket",
                }
            )
        )

    case = pd.concat(frames, ignore_index=True)
    if not np.isclose(case["label"].mean(), GLOBAL_RATE):
        raise RuntimeError("synthetic case failed to preserve the declared global rate")
    return case


def _target_row(
    *,
    scenario: str,
    effect_pp: int,
    target: str,
    target_mask: np.ndarray,
    metric: str,
    scores: np.ndarray,
    threshold: float,
    partition: Partition,
    types: np.ndarray,
) -> dict:
    intersections = np.asarray(
        [int(target_mask[np.asarray(region["points"], dtype=int)].sum()) for region in partition.regions]
    )
    region_idx = int(np.argmax(intersections))
    region = partition.regions[region_idx]
    points = np.asarray(region["points"], dtype=int)
    intersection_n = int(intersections[region_idx])
    target_n = int(target_mask.sum())
    cluster_n = int(len(points))
    union_n = target_n + cluster_n - intersection_n
    score = float(scores[region_idx])
    p = int(types[points].sum())
    rho_in = p / cluster_n
    rho_out = (int(types.sum()) - p) / (len(types) - cluster_n)
    direction_value = score if metric == "local_z" else rho_in - rho_out
    valid_scores = np.abs(scores[np.isfinite(scores)])
    target_detected = bool(abs(score) >= threshold)
    n_detected_map = int(np.count_nonzero(valid_scores >= threshold))
    partition_precision = intersection_n / cluster_n
    partition_recall = intersection_n / target_n
    partition_iou = intersection_n / union_n

    return {
        "scenario": scenario,
        "effect_pp": effect_pp,
        "target": target,
        "metric": metric,
        "cluster_label": int(region["cluster_label"]),
        "target_n": target_n,
        "cluster_n": cluster_n,
        "intersection_n": intersection_n,
        "partition_target_precision": partition_precision,
        "partition_target_recall": partition_recall,
        "partition_target_iou": partition_iou,
        "detected_target_precision": partition_precision if target_detected else 0.0,
        "detected_target_recall": partition_recall if target_detected else 0.0,
        "detected_target_iou": partition_iou if target_detected else 0.0,
        "score": score,
        "threshold": threshold,
        "evidence_ratio": abs(score) / threshold,
        "detected": target_detected,
        "direction": (
            "negative" if direction_value < 0 else "positive" if direction_value > 0 else "neutral"
        ),
        "n_detected_map": n_detected_map,
        "off_target_detections": n_detected_map - int(target_detected),
        "max_evidence_ratio_map": float(valid_scores.max() / threshold),
    }


def _fair_map_row(
    metric: str,
    scores: np.ndarray,
    threshold: float,
    partition: Partition,
) -> dict:
    finite = np.flatnonzero(np.isfinite(scores))
    max_idx = int(finite[np.argmax(np.abs(scores[finite]))])
    max_score = float(scores[max_idx])
    n_detected = int(np.count_nonzero(np.abs(scores[finite]) >= threshold))
    return {
        "scenario": "fair",
        "effect_pp": 0,
        "target": "mapa",
        "metric": metric,
        "cluster_label": int(partition.regions[max_idx]["cluster_label"]),
        "target_n": 0,
        "cluster_n": int(len(partition.regions[max_idx]["points"])),
        "intersection_n": 0,
        "partition_target_precision": float("nan"),
        "partition_target_recall": float("nan"),
        "partition_target_iou": float("nan"),
        "detected_target_precision": float("nan"),
        "detected_target_recall": float("nan"),
        "detected_target_iou": float("nan"),
        "score": max_score,
        "threshold": threshold,
        "evidence_ratio": abs(max_score) / threshold,
        "detected": bool(n_detected),
        "direction": "negative" if max_score < 0 else "positive" if max_score > 0 else "neutral",
        "n_detected_map": n_detected,
        "off_target_detections": float("nan"),
        "max_evidence_ratio_map": abs(max_score) / threshold,
    }


def run_effect_sweep(
    *,
    effects_pp: Iterable[int] = EFFECTS_PP,
    n_alt_worlds: int = 1000,
    signif_level: float = 0.005,
    seed: int = 42,
    min_cluster_frac: float = 0.01,
    min_samples: int = 25,
) -> tuple[pd.DataFrame, dict]:
    """Evaluate local-z and SUL over one fixed HDBSCAN partition.

    One fair geography calibrates both Monte Carlo rulers. Every alternative
    preserves that geography and the same global rate, so the partition and
    thresholds are reused rather than re-selected for each effect.
    """
    effects = tuple(int(value) for value in effects_pp)
    if len(set(effects)) != len(effects):
        raise ValueError("effects_pp must not contain duplicate values")
    invalid = sorted(set(effects) - set(EFFECTS_PP))
    if invalid:
        raise ValueError(f"effects_pp contains unsupported values: {invalid}")
    if n_alt_worlds <= 0:
        raise ValueError("n_alt_worlds must be positive")
    if not 0.0 < signif_level < 1.0:
        raise ValueError("signif_level must be between 0 and 1")
    min_worlds = math.ceil(1.0 / signif_level)
    if n_alt_worlds < min_worlds:
        raise ValueError(
            f"n_alt_worlds must be at least {min_worlds} for signif_level={signif_level}"
        )

    fair = generate_effect_case("fair", effect_pp=0, seed=seed)
    partition = fit_hdbscan_partition(
        fair,
        min_cluster_frac=min_cluster_frac,
        min_samples=min_samples,
    )
    if not partition.regions:
        raise RuntimeError("benchmark geometry produced no HDBSCAN clusters")
    adjacency = build_delaunay_adjacency(partition, fair)
    fair_types = fair["label"].to_numpy(dtype=int)
    fair_ctx = MetricContext(
        n_total=len(fair),
        p_total=int(fair_types.sum()),
        adjacency=adjacency,
        rng=np.random.default_rng(seed),
    )

    thresholds: dict[str, float] = {}
    for offset, metric in enumerate(("local_z", "sul")):
        null = simulate_null_metric(
            get_metric(metric),
            partition,
            fair_ctx,
            n_alt_worlds,
            len(fair),
            int(fair_types.sum()),
            seed=seed + 1_000 + offset,
        )
        thresholds[metric] = significance_threshold(signif_level, null)

    rows: list[dict] = []
    fair_results = {
        metric: get_metric(metric)(partition, fair_types, fair_ctx).per_cluster
        for metric in ("local_z", "sul")
    }
    for metric, scores in fair_results.items():
        rows.append(_fair_map_row(metric, scores, thresholds[metric], partition))

    for scenario in ("global", "local", "both"):
        targets = ("global",) if scenario == "global" else ("local",)
        if scenario == "both":
            targets = ("local", "global")
        for effect_pp in effects:
            case = generate_effect_case(scenario, effect_pp=effect_pp, seed=seed)
            types = case["label"].to_numpy(dtype=int)
            ctx = MetricContext(
                n_total=len(case),
                p_total=int(types.sum()),
                adjacency=adjacency,
                rng=np.random.default_rng(seed),
            )
            for metric in ("local_z", "sul"):
                scores = np.asarray(
                    get_metric(metric)(partition, types, ctx).per_cluster,
                    dtype=float,
                )
                for target in targets:
                    rows.append(
                        _target_row(
                            scenario=scenario,
                            effect_pp=effect_pp,
                            target=target,
                            target_mask=case[f"target_{target}"].to_numpy(dtype=bool),
                            metric=metric,
                            scores=scores,
                            threshold=thresholds[metric],
                            partition=partition,
                            types=types,
                        )
                    )

    assigned_n = sum(len(region["points"]) for region in partition.regions)
    metadata = {
        "benchmark_kind": "deterministic_effect_sweep",
        "n_total": len(fair),
        "global_rate": GLOBAL_RATE,
        "target_n": int(fair["target_local"].sum()),
        "n_clusters": len(partition.regions),
        "assigned_n": assigned_n,
        "coverage_rate": assigned_n / len(fair),
        "min_cluster_frac": min_cluster_frac,
        "min_samples": min_samples,
        "n_alt_worlds": n_alt_worlds,
        "mc_min_worlds": min_worlds,
        "mc_resolution": 1.0 / n_alt_worlds,
        "signif_level": signif_level,
        "seed": seed,
        "effects_pp": list(effects),
        "case_rates": {
            "fair_0": _role_rates("fair", 0),
            **{
                f"{scenario}_{effect}": _role_rates(scenario, effect)
                for scenario in ("global", "local", "both")
                for effect in effects
            },
        },
        "thresholds": thresholds,
    }
    return pd.DataFrame(rows), metadata


def _metric_label(metric: str) -> str:
    return "local-z" if metric == "local_z" else "SUL"


def _scenario_label(scenario: str) -> str:
    return {
        "fair": "Mapa justo",
        "global": "Bolsão global",
        "local": "Bolsão local",
        "both": "Bolsões simultâneos",
    }[scenario]


def _detection_cell(row: pd.Series) -> str:
    verdict = "detecta" if bool(row["detected"]) else "sem detecção"
    return (
        f"{verdict} ({float(row['evidence_ratio']):.2f}×; "
        f"mapa {int(row['n_detected_map'])}; "
        f"fora do alvo {int(row['off_target_detections'])})"
    )


def build_markdown_summary(results: pd.DataFrame, metadata: dict) -> str:
    """Build a presentation-ready summary without cross-scale raw scores."""
    fair = results[results["scenario"] == "fair"].set_index("metric")
    lines = [
        "# Benchmark sintético inicial — local-z × SUL",
        "",
        (
            f"> varredura determinística: uma geometria/seed, N = {metadata['n_total']:,}, "
            f"{metadata['n_clusters']} clusters, {metadata['n_alt_worlds']} mundos nulos e "
            f"α global = {metadata['signif_level']:.3f}. Não é ainda uma estimativa de poder."
        ).replace(",", "."),
        "",
        "## Sanity check do mapa justo",
        "",
        "| Cenário | local-z | SUL |",
        "|---|---:|---:|",
        (
            "| Mapa justo | "
            f"{int(fair.loc['local_z', 'n_detected_map'])} detecção(ões) · "
            f"máx. {float(fair.loc['local_z', 'max_evidence_ratio_map']):.2f}× | "
            f"{int(fair.loc['sul', 'n_detected_map'])} detecção(ões) · "
            f"máx. {float(fair.loc['sul', 'max_evidence_ratio_map']):.2f}× |"
        ),
        "",
        "## Recuperação dos bolsões plantados",
        "",
        "`×` é a razão de evidência: score em módulo ÷ limiar da própria métrica; ≥ 1 detecta.",
        "",
        "| Cenário | Alvo | Métrica | "
        + " | ".join(f"{effect} p.p." for effect in metadata["effects_pp"])
        + " |",
        "|---|---|---|" + "---:|" * len(metadata["effects_pp"]),
    ]
    alternatives = results[results["scenario"] != "fair"]
    for scenario in ("global", "local", "both"):
        scenario_rows = alternatives[alternatives["scenario"] == scenario]
        for target in ("local", "global"):
            target_rows = scenario_rows[scenario_rows["target"] == target]
            if target_rows.empty:
                continue
            for metric in ("local_z", "sul"):
                metric_rows = target_rows[target_rows["metric"] == metric].set_index("effect_pp")
                cells = [
                    _detection_cell(metric_rows.loc[effect])
                    for effect in metadata["effects_pp"]
                ]
                lines.append(
                    f"| {_scenario_label(scenario)} | {target} | {_metric_label(metric)} | "
                    + " | ".join(cells)
                    + " |"
                )
    lines += [
        "",
        "## Como ler",
        "",
        "- Bolsão local: a taxa do alvo coincide com a global, mas difere dos peers.",
        "- Bolsão global: alvo e peers se afastam juntos da taxa global.",
        "- Bolsões simultâneos: os dois mecanismos coexistem na mesma geografia.",
        "- `partition_target_*` mede o alinhamento do melhor cluster HDBSCAN com o alvo, "
        "mesmo sem detecção; não é desempenho do detector.",
        "- `detected_target_*` zera quando o cluster-alvo não cruza o limiar; ainda é uma "
        "leitura do alvo, não penaliza todas as detecções fora dele.",
        "- `n_detected_map` e `off_target_detections` expõem as outras regiões alteradas "
        "pelo contexto (peers e compensação distribuída). Este controle não estima "
        "precisão global nem falso alarme.",
    ]
    return "\n".join(lines) + "\n"


def _effect_summary(results: pd.DataFrame, scenario: str, target: str, metric: str) -> str:
    subset = results[
        (results["scenario"] == scenario)
        & (results["target"] == target)
        & (results["metric"] == metric)
    ].sort_values("effect_pp")
    detected = [str(int(value)) for value in subset.loc[subset["detected"], "effect_pp"]]
    missed = [str(int(value)) for value in subset.loc[~subset["detected"], "effect_pp"]]
    parts = []
    if detected:
        parts.append("detecta em " + ", ".join(detected) + " p.p.")
    if missed:
        parts.append("não detecta em " + ", ".join(missed) + " p.p.")
    return "; ".join(parts) if parts else "—"


def _fmt_int_pt(value: int | float) -> str:
    return f"{int(value):,}".replace(",", ".")


def build_comparison_slide(
    results: pd.DataFrame,
    metadata: dict,
    *,
    lar_run_dir: Path | None = None,
    authors_run_dir: Path | None = None,
) -> str:
    """Build the screenshot-ready evidence table from current artifacts."""
    fair = results[results["scenario"] == "fair"].set_index("metric")
    rows = [
        (
            "Mapa justo determinístico",
            f"local-z: {int(fair.loc['local_z', 'n_detected_map'])} detecções",
            f"SUL: {int(fair.loc['sul', 'n_detected_map'])} detecções",
            "Sanity check de uma execução; não estima falso alarme",
        ),
        (
            "Cluster-alvo local",
            "local-z — alvo local: " + _effect_summary(results, "local", "local", "local_z"),
            "SUL — alvo local: " + _effect_summary(results, "local", "local", "sul"),
            "O baseline de peers recupera o contraste local do alvo",
        ),
        (
            "Cluster-alvo global",
            "local-z — alvo global: " + _effect_summary(results, "global", "global", "local_z"),
            "SUL — alvo global: " + _effect_summary(results, "global", "global", "sul"),
            "O baseline global recupera o contraste acompanhado pelos peers",
        ),
        (
            "Dois alvos simultâneos",
            "local-z — alvo local: " + _effect_summary(results, "both", "local", "local_z"),
            "SUL — alvo global: " + _effect_summary(results, "both", "global", "sul"),
            "Cada métrica recupera o componente ligado à sua referência",
        ),
    ]

    detection_sets_path = (
        lar_run_dir / "exploration/comparisons/sul_vs_local_z/detection_sets.csv"
        if lar_run_dir is not None else None
    )
    if detection_sets_path is not None and detection_sets_path.exists():
        sets = pd.read_csv(detection_sets_path).set_index("set")["n_clusters"]
        both = int(sets.get("ambas", 0))
        only_local = int(sets.get("somente_local_z", 0))
        only_sul = int(sets.get("somente_sul", 0))
        neither = int(sets.get("nenhuma", 0))
        rows.append(
            (
                "LAR · mesmos clusters HDBSCAN",
                f"local-z: {both + only_local} detecções · {only_local} exclusivas",
                f"SUL: {both + only_sul} detecções · {only_sul} exclusivas",
                f"{both} em comum e {neither} por nenhuma; referências distintas",
            )
        )

    hdbscan_path = (
        authors_run_dir / "hdbscan_lar_comparison.csv"
        if authors_run_dir is not None else None
    )
    authors_path = (
        authors_run_dir / "unrestricted_lar_regions.csv"
        if authors_run_dir is not None else None
    )
    if (
        hdbscan_path is not None
        and authors_path is not None
        and hdbscan_path.exists()
        and authors_path.exists()
    ):
        hdbscan = pd.read_csv(hdbscan_path).iloc[-1]
        authors = pd.read_csv(authors_path)
        authors = authors[authors.get("method", "") == "kmeans_scan"].iloc[-1]
        rows.append(
            (
                "LAR · geometria sob a mesma SUL",
                (
                    f"HDBSCAN + SUL: {_fmt_int_pt(hdbscan['n_regions'])} regiões disjuntas, "
                    f"{_fmt_int_pt(hdbscan['significant_regions'])} significativas"
                ),
                (
                    f"Autores/KMeans + SUL: {_fmt_int_pt(authors['n_regions'])} candidatas, "
                    f"{_fmt_int_pt(authors['significant_regions'])} significativas → "
                    f"{_fmt_int_pt(authors['non_overlapping_regions'])} sem sobreposição"
                ),
                "Isola o efeito da partição; não demonstra vencedor",
            )
        )

    lines = [
        "# O que a comparação empírica já sustenta",
        "",
        "| Experimento | Leitura A | Leitura B | Conclusão permitida |",
        "|---|---|---|---|",
    ]
    lines.extend(f"| {experiment} | {local} | {sul} | {conclusion} |" for experiment, local, sul, conclusion in rows)
    lines += [
        "",
        (
            f"> Sintético inicial: N={_fmt_int_pt(metadata['n_total'])}, alvo n="
            f"{_fmt_int_pt(metadata['target_n'])}, {metadata['n_clusters']} clusters, "
            f"cobertura {metadata['coverage_rate']:.0%}, uma geometria/seed, "
            f"{metadata['n_alt_worlds']} mundos e α global={metadata['signif_level']:.3f}."
        ),
        "> O melhor cluster HDBSCAN coincide com o alvo nesta geometria separável; isso mede "
        "alinhamento da partição. As alterações de contexto também geram detecções no mapa, "
        "portanto ainda faltam repetições e um gabarito completo para poder, falso alarme e "
        "precisão global.",
    ]
    return "\n".join(lines) + "\n"


def benchmark_figure(results: pd.DataFrame, metadata: dict):
    """Two-panel slide figure for the pure local and pure global controls."""
    fig, axes = plt.subplots(1, 2, figsize=(13.333, 7.5), facecolor=SURFACE, sharey=True)
    panels = (("local", "local", "Cluster-alvo local"), ("global", "global", "Cluster-alvo global"))
    colors = {"local_z": CATEGORICAL[0], "sul": CATEGORICAL[2]}
    markers = {"local_z": "o", "sul": "s"}

    for ax, (scenario, target, title) in zip(axes, panels):
        subset = results[(results["scenario"] == scenario) & (results["target"] == target)]
        for metric in ("local_z", "sul"):
            series = subset[subset["metric"] == metric].sort_values("effect_pp")
            ax.plot(
                series["effect_pp"],
                series["evidence_ratio"],
                color=colors[metric],
                marker=markers[metric],
                linewidth=2.5,
                markersize=8,
                label=_metric_label(metric),
            )
        ax.axhline(1.0, color=INK_SECONDARY, linestyle="--", linewidth=1.8, label="limiar")
        ax.set_title(title, fontsize=17, color=INK_PRIMARY, pad=12)
        ax.set_xlabel("Diferença de taxa plantada (p.p.)", fontsize=13)
        ax.set_xticks(list(metadata["effects_pp"]))
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.tick_params(labelsize=12)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Razão de evidência (|score| ÷ limiar)", fontsize=13)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=3,
        frameon=False,
        fontsize=13,
    )
    fig.suptitle("As métricas respondem a referências diferentes", fontsize=22, y=0.98)
    fig.text(
        0.5,
        0.04,
        (
            f"Uma partição HDBSCAN fixa · N={metadata['n_total']:,} · "
            f"{metadata['n_alt_worlds']} mundos nulos · linha 1× = detecção"
        ).replace(",", "."),
        ha="center",
        fontsize=11,
        color=INK_SECONDARY,
    )
    fig.text(
        0.5,
        0.018,
        "Curvas do cluster-alvo; alterações de contexto geram outras detecções no mapa (ver tabela/CSV).",
        ha="center",
        fontsize=10,
        color=INK_SECONDARY,
    )
    fig.tight_layout(rect=(0.03, 0.085, 0.98, 0.81))
    return fig


def write_benchmark_artifacts(
    results: pd.DataFrame,
    metadata: dict,
    out_dir: Path,
    *,
    lar_run_dir: Path | None = None,
    authors_run_dir: Path | None = None,
) -> list[Path]:
    """Persist the evidence before rendering the optional slide figure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "benchmark_results.csv"
    markdown_path = out_dir / "benchmark_summary.md"
    metadata_path = out_dir / "benchmark_metadata.json"
    comparison_path = out_dir / "comparison_slide.md"
    results.to_csv(results_path, index=False)
    markdown_path.write_text(build_markdown_summary(results, metadata), encoding="utf-8")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    comparison_path.write_text(
        build_comparison_slide(
            results,
            metadata,
            lar_run_dir=lar_run_dir,
            authors_run_dir=authors_run_dir,
        ),
        encoding="utf-8",
    )

    figure = benchmark_figure(results, metadata)
    png_path = out_dir / "benchmark_effect_sweep.png"
    pdf_path = out_dir / "benchmark_effect_sweep.pdf"
    figure.savefig(png_path, dpi=180, bbox_inches="tight", facecolor=figure.get_facecolor())
    figure.savefig(pdf_path, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
    return [
        results_path,
        markdown_path,
        metadata_path,
        comparison_path,
        png_path,
        pdf_path,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("outputs/synth_benchmark_initial"))
    parser.add_argument("--effects-pp", default="5,10,20,30")
    parser.add_argument("--n-alt-worlds", type=int, default=1000)
    parser.add_argument("--signif-level", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-cluster-frac", type=float, default=0.01)
    parser.add_argument("--min-samples", type=int, default=25)
    parser.add_argument("--lar-run", type=Path, default=Path("outputs/lar_best_hdbscan"))
    parser.add_argument(
        "--authors-run",
        type=Path,
        default=Path("outputs/lar_unrestricted_authors_vs_hdbscan"),
    )
    args = parser.parse_args()

    effects = tuple(int(value.strip()) for value in args.effects_pp.split(",") if value.strip())
    results, metadata = run_effect_sweep(
        effects_pp=effects,
        n_alt_worlds=args.n_alt_worlds,
        signif_level=args.signif_level,
        seed=args.seed,
        min_cluster_frac=args.min_cluster_frac,
        min_samples=args.min_samples,
    )
    written = write_benchmark_artifacts(
        results,
        metadata,
        args.out,
        lar_run_dir=args.lar_run,
        authors_run_dir=args.authors_run,
    )
    print(f"Benchmark written to {args.out}")
    print(f"Partition: {metadata['n_clusters']} clusters, coverage {metadata['coverage_rate']:.1%}")
    for path in written:
        print(f"- {path.name}")


if __name__ == "__main__":
    main()
