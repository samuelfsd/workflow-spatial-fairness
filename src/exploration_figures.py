"""Matplotlib figures for the single-partition cluster exploration report."""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from exploration import ExplorationTables
from figure_style import apply_presentation_style
from palette import (
    CATEGORICAL,
    COLOR_NOT_EVALUATED,
    DETECTION_COLORS,
    DETECTION_LABELS,
    GRID,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SURFACE,
)


DETECTION_MARKERS = {
    "negative": "v",
    "positive": "^",
    "neutral": "o",
    None: "x",
}


def _figure(nrows: int = 1, ncols: int = 1, **kwargs):
    fig, axes = plt.subplots(
        nrows, ncols, figsize=kwargs.pop("figsize", (13.333, 7.5)),
        facecolor=SURFACE, **kwargs,
    )
    if isinstance(axes, np.ndarray):
        iterable = axes.flat
    else:
        iterable = [axes]
    for ax in iterable:
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.8)
        ax.tick_params(colors=INK_SECONDARY)
    return fig, axes


def _footer(fig: Figure, text: str) -> None:
    fig.text(0.01, 0.012, text, color=INK_MUTED, ha="left")


def _detected(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["detection_class"].isin(["negative", "positive"])]


def _not_evaluated_reason(row: pd.Series) -> str:
    reason = row.get("evaluation_reason")
    return "motivo não registrado" if pd.isna(reason) or not reason else str(reason)


def cover_figure(
    tables: ExplorationTables, manifest: dict, primary_metric: str
) -> Figure:
    fig = plt.figure(figsize=(13.333, 7.5), facecolor=SURFACE)
    ax = fig.add_axes((0.07, 0.08, 0.86, 0.84))
    ax.axis("off")
    dataset = manifest["dataset"]
    outcome = dataset["outcome"]
    threshold_row = tables.cluster_features.iloc[0] if len(tables.cluster_features) else None
    threshold = threshold_row["signif_threshold"] if threshold_row is not None else float("nan")
    reference = threshold_row["primary_reference"] if threshold_row is not None else "—"
    n_worlds = int(threshold_row["mc_worlds"]) if threshold_row is not None else "—"
    signif = threshold_row["signif_level"] if threshold_row is not None else "—"
    lines = [
        "Exploração dos clusters e análise de injustiça",
        f"Dataset: {dataset['name']} · uma única partição",
        "",
        f"Outcome positivo: {outcome['positive_label']}",
        f"Outcome negativo: {outcome['negative_label']}",
        f"Maior taxa: {outcome['desirability']}",
        "",
        f"Métrica primária: {primary_metric} · referência: {reference}",
        f"Limiar Monte Carlo: {threshold:.4g} · mundos: {n_worlds} · α={signif}",
        "",
        "A detection class descreve evidência estatística significativa.",
        "“Nada detectado” não significa “justo”; “não avaliado” é um status separado.",
    ]
    ax.text(0.0, 1.0, lines[0], va="top", fontsize=25, color=INK_PRIMARY, weight="bold")
    ax.text(0.0, 0.89, "\n".join(lines[1:]), va="top", fontsize=14,
            color=INK_SECONDARY, linespacing=1.6)
    _footer(fig, "Relatório exploratório; não compara métodos de agrupamento.")
    return fig


def coverage_figure(tables: ExplorationTables) -> Figure:
    frame = tables.coverage_audit[
        tables.coverage_audit["scope"].isin(["assigned", "unassigned"])
    ]
    fig, axes = _figure(1, 2)
    ax, rate_ax = axes
    x = np.arange(len(frame))
    ax.bar(x, frame["p"], color=CATEGORICAL[0], label="positivos")
    ax.bar(x, frame["n_neg"], bottom=frame["p"], color=CATEGORICAL[2], label="negativos")
    ax.set_xticks(x, ["atribuídos", "não atribuídos"])
    ax.set_ylabel("pontos")
    ax.set_title("Cobertura e composição", loc="left", color=INK_PRIMARY)
    ax.legend(frameon=False)
    rate_ax.bar(x, frame["rho"], color=CATEGORICAL[1])
    rate_ax.set_ylim(0, 1)
    rate_ax.set_xticks(x, ["atribuídos", "não atribuídos"])
    rate_ax.set_ylabel("taxa de positivos")
    rate_ax.set_title("Perfil do que ficou dentro e fora", loc="left", color=INK_PRIMARY)
    for idx, row in frame.reset_index(drop=True).iterrows():
        rate_ax.text(idx, row["rho"] + 0.03, f"{row['rho']:.1%}\nn={int(row['n'])}", ha="center")
    _footer(fig, "Não atribuídos permanecem fora dos clusters e das métricas.")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def size_distribution_figure(tables: ExplorationTables) -> Figure:
    frame = tables.cluster_features
    summary = tables.distribution_summary.set_index("metric").loc["n"]
    fig, ax = _figure()
    bins = min(max(5, int(np.sqrt(max(len(frame), 1)))), 30)
    ax.hist(frame["n"], bins=bins, color=CATEGORICAL[0], alpha=0.78, edgecolor=SURFACE)
    for key, label, style in (
        ("mean", "média", "-"), ("median", "mediana", "--"),
        ("q1", "Q1", ":"), ("q3", "Q3", ":"),
    ):
        ax.axvline(summary[key], color=INK_PRIMARY, linestyle=style, linewidth=1.4, label=label)
    for _, row in _detected(frame).iterrows():
        detection = row["detection_class"]
        ax.scatter(
            row["n"], 0, marker=DETECTION_MARKERS[detection], s=80,
            facecolors=DETECTION_COLORS[detection], edgecolors=INK_PRIMARY, zorder=5,
        )
        ax.annotate(
            str(int(row["cluster_label"])),
            (row["n"], 0),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
        )
    not_evaluated = frame[frame["evaluation_status"] != "avaliado"]
    for index, (_, row) in enumerate(not_evaluated.iterrows()):
        ax.scatter(
            row["n"], 0, marker="x", s=90, color=COLOR_NOT_EVALUATED,
            linewidths=1.8, zorder=6,
            label=("não avaliado — motivo junto ao rótulo" if index == 0 else None),
        )
        ax.annotate(
            f"{int(row['cluster_label'])}: {_not_evaluated_reason(row)}",
            (row["n"], 0), xytext=(5, 15), textcoords="offset points",
            ha="left", color=COLOR_NOT_EVALUATED,
        )
    ax.set_xlabel("tamanho do cluster (pontos)")
    ax.set_ylabel("quantidade de clusters")
    ax.set_title("Distribuição dos tamanhos", loc="left", color=INK_PRIMARY)
    ax.legend(frameon=False, ncols=4)
    _footer(fig, "Linhas: média, mediana e quartis. Marcas na base: detecções e não avaliados.")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def balance_ranking_figure(tables: ExplorationTables) -> Figure:
    frame = tables.cluster_features.sort_values("n", ascending=True).reset_index(drop=True)
    fig, axes = _figure(1, 2)
    count_ax, rate_ax = axes
    y = np.arange(len(frame))
    count_ax.barh(y, frame["p"], color=CATEGORICAL[0], label="positivos")
    count_ax.barh(y, frame["n_neg"], left=frame["p"], color=CATEGORICAL[2], label="negativos")
    count_ax.set_yticks(y, frame["cluster_label"].astype(str))
    count_ax.set_xlabel("pontos")
    count_ax.set_ylabel("cluster")
    count_ax.set_title("Ranking de tamanho e composição", loc="left", color=INK_PRIMARY)
    count_ax.legend(frameon=False)
    rate_ax.scatter(frame["rho_in"], y, color=CATEGORICAL[1], s=45, label="taxa do cluster")
    rate_ax.axvline(frame["rho_global"].iloc[0], color=INK_PRIMARY, linestyle="--", label="taxa global")
    for value, label, style in (
        (frame["rho_in"].median(), "mediana dos clusters", "-"),
        (frame["rho_in"].quantile(0.25), "Q1 dos clusters", ":"),
        (frame["rho_in"].quantile(0.75), "Q3 dos clusters", ":"),
    ):
        rate_ax.axvline(value, color=INK_MUTED, linestyle=style, linewidth=1, label=label)
    used_labels: set[str] = set()
    for idx, row in frame.iterrows():
        detection = row["detection_class"] if row["evaluation_status"] == "avaliado" else None
        marker_label = (
            DETECTION_LABELS[detection]
            if detection else "não avaliado — motivo anotado"
        )
        marker_color = DETECTION_COLORS[detection] if detection else COLOR_NOT_EVALUATED
        rate_ax.scatter(
            row["rho_in"], idx, marker=DETECTION_MARKERS[detection], s=85,
            color=marker_color, zorder=4,
            label=(marker_label if marker_label not in used_labels else None),
        )
        used_labels.add(marker_label)
        if detection is None:
            rate_ax.annotate(
                _not_evaluated_reason(row), (row["rho_in"], idx),
                xytext=(6, 0), textcoords="offset points", va="center",
                color=COLOR_NOT_EVALUATED,
            )
    rate_ax.set_xlim(0, 1)
    rate_ax.set_yticks(y, [])
    rate_ax.set_xlabel("percentual positivo")
    rate_ax.set_title("Taxa e detection class", loc="left", color=INK_PRIMARY)
    rate_ax.legend(frameon=False)
    _footer(fig, "Outcomes usam cores categóricas; detection class usa símbolo e contorno próprios.")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def size_dispersion_figure(tables: ExplorationTables) -> Figure:
    frame = tables.cluster_features
    fig, ax = _figure()
    scatter = ax.scatter(
        frame["n"], frame["distance_mean_km"], c=frame["rho_in"], cmap="viridis",
        s=75, vmin=0, vmax=1, alpha=0.75,
    )
    used_labels: set[str] = set()
    for _, row in frame.iterrows():
        detection = row["detection_class"] if row["evaluation_status"] == "avaliado" else None
        marker_label = DETECTION_LABELS[detection] if detection else "não avaliado — motivo anotado"
        marker_color = DETECTION_COLORS[detection] if detection else COLOR_NOT_EVALUATED
        marker_options = (
            {"color": marker_color}
            if detection is None
            else {"facecolors": "none", "edgecolors": marker_color}
        )
        ax.scatter(
            row["n"], row["distance_mean_km"], marker=DETECTION_MARKERS[detection],
            s=90, linewidths=1.5,
            label=(marker_label if marker_label not in used_labels else None),
            **marker_options,
        )
        used_labels.add(marker_label)
        if detection is None:
            ax.annotate(
                f"{int(row['cluster_label'])}: {_not_evaluated_reason(row)}",
                (row["n"], row["distance_mean_km"]), xytext=(5, 5),
                textcoords="offset points", color=COLOR_NOT_EVALUATED,
            )
    positive_sizes = frame.loc[frame["n"] > 0, "n"]
    if len(positive_sizes) and positive_sizes.max() / positive_sizes.min() >= 10:
        ax.set_xscale("log")
    ax.set_xlabel("tamanho do cluster")
    ax.set_ylabel("raio médio interno (km)")
    ax.set_title("Tamanho × dispersão espacial", loc="left", color=INK_PRIMARY)
    fig.colorbar(scatter, ax=ax, label="taxa de positivos")
    ax.legend(frameon=False)
    _footer(fig, "Área dos marcadores é constante; preenchimento representa a taxa, não o tamanho.")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def deviations_figure(tables: ExplorationTables) -> Figure:
    frame = tables.cluster_features
    fig, ax = _figure()
    ax.axhline(0, color=INK_MUTED, linewidth=1)
    ax.axvline(0, color=INK_MUTED, linewidth=1)
    used_labels: set[str] = set()
    for _, row in frame.iterrows():
        detection = row["detection_class"] if row["evaluation_status"] == "avaliado" else None
        marker_label = DETECTION_LABELS[detection] if detection else "não avaliado — motivo anotado"
        ax.scatter(
            row["global_deviation"], row["peer_deviation"],
            marker=DETECTION_MARKERS[detection], s=80,
            color=(DETECTION_COLORS[detection] if detection else COLOR_NOT_EVALUATED),
            alpha=0.8, label=(marker_label if marker_label not in used_labels else None),
        )
        used_labels.add(marker_label)
        if detection in ("negative", "positive") or bool(row.get("auto_selected", False)):
            ax.annotate(str(int(row["cluster_label"])),
                        (row["global_deviation"], row["peer_deviation"]),
                        xytext=(4, 4), textcoords="offset points")
        elif detection is None:
            ax.annotate(
                f"{int(row['cluster_label'])}: {_not_evaluated_reason(row)}",
                (row["global_deviation"], row["peer_deviation"]),
                xytext=(4, 4), textcoords="offset points", color=COLOR_NOT_EVALUATED,
            )
    ax.set_xlabel("desvio da taxa global")
    ax.set_ylabel("desvio da taxa dos peers")
    ax.set_title("Referência global × referência local", loc="left", color=INK_PRIMARY)
    ax.legend(frameon=False)
    _footer(fig, "Contrastes são descritivos; posição no gráfico não estabelece causalidade.")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def detection_distributions_figure(tables: ExplorationTables) -> Figure:
    frame = tables.cluster_features[tables.cluster_features["evaluation_status"] == "avaliado"]
    fig, axes = _figure(2, 3)
    classes = [name for name in ("negative", "positive", "neutral")
               if (frame["detection_class"] == name).any()]
    for column, (metric, title) in enumerate(
        (("n", "Tamanho"), ("internal_predominance", "Predominância"),
         ("distance_mean_km", "Dispersão (km)")),
    ):
        box_ax = axes[0, column]
        values = [frame.loc[frame["detection_class"] == name, metric].dropna() for name in classes]
        if values:
            boxes = box_ax.boxplot(
                values, tick_labels=[DETECTION_LABELS[name] for name in classes],
                patch_artist=True,
            )
            for box, name in zip(boxes["boxes"], classes):
                box.set_facecolor(DETECTION_COLORS[name])
                box.set_alpha(0.55)
        box_ax.set_title(f"{title} · boxplot", loc="left", color=INK_PRIMARY)
        box_ax.tick_params(axis="x", rotation=25)
        ecdf_ax = axes[1, column]
        for name, class_values in zip(classes, values):
            sorted_values = np.sort(class_values.to_numpy(dtype=float))
            if not len(sorted_values):
                continue
            ecdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
            ecdf_ax.step(
                sorted_values, ecdf, where="post", color=DETECTION_COLORS[name],
                linestyle={"negative": "--", "positive": "-", "neutral": ":"}[name],
                label=DETECTION_LABELS[name],
            )
        ecdf_ax.set_title(f"{title} · ECDF", loc="left", color=INK_PRIMARY)
        ecdf_ax.set_ylabel("fração acumulada na classe")
        ecdf_ax.set_ylim(0, 1)
        ecdf_ax.legend(frameon=False)
    _footer(
        fig,
        "Somente clusters avaliados; cada cluster tem o mesmo peso. Boxplots e ECDFs são exploratórios.",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def heatmap_figure(tables: ExplorationTables) -> Figure:
    features = tables.cluster_features.copy()
    status_order = features["evaluation_status"].map({"avaliado": 0}).fillna(1)
    class_order = features["detection_class"].map(
        {"negative": 0, "positive": 1, "neutral": 2}
    ).fillna(3)
    order = pd.DataFrame(
        {
            "index": features.index,
            "status": status_order,
            "class": class_order,
            "magnitude": -features["primary_score"].abs().fillna(-np.inf),
            "label": features["cluster_label"],
        }
    ).sort_values(["status", "class", "magnitude", "label"], kind="stable")
    heatmap = tables.heatmap.set_index("cluster_label").loc[
        features.loc[order["index"], "cluster_label"]
    ]
    values = heatmap.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)
    fig = plt.figure(figsize=(13.333, 7.5), facecolor=SURFACE)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.6, 6.4), wspace=0.05)
    status_ax = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[0, 1])
    for axis in (status_ax, ax):
        axis.set_facecolor(SURFACE)
    image = ax.imshow(masked, aspect="auto", cmap="coolwarm", vmin=-3, vmax=3)
    metric_labels = {
        "n": "tamanho", "rho_in": "taxa positiva",
        "internal_predominance": "predominância", "distance_mean_km": "raio médio",
        "distance_p95_km": "raio p95",
        "class_centroid_separation_km": "separação das classes",
        "primary_score": "primary score",
    }
    zero_iqr = set()
    for column in heatmap.columns:
        finite = pd.to_numeric(features[column], errors="coerce").dropna()
        if len(finite) and float(finite.quantile(0.75) - finite.quantile(0.25)) == 0.0:
            zero_iqr.add(column)
    labels = [
        f"{metric_labels.get(column, column)}{' †' if column in zero_iqr else ''}"
        for column in heatmap.columns
    ]
    ax.set_xticks(np.arange(len(heatmap.columns)), labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(heatmap)), [])
    ax.set_xlabel("característica (mediana/IQR; clipping visual em ±3)")
    ax.set_ylabel("cluster")
    ax.set_title("Perfil robusto dos clusters", loc="left", color=INK_PRIMARY)
    for row, col in zip(*np.where(np.isnan(values))):
        ax.text(col, row, "NA", ha="center", va="center", color=INK_PRIMARY)

    ordered_features = features.loc[order["index"]].reset_index(drop=True)
    status_colors = np.ones((len(ordered_features), 3, 4), dtype=float)
    for row_index, (_, row) in enumerate(ordered_features.iterrows()):
        evaluated = row["evaluation_status"] == "avaliado"
        status_colors[row_index, 0] = matplotlib.colors.to_rgba(
            CATEGORICAL[0] if evaluated else COLOR_NOT_EVALUATED
        )
        status_colors[row_index, 1] = matplotlib.colors.to_rgba(
            DETECTION_COLORS.get(row["detection_class"], COLOR_NOT_EVALUATED)
        )
        status_colors[row_index, 2] = matplotlib.colors.to_rgba(
            CATEGORICAL[1] if bool(row.get("auto_selected", False)) else SURFACE
        )
        symbol = (
            {"negative": "▼", "positive": "▲", "neutral": "○"}.get(
                row["detection_class"], "×"
            )
        )
        status_ax.text(1, row_index, symbol, ha="center", va="center", color=INK_PRIMARY)
    status_ax.imshow(status_colors, aspect="auto")
    status_ax.set_xticks([0, 1, 2], ["avaliação", "classe", "anexo"], rotation=35, ha="right")
    status_labels = []
    for _, row in ordered_features.iterrows():
        label = str(int(row["cluster_label"]))
        if row["evaluation_status"] != "avaliado":
            label += f" · × {_not_evaluated_reason(row)}"
        status_labels.append(label)
    status_ax.set_yticks(np.arange(len(ordered_features)), status_labels)
    status_ax.set_ylabel("cluster e status")
    status_ax.set_title("status", loc="left", color=INK_PRIMARY)
    fig.colorbar(image, ax=ax, label="escala robusta")
    selected_n = int(ordered_features.get("auto_selected", pd.Series(dtype=bool)).sum())
    _footer(
        fig,
        f"† IQR zero: sem variação. NA permanece NA. Seleção automática para o anexo: "
        f"{selected_n}/{len(ordered_features)} clusters. Cores do heatmap são relativas, não significância.",
    )
    fig.subplots_adjust(left=0.14, right=0.94, bottom=0.22, top=0.90)
    return fig


def core_figures(
    tables: ExplorationTables, manifest: dict, primary_metric: str
) -> list[tuple[str, Figure]]:
    figures = [
        ("01_capa", cover_figure(tables, manifest, primary_metric)),
        ("02_cobertura", coverage_figure(tables)),
        ("03_tamanhos", size_distribution_figure(tables)),
        ("04_balanceamento", balance_ranking_figure(tables)),
        ("05_tamanho_dispersao", size_dispersion_figure(tables)),
        ("06_desvios_global_peers", deviations_figure(tables)),
        ("07_distribuicoes_detection", detection_distributions_figure(tables)),
        ("08_heatmap_robusto", heatmap_figure(tables)),
    ]
    for _, figure in figures:
        apply_presentation_style(figure)
    return figures
