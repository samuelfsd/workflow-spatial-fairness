"""Supplementary exploratory calculations and traceable factual summaries."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from figure_style import apply_presentation_style
from palette import (
    CATEGORICAL,
    DETECTION_COLORS,
    DETECTION_LABELS,
    INK_PRIMARY,
    SURFACE,
)


METRIC_LABELS = {
    "n": "tamanho do cluster (pontos)",
    "rho_in": "taxa de positivos",
    "internal_predominance": "predominância interna",
    "distance_mean_km": "raio médio interno (km)",
    "class_centroid_separation_km": "separação dos centroides das classes (km)",
    "peer_deviation": "desvio da taxa dos peers",
    "evidence_ratio": "evidência relativa (|score|/limiar)",
}


def spearman_pairwise(
    frame: pd.DataFrame, metrics: list[str], *, minimum_pair_n: int = 10
) -> pd.DataFrame:
    """Pairwise Spearman correlations with n, deliberately without p-values."""
    rows = []
    for left_index, left in enumerate(metrics):
        for right in metrics[left_index + 1 :]:
            values = frame[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(values) < minimum_pair_n:
                continue
            rows.append(
                {
                    "metric_a": left,
                    "metric_b": right,
                    "spearman_rho": float(values[left].corr(values[right], method="spearman")),
                    "n_pair": len(values),
                }
            )
    return pd.DataFrame(
        rows, columns=["metric_a", "metric_b", "spearman_rho", "n_pair"]
    )


def factual_summary(
    cluster_features: pd.DataFrame, selected_clusters: pd.DataFrame
) -> dict[str, Any]:
    """Machine-readable facts and caveats; never automated causal prose."""
    facts = {
        "n_clusters": len(cluster_features),
        "n_selected_clusters": int(selected_clusters["cluster_label"].nunique())
        if not selected_clusters.empty else 0,
        "assigned_points": int(cluster_features["n"].sum())
        if "n" in cluster_features else 0,
    }
    rankings = {}
    for metric in ("n", "rho_in", "internal_predominance", "distance_mean_km", "evidence_ratio"):
        if metric not in cluster_features:
            continue
        top = cluster_features[["cluster_label", metric]].dropna().nlargest(5, metric)
        rankings[metric] = top.to_dict("records")
    criteria = (
        selected_clusters.groupby("cluster_label")["reason"].apply(list).to_dict()
        if not selected_clusters.empty else {}
    )
    return {
        "facts": facts,
        "rankings": rankings,
        "selection_criteria": criteria,
        "caveats": [
            "Associações visuais são exploratórias e não estabelecem causalidade.",
            "Primary score depende de tamanho, taxa e referência estatística.",
            "Não atribuído permanece fora das métricas de cluster.",
        ],
    }


def supplementary_figures(frame: pd.DataFrame) -> list[tuple[str, Figure]]:
    """Additional bivariate views kept outside the confirmatory-looking core."""
    definitions = (
        ("n", "rho_in", "Tamanho × taxa positiva"),
        ("distance_mean_km", "internal_predominance", "Dispersão × predominância"),
        ("class_centroid_separation_km", "peer_deviation", "Separação × desvio dos peers"),
        ("n", "evidence_ratio", "Tamanho × evidência relativa"),
    )
    figures = []
    for index, (x_name, y_name, title) in enumerate(definitions, start=1):
        fig, ax = plt.subplots(figsize=(13.333, 7.5), facecolor=SURFACE)
        valid = frame[[x_name, y_name]].dropna()
        ax.scatter(valid[x_name], valid[y_name], color=CATEGORICAL[(index - 1) % len(CATEGORICAL)],
                   alpha=0.7, s=55)
        ax.set_xlabel(METRIC_LABELS.get(x_name, x_name))
        ax.set_ylabel(METRIC_LABELS.get(y_name, y_name))
        ax.set_title(title, loc="left", color=INK_PRIMARY)
        fig.text(0.01, 0.01, "Leitura exploratória; não estabelece causalidade nem validação confirmatória.",
                 )
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        figures.append((f"scatter_{index:02d}_{x_name}_{y_name}", fig))

    ranking, axes = plt.subplots(1, 2, figsize=(13.333, 7.5), facecolor=SURFACE)
    for ax, metric, title in (
        (axes[0], "n", "Maiores clusters"),
        (axes[1], "evidence_ratio", "Maior evidência relativa"),
    ):
        ranked = frame[["cluster_label", metric]].dropna().nlargest(20, metric).sort_values(metric)
        ax.barh(ranked["cluster_label"].astype(str), ranked[metric], color=CATEGORICAL[0])
        ax.set_xlabel(METRIC_LABELS.get(metric, metric))
        ax.set_ylabel("cluster")
        ax.set_title(title, loc="left")
    ranking.suptitle("Rankings completos no CSV; top 20 para leitura visual", x=0.01, ha="left")
    ranking.tight_layout(rect=(0, 0, 1, 0.94))
    figures.append(("ranking_top20", ranking))

    evaluated = frame[frame["evaluation_status"] == "avaliado"]
    ecdf, axes = plt.subplots(1, 3, figsize=(13.333, 7.5), facecolor=SURFACE)
    for ax, metric in zip(axes, ("n", "internal_predominance", "distance_mean_km")):
        for detection_class in ("negative", "positive", "neutral"):
            values = np.sort(
                evaluated.loc[evaluated["detection_class"] == detection_class, metric]
                .dropna().to_numpy(dtype=float)
            )
            if not len(values):
                continue
            y = np.arange(1, len(values) + 1) / len(values)
            ax.step(
                values, y, where="post", label=DETECTION_LABELS[detection_class],
                color=DETECTION_COLORS[detection_class],
                linestyle={"negative": "--", "positive": "-", "neutral": ":"}[detection_class],
            )
        ax.set_title(METRIC_LABELS.get(metric, metric), loc="left")
        ax.set_ylim(0, 1)
        ax.set_ylabel("ECDF dentro da classe")
        ax.legend(frameon=False)
    ecdf.suptitle("Distribuições por detection class · somente avaliados", x=0.01, ha="left")
    ecdf.tight_layout(rect=(0, 0, 1, 0.94))
    figures.append(("ecdf_detection_classes", ecdf))
    for _, figure in figures:
        apply_presentation_style(figure)
    return figures
