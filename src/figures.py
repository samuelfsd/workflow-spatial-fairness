"""Matplotlib figures for the analytical charts (ADR-0005).

Charts are matplotlib, maps stay folium: the deliverable is a dissertation and a
presentation, and HTML goes into neither. Every figure is written as PNG (quick
look) plus vector PDF (LaTeX/slides), and `save_pdf_report` collects a run's
figures into one multipage PDF to open in a meeting.

Sizing targets projection: 16:9 at a legible base font, labels in pt-BR, values
annotated selectively (never a number on every bar), grid and axes recessive.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # WSL has no display; never call plt.show()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from palette import (
    CATEGORICAL,
    DETECTION_COLORS,
    DETECTION_LABELS,
    GRID,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SURFACE,
)

#: 16:9 at a size that stays legible when projected.
SLIDE_SIZE = (10.0, 5.625)

#: Above this many bars, only the extremes get a value label.
_LABEL_ALL_BELOW = 13

_RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK_PRIMARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "grid.color": GRID,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,  # recessive grid stays behind the marks
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.autolayout": False,
}


def _new_figure(nrows: int = 1, height: float | None = None, **kwargs) -> tuple[Figure, Any]:
    with plt.rc_context(_RC):
        figsize = (SLIDE_SIZE[0], height or SLIDE_SIZE[1])
        fig, axes = plt.subplots(nrows=nrows, figsize=figsize, **kwargs)
    return fig, axes


def _fmt_value(value: float) -> str:
    magnitude = abs(value)
    if magnitude == 0:
        return "0"
    if magnitude >= 100:
        return f"{value:,.0f}".replace(",", ".")
    if magnitude >= 1:
        return f"{value:.2f}".replace(".", ",")
    return f"{value:.3f}".replace(".", ",")


def _annotate_selectively(ax, bars, values: list[float]) -> None:
    """Label every bar when there are few; otherwise only the two extremes.

    A number on every one of 42 bars is noise, but a small panel reads better
    with the values visible — so the rule is by count, not by taste.
    """
    finite = [(idx, value) for idx, value in enumerate(values) if not math.isnan(value)]
    if not finite:
        return

    if len(finite) < _LABEL_ALL_BELOW:
        chosen = {idx for idx, _ in finite}
    else:
        chosen = {min(finite, key=lambda item: item[1])[0], max(finite, key=lambda item: item[1])[0]}

    for idx in chosen:
        bar, value = bars[idx], values[idx]
        offset = 3 if value >= 0 else -3
        ax.annotate(
            _fmt_value(value),
            (bar.get_x() + bar.get_width() / 2.0, value),
            textcoords="offset points",
            xytext=(0, offset),
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
            color=INK_SECONDARY,
        )


def _cluster_ticks(ax, labels: list[Any]) -> None:
    step = 1 if len(labels) <= 25 else 2
    positions = list(range(0, len(labels), step))
    ax.set_xticks(positions)
    ax.set_xticklabels([str(labels[i]) for i in positions], rotation=0)


def metric_panels_figure(
    panels: list[dict],
    *,
    dataset: str,
    method: str,
) -> Figure:
    """Small-multiples: one panel per metric, one bar per cluster, shared order.

    Each panel dict: {"name", "labels", "values", "directions", "significant",
    "threshold", "analytic_threshold", "signed", "caption"}. Bar height is the
    metric's native value; bar color is the detection class. Dashed line = Monte
    Carlo threshold; dotted = analytic (Šidák) cross-check, drawn only for
    standardized metrics.
    """
    if not panels:
        raise ValueError("metric_panels_figure needs at least one panel")

    height = max(SLIDE_SIZE[1], 1.7 * len(panels) + 1.6)
    fig, axes = _new_figure(nrows=len(panels), height=height, sharex=True)
    axes = np.atleast_1d(axes)

    for ax, panel in zip(axes, panels):
        values = [float(value) for value in panel["values"]]
        colors = [
            DETECTION_COLORS[direction if significant else "neutral"]
            for direction, significant in zip(panel["directions"], panel["significant"])
        ]
        plotted = [0.0 if math.isnan(value) else value for value in values]
        bars = ax.bar(range(len(values)), plotted, color=colors, width=0.72, linewidth=0)

        # Monte Carlo reads as the ruler in force; the analytic band is a lighter
        # cross-check sitting just above it (they nearly coincide by design).
        rulers = (("threshold", "--", 1.4, 0.75), ("analytic_threshold", ":", 1.2, 0.45))
        for threshold_key, style, line_width, alpha in rulers:
            threshold = panel.get(threshold_key)
            if threshold is None or (isinstance(threshold, float) and math.isnan(threshold)):
                continue
            if float(threshold) <= 0.0:
                # A zero threshold would mark everything significant; the run that
                # produced it is degenerate, so drawing the line would mislead.
                continue
            for sign in ((1, -1) if panel.get("signed") else (1,)):
                ax.axhline(
                    sign * float(threshold),
                    linestyle=style,
                    linewidth=line_width,
                    color=INK_PRIMARY,
                    alpha=alpha,
                )

        if panel.get("signed"):
            ax.axhline(0.0, linewidth=1.0, color=GRID)
        _annotate_selectively(ax, bars, values)
        # Headroom so value labels never land on the panel caption above them.
        ax.margins(y=0.24)
        if all(math.isnan(value) for value in values):
            ax.text(
                0.5,
                0.5,
                "não avaliado nesta partição",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color=INK_MUTED,
            )
        ax.set_ylabel(panel["name"], rotation=90, labelpad=8)
        caption = panel.get("caption", "")
        if caption:
            ax.set_title(caption, loc="left", fontsize=9, color=INK_MUTED, pad=6)

    _cluster_ticks(axes[-1], list(panels[0]["labels"]))
    axes[-1].set_xlabel("cluster")

    handles = [Patch(facecolor=DETECTION_COLORS[key], label=DETECTION_LABELS[key]) for key in DETECTION_COLORS]
    handles += [
        Line2D([0], [0], linestyle="--", color=INK_PRIMARY, alpha=0.6, label="limiar Monte Carlo"),
        Line2D([0], [0], linestyle=":", color=INK_PRIMARY, alpha=0.6, label="limiar analítico (Šidák)"),
    ]
    fig.suptitle(
        f"Métricas por cluster · {dataset} ({method})",
        x=0.01,
        y=0.995,
        ha="left",
        fontsize=13,
        color=INK_PRIMARY,
    )
    # Legend below the title, never on top of it.
    fig.legend(
        handles=handles,
        loc="upper left",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.01, 0.955),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    return fig


def balance_figure(frame: pd.DataFrame, *, dataset: str, method: str) -> Figure:
    """Stacked balance per cluster: positives and negatives adding up to `n`.

    Deliberately **not** red/green: point outcome is not a detection class, and
    reusing those hues would collide with the cluster's verdict language.
    """
    fig, ax = _new_figure()
    labels = list(frame["cluster_label"])
    positions = range(len(frame))
    positives = frame["p"].to_numpy(dtype=float)
    negatives = frame["n_neg"].to_numpy(dtype=float)

    # 1px surface edge keeps the two stacked fills from touching.
    ax.bar(positions, positives, color=CATEGORICAL[0], width=0.72,
           edgecolor=SURFACE, linewidth=1.0, label="positivos (outcome = 1)")
    ax.bar(positions, negatives, bottom=positives, color=CATEGORICAL[1], width=0.72,
           edgecolor=SURFACE, linewidth=1.0, label="negativos (outcome = 0)")

    ax.set_ylabel("pontos no cluster", rotation=90)
    ax.set_xlabel("cluster")
    _cluster_ticks(ax, labels)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title(
        f"Balanceamento por cluster · {dataset} ({method})",
        loc="left",
        color=INK_PRIMARY,
    )
    fig.tight_layout()
    return fig


def dispersion_figure(table: pd.DataFrame, *, dataset: str, stat: str = "cv") -> Figure:
    """Grouped bars of one dispersion statistic per variable, one group per config.

    Defaults to the CV because it is the only scale-free reading — the raw sigma
    of counts is not comparable between variables or between configurations.
    """
    rows = table.xs(stat, level="stat")
    fig, ax = _new_figure()
    variables = list(rows.index)
    configs = list(rows.columns)
    width = 0.8 / max(len(configs), 1)

    for idx, config in enumerate(configs):
        offsets = [position + idx * width - 0.4 + width / 2 for position in range(len(variables))]
        values = [float(value) for value in rows[config]]
        bars = ax.bar(offsets, values, width=width * 0.9, color=CATEGORICAL[idx % len(CATEGORICAL)], label=str(config), linewidth=0)
        if len(configs) * len(variables) < _LABEL_ALL_BELOW:
            _annotate_selectively(ax, bars, values)

    ax.set_xticks(range(len(variables)))
    ax.set_xticklabels(variables)
    label = {"cv": "coeficiente de variação (σ/média)", "std": "desvio padrão", "var": "variância"}
    ax.set_ylabel(label.get(stat, stat), rotation=90)
    ax.set_xlabel("variável (entre clusters)")
    if len(configs) > 1:
        ax.legend(frameon=False, loc="upper right", title="configuração")
    ax.set_title(
        f"Dispersão entre clusters · {dataset}",
        loc="left",
        color=INK_PRIMARY,
    )
    fig.tight_layout()
    return fig


#: The readings of the partition profile, as (column, title, how to read it, format).
PROFILE_READINGS = (
    ("cluster_size_cv", "CV do tamanho dos clusters", "menor = clusters mais comparáveis", "num"),
    ("noise_rate", "Fração não atribuída", "menor = mais cobertura (o custo do cap)", "pct"),
    ("rho_sigma", "σ das taxas entre clusters", "não deve desabar: é o sinal", "num"),
    ("raio_medio_km_mean", "Raio médio (km)", "menor = clusters mais compactos", "num"),
)


def profile_figure(profile: pd.DataFrame, *, dataset: str, readings=PROFILE_READINGS) -> Figure:
    """The partition profile as small multiples: one panel per reading.

    This is the decision figure for the size cap (ADR-0001). The readings live on
    different scales — a coefficient of variation and a percentage of points — so
    they get **separate panels**, never a second y-axis. Colour identifies the
    configuration and is consistent with `dispersion_figure`, so a configuration
    keeps its identity across the whole report.
    """
    available = [item for item in readings if item[0] in profile.columns]
    if not available or profile.empty:
        raise ValueError("profile_figure needs a non-empty profile with at least one reading")

    labels = [str(value) for value in profile["config"]]
    ncols = min(len(available), 2)
    nrows = math.ceil(len(available) / ncols)
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(SLIDE_SIZE[0], max(SLIDE_SIZE[1], 3.1 * nrows)),
        )
    flat = np.atleast_1d(axes).ravel()

    for ax, (column, title, hint, kind) in zip(flat, available):
        values = [float(value) for value in profile[column]]
        colors = [CATEGORICAL[idx % len(CATEGORICAL)] for idx in range(len(values))]
        bars = ax.bar(range(len(values)), values, color=colors, width=0.62, linewidth=0)
        for bar, value in zip(bars, values):
            text = f"{value:.1%}".replace(".", ",") if kind == "pct" else _fmt_value(value)
            ax.annotate(
                text,
                (bar.get_x() + bar.get_width() / 2.0, value),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=9,
                color=INK_SECONDARY,
            )
        ax.margins(y=0.22)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
        ax.set_title(f"{title}\n{hint}", loc="left", fontsize=10, color=INK_PRIMARY, pad=6)
        if kind == "pct":
            ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")

    for ax in flat[len(available):]:
        ax.set_visible(False)

    fig.suptitle(
        f"Perfil das partições · {dataset}",
        x=0.01,
        y=0.995,
        ha="left",
        fontsize=13,
        color=INK_PRIMARY,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def cluster_card_figure(card: dict, *, dataset: str, granularity: str) -> Figure:
    """One cluster in depth: subcluster rates (intra) + three reference rates (extra).

    Bars are the ingredients of `gini_subcluster`, sorted so a pocket lands at
    the end; `n` is annotated on every bar because a 25-point pocket and a
    4.000-point pocket are not the same news. The Gini is the caption, not the
    chart — a cluster's Gini is a single number.
    """
    subclusters = card["subclusters"].sort_values("rho", ascending=True).reset_index(drop=True)
    fig, ax = _new_figure()

    positions = range(len(subclusters))
    rates = subclusters["rho"].to_numpy(dtype=float)
    bars = ax.bar(positions, rates, color=CATEGORICAL[0], width=0.68, linewidth=0)
    for bar, size in zip(bars, subclusters["n"]):
        height = bar.get_height()
        # Sizes sit *inside* the bar: above it they would land on the reference
        # rate lines, which are the other half of this figure's message.
        inside = height >= 0.12
        ax.annotate(
            f"n={int(size)}",
            (bar.get_x() + bar.get_width() / 2.0, height),
            textcoords="offset points",
            xytext=(0, -6 if inside else 3),
            ha="center",
            va="top" if inside else "bottom",
            fontsize=9,
            color=SURFACE if inside else INK_SECONDARY,
        )

    references = (
        ("rho_in", "taxa do cluster (ρ_in)", "-"),
        ("rho_peer", "taxa dos vizinhos (ρ_peer)", "--"),
        ("rho_global", "taxa global do mapa", ":"),
    )
    for key, label, style in references:
        value = card.get(key)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        ax.axhline(float(value), linestyle=style, linewidth=1.6, color=INK_PRIMARY, alpha=0.65, label=label)

    # Rates are proportions: anchor the axis at 0 and 1 so bar heights are honest.
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("taxa de positivos do subcluster", rotation=90)
    ax.set_xlabel("subcluster (ordenado pela taxa)")
    ax.set_xticks(list(positions))
    ax.set_xticklabels([str(idx) for idx in subclusters["subcluster"]])
    ax.legend(frameon=False, loc="upper right")

    headline = (
        f"Ficha do cluster {card['cluster_label']} · {dataset} · granularidade {granularity}"
    )
    caption = (
        f"n={card['n']} · ρ_in={_fmt_value(card['rho_in'])} · "
        f"gini_subcluster={_fmt_value(card['gini_subcluster'])}"
    )
    if card.get("homogeneous"):
        caption += " · não se subdivide nesta granularidade (homogêneo por dentro)"
    ax.set_title(f"{headline}\n{caption}", loc="left", fontsize=11, color=INK_PRIMARY)
    fig.tight_layout()
    return fig


def save_figure(fig: Figure, output_base: Path) -> list[Path]:
    """Write one figure as PNG (quick look) and vector PDF (LaTeX/slides)."""
    output_base.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in (".png", ".pdf"):
        path = output_base.with_suffix(suffix)
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
        written.append(path)
    return written


def save_pdf_report(figures: list[Figure], output_path: Path) -> Path:
    """Collect figures into one multipage PDF, in narrative order."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        for fig in figures:
            pdf.savefig(fig, bbox_inches="tight", facecolor=SURFACE)
    return output_path


def close(*figures: Figure) -> None:
    """Release figures (the Agg backend keeps them alive until closed)."""
    for fig in figures:
        plt.close(fig)
