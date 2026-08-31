"""Compact benchmark figures derived only from canonical long-form tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from figures import save_figure, save_pdf_report
from palette import CATEGORICAL, GRID, INK_PRIMARY


DATASET_LABELS = {
    "semisynth": "SemiSynth\njusto",
    "synth_unfair": "Synth\ninjusto",
    "synth_fair": "Synth adicional\njusto",
    "lar": "LAR",
    "crime": "Crime",
}
METRIC_LABELS = {
    "sul": "SUL",
    "local_z": "local-z",
    "peer_rate_difference": "diferença de taxa",
    "peer_log_rate_ratio": "log razão de taxas",
    "peer_gini_gap": "contraste de Gini",
}
DETECTOR_ORDER = {
    "Grade + SUL": 0,
    "Varredura + SUL": 1,
    "HDBSCAN + SUL": 2,
    "HDBSCAN + local-z": 3,
    "HDBSCAN + diferença de taxa": 4,
    "HDBSCAN + log razão de taxas": 5,
    "HDBSCAN + contraste de Gini": 6,
}


def _figure(ncols: int = 1):
    return plt.subplots(1, ncols, figsize=(10, 5.625), squeeze=False)


def _summary(canonical: pd.DataFrame) -> pd.DataFrame:
    """Collapse one-record-per-quantity input into one row per detector."""
    if canonical.empty or not {"quantity", "value"}.issubset(canonical.columns):
        return pd.DataFrame()
    identity = [
        column for column in (
            "dataset", "source", "protocol", "experiment", "method",
            "region_system", "metric",
        ) if column in canonical.columns
    ]
    return (
        canonical.groupby(identity + ["quantity"], dropna=False, sort=False)["value"]
        .first()
        .unstack("quantity")
        .reset_index()
    )


def _short_label(row) -> str:
    method = str(row.method)
    metric = METRIC_LABELS.get(str(row.metric), str(row.metric))
    if method == "kmeans_scan":
        return "Varredura + SUL"
    if method == "grid":
        return "Grade + SUL"
    if method == "hdbscan":
        return f"HDBSCAN + {metric}"
    return f"{method} + {metric}"


def _as_bool(value) -> bool:
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"false", "não", "nao", "0"}:
            return False
        if normalized in {"true", "sim", "1"}:
            return True
    return bool(value)


def _key_detectors(summary: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Keep only presentation configurations; never one row per raw checkpoint."""
    if summary.empty:
        return summary
    frame = summary[
        summary.get("dataset", pd.Series(index=summary.index, dtype=str)).eq(dataset)
        & summary.get("source", pd.Series("local", index=summary.index)).eq("local")
    ].copy()
    if frame.empty:
        return frame
    if frame.get("protocol", pd.Series(index=frame.index, dtype=str)).eq(
        "standardized"
    ).any():
        frame = frame[frame["protocol"].eq("standardized")].copy()
    region = frame.get("region_system", pd.Series("", index=frame.index)).astype(str)
    method = frame.get("method", pd.Series("", index=frame.index)).astype(str)
    metric = frame.get("metric", pd.Series("", index=frame.index)).astype(str)
    keep = (
        (method.eq("grid") & metric.eq("sul"))
        | (method.eq("kmeans_scan") & metric.eq("sul"))
        | (
            method.eq("hdbscan")
            & region.eq("hdbscan_frac_0.005")
            & metric.isin(set(METRIC_LABELS))
        )
    )
    frame = frame[keep].copy()
    if frame.empty:
        return frame
    frame["detector_label"] = [_short_label(row) for row in frame.itertuples()]
    frame["detector_order"] = frame["detector_label"].map(DETECTOR_ORDER).fillna(99)
    frame["candidate_sort"] = pd.to_numeric(
        frame.get("candidate_regions", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    ).fillna(-1)
    # LAR has two grids; retain the higher-resolution one under one short label.
    return frame.sort_values(
        ["detector_order", "candidate_sort"], ascending=[True, False], kind="stable"
    ).drop_duplicates("detector_label", keep="first").sort_values(
        "detector_order", kind="stable"
    )


def _synthetic_matrix(canonical: pd.DataFrame):
    fig, axes = _figure(1)
    ax = axes[0, 0]
    summary = _summary(canonical)
    rows = []
    for dataset in ("semisynth", "synth_unfair", "synth_fair"):
        selected = _key_detectors(summary, dataset)
        for row in selected.itertuples():
            if not hasattr(row, "unfairness_detected") or pd.isna(row.unfairness_detected):
                continue
            detected = _as_bool(row.unfairness_detected)
            expected = dataset == "synth_unfair"
            rows.append({
                "dataset": dataset,
                "detector": row.detector_label,
                "correct": detected == expected,
                "detected": detected,
                "significant_regions": getattr(row, "significant_regions", np.nan),
            })
    frame = pd.DataFrame(rows)
    datasets = [
        dataset for dataset in ("semisynth", "synth_unfair", "synth_fair")
        if frame.empty or dataset in set(frame.get("dataset", []))
    ]
    detectors = sorted(
        frame.get("detector", pd.Series(dtype=str)).dropna().unique(),
        key=lambda value: DETECTOR_ORDER.get(value, 99),
    )
    if not detectors:
        detectors = ["sem resultados canônicos"]
    matrix = np.zeros((len(detectors), len(datasets)), dtype=int)
    annotations = np.full(matrix.shape, "—", dtype=object)
    for row in frame.itertuples():
        y = detectors.index(row.detector)
        x = datasets.index(row.dataset)
        matrix[y, x] = 1 if row.correct else 2
        count = row.significant_regions
        annotations[y, x] = (
            f"{int(float(count))} reg."
            if pd.notna(count)
            else "detecta" if row.detected else "silêncio"
        )
    cmap = ListedColormap([
        "#ECEFF1", CATEGORICAL[0], CATEGORICAL[2]
    ])
    norm = BoundaryNorm([-.5, .5, 1.5, 2.5], cmap.N)
    ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    for y in range(len(detectors)):
        for x in range(len(datasets)):
            ax.text(x, y, annotations[y, x], ha="center", va="center", fontsize=9)
    detector_ticks = [label.replace(" + ", "\n+ ") for label in detectors]
    ax.set_xticks(
        range(len(datasets)), [DATASET_LABELS[item] for item in datasets],
        rotation=0, ha="center",
    )
    ax.set_yticks(range(len(detectors)), detector_ticks)
    ax.set_title("Controles sintéticos — detector × cenário")
    ax.set_xlabel("Cenário com desenho conhecido")
    ax.set_ylabel("Configuração comparada")
    ax.legend(
        handles=[
            Patch(facecolor=CATEGORICAL[0], label="✓ veredito correto"),
            Patch(facecolor=CATEGORICAL[2], label="× veredito incorreto"),
            Patch(facecolor="#ECEFF1", label="não disponível"),
        ],
        loc="upper center", bbox_to_anchor=(.5, 1.18), ncol=3, frameon=False,
    )
    fig.suptitle(
        "Benchmark sintético: uma célula por cenário e detector",
        color=INK_PRIMARY, fontsize=15,
    )
    fig.tight_layout()
    return fig


def _local_global(canonical: pd.DataFrame):
    fig, axes = _figure(2)
    summary = _summary(canonical)
    for ax, dataset in zip(axes[0], ("lar", "crime"), strict=True):
        frame = _key_detectors(summary, dataset).copy()
        # This panel is deliberately rate-only. The Gini candidate has its own
        # indicator/reference columns and must not be drawn as a rate contrast.
        frame = frame[~frame["metric"].eq("peer_gini_gap")]
        frame["local_rate"] = pd.to_numeric(
            frame.get("best_region_rate", pd.Series(np.nan, index=frame.index)),
            errors="coerce",
        )
        reference = frame.get(
            "best_reference_rate", pd.Series(np.nan, index=frame.index)
        )
        frame["reference_rate"] = pd.to_numeric(reference, errors="coerce")
        fallback = pd.to_numeric(
            frame.get("global_rate", pd.Series(np.nan, index=frame.index)),
            errors="coerce",
        )
        frame["reference_rate"] = frame["reference_rate"].fillna(fallback)
        frame = frame.dropna(subset=["local_rate", "reference_rate"])
        y = np.arange(len(frame))
        for idx, row in enumerate(frame.itertuples()):
            ax.plot(
                [row.reference_rate, row.local_rate], [idx, idx],
                color=GRID, linewidth=3,
            )
            ax.scatter(
                row.reference_rate, idx, s=55, facecolor="white",
                edgecolor=INK_PRIMARY, zorder=3,
            )
            ax.scatter(
                row.local_rate, idx, s=70, color=CATEGORICAL[1],
                edgecolor="white", zorder=4,
            )
            ax.text(
                min(1.0, max(row.reference_rate, row.local_rate) + .015), idx,
                f"{(row.local_rate - row.reference_rate) * 100:+.1f} p.p.",
                va="center", fontsize=8,
            )
        ax.set_yticks(y, frame.get("detector_label", pd.Series(dtype=str)))
        ax.set_xlim(0, 1)
        ax.set_xlabel("Taxa")
        ax.set_title(f"Taxa local e referência — {DATASET_LABELS[dataset]}")
        ax.grid(axis="x", color=GRID, alpha=.55)
    fig.legend(
        handles=[
            Line2D(
                [], [], marker="o", linestyle="", markerfacecolor="white",
                markeredgecolor=INK_PRIMARY, label="referência",
            ),
            Line2D(
                [], [], marker="o", linestyle="",
                color=CATEGORICAL[1], label="taxa local",
            ),
        ],
        frameon=False, loc="upper center", bbox_to_anchor=(.5, .93), ncol=2,
    )
    fig.suptitle(
        "Efeito em unidade comum: taxa local versus sua própria referência",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, .88))
    return fig


def _workload(canonical: pd.DataFrame):
    fig, axes = _figure(2)
    summary = _summary(canonical)
    quantities = [
        ("candidate_regions", "avaliadas/candidatas", CATEGORICAL[0]),
        ("significant_regions", "significativas", CATEGORICAL[1]),
        ("consolidated_regions", "consolidadas", CATEGORICAL[2]),
    ]
    for ax, dataset in zip(axes[0], ("lar", "crime"), strict=True):
        frame = _key_detectors(summary, dataset).copy()
        y = np.arange(len(frame)); height = .22
        for offset, (quantity, label, color) in enumerate(quantities):
            values = pd.to_numeric(
                frame.get(quantity, pd.Series(np.nan, index=frame.index)),
                errors="coerce",
            )
            positions = y + (offset - 1) * height
            ax.barh(positions, values.fillna(0), height, label=label, color=color)
            for position, value in zip(positions, values, strict=True):
                if pd.notna(value):
                    rendered = f"{int(value):,}".replace(",", ".")
                    ax.text(float(value), position, f" {rendered}", va="center", fontsize=8)
        ax.set_yticks(y, frame.get("detector_label", pd.Series(dtype=str)))
        ax.invert_yaxis()
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlabel("Número de regiões — escala log após 1")
        ax.set_title(f"Carga da auditoria — {DATASET_LABELS[dataset]}")
        ax.grid(axis="x", color=GRID, alpha=.45)
    handles = [
        Patch(facecolor=color, label=label)
        for _, label, color in quantities
    ]
    fig.legend(
        handles=handles, frameon=False, loc="upper center",
        bbox_to_anchor=(.5, .93), ncol=3,
    )
    fig.suptitle("Carga operacional por configuração canônica", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .88))
    return fig


def build_benchmark_figures(
    tables: dict[str, pd.DataFrame],
) -> list[tuple[str, object]]:
    canonical = tables["canonical"]
    return [
        ("01_controles_sinteticos", _synthetic_matrix(canonical)),
        ("02_taxa_local_global", _local_global(canonical)),
        ("03_carga_auditoria", _workload(canonical)),
    ]


def render_benchmark_figures(
    tables: dict[str, pd.DataFrame], output_dir: Path,
) -> list[object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = build_benchmark_figures(tables)
    figures = [figure for _, figure in pairs]
    for name, figure in pairs:
        save_figure(figure, output_dir / name)
    save_pdf_report(figures, output_dir / "benchmark_quantitativo.pdf")
    # Publication owns final closure so it can also clean up on renderer failure.
    return figures
