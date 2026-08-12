"""Selection, detail utilities and honest multiscale internal evidence."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from clustering.base import Partition
from clustering.internal import InternalSubdivision, diagnostic_density_subdivision
from data_loading import LoadedDataset
from figure_style import apply_presentation_style
from metrics.group_fairness import calculate_gini
from metrics.registry import get_primary_capabilities
from palette import CATEGORICAL, DETECTION_COLORS, INK_PRIMARY, INK_SECONDARY, SURFACE


OUTLIER_METRICS = (
    "rho_in",
    "internal_predominance",
    "global_deviation",
    "distance_mean_km",
    "distance_p95_km",
)


def large_cluster_labels(frame: pd.DataFrame) -> set[int]:
    """Leave-one-out mean + one sample sigma tail over cluster sizes."""
    labels: set[int] = set()
    values = frame[["cluster_label", "n"]].dropna()
    for _, row in values.iterrows():
        others = values.loc[values["cluster_label"] != row["cluster_label"], "n"].astype(float)
        if len(others) < 2:
            continue
        threshold = float(others.mean() + others.std(ddof=1))
        if float(row["n"]) > threshold:
            labels.add(int(row["cluster_label"]))
    return labels


def _tukey_reasons(frame: pd.DataFrame) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    for metric in OUTLIER_METRICS:
        finite = frame[["cluster_label", metric]].dropna()
        if len(finite) < 8:
            continue
        q1 = float(finite[metric].quantile(0.25))
        q3 = float(finite[metric].quantile(0.75))
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        for _, row in finite.iterrows():
            value = float(row[metric])
            if value < lower:
                rows.append({"cluster_label": int(row["cluster_label"]), "reason": f"tukey_low:{metric}"})
            elif value > upper:
                rows.append({"cluster_label": int(row["cluster_label"]), "reason": f"tukey_high:{metric}"})
    return rows


def select_clusters(frame: pd.DataFrame, selection: str = "auto") -> pd.DataFrame:
    """Relational selection table: one cluster may retain several reasons."""
    available = set(frame["cluster_label"].astype(int))
    rows: list[dict[str, int | str]] = []
    if selection == "all":
        rows = [{"cluster_label": label, "reason": "selection_all"} for label in sorted(available)]
    elif selection == "auto":
        detected = frame[
            (frame["evaluation_status"] == "avaliado")
            & frame["detection_class"].isin(["negative", "positive"])
        ]
        rows.extend(
            {
                "cluster_label": int(row["cluster_label"]),
                "reason": f"detection:{row['detection_class']}",
            }
            for _, row in detected.iterrows()
        )
        rows.extend(
            {"cluster_label": label, "reason": "cluster_grande_leave_one_out"}
            for label in sorted(large_cluster_labels(frame))
        )
        rows.extend(_tukey_reasons(frame))
    else:
        try:
            requested = {int(value.strip()) for value in selection.split(",") if value.strip()}
        except ValueError as exc:
            raise ValueError("Detail selection must be auto, all, or comma-separated labels") from exc
        unknown = requested.difference(available)
        if unknown:
            raise ValueError(f"Unknown cluster labels in detail selection: {sorted(unknown)}")
        rows = [
            {"cluster_label": label, "reason": "selection_explicit"}
            for label in sorted(requested)
        ]

    if not rows:
        return pd.DataFrame(columns=["cluster_label", "reason"])
    return pd.DataFrame(rows).drop_duplicates().sort_values(
        ["cluster_label", "reason"], kind="stable"
    ).reset_index(drop=True)


def shared_histogram_bins(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    """Freedman-Diaconis shared edges with deterministic Sturges fallback."""
    values = np.concatenate(
        [np.asarray(positive, dtype=float), np.asarray(negative, dtype=float)]
    )
    values = values[np.isfinite(values)]
    if not len(values):
        return np.array([0.0, 1.0])
    minimum, maximum = float(values.min()), float(values.max())
    if minimum == maximum:
        padding = 0.5 if minimum == 0 else abs(minimum) * 0.05
        return np.array([minimum - padding, maximum + padding])
    iqr = float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
    width = 2.0 * iqr / np.cbrt(len(values)) if iqr > 0 else 0.0
    if width > 0 and math.isfinite(width):
        n_bins = int(math.ceil((maximum - minimum) / width))
    else:
        n_bins = int(math.ceil(math.log2(len(values)) + 1))
    n_bins = min(100, max(1, n_bins))
    return np.linspace(minimum, maximum, n_bins + 1)


def stratified_point_sample(
    points: list[int], outcomes: np.ndarray, *, max_points: int = 5000, seed: int = 42
) -> list[int]:
    """Deterministic outcome-stratified IDs for the geographic scatter only."""
    points_array = np.asarray(points, dtype=int)
    if len(points_array) <= max_points:
        return sorted(points_array.tolist())
    values = np.asarray(outcomes)[points_array]
    groups = {value: points_array[values == value] for value in sorted(np.unique(values))}
    rng = np.random.default_rng(seed)
    allocations = {
        value: max(1, int(round(max_points * len(group) / len(points_array))))
        for value, group in groups.items()
    }
    while sum(allocations.values()) > max_points:
        value = max(allocations, key=lambda item: (allocations[item], len(groups[item])))
        if allocations[value] > 1:
            allocations[value] -= 1
        else:
            break
    while sum(allocations.values()) < max_points:
        candidates = [value for value, group in groups.items() if allocations[value] < len(group)]
        if not candidates:
            break
        value = max(candidates, key=lambda item: len(groups[item]) - allocations[item])
        allocations[value] += 1
    selected = []
    for value, group in groups.items():
        count = min(allocations[value], len(group))
        selected.extend(rng.choice(group, size=count, replace=False).astype(int).tolist())
    return sorted(selected)


def paginate_cluster_bundles(
    pages_by_cluster: dict[int, int], *, target_pages: int = 50
) -> pd.DataFrame:
    """Assign whole cluster bundles to volumes; no bundle crosses a boundary."""
    rows = []
    volume = 1
    next_page = 1
    for label, page_count in pages_by_cluster.items():
        page_count = int(page_count)
        if next_page > 1 and next_page - 1 + page_count > target_pages:
            volume += 1
            next_page = 1
        rows.append(
            {
                "cluster_label": int(label),
                "volume": volume,
                "page_start": next_page,
                "page_end": next_page + page_count - 1,
                "page_count": page_count,
            }
        )
        next_page += page_count
    return pd.DataFrame(rows, columns=["cluster_label", "volume", "page_start", "page_end", "page_count"])


@dataclass
class InternalEvidence:
    components: pd.DataFrame
    scale_summary: pd.DataFrame
    overlap: pd.DataFrame


def point_distance_frame(
    dataset: LoadedDataset, points: list[int]
) -> pd.DataFrame:
    """Exact per-point distance to the parent centroid; no pairwise matrix."""
    subset = dataset.df.iloc[points]
    lat = subset["lat"].to_numpy(dtype=float)
    lon = subset["lon"].to_numpy(dtype=float)
    lat0, lon0 = float(lat.mean()), float(lon.mean())
    lat_rad, lon_rad = np.radians(lat), np.radians(lon)
    lat0_rad, lon0_rad = math.radians(lat0), math.radians(lon0)
    inner = (
        np.sin((lat_rad - lat0_rad) / 2.0) ** 2
        + np.cos(lat_rad) * math.cos(lat0_rad)
        * np.sin((lon_rad - lon0_rad) / 2.0) ** 2
    )
    distances = 2.0 * 6371.0088 * np.arcsin(np.sqrt(np.clip(inner, 0.0, 1.0)))
    return pd.DataFrame(
        {
            "point_id": points,
            "outcome": dataset.types[points],
            "distance_km": distances,
            "lat": lat,
            "lon": lon,
        }
    )


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.sort(np.asarray(values, dtype=float))
    return values, np.arange(1, len(values) + 1) / len(values) if len(values) else np.array([])


def cluster_detail_figures(
    dataset: LoadedDataset,
    partition: Partition,
    cluster_features: pd.DataFrame,
    *,
    cluster_label: int,
    reasons: list[str],
    seed: int,
) -> list[tuple[str, Figure]]:
    """Base, exact distributions and sampled geography for one selected cluster."""
    region = next(
        region for region in partition.regions
        if int(region["cluster_label"]) == int(cluster_label)
    )
    row = cluster_features.set_index("cluster_label").loc[cluster_label]
    distances = point_distance_frame(dataset, list(region["points"]))
    positive = distances.loc[distances["outcome"] == 1, "distance_km"].to_numpy()
    negative = distances.loc[distances["outcome"] == 0, "distance_km"].to_numpy()

    # Page 1: identity, selection, composition, references and primary verdict.
    base = plt.figure(figsize=(13.333, 7.5), facecolor=SURFACE)
    ax = base.add_axes((0.07, 0.08, 0.86, 0.84))
    ax.axis("off")
    lines = [
        f"Cluster {cluster_label} · ficha-base",
        f"Seleção: {', '.join(reasons)}",
        "",
        f"n={int(row['n'])} · positivos={int(row['p'])} · negativos={int(row['n_neg'])}",
        f"ρ_in={row['rho_in']:.3f} · ρ_global={row['rho_global']:.3f} · "
        f"ρ_peer={row['rho_peer']:.3f} · ρ_out={row['rho_out']:.3f}",
        f"primary={row['primary_metric']} · score={row['primary_score']:.4g} · "
        f"limiar={row['signif_threshold']:.4g} · evidência relativa={row['evidence_ratio']:.3g}",
        f"status={row['evaluation_status']} · motivo={row['evaluation_reason'] or '—'}",
        f"detection class={row['detection_class'] or 'não avaliado'} · "
        f"direção={row['direction'] or '—'}",
        "",
        f"raio médio={row['distance_mean_km']:.3f} km · p95={row['distance_p95_km']:.3f} km",
        f"separação entre centroides das classes={row['class_centroid_separation_km']:.3f} km",
    ]
    ax.text(0, 1, lines[0], va="top", fontsize=23, weight="bold", color=INK_PRIMARY)
    ax.text(0, 0.88, "\n".join(lines[1:]), va="top", fontsize=13.5,
            color=INK_SECONDARY, linespacing=1.55)

    # Page 2: percentages in shared bins plus bin-independent ECDF and exact summaries.
    distribution = plt.figure(figsize=(13.333, 7.5), facecolor=SURFACE)
    grid = distribution.add_gridspec(2, 2, height_ratios=(4.0, 1.15))
    axes = (distribution.add_subplot(grid[0, 0]), distribution.add_subplot(grid[0, 1]))
    summary_ax = distribution.add_subplot(grid[1, :])
    summary_ax.axis("off")
    bins = shared_histogram_bins(positive, negative)
    if len(positive):
        axes[0].hist(
            positive, bins=bins, weights=np.full(len(positive), 100.0 / len(positive)),
            histtype="step", linewidth=2, color=CATEGORICAL[0], label="positivos",
        )
    if len(negative):
        axes[0].hist(
            negative, bins=bins, weights=np.full(len(negative), 100.0 / len(negative)),
            histtype="step", linewidth=2, color=CATEGORICAL[2], label="negativos",
        )
    axes[0].set_title("Distância ao centroide geral · percentual", loc="left")
    axes[0].set_xlabel("km")
    axes[0].set_ylabel("% de pontos dentro da classe")
    axes[0].legend(frameon=False)
    for values, label, color in (
        (positive, "positivos", CATEGORICAL[0]),
        (negative, "negativos", CATEGORICAL[2]),
    ):
        x, y = _ecdf(values)
        axes[1].step(x, y, where="post", color=color, linewidth=2, label=label)
    axes[1].set_title("ECDF compartilhada", loc="left")
    axes[1].set_xlabel("km")
    axes[1].set_ylabel("fração acumulada dentro da classe")
    axes[1].set_ylim(0, 1)
    axes[1].legend(frameon=False)
    summary_lines = []
    for values, label in ((positive, "positivos"), (negative, "negativos")):
        if not len(values):
            summary_lines.append(f"{label}: classe ausente")
            continue
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        summary_lines.append(
            f"{label}: média={np.mean(values):.3f} km · Q1={q1:.3f} · "
            f"mediana={median:.3f} · Q3={q3:.3f}"
        )
    summary_ax.text(
        0.0, 0.95, "Faixa-resumo da população completa\n" + "\n".join(summary_lines),
        va="top", color=INK_SECONDARY, linespacing=1.35,
    )
    distribution.suptitle(f"Cluster {cluster_label} · distribuição interna exata", x=0.01, ha="left")
    distribution.tight_layout(rect=(0, 0, 1, 0.93))

    # Page 3: the only sampled analytical view.
    sampled = stratified_point_sample(
        list(region["points"]), dataset.types, max_points=5000, seed=seed + cluster_label
    )
    sampled_frame = dataset.df.iloc[sampled]
    sampled_outcomes = dataset.types[sampled]
    geography, geo_ax = plt.subplots(figsize=(13.333, 7.5), facecolor=SURFACE)
    for value, label, color in (
        (1, "positivos", CATEGORICAL[0]), (0, "negativos", CATEGORICAL[2])
    ):
        mask = sampled_outcomes == value
        geo_ax.scatter(sampled_frame.loc[mask, "lon"], sampled_frame.loc[mask, "lat"],
                       s=10, alpha=0.55, color=color, label=label)
    full_frame = dataset.df.iloc[list(region["points"])]
    centroids = [
        (
            float(full_frame["lon"].mean()), float(full_frame["lat"].mean()),
            "centroide geral (população)", "X", INK_PRIMARY,
        )
    ]
    for value, label, marker, color in (
        (1, "centroide positivo (população)", "*", CATEGORICAL[0]),
        (0, "centroide negativo (população)", "D", CATEGORICAL[2]),
    ):
        class_frame = full_frame.loc[dataset.types[list(region["points"])] == value]
        if not class_frame.empty:
            centroids.append(
                (
                    float(class_frame["lon"].mean()), float(class_frame["lat"].mean()),
                    label, marker, color,
                )
            )
    for longitude, latitude, label, marker, color in centroids:
        geo_ax.scatter(
            longitude, latitude, s=180, marker=marker, color=color,
            edgecolors=SURFACE, linewidths=1.2, label=label, zorder=8,
        )
    geo_ax.set_title(
        f"Cluster {cluster_label} · geografia (amostra estratificada {len(sampled)}/{len(region['points'])}, seed={seed + cluster_label})",
        loc="left",
    )
    geo_ax.set_xlabel("longitude")
    geo_ax.set_ylabel("latitude")
    geo_ax.legend(frameon=False)
    geography.text(
        0.01, 0.01,
        "Pontos podem ser amostrados; os três centroides usam sempre a população completa do cluster.",
        color=INK_SECONDARY,
    )
    geography.tight_layout(rect=(0, 0.05, 1, 1))
    for figure in (base, distribution, geography):
        apply_presentation_style(figure)
    return [("base", base), ("distribuicao", distribution), ("geografia", geography)]


def internal_evidence_figures(
    evidence: InternalEvidence, *, cluster_label: int
) -> list[tuple[str, Figure]]:
    """Rate/reference pages for g1/g2 plus a factual Jaccard heatmap."""
    figures = []
    for scale in ("g1", "g2"):
        components = evidence.components[evidence.components["scale"] == scale].copy()
        summary = evidence.scale_summary.set_index("scale").loc[scale]
        fig, ax = plt.subplots(figsize=(13.333, 7.5), facecolor=SURFACE)
        if len(components):
            labels = components["component"].astype(str)
            colors = [CATEGORICAL[2] if value else CATEGORICAL[0]
                      for value in components["is_residue"]]
            ax.bar(labels, components["rho"], color=colors)
            ax.axhline(summary["rho_reference"], color=INK_PRIMARY, linestyle="--",
                       label=f"referência {summary['reference_type']}")
            for idx, (_, row) in enumerate(components.iterrows()):
                ax.text(idx, row["rho"] + 0.03,
                        f"n={int(row['n'])}\nparcela={row['signed_contribution']:.3g}",
                        ha="center")
            ax.legend(frameon=False)
            ax.set_ylim(0, 1)
            ax.set_ylabel("taxa de positivos")
        else:
            ax.axis("off")
            ax.text(
                0.5, 0.5,
                f"{summary['subdivision_status']}\n"
                "Nenhum componente condensado nem resíduo foi produzido nesta granularidade.",
                ha="center", va="center", color=INK_SECONDARY,
            )
        ax.set_title(
            f"Cluster {cluster_label} · {scale} · {summary['subdivision_status']} · "
            f"cobertura condensada={summary['internal_coverage_rate']:.1%} · "
            f"Gini={summary['gini_subcluster']:.3g}",
            loc="left",
        )
        figures.append((f"internal_{scale}", fig))
    if not evidence.overlap.empty:
        matrix = evidence.overlap.pivot(
            index="g1_component", columns="g2_component", values="jaccard"
        )
        fig, ax = plt.subplots(figsize=(13.333, 7.5), facecolor=SURFACE)
        image = ax.imshow(matrix.to_numpy(dtype=float), vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(matrix.columns)), matrix.columns)
        ax.set_yticks(range(len(matrix.index)), matrix.index)
        ax.set_xlabel("componente g2")
        ax.set_ylabel("componente g1")
        ax.set_title(f"Cluster {cluster_label} · sobreposição Jaccard g1 × g2", loc="left")
        fig.colorbar(image, ax=ax, label="Jaccard")
        figures.append(("internal_overlap", fig))
    for _, figure in figures:
        apply_presentation_style(figure)
    return figures


def _component_rows(
    subdivision: InternalSubdivision,
    *,
    scale: str,
    types: np.ndarray,
    parent_n: int,
    rho_reference: float,
) -> list[dict]:
    components: list[tuple[str, list[int]]] = [
        (str(index), points) for index, points in enumerate(subdivision.subclusters)
    ]
    if subdivision.residue:
        components.append(("residue", subdivision.residue))
    rows = []
    for component, points in components:
        n = len(points)
        p = int(types[points].sum()) if n else 0
        rho = p / n if n else float("nan")
        contribution = (n / parent_n) * (rho - rho_reference) if parent_n else float("nan")
        rows.append(
            {
                "scale": scale,
                "component": component,
                "is_residue": component == "residue",
                "point_ids": json.dumps(points),
                "n": n,
                "p": p,
                "n_neg": n - p,
                "rho": rho,
                "signed_contribution": contribution,
                "contribution_magnitude": abs(contribution),
                "direction": (
                    "negative" if contribution < 0 else "positive" if contribution > 0 else "neutral"
                ),
            }
        )
    magnitude_sum = sum(float(row["contribution_magnitude"]) for row in rows)
    for row in rows:
        row["magnitude_share"] = (
            float(row["contribution_magnitude"]) / magnitude_sum
            if magnitude_sum > 0 else float("nan")
        )
    return rows


def _overlap_frame(components: pd.DataFrame) -> pd.DataFrame:
    g1 = components[components["scale"] == "g1"]
    g2 = components[components["scale"] == "g2"]
    columns = [
        "g1_component", "g2_component", "intersection_n", "jaccard",
        "g1_rho", "g2_rho", "rho_change", "g1_direction", "g2_direction",
        "direction_changed", "g1_signed_contribution", "g2_signed_contribution",
        "contribution_change", "g1_residue_n", "g2_residue_n",
        "residue_n_change", "is_best_match_for_g1",
    ]
    rows = []
    g1_residue_n = int(g1.loc[g1["is_residue"], "n"].sum()) if not g1.empty else 0
    g2_residue_n = int(g2.loc[g2["is_residue"], "n"].sum()) if not g2.empty else 0
    for _, left in g1.iterrows():
        left_ids = set(json.loads(left["point_ids"]))
        candidates = []
        for _, right in g2.iterrows():
            right_ids = set(json.loads(right["point_ids"]))
            intersection = len(left_ids.intersection(right_ids))
            union = len(left_ids.union(right_ids))
            jaccard = intersection / union if union else float("nan")
            candidates.append(
                {
                    "g1_component": left["component"],
                    "g2_component": right["component"],
                    "intersection_n": intersection,
                    "jaccard": jaccard,
                    "g1_rho": float(left["rho"]),
                    "g2_rho": float(right["rho"]),
                    "rho_change": float(right["rho"]) - float(left["rho"]),
                    "g1_direction": left["direction"],
                    "g2_direction": right["direction"],
                    "direction_changed": left["direction"] != right["direction"],
                    "g1_signed_contribution": float(left["signed_contribution"]),
                    "g2_signed_contribution": float(right["signed_contribution"]),
                    "contribution_change": (
                        float(right["signed_contribution"]) - float(left["signed_contribution"])
                    ),
                    "g1_residue_n": g1_residue_n,
                    "g2_residue_n": g2_residue_n,
                    "residue_n_change": g2_residue_n - g1_residue_n,
                }
            )
        best_index = (
            max(range(len(candidates)), key=lambda idx: candidates[idx]["jaccard"])
            if candidates else None
        )
        for idx, candidate in enumerate(candidates):
            candidate["is_best_match_for_g1"] = idx == best_index
            rows.append(candidate)
    return pd.DataFrame(rows, columns=columns)


def analyze_cluster_internal(
    dataset: LoadedDataset,
    partition: Partition,
    cluster_features: pd.DataFrame,
    *,
    cluster_label: int,
    primary_metric: str,
    subdividers: dict[str, Callable[[list[int]], InternalSubdivision]] | None = None,
) -> InternalEvidence:
    """Two-scale components relative to the parent's declared primary reference."""
    region = next(
        (region for region in partition.regions if int(region["cluster_label"]) == cluster_label),
        None,
    )
    if region is None:
        raise ValueError(f"Unknown cluster label: {cluster_label}")
    parent = cluster_features.set_index("cluster_label").loc[cluster_label]
    capabilities = get_primary_capabilities(primary_metric)
    rho_reference = float(
        parent["rho_peer"] if capabilities.rate_reference == "peers" else parent["rho_out"]
    )
    base_size = int(partition.params.get("min_cluster_size", 25))
    min_samples = int(partition.params.get("min_samples", 60))
    if subdividers is None:
        subdividers = {
            "g1": lambda points: diagnostic_density_subdivision(
                dataset.df, points, base_size, min_samples
            ),
            "g2": lambda points: diagnostic_density_subdivision(
                dataset.df, points, 2 * base_size, min_samples
            ),
        }

    rows = []
    summaries = []
    for scale in ("g1", "g2"):
        subdivision = subdividers[scale](list(region["points"]))
        scale_rows = _component_rows(
            subdivision,
            scale=scale,
            types=dataset.types,
            parent_n=len(region["points"]),
            rho_reference=rho_reference,
        )
        rows.extend(scale_rows)
        rates = [
            dataset.types[group].mean() for group in subdivision.subclusters if len(group)
        ]
        summaries.append(
            {
                "scale": scale,
                "min_cluster_size": subdivision.min_cluster_size,
                "n_subclusters": len(subdivision.subclusters),
                "residue_n": subdivision.residue_n,
                "internal_coverage_rate": subdivision.coverage_rate,
                "gini_subcluster": calculate_gini(rates) if rates else float("nan"),
                "subdivision_status": subdivision.status,
                "rho_parent": float(parent["rho_in"]),
                "rho_reference": rho_reference,
                "reference_type": capabilities.rate_reference,
                "reference_reason": (
                    None if math.isfinite(rho_reference) else "referencia_de_taxa_nao_finita"
                ),
            }
        )
    components = pd.DataFrame(rows)
    scale_summary = pd.DataFrame(summaries)
    if len(scale_summary) == 2:
        g1_residue = int(
            scale_summary.loc[scale_summary["scale"] == "g1", "residue_n"].iloc[0]
        )
        scale_summary["residue_n_change_from_g1"] = (
            scale_summary["residue_n"] - g1_residue
        )
    return InternalEvidence(
        components=components,
        scale_summary=scale_summary,
        overlap=_overlap_frame(components),
    )
