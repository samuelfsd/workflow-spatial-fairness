"""Transactional materialization of the cluster exploration report."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from data_loading import LoadedDataset, load_dataset
from exploration import ExplorationTables, build_exploration_tables
from exploration import partition_from_snapshot
from exploration_details import (
    analyze_cluster_internal,
    cluster_detail_figures,
    internal_evidence_figures,
    large_cluster_labels,
    paginate_cluster_bundles,
    select_clusters,
)
from exploration_figures import core_figures
from exploration_supplements import factual_summary, spearman_pairwise, supplementary_figures
from figures import close, save_figure, save_pdf_report
from run_snapshot import MANIFEST_NAME, load_run_snapshot, mark_exploration_complete


def _write_tables(tables: ExplorationTables, output_dir: Path) -> None:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "cluster_features", "coverage_audit", "distribution_summary",
        "detection_summary", "rankings", "heatmap",
    ):
        getattr(tables, name).to_csv(tables_dir / f"{name}.csv", index=False)


def _publish_directory(staging: Path, destination: Path) -> None:
    """Atomically expose a complete directory while retaining the last good one."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = destination.with_name(f".{destination.name}.backup-{uuid.uuid4().hex}")
    moved_previous = False
    try:
        if destination.exists():
            os.replace(destination, backup)
            moved_previous = True
        os.replace(staging, destination)
    except Exception:
        if moved_previous and not destination.exists() and backup.exists():
            os.replace(backup, destination)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)


def _recover_interrupted_publication(destination: Path) -> list[str]:
    """Conservatively recover one interrupted staging/backup transaction."""
    parent = destination.parent
    if not parent.exists():
        return []
    stagings = sorted(parent.glob(f".{destination.name}.staging-*"))
    backups = sorted(parent.glob(f".{destination.name}.backup-*"))
    if len(stagings) > 1 or len(backups) > 1:
        raise RuntimeError(
            f"Ambiguous interrupted publication for {destination}: "
            f"{len(stagings)} stagings, {len(backups)} backups"
        )
    actions = []
    staging = stagings[0] if stagings else None
    backup = backups[0] if backups else None
    if destination.exists():
        if staging:
            shutil.rmtree(staging)
            actions.append("staging_obsoleto_removido")
        if backup:
            shutil.rmtree(backup)
            actions.append("backup_obsoleto_removido")
        return actions

    staging_complete = False
    if staging and (staging / "report_manifest.json").exists():
        try:
            staging_complete = bool(
                json.loads((staging / "report_manifest.json").read_text(encoding="utf-8"))
                .get("complete")
            )
        except (OSError, json.JSONDecodeError):
            staging_complete = False
    if staging_complete:
        os.replace(staging, destination)
        actions.append("staging_completo_publicado")
        if backup:
            shutil.rmtree(backup)
            actions.append("backup_substituido_removido")
    elif backup:
        os.replace(backup, destination)
        actions.append("ultimo_backup_restaurado")
        if staging:
            shutil.rmtree(staging)
            actions.append("staging_incompleto_removido")
    elif staging:
        raise RuntimeError(
            f"Interrupted publication has only an incomplete staging directory: {staging}"
        )
    return actions


def generate_cluster_exploration(
    run_dir: Path,
    *,
    primary_metric: str,
    profile: str = "full",
    output_dir: Path | None = None,
    dataset: LoadedDataset | None = None,
    detail_selection: str = "auto",
    custom_families: set[str] | None = None,
    progress: Callable[[str], None] | None = None,
    core_figure_builder: Callable = core_figures,
) -> Path:
    """Regenerate one primary reading from a validated snapshot only."""
    if profile not in {"full", "core", "custom", "none"}:
        raise ValueError("profile must be one of: full, core, custom, none")
    allowed_families = {"details", "multiscale", "supplements"}
    families = set(custom_families or ())
    if profile == "full":
        families = allowed_families
    elif profile in {"core", "none"}:
        families = set()
    elif not families or not families.issubset(allowed_families):
        raise ValueError(
            "custom profile requires --families from: details,multiscale,supplements"
        )
    if "multiscale" in families:
        families.add("details")
    notify = progress or (lambda message: None)
    run_dir = Path(run_dir)
    if dataset is None:
        manifest_path = run_dir / MANIFEST_NAME
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Run sem snapshot versionado: {manifest_path}. Regeneração aproximada é recusada."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        dataset = load_dataset(str(manifest.get("dataset", {}).get("name", "")))

    # Validation and all calculations precede staging: a bad snapshot cannot
    # create even an empty final report directory.
    notify("Validando snapshot versionado")
    snapshot = load_run_snapshot(run_dir, dataset)
    notify("Construindo tabelas canônicas")
    tables = build_exploration_tables(dataset, snapshot, primary_metric)
    destination = Path(output_dir) if output_dir else run_dir / "exploration" / primary_metric
    if profile == "none":
        return destination

    recovery_actions = _recover_interrupted_publication(destination)
    for action in recovery_actions:
        notify(f"Recuperação transacional: {action}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    figures = []
    try:
        selected = pd.DataFrame(columns=["cluster_label", "reason"])
        _write_tables(tables, staging)
        notify("Renderizando as oito páginas globais")
        global_dir = staging / "figures" / "global"
        figure_pairs = core_figure_builder(tables, snapshot.manifest, primary_metric)
        figures = [figure for _, figure in figure_pairs]
        for name, figure in figure_pairs:
            save_figure(figure, global_dir / name)
        save_pdf_report(figures, staging / "cluster_exploration.pdf")
        close(*figures)
        figures = []
        if "details" in families:
            partition = partition_from_snapshot(snapshot, dataset.n_total)
            selected = select_clusters(tables.cluster_features, detail_selection)
            selected.to_csv(staging / "tables" / "selected_clusters.csv", index=False)
            selected_labels = selected["cluster_label"].drop_duplicates().astype(int).tolist()
            reasons_by_label = (
                selected.groupby("cluster_label")["reason"].apply(list).to_dict()
                if not selected.empty else {}
            )
            large_labels = large_cluster_labels(tables.cluster_features)
            internal_components = []
            internal_summaries = []
            internal_overlaps = []
            details_dir = staging / "figures" / "details"
            evidence_by_label = {}
            if "multiscale" in families:
                for label in selected_labels:
                    if label not in large_labels:
                        continue
                    notify(f"Calculando inspeção multiescala do cluster {label}")
                    evidence_by_label[label] = analyze_cluster_internal(
                        dataset,
                        partition,
                        tables.cluster_features,
                        cluster_label=label,
                        primary_metric=primary_metric,
                    )
            page_index = paginate_cluster_bundles(
                {
                    label: 3
                    + (
                        2 + (1 if not evidence_by_label[label].overlap.empty else 0)
                        if label in evidence_by_label else 0
                    )
                    for label in selected_labels
                },
                target_pages=50,
            )
            page_index.to_csv(staging / "tables" / "cluster_detail_index.csv", index=False)
            for volume, volume_rows in page_index.groupby("volume"):
                volume_figures = []
                try:
                    for label in volume_rows["cluster_label"].astype(int):
                        notify(f"Renderizando detalhes do cluster {label}")
                        bundle = cluster_detail_figures(
                            dataset,
                            partition,
                            tables.cluster_features,
                            cluster_label=label,
                            reasons=reasons_by_label[label],
                            seed=int(snapshot.manifest["run"]["seed"]),
                        )
                        if label in evidence_by_label:
                            evidence = evidence_by_label[label]
                            for frame in (
                                evidence.components,
                                evidence.scale_summary,
                                evidence.overlap,
                            ):
                                frame.insert(0, "cluster_label", label)
                            internal_components.append(evidence.components)
                            internal_summaries.append(evidence.scale_summary)
                            internal_overlaps.append(evidence.overlap)
                            bundle.extend(
                                internal_evidence_figures(evidence, cluster_label=label)
                            )
                        for name, figure in bundle:
                            save_figure(
                                figure,
                                details_dir / f"cluster_{label}_{name}",
                            )
                        volume_figures.extend(figure for _, figure in bundle)
                    save_pdf_report(
                        volume_figures,
                        staging / f"cluster_details_{int(volume):03d}.pdf",
                    )
                finally:
                    if volume_figures:
                        close(*volume_figures)

            empty_components = pd.DataFrame(
                columns=["cluster_label", "scale", "component", "is_residue", "point_ids",
                         "n", "p", "n_neg", "rho", "signed_contribution",
                         "contribution_magnitude", "direction", "magnitude_share"]
            )
            components_frame = (
                pd.concat(internal_components, ignore_index=True)
                if internal_components else empty_components
            )
            components_frame.to_csv(
                staging / "tables" / "internal_subclusters.csv", index=False
            )
            components_frame.drop(columns=["point_ids"], errors="ignore").to_csv(
                staging / "tables" / "internal_contributions.csv", index=False
            )
            (
                pd.concat(internal_summaries, ignore_index=True)
                if internal_summaries else pd.DataFrame()
            ).to_csv(staging / "tables" / "internal_scale_summary.csv", index=False)
            overlaps_frame = (
                pd.concat(internal_overlaps, ignore_index=True)
                if internal_overlaps else pd.DataFrame()
            )
            overlaps_frame.to_csv(
                staging / "tables" / "multiscale_overlap.csv", index=False
            )
            for value_column, filename in (
                ("intersection_n", "multiscale_intersection_matrix.csv"),
                ("jaccard", "multiscale_jaccard_matrix.csv"),
            ):
                matrix = (
                    overlaps_frame.pivot_table(
                        index=["cluster_label", "g1_component"],
                        columns="g2_component",
                        values=value_column,
                        aggfunc="first",
                        dropna=False,
                    ).reset_index()
                    if not overlaps_frame.empty else pd.DataFrame()
                )
                matrix.to_csv(staging / "tables" / filename, index=False)

        if "supplements" in families:
            supplementary_dir = staging / "figures" / "supplementary"
            supplementary = supplementary_figures(tables.cluster_features)
            try:
                for name, figure in supplementary:
                    save_figure(figure, supplementary_dir / name)
                save_pdf_report(
                    [figure for _, figure in supplementary],
                    staging / "cluster_supplements.pdf",
                )
            finally:
                close(*(figure for _, figure in supplementary))
            correlation_metrics = [
                "n", "rho_in", "internal_predominance", "global_deviation",
                "peer_deviation", "distance_mean_km", "distance_p95_km",
                "class_centroid_separation_km", "evidence_ratio",
            ]
            correlations = spearman_pairwise(
                tables.cluster_features, correlation_metrics
            )
            if correlations.empty:
                (staging / "tables" / "correlations_omitted.json").write_text(
                    json.dumps(
                        {"reason": "menos_de_10_valores_validos_por_par"},
                        ensure_ascii=False,
                        indent=2,
                    ) + "\n",
                    encoding="utf-8",
                )
            else:
                correlations.to_csv(staging / "tables" / "correlations.csv", index=False)
            (staging / "analysis_summary.json").write_text(
                json.dumps(
                    factual_summary(tables.cluster_features, selected),
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ) + "\n",
                encoding="utf-8",
            )
        (staging / "report_manifest.json").write_text(
            json.dumps(
                {
                    "snapshot_schema_version": snapshot.manifest["schema_version"],
                    "primary_metric": primary_metric,
                    "profile": profile,
                    "complete": True,
                    "n_clusters": len(tables.cluster_features),
                    "detail_selection": detail_selection,
                    "families": sorted(families),
                    "recovery_actions": recovery_actions,
                },
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        _publish_directory(staging, destination)
        mark_exploration_complete(
            run_dir,
            primary_metric=primary_metric,
            profile=profile,
            report_dir=destination,
        )
    finally:
        if figures:
            close(*figures)
        if staging.exists():
            shutil.rmtree(staging)
    return destination
