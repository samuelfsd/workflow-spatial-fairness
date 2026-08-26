"""Illustrative LAR/Crime maps rendered from persisted benchmark regions."""

from __future__ import annotations

import html
import json
from pathlib import Path

import folium
import numpy as np
import pandas as pd

from palette import CATEGORICAL, COLOR_NOT_EVALUATED, DETECTION_COLORS


def _json_dict(value) -> dict:
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _geometry(value) -> list[list[float]]:
    if not isinstance(value, str):
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _extent(frame: pd.DataFrame) -> list[list[float]]:
    points = [point for value in frame.get("geometry", []) for point in _geometry(value)]
    if not points:
        return [[-1.0, -1.0], [1.0, 1.0]]
    latitudes = [float(point[0]) for point in points]
    longitudes = [float(point[1]) for point in points]
    return [[min(latitudes), min(longitudes)], [max(latitudes), max(longitudes)]]


def _panel_name(row: pd.Series) -> str:
    if row["metric"] == "meanvar":
        return "grade_meanvar"
    if row["metric"] == "local_z":
        return "hdbscan_local_z"
    if row["method"] == "kmeans_scan":
        return "kmeans_scan_sul"
    return "grade_sul"


def _eligible(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    selected = frame[
        frame["dataset"].isin(["lar", "crime"])
        & frame["metric"].isin(["sul", "meanvar", "local_z"])
    ].copy()
    selected = selected[
        (
            selected["method"].eq("hdbscan")
            & selected["metric"].eq("local_z")
            & selected["protocol"].eq("standardized")
        )
        | (
            ~selected["method"].eq("hdbscan")
            & selected["protocol"].eq("reproduction")
        )
    ]
    if "params" in selected:
        parsed = selected["params"].map(_json_dict)
        direction_ok = parsed.map(lambda value: value.get("direction", "both") == "both")
        frac_ok = parsed.map(lambda value: value.get("min_cluster_frac", 0.005) == 0.005)
        selected = selected[direction_ok & frac_ok]
    selected["panel"] = selected.apply(_panel_name, axis=1)
    return selected


def _tooltip(row: pd.Series) -> str:
    source = row.get("source", "local")
    return (
        f"método={html.escape(str(row.get('metric')))}; sistema={html.escape(str(row.get('partitioning')))}; "
        f"n={int(row.get('n', 0))}; taxa local={float(row.get('rho_in', np.nan)):.3f}; "
        f"taxa de referência={float(row.get('rho_reference', np.nan)):.3f}; fonte={html.escape(str(source))}"
    )


def _color(row: pd.Series) -> str:
    if row.get("metric") == "meanvar":
        return CATEGORICAL[0]
    status = str(row.get("evaluation_status", ""))
    if status not in {"evaluated", "avaliado"}:
        return COLOR_NOT_EVALUATED
    significant = row.get("significant")
    if not (significant is True or significant == 1):
        return DETECTION_COLORS["neutral"]
    detection = row.get("detection_class") or row.get("direction") or "neutral"
    return DETECTION_COLORS.get(str(detection), DETECTION_COLORS["neutral"])


def _render_panel(frame: pd.DataFrame, extent: list[list[float]], path: Path, title: str) -> None:
    center = [(extent[0][0] + extent[1][0]) / 2, (extent[0][1] + extent[1][1]) / 2]
    mapit = folium.Map(location=center, tiles="CartoDB positron")
    for _, row in frame.sort_values(["score", "region_id"], ascending=[False, True], na_position="last").iterrows():
        geometry = _geometry(row.get("geometry"))
        if len(geometry) < 2:
            continue
        color = _color(row)
        if len(geometry) >= 3:
            folium.Polygon(
                geometry, color=color, fill=True, fill_color=color, fill_opacity=.16,
                weight=2, tooltip=_tooltip(row),
            ).add_to(mapit)
        else:
            folium.PolyLine(geometry, color=color, weight=3, tooltip=_tooltip(row)).add_to(mapit)
    meanvar_note = "MeanVar: ranking não direcional; azul não é detection class." if frame["metric"].eq("meanvar").any() else "Cores: detection class calibrada; cinza = nada detectado."
    legend = (
        '<div style="position:fixed;bottom:20px;left:20px;z-index:9999;background:white;padding:10px;border:1px solid #777">'
        f"<b>{html.escape(title)}</b><br>Extensão compartilhada no dataset.<br>{meanvar_note}<br>"
        "Tooltips: método, sistema, n, taxa local, taxa de referência e fonte.</div>"
    )
    mapit.get_root().html.add_child(folium.Element(legend))
    mapit.fit_bounds(extent)
    path.parent.mkdir(parents=True, exist_ok=True)
    mapit.save(str(path))


def render_comparative_maps(canonical: pd.DataFrame, regions: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Render one HTML per method panel; never recompute scores or verdicts."""
    del canonical  # canonical numbers are already denormalized into persisted region rows
    selected = _eligible(regions)
    output_dir = Path(output_dir)
    paths: list[Path] = []
    for dataset in ("lar", "crime"):
        dataset_rows = selected[selected["dataset"].eq(dataset)]
        if dataset_rows.empty:
            continue
        extent = _extent(dataset_rows)
        for panel, panel_rows in dataset_rows.groupby("panel", sort=True):
            path = output_dir / f"{dataset}_{panel}.html"
            _render_panel(panel_rows, extent, path, f"{dataset.upper()} — {panel}")
            paths.append(path)
    return paths
