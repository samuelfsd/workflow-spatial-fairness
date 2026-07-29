"""Build a markdown table comparing organic clustering vs the authors' baselines.

Reads the per-dataset CSVs produced by the experiment commands
(`unrestricted_{ds}_regions.csv`, `one_partitioning_{ds}.csv`,
`hdbscan_{ds}_comparison.csv`) and prints a metrics-by-method markdown table
ready to paste into docs/PIPELINE.md.

Usage:
    uv run python src/comparison_table.py --dataset lar --out outputs
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def _is_num(value) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def _fmt_int(value) -> str:
    return f"{int(value):,}".replace(",", ".") if _is_num(value) else "—"


def _fmt_float(value, digits: int = 1) -> str:
    return f"{value:.{digits}f}" if _is_num(value) else "—"


def _fmt_pct(value) -> str:
    return f"{value * 100:.1f}%" if _is_num(value) else "—"


def load_methods(out_dir: Path, dataset: str, include: set[str]) -> list[tuple[str, pd.Series]]:
    """Return (column label, result row) pairs for every requested method found on disk."""
    methods: list[tuple[str, pd.Series]] = []

    unrestricted = out_dir / f"unrestricted_{dataset}_regions.csv"
    if "kmeans" in include and unrestricted.exists():
        frame = pd.read_csv(unrestricted)
        for _, row in frame[frame["method"] == "kmeans_scan"].tail(1).iterrows():
            methods.append(("Quadrados KMeans (autores)", row))

    one_partitioning = out_dir / f"one_partitioning_{dataset}.csv"
    if "grid" in include and one_partitioning.exists():
        for _, row in pd.read_csv(one_partitioning).iterrows():
            params = json.loads(row["params"])
            methods.append((f"Grade {params['lon_n']}×{params['lat_n']} (autores)", row))

    hdbscan = out_dir / f"hdbscan_{dataset}_comparison.csv"
    if "hdbscan" in include and hdbscan.exists():
        frame = pd.read_csv(hdbscan).drop_duplicates(subset="params", keep="last")
        for _, row in frame.iterrows():
            params = json.loads(row["params"])
            label = f"HDBSCAN frac={params['min_cluster_frac']} (este trabalho)"
            methods.append((label, row))

    return methods


def build_table(methods: list[tuple[str, pd.Series]]) -> str:
    def significant(row: pd.Series) -> str:
        text = _fmt_int(row.get("significant_regions"))
        non_overlapping = row.get("non_overlapping_regions")
        if _is_num(non_overlapping):
            text += f" → {_fmt_int(non_overlapping)} sem sobreposição"
        return text

    def best_region(row: pd.Series) -> str:
        n, rate = row.get("best_region_n"), row.get("best_region_rate")
        if not _is_num(n):
            return "—"
        return f"{_fmt_int(n)} · {_fmt_pct(rate)}"

    def unassigned(row: pd.Series) -> str:
        noise_rate = row.get("noise_rate")
        return _fmt_pct(noise_rate) if _is_num(noise_rate) else "não medido"

    rows = [
        ("Regiões avaliadas", lambda r: _fmt_int(r.get("n_regions"))),
        ("SUL máximo", lambda r: _fmt_float(r.get("max_sul"))),
        ("Limiar Monte Carlo", lambda r: _fmt_float(r.get("signif_threshold"))),
        ("Regiões significativas", significant),
        ("Melhor região (n · taxa)", best_region),
        ("MeanVar", lambda r: _fmt_float(r.get("meanvar"), digits=4)),
        ("Gini", lambda r: _fmt_float(r.get("gini"), digits=3)),
        ("Pontos não atribuídos", unassigned),
    ]

    labels = [label for label, _ in methods]
    lines = [
        "| Métrica | " + " | ".join(labels) + " |",
        "|---" * (len(labels) + 1) + "|",
    ]
    for name, extract in rows:
        cells = [extract(row) for _, row in methods]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Markdown comparison table from experiment CSVs.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--methods",
        default="kmeans,grid,hdbscan",
        help="Comma-separated methods to include: kmeans, grid, hdbscan.",
    )
    args = parser.parse_args()

    include = {name.strip() for name in args.methods.split(",") if name.strip()}
    methods = load_methods(args.out, args.dataset, include)
    if not methods:
        raise SystemExit(
            f"No result CSVs found for dataset '{args.dataset}' in {args.out}/. "
            "Run the experiment commands first."
        )

    first = methods[0][1]
    print(f"### {args.dataset} (N = {_fmt_int(first['N'])}, taxa global {_fmt_pct(first['global_rate'])})\n")
    print(build_table(methods))


if __name__ == "__main__":
    main()
