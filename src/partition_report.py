"""Descriptive report comparing partition configurations — the ADR-0001 material.

This is **not** a pipeline stage and produces **no fairness verdict** (see
`CONTEXT.md`, "Relatório de partição"): it answers the question the advisors
actually asked — *are these clusters balanced, and what does the size cap do to
them?* — by reading balance, dispersion and compactness for each configuration
side by side.

No Monte Carlo, so it runs in seconds and can be regenerated freely.

Usage:
    uv run python src/partition_report.py --dataset lar --min-cluster-frac 0.005 \
        --clustering hdbscan,capped_hdbscan --max-cluster-sizes 1000,2000 --out outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from clustering.base import Partition
from clustering.hdbscan import effective_min_cluster_size
from clustering.registry import get_partitioner, partitioner_names
from data_loading import dataset_names, load_dataset
from descriptives import (
    cluster_frame,
    compare_configs,
    dataset_balance,
    dispersion_summary,
    expected_sigma_ratio,
    partition_profile,
)
from figures import (
    balance_figure,
    close,
    dispersion_figure,
    profile_figure,
    save_figure,
    save_pdf_report,
)


def _parse_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_ints(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def config_label(method: str, cap: int | None) -> str:
    """Human-readable name for a configuration — the axis label of every figure.

    Deliberately not `hdbscan cap=1000`: the reader should not need to remember
    which partitioner implements which cap mechanism. "teto" (ceiling) rather
    than "cap" because it is a *target*, not a guarantee — the recursive split
    leaves pieces above it whenever density refuses to divide them, and the
    "Teto cumprido" reading is what reports that per dataset. The exact
    partitioner stays in the `method` column of the profile CSV.
    """
    if cap is None:
        return "sem cap (orgânico)"
    mechanism = "cap nativo" if method == "hdbscan" else "redivisão"
    return f"{mechanism} (teto {cap})"


def config_slug(method: str, cap: int | None) -> str:
    """ASCII file-name fragment for a configuration."""
    return method if cap is None else f"{method}_cap{cap}"


def build_configs(
    df: pd.DataFrame,
    *,
    methods: tuple[str, ...],
    min_cluster_frac: float,
    max_cluster_sizes: tuple[int, ...],
    min_samples: int,
) -> tuple[dict[str, tuple[Partition, str, int | None]], list[str]]:
    """Fit one partition per configuration, plus the list of skipped ones.

    Returns `label -> (partition, file slug, cap)` and a list of human-readable
    reasons for configurations that were **not** fitted.

    Three alternatives, which are exactly the ones ADR-0001 weighs: `hdbscan`
    without a cap (the organic default), `hdbscan` with a cap (HDBSCAN's own EOM
    limit — coverage drops), and `capped_hdbscan` (recursive density split —
    coverage preserved).

    A cap at or below `effective_min_cluster_size` is **arithmetically
    impossible** — it asks for clusters both smaller and larger than the same
    number — and HDBSCAN answers with zero clusters. Such a configuration is
    skipped with a reason instead of being reported as a degenerate row: the run
    is not aborted, because one bad cap should not cost a whole clustering sweep.
    """
    floor = effective_min_cluster_size(len(df), min_cluster_frac)
    configs: dict[str, tuple[Partition, str, int | None]] = {}
    skipped: list[str] = []

    for method in methods:
        fit = get_partitioner(method)
        # Plain hdbscan is also reported uncapped (the organic default); the
        # recursive split has no meaning without a cap, so it is cap-only.
        caps: tuple[int | None, ...] = (
            (None, *max_cluster_sizes) if method == "hdbscan" else tuple(max_cluster_sizes)
        )
        for cap in caps:
            label = config_label(method, cap)
            if cap is not None and cap <= floor:
                skipped.append(
                    f"{label}: impossível — o tamanho mínimo de cluster nesta fração é "
                    f"{floor} pontos, então um teto de {cap} descreve conjunto vazio"
                )
                continue
            extra: dict[str, Any] = {"min_samples": min_samples}
            if cap is not None:
                extra["max_cluster_size"] = cap
            partition = fit(df, (min_cluster_frac,), **extra)[0]
            configs[label] = (partition, config_slug(method, cap), cap)

    return configs, skipped


def profile_table(
    configs: dict[str, tuple[Partition, str, int | None]],
    frames: dict[str, pd.DataFrame],
    *,
    n_total: int,
    global_rate: float,
) -> pd.DataFrame:
    """One row per configuration: regions, unassigned share, and headline spreads.

    Keeps the exact partitioner in the `method` column (from `partition_profile`)
    so the human-readable `config` label never costs reproducibility, and reports
    `cap_compliance` — the share of clusters actually at or below the ceiling,
    because the recursive split treats it as a target, not a guarantee.
    """
    rows = []
    for label, (partition, _, cap) in configs.items():
        frame = frames[label]
        summary = dispersion_summary(frame)
        sigma_neg = summary.loc["n_neg", "std"]
        rows.append(
            {
                "config": label,
                **partition_profile(partition, n_total),
                "cap": cap,
                "cap_compliance": (
                    float((frame["n"] <= cap).mean()) if cap is not None and len(frame) else float("nan")
                ),
                "cluster_size_cv": summary.loc["n", "cv"],
                "cluster_size_min": summary.loc["n", "min"],
                "cluster_size_max": summary.loc["n", "max"],
                "rho_sigma": summary.loc["rho", "std"],
                "raio_medio_km_mean": summary.loc["raio_medio_km", "mean"],
                "sigma_ratio_p_over_neg": (
                    summary.loc["p", "std"] / sigma_neg if sigma_neg else float("nan")
                ),
                "sigma_ratio_expected": expected_sigma_ratio(global_rate),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(profile: pd.DataFrame) -> str:
    """Render the profile as a markdown table ready to paste into the docs."""
    rows = [
        ("Clusters", "n_regions", "{:.0f}"),
        ("Menor cluster (n)", "cluster_size_min", "{:.0f}"),
        ("Maior cluster (n)", "cluster_size_max", "{:.0f}"),
        ("CV do tamanho", "cluster_size_cv", "{:.2f}"),
        ("σ das taxas", "rho_sigma", "{:.4f}"),
        ("Raio médio (km)", "raio_medio_km_mean", "{:.1f}"),
        ("σ(p)/σ(neg) observado", "sigma_ratio_p_over_neg", "{:.2f}"),
        ("σ(p)/σ(neg) esperado", "sigma_ratio_expected", "{:.2f}"),
        ("Não atribuídos", "noise_rate", "{:.1%}"),
        ("**Teto cumprido**", "cap_compliance", "{:.0%}"),
        ("Clusters acima do teto", "over_cap", "{:.0f}"),
        ("Divisões forçadas", "forced_uncapped", "{:.0f}"),
    ]
    labels = list(profile["config"])
    lines = [
        "| Leitura | " + " | ".join(labels) + " |",
        "|---" * (len(labels) + 1) + "|",
    ]
    for title, column, fmt in rows:
        cells = []
        for _, row in profile.iterrows():
            value = row[column]
            cells.append("—" if pd.isna(value) else fmt.format(value))
        lines.append(f"| {title} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=dataset_names(), default="lar")
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument("--min-cluster-frac", type=float, default=0.005)
    parser.add_argument(
        "--clustering",
        type=_parse_list,
        default=("hdbscan", "capped_hdbscan"),
        help=f"Comma-separated partitioners to compare. Available: {partitioner_names()}",
    )
    parser.add_argument(
        "--max-cluster-sizes",
        type=_parse_ints,
        default=(1000, 2000),
        help="Comma-separated size caps to sweep (ADR-0001). Empty = no cap at all.",
    )
    parser.add_argument("--hdbscan-min-samples", type=int, default=60)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    print(f"Dataset {dataset.name}: " + ", ".join(f"{k}={v}" for k, v in dataset_balance(dataset.types).items()))

    floor = effective_min_cluster_size(dataset.n_total, args.min_cluster_frac)
    print(f"Tamanho mínimo de cluster nesta fração: {floor} pontos (piso de qualquer teto)")

    configs, skipped = build_configs(
        dataset.df,
        methods=args.clustering,
        min_cluster_frac=args.min_cluster_frac,
        max_cluster_sizes=args.max_cluster_sizes,
        min_samples=args.hdbscan_min_samples,
    )
    for reason in skipped:
        print(f"  ⚠ pulada — {reason}")
    if not configs:
        raise SystemExit(
            "Nenhuma configuração de partição para reportar — verifique "
            "--clustering/--max-cluster-sizes (o teto precisa ser maior que o piso acima)."
        )

    frames = {
        label: cluster_frame(dataset.df, partition, dataset.types)
        for label, (partition, _, _) in configs.items()
    }
    dispersion = compare_configs(frames)
    profile = profile_table(
        configs, frames, n_total=dataset.n_total, global_rate=dataset.global_rate
    )

    args.out.mkdir(parents=True, exist_ok=True)
    clusters = pd.concat(
        [frame.assign(config=label) for label, frame in frames.items()], ignore_index=True
    )
    clusters.to_csv(args.out / f"partition_report_{dataset.name}_clusters.csv", index=False)
    dispersion.to_csv(args.out / f"partition_report_{dataset.name}_dispersion.csv")
    profile.to_csv(args.out / f"partition_report_{dataset.name}_profile.csv", index=False)

    figures_dir = args.out / "figures"
    # The profile leads the report: it is the figure the cap decision is read from.
    figures = [profile_figure(profile, dataset=dataset.name)]
    save_figure(figures[0], figures_dir / f"partition_report_{dataset.name}_profile")

    dispersion_fig = dispersion_figure(dispersion, dataset=dataset.name)
    save_figure(dispersion_fig, figures_dir / f"partition_report_{dataset.name}_dispersion")
    figures.append(dispersion_fig)

    for label, frame in frames.items():
        slug = configs[label][1]  # (partition, slug, cap)
        figure = balance_figure(frame, dataset=dataset.name, method=label)
        save_figure(figure, figures_dir / f"partition_report_{dataset.name}_balance_{slug}")
        figures.append(figure)

    save_pdf_report(figures, figures_dir / f"partition_report_{dataset.name}.pdf")
    close(*figures)

    print()
    print(f"### Relatório de partição — {dataset.name} (frac {args.min_cluster_frac})\n")
    print(markdown_table(profile))

    # A configuration that stops covering the map stops being an audit; say so
    # next to the number instead of leaving it to be noticed. The loss is only
    # *the ceiling's cost* when it exceeds the uncapped baseline — an uncapped
    # partition that already leaves half the map out is a property of the data.
    uncapped = profile[profile["cap"].isna()]
    baseline_noise = float(uncapped["noise_rate"].iloc[0]) if len(uncapped) else float("nan")

    for _, row in profile.iterrows():
        if row["noise_rate"] > 0.5:
            extra = ""
            if pd.notna(row["cap"]) and pd.notna(baseline_noise):
                delta = row["noise_rate"] - baseline_noise
                if delta > 0.01:
                    extra = (
                        f" — {delta:.1%} acima da partição sem teto, e essa parte é o custo do teto"
                    )
            print(
                f"\n⚠ {row['config']}: {row['noise_rate']:.1%} dos pontos ficam fora da comparação{extra}."
            )
        if pd.notna(row.get("cap_compliance")) and row["cap_compliance"] < 1.0:
            print(
                f"\n⚠ {row['config']}: só {row['cap_compliance']:.0%} dos clusters respeitam o teto "
                f"(o maior tem {row['cluster_size_max']:.0f} pontos) — a densidade se recusa a dividir "
                f"o resto, então o teto é meta, não garantia."
            )

    print(f"\nArquivos e figuras escritos em {args.out}")


if __name__ == "__main__":
    main()
