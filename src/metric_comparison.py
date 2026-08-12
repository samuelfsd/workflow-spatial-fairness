"""Explicit comparison of two eligible primaries on one immutable snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data_loading import LoadedDataset
from exploration import build_exploration_tables
from figure_style import apply_presentation_style
from run_snapshot import RunSnapshot
from figures import close, save_figure
from palette import CATEGORICAL, SURFACE


@dataclass
class PrimaryComparison:
    joint: pd.DataFrame
    concordance: pd.DataFrame
    sets: pd.DataFrame


def compare_primary_metrics(
    dataset: LoadedDataset,
    snapshot: RunSnapshot,
    first: str,
    second: str,
) -> PrimaryComparison:
    """Compare derived readings; never create a hybrid class or choose a winner."""
    first_frame = build_exploration_tables(dataset, snapshot, first).cluster_features
    second_frame = build_exploration_tables(dataset, snapshot, second).cluster_features
    first_columns = {
        "primary_score": f"{first}_score",
        "signif_threshold": f"{first}_threshold",
        "evidence_ratio": f"{first}_evidence_ratio",
        "detection_class": f"{first}_detection_class",
        "evaluation_status": f"{first}_evaluation_status",
        "direction": f"{first}_direction",
    }
    second_columns = {
        "primary_score": f"{second}_score",
        "signif_threshold": f"{second}_threshold",
        "evidence_ratio": f"{second}_evidence_ratio",
        "detection_class": f"{second}_detection_class",
        "evaluation_status": f"{second}_evaluation_status",
        "direction": f"{second}_direction",
    }
    shared = ["cluster_label", "rho_in", "rho_peer", "rho_out"]
    joint = first_frame[shared + list(first_columns)].rename(columns=first_columns).merge(
        second_frame[["cluster_label"] + list(second_columns)].rename(columns=second_columns),
        on="cluster_label",
        validate="one_to_one",
    )

    concordance = pd.crosstab(
        joint[f"{first}_detection_class"].fillna("não avaliado"),
        joint[f"{second}_detection_class"].fillna("não avaliado"),
        dropna=False,
    ).rename_axis(index=f"{first}_class", columns=f"{second}_class")

    first_detected = joint[f"{first}_detection_class"].isin(["negative", "positive"])
    second_detected = joint[f"{second}_detection_class"].isin(["negative", "positive"])
    membership = {
        "ambas": first_detected & second_detected,
        f"somente_{first}": first_detected & ~second_detected,
        f"somente_{second}": ~first_detected & second_detected,
        "nenhuma": ~first_detected & ~second_detected,
    }
    sets = pd.DataFrame(
        [
            {
                "set": name,
                "n_clusters": int(mask.sum()),
                "cluster_labels": ",".join(
                    str(int(label)) for label in joint.loc[mask, "cluster_label"]
                ),
            }
            for name, mask in membership.items()
        ]
    )
    return PrimaryComparison(joint=joint, concordance=concordance, sets=sets)


def write_primary_comparison(
    comparison: PrimaryComparison,
    output_dir: Path,
    *,
    first: str,
    second: str,
) -> Path:
    """Publish explicit comparison evidence, with no combined verdict."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.joint.to_csv(output_dir / "primary_comparison.csv", index=False)
    comparison.concordance.to_csv(output_dir / "detection_concordance.csv")
    comparison.sets.to_csv(output_dir / "detection_sets.csv", index=False)

    fig, ax = plt.subplots(figsize=(13.333, 7.5), facecolor=SURFACE)
    joint = comparison.joint.dropna(
        subset=[f"{first}_evidence_ratio", f"{second}_evidence_ratio"]
    )
    for index, direction in enumerate(("negative", "positive", "neutral")):
        subset = joint[joint[f"{first}_direction"] == direction]
        if subset.empty:
            continue
        ax.scatter(
            subset[f"{first}_evidence_ratio"],
            subset[f"{second}_evidence_ratio"],
            c=CATEGORICAL[index], alpha=0.7, s=60,
            marker={"negative": "v", "positive": "^", "neutral": "o"}[direction],
            label=f"direção {first}: {direction}",
        )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(f"|score|/limiar · {first}")
    ax.set_ylabel(f"|score|/limiar · {second}")
    ax.set_title(f"Evidência relativa · {first} × {second}", loc="left")
    ax.legend(frameon=False)
    fig.text(
        0.01, 0.01,
        "Mesma partição; referências estatísticas distintas. Não há classe híbrida nem escolha de vencedor.",
    )
    apply_presentation_style(fig)
    try:
        save_figure(fig, output_dir / "evidence_scatter")
    finally:
        close(fig)
    return output_dir
