"""Dataset loading and normalization for the spatial fairness experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    filename: str
    label_column: str
    radii_start: float
    radii_stop: float
    radii_step: float
    fixed_grids: tuple[tuple[int, int], ...]


@dataclass
class LoadedDataset:
    name: str
    df: pd.DataFrame
    types: np.ndarray
    n_total: int
    p_total: int
    radii: np.ndarray
    fixed_grids: tuple[tuple[int, int], ...]

    @property
    def global_rate(self) -> float:
        return self.p_total / self.n_total if self.n_total else 0.0


DATASET_SPECS: dict[str, DatasetSpec] = {
    "lar": DatasetSpec("lar", "LAR.csv", "action_taken", 0.05, 1.01, 0.05, ((100, 50), (25, 12))),
    "crime": DatasetSpec("crime", "Crime.csv", "pred", 0.005, 0.1, 0.005, ((20, 20),)),
    "semisynth": DatasetSpec("semisynth", "Semisynth.csv", "label", 0.01, 0.2, 0.01, ((20, 20),)),
    "synth_fair": DatasetSpec("synth_fair", "Synth_fair.csv", "label", 0.01, 0.2, 0.01, ((20, 20),)),
    "synth_unfair": DatasetSpec("synth_unfair", "Synth_unfair.csv", "label", 0.01, 0.2, 0.01, ((20, 20),)),
}


def dataset_names() -> list[str]:
    return list(DATASET_SPECS)


def load_dataset(name: str) -> LoadedDataset:
    """Load one dataset and normalize its outcome to a binary `outcome` column."""
    if name not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset: {name}")

    spec = DATASET_SPECS[name]
    path = REPO_ROOT / "datasets" / "old" / spec.filename
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, index_col=0)
    df.reset_index(drop=True, inplace=True)

    required = {"lat", "lon", spec.label_column}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{spec.filename} is missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["lat", "lon", spec.label_column]).copy()
    if spec.label_column == "action_taken":
        df["outcome"] = df[spec.label_column].replace({3: 0}).astype(int)
    else:
        df["outcome"] = df[spec.label_column].astype(int)

    if not set(df["outcome"].unique()).issubset({0, 1}):
        raise ValueError(f"{spec.filename} outcome must be binary after normalization")

    df.reset_index(drop=True, inplace=True)
    types = df["outcome"].to_numpy(dtype=int)
    radii = np.round(np.arange(spec.radii_start, spec.radii_stop, spec.radii_step), 10)

    return LoadedDataset(
        name=name,
        df=df,
        types=types,
        n_total=len(df),
        p_total=int(types.sum()),
        radii=radii,
        fixed_grids=spec.fixed_grids,
    )
