"""Synthetic control that separates the two reference frames.

`Synth_unfair` plants a pocket that deviates from the global rate **and** from its
neighbourhood, so both the SUL and the local z-score fire and nothing is
discriminated. This control plants **two** pockets far apart:

- **local pocket** — its rate *is* the global rate, but its neighbours sit well
  above it. The global baseline is blind **by construction** (the log-likelihood
  ratio collapses to zero), while the peer baseline sees it clearly. This is what
  makes the peer baseline *necessary* rather than merely different.
- **global pocket** — its rate deviates from the global one and its neighbours
  follow it down. The SUL fires; the local z-score stays quiet. This is what
  shows the local z-score is not "a more sensitive SUL" — it answers another
  question, and the two are complementary.

Blob rates are calibrated so the map-wide rate is *exactly* the declared one, and
positives are assigned by exact count (not by a coin flip), so the ground truth
is deterministic rather than approximate.

Usage:
    uv run python src/synth_data.py --out datasets/synth_local/data
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Blob:
    """One dense blob of points: where it sits, how many, and what share is positive."""

    lat: float
    lon: float
    n: int
    rate: float
    role: str


@dataclass(frozen=True)
class LocalControlSpec:
    """Declared ground truth of the control — the table the validation fills in."""

    n_total: int
    global_rate: float
    local_pocket_rate: float
    local_peer_rate: float
    global_pocket_rate: float
    global_peer_rate: float
    filler_rate: float
    blob_n: int
    blob_spread_deg: float
    blobs: tuple[Blob, ...] = field(repr=False)


def _hex_ring(lat: float, lon: float, radius: float) -> list[tuple[float, float]]:
    """Six points around a centre — a tight neighbourhood whose Delaunay
    triangulation makes the centre adjacent to all of them."""
    angles = np.deg2rad(np.arange(0, 360, 60))
    return [(lat + radius * float(np.sin(a)), lon + radius * float(np.cos(a))) for a in angles]


def _square_ring(lat: float, lon: float, radius: float) -> list[tuple[float, float]]:
    return [(lat + radius, lon), (lat - radius, lon), (lat, lon + radius), (lat, lon - radius)]


def _build_spec() -> LocalControlSpec:
    blob_n = 800
    local_center = (34.0, -118.0)
    global_center = (37.0, -122.0)
    ring_radius = 0.6

    local_pocket_rate, local_peer_rate = 0.50, 0.75
    global_pocket_rate = 0.30
    filler_rate = 0.40

    blobs = [Blob(*local_center, blob_n, local_pocket_rate, "local_pocket")]
    blobs += [Blob(lat, lon, blob_n, local_peer_rate, "local_peer") for lat, lon in _hex_ring(*local_center, ring_radius)]
    blobs += [Blob(*global_center, blob_n, global_pocket_rate, "global_pocket")]
    blobs += [
        Blob(lat, lon, blob_n, global_pocket_rate, "global_peer")
        for lat, lon in _square_ring(*global_center, ring_radius)
    ]
    # Filler blobs, far from both neighbourhoods, calibrated to pull the map-wide
    # rate to exactly 0.50 (6800 positives over 13600 points).
    blobs += [Blob(40.0, -100.0 + 1.5 * i, blob_n, filler_rate, "filler") for i in range(5)]

    n_total = sum(blob.n for blob in blobs)
    p_total = sum(round(blob.n * blob.rate) for blob in blobs)

    return LocalControlSpec(
        n_total=n_total,
        global_rate=p_total / n_total,
        local_pocket_rate=local_pocket_rate,
        local_peer_rate=local_peer_rate,
        global_pocket_rate=global_pocket_rate,
        global_peer_rate=global_pocket_rate,  # peers follow the pocket, by design
        filler_rate=filler_rate,
        blob_n=blob_n,
        blob_spread_deg=0.05,
        blobs=tuple(blobs),
    )


LOCAL_CONTROL = _build_spec()


def generate_local_control(seed: int = 42, spec: LocalControlSpec = LOCAL_CONTROL) -> pd.DataFrame:
    """Build the control dataset: `lat`, `lon`, binary `label`.

    Coordinates are gaussian around each blob centre; positives are assigned by
    exact count within the blob, then permuted, so the ground-truth rates hold
    exactly for any seed.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for blob in spec.blobs:
        lat = rng.normal(blob.lat, spec.blob_spread_deg, blob.n)
        lon = rng.normal(blob.lon, spec.blob_spread_deg, blob.n)
        positives = round(blob.n * blob.rate)
        labels = np.zeros(blob.n, dtype=int)
        labels[:positives] = 1
        frames.append(
            pd.DataFrame({"lat": lat, "lon": lon, "label": rng.permutation(labels)})
        )

    return pd.concat(frames, ignore_index=True)


def ground_truth_table(spec: LocalControlSpec = LOCAL_CONTROL) -> pd.DataFrame:
    """The expected verdicts, so the validation table comes out of the spec itself."""
    return pd.DataFrame(
        [
            {
                "bolsão": "local (puramente local)",
                "ρ do bolsão": spec.local_pocket_rate,
                "ρ da vizinhança": spec.local_peer_rate,
                "ρ global": spec.global_rate,
                "SUL esperada": "≈ 0 (cega por construção)",
                "local z esperado": "fortemente negativo",
            },
            {
                "bolsão": "global (puramente global)",
                "ρ do bolsão": spec.global_pocket_rate,
                "ρ da vizinhança": spec.global_peer_rate,
                "ρ global": spec.global_rate,
                "SUL esperada": "alta",
                "local z esperado": "≈ 0 (vizinhos acompanham)",
            },
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("datasets/synth_local/data"))
    parser.add_argument("--filename", default="Synth_local.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = generate_local_control(seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / args.filename
    df.to_csv(path)  # index written: loaders read with index_col=0

    print(f"Wrote {len(df)} points to {path} (seed {args.seed})")
    print(f"Global rate: {df['label'].mean():.4f}")
    print()
    print(ground_truth_table().to_string(index=False))


if __name__ == "__main__":
    main()
