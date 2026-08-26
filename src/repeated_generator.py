"""Method-independent geographies, spatial truth roles and mirrored outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


ROLES = ("focal_target", "manipulated_context", "compensation", "null_background")


@dataclass
class GeneratedGeography:
    points: pd.DataFrame
    metadata: dict[str, Any]


def _coordinates(n: int, family: str, rng: np.random.Generator, source: pd.DataFrame | None) -> tuple[np.ndarray, str | None]:
    if family == "uniform":
        return rng.uniform(0, 1, size=(n, 2)), None
    if family == "clustered":
        centers = np.array([[.25, .25], [.72, .30], [.45, .76]])
        chosen = rng.integers(0, len(centers), size=n)
        return np.clip(centers[chosen] + rng.normal(0, .085, size=(n, 2)), 0, 1), None
    if family == "realistic_irregular":
        if source is None or not {"lat", "lon"}.issubset(source.columns):
            raise ValueError("fonte de coordenadas lat/lon é obrigatória para geografia irregular")
        chosen = rng.choice(len(source), size=n, replace=len(source) < n)
        source_name = str(source.attrs.get("source_name", "declared_coordinate_source"))
        return source.iloc[chosen][["lat", "lon"]].to_numpy(dtype=float), source_name
    raise ValueError(f"família de geografia desconhecida: {family}")


def _shape_score(coords: np.ndarray, shape: str) -> tuple[np.ndarray, dict[str, float]]:
    center = np.median(coords, axis=0)
    scale = np.ptp(coords, axis=0)
    scale[scale == 0] = 1.0
    normalized = (coords - center) / scale
    x, y = normalized[:, 1], normalized[:, 0]
    if shape == "circle":
        score = x * x + y * y
        params = {"rotation_degrees": 0.0, "aspect_ratio": 1.0}
    elif shape == "rotated_ellipse":
        angle = np.deg2rad(32)
        xr = x * np.cos(angle) + y * np.sin(angle)
        yr = -x * np.sin(angle) + y * np.cos(angle)
        score = (xr / 2.8) ** 2 + yr**2
        params = {"rotation_degrees": 32.0, "aspect_ratio": 2.8}
    elif shape == "irregular_nonconvex":
        radius = np.hypot(x, y)
        theta = np.arctan2(y, x)
        boundary = 1.0 + .38 * np.cos(3 * theta)
        score = radius / boundary
        params = {"lobes": 3.0, "amplitude": .38}
    else:
        raise ValueError(f"forma de alvo desconhecida: {shape}")
    return score, params


def generate_geography(
    n_points: int,
    family: str,
    target_shape: str,
    support_frac: float,
    geometry_seed: int,
    *,
    coordinate_source: pd.DataFrame | None = None,
    support_tolerance: float = .002,
) -> GeneratedGeography:
    if n_points < 20 or not 0 < support_frac < .25:
        raise ValueError("N ou suporte torna o alvo impossível")
    rng = np.random.default_rng(geometry_seed)
    coords, source_name = _coordinates(n_points, family, rng, coordinate_source)
    score, shape_params = _shape_score(coords, target_shape)
    target_n = max(1, int(round(n_points * support_frac)))
    if abs(target_n / n_points - support_frac) > support_tolerance + 1 / n_points:
        raise ValueError("suporte realizado fora da tolerância declarada")
    order = np.argsort(score, kind="stable")
    context_n = min(target_n, n_points - target_n)
    # Twice the focal support is reserved for balancing even the strongest
    # predeclared same-direction target+context effect without contaminating
    # the null background.
    compensation_n = min(2 * target_n, n_points - target_n - context_n)
    roles = np.full(n_points, "null_background", dtype=object)
    roles[order[:target_n]] = "focal_target"
    roles[order[target_n:target_n + context_n]] = "manipulated_context"
    if compensation_n:
        roles[order[-compensation_n:]] = "compensation"
    frame = pd.DataFrame({
        "point_id": np.arange(n_points, dtype=int), "lat": coords[:, 0], "lon": coords[:, 1],
        "role": roles,
    })
    metadata = {
        "geometry_seed": geometry_seed, "family": family, "coordinate_source": source_name,
        "target_shape": target_shape, "shape_params": shape_params,
        "support_requested": support_frac, "support_realized": target_n / n_points,
        "target_n": target_n,
        "bounding_box": [float(frame.lon.min()), float(frame.lat.min()), float(frame.lon.max()), float(frame.lat.max())],
        "role_masks": {role: frame.loc[frame.role.eq(role), "point_id"].astype(int).tolist() for role in ROLES},
    }
    return GeneratedGeography(frame, metadata)


def _sample_exact(rng: np.random.Generator, candidates: np.ndarray, count: int) -> np.ndarray:
    count = max(0, min(int(count), len(candidates)))
    return rng.choice(candidates, size=count, replace=False) if count else np.array([], dtype=int)


def generate_outcomes(
    points: pd.DataFrame,
    condition: str,
    effect_pp: float,
    global_rate: float,
    outcome_seed: int,
) -> np.ndarray:
    """Generate a fixed-total outcome; mirrored names invert only planted direction."""
    n = len(points); total = int(round(global_rate * n))
    rng = np.random.default_rng(outcome_seed)
    outcome = np.zeros(n, dtype=int)
    if condition == "fair":
        outcome[_sample_exact(rng, np.arange(n), total)] = 1
        return outcome
    sign = 1 if condition.endswith("positive") else -1
    delta = effect_pp / 100.0
    roles = points["role"].to_numpy()
    if not (
        condition.startswith("local")
        or condition.startswith("global")
        or condition.startswith("simultaneous")
    ):
        raise ValueError(f"condição desconhecida: {condition}")

    role_ids = {role: np.flatnonzero(roles == role) for role in ROLES}
    rates = {role: global_rate for role in ROLES}
    if condition.startswith("local"):
        rates["focal_target"] += sign * delta
    elif condition.startswith("global"):
        # Target and its immediate context move together: signal versus the
        # global/outside baseline, little target-versus-peer contrast.
        rates["focal_target"] += sign * delta
        rates["manipulated_context"] += sign * delta
    elif condition == "simultaneous_opposite":
        rates["focal_target"] += delta
        rates["manipulated_context"] -= delta
    else:
        # Same-direction context shift plus a stronger focal shift loads both
        # the global and peer references without changing the planted sign.
        rates["focal_target"] += sign * delta
        rates["manipulated_context"] += sign * delta / 2

    counts = {
        role: int(round(np.clip(rate, 0, 1) * len(role_ids[role])))
        for role, rate in rates.items()
    }
    # Preserve the declared global rate by assigning the balancing adjustment
    # to the explicit compensation role first, then the null background.
    residual = total - sum(counts.values())
    role = "compensation"
    if residual > 0:
        change = min(residual, len(role_ids[role]) - counts[role])
    else:
        change = -min(-residual, counts[role])
    counts[role] += change
    residual -= change
    if residual:
        raise ValueError(
            "região de compensação insuficiente para preservar a taxa global sem contaminar o fundo nulo"
        )
    for role in ROLES:
        outcome[_sample_exact(rng, role_ids[role], counts[role])] = 1
    return outcome
