"""Frozen, inspectable plan expansion for the repeated spatial benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd


FAMILIES = ("uniform", "clustered", "realistic_irregular")
CORE_CONDITIONS = (
    "fair", "local_positive", "local_negative", "global_positive",
    "global_negative", "simultaneous_positive", "simultaneous_negative",
)


@dataclass(frozen=True)
class RepeatedPlan:
    schema_version: int = 1
    plan_id: str = "repeated-spatial-v1"
    n_points: int | None = None
    reference_grid: tuple[int, int] | None = None
    geometry_seeds: tuple[int, ...] = tuple(range(50))
    unfair_outcomes_per_geometry: int = 20
    fair_outcomes_per_geometry: int = 100
    global_rate: float = .5
    coordinate_source_dataset: str = "lar"
    null_worlds: int = 5000
    alpha: float = .005
    null_seed: int = 42000
    bootstrap_seed: int = 43000
    bootstrap_repetitions: int = 10000
    kmeans_seeds: int = 100
    scan_radii: tuple[float, ...] = (.02, .04, .06, .08, .10)
    methods: tuple[str, ...] = (
        "hdbscan_local_z",
        "hdbscan_peer_rate_difference",
        "hdbscan_peer_log_rate_ratio",
        "hdbscan_peer_gini_gap",
        "hdbscan_sul",
        "grid_sul",
        "scan_sul",
    )
    stresses: tuple[dict[str, Any], ...] = field(default_factory=lambda: (
        {"id": "stress_irregular_small_strong", "family": "realistic_irregular", "condition": "local_positive", "effect_pp": 30.0, "support_frac": .005, "target_shape": "irregular_nonconvex", "hdbscan_frac": .005},
        {"id": "stress_clustered_opposite", "family": "clustered", "condition": "simultaneous_opposite", "effect_pp": 20.0, "support_frac": .02, "target_shape": "rotated_ellipse", "hdbscan_frac": .01},
    ))

    def validate(self) -> "RepeatedPlan":
        if self.schema_version != 1:
            raise ValueError("schema do plano repetido não suportado")
        if self.n_points is None or self.n_points <= 0:
            raise ValueError("N deve ser congelado explicitamente no plano")
        if self.reference_grid is None or min(self.reference_grid) <= 0:
            raise ValueError("grade de referência deve ser congelada explicitamente")
        if not 0 < self.global_rate < 1:
            raise ValueError("taxa global deve estar entre zero e um")
        if not self.coordinate_source_dataset.strip():
            raise ValueError("fonte de coordenadas irregulares deve estar declarada")
        if self.null_worlds <= 0 or not 0 < self.alpha < 1:
            raise ValueError("mundos nulos e alfa devem ser explícitos e válidos")
        if not self.geometry_seeds or not self.methods:
            raise ValueError("seeds de geografia e métodos são obrigatórios")
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def _row(
    *, layer: str, family: str, condition: str, effect_pp: float = 10.0,
    support_frac: float = .02, target_shape: str = "circle",
    hdbscan_frac: float = .005, changed_factor: str | None = None,
    stress_id: str | None = None,
) -> dict[str, Any]:
    parts = [layer, family, condition, f"e{effect_pp:g}", f"s{support_frac:g}", target_shape, f"h{hdbscan_frac:g}"]
    if stress_id:
        parts.append(stress_id)
    return {
        "scenario_id": "__".join(parts), "layer": layer, "family": family,
        "condition": condition, "effect_pp": float(effect_pp),
        "support_frac": float(support_frac), "target_shape": target_shape,
        "hdbscan_frac": float(hdbscan_frac), "changed_factor": changed_factor,
        "reference_effect_pp": 10.0, "reference_support_frac": .02,
        "reference_target_shape": "circle", "reference_hdbscan_frac": .005,
    }


def expand_plan(plan: RepeatedPlan) -> pd.DataFrame:
    """Expand only declared layers; deliberately avoids a Cartesian sensitivity grid."""
    plan.validate()
    rows = [
        _row(layer="core", family=family, condition=condition)
        for family in FAMILIES for condition in CORE_CONDITIONS
    ]
    reference = {"layer": "sensitivity", "family": "uniform", "condition": "local_positive"}
    rows.extend(_row(**reference, effect_pp=value, changed_factor="effect_pp") for value in (5.0, 20.0, 30.0))
    rows.extend(_row(**reference, support_frac=value, changed_factor="support_frac") for value in (.005, .01, .05))
    rows.extend(_row(**reference, target_shape=value, changed_factor="target_shape") for value in ("rotated_ellipse", "irregular_nonconvex"))
    rows.extend(_row(**reference, hdbscan_frac=value, changed_factor="hdbscan_frac") for value in (.01, .02))
    for stress in plan.stresses:
        rows.append(_row(
            layer="stress", family=str(stress["family"]), condition=str(stress["condition"]),
            effect_pp=float(stress["effect_pp"]), support_frac=float(stress["support_frac"]),
            target_shape=str(stress["target_shape"]), hdbscan_frac=float(stress["hdbscan_frac"]),
            changed_factor="predeclared_combination", stress_id=str(stress["id"]),
        ))
    frame = pd.DataFrame(rows)
    if frame["scenario_id"].duplicated().any():
        raise ValueError("plano expandido contém identificadores duplicados")
    frame["target_n"] = (frame["support_frac"] * int(plan.n_points)).round().astype(int)
    return frame.reset_index(drop=True)
