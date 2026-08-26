"""Block-bootstrap validity gate and predeclared paired recovery contrasts."""

from __future__ import annotations

import numpy as np
import pandas as pd


CONTRASTS = (
    (
        "hdbscan_peer_rate_difference", "hdbscan_local_z",
        "diferença de taxa versus local z-score nos mesmos peers",
    ),
    (
        "hdbscan_peer_log_rate_ratio", "hdbscan_local_z",
        "log da razão de taxas versus local z-score nos mesmos peers",
    ),
    ("hdbscan_local_z", "hdbscan_sul", "local z-score versus SUL no HDBSCAN"),
    ("hdbscan_sul", "scan_sul", "HDBSCAN versus varredura com SUL"),
    ("hdbscan_sul", "grid_sul", "HDBSCAN versus grade com SUL"),
    ("hdbscan_local_z", "scan_sul", "HDBSCAN + local z-score versus varredura + SUL"),
    ("hdbscan_local_z", "grid_sul", "HDBSCAN + local z-score versus grade + SUL"),
)


def _block_means(frame: pd.DataFrame, value: str, n_bootstrap: int, seed: int) -> np.ndarray:
    geometry_ids = frame["geometry_seed"].drop_duplicates().to_numpy()
    if not len(geometry_ids):
        return np.array([], dtype=float)
    blocks = {key: group[value].astype(float).to_numpy() for key, group in frame.groupby("geometry_seed", sort=False)}
    rng = np.random.default_rng(seed)
    values = np.empty(n_bootstrap, dtype=float)
    for idx in range(n_bootstrap):
        sampled = rng.choice(geometry_ids, size=len(geometry_ids), replace=True)
        values[idx] = np.concatenate([blocks[key] for key in sampled]).mean()
    return values


def block_fwer_gate(frame: pd.DataFrame, *, n_bootstrap: int = 10000, seed: int = 43000) -> pd.DataFrame:
    rows = []
    for offset, (keys, group) in enumerate(frame.groupby(["family", "method_id"], sort=True)):
        bootstrap = _block_means(group, "familywise_false_alarm", n_bootstrap, seed + offset)
        estimate = float(group["familywise_false_alarm"].astype(float).mean())
        lower, upper = np.quantile(bootstrap, [.025, .975]) if len(bootstrap) else (np.nan, np.nan)
        upper_one = float(np.quantile(bootstrap, .95)) if len(bootstrap) else np.nan
        rows.append({
            "family": keys[0], "method_id": keys[1], "fwer": estimate,
            "ci95_lower": float(lower), "ci95_upper": float(upper),
            "upper_one_sided_95": upper_one, "gate_threshold": .01,
            "gate_passed": bool(upper_one <= .01), "nominal_alpha": .005,
            "bootstrap_seed": seed + offset, "bootstrap_repetitions": n_bootstrap,
        })
    return pd.DataFrame(rows)


def paired_recovery_contrast(
    frame: pd.DataFrame,
    first_method: str,
    second_method: str,
    *,
    n_bootstrap: int = 10000,
    seed: int = 43000,
    gates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    keys = ["family", "scenario_id", "geometry_seed", "outcome_seed"]
    first = frame[frame["method_id"].eq(first_method)][keys + ["correct_recovery"]].rename(columns={"correct_recovery": "first"})
    second = frame[frame["method_id"].eq(second_method)][keys + ["correct_recovery"]].rename(columns={"correct_recovery": "second"})
    paired = first.merge(second, on=keys, how="inner")
    rows = []
    for offset, ((family, scenario), group) in enumerate(paired.groupby(["family", "scenario_id"], sort=True)):
        group = group.copy(); group["difference"] = group["first"].astype(float) - group["second"].astype(float)
        bootstrap = _block_means(group, "difference", n_bootstrap, seed + offset)
        lower, upper = np.quantile(bootstrap, [.025, .975]) * 100 if len(bootstrap) else (np.nan, np.nan)
        difference = float(group["difference"].mean() * 100)
        valid = True
        if gates is not None:
            required = gates[gates["family"].eq(family) & gates["method_id"].isin([first_method, second_method])]
            valid = len(required) == 2 and bool(required["gate_passed"].all())
        if not valid:
            conclusion = "bloqueado_pelo_gate"
        elif lower <= 0 <= upper:
            conclusion = "inconclusivo"
        else:
            conclusion = "favorece_primeiro" if difference > 0 else "favorece_segundo"
        rows.append({
            "family": family, "scenario_id": scenario, "first_method": first_method,
            "second_method": second_method, "first_recovery": float(group["first"].mean()),
            "second_recovery": float(group["second"].mean()), "difference_pp": difference,
            "ci95_lower_pp": float(lower), "ci95_upper_pp": float(upper),
            "conclusion": conclusion, "bootstrap_seed": seed + offset,
            "bootstrap_repetitions": n_bootstrap,
        })
    return pd.DataFrame(rows)


def all_predeclared_contrasts(frame: pd.DataFrame, *, gates: pd.DataFrame | None = None, n_bootstrap: int = 10000, seed: int = 43000) -> pd.DataFrame:
    outputs = []
    for idx, (first, second, label) in enumerate(CONTRASTS):
        result = paired_recovery_contrast(frame, first, second, n_bootstrap=n_bootstrap, seed=seed + idx * n_bootstrap, gates=gates)
        result["contrast"] = label
        outputs.append(result)
    return pd.concat(outputs, ignore_index=True, sort=False) if outputs else pd.DataFrame()
