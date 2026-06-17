"""Fairness metrics and scans used by the original spatial audit."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def get_simple_stats(points: Iterable[int], types: np.ndarray) -> tuple[int, int, float]:
    """Return region size, positive count, and local positive rate."""
    point_ids = np.asarray(list(points), dtype=int)
    n = int(len(point_ids))
    if n == 0:
        return 0, 0, np.nan

    p = int(types[point_ids].sum())
    return n, p, p / n


def _safe_x_log_prob(x: int | float, prob: float) -> float:
    if x <= 0 or prob <= 0:
        return 0.0
    return float(x) * math.log(prob)


def _max_log_likelihood(n: int, p: int, n_total: int, p_total: int) -> float:
    rho = p_total / n_total
    l0max = _safe_x_log_prob(p_total, rho) + _safe_x_log_prob(n_total - p_total, 1 - rho)

    if n == 0 or n == n_total:
        return l0max

    rho_in = p / n
    rho_out = (p_total - p) / (n_total - n)

    return (
        _safe_x_log_prob(p, rho_in)
        + _safe_x_log_prob(n - p, 1 - rho_in)
        + _safe_x_log_prob(p_total - p, rho_out)
        + _safe_x_log_prob(n_total - n - (p_total - p), 1 - rho_out)
    )


def calculate_sul(
    n: int,
    p: int,
    n_total: int,
    p_total: int,
    direction: str = "both",
) -> float:
    """Return Spatial Unfairness Likelihood, preserving the authors' statistic."""
    if n_total == 0 or n == 0 or n == n_total:
        return 0.0

    rho = p_total / n_total
    l0max = _safe_x_log_prob(p_total, rho) + _safe_x_log_prob(n_total - p_total, 1 - rho)

    rho_in = p / n
    rho_out = (p_total - p) / (n_total - n)

    if direction == "less_in" and rho_in >= rho_out:
        return 0.0
    if direction == "less_out" and rho_in <= rho_out:
        return 0.0

    return _max_log_likelihood(n, p, n_total, p_total) - l0max


def calculate_meanvar(rhos: Iterable[float]) -> float:
    """Return the MeanVar score used by the partitioning baseline."""
    values = np.asarray(list(rhos), dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return 0.0

    mean_rho = float(np.mean(values))
    return float(np.mean((values - mean_rho) ** 2))


def scan_regions(
    regions: list[dict],
    types: np.ndarray,
    n_total: int,
    p_total: int,
    direction: str = "both",
) -> tuple[dict | None, float, np.ndarray]:
    """Score all regions by SUL and return the best region and all scores."""
    if not regions:
        return None, 0.0, np.asarray([], dtype=float)

    statistics = []
    for region in regions:
        n, p, _ = get_simple_stats(region["points"], types)
        statistics.append(calculate_sul(n, p, n_total, p_total, direction=direction))

    scores = np.asarray(statistics, dtype=float)
    idx = int(np.argmax(scores))
    return regions[idx], float(scores[idx]), scores


def scan_partitioning(regions: list[dict], types: np.ndarray) -> tuple[dict | None, float, np.ndarray, np.ndarray]:
    """Run the MeanVar-style partition scan from the original notebook."""
    if not regions:
        return None, 0.0, np.asarray([], dtype=float), np.asarray([], dtype=float)

    rhos = []
    for region in regions:
        _, _, rho = get_simple_stats(region["points"], types)
        rhos.append(rho)

    rho_values = np.asarray(rhos, dtype=float)
    if np.all(np.isnan(rho_values)):
        scores = np.zeros_like(rho_values)
        return regions[0], 0.0, scores, rho_values

    mean_rho = float(np.nanmean(rho_values))
    scores = (rho_values - mean_rho) ** 2

    idx = int(np.nanargmax(scores))
    return regions[idx], float(scores[idx]), scores, rho_values


def get_random_types(n_total: int, p_total: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Create one alternate world with the global positive rate."""
    generator = rng if rng is not None else np.random.default_rng()
    return generator.binomial(size=n_total, n=1, p=p_total / n_total)


def get_signif_threshold(
    signif_level: float,
    n_alt_worlds: int,
    regions: list[dict],
    n_total: int,
    p_total: int,
    seed: int | None = None,
) -> float:
    """Return the Monte Carlo SUL threshold used by the authors."""
    if n_alt_worlds <= 0 or not regions:
        return 0.0

    rng = np.random.default_rng(seed)
    alt_max_scores = []
    for _ in range(n_alt_worlds):
        alt_types = get_random_types(n_total, p_total, rng=rng)
        _, alt_max, _ = scan_regions(regions, alt_types, n_total, p_total)
        alt_max_scores.append(alt_max)

    alt_max_scores.sort(reverse=True)
    idx = min(int(signif_level * n_alt_worlds), len(alt_max_scores) - 1)
    return float(alt_max_scores[idx])


def select_significant_regions(
    regions: list[dict],
    statistics: np.ndarray,
    threshold: float,
) -> list[dict]:
    """Select regions at or above the threshold, ordered by descending SUL."""
    if not regions or len(statistics) == 0:
        return []

    indexes = np.argsort(statistics)[::-1]
    return [regions[int(i)] for i in indexes if statistics[int(i)] >= threshold]
