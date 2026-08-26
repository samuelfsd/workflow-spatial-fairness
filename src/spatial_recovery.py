"""Pure recovery metrics against point-level spatial ground-truth roles."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Iterable, Mapping

import pandas as pd


ROLES = ("focal_target", "manipulated_context", "compensation", "null_background")


def _ids(region: dict[str, Any]) -> set[int]:
    value = region.get("point_ids", region.get("points", []))
    return {int(point) for point in value}


def evaluate_spatial_recovery(
    truth: pd.DataFrame,
    detected_regions: Iterable[dict[str, Any]],
    *,
    expected_direction: str | None,
    fair: bool,
    evaluated_point_ids: Iterable[int] | None = None,
    direction_required: bool = True,
    recovery_eligible: bool = True,
) -> dict[str, Any]:
    regions = list(detected_regions)
    significant = [region for region in regions if bool(region.get("significant", False))]
    consolidated = [region for region in significant if bool(region.get("consolidated", True))]
    scenario_detected = bool(significant)
    correctly_directed = [
        region for region in consolidated
        if expected_direction is not None and region.get("direction") == expected_direction
    ]
    all_detected = set().union(*(_ids(region) for region in consolidated)) if consolidated else set()
    directional_regions = correctly_directed if direction_required else consolidated
    directional_detected = (
        set().union(*(_ids(region) for region in directional_regions))
        if directional_regions else set()
    )
    target = set(truth.loc[truth["role"].eq("focal_target"), "point_id"].astype(int))
    intersection = all_detected & target
    false_positive = all_detected - target
    false_negative = target - all_detected
    universe = set(truth["point_id"].astype(int))
    true_negative = universe - (target | all_detected)
    precision = len(intersection) / len(all_detected) if all_detected else 0.0
    recall = len(intersection) / len(target) if target else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = all_detected | target
    iou = len(intersection) / len(union) if union else 0.0
    directional_intersection = directional_detected & target
    directional_precision = (
        len(directional_intersection) / len(directional_detected)
        if directional_detected else 0.0
    )
    directional_recall = (
        len(directional_intersection) / len(target) if target else 0.0
    )
    directional_f1 = (
        2 * directional_precision * directional_recall
        / (directional_precision + directional_recall)
        if directional_precision + directional_recall else 0.0
    )
    directional_union = directional_detected | target
    directional_iou = (
        len(directional_intersection) / len(directional_union)
        if directional_union else 0.0
    )
    evaluated = universe if evaluated_point_ids is None else {int(point) for point in evaluated_point_ids}
    target_covered = target & evaluated
    direction_satisfied = bool(correctly_directed) if direction_required else bool(consolidated)
    result: dict[str, Any] = {
        "scenario_detected": scenario_detected,
        "familywise_false_alarm": bool(fair and scenario_detected),
        "direction_required": bool(direction_required),
        "recovery_eligible": bool(recovery_eligible),
        "correct_direction_detected": bool(correctly_directed) if direction_required else None,
        "correct_recovery": (
            bool(
                not fair
                and direction_satisfied
                and directional_precision >= .5
                and directional_recall >= .5
            )
            if recovery_eligible else None
        ),
        "true_positive_n": len(intersection),
        "false_positive_n": len(false_positive),
        "false_negative_n": len(false_negative),
        "true_negative_n": len(true_negative),
        "precision": precision, "recall": recall, "f1": f1, "iou": iou,
        "directional_precision": directional_precision,
        "directional_recall": directional_recall,
        "directional_f1": directional_f1,
        "directional_iou": directional_iou,
        "recovered_n": len(all_detected),
        "directional_recovered_n": len(directional_detected),
        "target_n": len(target),
        "target_covered_n": len(target_covered),
        "target_coverage": len(target_covered) / len(target) if target else 1.0,
        "unassigned_target_n": len(target - evaluated),
        "detected_point_ids": sorted(all_detected),
        "directional_detected_point_ids": sorted(directional_detected),
        "all_detected_point_ids": sorted(all_detected),
        "raw_significant_regions": len(significant), "consolidated_regions": len(consolidated),
    }
    for role in ROLES:
        role_ids = set(truth.loc[truth["role"].eq(role), "point_id"].astype(int))
        detected_role_n = len(all_detected & role_ids)
        result[f"role_{role}_n"] = detected_role_n
        result[f"role_{role}_total_n"] = len(role_ids)
        result[f"role_{role}_rate"] = (
            detected_role_n / len(role_ids) if role_ids else 0.0
        )
    result["spatial_false_alarm"] = bool(not fair and result["role_null_background_n"] > 0)
    return result


def compare_method_detection_sets(
    method_points: Mapping[str, Iterable[int]],
) -> pd.DataFrame:
    """Return point-level pairwise agreement without treating agreement as truth."""
    normalized = {
        str(method): {int(point) for point in points}
        for method, points in method_points.items()
    }
    rows = []
    for first, second in combinations(sorted(normalized), 2):
        first_points = normalized[first]
        second_points = normalized[second]
        intersection = first_points & second_points
        union = first_points | second_points
        rows.append({
            "first_method": first,
            "second_method": second,
            "first_detected_n": len(first_points),
            "second_detected_n": len(second_points),
            "intersection_n": len(intersection),
            "first_only_n": len(first_points - second_points),
            "second_only_n": len(second_points - first_points),
            "union_n": len(union),
            "point_jaccard": len(intersection) / len(union) if union else 1.0,
        })
    return pd.DataFrame(rows)
