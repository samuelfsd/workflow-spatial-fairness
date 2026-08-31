"""Safe CLI-facing workflow for trial, official run and checkpoint-only report."""

from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path

from benchmark_report import load_checkpoint_results
from repeated_benchmark import RepeatedBenchmarkRunner
from repeated_plan import RepeatedPlan, expand_plan


def load_repeated_plan(path: Path) -> RepeatedPlan:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(RepeatedPlan)}
    unknown = sorted(set(payload).difference(allowed))
    if unknown:
        raise ValueError(f"campos desconhecidos no plano: {unknown}")
    for name in ("reference_grid", "geometry_seeds", "scan_radii", "methods", "stresses"):
        if payload.get(name) is not None:
            payload[name] = tuple(payload[name])
    return RepeatedPlan(**payload).validate()


def _results(output_root: Path, plan: RepeatedPlan):
    expected = json.loads(json.dumps(plan.to_dict()))
    frame = load_checkpoint_results(
        output_root / "checkpoints" / "results",
        compact_point_ids=True,
        expected_plan=expected,
    )
    if frame.empty:
        return frame
    frame = frame[frame["record_type"].eq("repeated_result")].copy()
    realization = ["scenario_id", "geometry_seed", "outcome_seed", "method_id"]
    duplicates = frame.duplicated(realization, keep=False)
    if duplicates.any():
        raise ValueError(
            "checkpoints duplicados para a mesma realização e método; use uma saída separada"
        )
    return frame


def _trial_plan(plan: RepeatedPlan) -> RepeatedPlan:
    """Return the exact reduced plan persisted by the performance trial."""
    return replace(
        plan, plan_id=f"{plan.plan_id}-trial", geometry_seeds=plan.geometry_seeds[:2],
        fair_outcomes_per_geometry=min(2, plan.fair_outcomes_per_geometry),
        unfair_outcomes_per_geometry=min(2, plan.unfair_outcomes_per_geometry),
        null_worlds=min(20, plan.null_worlds),
        bootstrap_repetitions=min(100, plan.bootstrap_repetitions),
    )


def run_repeated_workflow(
    plan_path: Path,
    output_root: Path,
    *,
    phase: str,
    confirm_official: bool = False,
    coordinate_source=None,
) -> Path:
    plan = load_repeated_plan(plan_path)
    output_root = Path(output_root)
    if phase == "run" and not confirm_official:
        raise PermissionError("a bateria oficial exige --confirm-official")
    if phase == "trial":
        trial = _trial_plan(plan)
        scenarios = expand_plan(trial)
        selected = scenarios[scenarios["layer"].eq("core")].head(2)["scenario_id"].tolist()
        RepeatedBenchmarkRunner(trial, output_root / "checkpoints", coordinate_source=coordinate_source).run(scenario_ids=selected)
        return output_root / "checkpoints"
    if phase == "run":
        RepeatedBenchmarkRunner(plan, output_root / "checkpoints", coordinate_source=coordinate_source).run()
        return output_root / "checkpoints"
    if phase == "report":
        from repeated_report import publish_repeated_report

        try:
            results = _results(output_root, plan)
            report_plan = plan
        except ValueError as official_error:
            try:
                report_plan = _trial_plan(plan)
                results = _results(output_root, report_plan)
            except ValueError:
                raise official_error
        if results.empty:
            raise FileNotFoundError("nenhum checkpoint de resultado repetido disponível")
        return publish_repeated_report(
            results, output_root / "report",
            n_bootstrap=report_plan.bootstrap_repetitions,
            seed=report_plan.bootstrap_seed,
            plan_metadata=report_plan.to_dict(),
        )
    raise ValueError("phase deve ser trial, run ou report")
