"""Incremental, checkpointed executor for the repeated spatial benchmark."""

from __future__ import annotations

import hashlib
import json
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from benchmark_checkpoint import BenchmarkUnitSpec, checkpoint_state, load_benchmark_checkpoint, publish_benchmark_checkpoint
from benchmark_sacharidis import code_provenance
from clustering.base import Partition
from clustering.hdbscan import fit_hdbscan_partition
from clustering.internal import InternalSubdivision, diagnostic_density_subdivision
from metrics.base import MetricContext
from metrics.group_fairness import classify_direction, scan_regions, select_significant_regions, simulate_null_max_suls
from metrics.neighbors import build_delaunay_adjacency
from metrics.registry import get_metric, get_metric_definition
from metrics.significance import significance_threshold, simulate_null_metric
from regions import create_grid_from_dataset, create_regions, create_rtree, create_seeds, filter_non_overlapping_regions, query_range
from repeated_generator import GeneratedGeography, generate_geography, generate_outcomes
from repeated_plan import RepeatedPlan, expand_plan
from spatial_recovery import evaluate_spatial_recovery


@dataclass(frozen=True)
class BenchmarkMethodSpec:
    """One explicit detector configuration in the repeated benchmark."""

    system: str
    metric: str
    direction_required: bool = True
    confirmatory: bool = True


BENCHMARK_METHODS: dict[str, BenchmarkMethodSpec] = {
    "hdbscan_local_z": BenchmarkMethodSpec("hdbscan", "local_z"),
    "hdbscan_peer_rate_difference": BenchmarkMethodSpec(
        "hdbscan", "peer_rate_difference",
        direction_required=bool(
            get_metric_definition("peer_rate_difference").outcome_direction
        ),
        confirmatory=get_metric_definition(
            "peer_rate_difference"
        ).confirmatory_candidate,
    ),
    "hdbscan_peer_log_rate_ratio": BenchmarkMethodSpec(
        "hdbscan", "peer_log_rate_ratio",
        direction_required=bool(
            get_metric_definition("peer_log_rate_ratio").outcome_direction
        ),
        confirmatory=get_metric_definition(
            "peer_log_rate_ratio"
        ).confirmatory_candidate,
    ),
    "hdbscan_peer_gini_gap": BenchmarkMethodSpec(
        "hdbscan", "peer_gini_gap",
        direction_required=bool(
            get_metric_definition("peer_gini_gap").outcome_direction
        ),
        confirmatory=get_metric_definition("peer_gini_gap").confirmatory_candidate,
    ),
    "hdbscan_sul": BenchmarkMethodSpec("hdbscan", "sul"),
    "grid_sul": BenchmarkMethodSpec("grid", "sul"),
    "scan_sul": BenchmarkMethodSpec("scan", "sul"),
}


def benchmark_method_spec(method_id: str) -> BenchmarkMethodSpec:
    try:
        return BENCHMARK_METHODS[method_id]
    except KeyError as exc:
        raise ValueError(
            f"método repetido desconhecido: {method_id}; "
            f"disponíveis: {sorted(BENCHMARK_METHODS)}"
        ) from exc


def _digest_points(points: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(points[["lat", "lon"]], index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _partition_frame(partition: Partition) -> pd.DataFrame:
    columns = ["record_type", "cluster_label", "point_ids", "region"]
    return pd.DataFrame([
        {"record_type": "region", "cluster_label": region.get("cluster_label", idx),
         "point_ids": (
             None if partition.method == "scan"
             else json.dumps([int(point) for point in region["points"]])
         ),
         "region": json.dumps({key: value for key, value in region.items() if key != "points"}, sort_keys=True)}
        for idx, region in enumerate(partition.regions)
    ], columns=columns)


def _partition_from_frame(frame: pd.DataFrame, method: str, params: dict[str, Any], points_frame: pd.DataFrame) -> Partition:
    n = len(points_frame)
    labels = np.full(n, -1, dtype=int); regions = []
    rtree = create_rtree(points_frame) if method == "scan" else None
    for _, row in frame.iterrows():
        metadata = json.loads(row["region"])
        if method == "scan":
            points = query_range(
                points_frame, rtree, int(metadata["center"]), float(metadata["radius"])
            )
        else:
            points = [int(value) for value in json.loads(row["point_ids"])]
        label = int(row["cluster_label"]); metadata.update({"points": points, "cluster_label": label})
        regions.append(metadata)
        if method != "scan":
            labels[points] = label
    return Partition(method=method, params=params, labels=labels, regions=regions, noise_points=np.flatnonzero(labels < 0).astype(int).tolist())


class RepeatedBenchmarkRunner:
    def __init__(self, plan: RepeatedPlan, output_root: Path, *, coordinate_source: pd.DataFrame | None = None) -> None:
        self.plan = plan.validate()
        self.output_root = Path(output_root)
        self.coordinate_source = coordinate_source
        if coordinate_source is not None:
            source_dataset = coordinate_source.attrs.get("dataset_name")
            if source_dataset != self.plan.coordinate_source_dataset:
                raise ValueError(
                    "fonte de coordenadas não corresponde ao plano: "
                    f"esperada {self.plan.coordinate_source_dataset!r}, "
                    f"recebida {source_dataset!r}"
                )
        self.provenance = code_provenance()
        self._partitions: dict[tuple, Partition] = {}
        self._thresholds: dict[tuple, float] = {}
        self._subdivisions: dict[tuple[str, tuple[int, ...]], InternalSubdivision] = {}

    def calibration_key(
        self, geometry_hash: str, method_id: str, metric: str, p_total: int,
        partition_identity: str = "unspecified",
    ) -> tuple:
        return (geometry_hash, method_id, metric, partition_identity, int(p_total), self.plan.n_points, self.plan.null_worlds, self.plan.alpha, self.plan.null_seed)

    def _partition_spec(self, geography: GeneratedGeography, system: str, params: dict[str, Any]) -> BenchmarkUnitSpec:
        return BenchmarkUnitSpec(
            dataset=f"repeated:{geography.metadata['family']}", dataset_sha256=_digest_points(geography.points),
            protocol="repeated-partition", partitioning=system, metric="none", params=params,
            seed=int(geography.metadata["geometry_seed"]), n_alt_worlds=0, code_provenance=self.provenance,
        )

    def _build_partition(self, geography: GeneratedGeography, system: str, scenario: pd.Series) -> Partition:
        points = geography.points
        if system == "hdbscan":
            return fit_hdbscan_partition(points, min_cluster_frac=float(scenario.hdbscan_frac), min_samples=60)
        rtree = create_rtree(points)
        if system == "grid":
            lon_n, lat_n = self.plan.reference_grid
            _, _, raw = create_grid_from_dataset(points, rtree, lon_n=lon_n, lat_n=lat_n)
            regions = []; labels = np.full(len(points), -1, dtype=int)
            for label, region in enumerate(raw):
                item = dict(region, cluster_label=label); regions.append(item); labels[item["points"]] = label
            return Partition("grid", {"lon_n": lon_n, "lat_n": lat_n}, labels, regions, np.flatnonzero(labels < 0).astype(int).tolist())
        if system == "scan":
            seeds = create_seeds(points, rtree, min(self.plan.kmeans_seeds, len(points)), random_state=int(geography.metadata["geometry_seed"]))
            regions = create_regions(points, rtree, seeds, np.asarray(self.plan.scan_radii, dtype=float))
            for label, region in enumerate(regions):
                region["cluster_label"] = label
            return Partition("scan", {"kmeans_seeds": self.plan.kmeans_seeds, "radii": list(self.plan.scan_radii)}, np.full(len(points), -1), regions, list(range(len(points))))
        raise ValueError(f"sistema desconhecido: {system}")

    def _partition(self, geography: GeneratedGeography, system: str, scenario: pd.Series) -> Partition:
        params = {"system": system, "hdbscan_frac": float(scenario.hdbscan_frac), "grid": self.plan.reference_grid, "scan_radii": self.plan.scan_radii, "kmeans_seeds": self.plan.kmeans_seeds}
        if system != "hdbscan":
            params.pop("hdbscan_frac")
        spec = self._partition_spec(geography, system, params)
        key = (spec.dataset_sha256, system, json.dumps(params, sort_keys=True))
        if key in self._partitions:
            return self._partitions[key]
        logical_system = (
            f"hdbscan_frac_{float(scenario.hdbscan_frac):g}"
            if system == "hdbscan" else system
        )
        path = self.output_root / "partitions" / geography.metadata["family"] / str(geography.metadata["geometry_seed"]) / logical_system
        if checkpoint_state(path, spec) == "complete":
            checkpoint = load_benchmark_checkpoint(path, spec)
            stored_params = checkpoint.metadata.get("partition_params")
            if not isinstance(stored_params, dict):
                raise ValueError(f"checkpoint de partição sem parâmetros: {path}")
            partition = _partition_from_frame(
                checkpoint.results, system, stored_params, geography.points
            )
        else:
            partition = self._build_partition(geography, system, scenario)
            publish_benchmark_checkpoint(
                path, spec, _partition_frame(partition),
                metadata={
                    "n_points": len(geography.points),
                    "partition_params": partition.params,
                },
            )
        self._partitions[key] = partition
        return partition

    def _calibration(self, geography: GeneratedGeography, partition: Partition, method_id: str, metric_name: str, p_total: int) -> tuple[float, MetricContext]:
        geometry_hash = _digest_points(geography.points)
        partition_identity = hashlib.sha256(
            json.dumps(partition.params, sort_keys=True).encode("utf-8")
        ).hexdigest()
        key = self.calibration_key(
            geometry_hash, method_id, metric_name, p_total, partition_identity
        )
        definition = get_metric_definition(metric_name)
        adjacency = (
            build_delaunay_adjacency(partition, geography.points)
            if "neighbors" in definition.needs else {}
        )
        internal_subdivider = None
        if "subclusters" in definition.needs:
            min_cluster_size = int(partition.params.get("min_cluster_size", 25))
            min_samples = int(partition.params.get("min_samples", 60))

            def subdivide(points: list[int]) -> InternalSubdivision:
                cache_key = (partition_identity, tuple(int(point) for point in points))
                if cache_key not in self._subdivisions:
                    self._subdivisions[cache_key] = diagnostic_density_subdivision(
                        geography.points,
                        points,
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                    )
                return self._subdivisions[cache_key]

            internal_subdivider = subdivide
        ctx = MetricContext(
            n_total=len(geography.points),
            p_total=p_total,
            adjacency=adjacency,
            rng=np.random.default_rng(self.plan.null_seed),
            internal_subdivider=internal_subdivider,
        )
        if key in self._thresholds:
            return self._thresholds[key], ctx
        spec = BenchmarkUnitSpec(
            dataset=f"repeated:{geography.metadata['family']}", dataset_sha256=geometry_hash,
            protocol="repeated-null", partitioning=method_id, metric=metric_name,
            params={
                "p_total": p_total, "alpha": self.plan.alpha,
                "partition_identity": partition_identity,
                "partition_params": partition.params,
            }, seed=self.plan.null_seed,
            n_alt_worlds=self.plan.null_worlds, code_provenance=self.provenance,
        )
        partition_label = str(
            partition.params.get(
                "min_cluster_frac",
                partition.params.get("hdbscan_frac", partition.params.get("system", partition.method)),
            )
        )
        path = self.output_root / "calibrations" / geography.metadata["family"] / str(geography.metadata["geometry_seed"]) / f"{method_id}__{metric_name}__{partition_label}"
        if checkpoint_state(path, spec) == "complete":
            threshold = float(load_benchmark_checkpoint(path, spec).results.iloc[0]["threshold"])
        else:
            if partition.method == "scan":
                null = simulate_null_max_suls(self.plan.null_worlds, partition.regions, len(geography.points), p_total, seed=self.plan.null_seed)
            else:
                null = simulate_null_metric(get_metric(metric_name), partition, ctx, self.plan.null_worlds, len(geography.points), p_total, seed=self.plan.null_seed)
            threshold = significance_threshold(self.plan.alpha, null)
            publish_benchmark_checkpoint(path, spec, pd.DataFrame([{"record_type": "calibration", "threshold": threshold}]), metadata={"null_min": float(null.min()) if len(null) else None, "null_max": float(null.max()) if len(null) else None})
        self._thresholds[key] = threshold
        return threshold, ctx

    def _detect(self, geography: GeneratedGeography, partition: Partition, outcomes: np.ndarray, method_id: str, metric_name: str) -> tuple[list[dict[str, Any]], float, float]:
        n_total = len(outcomes); p_total = int(outcomes.sum())
        threshold, ctx = self._calibration(geography, partition, method_id, metric_name, p_total)
        if partition.method == "scan":
            _, best_score, scores = scan_regions(partition.regions, outcomes, n_total, p_total)
            significant = select_significant_regions(partition.regions, scores, threshold)
            consolidated = filter_non_overlapping_regions(significant, geography.points)
            significant_ids = {id(region) for region in significant}; consolidated_ids = {id(region) for region in consolidated}
            detected = []
            for region, score in zip(partition.regions, scores, strict=True):
                point_ids = list(region["points"]); n = len(point_ids); p = int(outcomes[point_ids].sum())
                detected.append({"point_ids": point_ids, "direction": classify_direction(n, p, n_total, p_total), "score": float(score), "significant": id(region) in significant_ids, "consolidated": id(region) in consolidated_ids})
            return detected, threshold, best_score
        definition = get_metric_definition(metric_name)
        metric = definition.compute
        scores = np.asarray(metric(partition, outcomes, ctx).per_cluster, dtype=float)
        detected = []
        for region, score in zip(partition.regions, scores, strict=True):
            point_ids = list(region["points"]); n = len(point_ids); p = int(outcomes[point_ids].sum())
            if metric_name == "local_z" or (
                definition.benchmark_candidate and definition.outcome_direction
            ):
                direction = (
                    "positive" if score > 0
                    else "negative" if score < 0
                    else "neutral"
                )
            elif definition.benchmark_candidate:
                direction = None
            else:
                direction = classify_direction(n, p, n_total, p_total)
            significant = bool(
                np.isfinite(score)
                and (abs(score) >= threshold if threshold > 0 else abs(score) > 0)
            )
            detected.append({"point_ids": point_ids, "direction": direction, "score": float(score), "significant": significant, "consolidated": True})
        finite = np.abs(scores[np.isfinite(scores)])
        return detected, threshold, float(finite.max()) if len(finite) else 0.0

    def _result_spec(self, geography: GeneratedGeography, scenario: pd.Series, method_id: str, metric: str, outcome_seed: int) -> BenchmarkUnitSpec:
        return BenchmarkUnitSpec(
            dataset=f"repeated:{scenario.scenario_id}", dataset_sha256=_digest_points(geography.points),
            protocol="repeated-outcome", partitioning=method_id, metric=metric,
            params={
                "plan": self.plan.to_dict(), "scenario": scenario.to_dict(),
                "global_rate": self.plan.global_rate,
            },
            seed=outcome_seed, n_alt_worlds=self.plan.null_worlds, code_provenance=self.provenance,
        )

    def _publish_truth(
        self,
        geography: GeneratedGeography,
        scenario: pd.Series,
        outcomes: np.ndarray,
        outcome_seed: int,
    ) -> None:
        """Persist the method-independent point truth before any detector runs."""
        spec = BenchmarkUnitSpec(
            dataset=f"repeated:{scenario.scenario_id}",
            dataset_sha256=_digest_points(geography.points),
            protocol="repeated-truth",
            partitioning="none",
            metric="ground_truth",
            params={
                "plan": self.plan.to_dict(),
                "scenario": scenario.to_dict(),
                "global_rate": self.plan.global_rate,
            },
            seed=outcome_seed,
            n_alt_worlds=0,
            code_provenance=self.provenance,
        )
        path = (
            self.output_root / "truth" / str(scenario.scenario_id)
            / str(geography.metadata["geometry_seed"])
            / f"outcome_{outcome_seed}"
        )
        truth = geography.points[["point_id", "lat", "lon", "role"]].copy()
        truth.insert(0, "record_type", "truth_point")
        truth["outcome"] = outcomes.astype(int)
        truth["scenario_id"] = str(scenario.scenario_id)
        truth["geometry_seed"] = int(geography.metadata["geometry_seed"])
        truth["outcome_seed"] = int(outcome_seed)
        truth["null_seed"] = int(self.plan.null_seed)
        publish_benchmark_checkpoint(
            path,
            spec,
            truth,
            metadata={"geometry": geography.metadata},
        )

    def _publish_detection_trace(
        self,
        geography: GeneratedGeography,
        scenario: pd.Series,
        detected: list[dict[str, Any]],
        method_id: str,
        metric: str,
        outcome_seed: int,
    ) -> None:
        """Persist candidate/raw-significant/consolidated states per region."""
        spec = BenchmarkUnitSpec(
            dataset=f"repeated:{scenario.scenario_id}",
            dataset_sha256=_digest_points(geography.points),
            protocol="repeated-detections",
            partitioning=method_id,
            metric=metric,
            params={"plan": self.plan.to_dict(), "scenario": scenario.to_dict()},
            seed=outcome_seed,
            n_alt_worlds=self.plan.null_worlds,
            code_provenance=self.provenance,
        )
        path = (
            self.output_root / "detections" / str(scenario.scenario_id)
            / str(geography.metadata["geometry_seed"])
            / method_id / f"outcome_{outcome_seed}"
        )
        records = []
        for region_id, region in enumerate(detected):
            selected = bool(region.get("significant", False))
            consolidated = bool(region.get("consolidated", True))
            records.append({
                "record_type": "detection_region",
                "region_id": region_id,
                "method_id": method_id,
                "metric": metric,
                "candidate": True,
                "significant": selected,
                "consolidated": consolidated,
                "direction": region.get("direction"),
                "score": region.get("score"),
                # Full membership is necessary only for the selected union;
                # every scan candidate remains reconstructable from its frozen
                # center/radius in the partition checkpoint.
                "point_ids": json.dumps(region["point_ids"])
                if selected and consolidated else None,
            })
        publish_benchmark_checkpoint(
            path,
            spec,
            pd.DataFrame(records, columns=[
                "record_type", "region_id", "method_id", "metric",
                "candidate", "significant", "consolidated", "direction",
                "score", "point_ids",
            ]),
            metadata={
                "candidate_regions": len(records),
                "significant_regions": sum(bool(row["significant"]) for row in records),
                "consolidated_regions": sum(
                    bool(row["significant"] and row["consolidated"])
                    for row in records
                ),
            },
        )

    def run(self, *, layers: Iterable[str] | None = None, scenario_ids: Iterable[str] | None = None) -> pd.DataFrame:
        scenarios = expand_plan(self.plan)
        if layers is not None:
            scenarios = scenarios[scenarios["layer"].isin(set(layers))]
        if scenario_ids is not None:
            scenarios = scenarios[scenarios["scenario_id"].isin(set(scenario_ids))]
        rows = []
        for _, scenario in scenarios.iterrows():
            for geometry_seed in self.plan.geometry_seeds:
                geography = generate_geography(
                    int(self.plan.n_points), scenario.family, scenario.target_shape,
                    float(scenario.support_frac), int(geometry_seed), coordinate_source=self.coordinate_source,
                )
                outcome_count = self.plan.fair_outcomes_per_geometry if scenario.condition == "fair" else self.plan.unfair_outcomes_per_geometry
                for outcome_idx in range(outcome_count):
                    outcome_seed = int(geometry_seed) * 100000 + outcome_idx + 1
                    outcomes = generate_outcomes(geography.points, scenario.condition, float(scenario.effect_pp), self.plan.global_rate, outcome_seed)
                    self._publish_truth(
                        geography, scenario, outcomes, outcome_seed
                    )
                    expected = None if scenario.condition == "fair" else ("positive" if scenario.condition.endswith("positive") or scenario.condition == "simultaneous_opposite" else "negative")
                    for method_id in self.plan.methods:
                        method_spec = benchmark_method_spec(method_id)
                        system = method_spec.system
                        metric = method_spec.metric
                        spec = self._result_spec(geography, scenario, method_id, metric, outcome_seed)
                        path = self.output_root / "results" / scenario.scenario_id / str(geometry_seed) / method_id / f"outcome_{outcome_seed}"
                        if checkpoint_state(path, spec) == "complete":
                            rows.append(load_benchmark_checkpoint(path, spec).results.iloc[0].to_dict()); continue
                        partition = self._partition(geography, system, scenario)
                        started = time.perf_counter()
                        tracemalloc.start()
                        detected, threshold, best_score = self._detect(geography, partition, outcomes, method_id, metric)
                        self._publish_detection_trace(
                            geography, scenario, detected, method_id, metric,
                            outcome_seed,
                        )
                        _, evaluation_python_peak_bytes = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                        elapsed_seconds = time.perf_counter() - started
                        evaluated_point_ids = set().union(*(
                            set(int(point) for point in region["points"])
                            for region in partition.regions
                        )) if partition.regions else set()
                        recovery = evaluate_spatial_recovery(
                            geography.points,
                            detected,
                            expected_direction=expected,
                            fair=scenario.condition == "fair",
                            evaluated_point_ids=evaluated_point_ids,
                            direction_required=method_spec.direction_required,
                            recovery_eligible=method_spec.confirmatory,
                        )
                        evaluation_coverage = len(evaluated_point_ids) / len(outcomes)
                        detected_coverage = (
                            len(recovery["all_detected_point_ids"]) / len(outcomes)
                        )
                        recovery["detected_point_ids"] = json.dumps(
                            recovery["detected_point_ids"]
                        )
                        recovery["directional_detected_point_ids"] = json.dumps(
                            recovery["directional_detected_point_ids"]
                        )
                        recovery["all_detected_point_ids"] = json.dumps(
                            recovery["all_detected_point_ids"]
                        )
                        row = {
                            "record_type": "repeated_result", "plan_id": self.plan.plan_id,
                            "scenario_id": scenario.scenario_id, "layer": scenario.layer,
                            "family": scenario.family, "condition": scenario.condition,
                            "changed_factor": scenario.changed_factor, "method_id": method_id,
                            "partitioning": system, "metric": metric,
                            "confirmatory_method": method_spec.confirmatory,
                            "geometry_seed": int(geometry_seed), "outcome_seed": outcome_seed,
                            "null_seed": self.plan.null_seed, "N": len(outcomes), "P": int(outcomes.sum()),
                            "global_rate": float(outcomes.mean()), "target_n": geography.metadata["target_n"],
                            "target_support": geography.metadata["support_realized"],
                            "effect_pp": float(scenario.effect_pp), "expected_direction": expected,
                            "score": best_score, "threshold": threshold,
                            "coverage": evaluation_coverage,
                            "evaluation_coverage": evaluation_coverage,
                            "detected_coverage": detected_coverage,
                            "candidate_regions": len(partition.regions),
                            "elapsed_seconds": elapsed_seconds,
                            "evaluation_python_peak_bytes": int(evaluation_python_peak_bytes),
                            **recovery,
                        }
                        result = pd.DataFrame([row])
                        publish_benchmark_checkpoint(path, spec, result, metadata={"geometry": geography.metadata})
                        rows.append(row)
        return pd.DataFrame(rows).sort_values(["scenario_id", "geometry_seed", "outcome_seed", "method_id"], kind="stable").reset_index(drop=True) if rows else pd.DataFrame()
