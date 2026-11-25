"""End-to-end application API for the research candidate-field estimator."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..application.mapping import (
    MappingComputation, MappingInputs, compute_mapping,
)
from ..application.planimetric import (
    TilingPlan, prepare_planimetric_inputs,
)
from ..dsm_registration import register
from ..vertical_estimation import component_shifts
from .candidate_field import candidate_responsibilities, fit_candidate_field_em
from .dsm_covariance import dsm_candidate_covariance


@dataclass(frozen=True)
class ResearchPlanimetricInputs:
    visible_dsm: Path
    thermal_dsm: Path
    visible_world: Mapping[str, Any]
    resolution_m: float = 0.1
    tile_m: float = 40.0
    step_m: float = 20.0
    prior_radius_m: float = 2.0
    field_max_distance_m: float = 30.0
    warp_threads: int = 8
    ecc_workers: int = 8
    max_iterations: int = 6
    outlier_prior: float = 0.1
    outlier_radius_m: float = 4.0
    dsm_covariance: bool = False
    sigma_bounds: tuple[float, float] = (0.05, 3.0)
    sigma_prior_m: float = 0.5
    sigma_prior_scale_m: float = 1.0
    sigma_prior_weight: float = 0.1
    temperature_schedule: tuple[float, ...] = (1.0,)
    ridge: float = 0.1
    kernel_length_m: float | None = None


@dataclass(frozen=True)
class ResearchPlanimetricResult:
    field: Any
    records: tuple[dict, ...]
    inliers: tuple[dict, ...]
    components: tuple[dict, ...]
    responsibilities: dict
    history: tuple[dict, ...]
    sigma_f_m: float
    outliers: int
    joint_shape: tuple[int, int]
    overlap_extent_m: tuple[float, float]
    tiling: TilingPlan
    field_support: Any
    attempts: tuple[dict, ...]


def _validate(inputs: ResearchPlanimetricInputs) -> None:
    for name in ("resolution_m", "tile_m", "step_m", "prior_radius_m",
                 "field_max_distance_m"):
        value = float(getattr(inputs, name))
        if not value > 0:
            raise ValueError("%s must be positive" % name)
    if inputs.max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    if not 0.0 < inputs.outlier_prior < 1.0:
        raise ValueError("outlier_prior must be in (0, 1)")
    if inputs.outlier_radius_m <= 0:
        raise ValueError("outlier_radius_m must be positive")
    lower, upper = (float(v) for v in inputs.sigma_bounds)
    if not 0.0 < lower < upper:
        raise ValueError("sigma_bounds must be positive and increasing")
    if not inputs.temperature_schedule:
        raise ValueError("temperature_schedule must not be empty")


@dataclass(frozen=True)
class ResearchPipelineInputs:
    planimetric: ResearchPlanimetricInputs
    visible_ortho: Path
    thermal_ortho: Path
    grid_step: int = 16
    field_max_distance_m: float | None = None


@dataclass(frozen=True)
class ResearchPipelineResult:
    planimetric: ResearchPlanimetricResult
    mapping: MappingComputation


def run_research_pipeline(inputs: ResearchPipelineInputs) -> ResearchPipelineResult:
    """Run research field estimation and in-memory panel/grid mapping."""
    planimetric = estimate_research_planimetric(inputs.planimetric)
    mapping = compute_mapping(MappingInputs(
        visible_world=inputs.planimetric.visible_world,
        field=planimetric.field,
        visible_ortho=inputs.visible_ortho,
        thermal_ortho=inputs.thermal_ortho,
        residual_model_path=Path("in_memory"),
        grid_step=inputs.grid_step,
        field_max_distance_m=inputs.field_max_distance_m,
    ))
    return ResearchPipelineResult(planimetric=planimetric, mapping=mapping)


def estimate_research_planimetric(
        inputs: ResearchPlanimetricInputs) -> ResearchPlanimetricResult:
    """Run DSM loading, all-raw candidate extraction, and joint EM."""
    _validate(inputs)
    resolution = float(inputs.resolution_m)
    prepared = prepare_planimetric_inputs(
        inputs.visible_dsm, inputs.thermal_dsm,
        resolution_m=resolution, tile_m=inputs.tile_m,
        step_m=inputs.step_m,
        field_max_distance_m=inputs.field_max_distance_m,
        warp_threads=inputs.warp_threads)
    joint = prepared.joint
    transform = prepared.transform
    plan = prepared.tiling
    field_support = prepared.field_support
    overlap_width_m, overlap_height_m = prepared.overlap_extent_m
    visible_values = prepared.visible_values
    thermal_values = prepared.thermal_values
    covariance_fn = (dsm_candidate_covariance
                     if inputs.dsm_covariance else None)
    records, _ = register(
        visible_values, thermal_values, joint, transform, resolution,
        plan.tile_m, plan.step_m, inputs.ecc_workers,
        prior_radius_m=inputs.prior_radius_m,
        include_boundary=True, apply_domain=False,
        covariance_fn=covariance_fn)
    if not records:
        raise ValueError("no usable DSM tiles for the research estimator")
    field, history = fit_candidate_field_em(
        records, resolution_m=resolution, tile_m=plan.tile_m,
        max_distance_m=field_support.max_distance_m,
        outlier_prior=inputs.outlier_prior,
        outlier_radius_m=inputs.outlier_radius_m,
        sigma_bounds=inputs.sigma_bounds,
        sigma_prior_m=inputs.sigma_prior_m,
        sigma_prior_scale_m=inputs.sigma_prior_scale_m,
        sigma_prior_weight=inputs.sigma_prior_weight,
        temperature_schedule=inputs.temperature_schedule,
        ridge=inputs.ridge, kernel_length_m=inputs.kernel_length_m,
        max_iterations=inputs.max_iterations)
    final_sigma = float(history[-1]["sigma_estimated_m"])
    responsibilities = candidate_responsibilities(
        records, field, resolution_m=resolution,
        sigma_f_m=final_sigma,
        outlier_prior=inputs.outlier_prior,
        outlier_radius_m=inputs.outlier_radius_m)
    inliers = tuple(record for index, record in enumerate(records)
                    if responsibilities["q0"][index] <= 0.5)
    components = component_shifts(inputs.visible_world["panels"], field)
    return ResearchPlanimetricResult(
        field=field, records=tuple(records), inliers=inliers,
        components=tuple(components), responsibilities=responsibilities,
        history=tuple(history), sigma_f_m=final_sigma,
        outliers=len(records) - len(inliers),
        joint_shape=tuple(joint.shape),
        overlap_extent_m=(overlap_width_m, overlap_height_m),
        tiling=TilingPlan(
            tile_m=plan.tile_m, step_m=plan.step_m,
            tile_px=int(round(plan.tile_m / resolution)),
            step_px=int(round(plan.step_m / resolution)),
            overlap_width_m=overlap_width_m,
            overlap_height_m=overlap_height_m,
            requested_tile_m=plan.requested_tile_m,
            requested_step_m=plan.requested_step_m),
        field_support=field_support,
        attempts=tuple([{
            "attempt": 1, "tile_m": plan.tile_m, "step_m": plan.step_m,
            "records": len(records), "inliers": len(inliers),
        }]),
    )