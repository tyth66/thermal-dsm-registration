"""Application use case: estimate the planimetric DSM shift field."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..dsm_registration import (
    cross_validate, cross_validate_spatial, highpass, load_grid, register,
    reject_outliers,
)
from ..residual_model import fit_regularized_field
from ..vertical_estimation import component_shifts

MINIMUM_TILE_PIXELS = 8
MINIMUM_ACCEPTED_TILES = 6
MAXIMUM_TILING_ATTEMPTS = 3
# Empirically derived from the 141/277 baseline: 40 m tile / 202 m short overlap.
TARGET_TILE_SHORT_RATIO = 0.20
# Field support scales with the short overlap edge. The baseline uses a 30 m
# support distance on a 202 m short edge (0.15).
TARGET_FIELD_MAX_DISTANCE_SHORT_RATIO = 0.30
MIN_FIELD_MAX_DISTANCE_M = 10.0
MAX_FIELD_MAX_DISTANCE_M = 30.0


@dataclass(frozen=True)
class PlanimetricInputs:
    visible_dsm: Path
    thermal_dsm: Path
    visible_world: Mapping[str, Any]
    resolution_m: float = 0.1
    tile_m: float = 40.0
    step_m: float = 20.0
    highpass_sigma_m: float = 1.5
    min_cc: float = 0.5
    max_shift_m: float = 1.5
    field_max_distance_m: float = 30.0
    warp_threads: int = 8
    ecc_workers: int = 8
    adaptive_tiling: bool = True
    adaptive_field_support: bool = True
    regularized_kernel_m: float | None = None
    regularized_ridge: float = 0.1


@dataclass(frozen=True)
class TilingPlan:
    tile_m: float
    step_m: float
    tile_px: int
    step_px: int
    overlap_width_m: float
    overlap_height_m: float
    requested_tile_m: float
    requested_step_m: float


@dataclass(frozen=True)
class FieldSupportPlan:
    max_distance_m: float
    short_overlap_m: float
    requested_max_distance_m: float
    adaptive: bool


@dataclass(frozen=True)
class PlanimetricResult:
    field: PlanimetricField
    records: tuple[dict, ...]
    accepted: tuple[dict, ...]
    components: tuple[dict, ...]
    outliers: int
    cv_checker: Mapping[str, Any]
    cv_quadrant: Mapping[str, Any]
    cv_block: Mapping[str, Any]
    joint_shape: tuple[int, int]
    overlap_extent_m: tuple[float, float]
    tiling: TilingPlan
    field_support: FieldSupportPlan
    attempts: tuple[dict, ...]


def adaptive_field_support(overlap_width_m: float,
                           overlap_height_m: float,
                           requested_max_distance_m: float,
                           adaptive: bool = True) -> FieldSupportPlan:
    """Scale IDW radius/support distance with the short overlap edge.

    The 141/277 baseline is the reference: 90 m radius and 30 m support on a
    202 m short edge. Small stations otherwise inherit parameters that cover
    their whole DSM overlap, which turns IDW into a near-global correction.
    """
    if overlap_width_m <= 0 or overlap_height_m <= 0:
        raise ValueError("empty DSM overlap extent")
    if requested_max_distance_m <= 0:
        raise ValueError("field support distance must be positive")
    short_m = min(float(overlap_width_m), float(overlap_height_m))
    if adaptive:
        max_distance_m = min(MAX_FIELD_MAX_DISTANCE_M,
                             max(MIN_FIELD_MAX_DISTANCE_M,
                                 TARGET_FIELD_MAX_DISTANCE_SHORT_RATIO * short_m))
    else:
        max_distance_m = float(requested_max_distance_m)
    return FieldSupportPlan(
        max_distance_m=float(max_distance_m), short_overlap_m=short_m,
        requested_max_distance_m=float(requested_max_distance_m),
        adaptive=bool(adaptive))


def adaptive_tiling(overlap_width_m: float, overlap_height_m: float,
                    resolution_m: float, requested_tile_m: float,
                    requested_step_m: float) -> TilingPlan:
    """Cap the requested tile/step by the DSM overlap extent and resolution.

    The baseline 141/277 profile uses a 40 m tile on 202 m of short-overlap
    extent (ratio ~0.20). Keep that ratio as the scale-invariant target so a
    small station gets a proportionally smaller tile instead of inheriting the
    absolute 40 m default. The result is quantized to whole pixels so
    registration and reporting agree.
    """
    if overlap_width_m <= 0 or overlap_height_m <= 0:
        raise ValueError("empty DSM overlap extent")
    if resolution_m <= 0 or requested_tile_m <= 0 or requested_step_m <= 0:
        raise ValueError("resolution, tile and step must be positive")
    short_m = min(overlap_width_m, overlap_height_m)
    baseline_ratio_tile_m = max(short_m * TARGET_TILE_SHORT_RATIO,
                                MINIMUM_TILE_PIXELS * resolution_m)
    tile_m = min(requested_tile_m, baseline_ratio_tile_m)
    step_m = min(requested_step_m, tile_m / 2.0)
    step_m = max(step_m, resolution_m)
    tile_px = max(MINIMUM_TILE_PIXELS, int(round(tile_m / resolution_m)))
    step_px = max(1, int(round(step_m / resolution_m)))
    if step_px >= tile_px:
        step_px = max(1, tile_px // 2)
    return TilingPlan(
        tile_m=tile_px * resolution_m,
        step_m=step_px * resolution_m,
        tile_px=tile_px,
        step_px=step_px,
        overlap_width_m=float(overlap_width_m),
        overlap_height_m=float(overlap_height_m),
        requested_tile_m=float(requested_tile_m),
        requested_step_m=float(requested_step_m),
    )


def estimate_planimetric(inputs: PlanimetricInputs) -> PlanimetricResult:
    resolution = inputs.resolution_m
    visible_z, visible_valid, thermal_z, thermal_valid, transform, _ = load_grid(
        inputs.visible_dsm, inputs.thermal_dsm, resolution, inputs.warp_threads)
    joint = visible_valid & thermal_valid
    overlap_height_m = joint.shape[0] * resolution
    overlap_width_m = joint.shape[1] * resolution
    field_support = adaptive_field_support(
        overlap_width_m, overlap_height_m, inputs.field_max_distance_m,
        inputs.adaptive_field_support)
    if inputs.adaptive_tiling:
        plan = adaptive_tiling(overlap_width_m, overlap_height_m, resolution,
                               inputs.tile_m, inputs.step_m)
        maximum_attempts = MAXIMUM_TILING_ATTEMPTS
    else:
        tile_px = max(1, int(round(inputs.tile_m / resolution)))
        step_px = max(1, int(round(inputs.step_m / resolution)))
        plan = TilingPlan(
            tile_m=tile_px * resolution, step_m=step_px * resolution,
            tile_px=tile_px, step_px=step_px,
            overlap_width_m=overlap_width_m, overlap_height_m=overlap_height_m,
            requested_tile_m=inputs.tile_m, requested_step_m=inputs.step_m)
        maximum_attempts = 1
    sigma_px = inputs.highpass_sigma_m / resolution
    visible_high = highpass(visible_z, visible_valid, sigma_px)
    thermal_high = highpass(thermal_z, thermal_valid, sigma_px)

    attempts: list[dict] = []
    best = None
    tile_m, step_m = plan.tile_m, plan.step_m
    for attempt in range(1, maximum_attempts + 1):
        records = register(visible_high, thermal_high, joint, transform,
                           resolution, tile_m, step_m, inputs.min_cc,
                           inputs.max_shift_m, inputs.ecc_workers)
        accepted_raw = [record for record in records if record["accepted"]]
        attempts.append({
            "attempt": attempt, "tile_m": tile_m, "step_m": step_m,
            "tile_px": int(round(tile_m / resolution)),
            "step_px": int(round(step_m / resolution)),
            "records": len(records), "accepted": len(accepted_raw),
        })
        if best is None or len(accepted_raw) > len(best[0]):
            best = (accepted_raw, records, tile_m, step_m)
        if not inputs.adaptive_tiling:
            break
        if len(accepted_raw) >= MINIMUM_ACCEPTED_TILES:
            break
        tile_m = max(plan.tile_m / (2 ** attempt), MINIMUM_TILE_PIXELS * resolution)
        step_m = max(plan.step_m / (2 ** attempt), resolution)

    if best is None or not best[0]:
        raise ValueError(
            "no accepted DSM tiles; overlap=({:.1f} x {:.1f} m), "
            "tile=({:.2f}..{:.2f} m), attempts={}".format(
                overlap_width_m, overlap_height_m, plan.tile_m, tile_m, attempts))
    accepted_raw, records, tile_m, step_m = best
    outliers = reject_outliers(records)
    accepted = [record for record in records
                if record["accepted"] and not record["outlier"]]
    if not accepted:
        raise ValueError("all accepted DSM tiles were rejected as outliers")
    field = fit_regularized_field(
        accepted, tile_m=tile_m,
        max_distance_m=field_support.max_distance_m,
        kernel_length_m=inputs.regularized_kernel_m,
        ridge=inputs.regularized_ridge)
    cv_checker = cross_validate(
        records, "checkerboard", tile_m=tile_m,
        max_distance_m=field.max_distance_m,
        kernel_length_m=field.kernel_length_m,
        ridge=inputs.regularized_ridge)
    cv_quadrant = cross_validate(
        records, "quadrant", tile_m=tile_m,
        max_distance_m=field.max_distance_m,
        kernel_length_m=field.kernel_length_m,
        ridge=inputs.regularized_ridge)
    components = component_shifts(inputs.visible_world["panels"], field)
    cv_block = cross_validate_spatial(
        records, tile_m=tile_m,
        max_distance_m=field.max_distance_m,
        kernel_length_m=field.kernel_length_m,
        ridge=inputs.regularized_ridge)
    return PlanimetricResult(
        field=field, records=tuple(records), accepted=tuple(accepted),
        components=tuple(components), outliers=outliers,
        cv_checker=cv_checker, cv_quadrant=cv_quadrant, cv_block=cv_block,
        joint_shape=tuple(joint.shape),
        overlap_extent_m=(overlap_width_m, overlap_height_m),
        field_support=field_support,
        tiling=TilingPlan(
            tile_m=tile_m, step_m=step_m,
            tile_px=int(round(tile_m / resolution)),
            step_px=int(round(step_m / resolution)),
            overlap_width_m=overlap_width_m,
            overlap_height_m=overlap_height_m,
            requested_tile_m=plan.requested_tile_m,
            requested_step_m=plan.requested_step_m),
        attempts=tuple(attempts))
