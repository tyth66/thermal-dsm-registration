"""Application use case: estimate the planimetric DSM shift field."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..dsm_registration import highpass, load_grid, register
from ..residual_model import fit_regularized_field
from ..vertical_estimation import component_shifts

MINIMUM_TILE_PIXELS = 8
MINIMUM_ACCEPTED_TILES = 6
MAXIMUM_TILING_ATTEMPTS = 3
# Tile area target: 1% of the common DSM overlap, with 50% overlap between
# adjacent tiles.  The resulting tile is quantized to an even pixel count so
# step = tile / 2 is exact.
TARGET_TILE_AREA_RATIO = 0.01
# Field support ratio: 0.30 is the CONFIRMED design ratio (not 0.15). The
# adaptive formula is max_distance = min(30, max(10, 0.30 * short_edge)), so
# adaptation only bites when 0.30 * short_edge < 30 m, i.e. short_edge < 100 m.
# Stations whose short edge is 100-202 m are capped by the 30 m ceiling; the
# 141/277 baseline (202 m short edge) therefore uses 30 m. The 10 m floor covers
# very small overlaps where the ratio alone would shrink support below 10 m.
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
    highpass_sigma_m: float = 0.0
    prior_radius_m: float = 2.0
    field_max_distance_m: float = 30.0
    warp_threads: int = 8
    ecc_workers: int = 8
    regularized_kernel_m: float | None = None
    regularized_ridge: float = 0.1


PLANIMETRIC_SCALE_PARAMETERS = ("resolution_m", "tile_m", "step_m",
                                "highpass_sigma_m", "prior_radius_m",
                                "field_max_distance_m")


def validate_planimetric_inputs(inputs: PlanimetricInputs) -> None:
    """Reject non-positive scale parameters before any DSM work starts.

    "not value > 0" is used instead of "value <= 0" so NaN is rejected too;
    each error names the offending field.
    """
    for name in PLANIMETRIC_SCALE_PARAMETERS:
        value = getattr(inputs, name)
        if name == "highpass_sigma_m" and value == 0.0:
            continue
        if not value > 0:
            raise ValueError("%s must be positive" % name)


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


@dataclass(frozen=True)
class PlanimetricResult:
    field: PlanimetricField
    records: tuple[dict, ...]
    accepted: tuple[dict, ...]
    components: tuple[dict, ...]
    outliers: int
    joint_shape: tuple[int, int]
    overlap_extent_m: tuple[float, float]
    tiling: TilingPlan
    field_support: FieldSupportPlan
    attempts: tuple[dict, ...]
    highpass_sigma_m: float | None = None


def adaptive_field_support(overlap_width_m: float,
                           overlap_height_m: float,
                           requested_max_distance_m: float) -> FieldSupportPlan:
    """Scale the field support distance with the short overlap edge.

    max_distance = min(30, max(10, 0.30 * short_edge)). The 0.30 ratio is the
    confirmed design value: adaptation only takes effect below a 100 m short
    edge, while the 141/277 baseline (202 m) sits at the 30 m cap. Without the
    ratio, small stations would inherit a support distance covering their whole
    DSM overlap, which turns the field correction near-global.
    """
    if overlap_width_m <= 0 or overlap_height_m <= 0:
        raise ValueError("empty DSM overlap extent")
    if requested_max_distance_m <= 0:
        raise ValueError("field support distance must be positive")
    short_m = min(float(overlap_width_m), float(overlap_height_m))
    max_distance_m = min(MAX_FIELD_MAX_DISTANCE_M,
                         max(MIN_FIELD_MAX_DISTANCE_M,
                             TARGET_FIELD_MAX_DISTANCE_SHORT_RATIO * short_m))
    return FieldSupportPlan(
        max_distance_m=float(max_distance_m), short_overlap_m=short_m,
        requested_max_distance_m=float(requested_max_distance_m))


def adaptive_tiling(overlap_width_m: float, overlap_height_m: float,
                    resolution_m: float, requested_tile_m: float,
                    requested_step_m: float) -> TilingPlan:
    """Choose a tile whose area is 1% of the DSM overlap, at 50% overlap.

    The requested tile remains an upper bound.  The pixel count is rounded to
    an even number so the 50% step is exact after quantization.
    """
    if overlap_width_m <= 0 or overlap_height_m <= 0:
        raise ValueError("empty DSM overlap extent")
    if resolution_m <= 0 or requested_tile_m <= 0 or requested_step_m <= 0:
        raise ValueError("resolution, tile and step must be positive")
    requested_px = max(MINIMUM_TILE_PIXELS,
                       int(round(requested_tile_m / resolution_m)))
    target_px = np.sqrt(TARGET_TILE_AREA_RATIO * overlap_width_m
                        * overlap_height_m) / resolution_m
    target_px = max(MINIMUM_TILE_PIXELS,
                    int(round(target_px / 2.0)) * 2)
    tile_px = min(requested_px, target_px)
    step_px = max(1, tile_px // 2)
    tile_m = tile_px * resolution_m
    step_m = step_px * resolution_m
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
    validate_planimetric_inputs(inputs)
    resolution = inputs.resolution_m
    visible_z, visible_valid, thermal_z, thermal_valid, transform, _ = load_grid(
        inputs.visible_dsm, inputs.thermal_dsm, resolution, inputs.warp_threads)
    joint = visible_valid & thermal_valid
    overlap_height_m = joint.shape[0] * resolution
    overlap_width_m = joint.shape[1] * resolution
    field_support = adaptive_field_support(
        overlap_width_m, overlap_height_m, inputs.field_max_distance_m)
    plan = adaptive_tiling(overlap_width_m, overlap_height_m, resolution,
                           inputs.tile_m, inputs.step_m)
    maximum_attempts = MAXIMUM_TILING_ATTEMPTS
    highpass_sigma_m = float(inputs.highpass_sigma_m)
    if highpass_sigma_m > 0.0:
        sigma_px = highpass_sigma_m / resolution
        visible_high = highpass(visible_z, visible_valid, sigma_px)
        thermal_high = highpass(thermal_z, thermal_valid, sigma_px)
    else:
        visible_high = np.where(visible_valid, visible_z, 0.0).astype(np.float32)
        thermal_high = np.where(thermal_valid, thermal_z, 0.0).astype(np.float32)

    attempts: list[dict] = []
    best = None
    tile_m, step_m = plan.tile_m, plan.step_m
    for attempt in range(1, maximum_attempts + 1):
        records, displacement = register(
            visible_high, thermal_high, joint, transform, resolution,
            tile_m, step_m, inputs.ecc_workers,
            prior_radius_m=inputs.prior_radius_m,
            include_boundary=True)
        accepted_raw = [record for record in records
                        if record.get("consistent", True)
                        and not record.get("ambiguous", False)]
        attempts.append({
            "attempt": attempt, "tile_m": tile_m, "step_m": step_m,
            "tile_px": int(round(tile_m / resolution)),
            "step_px": int(round(step_m / resolution)),
            "records": len(records), "accepted": len(accepted_raw),
            "ambiguous": sum(bool(record.get("ambiguous", False))
                             for record in records),
            "inconsistent": sum(not record.get("consistent", True)
                                for record in records),
            "invalid_displacements": displacement["rejected"],
            "displacement_domain_m": {
                key: displacement[key]
                for key in ("center_dx", "bound_dx", "center_dy", "bound_dy")},
        })
        if best is None or len(accepted_raw) > len(best[0]):
            best = (accepted_raw, records, tile_m, step_m)
        if len(accepted_raw) >= MINIMUM_ACCEPTED_TILES:
            break
        tile_m = max(plan.tile_m / (2 ** attempt), MINIMUM_TILE_PIXELS * resolution)
        step_m = max(plan.step_m / (2 ** attempt), resolution)

    if best is None or not best[0]:
        raise ValueError(
            "no usable DSM tiles after consistency/ambiguity filtering; "
            "overlap=({:.1f} x {:.1f} m), tile=({:.2f}..{:.2f} m), "
            "attempts={}".format(
                overlap_width_m, overlap_height_m, plan.tile_m, tile_m, attempts))
    accepted_raw, records, tile_m, step_m = best
    accepted = list(accepted_raw)
    outliers = len(records) - len(accepted)
    if not accepted:
        raise ValueError("no usable DSM tiles produced a finite registration")
    field = fit_regularized_field(
        accepted, tile_m=tile_m,
        max_distance_m=field_support.max_distance_m,
        kernel_length_m=inputs.regularized_kernel_m,
        ridge=inputs.regularized_ridge)
    components = component_shifts(inputs.visible_world["panels"], field)
    return PlanimetricResult(
        field=field, records=tuple(records), accepted=tuple(accepted),
        components=tuple(components), outliers=outliers,
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
        attempts=tuple(attempts), highpass_sigma_m=float(highpass_sigma_m))
