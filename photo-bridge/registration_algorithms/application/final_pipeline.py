
"""Application use case: final-only in-memory position pipeline.

Two stages: planimetric field estimation -> visible-to-thermal position
mapping. The former panel/thermal geometry and array vertical bias stages are
gone: they never influenced the mapped pixel position.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .artifacts import write_json
from .mapping import MappingInputs, compute_mapping, mapping_result_payload
from .planimetric import PlanimetricInputs, estimate_planimetric


@dataclass(frozen=True)
class FinalPipelineInputs:
    world: Mapping[str, Any]
    planimetric: PlanimetricInputs
    visible_ortho: Path
    thermal_ortho: Path
    output_json: Path
    grid_step: int = 16


@dataclass(frozen=True)
class FinalPipelineResult:
    final_json: Path
    summary: Mapping[str, Any]
    stages: tuple[dict, ...]


def planimetric_provenance(result) -> dict:
    """Serializable provenance of the planimetric estimation stage.

    The payload summary carries the estimation parameters that produced the
    field (tiling, field support, retry attempts) plus the fitted kernel length
    and ridge, so a delivered mapping can be traced back to its field fit.
    """
    tiling = result.tiling
    field_support = result.field_support
    return {
        "tiling": {
            "tile_m": float(tiling.tile_m),
            "step_m": float(tiling.step_m),
            "tile_px": int(tiling.tile_px),
            "step_px": int(tiling.step_px),
            "overlap_width_m": float(tiling.overlap_width_m),
            "overlap_height_m": float(tiling.overlap_height_m),
            "requested_tile_m": float(tiling.requested_tile_m),
            "requested_step_m": float(tiling.requested_step_m),
        },
        "field_support": {
            "max_distance_m": float(field_support.max_distance_m),
            "short_overlap_m": float(field_support.short_overlap_m),
            "requested_max_distance_m": float(
                field_support.requested_max_distance_m),
        },
        "attempts": [dict(attempt) for attempt in result.attempts],
        "kernel_length_m": float(result.field.kernel_length_m),
        "ridge": float(result.field.ridge),
    }


def run_final_pipeline(inputs: FinalPipelineInputs) -> FinalPipelineResult:
    stages: list[dict] = []
    planimetric = estimate_planimetric(inputs.planimetric)
    stages.append({"stage": "planimetric_residual", "in_memory": True})
    computation = compute_mapping(MappingInputs(
        visible_world=inputs.world,
        field=planimetric.field,
        visible_ortho=inputs.visible_ortho,
        thermal_ortho=inputs.thermal_ortho,
        residual_model_path=Path("in_memory"),
        grid_step=inputs.grid_step,
    ))
    computation.summary["planimetric"] = planimetric_provenance(planimetric)
    computation.summary["outputs"] = [str(inputs.output_json)]
    computation.summary["residual_model"] = "in_memory"
    write_json(inputs.output_json, mapping_result_payload(computation),
               compact=True)
    stages.append({"stage": "visible_to_thermal_mapping", "in_memory": True})
    return FinalPipelineResult(final_json=inputs.output_json,
                               summary=dict(computation.summary),
                               stages=tuple(stages))
