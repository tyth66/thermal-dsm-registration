
"""Application use case: final-only in-memory pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..domain import GeometryModel
from .artifacts import write_json
from .bias import BiasInputs, fit_bias
from .context import PipelineContext
from .geometry import GeometryInputs, build_geometry
from .mapping import MappingInputs, compute_mapping, mapping_result_payload
from .panel_model import PanelModelInputs, build_panel_model
from .planimetric import PlanimetricInputs, estimate_planimetric


@dataclass(frozen=True)
class FinalPipelineInputs:
    world: Mapping[str, Any]
    planimetric: PlanimetricInputs
    visible_dsm: Path
    thermal_dsm: Path
    visible_ortho: Path
    thermal_ortho: Path
    output_json: Path
    grid_step: int = 16
    array_plane_shrinkage: float = 0.5
    erode_m: float = 0.35
    minimum_pixels: int = 60
    maximum_rms: float = 0.20
    workers: int = 1
    panel_mapping: str = "rigid"


@dataclass(frozen=True)
class FinalPipelineResult:
    final_json: Path
    summary: Mapping[str, Any]
    stages: tuple[dict, ...]


def run_final_pipeline(inputs: FinalPipelineInputs) -> FinalPipelineResult:
    stages: list[dict] = []
    planimetric = estimate_planimetric(inputs.planimetric)
    stages.append({"stage": "planimetric_residual", "in_memory": True})
    components = {row["panel_id"]: row for row in planimetric.components}
    field = planimetric.field
    panel = build_panel_model(PanelModelInputs(
        visible_world=inputs.world,
        components=components,
        field=field,
        visible_dsm=inputs.visible_dsm,
        thermal_dsm=inputs.thermal_dsm,
        erode_m=inputs.erode_m,
        minimum_pixels=inputs.minimum_pixels,
        maximum_rms=inputs.maximum_rms,
        workers=inputs.workers,
        model_inputs={"planimetric_mode": "in_memory"},
        provenance={"maximum_panel_plane_rms_m": inputs.maximum_rms},
    ))
    stages.append({"stage": "panel_vertical_model", "in_memory": True})
    geometry = build_geometry(GeometryInputs(
        visible_dsm_path=inputs.visible_dsm,
        thermal_dsm_path=inputs.thermal_dsm,
        visible_world=inputs.world,
        component_rows=components,
        erode_m=inputs.erode_m,
        field=field,
        samples_cache=panel.samples,
        plane_cache=panel.plane_cache,
    ))
    stages.append({"stage": "vertical_geometry", "in_memory": True})
    bias = fit_bias(BiasInputs(
        panel_model=panel.model,
        geometry_rows={row["panel_id"]: row for row in geometry.rows},
        components=components,
        minimum_support=20,
        plane_shrinkage=inputs.array_plane_shrinkage,
    ))
    stages.append({"stage": "array_vertical_bias", "in_memory": True})
    context = PipelineContext(
        world=inputs.world,
        planimetric=field,
        panel_model=bias.corrected,
        geometry=GeometryModel.from_rows(list(geometry.rows)),
        provenance={
            "planimetric_field": "in_memory",
            "tiles": "in_memory",
            "geometry": "in_memory",
            "thermal_panel_model": "in_memory",
            "array_height_correction": bias.corrected.provenance.get(
                "array_height_correction"),
        },
    )
    canonical = context.compose()
    stages.append({"stage": "compose_residual_model", "in_memory": True})
    computation = compute_mapping(MappingInputs(
        visible_world=inputs.world,
        canonical=canonical,
        visible_ortho=inputs.visible_ortho,
        visible_dsm=inputs.visible_dsm,
        thermal_ortho=inputs.thermal_ortho,
        residual_model_path=Path("in_memory"),
        grid_step=inputs.grid_step,
        panel_mapping=inputs.panel_mapping,
    ))
    computation.summary["outputs"] = [str(inputs.output_json)]
    computation.summary["residual_model"] = "in_memory"
    write_json(inputs.output_json, mapping_result_payload(computation),
               compact=True)
    stages.append({"stage": "visible_to_thermal_mapping", "in_memory": True})
    return FinalPipelineResult(final_json=inputs.output_json,
                               summary=dict(computation.summary),
                               stages=tuple(stages))
