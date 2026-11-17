
"""Application use case: build the thermal panel vertical model in memory."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..domain import PanelVerticalModel
from ..panel_thermal_model import build_panel_thermal_model
from ..residual_model import PlanimetricField


@dataclass(frozen=True)
class PanelModelInputs:
    visible_world: Mapping[str, Any]
    components: Mapping[str, Mapping[str, Any]]
    field: PlanimetricField
    visible_dsm: Path
    thermal_dsm: Path
    erode_m: float = 0.35
    minimum_pixels: int = 60
    maximum_rms: float = 0.20
    workers: int = 1
    model_inputs: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PanelModelResult:
    model: PanelVerticalModel
    samples: Mapping[str, Any]
    plane_cache: Mapping[str, Any]
    validation: Mapping[str, Any]
    z_source_counts: Mapping[str, int]


def build_panel_model(inputs: PanelModelInputs) -> PanelModelResult:
    arrays, panels, counts, validation, samples, plane_cache = (
        build_panel_thermal_model(
            dict(inputs.visible_world), inputs.components, inputs.visible_dsm,
            inputs.thermal_dsm, inputs.erode_m, inputs.minimum_pixels,
            inputs.maximum_rms, inputs.field, return_samples=True,
            workers=inputs.workers))
    model = PanelVerticalModel.from_parts(
        arrays, panels,
        inputs=inputs.model_inputs,
        provenance={"kind": "panel_level_vertical_geometry",
                    **dict(inputs.provenance),
                    "z_source_counts": counts},
    )
    return PanelModelResult(model=model, samples=samples,
                            plane_cache=plane_cache, validation=validation,
                            z_source_counts=counts)
