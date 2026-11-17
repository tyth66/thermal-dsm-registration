
"""Compose the canonical residual model from typed panel/geometry domain data."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..domain import (
    CanonicalResidualModel, GeometryModel, PanelVerticalBundle, PanelVerticalModel,
)
from ..residual_model import PlanimetricField


@dataclass(frozen=True)
class ComposeInputs:
    world_panels: tuple[Mapping[str, Any], ...]
    planimetric: PlanimetricField
    panel_model: PanelVerticalModel
    geometry: GeometryModel
    provenance: Mapping[str, Any]


def compose_residual_model(inputs: ComposeInputs) -> CanonicalResidualModel:
    panels = inputs.world_panels
    panel_ids = [str(panel["panel_id"]) for panel in panels]
    array_by_panel = {str(panel["panel_id"]): str(panel["array_id"])
                      for panel in panels}
    available = sum(
        inputs.panel_model.panels.get(panel_id) is not None
        and inputs.panel_model.panels[panel_id].center_xyz is not None
        for panel_id in panel_ids)
    coverage = {
        "visible_panels": len(panels),
        "mapping_panels": len(panels),
        "available_panels": int(available),
        "unavailable_panels": len(panels) - int(available),
        "thermal_surface_available_panels": int(available),
        "thermal_surface_unavailable_panels": len(panels) - int(available),
    }
    bundle = PanelVerticalBundle(panel_model=inputs.panel_model,
                                 geometry=inputs.geometry)
    return bundle.to_canonical(
        inputs.planimetric, panel_ids, array_by_panel,
        coverage=coverage, provenance=inputs.provenance)
