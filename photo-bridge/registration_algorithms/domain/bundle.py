
"""Combine the thermal panel model, geometry, and bias into canonical residuals."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..residual_model import PlanimetricField
from .geometry import GeometryModel
from .panel import PanelVerticalModel
from .residual import CanonicalResidualModel, PanelResidual


@dataclass(frozen=True)
class PanelVerticalBundle:
    panel_model: PanelVerticalModel
    geometry: GeometryModel

    def to_canonical(self, field: PlanimetricField,
                     panel_order: list[str],
                     array_by_panel: Mapping[str, str],
                     coverage: Mapping[str, Any],
                     provenance: Mapping[str, Any]) -> CanonicalResidualModel:
        panels = {}
        for panel_id in panel_order:
            panel = self.panel_model.panels.get(panel_id)
            geometry = self.geometry.panels.get(panel_id)
            visible_plane = None if geometry is None else geometry.visible_plane
            thermal_plane = (None if panel is None or panel.panel_plane is None
                             else panel.panel_plane.plane)
            method = ("array_bias" if panel is None
                      or panel.vertical_correction_method is None
                      else panel.vertical_correction_method)
            panels[panel_id] = PanelResidual(
                array_id=str(array_by_panel[panel_id]),
                visible_plane=visible_plane,
                thermal_plane=thermal_plane,
                thermal_correction_method=method,
            )
        return CanonicalResidualModel.from_parts(
            field, {panel_id: panel.to_row() for panel_id, panel in panels.items()},
            coverage=coverage, provenance=provenance, version="2.0")
