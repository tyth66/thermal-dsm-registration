
"""In-memory state for the residual-mapping application pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..domain import CanonicalResidualModel, GeometryModel, PanelVerticalModel
from ..residual_model import PlanimetricField
from .compose import ComposeInputs, compose_residual_model


@dataclass
class PipelineContext:
    world: Mapping[str, Any]
    planimetric: PlanimetricField
    panel_model: PanelVerticalModel
    geometry: GeometryModel
    provenance: Mapping[str, Any]
    canonical: CanonicalResidualModel | None = None
    state: dict[str, Any] = field(default_factory=dict)

    @property
    def available_panels(self) -> int:
        return sum(1 for panel in self.panel_model.panels.values()
                   if panel.center_xyz is not None)

    def compose(self) -> CanonicalResidualModel:
        if self.canonical is None:
            self.canonical = compose_residual_model(ComposeInputs(
                world_panels=tuple(self.world["panels"]),
                planimetric=self.planimetric,
                panel_model=self.panel_model,
                geometry=self.geometry,
                provenance=self.provenance,
            ))
        return self.canonical
