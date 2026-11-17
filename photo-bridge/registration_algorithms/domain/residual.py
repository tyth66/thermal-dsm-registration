
"""Canonical residual model as typed domain objects.

The JSON schema is an adapter boundary; application code should pass these
objects instead of raw dictionaries.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..local_plane import LocalPlane
from ..residual_model import PlanimetricField, validate_residual_model


def _plane_from_model(value: Any) -> LocalPlane | None:
    return None if value is None else LocalPlane.from_model(value)


@dataclass(frozen=True)
class PanelResidual:
    array_id: str | None
    visible_plane: LocalPlane | None
    thermal_plane: LocalPlane | None
    thermal_correction_method: str

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "PanelResidual":
        return cls(
            array_id=(None if row.get("array_id") is None else str(row["array_id"])),
            visible_plane=_plane_from_model(row.get("visible_plane")),
            thermal_plane=_plane_from_model(row.get("thermal_plane")),
            thermal_correction_method=str(
                row.get("thermal_correction_method", "array_bias")),
        )

    def to_row(self) -> dict:
        return {
            "array_id": self.array_id,
            "visible_plane": (None if self.visible_plane is None
                              else self.visible_plane.to_dict()),
            "thermal_plane": (None if self.thermal_plane is None
                              else self.thermal_plane.to_dict()),
            "thermal_correction_method": self.thermal_correction_method,
        }


@dataclass(frozen=True)
class CanonicalResidualModel:
    version: str
    planimetric: PlanimetricField
    panels: Mapping[str, PanelResidual]
    coverage: Mapping[str, Any]
    provenance: Mapping[str, Any]

    @classmethod
    def from_dict(cls, model: Mapping[str, Any]) -> "CanonicalResidualModel":
        validate_residual_model(dict(model))
        vertical = model["vertical"]
        return cls(
            version=str(model.get("version", "2.0")),
            planimetric=PlanimetricField.from_model(model["planimetric"]),
            panels={panel_id: PanelResidual.from_row(row)
                    for panel_id, row in vertical["panels"].items()},
            coverage=dict(vertical.get("coverage", {})),
            provenance=dict(model.get("provenance", {})),
        )

    @classmethod
    def from_parts(cls, planimetric: PlanimetricField,
                   vertical_rows: Mapping[str, Mapping[str, Any]],
                   coverage: Mapping[str, Any], provenance: Mapping[str, Any],
                   version: str = "2.0") -> "CanonicalResidualModel":
        return cls(
            version=str(version),
            planimetric=planimetric,
            panels={panel_id: PanelResidual.from_row(row)
                    for panel_id, row in vertical_rows.items()},
            coverage=dict(coverage),
            provenance=dict(provenance),
        )

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "planimetric": self.planimetric.to_model(),
            "vertical": {
                "panels": {panel_id: panel.to_row()
                           for panel_id, panel in self.panels.items()},
                "coverage": dict(self.coverage),
            },
            "provenance": dict(self.provenance),
        }

    def mapping_panels(self) -> dict[str, dict]:
        """Shape consumed by registration_algorithms.forward_mapping."""
        return {
            panel_id: {
                "visible_plane": panel.visible_plane,
                "thermal_plane": panel.thermal_plane,
                "thermal_correction_method": panel.thermal_correction_method,
                "array_id": panel.array_id,
            }
            for panel_id, panel in self.panels.items()
        }
