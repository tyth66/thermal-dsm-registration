
"""Typed panel-level vertical model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..local_plane import LocalPlane


@dataclass(frozen=True)
class PanelPlane:
    plane: LocalPlane
    rms_m: float | None = None
    support: int | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "PanelPlane | None":
        if value is None:
            return None
        plane = LocalPlane.from_model(value)
        return cls(plane=plane,
                   rms_m=(None if value.get("rms_m") is None else float(value["rms_m"])),
                   support=(None if value.get("support") is None else int(value["support"])))

    def to_dict(self) -> dict:
        result = self.plane.to_dict()
        if self.rms_m is not None:
            result["rms_m"] = float(self.rms_m)
        if self.support is not None:
            result["support"] = int(self.support)
        return result


@dataclass(frozen=True)
class ArraySurface:
    plane: LocalPlane | None
    quality_status: str
    support: int

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "ArraySurface":
        plane = None if row.get("plane") is None else LocalPlane.from_model(row["plane"])
        return cls(plane=plane,
                   quality_status=str(row.get("quality_status", "surface_unavailable")),
                   support=int(row.get("support", 0)))

    def to_row(self) -> dict:
        return {
            "plane": None if self.plane is None else self.plane.to_dict(),
            "quality_status": self.quality_status,
            "support": self.support,
        }


@dataclass(frozen=True)
class PanelVertical:
    panel_id: str
    array_id: str
    center_xyz: tuple[float, float, float] | None
    corners_xyz: tuple[tuple[float, float, float], ...] | None
    z_source: str
    panel_plane: PanelPlane | None
    vertical_bias_m: float | None = None
    vertical_correction_method: str | None = None

    @classmethod
    def from_row(cls, panel_id: str, row: Mapping[str, Any]) -> "PanelVertical":
        center = row.get("center_xyz")
        corners = row.get("corners_xyz")
        return cls(
            panel_id=str(panel_id),
            array_id=str(row.get("array_id", "")),
            center_xyz=(None if center is None else tuple(float(value) for value in center)),
            corners_xyz=(None if corners is None else tuple(
                tuple(float(value) for value in point) for point in corners)),
            z_source=str(row.get("z_source", "unavailable")),
            panel_plane=PanelPlane.from_dict(row.get("panel_plane")),
            vertical_bias_m=(None if row.get("vertical_bias_m") is None
                             else float(row["vertical_bias_m"])),
            vertical_correction_method=(None if row.get("vertical_correction_method") is None
                                        else str(row["vertical_correction_method"])),
        )

    def to_row(self) -> dict:
        row = {
            "array_id": self.array_id,
            "center_xyz": None if self.center_xyz is None else list(self.center_xyz),
            "corners_xyz": (None if self.corners_xyz is None
                            else [list(point) for point in self.corners_xyz]),
            "z_source": self.z_source,
            "panel_plane": None if self.panel_plane is None else self.panel_plane.to_dict(),
        }
        if self.vertical_bias_m is not None:
            row["vertical_bias_m"] = float(self.vertical_bias_m)
        if self.vertical_correction_method is not None:
            row["vertical_correction_method"] = self.vertical_correction_method
        return row


@dataclass(frozen=True)
class PanelVerticalModel:
    plane_representation: str
    footprint_sampling: str
    inputs: Mapping[str, Any]
    arrays: Mapping[str, ArraySurface]
    panels: Mapping[str, PanelVertical]
    provenance: Mapping[str, Any]
    array_height_corrections: Mapping[str, Any] | None = None

    @classmethod
    def from_dict(cls, model: Mapping[str, Any]) -> "PanelVerticalModel":
        return cls(
            plane_representation=str(model.get("plane_representation", "local_en/v1")),
            footprint_sampling=str(model.get("footprint_sampling", "sample_footprint/v1")),
            inputs=dict(model.get("inputs", {})),
            arrays={array_id: ArraySurface.from_row(row)
                    for array_id, row in model.get("arrays", {}).items()},
            panels={panel_id: PanelVertical.from_row(panel_id, row)
                    for panel_id, row in model["panels"].items()},
            provenance=dict(model.get("provenance", {})),
            array_height_corrections=(None if model.get("array_height_corrections") is None
                                      else dict(model["array_height_corrections"])),
        )

    @classmethod
    def from_parts(cls, arrays: Mapping[str, Mapping[str, Any]],
                   panels: Mapping[str, Mapping[str, Any]],
                   *, inputs: Mapping[str, Any] | None = None,
                   provenance: Mapping[str, Any] | None = None,
                   plane_representation: str = "local_en/v1",
                   footprint_sampling: str = "sample_footprint/v1") -> "PanelVerticalModel":
        return cls(
            plane_representation=plane_representation,
            footprint_sampling=footprint_sampling,
            inputs=dict(inputs or {}),
            arrays={array_id: ArraySurface.from_row(row)
                    for array_id, row in arrays.items()},
            panels={panel_id: PanelVertical.from_row(panel_id, row)
                    for panel_id, row in panels.items()},
            provenance=dict(provenance or {}),
        )

    def to_dict(self) -> dict:
        result = {
            "plane_representation": self.plane_representation,
            "footprint_sampling": self.footprint_sampling,
            "inputs": dict(self.inputs),
            "arrays": {array_id: array.to_row()
                       for array_id, array in self.arrays.items()},
            "panels": {panel_id: panel.to_row()
                       for panel_id, panel in self.panels.items()},
            "provenance": dict(self.provenance),
        }
        if self.array_height_corrections is not None:
            result["array_height_corrections"] = dict(self.array_height_corrections)
        return result
