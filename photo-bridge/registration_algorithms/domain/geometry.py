
"""Typed panel-level geometry derived from the visible/thermal footprints."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..local_plane import LocalPlane, plane_from_row


def _number(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value in (None, "", "None"):
        return None
    return float(value)


def _plane(row: Mapping[str, Any], prefix: str) -> LocalPlane | None:
    return plane_from_row(dict(row), prefix)


@dataclass(frozen=True)
class PanelGeometry:
    panel_id: str
    array_id: str
    visible_plane: LocalPlane | None
    thermal_plane: LocalPlane | None
    visible_plane_rms_m: float | None
    thermal_plane_rms_m: float | None
    visible_offset_m: float | None
    thermal_offset_m: float | None
    delta_offset_m: float | None
    visible_tilt_a: float | None
    visible_tilt_b: float | None
    thermal_tilt_a: float | None
    thermal_tilt_b: float | None
    delta_tilt_a: float | None
    delta_tilt_b: float | None
    visible_samples: int | None
    thermal_samples: int | None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "PanelGeometry":
        def integer(key: str) -> int | None:
            value = row.get(key)
            return None if value in (None, "", "None") else int(value)
        return cls(
            panel_id=str(row["panel_id"]),
            array_id=str(row.get("array_id", "")),
            visible_plane=_plane(row, "visible_plane"),
            thermal_plane=_plane(row, "thermal_plane"),
            visible_plane_rms_m=_number(row, "visible_plane_rms_m"),
            thermal_plane_rms_m=_number(row, "thermal_plane_rms_m"),
            visible_offset_m=_number(row, "visible_offset_m"),
            thermal_offset_m=_number(row, "thermal_offset_m"),
            delta_offset_m=_number(row, "delta_offset_m"),
            visible_tilt_a=_number(row, "visible_tilt_a"),
            visible_tilt_b=_number(row, "visible_tilt_b"),
            thermal_tilt_a=_number(row, "thermal_tilt_a"),
            thermal_tilt_b=_number(row, "thermal_tilt_b"),
            delta_tilt_a=_number(row, "delta_tilt_a"),
            delta_tilt_b=_number(row, "delta_tilt_b"),
            visible_samples=integer("visible_samples"),
            thermal_samples=integer("thermal_samples"),
        )


@dataclass(frozen=True)
class GeometryModel:
    panels: Mapping[str, PanelGeometry]

    @classmethod
    def from_rows(cls, rows: list[Mapping[str, Any]]) -> "GeometryModel":
        return cls(panels={row["panel_id"]: PanelGeometry.from_row(row) for row in rows})
