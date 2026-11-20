"""Planimetric component shifts for the position-mapping pipeline."""
from __future__ import annotations

import numpy as np

from .residual_model import PlanimetricField


def component_shifts(visible_panels, field: PlanimetricField):
    """Planimetric shifts/support for every visible component.

    The pipeline maps positions only: no vertical model and no thermal DSM read
    participates here.
    """
    rows = []
    if not visible_panels:
        return rows
    centers = np.asarray([panel["center_xyz"][:2] for panel in visible_panels], float)
    corners = np.asarray([np.asarray(panel["corners_xyz"], float)[:, :2]
                          for panel in visible_panels], float)
    dx, dy, weights, _ = field.evaluate(centers[:, 0], centers[:, 1])
    support_points = np.concatenate([centers[:, None, :], corners], axis=1).reshape(-1, 2)
    support_valid = field.supports_points(support_points).reshape(len(visible_panels), 5)
    for index, panel in enumerate(visible_panels):
        east, north = (float(value) for value in panel["center_xyz"][:2])
        rows.append({
            "panel_id": panel["panel_id"], "array_id": str(panel["array_id"]),
            "east": east, "north": north,
            "delta_e_thermal_vs_visible_m": float(dx[index]),
            "delta_n_thermal_vs_visible_m": float(dy[index]),
            "horizontal_confidence": float(weights[index]),
            "horizontal_valid": int(bool(support_valid[index].all())),
        })
    return rows
