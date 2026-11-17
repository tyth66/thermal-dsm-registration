"""Build the visible world model from standard panel OBBs, visible ortho and DSM."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS

from .array_surface import fit_array_plane
from .dsm_sampling import center_elevation, sample_footprint
from .local_plane import LocalPlane


def _is_metre_projected(crs: CRS) -> bool:
    if not crs.is_projected:
        return False
    unit_name, unit_factor = crs.linear_units_factor
    return (unit_name.lower() in ("metre", "meter", "metres", "meters")
            and math.isclose(float(unit_factor), 1.0, abs_tol=1e-12))


def _corners(panel: dict, transform) -> np.ndarray:
    x, y = float(panel["x"]), float(panel["y"])
    width, height = float(panel["width"]), float(panel["height"])
    if not all(math.isfinite(v) for v in (x, y, width, height)) or width <= 0 or height <= 0:
        raise ValueError("OBB x/y/width/height must be finite and dimensions positive")
    a = -math.radians(float(panel["rotation"]))
    ux, uy = math.cos(a), math.sin(a)
    vx, vy = -math.sin(a), math.cos(a)
    xy = np.array([
        [x + width / 2 * ux + height / 2 * vx,
         y + width / 2 * uy + height / 2 * vy],
        [x - width / 2 * ux + height / 2 * vx,
         y - width / 2 * uy + height / 2 * vy],
        [x - width / 2 * ux - height / 2 * vx,
         y - width / 2 * uy - height / 2 * vy],
        [x + width / 2 * ux - height / 2 * vx,
         y + width / 2 * uy - height / 2 * vy],
    ], dtype=float)
    en = np.empty_like(xy)
    en[:, 0] = transform.a * xy[:, 0] + transform.b * xy[:, 1] + transform.c
    en[:, 1] = transform.d * xy[:, 0] + transform.e * xy[:, 1] + transform.f
    return en


def _diagnostic(stage: str, code: str, item: str, message: str) -> dict:
    return {"stage": stage, "code": code, "item": item, "message": message,
            "severity": "warning"}


def build_visible_world_model(panels_json: Path, visible_ortho: Path,
                              visible_dsm: Path, dsm_erode_m: float = 0.35,
                              minimum_dsm_pixels: int = 20) -> dict:
    diagnostics = []
    panels_doc = json.loads(Path(panels_json).read_text(encoding="utf-8"))
    if panels_doc.get("schema_version") != "standard-panels/v1" or not isinstance(panels_doc.get("panels"), list):
        raise ValueError("panels input must follow standard-panels/v1 with a panels list")
    raw_panels = panels_doc["panels"]
    valid_rows = []
    ids = set()
    for index, row in enumerate(raw_panels):
        if not isinstance(row, dict) or row.get("id") is None or not str(row["id"]).strip():
            diagnostics.append(_diagnostic("world_model", "invalid_panel", "index:%d" % index,
                                           "component needs a non-empty id"))
            continue
        panel_id = str(row["id"])
        if panel_id in ids:
            raise ValueError("duplicate panel ids are ambiguous: %s" % panel_id)
        ids.add(panel_id)
        valid_rows.append(row)
    if (not math.isfinite(dsm_erode_m) or dsm_erode_m < 0 or minimum_dsm_pixels < 1):
        raise ValueError("DSM sampling options are outside their valid ranges")

    with rasterio.open(visible_ortho) as ortho, rasterio.open(visible_dsm) as dsm:
        if ortho.crs is None or dsm.crs is None:
            raise ValueError("visible orthophoto and DSM must both declare a CRS")
        ortho_crs, dsm_crs = CRS.from_user_input(ortho.crs), CRS.from_user_input(dsm.crs)
        if ortho_crs != dsm_crs:
            raise ValueError("orthophoto and DSM CRS differ: %s != %s" % (ortho_crs, dsm_crs))
        if not _is_metre_projected(ortho_crs):
            raise ValueError("world model requires a projected metre CRS, got %s" % ortho_crs)
        records = []
        for raw in valid_rows:
            panel_id = str(raw["id"])
            array_id = "" if raw.get("arrayId") is None else str(raw["arrayId"])
            try:
                if not array_id:
                    raise ValueError("arrayId is required")
                rotation = float(raw["rotation"])
                if not math.isfinite(rotation):
                    raise ValueError("OBB rotation must be finite")
                corners = _corners(raw, ortho.transform)
                center = corners.mean(axis=0)
                footprint = sample_footprint(dsm, corners, dsm_erode_m)
                footprint["sample_usable"] = (
                    footprint["valid_pixels"] >= minimum_dsm_pixels
                    and footprint["median_z"] is not None)
                if not footprint["sample_usable"]:
                    diagnostics.append(_diagnostic(
                        "dsm_sampling", "insufficient_valid_samples", panel_id,
                        "valid pixels=%d, valid ratio=%.6f" % (
                            footprint["valid_pixels"], footprint["valid_ratio"])))
                records.append({"panel_id": panel_id, "array_id": array_id,
                                "center": center, "corners": corners,
                                "sample": footprint})
            except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
                diagnostics.append(_diagnostic("world_model", "invalid_panel", panel_id, str(exc)))

        by_array = {}
        for record in records:
            by_array.setdefault(record["array_id"], []).append(record)
        surfaces = {}
        for array_id, items in sorted(by_array.items()):
            samples = [(float(item["center"][0]), float(item["center"][1]),
                        float(item["sample"]["median_z"]))
                       for item in items if item["sample"]["sample_usable"]]
            mads = [float(item["sample"]["mad_z"]) for item in items
                    if item["sample"]["sample_usable"]]
            model = fit_array_plane(samples, component_mads=mads, panel_count=len(items))
            model["array_id"] = array_id
            model["quality_scope"] = ("legacy_M0_DSM_spatial_holdout_consistency; "
                                      "not physical truth")
            surfaces[array_id] = model
            if model["failure_reason"]:
                diagnostics.append(_diagnostic(
                    "surface_fit", model["failure_reason"], array_id,
                    "array surface status=%s; support=%s" % (
                        model["quality_status"], model["support"])))

        world_panels = []
        panel_quality = {}
        for item in records:
            panel_id, array_id = item["panel_id"], item["array_id"]
            footprint = item["sample"]
            surface = surfaces[array_id]
            plane_model = surface.get("plane")
            if plane_model is None and all(surface.get(key) is not None and
                                           math.isfinite(surface[key])
                                           for key in ("plane_a", "plane_b", "plane_c")):
                # Compatibility for previously serialized surfaces.
                plane_model = [surface["plane_a"], surface["plane_b"], surface["plane_c"]]
            try:
                plane = LocalPlane.from_model(plane_model) if plane_model is not None else None
            except ValueError:
                plane = None
            has_plane = plane is not None
            if has_plane and surface["quality_status"] not in (
                    "surface_unavailable", "surface_model_not_applicable"):
                center_z = float(plane.evaluate(item["center"][0], item["center"][1]))
                corner_z = np.asarray(plane.evaluate(item["corners"][:, 0],
                                                     item["corners"][:, 1]), dtype=float)
                z_source = "m0_plane"
            else:
                center_z = center_elevation(dsm, item["center"])
                corner_z = (np.full(4, center_z, dtype=float)
                            if center_z is not None else None)
                z_source = "site_z_fallback"
            panel_quality[panel_id] = {
                "array_id": array_id, "dsm_median_z": footprint["median_z"],
                "dsm_mad_z": footprint["mad_z"],
                "dsm_valid_pixels": footprint["valid_pixels"],
                "dsm_total_pixels": footprint["total_pixels"],
                "dsm_valid_ratio": footprint["valid_ratio"],
                "dsm_sample_usable": footprint["sample_usable"],
                "z_source": z_source,
                "quality_status": surface["quality_status"],
                "model_valid": z_source == "m0_plane",
                "z_reason": (None if z_source == "m0_plane"
                             else "invalid_plane:missing_plane_model"),
                "surface_failure_reason": surface["failure_reason"],
                "surface_model_id": "array-plane:%s" % array_id,
            }
            if center_z is None or corner_z is None or not np.isfinite([center_z, *corner_z]).all():
                panel_quality[panel_id]["z_source"] = "site_z_unavailable"
                panel_quality[panel_id]["z_reason"] += ";site_z_unavailable"
                diagnostics.append(_diagnostic(
                    "world_model", "elevation_unavailable", panel_id,
                    "no usable array plane or valid DSM pixel at panel center"))
                continue
            world_panels.append({
                "panel_id": panel_id, "array_id": array_id,
                "center_xyz": [float(item["center"][0]), float(item["center"][1]),
                               float(center_z)],
                "corners_xyz": [[float(row[0]), float(row[1]), float(row[2])]
                                for row in np.column_stack([item["corners"], corner_z])],
                "z_source": z_source,
                "surface_model_id": "array-plane:%s" % array_id,
            })

    return {
        "plane_representation": "local_en/v1",
        "panels": world_panels,
        "surfaces": surfaces,
        "panel_quality": panel_quality,
        "diagnostics": diagnostics,
        "crs": ortho_crs.to_string(),
        "input_panel_count": len(raw_panels),
    }
