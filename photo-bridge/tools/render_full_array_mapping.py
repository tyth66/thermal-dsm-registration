"""Render full-array current-algorithm mapping overviews."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

POINT_NAMES = ("center", "corner_0", "corner_1", "corner_2", "corner_3")


def _load_world(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _panel_arrays(world):
    return {panel["panel_id"]: str(panel["array_id"])
            for panel in world["panels"]}


def _read_image(path, target_width):
    with rasterio.open(path) as dataset:
        factor = target_width / dataset.width
        height = max(1, int(round(dataset.height * factor)))
        rgb = dataset.read([1, 2, 3], out_shape=(3, height, target_width),
                           resampling=rasterio.enums.Resampling.average)
        alpha = dataset.read(4, out_shape=(height, target_width),
                             resampling=rasterio.enums.Resampling.average)
    rgb = np.transpose(rgb, (1, 2, 0)).astype(np.float32) / 255.0
    alpha = alpha.astype(np.float32) / 255.0
    return rgb, alpha, height, target_width


def _panels_from_json(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    panels = []
    for panel in payload["panels"]:
        center = panel["center"]
        corners = panel["corners"]
        panels.append({
            "panel_id": panel["panel_id"],
            "center": center,
            "corners": corners,
            "corrected_valid": int(center.get("corrected_valid", 0)),
            "field_valid": int(center.get("field_valid", 1)),
            "conservative_valid": (None if center.get("conservative_valid") is None
                                   else int(center["conservative_valid"])),
            "valid": int(center.get("valid", 0)),
        })
    return panels


def _panels_from_csv(path):
    grouped = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(row["panel_id"], {})[row["point"]] = row
    panels = []
    for panel_id, points in grouped.items():
        center = points.get("center")
        corners = [points.get(name) for name in POINT_NAMES[1:]]
        if center is None or any(corner is None for corner in corners):
            continue
        panels.append({
            "panel_id": panel_id,
            "center": center,
            "corners": corners,
            "corrected_valid": int(center.get("corrected_valid", 0)),
            "field_valid": int(center.get("field_valid", 1)),
            "conservative_valid": (None if center.get("conservative_valid") is None
                                   else int(center["conservative_valid"])),
            "valid": int(center.get("valid", 0)),
        })
    return panels


def _xy(points, prefix):
    return [(float(point[prefix + "_u"]), float(point[prefix + "_v"]))
            for point in points]


def _render_axis(axis, image, alpha, panels, array_of_panel, prefix,
                 scale_x, scale_y, show_labels, title, offset=(0, 0)):
    height, width = image.shape[:2]
    offset_x, offset_y = offset
    axis.imshow(image, extent=(0, width, height, 0), alpha=alpha)
    valid_polygons = []
    invalid_polygons = []
    unsupported_polygons = []
    low_confidence_polygons = []
    for panel in panels:
        points = _xy(panel["corners"], prefix)
        if not all(bool(np.isfinite([x, y]).all()) for x, y in points):
            continue
        scaled = [(x * scale_x - offset_x, y * scale_y - offset_y)
                  for x, y in points]
        if not panel["corrected_valid"]:
            invalid_polygons.append(scaled)
        elif not panel["field_valid"]:
            unsupported_polygons.append(scaled)
        elif panel.get("conservative_valid") == 0:
            low_confidence_polygons.append(scaled)
        else:
            valid_polygons.append(scaled)
    axis.add_collection(PolyCollection(
        valid_polygons, facecolors=(0, 0, 0, 0),
        edgecolors=(0.1, 1.0, 0.25, 0.85), linewidths=0.45))
    axis.add_collection(PolyCollection(
        invalid_polygons, facecolors=(0, 0, 0, 0),
        edgecolors=(1.0, 0.55, 0.1, 0.9), linewidths=0.5))
    axis.add_collection(PolyCollection(
        unsupported_polygons, facecolors=(0, 0, 0, 0),
        edgecolors=(1.0, 0.1, 0.1, 0.9), linewidths=0.5))
    axis.add_collection(PolyCollection(
        low_confidence_polygons, facecolors=(0, 0, 0, 0),
        edgecolors=(1.0, 0.85, 0.1, 0.9), linewidths=0.5))
    if show_labels:
        by_array = {}
        for panel in panels:
            points = _xy([panel["center"]], prefix)
            if not points or not bool(np.isfinite(points[0]).all()):
                continue
            array_id = array_of_panel.get(panel["panel_id"], "?")
            by_array.setdefault(array_id, []).append(
                (points[0][0] * scale_x - offset_x,
                 points[0][1] * scale_y - offset_y))
        for array_id, points in sorted(by_array.items()):
            x = float(np.median([point[0] for point in points]))
            y = float(np.median([point[1] for point in points]))
            axis.text(x, y, "A" + array_id, color="white", fontsize=7.0,
                      ha="center", va="center",
                      bbox=dict(boxstyle="round,pad=0.12", fc="black",
                                ec="none", alpha=0.55))
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(title, fontsize=10, pad=4)


def _panel_bbox(panels, prefix, scale, image_width, image_height,
                margin_ratio=0.025):
    """Clipped pixel bbox of the finite panel corners (full image when empty)."""
    xs = []
    ys = []
    for panel in panels:
        for point in panel["corners"]:
            try:
                x = float(point[prefix + "_u"]) * scale
                y = float(point[prefix + "_v"]) * scale
            except (KeyError, TypeError, ValueError):
                continue
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            xs.append(x)
            ys.append(y)
    if not xs:
        return 0, 0, image_width, image_height
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    pad = max(x1 - x0, y1 - y0) * margin_ratio
    x0 = min(max(0, int(np.floor(x0 - pad))), image_width - 1)
    y0 = min(max(0, int(np.floor(y0 - pad))), image_height - 1)
    x1 = min(image_width, int(np.ceil(x1 + pad)))
    y1 = min(image_height, int(np.ceil(y1 + pad)))
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def _crop(image, alpha, bbox):
    x, y, width, height = bbox
    return image[y:y + height, x:x + width], alpha[y:y + height, x:x + width]


def _add_scale_bar(axis, width, height, meters_per_pixel, rendered_width,
                   original_width):
    factor = rendered_width / original_width
    pixels_per_meter = factor / meters_per_pixel
    bar_px = 100.0 * pixels_per_meter
    x0 = width * 0.04
    y0 = height * 0.96
    axis.plot([x0, x0 + bar_px], [y0, y0], color="white", lw=2.5)
    axis.text(x0 + bar_px / 2, y0 - height * 0.012, "100 m",
              color="white", ha="center", va="bottom", fontsize=9)


def _raster_metadata(path):
    """Ground resolution (m/px) and pixel dimensions of a raster."""
    with rasterio.open(path) as dataset:
        return abs(dataset.transform.a), dataset.width, dataset.height


def render(args):
    world = _load_world(args.world_model)
    array_of_panel = _panel_arrays(world)
    if args.result_json is not None:
        panels = _panels_from_json(args.result_json)
    else:
        panels = _panels_from_csv(args.panel_csv)
    thermal_resolution, thermal_width, _ = _raster_metadata(args.thermal_ortho)
    _, visible_width, _ = _raster_metadata(args.visible_ortho)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    station = args.station_name

    # Full-array thermal image.
    target_width = args.overview_width
    thermal_rgb, thermal_alpha, thermal_h, _ = _read_image(
        args.thermal_ortho, target_width)
    thermal_scale = target_width / thermal_width
    thermal_bbox = _panel_bbox(panels, "thermal", thermal_scale,
                               target_width, thermal_h)
    thermal_crop, thermal_crop_alpha = _crop(
        thermal_rgb, thermal_alpha, thermal_bbox)
    crop_h, crop_w = thermal_crop.shape[:2]
    figure = plt.figure(figsize=(crop_w / 120, crop_h / 120), dpi=120)
    axis = figure.add_axes([0.005, 0.005, 0.99, 0.985])
    _render_axis(
        axis, thermal_crop, thermal_crop_alpha, panels, array_of_panel, "thermal",
        thermal_scale, thermal_scale, True,
        "%s: current algorithm mapping" % station,
        offset=(thermal_bbox[0], thermal_bbox[1]))
    _add_scale_bar(axis, crop_w, crop_h, thermal_resolution,
                   target_width, thermal_width)
    corrected_count = sum(panel["corrected_valid"] for panel in panels)
    conservative_count = sum(
        panel["corrected_valid"] if panel.get("conservative_valid") is None
        else panel["conservative_valid"] for panel in panels)
    axis.text(
        0.01, 0.985,
        "%s  panels=%d  arrays=%d  corrected_valid=%d  conservative_valid=%d  thermal resolve=%.3f m/px" %
        (station, len(panels), len(set(array_of_panel.values())), corrected_count,
         conservative_count, thermal_resolution),
        transform=axis.transAxes, color="white", fontsize=11, va="top",
        bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.55))
    axis.legend(handles=[
        Patch(edgecolor=(0.1, 1.0, 0.25, 0.85), facecolor="none", label="corrected + field valid"),
        Patch(edgecolor=(1.0, 0.55, 0.1, 0.9), facecolor="none", label="no corrected coverage"),
        Patch(edgecolor=(1.0, 0.1, 0.1, 0.9), facecolor="none", label="field unsupported"),
        Patch(edgecolor=(1.0, 0.85, 0.1, 0.9), facecolor="none", label="conservative gate failed"),
    ], loc="lower left", fontsize=9, framealpha=0.55, facecolor="black",
       labelcolor="white")
    figure.savefig(out_dir / ("%s_full_arrays.png" % station), dpi=120,
                   facecolor="black")
    plt.close(figure)

    # Side-by-side visible and thermal.
    side_width = args.side_width
    visible_rgb, visible_alpha, visible_h, _ = _read_image(
        args.visible_ortho, side_width)
    thermal_rgb2, thermal_alpha2, thermal_h2, _ = _read_image(
        args.thermal_ortho, side_width)
    visible_scale = side_width / visible_width
    thermal_side_scale = side_width / thermal_width
    visible_bbox = _panel_bbox(panels, "visible", visible_scale,
                               side_width, visible_h)
    thermal_bbox2 = _panel_bbox(panels, "thermal", thermal_side_scale,
                                side_width, thermal_h2)
    visible_crop, visible_crop_alpha = _crop(
        visible_rgb, visible_alpha, visible_bbox)
    thermal_crop2, thermal_crop_alpha2 = _crop(
        thermal_rgb2, thermal_alpha2, thermal_bbox2)
    fig_h = max(visible_crop.shape[0], thermal_crop2.shape[0]) / 120.0
    figure = plt.figure(figsize=(side_width * 2 / 120, fig_h), dpi=120)
    ax_visible = figure.add_axes([0.005, 0.005, 0.49, 0.985])
    ax_thermal = figure.add_axes([0.505, 0.005, 0.49, 0.985])
    _render_axis(
        ax_visible, visible_crop, visible_crop_alpha, panels, array_of_panel,
        "visible", visible_scale, visible_scale,
        True, "%s: visible reference" % station,
        offset=(visible_bbox[0], visible_bbox[1]))
    _render_axis(
        ax_thermal, thermal_crop2, thermal_crop_alpha2, panels, array_of_panel,
        "thermal", thermal_side_scale, thermal_side_scale,
        True, "%s: current corrected mapping" % station,
        offset=(thermal_bbox2[0], thermal_bbox2[1]))
    figure.savefig(out_dir / ("%s_visible_vs_thermal.png" % station),
                   dpi=120, facecolor="black")
    plt.close(figure)

    return {
        "station": station,
        "panels": len(panels),
        "arrays": len(set(array_of_panel.values())),
        "corrected_valid": corrected_count,
        "conservative_valid": conservative_count,
        "outputs": [
            str(out_dir / ("%s_full_arrays.png" % station)),
            str(out_dir / ("%s_visible_vs_thermal.png" % station)),
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--station-name", required=True)
    parser.add_argument("--thermal-ortho", type=Path, required=True)
    parser.add_argument("--visible-ortho", type=Path, required=True)
    parser.add_argument("--world-model", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--panel-csv", type=Path)
    group.add_argument("--result-json", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--overview-width", type=int, default=4200)
    parser.add_argument("--side-width", type=int, default=3000)
    args = parser.parse_args(argv)
    print(json.dumps(render(args), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
