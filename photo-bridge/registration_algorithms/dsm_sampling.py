"""Generic DSM footprint sampling shared by world-model builders.

The helpers rasterize a component OBB footprint (world metres) into a DSM raster,
apply the fixed metric elliptical erosion, and summarize valid elevations. They are
I/O helpers over an already-open rasterio dataset; plane fitting lives in surface.py.
"""
from __future__ import annotations

import math

import numpy as np
import cv2
from rasterio.windows import Window

MIN_ELEVATION_M = -100.0
MAX_ELEVATION_M = 80.0


def window_for_points(points: np.ndarray, inverse, width: int,
                      height: int) -> Window | None:
    """Smallest integer raster window containing world points, clipped to the raster."""
    pix = np.asarray([inverse @ (float(x), float(y)) for x, y in points], dtype=float)
    c0 = max(0, int(math.floor(float(pix[:, 0].min()))))
    r0 = max(0, int(math.floor(float(pix[:, 1].min()))))
    c1 = min(width, int(math.ceil(float(pix[:, 0].max()))) + 1)
    r1 = min(height, int(math.ceil(float(pix[:, 1].max()))) + 1)
    if c1 <= c0 or r1 <= r0:
        return None
    return Window(c0, r0, c1 - c0, r1 - r0)


def dsm_pixels(ds, window: Window):
    """Read one DSM window and build its validity mask (finite, mask, nodata)."""
    data = ds.read(1, window=window, masked=False)
    valid = (ds.read_masks(1, window=window) > 0) & np.isfinite(data)
    if ds.nodata is not None:
        if math.isnan(float(ds.nodata)):
            valid &= np.isfinite(data)
        else:
            valid &= data != float(ds.nodata)
    return data, valid


def sample_footprint(ds, corners: np.ndarray, erode_m: float, *,
                     include_samples: bool = False) -> dict:
    """Sample a polygon interior after a fixed metric elliptical erosion.

    Returns median/MAD of valid elevations plus pixel-support statistics. The
    elevation range is the legacy visible-side range (-100, 80) m; callers that
    need another range must filter afterwards.
    With include_samples, samples contains aligned E/N/Z arrays from exactly
    the same usable mask; E/N are world coordinates of pixel centers.
    """
    corners = np.asarray(corners, dtype=float).reshape(4, 2)
    if not np.isfinite(corners).all() or not math.isfinite(float(erode_m)) or erode_m < 0:
        raise ValueError("corners must be finite and erode_m nonnegative")
    window = window_for_points(corners, ~ds.transform, ds.width, ds.height)
    if window is None:
        result = {"median_z": None, "mad_z": None, "valid_pixels": 0,
                  "total_pixels": 0, "valid_ratio": 0.0}
        if include_samples:
            result["samples"] = tuple(np.empty(0, dtype=float) for _ in range(3))
        return result
    values, valid = dsm_pixels(ds, window)
    pixel_corners = np.asarray([~ds.transform @ (float(e), float(n)) for e, n in corners])
    local = np.rint(pixel_corners - [window.col_off, window.row_off]).astype(np.int32)
    mask = np.zeros(values.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [local.reshape(-1, 1, 2)], 1)

    transform = ds.transform
    pixel_x_m = math.hypot(float(transform.a), float(transform.d))
    pixel_y_m = math.hypot(float(transform.b), float(transform.e))
    if pixel_x_m <= 0 or pixel_y_m <= 0:
        raise ValueError("DSM transform has a zero-sized pixel axis")
    radius_x = int(round(float(erode_m) / pixel_x_m))
    radius_y = int(round(float(erode_m) / pixel_y_m))
    if radius_x or radius_y:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (2 * radius_x + 1, 2 * radius_y + 1))
        mask = cv2.erode(mask, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    eroded = mask > 0
    total = int(eroded.sum())
    usable = (eroded & valid & np.isfinite(values)
              & (values > MIN_ELEVATION_M) & (values < MAX_ELEVATION_M))
    samples = values[usable]
    median = float(np.median(samples)) if len(samples) else None
    result = {"median_z": median,
              "mad_z": float(np.median(np.abs(samples - median))) if len(samples) else None,
              "valid_pixels": int(len(samples)), "total_pixels": total,
              "valid_ratio": float(len(samples) / total) if total else 0.0}
    if include_samples:
        rows, columns = np.nonzero(usable)
        east, north = ds.transform @ (columns + window.col_off + 0.5,
                                      rows + window.row_off + 0.5)
        result["samples"] = (np.asarray(east, float), np.asarray(north, float),
                             samples)
    return result


def sample_footprint_points(ds, corners, erode_m: float, minimum_pixels: int = 60):
    """Return E/N/Z samples for plane fitting, or None below required support."""
    result = sample_footprint(ds, corners, erode_m, include_samples=True)
    if result["valid_pixels"] < minimum_pixels or result["valid_pixels"] == 0:
        return None
    return result["samples"]


def center_elevation(ds, point: np.ndarray) -> float | None:
    """Elevation of the DSM pixel containing a world point; never borrows nearby pixels."""
    col_f, row_f = ~ds.transform @ (float(point[0]), float(point[1]))
    col, row = int(math.floor(col_f)), int(math.floor(row_f))
    if not (0 <= col < ds.width and 0 <= row < ds.height):
        return None
    values, valid = dsm_pixels(ds, Window(col, row, 1, 1))
    z = float(values[0, 0])
    return z if valid[0, 0] and MIN_ELEVATION_M < z < MAX_ELEVATION_M else None

