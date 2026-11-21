"""Masked ECC translation search for DSM tiles.

The OpenCV ECC path compares the whole tile, including zero-filled nodata
pixels.  This module computes the same normalised correlation over the
candidate-dependent joint mask instead:

    M_s(x) = m_visible(x) & m_thermal(x + s)

All sums are obtained with one FFT cross-correlation per term, so a complete
integer-shift surface costs six FFTs per tile.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter

DEFAULT_TOP_K = 5
DEFAULT_NMS_PX = 2
AMBIGUITY_MARGIN = 0.02
FB_TOLERANCE_M = 0.20
SIGMA0_M = 0.05
SOFTMAX_BETA = 50.0


def _corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Linear cross-correlation: C[s] = sum_x a(x) * b(x + s).

    The result is fftshifted so the zero shift sits at (H, W).
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    height, width = a.shape
    shape = (2 * height, 2 * width)
    fa = np.fft.rfft2(a, s=shape)
    fb = np.fft.rfft2(b, s=shape)
    correlation = np.fft.irfft2(np.conj(fa) * fb, s=shape)
    return np.fft.fftshift(correlation)


def masked_ecc_surface(visible: np.ndarray, thermal: np.ndarray,
                       joint: np.ndarray, max_shift_px: int | None = None,
                       visible_mask: np.ndarray | None = None,
                       thermal_mask: np.ndarray | None = None):
    """Return integer shifts, ECC and common-mask area for one tile.

    visible/thermal are high-pass arrays; joint is the boolean common mask.
    Invalid pixels are excluded by the mask, not filled with zero.
    """
    visible = np.asarray(visible, float)
    thermal = np.asarray(thermal, float)
    joint = np.asarray(joint, bool)
    if visible.shape != thermal.shape or visible.shape != joint.shape:
        raise ValueError("visible, thermal and joint must share one shape")
    height, width = joint.shape
    mv = (joint if visible_mask is None else visible_mask)
    mt = (joint if thermal_mask is None else thermal_mask)
    mv = np.asarray(mv, float)
    mt = np.asarray(mt, float)
    v = np.where(mv > 0, visible, 0.0)
    t = np.where(mt > 0, thermal, 0.0)

    # C_m(s) = sum_x mv(x) mt(x+s); with one shared joint mask this reduces to
    # the previous J(x)J(x+s) overlap.
    c_m = _corr(mv, mt)
    c_v = _corr(mv * v, mt)
    c_t = _corr(mv, mt * t)
    c_vv = _corr(mv * v * v, mt)
    c_tt = _corr(mv, mt * t * t)
    c_vt = _corr(mv * v, mt * t)

    if max_shift_px is None:
        max_shift_px = min(height, width) // 2
    limit = max(0, min(int(max_shift_px), height - 1, width - 1))
    shifts = np.arange(-limit, limit + 1)
    sl = (slice(height - limit, height + limit + 1),
          slice(width - limit, width + limit + 1))
    m = c_m[sl]
    v_sum = c_v[sl]
    t_sum = c_t[sl]
    vv_sum = c_vv[sl]
    tt_sum = c_tt[sl]
    vt_sum = c_vt[sl]

    # At least half of the zero-shift common area must remain visible; for a
    # fully valid tile this is the 0.5*|T| contract in the plan.
    zero_shift_area = float(c_m[height, width])
    valid = m > 0.0
    if zero_shift_area > 0:
        valid &= m >= 0.5 * zero_shift_area
    ecc = np.full(m.shape, np.nan, float)
    if np.any(valid):
        mv = v_sum[valid] / m[valid]
        mt = t_sum[valid] / m[valid]
        numerator = vt_sum[valid] - v_sum[valid] * t_sum[valid] / m[valid]
        var_v = vv_sum[valid] - v_sum[valid] ** 2 / m[valid]
        var_t = tt_sum[valid] - t_sum[valid] ** 2 / m[valid]
        var_v = np.maximum(var_v, 0.0)
        var_t = np.maximum(var_t, 0.0)
        denominator = np.sqrt(var_v * var_t)
        good = denominator > 1e-12
        values = np.full(len(denominator), np.nan, float)
        values[good] = numerator[good] / denominator[good]
        ecc[valid] = values
        # Keep the means for diagnostics? not needed by callers.
    return shifts, ecc, m


def _subpixel(ecc: np.ndarray, iy: int, ix: int) -> tuple[float, float]:
    """Parabolic subpixel offset around a sampled peak, clamped to +/-0.5 px."""
    if not (0 < iy < ecc.shape[0] - 1 and 0 < ix < ecc.shape[1] - 1):
        return 0.0, 0.0
    center = ecc[iy, ix]
    left, right = ecc[iy, ix - 1], ecc[iy, ix + 1]
    up, down = ecc[iy - 1, ix], ecc[iy + 1, ix]
    if not np.all(np.isfinite([center, left, right, up, down])):
        return 0.0, 0.0
    denom_x = left - 2.0 * center + right
    denom_y = up - 2.0 * center + down
    dx = 0.5 * (left - right) / denom_x if abs(denom_x) > 1e-12 else 0.0
    dy = 0.5 * (up - down) / denom_y if abs(denom_y) > 1e-12 else 0.0
    return float(np.clip(dx, -0.5, 0.5)), float(np.clip(dy, -0.5, 0.5))


def masked_ecc_candidates(visible: np.ndarray, thermal: np.ndarray,
                          joint: np.ndarray, resolution_m: float,
                          top_k: int = DEFAULT_TOP_K,
                          nms_px: int = DEFAULT_NMS_PX,
                          max_shift_px: int | None = None,
                          prior: tuple[float, float] | None = None,
                          radius_px: float | None = None,
                          visible_mask: np.ndarray | None = None,
                          thermal_mask: np.ndarray | None = None) -> list[dict]:
    """Return the strongest masked-ECC translation peaks.

    Each dict carries subpixel sx/sy in pixels, cc, and integer peak indices.
    """
    shifts, ecc, _ = masked_ecc_surface(
        visible, thermal, joint, max_shift_px,
        visible_mask=visible_mask, thermal_mask=thermal_mask)
    valid = np.isfinite(ecc)
    if not np.any(valid):
        return []
    surface = np.where(valid, ecc, -np.inf)
    local_max = maximum_filter(surface, size=2 * nms_px + 1,
                               mode="constant", cval=-np.inf)
    peak_indices = np.argwhere(valid & (surface >= local_max - 1e-12))
    if len(peak_indices) == 0:
        iy, ix = np.unravel_index(np.nanargmax(ecc), ecc.shape)
        peak_indices = np.array([[iy, ix]])
    scores = ecc[peak_indices[:, 0], peak_indices[:, 1]]
    order = np.argsort(scores)[::-1]
    chosen: list[dict] = []
    for index in order:
        iy, ix = int(peak_indices[index, 0]), int(peak_indices[index, 1])
        sx = float(shifts[ix])
        sy = float(shifts[iy])
        if prior is not None and radius_px is not None:
            if np.hypot(sx - prior[0], sy - prior[1]) > radius_px:
                continue
        if any((sx - item["sx"]) ** 2 + (sy - item["sy"]) ** 2
               < nms_px ** 2 for item in chosen):
            continue
        dx, dy = _subpixel(ecc, iy, ix)
        chosen.append({
            "sx": sx + dx, "sy": sy + dy,
            "cc": float(ecc[iy, ix]),
            "sx_int": int(sx), "sy_int": int(sy),
        })
        if len(chosen) >= top_k:
            break
    return chosen


def estimate_period_m(visible: np.ndarray, joint: np.ndarray,
                      resolution_m: float, max_period_m: float = 50.0) -> float | None:
    """Estimate a dominant repeat period from the masked high-pass tile.

    Returns None when the autocorrelation has no clear off-centre peak.  The
    value is diagnostic only; the ambiguity gate falls back to separation.
    """
    visible = np.asarray(visible, float)
    joint = np.asarray(joint, bool)
    if not np.any(joint):
        return None
    v = np.where(joint, visible, 0.0)
    mask = joint.astype(float)
    c = _corr(mask * v, mask * v)
    height, width = joint.shape
    center = float(c[height, width])
    if center <= 0:
        return None
    limit = min(height, width) // 2
    best = None
    for iy in range(height - limit, height + limit + 1):
        for ix in range(width - limit, width + limit + 1):
            sy, sx = iy - height, ix - width
            if sy == 0 and sx == 0:
                continue
            distance = float(np.hypot(sy, sx))
            if distance < 10.0:
                continue
            value = c[iy, ix] / center
            if value < 0.40:
                continue
            period_m = distance * resolution_m
            if period_m < 2.0 or period_m > max_period_m:
                continue
            if best is None or value > best[0]:
                best = (value, period_m)
    return None if best is None else float(best[1])
