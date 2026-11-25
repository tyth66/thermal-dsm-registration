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
from scipy.ndimage import maximum_filter, shift as ndi_shift

DEFAULT_TOP_K = 5
DEFAULT_NMS_PX = 2
SIGMA0_M = 0.05
SOFTMAX_BETA = 50.0
PEAK_COVARIANCE_SCALE_PX2 = 1.0
PEAK_COVARIANCE_RATIO_MIN = 0.05
PEAK_COVARIANCE_RATIO_MAX = 10.0
DSM_COVARIANCE_MIN_PX2 = 1e-4
DSM_COVARIANCE_MAX_PX2 = 100.0
DSM_MIN_OVERLAP_PX = 50


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
                       joint: np.ndarray, max_shift_px: int | None = None):
    """Return integer shifts, ECC and common-mask area for one tile.

    visible/thermal are masked DSM arrays; joint is the boolean common mask.
    Invalid pixels are excluded by the mask, not filled with zero.
    """
    visible = np.asarray(visible, float)
    thermal = np.asarray(thermal, float)
    joint = np.asarray(joint, bool)
    if visible.shape != thermal.shape or visible.shape != joint.shape:
        raise ValueError("visible, thermal and joint must share one shape")
    height, width = joint.shape
    mask = joint.astype(float)
    v = np.where(joint, visible, 0.0)
    t = np.where(joint, thermal, 0.0)

    c_m = _corr(mask, mask)
    c_v = _corr(mask * v, mask)
    c_t = _corr(mask, mask * t)
    c_vv = _corr(mask * v * v, mask)
    c_tt = _corr(mask, mask * t * t)
    c_vt = _corr(mask * v, mask * t)

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


def _peak_curvature(ecc: np.ndarray, iy: int, ix: int):
    """Positive-definite Q = -Hessian(rho) at an interior peak, or None."""
    if not (0 < iy < ecc.shape[0] - 1 and 0 < ix < ecc.shape[1] - 1):
        return None
    center = float(ecc[iy, ix])
    hxx = float(ecc[iy, ix - 1]) - 2.0 * center + float(ecc[iy, ix + 1])
    hyy = float(ecc[iy - 1, ix]) - 2.0 * center + float(ecc[iy + 1, ix])
    hxy = 0.25 * (float(ecc[iy + 1, ix + 1])
                  - float(ecc[iy + 1, ix - 1])
                  - float(ecc[iy - 1, ix + 1])
                  + float(ecc[iy - 1, ix - 1]))
    precision = -np.array([[hxx, hxy], [hxy, hyy]], float)
    precision = 0.5 * (precision + precision.T)
    eigenvalue = np.linalg.eigvalsh(precision)
    if not np.all(np.isfinite(eigenvalue)) or np.any(eigenvalue <= 1e-9):
        return None
    return precision


def _peak_covariance(ecc: np.ndarray, iy: int, ix: int) -> np.ndarray:
    """Normalized inverse-Hessian covariance in px^2 (engineering fallback)."""
    precision = _peak_curvature(ecc, iy, ix)
    if precision is None:
        return np.eye(2)
    covariance = np.linalg.inv(precision)
    covariance = 0.5 * (covariance + covariance.T)
    mean_variance = 0.5 * float(np.trace(covariance))
    if not np.isfinite(mean_variance) or mean_variance <= 0:
        return PEAK_COVARIANCE_SCALE_PX2 * np.eye(2)
    covariance = covariance * (PEAK_COVARIANCE_SCALE_PX2 / mean_variance)
    eigenvalue, eigenvector = np.linalg.eigh(covariance)
    eigenvalue = np.clip(eigenvalue,
                         PEAK_COVARIANCE_SCALE_PX2 * PEAK_COVARIANCE_RATIO_MIN,
                         PEAK_COVARIANCE_SCALE_PX2 * PEAK_COVARIANCE_RATIO_MAX)
    return eigenvector @ np.diag(eigenvalue) @ eigenvector.T


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


def _dsm_regression(visible, thermal, joint, sx, sy):
    """Fit z_v(x)=alpha*z_t(x+s)+b over the shifted common mask."""
    shifted = ndi_shift(np.asarray(thermal, float),
                         shift=(-float(sy), -float(sx)), order=1,
                         mode="constant", cval=np.nan, prefilter=False)
    valid = (np.asarray(joint, bool) & np.isfinite(shifted)
             & np.isfinite(visible))
    count = int(valid.sum())
    if count < DSM_MIN_OVERLAP_PX:
        return None
    v = np.asarray(visible, float)[valid]
    t = shifted[valid]
    t_centered = t - t.mean()
    v_centered = v - v.mean()
    denominator = float(np.sum(t_centered ** 2))
    if denominator <= 1e-12:
        return None
    alpha = float(np.sum(t_centered * v_centered) / denominator)
    residual = v_centered - alpha * t_centered
    sse = float(np.sum(residual ** 2))
    sst = float(np.sum(v_centered ** 2))
    sigma_z2 = sse / max(count - 2, 1)
    if not np.isfinite(sigma_z2) or sigma_z2 <= 0.0 or sst <= 0.0:
        return None
    return count, sst, sigma_z2, alpha


def _dsm_candidate_covariance(visible, thermal, joint, sx, sy, ecc, iy, ix):
    """DSM-profile covariance C=(beta_eff Q)^-1 for one candidate."""
    fallback = _peak_covariance(ecc, iy, ix)
    curvature = _peak_curvature(ecc, iy, ix)
    result = {
        "covariance_px": fallback,
        "beta_eff": None,
        "sst": None,
        "sigma_z2": None,
        "covariance_source": "fallback",
    }
    if curvature is None:
        return result
    regression = _dsm_regression(visible, thermal, joint, sx, sy)
    if regression is None:
        return result
    count, sst, sigma_z2, _ = regression
    rho = float(ecc[iy, ix])
    if not np.isfinite(rho) or rho <= 0.0:
        return result
    beta_eff = float(sst / sigma_z2 * rho)
    if not np.isfinite(beta_eff) or beta_eff <= 0.0:
        return result
    covariance = np.linalg.inv(beta_eff * curvature)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalue, eigenvector = np.linalg.eigh(covariance)
    clipped = np.clip(eigenvalue,
                       DSM_COVARIANCE_MIN_PX2, DSM_COVARIANCE_MAX_PX2)
    covariance = eigenvector @ np.diag(clipped) @ eigenvector.T
    result.update({
        "covariance_px": covariance,
        "beta_eff": beta_eff,
        "sst": sst,
        "sigma_z2": sigma_z2,
        "covariance_source": ("dsm" if np.allclose(clipped, eigenvalue)
                               else "dsm_clipped"),
        "dsm_overlap_px": count,
    })
    return result


def masked_ecc_candidates(visible: np.ndarray, thermal: np.ndarray,
                          joint: np.ndarray, resolution_m: float,
                          top_k: int = DEFAULT_TOP_K,
                          nms_px: int = DEFAULT_NMS_PX,
                          max_shift_px: int | None = None,
                          prior: tuple[float, float] | None = None,
                          radius_px: float | None = None,
                          dsm_covariance: bool = False) -> list[dict]:
    """Return the strongest masked-ECC translation peaks.

    Each dict carries subpixel sx/sy in pixels, cc, and integer peak indices.
    With ``dsm_covariance=True`` the reported covariance_px comes from the
    DSM height-profile likelihood C=(beta_eff Q)^-1; otherwise it is the
    normalized engineering inverse Hessian.
    """
    shifts, ecc, _ = masked_ecc_surface(visible, thermal, joint, max_shift_px)
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
        candidate = {
            "sx": sx + dx, "sy": sy + dy,
            "cc": float(ecc[iy, ix]),
            "sx_int": int(sx), "sy_int": int(sy),
            "covariance_px": _peak_covariance(ecc, iy, ix).tolist(),
        }
        if dsm_covariance:
            statistics = _dsm_candidate_covariance(
                visible, thermal, joint, candidate["sx"], candidate["sy"],
                ecc, iy, ix)
            covariance = statistics.pop("covariance_px")
            candidate["covariance_px"] = covariance.tolist()
            candidate.update(statistics)
        chosen.append(candidate)
        if len(chosen) >= top_k:
            break
    return chosen


CANDIDATE_MODE_MASS_MIN = 0.73
CANDIDATE_MODE_BANDWIDTH_PX = 2.0
CANDIDATE_MODE_DEDUPE_PX = 0.5


def candidate_mode_quality(candidates, resolution_m: float,
                           bandwidth_px: float = CANDIDATE_MODE_BANDWIDTH_PX):
    """Return 2-D mean-shift mode masses for the top ECC candidates.

    Candidate scores are converted to softmax weights with the same beta as
    the covariance estimator.  Mean-shift with a Gaussian kernel groups peaks
    that are closer than one NMS radius; the remaining modes are assigned the
    candidate mass.  A top mode below CANDIDATE_MODE_MASS_MIN is a competing
    mode rather than a unique peak.
    """
    if not candidates:
        return {"mode_count": 0, "top_mass": 0.0, "second_mass": 0.0,
                "mode_separation_m": 0.0}
    positions = np.asarray([[item["sx"], item["sy"]]
                            for item in candidates], float)
    scores = np.asarray([item["cc"] for item in candidates], float)
    logits = SOFTMAX_BETA * (scores - float(np.max(scores)))
    weights = np.exp(logits)
    total_weight = float(weights.sum())
    weights = weights / total_weight if total_weight > 0 else np.full(
        len(candidates), 1.0 / len(candidates))

    modes = []
    bandwidth = float(bandwidth_px)
    for start in positions:
        center = start.copy()
        for _ in range(50):
            delta = positions - center
            kernel = np.exp(-0.5 * np.einsum("ij,ij->i", delta, delta)
                            / (bandwidth * bandwidth))
            local = weights * kernel
            local_total = float(local.sum())
            if local_total <= 0:
                break
            updated = (local[:, None] * positions).sum(axis=0) / local_total
            if np.linalg.norm(updated - center) < 1e-6:
                center = updated
                break
            center = updated
        if not any(np.linalg.norm(center - mode) <= CANDIDATE_MODE_DEDUPE_PX
                   for mode in modes):
            modes.append(center)

    if not modes:
        return {"mode_count": 0, "top_mass": 0.0, "second_mass": 0.0,
                "mode_separation_m": 0.0}
    mode_array = np.asarray(modes, float)
    squared = np.sum((positions[:, None, :] - mode_array[None, :, :]) ** 2,
                     axis=2)
    labels = np.argmin(squared, axis=1)
    masses = np.bincount(labels, weights=weights, minlength=len(modes))
    order = np.argsort(masses)[::-1]
    top_mass = float(masses[order[0]])
    second_mass = float(masses[order[1]]) if len(order) > 1 else 0.0
    if len(order) > 1:
        separation_px = float(np.linalg.norm(
            mode_array[order[0]] - mode_array[order[1]]))
    else:
        separation_px = 0.0
    return {"mode_count": int(len(modes)), "top_mass": top_mass,
            "second_mass": second_mass,
            "mode_separation_m": separation_px * float(resolution_m)}


def estimate_period_m(visible: np.ndarray, joint: np.ndarray,
                      resolution_m: float, max_period_m: float = 50.0) -> float | None:
    """Estimate a dominant repeat period from the masked DSM tile.

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
