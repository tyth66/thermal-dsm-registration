"""DSM height-profile candidate covariance (research only)."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import shift as ndi_shift

from ..masked_ecc import _peak_covariance, _peak_curvature

DSM_COVARIANCE_MIN_PX2 = 1e-4
DSM_COVARIANCE_MAX_PX2 = 100.0
DSM_MIN_OVERLAP_PX = 50


def _regression(visible, thermal, joint, sx, sy):
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


def dsm_candidate_covariance(visible, thermal, joint, sx, sy, ecc, iy, ix):
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
    regression = _regression(visible, thermal, joint, sx, sy)
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