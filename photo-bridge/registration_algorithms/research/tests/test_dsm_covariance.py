"""DSM-profile peak covariance tests."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, shift as ndi_shift

from registration_algorithms.masked_ecc import masked_ecc_candidates
from registration_algorithms.research.dsm_covariance import (
    DSM_COVARIANCE_MAX_PX2, DSM_COVARIANCE_MIN_PX2,
    dsm_candidate_covariance,
)


def _tile(noise=0.03):
    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:120, 0:120]
    base = (np.exp(-((x - 45) ** 2 + (y - 55) ** 2) / 200)
            + 0.7 * np.exp(-((x - 80) ** 2 + (y - 30) ** 2) / 120)
            + 0.3 * np.exp(-((x - 20) ** 2 + (y - 90) ** 2) / 80))
    visible = gaussian_filter(base, 1.5) + noise * rng.normal(size=base.shape)
    thermal = ndi_shift(visible, shift=(2, 3), order=1, mode="nearest")
    return visible, thermal


def test_dsm_profile_covariance_is_positive_and_physical():
    visible, thermal = _tile()
    joint = np.ones_like(visible, bool)
    candidates = masked_ecc_candidates(
        visible, thermal, joint, resolution_m=0.1, top_k=1,
        covariance_fn=dsm_candidate_covariance)
    assert candidates
    candidate = candidates[0]
    assert candidate["sx"] == pytest.approx(3.0, abs=0.2)
    assert candidate["sy"] == pytest.approx(2.0, abs=0.2)
    assert candidate["covariance_source"].startswith("dsm")
    assert candidate["beta_eff"] > 0.0
    assert candidate["sigma_z2"] > 0.0
    covariance = np.asarray(candidate["covariance_px"], float)
    eigenvalue = np.linalg.eigvalsh(covariance)
    assert np.all(eigenvalue > 0.0)
    assert np.all(eigenvalue >= DSM_COVARIANCE_MIN_PX2 - 1e-12)
    assert np.all(eigenvalue <= DSM_COVARIANCE_MAX_PX2 + 1e-12)


def test_default_covariance_remains_normalized():
    visible, thermal = _tile()
    joint = np.ones_like(visible, bool)
    candidates = masked_ecc_candidates(
        visible, thermal, joint, resolution_m=0.1, top_k=1)
    covariance = np.asarray(candidates[0]["covariance_px"], float)
    assert np.isfinite(covariance).all()
    assert np.linalg.eigvalsh(covariance).min() > 0.0
    assert "covariance_source" not in candidates[0]