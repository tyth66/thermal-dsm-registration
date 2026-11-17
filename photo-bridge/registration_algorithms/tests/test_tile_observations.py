"""Tests for tile-level candidate displacement observations."""
from __future__ import annotations

import numpy as np

from registration_algorithms.tile_observations import tile_candidates


def _tile(shift_x: int, shift_y: int):
    rng = np.random.default_rng(7)
    base = rng.normal(0.0, 1.0, (48, 48)).astype(np.float32)
    # Add a non-periodic edge so the phase candidate is unambiguous.
    base[10:16, 8:34] += 3.0
    base[30:34, 12:30] -= 2.0
    shifted = np.roll(base, shift=(shift_y, shift_x), axis=(0, 1))
    return base, shifted


def test_phase_candidates_recover_known_shift():
    base, shifted = _tile(7, -5)
    mask = np.ones_like(base, dtype=bool)
    candidates = tile_candidates(base, shifted, mask, resolution_m=0.1,
                                 top_k=4, min_separation_px=2)
    assert candidates
    best = candidates[0]
    assert abs(best["dx_m"] - 7 * 0.1) <= 0.4
    assert abs(best["dy_m"] - 5 * 0.1) <= 0.4
    assert best["score"] > 0.5
