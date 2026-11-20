"""Robust array M0 fitting in locally centered EN coordinates."""
from __future__ import annotations

import numpy as np

from .local_plane import LocalPlane


ROBUST_ROUNDS = 4


def _plane(samples: np.ndarray) -> tuple[LocalPlane | None, int]:
    """Least-squares fit in a locally centered EN frame."""
    e0, n0, z0 = map(float, np.mean(samples, axis=0))
    design = np.column_stack([samples[:, 0] - e0, samples[:, 1] - n0,
                              np.ones(len(samples))])
    centered_z = samples[:, 2] - z0
    coeff, _, rank, _ = np.linalg.lstsq(design, centered_z, rcond=None)
    plane = (LocalPlane(e0, n0, z0 + float(coeff[2]),
                        float(coeff[0]), float(coeff[1]))
             if np.isfinite(coeff).all() else None)
    return plane, int(rank)


def _robust_plane(samples: np.ndarray) -> dict:
    n = len(samples)
    keep = np.ones(n, dtype=bool)
    rejected_update = False
    rounds = 0
    threshold = None
    plane = None
    rank = 0
    for _ in range(ROBUST_ROUNDS):
        rounds += 1
        fitted_keep = keep.copy()
        plane, rank = _plane(samples[fitted_keep])
        if plane is None:
            return {'coeff': None, 'keep': keep, 'rank': rank, 'rounds': rounds,
                    'failure_reason': 'non_finite_coefficients', 'threshold_m': threshold}
        if rank < 3:
            # Computation guard, not a support gate: a plane is only defined
            # when the design is identifiable from the available samples.
            return {'coeff': None, 'keep': keep, 'rank': rank, 'rounds': rounds,
                    'failure_reason': 'rank_below_3', 'threshold_m': threshold}
        residual = samples[:, 2] - plane.evaluate(samples[:, 0], samples[:, 1])
        center = float(np.median(residual))
        mad = float(np.median(np.abs(residual - center)))
        scale = 1.4826 * mad
        # Clip around the residual median.  The least-squares intercept already
        # keeps the kept-sample residual mean near zero, so this only hardens
        # the first round against one-sided contamination.  The 3.0x MAD
        # multiplier and the 0.05 m floor are deliberate calibration for this
        # stage (panel-median array surfaces) and must not change.
        threshold = max(3.0 * scale, 0.05)
        candidate = np.abs(residual - center) <= threshold
        if np.array_equal(candidate, keep):
            break
        minimum_keep = max(3, int(np.ceil(0.5 * n)))
        if int(candidate.sum()) < minimum_keep:
            rejected_update = True
            break
        keep = candidate
    return {'coeff': plane, 'keep': fitted_keep, 'rank': rank, 'rounds': rounds,
            'failure_reason': 'rejected_small_inlier_update' if rejected_update else None,
            'threshold_m': threshold}


def _predict(plane: LocalPlane, en: np.ndarray) -> np.ndarray:
    return np.asarray(plane.evaluate(en[:, 0], en[:, 1]), dtype=float)


def _p95(values) -> float | None:
    a = np.asarray(values, dtype=float)
    return float(np.percentile(a, 95)) if len(a) else None


def fit_array_plane(samples: list[tuple[float, float, float]],
                    component_mads: list[float] | None = None,
                    panel_count: int | None = None) -> dict:
    """Fit a locally centered robust M0 and report its computation status.

    component_mads aligns with valid samples; panel_count counts every input
    panel in the array, including panels without usable DSM samples. There is
    no minimum-support admission gate: the plane is fitted whenever the rank-3
    design is identifiable, otherwise the surface is unavailable.
    """
    raw = np.asarray(samples, dtype=float)
    if raw.size == 0:
        raw = np.empty((0, 3), dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 3 or not np.isfinite(raw).all():
        raise ValueError('samples must be finite (E, N, Z) triples')
    valid_count = int(len(raw))
    count_all = valid_count if panel_count is None else int(panel_count)
    if count_all < valid_count:
        raise ValueError('panel_count cannot be smaller than valid sample count')
    mads = np.asarray([] if component_mads is None else component_mads, dtype=float)
    if mads.size and (mads.ndim != 1 or not np.isfinite(mads).all()):
        raise ValueError('component_mads must contain finite values')
    component_mad = float(np.median(mads)) if len(mads) else None

    base = {'support': valid_count, 'valid_panels': valid_count,
            'panel_count': count_all, 'residual_mad': None, 'residual_rmse': None,
            'in_sample_residual_p95_m': None, 'component_mad_m': component_mad,
            'robust_inlier_support': 0, 'robust_rounds': 0,
            'robust_threshold_m': None, 'robust_failure_reason': None,
            'failure_reason': None}
    if valid_count < 3:
        # Rank-3 identifiability is impossible below three samples; this is a
        # computation guard rather than the removed minimum-support gate.
        return {**base, 'plane': None, 'plane_a': None, 'plane_b': None,
                'plane_c': None,
                'failure_reason': 'valid_panels_%d_lt_3' % valid_count}

    fitted = _robust_plane(raw)
    coeff = fitted['coeff']
    if coeff is None:
        return {**base, 'plane': None, 'plane_a': None, 'plane_b': None,
                'plane_c': None, 'failure_reason': fitted['failure_reason']}
    residuals = raw[:, 2] - _predict(coeff, raw[:, :2])
    median_residual = float(np.median(residuals))
    residual_mad = float(np.median(np.abs(residuals - median_residual)))
    plane = coeff
    a, b = plane.a, plane.b
    # Legacy global intercept is retained as a read-compatibility field only.
    c = plane.z0 - a * plane.e0 - b * plane.n0
    return {
        **base,
        'plane': plane.to_dict(),
        'plane_a': a, 'plane_b': b, 'plane_c': c,
        'support': valid_count, 'robust_inlier_support': int(fitted['keep'].sum()),
        'robust_rounds': int(fitted['rounds']), 'robust_threshold_m': fitted['threshold_m'],
        'robust_failure_reason': fitted['failure_reason'],
        'residual_mad': residual_mad,
        'residual_rmse': float(np.sqrt(np.mean(residuals ** 2))),
        'in_sample_residual_p95_m': _p95(np.abs(residuals)),
        'component_mad_m': component_mad,
        'failure_reason': None,
    }
