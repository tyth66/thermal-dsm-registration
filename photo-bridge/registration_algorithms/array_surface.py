"""Robust array M0 fitting and spatial holdout diagnostics."""
from __future__ import annotations

import numpy as np

from .local_plane import LocalPlane


MIN_PANELS = 8
MIN_TRAIN = 8
MIN_TEST = 3
N_FOLDS = 5
BUFFER_M = 1.5
ROBUST_ROUNDS = 4
RESIDUAL_LIMIT_M = 0.15


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
        residual = samples[:, 2] - plane.evaluate(samples[:, 0], samples[:, 1])
        center = float(np.median(residual))
        mad = float(np.median(np.abs(residual - center)))
        scale = 1.4826 * mad
        threshold = max(3.0 * scale, 0.05)
        candidate = np.abs(residual) <= threshold
        if np.array_equal(candidate, keep):
            break
        minimum_keep = max(6, int(np.ceil(0.5 * n)))
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


def _spatial_folds(en: np.ndarray, z: np.ndarray) -> tuple[list[dict], dict]:
    """Five contiguous PCA-axis folds with a 1.5 m exclusion buffer."""
    center = en.mean(axis=0)
    centered = en - center
    cov = centered.T @ centered / max(1, len(en))
    eigval, eigvec = np.linalg.eigh(cov)
    axis = eigvec[:, int(np.argmax(eigval))]
    # Preserve the legacy eigensolver sign. Flipping the PCA axis changes which
    # panels enter each unequal-sized fold, even though it is the same axis.
    coordinate = centered @ axis
    order = np.argsort(coordinate, kind='stable')
    chunks = np.array_split(order, N_FOLDS)
    folds = []
    edge_p95_values = []
    feasible_count = 0
    for fold_index, test_idx in enumerate(chunks):
        tcoord = coordinate[test_idx]
        lo, hi = float(np.min(tcoord)), float(np.max(tcoord))
        train_idx = np.flatnonzero((coordinate < lo - BUFFER_M) |
                                   (coordinate > hi + BUFFER_M))
        fold = {'fold': fold_index, 'side': 'west' if fold_index == 0 else
                ('east' if fold_index == N_FOLDS - 1 else 'interior'),
                'test_count': int(len(test_idx)), 'train_count': int(len(train_idx)),
                'test_s_min': lo, 'test_s_max': hi,
                'train_test_buffer_m': BUFFER_M, 'feasible': False}
        if len(train_idx):
            train_s = coordinate[train_idx]
            fold['train_s_min'] = float(train_s.min())
            fold['train_s_max'] = float(train_s.max())
            fold['nearest_train_distance_m'] = float(np.min(np.abs(
                np.concatenate([lo - train_s[train_s < lo],
                                train_s[train_s > hi] - hi]))))
        if len(train_idx) < MIN_TRAIN or len(test_idx) < MIN_TEST:
            fold['failure_reason'] = 'insufficient_train_or_test'
            folds.append(fold)
            continue
        train_fit = _robust_plane(np.column_stack([en[train_idx], z[train_idx]]))
        if train_fit['coeff'] is None:
            fold['failure_reason'] = train_fit['failure_reason'] or 'train_fit_failed'
            folds.append(fold)
            continue
        residuals = z[test_idx] - _predict(train_fit['coeff'], en[test_idx])
        abs_residuals = np.abs(residuals)
        fold.update(feasible=True, failure_reason=None,
                    median_abs_residual_m=float(np.median(abs_residuals)),
                    p95_abs_residual_m=_p95(abs_residuals),
                    max_abs_residual_m=float(np.max(abs_residuals)),
                    rmse_m=float(np.sqrt(np.mean(residuals ** 2))),
                    bias_m=float(np.mean(residuals)),
                    train_robust_support=int(train_fit['keep'].sum()),
                    train_robust_rounds=int(train_fit['rounds']))
        feasible_count += 1
        if fold_index in (0, N_FOLDS - 1):
            edge_p95_values.append(fold['p95_abs_residual_m'])
        folds.append(fold)
    summary = {'feasible_folds': feasible_count,
               'edge_p95_m': max(edge_p95_values) if edge_p95_values else None,
               'axis': [float(axis[0]), float(axis[1])],
               'axis_convention': 'numpy_eigh_native_sign',
               'buffer_m': BUFFER_M,
               'span_m': float(np.ptp(coordinate)),
               'axis_angle_deg_vs_E': float(np.degrees(np.arctan2(axis[1], axis[0]))),
               'pca_explained_ratio': float(np.max(eigval) / np.sum(eigval)) if np.sum(eigval) > 0 else None}
    return folds, summary


def gate(valid_panels: int, panel_count: int, feasible_folds: int,
         residual_p95_m: float | None, component_mad_m: float | None) -> dict:
    """Training-side diagnostic gate; it never controls coefficient use."""
    reasons = []
    if valid_panels < MIN_PANELS:
        reasons.append('valid_panels_lt_8')
    if panel_count < 20:
        reasons.append('panel_count_lt_20')
    if feasible_folds == 0:
        reasons.append('feasible_folds_eq_0')
    if residual_p95_m is not None and residual_p95_m > RESIDUAL_LIMIT_M:
        reasons.append('residual_p95_gt_0.15m')
    if component_mad_m is not None and component_mad_m > RESIDUAL_LIMIT_M:
        reasons.append('component_mad_gt_0.15m')
    return {'decision': 'reject_enter_direct' if reasons else 'pass', 'reasons': reasons}


def _quality_status(residual_p95: float | None, feasible_folds: int,
                    edge_p95: float | None, component_mad: float | None) -> tuple[str, str | None]:
    if residual_p95 is not None and residual_p95 > RESIDUAL_LIMIT_M:
        return 'surface_high_residual', f'in_sample_p95_{residual_p95:.3f}_gt_0.150'
    if feasible_folds == 0:
        return 'surface_low_support', 'no_feasible_holdout_fold'
    if edge_p95 is not None and edge_p95 > RESIDUAL_LIMIT_M:
        return 'surface_extrapolation_risk', f'edge_p95_{edge_p95:.3f}_gt_0.150'
    if component_mad is not None and component_mad > RESIDUAL_LIMIT_M:
        return 'surface_high_residual', f'component_mad_{component_mad:.3f}_gt_0.150'
    return 'surface_valid', None


def fit_array_plane(samples: list[tuple[float, float, float]],
                    component_mads: list[float] | None = None,
                    panel_count: int | None = None) -> dict:
    """Fit a locally centered robust M0 and return spatial/quality evidence.

    ``component_mads`` aligns with valid samples; ``panel_count`` counts every
    input panel in the array, including panels without usable DSM samples.
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
            'holdout_metrics': {'folds': [], 'summary': {
                'feasible_folds': 0, 'edge_p95_m': None, 'axis': None,
                'axis_convention': 'numpy_eigh_native_sign',
                'buffer_m': BUFFER_M,
                'span_m': None, 'axis_angle_deg_vs_E': None,
                'pca_explained_ratio': None}},
            'gate': gate(valid_count, count_all, 0, None, component_mad)}
    if valid_count < MIN_PANELS:
        return {**base, 'plane': None, 'plane_a': None, 'plane_b': None, 'plane_c': None,
                'quality_status': 'surface_low_support',
                'failure_reason': f'valid_panels_{valid_count}_lt_8'}

    fitted = _robust_plane(raw)
    coeff = fitted['coeff']
    if coeff is None:
        return {**base, 'plane': None, 'plane_a': None, 'plane_b': None, 'plane_c': None,
                'quality_status': 'surface_unavailable',
                'failure_reason': fitted['failure_reason']}
    residuals = raw[:, 2] - _predict(coeff, raw[:, :2])
    median_residual = float(np.median(residuals))
    residual_mad = float(np.median(np.abs(residuals - median_residual)))
    folds, holdout_summary = _spatial_folds(raw[:, :2], raw[:, 2])
    status, reason = _quality_status(_p95(np.abs(residuals)),
                                     holdout_summary['feasible_folds'],
                                     holdout_summary['edge_p95_m'], component_mad)
    gate_result = gate(valid_count, count_all, holdout_summary['feasible_folds'],
                       _p95(np.abs(residuals)), component_mad)
    plane = coeff
    a, b = plane.a, plane.b
    # Legacy global intercept is retained as a diagnostic/read-compatibility field only.
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
        'holdout_metrics': {'folds': folds, 'summary': holdout_summary},
        'quality_status': status, 'failure_reason': reason,
        'gate': gate_result,
    }
