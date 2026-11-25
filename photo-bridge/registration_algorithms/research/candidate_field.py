"""Joint candidate-field EM with a tile-level outlier component.

The full candidate-set density is

    p(S_i | f) = prod_k u_i(s_ik) *
                 [a0 + (1 - a0) sum_j pi_ij B_ij(f)]

    B_ij(f) = N(s_ij; f(p_i), C_ij + sigma_f^2 I) / u_i(s_ij)

where u_i is the false-peak background density.  The common product cancels
from responsibilities, so the EM code carries the dimensionless evidence
ratio B_ij.  The posterior tile mass m_i = 1 - q_i0 is what lets an all-raw
tile leave the field fit; normalized candidate weights alone sum to one.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logsumexp
from scipy.spatial import cKDTree

from ..residual_model import RegularizedPlanimetricField, fit_regularized_field

CANDIDATE_CORE_FLOOR_M = 0.05
AMBIGUITY_MASS_MIN = 0.73


def _candidate_arrays(records, resolution_m):
    """Return padded candidate positions, scores and covariances in metres."""
    count = len(records)
    if count == 0:
        raise ValueError("candidate-field EM requires at least one tile")
    candidates = [record.get("candidates") for record in records]
    if any(not item for item in candidates):
        raise ValueError("every tile must carry a non-empty candidates list")
    width = max(len(item) for item in candidates)
    positions = np.full((count, width, 2), np.nan)
    scores = np.full((count, width), -np.inf)
    covariances = np.full((count, width, 2, 2), np.nan)
    counts = np.zeros(count, int)
    flip = np.diag([float(resolution_m), -float(resolution_m)])
    for index, items in enumerate(candidates):
        counts[index] = len(items)
        for column, item in enumerate(items):
            positions[index, column] = [
                float(item["sx"]) * resolution_m,
                -float(item["sy"]) * resolution_m,
            ]
            scores[index, column] = float(item["cc"])
            covariance_px = np.asarray(
                item.get("covariance_px", np.eye(2)), float)
            covariances[index, column] = (
                flip @ covariance_px @ flip.T
                + CANDIDATE_CORE_FLOOR_M ** 2 * np.eye(2))
    return positions, scores, covariances, counts


def _log_gaussian(delta, covariance):
    """Log N(delta; 0, covariance), including the normalizing constant."""
    sign, log_det = np.linalg.slogdet(covariance)
    if np.any(sign <= 0):
        raise ValueError("candidate covariance must be positive definite")
    distance = np.einsum("ij,ijk,ik->i", delta,
                         np.linalg.inv(covariance), delta)
    return -0.5 * (2.0 * np.log(2.0 * np.pi) + log_det + distance)


def _tile_covariances(records):
    result = np.empty((len(records), 2, 2), float)
    for index, record in enumerate(records):
        sigma_e = float(record["sigma_e"])
        sigma_n = float(record["sigma_n"])
        sigma_en = float(record["sigma_en"])
        result[index] = [[sigma_e ** 2, sigma_en],
                         [sigma_en, sigma_n ** 2]]
    return result


def _background_array(value, count: int, width: int, radius_m: float):
    if value is None:
        value = 1.0 / (2.0 * float(radius_m)) ** 2
    array = np.asarray(value, float)
    if array.ndim == 0:
        array = np.full((count, width), float(array))
    elif array.ndim == 1:
        if len(array) == count:
            array = np.broadcast_to(array[:, None], (count, width))
        elif len(array) == width:
            array = np.broadcast_to(array[None, :], (count, width))
        else:
            raise ValueError(
                "background_density must have length N or K when 1-D")
    elif array.ndim == 2:
        array = np.broadcast_to(array, (count, width))
    else:
        raise ValueError("background_density must be scalar, 1-D or 2-D")
    if np.any(~np.isfinite(array)) or np.any(array <= 0):
        raise ValueError("background_density must be positive and finite")
    return array


def candidate_responsibilities(records, field, *, resolution_m: float,
                               beta: float = 50.0,
                               sigma_f_m: float = 0.5,
                               outlier_prior: float = 0.1,
                               outlier_radius_m: float = 4.0,
                               background_density=None,
                               temperature: float = 1.0):
    """Return q_ij, q_i0 and effective tile mass m_i for every tile.

    Responsibilities use the dimensionless evidence ratio
    B_ij = N(s_ij; f(p_i), V_ij) / u_i(s_ij).  The common product of
    background densities over the candidate set cancels from q_i0/q_ij, so it
    is not carried here.  ``background_density`` may be a scalar, an (N,)
    array (one density per tile) or an (N, K) array.
    """
    if not 0.0 < outlier_prior < 1.0:
        raise ValueError("outlier_prior must be in (0, 1)")
    if outlier_radius_m <= 0:
        raise ValueError("outlier_radius_m must be positive")
    positions, scores, covariances, counts = _candidate_arrays(
        records, resolution_m)
    background = _background_array(
        background_density, len(records), scores.shape[1], outlier_radius_m)
    east = np.asarray([record["east"] for record in records], float)
    north = np.asarray([record["north"] for record in records], float)
    field_east, field_north, _, _ = field.evaluate(east, north)
    field_shift = np.column_stack([field_east, field_north])
    q = np.zeros_like(scores)
    q0 = np.zeros(len(records))
    tile_mass = np.zeros(len(records))
    candidate_weights = np.zeros_like(scores)
    precision = np.zeros((len(records), 2, 2))
    ambiguity_loss = np.zeros((len(records), 2, 2))
    information = np.zeros((len(records), 2, 2))
    validity_entropy = np.zeros(len(records))
    alias_entropy = np.zeros(len(records))
    total_entropy = np.zeros(len(records))
    log_match_evidence = np.full(len(records), -np.inf)
    match_evidence = np.zeros(len(records))
    log_odds = np.full(len(records), -np.inf)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    log_prior_odds = np.log1p(-outlier_prior) - np.log(outlier_prior)
    for index in range(len(records)):
        count = int(counts[index])
        weight = scores[index, :count].copy()
        weight = np.exp(beta * (weight - np.max(weight)))
        weight /= np.sum(weight)
        covariance = covariances[index, :count] + (
            float(sigma_f_m) ** 2) * np.eye(2)
        inverse_covariance = np.linalg.inv(covariance)
        delta = positions[index, :count] - field_shift[index]
        log_likelihood = _log_gaussian(delta, covariance)
        density = background[index, :count]
        log_b = log_likelihood - np.log(density)
        log_e = logsumexp(np.log(np.maximum(weight, 1e-300)) + log_b)
        log_match_evidence[index] = log_e
        match_evidence[index] = float(np.exp(log_e))
        log_odds[index] = (log_prior_odds + log_e) / float(temperature)
        mass = float(expit(log_odds[index]))
        tile_mass[index] = mass
        q0[index] = 1.0 - mass
        w = np.exp((np.log(np.maximum(weight, 1e-300)) + log_b - log_e)
                   / float(temperature))
        w /= np.sum(w)
        candidate_weights[index, :count] = w
        q[index, :count] = mass * w
        if 0.0 < mass < 1.0:
            validity_entropy[index] = (-mass * np.log(mass)
                                       - (1.0 - mass) * np.log1p(-mass))
        positive = w > 0.0
        alias_entropy[index] = -float(np.sum(w[positive] * np.log(w[positive])))
        total_entropy[index] = (validity_entropy[index]
                                + mass * alias_entropy[index])
        v = np.einsum("kij,kj->ki", inverse_covariance, delta)
        weighted_v = q[index, :count] @ v
        weighted_vv = np.einsum(
            "k,ki,kj->ij", q[index, :count], v, v)
        precision[index] = np.einsum(
            "k,kij->ij", q[index, :count], inverse_covariance)
        ambiguity_loss[index] = (weighted_vv
                                 - np.outer(weighted_v, weighted_v))
        information[index] = precision[index] - ambiguity_loss[index]
    return {
        "q": q,
        "q0": q0,
        "tile_mass": tile_mass,
        "candidate_weights": candidate_weights,
        "precision": precision,
        "ambiguity_loss": ambiguity_loss,
        "information": information,
        "validity_entropy": validity_entropy,
        "alias_entropy": alias_entropy,
        "total_entropy": total_entropy,
        "match_evidence": match_evidence,
        "log_match_evidence": log_match_evidence,
        "log_odds": log_odds,
        "positions": positions,
        "scores": scores,
        "covariances": covariances,
        "counts": counts,
    }


def candidate_log_likelihood(records, field, *, resolution_m: float,
                             beta: float = 50.0,
                             sigma_f_m: float = 0.5,
                             outlier_prior: float = 0.1,
                             outlier_radius_m: float = 4.0,
                             background_density=None):
    """Full per-tile candidate-set log density, including the common factor."""
    if not 0.0 < outlier_prior < 1.0:
        raise ValueError("outlier_prior must be in (0, 1)")
    if outlier_radius_m <= 0:
        raise ValueError("outlier_radius_m must be positive")
    positions, scores, covariances, counts = _candidate_arrays(
        records, resolution_m)
    background = _background_array(
        background_density, len(records), scores.shape[1], outlier_radius_m)
    east = np.asarray([record["east"] for record in records], float)
    north = np.asarray([record["north"] for record in records], float)
    field_east, field_north, _, _ = field.evaluate(east, north)
    field_shift = np.column_stack([field_east, field_north])
    log_density = np.zeros(len(records))
    for index in range(len(records)):
        count = int(counts[index])
        weight = scores[index, :count].copy()
        weight = np.exp(beta * (weight - np.max(weight)))
        weight /= np.sum(weight)
        covariance = covariances[index, :count] + (
            float(sigma_f_m) ** 2) * np.eye(2)
        delta = positions[index, :count] - field_shift[index]
        log_evidence = (_log_gaussian(delta, covariance)
                        - np.log(background[index, :count]))
        log_bracket = np.logaddexp(
            np.log(outlier_prior),
            np.log1p(-outlier_prior)
            + logsumexp(np.log(np.maximum(weight, 1e-300))
                        + log_evidence))
        log_density[index] = (np.sum(np.log(background[index, :count]))
                              + log_bracket)
    return {
        "log_likelihood": log_density,
        "total": float(np.sum(log_density)),
        "background_density": background,
    }


def estimate_sigma_f(records, field, responsibilities, *,
                       sigma_bounds=(0.05, 3.0),
                       prior_sigma_m: float = 0.5,
                       prior_scale_m: float = 1.0,
                       prior_weight: float = 0.1,
                       min_mass: float = 8.0):
    """Bounded MLE/MAP estimate of the field-error scale."""
    lower, upper = (float(value) for value in sigma_bounds)
    if not 0.0 < lower < upper:
        raise ValueError("sigma_bounds must be positive and increasing")
    mass = float(np.sum(responsibilities["tile_mass"]))
    if mass < min_mass:
        return {"sigma_f_m": float(prior_sigma_m), "variance_m2":
                float(prior_sigma_m) ** 2, "mass": mass,
                "frozen": True, "objective": None}
    positions = responsibilities["positions"]
    covariances = responsibilities["covariances"]
    counts = responsibilities["counts"]
    q = responsibilities["q"]
    east = np.asarray([record["east"] for record in records], float)
    north = np.asarray([record["north"] for record in records], float)
    dx, dy, _, _ = field.evaluate(east, north)
    field_shift = np.column_stack([dx, dy])
    valid = np.arange(positions.shape[1])[None, :] < counts[:, None]
    residuals = (positions - field_shift[:, None, :])[valid]
    candidate_covariances = covariances[valid]
    candidate_q = q[valid]

    def objective(variance):
        variance = float(variance)
        a = candidate_covariances[:, 0, 0] + variance
        b = candidate_covariances[:, 0, 1]
        c = candidate_covariances[:, 1, 1] + variance
        determinant = a * c - b * b
        if np.any(determinant <= 0.0):
            return np.inf
        re = residuals[:, 0]
        rn = residuals[:, 1]
        quadratic = (c * re * re - 2.0 * b * re * rn + a * rn * rn) / determinant
        total = 0.5 * float(np.sum(candidate_q * (np.log(determinant) + quadratic)))
        if prior_weight > 0.0:
            total += 0.5 * prior_weight * (
                (np.sqrt(variance) - prior_sigma_m) / prior_scale_m) ** 2
        return float(total)

    result = minimize_scalar(
        objective, bounds=(lower ** 2, upper ** 2), method="bounded",
        options={"xatol": 1e-8})
    if not result.success or not np.isfinite(result.fun):
        return {"sigma_f_m": float(prior_sigma_m), "variance_m2":
                float(prior_sigma_m) ** 2, "mass": mass,
                "frozen": True, "objective": None}
    return {"sigma_f_m": float(np.sqrt(result.x)),
            "variance_m2": float(result.x), "mass": mass,
            "frozen": False, "objective": float(result.fun)}


def _exact_field(records, responsibilities, *, sigma_f_m: float,
                 tile_m: float, max_distance_m: float, ridge: float,
                 kernel_length_m=None):
    """Exact weighted-GLS M-step from per-tile precision W_i and rhs h_i."""
    east = np.asarray([record["east"] for record in records], float)
    north = np.asarray([record["north"] for record in records], float)
    centers = np.column_stack([east, north])
    center = np.array([float(np.mean(east)), float(np.mean(north))])
    centered = centers - center
    q = responsibilities["q"]
    positions = responsibilities["positions"]
    covariances = responsibilities["covariances"]
    scores = responsibilities["scores"]
    counts = responsibilities["counts"]
    if kernel_length_m is None:
        tree = cKDTree(centers)
        spacing = tree.query(centers, k=2)[0][:, 1]
        spacing = spacing[np.isfinite(spacing) & (spacing > 0)]
        spacing_median = (float(np.median(spacing))
                           if len(spacing) else float(tile_m))
        kernel_length_m = max(2.5 * float(tile_m), 4.0 * spacing_median)
    kernel_length_m = max(float(kernel_length_m), 1e-6)
    squared = ((centers[:, None, 0] - centers[None, :, 0]) ** 2
               + (centers[:, None, 1] - centers[None, :, 1]) ** 2)
    kernel = np.exp(-0.5 * squared / (kernel_length_m ** 2))
    count = len(records)
    dimension = 4 + 2 * count
    design = np.zeros((2 * count, dimension))
    design[0::2, 0] = centered[:, 0]
    design[0::2, 1] = -centered[:, 1]
    design[0::2, 2] = 1.0
    design[1::2, 0] = centered[:, 1]
    design[1::2, 1] = centered[:, 0]
    design[1::2, 3] = 1.0
    design[0::2, 4::2] = kernel
    design[1::2, 5::2] = kernel
    precisions = np.zeros((count, 2, 2))
    rhs = np.zeros((count, 2))
    effective = np.zeros((count, 2))
    for index in range(count):
        tile_precision = np.zeros((2, 2))
        tile_rhs = np.zeros(2)
        item_count = int(counts[index])
        for column in range(item_count):
            covariance = (covariances[index, column]
                          + float(sigma_f_m) ** 2 * np.eye(2))
            precision = np.linalg.inv(covariance)
            weight = float(q[index, column])
            tile_precision += weight * precision
            tile_rhs += weight * precision @ positions[index, column]
        precisions[index] = tile_precision
        rhs[index] = tile_rhs
        if abs(np.trace(tile_precision)) > 1e-12:
            effective[index] = np.linalg.solve(
                tile_precision + 1e-12 * np.eye(2), tile_rhs)
        elif item_count:
            effective[index] = positions[index, :item_count].mean(axis=0)
    stacked_design = []
    stacked_rhs = []
    for index in range(count):
        if abs(np.trace(precisions[index])) <= 1e-12:
            continue
        factor = np.linalg.cholesky(precisions[index])
        stacked_design.append(
            factor.T @ design[2 * index:2 * index + 2])
        stacked_rhs.append(factor.T @ effective[index])
    prior = np.full(dimension, 1e-9)
    prior[4:] = float(ridge)
    stacked_design.append(np.diag(np.sqrt(prior)))
    stacked_rhs.append(np.zeros(dimension))
    theta = np.linalg.lstsq(
        np.vstack(stacked_design), np.concatenate(stacked_rhs),
        rcond=None)[0]
    gradient = np.array([[theta[0], -theta[1]], [theta[1], theta[0]]])
    translation = theta[2:4]
    coefficients = np.column_stack([theta[4::2], theta[5::2]])
    fitted = centered @ gradient.T + translation + kernel @ coefficients
    residual = effective - fitted
    tile_mass = responsibilities["tile_mass"]
    tiles = []
    for index, record in enumerate(records):
        item_count = int(counts[index])
        mass = float(tile_mass[index])
        if mass > 1e-12 and item_count:
            cc = float(np.sum(q[index, :item_count]
                             * scores[index, :item_count]) / mass)
        else:
            cc = 0.0
        tiles.append({
            "east": float(east[index]),
            "north": float(north[index]),
            "cc": cc,
            "residual_e": float(residual[index, 0]),
            "residual_n": float(residual[index, 1]),
        })
    return RegularizedPlanimetricField(
        tiles=tiles, centers=centers, weights=tile_mass,
        coefficients=coefficients, displacement_gradient=gradient,
        translation=translation, center=center,
        kernel_length_m=kernel_length_m, ridge=float(ridge),
        max_distance_m=float(max_distance_m), tile_m=float(tile_m))


def _design_points(points, center, kernel_centers, kernel_length_m):
    """Design matrix Phi mapping theta to f(p) at arbitrary points."""
    points = np.asarray(points, float)
    centered = points - np.asarray(center, float)
    squared = ((points[:, None, 0] - kernel_centers[None, :, 0]) ** 2
               + (points[:, None, 1] - kernel_centers[None, :, 1]) ** 2)
    kernel = np.exp(-0.5 * squared / (float(kernel_length_m) ** 2))
    count = len(points)
    kernel_count = len(kernel_centers)
    design = np.zeros((2 * count, 4 + 2 * kernel_count))
    design[0::2, 0] = centered[:, 0]
    design[0::2, 1] = -centered[:, 1]
    design[0::2, 2] = 1.0
    design[1::2, 0] = centered[:, 1]
    design[1::2, 1] = centered[:, 0]
    design[1::2, 3] = 1.0
    design[0::2, 4::2] = kernel
    design[1::2, 5::2] = kernel
    return design


def field_posterior_uncertainty(records, field, responsibilities, *,
                                  ridge: float = 0.1):
    """Local Laplace Hessian and covariance of the field parameters."""
    east = np.asarray([record["east"] for record in records], float)
    north = np.asarray([record["north"] for record in records], float)
    points = np.column_stack([east, north])
    design = _design_points(points, field.center, field.centers,
                            field.kernel_length_m)
    dimension = design.shape[1]
    hessian = np.zeros((dimension, dimension))
    information = responsibilities["information"]
    for index in range(len(records)):
        block = design[2 * index:2 * index + 2]
        hessian += block.T @ information[index] @ block
    prior = np.full(dimension, 1e-9)
    prior[4:] = float(ridge)
    hessian[np.diag_indices(dimension)] += prior
    eigenvalue = np.linalg.eigvalsh(hessian)
    valid = bool(np.min(eigenvalue) > 0.0)
    covariance = (np.linalg.inv(hessian) if valid
                  else np.linalg.pinv(hessian))
    return {
        "hessian": hessian,
        "covariance": covariance,
        "min_eigenvalue": float(np.min(eigenvalue)),
        "valid": valid,
    }


def predict_displacement_covariance(field, uncertainty, east, north,
                                    sigma_f_m: float = 0.5):
    """Predict Sigma_f(p) and Sigma_d(p)=Sigma_f+sigma_f^2 I."""
    east = np.asarray(east, float)
    north = np.asarray(north, float)
    points = np.column_stack([east, north])
    design = _design_points(points, field.center, field.centers,
                            field.kernel_length_m)
    covariance_theta = uncertainty["covariance"]
    count = len(points)
    field_covariance = np.empty((count, 2, 2))
    for index in range(count):
        block = design[2 * index:2 * index + 2]
        field_covariance[index] = block @ covariance_theta @ block.T
    displacement_covariance = (field_covariance
                               + float(sigma_f_m) ** 2 * np.eye(2))
    return {
        "field_covariance": field_covariance,
        "displacement_covariance": displacement_covariance,
    }


def _effective_rows(records, responsibilities):
    """Moment-matched diagnostic rows; not used by the exact M-step."""
    q = responsibilities["q"]
    q0 = responsibilities["q0"]
    tile_mass = responsibilities["tile_mass"]
    positions = responsibilities["positions"]
    covariances = responsibilities["covariances"]
    scores = responsibilities["scores"]
    counts = responsibilities["counts"]
    tile_covariances = _tile_covariances(records)
    rows = []
    for index, record in enumerate(records):
        mass = float(tile_mass[index])
        if mass <= 1e-8:
            continue
        count = int(counts[index])
        weights = q[index, :count] / mass
        displacement = weights @ positions[index, :count]
        spread = positions[index, :count] - displacement
        covariance = (tile_covariances[index]
                      + np.einsum("j,ja,jb->ab", weights, spread, spread))
        variance_e = max(float(covariance[0, 0]), 1e-8)
        variance_n = max(float(covariance[1, 1]), 1e-8)
        covariance_en = float(np.clip(
            covariance[0, 1],
            -0.99 * np.sqrt(variance_e * variance_n),
            0.99 * np.sqrt(variance_e * variance_n)))
        rows.append({
            "east": float(record["east"]),
            "north": float(record["north"]),
            "dx_m": float(displacement[0]),
            "dy_m": float(displacement[1]),
            "cc": float(weights @ scores[index, :count]),
            "quality_weight": float(record.get("quality_weight", 1.0)) * mass,
            "sigma_e": float(np.sqrt(variance_e)),
            "sigma_n": float(np.sqrt(variance_n)),
            "sigma_en": covariance_en,
        })
    if not rows:
        raise ValueError("candidate-field EM rejected every tile as outlier")
    return rows


def initial_candidate_field(records, *, resolution_m: float,
                            tile_m: float, max_distance_m: float = 30.0):
    """Fit an initial field from each tile's top candidate."""
    rows = []
    for record in records:
        candidates = record.get("candidates") or []
        if not candidates:
            raise ValueError("initial field requires a candidate on every tile")
        candidate = candidates[0]
        covariance_px = np.asarray(
            candidate.get("covariance_px", np.eye(2)), float)
        flip = np.diag([float(resolution_m), -float(resolution_m)])
        covariance = (flip @ covariance_px @ flip.T
                      + CANDIDATE_CORE_FLOOR_M ** 2 * np.eye(2))
        variance_e = max(float(covariance[0, 0]), 1e-8)
        variance_n = max(float(covariance[1, 1]), 1e-8)
        rows.append({
            "east": float(record["east"]),
            "north": float(record["north"]),
            "dx_m": float(candidate["sx"]) * resolution_m,
            "dy_m": -float(candidate["sy"]) * resolution_m,
            "cc": float(candidate["cc"]),
            "quality_weight": float(record.get("quality_weight", 1.0)),
            "sigma_e": float(np.sqrt(variance_e)),
            "sigma_n": float(np.sqrt(variance_n)),
            "sigma_en": float(np.clip(
                covariance[0, 1],
                -0.99 * np.sqrt(variance_e * variance_n),
                0.99 * np.sqrt(variance_e * variance_n))),
        })
    return fit_regularized_field(rows, tile_m=tile_m,
                                 max_distance_m=max_distance_m)


def fit_candidate_field_em(records, *, resolution_m: float, tile_m: float,
                           initial_field=None, max_distance_m: float = 30.0,
                           beta: float = 50.0,
                           outlier_prior: float = 0.1,
                           outlier_radius_m: float = 4.0,
                           background_density=None,
                           sigma_bounds=(0.05, 3.0),
                           sigma_prior_m: float = 0.5,
                           sigma_prior_scale_m: float = 1.0,
                           sigma_prior_weight: float = 0.1,
                           temperature_schedule=(1.0,),
                           ridge: float = 0.1,
                           kernel_length_m=None,
                           max_iterations: int = 6):
    """Run deterministic-annealing MAP-EM and return (field, history)."""
    if not temperature_schedule:
        raise ValueError("temperature_schedule must not be empty")
    if not np.isclose(float(temperature_schedule[-1]), 1.0):
        raise ValueError("temperature_schedule must end at 1.0")
    field = initial_field
    if field is None:
        field = initial_candidate_field(
            records, resolution_m=resolution_m, tile_m=tile_m,
            max_distance_m=max_distance_m)
    east = np.asarray([record["east"] for record in records], float)
    north = np.asarray([record["north"] for record in records], float)
    history = []
    sigma_f_m = float(sigma_prior_m)
    iterations = max(1, int(max_iterations), len(temperature_schedule))
    for iteration in range(iterations):
        temperature = float(temperature_schedule[
            min(iteration, len(temperature_schedule) - 1)])
        responsibilities = candidate_responsibilities(
            records, field, resolution_m=resolution_m, beta=beta,
            sigma_f_m=sigma_f_m, outlier_prior=outlier_prior,
            outlier_radius_m=outlier_radius_m,
            background_density=background_density, temperature=temperature)
        updated = _exact_field(
            records, responsibilities, sigma_f_m=sigma_f_m, tile_m=tile_m,
            max_distance_m=max_distance_m, ridge=ridge,
            kernel_length_m=kernel_length_m)
        previous_east, previous_north, _, _ = field.evaluate(east, north)
        current_east, current_north, _, _ = updated.evaluate(east, north)
        change = np.hypot(current_east - previous_east,
                          current_north - previous_north)
        sigma_info = estimate_sigma_f(
            records, updated, responsibilities, sigma_bounds=sigma_bounds,
            prior_sigma_m=sigma_prior_m, prior_scale_m=sigma_prior_scale_m,
            prior_weight=sigma_prior_weight)
        q = responsibilities["q"]
        q0 = responsibilities["q0"]
        info_eigenvalue = np.linalg.eigvalsh(
            responsibilities["information"])
        top_mass = np.max(q, axis=1)
        ambiguous = ((responsibilities["tile_mass"] > 0.5)
                     & (top_mass < AMBIGUITY_MASS_MIN))
        history.append({
            "iteration": iteration + 1,
            "sigma_f_m": sigma_f_m,
            "sigma_estimated_m": sigma_info["sigma_f_m"],
            "temperature": temperature,
            "outlier_tiles": int(np.sum(q0 > 0.5)),
            "outlier_mass_mean": float(np.mean(q0)),
            "ambiguous_tiles": int(np.sum(ambiguous)),
            "min_information_eigenvalue": float(np.min(info_eigenvalue)),
            "ambiguity_loss_mean": float(np.mean(np.trace(
                responsibilities["ambiguity_loss"], axis1=1, axis2=2))),
            "validity_entropy_mean": float(np.mean(
                responsibilities["validity_entropy"])),
            "alias_entropy_mean": float(np.mean(
                responsibilities["alias_entropy"])),
            "total_entropy_mean": float(np.mean(
                responsibilities["total_entropy"])),
            "field_change_mean_m": float(np.mean(change)),
            "field_change_max_m": float(np.max(change)),
        })
        field = updated
        sigma_f_m = float(sigma_info["sigma_f_m"])
    return field, history
