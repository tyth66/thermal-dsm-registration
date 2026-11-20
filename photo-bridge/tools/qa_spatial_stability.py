"""Spatial geometric stability QA for the frozen planimetric pipeline.

Read-only diagnostic: it imports the production estimator and regularized field
fitter and never changes them. Three report-only metric families:

  1. spatial support     - tile- and panel-level nearest-tile distance and
                           local k-NN density, per-block counts/bboxes and
                           station-wide coverage balance;
  2. parameter stability - leave-one-block-out Similarity(2) rotation/scale/
                           translation deltas against the full field, plus the
                           fold kernel length. Each fold is refitted at the
                           station's production tile scale and support radius,
                           so the fold field differs from the full field only by
                           the held-out observations: held-out tiles never take
                           part in the fit, and the kernel length is estimated
                           from training spacing only;
  3. component sensitivity - per-panel U = || rigid_delta(fold field of its
                           own block) - rigid_delta(full field) || with the
                           rigid five-point mean semantics of the production
                           mapping, station P50/P95/max, per-block breakdown
                           and descriptive high-sensitivity clusters.

Blocks are cut along the PCA principal axis of the accepted tile centres:
centres are projected onto the first principal direction and split into
equal-count contiguous slices, so a rotated array is split along its own long
axis instead of degenerating under EN median quadrants (blocks=2 is the
single-cut case). Panels are assigned with the same axis and boundaries. Every
block/fold reports holdout_ratio = held_tiles / total_tiles; U_k depends on that
ratio, so only compare blocks with similar held/total ratios.

No thresholds, no admission control, no gates: every number is descriptive and
cross-station comparable. The high-sensitivity cluster reference is purely a
reporting aid (pitch/2 of the panel lattice), never a decision rule.

Usage:
  python3 tools/qa_spatial_stability.py \
      --visible-dsm <visible.dsm.tif> --thermal-dsm <thermal.dsm.tif> \
      --result-json <v3 mapping payload> [--station-name <name>] \
      [--blocks 4] [--out-dir <dir>] [--force]

The result JSON supplies the panel EN geometry (each panel centre + four corners
with east/north_visible). Output: <out-dir>/<station_name>_qa.json with schema
spatial_stability_qa/v1. When --station-name is omitted the result JSON's parent
directory name is used; an existing output is never overwritten without --force.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from registration_algorithms.application import (  # noqa: E402
    PlanimetricInputs, estimate_planimetric,
)
from registration_algorithms.residual_model import (  # noqa: E402
    fit_regularized_field,
)

SCHEMA = "spatial_stability_qa/v1"
PRODUCTION_PARAMS = dict(resolution_m=0.1, tile_m=40.0, step_m=20.0,
                         highpass_sigma_m=1.5, field_max_distance_m=30.0,
                         warp_threads=8, ecc_workers=8)
RIDGE = 0.1  # production default (PlanimetricInputs.regularized_ridge)
MINIMUM_TRAIN_TILES = 8
CLUSTER_EPS_M = 3.0
DENSITY_K = 6


def summarize(values) -> dict:
    """Percentile summary of finite values (None entries are ignored)."""
    array = np.asarray([value for value in values
                        if value is not None and np.isfinite(value)], float)
    if not len(array):
        return {"n": 0}
    return {"n": int(len(array)),
            "p50": float(np.percentile(array, 50)),
            "p95": float(np.percentile(array, 95)),
            "max": float(array.max()),
            "min": float(array.min())}


def _realized_blocks(labels) -> dict:
    """Label -> tile count, in label order."""
    text = [str(label) for label in labels]
    return {name: int(text.count(name)) for name in sorted(set(text))}


def _principal_axis(centers: np.ndarray):
    """First PCA direction of the tile centres, or None without directional spread.

    Returns (center, unit_direction, projection). The direction is canonicalised
    so its east component is non-negative, which makes axis_angle_deg stable
    under the SVD's sign ambiguity. A degenerate input (all centres collapsed to
    one location, non-finite values) returns None so the caller can fall back to
    the EN split.
    """
    center = centers.mean(axis=0)
    centered = centers - center
    if not np.isfinite(centered).all():
        return None
    try:
        _, singular, vectors = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    spread = float(np.abs(centered).max()) if len(centered) else 0.0
    if (not len(singular) or not np.isfinite(singular[0])
            or singular[0] <= 1e-9 * max(spread, 1.0)):
        return None
    direction = np.asarray(vectors[0], float)
    norm = float(np.hypot(direction[0], direction[1]))
    if not np.isfinite(norm) or norm <= 0.0:
        return None
    direction = direction / norm
    if direction[0] < 0.0 or (direction[0] == 0.0 and direction[1] < 0.0):
        direction = -direction
    return center, direction, centered @ direction


def _median_quadrant_split(centers, blocks):
    """EN median-quadrant fallback (degenerate PCA, blocks == 4)."""
    median_east = float(np.median(centers[:, 0]))
    median_north = float(np.median(centers[:, 1]))

    def assign(points):
        pts = np.asarray(points, float).reshape(-1, 2)
        names = np.empty(len(pts), dtype=object)
        for index, (point_east, point_north) in enumerate(pts):
            names[index] = (("N" if point_north >= median_north else "S")
                            + ("E" if point_east >= median_east else "W"))
        return names

    scheme = {"method": "median_quadrants", "blocks": blocks,
              "median_east": median_east, "median_north": median_north,
              "realized": _realized_blocks(assign(centers))}
    return scheme, assign


def _axis_quantile_split(centers, blocks):
    """Longer-EN-axis equal-count quantile fallback (degenerate PCA)."""
    east = centers[:, 0]
    north = centers[:, 1]
    axis = 0 if float(np.ptp(east)) >= float(np.ptp(north)) else 1
    edges = np.quantile(centers[:, axis],
                        np.linspace(0.0, 1.0, blocks + 1)[1:-1])

    def assign(points):
        pts = np.asarray(points, float).reshape(-1, 2)
        indices = np.searchsorted(edges, pts[:, axis], side="right")
        return np.asarray(["B%d" % index for index in indices], dtype=object)

    scheme = {"method": "axis_quantiles", "blocks": blocks,
              "axis": "east" if axis == 0 else "north",
              "edges": [float(edge) for edge in edges],
              "realized": _realized_blocks(assign(centers))}
    return scheme, assign


def partition_scheme(tile_centers, blocks: int = 4):
    """Return (scheme, assign) for the spatial split.

    The split runs along the PCA principal axis of the tile centres: centres are
    projected onto the first principal direction and cut into equal-count
    contiguous slices (np.array_split of the projected order). This keeps a
    rotated array (e.g. -33 deg) split along its own long axis instead of
    degenerating under EN median quadrants. blocks == 2 is the single-cut case.
    Panels are assigned with the same axis and slice boundaries, so tile and
    panel labels always agree. Degenerate centres (no directional spread, or
    fewer centres than blocks) fall back to the deterministic EN median/quantile
    split.
    """
    centers = np.asarray(tile_centers, float)
    if centers.ndim != 2 or centers.shape[1] != 2 or not len(centers):
        raise ValueError("tile_centers must be a non-empty (N, 2) array")
    blocks = int(blocks)
    if blocks < 2:
        raise ValueError("blocks must be at least 2")
    axis = _principal_axis(centers) if len(centers) >= blocks else None
    if axis is None:
        if blocks == 4:
            return _median_quadrant_split(centers, blocks)
        return _axis_quantile_split(centers, blocks)
    center, direction, projection = axis
    order = np.argsort(projection, kind="stable")
    groups = np.array_split(order, blocks)
    edges = np.asarray([
        (float(projection[left[-1]]) + float(projection[right[0]])) / 2.0
        for left, right in zip(groups[:-1], groups[1:])], float)
    # Regular tile lattices project onto repeated values, so a slice boundary
    # can fall inside a tie group. The array_split groups stay authoritative for
    # the tiles themselves; arbitrary points (panels) use the slice boundaries.
    tile_labels = np.empty(len(centers), dtype=object)
    for index, group in enumerate(groups):
        tile_labels[group] = "B%d" % index
    tile_of_point = {(float(point[0]), float(point[1])): str(label)
                     for point, label in zip(centers, tile_labels)}

    def assign(points):
        pts = np.asarray(points, float).reshape(-1, 2)
        projected = (pts - center) @ direction
        indices = np.searchsorted(edges, projected, side="right")
        labels = np.asarray(["B%d" % index for index in indices], dtype=object)
        for position, (east, north) in enumerate(pts):
            exact = tile_of_point.get((float(east), float(north)))
            if exact is not None:
                labels[position] = exact
        return labels

    scheme = {"method": "pca_axis", "blocks": blocks,
              "axis_direction": [float(direction[0]), float(direction[1])],
              "axis_angle_deg": float(np.degrees(
                  np.arctan2(direction[1], direction[0]))),
              "center": [float(center[0]), float(center[1])],
              "edges": [float(edge) for edge in edges],
              "realized": _realized_blocks(tile_labels)}
    return scheme, assign


def tiles_from_records(records) -> list[dict]:
    """Normalise planimetric registration records to QA tile rows."""
    tiles = []
    for record in records:
        tiles.append({"row": int(record["row"]),
                      "column": int(record["column"]),
                      "east": float(record["east"]),
                      "north": float(record["north"]),
                      "dx_m": float(record["dx_m"]),
                      "dy_m": float(record["dy_m"]),
                      "cc": float(record["cc"])})
    return tiles


def tile_centers(tiles) -> np.ndarray:
    return np.asarray([[tile["east"], tile["north"]] for tile in tiles], float)


def panel_geometry_from_result(result_doc) -> tuple[list[np.ndarray], list[str]]:
    """Per-panel five EN points (centre + four corners) and panel ids."""
    points = []
    ids = []
    for panel in result_doc["panels"]:
        five = [panel["center"]] + list(panel["corners"])
        if len(five) != 5:
            raise ValueError("panel %s does not carry five points"
                             % panel.get("panel_id"))
        points.append(np.asarray(
            [[float(point["east_visible"]), float(point["north_visible"])]
             for point in five], float))
        ids.append(str(panel["panel_id"]))
    return points, ids


def rigid_delta(field, points) -> np.ndarray:
    """Rigid five-point displacement of a panel under a planimetric field."""
    pts = np.asarray(points, float)
    delta_e, delta_n, _, _ = field.evaluate(pts[:, 0], pts[:, 1])
    return np.array([float(np.mean(delta_e)), float(np.mean(delta_n))])


def fit_field(rows, tile_m: float, max_distance_m: float,
              ridge: float = RIDGE):
    """The frozen production field estimator at the station's production scale.

    tile_m and max_distance_m must come from the production estimation result
    (TilingPlan.tile_m, FieldSupportPlan.max_distance_m), never from a fixed
    constant: a leave-block-out fold then differs from the full field only by
    the held-out observations, not by estimator parameters. Kernel length stays
    auto (production regularized_kernel_m is None).
    """
    return fit_regularized_field(rows, tile_m=float(tile_m),
                                 max_distance_m=float(max_distance_m),
                                 kernel_length_m=None, ridge=float(ridge))


def _fit_rows(tiles):
    return [{"east": tile["east"], "north": tile["north"],
             "dx_m": tile["dx_m"], "dy_m": tile["dy_m"],
             "cc": max(0.05, float(tile["cc"]))} for tile in tiles]


def fit_fold_fields(tiles, labels, panel_labels, field_all, tile_m: float,
                     max_distance_m: float):
    """Leave-one-block-out fits plus their parameter deltas vs the full field.

    The block set is the union of the tile and panel labels, so a block that
    only holds panels (no accepted tiles) is still reported instead of silently
    disappearing from the fold table. Each fold is refitted with the production
    tile scale and support radius, so the fold field differs from the full field
    only by the held-out observations. Returns (folds, fold_fields). A block
    whose held-out tile set is empty, or whose training set has fewer than
    MINIMUM_TRAIN_TILES tiles, is skipped with an explicit reason; the tool never
    raises for sparse stations.
    """
    labels = np.asarray(labels, dtype=object)
    panel_labels = np.asarray(panel_labels, dtype=object)
    full_similarity = field_all.to_model()["similarity"]
    folds = []
    fold_fields = {}
    total_tiles = len(tiles)
    def holdout_ratio(held_count):
        return float(held_count) / float(total_tiles) if total_tiles else 0.0
    block_names = sorted(set(labels.tolist()) | set(panel_labels.tolist()))
    for name in block_names:
        train = [tile for tile, label in zip(tiles, labels) if label != name]
        held = [tile for tile, label in zip(tiles, labels) if label == name]
        if not held:
            folds.append({"block": name, "skipped": True,
                          "skip_reason": "empty held-out block",
                          "held_tiles": 0, "train_tiles": len(train),
                          "holdout_ratio": holdout_ratio(0)})
            continue
        if len(train) < MINIMUM_TRAIN_TILES:
            folds.append({"block": name, "skipped": True,
                          "skip_reason": "train tiles %d < %d" % (
                              len(train), MINIMUM_TRAIN_TILES),
                          "held_tiles": len(held), "train_tiles": len(train),
                          "holdout_ratio": holdout_ratio(len(held))})
            continue
        field = fit_field(_fit_rows(train), tile_m, max_distance_m)
        fold_fields[name] = field
        similarity = field.to_model()["similarity"]
        translation_delta = (np.asarray(similarity["translation"], float)
                             - np.asarray(full_similarity["translation"], float))
        folds.append({
            "block": name, "skipped": False,
            "held_tiles": len(held), "train_tiles": len(train),
            "holdout_ratio": holdout_ratio(len(held)),
            "kernel_length_m": float(field.kernel_length_m),
            "centers": len(field.centers),
            "rotation_deg": float(similarity["rotation_deg"]),
            "rotation_delta_deg": float(similarity["rotation_deg"]
                                        - full_similarity["rotation_deg"]),
            "scale": float(similarity["scale"]),
            "scale_delta": float(similarity["scale"] - full_similarity["scale"]),
            "translation": [float(value) for value in similarity["translation"]],
            "translation_delta_m": [float(value) for value in translation_delta],
        })
    return folds, fold_fields


def panel_sensitivity(field_all, fold_fields, panel_points, panel_centers,
                      assign):
    """Per-panel U = rigid_delta(fold of its block) - rigid_delta(full)."""
    labels = np.asarray(assign(panel_centers), dtype=object)
    values = np.full(len(panel_points), np.nan)
    skipped_blocks = set()
    for index, (points, label) in enumerate(zip(panel_points, labels)):
        field = fold_fields.get(label)
        if field is None:
            skipped_blocks.add(str(label))
            continue
        values[index] = float(np.linalg.norm(rigid_delta(field, points)
                                             - rigid_delta(field_all, points)))
    return values, labels, sorted(skipped_blocks)


def high_sensitivity_clusters(panel_centers, values, threshold_m: float,
                              eps_m: float = CLUSTER_EPS_M) -> dict:
    """Descriptive connected-component clusters of high-sensitivity panels."""
    centers = np.asarray(panel_centers, float)
    values = np.asarray(values, float)
    mask = np.isfinite(values) & (values >= float(threshold_m))
    count = int(mask.sum())
    if count == 0:
        return {"threshold_m": float(threshold_m), "panels": 0,
                "cluster_count": 0, "max_cluster_size": 0, "clusters": []}
    members = centers[mask]
    tree = cKDTree(members)
    parent = np.arange(len(members))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left, right in tree.query_pairs(r=float(eps_m)):
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_left] = root_right
    labels = np.asarray([find(index) for index in range(len(members))])
    clusters = []
    for label in sorted(set(labels.tolist())):
        group = members[labels == label]
        clusters.append({
            "panels": int(len(group)),
            "centroid_east": float(group[:, 0].mean()),
            "centroid_north": float(group[:, 1].mean()),
            "span_east_m": float(np.ptp(group[:, 0])),
            "span_north_m": float(np.ptp(group[:, 1])),
        })
    clusters.sort(key=lambda item: item["panels"], reverse=True)
    return {"threshold_m": float(threshold_m), "panels": count,
            "cluster_count": len(clusters),
            "max_cluster_size": clusters[0]["panels"],
            "clusters": clusters}


def spatial_support(tile_centers_array, panel_centers) -> dict:
    """Nearest-tile distance and local k-NN density for tiles and panels."""
    centers = np.asarray(tile_centers_array, float)
    panels = np.asarray(panel_centers, float)
    k_tiles = min(DENSITY_K, max(len(centers) - 1, 1))
    tile_tree = cKDTree(centers)
    tile_nearest = tile_tree.query(centers, k=2)[0][:, 1] if len(centers) > 1 \
        else np.asarray([np.nan])
    tile_density = tile_tree.query(centers, k=k_tiles + 1)[0][:, -1] \
        if len(centers) > k_tiles else np.asarray([np.nan])
    if len(panels):
        panel_nearest = tile_tree.query(panels, k=1)[0]
        panel_density = tile_tree.query(panels, k=min(DENSITY_K, len(centers)))[0][:, -1]
    else:
        panel_nearest = np.asarray([])
        panel_density = np.asarray([])
    return {
        "tiles": {"count": int(len(centers)),
                  "nearest_tile_distance_m": summarize(tile_nearest),
                  "knn6_tile_distance_m": summarize(tile_density)},
        "panels": {"count": int(len(panels)),
                   "nearest_tile_distance_m": summarize(panel_nearest),
                   "knn6_tile_distance_m": summarize(panel_density)},
    }


def station_blocks(tiles, labels, panels, panel_labels) -> list[dict]:
    """Per-block tile/panel counts and bounding boxes.

    The block set is the union of the tile and panel labels: a block that only
    holds panels is reported with tiles=0, skipped=true and the explicit
    empty-held-out reason, so its panels stay visible in the per-block
    breakdown (and are then reported as sensitivity n/a by summarize()).
    """
    labels = np.asarray(labels, dtype=object)
    panel_labels = np.asarray(panel_labels, dtype=object)
    total_tiles = len(tiles)
    blocks = []
    for name in sorted(set(labels.tolist()) | set(panel_labels.tolist())):
        block_tiles = [tile for tile, label in zip(tiles, labels) if label == name]
        block_panels = np.asarray([center for center, label
                                   in zip(panels, panel_labels) if label == name],
                                  float)
        block = {"block": str(name), "tiles": len(block_tiles),
                 "panels": int(len(block_panels)),
                 "holdout_ratio": (float(len(block_tiles)) / float(total_tiles)
                                   if total_tiles else 0.0),
                 "skipped": not block_tiles}
        if block_tiles:
            east = [tile["east"] for tile in block_tiles]
            north = [tile["north"] for tile in block_tiles]
            block["tile_bbox_east"] = [float(min(east)), float(max(east))]
            block["tile_bbox_north"] = [float(min(north)), float(max(north))]
        else:
            block["tile_bbox_east"] = None
            block["tile_bbox_north"] = None
            block["skip_reason"] = "empty held-out block"
        if len(block_panels):
            block["panel_bbox_east"] = [float(block_panels[:, 0].min()),
                                        float(block_panels[:, 0].max())]
            block["panel_bbox_north"] = [float(block_panels[:, 1].min()),
                                         float(block_panels[:, 1].max())]
        blocks.append(block)
    return blocks


def coverage_balance(blocks) -> dict:
    """Tile/panel count spread across blocks, including tile-empty blocks."""
    tile_counts = np.asarray([block["tiles"] for block in blocks], float)
    panel_counts = np.asarray([block["panels"] for block in blocks], float)
    zero_tile_blocks = [block["block"] for block in blocks
                        if not block["tiles"]]
    def ratio(counts):
        positive = counts[counts > 0]
        return (float(positive.max() / positive.min()) if len(positive) else None)
    return {
        "tile_count_min": int(tile_counts.min()) if len(tile_counts) else 0,
        "tile_count_max": int(tile_counts.max()) if len(tile_counts) else 0,
        "tile_count_max_over_min": ratio(tile_counts),
        "zero_tile_blocks": zero_tile_blocks,
        "panel_count_min": int(panel_counts.min()) if len(panel_counts) else 0,
        "panel_count_max": int(panel_counts.max()) if len(panel_counts) else 0,
        "panel_count_max_over_min": ratio(panel_counts),
    }


def estimate_panel_pitch(panel_centers) -> float | None:
    """Panel lattice pitch: median nearest-neighbour distance of panel centres."""
    centers = np.asarray(panel_centers, float)
    if len(centers) < 2:
        return None
    distances = cKDTree(centers).query(centers, k=2)[0][:, 1]
    return float(np.median(distances))


def build_payload(args, estimation, result_doc) -> dict:
    tiles = tiles_from_records(estimation.accepted)
    centers = tile_centers(tiles)
    scheme, assign = partition_scheme(centers, args.blocks)
    tile_labels = assign(centers)
    panel_points, panel_ids = panel_geometry_from_result(result_doc)
    panel_centers = np.asarray([points.mean(axis=0) for points in panel_points], float)
    panel_labels = assign(panel_centers)
    field_all = estimation.field
    # Production scale for every fold fit: the fold field then differs from the
    # full field only by the held-out observations.
    tile_m = float(estimation.tiling.tile_m)
    max_distance_m = float(estimation.field_support.max_distance_m)
    folds, fold_fields = fit_fold_fields(tiles, tile_labels, panel_labels,
                                         field_all, tile_m, max_distance_m)
    sensitivity, panel_labels, skipped_blocks = panel_sensitivity(
        field_all, fold_fields, panel_points, panel_centers, assign)
    pitch = estimate_panel_pitch(panel_centers)
    half_pitch = None if pitch is None else pitch / 2.0
    fold_by_name = {fold["block"]: fold for fold in folds}
    blocks = station_blocks(tiles, tile_labels, panel_centers, panel_labels)
    support = spatial_support(centers, panel_centers)
    station_values = sensitivity[np.isfinite(sensitivity)]
    site_clusters = (high_sensitivity_clusters(panel_centers, sensitivity,
                                               half_pitch)
                     if half_pitch else None)
    by_block = []
    for block in blocks:
        name = block["block"]
        mask = panel_labels == name
        block_values = sensitivity[mask]
        block_clusters = (high_sensitivity_clusters(panel_centers[mask],
                                                    block_values, half_pitch)
                          if half_pitch else None)
        block.update({
            "sensitivity": summarize(block_values),
            "panels_ge_half_pitch": int(np.sum(
                np.isfinite(block_values) & (block_values
                                             >= (half_pitch or np.inf)))),
            "high_sensitivity_clusters": block_clusters,
            "fold_skipped": bool(fold_by_name.get(name, {}).get("skipped", True)),
        })
        by_block.append(block)
    warnings = []
    if len(tiles) < 2 * MINIMUM_TRAIN_TILES:
        warnings.append("fewer than %d accepted tiles; leave-block-out folds may "
                        "be skipped" % (2 * MINIMUM_TRAIN_TILES))
    for fold in folds:
        if fold.get("skipped"):
            warnings.append("block %s fold skipped: %s"
                            % (fold["block"], fold["skip_reason"]))
    if skipped_blocks:
        warnings.append("panels in blocks %s have no fold sensitivity (skip)"
                        % skipped_blocks)
    similarity = field_all.to_model()["similarity"]
    payload = {
        "schema": SCHEMA,
        "inputs": {
            "visible_dsm": str(args.visible_dsm),
            "thermal_dsm": str(args.thermal_dsm),
            "result_json": str(args.result_json),
            "blocks": int(args.blocks),
        },
        "planimetric": {
            "accepted_tiles": len(estimation.accepted),
            "records": len(estimation.records),
            "tiling": {"tile_m": float(estimation.tiling.tile_m),
                       "step_m": float(estimation.tiling.step_m),
                       "tile_px": int(estimation.tiling.tile_px),
                       "step_px": int(estimation.tiling.step_px)},
            "field_fit": {"tile_m": tile_m,
                          "max_distance_m": max_distance_m,
                          "kernel_length_m": None,
                          "ridge": RIDGE},
            "field": {"rotation_deg": float(similarity["rotation_deg"]),
                      "scale": float(similarity["scale"]),
                      "translation": [float(value) for value
                                      in similarity["translation"]],
                      "kernel_length_m": float(field_all.kernel_length_m),
                      "centers": len(field_all.centers)},
        },
        "partition": scheme,
        "spatial_support": {
            **support,
            "blocks": blocks,
            "coverage_balance": coverage_balance(blocks),
        },
        "global_parameter_stability": {"folds": folds},
        "component_sensitivity": {
            "definition": "U = || mean_5pt(fold field of panel block) - "
                          "mean_5pt(full field) || (rigid five-point mean)",
            "station": summarize(station_values),
            "note": "U_k depends on holdout_ratio; compare blocks with similar "
                    "held/total ratios",
            "pitch_m": pitch,
            "half_pitch_m": half_pitch,
            "panels_ge_half_pitch": int(np.sum(
                np.isfinite(sensitivity) & (sensitivity
                                            >= (half_pitch or np.inf)))),
            "site_high_sensitivity_clusters": site_clusters,
            "by_block": by_block,
            "panel_ids": panel_ids,
            "panel_blocks": [str(label) for label in panel_labels],
            "panel_u_m": [None if not np.isfinite(value) else float(value)
                          for value in sensitivity],
        },
        "warnings": warnings,
    }
    return payload


def _station_name(args) -> str:
    """Explicit --station-name, else the result JSON's parent directory name."""
    station_name = getattr(args, "station_name", None)
    if station_name:
        return str(station_name)
    name = Path(args.result_json).resolve().parent.name
    if not name:
        raise ValueError("cannot infer station name from %s; pass --station-name"
                         % args.result_json)
    return name


def run(args) -> Path:
    result_json = Path(args.result_json)
    out_dir = Path(args.out_dir)
    out_path = out_dir / ("%s_qa.json" % _station_name(args))
    if out_path.exists() and not getattr(args, "force", False):
        raise FileExistsError("%s already exists (pass --force to overwrite)"
                              % out_path)
    result_doc = json.loads(result_json.read_text(encoding="utf-8"))
    estimation = estimate_planimetric(PlanimetricInputs(
        visible_dsm=Path(args.visible_dsm),
        thermal_dsm=Path(args.thermal_dsm),
        visible_world={"panels": []},
        **PRODUCTION_PARAMS))
    payload = build_payload(args, estimation, result_doc)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=1))
    return out_path


def _summary_text(summary) -> str:
    """'P50=.. P95=.. max=.. m' or 'n/a' when the sample is empty."""
    if not summary.get("n", 0):
        return "P50=n/a P95=n/a max=n/a"
    return "P50=%.4f P95=%.4f max=%.4f m" % (
        summary.get("p50", float("nan")),
        summary.get("p95", float("nan")),
        summary.get("max", float("nan")))


def print_summary(payload, out_path) -> None:
    stem = out_path.stem
    station_name = stem[:-len("_qa")] if stem.endswith("_qa") else stem
    print("spatial_stability_qa/v1  station=%s" % station_name)
    planimetric = payload["planimetric"]
    field = planimetric["field"]
    print("tiles: accepted=%d records=%d  full field: rotation=%.6f deg "
          "scale=%.9f L=%.1f m centers=%d"
          % (planimetric["accepted_tiles"], planimetric["records"],
             field["rotation_deg"], field["scale"], field["kernel_length_m"],
             field["centers"]))
    fit = planimetric["field_fit"]
    print("fold fits: production scale tile_m=%.3f m max_distance=%.2f m "
          "ridge=%.2f (kernel auto from training spacing)"
          % (fit["tile_m"], fit["max_distance_m"], fit["ridge"]))
    balance = payload["spatial_support"]["coverage_balance"]
    print("blocks: %d  tile counts %d..%d (max/min %s)  panel counts %d..%d "
          "(max/min %s)"
          % (len(payload["spatial_support"]["blocks"]),
             balance["tile_count_min"], balance["tile_count_max"],
             "n/a" if balance["tile_count_max_over_min"] is None
             else "%.2f" % balance["tile_count_max_over_min"],
             balance["panel_count_min"], balance["panel_count_max"],
             "n/a" if balance["panel_count_max_over_min"] is None
             else "%.2f" % balance["panel_count_max_over_min"]))
    partition = payload.get("partition", {})
    print("partition: method=%s blocks=%d realized=%s"
          % (partition.get("method"), partition.get("blocks", 0),
             partition.get("realized")))
    for fold in payload["global_parameter_stability"]["folds"]:
        if fold.get("skipped"):
            print("  fold %-3s SKIPPED (%s) holdout=%.3f"
                  % (fold["block"], fold["skip_reason"],
                     fold.get("holdout_ratio", 0.0)))
        else:
            print("  fold %-3s held=%2d train=%2d holdout=%.3f L=%.1f m "
                  "rotation_delta=%+.4f deg scale_delta=%+.2e"
                  % (fold["block"], fold["held_tiles"], fold["train_tiles"],
                     fold.get("holdout_ratio", 0.0), fold["kernel_length_m"],
                     fold["rotation_delta_deg"], fold["scale_delta"]))
    sensitivity = payload["component_sensitivity"]
    station = sensitivity["station"]
    pitch = sensitivity.get("pitch_m")
    print("panel sensitivity U: %s (n=%d)  pitch=%s"
          % (_summary_text(station), station.get("n", 0),
             "n/a" if pitch is None else "%.4f m" % pitch))
    if sensitivity.get("note"):
        print("note:", sensitivity["note"])
    for block in sensitivity["by_block"]:
        clusters = block["high_sensitivity_clusters"] or {}
        print("  block %-3s panels=%4d tiles=%3d holdout=%.3f U %s "
              "panels>=pitch/2=%d/%d high clusters=%d max_cluster=%d"
              % (block["block"], block["panels"], block["tiles"],
                 block["holdout_ratio"], _summary_text(block["sensitivity"]),
                 block["panels_ge_half_pitch"], block["panels"],
                 clusters.get("cluster_count", 0),
                 clusters.get("max_cluster_size", 0)))
    for warning in payload["warnings"]:
        print("warning:", warning)
    print("written:", out_path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Spatial geometric stability QA (report-only, non-gating)")
    parser.add_argument("--visible-dsm", type=Path, required=True)
    parser.add_argument("--thermal-dsm", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True,
                        help="v3 mapping payload with panel EN geometry")
    parser.add_argument("--station-name", default=None,
                        help="output label; defaults to the result JSON's "
                             "parent directory name")
    parser.add_argument("--blocks", type=int, default=4,
                        help="4 = median quadrants; otherwise equal-count "
                             "quantile slices along the longer axis")
    parser.add_argument("--out-dir", type=Path, default=Path("spatial_stability_qa"))
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing <station_name>_qa.json")
    args = parser.parse_args(argv)
    try:
        out_path = run(args)
    except FileExistsError as error:
        parser.error(str(error))
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    print_summary(payload, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
