"""Reproducible 13-scheme internal baseline for the frozen candidate model."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from registration_algorithms.application.planimetric import (  # noqa: E402
    PlanimetricInputs, estimate_planimetric)
from registration_algorithms.residual_model import fit_regularized_field  # noqa: E402


def _rows(records):
    keys = ("east", "north", "dx_m", "dy_m", "cc", "quality_weight")
    return [{key: record[key] for key in keys} for record in records]


def _folds(xs, ys, step_m):
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    width, height = xmax - xmin, ymax - ymin
    ix = np.round((xs - xmin) / max(step_m, 1e-6)).astype(int)
    iy = np.round((ys - ymin) / max(step_m, 1e-6)).astype(int)
    folds = {}

    def add(name, mask):
        mask = np.asarray(mask, bool)
        if mask.sum() == 0 or (~mask).sum() < 8:
            return
        folds.setdefault(name, []).append(mask)

    points = np.column_stack([xs, ys])
    centered = points - points.mean(axis=0)
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    axis = vectors[0] / np.linalg.norm(vectors[0])
    if axis[0] < 0:
        axis = -axis
    order = np.argsort(centered @ axis, kind="stable")
    for group in np.array_split(order, 4):
        add("pca4", np.isin(np.arange(len(xs)), group))

    for count in (3, 4):
        cx = np.clip(((xs - xmin) / max(width, 1e-9) * count).astype(int), 0, count - 1)
        cy = np.clip(((ys - ymin) / max(height, 1e-9) * count).astype(int), 0, count - 1)
        for i in range(count):
            for j in range(count):
                add("grid%d" % count, (cx == i) & (cy == j))

    distance = np.minimum.reduce([xs - xmin, xmax - xs, ys - ymin, ymax - ys])
    add("edge_ring", distance <= 0.20 * min(width, height))
    add("edge_N", ys >= ymax - 0.20 * height)
    add("edge_S", ys <= ymin + 0.20 * height)
    add("edge_E", xs >= xmax - 0.20 * width)
    add("edge_W", xs <= xmin + 0.20 * width)
    edge = distance <= 0.20 * min(width, height)
    edge_indices = np.where(edge)[0]
    edge_indices = edge_indices[np.argsort(xs[edge_indices] * 1000 + ys[edge_indices])]
    half = np.zeros(len(xs), bool)
    half[edge_indices[::2]] = True
    add("edge_half", half)

    for xlo, xhi in ((xmin, xmin + 0.30 * width), (xmax - 0.30 * width, xmax)):
        for ylo, yhi in ((ymin, ymin + 0.30 * height), (ymax - 0.30 * height, ymax)):
            add("corner4", (xs >= xlo) & (xs <= xhi) & (ys >= ylo) & (ys <= yhi))

    parity = (ix + iy) % 2
    add("checker", parity == 0)
    add("checker", parity == 1)
    for seed in range(2):
        rng = np.random.default_rng(seed)
        add("random25", rng.random(len(xs)) < 0.25)
    gx, gy = ix // 2, iy // 2
    for a, b in sorted(set(zip(gx.tolist(), gy.tolist()))):
        add("local2x2", (gx == a) & (gy == b))
    return folds


def evaluate(name, visible_dsm, thermal_dsm):
    result = estimate_planimetric(PlanimetricInputs(
        visible_dsm=visible_dsm, thermal_dsm=thermal_dsm,
        visible_world={"panels": []}, resolution_m=0.1,
        tile_m=40.0, step_m=20.0, ecc_workers=8, warp_threads=8))
    records = list(result.accepted)
    xs = np.asarray([row["east"] for row in records], float)
    ys = np.asarray([row["north"] for row in records], float)
    vf = np.asarray([row["valid_fraction"] for row in records], float)
    dx = np.asarray([row["dx_m"] for row in records], float)
    dy = np.asarray([row["dy_m"] for row in records], float)
    folds = _folds(xs, ys, result.tiling.step_m)
    errors = {}
    for scheme, tests in folds.items():
        values = []
        for test in tests:
            train = [records[i] for i in np.where(~test)[0]]
            held = np.where(test & (vf >= 0.8))[0]
            if not len(held):
                continue
            field = fit_regularized_field(
                _rows(train), tile_m=result.tiling.tile_m,
                max_distance_m=result.field_support.max_distance_m,
                kernel_length_m=None, ridge=0.1)
            de, dn, _, _ = field.evaluate(xs[held], ys[held])
            values.extend(np.hypot(de - dx[held], dn - dy[held]).tolist())
        if values:
            errors[scheme] = float(np.mean(values))
    return {
        "accepted": len(records),
        "records": len(result.records),
        "ambiguous": sum(bool(row["ambiguous"]) for row in result.records),
        "mean_holdout_error_m": float(np.mean(list(errors.values()))) if errors else None,
        "schemes": errors,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", action="append", required=True,
                        help="NAME:VISIBLE_DSM:THERMAL_DSM")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    output = {}
    for spec in args.station:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError("station must be NAME:VISIBLE_DSM:THERMAL_DSM")
        name, visible, thermal = parts
        output[name] = evaluate(name, Path(visible), Path(thermal))
    text = json.dumps(output, indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
