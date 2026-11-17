"""Numerically stable height planes in a locally centered EN coordinate frame."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LocalPlane:
    """Plane ``z = z0 + a * (east-e0) + b * (north-n0)``."""

    e0: float
    n0: float
    z0: float
    a: float
    b: float

    def __post_init__(self) -> None:
        try:
            finite = all(math.isfinite(float(value))
                         for value in (self.e0, self.n0, self.z0, self.a, self.b))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("local plane parameters must be finite numbers") from exc
        if not finite:
            raise ValueError("local plane parameters must be finite numbers")

    def evaluate(self, east, north):
        """Evaluate the plane for scalar or broadcast-compatible arrays."""
        return self.z0 + self.a * (np.asarray(east) - self.e0) + self.b * (
            np.asarray(north) - self.n0)

    def to_dict(self) -> dict[str, float | str]:
        return {"type": "local_plane", "e0": float(self.e0), "n0": float(self.n0),
                "z0": float(self.z0), "a": float(self.a), "b": float(self.b)}

    @classmethod
    def from_model(cls, model: Any) -> "LocalPlane":
        if isinstance(model, cls):
            return model
        if isinstance(model, dict):
            if model.get("type") == "local_plane" or all(
                    key in model for key in ("e0", "n0", "z0", "a", "b")):
                try:
                    return cls(*(float(model[key]) for key in ("e0", "n0", "z0", "a", "b")))
                except (KeyError, TypeError) as exc:
                    raise ValueError("invalid local plane model") from exc
            try:
                return cls.from_model([model["plane_a"], model["plane_b"], model["plane_c"]])
            except (KeyError, TypeError) as exc:
                raise ValueError("unrecognized plane model") from exc
        try:
            coefficients = np.asarray(model, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("plane model must contain three legacy coefficients") from exc
        if coefficients.shape != (3,) or not np.isfinite(coefficients).all():
            raise ValueError("legacy plane coefficients must be three finite values")
        a, b, c = map(float, coefficients)
        return cls(0.0, 0.0, c, a, b)

    def shifted(self, offset: float) -> "LocalPlane":
        return LocalPlane(self.e0, self.n0, self.z0 + float(offset), self.a, self.b)

    def subtract(self, correction: "LocalPlane") -> "LocalPlane":
        """Subtract a height plane, retaining this plane's coordinate origin."""
        return LocalPlane(self.e0, self.n0,
                          self.z0 - float(correction.evaluate(self.e0, self.n0)),
                          self.a - correction.a, self.b - correction.b)


def fit_local_plane(east, north, values, rounds: int = 4) -> LocalPlane:
    """Fit a robust plane after centering EN and Z at their fixed sample means.

    Coordinates are centered once using all finite input samples. Robust clipping
    changes only the regression support; it never moves the coordinate origin.
    """
    e = np.asarray(east, dtype=float)
    n = np.asarray(north, dtype=float)
    z = np.asarray(values, dtype=float)
    if e.ndim != 1 or n.ndim != 1 or z.ndim != 1 or not (len(e) == len(n) == len(z)):
        raise ValueError("east, north and values must be equal-length vectors")
    if not len(e) or not (np.isfinite(e).all() and np.isfinite(n).all() and np.isfinite(z).all()):
        raise ValueError("local plane samples must be non-empty and finite")
    if not isinstance(rounds, int) or rounds < 1:
        raise ValueError("rounds must be a positive integer")
    e0, n0, z0 = float(np.mean(e)), float(np.mean(n)), float(np.mean(z))
    design = np.column_stack([e - e0, n - n0, np.ones(len(e))])
    centered_z = z - z0
    keep = np.ones(len(e), dtype=bool)
    coefficients = None
    for _ in range(rounds):
        coefficients, _, rank, _ = np.linalg.lstsq(design[keep], centered_z[keep], rcond=None)
        if int(rank) < 3:
            raise ValueError("local plane design rank must be 3")
        residual = centered_z - design @ coefficients
        center = float(np.median(residual))
        mad = float(np.median(np.abs(residual - center)))
        threshold = max(2.5 * 1.4826 * mad, 0.02)
        candidate = np.abs(residual - center) <= threshold
        if np.array_equal(candidate, keep):
            break
        if int(candidate.sum()) < 3:
            break
        keep = candidate
    # Refit after the last support update so returned coefficients match keep.
    coefficients, _, rank, _ = np.linalg.lstsq(design[keep], centered_z[keep], rcond=None)
    if int(rank) < 3:
        raise ValueError("local plane design rank must be 3")
    if not np.isfinite(coefficients).all():
        raise ValueError("local plane fit produced non-finite coefficients")
    return LocalPlane(e0, n0, z0 + float(coefficients[2]),
                      float(coefficients[0]), float(coefficients[1]))


def plane_from_row(row: dict, prefix: str = "visible_plane") -> LocalPlane | None:
    """Read a local plane from CSV fields, falling back to legacy a/b/c fields."""
    local_keys = tuple(f"{prefix}_{key}" for key in ("e0", "n0", "z0", "a", "b"))
    anchor_keys = local_keys[:3]
    local_present = [key in row and row[key] not in (None, "") for key in anchor_keys]
    present = [key in row and row[key] not in (None, "") for key in local_keys]
    if any(local_present):
        if not all(present):
            raise ValueError(f"incomplete local plane fields for {prefix}")
        try:
            return LocalPlane(*(float(row[key]) for key in local_keys))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid local plane fields for {prefix}") from exc
    legacy_keys = tuple(f"{prefix}_{key}" for key in ("a", "b", "c"))
    legacy_present = [key in row and row[key] not in (None, "") for key in legacy_keys]
    if any(legacy_present):
        if not all(legacy_present):
            raise ValueError(f"incomplete legacy plane fields for {prefix}")
        return LocalPlane.from_model([row[key] for key in legacy_keys])
    return None
