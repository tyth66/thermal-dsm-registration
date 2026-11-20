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

