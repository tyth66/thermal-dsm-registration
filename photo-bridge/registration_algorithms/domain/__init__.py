
"""Typed domain contracts for the residual-mapping pipeline."""
from .bundle import PanelVerticalBundle
from .geometry import GeometryModel, PanelGeometry
from .panel import ArraySurface, PanelPlane, PanelVertical, PanelVerticalModel
from .residual import CanonicalResidualModel, PanelResidual

__all__ = [
    "ArraySurface",
    "CanonicalResidualModel",
    "GeometryModel",
    "PanelGeometry",
    "PanelPlane",
    "PanelResidual",
    "PanelVertical",
    "PanelVerticalBundle",
    "PanelVerticalModel",
]
