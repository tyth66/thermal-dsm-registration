
"""Application use case: fit the per-array vertical bias model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..array_vertical_bias import build_bias_rows, fit_array_bias, validate_bias
from ..domain import PanelVerticalModel


@dataclass(frozen=True)
class BiasInputs:
    panel_model: PanelVerticalModel
    geometry_rows: Mapping[str, Mapping[str, Any]]
    components: Mapping[str, Mapping[str, Any]]
    minimum_support: int
    plane_shrinkage: float


@dataclass(frozen=True)
class BiasResult:
    corrected: PanelVerticalModel
    rows: tuple[dict, ...]
    biases: Mapping[str, dict]
    component_rows: tuple[dict, ...]
    validation: Mapping[str, Any]


def fit_bias(inputs: BiasInputs) -> BiasResult:
    rows = build_bias_rows(inputs.panel_model.to_dict(),
                           dict(inputs.geometry_rows),
                           dict(inputs.components))
    biases, corrected, component_rows = fit_array_bias(
        inputs.panel_model.to_dict(), rows, inputs.minimum_support,
        plane_shrinkage=inputs.plane_shrinkage)
    validation = validate_bias(rows, inputs.minimum_support,
                               plane_shrinkage=inputs.plane_shrinkage)
    return BiasResult(
        corrected=PanelVerticalModel.from_dict(corrected),
        rows=tuple(rows), biases=biases, component_rows=tuple(component_rows),
        validation=validation)
