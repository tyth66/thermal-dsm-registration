"""Panel-level conservative validity contract."""
from __future__ import annotations

from registration_algorithms.forward_mapping import _component_conservative


def _rows(center, corners):
    rows = [{"point": "center", "conservative_valid": center}]
    rows.extend({"point": "corner_%d" % index, "conservative_valid": value}
                for index, value in enumerate(corners))
    return rows


def test_component_conservative_requires_a_common_valid_core():
    # The old component-gate combination could be true here even though no
    # point satisfies source & field & target.
    assert _component_conservative(_rows(0, (0, 0, 0, 1))) is False
    assert _component_conservative(_rows(0, (1, 1, 1, 1))) is False
    assert _component_conservative(_rows(1, (1, 1, 0, 0))) is True
    assert _component_conservative(_rows(1, (1, 0, 0, 0))) is False
