from __future__ import annotations

import json

import pytest

from scripts.run_qwen_vl_physics_estimation import (
    PhysicsEstimationError,
    extract_json_object,
    normalize_qwen_estimate,
)

VALID = {
    "dimensions_m_width_depth_height": [0.55, 0.45, 0.62],
    "mass_kg": 18,
    "density_kg_m3": 520,
    "static_friction": 0.55,
    "dynamic_friction": 0.4,
    "restitution": 0.05,
    "confidence": 0.72,
    "limitations": ["Material and wall thickness are inferred."],
}


def test_extract_and_normalize_qwen_prior() -> None:
    result = normalize_qwen_estimate(extract_json_object(json.dumps(VALID)))
    assert result.status == "unvalidated_estimate"
    assert result.source == "qwen_vl_prior"
    assert result.scale_basis == "category_prior"
    assert result.dimensions_m == (0.55, 0.45, 0.62)
    assert result.confidence == 0.6
    assert any("not measurements" in item for item in result.limitations)


def test_extract_fenced_json() -> None:
    assert extract_json_object(f"```json\n{json.dumps(VALID)}\n```")["mass_kg"] == 18


def test_dynamic_friction_must_not_exceed_static() -> None:
    with pytest.raises(PhysicsEstimationError, match="dynamic_friction"):
        normalize_qwen_estimate({**VALID, "dynamic_friction": 0.8})


def test_dimensions_are_bounded() -> None:
    with pytest.raises(PhysicsEstimationError, match=r"dimensions_m\[0\]"):
        normalize_qwen_estimate(
            {**VALID, "dimensions_m_width_depth_height": [100, 0.4, 0.5]}
        )
