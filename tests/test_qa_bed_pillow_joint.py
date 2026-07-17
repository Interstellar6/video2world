from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "qa_bed_pillow_joint.py"
sys.path.insert(0, str(SCRIPT_PATH.parent))
SPEC = importlib.util.spec_from_file_location("qa_bed_pillow_joint", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_allowed_penetration_uses_stricter_limit() -> None:
    assert MODULE.allowed_penetration(3.5, 0.2, 0.1) == pytest.approx(0.2)
    assert MODULE.allowed_penetration(1.0, 0.2, 0.1) == pytest.approx(0.1)


def test_rejected_contact_recommends_lower_canonical_top() -> None:
    result = MODULE.evaluate_contact_values(
        pillow_bottom_down=2.6,
        pillow_thickness=3.0,
        mattress_top_down=1.9,
        current_mattress_top_canonical_y=0.4,
        world_down_per_negative_canonical_y=12.5,
        absolute_limit=0.2,
        thickness_fraction=0.1,
        maximum_support_gap=0.2,
    )
    assert result["status"] == "rejected"
    assert result["metrics"]["penetration_scene_units"] == pytest.approx(0.7)
    assert result["recommended_mattress_adjustment"][
        "required_downward_shift_scene_units"
    ] == pytest.approx(0.5)
    assert result["recommended_mattress_adjustment"][
        "required_canonical_y_reduction"
    ] == pytest.approx(0.04)
    assert result["recommended_mattress_adjustment"][
        "maximum_mattress_top_canonical_y"
    ] == pytest.approx(0.36)


def test_contact_at_limit_passes_without_adjustment() -> None:
    result = MODULE.evaluate_contact_values(
        pillow_bottom_down=2.1,
        pillow_thickness=2.0,
        mattress_top_down=1.9,
        current_mattress_top_canonical_y=0.4,
        world_down_per_negative_canonical_y=10.0,
        absolute_limit=0.2,
        thickness_fraction=0.1,
        maximum_support_gap=0.2,
    )
    assert result["status"] == "passed"
    assert result["recommended_mattress_adjustment"][
        "required_downward_shift_scene_units"
    ] == pytest.approx(0.0)


def test_support_gap_over_limit_rejects_contact() -> None:
    result = MODULE.evaluate_contact_values(
        pillow_bottom_down=1.6,
        pillow_thickness=2.0,
        mattress_top_down=1.9,
        current_mattress_top_canonical_y=0.4,
        world_down_per_negative_canonical_y=10.0,
        absolute_limit=0.2,
        thickness_fraction=0.1,
        maximum_support_gap=0.2,
    )
    assert result["status"] == "rejected"
    assert result["gates"]["penetration_within_limit"] is True
    assert result["gates"]["support_gap_within_limit"] is False


def test_floor_gate_uses_separate_floor_support_report() -> None:
    gate = MODULE.make_bed_floor_support_gate(
        {"all_acceptance_gates_passed": True, "support_contact": None},
        {"support_contact": {"passed": True, "minimum_gap": 0.0}},
    )
    assert gate["status"] == "passed"
    assert gate["support_contact"] == {"passed": True, "minimum_gap": 0.0}


def test_headboard_relation_uses_first_sufficient_sampling(monkeypatch) -> None:
    calls = []

    def fake_depth_bins(vertices, camera, bin_size_px):
        calls.append(bin_size_px)
        return np.zeros((1, 1), dtype=np.float64)

    def fake_mask_bins(mask, bin_size_px):
        return np.ones((1, 1), dtype=bool)

    def fake_evaluate_depth_order(**kwargs):
        bin_size = calls[-1]
        shared = 0 if bin_size < 12 else 2
        return {
            "status": "passed" if shared else "rejected",
            "metrics": {
                "shared_depth_bins": shared,
                "minimum_rear_minus_front_depth_m": 1.0 if shared else None,
            },
            "gates": {},
        }

    monkeypatch.setattr(MODULE, "depth_bins", fake_depth_bins)
    monkeypatch.setattr(MODULE, "mask_bins", fake_mask_bins)
    monkeypatch.setattr(MODULE, "evaluate_depth_order", fake_evaluate_depth_order)
    result = MODULE.evaluate_headboard_relation(
        pillow_vertices=np.zeros((3, 3)),
        pillow_mask=np.ones((2, 2), dtype=bool),
        headboard_vertices=np.zeros((3, 3)),
        headboard_mask=np.ones((2, 2), dtype=bool),
        camera={},
        base_bin_size_px=4,
        fallback_multipliers=[1, 2, 3, 4],
        minimum_shared_bins=1,
        minimum_depth_margin=0.05,
    )
    assert result["status"] == "passed"
    assert result["sampling"]["selected_bin_size_px"] == 12
    assert result["sampling"]["fallback_used"] is True
