from __future__ import annotations

import importlib.util
import json
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


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


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


def test_load_pillow_surfaces_scene_fit_rejection_context(tmp_path: Path) -> None:
    mesh_path = tmp_path / "pillow.glb"
    mesh_path.write_bytes(b"pillow candidate")
    report_path = tmp_path / "pillow-scene-fit.json"
    write_json(
        report_path,
        {
            "object_id": "pillow-a",
            "mesh": {"glb_sha256": MODULE.sha256_file(mesh_path)},
            "all_acceptance_gates_passed": False,
            "acceptance_gates": {
                "source_camera_iou": False,
                "scene_support_contact": True,
            },
            "promotion_blockers": ["source_camera_iou"],
            "next_action": {
                "action": "rerun_scene_fit",
                "blocker": "source-camera silhouette coverage is too low",
            },
        },
    )

    with pytest.raises(ValueError) as exc_info:
        MODULE.load_pillow(
            {
                "object_id": "pillow-a",
                "mesh": str(mesh_path),
                "scene_fit_report": str(report_path),
            },
            base=tmp_path,
            camera={},
        )

    message = str(exc_info.value)
    assert "scene fit is not accepted for pillow-a" in message
    assert "all_acceptance_gates_passed=false" in message
    assert "failed_acceptance_gates=source_camera_iou" in message
    assert "promotion_blockers=source_camera_iou" in message
    assert "next_action=rerun_scene_fit" in message
    assert str(report_path) in message


def test_joint_review_surfaces_bed_scene_fit_rejection_context(tmp_path: Path) -> None:
    bed_mesh = tmp_path / "bed.glb"
    bed_mesh.write_bytes(b"bed candidate")
    bed_report = tmp_path / "bed-scene-fit.json"
    write_json(
        bed_report,
        {
            "sources": {"mesh": {"sha256": MODULE.sha256_file(bed_mesh)}},
            "all_acceptance_gates_passed": False,
            "acceptance_gates": {
                "scene_floor_contact": True,
                "source_camera_silhouette_iou": False,
            },
            "promotion_blockers": ["source_camera_silhouette_iou"],
        },
    )
    cameras = tmp_path / "cameras.json"
    write_json(
        cameras,
        [
            {
                "img_name": "000064",
                "width": 64,
                "height": 64,
                "fx": 40.0,
                "fy": 40.0,
                "rotation": np.eye(3).tolist(),
                "position": [0.0, 0.0, 0.0],
            }
        ],
    )
    pairwise = tmp_path / "pairwise.json"
    write_json(pairwise, {"status": "passed", "promotion_allowed": True})
    spec = tmp_path / "joint-spec.json"
    write_json(
        spec,
        {
            "frame_id": "000064",
            "cameras": str(cameras),
            "source_frame": str(tmp_path / "source.png"),
            "pairwise_review": str(pairwise),
            "bed": {
                "mesh": str(bed_mesh),
                "scene_fit_report": str(bed_report),
                "mattress_geometries": ["mattress"],
                "headboard_geometries": ["headboard"],
            },
        },
    )

    with pytest.raises(ValueError) as exc_info:
        MODULE.review_joint(spec, tmp_path / "joint-output")

    message = str(exc_info.value)
    assert "bed scene fit is not accepted" in message
    assert "all_acceptance_gates_passed=false" in message
    assert "failed_acceptance_gates=source_camera_silhouette_iou" in message
    assert "promotion_blockers=source_camera_silhouette_iou" in message
    assert str(bed_report) in message
