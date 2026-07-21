from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "render_source_camera_pbr_layer.py"
SPEC = importlib.util.spec_from_file_location("render_source_camera_pbr_layer", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def exr_attribute(name: str, kind: str, value: bytes) -> bytes:
    return (
        name.encode("ascii")
        + b"\x00"
        + kind.encode("ascii")
        + b"\x00"
        + struct.pack("<I", len(value))
        + value
    )


def write_float_exr(path: Path, value: np.ndarray, *, compression: int = 0) -> None:
    height, width = value.shape
    channel = (
        b"Depth.V\x00"
        + struct.pack("<i", 2)
        + b"\x00\x00\x00\x00"
        + struct.pack("<ii", 1, 1)
        + b"\x00"
    )
    window = struct.pack("<4i", 0, 0, width - 1, height - 1)
    header = b"".join(
        (
            struct.pack("<II", 20000630, 2),
            exr_attribute("channels", "chlist", channel),
            exr_attribute("compression", "compression", bytes([compression])),
            exr_attribute("dataWindow", "box2i", window),
            exr_attribute("displayWindow", "box2i", window),
            exr_attribute("lineOrder", "lineOrder", b"\x00"),
            b"\x00",
        )
    )
    chunk_size = 8 + width * 4
    first_chunk = len(header) + height * 8
    offsets = b"".join(struct.pack("<Q", first_chunk + row * chunk_size) for row in range(height))
    chunks = b"".join(
        struct.pack("<iI", row, width * 4) + np.asarray(value[row], dtype="<f4").tobytes()
        for row in range(height)
    )
    path.write_bytes(header + offsets + chunks)


def test_camera_to_blender_matrix_preserves_right_down_forward_convention() -> None:
    camera = {
        "position": np.asarray([2.0, 3.0, 4.0]),
        "rotation": np.eye(3),
    }

    matrix = MODULE.camera_to_blender_world_matrix(camera)

    expected_position = MODULE.RAW_TO_BLENDER @ np.asarray([2.0, 3.0, 4.0, 1.0])
    expected_forward = MODULE.RAW_TO_BLENDER[:3, :3] @ np.asarray([0.0, 0.0, 1.0])
    expected_down = MODULE.RAW_TO_BLENDER[:3, :3] @ np.asarray([0.0, 1.0, 0.0])
    assert np.allclose(matrix[:3, 3], expected_position[:3])
    assert np.allclose(matrix[:3, :3] @ [0.0, 0.0, -1.0], expected_forward)
    assert np.allclose(matrix[:3, :3] @ [0.0, -1.0, 0.0], expected_down)
    assert np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3))


def test_scene_fit_transform_supports_affine_and_baked_pivot() -> None:
    affine = np.eye(4)
    affine[:3, 3] = [1.0, 2.0, 3.0]

    mode, matrix = MODULE.scene_fit_raw_transform(
        {
            "runtime_transform": {
                "asset_coordinates_baked": False,
                "matrix_row_major": affine.tolist(),
            }
        }
    )
    assert mode == "full_affine_runtime_transform"
    assert np.array_equal(matrix, affine)

    mode, matrix = MODULE.scene_fit_raw_transform(
        {"baked_relative_transform": {"runtime_pivot": [4.0, 5.0, 6.0]}}
    )
    assert mode == "baked_linear_plus_runtime_pivot"
    assert np.array_equal(matrix[:3, 3], [4.0, 5.0, 6.0])

    mode, matrix = MODULE.scene_fit_raw_transform(
        {
            "kind": "video2world.completed_object_scene_fit",
            "scene_local_runtime": {"placement": {"pivot": [7.0, 8.0, 9.0]}},
        }
    )
    assert mode == "scene_local_rotation_scale_baked_plus_runtime_pivot"
    assert np.array_equal(matrix[:3, 3], [7.0, 8.0, 9.0])


def test_scene_fit_transform_fails_closed_on_ambiguous_or_non_affine_values() -> None:
    with pytest.raises(ValueError, match="no supported"):
        MODULE.scene_fit_raw_transform({})
    bad = np.eye(4)
    bad[3, 0] = 0.1
    with pytest.raises(ValueError, match="affine"):
        MODULE.scene_fit_raw_transform({"runtime_transform": {"matrix_row_major": bad.tolist()}})


def test_fitted_mesh_hash_takes_precedence_over_source_asset_hash() -> None:
    assert (
        MODULE.expected_mesh_sha256(
            {
                "mesh": {"glb_sha256": "fitted"},
                "sources": {"mesh": {"sha256": "source"}},
            }
        )
        == "fitted"
    )


def test_completed_object_scene_local_hash_takes_precedence() -> None:
    assert (
        MODULE.expected_mesh_sha256(
            {
                "exports": {"scene_local": {"glb": {"sha256": "scene-local"}}},
                "mesh": {"glb_sha256": "legacy-fitted"},
                "sources": {"mesh": {"sha256": "source"}},
            }
        )
        == "scene-local"
    )


def test_neutral_review_lighting_contract_is_distance_invariant() -> None:
    contract = MODULE.lighting_profile_contract("neutral-review")

    assert contract["profile"] == "neutral-review"
    assert contract["film_transparent"] is True
    assert contract["distance_dependence"] == "none"
    assert contract["world"]["mode"] == "nodes_background"
    assert contract["world"]["strength"] > 0
    assert contract["view_settings"]["exposure"] == 1.0
    assert len(contract["lights"]) == 2
    assert {light["type"] for light in contract["lights"]} == {"SUN"}
    for light in contract["lights"]:
        assert light["energy"] > 0
        assert light["direction_frame"] == "camera_local"
        assert np.linalg.norm(light["direction"]) == pytest.approx(1.0)


def test_legacy_point_lighting_contract_preserves_old_energy_and_locations() -> None:
    contract = MODULE.lighting_profile_contract("legacy-point")

    assert contract["distance_dependence"] == "inverse_square"
    assert contract["world"]["color_rgba"] == [0.08, 0.08, 0.08, 1.0]
    assert contract["view_settings"]["exposure"] == 0.0
    assert [light["type"] for light in contract["lights"]] == ["POINT", "POINT"]
    assert [light["energy"] for light in contract["lights"]] == [1400.0, 1000.0]
    assert contract["lights"][1]["location"] == [1.0, -4.0, 8.0]


def test_lighting_contract_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError, match="unsupported lighting profile"):
        MODULE.lighting_profile_contract("unknown")


def test_rejected_scene_fit_requires_explicit_review_override(tmp_path: Path) -> None:
    mesh = tmp_path / "candidate.glb"
    mesh.write_bytes(b"review candidate")
    report = {
        "kind": "video2world.completed_object_scene_fit",
        "object_id": "plant",
        "status": "rejected",
        "all_acceptance_gates_passed": False,
        "exports": {"scene_local": {"glb": {"sha256": MODULE.sha256_file(mesh)}}},
        "scene_local_runtime": {"placement": {"pivot": [1.0, 2.0, 3.0]}},
    }

    with pytest.raises(ValueError, match="not accepted"):
        MODULE.validate_scene_fit(report, mesh, "plant")

    disposition = MODULE.validate_scene_fit(
        report,
        mesh,
        "plant",
        allow_rejected_for_review=True,
    )
    assert disposition == {
        "report_status": "rejected",
        "all_acceptance_gates_passed": False,
        "review_override_used": True,
        "promotion_eligible": False,
    }


def test_cli_defaults_to_neutral_review_lighting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render_source_camera_pbr_layer.py",
            "--mesh",
            "object.glb",
            "--scene-fit-report",
            "fit.json",
            "--cameras",
            "cameras.json",
            "--layer-id",
            "plant",
            "--frame-id",
            "000001",
            "--output",
            "output",
        ],
    )

    args = MODULE.parse_args()

    assert args.lighting_profile == "neutral-review"
    assert args.allow_rejected_scene_fit_for_review is False


def test_uncompressed_float_exr_reader_preserves_scanline_orientation(tmp_path: Path) -> None:
    expected = np.asarray(
        [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]],
        dtype=np.float32,
    )
    path = tmp_path / "depth.exr"
    write_float_exr(path, expected)

    actual = MODULE.read_uncompressed_float_exr(path)

    assert actual.dtype == np.float32
    assert np.array_equal(actual, expected)


def test_exr_reader_rejects_compression_instead_of_guessing(tmp_path: Path) -> None:
    path = tmp_path / "compressed.exr"
    write_float_exr(path, np.ones((2, 3), dtype=np.float32), compression=3)

    with pytest.raises(ValueError, match="no compression"):
        MODULE.read_uncompressed_float_exr(path)


def test_materialize_depth_output_requires_the_exact_target(tmp_path: Path) -> None:
    target = tmp_path / "evidence" / "layer_frame.exr"
    wrong = tmp_path / "calibration" / target.name
    wrong.parent.mkdir()
    wrong.write_bytes(b"depth")

    with pytest.raises(ValueError, match="exact target"):
        MODULE.materialize_depth_output(object(), target)

    target.parent.mkdir()
    target.write_bytes(b"depth")
    assert MODULE.materialize_depth_output(object(), target) == target


def test_camera_loader_rejects_non_centered_principal_point(tmp_path: Path) -> None:
    path = tmp_path / "cameras.json"
    write_json(
        path,
        [
            {
                "img_name": "000067",
                "position": [0.0, 0.0, 0.0],
                "rotation": np.eye(3).tolist(),
                "width": 1280,
                "height": 720,
                "fx": 640.0,
                "fy": 680.0,
                "cx": 600.0,
                "cy": 360.0,
            }
        ],
    )

    with pytest.raises(ValueError, match="principal"):
        MODULE.load_camera(path, "000067")


def test_mask_iou_and_bbox_report_camera_alignment() -> None:
    left = np.zeros((5, 6), dtype=bool)
    right = np.zeros_like(left)
    left[1:4, 2:5] = True
    right[1:4, 2:5] = True

    assert MODULE.mask_bbox(left) == [2, 1, 5, 4]
    assert MODULE.mask_iou(left, right) == 1.0


def test_far_clip_alpha_edge_is_planned_for_sanitization() -> None:
    alpha = np.ones((10, 100), dtype=np.float32)
    depth = np.full_like(alpha, 12.0)
    depth[0, 0] = 996.0

    plan = MODULE.plan_alpha_depth_sanitization(
        alpha=alpha,
        raw_depth=depth,
        geometry_depth_min=11.0,
        geometry_depth_max=13.0,
        clip_start=0.01,
        clip_end=1000.0,
        alpha_threshold=0.5,
    )

    assert plan["sanitization_mask"].sum() == 1
    assert plan["final_alpha_mask"].sum() == 999
    assert plan["sanitized_fraction"] == pytest.approx(0.001)


def test_far_clip_alpha_sample_one_pixel_inside_antialiased_edge_is_sanitized() -> None:
    alpha = np.ones((100, 100), dtype=np.float32)
    alpha[:, 50:] = 0.0
    depth = np.full_like(alpha, 12.0)
    depth[50, 48] = 996.0

    plan = MODULE.plan_alpha_depth_sanitization(
        alpha=alpha,
        raw_depth=depth,
        geometry_depth_min=11.0,
        geometry_depth_max=13.0,
        clip_start=0.01,
        clip_end=1000.0,
        alpha_threshold=0.5,
    )

    assert plan["sanitization_mask"].sum() == 1
    assert plan["spatial_sanitization_mask"].sum() == 1
    assert plan["low_coverage_sanitization_mask"].sum() == 0
    assert plan["maximum_boundary_distance_px"] == 1


def test_far_clip_alpha_sample_two_pixels_inside_edge_fails_closed() -> None:
    alpha = np.ones((100, 100), dtype=np.float32)
    alpha[:, 50:] = 0.0
    depth = np.full_like(alpha, 12.0)
    depth[50, 47] = 996.0

    with pytest.raises(ValueError, match="boundary collar"):
        MODULE.plan_alpha_depth_sanitization(
            alpha=alpha,
            raw_depth=depth,
            geometry_depth_min=11.0,
            geometry_depth_max=13.0,
            clip_start=0.01,
            clip_end=1000.0,
            alpha_threshold=0.5,
        )


def test_low_coverage_far_clip_sample_inside_binary_silhouette_is_sanitized() -> None:
    alpha = np.ones((100, 100), dtype=np.float32)
    alpha[50, 50] = 0.55
    depth = np.full_like(alpha, 12.0)
    depth[50, 50] = 996.0

    plan = MODULE.plan_alpha_depth_sanitization(
        alpha=alpha,
        raw_depth=depth,
        geometry_depth_min=11.0,
        geometry_depth_max=13.0,
        clip_start=0.01,
        clip_end=1000.0,
        alpha_threshold=0.5,
    )

    assert plan["sanitization_mask"].sum() == 1
    assert plan["spatial_sanitization_mask"].sum() == 0
    assert plan["low_coverage_sanitization_mask"].sum() == 1
    assert plan["maximum_sanitizable_alpha"] == 0.6


def test_high_coverage_far_clip_sample_inside_binary_silhouette_fails_closed() -> None:
    alpha = np.ones((100, 100), dtype=np.float32)
    alpha[50, 50] = 0.61
    depth = np.full_like(alpha, 12.0)
    depth[50, 50] = 996.0

    with pytest.raises(ValueError, match="low-coverage"):
        MODULE.plan_alpha_depth_sanitization(
            alpha=alpha,
            raw_depth=depth,
            geometry_depth_min=11.0,
            geometry_depth_max=13.0,
            clip_start=0.01,
            clip_end=1000.0,
            alpha_threshold=0.5,
        )


def test_geometry_derived_binary_matte_records_bounded_clear_and_fill() -> None:
    alpha = np.zeros((100, 100), dtype=np.float32)
    alpha[20:80, 20:80] = 1.0
    depth = np.full_like(alpha, 996.0)
    depth[20:80, 20:80] = 12.0
    depth[50, 50] = 996.0
    depth[19, 50] = 12.0

    plan = MODULE.plan_geometry_derived_binary_matte(
        alpha=alpha,
        raw_depth=depth,
        geometry_depth_min=11.0,
        geometry_depth_max=13.0,
        clip_start=0.01,
        clip_end=1000.0,
        alpha_threshold=0.5,
    )

    assert plan["cleared_mask"].sum() == 1
    assert plan["filled_mask"].sum() == 1
    assert plan["changed_mask"].sum() == 2
    assert plan["final_alpha_mask"].sum() == plan["plausible_raw_mask"].sum()
    assert plan["initial_alpha_vs_depth_iou"] >= 0.98
    assert plan["changed_union_fraction"] <= 0.02


def test_geometry_derived_binary_matte_rejects_unbounded_support_change() -> None:
    alpha = np.zeros((100, 100), dtype=np.float32)
    alpha[10:40, 10:40] = 1.0
    depth = np.full_like(alpha, 996.0)
    depth[60:90, 60:90] = 12.0

    with pytest.raises(ValueError, match="IoU"):
        MODULE.plan_geometry_derived_binary_matte(
            alpha=alpha,
            raw_depth=depth,
            geometry_depth_min=11.0,
            geometry_depth_max=13.0,
            clip_start=0.01,
            clip_end=1000.0,
            alpha_threshold=0.5,
        )


def test_geometry_derived_binary_matte_rejects_exactly_two_percent_change() -> None:
    alpha = np.ones((10, 100), dtype=np.float32)
    depth = np.full_like(alpha, 12.0)
    depth[:, :2] = 996.0

    with pytest.raises(ValueError, match="unbounded"):
        MODULE.plan_geometry_derived_binary_matte(
            alpha=alpha,
            raw_depth=depth,
            geometry_depth_min=11.0,
            geometry_depth_max=13.0,
            clip_start=0.01,
            clip_end=1000.0,
            alpha_threshold=0.5,
        )


def test_far_clip_sanitization_fails_above_fraction_limit() -> None:
    alpha = np.ones((10, 100), dtype=np.float32)
    depth = np.full_like(alpha, 12.0)
    depth[0, :2] = 996.0

    with pytest.raises(ValueError, match="too many"):
        MODULE.plan_alpha_depth_sanitization(
            alpha=alpha,
            raw_depth=depth,
            geometry_depth_min=11.0,
            geometry_depth_max=13.0,
            clip_start=0.01,
            clip_end=1000.0,
            alpha_threshold=0.5,
        )


def test_small_non_edge_depth_error_fails_closed() -> None:
    alpha = np.ones((100, 100), dtype=np.float32)
    depth = np.full_like(alpha, 12.0)
    depth[50, 50] = 996.0

    with pytest.raises(ValueError, match="boundary collar"):
        MODULE.plan_alpha_depth_sanitization(
            alpha=alpha,
            raw_depth=depth,
            geometry_depth_min=11.0,
            geometry_depth_max=13.0,
            clip_start=0.01,
            clip_end=1000.0,
            alpha_threshold=0.5,
        )


def test_large_alpha_depth_misalignment_fails_closed() -> None:
    alpha = np.ones((20, 20), dtype=np.float32)
    depth = np.full_like(alpha, 996.0)

    with pytest.raises(ValueError, match="too many"):
        MODULE.plan_alpha_depth_sanitization(
            alpha=alpha,
            raw_depth=depth,
            geometry_depth_min=11.0,
            geometry_depth_max=13.0,
            clip_start=0.01,
            clip_end=1000.0,
            alpha_threshold=0.5,
        )


def test_silhouette_alignment_metrics_are_exact_for_equal_masks() -> None:
    mask = np.zeros((8, 10), dtype=bool)
    mask[2:7, 3:9] = True

    metrics = MODULE.silhouette_alignment_metrics(mask, mask.copy())

    assert metrics["mask_iou"] == 1.0
    assert metrics["rendered_precision"] == 1.0
    assert metrics["analytic_recall"] == 1.0
    assert metrics["bbox_iou"] == 1.0
    assert metrics["center_error_px"] == 0.0
