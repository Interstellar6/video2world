from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "complete_planar_background.py"
SPEC = importlib.util.spec_from_file_location("complete_planar_background", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def candidate(offset: float, support: float) -> dict[str, Any]:
    return {
        "offset": offset,
        "support_area": support,
        "inlier_faces": 100,
        "rms_residual": 0.01,
        "bounds_uv": (-20.0, 20.0, -20.0, 20.0),
    }


def plane(
    plane_id: int,
    normal: tuple[float, float, float],
    offset: float,
    basis_u: tuple[float, float, float],
    basis_v: tuple[float, float, float],
) -> Any:
    return MODULE.Plane(
        plane_id=plane_id,
        axis_index=plane_id - 1,
        side="positive",
        semantic_role="structural_boundary",
        normal=np.asarray(normal, dtype=np.float64),
        offset=offset,
        basis_u=np.asarray(basis_u, dtype=np.float64),
        basis_v=np.asarray(basis_v, dtype=np.float64),
        bounds_uv=(-20.0, 20.0, -20.0, 20.0),
        support_area=100.0,
        inlier_faces=100,
        rms_residual=0.01,
        tsdf_offset=offset,
        anchor_role=None,
        anchor_inliers=0,
        anchor_fraction=0.0,
    )


def test_fit_manhattan_axes_recovers_rotated_orthogonal_frame() -> None:
    angle = np.deg2rad(27.0)
    expected = np.asarray(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    rng = np.random.default_rng(7)
    normals = []
    areas = []
    for axis_index, count in enumerate((700, 500, 600)):
        noisy = expected[axis_index] + rng.normal(0.0, 0.015, size=(count, 3))
        noisy /= np.linalg.norm(noisy, axis=1, keepdims=True)
        noisy[::2] *= -1.0
        normals.append(noisy)
        areas.append(np.full(count, 1.0 + axis_index * 0.1))

    fitted, diagnostics = MODULE.fit_manhattan_axes(
        np.vstack(normals),
        np.concatenate(areas),
        normal_bucket_size=0.05,
        angular_tolerance_degrees=12.0,
    )

    alignment = np.abs(fitted @ expected.T)
    assert np.all(alignment.max(axis=1) > 0.999)
    assert diagnostics["maximum_axis_dot_error"] < 1e-10


def test_structure_anchor_selects_floor_instead_of_larger_bed_plane() -> None:
    axes = np.eye(3)
    candidates = [
        [],
        [candidate(-4.0, 30.0), candidate(1.0, 150.0), candidate(4.0, 25.0)],
        [],
    ]
    floor_points = np.column_stack(
        [
            np.linspace(-3.0, 3.0, 200),
            np.full(200, 4.02),
            np.linspace(2.0, 9.0, 200),
        ]
    )
    ceiling_points = floor_points.copy()
    ceiling_points[:, 1] = -4.01
    anchors = {"floor": floor_points, "ceiling": ceiling_points}
    MODULE.annotate_candidates_with_anchors(
        axes,
        candidates,
        anchors,
        assignment_tolerance=0.15,
    )

    selected = MODULE.select_envelope_planes(
        axes,
        candidates,
        np.asarray([[0.0, 0.0, 0.0]]),
        anchors,
        minimum_relative_support=0.05,
        camera_clearance=0.25,
        extent_margin=0.2,
        minimum_anchor_fraction=0.05,
        anchor_assignment_tolerance=0.15,
    )

    offsets = {item.semantic_role: item.offset for item in selected}
    assert offsets["floor"] == pytest.approx(4.02)
    assert offsets["ceiling"] == pytest.approx(-4.01)
    assert all(abs(item.offset - 1.0) > 0.5 for item in selected)
    assert next(item for item in selected if item.semantic_role == "floor").anchor_inliers == 200


def test_intersection_chooses_nearest_room_boundary() -> None:
    floor = plane(1, (0.0, 1.0, 0.0), 4.0, (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    front_wall = plane(
        2,
        (0.0, 0.0, 1.0),
        10.0,
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    )
    rays = np.asarray([[0.0, 0.5, 1.0], [0.0, 0.1, 1.0]])

    plane_ids, depth, points = MODULE.intersect_envelope(
        rays,
        np.zeros(3),
        [floor, front_wall],
    )

    assert plane_ids.tolist() == [1, 2]
    assert depth.tolist() == pytest.approx([8.0, 10.0])
    assert points[0].tolist() == pytest.approx([0.0, 4.0, 8.0])
    assert points[1].tolist() == pytest.approx([0.0, 1.0, 10.0])


def test_upstream_prefill_validation_is_fail_closed_and_records_coverage(
    tmp_path: Path,
) -> None:
    records = [
        {"frame_id": "000001", "coverage_fraction": 0.2},
        {"frame_id": "000002", "coverage_fraction": 0.4},
    ]
    path = tmp_path / "report.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "technical_passed",
                "purpose": "measured donor prefill",
                "promotion_blocker": "semantic texture continuity review is required",
                "next_action": {
                    "action": "run_constrained_residual_completion_then_semantic_cross_view_review",
                    "blocker": "unresolved_residual_requires_reviewed_completion",
                    "no_support_frame_ids": [],
                    "unresolved_unobserved_pixels": 12,
                    "promotion_approved": False,
                },
                "gates": {
                    "outside_removal_mask_rgb_exact": True,
                    "all_residual_masks_subset_of_removal_masks": True,
                },
                "frame_records": records,
            }
        ),
        encoding="utf-8",
    )

    lineage = MODULE.validate_upstream_prefill(path, records)

    assert lineage["coverage_fraction"]["mean"] == pytest.approx(0.3)
    assert lineage["lineage_role"].endswith("only its residual masks")
    assert (
        lineage["next_action"]["action"]
        == "run_constrained_residual_completion_then_semantic_cross_view_review"
    )
    assert lineage["promotion_blocker"] == "semantic texture continuity review is required"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["gates"]["all_residual_masks_subset_of_removal_masks"] = False
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError) as error:
        MODULE.validate_upstream_prefill(path, records)
    message = str(error.value)
    assert "exactness/subset" in message
    assert "run_constrained_residual_completion_then_semantic_cross_view_review" in message
    assert "unresolved_residual_requires_reviewed_completion" in message


def test_upstream_prefill_validation_surfaces_failed_next_action(tmp_path: Path) -> None:
    records = [
        {"frame_id": "000001", "coverage_fraction": 0.0},
        {"frame_id": "000002", "coverage_fraction": 0.1},
    ]
    path = tmp_path / "report.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "technical_failed_no_support",
                "promotion_blocker": "one or more target frames have no configured donor support",
                "next_action": {
                    "action": "add_observed_donor_or_switch_to_constrained_generation_for_residual",
                    "blocker": "no_guard_stable_measured_donor_support",
                    "failed_frame_ids": ["000064"],
                    "first_failed_frame_id": "000064",
                    "not_evaluable_pair_ids": ["0016_to_0017"],
                    "not_evaluable_triplet_center_frame_ids": ["000064"],
                    "no_support_frame_ids": ["000001"],
                    "unresolved_unobserved_pixels": 24,
                    "promotion_approved": False,
                },
                "gates": {
                    "outside_removal_mask_rgb_exact": True,
                    "all_residual_masks_subset_of_removal_masks": True,
                },
                "frame_records": records,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as error:
        MODULE.validate_upstream_prefill(path, records)
    message = str(error.value)
    assert "did not pass its technical gates" in message
    assert "status=technical_failed_no_support" in message
    assert "add_observed_donor_or_switch_to_constrained_generation_for_residual" in message
    assert "no_guard_stable_measured_donor_support" in message
    assert "failed_frame_ids=000064" in message
    assert "first_failed_frame_id=000064" in message
    assert "not_evaluable_pair_ids=0016_to_0017" in message
    assert "not_evaluable_triplet_center_frame_ids=000064" in message
    assert "no_support_frame_ids=000001" in message


def test_record_key_auto_prefers_prefill_and_residual_assets() -> None:
    records = [
        {
            "prefill_frame": "frames/0000.png",
            "source_frame": "source/0000.png",
            "residual_mask": "masks/0000.png",
            "removal_mask": "source-mask/0000.png",
        }
    ]
    frame_key = MODULE.select_record_key(
        records,
        "auto",
        ("prefill_frame", "input_frame", "source_frame"),
    )
    mask_key = MODULE.select_record_key(
        records,
        "auto",
        ("residual_mask", "input_mask", "removal_mask"),
    )
    assert frame_key == "prefill_frame"
    assert mask_key == "residual_mask"


def test_cumulative_supplement_overrides_upstream_measured_partition() -> None:
    original = np.zeros((4, 6), dtype=bool)
    original[1:3, 1:5] = True
    base = np.zeros_like(original)
    base[1:3, 3:5] = True
    supplement = np.zeros_like(original)
    supplement[1:3, 1:3] = True
    supplement[0, 1] = True

    cumulative, original_union, upstream_kept, receipt = MODULE.compose_cumulative_removal(
        base,
        original,
        [supplement],
    )

    assert np.array_equal(cumulative, base | supplement)
    assert np.array_equal(original_union, original | supplement)
    assert not upstream_kept.any()
    assert receipt["supplement_added_outside_base_pixels"] == int(supplement.sum())
    assert receipt["partition_exact"] is True
    assert receipt["identity_precedence"].startswith("supplement residual overrides")

    expanded, expanded_union, expanded_kept, expanded_receipt = (
        MODULE.add_cumulative_safety_margin(
            cumulative,
            original_union,
            upstream_kept,
            receipt,
            dilation=1,
        )
    )
    assert expanded.sum() > cumulative.sum()
    assert np.array_equal(expanded | expanded_kept, expanded_union)
    assert expanded_receipt["cumulative_safety_dilation_pixels"] == 1
    assert expanded_receipt["cumulative_safety_margin_added_pixels"] > 0


def test_mask_directory_resolver_accepts_sequence_or_frame_stem(tmp_path: Path) -> None:
    mask_path = tmp_path / "0007.png"
    mask_path.write_bytes(b"mask")
    record = {"sequence_index": 7, "frame_id": "000055"}

    resolved = MODULE.resolve_mask_from_directory(tmp_path, record)

    assert resolved == mask_path.resolve()


def test_supplement_manifest_joins_by_frame_and_binds_receipt(tmp_path: Path) -> None:
    masks = tmp_path / "masks"
    masks.mkdir()
    target_records = [
        {"sequence_index": 0, "frame_id": "000048"},
        {"sequence_index": 1, "frame_id": "000049"},
    ]
    frame_records = []
    for record in target_records:
        path = masks / f"{record['sequence_index']:04d}.png"
        path.write_bytes(f"mask-{record['frame_id']}".encode())
        frame_records.append(
            {
                **record,
                "union_mask": f"masks/{path.name}",
                "union_mask_sha256": MODULE.sha256_file(path),
                "removed_object_ids": ["front", "left", "right"],
            }
        )
    manifest_path = tmp_path / "cumulative_removal_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "kind": "video2world.cumulative_removal_manifest",
                "status": "ready_for_measured_multiview_prefill",
                "round_index": 3,
                "removed_object_ids": ["front", "left", "right"],
                "frame_records": frame_records,
            }
        ),
        encoding="utf-8",
    )
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(
        json.dumps({"manifest_sha256": MODULE.sha256_file(manifest_path)}),
        encoding="utf-8",
    )

    lineage, assets = MODULE.load_supplement_mask_manifests(
        [manifest_path],
        target_records,
    )

    assert lineage[0]["receipt"]["sha256"] == MODULE.sha256_file(receipt_path)
    assert assets["000048"][0]["path"] == (masks / "0000.png").resolve()
    assert assets["000049"][0]["removed_object_ids"] == ["front", "left", "right"]


def test_object_anchor_manifest_uses_nested_mask_and_object_id(tmp_path: Path) -> None:
    masks = tmp_path / "masks"
    masks.mkdir()
    mask_path = masks / "000048.png"
    mask_path.write_bytes(b"depth-visible-anchor-mask")
    manifest_path = tmp_path / "object_anchor_supplement_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "kind": "video2world.object_anchor_projection_mask_manifest",
                "status": "technical_passed",
                "object_id": "sam3_bed_01",
                "frame_records": [
                    {
                        "sequence_index": 0,
                        "frame_id": "000048",
                        "mask": {
                            "path": "masks/000048.png",
                            "sha256": MODULE.sha256_file(mask_path),
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "receipt.json").write_text(
        json.dumps({"manifest_sha256": MODULE.sha256_file(manifest_path)}),
        encoding="utf-8",
    )

    lineage, assets = MODULE.load_supplement_mask_manifests(
        [manifest_path],
        [{"sequence_index": 0, "frame_id": "000048"}],
    )

    assert lineage[0]["removed_object_ids"] == ["sam3_bed_01"]
    assert assets["000048"][0]["path"] == mask_path.resolve()
    assert assets["000048"][0]["removed_object_ids"] == ["sam3_bed_01"]


def test_texture_donor_exclusion_uses_full_cumulative_original_union(
    tmp_path: Path,
) -> None:
    base_dir = tmp_path / "base"
    original_dir = tmp_path / "original"
    supplement_dir = tmp_path / "supplement"
    for directory in (base_dir, original_dir, supplement_dir):
        directory.mkdir()
    base = np.zeros((7, 9), dtype=bool)
    base[3, 4] = True
    original = base.copy()
    original[0, 0] = True
    supplement = np.zeros_like(base)
    supplement[3, 5] = True
    for path, mask in (
        (base_dir / "0000.png", base),
        (original_dir / "0000.png", original),
        (supplement_dir / "000048.png", supplement),
    ):
        Image.fromarray(mask.astype(np.uint8) * 255).save(path)

    prepared = MODULE.prepare_cumulative_removal_inputs(
        [{"sequence_index": 0, "frame_id": "000048", "residual_mask": "unused"}],
        manifest_dir=tmp_path,
        target_mask_key="residual_mask",
        masks_override=base_dir,
        original_removal_masks_dir=original_dir,
        supplement_mask_dirs=[],
        supplement_manifest_assets={
            "000048": [{"path": supplement_dir / "000048.png"}]
        },
        cumulative_mask_dilation=1,
        texture_donor_exclusion_dilation=2,
        output_dir=tmp_path / "cumulative",
        donor_exclusion_dir=tmp_path / "donor-exclusion",
    )["000048"]

    cumulative = MODULE.mask_bool(prepared["cumulative_mask_path"])
    donor_exclusion = MODULE.mask_bool(prepared["donor_exclusion_mask_path"])
    expected_cumulative = MODULE.binary_dilate(base | supplement, 1)
    expected_donor_exclusion = MODULE.binary_dilate(
        original | supplement | expected_cumulative,
        2,
    )
    assert np.array_equal(cumulative, expected_cumulative)
    assert np.array_equal(donor_exclusion, expected_donor_exclusion)
    assert donor_exclusion[0, 0]
    assert not cumulative[0, 0]


def test_structural_shadow_support_rejects_nonplanar_neighbor_depth() -> None:
    depth = np.asarray([[5.0, 3.0]], dtype=np.float32)
    structural = plane(
        1,
        (0.0, 0.0, 1.0),
        5.0,
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    )
    intrinsic = {
        "fx": 1.0,
        "fy": 1.0,
        "cx": 0.0,
        "cy": 0.0,
        "w": 2.0,
        "h": 1.0,
    }

    supported, counts = MODULE.structural_plane_support_for_pixels(
        np.asarray([0, 1]),
        np.asarray([0, 0]),
        depth,
        intrinsic,
        np.eye(4),
        [structural],
        plane_tolerance=0.2,
    )

    assert supported.tolist() == [True, False]
    assert counts == {"1": 1}
