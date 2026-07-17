from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.materialize_clean_scene_reconstruction_input import (
    materialize_package,
    sha256_file,
    validate_acceptance,
)


@dataclass(frozen=True)
class AcceptedFixture:
    texture_report: Path
    geometry_report: Path
    camera_info: Path
    completed_frames: dict[str, Path]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def accepted_planar_fixture(tmp_path: Path) -> AcceptedFixture:
    source_root = tmp_path / "accepted-planar"
    frames_dir = source_root / "frames"
    source_dir = source_root / "source"
    masks_dir = source_root / "synthetic_masks"
    depth_dir = source_root / "geometry_depth"
    frame_ids = ("000048", "000067")
    texture_records = []
    geometry_records = []
    completed_frames: dict[str, Path] = {}
    for sequence_index, frame_id in enumerate(frame_ids):
        source = np.full((3, 4, 3), 40 + sequence_index * 20, dtype=np.uint8)
        mask = np.zeros((3, 4), dtype=np.uint8)
        mask[1, 1 + sequence_index] = 255
        completed = source.copy()
        completed[mask > 0] = [190, 180, 170]
        source_path = source_dir / f"{sequence_index:04d}.png"
        completed_path = frames_dir / f"{sequence_index:04d}.png"
        mask_path = masks_dir / f"{sequence_index:04d}.png"
        depth_path = depth_dir / f"{sequence_index:04d}.npz"
        for path in (source_path, completed_path, mask_path, depth_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(source).save(source_path)
        Image.fromarray(completed).save(completed_path)
        Image.fromarray(mask).save(mask_path)
        np.savez(depth_path, depth=np.full((3, 4), 2.5 + sequence_index, dtype=np.float32))
        completed_frames[frame_id] = completed_path
        texture_records.append(
            {
                "sequence_index": sequence_index,
                "frame_id": frame_id,
                "source_frame": str(source_path.relative_to(source_root)),
                "source_frame_sha256": sha256_file(source_path),
                "completed_frame": str(completed_path.relative_to(source_root)),
                "completed_frame_sha256": sha256_file(completed_path),
                "synthetic_mask": str(mask_path.relative_to(source_root)),
                "synthetic_mask_sha256": sha256_file(mask_path),
                "synthetic_pixels": 1,
                "synthetic_pixels_assigned_from_shared_atlas": 1,
                "all_synthetic_pixels_assigned": True,
                "outside_synthetic_mask_rgb_exact": True,
            }
        )
        geometry_records.append(
            {
                "sequence_index": sequence_index,
                "frame_id": frame_id,
                "geometry_depth": str(depth_path.relative_to(source_root)),
                "geometry_depth_sha256": sha256_file(depth_path),
                "outside_removal_mask_rgb_exact": True,
                "texture_partition_exact": True,
            }
        )

    intrinsic = {
        "camera_id": "1",
        "model": "PINHOLE",
        "w": 4,
        "h": 3,
        "fx": 5.0,
        "fy": 5.5,
        "cx": 2.0,
        "cy": 1.5,
        "params": [5.0, 5.5, 2.0, 1.5],
    }
    camera_info = {
        "schema_version": 1,
        "source": "test_colmap_text",
        "extrinsic_type": "world_to_camera",
        "intrinsic": intrinsic,
        "intrinsics": {"1": intrinsic},
        "extrinsic": {
            "000048": np.eye(4).tolist(),
            "000067": [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "000099": np.eye(4).tolist(),
        },
        "frame_camera_ids": {frame_id: "1" for frame_id in (*frame_ids, "000099")},
        "images": {
            frame_id: {
                "image_id": str(index + 1),
                "camera_id": "1",
                "name": f"{frame_id}.png",
            }
            for index, frame_id in enumerate((*frame_ids, "000099"))
        },
    }
    camera_path = source_root / "camera_info.json"
    write_json(camera_path, camera_info)
    geometry_report = {
        "schema_version": 1,
        "status": "geometry_completed_texture_pending",
        "camera_extrinsic_type": "world_to_camera",
        "camera_info": str(camera_path),
        "camera_info_sha256": sha256_file(camera_path),
        "gates": {
            "minimum_geometry_coverage": {"passed": True, "observed": 1.0},
            "outside_removal_mask_rgb_exact": True,
            "texture_masks_partition_removal_mask": True,
            "new_depth_normal_estimation_before_pgsr": "required_after_final_rgb_acceptance",
        },
        "frame_records": geometry_records,
    }
    geometry_path = source_root / "planar_background_report.json"
    write_json(geometry_path, geometry_report)
    texture_report = {
        "schema_version": 1,
        "status": "accepted_for_round04_clean_plate",
        "promotion_approved": True,
        "eligible_as_round04_clean_plate": True,
        "promotion_blocker": None,
        "planar_geometry_report": geometry_path.name,
        "planar_geometry_report_sha256": sha256_file(geometry_path),
        "gates": {
            "measured_atlas_texels_rgb_exact": True,
            "outside_synthetic_mask_rgb_exact": True,
            "protected_neighbors_outside_synthetic_mask_rgb_exact": True,
            "all_synthetic_pixels_assigned_from_shared_atlas": True,
            "same_atlas_texel_has_identical_rgb_across_views": True,
            "rejected_texture_inputs_not_promotable": True,
            "wall_boundary_color_continuity": {"passed": True},
            "wall_boundary_low_frequency_gradient_continuity": {"passed": True},
            "visual_quality": "passed",
            "new_depth_normal_estimation_before_pgsr": "required_after_visual_acceptance",
        },
        "frame_records": texture_records,
    }
    texture_path = source_root / "planar_texture_report.json"
    write_json(texture_path, texture_report)
    return AcceptedFixture(
        texture_report=texture_path,
        geometry_report=geometry_path,
        camera_info=camera_path,
        completed_frames=completed_frames,
    )


def configure_limited_report(
    fixture: AcceptedFixture,
    *,
    failed_boundary_gate: str = "wall_boundary_color_continuity",
) -> dict:
    report = read_json(fixture.texture_report)
    failed_gate = {
        "threshold_p95_abs_rgb_delta": 8.0,
        "observed_maximum": 12.5,
        "passed": False,
    }
    overridden_gates = {
        "visual_quality": "pending_human_or_vlm_review",
        failed_boundary_gate: failed_gate,
    }
    report["gates"][failed_boundary_gate] = failed_gate
    report["gates"]["visual_quality"] = "accepted_with_limitations"
    report["promotion_approved"] = False
    report["eligible_as_round04_clean_plate"] = False
    report["acceptance_scope"] = "current_demo_only"
    report["accepted_with_limitations"] = True
    report["demo_use_approved"] = True
    report["eligible_as_current_demo_round04_clean_plate"] = True
    report["overridden_gates"] = overridden_gates
    candidate_path = fixture.texture_report.parent / "limited_candidate_report.json"
    write_json(
        candidate_path,
        {
            "schema_version": 1,
            "status": "texture_candidate_review_pending",
            "promotion_approved": False,
            "eligible_as_round04_clean_plate": False,
        },
    )
    authority = {
        "kind": "user",
        "reviewer_id": "demo-owner@example.test",
        "scope": "current_demo_only",
        "reviewed_at": "2026-07-17T15:00:00+08:00",
    }
    review_limitations = ["Accepted only for the current demo."]
    override_gate_keys = ["visual_quality", failed_boundary_gate]
    review_path = fixture.texture_report.parent / "limited_visual_review.json"
    write_json(
        review_path,
        {
            "schema_version": 1,
            "kind": "video2world.planar_texture_visual_review",
            "status": "completed",
            "decision": "accepted_with_limitations",
            "acceptance_scope": "current_demo_only",
            "candidate_report_sha256": sha256_file(candidate_path),
            "review_authority": authority,
            "limitations": review_limitations,
            "override_gate_keys": override_gate_keys,
        },
    )
    report["acceptance"] = {
        "acceptance_scope": "current_demo_only",
        "accepted_with_limitations": True,
        "promotion_approved": False,
        "eligible_as_round04_clean_plate": False,
        "demo_use_approved": True,
        "eligible_as_current_demo_round04_clean_plate": True,
        "override_gate_keys": override_gate_keys,
        "overridden_gates": overridden_gates,
        "candidate_report": {
            "path": str(candidate_path),
            "sha256": sha256_file(candidate_path),
            "status": "texture_candidate_review_pending",
            "promotion_approved": False,
            "eligible_as_round04_clean_plate": False,
        },
        "visual_review": {
            "path": str(review_path),
            "sha256": sha256_file(review_path),
            "status": "completed",
            "decision": "accepted_with_limitations",
        },
        "review_authority": authority,
        "review_limitations": review_limitations,
    }
    write_json(fixture.texture_report, report)
    return failed_gate


def test_materializes_true_frame_ids_subset_cameras_and_holi_transforms(
    accepted_planar_fixture: AcceptedFixture,
    tmp_path: Path,
) -> None:
    output = tmp_path / "package-copy"
    manifest, receipt = materialize_package(
        texture_report_path=accepted_planar_fixture.texture_report,
        camera_info_path=accepted_planar_fixture.camera_info,
        output=output,
        scene_id="bedroom_4_clean",
        storage_mode="copy",
    )
    assert sorted(path.name for path in (output / "dslr/resized_undistorted_images").iterdir()) == [
        "000048.png",
        "000067.png",
    ]
    assert sorted(path.name for path in (output / "provenance/synthetic_masks").iterdir()) == [
        "000048.png",
        "000067.png",
    ]
    assert sorted(
        path.name for path in (output / "provenance/geometry_depth_visibility_evidence").iterdir()
    ) == ["000048.npz", "000067.npz"]
    camera = read_json(output / "camera_info.json")
    assert list(camera["extrinsic"]) == ["000048", "000067"]
    assert list(camera["images"]) == ["000048", "000067"]
    transforms = read_json(output / "dslr/nerfstudio/transforms_undistorted.json")
    assert [frame["file_path"] for frame in transforms["frames"]] == [
        "000048.png",
        "000067.png",
    ]
    assert transforms["frames"][0]["transform_matrix"] == [
        [1.0, -0.0, -0.0, 0.0],
        [0.0, -1.0, -0.0, 0.0],
        [0.0, -0.0, -1.0, 0.0],
        [0.0, -0.0, -0.0, 1.0],
    ]
    assert transforms["coordinate_convention"]["conversion"] == (
        "inverse(world_to_camera), then negate camera Y/Z columns"
    )
    assert manifest["status"] == "ready_for_fresh_depth_and_normal_estimation"
    assert manifest["source_planar_texture_report"]["promotion_approved"] is True
    assert manifest["source_planar_texture_report"]["eligible_as_round04_clean_plate"] is True
    assert "accepted_with_limitations" not in manifest["source_planar_texture_report"]
    assert manifest["stage_status"] == {
        "accepted_clean_rgb": "ready",
        "camera_subset": "ready",
        "holi_pgsr_camera_transforms": "ready",
        "fresh_depth_estimation": "pending",
        "fresh_normal_estimation": "pending",
        "pgsr_optimization": "pending",
        "tsdf_fusion": "pending",
    }
    assert all(
        frame["structural_geometry_depth"]["allowed_as_new_da3_depth"] is False
        for frame in manifest["frames"]
    )
    assert receipt["frame_count"] == 2
    assert receipt["acceptance_gate"]["promotion_approved"] is True
    assert receipt["acceptance_gate"]["eligible_as_round04_clean_plate"] is True
    assert "accepted_with_limitations" not in receipt["acceptance_gate"]
    assert receipt["materialized_source_file_count"] == 6
    assert all(not Path(item["destination"]).is_absolute() for item in receipt["materializations"])
    assert sha256_file(output / "manifest.json") == receipt["manifest"]["sha256"]


def test_materializes_limited_scope_and_preserves_failed_boundary_evidence(
    accepted_planar_fixture: AcceptedFixture,
    tmp_path: Path,
) -> None:
    failed_gate = configure_limited_report(accepted_planar_fixture)
    output = tmp_path / "package-limited"

    manifest, receipt = materialize_package(
        texture_report_path=accepted_planar_fixture.texture_report,
        camera_info_path=accepted_planar_fixture.camera_info,
        output=output,
        scene_id="bedroom_4_clean_demo_only",
        storage_mode="copy",
    )

    manifest_gate = manifest["source_planar_texture_report"]
    receipt_gate = receipt["acceptance_gate"]
    for gate in (manifest_gate, receipt_gate):
        assert gate["acceptance_scope"] == "current_demo_only"
        assert gate["accepted_with_limitations"] is True
        assert gate["promotion_approved"] is False
        assert gate["eligible_as_round04_clean_plate"] is False
        assert gate["demo_use_approved"] is True
        assert gate["eligible_as_current_demo_round04_clean_plate"] is True
        assert gate["decision"] == "accepted_with_limitations"
        assert gate["overridden_gates"]["wall_boundary_color_continuity"] == failed_gate
        assert gate["overridden_gates"]["wall_boundary_color_continuity"]["passed"] is False
    assert read_json(output / "manifest.json")["source_planar_texture_report"] == manifest_gate
    assert read_json(output / "receipt.json")["acceptance_gate"] == receipt_gate


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("unknown_override", "forbidden overridden gates"),
        ("missing_failed_boundary", "did not preserve failed boundary gate evidence"),
        ("tampered_failure_evidence", "did not preserve failed boundary gate evidence"),
    ],
)
def test_rejects_invalid_limited_materializer_contract(
    accepted_planar_fixture: AcceptedFixture,
    tmp_path: Path,
    mutation: str,
    expected: str,
) -> None:
    configure_limited_report(accepted_planar_fixture)
    report = read_json(accepted_planar_fixture.texture_report)
    if mutation == "unknown_override":
        report["overridden_gates"]["measured_atlas_texels_rgb_exact"] = True
        report["acceptance"]["overridden_gates"]["measured_atlas_texels_rgb_exact"] = True
        report["acceptance"]["override_gate_keys"].append("measured_atlas_texels_rgb_exact")
    elif mutation == "missing_failed_boundary":
        report["overridden_gates"].pop("wall_boundary_color_continuity")
        report["acceptance"]["overridden_gates"].pop("wall_boundary_color_continuity")
        report["acceptance"]["override_gate_keys"].remove("wall_boundary_color_continuity")
    else:
        report["overridden_gates"]["wall_boundary_color_continuity"]["observed_maximum"] = 7.0
        report["acceptance"]["overridden_gates"] = report["overridden_gates"]
    write_json(accepted_planar_fixture.texture_report, report)
    output = tmp_path / f"invalid-limited-{mutation}"

    with pytest.raises(RuntimeError, match=expected):
        materialize_package(
            texture_report_path=accepted_planar_fixture.texture_report,
            camera_info_path=accepted_planar_fixture.camera_info,
            output=output,
            scene_id="bedroom_4_clean_demo_only",
            storage_mode="copy",
        )

    assert not output.exists()


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("promotion_approved", True, "cannot approve general promotion"),
        (
            "eligible_as_current_demo_round04_clean_plate",
            False,
            "not eligible for the current demo",
        ),
    ],
)
def test_rejects_limited_report_with_wrong_permission_bits(
    accepted_planar_fixture: AcceptedFixture,
    tmp_path: Path,
    field: str,
    value: bool,
    expected: str,
) -> None:
    configure_limited_report(accepted_planar_fixture)
    report = read_json(accepted_planar_fixture.texture_report)
    report[field] = value
    write_json(accepted_planar_fixture.texture_report, report)
    output = tmp_path / f"invalid-limited-permission-{field}"

    with pytest.raises(RuntimeError, match=expected):
        materialize_package(
            texture_report_path=accepted_planar_fixture.texture_report,
            camera_info_path=accepted_planar_fixture.camera_info,
            output=output,
            scene_id="bedroom_4_clean_demo_only",
            storage_mode="copy",
        )

    assert not output.exists()


def test_rejects_limited_report_after_user_review_is_modified(
    accepted_planar_fixture: AcceptedFixture,
    tmp_path: Path,
) -> None:
    configure_limited_report(accepted_planar_fixture)
    report = read_json(accepted_planar_fixture.texture_report)
    review_path = Path(report["acceptance"]["visual_review"]["path"])
    review = read_json(review_path)
    review["limitations"].append("Modified after signing.")
    write_json(review_path, review)
    output = tmp_path / "invalid-modified-user-review"

    with pytest.raises(RuntimeError, match="visual review sha256 mismatch"):
        materialize_package(
            texture_report_path=accepted_planar_fixture.texture_report,
            camera_info_path=accepted_planar_fixture.camera_info,
            output=output,
            scene_id="bedroom_4_clean_demo_only",
            storage_mode="copy",
        )

    assert not output.exists()


def test_strict_acceptance_still_rejects_failed_boundary_gate(
    accepted_planar_fixture: AcceptedFixture,
) -> None:
    report = read_json(accepted_planar_fixture.texture_report)
    report["gates"]["wall_boundary_color_continuity"] = {
        "threshold": 8.0,
        "observed": 12.5,
        "passed": False,
    }

    with pytest.raises(RuntimeError, match="planar texture gate did not pass"):
        validate_acceptance(report)


def test_hardlink_mode_preserves_hash_and_inode(
    accepted_planar_fixture: AcceptedFixture,
    tmp_path: Path,
) -> None:
    output = tmp_path / "package-hardlink"
    _, receipt = materialize_package(
        texture_report_path=accepted_planar_fixture.texture_report,
        camera_info_path=accepted_planar_fixture.camera_info,
        output=output,
        scene_id="bedroom_4_clean",
        storage_mode="hardlink",
    )
    source = accepted_planar_fixture.completed_frames["000048"]
    destination = output / "dslr/resized_undistorted_images/000048.png"
    assert source.stat().st_ino == destination.stat().st_ino
    assert sha256_file(source) == sha256_file(destination)
    assert all(item["hardlink_inode_match"] is True for item in receipt["materializations"])


def test_rejects_pending_or_ineligible_texture_report_without_creating_output(
    accepted_planar_fixture: AcceptedFixture,
    tmp_path: Path,
) -> None:
    report = read_json(accepted_planar_fixture.texture_report)
    report["promotion_approved"] = False
    report["eligible_as_round04_clean_plate"] = False
    write_json(accepted_planar_fixture.texture_report, report)
    output = tmp_path / "must-not-exist"
    with pytest.raises(RuntimeError, match="promotion is not approved"):
        materialize_package(
            texture_report_path=accepted_planar_fixture.texture_report,
            camera_info_path=accepted_planar_fixture.camera_info,
            output=output,
            scene_id="bedroom_4_clean",
            storage_mode="copy",
        )
    assert not output.exists()


def test_rejects_nonempty_output_before_materialization(
    accepted_planar_fixture: AcceptedFixture,
    tmp_path: Path,
) -> None:
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "keep.txt").write_text("user data", encoding="utf-8")
    with pytest.raises(RuntimeError, match="output directory is not empty"):
        materialize_package(
            texture_report_path=accepted_planar_fixture.texture_report,
            camera_info_path=accepted_planar_fixture.camera_info,
            output=output,
            scene_id="bedroom_4_clean",
            storage_mode="copy",
        )
    assert (output / "keep.txt").read_text(encoding="utf-8") == "user data"


def test_recomputes_outside_mask_exactness_instead_of_trusting_report_flag(
    accepted_planar_fixture: AcceptedFixture,
    tmp_path: Path,
) -> None:
    report = read_json(accepted_planar_fixture.texture_report)
    completed_path = accepted_planar_fixture.completed_frames["000048"]
    completed = np.asarray(Image.open(completed_path).convert("RGB"), dtype=np.uint8).copy()
    completed[0, 0] = [1, 2, 3]
    Image.fromarray(completed).save(completed_path)
    report["frame_records"][0]["completed_frame_sha256"] = sha256_file(completed_path)
    write_json(accepted_planar_fixture.texture_report, report)
    output = tmp_path / "must-not-exist"
    with pytest.raises(RuntimeError, match="outside-mask RGB is not byte-exact"):
        materialize_package(
            texture_report_path=accepted_planar_fixture.texture_report,
            camera_info_path=accepted_planar_fixture.camera_info,
            output=output,
            scene_id="bedroom_4_clean",
            storage_mode="copy",
        )
    assert not output.exists()
