from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "prefill_clean_plate_multiview.py"
SPEC = importlib.util.spec_from_file_location("prefill_clean_plate_multiview", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_args(root: Path, *, donor_mask_index: Path | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        input_manifest=root / "input_manifest.json",
        camera_info=root / "camera_info.json",
        donor_frames_dir=root / "frames",
        depth_dir=root / "depth",
        donor_mask_index=donor_mask_index,
        donor_mask_label="pillow",
        donor_mask_min_score=None,
        donor_mask_dilation=0,
        target_frame_ids="000000",
        donor_frame_ids="000001,000002",
        exclude_same_frame=True,
        sample_stride=1,
        splat_radius=0,
        min_support=2,
        absolute_depth_tolerance=0.05,
        relative_depth_tolerance=0.01,
        contact_sheet_samples=1,
        output=root / "output",
    )


def make_fixture(root: Path) -> tuple[np.ndarray, np.ndarray]:
    height, width = 6, 8
    frames = root / "frames"
    depth = root / "depth"
    masks = root / "masks"
    frames.mkdir(parents=True)
    depth.mkdir(parents=True)
    masks.mkdir(parents=True)
    source = np.full((height, width, 3), [180, 180, 180], dtype=np.uint8)
    donor = np.zeros((height, width, 3), dtype=np.uint8)
    donor[:, :, 0] = np.arange(width, dtype=np.uint8)[None, :] * 10
    donor[:, :, 1] = np.arange(height, dtype=np.uint8)[:, None] * 20
    donor[:, :, 2] = 40
    Image.fromarray(source).save(frames / "000000.png")
    Image.fromarray(donor).save(frames / "000001.png")
    Image.fromarray(donor).save(frames / "000002.png")
    for frame_id in ("000000", "000001", "000002"):
        np.save(depth / f"{frame_id}.npy", np.full((height, width), 2.0, dtype=np.float32))
    removal = np.zeros((height, width), dtype=np.uint8)
    removal[2:4, 3:6] = 255
    Image.fromarray(removal).save(masks / "target.png")
    intrinsic = {"fx": 20.0, "fy": 20.0, "cx": 4.0, "cy": 3.0, "w": width, "h": height}
    identity = np.eye(4).tolist()
    write_json(
        root / "camera_info.json",
        {
            "extrinsic_type": "world_to_camera",
            "intrinsic": intrinsic,
            "extrinsic": {frame_id: identity for frame_id in ("000000", "000001", "000002")},
        },
    )
    write_json(
        root / "input_manifest.json",
        {
            "frame_records": [
                {
                    "sequence_index": 0,
                    "frame_id": "000000",
                    "source_frame": str(frames / "000000.png"),
                    "union_mask": str(masks / "target.png"),
                }
            ]
        },
    )
    return source, removal > 0


def make_cumulative_contract(root: Path) -> Path:
    removal_path = root / "masks" / "target.png"
    removal_sha = MODULE.sha256_file(removal_path)
    records = []
    index_items = []
    assets = {}
    for sequence_index, frame_id in enumerate(("000000", "000001", "000002")):
        frame_path = root / "frames" / f"{frame_id}.png"
        records.append(
            {
                "sequence_index": sequence_index,
                "frame_id": frame_id,
                "source_frame": str(frame_path),
                "union_mask": str(removal_path),
                "union_mask_sha256": removal_sha,
                "donor_exclusion_mask": str(removal_path),
                "donor_exclusion_mask_sha256": removal_sha,
            }
        )
        index_items.append(
            {
                "frame_id": frame_id,
                "image": f"{frame_id}.png",
                "source_rgb_path": str(frame_path),
                "source_rgb_sha256": MODULE.sha256_file(frame_path),
                "label": "cumulative_removed",
                "mask_path": str(removal_path),
                "mask_sha256": removal_sha,
            }
        )
        assets[frame_id] = {
            "path": str(frame_path),
            "sha256": MODULE.sha256_file(frame_path),
            "role": "original_observed_rgb_only",
        }
    index_path = root / "donor_exclusion_index.json"
    write_json(index_path, {"items": index_items})
    manifest_path = root / "input_manifest.json"
    write_json(
        manifest_path,
        {
            "kind": "video2world.cumulative_removal_manifest",
            "status": "ready_for_measured_multiview_prefill",
            "round_index": 1,
            "removed_object_ids": ["front"],
            "lineage": {"previous_round_index": None},
            "gates": {
                "strict_front_to_back_order": True,
                "input_is_immediately_previous_round": True,
                "all_previous_masks_are_accumulated": True,
                "all_removed_objects_are_excluded_from_donors": True,
                "donors_are_original_observed_rgb": True,
                "unresolved_residual_is_never_donor_evidence": True,
            },
            "donor_contract": {
                "role": "original_observed_rgb_only",
                "generated_or_prefilled_rgb_as_donor": False,
                "unresolved_residual_as_donor": False,
                "exclusion_label": "cumulative_removed",
                "exclusion_index": str(index_path),
                "exclusion_index_sha256": MODULE.sha256_file(index_path),
                "assets": assets,
            },
            "frame_records": records,
        },
    )
    return index_path


def make_associated_cumulative_contract(root: Path) -> Path:
    removal_path = root / "masks" / "target.png"
    removal_sha = MODULE.sha256_file(removal_path)
    object_ids = ["front", "left"]
    eligible_ids = ["000001", "000002"]
    associated_root = root / "associated_donor_evidence"
    items = []
    donor_unions = {}
    assets = {}
    for frame_id in eligible_ids:
        frame_path = root / "frames" / f"{frame_id}.png"
        assets[frame_id] = {
            "path": str(frame_path),
            "sha256": MODULE.sha256_file(frame_path),
            "role": "original_observed_rgb_only",
        }
        union = np.zeros((6, 8), dtype=bool)
        for object_index, object_id in enumerate(object_ids):
            mask = np.zeros((6, 8), dtype=np.uint8)
            if object_index == 0:
                mask[2:4, 3:5] = 255
            else:
                mask[2:4, 5:6] = 255
            union |= mask > 0
            relative_mask = Path("masks") / frame_id / f"{object_id}.png"
            mask_path = associated_root / relative_mask
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask).save(mask_path)
            items.append(
                {
                    "frame_id": frame_id,
                    "image": frame_path.name,
                    "source_rgb_path": str(frame_path),
                    "source_rgb_sha256": MODULE.sha256_file(frame_path),
                    "label": "associated_removed_object",
                    "object_id": object_id,
                    "detection_id": f"{frame_id}:mask:{object_index:06d}",
                    "score": 0.95,
                    "mask_path": relative_mask.as_posix(),
                    "mask_sha256": MODULE.sha256_file(mask_path),
                    "mask_pixels": int((mask > 0).sum()),
                    "physical_identity_source": "association_explicit_3d_anchor",
                }
            )
        union_path = associated_root / "unions" / f"{frame_id}.png"
        union_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(union.astype(np.uint8) * 255).save(union_path)
        donor_unions[frame_id] = {
            "path": f"unions/{frame_id}.png",
            "sha256": MODULE.sha256_file(union_path),
            "pixels": int(union.sum()),
            "object_ids": object_ids,
            "component_mask_count": len(object_ids),
        }
    index_path = associated_root / "mask_index.json"
    write_json(
        index_path,
        {
            "schema_version": 1,
            "kind": MODULE.ASSOCIATED_DONOR_INDEX_KIND,
            "status": "ready_for_measured_multiview_prefill",
            "selected_object_ids": object_ids,
            "target_removed_object_union": object_ids,
            "label": "associated_removed_object",
            "eligible_donor_frame_ids": eligible_ids,
            "items": items,
            "donor_unions": donor_unions,
            "gates": {
                "physical_identity_from_explicit_3d_anchors": True,
                "every_item_bound_to_exact_source_rgb_sha256": True,
                "every_mask_sha256_verified": True,
                "incomplete_frames_excluded_from_donors": True,
                "selected_object_ids_equal_cumulative_removed_object_union": True,
                "every_eligible_donor_has_every_removed_object_mask": True,
                "per_donor_multi_mask_union_materialized": True,
                "per_frame_physical_assignments_are_one_to_one": True,
            },
        },
    )
    manifest_path = root / "input_manifest.json"
    write_json(
        manifest_path,
        {
            "kind": "video2world.cumulative_removal_manifest",
            "status": "ready_for_measured_multiview_prefill",
            "round_index": 2,
            "removed_object_ids": object_ids,
            "lineage": {"previous_round_index": 1},
            "gates": {
                "strict_front_to_back_order": True,
                "input_is_immediately_previous_round": True,
                "all_previous_masks_are_accumulated": True,
                "all_removed_objects_are_excluded_from_donors": True,
                "donors_are_original_observed_rgb": True,
                "unresolved_residual_is_never_donor_evidence": True,
            },
            "donor_contract": {
                "role": "original_observed_rgb_only",
                "generated_or_prefilled_rgb_as_donor": False,
                "unresolved_residual_as_donor": False,
                "exclusion_mode": "associated_physical_instance_masks",
                "exclusion_label": "associated_removed_object",
                "exclusion_index": str(index_path),
                "exclusion_index_sha256": MODULE.sha256_file(index_path),
                "eligible_donor_frame_ids": eligible_ids,
                "physical_instance_object_ids": object_ids,
                "per_donor_multi_mask_union": True,
                "assets": assets,
            },
            "frame_records": [
                {
                    "sequence_index": 0,
                    "frame_id": "000000",
                    "source_frame": str(root / "frames" / "000000.png"),
                    "union_mask": str(removal_path),
                    "union_mask_sha256": removal_sha,
                    "donor_exclusion_mask": str(removal_path),
                    "donor_exclusion_mask_sha256": removal_sha,
                    "removed_object_ids": object_ids,
                }
            ],
        },
    )
    return index_path


def refresh_associated_index_hash(root: Path, index_path: Path) -> None:
    manifest_path = root / "input_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["donor_contract"]["exclusion_index_sha256"] = MODULE.sha256_file(index_path)
    write_json(manifest_path, manifest)


def test_prefill_fuses_donors_and_preserves_outside_mask_exactly(tmp_path: Path) -> None:
    source, removal = make_fixture(tmp_path)

    report = MODULE.build_prefill(make_args(tmp_path))

    output = np.asarray(Image.open(tmp_path / "output" / "frames" / "0000.png"))
    donor = np.asarray(Image.open(tmp_path / "frames" / "000001.png"))
    residual = np.asarray(Image.open(tmp_path / "output" / "masks" / "0000.png")) > 0
    measured_depth_path = tmp_path / "output" / "depth" / "0000.npy"
    measured_depth = np.load(measured_depth_path)
    assert np.array_equal(output[~removal], source[~removal])
    assert np.array_equal(output[removal], donor[removal])
    assert not residual.any()
    assert measured_depth.dtype == np.float32
    assert np.all(np.isfinite(measured_depth[removal]))
    assert np.all(measured_depth[removal] > 0)
    assert np.all(np.isnan(measured_depth[~removal]))
    assert report["status"] == "technical_passed"
    assert report["promotion_approved"] is False
    assert report["frame_records"][0]["coverage_fraction"] == 1.0
    assert report["gates"]["new_depth_normal_estimation"].startswith("required")
    record = report["frame_records"][0]
    assert record["measured_depth_sha256"] == MODULE.sha256_file(measured_depth_path)
    assert record["measured_depth_role"] == "fused_multiview_measured_depth"
    assert record["measured_depth_semantics"] == "target_camera_positive_z_axis"
    assert record["measured_depth_valid_pixels"] == record["covered_pixels"]
    assert report["gates"]["fused_measured_metric_depth_materialized"] is True
    assert report["pixel_provenance"]["generated_pixels"] == 0
    assert report["pixel_provenance"]["propainter_pixels"] == 0
    receipt = json.loads(
        (tmp_path / "output" / "multiview_prefill_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["report_sha256"] == MODULE.sha256_file(
        tmp_path / "output" / "multiview_prefill_report.json"
    )
    assert receipt["fused_metric_depth_artifacts_are_hashed_in_report"] is True
    assert receipt["measured_provenance_excludes_generated_pixels"] is True
    assert receipt["measured_provenance_excludes_propainter_pixels"] is True


def test_donor_object_masks_prevent_copying_occluder_pixels(tmp_path: Path) -> None:
    _, removal = make_fixture(tmp_path)
    donor_mask = removal.astype(np.uint8) * 255
    donor_mask_paths = []
    for frame_id in ("000001", "000002"):
        path = tmp_path / "masks" / f"donor_{frame_id}.png"
        Image.fromarray(donor_mask).save(path)
        donor_mask_paths.append(path)
    mask_index = tmp_path / "donor_mask_index.json"
    write_json(
        mask_index,
        {
            "items": [
                {
                    "frame_id": frame_id,
                    "image": f"{frame_id}.png",
                    "label": "pillow",
                    "score": 0.99,
                    "mask_path": str(path),
                    "mask_sha256": MODULE.sha256_file(path),
                    "source_rgb_sha256": MODULE.sha256_file(
                        tmp_path / "frames" / f"{frame_id}.png"
                    ),
                }
                for frame_id, path in zip(("000001", "000002"), donor_mask_paths, strict=True)
            ]
        },
    )

    report = MODULE.build_prefill(make_args(tmp_path, donor_mask_index=mask_index))

    residual = np.asarray(Image.open(tmp_path / "output" / "masks" / "0000.png")) > 0
    measured_depth = np.load(tmp_path / "output" / "depth" / "0000.npy")
    assert np.array_equal(residual, removal)
    assert np.all(np.isnan(measured_depth))
    assert report["frame_records"][0]["coverage_fraction"] == 0.0
    assert report["status"] == "technical_failed_no_support"
    assert report["gates"]["donor_support_available_for_every_target"] is False
    assert report["donor_mask_index_contract"] == {
        "item_count": 2,
        "frame_count": 2,
        "mask_sha256_verified": True,
        "source_rgb_sha256_bound": True,
    }
    assert report["gates"]["donor_exclusion_masks_content_verified"] is True
    assert report["gates"]["donor_exclusion_masks_bound_to_exact_source_rgb"] is True


def test_donor_mask_index_rejects_tampered_mask_hash(tmp_path: Path) -> None:
    make_fixture(tmp_path)
    mask_path = tmp_path / "masks" / "donor.png"
    Image.fromarray(np.zeros((6, 8), dtype=np.uint8)).save(mask_path)
    index_path = tmp_path / "donor_mask_index.json"
    write_json(
        index_path,
        {
            "items": [
                {
                    "frame_id": "000001",
                    "image": "000001.png",
                    "label": "pillow",
                    "score": 0.99,
                    "mask_path": str(mask_path),
                    "mask_sha256": "0" * 64,
                    "source_rgb_sha256": MODULE.sha256_file(tmp_path / "frames" / "000001.png"),
                }
            ]
        },
    )

    with pytest.raises(ValueError, match="mask SHA-256 mismatch"):
        MODULE.build_prefill(make_args(tmp_path, donor_mask_index=index_path))


def test_donor_mask_index_rejects_stale_source_binding(tmp_path: Path) -> None:
    make_fixture(tmp_path)
    mask_path = tmp_path / "masks" / "donor.png"
    Image.fromarray(np.zeros((6, 8), dtype=np.uint8)).save(mask_path)
    index_path = tmp_path / "donor_mask_index.json"
    write_json(
        index_path,
        {
            "items": [
                {
                    "frame_id": "000001",
                    "image": "000001.png",
                    "label": "pillow",
                    "score": 0.99,
                    "mask_path": str(mask_path),
                    "mask_sha256": MODULE.sha256_file(mask_path),
                    "source_rgb_sha256": "0" * 64,
                }
            ]
        },
    )

    with pytest.raises(ValueError, match="source RGB SHA-256 mismatch"):
        MODULE.build_prefill(make_args(tmp_path, donor_mask_index=index_path))


def test_robust_fuse_rejects_a_far_depth_outlier() -> None:
    depths = np.array([[2.00], [2.01], [5.00]], dtype=np.float32)
    colors = np.array([[[240, 240, 240]], [[220, 220, 220]], [[255, 0, 0]]], dtype=np.float32)

    color, depth, support, accepted = MODULE.robust_fuse(
        depths,
        colors,
        absolute_depth_tolerance=0.05,
        relative_depth_tolerance=0.01,
        min_support=2,
    )

    assert accepted.tolist() == [True]
    assert support.tolist() == [2]
    assert color.tolist() == [[230, 230, 230]]
    assert depth.tolist() == pytest.approx([2.005])


def test_camera_convention_is_fail_closed(tmp_path: Path) -> None:
    make_fixture(tmp_path)
    camera_path = tmp_path / "camera_info.json"
    value = json.loads(camera_path.read_text(encoding="utf-8"))
    value["extrinsic_type"] = "camera_to_world"
    write_json(camera_path, value)

    with pytest.raises(ValueError, match="world_to_camera"):
        MODULE.build_prefill(make_args(tmp_path))


def test_cumulative_contract_is_enforced_and_reported(tmp_path: Path) -> None:
    make_fixture(tmp_path)
    index_path = make_cumulative_contract(tmp_path)
    args = make_args(tmp_path, donor_mask_index=index_path)
    args.donor_mask_label = "cumulative_removed"

    report = MODULE.build_prefill(args)

    contract = report["cumulative_removal_contract"]
    assert contract["enforced"] is True
    assert contract["removed_object_ids"] == ["front"]
    assert contract["unresolved_residual_as_donor"] is False
    assert report["pixel_provenance"]["generated_pixels"] == 0
    assert report["pixel_provenance"]["propainter_pixels"] == 0
    assert report["gates"]["cumulative_front_to_back_contract_enforced"] is True
    assert report["gates"]["donor_exclusion_masks_content_verified"] is True
    assert report["gates"]["donor_exclusion_masks_bound_to_exact_source_rgb"] is True


def test_cumulative_contract_rejects_a_modified_donor_frame(tmp_path: Path) -> None:
    make_fixture(tmp_path)
    index_path = make_cumulative_contract(tmp_path)
    Image.fromarray(np.full((6, 8, 3), 255, dtype=np.uint8)).save(
        tmp_path / "frames" / "000001.png"
    )
    args = make_args(tmp_path, donor_mask_index=index_path)
    args.donor_mask_label = "cumulative_removed"

    with pytest.raises(ValueError, match="source RGB SHA-256 mismatch"):
        MODULE.build_prefill(args)


def test_associated_cumulative_contract_uses_eligible_multi_mask_donors(
    tmp_path: Path,
) -> None:
    _, removal = make_fixture(tmp_path)
    index_path = make_associated_cumulative_contract(tmp_path)
    args = make_args(tmp_path, donor_mask_index=index_path)
    args.donor_mask_label = "associated_removed_object"

    report = MODULE.build_prefill(args)

    contract = report["cumulative_removal_contract"]
    assert contract["donor_exclusion_mode"] == "associated_physical_instance_masks"
    assert contract["eligible_donor_frame_ids"] == ["000001", "000002"]
    assert contract["per_donor_multi_mask_union"] is True
    assert report["config"]["donor_frame_ids"] == ["000001", "000002"]
    assert report["gates"]["physical_instance_donor_exclusion_enforced"] is True
    assert report["gates"]["donor_support_not_boundary_concentrated"] is False
    assert report["config"]["donor_mask_dilation"] == 0
    assert report["config"]["boundary_guard_extra_dilation"] == 4
    assert report["donor_mask_index_contract"]["physical_instance_identity_bound"] is True
    assert report["donor_mask_index_contract"]["per_donor_mask_counts"] == {
        "000001": 2,
        "000002": 2,
    }
    assert all(len(asset["exclusion_masks"]) == 2 for asset in report["donor_assets"].values())
    residual = np.asarray(Image.open(tmp_path / "output" / "masks" / "0000.png")) > 0
    assert np.array_equal(residual, removal)
    assert report["status"] == "technical_failed_no_support"
    assert report["no_support_frame_ids"] == ["000000"]
    assert report["gates"]["donor_support_available_for_every_target"] is False


def test_associated_contract_rejects_noneligible_donor_selection(tmp_path: Path) -> None:
    make_fixture(tmp_path)
    index_path = make_associated_cumulative_contract(tmp_path)
    args = make_args(tmp_path, donor_mask_index=index_path)
    args.donor_mask_label = "associated_removed_object"
    args.donor_frame_ids = "000001"

    with pytest.raises(ValueError, match="exactly equal associated eligible donor ids"):
        MODULE.build_prefill(args)


def test_prefill_rejects_raw_associated_index_before_materialization(tmp_path: Path) -> None:
    make_fixture(tmp_path)
    index_path = make_associated_cumulative_contract(tmp_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["status"] = MODULE.RAW_ASSOCIATED_INDEX_STATUS
    write_json(index_path, index)
    refresh_associated_index_hash(tmp_path, index_path)
    args = make_args(tmp_path, donor_mask_index=index_path)
    args.donor_mask_label = "associated_removed_object"

    with pytest.raises(ValueError, match="requires cumulative materialization"):
        MODULE.build_prefill(args)

    assert not args.output.exists()


def test_prefill_revalidates_legacy_materialized_index_without_new_uniqueness_gate(
    tmp_path: Path,
) -> None:
    make_fixture(tmp_path)
    index_path = make_associated_cumulative_contract(tmp_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    del index["gates"]["per_frame_physical_assignments_are_one_to_one"]
    write_json(index_path, index)
    refresh_associated_index_hash(tmp_path, index_path)
    args = make_args(tmp_path, donor_mask_index=index_path)
    args.donor_mask_label = "associated_removed_object"

    report = MODULE.build_prefill(args)

    assert report["gates"]["physical_instance_donor_exclusion_enforced"] is True
    assert report["status"] == "technical_failed_no_support"


def test_prefill_rejects_materialized_associated_index_without_cumulative_contract(
    tmp_path: Path,
) -> None:
    make_fixture(tmp_path)
    manifest_path = tmp_path / "input_manifest.json"
    generic_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    index_path = make_associated_cumulative_contract(tmp_path)
    write_json(manifest_path, generic_manifest)
    args = make_args(tmp_path, donor_mask_index=index_path)
    args.donor_mask_label = "associated_removed_object"

    with pytest.raises(ValueError, match="requires its cumulative removal contract"):
        MODULE.build_prefill(args)

    assert not args.output.exists()


def test_prefill_rejects_reused_physical_detection(tmp_path: Path) -> None:
    make_fixture(tmp_path)
    index_path = make_associated_cumulative_contract(tmp_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    for frame_id in ("000001", "000002"):
        front, left = [item for item in index["items"] if item["frame_id"] == frame_id]
        left["detection_id"] = front["detection_id"]
    write_json(index_path, index)
    refresh_associated_index_hash(tmp_path, index_path)
    args = make_args(tmp_path, donor_mask_index=index_path)
    args.donor_mask_label = "associated_removed_object"

    with pytest.raises(ValueError, match="physical detection is reused"):
        MODULE.build_prefill(args)

    assert not args.output.exists()


def test_prefill_rejects_reused_physical_mask_path_or_hash(tmp_path: Path) -> None:
    make_fixture(tmp_path)
    index_path = make_associated_cumulative_contract(tmp_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    for frame_id in ("000001", "000002"):
        front, left = [item for item in index["items"] if item["frame_id"] == frame_id]
        left.update(
            mask_path=front["mask_path"],
            mask_sha256=front["mask_sha256"],
            mask_pixels=front["mask_pixels"],
        )
        union_record = index["donor_unions"][frame_id]
        union_path = index_path.parent / union_record["path"]
        front_mask = MODULE.mask_bool(index_path.parent / front["mask_path"])
        Image.fromarray(front_mask.astype(np.uint8) * 255).save(union_path)
        union_record["sha256"] = MODULE.sha256_file(union_path)
        union_record["pixels"] = int(front_mask.sum())
    write_json(index_path, index)
    refresh_associated_index_hash(tmp_path, index_path)
    args = make_args(tmp_path, donor_mask_index=index_path)
    args.donor_mask_label = "associated_removed_object"

    with pytest.raises(ValueError, match="mask path or SHA-256 is reused"):
        MODULE.build_prefill(args)

    assert not args.output.exists()


@pytest.mark.parametrize(
    ("field", "expected_error"),
    [
        ("object_id", "invalid frame or object identity"),
        ("detection_id", "physical detection identity"),
        ("physical_identity_source", "category-level rather than physical-instance"),
        ("mask_sha256", "no mask SHA-256"),
        ("source_rgb_sha256", "no source RGB SHA-256"),
    ],
)
def test_associated_contract_rejects_category_or_unbound_items(
    tmp_path: Path,
    field: str,
    expected_error: str,
) -> None:
    make_fixture(tmp_path)
    index_path = make_associated_cumulative_contract(tmp_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    del index["items"][0][field]
    write_json(index_path, index)
    refresh_associated_index_hash(tmp_path, index_path)
    args = make_args(tmp_path, donor_mask_index=index_path)
    args.donor_mask_label = "associated_removed_object"

    with pytest.raises(ValueError, match=expected_error):
        MODULE.build_prefill(args)


def test_associated_contract_rejects_union_not_equal_to_object_masks(
    tmp_path: Path,
) -> None:
    make_fixture(tmp_path)
    index_path = make_associated_cumulative_contract(tmp_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    union_path = index_path.parent / index["donor_unions"]["000001"]["path"]
    Image.fromarray(np.zeros((6, 8), dtype=np.uint8)).save(union_path)
    index["donor_unions"]["000001"]["sha256"] = MODULE.sha256_file(union_path)
    index["donor_unions"]["000001"]["pixels"] = 0
    write_json(index_path, index)
    refresh_associated_index_hash(tmp_path, index_path)
    args = make_args(tmp_path, donor_mask_index=index_path)
    args.donor_mask_label = "associated_removed_object"

    with pytest.raises(ValueError, match="differs from its object masks"):
        MODULE.build_prefill(args)


def test_associated_boundary_concentrated_support_stays_unresolved(
    tmp_path: Path,
) -> None:
    _, removal = make_fixture(tmp_path)
    index_path = make_associated_cumulative_contract(tmp_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    for frame_id in ("000001", "000002"):
        union = np.zeros((6, 8), dtype=bool)
        for item in [value for value in index["items"] if value["frame_id"] == frame_id]:
            mask = np.zeros((6, 8), dtype=np.uint8)
            column = 2 if item["object_id"] == "front" else 6
            mask[2:4, column] = 255
            mask_path = index_path.parent / item["mask_path"]
            Image.fromarray(mask).save(mask_path)
            item["mask_sha256"] = MODULE.sha256_file(mask_path)
            item["mask_pixels"] = int((mask > 0).sum())
            union |= mask > 0
        union_record = index["donor_unions"][frame_id]
        union_path = index_path.parent / union_record["path"]
        Image.fromarray(union.astype(np.uint8) * 255).save(union_path)
        union_record["sha256"] = MODULE.sha256_file(union_path)
        union_record["pixels"] = int(union.sum())
    write_json(index_path, index)
    refresh_associated_index_hash(tmp_path, index_path)
    args = make_args(tmp_path, donor_mask_index=index_path)
    args.donor_mask_label = "associated_removed_object"
    args.boundary_guard_extra_dilation = 2
    args.minimum_boundary_guard_retention = 0.5

    report = MODULE.build_prefill(args)

    concentration = report["frame_records"][0]["donor_boundary_concentration"]
    assert concentration["configured_dilation_px"] == 0
    assert concentration["guard_dilation_px"] == 2
    assert concentration["configured_supported_pixels"] == int(removal.sum())
    assert concentration["guard_supported_pixels"] == 0
    assert concentration["guard_retention_fraction"] == 0.0
    assert concentration["passed"] is False
    assert concentration["final_measured_pixels_are_guard_stable_only"] is True
    assert report["gates"]["donor_support_not_boundary_concentrated"] is False
    assert report["status"] == "technical_failed"
    assert report["frame_records"][0]["covered_pixels"] == 0
    residual = np.asarray(Image.open(tmp_path / "output" / "masks" / "0000.png")) > 0
    assert np.array_equal(residual, removal)


def test_prefill_failure_cleans_staging_and_allows_retry(tmp_path: Path) -> None:
    make_fixture(tmp_path)
    camera_path = tmp_path / "camera_info.json"
    camera = json.loads(camera_path.read_text(encoding="utf-8"))
    camera["intrinsic"]["w"] = 999
    write_json(camera_path, camera)
    args = make_args(tmp_path)

    with pytest.raises(ValueError, match="camera/image shape mismatch"):
        MODULE.build_prefill(args)

    assert not args.output.exists()
    assert not list(tmp_path.glob(".output.staging-*"))

    camera["intrinsic"]["w"] = 8
    write_json(camera_path, camera)
    report = MODULE.build_prefill(args)

    assert report["status"] == "technical_passed"
    assert args.output.is_dir()
    assert Path(report["frame_records"][0]["prefill_frame"]).is_file()
    assert str(args.output) in report["frame_records"][0]["prefill_frame"]
    report_path = args.output / "multiview_prefill_report.json"
    assert ".output.staging-" not in report_path.read_text(encoding="utf-8")
    receipt = json.loads(
        (args.output / "multiview_prefill_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["report_sha256"] == MODULE.sha256_file(report_path)
