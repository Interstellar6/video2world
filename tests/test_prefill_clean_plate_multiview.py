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
                    "image": f"{frame_id}.png",
                    "label": "pillow",
                    "score": 0.99,
                    "mask_path": str(path),
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


def test_cumulative_contract_rejects_a_modified_donor_frame(tmp_path: Path) -> None:
    make_fixture(tmp_path)
    index_path = make_cumulative_contract(tmp_path)
    Image.fromarray(np.full((6, 8, 3), 255, dtype=np.uint8)).save(
        tmp_path / "frames" / "000001.png"
    )
    args = make_args(tmp_path, donor_mask_index=index_path)
    args.donor_mask_label = "cumulative_removed"

    with pytest.raises(ValueError, match="not the contracted original"):
        MODULE.build_prefill(args)
