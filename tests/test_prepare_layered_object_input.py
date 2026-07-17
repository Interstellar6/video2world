from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from scripts.prepare_layered_object_input import (
    build_parser,
    prepare,
    sha256_file,
    sha256_json,
    view_angle_degrees,
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def make_mask(path: Path, *, bbox: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros((60, 80), dtype=np.uint8)
    x0, y0, x1, y1 = bbox
    mask[y0:y1, x0:x1] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(path)
    return mask > 0


def make_frame(path: Path, mask: np.ndarray, color: tuple[int, int, int]) -> None:
    rgb = np.full((60, 80, 3), [30, 35, 40], dtype=np.uint8)
    rgb[mask] = color
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)


def evidence(
    *,
    frame_id: str,
    detection_id: str,
    frame_hash: str,
    mask_hash: str,
    quality: float,
    camera_center: list[float],
    view_direction: list[float],
) -> dict[str, Any]:
    return {
        "frame_id": frame_id,
        "detection_id": detection_id,
        "object_id": "physical_pillow",
        "label": "pillow",
        "quality_score": quality,
        "camera_center_world": camera_center,
        "object_view_direction_world": view_direction,
        "frame_sha256": frame_hash,
        "mask_sha256": mask_hash,
        "peeled_object_residual": False,
        "metrics": {
            "eligible": True,
            "quality_score": quality,
            "gates": {
                "minimum_visible_projected_pixels": True,
                "minimum_mask_hit_ratio": True,
                "minimum_dilated_support_ratio": True,
                "minimum_bbox_iou": True,
                "minimum_sam3_score": True,
            },
        },
    }


def source_record(path: Path, *, frame_id: str) -> dict[str, Any]:
    # The reported resolved path can be the target of a peeled-frame symlink, so
    # its basename need not be frame_id.png. The content hash remains authoritative.
    return {
        "frame_id": frame_id,
        "resolved_path": f"/remote/propainter_output/{int(frame_id) - 10:04d}.png",
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "dimensions": [80, 60],
    }


def mask_record(
    path: Path,
    *,
    frame_id: str,
    detection_id: str,
    bbox: list[int],
) -> dict[str, Any]:
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    return {
        "detection_id": detection_id,
        "item_index": 999,
        "frame_id": frame_id,
        "label": "pillow",
        "score": 0.95,
        "declared_path": path.name,
        "resolved_path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "area_pixels": area,
        "actual_bbox_xyxy": bbox,
    }


def write_association(root: Path) -> tuple[Path, Path, dict[str, Any]]:
    frames_dir = root / "frames"
    first_mask_path = root / "masks" / "looks_like_a_different_object_02.png"
    second_mask_path = root / "masks" / "looks_like_a_different_object_01.png"
    decoy_mask_path = root / "masks" / "physical_pillow_looks_right_but_is_decoy.png"
    first_mask = make_mask(first_mask_path, bbox=(10, 15, 30, 35))
    second_mask = make_mask(second_mask_path, bbox=(20, 10, 45, 34))
    make_mask(decoy_mask_path, bbox=(50, 40, 70, 55))
    first_frame = frames_dir / "000010.png"
    second_frame = frames_dir / "000020.png"
    make_frame(first_frame, first_mask, (218, 218, 212))
    make_frame(second_frame, second_mask, (222, 221, 215))

    first_detection = "opaque:first-detection"
    second_detection = "opaque:second-detection"
    first = evidence(
        frame_id="000010",
        detection_id=first_detection,
        frame_hash=sha256_file(first_frame),
        mask_hash=sha256_file(first_mask_path),
        quality=0.81,
        camera_center=[0.0, 0.0, 0.0],
        view_direction=[0.0, 0.0, 1.0],
    )
    second = evidence(
        frame_id="000020",
        detection_id=second_detection,
        frame_hash=sha256_file(second_frame),
        mask_hash=sha256_file(second_mask_path),
        quality=0.96,
        camera_center=[1.0, 0.0, 0.0],
        view_direction=[0.2, 0.0, 0.98],
    )
    baseline = 1.0
    angle = view_angle_degrees(
        np.asarray(first["object_view_direction_world"]),
        np.asarray(second["object_view_direction_world"]),
    )
    sources = {
        "anchors": [],
        "camera_info": {},
        "sam3_mask_index": {},
        "frames": [
            source_record(first_frame, frame_id="000010"),
            source_record(second_frame, frame_id="000020"),
        ],
        # Deliberately reverse physical-looking filename and array order. The
        # exact detection_id join is the only valid identity operation.
        "masks": [
            mask_record(
                decoy_mask_path,
                frame_id="000010",
                detection_id="opaque:decoy",
                bbox=[50, 40, 70, 55],
            ),
            mask_record(
                second_mask_path,
                frame_id="000020",
                detection_id=second_detection,
                bbox=[20, 10, 45, 34],
            ),
            mask_record(
                first_mask_path,
                frame_id="000010",
                detection_id=first_detection,
                bbox=[10, 15, 30, 35],
            ),
        ],
    }
    report = {
        "schema_version": 1,
        "status": "passed",
        "association_contract": {
            "physical_identity_source": "explicit --anchor object_id=PLY only",
            "forbidden_identity_sources": [
                "SAM3 mask filename",
                "SAM3 per-frame ordinal suffix",
                "cross-frame item order",
            ],
        },
        "thresholds": {
            "minimum_evidence_frames": 2,
            "minimum_camera_baseline_scene_units": 0.5,
            "minimum_object_view_angle_degrees": 5.0,
            "separation_mode": "both",
        },
        "sources": sources,
        "source_set_sha256": sha256_json(sources),
        "peeled_object_ids": ["already_peeled_front"],
        "next_layer_target_ids": ["physical_pillow"],
        "objects": [
            {
                "object_id": "physical_pillow",
                "peeled": False,
                "disposition": "accepted_next_layer_target",
                "evidence_gate_applies": True,
                "evidence_gate_passed": True,
                "selected_evidence_frame_count": 2,
                "selected_evidence": [first, second],
                "selected_frame_separation": [
                    {
                        "left_frame_id": "000010",
                        "right_frame_id": "000020",
                        "camera_baseline_scene_units": baseline,
                        "object_view_angle_degrees": angle,
                        "baseline_passed": True,
                        "view_angle_passed": True,
                        "separation_mode": "both",
                        "passed": True,
                    }
                ],
            }
        ],
        "gates": {
            "passed": True,
            "failure_reasons": [],
            "all_non_peeled_objects_have_separated_evidence": True,
            "peeled_objects_excluded_from_next_layer_targets": True,
        },
    }
    association_path = root / "association.json"
    write_json(association_path, report)
    write_json(
        association_path.with_suffix(".json.sha256.json"),
        {
            "schema_version": 1,
            "report_path": str(association_path),
            "report_sha256": sha256_file(association_path),
            "report_bytes": association_path.stat().st_size,
        },
    )
    return association_path, frames_dir, report


def rewrite_association(path: Path, report: dict[str, Any]) -> None:
    write_json(path, report)
    write_json(
        path.with_suffix(".json.sha256.json"),
        {
            "schema_version": 1,
            "report_path": str(path),
            "report_sha256": sha256_file(path),
            "report_bytes": path.stat().st_size,
        },
    )


def make_args(association: Path, frames_dir: Path, output_dir: Path):
    return build_parser().parse_args(
        [
            "--association",
            str(association),
            "--object-id",
            "physical_pillow",
            "--frames-dir",
            str(frames_dir),
            "--output-dir",
            str(output_dir),
        ]
    )


def test_builds_portable_amodal_manifest_from_exact_selected_evidence(tmp_path: Path) -> None:
    association, frames_dir, _ = write_association(tmp_path)
    output_dir = tmp_path / "output"

    report = prepare(make_args(association, frames_dir, output_dir))

    assert report["status"] == "passed"
    assert report["source_evidence_frame_id"] == "000020"
    assert report["identity_contract"]["mask_lookup"].startswith("exact equality join")
    assert report["crop_policy"]["mode"] == "independent_per_selected_bbox"
    evidence = {item["frame_id"]: item for item in report["artifacts"]["evidence_frames"]}
    assert evidence["000010"]["crop_bbox_xyxy"] == [2, 7, 38, 43]
    assert evidence["000020"]["crop_bbox_xyxy"] == [12, 2, 53, 42]

    manifest_path = output_dir / "scene_reference_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["identity_revalidated"] is True
    assert manifest["identity_revalidated_by"].startswith("explicit_3d_anchor")
    assert {item["detection_id"] for item in manifest["frames"]} == {
        "opaque:first-detection",
        "opaque:second-detection",
    }
    assert sum(item["is_source_frame"] for item in manifest["frames"]) == 1
    assert next(item for item in manifest["frames"] if item["is_source_frame"])[
        "frame_id"
    ] == "000020"

    source_entry = next(item for item in manifest["frames"] if item["is_source_frame"])
    source_rgba = np.asarray(Image.open(output_dir / "source_rgba.png").convert("RGBA"))
    reference_rgba = np.asarray(
        Image.open(output_dir / source_entry["image_path"]).convert("RGBA")
    )
    reference_mask = np.asarray(
        Image.open(output_dir / source_entry["mask_path"]).convert("L")
    )
    assert source_entry["bbox_xyxy"] == [0, 0, source_rgba.shape[1], source_rgba.shape[0]]
    assert np.array_equal(source_rgba, reference_rgba)
    assert np.array_equal(source_rgba[..., 3], reference_mask)

    receipt = json.loads(
        (output_dir / "layered_object_input_report.json.sha256.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["report_sha256"] == sha256_file(
        output_dir / "layered_object_input_report.json"
    )
    assert receipt["source_rgba_sha256"] == sha256_file(output_dir / "source_rgba.png")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda report: report.update(status="failed"), "status=passed"),
        (
            lambda report: report["objects"][0].update(
                disposition="failed_insufficient_separated_evidence"
            ),
            "accepted next-layer target",
        ),
        (
            lambda report: report["objects"][0]["selected_frame_separation"][0].update(
                passed=False
            ),
            "fails separation gates",
        ),
    ],
)
def test_fails_closed_on_report_or_object_gate_changes(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    association, frames_dir, report = write_association(tmp_path)
    mutation(report)
    rewrite_association(association, report)

    with pytest.raises(ValueError, match=message):
        prepare(make_args(association, frames_dir, tmp_path / "output"))


def test_fails_closed_when_selected_frame_content_no_longer_matches_hash(tmp_path: Path) -> None:
    association, frames_dir, _ = write_association(tmp_path)
    frame = np.asarray(Image.open(frames_dir / "000020.png").convert("RGB")).copy()
    frame[0, 0] = [255, 0, 0]
    Image.fromarray(frame).save(frames_dir / "000020.png")

    with pytest.raises(ValueError, match="sha256 mismatch"):
        prepare(make_args(association, frames_dir, tmp_path / "output"))


def test_never_falls_back_to_filename_or_mask_array_order_for_identity(tmp_path: Path) -> None:
    association, frames_dir, report = write_association(tmp_path)
    report["objects"][0]["selected_evidence"][0]["detection_id"] = "missing:opaque-id"
    rewrite_association(association, report)

    with pytest.raises(ValueError, match=r"exact sources\.frames/sources\.masks"):
        prepare(make_args(association, frames_dir, tmp_path / "output"))
