from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from scripts.prepare_propainter_peel_round import (
    EXPLICIT_IDENTITY_SOURCE,
    FORBIDDEN_IDENTITY_SOURCES,
    complete_mask,
    materialize_round,
    sha256_file,
    sha256_json,
)

OBJECT_ID = "sam3_pillow_left"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_assignment(frame_id: str, detection_id: str, frame_sha: str, mask_sha: str) -> dict:
    gates = {
        "minimum_visible_projected_pixels": True,
        "minimum_mask_hit_ratio": True,
        "minimum_dilated_support_ratio": True,
        "minimum_bbox_iou": True,
        "minimum_sam3_score": True,
    }
    return {
        "frame_id": frame_id,
        "detection_id": detection_id,
        "object_id": OBJECT_ID,
        "label": "pillow",
        "quality_score": 0.97,
        "frame_sha256": frame_sha,
        "mask_sha256": mask_sha,
        "peeled_object_residual": False,
        "metrics": {
            "detection_id": detection_id,
            "eligible": True,
            "quality_score": 0.97,
            "gates": gates,
        },
    }


def seal_report(report: dict[str, Any]) -> dict[str, Any]:
    report["thresholds_sha256"] = sha256_json(report["thresholds"])
    report["source_set_sha256"] = sha256_json(report["sources"])
    return report


def make_fixture(root: Path) -> tuple[dict[str, Any], Path, Path]:
    frames_dir = root / "frames"
    masks_dir = root / "masks"
    frames_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)

    frame_sources: list[dict[str, Any]] = []
    mask_sources: list[dict[str, Any]] = []
    frame_records: list[dict[str, Any]] = []
    object_assignments: list[dict[str, Any]] = []
    # The report is intentionally reverse ordered. Output must be frame-id ordered.
    fixtures = [
        ("000002", "det-physical-b", "mask_named_ordinal_01.png", (4, 1, 7, 4)),
        ("000001", "det-physical-a", "mask_named_ordinal_99.png", (1, 2, 4, 5)),
    ]
    for frame_id, detection_id, mask_name, (left, top, right, bottom) in fixtures:
        frame_path = frames_dir / f"{frame_id}.png"
        Image.new("RGB", (8, 6), (int(frame_id[-1]) * 30, 80, 120)).save(frame_path)
        mask_path = masks_dir / mask_name
        mask = Image.new("L", (8, 6), 0)
        for y in range(top, bottom):
            for x in range(left, right):
                mask.putpixel((x, y), 255)
        mask.save(mask_path)

        frame_sha = sha256_file(frame_path)
        mask_sha = sha256_file(mask_path)
        assignment = make_assignment(frame_id, detection_id, frame_sha, mask_sha)
        object_assignments.append(copy.deepcopy(assignment))
        frame_sources.append(
            {
                "frame_id": frame_id,
                "resolved_path": f"/remote/previous-clean-plate/{frame_id}.png",
                "sha256": frame_sha,
                "bytes": frame_path.stat().st_size,
                "dimensions": [8, 6],
            }
        )
        mask_sources.append(
            {
                "detection_id": detection_id,
                "item_index": len(mask_sources),
                "frame_id": frame_id,
                "label": "pillow",
                "score": 0.98,
                "declared_path": str(mask_path),
                "resolved_path": str(mask_path),
                "sha256": mask_sha,
                "bytes": mask_path.stat().st_size,
                "area_pixels": (right - left) * (bottom - top),
                "actual_bbox_xyxy": [left, top, right, bottom],
            }
        )
        frame_records.append(
            {
                "frame_id": frame_id,
                "assignments": [assignment],
                "pair_metrics": [
                    {
                        "object_id": OBJECT_ID,
                        "detection_id": detection_id,
                        "eligible": True,
                        "quality_score": 0.97,
                        "gates": assignment["metrics"]["gates"],
                    }
                ],
            }
        )

    thresholds = {
        "minimum_evidence_frames": 2,
        "minimum_camera_baseline_scene_units": 0.1,
        "minimum_object_view_angle_degrees": 4.0,
    }
    report = {
        "schema_version": 1,
        "status": "passed",
        "association_contract": {
            "physical_identity_source": EXPLICIT_IDENTITY_SOURCE,
            "forbidden_identity_sources": sorted(FORBIDDEN_IDENTITY_SOURCES),
        },
        "thresholds": thresholds,
        "sources": {
            "anchors": [
                {
                    "object_id": OBJECT_ID,
                    "resolved_path": "/remote/anchors/left.ply",
                    "sha256": "a" * 64,
                    "bytes": 123,
                    "point_count": 42,
                }
            ],
            "camera_info": {"sha256": "b" * 64},
            "sam3_mask_index": {"sha256": "c" * 64},
            "frames": frame_sources,
            "masks": mask_sources,
        },
        "peeled_object_ids": ["sam3_pillow_front"],
        "next_layer_target_ids": [OBJECT_ID],
        "objects": [
            {
                "object_id": OBJECT_ID,
                "peeled": False,
                "disposition": "accepted_next_layer_target",
                "evidence_gate_applies": True,
                "evidence_gate_passed": True,
                "high_confidence_assignment_count": 2,
                "selected_evidence_frame_count": 2,
                "selected_evidence": copy.deepcopy(object_assignments),
                "all_high_confidence_assignments": object_assignments,
            }
        ],
        "frames": frame_records,
        "gates": {
            "passed": True,
            "failure_reasons": [],
            "all_non_peeled_objects_have_separated_evidence": True,
            "peeled_objects_excluded_from_next_layer_targets": True,
        },
    }
    seal_report(report)
    association = root / "association.json"
    write_json(association, report)
    return report, association, frames_dir


def args_for(association: Path, frames_dir: Path, output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        association=association,
        object_id=OBJECT_ID,
        frames_dir=frames_dir,
        output_dir=output,
    )


def rewrite_report(path: Path, report: dict[str, Any]) -> None:
    seal_report(report)
    write_json(path, report)


def replace_detection_mask(
    report: dict[str, Any],
    *,
    detection_id: str,
    mask: Image.Image,
) -> None:
    mask_source = next(
        item for item in report["sources"]["masks"] if item["detection_id"] == detection_id
    )
    mask_path = Path(mask_source["resolved_path"])
    binary = mask.convert("L").point(lambda value: 255 if value > 0 else 0)
    binary.save(mask_path)
    bbox = binary.getbbox()
    assert bbox is not None
    mask_source.update(
        sha256=sha256_file(mask_path),
        bytes=mask_path.stat().st_size,
        area_pixels=sum(binary.histogram()[1:]),
        actual_bbox_xyxy=list(bbox),
    )
    for frame in report["frames"]:
        for assignment in frame["assignments"]:
            if assignment.get("detection_id") == detection_id:
                assignment["mask_sha256"] = mask_source["sha256"]
    for object_record in report["objects"]:
        for field in ("selected_evidence", "all_high_confidence_assignments"):
            for assignment in object_record[field]:
                if assignment.get("detection_id") == detection_id:
                    assignment["mask_sha256"] = mask_source["sha256"]


def rectangular_ring_mask(
    *,
    size: tuple[int, int],
    outer: tuple[int, int, int, int],
    inner: tuple[int, int, int, int],
    channel: tuple[int, int, int, int] | None = None,
) -> Image.Image:
    mask = Image.new("L", size, 0)
    for y in range(outer[1], outer[3]):
        for x in range(outer[0], outer[2]):
            mask.putpixel((x, y), 255)
    for y in range(inner[1], inner[3]):
        for x in range(inner[0], inner[2]):
            mask.putpixel((x, y), 0)
    if channel is not None:
        for y in range(channel[1], channel[3]):
            for x in range(channel[0], channel[2]):
                mask.putpixel((x, y), 0)
    return mask


def test_materializes_portable_inputs_by_detection_id_and_sorted_frame_id(
    tmp_path: Path,
) -> None:
    report, association, frames_dir = make_fixture(tmp_path)
    output = tmp_path / "round02_left"

    manifest = materialize_round(args_for(association, frames_dir, output))

    assert manifest["status"] == "ready_for_propainter"
    assert manifest["object_id"] == OBJECT_ID
    assert manifest["frame_ids"] == ["000001", "000002"]
    assert manifest["previous_clean_plate_source_set_sha256"] == report["source_set_sha256"]
    assert manifest["association"]["thresholds"] == report["thresholds"]
    assert manifest["association"]["sha256"] == sha256_file(association)
    assert manifest["portable_layout"]["frames_dir"] == "input/frames"
    assert manifest["frame_records"][0]["detection_id"] == "det-physical-a"
    assert manifest["frame_records"][1]["detection_id"] == "det-physical-b"
    first_mask = Image.open(output / "input" / "masks" / "0000.png").convert("L")
    second_mask = Image.open(output / "input" / "masks" / "0001.png").convert("L")
    assert first_mask.getbbox() == (1, 2, 4, 5)
    assert second_mask.getbbox() == (4, 1, 7, 4)
    assert manifest["mask_completion"]["policy"] == "observed"
    assert manifest["mask_completion"]["added_pixels"] == 0
    assert manifest["mask_completion"]["components"] == []
    first_record = manifest["frame_records"][0]
    assert first_record["source_mask"]["area_pixels"] == 9
    assert first_record["input_mask"]["area_pixels"] == 9
    assert first_record["mask_completion"]["observed_is_subset"] is True
    assert sha256_file(output / "provenance" / "association.json") == sha256_file(association)
    receipt = json.loads((output / "input_manifest.json.sha256.json").read_text(encoding="utf-8"))
    assert receipt["manifest_path"] == "input_manifest.json"
    assert receipt["manifest_sha256"] == sha256_file(output / "input_manifest.json")


def test_fill_enclosed_holes_materializes_completed_mask_and_audits_geometry(
    tmp_path: Path,
) -> None:
    report, association, frames_dir = make_fixture(tmp_path)
    observed = rectangular_ring_mask(
        size=(8, 6),
        outer=(1, 1, 7, 5),
        inner=(2, 2, 6, 4),
    )
    replace_detection_mask(report, detection_id="det-physical-a", mask=observed)
    rewrite_report(association, report)
    output = tmp_path / "filled-hole"
    args = args_for(association, frames_dir, output)
    args.mask_completion_policy = "fill-enclosed-holes"
    args.min_enclosed_hole_area_px = 2
    args.min_enclosed_hole_area_ratio = 0.0
    args.cavity_bridge_radius = 5

    manifest = materialize_round(args)

    record = manifest["frame_records"][0]
    assert record["source_mask"]["area_pixels"] == 16
    assert record["source_mask"]["bbox_xyxy_exclusive"] == [1, 1, 7, 5]
    assert record["input_mask"]["area_pixels"] == 24
    assert record["input_mask"]["bbox_xyxy_exclusive"] == [1, 1, 7, 5]
    completion = record["mask_completion"]
    assert completion["added_pixels"] == 8
    assert completion["added_ratio"] == 0.5
    assert completion["components"] == [
        {"area_pixels": 8, "bbox_xyxy_exclusive": [2, 2, 6, 4]}
    ]
    assert completion["bbox_xyxy_exclusive"] == [2, 2, 6, 4]
    assert completion["observed_geometry"]["area_pixels"] == 16
    assert completion["completed_geometry"]["area_pixels"] == 24
    assert manifest["mask_completion"]["added_pixels"] == 8
    assert manifest["mask_completion"]["component_count"] == 1
    assert manifest["mask_completion"]["bbox_xyxy_exclusive_by_frame"]["000001"] == [
        2,
        2,
        6,
        4,
    ]
    completed = Image.open(output / "input" / "masks" / "0000.png").convert("L")
    assert completed.getpixel((3, 3)) == 255


def test_fill_enclosed_holes_keeps_a_cavity_connected_to_border_open() -> None:
    observed = rectangular_ring_mask(
        size=(15, 15),
        outer=(2, 0, 13, 13),
        inner=(5, 4, 10, 10),
        channel=(7, 0, 8, 5),
    )

    completed, audit = complete_mask(
        observed,
        policy="fill-enclosed-holes",
        minimum_area_pixels=1,
        minimum_area_ratio=0.0,
        cavity_bridge_radius=5,
    )

    assert completed.tobytes() == observed.tobytes()
    assert audit["added_pixels"] == 0
    assert audit["components"] == []
    assert completed.getpixel((7, 7)) == 0


def test_bridge_fill_cavities_fills_narrow_mouth_but_never_copies_closing_contour() -> None:
    observed = rectangular_ring_mask(
        size=(25, 25),
        outer=(3, 0, 22, 22),
        inner=(7, 7, 18, 18),
        channel=(11, 0, 14, 8),
    )

    completed, audit = complete_mask(
        observed,
        policy="bridge-fill-cavities",
        minimum_area_pixels=1,
        minimum_area_ratio=0.0,
        cavity_bridge_radius=2,
    )

    assert audit["topology_method"] == "binary_closing_for_topology_only"
    assert audit["added_pixels"] > 0
    assert audit["component_count"] == 1
    assert completed.getpixel((8, 10)) == 255
    assert completed.getpixel((12, 2)) == 0
    assert completed.getpixel((0, 0)) == 0
    for observed_value, completed_value in zip(
        observed.tobytes(), completed.tobytes(), strict=True
    ):
        assert not observed_value or completed_value


def test_bridge_fill_cavities_does_not_fill_large_external_concavity() -> None:
    observed = rectangular_ring_mask(
        size=(25, 25),
        outer=(3, 0, 22, 22),
        inner=(7, 7, 18, 18),
        channel=(9, 0, 16, 8),
    )

    completed, audit = complete_mask(
        observed,
        policy="bridge-fill-cavities",
        minimum_area_pixels=1,
        minimum_area_ratio=0.0,
        cavity_bridge_radius=2,
    )

    assert completed.tobytes() == observed.tobytes()
    assert audit["added_pixels"] == 0
    assert audit["component_count"] == 0


def test_enclosed_hole_area_threshold_rejects_small_components() -> None:
    observed = Image.new("L", (12, 9), 255)
    observed.putpixel((2, 2), 0)
    for y in range(3, 5):
        for x in range(6, 8):
            observed.putpixel((x, y), 0)

    completed, audit = complete_mask(
        observed,
        policy="fill-enclosed-holes",
        minimum_area_pixels=2,
        minimum_area_ratio=0.0,
        cavity_bridge_radius=5,
    )

    assert completed.getpixel((2, 2)) == 0
    assert completed.getpixel((6, 3)) == 255
    assert audit["added_pixels"] == 4
    assert audit["rejected_below_threshold_component_count"] == 1


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(status="failed"), "status must be 'passed'"),
        (
            lambda value: value["gates"].update(passed=False),
            "association.gates.passed must be true",
        ),
        (
            lambda value: value["objects"][0].update(
                disposition="failed_insufficient_separated_evidence"
            ),
            "not an accepted_next_layer_target",
        ),
        (
            lambda value: value["association_contract"].update(
                physical_identity_source="mask filename"
            ),
            "not sourced from explicit 3D anchors",
        ),
    ],
)
def test_rejects_failed_association_or_object_gate(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    report, association, frames_dir = make_fixture(tmp_path)
    mutation(report)
    rewrite_report(association, report)
    output = tmp_path / "rejected"

    with pytest.raises(ValueError, match=message):
        materialize_round(args_for(association, frames_dir, output))

    assert not output.exists()


def test_missing_assignment_for_any_source_frame_fails_closed(tmp_path: Path) -> None:
    report, association, frames_dir = make_fixture(tmp_path)
    report["frames"][0]["assignments"] = []
    rewrite_report(association, report)
    output = tmp_path / "missing-assignment"

    with pytest.raises(ValueError, match="exactly one assignment"):
        materialize_round(args_for(association, frames_dir, output))

    assert not output.exists()


def test_rejects_tampered_frame_bytes(tmp_path: Path) -> None:
    _, association, frames_dir = make_fixture(tmp_path)
    Image.new("RGB", (8, 6), (255, 0, 0)).save(frames_dir / "000001.png")
    output = tmp_path / "tampered-frame"

    with pytest.raises(FileNotFoundError, match=r"no frame.*matches"):
        materialize_round(args_for(association, frames_dir, output))

    assert not output.exists()


def test_rejects_declared_frame_dimension_mismatch(tmp_path: Path) -> None:
    report, association, frames_dir = make_fixture(tmp_path)
    report["sources"]["frames"][0]["dimensions"] = [9, 6]
    rewrite_report(association, report)
    output = tmp_path / "wrong-dimensions"

    with pytest.raises(ValueError, match="dimensions do not match"):
        materialize_round(args_for(association, frames_dir, output))

    assert not output.exists()


def test_rejects_assignment_not_backed_by_explicit_anchor(tmp_path: Path) -> None:
    report, association, frames_dir = make_fixture(tmp_path)
    report["frames"][0]["assignments"].append(
        {
            "frame_id": "000002",
            "detection_id": "unanchored-detection",
            "object_id": "unknown_object",
        }
    )
    rewrite_report(association, report)
    output = tmp_path / "unanchored"

    with pytest.raises(ValueError, match="has no explicit 3D anchor"):
        materialize_round(args_for(association, frames_dir, output))

    assert not output.exists()
