from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from scripts import materialize_clean_plate_sam_semantic_run_receipt as run_producer
from scripts import qa_clean_plate_sam_semantics as semantic_producer

FRAME_IDS = tuple(f"{index:06d}" for index in range(48, 73))
HEIGHT = 6
WIDTH = 8
TARGET_ID = "physical_target"
REMAINING_ID = "physical_remaining"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_record(path: Path) -> dict[str, Any]:
    digest, size = run_producer.sha256_file(path)
    return {"path": str(path), "sha256": digest, "size_bytes": size}


def save_rgb(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value.astype(np.uint8)).save(path)


def save_mask(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value.astype(np.uint8) * 255).save(path)


def make_selection(root: Path) -> tuple[Path, dict[str, dict[str, Path]]]:
    records = []
    paths: dict[str, dict[str, Path]] = {}
    for sequence_index, frame_id in enumerate(FRAME_IDS):
        yy, xx = np.mgrid[:HEIGHT, :WIDTH]
        source = np.stack(
            (
                20 + xx * 3 + sequence_index,
                30 + yy * 4 + sequence_index,
                np.full_like(xx, 60 + sequence_index),
            ),
            axis=2,
        ).astype(np.uint8)
        core = np.zeros((HEIGHT, WIDTH), dtype=bool)
        core[2:4, 1:3] = True
        editable = np.zeros((HEIGHT, WIDTH), dtype=bool)
        editable[1:5, 0:4] = True
        composite = source.copy()
        composite[editable] = np.clip(composite[editable].astype(np.int16) + 10, 0, 255)
        source_path = root / "source" / f"{frame_id}.png"
        composite_path = root / "selection" / "frames" / f"{sequence_index:04d}.png"
        core_path = root / "selection" / "core" / f"{sequence_index:04d}.png"
        editable_path = root / "selection" / "editable" / f"{sequence_index:04d}.png"
        save_rgb(source_path, source)
        save_rgb(composite_path, composite)
        save_mask(core_path, core)
        save_mask(editable_path, editable)
        records.append(
            {
                "sequence_index": sequence_index,
                "frame_id": frame_id,
                "source_rgb": file_record(source_path),
                "outputs": {
                    "composite_rgb": file_record(composite_path),
                    "core_mask": file_record(core_path),
                    "editable_mask": file_record(editable_path),
                },
            }
        )
        paths[frame_id] = {
            "source": source_path,
            "composite": composite_path,
            "core": core_path,
            "editable": editable_path,
        }
    selection_path = root / "selection" / "selection_receipt.json"
    write_json(
        selection_path,
        {
            "schema_version": 1,
            "kind": run_producer.SELECTION_KIND,
            "status": run_producer.SELECTION_STATUS,
            "promotion_allowed": False,
            "ordered_frame_ids": list(FRAME_IDS),
            "frames": records,
        },
    )
    return selection_path, paths


def mask_for(object_id: str, *, overlap: bool = False) -> np.ndarray:
    value = np.zeros((HEIGHT, WIDTH), dtype=bool)
    if object_id == TARGET_ID:
        value[2:4, 1:3] = True
    elif object_id == REMAINING_ID:
        value[2:4, 5:7] = True
    else:
        value[1:3, 4:6] = True
        if overlap:
            value[2, 5] = True
    return value


def make_mask_index(
    root: Path,
    *,
    name: str,
    object_ids: tuple[str, ...],
    image_paths: dict[str, Path],
    residual_target: bool = False,
    overlapping_second_remaining: bool = False,
) -> tuple[Path, dict[str, list[dict[str, Any]]]]:
    items = []
    by_frame: dict[str, list[dict[str, Any]]] = {frame_id: [] for frame_id in FRAME_IDS}
    image_root = root / name / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    for frame_id in FRAME_IDS:
        shutil.copyfile(image_paths[frame_id], image_root / f"{frame_id}.png")
        frame_objects = list(object_ids)
        if residual_target and TARGET_ID not in frame_objects:
            frame_objects.append(TARGET_ID)
        for object_id in frame_objects:
            item_index = len(items)
            overlap = overlapping_second_remaining and object_id == "physical_remaining_two"
            mask = mask_for(object_id, overlap=overlap)
            mask_path = root / name / "masks" / frame_id / f"mask_{item_index:06d}.png"
            save_mask(mask_path, mask)
            rows, columns = np.nonzero(mask)
            item = {
                "image": f"{frame_id}.png",
                "label": "pillow",
                "mask_path": str(mask_path),
                "bbox": [
                    float(columns.min()),
                    float(rows.min()),
                    float(columns.max() + 1),
                    float(rows.max() + 1),
                ],
                "score": 0.95,
                "_object_id": object_id,
                "_item_index": item_index,
            }
            items.append({key: value for key, value in item.items() if not key.startswith("_")})
            by_frame[frame_id].append(item)
    index_path = root / name / "mask_index.json"
    write_json(
        index_path,
        {
            "scene": name,
            "image_root": str(image_root),
            "items": items,
            "missing_images": [],
        },
    )
    return index_path, by_frame


def make_legacy_receipt(
    root: Path,
    mask_index: Path,
    *,
    round_index: int = 1,
    target_object_id: str = TARGET_ID,
) -> Path:
    mask_sha, mask_size = run_producer.sha256_file(mask_index)
    path = root / "legacy_sam_run_receipt.json"
    write_json(
        path,
        {
            "schema_version": 1,
            "kind": run_producer.LEGACY_KIND,
            "status": run_producer.LEGACY_STATUS,
            "claim_scope": f"SAM3 reinspection after removing only {target_object_id}",
            "input_clean_plate": {"round": round_index, "frame_count": 25},
            "sources": {
                "runner": {
                    "path": "/audited/sam3_holi_runner.py",
                    "sha256": run_producer.EXPECTED_RUNNER_SHA256,
                },
                "checkpoint": {
                    "path": "/audited/sam3.pt",
                    "sha256": run_producer.EXPECTED_CHECKPOINT_SHA256,
                    "bytes": run_producer.EXPECTED_CHECKPOINT_BYTES,
                },
            },
            "outputs": {
                "mask_index": {
                    "path": str(mask_index),
                    "sha256": mask_sha,
                    "bytes": mask_size,
                    "frame_count": 25,
                    "missing_images": [],
                }
            },
        },
    )
    return path


def association_mask_record(item: dict[str, Any], frame_id: str) -> dict[str, Any]:
    path = Path(item["mask_path"])
    digest, size = run_producer.sha256_file(path)
    with Image.open(path) as image:
        pixels = int(np.count_nonzero(np.asarray(image.convert("L"), dtype=np.uint8)))
    return {
        "detection_id": f"{frame_id}:mask:{item['_item_index']:06d}",
        "item_index": item["_item_index"],
        "frame_id": frame_id,
        "label": item["label"],
        "score": item["score"],
        "resolved_path": str(path),
        "sha256": digest,
        "bytes": size,
        "area_pixels": pixels,
    }


def make_association(
    root: Path,
    *,
    name: str,
    index_path: Path,
    index_items: dict[str, list[dict[str, Any]]],
    frame_paths: dict[str, dict[str, Path]],
    source_key: str,
    assigned_object_ids: tuple[str, ...],
    peeled_target: bool,
    unassigned_target: bool = False,
    anchors: tuple[str, ...] = (TARGET_ID, REMAINING_ID),
) -> Path:
    source_frames = []
    source_masks = []
    frame_records = []
    for frame_id in FRAME_IDS:
        frame_path = frame_paths[frame_id][source_key]
        frame_sha, frame_size = run_producer.sha256_file(frame_path)
        source_frames.append(
            {
                "frame_id": frame_id,
                "resolved_path": str(frame_path),
                "sha256": frame_sha,
                "bytes": frame_size,
                "dimensions": [WIDTH, HEIGHT],
            }
        )
        items_by_object = {item["_object_id"]: item for item in index_items[frame_id]}
        assignments = []
        assigned_detection_ids = set()
        for object_id in assigned_object_ids:
            item = items_by_object[object_id]
            mask_record = association_mask_record(item, frame_id)
            detection_id = mask_record["detection_id"]
            assignments.append(
                {
                    "frame_id": frame_id,
                    "detection_id": detection_id,
                    "object_id": object_id,
                    "label": item["label"],
                    "frame_sha256": frame_sha,
                    "mask_sha256": mask_record["sha256"],
                    "metrics": {
                        "detection_id": detection_id,
                        "eligible": True,
                        "gates": {
                            "minimum_visible_projected_pixels": True,
                            "minimum_mask_hit_ratio": True,
                            "minimum_dilated_support_ratio": True,
                            "minimum_bbox_iou": True,
                            "minimum_sam3_score": True,
                        },
                    },
                }
            )
            assigned_detection_ids.add(detection_id)
        frame_mask_records = [
            association_mask_record(item, frame_id) for item in index_items[frame_id]
        ]
        source_masks.extend(frame_mask_records)
        all_detection_ids = {record["detection_id"] for record in frame_mask_records}
        unassigned = sorted(all_detection_ids - assigned_detection_ids)
        if not unassigned_target:
            assert not unassigned
        camera = {
            "extrinsic_type": "world_to_camera",
            "world_to_camera": np.eye(4).tolist(),
            "camera_center_world": [0.0, 0.0, 0.0],
            "rotation_orthonormal_error": 0.0,
            "intrinsic": {
                "fx": 8.0,
                "fy": 8.0,
                "cx": 4.0,
                "cy": 3.0,
                "w": 8.0,
                "h": 6.0,
            },
        }
        frame_records.append(
            {
                "frame_id": frame_id,
                "camera": camera,
                "assignments": assignments,
                "unassigned_detection_ids": unassigned,
            }
        )
    index_sha, index_size = run_producer.sha256_file(index_path)
    thresholds = {
        "minimum_visible_projected_pixels": 1,
        "minimum_mask_hit_ratio": 0.5,
        "minimum_dilated_support_ratio": 0.2,
        "minimum_bbox_iou": 0.1,
        "minimum_sam3_score": 0.5,
        "projection_support_dilation_pixels": 1,
        "occlusion_absolute_depth_tolerance": 0.02,
        "occlusion_relative_depth_tolerance": 0.005,
        "minimum_evidence_frames": 2,
        "minimum_camera_baseline_scene_units": 0.1,
        "minimum_object_view_angle_degrees": 4.0,
        "separation_mode": "both",
        "maximum_rotation_orthonormal_error": 0.001,
        "assignment_weights": {
            "mask_hit_ratio": 0.45,
            "dilated_support_ratio": 0.30,
            "bbox_iou": 0.20,
            "sam3_score": 0.05,
        },
    }
    association_asset_root = root / "association-assets"
    camera_path = association_asset_root / "camera_info.json"
    if not camera_path.exists():
        write_json(camera_path, {"camera_model": "PINHOLE", "frames": list(FRAME_IDS)})
    camera_sha, camera_size = run_producer.sha256_file(camera_path)
    anchor_records = []
    for object_id in anchors:
        anchor_path = association_asset_root / "anchors" / f"{object_id}.ply"
        if not anchor_path.exists():
            anchor_path.parent.mkdir(parents=True, exist_ok=True)
            anchor_path.write_text(
                f"ply\nformat ascii 1.0\ncomment {object_id}\nelement vertex 0\nend_header\n",
                encoding="utf-8",
            )
        anchor_sha, anchor_size = run_producer.sha256_file(anchor_path)
        anchor_records.append(
            {
                "object_id": object_id,
                "declared_path": str(anchor_path),
                "resolved_path": str(anchor_path),
                "sha256": anchor_sha,
                "bytes": anchor_size,
                "point_count": 10,
            }
        )
    association_script = Path(semantic_producer.__file__).with_name(
        "associate_multiframe_instance_masks.py"
    )
    association_script_sha, association_script_size = run_producer.sha256_file(association_script)
    sources = {
        "anchors": anchor_records,
        "camera_info": {
            "resolved_path": str(camera_path),
            "sha256": camera_sha,
            "bytes": camera_size,
            "extrinsic_type": "world_to_camera",
        },
        "sam3_mask_index": {
            "resolved_path": str(index_path),
            "sha256": index_sha,
            "bytes": index_size,
            "loaded_detection_count": len(source_masks),
        },
        "frames": source_frames,
        "masks": source_masks,
    }
    report_path = root / f"{name}_association.json"
    write_json(
        report_path,
        {
            "schema_version": 1,
            "status": "passed",
            "execution": {
                "script": {
                    "path": str(association_script),
                    "sha256": association_script_sha,
                    "bytes": association_script_size,
                }
            },
            "association_contract": {
                "physical_identity_source": semantic_producer.EXPLICIT_IDENTITY_SOURCE,
                "forbidden_identity_sources": sorted(semantic_producer.FORBIDDEN_IDENTITY_SOURCES),
            },
            "thresholds": thresholds,
            "thresholds_sha256": semantic_producer.sha256_json(thresholds),
            "sources": sources,
            "source_set_sha256": semantic_producer.sha256_json(sources),
            "peeled_object_ids": [TARGET_ID] if peeled_target else [],
            "frames": frame_records,
            "gates": {"passed": True, "failure_reasons": []},
        },
    )
    return report_path


def make_fixture(
    root: Path,
    *,
    residual_target: bool = False,
    missing_remaining: bool = False,
    overlapping_labels: bool = False,
) -> dict[str, Path]:
    selection, frame_paths = make_selection(root)
    baseline_objects = (TARGET_ID, REMAINING_ID)
    candidate_objects: tuple[str, ...] = (REMAINING_ID,)
    anchors: tuple[str, ...] = (TARGET_ID, REMAINING_ID)
    if overlapping_labels:
        baseline_objects += ("physical_remaining_two",)
        candidate_objects += ("physical_remaining_two",)
        anchors += ("physical_remaining_two",)
    baseline_index, baseline_items = make_mask_index(
        root,
        name="baseline",
        object_ids=baseline_objects,
        image_paths={frame_id: frame_paths[frame_id]["source"] for frame_id in FRAME_IDS},
        overlapping_second_remaining=overlapping_labels,
    )
    candidate_index, candidate_items = make_mask_index(
        root,
        name="candidate",
        object_ids=candidate_objects,
        image_paths={frame_id: frame_paths[frame_id]["composite"] for frame_id in FRAME_IDS},
        residual_target=residual_target,
        overlapping_second_remaining=overlapping_labels,
    )
    legacy = make_legacy_receipt(root, candidate_index)
    baseline_association = make_association(
        root,
        name="baseline",
        index_path=baseline_index,
        index_items=baseline_items,
        frame_paths=frame_paths,
        source_key="source",
        assigned_object_ids=baseline_objects,
        peeled_target=False,
        anchors=anchors,
    )
    assigned_candidate = () if missing_remaining else candidate_objects
    candidate_association = make_association(
        root,
        name="candidate",
        index_path=candidate_index,
        index_items=candidate_items,
        frame_paths=frame_paths,
        source_key="composite",
        assigned_object_ids=assigned_candidate,
        peeled_target=True,
        unassigned_target=residual_target or missing_remaining,
        anchors=anchors,
    )
    return {
        "selection": selection,
        "candidate_index": candidate_index,
        "legacy": legacy,
        "baseline_association": baseline_association,
        "candidate_association": candidate_association,
    }


def materialize_run(paths: dict[str, Path], output: Path) -> Path:
    receipt, _ = run_producer.materialize(
        selection_receipt=paths["selection"],
        legacy_sam_run_receipt=paths["legacy"],
        mask_index_path=paths["candidate_index"],
        round_index=1,
        target_object_id=TARGET_ID,
        output_dir=output,
    )
    assert receipt["status"] == run_producer.OUTPUT_STATUS
    return output / "sam_run_receipt.json"


def update_legacy_mask_index_binding(paths: dict[str, Path]) -> None:
    legacy = json.loads(paths["legacy"].read_text(encoding="utf-8"))
    digest, size = run_producer.sha256_file(paths["candidate_index"])
    legacy["outputs"]["mask_index"].update(sha256=digest, bytes=size)
    write_json(paths["legacy"], legacy)


def produce_semantics(paths: dict[str, Path], sam_run: Path, output: Path) -> dict[str, Any]:
    return semantic_producer.produce(
        selection_receipt=paths["selection"],
        sam_run_receipt=sam_run,
        baseline_association_path=paths["baseline_association"],
        candidate_association_path=paths["candidate_association"],
        round_index=1,
        target_object_id=TARGET_ID,
        output_dir=output,
    )


def test_materializes_strict_25_frame_sam_run_and_passing_semantic_qa(
    tmp_path: Path,
) -> None:
    paths = make_fixture(tmp_path)
    run_output = tmp_path / "strict-run"
    sam_run = materialize_run(paths, run_output)

    report = produce_semantics(paths, sam_run, tmp_path / "semantic-output")

    assert report["status"] == semantic_producer.PASS_STATUS
    assert report["frame_count"] == 25
    assert report["label_registry"] == {REMAINING_ID: 1}
    assert report["aggregate"]["all_target_object_absent"] is True
    assert report["aggregate"]["all_remaining_object_semantics_preserved"] is True
    assert len(list((run_output / "masks").glob("*/*.png"))) == 25
    sam_receipt = json.loads(sam_run.read_text(encoding="utf-8"))
    producer = run_output / sam_receipt["producer"]["script"]["path"]
    assert producer.is_file()
    assert run_producer.sha256_file(producer)[0] == semantic_producer.SAM_RUN_PRODUCER_SCRIPT_SHA256
    for frame in sam_receipt["frame_records"]:
        assert frame["actual_sam_input_sha256"] == frame["accepted_composite_sha256"]
        assert (run_output / frame["actual_sam_input"]["path"]).is_file()
        assert (run_output / frame["accepted_composite"]["path"]).is_file()
    assert len(list((tmp_path / "semantic-output" / "contributor_labels").glob("*.png"))) == 25


@pytest.mark.parametrize("mutation", ["old_source_bytes", "wrong_image_root", "missing_frame"])
def test_sam_run_rejects_unbound_or_missing_actual_inputs(
    tmp_path: Path,
    mutation: str,
) -> None:
    paths = make_fixture(tmp_path)
    index = json.loads(paths["candidate_index"].read_text(encoding="utf-8"))
    first_image = Path(index["image_root"]) / index["items"][0]["image"]
    if mutation == "old_source_bytes":
        selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
        shutil.copyfile(selection["frames"][0]["source_rgb"]["path"], first_image)
        expected = "differs from selected composite"
    elif mutation == "wrong_image_root":
        index["image_root"] = str(tmp_path / "wrong-image-root")
        write_json(paths["candidate_index"], index)
        update_legacy_mask_index_binding(paths)
        expected = "image_root does not exist"
    else:
        first_image.unlink()
        expected = "SAM input does not exist"

    with pytest.raises(run_producer.ContractError, match=expected):
        materialize_run(paths, tmp_path / "strict-run")


def test_sam_run_rejects_mask_index_declaring_missing_images(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    index = json.loads(paths["candidate_index"].read_text(encoding="utf-8"))
    index["missing_images"] = [index["items"][0]["image"]]
    write_json(paths["candidate_index"], index)
    update_legacy_mask_index_binding(paths)

    with pytest.raises(run_producer.ContractError, match="missing SAM input images"):
        materialize_run(paths, tmp_path / "strict-run")


@pytest.mark.parametrize("mutation", ["same_path", "same_sha"])
def test_sam_run_rejects_duplicate_mask_evidence_within_frame(
    tmp_path: Path,
    mutation: str,
) -> None:
    paths = make_fixture(tmp_path)
    index = json.loads(paths["candidate_index"].read_text(encoding="utf-8"))
    duplicate = dict(index["items"][0])
    if mutation == "same_sha":
        duplicate_path = tmp_path / "candidate" / "masks" / FRAME_IDS[0] / "duplicate.png"
        shutil.copyfile(duplicate["mask_path"], duplicate_path)
        duplicate["mask_path"] = str(duplicate_path)
        expected = "reuses mask bytes"
    else:
        expected = "reuses a mask path"
    index["items"].append(duplicate)
    write_json(paths["candidate_index"], index)
    update_legacy_mask_index_binding(paths)

    with pytest.raises(run_producer.ContractError, match=expected):
        materialize_run(paths, tmp_path / "strict-run")


def test_target_assignment_in_candidate_fails_immutable_report(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    run_output = tmp_path / "strict-run"
    sam_run = materialize_run(paths, run_output)
    candidate = json.loads(paths["candidate_association"].read_text(encoding="utf-8"))
    for frame in candidate["frames"]:
        frame["assignments"][0]["object_id"] = TARGET_ID
    write_json(paths["candidate_association"], candidate)

    report = produce_semantics(paths, sam_run, tmp_path / "semantic-output")

    assert report["status"] == semantic_producer.FAIL_STATUS
    assert report["sam_semantic_gate_passed"] is False
    assert (tmp_path / "semantic-output" / "semantic_qa_report.json").is_file()


def test_missing_remaining_object_fails_semantic_preservation(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path, missing_remaining=True)
    sam_run = materialize_run(paths, tmp_path / "strict-run")

    report = produce_semantics(paths, sam_run, tmp_path / "semantic-output")

    assert report["status"] == semantic_producer.FAIL_STATUS
    assert report["aggregate"]["all_remaining_object_semantics_preserved"] is False


def test_unassigned_same_label_core_residual_fails_target_absence(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path, residual_target=True)
    sam_run = materialize_run(paths, tmp_path / "strict-run")

    report = produce_semantics(paths, sam_run, tmp_path / "semantic-output")

    assert report["status"] == semantic_producer.FAIL_STATUS
    assert report["aggregate"]["all_unassigned_target_overlap_absent"] is False


def test_wrong_candidate_source_or_mask_hash_fails_closed(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path / "source")
    sam_run = materialize_run(paths, tmp_path / "source" / "strict-run")
    candidate = json.loads(paths["candidate_association"].read_text(encoding="utf-8"))
    candidate["sources"]["frames"][0]["sha256"] = "0" * 64
    candidate["source_set_sha256"] = semantic_producer.sha256_json(candidate["sources"])
    write_json(paths["candidate_association"], candidate)
    report = produce_semantics(paths, sam_run, tmp_path / "source" / "semantic-output")
    assert report["status"] == semantic_producer.FAIL_STATUS
    assert "source 000048 SHA mismatch" in report["error"]["message"]

    paths = make_fixture(tmp_path / "mask")
    sam_run = materialize_run(paths, tmp_path / "mask" / "strict-run")
    candidate = json.loads(paths["candidate_association"].read_text(encoding="utf-8"))
    candidate["sources"]["masks"][0]["sha256"] = "0" * 64
    candidate["source_set_sha256"] = semantic_producer.sha256_json(candidate["sources"])
    write_json(paths["candidate_association"], candidate)
    report = produce_semantics(paths, sam_run, tmp_path / "mask" / "semantic-output")
    assert report["status"] == semantic_producer.FAIL_STATUS
    assert "mask 000048:mask:000000 SHA mismatch" in report["error"]["message"]


@pytest.mark.parametrize("mutation", ["script_path", "script_bytes", "camera", "anchor"])
def test_semantic_qa_reopens_association_source_assets(
    tmp_path: Path,
    mutation: str,
) -> None:
    paths = make_fixture(tmp_path)
    sam_run = materialize_run(paths, tmp_path / "strict-run")
    candidate = json.loads(paths["candidate_association"].read_text(encoding="utf-8"))
    if mutation == "script_path":
        candidate["execution"]["script"]["path"] = str(tmp_path / "missing-script.py")
        expected = "producer script does not exist"
    elif mutation == "script_bytes":
        candidate["execution"]["script"].pop("bytes")
        expected = "producer script.bytes"
    elif mutation == "camera":
        camera_path = Path(candidate["sources"]["camera_info"]["resolved_path"])
        camera_path.write_text("tampered camera bytes\n", encoding="utf-8")
        expected = "camera_info SHA mismatch"
    else:
        anchor_path = Path(candidate["sources"]["anchors"][0]["resolved_path"])
        anchor_path.write_text("tampered anchor bytes\n", encoding="utf-8")
        expected = "anchor .* SHA mismatch"
    write_json(paths["candidate_association"], candidate)

    report = produce_semantics(paths, sam_run, tmp_path / "semantic-output")

    assert report["status"] == semantic_producer.FAIL_STATUS
    assert re.search(expected, report["error"]["message"])


def test_semantic_qa_supports_explicit_association_path_remaps(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    sam_run = materialize_run(paths, tmp_path / "strict-run")
    script_root = Path(semantic_producer.__file__).parent
    remote_data = Path("/remote/semantic-data")
    remote_scripts = Path("/remote/semantic-scripts")
    for key in ("baseline_association", "candidate_association"):
        association = json.loads(paths[key].read_text(encoding="utf-8"))
        script_path = Path(association["execution"]["script"]["path"])
        association["execution"]["script"]["path"] = str(
            remote_scripts / script_path.relative_to(script_root)
        )
        for anchor in association["sources"]["anchors"]:
            local_path = Path(anchor["resolved_path"])
            anchor["resolved_path"] = str(remote_data / local_path.relative_to(tmp_path))
        camera = association["sources"]["camera_info"]
        camera_path = Path(camera["resolved_path"])
        camera["resolved_path"] = str(remote_data / camera_path.relative_to(tmp_path))
        mask_index = association["sources"]["sam3_mask_index"]
        index_path = Path(mask_index["resolved_path"])
        mask_index["resolved_path"] = str(remote_data / index_path.relative_to(tmp_path))
        for source in association["sources"]["frames"]:
            source_path = Path(source["resolved_path"])
            source["resolved_path"] = str(remote_data / source_path.relative_to(tmp_path))
        for mask in association["sources"]["masks"]:
            mask_path = Path(mask["resolved_path"])
            mask["resolved_path"] = str(remote_data / mask_path.relative_to(tmp_path))
        association["source_set_sha256"] = semantic_producer.sha256_json(association["sources"])
        write_json(paths[key], association)

    report = semantic_producer.produce(
        selection_receipt=paths["selection"],
        sam_run_receipt=sam_run,
        baseline_association_path=paths["baseline_association"],
        candidate_association_path=paths["candidate_association"],
        round_index=1,
        target_object_id=TARGET_ID,
        output_dir=tmp_path / "semantic-output",
        association_path_remaps=[
            f"{remote_data}={tmp_path}",
            f"{remote_scripts}={script_root}",
        ],
    )

    assert report["status"] == semantic_producer.PASS_STATUS


def test_semantic_publish_failure_cleans_same_parent_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = make_fixture(tmp_path)
    sam_run = materialize_run(paths, tmp_path / "strict-run")

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("injected rename failure")

    monkeypatch.setattr(semantic_producer.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected rename failure"):
        produce_semantics(paths, sam_run, tmp_path / "semantic-output")

    assert not (tmp_path / "semantic-output").exists()
    assert not list(tmp_path.glob(".semantic-output.staging-*"))


def test_sam_run_publish_failure_cleans_same_parent_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = make_fixture(tmp_path)

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("injected SAM rename failure")

    monkeypatch.setattr(run_producer.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected SAM rename failure"):
        materialize_run(paths, tmp_path / "strict-run")

    assert not (tmp_path / "strict-run").exists()
    assert not list(tmp_path.glob(".strict-run.staging-*"))


def test_semantic_qa_rejects_tampered_portable_sam_producer_bytes(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    sam_run = materialize_run(paths, tmp_path / "strict-run")
    receipt = json.loads(sam_run.read_text(encoding="utf-8"))
    producer_path = sam_run.parent / receipt["producer"]["script"]["path"]
    producer_path.write_text("tampered producer bytes\n", encoding="utf-8")

    report = produce_semantics(paths, sam_run, tmp_path / "semantic-output")

    assert report["status"] == semantic_producer.FAIL_STATUS
    assert "producer.script SHA mismatch" in report["error"]["message"]


def test_semantic_qa_rechecks_portable_mask_index_missing_images(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    sam_run = materialize_run(paths, tmp_path / "strict-run")
    receipt = json.loads(sam_run.read_text(encoding="utf-8"))
    index_path = sam_run.parent / receipt["mask_index"]["path"]
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["missing_images"] = [index["items"][0]["image"]]
    write_json(index_path, index)
    digest, size = run_producer.sha256_file(index_path)
    for key in ("mask_index",):
        receipt[key].update(sha256=digest, size_bytes=size)
    receipt["evidence"]["mask_index"].update(sha256=digest, size_bytes=size)
    write_json(sam_run, receipt)

    report = produce_semantics(paths, sam_run, tmp_path / "semantic-output")

    assert report["status"] == semantic_producer.FAIL_STATUS
    assert "declares missing input images" in report["error"]["message"]


def test_semantic_qa_rejects_reused_baseline_mask_path_and_bytes(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    sam_run = materialize_run(paths, tmp_path / "strict-run")
    association = json.loads(paths["baseline_association"].read_text(encoding="utf-8"))
    first, second = association["sources"]["masks"][:2]
    second.update(
        resolved_path=first["resolved_path"],
        sha256=first["sha256"],
        bytes=first["bytes"],
        area_pixels=first["area_pixels"],
    )
    second_assignment = next(
        item
        for item in association["frames"][0]["assignments"]
        if item["detection_id"] == second["detection_id"]
    )
    second_assignment["mask_sha256"] = first["sha256"]
    association["source_set_sha256"] = semantic_producer.sha256_json(association["sources"])
    write_json(paths["baseline_association"], association)

    report = produce_semantics(paths, sam_run, tmp_path / "semantic-output")

    assert report["status"] == semantic_producer.FAIL_STATUS
    assert "reuses a mask path in frame 000048" in report["error"]["message"]


@pytest.mark.parametrize("mutation", ["filename_identity", "ordinal_identity"])
def test_filename_or_ordinal_cannot_supply_physical_identity(
    tmp_path: Path,
    mutation: str,
) -> None:
    paths = make_fixture(tmp_path)
    sam_run = materialize_run(paths, tmp_path / "strict-run")
    candidate = json.loads(paths["candidate_association"].read_text(encoding="utf-8"))
    assignment = candidate["frames"][0]["assignments"][0]
    if mutation == "filename_identity":
        assignment["object_id"] = "mask_000000"
    else:
        assignment["detection_id"] = f"{FRAME_IDS[0]}:mask:999999"
    write_json(paths["candidate_association"], candidate)

    report = produce_semantics(paths, sam_run, tmp_path / "semantic-output")

    assert report["status"] == semantic_producer.FAIL_STATUS


def test_overlapping_assigned_contributor_masks_fail_closed(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path, overlapping_labels=True)
    sam_run = materialize_run(paths, tmp_path / "strict-run")

    report = produce_semantics(paths, sam_run, tmp_path / "semantic-output")

    assert report["status"] == semantic_producer.FAIL_STATUS
    assert "assigned masks overlap" in report["error"]["message"]


def test_24_frame_mask_index_and_failed_receipts_do_not_pass(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path / "short")
    index = json.loads(paths["candidate_index"].read_text(encoding="utf-8"))
    index["items"] = [item for item in index["items"] if item["image"] != f"{FRAME_IDS[-1]}.png"]
    write_json(paths["candidate_index"], index)
    legacy = json.loads(paths["legacy"].read_text(encoding="utf-8"))
    digest, size = run_producer.sha256_file(paths["candidate_index"])
    legacy["outputs"]["mask_index"].update(sha256=digest, bytes=size)
    write_json(paths["legacy"], legacy)
    with pytest.raises(run_producer.ContractError, match="cover all 25"):
        materialize_run(paths, tmp_path / "short" / "strict-run")

    paths = make_fixture(tmp_path / "failed")
    legacy = json.loads(paths["legacy"].read_text(encoding="utf-8"))
    legacy["status"] = "failed"
    write_json(paths["legacy"], legacy)
    with pytest.raises(run_producer.ContractError, match="did not complete"):
        materialize_run(paths, tmp_path / "failed" / "strict-run")


def test_rejects_canonical_output_and_path_style_image_identity(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path / "canonical")
    with pytest.raises(run_producer.ContractError, match="canonical/live"):
        materialize_run(paths, tmp_path / "web" / "public" / "sam-run")

    paths = make_fixture(tmp_path / "image-path")
    index = json.loads(paths["candidate_index"].read_text(encoding="utf-8"))
    index["items"][0]["image"] = f"folder/{FRAME_IDS[0]}.png"
    write_json(paths["candidate_index"], index)
    legacy = json.loads(paths["legacy"].read_text(encoding="utf-8"))
    digest, size = run_producer.sha256_file(paths["candidate_index"])
    legacy["outputs"]["mask_index"].update(sha256=digest, bytes=size)
    write_json(paths["legacy"], legacy)
    with pytest.raises(run_producer.ContractError, match="filename"):
        materialize_run(paths, tmp_path / "image-path" / "strict-run")
