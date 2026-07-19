#!/usr/bin/env python3
"""Produce fail-closed SAM3 semantic QA for an accepted clean-plate candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    from materialize_clean_plate_sam_semantic_run_receipt import (
        OUTPUT_KIND as SAM_RUN_KIND,
    )
    from materialize_clean_plate_sam_semantic_run_receipt import (
        OUTPUT_STATUS as SAM_RUN_STATUS,
    )
    from materialize_clean_plate_sam_semantic_run_receipt import (
        Asset,
        SourceFrame,
        asset_record,
        canonical_sha256,
        load_selection,
        parse_path_remaps,
        require_dict,
        require_list,
        require_sha256,
        require_string,
        resolve_asset,
        resolve_remapped_path,
        sha256_file,
    )
except ModuleNotFoundError:  # Imported as scripts.qa_clean_plate_sam_semantics in tests.
    from scripts.materialize_clean_plate_sam_semantic_run_receipt import (
        OUTPUT_KIND as SAM_RUN_KIND,
    )
    from scripts.materialize_clean_plate_sam_semantic_run_receipt import (
        OUTPUT_STATUS as SAM_RUN_STATUS,
    )
    from scripts.materialize_clean_plate_sam_semantic_run_receipt import (
        Asset,
        SourceFrame,
        asset_record,
        canonical_sha256,
        load_selection,
        parse_path_remaps,
        require_dict,
        require_list,
        require_sha256,
        require_string,
        resolve_asset,
        resolve_remapped_path,
        sha256_file,
    )

REPORT_KIND = "video2world.clean_plate_sam_semantic_qa_report"
PASS_STATUS = "technical_passed_semantic_only"
FAIL_STATUS = "technical_failed_semantic_only"
ASSOCIATION_SCRIPT_SHA256 = "4a12fa9834aa58e6f31d3bc5ded731717d6c6c7e3aa18da5c0cb57cf09f50171"
SAM_RUN_PRODUCER_SCRIPT_SHA256 = "9ca549becab2db94913909055aac84089e471901ec78e31934a1c7470276dcce"
EXPECTED_FRAME_COUNT = 25
MAX_UNASSIGNED_TARGET_LABEL_CORE_OVERLAP_FRACTION = 0.05
EXPLICIT_IDENTITY_SOURCE = "explicit --anchor object_id=PLY only"
FORBIDDEN_IDENTITY_SOURCES = {
    "SAM3 mask filename",
    "SAM3 per-frame ordinal suffix",
    "cross-frame item order",
}
SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class SemanticQAError(ValueError):
    """Semantic evidence is incomplete or violates the physical identity contract."""


@dataclass(frozen=True)
class MaskEvidence:
    detection_id: str
    item_index: int
    frame_id: str
    label: str
    score: float
    asset: Asset
    values: np.ndarray


@dataclass(frozen=True)
class AssociationFrame:
    frame_id: str
    source_rgb: Asset
    assignments: dict[str, dict[str, Any]]
    unassigned_detection_ids: tuple[str, ...]
    camera_sha256: str


@dataclass(frozen=True)
class Association:
    asset: Asset
    value: dict[str, Any]
    anchors: dict[str, str]
    anchor_assets: dict[str, Asset]
    producer_script: Asset
    camera_info: Asset
    camera_info_sha256: str
    thresholds_sha256: str
    mask_index_sha256: str
    masks: dict[str, MaskEvidence]
    frames: tuple[AssociationFrame, ...]


def local_require(condition: bool, message: str) -> None:
    if not condition:
        raise SemanticQAError(message)


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_association_asset(
    record: dict[str, Any],
    *,
    path_key: str,
    relative_to: Path,
    remaps: list[tuple[Path, Path]],
    label: str,
) -> Asset:
    path = resolve_remapped_path(
        require_string(record.get(path_key), f"{label}.{path_key}"),
        relative_to=relative_to,
        remaps=remaps,
    )
    local_require(path.is_file(), f"{label} does not exist: {path}")
    expected_sha = require_sha256(record.get("sha256"), f"{label}.sha256")
    declared_size = record.get("bytes", record.get("size_bytes"))
    local_require(
        isinstance(declared_size, int)
        and not isinstance(declared_size, bool)
        and declared_size >= 0,
        f"{label}.bytes must be a non-negative integer",
    )
    actual_sha, actual_size = sha256_file(path)
    local_require(actual_sha == expected_sha, f"{label} SHA mismatch")
    local_require(actual_size == declared_size, f"{label} byte size mismatch")
    return Asset(path, actual_sha, actual_size)


def load_binary_mask(asset: Asset, *, expected_shape: tuple[int, int], label: str) -> np.ndarray:
    try:
        with Image.open(asset.path) as image:
            values = np.asarray(image.convert("L"), dtype=np.uint8)
    except OSError as exc:
        raise SemanticQAError(f"cannot decode {label}: {asset.path}") from exc
    local_require(values.shape == expected_shape, f"{label} dimensions differ from source")
    mask = values > 0
    local_require(bool(mask.any()), f"{label} is empty")
    return mask


def load_strict_sam_run(
    path: Path,
    *,
    selection: tuple[SourceFrame, ...],
    round_index: int,
    target_object_id: str,
) -> tuple[Asset, dict[str, Any], dict[str, MaskEvidence]]:
    run_path = path.expanduser().resolve()
    local_require(run_path.is_file(), f"strict SAM run receipt does not exist: {run_path}")
    digest, size = sha256_file(run_path)
    asset = Asset(run_path, digest, size)
    value = json.loads(run_path.read_text(encoding="utf-8"))
    local_require(isinstance(value, dict), "strict SAM run receipt must be an object")
    local_require(value.get("schema_version") == 1, "strict SAM run schema is invalid")
    local_require(value.get("kind") == SAM_RUN_KIND, "strict SAM run kind is invalid")
    local_require(value.get("status") == SAM_RUN_STATUS, "strict SAM run status is not passed")
    local_require(value.get("review_only") is True, "strict SAM run must be review-only")
    local_require(value.get("promotion_approved") is False, "strict SAM run exceeded authority")
    local_require(value.get("round_index") == round_index, "strict SAM run round mismatch")
    local_require(
        value.get("target_object_id") == target_object_id,
        "strict SAM run target_object_id mismatch",
    )
    local_require(value.get("frame_count") == EXPECTED_FRAME_COUNT, "strict SAM run != 25 frames")
    ordered = value.get("ordered_frame_ids")
    expected_ids = [frame.frame_id for frame in selection]
    local_require(ordered == expected_ids, "strict SAM run frame order differs from selection")
    producer = require_dict(value.get("producer"), "strict SAM run producer")
    script = require_dict(producer.get("script"), "strict SAM run producer.script")
    producer_script = resolve_association_asset(
        script,
        path_key="path",
        relative_to=run_path.parent,
        remaps=[],
        label="strict SAM run producer.script",
    )
    local_require(
        producer_script.path.is_relative_to(run_path.parent),
        "strict SAM run producer script escapes the run package",
    )
    local_require(
        producer_script.sha256 == SAM_RUN_PRODUCER_SCRIPT_SHA256,
        "strict SAM run producer script SHA-256 is not audited",
    )
    mask_index = resolve_association_asset(
        require_dict(value.get("mask_index"), "strict SAM run.mask_index"),
        path_key="path",
        relative_to=run_path.parent,
        remaps=[],
        label="strict SAM run.mask_index",
    )
    local_require(mask_index.path.is_relative_to(run_path.parent), "strict SAM mask index escapes")
    mask_index_value = json.loads(mask_index.path.read_text(encoding="utf-8"))
    local_require(isinstance(mask_index_value, dict), "strict SAM mask index must be an object")
    local_require(
        mask_index_value.get("missing_images") == [],
        "strict SAM mask index declares missing input images",
    )
    records = require_list(value.get("frame_records"), "strict SAM run.frame_records")
    local_require(len(records) == EXPECTED_FRAME_COUNT, "strict SAM run needs 25 frame records")
    detections: dict[str, MaskEvidence] = {}
    detection_set: list[dict[str, Any]] = []
    frame_set: list[dict[str, Any]] = []
    for frame, raw_record in zip(selection, records, strict=True):
        record = require_dict(raw_record, f"strict SAM frame {frame.frame_id}")
        local_require(record.get("sequence_index") == frame.sequence_index, "SAM sequence mismatch")
        local_require(record.get("frame_id") == frame.frame_id, "SAM frame order mismatch")
        local_require(
            record.get("source_role") == "actual_sam_input_equals_accepted_clean_plate_composite",
            "strict SAM source role is invalid",
        )
        actual_sam_input = resolve_association_asset(
            require_dict(record.get("actual_sam_input"), "strict SAM actual_sam_input"),
            path_key="path",
            relative_to=run_path.parent,
            remaps=[],
            label=f"strict SAM {frame.frame_id}.actual_sam_input",
        )
        accepted_composite = resolve_association_asset(
            require_dict(record.get("accepted_composite"), "strict SAM accepted_composite"),
            path_key="path",
            relative_to=run_path.parent,
            remaps=[],
            label=f"strict SAM {frame.frame_id}.accepted_composite",
        )
        source = resolve_association_asset(
            require_dict(record.get("source_rgb"), "strict SAM source_rgb"),
            path_key="path",
            relative_to=run_path.parent,
            remaps=[],
            label=f"strict SAM {frame.frame_id}.source_rgb",
        )
        for source_asset, label in (
            (actual_sam_input, "actual_sam_input"),
            (accepted_composite, "accepted_composite"),
            (source, "source_rgb"),
        ):
            local_require(
                source_asset.path.is_relative_to(run_path.parent),
                f"strict SAM {frame.frame_id}.{label} escapes the run package",
            )
        local_require(
            actual_sam_input.sha256
            == accepted_composite.sha256
            == source.sha256
            == frame.composite.sha256,
            "strict SAM actual input/composite source binding mismatch",
        )
        local_require(
            record.get("actual_sam_input_sha256") == actual_sam_input.sha256,
            "SAM actual input hash mismatch",
        )
        local_require(
            record.get("accepted_composite_sha256") == accepted_composite.sha256,
            "SAM accepted composite hash mismatch",
        )
        local_require(record.get("source_rgb_sha256") == source.sha256, "SAM source hash mismatch")
        frame_set.append(
            {
                "sequence_index": frame.sequence_index,
                "frame_id": frame.frame_id,
                "actual_sam_input_sha256": actual_sam_input.sha256,
                "accepted_composite_sha256": accepted_composite.sha256,
            }
        )
        frame_detections = require_list(
            record.get("detections"), f"SAM {frame.frame_id}.detections"
        )
        local_require(
            bool(frame_detections), f"strict SAM frame {frame.frame_id} has no detections"
        )
        frame_mask_paths: set[Path] = set()
        frame_mask_sha256: set[str] = set()
        for raw_detection in frame_detections:
            detection = require_dict(raw_detection, f"strict SAM detection {frame.frame_id}")
            item_index = detection.get("item_index")
            local_require(
                isinstance(item_index, int)
                and not isinstance(item_index, bool)
                and item_index >= 0,
                "strict SAM item_index is invalid",
            )
            expected_detection_id = f"{frame.frame_id}:mask:{item_index:06d}"
            local_require(
                detection.get("detection_id") == expected_detection_id,
                "strict SAM detection identity differs from frame/item index",
            )
            local_require(expected_detection_id not in detections, "duplicate strict SAM detection")
            label = require_string(detection.get("label"), "strict SAM detection.label")
            score = detection.get("score")
            local_require(
                isinstance(score, int | float)
                and not isinstance(score, bool)
                and math.isfinite(float(score)),
                "strict SAM detection score is invalid",
            )
            mask = resolve_association_asset(
                require_dict(detection.get("mask"), "strict SAM mask"),
                path_key="path",
                relative_to=run_path.parent,
                remaps=[],
                label=f"strict SAM {expected_detection_id}.mask",
            )
            local_require(mask.path.is_relative_to(run_path.parent), "SAM mask escapes run package")
            local_require(mask.path not in frame_mask_paths, "duplicate strict SAM mask path")
            local_require(mask.sha256 not in frame_mask_sha256, "duplicate strict SAM mask bytes")
            frame_mask_paths.add(mask.path)
            frame_mask_sha256.add(mask.sha256)
            local_require(mask.sha256 == detection.get("mask_sha256"), "SAM mask hash mismatch")
            local_require(
                detection.get("source_rgb_sha256") == source.sha256,
                "SAM detection is not source-bound",
            )
            local_require(
                detection.get("actual_sam_input_sha256") == actual_sam_input.sha256,
                "SAM detection is not actual-input-bound",
            )
            local_require(
                detection.get("accepted_composite_sha256") == accepted_composite.sha256,
                "SAM detection is not accepted-composite-bound",
            )
            values = load_binary_mask(
                mask,
                expected_shape=(frame.height, frame.width),
                label=f"strict SAM {expected_detection_id}.mask",
            )
            evidence = MaskEvidence(
                detection_id=expected_detection_id,
                item_index=item_index,
                frame_id=frame.frame_id,
                label=label,
                score=float(score),
                asset=mask,
                values=values,
            )
            detections[expected_detection_id] = evidence
            detection_set.append(
                {
                    "item_index": item_index,
                    "detection_id": expected_detection_id,
                    "frame_id": frame.frame_id,
                    "label": label,
                    "source_rgb_sha256": source.sha256,
                    "actual_sam_input_sha256": actual_sam_input.sha256,
                    "accepted_composite_sha256": accepted_composite.sha256,
                    "mask_sha256": mask.sha256,
                }
            )
    aggregate = require_dict(value.get("aggregate"), "strict SAM run.aggregate")
    for key in (
        "all_source_frames_hash_bound",
        "all_actual_sam_inputs_hash_bound",
        "all_actual_sam_inputs_match_accepted_composites",
        "all_detection_masks_hash_verified",
        "all_detection_masks_source_bound",
        "all_25_frames_have_detections",
    ):
        local_require(aggregate.get(key) is True, f"strict SAM aggregate.{key} is not true")
    local_require(
        aggregate.get("source_frame_set_sha256") == canonical_sha256(frame_set),
        "strict SAM source frame set digest mismatch",
    )
    local_require(
        aggregate.get("detection_set_sha256") == canonical_sha256(detection_set),
        "strict SAM detection set digest mismatch",
    )
    local_require(
        aggregate.get("detection_count") == len(detections),
        "strict SAM detection_count mismatch",
    )
    return asset, {**value, "_mask_index_sha256": mask_index.sha256}, detections


def validate_association(
    path: Path,
    *,
    selection: tuple[SourceFrame, ...],
    source_role: str,
    strict_detections: dict[str, MaskEvidence] | None,
    strict_mask_index_sha256: str | None,
    path_remaps: list[tuple[Path, Path]],
) -> Association:
    report_path = path.expanduser().resolve()
    local_require(report_path.is_file(), f"{source_role} association does not exist: {report_path}")
    digest, size = sha256_file(report_path)
    asset = Asset(report_path, digest, size)
    value = json.loads(report_path.read_text(encoding="utf-8"))
    local_require(isinstance(value, dict), f"{source_role} association must be an object")
    local_require(value.get("schema_version") == 1, f"{source_role} association schema invalid")
    local_require(value.get("status") == "passed", f"{source_role} association status failed")
    execution = require_dict(value.get("execution"), f"{source_role} association.execution")
    script = require_dict(execution.get("script"), f"{source_role} association script")
    local_require(
        require_sha256(script.get("sha256"), f"{source_role} association script.sha256")
        == ASSOCIATION_SCRIPT_SHA256,
        f"{source_role} association producer script SHA-256 is not audited",
    )
    producer_script = resolve_association_asset(
        script,
        path_key="path",
        relative_to=report_path.parent,
        remaps=path_remaps,
        label=f"{source_role} association producer script",
    )
    contract = require_dict(
        value.get("association_contract"), f"{source_role} association contract"
    )
    local_require(
        contract.get("physical_identity_source") == EXPLICIT_IDENTITY_SOURCE,
        f"{source_role} association does not use explicit anchors",
    )
    local_require(
        set(require_list(contract.get("forbidden_identity_sources"), "forbidden identities"))
        == FORBIDDEN_IDENTITY_SOURCES,
        f"{source_role} association forbidden identity contract differs",
    )
    gates = require_dict(value.get("gates"), f"{source_role} association.gates")
    local_require(gates.get("passed") is True, f"{source_role} association gates failed")
    thresholds = require_dict(value.get("thresholds"), f"{source_role} thresholds")
    thresholds_sha = sha256_json(thresholds)
    local_require(
        value.get("thresholds_sha256") == thresholds_sha,
        f"{source_role} thresholds digest mismatch",
    )
    sources = require_dict(value.get("sources"), f"{source_role} association.sources")
    local_require(
        value.get("source_set_sha256") == sha256_json(sources),
        f"{source_role} source set digest mismatch",
    )
    anchors: dict[str, str] = {}
    anchor_assets: dict[str, Asset] = {}
    for raw_anchor in require_list(sources.get("anchors"), f"{source_role} anchors"):
        anchor = require_dict(raw_anchor, f"{source_role} anchor")
        object_id = require_string(anchor.get("object_id"), f"{source_role} anchor.object_id")
        local_require(SAFE_ID_PATTERN.fullmatch(object_id) is not None, "unsafe anchor object ID")
        local_require(object_id not in anchors, f"duplicate anchor {object_id}")
        anchor_asset = resolve_association_asset(
            anchor,
            path_key="resolved_path",
            relative_to=report_path.parent,
            remaps=path_remaps,
            label=f"{source_role} anchor {object_id}",
        )
        anchors[object_id] = anchor_asset.sha256
        anchor_assets[object_id] = anchor_asset
    local_require(bool(anchors), f"{source_role} association has no anchors")
    camera = require_dict(sources.get("camera_info"), f"{source_role} camera_info")
    camera_info = resolve_association_asset(
        camera,
        path_key="resolved_path",
        relative_to=report_path.parent,
        remaps=path_remaps,
        label=f"{source_role} camera_info",
    )
    camera_sha = camera_info.sha256
    local_require(
        camera.get("extrinsic_type") == "world_to_camera",
        f"{source_role} camera convention is invalid",
    )
    mask_index = require_dict(sources.get("sam3_mask_index"), f"{source_role} mask index")
    mask_index_asset = resolve_association_asset(
        mask_index,
        path_key="resolved_path",
        relative_to=report_path.parent,
        remaps=path_remaps,
        label=f"{source_role} mask index",
    )
    mask_index_sha = mask_index_asset.sha256
    if strict_mask_index_sha256 is not None:
        local_require(
            mask_index_sha == strict_mask_index_sha256,
            "candidate association uses another SAM mask index",
        )

    source_records = require_list(sources.get("frames"), f"{source_role} source frames")
    local_require(len(source_records) == EXPECTED_FRAME_COUNT, f"{source_role} needs 25 sources")
    source_by_id: dict[str, Asset] = {}
    dimensions_by_id = {frame.frame_id: (frame.height, frame.width) for frame in selection}
    for raw_source in source_records:
        record = require_dict(raw_source, f"{source_role} source frame")
        frame_id = require_string(record.get("frame_id"), f"{source_role} source.frame_id")
        local_require(frame_id in dimensions_by_id, f"{source_role} source has unknown frame")
        local_require(frame_id not in source_by_id, f"{source_role} source repeats {frame_id}")
        source_asset = resolve_association_asset(
            record,
            path_key="resolved_path",
            relative_to=report_path.parent,
            remaps=path_remaps,
            label=f"{source_role} source {frame_id}",
        )
        expected_dimensions = dimensions_by_id[frame_id]
        local_require(
            record.get("dimensions") == [expected_dimensions[1], expected_dimensions[0]],
            f"{source_role} source dimensions mismatch",
        )
        source_by_id[frame_id] = source_asset
    local_require(set(source_by_id) == set(dimensions_by_id), f"{source_role} sources != 25 IDs")

    masks: dict[str, MaskEvidence] = {}
    masks_by_frame: dict[str, set[str]] = {frame.frame_id: set() for frame in selection}
    mask_paths_by_frame: dict[str, set[Path]] = {frame.frame_id: set() for frame in selection}
    mask_sha256_by_frame: dict[str, set[str]] = {frame.frame_id: set() for frame in selection}
    for raw_mask in require_list(sources.get("masks"), f"{source_role} source masks"):
        record = require_dict(raw_mask, f"{source_role} source mask")
        item_index = record.get("item_index")
        local_require(
            isinstance(item_index, int) and not isinstance(item_index, bool) and item_index >= 0,
            f"{source_role} mask item_index is invalid",
        )
        frame_id = require_string(record.get("frame_id"), f"{source_role} mask.frame_id")
        detection_id = require_string(
            record.get("detection_id"), f"{source_role} mask.detection_id"
        )
        local_require(
            detection_id == f"{frame_id}:mask:{item_index:06d}",
            f"{source_role} detection ID is not frame/item-index exact",
        )
        local_require(detection_id not in masks, f"{source_role} duplicate detection ID")
        local_require(frame_id in source_by_id, f"{source_role} mask has unknown frame")
        mask_asset = resolve_association_asset(
            record,
            path_key="resolved_path",
            relative_to=report_path.parent,
            remaps=path_remaps,
            label=f"{source_role} mask {detection_id}",
        )
        local_require(
            mask_asset.path not in mask_paths_by_frame[frame_id],
            f"{source_role} reuses a mask path in frame {frame_id}",
        )
        local_require(
            mask_asset.sha256 not in mask_sha256_by_frame[frame_id],
            f"{source_role} reuses mask bytes in frame {frame_id}",
        )
        mask_paths_by_frame[frame_id].add(mask_asset.path)
        mask_sha256_by_frame[frame_id].add(mask_asset.sha256)
        frame_shape = dimensions_by_id[frame_id]
        values = load_binary_mask(
            mask_asset,
            expected_shape=frame_shape,
            label=f"{source_role} {detection_id}",
        )
        local_require(
            int(values.sum()) == record.get("area_pixels"),
            f"{source_role} mask area mismatch",
        )
        label = require_string(record.get("label"), f"{source_role} mask.label")
        score = record.get("score")
        local_require(
            isinstance(score, int | float)
            and not isinstance(score, bool)
            and math.isfinite(float(score)),
            f"{source_role} mask score is invalid",
        )
        evidence = MaskEvidence(
            detection_id=detection_id,
            item_index=item_index,
            frame_id=frame_id,
            label=label,
            score=float(score),
            asset=mask_asset,
            values=values,
        )
        masks[detection_id] = evidence
        masks_by_frame[frame_id].add(detection_id)
    local_require(
        mask_index.get("loaded_detection_count") == len(masks),
        f"{source_role} mask index loaded_detection_count mismatch",
    )
    if strict_detections is not None:
        local_require(
            set(masks) == set(strict_detections),
            "candidate association detection set differs",
        )
        for detection_id, mask in masks.items():
            strict = strict_detections[detection_id]
            local_require(
                (mask.asset.sha256, mask.frame_id, mask.label)
                == (strict.asset.sha256, strict.frame_id, strict.label),
                f"candidate association mask differs from strict SAM run: {detection_id}",
            )

    frame_values = require_list(value.get("frames"), f"{source_role} association.frames")
    local_require(
        len(frame_values) == EXPECTED_FRAME_COUNT,
        f"{source_role} needs 25 frame reports",
    )
    frame_map = {
        record.get("frame_id"): record for record in frame_values if isinstance(record, dict)
    }
    local_require(set(frame_map) == set(source_by_id), f"{source_role} frame reports != sources")
    frames: list[AssociationFrame] = []
    for selected in selection:
        frame_id = selected.frame_id
        record = require_dict(frame_map.get(frame_id), f"{source_role} frame {frame_id}")
        camera_record = require_dict(record.get("camera"), f"{source_role} frame.camera")
        local_require(
            camera_record.get("extrinsic_type") == "world_to_camera",
            f"{source_role} frame camera convention invalid",
        )
        camera_digest = sha256_json(camera_record)
        assignments: dict[str, dict[str, Any]] = {}
        assigned_detection_ids: set[str] = set()
        for raw_assignment in require_list(record.get("assignments"), "frame.assignments"):
            assignment = require_dict(raw_assignment, f"{source_role} assignment")
            object_id = require_string(assignment.get("object_id"), "assignment.object_id")
            local_require(object_id in anchors, f"assignment object {object_id} has no anchor")
            local_require(object_id not in assignments, f"duplicate assignment for {object_id}")
            detection_id = require_string(assignment.get("detection_id"), "assignment.detection_id")
            local_require(
                detection_id in masks_by_frame[frame_id],
                "assignment detection frame mismatch",
            )
            local_require(
                detection_id not in assigned_detection_ids,
                "detection reused across objects",
            )
            mask = masks[detection_id]
            local_require(assignment.get("frame_id") == frame_id, "assignment frame_id mismatch")
            local_require(
                assignment.get("frame_sha256") == source_by_id[frame_id].sha256,
                "assignment is bound to another source RGB",
            )
            local_require(
                assignment.get("mask_sha256") == mask.asset.sha256,
                "assignment mask mismatch",
            )
            local_require(assignment.get("label") == mask.label, "assignment label mismatch")
            metrics = require_dict(assignment.get("metrics"), "assignment.metrics")
            local_require(metrics.get("detection_id") == detection_id, "metric detection mismatch")
            local_require(metrics.get("eligible") is True, "assignment metric is not eligible")
            metric_gates = require_dict(metrics.get("gates"), "assignment.metrics.gates")
            local_require(
                bool(metric_gates) and all(value is True for value in metric_gates.values()),
                "assignment metric gates did not pass",
            )
            assignments[object_id] = assignment
            assigned_detection_ids.add(detection_id)
        raw_unassigned = require_list(record.get("unassigned_detection_ids"), "unassigned IDs")
        unassigned = tuple(
            require_string(item, "unassigned detection ID") for item in raw_unassigned
        )
        local_require(len(set(unassigned)) == len(unassigned), "duplicate unassigned detection ID")
        local_require(
            set(unassigned).isdisjoint(assigned_detection_ids),
            "detection is both assigned and unassigned",
        )
        local_require(
            set(unassigned) | assigned_detection_ids == masks_by_frame[frame_id],
            "assigned/unassigned detections do not close to source masks",
        )
        frames.append(
            AssociationFrame(
                frame_id=frame_id,
                source_rgb=source_by_id[frame_id],
                assignments=assignments,
                unassigned_detection_ids=unassigned,
                camera_sha256=camera_digest,
            )
        )
    return Association(
        asset=asset,
        value=value,
        anchors=anchors,
        anchor_assets=anchor_assets,
        producer_script=producer_script,
        camera_info=camera_info,
        camera_info_sha256=camera_sha,
        thresholds_sha256=thresholds_sha,
        mask_index_sha256=mask_index_sha,
        masks=masks,
        frames=tuple(frames),
    )


def validate_source_bindings(
    *,
    selection_asset: Asset,
    selection_value: dict[str, Any],
    selection: tuple[SourceFrame, ...],
    baseline: Association,
    candidate: Association,
) -> None:
    records = require_list(selection_value.get("frames"), "selection.frames")
    for frame, raw_record, baseline_frame, candidate_frame in zip(
        selection, records, baseline.frames, candidate.frames, strict=True
    ):
        record = require_dict(raw_record, f"selection {frame.frame_id}")
        source = resolve_asset(
            record.get("source_rgb"),
            relative_to=selection_asset.path.parent,
            label=f"selection {frame.frame_id}.source_rgb",
        )
        local_require(
            baseline_frame.source_rgb.sha256 == source.sha256,
            f"baseline association source mismatch for {frame.frame_id}",
        )
        local_require(
            candidate_frame.source_rgb.sha256 == frame.composite.sha256,
            f"candidate association source mismatch for {frame.frame_id}",
        )


def prepare_destination(value: Path) -> Path:
    lexical = value.expanduser()
    local_require(not lexical.is_symlink(), f"output directory cannot be a symlink: {lexical}")
    destination = lexical.resolve()
    local_require(
        not destination.exists(), f"refusing to overwrite output directory: {destination}"
    )
    forbidden = {("web", "public"), ("public", "worlds")}
    local_require(
        not any(pair in forbidden for pair in pairwise(destination.parts)),
        "output directory may not target canonical/live web assets",
    )
    return destination


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def produce(
    *,
    selection_receipt: Path,
    sam_run_receipt: Path,
    baseline_association_path: Path,
    candidate_association_path: Path,
    round_index: int,
    target_object_id: str,
    output_dir: Path,
    association_path_remaps: list[str] | None = None,
) -> dict[str, Any]:
    destination = prepare_destination(output_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve()
    script_sha, script_size = sha256_file(script_path)
    producer_script_record = asset_record(
        Asset(script_path, script_sha, script_size),
        path="evidence/producer_script.py",
    )
    timestamp = datetime.now(timezone.utc).isoformat()  # noqa: UP017
    base_report: dict[str, Any] = {
        "schema_version": 1,
        "kind": REPORT_KIND,
        "status": FAIL_STATUS,
        "created_at": timestamp,
        "review_only": True,
        "promotion_approved": False,
        "round_index": round_index,
        "target_object_id": target_object_id,
        "producer": {"script": producer_script_record},
        "canonical_or_live_manifest_modified": False,
    }
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    report = dict(base_report)
    try:
        producer_copy = staging / producer_script_record["path"]
        producer_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(script_path, producer_copy)
        copied_sha, copied_size = sha256_file(producer_copy)
        local_require(
            (copied_sha, copied_size) == (script_sha, script_size),
            "semantic QA producer copy changed bytes",
        )
        local_require(round_index > 0, "round_index must be positive")
        local_require(
            SAFE_ID_PATTERN.fullmatch(target_object_id) is not None,
            "target_object_id is unsafe",
        )
        path_remaps = parse_path_remaps(association_path_remaps or [])
        selection_asset, selection = load_selection(selection_receipt)
        selection_value = json.loads(selection_asset.path.read_text(encoding="utf-8"))
        sam_asset, sam_value, strict_detections = load_strict_sam_run(
            sam_run_receipt,
            selection=selection,
            round_index=round_index,
            target_object_id=target_object_id,
        )
        baseline = validate_association(
            baseline_association_path,
            selection=selection,
            source_role="baseline",
            strict_detections=None,
            strict_mask_index_sha256=None,
            path_remaps=path_remaps,
        )
        candidate = validate_association(
            candidate_association_path,
            selection=selection,
            source_role="candidate",
            strict_detections=strict_detections,
            strict_mask_index_sha256=sam_value["_mask_index_sha256"],
            path_remaps=path_remaps,
        )
        validate_source_bindings(
            selection_asset=selection_asset,
            selection_value=selection_value,
            selection=selection,
            baseline=baseline,
            candidate=candidate,
        )
        local_require(baseline.anchors == candidate.anchors, "association anchors differ")
        local_require(
            {object_id: asset.size_bytes for object_id, asset in baseline.anchor_assets.items()}
            == {
                object_id: asset.size_bytes for object_id, asset in candidate.anchor_assets.items()
            },
            "association anchor byte sizes differ",
        )
        local_require(
            (baseline.producer_script.sha256, baseline.producer_script.size_bytes)
            == (candidate.producer_script.sha256, candidate.producer_script.size_bytes),
            "association producer script assets differ",
        )
        local_require(
            baseline.camera_info_sha256 == candidate.camera_info_sha256,
            "association camera_info differs",
        )
        local_require(
            baseline.camera_info.size_bytes == candidate.camera_info.size_bytes,
            "association camera_info byte sizes differ",
        )
        local_require(
            baseline.thresholds_sha256 == candidate.thresholds_sha256,
            "association thresholds differ",
        )
        local_require(
            [frame.camera_sha256 for frame in baseline.frames]
            == [frame.camera_sha256 for frame in candidate.frames],
            "per-frame association cameras differ",
        )
        local_require(target_object_id in baseline.anchors, "target object has no explicit anchor")

        baseline_target_assignments = [
            assignment
            for frame in baseline.frames
            if (assignment := frame.assignments.get(target_object_id)) is not None
        ]
        local_require(bool(baseline_target_assignments), "target object is not visible in baseline")
        target_labels = {
            require_string(assignment.get("label"), "baseline target assignment.label")
            for assignment in baseline_target_assignments
        }
        local_require(len(target_labels) == 1, "baseline target has inconsistent semantic labels")
        target_label = next(iter(target_labels))

        remaining_ids = sorted(
            {
                object_id
                for frame in baseline.frames
                for object_id in frame.assignments
                if object_id != target_object_id
            }
        )
        local_require(bool(remaining_ids), "baseline has no remaining physical objects")
        label_registry = {object_id: index + 1 for index, object_id in enumerate(remaining_ids)}
        label_pixel_counts = {object_id: 0 for object_id in remaining_ids}
        label_frame_counts = {object_id: 0 for object_id in remaining_ids}
        frame_evaluations: list[dict[str, Any]] = []
        label_outputs: list[tuple[SourceFrame, np.ndarray, dict[str, Any]]] = []
        all_target_absent = True
        all_unassigned_absent = True
        all_remaining_preserved = True
        selection_records = require_list(selection_value.get("frames"), "selection.frames")

        for selected, selection_record, baseline_frame, candidate_frame in zip(
            selection, selection_records, baseline.frames, candidate.frames, strict=True
        ):
            selection_outputs = require_dict(
                require_dict(selection_record, "selection frame").get("outputs"),
                f"selection {selected.frame_id}.outputs",
            )
            core_asset = resolve_asset(
                selection_outputs.get("core_mask"),
                relative_to=selection_asset.path.parent,
                label=f"selection {selected.frame_id}.core_mask",
            )
            editable_asset = resolve_asset(
                selection_outputs.get("editable_mask"),
                relative_to=selection_asset.path.parent,
                label=f"selection {selected.frame_id}.editable_mask",
            )
            core = load_binary_mask(
                core_asset,
                expected_shape=(selected.height, selected.width),
                label=f"selection {selected.frame_id}.core_mask",
            )
            target_assignment_absent = target_object_id not in candidate_frame.assignments
            all_target_absent &= target_assignment_absent

            unassigned_records: list[dict[str, Any]] = []
            frame_unassigned_absent = True
            for detection_id in candidate_frame.unassigned_detection_ids:
                detection = candidate.masks[detection_id]
                if detection.label != target_label:
                    continue
                overlap_pixels = int(np.count_nonzero(detection.values & core))
                overlap_fraction = overlap_pixels / int(core.sum())
                significant = overlap_fraction > MAX_UNASSIGNED_TARGET_LABEL_CORE_OVERLAP_FRACTION
                frame_unassigned_absent &= not significant
                unassigned_records.append(
                    {
                        "detection_id": detection_id,
                        "label": detection.label,
                        "core_overlap_pixels": overlap_pixels,
                        "core_overlap_fraction": overlap_fraction,
                        "significant": significant,
                    }
                )
            all_unassigned_absent &= frame_unassigned_absent

            required_remaining = sorted(
                object_id
                for object_id in baseline_frame.assignments
                if object_id != target_object_id
            )
            missing_remaining = [
                object_id
                for object_id in required_remaining
                if object_id not in candidate_frame.assignments
            ]
            frame_remaining_preserved = not missing_remaining
            all_remaining_preserved &= frame_remaining_preserved
            extra_candidate = sorted(
                set(candidate_frame.assignments) - set(remaining_ids) - {target_object_id}
            )
            local_require(
                not extra_candidate,
                f"candidate has objects not visible in baseline: {extra_candidate}",
            )

            labels = np.zeros((selected.height, selected.width), dtype=np.uint16)
            assigned_mask_records: list[dict[str, Any]] = []
            for object_id in remaining_ids:
                assignment = candidate_frame.assignments.get(object_id)
                if assignment is None:
                    continue
                detection_id = str(assignment["detection_id"])
                mask = candidate.masks[detection_id]
                local_require(
                    not bool(np.any((labels > 0) & mask.values)),
                    f"candidate assigned masks overlap in frame {selected.frame_id}",
                )
                label_value = label_registry[object_id]
                labels[mask.values] = label_value
                pixels = int(mask.values.sum())
                label_pixel_counts[object_id] += pixels
                label_frame_counts[object_id] += 1
                assigned_mask_records.append(
                    {
                        "object_id": object_id,
                        "label": label_value,
                        "detection_id": detection_id,
                        "mask_sha256": mask.asset.sha256,
                        "mask_pixels": pixels,
                    }
                )
            frame_passed = (
                target_assignment_absent and frame_unassigned_absent and frame_remaining_preserved
            )
            frame_evaluations.append(
                {
                    "sequence_index": selected.sequence_index,
                    "frame_id": selected.frame_id,
                    "inputs": {
                        "composite_rgb": asset_record(selected.composite),
                        "core_mask": asset_record(core_asset),
                        "editable_mask": asset_record(editable_asset),
                    },
                    "target_label": target_label,
                    "target_assignment_absent": target_assignment_absent,
                    "unassigned_same_label_core_overlaps": unassigned_records,
                    "required_remaining_object_ids": required_remaining,
                    "missing_remaining_object_ids": missing_remaining,
                    "assigned_contributor_masks": assigned_mask_records,
                    "gates": {
                        "source_binding_verified": True,
                        "target_object_absent": target_assignment_absent,
                        "no_unassigned_target_semantic_overlap": frame_unassigned_absent,
                        "remaining_object_semantics_preserved": frame_remaining_preserved,
                    },
                    "passed": frame_passed,
                }
            )
            label_outputs.append((selected, labels, frame_evaluations[-1]))

        passed = all_target_absent and all_unassigned_absent and all_remaining_preserved
        status = PASS_STATUS if passed else FAIL_STATUS
        report = {
            **base_report,
            "status": status,
            "sam_semantic_gate_passed": passed,
            "frame_count": EXPECTED_FRAME_COUNT,
            "ordered_frame_ids": [frame.frame_id for frame in selection],
            "candidate_selection_receipt": asset_record(selection_asset),
            "sam_run_receipt": asset_record(sam_asset),
            "baseline_physical_association": asset_record(baseline.asset),
            "candidate_physical_association": asset_record(candidate.asset),
            "target_label": target_label,
            "label_registry": label_registry,
            "association_bindings": {
                "producer_script_sha256": ASSOCIATION_SCRIPT_SHA256,
                "producer_script_size_bytes": baseline.producer_script.size_bytes,
                "camera_info_sha256": baseline.camera_info_sha256,
                "camera_info_size_bytes": baseline.camera_info.size_bytes,
                "anchors": baseline.anchors,
                "anchor_size_bytes": {
                    object_id: baseline.anchor_assets[object_id].size_bytes
                    for object_id in sorted(baseline.anchor_assets)
                },
                "thresholds_sha256": baseline.thresholds_sha256,
                "baseline_mask_index_sha256": baseline.mask_index_sha256,
                "candidate_mask_index_sha256": candidate.mask_index_sha256,
                "strict_sam_mask_index_sha256": sam_value["_mask_index_sha256"],
                "baseline_source_role": "selection_source_rgb",
                "candidate_source_role": "selection_composite_rgb",
            },
            "aggregate": {
                "all_frames_evaluated": True,
                "all_target_object_absent": all_target_absent,
                "all_unassigned_target_overlap_absent": all_unassigned_absent,
                "all_remaining_object_semantics_preserved": all_remaining_preserved,
                "all_registered_remaining_labels_observed": all(
                    label_pixel_counts[object_id] > 0 for object_id in remaining_ids
                ),
                "remaining_label_preservation": {
                    object_id: {
                        "label": label_registry[object_id],
                        "frame_occurrence_count": label_frame_counts[object_id],
                        "pixel_count": label_pixel_counts[object_id],
                        "preserved": label_pixel_counts[object_id] > 0,
                    }
                    for object_id in remaining_ids
                },
            },
            "frame_records": frame_evaluations,
        }

        for selected, labels, frame_record in label_outputs:
            output_path = staging / "contributor_labels" / f"{selected.sequence_index:04d}.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(labels).save(output_path)
            output_sha, output_size = sha256_file(output_path)
            frame_record["outputs"] = {
                "contributor_labels": {
                    "path": output_path.relative_to(staging).as_posix(),
                    "sha256": output_sha,
                    "size_bytes": output_size,
                    "source_rgb_sha256": selected.composite.sha256,
                }
            }
    except Exception as exc:
        report = {
            **base_report,
            "status": FAIL_STATUS,
            "sam_semantic_gate_passed": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }

    try:
        report_path = staging / "semantic_qa_report.json"
        write_json(report_path, report)
        report_sha, report_size = sha256_file(report_path)
        receipt = {
            "schema_version": 1,
            "kind": "video2world.clean_plate_sam_semantic_qa_receipt",
            "status": report["status"],
            "report": {
                "path": "semantic_qa_report.json",
                "sha256": report_sha,
                "size_bytes": report_size,
            },
            "same_parent_staging": True,
            "atomic_directory_rename": True,
            "canonical_or_live_manifest_modified": False,
        }
        write_json(staging / "receipt.json", receipt)
        local_require(
            not destination.exists(),
            f"output directory appeared during run: {destination}",
        )
        os.replace(staging, destination)
        return report
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-receipt", required=True, type=Path)
    parser.add_argument("--sam-run-receipt", required=True, type=Path)
    parser.add_argument("--baseline-association", required=True, type=Path)
    parser.add_argument("--candidate-association", required=True, type=Path)
    parser.add_argument("--round-index", required=True, type=int)
    parser.add_argument("--target-object-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--association-path-remap", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = produce(
        selection_receipt=args.selection_receipt,
        sam_run_receipt=args.sam_run_receipt,
        baseline_association_path=args.baseline_association,
        candidate_association_path=args.candidate_association,
        round_index=args.round_index,
        target_object_id=args.target_object_id,
        output_dir=args.output_dir,
        association_path_remaps=args.association_path_remap,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output_dir": str(args.output_dir.expanduser().resolve()),
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == PASS_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
