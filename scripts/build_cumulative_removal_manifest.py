#!/usr/bin/env python3
"""Build a strict front-to-back cumulative object-removal manifest.

Each round unions the current object's observed mask with every earlier removal
mask. Round 2 and later source RGB must come from the immediately preceding
multi-view prefill report. Donors remain original observed RGB and are paired
with the cumulative exclusion mask, so neither removed objects nor unresolved
pixels can become donor evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

KIND = "video2world.cumulative_removal_manifest"
DONOR_INDEX_KIND = "video2world.cumulative_donor_exclusion_index"
ASSOCIATED_DONOR_INDEX_KIND = "video2world.associated_object_donor_exclusion_index"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
PATH_COMPONENT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
RAW_ASSOCIATED_INDEX_STATUS = "ready_for_cumulative_manifest_materialization"
MATERIALIZED_ASSOCIATED_INDEX_STATUS = "ready_for_measured_multiview_prefill"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def require_records(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty array")
    if not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{label} entries must be objects")
    return value


def require_string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty array")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{label} entries must be non-empty strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{label} contains duplicates")
    return list(value)


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def require_path_component(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or value in {".", ".."}
        or PATH_COMPONENT_PATTERN.fullmatch(value) is None
    ):
        raise ValueError(f"{label} must be a safe single path component")
    return value


def contained_output_path(root: Path, relative_path: Path, label: str) -> Path:
    root = root.resolve()
    destination = (root / relative_path).resolve()
    try:
        destination.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes its output root") from error
    return destination


def resolve_path(value: str, *, relative_to: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def resolve_asset(record: dict[str, Any], key: str, *, manifest_path: Path) -> Path:
    value = record.get(key)
    if isinstance(value, str):
        path_value = value
    elif isinstance(value, dict) and isinstance(value.get("path"), str):
        path_value = value["path"]
    else:
        raise ValueError(f"frame {record.get('frame_id')} has no {key} path")
    path = resolve_path(path_value, relative_to=manifest_path.parent)
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_sha: Any = None
    if isinstance(value, dict):
        expected_sha = value.get("sha256")
    expected_sha = expected_sha or record.get(f"{key}_sha256")
    if isinstance(expected_sha, str) and sha256_file(path) != expected_sha:
        raise ValueError(f"frame {record.get('frame_id')} {key} SHA-256 mismatch")
    return path


def mask_array(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) > 0


def image_dimensions(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def frame_map(manifest: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in require_records(manifest.get("frame_records"), f"{label}.frame_records"):
        frame_id = require_path_component(record.get("frame_id"), f"{label} frame_id")
        if frame_id in result:
            raise ValueError(f"{label} repeats frame {frame_id}")
        result[frame_id] = record
    return result


def resolve_raw_frame(frames_dir: Path, frame_id: str) -> Path:
    require_path_component(frame_id, "original donor frame_id")
    matches = sorted(frames_dir.glob(f"{frame_id}.*"))
    if len(matches) != 1:
        raise ValueError(f"expected one original donor RGB for {frame_id}, found {len(matches)}")
    return matches[0].resolve()


def current_mask(record: dict[str, Any], *, manifest_path: Path) -> Path:
    for key in ("input_mask", "union_mask", "source_mask"):
        try:
            return resolve_asset(record, key, manifest_path=manifest_path)
        except ValueError:
            continue
    raise ValueError(f"frame {record.get('frame_id')} has no current object mask")


def initial_source(record: dict[str, Any], *, manifest_path: Path) -> Path:
    for key in ("input_frame", "source_frame"):
        try:
            return resolve_asset(record, key, manifest_path=manifest_path)
        except ValueError:
            continue
    raise ValueError(f"frame {record.get('frame_id')} has no source frame")


def segmentation_source_sha256(record: dict[str, Any], *, manifest_path: Path) -> str:
    """Return the exact RGB byte hash on which the current mask was segmented."""

    source = record.get("source_frame")
    if isinstance(source, dict) and isinstance(source.get("sha256"), str):
        return source["sha256"]
    if isinstance(source, str) and isinstance(record.get("source_frame_sha256"), str):
        return record["source_frame_sha256"]

    assignment = record.get("association_assignment")
    if isinstance(assignment, dict) and isinstance(assignment.get("frame_sha256"), str):
        return assignment["frame_sha256"]

    source_path = initial_source(record, manifest_path=manifest_path)
    return sha256_file(source_path)


def associated_index_header(
    path: Path,
    *,
    expected_removed_object_ids: list[str],
) -> tuple[dict[str, Any], list[str]]:
    value = require_dict(read_json(path), "associated donor index")
    if value.get("kind") != ASSOCIATED_DONOR_INDEX_KIND:
        raise ValueError("associated donor index kind is invalid")
    status = value.get("status")
    if status not in {
        RAW_ASSOCIATED_INDEX_STATUS,
        MATERIALIZED_ASSOCIATED_INDEX_STATUS,
    }:
        raise ValueError("associated donor index status is not ready for materialization")
    gates = require_dict(value.get("gates"), "associated donor index gates")
    required_gates = (
        "physical_identity_from_explicit_3d_anchors",
        "every_item_bound_to_exact_source_rgb_sha256",
        "every_mask_sha256_verified",
        "incomplete_frames_excluded_from_donors",
    )
    if any(gates.get(key) is not True for key in required_gates):
        raise ValueError("associated donor index gates did not all pass")
    selected_object_ids = require_string_list(
        value.get("selected_object_ids"),
        "associated donor index selected_object_ids",
    )
    for object_id in selected_object_ids:
        require_path_component(object_id, "associated donor object_id")
    if selected_object_ids != expected_removed_object_ids:
        raise ValueError(
            "associated donor object ids do not equal the cumulative removed-object union"
        )
    eligible_frame_ids = require_string_list(
        value.get("eligible_donor_frame_ids"),
        "associated donor index eligible_donor_frame_ids",
    )
    for frame_id in eligible_frame_ids:
        require_path_component(frame_id, "associated eligible donor frame_id")
    if len(eligible_frame_ids) < 2:
        raise ValueError("associated donor index has fewer than two eligible donor frames")
    label = value.get("label")
    if not isinstance(label, str) or not label:
        raise ValueError("associated donor index label is missing")
    require_records(value.get("items"), "associated donor index items")
    return value, eligible_frame_ids


def materialize_associated_donor_evidence(
    *,
    source_index_path: Path,
    source_index: dict[str, Any],
    expected_removed_object_ids: list[str],
    eligible_frame_ids: list[str],
    donor_assets: dict[str, dict[str, Any]],
    staging: Path,
) -> dict[str, Any]:
    evidence_root = staging / "associated_donor_evidence"
    masks_root = evidence_root / "masks"
    unions_root = evidence_root / "unions"
    masks_root.mkdir(parents=True)
    unions_root.mkdir(parents=True)
    label = str(source_index["label"])
    expected_pairs = {
        (frame_id, object_id)
        for frame_id in eligible_frame_ids
        for object_id in expected_removed_object_ids
    }
    portable_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    masks_by_frame: dict[str, list[np.ndarray]] = {frame_id: [] for frame_id in eligible_frame_ids}
    detection_ids_by_frame: dict[str, set[str]] = {
        frame_id: set() for frame_id in eligible_frame_ids
    }
    mask_paths_by_frame: dict[str, set[Path]] = {frame_id: set() for frame_id in eligible_frame_ids}
    mask_hashes_by_frame: dict[str, set[str]] = {frame_id: set() for frame_id in eligible_frame_ids}

    for raw_item in require_records(source_index.get("items"), "associated donor index items"):
        frame_id_value = raw_item.get("frame_id")
        image = raw_item.get("image")
        object_id_value = raw_item.get("object_id")
        frame_id = (
            require_path_component(frame_id_value, "associated donor item frame_id")
            if isinstance(frame_id_value, str)
            else frame_id_value
        )
        object_id = (
            require_path_component(object_id_value, "associated donor item object_id")
            if isinstance(object_id_value, str)
            else object_id_value
        )
        pair = (frame_id, object_id)
        if (
            not isinstance(frame_id, str)
            or frame_id not in donor_assets
            or not isinstance(image, str)
            or Path(image).name != image
            or Path(image).stem != frame_id
        ):
            raise ValueError("associated donor item has an invalid eligible frame identity")
        if not isinstance(object_id, str) or object_id not in expected_removed_object_ids:
            raise ValueError("associated donor item has no requested physical object identity")
        if pair in portable_by_pair:
            raise ValueError(f"associated donor index repeats {frame_id}/{object_id}")
        if raw_item.get("label") != label:
            raise ValueError("associated donor item label differs from its index")
        detection_id = require_path_component(
            raw_item.get("detection_id"),
            f"associated donor detection_id for {frame_id}/{object_id}",
        )
        if detection_id in detection_ids_by_frame[frame_id]:
            raise ValueError(
                f"associated donor physical detection is reused in {frame_id}: {detection_id}"
            )
        detection_ids_by_frame[frame_id].add(detection_id)
        if raw_item.get("physical_identity_source") != "association_explicit_3d_anchor":
            raise ValueError(
                "associated donor item is category-level rather than physical-instance"
            )

        donor_asset = donor_assets[frame_id]
        source_sha = require_sha256(
            raw_item.get("source_rgb_sha256"),
            f"associated donor source RGB {frame_id}",
        )
        if source_sha != donor_asset["sha256"]:
            raise ValueError(f"associated donor source RGB SHA-256 mismatch for {frame_id}")
        if not isinstance(raw_item.get("source_rgb_path"), str):
            raise ValueError("associated donor item source_rgb_path is missing")

        mask_value = raw_item.get("mask_path")
        if not isinstance(mask_value, str):
            raise ValueError("associated donor item mask_path is missing")
        source_mask_path = resolve_path(mask_value, relative_to=source_index_path.parent)
        if not source_mask_path.is_file():
            raise FileNotFoundError(source_mask_path)
        mask_sha = require_sha256(
            raw_item.get("mask_sha256"),
            f"associated donor mask {frame_id}/{object_id}",
        )
        if sha256_file(source_mask_path) != mask_sha:
            raise ValueError(f"associated donor mask SHA-256 mismatch for {frame_id}/{object_id}")
        if (
            source_mask_path in mask_paths_by_frame[frame_id]
            or mask_sha in mask_hashes_by_frame[frame_id]
        ):
            raise ValueError(
                f"associated donor physical mask path or SHA-256 is reused in {frame_id}"
            )
        mask_paths_by_frame[frame_id].add(source_mask_path)
        mask_hashes_by_frame[frame_id].add(mask_sha)
        mask = mask_array(source_mask_path)
        if list(mask.shape[::-1]) != donor_asset["dimensions"]:
            raise ValueError(
                f"associated donor mask dimensions mismatch for {frame_id}/{object_id}"
            )
        declared_pixels = raw_item.get("mask_pixels")
        if not isinstance(declared_pixels, int) or declared_pixels != int(mask.sum()):
            raise ValueError(
                f"associated donor mask pixel count mismatch for {frame_id}/{object_id}"
            )
        score = raw_item.get("score")
        if not isinstance(score, int | float):
            raise ValueError(f"associated donor mask score is missing for {frame_id}/{object_id}")

        relative_mask = Path("masks") / frame_id / f"{object_id}.png"
        copied_mask = contained_output_path(
            evidence_root, relative_mask, f"associated donor mask {frame_id}/{object_id}"
        )
        copied_mask.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_mask_path, copied_mask)
        if sha256_file(copied_mask) != mask_sha:
            raise ValueError(f"associated donor mask changed while copying {frame_id}/{object_id}")
        portable_by_pair[(frame_id, object_id)] = {
            "frame_id": frame_id,
            "image": Path(str(donor_asset["path"])).name,
            "source_rgb_path": donor_asset["path"],
            "source_rgb_sha256": source_sha,
            "label": label,
            "object_id": object_id,
            "detection_id": detection_id,
            "score": float(score),
            "mask_path": relative_mask.as_posix(),
            "mask_sha256": mask_sha,
            "mask_pixels": declared_pixels,
            "physical_identity_source": "association_explicit_3d_anchor",
        }
        masks_by_frame[frame_id].append(mask)

    if set(portable_by_pair) != expected_pairs:
        missing = sorted(expected_pairs - set(portable_by_pair))
        extra = sorted(set(portable_by_pair) - expected_pairs)
        raise ValueError(
            f"associated donor masks do not cover the exact removed-object union: "
            f"missing={missing}, extra={extra}"
        )

    donor_unions: dict[str, dict[str, Any]] = {}
    for frame_id in eligible_frame_ids:
        masks = masks_by_frame[frame_id]
        union = np.logical_or.reduce(masks)
        union_path = contained_output_path(
            unions_root, Path(f"{frame_id}.png"), f"associated donor union {frame_id}"
        )
        Image.fromarray(union.astype(np.uint8) * 255).save(union_path)
        donor_unions[frame_id] = {
            "path": f"unions/{frame_id}.png",
            "sha256": sha256_file(union_path),
            "pixels": int(union.sum()),
            "object_ids": expected_removed_object_ids,
            "component_mask_count": len(masks),
        }

    portable_items = [
        portable_by_pair[(frame_id, object_id)]
        for frame_id in eligible_frame_ids
        for object_id in expected_removed_object_ids
    ]
    portable_index = {
        "schema_version": 1,
        "kind": ASSOCIATED_DONOR_INDEX_KIND,
        "status": "ready_for_measured_multiview_prefill",
        "source_index": "source_mask_index.json",
        "source_index_sha256": sha256_file(source_index_path),
        "association_report_sha256": source_index.get("association_report_sha256"),
        "selected_object_ids": expected_removed_object_ids,
        "target_removed_object_union": expected_removed_object_ids,
        "label": label,
        "eligible_donor_frame_ids": eligible_frame_ids,
        "items": portable_items,
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
    }
    portable_index_path = evidence_root / "mask_index.json"
    write_json(portable_index_path, portable_index)
    shutil.copyfile(source_index_path, evidence_root / "source_mask_index.json")
    return {
        "mode": "associated_physical_instance_masks",
        "label": label,
        "index": "associated_donor_evidence/mask_index.json",
        "index_sha256": sha256_file(portable_index_path),
        "source_index": "associated_donor_evidence/source_mask_index.json",
        "declared_source_index": str(source_index_path),
        "source_index_sha256": sha256_file(source_index_path),
        "selected_object_ids": expected_removed_object_ids,
        "eligible_donor_frame_ids": eligible_frame_ids,
        "per_donor_mask_count": len(expected_removed_object_ids),
    }


def previous_prefill_blocker_summary(previous_report: dict[str, Any]) -> str:
    details: list[str] = []
    status = previous_report.get("status")
    if isinstance(status, str) and status:
        details.append(f"status={status}")
    promotion_blocker = previous_report.get("promotion_blocker")
    if isinstance(promotion_blocker, str) and promotion_blocker:
        details.append(f"promotion_blocker={promotion_blocker}")
    next_action = previous_report.get("next_action")
    if isinstance(next_action, dict):
        action = next_action.get("action")
        blocker = next_action.get("blocker")
        if isinstance(action, str) and action:
            details.append(f"next_action={action}")
        if isinstance(blocker, str) and blocker:
            details.append(f"next_blocker={blocker}")
        failed_frame_ids = next_action.get("failed_frame_ids")
        if isinstance(failed_frame_ids, list) and failed_frame_ids:
            frames = [
                frame_id
                for frame_id in failed_frame_ids
                if isinstance(frame_id, str) and frame_id
            ]
            if frames:
                details.append(f"failed_frame_ids={','.join(frames)}")
        first_failed_frame_id = next_action.get("first_failed_frame_id")
        if isinstance(first_failed_frame_id, str) and first_failed_frame_id:
            details.append(f"first_failed_frame_id={first_failed_frame_id}")
        for key in (
            "failed_pair_ids",
            "failed_triplet_center_frame_ids",
            "not_evaluable_pair_ids",
            "not_evaluable_triplet_center_frame_ids",
        ):
            values = next_action.get(key)
            if isinstance(values, list) and values:
                kept = [value for value in values if isinstance(value, str) and value]
                if kept:
                    details.append(f"{key}={','.join(kept)}")
        no_support_frame_ids = next_action.get("no_support_frame_ids")
        if isinstance(no_support_frame_ids, list) and no_support_frame_ids:
            frames = [
                frame_id
                for frame_id in no_support_frame_ids
                if isinstance(frame_id, str) and frame_id
            ]
            if frames:
                details.append(f"no_support_frame_ids={','.join(frames)}")
        unresolved = next_action.get("unresolved_unobserved_pixels")
        if isinstance(unresolved, int):
            details.append(f"unresolved_unobserved_pixels={unresolved}")
    provenance = previous_report.get("pixel_provenance")
    if isinstance(provenance, dict):
        unresolved = provenance.get("unresolved_unobserved_pixels")
        if isinstance(unresolved, int):
            details.append(f"pixel_provenance.unresolved_unobserved_pixels={unresolved}")
    if not details:
        return ""
    return " [" + "; ".join(details) + "]"


def previous_prefill_error(message: str, previous_report: dict[str, Any]) -> ValueError:
    return ValueError(message + previous_prefill_blocker_summary(previous_report))


def validate_previous_lineage(
    *,
    current_round_index: int,
    previous_manifest_path: Path,
    previous_report_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    previous_manifest = require_dict(read_json(previous_manifest_path), "previous manifest")
    if previous_manifest.get("kind") != KIND:
        raise ValueError("previous manifest is not a cumulative-removal manifest")
    if previous_manifest.get("round_index") != current_round_index - 1:
        raise ValueError("previous manifest is not the immediately preceding round")
    previous_report = require_dict(read_json(previous_report_path), "previous prefill report")
    if previous_report.get("status") != "technical_passed":
        raise previous_prefill_error(
            "previous prefill report did not technically pass", previous_report
        )
    gates = require_dict(previous_report.get("gates"), "previous report gates")
    if gates.get("outside_removal_mask_rgb_exact") is not True:
        raise previous_prefill_error(
            "previous prefill did not preserve RGB outside its removal mask",
            previous_report,
        )
    if gates.get("all_residual_masks_subset_of_removal_masks") is not True:
        raise previous_prefill_error(
            "previous residual masks were not subsets of removal masks",
            previous_report,
        )
    if gates.get("donor_support_available_for_every_target") is not True:
        raise previous_prefill_error(
            "previous prefill has one or more targets with no donor support",
            previous_report,
        )
    if gates.get("donor_support_not_boundary_concentrated") is not True:
        raise previous_prefill_error(
            "previous prefill donor support failed its boundary guard",
            previous_report,
        )
    provenance = require_dict(
        previous_report.get("pixel_provenance"), "previous report pixel provenance"
    )
    if provenance.get("generated_pixels") != 0:
        raise ValueError("previous prefill report contains or omits generated pixel provenance")
    if provenance.get("propainter_pixels") != 0:
        raise ValueError("previous prefill report contains or omits ProPainter pixel provenance")
    if provenance.get("unresolved_pixels_are_not_valid_donor_or_geometry_evidence") is not True:
        raise ValueError("previous prefill report allows unresolved pixels as evidence")
    measured_pixels = provenance.get("measured_multiview_pixels")
    if not isinstance(measured_pixels, int) or measured_pixels < 1:
        raise previous_prefill_error(
            "previous prefill has no measured multi-view pixels", previous_report
        )
    expected_manifest_sha = sha256_file(previous_manifest_path)
    if previous_report.get("input_manifest_sha256") != expected_manifest_sha:
        raise ValueError("previous report does not hash the previous cumulative manifest")
    report_records = frame_map(previous_report, "previous report")
    manifest_records = frame_map(previous_manifest, "previous manifest")
    if set(report_records) != set(manifest_records):
        raise ValueError("previous report and manifest cover different frames")
    for frame_id, record in report_records.items():
        covered_pixels = record.get("covered_pixels")
        if not isinstance(covered_pixels, int) or covered_pixels < 1:
            raise previous_prefill_error(
                f"previous prefill target {frame_id} has no measured donor support",
                previous_report,
            )
    return manifest_records, report_records, previous_report


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    current_manifest_path = args.current_object_manifest.expanduser().resolve()
    donor_frames_dir = args.observed_donor_frames_dir.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    if args.round_index < 1:
        raise ValueError("round-index must be positive")
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    if not donor_frames_dir.is_dir():
        raise NotADirectoryError(donor_frames_dir)

    current_manifest = require_dict(read_json(current_manifest_path), "current object manifest")
    current_records = frame_map(current_manifest, "current object manifest")
    frame_ids = sorted(current_records)
    manifest_object_id = current_manifest.get("object_id")
    object_id = args.object_id or manifest_object_id
    if not isinstance(object_id, str) or not object_id:
        raise ValueError("object-id is required when the current manifest has no object_id")
    require_path_component(object_id, "current object_id")
    if isinstance(manifest_object_id, str) and manifest_object_id != object_id:
        raise ValueError("object-id does not match the current object manifest")

    previous_manifest_path = (
        args.previous_cumulative_manifest.expanduser().resolve()
        if args.previous_cumulative_manifest
        else None
    )
    previous_report_path = (
        args.previous_prefill_report.expanduser().resolve()
        if args.previous_prefill_report
        else None
    )
    if args.round_index == 1 and (previous_manifest_path or previous_report_path):
        raise ValueError("round 1 cannot have previous-round lineage")
    if args.round_index > 1 and not (previous_manifest_path and previous_report_path):
        raise ValueError("round 2+ requires previous cumulative manifest and prefill report")

    previous_records: dict[str, dict[str, Any]] = {}
    previous_report_records: dict[str, dict[str, Any]] = {}
    previous_report: dict[str, Any] | None = None
    if previous_manifest_path and previous_report_path:
        previous_records, previous_report_records, previous_report = validate_previous_lineage(
            current_round_index=args.round_index,
            previous_manifest_path=previous_manifest_path,
            previous_report_path=previous_report_path,
        )
        if set(previous_records) != set(current_records):
            raise ValueError("current and previous manifests cover different frames")

    prior_removed_object_ids: list[str] = []
    if previous_records:
        prior_unions = {
            tuple(record.get("removed_object_ids", [])) for record in previous_records.values()
        }
        if len(prior_unions) != 1:
            raise ValueError("previous cumulative frames disagree on removed-object union")
        prior_removed_object_ids = list(next(iter(prior_unions)))
        if len(prior_removed_object_ids) != args.round_index - 1:
            raise ValueError("previous removed-object count does not match round index")
    expected_removed_object_ids = [*prior_removed_object_ids, object_id]
    if len(set(expected_removed_object_ids)) != len(expected_removed_object_ids):
        raise ValueError("the same object appears in more than one peel round")

    associated_index_path = (
        getattr(args, "associated_donor_index", None).expanduser().resolve()
        if getattr(args, "associated_donor_index", None)
        else None
    )
    associated_index: dict[str, Any] | None = None
    eligible_donor_frame_ids: list[str] | None = None
    if associated_index_path is not None:
        if not associated_index_path.is_file():
            raise FileNotFoundError(associated_index_path)
        associated_index, eligible_donor_frame_ids = associated_index_header(
            associated_index_path,
            expected_removed_object_ids=expected_removed_object_ids,
        )

    target_raw_assets: dict[str, dict[str, Any]] = {}
    for frame_id in frame_ids:
        donor_path = resolve_raw_frame(donor_frames_dir, frame_id)
        target_raw_assets[frame_id] = {
            "path": str(donor_path),
            "sha256": sha256_file(donor_path),
            "bytes": donor_path.stat().st_size,
            "dimensions": list(image_dimensions(donor_path)),
            "role": "original_observed_rgb_only",
        }
    donor_assets: dict[str, dict[str, Any]] = {}
    for frame_id in eligible_donor_frame_ids or frame_ids:
        donor_path = resolve_raw_frame(donor_frames_dir, frame_id)
        donor_assets[frame_id] = {
            "path": str(donor_path),
            "sha256": sha256_file(donor_path),
            "bytes": donor_path.stat().st_size,
            "dimensions": list(image_dimensions(donor_path)),
            "role": "original_observed_rgb_only",
        }

    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_root.parent,
        prefix=f".{output_root.name}.staging-",
    ) as temporary_value:
        staging = Path(temporary_value)
        masks_dir = staging / "masks"
        masks_dir.mkdir(parents=True)
        output_records: list[dict[str, Any]] = []
        donor_index_items: list[dict[str, Any]] = []
        associated_contract: dict[str, Any] | None = None
        if (
            associated_index_path is not None
            and associated_index is not None
            and eligible_donor_frame_ids is not None
        ):
            associated_contract = materialize_associated_donor_evidence(
                source_index_path=associated_index_path,
                source_index=associated_index,
                expected_removed_object_ids=expected_removed_object_ids,
                eligible_frame_ids=eligible_donor_frame_ids,
                donor_assets=donor_assets,
                staging=staging,
            )

        for sequence_index, frame_id in enumerate(frame_ids):
            record = current_records[frame_id]
            object_mask_path = current_mask(record, manifest_path=current_manifest_path)
            object_mask = mask_array(object_mask_path)
            mask_source_sha = segmentation_source_sha256(
                record,
                manifest_path=current_manifest_path,
            )
            raw_asset = target_raw_assets[frame_id]
            raw_dimensions = tuple(raw_asset["dimensions"])
            if object_mask.shape[::-1] != raw_dimensions:
                raise ValueError(f"current mask dimensions do not match raw RGB for {frame_id}")

            components = [
                {
                    "object_id": object_id,
                    "round_index": args.round_index,
                    "role": "current_observed_object_mask",
                    "path": str(object_mask_path),
                    "sha256": sha256_file(object_mask_path),
                    "pixels": int(object_mask.sum()),
                    "segmentation_source_rgb_sha256": mask_source_sha,
                }
            ]
            cumulative_mask = object_mask.copy()
            input_lineage: dict[str, Any]
            if args.round_index == 1:
                source_path = initial_source(record, manifest_path=current_manifest_path)
                source_sha = sha256_file(source_path)
                if source_sha != raw_asset["sha256"]:
                    raise ValueError(f"round 1 source is not original observed RGB for {frame_id}")
                if mask_source_sha != source_sha:
                    raise ValueError(
                        f"current object mask for {frame_id} was not segmented on the exact "
                        "round 1 source RGB"
                    )
                prior_object_ids: list[str] = []
                input_lineage = {
                    "role": "original_observed_rgb",
                    "source_sha256": source_sha,
                }
            else:
                previous_record = previous_records[frame_id]
                previous_mask_path = resolve_asset(
                    previous_record,
                    "union_mask",
                    manifest_path=previous_manifest_path,  # type: ignore[arg-type]
                )
                previous_mask = mask_array(previous_mask_path)
                if previous_mask.shape != object_mask.shape:
                    raise ValueError(f"previous/current mask shape mismatch for {frame_id}")
                cumulative_mask |= previous_mask
                prior_object_ids = list(previous_record.get("removed_object_ids", []))
                if len(prior_object_ids) != args.round_index - 1:
                    raise ValueError("previous removed-object count does not match round index")
                components = list(previous_record.get("mask_components", [])) + components

                report_record = previous_report_records[frame_id]
                source_path = resolve_path(
                    str(report_record["prefill_frame"]),
                    relative_to=previous_report_path.parent,  # type: ignore[union-attr]
                )
                if sha256_file(source_path) != report_record.get("prefill_frame_sha256"):
                    raise ValueError(f"previous prefill frame SHA-256 mismatch for {frame_id}")
                if mask_source_sha != report_record.get("prefill_frame_sha256"):
                    raise ValueError(
                        f"current object mask for {frame_id} was not segmented on the exact "
                        "immediately previous prefill RGB"
                    )
                residual_path = resolve_path(
                    str(report_record["residual_mask"]),
                    relative_to=previous_report_path.parent,  # type: ignore[union-attr]
                )
                residual = mask_array(residual_path)
                if residual.shape != previous_mask.shape or np.any(residual & ~previous_mask):
                    raise ValueError(f"previous residual escapes removal mask for {frame_id}")
                input_lineage = {
                    "role": "immediately_previous_measured_prefill_with_unresolved_mask",
                    "previous_prefill_frame_sha256": report_record["prefill_frame_sha256"],
                    "segmentation_source_rgb_sha256": mask_source_sha,
                    "previous_residual_mask_sha256": report_record["residual_mask_sha256"],
                    "previous_residual_pixels": report_record["residual_mask_pixels"],
                    "previous_unresolved_pixels_remain_non_donor": True,
                }

            removed_object_ids = [*prior_object_ids, object_id]
            if len(set(removed_object_ids)) != len(removed_object_ids):
                raise ValueError("the same object appears in more than one peel round")
            if removed_object_ids != expected_removed_object_ids:
                raise ValueError("frame removed-object ids differ from the target union")
            output_name = f"{sequence_index:04d}.png"
            cumulative_path = masks_dir / output_name
            Image.fromarray(cumulative_mask.astype(np.uint8) * 255).save(cumulative_path)
            relative_mask_path = f"masks/{output_name}"
            cumulative_sha = sha256_file(cumulative_path)
            source_sha = sha256_file(source_path)
            output_records.append(
                {
                    "sequence_index": sequence_index,
                    "frame_id": frame_id,
                    "source_frame": str(source_path),
                    "source_frame_sha256": source_sha,
                    "source_role": input_lineage["role"],
                    "union_mask": relative_mask_path,
                    "union_mask_sha256": cumulative_sha,
                    "union_mask_pixels": int(cumulative_mask.sum()),
                    "donor_exclusion_mask": relative_mask_path,
                    "donor_exclusion_mask_sha256": cumulative_sha,
                    "removed_object_ids": removed_object_ids,
                    "mask_components": components,
                    "current_object_mask_sha256": sha256_file(object_mask_path),
                    "current_object_mask_pixels": int(object_mask.sum()),
                    "current_object_mask_segmentation_source_rgb_sha256": mask_source_sha,
                    "input_lineage": input_lineage,
                }
            )
            donor_index_items.append(
                {
                    "image": f"{frame_id}.png",
                    "frame_id": frame_id,
                    "source_rgb_path": raw_asset["path"],
                    "source_rgb_sha256": raw_asset["sha256"],
                    "label": "cumulative_removed",
                    "mask_path": relative_mask_path,
                    "mask_sha256": cumulative_sha,
                    "removed_object_ids": removed_object_ids,
                    "unresolved_residual_allowed_as_donor": False,
                }
            )

        donor_index = {
            "schema_version": 1,
            "kind": DONOR_INDEX_KIND,
            "round_index": args.round_index,
            "removed_object_ids": output_records[0]["removed_object_ids"],
            "label": "cumulative_removed",
            "items": donor_index_items,
        }
        donor_index_path = staging / "donor_exclusion_index.json"
        write_json(donor_index_path, donor_index)

        lineage: dict[str, Any] = {
            "previous_round_index": args.round_index - 1 if args.round_index > 1 else None,
            "previous_cumulative_manifest": (
                str(previous_manifest_path) if previous_manifest_path else None
            ),
            "previous_cumulative_manifest_sha256": (
                sha256_file(previous_manifest_path) if previous_manifest_path else None
            ),
            "previous_prefill_report": str(previous_report_path) if previous_report_path else None,
            "previous_prefill_report_sha256": (
                sha256_file(previous_report_path) if previous_report_path else None
            ),
            "previous_prefill_status": previous_report.get("status") if previous_report else None,
        }
        donor_contract: dict[str, Any] = {
            "role": "original_observed_rgb_only",
            "frames_dir": str(donor_frames_dir),
            "assets": donor_assets,
            "generated_or_prefilled_rgb_as_donor": False,
            "unresolved_residual_as_donor": False,
        }
        if associated_contract is None:
            donor_contract.update(
                {
                    "exclusion_label": "cumulative_removed",
                    "exclusion_index": "donor_exclusion_index.json",
                    "exclusion_index_sha256": sha256_file(donor_index_path),
                }
            )
        else:
            donor_contract.update(
                {
                    "exclusion_mode": associated_contract["mode"],
                    "exclusion_label": associated_contract["label"],
                    "exclusion_index": associated_contract["index"],
                    "exclusion_index_sha256": associated_contract["index_sha256"],
                    "associated_source_index": associated_contract["source_index"],
                    "associated_source_index_sha256": associated_contract["source_index_sha256"],
                    "associated_source_index_declared_path": associated_contract[
                        "declared_source_index"
                    ],
                    "eligible_donor_frame_ids": associated_contract["eligible_donor_frame_ids"],
                    "physical_instance_object_ids": associated_contract["selected_object_ids"],
                    "per_donor_multi_mask_union": True,
                    "per_donor_mask_count": associated_contract["per_donor_mask_count"],
                    "legacy_target_cumulative_exclusion_index": ("donor_exclusion_index.json"),
                    "legacy_target_cumulative_exclusion_index_sha256": sha256_file(
                        donor_index_path
                    ),
                }
            )

        manifest = {
            "schema_version": 1,
            "kind": KIND,
            "status": "ready_for_measured_multiview_prefill",
            "created_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 - mil8 is 3.10
            "round_index": args.round_index,
            "current_object_id": object_id,
            "removed_object_ids": output_records[0]["removed_object_ids"],
            "current_object_manifest": str(current_manifest_path),
            "current_object_manifest_sha256": sha256_file(current_manifest_path),
            "lineage": lineage,
            "donor_contract": donor_contract,
            "frame_count": len(output_records),
            "frame_records": output_records,
            "gates": {
                "strict_front_to_back_order": True,
                "input_is_immediately_previous_round": True,
                "all_previous_masks_are_accumulated": True,
                "all_removed_objects_are_excluded_from_donors": True,
                "donors_are_original_observed_rgb": True,
                "unresolved_residual_is_never_donor_evidence": True,
                "previous_prefill_excludes_generated_and_propainter_pixels": True,
                "current_object_mask_is_bound_to_exact_scene_source_rgb": True,
                "donor_exclusion_object_union_matches_removed_object_ids": True,
                "associated_physical_instance_donor_exclusion_enforced": (
                    associated_contract is not None
                ),
            },
            "pixel_provenance_classes": {
                "outside_cumulative_mask": "source scene RGB preserved exactly",
                "measured_multiview": "pending calibrated RGB-D donor reprojection",
                "unresolved_unobserved": "pending; must remain in residual mask",
                "generated": "forbidden in this measurement stage",
            },
        }
        manifest_path = staging / "cumulative_removal_manifest.json"
        write_json(manifest_path, manifest)
        receipt = {
            "schema_version": 1,
            "kind": "video2world.cumulative_removal_manifest_receipt",
            "manifest": "cumulative_removal_manifest.json",
            "manifest_sha256": sha256_file(manifest_path),
            "donor_exclusion_index": "donor_exclusion_index.json",
            "donor_exclusion_index_sha256": sha256_file(donor_index_path),
            "associated_donor_exclusion_index": (
                associated_contract["index"] if associated_contract else None
            ),
            "associated_donor_exclusion_index_sha256": (
                associated_contract["index_sha256"] if associated_contract else None
            ),
        }
        write_json(staging / "receipt.json", receipt)
        shutil.move(str(staging), output_root)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-index", type=int, required=True)
    parser.add_argument("--object-id")
    parser.add_argument("--current-object-manifest", type=Path, required=True)
    parser.add_argument("--previous-cumulative-manifest", type=Path)
    parser.add_argument("--previous-prefill-report", type=Path)
    parser.add_argument("--observed-donor-frames-dir", type=Path, required=True)
    parser.add_argument("--associated-donor-index", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = build_manifest(args)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "round_index": manifest["round_index"],
                "removed_object_ids": manifest["removed_object_ids"],
                "frame_count": manifest["frame_count"],
                "output": str(args.output / "cumulative_removal_manifest.json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
