#!/usr/bin/env python3
"""Build a source-bound donor exclusion index from physical mask associations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

PATH_COMPONENT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
OBJECT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
EXPLICIT_IDENTITY_SOURCE = "explicit --anchor object_id=PLY only"
EXPECTED_FORBIDDEN_IDENTITY_SOURCES = [
    "SAM3 mask filename",
    "SAM3 per-frame ordinal suffix",
    "cross-frame item order",
]
ASSOCIATION_REPORT_KIND = "video2world.multiframe_instance_mask_association_report"
RAW_INDEX_STATUS = "ready_for_cumulative_manifest_materialization"


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


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
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


def parse_object_ids(values: list[str]) -> list[str]:
    object_ids: list[str] = []
    for value in values:
        object_ids.extend(item.strip() for item in value.split(",") if item.strip())
    if not object_ids:
        raise ValueError("at least one --object-id is required")
    if any(not OBJECT_ID_PATTERN.fullmatch(item) for item in object_ids):
        raise ValueError("object ids must contain only letters, digits, dot, dash, or underscore")
    if len(set(object_ids)) != len(object_ids):
        raise ValueError("object ids contain duplicates")
    return object_ids


def source_maps(report: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    sources = require_dict(report.get("sources"), "association.sources")
    frame_map: dict[str, Any] = {}
    for raw in require_list(sources.get("frames"), "association.sources.frames"):
        record = require_dict(raw, "association source frame")
        frame_id = require_path_component(record.get("frame_id"), "association source frame_id")
        if frame_id in frame_map:
            raise ValueError("association source frames repeat or omit frame_id")
        frame_map[frame_id] = record
    mask_map: dict[str, Any] = {}
    for raw in require_list(sources.get("masks"), "association.sources.masks"):
        record = require_dict(raw, "association source mask")
        detection_id = require_path_component(
            record.get("detection_id"), "association source detection_id"
        )
        if detection_id in mask_map:
            raise ValueError("association source masks repeat or omit detection_id")
        require_path_component(record.get("frame_id"), "association source mask frame_id")
        mask_map[detection_id] = record
    return frame_map, mask_map


def validate_association(report: dict[str, Any], object_ids: list[str]) -> None:
    report_kind = report.get("kind")
    if report_kind is not None and report_kind != ASSOCIATION_REPORT_KIND:
        raise ValueError("association report kind is invalid")
    if report.get("status") != "passed":
        raise ValueError("association report status is not passed")
    gates = require_dict(report.get("gates"), "association gates")
    if gates.get("passed") is not True:
        raise ValueError("association report gates.passed is not true")
    contract = require_dict(report.get("association_contract"), "association contract")
    if contract.get("physical_identity_source") != EXPLICIT_IDENTITY_SOURCE:
        raise ValueError("association does not bind identity to explicit physical anchors")
    forbidden_sources = require_list(
        contract.get("forbidden_identity_sources"),
        "association forbidden identity sources",
    )
    if (
        any(not isinstance(item, str) or not item for item in forbidden_sources)
        or len(set(forbidden_sources)) != len(forbidden_sources)
        or not set(EXPECTED_FORBIDDEN_IDENTITY_SOURCES).issubset(forbidden_sources)
    ):
        raise ValueError("association forbidden identity sources contract is incomplete")
    sources = require_dict(report.get("sources"), "association.sources")
    anchor_ids: set[str] = set()
    for item in require_list(sources.get("anchors"), "association.sources.anchors"):
        anchor_id = require_path_component(
            require_dict(item, "association anchor").get("object_id"),
            "association anchor object_id",
        )
        if anchor_id in anchor_ids:
            raise ValueError("association physical anchors repeat object_id")
        anchor_ids.add(anchor_id)
    missing = set(object_ids) - anchor_ids
    if missing:
        raise ValueError(f"requested object ids have no physical anchor: {sorted(missing)}")


def build_index(
    *,
    association_path: Path,
    object_ids: list[str],
    output_root: Path,
    label: str,
) -> dict[str, Any]:
    association_path = association_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not association_path.is_file():
        raise FileNotFoundError(association_path)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_root}")
    object_ids = parse_object_ids(object_ids)
    require_path_component(label, "label")

    report = require_dict(read_json(association_path), "association report")
    validate_association(report, object_ids)
    source_frames, source_masks = source_maps(report)
    association_frames = require_list(report.get("frames"), "association.frames")
    if len(association_frames) != len(source_frames):
        raise ValueError("association frames and source frames have different coverage")

    if output_root.exists():
        output_root.rmdir()
    staging = output_root.with_name(f".{output_root.name}.{os.getpid()}.tmp")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        items: list[dict[str, Any]] = []
        eligible_frame_ids: list[str] = []
        dropped_frames: list[dict[str, Any]] = []
        seen_frame_ids: set[str] = set()
        total_mask_pixels = 0

        for raw_frame in association_frames:
            frame = require_dict(raw_frame, "association frame")
            frame_id = require_path_component(frame.get("frame_id"), "association frame_id")
            if frame_id in seen_frame_ids:
                raise ValueError("association frames repeat or omit frame_id")
            seen_frame_ids.add(frame_id)
            source_record = require_dict(source_frames.get(frame_id), f"source frame {frame_id}")
            source_path = Path(str(source_record.get("resolved_path"))).expanduser().resolve()
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            if source_path.stem != frame_id:
                raise ValueError(f"source RGB filename does not match frame_id {frame_id}")
            source_sha = sha256_file(source_path)
            if source_sha != source_record.get("sha256"):
                raise ValueError(f"source RGB SHA-256 mismatch for {frame_id}")

            assignments_by_object: dict[str, dict[str, Any]] = {}
            all_detection_ids: set[str] = set()
            all_mask_paths: set[Path] = set()
            all_mask_hashes: set[str] = set()
            for raw_assignment in require_list(frame.get("assignments"), "frame assignments"):
                assignment = require_dict(raw_assignment, "frame assignment")
                object_id = require_path_component(
                    assignment.get("object_id"), f"assignment object_id in {frame_id}"
                )
                if assignment.get("frame_sha256") != source_sha:
                    raise ValueError(
                        "assignment is not bound to the exact source RGB for "
                        f"{frame_id}/{object_id}"
                    )
                detection_id = require_path_component(
                    assignment.get("detection_id"),
                    f"assignment detection_id for {frame_id}/{object_id}",
                )
                if detection_id in all_detection_ids:
                    raise ValueError(
                        f"physical detection is reused across objects in {frame_id}: {detection_id}"
                    )
                all_detection_ids.add(detection_id)
                mask_record = require_dict(
                    source_masks.get(detection_id), f"source mask {detection_id}"
                )
                if mask_record.get("frame_id") != frame_id:
                    raise ValueError(f"mask frame mismatch for {detection_id}")
                mask_path = Path(str(mask_record.get("resolved_path"))).expanduser().resolve()
                if not mask_path.is_file():
                    raise FileNotFoundError(mask_path)
                mask_sha = sha256_file(mask_path)
                if mask_sha != assignment.get("mask_sha256") or mask_sha != mask_record.get(
                    "sha256"
                ):
                    raise ValueError(f"mask SHA-256 mismatch for {detection_id}")
                if mask_path in all_mask_paths or mask_sha in all_mask_hashes:
                    raise ValueError(
                        f"physical mask path or SHA-256 is reused across objects in {frame_id}"
                    )
                all_mask_paths.add(mask_path)
                all_mask_hashes.add(mask_sha)
                if object_id not in object_ids:
                    continue
                if object_id in assignments_by_object:
                    raise ValueError(f"duplicate assignment for {frame_id}/{object_id}")
                assignments_by_object[str(object_id)] = assignment
            missing_ids = [item for item in object_ids if item not in assignments_by_object]
            if missing_ids:
                dropped_frames.append(
                    {
                        "frame_id": frame_id,
                        "reason": "missing_requested_physical_assignment",
                        "missing_object_ids": missing_ids,
                        "source_rgb_sha256": source_sha,
                    }
                )
                continue

            frame_items: list[dict[str, Any]] = []
            seen_detection_ids: set[str] = set()
            seen_mask_paths: set[Path] = set()
            seen_mask_hashes: set[str] = set()
            for object_id in object_ids:
                assignment = assignments_by_object[object_id]
                if assignment.get("frame_sha256") != source_sha:
                    raise ValueError(
                        "assignment is not bound to the exact source RGB for "
                        f"{frame_id}/{object_id}"
                    )
                detection_id = require_path_component(
                    assignment.get("detection_id"),
                    f"assignment detection_id for {frame_id}/{object_id}",
                )
                if detection_id in seen_detection_ids:
                    raise ValueError(
                        f"physical detection is reused across objects in {frame_id}: {detection_id}"
                    )
                seen_detection_ids.add(detection_id)
                mask_record = require_dict(
                    source_masks.get(detection_id), f"source mask {detection_id}"
                )
                if mask_record.get("frame_id") != frame_id:
                    raise ValueError(f"mask frame mismatch for {detection_id}")
                mask_path = Path(str(mask_record.get("resolved_path"))).expanduser().resolve()
                if not mask_path.is_file():
                    raise FileNotFoundError(mask_path)
                mask_sha = sha256_file(mask_path)
                if mask_sha != assignment.get("mask_sha256") or mask_sha != mask_record.get(
                    "sha256"
                ):
                    raise ValueError(f"mask SHA-256 mismatch for {detection_id}")
                if mask_path in seen_mask_paths or mask_sha in seen_mask_hashes:
                    raise ValueError(
                        f"physical mask path or SHA-256 is reused across objects in {frame_id}"
                    )
                seen_mask_paths.add(mask_path)
                seen_mask_hashes.add(mask_sha)
                with Image.open(mask_path) as image:
                    mask = image.convert("L")
                    dimensions = [mask.width, mask.height]
                    pixels = sum(value > 0 for value in mask.getdata())
                expected_dimensions = source_record.get("dimensions")
                if expected_dimensions != dimensions:
                    raise ValueError(f"mask/source dimensions mismatch for {detection_id}")
                if pixels < 1 or pixels != mask_record.get("area_pixels"):
                    raise ValueError(f"mask area mismatch for {detection_id}")

                relative_mask = Path("masks") / frame_id / f"{object_id}.png"
                copied_mask = contained_output_path(
                    staging, relative_mask, f"copied mask {frame_id}/{object_id}"
                )
                copied_mask.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(mask_path, copied_mask)
                if sha256_file(copied_mask) != mask_sha:
                    raise ValueError(f"copied mask changed bytes for {detection_id}")
                score = mask_record.get("score")
                if not isinstance(score, int | float):
                    raise ValueError(f"mask score is missing for {detection_id}")
                total_mask_pixels += pixels
                frame_items.append(
                    {
                        "frame_id": frame_id,
                        "image": source_path.name,
                        "source_rgb_path": str(source_path),
                        "source_rgb_sha256": source_sha,
                        "label": label,
                        "object_id": object_id,
                        "detection_id": detection_id,
                        "score": float(score),
                        "mask_path": relative_mask.as_posix(),
                        "mask_sha256": mask_sha,
                        "mask_pixels": pixels,
                        "physical_identity_source": "association_explicit_3d_anchor",
                    }
                )
            items.extend(frame_items)
            eligible_frame_ids.append(frame_id)

        if seen_frame_ids != set(source_frames):
            raise ValueError("association frames do not exactly cover source frames")
        if len(eligible_frame_ids) < 2:
            raise ValueError("fewer than two source-bound donor frames remain")

        index = {
            "schema_version": 1,
            "kind": "video2world.associated_object_donor_exclusion_index",
            "status": RAW_INDEX_STATUS,
            "association_report": str(association_path),
            "association_report_sha256": sha256_file(association_path),
            "selected_object_ids": object_ids,
            "label": label,
            "eligible_donor_frame_ids": eligible_frame_ids,
            "items": items,
            "gates": {
                "physical_identity_from_explicit_3d_anchors": True,
                "every_item_bound_to_exact_source_rgb_sha256": True,
                "every_mask_sha256_verified": True,
                "incomplete_frames_excluded_from_donors": True,
                "per_frame_physical_assignments_are_one_to_one": True,
            },
        }
        index_path = staging / "mask_index.json"
        write_json(index_path, index)
        frame_ids_path = staging / "donor_frame_ids.txt"
        frame_ids_path.write_text("\n".join(eligible_frame_ids) + "\n", encoding="ascii")
        build_report = {
            "schema_version": 1,
            "kind": "video2world.associated_object_donor_exclusion_build_report",
            "created_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
            "status": "technical_passed_pending_cumulative_materialization",
            "promotion_approved": False,
            "association_report_sha256": sha256_file(association_path),
            "selected_object_ids": object_ids,
            "source_frame_count": len(source_frames),
            "eligible_donor_frame_count": len(eligible_frame_ids),
            "dropped_frame_count": len(dropped_frames),
            "dropped_frames": dropped_frames,
            "mask_item_count": len(items),
            "mask_pixel_sum_nonexclusive_across_objects_and_frames": total_mask_pixels,
            "donor_frame_ids_csv": ",".join(eligible_frame_ids),
            "next_stage": (
                "pass mask_index.json to build_cumulative_removal_manifest.py "
                "--associated-donor-index before measured RGB-D prefill"
            ),
        }
        report_path = staging / "build_report.json"
        write_json(report_path, build_report)
        receipt = {
            "schema_version": 1,
            "kind": "video2world.associated_object_donor_exclusion_receipt",
            "mask_index": "mask_index.json",
            "mask_index_sha256": sha256_file(index_path),
            "donor_frame_ids": "donor_frame_ids.txt",
            "donor_frame_ids_sha256": sha256_file(frame_ids_path),
            "build_report": "build_report.json",
            "build_report_sha256": sha256_file(report_path),
            "copied_mask_count": len(items),
        }
        write_json(staging / "receipt.json", receipt)
        staging.replace(output_root)
        return build_report
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--association-report", required=True, type=Path)
    parser.add_argument("--object-id", action="append", default=[])
    parser.add_argument("--label", default="associated_removed_object")
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        object_ids = parse_object_ids(args.object_id)
        report = build_index(
            association_path=args.association_report,
            object_ids=object_ids,
            output_root=args.output,
            label=args.label,
        )
    except (FileNotFoundError, KeyError, OSError, ValueError) as error:
        parser.exit(2, f"associated mask index build failed closed: {error}\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "eligible_donor_frame_count": report["eligible_donor_frame_count"],
                "dropped_frame_count": report["dropped_frame_count"],
                "output": str(args.output.expanduser().resolve()),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
