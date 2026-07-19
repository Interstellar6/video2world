#!/usr/bin/env python3
"""Materialize an audited SAM3 run as immutable semantic evidence.

The legacy Holi-Spatial receipt is not sufficient by itself: it does not bind
each mask to the exact selected clean-plate composite. This bridge validates
the audited runner/checkpoint, the complete 25-frame mask index, every decoded
mask byte, and every selected composite before atomically publishing a portable
review-only receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

OUTPUT_KIND = "video2world.clean_plate_sam_semantic_run_receipt"
OUTPUT_STATUS = "technical_passed_semantic_evidence_materialized"
SELECTION_KIND = "video2world.clean_plate_candidate_selection_receipt"
SELECTION_STATUS = "materialized_candidate_selection_pending_human_review"
LEGACY_KIND = "video2world.layered_peel.sam3_reinspection"
LEGACY_STATUS = "completed"
EXPECTED_FRAME_COUNT = 25
EXPECTED_RUNNER_SHA256 = "93b34a5fe50a8ff000b047479ca3ecea1f16f65b74e5cda6b20461120d1f3e44"
EXPECTED_CHECKPOINT_SHA256 = "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
EXPECTED_CHECKPOINT_BYTES = 3_450_062_241
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ContractError(ValueError):
    """The legacy evidence cannot authorize a strict SAM run receipt."""


@dataclass(frozen=True)
class Asset:
    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class SourceFrame:
    sequence_index: int
    frame_id: str
    composite: Asset
    width: int
    height: int


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def require_dict(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be an array")
    return value


def require_string(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value.strip()), f"{label} must be non-empty")
    return value


def require_sha256(value: Any, label: str) -> str:
    require(
        isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None,
        f"{label} must be a lowercase SHA-256 digest",
    )
    return value


def asset_record(asset: Asset, *, path: str | None = None) -> dict[str, Any]:
    return {
        "path": path if path is not None else str(asset.path),
        "sha256": asset.sha256,
        "size_bytes": asset.size_bytes,
    }


def resolve_asset(value: Any, *, relative_to: Path, label: str) -> Asset:
    record = require_dict(value, label)
    path = Path(require_string(record.get("path"), f"{label}.path")).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    path = path.resolve()
    require(path.is_file(), f"{label} does not exist: {path}")
    expected = require_sha256(record.get("sha256"), f"{label}.sha256")
    digest, size = sha256_file(path)
    require(digest == expected, f"{label} SHA-256 mismatch")
    return Asset(path=path, sha256=digest, size_bytes=size)


def parse_path_remaps(values: list[str]) -> list[tuple[Path, Path]]:
    result: list[tuple[Path, Path]] = []
    for raw in values:
        source, separator, destination = raw.partition("=")
        require(bool(separator and source and destination), f"invalid path remap: {raw}")
        source_path = Path(source).expanduser()
        destination_path = Path(destination).expanduser()
        require(source_path.is_absolute(), f"path remap source must be absolute: {raw}")
        require(destination_path.is_absolute(), f"path remap target must be absolute: {raw}")
        result.append((source_path, destination_path))
    return result


def resolve_remapped_path(
    value: str,
    *,
    relative_to: Path,
    remaps: list[tuple[Path, Path]],
) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        for source, destination in remaps:
            try:
                relative = path.relative_to(source)
            except ValueError:
                continue
            return (destination / relative).resolve()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def load_selection(path: Path) -> tuple[Asset, tuple[SourceFrame, ...]]:
    selection_path = path.expanduser().resolve()
    require(selection_path.is_file(), f"selection receipt does not exist: {selection_path}")
    digest, size = sha256_file(selection_path)
    value = read_json(selection_path)
    require(value.get("schema_version") == 1, "selection schema is invalid")
    require(value.get("kind") == SELECTION_KIND, "selection kind is invalid")
    require(value.get("status") == SELECTION_STATUS, "selection status is not review-ready")
    require(value.get("promotion_allowed") is False, "selection exceeded scoped authority")
    ordered_ids = require_list(value.get("ordered_frame_ids"), "selection.ordered_frame_ids")
    require(len(ordered_ids) == EXPECTED_FRAME_COUNT, "selection must contain exactly 25 IDs")
    require(len(set(ordered_ids)) == EXPECTED_FRAME_COUNT, "selection frame IDs are not unique")
    records = require_list(value.get("frames"), "selection.frames")
    require(len(records) == EXPECTED_FRAME_COUNT, "selection must contain exactly 25 records")
    frames: list[SourceFrame] = []
    for index, (frame_id, raw_record) in enumerate(zip(ordered_ids, records, strict=True)):
        require(
            isinstance(frame_id, str) and SAFE_ID_PATTERN.fullmatch(frame_id) is not None,
            f"selection frame ID {index} is unsafe",
        )
        record = require_dict(raw_record, f"selection.frames[{index}]")
        require(record.get("sequence_index") == index, "selection sequence indexes are invalid")
        require(record.get("frame_id") == frame_id, "selection frame order is invalid")
        outputs = require_dict(record.get("outputs"), f"selection {frame_id}.outputs")
        composite = resolve_asset(
            outputs.get("composite_rgb"),
            relative_to=selection_path.parent,
            label=f"selection {frame_id}.composite_rgb",
        )
        try:
            with Image.open(composite.path) as image:
                image.load()
                require(image.mode == "RGB", f"selection {frame_id} composite must be RGB")
                width, height = image.size
        except OSError as exc:
            raise ContractError(f"cannot decode selection composite {composite.path}") from exc
        frames.append(
            SourceFrame(
                sequence_index=index,
                frame_id=frame_id,
                composite=composite,
                width=width,
                height=height,
            )
        )
    return Asset(selection_path, digest, size), tuple(frames)


def validate_legacy_receipt(
    path: Path,
    *,
    mask_index: Asset,
    round_index: int,
    target_object_id: str,
) -> Asset:
    legacy_path = path.expanduser().resolve()
    require(legacy_path.is_file(), f"legacy SAM receipt does not exist: {legacy_path}")
    digest, size = sha256_file(legacy_path)
    value = read_json(legacy_path)
    require(value.get("schema_version") == 1, "legacy SAM receipt schema is invalid")
    require(value.get("kind") == LEGACY_KIND, "legacy SAM receipt kind is invalid")
    require(value.get("status") == LEGACY_STATUS, "legacy SAM run did not complete")
    clean_plate = require_dict(value.get("input_clean_plate"), "legacy.input_clean_plate")
    require(clean_plate.get("round") == round_index, "legacy SAM round mismatch")
    require(clean_plate.get("frame_count") == EXPECTED_FRAME_COUNT, "legacy frame_count != 25")
    claim_scope = require_string(value.get("claim_scope"), "legacy.claim_scope")
    require(target_object_id in claim_scope, "legacy claim scope does not name the target object")
    sources = require_dict(value.get("sources"), "legacy.sources")
    runner = require_dict(sources.get("runner"), "legacy.sources.runner")
    require(
        require_sha256(runner.get("sha256"), "legacy runner.sha256") == EXPECTED_RUNNER_SHA256,
        "legacy SAM runner is not audited",
    )
    checkpoint = require_dict(sources.get("checkpoint"), "legacy.sources.checkpoint")
    require(
        require_sha256(checkpoint.get("sha256"), "legacy checkpoint.sha256")
        == EXPECTED_CHECKPOINT_SHA256,
        "legacy SAM checkpoint is not audited",
    )
    require(checkpoint.get("bytes") == EXPECTED_CHECKPOINT_BYTES, "legacy checkpoint size mismatch")
    output = require_dict(
        require_dict(value.get("outputs"), "legacy.outputs").get("mask_index"),
        "legacy.outputs.mask_index",
    )
    require(output.get("sha256") == mask_index.sha256, "legacy receipt binds another mask index")
    require(
        output.get("frame_count") == EXPECTED_FRAME_COUNT,
        "legacy mask index frame_count != 25",
    )
    require(output.get("missing_images") == [], "legacy SAM run has missing images")
    return Asset(legacy_path, digest, size)


def copy_verified(source: Asset, destination: Path) -> Asset:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source.path, destination)
    digest, size = sha256_file(destination)
    require(digest == source.sha256 and size == source.size_bytes, f"copy changed {source.path}")
    return Asset(destination, digest, size)


def relative_record(asset: Asset, root: Path) -> dict[str, Any]:
    resolved_root = root.resolve()
    resolved = asset.path.resolve()
    require(resolved.is_relative_to(resolved_root), f"output escaped staging root: {resolved}")
    return asset_record(asset, path=resolved.relative_to(resolved_root).as_posix())


def prepare_destination(value: Path) -> Path:
    lexical = value.expanduser()
    require(not lexical.is_symlink(), f"output directory cannot be a symlink: {lexical}")
    destination = lexical.resolve()
    require(not destination.exists(), f"refusing to overwrite output directory: {destination}")
    forbidden = {("web", "public"), ("public", "worlds")}
    require(
        not any(pair in forbidden for pair in pairwise(destination.parts)),
        "output directory may not target canonical/live web assets",
    )
    return destination


def materialize(
    *,
    selection_receipt: Path,
    legacy_sam_run_receipt: Path,
    mask_index_path: Path,
    round_index: int,
    target_object_id: str,
    output_dir: Path,
    path_remaps: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    require(round_index > 0, "round_index must be positive")
    require(SAFE_ID_PATTERN.fullmatch(target_object_id) is not None, "target_object_id is unsafe")
    selection_asset, frames = load_selection(selection_receipt)
    mask_index_resolved = mask_index_path.expanduser().resolve()
    require(mask_index_resolved.is_file(), f"mask index does not exist: {mask_index_resolved}")
    mask_index_sha, mask_index_size = sha256_file(mask_index_resolved)
    mask_index_asset = Asset(mask_index_resolved, mask_index_sha, mask_index_size)
    legacy_asset = validate_legacy_receipt(
        legacy_sam_run_receipt,
        mask_index=mask_index_asset,
        round_index=round_index,
        target_object_id=target_object_id,
    )
    index = read_json(mask_index_resolved)
    require(index.get("missing_images") == [], "mask index has missing SAM input images")
    remaps = parse_path_remaps(path_remaps or [])
    image_root = resolve_remapped_path(
        require_string(index.get("image_root"), "mask_index.image_root"),
        relative_to=mask_index_resolved.parent,
        remaps=remaps,
    )
    require(image_root.is_dir(), f"mask index image_root does not exist: {image_root}")
    items = require_list(index.get("items"), "mask_index.items")
    require(bool(items), "mask index must contain detections")
    frame_map = {frame.frame_id: frame for frame in frames}
    detections_by_frame: dict[str, list[dict[str, Any]]] = {frame.frame_id: [] for frame in frames}
    resolved_masks: list[tuple[dict[str, Any], Asset]] = []
    sam_inputs: dict[str, Asset] = {}
    seen_detection_ids: set[str] = set()
    seen_mask_paths: dict[str, set[Path]] = {frame.frame_id: set() for frame in frames}
    seen_mask_sha256: dict[str, set[str]] = {frame.frame_id: set() for frame in frames}
    for item_index, raw_item in enumerate(items):
        item = require_dict(raw_item, f"mask_index.items[{item_index}]")
        image_value = require_string(item.get("image"), f"mask item {item_index}.image")
        frame_id = Path(image_value).stem
        require(
            frame_id in frame_map,
            f"mask item {item_index} references unknown frame {frame_id}",
        )
        require(
            Path(image_value).name == image_value,
            f"mask item {item_index}.image must be a filename, not an identity path",
        )
        sam_input_path = (image_root / image_value).resolve()
        require(
            sam_input_path.parent == image_root,
            f"mask item {item_index}.image escapes image_root",
        )
        require(
            sam_input_path.is_file(),
            f"mask item {item_index} SAM input does not exist: {sam_input_path}",
        )
        sam_input_sha, sam_input_size = sha256_file(sam_input_path)
        sam_input = Asset(sam_input_path, sam_input_sha, sam_input_size)
        frame = frame_map[frame_id]
        require(
            sam_input.sha256 == frame.composite.sha256
            and sam_input.size_bytes == frame.composite.size_bytes,
            f"mask item {item_index} SAM input differs from selected composite for {frame_id}",
        )
        try:
            with Image.open(sam_input.path) as image:
                image.load()
                require(image.mode == "RGB", f"SAM input {frame_id} must be RGB")
                require(
                    image.size == (frame.width, frame.height),
                    f"SAM input {frame_id} dimensions differ from selected composite",
                )
        except OSError as exc:
            raise ContractError(f"cannot decode SAM input {sam_input.path}") from exc
        previous_sam_input = sam_inputs.get(frame_id)
        if previous_sam_input is None:
            sam_inputs[frame_id] = sam_input
        else:
            require(
                previous_sam_input.path == sam_input.path
                and previous_sam_input.sha256 == sam_input.sha256,
                f"mask items for {frame_id} do not share one exact SAM input",
            )
        label = require_string(item.get("label"), f"mask item {item_index}.label")
        score = item.get("score")
        require(
            isinstance(score, int | float) and not isinstance(score, bool) and math.isfinite(score),
            f"mask item {item_index}.score must be finite",
        )
        mask_path = resolve_remapped_path(
            require_string(item.get("mask_path"), f"mask item {item_index}.mask_path"),
            relative_to=mask_index_resolved.parent,
            remaps=remaps,
        )
        require(mask_path.is_file(), f"mask item {item_index} does not exist: {mask_path}")
        mask_sha, mask_size = sha256_file(mask_path)
        mask_asset = Asset(mask_path, mask_sha, mask_size)
        require(
            mask_path not in seen_mask_paths[frame_id],
            f"mask item {item_index} reuses a mask path in frame {frame_id}",
        )
        require(
            mask_sha not in seen_mask_sha256[frame_id],
            f"mask item {item_index} reuses mask bytes in frame {frame_id}",
        )
        seen_mask_paths[frame_id].add(mask_path)
        seen_mask_sha256[frame_id].add(mask_sha)
        try:
            with Image.open(mask_path) as image:
                values = np.asarray(image.convert("L"), dtype=np.uint8)
        except OSError as exc:
            raise ContractError(f"cannot decode mask item {item_index}") from exc
        require(
            values.shape == (frame.height, frame.width),
            f"mask item {item_index} size mismatch",
        )
        require(bool(np.any(values > 0)), f"mask item {item_index} is empty")
        detection_id = f"{frame_id}:mask:{item_index:06d}"
        require(detection_id not in seen_detection_ids, f"duplicate detection ID {detection_id}")
        seen_detection_ids.add(detection_id)
        detection = {
            "item_index": item_index,
            "detection_id": detection_id,
            "frame_id": frame_id,
            "image": image_value,
            "label": label,
            "score": float(score),
            "mask_pixels": int(np.count_nonzero(values)),
            "mask_sha256": mask_sha,
        }
        detections_by_frame[frame_id].append(detection)
        resolved_masks.append((detection, mask_asset))
    require(
        all(detections_by_frame[frame.frame_id] for frame in frames),
        "mask index does not cover all 25 selected frames",
    )
    require(
        set(sam_inputs) == set(frame_map),
        "mask index SAM input set does not close to all 25 selected frames",
    )

    destination = prepare_destination(output_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        script_path = Path(__file__).resolve()
        script_sha, script_size = sha256_file(script_path)
        script_asset = Asset(script_path, script_sha, script_size)
        evidence = {
            "producer_script": relative_record(
                copy_verified(script_asset, staging / "evidence" / "producer_script.py"),
                staging,
            ),
            "selection_receipt": relative_record(
                copy_verified(selection_asset, staging / "evidence" / "selection_receipt.json"),
                staging,
            ),
            "legacy_sam_run_receipt": relative_record(
                copy_verified(legacy_asset, staging / "evidence" / "legacy_sam_run_receipt.json"),
                staging,
            ),
            "mask_index": relative_record(
                copy_verified(mask_index_asset, staging / "evidence" / "mask_index.json"),
                staging,
            ),
        }
        output_frames: list[dict[str, Any]] = []
        frame_set: list[dict[str, Any]] = []
        mask_outputs: dict[str, dict[str, Any]] = {}
        for detection, source_mask in resolved_masks:
            mask_output = copy_verified(
                source_mask,
                staging
                / "masks"
                / detection["frame_id"]
                / f"{int(detection['item_index']):06d}.png",
            )
            mask_outputs[detection["detection_id"]] = relative_record(mask_output, staging)
        detection_set: list[dict[str, Any]] = []
        for frame in frames:
            sam_input_output = copy_verified(
                sam_inputs[frame.frame_id],
                staging / "sam_input_frames" / f"{frame.sequence_index:04d}.png",
            )
            composite_output = copy_verified(
                frame.composite,
                staging / "accepted_composites" / f"{frame.sequence_index:04d}.png",
            )
            require(
                sam_input_output.sha256 == composite_output.sha256
                and sam_input_output.size_bytes == composite_output.size_bytes,
                f"portable SAM input/composite binding changed for {frame.frame_id}",
            )
            sam_input_record = relative_record(sam_input_output, staging)
            composite_record = relative_record(composite_output, staging)
            frame_detections = []
            for detection in detections_by_frame[frame.frame_id]:
                output_detection = {
                    **detection,
                    "mask": mask_outputs[detection["detection_id"]],
                    "source_rgb_sha256": sam_input_output.sha256,
                    "actual_sam_input_sha256": sam_input_output.sha256,
                    "accepted_composite_sha256": composite_output.sha256,
                }
                frame_detections.append(output_detection)
                detection_set.append(
                    {
                        "item_index": detection["item_index"],
                        "detection_id": detection["detection_id"],
                        "frame_id": frame.frame_id,
                        "label": detection["label"],
                        "source_rgb_sha256": sam_input_output.sha256,
                        "actual_sam_input_sha256": sam_input_output.sha256,
                        "accepted_composite_sha256": composite_output.sha256,
                        "mask_sha256": detection["mask_sha256"],
                    }
                )
            output_frames.append(
                {
                    "sequence_index": frame.sequence_index,
                    "frame_id": frame.frame_id,
                    "source_role": "actual_sam_input_equals_accepted_clean_plate_composite",
                    "actual_sam_input": sam_input_record,
                    "actual_sam_input_sha256": sam_input_output.sha256,
                    "accepted_composite": composite_record,
                    "accepted_composite_sha256": composite_output.sha256,
                    "source_rgb": sam_input_record,
                    "source_rgb_sha256": sam_input_output.sha256,
                    "detections": frame_detections,
                }
            )
            frame_set.append(
                {
                    "sequence_index": frame.sequence_index,
                    "frame_id": frame.frame_id,
                    "actual_sam_input_sha256": sam_input_output.sha256,
                    "accepted_composite_sha256": composite_output.sha256,
                }
            )
        timestamp = datetime.now(timezone.utc).isoformat()  # noqa: UP017
        receipt = {
            "schema_version": 1,
            "kind": OUTPUT_KIND,
            "status": OUTPUT_STATUS,
            "created_at": timestamp,
            "review_only": True,
            "promotion_approved": False,
            "round_index": round_index,
            "target_object_id": target_object_id,
            "frame_count": EXPECTED_FRAME_COUNT,
            "ordered_frame_ids": [frame.frame_id for frame in frames],
            "producer": {"script": evidence["producer_script"]},
            "model": {
                "family": "sam3",
                "implementation": "holi_spatial_sam3_holi_runner",
                "runner_sha256": EXPECTED_RUNNER_SHA256,
                "checkpoint": {
                    "path": "legacy://holi-spatial/checkpoints/sam3/sam3.pt",
                    "sha256": EXPECTED_CHECKPOINT_SHA256,
                    "bytes": EXPECTED_CHECKPOINT_BYTES,
                },
            },
            "evidence": evidence,
            "mask_index": evidence["mask_index"],
            "frame_records": output_frames,
            "aggregate": {
                "all_source_frames_hash_bound": True,
                "all_actual_sam_inputs_hash_bound": True,
                "all_actual_sam_inputs_match_accepted_composites": True,
                "all_detection_masks_hash_verified": True,
                "all_detection_masks_source_bound": True,
                "all_25_frames_have_detections": True,
                "source_frame_set_sha256": canonical_sha256(frame_set),
                "detection_set_sha256": canonical_sha256(detection_set),
                "detection_count": len(detection_set),
            },
        }
        receipt_path = staging / "sam_run_receipt.json"
        receipt_path.write_text(
            json.dumps(receipt, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        receipt_sha, receipt_size = sha256_file(receipt_path)
        materialization_receipt = {
            "schema_version": 1,
            "kind": "video2world.clean_plate_sam_semantic_run_materialization_receipt",
            "status": OUTPUT_STATUS,
            "sam_run_receipt": {
                "path": "sam_run_receipt.json",
                "sha256": receipt_sha,
                "size_bytes": receipt_size,
            },
            "same_parent_staging": True,
            "atomic_directory_rename": True,
            "canonical_or_live_manifest_modified": False,
        }
        (staging / "receipt.json").write_text(
            json.dumps(materialization_receipt, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        require(not destination.exists(), f"output directory appeared during run: {destination}")
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return receipt, materialization_receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-receipt", required=True, type=Path)
    parser.add_argument("--legacy-sam-run-receipt", required=True, type=Path)
    parser.add_argument("--mask-index", required=True, type=Path)
    parser.add_argument("--round-index", required=True, type=int)
    parser.add_argument("--target-object-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--path-remap", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt, _materialization = materialize(
            selection_receipt=args.selection_receipt,
            legacy_sam_run_receipt=args.legacy_sam_run_receipt,
            mask_index_path=args.mask_index,
            round_index=args.round_index,
            target_object_id=args.target_object_id,
            output_dir=args.output_dir,
            path_remaps=args.path_remap,
        )
    except (ContractError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "frame_count": receipt["frame_count"],
                "detection_count": receipt["aggregate"]["detection_count"],
                "output_dir": str(args.output_dir.expanduser().resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
