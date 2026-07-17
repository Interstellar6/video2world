#!/usr/bin/env python3
"""Build auditable per-object RGBA inputs from geometric multi-frame evidence.

The association report is the only identity authority. This script joins a
selected evidence record to ``sources.masks`` by the opaque ``detection_id``;
mask filenames, SAM3 ordinal suffixes, and list order never determine identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import PIL
    from PIL import Image
except ImportError as exc:  # pragma: no cover - exercised by the CLI error path
    raise SystemExit(
        "prepare_layered_object_input.py requires numpy and Pillow. "
        "Install the geometry extras with: python -m pip install -e '.[geometry]'"
    ) from exc


SCHEMA_VERSION = 1
OBJECT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
EXPECTED_FORBIDDEN_IDENTITY_SOURCES = {
    "SAM3 mask filename",
    "SAM3 per-frame ordinal suffix",
    "cross-frame item order",
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def require_object(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def require_unique_index(
    values: Any,
    *,
    key: str,
    description: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(values, list):
        raise ValueError(f"{description} must be an array")
    result: dict[str, dict[str, Any]] = {}
    for item in values:
        record = require_object(item, f"each {description} entry")
        identifier = str(record.get(key, "")).strip()
        if not identifier:
            raise ValueError(f"each {description} entry requires {key}")
        if identifier in result:
            raise ValueError(f"duplicate {description} {key}: {identifier}")
        result[identifier] = record
    return result


def normalize_category(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").replace("-", " ").split())


def resolve_reported_path(value: Any, *, relative_to: Path, description: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{description} requires a non-empty resolved_path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def resolve_frame_path(frames_dir: Path, source: dict[str, Any], frame_id: str) -> Path:
    resolved_value = source.get("resolved_path")
    if not isinstance(resolved_value, str) or not resolved_value.strip():
        raise ValueError(f"frame source {frame_id} requires resolved_path")
    reported_name = Path(resolved_value).name
    if not reported_name:
        raise ValueError(f"frame source {frame_id} has an invalid resolved_path")
    candidates = [frames_dir / reported_name]
    candidates.extend(
        child for child in frames_dir.iterdir() if child.is_file() and child.stem == frame_id
    )
    unique_candidates = sorted({path.absolute() for path in candidates if path.is_file()})
    if not unique_candidates:
        raise FileNotFoundError(
            f"no RGB file for frame {frame_id} or reported basename {reported_name} "
            f"under {frames_dir}"
        )
    declared_hash = source.get("sha256")
    matching = [path for path in unique_candidates if sha256_file(path) == declared_hash]
    if not matching:
        raise ValueError(
            f"frame {frame_id} sha256 mismatch: no RGB candidate under --frames-dir "
            "matches the declared source hash"
        )
    return matching[0]


def verify_hashed_file(
    path: Path,
    source: dict[str, Any],
    *,
    expected_hash: Any,
    description: str,
) -> str:
    declared_hash = source.get("sha256")
    if not isinstance(declared_hash, str) or len(declared_hash) != 64:
        raise ValueError(f"{description} has no valid source sha256")
    if expected_hash != declared_hash:
        raise ValueError(f"{description} selected-evidence hash disagrees with sources")
    actual_hash = sha256_file(path)
    if actual_hash != declared_hash:
        raise ValueError(
            f"{description} sha256 mismatch: expected {declared_hash}, got {actual_hash}"
        )
    declared_bytes = source.get("bytes")
    if not isinstance(declared_bytes, int) or declared_bytes != path.stat().st_size:
        raise ValueError(f"{description} byte count does not match sources")
    return actual_hash


def vector(value: Any, description: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{description} must contain three finite numbers")
    return array


def view_angle_degrees(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        raise ValueError("selected evidence contains a zero-length object view direction")
    cosine = float(np.dot(left, right) / (left_norm * right_norm))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def validate_separation(
    selected: list[dict[str, Any]],
    separation: Any,
    thresholds: dict[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(separation, list):
        raise ValueError("selected_frame_separation must be an array")
    minimum_baseline = float(thresholds.get("minimum_camera_baseline_scene_units"))
    minimum_angle = float(thresholds.get("minimum_object_view_angle_degrees"))
    mode = thresholds.get("separation_mode")
    if mode not in {"either", "both"}:
        raise ValueError("thresholds.separation_mode must be either or both")
    if minimum_baseline < 0 or minimum_angle < 0:
        raise ValueError("separation thresholds must be non-negative")

    by_frame = {str(item["frame_id"]): item for item in selected}
    expected_pairs = {
        frozenset((left, right))
        for left_index, left in enumerate(by_frame)
        for right in list(by_frame)[left_index + 1 :]
    }
    seen_pairs: set[frozenset[str]] = set()
    validated: list[dict[str, Any]] = []
    for raw_pair in separation:
        pair = require_object(raw_pair, "each selected-frame separation record")
        left_id = str(pair.get("left_frame_id", ""))
        right_id = str(pair.get("right_frame_id", ""))
        key = frozenset((left_id, right_id))
        if left_id == right_id or key not in expected_pairs or key in seen_pairs:
            raise ValueError("selected_frame_separation does not exactly cover selected frames")
        seen_pairs.add(key)
        left = by_frame[left_id]
        right = by_frame[right_id]
        left_center = vector(left.get("camera_center_world"), f"camera center for {left_id}")
        right_center = vector(right.get("camera_center_world"), f"camera center for {right_id}")
        left_view = vector(left.get("object_view_direction_world"), f"view direction for {left_id}")
        right_view = vector(
            right.get("object_view_direction_world"), f"view direction for {right_id}"
        )
        baseline = float(np.linalg.norm(left_center - right_center))
        angle = view_angle_degrees(left_view, right_view)
        baseline_passed = baseline >= minimum_baseline
        angle_passed = angle >= minimum_angle
        passed = (
            baseline_passed and angle_passed
            if mode == "both"
            else baseline_passed or angle_passed
        )
        recorded_baseline = pair.get("camera_baseline_scene_units")
        recorded_angle = pair.get("object_view_angle_degrees")
        if not isinstance(recorded_baseline, int | float) or not math.isclose(
            float(recorded_baseline), baseline, rel_tol=1e-7, abs_tol=1e-7
        ):
            raise ValueError(f"separation baseline was not reproduced for {left_id}/{right_id}")
        if not isinstance(recorded_angle, int | float) or not math.isclose(
            float(recorded_angle), angle, rel_tol=1e-7, abs_tol=1e-7
        ):
            raise ValueError(f"separation view angle was not reproduced for {left_id}/{right_id}")
        if (
            pair.get("baseline_passed") is not baseline_passed
            or pair.get("view_angle_passed") is not angle_passed
            or pair.get("passed") is not passed
            or pair.get("separation_mode") != mode
            or not passed
        ):
            raise ValueError(f"selected frame pair {left_id}/{right_id} fails separation gates")
        validated.append(
            {
                "left_frame_id": left_id,
                "right_frame_id": right_id,
                "camera_baseline_scene_units": baseline,
                "object_view_angle_degrees": angle,
                "passed": True,
            }
        )
    if seen_pairs != expected_pairs:
        raise ValueError("selected_frame_separation is incomplete")
    return validated


def actual_mask_bbox(mask: np.ndarray) -> list[int]:
    rows, columns = np.nonzero(mask)
    if not len(rows):
        raise ValueError("selected source mask is empty")
    return [
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    ]


def validate_bbox(value: Any, *, width: int, height: int, description: str) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(not isinstance(item, int) for item in value)
    ):
        raise ValueError(f"{description} must contain four integers")
    x0, y0, x1, y1 = value
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"{description} is outside the source frame")
    return value


def padded_crop_bbox(
    bbox: list[int],
    *,
    width: int,
    height: int,
    padding_ratio: float,
    minimum_padding_pixels: int,
) -> list[int]:
    x0, y0, x1, y1 = bbox
    pad_x = max(minimum_padding_pixels, math.ceil((x1 - x0) * padding_ratio))
    pad_y = max(minimum_padding_pixels, math.ceil((y1 - y0) * padding_ratio))
    return [max(0, x0 - pad_x), max(0, y0 - pad_y), min(width, x1 + pad_x), min(height, y1 + pad_y)]


def safe_frame_stem(frame_id: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", frame_id).strip("._-") or "frame"
    suffix = hashlib.sha256(frame_id.encode("utf-8")).hexdigest()[:10]
    return f"{readable}_{suffix}"


def validate_association_receipt(association_path: Path) -> dict[str, Any]:
    receipt_path = association_path.with_suffix(association_path.suffix + ".sha256.json")
    result: dict[str, Any] = {"path": str(receipt_path), "present": receipt_path.is_file()}
    if not receipt_path.is_file():
        return result
    receipt = require_object(read_json(receipt_path), "association hash receipt")
    actual_hash = sha256_file(association_path)
    if receipt.get("report_sha256") != actual_hash:
        raise ValueError("association hash receipt does not match association.json")
    if receipt.get("report_bytes") != association_path.stat().st_size:
        raise ValueError("association hash receipt byte count does not match association.json")
    return {**result, "verified": True, "sha256": sha256_file(receipt_path)}


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    association_path = args.association.expanduser().resolve()
    frames_dir = args.frames_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not association_path.is_file():
        raise FileNotFoundError(association_path)
    if not frames_dir.is_dir():
        raise FileNotFoundError(frames_dir)
    if not OBJECT_ID_PATTERN.fullmatch(args.object_id):
        raise ValueError("--object-id must match [A-Za-z0-9][A-Za-z0-9_.-]*")
    if not 0 <= args.padding_ratio <= 1:
        raise ValueError("--padding-ratio must be between 0 and 1")
    if args.min_padding_pixels < 0:
        raise ValueError("--min-padding-pixels must be non-negative")

    report = require_object(read_json(association_path), "association report")
    gates = require_object(report.get("gates"), "association gates")
    if report.get("schema_version") != 1 or report.get("status") != "passed":
        raise ValueError("association report must have schema_version=1 and status=passed")
    if (
        gates.get("passed") is not True
        or gates.get("failure_reasons") != []
        or gates.get("all_non_peeled_objects_have_separated_evidence") is not True
        or gates.get("peeled_objects_excluded_from_next_layer_targets") is not True
    ):
        raise ValueError("association report gates did not pass cleanly")
    receipt_audit = validate_association_receipt(association_path)

    contract = require_object(report.get("association_contract"), "association contract")
    forbidden_sources = set(contract.get("forbidden_identity_sources", []))
    if (
        contract.get("physical_identity_source") != "explicit --anchor object_id=PLY only"
        or not forbidden_sources >= EXPECTED_FORBIDDEN_IDENTITY_SOURCES
    ):
        raise ValueError(
            "association report does not carry the required physical identity contract"
        )

    sources = require_object(report.get("sources"), "association sources")
    declared_source_set_hash = report.get("source_set_sha256")
    if declared_source_set_hash != sha256_json(sources):
        raise ValueError("association source_set_sha256 does not match sources")
    frame_sources = require_unique_index(
        sources.get("frames"), key="frame_id", description="frames"
    )
    mask_sources = require_unique_index(
        sources.get("masks"), key="detection_id", description="masks"
    )
    objects = require_unique_index(report.get("objects"), key="object_id", description="objects")
    if args.object_id not in objects:
        raise ValueError(f"object {args.object_id} is absent from association report")
    object_record = objects[args.object_id]
    if (
        object_record.get("peeled") is not False
        or object_record.get("disposition") != "accepted_next_layer_target"
        or object_record.get("evidence_gate_applies") is not True
        or object_record.get("evidence_gate_passed") is not True
        or args.object_id not in report.get("next_layer_target_ids", [])
    ):
        raise ValueError(f"object {args.object_id} is not an accepted next-layer target")

    raw_selected = object_record.get("selected_evidence")
    if not isinstance(raw_selected, list) or len(raw_selected) < 2:
        raise ValueError("accepted object requires at least two selected evidence frames")
    if object_record.get("selected_evidence_frame_count") != len(raw_selected):
        raise ValueError("selected_evidence_frame_count does not match selected_evidence")
    selected = [require_object(item, "each selected evidence record") for item in raw_selected]
    frame_ids = [str(item.get("frame_id", "")) for item in selected]
    detection_ids = [str(item.get("detection_id", "")) for item in selected]
    if not all(frame_ids) or len(set(frame_ids)) != len(frame_ids):
        raise ValueError("selected evidence frame ids must be non-empty and unique")
    if not all(detection_ids) or len(set(detection_ids)) != len(detection_ids):
        raise ValueError("selected evidence detection ids must be non-empty and unique")
    minimum_frames = require_object(report.get("thresholds"), "association thresholds").get(
        "minimum_evidence_frames"
    )
    if not isinstance(minimum_frames, int) or minimum_frames < 2 or len(selected) < minimum_frames:
        raise ValueError("selected evidence does not satisfy the declared minimum frame count")

    labels: set[str] = set()
    for item in selected:
        quality = item.get("quality_score")
        metrics = require_object(item.get("metrics"), "selected evidence metrics")
        metric_gates = require_object(metrics.get("gates"), "selected evidence metric gates")
        if (
            item.get("object_id") != args.object_id
            or item.get("peeled_object_residual") is not False
            or metrics.get("eligible") is not True
            or not metric_gates
            or any(value is not True for value in metric_gates.values())
        ):
            raise ValueError("selected evidence does not carry clean geometric-association gates")
        if not isinstance(quality, int | float) or not math.isfinite(float(quality)):
            raise ValueError("selected evidence quality_score must be finite")
        if not math.isclose(float(quality), float(metrics.get("quality_score")), abs_tol=1e-12):
            raise ValueError("selected evidence quality_score disagrees with metrics")
        label = normalize_category(str(item.get("label", "")))
        if not label:
            raise ValueError("selected evidence requires a non-empty category label")
        labels.add(label)
    if len(labels) != 1:
        raise ValueError("selected evidence category labels disagree across frames")
    category = next(iter(labels))
    thresholds = require_object(report.get("thresholds"), "association thresholds")
    separation_audit = validate_separation(
        selected,
        object_record.get("selected_frame_separation"),
        thresholds,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    references_dir = output_dir / "references"
    references_dir.mkdir(parents=True, exist_ok=True)
    selected_outputs: list[dict[str, Any]] = []
    for evidence in selected:
        frame_id = str(evidence["frame_id"])
        detection_id = str(evidence["detection_id"])
        if frame_id not in frame_sources or detection_id not in mask_sources:
            raise ValueError("selected evidence has no exact sources.frames/sources.masks record")
        frame_source = frame_sources[frame_id]
        mask_source = mask_sources[detection_id]
        if mask_source.get("frame_id") != frame_id:
            raise ValueError("mask detection_id resolves to a different frame")
        if normalize_category(str(mask_source.get("label", ""))) != category:
            raise ValueError("mask source label disagrees with selected evidence")
        frame_path = resolve_frame_path(frames_dir, frame_source, frame_id)
        mask_path = resolve_reported_path(
            mask_source.get("resolved_path"),
            relative_to=association_path.parent,
            description=f"mask source {detection_id}",
        )
        frame_hash = verify_hashed_file(
            frame_path,
            frame_source,
            expected_hash=evidence.get("frame_sha256"),
            description=f"frame {frame_id}",
        )
        mask_hash = verify_hashed_file(
            mask_path,
            mask_source,
            expected_hash=evidence.get("mask_sha256"),
            description=f"mask {detection_id}",
        )
        with Image.open(frame_path) as image_handle:
            rgb = np.asarray(image_handle.convert("RGB"), dtype=np.uint8)
        with Image.open(mask_path) as mask_handle:
            mask = np.asarray(mask_handle.convert("L"), dtype=np.uint8) > 0
        if mask.shape != rgb.shape[:2]:
            raise ValueError(f"mask {detection_id} dimensions do not match frame {frame_id}")
        dimensions = frame_source.get("dimensions")
        if dimensions != [rgb.shape[1], rgb.shape[0]]:
            raise ValueError(f"frame {frame_id} dimensions do not match sources")
        bbox = actual_mask_bbox(mask)
        declared_bbox = validate_bbox(
            mask_source.get("actual_bbox_xyxy"),
            width=rgb.shape[1],
            height=rgb.shape[0],
            description=f"mask {detection_id} actual_bbox_xyxy",
        )
        if declared_bbox != bbox or mask_source.get("area_pixels") != int(mask.sum()):
            raise ValueError(f"mask {detection_id} decoded geometry does not match sources")
        crop_bbox = padded_crop_bbox(
            bbox,
            width=rgb.shape[1],
            height=rgb.shape[0],
            padding_ratio=args.padding_ratio,
            minimum_padding_pixels=args.min_padding_pixels,
        )
        x0, y0, x1, y1 = crop_bbox
        crop_rgb = rgb[y0:y1, x0:x1].copy()
        crop_mask = mask[y0:y1, x0:x1]
        crop_rgb[~crop_mask] = 0
        rgba = np.dstack([crop_rgb, crop_mask.astype(np.uint8) * 255]).astype(np.uint8)
        stem = safe_frame_stem(frame_id)
        rgba_path = references_dir / f"{stem}.rgba.png"
        mask_output_path = references_dir / f"{stem}.mask.png"
        Image.fromarray(rgba).save(rgba_path)
        Image.fromarray(crop_mask.astype(np.uint8) * 255).save(mask_output_path)
        selected_outputs.append(
            {
                "frame_id": frame_id,
                "detection_id": detection_id,
                "quality_score": float(evidence["quality_score"]),
                "source_frame": {
                    "path": str(frame_path),
                    "sha256": frame_hash,
                    "bytes": frame_path.stat().st_size,
                },
                "source_mask": {
                    "path": str(mask_path),
                    "sha256": mask_hash,
                    "bytes": mask_path.stat().st_size,
                },
                "object_bbox_xyxy": bbox,
                "crop_bbox_xyxy": crop_bbox,
                "crop_dimensions": [x1 - x0, y1 - y0],
                "foreground_pixels": int(crop_mask.sum()),
                "rgba_path": rgba_path,
                "mask_output_path": mask_output_path,
            }
        )

    source = sorted(
        selected_outputs,
        key=lambda item: (-item["quality_score"], item["frame_id"], item["detection_id"]),
    )[0]
    source_rgba_path = output_dir / "source_rgba.png"
    shutil.copyfile(source["rgba_path"], source_rgba_path)
    manifest_path = output_dir / "scene_reference_manifest.json"
    manifest_frames: list[dict[str, Any]] = []
    for item in sorted(
        selected_outputs,
        key=lambda value: (value["frame_id"], value["detection_id"]),
    ):
        rgba_path = item["rgba_path"]
        mask_output_path = item["mask_output_path"]
        width, height = item["crop_dimensions"]
        manifest_frames.append(
            {
                "frame_id": item["frame_id"],
                "detection_id": item["detection_id"],
                "image_path": str(rgba_path.relative_to(output_dir)),
                "image_sha256": sha256_file(rgba_path),
                "mask_path": str(mask_output_path.relative_to(output_dir)),
                "mask_sha256": sha256_file(mask_output_path),
                "bbox_xyxy": [0, 0, width, height],
                "source_scene_crop_bbox_xyxy": item["crop_bbox_xyxy"],
                "quality_score": item["quality_score"],
                "is_source_frame": item is source,
            }
        )
    manifest = {
        "schema_version": 1,
        "kind": "video2world.layered_object_scene_references",
        "object_id": args.object_id,
        "category": category,
        "identity_revalidated": True,
        "identity_revalidated_by": "explicit_3d_anchor_calibrated_multiframe_association",
        "expected_appearance": {"color_family": args.expected_color_family},
        "association": {
            "path": str(association_path),
            "sha256": sha256_file(association_path),
            "source_set_sha256": declared_source_set_hash,
            "selected_evidence_only": True,
        },
        "frames": manifest_frames,
    }
    write_json_atomic(manifest_path, manifest)

    artifact_records: dict[str, Any] = {
        "source_rgba": {
            "path": str(source_rgba_path),
            "sha256": sha256_file(source_rgba_path),
            "bytes": source_rgba_path.stat().st_size,
        },
        "scene_reference_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "bytes": manifest_path.stat().st_size,
        },
        "evidence_frames": [
            {
                "frame_id": item["frame_id"],
                "detection_id": item["detection_id"],
                "rgba": {
                    "path": str(item["rgba_path"]),
                    "sha256": sha256_file(item["rgba_path"]),
                    "bytes": item["rgba_path"].stat().st_size,
                },
                "mask": {
                    "path": str(item["mask_output_path"]),
                    "sha256": sha256_file(item["mask_output_path"]),
                    "bytes": item["mask_output_path"].stat().st_size,
                },
                "source_frame": item["source_frame"],
                "source_mask": item["source_mask"],
                "object_bbox_xyxy": item["object_bbox_xyxy"],
                "crop_bbox_xyxy": item["crop_bbox_xyxy"],
                "crop_dimensions": item["crop_dimensions"],
                "quality_score": item["quality_score"],
            }
            for item in selected_outputs
        ],
    }
    output_report_path = output_dir / "layered_object_input_report.json"
    output_report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "video2world.layered_object_input",
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 - mil8 uses 3.10
        "object_id": args.object_id,
        "category": category,
        "source_evidence_frame_id": source["frame_id"],
        "source_evidence_detection_id": source["detection_id"],
        "association": {
            "path": str(association_path),
            "sha256": sha256_file(association_path),
            "bytes": association_path.stat().st_size,
            "receipt": receipt_audit,
            "source_set_sha256": declared_source_set_hash,
        },
        "identity_contract": {
            "authority": "association.objects[].selected_evidence[].detection_id",
            "mask_lookup": "exact equality join against association.sources.masks[].detection_id",
            "forbidden_identity_sources": sorted(EXPECTED_FORBIDDEN_IDENTITY_SOURCES),
        },
        "crop_policy": {
            "mode": "independent_per_selected_bbox",
            "padding_ratio": args.padding_ratio,
            "minimum_padding_pixels": args.min_padding_pixels,
            "clamped_to_frame": True,
            "resized": False,
        },
        "selected_frame_separation": separation_audit,
        "gates": {
            "association_status_passed": True,
            "association_gates_passed": True,
            "object_is_accepted_next_layer_target": True,
            "minimum_two_selected_evidence_frames": len(selected_outputs) >= 2,
            "all_selected_frame_pairs_separated": all(
                pair["passed"] for pair in separation_audit
            ),
            "association_source_set_hash_verified": True,
            "all_selected_frame_and_mask_hashes_verified": True,
            "identity_join_uses_detection_id_only": True,
        },
        "artifacts": artifact_records,
        "execution": {
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "argv": list(sys.argv),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": PIL.__version__,
        },
        "output_receipt": {
            "path": str(output_report_path.with_suffix(output_report_path.suffix + ".sha256.json")),
            "note": (
                "The sidecar hashes this report after serialization to avoid a recursive self-hash."
            ),
        },
    }
    write_json_atomic(output_report_path, output_report)
    receipt_path = output_report_path.with_suffix(output_report_path.suffix + ".sha256.json")
    write_json_atomic(
        receipt_path,
        {
            "schema_version": 1,
            "report_path": str(output_report_path),
            "report_sha256": sha256_file(output_report_path),
            "report_bytes": output_report_path.stat().st_size,
            "source_rgba_sha256": sha256_file(source_rgba_path),
            "scene_reference_manifest_sha256": sha256_file(manifest_path),
        },
    )
    return output_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--association", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--padding-ratio", type=float, default=0.12)
    parser.add_argument("--min-padding-pixels", type=int, default=8)
    parser.add_argument(
        "--expected-color-family",
        choices=("any", "light_white_offwhite"),
        default="any",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = prepare(args)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
