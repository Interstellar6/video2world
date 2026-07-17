#!/usr/bin/env python3
"""Materialize one identity-safe ProPainter peel round from a 3D association report.

The selected mask is resolved exclusively through a frame assignment's
``detection_id``. SAM3 filenames, ordinal suffixes, and cross-frame item order
are never used as physical object identity.
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
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import PIL
from PIL import Image, ImageFilter

SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
EXPLICIT_IDENTITY_SOURCE = "explicit --anchor object_id=PLY only"
FORBIDDEN_IDENTITY_SOURCES = {
    "SAM3 mask filename",
    "SAM3 per-frame ordinal suffix",
    "cross-frame item order",
}
MASK_COMPLETION_POLICIES = {
    "observed",
    "fill-enclosed-holes",
    "bridge-fill-cavities",
}


@dataclass(frozen=True)
class PreparedFrame:
    sequence_index: int
    frame_id: str
    detection_id: str
    assignment: dict[str, Any]
    frame_source: dict[str, Any]
    frame_path: Path
    frame_dimensions: tuple[int, int]
    mask_source: dict[str, Any]
    mask_path: Path
    mask_dimensions: tuple[int, int]
    mask_pixels: int
    mask_bbox_xyxy: list[int]


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


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def require_positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def unique_by(items: list[Any], key: str, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, raw_item in enumerate(items):
        item = require_dict(raw_item, f"{label}[{index}]")
        value = item.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label}[{index}].{key} must be a non-empty string")
        if value in result:
            raise ValueError(f"duplicate {label} {key}: {value}")
        result[value] = item
    return result


def validate_report_gate(report: dict[str, Any]) -> None:
    if report.get("schema_version") != 1:
        raise ValueError("association.schema_version must be 1")
    if report.get("status") != "passed":
        raise ValueError("association.status must be 'passed'")

    gates = require_dict(report.get("gates"), "association.gates")
    if gates.get("passed") is not True:
        raise ValueError("association.gates.passed must be true")
    if gates.get("failure_reasons") != []:
        raise ValueError("association.gates.failure_reasons must be empty")
    if gates.get("all_non_peeled_objects_have_separated_evidence") is not True:
        raise ValueError(
            "association gate all_non_peeled_objects_have_separated_evidence must pass"
        )
    if gates.get("peeled_objects_excluded_from_next_layer_targets") is not True:
        raise ValueError(
            "association gate peeled_objects_excluded_from_next_layer_targets must pass"
        )

    contract = require_dict(report.get("association_contract"), "association_contract")
    if contract.get("physical_identity_source") != EXPLICIT_IDENTITY_SOURCE:
        raise ValueError("association identity is not sourced from explicit 3D anchors")
    forbidden = set(
        require_list(contract.get("forbidden_identity_sources"), "forbidden_identity_sources")
    )
    if not FORBIDDEN_IDENTITY_SOURCES.issubset(forbidden):
        raise ValueError("association contract does not forbid filename/ordinal identity")

    thresholds = require_dict(report.get("thresholds"), "association.thresholds")
    if not thresholds:
        raise ValueError("association.thresholds must not be empty")
    declared_thresholds_sha = require_sha256(
        report.get("thresholds_sha256"), "association.thresholds_sha256"
    )
    if sha256_json(thresholds) != declared_thresholds_sha:
        raise ValueError("association.thresholds_sha256 does not match thresholds")

    sources = require_dict(report.get("sources"), "association.sources")
    declared_source_set_sha = require_sha256(
        report.get("source_set_sha256"), "association.source_set_sha256"
    )
    if sha256_json(sources) != declared_source_set_sha:
        raise ValueError("association.source_set_sha256 does not match sources")


def select_object(report: dict[str, Any], object_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if not SAFE_ID.fullmatch(object_id):
        raise ValueError(f"unsafe object id: {object_id!r}")
    objects = unique_by(
        require_list(report.get("objects"), "association.objects"),
        "object_id",
        "association.objects",
    )
    if object_id not in objects:
        raise ValueError(f"object {object_id!r} is absent from association.objects")
    object_record = objects[object_id]
    if object_record.get("peeled") is not False:
        raise ValueError(f"object {object_id!r} is already peeled")
    if object_record.get("disposition") != "accepted_next_layer_target":
        raise ValueError(f"object {object_id!r} is not an accepted_next_layer_target")
    if object_record.get("evidence_gate_applies") is not True:
        raise ValueError(f"object {object_id!r} has no applicable evidence gate")
    if object_record.get("evidence_gate_passed") is not True:
        raise ValueError(f"object {object_id!r} did not pass its evidence gate")

    next_targets = require_list(
        report.get("next_layer_target_ids"), "association.next_layer_target_ids"
    )
    if object_id not in next_targets:
        raise ValueError(f"object {object_id!r} is not listed in next_layer_target_ids")
    peeled_ids = require_list(report.get("peeled_object_ids"), "association.peeled_object_ids")
    if object_id in peeled_ids:
        raise ValueError(f"object {object_id!r} appears in peeled_object_ids")

    sources = require_dict(report.get("sources"), "association.sources")
    anchors = unique_by(
        require_list(sources.get("anchors"), "association.sources.anchors"),
        "object_id",
        "association.sources.anchors",
    )
    if object_id not in anchors:
        raise ValueError(f"object {object_id!r} has no explicit 3D anchor source")
    anchor = anchors[object_id]
    require_sha256(anchor.get("sha256"), f"anchor {object_id}.sha256")
    require_positive_int(anchor.get("bytes"), f"anchor {object_id}.bytes")
    require_positive_int(anchor.get("point_count"), f"anchor {object_id}.point_count")
    return object_record, anchor


def resolve_frame_path(
    frames_dir: Path,
    *,
    frame_id: str,
    source: dict[str, Any],
) -> Path:
    expected_sha = require_sha256(source.get("sha256"), f"frame {frame_id}.sha256")
    declared_names: list[str] = []
    for field in ("resolved_path", "declared_path"):
        value = source.get(field)
        if isinstance(value, str) and value:
            declared_names.append(Path(value).name)

    candidates: list[Path] = []
    for name in declared_names:
        candidate = frames_dir / name
        if candidate.is_file() and candidate not in candidates:
            candidates.append(candidate)
    for candidate in sorted(frames_dir.glob(f"{frame_id}.*")):
        if candidate.is_file() and candidate not in candidates:
            candidates.append(candidate)
    matching = [
        candidate.resolve() for candidate in candidates if sha256_file(candidate) == expected_sha
    ]
    unique_matching = list(dict.fromkeys(matching))
    if not unique_matching:
        raise FileNotFoundError(
            f"no frame in {frames_dir} matches frame {frame_id} SHA-256 {expected_sha}"
        )
    if len(unique_matching) != 1:
        raise ValueError(f"frame {frame_id} resolves ambiguously inside {frames_dir}")
    return unique_matching[0]


def resolve_report_path(value: Any, *, association_path: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = association_path.parent / path
    return path.resolve()


def resolve_mask_path(source: dict[str, Any], *, association_path: Path) -> Path:
    expected_sha = require_sha256(source.get("sha256"), "mask source sha256")
    candidates: list[Path] = []
    for field in ("resolved_path", "declared_path"):
        value = source.get(field)
        if not isinstance(value, str) or not value:
            continue
        candidate = resolve_report_path(
            value,
            association_path=association_path,
            label=f"mask source {field}",
        )
        if candidate.is_file() and candidate not in candidates:
            candidates.append(candidate)
    matching = [candidate for candidate in candidates if sha256_file(candidate) == expected_sha]
    if not matching:
        raise FileNotFoundError(
            f"no declared mask path matches detection {source.get('detection_id')} SHA-256"
        )
    if len(matching) != 1:
        resolved = {candidate.resolve() for candidate in matching}
        if len(resolved) != 1:
            raise ValueError(f"mask {source.get('detection_id')} resolves ambiguously")
    return matching[0].resolve()


def image_dimensions(path: Path, *, label: str) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
    except (OSError, ValueError) as error:
        raise ValueError(f"{label} is not a decodable image: {path}") from error
    if width <= 0 or height <= 0:
        raise ValueError(f"{label} has non-positive dimensions: {path}")
    return width, height


def mask_geometry(path: Path) -> tuple[tuple[int, int], int, list[int]]:
    try:
        with Image.open(path) as image:
            mask = image.convert("L").point(lambda value: 255 if value > 0 else 0)
            mask.load()
    except (OSError, ValueError) as error:
        raise ValueError(f"mask is not a decodable image: {path}") from error
    bbox = mask.getbbox()
    if bbox is None:
        raise ValueError(f"mask is empty: {path}")
    histogram = mask.histogram()
    pixels = int(sum(histogram[1:]))
    left, top, right, bottom = bbox
    # This matches associate_multiframe_instance_masks.mask_bbox: right/bottom exclusive.
    return mask.size, pixels, [left, top, right, bottom]


def image_mask_geometry(mask: Image.Image) -> tuple[tuple[int, int], int, list[int]]:
    binary = mask.convert("L").point(lambda value: 255 if value > 0 else 0)
    bbox = binary.getbbox()
    if bbox is None:
        raise ValueError("completed mask is empty")
    pixels = int(sum(binary.histogram()[1:]))
    left, top, right, bottom = bbox
    return binary.size, pixels, [left, top, right, bottom]


def pixel_bbox(indices: list[int], *, width: int) -> list[int] | None:
    if not indices:
        return None
    xs = [index % width for index in indices]
    ys = [index // width for index in indices]
    return [min(xs), min(ys), max(xs) + 1, max(ys) + 1]


def complete_mask(
    observed_mask: Image.Image,
    *,
    policy: str,
    minimum_area_pixels: int,
    minimum_area_ratio: float,
    cavity_bridge_radius: int,
    hole_connectivity: int = 4,
) -> tuple[Image.Image, dict[str, Any]]:
    if policy not in MASK_COMPLETION_POLICIES:
        raise ValueError(
            "mask_completion_policy must be one of: "
            + ", ".join(sorted(MASK_COMPLETION_POLICIES))
        )
    if (
        not isinstance(minimum_area_pixels, int)
        or isinstance(minimum_area_pixels, bool)
        or minimum_area_pixels < 0
    ):
        raise ValueError("min_enclosed_hole_area_px must be a non-negative integer")
    if (
        not isinstance(minimum_area_ratio, int | float)
        or isinstance(minimum_area_ratio, bool)
        or not math.isfinite(minimum_area_ratio)
        or not 0.0 <= minimum_area_ratio <= 1.0
    ):
        raise ValueError("min_enclosed_hole_area_ratio must be finite and in [0, 1]")
    if (
        not isinstance(cavity_bridge_radius, int)
        or isinstance(cavity_bridge_radius, bool)
        or cavity_bridge_radius < 0
    ):
        raise ValueError("cavity_bridge_radius must be a non-negative integer")
    if hole_connectivity not in {4, 8}:
        raise ValueError("hole_connectivity must be 4 or 8")

    binary = observed_mask.convert("L").point(lambda value: 255 if value > 0 else 0)
    width, height = binary.size
    frame_pixels = width * height
    effective_minimum = max(
        1,
        minimum_area_pixels,
        math.ceil(minimum_area_ratio * frame_pixels),
    )
    observed = bytearray(binary.tobytes())
    for index, value in enumerate(observed):
        observed[index] = 1 if value else 0
    completed = observed.copy()
    topology = observed
    topology_method = "observed_binary_mask"
    if policy == "bridge-fill-cavities":
        if cavity_bridge_radius == 0:
            closed = binary
        else:
            kernel_size = cavity_bridge_radius * 2 + 1
            padded = Image.new(
                "L",
                (width + cavity_bridge_radius * 2, height + cavity_bridge_radius * 2),
                0,
            )
            padded.paste(binary, (cavity_bridge_radius, cavity_bridge_radius))
            closed_padded = padded.filter(ImageFilter.MaxFilter(kernel_size)).filter(
                ImageFilter.MinFilter(kernel_size)
            )
            closed = closed_padded.crop(
                (
                    cavity_bridge_radius,
                    cavity_bridge_radius,
                    cavity_bridge_radius + width,
                    cavity_bridge_radius + height,
                )
            )
        topology = bytearray(closed.tobytes())
        for index, value in enumerate(topology):
            topology[index] = 1 if value else 0
        topology_method = "binary_closing_for_topology_only"

    components: list[dict[str, Any]] = []
    enclosed_component_count = 0
    rejected_below_threshold_count = 0
    if policy in {"fill-enclosed-holes", "bridge-fill-cavities"}:
        visited = bytearray(frame_pixels)
        for start in range(frame_pixels):
            if topology[start] or visited[start]:
                continue
            visited[start] = 1
            component = [start]
            cursor = 0
            touches_border = False
            while cursor < len(component):
                index = component[cursor]
                cursor += 1
                x = index % width
                y = index // width
                if x == 0 or y == 0 or x == width - 1 or y == height - 1:
                    touches_border = True
                min_y = max(0, y - 1)
                max_y = min(height - 1, y + 1)
                min_x = max(0, x - 1)
                max_x = min(width - 1, x + 1)
                for neighbor_y in range(min_y, max_y + 1):
                    row_offset = neighbor_y * width
                    for neighbor_x in range(min_x, max_x + 1):
                        neighbor = row_offset + neighbor_x
                        if (
                            hole_connectivity == 4
                            and neighbor_x != x
                            and neighbor_y != y
                        ):
                            continue
                        if neighbor == index or topology[neighbor] or visited[neighbor]:
                            continue
                        visited[neighbor] = 1
                        component.append(neighbor)
            if touches_border:
                continue
            enclosed_component_count += 1
            added_component = [index for index in component if not observed[index]]
            if len(added_component) < effective_minimum:
                rejected_below_threshold_count += 1
                continue
            for index in added_component:
                completed[index] = 1
            components.append(
                {
                    "area_pixels": len(added_component),
                    "bbox_xyxy_exclusive": pixel_bbox(added_component, width=width),
                }
            )

    if any(value and not completed[index] for index, value in enumerate(observed)):
        raise ValueError("completed mask does not contain every observed mask pixel")
    completed_image = Image.frombytes(
        "L",
        (width, height),
        bytes(255 if value else 0 for value in completed),
    )
    _, observed_pixels, observed_bbox = image_mask_geometry(binary)
    _, completed_pixels, completed_bbox = image_mask_geometry(completed_image)
    added_indices = [
        index for index, value in enumerate(completed) if value and not observed[index]
    ]
    added_pixels = len(added_indices)
    if completed_pixels != observed_pixels + added_pixels:
        raise ValueError("completed mask pixel accounting is inconsistent")
    return completed_image, {
        "policy": policy,
        "connectivity": hole_connectivity,
        "topology_method": topology_method,
        "cavity_bridge_radius": cavity_bridge_radius,
        "cavity_bridge_kernel_size": cavity_bridge_radius * 2 + 1,
        "closing_pixels_are_never_copied_to_completed_mask": True,
        "minimum_enclosed_hole_area": {
            "pixels": minimum_area_pixels,
            "ratio_of_frame": minimum_area_ratio,
            "effective_pixels": effective_minimum,
        },
        "observed_geometry": {
            "area_pixels": observed_pixels,
            "bbox_xyxy_exclusive": observed_bbox,
        },
        "completed_geometry": {
            "area_pixels": completed_pixels,
            "bbox_xyxy_exclusive": completed_bbox,
        },
        "observed_is_subset": True,
        "added_pixels": added_pixels,
        "added_ratio": added_pixels / observed_pixels,
        "added_ratio_denominator": "observed_mask_pixels",
        "components": components,
        "component_count": len(components),
        "bbox_xyxy_exclusive": pixel_bbox(added_indices, width=width),
        "enclosed_component_count": enclosed_component_count,
        "rejected_below_threshold_component_count": rejected_below_threshold_count,
    }


def validate_source_file(
    path: Path,
    source: dict[str, Any],
    *,
    sha_label: str,
    bytes_label: str,
) -> None:
    expected_sha = require_sha256(source.get("sha256"), sha_label)
    expected_bytes = require_positive_int(source.get("bytes"), bytes_label)
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"{bytes_label} does not match the source file")
    if sha256_file(path) != expected_sha:
        raise ValueError(f"{sha_label} does not match the source file")


def assignment_key(value: dict[str, Any]) -> tuple[str, str, str, str]:
    frame_id = value.get("frame_id")
    detection_id = value.get("detection_id")
    frame_sha = value.get("frame_sha256")
    mask_sha = value.get("mask_sha256")
    if not all(isinstance(item, str) and item for item in (frame_id, detection_id)):
        raise ValueError("assignment is missing frame_id or detection_id")
    return (
        frame_id,
        detection_id,
        require_sha256(frame_sha, f"assignment {frame_id}.frame_sha256"),
        require_sha256(mask_sha, f"assignment {frame_id}.mask_sha256"),
    )


def validate_assignment_metrics(
    assignment: dict[str, Any],
    *,
    frame_record: dict[str, Any],
) -> None:
    metrics = require_dict(assignment.get("metrics"), "assignment.metrics")
    if metrics.get("detection_id") != assignment.get("detection_id"):
        raise ValueError("assignment.metrics.detection_id does not match assignment")
    if metrics.get("eligible") is not True:
        raise ValueError("assignment metrics are not eligible")
    quality = metrics.get("quality_score")
    if (
        not isinstance(quality, int | float)
        or isinstance(quality, bool)
        or not math.isfinite(quality)
    ):
        raise ValueError("assignment quality_score must be finite")
    metric_gates = require_dict(metrics.get("gates"), "assignment.metrics.gates")
    if not metric_gates or any(value is not True for value in metric_gates.values()):
        raise ValueError("assignment metric gates did not all pass")

    matching_pairs = [
        require_dict(item, "frame pair metric")
        for item in require_list(frame_record.get("pair_metrics"), "frame.pair_metrics")
        if isinstance(item, dict)
        and item.get("object_id") == assignment.get("object_id")
        and item.get("detection_id") == assignment.get("detection_id")
    ]
    if len(matching_pairs) != 1:
        raise ValueError("assignment has no unique calibrated object/detection pair metric")
    pair = matching_pairs[0]
    if pair.get("eligible") is not True:
        raise ValueError("assignment pair metric is not eligible")
    pair_gates = require_dict(pair.get("gates"), "assignment pair metric gates")
    if not pair_gates or any(value is not True for value in pair_gates.values()):
        raise ValueError("assignment pair metric gates did not all pass")


def validate_and_resolve(
    *,
    association_path: Path,
    frames_dir: Path,
    object_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[PreparedFrame]]:
    report = require_dict(read_json(association_path), "association")
    validate_report_gate(report)
    object_record, anchor = select_object(report, object_id)
    sources = require_dict(report.get("sources"), "association.sources")
    frame_sources = unique_by(
        require_list(sources.get("frames"), "association.sources.frames"),
        "frame_id",
        "association.sources.frames",
    )
    mask_sources = unique_by(
        require_list(sources.get("masks"), "association.sources.masks"),
        "detection_id",
        "association.sources.masks",
    )
    frame_records = unique_by(
        require_list(report.get("frames"), "association.frames"),
        "frame_id",
        "association.frames",
    )
    if set(frame_records) != set(frame_sources):
        raise ValueError("association.frames and association.sources.frames cover different frames")

    anchor_ids = {
        item.get("object_id")
        for item in require_list(sources.get("anchors"), "association.sources.anchors")
        if isinstance(item, dict)
    }
    for frame_id, frame_record in frame_records.items():
        assignments = require_list(frame_record.get("assignments"), f"frame {frame_id}.assignments")
        detection_ids: set[str] = set()
        for raw_assignment in assignments:
            assignment = require_dict(raw_assignment, f"frame {frame_id} assignment")
            detection_id = assignment.get("detection_id")
            if not isinstance(detection_id, str) or not detection_id:
                raise ValueError(f"frame {frame_id} assignment has no detection_id")
            if detection_id in detection_ids:
                raise ValueError(
                    f"frame {frame_id} assigns detection {detection_id} more than once"
                )
            detection_ids.add(detection_id)
            if assignment.get("object_id") not in anchor_ids:
                raise ValueError(f"frame {frame_id} assignment has no explicit 3D anchor")

    high_confidence_assignments = require_list(
        object_record.get("all_high_confidence_assignments"),
        f"object {object_id}.all_high_confidence_assignments",
    )
    high_confidence_keys = {
        assignment_key(require_dict(item, "object high-confidence assignment"))
        for item in high_confidence_assignments
    }
    if object_record.get("high_confidence_assignment_count") != len(high_confidence_keys):
        raise ValueError("object high_confidence_assignment_count is inconsistent")

    prepared: list[PreparedFrame] = []
    for sequence_index, frame_id in enumerate(sorted(frame_records)):
        if not SAFE_ID.fullmatch(frame_id):
            raise ValueError(f"unsafe frame id: {frame_id!r}")
        frame_record = frame_records[frame_id]
        matches = [
            require_dict(item, f"frame {frame_id} assignment")
            for item in require_list(
                frame_record.get("assignments"), f"frame {frame_id}.assignments"
            )
            if isinstance(item, dict) and item.get("object_id") == object_id
        ]
        if len(matches) != 1:
            raise ValueError(
                f"frame {frame_id} must contain exactly one assignment for {object_id}; "
                f"found {len(matches)}"
            )
        assignment = matches[0]
        if assignment.get("frame_id") != frame_id:
            raise ValueError(f"frame {frame_id} assignment frame_id mismatch")
        if assignment.get("peeled_object_residual") is not False:
            raise ValueError(f"frame {frame_id} assignment is marked as a peeled residual")
        key = assignment_key(assignment)
        if key not in high_confidence_keys:
            raise ValueError(f"frame {frame_id} assignment is absent from object evidence")
        validate_assignment_metrics(assignment, frame_record=frame_record)

        frame_source = frame_sources[frame_id]
        if assignment.get("frame_sha256") != frame_source.get("sha256"):
            raise ValueError(f"frame {frame_id} assignment/source SHA-256 mismatch")
        dimensions = frame_source.get("dimensions")
        if (
            not isinstance(dimensions, list)
            or len(dimensions) != 2
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in dimensions
            )
        ):
            raise ValueError(f"frame {frame_id} source dimensions must be [width, height]")
        frame_path = resolve_frame_path(
            frames_dir,
            frame_id=frame_id,
            source=frame_source,
        )
        validate_source_file(
            frame_path,
            frame_source,
            sha_label=f"frame {frame_id}.sha256",
            bytes_label=f"frame {frame_id}.bytes",
        )
        actual_frame_dimensions = image_dimensions(frame_path, label=f"frame {frame_id}")
        if list(actual_frame_dimensions) != dimensions:
            raise ValueError(f"frame {frame_id} dimensions do not match association source")

        detection_id = assignment["detection_id"]
        if detection_id not in mask_sources:
            raise ValueError(f"frame {frame_id} detection {detection_id} has no mask source")
        mask_source = mask_sources[detection_id]
        if mask_source.get("frame_id") != frame_id:
            raise ValueError(f"mask source {detection_id} belongs to a different frame")
        if assignment.get("mask_sha256") != mask_source.get("sha256"):
            raise ValueError(f"frame {frame_id} assignment/mask SHA-256 mismatch")
        validate_source_file(
            resolve_mask_path(mask_source, association_path=association_path),
            mask_source,
            sha_label=f"mask {detection_id}.sha256",
            bytes_label=f"mask {detection_id}.bytes",
        )
        mask_path = resolve_mask_path(mask_source, association_path=association_path)
        mask_dimensions, mask_pixels, mask_bbox = mask_geometry(mask_path)
        if mask_dimensions != actual_frame_dimensions:
            raise ValueError(f"mask {detection_id} dimensions do not match frame {frame_id}")
        if mask_source.get("area_pixels") != mask_pixels:
            raise ValueError(f"mask {detection_id} area_pixels does not match decoded mask")
        if mask_source.get("actual_bbox_xyxy") != mask_bbox:
            raise ValueError(f"mask {detection_id} bbox does not match decoded mask")

        prepared.append(
            PreparedFrame(
                sequence_index=sequence_index,
                frame_id=frame_id,
                detection_id=detection_id,
                assignment=assignment,
                frame_source=frame_source,
                frame_path=frame_path,
                frame_dimensions=actual_frame_dimensions,
                mask_source=mask_source,
                mask_path=mask_path,
                mask_dimensions=mask_dimensions,
                mask_pixels=mask_pixels,
                mask_bbox_xyxy=mask_bbox,
            )
        )

    if len(high_confidence_keys) != len(prepared):
        raise ValueError("object evidence does not contain exactly one assignment per source frame")
    return report, object_record, anchor, prepared


def materialize_round(args: argparse.Namespace) -> dict[str, Any]:
    association_path = args.association.expanduser().resolve()
    frames_dir = args.frames_dir.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    mask_completion_policy = getattr(args, "mask_completion_policy", "observed")
    min_enclosed_hole_area_px = getattr(args, "min_enclosed_hole_area_px", 1)
    min_enclosed_hole_area_ratio = getattr(args, "min_enclosed_hole_area_ratio", 0.0)
    cavity_bridge_radius = getattr(args, "cavity_bridge_radius", 5)
    hole_connectivity = getattr(args, "hole_connectivity", 4)
    if not association_path.is_file():
        raise FileNotFoundError(association_path)
    if not frames_dir.is_dir():
        raise NotADirectoryError(frames_dir)
    if output_root.exists():
        raise FileExistsError(f"output directory already exists: {output_root}")

    report, object_record, anchor, prepared = validate_and_resolve(
        association_path=association_path,
        frames_dir=frames_dir,
        object_id=args.object_id,
    )
    association_sha = sha256_file(association_path)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_root.parent,
        prefix=f".{output_root.name}.staging-",
    ) as temporary_value:
        staging = Path(temporary_value)
        input_frames = staging / "input" / "frames"
        input_masks = staging / "input" / "masks"
        provenance = staging / "provenance"
        input_frames.mkdir(parents=True)
        input_masks.mkdir(parents=True)
        provenance.mkdir(parents=True)
        association_snapshot = provenance / "association.json"
        shutil.copy2(association_path, association_snapshot)
        if sha256_file(association_snapshot) != association_sha:
            raise ValueError("association snapshot SHA-256 changed during copy")

        records: list[dict[str, Any]] = []
        for item in prepared:
            output_name = f"{item.sequence_index:04d}.png"
            output_frame = input_frames / output_name
            output_mask = input_masks / output_name
            with Image.open(item.frame_path) as image:
                image.convert("RGB").save(output_frame, format="PNG")
            with Image.open(item.mask_path) as image:
                completed_mask, completion_record = complete_mask(
                    image,
                    policy=mask_completion_policy,
                    minimum_area_pixels=min_enclosed_hole_area_px,
                    minimum_area_ratio=min_enclosed_hole_area_ratio,
                    cavity_bridge_radius=cavity_bridge_radius,
                    hole_connectivity=hole_connectivity,
                )
                completed_mask.save(output_mask, format="PNG")
            if image_dimensions(output_frame, label="materialized frame") != item.frame_dimensions:
                raise ValueError(f"materialized frame dimensions changed for {item.frame_id}")
            output_mask_dimensions, output_mask_pixels, output_mask_bbox = mask_geometry(
                output_mask
            )
            observed_geometry = completion_record["observed_geometry"]
            completed_geometry = completion_record["completed_geometry"]
            if (
                observed_geometry["area_pixels"] != item.mask_pixels
                or observed_geometry["bbox_xyxy_exclusive"] != item.mask_bbox_xyxy
            ):
                raise ValueError(f"observed mask geometry changed for {item.frame_id}")
            if (
                output_mask_dimensions != item.mask_dimensions
                or output_mask_pixels != completed_geometry["area_pixels"]
                or output_mask_bbox != completed_geometry["bbox_xyxy_exclusive"]
            ):
                raise ValueError(f"completed mask geometry changed for {item.frame_id}")

            records.append(
                {
                    "sequence_index": item.sequence_index,
                    "frame_id": item.frame_id,
                    "detection_id": item.detection_id,
                    "identity_source": "explicit_3d_anchor_assignment",
                    "association_assignment": {
                        "quality_score": item.assignment.get("quality_score"),
                        "frame_sha256": item.assignment["frame_sha256"],
                        "mask_sha256": item.assignment["mask_sha256"],
                        "metric_gates": item.assignment["metrics"]["gates"],
                    },
                    "source_frame": {
                        "sha256": item.frame_source["sha256"],
                        "bytes": item.frame_source["bytes"],
                        "dimensions": list(item.frame_dimensions),
                    },
                    "source_mask": {
                        "sha256": item.mask_source["sha256"],
                        "bytes": item.mask_source["bytes"],
                        "dimensions": list(item.mask_dimensions),
                        "area_pixels": item.mask_pixels,
                        "bbox_xyxy_exclusive": item.mask_bbox_xyxy,
                        "geometry_role": "observed_sam3_mask",
                    },
                    "input_frame": {
                        "path": f"input/frames/{output_name}",
                        "sha256": sha256_file(output_frame),
                        "bytes": output_frame.stat().st_size,
                        "dimensions": list(item.frame_dimensions),
                    },
                    "input_mask": {
                        "path": f"input/masks/{output_name}",
                        "sha256": sha256_file(output_mask),
                        "bytes": output_mask.stat().st_size,
                        "dimensions": list(item.mask_dimensions),
                        "area_pixels": output_mask_pixels,
                        "bbox_xyxy_exclusive": output_mask_bbox,
                        "geometry_role": "completed_propainter_mask",
                    },
                    "mask_completion": completion_record,
                }
            )

        observed_pixel_total = sum(
            item["mask_completion"]["observed_geometry"]["area_pixels"] for item in records
        )
        completed_pixel_total = sum(
            item["mask_completion"]["completed_geometry"]["area_pixels"] for item in records
        )
        added_pixel_total = sum(item["mask_completion"]["added_pixels"] for item in records)
        aggregate_components = [
            {
                "frame_id": item["frame_id"],
                "component_index": component_index,
                **component,
            }
            for item in records
            for component_index, component in enumerate(
                item["mask_completion"]["components"]
            )
        ]
        mask_completion_manifest = {
            "policy": mask_completion_policy,
            "connectivity": hole_connectivity,
            "topology_method": (
                "binary_closing_for_topology_only"
                if mask_completion_policy == "bridge-fill-cavities"
                else "observed_binary_mask"
            ),
            "cavity_bridge_radius": cavity_bridge_radius,
            "cavity_bridge_kernel_size": cavity_bridge_radius * 2 + 1,
            "closing_pixels_are_never_copied_to_completed_mask": True,
            "minimum_enclosed_hole_area": {
                "pixels": min_enclosed_hole_area_px,
                "ratio_of_frame": min_enclosed_hole_area_ratio,
                "effective_pixels_are_computed_per_frame": True,
            },
            "observed_pixels": observed_pixel_total,
            "completed_pixels": completed_pixel_total,
            "added_pixels": added_pixel_total,
            "added_ratio": added_pixel_total / observed_pixel_total,
            "added_ratio_denominator": "observed_mask_pixels",
            "components": aggregate_components,
            "component_count": len(aggregate_components),
            "bbox_xyxy_exclusive_by_frame": {
                item["frame_id"]: item["mask_completion"]["bbox_xyxy_exclusive"]
                for item in records
            },
            "observed_is_subset_in_every_frame": all(
                item["mask_completion"]["observed_is_subset"] for item in records
            ),
        }

        prepared_source_set = {
            "object_id": args.object_id,
            "previous_clean_plate_source_set_sha256": report["source_set_sha256"],
            "frames": [
                {
                    "frame_id": item["frame_id"],
                    "detection_id": item["detection_id"],
                    "input_frame_sha256": item["input_frame"]["sha256"],
                    "input_mask_sha256": item["input_mask"]["sha256"],
                }
                for item in records
            ],
        }
        manifest = {
            "schema_version": 1,
            "status": "ready_for_propainter",
            "created_at": datetime.now(UTC).isoformat(),
            "purpose": "identity-safe front-to-back object peel input",
            "method": (
                "explicit 3D anchor assignment -> detection_id -> exact SAM3 mask -> "
                "optional enclosed-hole completion"
            ),
            "object_id": args.object_id,
            "object_gate": {
                "disposition": object_record["disposition"],
                "evidence_gate_passed": object_record["evidence_gate_passed"],
                "explicit_anchor_sha256": anchor["sha256"],
                "explicit_anchor_point_count": anchor["point_count"],
            },
            "identity_contract": {
                "physical_identity_source": EXPLICIT_IDENTITY_SOURCE,
                "selected_mask_key": "association.frames[].assignments[].detection_id",
                "forbidden_identity_sources": sorted(FORBIDDEN_IDENTITY_SOURCES),
            },
            "previous_clean_plate_source_set_sha256": report["source_set_sha256"],
            "association": {
                "path": "provenance/association.json",
                "sha256": association_sha,
                "schema_version": report["schema_version"],
                "thresholds": report["thresholds"],
                "thresholds_sha256": report["thresholds_sha256"],
                "source_set_sha256": report["source_set_sha256"],
            },
            "portable_layout": {
                "paths_are_relative_to": "input_manifest.json parent",
                "frames_dir": "input/frames",
                "masks_dir": "input/masks",
                "association_snapshot": "provenance/association.json",
            },
            "frame_count": len(records),
            "frame_ids": [item["frame_id"] for item in records],
            "frame_records": records,
            "mask_completion": mask_completion_manifest,
            "prepared_source_set_sha256": sha256_json(prepared_source_set),
            "gates": {
                "association_passed": True,
                "object_accepted_next_layer_target": True,
                "explicit_3d_anchor_present": True,
                "one_assignment_per_frame": True,
                "assignment_and_source_hashes_verified": True,
                "frame_and_mask_dimensions_verified": True,
                "observed_mask_subset_of_completed_mask": True,
                "portable_assets_copied": True,
            },
            "execution": {
                "script": {
                    "path": Path(__file__).name,
                    "sha256": sha256_file(Path(__file__).resolve()),
                },
                "argv": list(sys.argv),
                "python": platform.python_version(),
                "pillow": PIL.__version__,
                "copy_mode": "decoded PNG copy; no symlink dependency",
            },
        }
        manifest_path = staging / "input_manifest.json"
        write_json(manifest_path, manifest)
        write_json(
            staging / "input_manifest.json.sha256.json",
            {
                "schema_version": 1,
                "manifest_path": "input_manifest.json",
                "manifest_sha256": sha256_file(manifest_path),
                "manifest_bytes": manifest_path.stat().st_size,
            },
        )
        os.replace(staging, output_root)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--association", required=True, type=Path)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--frames-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--mask-completion-policy",
        choices=sorted(MASK_COMPLETION_POLICIES),
        default="observed",
        help="Keep the observed SAM3 mask or fill only enclosed background components.",
    )
    parser.add_argument(
        "--min-enclosed-hole-area-px",
        type=int,
        default=1,
        help="Minimum enclosed-hole area in pixels; combined with the ratio using max().",
    )
    parser.add_argument(
        "--min-enclosed-hole-area-ratio",
        type=float,
        default=0.0,
        help="Minimum enclosed-hole area as a ratio of frame pixels; combined using max().",
    )
    parser.add_argument(
        "--cavity-bridge-radius",
        type=int,
        default=5,
        help=(
            "Square binary-closing radius used only to classify narrow-mouth cavities; "
            "closing contour pixels are never copied into the output mask."
        ),
    )
    parser.add_argument(
        "--hole-connectivity",
        type=int,
        choices=(4, 8),
        default=4,
        help="Background connectivity used by hole filling; 4 matches scipy defaults.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        manifest = materialize_round(args)
    except (FileNotFoundError, NotADirectoryError, FileExistsError, ValueError) as error:
        parser.exit(2, f"peel-round preparation failed closed: {error}\n")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "object_id": manifest["object_id"],
                "frame_count": manifest["frame_count"],
                "manifest": str(args.output_dir / "input_manifest.json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
