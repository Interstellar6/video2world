#!/usr/bin/env python3
"""Prepare auditable amodal RGBA inputs for single-image 3D reconstruction.

The script is deliberately conservative. ``auto`` applies a category profile only
when the category has a known silhouette prior. Unknown categories are preserved
instead of being forced into a rectangle or convex hull. The soft-rectangle profile
is intended for pillows and cushions whose observed mask contains a deep occlusion
notch. Multi-depth structures such as beds are preserved by ``auto`` because a 2D
alpha cavity can mix hidden solid surfaces with real negative space. Their holes may
only be filled with the explicit ``fill_enclosed_holes`` policy after depth-aware
review; exterior bridge pixels are never added.

Optional runtime dependencies:

    python -m pip install numpy Pillow scipy
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import PIL
    import scipy
    from PIL import Image, ImageDraw, ImageOps
    from scipy import ndimage
except ImportError as exc:  # pragma: no cover - exercised by the CLI error path
    raise SystemExit(
        "prepare_amodal_object_rgba.py requires numpy, Pillow, and scipy. "
        "Install them in an isolated environment with: "
        "python -m pip install numpy Pillow scipy"
    ) from exc


SCHEMA_VERSION = 1
PROVIDER_NAME = "video2world.amodal_rgba.soft_rectangle_nearest_v1"
SOFT_RECTANGLE_CATEGORIES = {
    "bed pillow",
    "cushion",
    "decorative pillow",
    "pillow",
    "seat cushion",
    "throw pillow",
}
MULTI_DEPTH_STRUCTURE_CATEGORIES = {
    "bed",
    "bed frame",
    "headboard",
}
SHAPE_POLICIES = {"auto", "fill_enclosed_holes", "preserve", "soft_rectangle"}


@dataclass(frozen=True)
class CompletionConfig:
    object_id: str
    category: str
    attempt: int
    shape_policy: str
    alpha_threshold: int
    corner_radius_ratio: float
    min_candidate_add_ratio: float
    min_occlusion_depth_ratio: float
    min_occlusion_depth_px: int
    min_component_area_ratio: float
    cavity_bridge_radius: int
    seed: int
    provider: str
    retry_prompt: str
    scene_reference_manifest: str
    scene_reference_manifest_sha256: str


@dataclass(frozen=True)
class CompletionResult:
    source_rgba: np.ndarray
    original_alpha: np.ndarray
    completed_alpha: np.ndarray
    original_mask: np.ndarray
    completed_mask: np.ndarray
    added_mask: np.ndarray
    completed_rgba: np.ndarray
    shape_metrics: dict[str, Any]
    color_metrics: dict[str, Any]
    gates: dict[str, bool]
    method_trace: dict[str, Any]


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalized_category(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").replace("-", " ").split())


def resolve_shape_policy(category: str, requested: str) -> tuple[str, str]:
    if requested not in SHAPE_POLICIES:
        raise ValueError(f"unsupported shape policy: {requested}")
    if requested != "auto":
        return requested, "explicit_cli_policy"
    if normalized_category(category) in SOFT_RECTANGLE_CATEGORIES:
        return "soft_rectangle", "known_soft_object_category_profile"
    if normalized_category(category) in MULTI_DEPTH_STRUCTURE_CATEGORIES:
        return "preserve", "known_multi_depth_structure_requires_depth_aware_amodal"
    return "preserve", "unknown_category_fail_closed"


def mask_bbox(mask: np.ndarray) -> list[int] | None:
    rows, columns = np.nonzero(mask)
    if not len(columns):
        return None
    return [
        int(columns.min()),
        int(rows.min()),
        int(columns.max()),
        int(rows.max()),
    ]


def mask_component_count(mask: np.ndarray) -> int:
    _, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    return int(count)


def mask_perimeter(mask: np.ndarray) -> int:
    if not np.any(mask):
        return 0
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=np.uint8))
    return int(np.count_nonzero(mask & ~eroded))


def describe_mask(mask: np.ndarray) -> dict[str, Any]:
    area = int(np.count_nonzero(mask))
    bbox = mask_bbox(mask)
    perimeter = mask_perimeter(mask)
    if bbox is None:
        return {
            "area_pixels": 0,
            "bbox_xyxy": None,
            "bbox_width": 0,
            "bbox_height": 0,
            "bbox_aspect_ratio": None,
            "bbox_fill_ratio": None,
            "connected_components": 0,
            "perimeter_pixels": 0,
            "compactness": None,
        }
    width = bbox[2] - bbox[0] + 1
    height = bbox[3] - bbox[1] + 1
    compactness = 4.0 * math.pi * area / (perimeter * perimeter) if perimeter else None
    return {
        "area_pixels": area,
        "bbox_xyxy": bbox,
        "bbox_width": width,
        "bbox_height": height,
        "bbox_aspect_ratio": float(width / height),
        "bbox_fill_ratio": float(area / (width * height)),
        "connected_components": mask_component_count(mask),
        "perimeter_pixels": perimeter,
        "compactness": float(compactness) if compactness is not None else None,
    }


def rounded_rectangle_candidate(
    mask: np.ndarray,
    *,
    corner_radius_ratio: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    bbox = mask_bbox(mask)
    if bbox is None:
        raise ValueError("source RGBA has no visible alpha pixels")
    x0, y0, x1, y1 = bbox
    width = x1 - x0 + 1
    height = y1 - y0 + 1
    radius = max(1, round(min(width, height) * corner_radius_ratio))
    canvas = Image.new("1", (mask.shape[1], mask.shape[0]), 0)
    ImageDraw.Draw(canvas).rounded_rectangle((x0, y0, x1, y1), radius=radius, fill=1)
    prior = np.asarray(canvas, dtype=bool)
    candidate = prior | mask
    return candidate, {
        "prior_bbox_xyxy": bbox,
        "corner_radius_px": radius,
        "corner_radius_ratio": corner_radius_ratio,
        "candidate_area_pixels": int(np.count_nonzero(candidate)),
    }


def enclosed_hole_candidate(
    mask: np.ndarray,
    *,
    min_component_area_ratio: float,
    cavity_bridge_radius: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return native holes plus cavities isolated by a topology-only bridge.

    Native enclosed holes are accepted directly. A morphological closing can also
    sever a narrow mouth between an occlusion cavity and the exterior background.
    The closing is used only to classify topology: pixels introduced by closing an
    exterior-connected component are never copied into the output alpha.
    """

    if cavity_bridge_radius < 0:
        raise ValueError("cavity_bridge_radius must be non-negative")
    original_area = int(np.count_nonzero(mask))
    minimum_component_area = max(
        16,
        round(original_area * min_component_area_ratio),
    )
    structure = np.ones((3, 3), dtype=np.uint8)
    original_labels, original_count = ndimage.label(
        ~mask,
        structure=structure,
    )
    height, width = mask.shape

    def component_record(
        labels: np.ndarray,
        component_id: int,
        *,
        stage: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        component = labels == component_id
        rows, columns = np.nonzero(component)
        area = len(rows)
        touches_crop_boundary = bool(
            np.any(rows == 0)
            or np.any(rows == height - 1)
            or np.any(columns == 0)
            or np.any(columns == width - 1)
        )
        return component, {
            "stage": stage,
            "component_id": component_id,
            "area_pixels": area,
            "area_to_original_ratio": float(area / original_area),
            "bbox_xyxy": mask_bbox(component),
            "touches_crop_boundary": touches_crop_boundary,
            "boundary_connectivity": (
                "connected_to_crop_boundary" if touches_crop_boundary else "enclosed"
            ),
            "passes_minimum_area": area >= minimum_component_area,
        }

    added_native = np.zeros_like(mask, dtype=bool)
    original_components: list[dict[str, Any]] = []
    original_component_boundary: dict[int, bool] = {}
    accepted_native_holes: list[dict[str, Any]] = []
    for component_id in range(1, original_count + 1):
        component, record = component_record(
            original_labels,
            component_id,
            stage="original_mask",
        )
        original_component_boundary[component_id] = record["touches_crop_boundary"]
        accepted = not record["touches_crop_boundary"] and record["passes_minimum_area"]
        record["accepted"] = accepted
        if record["touches_crop_boundary"]:
            record["disposition"] = "rejected_connected_to_crop_boundary"
        elif not record["passes_minimum_area"]:
            record["disposition"] = "rejected_below_minimum_area"
        else:
            record["disposition"] = "accepted_native_enclosed_hole"
            added_native |= component
            accepted_native_holes.append(record.copy())
        original_components.append(record)

    kernel_size = cavity_bridge_radius * 2 + 1
    if cavity_bridge_radius:
        closed_for_topology = ndimage.binary_closing(
            mask,
            structure=np.ones((kernel_size, kernel_size), dtype=np.uint8),
        )
    else:
        closed_for_topology = mask.copy()
    bridge_pixels = closed_for_topology & ~mask
    closed_labels, closed_count = ndimage.label(
        ~closed_for_topology,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    added_bridged = np.zeros_like(mask, dtype=bool)
    closed_components: list[dict[str, Any]] = []
    accepted_bridged_cavities: list[dict[str, Any]] = []
    for component_id in range(1, closed_count + 1):
        component, record = component_record(
            closed_labels,
            component_id,
            stage="bridge_closed_topology",
        )
        source_component_ids = sorted(
            int(item) for item in np.unique(original_labels[component]) if item != 0
        )
        was_connected_to_crop_boundary = any(
            original_component_boundary[item] for item in source_component_ids
        )
        accepted = bool(
            not record["touches_crop_boundary"]
            and record["passes_minimum_area"]
            and was_connected_to_crop_boundary
        )
        record.update(
            {
                "source_original_component_ids": source_component_ids,
                "source_was_connected_to_crop_boundary": was_connected_to_crop_boundary,
                "accepted": accepted,
            }
        )
        if record["touches_crop_boundary"]:
            record["disposition"] = "rejected_connected_to_crop_boundary_after_bridge"
        elif not record["passes_minimum_area"]:
            record["disposition"] = "rejected_below_minimum_area"
        elif not was_connected_to_crop_boundary:
            record["disposition"] = "already_handled_as_native_enclosed_hole"
        else:
            record["disposition"] = "accepted_narrow_mouth_cavity"
            # The component is background even after closing, so the synthetic
            # bridge itself is deliberately absent from this output mask.
            cavity_pixels = component & ~mask
            added_bridged |= cavity_pixels
            accepted_record = record.copy()
            accepted_record["added_pixels"] = int(np.count_nonzero(cavity_pixels))
            accepted_bridged_cavities.append(accepted_record)
        closed_components.append(record)

    original_exterior_ids = [
        component_id
        for component_id, touches_boundary in original_component_boundary.items()
        if touches_boundary
    ]
    original_exterior = np.isin(original_labels, original_exterior_ids)
    exterior_bridge_pixels = bridge_pixels & original_exterior
    added = added_native | added_bridged
    exterior_bridge_pixels_added = int(np.count_nonzero(added & exterior_bridge_pixels))
    if exterior_bridge_pixels_added:
        raise RuntimeError("topology-only exterior bridge pixels entered the output alpha")

    completion_applied = bool(np.any(added))
    if np.any(added_bridged) and np.any(added_native):
        completion_reason = "native_holes_and_narrow_mouth_cavities_completed"
    elif np.any(added_bridged):
        completion_reason = "narrow_mouth_occlusion_cavities_completed"
    elif np.any(added_native):
        completion_reason = "enclosed_alpha_holes_completed"
    elif any(not item["touches_crop_boundary"] for item in original_components):
        completion_reason = "enclosed_alpha_holes_below_minimum_area"
    else:
        completion_reason = "no_enclosed_alpha_holes"
    return added, {
        "completion_applied": completion_applied,
        "completion_reason": completion_reason,
        "minimum_component_area_pixels": minimum_component_area,
        "minimum_component_area_ratio": min_component_area_ratio,
        "cavity_bridge_radius_pixels": cavity_bridge_radius,
        "cavity_bridge_kernel_shape": [kernel_size, kernel_size],
        "bridge_is_topology_only": True,
        "bridge_candidate_pixels": int(np.count_nonzero(bridge_pixels)),
        "bridge_candidate_bbox_xyxy": mask_bbox(bridge_pixels),
        "exterior_bridge_candidate_pixels": int(np.count_nonzero(exterior_bridge_pixels)),
        "exterior_bridge_pixels_added_to_output": exterior_bridge_pixels_added,
        "background_component_count": original_count,
        "background_components": original_components,
        "post_bridge_background_component_count": closed_count,
        "post_bridge_background_components": closed_components,
        "accepted_enclosed_holes": accepted_native_holes,
        "accepted_bridged_cavities": accepted_bridged_cavities,
        "native_hole_pixels": int(np.count_nonzero(added_native)),
        "bridged_cavity_pixels": int(np.count_nonzero(added_bridged)),
        "accepted_hole_pixels": int(np.count_nonzero(added)),
    }


def infer_added_region(
    original_mask: np.ndarray,
    *,
    category: str,
    requested_policy: str,
    corner_radius_ratio: float,
    min_candidate_add_ratio: float,
    min_occlusion_depth_ratio: float,
    min_occlusion_depth_px: int,
    min_component_area_ratio: float,
    cavity_bridge_radius: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    resolved_policy, policy_reason = resolve_shape_policy(category, requested_policy)
    empty = np.zeros_like(original_mask, dtype=bool)
    if resolved_policy == "preserve":
        return empty, {
            "requested_shape_policy": requested_policy,
            "resolved_shape_policy": resolved_policy,
            "policy_reason": policy_reason,
            "completion_applied": False,
            "completion_reason": "category_has_no_safe_amodal_shape_prior",
            "candidate_add_pixels": 0,
            "candidate_add_ratio": 0.0,
            "accepted_deep_components": [],
        }

    if resolved_policy == "fill_enclosed_holes":
        added, hole_trace = enclosed_hole_candidate(
            original_mask,
            min_component_area_ratio=min_component_area_ratio,
            cavity_bridge_radius=cavity_bridge_radius,
        )
        return added, {
            "requested_shape_policy": requested_policy,
            "resolved_shape_policy": resolved_policy,
            "policy_reason": policy_reason,
            **hole_trace,
        }

    candidate, candidate_trace = rounded_rectangle_candidate(
        original_mask,
        corner_radius_ratio=corner_radius_ratio,
    )
    candidate_add = candidate & ~original_mask
    original_area = int(np.count_nonzero(original_mask))
    candidate_add_pixels = int(np.count_nonzero(candidate_add))
    candidate_add_ratio = candidate_add_pixels / original_area
    bbox = mask_bbox(original_mask)
    assert bbox is not None
    min_bbox_dimension = min(bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1)
    minimum_depth = max(
        min_occlusion_depth_px,
        round(min_bbox_dimension * min_occlusion_depth_ratio),
    )
    minimum_component_area = max(
        16,
        round(original_area * min_component_area_ratio),
    )
    interior_distance = ndimage.distance_transform_edt(candidate)
    deep_pixels = candidate_add & (interior_distance >= minimum_depth)
    labels, count = ndimage.label(
        deep_pixels,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    added = np.zeros_like(original_mask, dtype=bool)
    components: list[dict[str, Any]] = []
    if candidate_add_ratio >= min_candidate_add_ratio:
        for component_id in range(1, count + 1):
            component = labels == component_id
            component_area = int(np.count_nonzero(component))
            if component_area < minimum_component_area:
                continue
            distance_to_seed = ndimage.distance_transform_edt(~component)
            local_added = candidate_add & (distance_to_seed <= minimum_depth + 2)
            added |= local_added
            components.append(
                {
                    "component_id": component_id,
                    "deep_area_pixels": component_area,
                    "deep_bbox_xyxy": mask_bbox(component),
                    "max_interior_depth_px": float(interior_distance[component].max()),
                    "grown_added_pixels": int(np.count_nonzero(local_added)),
                }
            )

    completion_applied = bool(np.any(added))
    if candidate_add_ratio < min_candidate_add_ratio:
        completion_reason = "candidate_difference_below_occlusion_threshold"
    elif not components:
        completion_reason = "no_deep_occlusion_component_passed_gates"
    else:
        completion_reason = "deep_occlusion_notches_completed_with_soft_rectangle_prior"
    return added, {
        "requested_shape_policy": requested_policy,
        "resolved_shape_policy": resolved_policy,
        "policy_reason": policy_reason,
        "completion_applied": completion_applied,
        "completion_reason": completion_reason,
        "candidate_add_pixels": candidate_add_pixels,
        "candidate_add_ratio": float(candidate_add_ratio),
        "minimum_candidate_add_ratio": min_candidate_add_ratio,
        "minimum_occlusion_depth_px": minimum_depth,
        "minimum_component_area_pixels": minimum_component_area,
        "deep_candidate_pixels": int(np.count_nonzero(deep_pixels)),
        "accepted_deep_components": components,
        **candidate_trace,
    }


def channel_statistics(pixels: np.ndarray) -> dict[str, Any] | None:
    if not len(pixels):
        return None
    values = pixels.astype(np.float64)
    luminance = (values * np.asarray([0.2126, 0.7152, 0.0722])).sum(axis=1)
    return {
        "count": len(values),
        "mean_rgb": [float(item) for item in values.mean(axis=0)],
        "median_rgb": [float(item) for item in np.median(values, axis=0)],
        "std_rgb": [float(item) for item in values.std(axis=0)],
        "min_rgb": [int(item) for item in values.min(axis=0)],
        "max_rgb": [int(item) for item in values.max(axis=0)],
        "mean_luminance": float(luminance.mean()),
        "p05_luminance": float(np.percentile(luminance, 5)),
        "p95_luminance": float(np.percentile(luminance, 95)),
    }


def reference_path(value: str, manifest_path: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def crop_reference_mask(
    mask_path: Path,
    *,
    image_size: tuple[int, int],
    bbox: list[int],
) -> np.ndarray:
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) >= 127
    x0, y0, x1, y1 = bbox
    crop_shape = (y1 - y0, x1 - x0)
    if mask.shape == (image_size[1], image_size[0]):
        return mask[y0:y1, x0:x1]
    if mask.shape == crop_shape:
        return mask
    raise ValueError(
        f"reference mask {mask_path} must match the full frame or bbox crop; "
        f"got {mask.shape}, expected {(image_size[1], image_size[0])} or {crop_shape}"
    )


def white_appearance_metrics(rgb: np.ndarray) -> dict[str, Any]:
    values = rgb.astype(np.float64)
    luminance = (values * np.asarray([0.2126, 0.7152, 0.0722])).sum(axis=1)
    chroma = values.max(axis=1) - values.min(axis=1)
    white_like = (luminance >= 115) & (chroma <= 28)
    median_rgb = np.median(values, axis=0)
    chromaticity = median_rgb / max(float(median_rgb.sum()), 1.0)
    return {
        "median_rgb": [float(item) for item in median_rgb],
        "median_luminance": float(np.median(luminance)),
        "median_chroma": float(np.median(chroma)),
        "white_like_fraction": float(np.mean(white_like)),
        "median_rgb_chromaticity": [float(item) for item in chromaticity],
    }


def audit_scene_references(
    *,
    manifest_path: Path,
    source_rgba: np.ndarray,
    original_mask: np.ndarray,
    object_id: str,
    category: str,
) -> dict[str, Any]:
    value = read_json(manifest_path)
    if not isinstance(value, dict):
        raise ValueError("scene reference manifest must be a JSON object")
    frames = value.get("frames")
    if not isinstance(frames, list) or len(frames) < 2:
        raise ValueError("scene reference manifest must contain at least two frames")
    if value.get("object_id") != object_id:
        raise ValueError("scene reference manifest object_id does not match --object-id")
    if normalized_category(str(value.get("category", ""))) != normalized_category(category):
        raise ValueError("scene reference manifest category does not match --category")

    frame_reports: list[dict[str, Any]] = []
    source_reports: list[dict[str, Any]] = []
    frame_ids: set[str] = set()
    for entry in frames:
        if not isinstance(entry, dict):
            raise ValueError("each scene reference frame must be a JSON object")
        frame_id = str(entry.get("frame_id", "")).strip()
        if not frame_id or frame_id in frame_ids:
            raise ValueError("scene reference frame_id values must be non-empty and unique")
        frame_ids.add(frame_id)
        image_path = reference_path(str(entry.get("image_path", "")), manifest_path)
        mask_path = reference_path(str(entry.get("mask_path", "")), manifest_path)
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"missing scene reference image or mask for frame {frame_id}")
        image = Image.open(image_path).convert("RGB")
        bbox_value = entry.get("bbox_xyxy")
        if (
            not isinstance(bbox_value, list)
            or len(bbox_value) != 4
            or any(not isinstance(item, int) for item in bbox_value)
        ):
            raise ValueError(f"frame {frame_id} bbox_xyxy must contain four integers")
        x0, y0, x1, y1 = bbox_value
        if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
            raise ValueError(f"frame {frame_id} bbox_xyxy is outside the scene image")
        reference_mask = crop_reference_mask(
            mask_path,
            image_size=image.size,
            bbox=bbox_value,
        )
        if not np.any(reference_mask):
            raise ValueError(f"frame {frame_id} reference mask is empty")
        crop = np.asarray(image.crop((x0, y0, x1, y1)), dtype=np.uint8)
        appearance = white_appearance_metrics(crop[reference_mask])
        report = {
            "frame_id": frame_id,
            "image_path": str(image_path),
            "image_sha256": sha256_file(image_path),
            "mask_path": str(mask_path),
            "mask_sha256": sha256_file(mask_path),
            "bbox_xyxy": bbox_value,
            "foreground_pixels": int(np.count_nonzero(reference_mask)),
            "appearance": appearance,
            "is_source_frame": bool(entry.get("is_source_frame", False)),
        }
        if report["is_source_frame"]:
            shapes_match = (
                crop.shape[:2] == source_rgba.shape[:2]
                and reference_mask.shape == original_mask.shape
            )
            if not shapes_match:
                report["source_alignment"] = {
                    "shape_matches": False,
                    "mask_iou": 0.0,
                    "mean_absolute_rgb_delta": None,
                }
            else:
                union = np.count_nonzero(reference_mask | original_mask)
                intersection = np.count_nonzero(reference_mask & original_mask)
                mask_iou = float(intersection / union) if union else 0.0
                mean_delta = float(
                    np.abs(
                        crop[original_mask].astype(np.int16)
                        - source_rgba[..., :3][original_mask].astype(np.int16)
                    ).mean()
                )
                report["source_alignment"] = {
                    "shape_matches": True,
                    "mask_iou": mask_iou,
                    "mean_absolute_rgb_delta": mean_delta,
                }
            source_reports.append(report)
        frame_reports.append(report)

    chromaticities = np.asarray(
        [report["appearance"]["median_rgb_chromaticity"] for report in frame_reports]
    )
    maximum_chromaticity_distance = 0.0
    for first in range(len(chromaticities)):
        for second in range(first + 1, len(chromaticities)):
            maximum_chromaticity_distance = max(
                maximum_chromaticity_distance,
                float(np.linalg.norm(chromaticities[first] - chromaticities[second])),
            )

    expected = value.get("expected_appearance")
    if not isinstance(expected, dict):
        raise ValueError("scene reference manifest requires expected_appearance")
    color_family = str(expected.get("color_family", "")).strip()
    if color_family not in {"any", "light_white_offwhite"}:
        raise ValueError("expected_appearance.color_family must be any or light_white_offwhite")
    if color_family == "light_white_offwhite":
        passing_frames = sum(
            report["appearance"]["white_like_fraction"] >= 0.35
            and report["appearance"]["median_chroma"] <= 45
            for report in frame_reports
        )
        expected_appearance_met = passing_frames >= math.ceil(len(frame_reports) / 2)
    else:
        passing_frames = len(frame_reports)
        expected_appearance_met = True

    source_alignment = (
        source_reports[0].get("source_alignment", {}) if len(source_reports) == 1 else {}
    )
    gates = {
        "at_least_two_scene_reference_frames": len(frame_reports) >= 2,
        "exactly_one_source_frame_reference": len(source_reports) == 1,
        "source_frame_shape_matches_rgba": source_alignment.get("shape_matches") is True,
        "source_frame_mask_matches_rgba": source_alignment.get("mask_iou", 0.0) >= 0.98,
        "source_frame_rgb_matches_rgba": (
            source_alignment.get("mean_absolute_rgb_delta") is not None
            and source_alignment["mean_absolute_rgb_delta"] <= 2.0
        ),
        "cross_frame_foreground_chromaticity_consistent": maximum_chromaticity_distance <= 0.18,
        "expected_appearance_contract_met": expected_appearance_met,
        "same_instance_identity_revalidated": value.get("identity_revalidated") is True,
    }
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "frame_count": len(frame_reports),
        "frames": frame_reports,
        "expected_appearance": expected,
        "expected_appearance_passing_frames": passing_frames,
        "maximum_pairwise_median_chromaticity_distance": maximum_chromaticity_distance,
        "identity_revalidated": value.get("identity_revalidated") is True,
        "identity_revalidated_by": value.get("identity_revalidated_by"),
        "gates": gates,
        "passed": all(gates.values()),
        "scope_note": (
            "These gates verify multi-frame mask/color consistency and require an explicit "
            "same-instance decision; they do not infer hidden geometry truth."
        ),
    }


def complete_rgba(source_rgba: np.ndarray, config: CompletionConfig) -> CompletionResult:
    if source_rgba.ndim != 3 or source_rgba.shape[2] != 4:
        raise ValueError("source image must be RGBA")
    if source_rgba.dtype != np.uint8:
        raise ValueError("source image must use uint8 channels")
    original_alpha = source_rgba[..., 3].copy()
    original_mask = original_alpha >= config.alpha_threshold
    if not np.any(original_mask):
        raise ValueError("source RGBA has no alpha pixels at or above the threshold")
    added_mask, method_trace = infer_added_region(
        original_mask,
        category=config.category,
        requested_policy=config.shape_policy,
        corner_radius_ratio=config.corner_radius_ratio,
        min_candidate_add_ratio=config.min_candidate_add_ratio,
        min_occlusion_depth_ratio=config.min_occlusion_depth_ratio,
        min_occlusion_depth_px=config.min_occlusion_depth_px,
        min_component_area_ratio=config.min_component_area_ratio,
        cavity_bridge_radius=config.cavity_bridge_radius,
    )
    completed_mask = original_mask | added_mask
    completed_alpha = original_alpha.copy()
    completed_alpha[added_mask] = 255

    source_rgb = source_rgba[..., :3]
    completed_rgb = source_rgb.copy()
    nearest_source_is_visible = True
    if np.any(added_mask):
        _, nearest_indices = ndimage.distance_transform_edt(
            ~original_mask,
            return_indices=True,
        )
        nearest_rows = nearest_indices[0][added_mask]
        nearest_columns = nearest_indices[1][added_mask]
        nearest_source_is_visible = bool(np.all(original_mask[nearest_rows, nearest_columns]))
        completed_rgb[added_mask] = source_rgb[nearest_rows, nearest_columns]
    completed_rgb[~completed_mask] = 0
    completed_rgba = np.dstack([completed_rgb, completed_alpha]).astype(np.uint8)

    visible_unchanged = bool(
        np.array_equal(completed_rgba[original_mask, :3], source_rgb[original_mask])
        and np.array_equal(completed_alpha[original_mask], original_alpha[original_mask])
    )
    transparent_nonzero = int(np.count_nonzero(np.any(completed_rgb[~completed_mask] != 0, axis=1)))
    original_colors = channel_statistics(source_rgb[original_mask])
    added_colors = channel_statistics(completed_rgb[added_mask])
    color_delta = None
    if original_colors is not None and added_colors is not None:
        color_delta = float(
            np.linalg.norm(
                np.asarray(original_colors["mean_rgb"]) - np.asarray(added_colors["mean_rgb"])
            )
        )
    original_shape = describe_mask(original_mask)
    completed_shape = describe_mask(completed_mask)
    added_pixels = int(np.count_nonzero(added_mask))
    original_pixels = int(np.count_nonzero(original_mask))
    shape_metrics = {
        "original": original_shape,
        "completed": completed_shape,
        "added_pixels": added_pixels,
        "added_to_original_area_ratio": float(added_pixels / original_pixels),
        "alpha_iou_original_to_completed": float(
            original_pixels / (original_pixels + added_pixels)
        ),
        "original_pixels_removed": int(np.count_nonzero(original_mask & ~completed_mask)),
    }
    color_metrics = {
        "texture_provider": "nearest_visible_foreground",
        "original_visible": original_colors,
        "added_region": added_colors,
        "mean_rgb_delta_l2": color_delta,
        "original_visible_pixels_byte_identical": visible_unchanged,
        "added_pixels_source_only_original_foreground": nearest_source_is_visible,
        "transparent_rgb_nonzero_pixels": transparent_nonzero,
    }
    gates = {
        "original_visible_pixels_preserved": visible_unchanged,
        "no_original_alpha_removed": not bool(np.any(original_mask & ~completed_mask)),
        "added_texture_comes_from_visible_foreground": nearest_source_is_visible,
        "transparent_background_rgb_is_zero": transparent_nonzero == 0,
        "completed_alpha_is_nonempty": bool(np.any(completed_mask)),
        "completed_alpha_single_component": mask_component_count(completed_mask) == 1,
    }
    return CompletionResult(
        source_rgba=source_rgba.copy(),
        original_alpha=original_alpha,
        completed_alpha=completed_alpha,
        original_mask=original_mask,
        completed_mask=completed_mask,
        added_mask=added_mask,
        completed_rgba=completed_rgba,
        shape_metrics=shape_metrics,
        color_metrics=color_metrics,
        gates=gates,
        method_trace=method_trace,
    )


def checkerboard(size: tuple[int, int], tile: int = 12) -> Image.Image:
    width, height = size
    rows, columns = np.indices((height, width))
    values = np.where(((rows // tile) + (columns // tile)) % 2, 196, 232).astype(np.uint8)
    rgb = np.repeat(values[..., None], 3, axis=2)
    return Image.fromarray(rgb).convert("RGBA")


def composite_rgba(value: np.ndarray) -> Image.Image:
    foreground = Image.fromarray(value)
    return Image.alpha_composite(checkerboard(foreground.size), foreground).convert("RGB")


def contact_sheet_panel(image: Image.Image, label: str, size: tuple[int, int]) -> Image.Image:
    width, height = size
    panel = Image.new("RGB", size, (20, 22, 26))
    contained = ImageOps.contain(image.convert("RGB"), (width - 20, height - 44))
    x = (width - contained.width) // 2
    y = 32 + (height - 36 - contained.height) // 2
    panel.paste(contained, (x, y))
    ImageDraw.Draw(panel).text((10, 9), label, fill=(238, 240, 244))
    return panel


def build_contact_sheet(result: CompletionResult) -> Image.Image:
    alpha_original = Image.fromarray(result.original_alpha).convert("RGB")
    alpha_completed = Image.fromarray(result.completed_alpha).convert("RGB")
    overlay = result.source_rgba.copy()
    overlay[result.added_mask, :3] = np.asarray([255, 54, 72], dtype=np.uint8)
    overlay[result.added_mask, 3] = 255
    panels = [
        contact_sheet_panel(composite_rgba(result.source_rgba), "source RGBA", (230, 250)),
        contact_sheet_panel(alpha_original, "original alpha", (230, 250)),
        contact_sheet_panel(alpha_completed, "completed alpha", (230, 250)),
        contact_sheet_panel(composite_rgba(overlay), "added region (red)", (230, 250)),
        contact_sheet_panel(composite_rgba(result.completed_rgba), "completed RGBA", (230, 250)),
    ]
    sheet = Image.new("RGB", (sum(panel.width for panel in panels), 250), (20, 22, 26))
    x = 0
    for panel in panels:
        sheet.paste(panel, (x, 0))
        x += panel.width
    return sheet


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "source_rgba": output_dir / "source_rgba.png",
        "original_alpha": output_dir / "original_alpha.png",
        "completed_alpha": output_dir / "completed_alpha.png",
        "added_region": output_dir / "added_region.png",
        "completed_rgba": output_dir / "completed_rgba.png",
        "contact_sheet": output_dir / "before_after_contact_sheet.png",
        "audit": output_dir / "amodal_rgba_audit.json",
    }


def build_retry_prompt(object_id: str, category: str, preserve_description: str | None) -> str:
    details = preserve_description or (
        "Preserve the observed base color, local texture or pattern, material, "
        "silhouette proportions, and orientation exactly."
    )
    return (
        f"Retry image-to-3D reconstruction for exactly one {category} ({object_id}). "
        "The supplied RGBA is an auditable amodal front-view condition: its original visible "
        "pixels are unchanged and only category-safe occlusion regions were filled from nearby "
        "visible foreground texture. Do not reinterpret the category or merge adjacent objects. "
        f"{details} Generate a closed, connected, volumetric object with plausible front, back, "
        "left, right, top, and bottom surfaces. The rear must not be an open hole or a flat copy "
        "of the front. Keep thickness plausible for the category; avoid detached pieces, "
        "unexpected openings, severe color shifts, shape changes, and texture from the transparent "
        "background. Review all six orthographic views before accepting the result."
    )


def write_result(
    *,
    source_path: Path,
    output_dir: Path,
    result: CompletionResult,
    config: CompletionConfig,
    scene_reference_audit: dict[str, Any],
    mirror_dir: Path | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(output_dir)
    shutil.copyfile(source_path, paths["source_rgba"])
    Image.fromarray(result.original_alpha).save(paths["original_alpha"])
    Image.fromarray(result.completed_alpha).save(paths["completed_alpha"])
    Image.fromarray(result.added_mask.astype(np.uint8) * 255).save(paths["added_region"])
    Image.fromarray(result.completed_rgba).save(paths["completed_rgba"])
    build_contact_sheet(result).save(paths["contact_sheet"])

    config_value = asdict(config)
    artifacts = {
        key: {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for key, path in paths.items()
        if key != "audit"
    }
    audit = {
        "schema_version": SCHEMA_VERSION,
        "kind": "video2world.amodal_rgba_retry_input",
        "created_at": datetime.now(UTC).isoformat(),
        "object_id": config.object_id,
        "category": config.category,
        "attempt": config.attempt,
        "source": {
            "path": str(source_path.resolve()),
            "sha256": sha256_file(source_path),
            "size_bytes": source_path.stat().st_size,
            "width": int(result.source_rgba.shape[1]),
            "height": int(result.source_rgba.shape[0]),
        },
        "config": config_value,
        "config_sha256": sha256_json(config_value),
        "method_trace": result.method_trace,
        "shape_metrics": result.shape_metrics,
        "color_metrics": result.color_metrics,
        "scene_reference_audit": scene_reference_audit,
        "acceptance_gates": {
            **result.gates,
            **{
                f"scene_reference.{key}": gate
                for key, gate in scene_reference_audit["gates"].items()
            },
        },
        "all_acceptance_gates_passed": (
            all(result.gates.values()) and scene_reference_audit["passed"]
        ),
        "artifacts": artifacts,
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "scipy": scipy.__version__,
        },
    }
    write_json(paths["audit"], audit)
    if mirror_dir is not None:
        mirror_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            paths["completed_rgba"],
            mirror_dir / f"{config.object_id}_attempt{config.attempt}_amodal_rgba.png",
        )
        shutil.copyfile(
            paths["audit"],
            mirror_dir / f"{config.object_id}_attempt{config.attempt}_amodal_rgba_audit.json",
        )
    return audit


def process_one(
    *,
    source_path: Path,
    output_dir: Path,
    object_id: str,
    category: str,
    attempt: int,
    shape_policy: str,
    alpha_threshold: int,
    corner_radius_ratio: float,
    min_candidate_add_ratio: float,
    min_occlusion_depth_ratio: float,
    min_occlusion_depth_px: int,
    min_component_area_ratio: float,
    cavity_bridge_radius: int,
    seed: int,
    provider: str,
    retry_prompt: str,
    scene_reference_manifest: Path,
    mirror_dir: Path | None = None,
) -> dict[str, Any]:
    source_rgba = np.asarray(Image.open(source_path).convert("RGBA"), dtype=np.uint8)
    original_mask = source_rgba[..., 3] >= alpha_threshold
    scene_reference_audit = audit_scene_references(
        manifest_path=scene_reference_manifest,
        source_rgba=source_rgba,
        original_mask=original_mask,
        object_id=object_id,
        category=category,
    )
    if not scene_reference_audit["passed"]:
        failed = [name for name, passed in scene_reference_audit["gates"].items() if not passed]
        raise ValueError(
            "scene reference audit rejected input before amodal completion: " + ", ".join(failed)
        )
    config = CompletionConfig(
        object_id=object_id,
        category=category,
        attempt=attempt,
        shape_policy=shape_policy,
        alpha_threshold=alpha_threshold,
        corner_radius_ratio=corner_radius_ratio,
        min_candidate_add_ratio=min_candidate_add_ratio,
        min_occlusion_depth_ratio=min_occlusion_depth_ratio,
        min_occlusion_depth_px=min_occlusion_depth_px,
        min_component_area_ratio=min_component_area_ratio,
        cavity_bridge_radius=cavity_bridge_radius,
        seed=seed,
        provider=provider,
        retry_prompt=retry_prompt,
        scene_reference_manifest=str(scene_reference_manifest),
        scene_reference_manifest_sha256=sha256_file(scene_reference_manifest),
    )
    result = complete_rgba(source_rgba, config)
    return write_result(
        source_path=source_path,
        output_dir=output_dir,
        result=result,
        config=config,
        scene_reference_audit=scene_reference_audit,
        mirror_dir=mirror_dir,
    )


def synthetic_rgba(
    *,
    size: tuple[int, int],
    bbox: tuple[int, int, int, int],
    radius: int,
    cutout: tuple[int, int, int, int] | None,
) -> np.ndarray:
    width, height = size
    alpha_image = Image.new("L", size, 0)
    draw = ImageDraw.Draw(alpha_image)
    draw.rounded_rectangle(bbox, radius=radius, fill=255)
    if cutout is not None:
        draw.rounded_rectangle(cutout, radius=max(2, radius // 2), fill=0)
    alpha = np.asarray(alpha_image, dtype=np.uint8)
    rows, columns = np.indices((height, width))
    rgb = np.stack(
        [
            205 + (columns % 23),
            210 + (rows % 19),
            215 + ((columns + rows) % 17),
        ],
        axis=2,
    ).astype(np.uint8)
    rgb[alpha == 0] = np.asarray([255, 0, 255], dtype=np.uint8)
    return np.dstack([rgb, alpha]).astype(np.uint8)


def run_self_test(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pillow_source = output_dir / "synthetic_pillow_source.png"
    complete_source = output_dir / "synthetic_complete_pillow_source.png"
    unknown_source = output_dir / "synthetic_unknown_source.png"
    Image.fromarray(
        synthetic_rgba(
            size=(180, 140),
            bbox=(20, 20, 159, 119),
            radius=16,
            cutout=(91, 55, 170, 114),
        ),
    ).save(pillow_source)
    Image.fromarray(
        synthetic_rgba(
            size=(180, 140),
            bbox=(20, 20, 159, 119),
            radius=16,
            cutout=None,
        ),
    ).save(complete_source)
    unknown = synthetic_rgba(
        size=(180, 140),
        bbox=(30, 20, 149, 119),
        radius=12,
        cutout=(75, 45, 115, 105),
    )
    Image.fromarray(unknown).save(unknown_source)

    def make_reference_manifest(
        source: Path,
        *,
        object_id: str,
        category: str,
    ) -> Path:
        reference_dir = output_dir / f"{object_id}_references"
        reference_dir.mkdir(parents=True, exist_ok=True)
        image = Image.open(source).convert("RGBA")
        alpha = np.asarray(image, dtype=np.uint8)[..., 3]
        mask_first = reference_dir / "mask_first.png"
        mask_second = reference_dir / "mask_second.png"
        image_second = reference_dir / "frame_second.png"
        Image.fromarray(alpha).save(mask_first)
        Image.fromarray(alpha).save(mask_second)
        image.save(image_second)
        manifest = reference_dir / "scene_references.json"
        bbox = [0, 0, image.width, image.height]
        write_json(
            manifest,
            {
                "schema_version": 1,
                "object_id": object_id,
                "category": category,
                "identity_revalidated": True,
                "identity_revalidated_by": "synthetic_self_test",
                "expected_appearance": {"color_family": "any"},
                "frames": [
                    {
                        "frame_id": "synthetic_first",
                        "image_path": str(source.resolve()),
                        "mask_path": str(mask_first.resolve()),
                        "bbox_xyxy": bbox,
                        "is_source_frame": True,
                    },
                    {
                        "frame_id": "synthetic_second",
                        "image_path": str(image_second.resolve()),
                        "mask_path": str(mask_second.resolve()),
                        "bbox_xyxy": bbox,
                        "is_source_frame": False,
                    },
                ],
            },
        )
        return manifest

    common = {
        "attempt": 2,
        "shape_policy": "auto",
        "alpha_threshold": 127,
        "corner_radius_ratio": 0.10,
        "min_candidate_add_ratio": 0.24,
        "min_occlusion_depth_ratio": 0.16,
        "min_occlusion_depth_px": 8,
        "min_component_area_ratio": 0.002,
        "cavity_bridge_radius": 5,
        "seed": 20260717,
        "provider": PROVIDER_NAME,
        "retry_prompt": "synthetic self-test only",
    }
    pillow = process_one(
        source_path=pillow_source,
        output_dir=output_dir / "pillow",
        object_id="synthetic_pillow",
        category="pillow",
        scene_reference_manifest=make_reference_manifest(
            pillow_source,
            object_id="synthetic_pillow",
            category="pillow",
        ),
        **common,
    )
    complete = process_one(
        source_path=complete_source,
        output_dir=output_dir / "complete_pillow",
        object_id="synthetic_complete_pillow",
        category="pillow",
        scene_reference_manifest=make_reference_manifest(
            complete_source,
            object_id="synthetic_complete_pillow",
            category="pillow",
        ),
        **common,
    )
    unknown_result = process_one(
        source_path=unknown_source,
        output_dir=output_dir / "unknown_category",
        object_id="synthetic_unknown",
        category="plant",
        scene_reference_manifest=make_reference_manifest(
            unknown_source,
            object_id="synthetic_unknown",
            category="plant",
        ),
        **common,
    )
    one_frame_manifest = output_dir / "one_frame_scene_reference.json"
    write_json(
        one_frame_manifest,
        {
            "schema_version": 1,
            "object_id": "synthetic_pillow",
            "category": "pillow",
            "identity_revalidated": True,
            "expected_appearance": {"color_family": "any"},
            "frames": [
                {
                    "frame_id": "only_frame",
                    "image_path": str(pillow_source.resolve()),
                    "mask_path": str(
                        (output_dir / "synthetic_pillow_references" / "mask_first.png").resolve()
                    ),
                    "bbox_xyxy": [0, 0, 180, 140],
                    "is_source_frame": True,
                }
            ],
        },
    )
    single_frame_rejected = False
    try:
        audit_scene_references(
            manifest_path=one_frame_manifest,
            source_rgba=np.asarray(Image.open(pillow_source).convert("RGBA"), dtype=np.uint8),
            original_mask=(
                np.asarray(Image.open(pillow_source).convert("RGBA"), dtype=np.uint8)[..., 3] >= 127
            ),
            object_id="synthetic_pillow",
            category="pillow",
        )
    except ValueError:
        single_frame_rejected = True
    checks = {
        "occluded_pillow_adds_pixels": pillow["shape_metrics"]["added_pixels"] > 0,
        "occluded_pillow_preserves_visible_rgb": pillow["acceptance_gates"][
            "original_visible_pixels_preserved"
        ],
        "occluded_pillow_has_no_transparent_rgb_leak": pillow["acceptance_gates"][
            "transparent_background_rgb_is_zero"
        ],
        "complete_pillow_is_minimal_change": complete["shape_metrics"]["added_pixels"] == 0,
        "unknown_category_is_not_rectangularized": unknown_result["shape_metrics"]["added_pixels"]
        == 0,
        "unknown_category_failed_closed": unknown_result["method_trace"]["resolved_shape_policy"]
        == "preserve",
        "single_frame_scene_reference_is_rejected": single_frame_rejected,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "video2world.amodal_rgba_synthetic_self_test",
        "checks": checks,
        "passed": all(checks.values()),
    }
    write_json(output_dir / "self_test_report.json", report)
    if not report["passed"]:
        raise RuntimeError(f"synthetic self-test failed: {checks}")
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--source-rgba", type=Path)
    value.add_argument("--output-dir", type=Path)
    value.add_argument("--object-id")
    value.add_argument("--category")
    value.add_argument("--attempt", type=int, default=2)
    value.add_argument("--shape-policy", choices=sorted(SHAPE_POLICIES), default="auto")
    value.add_argument("--alpha-threshold", type=int, default=127)
    value.add_argument("--corner-radius-ratio", type=float, default=0.10)
    value.add_argument("--min-candidate-add-ratio", type=float, default=0.24)
    value.add_argument("--min-occlusion-depth-ratio", type=float, default=0.16)
    value.add_argument("--min-occlusion-depth-px", type=int, default=8)
    value.add_argument("--min-component-area-ratio", type=float, default=0.002)
    value.add_argument(
        "--cavity-bridge-radius",
        type=int,
        default=5,
        help=(
            "Radius in pixels for the topology-only closing used to isolate narrow-mouth "
            "cavities; closing pixels are never copied into the output alpha"
        ),
    )
    value.add_argument("--seed", type=int, default=20260717)
    value.add_argument("--provider", default=PROVIDER_NAME)
    value.add_argument("--retry-prompt")
    value.add_argument("--preserve-description")
    value.add_argument(
        "--scene-reference-manifest",
        type=Path,
        help=(
            "JSON with at least two original video frames, per-frame masks/bboxes, an expected "
            "appearance contract, and an explicit same-instance revalidation decision"
        ),
    )
    value.add_argument("--mirror-dir", type=Path)
    value.add_argument("--self-test", type=Path, metavar="OUTPUT_DIR")
    return value


def validate_args(args: argparse.Namespace) -> None:
    if args.self_test is not None:
        return
    missing = [
        name
        for name in (
            "source_rgba",
            "output_dir",
            "object_id",
            "category",
            "scene_reference_manifest",
        )
        if getattr(args, name) in {None, ""}
    ]
    if missing:
        raise ValueError(f"missing required arguments: {', '.join(missing)}")
    if not args.source_rgba.is_file():
        raise FileNotFoundError(args.source_rgba)
    if not args.scene_reference_manifest.is_file():
        raise FileNotFoundError(args.scene_reference_manifest)
    if not 1 <= args.alpha_threshold <= 255:
        raise ValueError("--alpha-threshold must be between 1 and 255")
    for name in (
        "corner_radius_ratio",
        "min_candidate_add_ratio",
        "min_occlusion_depth_ratio",
        "min_component_area_ratio",
    ):
        if not 0 <= getattr(args, name) <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1")
    if args.min_occlusion_depth_px < 1:
        raise ValueError("--min-occlusion-depth-px must be positive")
    if args.cavity_bridge_radius < 0:
        raise ValueError("--cavity-bridge-radius must be non-negative")
    if args.attempt < 1:
        raise ValueError("--attempt must be positive")


def main() -> int:
    args = parser().parse_args()
    try:
        validate_args(args)
        if args.self_test is not None:
            report = run_self_test(args.self_test.resolve())
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0
        prompt = args.retry_prompt or build_retry_prompt(
            args.object_id,
            args.category,
            args.preserve_description,
        )
        audit = process_one(
            source_path=args.source_rgba.resolve(),
            output_dir=args.output_dir.resolve(),
            object_id=args.object_id,
            category=args.category,
            attempt=args.attempt,
            shape_policy=args.shape_policy,
            alpha_threshold=args.alpha_threshold,
            corner_radius_ratio=args.corner_radius_ratio,
            min_candidate_add_ratio=args.min_candidate_add_ratio,
            min_occlusion_depth_ratio=args.min_occlusion_depth_ratio,
            min_occlusion_depth_px=args.min_occlusion_depth_px,
            min_component_area_ratio=args.min_component_area_ratio,
            cavity_bridge_radius=args.cavity_bridge_radius,
            seed=args.seed,
            provider=args.provider,
            retry_prompt=prompt,
            scene_reference_manifest=args.scene_reference_manifest.resolve(),
            mirror_dir=args.mirror_dir.resolve() if args.mirror_dir else None,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
