#!/usr/bin/env python3
"""Associate frame-local SAM3 masks with physical 3D object anchors.

Identity comes only from an explicit ``object_id=PLY`` anchor. Per-frame SAM3
detections are associated by calibrated projection evidence; mask filenames and
per-frame ordinal suffixes never participate in identity assignment.

The projection convention matches the accepted Video2Mesh Bedroom4 verifier:
``camera_xyz = world_xyz @ R.T + t`` for a ``world_to_camera`` extrinsic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import PIL
from PIL import Image
from plyfile import PlyData

OBJECT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
WEIGHTS = {
    "mask_hit_ratio": 0.45,
    "dilated_support_ratio": 0.30,
    "bbox_iou": 0.20,
    "sam3_score": 0.05,
}


@dataclass(frozen=True)
class Anchor:
    object_id: str
    declared_path: str
    path: Path
    sha256: str
    size: int
    points: np.ndarray
    centroid: np.ndarray
    rgb: dict[str, Any] | None


@dataclass(frozen=True)
class Detection:
    detection_id: str
    item_index: int
    frame_id: str
    image: str
    label: str | None
    score: float
    declared_path: str
    path: Path
    sha256: str
    size: int
    mask: np.ndarray
    area_pixels: int
    actual_bbox_xyxy: list[int]
    declared_bbox_xyxy: list[float]


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
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_path_remaps(values: list[str]) -> list[tuple[Path, Path]]:
    remaps: list[tuple[Path, Path]] = []
    for value in values:
        source_value, separator, target_value = value.partition("=")
        if not separator or not source_value or not target_value:
            raise ValueError(f"path remap must be ABSOLUTE_SOURCE=LOCAL_TARGET: {value!r}")
        source = Path(source_value).expanduser()
        target = Path(target_value).expanduser()
        if not source.is_absolute() or not target.is_absolute():
            raise ValueError(f"path remap endpoints must be absolute: {value!r}")
        remaps.append((source, target))
    return remaps


def resolve_input_path(
    value: str,
    *,
    relative_to: Path,
    path_remaps: list[tuple[Path, Path]],
) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        for source, target in path_remaps:
            try:
                relative = path.relative_to(source)
            except ValueError:
                continue
            return (target / relative).resolve()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def rgb_stats(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values)
    if array.size == 0:
        return {
            "pixel_count": 0,
            "mean_rgb": None,
            "median_rgb": None,
            "std_rgb": None,
            "p10_rgb": None,
            "p90_rgb": None,
        }
    array = array.reshape(-1, 3).astype(np.float64, copy=False)

    def vector(value: np.ndarray) -> list[float]:
        return [float(item) for item in value]

    return {
        "pixel_count": len(array),
        "mean_rgb": vector(array.mean(axis=0)),
        "median_rgb": vector(np.median(array, axis=0)),
        "std_rgb": vector(array.std(axis=0)),
        "p10_rgb": vector(np.percentile(array, 10, axis=0)),
        "p90_rgb": vector(np.percentile(array, 90, axis=0)),
    }


def parse_anchor_values(values: list[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    seen: set[str] = set()
    for value in values:
        object_id, separator, path = value.partition("=")
        if not separator or not OBJECT_ID_PATTERN.fullmatch(object_id):
            raise ValueError(f"anchor must be object_id=PLY with a safe object id: {value!r}")
        if object_id in seen:
            raise ValueError(f"duplicate anchor object id: {object_id}")
        if not path:
            raise ValueError(f"anchor PLY path is empty: {value!r}")
        seen.add(object_id)
        parsed.append((object_id, path))
    if not parsed:
        raise ValueError("at least one --anchor object_id=PLY is required")
    return parsed


def load_anchor(object_id: str, declared_path: str) -> Anchor:
    path = Path(declared_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    ply = PlyData.read(path)
    try:
        vertex = ply["vertex"].data
    except KeyError as error:
        raise ValueError(f"anchor PLY has no vertex element: {path}") from error
    names = set(vertex.dtype.names or ())
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError(f"anchor PLY must contain x/y/z: {path}")
    points = np.column_stack([vertex[axis] for axis in ("x", "y", "z")]).astype(
        np.float64, copy=False
    )
    if not len(points):
        raise ValueError(f"anchor PLY is empty: {path}")
    if not np.isfinite(points).all():
        raise ValueError(f"anchor PLY contains non-finite XYZ values: {path}")
    colors = None
    if {"red", "green", "blue"}.issubset(names):
        colors = rgb_stats(
            np.column_stack([vertex[channel] for channel in ("red", "green", "blue")])
        )
    return Anchor(
        object_id=object_id,
        declared_path=declared_path,
        path=path,
        sha256=sha256_file(path),
        size=path.stat().st_size,
        points=points,
        centroid=points.mean(axis=0),
        rgb=colors,
    )


def load_camera_info(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError("camera_info must be a JSON object")
    if value.get("extrinsic_type") != "world_to_camera":
        raise ValueError("camera_info.extrinsic_type must be 'world_to_camera'")
    if not isinstance(value.get("extrinsic"), dict):
        raise ValueError("camera_info.extrinsic must be a frame-keyed object")
    return value


def frame_intrinsic(camera_info: dict[str, Any], frame_id: str) -> dict[str, float]:
    intrinsics = camera_info.get("intrinsics")
    frame_camera_ids = camera_info.get("frame_camera_ids")
    value: Any = None
    if isinstance(intrinsics, dict):
        if frame_id in intrinsics:
            value = intrinsics[frame_id]
        elif isinstance(frame_camera_ids, dict):
            value = intrinsics.get(str(frame_camera_ids.get(frame_id, "")))
        elif len(intrinsics) == 1:
            value = next(iter(intrinsics.values()))
    if value is None:
        value = camera_info.get("intrinsic")
    required = ("fx", "fy", "cx", "cy", "w", "h")
    if not isinstance(value, dict) or not all(
        isinstance(value.get(key), int | float) for key in required
    ):
        raise ValueError(f"intrinsic for {frame_id} is missing numeric {required}")
    result = {key: float(value[key]) for key in required}
    if result["fx"] <= 0 or result["fy"] <= 0 or result["w"] <= 0 or result["h"] <= 0:
        raise ValueError(f"intrinsic for {frame_id} has non-positive dimensions/focal length")
    if not float(result["w"]).is_integer() or not float(result["h"]).is_integer():
        raise ValueError(f"intrinsic for {frame_id} width/height must be integers")
    return result


def world_to_camera(
    camera_info: dict[str, Any],
    frame_id: str,
    *,
    maximum_orthonormal_error: float,
) -> tuple[np.ndarray, float]:
    matrix = np.asarray(camera_info["extrinsic"].get(frame_id), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"invalid world-to-camera matrix for frame {frame_id}")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"invalid homogeneous row for frame {frame_id}")
    rotation = matrix[:3, :3]
    error = float(np.max(np.abs(rotation @ rotation.T - np.eye(3))))
    determinant = float(np.linalg.det(rotation))
    if error > maximum_orthonormal_error or abs(determinant - 1.0) > maximum_orthonormal_error:
        raise ValueError(
            f"frame {frame_id} rotation is not orthonormal: error={error}, det={determinant}"
        )
    return matrix, error


def camera_center(matrix: np.ndarray) -> np.ndarray:
    return -(matrix[:3, :3].T @ matrix[:3, 3])


def mask_bbox(mask: np.ndarray) -> list[int]:
    rows, columns = np.nonzero(mask)
    if not len(rows):
        raise ValueError("SAM3 mask is empty")
    return [
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    ]


def bbox_iou(left: list[int] | list[float], right: list[int] | list[float]) -> float:
    x0 = max(float(left[0]), float(right[0]))
    y0 = max(float(left[1]), float(right[1]))
    x1 = min(float(left[2]), float(right[2]))
    y1 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(
        0.0, float(left[3]) - float(left[1])
    )
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(
        0.0, float(right[3]) - float(right[1])
    )
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def binary_dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
    result = mask.astype(bool, copy=True)
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        result = (
            padded[1:-1, 1:-1]
            | padded[:-2, 1:-1]
            | padded[2:, 1:-1]
            | padded[1:-1, :-2]
            | padded[1:-1, 2:]
            | padded[:-2, :-2]
            | padded[:-2, 2:]
            | padded[2:, :-2]
            | padded[2:, 2:]
        )
    return result


def resolve_frame_path(frames_dir: Path, image_value: str) -> Path:
    image_name = Path(image_value).name
    exact = frames_dir / image_name
    if exact.is_file():
        return exact.resolve()
    frame_id = Path(image_name).stem
    matches = sorted(path for path in frames_dir.glob(f"{frame_id}.*") if path.is_file())
    if len(matches) != 1:
        raise ValueError(f"expected one RGB frame for {frame_id}, found {len(matches)}")
    return matches[0].resolve()


def load_detections(
    index_path: Path,
    *,
    labels: set[str] | None,
    path_remaps: list[tuple[Path, Path]],
) -> tuple[list[Detection], int]:
    value = read_json(index_path)
    if not isinstance(value, dict) or not isinstance(value.get("items"), list):
        raise ValueError("SAM3 mask index must contain an items array")
    detections: list[Detection] = []
    skipped = 0
    for item_index, item in enumerate(value["items"]):
        if not isinstance(item, dict):
            raise ValueError(f"SAM3 item {item_index} must be an object")
        label_value = item.get("label")
        label = str(label_value) if label_value is not None else None
        if labels is not None and label not in labels:
            skipped += 1
            continue
        image = item.get("image")
        mask_path_value = item.get("mask_path")
        score = item.get("score")
        bbox = item.get("bbox")
        if not isinstance(image, str) or not isinstance(mask_path_value, str):
            raise ValueError(f"SAM3 item {item_index} requires image and mask_path strings")
        if not isinstance(score, int | float) or not math.isfinite(float(score)):
            raise ValueError(f"SAM3 item {item_index} requires a finite numeric score")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(isinstance(value, int | float) for value in bbox)
        ):
            raise ValueError(f"SAM3 item {item_index} requires numeric bbox=[x0,y0,x1,y1]")
        declared_bbox = [float(value) for value in bbox]
        if declared_bbox[2] <= declared_bbox[0] or declared_bbox[3] <= declared_bbox[1]:
            raise ValueError(f"SAM3 item {item_index} bbox is empty or inverted")
        path = resolve_input_path(
            mask_path_value,
            relative_to=index_path.parent,
            path_remaps=path_remaps,
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image_handle:
            mask = np.asarray(image_handle.convert("L"), dtype=np.uint8) > 0
        actual_bbox = mask_bbox(mask)
        frame_id = Path(image).stem
        detections.append(
            Detection(
                detection_id=f"{frame_id}:mask:{item_index:06d}",
                item_index=item_index,
                frame_id=frame_id,
                image=image,
                label=label,
                score=float(score),
                declared_path=mask_path_value,
                path=path,
                sha256=sha256_file(path),
                size=path.stat().st_size,
                mask=mask,
                area_pixels=int(mask.sum()),
                actual_bbox_xyxy=actual_bbox,
                declared_bbox_xyxy=declared_bbox,
            )
        )
    if not detections:
        raise ValueError("no SAM3 detections remain after label filtering")
    return detections, skipped


def anchor_depth_map(
    points: np.ndarray,
    matrix: np.ndarray,
    intrinsic: dict[str, float],
) -> np.ndarray:
    width = int(intrinsic["w"])
    height = int(intrinsic["h"])
    # np.einsum expresses points @ R.T without the spurious Accelerate matmul
    # floating-point warnings seen for tall Nx3 arrays on macOS.
    camera = np.einsum("ni,ji->nj", points, matrix[:3, :3], optimize=False)
    camera += matrix[:3, 3]
    depth = camera[:, 2]
    valid = np.isfinite(depth) & (depth > 1e-6)
    x = np.full(len(points), -1, dtype=np.int64)
    y = np.full(len(points), -1, dtype=np.int64)
    x[valid] = np.rint(intrinsic["fx"] * camera[valid, 0] / depth[valid] + intrinsic["cx"]).astype(
        np.int64
    )
    y[valid] = np.rint(intrinsic["fy"] * camera[valid, 1] / depth[valid] + intrinsic["cy"]).astype(
        np.int64
    )
    inside = valid & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    result = np.full(height * width, np.inf, dtype=np.float64)
    if np.any(inside):
        flat = y[inside] * width + x[inside]
        np.minimum.at(result, flat, depth[inside])
    return result


def visible_anchor_projections(
    depth_maps: list[np.ndarray],
    *,
    width: int,
    height: int,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> list[dict[str, Any]]:
    if not depth_maps:
        return []
    global_depth = np.minimum.reduce(depth_maps)
    finite_global = np.isfinite(global_depth)
    tolerance = np.full_like(global_depth, absolute_tolerance)
    tolerance[finite_global] = np.maximum(
        absolute_tolerance,
        relative_tolerance * global_depth[finite_global],
    )
    result: list[dict[str, Any]] = []
    for depth_map in depth_maps:
        projected = np.isfinite(depth_map)
        visible = projected & finite_global & (depth_map <= global_depth + tolerance)
        flat_pixels = np.flatnonzero(visible)
        support = np.zeros((height, width), dtype=bool)
        support.reshape(-1)[flat_pixels] = True
        result.append(
            {
                "support": support,
                "flat_pixels": flat_pixels,
                "projected_unique_pixels": int(projected.sum()),
                "visible_unique_pixels": int(visible.sum()),
                "occluded_unique_pixels": int((projected & ~visible).sum()),
                "visible_depth_minimum": (
                    float(depth_map[visible].min()) if np.any(visible) else None
                ),
                "visible_depth_median": (
                    float(np.median(depth_map[visible])) if np.any(visible) else None
                ),
                "visible_depth_maximum": (
                    float(depth_map[visible].max()) if np.any(visible) else None
                ),
            }
        )
    return result


def pair_metric(
    projection: dict[str, Any],
    detection: Detection,
    *,
    rgb: np.ndarray,
    support_dilation: int,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    support = projection["support"]
    visible_count = int(projection["visible_unique_pixels"])
    exact_intersection = support & detection.mask
    hit_count = int(exact_intersection.sum())
    mask_hit_ratio = hit_count / visible_count if visible_count else 0.0
    dilated_support = binary_dilate(support, support_dilation)
    supported_mask_pixels = int((dilated_support & detection.mask).sum())
    dilated_support_ratio = supported_mask_pixels / detection.area_pixels
    if visible_count:
        projected_bbox = mask_bbox(support)
        bounds_iou = bbox_iou(projected_bbox, detection.actual_bbox_xyxy)
    else:
        projected_bbox = None
        bounds_iou = 0.0
    gates = {
        "minimum_visible_projected_pixels": (
            visible_count >= thresholds["minimum_visible_projected_pixels"]
        ),
        "minimum_mask_hit_ratio": mask_hit_ratio >= thresholds["minimum_mask_hit_ratio"],
        "minimum_dilated_support_ratio": (
            dilated_support_ratio >= thresholds["minimum_dilated_support_ratio"]
        ),
        "minimum_bbox_iou": bounds_iou >= thresholds["minimum_bbox_iou"],
        "minimum_sam3_score": detection.score >= thresholds["minimum_sam3_score"],
    }
    quality = (
        WEIGHTS["mask_hit_ratio"] * mask_hit_ratio
        + WEIGHTS["dilated_support_ratio"] * dilated_support_ratio
        + WEIGHTS["bbox_iou"] * bounds_iou
        + WEIGHTS["sam3_score"] * min(1.0, max(0.0, detection.score))
    )
    return {
        "detection_id": detection.detection_id,
        "eligible": all(gates.values()),
        "quality_score": float(quality),
        "gates": gates,
        "visible_projected_pixels": visible_count,
        "projected_unique_pixels_before_occlusion": int(projection["projected_unique_pixels"]),
        "occluded_projected_pixels": int(projection["occluded_unique_pixels"]),
        "projected_bbox_xyxy": projected_bbox,
        "mask_hit_pixels": hit_count,
        "mask_hit_ratio": float(mask_hit_ratio),
        "mask_pixels_supported_after_projection_dilation": supported_mask_pixels,
        "dilated_support_ratio": float(dilated_support_ratio),
        "bbox_iou": float(bounds_iou),
        "sam3_score": detection.score,
        "depth": {
            "minimum": projection["visible_depth_minimum"],
            "median": projection["visible_depth_median"],
            "maximum": projection["visible_depth_maximum"],
        },
        "rgb_stats": {
            "sam3_mask": rgb_stats(rgb[detection.mask]),
            "exact_geometry_mask_intersection": rgb_stats(rgb[exact_intersection]),
        },
    }


def maximum_weight_assignment(weights: np.ndarray) -> list[int]:
    """Return one column per row using rectangular Hungarian assignment.

    Callers append one zero-valued dummy column per row. Ineligible real pairs
    therefore lose to an unmatched dummy without requiring SciPy.
    """

    if weights.ndim != 2:
        raise ValueError("assignment weights must be a matrix")
    rows, columns = weights.shape
    if rows > columns:
        raise ValueError("Hungarian assignment requires rows <= columns")
    if not rows:
        return []
    maximum = float(np.max(weights))
    cost = maximum - weights
    u = np.zeros(rows + 1, dtype=np.float64)
    v = np.zeros(columns + 1, dtype=np.float64)
    matched_row = np.zeros(columns + 1, dtype=np.int64)
    previous_column = np.zeros(columns + 1, dtype=np.int64)
    for row in range(1, rows + 1):
        matched_row[0] = row
        minimum = np.full(columns + 1, np.inf, dtype=np.float64)
        used = np.zeros(columns + 1, dtype=bool)
        column0 = 0
        while True:
            used[column0] = True
            row0 = int(matched_row[column0])
            delta = np.inf
            column1 = 0
            for column in range(1, columns + 1):
                if used[column]:
                    continue
                current = cost[row0 - 1, column - 1] - u[row0] - v[column]
                if current < minimum[column]:
                    minimum[column] = current
                    previous_column[column] = column0
                if minimum[column] < delta:
                    delta = minimum[column]
                    column1 = column
            for column in range(columns + 1):
                if used[column]:
                    u[matched_row[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if matched_row[column0] == 0:
                break
        while True:
            column1 = int(previous_column[column0])
            matched_row[column0] = matched_row[column1]
            column0 = column1
            if column0 == 0:
                break
    assignment = [-1] * rows
    for column in range(1, columns + 1):
        row = int(matched_row[column])
        if row:
            assignment[row - 1] = column - 1
    return assignment


def angle_degrees(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    cosine = float(np.dot(left, right) / (left_norm * right_norm))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def separation_metric(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    minimum_baseline: float,
    minimum_view_angle_degrees: float,
    mode: str,
) -> dict[str, Any]:
    left_center = np.asarray(left["camera_center_world"], dtype=np.float64)
    right_center = np.asarray(right["camera_center_world"], dtype=np.float64)
    left_view = np.asarray(left["object_view_direction_world"], dtype=np.float64)
    right_view = np.asarray(right["object_view_direction_world"], dtype=np.float64)
    baseline = float(np.linalg.norm(left_center - right_center))
    view_angle = angle_degrees(left_view, right_view)
    baseline_passed = baseline >= minimum_baseline
    view_angle_passed = view_angle >= minimum_view_angle_degrees
    passed = (
        baseline_passed and view_angle_passed
        if mode == "both"
        else baseline_passed or view_angle_passed
    )
    return {
        "left_frame_id": left["frame_id"],
        "right_frame_id": right["frame_id"],
        "camera_baseline_scene_units": baseline,
        "object_view_angle_degrees": view_angle,
        "baseline_passed": baseline_passed,
        "view_angle_passed": view_angle_passed,
        "separation_mode": mode,
        "passed": passed,
    }


def select_separated_evidence(
    candidates: list[dict[str, Any]],
    *,
    minimum_frames: int,
    minimum_baseline: float,
    minimum_view_angle_degrees: float,
    mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(
        candidates,
        key=lambda item: (-item["quality_score"], item["frame_id"], item["detection_id"]),
    )
    if minimum_frames == 1 and ordered:
        return ordered[:1], []
    best: list[dict[str, Any]] = []
    best_pairs: list[dict[str, Any]] = []
    best_key: tuple[float, tuple[str, ...]] | None = None
    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1 :]:
            first_pair = separation_metric(
                left,
                right,
                minimum_baseline=minimum_baseline,
                minimum_view_angle_degrees=minimum_view_angle_degrees,
                mode=mode,
            )
            if not first_pair["passed"]:
                continue
            selected = [left, right]
            for candidate in ordered:
                if candidate in selected:
                    continue
                if all(
                    separation_metric(
                        candidate,
                        chosen,
                        minimum_baseline=minimum_baseline,
                        minimum_view_angle_degrees=minimum_view_angle_degrees,
                        mode=mode,
                    )["passed"]
                    for chosen in selected
                ):
                    selected.append(candidate)
                if len(selected) >= minimum_frames:
                    break
            if len(selected) < minimum_frames:
                continue
            selected = selected[:minimum_frames]
            pairs = [
                separation_metric(
                    selected[first],
                    selected[second],
                    minimum_baseline=minimum_baseline,
                    minimum_view_angle_degrees=minimum_view_angle_degrees,
                    mode=mode,
                )
                for first in range(len(selected))
                for second in range(first + 1, len(selected))
            ]
            key = (
                sum(item["quality_score"] for item in selected),
                tuple(item["frame_id"] for item in selected),
            )
            if best_key is None or key > best_key:
                best_key = key
                best = selected
                best_pairs = pairs
    return best, best_pairs


def parse_peeled_ids(values: list[str]) -> set[str]:
    result = {item.strip() for value in values for item in value.split(",") if item.strip()}
    invalid = sorted(item for item in result if not OBJECT_ID_PATTERN.fullmatch(item))
    if invalid:
        raise ValueError(f"invalid peeled object ids: {invalid}")
    return result


def associate(args: argparse.Namespace) -> dict[str, Any]:
    camera_info_path = args.camera_info.expanduser().resolve()
    mask_index_path = args.mask_index.expanduser().resolve()
    frames_dir = args.frames_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    path_remaps = parse_path_remaps(args.path_remap)
    anchor_values = parse_anchor_values(args.anchor)
    anchors = [load_anchor(object_id, path) for object_id, path in anchor_values]
    peeled_ids = parse_peeled_ids(args.peeled_object)
    anchor_ids = {anchor.object_id for anchor in anchors}
    unknown_peeled = sorted(peeled_ids - anchor_ids)
    if unknown_peeled:
        raise ValueError(f"peeled object ids have no explicit 3D anchor: {unknown_peeled}")

    camera_info = load_camera_info(camera_info_path)
    selected_labels = set(args.mask_label) if args.mask_label else None
    detections, skipped_label_count = load_detections(
        mask_index_path,
        labels=selected_labels,
        path_remaps=path_remaps,
    )
    detections_by_frame: dict[str, list[Detection]] = defaultdict(list)
    for detection in detections:
        detections_by_frame[detection.frame_id].append(detection)

    thresholds = {
        "minimum_visible_projected_pixels": args.min_visible_projected_pixels,
        "minimum_mask_hit_ratio": args.min_mask_hit_ratio,
        "minimum_dilated_support_ratio": args.min_dilated_support_ratio,
        "minimum_bbox_iou": args.min_bbox_iou,
        "minimum_sam3_score": args.min_sam3_score,
        "projection_support_dilation_pixels": args.support_dilation,
        "occlusion_absolute_depth_tolerance": args.occlusion_absolute_depth_tolerance,
        "occlusion_relative_depth_tolerance": args.occlusion_relative_depth_tolerance,
        "minimum_evidence_frames": args.min_evidence_frames,
        "minimum_camera_baseline_scene_units": args.min_camera_baseline,
        "minimum_object_view_angle_degrees": args.min_view_angle_degrees,
        "separation_mode": args.separation_mode,
        "maximum_rotation_orthonormal_error": args.max_rotation_orthonormal_error,
        "assignment_weights": WEIGHTS,
    }
    frame_records: list[dict[str, Any]] = []
    assignments_by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    mask_source_records: list[dict[str, Any]] = []
    frame_source_records: dict[str, dict[str, Any]] = {}
    residual_warnings: list[dict[str, Any]] = []

    for frame_id in sorted(detections_by_frame):
        frame_detections = sorted(detections_by_frame[frame_id], key=lambda item: item.detection_id)
        intrinsic = frame_intrinsic(camera_info, frame_id)
        matrix, orthonormal_error = world_to_camera(
            camera_info,
            frame_id,
            maximum_orthonormal_error=args.max_rotation_orthonormal_error,
        )
        frame_path = resolve_frame_path(frames_dir, frame_detections[0].image)
        with Image.open(frame_path) as image_handle:
            rgb = np.asarray(image_handle.convert("RGB"), dtype=np.uint8)
        expected_shape = (int(intrinsic["h"]), int(intrinsic["w"]))
        if rgb.shape[:2] != expected_shape:
            raise ValueError(
                f"frame {frame_id} shape {rgb.shape[:2]} does not match camera {expected_shape}"
            )
        for detection in frame_detections:
            if detection.mask.shape != expected_shape:
                raise ValueError(
                    f"mask {detection.detection_id} shape {detection.mask.shape} "
                    f"does not match camera {expected_shape}"
                )
            resolved_frame = resolve_frame_path(frames_dir, detection.image)
            if resolved_frame != frame_path:
                raise ValueError(f"frame {frame_id} resolves to multiple RGB files")
            mask_source_records.append(
                {
                    "detection_id": detection.detection_id,
                    "item_index": detection.item_index,
                    "frame_id": frame_id,
                    "label": detection.label,
                    "score": detection.score,
                    "declared_path": detection.declared_path,
                    "resolved_path": str(detection.path),
                    "sha256": detection.sha256,
                    "bytes": detection.size,
                    "area_pixels": detection.area_pixels,
                    "declared_bbox_xyxy": detection.declared_bbox_xyxy,
                    "actual_bbox_xyxy": detection.actual_bbox_xyxy,
                    "declared_actual_bbox_iou": bbox_iou(
                        detection.declared_bbox_xyxy, detection.actual_bbox_xyxy
                    ),
                }
            )
        frame_hash = sha256_file(frame_path)
        frame_source_records[frame_id] = {
            "frame_id": frame_id,
            "resolved_path": str(frame_path),
            "sha256": frame_hash,
            "bytes": frame_path.stat().st_size,
            "dimensions": [int(intrinsic["w"]), int(intrinsic["h"])],
        }

        depth_maps = [anchor_depth_map(anchor.points, matrix, intrinsic) for anchor in anchors]
        projections = visible_anchor_projections(
            depth_maps,
            width=int(intrinsic["w"]),
            height=int(intrinsic["h"]),
            absolute_tolerance=args.occlusion_absolute_depth_tolerance,
            relative_tolerance=args.occlusion_relative_depth_tolerance,
        )
        metrics: list[list[dict[str, Any]]] = []
        for projection in projections:
            metrics.append(
                [
                    pair_metric(
                        projection,
                        detection,
                        rgb=rgb,
                        support_dilation=args.support_dilation,
                        thresholds=thresholds,
                    )
                    for detection in frame_detections
                ]
            )
        invalid_weight = -1_000_000.0
        weights = np.zeros((len(anchors), len(frame_detections) + len(anchors)))
        for anchor_index in range(len(anchors)):
            for detection_index in range(len(frame_detections)):
                metric = metrics[anchor_index][detection_index]
                weights[anchor_index, detection_index] = (
                    metric["quality_score"] if metric["eligible"] else invalid_weight
                )
        assignment_columns = maximum_weight_assignment(weights)
        assigned_detection_indices: set[int] = set()
        frame_assignments: list[dict[str, Any]] = []
        center = camera_center(matrix)
        for anchor_index, column in enumerate(assignment_columns):
            if column < 0 or column >= len(frame_detections):
                continue
            metric = metrics[anchor_index][column]
            if not metric["eligible"]:
                continue
            anchor = anchors[anchor_index]
            detection = frame_detections[column]
            assigned_detection_indices.add(column)
            direction = anchor.centroid - center
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm > 1e-12:
                direction = direction / direction_norm
            assignment = {
                "frame_id": frame_id,
                "detection_id": detection.detection_id,
                "object_id": anchor.object_id,
                "label": detection.label,
                "quality_score": metric["quality_score"],
                "camera_center_world": center.tolist(),
                "object_view_direction_world": direction.tolist(),
                "frame_sha256": frame_hash,
                "mask_sha256": detection.sha256,
                "peeled_object_residual": anchor.object_id in peeled_ids,
                "metrics": metric,
            }
            frame_assignments.append(assignment)
            assignments_by_object[anchor.object_id].append(assignment)
            if anchor.object_id in peeled_ids:
                residual_warnings.append(
                    {
                        "code": "peeled_object_detection_residual",
                        "object_id": anchor.object_id,
                        "frame_id": frame_id,
                        "detection_id": detection.detection_id,
                        "quality_score": metric["quality_score"],
                        "message": (
                            "A high-confidence SAM3 detection still matches a peeled 3D anchor; "
                            "it is excluded from next_layer_target_ids."
                        ),
                    }
                )
        frame_records.append(
            {
                "frame_id": frame_id,
                "camera": {
                    "extrinsic_type": "world_to_camera",
                    "world_to_camera": matrix.tolist(),
                    "camera_center_world": center.tolist(),
                    "rotation_orthonormal_error": orthonormal_error,
                    "intrinsic": intrinsic,
                },
                "assignments": frame_assignments,
                "unassigned_detection_ids": [
                    detection.detection_id
                    for index, detection in enumerate(frame_detections)
                    if index not in assigned_detection_indices
                ],
                "pair_metrics": [
                    {
                        "object_id": anchor.object_id,
                        "detection_id": detection.detection_id,
                        **metrics[anchor_index][detection_index],
                    }
                    for anchor_index, anchor in enumerate(anchors)
                    for detection_index, detection in enumerate(frame_detections)
                ],
            }
        )

    object_records: list[dict[str, Any]] = []
    failure_reasons: list[str] = []
    next_layer_target_ids: list[str] = []
    for anchor in anchors:
        candidates = assignments_by_object.get(anchor.object_id, [])
        selected, separation_pairs = select_separated_evidence(
            candidates,
            minimum_frames=args.min_evidence_frames,
            minimum_baseline=args.min_camera_baseline,
            minimum_view_angle_degrees=args.min_view_angle_degrees,
            mode=args.separation_mode,
        )
        is_peeled = anchor.object_id in peeled_ids
        evidence_passed = len(selected) >= args.min_evidence_frames
        if is_peeled:
            disposition = "peeled_residual_warning" if candidates else "peeled_absent_as_expected"
        elif evidence_passed:
            disposition = "accepted_next_layer_target"
            next_layer_target_ids.append(anchor.object_id)
        else:
            disposition = "failed_insufficient_separated_evidence"
            failure_reasons.append(
                f"{anchor.object_id}: only {len(candidates)} high-confidence assignments and "
                f"{len(selected)} separated evidence frames; require {args.min_evidence_frames}"
            )
        object_records.append(
            {
                "object_id": anchor.object_id,
                "peeled": is_peeled,
                "disposition": disposition,
                "evidence_gate_applies": not is_peeled,
                "evidence_gate_passed": evidence_passed if not is_peeled else None,
                "high_confidence_assignment_count": len(candidates),
                "selected_evidence_frame_count": len(selected),
                "selected_evidence": selected,
                "selected_frame_separation": separation_pairs,
                "all_high_confidence_assignments": candidates,
            }
        )

    next_layer_target_ids.sort()
    status = "passed" if not failure_reasons else "failed"
    script_path = Path(__file__).resolve()
    source_inputs = {
        "anchors": [
            {
                "object_id": anchor.object_id,
                "declared_path": anchor.declared_path,
                "resolved_path": str(anchor.path),
                "sha256": anchor.sha256,
                "bytes": anchor.size,
                "point_count": len(anchor.points),
                "centroid_world": anchor.centroid.tolist(),
                "vertex_rgb_stats": anchor.rgb,
            }
            for anchor in anchors
        ],
        "camera_info": {
            "resolved_path": str(camera_info_path),
            "sha256": sha256_file(camera_info_path),
            "bytes": camera_info_path.stat().st_size,
            "extrinsic_type": "world_to_camera",
        },
        "sam3_mask_index": {
            "resolved_path": str(mask_index_path),
            "sha256": sha256_file(mask_index_path),
            "bytes": mask_index_path.stat().st_size,
            "selected_labels": sorted(selected_labels) if selected_labels else None,
            "loaded_detection_count": len(detections),
            "skipped_label_count": skipped_label_count,
        },
        "frames": [frame_source_records[key] for key in sorted(frame_source_records)],
        "masks": sorted(mask_source_records, key=lambda item: item["detection_id"]),
    }
    report = {
        "schema_version": 1,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 - mil8 is Python 3.10
        "execution": {
            "script": {"path": str(script_path), "sha256": sha256_file(script_path)},
            "argv": list(sys.argv),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "plyfile": "installed",
        },
        "association_contract": {
            "physical_identity_source": "explicit --anchor object_id=PLY only",
            "forbidden_identity_sources": [
                "SAM3 mask filename",
                "SAM3 per-frame ordinal suffix",
                "cross-frame item order",
            ],
            "projection": {
                "extrinsic_type": "world_to_camera",
                "world_to_camera_formula": "camera_xyz = world_xyz @ R.T + t",
                "pixel_formula": "round(focal * camera_xy / camera_z + principal_point)",
                "visibility": (
                    "one minimum-depth sample per anchor/pixel, then a cross-anchor z-buffer "
                    "with absolute-or-relative tolerance"
                ),
            },
            "metrics": {
                "mask_hit_ratio": (
                    "exact visible projected anchor pixels inside the SAM3 mask divided by "
                    "visible projected anchor pixels"
                ),
                "dilated_support_ratio": (
                    "SAM3 mask pixels covered by the dilated visible anchor projection divided "
                    "by SAM3 mask pixels"
                ),
                "bbox_iou": "visible anchor projection bbox versus decoded PNG mask bbox",
                "assignment": (
                    "maximum-weight one-to-one assignment per frame with dummy unmatched columns"
                ),
            },
        },
        "thresholds": thresholds,
        "thresholds_sha256": sha256_json(thresholds),
        "sources": source_inputs,
        "source_set_sha256": sha256_json(source_inputs),
        "peeled_object_ids": sorted(peeled_ids),
        "residual_warnings": residual_warnings,
        "next_layer_target_ids": next_layer_target_ids,
        "objects": object_records,
        "frames": frame_records,
        "gates": {
            "passed": not failure_reasons,
            "failure_reasons": failure_reasons,
            "all_non_peeled_objects_have_separated_evidence": not failure_reasons,
            "peeled_objects_excluded_from_next_layer_targets": not bool(
                peeled_ids & set(next_layer_target_ids)
            ),
        },
        "output_receipt": {
            "path": str(output_path.with_suffix(output_path.suffix + ".sha256.json")),
            "note": (
                "The sidecar hashes this report after serialization to avoid a recursive self-hash."
            ),
        },
    }
    write_json_atomic(output_path, report)
    receipt_path = output_path.with_suffix(output_path.suffix + ".sha256.json")
    write_json_atomic(
        receipt_path,
        {
            "schema_version": 1,
            "report_path": str(output_path),
            "report_sha256": sha256_file(output_path),
            "report_bytes": output_path.stat().st_size,
        },
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--anchor",
        action="append",
        default=[],
        metavar="OBJECT_ID=PLY",
        help="Explicit physical 3D identity anchor; repeat once per object.",
    )
    parser.add_argument("--camera-info", required=True, type=Path)
    parser.add_argument("--mask-index", required=True, type=Path)
    parser.add_argument("--frames-dir", required=True, type=Path)
    parser.add_argument(
        "--peeled-object",
        action="append",
        default=[],
        metavar="OBJECT_ID[,OBJECT_ID...]",
        help="Already peeled physical ids; repeat or pass a comma-separated list.",
    )
    parser.add_argument(
        "--mask-label",
        action="append",
        default=[],
        help="Optional SAM3 category label filter; repeatable and not used as instance identity.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--path-remap",
        action="append",
        default=[],
        metavar="ABSOLUTE_SOURCE=LOCAL_TARGET",
        help="Remap declared absolute mask paths to a local mirror; repeatable.",
    )
    parser.add_argument("--min-visible-projected-pixels", type=int, default=25)
    parser.add_argument("--min-mask-hit-ratio", type=float, default=0.55)
    parser.add_argument("--min-dilated-support-ratio", type=float, default=0.20)
    parser.add_argument("--min-bbox-iou", type=float, default=0.30)
    parser.add_argument("--min-sam3-score", type=float, default=0.50)
    parser.add_argument("--support-dilation", type=int, default=4)
    parser.add_argument("--occlusion-absolute-depth-tolerance", type=float, default=0.02)
    parser.add_argument("--occlusion-relative-depth-tolerance", type=float, default=0.005)
    parser.add_argument("--min-evidence-frames", type=int, default=2)
    parser.add_argument("--min-camera-baseline", type=float, default=0.10)
    parser.add_argument("--min-view-angle-degrees", type=float, default=5.0)
    parser.add_argument("--separation-mode", choices=("both", "either"), default="both")
    parser.add_argument("--max-rotation-orthonormal-error", type=float, default=1e-3)
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not args.anchor:
        parser.error("at least one --anchor OBJECT_ID=PLY is required")
    unit_intervals = (
        "min_mask_hit_ratio",
        "min_dilated_support_ratio",
        "min_bbox_iou",
        "min_sam3_score",
        "occlusion_relative_depth_tolerance",
    )
    for name in unit_intervals:
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    nonnegative = (
        "support_dilation",
        "occlusion_absolute_depth_tolerance",
        "min_camera_baseline",
        "min_view_angle_degrees",
        "max_rotation_orthonormal_error",
    )
    for name in nonnegative:
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if args.min_visible_projected_pixels < 1:
        parser.error("--min-visible-projected-pixels must be at least 1")
    if args.min_evidence_frames < 2:
        parser.error("--min-evidence-frames must be at least 2")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    try:
        report = associate(args)
    except (FileNotFoundError, KeyError, OSError, ValueError) as error:
        parser.exit(2, f"association failed closed: {error}\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(args.output.expanduser().resolve()),
                "next_layer_target_ids": report["next_layer_target_ids"],
                "residual_warning_count": len(report["residual_warnings"]),
                "failure_reasons": report["gates"]["failure_reasons"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
