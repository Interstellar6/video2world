#!/usr/bin/env python3
"""Associate targeted SAM3 potted-plant masks with exact scene anchors.

The tool consumes independently generated text-prompt masks. It assigns complete
``potted plant`` candidates to physical objects by projecting their hash-bound
3D anchors, verifies a separately prompted ``flower pot`` or ``planter`` mask
immediately below each anchor-associated foliage mask, and rejects candidates
that absorb neighboring nightstand or lamp pixels. No mask dilation, bbox
expansion, inpainting, or category relabeling is performed.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps

from scripts.fit_completed_object_to_scene import load_ply_points
from scripts.qa_scene_camera_silhouette import load_camera
from video2world.hashing import atomic_write_json, sha256_file

CONFIG_KIND = "video2world.targeted_potted_plant_mask_config"
SOURCE_AUDIT_KIND = "video2world.multiview_scene_fit_evidence_audit"
MANIFEST_KIND = "video2world.targeted_potted_plant_mask_manifest"


@dataclass(frozen=True)
class MaskCandidate:
    frame_id: str
    label: str
    score: float
    path: Path
    declared_path: str
    declared_bbox: list[float]
    rle_size: list[int]
    sha256: str
    size_bytes: int
    mask: np.ndarray
    bbox: list[int]
    pixel_count: int


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def resolve_path(value: Any, *, relative_to: Path, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def verified_path(record: Any, *, relative_to: Path, label: str) -> tuple[Path, str, int]:
    require(isinstance(record, dict), f"{label} must be an object")
    path = resolve_path(record.get("path"), relative_to=relative_to, label=f"{label}.path")
    require(path.is_file(), f"{label} is not a file: {path}")
    expected = record.get("sha256")
    require(
        isinstance(expected, str) and len(expected) == 64,
        f"{label}.sha256 must be a 64-character digest",
    )
    observed, size = sha256_file(path)
    require(observed == expected, f"{label} SHA-256 mismatch: {observed} != {expected}")
    return path, observed, size


def _number(value: Any, *, label: str, minimum: float, maximum: float) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be numeric",
    )
    result = float(value)
    require(math.isfinite(result), f"{label} must be finite")
    require(minimum <= result <= maximum, f"{label} must be in [{minimum}, {maximum}]")
    return result


def _integer(value: Any, *, label: str, minimum: int) -> int:
    require(isinstance(value, int) and not isinstance(value, bool), f"{label} must be integer")
    require(value >= minimum, f"{label} must be >= {minimum}")
    return value


def _string_list(value: Any, *, label: str) -> list[str]:
    require(isinstance(value, list) and value, f"{label} must be a non-empty list")
    require(all(isinstance(item, str) and item for item in value), f"{label} is invalid")
    result = [item.casefold() for item in value]
    require(len(result) == len(set(result)), f"{label} must not contain duplicates")
    return result


def _validate_remote_artifact(record: Any, *, label: str) -> dict[str, Any]:
    require(isinstance(record, dict), f"{label} must be an object")
    path = record.get("path")
    digest = record.get("sha256")
    size = record.get("size_bytes")
    require(isinstance(path, str) and path.startswith("/"), f"{label}.path must be absolute")
    require(isinstance(digest, str) and len(digest) == 64, f"{label}.sha256 is invalid")
    require(isinstance(size, int) and size > 0, f"{label}.size_bytes must be positive")
    return {
        "path": path,
        "sha256": digest,
        "size_bytes": size,
        "verification_scope": "sha256_observed_on_remote_inference_host",
    }


def parse_config(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = read_json(config_path)
    require(config.get("schema_version") == 1, "config.schema_version must be 1")
    require(config.get("kind") == CONFIG_KIND, f"config.kind must be {CONFIG_KIND}")
    base = config_path.parent
    source_audit_path, source_audit_sha, source_audit_size = verified_path(
        config.get("source_audit"), relative_to=base, label="source_audit"
    )
    cameras_path, cameras_sha, cameras_size = verified_path(
        config.get("cameras"), relative_to=base, label="cameras"
    )
    original_index_path, original_index_sha, original_index_size = verified_path(
        config.get("original_mask_index"), relative_to=base, label="original_mask_index"
    )
    targeted_index_path, targeted_index_sha, targeted_index_size = verified_path(
        config.get("targeted_mask_index"), relative_to=base, label="targeted_mask_index"
    )
    original_root = resolve_path(
        config.get("original_mask_root"), relative_to=base, label="original_mask_root"
    )
    targeted_root = resolve_path(
        config.get("targeted_mask_root"), relative_to=base, label="targeted_mask_root"
    )
    require(original_root.is_dir(), f"original_mask_root is not a directory: {original_root}")
    require(targeted_root.is_dir(), f"targeted_mask_root is not a directory: {targeted_root}")
    object_ids = config.get("object_ids")
    require(isinstance(object_ids, list) and object_ids, "object_ids must be non-empty")
    require(
        all(isinstance(item, str) and item for item in object_ids),
        "object_ids entries must be non-empty strings",
    )
    require(len(object_ids) == len(set(object_ids)), "object_ids must be unique")
    prompt_contract = config.get("prompt_contract")
    require(isinstance(prompt_contract, dict), "prompt_contract must be an object")
    complete_label = prompt_contract.get("complete_instance")
    require(
        isinstance(complete_label, str) and complete_label,
        "prompt_contract.complete_instance must be non-empty",
    )
    complete_label = complete_label.casefold()
    part_labels = _string_list(
        prompt_contract.get("container_parts"), label="prompt_contract.container_parts"
    )
    contaminant_labels = _string_list(
        prompt_contract.get("contaminant_labels"), label="prompt_contract.contaminant_labels"
    )
    require(complete_label not in part_labels, "complete and part prompt labels must differ")
    thresholds = config.get("thresholds")
    require(isinstance(thresholds, dict), "thresholds must be an object")
    parsed_thresholds = {
        "minimum_complete_anchor_hit_ratio": _number(
            thresholds.get("minimum_complete_anchor_hit_ratio"),
            label="thresholds.minimum_complete_anchor_hit_ratio",
            minimum=0.0,
            maximum=1.0,
        ),
        "minimum_complete_anchor_hit_margin": _number(
            thresholds.get("minimum_complete_anchor_hit_margin"),
            label="thresholds.minimum_complete_anchor_hit_margin",
            minimum=0.0,
            maximum=1.0,
        ),
        "minimum_foliage_coverage": _number(
            thresholds.get("minimum_foliage_coverage"),
            label="thresholds.minimum_foliage_coverage",
            minimum=0.0,
            maximum=1.0,
        ),
        "minimum_container_inside_complete": _number(
            thresholds.get("minimum_container_inside_complete"),
            label="thresholds.minimum_container_inside_complete",
            minimum=0.0,
            maximum=1.0,
        ),
        "minimum_container_outside_foliage": _number(
            thresholds.get("minimum_container_outside_foliage"),
            label="thresholds.minimum_container_outside_foliage",
            minimum=0.0,
            maximum=1.0,
        ),
        "maximum_contaminant_overlap": _number(
            thresholds.get("maximum_contaminant_overlap"),
            label="thresholds.maximum_contaminant_overlap",
            minimum=0.0,
            maximum=1.0,
        ),
        "maximum_horizontal_center_offset_foliage_widths": _number(
            thresholds.get("maximum_horizontal_center_offset_foliage_widths"),
            label="thresholds.maximum_horizontal_center_offset_foliage_widths",
            minimum=0.0,
            maximum=2.0,
        ),
        "maximum_container_top_gap_foliage_heights": _number(
            thresholds.get("maximum_container_top_gap_foliage_heights"),
            label="thresholds.maximum_container_top_gap_foliage_heights",
            minimum=0.0,
            maximum=2.0,
        ),
        "minimum_complete_bottom_extension_foliage_heights": _number(
            thresholds.get("minimum_complete_bottom_extension_foliage_heights"),
            label="thresholds.minimum_complete_bottom_extension_foliage_heights",
            minimum=0.0,
            maximum=3.0,
        ),
    }
    review = config.get("review")
    require(isinstance(review, dict), "review must be an object")
    parsed_review = {
        "columns": _integer(review.get("columns"), label="review.columns", minimum=1),
        "panel_width": _integer(review.get("panel_width"), label="review.panel_width", minimum=120),
        "panel_height": _integer(
            review.get("panel_height"), label="review.panel_height", minimum=100
        ),
        "crop_padding_ratio": _number(
            review.get("crop_padding_ratio"),
            label="review.crop_padding_ratio",
            minimum=0.1,
            maximum=1.5,
        ),
    }
    inference = config.get("inference")
    require(isinstance(inference, dict), "inference must be an object")
    scene_path, scene_sha, scene_size = verified_path(
        inference.get("scene_config"), relative_to=base, label="inference.scene_config"
    )
    runner_path, runner_sha, runner_size = verified_path(
        inference.get("runner"), relative_to=base, label="inference.runner"
    )
    log_path, log_sha, log_size = verified_path(
        inference.get("log"), relative_to=base, label="inference.log"
    )
    argv = inference.get("argv")
    require(
        isinstance(argv, list) and argv and all(isinstance(item, str) for item in argv),
        "inference.argv must be a non-empty string array",
    )
    environment = inference.get("environment")
    require(isinstance(environment, dict) and environment, "inference.environment is invalid")
    require(
        all(isinstance(key, str) and isinstance(value, str) for key, value in environment.items()),
        "inference.environment must contain string values",
    )
    remote_host = inference.get("remote_host")
    remote_output = inference.get("remote_output_directory")
    require(isinstance(remote_host, str) and remote_host, "inference.remote_host is invalid")
    require(
        isinstance(remote_output, str) and remote_output.startswith("/"),
        "inference.remote_output_directory must be absolute",
    )
    source_files = inference.get("sam3_source_files")
    require(isinstance(source_files, list) and source_files, "sam3_source_files is invalid")
    output_directory = resolve_path(
        config.get("output_directory"), relative_to=base, label="output_directory"
    )
    config_sha, config_size = sha256_file(config_path)
    return {
        "path": config_path,
        "sha256": config_sha,
        "size_bytes": config_size,
        "source_audit": (source_audit_path, source_audit_sha, source_audit_size),
        "cameras": (cameras_path, cameras_sha, cameras_size),
        "original_index": (original_index_path, original_index_sha, original_index_size),
        "targeted_index": (targeted_index_path, targeted_index_sha, targeted_index_size),
        "original_root": original_root,
        "targeted_root": targeted_root,
        "object_ids": object_ids,
        "complete_label": complete_label,
        "part_labels": part_labels,
        "contaminant_labels": contaminant_labels,
        "thresholds": parsed_thresholds,
        "review": parsed_review,
        "output_directory": output_directory,
        "inference": {
            "remote_host": remote_host,
            "remote_output_directory": remote_output,
            "scene_config": {
                "path": str(scene_path),
                "sha256": scene_sha,
                "size_bytes": scene_size,
            },
            "runner": {
                "path": str(runner_path),
                "sha256": runner_sha,
                "size_bytes": runner_size,
            },
            "log": {"path": str(log_path), "sha256": log_sha, "size_bytes": log_size},
            "checkpoint": _validate_remote_artifact(
                inference.get("checkpoint"), label="inference.checkpoint"
            ),
            "sam3_source_files": [
                _validate_remote_artifact(item, label=f"inference.sam3_source_files[{index}]")
                for index, item in enumerate(source_files)
            ],
            "argv": argv,
            "environment": environment,
        },
    }


def _bbox(mask: np.ndarray) -> list[int]:
    rows, columns = np.nonzero(mask)
    require(len(rows) > 0, "mask is empty")
    return [
        int(columns.min()),
        int(rows.min()),
        int(columns.max() + 1),
        int(rows.max() + 1),
    ]


def _mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    rows, columns = np.nonzero(mask)
    require(len(rows) > 0, "mask is empty")
    return float(columns.mean()), float(rows.mean())


def _load_mask(path: Path, *, shape: tuple[int, int], label: str) -> np.ndarray:
    require(path.is_file(), f"{label} is missing: {path}")
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L")) > 0
    require(mask.shape == shape, f"{label} dimensions are {mask.shape}, expected {shape}")
    require(np.any(mask), f"{label} is empty: {path}")
    return mask


def _decode_compressed_coco_rle(
    record: Any,
    *,
    shape: tuple[int, int],
    label: str,
) -> np.ndarray:
    """Decode the compressed COCO RLE emitted by the SAM3 runner.

    Keeping this small decoder local avoids making pycocotools a runtime
    dependency while still binding every synced PNG to the hash-bound index.
    """

    require(isinstance(record, dict), f"{label}.mask_rle must be an object")
    size = record.get("size")
    require(
        isinstance(size, list)
        and len(size) == 2
        and all(isinstance(value, int) and not isinstance(value, bool) for value in size),
        f"{label}.mask_rle.size must contain two integers",
    )
    require(tuple(size) == shape, f"{label}.mask_rle.size is {size}, expected {list(shape)}")
    encoded = record.get("counts")
    require(
        isinstance(encoded, str) and encoded,
        f"{label}.mask_rle.counts must be a non-empty compressed string",
    )

    counts: list[int] = []
    position = 0
    while position < len(encoded):
        value = 0
        shift = 0
        continuation = True
        terminal = 0
        while continuation:
            require(position < len(encoded), f"{label}.mask_rle.counts is truncated")
            terminal = ord(encoded[position]) - 48
            require(0 <= terminal <= 0x3F, f"{label}.mask_rle.counts has invalid bytes")
            value |= (terminal & 0x1F) << shift
            continuation = bool(terminal & 0x20)
            position += 1
            shift += 5
            require(shift <= 65, f"{label}.mask_rle.counts contains an oversized run")
        if terminal & 0x10:
            value |= -1 << shift
        if len(counts) > 2:
            value += counts[-2]
        require(value >= 0, f"{label}.mask_rle.counts contains a negative run")
        counts.append(value)

    flat_size = shape[0] * shape[1]
    require(sum(counts) == flat_size, f"{label}.mask_rle.counts does not cover the image")
    flat = np.zeros(flat_size, dtype=bool)
    offset = 0
    foreground = False
    for count in counts:
        if foreground:
            flat[offset : offset + count] = True
        offset += count
        foreground = not foreground
    return flat.reshape(shape, order="F")


def _load_cameras(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, list) and payload, "cameras must be a non-empty array")
    cameras: dict[str, dict[str, Any]] = {}
    for item in payload:
        require(isinstance(item, dict), "camera entries must be objects")
        frame_id = str(item.get("img_name"))
        require(frame_id not in cameras, f"duplicate camera: {frame_id}")
        cameras[frame_id] = load_camera(path, frame_id)
    return cameras


def _project_anchor(points: np.ndarray, camera: dict[str, Any]) -> dict[str, Any]:
    relative = points - camera["position"]
    camera_points = np.einsum("ni,ij->nj", relative, camera["rotation"], optimize=False)
    depth = camera_points[:, 2]
    visible = depth > 1e-6
    x = np.full(len(points), np.nan, dtype=np.float64)
    y = np.full(len(points), np.nan, dtype=np.float64)
    x[visible] = camera["fx"] * camera_points[visible, 0] / depth[visible] + camera["width"] / 2
    y[visible] = camera["height"] / 2 + camera["fy"] * camera_points[visible, 1] / depth[visible]
    inside = visible & (x >= 0) & (x < camera["width"]) & (y >= 0) & (y < camera["height"])
    require(np.any(inside), "anchor has no projected points inside the frame")
    projected = np.column_stack((x[inside], y[inside]))
    pixels = np.rint(projected).astype(np.int64)
    pixels[:, 0] = np.clip(pixels[:, 0], 0, camera["width"] - 1)
    pixels[:, 1] = np.clip(pixels[:, 1], 0, camera["height"] - 1)
    return {
        "pixels": pixels,
        "point_count": len(pixels),
        "bbox_xyxy": [
            float(projected[:, 0].min()),
            float(projected[:, 1].min()),
            float(projected[:, 0].max()),
            float(projected[:, 1].max()),
        ],
    }


def _load_targeted_candidates(
    *, index_path: Path, root: Path, cameras: dict[str, dict[str, Any]]
) -> dict[str, dict[str, list[MaskCandidate]]]:
    payload = read_json(index_path)
    require(payload.get("scene") == "bedroom_4", "targeted mask index scene must be bedroom_4")
    require(payload.get("missing_images") == [], "targeted mask index contains missing images")
    items = payload.get("items")
    require(isinstance(items, list) and items, "targeted mask index items must be non-empty")
    result: dict[str, dict[str, list[MaskCandidate]]] = {}
    for item in items:
        require(isinstance(item, dict), "targeted mask index items must be objects")
        image_name = item.get("image")
        label = item.get("label")
        declared_path = item.get("mask_path")
        require(
            isinstance(image_name, str)
            and isinstance(label, str)
            and isinstance(declared_path, str),
            "targeted mask index item is incomplete",
        )
        frame_id = Path(image_name).stem
        require(frame_id in cameras, f"targeted mask frame has no camera: {frame_id}")
        score = float(item.get("score"))
        require(math.isfinite(score) and 0 <= score <= 1, "targeted mask score is invalid")
        declared_bbox = [float(value) for value in item.get("bbox", [])]
        require(len(declared_bbox) == 4, "targeted mask bbox is invalid")
        path = (root / frame_id / Path(declared_path).name).resolve()
        camera = cameras[frame_id]
        mask = _load_mask(
            path,
            shape=(camera["height"], camera["width"]),
            label=f"targeted mask {frame_id}/{path.name}",
        )
        rle_mask = _decode_compressed_coco_rle(
            item.get("mask_rle"),
            shape=mask.shape,
            label=f"targeted mask index item {frame_id}/{path.name}",
        )
        require(
            np.array_equal(mask, rle_mask),
            f"targeted mask PNG differs from hash-bound index RLE: {frame_id}/{path.name}",
        )
        digest, size = sha256_file(path)
        candidate = MaskCandidate(
            frame_id=frame_id,
            label=label.casefold(),
            score=score,
            path=path,
            declared_path=declared_path,
            declared_bbox=declared_bbox,
            rle_size=list(mask.shape),
            sha256=digest,
            size_bytes=size,
            mask=mask,
            bbox=_bbox(mask),
            pixel_count=int(np.count_nonzero(mask)),
        )
        result.setdefault(frame_id, {}).setdefault(candidate.label, []).append(candidate)
    for frame in result.values():
        for candidates in frame.values():
            candidates.sort(key=lambda candidate: candidate.path.name)
    return result


def _load_original_index(path: Path) -> dict[str, dict[str, list[str]]]:
    payload = read_json(path)
    items = payload.get("items")
    require(isinstance(items, list) and items, "original mask index items must be non-empty")
    result: dict[str, dict[str, list[str]]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        image_name = item.get("image")
        label = item.get("label")
        declared_path = item.get("mask_path")
        if not all(
            isinstance(value, str) and value for value in (image_name, label, declared_path)
        ):
            continue
        result.setdefault(Path(image_name).stem, {}).setdefault(label.casefold(), []).append(
            Path(declared_path).name
        )
    return result


def _best_assignment(
    *,
    object_ids: list[str],
    candidates: list[MaskCandidate],
    score: Callable[[str, MaskCandidate], float],
    label: str,
) -> dict[str, MaskCandidate]:
    require(len(candidates) >= len(object_ids), f"not enough {label} candidates")
    require(len(candidates) <= 8, f"too many {label} candidates for exact assignment")
    scored: list[tuple[float, tuple[str, ...], tuple[MaskCandidate, ...]]] = []
    for assignment in itertools.permutations(candidates, len(object_ids)):
        total = sum(
            score(object_id, candidate)
            for object_id, candidate in zip(object_ids, assignment, strict=True)
        )
        names = tuple(candidate.path.name for candidate in assignment)
        scored.append((total, names, assignment))
    _, _, selected = max(scored, key=lambda item: (item[0], tuple(reversed(item[1]))))
    return dict(zip(object_ids, selected, strict=True))


def _relative(path: Path, *, relative_to: Path) -> str:
    return Path(os.path.relpath(path, relative_to)).as_posix()


def _copy_exact(source: Path, target: Path) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)
    source_sha, source_size = sha256_file(source)
    target_sha, target_size = sha256_file(target)
    require(
        (target_sha, target_size) == (source_sha, source_size),
        f"copied mask bytes changed: {source} -> {target}",
    )
    return {"path": target, "sha256": target_sha, "size_bytes": target_size}


def _mask_record(candidate: MaskCandidate) -> dict[str, Any]:
    return {
        "path": str(candidate.path),
        "declared_remote_path": candidate.declared_path,
        "sha256": candidate.sha256,
        "size_bytes": candidate.size_bytes,
        "label": candidate.label,
        "sam3_score": candidate.score,
        "mask_index_bbox_xyxy": candidate.declared_bbox,
        "mask_index_rle_size_hw": candidate.rle_size,
        "png_exactly_matches_mask_index_rle": True,
        "bbox_xyxy_exclusive": candidate.bbox,
        "pixel_count": candidate.pixel_count,
    }


def _overlay(source: Image.Image, mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    dark = ImageEnhance.Brightness(source).enhance(0.34)
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
    foreground = Image.composite(source, dark, mask_image)
    color_image = Image.new("RGB", source.size, color)
    tinted = Image.blend(foreground, color_image, 0.34)
    return Image.composite(tinted, dark, mask_image)


def _fit_panel(image: Image.Image, *, width: int, height: int) -> Image.Image:
    fitted = ImageOps.contain(image, (width, height), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (width, height), (21, 24, 28))
    panel.paste(fitted, ((width - fitted.width) // 2, (height - fitted.height) // 2))
    return panel


def _crop_box(
    masks: list[np.ndarray], projection: list[float], *, width: int, height: int, ratio: float
) -> list[int]:
    combined = np.logical_or.reduce(masks)
    box = _bbox(combined)
    x0 = min(box[0], math.floor(projection[0]))
    y0 = min(box[1], math.floor(projection[1]))
    x1 = max(box[2], math.ceil(projection[2]))
    y1 = max(box[3], math.ceil(projection[3]))
    pad_x = max(12, math.ceil((x1 - x0) * ratio))
    pad_y = max(12, math.ceil((y1 - y0) * ratio))
    return [max(0, x0 - pad_x), max(0, y0 - pad_y), min(width, x1 + pad_x), min(height, y1 + pad_y)]


def _draw_projection(panel: Image.Image, projection: list[float], crop: list[int]) -> None:
    crop_width = crop[2] - crop[0]
    crop_height = crop[3] - crop[1]
    scale = min(panel.width / crop_width, panel.height / crop_height)
    rendered_width = crop_width * scale
    rendered_height = crop_height * scale
    offset_x = (panel.width - rendered_width) / 2
    offset_y = (panel.height - rendered_height) / 2
    points = [
        offset_x + (projection[0] - crop[0]) * scale,
        offset_y + (projection[1] - crop[1]) * scale,
        offset_x + (projection[2] - crop[0]) * scale,
        offset_y + (projection[3] - crop[1]) * scale,
    ]
    ImageDraw.Draw(panel).rectangle(points, outline=(255, 176, 32), width=2)


def _render_tile(
    *,
    source: Image.Image,
    foliage: np.ndarray,
    container: np.ndarray,
    complete: np.ndarray,
    projection: list[float],
    title: str,
    panel_width: int,
    panel_height: int,
    padding_ratio: float,
) -> Image.Image:
    crop = _crop_box(
        [foliage, container, complete],
        projection,
        width=source.width,
        height=source.height,
        ratio=padding_ratio,
    )
    images = [
        source.crop(crop),
        _overlay(source, foliage, (18, 210, 116)).crop(crop),
        _overlay(source, container, (40, 190, 235)).crop(crop),
        _overlay(source, complete, (236, 116, 35)).crop(crop),
    ]
    panels = [_fit_panel(image, width=panel_width, height=panel_height) for image in images]
    for panel in panels:
        _draw_projection(panel, projection, crop)
    label_height = 38
    tile = Image.new("RGB", (panel_width * 4, panel_height + label_height), (13, 16, 20))
    draw = ImageDraw.Draw(tile)
    draw.text((6, 4), title, fill=(238, 241, 245), font=ImageFont.load_default())
    draw.text(
        (6, 20),
        "source | original foliage | targeted pot | targeted potted plant",
        fill=(160, 169, 180),
        font=ImageFont.load_default(),
    )
    for index, panel in enumerate(panels):
        tile.paste(panel, (index * panel_width, label_height))
    return tile


def _contact_sheet(tiles: list[Image.Image], *, columns: int, output_path: Path) -> dict[str, Any]:
    require(tiles, "contact sheet requires tiles")
    rows = math.ceil(len(tiles) / columns)
    sheet = Image.new("RGB", (tiles[0].width * columns, tiles[0].height * rows), (12, 15, 18))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * tile.width, (index // columns) * tile.height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp.png")
    sheet.save(temporary, format="PNG", optimize=True)
    os.replace(temporary, output_path)
    digest, size = sha256_file(output_path)
    return {
        "path": output_path,
        "sha256": digest,
        "size_bytes": size,
        "dimensions_wh": [sheet.width, sheet.height],
        "columns": columns,
        "rows": rows,
    }


def materialize(config_path: Path) -> dict[str, Any]:
    parsed = parse_config(config_path)
    source_audit_path, source_audit_sha, source_audit_size = parsed["source_audit"]
    cameras_path, cameras_sha, cameras_size = parsed["cameras"]
    original_index_path, original_index_sha, original_index_size = parsed["original_index"]
    targeted_index_path, targeted_index_sha, targeted_index_size = parsed["targeted_index"]
    source_audit = read_json(source_audit_path)
    require(
        source_audit.get("kind") == SOURCE_AUDIT_KIND,
        f"source audit kind must be {SOURCE_AUDIT_KIND}",
    )
    require(
        source_audit.get("status") == "source_multiview_modal_masks_ready_for_scene_fit",
        "source audit status is not ready for scene fit",
    )
    cameras = _load_cameras(cameras_path)
    candidates = _load_targeted_candidates(
        index_path=targeted_index_path,
        root=parsed["targeted_root"],
        cameras=cameras,
    )
    original_index = _load_original_index(original_index_path)
    audit_objects = {
        item.get("object_id"): item
        for item in source_audit.get("objects", [])
        if isinstance(item, dict)
    }
    object_ids = parsed["object_ids"]
    require(
        set(object_ids).issubset(audit_objects), "configured object is missing from source audit"
    )
    object_views: dict[str, dict[str, dict[str, Any]]] = {}
    anchor_points: dict[str, np.ndarray] = {}
    anchor_records: dict[str, dict[str, Any]] = {}
    frame_sets: list[set[str]] = []
    for object_id in object_ids:
        raw = audit_objects[object_id]
        views = raw.get("views")
        require(isinstance(views, list) and views, f"{object_id} has no source views")
        by_frame = {str(view.get("frame_id")): view for view in views if isinstance(view, dict)}
        require(len(by_frame) == len(views), f"{object_id} has duplicate frame ids")
        object_views[object_id] = by_frame
        frame_sets.append(set(by_frame))
        source_anchor = raw.get("source_anchor")
        require(isinstance(source_anchor, dict), f"{object_id} source_anchor is missing")
        anchor_path = Path(str(source_anchor.get("path"))).expanduser().resolve()
        expected_sha = source_anchor.get("sha256")
        require(anchor_path.is_file(), f"{object_id} source anchor is missing: {anchor_path}")
        require(isinstance(expected_sha, str) and len(expected_sha) == 64, "anchor SHA is invalid")
        points, observed_sha = load_ply_points(anchor_path, expected_sha256=expected_sha)
        anchor_points[object_id] = points
        anchor_records[object_id] = {
            "path": str(anchor_path),
            "sha256": observed_sha,
            "size_bytes": anchor_path.stat().st_size,
            "point_count": len(points),
        }
    require(all(frame_set == frame_sets[0] for frame_set in frame_sets), "object frame sets differ")
    frame_ids = sorted(frame_sets[0])
    require(set(frame_ids) == set(candidates), "targeted mask frames differ from source audit")
    output_directory: Path = parsed["output_directory"]
    manifest_path = output_directory / "targeted-potted-plant-mask-manifest.json"
    manifest_dir = manifest_path.parent
    object_reports: dict[str, dict[str, Any]] = {
        object_id: {"object_id": object_id, "frames": [], "tiles": []} for object_id in object_ids
    }
    thresholds = parsed["thresholds"]
    for frame_id in frame_ids:
        camera = cameras.get(frame_id)
        require(camera is not None, f"camera is missing for {frame_id}")
        shape = (camera["height"], camera["width"])
        projections = {
            object_id: _project_anchor(anchor_points[object_id], camera) for object_id in object_ids
        }
        complete_candidates = candidates[frame_id].get(parsed["complete_label"], [])
        require(
            len(complete_candidates) == len(object_ids),
            f"{frame_id} requires exactly {len(object_ids)} complete candidates",
        )
        complete_hits = {
            object_id: {
                candidate.path.name: float(
                    np.mean(
                        candidate.mask[
                            projections[object_id]["pixels"][:, 1],
                            projections[object_id]["pixels"][:, 0],
                        ]
                    )
                )
                for candidate in complete_candidates
            }
            for object_id in object_ids
        }
        complete_assignment = _best_assignment(
            object_ids=object_ids,
            candidates=complete_candidates,
            score=lambda object_id, candidate, hits=complete_hits: hits[object_id][
                candidate.path.name
            ],
            label=parsed["complete_label"],
        )
        source_images: dict[str, Image.Image] = {}
        foliage_masks: dict[str, np.ndarray] = {}
        contaminant_union = np.zeros(shape, dtype=bool)
        contaminant_records: list[dict[str, Any]] = []
        for contaminant_label in parsed["contaminant_labels"]:
            for file_name in original_index.get(frame_id, {}).get(contaminant_label, []):
                path = (parsed["original_root"] / frame_id / file_name).resolve()
                mask = _load_mask(
                    path, shape=shape, label=f"contaminant mask {frame_id}/{file_name}"
                )
                contaminant_union |= mask
                digest, size = sha256_file(path)
                contaminant_records.append(
                    {
                        "label": contaminant_label,
                        "path": str(path),
                        "sha256": digest,
                        "size_bytes": size,
                        "pixel_count": int(np.count_nonzero(mask)),
                    }
                )
        require(contaminant_records, f"no contaminant masks exist for {frame_id}")
        part_assignments: dict[str, dict[str, MaskCandidate]] = {}
        for object_id in object_ids:
            view = object_views[object_id][frame_id]
            require(
                view.get("association", {}).get("status") == "passed", "source association failed"
            )
            source_record = view.get("source_rgb")
            foliage_record = view.get("mask")
            require(isinstance(source_record, dict), "source RGB record is missing")
            require(isinstance(foliage_record, dict), "foliage mask record is missing")
            source_path = Path(str(source_record.get("path"))).expanduser().resolve()
            source_sha, _ = sha256_file(source_path)
            require(
                source_sha == source_record.get("sha256"), f"source RGB hash changed: {source_path}"
            )
            source = Image.open(source_path).convert("RGB")
            require(source.size == (camera["width"], camera["height"]), "source RGB size mismatch")
            source_images[object_id] = source
            foliage_path = Path(str(foliage_record.get("path"))).expanduser().resolve()
            foliage_sha, _ = sha256_file(foliage_path)
            require(
                foliage_sha == foliage_record.get("sha256"),
                f"foliage mask hash changed: {foliage_path}",
            )
            foliage_masks[object_id] = _load_mask(
                foliage_path, shape=shape, label=f"foliage mask {object_id}/{frame_id}"
            )
        for part_label in parsed["part_labels"]:
            part_candidates = candidates[frame_id].get(part_label, [])
            require(
                len(part_candidates) == len(object_ids),
                f"{frame_id} requires exactly {len(object_ids)} {part_label} candidates",
            )

            def part_score(
                object_id: str,
                candidate: MaskCandidate,
                assignments: dict[str, MaskCandidate] = complete_assignment,
                foliage_by_object: dict[str, np.ndarray] = foliage_masks,
            ) -> float:
                complete = assignments[object_id].mask
                foliage = foliage_by_object[object_id]
                inside = float(np.count_nonzero(candidate.mask & complete) / candidate.pixel_count)
                candidate_x, _ = _mask_centroid(candidate.mask)
                foliage_x, _ = _mask_centroid(foliage)
                foliage_width = max(1, _bbox(foliage)[2] - _bbox(foliage)[0])
                horizontal = abs(candidate_x - foliage_x) / foliage_width
                return inside - 0.05 * horizontal + 0.02 * candidate.score

            part_assignments[part_label] = _best_assignment(
                object_ids=object_ids,
                candidates=part_candidates,
                score=part_score,
                label=part_label,
            )
        for object_id in object_ids:
            view = object_views[object_id][frame_id]
            source = source_images[object_id]
            foliage = foliage_masks[object_id]
            complete_candidate = complete_assignment[object_id]
            complete = complete_candidate.mask
            selected_hit = complete_hits[object_id][complete_candidate.path.name]
            runner_up_hit = max(
                complete_hits[object_id][candidate.path.name]
                for candidate in complete_candidates
                if candidate.path != complete_candidate.path
            )
            hit_margin = selected_hit - runner_up_hit
            foliage_pixels = int(np.count_nonzero(foliage))
            foliage_coverage = float(np.count_nonzero(complete & foliage) / foliage_pixels)
            complete_contamination = float(
                np.count_nonzero(complete & contaminant_union) / complete_candidate.pixel_count
            )
            foliage_box = _bbox(foliage)
            foliage_width = max(1, foliage_box[2] - foliage_box[0])
            foliage_height = max(1, foliage_box[3] - foliage_box[1])
            foliage_x, foliage_y = _mask_centroid(foliage)
            complete_bottom_extension = float(
                (complete_candidate.bbox[3] - foliage_box[3]) / foliage_height
            )
            part_records = []
            for part_label in parsed["part_labels"]:
                part = part_assignments[part_label][object_id]
                part_x, part_y = _mask_centroid(part.mask)
                inside_complete = float(np.count_nonzero(part.mask & complete) / part.pixel_count)
                outside_foliage = float(np.count_nonzero(part.mask & ~foliage) / part.pixel_count)
                contamination = float(
                    np.count_nonzero(part.mask & contaminant_union) / part.pixel_count
                )
                horizontal_offset = float(abs(part_x - foliage_x) / foliage_width)
                vertical_offset = float((part_y - foliage_y) / foliage_height)
                top_gap = float(max(0, part.bbox[1] - foliage_box[3]) / foliage_height)
                gates = {
                    "container_inside_complete": inside_complete
                    >= thresholds["minimum_container_inside_complete"],
                    "container_outside_foliage": outside_foliage
                    >= thresholds["minimum_container_outside_foliage"],
                    "contaminant_overlap": contamination
                    <= thresholds["maximum_contaminant_overlap"],
                    "horizontal_adjacency": horizontal_offset
                    <= thresholds["maximum_horizontal_center_offset_foliage_widths"],
                    "container_centroid_below_foliage": vertical_offset > 0,
                    "vertical_adjacency": top_gap
                    <= thresholds["maximum_container_top_gap_foliage_heights"],
                }
                part_records.append(
                    {
                        "candidate": _mask_record(part),
                        "association": {
                            "method": (
                                "one_to_one_assignment_by_complete-mask containment "
                                "and horizontal adjacency"
                            ),
                            "container_inside_complete_ratio": inside_complete,
                            "container_outside_foliage_ratio": outside_foliage,
                            "contaminant_overlap_ratio": contamination,
                            "horizontal_center_offset_foliage_widths": horizontal_offset,
                            "vertical_center_offset_foliage_heights": vertical_offset,
                            "container_top_gap_foliage_heights": top_gap,
                        },
                        "gates": gates,
                        "passed": all(gates.values()),
                    }
                )
            passed_parts = [record for record in part_records if record["passed"]]
            require(passed_parts, f"no container-part prompt passed for {object_id}/{frame_id}")
            selected_part = max(
                passed_parts,
                key=lambda record: (
                    record["association"]["container_inside_complete_ratio"],
                    -record["association"]["contaminant_overlap_ratio"],
                    record["candidate"]["sam3_score"],
                    record["candidate"]["label"],
                ),
            )
            selected_part_candidate = next(
                part_assignments[label][object_id]
                for label in parsed["part_labels"]
                if label == selected_part["candidate"]["label"]
            )
            selected_part_inside = selected_part["association"]["container_inside_complete_ratio"]
            complete_gates = {
                "anchor_hit_ratio": selected_hit >= thresholds["minimum_complete_anchor_hit_ratio"],
                "anchor_hit_margin": hit_margin >= thresholds["minimum_complete_anchor_hit_margin"],
                "foliage_coverage": foliage_coverage >= thresholds["minimum_foliage_coverage"],
                "selected_container_coverage": selected_part_inside
                >= thresholds["minimum_container_inside_complete"],
                "contaminant_overlap": complete_contamination
                <= thresholds["maximum_contaminant_overlap"],
                "bottom_extension": complete_bottom_extension
                >= thresholds["minimum_complete_bottom_extension_foliage_heights"],
            }
            require(
                all(complete_gates.values()),
                f"complete potted-plant gates failed for {object_id}/{frame_id}: {complete_gates}",
            )
            complete_output = _copy_exact(
                complete_candidate.path,
                output_directory / "masks" / object_id / f"{frame_id}.png",
            )
            part_output = _copy_exact(
                selected_part_candidate.path,
                output_directory / "parts" / object_id / f"{frame_id}.pot.png",
            )
            source_path = Path(str(view["source_rgb"]["path"])).resolve()
            source_sha, source_size = sha256_file(source_path)
            foliage_path = Path(str(view["mask"]["path"])).resolve()
            foliage_sha, foliage_size = sha256_file(foliage_path)
            title = (
                f"{frame_id} {selected_part_candidate.label} "
                f"anchor={selected_hit:.3f} pot-in={selected_part_inside:.3f} "
                f"neighbor={selected_part['association']['contaminant_overlap_ratio']:.4f}"
            )
            tile = _render_tile(
                source=source,
                foliage=foliage,
                container=selected_part_candidate.mask,
                complete=complete,
                projection=projections[object_id]["bbox_xyxy"],
                title=title,
                panel_width=parsed["review"]["panel_width"],
                panel_height=parsed["review"]["panel_height"],
                padding_ratio=parsed["review"]["crop_padding_ratio"],
            )
            object_reports[object_id]["tiles"].append(tile)
            object_reports[object_id]["frames"].append(
                {
                    "frame_id": frame_id,
                    "source_rgb": {
                        "path": str(source_path),
                        "sha256": source_sha,
                        "size_bytes": source_size,
                        "dimensions_wh": [source.width, source.height],
                    },
                    "anchor_projection": {
                        "method": "exact_source_3d_anchor_projection",
                        "point_count_inside_image": projections[object_id]["point_count"],
                        "bbox_xyxy": projections[object_id]["bbox_xyxy"],
                    },
                    "original_anchor_associated_foliage_mask": {
                        "path": str(foliage_path),
                        "sha256": foliage_sha,
                        "size_bytes": foliage_size,
                        "pixel_count": foliage_pixels,
                        "bbox_xyxy_exclusive": foliage_box,
                    },
                    "complete_instance_source_mask": _mask_record(complete_candidate),
                    "complete_instance_output_mask": {
                        "path": _relative(complete_output["path"], relative_to=manifest_dir),
                        "sha256": complete_output["sha256"],
                        "size_bytes": complete_output["size_bytes"],
                        "byte_identical_to_source_mask": True,
                    },
                    "selected_container_source_mask": selected_part["candidate"],
                    "selected_container_output_mask": {
                        "path": _relative(part_output["path"], relative_to=manifest_dir),
                        "sha256": part_output["sha256"],
                        "size_bytes": part_output["size_bytes"],
                        "byte_identical_to_source_mask": True,
                    },
                    "all_container_prompt_associations": part_records,
                    "association": {
                        "complete_instance_method": (
                            "one_to_one_maximum exact 3D anchor projected-point hit assignment"
                        ),
                        "selected_anchor_hit_ratio": selected_hit,
                        "runner_up_anchor_hit_ratio": runner_up_hit,
                        "anchor_hit_margin": hit_margin,
                        "original_foliage_coverage_ratio": foliage_coverage,
                        "complete_contaminant_overlap_ratio": complete_contamination,
                        "complete_bottom_extension_foliage_heights": complete_bottom_extension,
                        "selected_container_method": (
                            "best independently prompted real mask passing spatial "
                            "and contamination gates"
                        ),
                    },
                    "contaminant_masks": contaminant_records,
                    "gates": complete_gates,
                    "status": "passed",
                }
            )
    objects_payload = []
    for object_id in object_ids:
        review_path = (
            output_directory / "review" / object_id / "all-frames-targeted-pot-contact-sheet.png"
        )
        contact = _contact_sheet(
            object_reports[object_id].pop("tiles"),
            columns=parsed["review"]["columns"],
            output_path=review_path,
        )
        contact["path"] = _relative(contact["path"], relative_to=manifest_dir)
        object_reports[object_id]["contact_sheet"] = contact
        object_reports[object_id]["frame_count"] = len(object_reports[object_id]["frames"])
        object_reports[object_id]["status"] = "passed"
        object_reports[object_id]["anchor"] = anchor_records[object_id]
        objects_payload.append(object_reports[object_id])
    manifest = {
        "schema_version": 1,
        "kind": MANIFEST_KIND,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "passed",
        "consumable_for_object_conditioning": True,
        "claim_scope": (
            "Hash-bound real SAM3 text-prompt modal masks for visible potted plants and their "
            "container parts. No amodal completion, morphology, bbox expansion, inpainting, "
            "or support-surface relabeling is claimed."
        ),
        "config": {
            "path": str(parsed["path"]),
            "sha256": parsed["sha256"],
            "size_bytes": parsed["size_bytes"],
        },
        "inputs": {
            "source_audit": {
                "path": str(source_audit_path),
                "sha256": source_audit_sha,
                "size_bytes": source_audit_size,
            },
            "cameras": {
                "path": str(cameras_path),
                "sha256": cameras_sha,
                "size_bytes": cameras_size,
            },
            "original_mask_index": {
                "path": str(original_index_path),
                "sha256": original_index_sha,
                "size_bytes": original_index_size,
            },
            "targeted_mask_index": {
                "path": str(targeted_index_path),
                "sha256": targeted_index_sha,
                "size_bytes": targeted_index_size,
            },
        },
        "inference": parsed["inference"],
        "prompt_contract": {
            "complete_instance": parsed["complete_label"],
            "container_parts": parsed["part_labels"],
            "contaminant_labels": parsed["contaminant_labels"],
        },
        "thresholds": thresholds,
        "frame_ids": frame_ids,
        "objects": objects_payload,
        "limitations": [
            "Masks are modal source-view evidence and do not reveal occluded backs or bottoms.",
            "The camera arc is the existing Bedroom4 000048-000072 source sequence.",
            (
                "The complete mask is copied byte-for-byte from the potted plant SAM3 prompt; "
                "the independent pot prompt is validation evidence, not a synthetic union."
            ),
        ],
    }
    atomic_write_json(manifest_path, manifest)
    manifest_sha, manifest_size = sha256_file(manifest_path)
    view_manifests = []
    for object_payload in objects_payload:
        object_id = object_payload["object_id"]
        selected_frames = (
            audit_objects[object_id].get("frame_selection", {}).get("selected_frame_ids")
        )
        require(
            isinstance(selected_frames, list) and selected_frames,
            f"{object_id} source audit has no selected frames",
        )
        frames_by_id = {frame["frame_id"]: frame for frame in object_payload["frames"]}
        require(
            set(selected_frames).issubset(frames_by_id),
            f"{object_id} selected frames are absent from targeted outputs",
        )
        view_path = output_directory / "views" / f"{object_id}.json"
        view_payload = {
            "schema_version": 1,
            "kind": "video2world.completed_object_source_view_manifest",
            "object_id": object_id,
            "evidence_scope": "targeted_sam3_complete_potted_plant_modal_masks",
            "source_audit": {
                "path": _relative(manifest_path, relative_to=view_path.parent),
                "sha256": manifest_sha,
            },
            "views": [
                {
                    "frame_id": frame_id,
                    "is_source_view": True,
                    "observed_mask": {
                        "path": _relative(
                            (
                                manifest_dir
                                / frames_by_id[frame_id]["complete_instance_output_mask"]["path"]
                            ).resolve(),
                            relative_to=view_path.parent,
                        ),
                        "sha256": frames_by_id[frame_id]["complete_instance_output_mask"]["sha256"],
                    },
                    "occluder_masks": [],
                }
                for frame_id in selected_frames
            ],
        }
        atomic_write_json(view_path, view_payload)
        view_sha, view_size = sha256_file(view_path)
        view_manifests.append(
            {
                "object_id": object_id,
                "path": str(view_path),
                "sha256": view_sha,
                "size_bytes": view_size,
                "selected_frame_ids": selected_frames,
            }
        )
    return {
        "status": "passed",
        "manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha,
            "size_bytes": manifest_size,
        },
        "object_count": len(objects_payload),
        "frame_count": len(frame_ids),
        "view_manifests": view_manifests,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Associate targeted potted-plant SAM3 masks with exact 3D anchors."
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(materialize(args.config), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
