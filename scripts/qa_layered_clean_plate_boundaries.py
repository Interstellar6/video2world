#!/usr/bin/env python3
"""Fail-closed boundary QA for scene-independent layered clean-plate frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

MANIFEST_KIND = "video2world.layered_clean_plate_boundary_qa_manifest"
REPORT_KIND = "video2world.layered_clean_plate_boundary_qa_report"
THRESHOLD_KEYS = (
    "minimum_mask_iou",
    "minimum_mask_precision",
    "minimum_mask_recall",
    "maximum_boundary_distance_p95_px",
    "maximum_seam_color_p95_abs_rgb_delta",
    "maximum_seam_gradient_p95_abs_rgb_delta",
    "minimum_core_to_local_ring_laplacian_energy_ratio",
)
TEXTURE_SAMPLING_KEYS = (
    "core_erosion_px",
    "local_ring_inner_px",
    "local_ring_outer_px",
    "minimum_region_pixels",
    "winsor_quantile",
)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
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


def require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def require_finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def resolve_asset(
    value: Any,
    *,
    manifest_path: Path,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    asset = require_dict(value, label)
    declared_path = asset.get("path")
    if not isinstance(declared_path, str) or not declared_path:
        raise ValueError(f"{label}.path must be a non-empty string")
    path = Path(declared_path).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_sha256 = require_sha256(asset.get("sha256"), f"{label}.sha256")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"{label}.sha256 does not match the input file")
    return path, {"path": str(path), "sha256": actual_sha256, "bytes": path.stat().st_size}


def validate_thresholds(value: Any) -> dict[str, float]:
    raw = require_dict(value, "thresholds")
    missing = [key for key in THRESHOLD_KEYS if key not in raw]
    if missing:
        raise ValueError(f"thresholds is missing required keys: {', '.join(missing)}")
    unknown = sorted(set(raw) - set(THRESHOLD_KEYS))
    if unknown:
        raise ValueError(f"thresholds has unknown keys: {', '.join(unknown)}")
    result = {key: require_finite_number(raw[key], f"thresholds.{key}") for key in raw}
    for key in (
        "minimum_mask_iou",
        "minimum_mask_precision",
        "minimum_mask_recall",
        "minimum_core_to_local_ring_laplacian_energy_ratio",
    ):
        if result[key] < 0.0:
            raise ValueError(f"thresholds.{key} must be non-negative")
    for key in ("minimum_mask_iou", "minimum_mask_precision", "minimum_mask_recall"):
        if result[key] > 1.0:
            raise ValueError(f"thresholds.{key} must be in [0, 1]")
    for key in (
        "maximum_boundary_distance_p95_px",
        "maximum_seam_color_p95_abs_rgb_delta",
        "maximum_seam_gradient_p95_abs_rgb_delta",
    ):
        if result[key] < 0.0:
            raise ValueError(f"thresholds.{key} must be non-negative")
    return result


def require_non_negative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def validate_texture_sampling(value: Any) -> dict[str, float | int]:
    raw = require_dict(value, "texture_sampling")
    missing = [key for key in TEXTURE_SAMPLING_KEYS if key not in raw]
    if missing:
        raise ValueError(f"texture_sampling is missing required keys: {', '.join(missing)}")
    unknown = sorted(set(raw) - set(TEXTURE_SAMPLING_KEYS))
    if unknown:
        raise ValueError(f"texture_sampling has unknown keys: {', '.join(unknown)}")
    core_erosion = require_non_negative_int(
        raw["core_erosion_px"], "texture_sampling.core_erosion_px"
    )
    ring_inner = require_non_negative_int(
        raw["local_ring_inner_px"], "texture_sampling.local_ring_inner_px"
    )
    ring_outer = require_non_negative_int(
        raw["local_ring_outer_px"], "texture_sampling.local_ring_outer_px"
    )
    minimum_pixels = require_non_negative_int(
        raw["minimum_region_pixels"], "texture_sampling.minimum_region_pixels"
    )
    winsor_quantile = require_finite_number(
        raw["winsor_quantile"], "texture_sampling.winsor_quantile"
    )
    if core_erosion < 1:
        raise ValueError("texture_sampling.core_erosion_px must be at least one")
    if ring_inner < 1:
        raise ValueError("texture_sampling.local_ring_inner_px must be at least one")
    if ring_outer <= ring_inner:
        raise ValueError("texture_sampling.local_ring_outer_px must exceed local_ring_inner_px")
    if minimum_pixels < 1:
        raise ValueError("texture_sampling.minimum_region_pixels must be at least one")
    if not 0.5 <= winsor_quantile < 1.0:
        raise ValueError("texture_sampling.winsor_quantile must be in [0.5, 1.0)")
    return {
        "core_erosion_px": core_erosion,
        "local_ring_inner_px": ring_inner,
        "local_ring_outer_px": ring_outer,
        "minimum_region_pixels": minimum_pixels,
        "winsor_quantile": winsor_quantile,
    }


def load_mask(
    value: Any,
    *,
    manifest_path: Path,
    label: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    path, asset_record = resolve_asset(value, manifest_path=manifest_path, label=label)
    asset = require_dict(value, label)
    channel = asset.get("channel")
    if channel not in {"luma", "alpha"}:
        raise ValueError(f"{label}.channel must be 'luma' or 'alpha'")
    threshold = asset.get("threshold")
    if not isinstance(threshold, int) or isinstance(threshold, bool) or not 1 <= threshold <= 255:
        raise ValueError(f"{label}.threshold must be an integer in [1, 255]")
    with Image.open(path) as image:
        if channel == "alpha":
            if "A" not in image.getbands():
                raise ValueError(f"{label} requests alpha but the image has no alpha channel")
            values = np.asarray(image.getchannel("A"), dtype=np.uint8)
        else:
            values = np.asarray(image.convert("L"), dtype=np.uint8)
    mask = values >= threshold
    pixels = int(mask.sum())
    if pixels == 0:
        raise ValueError(f"{label} is empty")
    if pixels == mask.size:
        raise ValueError(f"{label} covers the full frame and has no measurable boundary")
    asset_record.update(
        {
            "channel": channel,
            "threshold": threshold,
            "dimensions": [int(mask.shape[1]), int(mask.shape[0])],
            "area_pixels": pixels,
        }
    )
    return mask, asset_record


def load_rgb(
    value: Any,
    *,
    manifest_path: Path,
    label: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    path, asset_record = resolve_asset(value, manifest_path=manifest_path, label=label)
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    asset_record["dimensions"] = [int(rgb.shape[1]), int(rgb.shape[0])]
    return rgb, asset_record


def inner_boundary(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(
        mask,
        structure=np.ones((3, 3), dtype=bool),
        border_value=0,
    )


def mask_alignment_metrics(removal: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    intersection = int((removal & target).sum())
    removal_pixels = int(removal.sum())
    target_pixels = int(target.sum())
    union = removal_pixels + target_pixels - intersection
    removal_boundary = inner_boundary(removal)
    target_boundary = inner_boundary(target)
    if not removal_boundary.any() or not target_boundary.any():
        raise ValueError("removal mask and target matte must both have non-empty boundaries")
    distance_to_target = ndimage.distance_transform_edt(~target_boundary)[removal_boundary]
    distance_to_removal = ndimage.distance_transform_edt(~removal_boundary)[target_boundary]
    symmetric_distance = np.concatenate((distance_to_target, distance_to_removal))
    return {
        "removal_mask_pixels": removal_pixels,
        "target_matte_pixels": target_pixels,
        "intersection_pixels": intersection,
        "union_pixels": union,
        "mask_iou": intersection / union,
        "mask_precision": intersection / removal_pixels,
        "mask_recall": intersection / target_pixels,
        "removal_only_pixels": removal_pixels - intersection,
        "target_only_pixels": target_pixels - intersection,
        "boundary_sample_count": int(symmetric_distance.size),
        "boundary_distance_median_px": float(np.quantile(symmetric_distance, 0.5)),
        "boundary_distance_p95_px": float(np.quantile(symmetric_distance, 0.95)),
        "boundary_distance_max_px": float(symmetric_distance.max()),
    }


def seam_continuity_metrics(
    composite: np.ndarray,
    removal: np.ndarray,
) -> dict[str, float | int]:
    color_delta: list[float] = []
    gradient_delta: list[float] = []
    height, width = removal.shape

    def add_edge(
        outside: np.ndarray,
        inside: np.ndarray,
        outside_previous: np.ndarray | None,
        inside_next: np.ndarray | None,
    ) -> None:
        cross_gradient = inside.astype(np.float64) - outside
        color_delta.append(float(np.abs(cross_gradient).mean()))
        if outside_previous is not None:
            outside_gradient = outside.astype(np.float64) - outside_previous
            gradient_delta.append(float(np.abs(cross_gradient - outside_gradient).mean()))
        if inside_next is not None:
            inside_gradient = inside_next.astype(np.float64) - inside
            gradient_delta.append(float(np.abs(cross_gradient - inside_gradient).mean()))

    for row in range(height):
        for column in np.flatnonzero(removal[row, 1:] != removal[row, :-1]):
            if removal[row, column + 1]:
                add_edge(
                    composite[row, column],
                    composite[row, column + 1],
                    composite[row, column - 1]
                    if column > 0 and not removal[row, column - 1]
                    else None,
                    composite[row, column + 2]
                    if column + 2 < width and removal[row, column + 2]
                    else None,
                )
            else:
                add_edge(
                    composite[row, column + 1],
                    composite[row, column],
                    composite[row, column + 2]
                    if column + 2 < width and not removal[row, column + 2]
                    else None,
                    composite[row, column - 1] if column > 0 and removal[row, column - 1] else None,
                )
    for row in range(height - 1):
        for column in np.flatnonzero(removal[row + 1] != removal[row]):
            if removal[row + 1, column]:
                add_edge(
                    composite[row, column],
                    composite[row + 1, column],
                    composite[row - 1, column]
                    if row > 0 and not removal[row - 1, column]
                    else None,
                    composite[row + 2, column]
                    if row + 2 < height and removal[row + 2, column]
                    else None,
                )
            else:
                add_edge(
                    composite[row + 1, column],
                    composite[row, column],
                    composite[row + 2, column]
                    if row + 2 < height and not removal[row + 2, column]
                    else None,
                    composite[row - 1, column] if row > 0 and removal[row - 1, column] else None,
                )
    if not color_delta:
        raise ValueError("removal mask has no measurable composite boundary edges")
    if not gradient_delta:
        raise ValueError("removal mask has no measurable boundary-normal gradient pairs")
    return {
        "boundary_edge_count": len(color_delta),
        "seam_color_mean_abs_rgb_delta": float(np.mean(color_delta)),
        "seam_color_p95_abs_rgb_delta": float(np.quantile(color_delta, 0.95)),
        "boundary_normal_gradient_pair_count": len(gradient_delta),
        "seam_gradient_mean_abs_rgb_delta": float(np.mean(gradient_delta)),
        "seam_gradient_p95_abs_rgb_delta": float(np.quantile(gradient_delta, 0.95)),
    }


def local_texture_metrics(
    composite: np.ndarray,
    removal_core: np.ndarray,
    editable: np.ndarray,
    config: dict[str, float | int],
) -> dict[str, float | int]:
    structure = np.ones((3, 3), dtype=bool)
    core = ndimage.binary_erosion(
        removal_core,
        structure=structure,
        iterations=int(config["core_erosion_px"]),
        border_value=0,
    )
    ring_inner = ndimage.binary_dilation(
        editable,
        structure=structure,
        iterations=int(config["local_ring_inner_px"]),
        border_value=0,
    )
    ring_outer = ndimage.binary_dilation(
        editable,
        structure=structure,
        iterations=int(config["local_ring_outer_px"]),
        border_value=0,
    )
    local_ring = ring_outer & ~ring_inner
    core_pixels = int(core.sum())
    ring_pixels = int(local_ring.sum())
    minimum_pixels = int(config["minimum_region_pixels"])
    if core_pixels < minimum_pixels:
        raise ValueError(
            f"eroded removal core has {core_pixels} pixels; at least {minimum_pixels} required"
        )
    if ring_pixels < minimum_pixels:
        raise ValueError(
            f"outside local ring has {ring_pixels} pixels; at least {minimum_pixels} required"
        )

    rgb = composite.astype(np.float64)
    luma = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    laplacian = ndimage.laplace(luma, mode="reflect")
    energy = np.square(laplacian)
    core_energy = energy[core]
    ring_energy = energy[local_ring]
    combined = np.concatenate((core_energy, ring_energy))
    winsor_quantile = float(config["winsor_quantile"])
    energy_cap = float(np.quantile(combined, winsor_quantile))
    if energy_cap <= np.finfo(np.float64).eps:
        raise ValueError("core and local ring contain no measurable Laplacian texture energy")
    robust_core_energy = float(np.minimum(core_energy, energy_cap).mean())
    robust_ring_energy = float(np.minimum(ring_energy, energy_cap).mean())
    if robust_ring_energy <= np.finfo(np.float64).eps:
        raise ValueError("outside local ring contains no measurable Laplacian texture energy")
    return {
        "core_pixels": core_pixels,
        "local_ring_pixels": ring_pixels,
        "core_laplacian_energy_raw_mean": float(core_energy.mean()),
        "local_ring_laplacian_energy_raw_mean": float(ring_energy.mean()),
        "winsor_quantile": winsor_quantile,
        "winsor_energy_cap": energy_cap,
        "core_laplacian_energy_robust_mean": robust_core_energy,
        "local_ring_laplacian_energy_robust_mean": robust_ring_energy,
        "core_to_local_ring_laplacian_energy_ratio": (robust_core_energy / robust_ring_energy),
    }


def frame_gates(
    alignment: dict[str, float | int],
    seam: dict[str, float | int],
    texture: dict[str, float | int],
    thresholds: dict[str, float],
) -> dict[str, dict[str, float | bool]]:
    definitions = (
        ("mask_iou_gte", "mask_iou", "minimum_mask_iou", True),
        ("mask_precision_gte", "mask_precision", "minimum_mask_precision", True),
        ("mask_recall_gte", "mask_recall", "minimum_mask_recall", True),
        (
            "boundary_distance_p95_lte",
            "boundary_distance_p95_px",
            "maximum_boundary_distance_p95_px",
            False,
        ),
        (
            "seam_color_p95_lte",
            "seam_color_p95_abs_rgb_delta",
            "maximum_seam_color_p95_abs_rgb_delta",
            False,
        ),
        (
            "seam_gradient_p95_lte",
            "seam_gradient_p95_abs_rgb_delta",
            "maximum_seam_gradient_p95_abs_rgb_delta",
            False,
        ),
        (
            "core_to_local_ring_laplacian_energy_ratio_gte",
            "core_to_local_ring_laplacian_energy_ratio",
            "minimum_core_to_local_ring_laplacian_energy_ratio",
            True,
        ),
    )
    gates: dict[str, dict[str, float | bool]] = {}
    for gate_name, metric_name, threshold_name, is_minimum in definitions:
        if metric_name in alignment:
            source = alignment
        elif metric_name in seam:
            source = seam
        else:
            source = texture
        observed = float(source[metric_name])
        threshold = thresholds[threshold_name]
        gates[gate_name] = {
            "observed": observed,
            "threshold": threshold,
            "passed": observed >= threshold if is_minimum else observed <= threshold,
        }
    return gates


def make_overlay(
    base: np.ndarray,
    removal_core: np.ndarray,
    target: np.ndarray,
    editable: np.ndarray,
) -> Image.Image:
    result = base.astype(np.float32).copy()
    selections = (
        (editable & ~removal_core, np.asarray([255, 185, 35], dtype=np.float32)),
        (removal_core & target, np.asarray([40, 220, 100], dtype=np.float32)),
        (removal_core & ~target, np.asarray([255, 55, 55], dtype=np.float32)),
        (target & ~removal_core, np.asarray([0, 215, 255], dtype=np.float32)),
    )
    for selection, color in selections:
        result[selection] = result[selection] * 0.35 + color * 0.65
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def labeled_tile(image: Image.Image, label: str, size: tuple[int, int]) -> Image.Image:
    label_height = 22
    tile = Image.new("RGB", (size[0], size[1] + label_height), "#111111")
    contained = image.convert("RGB")
    contained.thumbnail(size, Image.Resampling.LANCZOS)
    x = (size[0] - contained.width) // 2
    y = label_height + (size[1] - contained.height) // 2
    tile.paste(contained, (x, y))
    ImageDraw.Draw(tile).text((6, 6), label, fill="#f0f0f0")
    return tile


def sample_indexes(frame_count: int, samples: int) -> set[int]:
    if samples <= 0:
        return set()
    count = min(frame_count, samples)
    return {round(value) for value in np.linspace(0, frame_count - 1, count)}


def make_contact_sheet(
    rows: list[tuple[str, Image.Image, Image.Image, Image.Image]],
    path: Path,
) -> None:
    if not rows:
        raise ValueError("contact sheet requested with zero samples")
    tile_size = (420, 236)
    tiled_rows = []
    for frame_id, base, composite, overlay in rows:
        tiled_rows.append(
            (
                labeled_tile(base, f"{frame_id}: source/base", tile_size),
                labeled_tile(composite, f"{frame_id}: composite", tile_size),
                labeled_tile(
                    overlay,
                    (
                        f"{frame_id}: green core-target / red core-only / "
                        "cyan target-only / yellow collar"
                    ),
                    tile_size,
                ),
            )
        )
    tile_width = tiled_rows[0][0].width
    tile_height = tiled_rows[0][0].height
    sheet = Image.new("RGB", (tile_width * 3, tile_height * len(tiled_rows)), "#111111")
    for row_index, row in enumerate(tiled_rows):
        for column_index, tile in enumerate(row):
            sheet.paste(tile, (column_index * tile_width, row_index * tile_height))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def aggregate_gates(
    records: list[dict[str, Any]],
    thresholds: dict[str, float],
) -> dict[str, dict[str, float | bool]]:
    definitions = (
        ("mask_iou_gte", "mask_iou", "minimum_mask_iou", min, True),
        ("mask_precision_gte", "mask_precision", "minimum_mask_precision", min, True),
        ("mask_recall_gte", "mask_recall", "minimum_mask_recall", min, True),
        (
            "boundary_distance_p95_lte",
            "boundary_distance_p95_px",
            "maximum_boundary_distance_p95_px",
            max,
            False,
        ),
        (
            "seam_color_p95_lte",
            "seam_color_p95_abs_rgb_delta",
            "maximum_seam_color_p95_abs_rgb_delta",
            max,
            False,
        ),
        (
            "seam_gradient_p95_lte",
            "seam_gradient_p95_abs_rgb_delta",
            "maximum_seam_gradient_p95_abs_rgb_delta",
            max,
            False,
        ),
        (
            "core_to_local_ring_laplacian_energy_ratio_gte",
            "core_to_local_ring_laplacian_energy_ratio",
            "minimum_core_to_local_ring_laplacian_energy_ratio",
            min,
            True,
        ),
    )
    gates: dict[str, dict[str, float | bool]] = {}
    for gate_name, metric_name, threshold_name, reducer, is_minimum in definitions:
        values = []
        for record in records:
            if metric_name in record["alignment"]:
                source = record["alignment"]
            elif metric_name in record["seam"]:
                source = record["seam"]
            else:
                source = record["texture"]
            values.append(float(source[metric_name]))
        observed = float(reducer(values))
        threshold = thresholds[threshold_name]
        gates[gate_name] = {
            "observed_worst": observed,
            "threshold": threshold,
            "passed": observed >= threshold if is_minimum else observed <= threshold,
        }
    return gates


def validate(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.input_manifest.expanduser().resolve()
    manifest = require_dict(read_json(manifest_path), "manifest")
    if manifest.get("schema_version") != 1:
        raise ValueError("manifest.schema_version must equal 1")
    if manifest.get("kind") != MANIFEST_KIND:
        raise ValueError(f"manifest.kind must equal {MANIFEST_KIND!r}")
    thresholds = validate_thresholds(manifest.get("thresholds"))
    texture_sampling = validate_texture_sampling(manifest.get("texture_sampling"))
    raw_records = require_list(manifest.get("frame_records"), "frame_records")
    if not raw_records:
        raise ValueError("frame_records must not be empty")
    if manifest.get("frame_count") != len(raw_records):
        raise ValueError("manifest.frame_count does not match frame_records")
    samples = sample_indexes(len(raw_records), args.contact_sheet_samples)
    records: list[dict[str, Any]] = []
    contact_rows: list[tuple[str, Image.Image, Image.Image, Image.Image]] = []
    frame_ids: set[str] = set()
    sequence_indexes: list[int] = []

    for index, raw_record in enumerate(raw_records):
        record = require_dict(raw_record, f"frame_records[{index}]")
        sequence_index = record.get("sequence_index")
        if not isinstance(sequence_index, int) or isinstance(sequence_index, bool):
            raise ValueError(f"frame_records[{index}].sequence_index must be an integer")
        sequence_indexes.append(sequence_index)
        frame_id = record.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id:
            raise ValueError(f"frame_records[{index}].frame_id must be a non-empty string")
        if frame_id in frame_ids:
            raise ValueError(f"duplicate frame_id: {frame_id}")
        frame_ids.add(frame_id)
        prefix = f"frame_records[{index}]"
        removal_core, removal_core_asset = load_mask(
            record.get("removal_core_mask"),
            manifest_path=manifest_path,
            label=f"{prefix}.removal_core_mask",
        )
        target, target_asset = load_mask(
            record.get("target_matte"), manifest_path=manifest_path, label=f"{prefix}.target_matte"
        )
        editable, editable_asset = load_mask(
            record.get("editable_mask"),
            manifest_path=manifest_path,
            label=f"{prefix}.editable_mask",
        )
        composite, composite_asset = load_rgb(
            record.get("composite_rgb"),
            manifest_path=manifest_path,
            label=f"{prefix}.composite_rgb",
        )
        if (
            removal_core.shape != target.shape
            or removal_core.shape != editable.shape
            or removal_core.shape != composite.shape[:2]
        ):
            raise ValueError(f"{prefix} asset dimensions do not match")
        if np.any(removal_core & ~editable):
            raise ValueError(f"{prefix}.editable_mask must contain removal_core_mask")
        source_asset = None
        source = composite
        if record.get("source_rgb") is not None:
            source, source_asset = load_rgb(
                record.get("source_rgb"),
                manifest_path=manifest_path,
                label=f"{prefix}.source_rgb",
            )
            if source.shape != composite.shape:
                raise ValueError(f"{prefix}.source_rgb dimensions do not match composite_rgb")
        alignment = mask_alignment_metrics(removal_core, target)
        seam = seam_continuity_metrics(composite, editable)
        texture = local_texture_metrics(
            composite,
            removal_core,
            editable,
            texture_sampling,
        )
        gates = frame_gates(alignment, seam, texture, thresholds)
        overlay = make_overlay(source, removal_core, target, editable)
        output_record: dict[str, Any] = {
            "sequence_index": sequence_index,
            "frame_id": frame_id,
            "inputs": {
                "removal_core_mask": removal_core_asset,
                "target_matte": target_asset,
                "editable_mask": editable_asset,
                "composite_rgb": composite_asset,
                "source_rgb": source_asset,
            },
            "alignment": alignment,
            "seam": seam,
            "texture": texture,
            "gates": gates,
            "passed": all(bool(gate["passed"]) for gate in gates.values()),
        }
        if args.overlay_dir is not None:
            overlay_path = args.overlay_dir.expanduser().resolve() / f"{sequence_index:04d}.png"
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            overlay.save(overlay_path)
            output_record["overlay"] = {
                "path": str(overlay_path),
                "sha256": sha256_file(overlay_path),
                "bytes": overlay_path.stat().st_size,
            }
        records.append(output_record)
        if index in samples:
            contact_rows.append(
                (
                    frame_id,
                    Image.fromarray(source),
                    Image.fromarray(composite),
                    overlay,
                )
            )

    if sequence_indexes != list(range(len(raw_records))):
        raise ValueError(
            "frame_records sequence_index values must be contiguous and ordered from 0"
        )
    gates = aggregate_gates(records, thresholds)
    passed = all(bool(gate["passed"]) for gate in gates.values())
    outputs: dict[str, Any] = {}
    if args.contact_sheet is not None:
        contact_sheet_path = args.contact_sheet.expanduser().resolve()
        make_contact_sheet(contact_rows, contact_sheet_path)
        outputs["contact_sheet"] = {
            "path": str(contact_sheet_path),
            "sha256": sha256_file(contact_sheet_path),
            "bytes": contact_sheet_path.stat().st_size,
            "sample_count": len(contact_rows),
        }
    report = {
        "schema_version": 1,
        "kind": REPORT_KIND,
        "status": "technical_passed" if passed else "technical_failed",
        "boundary_texture_gate_passed": passed,
        "promotion_scope": "boundary_and_local_texture_only",
        "promotion_approved": False,
        "created_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "input_manifest": str(manifest_path),
        "input_manifest_sha256": sha256_file(manifest_path),
        "frame_count": len(records),
        "thresholds": thresholds,
        "texture_sampling": texture_sampling,
        "metric_contract": {
            "mask_precision": "intersection / removal-core-mask pixels",
            "mask_recall": "intersection / target-matte pixels",
            "boundary_distance": (
                "symmetric distance between inner 8-connected removal-core and "
                "target-matte boundaries"
            ),
            "seam_color": "mean absolute RGB delta across editable-mask outer boundary edges",
            "seam_gradient": (
                "mean absolute RGB difference between each cross-boundary gradient and "
                "available one-pixel normal gradients on both sides"
            ),
            "local_texture": (
                "winsorized mean squared luminance Laplacian in an eroded removal core "
                "divided by the same energy in a local ring outside the editable mask"
            ),
            "overlay_colors": {
                "green": "removal core and target matte overlap",
                "red": "removal core only",
                "cyan": "target matte only",
                "yellow": "editable soft collar outside the removal core",
            },
        },
        "gates": gates,
        "frame_records": records,
        "outputs": outputs,
    }
    write_json(args.output_report.expanduser().resolve(), report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    parser.add_argument("--overlay-dir", type=Path)
    parser.add_argument("--contact-sheet", type=Path)
    parser.add_argument("--contact-sheet-samples", type=int, default=12)
    args = parser.parse_args()
    if args.contact_sheet_samples < 1:
        parser.error("--contact-sheet-samples must be positive")
    return args


def main() -> None:
    report = validate(parse_args())
    raise SystemExit(0 if report["boundary_texture_gate_passed"] else 1)


if __name__ == "__main__":
    main()
