#!/usr/bin/env python3
"""Materialize hash-bound source RGBA conditions from a source-view audit.

This tool is intentionally non-generative. It copies RGB values only where the
selected SAM mask is foreground, makes every other pixel transparent, and then
creates an aspect-preserving Lanczos 1024-square condition. Views whose mask
touches the source-frame edge are rejected before ranking.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from video2world.hashing import atomic_write_json, sha256_file

CONFIG_KIND = "video2world.source_rgba_condition_config"
AUDIT_KIND = "video2world.direct_trellis2_source_view_audit"
REPORT_KIND = "video2world.source_rgba_conditions_report"
OBJECT_REPORT_KIND = "video2world.source_rgba_condition_report"


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


def verified_hash(path: Path, expected: Any, *, label: str) -> tuple[str, int]:
    require(path.is_file(), f"{label} is not a file: {path}")
    require(
        isinstance(expected, str) and len(expected) == 64,
        f"{label}.sha256 must be a 64-character digest",
    )
    observed, size = sha256_file(path)
    require(observed == expected, f"{label} SHA-256 mismatch: {observed} != {expected}")
    return observed, size


def mask_bbox(mask: np.ndarray) -> list[int]:
    rows, columns = np.nonzero(mask)
    require(len(columns) > 0, "selected SAM mask is empty")
    return [
        int(columns.min()),
        int(rows.min()),
        int(columns.max() + 1),
        int(rows.max() + 1),
    ]


def _integer(value: Any, *, label: str, minimum: int) -> int:
    require(isinstance(value, int) and not isinstance(value, bool), f"{label} must be an integer")
    require(value >= minimum, f"{label} must be >= {minimum}")
    return value


def _number(value: Any, *, label: str, minimum: float, maximum: float) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be numeric",
    )
    result = float(value)
    require(math.isfinite(result), f"{label} must be finite")
    require(minimum <= result <= maximum, f"{label} must be inside [{minimum}, {maximum}]")
    return result


def parse_config(config_path: Path) -> dict[str, Any]:
    config = read_json(config_path)
    require(config.get("kind") == CONFIG_KIND, f"config.kind must be {CONFIG_KIND!r}")
    base = config_path.parent

    audit_record = config.get("audit")
    require(isinstance(audit_record, dict), "config.audit must be an object")
    audit_path = resolve_path(audit_record.get("path"), relative_to=base, label="audit.path")
    audit_hash, audit_size = verified_hash(
        audit_path,
        audit_record.get("sha256"),
        label="audit",
    )

    object_ids = config.get("object_ids")
    require(isinstance(object_ids, list) and object_ids, "config.object_ids must be non-empty")
    require(
        all(isinstance(value, str) and value for value in object_ids),
        "config.object_ids entries must be non-empty strings",
    )
    require(len(set(object_ids)) == len(object_ids), "config.object_ids must be unique")

    selection = config.get("selection")
    require(isinstance(selection, dict), "config.selection must be an object")
    require(
        selection.get("ranking") == "foreground_pixel_count_desc_then_frame_id",
        "selection.ranking must be foreground_pixel_count_desc_then_frame_id",
    )
    minimum_edge_distance_px = _integer(
        selection.get("minimum_edge_distance_px"),
        label="selection.minimum_edge_distance_px",
        minimum=1,
    )

    crop = config.get("crop")
    require(isinstance(crop, dict), "config.crop must be an object")
    padding_ratio = _number(
        crop.get("padding_ratio"),
        label="crop.padding_ratio",
        minimum=0.10,
        maximum=0.15,
    )
    condition_size = _integer(
        crop.get("condition_size"),
        label="crop.condition_size",
        minimum=64,
    )
    require(
        crop.get("background") == "transparent",
        "crop.background must be transparent",
    )
    emit_white_background = crop.get("emit_white_background", True)
    require(isinstance(emit_white_background, bool), "crop.emit_white_background must be boolean")

    output_root = resolve_path(
        config.get("output_directory"),
        relative_to=base,
        label="output_directory",
    )
    return {
        "raw": config,
        "path": config_path,
        "sha256": sha256_file(config_path)[0],
        "audit_path": audit_path,
        "audit_sha256": audit_hash,
        "audit_size_bytes": audit_size,
        "object_ids": object_ids,
        "minimum_edge_distance_px": minimum_edge_distance_px,
        "padding_ratio": padding_ratio,
        "condition_size": condition_size,
        "emit_white_background": emit_white_background,
        "output_root": output_root,
    }


def load_view(view: dict[str, Any], *, minimum_edge_distance_px: int) -> dict[str, Any]:
    frame_id = view.get("frame_id")
    require(isinstance(frame_id, str) and frame_id, "audit view.frame_id is invalid")
    association = view.get("association")
    require(isinstance(association, dict), f"{frame_id}: association must be an object")
    require(
        association.get("status") == "passed_unambiguous_for_selected_frame",
        f"{frame_id}: mask association is not unambiguously passed",
    )

    source_record = view.get("source_rgb")
    mask_record = view.get("mask")
    require(isinstance(source_record, dict), f"{frame_id}: source_rgb must be an object")
    require(isinstance(mask_record, dict), f"{frame_id}: mask must be an object")
    require(
        source_record.get("authoritative_receipt_hash_match") is True,
        f"{frame_id}: source RGB is not marked as authoritative receipt hash matched",
    )
    source_path = resolve_path(
        source_record.get("absolute_path"),
        relative_to=Path.cwd(),
        label=f"{frame_id}.source_rgb.absolute_path",
    )
    mask_path = resolve_path(
        mask_record.get("absolute_path"),
        relative_to=Path.cwd(),
        label=f"{frame_id}.mask.absolute_path",
    )
    source_hash, source_size = verified_hash(
        source_path,
        source_record.get("sha256"),
        label=f"{frame_id}.source_rgb",
    )
    mask_hash, mask_size = verified_hash(
        mask_path,
        mask_record.get("sha256"),
        label=f"{frame_id}.mask",
    )

    with Image.open(source_path) as image:
        source = np.asarray(image.convert("RGB"), dtype=np.uint8)
    with Image.open(mask_path) as image:
        mask = np.asarray(image.convert("L"), dtype=np.uint8) > 0
    height, width = source.shape[:2]
    require(mask.shape == (height, width), f"{frame_id}: source RGB and mask dimensions differ")
    declared_dimensions = source_record.get("dimensions_wh")
    require(
        declared_dimensions == [width, height],
        f"{frame_id}: source dimensions differ from audit: "
        f"{[width, height]} != {declared_dimensions}",
    )
    bbox = mask_bbox(mask)
    declared_bbox = mask_record.get("pixel_bbox_xyxy_exclusive")
    require(
        declared_bbox == bbox,
        f"{frame_id}: mask bbox differs from audit: {bbox} != {declared_bbox}",
    )
    pixel_count = int(np.count_nonzero(mask))
    require(
        mask_record.get("pixel_count") == pixel_count,
        f"{frame_id}: mask pixel count differs from audit: "
        f"{pixel_count} != {mask_record.get('pixel_count')}",
    )
    x0, y0, x1, y1 = bbox
    edge_distances = {
        "left": x0,
        "top": y0,
        "right": width - x1,
        "bottom": height - y1,
    }
    minimum_observed = min(edge_distances.values())
    eligible = minimum_observed >= minimum_edge_distance_px
    return {
        "frame_id": frame_id,
        "source_path": source_path,
        "source_repo_uri": source_record.get("repo_uri"),
        "source_upstream_path": source_record.get("authoritative_upstream_path"),
        "source_sha256": source_hash,
        "source_size_bytes": source_size,
        "mask_path": mask_path,
        "mask_file_name": mask_record.get("file_name"),
        "mask_sha256": mask_hash,
        "mask_size_bytes": mask_size,
        "source": source,
        "mask": mask,
        "dimensions_wh": [width, height],
        "bbox_xyxy_exclusive": bbox,
        "bbox_dimensions_wh": [x1 - x0, y1 - y0],
        "pixel_count": pixel_count,
        "edge_distances_px": edge_distances,
        "minimum_edge_distance_px": minimum_observed,
        "eligible_non_edge_view": eligible,
    }


def select_view(
    object_record: dict[str, Any],
    *,
    minimum_edge_distance_px: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    object_id = object_record.get("object_id")
    views = object_record.get("views")
    require(isinstance(views, list) and views, f"{object_id}: audit views must be non-empty")
    loaded = [load_view(view, minimum_edge_distance_px=minimum_edge_distance_px) for view in views]
    eligible = [view for view in loaded if view["eligible_non_edge_view"]]
    require(
        eligible,
        f"{object_id}: every audited SAM mask touches the source-frame edge "
        f"within {minimum_edge_distance_px}px; refusing to create a condition",
    )
    eligible.sort(key=lambda value: (-value["pixel_count"], value["frame_id"]))
    return eligible[0], loaded


def build_raw_crop(
    source: np.ndarray,
    mask: np.ndarray,
    bbox: list[int],
    *,
    padding_ratio: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    x0, y0, x1, y1 = bbox
    bbox_width = x1 - x0
    bbox_height = y1 - y0
    padding_x = math.ceil(bbox_width * padding_ratio)
    padding_y = math.ceil(bbox_height * padding_ratio)
    requested = [x0 - padding_x, y0 - padding_y, x1 + padding_x, y1 + padding_y]
    crop_width = bbox_width + 2 * padding_x
    crop_height = bbox_height + 2 * padding_y
    rgba = np.zeros((crop_height, crop_width, 4), dtype=np.uint8)

    source_height, source_width = mask.shape
    source_x0 = max(0, requested[0])
    source_y0 = max(0, requested[1])
    source_x1 = min(source_width, requested[2])
    source_y1 = min(source_height, requested[3])
    destination_x0 = source_x0 - requested[0]
    destination_y0 = source_y0 - requested[1]
    destination_x1 = destination_x0 + source_x1 - source_x0
    destination_y1 = destination_y0 + source_y1 - source_y0

    source_slice = source[source_y0:source_y1, source_x0:source_x1]
    mask_slice = mask[source_y0:source_y1, source_x0:source_x1]
    destination = rgba[destination_y0:destination_y1, destination_x0:destination_x1]
    destination[..., :3][mask_slice] = source_slice[mask_slice]
    destination[..., 3][mask_slice] = 255

    original_foreground = source[mask]
    crop_foreground = rgba[rgba[..., 3] == 255, :3]
    require(
        np.array_equal(original_foreground, crop_foreground),
        "raw RGBA crop did not preserve source foreground RGB bytes",
    )
    return rgba, {
        "padding_ratio": padding_ratio,
        "padding_xy_px": [padding_x, padding_y],
        "requested_source_crop_xyxy_exclusive": requested,
        "source_intersection_xyxy_exclusive": [source_x0, source_y0, source_x1, source_y1],
        "transparent_extension_ltrb_px": [
            max(0, -requested[0]),
            max(0, -requested[1]),
            max(0, requested[2] - source_width),
            max(0, requested[3] - source_height),
        ],
        "raw_crop_dimensions_wh": [crop_width, crop_height],
        "foreground_bbox_in_crop_xyxy_exclusive": [
            padding_x,
            padding_y,
            padding_x + bbox_width,
            padding_y + bbox_height,
        ],
        "foreground_rgb_byte_identical_to_source": True,
    }


def make_condition(raw_rgba: np.ndarray, *, size: int) -> tuple[Image.Image, dict[str, Any]]:
    raw = Image.fromarray(raw_rgba)
    scale = min(size / raw.width, size / raw.height)
    resized_wh = [max(1, round(raw.width * scale)), max(1, round(raw.height * scale))]
    resized = raw.resize(tuple(resized_wh), Image.Resampling.LANCZOS)
    condition = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    offset = [(size - resized.width) // 2, (size - resized.height) // 2]
    condition.alpha_composite(resized, tuple(offset))
    return condition, {
        "dimensions_wh": [size, size],
        "resampling": "Pillow.Image.Resampling.LANCZOS",
        "aspect_ratio_preserved": True,
        "resized_crop_dimensions_wh": resized_wh,
        "center_offset_xy_px": offset,
        "background": "transparent_rgba",
        "generative_completion_applied": False,
    }


def checkerboard(size: tuple[int, int], *, cell: int = 16) -> Image.Image:
    yy, xx = np.indices((size[1], size[0]))
    values = np.where(((xx // cell) + (yy // cell)) % 2 == 0, 62, 94).astype(np.uint8)
    rgb = np.repeat(values[..., None], 3, axis=2)
    return Image.fromarray(rgb).convert("RGBA")


def alpha_preview(image: Image.Image) -> Image.Image:
    return Image.alpha_composite(checkerboard(image.size), image.convert("RGBA")).convert("RGB")


def contain(
    image: Image.Image,
    size: tuple[int, int],
    *,
    background: tuple[int, int, int],
) -> Image.Image:
    result = Image.new("RGB", size, background)
    preview = image.convert("RGB")
    preview.thumbnail(size, Image.Resampling.LANCZOS)
    result.paste(preview, ((size[0] - preview.width) // 2, (size[1] - preview.height) // 2))
    return result


def make_overlay(source: np.ndarray, mask: np.ndarray, bbox: list[int]) -> Image.Image:
    base = Image.fromarray(source).convert("RGBA")
    tint = np.zeros((*mask.shape, 4), dtype=np.uint8)
    tint[mask] = [255, 72, 34, 112]
    base = Image.alpha_composite(base, Image.fromarray(tint))
    draw = ImageDraw.Draw(base)
    x0, y0, x1, y1 = bbox
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(255, 214, 63, 255), width=3)
    return base.convert("RGB")


def make_four_panel(
    *,
    source: np.ndarray,
    overlay: Image.Image,
    raw_rgba: Image.Image,
    condition: Image.Image,
    frame_id: str,
    bbox_wh: list[int],
) -> Image.Image:
    tile_size = (420, 300)
    label_height = 42
    panels = [
        (Image.fromarray(source), f"source frame {frame_id}"),
        (overlay, "selected SAM3 mask overlay"),
        (alpha_preview(raw_rgba), f"raw RGBA crop {bbox_wh[0]}x{bbox_wh[1]} bbox"),
        (alpha_preview(condition), "1024x1024 Lanczos RGBA condition"),
    ]
    sheet = Image.new(
        "RGB",
        (tile_size[0] * len(panels), tile_size[1] + label_height),
        (20, 22, 27),
    )
    font = ImageFont.load_default(size=15)
    draw = ImageDraw.Draw(sheet)
    for index, (panel, label) in enumerate(panels):
        tile = contain(panel, tile_size, background=(31, 34, 40))
        x = index * tile_size[0]
        sheet.paste(tile, (x, label_height))
        draw.text((x + 12, 12), label, fill=(236, 239, 244), font=font)
    return sheet


def artifact(path: Path, *, final_path: Path) -> dict[str, Any]:
    digest, size = sha256_file(path)
    return {
        "path": str(final_path),
        "sha256": digest,
        "size_bytes": size,
    }


def _candidate_record(view: dict[str, Any]) -> dict[str, Any]:
    return {
        "frame_id": view["frame_id"],
        "source_rgb": {
            "path": str(view["source_path"]),
            "repo_uri": view["source_repo_uri"],
            "authoritative_upstream_path": view["source_upstream_path"],
            "sha256": view["source_sha256"],
            "size_bytes": view["source_size_bytes"],
        },
        "sam3_mask": {
            "path": str(view["mask_path"]),
            "file_name": view["mask_file_name"],
            "sha256": view["mask_sha256"],
            "size_bytes": view["mask_size_bytes"],
        },
        "source_dimensions_wh": view["dimensions_wh"],
        "foreground_pixel_count": view["pixel_count"],
        "foreground_bbox_xyxy_exclusive": view["bbox_xyxy_exclusive"],
        "foreground_bbox_dimensions_wh": view["bbox_dimensions_wh"],
        "source_edge_distances_ltrb_px": view["edge_distances_px"],
        "minimum_source_edge_distance_px": view["minimum_edge_distance_px"],
        "eligible_non_edge_view": view["eligible_non_edge_view"],
    }


def materialize_object(
    *,
    object_record: dict[str, Any],
    staging_root: Path,
    final_root: Path,
    minimum_edge_distance_px: int,
    padding_ratio: float,
    condition_size: int,
    emit_white_background: bool,
    created_at: str,
) -> dict[str, Any]:
    object_id = object_record["object_id"]
    selected, candidates = select_view(
        object_record,
        minimum_edge_distance_px=minimum_edge_distance_px,
    )
    raw_rgba_array, crop_metrics = build_raw_crop(
        selected["source"],
        selected["mask"],
        selected["bbox_xyxy_exclusive"],
        padding_ratio=padding_ratio,
    )
    raw_rgba = Image.fromarray(raw_rgba_array)
    condition, condition_metrics = make_condition(raw_rgba_array, size=condition_size)
    overlay = make_overlay(
        selected["source"],
        selected["mask"],
        selected["bbox_xyxy_exclusive"],
    )
    object_staging = staging_root / object_id
    object_final = final_root / object_id
    object_staging.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "mask_overlay": object_staging / "mask-overlay.png",
        "raw_rgba_crop": object_staging / "raw-source-crop-rgba.png",
        "condition_rgba": object_staging / f"condition-{condition_size}-rgba.png",
        "four_panel": object_staging / "source-condition-four-panel.png",
    }
    overlay.save(output_paths["mask_overlay"], format="PNG", optimize=False)
    raw_rgba.save(output_paths["raw_rgba_crop"], format="PNG", optimize=False)
    condition.save(output_paths["condition_rgba"], format="PNG", optimize=False)
    if emit_white_background:
        white = Image.new("RGBA", condition.size, (255, 255, 255, 255))
        white.alpha_composite(condition)
        output_paths["condition_white_rgb"] = (
            object_staging / f"condition-{condition_size}-white.png"
        )
        white.convert("RGB").save(
            output_paths["condition_white_rgb"],
            format="PNG",
            optimize=False,
        )
    make_four_panel(
        source=selected["source"],
        overlay=overlay,
        raw_rgba=raw_rgba,
        condition=condition,
        frame_id=selected["frame_id"],
        bbox_wh=selected["bbox_dimensions_wh"],
    ).save(output_paths["four_panel"], format="PNG", optimize=False)

    selected_record = _candidate_record(selected)
    candidate_records = [_candidate_record(candidate) for candidate in candidates]
    outputs = {
        key: artifact(path, final_path=object_final / path.name)
        for key, path in output_paths.items()
    }
    report = {
        "kind": OBJECT_REPORT_KIND,
        "schema_version": 1,
        "created_at": created_at,
        "object_id": object_id,
        "category": object_record.get("category"),
        "selection": {
            "policy": "largest_foreground_pixel_count_among_non_edge_views_then_frame_id",
            "minimum_edge_distance_requirement_px": minimum_edge_distance_px,
            "selected_frame_id": selected["frame_id"],
            "selected": selected_record,
            "candidates": candidate_records,
        },
        "crop": crop_metrics,
        "condition": condition_metrics,
        "inputs_used": {
            "source_rgb_sha256": selected["source_sha256"],
            "sam3_mask_sha256": selected["mask_sha256"],
            "prompted_reference_rgba_used": False,
            "other_image_inputs_used": False,
        },
        "gates": {
            "source_rgb_hash_verified": True,
            "sam3_mask_hash_verified": True,
            "audit_dimensions_bbox_and_pixel_count_verified": True,
            "selected_view_does_not_touch_source_edge": True,
            "raw_foreground_rgb_byte_identical_to_source": True,
            "background_is_transparent": True,
            "generative_completion_applied": False,
        },
        "outputs": outputs,
        "claim_scope": (
            "Source-camera modal RGBA condition only. It contains no generated pixels, "
            "no amodal or back-side completion, and is not a reconstructed 3D result."
        ),
    }
    report_path = object_staging / "machine_report.json"
    atomic_write_json(report_path, report)
    report["machine_report"] = artifact(
        report_path,
        final_path=object_final / report_path.name,
    )
    return report


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    config = parse_config(config_path)
    audit = read_json(config["audit_path"])
    require(audit.get("kind") == AUDIT_KIND, f"audit.kind must be {AUDIT_KIND!r}")
    objects = audit.get("objects")
    require(isinstance(objects, list) and objects, "audit.objects must be non-empty")
    by_id: dict[str, dict[str, Any]] = {}
    for record in objects:
        require(isinstance(record, dict), "audit object entries must be objects")
        object_id = record.get("object_id")
        require(isinstance(object_id, str) and object_id, "audit object_id is invalid")
        require(object_id not in by_id, f"duplicate audit object_id: {object_id}")
        by_id[object_id] = record
    missing = [object_id for object_id in config["object_ids"] if object_id not in by_id]
    require(not missing, f"configured objects are absent from audit: {missing}")

    final_root = config["output_root"]
    staging_root = final_root.with_name(f".{final_root.name}.{os.getpid()}.staging")
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)
    created_at = datetime.now(UTC).isoformat()
    try:
        object_reports = [
            materialize_object(
                object_record=by_id[object_id],
                staging_root=staging_root,
                final_root=final_root,
                minimum_edge_distance_px=config["minimum_edge_distance_px"],
                padding_ratio=config["padding_ratio"],
                condition_size=config["condition_size"],
                emit_white_background=config["emit_white_background"],
                created_at=created_at,
            )
            for object_id in config["object_ids"]
        ]
        report = {
            "kind": REPORT_KIND,
            "schema_version": 1,
            "status": "technical_passed_source_conditions_materialized",
            "created_at": created_at,
            "config": {
                "path": str(config_path),
                "sha256": config["sha256"],
            },
            "source_view_audit": {
                "path": str(config["audit_path"]),
                "sha256": config["audit_sha256"],
                "size_bytes": config["audit_size_bytes"],
            },
            "policy": {
                "view_selection": (
                    "largest foreground mask among views whose four source-edge distances "
                    "are all at least minimum_edge_distance_px"
                ),
                "minimum_edge_distance_px": config["minimum_edge_distance_px"],
                "padding_ratio": config["padding_ratio"],
                "condition_size": config["condition_size"],
                "condition_background": "transparent_rgba",
                "resize": "aspect-preserving Lanczos centered on square canvas",
                "generation_or_inpainting": "forbidden_and_not_used",
                "forbidden_inputs": ["*_prompted_reference_rgba.png"],
            },
            "objects": object_reports,
            "gates": {
                "all_configured_objects_materialized": len(object_reports)
                == len(config["object_ids"]),
                "all_inputs_hash_verified": True,
                "all_selected_views_non_edge": True,
                "source_pixels_only": True,
                "production_manifest_modified": False,
                "trellis2_inference_run": False,
                "promotion_allowed": False,
            },
            "claim_scope": (
                f"{len(object_reports)} source-camera modal conditions for later TRELLIS2 "
                "input review. "
                "No reconstruction, clean plate, amodal completion, or production promotion."
            ),
        }
        atomic_write_json(staging_root / "source_rgba_conditions_report.json", report)
        if final_root.exists():
            require(final_root.is_dir(), f"output path is not a directory: {final_root}")
            shutil.rmtree(final_root)
        final_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging_root, final_root)
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    report = run(args.config)
    output_root = parse_config(args.config.expanduser().resolve())["output_root"]
    summary = {
        "report": str(output_root / "source_rgba_conditions_report.json"),
        "selected_frames": {
            record["object_id"]: record["selection"]["selected_frame_id"]
            for record in report["objects"]
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
