#!/usr/bin/env python3
"""Materialize hash-bound multiview evidence for segmented object-part audits.

The tool is deliberately non-generative. It renders source RGB crops beside the
selected modal mask for every audited frame and checks whether the SAM index
contains configured labels for required object parts. It never infers a missing
part mask or treats a support-surface mask as an object part.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps

from video2world.hashing import atomic_write_json, sha256_file

CONFIG_KIND = "video2world.multiview_mask_part_audit_config"
SOURCE_AUDIT_KIND = "video2world.multiview_scene_fit_evidence_audit"
REPORT_KIND = "video2world.multiview_mask_part_audit"


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


def _integer(value: Any, *, label: str, minimum: int) -> int:
    require(isinstance(value, int) and not isinstance(value, bool), f"{label} must be integer")
    require(value >= minimum, f"{label} must be >= {minimum}")
    return value


def _number(value: Any, *, label: str, minimum: float, maximum: float) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be numeric",
    )
    result = float(value)
    require(math.isfinite(result), f"{label} must be finite")
    require(minimum <= result <= maximum, f"{label} must be in [{minimum}, {maximum}]")
    return result


def _labels(value: Any, *, label: str) -> list[str]:
    require(isinstance(value, list) and value, f"{label} must be a non-empty list")
    require(all(isinstance(item, str) and item for item in value), f"{label} is invalid")
    result = [item.casefold() for item in value]
    require(len(result) == len(set(result)), f"{label} must not contain duplicates")
    return result


def parse_config(config_path: Path) -> dict[str, Any]:
    config = read_json(config_path)
    require(config.get("schema_version") == 1, "config.schema_version must be 1")
    require(config.get("kind") == CONFIG_KIND, f"config.kind must be {CONFIG_KIND}")
    base = config_path.parent
    audit_path, audit_sha, audit_size = verified_path(
        config.get("source_audit"), relative_to=base, label="source_audit"
    )
    mask_index_path, mask_index_sha, mask_index_size = verified_path(
        config.get("mask_index"), relative_to=base, label="mask_index"
    )
    object_ids = config.get("object_ids")
    require(isinstance(object_ids, list) and object_ids, "object_ids must be non-empty")
    require(
        all(isinstance(item, str) and item for item in object_ids),
        "object_ids entries must be non-empty strings",
    )
    require(len(object_ids) == len(set(object_ids)), "object_ids must be unique")
    aliases = config.get("part_label_aliases")
    require(isinstance(aliases, dict), "part_label_aliases must be an object")
    primary_labels = _labels(aliases.get("primary"), label="part_label_aliases.primary")
    required_labels = _labels(
        aliases.get("required_container"), label="part_label_aliases.required_container"
    )
    review = config.get("review")
    require(isinstance(review, dict), "review must be an object")
    padding_ratio = _number(
        review.get("crop_padding_ratio"),
        label="review.crop_padding_ratio",
        minimum=0.25,
        maximum=1.5,
    )
    columns = _integer(review.get("columns"), label="review.columns", minimum=1)
    panel_width = _integer(review.get("panel_width"), label="review.panel_width", minimum=120)
    panel_height = _integer(review.get("panel_height"), label="review.panel_height", minimum=100)
    output_directory = resolve_path(
        config.get("output_directory"), relative_to=base, label="output_directory"
    )
    config_sha, config_size = sha256_file(config_path)
    return {
        "path": config_path,
        "sha256": config_sha,
        "size_bytes": config_size,
        "audit_path": audit_path,
        "audit_sha256": audit_sha,
        "audit_size_bytes": audit_size,
        "mask_index_path": mask_index_path,
        "mask_index_sha256": mask_index_sha,
        "mask_index_size_bytes": mask_index_size,
        "object_ids": object_ids,
        "primary_labels": primary_labels,
        "required_labels": required_labels,
        "padding_ratio": padding_ratio,
        "columns": columns,
        "panel_width": panel_width,
        "panel_height": panel_height,
        "output_directory": output_directory,
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


def _crop_box(
    mask_box: list[int], projection_box: list[float], *, width: int, height: int, ratio: float
) -> list[int]:
    x0 = min(mask_box[0], math.floor(projection_box[0]))
    y0 = min(mask_box[1], math.floor(projection_box[1]))
    x1 = max(mask_box[2], math.ceil(projection_box[2]))
    y1 = max(mask_box[3], math.ceil(projection_box[3]))
    object_width = max(1, x1 - x0)
    object_height = max(1, y1 - y0)
    pad_x = max(12, math.ceil(object_width * ratio))
    pad_y = max(12, math.ceil(object_height * ratio))
    return [max(0, x0 - pad_x), max(0, y0 - pad_y), min(width, x1 + pad_x), min(height, y1 + pad_y)]


def _fit_panel(image: Image.Image, *, width: int, height: int) -> Image.Image:
    fitted = ImageOps.contain(image, (width, height), Image.Resampling.NEAREST)
    panel = Image.new("RGB", (width, height), (22, 25, 29))
    panel.paste(fitted, ((width - fitted.width) // 2, (height - fitted.height) // 2))
    return panel


def _draw_projection(image: Image.Image, projection: list[float], crop: list[int]) -> None:
    crop_width = crop[2] - crop[0]
    crop_height = crop[3] - crop[1]
    scale = min(image.width / crop_width, image.height / crop_height)
    rendered_width = crop_width * scale
    rendered_height = crop_height * scale
    offset_x = (image.width - rendered_width) / 2
    offset_y = (image.height - rendered_height) / 2
    points = [
        offset_x + (projection[0] - crop[0]) * scale,
        offset_y + (projection[1] - crop[1]) * scale,
        offset_x + (projection[2] - crop[0]) * scale,
        offset_y + (projection[3] - crop[1]) * scale,
    ]
    draw = ImageDraw.Draw(image)
    draw.rectangle(points, outline=(255, 176, 32), width=2)


def _render_tile(
    *,
    source: Image.Image,
    mask: np.ndarray,
    crop: list[int],
    projection: list[float],
    title: str,
    panel_width: int,
    panel_height: int,
) -> Image.Image:
    x0, y0, x1, y1 = crop
    source_crop = source.crop(crop)
    mask_crop = Image.fromarray((mask[y0:y1, x0:x1] * 255).astype(np.uint8))
    dark = ImageEnhance.Brightness(source_crop).enhance(0.30)
    foreground = Image.composite(source_crop, dark, mask_crop)
    green = Image.new("RGB", source_crop.size, (18, 210, 116))
    green_layer = Image.blend(foreground, green, 0.32)
    overlay = Image.composite(green_layer, dark, mask_crop)
    left = _fit_panel(source_crop, width=panel_width, height=panel_height)
    right = _fit_panel(overlay, width=panel_width, height=panel_height)
    _draw_projection(left, projection, crop)
    _draw_projection(right, projection, crop)
    header_height = 26
    tile = Image.new("RGB", (panel_width * 2 + 4, panel_height + header_height), (16, 18, 22))
    tile.paste(left, (0, header_height))
    tile.paste(right, (panel_width + 4, header_height))
    draw = ImageDraw.Draw(tile)
    draw.text((6, 6), title, fill=(236, 239, 244), font=ImageFont.load_default())
    return tile


def _frame_label_inventory(mask_index: dict[str, Any], frame_ids: set[str]) -> dict[str, Any]:
    items = mask_index.get("items")
    require(isinstance(items, list), "mask index items must be a list")
    per_frame: dict[str, dict[str, int]] = {frame_id: {} for frame_id in sorted(frame_ids)}
    for item in items:
        require(isinstance(item, dict), "mask index items must be objects")
        image_name = item.get("image")
        label = item.get("label")
        if not isinstance(image_name, str) or not isinstance(label, str):
            continue
        frame_id = Path(image_name).stem
        if frame_id not in per_frame:
            continue
        normalized = label.casefold()
        per_frame[frame_id][normalized] = per_frame[frame_id].get(normalized, 0) + 1
    aggregate: dict[str, int] = {}
    for counts in per_frame.values():
        for label, count in counts.items():
            aggregate[label] = aggregate.get(label, 0) + count
    return {"aggregate": dict(sorted(aggregate.items())), "per_frame": per_frame}


def materialize(config_path: Path) -> dict[str, Any]:
    config = parse_config(config_path.resolve())
    audit = read_json(config["audit_path"])
    require(
        audit.get("kind") == SOURCE_AUDIT_KIND, f"source audit kind must be {SOURCE_AUDIT_KIND}"
    )
    mask_index = read_json(config["mask_index_path"])
    audit_objects = audit.get("objects")
    require(isinstance(audit_objects, list), "source audit objects must be a list")
    by_id = {item.get("object_id"): item for item in audit_objects if isinstance(item, dict)}
    for object_id in config["object_ids"]:
        require(object_id in by_id, f"object_id is missing from source audit: {object_id}")
    all_frame_ids = {
        str(view.get("frame_id"))
        for object_id in config["object_ids"]
        for view in by_id[object_id].get("views", [])
        if isinstance(view, dict)
    }
    inventory = _frame_label_inventory(mask_index, all_frame_ids)
    missing_required = [
        label for label in config["required_labels"] if inventory["aggregate"].get(label, 0) == 0
    ]
    output_root: Path = config["output_directory"]
    output_root.mkdir(parents=True, exist_ok=True)
    report_objects = []
    for object_id in config["object_ids"]:
        object_record = by_id[object_id]
        views = object_record.get("views")
        require(isinstance(views, list) and views, f"{object_id}: views must be non-empty")
        tiles = []
        report_views = []
        for view in views:
            require(isinstance(view, dict), f"{object_id}: views entries must be objects")
            frame_id = view.get("frame_id")
            association = view.get("association")
            source_record = view.get("source_rgb")
            mask_record = view.get("mask")
            require(isinstance(frame_id, str) and frame_id, f"{object_id}: invalid frame_id")
            require(isinstance(association, dict), f"{object_id}/{frame_id}: invalid association")
            require(isinstance(source_record, dict), f"{object_id}/{frame_id}: invalid source_rgb")
            require(isinstance(mask_record, dict), f"{object_id}/{frame_id}: invalid mask")
            require(
                association.get("status") == "passed", f"{object_id}/{frame_id}: association failed"
            )
            source_path, source_sha, source_size = verified_path(
                {"path": source_record.get("path"), "sha256": source_record.get("sha256")},
                relative_to=config["audit_path"].parent,
                label=f"{object_id}/{frame_id}.source_rgb",
            )
            mask_path, mask_sha, mask_size = verified_path(
                {"path": mask_record.get("path"), "sha256": mask_record.get("sha256")},
                relative_to=config["audit_path"].parent,
                label=f"{object_id}/{frame_id}.mask",
            )
            with Image.open(source_path) as image:
                source = image.convert("RGB")
            with Image.open(mask_path) as image:
                mask = np.asarray(image.convert("L"), dtype=np.uint8) > 0
            require(
                mask.shape == (source.height, source.width),
                f"{object_id}/{frame_id}: dimensions differ",
            )
            observed_bbox = _bbox(mask)
            require(
                observed_bbox == mask_record.get("pixel_bbox_xyxy_exclusive"),
                f"{object_id}/{frame_id}: mask bbox differs from audit",
            )
            projection = association.get("projected_anchor_bbox_xyxy")
            require(
                isinstance(projection, list) and len(projection) == 4,
                f"{object_id}/{frame_id}: projection bbox is invalid",
            )
            projection = [float(value) for value in projection]
            crop = _crop_box(
                observed_bbox,
                projection,
                width=source.width,
                height=source.height,
                ratio=config["padding_ratio"],
            )
            title = (
                f"{frame_id}  {association.get('selected_mask_file_name')}  "
                f"hit={float(association.get('selected_hit_ratio')):.3f}"
            )
            tiles.append(
                _render_tile(
                    source=source,
                    mask=mask,
                    crop=crop,
                    projection=projection,
                    title=title,
                    panel_width=config["panel_width"],
                    panel_height=config["panel_height"],
                )
            )
            report_views.append(
                {
                    "frame_id": frame_id,
                    "source_rgb": {
                        "path": str(source_path),
                        "sha256": source_sha,
                        "size_bytes": source_size,
                    },
                    "selected_mask": {
                        "path": str(mask_path),
                        "sha256": mask_sha,
                        "size_bytes": mask_size,
                        "file_name": association.get("selected_mask_file_name"),
                        "bbox_xyxy_exclusive": observed_bbox,
                        "pixel_count": int(np.count_nonzero(mask)),
                    },
                    "association": {
                        "selected_hit_ratio": association.get("selected_hit_ratio"),
                        "hit_ratio_margin": association.get("hit_ratio_margin"),
                        "projected_anchor_bbox_xyxy": projection,
                    },
                    "review_crop_xyxy_exclusive": crop,
                }
            )
        columns = min(config["columns"], len(tiles))
        rows = math.ceil(len(tiles) / columns)
        tile_width = tiles[0].width
        tile_height = tiles[0].height
        sheet = Image.new("RGB", (tile_width * columns, tile_height * rows), (12, 14, 18))
        for index, tile in enumerate(tiles):
            sheet.paste(tile, ((index % columns) * tile_width, (index // columns) * tile_height))
        object_dir = output_root / object_id
        object_dir.mkdir(parents=True, exist_ok=True)
        contact_path = object_dir / "all-frames-source-mask-contact-sheet.png"
        sheet.save(contact_path, format="PNG", optimize=True)
        contact_sha, contact_size = sha256_file(contact_path)
        report_objects.append(
            {
                "object_id": object_id,
                "frame_count": len(report_views),
                "status": "rejected_missing_required_part_mask"
                if missing_required
                else "manual_part_instance_association_required",
                "union_ready": False,
                "fail_closed_reasons": (
                    [
                        "The SAM index has no mask under any configured required-container label.",
                        "A support-surface mask must not be cropped or relabeled as the "
                        "missing object part.",
                        "The selected modal plant masks alone are insufficient to prove "
                        "complete potted-plant coverage.",
                    ]
                    if missing_required
                    else [
                        "Required-part labels exist, but their physical instance association "
                        "is not proven by this audit."
                    ]
                ),
                "contact_sheet": {
                    "path": str(contact_path),
                    "sha256": contact_sha,
                    "size_bytes": contact_size,
                    "layout": {
                        "columns": columns,
                        "rows": rows,
                        "panel_semantics": ["source_rgb", "selected_modal_mask_overlay"],
                    },
                },
                "views": report_views,
            }
        )
    report = {
        "schema_version": 1,
        "kind": REPORT_KIND,
        "status": "rejected_missing_required_part_mask"
        if missing_required
        else "manual_part_instance_association_required",
        "claim_scope": (
            "Hash-bound source/mask inspection only. No mask is generated, expanded, "
            "relabeled, or unioned."
        ),
        "config": {
            "path": str(config["path"]),
            "sha256": config["sha256"],
            "size_bytes": config["size_bytes"],
        },
        "inputs": {
            "source_audit": {
                "path": str(config["audit_path"]),
                "sha256": config["audit_sha256"],
                "size_bytes": config["audit_size_bytes"],
            },
            "mask_index": {
                "path": str(config["mask_index_path"]),
                "sha256": config["mask_index_sha256"],
                "size_bytes": config["mask_index_size_bytes"],
            },
        },
        "part_label_contract": {
            "primary_aliases": config["primary_labels"],
            "required_container_aliases": config["required_labels"],
            "missing_required_container_aliases": missing_required,
        },
        "mask_index_label_inventory": inventory,
        "objects": report_objects,
    }
    report_path = output_root / "mask-part-audit.json"
    atomic_write_json(report_path, report)
    report_sha, report_size = sha256_file(report_path)
    return {
        "report": {"path": str(report_path), "sha256": report_sha, "size_bytes": report_size},
        "status": report["status"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    result = materialize(parse_args().config)
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
