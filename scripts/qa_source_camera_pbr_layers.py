#!/usr/bin/env python3
"""Audit source-camera PBR layers against analytic scene-fit silhouettes."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from qa_scene_camera_silhouette import (
    bbox_iou,
    load_camera,
    make_overlay,
    mask_bbox,
    mask_center,
    project_mesh_silhouette,
)
from render_source_camera_pbr_layer import (
    PBR_RECEIPT_KIND,
    read_json,
    scene_fit_raw_transform,
    sha256_file,
    write_json,
)

EXPECTED_LAYER_ORDER = (
    "sam3_pillow_front",
    "sam3_pillow_left",
    "sam3_pillow_right",
    "sam3_bed_01",
)
ROUND_REMAINING = {
    "round01_front_pillow": [
        "sam3_pillow_left",
        "sam3_pillow_right",
        "sam3_bed_01",
    ],
    "round02_left_pillow": ["sam3_pillow_right", "sam3_bed_01"],
    "round03_right_pillow": ["sam3_bed_01"],
    "round04_bed": [],
}


def resolve_path(value: str, *, relative_to: Path) -> Path:
    path = Path(value).expanduser()
    return (relative_to / path).resolve() if not path.is_absolute() else path.resolve()


def validate_render_receipt(
    path: Path,
    *,
    layer_id: str,
    frame_id: str,
) -> dict[str, Any]:
    receipt = read_json(path)
    if not isinstance(receipt, dict) or receipt.get("kind") != PBR_RECEIPT_KIND:
        raise ValueError(f"invalid PBR receipt for {layer_id}")
    if receipt.get("status") != "technical_passed":
        raise ValueError(f"PBR receipt did not pass for {layer_id}")
    if receipt.get("layer_id") != layer_id or receipt.get("frame_id") != frame_id:
        raise ValueError(f"PBR receipt identity mismatch for {layer_id}")
    if receipt.get("role") != "downstream_object_render":
        raise ValueError(f"PBR receipt role mismatch for {layer_id}")
    if receipt.get("claims_measured_donor") is not False:
        raise ValueError(f"PBR receipt claims measured provenance for {layer_id}")
    if not all(receipt.get("gates", {}).values()):
        raise ValueError(f"one or more PBR receipt gates failed for {layer_id}")
    return receipt


def analytic_silhouette(
    mesh_path: Path,
    report: dict[str, Any],
    camera: dict[str, Any],
) -> np.ndarray:
    loaded = trimesh.load(mesh_path, force="scene")
    mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded
    mode, transform = scene_fit_raw_transform(report)
    return project_mesh_silhouette(
        mesh,
        runtime_pivot=transform[:3, 3] if mode == "baked_linear_plus_runtime_pivot" else None,
        camera=camera,
        image_y_axis="down",
        scene_transform=transform if mode == "full_affine_runtime_transform" else None,
    )


def layer_metrics(expected: np.ndarray, rendered: np.ndarray) -> dict[str, Any]:
    if not np.any(expected) or not np.any(rendered):
        raise ValueError("analytic and rendered silhouettes must be non-empty")
    intersection = int(np.count_nonzero(expected & rendered))
    union = int(np.count_nonzero(expected | rendered))
    expected_pixels = int(expected.sum())
    rendered_pixels = int(rendered.sum())
    expected_bbox = mask_bbox(expected)
    rendered_bbox = mask_bbox(rendered)
    assert expected_bbox is not None and rendered_bbox is not None
    return {
        "analytic_pixels": expected_pixels,
        "rendered_alpha_pixels": rendered_pixels,
        "intersection_pixels": intersection,
        "union_pixels": union,
        "mask_iou": intersection / union,
        "rendered_precision": intersection / rendered_pixels,
        "analytic_recall": intersection / expected_pixels,
        "analytic_bbox_xyxy": expected_bbox,
        "rendered_bbox_xyxy": rendered_bbox,
        "bbox_iou": bbox_iou(expected_bbox, rendered_bbox),
        "center_error_px": float(np.linalg.norm(mask_center(expected) - mask_center(rendered))),
    }


def alpha_composite(source: Image.Image, rgba_path: Path) -> Image.Image:
    layer = Image.open(rgba_path).convert("RGBA")
    if layer.size != source.size:
        raise ValueError("PBR RGBA dimensions do not match source frame")
    result = Image.alpha_composite(source.convert("RGBA"), layer).convert("RGB")
    layer.close()
    return result


def nearest_layer_composite(
    source: Image.Image,
    layer_assets: list[tuple[Path, Path]],
) -> Image.Image:
    height, width = source.height, source.width
    best_depth = np.full((height, width), np.inf, dtype=np.float32)
    selected = np.zeros((height, width, 4), dtype=np.uint8)
    for rgba_path, depth_path in layer_assets:
        rgba = np.asarray(Image.open(rgba_path).convert("RGBA"), dtype=np.uint8)
        depth = np.load(depth_path)
        if rgba.shape[:2] != (height, width) or depth.shape != (height, width):
            raise ValueError("PBR layer dimensions differ during QA composite")
        candidate = (rgba[:, :, 3] >= 128) & np.isfinite(depth) & (depth > 0)
        replace = candidate & (depth < best_depth)
        best_depth[replace] = depth[replace]
        selected[replace] = rgba[replace]
    return Image.alpha_composite(
        source.convert("RGBA"),
        Image.fromarray(selected),
    ).convert("RGB")


def make_contact_sheet(tiles: list[tuple[str, Image.Image]], path: Path) -> None:
    if len(tiles) != 6:
        raise ValueError("PBR QA contact sheet requires exactly six tiles")
    tile_width, tile_height, label_height = 480, 270, 24
    sheet = Image.new("RGB", (tile_width * 3, (tile_height + label_height) * 2), "#111111")
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(tiles):
        row, column = divmod(index, 3)
        x = column * tile_width
        y = row * (tile_height + label_height)
        tile = image.convert("RGB").resize((tile_width, tile_height), Image.Resampling.LANCZOS)
        draw.text((x + 8, y + 5), label, fill="#f4f4f4")
        sheet.paste(tile, (x, y + label_height))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def build_qa(spec_path: Path, output_root: Path) -> dict[str, Any]:
    spec = read_json(spec_path)
    if not isinstance(spec, dict) or not isinstance(spec.get("layers"), list):
        raise ValueError("PBR QA spec must contain layers")
    frame_id = str(spec.get("frame_id"))
    layer_ids = [str(item.get("layer_id")) for item in spec["layers"]]
    if tuple(layer_ids) != EXPECTED_LAYER_ORDER:
        raise ValueError("PBR QA layers are absent, duplicated, or out of fixed order")
    base = spec_path.parent
    cameras_path = resolve_path(str(spec["cameras"]), relative_to=base)
    source_path = resolve_path(str(spec["source_frame"]), relative_to=base)
    camera = load_camera(cameras_path, frame_id)
    source = Image.open(source_path).convert("RGB")
    if source.size != (camera["width"], camera["height"]):
        raise ValueError("source frame dimensions do not match camera")
    thresholds = spec.get("thresholds", {})
    required_thresholds = (
        "minimum_mask_iou",
        "minimum_precision",
        "minimum_recall",
        "minimum_bbox_iou",
        "maximum_center_error_px",
        "minimum_observed_bbox_iou",
        "minimum_observed_recall",
        "maximum_observed_center_error_px",
    )
    if any(not isinstance(thresholds.get(key), int | float) for key in required_thresholds):
        raise ValueError("PBR QA thresholds are incomplete")

    layer_reports: list[dict[str, Any]] = []
    render_index_layers: dict[str, Any] = {}
    composited_tiles: dict[str, Image.Image] = {}
    observed_overlay_tiles: dict[str, Image.Image] = {}
    nearest_assets: list[tuple[Path, Path]] = []
    for item in spec["layers"]:
        layer_id = str(item["layer_id"])
        mesh_path = resolve_path(str(item["mesh"]), relative_to=base)
        report_path = resolve_path(str(item["scene_fit_report"]), relative_to=base)
        render_root = resolve_path(str(item["render_root"]), relative_to=base)
        observed_mask_path = resolve_path(str(item["observed_mask"]), relative_to=base)
        receipt_path = render_root / "receipts" / f"{frame_id}.json"
        receipt = validate_render_receipt(receipt_path, layer_id=layer_id, frame_id=frame_id)
        rgba_path = resolve_path(str(receipt["rgba"]), relative_to=receipt_path.parent)
        depth_path = resolve_path(str(receipt["depth"]), relative_to=receipt_path.parent)
        if receipt.get("rgba_sha256") != sha256_file(rgba_path):
            raise ValueError(f"RGBA hash mismatch for {layer_id}")
        if receipt.get("depth_sha256") != sha256_file(depth_path):
            raise ValueError(f"depth hash mismatch for {layer_id}")
        report = read_json(report_path)
        expected = analytic_silhouette(mesh_path, report, camera)
        rgba = np.asarray(Image.open(rgba_path).convert("RGBA"), dtype=np.uint8)
        rendered = rgba[:, :, 3] >= 128
        observed_image = Image.open(observed_mask_path).convert("L")
        if observed_image.size != source.size:
            raise ValueError(f"observed SAM mask dimensions differ for {layer_id}")
        observed = np.asarray(observed_image, dtype=np.uint8) > 0
        observed_image.close()
        depth = np.load(depth_path)
        if depth.shape != rendered.shape:
            raise ValueError(f"depth shape mismatch for {layer_id}")
        if not np.array_equal(rendered, np.isfinite(depth) & (depth > 0)):
            raise ValueError(f"alpha/depth support mismatch for {layer_id}")
        metrics = layer_metrics(expected, rendered)
        observed_metrics = layer_metrics(observed, rendered)
        gates = {
            "mask_iou": metrics["mask_iou"] >= thresholds["minimum_mask_iou"],
            "precision": metrics["rendered_precision"] >= thresholds["minimum_precision"],
            "recall": metrics["analytic_recall"] >= thresholds["minimum_recall"],
            "bbox_iou": metrics["bbox_iou"] >= thresholds["minimum_bbox_iou"],
            "center_error": metrics["center_error_px"] <= thresholds["maximum_center_error_px"],
            "alpha_support_equals_metric_depth_support": True,
            "observed_bbox_iou": (
                observed_metrics["bbox_iou"] >= thresholds["minimum_observed_bbox_iou"]
            ),
            "observed_recall": (
                observed_metrics["analytic_recall"] >= thresholds["minimum_observed_recall"]
            ),
            "observed_center_error": (
                observed_metrics["center_error_px"]
                <= thresholds["maximum_observed_center_error_px"]
            ),
        }
        overlay_path = output_root / "overlays" / f"{layer_id}.png"
        silhouette_path = output_root / "analytic-silhouettes" / f"{layer_id}.png"
        observed_overlay_path = output_root / "observed-overlays" / f"{layer_id}.png"
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        silhouette_path.parent.mkdir(parents=True, exist_ok=True)
        observed_overlay_path.parent.mkdir(parents=True, exist_ok=True)
        make_overlay(source, expected, rendered).save(overlay_path)
        observed_overlay = make_overlay(source, observed, rendered)
        observed_overlay.save(observed_overlay_path)
        Image.fromarray(expected.astype(np.uint8) * 255).save(silhouette_path)
        layer_reports.append(
            {
                "layer_id": layer_id,
                "status": "passed" if all(gates.values()) else "rejected",
                "mesh_watertight": report.get("mesh", {}).get("watertight"),
                "metrics": metrics,
                "observed_sam_vs_rendered_metrics": observed_metrics,
                "observed_sam_claim_scope": (
                    "Visible SAM mask versus full PBR layer; amodal regions behind other objects "
                    "may lower mask IoU, so bbox, recall, and center are the alignment gates."
                ),
                "thresholds": thresholds,
                "gates": gates,
                "sources": {
                    "mesh": {"path": str(mesh_path), "sha256": sha256_file(mesh_path)},
                    "scene_fit_report": {
                        "path": str(report_path),
                        "sha256": sha256_file(report_path),
                    },
                    "render_receipt": {
                        "path": str(receipt_path),
                        "sha256": sha256_file(receipt_path),
                    },
                    "observed_sam_mask": {
                        "path": str(observed_mask_path),
                        "sha256": sha256_file(observed_mask_path),
                    },
                },
                "outputs": {
                    "overlay": {"path": str(overlay_path), "sha256": sha256_file(overlay_path)},
                    "analytic_silhouette": {
                        "path": str(silhouette_path),
                        "sha256": sha256_file(silhouette_path),
                    },
                    "observed_sam_overlay": {
                        "path": str(observed_overlay_path),
                        "sha256": sha256_file(observed_overlay_path),
                    },
                },
            }
        )
        render_index_layers[layer_id] = {
            "rgba": {"path": str(rgba_path), "sha256": sha256_file(rgba_path)},
            "depth": {"path": str(depth_path), "sha256": sha256_file(depth_path)},
            "receipt": {"path": str(receipt_path), "sha256": sha256_file(receipt_path)},
        }
        composited_tiles[layer_id] = alpha_composite(source, rgba_path)
        observed_overlay_tiles[layer_id] = observed_overlay
        nearest_assets.append((rgba_path, depth_path))

    contact_path = output_root / "source_camera_pbr_qa_contact_sheet.png"
    nearest = nearest_layer_composite(source, nearest_assets)
    make_contact_sheet(
        [
            (f"source {frame_id}", source),
            ("front pillow", composited_tiles["sam3_pillow_front"]),
            ("left pillow", composited_tiles["sam3_pillow_left"]),
            ("right pillow", composited_tiles["sam3_pillow_right"]),
            ("bed v2", composited_tiles["sam3_bed_01"]),
            ("nearest-depth QA overlay (not final)", nearest),
        ],
        contact_path,
    )
    observed_contact_path = output_root / "observed_sam_vs_pbr_contact_sheet.png"
    make_contact_sheet(
        [
            (f"source {frame_id}", source),
            ("front: SAM vs PBR", observed_overlay_tiles["sam3_pillow_front"]),
            ("left: SAM vs PBR", observed_overlay_tiles["sam3_pillow_left"]),
            ("right: SAM vs PBR", observed_overlay_tiles["sam3_pillow_right"]),
            ("bed: SAM vs PBR", observed_overlay_tiles["sam3_bed_01"]),
            ("nearest-depth QA overlay (not final)", nearest),
        ],
        observed_contact_path,
    )
    passed = all(item["status"] == "passed" for item in layer_reports)
    render_index = {
        "schema_version": 1,
        "kind": "video2world.pbr_layer_render_index",
        "status": "qa_passed_single_frame_only" if passed else "rejected",
        "promotion_approved": False,
        "frame_ids": [frame_id],
        "layer_order": list(EXPECTED_LAYER_ORDER),
        "frames": {frame_id: render_index_layers},
        "round_remaining_object_ids": ROUND_REMAINING,
        "contract": {
            "removed_object_must_not_reappear_in_same_or_later_round": True,
            "previous_round_unresolved_must_not_feed_next_round": True,
            "this_index_is_not_a_final_composite": True,
        },
    }
    render_index_path = output_root / "render_index.json"
    write_json(render_index_path, render_index)
    write_json(
        output_root / "render_index_receipt.json",
        {
            "schema_version": 1,
            "kind": "video2world.pbr_layer_render_index_receipt",
            "render_index": str(render_index_path),
            "render_index_sha256": sha256_file(render_index_path),
            "frame_count": 1,
            "layer_count": len(EXPECTED_LAYER_ORDER),
        },
    )
    report = {
        "schema_version": 1,
        "kind": "video2world.source_camera_pbr_layer_qa",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "passed_single_frame_only" if passed else "rejected",
        "promotion_approved": False,
        "promotion_blocker": "000067 QA does not authorize 25-frame or final composition",
        "frame_id": frame_id,
        "camera_convention": "camera_to_world_right_down_forward; image_y_down",
        "sources": {
            "spec": {"path": str(spec_path), "sha256": sha256_file(spec_path)},
            "cameras": {"path": str(cameras_path), "sha256": sha256_file(cameras_path)},
            "source_frame": {"path": str(source_path), "sha256": sha256_file(source_path)},
        },
        "layers": layer_reports,
        "outputs": {
            "contact_sheet": {"path": str(contact_path), "sha256": sha256_file(contact_path)},
            "observed_sam_contact_sheet": {
                "path": str(observed_contact_path),
                "sha256": sha256_file(observed_contact_path),
            },
            "render_index": {
                "path": str(render_index_path),
                "sha256": sha256_file(render_index_path),
            },
        },
    }
    write_json(output_root / "source_camera_pbr_layer_qa.json", report)
    source.close()
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_qa(args.spec.expanduser().resolve(), args.output.expanduser().resolve())
    print(
        json.dumps(
            {
                "status": report["status"],
                "frame_id": report["frame_id"],
                "report": str(args.output / "source_camera_pbr_layer_qa.json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
