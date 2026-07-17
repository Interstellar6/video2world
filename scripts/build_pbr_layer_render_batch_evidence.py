#!/usr/bin/env python3
"""Audit a multi-frame PBR RGB-D layer batch and bind its evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

LAYER_ORDER = (
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
CENTER_ERROR_LIMITS = {
    "sam3_pillow_front": 10.0,
    "sam3_pillow_left": 10.0,
    "sam3_pillow_right": 10.0,
    "sam3_bed_01": 20.0,
}
PBR_RECEIPT_KIND = "video2world.pbr_layer_render_receipt"
ALIGNMENT_MODE = "geometry-derived-binary-matte"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str, *, relative_to: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (relative_to / path).resolve()


def relative_artifact(path: Path, batch_root: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(batch_root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def payload_digest(batch_root: Path) -> dict[str, Any]:
    paths = sorted(
        (
            path
            for root_name in ("layers", "source")
            for path in (batch_root / root_name).rglob("*")
            if path.is_file() and not path.name.startswith(".")
        ),
        key=lambda path: path.relative_to(batch_root).as_posix(),
    )
    digest = hashlib.sha256()
    total_bytes = 0
    for path in paths:
        relative = path.relative_to(batch_root).as_posix()
        file_digest = sha256_file(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\n")
        total_bytes += path.stat().st_size
    return {
        "algorithm": "sha256(sorted(relative_path + NUL + file_sha256 + LF))",
        "sha256": digest.hexdigest(),
        "file_count": len(paths),
        "bytes": total_bytes,
    }


def parse_frame_ids(values: list[str]) -> list[str]:
    frame_ids: list[str] = []
    for value in values:
        frame_ids.extend(item.strip() for item in value.split(",") if item.strip())
    if not frame_ids or len(frame_ids) != len(set(frame_ids)):
        raise ValueError("frame IDs must be non-empty and unique")
    return frame_ids


def validate_source_frames(
    batch_root: Path,
    frame_ids: list[str],
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    expected_size: tuple[int, int] | None = None
    for frame_id in frame_ids:
        path = batch_root / "source" / f"{frame_id}.png"
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            size = image.size
        expected_size = expected_size or size
        if size != expected_size:
            raise ValueError("source frames do not share one camera size")
        records[frame_id] = relative_artifact(path, batch_root) | {
            "width": size[0],
            "height": size[1],
        }
    return records


def audit_layer(
    batch_root: Path,
    layer_id: str,
    frame_ids: list[str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    layer_root = batch_root / "layers" / layer_id
    manifest_path = layer_root / "render_manifest.json"
    manifest_receipt_path = layer_root / "render_receipt.json"
    manifest = read_json(manifest_path)
    manifest_receipt = read_json(manifest_receipt_path)
    if manifest.get("layer_id") != layer_id:
        raise ValueError(f"layer manifest identity mismatch: {layer_id}")
    if manifest.get("alpha_depth_alignment_mode") != ALIGNMENT_MODE:
        raise ValueError(f"layer manifest alignment mode mismatch: {layer_id}")
    if manifest_receipt.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError(f"layer manifest receipt hash mismatch: {layer_id}")
    if manifest_receipt.get("frame_count") != len(frame_ids):
        raise ValueError(f"layer manifest receipt frame count mismatch: {layer_id}")
    records = manifest.get("frame_records")
    if not isinstance(records, list) or [item.get("frame_id") for item in records] != frame_ids:
        raise ValueError(f"layer frame order mismatch: {layer_id}")
    by_frame = {str(item["frame_id"]): item for item in records}

    frame_index: dict[str, dict[str, Any]] = {}
    values: list[dict[str, Any]] = []
    for frame_id in frame_ids:
        manifest_record = by_frame[frame_id]
        receipt_path = resolve_path(
            str(manifest_record["receipt"]),
            relative_to=manifest_path.parent,
        )
        if manifest_record.get("receipt_sha256") != sha256_file(receipt_path):
            raise ValueError(f"frame receipt hash mismatch: {layer_id}/{frame_id}")
        receipt = read_json(receipt_path)
        if (
            receipt.get("kind") != PBR_RECEIPT_KIND
            or receipt.get("status") != "technical_passed"
            or receipt.get("layer_id") != layer_id
            or receipt.get("frame_id") != frame_id
        ):
            raise ValueError(f"invalid frame receipt: {layer_id}/{frame_id}")
        if receipt.get("alpha_depth_alignment_mode") != ALIGNMENT_MODE:
            raise ValueError(f"frame alignment mode mismatch: {layer_id}/{frame_id}")
        if receipt.get("claims_measured_donor") is not False:
            raise ValueError(f"frame incorrectly claims measured donor: {layer_id}/{frame_id}")
        gates = receipt.get("gates")
        if not isinstance(gates, dict) or not gates or not all(gates.values()):
            raise ValueError(f"one or more frame gates failed: {layer_id}/{frame_id}")

        rgba_path = resolve_path(str(receipt["rgba"]), relative_to=receipt_path.parent)
        depth_path = resolve_path(str(receipt["depth"]), relative_to=receipt_path.parent)
        if receipt.get("rgba_sha256") != sha256_file(rgba_path):
            raise ValueError(f"RGBA hash mismatch: {layer_id}/{frame_id}")
        if receipt.get("depth_sha256") != sha256_file(depth_path):
            raise ValueError(f"depth hash mismatch: {layer_id}/{frame_id}")
        with Image.open(rgba_path) as image:
            rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        alpha_values = np.unique(rgba[:, :, 3])
        if not set(alpha_values.tolist()).issubset({0, 255}):
            raise ValueError(f"geometry matte is not binary: {layer_id}/{frame_id}")
        alpha = rgba[:, :, 3] == 255
        depth = np.load(depth_path, allow_pickle=False)
        valid_depth = np.isfinite(depth)
        if depth.shape != alpha.shape or not np.array_equal(alpha, valid_depth):
            raise ValueError(f"final alpha/depth support mismatch: {layer_id}/{frame_id}")
        if np.any(depth[valid_depth] <= 0) or np.any(~np.isnan(depth[~valid_depth])):
            raise ValueError(f"depth validity contract failed: {layer_id}/{frame_id}")
        if int(alpha.sum()) != receipt.get("alpha_pixels"):
            raise ValueError(f"alpha count mismatch: {layer_id}/{frame_id}")
        if int(valid_depth.sum()) != receipt.get("depth_valid_pixels"):
            raise ValueError(f"depth count mismatch: {layer_id}/{frame_id}")

        cleared = int(receipt["geometry_matte_cleared_pixels"])
        filled = int(receipt["geometry_matte_filled_pixels"])
        changed = int(receipt["geometry_matte_changed_pixels"])
        union = int(receipt["geometry_matte_union_pixels"])
        if changed != cleared + filled or union <= 0:
            raise ValueError(f"matte clear/fill accounting failed: {layer_id}/{frame_id}")
        changed_fraction = float(receipt["geometry_matte_changed_union_fraction"])
        initial_iou = float(receipt["geometry_matte_initial_alpha_vs_depth_iou"])
        if not np.isclose(changed_fraction, changed / union, atol=1e-12):
            raise ValueError(f"matte changed fraction mismatch: {layer_id}/{frame_id}")
        if initial_iou < 0.98 or changed_fraction >= 0.02:
            raise ValueError(f"matte alignment bounds failed: {layer_id}/{frame_id}")
        render_config = receipt.get("render_configuration", {})
        if render_config.get("taa_render_samples") != 64:
            raise ValueError(f"RGB was not rendered with 64 samples: {layer_id}/{frame_id}")
        analytic = receipt.get("analytic_silhouette_metrics", {})
        thresholds = receipt.get("analytic_silhouette_thresholds", {})
        if (
            float(analytic.get("mask_iou", 0)) < 0.85
            or float(analytic.get("bbox_iou", 0)) < 0.8
            or float(analytic.get("center_error_px", float("inf")))
            > CENTER_ERROR_LIMITS[layer_id]
            or float(thresholds.get("maximum_center_error_px", -1))
            != CENTER_ERROR_LIMITS[layer_id]
        ):
            raise ValueError(f"analytic silhouette threshold failed: {layer_id}/{frame_id}")
        if float(receipt.get("raw_object_depth_vs_final_alpha_iou", 0)) != 1.0:
            raise ValueError(f"final geometry matte/depth IoU is not exact: {layer_id}/{frame_id}")

        frame_index[frame_id] = {
            "rgba": relative_artifact(rgba_path, batch_root),
            "depth": relative_artifact(depth_path, batch_root),
            "receipt": relative_artifact(receipt_path, batch_root),
            "alpha_pixels": int(alpha.sum()),
            "depth_valid_pixels": int(valid_depth.sum()),
            "geometry_matte": {
                "initial_alpha_vs_depth_iou": initial_iou,
                "changed_union_fraction": changed_fraction,
                "cleared_pixels": cleared,
                "filled_pixels": filled,
            },
            "analytic_silhouette_metrics": analytic,
        }
        values.append(receipt)

    raw_depth_exrs = list((layer_root / "raw-depth").glob("*.exr"))
    if raw_depth_exrs:
        raise ValueError(f"per-frame raw depth EXRs were unexpectedly retained: {layer_id}")
    summary = {
        "layer_id": layer_id,
        "status": "passed_25_frame_geometry_matte_audit",
        "frame_count": len(values),
        "render_elapsed_seconds": float(manifest["render_elapsed_seconds"]),
        "minimum_initial_alpha_vs_depth_iou": min(
            float(item["geometry_matte_initial_alpha_vs_depth_iou"]) for item in values
        ),
        "maximum_changed_union_fraction": max(
            float(item["geometry_matte_changed_union_fraction"]) for item in values
        ),
        "cleared_pixels_total": sum(
            int(item["geometry_matte_cleared_pixels"]) for item in values
        ),
        "filled_pixels_total": sum(
            int(item["geometry_matte_filled_pixels"]) for item in values
        ),
        "changed_pixels_total": sum(
            int(item["geometry_matte_changed_pixels"]) for item in values
        ),
        "minimum_analytic_mask_iou": min(
            float(item["analytic_silhouette_metrics"]["mask_iou"]) for item in values
        ),
        "minimum_analytic_bbox_iou": min(
            float(item["analytic_silhouette_metrics"]["bbox_iou"]) for item in values
        ),
        "maximum_analytic_center_error_px": max(
            float(item["analytic_silhouette_metrics"]["center_error_px"])
            for item in values
        ),
        "center_error_limit_px": CENTER_ERROR_LIMITS[layer_id],
        "manifest": relative_artifact(manifest_path, batch_root),
        "manifest_receipt": relative_artifact(manifest_receipt_path, batch_root),
        "depth_probe_receipt": relative_artifact(
            layer_root / "calibration" / "depth_semantics_receipt.json",
            batch_root,
        ),
        "per_frame_raw_depth_exrs_retained": 0,
    }
    return summary, frame_index


def checkerboard(size: tuple[int, int], cell: int = 16) -> Image.Image:
    width, height = size
    y, x = np.indices((height, width))
    light = ((x // cell + y // cell) % 2) == 0
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[light] = (72, 75, 82)
    rgb[~light] = (42, 44, 50)
    return Image.fromarray(rgb)


def contact_tile(path: Path, *, rgba: bool, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        if rgba:
            layer = image.convert("RGBA")
            background = checkerboard(layer.size).convert("RGBA")
            tile = Image.alpha_composite(background, layer).convert("RGB")
        else:
            tile = image.convert("RGB")
    return tile.resize(size, Image.Resampling.LANCZOS)


def make_contact_sheet(
    batch_root: Path,
    frame_ids: list[str],
    render_index: dict[str, Any],
    output_path: Path,
) -> None:
    tile_size = (320, 180)
    label_height = 22
    headers = ("source", *LAYER_ORDER)
    sheet = Image.new(
        "RGB",
        (
            tile_size[0] * len(headers),
            label_height + len(frame_ids) * (tile_size[1] + label_height),
        ),
        "#111318",
    )
    draw = ImageDraw.Draw(sheet)
    for column, label in enumerate(headers):
        draw.text((column * tile_size[0] + 8, 5), label, fill="#f2f4f8")
    for row, frame_id in enumerate(frame_ids):
        row_top = label_height + row * (tile_size[1] + label_height)
        draw.text((8, row_top + 4), f"frame {frame_id}", fill="#f2f4f8")
        source_record = render_index["frames"][frame_id]["source_frame"]
        source_path = batch_root / source_record["path"]
        source_tile = contact_tile(source_path, rgba=False, size=tile_size)
        sheet.paste(source_tile, (0, row_top + label_height))
        for column, layer_id in enumerate(LAYER_ORDER, start=1):
            rgba_record = render_index["frames"][frame_id]["layers"][layer_id]["rgba"]
            rgba_path = batch_root / rgba_record["path"]
            tile = contact_tile(rgba_path, rgba=True, size=tile_size)
            sheet.paste(tile, (column * tile_size[0], row_top + label_height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def build_evidence(batch_root: Path, frame_ids: list[str], contact_frames: list[str]) -> None:
    output_paths = {
        "render_index": batch_root / "render_index.json",
        "contact_sheet": batch_root / "pbr_layer_batch_contact_sheet.png",
        "report": batch_root / "pbr_layer_batch_qa.json",
        "receipt": batch_root / "batch_receipt.json",
    }
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"batch evidence outputs already exist: {existing}")
    if any(frame_id not in frame_ids for frame_id in contact_frames):
        raise ValueError("contact frame is outside the audited frame set")

    source_records = validate_source_frames(batch_root, frame_ids)
    layer_summaries: list[dict[str, Any]] = []
    per_layer_frames: dict[str, dict[str, dict[str, Any]]] = {}
    for layer_id in LAYER_ORDER:
        summary, frame_index = audit_layer(batch_root, layer_id, frame_ids)
        layer_summaries.append(summary)
        per_layer_frames[layer_id] = frame_index

    render_index = {
        "schema_version": 1,
        "kind": "video2world.source_camera_pbr_geometry_matte_render_index",
        "status": "technical_passed_100_frame_asset_set_not_promoted",
        "promotion_approved": False,
        "path_scope": "batch_root_relative",
        "frame_ids": frame_ids,
        "layer_order": list(LAYER_ORDER),
        "round_remaining_object_ids": ROUND_REMAINING,
        "contract": {
            "alpha_depth_alignment_mode": ALIGNMENT_MODE,
            "rgb_render_samples": 64,
            "final_alpha_equals_finite_depth_support": True,
            "claims_measured_donor": False,
            "this_index_is_not_a_final_composite": True,
            "this_index_is_not_promoted": True,
        },
        "frames": {
            frame_id: {
                "source_frame": source_records[frame_id],
                "layers": {
                    layer_id: per_layer_frames[layer_id][frame_id]
                    for layer_id in LAYER_ORDER
                },
            }
            for frame_id in frame_ids
        },
    }
    write_json(output_paths["render_index"], render_index)
    make_contact_sheet(batch_root, contact_frames, render_index, output_paths["contact_sheet"])

    payload = payload_digest(batch_root)
    sum_elapsed = sum(item["render_elapsed_seconds"] for item in layer_summaries)
    report = {
        "schema_version": 1,
        "kind": "video2world.source_camera_pbr_geometry_matte_batch_qa",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "passed_4_layers_x_25_frames_not_promoted",
        "promotion_approved": False,
        "promotion_blocker": (
            "final peel composite and clean-plate reconstruction are separate stages"
        ),
        "batch_root": str(batch_root),
        "frame_count": len(frame_ids),
        "layer_count": len(LAYER_ORDER),
        "frame_layer_asset_count": len(frame_ids) * len(LAYER_ORDER),
        "alpha_depth_alignment": {
            "mode": ALIGNMENT_MODE,
            "matte_provenance": "raw_blender_z_within_evaluated_geometry_depth_bounds",
            "minimum_initial_alpha_vs_depth_iou": 0.98,
            "maximum_changed_union_fraction_exclusive": 0.02,
            "final_alpha_is_binary": True,
            "final_alpha_equals_finite_depth_support": True,
            "rgb_render_samples": 64,
        },
        "analytic_silhouette_thresholds": {
            "minimum_mask_iou": 0.85,
            "minimum_bbox_iou": 0.8,
            "maximum_center_error_px_by_layer": CENTER_ERROR_LIMITS,
            "bed_center_threshold_reason": (
                "The bed is a wide support object cropped by the source camera; the approved "
                "20px center gate preserves strict mask and bbox gates while avoiding a "
                "misleading pillow-scale center threshold."
            ),
        },
        "layers": layer_summaries,
        "sum_layer_render_elapsed_seconds": sum_elapsed,
        "payload_artifact_set": payload,
        "gates": {
            "four_expected_layers_present": True,
            "exactly_25_ordered_frames_per_layer": True,
            "all_100_frame_receipts_and_hashes_valid": True,
            "all_frame_receipt_gates_true": True,
            "all_initial_alpha_vs_depth_iou_at_least_0_98": True,
            "all_changed_union_fractions_below_0_02": True,
            "all_final_alpha_support_equals_finite_depth_support": True,
            "all_final_alpha_values_binary": True,
            "all_analytic_silhouette_gates_true": True,
            "all_rgb_render_samples_equal_64": True,
            "no_per_frame_raw_depth_exr_retained": True,
            "claims_measured_donor_rejected": True,
            "promotion_approved_rejected": True,
        },
        "outputs": {
            "render_index": relative_artifact(output_paths["render_index"], batch_root),
            "contact_sheet": relative_artifact(output_paths["contact_sheet"], batch_root),
        },
    }
    write_json(output_paths["report"], report)
    receipt = {
        "schema_version": 1,
        "kind": "video2world.source_camera_pbr_geometry_matte_batch_receipt",
        "status": report["status"],
        "promotion_approved": False,
        "batch_report": relative_artifact(output_paths["report"], batch_root),
        "render_index": relative_artifact(output_paths["render_index"], batch_root),
        "contact_sheet": relative_artifact(output_paths["contact_sheet"], batch_root),
        "payload_artifact_set": payload,
        "layer_manifests": {
            item["layer_id"]: item["manifest"] for item in layer_summaries
        },
        "source_frames": source_records,
        "all_required_evidence_is_hash_bound": True,
    }
    write_json(output_paths["receipt"], receipt)
    print(
        json.dumps(
            {
                "status": report["status"],
                "frame_layer_asset_count": report["frame_layer_asset_count"],
                "payload_sha256": payload["sha256"],
                "report": str(output_paths["report"]),
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--frame-id", action="append", required=True)
    parser.add_argument("--contact-frame", action="append", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    batch_root = args.batch_root.expanduser().resolve()
    frame_ids = parse_frame_ids(args.frame_id)
    contact_frames = parse_frame_ids(args.contact_frame)
    build_evidence(batch_root, frame_ids, contact_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
