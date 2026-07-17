#!/usr/bin/env python3
"""Prefill object-removal masks with calibrated multi-view donor reprojection.

The script forward-projects unmasked donor RGB-D pixels through world-to-camera
calibration, applies a per-donor z-buffer, robustly fuses depth-consistent donor
samples, and writes ProPainter-compatible residual masks. Existing scene pixels
outside each removal mask are copied byte-for-byte at the RGB array level.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


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


def image_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def mask_bool(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) > 0


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


def parse_frame_ids(value: str | None) -> list[str] | None:
    if value is None:
        return None
    frame_ids = [Path(item.strip()).stem for item in value.split(",") if item.strip()]
    if not frame_ids:
        raise ValueError("frame-id list is empty")
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError("frame-id list contains duplicates")
    return frame_ids


def resolve_path(value: str, *, relative_to: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def resolve_frame_path(frames_dir: Path, frame_id: str) -> Path:
    matches = sorted(frames_dir.glob(f"{frame_id}.*"))
    if len(matches) != 1:
        raise ValueError(f"expected one RGB frame for {frame_id}, found {len(matches)}")
    return matches[0].resolve()


def resolve_depth_path(depth_dir: Path, frame_id: str) -> Path:
    matches = sorted(depth_dir.glob(f"{frame_id}.*"))
    if len(matches) != 1:
        raise ValueError(f"expected one depth array for {frame_id}, found {len(matches)}")
    if matches[0].suffix != ".npy":
        raise ValueError(f"depth must be a .npy array: {matches[0]}")
    return matches[0].resolve()


def load_camera_info(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError("camera info must be a JSON object")
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
            camera_id = str(frame_camera_ids.get(frame_id, ""))
            value = intrinsics.get(camera_id)
        elif len(intrinsics) == 1:
            value = next(iter(intrinsics.values()))
    if value is None:
        value = camera_info.get("intrinsic")
    if not isinstance(value, dict):
        raise ValueError(f"no intrinsic calibration for frame {frame_id}")
    required = ("fx", "fy", "cx", "cy", "w", "h")
    if not all(isinstance(value.get(key), int | float) for key in required):
        raise ValueError(f"intrinsic for {frame_id} is missing numeric {required}")
    return {key: float(value[key]) for key in required}


def world_to_camera(camera_info: dict[str, Any], frame_id: str) -> np.ndarray:
    value = camera_info["extrinsic"].get(frame_id)
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"invalid world-to-camera matrix for frame {frame_id}")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"invalid homogeneous row for frame {frame_id}")
    return matrix


def camera_center(matrix: np.ndarray) -> np.ndarray:
    return -(matrix[:3, :3].T @ matrix[:3, 3])


def load_mask_index(
    path: Path | None,
    *,
    label: str | None,
    min_score: float | None,
) -> dict[str, list[Path]]:
    if path is None:
        return {}
    value = read_json(path)
    if not isinstance(value, dict) or not isinstance(value.get("items"), list):
        raise ValueError("donor mask index must contain an items array")
    result: dict[str, list[Path]] = defaultdict(list)
    for item in value["items"]:
        if not isinstance(item, dict):
            raise ValueError("donor mask index items must be objects")
        if label is not None and item.get("label") != label:
            continue
        score = item.get("score")
        if min_score is not None and (
            not isinstance(score, int | float) or float(score) < min_score
        ):
            continue
        image = item.get("image")
        mask_path = item.get("mask_path")
        if not isinstance(image, str) or not isinstance(mask_path, str):
            raise ValueError("donor mask items require image and mask_path strings")
        resolved = resolve_path(mask_path, relative_to=path.parent)
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        result[Path(image).stem].append(resolved)
    return dict(result)


def union_masks(paths: list[Path], shape: tuple[int, int]) -> np.ndarray:
    union = np.zeros(shape, dtype=bool)
    for path in paths:
        mask = mask_bool(path)
        if mask.shape != shape:
            raise ValueError(f"mask {path} has shape {mask.shape}, expected {shape}")
        union |= mask
    return union


def load_cumulative_contract(
    *,
    input_manifest_path: Path,
    input_manifest: dict[str, Any],
    donor_mask_index_path: Path | None,
    donor_mask_label: str | None,
) -> dict[str, Any] | None:
    if input_manifest.get("kind") != "video2world.cumulative_removal_manifest":
        return None
    if input_manifest.get("status") != "ready_for_measured_multiview_prefill":
        raise ValueError("cumulative removal manifest is not ready for measured prefill")
    gates = input_manifest.get("gates")
    required_gates = (
        "strict_front_to_back_order",
        "input_is_immediately_previous_round",
        "all_previous_masks_are_accumulated",
        "all_removed_objects_are_excluded_from_donors",
        "donors_are_original_observed_rgb",
        "unresolved_residual_is_never_donor_evidence",
    )
    if not isinstance(gates, dict) or any(gates.get(key) is not True for key in required_gates):
        raise ValueError("cumulative removal manifest gates did not all pass")
    donor_contract = input_manifest.get("donor_contract")
    if not isinstance(donor_contract, dict):
        raise ValueError("cumulative removal manifest has no donor contract")
    if donor_contract.get("role") != "original_observed_rgb_only":
        raise ValueError("cumulative donor role must be original_observed_rgb_only")
    if donor_contract.get("generated_or_prefilled_rgb_as_donor") is not False:
        raise ValueError("generated or prefilled RGB is not allowed as cumulative donor evidence")
    if donor_contract.get("unresolved_residual_as_donor") is not False:
        raise ValueError("unresolved residual is not allowed as cumulative donor evidence")
    if donor_mask_index_path is None:
        raise ValueError("cumulative measured prefill requires its donor exclusion index")
    expected_index_path = resolve_path(
        str(donor_contract.get("exclusion_index")),
        relative_to=input_manifest_path.parent,
    )
    if expected_index_path != donor_mask_index_path:
        raise ValueError("donor exclusion index does not match cumulative manifest")
    expected_index_sha = donor_contract.get("exclusion_index_sha256")
    if (
        not isinstance(expected_index_sha, str)
        or sha256_file(expected_index_path) != expected_index_sha
    ):
        raise ValueError("cumulative donor exclusion index SHA-256 mismatch")
    exclusion_label = donor_contract.get("exclusion_label")
    if donor_mask_label != exclusion_label:
        raise ValueError("donor mask label does not match cumulative exclusion label")

    records = input_manifest.get("frame_records")
    if not isinstance(records, list) or not records:
        raise ValueError("cumulative removal manifest has no frame records")
    expected_masks: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("frame_id"), str):
            raise ValueError("invalid cumulative frame record")
        frame_id = record["frame_id"]
        mask_path = resolve_path(
            str(record.get("donor_exclusion_mask")),
            relative_to=input_manifest_path.parent,
        )
        expected_sha = record.get("donor_exclusion_mask_sha256")
        if not isinstance(expected_sha, str) or sha256_file(mask_path) != expected_sha:
            raise ValueError(f"cumulative donor exclusion mask mismatch for {frame_id}")
        if expected_sha != record.get("union_mask_sha256"):
            raise ValueError(f"donor exclusion is not the cumulative removal mask for {frame_id}")
        expected_masks[frame_id] = {"path": mask_path, "sha256": expected_sha}

    index = read_json(expected_index_path)
    if not isinstance(index, dict) or not isinstance(index.get("items"), list):
        raise ValueError("invalid cumulative donor exclusion index")
    indexed_masks: dict[str, dict[str, Any]] = {}
    for item in index["items"]:
        if not isinstance(item, dict) or item.get("label") != exclusion_label:
            raise ValueError("cumulative donor index has an invalid item")
        frame_id = item.get("frame_id") or Path(str(item.get("image"))).stem
        if not isinstance(frame_id, str) or frame_id in indexed_masks:
            raise ValueError("cumulative donor index repeats or omits a frame id")
        mask_path = resolve_path(str(item.get("mask_path")), relative_to=expected_index_path.parent)
        mask_sha = sha256_file(mask_path)
        if item.get("mask_sha256") != mask_sha:
            raise ValueError(f"cumulative donor index mask SHA-256 mismatch for {frame_id}")
        indexed_masks[frame_id] = {"path": mask_path, "sha256": mask_sha}
    if set(indexed_masks) != set(expected_masks):
        raise ValueError("cumulative donor index and manifest cover different frames")
    for frame_id, expected in expected_masks.items():
        indexed = indexed_masks[frame_id]
        if indexed["path"] != expected["path"] or indexed["sha256"] != expected["sha256"]:
            raise ValueError(f"cumulative donor index does not match frame {frame_id}")

    assets = donor_contract.get("assets")
    if not isinstance(assets, dict) or set(assets) != set(expected_masks):
        raise ValueError("cumulative original-donor assets do not cover the target frames")
    return {
        "round_index": input_manifest.get("round_index"),
        "removed_object_ids": input_manifest.get("removed_object_ids"),
        "manifest_sha256": sha256_file(input_manifest_path),
        "donor_assets": assets,
        "expected_masks": expected_masks,
        "lineage": input_manifest.get("lineage"),
    }


def donor_projection(
    *,
    donor_rgb: np.ndarray,
    donor_depth: np.ndarray,
    donor_exclusion: np.ndarray,
    donor_intrinsic: dict[str, float],
    donor_world_to_camera: np.ndarray,
    target_intrinsic: dict[str, float],
    target_world_to_camera: np.ndarray,
    target_width: int,
    target_height: int,
    sample_stride: int,
    splat_radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = donor_depth.shape
    y_grid, x_grid = np.mgrid[0:height:sample_stride, 0:width:sample_stride]
    x = x_grid.reshape(-1)
    y = y_grid.reshape(-1)
    depth = donor_depth[y, x].astype(np.float64, copy=False)
    valid = np.isfinite(depth) & (depth > 0) & ~donor_exclusion[y, x]
    if not np.any(valid):
        empty_int = np.empty(0, dtype=np.int64)
        return empty_int, np.empty(0, dtype=np.float64), np.empty((0, 3), dtype=np.uint8)

    x = x[valid]
    y = y[valid]
    depth = depth[valid]
    colors = donor_rgb[y, x]
    camera_points = np.stack(
        [
            (x - donor_intrinsic["cx"]) * depth / donor_intrinsic["fx"],
            (y - donor_intrinsic["cy"]) * depth / donor_intrinsic["fy"],
            depth,
        ],
        axis=1,
    )
    donor_rotation = donor_world_to_camera[:3, :3]
    donor_translation = donor_world_to_camera[:3, 3]
    world_points = (camera_points - donor_translation) @ donor_rotation
    target_camera = world_points @ target_world_to_camera[:3, :3].T + target_world_to_camera[:3, 3]
    target_depth = target_camera[:, 2]
    valid = np.isfinite(target_depth) & (target_depth > 1e-6)
    if not np.any(valid):
        empty_int = np.empty(0, dtype=np.int64)
        return empty_int, np.empty(0, dtype=np.float64), np.empty((0, 3), dtype=np.uint8)
    target_camera = target_camera[valid]
    target_depth = target_depth[valid]
    colors = colors[valid]
    projected_x = np.rint(
        target_intrinsic["fx"] * target_camera[:, 0] / target_depth + target_intrinsic["cx"]
    ).astype(np.int64)
    projected_y = np.rint(
        target_intrinsic["fy"] * target_camera[:, 1] / target_depth + target_intrinsic["cy"]
    ).astype(np.int64)

    flat_parts: list[np.ndarray] = []
    depth_parts: list[np.ndarray] = []
    color_parts: list[np.ndarray] = []
    for offset_y in range(-splat_radius, splat_radius + 1):
        for offset_x in range(-splat_radius, splat_radius + 1):
            target_x = projected_x + offset_x
            target_y = projected_y + offset_y
            inside = (
                (target_x >= 0)
                & (target_x < target_width)
                & (target_y >= 0)
                & (target_y < target_height)
            )
            if not np.any(inside):
                continue
            flat_parts.append(target_y[inside] * target_width + target_x[inside])
            depth_parts.append(target_depth[inside])
            color_parts.append(colors[inside])
    if not flat_parts:
        empty_int = np.empty(0, dtype=np.int64)
        return empty_int, np.empty(0, dtype=np.float64), np.empty((0, 3), dtype=np.uint8)

    flat = np.concatenate(flat_parts)
    projected_depth = np.concatenate(depth_parts)
    projected_color = np.concatenate(color_parts)
    order = np.lexsort((projected_depth, flat))
    flat = flat[order]
    projected_depth = projected_depth[order]
    projected_color = projected_color[order]
    first = np.empty(len(flat), dtype=bool)
    first[0] = True
    first[1:] = flat[1:] != flat[:-1]
    return flat[first], projected_depth[first], projected_color[first]


def robust_fuse(
    depths: np.ndarray,
    colors: np.ndarray,
    *,
    absolute_depth_tolerance: float,
    relative_depth_tolerance: float,
    min_support: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    median_depth = np.ma.median(np.ma.masked_invalid(depths), axis=0).filled(np.nan)
    tolerance = np.maximum(
        absolute_depth_tolerance,
        np.abs(median_depth) * relative_depth_tolerance,
    )
    inlier = np.isfinite(depths) & (np.abs(depths - median_depth[None, :]) <= tolerance[None, :])
    support = inlier.sum(axis=0).astype(np.uint16)
    accepted = support >= min_support
    fused_color = np.zeros((depths.shape[1], 3), dtype=np.uint8)
    for channel in range(3):
        values = np.ma.masked_where(~inlier, colors[:, :, channel])
        median_color = np.ma.median(values, axis=0).filled(0.0)
        fused_color[:, channel] = np.clip(np.rint(median_color), 0, 255).astype(np.uint8)
    fused_depth = (
        np.ma.median(np.ma.masked_where(~inlier, depths), axis=0)
        .filled(np.nan)
        .astype(np.float32)
    )
    fused_depth[~accepted] = np.nan
    return fused_color, fused_depth, support, accepted


def make_contact_sheet(records: list[dict[str, Any]], path: Path, samples: int) -> None:
    sample_count = min(samples, len(records))
    if sample_count <= 0:
        raise ValueError("contact sheet needs at least one record")
    indices = np.linspace(0, len(records) - 1, sample_count, dtype=int)
    cell_width, cell_height, label_height = 360, 203, 24
    sheet = Image.new(
        "RGB",
        (cell_width * 4, (cell_height + label_height) * sample_count),
        "#111111",
    )
    draw = ImageDraw.Draw(sheet)
    for row, index in enumerate(indices):
        record = records[int(index)]
        source = Image.open(record["source_frame"]).convert("RGB")
        output = Image.open(record["prefill_frame"]).convert("RGB")
        removal = Image.open(record["removal_mask"]).convert("L")
        residual = Image.open(record["residual_mask"]).convert("L")
        removal_overlay = source.copy()
        removal_overlay.paste(Image.new("RGB", source.size, (255, 0, 170)), mask=removal)
        residual_overlay = output.copy()
        residual_overlay.paste(Image.new("RGB", output.size, (255, 80, 0)), mask=residual)
        images = (source, removal_overlay, output, residual_overlay)
        titles = (
            f"source {record['frame_id']}",
            "removal mask",
            f"donor prefill {record['coverage_fraction']:.1%}",
            "unresolved residual",
        )
        y = row * (cell_height + label_height)
        for column, (image, title) in enumerate(zip(images, titles, strict=True)):
            image.thumbnail((cell_width, cell_height), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (cell_width, cell_height), "#202020")
            offset = (
                (cell_width - image.width) // 2,
                (cell_height - image.height) // 2,
            )
            canvas.paste(image, offset)
            x = column * cell_width
            sheet.paste(canvas, (x, y + label_height))
            draw.text((x + 8, y + 6), title, fill="#f4f4f4")
        for image in images:
            image.close()
        removal.close()
        residual.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def write_output_receipt(output_root: Path) -> dict[str, Any]:
    report_path = output_root / "multiview_prefill_report.json"
    contact_sheet_path = output_root / "multiview_prefill_contact_sheet.png"
    report = read_json(report_path)
    if not isinstance(report, dict) or not isinstance(report.get("frame_records"), list):
        raise ValueError("cannot receipt an invalid multi-view prefill report")
    provenance = report.get("pixel_provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("generated_pixels") != 0
        or provenance.get("propainter_pixels") != 0
    ):
        raise ValueError(
            "cannot receipt measured prefill without zero generated and ProPainter pixels"
        )
    receipt = {
        "schema_version": 2,
        "kind": "video2world.multiview_prefill_receipt",
        "report": report_path.name,
        "report_sha256": sha256_file(report_path),
        "contact_sheet": contact_sheet_path.name,
        "contact_sheet_sha256": sha256_file(contact_sheet_path),
        "frame_count": len(report["frame_records"]),
        "frame_artifacts_are_hashed_in_report": True,
        "fused_metric_depth_artifacts_are_hashed_in_report": True,
        "measured_provenance_excludes_generated_pixels": True,
        "measured_provenance_excludes_propainter_pixels": True,
    }
    write_json(output_root / "multiview_prefill_receipt.json", receipt)
    return receipt


def build_prefill(args: argparse.Namespace) -> dict[str, Any]:
    input_manifest_path = args.input_manifest.expanduser().resolve()
    camera_info_path = args.camera_info.expanduser().resolve()
    donor_frames_dir = args.donor_frames_dir.expanduser().resolve()
    depth_dir = args.depth_dir.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    donor_mask_index_path = (
        args.donor_mask_index.expanduser().resolve() if args.donor_mask_index else None
    )
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_root}")
    if args.sample_stride < 1 or args.splat_radius < 0 or args.min_support < 1:
        raise ValueError("sample_stride/min_support must be positive and splat_radius non-negative")
    if args.absolute_depth_tolerance < 0 or args.relative_depth_tolerance < 0:
        raise ValueError("depth tolerances must be non-negative")

    input_manifest = read_json(input_manifest_path)
    if not isinstance(input_manifest, dict) or not isinstance(
        input_manifest.get("frame_records"), list
    ):
        raise ValueError("input manifest must contain frame_records")
    cumulative_contract = load_cumulative_contract(
        input_manifest_path=input_manifest_path,
        input_manifest=input_manifest,
        donor_mask_index_path=donor_mask_index_path,
        donor_mask_label=args.donor_mask_label,
    )
    camera_info = load_camera_info(camera_info_path)
    requested_targets = parse_frame_ids(args.target_frame_ids)
    target_records = [
        record
        for record in input_manifest["frame_records"]
        if requested_targets is None or str(record.get("frame_id")) in requested_targets
    ]
    selected_target_ids = {str(record.get("frame_id")) for record in target_records}
    if requested_targets is not None and selected_target_ids != set(requested_targets):
        raise ValueError("one or more requested target frames are absent from the input manifest")
    if not target_records:
        raise ValueError("no target records selected")

    donor_masks = load_mask_index(
        donor_mask_index_path,
        label=args.donor_mask_label,
        min_score=args.donor_mask_min_score,
    )
    donor_ids = parse_frame_ids(args.donor_frame_ids)
    if donor_ids is None:
        donor_ids = sorted(camera_info["extrinsic"])
    donor_assets: dict[str, dict[str, Any]] = {}
    donor_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for frame_id in donor_ids:
        frame_path = resolve_frame_path(donor_frames_dir, frame_id)
        depth_path = resolve_depth_path(depth_dir, frame_id)
        rgb = image_rgb(frame_path)
        depth = np.load(depth_path)
        if depth.shape != rgb.shape[:2]:
            raise ValueError(f"depth/RGB shape mismatch for donor {frame_id}")
        exclusion = union_masks(donor_masks.get(frame_id, []), depth.shape)
        exclusion = binary_dilate(exclusion, args.donor_mask_dilation)
        frame_sha = sha256_file(frame_path)
        if cumulative_contract is not None:
            expected_asset = cumulative_contract["donor_assets"].get(frame_id)
            if not isinstance(expected_asset, dict) or expected_asset.get("role") != (
                "original_observed_rgb_only"
            ):
                raise ValueError(f"cumulative donor {frame_id} has no original-RGB contract")
            if expected_asset.get("sha256") != frame_sha:
                raise ValueError(
                    f"donor {frame_id} is not the contracted original observed RGB"
                )
            expected_mask = cumulative_contract["expected_masks"][frame_id]
            actual_masks = donor_masks.get(frame_id, [])
            if len(actual_masks) != 1 or actual_masks[0] != expected_mask["path"]:
                raise ValueError(
                    f"donor {frame_id} is not paired with its cumulative exclusion mask"
                )
        donor_cache[frame_id] = (rgb, depth, exclusion)
        donor_assets[frame_id] = {
            "frame": str(frame_path),
            "frame_sha256": frame_sha,
            "depth": str(depth_path),
            "depth_sha256": sha256_file(depth_path),
            "depth_role": "reprojection_evidence_only",
            "exclusion_masks": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in donor_masks.get(frame_id, [])
            ],
            "excluded_pixels_after_dilation": int(exclusion.sum()),
        }

    frame_output_dir = output_root / "frames"
    residual_output_dir = output_root / "masks"
    support_output_dir = output_root / "support"
    depth_output_dir = output_root / "depth"
    frame_output_dir.mkdir(parents=True, exist_ok=True)
    residual_output_dir.mkdir(parents=True, exist_ok=True)
    support_output_dir.mkdir(parents=True, exist_ok=True)
    depth_output_dir.mkdir(parents=True, exist_ok=True)
    output_records: list[dict[str, Any]] = []

    for output_index, input_record in enumerate(target_records):
        frame_id = str(input_record["frame_id"])
        source_path = resolve_path(
            str(input_record["source_frame"]), relative_to=input_manifest_path.parent
        )
        mask_path = resolve_path(
            str(input_record["union_mask"]), relative_to=input_manifest_path.parent
        )
        source = image_rgb(source_path)
        removal_mask = mask_bool(mask_path)
        height, width = source.shape[:2]
        if removal_mask.shape != (height, width):
            raise ValueError(f"source/removal-mask shape mismatch for target {frame_id}")
        target_intrinsic = frame_intrinsic(camera_info, frame_id)
        if (int(target_intrinsic["w"]), int(target_intrinsic["h"])) != (width, height):
            raise ValueError(f"camera/image shape mismatch for target {frame_id}")
        target_w2c = world_to_camera(camera_info, frame_id)
        mask_flat = np.flatnonzero(removal_mask.reshape(-1))
        pixel_to_mask = np.full(height * width, -1, dtype=np.int64)
        pixel_to_mask[mask_flat] = np.arange(len(mask_flat), dtype=np.int64)
        candidate_depths = np.full((len(donor_ids), len(mask_flat)), np.nan, dtype=np.float32)
        candidate_colors = np.full((len(donor_ids), len(mask_flat), 3), np.nan, dtype=np.float32)
        target_center = camera_center(target_w2c)
        used_donors: list[dict[str, Any]] = []

        for donor_index, donor_id in enumerate(donor_ids):
            if args.exclude_same_frame and donor_id == frame_id:
                continue
            donor_rgb, donor_depth, donor_exclusion = donor_cache[donor_id]
            donor_intrinsic = frame_intrinsic(camera_info, donor_id)
            donor_w2c = world_to_camera(camera_info, donor_id)
            flat, projected_depth, projected_color = donor_projection(
                donor_rgb=donor_rgb,
                donor_depth=donor_depth,
                donor_exclusion=donor_exclusion,
                donor_intrinsic=donor_intrinsic,
                donor_world_to_camera=donor_w2c,
                target_intrinsic=target_intrinsic,
                target_world_to_camera=target_w2c,
                target_width=width,
                target_height=height,
                sample_stride=args.sample_stride,
                splat_radius=args.splat_radius,
            )
            indices = pixel_to_mask[flat]
            inside_removal = indices >= 0
            indices = indices[inside_removal]
            if len(indices):
                candidate_depths[donor_index, indices] = projected_depth[inside_removal]
                candidate_colors[donor_index, indices] = projected_color[inside_removal]
            used_donors.append(
                {
                    "frame_id": donor_id,
                    "camera_baseline": float(
                        np.linalg.norm(camera_center(donor_w2c) - target_center)
                    ),
                    "projected_mask_pixels": len(np.unique(indices)),
                    "excluded_object_pixels": int(donor_exclusion.sum()),
                }
            )

        fused_color, fused_depth, support, accepted = robust_fuse(
            candidate_depths,
            candidate_colors,
            absolute_depth_tolerance=args.absolute_depth_tolerance,
            relative_depth_tolerance=args.relative_depth_tolerance,
            min_support=args.min_support,
        )
        output = source.copy()
        output_flat = output.reshape(-1, 3)
        output_flat[mask_flat[accepted]] = fused_color[accepted]
        residual_flat = np.zeros(height * width, dtype=bool)
        residual_flat[mask_flat[~accepted]] = True
        residual = residual_flat.reshape(height, width)
        support_image = np.zeros(height * width, dtype=np.uint16)
        support_image[mask_flat] = support
        support_image = support_image.reshape(height, width)
        measured_depth = np.full((height, width), np.nan, dtype=np.float32)
        measured_depth.reshape(-1)[mask_flat[accepted]] = fused_depth[accepted]
        valid_measured_depth = np.isfinite(measured_depth) & (measured_depth > 0)
        expected_measured = removal_mask & ~residual
        if not np.array_equal(valid_measured_depth, expected_measured):
            raise ValueError(f"fused measured depth support mismatch for target {frame_id}")

        name = f"{output_index:04d}.png"
        output_path = frame_output_dir / name
        residual_path = residual_output_dir / name
        support_path = support_output_dir / name
        depth_path = depth_output_dir / f"{output_index:04d}.npy"
        Image.fromarray(output).save(output_path)
        Image.fromarray(residual.astype(np.uint8) * 255).save(residual_path)
        max_support = max(int(support.max(initial=0)), 1)
        Image.fromarray(
            np.clip(np.rint(support_image * (255.0 / max_support)), 0, 255).astype(np.uint8),
        ).save(support_path)
        np.save(depth_path, measured_depth, allow_pickle=False)
        outside_exact = bool(np.array_equal(output[~removal_mask], source[~removal_mask]))
        coverage_fraction = float(accepted.mean()) if len(accepted) else 0.0
        output_records.append(
            {
                "sequence_index": output_index,
                "frame_id": frame_id,
                "source_frame": str(source_path),
                "source_frame_sha256": sha256_file(source_path),
                "removal_mask": str(mask_path),
                "removal_mask_sha256": sha256_file(mask_path),
                "removal_mask_pixels": int(removal_mask.sum()),
                "prefill_frame": str(output_path),
                "prefill_frame_sha256": sha256_file(output_path),
                "residual_mask": str(residual_path),
                "residual_mask_sha256": sha256_file(residual_path),
                "residual_mask_pixels": int(residual.sum()),
                "support_visualization": str(support_path),
                "support_visualization_sha256": sha256_file(support_path),
                "measured_depth": str(depth_path),
                "measured_depth_sha256": sha256_file(depth_path),
                "measured_depth_role": "fused_multiview_measured_depth",
                "measured_depth_unit": "meter",
                "measured_depth_semantics": "target_camera_positive_z_axis",
                "measured_depth_valid_pixels": int(valid_measured_depth.sum()),
                "measured_depth_validity_matches_measured_rgb": True,
                "covered_pixels": int(accepted.sum()),
                "coverage_fraction": coverage_fraction,
                "support_min_accepted": int(support[accepted].min()) if np.any(accepted) else None,
                "support_median_accepted": (
                    float(np.median(support[accepted])) if np.any(accepted) else None
                ),
                "support_max": int(support.max(initial=0)),
                "outside_removal_mask_rgb_exact": outside_exact,
                "donors": used_donors,
            }
        )

    outside_exact_passed = all(
        record["outside_removal_mask_rgb_exact"] for record in output_records
    )
    total_removal_pixels = sum(record["removal_mask_pixels"] for record in output_records)
    total_measured_pixels = sum(record["covered_pixels"] for record in output_records)
    total_unresolved_pixels = sum(record["residual_mask_pixels"] for record in output_records)
    report = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 - mil8 uses Python 3.10
        "purpose": "calibrated multi-view donor prefill before residual video inpainting",
        "status": "technical_passed" if outside_exact_passed else "technical_failed",
        "promotion_approved": False,
        "promotion_blocker": "semantic texture continuity review is required",
        "input_manifest": str(input_manifest_path),
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "camera_info": str(camera_info_path),
        "camera_info_sha256": sha256_file(camera_info_path),
        "camera_extrinsic_type": "world_to_camera",
        "cumulative_removal_contract": (
            {
                "enforced": True,
                "round_index": cumulative_contract["round_index"],
                "removed_object_ids": cumulative_contract["removed_object_ids"],
                "manifest_sha256": cumulative_contract["manifest_sha256"],
                "lineage": cumulative_contract["lineage"],
                "donors_are_original_observed_rgb": True,
                "all_removed_objects_excluded_from_donors": True,
                "unresolved_residual_as_donor": False,
            }
            if cumulative_contract is not None
            else {"enforced": False}
        ),
        "donor_frames_dir": str(donor_frames_dir),
        "depth_dir": str(depth_dir),
        "donor_mask_index": str(donor_mask_index_path) if donor_mask_index_path else None,
        "donor_mask_index_sha256": (
            sha256_file(donor_mask_index_path) if donor_mask_index_path else None
        ),
        "config": {
            "target_frame_ids": [record["frame_id"] for record in output_records],
            "donor_frame_ids": donor_ids,
            "donor_mask_label": args.donor_mask_label,
            "donor_mask_min_score": args.donor_mask_min_score,
            "donor_mask_dilation": args.donor_mask_dilation,
            "exclude_same_frame": args.exclude_same_frame,
            "sample_stride": args.sample_stride,
            "splat_radius": args.splat_radius,
            "min_support": args.min_support,
            "absolute_depth_tolerance": args.absolute_depth_tolerance,
            "relative_depth_tolerance": args.relative_depth_tolerance,
            "fusion": "per_donor_nearest_z_then_cross_donor_median_depth_cluster",
        },
        "gates": {
            "outside_removal_mask_rgb_exact": outside_exact_passed,
            "all_residual_masks_subset_of_removal_masks": True,
            "cumulative_front_to_back_contract_enforced": cumulative_contract is not None,
            "semantic_texture_continuity": "pending_human_or_vlm_review",
            "new_depth_normal_estimation": "required_before_scene_reconstruction",
            "fused_measured_metric_depth_materialized": True,
        },
        "pixel_provenance": {
            "denominator": "cumulative removal-mask pixels across selected frames",
            "cumulative_removal_pixels": total_removal_pixels,
            "measured_multiview_pixels": total_measured_pixels,
            "measured_multiview_fraction": (
                total_measured_pixels / total_removal_pixels if total_removal_pixels else 0.0
            ),
            "unresolved_unobserved_pixels": total_unresolved_pixels,
            "unresolved_unobserved_fraction": (
                total_unresolved_pixels / total_removal_pixels if total_removal_pixels else 0.0
            ),
            "generated_pixels": 0,
            "propainter_pixels": 0,
            "classes_are_mutually_exclusive": True,
            "unresolved_pixels_are_not_valid_donor_or_geometry_evidence": True,
        },
        "donor_assets": donor_assets,
        "frame_records": output_records,
        "next_stage": {
            "residual_handling": (
                "keep masks/*.png as explicitly unresolved; recover with a cross-view-consistent "
                "background atlas or scene geometry before reconstruction"
                if cumulative_contract is not None
                else "run ProPainter only on masks/*.png residual holes for visual reinspection"
            ),
            "geometry": (
                "rerun depth and normal estimation on accepted clean plates before PGSR/TSDF; "
                "the fused donor depth is sparse measured evidence only and must not be treated "
                "as a dense final clean-plate depth map"
            ),
        },
        "limitations": [
            "Forward splatting cannot reveal a surface that is absent from every donor view.",
            "DA3 depth noise can misregister high-frequency bedding texture across views.",
            (
                "Technical coverage and exact outside-mask preservation do not certify "
                "semantic quality."
            ),
        ],
        "output_receipt": "multiview_prefill_receipt.json",
    }
    report_path = output_root / "multiview_prefill_report.json"
    contact_sheet_path = output_root / "multiview_prefill_contact_sheet.png"
    write_json(report_path, report)
    make_contact_sheet(
        output_records,
        contact_sheet_path,
        args.contact_sheet_samples,
    )
    write_output_receipt(output_root)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--camera-info", type=Path, required=True)
    parser.add_argument("--donor-frames-dir", type=Path, required=True)
    parser.add_argument("--depth-dir", type=Path, required=True)
    parser.add_argument("--donor-mask-index", type=Path)
    parser.add_argument("--donor-mask-label", default="pillow")
    parser.add_argument("--donor-mask-min-score", type=float)
    parser.add_argument("--donor-mask-dilation", type=int, default=4)
    parser.add_argument("--target-frame-ids")
    parser.add_argument("--donor-frame-ids")
    parser.add_argument("--exclude-same-frame", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument("--splat-radius", type=int, default=1)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--absolute-depth-tolerance", type=float, default=0.08)
    parser.add_argument("--relative-depth-tolerance", type=float, default=0.01)
    parser.add_argument("--contact-sheet-samples", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = build_prefill(args)
    summary = {
        "status": report["status"],
        "promotion_approved": report["promotion_approved"],
        "frame_count": len(report["frame_records"]),
        "coverage": {
            record["frame_id"]: record["coverage_fraction"] for record in report["frame_records"]
        },
        "report": str(args.output / "multiview_prefill_report.json"),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
