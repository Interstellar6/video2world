#!/usr/bin/env python3
"""Project a 3D object anchor into depth-visible per-frame supplement masks.

Every output pixel must be supported by a projected anchor depth and a finite
scene-depth sample within the configured tolerance. Splatting propagates anchor
depth into a small neighborhood and rechecks visibility there. Closing is
bounded by a larger depth-consistent anchor envelope; this tool never dilates a
SAM mask and never assigns pixels without 3D/depth evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData

MANIFEST_KIND = "video2world.object_anchor_projection_mask_manifest"
RECEIPT_KIND = "video2world.object_anchor_projection_mask_receipt"


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


def digest_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_frame_ids(value: str | None, camera_info: dict[str, Any]) -> list[str]:
    if value is None:
        result = sorted(camera_info["extrinsic"])
    else:
        result = [Path(item.strip()).stem for item in value.split(",") if item.strip()]
    if not result or len(result) != len(set(result)):
        raise ValueError("frame ids must be non-empty and unique")
    unknown = [frame_id for frame_id in result if frame_id not in camera_info["extrinsic"]]
    if unknown:
        raise ValueError(f"camera_info has no extrinsic for frames: {unknown}")
    return result


def load_anchor(path: Path, *, point_stride: int) -> tuple[np.ndarray, int]:
    ply = PlyData.read(path)
    try:
        vertex = ply["vertex"].data
    except KeyError as error:
        raise ValueError("anchor PLY has no vertex element") from error
    names = set(vertex.dtype.names or ())
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError("anchor PLY must contain x/y/z")
    points = np.column_stack([vertex[axis] for axis in ("x", "y", "z")]).astype(
        np.float64, copy=False
    )
    if not len(points) or not np.isfinite(points).all():
        raise ValueError("anchor PLY must contain finite points")
    return points[::point_stride], len(points)


def load_camera_info(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict) or value.get("extrinsic_type") != "world_to_camera":
        raise ValueError("camera_info.extrinsic_type must be world_to_camera")
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
    if result["fx"] <= 0 or result["fy"] <= 0:
        raise ValueError(f"intrinsic for {frame_id} has invalid focal length")
    if not result["w"].is_integer() or not result["h"].is_integer():
        raise ValueError(f"intrinsic for {frame_id} width/height must be integers")
    return result


def world_to_camera(
    camera_info: dict[str, Any], frame_id: str, *, maximum_orthonormal_error: float
) -> tuple[np.ndarray, float]:
    matrix = np.asarray(camera_info["extrinsic"].get(frame_id), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"invalid world-to-camera matrix for frame {frame_id}")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"invalid homogeneous row for frame {frame_id}")
    rotation = matrix[:3, :3]
    error = float(np.max(np.abs(rotation @ rotation.T - np.eye(3))))
    determinant = float(np.linalg.det(rotation))
    if error > maximum_orthonormal_error or abs(determinant - 1) > (
        maximum_orthonormal_error
    ):
        raise ValueError(f"frame {frame_id} rotation is not orthonormal")
    return matrix, error


def resolve_frame_path(directory: Path, frame_id: str, *, suffix: str | None = None) -> Path:
    if suffix is not None:
        path = directory / f"{frame_id}{suffix}"
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.resolve()
    matches = sorted(path for path in directory.glob(f"{frame_id}.*") if path.is_file())
    if len(matches) != 1:
        raise ValueError(f"expected one frame for {frame_id}, found {len(matches)}")
    return matches[0].resolve()


def project_anchor_depth(
    points: np.ndarray,
    matrix: np.ndarray,
    intrinsic: dict[str, float],
) -> tuple[np.ndarray, dict[str, int]]:
    width = int(intrinsic["w"])
    height = int(intrinsic["h"])
    camera = np.einsum("ni,ji->nj", points, matrix[:3, :3], optimize=False)
    camera += matrix[:3, 3]
    depth = camera[:, 2]
    positive = np.isfinite(depth) & (depth > 1e-6)
    x = np.full(len(points), -1, dtype=np.int64)
    y = np.full(len(points), -1, dtype=np.int64)
    x[positive] = np.rint(
        intrinsic["fx"] * camera[positive, 0] / depth[positive] + intrinsic["cx"]
    ).astype(np.int64)
    y[positive] = np.rint(
        intrinsic["fy"] * camera[positive, 1] / depth[positive] + intrinsic["cy"]
    ).astype(np.int64)
    inside = positive & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    result = np.full(height * width, np.inf, dtype=np.float64)
    if np.any(inside):
        flat = y[inside] * width + x[inside]
        np.minimum.at(result, flat, depth[inside])
    return result.reshape(height, width), {
        "anchor_points": len(points),
        "positive_camera_depth_points": int(positive.sum()),
        "inside_image_points": int(inside.sum()),
        "projected_unique_pixels": int(np.isfinite(result).sum()),
    }


def depth_consistent(
    anchor_depth: np.ndarray,
    scene_depth: np.ndarray,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    projected = np.isfinite(anchor_depth)
    observed = np.isfinite(scene_depth) & (scene_depth > 0)
    tolerance = np.maximum(absolute_tolerance, np.abs(scene_depth) * relative_tolerance)
    delta = anchor_depth - scene_depth
    visible = projected & observed & (np.abs(delta) <= tolerance)
    diagnostics = {
        "missing_scene_depth": projected & ~observed,
        "occluded_behind_scene": projected & observed & (delta > tolerance),
        "front_depth_mismatch": projected & observed & (delta < -tolerance),
    }
    return visible, diagnostics


def shifted_slices(
    height: int, width: int, offset_y: int, offset_x: int
) -> tuple[tuple[slice, slice], tuple[slice, slice]]:
    source_y = slice(max(0, -offset_y), min(height, height - offset_y))
    source_x = slice(max(0, -offset_x), min(width, width - offset_x))
    target_y = slice(max(0, offset_y), min(height, height + offset_y))
    target_x = slice(max(0, offset_x), min(width, width + offset_x))
    return (source_y, source_x), (target_y, target_x)


def depth_aware_splat(
    anchor_depth: np.ndarray,
    scene_depth: np.ndarray,
    *,
    radius: int,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = anchor_depth.shape
    propagated = np.full_like(anchor_depth, np.inf, dtype=np.float64)
    for offset_y in range(-radius, radius + 1):
        for offset_x in range(-radius, radius + 1):
            source, target = shifted_slices(height, width, offset_y, offset_x)
            np.minimum(propagated[target], anchor_depth[source], out=propagated[target])
    visible_raw, _ = depth_consistent(
        anchor_depth,
        scene_depth,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    propagated_visible, _ = depth_consistent(
        propagated,
        scene_depth,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    return visible_raw | propagated_visible, propagated


def binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    result = mask.astype(bool, copy=True)
    for _ in range(radius):
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


def binary_erode(mask: np.ndarray, radius: int) -> np.ndarray:
    result = mask.astype(bool, copy=True)
    for _ in range(radius):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        result = (
            padded[1:-1, 1:-1]
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
            & padded[:-2, :-2]
            & padded[:-2, 2:]
            & padded[2:, :-2]
            & padded[2:, 2:]
        )
    return result


def depth_bounded_close(
    support: np.ndarray,
    anchor_depth: np.ndarray,
    scene_depth: np.ndarray,
    *,
    splat_radius: int,
    closing_radius: int,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    if closing_radius == 0:
        return support.copy(), support.copy()
    closed = binary_erode(binary_dilate(support, closing_radius), closing_radius)
    envelope, _ = depth_aware_splat(
        anchor_depth,
        scene_depth,
        radius=splat_radius + closing_radius,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    return support | (closed & envelope), support | envelope


def mask_bbox(mask: np.ndarray) -> list[int] | None:
    rows, columns = np.nonzero(mask)
    if not len(rows):
        return None
    return [
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    ]


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def make_contact_sheet(
    records: list[dict[str, Any]],
    output: Path,
    *,
    samples: int,
    selected_frame_ids: list[str] | None,
    relative_to: Path,
) -> None:
    records_with_rgb = [record for record in records if record.get("source_rgb")]
    by_frame = {record["frame_id"]: record for record in records_with_rgb}
    if selected_frame_ids is not None:
        unknown = [frame_id for frame_id in selected_frame_ids if frame_id not in by_frame]
        if unknown:
            raise ValueError(f"contact sheet frame ids have no RGB inputs: {unknown}")
        selected_records = [by_frame[frame_id] for frame_id in selected_frame_ids]
    else:
        sample_count = min(samples, len(records_with_rgb))
        indices = np.linspace(0, len(records_with_rgb) - 1, sample_count, dtype=int)
        selected_records = [records_with_rgb[int(index)] for index in indices]
    sample_count = len(selected_records)
    if sample_count == 0:
        return
    cell_width, cell_height, label_height = 360, 203, 24
    sheet = Image.new(
        "RGB", (cell_width * 4, (cell_height + label_height) * sample_count), "#111111"
    )
    draw = ImageDraw.Draw(sheet)
    for row, record in enumerate(selected_records):
        source = Image.open(record["source_rgb"]).convert("RGB")
        raw = Image.open(relative_to / record["diagnostics"]["visible_raw"]["path"]).convert("L")
        final = Image.open(relative_to / record["mask"]["path"]).convert("L")
        behind = Image.open(
            relative_to / record["diagnostics"]["occluded_behind"]["path"]
        ).convert("L")
        raw_overlay = source.copy()
        raw_overlay.paste(Image.new("RGB", source.size, (0, 255, 100)), mask=raw)
        final_overlay = source.copy()
        final_overlay.paste(Image.new("RGB", source.size, (255, 0, 170)), mask=final)
        rejection_overlay = source.copy()
        rejection_overlay.paste(Image.new("RGB", source.size, (255, 70, 0)), mask=behind)
        images = (source, raw_overlay, final_overlay, rejection_overlay)
        titles = (
            f"source {record['frame_id']}",
            f"visible anchor {record['coverage']['visible_raw_pixels']}",
            f"final supplement {record['coverage']['final_mask_pixels']}",
            f"behind scene {record['coverage']['occluded_behind_pixels']}",
        )
        y = row * (cell_height + label_height)
        for column, (image, title) in enumerate(zip(images, titles, strict=True)):
            image.thumbnail((cell_width, cell_height), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (cell_width, cell_height), "#202020")
            canvas.paste(
                image,
                ((cell_width - image.width) // 2, (cell_height - image.height) // 2),
            )
            x = column * cell_width
            sheet.paste(canvas, (x, y + label_height))
            draw.text((x + 8, y + 6), title, fill="#f4f4f4")
        for image in images:
            image.close()
        raw.close()
        final.close()
        behind.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def project(args: argparse.Namespace) -> dict[str, Any]:
    anchor_path = args.anchor.expanduser().resolve()
    camera_path = args.camera_info.expanduser().resolve()
    depth_dir = args.depth_dir.expanduser().resolve()
    frames_dir = args.frames_dir.expanduser().resolve() if args.frames_dir else None
    output_root = args.output.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    if not anchor_path.is_file() or not camera_path.is_file() or not depth_dir.is_dir():
        raise FileNotFoundError("anchor, camera_info, or depth directory is missing")
    if frames_dir is not None and not frames_dir.is_dir():
        raise NotADirectoryError(frames_dir)
    if args.point_stride < 1 or args.splat_radius < 0 or args.closing_radius < 0:
        raise ValueError("point stride must be positive and radii non-negative")
    if args.absolute_depth_tolerance < 0 or args.relative_depth_tolerance < 0:
        raise ValueError("depth tolerances must be non-negative")
    if args.minimum_visible_pixels < 1:
        raise ValueError("minimum-visible-pixels must be positive")
    if args.round_index < 1:
        raise ValueError("round-index must be positive")
    if args.contact_sheet_samples < 0:
        raise ValueError("contact-sheet-samples must be non-negative")

    camera_info = load_camera_info(camera_path)
    frame_ids = parse_frame_ids(args.frame_ids, camera_info)
    points, source_point_count = load_anchor(anchor_path, point_stride=args.point_stride)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_root.parent, prefix=f".{output_root.name}.staging-"
    ) as temporary_value:
        staging = Path(temporary_value)
        records: list[dict[str, Any]] = []
        for sequence_index, frame_id in enumerate(frame_ids):
            intrinsic = frame_intrinsic(camera_info, frame_id)
            matrix, rotation_error = world_to_camera(
                camera_info,
                frame_id,
                maximum_orthonormal_error=args.maximum_rotation_orthonormal_error,
            )
            depth_path = resolve_frame_path(depth_dir, frame_id, suffix=".npy")
            scene_depth = np.load(depth_path, allow_pickle=False).astype(np.float64, copy=False)
            expected_shape = (int(intrinsic["h"]), int(intrinsic["w"]))
            if scene_depth.shape != expected_shape:
                raise ValueError(f"depth shape mismatch for frame {frame_id}")
            anchor_depth, projection_counts = project_anchor_depth(points, matrix, intrinsic)
            visible_raw, diagnostics = depth_consistent(
                anchor_depth,
                scene_depth,
                absolute_tolerance=args.absolute_depth_tolerance,
                relative_tolerance=args.relative_depth_tolerance,
            )
            splatted, _ = depth_aware_splat(
                anchor_depth,
                scene_depth,
                radius=args.splat_radius,
                absolute_tolerance=args.absolute_depth_tolerance,
                relative_tolerance=args.relative_depth_tolerance,
            )
            final_mask, closing_envelope = depth_bounded_close(
                splatted,
                anchor_depth,
                scene_depth,
                splat_radius=args.splat_radius,
                closing_radius=args.closing_radius,
                absolute_tolerance=args.absolute_depth_tolerance,
                relative_tolerance=args.relative_depth_tolerance,
            )
            visible_subset_final = bool(not np.any(visible_raw & ~final_mask))
            final_subset_envelope = bool(not np.any(final_mask & ~closing_envelope))
            if not visible_subset_final:
                raise ValueError(f"frame {frame_id} final mask removed visible anchor evidence")
            if not final_subset_envelope:
                raise ValueError(f"frame {frame_id} final mask escapes depth-consistent envelope")
            if int(visible_raw.sum()) < args.minimum_visible_pixels:
                raise ValueError(
                    f"frame {frame_id} has only {int(visible_raw.sum())} visible anchor pixels"
                )
            mask_path = staging / "masks" / f"{frame_id}.png"
            visible_path = staging / "diagnostics" / "visible_raw" / f"{frame_id}.png"
            behind_path = staging / "diagnostics" / "occluded_behind" / f"{frame_id}.png"
            front_path = staging / "diagnostics" / "front_mismatch" / f"{frame_id}.png"
            missing_path = staging / "diagnostics" / "missing_depth" / f"{frame_id}.png"
            save_mask(mask_path, final_mask)
            save_mask(visible_path, visible_raw)
            save_mask(behind_path, diagnostics["occluded_behind_scene"])
            save_mask(front_path, diagnostics["front_depth_mismatch"])
            save_mask(missing_path, diagnostics["missing_scene_depth"])
            source_path = (
                resolve_frame_path(frames_dir, frame_id) if frames_dir is not None else None
            )
            records.append(
                {
                    "sequence_index": sequence_index,
                    "frame_id": frame_id,
                    "round_index": args.round_index,
                    "removed_object_ids": [args.object_id],
                    "source_rgb": str(source_path) if source_path else None,
                    "source_rgb_sha256": sha256_file(source_path) if source_path else None,
                    "scene_depth": str(depth_path),
                    "scene_depth_sha256": sha256_file(depth_path),
                    "camera_rotation_orthonormal_error": rotation_error,
                    "projection": projection_counts,
                    "coverage": {
                        "visible_raw_pixels": int(visible_raw.sum()),
                        "depth_aware_splat_pixels": int(splatted.sum()),
                        "closing_envelope_pixels": int(closing_envelope.sum()),
                        "final_mask_pixels": int(final_mask.sum()),
                        "final_fraction_of_image": float(final_mask.mean()),
                        "occluded_behind_pixels": int(
                            diagnostics["occluded_behind_scene"].sum()
                        ),
                        "front_depth_mismatch_pixels": int(
                            diagnostics["front_depth_mismatch"].sum()
                        ),
                        "missing_scene_depth_pixels": int(
                            diagnostics["missing_scene_depth"].sum()
                        ),
                    },
                    "mask": {
                        "path": f"masks/{frame_id}.png",
                        "sha256": sha256_file(mask_path),
                        "bbox_xyxy_exclusive": mask_bbox(final_mask),
                        "provenance": "depth_visible_object_anchor_projection",
                    },
                    "union_mask": f"masks/{frame_id}.png",
                    "union_mask_sha256": sha256_file(mask_path),
                    "diagnostics": {
                        "visible_raw": {
                            "path": f"diagnostics/visible_raw/{frame_id}.png",
                            "sha256": sha256_file(visible_path),
                        },
                        "occluded_behind": {
                            "path": f"diagnostics/occluded_behind/{frame_id}.png",
                            "sha256": sha256_file(behind_path),
                        },
                        "front_mismatch": {
                            "path": f"diagnostics/front_mismatch/{frame_id}.png",
                            "sha256": sha256_file(front_path),
                        },
                        "missing_depth": {
                            "path": f"diagnostics/missing_depth/{frame_id}.png",
                            "sha256": sha256_file(missing_path),
                        },
                    },
                    "gates": {
                        "minimum_visible_pixels": int(visible_raw.sum())
                        >= args.minimum_visible_pixels,
                        "visible_raw_is_subset_of_final_mask": visible_subset_final,
                        "final_mask_is_subset_of_depth_consistent_anchor_envelope": (
                            final_subset_envelope
                        ),
                        "sam_mask_dilation_used": False,
                    },
                }
            )

        mask_set = [
            {"frame_id": record["frame_id"], "mask_sha256": record["mask"]["sha256"]}
            for record in records
        ]
        final_counts = [record["coverage"]["final_mask_pixels"] for record in records]
        report = {
            "schema_version": 1,
            "kind": MANIFEST_KIND,
            "status": "technical_passed",
            "created_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 - mil8 is 3.10
            "object_id": args.object_id,
            "round_index": args.round_index,
            "removed_object_ids": [args.object_id],
            "anchor": {
                "path": str(anchor_path),
                "sha256": sha256_file(anchor_path),
                "bytes": anchor_path.stat().st_size,
                "source_point_count": source_point_count,
                "projected_point_count": len(points),
                "point_stride": args.point_stride,
            },
            "camera_info": {
                "path": str(camera_path),
                "sha256": sha256_file(camera_path),
                "extrinsic_type": "world_to_camera",
            },
            "depth_dir": str(depth_dir),
            "frame_count": len(records),
            "frame_ids": frame_ids,
            "config": {
                "absolute_depth_tolerance": args.absolute_depth_tolerance,
                "relative_depth_tolerance": args.relative_depth_tolerance,
                "splat_radius": args.splat_radius,
                "closing_radius": args.closing_radius,
                "minimum_visible_pixels": args.minimum_visible_pixels,
                "maximum_rotation_orthonormal_error": (
                    args.maximum_rotation_orthonormal_error
                ),
                "visibility_rule": (
                    "abs(projected_anchor_depth - observed_DA3_depth) <= "
                    "max(absolute_tolerance, relative_tolerance * observed_depth)"
                ),
                "splat_rule": "propagate anchor depth then rerun the visibility rule",
                "closing_rule": (
                    "binary close intersected with a depth-consistent anchor envelope"
                ),
                "sam_mask_dilation": "forbidden",
            },
            "frame_records": records,
            "coverage_summary": {
                "final_mask_pixels_total": int(sum(final_counts)),
                "final_mask_pixels_minimum": int(min(final_counts)),
                "final_mask_pixels_median": float(np.median(final_counts)),
                "final_mask_pixels_mean": float(np.mean(final_counts)),
                "final_mask_pixels_maximum": int(max(final_counts)),
            },
            "mask_set_sha256": digest_json(mask_set),
            "gates": {
                "all_frames_have_minimum_visible_anchor_pixels": True,
                "all_visible_raw_masks_are_subsets_of_final_masks": all(
                    record["gates"]["visible_raw_is_subset_of_final_mask"]
                    for record in records
                ),
                "all_final_masks_are_subsets_of_depth_consistent_anchor_envelopes": all(
                    record["gates"][
                        "final_mask_is_subset_of_depth_consistent_anchor_envelope"
                    ]
                    for record in records
                ),
                "camera_convention_verified": True,
                "all_inputs_and_outputs_hashed": True,
                "sam_mask_dilation_used": False,
            },
            "usage": {
                "join_key": "frame_records[].frame_id",
                "mask_path": "frame_records[].mask.path relative to this manifest",
                "intended_role": "supplement-only union with the current bed removal mask",
                "not_a_claim": "this projection is not an amodal bed reconstruction",
            },
            "promotion_approved": False,
            "promotion_blocker": "consumer must inspect union coverage and downstream clean plate",
        }
        manifest_path = staging / "object_anchor_supplement_manifest.json"
        write_json(manifest_path, report)
        contact_path = staging / "object_anchor_projection_contact_sheet.png"
        make_contact_sheet(
            records,
            contact_path,
            samples=args.contact_sheet_samples,
            selected_frame_ids=(
                [
                    Path(item.strip()).stem
                    for item in args.contact_sheet_frame_ids.split(",")
                    if item.strip()
                ]
                if args.contact_sheet_frame_ids
                else None
            ),
            relative_to=staging,
        )
        receipt = {
            "schema_version": 1,
            "kind": RECEIPT_KIND,
            "object_id": args.object_id,
            "round_index": args.round_index,
            "removed_object_ids": [args.object_id],
            "manifest": manifest_path.name,
            "manifest_sha256": sha256_file(manifest_path),
            "mask_set_sha256": report["mask_set_sha256"],
            "contact_sheet": contact_path.name if contact_path.is_file() else None,
            "contact_sheet_sha256": (
                sha256_file(contact_path) if contact_path.is_file() else None
            ),
            "frame_count": len(records),
            "gates": {
                "all_visible_raw_masks_are_subsets_of_final_masks": report["gates"][
                    "all_visible_raw_masks_are_subsets_of_final_masks"
                ],
                "all_final_masks_are_subsets_of_depth_consistent_anchor_envelopes": (
                    report["gates"][
                        "all_final_masks_are_subsets_of_depth_consistent_anchor_envelopes"
                    ]
                ),
                "sam_mask_dilation_used": False,
            },
            "promotion_approved": False,
        }
        write_json(staging / "receipt.json", receipt)
        shutil.move(str(staging), output_root)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--round-index", type=int, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--camera-info", type=Path, required=True)
    parser.add_argument("--depth-dir", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path)
    parser.add_argument("--frame-ids")
    parser.add_argument("--point-stride", type=int, default=1)
    parser.add_argument("--absolute-depth-tolerance", type=float, default=0.08)
    parser.add_argument("--relative-depth-tolerance", type=float, default=0.01)
    parser.add_argument("--splat-radius", type=int, default=1)
    parser.add_argument("--closing-radius", type=int, default=1)
    parser.add_argument("--minimum-visible-pixels", type=int, default=25)
    parser.add_argument("--maximum-rotation-orthonormal-error", type=float, default=1e-3)
    parser.add_argument("--contact-sheet-samples", type=int, default=7)
    parser.add_argument("--contact-sheet-frame-ids")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = project(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "object_id": report["object_id"],
                "frame_count": report["frame_count"],
                "coverage_summary": report["coverage_summary"],
                "manifest": str(args.output / "object_anchor_supplement_manifest.json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
