#!/usr/bin/env python3
"""Recover large occluded room surfaces with TSDF-fitted structural planes.

This stage is deliberately geometry-first. It fits a Manhattan room envelope
from a scene TSDF mesh, intersects object-removal rays with that envelope, and
reprojects only measured RGB-D donor samples into per-plane texture atlases.
Pixels whose geometry is known but whose texture was never observed are emitted
as a separate generative-texture mask; they are never silently painted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class Plane:
    plane_id: int
    axis_index: int
    side: str
    semantic_role: str
    normal: np.ndarray
    offset: float
    basis_u: np.ndarray
    basis_v: np.ndarray
    bounds_uv: tuple[float, float, float, float]
    support_area: float
    inlier_faces: int
    rms_residual: float
    tsdf_offset: float
    anchor_role: str | None
    anchor_inliers: int
    anchor_fraction: float


@dataclass
class TextureAtlas:
    plane_id: int
    origin_uv: tuple[float, float]
    texel_size: float
    color: np.ndarray
    frame_support: np.ndarray
    observed: np.ndarray


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


def resolve_path(value: str, *, relative_to: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def frame_intrinsic(camera_info: dict[str, Any], frame_id: str) -> dict[str, float]:
    intrinsics = camera_info.get("intrinsics")
    camera_ids = camera_info.get("frame_camera_ids")
    value: Any = None
    if isinstance(intrinsics, dict):
        if frame_id in intrinsics:
            value = intrinsics[frame_id]
        elif isinstance(camera_ids, dict):
            value = intrinsics.get(str(camera_ids.get(frame_id, "")))
        elif len(intrinsics) == 1:
            value = next(iter(intrinsics.values()))
    if value is None:
        value = camera_info.get("intrinsic")
    required = ("fx", "fy", "cx", "cy", "w", "h")
    if not isinstance(value, dict) or not all(
        isinstance(value.get(key), int | float) for key in required
    ):
        raise ValueError(f"no complete intrinsic calibration for frame {frame_id}")
    return {key: float(value[key]) for key in required}


def world_to_camera(camera_info: dict[str, Any], frame_id: str) -> np.ndarray:
    value = camera_info.get("extrinsic", {}).get(frame_id)
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"invalid world-to-camera matrix for frame {frame_id}")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"invalid homogeneous row for frame {frame_id}")
    return matrix


def camera_center(matrix: np.ndarray) -> np.ndarray:
    return -(matrix[:3, :3].T @ matrix[:3, 3])


def row_dot(vectors: np.ndarray, direction: np.ndarray) -> np.ndarray:
    return np.einsum("ni,i->n", vectors, direction, optimize=True)


def row_transform(vectors: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.einsum("ni,ij->nj", vectors, matrix, optimize=True)


def record_asset_path(
    record: dict[str, Any],
    key: str,
    *,
    manifest_dir: Path,
    override_dir: Path | None,
) -> Path:
    if override_dir is not None:
        sequence_index = int(record["sequence_index"])
        matches = sorted(override_dir.glob(f"{sequence_index:04d}.*"))
        if len(matches) != 1:
            raise ValueError(
                f"expected one override {key} for sequence {sequence_index}, found {len(matches)}"
            )
        return matches[0].resolve()
    value = record.get(key)
    if isinstance(value, dict):
        value = value.get("path")
    if not isinstance(value, str):
        raise ValueError(f"frame record is missing {key}.path")
    return resolve_path(value, relative_to=manifest_dir)


def resolve_mask_from_directory(
    directory: Path,
    record: dict[str, Any],
) -> Path:
    sequence_stem = f"{int(record['sequence_index']):04d}"
    frame_stem = str(record["frame_id"])
    matches = {
        path.resolve()
        for stem in (sequence_stem, frame_stem)
        for path in directory.glob(f"{stem}.*")
        if path.is_file()
    }
    if len(matches) != 1:
        raise ValueError(
            f"expected one mask for sequence={sequence_stem} frame={frame_stem} in "
            f"{directory}, found {len(matches)}"
        )
    return next(iter(matches))


def compose_cumulative_removal(
    base_residual: np.ndarray,
    original_removal: np.ndarray,
    supplements: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if base_residual.shape != original_removal.shape:
        raise ValueError("base residual and original removal masks differ in shape")
    if np.any(base_residual & ~original_removal):
        raise ValueError("base residual mask is not a subset of the original removal mask")
    supplement_union = np.zeros_like(base_residual, dtype=bool)
    for supplement in supplements:
        if supplement.shape != base_residual.shape:
            raise ValueError("supplement mask shape differs from the target frame")
        supplement_union |= supplement
    cumulative = base_residual | supplement_union
    original_union = original_removal | supplement_union
    upstream_kept = original_removal & ~base_residual & ~supplement_union
    if not np.array_equal(upstream_kept | cumulative, original_union):
        raise ValueError("cumulative removal partition is inconsistent")
    return cumulative, original_union, upstream_kept, {
        "base_residual_pixels": int(base_residual.sum()),
        "original_removal_pixels": int(original_removal.sum()),
        "supplement_union_pixels": int(supplement_union.sum()),
        "supplement_added_outside_base_pixels": int((supplement_union & ~base_residual).sum()),
        "supplement_overlaps_base_pixels": int((supplement_union & base_residual).sum()),
        "upstream_measured_pixels_kept": int(upstream_kept.sum()),
        "cumulative_removal_pixels": int(cumulative.sum()),
        "cumulative_original_union_pixels": int(original_union.sum()),
        "partition_exact": True,
        "identity_precedence": "supplement residual overrides upstream measured donor prefill",
    }


def add_cumulative_safety_margin(
    cumulative: np.ndarray,
    original_union: np.ndarray,
    upstream_kept: np.ndarray,
    receipt: dict[str, Any],
    *,
    dilation: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if dilation < 0:
        raise ValueError("cumulative mask dilation must be non-negative")
    expanded = binary_dilate(cumulative, dilation)
    added = expanded & ~cumulative
    original_union = original_union | expanded
    upstream_kept = upstream_kept & ~expanded
    updated = {
        **receipt,
        "cumulative_safety_dilation_pixels": dilation,
        "cumulative_safety_margin_added_pixels": int(added.sum()),
        "cumulative_removal_pixels_before_safety_margin": int(cumulative.sum()),
        "cumulative_removal_pixels": int(expanded.sum()),
        "cumulative_original_union_pixels": int(original_union.sum()),
        "upstream_measured_pixels_kept": int(upstream_kept.sum()),
    }
    if not np.array_equal(upstream_kept | expanded, original_union):
        raise ValueError("dilated cumulative removal partition is inconsistent")
    return expanded, original_union, upstream_kept, updated


def prepare_cumulative_removal_inputs(
    records: list[dict[str, Any]],
    *,
    manifest_dir: Path,
    target_mask_key: str,
    masks_override: Path | None,
    original_removal_masks_dir: Path | None,
    supplement_mask_dirs: list[Path],
    supplement_manifest_assets: dict[str, list[dict[str, Any]]],
    cumulative_mask_dilation: int,
    texture_donor_exclusion_dilation: int,
    output_dir: Path,
    donor_exclusion_dir: Path,
) -> dict[str, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    donor_exclusion_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, dict[str, Any]] = {}
    for output_index, record in enumerate(records):
        frame_id = str(record["frame_id"])
        base_mask_path = record_asset_path(
            record,
            target_mask_key,
            manifest_dir=manifest_dir,
            override_dir=masks_override,
        )
        if original_removal_masks_dir is not None:
            original_mask_path = resolve_mask_from_directory(
                original_removal_masks_dir,
                record,
            )
        elif "removal_mask" in record:
            original_mask_path = record_asset_path(
                record,
                "removal_mask",
                manifest_dir=manifest_dir,
                override_dir=None,
            )
        else:
            original_mask_path = base_mask_path
        supplement_asset_records = [
            {
                "path": resolve_mask_from_directory(directory, record),
                "source": "directory",
                "directory": str(directory),
            }
            for directory in supplement_mask_dirs
        ]
        supplement_asset_records.extend(supplement_manifest_assets[frame_id])
        deduplicated_assets: list[dict[str, Any]] = []
        seen_supplements: set[Path] = set()
        for asset in supplement_asset_records:
            path = Path(asset["path"]).resolve()
            if path in seen_supplements:
                continue
            seen_supplements.add(path)
            deduplicated_assets.append({**asset, "path": path})
        supplement_asset_records = deduplicated_assets
        supplement_paths = [Path(asset["path"]) for asset in supplement_asset_records]
        base_removal = mask_bool(base_mask_path)
        original_removal = mask_bool(original_mask_path)
        supplement_masks = [mask_bool(path) for path in supplement_paths]
        removal, original_union, upstream_kept, receipt = compose_cumulative_removal(
            base_removal,
            original_removal,
            supplement_masks,
        )
        removal, original_union, upstream_kept, receipt = add_cumulative_safety_margin(
            removal,
            original_union,
            upstream_kept,
            receipt,
            dilation=cumulative_mask_dilation,
        )
        if texture_donor_exclusion_dilation < 0:
            raise ValueError("texture donor exclusion dilation must be non-negative")
        donor_exclusion = binary_dilate(
            original_union,
            texture_donor_exclusion_dilation,
        )
        donor_exclusion_receipt = {
            "dilation_pixels": texture_donor_exclusion_dilation,
            "pixels_before_dilation": int(original_union.sum()),
            "added_shadow_collar_pixels": int((donor_exclusion & ~original_union).sum()),
            "final_pixels": int(donor_exclusion.sum()),
            "role": (
                "texture-donor-only object edge and cast-shadow exclusion; does not expand the "
                "RGB output mutation mask"
            ),
        }
        cumulative_mask_path = output_dir / f"{output_index:04d}.png"
        donor_exclusion_mask_path = donor_exclusion_dir / f"{output_index:04d}.png"
        Image.fromarray(removal.astype(np.uint8) * 255).save(cumulative_mask_path)
        Image.fromarray(donor_exclusion.astype(np.uint8) * 255).save(
            donor_exclusion_mask_path
        )
        result[frame_id] = {
            "base_mask_path": base_mask_path,
            "original_mask_path": original_mask_path,
            "supplement_asset_records": supplement_asset_records,
            "supplement_paths": supplement_paths,
            "supplement_masks": supplement_masks,
            "removal": removal,
            "original_union": original_union,
            "upstream_kept": upstream_kept,
            "receipt": receipt,
            "cumulative_mask_path": cumulative_mask_path,
            "donor_exclusion_mask_path": donor_exclusion_mask_path,
            "donor_exclusion": donor_exclusion,
            "donor_exclusion_receipt": donor_exclusion_receipt,
        }
    return result


def structural_plane_support_for_pixels(
    x: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    intrinsic: dict[str, float],
    matrix: np.ndarray,
    planes: list[Plane],
    *,
    plane_tolerance: float,
) -> tuple[np.ndarray, dict[str, int]]:
    if plane_tolerance < 0:
        raise ValueError("structural shadow plane tolerance must be non-negative")
    selected = np.zeros(len(x), dtype=bool)
    plane_counts: dict[str, int] = {}
    sampled_depth = depth[y, x].astype(np.float64, copy=False)
    finite = np.isfinite(sampled_depth) & (sampled_depth > 0.0)
    if not np.any(finite):
        return selected, {str(plane.plane_id): 0 for plane in planes}
    camera_points = np.stack(
        [
            (x[finite] - intrinsic["cx"]) * sampled_depth[finite] / intrinsic["fx"],
            (y[finite] - intrinsic["cy"]) * sampled_depth[finite] / intrinsic["fy"],
            sampled_depth[finite],
        ],
        axis=1,
    )
    world_points = row_transform(camera_points - matrix[:3, 3], matrix[:3, :3])
    finite_indices = np.flatnonzero(finite)
    for plane in planes:
        distance = np.abs(row_dot(world_points, plane.normal) - plane.offset)
        u = row_dot(world_points, plane.basis_u)
        v = row_dot(world_points, plane.basis_v)
        u0, u1, v0, v1 = plane.bounds_uv
        supported = (
            (distance <= plane_tolerance)
            & (u >= u0)
            & (u <= u1)
            & (v >= v0)
            & (v <= v1)
        )
        selected[finite_indices[supported]] = True
        plane_counts[str(plane.plane_id)] = int(supported.sum())
    return selected, plane_counts


def add_structural_shadow_collars(
    cumulative_inputs: dict[str, dict[str, Any]],
    records: list[dict[str, Any]],
    *,
    radius: int,
    plane_tolerance: float,
    texture_donor_exclusion_dilation: int,
    depth_dir: Path,
    camera_info: dict[str, Any],
    planes: list[Plane],
    protected_manifest_assets: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    if radius < 0:
        raise ValueError("structural shadow collar radius must be non-negative")
    total_candidates = 0
    total_added = 0
    total_protected = 0
    for record in records:
        frame_id = str(record["frame_id"])
        item = cumulative_inputs[frame_id]
        removal = item["removal"]
        protected_asset_records = protected_manifest_assets[frame_id]
        protected_union = np.zeros_like(removal)
        for asset in protected_asset_records:
            protected_mask = mask_bool(Path(asset["path"]))
            if protected_mask.shape != removal.shape:
                raise ValueError(f"protected mask shape mismatch for {frame_id}")
            protected_union |= protected_mask
        expanded = binary_dilate(removal, radius)
        candidate = expanded & ~removal
        total_candidates += int(candidate.sum())
        added = np.zeros_like(removal)
        plane_counts = {str(plane.plane_id): 0 for plane in planes}
        if np.any(candidate):
            depth_path = (depth_dir / f"{frame_id}.npy").resolve()
            depth = np.load(depth_path, allow_pickle=False)
            if depth.shape != removal.shape:
                raise ValueError(f"shadow collar depth/mask shape mismatch for {frame_id}")
            y, x = np.nonzero(candidate)
            supported, plane_counts = structural_plane_support_for_pixels(
                x,
                y,
                depth,
                frame_intrinsic(camera_info, frame_id),
                world_to_camera(camera_info, frame_id),
                planes,
                plane_tolerance=plane_tolerance,
            )
            added[y[supported], x[supported]] = True
        protected_overlap = added & protected_union
        added &= ~protected_union
        protected = candidate & ~added
        updated_removal = removal | added
        updated_original_union = item["original_union"] | added
        updated_upstream_kept = item["upstream_kept"] & ~added
        if not np.array_equal(
            updated_upstream_kept | updated_removal,
            updated_original_union,
        ):
            raise ValueError(f"structural shadow collar partition failed for {frame_id}")
        donor_exclusion = binary_dilate(
            updated_original_union,
            texture_donor_exclusion_dilation,
        )
        Image.fromarray(updated_removal.astype(np.uint8) * 255).save(
            item["cumulative_mask_path"]
        )
        Image.fromarray(donor_exclusion.astype(np.uint8) * 255).save(
            item["donor_exclusion_mask_path"]
        )
        item["removal"] = updated_removal
        item["original_union"] = updated_original_union
        item["upstream_kept"] = updated_upstream_kept
        item["donor_exclusion"] = donor_exclusion
        item["receipt"] = {
            **item["receipt"],
            "structural_shadow_collar_radius_pixels": radius,
            "structural_shadow_plane_tolerance": plane_tolerance,
            "structural_shadow_candidate_pixels": int(candidate.sum()),
            "structural_shadow_added_pixels": int(added.sum()),
            "protected_nonstructural_neighbor_pixels": int(protected.sum()),
            "protected_anchor_pixels": int(protected_union.sum()),
            "protected_anchor_overlap_suppressed_pixels": int(protected_overlap.sum()),
            "protected_anchor_masks": [
                {
                    "path": str(asset["path"]),
                    "sha256": asset["sha256"],
                    "manifest_sha256": asset["manifest_sha256"],
                    "protected_object_ids": asset["removed_object_ids"],
                }
                for asset in protected_asset_records
            ],
            "structural_shadow_added_by_plane": plane_counts,
            "cumulative_removal_pixels": int(updated_removal.sum()),
            "cumulative_original_union_pixels": int(updated_original_union.sum()),
            "upstream_measured_pixels_kept": int(updated_upstream_kept.sum()),
        }
        item["donor_exclusion_receipt"] = {
            **item["donor_exclusion_receipt"],
            "pixels_before_dilation": int(updated_original_union.sum()),
            "added_shadow_collar_pixels": int(
                (donor_exclusion & ~updated_original_union).sum()
            ),
            "final_pixels": int(donor_exclusion.sum()),
        }
        total_added += int(added.sum())
        total_protected += int(protected.sum())
    return {
        "radius_pixels": radius,
        "plane_tolerance": plane_tolerance,
        "candidate_pixels": total_candidates,
        "added_structural_pixels": total_added,
        "protected_nonstructural_neighbor_pixels": total_protected,
        "all_added_pixels_are_depth_consistent_with_a_structural_plane": True,
        "protected_anchor_masks_subtracted_only_from_shadow_collar_additions": True,
    }


def load_supplement_mask_manifests(
    paths: list[Path],
    target_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    target_by_frame = {str(record["frame_id"]): record for record in target_records}
    assets: dict[str, list[dict[str, Any]]] = {frame_id: [] for frame_id in target_by_frame}
    lineage: list[dict[str, Any]] = []
    for path in paths:
        value = read_json(path)
        if not isinstance(value, dict) or not isinstance(value.get("frame_records"), list):
            raise ValueError(f"supplement manifest must contain frame_records: {path}")
        if value.get("status") not in {
            "ready_for_measured_multiview_prefill",
            "technical_passed",
        }:
            raise ValueError(f"supplement manifest status is not consumable: {path}")
        manifest_sha = sha256_file(path)
        manifest_frames: set[str] = set()
        for item in value["frame_records"]:
            frame_id = str(item.get("frame_id"))
            if frame_id not in target_by_frame:
                continue
            if frame_id in manifest_frames:
                raise ValueError(f"duplicate frame {frame_id} in supplement manifest {path}")
            manifest_frames.add(frame_id)
            target = target_by_frame[frame_id]
            if int(item.get("sequence_index", -1)) != int(target["sequence_index"]):
                raise ValueError(f"supplement sequence mismatch for frame {frame_id}")
            relative_mask = item.get("union_mask")
            declared_sha = item.get("union_mask_sha256")
            if relative_mask is None or declared_sha is None:
                mask_record = item.get("mask")
                if isinstance(mask_record, dict):
                    relative_mask = mask_record.get("path")
                    declared_sha = mask_record.get("sha256")
            if not isinstance(relative_mask, str) or not isinstance(declared_sha, str):
                raise ValueError(
                    f"supplement frame {frame_id} lacks union_mask/hash or mask.path/hash"
                )
            mask_path = resolve_path(relative_mask, relative_to=path.parent)
            actual_sha = sha256_file(mask_path)
            if actual_sha != declared_sha:
                raise ValueError(f"supplement mask hash mismatch for frame {frame_id}")
            removed_object_ids = item.get(
                "removed_object_ids",
                value.get("removed_object_ids"),
            )
            if removed_object_ids is None and isinstance(value.get("object_id"), str):
                removed_object_ids = [value["object_id"]]
            if not isinstance(removed_object_ids, list) or not all(
                isinstance(object_id, str) for object_id in removed_object_ids
            ):
                raise ValueError(
                    f"supplement frame {frame_id} has invalid removed_object_ids"
                )
            assets[frame_id].append(
                {
                    "path": mask_path,
                    "sha256": actual_sha,
                    "manifest": path,
                    "manifest_sha256": manifest_sha,
                    "removed_object_ids": removed_object_ids,
                }
            )
        if manifest_frames != set(target_by_frame):
            missing = sorted(set(target_by_frame) - manifest_frames)
            raise ValueError(f"supplement manifest is missing selected frames: {missing}")
        receipt_path = path.parent / "receipt.json"
        receipt_record = None
        if receipt_path.is_file():
            receipt = read_json(receipt_path)
            if not isinstance(receipt, dict) or receipt.get("manifest_sha256") != manifest_sha:
                raise ValueError(f"supplement receipt does not bind manifest hash: {receipt_path}")
            receipt_record = {
                "path": str(receipt_path),
                "sha256": sha256_file(receipt_path),
                "manifest_sha256": receipt["manifest_sha256"],
                "donor_exclusion_index_sha256": receipt.get(
                    "donor_exclusion_index_sha256"
                ),
            }
        removed_object_ids = value.get("removed_object_ids")
        if removed_object_ids is None and isinstance(value.get("object_id"), str):
            removed_object_ids = [value["object_id"]]
        lineage.append(
            {
                "path": str(path),
                "sha256": manifest_sha,
                "kind": value.get("kind"),
                "status": value.get("status"),
                "round_index": value.get("round_index"),
                "removed_object_ids": removed_object_ids or [],
                "frame_count": len(manifest_frames),
                "receipt": receipt_record,
            }
        )
    return lineage, assets


def build_protected_sam_mask_assets(
    root: Path | None,
    labels: list[str],
    records: list[dict[str, Any]],
    *,
    dilation: int,
    output_dir: Path,
) -> tuple[dict[str, Any] | None, dict[str, list[dict[str, Any]]]]:
    assets = {str(record["frame_id"]): [] for record in records}
    if root is None:
        return None, assets
    if not root.is_dir():
        raise NotADirectoryError(root)
    if not labels:
        raise ValueError("protected SAM mask root requires at least one protected label")
    if dilation < 0:
        raise ValueError("protected SAM mask dilation must be non-negative")
    index_path = root / "mask_index.json"
    index_sha = sha256_file(index_path) if index_path.is_file() else None
    output_dir.mkdir(parents=True, exist_ok=True)
    source_mask_count = 0
    protected_pixel_total = 0
    frame_records: list[dict[str, Any]] = []
    for output_index, record in enumerate(records):
        frame_id = str(record["frame_id"])
        frame_root = root / frame_id
        if not frame_root.is_dir():
            raise NotADirectoryError(frame_root)
        source_paths = sorted(
            {
                path.resolve()
                for label in labels
                for path in frame_root.glob(f"{label}*.png")
                if path.is_file()
            }
        )
        if not source_paths:
            raise ValueError(f"no protected SAM masks found for frame {frame_id}")
        source_masks = [mask_bool(path) for path in source_paths]
        shape = source_masks[0].shape
        if any(mask.shape != shape for mask in source_masks):
            raise ValueError(f"protected SAM masks differ in shape for frame {frame_id}")
        union = np.zeros(shape, dtype=bool)
        for mask in source_masks:
            union |= mask
        union = binary_dilate(union, dilation)
        output_path = output_dir / f"{output_index:04d}.png"
        Image.fromarray(union.astype(np.uint8) * 255).save(output_path)
        source_records = [
            {"path": str(path), "sha256": sha256_file(path)} for path in source_paths
        ]
        asset = {
            "path": output_path,
            "sha256": sha256_file(output_path),
            "manifest_sha256": index_sha,
            "removed_object_ids": labels,
            "source_masks": source_records,
        }
        assets[frame_id].append(asset)
        source_mask_count += len(source_paths)
        protected_pixel_total += int(union.sum())
        frame_records.append(
            {
                "sequence_index": int(record["sequence_index"]),
                "frame_id": frame_id,
                "union_mask": str(output_path),
                "union_mask_sha256": asset["sha256"],
                "pixels": int(union.sum()),
                "source_masks": source_records,
            }
        )
    return {
        "kind": "video2world.protected_neighbor_sam_mask_set",
        "root": str(root),
        "mask_index": str(index_path) if index_path.is_file() else None,
        "mask_index_sha256": index_sha,
        "protected_labels": labels,
        "dilation_pixels": dilation,
        "frame_count": len(frame_records),
        "source_mask_count": source_mask_count,
        "protected_pixels_total": protected_pixel_total,
        "frame_records": frame_records,
    }, assets


def select_record_key(
    records: list[dict[str, Any]],
    requested: str,
    candidates: tuple[str, ...],
) -> str:
    if requested != "auto":
        if not all(requested in record for record in records):
            raise ValueError(f"record key {requested!r} is absent from one or more frames")
        return requested
    for candidate in candidates:
        if all(candidate in record for record in records):
            return candidate
    raise ValueError(f"none of the record keys {candidates} is available for every frame")


def validate_upstream_prefill(path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    report = read_json(path)
    if not isinstance(report, dict) or not isinstance(report.get("frame_records"), list):
        raise ValueError("upstream prefill report must contain frame_records")
    if report.get("status") != "technical_passed":
        raise ValueError("upstream prefill report did not pass its technical gates")
    gates = report.get("gates")
    if not isinstance(gates, dict):
        raise ValueError("upstream prefill report is missing gates")
    required_gates = (
        "outside_removal_mask_rgb_exact",
        "all_residual_masks_subset_of_removal_masks",
    )
    if not all(gates.get(key) is True for key in required_gates):
        raise ValueError("upstream prefill exactness/subset gates did not pass")
    upstream_ids = [str(record.get("frame_id")) for record in report["frame_records"]]
    selected_ids = [str(record.get("frame_id")) for record in records]
    if upstream_ids != selected_ids:
        raise ValueError("upstream prefill frame order differs from selected target records")
    coverage = [float(record.get("coverage_fraction", 0.0)) for record in report["frame_records"]]
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "schema_version": report.get("schema_version"),
        "status": report["status"],
        "purpose": report.get("purpose"),
        "gates": {key: gates[key] for key in required_gates},
        "frame_count": len(upstream_ids),
        "coverage_fraction": {
            "minimum": min(coverage),
            "mean": float(np.mean(coverage)),
            "median": float(np.median(coverage)),
            "maximum": max(coverage),
        },
        "lineage_role": (
            "measured multi-view donor prefill; planar completion consumes only its residual masks"
        ),
    }


def load_camera_info(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError("camera info must be a JSON object")
    if value.get("extrinsic_type") != "world_to_camera":
        raise ValueError("camera_info.extrinsic_type must be 'world_to_camera'")
    if not isinstance(value.get("extrinsic"), dict):
        raise ValueError("camera_info.extrinsic must be frame keyed")
    return value


def load_mesh_triangles(path: Path, max_faces: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        from plyfile import PlyData
    except ImportError as error:  # pragma: no cover - environment contract
        raise RuntimeError("install video2world[geometry] to read TSDF PLY meshes") from error

    ply = PlyData.read(str(path))
    vertex = ply["vertex"].data
    vertices = np.column_stack([vertex[axis] for axis in ("x", "y", "z")]).astype(
        np.float64,
        copy=False,
    )
    face_data = ply["face"].data
    face_key = "vertex_indices" if "vertex_indices" in face_data.dtype.names else "vertex_index"
    faces = np.asarray([np.asarray(item, dtype=np.int64) for item in face_data[face_key]])
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("TSDF mesh must contain triangular faces")
    if max_faces > 0 and len(faces) > max_faces:
        indices = np.linspace(0, len(faces) - 1, max_faces, dtype=np.int64)
        faces = faces[indices]
    return vertices, faces


def parse_structure_anchor_specs(values: list[str] | None) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values or []:
        role, separator, raw_path = value.partition("=")
        role = role.strip()
        if not separator or not role or not raw_path.strip():
            raise ValueError("structure anchors must use role=/path/to/anchor.ply")
        if role in result:
            raise ValueError(f"duplicate structure anchor role: {role}")
        path = Path(raw_path.strip()).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        result[role] = path
    return result


def load_structure_anchors(
    paths: dict[str, Path],
    maximum_points: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    try:
        from plyfile import PlyData
    except ImportError as error:  # pragma: no cover - environment contract
        raise RuntimeError("install video2world[geometry] to read structure anchors") from error

    anchors: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    for role, path in sorted(paths.items()):
        vertex = PlyData.read(str(path))["vertex"].data
        points = np.column_stack([vertex[axis] for axis in ("x", "y", "z")]).astype(
            np.float64,
            copy=False,
        )
        points = points[np.isfinite(points).all(axis=1)]
        original_count = len(points)
        if maximum_points > 0 and len(points) > maximum_points:
            indices = np.linspace(0, len(points) - 1, maximum_points, dtype=np.int64)
            points = points[indices]
        if not len(points):
            raise ValueError(f"structure anchor {role!r} has no finite points")
        anchors[role] = points
        records.append(
            {
                "role": role,
                "path": str(path),
                "sha256": sha256_file(path),
                "point_count": original_count,
                "sampled_point_count": len(points),
                "bounds": [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
            }
        )
    return anchors, records


def face_geometry(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, ...]:
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    finite = (
        np.isfinite(triangles).all(axis=(1, 2))
        & np.isfinite(cross).all(axis=1)
        & np.isfinite(double_area)
        & (double_area > 1e-10)
    )
    centers = triangles[finite].mean(axis=1)
    normals = cross[finite] / double_area[finite, None]
    areas = double_area[finite] * 0.5
    return centers, normals, areas


def canonicalize_normals(normals: np.ndarray) -> np.ndarray:
    result = normals.copy()
    dominant = np.argmax(np.abs(result), axis=1)
    signs = np.where(result[np.arange(len(result)), dominant] < 0.0, -1.0, 1.0)
    result *= signs[:, None]
    return result


def closest_orthonormal_frame(matrix: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(matrix, full_matrices=False)
    result = u @ vh
    if np.linalg.det(result) < 0:
        u[:, -1] *= -1
        result = u @ vh
    return result


def fit_manhattan_axes(
    normals: np.ndarray,
    areas: np.ndarray,
    *,
    normal_bucket_size: float,
    angular_tolerance_degrees: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    canonical = canonicalize_normals(normals)
    quantized = np.rint(canonical / normal_bucket_size).astype(np.int16)
    unique, inverse = np.unique(quantized, axis=0, return_inverse=True)
    weights = np.bincount(inverse, weights=areas)
    order = np.argsort(weights)[::-1]
    candidates: list[np.ndarray] = []
    candidate_weights: list[float] = []
    for index in order:
        direction = unique[index].astype(np.float64) * normal_bucket_size
        norm = np.linalg.norm(direction)
        if norm <= 1e-8:
            continue
        direction /= norm
        if any(abs(float(direction @ existing)) > 0.94 for existing in candidates):
            continue
        candidates.append(direction)
        candidate_weights.append(float(weights[index]))
        if len(candidates) >= 64:
            break
    if len(candidates) < 2:
        raise ValueError("mesh normals do not contain two independent dominant directions")
    first = candidates[0]
    second = next(
        (candidate for candidate in candidates[1:] if abs(float(candidate @ first)) < 0.35),
        None,
    )
    if second is None:
        raise ValueError("mesh normals do not contain an orthogonal Manhattan direction")
    second = second - first * float(second @ first)
    second /= np.linalg.norm(second)
    axes = np.stack([first, second, np.cross(first, second)])
    axes = closest_orthonormal_frame(axes)
    cosine_threshold = math.cos(math.radians(angular_tolerance_degrees))
    for _ in range(5):
        alignment = np.abs(row_transform(normals, axes.T))
        assignment = np.argmax(alignment, axis=1)
        refined: list[np.ndarray] = []
        for axis_index in range(3):
            selected = (assignment == axis_index) & (
                alignment[:, axis_index] >= cosine_threshold
            )
            if not np.any(selected):
                raise ValueError(f"no mesh support for Manhattan axis {axis_index}")
            weighted_outer = np.einsum(
                "n,ni,nj->ij",
                areas[selected],
                normals[selected],
                normals[selected],
            )
            values, vectors = np.linalg.eigh(weighted_outer)
            direction = vectors[:, int(np.argmax(values))]
            if float(direction @ axes[axis_index]) < 0.0:
                direction *= -1.0
            refined.append(direction)
        axes = closest_orthonormal_frame(np.stack(refined))
    alignment = np.abs(row_transform(normals, axes.T))
    assignment = np.argmax(alignment, axis=1)
    support = [
        float(
            areas[
                (assignment == index) & (alignment[:, index] >= cosine_threshold)
            ].sum()
        )
        for index in range(3)
    ]
    diagnostics = {
        "bucket_candidate_count": len(candidates),
        "top_bucket_weights": candidate_weights[:10],
        "axis_support_area": support,
        "maximum_axis_dot_error": float(
            np.max(np.abs(axes @ axes.T - np.eye(3)))
        ),
    }
    return axes, diagnostics


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cutoff = float(weights.sum()) * 0.5
    return float(values[min(int(np.searchsorted(np.cumsum(weights), cutoff)), len(values) - 1)])


def plane_basis(axes: np.ndarray, axis_index: int) -> tuple[np.ndarray, np.ndarray]:
    others = [index for index in range(3) if index != axis_index]
    return axes[others[0]], axes[others[1]]


def fit_axis_plane_candidates(
    centers: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    axes: np.ndarray,
    axis_index: int,
    *,
    angular_tolerance_degrees: float,
    offset_bin_size: float,
    plane_tolerance: float,
    peak_separation: float,
    maximum_peaks: int,
) -> list[dict[str, Any]]:
    axis = axes[axis_index]
    alignment = np.abs(row_dot(normals, axis))
    selected = alignment >= math.cos(math.radians(angular_tolerance_degrees))
    projection = row_dot(centers[selected], axis)
    selected_areas = areas[selected]
    if not len(projection):
        return []
    low = math.floor(float(projection.min()) / offset_bin_size) * offset_bin_size
    high = math.ceil(float(projection.max()) / offset_bin_size) * offset_bin_size
    edges = np.arange(low, high + 2.0 * offset_bin_size, offset_bin_size)
    histogram, edges = np.histogram(projection, bins=edges, weights=selected_areas)
    smooth = np.convolve(histogram, np.array([1, 2, 3, 2, 1]) / 9.0, mode="same")
    local = np.flatnonzero(
        (smooth >= np.r_[0.0, smooth[:-1]])
        & (smooth >= np.r_[smooth[1:], 0.0])
        & (histogram > 0.0)
    )
    basis_u, basis_v = plane_basis(axes, axis_index)
    candidates: list[dict[str, Any]] = []
    for bin_index in local[np.argsort(smooth[local])[::-1]]:
        initial = float((edges[bin_index] + edges[bin_index + 1]) * 0.5)
        if any(abs(initial - item["offset"]) < peak_separation for item in candidates):
            continue
        neighborhood = np.abs(projection - initial) <= max(
            plane_tolerance * 2.0,
            offset_bin_size * 2.0,
        )
        if not np.any(neighborhood):
            continue
        offset = weighted_median(projection[neighborhood], selected_areas[neighborhood])
        inlier = np.abs(projection - offset) <= plane_tolerance
        if not np.any(inlier):
            continue
        inlier_centers = centers[selected][inlier]
        inlier_areas = selected_areas[inlier]
        residual = projection[inlier] - offset
        u = row_dot(inlier_centers, basis_u)
        v = row_dot(inlier_centers, basis_v)
        bounds = (
            float(np.quantile(u, 0.002)),
            float(np.quantile(u, 0.998)),
            float(np.quantile(v, 0.002)),
            float(np.quantile(v, 0.998)),
        )
        candidates.append(
            {
                "offset": offset,
                "support_area": float(inlier_areas.sum()),
                "inlier_faces": int(inlier.sum()),
                "rms_residual": float(
                    np.sqrt(np.average(np.square(residual), weights=inlier_areas))
                ),
                "bounds_uv": bounds,
            }
        )
        if len(candidates) >= maximum_peaks:
            break
    return sorted(candidates, key=lambda item: item["offset"])


def annotate_candidates_with_anchors(
    axes: np.ndarray,
    candidates_by_axis: list[list[dict[str, Any]]],
    anchors: dict[str, np.ndarray],
    *,
    assignment_tolerance: float,
) -> dict[str, Any]:
    compatibility: dict[str, dict[str, Any]] = {}
    for role, points in anchors.items():
        coordinates = row_transform(points, axes.T)
        spans = np.quantile(coordinates, 0.998, axis=0) - np.quantile(
            coordinates,
            0.002,
            axis=0,
        )
        order = np.argsort(spans)
        planar = bool(spans[order[0]] <= max(spans[order[1]] * 0.25, 1e-6))
        compatible_axes = [int(order[0])] if planar else [0, 1, 2]
        compatibility[role] = {
            "projection_spans": spans.tolist(),
            "planar_anchor": planar,
            "compatible_axis_indices": compatible_axes,
        }
    for axis_index, candidates in enumerate(candidates_by_axis):
        axis = axes[axis_index]
        projected = {role: row_dot(points, axis) for role, points in anchors.items()}
        for candidate in candidates:
            support: dict[str, dict[str, Any]] = {}
            for role, values in projected.items():
                residual = np.abs(values - float(candidate["offset"]))
                axis_compatible = axis_index in compatibility[role]["compatible_axis_indices"]
                inlier = (residual <= assignment_tolerance) if axis_compatible else np.zeros(
                    len(residual),
                    dtype=bool,
                )
                support[role] = {
                    "inliers": int(inlier.sum()),
                    "fraction": float(inlier.mean()),
                    "axis_compatible": axis_compatible,
                    "median_abs_residual": (
                        float(np.median(residual[inlier])) if np.any(inlier) else None
                    ),
                }
            candidate["anchor_support"] = support
            candidate["anchor_score"] = float(
                sum(float(item["fraction"]) for item in support.values())
            )
    return compatibility


def select_envelope_planes(
    axes: np.ndarray,
    candidates_by_axis: list[list[dict[str, Any]]],
    camera_centers: np.ndarray,
    anchors: dict[str, np.ndarray],
    *,
    minimum_relative_support: float,
    camera_clearance: float,
    extent_margin: float,
    minimum_anchor_fraction: float,
    anchor_assignment_tolerance: float,
) -> list[Plane]:
    planes: list[Plane] = []
    for axis_index, candidates in enumerate(candidates_by_axis):
        if not candidates:
            continue
        max_support = max(item["support_area"] for item in candidates)
        structural = [
            item
            for item in candidates
            if item["support_area"] >= max_support * minimum_relative_support
        ]
        camera_coordinate = row_dot(camera_centers, axes[axis_index])
        side_candidates = {
            "negative": [
                item
                for item in structural
                if item["offset"] < float(camera_coordinate.min()) - camera_clearance
            ],
            "positive": [
                item
                for item in structural
                if item["offset"] > float(camera_coordinate.max()) + camera_clearance
            ],
        }
        basis_u, basis_v = plane_basis(axes, axis_index)
        for side, eligible in side_candidates.items():
            if not eligible:
                continue
            anchored = [
                item
                for item in eligible
                if float(item.get("anchor_score", 0.0)) >= minimum_anchor_fraction
            ]
            winner = max(
                anchored or eligible,
                key=lambda item: (
                    float(item.get("anchor_score", 0.0)),
                    float(item["support_area"]),
                ),
            )
            tsdf_offset = float(winner["offset"])
            anchor_support = winner.get("anchor_support", {})
            anchor_role = None
            anchor_inliers = 0
            anchor_fraction = 0.0
            offset = tsdf_offset
            anchor_points: np.ndarray | None = None
            if isinstance(anchor_support, dict) and anchor_support:
                anchor_role, best = max(
                    anchor_support.items(),
                    key=lambda item: float(item[1]["fraction"]),
                )
                anchor_fraction = float(best["fraction"])
                anchor_inliers = int(best["inliers"])
                if anchor_fraction >= minimum_anchor_fraction:
                    values = row_dot(anchors[anchor_role], axes[axis_index])
                    inlier = np.abs(values - tsdf_offset) <= anchor_assignment_tolerance
                    anchor_points = anchors[anchor_role][inlier]
                    offset = float(np.median(values[inlier]))
                else:
                    anchor_role = None
                    anchor_inliers = 0
                    anchor_fraction = 0.0
            u0, u1, v0, v1 = winner["bounds_uv"]
            if anchor_points is not None and len(anchor_points):
                anchor_u = row_dot(anchor_points, basis_u)
                anchor_v = row_dot(anchor_points, basis_v)
                u0 = min(u0, float(np.quantile(anchor_u, 0.002)))
                u1 = max(u1, float(np.quantile(anchor_u, 0.998)))
                v0 = min(v0, float(np.quantile(anchor_v, 0.002)))
                v1 = max(v1, float(np.quantile(anchor_v, 0.998)))
            planes.append(
                Plane(
                    plane_id=len(planes) + 1,
                    axis_index=axis_index,
                    side=side,
                    semantic_role=anchor_role or "structural_boundary",
                    normal=axes[axis_index].copy(),
                    offset=offset,
                    basis_u=basis_u.copy(),
                    basis_v=basis_v.copy(),
                    bounds_uv=(
                        u0 - extent_margin,
                        u1 + extent_margin,
                        v0 - extent_margin,
                        v1 + extent_margin,
                    ),
                    support_area=float(winner["support_area"]),
                    inlier_faces=int(winner["inlier_faces"]),
                    rms_residual=float(winner["rms_residual"]),
                    tsdf_offset=tsdf_offset,
                    anchor_role=anchor_role,
                    anchor_inliers=anchor_inliers,
                    anchor_fraction=anchor_fraction,
                )
            )
    if not planes:
        raise ValueError("no structural envelope planes bracket the target cameras")
    return planes


def rays_for_pixels(
    x: np.ndarray,
    y: np.ndarray,
    intrinsic: dict[str, float],
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    camera_rays = np.stack(
        [
            (x - intrinsic["cx"]) / intrinsic["fx"],
            (y - intrinsic["cy"]) / intrinsic["fy"],
            np.ones_like(x, dtype=np.float64),
        ],
        axis=1,
    )
    world_rays = row_transform(camera_rays, matrix[:3, :3])
    return world_rays, camera_center(matrix)


def intersect_envelope(
    world_rays: np.ndarray,
    center: np.ndarray,
    planes: list[Plane],
    *,
    minimum_depth: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(world_rays)
    candidate_depth = np.full((len(planes), count), np.inf, dtype=np.float64)
    candidate_points = np.full((len(planes), count, 3), np.nan, dtype=np.float64)
    for index, plane in enumerate(planes):
        denominator = row_dot(world_rays, plane.normal)
        valid = np.abs(denominator) > 1e-9
        depth = np.full(count, np.inf, dtype=np.float64)
        depth[valid] = (plane.offset - float(center @ plane.normal)) / denominator[valid]
        valid &= np.isfinite(depth) & (depth > minimum_depth)
        points = center[None, :] + world_rays * depth[:, None]
        u = row_dot(points, plane.basis_u)
        v = row_dot(points, plane.basis_v)
        u0, u1, v0, v1 = plane.bounds_uv
        valid &= (u >= u0) & (u <= u1) & (v >= v0) & (v <= v1)
        candidate_depth[index, valid] = depth[valid]
        candidate_points[index, valid] = points[valid]
    winner = np.argmin(candidate_depth, axis=0)
    depth = candidate_depth[winner, np.arange(count)]
    valid = np.isfinite(depth)
    plane_ids = np.zeros(count, dtype=np.uint16)
    plane_ids[valid] = np.asarray([planes[index].plane_id for index in winner[valid]])
    points = np.full((count, 3), np.nan, dtype=np.float64)
    points[valid] = candidate_points[winner[valid], np.flatnonzero(valid)]
    depth[~valid] = np.nan
    return plane_ids, depth, points


def atlas_shape(
    plane: Plane,
    requested_texel_size: float,
    maximum_dimension: int,
) -> tuple[float, int, int]:
    u0, u1, v0, v1 = plane.bounds_uv
    texel_size = requested_texel_size
    width = max(1, math.ceil((u1 - u0) / texel_size))
    height = max(1, math.ceil((v1 - v0) / texel_size))
    scale = max(width / maximum_dimension, height / maximum_dimension, 1.0)
    texel_size *= scale
    width = max(1, math.ceil((u1 - u0) / texel_size))
    height = max(1, math.ceil((v1 - v0) / texel_size))
    return texel_size, width, height


def build_texture_atlases(
    records: list[dict[str, Any]],
    *,
    manifest_dir: Path,
    frames_override: Path | None,
    masks_override: Path | None,
    frame_record_key: str,
    mask_record_key: str,
    depth_dir: Path,
    camera_info: dict[str, Any],
    planes: list[Plane],
    sample_stride: int,
    plane_tolerance: float,
    texel_size: float,
    maximum_dimension: int,
    minimum_frame_support: int,
    exclusion_mask_paths: dict[str, Path] | None = None,
) -> tuple[dict[int, TextureAtlas], list[dict[str, Any]]]:
    accumulators: dict[int, dict[str, np.ndarray | float | tuple[float, float]]] = {}
    for plane in planes:
        actual_texel, width, height = atlas_shape(plane, texel_size, maximum_dimension)
        u0, _, v0, _ = plane.bounds_uv
        accumulators[plane.plane_id] = {
            "origin": (u0, v0),
            "texel_size": actual_texel,
            "color_sum": np.zeros((height, width, 3), dtype=np.float64),
            "frame_support": np.zeros((height, width), dtype=np.uint16),
        }
    donor_records: list[dict[str, Any]] = []
    for record in records:
        frame_id = str(record["frame_id"])
        frame_path = record_asset_path(
            record,
            frame_record_key,
            manifest_dir=manifest_dir,
            override_dir=frames_override,
        )
        mask_path = (
            exclusion_mask_paths[frame_id]
            if exclusion_mask_paths is not None
            else record_asset_path(
                record,
                mask_record_key,
                manifest_dir=manifest_dir,
                override_dir=masks_override,
            )
        )
        depth_path = (depth_dir / f"{frame_id}.npy").resolve()
        if not depth_path.is_file():
            raise FileNotFoundError(depth_path)
        rgb = image_rgb(frame_path)
        removal = mask_bool(mask_path)
        depth = np.load(depth_path)
        if depth.shape != rgb.shape[:2] or removal.shape != depth.shape:
            raise ValueError(f"RGB/depth/mask shape mismatch for donor {frame_id}")
        intrinsic = frame_intrinsic(camera_info, frame_id)
        matrix = world_to_camera(camera_info, frame_id)
        y, x = np.mgrid[0 : depth.shape[0] : sample_stride, 0 : depth.shape[1] : sample_stride]
        sampled_depth = depth[::sample_stride, ::sample_stride]
        sampled_mask = removal[::sample_stride, ::sample_stride]
        valid = np.isfinite(sampled_depth) & (sampled_depth > 0.0) & ~sampled_mask
        x = x[valid].astype(np.float64, copy=False)
        y = y[valid].astype(np.float64, copy=False)
        sampled_depth = sampled_depth[valid].astype(np.float64, copy=False)
        colors = rgb[::sample_stride, ::sample_stride][valid]
        camera_points = np.stack(
            [
                (x - intrinsic["cx"]) * sampled_depth / intrinsic["fx"],
                (y - intrinsic["cy"]) * sampled_depth / intrinsic["fy"],
                sampled_depth,
            ],
            axis=1,
        )
        world_points = row_transform(camera_points - matrix[:3, 3], matrix[:3, :3])
        per_plane: dict[str, int] = {}
        for plane in planes:
            accumulator = accumulators[plane.plane_id]
            origin_u, origin_v = accumulator["origin"]  # type: ignore[misc]
            actual_texel = float(accumulator["texel_size"])
            color_sum = accumulator["color_sum"]
            frame_support = accumulator["frame_support"]
            assert isinstance(color_sum, np.ndarray)
            assert isinstance(frame_support, np.ndarray)
            distance = np.abs(row_dot(world_points, plane.normal) - plane.offset)
            near = distance <= plane_tolerance
            u = row_dot(world_points, plane.basis_u)
            v = row_dot(world_points, plane.basis_v)
            column = np.floor((u - origin_u) / actual_texel).astype(np.int64)
            row = np.floor((v - origin_v) / actual_texel).astype(np.int64)
            inside = (
                near
                & (column >= 0)
                & (column < color_sum.shape[1])
                & (row >= 0)
                & (row < color_sum.shape[0])
            )
            if not np.any(inside):
                per_plane[str(plane.plane_id)] = 0
                continue
            flat = row[inside] * color_sum.shape[1] + column[inside]
            unique, inverse = np.unique(flat, return_inverse=True)
            counts = np.bincount(inverse)
            frame_color = np.column_stack(
                [np.bincount(inverse, weights=colors[inside, channel]) for channel in range(3)]
            ) / counts[:, None]
            unique_row = unique // color_sum.shape[1]
            unique_column = unique % color_sum.shape[1]
            color_sum[unique_row, unique_column] += frame_color
            frame_support[unique_row, unique_column] += 1
            per_plane[str(plane.plane_id)] = len(unique)
        donor_records.append(
            {
                "frame_id": frame_id,
                "frame": str(frame_path),
                "frame_sha256": sha256_file(frame_path),
                "mask": str(mask_path),
                "mask_sha256": sha256_file(mask_path),
                "mask_role": (
                    "front_to_back_cumulative_removal_union"
                    if exclusion_mask_paths is not None
                    else "configured_texture_exclusion"
                ),
                "depth": str(depth_path),
                "depth_sha256": sha256_file(depth_path),
                "depth_role": "measured_texture_visibility_evidence_only",
                "atlas_texels_contributed": per_plane,
            }
        )
    atlases: dict[int, TextureAtlas] = {}
    for plane in planes:
        accumulator = accumulators[plane.plane_id]
        color_sum = accumulator["color_sum"]
        frame_support = accumulator["frame_support"]
        assert isinstance(color_sum, np.ndarray)
        assert isinstance(frame_support, np.ndarray)
        divisor = np.maximum(frame_support, 1)[..., None]
        color = np.clip(np.rint(color_sum / divisor), 0, 255).astype(np.uint8)
        atlases[plane.plane_id] = TextureAtlas(
            plane_id=plane.plane_id,
            origin_uv=accumulator["origin"],  # type: ignore[arg-type]
            texel_size=float(accumulator["texel_size"]),
            color=color,
            frame_support=frame_support,
            observed=frame_support >= minimum_frame_support,
        )
    return atlases, donor_records


def lookup_atlas(
    atlas: TextureAtlas,
    plane: Plane,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    u = row_dot(points, plane.basis_u)
    v = row_dot(points, plane.basis_v)
    column = np.floor((u - atlas.origin_uv[0]) / atlas.texel_size).astype(np.int64)
    row = np.floor((v - atlas.origin_uv[1]) / atlas.texel_size).astype(np.int64)
    inside = (
        np.isfinite(points).all(axis=1)
        & (column >= 0)
        & (column < atlas.color.shape[1])
        & (row >= 0)
        & (row < atlas.color.shape[0])
    )
    observed = np.zeros(len(points), dtype=bool)
    colors = np.zeros((len(points), 3), dtype=np.uint8)
    observed[inside] = atlas.observed[row[inside], column[inside]]
    colors[inside] = atlas.color[row[inside], column[inside]]
    return colors, observed


def write_patch_mesh(
    planes: list[Plane],
    occupied_uv: dict[int, set[tuple[int, int]]],
    path: Path,
    grid_size: float,
) -> dict[str, int]:
    try:
        from plyfile import PlyData, PlyElement
    except ImportError as error:  # pragma: no cover - environment contract
        raise RuntimeError("install video2world[geometry] to write PLY patches") from error

    palette = np.asarray(
        [[70, 170, 255], [255, 184, 77], [108, 214, 126], [219, 111, 255], [255, 105, 97]],
        dtype=np.uint8,
    )
    vertices: list[tuple[float, float, float, int, int, int, int]] = []
    faces: list[tuple[np.ndarray, int]] = []
    for plane in planes:
        color = palette[(plane.plane_id - 1) % len(palette)]
        origin = plane.normal * plane.offset
        for cell_u, cell_v in sorted(occupied_uv.get(plane.plane_id, set())):
            u0 = cell_u * grid_size
            v0 = cell_v * grid_size
            corners = (
                origin + plane.basis_u * u0 + plane.basis_v * v0,
                origin + plane.basis_u * (u0 + grid_size) + plane.basis_v * v0,
                origin
                + plane.basis_u * (u0 + grid_size)
                + plane.basis_v * (v0 + grid_size),
                origin + plane.basis_u * u0 + plane.basis_v * (v0 + grid_size),
            )
            start = len(vertices)
            vertices.extend(
                (
                    float(point[0]),
                    float(point[1]),
                    float(point[2]),
                    int(color[0]),
                    int(color[1]),
                    int(color[2]),
                    plane.plane_id,
                )
                for point in corners
            )
            faces.append(
                (np.asarray([start, start + 1, start + 2], dtype=np.int32), plane.plane_id)
            )
            faces.append(
                (np.asarray([start, start + 2, start + 3], dtype=np.int32), plane.plane_id)
            )
    if not vertices:
        raise ValueError("no occupied planar cells were generated")
    vertex_dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("plane_id", "u2"),
    ]
    face_dtype = [("vertex_indices", "i4", (3,)), ("plane_id", "u2")]
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData(
        [
            PlyElement.describe(np.asarray(vertices, dtype=vertex_dtype), "vertex"),
            PlyElement.describe(np.asarray(faces, dtype=face_dtype), "face"),
        ],
        text=False,
    ).write(str(path))
    return {"vertices": len(vertices), "faces": len(faces)}


def make_contact_sheet(records: list[dict[str, Any]], path: Path, samples: int) -> None:
    count = min(samples, len(records))
    indices = np.linspace(0, len(records) - 1, count, dtype=int)
    width, height, label = 288, 162, 22
    sheet = Image.new("RGB", (width * 5, (height + label) * count), "#111111")
    draw = ImageDraw.Draw(sheet)
    for row_index, record_index in enumerate(indices):
        record = records[int(record_index)]
        source = Image.open(record["source_frame"]).convert("RGB")
        removal = Image.open(record["removal_mask"]).convert("L")
        labels = Image.open(record["plane_labels"]).convert("RGB")
        prefill = Image.open(record["prefill_frame"]).convert("RGB")
        generated = Image.open(record["generative_texture_mask"]).convert("L")
        removal_overlay = source.copy()
        removal_overlay.paste(Image.new("RGB", source.size, (255, 0, 170)), mask=removal)
        generation_overlay = prefill.copy()
        generation_overlay.paste(Image.new("RGB", source.size, (255, 65, 40)), mask=generated)
        images = (source, removal_overlay, labels, prefill, generation_overlay)
        titles = (
            f"source {record['frame_id']}",
            "object removal mask",
            "TSDF envelope plane ID",
            f"measured texture {record['observed_texture_fraction']:.1%}",
            "generation-only boundary",
        )
        y_offset = row_index * (height + label)
        for column, (image, title) in enumerate(zip(images, titles, strict=True)):
            image.thumbnail((width, height), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (width, height), "#202020")
            canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
            x_offset = column * width
            sheet.paste(canvas, (x_offset, y_offset + label))
            draw.text((x_offset + 6, y_offset + 5), title, fill="#f4f4f4")
        for image in images:
            image.close()
        removal.close()
        generated.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def plane_to_json(plane: Plane) -> dict[str, Any]:
    return {
        "plane_id": plane.plane_id,
        "axis_index": plane.axis_index,
        "side": plane.side,
        "semantic_role": plane.semantic_role,
        "normal": plane.normal.tolist(),
        "offset": plane.offset,
        "equation": "normal dot world_xyz = offset",
        "basis_u": plane.basis_u.tolist(),
        "basis_v": plane.basis_v.tolist(),
        "bounds_uv": list(plane.bounds_uv),
        "support_area_sampled": plane.support_area,
        "inlier_faces_sampled": plane.inlier_faces,
        "rms_plane_residual": plane.rms_residual,
        "tsdf_candidate_offset": plane.tsdf_offset,
        "anchor_role": plane.anchor_role,
        "anchor_inliers_sampled": plane.anchor_inliers,
        "anchor_fraction_sampled": plane.anchor_fraction,
        "anchor_offset_correction": plane.offset - plane.tsdf_offset,
    }


def build_planar_completion(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.input_manifest.expanduser().resolve()
    camera_path = args.camera_info.expanduser().resolve()
    mesh_path = args.mesh.expanduser().resolve()
    depth_dir = args.depth_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    frames_override = args.frames_dir.expanduser().resolve() if args.frames_dir else None
    masks_override = args.masks_dir.expanduser().resolve() if args.masks_dir else None
    texture_frames_override = (
        args.texture_frames_dir.expanduser().resolve()
        if args.texture_frames_dir
        else frames_override
    )
    texture_masks_override = (
        args.texture_masks_dir.expanduser().resolve()
        if args.texture_masks_dir
        else masks_override
    )
    original_removal_masks_dir = (
        args.original_removal_masks_dir.expanduser().resolve()
        if args.original_removal_masks_dir
        else texture_masks_override
    )
    supplement_mask_dirs = [
        path.expanduser().resolve() for path in (args.supplement_mask_dir or [])
    ]
    supplement_manifest_paths = [
        path.expanduser().resolve() for path in (args.supplement_mask_manifest or [])
    ]
    protected_manifest_paths = [
        path.expanduser().resolve() for path in (args.protected_mask_manifest or [])
    ]
    protected_sam_mask_root = (
        args.protected_sam_mask_root.expanduser().resolve()
        if args.protected_sam_mask_root
        else None
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("frame_records"), list):
        raise ValueError("input manifest must contain frame_records")
    records = list(manifest["frame_records"])
    if args.target_frame_ids:
        requested = {item.strip() for item in args.target_frame_ids.split(",") if item.strip()}
        records = [record for record in records if str(record.get("frame_id")) in requested]
        if {str(record.get("frame_id")) for record in records} != requested:
            raise ValueError("one or more requested target frames are absent")
    if not records:
        raise ValueError("no target frames selected")
    supplement_manifest_lineage, supplement_manifest_assets = (
        load_supplement_mask_manifests(supplement_manifest_paths, records)
    )
    protected_manifest_lineage, protected_manifest_assets = (
        load_supplement_mask_manifests(protected_manifest_paths, records)
    )
    protected_sam_mask_set, protected_sam_assets = build_protected_sam_mask_assets(
        protected_sam_mask_root,
        args.protected_sam_label or [],
        records,
        dilation=args.protected_sam_mask_dilation,
        output_dir=output / "protected_neighbor_masks",
    )
    for frame_id, assets in protected_sam_assets.items():
        protected_manifest_assets[frame_id].extend(assets)
    target_frame_key = select_record_key(
        records,
        args.frame_record_key,
        ("prefill_frame", "input_frame", "source_frame"),
    )
    target_mask_key = select_record_key(
        records,
        args.mask_record_key,
        ("residual_mask", "input_mask", "removal_mask", "union_mask"),
    )
    texture_frame_key = select_record_key(
        records,
        args.texture_frame_record_key,
        ("source_frame", "input_frame", "prefill_frame"),
    )
    texture_mask_key = select_record_key(
        records,
        args.texture_mask_record_key,
        ("removal_mask", "input_mask", "residual_mask", "union_mask"),
    )
    upstream_path = (
        args.upstream_prefill_report.expanduser().resolve()
        if args.upstream_prefill_report
        else (
            manifest_path
            if target_frame_key == "prefill_frame" and target_mask_key == "residual_mask"
            else None
        )
    )
    upstream_prefill = (
        validate_upstream_prefill(upstream_path, records) if upstream_path else None
    )
    cumulative_inputs = prepare_cumulative_removal_inputs(
        records,
        manifest_dir=manifest_path.parent,
        target_mask_key=target_mask_key,
        masks_override=masks_override,
        original_removal_masks_dir=original_removal_masks_dir,
        supplement_mask_dirs=supplement_mask_dirs,
        supplement_manifest_assets=supplement_manifest_assets,
        cumulative_mask_dilation=args.cumulative_mask_dilation,
        texture_donor_exclusion_dilation=args.texture_donor_exclusion_dilation,
        output_dir=output / "cumulative_masks",
        donor_exclusion_dir=output / "texture_donor_exclusion_masks",
    )
    camera_info = load_camera_info(camera_path)
    anchor_paths = parse_structure_anchor_specs(args.structure_anchor)
    anchors, anchor_records = load_structure_anchors(
        anchor_paths,
        args.maximum_anchor_points,
    )
    camera_centers = np.stack(
        [camera_center(world_to_camera(camera_info, str(record["frame_id"]))) for record in records]
    )
    vertices, faces = load_mesh_triangles(mesh_path, args.maximum_sampled_faces)
    centers, normals, areas = face_geometry(vertices, faces)
    axes, axis_diagnostics = fit_manhattan_axes(
        normals,
        areas,
        normal_bucket_size=args.normal_bucket_size,
        angular_tolerance_degrees=args.normal_angle_tolerance,
    )
    candidates_by_axis = [
        fit_axis_plane_candidates(
            centers,
            normals,
            areas,
            axes,
            axis_index,
            angular_tolerance_degrees=args.normal_angle_tolerance,
            offset_bin_size=args.offset_bin_size,
            plane_tolerance=args.plane_fit_tolerance,
            peak_separation=args.peak_separation,
            maximum_peaks=args.maximum_peaks_per_axis,
        )
        for axis_index in range(3)
    ]
    anchor_axis_diagnostics = annotate_candidates_with_anchors(
        axes,
        candidates_by_axis,
        anchors,
        assignment_tolerance=args.anchor_assignment_tolerance,
    )
    planes = select_envelope_planes(
        axes,
        candidates_by_axis,
        camera_centers,
        anchors,
        minimum_relative_support=args.minimum_relative_plane_support,
        camera_clearance=args.camera_clearance,
        extent_margin=args.plane_extent_margin,
        minimum_anchor_fraction=args.minimum_anchor_fraction,
        anchor_assignment_tolerance=args.anchor_assignment_tolerance,
    )
    structural_shadow_collar = add_structural_shadow_collars(
        cumulative_inputs,
        records,
        radius=args.structural_shadow_collar_radius,
        plane_tolerance=args.structural_shadow_plane_tolerance,
        texture_donor_exclusion_dilation=args.texture_donor_exclusion_dilation,
        depth_dir=depth_dir,
        camera_info=camera_info,
        planes=planes,
        protected_manifest_assets=protected_manifest_assets,
    )
    atlases, donor_records = build_texture_atlases(
        records,
        manifest_dir=manifest_path.parent,
        frames_override=texture_frames_override,
        masks_override=texture_masks_override,
        frame_record_key=texture_frame_key,
        mask_record_key=texture_mask_key,
        depth_dir=depth_dir,
        camera_info=camera_info,
        planes=planes,
        sample_stride=args.texture_sample_stride,
        plane_tolerance=args.texture_plane_tolerance,
        texel_size=args.texture_texel_size,
        maximum_dimension=args.maximum_atlas_dimension,
        minimum_frame_support=args.minimum_texture_frame_support,
        exclusion_mask_paths={
            frame_id: value["donor_exclusion_mask_path"]
            for frame_id, value in cumulative_inputs.items()
        },
    )
    atlas_dir = output / "atlases"
    atlas_dir.mkdir(parents=True, exist_ok=True)
    atlas_records: list[dict[str, Any]] = []
    for plane in planes:
        atlas = atlases[plane.plane_id]
        color_path = atlas_dir / f"plane_{plane.plane_id:02d}_measured_rgb.png"
        support_path = atlas_dir / f"plane_{plane.plane_id:02d}_frame_support.png"
        observed_path = atlas_dir / f"plane_{plane.plane_id:02d}_observed_mask.png"
        Image.fromarray(atlas.color).save(color_path)
        support_scale = 255.0 / max(int(atlas.frame_support.max(initial=0)), 1)
        Image.fromarray(
            np.clip(np.rint(atlas.frame_support * support_scale), 0, 255).astype(np.uint8)
        ).save(support_path)
        Image.fromarray(atlas.observed.astype(np.uint8) * 255).save(observed_path)
        atlas_records.append(
            {
                "plane_id": plane.plane_id,
                "measured_rgb": str(color_path),
                "measured_rgb_sha256": sha256_file(color_path),
                "frame_support": str(support_path),
                "observed_mask": str(observed_path),
                "observed_mask_sha256": sha256_file(observed_path),
                "dimensions": [atlas.color.shape[1], atlas.color.shape[0]],
                "origin_uv": list(atlas.origin_uv),
                "texel_size_world_units": atlas.texel_size,
                "observed_texels": int(atlas.observed.sum()),
                "total_texels": int(atlas.observed.size),
            }
        )
    directories = {
        name: output / name
        for name in (
            "frames",
            "geometry_depth",
            "plane_ids",
            "plane_labels",
            "observed_texture_masks",
            "generative_texture_masks",
            "unresolved_geometry_masks",
            "cumulative_masks",
        )
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    palette = np.asarray(
        [
            [0, 0, 0],
            [70, 170, 255],
            [255, 184, 77],
            [108, 214, 126],
            [219, 111, 255],
            [255, 105, 97],
        ],
        dtype=np.uint8,
    )
    output_records: list[dict[str, Any]] = []
    occupied_uv: dict[int, set[tuple[int, int]]] = {
        plane.plane_id: set() for plane in planes
    }
    for output_index, record in enumerate(records):
        frame_id = str(record["frame_id"])
        frame_path = record_asset_path(
            record,
            target_frame_key,
            manifest_dir=manifest_path.parent,
            override_dir=frames_override,
        )
        cumulative_input = cumulative_inputs[frame_id]
        base_mask_path = cumulative_input["base_mask_path"]
        original_mask_path = cumulative_input["original_mask_path"]
        supplement_asset_records = cumulative_input["supplement_asset_records"]
        supplement_paths = cumulative_input["supplement_paths"]
        source = image_rgb(frame_path)
        supplement_masks = cumulative_input["supplement_masks"]
        removal = cumulative_input["removal"]
        original_union = cumulative_input["original_union"]
        upstream_kept = cumulative_input["upstream_kept"]
        cumulative_receipt = cumulative_input["receipt"]
        height, width = source.shape[:2]
        if removal.shape != (height, width):
            raise ValueError(f"source/mask shape mismatch for target {frame_id}")
        intrinsic = frame_intrinsic(camera_info, frame_id)
        if (int(intrinsic["w"]), int(intrinsic["h"])) != (width, height):
            raise ValueError(f"camera/image shape mismatch for target {frame_id}")
        matrix = world_to_camera(camera_info, frame_id)
        y, x = np.nonzero(removal)
        world_rays, center = rays_for_pixels(
            x.astype(np.float64), y.astype(np.float64), intrinsic, matrix
        )
        plane_ids, depth, points = intersect_envelope(world_rays, center, planes)
        observed = np.zeros(len(x), dtype=bool)
        measured_color = np.zeros((len(x), 3), dtype=np.uint8)
        for plane in planes:
            selected = plane_ids == plane.plane_id
            if not np.any(selected):
                continue
            colors, supported = lookup_atlas(atlases[plane.plane_id], plane, points[selected])
            measured_color[selected] = colors
            observed[selected] = supported
            u = row_dot(points[selected], plane.basis_u)
            v = row_dot(points[selected], plane.basis_v)
            finite = np.isfinite(u) & np.isfinite(v)
            cells = np.column_stack(
                [
                    np.floor(u[finite] / args.geometry_grid_size).astype(np.int64),
                    np.floor(v[finite] / args.geometry_grid_size).astype(np.int64),
                ]
            )
            occupied_uv[plane.plane_id].update(map(tuple, np.unique(cells, axis=0)))
        assigned = plane_ids > 0
        observed &= assigned
        generative = assigned & ~observed
        unresolved = ~assigned
        result = source.copy()
        result[y[observed], x[observed]] = measured_color[observed]
        plane_image = np.zeros((height, width), dtype=np.uint16)
        plane_image[y, x] = plane_ids
        depth_image = np.full((height, width), np.nan, dtype=np.float32)
        depth_image[y, x] = depth.astype(np.float32)
        observed_image = np.zeros((height, width), dtype=bool)
        observed_image[y, x] = observed
        generative_image = np.zeros((height, width), dtype=bool)
        generative_image[y, x] = generative
        unresolved_image = np.zeros((height, width), dtype=bool)
        unresolved_image[y, x] = unresolved
        label_image = palette[np.minimum(plane_image, len(palette) - 1)]
        name = f"{output_index:04d}"
        result_path = directories["frames"] / f"{name}.png"
        depth_path = directories["geometry_depth"] / f"{name}.npz"
        plane_path = directories["plane_ids"] / f"{name}.png"
        label_path = directories["plane_labels"] / f"{name}.png"
        observed_path = directories["observed_texture_masks"] / f"{name}.png"
        generative_path = directories["generative_texture_masks"] / f"{name}.png"
        unresolved_path = directories["unresolved_geometry_masks"] / f"{name}.png"
        cumulative_mask_path = cumulative_input["cumulative_mask_path"]
        Image.fromarray(result).save(result_path)
        np.savez_compressed(depth_path, depth=depth_image)
        Image.fromarray(plane_image).save(plane_path)
        Image.fromarray(label_image).save(label_path)
        Image.fromarray(observed_image.astype(np.uint8) * 255).save(observed_path)
        Image.fromarray(generative_image.astype(np.uint8) * 255).save(generative_path)
        Image.fromarray(unresolved_image.astype(np.uint8) * 255).save(unresolved_path)
        removal_count = max(int(removal.sum()), 1)
        original_removal_pixels = int(original_union.sum())
        upstream_measured_pixels = int(upstream_kept.sum())
        final_measured_pixels = upstream_measured_pixels + int(observed.sum())
        original_partition_exact = (
            final_measured_pixels + int(generative.sum()) + int(unresolved.sum())
            == original_removal_pixels
        )
        partition_exact = bool(
            np.array_equal(observed_image | generative_image | unresolved_image, removal)
            and not np.any(observed_image & generative_image)
            and not np.any(observed_image & unresolved_image)
            and not np.any(generative_image & unresolved_image)
        )
        outside_exact = bool(np.array_equal(result[~removal], source[~removal]))
        output_records.append(
            {
                "sequence_index": output_index,
                "frame_id": frame_id,
                "source_frame": str(frame_path),
                "source_frame_sha256": sha256_file(frame_path),
                "base_residual_mask": str(base_mask_path),
                "base_residual_mask_sha256": sha256_file(base_mask_path),
                "original_removal_mask": str(original_mask_path),
                "original_removal_mask_sha256": sha256_file(original_mask_path),
                "supplement_masks": [
                    {
                        **{
                            key: value
                            for key, value in asset.items()
                            if key != "path" and not isinstance(value, Path)
                        },
                        "path": str(path),
                        "sha256": sha256_file(path),
                        "pixels": int(mask.sum()),
                    }
                    for asset, path, mask in zip(
                        supplement_asset_records,
                        supplement_paths,
                        supplement_masks,
                        strict=True,
                    )
                ],
                "cumulative_mask_receipt": cumulative_receipt,
                "removal_mask": str(cumulative_mask_path),
                "removal_mask_sha256": sha256_file(cumulative_mask_path),
                "removal_mask_pixels": int(removal.sum()),
                "texture_donor_exclusion_mask": str(
                    cumulative_input["donor_exclusion_mask_path"]
                ),
                "texture_donor_exclusion_mask_sha256": sha256_file(
                    cumulative_input["donor_exclusion_mask_path"]
                ),
                "texture_donor_exclusion_pixels": int(
                    cumulative_input["donor_exclusion"].sum()
                ),
                "texture_donor_exclusion_receipt": cumulative_input[
                    "donor_exclusion_receipt"
                ],
                "prefill_frame": str(result_path),
                "prefill_frame_sha256": sha256_file(result_path),
                "geometry_depth": str(depth_path),
                "geometry_depth_sha256": sha256_file(depth_path),
                "plane_ids": str(plane_path),
                "plane_labels": str(label_path),
                "observed_texture_mask": str(observed_path),
                "generative_texture_mask": str(generative_path),
                "unresolved_geometry_mask": str(unresolved_path),
                "geometry_assigned_pixels": int(assigned.sum()),
                "geometry_coverage_fraction": float(assigned.sum() / removal_count),
                "observed_texture_pixels": int(observed.sum()),
                "observed_texture_fraction": float(observed.sum() / removal_count),
                "generative_texture_pixels": int(generative.sum()),
                "generative_texture_fraction": float(generative.sum() / removal_count),
                "unresolved_geometry_pixels": int(unresolved.sum()),
                "original_removal_mask_pixels": original_removal_pixels,
                "upstream_measured_texture_pixels": upstream_measured_pixels,
                "final_measured_texture_pixels": final_measured_pixels,
                "final_measured_texture_fraction_of_original": (
                    final_measured_pixels / max(original_removal_pixels, 1)
                ),
                "synthetic_texture_pending_pixels": int(generative.sum()),
                "synthetic_texture_pending_fraction_of_original": (
                    int(generative.sum()) / max(original_removal_pixels, 1)
                ),
                "original_texture_partition_exact": original_partition_exact,
                "outside_removal_mask_rgb_exact": outside_exact,
                "texture_partition_exact": partition_exact,
                "plane_pixel_counts": {
                    str(plane.plane_id): int((plane_ids == plane.plane_id).sum())
                    for plane in planes
                },
            }
        )
    patch_path = output / "planar_completion_patch.ply"
    patch_counts = write_patch_mesh(
        planes,
        occupied_uv,
        patch_path,
        args.geometry_grid_size,
    )
    minimum_geometry = min(
        record["geometry_coverage_fraction"] for record in output_records
    )
    outside_exact = all(record["outside_removal_mask_rgb_exact"] for record in output_records)
    partitions_exact = all(record["texture_partition_exact"] for record in output_records)
    original_partitions_exact = all(
        record["original_texture_partition_exact"] for record in output_records
    )
    original_pixel_total = sum(
        record["original_removal_mask_pixels"] for record in output_records
    )
    measured_pixel_total = sum(
        record["final_measured_texture_pixels"] for record in output_records
    )
    synthetic_pixel_total = sum(
        record["synthetic_texture_pending_pixels"] for record in output_records
    )
    geometry_passed = minimum_geometry >= args.minimum_geometry_coverage
    status = "geometry_completed_texture_pending" if (
        geometry_passed and outside_exact and partitions_exact and original_partitions_exact
    ) else "technical_failed"
    support_plane = None
    floor_planes = [plane for plane in planes if plane.semantic_role == "floor"]
    if floor_planes:
        floor = max(floor_planes, key=lambda plane: plane.anchor_fraction)
        interior_sign = (
            1.0
            if float(np.mean(row_dot(camera_centers, floor.normal))) > floor.offset
            else -1.0
        )
        anchor_record = next(
            (record for record in anchor_records if record["role"] == floor.anchor_role),
            None,
        )
        support_plane = {
            "plane_id": floor.plane_id,
            "semantic_role": "floor",
            "normal": floor.normal.tolist(),
            "interior_normal": (floor.normal * interior_sign).tolist(),
            "offset": floor.offset,
            "equation": "normal dot world_xyz = offset",
            "source": "Holi structure anchor fixes offset; TSDF fixes Manhattan normal/topology",
            "anchor": anchor_record,
            "anchor_fraction_sampled": floor.anchor_fraction,
            "anchor_inliers_sampled": floor.anchor_inliers,
            "rms_tsdf_plane_residual": floor.rms_residual,
            "scene_fit_role": "gravity/support constraint for grounded object placement",
        }
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "purpose": "TSDF-fitted structural-plane completion after front-to-back object peeling",
        "status": status,
        "promotion_approved": False,
        "promotion_blocker": (
            "generative texture and cross-view RGB review remain required; geometry success alone "
            "cannot promote a final clean plate"
        ),
        "input_manifest": str(manifest_path),
        "input_manifest_sha256": sha256_file(manifest_path),
        "upstream_multiview_prefill": upstream_prefill,
        "camera_info": str(camera_path),
        "camera_info_sha256": sha256_file(camera_path),
        "camera_extrinsic_type": "world_to_camera",
        "tsdf_mesh": str(mesh_path),
        "tsdf_mesh_sha256": sha256_file(mesh_path),
        "mesh_sample": {
            "vertices": len(vertices),
            "faces_loaded": len(faces),
            "valid_faces": len(areas),
            "maximum_sampled_faces": args.maximum_sampled_faces,
        },
        "structure_anchors": anchor_records,
        "structure_anchor_axis_diagnostics": anchor_axis_diagnostics,
        "structure_anchor_mode": "preferred" if anchor_records else "tsdf_only_fallback",
        "supplement_mask_manifests": supplement_manifest_lineage,
        "protected_mask_manifests": protected_manifest_lineage,
        "protected_sam_mask_set": protected_sam_mask_set,
        "structural_shadow_collar": structural_shadow_collar,
        "manhattan_axes": axes.tolist(),
        "manhattan_diagnostics": axis_diagnostics,
        "plane_candidates": candidates_by_axis,
        "selected_envelope_planes": [plane_to_json(plane) for plane in planes],
        "support_plane": support_plane,
        "texture_atlases": atlas_records,
        "texture_coverage": {
            "denominator": "original removal-mask pixels across selected frames",
            "original_mask_pixels": original_pixel_total,
            "observed_pixels": measured_pixel_total,
            "observed_fraction": measured_pixel_total / max(original_pixel_total, 1),
            "synthetic_pending_pixels": synthetic_pixel_total,
            "synthetic_pending_fraction": synthetic_pixel_total
            / max(original_pixel_total, 1),
            "unresolved_geometry_pixels": sum(
                record["unresolved_geometry_pixels"] for record in output_records
            ),
        },
        "donor_records": donor_records,
        "planar_completion_patch": {
            "path": str(patch_path),
            "sha256": sha256_file(patch_path),
            **patch_counts,
            "role": "geometry-only hidden-surface patch; not final textured scene mesh",
        },
        "config": {
            "target_frame_record_key": target_frame_key,
            "target_mask_record_key": target_mask_key,
            "texture_frame_record_key": texture_frame_key,
            "texture_mask_record_key": texture_mask_key,
            "normal_angle_tolerance_degrees": args.normal_angle_tolerance,
            "offset_bin_size": args.offset_bin_size,
            "plane_fit_tolerance": args.plane_fit_tolerance,
            "texture_plane_tolerance": args.texture_plane_tolerance,
            "minimum_relative_plane_support": args.minimum_relative_plane_support,
            "camera_clearance": args.camera_clearance,
            "plane_extent_margin": args.plane_extent_margin,
            "texture_sample_stride": args.texture_sample_stride,
            "texture_texel_size": args.texture_texel_size,
            "minimum_texture_frame_support": args.minimum_texture_frame_support,
            "geometry_grid_size": args.geometry_grid_size,
            "minimum_anchor_fraction": args.minimum_anchor_fraction,
            "anchor_assignment_tolerance": args.anchor_assignment_tolerance,
            "original_removal_masks_dir": (
                str(original_removal_masks_dir) if original_removal_masks_dir else None
            ),
            "supplement_mask_dirs": [str(path) for path in supplement_mask_dirs],
            "supplement_mask_manifests": [
                str(path) for path in supplement_manifest_paths
            ],
            "protected_mask_manifests": [
                str(path) for path in protected_manifest_paths
            ],
            "protected_sam_mask_root": (
                str(protected_sam_mask_root) if protected_sam_mask_root else None
            ),
            "protected_sam_labels": args.protected_sam_label or [],
            "protected_sam_mask_dilation": args.protected_sam_mask_dilation,
            "cumulative_mask_dilation": args.cumulative_mask_dilation,
            "texture_donor_exclusion_dilation": (
                args.texture_donor_exclusion_dilation
            ),
            "structural_shadow_collar_radius": args.structural_shadow_collar_radius,
            "structural_shadow_plane_tolerance": (
                args.structural_shadow_plane_tolerance
            ),
        },
        "gates": {
            "manhattan_axes_orthonormal": axis_diagnostics["maximum_axis_dot_error"] <= 1e-5,
            "minimum_geometry_coverage": {
                "threshold": args.minimum_geometry_coverage,
                "observed": minimum_geometry,
                "passed": geometry_passed,
            },
            "outside_removal_mask_rgb_exact": outside_exact,
            "texture_masks_partition_removal_mask": partitions_exact,
            "upstream_observed_plus_residual_partition_original_mask": (
                original_partitions_exact
            ),
            "cumulative_supplement_union_partition_exact": original_partitions_exact,
            "all_texture_donors_exclude_cumulative_original_union": all(
                record["mask_role"] == "front_to_back_cumulative_removal_union"
                for record in donor_records
            ),
            "structural_shadow_collar_only_adds_plane_consistent_pixels": (
                structural_shadow_collar[
                    "all_added_pixels_are_depth_consistent_with_a_structural_plane"
                ]
            ),
            "protected_neighbor_masks_subtracted_only_from_shadow_collar": (
                structural_shadow_collar[
                    "protected_anchor_masks_subtracted_only_from_shadow_collar_additions"
                ]
            ),
            "generated_texture_present": False,
            "cross_view_rgb_continuity": "pending",
            "new_depth_normal_estimation_before_pgsr": "required_after_final_rgb_acceptance",
        },
        "texture_provenance_contract": {
            "measured_texture": (
                "only DA3-depth samples outside the full front-to-back cumulative original-union "
                "mask and within the fitted TSDF plane tolerance"
            ),
            "generative_texture": (
                "only generative_texture_masks/*.png; a later model may fill these pixels but "
                "must preserve plane geometry and report synthetic provenance"
            ),
            "forbidden": [
                "painting unresolved_geometry_masks as if geometry were known",
                "using ProPainter blur as structural evidence",
                "reusing old DA3 depth as final reconstructed depth after RGB generation",
            ],
        },
        "frame_records": output_records,
        "next_stage": {
            "texture_generation": (
                "fill only generative_texture_masks with geometry-conditioned, plane-coordinate "
                "consistent texture; keep measured texels and outside-mask RGB fixed"
            ),
            "validation": (
                "warp accepted texture across calibrated views, reject seams or plane-boundary "
                "drift, then rerun depth/normal estimation and PGSR/TSDF"
            ),
        },
        "limitations": [
            (
                "The method completes planar room structure, not curved or articulated hidden "
                "geometry."
            ),
            "A plane can be geometrically complete while its hidden texture remains unobserved.",
            "TSDF and DA3 errors are recorded separately and cannot certify final RGB quality.",
        ],
    }
    report_path = output / "planar_background_report.json"
    write_json(report_path, report)
    make_contact_sheet(
        output_records,
        output / "planar_background_contact_sheet.png",
        args.contact_sheet_samples,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--camera-info", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--depth-dir", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path)
    parser.add_argument("--masks-dir", type=Path)
    parser.add_argument("--texture-frames-dir", type=Path)
    parser.add_argument("--texture-masks-dir", type=Path)
    parser.add_argument("--original-removal-masks-dir", type=Path)
    parser.add_argument(
        "--supplement-mask-dir",
        type=Path,
        action="append",
        help="repeatable prior-layer residual mask directory unioned before planar completion",
    )
    parser.add_argument(
        "--supplement-mask-manifest",
        type=Path,
        action="append",
        help="repeatable authoritative frame_id -> cumulative union_mask manifest",
    )
    parser.add_argument(
        "--protected-mask-manifest",
        type=Path,
        action="append",
        help="repeatable protected-neighbor mask manifest subtracted only from shadow collar",
    )
    parser.add_argument("--protected-sam-mask-root", type=Path)
    parser.add_argument("--protected-sam-label", action="append")
    parser.add_argument("--protected-sam-mask-dilation", type=int, default=0)
    parser.add_argument("--cumulative-mask-dilation", type=int, default=0)
    parser.add_argument("--texture-donor-exclusion-dilation", type=int, default=0)
    parser.add_argument("--structural-shadow-collar-radius", type=int, default=0)
    parser.add_argument("--structural-shadow-plane-tolerance", type=float, default=0.35)
    parser.add_argument("--frame-record-key", default="auto")
    parser.add_argument("--mask-record-key", default="auto")
    parser.add_argument("--texture-frame-record-key", default="auto")
    parser.add_argument("--texture-mask-record-key", default="auto")
    parser.add_argument("--upstream-prefill-report", type=Path)
    parser.add_argument(
        "--structure-anchor",
        action="append",
        help="repeatable semantic structure anchor in role=/path/to/anchor.ply form",
    )
    parser.add_argument("--maximum-anchor-points", type=int, default=250_000)
    parser.add_argument("--minimum-anchor-fraction", type=float, default=0.01)
    parser.add_argument("--anchor-assignment-tolerance", type=float, default=0.35)
    parser.add_argument("--target-frame-ids")
    parser.add_argument("--maximum-sampled-faces", type=int, default=350_000)
    parser.add_argument("--normal-bucket-size", type=float, default=0.05)
    parser.add_argument("--normal-angle-tolerance", type=float, default=12.0)
    parser.add_argument("--offset-bin-size", type=float, default=0.1)
    parser.add_argument("--plane-fit-tolerance", type=float, default=0.15)
    parser.add_argument("--peak-separation", type=float, default=0.4)
    parser.add_argument("--maximum-peaks-per-axis", type=int, default=16)
    parser.add_argument("--minimum-relative-plane-support", type=float, default=0.05)
    parser.add_argument("--camera-clearance", type=float, default=0.5)
    parser.add_argument("--plane-extent-margin", type=float, default=0.6)
    parser.add_argument("--texture-sample-stride", type=int, default=3)
    parser.add_argument("--texture-plane-tolerance", type=float, default=0.35)
    parser.add_argument("--texture-texel-size", type=float, default=0.08)
    parser.add_argument("--maximum-atlas-dimension", type=int, default=2048)
    parser.add_argument("--minimum-texture-frame-support", type=int, default=2)
    parser.add_argument("--geometry-grid-size", type=float, default=0.1)
    parser.add_argument("--minimum-geometry-coverage", type=float, default=0.95)
    parser.add_argument("--contact-sheet-samples", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = build_planar_completion(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "promotion_approved": report["promotion_approved"],
                "selected_plane_count": len(report["selected_envelope_planes"]),
                "minimum_geometry_coverage": report["gates"]["minimum_geometry_coverage"],
                "report": str(args.output / "planar_background_report.json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
