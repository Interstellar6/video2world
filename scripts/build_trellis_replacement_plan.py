#!/usr/bin/env python3
"""Build placement sidecars for replacing scan point-cloud objects with meshes."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


PLANAR_CATEGORIES = {"window", "door", "picture", "mirror", "wall art"}
HORIZONTAL_SUPPORT_CATEGORIES = {"bed", "nightstand", "table", "desk", "chair", "sofa"}
PREFERRED_ASSET_NAMES = (
    "asset_pbr.glb",
    "asset_pbr.obj",
    "{object_id}.glb",
    "{object_id}.obj",
    "mesh.glb",
    "mesh.obj",
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
    temporary.replace(path)


def read_ascii_ply_xyz(path: Path) -> np.ndarray:
    header_lines = 0
    vertex_count: int | None = None
    ascii_format = False
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            header_lines += 1
            stripped = line.strip()
            if stripped == "format ascii 1.0":
                ascii_format = True
            if stripped.startswith("element vertex "):
                vertex_count = int(stripped.split()[-1])
            if stripped == "end_header":
                break
        else:
            raise ValueError(f"PLY header has no end_header: {path}")
    if not ascii_format:
        raise ValueError(f"Only ASCII PLY is supported for placement planning: {path}")
    if vertex_count is None or vertex_count <= 0:
        raise ValueError(f"PLY has no vertices: {path}")
    points = np.loadtxt(path, dtype=np.float64, skiprows=header_lines, usecols=(0, 1, 2))
    points = np.atleast_2d(points)
    if len(points) != vertex_count:
        raise ValueError(f"PLY vertex count mismatch for {path}: header={vertex_count}, parsed={len(points)}")
    return points


def robust_bounds(points: np.ndarray, percentile: float) -> tuple[np.ndarray, np.ndarray]:
    low = (100.0 - percentile) / 2.0
    high = 100.0 - low
    return np.percentile(points, low, axis=0), np.percentile(points, high, axis=0)


def robust_pca_bounds(points: np.ndarray, basis: np.ndarray, percentile: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    origin = points.mean(axis=0)
    local = (points - origin) @ basis
    local_min, local_max = robust_bounds(local, percentile)
    local_center = (local_min + local_max) / 2.0
    scene_center = origin + local_center @ basis.T
    return local_min, local_max, scene_center


def planar_inlier_points(
    points: np.ndarray,
    basis: np.ndarray,
    *,
    bounds_percentile: float,
    maximum_thickness_ratio: float = 0.12,
    minimum_inlier_fraction: float = 0.15,
) -> tuple[np.ndarray, dict[str, Any]]:
    origin = points.mean(axis=0)
    local = (points - origin) @ basis
    local_min, local_max = robust_bounds(local, bounds_percentile)
    local_extents = safe_extents(local_min, local_max)
    in_plane_extent = float(max(local_extents[0], local_extents[1], 1e-5))
    full_thickness = float(local_extents[2])
    target_width = max(in_plane_extent * maximum_thickness_ratio, 1e-5)
    if full_thickness <= target_width:
        return points, {
            "mode": "planar_pca_slab_not_needed",
            "input_points": int(len(points)),
            "kept_points": int(len(points)),
            "kept_fraction": 1.0,
            "target_slab_width": target_width,
            "full_robust_thickness": full_thickness,
            "maximum_thickness_ratio": maximum_thickness_ratio,
        }

    z = local[:, 2]
    order = np.argsort(z)
    sorted_z = z[order]
    best_start = 0
    best_end = 0
    end = 0
    for start in range(len(sorted_z)):
        while end < len(sorted_z) and sorted_z[end] - sorted_z[start] <= target_width:
            end += 1
        if end - start > best_end - best_start:
            best_start, best_end = start, end
    selected = order[best_start:best_end]
    minimum_count = max(8, int(math.ceil(len(points) * minimum_inlier_fraction)))
    if len(selected) < minimum_count:
        return points, {
            "mode": "planar_pca_slab_rejected_too_few_inliers",
            "input_points": int(len(points)),
            "kept_points": int(len(points)),
            "candidate_kept_points": int(len(selected)),
            "kept_fraction": 1.0,
            "candidate_kept_fraction": float(len(selected) / max(1, len(points))),
            "target_slab_width": target_width,
            "full_robust_thickness": full_thickness,
            "minimum_inlier_fraction": minimum_inlier_fraction,
            "maximum_thickness_ratio": maximum_thickness_ratio,
        }
    kept = points[np.sort(selected)]
    return kept, {
        "mode": "planar_pca_densest_slab",
        "input_points": int(len(points)),
        "kept_points": int(len(kept)),
        "kept_fraction": float(len(kept) / max(1, len(points))),
        "target_slab_width": target_width,
        "full_robust_thickness": full_thickness,
        "slab_min": float(sorted_z[best_start]),
        "slab_max": float(sorted_z[best_end - 1]),
        "maximum_thickness_ratio": maximum_thickness_ratio,
        "minimum_inlier_fraction": minimum_inlier_fraction,
    }


def pca_basis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = points - points.mean(axis=0, keepdims=True)
    covariance = np.cov(centered.T)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = values[order]
    basis = vectors[:, order]
    if np.linalg.det(basis) < 0:
        basis[:, 2] *= -1.0
    return basis, values


def safe_extents(minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
    return np.maximum(maximum - minimum, np.full(3, 1e-5, dtype=np.float64))


def mesh_bounds(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import trimesh

    loaded = trimesh.load(path, force="scene" if path.suffix.lower() == ".glb" else "mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = [geometry for geometry in loaded.geometry.values() if not geometry.is_empty]
        if not geometries:
            raise ValueError(f"Generated asset has no mesh geometry: {path}")
        mesh = trimesh.util.concatenate(geometries)
    else:
        mesh = loaded
    if mesh.is_empty:
        raise ValueError(f"Generated asset has no mesh geometry: {path}")
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    extents = safe_extents(bounds[0], bounds[1])
    return bounds[0], bounds[1], {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "bounds": bounds.tolist(),
        "extents": extents.tolist(),
    }


def matrix_from_trs(position: np.ndarray, rotation: np.ndarray, scale: np.ndarray, source_center: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation @ np.diag(scale)
    matrix[:3, 3] = position - matrix[:3, :3] @ source_center
    return matrix


def find_asset(asset_root: Path, object_id: str, explicit_path: str | None) -> Path:
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if not path.is_absolute():
            path = asset_root / path
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.resolve()
    object_dir = asset_root / object_id
    for template in PREFERRED_ASSET_NAMES:
        path = object_dir / template.format(object_id=object_id)
        if path.is_file():
            return path.resolve()
    direct_candidates = sorted(
        [path for path in object_dir.glob("*.glb")] + [path for path in object_dir.glob("*.obj")]
    )
    if direct_candidates:
        return direct_candidates[0].resolve()
    raise FileNotFoundError(f"No generated GLB/OBJ found for {object_id} under {asset_root}")


def planar_rotation_from_basis(basis: np.ndarray) -> np.ndarray:
    rotation = basis.copy()
    if np.linalg.det(rotation) < 0:
        rotation[:, 2] *= -1.0
    return rotation


def replacement_for_job(job: dict[str, Any], asset_root: Path, bounds_percentile: float) -> dict[str, Any]:
    object_id = str(job["object_id"])
    category = str(job.get("category") or "")
    source_cloud = Path(str(job["source_point_cloud"])).expanduser().resolve()
    asset_path = find_asset(asset_root.resolve(), object_id, job.get("generated_asset"))
    source_points = read_ascii_ply_xyz(source_cloud)
    category_key = category.lower()
    placement_points = source_points
    placement_filter: dict[str, Any] = {
        "mode": "all_source_points",
        "input_points": int(len(source_points)),
        "kept_points": int(len(source_points)),
        "kept_fraction": 1.0,
    }
    initial_basis, initial_eigenvalues = pca_basis(source_points)
    if category_key in PLANAR_CATEGORIES:
        placement_points, placement_filter = planar_inlier_points(
            source_points,
            initial_basis,
            bounds_percentile=bounds_percentile,
        )
    target_min, target_max = robust_bounds(placement_points, bounds_percentile)
    target_center = (target_min + target_max) / 2.0
    target_extents = safe_extents(target_min, target_max)
    basis, eigenvalues = pca_basis(placement_points)
    source_min, source_max, generated_stats = mesh_bounds(asset_path)
    source_center = (source_min + source_max) / 2.0
    source_extents = safe_extents(source_min, source_max)

    if category_key in PLANAR_CATEGORIES:
        rotation = planar_rotation_from_basis(basis)
        target_local_min, target_local_max, target_center = robust_pca_bounds(
            placement_points,
            basis,
            bounds_percentile,
        )
        target_axis_extents = safe_extents(target_local_min, target_local_max)
        planar_scale = max(target_axis_extents[:2]) / max(source_extents[:2])
        scale = np.asarray(
            [
                target_axis_extents[0] / source_extents[0],
                target_axis_extents[1] / source_extents[1],
                min(target_axis_extents[2] / source_extents[2], planar_scale * 0.18),
            ],
            dtype=np.float64,
        )
        placement_mode = "planar_pca_preserve_scene_plane"
        planar_thickness_ratio = float(target_axis_extents[2] / max(target_axis_extents[0], target_axis_extents[1], 1e-5))
    else:
        rotation = np.eye(3, dtype=np.float64)
        scale = target_extents / source_extents
        placement_mode = "axis_aligned_robust_bbox"
        planar_thickness_ratio = None
        if category_key in HORIZONTAL_SUPPORT_CATEGORIES:
            target_center[2] = target_min[2] + target_extents[2] / 2.0

    scale = np.clip(scale, 1e-5, 1e5)
    transform = matrix_from_trs(target_center, rotation, scale, source_center)
    warnings: list[str] = []
    if planar_thickness_ratio is not None and planar_thickness_ratio > 0.18:
        warnings.append(
            "planar_source_cloud_has_large_depth_thickness; likely includes through-window/background points"
        )
    if placement_filter["kept_fraction"] < 0.5:
        warnings.append("placement_uses_minor_planar_slab_subset_of_source_cloud")
    return {
        "object_id": object_id,
        "name": job.get("name"),
        "category": category,
        "status": "planned_with_warnings" if warnings else "planned",
        "placement_mode": placement_mode,
        "generated_asset": str(asset_path),
        "source_point_cloud": str(source_cloud),
        "source_frame_id": job.get("frame_id"),
        "source_image": job.get("source_image"),
        "target_bounds_percentile": float(bounds_percentile),
        "placement_source_filter": placement_filter,
        "target_bounds": [target_min.tolist(), target_max.tolist()],
        "target_center": target_center.tolist(),
        "target_extents": target_extents.tolist(),
        "target_pca_bounds": (
            [target_local_min.tolist(), target_local_max.tolist()]
            if category_key in PLANAR_CATEGORIES
            else None
        ),
        "target_pca_extents": (
            target_axis_extents.tolist()
            if category_key in PLANAR_CATEGORIES
            else None
        ),
        "source_asset_bounds": [source_min.tolist(), source_max.tolist()],
        "source_asset_center": source_center.tolist(),
        "source_asset_extents": source_extents.tolist(),
        "rotation_matrix": rotation.tolist(),
        "scale": scale.tolist(),
        "translation": transform[:3, 3].tolist(),
        "world_from_asset": transform.tolist(),
        "pca_basis": basis.tolist(),
        "pca_eigenvalues": eigenvalues.tolist(),
        "initial_pca_basis": initial_basis.tolist(),
        "initial_pca_eigenvalues": initial_eigenvalues.tolist(),
        "front_back_audit_required": category_key in PLANAR_CATEGORIES,
        "warnings": warnings,
        "planar_thickness_ratio": planar_thickness_ratio,
        "quality": {
            "input_selection_quality": job.get("selection_quality"),
            "input_selection_quality_breakdown": job.get("selection_quality_breakdown"),
            "generated_mesh": generated_stats,
            "notes": (
                "Initial transform from original object point cloud. Planar assets preserve the observed "
                "scene plane; symmetric or soft objects still need visual acceptance before replacement."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--objects", nargs="*")
    parser.add_argument("--bounds-percentile", type=float, default=96.0)
    parser.add_argument("--allow-missing-assets", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not (50.0 <= args.bounds_percentile <= 100.0):
        raise ValueError("--bounds-percentile must be in [50, 100]")
    manifest = read_json(args.input_manifest)
    prepared = manifest.get("prepared")
    if not isinstance(prepared, list):
        raise ValueError(f"Missing prepared list in {args.input_manifest}")
    wanted = set(args.objects or [])
    placements: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}
    for item in prepared:
        if not isinstance(item, dict):
            continue
        object_id = str(item.get("object_id") or "")
        if wanted and object_id not in wanted:
            continue
        try:
            placements.append(replacement_for_job(item, args.asset_root.resolve(), args.bounds_percentile))
            print(f"planned {object_id}", flush=True)
        except Exception as exc:
            if not args.allow_missing_assets:
                raise
            skipped[object_id] = f"{type(exc).__name__}: {exc}"
            print(f"skipped {object_id}: {skipped[object_id]}", flush=True)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "method": "trellis_asset_replacement_plan_from_original_object_point_cloud",
        "input_manifest": str(args.input_manifest.resolve()),
        "asset_root": str(args.asset_root.resolve()),
        "parameters": {
            "objects": sorted(wanted),
            "bounds_percentile": float(args.bounds_percentile),
            "allow_missing_assets": bool(args.allow_missing_assets),
        },
        "planned_count": len(placements),
        "skipped_count": len(skipped),
        "placements": placements,
        "skipped": skipped,
    }
    write_json(args.output.resolve(), report)
    print(f"wrote replacement plan: {args.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
