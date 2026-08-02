#!/usr/bin/env python3
"""Materialize calibrated multiview placement evidence for front-half assets.

This adapter joins the projection-ranked single-image inputs used for
TRELLIS.2 with the repository's calibrated silhouette fitter. It is deliberately
pre-QA and pre-scene-graph: it writes source-mask evidence and filtered 3D
anchors, but never mutates or promotes a scene replacement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    from scripts.build_trellis_replacement_plan import (
        PLANAR_CATEGORIES,
        pca_basis,
        planar_inlier_points,
        planar_ransac_consensus_points,
        read_ascii_ply_xyz,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from build_trellis_replacement_plan import (  # type: ignore[no-redef]
        PLANAR_CATEGORIES,
        pca_basis,
        planar_inlier_points,
        planar_ransac_consensus_points,
        read_ascii_ply_xyz,
    )


KIND = "video2world.front_half_scene_fit_evidence"
VIEW_KIND = "video2world.completed_object_source_view_manifest"


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path: Path, *, parent: Path) -> str:
    return Path(os.path.relpath(path, parent)).as_posix()


def intrinsic_for_frame(camera_info: dict[str, Any], frame_id: str) -> dict[str, float]:
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
    if not isinstance(value, dict) or not all(isinstance(value.get(key), int | float) for key in required):
        raise ValueError(f"Missing numeric intrinsic for frame {frame_id}: {required}")
    return {key: float(value[key]) for key in required}


def camera_records_from_camera_info(
    camera_info: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Convert Video2Mesh world-to-camera matrices into silhouette-fit cameras."""
    if camera_info.get("extrinsic_type") != "world_to_camera":
        raise ValueError("camera_info.extrinsic_type must be world_to_camera")
    extrinsics = camera_info.get("extrinsic")
    if not isinstance(extrinsics, dict) or not extrinsics:
        raise ValueError("camera_info.extrinsic must be a non-empty object")
    records: list[dict[str, Any]] = []
    internal: dict[str, dict[str, Any]] = {}
    for frame_id in sorted(extrinsics):
        world_to_camera = np.asarray(extrinsics[frame_id], dtype=np.float64)
        if world_to_camera.shape != (4, 4) or not np.isfinite(world_to_camera).all():
            raise ValueError(f"Invalid world-to-camera matrix for {frame_id}")
        rotation_world_to_camera = world_to_camera[:3, :3]
        if not np.allclose(rotation_world_to_camera @ rotation_world_to_camera.T, np.eye(3), atol=1e-5):
            raise ValueError(f"Camera rotation is not orthonormal for {frame_id}")
        translation = world_to_camera[:3, 3]
        position = -rotation_world_to_camera.T @ translation
        intrinsic = intrinsic_for_frame(camera_info, frame_id)
        camera = {
            "frame_id": frame_id,
            "position": position,
            # Row-vector projection: (world - position) @ rotation.
            "rotation": rotation_world_to_camera.T,
            "fx": intrinsic["fx"],
            "fy": intrinsic["fy"],
            "cx": intrinsic["cx"],
            "cy": intrinsic["cy"],
            "width": int(round(intrinsic["w"])),
            "height": int(round(intrinsic["h"])),
        }
        if camera["width"] <= 0 or camera["height"] <= 0:
            raise ValueError(f"Camera dimensions must be positive for {frame_id}")
        internal[frame_id] = camera
        records.append(
            {
                "img_name": frame_id,
                "width": camera["width"],
                "height": camera["height"],
                "position": position.tolist(),
                "rotation": rotation_world_to_camera.T.tolist(),
                "fx": intrinsic["fx"],
                "fy": intrinsic["fy"],
                "cx": intrinsic["cx"],
                "cy": intrinsic["cy"],
                "source_extrinsic_type": "world_to_camera",
            }
        )
    return records, internal


def project_points(points: np.ndarray, camera: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera_points = (points - camera["position"]) @ camera["rotation"]
    depth = camera_points[:, 2]
    visible = depth > 1e-6
    safe_depth = np.where(visible, depth, 1.0)
    x = camera["fx"] * camera_points[:, 0] / safe_depth + camera["cx"]
    y = camera["fy"] * camera_points[:, 1] / safe_depth + camera["cy"]
    return x, y, visible


def mask_support(points: np.ndarray, camera: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    x, y, visible = project_points(points, camera)
    xi = np.rint(x).astype(np.int64)
    yi = np.rint(y).astype(np.int64)
    inside = (
        visible
        & (xi >= 0)
        & (xi < camera["width"])
        & (yi >= 0)
        & (yi < camera["height"])
    )
    projected_count = int(inside.sum())
    if projected_count == 0:
        return {
            "projected_points": 0,
            "mask_support_points": 0,
            "support_ratio": 0.0,
            "mask_pixel_count": int(mask.sum()),
        }
    hits = mask[yi[inside], xi[inside]]
    return {
        "projected_points": projected_count,
        "mask_support_points": int(hits.sum()),
        "support_ratio": float(hits.mean()),
        "mask_pixel_count": int(mask.sum()),
    }


def restrict_planar_anchor_to_source_crop(
    points: np.ndarray,
    *,
    camera: dict[str, Any],
    source_mask: np.ndarray,
    bbox_xyxy: list[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep the physical planar instance visible in the TRELLIS source crop.

    A detector category can merge neighboring windows into one 3D plane. The
    single image chosen for TRELLIS is already an instance-level crop, so it is
    the least ambiguous place to split that planar anchor before scene fitting.
    """
    if len(bbox_xyxy) != 4:
        return points, {
            "mode": "source_crop_filter_rejected_invalid_bbox",
            "input_points": int(len(points)),
            "kept_points": int(len(points)),
            "kept_fraction": 1.0,
        }
    x0, y0, x1, y1 = [int(value) for value in bbox_xyxy]
    x, y, visible = project_points(points, camera)
    xi = np.rint(x).astype(np.int64)
    yi = np.rint(y).astype(np.int64)
    in_image = (
        visible
        & (xi >= 0)
        & (xi < camera["width"])
        & (yi >= 0)
        & (yi < camera["height"])
    )
    in_crop = in_image & (xi >= x0) & (xi < x1) & (yi >= y0) & (yi < y1)
    selected = np.zeros(len(points), dtype=bool)
    selected_indices = np.flatnonzero(in_crop)
    if len(selected_indices):
        selected[selected_indices] = source_mask[yi[selected_indices], xi[selected_indices]]
    minimum_count = min(len(points), max(32, int(math.ceil(len(points) * 0.1))))
    if int(selected.sum()) < minimum_count:
        return points, {
            "mode": "source_crop_filter_rejected_too_few_points",
            "input_points": int(len(points)),
            "kept_points": int(len(points)),
            "candidate_kept_points": int(selected.sum()),
            "kept_fraction": 1.0,
            "minimum_point_count": minimum_count,
            "bbox_xyxy": [x0, y0, x1, y1],
        }
    return points[selected], {
        "mode": "source_crop_and_mask_projected_anchor",
        "input_points": int(len(points)),
        "kept_points": int(selected.sum()),
        "kept_fraction": float(selected.mean()),
        "minimum_point_count": minimum_count,
        "bbox_xyxy": [x0, y0, x1, y1],
    }


def seeded_primary_component(
    mask: np.ndarray,
    *,
    points: np.ndarray,
    camera: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select one modal mask component rather than a category-wide union."""
    from scipy import ndimage

    labels, component_count = ndimage.label(mask)
    if component_count <= 1:
        return mask, {
            "mode": "single_modal_component",
            "component_count": int(component_count),
            "input_pixels": int(mask.sum()),
            "kept_pixels": int(mask.sum()),
        }
    x, y, visible = project_points(points, camera)
    xi = np.rint(x).astype(np.int64)
    yi = np.rint(y).astype(np.int64)
    inside = (
        visible
        & (xi >= 0)
        & (xi < camera["width"])
        & (yi >= 0)
        & (yi < camera["height"])
    )
    projected_indices = np.flatnonzero(inside)
    seed_labels = labels[yi[projected_indices], xi[projected_indices]] if len(projected_indices) else np.empty(0, dtype=np.int64)
    sizes = np.bincount(labels.ravel())
    seed_counts = np.bincount(seed_labels, minlength=len(sizes)) if len(seed_labels) else np.zeros(len(sizes), dtype=np.int64)
    eligible_labels = [label for label in range(1, len(sizes)) if seed_counts[label] >= 8]
    if not eligible_labels:
        return mask, {
            "mode": "seeded_component_filter_rejected_no_seeded_component",
            "component_count": int(component_count),
            "input_pixels": int(mask.sum()),
            "kept_pixels": int(mask.sum()),
            "projected_seed_points": int(len(projected_indices)),
            "component_seed_points": {str(label): int(count) for label, count in enumerate(seed_counts) if label and count},
        }
    selected_label = max(eligible_labels, key=lambda label: (int(seed_counts[label]), int(sizes[label]), -label))
    selected = labels == selected_label
    return selected, {
        "mode": "projected_anchor_primary_component",
        "component_count": int(component_count),
        "selected_label": int(selected_label),
        "input_pixels": int(mask.sum()),
        "kept_pixels": int(selected.sum()),
        "projected_seed_points": int(len(projected_indices)),
        "selected_component_seed_points": int(seed_counts[selected_label]),
        "component_seed_points": {str(label): int(count) for label, count in enumerate(seed_counts) if label and count},
    }


def view_ray_angle(left: np.ndarray, right: np.ndarray) -> float:
    cosine = float(np.clip(np.dot(left, right), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def choose_source_anchored_views(
    candidates: list[dict[str, Any]],
    *,
    cameras: dict[str, dict[str, Any]],
    anchor_center: np.ndarray,
    preferred_source_frame: str,
    count: int,
) -> tuple[list[str], dict[str, Any]]:
    if count < 1:
        raise ValueError("view count must be positive")
    if not candidates:
        return [], {
            "policy": "source_anchored_maximum_baseline",
            "status": "no_eligible_source_masks",
            "selected_frame_ids": [],
        }
    by_frame = {str(candidate["frame_id"]): candidate for candidate in candidates}
    primary = preferred_source_frame if preferred_source_frame in by_frame else max(
        candidates,
        key=lambda candidate: (
            float(candidate["support_ratio"]),
            int(candidate["mask_support_points"]),
            str(candidate["frame_id"]),
        ),
    )["frame_id"]
    rays: dict[str, np.ndarray] = {}
    for frame_id in sorted(by_frame):
        ray = anchor_center - cameras[frame_id]["position"]
        norm = float(np.linalg.norm(ray))
        if norm > 1e-9:
            rays[frame_id] = ray / norm
    selected = [str(primary)]
    additions: list[dict[str, Any]] = []
    while len(selected) < min(count, len(rays)):
        remaining = [frame_id for frame_id in sorted(rays) if frame_id not in selected]
        if not remaining:
            break
        scored = []
        for frame_id in remaining:
            angles = [view_ray_angle(rays[frame_id], rays[chosen]) for chosen in selected]
            scored.append(
                (
                    min(angles),
                    float(by_frame[frame_id]["support_ratio"]),
                    int(by_frame[frame_id]["mask_support_points"]),
                    frame_id,
                    angles,
                )
            )
        minimum_angle, _, _, frame_id, angles = max(scored)
        selected.append(frame_id)
        additions.append(
            {
                "frame_id": frame_id,
                "minimum_angle_to_selected_deg": minimum_angle,
                "angles_to_prior_selected_deg": angles,
            }
        )
    return selected, {
        "policy": "preferred_trellis_source_then_maximin_camera_to_anchor_ray_baseline",
        "preferred_source_frame": preferred_source_frame,
        "selected_source_frame": primary,
        "preferred_source_retained": primary == preferred_source_frame,
        "maximin_additions": additions,
        "selected_frame_ids": selected,
    }


def write_ascii_xyz_ply(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for x, y, z in points:
            handle.write(f"{x:.9g} {y:.9g} {z:.9g}\n")


def placement_anchor(
    points: np.ndarray,
    *,
    category: str,
    ransac_iterations: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if category.lower() not in PLANAR_CATEGORIES:
        return points, {
            "mode": "original_object_point_cloud",
            "input_points": int(len(points)),
            "kept_points": int(len(points)),
            "kept_fraction": 1.0,
        }
    filtered, report = planar_ransac_consensus_points(points, iterations=ransac_iterations)
    if report["mode"] == "planar_ransac_consensus":
        return filtered, report
    basis, _ = pca_basis(points)
    fallback, fallback_report = planar_inlier_points(
        points,
        basis,
        bounds_percentile=96.0,
    )
    fallback_report["ransac_attempt"] = report
    return fallback, fallback_report


def prepare_front_half_scene_fit_evidence(
    *,
    input_manifest_path: Path,
    output_dir: Path,
    objects: set[str] | None = None,
    view_count: int = 3,
    mask_probability: float = 0.6,
    minimum_projected_points: int = 40,
    minimum_mask_support_ratio: float = 0.5,
    planar_ransac_iterations: int = 6000,
) -> dict[str, Any]:
    if view_count < 1:
        raise ValueError("view_count must be positive")
    if not 0 < mask_probability <= 1:
        raise ValueError("mask_probability must be in (0, 1]")
    if minimum_projected_points < 1:
        raise ValueError("minimum_projected_points must be positive")
    if not 0 <= minimum_mask_support_ratio <= 1:
        raise ValueError("minimum_mask_support_ratio must be in [0, 1]")
    input_manifest_path = input_manifest_path.resolve()
    output_dir = output_dir.resolve()
    manifest = read_json(input_manifest_path)
    prepared = manifest.get("prepared")
    if not isinstance(prepared, list):
        raise ValueError("input manifest requires a prepared list")
    camera_info_path = Path(str(manifest.get("camera_info") or "")).expanduser().resolve()
    if not camera_info_path.is_file():
        raise FileNotFoundError(f"camera_info is missing: {camera_info_path}")
    camera_info = read_json(camera_info_path)
    camera_records, cameras = camera_records_from_camera_info(camera_info)
    frames_dir = Path(str(manifest.get("frames_dir") or "")).expanduser().resolve()
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"frames_dir is missing: {frames_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    cameras_path = output_dir / "cameras.json"
    write_json(cameras_path, camera_records)
    views_dir = output_dir / "views"
    masks_dir = output_dir / "masks"
    anchors_dir = output_dir / "placement_anchors"
    wanted = objects or set()
    object_records: list[dict[str, Any]] = []
    view_manifest_records: list[dict[str, Any]] = []

    for item in prepared:
        if not isinstance(item, dict):
            continue
        object_id = str(item.get("object_id") or "")
        if not object_id or (wanted and object_id not in wanted):
            continue
        category = str(item.get("category") or "")
        source_cloud = Path(str(item.get("source_point_cloud") or "")).expanduser().resolve()
        source_masks = Path(str(item.get("mask_dir") or "")).expanduser().resolve()
        source_frame = str(item.get("frame_id") or "")
        record: dict[str, Any] = {
            "object_id": object_id,
            "category": category,
            "source_frame_id": source_frame,
            "front_back_audit_required": category.lower() in PLANAR_CATEGORIES,
        }
        if not source_cloud.is_file() or not source_masks.is_dir() or source_frame not in cameras:
            record.update(
                {
                    "status": "skipped_missing_source_cloud_masks_or_camera",
                    "source_point_cloud": str(source_cloud),
                    "source_mask_dir": str(source_masks),
                }
            )
            object_records.append(record)
            continue
        source_points = read_ascii_ply_xyz(source_cloud)
        threshold = int(round(mask_probability * 255.0))
        anchor_points, anchor_report = placement_anchor(
            source_points,
            category=category,
            ransac_iterations=planar_ransac_iterations,
        )
        if category.lower() in PLANAR_CATEGORIES:
            source_mask_path = Path(str(item.get("source_mask") or "")).expanduser().resolve()
            source_bbox = item.get("bbox_xyxy")
            crop_filter: dict[str, Any]
            if source_mask_path.is_file() and isinstance(source_bbox, list):
                with Image.open(source_mask_path) as source_mask_image:
                    source_mask = np.asarray(source_mask_image.convert("L"), dtype=np.uint8) >= threshold
                source_camera = cameras[source_frame]
                if source_mask.shape == (source_camera["height"], source_camera["width"]):
                    anchor_points, crop_filter = restrict_planar_anchor_to_source_crop(
                        anchor_points,
                        camera=source_camera,
                        source_mask=source_mask,
                        bbox_xyxy=[int(value) for value in source_bbox],
                    )
                else:
                    crop_filter = {
                        "mode": "source_crop_filter_rejected_mask_shape_mismatch",
                        "input_points": int(len(anchor_points)),
                        "kept_points": int(len(anchor_points)),
                        "kept_fraction": 1.0,
                    }
            else:
                crop_filter = {
                    "mode": "source_crop_filter_rejected_missing_source_mask_or_bbox",
                    "input_points": int(len(anchor_points)),
                    "kept_points": int(len(anchor_points)),
                    "kept_fraction": 1.0,
                }
            anchor_report["source_view_crop_filter"] = crop_filter
        anchor_path = anchors_dir / f"{object_id}.ply"
        write_ascii_xyz_ply(anchor_path, anchor_points)
        candidates: list[dict[str, Any]] = []
        masks_by_frame: dict[str, tuple[np.ndarray, Path, dict[str, Any]]] = {}
        for mask_path in sorted(source_masks.glob("*.png")):
            frame_id = mask_path.stem
            camera = cameras.get(frame_id)
            if camera is None:
                continue
            with Image.open(mask_path) as source_mask:
                mask = np.asarray(source_mask.convert("L"), dtype=np.uint8) >= threshold
            if mask.shape != (camera["height"], camera["width"]):
                continue
            modal_mask, component_filter = seeded_primary_component(
                mask,
                points=anchor_points,
                camera=camera,
            )
            support = mask_support(anchor_points, camera, modal_mask)
            support["frame_id"] = frame_id
            support["source_mask"] = str(mask_path)
            support["component_filter"] = component_filter
            support["eligible"] = bool(
                support["projected_points"] >= minimum_projected_points
                and support["support_ratio"] >= minimum_mask_support_ratio
                and support["mask_pixel_count"] > 0
            )
            candidates.append(support)
            masks_by_frame[frame_id] = (modal_mask, mask_path, component_filter)
        eligible = [candidate for candidate in candidates if candidate["eligible"]]
        selected_frames, selection = choose_source_anchored_views(
            eligible,
            cameras=cameras,
            anchor_center=np.median(anchor_points, axis=0),
            preferred_source_frame=source_frame,
            count=view_count,
        )
        view_entries: list[dict[str, Any]] = []
        selected_source = str(selection.get("selected_source_frame") or "")
        for frame_id in selected_frames:
            mask, source_mask_path, component_filter = masks_by_frame[frame_id]
            output_mask = masks_dir / object_id / f"{frame_id}.png"
            output_mask.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(output_mask)
            source_rgb = frames_dir / f"{frame_id}.png"
            view_entries.append(
                {
                    "frame_id": frame_id,
                    "is_source_view": frame_id == selected_source,
                    "observed_mask": {
                        "path": relative_path(output_mask, parent=views_dir),
                        "sha256": sha256_file(output_mask),
                    },
                    "source_modal_mask": {
                        "path": str(source_mask_path),
                        "sha256": sha256_file(source_mask_path),
                    },
                    "component_filter": component_filter,
                    "source_rgb": (
                        {"path": str(source_rgb), "sha256": sha256_file(source_rgb)}
                        if source_rgb.is_file()
                        else None
                    ),
                    "occluder_masks": [],
                }
            )
        view_manifest = {
            "schema_version": 1,
            "kind": VIEW_KIND,
            "evidence_scope": (
                "multiview_source_modal_masks"
                if len(view_entries) >= 2
                else "single_view_source_modal_mask_provisional"
            ),
            "object_id": object_id,
            "views": view_entries,
        }
        view_manifest_path = views_dir / f"{object_id}.json"
        if view_entries:
            write_json(view_manifest_path, view_manifest)
            view_manifest_records.append(
                {
                    "object_id": object_id,
                    "path": str(view_manifest_path),
                    "sha256": sha256_file(view_manifest_path),
                    "frame_ids": selected_frames,
                }
            )
        record.update(
            {
                "status": (
                    "multiview_scene_fit_evidence_ready"
                    if len(view_entries) >= 2
                    else (
                        "single_view_scene_fit_evidence_provisional"
                        if len(view_entries) == 1
                        else "skipped_no_eligible_source_masks"
                    )
                ),
                "source_point_cloud": {"path": str(source_cloud), "sha256": sha256_file(source_cloud)},
                "placement_anchor": {
                    "path": str(anchor_path),
                    "sha256": sha256_file(anchor_path),
                    "point_count": int(len(anchor_points)),
                    "filter": anchor_report,
                },
                "candidate_frames": candidates,
                "eligible_frame_ids": [str(candidate["frame_id"]) for candidate in eligible],
                "frame_selection": selection,
                "view_manifest": (
                    {"path": str(view_manifest_path), "sha256": sha256_file(view_manifest_path)}
                    if view_entries
                    else None
                ),
            }
        )
        object_records.append(record)

    report = {
        "schema_version": 1,
        "kind": KIND,
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "front_half_object_completion_placement_pre_qa_pre_scene_graph",
        "claim_scope": (
            "Calibrated source-mask and anchor evidence for placement only. It does not prove "
            "unobserved geometry, front/back material orientation, or accept a scene replacement."
        ),
        "input_manifest": {"path": str(input_manifest_path), "sha256": sha256_file(input_manifest_path)},
        "camera_info": {"path": str(camera_info_path), "sha256": sha256_file(camera_info_path)},
        "camera_conversion": {
            "path": str(cameras_path),
            "sha256": sha256_file(cameras_path),
            "contract": "camera_points = (world_point - camera_position) @ camera_rotation",
            "source_contract": "camera_info world_to_camera",
        },
        "parameters": {
            "objects": sorted(wanted),
            "view_count": view_count,
            "mask_probability": mask_probability,
            "minimum_projected_points": minimum_projected_points,
            "minimum_mask_support_ratio": minimum_mask_support_ratio,
            "planar_ransac_iterations": planar_ransac_iterations,
        },
        "objects": object_records,
        "view_manifests": view_manifest_records,
    }
    report_path = output_dir / "front_half_scene_fit_evidence.json"
    write_json(report_path, report)
    return {"report": str(report_path), "objects": object_records}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--objects", nargs="*")
    parser.add_argument("--view-count", type=int, default=3)
    parser.add_argument("--mask-probability", type=float, default=0.6)
    parser.add_argument("--minimum-projected-points", type=int, default=40)
    parser.add_argument("--minimum-mask-support-ratio", type=float, default=0.5)
    parser.add_argument("--planar-ransac-iterations", type=int, default=6000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = prepare_front_half_scene_fit_evidence(
        input_manifest_path=args.input_manifest.expanduser(),
        output_dir=args.output_dir.expanduser(),
        objects=set(args.objects or []),
        view_count=args.view_count,
        mask_probability=args.mask_probability,
        minimum_projected_points=args.minimum_projected_points,
        minimum_mask_support_ratio=args.minimum_mask_support_ratio,
        planar_ransac_iterations=args.planar_ransac_iterations,
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
