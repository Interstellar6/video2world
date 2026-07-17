#!/usr/bin/env python3
"""Fit one unified GLB to an object anchor under a structural support plane."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image
from plyfile import PlyData

from scripts.qa_scene_camera_silhouette import (
    bbox_iou,
    load_camera,
    make_overlay,
    mask_bbox,
    mask_center,
    project_mesh_silhouette,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def resolve_path(value: str, *, relative_to: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def load_ply_points(path: Path, *, expected_sha256: str | None = None) -> tuple[np.ndarray, str]:
    observed_sha256 = sha256_file(path)
    if expected_sha256 is not None and expected_sha256 != observed_sha256:
        raise ValueError(f"PLY hash mismatch for {path}")
    ply = PlyData.read(path)
    if "vertex" not in ply:
        raise ValueError(f"PLY has no vertex element: {path}")
    vertex = ply["vertex"].data
    names = set(vertex.dtype.names or ())
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError(f"PLY is missing coordinates: {path}")
    points = np.column_stack([vertex[name] for name in ("x", "y", "z")]).astype(
        np.float64
    )
    if len(points) < 64 or not np.isfinite(points).all():
        raise ValueError(f"PLY requires at least 64 finite points: {path}")
    return points, observed_sha256


def sampled(points: np.ndarray, limit: int = 250_000) -> tuple[np.ndarray, int]:
    stride = max(1, len(points) // limit)
    return points[::stride], stride


def support_constrained_alignment(
    object_points: np.ndarray,
    support_points: np.ndarray,
    camera_positions: np.ndarray,
    *,
    robust_quantiles: tuple[float, float] = (0.005, 0.995),
) -> dict[str, Any]:
    object_sample, object_stride = sampled(object_points)
    support_sample, support_stride = sampled(support_points)
    support_seed = np.median(support_sample, axis=0)
    support_covariance = np.cov((support_sample - support_seed).T)
    support_values, support_vectors = np.linalg.eigh(support_covariance)
    down = support_vectors[:, int(np.argmin(support_values))]
    support_offset = float(
        np.median(np.einsum("ni,i->n", support_sample, down, optimize=False))
    )
    object_seed = np.median(object_sample, axis=0)
    object_offset = float(np.dot(object_seed, down))
    if support_offset < object_offset:
        down = -down
        support_offset = -support_offset
        object_offset = -object_offset
    up = -down

    object_covariance = np.cov((object_sample - object_seed).T)
    object_values, object_vectors = np.linalg.eigh(object_covariance)
    length_axis = object_vectors[:, int(np.argmax(object_values))]
    length_axis -= up * float(np.dot(length_axis, up))
    length_norm = float(np.linalg.norm(length_axis))
    if length_norm <= 1e-8:
        raise ValueError("object longest axis is parallel to the support normal")
    length_axis /= length_norm
    camera_center = np.mean(camera_positions, axis=0)
    toward_camera = camera_center - object_seed
    toward_camera -= up * float(np.dot(toward_camera, up))
    if float(np.dot(length_axis, toward_camera)) < 0:
        length_axis *= -1
    width_axis = np.cross(up, length_axis)
    width_axis /= np.linalg.norm(width_axis)
    axes = np.column_stack((width_axis, up, length_axis))
    if not np.isclose(np.linalg.det(axes), 1.0, atol=1e-5):
        raise ValueError("support-constrained axes are not right handed")

    projected = np.einsum(
        "ni,ij->nj",
        object_sample - object_seed,
        axes,
        optimize=False,
    )
    lower, upper = np.quantile(projected, robust_quantiles, axis=0)
    dimensions = upper - lower
    if not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
        raise ValueError("support-constrained robust extents are invalid")
    robust_center = object_seed + np.einsum(
        "ij,j->i",
        axes,
        (lower + upper) / 2.0,
        optimize=False,
    )
    origin = robust_center + down * (
        support_offset - float(np.dot(robust_center, down))
    )
    support_distances = (
        np.einsum("ni,i->n", support_sample, down, optimize=False) - support_offset
    )
    return {
        "axes_columns": axes,
        "initial_dimensions": dimensions,
        "support_origin": origin,
        "report": {
            "support_down_normal": down.tolist(),
            "object_up_axis": up.tolist(),
            "support_plane_offset": support_offset,
            "support_plane_equation": [*down.tolist(), -support_offset],
            "support_eigenvalues_ascending": support_values.tolist(),
            "support_distance_median_abs": float(np.median(np.abs(support_distances))),
            "support_distance_p95_abs": float(np.quantile(np.abs(support_distances), 0.95)),
            "object_eigenvalues_ascending": object_values.tolist(),
            "axes_columns": axes.tolist(),
            "robust_quantiles": list(robust_quantiles),
            "initial_dimensions_xyz": dimensions.tolist(),
            "robust_center": robust_center.tolist(),
            "support_origin": origin.tolist(),
            "object_analysis_stride": object_stride,
            "support_analysis_stride": support_stride,
        },
    }


def load_view_manifest(path: Path, *, object_id: str, cameras_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("object_id") != object_id:
        raise ValueError("view manifest object_id does not match")
    values = payload.get("views")
    if not isinstance(values, list) or len(values) < 2:
        raise ValueError("view manifest requires at least two views")
    result = []
    for value in values:
        if not isinstance(value, dict) or not isinstance(value.get("frame_id"), str):
            raise ValueError("view entries require frame_id")
        frame_id = value["frame_id"]
        observed_path = resolve_path(str(value.get("observed_mask")), relative_to=path.parent)
        if not observed_path.is_file():
            raise FileNotFoundError(observed_path)
        expected_observed = value.get("observed_mask_sha256")
        if expected_observed is not None and sha256_file(observed_path) != expected_observed:
            raise ValueError(f"observed-mask hash mismatch for {frame_id}")
        observed = np.asarray(Image.open(observed_path).convert("L")) > 0
        camera = load_camera(cameras_path, frame_id)
        if observed.shape != (camera["height"], camera["width"]):
            raise ValueError(f"observed-mask dimensions do not match camera for {frame_id}")
        occluder_paths = []
        occluder_union = np.zeros_like(observed)
        for occluder in value.get("occluder_masks", []):
            if not isinstance(occluder, dict):
                raise ValueError("occluder entries must be objects")
            occluder_path = resolve_path(str(occluder.get("path")), relative_to=path.parent)
            if not occluder_path.is_file():
                raise FileNotFoundError(occluder_path)
            expected = occluder.get("sha256")
            if expected is not None and sha256_file(occluder_path) != expected:
                raise ValueError(f"occluder-mask hash mismatch for {frame_id}")
            mask = np.asarray(Image.open(occluder_path).convert("L")) > 0
            if mask.shape != observed.shape:
                raise ValueError(f"occluder dimensions do not match camera for {frame_id}")
            occluder_union |= mask
            occluder_paths.append(occluder_path)
        source_path = None
        if value.get("source_frame") is not None:
            source_path = resolve_path(str(value["source_frame"]), relative_to=path.parent)
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            expected_source = value.get("source_frame_sha256")
            if expected_source is not None and sha256_file(source_path) != expected_source:
                raise ValueError(f"source-frame hash mismatch for {frame_id}")
        result.append(
            {
                "frame_id": frame_id,
                "camera": camera,
                "observed_path": observed_path,
                "observed_raw": observed,
                "observed": observed & ~occluder_union,
                "occluder_paths": occluder_paths,
                "occluder_union": occluder_union,
                "source_path": source_path,
            }
        )
    if len({item["frame_id"] for item in result}) != len(result):
        raise ValueError("view manifest contains duplicate frame IDs")
    return result


def metrics_for_masks(observed: np.ndarray, rendered: np.ndarray) -> dict[str, Any]:
    intersection = int(np.count_nonzero(observed & rendered))
    union = int(np.count_nonzero(observed | rendered))
    observed_bbox = mask_bbox(observed)
    rendered_bbox = mask_bbox(rendered)
    if union == 0 or observed_bbox is None or rendered_bbox is None:
        raise ValueError("observed and rendered visible silhouettes must be nonempty")
    return {
        "mask_iou": intersection / union,
        "bbox_iou": bbox_iou(observed_bbox, rendered_bbox),
        "center_error_px": float(np.linalg.norm(mask_center(observed) - mask_center(rendered))),
        "observed_pixels": int(np.count_nonzero(observed)),
        "rendered_pixels": int(np.count_nonzero(rendered)),
        "intersection_pixels": intersection,
        "union_pixels": union,
        "observed_bbox_xyxy": observed_bbox,
        "rendered_bbox_xyxy": rendered_bbox,
    }


def transform_from_parameters(
    axes: np.ndarray,
    dimensions: np.ndarray,
    mesh_extents: np.ndarray,
    support_origin: np.ndarray,
    offset: np.ndarray,
) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = axes @ np.diag(dimensions / mesh_extents)
    matrix[:3, 3] = support_origin + np.einsum(
        "ij,j->i", axes, offset, optimize=False
    )
    return matrix


def evaluate_candidate(
    mesh: trimesh.Trimesh,
    views: list[dict[str, Any]],
    matrix: np.ndarray,
    *,
    maximum_center_error_px: float,
) -> tuple[float, list[dict[str, Any]]]:
    records = []
    for view in views:
        rendered_raw = project_mesh_silhouette(
            mesh,
            runtime_pivot=None,
            camera=view["camera"],
            scene_transform=matrix,
        )
        rendered = rendered_raw & ~view["occluder_union"]
        metrics = metrics_for_masks(view["observed"], rendered)
        metrics["frame_id"] = view["frame_id"]
        metrics["rendered_raw_pixels"] = int(np.count_nonzero(rendered_raw))
        metrics["rendered_occluded_pixels"] = int(
            np.count_nonzero(rendered_raw & view["occluder_union"])
        )
        records.append(metrics)
    minimum_iou = min(item["mask_iou"] for item in records)
    mean_iou = float(np.mean([item["mask_iou"] for item in records]))
    minimum_bbox = min(item["bbox_iou"] for item in records)
    maximum_center = max(item["center_error_px"] for item in records)
    score = (
        minimum_iou
        + 0.35 * mean_iou
        + 0.08 * minimum_bbox
        - 0.003 * max(0.0, maximum_center - maximum_center_error_px)
    )
    return score, records


def refine_parameters(
    mesh: trimesh.Trimesh,
    views: list[dict[str, Any]],
    alignment: dict[str, Any],
    *,
    cycles: int,
    shrink: float,
    maximum_center_error_px: float,
) -> dict[str, Any]:
    dimensions = np.asarray(alignment["initial_dimensions"], dtype=np.float64)
    parameters = np.concatenate((dimensions, np.zeros(3, dtype=np.float64)))
    deltas = np.concatenate((dimensions * 0.12, dimensions * 0.08))

    def candidate(value: np.ndarray) -> tuple[float, list[dict[str, Any]], np.ndarray]:
        matrix = transform_from_parameters(
            alignment["axes_columns"],
            value[:3],
            np.asarray(mesh.extents, dtype=np.float64),
            alignment["support_origin"],
            value[3:],
        )
        score, records = evaluate_candidate(
            mesh,
            views,
            matrix,
            maximum_center_error_px=maximum_center_error_px,
        )
        return score, records, matrix

    score, records, matrix = candidate(parameters)
    trace = [
        {
            "cycle": -1,
            "score": score,
            "dimensions_xyz": parameters[:3].tolist(),
            "offset_xyz": parameters[3:].tolist(),
            "views": records,
        }
    ]
    for cycle in range(cycles):
        for parameter_index in range(6):
            best = (score, parameters.copy(), records, matrix)
            for multiplier in np.linspace(-2.0, 2.0, 9):
                trial = parameters.copy()
                trial[parameter_index] += multiplier * deltas[parameter_index]
                if np.any(trial[:3] <= 1e-6):
                    continue
                trial_score, trial_records, trial_matrix = candidate(trial)
                if trial_score > best[0]:
                    best = (trial_score, trial, trial_records, trial_matrix)
            score, parameters, records, matrix = best
        trace.append(
            {
                "cycle": cycle,
                "score": score,
                "dimensions_xyz": parameters[:3].tolist(),
                "offset_xyz": parameters[3:].tolist(),
                "views": records,
            }
        )
        deltas *= shrink
    return {
        "score": score,
        "dimensions": parameters[:3],
        "offset": parameters[3:],
        "matrix": matrix,
        "view_records": records,
        "trace": trace,
    }


def refine_supported_scene_fit(
    *,
    object_id: str,
    mesh_source: Path,
    object_anchor_ply: Path,
    support_anchor_ply: Path,
    cameras_path: Path,
    view_manifest_path: Path,
    output_dir: Path,
    expected_object_anchor_sha256: str | None = None,
    expected_support_anchor_sha256: str | None = None,
    cycles: int = 5,
    shrink: float = 0.48,
    minimum_mask_iou: float = 0.65,
    minimum_bbox_iou: float = 0.8,
    maximum_center_error_px: float = 20.0,
    maximum_scale_anisotropy: float = 2.0,
) -> dict[str, Any]:
    if cycles < 1 or not 0 < shrink < 1:
        raise ValueError("cycles must be positive and shrink must be in (0, 1)")
    if maximum_scale_anisotropy < 1:
        raise ValueError("maximum_scale_anisotropy must be at least one")
    object_points, object_anchor_sha = load_ply_points(
        object_anchor_ply, expected_sha256=expected_object_anchor_sha256
    )
    support_points, support_anchor_sha = load_ply_points(
        support_anchor_ply, expected_sha256=expected_support_anchor_sha256
    )
    views = load_view_manifest(
        view_manifest_path, object_id=object_id, cameras_path=cameras_path
    )
    camera_positions = np.asarray([view["camera"]["position"] for view in views])
    alignment = support_constrained_alignment(
        object_points,
        support_points,
        camera_positions,
    )
    loaded = trimesh.load(mesh_source, force="scene")
    scene = loaded if isinstance(loaded, trimesh.Scene) else trimesh.Scene(loaded)
    mesh = scene.to_geometry()
    result = refine_parameters(
        mesh,
        views,
        alignment,
        cycles=cycles,
        shrink=shrink,
        maximum_center_error_px=maximum_center_error_px,
    )
    matrix = result["matrix"]
    axis_scales = np.linalg.norm(matrix[:3, :3], axis=0)
    scale_anisotropy = float(np.max(axis_scales) / np.min(axis_scales))
    rotation = matrix[:3, :3] / axis_scales[None, :]
    vertices_world = np.einsum(
        "ni,ji->nj",
        np.asarray(mesh.vertices, dtype=np.float64),
        matrix[:3, :3],
        optimize=False,
    ) + matrix[:3, 3]
    support_down = np.asarray(
        alignment["report"]["support_down_normal"], dtype=np.float64
    )
    support_offset = float(alignment["report"]["support_plane_offset"])
    support_signed_distances = (
        np.einsum("ni,i->n", vertices_world, support_down, optimize=False)
        - support_offset
    )
    closest_support_distance = float(np.max(support_signed_distances))
    support_tolerance = max(1e-4, float(result["dimensions"][1]) * 0.005)
    support_penetration = max(0.0, closest_support_distance)
    support_gap = max(0.0, -closest_support_distance)
    parents = dict(scene.graph.transforms.parents)
    base_frame = scene.graph.base_frame
    root_nodes = sorted(
        node for node, parent in parents.items() if parent == base_frame and node != base_frame
    )
    material_types = [
        type(getattr(getattr(geometry, "visual", None), "material", None)).__name__
        for geometry in scene.geometry.values()
    ]
    view_gates = []
    for record in result["view_records"]:
        view_gates.append(
            {
                "frame_id": record["frame_id"],
                "mask_iou": record["mask_iou"] >= minimum_mask_iou,
                "bbox_iou": record["bbox_iou"] >= minimum_bbox_iou,
                "center_error_px": record["center_error_px"] <= maximum_center_error_px,
            }
        )
    gates = {
        "support_plane_finite": bool(
            np.isfinite(np.asarray(alignment["report"]["support_plane_equation"])).all()
        ),
        "support_constrained_rotation": bool(
            np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            and np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
        ),
        "scale_anisotropy_within_limit": scale_anisotropy <= maximum_scale_anisotropy,
        "single_logical_root": root_nodes == [object_id],
        "pbr_materials": bool(material_types)
        and all(value == "PBRMaterial" for value in material_types),
        "collision_topology_ready": bool(mesh.is_watertight),
        "support_contact": (
            support_penetration <= support_tolerance and support_gap <= support_tolerance
        ),
        "multi_view_silhouette": all(all(gate.values()) for gate in view_gates),
    }
    all_gates_passed = all(gates.values())

    evidence_dir = output_dir / "evidence"
    review_dir = output_dir / "source_camera_review"
    copied_views = []
    normalized_views = []
    for view, metrics in zip(views, result["view_records"], strict=True):
        frame_id = view["frame_id"]
        observed_destination = evidence_dir / f"{frame_id}.observed.png"
        observed_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(view["observed_path"], observed_destination)
        copied_occluders = []
        for index, source in enumerate(view["occluder_paths"]):
            destination = evidence_dir / f"{frame_id}.occluder_{index:02d}.png"
            shutil.copy2(source, destination)
            copied_occluders.append(
                {"path": str(destination), "sha256": sha256_file(destination)}
            )
        copied_source = None
        if view["source_path"] is not None:
            source_destination = evidence_dir / f"{frame_id}.source.png"
            shutil.copy2(view["source_path"], source_destination)
            copied_source = {
                "path": str(source_destination),
                "sha256": sha256_file(source_destination),
            }
        source_image = (
            Image.open(view["source_path"]).convert("RGB")
            if view["source_path"] is not None
            else None
        )
        rendered_raw = project_mesh_silhouette(
            mesh,
            runtime_pivot=None,
            camera=view["camera"],
            scene_transform=matrix,
        )
        rendered = rendered_raw & ~view["occluder_union"]
        frame_review_dir = review_dir / frame_id
        frame_review_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = frame_review_dir / "source_camera_silhouette_overlay.png"
        rendered_path = frame_review_dir / "rendered_visible_silhouette.png"
        make_overlay(source_image, view["observed"], rendered).save(overlay_path)
        Image.fromarray(rendered.astype(np.uint8) * 255).save(rendered_path)
        if source_image is not None:
            source_image.close()
        copied_views.append(
            {
                "frame_id": frame_id,
                "observed_mask": {
                    "path": str(observed_destination),
                    "sha256": sha256_file(observed_destination),
                },
                "occluder_masks": copied_occluders,
                "source_frame": copied_source,
                "metrics": metrics,
                "gates": next(item for item in view_gates if item["frame_id"] == frame_id),
                "overlay": {"path": str(overlay_path), "sha256": sha256_file(overlay_path)},
                "rendered_visible_silhouette": {
                    "path": str(rendered_path),
                    "sha256": sha256_file(rendered_path),
                },
            }
        )
        normalized_views.append(
            {
                "frame_id": frame_id,
                "observed_mask": str(observed_destination),
                "observed_mask_sha256": sha256_file(observed_destination),
                "occluder_masks": copied_occluders,
                "source_frame": copied_source["path"] if copied_source else None,
                "source_frame_sha256": copied_source["sha256"] if copied_source else None,
            }
        )

    normalized_manifest_path = evidence_dir / "view_manifest.normalized.json"
    write_json(
        normalized_manifest_path,
        {
            "schema_version": 1,
            "kind": "video2world.scene_fit_view_evidence",
            "object_id": object_id,
            "views": normalized_views,
        },
    )

    report = {
        "schema_version": 1,
        "kind": "video2world.support_constrained_multiview_scene_fit",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "technical_gates_passed" if all_gates_passed else "rejected",
        "promotion_allowed": False,
        "promotion_status": "held_pending_scene_browser_review",
        "object_id": object_id,
        "representation_mode": "unified_mesh_visual_logic_collision",
        "sources": {
            "mesh": {"path": str(mesh_source), "sha256": sha256_file(mesh_source)},
            "object_anchor": {
                "path": str(object_anchor_ply),
                "sha256": object_anchor_sha,
                "point_count": len(object_points),
            },
            "support_anchor": {
                "path": str(support_anchor_ply),
                "sha256": support_anchor_sha,
                "point_count": len(support_points),
            },
            "cameras": {"path": str(cameras_path), "sha256": sha256_file(cameras_path)},
            "view_manifest": {
                "path": str(view_manifest_path),
                "sha256": sha256_file(view_manifest_path),
            },
            "normalized_view_manifest": {
                "path": str(normalized_manifest_path),
                "sha256": sha256_file(normalized_manifest_path),
            },
        },
        "support_alignment": alignment["report"],
        "support_contact": {
            "signed_distance_convention": "positive_below_support_plane",
            "closest_signed_distance": closest_support_distance,
            "maximum_penetration": support_penetration,
            "minimum_gap": support_gap,
            "tolerance": support_tolerance,
            "passed": support_penetration <= support_tolerance
            and support_gap <= support_tolerance,
        },
        "search": {
            "method": "shared_transform_coordinate_descent",
            "objective": "minimum_view_iou_plus_mean_iou_bbox_and_center_terms",
            "cycles": cycles,
            "shrink": shrink,
            "score": result["score"],
            "initial_dimensions_xyz": alignment["initial_dimensions"].tolist(),
            "final_dimensions_xyz": result["dimensions"].tolist(),
            "final_offset_xyz": result["offset"].tolist(),
            "trace": result["trace"],
        },
        "runtime_transform": {
            "matrix_row_major": matrix.tolist(),
            "asset_coordinates_baked": False,
            "translation": matrix[:3, 3].tolist(),
            "rotation_matrix_row_major": rotation.tolist(),
            "scale_xyz": axis_scales.tolist(),
            "scale_anisotropy_ratio": scale_anisotropy,
        },
        "thresholds": {
            "minimum_mask_iou": minimum_mask_iou,
            "minimum_bbox_iou": minimum_bbox_iou,
            "maximum_center_error_px": maximum_center_error_px,
            "maximum_scale_anisotropy": maximum_scale_anisotropy,
        },
        "views": copied_views,
        "logical_entity": {
            "root_node": object_id,
            "selection_owner": object_id,
            "motion_owner": object_id,
            "collision_owner": object_id,
        },
        "mesh": {
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "watertight": bool(mesh.is_watertight),
            "material_types": material_types,
        },
        "acceptance_gates": gates,
        "view_acceptance_gates": view_gates,
        "all_acceptance_gates_passed": all_gates_passed,
        "promotion_blockers": [
            "scene_interpenetration_gate",
            "support_contact_browser_review",
            "independent_pillow_replay",
            "accepted_clean_plate_or_static_scene_carve",
        ],
    }
    write_json(output_dir / "scene_fit_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--mesh-source", type=Path, required=True)
    parser.add_argument("--object-anchor-ply", type=Path, required=True)
    parser.add_argument("--support-anchor-ply", type=Path, required=True)
    parser.add_argument("--expected-object-anchor-sha256")
    parser.add_argument("--expected-support-anchor-sha256")
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--view-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--shrink", type=float, default=0.48)
    parser.add_argument("--minimum-mask-iou", type=float, default=0.65)
    parser.add_argument("--minimum-bbox-iou", type=float, default=0.8)
    parser.add_argument("--maximum-center-error-px", type=float, default=20.0)
    parser.add_argument("--maximum-scale-anisotropy", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = refine_supported_scene_fit(
        object_id=args.object_id,
        mesh_source=args.mesh_source.expanduser().resolve(),
        object_anchor_ply=args.object_anchor_ply.expanduser().resolve(),
        support_anchor_ply=args.support_anchor_ply.expanduser().resolve(),
        cameras_path=args.cameras.expanduser().resolve(),
        view_manifest_path=args.view_manifest.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        expected_object_anchor_sha256=args.expected_object_anchor_sha256,
        expected_support_anchor_sha256=args.expected_support_anchor_sha256,
        cycles=args.cycles,
        shrink=args.shrink,
        minimum_mask_iou=args.minimum_mask_iou,
        minimum_bbox_iou=args.minimum_bbox_iou,
        maximum_center_error_px=args.maximum_center_error_px,
        maximum_scale_anisotropy=args.maximum_scale_anisotropy,
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True))
    return 0 if report["all_acceptance_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
