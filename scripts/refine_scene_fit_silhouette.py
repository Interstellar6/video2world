#!/usr/bin/env python3
"""Refine one anchored scene-fit mesh against a calibrated source silhouette.

The search is deliberately restricted to transformations that preserve the
object's independent 3D anchor: tangent-plane scale, rotation about the OBB
thickness axis, and a small tangent-plane translation.  Thickness and the
observed front-surface depth are never optimized.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from trimesh.exchange.obj import export_obj


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


def load_support_plane_evidence(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    support_plane = payload.get("support_plane")
    if not isinstance(support_plane, dict):
        raise ValueError("support plane report has no support_plane object")
    interior_normal = np.asarray(support_plane.get("interior_normal"), dtype=np.float64)
    if interior_normal.shape != (3,) or not np.isfinite(interior_normal).all():
        raise ValueError("support plane report has no finite interior_normal")
    interior_normal /= np.linalg.norm(interior_normal)
    gravity_direction = -interior_normal
    evidence = {
        "path": str(path),
        "sha256": sha256_file(path),
        "plane_id": support_plane.get("plane_id"),
        "semantic_role": support_plane.get("semantic_role"),
        "interior_normal": interior_normal.tolist(),
        "derived_downward_gravity_direction": gravity_direction.tolist(),
        "sign_semantics": (
            "interior_normal points into the room; downward gravity used by the "
            "max-extreme support check is its negation"
        ),
        "offset": support_plane.get("offset"),
        "equation": support_plane.get("equation"),
        "anchor": support_plane.get("anchor"),
        "anchor_fraction_sampled": support_plane.get("anchor_fraction_sampled"),
        "rms_tsdf_plane_residual": support_plane.get("rms_tsdf_plane_residual"),
    }
    return gravity_direction, evidence


def determinant_3x3(matrix: np.ndarray) -> float:
    (a, b, c), (d, e, f), (g, h, i) = np.asarray(matrix, dtype=np.float64).tolist()
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def load_camera(path: Path, frame_id: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("camera file must contain an array")
    matches = [item for item in payload if str(item.get("img_name")) == frame_id]
    if len(matches) != 1:
        raise ValueError(f"expected one camera for {frame_id!r}, found {len(matches)}")
    camera = matches[0]
    position = np.asarray(camera.get("position"), dtype=np.float64)
    rotation = np.asarray(camera.get("rotation"), dtype=np.float64)
    if position.shape != (3,) or rotation.shape != (3, 3):
        raise ValueError("camera position/rotation has invalid dimensions")
    if not np.isfinite(position).all() or not np.isfinite(rotation).all():
        raise ValueError("camera contains non-finite values")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("camera rotation is not orthonormal")
    width = int(camera.get("width", 0))
    height = int(camera.get("height", 0))
    fx = float(camera.get("fx", 0))
    fy = float(camera.get("fy", 0))
    if width <= 0 or height <= 0 or fx <= 0 or fy <= 0:
        raise ValueError("camera intrinsics must be positive")
    return {
        "position": position,
        "rotation": rotation,
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fy,
    }


def mask_bbox(mask: np.ndarray) -> list[int] | None:
    y, x = np.nonzero(mask)
    if len(x) == 0:
        return None
    return [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1]


def mask_center(mask: np.ndarray) -> np.ndarray:
    y, x = np.nonzero(mask)
    if len(x) == 0:
        raise ValueError("cannot compute center of an empty mask")
    return np.asarray([x.mean(), y.mean()], dtype=np.float64)


def bbox_iou(left: list[int], right: list[int]) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def project_mesh_silhouette(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    runtime_pivot: np.ndarray,
    camera: dict[str, Any],
    image_y_axis: str,
) -> np.ndarray:
    if image_y_axis not in {"down", "up"}:
        raise ValueError("image_y_axis must be 'down' or 'up'")
    vertices_world = vertices + runtime_pivot
    relative = vertices_world - camera["position"]
    camera_points = np.einsum("ni,ij->nj", relative, camera["rotation"])
    depth = camera_points[:, 2]
    visible = depth > 1e-6
    projected = np.full((len(camera_points), 2), np.nan, dtype=np.float64)
    projected[visible, 0] = (
        camera["fx"] * camera_points[visible, 0] / depth[visible] + camera["width"] / 2
    )
    y_sign = 1.0 if image_y_axis == "down" else -1.0
    projected[visible, 1] = (
        camera["height"] / 2 + y_sign * camera["fy"] * camera_points[visible, 1] / depth[visible]
    )
    canvas = Image.new("1", (camera["width"], camera["height"]), 0)
    draw = ImageDraw.Draw(canvas)
    for face in faces:
        if not np.all(visible[face]):
            continue
        polygon = [tuple(projected[index]) for index in face]
        if np.isfinite(np.asarray(polygon)).all():
            draw.polygon(polygon, fill=1)
    return np.asarray(canvas, dtype=bool)


def silhouette_metrics(
    observed: np.ndarray,
    rendered: np.ndarray,
    ignored_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    ignored = np.zeros_like(observed) if ignored_mask is None else ignored_mask
    if ignored.shape != observed.shape:
        raise ValueError("ignored mask dimensions do not match silhouette")
    evaluated_observed = observed & ~ignored
    evaluated_rendered = rendered & ~ignored
    if not evaluated_observed.any() or not evaluated_rendered.any():
        raise ValueError("observed and rendered masks must both be non-empty")
    intersection = int(np.count_nonzero(evaluated_observed & evaluated_rendered))
    union = int(np.count_nonzero(evaluated_observed | evaluated_rendered))
    observed_pixels = int(np.count_nonzero(evaluated_observed))
    rendered_pixels = int(np.count_nonzero(evaluated_rendered))
    observed_bbox = mask_bbox(observed)
    rendered_bbox = mask_bbox(rendered)
    assert observed_bbox is not None and rendered_bbox is not None
    return {
        "observed_pixels": observed_pixels,
        "rendered_pixels": rendered_pixels,
        "intersection_pixels": intersection,
        "union_pixels": union,
        "mask_iou": intersection / union,
        "rendered_precision": intersection / rendered_pixels,
        "observed_recall": intersection / observed_pixels,
        "ignored_pixels": int(np.count_nonzero(ignored)),
        "observed_pixels_inside_ignored_region": int(np.count_nonzero(observed & ignored)),
        "rendered_pixels_inside_ignored_region": int(np.count_nonzero(rendered & ignored)),
        "pixel_evaluation_domain": (
            "outside_explicit_occluder_union" if ignored.any() else "full_image"
        ),
        "observed_bbox_xyxy": observed_bbox,
        "rendered_bbox_xyxy": rendered_bbox,
        "bbox_iou": bbox_iou(observed_bbox, rendered_bbox),
        "center_error_px": float(
            np.linalg.norm(mask_center(evaluated_observed) - mask_center(evaluated_rendered))
        ),
    }


def rotation_about_axis(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    normalized = np.array(axis, dtype=np.float64, copy=True)
    normalized /= np.linalg.norm(normalized)
    x, y, z = normalized
    angle = math.radians(angle_deg)
    c = math.cos(angle)
    s = math.sin(angle)
    one_minus_c = 1.0 - c
    return np.asarray(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ],
        dtype=np.float64,
    )


def refinement_linear(
    axes: np.ndarray,
    thickness_axis: int,
    scale_axis_0: float,
    scale_axis_1: float,
    angle_deg: float,
) -> np.ndarray:
    tangent_axes = [index for index in range(3) if index != thickness_axis]
    local_scale = np.ones(3, dtype=np.float64)
    local_scale[tangent_axes] = [scale_axis_0, scale_axis_1]
    scale_world = axes @ np.diag(local_scale) @ axes.T
    rotation = rotation_about_axis(axes[:, thickness_axis], angle_deg)
    return rotation @ scale_world


def cluster_mesh(mesh: trimesh.Trimesh, resolution: int) -> trimesh.Trimesh:
    if resolution < 4:
        raise ValueError("search mesh resolution must be at least 4")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    lower = vertices.min(axis=0)
    extents = np.maximum(np.ptp(vertices, axis=0), 1e-12)
    cells = np.floor((vertices - lower) / extents * (resolution - 1)).astype(np.int64)
    unique_cells, inverse = np.unique(cells, axis=0, return_inverse=True)
    accumulated = np.zeros((len(unique_cells), 3), dtype=np.float64)
    np.add.at(accumulated, inverse, vertices)
    clustered_vertices = accumulated / np.bincount(inverse)[:, None]
    clustered_faces = inverse[np.asarray(mesh.faces, dtype=np.int64)]
    nondegenerate = (
        (clustered_faces[:, 0] != clustered_faces[:, 1])
        & (clustered_faces[:, 0] != clustered_faces[:, 2])
        & (clustered_faces[:, 1] != clustered_faces[:, 2])
    )
    clustered_faces = clustered_faces[nondegenerate]
    sorted_faces = np.sort(clustered_faces, axis=1)
    _, unique_indices = np.unique(sorted_faces, axis=0, return_index=True)
    clustered_faces = clustered_faces[np.sort(unique_indices)]
    if len(clustered_vertices) < 4 or len(clustered_faces) < 2:
        raise ValueError("search mesh collapsed below a usable surface")
    return trimesh.Trimesh(clustered_vertices, clustered_faces, process=False)


def project_point(point: np.ndarray, camera: dict[str, Any], image_y_axis: str) -> np.ndarray:
    camera_point = (point - camera["position"]) @ camera["rotation"]
    if camera_point[2] <= 1e-6:
        raise ValueError("anchor projects behind the camera")
    y_sign = 1.0 if image_y_axis == "down" else -1.0
    return np.asarray(
        [
            camera["fx"] * camera_point[0] / camera_point[2] + camera["width"] / 2,
            camera["height"] / 2 + y_sign * camera["fy"] * camera_point[1] / camera_point[2],
        ],
        dtype=np.float64,
    )


def tangent_translation_for_center(
    *,
    rendered: np.ndarray,
    observed: np.ndarray,
    pivot: np.ndarray,
    tangent_basis: np.ndarray,
    camera: dict[str, Any],
    image_y_axis: str,
    maximum_translation: float,
    ignored_mask: np.ndarray | None = None,
) -> np.ndarray:
    ignored = np.zeros_like(observed) if ignored_mask is None else ignored_mask
    desired_delta = mask_center(observed & ~ignored) - mask_center(rendered & ~ignored)
    epsilon = max(1e-4, maximum_translation * 1e-3)
    projected_pivot = project_point(pivot, camera, image_y_axis)
    jacobian = np.column_stack(
        [
            (
                project_point(pivot + epsilon * tangent_basis[:, index], camera, image_y_axis)
                - projected_pivot
            )
            / epsilon
            for index in range(2)
        ]
    )
    coefficients = np.linalg.lstsq(jacobian, desired_delta, rcond=None)[0]
    length = float(np.linalg.norm(coefficients))
    if length > maximum_translation:
        coefficients *= maximum_translation / length
    return coefficients


@dataclass(frozen=True)
class Candidate:
    scale_axis_0: float
    scale_axis_1: float
    angle_deg: float
    translation_coefficients: tuple[float, float]
    metrics: dict[str, Any]
    score: float


def candidate_score(
    metrics: dict[str, Any],
    *,
    minimum_mask_iou: float,
    minimum_bbox_iou: float,
    maximum_center_error_px: float,
    scale_axis_0: float,
    scale_axis_1: float,
    angle_deg: float,
    translation_length: float,
    maximum_angle_deg: float,
    maximum_translation: float,
) -> float:
    center_ratio = (
        1.2
        if metrics["center_error_px"] <= 1e-6
        else min(1.2, maximum_center_error_px / metrics["center_error_px"])
    )
    gate_floor = min(
        metrics["mask_iou"] / minimum_mask_iou,
        metrics["bbox_iou"] / minimum_bbox_iou,
        center_ratio,
    )
    regularization = (
        0.015 * (scale_axis_0 - 1.0) ** 2
        + 0.015 * (scale_axis_1 - 1.0) ** 2
        + 0.01 * (angle_deg / maximum_angle_deg) ** 2
        + 0.01 * (translation_length / maximum_translation) ** 2
    )
    return (
        3.0 * gate_floor
        + metrics["mask_iou"]
        + 0.5 * metrics["bbox_iou"]
        - 0.002 * metrics["center_error_px"]
        - regularization
    )


def evaluate_candidate(
    *,
    base_vertices: np.ndarray,
    faces: np.ndarray,
    axes: np.ndarray,
    thickness_axis: int,
    base_pivot: np.ndarray,
    observed: np.ndarray,
    camera: dict[str, Any],
    image_y_axis: str,
    scale_axis_0: float,
    scale_axis_1: float,
    angle_deg: float,
    maximum_translation: float,
    minimum_mask_iou: float,
    minimum_bbox_iou: float,
    maximum_center_error_px: float,
    maximum_angle_deg: float,
    ignored_mask: np.ndarray | None = None,
) -> Candidate:
    linear = refinement_linear(
        axes,
        thickness_axis,
        scale_axis_0,
        scale_axis_1,
        angle_deg,
    )
    transformed = np.einsum("ni,ji->nj", base_vertices, linear)
    initial_render = project_mesh_silhouette(
        transformed,
        faces,
        runtime_pivot=base_pivot,
        camera=camera,
        image_y_axis=image_y_axis,
    )
    tangent_indices = [index for index in range(3) if index != thickness_axis]
    tangent_basis = axes[:, tangent_indices]
    translation = tangent_translation_for_center(
        rendered=initial_render,
        observed=observed,
        pivot=base_pivot,
        tangent_basis=tangent_basis,
        camera=camera,
        image_y_axis=image_y_axis,
        maximum_translation=maximum_translation,
        ignored_mask=ignored_mask,
    )
    pivot = base_pivot + tangent_basis @ translation
    rendered = project_mesh_silhouette(
        transformed,
        faces,
        runtime_pivot=pivot,
        camera=camera,
        image_y_axis=image_y_axis,
    )
    metrics = silhouette_metrics(observed, rendered, ignored_mask)
    score = candidate_score(
        metrics,
        minimum_mask_iou=minimum_mask_iou,
        minimum_bbox_iou=minimum_bbox_iou,
        maximum_center_error_px=maximum_center_error_px,
        scale_axis_0=scale_axis_0,
        scale_axis_1=scale_axis_1,
        angle_deg=angle_deg,
        translation_length=float(np.linalg.norm(translation)),
        maximum_angle_deg=maximum_angle_deg,
        maximum_translation=maximum_translation,
    )
    return Candidate(
        scale_axis_0=scale_axis_0,
        scale_axis_1=scale_axis_1,
        angle_deg=angle_deg,
        translation_coefficients=(float(translation[0]), float(translation[1])),
        metrics=metrics,
        score=score,
    )


def candidate_passes(
    candidate: Candidate,
    *,
    minimum_mask_iou: float,
    minimum_bbox_iou: float,
    maximum_center_error_px: float,
) -> bool:
    return (
        candidate.metrics["mask_iou"] >= minimum_mask_iou
        and candidate.metrics["bbox_iou"] >= minimum_bbox_iou
        and candidate.metrics["center_error_px"] <= maximum_center_error_px
    )


def search_refinement(
    *,
    mesh: trimesh.Trimesh,
    axes: np.ndarray,
    thickness_axis: int,
    base_pivot: np.ndarray,
    observed: np.ndarray,
    camera: dict[str, Any],
    image_y_axis: str = "down",
    minimum_scale: float = 0.75,
    maximum_scale: float = 1.35,
    maximum_angle_deg: float = 55.0,
    maximum_translation: float = 0.25,
    minimum_mask_iou: float = 0.65,
    minimum_bbox_iou: float = 0.8,
    maximum_center_error_px: float = 20.0,
    coarse_resolution: int = 24,
    fine_resolution: int = 40,
    ignored_mask: np.ndarray | None = None,
) -> tuple[Candidate, dict[str, Any]]:
    if not 0 < minimum_scale <= 1 <= maximum_scale:
        raise ValueError("scale bounds must be positive and contain 1")
    if maximum_angle_deg <= 0 or maximum_translation <= 0:
        raise ValueError("angle and translation bounds must be positive")
    if thickness_axis not in {0, 1, 2}:
        raise ValueError("thickness_axis must be 0, 1, or 2")
    if axes.shape != (3, 3) or not np.allclose(axes.T @ axes, np.eye(3), atol=1e-5):
        raise ValueError("axes must be an orthonormal 3x3 matrix")
    coarse_mesh = cluster_mesh(mesh, coarse_resolution)
    fine_mesh = cluster_mesh(mesh, fine_resolution)
    common = {
        "axes": axes,
        "thickness_axis": thickness_axis,
        "base_pivot": base_pivot,
        "observed": observed,
        "camera": camera,
        "image_y_axis": image_y_axis,
        "maximum_translation": maximum_translation,
        "minimum_mask_iou": minimum_mask_iou,
        "minimum_bbox_iou": minimum_bbox_iou,
        "maximum_center_error_px": maximum_center_error_px,
        "maximum_angle_deg": maximum_angle_deg,
        "ignored_mask": ignored_mask,
    }
    scale_values = np.linspace(minimum_scale, maximum_scale, 5)
    angle_values = np.linspace(-maximum_angle_deg, maximum_angle_deg, 12)
    coarse_candidates = [
        evaluate_candidate(
            base_vertices=np.asarray(coarse_mesh.vertices),
            faces=np.asarray(coarse_mesh.faces),
            scale_axis_0=float(scale_axis_0),
            scale_axis_1=float(scale_axis_1),
            angle_deg=float(angle),
            **common,
        )
        for angle in angle_values
        for scale_axis_0 in scale_values
        for scale_axis_1 in scale_values
    ]
    ranked = sorted(coarse_candidates, key=lambda candidate: candidate.score, reverse=True)
    starts: list[Candidate] = []
    for candidate in ranked:
        if all(abs(candidate.angle_deg - prior.angle_deg) >= 8.0 for prior in starts):
            starts.append(candidate)
        if len(starts) == 3:
            break
    if not starts:
        raise RuntimeError("silhouette search produced no candidates")

    fine_candidates: list[Candidate] = []
    for start in starts:
        current = evaluate_candidate(
            base_vertices=np.asarray(fine_mesh.vertices),
            faces=np.asarray(fine_mesh.faces),
            scale_axis_0=start.scale_axis_0,
            scale_axis_1=start.scale_axis_1,
            angle_deg=start.angle_deg,
            **common,
        )
        steps = [
            ((maximum_scale - minimum_scale) / 8.0, (maximum_scale - minimum_scale) / 8.0, 5.0),
            ((maximum_scale - minimum_scale) / 16.0, (maximum_scale - minimum_scale) / 16.0, 2.5),
            ((maximum_scale - minimum_scale) / 32.0, (maximum_scale - minimum_scale) / 32.0, 1.25),
        ]
        for scale_step_0, scale_step_1, angle_step in steps:
            improved = True
            while improved:
                improved = False
                neighbours: list[Candidate] = []
                for delta_0, delta_1, delta_angle in (
                    (scale_step_0, 0.0, 0.0),
                    (-scale_step_0, 0.0, 0.0),
                    (0.0, scale_step_1, 0.0),
                    (0.0, -scale_step_1, 0.0),
                    (0.0, 0.0, angle_step),
                    (0.0, 0.0, -angle_step),
                ):
                    scale_0 = current.scale_axis_0 + delta_0
                    scale_1 = current.scale_axis_1 + delta_1
                    angle = current.angle_deg + delta_angle
                    if (
                        minimum_scale <= scale_0 <= maximum_scale
                        and minimum_scale <= scale_1 <= maximum_scale
                        and abs(angle) <= maximum_angle_deg
                    ):
                        neighbours.append(
                            evaluate_candidate(
                                base_vertices=np.asarray(fine_mesh.vertices),
                                faces=np.asarray(fine_mesh.faces),
                                scale_axis_0=scale_0,
                                scale_axis_1=scale_1,
                                angle_deg=angle,
                                **common,
                            )
                        )
                best_neighbour = max(neighbours, key=lambda item: item.score)
                if best_neighbour.score > current.score + 1e-9:
                    current = best_neighbour
                    improved = True
        fine_candidates.append(current)
    passing = [
        candidate
        for candidate in fine_candidates
        if candidate_passes(
            candidate,
            minimum_mask_iou=minimum_mask_iou,
            minimum_bbox_iou=minimum_bbox_iou,
            maximum_center_error_px=maximum_center_error_px,
        )
    ]
    selected_pool = passing or fine_candidates
    selected = max(selected_pool, key=lambda candidate: candidate.score)
    audit = {
        "coarse_search_mesh": {
            "resolution": coarse_resolution,
            "vertices": len(coarse_mesh.vertices),
            "faces": len(coarse_mesh.faces),
        },
        "fine_search_mesh": {
            "resolution": fine_resolution,
            "vertices": len(fine_mesh.vertices),
            "faces": len(fine_mesh.faces),
        },
        "coarse_candidate_count": len(coarse_candidates),
        "fine_start_count": len(starts),
        "fine_candidates": [candidate_to_json(item) for item in fine_candidates],
        "selected_from_passing_pool": bool(passing),
    }
    return selected, audit


def candidate_to_json(candidate: Candidate) -> dict[str, Any]:
    return {
        "scale_axis_0": candidate.scale_axis_0,
        "scale_axis_1": candidate.scale_axis_1,
        "angle_deg": candidate.angle_deg,
        "translation_coefficients": list(candidate.translation_coefficients),
        "translation_length": float(np.linalg.norm(candidate.translation_coefficients)),
        "score": candidate.score,
        "metrics": candidate.metrics,
    }


def make_overlay(
    source: Image.Image | None,
    observed: np.ndarray,
    rendered: np.ndarray,
) -> Image.Image:
    if source is None:
        background = np.full((*observed.shape, 3), 32, dtype=np.uint8)
    else:
        background = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
    overlay = background.astype(np.float32)
    for selection, color in (
        (observed & ~rendered, np.asarray([0, 220, 120], dtype=np.float32)),
        (rendered & ~observed, np.asarray([255, 70, 70], dtype=np.float32)),
        (observed & rendered, np.asarray([255, 210, 0], dtype=np.float32)),
    ):
        overlay[selection] = overlay[selection] * 0.35 + color * 0.65
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def support_and_envelope_review(
    *,
    world_vertices: np.ndarray,
    source_world_vertices: np.ndarray,
    report: dict[str, Any],
    observed: np.ndarray,
    rendered: np.ndarray,
    gravity_direction: np.ndarray,
    maximum_support_drift: float,
    maximum_anchor_overflow_ratio: float,
    maximum_support_edge_error_px: float,
    front_direction: np.ndarray | None = None,
) -> dict[str, Any]:
    gravity = np.array(gravity_direction, dtype=np.float64, copy=True)
    gravity /= np.linalg.norm(gravity)
    source_support = float(np.max(np.einsum("ni,i->n", source_world_vertices, gravity)))
    fitted_support = float(np.max(np.einsum("ni,i->n", world_vertices, gravity)))
    support_drift = abs(fitted_support - source_support)
    target = report["target_obb"]
    target_center = np.asarray(target["center"], dtype=np.float64)
    target_extents = np.asarray(target["extents"], dtype=np.float64)
    target_axes = np.asarray(target["axes_columns"], dtype=np.float64)
    thickness_axis = int(report["placement_evidence"]["thickness_axis"])
    tangent_indices = [index for index in range(3) if index != thickness_axis]
    local = np.einsum("ni,ij->nj", world_vertices - target_center, target_axes)
    tangent_ratio = np.abs(local[:, tangent_indices]) / (target_extents[tangent_indices] / 2.0)
    maximum_tangent_ratio = float(np.max(tangent_ratio))
    tangent_overflow_fraction = float(np.mean(np.any(tangent_ratio > 1.0, axis=1)))
    if front_direction is None:
        completion_sign = float(report["placement_evidence"]["completion_direction_sign"])
        front_normal = completion_sign * target_axes[:, thickness_axis]
    else:
        front_normal = np.array(front_direction, dtype=np.float64, copy=True)
        front_normal /= np.linalg.norm(front_normal)
    source_front_min = float(
        np.min(np.einsum("ni,i->n", source_world_vertices - target_center, front_normal))
    )
    fitted_front_min = float(
        np.min(np.einsum("ni,i->n", world_vertices - target_center, front_normal))
    )
    front_plane_additional_overshoot = max(0.0, source_front_min - fitted_front_min)
    observed_bbox = mask_bbox(observed)
    rendered_bbox = mask_bbox(rendered)
    assert observed_bbox is not None and rendered_bbox is not None
    support_edge_error_px = abs(observed_bbox[3] - rendered_bbox[3])
    gates = {
        "support_plane_drift": support_drift <= maximum_support_drift,
        "source_camera_support_edge": support_edge_error_px <= maximum_support_edge_error_px,
        "anchor_tangent_envelope": maximum_tangent_ratio <= 1.0 + maximum_anchor_overflow_ratio,
        "no_additional_front_plane_overshoot": front_plane_additional_overshoot <= 1e-6,
    }
    return {
        "status": "passed" if all(gates.values()) else "rejected",
        "claim_scope": (
            "Anchor-envelope, source-camera support-edge, and preserved-front-plane proxies; "
            "not a signed scene-mesh penetration proof."
        ),
        "gravity_direction": gravity.tolist(),
        "metrics": {
            "source_support_coordinate": source_support,
            "fitted_support_coordinate": fitted_support,
            "support_drift": support_drift,
            "source_camera_support_edge_error_px": support_edge_error_px,
            "maximum_tangent_anchor_ratio": maximum_tangent_ratio,
            "tangent_overflow_vertex_fraction": tangent_overflow_fraction,
            "source_front_plane_min": source_front_min,
            "fitted_front_plane_min": fitted_front_min,
            "additional_front_plane_overshoot": front_plane_additional_overshoot,
        },
        "thresholds": {
            "maximum_support_drift": maximum_support_drift,
            "maximum_support_edge_error_px": maximum_support_edge_error_px,
            "maximum_anchor_overflow_ratio": maximum_anchor_overflow_ratio,
        },
        "gates": gates,
    }


def refine_scene_fit(
    *,
    mesh_path: Path,
    scene_fit_report_path: Path,
    cameras_path: Path,
    frame_id: str,
    observed_mask_path: Path,
    output_dir: Path,
    source_frame_path: Path | None = None,
    image_y_axis: str = "down",
    minimum_scale: float = 0.75,
    maximum_scale: float = 1.35,
    maximum_angle_deg: float = 55.0,
    maximum_translation_fraction: float = 0.08,
    minimum_mask_iou: float = 0.65,
    minimum_bbox_iou: float = 0.8,
    maximum_center_error_px: float = 20.0,
    gravity_direction: tuple[float, float, float] = (0.0, 1.0, 0.0),
    maximum_support_drift_fraction: float = 0.15,
    maximum_anchor_overflow_ratio: float = 0.10,
    maximum_support_edge_error_px: float = 12.0,
    coarse_resolution: int = 24,
    fine_resolution: int = 40,
    transform_basis_mode: str = "target_obb_tangent",
    occluder_mask_paths: tuple[Path, ...] = (),
    support_plane_report_path: Path | None = None,
) -> dict[str, Any]:
    report = json.loads(scene_fit_report_path.read_text(encoding="utf-8"))
    if maximum_translation_fraction <= 0:
        raise ValueError("maximum_translation_fraction must be positive")
    if report.get("all_acceptance_gates_passed") is not True:
        raise ValueError("source scene fit did not pass its technical gates")
    object_id = str(report.get("object_id", ""))
    if not object_id:
        raise ValueError("source scene fit has no object_id")
    target = report.get("target_obb")
    placement = report.get("placement_evidence")
    transform = report.get("baked_relative_transform")
    if (
        not isinstance(target, dict)
        or not isinstance(placement, dict)
        or not isinstance(transform, dict)
    ):
        raise ValueError("source scene fit is missing placement evidence")
    axes = np.asarray(target.get("axes_columns"), dtype=np.float64)
    base_pivot = np.asarray(transform.get("runtime_pivot"), dtype=np.float64)
    thickness_axis = int(placement.get("thickness_axis", -1))
    if axes.shape != (3, 3) or base_pivot.shape != (3,):
        raise ValueError("source scene fit contains invalid axes or pivot")
    tangent_indices = [index for index in range(3) if index != thickness_axis]
    completed_extents = np.asarray(report["completed_obb"]["extents"], dtype=np.float64)
    maximum_translation = maximum_translation_fraction * float(
        np.min(completed_extents[tangent_indices])
    )
    maximum_support_drift = maximum_support_drift_fraction * float(
        np.min(completed_extents[tangent_indices])
    )
    support_plane_evidence: dict[str, Any] | None = None
    if support_plane_report_path is not None:
        derived_gravity, support_plane_evidence = load_support_plane_evidence(
            support_plane_report_path
        )
        gravity_direction = tuple(derived_gravity.tolist())
    camera = load_camera(cameras_path, frame_id)
    if transform_basis_mode == "target_obb_tangent":
        search_axes = axes
        search_thickness_axis = thickness_axis
        front_direction = float(placement["completion_direction_sign"]) * axes[:, thickness_axis]
    elif transform_basis_mode == "camera_image_plane":
        search_axes = np.asarray(camera["rotation"], dtype=np.float64)
        search_thickness_axis = 2
        front_direction = search_axes[:, 2]
    else:
        raise ValueError(f"unsupported transform_basis_mode: {transform_basis_mode!r}")
    observed_image = Image.open(observed_mask_path).convert("L")
    if observed_image.size != (camera["width"], camera["height"]):
        raise ValueError("observed mask dimensions do not match camera")
    observed = np.asarray(observed_image) > 0
    ignored_mask = np.zeros_like(observed)
    occluder_sources: list[dict[str, Any]] = []
    for occluder_path in occluder_mask_paths:
        occluder_image = Image.open(occluder_path).convert("L")
        if occluder_image.size != observed_image.size:
            raise ValueError("occluder mask dimensions do not match observed mask")
        occluder = np.asarray(occluder_image) > 0
        ignored_mask |= occluder
        occluder_sources.append(
            {
                "path": str(occluder_path),
                "sha256": sha256_file(occluder_path),
                "pixels": int(np.count_nonzero(occluder)),
            }
        )
    observed_occluder_overlap = int(np.count_nonzero(observed & ignored_mask))
    if observed_occluder_overlap:
        raise ValueError(
            "observed object mask overlaps the explicit occluder union; masks are not depth ordered"
        )
    loaded = trimesh.load(mesh_path, force="scene")
    source_scene = loaded if isinstance(loaded, trimesh.Scene) else trimesh.Scene(loaded)
    source_mesh = source_scene.to_geometry()
    selected, search_audit = search_refinement(
        mesh=source_mesh,
        axes=search_axes,
        thickness_axis=search_thickness_axis,
        base_pivot=base_pivot,
        observed=observed,
        camera=camera,
        image_y_axis=image_y_axis,
        minimum_scale=minimum_scale,
        maximum_scale=maximum_scale,
        maximum_angle_deg=maximum_angle_deg,
        maximum_translation=maximum_translation,
        minimum_mask_iou=minimum_mask_iou,
        minimum_bbox_iou=minimum_bbox_iou,
        maximum_center_error_px=maximum_center_error_px,
        coarse_resolution=coarse_resolution,
        fine_resolution=fine_resolution,
        ignored_mask=ignored_mask,
    )
    linear = refinement_linear(
        search_axes,
        search_thickness_axis,
        selected.scale_axis_0,
        selected.scale_axis_1,
        selected.angle_deg,
    )
    search_tangent_indices = [index for index in range(3) if index != search_thickness_axis]
    tangent_basis = search_axes[:, search_tangent_indices]
    translation_coefficients = np.asarray(selected.translation_coefficients, dtype=np.float64)
    translation = tangent_basis @ translation_coefficients
    final_pivot = base_pivot + translation
    source_vertices = np.asarray(source_mesh.vertices, dtype=np.float64)
    final_vertices = np.einsum("ni,ji->nj", source_vertices, linear)
    final_rendered = project_mesh_silhouette(
        final_vertices,
        np.asarray(source_mesh.faces, dtype=np.int64),
        runtime_pivot=final_pivot,
        camera=camera,
        image_y_axis=image_y_axis,
    )
    exact_metrics = silhouette_metrics(observed, final_rendered, ignored_mask)
    silhouette_gates = {
        "mask_iou": exact_metrics["mask_iou"] >= minimum_mask_iou,
        "bbox_iou": exact_metrics["bbox_iou"] >= minimum_bbox_iou,
        "center_error_px": exact_metrics["center_error_px"] <= maximum_center_error_px,
    }
    source_world_vertices = source_vertices + base_pivot
    world_vertices = final_vertices + final_pivot
    scene_geometry_review = support_and_envelope_review(
        world_vertices=world_vertices,
        source_world_vertices=source_world_vertices,
        report=report,
        observed=observed,
        rendered=final_rendered,
        gravity_direction=np.asarray(gravity_direction, dtype=np.float64),
        maximum_support_drift=maximum_support_drift,
        maximum_anchor_overflow_ratio=maximum_anchor_overflow_ratio,
        maximum_support_edge_error_px=maximum_support_edge_error_px,
        front_direction=front_direction,
    )
    scene_geometry_review["support_direction_semantics"] = (
        "downward direction; support surface is the maximum vertex projection"
    )
    scene_geometry_review["support_plane_evidence"] = support_plane_evidence

    output_dir.mkdir(parents=True, exist_ok=False)
    output_glb = output_dir / f"{object_id}.scene-fit.glb"
    output_obj = output_dir / f"{object_id}.scene-fit.obj"
    refined_scene = copy.deepcopy(source_scene)
    affine = np.eye(4)
    affine[:3, :3] = linear
    for geometry in refined_scene.geometry.values():
        geometry.apply_transform(affine)
    refined_scene.export(output_glb)
    merged = refined_scene.to_geometry()
    output_obj.write_text(export_obj(merged, include_color=True), encoding="utf-8")
    face_areas = np.asarray(merged.area_faces)
    mesh_gates = {
        "mesh_nonempty": len(merged.vertices) > 0 and len(merged.faces) > 0,
        "mesh_finite": bool(np.isfinite(np.asarray(merged.vertices)).all()),
        "mesh_no_degenerate_faces": int(np.count_nonzero(face_areas <= 1e-12)) == 0,
        "mesh_winding_consistent": bool(merged.is_winding_consistent),
        "collision_face_budget": len(merged.faces)
        <= int(report["collision_semantics"]["max_faces"]),
        "positive_refinement_determinant": determinant_3x3(linear) > 0,
    }
    base_linear = np.asarray(transform.get("linear_row_major"), dtype=np.float64)
    combined_linear = linear @ base_linear
    refined_axes = (
        rotation_about_axis(search_axes[:, search_thickness_axis], selected.angle_deg) @ axes
    )
    refined_extents = np.ptp(np.einsum("ni,ij->nj", final_vertices, refined_axes), axis=0)
    source_front_anchor_error = float(placement.get("front_surface_anchor_error", 0.0))
    front_anchor_error = float(math.hypot(source_front_anchor_error, np.linalg.norm(translation)))
    all_gates = (
        all(silhouette_gates.values())
        and all(mesh_gates.values())
        and scene_geometry_review["status"] == "passed"
    )
    refined_report = copy.deepcopy(report)
    refined_report.update(
        {
            "schema_version": 1,
            "kind": "video2world.silhouette_refined_object_scene_fit",
            "created_at": datetime.now(UTC).isoformat(),
            "status": "passed" if all_gates else "rejected",
            "promotion_status": (
                "held_pending_pairwise_scene_browser_review"
                if all_gates
                else "rejected_by_refinement_gates"
            ),
            "sources": {
                **report.get("sources", {}),
                "source_scene_fit_mesh": {"path": str(mesh_path), "sha256": sha256_file(mesh_path)},
                "source_scene_fit_report": {
                    "path": str(scene_fit_report_path),
                    "sha256": sha256_file(scene_fit_report_path),
                },
                "refinement_script": {
                    "path": str(Path(__file__).resolve()),
                    "sha256": sha256_file(Path(__file__).resolve()),
                },
                "cameras": {"path": str(cameras_path), "sha256": sha256_file(cameras_path)},
                "observed_mask": {
                    "path": str(observed_mask_path),
                    "sha256": sha256_file(observed_mask_path),
                },
                "occluder_masks": occluder_sources,
                "support_plane_report": support_plane_evidence,
            },
            "completed_obb": {
                "center": final_pivot.tolist(),
                "extents": refined_extents.tolist(),
                "axes_columns": refined_axes.tolist(),
            },
            "baked_relative_transform": {
                **transform,
                "linear_row_major": combined_linear.tolist(),
                "runtime_pivot": final_pivot.tolist(),
                "silhouette_refinement_linear_row_major": linear.tolist(),
                "silhouette_refinement_translation_world": translation.tolist(),
            },
            "placement_evidence": {
                **placement,
                "front_surface_anchor_error": front_anchor_error,
                "refinement_preserves_search_normal_extent": True,
                "target_obb_thickness_preserved": (transform_basis_mode == "target_obb_tangent"),
                "calibrated_camera_depth_preserved": (transform_basis_mode == "camera_image_plane"),
                "refinement_translation_is_tangent_to_search_normal": bool(
                    abs(
                        float(
                            np.dot(
                                translation,
                                search_axes[:, search_thickness_axis],
                            )
                        )
                    )
                    <= 1e-8
                ),
                "calibrated_camera_depth_translation_error": abs(
                    float(np.dot(translation, camera["rotation"][:, 2]))
                ),
            },
            "silhouette_refinement": {
                "constraints": {
                    "minimum_scale": minimum_scale,
                    "maximum_scale": maximum_scale,
                    "maximum_angle_deg": maximum_angle_deg,
                    "maximum_translation_fraction": maximum_translation_fraction,
                    "maximum_translation_world": maximum_translation,
                    "transform_basis_mode": transform_basis_mode,
                    "rotation_axis": (
                        "target_obb_thickness_axis"
                        if transform_basis_mode == "target_obb_tangent"
                        else "calibrated_camera_forward_axis"
                    ),
                    "translation_plane": (
                        "target_obb_tangent_plane"
                        if transform_basis_mode == "target_obb_tangent"
                        else "calibrated_camera_image_plane"
                    ),
                    "thickness_optimized": False,
                    "manual_target_bbox_used": False,
                    "explicit_occluder_count": len(occluder_sources),
                    "observed_occluder_overlap_pixels": observed_occluder_overlap,
                },
                "selected": candidate_to_json(selected),
                "search": search_audit,
                "exact_full_mesh_metrics": exact_metrics,
                "silhouette_gates": silhouette_gates,
            },
            "scene_geometry_review": scene_geometry_review,
            "mesh": {
                **report.get("mesh", {}),
                "vertices": len(merged.vertices),
                "faces": len(merged.faces),
                "bounds": np.asarray(merged.bounds).tolist(),
                "extents": np.asarray(merged.extents).tolist(),
                "degenerate_face_count": int(np.count_nonzero(face_areas <= 1e-12)),
                "finite_vertices": bool(np.isfinite(np.asarray(merged.vertices)).all()),
                "winding_consistent": bool(merged.is_winding_consistent),
                "watertight": bool(merged.is_watertight),
                "glb_sha256": sha256_file(output_glb),
                "glb_bytes": output_glb.stat().st_size,
                "obj_sha256": sha256_file(output_obj),
                "obj_bytes": output_obj.stat().st_size,
            },
            "acceptance_gates": {
                **mesh_gates,
                **{f"source_camera_{key}": value for key, value in silhouette_gates.items()},
                **{f"scene_{key}": value for key, value in scene_geometry_review["gates"].items()},
            },
            "all_acceptance_gates_passed": all_gates,
            "promotion_blockers": (
                [
                    "pairwise_scene_interpenetration_review",
                    "accepted_clean_plate_or_static_scene_carve",
                ]
                if all_gates
                else [
                    key
                    for key, value in {
                        **silhouette_gates,
                        **mesh_gates,
                        **scene_geometry_review["gates"],
                    }.items()
                    if not value
                ]
            ),
        }
    )
    report_path = output_dir / "scene_fit_report.json"
    write_json(report_path, refined_report)

    source_frame = Image.open(source_frame_path) if source_frame_path else None
    if source_frame is not None and source_frame.size != observed_image.size:
        raise ValueError("source frame dimensions do not match observed mask")
    review_dir = output_dir / "source_camera_review"
    review_dir.mkdir(parents=True)
    overlay_path = review_dir / "source_camera_silhouette_overlay.png"
    rendered_path = review_dir / "rendered_silhouette.png"
    evaluated_rendered = final_rendered & ~ignored_mask
    make_overlay(source_frame, observed & ~ignored_mask, evaluated_rendered).save(overlay_path)
    Image.fromarray(evaluated_rendered.astype(np.uint8) * 255).save(rendered_path)
    full_rendered_path = review_dir / "rendered_silhouette_full.png"
    Image.fromarray(final_rendered.astype(np.uint8) * 255).save(full_rendered_path)
    camera_receipt = {
        "schema_version": 1,
        "kind": "video2world.source_camera_silhouette_review",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "passed" if all(silhouette_gates.values()) else "rejected",
        "promotion_allowed": all(silhouette_gates.values()),
        "frame_id": frame_id,
        "camera_convention": {"rotation": "camera_to_world", "image_y_axis": image_y_axis},
        "sources": {
            "mesh": {"path": str(output_glb), "sha256": sha256_file(output_glb)},
            "scene_fit_report": {"path": str(report_path), "sha256": sha256_file(report_path)},
            "cameras": {"path": str(cameras_path), "sha256": sha256_file(cameras_path)},
            "observed_mask": {
                "path": str(observed_mask_path),
                "sha256": sha256_file(observed_mask_path),
            },
            "occluder_masks": occluder_sources,
        },
        "metrics": exact_metrics,
        "thresholds": {
            "minimum_mask_iou": minimum_mask_iou,
            "minimum_bbox_iou": minimum_bbox_iou,
            "maximum_center_error_px": maximum_center_error_px,
        },
        "gates": silhouette_gates,
        "outputs": {
            "overlay": {"path": str(overlay_path), "sha256": sha256_file(overlay_path)},
            "rendered_mask": {"path": str(rendered_path), "sha256": sha256_file(rendered_path)},
            "full_rendered_mask": {
                "path": str(full_rendered_path),
                "sha256": sha256_file(full_rendered_path),
            },
        },
    }
    write_json(review_dir / "source_camera_silhouette_review.json", camera_receipt)
    write_json(output_dir / "scene_geometry_review.json", scene_geometry_review)
    return refined_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--scene-fit-report", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--frame-id", required=True)
    parser.add_argument("--observed-mask", type=Path, required=True)
    parser.add_argument("--source-frame", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-y-axis", choices=("down", "up"), default="down")
    parser.add_argument("--minimum-scale", type=float, default=0.75)
    parser.add_argument("--maximum-scale", type=float, default=1.35)
    parser.add_argument("--maximum-angle-deg", type=float, default=55.0)
    parser.add_argument("--maximum-translation-fraction", type=float, default=0.08)
    parser.add_argument("--minimum-mask-iou", type=float, default=0.65)
    parser.add_argument("--minimum-bbox-iou", type=float, default=0.8)
    parser.add_argument("--maximum-center-error-px", type=float, default=20.0)
    parser.add_argument("--gravity-direction", type=float, nargs=3, default=(0.0, 1.0, 0.0))
    parser.add_argument("--maximum-support-drift-fraction", type=float, default=0.15)
    parser.add_argument("--maximum-anchor-overflow-ratio", type=float, default=0.10)
    parser.add_argument("--maximum-support-edge-error-px", type=float, default=12.0)
    parser.add_argument("--coarse-resolution", type=int, default=24)
    parser.add_argument("--fine-resolution", type=int, default=40)
    parser.add_argument(
        "--transform-basis-mode",
        choices=("target_obb_tangent", "camera_image_plane"),
        default="target_obb_tangent",
    )
    parser.add_argument("--occluder-mask", type=Path, action="append", default=[])
    parser.add_argument("--support-plane-report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = refine_scene_fit(
        mesh_path=args.mesh.expanduser().resolve(),
        scene_fit_report_path=args.scene_fit_report.expanduser().resolve(),
        cameras_path=args.cameras.expanduser().resolve(),
        frame_id=args.frame_id,
        observed_mask_path=args.observed_mask.expanduser().resolve(),
        source_frame_path=args.source_frame.expanduser().resolve() if args.source_frame else None,
        output_dir=args.output_dir.expanduser().resolve(),
        image_y_axis=args.image_y_axis,
        minimum_scale=args.minimum_scale,
        maximum_scale=args.maximum_scale,
        maximum_angle_deg=args.maximum_angle_deg,
        maximum_translation_fraction=args.maximum_translation_fraction,
        minimum_mask_iou=args.minimum_mask_iou,
        minimum_bbox_iou=args.minimum_bbox_iou,
        maximum_center_error_px=args.maximum_center_error_px,
        gravity_direction=tuple(args.gravity_direction),
        maximum_support_drift_fraction=args.maximum_support_drift_fraction,
        maximum_anchor_overflow_ratio=args.maximum_anchor_overflow_ratio,
        maximum_support_edge_error_px=args.maximum_support_edge_error_px,
        coarse_resolution=args.coarse_resolution,
        fine_resolution=args.fine_resolution,
        transform_basis_mode=args.transform_basis_mode,
        occluder_mask_paths=tuple(path.expanduser().resolve() for path in args.occluder_mask),
        support_plane_report_path=(
            args.support_plane_report.expanduser().resolve() if args.support_plane_report else None
        ),
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if report["all_acceptance_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
