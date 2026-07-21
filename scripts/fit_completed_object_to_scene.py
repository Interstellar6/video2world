#!/usr/bin/env python3
"""Fit a completed object asset to physical anchors and calibrated masks.

The output contract deliberately contains both a canonical-local asset for an
interactive runtime and a fully world-baked asset for tools which do not apply
the receipt transform.  GLB is the PBR authority; OBJ is geometry-only.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image
from plyfile import PlyData
from trimesh.exchange.obj import export_obj

from scripts.qa_scene_camera_silhouette import (
    bbox_iou,
    load_camera,
    mask_bbox,
    mask_center,
    project_mesh_silhouette,
)

AXIS_VECTORS = {
    "+X": np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
    "-X": np.asarray([-1.0, 0.0, 0.0], dtype=np.float64),
    "+Y": np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
    "-Y": np.asarray([0.0, -1.0, 0.0], dtype=np.float64),
    "+Z": np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
    "-Z": np.asarray([0.0, 0.0, -1.0], dtype=np.float64),
}


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
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise ValueError(f"PLY hash mismatch for {path}")
    ply = PlyData.read(path)
    if "vertex" not in ply:
        raise ValueError(f"PLY has no vertex element: {path}")
    vertex = ply["vertex"].data
    names = set(vertex.dtype.names or ())
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError(f"PLY is missing coordinates: {path}")
    points = np.column_stack([vertex[name] for name in ("x", "y", "z")]).astype(np.float64)
    if len(points) < 64 or not np.isfinite(points).all():
        raise ValueError(f"PLY requires at least 64 finite points: {path}")
    return points, observed_sha256


def sampled(points: np.ndarray, limit: int = 250_000) -> tuple[np.ndarray, int]:
    stride = max(1, len(points) // limit)
    return np.asarray(points[::stride], dtype=np.float64), stride


def _deterministic_axes(axes: np.ndarray) -> np.ndarray:
    result = np.asarray(axes, dtype=np.float64).copy()
    for axis_index in range(3):
        column = result[:, axis_index]
        dominant = int(np.argmax(np.abs(column)))
        if column[dominant] < 0:
            result[:, axis_index] *= -1
    if np.linalg.det(result) < 0:
        result[:, 2] *= -1
    return result


def robust_oriented_frame(
    points: np.ndarray,
    *,
    robust_quantiles: tuple[float, float] = (0.005, 0.995),
) -> dict[str, Any]:
    """Return a deterministic PCA frame with quantile rather than min/max bounds."""
    lower_q, upper_q = robust_quantiles
    if not 0 <= lower_q < upper_q <= 1:
        raise ValueError("robust quantiles must satisfy 0 <= lower < upper <= 1")
    values, stride = sampled(points)
    seed = np.median(values, axis=0)
    distance = np.linalg.norm(values - seed, axis=1)
    trim_limit = float(np.quantile(distance, upper_q))
    trimmed = values[distance <= trim_limit]
    minimum_points = min(32, len(values))
    if len(trimmed) < minimum_points or len(trimmed) < 4:
        raise ValueError("too few points remain after robust trimming")
    covariance = np.cov((trimmed - np.median(trimmed, axis=0)).T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    axes = _deterministic_axes(eigenvectors[:, order])
    projected = np.einsum("ni,ij->nj", values, axes, optimize=False)
    lower, upper = np.quantile(projected, robust_quantiles, axis=0)
    extents = upper - lower
    if np.any(extents <= 1e-9) or not np.isfinite(extents).all():
        raise ValueError("robust oriented extents are invalid")
    center_local = (lower + upper) / 2.0
    center = np.einsum("ij,j->i", axes, center_local, optimize=False)
    return {
        "center": center,
        "axes_columns": axes,
        "extents": extents,
        "lower": lower,
        "upper": upper,
        "eigenvalues_descending": eigenvalues[order],
        "quantiles": robust_quantiles,
        "sample_stride": stride,
        "trimmed_points": len(trimmed),
    }


def right_handed_axis_permutations() -> list[np.ndarray]:
    """Enumerate the 24 proper signed permutation matrices in SO(3)."""
    result: list[np.ndarray] = []
    for permutation in itertools.permutations(range(3)):
        base = np.eye(3, dtype=np.float64)[list(permutation)]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            candidate = np.diag(signs) @ base
            if np.isclose(np.linalg.det(candidate), 1.0, atol=1e-12):
                result.append(candidate)
    if len(result) != 24:
        raise AssertionError("proper signed axis enumeration must contain 24 matrices")
    return result


def _load_flattened_geometries(
    path: Path,
) -> tuple[list[tuple[str, trimesh.Trimesh]], dict[str, Any]]:
    if path.suffix.lower() not in {".glb", ".gltf", ".obj"}:
        raise ValueError("mesh source must be GLB, glTF, or OBJ")
    loaded = trimesh.load(path, force="scene")
    source_scene = loaded if isinstance(loaded, trimesh.Scene) else trimesh.Scene(loaded)
    flattened: list[tuple[str, trimesh.Trimesh]] = []
    graph_nodes = []
    for index, node_name in enumerate(sorted(source_scene.graph.nodes_geometry)):
        matrix, geometry_name = source_scene.graph[node_name]
        geometry = source_scene.geometry[geometry_name]
        if not isinstance(geometry, trimesh.Trimesh):
            continue
        copied = geometry.copy()
        copied.apply_transform(np.asarray(matrix, dtype=np.float64))
        flattened.append((f"geometry_{index:03d}", copied))
        graph_nodes.append(
            {
                "source_node": node_name,
                "source_geometry": geometry_name,
                "matrix_row_major": np.asarray(matrix, dtype=np.float64).tolist(),
            }
        )
    if not flattened:
        raise ValueError(f"mesh source contains no triangle geometry: {path}")
    material_types = [
        type(getattr(getattr(geometry, "visual", None), "material", None)).__name__
        for _, geometry in flattened
    ]
    return flattened, {
        "source_graph_nodes": graph_nodes,
        "geometry_count": len(flattened),
        "material_types": material_types,
    }


def _build_transformed_scene(
    geometries: list[tuple[str, trimesh.Trimesh]],
    matrix: np.ndarray,
    *,
    object_id: str,
) -> trimesh.Scene:
    scene = trimesh.Scene(base_frame="world")
    scene.graph.update(frame_from="world", frame_to=object_id, matrix=np.eye(4))
    for index, (source_name, geometry) in enumerate(geometries):
        copied = geometry.copy()
        copied.apply_transform(matrix)
        scene.add_geometry(
            copied,
            node_name=f"{object_id}__part_{index:03d}",
            geom_name=f"{object_id}.{source_name}",
            parent_node_name=object_id,
        )
    return scene


def _scene_mesh(scene: trimesh.Scene) -> trimesh.Trimesh:
    mesh = scene.to_geometry()
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
        raise ValueError("scene has no merged triangle mesh")
    return mesh


def _cluster_mesh(mesh: trimesh.Trimesh, bins: int) -> trimesh.Trimesh:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    lower = vertices.min(axis=0)
    span = np.maximum(vertices.max(axis=0) - lower, 1e-12)
    cells = np.floor((vertices - lower) / span * bins).astype(np.int64)
    cells = np.clip(cells, 0, bins - 1)
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    clustered_vertices = np.column_stack(
        [np.bincount(inverse, weights=vertices[:, axis]) / counts for axis in range(3)]
    )
    clustered_faces = inverse[faces]
    nondegenerate = (
        (clustered_faces[:, 0] != clustered_faces[:, 1])
        & (clustered_faces[:, 1] != clustered_faces[:, 2])
        & (clustered_faces[:, 0] != clustered_faces[:, 2])
    )
    clustered_faces = clustered_faces[nondegenerate]
    if len(clustered_faces) == 0:
        raise ValueError("vertex clustering removed every face")
    canonical_faces = np.sort(clustered_faces, axis=1)
    _, unique_indices = np.unique(canonical_faces, axis=0, return_index=True)
    clustered = trimesh.Trimesh(
        vertices=clustered_vertices,
        faces=clustered_faces[np.sort(unique_indices)],
        process=False,
    )
    clustered.remove_unreferenced_vertices()
    return clustered


def make_search_mesh(
    mesh: trimesh.Trimesh,
    *,
    maximum_faces: int,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Build a deterministic geometry-only mesh for repeated silhouette search."""
    if maximum_faces < 100:
        raise ValueError("maximum_search_faces must be at least 100")
    if len(mesh.faces) <= maximum_faces:
        return mesh.copy(), {
            "method": "full_mesh_below_limit",
            "source_faces": len(mesh.faces),
            "search_faces": len(mesh.faces),
            "search_vertices": len(mesh.vertices),
            "maximum_faces": maximum_faces,
        }
    selected = None
    selected_bins = None
    for bins in (64, 56, 48, 40, 32, 28, 24, 20, 16, 12, 10, 8, 6, 4):
        candidate = _cluster_mesh(mesh, bins)
        if len(candidate.faces) <= maximum_faces:
            selected = candidate
            selected_bins = bins
            break
    if selected is None:
        selected = _cluster_mesh(mesh, 3)
        selected_bins = 3
    return selected, {
        "method": "deterministic_vertex_clustering",
        "source_faces": len(mesh.faces),
        "search_faces": len(selected.faces),
        "search_vertices": len(selected.vertices),
        "maximum_faces": maximum_faces,
        "grid_bins_per_axis": selected_bins,
    }


def _write_scene_pair(
    scene: trimesh.Scene,
    *,
    glb_path: Path,
    obj_path: Path,
) -> None:
    glb_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(glb_path)
    obj_path.write_text(export_obj(_scene_mesh(scene), include_color=True), encoding="utf-8")


def _audit_geometry(path: Path) -> dict[str, Any]:
    loaded = trimesh.load(path, force="scene")
    scene = loaded if isinstance(loaded, trimesh.Scene) else trimesh.Scene(loaded)
    mesh = _scene_mesh(scene)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    bounds = np.asarray([vertices.min(axis=0), vertices.max(axis=0)])
    material_types = [
        type(getattr(getattr(geometry, "visual", None), "material", None)).__name__
        for geometry in scene.geometry.values()
    ]
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "bounds": bounds.tolist(),
        "extents": (bounds[1] - bounds[0]).tolist(),
        "center": ((bounds[0] + bounds[1]) / 2.0).tolist(),
        "finite_vertices": bool(np.isfinite(vertices).all()),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "material_types": material_types,
    }


def _geometry_vertices(path: Path) -> np.ndarray:
    loaded = trimesh.load(path, force="scene")
    scene = loaded if isinstance(loaded, trimesh.Scene) else trimesh.Scene(loaded)
    return np.asarray(_scene_mesh(scene).vertices, dtype=np.float64)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def load_view_manifest(
    path: Path,
    *,
    object_id: str,
    cameras_path: Path,
) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    declared_object = payload.get("object_id")
    if declared_object is not None and declared_object != object_id:
        raise ValueError("view manifest object_id does not match")
    values = payload.get("views")
    if not isinstance(values, list) or not values:
        raise ValueError("view manifest requires at least one view")
    result = []
    for index, value in enumerate(values):
        if not isinstance(value, dict) or not isinstance(value.get("frame_id"), str):
            raise ValueError("view entries require frame_id")
        frame_id = value["frame_id"]
        observed_value = value.get("observed_mask")
        if isinstance(observed_value, dict):
            observed_path_value = observed_value.get("path")
            expected_observed = observed_value.get("sha256")
        else:
            observed_path_value = observed_value
            expected_observed = value.get("observed_mask_sha256")
        observed_path = resolve_path(str(observed_path_value), relative_to=path.parent)
        if not observed_path.is_file():
            raise FileNotFoundError(observed_path)
        if expected_observed is not None and sha256_file(observed_path) != expected_observed:
            raise ValueError(f"observed-mask hash mismatch for {frame_id}")
        observed = np.asarray(Image.open(observed_path).convert("L")) > 0
        camera = load_camera(cameras_path, frame_id)
        if observed.shape != (camera["height"], camera["width"]):
            raise ValueError(f"observed-mask dimensions do not match camera for {frame_id}")
        occluder_union = np.zeros_like(observed)
        occluders = []
        for occluder in value.get("occluder_masks", []):
            if isinstance(occluder, dict):
                occluder_value = occluder.get("path")
                expected = occluder.get("sha256")
            else:
                occluder_value = occluder
                expected = None
            occluder_path = resolve_path(str(occluder_value), relative_to=path.parent)
            if not occluder_path.is_file():
                raise FileNotFoundError(occluder_path)
            if expected is not None and sha256_file(occluder_path) != expected:
                raise ValueError(f"occluder-mask hash mismatch for {frame_id}")
            mask = np.asarray(Image.open(occluder_path).convert("L")) > 0
            if mask.shape != observed.shape:
                raise ValueError(f"occluder dimensions do not match camera for {frame_id}")
            occluder_union |= mask
            occluders.append({"path": str(occluder_path), "sha256": sha256_file(occluder_path)})
        if not np.any(observed & ~occluder_union):
            raise ValueError(f"visible observed mask is empty for {frame_id}")
        result.append(
            {
                "index": index,
                "frame_id": frame_id,
                "camera": camera,
                "observed_path": observed_path,
                "observed_sha256": sha256_file(observed_path),
                "observed": observed & ~occluder_union,
                "occluder_union": occluder_union,
                "occluders": occluders,
                "is_source_view": bool(value.get("is_source_view", index == 0)),
            }
        )
    if len({view["frame_id"] for view in result}) != len(result):
        raise ValueError("view manifest contains duplicate frame IDs")
    return result


def load_support_contact_manifest(
    path: Path,
    *,
    object_id: str,
    views: list[dict[str, Any]],
) -> dict[str, Any]:
    """Attach independently hash-bound support-contact masks to calibrated views."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("object_id") != object_id:
        raise ValueError("support contact manifest object_id does not match")
    values = payload.get("views")
    if not isinstance(values, list) or not values:
        raise ValueError("support contact manifest requires at least one view")
    by_frame = {view["frame_id"]: view for view in views}
    observed_frames: set[str] = set()
    records = []
    for value in values:
        if not isinstance(value, dict) or not isinstance(value.get("frame_id"), str):
            raise ValueError("support contact entries require frame_id")
        frame_id = value["frame_id"]
        if frame_id in observed_frames:
            raise ValueError("support contact manifest contains duplicate frame IDs")
        if frame_id not in by_frame:
            raise ValueError(f"support contact frame is absent from view manifest: {frame_id}")
        mask_value = value.get("support_contact_mask")
        if isinstance(mask_value, dict):
            mask_path_value = mask_value.get("path")
            expected_sha256 = mask_value.get("sha256")
        else:
            mask_path_value = mask_value
            expected_sha256 = value.get("support_contact_mask_sha256")
        mask_path = resolve_path(str(mask_path_value), relative_to=path.parent)
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        observed_sha256 = sha256_file(mask_path)
        if expected_sha256 is not None and observed_sha256 != expected_sha256:
            raise ValueError(f"support-contact-mask hash mismatch for {frame_id}")
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        view = by_frame[frame_id]
        expected_shape = (view["camera"]["height"], view["camera"]["width"])
        if mask.shape != expected_shape or mask_bbox(mask) is None:
            raise ValueError(f"support contact mask is empty or has wrong dimensions: {frame_id}")
        view["support_contact_mask"] = mask
        view["support_contact_path"] = mask_path
        view["support_contact_sha256"] = observed_sha256
        observed_frames.add(frame_id)
        records.append(
            {
                "frame_id": frame_id,
                "support_contact_mask": {
                    "path": str(mask_path),
                    "sha256": observed_sha256,
                },
            }
        )
    expected_frames = set(by_frame)
    if observed_frames != expected_frames:
        missing = sorted(expected_frames - observed_frames)
        raise ValueError(f"support contact manifest is missing view frames: {missing}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "views": records,
    }


def _support_alignment(
    object_points: np.ndarray,
    support_points: np.ndarray,
    camera_positions: np.ndarray,
    *,
    robust_quantiles: tuple[float, float],
) -> dict[str, Any]:
    object_sample, object_stride = sampled(object_points)
    support_sample, support_stride = sampled(support_points)
    support_seed = np.median(support_sample, axis=0)
    support_distance = np.linalg.norm(support_sample - support_seed, axis=1)
    support_trimmed = support_sample[
        support_distance <= np.quantile(support_distance, robust_quantiles[1])
    ]
    covariance = np.cov((support_trimmed - np.median(support_trimmed, axis=0)).T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    down = eigenvectors[:, int(np.argmin(eigenvalues))]
    support_offset = float(np.median(np.einsum("ni,i->n", support_sample, down, optimize=False)))
    object_seed = np.median(object_sample, axis=0)
    if float(np.einsum("i,i->", object_seed, down, optimize=False)) > support_offset:
        down *= -1
        support_offset *= -1
    up = -down

    object_covariance = np.cov((object_sample - object_seed).T)
    object_values, object_vectors = np.linalg.eigh(object_covariance)
    length_axis = object_vectors[:, int(np.argmax(object_values))]
    length_axis -= up * float(np.einsum("i,i->", length_axis, up, optimize=False))
    if np.linalg.norm(length_axis) <= 1e-8:
        candidates = np.eye(3) - np.outer(up, up)
        length_axis = candidates[int(np.argmax(np.linalg.norm(candidates, axis=1)))]
    length_axis /= np.linalg.norm(length_axis)
    toward_camera = np.mean(camera_positions, axis=0) - object_seed
    toward_camera -= up * float(np.einsum("i,i->", toward_camera, up, optimize=False))
    if float(np.einsum("i,i->", length_axis, toward_camera, optimize=False)) < 0:
        length_axis *= -1
    width_axis = np.cross(up, length_axis)
    width_axis /= np.linalg.norm(width_axis)
    axes = np.column_stack((width_axis, up, length_axis))

    projected = np.einsum("ni,ij->nj", object_sample, axes, optimize=False)
    lower, upper = np.quantile(projected, robust_quantiles, axis=0)
    extents = upper - lower
    center = np.einsum("ij,j->i", axes, (lower + upper) / 2.0, optimize=False)
    support_distances = np.einsum("ni,i->n", support_sample, down, optimize=False) - support_offset
    return {
        "center": center,
        "axes_columns": axes,
        "extents": extents,
        "support": {
            "down_normal": down,
            "up_normal": up,
            "plane_offset": support_offset,
            "plane_equation": np.append(down, -support_offset),
            "eigenvalues_ascending": eigenvalues,
            "distance_median_abs": float(np.median(np.abs(support_distances))),
            "distance_p95_abs": float(np.quantile(np.abs(support_distances), 0.95)),
            "object_analysis_stride": object_stride,
            "support_analysis_stride": support_stride,
        },
    }


def _rotation_xyz(angles: np.ndarray) -> np.ndarray:
    x, y, z = angles
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)
    rx = np.asarray([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry = np.asarray([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz = np.asarray([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def _semantic_up_alignment(
    linear: np.ndarray,
    *,
    canonical_source_up: np.ndarray,
    world_up: np.ndarray,
) -> dict[str, Any]:
    mapped = np.einsum("ij,j->i", linear, canonical_source_up, optimize=False)
    norm = float(np.linalg.norm(mapped))
    if norm <= 1e-12 or not np.isfinite(norm):
        return {
            "mapped_source_up_world": [float("nan")] * 3,
            "dot": -1.0,
            "tilt_degrees": 180.0,
        }
    mapped /= norm
    dot = float(np.clip(np.dot(mapped, world_up), -1.0, 1.0))
    return {
        "mapped_source_up_world": mapped.tolist(),
        "dot": dot,
        "tilt_degrees": float(np.degrees(np.arccos(dot))),
    }


def _apply_plane_constraints(
    linear: np.ndarray,
    translation: np.ndarray,
    canonical_vertices: np.ndarray,
    *,
    support: dict[str, Any] | None,
    front: dict[str, Any] | None,
) -> tuple[np.ndarray, dict[str, float]]:
    rotated = np.einsum("ni,ji->nj", canonical_vertices, linear, optimize=False)
    rows = []
    residuals = []
    if support is not None:
        down = np.asarray(support["down_normal"], dtype=np.float64)
        current = float(np.max(np.einsum("ni,i->n", rotated + translation, down, optimize=False)))
        rows.append(down)
        residuals.append(float(support["plane_offset"]) - current)
    if front is not None:
        direction = np.asarray(front["view_direction"], dtype=np.float64)
        depths = np.einsum("ni,i->n", rotated + translation, direction, optimize=False)
        current = float(np.quantile(depths, front["mesh_surface_quantile"]))
        rows.append(direction)
        residuals.append(float(front["plane_offset"]) - current)
    if rows:
        correction = np.linalg.lstsq(
            np.asarray(rows, dtype=np.float64),
            np.asarray(residuals, dtype=np.float64),
            rcond=None,
        )[0]
        translation = translation + correction
    errors: dict[str, float] = {}
    world = rotated + translation
    if support is not None:
        errors["support_plane_error"] = abs(
            float(
                np.max(np.einsum("ni,i->n", world, support["down_normal"], optimize=False))
                - support["plane_offset"]
            )
        )
    if front is not None:
        depths = np.einsum("ni,i->n", world, front["view_direction"], optimize=False)
        errors["visible_front_plane_error"] = abs(
            float(np.quantile(depths, front["mesh_surface_quantile"]) - front["plane_offset"])
        )
    return translation, errors


def _mask_metrics(observed: np.ndarray, rendered: np.ndarray) -> dict[str, Any]:
    observed_bbox = mask_bbox(observed)
    rendered_bbox = mask_bbox(rendered)
    union = int(np.count_nonzero(observed | rendered))
    if observed_bbox is None or rendered_bbox is None or union == 0:
        return {
            "mask_iou": 0.0,
            "bbox_iou": 0.0,
            "center_error_px": float("inf"),
            "observed_pixels": int(np.count_nonzero(observed)),
            "rendered_pixels": int(np.count_nonzero(rendered)),
        }
    intersection = int(np.count_nonzero(observed & rendered))
    return {
        "mask_iou": intersection / union,
        "bbox_iou": bbox_iou(observed_bbox, rendered_bbox),
        "center_error_px": float(np.linalg.norm(mask_center(observed) - mask_center(rendered))),
        "observed_pixels": int(np.count_nonzero(observed)),
        "rendered_pixels": int(np.count_nonzero(rendered)),
    }


def _evaluate_views(
    mesh: trimesh.Trimesh,
    views: list[dict[str, Any]],
    matrix: np.ndarray,
    *,
    center_error_score_weight: float,
    support_contact_error_score_weight: float = 0.0,
) -> tuple[float, list[dict[str, Any]]]:
    records = []
    for view in views:
        rendered = project_mesh_silhouette(
            mesh,
            runtime_pivot=None,
            camera=view["camera"],
            scene_transform=matrix,
        )
        rendered = rendered & ~view["occluder_union"]
        metrics = _mask_metrics(view["observed"], rendered)
        metrics["frame_id"] = view["frame_id"]
        contact_mask = view.get("support_contact_mask")
        if contact_mask is not None:
            rendered_bbox = mask_bbox(rendered)
            contact_bbox = mask_bbox(contact_mask)
            if rendered_bbox is None or contact_bbox is None:
                signed_error = float("inf")
                rendered_bottom = None
                observed_bottom = None
            else:
                rendered_bottom = rendered_bbox[3] - 1
                observed_bottom = contact_bbox[3] - 1
                signed_error = float(rendered_bottom - observed_bottom)
            metrics["source_camera_support_contact"] = {
                "signed_bottom_error_px": signed_error,
                "absolute_bottom_error_px": abs(signed_error),
                "floating_gap_px": max(0.0, -signed_error),
                "image_plane_penetration_px": max(0.0, signed_error),
                "rendered_geometry_bottom_y_inclusive": rendered_bottom,
                "observed_contact_mask_bottom_y_inclusive": observed_bottom,
                "measurement": "calibrated_geometry_silhouette_vs_observed_contact_mask",
            }
        records.append(metrics)
    minimum_iou = min(record["mask_iou"] for record in records)
    mean_iou = float(np.mean([record["mask_iou"] for record in records]))
    minimum_bbox = min(record["bbox_iou"] for record in records)
    finite_centers = [record["center_error_px"] for record in records]
    maximum_center = max(finite_centers)
    score = minimum_iou + 0.35 * mean_iou + 0.08 * minimum_bbox
    score -= center_error_score_weight * maximum_center if np.isfinite(maximum_center) else 1e6
    contact_errors = [
        record["source_camera_support_contact"]["absolute_bottom_error_px"]
        for record in records
        if "source_camera_support_contact" in record
    ]
    if contact_errors:
        maximum_contact_error = max(contact_errors)
        score -= (
            support_contact_error_score_weight * maximum_contact_error
            if np.isfinite(maximum_contact_error)
            else 1e6
        )
    return float(score), records


def _candidate_matrix(
    *,
    permutation: np.ndarray,
    parameters: np.ndarray,
    source_extents: np.ndarray,
    target: dict[str, Any],
    canonical_vertices: np.ndarray,
    maximum_scale_multiplier: float,
    maximum_scale_anisotropy: float,
    maximum_rotation_radians: float,
    maximum_translation_ratio: float,
    support: dict[str, Any] | None,
    front: dict[str, Any] | None,
    semantic_up: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    log_limit = math.log(maximum_scale_multiplier)
    log_scale = np.clip(parameters[:3], -log_limit, log_limit)
    anisotropy_log_limit = math.log(maximum_scale_anisotropy)
    observed_log_range = float(np.ptp(log_scale))
    if observed_log_range > anisotropy_log_limit:
        log_midpoint = float((np.min(log_scale) + np.max(log_scale)) / 2.0)
        log_scale = np.clip(
            log_scale,
            log_midpoint - anisotropy_log_limit / 2.0,
            log_midpoint + anisotropy_log_limit / 2.0,
        )
    angles = np.clip(parameters[3:6], -maximum_rotation_radians, maximum_rotation_radians)
    translation_limit = np.asarray(target["extents"]) * maximum_translation_ratio
    local_offset = np.clip(parameters[6:9], -translation_limit, translation_limit)
    permuted_extents = np.einsum("ij,j->i", np.abs(permutation), source_extents, optimize=False)
    extent_ratios = np.asarray(target["extents"]) / permuted_extents
    uniform_scale = float(np.median(extent_ratios))
    initial_scale = np.full(3, uniform_scale, dtype=np.float64)
    scale = initial_scale * np.exp(log_scale)
    axes = np.asarray(target["axes_columns"], dtype=np.float64)
    linear = axes @ _rotation_xyz(angles) @ np.diag(scale) @ permutation
    translation = np.asarray(target["center"], dtype=np.float64) + np.einsum(
        "ij,j->i", axes, local_offset, optimize=False
    )
    translation, constraint_errors = _apply_plane_constraints(
        linear,
        translation,
        canonical_vertices,
        support=support,
        front=front,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = linear
    matrix[:3, 3] = translation
    semantic_up_alignment = None
    if semantic_up is not None:
        semantic_up_alignment = _semantic_up_alignment(
            linear,
            canonical_source_up=np.asarray(semantic_up["canonical_source_up"], dtype=np.float64),
            world_up=np.asarray(semantic_up["world_up"], dtype=np.float64),
        )
    return matrix, {
        "initial_scale_xyz": initial_scale.tolist(),
        "initial_uniform_scale": uniform_scale,
        "target_to_source_extent_ratios_xyz": extent_ratios.tolist(),
        "permuted_source_extents_xyz": permuted_extents.tolist(),
        "scale_xyz": scale.tolist(),
        "scale_anisotropy_ratio": float(np.max(scale) / np.min(scale)),
        "log_scale_delta_xyz": log_scale.tolist(),
        "rotation_delta_degrees_xyz": np.degrees(angles).tolist(),
        "translation_offset_target_axes": local_offset.tolist(),
        "constraint_errors": constraint_errors,
        "semantic_up_alignment": semantic_up_alignment,
    }


def _optimize_candidate(
    *,
    permutation_index: int,
    permutation: np.ndarray,
    mesh: trimesh.Trimesh,
    views: list[dict[str, Any]],
    source_extents: np.ndarray,
    target: dict[str, Any],
    canonical_vertices: np.ndarray,
    support: dict[str, Any] | None,
    front: dict[str, Any] | None,
    semantic_up: dict[str, Any] | None,
    cycles: int,
    shrink: float,
    maximum_scale_multiplier: float,
    maximum_scale_anisotropy: float,
    maximum_rotation_radians: float,
    maximum_translation_ratio: float,
    center_error_score_weight: float,
) -> dict[str, Any]:
    parameters = np.zeros(9, dtype=np.float64)
    deltas = np.concatenate(
        (
            np.full(3, math.log(min(maximum_scale_multiplier, 1.12))),
            np.full(3, max(maximum_rotation_radians / 2.0, 1e-6)),
            np.asarray(target["extents"]) * min(maximum_translation_ratio, 0.08),
        )
    )

    def evaluate(
        value: np.ndarray,
    ) -> tuple[float, np.ndarray, dict[str, Any], list[dict[str, Any]]]:
        matrix, fit = _candidate_matrix(
            permutation=permutation,
            parameters=value,
            source_extents=source_extents,
            target=target,
            canonical_vertices=canonical_vertices,
            maximum_scale_multiplier=maximum_scale_multiplier,
            maximum_scale_anisotropy=maximum_scale_anisotropy,
            maximum_rotation_radians=maximum_rotation_radians,
            maximum_translation_ratio=maximum_translation_ratio,
            support=support,
            front=front,
            semantic_up=semantic_up,
        )
        score, view_records = _evaluate_views(
            mesh,
            views,
            matrix,
            center_error_score_weight=center_error_score_weight,
        )
        score -= 0.01 * float(np.linalg.norm(value[:3]))
        if semantic_up is not None:
            tilt = float(fit["semantic_up_alignment"]["tilt_degrees"])
            maximum_tilt = float(semantic_up["maximum_tilt_degrees"])
            if tilt > maximum_tilt:
                # Silhouette agreement cannot override a physically invalid gravity axis.
                score -= 1_000.0 + 10.0 * (tilt - maximum_tilt)
        return score, matrix, fit, view_records

    score, matrix, fit, records = evaluate(parameters)
    trace = [{"cycle": -1, "score": score}]
    for cycle in range(cycles):
        for parameter_index in range(len(parameters)):
            best = (score, parameters.copy(), matrix, fit, records)
            for multiplier in (-1.0, 1.0):
                trial = parameters.copy()
                trial[parameter_index] += multiplier * deltas[parameter_index]
                trial_score, trial_matrix, trial_fit, trial_records = evaluate(trial)
                if trial_score > best[0]:
                    best = (
                        trial_score,
                        trial,
                        trial_matrix,
                        trial_fit,
                        trial_records,
                    )
            score, parameters, matrix, fit, records = best
        trace.append({"cycle": cycle, "score": score})
        deltas *= shrink
    return {
        "permutation_index": permutation_index,
        "permutation_matrix": permutation.tolist(),
        "score": score,
        "parameters": parameters.tolist(),
        "matrix": matrix,
        "fit": fit,
        "views": records,
        "trace": trace,
    }


def _load_validated_initial_scene_fit(
    path: Path,
    *,
    object_id: str,
    mesh_source: Path,
    object_anchor_ply: Path,
    cameras_path: Path,
    view_manifest_path: Path,
    canonical_matrix: np.ndarray,
    source_up_axis: str | None,
    world_up_axis: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("kind") != "video2world.completed_object_scene_fit":
        raise ValueError("initial scene-fit receipt kind mismatch")
    if receipt.get("schema_version") != 1 or receipt.get("object_id") != object_id:
        raise ValueError("initial scene-fit receipt object or schema mismatch")
    if receipt.get("all_acceptance_gates_passed") is not True:
        raise ValueError("initial scene-fit receipt did not pass its technical gates")
    sources = receipt.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("initial scene-fit receipt is missing sources")
    expected_sources = {
        "mesh": mesh_source,
        "object_anchor": object_anchor_ply,
        "cameras": cameras_path,
        "view_manifest": view_manifest_path,
    }
    source_checks = {}
    for label, current_path in expected_sources.items():
        source = sources.get(label)
        observed_sha256 = sha256_file(current_path)
        if not isinstance(source, dict) or source.get("sha256") != observed_sha256:
            raise ValueError(f"initial scene-fit {label} hash does not match current input")
        source_checks[label] = {
            "path": str(current_path),
            "sha256": observed_sha256,
            "matched": True,
        }
    initial_canonical = np.asarray(
        receipt.get("canonicalization", {}).get("source_to_canonical_matrix_row_major"),
        dtype=np.float64,
    )
    if initial_canonical.shape != (4, 4) or not np.allclose(
        initial_canonical, canonical_matrix, atol=1e-10, rtol=0
    ):
        raise ValueError("initial scene-fit canonicalization does not match current source")
    matrix = np.asarray(
        receipt.get("canonical_to_world", {}).get("matrix_row_major"), dtype=np.float64
    )
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-10)
    ):
        raise ValueError("initial scene-fit canonical_to_world matrix is invalid")
    orientation = receipt.get("semantic_orientation")
    if orientation is not None:
        if source_up_axis != orientation.get("source_up_axis"):
            raise ValueError("source_up_axis does not match initial scene-fit receipt")
        if world_up_axis != orientation.get("world_up_axis"):
            raise ValueError("world_up_axis does not match initial scene-fit receipt")
    elif source_up_axis is not None or world_up_axis is not None:
        raise ValueError("initial scene-fit receipt has no semantic orientation to refine")
    gates = receipt.get("acceptance_gates", {})
    if gates.get("view_silhouettes") is not True:
        raise ValueError("initial scene-fit receipt did not pass its silhouette gate")
    if orientation is not None and gates.get("semantic_up_alignment") is not True:
        raise ValueError("initial scene-fit receipt did not pass semantic up alignment")
    return receipt, {
        "path": str(path),
        "sha256": sha256_file(path),
        "source_checks": source_checks,
        "canonicalization_matched": True,
        "initial_technical_gates_passed": True,
        "initial_view_silhouettes_passed": True,
        "initial_semantic_up_passed": orientation is None
        or gates.get("semantic_up_alignment") is True,
    }


def _world_up_tangent_basis(world_up: np.ndarray) -> np.ndarray:
    world_up = np.asarray(world_up, dtype=np.float64)
    world_up /= np.linalg.norm(world_up)
    references = np.eye(3, dtype=np.float64)
    reference = references[int(np.argmin(np.abs(references @ world_up)))]
    tangent_x = reference - world_up * float(np.dot(reference, world_up))
    tangent_x /= np.linalg.norm(tangent_x)
    tangent_z = np.cross(tangent_x, world_up)
    tangent_z /= np.linalg.norm(tangent_z)
    basis = np.column_stack((tangent_x, world_up, tangent_z))
    if not np.isclose(np.linalg.det(basis), 1.0, atol=1e-10):
        raise ValueError("world-up tangent basis is not right handed")
    return basis


def _rotation_about_axis(axis: np.ndarray, radians: float) -> np.ndarray:
    x, y, z = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(radians) * skew + (1.0 - math.cos(radians)) * (skew @ skew)


def _support_signed_contact(
    canonical_vertices: np.ndarray,
    matrix: np.ndarray,
    support: dict[str, Any],
) -> dict[str, Any]:
    world = (
        np.einsum(
            "ni,ji->nj",
            canonical_vertices,
            np.asarray(matrix[:3, :3], dtype=np.float64),
            optimize=False,
        )
        + matrix[:3, 3]
    )
    down = np.asarray(support["down_normal"], dtype=np.float64)
    signed = np.einsum("ni,i->n", world, down, optimize=False) - float(
        support["plane_offset"]
    )
    deepest = float(np.max(signed))
    return {
        "signed_distance_convention": "positive_below_support_plane",
        "deepest_signed_distance": deepest,
        "maximum_penetration": max(0.0, deepest),
        "minimum_gap": max(0.0, -deepest),
        "penetrating_vertex_count": int(np.count_nonzero(signed > 0.0)),
        "penetrating_vertex_fraction": float(np.count_nonzero(signed > 0.0) / len(signed)),
        "signed_distance_quantiles": {
            key: float(value)
            for key, value in zip(
                ("minimum", "q50", "q95", "q99", "q999", "maximum"),
                np.quantile(signed, (0.0, 0.5, 0.95, 0.99, 0.999, 1.0)),
                strict=True,
            )
        },
    }


def _initial_refinement_candidate(
    *,
    initial_matrix: np.ndarray,
    parameters: np.ndarray,
    canonical_vertices: np.ndarray,
    basis: np.ndarray,
    bottom_pivot: np.ndarray,
    support: dict[str, Any],
    front: dict[str, Any] | None,
    maximum_horizontal_scale_multiplier: float,
    maximum_vertical_compression: float,
    maximum_rotation_radians: float,
    translation_limits: np.ndarray,
    maximum_support_penetration: float,
    semantic_up: dict[str, Any] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    horizontal_log_limit = math.log(maximum_horizontal_scale_multiplier)
    horizontal_log_scale = np.clip(
        parameters[[0, 2]], -horizontal_log_limit, horizontal_log_limit
    )
    vertical_compression = float(np.clip(parameters[1], 0.0, maximum_vertical_compression))
    yaw = float(np.clip(parameters[3], -maximum_rotation_radians, maximum_rotation_radians))
    tangent_offset = np.clip(parameters[4:6], -translation_limits, translation_limits)
    requested_penetration = float(
        np.clip(parameters[6], 0.0, maximum_support_penetration)
    )
    relative_scales = np.asarray(
        [
            math.exp(horizontal_log_scale[0]),
            1.0 - vertical_compression,
            math.exp(horizontal_log_scale[1]),
        ]
    )
    world_relative = (
        _rotation_about_axis(basis[:, 1], yaw) @ basis @ np.diag(relative_scales) @ basis.T
    )
    linear = world_relative @ initial_matrix[:3, :3]
    translation = bottom_pivot + world_relative @ (initial_matrix[:3, 3] - bottom_pivot)
    translation += basis[:, 0] * tangent_offset[0] + basis[:, 2] * tangent_offset[1]

    rotated = np.einsum("ni,ji->nj", canonical_vertices, linear, optimize=False)
    down = np.asarray(support["down_normal"], dtype=np.float64)
    support_current = float(
        np.max(np.einsum("ni,i->n", rotated + translation, down, optimize=False))
    )
    rows = [down]
    residuals = [float(support["plane_offset"]) + requested_penetration - support_current]
    if front is not None:
        direction = np.asarray(front["view_direction"], dtype=np.float64)
        depths = np.einsum("ni,i->n", rotated + translation, direction, optimize=False)
        rows.append(direction)
        residuals.append(
            float(front["plane_offset"])
            - float(np.quantile(depths, front["mesh_surface_quantile"]))
        )
    correction = np.linalg.lstsq(
        np.asarray(rows, dtype=np.float64), np.asarray(residuals, dtype=np.float64), rcond=None
    )[0]
    translation += correction
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = linear
    matrix[:3, 3] = translation
    support_contact = _support_signed_contact(canonical_vertices, matrix, support)
    support_target_error = abs(
        support_contact["deepest_signed_distance"] - requested_penetration
    )
    singular_values = np.linalg.svd(linear, compute_uv=False)
    initial_singular_values = np.linalg.svd(initial_matrix[:3, :3], compute_uv=False)
    semantic_up_alignment = None
    if semantic_up is not None:
        semantic_up_alignment = _semantic_up_alignment(
            linear,
            canonical_source_up=np.asarray(semantic_up["canonical_source_up"]),
            world_up=np.asarray(semantic_up["world_up"]),
        )
    constraint_errors = {
        "support_plane_error": support_target_error,
        "support_signed_penetration": support_contact["deepest_signed_distance"],
        "support_gap": support_contact["minimum_gap"],
    }
    if front is not None:
        world = rotated + translation
        depths = np.einsum("ni,i->n", world, front["view_direction"], optimize=False)
        constraint_errors["visible_front_plane_error"] = abs(
            float(np.quantile(depths, front["mesh_surface_quantile"]) - front["plane_offset"])
        )
    return matrix, {
        "initial_scale_xyz": initial_singular_values.tolist(),
        "initial_uniform_scale": float(np.median(initial_singular_values)),
        "target_to_source_extent_ratios_xyz": [1.0, 1.0, 1.0],
        "permuted_source_extents_xyz": [1.0, 1.0, 1.0],
        "scale_xyz": singular_values.tolist(),
        "scale_anisotropy_ratio": float(np.max(singular_values) / np.min(singular_values)),
        "log_scale_delta_xyz": [
            float(horizontal_log_scale[0]),
            math.log(1.0 - vertical_compression),
            float(horizontal_log_scale[1]),
        ],
        "refinement_scale_world_up_tangent_basis": relative_scales.tolist(),
        "vertical_compression_fraction": vertical_compression,
        "rotation_delta_degrees_xyz": [0.0, math.degrees(yaw), 0.0],
        "yaw_about_world_up_degrees": math.degrees(yaw),
        "translation_offset_target_axes": [
            float(tangent_offset[0]),
            float(np.dot(correction, basis[:, 1])),
            float(tangent_offset[1]),
        ],
        "support_contact": support_contact,
        "requested_support_penetration": requested_penetration,
        "constraint_errors": constraint_errors,
        "semantic_up_alignment": semantic_up_alignment,
    }


def _optimize_initial_scene_fit(
    *,
    initial_matrix: np.ndarray,
    mesh: trimesh.Trimesh,
    views: list[dict[str, Any]],
    canonical_vertices: np.ndarray,
    support: dict[str, Any],
    front: dict[str, Any] | None,
    semantic_up: dict[str, Any] | None,
    cycles: int,
    shrink: float,
    maximum_horizontal_scale_multiplier: float,
    maximum_vertical_compression: float,
    maximum_rotation_radians: float,
    maximum_translation_ratio: float,
    maximum_support_penetration: float,
    support_penetration_score_weight: float,
    center_error_score_weight: float,
    support_contact_error_score_weight: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if semantic_up is None:
        raise ValueError("initial scene-fit refinement requires a declared semantic world up")
    basis = _world_up_tangent_basis(np.asarray(semantic_up["world_up"], dtype=np.float64))
    initial_world = (
        np.einsum(
            "ni,ji->nj", canonical_vertices, initial_matrix[:3, :3], optimize=False
        )
        + initial_matrix[:3, 3]
    )
    world_down = -basis[:, 1]
    bottom_depth = float(
        np.max(np.einsum("ni,i->n", initial_world, world_down, optimize=False))
    )
    center = np.median(initial_world, axis=0)
    bottom_pivot = center + world_down * (bottom_depth - float(np.dot(center, world_down)))
    projected = np.einsum("ni,ij->nj", initial_world, basis, optimize=False)
    projected_lower, projected_upper = np.quantile(projected, (0.005, 0.995), axis=0)
    projected_extents = projected_upper - projected_lower
    translation_limits = projected_extents[[0, 2]] * maximum_translation_ratio
    initial_contact = _support_signed_contact(canonical_vertices, initial_matrix, support)
    parameters = np.zeros(7, dtype=np.float64)
    parameters[6] = min(
        initial_contact["maximum_penetration"], maximum_support_penetration
    )
    deltas = np.asarray(
        [
            math.log(min(maximum_horizontal_scale_multiplier, 1.08)),
            min(maximum_vertical_compression / 2.0, 0.08),
            math.log(min(maximum_horizontal_scale_multiplier, 1.08)),
            max(maximum_rotation_radians / 2.0, 1e-8),
            projected_extents[0] * min(maximum_translation_ratio, 0.08),
            projected_extents[2] * min(maximum_translation_ratio, 0.08),
            max(maximum_support_penetration / 2.0, 1e-8),
        ],
        dtype=np.float64,
    )

    def evaluate(
        value: np.ndarray,
    ) -> tuple[float, np.ndarray, dict[str, Any], list[dict[str, Any]]]:
        matrix, fit = _initial_refinement_candidate(
            initial_matrix=initial_matrix,
            parameters=value,
            canonical_vertices=canonical_vertices,
            basis=basis,
            bottom_pivot=bottom_pivot,
            support=support,
            front=front,
            maximum_horizontal_scale_multiplier=maximum_horizontal_scale_multiplier,
            maximum_vertical_compression=maximum_vertical_compression,
            maximum_rotation_radians=maximum_rotation_radians,
            translation_limits=translation_limits,
            maximum_support_penetration=maximum_support_penetration,
            semantic_up=semantic_up,
        )
        score, records = _evaluate_views(
            mesh,
            views,
            matrix,
            center_error_score_weight=center_error_score_weight,
            support_contact_error_score_weight=support_contact_error_score_weight,
        )
        regularization = (
            abs(fit["log_scale_delta_xyz"][0])
            + abs(fit["log_scale_delta_xyz"][1])
            + abs(fit["log_scale_delta_xyz"][2])
        )
        score -= 0.01 * regularization
        if maximum_support_penetration > 0:
            score -= support_penetration_score_weight * (
                fit["requested_support_penetration"] / maximum_support_penetration
            )
        tilt = float(fit["semantic_up_alignment"]["tilt_degrees"])
        maximum_tilt = float(semantic_up["maximum_tilt_degrees"])
        if tilt > maximum_tilt:
            score -= 1_000.0 + 10.0 * (tilt - maximum_tilt)
        return score, matrix, fit, records

    score, matrix, fit, records = evaluate(parameters)
    trace = [
        {
            "cycle": -1,
            "score": score,
            "parameters": parameters.tolist(),
            "views": records,
        }
    ]
    for cycle in range(cycles):
        for parameter_index in range(len(parameters)):
            best = (score, parameters.copy(), matrix, fit, records)
            for multiplier in (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0):
                trial = parameters.copy()
                trial[parameter_index] += multiplier * deltas[parameter_index]
                trial_score, trial_matrix, trial_fit, trial_records = evaluate(trial)
                if trial_score > best[0]:
                    best = (
                        trial_score,
                        trial,
                        trial_matrix,
                        trial_fit,
                        trial_records,
                    )
            score, parameters, matrix, fit, records = best
        trace.append(
            {
                "cycle": cycle,
                "score": score,
                "parameters": parameters.tolist(),
                "views": records,
            }
        )
        deltas *= shrink
    return (
        {
            "permutation_index": None,
            "permutation_matrix": None,
            "score": score,
            "parameters": parameters.tolist(),
            "matrix": matrix,
            "fit": fit,
            "views": records,
            "trace": trace,
        },
        {
            "world_up_tangent_basis_columns": basis.tolist(),
            "bottom_pivot_world": bottom_pivot.tolist(),
            "initial_support_contact": initial_contact,
            "translation_limits_tangent_axes": translation_limits.tolist(),
        },
    )


def fit_completed_object_to_scene(
    *,
    object_id: str,
    mesh_source: Path,
    object_anchor_ply: Path,
    cameras_path: Path,
    view_manifest_path: Path,
    output_dir: Path,
    support_anchor_ply: Path | None = None,
    initial_scene_fit_receipt: Path | None = None,
    support_contact_manifest_path: Path | None = None,
    placement_mode: str = "volume_center",
    expected_object_anchor_sha256: str | None = None,
    expected_support_anchor_sha256: str | None = None,
    robust_quantiles: tuple[float, float] = (0.005, 0.995),
    front_quantile: float = 0.05,
    cycles: int = 2,
    shrink: float = 0.5,
    refine_axis_candidates: int = 3,
    maximum_scale_multiplier: float = 1.35,
    maximum_scale_anisotropy: float = 1.15,
    maximum_rotation_degrees: float = 15.0,
    maximum_translation_ratio: float = 0.2,
    maximum_search_faces: int = 5_000,
    minimum_mask_iou: float = 0.5,
    minimum_bbox_iou: float = 0.6,
    maximum_center_error_px: float = 30.0,
    source_up_axis: str | None = None,
    world_up_axis: str | None = None,
    maximum_up_tilt_degrees: float = 20.0,
    center_error_score_weight: float = 0.01,
    maximum_vertical_compression: float = 0.25,
    maximum_support_penetration: float = 0.0,
    support_penetration_score_weight: float = 0.01,
    support_contact_error_score_weight: float = 0.0,
    maximum_support_contact_error_px: float = 15.0,
) -> dict[str, Any]:
    if placement_mode not in {"volume_center", "observed_front_surface"}:
        raise ValueError("placement_mode must be volume_center or observed_front_surface")
    if cycles < 0 or not 0 < shrink < 1:
        raise ValueError("cycles must be non-negative and shrink must be in (0, 1)")
    if not 1 <= refine_axis_candidates <= 24:
        raise ValueError("refine_axis_candidates must be in [1, 24]")
    if maximum_scale_multiplier < 1:
        raise ValueError("maximum_scale_multiplier must be at least one")
    if maximum_scale_anisotropy < 1:
        raise ValueError("maximum_scale_anisotropy must be at least one")
    if maximum_search_faces < 100:
        raise ValueError("maximum_search_faces must be at least 100")
    if not 0 <= front_quantile < 0.5:
        raise ValueError("front_quantile must be in [0, 0.5)")
    if (source_up_axis is None) != (world_up_axis is None):
        raise ValueError("source_up_axis and world_up_axis must be provided together")
    if source_up_axis is not None and source_up_axis not in AXIS_VECTORS:
        raise ValueError(f"unknown source_up_axis: {source_up_axis}")
    if world_up_axis is not None and world_up_axis not in AXIS_VECTORS:
        raise ValueError(f"unknown world_up_axis: {world_up_axis}")
    if not 0 <= maximum_up_tilt_degrees < 90:
        raise ValueError("maximum_up_tilt_degrees must be in [0, 90)")
    if center_error_score_weight < 0 or not math.isfinite(center_error_score_weight):
        raise ValueError("center_error_score_weight must be finite and non-negative")
    if not 0 <= maximum_vertical_compression < 1:
        raise ValueError("maximum_vertical_compression must be in [0, 1)")
    if maximum_support_penetration < 0 or not math.isfinite(maximum_support_penetration):
        raise ValueError("maximum_support_penetration must be finite and non-negative")
    if support_penetration_score_weight < 0 or not math.isfinite(
        support_penetration_score_weight
    ):
        raise ValueError("support_penetration_score_weight must be finite and non-negative")
    if support_contact_error_score_weight < 0 or not math.isfinite(
        support_contact_error_score_weight
    ):
        raise ValueError("support_contact_error_score_weight must be finite and non-negative")
    if maximum_support_contact_error_px < 0 or not math.isfinite(
        maximum_support_contact_error_px
    ):
        raise ValueError("maximum_support_contact_error_px must be finite and non-negative")
    if initial_scene_fit_receipt is not None and support_anchor_ply is None:
        raise ValueError("initial scene-fit refinement requires support_anchor_ply")
    if initial_scene_fit_receipt is not None and source_up_axis is None:
        raise ValueError("initial scene-fit refinement requires semantic up axes")

    mesh_source = mesh_source.resolve()
    object_anchor_ply = object_anchor_ply.resolve()
    cameras_path = cameras_path.resolve()
    view_manifest_path = view_manifest_path.resolve()
    output_dir = output_dir.resolve()
    support_anchor_ply = support_anchor_ply.resolve() if support_anchor_ply else None
    initial_scene_fit_receipt = (
        initial_scene_fit_receipt.resolve() if initial_scene_fit_receipt else None
    )
    support_contact_manifest_path = (
        support_contact_manifest_path.resolve() if support_contact_manifest_path else None
    )

    object_points, object_anchor_sha = load_ply_points(
        object_anchor_ply, expected_sha256=expected_object_anchor_sha256
    )
    views = load_view_manifest(
        view_manifest_path,
        object_id=object_id,
        cameras_path=cameras_path,
    )
    support_contact_source = None
    if support_contact_manifest_path is not None:
        support_contact_source = load_support_contact_manifest(
            support_contact_manifest_path,
            object_id=object_id,
            views=views,
        )
    camera_positions = np.asarray([view["camera"]["position"] for view in views])
    source_geometries, source_graph = _load_flattened_geometries(mesh_source)
    source_mesh = _scene_mesh(
        _build_transformed_scene(source_geometries, np.eye(4), object_id=object_id)
    )
    source_frame = robust_oriented_frame(
        np.asarray(source_mesh.vertices, dtype=np.float64),
        robust_quantiles=robust_quantiles,
    )
    canonical_matrix = np.eye(4, dtype=np.float64)
    canonical_matrix[:3, :3] = source_frame["axes_columns"].T
    canonical_matrix[:3, 3] = -np.einsum(
        "ij,j->i",
        source_frame["axes_columns"].T,
        source_frame["center"],
        optimize=False,
    )
    initial_receipt = None
    initial_receipt_validation = None
    initial_world_matrix = None
    if initial_scene_fit_receipt is not None:
        initial_receipt, initial_receipt_validation = _load_validated_initial_scene_fit(
            initial_scene_fit_receipt,
            object_id=object_id,
            mesh_source=mesh_source,
            object_anchor_ply=object_anchor_ply,
            cameras_path=cameras_path,
            view_manifest_path=view_manifest_path,
            canonical_matrix=canonical_matrix,
            source_up_axis=source_up_axis,
            world_up_axis=world_up_axis,
        )
        initial_world_matrix = np.asarray(
            initial_receipt["canonical_to_world"]["matrix_row_major"], dtype=np.float64
        )
    semantic_up = None
    if source_up_axis is not None and world_up_axis is not None:
        canonical_source_up = np.einsum(
            "ij,j->i",
            canonical_matrix[:3, :3],
            AXIS_VECTORS[source_up_axis],
            optimize=False,
        )
        canonical_source_up /= np.linalg.norm(canonical_source_up)
        semantic_up = {
            "source_up_axis": source_up_axis,
            "source_up_vector": AXIS_VECTORS[source_up_axis],
            "canonical_source_up": canonical_source_up,
            "world_up_axis": world_up_axis,
            "world_up": AXIS_VECTORS[world_up_axis],
            "maximum_tilt_degrees": maximum_up_tilt_degrees,
        }
    canonical_scene = _build_transformed_scene(
        source_geometries, canonical_matrix, object_id=object_id
    )
    canonical_mesh = _scene_mesh(canonical_scene)
    search_mesh, search_mesh_report = make_search_mesh(
        canonical_mesh,
        maximum_faces=maximum_search_faces,
    )
    canonical_vertices = np.asarray(canonical_mesh.vertices, dtype=np.float64)
    canonical_frame = robust_oriented_frame(
        canonical_vertices,
        robust_quantiles=robust_quantiles,
    )
    source_extents = np.asarray(source_frame["extents"], dtype=np.float64)

    support_points = None
    support_sha = None
    support = None
    support_alignment_target = None
    if support_anchor_ply is not None:
        support_points, support_sha = load_ply_points(
            support_anchor_ply,
            expected_sha256=expected_support_anchor_sha256,
        )
        support_alignment_target = _support_alignment(
            object_points,
            support_points,
            camera_positions,
            robust_quantiles=robust_quantiles,
        )
        support = support_alignment_target["support"]
        if initial_world_matrix is None:
            target = support_alignment_target
        else:
            initial_world_vertices = (
                np.einsum(
                    "ni,ji->nj",
                    canonical_vertices,
                    initial_world_matrix[:3, :3],
                    optimize=False,
                )
                + initial_world_matrix[:3, 3]
            )
            target = robust_oriented_frame(
                initial_world_vertices, robust_quantiles=robust_quantiles
            )
            target["support"] = support
            target["mode"] = "validated_initial_scene_fit_refinement"
    else:
        target = robust_oriented_frame(object_points, robust_quantiles=robust_quantiles)

    front = None
    if placement_mode == "observed_front_surface":
        source_views = [view for view in views if view["is_source_view"]]
        source_view = source_views[0] if source_views else views[0]
        camera_position = np.asarray(source_view["camera"]["position"], dtype=np.float64)
        view_direction = np.asarray(target["center"], dtype=np.float64) - camera_position
        view_direction /= np.linalg.norm(view_direction)
        front_depths = np.einsum("ni,i->n", object_points, view_direction, optimize=False)
        front_offset = float(np.quantile(front_depths, front_quantile))
        front = {
            "frame_id": source_view["frame_id"],
            "camera_position": camera_position,
            "view_direction": view_direction,
            "plane_offset": front_offset,
            "quantile": front_quantile,
            "mesh_surface_quantile": robust_quantiles[0],
        }

    permutations = right_handed_axis_permutations()
    initial_refinement_geometry = None
    if initial_world_matrix is not None:
        if support is None:
            raise AssertionError("validated initial refinement requires support geometry")
        initial_recomputed_score, initial_recomputed_views = _evaluate_views(
            canonical_mesh,
            views,
            initial_world_matrix,
            center_error_score_weight=center_error_score_weight,
            support_contact_error_score_weight=support_contact_error_score_weight,
        )
        initial_recomputed_silhouettes_passed = all(
            record["mask_iou"] >= minimum_mask_iou
            and record["bbox_iou"] >= minimum_bbox_iou
            and record["center_error_px"] <= maximum_center_error_px
            for record in initial_recomputed_views
        )
        if not initial_recomputed_silhouettes_passed:
            raise ValueError("initial scene-fit no longer passes current silhouette thresholds")
        if semantic_up is not None:
            initial_recomputed_orientation = _semantic_up_alignment(
                initial_world_matrix[:3, :3],
                canonical_source_up=np.asarray(semantic_up["canonical_source_up"]),
                world_up=np.asarray(semantic_up["world_up"]),
            )
            if initial_recomputed_orientation["tilt_degrees"] > maximum_up_tilt_degrees:
                raise ValueError("initial scene-fit no longer passes semantic up threshold")
        else:
            initial_recomputed_orientation = None
        initial_receipt_validation["recomputed_view_silhouettes_passed"] = True
        initial_receipt_validation["recomputed_score"] = initial_recomputed_score
        initial_receipt_validation["recomputed_views"] = initial_recomputed_views
        initial_receipt_validation["recomputed_semantic_up"] = initial_recomputed_orientation
        best, initial_refinement_geometry = _optimize_initial_scene_fit(
            initial_matrix=initial_world_matrix,
            mesh=search_mesh,
            views=views,
            canonical_vertices=canonical_vertices,
            support=support,
            front=front,
            semantic_up=semantic_up,
            cycles=cycles,
            shrink=shrink,
            maximum_horizontal_scale_multiplier=maximum_scale_multiplier,
            maximum_vertical_compression=maximum_vertical_compression,
            maximum_rotation_radians=math.radians(maximum_rotation_degrees),
            maximum_translation_ratio=maximum_translation_ratio,
            maximum_support_penetration=maximum_support_penetration,
            support_penetration_score_weight=support_penetration_score_weight,
            center_error_score_weight=center_error_score_weight,
            support_contact_error_score_weight=support_contact_error_score_weight,
        )
        initial_candidates = [best]
    else:
        initial_candidates = [
            _optimize_candidate(
                permutation_index=index,
                permutation=permutation,
                mesh=search_mesh,
                views=views,
                source_extents=source_extents,
                target=target,
                canonical_vertices=canonical_vertices,
                support=support,
                front=front,
                semantic_up=semantic_up,
                cycles=0,
                shrink=shrink,
                maximum_scale_multiplier=maximum_scale_multiplier,
                maximum_scale_anisotropy=maximum_scale_anisotropy,
                maximum_rotation_radians=math.radians(maximum_rotation_degrees),
                maximum_translation_ratio=maximum_translation_ratio,
                center_error_score_weight=center_error_score_weight,
            )
            for index, permutation in enumerate(permutations)
        ]
        ranked = sorted(initial_candidates, key=lambda item: item["score"], reverse=True)
        refined = []
        for initial in ranked[:refine_axis_candidates]:
            index = int(initial["permutation_index"])
            refined.append(
                _optimize_candidate(
                    permutation_index=index,
                    permutation=permutations[index],
                    mesh=search_mesh,
                    views=views,
                    source_extents=source_extents,
                    target=target,
                    canonical_vertices=canonical_vertices,
                    support=support,
                    front=front,
                    semantic_up=semantic_up,
                    cycles=cycles,
                    shrink=shrink,
                    maximum_scale_multiplier=maximum_scale_multiplier,
                    maximum_scale_anisotropy=maximum_scale_anisotropy,
                    maximum_rotation_radians=math.radians(maximum_rotation_degrees),
                    maximum_translation_ratio=maximum_translation_ratio,
                    center_error_score_weight=center_error_score_weight,
                )
            )
        best = max(refined, key=lambda item: item["score"])
    world_matrix = np.asarray(best["matrix"], dtype=np.float64)
    full_mesh_score, full_mesh_views = _evaluate_views(
        canonical_mesh,
        views,
        world_matrix,
        center_error_score_weight=center_error_score_weight,
        support_contact_error_score_weight=support_contact_error_score_weight,
    )
    best["search_mesh_score"] = best["score"]
    best["score"] = full_mesh_score
    best["views"] = full_mesh_views
    scene_local_matrix = world_matrix.copy()
    scene_local_matrix[:3, 3] = 0.0
    canonical_geometry_items = [
        (name, geometry) for name, geometry in canonical_scene.geometry.items()
    ]
    scene_local_scene = _build_transformed_scene(
        canonical_geometry_items,
        scene_local_matrix,
        object_id=object_id,
    )
    world_scene = _build_transformed_scene(
        canonical_geometry_items,
        world_matrix,
        object_id=object_id,
    )

    canonical_glb = output_dir / f"{object_id}.canonical.glb"
    canonical_obj = output_dir / f"{object_id}.canonical.obj"
    scene_local_glb = output_dir / f"{object_id}.scene-local.glb"
    scene_local_obj = output_dir / f"{object_id}.scene-local.obj"
    world_glb = output_dir / f"{object_id}.scene-space.glb"
    world_obj = output_dir / f"{object_id}.scene-space.obj"
    _write_scene_pair(canonical_scene, glb_path=canonical_glb, obj_path=canonical_obj)
    _write_scene_pair(
        scene_local_scene,
        glb_path=scene_local_glb,
        obj_path=scene_local_obj,
    )
    _write_scene_pair(world_scene, glb_path=world_glb, obj_path=world_obj)

    exports = {
        "canonical_local": {
            "coordinate_space": "canonical_local",
            "glb_role": "pbr_visual_logic_authority",
            "obj_role": "geometry_exchange_only",
            "glb": _audit_geometry(canonical_glb),
            "obj": _audit_geometry(canonical_obj),
        },
        "scene_local": {
            "coordinate_space": "scene_local_about_target_pivot",
            "placement": {
                "pivot": world_matrix[:3, 3].tolist(),
                "runtime_scale": [1.0, 1.0, 1.0],
                "runtime_rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "glb_role": "pbr_visual_logic_authority",
            "obj_role": "geometry_exchange_only",
            "glb": _audit_geometry(scene_local_glb),
            "obj": _audit_geometry(scene_local_obj),
        },
        "scene_space": {
            "coordinate_space": "world_baked",
            "runtime_transform": np.eye(4).tolist(),
            "glb_role": "pbr_visual_logic_authority",
            "obj_role": "geometry_exchange_only",
            "glb": _audit_geometry(world_glb),
            "obj": _audit_geometry(world_obj),
        },
    }
    local_vertices_from_bytes = _geometry_vertices(canonical_glb)
    predicted_world = (
        np.einsum(
            "ni,ji->nj",
            local_vertices_from_bytes,
            world_matrix[:3, :3],
            optimize=False,
        )
        + world_matrix[:3, 3]
    )
    predicted_bounds = np.asarray([predicted_world.min(axis=0), predicted_world.max(axis=0)])
    world_bounds = np.asarray(exports["scene_space"]["glb"]["bounds"])
    scene_local_bounds = np.asarray(exports["scene_local"]["glb"]["bounds"])
    scene_local_predicted_world = scene_local_bounds + world_matrix[:3, 3]
    bounds_scale = max(1.0, float(np.max(np.abs(world_bounds))))
    bounds_tolerance = bounds_scale * 2e-5
    glb_obj_match = all(
        np.allclose(
            exports[space]["glb"]["bounds"],
            exports[space]["obj"]["bounds"],
            atol=bounds_tolerance,
            rtol=0,
        )
        for space in ("canonical_local", "scene_local", "scene_space")
    )
    transform_round_trip = bool(
        np.allclose(predicted_bounds, world_bounds, atol=bounds_tolerance, rtol=0)
    )
    scene_local_round_trip = bool(
        np.allclose(
            scene_local_predicted_world,
            world_bounds,
            atol=bounds_tolerance,
            rtol=0,
        )
    )
    support_tolerance = max(1e-5, float(np.min(target["extents"])) * 0.01)
    constraint_errors = best["fit"]["constraint_errors"]
    support_contact = (
        _support_signed_contact(canonical_vertices, world_matrix, support)
        if support is not None
        else None
    )
    view_gates = []
    for record in best["views"]:
        gate = {
            "frame_id": record["frame_id"],
            "mask_iou": record["mask_iou"] >= minimum_mask_iou,
            "bbox_iou": record["bbox_iou"] >= minimum_bbox_iou,
            "center_error_px": record["center_error_px"] <= maximum_center_error_px,
        }
        if "source_camera_support_contact" in record:
            gate["support_contact_bottom_error_px"] = (
                record["source_camera_support_contact"]["absolute_bottom_error_px"]
                <= maximum_support_contact_error_px
            )
        view_gates.append(gate)
    gates = {
        "canonical_glb_obj_bounds_match": glb_obj_match,
        "world_bake_matches_canonical_transform": transform_round_trip,
        "finite_exported_geometry": all(
            exports[space][kind]["finite_vertices"]
            for space in ("canonical_local", "scene_local", "scene_space")
            for kind in ("glb", "obj")
        ),
        "pbr_glb_materials": all(
            exports[space]["glb"]["material_types"]
            and all(
                material_type == "PBRMaterial"
                for material_type in exports[space]["glb"]["material_types"]
            )
            for space in ("canonical_local", "scene_local", "scene_space")
        ),
        "visible_front_surface": constraint_errors.get("visible_front_plane_error", 0.0)
        <= support_tolerance,
        "scale_anisotropy": best["fit"]["scale_anisotropy_ratio"]
        <= maximum_scale_anisotropy + 1e-9,
        "scene_local_pivot_round_trip": scene_local_round_trip,
        "semantic_up_alignment": (
            semantic_up is None
            or best["fit"]["semantic_up_alignment"]["tilt_degrees"]
            <= maximum_up_tilt_degrees + 1e-9
        ),
        "view_silhouettes": all(
            all(value for key, value in gate.items() if key != "frame_id") for gate in view_gates
        ),
    }
    if initial_world_matrix is not None:
        gates["initial_scene_fit_validated"] = bool(
            initial_receipt_validation
            and all(
                value is True
                for key, value in initial_receipt_validation.items()
                if key.endswith("_matched") or key.endswith("_passed")
            )
        )
    else:
        gates["axis_permutations_enumerated"] = len(initial_candidates) == 24
    if support_contact is not None:
        gates["support_contact"] = bool(
            support_contact["minimum_gap"] <= support_tolerance
            and support_contact["maximum_penetration"]
            <= maximum_support_penetration + 1e-8
            and constraint_errors.get("support_plane_error", float("inf"))
            <= support_tolerance
        )
    all_gates_passed = all(gates.values())
    multiview_evidence = len(views) >= 2
    pbr_support_contact_pending = support_contact_source is not None
    promotion_allowed = (
        all_gates_passed and multiview_evidence and not pbr_support_contact_pending
    )

    candidate_summaries = [
        {
            "permutation_index": item["permutation_index"],
            "permutation_matrix": item["permutation_matrix"],
            "initial_score": item["score"],
            "views": item["views"],
        }
        for item in initial_candidates
    ]
    report = {
        "schema_version": 1,
        "kind": "video2world.completed_object_scene_fit",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "technical_gates_passed" if all_gates_passed else "rejected",
        "promotion_allowed": promotion_allowed,
        "promotion_status": (
            "authorized_by_multiview_technical_gates"
            if promotion_allowed
            else (
                "held_pending_multiview_evidence"
                if all_gates_passed and not multiview_evidence
                else (
                    "held_pending_material_corrected_pbr_support_contact_review"
                    if all_gates_passed and pbr_support_contact_pending
                    else "held_by_failed_technical_gates"
                )
            )
        ),
        "evidence_scope": (
            "calibrated_multiview" if multiview_evidence else "single_view_provisional"
        ),
        "object_id": object_id,
        "representation_mode": "unified_pbr_mesh_visual_logic_collision",
        "placement_mode": placement_mode,
        "support_policy": {
            "provided": support_anchor_ply is not None,
            "maximum_signed_penetration": maximum_support_penetration,
            "contract": (
                "support anchor must describe the object's immediate physical support; "
                "support contact is not reported as true when that surface is unavailable"
            ),
        },
        "support_contact": _json_value(support_contact),
        "pbr_support_contact_review": {
            "status": (
                "pending_material_corrected_pbr_review"
                if pbr_support_contact_pending
                else "not_requested"
            ),
            "final_pbr_alpha_verified": False,
            "geometry_contact_gate_is_not_final_pbr_alpha_gate": True,
            "contract": (
                "Calibrated analytic geometry contact guides fitting, but final PBR alpha "
                "contact must be re-reviewed after material and double-sided corrections."
            ),
        },
        "semantic_orientation": (
            {
                "source_up_axis": source_up_axis,
                "source_up_vector": AXIS_VECTORS[source_up_axis].tolist(),
                "canonical_source_up": semantic_up["canonical_source_up"].tolist(),
                "world_up_axis": world_up_axis,
                "world_up_vector": AXIS_VECTORS[world_up_axis].tolist(),
                "maximum_tilt_degrees": maximum_up_tilt_degrees,
                "selected_alignment": best["fit"]["semantic_up_alignment"],
                "contract": (
                    "The declared native semantic up axis must remain aligned with scene "
                    "gravity; source-camera silhouette score cannot override this gate."
                ),
            }
            if semantic_up is not None
            else None
        ),
        "sources": {
            "mesh": {"path": str(mesh_source), "sha256": sha256_file(mesh_source)},
            "object_anchor": {
                "path": str(object_anchor_ply),
                "sha256": object_anchor_sha,
                "point_count": len(object_points),
            },
            "support_anchor": (
                {
                    "path": str(support_anchor_ply),
                    "sha256": support_sha,
                    "point_count": len(support_points),
                }
                if support_anchor_ply is not None and support_points is not None
                else None
            ),
            "cameras": {"path": str(cameras_path), "sha256": sha256_file(cameras_path)},
            "view_manifest": {
                "path": str(view_manifest_path),
                "sha256": sha256_file(view_manifest_path),
            },
            "initial_scene_fit_receipt": initial_receipt_validation,
            "support_contact_manifest": support_contact_source,
            "views": [
                {
                    "frame_id": view["frame_id"],
                    "observed_mask": {
                        "path": str(view["observed_path"]),
                        "sha256": view["observed_sha256"],
                    },
                    "occluder_masks": view["occluders"],
                    "is_source_view": view["is_source_view"],
                }
                for view in views
            ],
        },
        "canonicalization": {
            "source_graph": source_graph,
            "source_robust_frame": _json_value(source_frame),
            "source_to_canonical_matrix_row_major": canonical_matrix.tolist(),
            "canonical_robust_frame": _json_value(canonical_frame),
        },
        "target": _json_value(target),
        "visible_front_surface": _json_value(front) if front is not None else None,
        "search": {
            "method": (
                "validated_initial_fit_bottom_pivot_support_refinement"
                if initial_world_matrix is not None
                else "24_axis_enumeration_then_bounded_coordinate_descent"
            ),
            "robust_quantiles": list(robust_quantiles),
            "cycles": cycles,
            "shrink": shrink,
            "refined_axis_candidates": refine_axis_candidates,
            "maximum_scale_multiplier": maximum_scale_multiplier,
            "maximum_scale_anisotropy": maximum_scale_anisotropy,
            "maximum_rotation_degrees": maximum_rotation_degrees,
            "maximum_translation_ratio": maximum_translation_ratio,
            "maximum_vertical_compression": maximum_vertical_compression,
            "maximum_support_penetration": maximum_support_penetration,
            "support_penetration_score_weight": support_penetration_score_weight,
            "support_contact_error_score_weight": support_contact_error_score_weight,
            "maximum_support_contact_error_px": maximum_support_contact_error_px,
            "center_error_score_weight": center_error_score_weight,
            "initial_refinement_geometry": initial_refinement_geometry,
            "projection_geometry": {
                **search_mesh_report,
                "final_metrics_recomputed_with_full_mesh": True,
            },
            "axis_candidates": candidate_summaries,
            "selected": {
                key: (value.tolist() if isinstance(value, np.ndarray) else value)
                for key, value in best.items()
                if key != "matrix"
            },
        },
        "canonical_to_world": {
            "matrix_row_major": world_matrix.tolist(),
            "asset_coordinates_baked": False,
        },
        "scene_local_runtime": {
            "asset_coordinates_baked": "rotation_and_scale_only",
            "placement": {
                "pivot": world_matrix[:3, 3].tolist(),
                "scale": [1.0, 1.0, 1.0],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "canonical_to_scene_local_matrix_row_major": scene_local_matrix.tolist(),
        },
        "world_baked_transform": {
            "matrix_row_major": np.eye(4).tolist(),
            "asset_coordinates_baked": True,
            "source_canonical_to_world_matrix_row_major": world_matrix.tolist(),
        },
        "exports": exports,
        "final_export_qa": {
            "bounds_tolerance": bounds_tolerance,
            "predicted_world_bounds_from_canonical_bytes": predicted_bounds.tolist(),
            "observed_world_bounds_from_export_bytes": world_bounds.tolist(),
            "predicted_world_bounds_from_scene_local_bytes_and_pivot": (
                scene_local_predicted_world.tolist()
            ),
            "glb_obj_bounds_match": glb_obj_match,
            "canonical_to_world_round_trip": transform_round_trip,
            "scene_local_pivot_round_trip": scene_local_round_trip,
        },
        "thresholds": {
            "minimum_mask_iou": minimum_mask_iou,
            "minimum_bbox_iou": minimum_bbox_iou,
            "maximum_center_error_px": maximum_center_error_px,
            "support_and_front_plane_tolerance": support_tolerance,
            "maximum_up_tilt_degrees": maximum_up_tilt_degrees,
            "maximum_support_penetration": maximum_support_penetration,
            "maximum_support_contact_error_px": maximum_support_contact_error_px,
        },
        "views": best["views"],
        "view_acceptance_gates": view_gates,
        "acceptance_gates": gates,
        "all_acceptance_gates_passed": all_gates_passed,
    }
    write_json(output_dir / "scene_fit_receipt.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--mesh-source", type=Path, required=True)
    parser.add_argument("--object-anchor-ply", type=Path, required=True)
    parser.add_argument("--support-anchor-ply", type=Path)
    parser.add_argument("--initial-scene-fit-receipt", type=Path)
    parser.add_argument("--support-contact-manifest", type=Path)
    parser.add_argument("--expected-object-anchor-sha256")
    parser.add_argument("--expected-support-anchor-sha256")
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--view-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--placement-mode",
        choices=("volume_center", "observed_front_surface"),
        default="volume_center",
    )
    parser.add_argument("--robust-lower-quantile", type=float, default=0.005)
    parser.add_argument("--robust-upper-quantile", type=float, default=0.995)
    parser.add_argument("--front-quantile", type=float, default=0.05)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--shrink", type=float, default=0.5)
    parser.add_argument("--refine-axis-candidates", type=int, default=3)
    parser.add_argument("--maximum-scale-multiplier", type=float, default=1.35)
    parser.add_argument("--maximum-scale-anisotropy", type=float, default=1.15)
    parser.add_argument("--maximum-rotation-degrees", type=float, default=15.0)
    parser.add_argument("--maximum-translation-ratio", type=float, default=0.2)
    parser.add_argument("--maximum-search-faces", type=int, default=5_000)
    parser.add_argument("--minimum-mask-iou", type=float, default=0.5)
    parser.add_argument("--minimum-bbox-iou", type=float, default=0.6)
    parser.add_argument("--maximum-center-error-px", type=float, default=30.0)
    parser.add_argument("--source-up-axis", choices=tuple(AXIS_VECTORS))
    parser.add_argument("--world-up-axis", choices=tuple(AXIS_VECTORS))
    parser.add_argument("--maximum-up-tilt-degrees", type=float, default=20.0)
    parser.add_argument("--center-error-score-weight", type=float, default=0.01)
    parser.add_argument("--maximum-vertical-compression", type=float, default=0.25)
    parser.add_argument("--maximum-support-penetration", type=float, default=0.0)
    parser.add_argument("--support-penetration-score-weight", type=float, default=0.01)
    parser.add_argument("--support-contact-error-score-weight", type=float, default=0.0)
    parser.add_argument("--maximum-support-contact-error-px", type=float, default=15.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = fit_completed_object_to_scene(
        object_id=args.object_id,
        mesh_source=args.mesh_source.expanduser(),
        object_anchor_ply=args.object_anchor_ply.expanduser(),
        support_anchor_ply=(
            args.support_anchor_ply.expanduser() if args.support_anchor_ply else None
        ),
        initial_scene_fit_receipt=(
            args.initial_scene_fit_receipt.expanduser()
            if args.initial_scene_fit_receipt
            else None
        ),
        support_contact_manifest_path=(
            args.support_contact_manifest.expanduser() if args.support_contact_manifest else None
        ),
        expected_object_anchor_sha256=args.expected_object_anchor_sha256,
        expected_support_anchor_sha256=args.expected_support_anchor_sha256,
        cameras_path=args.cameras.expanduser(),
        view_manifest_path=args.view_manifest.expanduser(),
        output_dir=args.output_dir.expanduser(),
        placement_mode=args.placement_mode,
        robust_quantiles=(args.robust_lower_quantile, args.robust_upper_quantile),
        front_quantile=args.front_quantile,
        cycles=args.cycles,
        shrink=args.shrink,
        refine_axis_candidates=args.refine_axis_candidates,
        maximum_scale_multiplier=args.maximum_scale_multiplier,
        maximum_scale_anisotropy=args.maximum_scale_anisotropy,
        maximum_rotation_degrees=args.maximum_rotation_degrees,
        maximum_translation_ratio=args.maximum_translation_ratio,
        maximum_search_faces=args.maximum_search_faces,
        minimum_mask_iou=args.minimum_mask_iou,
        minimum_bbox_iou=args.minimum_bbox_iou,
        maximum_center_error_px=args.maximum_center_error_px,
        source_up_axis=args.source_up_axis,
        world_up_axis=args.world_up_axis,
        maximum_up_tilt_degrees=args.maximum_up_tilt_degrees,
        center_error_score_weight=args.center_error_score_weight,
        maximum_vertical_compression=args.maximum_vertical_compression,
        maximum_support_penetration=args.maximum_support_penetration,
        support_penetration_score_weight=args.support_penetration_score_weight,
        support_contact_error_score_weight=args.support_contact_error_score_weight,
        maximum_support_contact_error_px=args.maximum_support_contact_error_px,
    )
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if report["all_acceptance_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
