#!/usr/bin/env python3
"""Extract a measured planar support subset from a larger object anchor PLY."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData, PlyElement
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, cKDTree

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


def _quantiles(values: np.ndarray) -> dict[str, float]:
    names = ("minimum", "q005", "q01", "q05", "median", "q95", "q99", "q995", "maximum")
    probabilities = (0.0, 0.005, 0.01, 0.05, 0.5, 0.95, 0.99, 0.995, 1.0)
    measured = np.quantile(values, probabilities)
    return {name: float(value) for name, value in zip(names, measured, strict=True)}


def load_vertex_data(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    observed_sha256 = sha256_file(path)
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise ValueError(f"PLY hash mismatch for {path}")
    ply = PlyData.read(path)
    if "vertex" not in ply:
        raise ValueError(f"PLY has no vertex element: {path}")
    vertex = np.array(ply["vertex"].data, copy=True)
    names = set(vertex.dtype.names or ())
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError(f"PLY is missing coordinates: {path}")
    points = np.column_stack([vertex[name] for name in ("x", "y", "z")]).astype(
        np.float64
    )
    if len(points) < 64 or not np.isfinite(points).all():
        raise ValueError(f"PLY requires at least 64 finite points: {path}")
    return vertex, points, observed_sha256


def _point_plane_values(points: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return np.einsum("ni,i->n", points, normal, optimize=False)


def _orient_normal(normal: np.ndarray, world_up: np.ndarray) -> np.ndarray:
    result = np.asarray(normal, dtype=np.float64)
    length = float(np.linalg.norm(result))
    if length <= 1e-12 or not np.isfinite(length):
        raise ValueError("plane normal is invalid")
    result = result / length
    if float(np.dot(result, world_up)) < 0:
        result *= -1
    return result


def _fit_pca_plane(points: np.ndarray, world_up: np.ndarray) -> tuple[np.ndarray, float]:
    if len(points) < 3:
        raise ValueError("at least three points are required to fit a plane")
    center = np.median(points, axis=0)
    covariance = np.cov((points - center).T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normal = _orient_normal(eigenvectors[:, int(np.argmin(eigenvalues))], world_up)
    offset = float(np.median(_point_plane_values(points, normal)))
    return normal, offset


def _plane_tilt_degrees(normal: np.ndarray, world_up: np.ndarray) -> float:
    dot = float(np.clip(np.dot(normal, world_up), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def fit_support_plane(
    support_points: np.ndarray,
    *,
    world_up: np.ndarray,
    candidate_top_fraction: float,
    ransac_trials: int,
    random_seed: int,
    inlier_distance: float,
    maximum_normal_tilt_degrees: float,
    refinement_iterations: int,
) -> dict[str, Any]:
    if not 0 < candidate_top_fraction < 1:
        raise ValueError("candidate_top_fraction must be in (0, 1)")
    if ransac_trials < 1 or refinement_iterations < 1:
        raise ValueError("RANSAC and refinement iteration counts must be positive")
    if inlier_distance <= 0:
        raise ValueError("inlier_distance must be positive")
    if not 0 <= maximum_normal_tilt_degrees < 90:
        raise ValueError("maximum_normal_tilt_degrees must be in [0, 90)")

    heights = _point_plane_values(support_points, world_up)
    height_threshold = float(np.quantile(heights, 1.0 - candidate_top_fraction))
    candidate_indices = np.flatnonzero(heights >= height_threshold)
    candidates = support_points[candidate_indices]
    if len(candidates) < 64:
        raise ValueError("too few points remain in the top support candidate band")

    generator = np.random.default_rng(random_seed)
    triplets = generator.integers(0, len(candidates), size=(ransac_trials, 3))
    first = candidates[triplets[:, 1]] - candidates[triplets[:, 0]]
    second = candidates[triplets[:, 2]] - candidates[triplets[:, 0]]
    normals = np.cross(first, second)
    lengths = np.linalg.norm(normals, axis=1)
    valid = np.isfinite(lengths) & (lengths > 1e-9)
    normals = normals[valid] / lengths[valid, None]
    origins = candidates[triplets[valid, 0]]
    orientation = _point_plane_values(normals, world_up)
    normals *= np.where(orientation >= 0, 1.0, -1.0)[:, None]

    best: tuple[tuple[int, float, float], np.ndarray, float] | None = None
    valid_plane_count = 0
    for normal, origin in zip(normals, origins, strict=True):
        tilt_degrees = _plane_tilt_degrees(normal, world_up)
        if tilt_degrees > maximum_normal_tilt_degrees:
            continue
        valid_plane_count += 1
        offset = float(np.dot(origin, normal))
        distances = np.abs(_point_plane_values(candidates, normal) - offset)
        inliers = distances <= inlier_distance
        count = int(np.count_nonzero(inliers))
        if count < 3:
            continue
        median_residual = float(np.median(distances[inliers]))
        score = (count, -median_residual, -tilt_degrees)
        if best is None or score > best[0]:
            best = (score, normal.copy(), offset)
    if best is None:
        raise ValueError("no support plane passed the world-up tilt constraint")

    _, normal, offset = best
    refinement_trace = []
    for iteration in range(refinement_iterations):
        distances = np.abs(_point_plane_values(candidates, normal) - offset)
        inliers = distances <= inlier_distance
        if int(np.count_nonzero(inliers)) < 64:
            raise ValueError("plane refinement retained fewer than 64 points")
        normal, offset = _fit_pca_plane(candidates[inliers], world_up)
        refinement_trace.append(
            {
                "iteration": iteration,
                "inlier_count": int(np.count_nonzero(inliers)),
                "normal": normal.tolist(),
                "offset": offset,
                "tilt_degrees": _plane_tilt_degrees(normal, world_up),
            }
        )

    distances = np.abs(_point_plane_values(candidates, normal) - offset)
    inlier_mask = distances <= inlier_distance
    inlier_indices = candidate_indices[inlier_mask]
    if len(inlier_indices) < 64:
        raise ValueError("final support plane retained fewer than 64 points")
    return {
        "normal": normal,
        "offset": offset,
        "candidate_indices": candidate_indices,
        "inlier_indices": inlier_indices,
        "height_threshold": height_threshold,
        "valid_ransac_plane_count": valid_plane_count,
        "refinement_trace": refinement_trace,
    }


def plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    references = np.eye(3, dtype=np.float64)
    alignment = np.abs(references @ normal)
    reference = references[int(np.argmin(alignment))]
    first = reference - normal * float(np.dot(reference, normal))
    first /= np.linalg.norm(first)
    second = np.cross(normal, first)
    second /= np.linalg.norm(second)
    return first, second


def retain_largest_planar_component(
    points: np.ndarray,
    indices: np.ndarray,
    *,
    normal: np.ndarray,
    connectivity_radius: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if connectivity_radius <= 0:
        raise ValueError("connectivity_radius must be positive")
    first, second = plane_basis(normal)
    coordinates = np.column_stack(
        [_point_plane_values(points, first), _point_plane_values(points, second)]
    )
    tree = cKDTree(coordinates)
    pairs = tree.query_pairs(connectivity_radius, output_type="ndarray")
    self_indices = np.arange(len(points), dtype=np.int64)
    if len(pairs):
        rows = np.concatenate([pairs[:, 0], pairs[:, 1], self_indices])
        columns = np.concatenate([pairs[:, 1], pairs[:, 0], self_indices])
    else:
        rows = self_indices
        columns = self_indices
    graph = coo_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, columns)),
        shape=(len(points), len(points)),
    )
    component_count, labels = connected_components(graph, directed=False)
    counts = np.bincount(labels, minlength=component_count)
    selected_label = int(np.argmax(counts))
    selected = labels == selected_label
    nearest, _ = tree.query(coordinates, k=min(2, len(points)))
    nearest_neighbor = nearest[:, 1] if nearest.ndim == 2 and nearest.shape[1] > 1 else nearest
    return indices[selected], {
        "connectivity_radius": connectivity_radius,
        "component_count": int(component_count),
        "largest_component_count": int(np.count_nonzero(selected)),
        "largest_component_fraction": float(np.mean(selected)),
        "second_largest_component_count": (
            int(np.sort(counts)[-2]) if component_count > 1 else 0
        ),
        "nearest_neighbor_distance_quantiles": _quantiles(nearest_neighbor),
    }


def _load_camera(path: Path, frame_id: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("camera file must contain an array")
    matches = [item for item in payload if str(item.get("img_name")) == frame_id]
    if len(matches) != 1:
        raise ValueError(f"expected one camera for {frame_id!r}, found {len(matches)}")
    camera = matches[0]
    position = np.asarray(camera["position"], dtype=np.float64)
    rotation = np.asarray(camera["rotation"], dtype=np.float64)
    if position.shape != (3,) or rotation.shape != (3, 3):
        raise ValueError(f"camera {frame_id} has invalid dimensions")
    return {
        "position": position,
        "rotation": rotation,
        "width": int(camera["width"]),
        "height": int(camera["height"]),
        "fx": float(camera["fx"]),
        "fy": float(camera["fy"]),
    }


def _project_points(points: np.ndarray, camera: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    relative = points - camera["position"]
    camera_points = np.einsum(
        "ni,ij->nj", relative, camera["rotation"], optimize=False
    )
    depth = camera_points[:, 2]
    visible = depth > 1e-6
    projected = np.full((len(points), 2), np.nan, dtype=np.float64)
    projected[visible, 0] = (
        camera["fx"] * camera_points[visible, 0] / depth[visible] + camera["width"] / 2
    )
    projected[visible, 1] = (
        camera["height"] / 2
        + camera["fy"] * camera_points[visible, 1] / depth[visible]
    )
    return projected, visible


def _draw_projected_points(
    draw: ImageDraw.ImageDraw,
    projected: np.ndarray,
    visible: np.ndarray,
    *,
    width: int,
    height: int,
    color: tuple[int, int, int, int],
    radius: int,
    maximum_points: int,
) -> int:
    indices = np.flatnonzero(visible)
    stride = max(1, math.ceil(len(indices) / maximum_points))
    drawn = 0
    for x, y in projected[indices[::stride]]:
        if 0 <= x < width and 0 <= y < height:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
            drawn += 1
    return drawn


def render_review_contact_sheet(
    *,
    camera_path: Path,
    review_frames: list[tuple[str, Path]],
    support_points: np.ndarray,
    selected_points: np.ndarray,
    object_points: np.ndarray,
    output_path: Path,
) -> dict[str, Any]:
    tiles = []
    frame_reports = []
    for frame_id, source_path in review_frames:
        camera = _load_camera(camera_path, frame_id)
        source = Image.open(source_path).convert("RGB")
        expected_size = (camera["width"], camera["height"])
        if source.size != expected_size:
            raise ValueError(f"source frame dimensions do not match camera for {frame_id}")
        draw = ImageDraw.Draw(source, "RGBA")
        support_projected, support_visible = _project_points(support_points, camera)
        selected_projected, selected_visible = _project_points(selected_points, camera)
        object_projected, object_visible = _project_points(object_points, camera)
        support_drawn = _draw_projected_points(
            draw,
            support_projected,
            support_visible,
            width=camera["width"],
            height=camera["height"],
            color=(255, 255, 255, 55),
            radius=1,
            maximum_points=10_000,
        )
        selected_drawn = _draw_projected_points(
            draw,
            selected_projected,
            selected_visible,
            width=camera["width"],
            height=camera["height"],
            color=(255, 0, 210, 210),
            radius=2,
            maximum_points=8_000,
        )
        object_drawn = _draw_projected_points(
            draw,
            object_projected,
            object_visible,
            width=camera["width"],
            height=camera["height"],
            color=(0, 145, 255, 190),
            radius=1,
            maximum_points=5_000,
        )
        draw.rectangle((0, 0, 615, 38), fill=(0, 0, 0, 190))
        draw.text(
            (10, 10),
            f"{frame_id}: magenta=support surface, blue=plant anchor",
            fill=(255, 255, 255, 255),
        )
        source.thumbnail((640, 360), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (640, 360), (24, 24, 24))
        tile.paste(source, ((640 - source.width) // 2, (360 - source.height) // 2))
        tiles.append(tile)
        frame_reports.append(
            {
                "frame_id": frame_id,
                "source": {"path": str(source_path), "sha256": sha256_file(source_path)},
                "projected_points_drawn": {
                    "support_anchor": support_drawn,
                    "selected_surface": selected_drawn,
                    "object_anchor": object_drawn,
                },
            }
        )

    columns = 2
    rows = math.ceil(len(tiles) / columns)
    contact = Image.new("RGB", (columns * 640, rows * 360), (24, 24, 24))
    for index, tile in enumerate(tiles):
        contact.paste(tile, ((index % columns) * 640, (index // columns) * 360))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    contact.save(output_path)
    return {
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "width": contact.width,
        "height": contact.height,
        "frames": frame_reports,
    }


def extract_planar_support_surface(
    *,
    support_anchor_ply: Path,
    object_anchor_ply: Path,
    output_ply: Path,
    report_path: Path,
    world_up_axis: str = "-Y",
    candidate_top_fraction: float = 0.3,
    ransac_trials: int = 4_000,
    random_seed: int = 20_260_720,
    inlier_distance: float = 0.02,
    maximum_normal_tilt_degrees: float = 10.0,
    refinement_iterations: int = 12,
    connectivity_radius: float = 0.075,
    expected_support_anchor_sha256: str | None = None,
    expected_object_anchor_sha256: str | None = None,
    camera_path: Path | None = None,
    review_frames: list[tuple[str, Path]] | None = None,
    review_contact_sheet_path: Path | None = None,
) -> dict[str, Any]:
    if world_up_axis not in AXIS_VECTORS:
        raise ValueError(f"unknown world_up_axis: {world_up_axis}")
    support_anchor_ply = support_anchor_ply.resolve()
    object_anchor_ply = object_anchor_ply.resolve()
    output_ply = output_ply.resolve()
    report_path = report_path.resolve()
    camera_path = camera_path.resolve() if camera_path is not None else None
    review_frames = [
        (frame_id, path.resolve()) for frame_id, path in (review_frames or [])
    ]
    review_contact_sheet_path = (
        review_contact_sheet_path.resolve() if review_contact_sheet_path is not None else None
    )
    input_paths = {support_anchor_ply, object_anchor_ply}
    if camera_path is not None:
        input_paths.add(camera_path)
    input_paths.update(path for _frame_id, path in review_frames)
    output_paths = [output_ply, report_path]
    if review_contact_sheet_path is not None:
        output_paths.append(review_contact_sheet_path)
    if len(set(output_paths)) != len(output_paths):
        raise ValueError("output PLY, report, and review contact sheet paths must be distinct")
    overwritten_inputs = sorted(str(path) for path in set(output_paths).intersection(input_paths))
    if overwritten_inputs:
        raise ValueError(f"output paths must not overwrite inputs: {overwritten_inputs}")
    if bool(review_frames) != bool(camera_path):
        raise ValueError("camera_path and review_frames must be provided together")
    if review_frames and review_contact_sheet_path is None:
        raise ValueError("review_contact_sheet_path is required with review_frames")

    support_vertex, support_points, support_sha256 = load_vertex_data(
        support_anchor_ply, expected_sha256=expected_support_anchor_sha256
    )
    _, object_points, object_sha256 = load_vertex_data(
        object_anchor_ply, expected_sha256=expected_object_anchor_sha256
    )
    world_up = AXIS_VECTORS[world_up_axis]
    plane = fit_support_plane(
        support_points,
        world_up=world_up,
        candidate_top_fraction=candidate_top_fraction,
        ransac_trials=ransac_trials,
        random_seed=random_seed,
        inlier_distance=inlier_distance,
        maximum_normal_tilt_degrees=maximum_normal_tilt_degrees,
        refinement_iterations=refinement_iterations,
    )
    inlier_indices = plane["inlier_indices"]
    component_indices, connectivity = retain_largest_planar_component(
        support_points[inlier_indices],
        inlier_indices,
        normal=plane["normal"],
        connectivity_radius=connectivity_radius,
    )
    normal, offset = _fit_pca_plane(support_points[component_indices], world_up)
    for _ in range(3):
        residuals = np.abs(_point_plane_values(support_points[component_indices], normal) - offset)
        component_indices = component_indices[residuals <= inlier_distance]
        if len(component_indices) < 64:
            raise ValueError("final connected support surface has fewer than 64 points")
        normal, offset = _fit_pca_plane(support_points[component_indices], world_up)
    selected_points = support_points[component_indices]
    residuals = np.abs(_point_plane_values(selected_points, normal) - offset)
    if float(np.max(residuals)) > inlier_distance + 1e-9:
        raise ValueError("final support points exceed the plane inlier distance")

    output_ply.parent.mkdir(parents=True, exist_ok=True)
    output_vertex = np.array(support_vertex[component_indices], copy=True)
    PlyData([PlyElement.describe(output_vertex, "vertex")], text=True).write(output_ply)
    output_sha256 = sha256_file(output_ply)

    first, second = plane_basis(normal)
    support_2d = np.column_stack(
        [_point_plane_values(selected_points, first), _point_plane_values(selected_points, second)]
    )
    hull = ConvexHull(support_2d)
    object_center = np.median(object_points, axis=0)
    center_distance = float(np.dot(object_center, normal) - offset)
    projected_center = object_center - normal * center_distance
    projected_center_2d = np.asarray(
        [np.dot(projected_center, first), np.dot(projected_center, second)]
    )
    hull_values = (
        hull.equations[:, :2] @ projected_center_2d + hull.equations[:, 2]
    )
    hull_normal_lengths = np.linalg.norm(hull.equations[:, :2], axis=1)
    center_inside_hull = bool(np.all(hull_values <= 1e-9))
    center_hull_margin = float(np.min(-hull_values / hull_normal_lengths))
    support_tree = cKDTree(support_2d)
    center_nearest_distance = float(support_tree.query(projected_center_2d, k=1)[0])
    local_radii = [
        connectivity_radius,
        2.0 * connectivity_radius,
        4.0 * connectivity_radius,
        8.0 * connectivity_radius,
    ]
    local_counts = {
        f"{radius:.9g}": len(support_tree.query_ball_point(projected_center_2d, radius))
        for radius in local_radii
    }
    object_signed_distances = _point_plane_values(object_points, normal) - offset
    tilt_degrees = _plane_tilt_degrees(normal, world_up)

    visual_review = None
    if review_frames:
        assert camera_path is not None
        assert review_contact_sheet_path is not None
        visual_review = render_review_contact_sheet(
            camera_path=camera_path,
            review_frames=review_frames,
            support_points=support_points,
            selected_points=selected_points,
            object_points=object_points,
            output_path=review_contact_sheet_path,
        )
        visual_review["camera_file"] = {
            "path": str(camera_path),
            "sha256": sha256_file(camera_path),
        }

    gates = {
        "output_has_at_least_64_points": len(selected_points) >= 64,
        "plane_tilt_within_limit": tilt_degrees <= maximum_normal_tilt_degrees + 1e-12,
        "p95_residual_within_inlier_distance": (
            float(np.quantile(residuals, 0.95)) <= inlier_distance + 1e-12
        ),
        "largest_component_fraction_at_least_0_95": (
            connectivity["largest_component_fraction"] >= 0.95
        ),
        "object_center_projection_inside_support_hull": center_inside_hull,
        "object_anchor_robustly_above_support": (
            float(np.quantile(object_signed_distances, 0.005)) > 0
        ),
        "observed_support_exists_within_four_connectivity_radii": (
            center_nearest_distance <= 4.0 * connectivity_radius
        ),
    }
    report = {
        "schema_version": 1,
        "kind": "video2world.planar_support_surface_receipt",
        "status": "passed" if all(gates.values()) else "rejected",
        "claim_scope": (
            "Measured points from the support object's top planar surface. The plane may "
            "interpolate through an occlusion gap, but does not reconstruct the supported object."
        ),
        "inputs": {
            "support_anchor": {
                "path": str(support_anchor_ply),
                "sha256": support_sha256,
                "point_count": len(support_points),
            },
            "object_anchor": {
                "path": str(object_anchor_ply),
                "sha256": object_sha256,
                "point_count": len(object_points),
            },
        },
        "parameters": {
            "world_up_axis": world_up_axis,
            "world_up_vector": world_up.tolist(),
            "candidate_top_fraction": candidate_top_fraction,
            "ransac_trials": ransac_trials,
            "random_seed": random_seed,
            "inlier_distance": inlier_distance,
            "maximum_normal_tilt_degrees": maximum_normal_tilt_degrees,
            "refinement_iterations": refinement_iterations,
            "connectivity_radius": connectivity_radius,
        },
        "selection": {
            "candidate_height_threshold": plane["height_threshold"],
            "candidate_point_count": len(plane["candidate_indices"]),
            "candidate_fraction_of_support_anchor": (
                len(plane["candidate_indices"]) / len(support_points)
            ),
            "valid_ransac_plane_count": plane["valid_ransac_plane_count"],
            "pre_connectivity_inlier_count": len(plane["inlier_indices"]),
            "connectivity": connectivity,
            "refinement_trace": plane["refinement_trace"],
        },
        "surface": {
            "up_normal": normal.tolist(),
            "down_normal_for_fit_consumer": (-normal).tolist(),
            "offset_for_dot_x_up_normal": offset,
            "plane_equation_up_normal": np.append(normal, -offset).tolist(),
            "tilt_from_world_up_degrees": tilt_degrees,
            "point_count": len(selected_points),
            "residual_abs_distance_quantiles": _quantiles(residuals),
            "bounds_min_xyz": selected_points.min(axis=0).tolist(),
            "bounds_max_xyz": selected_points.max(axis=0).tolist(),
            "robust_bounds_q005_xyz": np.quantile(selected_points, 0.005, axis=0).tolist(),
            "robust_bounds_q995_xyz": np.quantile(selected_points, 0.995, axis=0).tolist(),
            "planar_hull_area_scene_units_squared": float(hull.volume),
        },
        "object_relation": {
            "object_anchor_median_xyz": object_center.tolist(),
            "object_center_projected_to_surface_xyz": projected_center.tolist(),
            "object_center_signed_height_above_surface": center_distance,
            "object_anchor_signed_height_quantiles": _quantiles(object_signed_distances),
            "center_projection_inside_support_hull": center_inside_hull,
            "center_projection_hull_margin_scene_units": center_hull_margin,
            "nearest_observed_support_point_in_plane_scene_units": center_nearest_distance,
            "local_observed_support_counts_by_radius": local_counts,
            "occlusion_interpolation_required_at_object_center": (
                center_nearest_distance > connectivity_radius
            ),
        },
        "output": {
            "support_surface_ply": {
                "path": str(output_ply),
                "sha256": output_sha256,
                "point_count": len(selected_points),
                "format": "ascii PLY retaining source vertex properties",
            }
        },
        "visual_review": visual_review,
        "acceptance_gates": gates,
        "all_acceptance_gates_passed": all(gates.values()),
        "fit_completed_object_to_scene_recommendation": {
            "required_support_arguments": {
                "support_anchor_ply": str(output_ply),
                "expected_support_anchor_sha256": output_sha256,
            },
            "placement_mode": "volume_center",
            "source_up_axis": "+Y",
            "world_up_axis": world_up_axis,
            "maximum_up_tilt_degrees": 5.0,
            "maximum_rotation_degrees": 5.0,
            "maximum_scale_anisotropy": 1.15,
            "maximum_center_error_px": 10.0,
            "center_error_score_weight": 0.01,
            "warning": (
                "The object anchor is above the measured surface and does not directly prove pot "
                "contact. Use complete potted-plant masks plus the support constraint."
            ),
        },
    }
    write_json(report_path, report)
    return report


def _parse_review_frame(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("review frame must use FRAME_ID=PATH")
    frame_id, path_value = value.split("=", 1)
    if not frame_id or not path_value:
        raise argparse.ArgumentTypeError("review frame must use FRAME_ID=PATH")
    return frame_id, Path(path_value).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-anchor-ply", type=Path, required=True)
    parser.add_argument("--object-anchor-ply", type=Path, required=True)
    parser.add_argument("--output-ply", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-support-anchor-sha256")
    parser.add_argument("--expected-object-anchor-sha256")
    parser.add_argument("--world-up-axis", choices=tuple(AXIS_VECTORS), default="-Y")
    parser.add_argument("--candidate-top-fraction", type=float, default=0.3)
    parser.add_argument("--ransac-trials", type=int, default=4_000)
    parser.add_argument("--random-seed", type=int, default=20_260_720)
    parser.add_argument("--inlier-distance", type=float, default=0.02)
    parser.add_argument("--maximum-normal-tilt-degrees", type=float, default=10.0)
    parser.add_argument("--refinement-iterations", type=int, default=12)
    parser.add_argument("--connectivity-radius", type=float, default=0.075)
    parser.add_argument("--cameras", type=Path)
    parser.add_argument("--review-frame", action="append", type=_parse_review_frame, default=[])
    parser.add_argument("--review-contact-sheet", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = extract_planar_support_surface(
        support_anchor_ply=args.support_anchor_ply.expanduser(),
        object_anchor_ply=args.object_anchor_ply.expanduser(),
        output_ply=args.output_ply.expanduser(),
        report_path=args.report.expanduser(),
        world_up_axis=args.world_up_axis,
        candidate_top_fraction=args.candidate_top_fraction,
        ransac_trials=args.ransac_trials,
        random_seed=args.random_seed,
        inlier_distance=args.inlier_distance,
        maximum_normal_tilt_degrees=args.maximum_normal_tilt_degrees,
        refinement_iterations=args.refinement_iterations,
        connectivity_radius=args.connectivity_radius,
        expected_support_anchor_sha256=args.expected_support_anchor_sha256,
        expected_object_anchor_sha256=args.expected_object_anchor_sha256,
        camera_path=args.cameras.expanduser() if args.cameras else None,
        review_frames=args.review_frame,
        review_contact_sheet_path=(
            args.review_contact_sheet.expanduser() if args.review_contact_sheet else None
        ),
    )
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if report["all_acceptance_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
