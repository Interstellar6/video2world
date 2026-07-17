#!/usr/bin/env python3
"""Build a closed soft-object mesh and Gaussian surface from an audited RGBA view.

This is a deterministic fallback for semantically simple soft objects such as pillows.
It does not invent a second image. The observed front texture is mapped onto a closed
superellipsoid; hidden faces reuse the same material family, and every output is checked
for topology, thickness, color, and six-side appearance consistency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import PIL
import plyfile
import trimesh
from PIL import Image
from plyfile import PlyData, PlyElement

SH_C0 = 0.28209479177387814
GAUSSIAN_REST_FIELDS = 45
GAUSSIAN_PROPERTY_NAMES = (
    "x",
    "y",
    "z",
    "nx",
    "ny",
    "nz",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    *(f"f_rest_{index}" for index in range(GAUSSIAN_REST_FIELDS)),
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def signed_power(values: np.ndarray, exponent: float) -> np.ndarray:
    return np.sign(values) * np.power(np.abs(values), exponent)


def visible_bbox(alpha: np.ndarray) -> tuple[int, int, int, int]:
    rows, columns = np.nonzero(alpha >= 128)
    if not len(columns):
        raise ValueError("source RGBA has no visible pixels")
    return int(columns.min()), int(rows.min()), int(columns.max()), int(rows.max())


def extend_visible_texture(rgba: np.ndarray) -> np.ndarray:
    """Deterministically extend edge colors into transparent pixels without hallucination."""

    colors = rgba[..., :3].copy()
    known = rgba[..., 3] >= 128
    if not np.any(known):
        raise ValueError("source RGBA has no visible pixels")
    height, width = known.shape
    while not np.all(known):
        filled = False
        next_colors = colors.copy()
        next_known = known.copy()
        for dy, dx in ((-1, 0), (0, -1), (0, 1), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            source_y0 = max(0, -dy)
            source_y1 = min(height, height - dy)
            source_x0 = max(0, -dx)
            source_x1 = min(width, width - dx)
            target_y0 = source_y0 + dy
            target_y1 = source_y1 + dy
            target_x0 = source_x0 + dx
            target_x1 = source_x1 + dx
            source_known = known[source_y0:source_y1, source_x0:source_x1]
            target_unknown = ~next_known[target_y0:target_y1, target_x0:target_x1]
            accepted = source_known & target_unknown
            if not np.any(accepted):
                continue
            target = next_colors[target_y0:target_y1, target_x0:target_x1]
            source = colors[source_y0:source_y1, source_x0:source_x1]
            target[accepted] = source[accepted]
            next_known[target_y0:target_y1, target_x0:target_x1][accepted] = True
            filled = True
        if not filled:
            raise RuntimeError("could not extend the visible texture")
        colors, known = next_colors, next_known
    return colors


def sample_texture(colors: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    height, width = colors.shape[:2]
    x = np.clip(u, 0.0, 1.0) * (width - 1)
    y = np.clip(v, 0.0, 1.0) * (height - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = (x - x0)[:, None]
    wy = (y - y0)[:, None]
    top = colors[y0, x0] * (1.0 - wx) + colors[y0, x1] * wx
    bottom = colors[y1, x0] * (1.0 - wx) + colors[y1, x1] * wx
    return top * (1.0 - wy) + bottom * wy


def normalize_appearance(colors: np.ndarray, mode: str) -> np.ndarray:
    values = colors.astype(np.float64)
    if mode == "source":
        return np.clip(np.rint(values), 0, 255).astype(np.uint8)
    luminance = np.sum(values * np.asarray([0.2126, 0.7152, 0.0722]), axis=-1)
    median = float(np.median(luminance))
    if mode == "white_fabric":
        detail = np.clip((luminance - median) * 0.82, -54.0, 34.0)
        chroma = (values - luminance[:, None]) * 0.12
        target = np.asarray([205.0, 207.0, 208.0])
        normalized = target + detail[:, None] + chroma
        return np.clip(np.rint(normalized), 148, 240).astype(np.uint8)
    if mode == "source_light":
        target_luminance = max(172.0, median)
        detail = np.clip((luminance - median) * 0.78, -58.0, 42.0)
        chroma = (values - luminance[:, None]) * 0.62
        normalized = target_luminance + detail[:, None] + chroma
        return np.clip(np.rint(normalized), 92, 242).astype(np.uint8)
    raise ValueError(f"unsupported appearance mode: {mode}")


def build_superellipsoid(
    *,
    width: float,
    height: float,
    depth: float,
    latitude_segments: int,
    longitude_segments: int,
    shape_exponent: float,
) -> trimesh.Trimesh:
    if latitude_segments < 8 or longitude_segments < 16:
        raise ValueError("soft-object mesh requires at least 8 latitude and 16 longitude segments")
    half_width = width / 2.0
    half_height = height / 2.0
    half_depth = depth / 2.0
    latitudes = np.linspace(-math.pi / 2, math.pi / 2, latitude_segments + 1)[1:-1]
    longitudes = np.linspace(-math.pi, math.pi, longitude_segments, endpoint=False)
    vertices: list[list[float]] = []
    for latitude in latitudes:
        latitude_cos = math.cos(float(latitude)) ** shape_exponent
        y = half_height * math.copysign(
            abs(math.sin(float(latitude))) ** shape_exponent,
            float(latitude),
        )
        x = half_width * latitude_cos * signed_power(np.cos(longitudes), shape_exponent)
        z = half_depth * latitude_cos * signed_power(np.sin(longitudes), shape_exponent)
        edge = (np.abs(x / half_width) ** 7 + abs(y / half_height) ** 7) / 2.0
        z *= 1.0 - 0.10 * np.clip(edge, 0.0, 1.0)
        vertices.extend(np.column_stack((x, np.full_like(x, y), z)).tolist())

    ring_count = len(latitudes)
    bottom_index = len(vertices)
    vertices.append([0.0, -half_height, 0.0])
    top_index = len(vertices)
    vertices.append([0.0, half_height, 0.0])
    faces: list[list[int]] = []
    for longitude_index in range(longitude_segments):
        next_longitude = (longitude_index + 1) % longitude_segments
        faces.append([bottom_index, next_longitude, longitude_index])
    for ring_index in range(ring_count - 1):
        lower = ring_index * longitude_segments
        upper = (ring_index + 1) * longitude_segments
        for longitude_index in range(longitude_segments):
            next_longitude = (longitude_index + 1) % longitude_segments
            a = lower + longitude_index
            b = lower + next_longitude
            c = upper + next_longitude
            d = upper + longitude_index
            faces.extend(([a, b, c], [a, c, d]))
    last_ring = (ring_count - 1) * longitude_segments
    for longitude_index in range(longitude_segments):
        next_longitude = (longitude_index + 1) % longitude_segments
        faces.append([top_index, last_ring + longitude_index, last_ring + next_longitude])

    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def connected_component_count(mesh: trimesh.Trimesh) -> int:
    """Count face-connected components without optional scipy/networkx dependencies."""

    face_count = len(mesh.faces)
    if face_count == 0:
        return 0
    parents = np.arange(face_count, dtype=np.int64)

    def find(index: int) -> int:
        root = index
        while parents[root] != root:
            root = int(parents[root])
        while parents[index] != index:
            parent = int(parents[index])
            parents[index] = root
            index = parent
        return root

    for left, right in np.asarray(mesh.face_adjacency, dtype=np.int64):
        left_root = find(int(left))
        right_root = find(int(right))
        if left_root != right_root:
            parents[right_root] = left_root
    return len({find(index) for index in range(face_count)})


def color_mesh(
    mesh: trimesh.Trimesh,
    texture: np.ndarray,
    *,
    appearance_mode: str,
) -> np.ndarray:
    bounds = mesh.bounds
    width = bounds[1, 0] - bounds[0, 0]
    height = bounds[1, 1] - bounds[0, 1]
    u = (mesh.vertices[:, 0] - bounds[0, 0]) / width
    v = 1.0 - (mesh.vertices[:, 1] - bounds[0, 1]) / height
    back = mesh.vertices[:, 2] < 0
    u = np.where(back, 1.0 - u, u)
    sampled = sample_texture(texture, u, v)
    rgb = normalize_appearance(sampled, appearance_mode).astype(np.float64)

    # Collapse UVs at the top and bottom poles would otherwise create radial color fans.
    edge_rows = max(1, texture.shape[0] // 18)
    top_color = normalize_appearance(
        texture[:edge_rows].reshape(-1, 3).mean(axis=0, keepdims=True),
        appearance_mode,
    )[0]
    bottom_color = normalize_appearance(
        texture[-edge_rows:].reshape(-1, 3).mean(axis=0, keepdims=True),
        appearance_mode,
    )[0]
    center_y = (bounds[0, 1] + bounds[1, 1]) / 2.0
    normalized_y = np.abs((mesh.vertices[:, 1] - center_y) / (height / 2.0))
    pole_weight = np.clip((normalized_y - 0.76) / 0.22, 0.0, 1.0) ** 2
    pole_color = np.where(
        (mesh.vertices[:, 1] >= center_y)[:, None],
        top_color,
        bottom_color,
    )
    rgb = rgb * (1.0 - pole_weight[:, None]) + pole_color * pole_weight[:, None]
    rgb = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    rgba = np.column_stack((rgb, np.full(len(rgb), 255, dtype=np.uint8)))
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=rgba)
    return rgba


def sample_colored_surface(
    mesh: trimesh.Trimesh,
    count: int,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    probabilities = mesh.area_faces / mesh.area
    face_ids = rng.choice(len(mesh.faces), size=count, replace=True, p=probabilities)
    triangles = mesh.vertices[mesh.faces[face_ids]]
    first = rng.random(count)
    second = rng.random(count)
    root = np.sqrt(first)
    barycentric = np.column_stack((1.0 - root, root * (1.0 - second), root * second))
    points = np.einsum("ni,nij->nj", barycentric, triangles)
    vertex_colors = np.asarray(mesh.visual.vertex_colors, dtype=np.float64)[:, :3]
    triangle_colors = vertex_colors[mesh.faces[face_ids]]
    colors = np.einsum("ni,nij->nj", barycentric, triangle_colors)
    normals = mesh.face_normals[face_ids]
    return points, normals, np.clip(np.rint(colors), 0, 255).astype(np.uint8)


def normal_quaternions(normals: np.ndarray) -> np.ndarray:
    quaternions = np.zeros((len(normals), 4), dtype=np.float32)
    for index, normal in enumerate(normals):
        normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
        dot = float(np.clip(normal[2], -1.0, 1.0))
        if dot < -0.999999:
            quaternions[index] = [0.0, 1.0, 0.0, 0.0]
            continue
        cross = np.asarray([-normal[1], normal[0], 0.0], dtype=np.float64)
        quaternion = np.asarray([1.0 + dot, *cross], dtype=np.float64)
        quaternion /= max(float(np.linalg.norm(quaternion)), 1e-12)
        quaternions[index] = quaternion.astype(np.float32)
    return quaternions


def write_gaussian_ply(
    path: Path,
    mesh: trimesh.Trimesh,
    *,
    count: int,
    seed: int,
) -> dict[str, Any]:
    points, normals, colors = sample_colored_surface(mesh, count, seed=seed)
    if count <= 0:
        raise ValueError("Gaussian surface requires a positive sample count")
    dtype: list[tuple[str, str]] = [(name, "f4") for name in GAUSSIAN_PROPERTY_NAMES]
    data = np.zeros(count, dtype=dtype)
    for axis, name in enumerate(("x", "y", "z")):
        data[name] = points[:, axis].astype(np.float32)
    for axis, name in enumerate(("nx", "ny", "nz")):
        data[name] = normals[:, axis].astype(np.float32)
    normalized_colors = colors.astype(np.float32) / 255.0
    for channel, name in enumerate(("f_dc_0", "f_dc_1", "f_dc_2")):
        data[name] = (normalized_colors[:, channel] - 0.5) / SH_C0
    data["opacity"] = math.log(0.94 / 0.06)
    spacing = math.sqrt(float(mesh.area) / count) * 0.78
    data["scale_0"] = math.log(spacing)
    data["scale_1"] = math.log(spacing)
    data["scale_2"] = math.log(spacing * 0.30)
    quaternions = normal_quaternions(normals)
    for channel, name in enumerate(("rot_0", "rot_1", "rot_2", "rot_3")):
        data[name] = quaternions[:, channel]
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(data, "vertex")], text=False, byte_order="<").write(path)
    return {
        "vertex_count": count,
        "property_names": list(GAUSSIAN_PROPERTY_NAMES),
        "sha256": sha256_file(path),
        "surface_spacing": spacing,
        "mean_rgb": [float(value) for value in colors.mean(axis=0)],
        "median_rgb": [float(value) for value in np.median(colors, axis=0)],
    }


def color_metrics(colors: np.ndarray) -> dict[str, Any]:
    values = colors.astype(np.float64)
    luminance = np.sum(values * np.asarray([0.2126, 0.7152, 0.0722]), axis=-1)
    return {
        "count": len(values),
        "mean_rgb": [float(value) for value in values.mean(axis=0)],
        "median_rgb": [float(value) for value in np.median(values, axis=0)],
        "mean_luminance": float(luminance.mean()),
        "median_luminance": float(np.median(luminance)),
        "light_pixel_fraction": float(np.mean(luminance >= 160.0)),
    }


def six_side_metrics(mesh: trimesh.Trimesh, rgba: np.ndarray) -> dict[str, dict[str, Any]]:
    vertices = mesh.vertices
    bounds = mesh.bounds
    extents = mesh.extents
    axes = {
        "front": vertices[:, 2] >= bounds[1, 2] - extents[2] * 0.18,
        "back": vertices[:, 2] <= bounds[0, 2] + extents[2] * 0.18,
        "left": vertices[:, 0] <= bounds[0, 0] + extents[0] * 0.10,
        "right": vertices[:, 0] >= bounds[1, 0] - extents[0] * 0.10,
        "top": vertices[:, 1] >= bounds[1, 1] - extents[1] * 0.10,
        "bottom": vertices[:, 1] <= bounds[0, 1] + extents[1] * 0.10,
    }
    empty = [name for name, mask in axes.items() if not np.any(mask)]
    if empty:
        raise ValueError(f"six-side audit has empty vertex masks: {empty}")
    return {name: color_metrics(rgba[mask, :3]) for name, mask in axes.items()}


def _sorted_rgb(colors: np.ndarray) -> np.ndarray:
    rgb = np.asarray(colors, dtype=np.uint8)[:, :3]
    return rgb[np.lexsort((rgb[:, 2], rgb[:, 1], rgb[:, 0]))]


def compare_vertex_colors(expected: np.ndarray, observed: np.ndarray) -> dict[str, Any]:
    expected_rgb = _sorted_rgb(expected)
    observed_rgb = _sorted_rgb(observed)
    same_count = len(expected_rgb) == len(observed_rgb)
    max_channel_error = None
    if same_count:
        difference = np.abs(expected_rgb.astype(np.int16) - observed_rgb.astype(np.int16))
        max_channel_error = int(difference.max(initial=0))
    return {
        "expected_count": len(expected_rgb),
        "observed_count": len(observed_rgb),
        "max_channel_error": max_channel_error,
        "preserved": bool(same_count and max_channel_error is not None and max_channel_error <= 1),
    }


def read_glb_vertex_colors(path: Path) -> np.ndarray:
    scene = trimesh.load_scene(path)
    colors: list[np.ndarray] = []
    for geometry in scene.geometry.values():
        vertex_colors = getattr(geometry.visual, "vertex_colors", None)
        if vertex_colors is not None and len(vertex_colors) == len(geometry.vertices):
            colors.append(np.asarray(vertex_colors, dtype=np.uint8))
    if not colors:
        raise ValueError(f"GLB contains no per-vertex colors: {path}")
    return np.vstack(colors)


def read_obj_vertex_colors(path: Path) -> np.ndarray:
    values: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("v "):
            continue
        fields = line.split()
        if len(fields) < 7:
            raise ValueError(f"OBJ vertex has no RGB values: {line}")
        values.append([float(value) for value in fields[4:7]])
    if not values:
        raise ValueError(f"OBJ contains no colored vertices: {path}")
    colors = np.asarray(values, dtype=np.float64)
    if float(colors.max()) <= 1.0 + 1e-6:
        colors *= 255.0
    return np.column_stack(
        (
            np.clip(np.rint(colors), 0, 255).astype(np.uint8),
            np.full(len(colors), 255, dtype=np.uint8),
        )
    )


def export_assets(
    *,
    source_image: Path,
    output_dir: Path,
    object_id: str,
    appearance_mode: str,
    thickness_ratio: float,
    latitude_segments: int,
    longitude_segments: int,
    shape_exponent: float,
    gaussian_count: int,
    seed: int,
) -> dict[str, Any]:
    rgba_image = Image.open(source_image).convert("RGBA")
    rgba = np.asarray(rgba_image, dtype=np.uint8)
    x0, y0, x1, y1 = visible_bbox(rgba[..., 3])
    bbox_width = x1 - x0 + 1
    bbox_height = y1 - y0 + 1
    aspect = bbox_width / bbox_height
    height = 1.0
    width = aspect
    depth = min(width, height) * thickness_ratio
    texture = extend_visible_texture(rgba[y0 : y1 + 1, x0 : x1 + 1])
    mesh = build_superellipsoid(
        width=width,
        height=height,
        depth=depth,
        latitude_segments=latitude_segments,
        longitude_segments=longitude_segments,
        shape_exponent=shape_exponent,
    )
    vertex_colors = color_mesh(mesh, texture, appearance_mode=appearance_mode)
    output_dir.mkdir(parents=True, exist_ok=True)
    glb_path = output_dir / f"{object_id}.glb"
    obj_path = output_dir / f"{object_id}.obj"
    gaussian_path = output_dir / f"{object_id}_gaussian.ply"
    material_reference = output_dir / "material_reference.png"
    normalized_texture = normalize_appearance(texture.reshape((-1, 3)), appearance_mode).reshape(
        texture.shape
    )
    Image.fromarray(normalized_texture).save(material_reference)
    mesh.export(glb_path)
    obj_text = trimesh.exchange.obj.export_obj(mesh, include_color=True)
    obj_path.write_text(obj_text, encoding="utf-8")
    glb_color_audit = compare_vertex_colors(vertex_colors, read_glb_vertex_colors(glb_path))
    obj_color_audit = compare_vertex_colors(vertex_colors, read_obj_vertex_colors(obj_path))
    gaussian = write_gaussian_ply(gaussian_path, mesh, count=gaussian_count, seed=seed)
    extents = [float(value) for value in mesh.extents]
    planar = sorted(extents, reverse=True)[:2]
    generated_aspect = max(planar) / min(planar)
    component_count = connected_component_count(mesh)
    views = six_side_metrics(mesh, vertex_colors)
    view_luminance = [metrics["mean_luminance"] for metrics in views.values()]
    white_view_gate = all(
        metrics["median_luminance"] >= 190.0
        and metrics["light_pixel_fraction"] >= 0.95
        and min(metrics["mean_rgb"]) >= 180.0
        for metrics in views.values()
    )
    gates = {
        "finite_geometry": bool(np.isfinite(mesh.vertices).all()),
        "watertight_closed_surface": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "positive_volume": bool(mesh.volume > 0.0),
        "single_connected_component": component_count == 1,
        "source_aspect_preserved": (
            abs(math.log(generated_aspect / max(aspect, 1.0 / aspect))) <= 0.03
        ),
        "volumetric_thickness": 0.22 <= min(extents) / sorted(extents, reverse=True)[1] <= 0.42,
        "six_side_material_consistency": max(view_luminance) - min(view_luminance) <= 38.0,
        "white_fabric_is_white_on_all_sides": appearance_mode != "white_fabric" or white_view_gate,
        "glb_vertex_colors_preserved": glb_color_audit["preserved"],
        "obj_vertex_colors_preserved": obj_color_audit["preserved"],
    }
    artifacts = [glb_path, obj_path, gaussian_path, material_reference]
    report = {
        "schema_version": 1,
        "kind": "video2world.parametric_soft_object_completion",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "technical_gates_passed" if all(gates.values()) else "rejected",
        "promotion_status": (
            "held_pending_browser_six_view_review" if all(gates.values()) else "blocked"
        ),
        "object_id": object_id,
        "source": {
            "path": str(source_image.resolve()),
            "sha256": sha256_file(source_image),
            "width": rgba_image.width,
            "height": rgba_image.height,
            "visible_bbox_xyxy": [x0, y0, x1, y1],
            "visible_bbox_aspect": aspect,
            "visible_color": color_metrics(rgba[..., :3][rgba[..., 3] >= 128]),
        },
        "method": {
            "geometry": "closed colored superellipsoid with corner thickness falloff",
            "hidden_surface": "mirrored observed material family",
            "appearance_mode": appearance_mode,
            "thickness_ratio": thickness_ratio,
            "shape_exponent": shape_exponent,
            "latitude_segments": latitude_segments,
            "longitude_segments": longitude_segments,
            "seed": seed,
        },
        "mesh": {
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "extents": extents,
            "requested_width_height_depth": [width, height, depth],
            "observed_thickness_to_short_planar_ratio": (
                min(extents) / sorted(extents, reverse=True)[1]
            ),
            "volume": float(mesh.volume),
            "area": float(mesh.area),
            "connected_components": component_count,
            "watertight": bool(mesh.is_watertight),
            "winding_consistent": bool(mesh.is_winding_consistent),
        },
        "mesh_color_roundtrip": {
            "glb": glb_color_audit,
            "obj": obj_color_audit,
        },
        "gaussian": gaussian,
        "six_side_appearance": views,
        "browser_six_view_review": {
            "status": "pending" if all(gates.values()) else "blocked",
            "required_views": ["front", "back", "left", "right", "top", "bottom"],
            "promotion_allowed": False,
        },
        "acceptance_gates": gates,
        "all_acceptance_gates_passed": all(gates.values()),
        "artifacts": [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in artifacts
        ],
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "trimesh": trimesh.__version__,
            "plyfile": getattr(plyfile, "__version__", "unknown"),
        },
    }
    write_json(output_dir / "completion_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument(
        "--appearance-mode",
        choices=("source", "source_light", "white_fabric"),
        default="source",
    )
    parser.add_argument("--thickness-ratio", type=float, default=0.30)
    parser.add_argument("--latitude-segments", type=int, default=72)
    parser.add_argument("--longitude-segments", type=int, default=144)
    parser.add_argument("--shape-exponent", type=float, default=0.34)
    parser.add_argument("--gaussian-count", type=int, default=80_000)
    parser.add_argument("--seed", type=int, default=20260717)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.20 <= args.thickness_ratio <= 0.45:
        raise SystemExit("--thickness-ratio must be in [0.20, 0.45]")
    report = export_assets(
        source_image=args.source_image.resolve(),
        output_dir=args.output_dir.resolve(),
        object_id=args.object_id,
        appearance_mode=args.appearance_mode,
        thickness_ratio=args.thickness_ratio,
        latitude_segments=args.latitude_segments,
        longitude_segments=args.longitude_segments,
        shape_exponent=args.shape_exponent,
        gaussian_count=args.gaussian_count,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if report["all_acceptance_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
