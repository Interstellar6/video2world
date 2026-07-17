#!/usr/bin/env python3
"""Render one accepted scene-fit GLB into calibrated source-camera RGBA and Z.

Run this script with Blender and pass script arguments after ``--``. The
renderer fails closed unless a built-in plane probe proves that Blender's
Depth pass is positive camera-axis metric Z for the running Blender build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

RAW_TO_BLENDER = np.asarray(
    (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, -1.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ),
    dtype=np.float64,
)
PBR_RECEIPT_KIND = "video2world.pbr_layer_render_receipt"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_camera(path: Path, frame_id: str) -> dict[str, Any]:
    payload = read_json(path)
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
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(rotation)):
        raise ValueError("camera position/rotation contains non-finite values")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("camera rotation must be orthonormal")
    if np.linalg.det(rotation) <= 0:
        raise ValueError("camera rotation must be right handed")
    width = int(camera.get("width", 0))
    height = int(camera.get("height", 0))
    fx = float(camera.get("fx", 0.0))
    fy = float(camera.get("fy", 0.0))
    if width <= 0 or height <= 0 or fx <= 0 or fy <= 0:
        raise ValueError("camera intrinsics must be positive")
    cx = float(camera.get("cx", width / 2.0))
    cy = float(camera.get("cy", height / 2.0))
    if not np.isclose(cx, width / 2.0, atol=1e-6) or not np.isclose(
        cy, height / 2.0, atol=1e-6
    ):
        raise ValueError("non-centered principal points are not yet supported")
    return {
        "frame_id": frame_id,
        "position": position,
        "rotation": rotation,
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "raw": camera,
    }


def camera_to_blender_world_matrix(camera: dict[str, Any]) -> np.ndarray:
    position = RAW_TO_BLENDER @ np.append(camera["position"], 1.0)
    rotation = camera["rotation"]
    raw_axes = np.column_stack(
        (
            rotation[:, 0],
            -rotation[:, 1],
            -rotation[:, 2],
        )
    )
    blender_axes = RAW_TO_BLENDER[:3, :3] @ raw_axes
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = blender_axes
    matrix[:3, 3] = position[:3]
    return matrix


def scene_fit_raw_transform(report: dict[str, Any]) -> tuple[str, np.ndarray]:
    runtime = report.get("runtime_transform")
    baked = report.get("baked_relative_transform")
    if isinstance(runtime, dict):
        if runtime.get("asset_coordinates_baked") is True:
            raise ValueError("runtime transform contradicts baked asset coordinates")
        matrix = np.asarray(runtime.get("matrix_row_major"), dtype=np.float64)
        mode = "full_affine_runtime_transform"
    elif isinstance(baked, dict):
        pivot = np.asarray(baked.get("runtime_pivot"), dtype=np.float64)
        if pivot.shape != (3,) or not np.all(np.isfinite(pivot)):
            raise ValueError("baked scene-fit report has no finite runtime pivot")
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 3] = pivot
        mode = "baked_linear_plus_runtime_pivot"
    else:
        raise ValueError("scene-fit report has no supported placement transform")
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("scene-fit transform must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError("scene-fit transform must be affine")
    return mode, matrix


def raw_transform_to_blender(matrix: np.ndarray) -> np.ndarray:
    return RAW_TO_BLENDER @ matrix @ np.linalg.inv(RAW_TO_BLENDER)


def expected_mesh_sha256(report: dict[str, Any]) -> str:
    fitted_mesh_sha = report.get("mesh", {}).get("glb_sha256")
    if isinstance(fitted_mesh_sha, str):
        return fitted_mesh_sha
    affine_mesh_sha = report.get("sources", {}).get("mesh", {}).get("sha256")
    if isinstance(affine_mesh_sha, str):
        return affine_mesh_sha
    raise ValueError("scene-fit report does not bind its GLB")


def validate_scene_fit(report: dict[str, Any], mesh_path: Path, layer_id: str) -> None:
    if report.get("object_id") != layer_id:
        raise ValueError("layer id does not match scene-fit object id")
    if report.get("all_acceptance_gates_passed") is not True:
        raise ValueError("scene-fit report is not accepted")
    if sha256_file(mesh_path) != expected_mesh_sha256(report):
        raise ValueError("GLB hash does not match scene-fit report")
    scene_fit_raw_transform(report)


def read_cstring(stream: BinaryIO) -> str:
    value = bytearray()
    while True:
        char = stream.read(1)
        if not char:
            raise ValueError("unexpected EOF in EXR string")
        if char == b"\x00":
            return value.decode("ascii")
        value.extend(char)


def parse_exr_channels(value: bytes) -> list[tuple[str, int, int, int]]:
    result: list[tuple[str, int, int, int]] = []
    cursor = 0
    while cursor < len(value) and value[cursor] != 0:
        end = value.index(b"\x00", cursor)
        name = value[cursor:end].decode("ascii")
        cursor = end + 1
        if cursor + 16 > len(value):
            raise ValueError("truncated EXR channel list")
        pixel_type = struct.unpack_from("<i", value, cursor)[0]
        x_sampling, y_sampling = struct.unpack_from("<ii", value, cursor + 8)
        cursor += 16
        result.append((name, pixel_type, x_sampling, y_sampling))
    return result


def read_uncompressed_float_exr(path: Path, *, channel_name: str = "Depth.V") -> np.ndarray:
    with path.open("rb") as stream:
        if stream.read(4) != struct.pack("<I", 20000630):
            raise ValueError("depth output is not an OpenEXR file")
        version = struct.unpack("<I", stream.read(4))[0]
        if version != 2:
            raise ValueError(f"unsupported OpenEXR version/flags: {version}")
        attributes: dict[str, tuple[str, bytes]] = {}
        while True:
            name = read_cstring(stream)
            if not name:
                break
            kind = read_cstring(stream)
            size_raw = stream.read(4)
            if len(size_raw) != 4:
                raise ValueError("truncated OpenEXR attribute")
            size = struct.unpack("<I", size_raw)[0]
            value = stream.read(size)
            if len(value) != size:
                raise ValueError("truncated OpenEXR attribute value")
            attributes[name] = (kind, value)
        header_end = stream.tell()
        if attributes.get("compression", (None, None))[1] != b"\x00":
            raise ValueError("depth EXR must use no compression")
        data_window = attributes.get("dataWindow", (None, b""))[1]
        if len(data_window) != 16:
            raise ValueError("depth EXR has no valid data window")
        x0, y0, x1, y1 = struct.unpack("<4i", data_window)
        width, height = x1 - x0 + 1, y1 - y0 + 1
        if width <= 0 or height <= 0:
            raise ValueError("depth EXR has invalid dimensions")
        channels = parse_exr_channels(attributes.get("channels", (None, b""))[1])
        if channels != [(channel_name, 2, 1, 1)]:
            raise ValueError(f"depth EXR channel contract mismatch: {channels}")
        stream.seek(header_end)
        offsets_raw = stream.read(8 * height)
        if len(offsets_raw) != 8 * height:
            raise ValueError("truncated OpenEXR scanline offset table")
        offsets = struct.unpack(f"<{height}Q", offsets_raw)
        result = np.full((height, width), np.nan, dtype=np.float32)
        seen_rows: set[int] = set()
        for offset in offsets:
            stream.seek(offset)
            chunk_header = stream.read(8)
            if len(chunk_header) != 8:
                raise ValueError("truncated OpenEXR scanline chunk")
            y, size = struct.unpack("<iI", chunk_header)
            if y < y0 or y > y1 or y in seen_rows:
                raise ValueError("invalid or repeated OpenEXR scanline")
            payload = stream.read(size)
            if size != width * 4 or len(payload) != size:
                raise ValueError("unexpected OpenEXR float scanline size")
            result[y - y0] = np.frombuffer(payload, dtype="<f4")
            seen_rows.add(y)
        if len(seen_rows) != height:
            raise ValueError("OpenEXR is missing scanlines")
        return result


def parse_frame_ids(values: list[str]) -> list[str]:
    frame_ids: list[str] = []
    for value in values:
        frame_ids.extend(item.strip() for item in value.split(",") if item.strip())
    if not frame_ids or len(set(frame_ids)) != len(frame_ids):
        raise ValueError("frame ids must be non-empty and unique")
    return frame_ids


def reset_blender_scene(bpy: Any) -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def configure_depth_output(bpy: Any, scene: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    # Blender 5.1 captures this base when the File Output node is created.
    scene.render.filepath = str(path.parent) + "/"
    scene.view_layers[0].use_pass_z = True
    scene.use_nodes = True
    node_group = bpy.data.node_groups.new(
        f"video2world_depth_{path.stem}", "CompositorNodeTree"
    )
    scene.compositing_node_group = node_group
    render_layers = node_group.nodes.new("CompositorNodeRLayers")
    file_output = node_group.nodes.new("CompositorNodeOutputFile")
    file_output.file_name = path.name
    file_output.format.file_format = "OPEN_EXR_MULTILAYER"
    depth_item = file_output.file_output_items.new("FLOAT", "Depth")
    depth_item.override_node_format = True
    depth_item.format.file_format = "OPEN_EXR"
    depth_item.format.color_mode = "BW"
    depth_item.format.color_depth = "32"
    depth_item.format.exr_codec = "NONE"
    node_group.links.new(render_layers.outputs["Depth"], file_output.inputs["Depth"])


def materialize_depth_output(bpy: Any, path: Path) -> Path:
    del bpy
    if not path.is_file():
        raise ValueError(f"rendered depth EXR is absent from exact target: {path}")
    return path


def configure_render_scene(scene: Any, *, width: int, height: int, fx: float, fy: float) -> None:
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = fx / fy
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.world.color = (0.08, 0.08, 0.08)


def load_saved_rgba(bpy: Any, path: Path) -> tuple[Any, np.ndarray]:
    image = bpy.data.images.load(str(path), check_existing=False)
    width, height = image.size[:]
    pixels = np.asarray(image.pixels[:], dtype=np.float32)
    if pixels.size != width * height * 4:
        bpy.data.images.remove(image)
        raise ValueError("saved RGBA could not be read back")
    return image, np.flipud(pixels.reshape(height, width, 4).copy())


def save_render_alpha(bpy: Any, scene: Any, path: Path) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.data.images["Render Result"].save_render(filepath=str(path), scene=scene)
    image, rgba = load_saved_rgba(bpy, path)
    bpy.data.images.remove(image)
    return rgba[:, :, 3]


def clear_saved_rgba_alpha(bpy: Any, path: Path, clear_mask: np.ndarray) -> np.ndarray:
    image, rgba = load_saved_rgba(bpy, path)
    if clear_mask.shape != rgba.shape[:2]:
        bpy.data.images.remove(image)
        raise ValueError("alpha sanitization mask shape mismatch")
    rgba[clear_mask, 3] = 0.0
    image.pixels.foreach_set(np.flipud(rgba).reshape(-1))
    image.filepath_raw = str(path)
    image.file_format = "PNG"
    image.save()
    bpy.data.images.remove(image)
    verified_image, verified_rgba = load_saved_rgba(bpy, path)
    bpy.data.images.remove(verified_image)
    if np.any(verified_rgba[clear_mask, 3] > 0):
        raise ValueError("RGBA alpha sanitization did not persist")
    return verified_rgba[:, :, 3]


def write_saved_rgba_binary_alpha(
    bpy: Any,
    path: Path,
    alpha_mask: np.ndarray,
) -> np.ndarray:
    image, rgba = load_saved_rgba(bpy, path)
    if alpha_mask.shape != rgba.shape[:2]:
        bpy.data.images.remove(image)
        raise ValueError("geometry-derived alpha mask shape mismatch")
    rgba[:, :, 3] = alpha_mask.astype(np.float32)
    image.pixels.foreach_set(np.flipud(rgba).reshape(-1))
    image.filepath_raw = str(path)
    image.file_format = "PNG"
    image.save()
    bpy.data.images.remove(image)
    verified_image, verified_rgba = load_saved_rgba(bpy, path)
    bpy.data.images.remove(verified_image)
    verified_mask = verified_rgba[:, :, 3] >= 0.5
    if not np.array_equal(verified_mask, alpha_mask):
        raise ValueError("geometry-derived binary alpha did not persist")
    return verified_rgba[:, :, 3]


def load_binary_mask_png(bpy: Any, path: Path) -> np.ndarray:
    image, rgba = load_saved_rgba(bpy, path)
    bpy.data.images.remove(image)
    return rgba[:, :, 0] >= 0.5


def make_blender_camera(bpy: Any, Matrix: Any, camera: dict[str, Any]) -> Any:
    data = bpy.data.cameras.new("video2world_camera")
    camera_object = bpy.data.objects.new("video2world_camera", data)
    bpy.context.collection.objects.link(camera_object)
    camera_object.matrix_world = Matrix(camera_to_blender_world_matrix(camera).tolist())
    sensor_width = 36.0
    data.sensor_fit = "HORIZONTAL"
    data.sensor_width = sensor_width
    data.lens = camera["fx"] * sensor_width / camera["width"]
    data.clip_start = 0.01
    data.clip_end = 1000.0
    return camera_object


def add_point_light(bpy: Any, Vector: Any, name: str, location: np.ndarray, energy: float) -> None:
    data = bpy.data.lights.new(name=name, type="POINT")
    data.energy = energy
    data.shadow_soft_size = 4.0
    light = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(light)
    light.location = Vector(location.tolist())


def import_scene_fit_glb(
    bpy: Any,
    Matrix: Any,
    mesh_path: Path,
    raw_transform: np.ndarray,
    layer_id: str,
) -> list[Any]:
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(mesh_path))
    imported = [item for item in bpy.data.objects if item not in before]
    imported_set = set(imported)
    roots = [item for item in imported if item.parent not in imported_set]
    if not roots:
        raise ValueError("GLB import produced no logical root")
    transform = Matrix(raw_transform_to_blender(raw_transform).tolist())
    for root in roots:
        root.matrix_world = transform @ root.matrix_world
    for item in imported:
        item["video2world_entity"] = layer_id
    bpy.context.view_layer.update()
    return imported


def evaluated_geometry_depth_range(
    bpy: Any,
    camera_object: Any,
    imported: list[Any],
) -> tuple[float, float]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    camera_inverse = camera_object.matrix_world.inverted()
    depths: list[float] = []
    for item in imported:
        if item.type != "MESH":
            continue
        evaluated = item.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            for vertex in mesh.vertices:
                camera_point = camera_inverse @ (evaluated.matrix_world @ vertex.co)
                depth = -float(camera_point.z)
                if np.isfinite(depth) and depth > 0:
                    depths.append(depth)
        finally:
            evaluated.to_mesh_clear()
    if not depths:
        raise ValueError("evaluated GLB has no vertices in front of the source camera")
    return min(depths), max(depths)


def mask_bbox(mask: np.ndarray) -> list[int] | None:
    y, x = np.nonzero(mask)
    if len(x) == 0:
        return None
    return [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1]


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    union = int(np.count_nonzero(left | right))
    return float(np.count_nonzero(left & right) / union) if union else 0.0


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


def mask_center(mask: np.ndarray) -> np.ndarray:
    y, x = np.nonzero(mask)
    if len(x) == 0:
        raise ValueError("cannot compute center of an empty mask")
    return np.asarray([x.mean(), y.mean()], dtype=np.float64)


def silhouette_alignment_metrics(
    expected: np.ndarray,
    rendered: np.ndarray,
) -> dict[str, Any]:
    if expected.shape != rendered.shape or not np.any(expected) or not np.any(rendered):
        raise ValueError("analytic and rendered silhouettes must be non-empty and aligned")
    intersection = int(np.count_nonzero(expected & rendered))
    union = int(np.count_nonzero(expected | rendered))
    expected_bbox = mask_bbox(expected)
    rendered_bbox = mask_bbox(rendered)
    assert expected_bbox is not None and rendered_bbox is not None
    return {
        "mask_iou": intersection / union,
        "rendered_precision": intersection / int(rendered.sum()),
        "analytic_recall": intersection / int(expected.sum()),
        "bbox_iou": bbox_iou(expected_bbox, rendered_bbox),
        "center_error_px": float(np.linalg.norm(mask_center(expected) - mask_center(rendered))),
        "analytic_bbox_xyxy": expected_bbox,
        "rendered_bbox_xyxy": rendered_bbox,
    }


def binary_boundary(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    neighborhoods = [
        padded[y_offset : y_offset + mask.shape[0], x_offset : x_offset + mask.shape[1]]
        for y_offset in range(3)
        for x_offset in range(3)
    ]
    interior = np.logical_and.reduce(neighborhoods)
    return mask.astype(bool) & ~interior


def binary_boundary_collar(mask: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px < 0:
        raise ValueError("boundary collar radius must be non-negative")
    mask = mask.astype(bool)
    collar = binary_boundary(mask)
    for _ in range(radius_px):
        padded = np.pad(collar, 1, mode="constant", constant_values=False)
        neighborhoods = [
            padded[
                y_offset : y_offset + mask.shape[0],
                x_offset : x_offset + mask.shape[1],
            ]
            for y_offset in range(3)
            for x_offset in range(3)
        ]
        collar = np.logical_or.reduce(neighborhoods) & mask
    return collar


def plausible_geometry_depth_mask(
    *,
    raw_depth: np.ndarray,
    geometry_depth_min: float,
    geometry_depth_max: float,
    clip_start: float,
    clip_end: float,
) -> np.ndarray:
    if not 0 < clip_start < clip_end:
        raise ValueError("invalid depth clip range")
    if not 0 < geometry_depth_min <= geometry_depth_max < clip_end:
        raise ValueError("invalid evaluated geometry depth range")
    geometry_tolerance = max(1e-3, (geometry_depth_max - geometry_depth_min) * 1e-4)
    return (
        np.isfinite(raw_depth)
        & (raw_depth > clip_start)
        & (raw_depth < clip_end * 0.95)
        & (raw_depth >= geometry_depth_min - geometry_tolerance)
        & (raw_depth <= geometry_depth_max + geometry_tolerance)
    )


def plan_alpha_depth_sanitization(
    *,
    alpha: np.ndarray,
    raw_depth: np.ndarray,
    geometry_depth_min: float,
    geometry_depth_max: float,
    clip_start: float,
    clip_end: float,
    alpha_threshold: float,
    maximum_sanitized_fraction: float = 0.001,
    maximum_boundary_distance_px: int = 1,
    maximum_sanitizable_alpha: float = 0.6,
) -> dict[str, Any]:
    if alpha.shape != raw_depth.shape:
        raise ValueError("alpha and raw depth shape mismatch")
    if not 0 < clip_start < clip_end or not 0 < alpha_threshold <= 1:
        raise ValueError("invalid clip or alpha threshold")
    if not 0 <= maximum_sanitized_fraction <= 1:
        raise ValueError("invalid alpha sanitization fraction")
    if maximum_boundary_distance_px < 0:
        raise ValueError("invalid alpha sanitization boundary distance")
    if not alpha_threshold <= maximum_sanitizable_alpha <= 1.0:
        raise ValueError("invalid maximum sanitizable alpha")
    if not 0 < geometry_depth_min <= geometry_depth_max < clip_end:
        raise ValueError("invalid evaluated geometry depth range")
    initial_alpha = alpha >= alpha_threshold
    if not np.any(initial_alpha):
        raise ValueError("rendered layer has no opaque pixels")
    plausible_raw = plausible_geometry_depth_mask(
        raw_depth=raw_depth,
        geometry_depth_min=geometry_depth_min,
        geometry_depth_max=geometry_depth_max,
        clip_start=clip_start,
        clip_end=clip_end,
    )
    sanitization_mask = initial_alpha & ~plausible_raw
    sanitized_fraction = float(sanitization_mask.sum() / initial_alpha.sum())
    if sanitized_fraction > maximum_sanitized_fraction:
        raise ValueError("too many opaque pixels have depth outside evaluated geometry bounds")
    boundary_collar = binary_boundary_collar(
        initial_alpha,
        maximum_boundary_distance_px,
    )
    low_coverage_aa = initial_alpha & (alpha <= maximum_sanitizable_alpha)
    bounded_aa_support = boundary_collar | low_coverage_aa
    if np.any(sanitization_mask & ~bounded_aa_support):
        raise ValueError(
            "invalid opaque depth is outside the bounded alpha boundary collar or "
            "low-coverage antialiasing support"
        )
    spatial_sanitization_mask = sanitization_mask & boundary_collar
    low_coverage_sanitization_mask = (
        sanitization_mask & ~boundary_collar & low_coverage_aa
    )
    if not np.array_equal(
        spatial_sanitization_mask | low_coverage_sanitization_mask,
        sanitization_mask,
    ):
        raise ValueError("alpha sanitization path accounting is incomplete")
    return {
        "initial_alpha_mask": initial_alpha,
        "plausible_raw_mask": plausible_raw,
        "sanitization_mask": sanitization_mask,
        "spatial_sanitization_mask": spatial_sanitization_mask,
        "low_coverage_sanitization_mask": low_coverage_sanitization_mask,
        "sanitized_fraction": sanitized_fraction,
        "maximum_boundary_distance_px": maximum_boundary_distance_px,
        "maximum_sanitizable_alpha": maximum_sanitizable_alpha,
        "final_alpha_mask": initial_alpha & ~sanitization_mask,
    }


def plan_geometry_derived_binary_matte(
    *,
    alpha: np.ndarray,
    raw_depth: np.ndarray,
    geometry_depth_min: float,
    geometry_depth_max: float,
    clip_start: float,
    clip_end: float,
    alpha_threshold: float,
    minimum_initial_alpha_vs_depth_iou: float = 0.98,
    maximum_changed_union_fraction: float = 0.02,
) -> dict[str, Any]:
    if alpha.shape != raw_depth.shape:
        raise ValueError("alpha and raw depth shape mismatch")
    if not 0 < alpha_threshold <= 1:
        raise ValueError("invalid alpha threshold")
    if not 0 <= minimum_initial_alpha_vs_depth_iou <= 1:
        raise ValueError("invalid geometry-matte alignment IoU")
    if not 0 <= maximum_changed_union_fraction <= 1:
        raise ValueError("invalid geometry-matte changed fraction")
    initial_alpha = alpha >= alpha_threshold
    plausible_raw = plausible_geometry_depth_mask(
        raw_depth=raw_depth,
        geometry_depth_min=geometry_depth_min,
        geometry_depth_max=geometry_depth_max,
        clip_start=clip_start,
        clip_end=clip_end,
    )
    if not np.any(initial_alpha) or not np.any(plausible_raw):
        raise ValueError("geometry-derived matte inputs must be non-empty")
    union = initial_alpha | plausible_raw
    intersection = initial_alpha & plausible_raw
    cleared = initial_alpha & ~plausible_raw
    filled = plausible_raw & ~initial_alpha
    changed = cleared | filled
    initial_alpha_vs_depth_iou = float(intersection.sum() / union.sum())
    changed_union_fraction = float(changed.sum() / union.sum())
    if initial_alpha_vs_depth_iou < minimum_initial_alpha_vs_depth_iou:
        raise ValueError("initial RGBA alpha and geometry depth support IoU is too low")
    if changed_union_fraction >= maximum_changed_union_fraction:
        raise ValueError("geometry-derived matte would change an unbounded support fraction")
    return {
        "initial_alpha_mask": initial_alpha,
        "plausible_raw_mask": plausible_raw,
        "final_alpha_mask": plausible_raw,
        "cleared_mask": cleared,
        "filled_mask": filled,
        "changed_mask": changed,
        "union_pixels": int(union.sum()),
        "intersection_pixels": int(intersection.sum()),
        "initial_alpha_vs_depth_iou": initial_alpha_vs_depth_iou,
        "changed_union_fraction": changed_union_fraction,
        "minimum_initial_alpha_vs_depth_iou": minimum_initial_alpha_vs_depth_iou,
        "maximum_changed_union_fraction": maximum_changed_union_fraction,
    }


def calibrate_blender_depth(
    bpy: Any,
    Matrix: Any,
    output_root: Path,
) -> dict[str, Any]:
    reset_blender_scene(bpy)
    camera_data = bpy.data.cameras.new("video2world_depth_probe_camera")
    camera = bpy.data.objects.new("video2world_depth_probe_camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.matrix_world = Matrix.Identity(4)
    camera_data.sensor_fit = "HORIZONTAL"
    camera_data.sensor_width = 36.0
    camera_data.lens = 18.0
    camera_data.clip_start = 0.01
    camera_data.clip_end = 100.0
    bpy.ops.mesh.primitive_plane_add(size=40.0, location=(0.0, 0.0, -5.0))
    scene = bpy.context.scene
    scene.camera = camera
    configure_render_scene(scene, width=64, height=64, fx=32.0, fy=32.0)
    raw_path = output_root / "calibration" / "blender_depth_probe.exr"
    configure_depth_output(bpy, scene, raw_path)
    bpy.context.view_layer.update()
    bpy.ops.render.render()
    materialize_depth_output(bpy, raw_path)
    depth = read_uncompressed_float_exr(raw_path)
    finite = np.isfinite(depth) & (depth > 0) & (depth < 100.0)
    if not np.all(finite):
        raise ValueError("Blender depth probe did not cover the full calibration plane")
    maximum_error = float(np.max(np.abs(depth - 5.0)))
    spread = float(np.ptp(depth))
    passed = maximum_error <= 5e-3 and spread <= 5e-3
    receipt = {
        "schema_version": 1,
        "kind": "video2world.blender_depth_semantics_probe",
        "status": "passed" if passed else "rejected",
        "blender_version": bpy.app.version_string,
        "engine": "BLENDER_EEVEE",
        "probe_geometry": "fronto_parallel_plane_at_camera_axis_z_5",
        "depth_semantics": "positive_camera_axis_z",
        "expected_depth": 5.0,
        "observed_min": float(depth.min()),
        "observed_median": float(np.median(depth)),
        "observed_max": float(depth.max()),
        "maximum_absolute_error": maximum_error,
        "off_axis_spread": spread,
        "maximum_allowed_error": 5e-3,
        "raw_exr": str(raw_path),
        "raw_exr_sha256": sha256_file(raw_path),
    }
    write_json(output_root / "calibration" / "depth_semantics_receipt.json", receipt)
    if not passed:
        raise ValueError("Blender Depth pass is not reliable camera-axis metric Z")
    return receipt


def render_frame(
    *,
    bpy: Any,
    Matrix: Any,
    Vector: Any,
    mesh_path: Path,
    report: dict[str, Any],
    cameras_path: Path,
    frame_id: str,
    layer_id: str,
    output_root: Path,
    alpha_threshold: float,
    keep_raw_depth_exr: bool,
    depth_probe_receipt: dict[str, Any],
    analytic_silhouette_path: Path | None,
    minimum_silhouette_iou: float,
    minimum_silhouette_bbox_iou: float,
    maximum_silhouette_center_error_px: float,
    alpha_depth_alignment_mode: str,
) -> dict[str, Any]:
    frame_started = time.monotonic()
    reset_blender_scene(bpy)
    placement_mode, raw_transform = scene_fit_raw_transform(report)
    imported = import_scene_fit_glb(bpy, Matrix, mesh_path, raw_transform, layer_id)
    camera = load_camera(cameras_path, frame_id)
    camera_object = make_blender_camera(bpy, Matrix, camera)
    geometry_depth_min, geometry_depth_max = evaluated_geometry_depth_range(
        bpy, camera_object, imported
    )
    scene = bpy.context.scene
    scene.camera = camera_object
    configure_render_scene(
        scene,
        width=camera["width"],
        height=camera["height"],
        fx=camera["fx"],
        fy=camera["fy"],
    )
    raw_position = RAW_TO_BLENDER @ np.append(camera["position"], 1.0)
    add_point_light(
        bpy,
        Vector,
        "camera_fill",
        raw_position[:3] + np.asarray([0.0, 0.0, 1.0]),
        1400.0,
    )
    room_fill = RAW_TO_BLENDER @ np.asarray([1.0, -4.0, 8.0, 1.0])
    add_point_light(bpy, Vector, "room_fill", room_fill[:3], 1000.0)

    rgba_path = output_root / "rgba" / f"{frame_id}.png"
    depth_path = output_root / "depth" / f"{frame_id}.npy"
    raw_depth_path = output_root / "raw-depth" / f"{layer_id}_{frame_id}.exr"
    configure_depth_output(bpy, scene, raw_depth_path)
    bpy.context.view_layer.update()
    bpy.ops.render.render()
    materialize_depth_output(bpy, raw_depth_path)
    alpha = save_render_alpha(bpy, scene, rgba_path)
    raw_depth = read_uncompressed_float_exr(raw_depth_path)
    expected_shape = (camera["height"], camera["width"])
    if alpha.shape != expected_shape or raw_depth.shape != expected_shape:
        raise ValueError("RGBA/depth dimensions do not match source camera")
    if alpha_depth_alignment_mode == "bounded-sanitization":
        alignment = plan_alpha_depth_sanitization(
            alpha=alpha,
            raw_depth=raw_depth,
            geometry_depth_min=geometry_depth_min,
            geometry_depth_max=geometry_depth_max,
            clip_start=float(camera_object.data.clip_start),
            clip_end=float(camera_object.data.clip_end),
            alpha_threshold=alpha_threshold,
        )
        initial_alpha_mask = alignment["initial_alpha_mask"]
        plausible_raw = alignment["plausible_raw_mask"]
        invalid_alpha_depth = alignment["sanitization_mask"]
        if np.any(invalid_alpha_depth):
            alpha = clear_saved_rgba_alpha(bpy, rgba_path, invalid_alpha_depth)
        alignment_receipt = {
            "rgba_alpha_sanitized_pixels": int(invalid_alpha_depth.sum()),
            "rgba_alpha_sanitized_spatial_collar_pixels": int(
                alignment["spatial_sanitization_mask"].sum()
            ),
            "rgba_alpha_sanitized_low_coverage_outside_collar_pixels": int(
                alignment["low_coverage_sanitization_mask"].sum()
            ),
            "rgba_alpha_sanitized_fraction": alignment["sanitized_fraction"],
            "rgba_alpha_sanitization_policy": (
                "clear_if_depth_invalid_or_outside_geometry_within_bounded_alpha_edge"
            ),
            "rgba_alpha_sanitization_maximum_boundary_distance_px": alignment[
                "maximum_boundary_distance_px"
            ],
            "rgba_alpha_sanitization_maximum_low_coverage_alpha": alignment[
                "maximum_sanitizable_alpha"
            ],
        }
        alignment_gates = {
            "rgba_alpha_sanitization_fraction_below_0_001": (
                alignment["sanitized_fraction"] <= 0.001
            ),
            "rgba_alpha_sanitization_within_1px_boundary_collar": True,
            "rgba_alpha_sanitization_outside_collar_is_low_coverage_at_most_0_6": True,
            "rgba_alpha_sanitization_path_counts_sum_to_total": (
                int(alignment["spatial_sanitization_mask"].sum())
                + int(alignment["low_coverage_sanitization_mask"].sum())
                == int(invalid_alpha_depth.sum())
            ),
        }
    elif alpha_depth_alignment_mode == "geometry-derived-binary-matte":
        alignment = plan_geometry_derived_binary_matte(
            alpha=alpha,
            raw_depth=raw_depth,
            geometry_depth_min=geometry_depth_min,
            geometry_depth_max=geometry_depth_max,
            clip_start=float(camera_object.data.clip_start),
            clip_end=float(camera_object.data.clip_end),
            alpha_threshold=alpha_threshold,
        )
        initial_alpha_mask = alignment["initial_alpha_mask"]
        plausible_raw = alignment["plausible_raw_mask"]
        invalid_alpha_depth = alignment["cleared_mask"]
        alpha = write_saved_rgba_binary_alpha(
            bpy,
            rgba_path,
            alignment["final_alpha_mask"],
        )
        alignment_receipt = {
            "geometry_matte_provenance": (
                "raw_blender_z_within_evaluated_geometry_depth_bounds"
            ),
            "geometry_matte_binary_alpha": True,
            "geometry_matte_cleared_pixels": int(alignment["cleared_mask"].sum()),
            "geometry_matte_filled_pixels": int(alignment["filled_mask"].sum()),
            "geometry_matte_changed_pixels": int(alignment["changed_mask"].sum()),
            "geometry_matte_union_pixels": alignment["union_pixels"],
            "geometry_matte_intersection_pixels": alignment["intersection_pixels"],
            "geometry_matte_changed_union_fraction": alignment[
                "changed_union_fraction"
            ],
            "geometry_matte_initial_alpha_vs_depth_iou": alignment[
                "initial_alpha_vs_depth_iou"
            ],
            "geometry_matte_minimum_initial_alpha_vs_depth_iou": alignment[
                "minimum_initial_alpha_vs_depth_iou"
            ],
            "geometry_matte_maximum_changed_union_fraction": alignment[
                "maximum_changed_union_fraction"
            ],
            "geometry_matte_changed_union_fraction_limit_is_exclusive": True,
        }
        alignment_gates = {
            "geometry_matte_initial_alpha_vs_depth_iou_above_0_98": (
                alignment["initial_alpha_vs_depth_iou"] >= 0.98
            ),
            "geometry_matte_changed_union_fraction_below_0_02": (
                alignment["changed_union_fraction"] < 0.02
            ),
            "geometry_matte_is_binary": True,
        }
    else:
        raise ValueError(f"unsupported alpha/depth alignment mode: {alpha_depth_alignment_mode}")
    alpha_mask = alpha >= alpha_threshold
    if not np.array_equal(alpha_mask, alignment["final_alpha_mask"]):
        raise ValueError("final RGBA alpha differs from the planned alpha/depth alignment")
    measured_depth = np.full(expected_shape, np.nan, dtype=np.float32)
    measured_depth[alpha_mask] = raw_depth[alpha_mask]
    if np.any(alpha_mask & (~np.isfinite(measured_depth) | (measured_depth <= 0))):
        raise ValueError("opaque RGBA pixels still lack positive finite camera-axis depth")
    if not np.array_equal(np.isfinite(measured_depth), alpha_mask):
        raise ValueError("final RGBA alpha and depth validity support differ")
    raw_alpha_iou = mask_iou(plausible_raw, alpha_mask)
    if raw_alpha_iou < 0.98:
        raise ValueError("raw object depth and final RGBA alpha are not camera aligned")
    analytic_metrics = None
    analytic_gates: dict[str, bool] = {}
    if analytic_silhouette_path is not None:
        expected_silhouette = load_binary_mask_png(bpy, analytic_silhouette_path)
        analytic_metrics = silhouette_alignment_metrics(expected_silhouette, alpha_mask)
        analytic_gates = {
            "analytic_silhouette_mask_iou": (
                analytic_metrics["mask_iou"] >= minimum_silhouette_iou
            ),
            "analytic_silhouette_bbox_iou": (
                analytic_metrics["bbox_iou"] >= minimum_silhouette_bbox_iou
            ),
            "analytic_silhouette_center_error": (
                analytic_metrics["center_error_px"]
                <= maximum_silhouette_center_error_px
            ),
        }
        if not all(analytic_gates.values()):
            raise ValueError(
                f"analytic silhouette alignment failed for {layer_id} frame {frame_id}: "
                f"{analytic_metrics}"
            )
    depth_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(depth_path, measured_depth, allow_pickle=False)
    raw_exr_sha256 = sha256_file(raw_depth_path)
    if not keep_raw_depth_exr:
        raw_depth_path.unlink()

    receipt_path = output_root / "receipts" / f"{frame_id}.json"
    receipt = {
        "schema_version": 1,
        "kind": PBR_RECEIPT_KIND,
        "status": "technical_passed",
        "promotion_approved": False,
        "promotion_blocker": "source-camera silhouette and scene occlusion QA",
        "layer_id": layer_id,
        "frame_id": frame_id,
        "role": "downstream_object_render",
        "provenance_class": "downstream_object_render",
        "claims_measured_donor": False,
        "rgba": str(rgba_path),
        "rgba_sha256": sha256_file(rgba_path),
        "depth": str(depth_path),
        "depth_sha256": sha256_file(depth_path),
        "depth_unit": "meter",
        "depth_semantics": "positive_target_camera_axis_z",
        "depth_validity_scope": "alpha_greater_than_or_equal_to_threshold_only",
        "alpha_threshold": alpha_threshold,
        "alpha_depth_alignment_mode": alpha_depth_alignment_mode,
        "alpha_pixels_before_alignment": int(initial_alpha_mask.sum()),
        "alpha_pixels": int(alpha_mask.sum()),
        "depth_valid_pixels": int(np.isfinite(measured_depth).sum()),
        "depth_min": float(measured_depth[alpha_mask].min()),
        "depth_median": float(np.median(measured_depth[alpha_mask])),
        "depth_max": float(measured_depth[alpha_mask].max()),
        "evaluated_geometry_depth_min": geometry_depth_min,
        "evaluated_geometry_depth_max": geometry_depth_max,
        "raw_depth_outside_geometry_pixels": int(invalid_alpha_depth.sum()),
        **alignment_receipt,
        "raw_object_depth_bbox_xyxy": mask_bbox(plausible_raw),
        "final_alpha_bbox_xyxy": mask_bbox(alpha_mask),
        "raw_object_depth_vs_final_alpha_iou": raw_alpha_iou,
        "analytic_silhouette_metrics": analytic_metrics,
        "analytic_silhouette_thresholds": (
            {
                "minimum_mask_iou": minimum_silhouette_iou,
                "minimum_bbox_iou": minimum_silhouette_bbox_iou,
                "maximum_center_error_px": maximum_silhouette_center_error_px,
            }
            if analytic_silhouette_path is not None
            else None
        ),
        "elapsed_seconds": time.monotonic() - frame_started,
        "raw_depth_exr": str(raw_depth_path) if keep_raw_depth_exr else None,
        "raw_depth_exr_sha256": raw_exr_sha256,
        "placement_mode": placement_mode,
        "render_configuration": {
            "engine": "BLENDER_EEVEE",
            "taa_render_samples": int(scene.eevee.taa_render_samples),
            "pixel_filter_size": float(scene.render.filter_size),
            "rgb_sample_policy": "configured_eevee_render_samples",
            "alpha_policy": alpha_depth_alignment_mode,
        },
        "camera_convention": {
            "extrinsic": "camera_to_world",
            "rotation_columns": "right_down_forward",
            "image_y_axis": "down",
            "principal_point": "center",
        },
        "gates": {
            "depth_probe_passed": depth_probe_receipt["status"] == "passed",
            "rgba_camera_dimensions_match": True,
            "depth_camera_dimensions_match": True,
            "all_opaque_pixels_have_positive_finite_depth": True,
            **alignment_gates,
            "raw_object_depth_vs_final_alpha_iou_above_0_98": raw_alpha_iou >= 0.98,
            "final_alpha_support_equals_finite_depth_support": True,
            "depth_is_nan_outside_opaque_alpha": True,
            "claims_measured_donor_rejected": True,
            **analytic_gates,
        },
        "sources": {
            "mesh": {"path": str(mesh_path), "sha256": sha256_file(mesh_path)},
            "scene_fit_report_sha256": None,
            "cameras": {"path": str(cameras_path), "sha256": sha256_file(cameras_path)},
            "camera_record": camera["raw"],
            "analytic_silhouette": (
                {
                    "path": str(analytic_silhouette_path),
                    "sha256": sha256_file(analytic_silhouette_path),
                }
                if analytic_silhouette_path is not None
                else None
            ),
        },
    }
    write_json(receipt_path, receipt)
    return {"receipt_path": receipt_path, "receipt": receipt}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--scene-fit-report", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--layer-id", required=True)
    parser.add_argument("--frame-id", action="append", required=True)
    parser.add_argument("--alpha-threshold", type=float, default=0.5)
    parser.add_argument(
        "--alpha-depth-alignment-mode",
        choices=("bounded-sanitization", "geometry-derived-binary-matte"),
        default="bounded-sanitization",
    )
    parser.add_argument("--analytic-silhouette-dir", type=Path)
    parser.add_argument("--minimum-silhouette-iou", type=float, default=0.85)
    parser.add_argument("--minimum-silhouette-bbox-iou", type=float, default=0.8)
    parser.add_argument("--maximum-silhouette-center-error-px", type=float, default=10.0)
    parser.add_argument("--keep-raw-depth-exr", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    if not 0.0 < args.alpha_threshold <= 1.0:
        raise ValueError("alpha threshold must be in (0, 1]")
    output_root = args.output.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    mesh_path = args.mesh.expanduser().resolve()
    report_path = args.scene_fit_report.expanduser().resolve()
    cameras_path = args.cameras.expanduser().resolve()
    report = read_json(report_path)
    if not isinstance(report, dict):
        raise ValueError("scene-fit report must be a JSON object")
    validate_scene_fit(report, mesh_path, args.layer_id)
    frame_ids = parse_frame_ids(args.frame_id)
    analytic_silhouette_dir = (
        args.analytic_silhouette_dir.expanduser().resolve()
        if args.analytic_silhouette_dir
        else None
    )

    import bpy
    from mathutils import Matrix, Vector

    depth_probe_receipt = calibrate_blender_depth(bpy, Matrix, output_root)
    frame_records = []
    for frame_id in frame_ids:
        analytic_silhouette_path = (
            analytic_silhouette_dir / f"{frame_id}.png"
            if analytic_silhouette_dir is not None
            else None
        )
        if analytic_silhouette_path is not None and not analytic_silhouette_path.is_file():
            raise FileNotFoundError(analytic_silhouette_path)
        result = render_frame(
            bpy=bpy,
            Matrix=Matrix,
            Vector=Vector,
            mesh_path=mesh_path,
            report=report,
            cameras_path=cameras_path,
            frame_id=frame_id,
            layer_id=args.layer_id,
            output_root=output_root,
            alpha_threshold=args.alpha_threshold,
            keep_raw_depth_exr=args.keep_raw_depth_exr,
            depth_probe_receipt=depth_probe_receipt,
            analytic_silhouette_path=analytic_silhouette_path,
            minimum_silhouette_iou=args.minimum_silhouette_iou,
            minimum_silhouette_bbox_iou=args.minimum_silhouette_bbox_iou,
            maximum_silhouette_center_error_px=args.maximum_silhouette_center_error_px,
            alpha_depth_alignment_mode=args.alpha_depth_alignment_mode,
        )
        receipt = result["receipt"]
        receipt["sources"]["scene_fit_report"] = {
            "path": str(report_path),
            "sha256": sha256_file(report_path),
        }
        receipt["sources"].pop("scene_fit_report_sha256")
        write_json(result["receipt_path"], receipt)
        frame_records.append(
            {
                "frame_id": frame_id,
                "rgba": receipt["rgba"],
                "rgba_sha256": receipt["rgba_sha256"],
                "depth": receipt["depth"],
                "depth_sha256": receipt["depth_sha256"],
                "receipt": str(result["receipt_path"]),
                "receipt_sha256": sha256_file(result["receipt_path"]),
                "alpha_pixels": receipt["alpha_pixels"],
                "depth_valid_pixels": receipt["depth_valid_pixels"],
            }
        )

    manifest = {
        "schema_version": 1,
        "kind": "video2world.source_camera_pbr_layer_render_set",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "technical_passed_pending_silhouette_qa",
        "promotion_approved": False,
        "layer_id": args.layer_id,
        "role": "downstream_object_render",
        "claims_measured_donor": False,
        "alpha_depth_alignment_mode": args.alpha_depth_alignment_mode,
        "sources": {
            "mesh": {"path": str(mesh_path), "sha256": sha256_file(mesh_path)},
            "scene_fit_report": {
                "path": str(report_path),
                "sha256": sha256_file(report_path),
            },
            "cameras": {"path": str(cameras_path), "sha256": sha256_file(cameras_path)},
        },
        "depth_probe_receipt": {
            "path": str(output_root / "calibration" / "depth_semantics_receipt.json"),
            "sha256": sha256_file(
                output_root / "calibration" / "depth_semantics_receipt.json"
            ),
        },
        "analytic_silhouette_dir": (
            str(analytic_silhouette_dir) if analytic_silhouette_dir else None
        ),
        "render_elapsed_seconds": time.monotonic() - started,
        "frame_records": frame_records,
    }
    manifest_path = output_root / "render_manifest.json"
    write_json(manifest_path, manifest)
    write_json(
        output_root / "render_receipt.json",
        {
            "schema_version": 1,
            "kind": "video2world.source_camera_pbr_layer_render_set_receipt",
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "frame_count": len(frame_records),
            "frame_receipts_are_hashed_in_manifest": True,
            "claims_measured_donor": False,
        },
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "layer_id": args.layer_id,
                "frame_ids": frame_ids,
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
