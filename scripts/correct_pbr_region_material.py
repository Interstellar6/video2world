#!/usr/bin/env python3
"""Replace one geometry-selected PBR region without changing object geometry.

Run this script with Blender, for example::

    blender --background --python scripts/correct_pbr_region_material.py -- \
      --source-glb source.glb \
      --mask-manifest targeted-potted-plant-mask-manifest.json \
      --object-id sam3_plant_02 \
      --semantic-up-axis +Y \
      --output-glb corrected.glb \
      --receipt receipt.json

The lower-container selector operates on complete connected components.  A
component is selected only when every vertex stays below the configured height
cut and within the configured radial cylinder.  No UV painting or texture
editing is performed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import struct
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

GLTF_AXIS_TO_BLENDER = {
    "+X": np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
    "-X": np.asarray([-1.0, 0.0, 0.0], dtype=np.float64),
    "+Y": np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
    "-Y": np.asarray([0.0, 0.0, -1.0], dtype=np.float64),
    "+Z": np.asarray([0.0, -1.0, 0.0], dtype=np.float64),
    "-Z": np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


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


def srgb_to_linear(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.where(
        values <= 0.0031308,
        values * 12.92,
        1.055 * np.power(values, 1.0 / 2.4) - 0.055,
    )


def derive_neutral_metal_base_color(
    masked_rgb_srgb: np.ndarray,
    *,
    minimum_luminance_srgb: float = 0.06,
    maximum_luminance_srgb: float = 0.18,
) -> dict[str, Any]:
    """Derive a neutral dark-metal factor from robust real-mask statistics."""
    rgb = np.asarray(masked_rgb_srgb, dtype=np.float64)
    require(rgb.ndim == 2 and rgb.shape[1] == 3, "masked RGB samples must have shape Nx3")
    require(
        len(rgb) >= 64 and np.isfinite(rgb).all(),
        "at least 64 finite RGB samples are required",
    )
    require(np.all((rgb >= 0.0) & (rgb <= 1.0)), "RGB samples must be normalized sRGB")
    luminance = np.einsum("ni,i->n", rgb, [0.2126, 0.7152, 0.0722], optimize=False)
    observed_median = float(np.quantile(luminance, 0.5))
    target_srgb = float(np.clip(observed_median, minimum_luminance_srgb, maximum_luminance_srgb))
    target_linear = float(srgb_to_linear(np.asarray([target_srgb]))[0])
    return {
        "observed_rgb_srgb_quantiles": {
            str(quantile): np.quantile(rgb, quantile, axis=0).tolist()
            for quantile in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
        "observed_luminance_srgb_quantiles": {
            str(quantile): float(np.quantile(luminance, quantile))
            for quantile in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
        "neutralization": (
            "equal RGB channels; scene illuminant chroma is not baked into metal albedo"
        ),
        "target_base_color_srgb": [target_srgb, target_srgb, target_srgb, 1.0],
        "target_base_color_linear": [target_linear, target_linear, target_linear, 1.0],
        "clamp_srgb": [minimum_luminance_srgb, maximum_luminance_srgb],
    }


def component_passes_lower_cylinder(
    component: dict[str, float],
    *,
    maximum_height: float,
    maximum_radius: float,
) -> bool:
    return (
        component["maximum_height"] <= maximum_height
        and component["maximum_radius"] <= maximum_radius
    )


def _erode_binary(mask: np.ndarray) -> np.ndarray:
    eroded = np.asarray(mask, dtype=bool).copy()
    padded = np.pad(eroded, 1, mode="constant", constant_values=False)
    for offset_y in range(3):
        for offset_x in range(3):
            eroded &= padded[
                offset_y : offset_y + mask.shape[0],
                offset_x : offset_x + mask.shape[1],
            ]
    return eroded


def _image_pixels(image: Any) -> np.ndarray:
    width, height = image.size[:]
    values = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(values)
    return values.reshape(height, width, 4)


def _masked_color_statistics(
    bpy: Any,
    *,
    manifest_path: Path,
    object_id: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    objects = manifest.get("objects")
    require(isinstance(objects, list), "mask manifest has no objects list")
    records = [item for item in objects if item.get("object_id") == object_id]
    require(len(records) == 1, f"mask manifest must contain exactly one {object_id!r}")
    frames = records[0].get("frames")
    require(isinstance(frames, list) and frames, "target object has no mask frames")

    all_samples: list[np.ndarray] = []
    frame_reports: list[dict[str, Any]] = []
    for frame in frames:
        frame_id = str(frame.get("frame_id"))
        source_record = frame.get("source_rgb")
        mask_record = frame.get("selected_container_output_mask")
        require(isinstance(source_record, dict), f"{frame_id} has no source_rgb")
        require(isinstance(mask_record, dict), f"{frame_id} has no selected container mask")
        source_path = resolve_path(str(source_record.get("path")), relative_to=manifest_path.parent)
        mask_path = resolve_path(str(mask_record.get("path")), relative_to=manifest_path.parent)
        require(source_path.is_file(), f"source RGB is missing: {source_path}")
        require(mask_path.is_file(), f"pot mask is missing: {mask_path}")
        expected_source_hash = source_record.get("sha256")
        expected_mask_hash = mask_record.get("sha256")
        observed_source_hash = sha256_file(source_path)
        observed_mask_hash = sha256_file(mask_path)
        require(
            not isinstance(expected_source_hash, str)
            or expected_source_hash == observed_source_hash,
            f"source RGB hash mismatch: {source_path}",
        )
        require(
            not isinstance(expected_mask_hash, str) or expected_mask_hash == observed_mask_hash,
            f"pot mask hash mismatch: {mask_path}",
        )

        source = bpy.data.images.load(str(source_path), check_existing=False)
        mask_image = bpy.data.images.load(str(mask_path), check_existing=False)
        try:
            source_pixels = _image_pixels(source)
            mask_pixels = _image_pixels(mask_image)
        finally:
            bpy.data.images.remove(source)
            bpy.data.images.remove(mask_image)
        require(
            source_pixels.shape[:2] == mask_pixels.shape[:2],
            f"source/mask dimensions differ for {frame_id}",
        )
        mask = mask_pixels[:, :, 0] > 0.5
        eroded = _erode_binary(mask)
        sample_mask = eroded if int(eroded.sum()) >= 64 else mask
        samples = source_pixels[:, :, :3][sample_mask].astype(np.float64)
        require(len(samples) >= 64, f"too few pot pixels in {frame_id}")
        all_samples.append(samples)
        luma = np.einsum("ni,i->n", samples, [0.2126, 0.7152, 0.0722], optimize=False)
        frame_reports.append(
            {
                "frame_id": frame_id,
                "source_rgb": {
                    "path": str(source_path),
                    "sha256": observed_source_hash,
                },
                "pot_mask": {
                    "path": str(mask_path),
                    "sha256": observed_mask_hash,
                    "pixel_count": int(mask.sum()),
                    "eroded_pixel_count": int(eroded.sum()),
                },
                "sample_count": len(samples),
                "median_rgb_srgb": np.quantile(samples, 0.5, axis=0).tolist(),
                "median_luminance_srgb": float(np.quantile(luma, 0.5)),
            }
        )

    combined = np.concatenate(all_samples, axis=0)
    return combined, {
        "method": (
            "one-pixel 3x3 binary erosion, then aggregate real source RGB under "
            "hash-bound pot masks"
        ),
        "frame_count": len(frame_reports),
        "sample_count": len(combined),
        "frames": frame_reports,
    }


def _reset_scene(bpy: Any) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _import_glb(bpy: Any, path: Path) -> list[Any]:
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    imported = [item for item in bpy.context.scene.objects if item not in before]
    require(imported, f"GLB import created no objects: {path}")
    require(any(item.type == "MESH" for item in imported), f"GLB contains no mesh: {path}")
    return imported


def _mesh_world_coordinates(mesh_object: Any) -> np.ndarray:
    mesh = mesh_object.data
    local = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", local)
    local = local.reshape(-1, 3)
    matrix = np.asarray(mesh_object.matrix_world, dtype=np.float64)
    homogeneous = np.column_stack((local, np.ones(len(local), dtype=np.float64)))
    return np.einsum("ij,nj->ni", matrix, homogeneous, optimize=False)[:, :3]


def _pixel_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.float32).tobytes()).hexdigest()


def _principled_and_base_image(
    mesh_objects: list[Any],
    *,
    require_untextured_metallic_roughness: bool = True,
) -> tuple[Any, Any, Any]:
    materials = {
        material
        for mesh_object in mesh_objects
        for material in mesh_object.data.materials
        if material is not None
    }
    require(len(materials) == 1, "texture-region correction requires exactly one source material")
    material = next(iter(materials))
    require(material.use_nodes, "source material must use nodes")
    principled_nodes = [node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"]
    require(len(principled_nodes) == 1, "source material must have one Principled BSDF")
    principled = principled_nodes[0]
    base_input = principled.inputs["Base Color"]
    require(len(base_input.links) == 1, "source base color must have one texture link")
    image_node = base_input.links[0].from_node
    require(
        image_node.type == "TEX_IMAGE" and image_node.image is not None,
        "source base color link must come directly from an image texture",
    )
    if require_untextured_metallic_roughness:
        require(
            len(principled.inputs["Metallic"].links) == 0,
            "source metallic input is already textured",
        )
        require(
            len(principled.inputs["Roughness"].links) == 0,
            "source roughness input is already textured",
        )
    base_factor = np.asarray(base_input.default_value[:], dtype=np.float64)
    require(
        np.all((base_factor[:3] > 0.0) & (base_factor[:3] <= 1.0)) and 0.0 < base_factor[3] <= 1.0,
        "source base color factor must be positive and normalized",
    )
    return material, principled, image_node


def _glb_json(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        header = stream.read(12)
        require(len(header) == 12, f"GLB header is truncated: {path}")
        magic, version, total_length = struct.unpack("<4sII", header)
        require(magic == b"glTF" and version == 2, f"not a glTF 2 GLB: {path}")
        require(total_length == path.stat().st_size, f"GLB length header mismatch: {path}")
        while stream.tell() < total_length:
            chunk_header = stream.read(8)
            require(len(chunk_header) == 8, f"GLB chunk header is truncated: {path}")
            chunk_length, chunk_type = struct.unpack("<II", chunk_header)
            chunk = stream.read(chunk_length)
            require(len(chunk) == chunk_length, f"GLB chunk is truncated: {path}")
            if chunk_type == 0x4E4F534A:
                return json.loads(chunk.rstrip(b" \t\r\n\x00").decode("utf-8"))
    raise ValueError(f"GLB has no JSON chunk: {path}")


def _glb_payload(path: Path) -> tuple[dict[str, Any], bytes]:
    json_payload: dict[str, Any] | None = None
    binary_payload: bytes | None = None
    with path.open("rb") as stream:
        magic, version, total_length = struct.unpack("<4sII", stream.read(12))
        require(magic == b"glTF" and version == 2, f"not a glTF 2 GLB: {path}")
        require(total_length == path.stat().st_size, f"GLB length header mismatch: {path}")
        while stream.tell() < total_length:
            chunk_length, chunk_type = struct.unpack("<II", stream.read(8))
            chunk = stream.read(chunk_length)
            require(len(chunk) == chunk_length, f"GLB chunk is truncated: {path}")
            if chunk_type == 0x4E4F534A:
                json_payload = json.loads(chunk.rstrip(b" \t\r\n\x00").decode("utf-8"))
            elif chunk_type == 0x004E4942:
                binary_payload = chunk
    require(json_payload is not None, f"GLB has no JSON chunk: {path}")
    require(binary_payload is not None, f"GLB has no BIN chunk: {path}")
    return json_payload, binary_payload


def _write_glb(path: Path, payload: dict[str, Any], binary: bytes) -> None:
    json_bytes = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * ((-len(json_bytes)) % 4)
    binary += b"\x00" * ((-len(binary)) % 4)
    total_length = 12 + 8 + len(json_bytes) + 8 + len(binary)
    output = bytearray()
    output.extend(struct.pack("<4sII", b"glTF", 2, total_length))
    output.extend(struct.pack("<II", len(json_bytes), 0x4E4F534A))
    output.extend(json_bytes)
    output.extend(struct.pack("<II", len(binary), 0x004E4942))
    output.extend(binary)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(output)
    temporary.replace(path)


def _geometry_contract_sha256(payload: dict[str, Any]) -> str:
    accessors = payload.get("accessors", [])
    used_buffer_views = sorted(
        {
            accessor["bufferView"]
            for accessor in accessors
            if isinstance(accessor, dict) and isinstance(accessor.get("bufferView"), int)
        }
    )
    contract = {
        "accessors": accessors,
        "geometry_buffer_views": [
            payload.get("bufferViews", [])[index] for index in used_buffer_views
        ],
        "meshes": payload.get("meshes", []),
        "nodes": payload.get("nodes", []),
        "scene": payload.get("scene"),
        "scenes": payload.get("scenes", []),
    }
    encoded = json.dumps(contract, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _patch_source_geometry_with_donor_material(
    *,
    source_path: Path,
    donor_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    source_json, source_binary = _glb_payload(source_path)
    donor_json, donor_binary = _glb_payload(donor_path)
    donor_images = donor_json.get("images", [])
    donor_textures = donor_json.get("textures", [])
    donor_materials = donor_json.get("materials", [])
    require(donor_images and donor_textures and donor_materials, "material donor has no PBR images")

    output_json = copy.deepcopy(source_json)
    output_binary = bytearray(source_binary)
    output_views = output_json.setdefault("bufferViews", [])
    output_images: list[dict[str, Any]] = []
    for image in donor_images:
        source_view_index = image.get("bufferView")
        require(isinstance(source_view_index, int), "material donor image is not embedded")
        source_view = donor_json["bufferViews"][source_view_index]
        require(source_view.get("buffer", 0) == 0, "material donor image uses another buffer")
        start = int(source_view.get("byteOffset", 0))
        end = start + int(source_view["byteLength"])
        image_bytes = donor_binary[start:end]
        require(len(image_bytes) == source_view["byteLength"], "material donor image is truncated")
        output_binary.extend(b"\x00" * ((-len(output_binary)) % 4))
        output_view_index = len(output_views)
        output_views.append(
            {
                "buffer": 0,
                "byteOffset": len(output_binary),
                "byteLength": len(image_bytes),
            }
        )
        output_binary.extend(image_bytes)
        output_image = {
            key: copy.deepcopy(value)
            for key, value in image.items()
            if key not in {"bufferView", "uri"}
        }
        output_image["bufferView"] = output_view_index
        output_images.append(output_image)

    output_json["images"] = output_images
    output_json["textures"] = copy.deepcopy(donor_textures)
    output_json["materials"] = copy.deepcopy(donor_materials)
    if donor_json.get("samplers"):
        output_json["samplers"] = copy.deepcopy(donor_json["samplers"])
    else:
        output_json.pop("samplers", None)
    require(len(output_json.get("buffers", [])) == 1, "source GLB must have one binary buffer")
    output_json["buffers"][0]["byteLength"] = len(output_binary)
    source_contract = _geometry_contract_sha256(source_json)
    output_contract = _geometry_contract_sha256(output_json)
    require(
        source_contract == output_contract, "material patch changed the source geometry contract"
    )
    _write_glb(output_path, output_json, bytes(output_binary))
    return {
        "method": "source GLB geometry chunks plus Blender-exported PBR donor images/material",
        "source_geometry_contract_sha256": source_contract,
        "output_geometry_contract_sha256": output_contract,
        "geometry_contract_unchanged": True,
        "donor_image_count": len(output_images),
        "source_binary_prefix_size_bytes": len(source_binary),
        "output_binary_size_bytes": len(output_binary),
    }


def _glb_structure(path: Path) -> dict[str, Any]:
    payload = _glb_json(path)
    primitives = [
        primitive for mesh in payload.get("meshes", []) for primitive in mesh.get("primitives", [])
    ]
    materials = payload.get("materials", [])
    return {
        "mesh_count": len(payload.get("meshes", [])),
        "primitive_count": len(primitives),
        "material_count": len(materials),
        "base_color_texture_count": sum(
            "baseColorTexture" in material.get("pbrMetallicRoughness", {}) for material in materials
        ),
        "metallic_roughness_texture_count": sum(
            "metallicRoughnessTexture" in material.get("pbrMetallicRoughness", {})
            for material in materials
        ),
        "double_sided_material_count": sum(
            material.get("doubleSided") is True for material in materials
        ),
        "alpha_modes": [material.get("alphaMode", "OPAQUE") for material in materials],
    }


def _mesh_quality(mesh_object: Any) -> dict[str, Any]:
    mesh = mesh_object.data
    require(len(mesh.uv_layers) >= 1, f"mesh {mesh_object.name} has no UV layer")
    uv = np.empty(len(mesh.loops) * 2, dtype=np.float64)
    mesh.uv_layers.active.data.foreach_get("uv", uv)
    normals = np.empty(len(mesh.corner_normals) * 3, dtype=np.float64)
    mesh.corner_normals.foreach_get("vector", normals)
    normals = normals.reshape(-1, 3)
    normal_lengths = np.linalg.norm(normals, axis=1)
    return {
        "object_name": mesh_object.name,
        "vertex_count": len(mesh.vertices),
        "face_count": len(mesh.polygons),
        "loop_count": len(mesh.loops),
        "uv_layer_count": len(mesh.uv_layers),
        "uv_finite": bool(np.isfinite(uv).all()),
        "corner_normal_count": len(mesh.corner_normals),
        "normals_finite": bool(np.isfinite(normals).all()),
        "normal_length_range": [float(normal_lengths.min()), float(normal_lengths.max())],
        "material_names": [material.name if material else None for material in mesh.materials],
    }


def _canonical_triangle_hash(mesh_objects: list[Any], *, decimals: int = 7) -> str:
    rows: list[np.ndarray] = []
    for mesh_object in mesh_objects:
        mesh = mesh_object.data
        require(
            all(len(polygon.vertices) == 3 for polygon in mesh.polygons),
            f"mesh {mesh_object.name} is not fully triangulated",
        )
        coordinates = _mesh_world_coordinates(mesh_object)
        faces = np.empty((len(mesh.polygons), 3), dtype=np.int32)
        for index, polygon in enumerate(mesh.polygons):
            faces[index] = polygon.vertices[:]
        triangles = np.round(coordinates[faces], decimals=decimals)
        vertex_order = np.lexsort(
            (triangles[:, :, 2], triangles[:, :, 1], triangles[:, :, 0]),
            axis=1,
        )
        triangles = np.take_along_axis(triangles, vertex_order[:, :, None], axis=1)
        rows.append(triangles.reshape(-1, 9))
    flattened = np.concatenate(rows, axis=0)
    order = np.lexsort(tuple(flattened[:, index] for index in reversed(range(9))))
    digest = hashlib.sha256()
    digest.update(np.asarray(flattened[order], dtype=np.float64).tobytes())
    return digest.hexdigest()


def _geometry_snapshot(mesh_objects: list[Any]) -> dict[str, Any]:
    coordinates = np.concatenate([_mesh_world_coordinates(item) for item in mesh_objects], axis=0)
    fingerprint = hashlib.sha256()
    for mesh_object in mesh_objects:
        mesh = mesh_object.data
        local = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", local)
        fingerprint.update(mesh_object.name.encode("utf-8"))
        fingerprint.update(np.asarray(mesh_object.matrix_world, dtype=np.float64).tobytes())
        fingerprint.update(local.tobytes())
        for polygon in mesh.polygons:
            fingerprint.update(struct.pack("<I", len(polygon.vertices)))
            fingerprint.update(np.asarray(polygon.vertices[:], dtype=np.uint32).tobytes())
    return {
        "mesh_count": len(mesh_objects),
        "vertex_count": sum(len(item.data.vertices) for item in mesh_objects),
        "unique_world_vertex_count_at_1e7": len(
            np.unique(np.round(coordinates, decimals=7), axis=0)
        ),
        "face_count": sum(len(item.data.polygons) for item in mesh_objects),
        "world_bounds_min": coordinates.min(axis=0).tolist(),
        "world_bounds_max": coordinates.max(axis=0).tolist(),
        "object_local_topology_sha256": fingerprint.hexdigest(),
        "canonical_world_triangles_rounded_1e7_sha256": _canonical_triangle_hash(mesh_objects),
    }


def _component_roots(mesh: Any) -> np.ndarray:
    parent = np.arange(len(mesh.vertices), dtype=np.int32)
    rank = np.zeros(len(mesh.vertices), dtype=np.uint8)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        if rank[first_root] < rank[second_root]:
            parent[first_root] = second_root
        elif rank[first_root] > rank[second_root]:
            parent[second_root] = first_root
        else:
            parent[second_root] = first_root
            rank[first_root] += 1

    for edge in mesh.edges:
        union(int(edge.vertices[0]), int(edge.vertices[1]))
    for index in range(len(parent)):
        parent[index] = find(index)
    return parent


def _semantic_coordinates(
    world_coordinates: np.ndarray,
    *,
    semantic_up_axis: str,
) -> np.ndarray:
    up = GLTF_AXIS_TO_BLENDER[semantic_up_axis]
    up_index = int(np.argmax(np.abs(up)))
    horizontal_indices = [index for index in range(3) if index != up_index]
    height = world_coordinates[:, up_index] * float(up[up_index])
    return np.column_stack(
        (
            world_coordinates[:, horizontal_indices[0]],
            world_coordinates[:, horizontal_indices[1]],
            height,
        )
    )


def _select_lower_container_region(
    mesh_objects: list[Any],
    *,
    semantic_up_axis: str,
    top_height_ratio: float,
    radius_ratio: float,
) -> tuple[dict[str, Any], dict[str, list[int]]]:
    world = np.concatenate([_mesh_world_coordinates(item) for item in mesh_objects], axis=0)
    semantic = _semantic_coordinates(world, semantic_up_axis=semantic_up_axis)
    lower = semantic.min(axis=0)
    upper = semantic.max(axis=0)
    extents = upper - lower
    require(np.all(extents > 1e-8), "asset semantic bounds are degenerate")
    center_horizontal = (lower[:2] + upper[:2]) * 0.5
    maximum_height = float(lower[2] + top_height_ratio * extents[2])
    maximum_radius = float(max(extents[:2]) * 0.5 * radius_ratio)

    component_count = 0
    selected_component_count = 0
    selected_face_count = 0
    selected_vertex_coordinates: list[np.ndarray] = []
    mesh_reports: list[dict[str, Any]] = []
    selected_polygons: dict[str, list[int]] = {}
    for mesh_object in mesh_objects:
        mesh = mesh_object.data
        coordinates = _semantic_coordinates(
            _mesh_world_coordinates(mesh_object),
            semantic_up_axis=semantic_up_axis,
        )
        roots = _component_roots(mesh)
        order = np.argsort(roots, kind="stable")
        sorted_roots = roots[order]
        boundaries = np.flatnonzero(np.diff(sorted_roots)) + 1
        groups = np.split(order, boundaries)
        selected_roots: set[int] = set()
        component_records: list[dict[str, Any]] = []
        for vertex_ids in groups:
            values = coordinates[vertex_ids]
            radii = np.linalg.norm(values[:, :2] - center_horizontal, axis=1)
            record = {
                "root": int(roots[vertex_ids[0]]),
                "vertex_count": len(vertex_ids),
                "minimum_height": float(values[:, 2].min()),
                "maximum_height": float(values[:, 2].max()),
                "maximum_radius": float(radii.max()),
            }
            selected = component_passes_lower_cylinder(
                record,
                maximum_height=maximum_height,
                maximum_radius=maximum_radius,
            )
            record["selected"] = selected
            component_records.append(record)
            if selected:
                selected_roots.add(record["root"])
                selected_vertex_coordinates.append(values)

        mesh_selected_polygon_indices: list[int] = []
        for polygon in mesh.polygons:
            if int(roots[polygon.vertices[0]]) in selected_roots:
                mesh_selected_polygon_indices.append(polygon.index)
        selected_polygons[mesh_object.name] = mesh_selected_polygon_indices
        mesh_selected_faces = len(mesh_selected_polygon_indices)
        component_count += len(component_records)
        selected_component_count += len(selected_roots)
        selected_face_count += mesh_selected_faces
        mesh_reports.append(
            {
                "object_name": mesh_object.name,
                "component_count": len(component_records),
                "selected_component_count": len(selected_roots),
                "selected_face_count": mesh_selected_faces,
            }
        )

    total_faces = sum(len(item.data.polygons) for item in mesh_objects)
    require(selected_face_count > 0, "lower-container selector chose no faces")
    selected_coordinates = np.concatenate(selected_vertex_coordinates, axis=0)
    report = {
        "kind": "connected_components_inside_lower_semantic_cylinder",
        "semantic_up_axis_gltf": semantic_up_axis,
        "semantic_up_vector_blender_world": GLTF_AXIS_TO_BLENDER[semantic_up_axis].tolist(),
        "parameters": {
            "top_height_ratio": top_height_ratio,
            "radius_ratio_of_max_horizontal_half_extent": radius_ratio,
            "connectivity": "shared_mesh_edges",
            "component_rule": "all vertices must remain at or below top and inside radius",
        },
        "derived_thresholds": {
            "semantic_bounds_min": lower.tolist(),
            "semantic_bounds_max": upper.tolist(),
            "horizontal_center": center_horizontal.tolist(),
            "maximum_height": maximum_height,
            "maximum_radius": maximum_radius,
        },
        "component_count": component_count,
        "selected_component_count": selected_component_count,
        "selected_face_count": selected_face_count,
        "total_face_count": total_faces,
        "selected_face_fraction": selected_face_count / total_faces,
        "selected_semantic_bounds_min": selected_coordinates.min(axis=0).tolist(),
        "selected_semantic_bounds_max": selected_coordinates.max(axis=0).tolist(),
        "meshes": mesh_reports,
    }
    return report, selected_polygons


def _rasterize_triangle(mask: np.ndarray, points: np.ndarray) -> None:
    height, width = mask.shape
    pixel = np.asarray(points, dtype=np.float64) * [width - 1, height - 1]
    minimum = np.maximum(np.floor(pixel.min(axis=0)).astype(int), 0)
    maximum = np.minimum(np.ceil(pixel.max(axis=0)).astype(int), [width - 1, height - 1])
    if np.any(maximum < minimum):
        return
    x_values = np.arange(minimum[0], maximum[0] + 1, dtype=np.float64) + 0.5
    y_values = np.arange(minimum[1], maximum[1] + 1, dtype=np.float64) + 0.5
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    first, second, third = pixel
    denominator = (second[1] - third[1]) * (first[0] - third[0]) + (third[0] - second[0]) * (
        first[1] - third[1]
    )
    if abs(denominator) > 1e-12:
        first_weight = (
            (second[1] - third[1]) * (x_grid - third[0])
            + (third[0] - second[0]) * (y_grid - third[1])
        ) / denominator
        second_weight = (
            (third[1] - first[1]) * (x_grid - third[0])
            + (first[0] - third[0]) * (y_grid - third[1])
        ) / denominator
        third_weight = 1.0 - first_weight - second_weight
        inside = (first_weight >= -1e-6) & (second_weight >= -1e-6) & (third_weight >= -1e-6)
        mask[
            minimum[1] : maximum[1] + 1,
            minimum[0] : maximum[0] + 1,
        ] |= inside
    rounded = np.rint(pixel).astype(int)
    rounded[:, 0] = np.clip(rounded[:, 0], 0, width - 1)
    rounded[:, 1] = np.clip(rounded[:, 1], 0, height - 1)
    mask[rounded[:, 1], rounded[:, 0]] = True


def _dilate_binary(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    dilated = np.zeros_like(mask, dtype=bool)
    for offset_y in range(3):
        for offset_x in range(3):
            dilated |= padded[
                offset_y : offset_y + mask.shape[0],
                offset_x : offset_x + mask.shape[1],
            ]
    return dilated


def _selected_uv_mask(
    mesh_objects: list[Any],
    selected_polygons: dict[str, list[int]],
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    mask = np.zeros((height, width), dtype=bool)
    selected_face_count = 0
    selected_uv_min = np.asarray([math.inf, math.inf], dtype=np.float64)
    selected_uv_max = np.asarray([-math.inf, -math.inf], dtype=np.float64)
    for mesh_object in mesh_objects:
        mesh = mesh_object.data
        uv_data = mesh.uv_layers.active.data
        for polygon_index in selected_polygons[mesh_object.name]:
            polygon = mesh.polygons[polygon_index]
            require(len(polygon.loop_indices) == 3, "selected face is not triangulated")
            uv = np.asarray([uv_data[index].uv[:] for index in polygon.loop_indices])
            require(
                np.all((uv >= -1e-6) & (uv <= 1.0 + 1e-6)),
                "selected pot UV lies outside the base texture tile",
            )
            selected_uv_min = np.minimum(selected_uv_min, uv.min(axis=0))
            selected_uv_max = np.maximum(selected_uv_max, uv.max(axis=0))
            _rasterize_triangle(mask, np.clip(uv, 0.0, 1.0))
            selected_face_count += 1
    require(selected_face_count > 0 and mask.any(), "selected faces produced no UV texels")
    undilated_mask = mask
    dilated_mask = _dilate_binary(undilated_mask)

    undilated_overlap_sample_count = 0
    dilated_overlap_sample_count = 0
    unselected_sample_count = 0
    conflict_texels = np.zeros_like(undilated_mask, dtype=bool)
    for mesh_object in mesh_objects:
        mesh = mesh_object.data
        selected = np.zeros(len(mesh.polygons), dtype=bool)
        selected[selected_polygons[mesh_object.name]] = True
        uv_data = mesh.uv_layers.active.data
        require(
            all(
                len(polygon.loop_indices) == 3 and polygon.loop_start == polygon.index * 3
                for polygon in mesh.polygons
            ),
            "vectorized UV overlap audit requires contiguous triangles",
        )
        all_uv = np.empty(len(mesh.loops) * 2, dtype=np.float64)
        uv_data.foreach_get("uv", all_uv)
        all_uv = all_uv.reshape(-1, 3, 2)
        unselected_uv = all_uv[~selected]
        for start in range(0, len(unselected_uv), 50_000):
            chunk = unselected_uv[start : start + 50_000]
            samples = np.concatenate((chunk, chunk.mean(axis=1, keepdims=True)), axis=1).reshape(
                -1, 2
            )
            x = np.clip(np.rint(samples[:, 0] * (width - 1)).astype(int), 0, width - 1)
            y = np.clip(np.rint(samples[:, 1] * (height - 1)).astype(int), 0, height - 1)
            undilated_overlap_sample_count += int(undilated_mask[y, x].sum())
            dilated_overlap_sample_count += int(dilated_mask[y, x].sum())
            conflict = undilated_mask[y, x]
            conflict_texels[y[conflict], x[conflict]] = True
            unselected_sample_count += len(samples)
    conflict_exclusion = np.zeros_like(undilated_mask, dtype=bool)
    if dilated_overlap_sample_count == 0:
        mask = dilated_mask
        overlap_sample_count = 0
        selection_mode = "one_texel_dilation"
    elif undilated_overlap_sample_count == 0:
        mask = undilated_mask
        overlap_sample_count = 0
        selection_mode = "undilated"
    else:
        conflict_exclusion = _dilate_binary(conflict_texels)
        mask = undilated_mask & ~conflict_exclusion
        overlap_sample_count = 0
        selection_mode = "undilated_minus_one_texel_buffer_around_unselected_uv_samples"
    require(mask.any(), "UV conflict exclusion removed the complete selected region")
    require(
        overlap_sample_count == 0,
        (
            "selected pot UV texels overlap sampled unselected geometry; refusing texture edit: "
            f"{overlap_sample_count}/{unselected_sample_count} sampled UV positions"
        ),
    )
    return mask, {
        "method": "rasterized selected triangle UVs; seam dilation only when overlap audit passes",
        "one_texel_seam_dilation_applied": dilated_overlap_sample_count == 0,
        "selection_mode": selection_mode,
        "texture_dimensions_wh": [width, height],
        "selected_face_count": selected_face_count,
        "selected_texel_count": int(mask.sum()),
        "selected_texel_fraction": float(mask.mean()),
        "selected_uv_bounds_min": selected_uv_min.tolist(),
        "selected_uv_bounds_max": selected_uv_max.tolist(),
        "unselected_vertex_and_centroid_sample_count": unselected_sample_count,
        "unselected_overlap_sample_count": overlap_sample_count,
        "undilated_overlap_sample_count": undilated_overlap_sample_count,
        "dilated_overlap_sample_count": dilated_overlap_sample_count,
        "conflict_texel_count": int(conflict_texels.sum()),
        "conflict_exclusion_texel_count": int(conflict_exclusion.sum()),
    }


def _new_image_from_pixels(
    bpy: Any,
    *,
    name: str,
    pixels: np.ndarray,
    colorspace: str,
) -> Any:
    height, width, channels = pixels.shape
    require(channels == 4, "generated texture must be RGBA")
    image = bpy.data.images.new(name=name, width=width, height=height, alpha=True)
    image.colorspace_settings.name = colorspace
    image.pixels.foreach_set(np.asarray(pixels, dtype=np.float32).reshape(-1))
    image.update()
    image.pack()
    return image


def _apply_pbr_texture_region(
    bpy: Any,
    *,
    mesh_objects: list[Any],
    selected_polygons: dict[str, list[int]],
    target_base_srgb: list[float],
    metallic: float,
    roughness: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    material, principled, base_node = _principled_and_base_image(mesh_objects)
    base_image = base_node.image
    source_pixels = _image_pixels(base_image).astype(np.float32)
    height, width = source_pixels.shape[:2]
    uv_mask, uv_report = _selected_uv_mask(
        mesh_objects,
        selected_polygons,
        width=width,
        height=height,
    )
    base_input_default = np.asarray(
        principled.inputs["Base Color"].default_value[:], dtype=np.float64
    )
    target_texture_srgb = np.asarray(target_base_srgb[:3], dtype=np.float32)
    selected_alpha_before = source_pixels[uv_mask, 3].astype(np.float64)
    corrected_pixels = source_pixels.copy()
    corrected_pixels[uv_mask, :3] = target_texture_srgb
    corrected_pixels[uv_mask, 3] = 1.0
    corrected_base_image = _new_image_from_pixels(
        bpy,
        name=f"{base_image.name}_PotCorrected",
        pixels=corrected_pixels,
        colorspace=base_image.colorspace_settings.name,
    )
    base_node.image = corrected_base_image

    source_metallic = float(principled.inputs["Metallic"].default_value)
    source_roughness = float(principled.inputs["Roughness"].default_value)
    metallic_pixels = np.full((height, width, 4), source_metallic, dtype=np.float32)
    roughness_pixels = np.full((height, width, 4), source_roughness, dtype=np.float32)
    metallic_pixels[:, :, 3] = 1.0
    roughness_pixels[:, :, 3] = 1.0
    metallic_pixels[uv_mask, :3] = metallic
    roughness_pixels[uv_mask, :3] = roughness
    metallic_image = _new_image_from_pixels(
        bpy,
        name="PotMetallic_From_Geometry",
        pixels=metallic_pixels,
        colorspace="Non-Color",
    )
    roughness_image = _new_image_from_pixels(
        bpy,
        name="PotRoughness_From_Geometry",
        pixels=roughness_pixels,
        colorspace="Non-Color",
    )
    metallic_node = material.node_tree.nodes.new("ShaderNodeTexImage")
    metallic_node.name = "Pot Metallic Map From Geometry"
    metallic_node.image = metallic_image
    metallic_node.interpolation = base_node.interpolation
    metallic_node.extension = base_node.extension
    roughness_node = material.node_tree.nodes.new("ShaderNodeTexImage")
    roughness_node.name = "Pot Roughness Map From Geometry"
    roughness_node.image = roughness_image
    roughness_node.interpolation = base_node.interpolation
    roughness_node.extension = base_node.extension
    material.node_tree.links.new(metallic_node.outputs["Color"], principled.inputs["Metallic"])
    material.node_tree.links.new(roughness_node.outputs["Color"], principled.inputs["Roughness"])
    source_backface_culling = bool(material.use_backface_culling)
    material.use_backface_culling = False

    outside = ~uv_mask
    return (
        {
            "material_name": material.name,
            "source_base_texture": {
                "image_name": base_image.name,
                "dimensions_wh": [width, height],
                "full_pixel_float32_sha256": _pixel_sha256(source_pixels),
                "outside_selected_uv_pixel_float32_sha256": _pixel_sha256(source_pixels[outside]),
            },
            "corrected_base_texture": {
                "image_name": corrected_base_image.name,
                "full_pixel_float32_sha256": _pixel_sha256(corrected_pixels),
                "outside_selected_uv_pixel_float32_sha256": _pixel_sha256(
                    corrected_pixels[outside]
                ),
            },
            "source_metallic_factor": source_metallic,
            "source_roughness_factor": source_roughness,
            "target_metallic": metallic,
            "target_roughness": roughness,
            "base_color_link_replaces_unlinked_default": base_input_default.tolist(),
            "target_texture_rgb_srgb": target_texture_srgb.tolist(),
            "selected_alpha_before_quantiles": {
                str(quantile): float(np.quantile(selected_alpha_before, quantile))
                for quantile in (0.0, 0.05, 0.5, 0.95, 1.0)
            },
            "selected_alpha_after": 1.0,
            "source_use_backface_culling": source_backface_culling,
            "corrected_use_backface_culling": False,
            "glb_double_sided_requested": True,
            "uv_projection": uv_report,
        },
        source_pixels,
        uv_mask,
    )


def _validate_mesh_quality(reports: list[dict[str, Any]]) -> dict[str, bool]:
    return {
        "all_meshes_have_uv": all(item["uv_layer_count"] >= 1 for item in reports),
        "all_uv_values_finite": all(item["uv_finite"] for item in reports),
        "all_corner_normals_present": all(
            item["corner_normal_count"] == item["loop_count"] for item in reports
        ),
        "all_normals_finite": all(item["normals_finite"] for item in reports),
        "all_normal_lengths_unit": all(
            item["normal_length_range"][0] >= 0.99 and item["normal_length_range"][1] <= 1.01
            for item in reports
        ),
    }


def _add_area_light(bpy: Any, *, name: str, location: Any, center: Any, energy: float) -> None:
    from mathutils import Vector

    light_data = bpy.data.lights.new(name=name, type="AREA")
    light_data.energy = energy
    light_data.shape = "DISK"
    light_data.size = 4.0
    light = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(light)
    light.location = Vector(location)
    light.rotation_euler = (Vector(center) - light.location).to_track_quat("-Z", "Y").to_euler()


def _pbr_bottom_coverage_qa(
    bpy: Any,
    *,
    mesh_objects: list[Any],
    semantic_up_axis: str,
    output_directory: Path,
) -> dict[str, Any]:
    from mathutils import Matrix, Vector

    world = np.concatenate([_mesh_world_coordinates(item) for item in mesh_objects], axis=0)
    lower = world.min(axis=0)
    upper = world.max(axis=0)
    center = (lower + upper) * 0.5
    extents = upper - lower
    up = GLTF_AXIS_TO_BLENDER[semantic_up_axis]
    up_index = int(np.argmax(np.abs(up)))
    horizontal_indices = [index for index in range(3) if index != up_index]
    view_index = max(horizontal_indices, key=lambda index: extents[index])
    position_axis = np.zeros(3, dtype=np.float64)
    position_axis[view_index] = 1.0
    screen_x = np.cross(up, position_axis)
    screen_x /= np.linalg.norm(screen_x)

    camera_data = bpy.data.cameras.new("PotBottomCoverageCamera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = float(max(extents[up_index], np.dot(np.abs(screen_x), extents)) * 1.2)
    camera = bpy.data.objects.new("PotBottomCoverageCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    z_axis = Vector(position_axis.tolist()).normalized()
    y_axis = Vector(up.tolist()).normalized()
    x_axis = y_axis.cross(z_axis).normalized()
    matrix = Matrix.Identity(4)
    distance = max(3.0, float(extents.max()) * 5.0)
    camera_position = Vector(center.tolist()) + z_axis * distance
    for row in range(3):
        matrix[row][0] = x_axis[row]
        matrix[row][1] = y_axis[row]
        matrix[row][2] = z_axis[row]
        matrix[row][3] = camera_position[row]
    camera.matrix_world = matrix

    _add_area_light(
        bpy,
        name="PotBottomCoverageKey",
        location=center + np.asarray([2.5, -2.5, 3.0]) * max(float(extents.max()), 1.0),
        center=center,
        energy=700.0,
    )
    _add_area_light(
        bpy,
        name="PotBottomCoverageFill",
        location=center + np.asarray([-2.0, 2.5, 1.5]) * max(float(extents.max()), 1.0),
        center=center,
        energy=400.0,
    )

    scene = bpy.context.scene
    scene.camera = camera
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.look = "AgX - Medium High Contrast"
    output_directory.mkdir(parents=True, exist_ok=True)

    pbr_path = output_directory / "pbr-side-alpha.png"
    scene.view_layers[0].material_override = None
    scene.render.filepath = str(pbr_path)
    bpy.context.view_layer.update()
    bpy.ops.render.render(write_still=True)
    pbr_image = bpy.data.images.load(str(pbr_path), check_existing=False)
    pbr_alpha = _image_pixels(pbr_image)[:, :, 3].copy()
    bpy.data.images.remove(pbr_image)

    analytic_material = bpy.data.materials.new(name="PotBottomCoverageAnalyticOpaque")
    analytic_material.use_nodes = True
    analytic_material.use_backface_culling = False
    analytic_bsdf = next(
        node for node in analytic_material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"
    )
    analytic_bsdf.inputs["Base Color"].default_value = [0.8, 0.8, 0.8, 1.0]
    analytic_bsdf.inputs["Metallic"].default_value = 0.0
    analytic_bsdf.inputs["Roughness"].default_value = 1.0
    analytic_path = output_directory / "analytic-side-alpha.png"
    scene.view_layers[0].material_override = analytic_material
    scene.render.filepath = str(analytic_path)
    bpy.ops.render.render(write_still=True)
    analytic_image = bpy.data.images.load(str(analytic_path), check_existing=False)
    analytic_alpha = _image_pixels(analytic_image)[:, :, 3].copy()
    bpy.data.images.remove(analytic_image)
    scene.view_layers[0].material_override = None

    pbr_mask = pbr_alpha > 0.05
    analytic_mask = analytic_alpha > 0.05
    require(pbr_mask.any() and analytic_mask.any(), "bottom coverage renders are blank")
    pbr_rows = np.flatnonzero(pbr_mask.any(axis=1))
    analytic_rows = np.flatnonzero(analytic_mask.any(axis=1))
    pbr_bottom = int(pbr_rows.min())
    analytic_bottom = int(analytic_rows.min())
    vertical_span = int(analytic_rows.max() - analytic_rows.min() + 1)
    lower_band_top = analytic_bottom + max(4, round(vertical_span * 0.18))
    lower_band = analytic_mask.copy()
    lower_band[lower_band_top + 1 :, :] = False
    lower_coverage = float((pbr_mask & lower_band).sum() / lower_band.sum())
    union = pbr_mask | analytic_mask
    silhouette_iou = float((pbr_mask & analytic_mask).sum() / union.sum())
    bottom_gap = max(0, pbr_bottom - analytic_bottom)
    gates = {
        "pbr_render_nonblank": True,
        "analytic_render_nonblank": True,
        "pbr_bottom_gap_at_most_2px": bottom_gap <= 2,
        "lower_band_analytic_coverage_at_least_0_99": lower_coverage >= 0.99,
        "whole_silhouette_iou_at_least_0_99": silhouette_iou >= 0.99,
    }
    require(all(gates.values()), f"PBR bottom coverage QA failed: {gates}")
    return {
        "kind": "orthographic_pbr_vs_analytic_bottom_coverage",
        "semantic_up_axis_gltf": semantic_up_axis,
        "view_axis_blender_world": position_axis.tolist(),
        "resolution_wh": [512, 512],
        "alpha_threshold": 0.05,
        "pbr_bottom_row_from_image_bottom": pbr_bottom,
        "analytic_bottom_row_from_image_bottom": analytic_bottom,
        "pbr_bottom_gap_px": bottom_gap,
        "lower_band_analytic_coverage": lower_coverage,
        "whole_silhouette_iou": silhouette_iou,
        "gates": gates,
        "artifacts": {
            "pbr": {
                "path": str(pbr_path),
                "size_bytes": pbr_path.stat().st_size,
                "sha256": sha256_file(pbr_path),
            },
            "analytic": {
                "path": str(analytic_path),
                "size_bytes": analytic_path.stat().st_size,
                "sha256": sha256_file(analytic_path),
            },
        },
    }


def _export_selected(bpy: Any, imported: list[Any], output_path: Path) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for item in imported:
        item.select_set(True)
    if imported:
        bpy.context.view_layer.objects.active = imported[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(output_path),
        export_format="GLB",
        use_selection=True,
        export_yup=True,
        export_apply=False,
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
        export_image_format="AUTO",
        export_keep_originals=False,
        export_animations=False,
        export_cameras=False,
        export_lights=False,
        check_existing=False,
    )
    require(output_path.is_file() and output_path.stat().st_size > 1024, "GLB export failed")


def correct_region(args: argparse.Namespace) -> dict[str, Any]:
    import bpy

    source_path = args.source_glb.expanduser().resolve()
    manifest_path = args.mask_manifest.expanduser().resolve()
    output_path = args.output_glb.expanduser().resolve()
    receipt_path = args.receipt.expanduser().resolve()
    require(source_path.is_file(), f"source GLB is missing: {source_path}")
    require(manifest_path.is_file(), f"mask manifest is missing: {manifest_path}")
    require(source_path != output_path, "output GLB must differ from source GLB")

    samples, color_evidence = _masked_color_statistics(
        bpy,
        manifest_path=manifest_path,
        object_id=args.object_id,
    )
    color = derive_neutral_metal_base_color(samples)

    _reset_scene(bpy)
    imported = _import_glb(bpy, source_path)
    mesh_objects = [item for item in imported if item.type == "MESH"]
    source_snapshot = _geometry_snapshot(mesh_objects)
    source_quality = [_mesh_quality(item) for item in mesh_objects]
    source_structure = _glb_structure(source_path)

    selection, selected_polygons = _select_lower_container_region(
        mesh_objects,
        semantic_up_axis=args.semantic_up_axis,
        top_height_ratio=args.top_height_ratio,
        radius_ratio=args.radius_ratio,
    )
    require(
        args.minimum_selected_face_fraction
        <= selection["selected_face_fraction"]
        <= args.maximum_selected_face_fraction,
        "selected face fraction is outside fail-closed bounds",
    )
    texture_region, source_base_pixels, selected_uv_mask = _apply_pbr_texture_region(
        bpy,
        mesh_objects=mesh_objects,
        selected_polygons=selected_polygons,
        target_base_srgb=color["target_base_color_srgb"],
        metallic=args.metallic,
        roughness=args.roughness,
    )
    edited_snapshot = _geometry_snapshot(mesh_objects)
    require(
        source_snapshot["object_local_topology_sha256"]
        == edited_snapshot["object_local_topology_sha256"],
        "geometry changed while assigning the material",
    )
    require(
        source_snapshot["world_bounds_min"] == edited_snapshot["world_bounds_min"],
        "min bound changed",
    )
    require(
        source_snapshot["world_bounds_max"] == edited_snapshot["world_bounds_max"],
        "max bound changed",
    )
    require(
        texture_region["source_base_texture"]["outside_selected_uv_pixel_float32_sha256"]
        == texture_region["corrected_base_texture"]["outside_selected_uv_pixel_float32_sha256"],
        "source leaf texture pixels changed outside the selected UV region",
    )

    donor_path = output_path.with_suffix(".material-donor.glb")
    _export_selected(bpy, imported, donor_path)
    donor_sha256 = sha256_file(donor_path)
    donor_size_bytes = donor_path.stat().st_size
    material_patch = _patch_source_geometry_with_donor_material(
        source_path=source_path,
        donor_path=donor_path,
        output_path=output_path,
    )
    material_patch["temporary_donor_sha256"] = donor_sha256
    material_patch["temporary_donor_size_bytes"] = donor_size_bytes
    donor_path.unlink()

    _reset_scene(bpy)
    roundtrip_imported = _import_glb(bpy, output_path)
    roundtrip_meshes = [item for item in roundtrip_imported if item.type == "MESH"]
    roundtrip_snapshot = _geometry_snapshot(roundtrip_meshes)
    roundtrip_quality = [_mesh_quality(item) for item in roundtrip_meshes]
    roundtrip_structure = _glb_structure(output_path)
    _, _, roundtrip_base_node = _principled_and_base_image(
        roundtrip_meshes,
        require_untextured_metallic_roughness=False,
    )
    roundtrip_base_pixels = _image_pixels(roundtrip_base_node.image).astype(np.float32)
    require(
        roundtrip_base_pixels.shape == source_base_pixels.shape,
        "roundtrip base texture dimensions changed",
    )
    outside_uv_mask = ~selected_uv_mask
    outside_source_hash = _pixel_sha256(source_base_pixels[outside_uv_mask])
    outside_roundtrip_hash = _pixel_sha256(roundtrip_base_pixels[outside_uv_mask])
    target_texture_srgb = np.asarray(
        texture_region["target_texture_rgb_srgb"],
        dtype=np.float32,
    )
    selected_roundtrip_rgb = roundtrip_base_pixels[selected_uv_mask, :3]
    selected_roundtrip_alpha = roundtrip_base_pixels[selected_uv_mask, 3]
    quality_gates = _validate_mesh_quality(roundtrip_quality)
    bottom_coverage = _pbr_bottom_coverage_qa(
        bpy,
        mesh_objects=roundtrip_meshes,
        semantic_up_axis=args.semantic_up_axis,
        output_directory=receipt_path.parent / "bottom-coverage",
    )
    gates = {
        "source_hash_bound": True,
        "mask_manifest_hash_bound": True,
        "real_pot_color_samples": color_evidence["sample_count"] >= 64,
        "selected_face_fraction_in_bounds": True,
        "in_memory_topology_unchanged": True,
        "roundtrip_mesh_count_unchanged": (
            source_snapshot["mesh_count"] == roundtrip_snapshot["mesh_count"]
        ),
        "roundtrip_vertex_count_unchanged": (
            source_snapshot["vertex_count"] == roundtrip_snapshot["vertex_count"]
        ),
        "roundtrip_unique_vertex_count_unchanged": (
            source_snapshot["unique_world_vertex_count_at_1e7"]
            == roundtrip_snapshot["unique_world_vertex_count_at_1e7"]
        ),
        "roundtrip_face_count_unchanged": (
            source_snapshot["face_count"] == roundtrip_snapshot["face_count"]
        ),
        "roundtrip_world_triangle_geometry_unchanged": (
            source_snapshot["canonical_world_triangles_rounded_1e7_sha256"]
            == roundtrip_snapshot["canonical_world_triangles_rounded_1e7_sha256"]
        ),
        "roundtrip_bounds_unchanged_at_1e7": (
            np.allclose(
                source_snapshot["world_bounds_min"],
                roundtrip_snapshot["world_bounds_min"],
                rtol=0.0,
                atol=1e-7,
            )
            and np.allclose(
                source_snapshot["world_bounds_max"],
                roundtrip_snapshot["world_bounds_max"],
                rtol=0.0,
                atol=1e-7,
            )
        ),
        "roundtrip_primitive_count_unchanged": (
            source_structure["primitive_count"] == roundtrip_structure["primitive_count"]
        ),
        "roundtrip_material_count_unchanged": (
            source_structure["material_count"] == roundtrip_structure["material_count"]
        ),
        "roundtrip_base_color_texture_present": (
            roundtrip_structure["base_color_texture_count"] >= 1
        ),
        "roundtrip_metallic_roughness_texture_present": (
            roundtrip_structure["metallic_roughness_texture_count"] >= 1
        ),
        "roundtrip_materials_double_sided": (
            roundtrip_structure["material_count"] > 0
            and roundtrip_structure["double_sided_material_count"]
            == roundtrip_structure["material_count"]
        ),
        "source_leaf_texture_pixels_preserved_outside_selected_uv": (
            outside_source_hash == outside_roundtrip_hash
        ),
        "selected_base_color_matches_real_mask_target": bool(
            np.allclose(
                selected_roundtrip_rgb,
                target_texture_srgb,
                rtol=0.0,
                atol=1.0 / 255.0,
            )
        ),
        "selected_pot_texture_alpha_is_opaque": bool(
            np.allclose(selected_roundtrip_alpha, 1.0, rtol=0.0, atol=1.0 / 255.0)
        ),
        "pbr_bottom_coverage": all(bottom_coverage["gates"].values()),
        **quality_gates,
    }
    require(all(gates.values()), f"fail-closed GLB QA failed: {gates}")

    receipt = {
        "schema_version": 1,
        "kind": "video2world.pbr_region_material_correction",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "technical_passed_visual_review_pending",
        "claim_scope": (
            "Geometry is unchanged. Only complete connected components inside the lower semantic "
            "cylinder receive geometry-rasterized base-color and metallic-roughness texels derived "
            "from real SAM3 pot-mask RGB statistics."
        ),
        "object_id": args.object_id,
        "inputs": {
            "source_glb": {
                "path": str(source_path),
                "size_bytes": source_path.stat().st_size,
                "sha256": sha256_file(source_path),
            },
            "mask_manifest": {
                "path": str(manifest_path),
                "size_bytes": manifest_path.stat().st_size,
                "sha256": sha256_file(manifest_path),
            },
        },
        "color_evidence": color_evidence,
        "material": {
            "name": texture_region["material_name"],
            "base_color": color,
            "metallic": args.metallic,
            "roughness": args.roughness,
            "texture_region": texture_region,
            "source_leaf_texture_unchanged_outside_selected_uv": True,
        },
        "region_selector": selection,
        "geometry": {
            "source": source_snapshot,
            "after_material_assignment": edited_snapshot,
            "roundtrip_output": roundtrip_snapshot,
        },
        "glb_structure": {
            "source": source_structure,
            "roundtrip_output": roundtrip_structure,
        },
        "geometry_preserving_material_patch": material_patch,
        "pbr_bottom_coverage": bottom_coverage,
        "mesh_quality": {
            "source": source_quality,
            "roundtrip_output": roundtrip_quality,
        },
        "gates": gates,
        "output": {
            "corrected_glb": {
                "path": str(output_path),
                "size_bytes": output_path.stat().st_size,
                "sha256": sha256_file(output_path),
            }
        },
        "limitations": [
            "Real masks are modal and provide appearance evidence only for visible pot surfaces.",
            (
                "The target material neutralizes warm scene illumination instead of baking it "
                "into albedo."
            ),
            "Visual six-view review remains mandatory before scene-fit or candidate adoption.",
        ],
    }
    write_json(receipt_path, receipt)
    return receipt


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--source-glb", type=Path, required=True)
    result.add_argument("--mask-manifest", type=Path, required=True)
    result.add_argument("--object-id", required=True)
    result.add_argument("--semantic-up-axis", choices=sorted(GLTF_AXIS_TO_BLENDER), required=True)
    result.add_argument("--output-glb", type=Path, required=True)
    result.add_argument("--receipt", type=Path, required=True)
    result.add_argument("--top-height-ratio", type=float, default=0.34)
    result.add_argument("--radius-ratio", type=float, default=0.55)
    result.add_argument("--metallic", type=float, default=0.65)
    result.add_argument("--roughness", type=float, default=0.42)
    result.add_argument("--minimum-selected-face-fraction", type=float, default=0.03)
    result.add_argument("--maximum-selected-face-fraction", type=float, default=0.10)
    return result


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    args = parser().parse_args(argv)
    require(0.0 < args.top_height_ratio < 0.5, "top height ratio must be inside (0, 0.5)")
    require(0.0 < args.radius_ratio < 1.0, "radius ratio must be inside (0, 1)")
    require(0.0 <= args.metallic <= 1.0, "metallic must be inside [0, 1]")
    require(0.0 <= args.roughness <= 1.0, "roughness must be inside [0, 1]")
    receipt = correct_region(args)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output": receipt["output"]["corrected_glb"],
                "receipt": str(args.receipt.expanduser().resolve()),
                "selected_face_fraction": receipt["region_selector"]["selected_face_fraction"],
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
