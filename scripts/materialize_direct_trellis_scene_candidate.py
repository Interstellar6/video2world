#!/usr/bin/env python3
"""Materialize a local-only scene candidate from direct TRELLIS scene-fit receipts.

The materializer deliberately keeps the supplied PGSR scene PLY byte-for-byte.
It does not carve the scene and does not consume or generate a clean plate.  Each
adopted object uses the receipt's scene-local PBR GLB as its visual, logical,
selection, and collision asset.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import struct
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import trimesh
from plyfile import PlyData

from video2world.hashing import atomic_write_json, sha256_file

CONFIG_KIND = "video2world.direct_trellis_scene_candidate_config"
RECEIPT_KIND = "video2world.completed_object_scene_fit"
REPORT_KIND = "video2world.direct_trellis_scene_candidate_report"
LOD_QA_KIND = "video2world.pbr_browser_lod_qa"
RUNTIME_MAX_COLLISION_FACES = 100_000
UNADOPTED_OBJECT_POLICY = "original_static_scene_only_with_logical_ancestors"
LOGICAL_HIERARCHY_ROLE = "unrendered_unselectable_hierarchy_ancestor"
FORBIDDEN_PROXY_FIELDS = frozenset({"visual", "renderAsset", "colliderProxy"})
PGSR_REQUIRED_PROPERTIES = frozenset(
    {
        "x",
        "y",
        "z",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    }
)
HIERARCHY_FIELDS = (
    "semanticGranularity",
    "parentObjectId",
    "movesWithParent",
    "childObjectIds",
    "independentlyMovable",
)
LOD_REQUIRED_QA_GATES = (
    "reimportedAfterExport",
    "targetMet",
    "nonempty",
    "finiteGeometry",
    "nondegenerateTriangles",
    "windingConsistent",
    "windingRepairSuccessful",
    "pbrMaterialsPreserved",
    "materialSlotsPreserved",
    "textureImagesPresent",
    "uvLayersPreserved",
    "normalsPreserved",
    "boundsPreservedWithinTolerance",
)


@dataclass(frozen=True)
class GlbAudit:
    path: Path
    sha256: str
    size_bytes: int
    vertices: int
    faces: int
    bounds: list[list[float]]
    finite: bool
    degenerate_triangles: int
    nondegenerate: bool
    winding_consistent: bool
    watertight: bool
    material_types: list[str]
    pbr_materials: int
    texture_images: int
    uv_layers: int
    normal_loops: int
    glb_has_materials: bool
    glb_has_normals: bool
    glb_has_uvs: bool
    normals_finite: bool


@dataclass(frozen=True)
class PgsrAudit:
    path: Path
    sha256: str
    size_bytes: int
    vertex_count: int
    properties: list[str]
    bounds: list[list[float]]
    finite: bool
    binary_little_endian: bool


@dataclass(frozen=True)
class StaticColliderAudit:
    path: Path
    sha256: str
    size_bytes: int
    vertex_count: int
    face_count: int
    vertex_properties: list[str]
    vertex_position_type: str
    face_index_type: str
    bounds: list[list[float]]
    finite: bool
    face_indices_valid: bool
    degenerate_triangles: int
    binary_little_endian: bool


@dataclass(frozen=True)
class VerifiedObject:
    object_id: str
    receipt_path: Path
    receipt_sha256: str
    receipt: dict[str, Any]
    scene_fit_glb: GlbAudit
    glb: GlbAudit
    pivot: list[float]
    world_bounds: list[list[float]]
    override_qa_path: Path | None = None
    override_qa_sha256: str | None = None
    source_normalization: dict[str, Any] | None = None


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON {path}: {exc}") from exc
    require(isinstance(payload, dict), f"JSON root must be an object: {path}")
    return payload


def resolve_path(value: Any, *, relative_to: Path, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label}.path must be a non-empty string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    path = path.resolve()
    require(path.is_file(), f"{label} file does not exist: {path}")
    return path


def require_hash(path: Path, expected: Any, *, label: str) -> tuple[str, int]:
    require(
        isinstance(expected, str) and len(expected) == 64,
        f"{label}.sha256 must be a 64-character digest",
    )
    observed, size = sha256_file(path)
    require(observed == expected, f"{label} sha256 mismatch: {observed} != {expected}")
    return observed, size


def _local_fs_url(path: Path) -> str:
    return "/@fs" + quote(path.resolve().as_posix(), safe="/")


def _finite_vector(value: Any, *, length: int, label: str) -> list[float]:
    require(isinstance(value, list) and len(value) == length, f"{label} must have {length} values")
    result = [float(item) for item in value]
    require(all(math.isfinite(item) for item in result), f"{label} must be finite")
    return result


def _same_bounds(left: Any, right: Any, *, tolerance: float) -> bool:
    try:
        left_array = np.asarray(left, dtype=np.float64)
        right_array = np.asarray(right, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        left_array.shape == right_array.shape == (2, 3)
        and np.isfinite(left_array).all()
        and np.isfinite(right_array).all()
        and np.allclose(left_array, right_array, rtol=0, atol=tolerance)
    )


def audit_pgsr_ply(path: Path, *, expected_sha256: str) -> PgsrAudit:
    digest, size = require_hash(path, expected_sha256, label="static_scene")
    try:
        ply = PlyData.read(path, mmap=True)
    except Exception as exc:  # pragma: no cover - parser error detail is library-specific
        raise RuntimeError(f"cannot inspect PGSR PLY {path}: {exc}") from exc
    require("vertex" in ply, f"PGSR PLY has no vertex element: {path}")
    vertex = ply["vertex"]
    properties = [item.name for item in vertex.properties]
    missing = sorted(PGSR_REQUIRED_PROPERTIES.difference(properties))
    require(not missing, f"PGSR PLY is missing Gaussian properties: {missing}")
    require(vertex.count > 0, f"PGSR PLY is empty: {path}")
    positions = np.column_stack(
        [np.asarray(vertex.data[axis], dtype=np.float64) for axis in ("x", "y", "z")]
    )
    finite = bool(np.isfinite(positions).all())
    require(finite, f"PGSR PLY contains non-finite positions: {path}")
    binary_little_endian = bool(not ply.text and ply.byte_order == "<")
    require(binary_little_endian, "PGSR PLY must be binary little-endian")
    bounds = np.asarray([positions.min(axis=0), positions.max(axis=0)]).tolist()
    return PgsrAudit(
        path=path,
        sha256=digest,
        size_bytes=size,
        vertex_count=int(vertex.count),
        properties=properties,
        bounds=bounds,
        finite=finite,
        binary_little_endian=binary_little_endian,
    )


def audit_static_collider_ply(path: Path, *, expected_sha256: str) -> StaticColliderAudit:
    digest, size = require_hash(path, expected_sha256, label="static_collider")
    try:
        ply = PlyData.read(
            path,
            mmap="c",
            known_list_len={"face": {"vertex_indices": 3}},
        )
    except Exception as exc:  # pragma: no cover - parser error detail is library-specific
        raise RuntimeError(f"cannot inspect static collider PLY {path}: {exc}") from exc
    require("vertex" in ply and "face" in ply, "static collider PLY needs vertex and face elements")
    vertex = ply["vertex"]
    face = ply["face"]
    require(vertex.count > 0 and face.count > 0, "static collider PLY must be non-empty")
    vertex_properties = [item.name for item in vertex.properties]
    require(
        {"x", "y", "z"}.issubset(vertex_properties),
        "static collider PLY is missing x/y/z vertex properties",
    )
    require(
        "vertex_indices" in (face.data.dtype.names or ()),
        "static collider PLY is missing vertex_indices",
    )
    positions = np.column_stack(
        [np.asarray(vertex.data[axis], dtype=np.float64) for axis in ("x", "y", "z")]
    )
    finite = bool(np.isfinite(positions).all())
    require(finite, "static collider PLY contains non-finite vertices")
    indices = np.asarray(face.data["vertex_indices"])
    require(
        indices.shape == (face.count, 3) and np.issubdtype(indices.dtype, np.integer),
        "static collider PLY faces must all be triangles",
    )
    face_indices_valid = bool(
        indices.size > 0 and int(indices.min()) >= 0 and int(indices.max()) < vertex.count
    )
    require(face_indices_valid, "static collider PLY contains invalid face indices")
    duplicate_indices = int(
        np.sum(
            (indices[:, 0] == indices[:, 1])
            | (indices[:, 1] == indices[:, 2])
            | (indices[:, 0] == indices[:, 2])
        )
    )
    degenerate_triangles = duplicate_indices
    for offset in range(0, face.count, 250_000):
        triangle = indices[offset : offset + 250_000]
        edge_a = positions[triangle[:, 1]] - positions[triangle[:, 0]]
        edge_b = positions[triangle[:, 2]] - positions[triangle[:, 0]]
        area_twice = np.linalg.norm(np.cross(edge_a, edge_b), axis=1)
        geometric = ~np.isfinite(area_twice) | (area_twice <= 1e-14)
        if duplicate_indices:
            duplicated = (
                (triangle[:, 0] == triangle[:, 1])
                | (triangle[:, 1] == triangle[:, 2])
                | (triangle[:, 0] == triangle[:, 2])
            )
            geometric &= ~duplicated
        degenerate_triangles += int(np.sum(geometric))
    require(degenerate_triangles == 0, "static collider PLY contains degenerate triangles")
    binary_little_endian = bool(not ply.text and ply.byte_order == "<")
    require(binary_little_endian, "static collider PLY must be binary little-endian")
    position_dtype = vertex.data.dtype.fields["x"][0]
    bounds = np.asarray([positions.min(axis=0), positions.max(axis=0)]).tolist()
    return StaticColliderAudit(
        path=path,
        sha256=digest,
        size_bytes=size,
        vertex_count=int(vertex.count),
        face_count=int(face.count),
        vertex_properties=vertex_properties,
        vertex_position_type=str(position_dtype),
        face_index_type=str(indices.dtype),
        bounds=bounds,
        finite=finite,
        face_indices_valid=face_indices_valid,
        degenerate_triangles=degenerate_triangles,
        binary_little_endian=binary_little_endian,
    )


def _merged_mesh(scene: trimesh.Scene, *, label: str) -> trimesh.Trimesh:
    merged = scene.to_geometry()
    require(
        isinstance(merged, trimesh.Trimesh) and len(merged.vertices) > 0 and len(merged.faces) > 0,
        f"{label} contains no triangle geometry",
    )
    return merged


def _read_glb_document(path: Path, *, expected_size: int) -> dict[str, Any]:
    with path.open("rb") as stream:
        header = stream.read(12)
        require(len(header) == 12, f"GLB header is truncated: {path}")
        magic, version, declared_size = struct.unpack("<4sII", header)
        require(magic == b"glTF" and version == 2, f"asset is not GLB 2.0: {path}")
        require(declared_size == expected_size, f"GLB declared byte count mismatch: {path}")
        chunk_header = stream.read(8)
        require(len(chunk_header) == 8, f"GLB JSON chunk header is truncated: {path}")
        chunk_size, chunk_type = struct.unpack("<II", chunk_header)
        require(chunk_type == 0x4E4F534A, f"GLB first chunk is not JSON: {path}")
        try:
            document = json.loads(stream.read(chunk_size).rstrip(b" \t\r\n\0"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot decode GLB JSON: {path}: {exc}") from exc
    require(isinstance(document, dict), f"GLB JSON root must be an object: {path}")
    return document


def _referenced_texture_images(document: dict[str, Any], material_ids: set[int]) -> int:
    texture_ids: set[int] = set()
    materials = document.get("materials", [])
    for material_id in material_ids:
        require(0 <= material_id < len(materials), "GLB primitive material index is invalid")
        material = materials[material_id]
        pbr = material.get("pbrMetallicRoughness", {})
        for record in (
            pbr.get("baseColorTexture"),
            pbr.get("metallicRoughnessTexture"),
            material.get("normalTexture"),
            material.get("occlusionTexture"),
            material.get("emissiveTexture"),
        ):
            if isinstance(record, dict) and isinstance(record.get("index"), int):
                texture_ids.add(record["index"])
    textures = document.get("textures", [])
    image_ids: set[int] = set()
    for texture_id in texture_ids:
        require(0 <= texture_id < len(textures), "GLB material texture index is invalid")
        source = textures[texture_id].get("source")
        require(isinstance(source, int), "GLB texture source is missing")
        image_ids.add(source)
    require(
        all(0 <= image_id < len(document.get("images", [])) for image_id in image_ids),
        "GLB texture image index is invalid",
    )
    return len(image_ids)


def _glb_attribute_contract(document: dict[str, Any], *, faces: int) -> dict[str, Any]:
    meshes = document.get("meshes", [])
    primitives = [primitive for mesh in meshes for primitive in mesh.get("primitives", [])]
    require(primitives, "GLB JSON declares no mesh primitives")
    require(
        all(primitive.get("mode", 4) == 4 for primitive in primitives),
        "GLB contains non-triangle primitives",
    )
    material_ids = {
        int(primitive["material"]) for primitive in primitives if "material" in primitive
    }
    all_have_material = all("material" in primitive for primitive in primitives)
    all_have_normals = all("NORMAL" in primitive.get("attributes", {}) for primitive in primitives)
    all_have_uvs = all("TEXCOORD_0" in primitive.get("attributes", {}) for primitive in primitives)
    uv_layers = 0
    for mesh in meshes:
        layers = {
            name
            for primitive in mesh.get("primitives", [])
            for name in primitive.get("attributes", {})
            if name.startswith("TEXCOORD_")
        }
        uv_layers += len(layers)
    return {
        "pbr_materials": len(material_ids) if all_have_material else 0,
        "texture_images": _referenced_texture_images(document, material_ids),
        "uv_layers": uv_layers,
        "normal_loops": faces * 3,
        "glb_has_materials": all_have_material,
        "glb_has_normals": all_have_normals,
        "glb_has_uvs": all_have_uvs,
    }


def audit_unified_glb(
    path: Path,
    *,
    maximum_faces: int,
    enforce_runtime_geometry: bool = True,
) -> GlbAudit:
    require(path.suffix.lower() == ".glb", f"unified asset must be GLB: {path}")
    digest, size = sha256_file(path)
    document = _read_glb_document(path, expected_size=size)
    try:
        loaded = trimesh.load(path, force="scene", process=False)
    except Exception as exc:  # pragma: no cover - parser error detail is library-specific
        raise RuntimeError(f"cannot inspect GLB {path}: {exc}") from exc
    scene = loaded if isinstance(loaded, trimesh.Scene) else trimesh.Scene(loaded)
    require(bool(scene.geometry), f"GLB has no geometry: {path}")
    mesh = _merged_mesh(scene, label=str(path))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    bounds = np.asarray([vertices.min(axis=0), vertices.max(axis=0)])
    finite = bool(np.isfinite(vertices).all())
    face_areas = np.asarray(mesh.area_faces)
    degenerate_triangles = int(np.sum(~np.isfinite(face_areas) | (face_areas <= 1e-14)))
    nondegenerate = bool(len(mesh.faces) > 0 and degenerate_triangles == 0)
    winding_consistent = bool(mesh.is_winding_consistent)
    watertight = bool(mesh.is_watertight)
    normals_finite = bool(np.isfinite(mesh.vertex_normals).all())
    material_types = sorted(
        {
            type(getattr(getattr(geometry, "visual", None), "material", None)).__name__
            for geometry in scene.geometry.values()
        }
    )
    contract = _glb_attribute_contract(document, faces=len(mesh.faces))
    require(finite, f"GLB contains non-finite vertices: {path}")
    require(normals_finite, f"GLB contains non-finite normals: {path}")
    require(len(mesh.faces) > 0, f"GLB contains no faces: {path}")
    if enforce_runtime_geometry:
        require(nondegenerate, f"GLB contains degenerate faces: {path}")
        require(winding_consistent, f"GLB winding is inconsistent: {path}")
        require(len(mesh.faces) <= maximum_faces, f"GLB exceeds face budget: {path}")
    require(
        material_types == ["PBRMaterial"],
        f"GLB must use PBR materials for every geometry: {path} ({material_types})",
    )
    return GlbAudit(
        path=path,
        sha256=digest,
        size_bytes=size,
        vertices=len(mesh.vertices),
        faces=len(mesh.faces),
        bounds=bounds.tolist(),
        finite=finite,
        degenerate_triangles=degenerate_triangles,
        nondegenerate=nondegenerate,
        winding_consistent=winding_consistent,
        watertight=watertight,
        material_types=material_types,
        pbr_materials=contract["pbr_materials"],
        texture_images=contract["texture_images"],
        uv_layers=contract["uv_layers"],
        normal_loops=contract["normal_loops"],
        glb_has_materials=contract["glb_has_materials"],
        glb_has_normals=contract["glb_has_normals"],
        glb_has_uvs=contract["glb_has_uvs"],
        normals_finite=normals_finite,
    )


def _verify_receipt(
    receipt_path: Path,
    *,
    expected_sha256: str,
    maximum_faces: int,
    allow_runtime_asset_override: bool = False,
) -> VerifiedObject:
    receipt_sha256, _ = require_hash(
        receipt_path,
        expected_sha256,
        label="scene_fit_receipt",
    )
    receipt = read_json(receipt_path)
    require(receipt.get("schema_version") == 1, "scene-fit receipt schema version mismatch")
    require(receipt.get("kind") == RECEIPT_KIND, "scene-fit receipt kind mismatch")
    object_id = receipt.get("object_id")
    require(isinstance(object_id, str) and object_id, "scene-fit receipt object_id is missing")
    require(
        receipt.get("representation_mode") == "unified_pbr_mesh_visual_logic_collision",
        f"scene-fit receipt is not a unified PBR object: {object_id}",
    )
    require(
        receipt.get("status") == "technical_gates_passed"
        and receipt.get("all_acceptance_gates_passed") is True,
        f"scene-fit technical gates failed: {object_id}",
    )
    gates = receipt.get("acceptance_gates")
    require(
        isinstance(gates, dict) and gates and all(value is True for value in gates.values()),
        f"scene-fit receipt contains a failed gate: {object_id}",
    )
    require(
        receipt.get("final_export_qa", {}).get("scene_local_pivot_round_trip") is True,
        f"scene-local pivot round trip failed: {object_id}",
    )

    scene_local = receipt.get("exports", {}).get("scene_local")
    require(isinstance(scene_local, dict), f"scene-local export is missing: {object_id}")
    require(
        scene_local.get("coordinate_space") == "scene_local_about_target_pivot",
        f"scene-local coordinate space mismatch: {object_id}",
    )
    require(
        scene_local.get("glb_role") == "pbr_visual_logic_authority",
        f"scene-local GLB is not the PBR authority: {object_id}",
    )
    glb_record = scene_local.get("glb")
    require(isinstance(glb_record, dict), f"scene-local GLB record is missing: {object_id}")
    glb_path = resolve_path(
        glb_record.get("path"),
        relative_to=receipt_path.parent,
        label=f"{object_id}.scene_local.glb",
    )
    glb = audit_unified_glb(
        glb_path,
        maximum_faces=maximum_faces,
        enforce_runtime_geometry=not allow_runtime_asset_override,
    )
    require(glb.sha256 == glb_record.get("sha256"), f"final GLB hash mismatch: {object_id}")
    require(
        glb.size_bytes == glb_record.get("bytes"),
        f"final GLB byte count mismatch: {object_id}",
    )
    require(
        glb.vertices == glb_record.get("vertices"),
        f"final GLB vertex count mismatch: {object_id}",
    )
    require(glb.faces == glb_record.get("faces"), f"final GLB face count mismatch: {object_id}")
    tolerance = float(receipt.get("final_export_qa", {}).get("bounds_tolerance", 1e-5))
    require(
        math.isfinite(tolerance) and 0 < tolerance <= 1.0,
        f"invalid receipt bounds tolerance: {object_id}",
    )
    require(
        _same_bounds(glb.bounds, glb_record.get("bounds"), tolerance=tolerance),
        f"final GLB bounds mismatch: {object_id}",
    )
    require(
        glb_record.get("finite_vertices") is True
        and glb_record.get("winding_consistent") is glb.winding_consistent
        and glb_record.get("watertight") is glb.watertight,
        f"final GLB geometry claims mismatch: {object_id}",
    )
    require(
        sorted(glb_record.get("material_types", [])) == glb.material_types,
        f"final GLB material claims mismatch: {object_id}",
    )

    runtime = receipt.get("scene_local_runtime")
    require(isinstance(runtime, dict), f"scene-local runtime is missing: {object_id}")
    require(
        runtime.get("asset_coordinates_baked") == "rotation_and_scale_only",
        f"scene-local asset does not bake rotation and scale: {object_id}",
    )
    runtime_placement = runtime.get("placement", {})
    pivot = _finite_vector(
        runtime_placement.get("pivot"),
        length=3,
        label=f"{object_id}.scene_local_runtime.placement.pivot",
    )
    require(
        runtime_placement.get("scale") == [1.0, 1.0, 1.0]
        and runtime_placement.get("rotation_xyzw") == [0.0, 0.0, 0.0, 1.0],
        f"scene-local runtime placement is not pivot-only: {object_id}",
    )
    export_pivot = _finite_vector(
        scene_local.get("placement", {}).get("pivot"),
        length=3,
        label=f"{object_id}.exports.scene_local.placement.pivot",
    )
    require(np.allclose(pivot, export_pivot, rtol=0, atol=1e-9), f"pivot mismatch: {object_id}")
    require(
        scene_local.get("placement", {}).get("runtime_scale") == [1.0, 1.0, 1.0]
        and scene_local.get("placement", {}).get("runtime_rotation_xyzw") == [0.0, 0.0, 0.0, 1.0],
        f"scene-local runtime transform is not pivot-only: {object_id}",
    )
    bounds = np.asarray(glb.bounds, dtype=np.float64) + np.asarray(pivot, dtype=np.float64)
    require(
        _same_bounds(
            bounds,
            receipt.get("final_export_qa", {}).get(
                "predicted_world_bounds_from_scene_local_bytes_and_pivot"
            ),
            tolerance=tolerance,
        ),
        f"scene-local world bounds mismatch: {object_id}",
    )
    return VerifiedObject(
        object_id=object_id,
        receipt_path=receipt_path,
        receipt_sha256=receipt_sha256,
        receipt=receipt,
        scene_fit_glb=glb,
        glb=glb,
        pivot=pivot,
        world_bounds=bounds.tolist(),
    )


def _qa_bounds(value: Any, *, label: str) -> list[list[float]]:
    require(isinstance(value, dict), f"{label} bounds must be an object")
    minimum = _finite_vector(value.get("min"), length=3, label=f"{label}.bounds.min")
    maximum = _finite_vector(value.get("max"), length=3, label=f"{label}.bounds.max")
    require(
        all(high >= low for low, high in zip(minimum, maximum, strict=True)),
        f"{label} bounds are invalid",
    )
    coordinate_system = value.get("coordinateSystem")
    if coordinate_system == "blender_world_z_up":
        # Blender's glTF importer maps glTF (x, y, z) to Blender (x, -z, y).
        # Convert its axis-aligned bounds back to the scene-local glTF frame.
        return [
            [minimum[0], minimum[2], -maximum[1]],
            [maximum[0], maximum[2], -minimum[1]],
        ]
    require(
        coordinate_system in {"gltf_y_up", "scene_local_about_target_pivot"},
        f"{label} bounds coordinate system is unsupported: {coordinate_system}",
    )
    return [minimum, maximum]


def _verify_qa_source_normalization(
    qa: dict[str, Any],
    audit: GlbAudit,
    *,
    label: str,
) -> dict[str, Any] | None:
    input_record = qa.get("assetBinding", {}).get("input")
    source_record = qa.get("source")
    require(isinstance(input_record, dict), f"{label} input binding is missing")
    require(isinstance(source_record, dict), f"{label} source record is missing")

    audited_stats = (audit.faces, audit.normal_loops, audit.degenerate_triangles)
    input_stats = (
        input_record.get("faces"),
        input_record.get("normalLoops"),
        input_record.get("degenerateTriangles"),
    )
    source_stats = (
        source_record.get("faces"),
        source_record.get("normalLoops"),
        source_record.get("degenerateTriangles"),
    )
    if input_stats == audited_stats and source_stats == audited_stats:
        return None

    removed = audit.degenerate_triangles
    require(removed > 0, f"{label} source normalization has no audited degenerates to remove")
    expected_faces = audit.faces - removed
    expected_normal_loops = audit.normal_loops - (removed * 3)
    require(expected_faces > 0, f"{label} source normalization would remove all faces")
    require(
        audit.normal_loops == audit.faces * 3 and expected_normal_loops == expected_faces * 3,
        f"{label} source normalization normal-loop contract is invalid",
    )
    normalized_stats = (expected_faces, expected_normal_loops, 0)
    require(
        input_stats == normalized_stats,
        f"{label} input stats are not an exact audited-degenerate normalization",
    )
    require(
        source_stats == normalized_stats,
        f"{label} source stats are not an exact audited-degenerate normalization",
    )

    processing = qa.get("processing")
    require(isinstance(processing, dict), f"{label} processing record is missing")
    require(
        processing.get("tool") == "blender_headless_pbr_decimate",
        f"{label} source normalization was not recorded by the Blender decimator",
    )
    require(
        processing.get("sourceFacesAfterTriangulate") == expected_faces,
        f"{label} Blender normalized source face count mismatch",
    )
    return {
        "mode": "blender_removed_audited_degenerate_triangles",
        "tool": processing["tool"],
        "audited_source_faces": audit.faces,
        "audited_degenerate_triangles": removed,
        "normalized_source_faces": expected_faces,
        "audited_normal_loops": audit.normal_loops,
        "normalized_normal_loops": expected_normal_loops,
    }


def _verify_qa_asset_stats(
    record: Any,
    audit: GlbAudit,
    *,
    label: str,
    removed_degenerate_triangles: int = 0,
) -> None:
    require(isinstance(record, dict), f"{label} must be an object")
    require(
        isinstance(removed_degenerate_triangles, int)
        and 0 <= removed_degenerate_triangles <= audit.degenerate_triangles,
        f"{label} invalid source normalization count",
    )
    expected_faces = audit.faces - removed_degenerate_triangles
    expected_normal_loops = audit.normal_loops - (removed_degenerate_triangles * 3)
    expected_degenerate_triangles = (
        0 if removed_degenerate_triangles else audit.degenerate_triangles
    )
    require(record.get("sha256") == audit.sha256, f"{label} sha256 mismatch")
    require(record.get("bytes") == audit.size_bytes, f"{label} byte count mismatch")
    require(record.get("faces") == expected_faces, f"{label} face count mismatch")
    require(
        _same_bounds(_qa_bounds(record.get("bounds"), label=label), audit.bounds, tolerance=1e-7),
        f"{label} bounds mismatch",
    )
    require(record.get("pbrMaterials") == audit.pbr_materials, f"{label} PBR count mismatch")
    require(
        record.get("textureImages") == audit.texture_images,
        f"{label} texture image count mismatch",
    )
    require(record.get("uvLayers") == audit.uv_layers, f"{label} UV layer count mismatch")
    require(
        record.get("normalLoops") == expected_normal_loops,
        f"{label} normal loop count mismatch",
    )
    require(
        record.get("glbHasNormals") is audit.glb_has_normals,
        f"{label} normal attribute claim mismatch",
    )
    require(
        record.get("degenerateTriangles") == expected_degenerate_triangles,
        f"{label} degenerate triangle count mismatch",
    )
    require(
        record.get("windingConsistent") is audit.winding_consistent,
        f"{label} winding claim mismatch",
    )


def _verify_qa_reload_stats(
    record: Any,
    audit: GlbAudit,
    *,
    label: str,
    removed_degenerate_triangles: int = 0,
) -> None:
    require(isinstance(record, dict), f"{label} must be an object")
    require(
        isinstance(removed_degenerate_triangles, int)
        and 0 <= removed_degenerate_triangles <= audit.degenerate_triangles,
        f"{label} invalid source normalization count",
    )
    expected_faces = audit.faces - removed_degenerate_triangles
    expected_normal_loops = audit.normal_loops - (removed_degenerate_triangles * 3)
    expected_degenerate_triangles = (
        0 if removed_degenerate_triangles else audit.degenerate_triangles
    )
    require(record.get("sha256") == audit.sha256, f"{label} sha256 mismatch")
    require(record.get("bytes") == audit.size_bytes, f"{label} byte count mismatch")
    require(record.get("vertices") == audit.vertices, f"{label} vertex count mismatch")
    require(record.get("faces") == expected_faces, f"{label} face count mismatch")
    require(
        _same_bounds(_qa_bounds(record.get("bounds"), label=label), audit.bounds, tolerance=1e-7),
        f"{label} bounds mismatch",
    )
    require(record.get("finiteVertices") is audit.finite, f"{label} finite claim mismatch")
    require(
        record.get("normalsFinite") is audit.normals_finite,
        f"{label} finite normals claim mismatch",
    )
    require(
        record.get("degenerateTriangles") == expected_degenerate_triangles,
        f"{label} degenerate triangle count mismatch",
    )
    require(
        record.get("winding", {}).get("consistent") is audit.winding_consistent,
        f"{label} winding claim mismatch",
    )
    require(record.get("uvLayers") == audit.uv_layers, f"{label} UV layer count mismatch")
    require(
        record.get("normalLoops") == expected_normal_loops,
        f"{label} normal loop count mismatch",
    )
    require(
        record.get("materials", {}).get("pbrMaterialCount") == audit.pbr_materials,
        f"{label} PBR count mismatch",
    )
    require(
        record.get("materials", {}).get("textureImageCount") == audit.texture_images,
        f"{label} texture image count mismatch",
    )
    require(
        record.get("glbContract", {}).get("allPrimitivesHaveMaterial") is audit.glb_has_materials,
        f"{label} material attribute claim mismatch",
    )
    require(
        record.get("glbContract", {}).get("allPrimitivesHaveNormals") is audit.glb_has_normals,
        f"{label} normal attribute claim mismatch",
    )
    require(
        record.get("glbContract", {}).get("allPrimitivesHaveTexcoord0") is audit.glb_has_uvs,
        f"{label} UV attribute claim mismatch",
    )


def _verify_object_override(
    verified: VerifiedObject,
    override: dict[str, Any],
    *,
    config_root: Path,
    maximum_faces: int,
) -> VerifiedObject:
    object_id = verified.object_id
    require(
        override.get("placement_policy") == "inherit_scene_fit_receipt",
        f"override placement policy must inherit scene-fit receipt: {object_id}",
    )
    source_ref = override.get("source_asset")
    output_ref = override.get("browser_lod")
    qa_ref = override.get("qa")
    require(isinstance(source_ref, dict), f"override source_asset is missing: {object_id}")
    require(isinstance(output_ref, dict), f"override browser_lod is missing: {object_id}")
    require(isinstance(qa_ref, dict), f"override qa is missing: {object_id}")
    require(
        output_ref.get("coordinate_space") == "scene_local_about_target_pivot",
        f"override coordinate space must be scene-local: {object_id}",
    )
    require(
        output_ref.get("role") == "pbr_visual_logic_authority",
        f"override browser LOD role mismatch: {object_id}",
    )
    require(
        qa_ref.get("required_status") == "passed",
        f"override QA required_status must be passed: {object_id}",
    )

    source_path = resolve_path(
        source_ref.get("path"), relative_to=config_root, label=f"{object_id}.source_asset"
    )
    require(
        source_path == verified.scene_fit_glb.path,
        f"override source path is not the receipt scene-local GLB: {object_id}",
    )
    require(
        source_ref.get("sha256") == verified.scene_fit_glb.sha256,
        f"override source sha256 is not the receipt scene-local GLB: {object_id}",
    )

    qa_path = resolve_path(qa_ref.get("path"), relative_to=config_root, label=f"{object_id}.qa")
    qa_sha256, _ = require_hash(qa_path, qa_ref.get("sha256"), label=f"{object_id}.qa")
    qa = read_json(qa_path)
    require(qa.get("schemaVersion") == 1, f"override QA schema version mismatch: {object_id}")
    require(qa.get("kind") == LOD_QA_KIND, f"override QA kind mismatch: {object_id}")
    require(qa.get("status") == "passed", f"override QA status is not passed: {object_id}")
    failed_gates = [
        name for name in LOD_REQUIRED_QA_GATES if qa.get("qa", {}).get(name) is not True
    ]
    require(not failed_gates, f"override QA required gates failed for {object_id}: {failed_gates}")

    input_binding = qa.get("assetBinding", {}).get("input")
    output_binding = qa.get("assetBinding", {}).get("output")
    require(isinstance(input_binding, dict), f"override QA input binding is missing: {object_id}")
    require(isinstance(output_binding, dict), f"override QA output binding is missing: {object_id}")
    require(
        input_binding.get("sha256") == verified.scene_fit_glb.sha256,
        f"override QA input sha256 is not the receipt scene-local GLB: {object_id}",
    )
    qa_input_path = resolve_path(
        input_binding.get("path"), relative_to=qa_path.parent, label=f"{object_id}.qa.input"
    )
    require(qa_input_path == source_path, f"override QA input path mismatch: {object_id}")
    qa_source_path = resolve_path(
        qa.get("source", {}).get("path"),
        relative_to=qa_path.parent,
        label=f"{object_id}.qa.source",
    )
    require(qa_source_path == source_path, f"override QA source path mismatch: {object_id}")
    source_normalization = _verify_qa_source_normalization(
        qa,
        verified.scene_fit_glb,
        label=f"{object_id}.qa",
    )
    removed_degenerate_triangles = (
        int(source_normalization["audited_degenerate_triangles"])
        if source_normalization is not None
        else 0
    )
    _verify_qa_asset_stats(
        input_binding,
        verified.scene_fit_glb,
        label=f"{object_id}.qa.input",
        removed_degenerate_triangles=removed_degenerate_triangles,
    )
    _verify_qa_reload_stats(
        qa.get("source"),
        verified.scene_fit_glb,
        label=f"{object_id}.qa.source",
        removed_degenerate_triangles=removed_degenerate_triangles,
    )

    output_path = resolve_path(
        output_ref.get("path"), relative_to=config_root, label=f"{object_id}.browser_lod"
    )
    output_sha256, _ = require_hash(
        output_path, output_ref.get("sha256"), label=f"{object_id}.browser_lod"
    )
    require(
        output_binding.get("sha256") == output_sha256,
        f"override QA output sha256 mismatch: {object_id}",
    )
    qa_output_path = resolve_path(
        output_binding.get("path"), relative_to=qa_path.parent, label=f"{object_id}.qa.output"
    )
    require(qa_output_path == output_path, f"override QA output path mismatch: {object_id}")
    qa_reload_path = resolve_path(
        qa.get("output", {}).get("path"),
        relative_to=qa_path.parent,
        label=f"{object_id}.qa.reload",
    )
    require(qa_reload_path == output_path, f"override QA reload path mismatch: {object_id}")
    glb = audit_unified_glb(output_path, maximum_faces=maximum_faces)
    require(
        glb.nondegenerate and glb.degenerate_triangles == 0,
        f"override GLB contains degenerate triangles: {object_id}",
    )
    require(
        glb.pbr_materials > 0
        and glb.texture_images > 0
        and glb.uv_layers > 0
        and glb.glb_has_materials
        and glb.glb_has_uvs
        and glb.glb_has_normals
        and glb.normal_loops > 0,
        f"override GLB lacks required PBR, texture, UV, or normal data: {object_id}",
    )
    target_faces = qa.get("target", {}).get("faces")
    require(
        isinstance(target_faces, int)
        and target_faces > 0
        and glb.faces <= target_faces
        and qa.get("target", {}).get("met") is True,
        f"override GLB does not meet its QA face target: {object_id}",
    )
    _verify_qa_asset_stats(output_binding, glb, label=f"{object_id}.qa.output")
    _verify_qa_reload_stats(qa.get("output"), glb, label=f"{object_id}.qa.reload")

    tolerance = float(qa.get("qa", {}).get("boundsTolerance", -1))
    require(
        math.isfinite(tolerance) and tolerance >= 0,
        f"override QA bounds tolerance is invalid: {object_id}",
    )
    bounds_delta = float(
        np.max(
            np.abs(
                np.asarray(glb.bounds, dtype=np.float64)
                - np.asarray(verified.scene_fit_glb.bounds, dtype=np.float64)
            )
        )
    )
    require(
        bounds_delta <= tolerance + 1e-9,
        f"override GLB bounds drift exceeds QA tolerance: {object_id}",
    )
    require(
        math.isclose(
            float(qa.get("qa", {}).get("maxAbsBoundsDelta", math.inf)),
            bounds_delta,
            rel_tol=0,
            abs_tol=1e-7,
        ),
        f"override QA bounds delta mismatch: {object_id}",
    )
    world_bounds = (
        np.asarray(glb.bounds, dtype=np.float64) + np.asarray(verified.pivot, dtype=np.float64)
    ).tolist()
    return replace(
        verified,
        glb=glb,
        world_bounds=world_bounds,
        override_qa_path=qa_path,
        override_qa_sha256=qa_sha256,
        source_normalization=source_normalization,
    )


def _bbox(bounds: list[list[float]]) -> dict[str, Any]:
    values = np.asarray(bounds, dtype=np.float64)
    return {
        "coordinateFrame": "visual_native",
        "min": values[0].tolist(),
        "max": values[1].tolist(),
        "center": ((values[0] + values[1]) / 2.0).tolist(),
        "extent": (values[1] - values[0]).tolist(),
    }


def _assert_no_proxy_split(value: Any, *, object_id: str) -> None:
    if isinstance(value, dict):
        forbidden = FORBIDDEN_PROXY_FIELDS.intersection(value)
        require(not forbidden, f"legacy proxy fields remain on {object_id}: {sorted(forbidden)}")
        for child in value.values():
            _assert_no_proxy_split(child, object_id=object_id)
    elif isinstance(value, list):
        for child in value:
            _assert_no_proxy_split(child, object_id=object_id)


def _unified_object(base: dict[str, Any], verified: VerifiedObject) -> dict[str, Any]:
    result = copy.deepcopy(base)
    preserved_hierarchy = {key: copy.deepcopy(base.get(key)) for key in HIERARCHY_FIELDS}
    for key in FORBIDDEN_PROXY_FIELDS | {"carve"}:
        result.pop(key, None)
    result["bbox"] = _bbox(verified.world_bounds)
    result["fidelity"] = "direct_trellis_scene_fit_local_qa_candidate"
    result["visualOnly"] = False
    result["independentlyRecognized"] = True
    result["placement"] = {
        "coordinateFrame": "visual_native",
        "sourceCoordinateFrame": "scene_local_about_target_pivot",
        "transform": "scene_local_glb_plus_runtime_pivot",
        "assetCoordinatesBaked": True,
        "pivot": verified.pivot,
        "generatedCenter": [0, 0, 0],
        "scale": [1, 1, 1],
        "rotationEulerDeg": [0, 0, 0],
        "eulerOrder": "XYZ",
    }
    topology = "closed_volume" if verified.glb.watertight else "surface_bvh"
    result["collision"] = {
        "mode": "unified-glb",
        "topology": topology,
        "walkable": bool(base.get("collision", {}).get("walkable", False)),
        "characterCollision": True,
        "asset": {
            "id": f"{verified.object_id}_direct_trellis_scene_local_glb",
            "label": f"{base.get('label', verified.object_id)} scene-local PBR GLB",
            "url": _local_fs_url(verified.glb.path),
            "fileName": verified.glb.path.name,
            "fileType": "glb",
            "format": "gltf-binary",
            "size": verified.glb.size_bytes,
            "sha256": verified.glb.sha256,
            "vertices": verified.glb.vertices,
            "faces": verified.glb.faces,
            "bounds": verified.glb.bounds,
            "finite": verified.glb.finite,
            "nondegenerate": verified.glb.nondegenerate,
            "windingConsistent": verified.glb.winding_consistent,
            "watertight": verified.glb.watertight,
            "logicalRoot": verified.object_id,
            "sourcePath": str(verified.glb.path),
        },
        "gate": {
            "status": "passed",
            "surfaceCollision": "passed",
            "sceneFitReceiptSha256": verified.receipt_sha256,
            "finalBytesHash": "passed",
            "finalBounds": "passed",
            "faceBudget": "passed",
            "candidateBrowserQa": "pending",
            "localQaOnly": True,
        },
    }
    if verified.override_qa_sha256 is not None:
        result["collision"]["gate"].update(
            {
                "browserLodQaSha256": verified.override_qa_sha256,
                "sceneFitSourceGlbSha256": verified.scene_fit_glb.sha256,
                "browserLodOverride": "passed",
            }
        )
    result["limitations"] = [
        "Local QA candidate only; browser review is pending.",
        (
            "The original PGSR scene is intentionally uncarved, so its captured object "
            "remains visible under the completed mesh."
        ),
    ]
    require(
        {key: result.get(key) for key in HIERARCHY_FIELDS} == preserved_hierarchy,
        f"hierarchy metadata changed while materializing {verified.object_id}",
    )
    _assert_no_proxy_split(result, object_id=verified.object_id)
    return result


def _validate_hierarchy(objects: list[dict[str, Any]]) -> None:
    by_id = {item.get("id"): item for item in objects}
    require(
        len(by_id) == len(objects) and None not in by_id,
        "interactive object ids must be unique",
    )
    for object_id, item in by_id.items():
        parent_id = item.get("parentObjectId")
        if parent_id is not None:
            require(parent_id in by_id, f"missing parent {parent_id} for {object_id}")
            require(item.get("movesWithParent") is True, f"child {object_id} must move with parent")
        for child_id in item.get("childObjectIds", []) or []:
            require(child_id in by_id, f"missing child {child_id} for {object_id}")
            require(
                by_id[child_id].get("parentObjectId") == object_id,
                f"parent/child metadata mismatch: {object_id} -> {child_id}",
            )
    for object_id in by_id:
        seen: set[str] = set()
        cursor: str | None = object_id
        while cursor is not None:
            require(cursor not in seen, f"interactive object hierarchy cycle at {object_id}")
            seen.add(cursor)
            cursor = by_id[cursor].get("parentObjectId")


def _logical_ancestor(base: dict[str, Any]) -> dict[str, Any]:
    parent_id = base.get("parentObjectId")
    result = {
        "id": base["id"],
        "label": base.get("label", base["id"]),
        "semanticGranularity": (
            "independent_child_asset" if parent_id is not None else "independent_root_asset"
        ),
        "parentObjectId": parent_id,
        "movesWithParent": parent_id is not None,
        "childObjectIds": [],
        "independentlyMovable": False,
        "logicalHierarchyOnly": True,
        "logicalRole": LOGICAL_HIERARCHY_ROLE,
    }
    for key in ("name", "category", "aliases"):
        if key in base:
            result[key] = copy.deepcopy(base[key])
    return result


def _materialize_runtime_objects(
    base_objects: list[dict[str, Any]],
    verified_objects: dict[str, VerifiedObject],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    by_id = {item.get("id"): item for item in base_objects if isinstance(item, dict)}
    require(
        len(by_id) == len(base_objects), "base interactive object ids are invalid or duplicated"
    )
    adopted_ids = set(verified_objects)
    ancestor_ids: set[str] = set()
    for object_id in adopted_ids:
        cursor = by_id[object_id].get("parentObjectId")
        seen = {object_id}
        while cursor is not None:
            require(cursor in by_id, f"missing hierarchy ancestor {cursor} for {object_id}")
            require(cursor not in seen, f"base hierarchy cycle at {object_id}")
            seen.add(cursor)
            if cursor not in adopted_ids:
                ancestor_ids.add(cursor)
            cursor = by_id[cursor].get("parentObjectId")
    retained_ids = adopted_ids | ancestor_ids
    retained: list[dict[str, Any]] = []
    for base in base_objects:
        object_id = base["id"]
        if object_id not in retained_ids:
            continue
        retained.append(
            _unified_object(base, verified_objects[object_id])
            if object_id in adopted_ids
            else _logical_ancestor(base)
        )
    child_ids: dict[str, list[str]] = {object_id: [] for object_id in retained_ids}
    for item in retained:
        parent_id = item.get("parentObjectId")
        if parent_id is not None:
            child_ids[parent_id].append(item["id"])
    for item in retained:
        item["childObjectIds"] = child_ids[item["id"]]
    removed = [item["id"] for item in base_objects if item["id"] not in retained_ids]
    return retained, [item["id"] for item in retained if item["id"] in ancestor_ids], removed


def _assert_runtime_interactive_contract(
    objects: list[dict[str, Any]], *, adopted_ids: set[str]
) -> None:
    for item in objects:
        if item["id"] in adopted_ids:
            require(
                item.get("logicalHierarchyOnly") is not True, "adopted object became logical-only"
            )
            continue
        require(
            item.get("logicalHierarchyOnly") is True, "unadopted runtime object is not logical-only"
        )
        require(item.get("logicalRole") == LOGICAL_HIERARCHY_ROLE, "logical role is invalid")
        forbidden = {
            "placement",
            "collision",
            "visual",
            "renderAsset",
            "colliderProxy",
            "interaction",
            "carve",
            "sourceAnchor",
        }.intersection(item)
        require(
            not forbidden, f"logical ancestor retains runtime asset fields: {sorted(forbidden)}"
        )
        require(item.get("independentlyMovable") is False, "logical ancestor remains movable")
    serialized = json.dumps(objects, ensure_ascii=True, sort_keys=True).lower()
    require("/chunks/" not in serialized, "old interactive chunk references remain")
    require(
        "pending_strict_clean_scene_world" not in serialized,
        "old strict-clean interactive gate remains",
    )


def _static_visual(audit: PgsrAudit) -> dict[str, Any]:
    return {
        "id": "direct_original_uncarved_pgsr_full_scene",
        "label": "Original uncarved PGSR full-scene Gaussians (local QA)",
        "url": _local_fs_url(audit.path),
        "fileName": audit.path.name,
        "fileType": "ply",
        "format": "graphdeco-3dgs-ply",
        "renderer": "spark",
        "size": audit.size_bytes,
        "sha256": audit.sha256,
        "vertexCount": audit.vertex_count,
        "vertexProperties": audit.properties,
        "bbox": {"min": audit.bounds[0], "max": audit.bounds[1]},
        "sourcePath": str(audit.path),
        "byteForBytePreserved": True,
        "carved": False,
        "cleanPlateApplied": False,
    }


def _static_collider(audit: StaticColliderAudit) -> dict[str, Any]:
    return {
        "id": "direct_original_uncarved_scene_mesh",
        "label": "Original uncarved scene TSDF mesh (local QA)",
        "url": _local_fs_url(audit.path),
        "fileName": audit.path.name,
        "fileType": "ply",
        "format": "open3d-binary-little-endian-triangle-mesh-ply",
        "size": audit.size_bytes,
        "sha256": audit.sha256,
        "vertexCount": audit.vertex_count,
        "faceCount": audit.face_count,
        "vertexProperties": audit.vertex_properties,
        "vertexPositionType": audit.vertex_position_type,
        "faceIndexType": audit.face_index_type,
        "bbox": {"min": audit.bounds[0], "max": audit.bounds[1]},
        "finite": audit.finite,
        "faceIndicesValid": audit.face_indices_valid,
        "degenerateTriangles": audit.degenerate_triangles,
        "sourcePath": str(audit.path),
        "transportMode": "local_fs_single_file",
        "byteForBytePreserved": True,
        "carved": False,
        "cleanPlateApplied": False,
    }


def _assert_no_inherited_collider_transport(manifest: dict[str, Any]) -> None:
    assets = manifest.get("assets", {})
    require(
        set(key for key in assets if key.lower().startswith("collider")) == {"collider"},
        "legacy collider asset aliases remain",
    )
    collider = assets.get("collider", {})
    require("parts" not in collider, "static collider must not inherit chunk transport")
    collision_world = manifest.get("collisionWorld", {})
    require(
        collision_world.get("sceneAssetKey") == "collider",
        "collisionWorld must use original collider",
    )
    serialized = json.dumps(
        {"assets": assets, "collisionWorld": collision_world},
        ensure_ascii=True,
        sort_keys=True,
    ).lower()
    require("strict_clean" not in serialized, "strict-clean collider metadata remains")
    require("/chunks/" not in serialized, "strict-clean collider chunk references remain")


def build_candidate_manifest(
    base: dict[str, Any],
    *,
    base_sha256: str,
    candidate_version: str,
    static_scene: PgsrAudit,
    static_collider: StaticColliderAudit,
    verified_objects: dict[str, VerifiedObject],
) -> dict[str, Any]:
    require(base.get("contract") == "video2world-web-manifest-1.0.0", "base contract mismatch")
    require(base.get("schemaVersion") == 1, "base schema version mismatch")
    interactive = base.get("interactiveObjects")
    require(isinstance(interactive, list) and interactive, "base interactiveObjects are missing")
    by_id = {item.get("id"): item for item in interactive if isinstance(item, dict)}
    require(len(by_id) == len(interactive), "base interactive object ids are invalid or duplicated")
    missing = sorted(set(verified_objects).difference(by_id))
    require(not missing, f"scene-fit objects are absent from base manifest: {missing}")

    manifest = copy.deepcopy(base)
    manifest["version"] = candidate_version
    # These describe the superseded carved/chunked production build and would make the
    # direct original-scene candidate claim two incompatible representations.
    manifest.pop("interactiveObjectBuild", None)
    manifest.pop("productionBuild", None)
    assets = manifest.setdefault("assets", {})
    for key in list(assets):
        if key.lower().startswith("collider"):
            del assets[key]
    assets["visual"] = _static_visual(static_scene)
    assets["collider"] = _static_collider(static_collider)
    manifest["collisionWorld"] = {
        "baselineBoundsAssetKey": "collider",
        "sceneAssetKey": "collider",
        "objectFaceRemovalRequired": False,
        "replacementMode": "original_uncarved_scene_mesh",
        "gate": {
            "status": "passed_local_qa_input_audit",
            "byteForBytePreserved": True,
            "carved": False,
            "cleanPlateApplied": False,
            "vertexCount": static_collider.vertex_count,
            "faceCount": static_collider.face_count,
            "finite": static_collider.finite,
            "faceIndicesValid": static_collider.face_indices_valid,
            "degenerateTriangles": static_collider.degenerate_triangles,
        },
        "sceneAssetTransport": {
            "mode": "local_fs_single_file",
            "url": _local_fs_url(static_collider.path),
            "sha256": static_collider.sha256,
            "bytes": static_collider.size_bytes,
            "byteForBytePreserved": True,
        },
    }
    source_world = manifest.get("sourceWorld")
    if isinstance(source_world, dict):
        source_world["adoptionMode"] = "direct_original_uncarved_scene_local_qa_only"
    runtime_objects, logical_ancestor_ids, removed_object_ids = _materialize_runtime_objects(
        manifest["interactiveObjects"], verified_objects
    )
    manifest["interactiveObjects"] = runtime_objects
    _validate_hierarchy(manifest["interactiveObjects"])
    _assert_runtime_interactive_contract(
        manifest["interactiveObjects"], adopted_ids=set(verified_objects)
    )
    initial_state = manifest.setdefault("initialState", {})
    if initial_state.get("cameraFocusObjectId") not in verified_objects:
        initial_state["cameraFocusObjectId"] = next(iter(verified_objects))

    knowledge = manifest.get("sceneKnowledge", {}).get("objects")
    if isinstance(knowledge, list):
        for item in knowledge:
            object_id = item.get("id") if isinstance(item, dict) else None
            if object_id in verified_objects:
                item["bbox"] = _bbox(verified_objects[object_id].world_bounds)

    object_ids = list(verified_objects)
    manifest["candidateBuild"] = {
        "status": "local_qa_candidate_materialized",
        "promotionAllowed": False,
        "completionClaim": "direct_trellis_scene_fit_without_clean_plate_local_qa_only",
        "representation": (
            "one_scene_local_pbr_glb_per_object_for_visual_logic_selection_and_collision"
        ),
        "objectIds": object_ids,
        "unadoptedObjectPolicy": UNADOPTED_OBJECT_POLICY,
        "logicalAncestorObjectIds": logical_ancestor_ids,
        "removedUnadoptedObjectIds": removed_object_ids,
        "runtimeInteractiveObjectIds": [item["id"] for item in runtime_objects],
        "originalStaticSceneIsSoleRepresentationForRemovedObjects": True,
        "forbiddenProxyFields": sorted(FORBIDDEN_PROXY_FIELDS),
        "forbiddenProxyFieldsScope": "adopted_objects",
        "objectHierarchyMetadata": "preserved_from_base_manifest",
        "staticScene": {
            "mode": "original_uncarved_pgsr_full_scene",
            "sha256": static_scene.sha256,
            "visual": {
                "role": "original_uncarved_pgsr_full_scene",
                "sha256": static_scene.sha256,
                "bytes": static_scene.size_bytes,
                "vertexCount": static_scene.vertex_count,
            },
            "collider": {
                "role": "original_uncarved_scene_mesh",
                "sha256": static_collider.sha256,
                "bytes": static_collider.size_bytes,
                "vertexCount": static_collider.vertex_count,
                "faceCount": static_collider.face_count,
            },
            "byteForBytePreserved": True,
            "carvePerformed": False,
            "cleanPlateGenerated": False,
            "knownOverlap": "captured_objects_remain_in_the_static_scene",
        },
        "inputs": {
            "baseManifestSha256": base_sha256,
            "staticVisualSha256": static_scene.sha256,
            "staticColliderSha256": static_collider.sha256,
            "sceneFitReceipts": {
                object_id: verified_objects[object_id].receipt_sha256 for object_id in object_ids
            },
            "sceneLocalGlbs": {
                object_id: verified_objects[object_id].glb.sha256 for object_id in object_ids
            },
        },
        "localAssetTransport": "/@fs",
        "promotionBlockers": [
            "local_fs_asset_urls_are_not_deployable",
            "candidate_browser_qa_pending",
            "original_scene_object_overlap_is_intentional",
        ],
    }
    override_evidence = {
        object_id: {
            "sceneFitSourceGlbSha256": item.scene_fit_glb.sha256,
            "browserLodGlbSha256": item.glb.sha256,
            "browserLodQaSha256": item.override_qa_sha256,
            "placementPolicy": "inherit_scene_fit_receipt",
        }
        for object_id, item in verified_objects.items()
        if item.override_qa_sha256 is not None
    }
    if override_evidence:
        manifest["candidateBuild"]["objectOverrides"] = override_evidence
    _assert_no_inherited_collider_transport(manifest)
    serialized_manifest = json.dumps(manifest, ensure_ascii=True, sort_keys=True).lower()
    require("/chunks/" not in serialized_manifest, "legacy chunk references remain in candidate")
    require(
        "pending_strict_clean_scene_world" not in serialized_manifest,
        "legacy strict-clean gate remains in candidate",
    )
    return manifest


def _snapshot_public_manifests(project_root: Path) -> dict[str, tuple[str, int]]:
    public_root = (project_root / "web/public").resolve()
    if not public_root.is_dir():
        return {}
    return {
        str(path.resolve()): sha256_file(path)
        for path in sorted(public_root.rglob("manifest*.json"))
        if path.is_file()
    }


def _assert_local_output(project_root: Path, output: Path, report: Path) -> None:
    public_root = (project_root / "web/public").resolve()
    for path, label in ((output.resolve(), "candidate manifest"), (report.resolve(), "report")):
        require(
            path != public_root and public_root not in path.parents,
            f"{label} must stay outside web/public",
        )
    require(output.resolve() != report.resolve(), "candidate manifest and report must differ")


def _assert_outputs_do_not_overwrite_inputs(
    *,
    output: Path,
    report: Path,
    input_paths: set[Path],
) -> None:
    destinations = {output.resolve(), report.resolve()}
    aliases = sorted(str(path) for path in destinations.intersection(input_paths))
    require(not aliases, f"candidate outputs must not overwrite inputs: {aliases}")


def _load_config(
    config_path: Path,
) -> tuple[
    dict[str, Any],
    Path,
    str,
    Path,
    str,
    Path,
    str,
    list[tuple[Path, str]],
    dict[str, dict[str, Any]],
    int,
]:
    config = read_json(config_path)
    require(config.get("schema_version") == 1, "candidate config schema version mismatch")
    require(config.get("kind") == CONFIG_KIND, "candidate config kind mismatch")
    candidate_version = config.get("candidate_version")
    require(
        isinstance(candidate_version, str) and candidate_version,
        "candidate_version must be a non-empty string",
    )
    require(
        config.get("unadopted_object_policy") == UNADOPTED_OBJECT_POLICY,
        f"unadopted_object_policy must be {UNADOPTED_OBJECT_POLICY}",
    )
    config_root = config_path.parent
    base_ref = config.get("base_manifest")
    static_ref = config.get("static_scene")
    collider_ref = config.get("static_collider")
    require(isinstance(base_ref, dict), "base_manifest reference is missing")
    require(isinstance(static_ref, dict), "static_scene reference is missing")
    require(isinstance(collider_ref, dict), "static_collider reference is missing")
    require(
        static_ref.get("role") == "original_uncarved_pgsr_full_scene",
        "static_scene.role must explicitly identify the original uncarved PGSR scene",
    )
    require(
        collider_ref.get("role") == "original_uncarved_scene_mesh",
        "static_collider.role must explicitly identify the original uncarved scene mesh",
    )
    base_path = resolve_path(base_ref.get("path"), relative_to=config_root, label="base_manifest")
    static_path = resolve_path(
        static_ref.get("path"), relative_to=config_root, label="static_scene"
    )
    collider_path = resolve_path(
        collider_ref.get("path"), relative_to=config_root, label="static_collider"
    )
    object_refs = config.get("objects")
    require(isinstance(object_refs, list) and object_refs, "objects must be a non-empty list")
    receipt_refs: list[tuple[Path, str]] = []
    declared_ids: set[str] = set()
    for index, item in enumerate(object_refs):
        require(isinstance(item, dict), f"objects[{index}] must be an object")
        object_id = item.get("object_id")
        require(isinstance(object_id, str) and object_id, f"objects[{index}].object_id is missing")
        require(object_id not in declared_ids, f"duplicate object_id in config: {object_id}")
        declared_ids.add(object_id)
        receipt_ref = item.get("scene_fit_receipt")
        require(isinstance(receipt_ref, dict), f"scene-fit receipt reference missing: {object_id}")
        receipt_path = resolve_path(
            receipt_ref.get("path"),
            relative_to=config_root,
            label=f"{object_id}.scene_fit_receipt",
        )
        receipt_sha = receipt_ref.get("sha256")
        require(isinstance(receipt_sha, str), f"scene-fit receipt sha256 missing: {object_id}")
        receipt_refs.append((receipt_path, receipt_sha))
    overrides = config.get("object_overrides", {})
    require(isinstance(overrides, dict), "object_overrides must be an object")
    unknown_overrides = sorted(set(overrides).difference(declared_ids))
    require(
        not unknown_overrides,
        f"object_overrides contains undeclared objects: {unknown_overrides}",
    )
    require(
        all(isinstance(value, dict) for value in overrides.values()),
        "each object_overrides value must be an object",
    )
    maximum_faces = int(config.get("maximum_collision_faces", RUNTIME_MAX_COLLISION_FACES))
    require(
        0 < maximum_faces <= RUNTIME_MAX_COLLISION_FACES,
        f"maximum_collision_faces must be between 1 and {RUNTIME_MAX_COLLISION_FACES}",
    )
    return (
        config,
        base_path,
        str(base_ref.get("sha256")),
        static_path,
        str(static_ref.get("sha256")),
        collider_path,
        str(collider_ref.get("sha256")),
        receipt_refs,
        overrides,
        maximum_faces,
    )


def _object_report(item: VerifiedObject) -> dict[str, Any]:
    report = {
        "scene_fit_receipt": {
            "path": str(item.receipt_path),
            "sha256": item.receipt_sha256,
        },
        "scene_local_glb": {
            "path": str(item.glb.path),
            "url": _local_fs_url(item.glb.path),
            "sha256": item.glb.sha256,
            "bytes": item.glb.size_bytes,
            "vertices": item.glb.vertices,
            "faces": item.glb.faces,
            "bounds": item.glb.bounds,
            "material_types": item.glb.material_types,
        },
        "runtime_pivot": item.pivot,
        "world_bounds": item.world_bounds,
        "unified_visual_logic_collision": True,
        "legacy_proxy_fields_absent": True,
    }
    if item.override_qa_path is not None:
        report["scene_fit_source_glb"] = {
            "path": str(item.scene_fit_glb.path),
            "sha256": item.scene_fit_glb.sha256,
            "bytes": item.scene_fit_glb.size_bytes,
            "faces": item.scene_fit_glb.faces,
            "bounds": item.scene_fit_glb.bounds,
        }
        report["browser_lod_override"] = {
            "qa_path": str(item.override_qa_path),
            "qa_sha256": item.override_qa_sha256,
            "coordinate_space": "scene_local_about_target_pivot",
            "placement_policy": "inherit_scene_fit_receipt",
        }
        if item.source_normalization is not None:
            report["browser_lod_override"]["source_normalization"] = item.source_normalization
    return report


def materialize_candidate(
    *,
    project_root: Path,
    config_path: Path,
    output_manifest: Path,
    report_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    project_root = project_root.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    output_manifest = output_manifest.expanduser().resolve()
    report_path = report_path.expanduser().resolve()
    _assert_local_output(project_root, output_manifest, report_path)
    protected_before = _snapshot_public_manifests(project_root)
    config_sha256, _ = sha256_file(config_path)
    (
        config,
        base_path,
        expected_base_sha,
        static_path,
        expected_static_sha,
        collider_path,
        expected_collider_sha,
        receipt_refs,
        overrides,
        maximum_faces,
    ) = _load_config(config_path)
    protected_inputs = {
        config_path,
        base_path,
        static_path,
        collider_path,
        *(receipt_path for receipt_path, _receipt_sha in receipt_refs),
    }
    for override in overrides.values():
        for key in ("source_asset", "browser_lod", "qa"):
            record = override.get(key)
            if not isinstance(record, dict):
                continue
            value = record.get("path")
            if not isinstance(value, str) or not value:
                continue
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = config_path.parent / path
            protected_inputs.add(path.resolve())
    _assert_outputs_do_not_overwrite_inputs(
        output=output_manifest,
        report=report_path,
        input_paths=protected_inputs,
    )
    base_sha256, _ = require_hash(base_path, expected_base_sha, label="base_manifest")
    static_scene = audit_pgsr_ply(static_path, expected_sha256=expected_static_sha)
    static_collider = audit_static_collider_ply(
        collider_path, expected_sha256=expected_collider_sha
    )
    verified_objects: dict[str, VerifiedObject] = {}
    declared_object_ids = [item["object_id"] for item in config["objects"]]
    for declared_id, (receipt_path, receipt_sha) in zip(
        declared_object_ids, receipt_refs, strict=True
    ):
        verified = _verify_receipt(
            receipt_path,
            expected_sha256=receipt_sha,
            maximum_faces=maximum_faces,
            allow_runtime_asset_override=declared_id in overrides,
        )
        require(
            verified.object_id == declared_id,
            f"receipt object_id mismatch: {verified.object_id} != {declared_id}",
        )
        if declared_id in overrides:
            verified = _verify_object_override(
                verified,
                overrides[declared_id],
                config_root=config_path.parent,
                maximum_faces=maximum_faces,
            )
        verified_objects[declared_id] = verified

    manifest = build_candidate_manifest(
        read_json(base_path),
        base_sha256=base_sha256,
        candidate_version=config["candidate_version"],
        static_scene=static_scene,
        static_collider=static_collider,
        verified_objects=verified_objects,
    )
    atomic_write_json(output_manifest, manifest)
    manifest_sha256, manifest_size = sha256_file(output_manifest)
    protected_after_manifest = _snapshot_public_manifests(project_root)
    require(protected_after_manifest == protected_before, "a public manifest changed")

    report = {
        "schema_version": 1,
        "kind": REPORT_KIND,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "local_qa_candidate_materialized",
        "promotion_allowed": False,
        "unadopted_object_policy": {
            "mode": UNADOPTED_OBJECT_POLICY,
            "logical_ancestor_object_ids": manifest["candidateBuild"]["logicalAncestorObjectIds"],
            "removed_object_ids": manifest["candidateBuild"]["removedUnadoptedObjectIds"],
            "original_static_scene_is_sole_representation": True,
        },
        "claim_scope": (
            "The original PGSR visual PLY and original TSDF collider PLY are referenced "
            "byte-for-byte without carve or clean plate. Each adopted scene-local PBR GLB is "
            "the unified visual, logical, selection, and collision asset. This is not a "
            "deployable or promoted manifest."
        ),
        "inputs": {
            "config": {"path": str(config_path), "sha256": config_sha256},
            "base_manifest": {"path": str(base_path), "sha256": base_sha256},
            "static_scene": {
                "path": str(static_scene.path),
                "sha256": static_scene.sha256,
                "bytes": static_scene.size_bytes,
                "vertex_count": static_scene.vertex_count,
                "bounds": static_scene.bounds,
                "byte_for_byte_preserved": True,
            },
            "static_collider": {
                "path": str(static_collider.path),
                "sha256": static_collider.sha256,
                "bytes": static_collider.size_bytes,
                "vertex_count": static_collider.vertex_count,
                "face_count": static_collider.face_count,
                "bounds": static_collider.bounds,
                "finite": static_collider.finite,
                "face_indices_valid": static_collider.face_indices_valid,
                "degenerate_triangles": static_collider.degenerate_triangles,
                "byte_for_byte_preserved": True,
                "carved": False,
                "clean_plate_applied": False,
            },
        },
        "objects": {
            object_id: _object_report(item) for object_id, item in verified_objects.items()
        },
        "output_manifest": {
            "path": str(output_manifest),
            "sha256": manifest_sha256,
            "bytes": manifest_size,
            "local_qa_only": True,
        },
        "clean_plate": {"generated": False, "consumed": False},
        "carve": {"performed": False},
        "public_manifests": {
            "mutated": False,
            "count": len(protected_before),
            "sha256_by_path": {path: digest for path, (digest, _size) in protected_before.items()},
        },
        "promotion_blockers": manifest["candidateBuild"]["promotionBlockers"],
    }
    atomic_write_json(report_path, report)
    require(
        _snapshot_public_manifests(project_root) == protected_before,
        "a public manifest changed",
    )
    return manifest, report


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest, report = materialize_candidate(
        project_root=args.project_root,
        config_path=args.config,
        output_manifest=args.output_manifest,
        report_path=args.report,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "version": manifest["version"],
                "manifest": str(args.output_manifest.expanduser().resolve()),
                "report": str(args.report.expanduser().resolve()),
                "promotion_allowed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
