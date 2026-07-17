#!/usr/bin/env python3
"""Materialize the Bedroom 4 unified-PBR pillow candidate from verified evidence."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import struct
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.materialize_bedroom4_world import materialize_file
from video2world.hashing import atomic_write_json, sha256_file

CANDIDATE_VERSION = "bedroom4-candidate-unified-pillow-20260717-static-recarve-semantic-covariance"
CANDIDATE_CREATED_AT = "2026-07-17T06:13:53+08:00"
OLD_ENSEMBLE_ID = "sam3_pillow_01"
FRONT_ID = "sam3_pillow_front"
REAR_IDS = ("sam3_pillow_left", "sam3_pillow_right")
FRONT_PIVOT = (-0.9132125483707731, -0.1424510127463641, 14.760714277636716)
FRONT_COLLIDER_CARVE_MARGIN = 0.08
FRONT_SUPPORT_BAND = 0.10
FRONT_SUPPORT_MIN_NORMAL_ALIGNMENT = 0.75
COLLIDER_CHUNK_SIZE = 1024 * 1024
COLLIDER_CHUNK_STEM = "collider_bedroom4_tsdf_static_carved_unified_pillow_ply"
COLLIDER_CHUNK_PUBLIC_PREFIX = "./worlds/bedroom4/chunks"
STATIC_VISUAL_CHUNK_STEM = "visual_bedroom4_static_front_recarve_semantic_covariance"
STATIC_VISUAL_CHUNK_PUBLIC_PREFIX = "./worlds/bedroom4/chunks"
STATIC_VISUAL_EXPECTED = {
    "sha256": "7f8a7d9a1fc07cee34754469d5fffe95d90640bfa9d5883b4432e83698e55c0d",
    "size": 203_804_211,
    "vertices": 821_785,
    "input_vertices": 823_391,
    "removed_vertices": 1_606,
    "parts": 195,
}
STABLE_BASE_MANIFEST_SHA256 = "3805f0e5bab09add424b3b78f9349cd2eca6d1262777ef683e13cda07695e82b"
CANONICAL_SIX_VIEW_REVIEW_SHA256 = (
    "c6e374bec1d4014dd5f61a788379b2ab739bbbf7ea4323db0a3fc58057f01664"
)
STATIC_RECARVE_REPORT_SHA256 = "8d119fb87b329b9abd122335b8d1c7d0b75be0f9bd53fe64b0a28ff3be4dc36b"
STATIC_RECARVE_BROWSER_REVIEW_SHA256 = (
    "e97d5554d1508b66722e946ddf0206297295c7dae5c316c41346b873ca15df2c"
)
FRONT_EXPECTED = {
    "sha256": "9d84ab4561d7c017c24c07a94abe5400073b0fafb97893a8a1eafca320c391d6",
    "size": 3_979_924,
    "vertices": 60_237,
    "faces": 97_082,
}
REAR_EXPECTED = {
    "sam3_pillow_left": {
        "sha256": "b8c45454d887a77a99b17558c0fa0f0e49d81ab00099e346aeffc940eacdb400",
        "size": 1_346_659,
        "points": 89_763,
    },
    "sam3_pillow_right": {
        "sha256": "09537489887ad6ba3cbccf9cbaae99142eccbb4a65ab82b8829d4fe140bfc63f",
        "size": 737_659,
        "points": 49_163,
    },
}


@dataclass(frozen=True)
class RgbPlyStats:
    point_count: int
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]
    header_bytes: int
    stride_bytes: int


@dataclass(frozen=True)
class BinaryTriangleMesh:
    label: str
    data: bytes
    header: str
    data_offset: int
    face_offset: int
    vertex_count: int
    face_count: int
    vertex_stride: int
    positions: array


@dataclass(frozen=True)
class CandidateEvidence:
    base_sha256: str
    scene_fit_sha256: str
    split_report_sha256: str
    clean_plate_review_sha256: str
    canonical_six_view_review_sha256: str
    front_visual_review_sha256: str
    static_recarve_report_sha256: str
    static_recarve_browser_review_sha256: str
    front_source: Path
    front_mesh: dict[str, Any]
    rear_sources: dict[str, Path]
    rear_records: dict[str, dict[str, Any]]
    front_split_aabb: dict[str, list[float]]
    base_collider_asset: dict[str, Any]
    clean_plate_review: dict[str, Any]
    canonical_six_view_review: dict[str, Any]
    front_visual_review: dict[str, Any]
    static_recarve_report: dict[str, Any]
    static_recarve_browser_review: dict[str, Any]
    world_up: tuple[float, float, float]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def preserve_stable_manifest(
    *, base_manifest: Path, promoted_manifest: Path, stable_manifest: Path
) -> None:
    """Preserve the fixed stable manifest before replacing the canonical demo manifest."""

    base_bytes = base_manifest.read_bytes()
    require(
        hashlib.sha256(base_bytes).hexdigest() == STABLE_BASE_MANIFEST_SHA256,
        "base manifest is not the fixed stable baseline",
    )
    if stable_manifest.exists():
        stable_bytes = stable_manifest.read_bytes()
        require(
            hashlib.sha256(stable_bytes).hexdigest() == STABLE_BASE_MANIFEST_SHA256,
            "stable manifest alias does not point to the fixed baseline",
        )
        require(stable_bytes == base_bytes, "stable manifest alias is not byte-identical to base")
        return

    require(promoted_manifest.is_file(), "canonical manifest is missing before stable preservation")
    current_bytes = promoted_manifest.read_bytes()
    require(
        hashlib.sha256(current_bytes).hexdigest() == STABLE_BASE_MANIFEST_SHA256,
        "canonical manifest is not the fixed stable baseline before promotion",
    )
    require(current_bytes == base_bytes, "canonical manifest differs byte-for-byte from base")
    atomic_write_bytes(stable_manifest, current_bytes)


def _close_vector(actual: list[float], expected: tuple[float, ...] | list[float]) -> bool:
    return len(actual) == len(expected) and all(
        math.isclose(float(left), float(right), rel_tol=0, abs_tol=1e-6)
        for left, right in zip(actual, expected, strict=True)
    )


def inspect_rgb_ply(path: Path) -> RgbPlyStats:
    """Validate the split RGB PLY payload and return its observed geometry."""

    with path.open("rb") as stream:
        header_lines: list[str] = []
        header_bytes = 0
        while True:
            raw_line = stream.readline()
            require(raw_line, f"PLY header is truncated: {path}")
            header_bytes += len(raw_line)
            require(header_bytes <= 1024 * 1024, f"PLY header exceeds one MiB: {path}")
            try:
                line = raw_line.decode("ascii").strip()
            except UnicodeDecodeError as exc:
                raise RuntimeError(f"PLY header is not ASCII: {path}") from exc
            header_lines.append(line)
            if line == "end_header":
                break

        require(header_lines[0] == "ply", f"invalid PLY magic: {path}")
        require(
            "format binary_little_endian 1.0" in header_lines,
            f"RGB split PLY must be binary little endian: {path}",
        )
        vertex_lines = [line for line in header_lines if line.startswith("element vertex ")]
        require(len(vertex_lines) == 1, f"RGB split PLY must declare one vertex element: {path}")
        point_count = int(vertex_lines[0].split()[2])
        expected_properties = [
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
        properties = [line for line in header_lines if line.startswith("property ")]
        require(properties == expected_properties, f"unexpected RGB split PLY properties: {path}")

        stride = struct.calcsize("<fffBBB")
        payload = stream.read()
        require(
            len(payload) == point_count * stride,
            f"RGB split PLY payload size does not match its vertex count: {path}",
        )
        minimum = [math.inf, math.inf, math.inf]
        maximum = [-math.inf, -math.inf, -math.inf]
        for x, y, z, _red, _green, _blue in struct.iter_unpack("<fffBBB", payload):
            require(
                all(math.isfinite(value) for value in (x, y, z)), f"non-finite PLY point: {path}"
            )
            for axis, value in enumerate((x, y, z)):
                minimum[axis] = min(minimum[axis], value)
                maximum[axis] = max(maximum[axis], value)
    require(point_count > 0, f"RGB split PLY contains no points: {path}")
    return RgbPlyStats(
        point_count=point_count,
        minimum=tuple(minimum),
        maximum=tuple(maximum),
        header_bytes=header_bytes,
        stride_bytes=stride,
    )


def parse_binary_triangle_mesh_ply(data: bytes, label: str) -> BinaryTriangleMesh:
    marker = b"end_header"
    marker_index = data.find(marker)
    require(marker_index >= 0, f"triangle PLY has no end_header: {label}")
    data_offset = marker_index + len(marker)
    while data_offset < len(data) and data[data_offset] in {10, 13}:
        data_offset += 1
    try:
        header = data[:data_offset].decode("ascii")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"triangle PLY header is not ASCII: {label}") from exc
    require(
        "format binary_little_endian 1.0" in header,
        f"triangle PLY must be binary little endian: {label}",
    )

    scalar_sizes = {
        "char": 1,
        "int8": 1,
        "uchar": 1,
        "uint8": 1,
        "short": 2,
        "int16": 2,
        "ushort": 2,
        "uint16": 2,
        "int": 4,
        "int32": 4,
        "uint": 4,
        "uint32": 4,
        "float": 4,
        "float32": 4,
        "double": 8,
        "float64": 8,
    }
    element: str | None = None
    vertex_count = 0
    face_count = 0
    vertex_stride = 0
    vertex_properties: dict[str, tuple[str, int]] = {}
    face_list: tuple[str, str, str] | None = None
    for line in header.splitlines():
        fields = line.strip().split()
        if not fields:
            continue
        if fields[0] == "element":
            require(len(fields) == 3, f"malformed PLY element: {label}")
            element = fields[1]
            if element == "vertex":
                vertex_count = int(fields[2])
            elif element == "face":
                face_count = int(fields[2])
            continue
        if fields[0] != "property":
            continue
        if element == "vertex":
            require(
                len(fields) == 3 and fields[1] != "list", f"unsupported vertex property: {label}"
            )
            scalar_type = fields[1]
            require(scalar_type in scalar_sizes, f"unsupported vertex type {scalar_type}: {label}")
            vertex_properties[fields[2]] = (scalar_type, vertex_stride)
            vertex_stride += scalar_sizes[scalar_type]
        elif element == "face":
            require(len(fields) == 5 and fields[1] == "list", f"unsupported face property: {label}")
            face_list = (fields[2], fields[3], fields[4])
    require(vertex_count > 0 and face_count > 0, f"triangle PLY is empty: {label}")
    require(face_list is not None, f"triangle PLY has no face list: {label}")
    require(
        face_list[0] in {"uchar", "uint8"}
        and face_list[1] in {"uint", "uint32"}
        and face_list[2] in {"vertex_index", "vertex_indices"},
        f"triangle PLY must use list uchar uint vertex_indices: {label}",
    )
    for axis in ("x", "y", "z"):
        require(
            vertex_properties.get(axis, (None,))[0] == "double",
            f"triangle PLY must use double {axis}: {label}",
        )
    vertex_payload_bytes = vertex_count * vertex_stride
    face_offset = data_offset + vertex_payload_bytes
    require(
        face_offset + face_count * 13 == len(data),
        f"triangle PLY payload size differs from fixed triangle declarations: {label}",
    )
    positions = array("d")
    axis_offsets = [vertex_properties[axis][1] for axis in ("x", "y", "z")]
    for vertex in range(vertex_count):
        offset = data_offset + vertex * vertex_stride
        values = [struct.unpack_from("<d", data, offset + axis)[0] for axis in axis_offsets]
        require(
            all(math.isfinite(value) for value in values), f"non-finite collider vertex: {label}"
        )
        positions.extend(values)
    return BinaryTriangleMesh(
        label=label,
        data=data,
        header=header,
        data_offset=data_offset,
        face_offset=face_offset,
        vertex_count=vertex_count,
        face_count=face_count,
        vertex_stride=vertex_stride,
        positions=positions,
    )


def _face_intersects_bounds(
    mesh: BinaryTriangleMesh,
    face_index: int,
    bounds: dict[str, list[float]],
) -> bool:
    offset = mesh.face_offset + face_index * 13
    require(mesh.data[offset] == 3, f"non-triangle face {face_index}: {mesh.label}")
    indices = struct.unpack_from("<III", mesh.data, offset + 1)
    require(
        all(index < mesh.vertex_count for index in indices), f"invalid face index: {mesh.label}"
    )
    for axis in range(3):
        values = [mesh.positions[index * 3 + axis] for index in indices]
        if max(values) < bounds["min"][axis] or min(values) > bounds["max"][axis]:
            return False
    return True


def _face_matches_support_policy(
    mesh: BinaryTriangleMesh,
    face_index: int,
    policy: dict[str, Any],
) -> bool:
    offset = mesh.face_offset + face_index * 13
    indices = struct.unpack_from("<III", mesh.data, offset + 1)
    vertices = [
        tuple(float(mesh.positions[index * 3 + axis]) for axis in range(3)) for index in indices
    ]
    ab = tuple(vertices[1][axis] - vertices[0][axis] for axis in range(3))
    ac = tuple(vertices[2][axis] - vertices[0][axis] for axis in range(3))
    normal = (
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    )
    normal_length = math.sqrt(sum(value * value for value in normal))
    if normal_length <= 1e-12:
        return False
    world_up = tuple(float(value) for value in policy["worldUp"])
    alignment = abs(sum(normal[axis] * world_up[axis] for axis in range(3)) / normal_length)
    if alignment < float(policy["minimumNormalAlignment"]):
        return False
    down = tuple(-value for value in world_up)
    centroid = tuple(sum(vertex[axis] for vertex in vertices) / 3 for axis in range(3))
    down_coordinate = sum(centroid[axis] * down[axis] for axis in range(3))
    boundary = float(policy["objectDownSideCoordinate"])
    return (
        boundary - float(policy["inwardBand"])
        <= down_coordinate
        <= boundary + float(policy["outwardBand"])
    )


def count_intersecting_faces(
    mesh: BinaryTriangleMesh,
    bounds: dict[str, list[float]],
) -> int:
    return sum(
        _face_intersects_bounds(mesh, face_index, bounds) for face_index in range(mesh.face_count)
    )


def carve_triangle_mesh_ply(
    data: bytes,
    *,
    label: str,
    bounds: dict[str, list[float]],
    support_protection: dict[str, Any] | None = None,
) -> tuple[bytes, dict[str, Any]]:
    mesh = parse_binary_triangle_mesh_ply(data, label)
    kept_faces = bytearray()
    removed_faces = 0
    protected_support_faces = 0
    for face_index in range(mesh.face_count):
        offset = mesh.face_offset + face_index * 13
        if _face_intersects_bounds(mesh, face_index, bounds):
            if support_protection and _face_matches_support_policy(
                mesh, face_index, support_protection
            ):
                protected_support_faces += 1
                kept_faces.extend(mesh.data[offset : offset + 13])
                continue
            removed_faces += 1
            continue
        kept_faces.extend(mesh.data[offset : offset + 13])
    require(removed_faces > 0, "front-pillow static collider carve removed zero faces")
    kept_face_count = mesh.face_count - removed_faces
    output_header = mesh.header.replace(
        f"element face {mesh.face_count}",
        f"element face {kept_face_count}",
    )
    require(output_header != mesh.header, "failed to update carved collider face count")
    output = b"".join(
        (
            output_header.encode("ascii"),
            mesh.data[mesh.data_offset : mesh.face_offset],
            bytes(kept_faces),
        )
    )
    verified = parse_binary_triangle_mesh_ply(output, f"{label}:carved")
    retained = count_intersecting_faces(verified, bounds)
    require(
        retained == protected_support_faces,
        "carved static collider retains an unprotected front-pillow intersecting face",
    )
    return output, {
        "inputFaceCount": mesh.face_count,
        "outputFaceCount": verified.face_count,
        "removedFaceCount": removed_faces,
        "vertexCount": verified.vertex_count,
        "finite": True,
        "faceIndicesValid": True,
        "retainedIntersectingFaces": retained,
        "retainedUnprotectedIntersectingFaces": retained - protected_support_faces,
        "protectedSupportFaceCount": protected_support_faces,
        "supportProtection": copy.deepcopy(support_protection),
        "carveBounds": bounds,
    }


def _read_verified_chunked_asset(asset: dict[str, Any], chunk_dir: Path) -> bytes:
    parts = asset.get("parts")
    require(isinstance(parts, list) and parts, "base static collider has no chunks")
    chunks: list[bytes] = []
    for index, part in enumerate(parts):
        require(isinstance(part, dict), f"invalid collider chunk record {index}")
        path = chunk_dir / Path(str(part["url"])).name
        sha256, size = sha256_file(path)
        require(size == part.get("size"), f"collider chunk size mismatch: {path}")
        require(sha256 == part.get("sha256"), f"collider chunk hash mismatch: {path}")
        chunks.append(path.read_bytes())
    data = b"".join(chunks)
    require(len(data) == asset.get("size"), "merged static collider size mismatch")
    require(
        hashlib.sha256(data).hexdigest() == asset.get("sha256"),
        "merged static collider hash mismatch",
    )
    return data


def write_verified_chunks(
    data: bytes,
    *,
    output_dir: Path,
    public_prefix: str,
    stem: str,
    chunk_size: int = COLLIDER_CHUNK_SIZE,
) -> list[dict[str, Any]]:
    """Write deterministic publish chunks and verify their reconstructed payload."""

    require(data, "cannot chunk an empty collider payload")
    require(
        0 < chunk_size <= COLLIDER_CHUNK_SIZE,
        f"publish chunk size must be between 1 and {COLLIDER_CHUNK_SIZE} bytes",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    parts: list[dict[str, Any]] = []
    expected_names: set[str] = set()
    for index, start in enumerate(range(0, len(data), chunk_size)):
        payload = data[start : start + chunk_size]
        name = f"{stem}.chunk{index:03d}"
        expected_names.add(name)
        path = output_dir / name
        digest = hashlib.sha256(payload).hexdigest()
        existing_matches = False
        if path.is_file():
            existing_sha, existing_size = sha256_file(path)
            existing_matches = existing_size == len(payload) and existing_sha == digest
        if not existing_matches:
            temporary = path.with_name(f".{name}.{os.getpid()}.tmp")
            temporary.unlink(missing_ok=True)
            try:
                temporary.write_bytes(payload)
                temporary_sha, temporary_size = sha256_file(temporary)
                require(
                    temporary_size == len(payload),
                    f"chunk size mismatch before publish: {name}",
                )
                require(temporary_sha == digest, f"chunk hash mismatch before publish: {name}")
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        parts.append(
            {
                "url": f"{public_prefix.rstrip('/')}/{name}",
                "size": len(payload),
                "sha256": digest,
            }
        )

    chunk_prefix = f"{stem}.chunk"
    for stale in output_dir.glob(f"{chunk_prefix}*"):
        suffix = stale.name.removeprefix(chunk_prefix)
        if stale.is_file() and suffix.isdigit() and stale.name not in expected_names:
            stale.unlink()

    reconstructed = hashlib.sha256()
    reconstructed_size = 0
    for index, part in enumerate(parts):
        path = output_dir / Path(str(part["url"])).name
        actual_sha, actual_size = sha256_file(path)
        require(actual_size == part["size"] <= chunk_size, f"published chunk size mismatch: {path}")
        require(actual_sha == part["sha256"], f"published chunk hash mismatch: {path}")
        payload = path.read_bytes()
        reconstructed.update(payload)
        reconstructed_size += len(payload)
        require(path.name.endswith(f".chunk{index:03d}"), "published chunk order is unstable")
    require(reconstructed_size == len(data), "reconstructed collider chunk size mismatch")
    require(
        reconstructed.hexdigest() == hashlib.sha256(data).hexdigest(),
        "reconstructed collider chunk hash mismatch",
    )
    return parts


def adopt_verified_chunks(
    source_parts: list[dict[str, Any]],
    *,
    source_dir: Path,
    output_dir: Path,
    public_prefix: str,
    stem: str,
    expected_sha256: str,
    expected_size: int,
    chunk_size: int = COLLIDER_CHUNK_SIZE,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Adopt verified source chunks into a canonical publish namespace."""

    require(source_parts, "cannot adopt an empty chunk list")
    require(0 < chunk_size <= COLLIDER_CHUNK_SIZE, "adopted chunks must not exceed one MiB")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_parts: list[dict[str, Any]] = []
    storage_modes: list[str] = []
    expected_names: set[str] = set()
    merged_digest = hashlib.sha256()
    merged_size = 0
    for index, source_part in enumerate(source_parts):
        require(isinstance(source_part, dict), f"invalid source chunk record {index}")
        source = source_dir / Path(str(source_part["url"])).name
        part_size = int(source_part["size"])
        part_sha = str(source_part["sha256"])
        require(0 < part_size <= chunk_size, f"source chunk exceeds publish limit: {source}")
        name = f"{stem}.chunk{index:03d}"
        expected_names.add(name)
        destination = output_dir / name
        record = materialize_file(
            source,
            destination,
            expected_sha256=part_sha,
            expected_size=part_size,
        )
        storage_modes.append(record.mode)
        with destination.open("rb") as stream:
            while block := stream.read(COLLIDER_CHUNK_SIZE):
                merged_digest.update(block)
                merged_size += len(block)
        output_parts.append(
            {
                "url": f"{public_prefix.rstrip('/')}/{name}",
                "size": part_size,
                "sha256": part_sha,
            }
        )

    chunk_prefix = f"{stem}.chunk"
    for stale in output_dir.glob(f"{chunk_prefix}*"):
        suffix = stale.name.removeprefix(chunk_prefix)
        if stale.is_file() and suffix.isdigit() and stale.name not in expected_names:
            stale.unlink()
    require(merged_size == expected_size, "adopted chunks reconstruct to the wrong byte size")
    require(
        merged_digest.hexdigest() == expected_sha256, "adopted chunks reconstruct to wrong hash"
    )
    return output_parts, storage_modes


def _verify_glb(path: Path, expected_size: int) -> None:
    with path.open("rb") as stream:
        header = stream.read(12)
    require(len(header) == 12, f"GLB header is truncated: {path}")
    magic, version, declared_size = struct.unpack("<4sII", header)
    require(magic == b"glTF" and version == 2, f"asset is not a GLB 2.0 file: {path}")
    require(declared_size == expected_size, f"GLB declared byte length differs from report: {path}")


def load_evidence(project_root: Path, base_manifest: Path) -> CandidateEvidence:
    completion_root = project_root / "examples" / "bedroom4" / "completion"
    front_root = completion_root / "trellis2_pillow_front_seed42"
    scene_fit_path = front_root / "scene_fit_silhouette_refined" / "scene_fit_report.json"
    front_source = front_root / "scene_fit_silhouette_refined" / "sam3_pillow_front.scene-fit.glb"
    split_report_path = (
        completion_root / "bedroom4_frame64_three_pillows" / "instance_cloud_split_report.json"
    )
    clean_review_path = completion_root / "clean_plate_round3_sdxl_anchor" / "visual_review.json"
    canonical_review_path = (
        front_root
        / "scene_fit_silhouette_refined"
        / "canonical_six_view_review"
        / "object_six_view_review.json"
    )
    front_visual_review_path = canonical_review_path.with_name("visual_review.json")
    static_recarve_report_path = front_root / "composition-review" / "static-recarve-report.json"
    static_recarve_browser_review_path = (
        front_root / "composition-review" / "static-recarve-browser-comparison.json"
    )
    scene_fit = read_json(scene_fit_path)
    split_report = read_json(split_report_path)
    clean_review = read_json(clean_review_path)
    canonical_review = read_json(canonical_review_path)
    front_visual_review = read_json(front_visual_review_path)
    static_recarve_report = read_json(static_recarve_report_path)
    static_recarve_browser_review = read_json(static_recarve_browser_review_path)
    base = read_json(base_manifest)
    base_sha, _ = sha256_file(base_manifest)
    scene_fit_sha, _ = sha256_file(scene_fit_path)
    split_report_sha, _ = sha256_file(split_report_path)
    clean_review_sha, _ = sha256_file(clean_review_path)
    canonical_review_sha, _ = sha256_file(canonical_review_path)
    front_visual_review_sha, _ = sha256_file(front_visual_review_path)
    static_recarve_report_sha, _ = sha256_file(static_recarve_report_path)
    static_recarve_browser_review_sha, _ = sha256_file(static_recarve_browser_review_path)

    require(base_sha == STABLE_BASE_MANIFEST_SHA256, "stable base manifest hash changed")

    require(scene_fit.get("status") == "technical_gates_passed", "scene fit did not pass")
    require(scene_fit.get("all_acceptance_gates_passed") is True, "scene fit gates failed")
    gates = scene_fit.get("acceptance_gates")
    require(isinstance(gates, dict) and all(gates.values()), "scene fit has a failed gate")
    mesh = scene_fit.get("mesh")
    require(isinstance(mesh, dict), "scene fit has no mesh report")
    require(
        scene_fit.get("material_profile") == "nonmetal_fabric",
        "front scene-fit GLB must use the audited nonmetal fabric material profile",
    )
    require(gates.get("material_profile_applied") is True, "fabric material gate failed")
    materials = mesh.get("materials")
    require(isinstance(materials, list) and materials, "front GLB has no PBR material audit")
    require(
        all(
            isinstance(item, dict)
            and item.get("metallic_factor_after") == 0.0
            and item.get("roughness_factor_after") == 1.0
            for item in materials
        ),
        "front GLB material is not nonmetal rough fabric",
    )
    for field, expected in FRONT_EXPECTED.items():
        report_field = (
            "glb_bytes" if field == "size" else "glb_sha256" if field == "sha256" else field
        )
        require(mesh.get(report_field) == expected, f"front mesh {report_field} changed")
    require(mesh.get("finite_vertices") is True, "front mesh contains non-finite vertices")
    require(mesh.get("degenerate_face_count") == 0, "front mesh contains degenerate faces")
    require(mesh.get("winding_consistent") is True, "front mesh winding is inconsistent")
    require(mesh.get("watertight") is False, "front surface topology claim changed")
    front_sha, front_size = sha256_file(front_source)
    require(front_sha == FRONT_EXPECTED["sha256"], "front GLB hash differs from scene-fit report")
    require(front_size == FRONT_EXPECTED["size"], "front GLB size differs from scene-fit report")
    _verify_glb(front_source, front_size)

    canonical_views = ["front", "right", "back", "left", "top", "bottom"]
    require(
        canonical_review_sha == CANONICAL_SIX_VIEW_REVIEW_SHA256,
        "canonical six-view receipt hash changed",
    )
    require(
        canonical_review.get("kind") == "video2world.canonical_object_six_view_review",
        "unexpected canonical six-view receipt kind",
    )
    require(
        canonical_review.get("canonicalViews") == canonical_views,
        "canonical six-view receipt does not cover the six orthogonal views",
    )
    require(
        canonical_review.get("horizontalOrbitAcceptedAsSixViewEvidence") is False,
        "horizontal orbit frames cannot substitute for canonical six-view evidence",
    )
    canonical_asset = canonical_review.get("asset")
    require(isinstance(canonical_asset, dict), "canonical six-view receipt has no asset")
    require(
        canonical_asset.get("sha256") == FRONT_EXPECTED["sha256"]
        and canonical_asset.get("sizeBytes") == FRONT_EXPECTED["size"]
        and canonical_review.get("triangleCount") == FRONT_EXPECTED["faces"],
        "canonical six-view receipt references a different front-pillow asset",
    )

    require(
        front_visual_review.get("kind") == "video2world.object_six_view_visual_review",
        "unexpected front visual-review kind",
    )
    require(
        front_visual_review.get("status") == "accepted_current_demo_only"
        and front_visual_review.get("promotion_allowed") is True
        and front_visual_review.get("promotion_scope") == "current_demo_only",
        "front visual review did not allow current-demo-only promotion",
    )
    reviewed_asset = front_visual_review.get("asset")
    require(isinstance(reviewed_asset, dict), "front visual review has no asset record")
    require(
        reviewed_asset.get("sha256") == FRONT_EXPECTED["sha256"]
        and reviewed_asset.get("size_bytes") == FRONT_EXPECTED["size"]
        and reviewed_asset.get("vertices") == FRONT_EXPECTED["vertices"]
        and reviewed_asset.get("faces") == FRONT_EXPECTED["faces"]
        and reviewed_asset.get("material_profile") == "nonmetal_fabric",
        "front visual review references a different or unaudited asset",
    )
    reviewed_canonical = front_visual_review.get("canonical_review")
    require(isinstance(reviewed_canonical, dict), "front visual review has no canonical receipt")
    require(
        reviewed_canonical.get("receipt_sha256") == canonical_review_sha
        and reviewed_canonical.get("views") == canonical_views
        and reviewed_canonical.get("horizontal_orbit_accepted_as_six_view") is False,
        "front visual review does not bind the canonical six-view receipt",
    )
    visual_gates = front_visual_review.get("gates")
    require(isinstance(visual_gates, dict) and all(visual_gates.values()), "visual gates failed")
    require(
        visual_gates.get("scene_interpenetration_not_obvious") is True,
        "scene interpenetration gate did not pass",
    )
    require(front_visual_review.get("blocking_issues") == [], "front review has blocking issues")
    warnings = front_visual_review.get("warnings")
    require(
        isinstance(warnings, list)
        and any(
            isinstance(warning, dict) and warning.get("issue_type") == "texture_hallucination"
            for warning in warnings
        ),
        "accepted floral texture warning is missing from the front review",
    )

    require(
        static_recarve_report_sha == STATIC_RECARVE_REPORT_SHA256,
        "static recarve report hash changed",
    )
    require(
        static_recarve_report.get("status") == "experimental_candidate_materialized"
        and static_recarve_report.get("mode") == "semantic-covariance",
        "semantic-covariance static recarve report is not promotable",
    )
    static_visual = static_recarve_report.get("staticVisual")
    require(isinstance(static_visual, dict), "static recarve report has no static visual")
    require(
        static_visual.get("outputSha256") == STATIC_VISUAL_EXPECTED["sha256"]
        and static_visual.get("outputBytes") == STATIC_VISUAL_EXPECTED["size"]
        and static_visual.get("outputVertexCount") == STATIC_VISUAL_EXPECTED["vertices"]
        and static_visual.get("inputVertexCount") == STATIC_VISUAL_EXPECTED["input_vertices"]
        and static_visual.get("removedVertexCount") == STATIC_VISUAL_EXPECTED["removed_vertices"]
        and static_visual.get("chunkCount") == STATIC_VISUAL_EXPECTED["parts"],
        "static recarve report values changed",
    )
    source_parts = static_visual.get("parts")
    require(
        isinstance(source_parts, list) and len(source_parts) == STATIC_VISUAL_EXPECTED["parts"],
        "static recarve report has the wrong number of chunks",
    )
    require(
        sum(int(part["size"]) for part in source_parts) == STATIC_VISUAL_EXPECTED["size"]
        and all(0 < int(part["size"]) <= COLLIDER_CHUNK_SIZE for part in source_parts)
        and all(len(str(part["sha256"])) == 64 for part in source_parts),
        "static recarve report contains invalid publish chunks",
    )

    require(
        static_recarve_browser_review_sha == STATIC_RECARVE_BROWSER_REVIEW_SHA256,
        "static recarve browser review hash changed",
    )
    require(
        static_recarve_browser_review.get("status") == "candidate_passed"
        and static_recarve_browser_review.get("failedChecks") == [],
        "static recarve browser comparison did not pass",
    )
    browser_checks = static_recarve_browser_review.get("checks")
    require(
        isinstance(browser_checks, dict) and all(browser_checks.values()),
        "static recarve browser comparison has a failed check",
    )
    decision = static_recarve_browser_review.get("decision")
    require(isinstance(decision, dict), "static recarve browser review has no decision")
    require(
        decision.get("currentDemoPromotion") == "recommended_with_known_limitation"
        and decision.get("backgroundAndBedCompletion") == "not_proven",
        "static recarve browser decision changed",
    )
    scene_review = front_visual_review.get("scene_review")
    require(isinstance(scene_review, dict), "front visual review has no scene review")
    scene_hard_gates = scene_review.get("hard_gates")
    require(
        scene_review.get("status") == "passed_current_demo_only"
        and scene_review.get("receipt_sha256") == static_recarve_browser_review_sha
        and scene_review.get("static_visual_sha256") == STATIC_VISUAL_EXPECTED["sha256"]
        and scene_review.get("retained_static_gaussians") == STATIC_VISUAL_EXPECTED["vertices"]
        and isinstance(scene_hard_gates, dict)
        and all(scene_hard_gates.values()),
        "front visual review is not bound to the passed scene recarve review",
    )

    require(split_report.get("status") == "passed", "pillow split report did not pass")
    instances = split_report.get("instances")
    require(isinstance(instances, list), "pillow split report has no instances")
    all_records = {
        str(record["object_id"]): record
        for record in instances
        if isinstance(record, dict) and record.get("object_id") in {FRONT_ID, *REAR_IDS}
    }
    require(FRONT_ID in all_records, "pillow split report lacks the front pillow record")
    records = {object_id: all_records[object_id] for object_id in REAR_IDS}
    require(set(records) == set(REAR_IDS), "pillow split report lacks rear pillow records")
    rear_sources: dict[str, Path] = {}
    for object_id in REAR_IDS:
        record = records[object_id]
        output = record.get("output")
        geometry = record.get("geometry")
        require(isinstance(output, dict) and isinstance(geometry, dict), f"incomplete {object_id}")
        source = (
            project_root
            / "examples"
            / "bedroom4"
            / "completion"
            / "bedroom4_frame64_three_pillows"
            / str(output["relative_path"])
        )
        expected = REAR_EXPECTED[object_id]
        sha256, size = sha256_file(source)
        require(sha256 == expected["sha256"] == output.get("sha256"), f"{object_id} hash mismatch")
        require(size == expected["size"] == output.get("bytes"), f"{object_id} size mismatch")
        stats = inspect_rgb_ply(source)
        require(
            stats.point_count == expected["points"] == record.get("point_count"),
            f"{object_id} point count mismatch",
        )
        aabb = geometry.get("aabb")
        require(isinstance(aabb, dict), f"{object_id} has no AABB")
        require(_close_vector(list(stats.minimum), aabb["minimum"]), f"{object_id} minimum differs")
        require(_close_vector(list(stats.maximum), aabb["maximum"]), f"{object_id} maximum differs")
        rear_sources[object_id] = source

    require(
        clean_review.get("status") == "accepted_for_current_demo", "clean plate is not accepted"
    )
    require(clean_review.get("selected_seed") == 2026071701, "unexpected clean-plate seed")
    scope = clean_review.get("scope")
    require(isinstance(scope, dict), "clean-plate review has no scope")
    require(
        "occluded bed geometry completion" in scope.get("not_proven", []),
        "clean-plate scope overclaims 3D background",
    )
    assets = base.get("assets")
    require(isinstance(assets, dict), "base manifest has no assets")
    base_collider = assets.get("colliderStaticCarved")
    require(isinstance(base_collider, dict), "base manifest has no carved static collider")
    front_geometry = all_records[FRONT_ID].get("geometry")
    require(isinstance(front_geometry, dict), "front split record has no geometry")
    front_split_aabb = front_geometry.get("aabb")
    require(isinstance(front_split_aabb, dict), "front split record has no AABB")
    coordinate_system = base.get("coordinateSystem")
    require(isinstance(coordinate_system, dict), "base manifest has no coordinate system")
    raw_world_up = coordinate_system.get("worldUp")
    require(
        isinstance(raw_world_up, list)
        and len(raw_world_up) == 3
        and all(math.isfinite(float(value)) for value in raw_world_up),
        "base manifest has an invalid world-up vector",
    )
    magnitude = math.sqrt(sum(float(value) ** 2 for value in raw_world_up))
    require(magnitude > 1e-8, "base manifest world-up vector is zero")
    world_up = tuple(float(value) / magnitude for value in raw_world_up)
    return CandidateEvidence(
        base_sha256=base_sha,
        scene_fit_sha256=scene_fit_sha,
        split_report_sha256=split_report_sha,
        clean_plate_review_sha256=clean_review_sha,
        canonical_six_view_review_sha256=canonical_review_sha,
        front_visual_review_sha256=front_visual_review_sha,
        static_recarve_report_sha256=static_recarve_report_sha,
        static_recarve_browser_review_sha256=static_recarve_browser_review_sha,
        front_source=front_source,
        front_mesh=mesh,
        rear_sources=rear_sources,
        rear_records=records,
        front_split_aabb=front_split_aabb,
        base_collider_asset=base_collider,
        clean_plate_review=clean_review,
        canonical_six_view_review=canonical_review,
        front_visual_review=front_visual_review,
        static_recarve_report=static_recarve_report,
        static_recarve_browser_review=static_recarve_browser_review,
        world_up=world_up,
    )


def _localized_description(kind: str) -> dict[str, Any]:
    if kind == FRONT_ID:
        return {
            "short": {
                "en": "A smaller light off-white square front pillow with softly rounded corners.",
                "zh": "一只较小的浅灰白色方形前排枕头, 四角柔和圆润。",
            },
            "appearance": {
                "en": (
                    "The completed PBR surface keeps the observed light neutral main color "
                    "and soft stuffed form."
                ),
                "zh": "补全后的 PBR 表面保持了观测到的浅中性色主色和柔软填充形态。",
            },
            "detailed": {
                "en": (
                    "A compact near-square cushion with a visibly thick completed back surface. "
                    "Minor backside pattern differences are accepted for this demo."
                ),
                "zh": (
                    "紧凑的近方形软垫, 具有可见厚度和完整背面; "
                    "背面花纹的轻微差异已按当前演示口径接受。"
                ),
            },
            "location": {
                "en": "On the bed in front of the two rear pillows.",
                "zh": "位于床上、两只后排枕头前方。",
            },
            "fidelity_caveat": {
                "en": (
                    "The backside was generated rather than observed. Minor backside texture "
                    "variation is accepted; obvious deformation, main-color error, missing "
                    "surfaces, or significant interpenetration remains blocking."
                ),
                "zh": (
                    "背面由模型生成而非直接观测。轻微背面纹理差异已接受; "
                    "明显形变、主色错误、缺面或显著穿模仍会阻断。"
                ),
            },
            "provider": "TRELLIS.2+scene_fit+user_demo_review",
            "model": "TRELLIS.2 seed 42",
            "confidence": None,
            "evidence_frames": ["000064"],
        }
    side = "left" if kind.endswith("left") else "right"
    zh_side = "左侧" if side == "left" else "右侧"
    return {
        "short": {
            "en": f"The {side} large pale patchwork rear pillow, restored as observed RGB points.",
            "zh": f"{zh_side}的大号浅色拼接后排枕头, 以观测 RGB 点云恢复。",
        },
        "appearance": {
            "en": (
                "Off-white and cream quilt-like panels under warm room lighting, with muted "
                "brown floral accents."
            ),
            "zh": "暖色室内光下的灰白与米白色绗缝拼接, 带低饱和棕色花卉点缀。",
        },
        "detailed": {
            "en": (
                "Only the observed visible surface is present; hidden backside geometry is "
                "not completed."
            ),
            "zh": "仅包含已观测可见表面; 隐藏背面几何未补全。",
        },
        "location": {
            "en": f"On the {side} rear side of the bed pillow group.",
            "zh": f"位于床上枕头组后排{zh_side}。",
        },
        "fidelity_caveat": {
            "en": (
                "Visible-surface RGB evidence only. It has selection bounds but no movement "
                "or collision interaction."
            ),
            "zh": "仅为可见表面 RGB 证据; 提供选择包围盒, 但不启用移动或碰撞交互。",
        },
        "provider": "Video2Mesh projection fusion+SAM3 instance split",
        "model": "geometry-derived split",
        "confidence": 0.9660567045211792 if side == "left" else 0.9715292453765869,
        "evidence_frames": ["000064"],
    }


def _bbox(minimum: list[float], maximum: list[float]) -> dict[str, Any]:
    center = [(low + high) / 2 for low, high in zip(minimum, maximum, strict=True)]
    extent = [high - low for low, high in zip(minimum, maximum, strict=True)]
    return {
        "coordinateFrame": "visual_native",
        "min": minimum,
        "max": maximum,
        "center": center,
        "extent": extent,
    }


def _front_world_bounds(evidence: CandidateEvidence) -> dict[str, list[float]]:
    local_min = [float(value) for value in evidence.front_mesh["bounds"][0]]
    local_max = [float(value) for value in evidence.front_mesh["bounds"][1]]
    return {
        "min": [value + FRONT_PIVOT[index] for index, value in enumerate(local_min)],
        "max": [value + FRONT_PIVOT[index] for index, value in enumerate(local_max)],
    }


def front_collider_carve_bounds(evidence: CandidateEvidence) -> dict[str, Any]:
    completed = _front_world_bounds(evidence)
    source = evidence.front_split_aabb
    union = {
        "min": [min(float(source["minimum"][axis]), completed["min"][axis]) for axis in range(3)],
        "max": [max(float(source["maximum"][axis]), completed["max"][axis]) for axis in range(3)],
    }
    expanded = {
        "min": [value - FRONT_COLLIDER_CARVE_MARGIN for value in union["min"]],
        "max": [value + FRONT_COLLIDER_CARVE_MARGIN for value in union["max"]],
    }
    return {
        "sourceSplitAabb": {
            "min": [float(value) for value in source["minimum"]],
            "max": [float(value) for value in source["maximum"]],
        },
        "completedWorldAabb": completed,
        "unionAabb": union,
        "margin": FRONT_COLLIDER_CARVE_MARGIN,
        "expandedAabb": expanded,
    }


def front_collider_support_policy(evidence: CandidateEvidence) -> dict[str, Any]:
    completed = _front_world_bounds(evidence)
    down = tuple(-value for value in evidence.world_up)
    down_side_coordinate = sum(
        down[axis] * (completed["max"][axis] if down[axis] >= 0 else completed["min"][axis])
        for axis in range(3)
    )
    return {
        "worldUp": list(evidence.world_up),
        "objectDownSideCoordinate": down_side_coordinate,
        "inwardBand": FRONT_SUPPORT_BAND,
        "outwardBand": FRONT_SUPPORT_BAND,
        "minimumNormalAlignment": FRONT_SUPPORT_MIN_NORMAL_ALIGNMENT,
        "classification": "down_side_near_world_up_support_surface",
    }


def _front_object(evidence: CandidateEvidence) -> dict[str, Any]:
    mesh = evidence.front_mesh
    local_min = [float(value) for value in mesh["bounds"][0]]
    local_max = [float(value) for value in mesh["bounds"][1]]
    world_bounds = _front_world_bounds(evidence)
    return {
        "id": FRONT_ID,
        "label": "前排浅色枕头 / Front light pillow",
        "name": "前排浅色枕头 / Front light pillow",
        "category": "pillow",
        "aliases": [
            "pillow",
            "枕头",
            "front pillow",
            "small pillow",
            "前排枕头",
            "小白枕",
        ],
        "description": _localized_description(FRONT_ID),
        "fidelity": "generated_completion_from_observed_front_evidence",
        "bbox": _bbox(world_bounds["min"], world_bounds["max"]),
        "independentlyRecognized": True,
        "visualOnly": False,
        "semanticGranularity": "independent_root_asset",
        "parentObjectId": None,
        "movesWithParent": False,
        "independentlyMovable": True,
        "placement": {
            "coordinateFrame": "visual_native",
            "sourceCoordinateFrame": "trellis2_scene_fit_local",
            "transform": "scene_fit_baked_glb_plus_runtime_pivot",
            "pivot": list(FRONT_PIVOT),
            "generatedCenter": [0, 0, 0],
            "scale": [1, 1, 1],
            "rotationEulerDeg": [0, 0, 0],
            "eulerOrder": "XYZ",
        },
        "collision": {
            "mode": "unified-glb",
            "topology": "surface_bvh",
            "walkable": False,
            "characterCollision": True,
            "asset": {
                "id": "sam3_pillow_front_scene_fit_unified_pbr_glb",
                "label": "Front pillow scene-fit PBR GLB",
                "url": "./worlds/bedroom4/objects/sam3_pillow_front.scene-fit.glb",
                "fileName": "sam3_pillow_front.scene-fit.glb",
                "fileType": "glb",
                "format": "gltf-binary",
                "size": FRONT_EXPECTED["size"],
                "sha256": FRONT_EXPECTED["sha256"],
                "vertices": FRONT_EXPECTED["vertices"],
                "faces": FRONT_EXPECTED["faces"],
                "bounds": [local_min, local_max],
                "finite": True,
                "nondegenerate": True,
                "windingConsistent": True,
                "watertight": False,
                "sourcePath": "repo://video2world/examples/bedroom4/completion/trellis2_pillow_front_seed42/scene_fit_silhouette_refined/sam3_pillow_front.scene-fit.glb",
            },
            "gate": {
                "status": "passed",
                "surfaceCollision": "passed",
                "rootProxyBrowserQa": "pending_promoted_manifest_recheck",
                "reason": (
                    "Scene-fit, canonical six-view, current-demo visual, and prior recarve "
                    "browser evidence passed. The newly promoted manifest transport still "
                    "requires a fresh root-proxy browser recheck; the surface remains open."
                ),
            },
        },
        "interaction": {
            "kind": "spin",
            "degrees": 360,
            "durationMs": 1250,
            "drag": "horizontal_yaw",
        },
        "limitations": [
            (
                "The GLB is a winding-consistent open surface (surface_bvh), not a watertight "
                "closed volume."
            ),
            (
                "Backside texture was generated; its minor pattern difference is accepted for "
                "the current demo."
            ),
            "Fresh root-proxy QA of the promoted manifest transport is pending.",
            "The bed and hidden background geometry are not reconstructed by this asset.",
        ],
    }


def _rear_object(object_id: str, evidence: CandidateEvidence) -> dict[str, Any]:
    record = evidence.rear_records[object_id]
    aabb = record["geometry"]["aabb"]
    minimum = [float(value) for value in aabb["minimum"]]
    maximum = [float(value) for value in aabb["maximum"]]
    center = [float(value) for value in aabb["center"]]
    extents = [float(value) for value in aabb["extents"]]
    side = "left" if object_id.endswith("left") else "right"
    zh_side = "左侧" if side == "left" else "右侧"
    expected = REAR_EXPECTED[object_id]
    return {
        "id": object_id,
        "label": f"{zh_side}后排拼接枕头 / {side.title()} rear patchwork pillow",
        "name": f"{zh_side}后排枕头",
        "category": "pillow",
        "aliases": [f"{side} rear pillow", f"{side} patchwork pillow", f"{zh_side}后排枕头"],
        "description": _localized_description(object_id),
        "fidelity": "observed_visible_surface_rgb_points",
        "bbox": _bbox(minimum, maximum),
        "independentlyRecognized": True,
        "visualOnly": True,
        "semanticGranularity": "independent_root_asset",
        "parentObjectId": None,
        "movesWithParent": False,
        "independentlyMovable": False,
        "placement": {
            "coordinateFrame": "visual_native",
            "sourceCoordinateFrame": "video2mesh_da3_world",
            "transform": "identity_holi_da3_to_pgsr_shared_frame",
            "pivot": center,
            "generatedCenter": center,
            "scale": [1, 1, 1],
            "rotationEulerDeg": [0, 0, 0],
            "eulerOrder": "XYZ",
        },
        "colliderProxy": {
            "type": "selection-box",
            "dimensions": extents,
            "center": [0, 0, 0],
            "collisionEnabled": False,
        },
        "visual": {
            "id": f"{object_id}_observed_rgb_points",
            "label": f"{side.title()} rear pillow observed RGB points",
            "url": f"./worlds/bedroom4/objects/{object_id}.ply",
            "fileName": f"{object_id}.ply",
            "fileType": "ply",
            "format": "rgb-point-cloud-ply",
            "renderer": "three-points",
            "pointSize": 0.03,
            "vertexCount": expected["points"],
            "primitiveKind": "rgb-point",
            "coordinateFrame": "visual_native",
            "bbox": {"min": minimum, "max": maximum},
            "size": expected["size"],
            "sha256": expected["sha256"],
            "sourcePath": f"repo://video2world/examples/bedroom4/completion/bedroom4_frame64_three_pillows/objects/{object_id}/{object_id}.ply",
        },
        "collision": {
            "mode": "none",
            "walkable": False,
            "characterCollision": False,
            "asset": None,
            "gate": {
                "status": "not_tested",
                "reason": (
                    "Observed visible-surface RGB points are selection-only and do not provide "
                    "collision geometry."
                ),
            },
        },
        "limitations": [
            "Only the observed visible RGB surface is restored.",
            "Hidden/backside geometry is not completed and collision is disabled.",
        ],
    }


def _knowledge_object(interactive: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": interactive["id"],
        "name": interactive["name"],
        "category": interactive["category"],
        "aliases": interactive["aliases"],
        "description": interactive["description"],
        "bbox": interactive["bbox"],
        "interactiveObjectId": interactive["id"],
        "independentlyRecognized": True,
        "visualOnly": interactive["visualOnly"],
        "limitations": interactive["limitations"],
    }


def _canonical_static_visual_parts(evidence: CandidateEvidence) -> list[dict[str, Any]]:
    source_parts = evidence.static_recarve_report["staticVisual"]["parts"]
    return [
        {
            "url": (
                f"{STATIC_VISUAL_CHUNK_PUBLIC_PREFIX}/{STATIC_VISUAL_CHUNK_STEM}.chunk{index:03d}"
            ),
            "size": int(part["size"]),
            "sha256": str(part["sha256"]),
        }
        for index, part in enumerate(source_parts)
    ]


def _promoted_static_visual(
    base_visual: dict[str, Any], evidence: CandidateEvidence
) -> dict[str, Any]:
    visual = copy.deepcopy(base_visual)
    previous_removed = int(visual["removedVertexCount"])
    previous_output = int(visual["vertexCount"])
    require(
        previous_output == STATIC_VISUAL_EXPECTED["input_vertices"],
        "static recarve input does not match the stable visual",
    )
    parts = _canonical_static_visual_parts(evidence)
    mode_metrics = evidence.static_recarve_report["analysis"]["modes"]["semantic-covariance"]
    browser = evidence.static_recarve_browser_review
    collateral = browser["summary"]["sceneCollateral"]
    visual.update(
        {
            "id": "bedroom_4_pgsr_static_front_recarve_semantic_covariance",
            "label": (
                "Bedroom 4 PGSR static Gaussian layer with interactive objects and residual "
                "front-pillow splats removed"
            ),
            "fileName": "bedroom_4_pgsr_static_front_recarve_semantic_covariance.ply",
            "vertexCount": STATIC_VISUAL_EXPECTED["vertices"],
            "size": STATIC_VISUAL_EXPECTED["size"],
            "sha256": STATIC_VISUAL_EXPECTED["sha256"],
            "parts": parts,
            "chunkSize": COLLIDER_CHUNK_SIZE,
            "transportMode": "verified_chunks",
            "directOversizedAssetPublished": False,
            "removedVertexCount": previous_removed + STATIC_VISUAL_EXPECTED["removed_vertices"],
            "inputVertexCount": previous_output,
            "latestRemovedVertexCount": STATIC_VISUAL_EXPECTED["removed_vertices"],
            "carveMethod": (f"{visual['carveMethod']}_then_front_residual_semantic_covariance"),
            "frontResidualCarve": {
                "status": "passed_current_demo_only",
                "mode": "semantic-covariance",
                "inputStaticGaussianCount": STATIC_VISUAL_EXPECTED["input_vertices"],
                "removedGaussianCount": STATIC_VISUAL_EXPECTED["removed_vertices"],
                "retainedStaticGaussianCount": STATIC_VISUAL_EXPECTED["vertices"],
                "metrics": copy.deepcopy(mode_metrics),
                "browserComparison": {
                    "status": browser["status"],
                    "report": (
                        "repo://video2world/examples/bedroom4/completion/"
                        "trellis2_pillow_front_seed42/composition-review/"
                        "static-recarve-browser-comparison.json"
                    ),
                    "sha256": evidence.static_recarve_browser_review_sha256,
                    "perimeterDarkLossFraction": collateral["perimeter"][
                        "darkLossFractionAbove35Luma"
                    ],
                    "supportBandDarkLossFraction": collateral["supportBand"][
                        "darkLossFractionAbove35Luma"
                    ],
                },
                "recarveReport": {
                    "path": (
                        "repo://video2world/examples/bedroom4/completion/"
                        "trellis2_pillow_front_seed42/composition-review/"
                        "static-recarve-report.json"
                    ),
                    "sha256": evidence.static_recarve_report_sha256,
                },
                "scope": "current_demo_only",
                "backgroundAndBedCompletion": "not_proven",
            },
        }
    )
    visual.pop("url", None)
    removed_by_object = visual.get("removedByObject")
    require(isinstance(removed_by_object, dict), "stable visual has no per-object removal record")
    if OLD_ENSEMBLE_ID in removed_by_object:
        removed_by_object["merged_three_pillow_anchor"] = removed_by_object.pop(OLD_ENSEMBLE_ID)
    removed_by_object["sam3_pillow_front_static_residual"] = STATIC_VISUAL_EXPECTED[
        "removed_vertices"
    ]
    carved_ids = visual.get("carvedObjectIds")
    require(isinstance(carved_ids, list), "stable visual has no carved object list")
    visual["carvedObjectIds"] = [
        "merged_three_pillow_anchor" if item == OLD_ENSEMBLE_ID else item for item in carved_ids
    ] + ["sam3_pillow_front_static_residual"]
    pillow_carve = visual.get("pillowCarve")
    if isinstance(pillow_carve, dict) and pillow_carve.get("objectId") == OLD_ENSEMBLE_ID:
        pillow_carve["objectId"] = "merged_three_pillow_anchor"
    return visual


def _candidate_collider_asset(
    evidence: CandidateEvidence,
    carve: dict[str, Any],
    parts: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline = evidence.base_collider_asset
    baseline_removed = int(baseline["removedFaceCount"])
    additional_removed = int(carve["removedFaceCount"])
    return {
        "id": "bedroom4_tsdf_static_carved_interactive_objects_and_unified_pillow",
        "label": (
            "Bedroom 4 TSDF static collider with legacy objects and unified front pillow removed"
        ),
        "fileName": "collider_bedroom4_tsdf_static_carved_unified_pillow.ply",
        "fileType": "ply",
        "format": "open3d-binary-little-endian-triangle-mesh-ply",
        "vertexCount": carve["vertexCount"],
        "faceCount": carve["outputFaceCount"],
        "bbox": copy.deepcopy(baseline["bbox"]),
        "size": carve["outputBytes"],
        "sha256": carve["outputSha256"],
        "parts": copy.deepcopy(parts),
        "chunkSize": COLLIDER_CHUNK_SIZE,
        "transportMode": "verified_chunks",
        "directOversizedAssetPublished": False,
        "baseline": {
            "id": baseline["id"],
            "sha256": baseline["sha256"],
            "size": baseline["size"],
            "faceCount": baseline["faceCount"],
            "removedFaceCount": baseline_removed,
            "carveMethod": baseline["carveMethod"],
        },
        "sourceSha256": baseline.get("sourceSha256"),
        "originalFaceCount": baseline["originalFaceCount"],
        "baselineRemovedFaceCount": baseline_removed,
        "additionalFrontPillowRemovedFaceCount": additional_removed,
        "removedFaceCount": baseline_removed + additional_removed,
        "removedByObject": {
            **copy.deepcopy(baseline["removedByObject"]),
            FRONT_ID: additional_removed,
        },
        "carveMargin": FRONT_COLLIDER_CARVE_MARGIN,
        "carveMethod": "front_split_and_completed_world_union_aabb_triangle_intersection",
        "frontPillowCarve": copy.deepcopy(carve["boundsEvidence"]),
        "finite": carve["finite"],
        "faceIndicesValid": carve["faceIndicesValid"],
        "retainedIntersectingFaces": carve["retainedIntersectingFaces"],
        "retainedUnprotectedIntersectingFaces": carve["retainedUnprotectedIntersectingFaces"],
        "protectedSupportFaceCount": carve["protectedSupportFaceCount"],
        "supportProtection": copy.deepcopy(carve["supportProtection"]),
    }


def build_candidate_manifest(
    base: dict[str, Any],
    evidence: CandidateEvidence,
    *,
    collider_carve: dict[str, Any],
    collider_parts: list[dict[str, Any]],
) -> dict[str, Any]:
    manifest = copy.deepcopy(base)
    require(manifest.get("schemaVersion") == 1, "base manifest schemaVersion changed")
    require(
        manifest.get("contract") == "video2world-web-manifest-1.0.0",
        "base manifest contract changed",
    )
    interactive = manifest.get("interactiveObjects")
    require(isinstance(interactive, list), "base manifest has no interactiveObjects")
    removed = [item for item in interactive if item.get("id") == OLD_ENSEMBLE_ID]
    require(len(removed) == 1, "base manifest must contain exactly one old pillow ensemble")
    interactive = [item for item in interactive if item.get("id") != OLD_ENSEMBLE_ID]
    front = _front_object(evidence)
    rears = [_rear_object(object_id, evidence) for object_id in REAR_IDS]
    manifest["interactiveObjects"] = [*interactive, front, *rears]

    assets = manifest.get("assets")
    require(isinstance(assets, dict), "base manifest has no assets")
    assets["colliderStaticCarved"] = _candidate_collider_asset(
        evidence, collider_carve, collider_parts
    )
    visual_asset = assets.get("visual")
    require(isinstance(visual_asset, dict), "base manifest has no static visual")
    assets["visual"] = _promoted_static_visual(visual_asset, evidence)

    collision_world = manifest.get("collisionWorld")
    require(isinstance(collision_world, dict), "base manifest has no collisionWorld")
    baseline_collision_world = copy.deepcopy(collision_world)
    base_collider = evidence.base_collider_asset
    total_removed = int(base_collider["removedFaceCount"]) + int(collider_carve["removedFaceCount"])
    collision_world.update(
        {
            "sceneAssetKey": "colliderStaticCarved",
            "replacementMode": (
                "baseline_static_carve_plus_front_split_completed_union_aabb_intersection"
            ),
            "objectFaceRemovalRequired": True,
            "baselineHistory": baseline_collision_world,
            "gate": {
                "status": "passed",
                "originalFaces": base_collider["originalFaceCount"],
                "baselineStaticFaces": base_collider["faceCount"],
                "staticFaces": collider_carve["outputFaceCount"],
                "baselineRemovedFaces": base_collider["removedFaceCount"],
                "additionalFrontPillowRemovedFaces": collider_carve["removedFaceCount"],
                "removedFaces": total_removed,
                "removedByObject": {
                    **copy.deepcopy(base_collider["removedByObject"]),
                    FRONT_ID: collider_carve["removedFaceCount"],
                },
                "retainedIntersectingFaces": collider_carve["retainedIntersectingFaces"],
                "retainedUnprotectedIntersectingFaces": collider_carve[
                    "retainedUnprotectedIntersectingFaces"
                ],
                "protectedSupportFaceCount": collider_carve["protectedSupportFaceCount"],
                "supportProtection": copy.deepcopy(collider_carve["supportProtection"]),
                "carveMethod": ("front_split_and_completed_world_union_aabb_triangle_intersection"),
                "frontPillowCarve": copy.deepcopy(collider_carve["boundsEvidence"]),
                "finite": collider_carve["finite"],
                "faceIndicesValid": collider_carve["faceIndicesValid"],
            },
            "sceneAssetTransport": {
                "mode": "verified_chunks",
                "parts": copy.deepcopy(collider_parts),
                "chunkSize": COLLIDER_CHUNK_SIZE,
                "size": collider_carve["outputBytes"],
                "sha256": collider_carve["outputSha256"],
                "directOversizedAssetPublished": False,
            },
        }
    )

    knowledge = manifest.get("sceneKnowledge")
    require(isinstance(knowledge, dict), "base manifest has no sceneKnowledge")
    knowledge_objects = knowledge.get("objects")
    relations = knowledge.get("relations")
    require(isinstance(knowledge_objects, list), "base sceneKnowledge has no objects")
    require(isinstance(relations, list), "base sceneKnowledge has no relations")
    knowledge["objects"] = [
        item for item in knowledge_objects if item.get("id") != OLD_ENSEMBLE_ID
    ] + [_knowledge_object(item) for item in (front, *rears)]
    knowledge["relations"] = [
        relation
        for relation in relations
        if relation.get("subject") != OLD_ENSEMBLE_ID and relation.get("object") != OLD_ENSEMBLE_ID
    ] + [
        {
            "subject": object_id,
            "predicate": "OnTopOf",
            "object": "sam3_bed_01",
            "targetLabel": {"zh": "床", "en": "the bed"},
            "confidence": (
                0.9723717570304871
                if object_id == FRONT_ID
                else float(evidence.rear_records[object_id]["mask"]["score"])
            ),
            "verified": True,
            "evidence": (
                "scene_fit_front_surface_anchor_and_user_current_demo_acceptance"
                if object_id == FRONT_ID
                else "sam3_frame_000064_geometry_split"
            ),
        }
        for object_id in (FRONT_ID, *REAR_IDS)
    ]

    build = manifest.get("interactiveObjectBuild")
    require(isinstance(build, dict), "base manifest has no interactiveObjectBuild")
    build.update(
        {
            "method": (
                "semantic_anchor_carve_plus_affine_baked_trellis_lod_plus_unified_pbr_front_"
                "plus_split_rgb_rear_plus_front_residual_semantic_covariance_recarve"
            ),
            "recognizedObjectCount": 7,
            "visualObjectCount": 7,
            "collidableObjectCount": 5,
            "objectRgbPointCount": sum(REAR_EXPECTED[item]["points"] for item in REAR_IDS),
            "objectUnifiedGlbCount": 1,
            "frontMeshVertexCount": FRONT_EXPECTED["vertices"],
            "staticSceneGaussianCount": STATIC_VISUAL_EXPECTED["vertices"],
            "removedSceneGaussianCount": 49_532,
            "totalVisualPrimitiveCount": 1_500_948,
            "existingObjectColliderFaceCount": 67_660,
            "frontSurfaceColliderFaceCount": FRONT_EXPECTED["faces"],
            "totalObjectColliderFaceCount": 164_742,
        }
    )
    removed_by_object = build.get("removedByObject")
    if isinstance(removed_by_object, dict) and OLD_ENSEMBLE_ID in removed_by_object:
        removed_by_object["merged_three_pillow_anchor"] = removed_by_object.pop(OLD_ENSEMBLE_ID)
    if isinstance(removed_by_object, dict):
        removed_by_object["sam3_pillow_front_static_residual"] = STATIC_VISUAL_EXPECTED[
            "removed_vertices"
        ]
    pillow_carve = build.get("pillowVisualCarve")
    if isinstance(pillow_carve, dict):
        pillow_carve["objectId"] = "merged_three_pillow_anchor"
        pillow_carve["replacement"] = {
            "front": FRONT_ID,
            "rearVisibleSurfaces": list(REAR_IDS),
            "oldEnsembleRetained": False,
        }
    build["pillowRestoration"] = {
        "status": "passed_current_demo_only",
        "frontUnifiedGlb": FRONT_ID,
        "rearVisibleSurfaceObjects": list(REAR_IDS),
        "rearRgbPointCount": sum(REAR_EXPECTED[item]["points"] for item in REAR_IDS),
        "oldEnsembleRetained": False,
        "backgroundAndBedCompletion": "not_proven",
    }

    production = manifest.get("productionBuild")
    if isinstance(production, dict):
        baseline_browser_qa = production.pop("browserQaReport", None)
        production.pop("pillowEntity", None)
        production.pop("pillowEntitySha256", None)
        production.pop("pillowVisualInteractionGate", None)
        production["baselineHistory"] = {
            "status": "passed_for_base_manifest_only",
            "browserQaReport": baseline_browser_qa,
            "statement": "This QA predates the unified-pillow candidate and does not validate it.",
        }
        production["candidatePillowIntegration"] = {
            "status": "materialized_current_demo_candidate",
            "frontObjectId": FRONT_ID,
            "rearObjectIds": list(REAR_IDS),
            "rootProxyBrowserQa": "pending_promoted_manifest_recheck",
            "priorRecarveBrowserEvidence": "passed",
            "staticVisualSha256": STATIC_VISUAL_EXPECTED["sha256"],
            "staticVisualVertexCount": STATIC_VISUAL_EXPECTED["vertices"],
        }
        production_carve = production.get("pillowVisualCarve")
        if isinstance(production_carve, dict):
            production_carve["objectId"] = "merged_three_pillow_anchor"

    selected = evidence.clean_plate_review["selected_candidate"]
    manifest["version"] = CANDIDATE_VERSION
    manifest["candidateBuild"] = {
        "schemaVersion": 1,
        "createdAt": CANDIDATE_CREATED_AT,
        "status": "materialized_current_demo_candidate",
        "promotionScope": "current_demo_only",
        "completionLayer": 1,
        "completionClaim": "front_pillow_current_demo_candidate_only",
        "nextRoundsPending": [
            "generalized_multiview_amodal_object_completion",
            "occluded_bed_and_background_geometry_reconstruction",
            "additional_scene_object_completion_and_review",
        ],
        "baseManifest": "repo://video2world/examples/bedroom4/manifest.production.json",
        "baseManifestSha256": evidence.base_sha256,
        "materializer": "repo://video2world/scripts/materialize_bedroom4_unified_pillow.py",
        "frontPillow": {
            "objectId": FRONT_ID,
            "representation": "one_scene_fit_pbr_glb_for_visual_selection_and_surface_collision",
            "technicalGates": "passed",
            "userCurrentDemoReview": "passed",
            "sceneReview": "passed_current_demo_only",
            "rootProxyBrowserQa": "pending_promoted_manifest_recheck",
            "canonicalSixViewReviewSha256": evidence.canonical_six_view_review_sha256,
            "visualReviewSha256": evidence.front_visual_review_sha256,
        },
        "rearPillows": {
            "objectIds": list(REAR_IDS),
            "representation": "observed_visible_surface_rgb_points_selection_only",
            "totalPointCount": sum(REAR_EXPECTED[item]["points"] for item in REAR_IDS),
        },
        "primitiveCounts": {
            "staticSceneGaussians": STATIC_VISUAL_EXPECTED["vertices"],
            "existingObjectGaussians": 480_000,
            "rearRgbPoints": 138_926,
            "frontMeshVertices": 60_237,
            "totalVisualPrimitives": 1_500_948,
            "existingObjectColliderFaces": 67_660,
            "frontSurfaceColliderFaces": 97_082,
            "totalObjectColliderFaces": 164_742,
            "collidableObjects": 5,
        },
        "sceneColliderCarve": {
            "status": "passed",
            "inputFaceCount": collider_carve["inputFaceCount"],
            "baselineFaceCount": base_collider["faceCount"],
            "outputFaceCount": collider_carve["outputFaceCount"],
            "removedFaceCount": collider_carve["removedFaceCount"],
            "additionalRemovedFaceCount": collider_carve["removedFaceCount"],
            "vertexCount": collider_carve["vertexCount"],
            "finite": collider_carve["finite"],
            "faceIndicesValid": collider_carve["faceIndicesValid"],
            "retainedIntersectingFaces": collider_carve["retainedIntersectingFaces"],
            "retainedUnprotectedIntersectingFaces": collider_carve[
                "retainedUnprotectedIntersectingFaces"
            ],
            "protectedSupportFaceCount": collider_carve["protectedSupportFaceCount"],
            "supportProtection": copy.deepcopy(collider_carve["supportProtection"]),
            "outputSha256": collider_carve["outputSha256"],
            "outputBytes": collider_carve["outputBytes"],
            "transportMode": "verified_chunks",
            "chunkCount": len(collider_parts),
            "chunkSize": COLLIDER_CHUNK_SIZE,
            "maximumChunkBytes": max(int(part["size"]) for part in collider_parts),
            "directOversizedAssetPublished": False,
            "boundsEvidence": copy.deepcopy(collider_carve["boundsEvidence"]),
        },
        "staticVisualResidualCarve": {
            "status": "passed_current_demo_only",
            "mode": "semantic-covariance",
            "inputStaticGaussianCount": STATIC_VISUAL_EXPECTED["input_vertices"],
            "removedStaticGaussianCount": STATIC_VISUAL_EXPECTED["removed_vertices"],
            "retainedStaticGaussianCount": STATIC_VISUAL_EXPECTED["vertices"],
            "outputSha256": STATIC_VISUAL_EXPECTED["sha256"],
            "outputBytes": STATIC_VISUAL_EXPECTED["size"],
            "chunkCount": STATIC_VISUAL_EXPECTED["parts"],
            "chunkSize": COLLIDER_CHUNK_SIZE,
            "recarveReport": {
                "path": (
                    "repo://video2world/examples/bedroom4/completion/"
                    "trellis2_pillow_front_seed42/composition-review/"
                    "static-recarve-report.json"
                ),
                "sha256": evidence.static_recarve_report_sha256,
            },
            "browserReview": {
                "path": (
                    "repo://video2world/examples/bedroom4/completion/"
                    "trellis2_pillow_front_seed42/composition-review/"
                    "static-recarve-browser-comparison.json"
                ),
                "sha256": evidence.static_recarve_browser_review_sha256,
                "status": evidence.static_recarve_browser_review["status"],
            },
            "promotionScope": "current_demo_only",
            "backgroundAndBedCompletion": "not_proven",
        },
        "qualityPolicy": {
            "accepted": (
                "overall shape is credible, main color is close, and interpenetration is not "
                "obvious"
            ),
            "minorAcceptedLimitations": [
                "minor texture or backside hallucination",
                "minor backside floral-pattern difference on the front pillow",
            ],
            "blockingConditions": [
                "obvious deformation",
                "wrong main color",
                "missing surface",
                "significant scene interpenetration",
            ],
        },
        "cleanPlateReview": {
            "status": evidence.clean_plate_review["status"],
            "selectedSeed": evidence.clean_plate_review["selected_seed"],
            "selectedCandidateSha256": selected["sha256"],
            "scope": "accepted_for_current_demo_only",
        },
        "backgroundGeometry": {
            "status": "not_proven",
            "statement": (
                "The accepted 2D clean plate does not prove object-free multi-view imagery or "
                "completed occluded bed/scene geometry. No clean 3D background is claimed."
            ),
        },
        "promotedManifestBrowserQa": {
            "status": "pending_promoted_manifest_recheck",
            "reason": (
                "The prior semantic-covariance candidate passed browser comparison, but the "
                "canonical manifest and new collider chunk transport require a fresh load."
            ),
        },
        "report": (
            "repo://video2world/examples/bedroom4/manifests/"
            "bedroom4.unified-pillow.candidate.report.json"
        ),
    }
    return manifest


def _stable_materialization_mode(source: Path, destination: Path) -> str:
    return "hardlink" if os.path.samefile(source, destination) else "verified_copy"


def materialize_candidate(
    *,
    project_root: Path,
    base_manifest: Path,
    output_manifest: Path,
    deployment_manifest: Path,
    promoted_manifest: Path,
    stable_manifest: Path,
    report_path: Path,
    asset_dir: Path,
    collider_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    project_root = project_root.resolve()
    base_manifest = base_manifest.resolve()
    preserve_stable_manifest(
        base_manifest=base_manifest,
        promoted_manifest=promoted_manifest.resolve(),
        stable_manifest=stable_manifest.resolve(),
    )
    evidence = load_evidence(project_root, base_manifest)
    output_assets = {
        FRONT_ID: asset_dir / "sam3_pillow_front.scene-fit.glb",
        **{object_id: asset_dir / f"{object_id}.ply" for object_id in REAR_IDS},
    }
    sources = {FRONT_ID: evidence.front_source, **evidence.rear_sources}
    expected = {FRONT_ID: FRONT_EXPECTED, **REAR_EXPECTED}
    materializations: list[dict[str, Any]] = []
    for object_id in (FRONT_ID, *REAR_IDS):
        record = materialize_file(
            sources[object_id],
            output_assets[object_id],
            expected_sha256=str(expected[object_id]["sha256"]),
            expected_size=int(expected[object_id]["size"]),
        )
        output_sha, output_size = sha256_file(record.path)
        require(output_sha == expected[object_id]["sha256"], f"output hash mismatch: {object_id}")
        require(output_size == expected[object_id]["size"], f"output size mismatch: {object_id}")
        materializations.append(
            {
                "object_id": object_id,
                "source": f"repo://video2world/{sources[object_id].relative_to(project_root).as_posix()}",
                "output": f"repo://video2world/{record.path.resolve().relative_to(project_root).as_posix()}",
                "storage": _stable_materialization_mode(sources[object_id], record.path),
                "sha256": output_sha,
                "bytes": output_size,
                "verified": True,
            }
        )

    publish_chunks = project_root / "web" / "public" / "worlds" / "bedroom4" / "chunks"
    source_static_chunks = (
        project_root
        / "web"
        / "public"
        / "worlds"
        / "bedroom4"
        / "recarve-candidates"
        / "semantic-covariance"
        / "chunks"
    )
    static_source_parts = evidence.static_recarve_report["staticVisual"]["parts"]
    static_parts, static_storage_modes = adopt_verified_chunks(
        static_source_parts,
        source_dir=source_static_chunks,
        output_dir=publish_chunks,
        public_prefix=STATIC_VISUAL_CHUNK_PUBLIC_PREFIX,
        stem=STATIC_VISUAL_CHUNK_STEM,
        expected_sha256=STATIC_VISUAL_EXPECTED["sha256"],
        expected_size=STATIC_VISUAL_EXPECTED["size"],
    )
    require(
        static_parts == _canonical_static_visual_parts(evidence),
        "adopted static visual chunks differ from the canonical manifest contract",
    )
    static_storage_counts = {
        mode: static_storage_modes.count(mode) for mode in sorted(set(static_storage_modes))
    }
    materializations.append(
        {
            "object_id": "scene_static_visual_front_residual_recarve",
            "source": (
                "repo://video2world/examples/bedroom4/completion/"
                "trellis2_pillow_front_seed42/composition-review/static-recarve-report.json"
            ),
            "output": "repo://video2world/web/public/worlds/bedroom4/chunks",
            "storage": "verified_chunk_adoption",
            "storage_modes": static_storage_counts,
            "sha256": STATIC_VISUAL_EXPECTED["sha256"],
            "bytes": STATIC_VISUAL_EXPECTED["size"],
            "vertices": STATIC_VISUAL_EXPECTED["vertices"],
            "chunk_count": len(static_parts),
            "maximum_chunk_bytes": max(int(part["size"]) for part in static_parts),
            "verified": True,
        }
    )

    collider_data = _read_verified_chunked_asset(evidence.base_collider_asset, publish_chunks)
    bounds_evidence = front_collider_carve_bounds(evidence)
    support_protection = front_collider_support_policy(evidence)
    carved_collider, collider_carve = carve_triangle_mesh_ply(
        collider_data,
        label="bedroom4 baseline static collider",
        bounds=bounds_evidence["expandedAabb"],
        support_protection=support_protection,
    )
    collider_path = collider_dir / "collider_bedroom4_tsdf_static_carved_unified_pillow.ply"
    collider_path.parent.mkdir(parents=True, exist_ok=True)
    collider_temporary = collider_path.with_name(f".{collider_path.name}.{os.getpid()}.tmp")
    collider_temporary.write_bytes(carved_collider)
    os.replace(collider_temporary, collider_path)
    collider_sha, collider_size = sha256_file(collider_path)
    require(
        collider_sha == hashlib.sha256(carved_collider).hexdigest(),
        "collider output hash mismatch",
    )
    require(collider_size == len(carved_collider), "collider output size mismatch")
    collider_parts = write_verified_chunks(
        carved_collider,
        output_dir=publish_chunks,
        public_prefix=COLLIDER_CHUNK_PUBLIC_PREFIX,
        stem=COLLIDER_CHUNK_STEM,
    )
    collider_carve.update(
        {
            "boundsEvidence": bounds_evidence,
            "outputSha256": collider_sha,
            "outputBytes": collider_size,
            "outputParts": copy.deepcopy(collider_parts),
            "chunkCount": len(collider_parts),
            "chunkSize": COLLIDER_CHUNK_SIZE,
            "maximumChunkBytes": max(int(part["size"]) for part in collider_parts),
            "directOversizedAssetPublished": False,
        }
    )
    materializations.append(
        {
            "object_id": "scene_static_collider_front_pillow_carve",
            "source": "verified_web_bundle_chunks:collider_bedroom4_tsdf_static_carved",
            "output": (
                f"repo://video2world/{collider_path.resolve().relative_to(project_root).as_posix()}"
            ),
            "storage": "generated_verified_triangle_carve",
            "sha256": collider_sha,
            "bytes": collider_size,
            "verified": True,
            "publish_transport": "verified_chunks",
            "publish_chunk_count": len(collider_parts),
            "maximum_publish_chunk_bytes": max(int(part["size"]) for part in collider_parts),
            "local_direct_audit_asset_retained": True,
            "release_package_published_as_direct_asset": False,
            "manifest_references_direct_audit_asset": False,
        }
    )

    manifest = build_candidate_manifest(
        read_json(base_manifest),
        evidence,
        collider_carve=collider_carve,
        collider_parts=collider_parts,
    )
    atomic_write_json(output_manifest, manifest)
    atomic_write_json(deployment_manifest, manifest)
    atomic_write_json(promoted_manifest, manifest)
    manifest_sha, manifest_size = sha256_file(output_manifest)
    deployment_sha, deployment_size = sha256_file(deployment_manifest)
    promoted_sha, promoted_size = sha256_file(promoted_manifest)
    stable_sha, stable_size = sha256_file(stable_manifest)
    require(deployment_sha == manifest_sha, "deployment manifest differs from example manifest")
    require(promoted_sha == manifest_sha, "canonical manifest differs from promoted candidate")
    require(
        stable_sha == STABLE_BASE_MANIFEST_SHA256,
        "stable manifest alias changed during promotion",
    )
    require(
        stable_manifest.read_bytes() == base_manifest.read_bytes(), "stable alias bytes changed"
    )
    script_path = Path(__file__).resolve()
    script_sha, _ = sha256_file(script_path)
    report = {
        "schema_version": "1.0",
        "kind": "video2world.bedroom4_unified_pillow_candidate_report",
        "created_at": CANDIDATE_CREATED_AT,
        "status": "materialized_pending_promoted_manifest_browser_qa",
        "candidate_version": CANDIDATE_VERSION,
        "completion_layer": 1,
        "completion_claim": "front_pillow_current_demo_candidate_only",
        "next_rounds_pending": [
            "generalized_multiview_amodal_object_completion",
            "occluded_bed_and_background_geometry_reconstruction",
            "additional_scene_object_completion_and_review",
        ],
        "materializer": {
            "path": f"repo://video2world/{script_path.relative_to(project_root).as_posix()}",
            "sha256": script_sha,
        },
        "inputs": {
            "base_manifest_sha256": evidence.base_sha256,
            "scene_fit_report_sha256": evidence.scene_fit_sha256,
            "instance_split_report_sha256": evidence.split_report_sha256,
            "clean_plate_visual_review_sha256": evidence.clean_plate_review_sha256,
            "canonical_six_view_review_sha256": evidence.canonical_six_view_review_sha256,
            "front_visual_review_sha256": evidence.front_visual_review_sha256,
            "static_recarve_report_sha256": evidence.static_recarve_report_sha256,
            "static_recarve_browser_review_sha256": (evidence.static_recarve_browser_review_sha256),
        },
        "output_manifest": {
            "path": f"repo://video2world/{output_manifest.resolve().relative_to(project_root).as_posix()}",
            "sha256": manifest_sha,
            "bytes": manifest_size,
            "validate_web_manifest": (
                "required_and_covered_by_tests/web/bedroom4-unified-pillow.test.js"
            ),
        },
        "deployment_manifest": {
            "path": (
                f"repo://video2world/"
                f"{deployment_manifest.resolve().relative_to(project_root).as_posix()}"
            ),
            "sha256": deployment_sha,
            "bytes": deployment_size,
            "gitignored_local_publish_artifact": True,
        },
        "promoted_manifest": {
            "path": (
                f"repo://video2world/"
                f"{promoted_manifest.resolve().relative_to(project_root).as_posix()}"
            ),
            "sha256": promoted_sha,
            "bytes": promoted_size,
            "byte_identical_to_deployment_manifest": True,
            "browser_qa": "pending_promoted_manifest_recheck",
        },
        "stable_manifest": {
            "path": (
                f"repo://video2world/"
                f"{stable_manifest.resolve().relative_to(project_root).as_posix()}"
            ),
            "sha256": stable_sha,
            "bytes": stable_size,
            "fixed_expected_sha256": STABLE_BASE_MANIFEST_SHA256,
            "byte_identical_to_base_manifest": True,
        },
        "materialized_assets": materializations,
        "front_pillow_gate": {
            "status": "passed_current_demo_only",
            "surface_collision": "passed",
            "basis": ["scene-fit technical gates", "user current-demo acceptance"],
            "root_proxy_browser_qa": "pending_promoted_manifest_recheck",
            "prior_recarve_browser_evidence": "passed",
            "topology": "surface_bvh",
            "watertight": False,
            "vertices": FRONT_EXPECTED["vertices"],
            "faces": FRONT_EXPECTED["faces"],
        },
        "primitive_counts": manifest["candidateBuild"]["primitiveCounts"],
        "scene_collider_carve": manifest["candidateBuild"]["sceneColliderCarve"],
        "static_visual_residual_carve": manifest["candidateBuild"]["staticVisualResidualCarve"],
        "quality_policy": manifest["candidateBuild"]["qualityPolicy"],
        "accepted_limitations": [
            (
                "The generated front-pillow backside has a minor floral-pattern difference; "
                "accepted and non-blocking under the current user review policy."
            ),
            (
                "Rear pillows restore observed visible RGB points only and do not claim completed "
                "hidden surfaces."
            ),
            (
                "Static residual removal does not reconstruct the hidden bed or background; "
                "perimeter dark-loss is 12.5426% and support-band dark-loss is 9.8490%."
            ),
        ],
        "clean_plate": manifest["candidateBuild"]["cleanPlateReview"],
        "background_geometry": manifest["candidateBuild"]["backgroundGeometry"],
        "promotion_scope": "current_demo_only",
        "promotion_blockers": ["promoted_manifest_root_proxy_browser_qa_pending"],
    }
    atomic_write_json(report_path, report)
    return manifest, report


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--base-manifest",
        type=Path,
        default=project_root / "examples" / "bedroom4" / "manifest.production.json",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=project_root
        / "examples"
        / "bedroom4"
        / "manifests"
        / "bedroom4.unified-pillow.candidate.web.json",
    )
    parser.add_argument(
        "--deployment-manifest",
        type=Path,
        default=project_root
        / "web"
        / "public"
        / "worlds"
        / "bedroom4"
        / "manifest.unified-pillow.json",
    )
    parser.add_argument(
        "--promoted-manifest",
        type=Path,
        default=project_root / "web" / "public" / "worlds" / "bedroom4" / "manifest.json",
    )
    parser.add_argument(
        "--stable-manifest",
        type=Path,
        default=project_root
        / "web"
        / "public"
        / "worlds"
        / "bedroom4"
        / "manifest.web-demo-baseline-stable.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=project_root
        / "examples"
        / "bedroom4"
        / "manifests"
        / "bedroom4.unified-pillow.candidate.report.json",
    )
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=project_root / "web" / "public" / "worlds" / "bedroom4" / "objects",
    )
    parser.add_argument(
        "--collider-dir",
        type=Path,
        default=project_root / "web" / "public" / "worlds" / "bedroom4" / "colliders",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest, report = materialize_candidate(
        project_root=args.project_root,
        base_manifest=args.base_manifest,
        output_manifest=args.output_manifest,
        deployment_manifest=args.deployment_manifest,
        promoted_manifest=args.promoted_manifest,
        stable_manifest=args.stable_manifest,
        report_path=args.report,
        asset_dir=args.asset_dir,
        collider_dir=args.collider_dir,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "version": manifest["version"],
                "manifest": str(args.output_manifest.resolve()),
                "deployment_manifest": str(args.deployment_manifest.resolve()),
                "promoted_manifest": str(args.promoted_manifest.resolve()),
                "stable_manifest": str(args.stable_manifest.resolve()),
                "report": str(args.report.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
