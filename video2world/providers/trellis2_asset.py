"""Official TRELLIS.2 mesh-first provider with fail-closed PBR GLB auditing."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import linecache
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from video2world.hashing import atomic_write_json, sha256_file

VERIFIED_SOURCE_REPOSITORY = "https://github.com/microsoft/TRELLIS.2.git"
VERIFIED_SOURCE_COMMIT = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
VERIFIED_MODEL_REPOSITORY = "microsoft/TRELLIS.2-4B"
VERIFIED_MODEL_REVISION = "af44b45f2e35a493886929c6d786e563ec68364d"
DEFAULT_CONFIG_FILE = "pipeline_512.json"
MAX_DECIMATION_FACES = 100_000
CANONICAL_SIX_VIEWS = ("front", "right", "back", "left", "top", "bottom")


class Trellis2AssetError(RuntimeError):
    """Raised when a TRELLIS.2 asset cannot satisfy the provider contract."""


@dataclass(frozen=True)
class Trellis2AssetRequest:
    trellis_source_dir: Path
    weights_dir: Path
    config_file: str
    input_image: Path
    output_dir: Path
    source_commit: str = VERIFIED_SOURCE_COMMIT
    model_repository: str = VERIFIED_MODEL_REPOSITORY
    model_revision: str = VERIFIED_MODEL_REVISION
    seed: int = 42
    pipeline_type: Literal["512", "1024", "1024_cascade", "1536_cascade"] = "512"
    decimation_target: int = MAX_DECIMATION_FACES
    texture_size: int = 1024
    allow_rmbg: bool = False
    debug_raw_mesh: bool = False
    debug_point_cloud: bool = False
    debug_convex: bool = False
    debug_surface_points: int = 200_000
    receipt_name: str = "trellis2_asset_receipt.json"


def artifact_record(path: Path) -> dict[str, Any]:
    digest, size = sha256_file(path)
    return {
        "path": str(path.resolve()),
        "size_bytes": size,
        "sha256": digest,
    }


def validate_generation_parameters(
    *,
    seed: int,
    decimation_target: int,
    texture_size: int,
    debug_surface_points: int,
) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise Trellis2AssetError("seed must be a non-negative signed 64-bit integer")
    if not 1 <= decimation_target <= MAX_DECIMATION_FACES:
        raise Trellis2AssetError(f"decimation_target must be between 1 and {MAX_DECIMATION_FACES}")
    if not 64 <= texture_size <= 8192 or texture_size & (texture_size - 1):
        raise Trellis2AssetError("texture_size must be a power of two between 64 and 8192")
    if not 1 <= debug_surface_points <= 5_000_000:
        raise Trellis2AssetError("debug_surface_points must be between 1 and 5000000")


def validate_input_image(path: Path, *, allow_rmbg: bool) -> dict[str, Any]:
    from PIL import Image

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise Trellis2AssetError(f"input image does not exist: {resolved}")
    try:
        with Image.open(resolved) as image:
            image.load()
            mode = image.mode
            width, height = image.size
            if width <= 0 or height <= 0:
                raise Trellis2AssetError("input image has invalid dimensions")
            alpha_range: list[int] | None = None
            if not allow_rmbg:
                if mode != "RGBA":
                    raise Trellis2AssetError(
                        "default TRELLIS.2 input must be RGBA; use --allow-rmbg explicitly "
                        "for RGB/background-removal input"
                    )
                alpha = image.getchannel("A")
                alpha_min, alpha_max = alpha.getextrema()
                alpha_range = [int(alpha_min), int(alpha_max)]
                if alpha_min == alpha_max:
                    raise Trellis2AssetError(
                        "RGBA input must contain both foreground and transparent background"
                    )
                if alpha_max == 0:
                    raise Trellis2AssetError("RGBA input alpha contains no foreground")
            elif mode not in {"RGB", "RGBA"}:
                raise Trellis2AssetError("RMBG input must be RGB or RGBA")
    except Trellis2AssetError:
        raise
    except Exception as exc:
        raise Trellis2AssetError(f"cannot decode input image {resolved}: {exc}") from exc
    digest, size = sha256_file(resolved)
    return {
        "path": str(resolved),
        "sha256": digest,
        "size_bytes": size,
        "mode": mode,
        "width": width,
        "height": height,
        "alpha_range": alpha_range,
        "background_removal": "explicit_opt_in" if allow_rmbg else "disabled_rgba_required",
    }


def resolve_config_path(weights_dir: Path, config_file: str) -> Path:
    candidate = Path(config_file).expanduser()
    if not candidate.is_absolute():
        candidate = weights_dir.expanduser().resolve() / candidate
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise Trellis2AssetError(f"TRELLIS.2 config file does not exist: {resolved}")
    return resolved


def pretrained_config_name(weights_dir: Path, config_path: Path) -> str:
    """Return the weights-relative config name expected by TRELLIS.2.

    The upstream loader tests ``<weights>/<config_file>`` to decide whether a
    model is local. Passing the already-resolved absolute path makes that test
    fail and incorrectly routes a local checkout through Hugging Face.
    """

    weights_root = weights_dir.expanduser().resolve()
    resolved = config_path.expanduser().resolve()
    try:
        relative = resolved.relative_to(weights_root)
    except ValueError as exc:
        raise Trellis2AssetError(
            f"TRELLIS.2 config must be inside the weights directory: {resolved}"
        ) from exc
    if not relative.parts:
        raise Trellis2AssetError("TRELLIS.2 config path cannot be the weights directory")
    return relative.as_posix()


def git_source_provenance(source_dir: Path, *, expected_commit: str) -> dict[str, Any]:
    resolved = source_dir.expanduser().resolve()
    if not resolved.is_dir():
        raise Trellis2AssetError(f"TRELLIS.2 source directory does not exist: {resolved}")
    try:
        commit = subprocess.run(
            ["git", "-C", str(resolved), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        remote = subprocess.run(
            ["git", "-C", str(resolved), "config", "--get", "remote.origin.url"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise Trellis2AssetError(f"cannot inspect TRELLIS.2 source checkout: {exc}") from exc
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise Trellis2AssetError(f"TRELLIS.2 source commit is invalid: {commit!r}")
    if commit != expected_commit:
        raise Trellis2AssetError(
            f"TRELLIS.2 source commit mismatch: expected {expected_commit}, got {commit}"
        )
    return {
        "repository": remote or VERIFIED_SOURCE_REPOSITORY,
        "path": str(resolved),
        "expected_commit": expected_commit,
        "commit": commit,
        "verified": True,
    }


def model_revision_evidence(weights_dir: Path, *, expected_revision: str) -> dict[str, Any]:
    resolved = weights_dir.expanduser().resolve()
    if not resolved.is_dir():
        raise Trellis2AssetError(f"TRELLIS.2 weights directory does not exist: {resolved}")
    metadata_files = sorted(resolved.glob(".cache/huggingface/download/**/*.metadata"))
    revisions: dict[str, str] = {}
    for metadata_path in metadata_files:
        try:
            first_line = metadata_path.read_text(encoding="utf-8").splitlines()[0].strip()
        except (OSError, IndexError, UnicodeError):
            continue
        if re.fullmatch(r"[0-9a-f]{40}", first_line):
            revisions[str(metadata_path.relative_to(resolved))] = first_line
    distinct = sorted(set(revisions.values()))
    mismatches = sorted(revision for revision in distinct if revision != expected_revision)
    if mismatches:
        raise Trellis2AssetError(
            "TRELLIS.2 model revision metadata mismatch: " + ", ".join(mismatches)
        )
    return {
        "path": str(resolved),
        "expected_revision": expected_revision,
        "metadata_file_count": len(revisions),
        "metadata_revisions": distinct,
        "verification": (
            "huggingface_download_metadata" if revisions else "declared_revision_no_local_metadata"
        ),
    }


def audit_triangle_mesh(vertices: Any, faces: Any) -> dict[str, Any]:
    import numpy as np

    vertex_array = np.asarray(vertices, dtype=np.float64)
    face_array = np.asarray(faces)
    finite_vertices = bool(
        vertex_array.ndim == 2
        and vertex_array.shape[1:] == (3,)
        and np.isfinite(vertex_array).all()
    )
    triangular_faces = bool(
        face_array.ndim == 2 and face_array.shape[1:] == (3,) and len(face_array) > 0
    )
    integer_indices = bool(np.issubdtype(face_array.dtype, np.integer))
    valid_indices = bool(
        triangular_faces
        and integer_indices
        and face_array.min(initial=0) >= 0
        and face_array.max(initial=-1) < len(vertex_array)
    )
    degenerate_faces = 0
    if finite_vertices and valid_indices:
        triangles = vertex_array[face_array]
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        extents = np.ptp(vertex_array, axis=0)
        scale = max(float(extents.max(initial=0.0)), 1e-9)
        threshold = max(scale * scale * 1e-12, 1e-18)
        degenerate_faces = int((np.linalg.norm(cross, axis=1) <= threshold).sum())
    return {
        "vertices": len(vertex_array) if vertex_array.ndim > 0 else 0,
        "faces": len(face_array) if face_array.ndim > 0 else 0,
        "finite_vertices": finite_vertices,
        "triangular_faces": triangular_faces,
        "integer_indices": integer_indices,
        "valid_indices": valid_indices,
        "degenerate_faces": degenerate_faces,
    }


def _unit_scalar_evidence(value: Any, *, default: float) -> dict[str, Any]:
    import numpy as np

    explicit = value is not None
    effective = value if explicit else default
    valid_scalar = not isinstance(effective, (bool, np.bool_))
    try:
        array = np.asarray(effective)
        valid_scalar = valid_scalar and array.ndim == 0
        number = float(array) if valid_scalar else None
    except (TypeError, ValueError, OverflowError):
        number = None
        valid_scalar = False
    finite = bool(valid_scalar and number is not None and np.isfinite(number))
    in_unit_range = bool(finite and 0.0 <= number <= 1.0)
    return {
        "value": number if finite else None,
        "source": "explicit" if explicit else "gltf_default",
        "default": default,
        "finite": finite,
        "in_unit_range": in_unit_range,
        "valid": finite and in_unit_range,
        "raw_repr": None if finite else repr(effective),
    }


def _base_color_factor_evidence(value: Any) -> dict[str, Any]:
    import numpy as np

    explicit = value is not None
    effective = np.ones(4, dtype=np.float64) if value is None else np.asarray(value)
    valid_shape = effective.shape == (4,)
    try:
        if np.issubdtype(effective.dtype, np.bool_):
            normalized = effective.astype(np.float64)
        elif np.issubdtype(effective.dtype, np.integer):
            normalized = effective.astype(np.float64) / np.iinfo(effective.dtype).max
        elif np.issubdtype(effective.dtype, np.floating):
            normalized = effective.astype(np.float64)
        else:
            normalized = np.asarray([], dtype=np.float64)
            valid_shape = False
    except (TypeError, ValueError, OverflowError):
        normalized = np.asarray([], dtype=np.float64)
        valid_shape = False
    finite = bool(valid_shape and np.isfinite(normalized).all())
    in_unit_range = bool(finite and (normalized >= 0.0).all() and (normalized <= 1.0).all())
    rgba = normalized.tolist() if finite else None
    return {
        "rgba": rgba,
        "source": "explicit" if explicit else "gltf_default",
        "finite": finite,
        "in_unit_range": in_unit_range,
        "valid": finite and in_unit_range,
        "alpha": rgba[3] if rgba is not None else None,
        "raw_dtype": str(getattr(effective, "dtype", type(effective).__name__)),
        "raw_repr": None if finite else repr(value),
    }


def _base_color_texture_evidence(image: Any) -> dict[str, Any]:
    import numpy as np

    if image is None:
        return {
            "present": False,
            "valid": True,
            "evidence_status": "not_present_optional",
        }
    try:
        pixels = np.asarray(image)
        if pixels.size == 0:
            raise ValueError("decoded image is empty")
        if np.issubdtype(pixels.dtype, np.bool_):
            normalized = pixels.astype(np.float64)
        elif np.issubdtype(pixels.dtype, np.integer):
            normalized = pixels.astype(np.float64) / np.iinfo(pixels.dtype).max
        elif np.issubdtype(pixels.dtype, np.floating):
            normalized = pixels.astype(np.float64)
        else:
            raise TypeError(f"unsupported decoded image dtype: {pixels.dtype}")
        finite = bool(np.isfinite(normalized).all())
        minimum = float(normalized.min()) if finite else None
        maximum = float(normalized.max()) if finite else None
        in_unit_range = bool(finite and minimum is not None and 0.0 <= minimum <= maximum <= 1.0)
        size = getattr(image, "size", None)
        valid_size = bool(
            isinstance(size, tuple)
            and len(size) == 2
            and all(isinstance(value, int) and value > 0 for value in size)
        )
        bands = list(image.getbands()) if callable(getattr(image, "getbands", None)) else []
        alpha_index = bands.index("A") if "A" in bands else None
        alpha_range = None
        if alpha_index is not None and normalized.ndim >= 3:
            alpha = normalized[..., alpha_index]
            alpha_range = [float(alpha.min()), float(alpha.max())]
        decoded = np.ascontiguousarray(pixels)
        return {
            "present": True,
            "valid": finite and in_unit_range and valid_size,
            "evidence_status": "decoded",
            "image_type": type(image).__name__,
            "format": getattr(image, "format", None),
            "mode": getattr(image, "mode", None),
            "size": list(size) if valid_size else None,
            "bands": bands,
            "decoded_shape": list(pixels.shape),
            "decoded_dtype": str(pixels.dtype),
            "decoded_pixel_sha256": hashlib.sha256(decoded.tobytes()).hexdigest(),
            "finite": finite,
            "normalized_range": [minimum, maximum],
            "in_unit_range": in_unit_range,
            "alpha_channel_present": alpha_index is not None,
            "alpha_normalized_range": alpha_range,
        }
    except Exception as exc:
        return {
            "present": True,
            "valid": False,
            "evidence_status": "decode_failed",
            "image_type": type(image).__name__,
            "error": f"{type(exc).__name__}: {exc}",
        }


def audit_pbr_material(material: Any) -> dict[str, Any]:
    import numpy as np

    material_type = type(material).__name__ if material is not None else "None"
    if material_type != "PBRMaterial":
        return {
            "material_type": material_type,
            "pbr_material": False,
            "technical_status": "failed",
            "failed_gates": ["pbr_material"],
        }

    data = getattr(material, "_data", {})
    data = data if isinstance(data, dict) else {}
    metallic = _unit_scalar_evidence(getattr(material, "metallicFactor", None), default=1.0)
    roughness = _unit_scalar_evidence(getattr(material, "roughnessFactor", None), default=1.0)
    base_color = _base_color_factor_evidence(getattr(material, "baseColorFactor", None))
    texture = _base_color_texture_evidence(getattr(material, "baseColorTexture", None))
    alpha_mode_value = getattr(material, "alphaMode", None)
    alpha_mode = alpha_mode_value if alpha_mode_value is not None else "OPAQUE"
    alpha_mode_valid = isinstance(alpha_mode, str) and alpha_mode in {"OPAQUE", "MASK", "BLEND"}
    alpha_cutoff = _unit_scalar_evidence(getattr(material, "alphaCutoff", None), default=0.5)
    double_sided_value = getattr(material, "doubleSided", False)
    double_sided_valid = isinstance(double_sided_value, (bool, np.bool_))
    double_sided = {
        "value": bool(double_sided_value) if double_sided_valid else None,
        "source": "explicit" if "doubleSided" in data else "gltf_default",
        "valid_boolean": double_sided_valid,
        "raw_repr": None if double_sided_valid else repr(double_sided_value),
    }
    gates = {
        "metallic_factor_finite_unit_range": metallic["valid"],
        "roughness_factor_finite_unit_range": roughness["valid"],
        "base_color_factor_finite_unit_range": base_color["valid"],
        "base_color_texture_evidence_valid": texture["valid"],
        "alpha_mode_valid": alpha_mode_valid,
        "alpha_cutoff_finite_unit_range": alpha_cutoff["valid"],
        "double_sided_boolean": double_sided_valid,
    }
    return {
        "material_type": material_type,
        "name": str(material.name) if getattr(material, "name", None) is not None else None,
        "pbr_material": True,
        "metallic_factor": metallic,
        "roughness_factor": roughness,
        "base_color_factor": base_color,
        "base_color_texture": texture,
        "alpha": {
            "mode": alpha_mode if isinstance(alpha_mode, str) else None,
            "mode_source": "explicit" if "alphaMode" in data else "gltf_default",
            "mode_valid": alpha_mode_valid,
            "cutoff": alpha_cutoff,
            "base_color_factor": base_color["alpha"],
            "texture_channel_present": texture.get("alpha_channel_present", False),
            "texture_normalized_range": texture.get("alpha_normalized_range"),
        },
        "double_sided": double_sided,
        "technical_gates": gates,
        "technical_status": "passed" if all(gates.values()) else "failed",
        "failed_gates": sorted(name for name, passed in gates.items() if not passed),
        "semantic_review": {
            "status": "pending_scene_fit_material_profile_or_vlm",
            "metallic_one_is_not_a_generic_failure": True,
        },
    }


def audit_pbr_glb(path: Path, *, face_limit: int = MAX_DECIMATION_FACES) -> dict[str, Any]:
    import numpy as np
    import trimesh

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise Trellis2AssetError(f"required asset_pbr.glb does not exist: {resolved}")
    if face_limit < 1 or face_limit > MAX_DECIMATION_FACES:
        raise Trellis2AssetError(f"face_limit must be between 1 and {MAX_DECIMATION_FACES}")
    try:
        loaded = trimesh.load(resolved, force="scene", process=False)
    except Exception as exc:
        raise Trellis2AssetError(f"cannot load final PBR GLB: {exc}") from exc
    scene = loaded if isinstance(loaded, trimesh.Scene) else trimesh.Scene(loaded)
    node_names = list(scene.graph.nodes_geometry)
    if not node_names:
        raise Trellis2AssetError("final PBR GLB contains no mesh nodes")

    mesh_records: list[dict[str, Any]] = []
    all_vertices: list[Any] = []
    total_faces = 0
    material_types: list[str] = []
    finite_transforms = True
    all_winding_consistent = True
    all_watertight = True
    for node_name in node_names:
        transform, geometry_name = scene.graph[node_name]
        transform_array = np.asarray(transform, dtype=np.float64)
        finite_transforms = finite_transforms and bool(np.isfinite(transform_array).all())
        mesh = scene.geometry[geometry_name].copy()
        if finite_transforms:
            mesh.apply_transform(transform_array)
        triangle_audit = audit_triangle_mesh(mesh.vertices, mesh.faces)
        material = getattr(getattr(mesh, "visual", None), "material", None)
        material_audit = audit_pbr_material(material)
        material_type = material_audit["material_type"]
        material_types.append(material_type)
        winding_consistent = bool(mesh.is_winding_consistent)
        watertight = bool(mesh.is_watertight)
        all_winding_consistent = all_winding_consistent and winding_consistent
        all_watertight = all_watertight and watertight
        total_faces += triangle_audit["faces"]
        if triangle_audit["finite_vertices"]:
            all_vertices.append(np.asarray(mesh.vertices, dtype=np.float64))
        mesh_records.append(
            {
                "node": str(node_name),
                "geometry": str(geometry_name),
                **triangle_audit,
                "winding_consistent": winding_consistent,
                "watertight": watertight,
                "material_type": material_type,
                "material": material_audit,
            }
        )

    bounds = None
    extents = None
    if all_vertices:
        merged_vertices = np.concatenate(all_vertices, axis=0)
        minimum = merged_vertices.min(axis=0)
        maximum = merged_vertices.max(axis=0)
        bounds = [minimum.tolist(), maximum.tolist()]
        extents = (maximum - minimum).tolist()
    pbr_material_count = sum(material_type == "PBRMaterial" for material_type in material_types)
    gates = {
        "nonempty_mesh": bool(mesh_records and total_faces > 0),
        "face_budget": 0 < total_faces <= face_limit <= MAX_DECIMATION_FACES,
        "finite_transforms": finite_transforms,
        "finite_vertices": all(record["finite_vertices"] for record in mesh_records),
        "valid_triangle_indices": all(
            record["triangular_faces"] and record["integer_indices"] and record["valid_indices"]
            for record in mesh_records
        ),
        "no_degenerate_faces": all(record["degenerate_faces"] == 0 for record in mesh_records),
        "winding_consistent": all_winding_consistent,
        "pbr_material_present": pbr_material_count == len(mesh_records),
        "pbr_material_factors_valid": all(
            record["material"]
            .get("technical_gates", {})
            .get("metallic_factor_finite_unit_range", False)
            and record["material"]
            .get("technical_gates", {})
            .get("roughness_factor_finite_unit_range", False)
            and record["material"]
            .get("technical_gates", {})
            .get("base_color_factor_finite_unit_range", False)
            for record in mesh_records
        ),
        "base_color_texture_evidence_valid": all(
            record["material"]
            .get("technical_gates", {})
            .get("base_color_texture_evidence_valid", False)
            for record in mesh_records
        ),
        "alpha_double_sided_evidence_valid": all(
            record["material"].get("technical_gates", {}).get("alpha_mode_valid", False)
            and record["material"]
            .get("technical_gates", {})
            .get("alpha_cutoff_finite_unit_range", False)
            and record["material"].get("technical_gates", {}).get("double_sided_boolean", False)
            for record in mesh_records
        ),
        "positive_extents": bool(extents and all(value > 0 for value in extents)),
    }
    topology = "closed_volume" if all_watertight else "surface_bvh"
    return {
        "asset": artifact_record(resolved),
        "mesh_count": len(mesh_records),
        "vertices": sum(record["vertices"] for record in mesh_records),
        "faces": total_faces,
        "bounds": bounds,
        "extents": extents,
        "material_count": len(material_types),
        "pbr_material_count": pbr_material_count,
        "material_types": material_types,
        "material_semantic_review": {
            "status": "pending_scene_fit_material_profile_or_vlm",
            "metallic_one_is_not_a_generic_failure": True,
        },
        "winding_consistent": all_winding_consistent,
        "watertight": all_watertight,
        "collision_topology": topology,
        "surface_blocking": True,
        "closed_volume_claim": topology == "closed_volume",
        "inside_outside_queries_allowed": topology == "closed_volume",
        "face_limit": face_limit,
        "meshes": mesh_records,
        "technical_gates": gates,
        "technical_status": "passed" if all(gates.values()) else "failed",
        "failed_gates": sorted(name for name, passed in gates.items() if not passed),
    }


def unified_pbr_glb_provenance(audit: dict[str, Any]) -> dict[str, Any]:
    """Return the manifest-ready topology evidence for a TRELLIS.2 PBR GLB."""

    faces = audit.get("faces")
    gates = audit.get("technical_gates")
    return {
        "mesh_count": audit.get("mesh_count"),
        "vertices": audit.get("vertices"),
        "faces": faces,
        "face_count": faces,
        "bounds": audit.get("bounds"),
        "extents": audit.get("extents"),
        "face_limit": audit.get("face_limit"),
        "winding_consistent": audit.get("winding_consistent"),
        "watertight": audit.get("watertight"),
        "surface_blocking": audit.get("surface_blocking"),
        "closed_volume_claim": audit.get("closed_volume_claim"),
        "inside_outside_queries_allowed": audit.get("inside_outside_queries_allowed"),
        "technical_status": audit.get("technical_status"),
        "technical_gates": dict(gates) if isinstance(gates, dict) else {},
    }


def canonical_six_view_contract() -> dict[str, Any]:
    return {
        "status": "pending",
        "required_receipt_kind": "video2world.canonical_object_six_view_review",
        "required_views": list(CANONICAL_SIX_VIEWS),
        "renderer": "web/object-review.html",
        "capture_driver": "scripts/capture_object_review.mjs",
        "orthogonal_axes_required": True,
        "horizontal_orbit_contact_sheet_is_not_six_view_evidence": True,
        "promotion_allowed": False,
        "severity_policy": {
            "blocking": [
                "shape_mismatch",
                "primary_color_category_error",
                "missing_surface_or_sheet_geometry",
                "disconnected_or_floating_parts",
                "significant_scene_interpenetration",
            ],
            "warning": [
                "minor_backside_texture_hallucination",
                "minor_material_or_pattern_drift",
            ],
        },
    }


def _install_python311_triton_compatibility() -> None:
    if sys.version_info < (3, 11) or getattr(inspect, "_video2world_trellis2_patch", False):
        return
    original_getsource = inspect.getsource

    def compatible_getsource(obj: Any) -> str:
        source = original_getsource(obj)
        if inspect.isfunction(obj) and re.search(r"^def\s+\w+\s*\(", source, re.MULTILINE) is None:
            filename = inspect.getsourcefile(obj)
            if filename:
                lines = linecache.getlines(filename)
                needle = f"def {obj.__name__}("
                start = max(0, obj.__code__.co_firstlineno - 1)
                for line_number in range(start, len(lines)):
                    if lines[line_number].lstrip().startswith(needle):
                        return "".join(inspect.getblock(lines[line_number:]))
        return source

    inspect.getsource = compatible_getsource
    inspect._video2world_trellis2_patch = True  # type: ignore[attr-defined]


def _disable_rmbg_for_rgba_input() -> None:
    import importlib

    from PIL import Image

    rembg = importlib.import_module("trellis2.pipelines.rembg")

    class AlphaInputOnly:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def to(self, device: Any = None) -> None:
            del device

        cuda = to
        cpu = to

        def __call__(self, image: Image.Image) -> Image.Image:
            del image
            raise RuntimeError("RMBG is disabled; provide the validated RGBA input")

    rembg.BiRefNet = AlphaInputOnly


def _raw_mesh_from_trellis(mesh: Any) -> tuple[Any, Any, Any, Any]:
    import numpy as np
    import trimesh

    vertex_attrs = mesh.query_vertex_attrs().detach().float().cpu().numpy()
    vertices = mesh.vertices.detach().float().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy().astype(np.int64)
    base_color = np.clip(vertex_attrs[:, mesh.layout["base_color"]], 0.0, 1.0)
    alpha = np.clip(vertex_attrs[:, mesh.layout["alpha"]], 0.0, 1.0)
    rgba = np.concatenate([base_color, alpha], axis=1)
    rgba_u8 = np.round(rgba * 255).astype(np.uint8)
    raw_mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=rgba_u8,
        process=False,
    )
    return raw_mesh, vertices, faces, rgba_u8


def run_official_trellis2(request: Trellis2AssetRequest, config_path: Path) -> dict[str, Any]:
    os.environ.setdefault("ATTN_BACKEND", "xformers")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")
    os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    _install_python311_triton_compatibility()

    source_dir = request.trellis_source_dir.expanduser().resolve()
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    if not request.allow_rmbg:
        _disable_rmbg_for_rgba_input()

    import o_voxel
    import torch
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    torch.set_float32_matmul_precision("high")
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
        str(request.weights_dir.expanduser().resolve()),
        config_file=pretrained_config_name(request.weights_dir, config_path),
    )
    pipeline.cuda()
    load_seconds = time.perf_counter() - load_started

    image = Image.open(request.input_image).convert("RGBA")
    processed = pipeline.preprocess_image(image)
    processed_path = request.output_dir / "processed_input.png"
    processed.save(processed_path)
    inference_started = time.perf_counter()
    meshes, latent = pipeline.run(
        processed,
        seed=request.seed,
        pipeline_type=request.pipeline_type,
        preprocess_image=False,
        return_latent=True,
    )
    if not meshes:
        raise Trellis2AssetError("TRELLIS.2 returned no mesh")
    mesh = meshes[0]
    _, _, resolution = latent
    inference_seconds = time.perf_counter() - inference_started
    export_started = time.perf_counter()
    pbr_scene = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        grid_size=int(resolution),
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=request.decimation_target,
        texture_size=request.texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
    )
    pbr_path = request.output_dir / "asset_pbr.glb"
    pbr_scene.export(pbr_path, extension_webp=True)
    pbr_export_seconds = time.perf_counter() - export_started

    debug_artifacts: list[dict[str, Any]] = []
    raw_mesh = None
    rgba_u8 = None
    if request.debug_raw_mesh or request.debug_point_cloud or request.debug_convex:
        raw_mesh, _, _, rgba_u8 = _raw_mesh_from_trellis(mesh)
    if request.debug_raw_mesh:
        assert raw_mesh is not None
        raw_path = request.output_dir / "debug_raw_vertex_color.glb"
        raw_mesh.export(raw_path)
        debug_artifacts.append(artifact_record(raw_path))
    if request.debug_point_cloud:
        import numpy as np
        import trimesh

        assert raw_mesh is not None
        assert rgba_u8 is not None
        points, face_ids = trimesh.sample.sample_surface(
            raw_mesh,
            request.debug_surface_points,
        )
        point_colors = rgba_u8[raw_mesh.faces[face_ids]].mean(axis=1).astype(np.uint8)
        cloud_path = request.output_dir / "debug_surface_points_rgb.ply"
        trimesh.points.PointCloud(points, colors=point_colors).export(cloud_path)
        debug_artifacts.append(artifact_record(cloud_path))
    if request.debug_convex:
        import trimesh

        assert raw_mesh is not None
        convex_path = request.output_dir / "debug_collision_convex.obj"
        convex_path.write_text(
            trimesh.exchange.obj.export_obj(raw_mesh.convex_hull),
            encoding="utf-8",
        )
        debug_artifacts.append(artifact_record(convex_path))

    return {
        "processed_input": artifact_record(processed_path),
        "pbr_asset": artifact_record(pbr_path),
        "debug_artifacts": debug_artifacts,
        "decoded_resolution": int(resolution),
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "pbr_export_seconds": pbr_export_seconds,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / (1024**3),
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_device": str(torch.cuda.get_device_name(0)),
        },
    }


Generator = Callable[[Trellis2AssetRequest, Path], dict[str, Any]]


def run_trellis2_asset(
    request: Trellis2AssetRequest,
    *,
    generator: Generator = run_official_trellis2,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    request.output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = request.output_dir / request.receipt_name
    timestamp = created_at or datetime.now(UTC)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise Trellis2AssetError("created_at must include timezone information")
    receipt: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": "video2world.trellis2_mesh_first_asset",
        "created_at": timestamp.isoformat(),
        "provider": "official_microsoft_trellis2_mesh_first",
        "status": "running",
        "promotion_allowed": False,
        "required_output": "asset_pbr.glb",
        "source": {},
        "model": {},
        "config": {},
        "input": {},
        "generation": {
            "seed": request.seed,
            "pipeline_type": request.pipeline_type,
            "decimation_target": request.decimation_target,
            "maximum_faces": MAX_DECIMATION_FACES,
            "texture_size": request.texture_size,
            "allow_rmbg": request.allow_rmbg,
        },
        "outputs": {},
        "technical_audit": None,
        "visual_review": canonical_six_view_contract(),
        "debug_exports": {
            "raw_mesh": request.debug_raw_mesh,
            "point_cloud": request.debug_point_cloud,
            "convex": request.debug_convex,
            "artifacts": [],
        },
    }
    try:
        validate_generation_parameters(
            seed=request.seed,
            decimation_target=request.decimation_target,
            texture_size=request.texture_size,
            debug_surface_points=request.debug_surface_points,
        )
        receipt["source"] = git_source_provenance(
            request.trellis_source_dir,
            expected_commit=request.source_commit,
        )
        receipt["model"] = {
            "repository": request.model_repository,
            "revision": request.model_revision,
            **model_revision_evidence(
                request.weights_dir,
                expected_revision=request.model_revision,
            ),
        }
        config_path = resolve_config_path(request.weights_dir, request.config_file)
        config_sha, config_size = sha256_file(config_path)
        receipt["config"] = {
            "path": str(config_path),
            "sha256": config_sha,
            "size_bytes": config_size,
        }
        receipt["input"] = validate_input_image(
            request.input_image,
            allow_rmbg=request.allow_rmbg,
        )
        generated = generator(request, config_path)
        pbr_path = request.output_dir / "asset_pbr.glb"
        if not pbr_path.is_file() or pbr_path.stat().st_size <= 0:
            raise Trellis2AssetError("official TRELLIS.2 export did not produce asset_pbr.glb")
        audit = audit_pbr_glb(pbr_path, face_limit=request.decimation_target)
        receipt["technical_audit"] = audit
        receipt["outputs"] = {
            "unified_pbr_glb": {
                **audit["asset"],
                "role": "unified_pbr_glb",
                "media_type": "model/gltf-binary",
                "status": "candidate",
                "collision_topology": audit["collision_topology"],
                "provenance": unified_pbr_glb_provenance(audit),
            },
            "processed_input": generated.get("processed_input"),
        }
        receipt["debug_exports"]["artifacts"] = generated.get("debug_artifacts", [])
        receipt["runtime"] = {
            key: value
            for key, value in generated.items()
            if key not in {"pbr_asset", "processed_input", "debug_artifacts"}
        }
        if audit["technical_status"] != "passed":
            raise Trellis2AssetError(
                "final asset_pbr.glb failed technical gates: " + ", ".join(audit["failed_gates"])
            )
        receipt["status"] = "technical_passed_visual_pending"
    except Exception as exc:
        receipt["status"] = "failed"
        receipt["promotion_allowed"] = False
        receipt["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        atomic_write_json(receipt_path, receipt)
        if isinstance(exc, Trellis2AssetError):
            raise
        raise Trellis2AssetError(str(exc)) from exc
    atomic_write_json(receipt_path, receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and audit one mesh-first PBR GLB with official TRELLIS.2."
    )
    parser.add_argument("--trellis-source-dir", type=Path, required=True)
    parser.add_argument("--weights", dest="weights_dir", type=Path, required=True)
    parser.add_argument("--config-file", default=DEFAULT_CONFIG_FILE)
    parser.add_argument("--input-rgba", dest="input_image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", default=VERIFIED_SOURCE_COMMIT)
    parser.add_argument("--model-repository", default=VERIFIED_MODEL_REPOSITORY)
    parser.add_argument("--model-revision", default=VERIFIED_MODEL_REVISION)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pipeline-type",
        choices=["512", "1024", "1024_cascade", "1536_cascade"],
        default="512",
    )
    parser.add_argument("--decimation-target", type=int, default=MAX_DECIMATION_FACES)
    parser.add_argument("--texture-size", type=int, default=1024)
    parser.add_argument(
        "--allow-rmbg",
        action="store_true",
        help="Explicitly allow TRELLIS.2 background removal; default input must be RGBA.",
    )
    parser.add_argument("--debug-raw-mesh", action="store_true")
    parser.add_argument("--debug-point-cloud", action="store_true")
    parser.add_argument("--debug-convex", action="store_true")
    parser.add_argument("--debug-surface-points", type=int, default=200_000)
    parser.add_argument("--receipt-name", default="trellis2_asset_receipt.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    request = Trellis2AssetRequest(**vars(args))
    try:
        receipt = run_trellis2_asset(request)
    except Trellis2AssetError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "asset": receipt["outputs"]["unified_pbr_glb"]["path"],
                "receipt": str((request.output_dir / request.receipt_name).resolve()),
                "visual_review": "pending canonical six-view receipt",
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
