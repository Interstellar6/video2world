#!/usr/bin/env python3
"""Materialize a local-only front-half scene composite from verified candidates.

This joins a multi-view carved static Gaussian PLY with world-baked TRELLIS
objects. It deliberately leaves the original static collider untouched and
does not publish or promote the resulting manifest.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
from plyfile import PlyData

from scripts.materialize_direct_trellis_scene_candidate import (
    audit_pgsr_ply,
    audit_unified_glb,
)
from video2world.hashing import atomic_write_json, sha256_file

CONFIG_KIND = "video2world.front_half_scene_candidate_config"
MANIFEST_KIND = "video2world.front_half_scene_candidate_manifest"
REPORT_KIND = "video2world.front_half_scene_candidate_report"
CARVE_KIND = "video2world.multiview_depth_static_gaussian_carve"
SCENE_FIT_KIND = "video2world.completed_object_scene_fit"
CARVED_STATIC_ROLE = "candidate_multiview_depth_carved_pgsr_scene"
ORIGINAL_STATIC_ROLE = "original_uncarved_pgsr_full_scene"
ORIGINAL_COLLIDER_ROLE = "original_uncarved_scene_mesh"


@dataclass(frozen=True)
class CarveEvidence:
    receipt_path: Path
    receipt_sha256: str
    source_static_scene: dict[str, Any]
    output_static_scene: dict[str, Any]
    analysis: dict[str, Any]


@dataclass(frozen=True)
class ObjectEvidence:
    object_id: str
    receipt_path: Path
    receipt_sha256: str
    category: str
    label: str
    glb_path: Path
    glb_sha256: str
    glb_bytes: int
    vertices: int
    faces: int
    bounds: list[list[float]]
    watertight: bool


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path, *, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {label}: {path}: {exc}") from exc


def resolve_path(value: Any, *, parent: Path, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label}.path must be a non-empty string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = parent / path
    path = path.resolve()
    require(path.is_file(), f"{label} does not exist: {path}")
    return path


def require_hash(path: Path, expected: Any, *, label: str) -> tuple[str, int]:
    require(
        isinstance(expected, str) and len(expected) == 64,
        f"{label}.sha256 must be a 64-character digest",
    )
    observed, size = sha256_file(path)
    require(observed == expected, f"{label} SHA-256 mismatch: {observed} != {expected}")
    return observed, size


def local_fs_url(path: Path) -> str:
    return "/@fs" + quote(path.resolve().as_posix(), safe="/")


def same_bounds(left: Any, right: Any, *, tolerance: float) -> bool:
    try:
        left_values = np.asarray(left, dtype=np.float64)
        right_values = np.asarray(right, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        left_values.shape == right_values.shape == (2, 3)
        and np.isfinite(left_values).all()
        and np.isfinite(right_values).all()
        and np.allclose(left_values, right_values, rtol=0, atol=tolerance)
    )


def audit_collider(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    digest, size = require_hash(path, expected_sha256, label="static_collider")
    try:
        ply = PlyData.read(path, mmap=True)
    except Exception as exc:  # pragma: no cover - parser detail differs by plyfile version
        raise RuntimeError(f"cannot inspect static collider {path}: {exc}") from exc
    require(not ply.text and ply.byte_order == "<", "static collider must be binary little-endian")
    require("vertex" in ply and "face" in ply, "static collider needs vertex and face elements")
    vertices = ply["vertex"].data
    faces = ply["face"].data
    names = set(vertices.dtype.names or ())
    require({"x", "y", "z"}.issubset(names), "static collider is missing x/y/z")
    face_names = set(faces.dtype.names or ())
    index_name = next(
        (name for name in ("vertex_indices", "vertex_index") if name in face_names),
        None,
    )
    require(index_name is not None, "static collider face indices are missing")
    require(len(vertices) > 0 and len(faces) > 0, "static collider is empty")
    positions = np.column_stack([vertices[axis] for axis in ("x", "y", "z")]).astype(np.float64)
    require(np.isfinite(positions).all(), "static collider has non-finite vertices")
    rows = [np.asarray(row, dtype=np.int64) for row in faces[index_name]]
    require(all(row.shape == (3,) for row in rows), "static collider has non-triangular faces")
    indices = np.vstack(rows)
    valid_indices = bool(indices.min() >= 0 and indices.max() < len(vertices))
    require(valid_indices, "static collider has invalid face indices")
    return {
        "path": str(path),
        "sha256": digest,
        "bytes": size,
        "vertex_count": len(vertices),
        "face_count": len(faces),
        "bounds": [positions.min(axis=0).tolist(), positions.max(axis=0).tolist()],
        "face_index_field": index_name,
        "face_indices_valid": valid_indices,
        "role": ORIGINAL_COLLIDER_ROLE,
    }


def verify_carve_evidence(
    *,
    static_path: Path,
    static_sha256: str,
    carve_reference: Any,
    parent: Path,
) -> CarveEvidence:
    require(isinstance(carve_reference, dict), "static_scene.carve_receipt is missing")
    receipt_path = resolve_path(
        carve_reference.get("path"), parent=parent, label="static_scene.carve_receipt"
    )
    receipt_sha, _receipt_size = require_hash(
        receipt_path,
        carve_reference.get("sha256"),
        label="static_scene.carve_receipt",
    )
    receipt = read_json(receipt_path, label="carve receipt")
    require(receipt.get("schema_version") == 1, "carve receipt schema version mismatch")
    require(receipt.get("kind") == CARVE_KIND, "carve receipt kind mismatch")
    require(
        receipt.get("status") == "candidate_materialized"
        and receipt.get("promotion_allowed") is False,
        "carve receipt must remain an unpromoted candidate",
    )
    source = receipt.get("input_static_scene")
    output = receipt.get("output_static_scene")
    analysis = receipt.get("analysis")
    require(
        isinstance(source, dict) and isinstance(output, dict), "carve receipt has no static refs"
    )
    require(isinstance(analysis, dict), "carve receipt has no analysis")
    source_path = resolve_path(source.get("path"), parent=receipt_path.parent, label="carve source")
    source_sha, source_size = require_hash(source_path, source.get("sha256"), label="carve source")
    output_path = resolve_path(output.get("path"), parent=receipt_path.parent, label="carve output")
    require(output_path == static_path, "carve output path does not match configured static scene")
    require(output.get("sha256") == static_sha256, "carve output hash does not match config")
    require(output.get("role") == CARVED_STATIC_ROLE, "carve output role mismatch")
    require(source.get("role") == ORIGINAL_STATIC_ROLE, "carve source role mismatch")
    require(
        source.get("sha256") == source_sha and source.get("bytes") == source_size,
        "carve source record mismatch",
    )
    require(
        int(analysis.get("selected_remove_count", 0)) > 0,
        "carve receipt removed no static Gaussians",
    )
    require(
        int(output.get("vertex_count", 0)) < int(source.get("vertex_count", 0)),
        "carve receipt output was not reduced",
    )
    return CarveEvidence(
        receipt_path=receipt_path,
        receipt_sha256=receipt_sha,
        source_static_scene={**source, "path": str(source_path)},
        output_static_scene={**output, "path": str(output_path)},
        analysis=analysis,
    )


def verify_object_evidence(
    *,
    reference: Any,
    parent: Path,
    maximum_faces: int,
) -> ObjectEvidence:
    require(isinstance(reference, dict), "object reference must be an object")
    object_id = reference.get("object_id")
    require(isinstance(object_id, str) and object_id, "object_id is missing")
    receipt_ref = reference.get("scene_fit_receipt")
    require(isinstance(receipt_ref, dict), f"scene_fit_receipt is missing: {object_id}")
    receipt_path = resolve_path(
        receipt_ref.get("path"), parent=parent, label=f"{object_id}.scene_fit_receipt"
    )
    receipt_sha, _receipt_size = require_hash(
        receipt_path,
        receipt_ref.get("sha256"),
        label=f"{object_id}.scene_fit_receipt",
    )
    receipt = read_json(receipt_path, label=f"{object_id} scene-fit receipt")
    require(receipt.get("schema_version") == 1, f"scene-fit schema mismatch: {object_id}")
    require(receipt.get("kind") == SCENE_FIT_KIND, f"scene-fit kind mismatch: {object_id}")
    require(receipt.get("object_id") == object_id, f"scene-fit object mismatch: {object_id}")
    require(
        receipt.get("status") == "technical_gates_passed"
        and receipt.get("all_acceptance_gates_passed") is True,
        f"scene-fit gates failed: {object_id}",
    )
    gates = receipt.get("acceptance_gates")
    require(
        isinstance(gates, dict) and gates and all(value is True for value in gates.values()),
        f"scene-fit acceptance gates failed: {object_id}",
    )
    scene_space = receipt.get("exports", {}).get("scene_space")
    require(isinstance(scene_space, dict), f"scene-space export is missing: {object_id}")
    require(
        scene_space.get("coordinate_space") == "world_baked", f"scene-space mismatch: {object_id}"
    )
    require(
        scene_space.get("glb_role") == "pbr_visual_logic_authority",
        f"scene-space PBR role mismatch: {object_id}",
    )
    transform = np.asarray(scene_space.get("runtime_transform"), dtype=np.float64)
    require(
        transform.shape == (4, 4) and np.allclose(transform, np.eye(4), atol=1e-9),
        f"scene-space runtime transform is not identity: {object_id}",
    )
    glb_record = scene_space.get("glb")
    require(isinstance(glb_record, dict), f"scene-space GLB record is missing: {object_id}")
    glb_path = resolve_path(
        glb_record.get("path"), parent=receipt_path.parent, label=f"{object_id}.scene_space.glb"
    )
    glb = audit_unified_glb(glb_path, maximum_faces=maximum_faces)
    require(glb.sha256 == glb_record.get("sha256"), f"scene-space GLB hash mismatch: {object_id}")
    require(
        glb.size_bytes == glb_record.get("bytes"),
        f"scene-space GLB byte mismatch: {object_id}",
    )
    require(
        glb.vertices == glb_record.get("vertices"), f"scene-space GLB vertex mismatch: {object_id}"
    )
    require(glb.faces == glb_record.get("faces"), f"scene-space GLB face mismatch: {object_id}")
    tolerance = float(receipt.get("final_export_qa", {}).get("bounds_tolerance", 0.0))
    require(
        0 < tolerance <= 1.0 and math.isfinite(tolerance),
        f"invalid GLB bounds tolerance: {object_id}",
    )
    require(
        same_bounds(glb.bounds, glb_record.get("bounds"), tolerance=tolerance),
        f"scene-space bounds mismatch: {object_id}",
    )
    require(
        receipt.get("final_export_qa", {}).get("canonical_to_world_round_trip") is True,
        f"world-bake round trip failed: {object_id}",
    )
    return ObjectEvidence(
        object_id=object_id,
        receipt_path=receipt_path,
        receipt_sha256=receipt_sha,
        category=str(reference.get("category") or "unknown"),
        label=str(reference.get("label") or object_id),
        glb_path=glb_path,
        glb_sha256=glb.sha256,
        glb_bytes=glb.size_bytes,
        vertices=glb.vertices,
        faces=glb.faces,
        bounds=glb.bounds,
        watertight=glb.watertight,
    )


def load_config(
    config_path: Path,
) -> tuple[dict[str, Any], Path, str, Any, Path, str, Sequence[Any], int]:
    config = read_json(config_path, label="candidate config")
    require(isinstance(config, dict), "candidate config root must be an object")
    require(config.get("schema_version") == 1, "candidate config schema version mismatch")
    require(config.get("kind") == CONFIG_KIND, "candidate config kind mismatch")
    require(
        isinstance(config.get("candidate_version"), str) and config["candidate_version"],
        "candidate_version must be non-empty",
    )
    static_ref = config.get("static_scene")
    collider_ref = config.get("static_collider")
    objects = config.get("objects")
    require(isinstance(static_ref, dict), "static_scene reference is missing")
    require(isinstance(collider_ref, dict), "static_collider reference is missing")
    require(isinstance(objects, list) and objects, "objects must be a non-empty list")
    require(static_ref.get("role") == CARVED_STATIC_ROLE, "static_scene role mismatch")
    require(
        collider_ref.get("role") == ORIGINAL_COLLIDER_ROLE,
        "static_collider role mismatch",
    )
    parent = config_path.parent
    static_path = resolve_path(static_ref.get("path"), parent=parent, label="static_scene")
    collider_path = resolve_path(collider_ref.get("path"), parent=parent, label="static_collider")
    maximum_faces = int(config.get("maximum_collision_faces", 100_000))
    require(0 < maximum_faces <= 100_000, "maximum_collision_faces must be in (0, 100000]")
    return (
        config,
        static_path,
        str(static_ref.get("sha256")),
        static_ref.get("carve_receipt"),
        collider_path,
        str(collider_ref.get("sha256")),
        objects,
        maximum_faces,
    )


def assert_local_outputs(project_root: Path, output: Path, report: Path) -> None:
    require(output != report, "candidate manifest and report must differ")
    public_root = (project_root / "web/public").resolve()
    require(
        output != public_root and public_root not in output.parents,
        "candidate manifest must not be written below web/public",
    )
    require(
        report != public_root and public_root not in report.parents,
        "candidate report must not be written below web/public",
    )
    require(not output.exists(), f"refusing to overwrite candidate manifest: {output}")
    require(not report.exists(), f"refusing to overwrite candidate report: {report}")


def object_manifest(item: ObjectEvidence) -> dict[str, Any]:
    topology = "closed_volume" if item.watertight else "surface_bvh"
    return {
        "id": item.object_id,
        "label": item.label,
        "category": item.category,
        "representation": "world_baked_pbr_glb_visual_logic_collision",
        "visual_logic_collision": {
            "url": local_fs_url(item.glb_path),
            "path": str(item.glb_path),
            "sha256": item.glb_sha256,
            "bytes": item.glb_bytes,
            "vertices": item.vertices,
            "faces": item.faces,
            "bounds": item.bounds,
            "topology": topology,
            "coordinate_space": "world_baked",
        },
        "placement": {
            "coordinate_space": "world_baked",
            "runtime_transform": "identity",
            "scale": [1.0, 1.0, 1.0],
            "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "translation": [0.0, 0.0, 0.0],
        },
        "scene_fit_receipt": {
            "path": str(item.receipt_path),
            "sha256": item.receipt_sha256,
        },
    }


def materialize_front_half_candidate(
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
    assert_local_outputs(project_root, output_manifest, report_path)
    config_sha, _config_size = sha256_file(config_path)
    (
        config,
        static_path,
        expected_static_sha,
        carve_reference,
        collider_path,
        expected_collider_sha,
        object_refs,
        maximum_faces,
    ) = load_config(config_path)
    static_scene = audit_pgsr_ply(static_path, expected_sha256=expected_static_sha)
    carve = verify_carve_evidence(
        static_path=static_path,
        static_sha256=static_scene.sha256,
        carve_reference=carve_reference,
        parent=config_path.parent,
    )
    require(
        static_scene.vertex_count == carve.output_static_scene.get("vertex_count"),
        "carve receipt static vertex count mismatch",
    )
    collider = audit_collider(collider_path, expected_sha256=expected_collider_sha)
    objects = [
        verify_object_evidence(
            reference=item,
            parent=config_path.parent,
            maximum_faces=maximum_faces,
        )
        for item in object_refs
    ]
    object_ids = [item.object_id for item in objects]
    require(len(object_ids) == len(set(object_ids)), "duplicate object_id in candidate config")
    manifest = {
        "schema_version": 1,
        "kind": MANIFEST_KIND,
        "candidate_version": config["candidate_version"],
        "status": "local_candidate_materialized",
        "promotion_allowed": False,
        "coordinate_space": "video2mesh_scene_world",
        "local_asset_transport": "/@fs",
        "static_scene": {
            "visual": {
                "url": local_fs_url(static_path),
                "path": str(static_path),
                "sha256": static_scene.sha256,
                "bytes": static_scene.size_bytes,
                "vertex_count": static_scene.vertex_count,
                "bounds": static_scene.bounds,
                "role": CARVED_STATIC_ROLE,
            },
            "source_before_carve": carve.source_static_scene,
            "carve_receipt": {
                "path": str(carve.receipt_path),
                "sha256": carve.receipt_sha256,
                "method": "multiview_modal_mask_depth_consistent_center_carve",
                "removed_static_gaussian_count": carve.analysis["selected_remove_count"],
                "strict_all_view_remove_count": carve.analysis["strict_all_view_remove_count"],
            },
        },
        "static_collider": {
            **collider,
            "status": "original_uncarved_mesh_reference",
            "replacement_status": "not_carved_for_this_visual_candidate",
        },
        "objects": [object_manifest(item) for item in objects],
        "composition": {
            "render_order": "carved_static_gaussians_then_world_baked_object_glbs",
            "object_ids": object_ids,
            "visual_overlap_policy": "only mask-depth-verified static centers were removed",
            "clean_plate_generated": False,
            "static_collider_updated": False,
        },
        "limitations": [
            "Local candidate only; no canonical scene file is changed.",
            (
                "The static collider remains the original uncarved mesh and is not "
                "a replacement proof."
            ),
            "The carved region is not a clean plate and does not reconstruct hidden background.",
            "Window assets remain excluded until their front/back and texture audit passes.",
        ],
    }
    atomic_write_json(output_manifest, manifest)
    manifest_sha, manifest_size = sha256_file(output_manifest)
    protected = [
        (config_path, config_sha, "candidate config"),
        (static_path, static_scene.sha256, "carved static scene"),
        (collider_path, collider["sha256"], "static collider"),
        (carve.receipt_path, carve.receipt_sha256, "carve receipt"),
        (
            Path(carve.source_static_scene["path"]),
            carve.source_static_scene["sha256"],
            "carve source",
        ),
    ]
    protected.extend(
        (item.receipt_path, item.receipt_sha256, f"{item.object_id} receipt") for item in objects
    )
    protected.extend(
        (item.glb_path, item.glb_sha256, f"{item.object_id} scene-space GLB") for item in objects
    )
    for path, digest, label in protected:
        require_hash(path, digest, label=label)
    report = {
        "schema_version": 1,
        "kind": REPORT_KIND,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "local_candidate_materialized",
        "promotion_allowed": False,
        "inputs": {
            "config": {"path": str(config_path), "sha256": config_sha},
            "carved_static_scene": manifest["static_scene"]["visual"],
            "source_static_scene": carve.source_static_scene,
            "carve_receipt": manifest["static_scene"]["carve_receipt"],
            "static_collider": collider,
            "objects": [
                {
                    "object_id": item.object_id,
                    "scene_fit_receipt": {
                        "path": str(item.receipt_path),
                        "sha256": item.receipt_sha256,
                    },
                    "scene_space_glb": {
                        "path": str(item.glb_path),
                        "sha256": item.glb_sha256,
                    },
                }
                for item in objects
            ],
        },
        "output_manifest": {
            "path": str(output_manifest),
            "sha256": manifest_sha,
            "bytes": manifest_size,
        },
        "limitations": manifest["limitations"],
    }
    atomic_write_json(report_path, report)
    return manifest, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest, report = materialize_front_half_candidate(
        project_root=args.project_root,
        config_path=args.config,
        output_manifest=args.output_manifest,
        report_path=args.report,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "objects": [item["id"] for item in manifest["objects"]],
                "manifest": report["output_manifest"]["path"],
                "report": str(args.report.expanduser().resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
