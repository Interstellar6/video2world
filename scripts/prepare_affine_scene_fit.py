#!/usr/bin/env python3
"""Validate an anchor-derived affine transform for one unified PBR GLB."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


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


def nested_value(payload: dict[str, Any], dotted_key: str) -> Any:
    current: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ValueError(f"transform receipt is missing {dotted_key!r}")
        current = current[part]
    return current


def audit_uniform_affine(matrix: np.ndarray) -> dict[str, Any]:
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("scene transform must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError("scene transform must be affine")
    linear = matrix[:3, :3]
    determinant = float(np.linalg.det(linear))
    if determinant <= 0:
        raise ValueError("scene transform must preserve orientation")
    scale_xyz = np.linalg.norm(linear, axis=0)
    if np.any(scale_xyz <= 1e-9):
        raise ValueError("scene transform has a zero scale axis")
    uniform_scale = float(np.mean(scale_xyz))
    scale_anisotropy_ratio = float(np.max(scale_xyz) / np.min(scale_xyz))
    rotation = linear / scale_xyz[None, :]
    orthonormal = bool(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5))
    uniform = bool(np.allclose(scale_xyz, uniform_scale, rtol=1e-5, atol=1e-7))
    right_handed = bool(np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5))
    return {
        "linear_row_major": linear.tolist(),
        "translation": matrix[:3, 3].tolist(),
        "scale_xyz": scale_xyz.tolist(),
        "uniform_scale": uniform_scale,
        "scale_anisotropy_ratio": scale_anisotropy_ratio,
        "rotation_matrix_row_major": rotation.tolist(),
        "determinant": determinant,
        "orthonormal_rotation": orthonormal,
        "uniform_scale_verified": uniform,
        "right_handed_rotation": right_handed,
    }


def prepare_affine_scene_fit(
    *,
    object_id: str,
    mesh_source: Path,
    transform_receipt: Path,
    output_dir: Path,
    matrix_key: str = "sources.anchor.canonical_to_anchor_row_major",
    collision_topology: str = "closed_volume",
    max_collision_faces: int = 100_000,
    scale_factors: tuple[float, float, float] = (1.0, 1.0, 1.0),
    canonical_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    allow_nonuniform_scale: bool = False,
    maximum_scale_anisotropy: float = 2.0,
) -> dict[str, Any]:
    if collision_topology not in {"closed_volume", "surface_bvh"}:
        raise ValueError("collision_topology must be closed_volume or surface_bvh")
    if max_collision_faces < 1:
        raise ValueError("max_collision_faces must be positive")
    factors = np.asarray(scale_factors, dtype=np.float64)
    offset = np.asarray(canonical_offset, dtype=np.float64)
    if factors.shape != (3,) or not np.isfinite(factors).all() or np.any(factors <= 0):
        raise ValueError("scale_factors must contain three positive finite values")
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("canonical_offset must contain three finite values")
    if not np.isfinite(maximum_scale_anisotropy) or maximum_scale_anisotropy < 1:
        raise ValueError("maximum_scale_anisotropy must be at least one")
    receipt = json.loads(transform_receipt.read_text(encoding="utf-8"))
    if receipt.get("object_id") != object_id:
        raise ValueError("transform receipt object_id does not match")
    output_asset = receipt.get("output", {}).get("unified_pbr_glb", {})
    expected_mesh_sha256 = output_asset.get("sha256")
    observed_mesh_sha256 = sha256_file(mesh_source)
    if expected_mesh_sha256 != observed_mesh_sha256:
        raise ValueError("mesh hash does not match the transform receipt")
    base_matrix = np.asarray(nested_value(receipt, matrix_key), dtype=np.float64)
    if base_matrix.shape != (4, 4) or not np.isfinite(base_matrix).all():
        raise ValueError("anchor transform must be a finite 4x4 matrix")
    refinement = np.eye(4, dtype=np.float64)
    refinement[:3, :3] = np.diag(factors)
    refinement[:3, 3] = offset
    matrix = base_matrix @ refinement
    transform_audit = audit_uniform_affine(matrix)

    loaded = trimesh.load(mesh_source, force="scene")
    scene = loaded if isinstance(loaded, trimesh.Scene) else trimesh.Scene(loaded)
    merged = scene.to_geometry()
    if len(merged.vertices) == 0 or len(merged.faces) == 0:
        raise ValueError("unified GLB is empty")
    parents = dict(scene.graph.transforms.parents)
    base_frame = scene.graph.base_frame
    root_nodes = sorted(
        node for node, parent in parents.items() if parent == base_frame and node != base_frame
    )
    component_nodes = sorted(node for node, parent in parents.items() if parent == object_id)
    materials = []
    for name, geometry in scene.geometry.items():
        material = getattr(getattr(geometry, "visual", None), "material", None)
        materials.append(
            {
                "geometry": name,
                "type": type(material).__name__ if material is not None else None,
                "metallic_factor": getattr(material, "metallicFactor", None),
                "roughness_factor": getattr(material, "roughnessFactor", None),
            }
        )

    vertices_world = np.einsum(
        "ni,ji->nj",
        np.asarray(merged.vertices, dtype=np.float64),
        matrix[:3, :3],
        optimize=False,
    ) + matrix[:3, 3]
    topology_ready = (
        bool(merged.is_watertight)
        if collision_topology == "closed_volume"
        else bool(merged.is_winding_consistent)
    )
    gates = {
        "receipt_asset_hash_match": expected_mesh_sha256 == observed_mesh_sha256,
        "finite_affine_transform": bool(np.isfinite(matrix).all()),
        "runtime_scale_supported": (
            transform_audit["uniform_scale_verified"] or allow_nonuniform_scale
        ),
        "scale_anisotropy_within_limit": (
            transform_audit["scale_anisotropy_ratio"] <= maximum_scale_anisotropy
        ),
        "orthonormal_rotation": transform_audit["orthonormal_rotation"],
        "right_handed_rotation": transform_audit["right_handed_rotation"],
        "single_logical_root": root_nodes == [object_id],
        "component_parenting": bool(component_nodes),
        "pbr_materials": bool(materials)
        and all(item["type"] == "PBRMaterial" for item in materials),
        "collision_face_budget": len(merged.faces) <= max_collision_faces,
        "collision_topology_ready": topology_ready,
        "world_vertices_finite": bool(np.isfinite(vertices_world).all()),
    }
    all_gates_passed = all(gates.values())
    report = {
        "schema_version": 1,
        "kind": "video2world.affine_object_scene_fit",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "technical_gates_passed" if all_gates_passed else "rejected",
        "promotion_status": "held_pending_scene_browser_review",
        "object_id": object_id,
        "representation_mode": "unified_mesh_visual_logic_collision",
        "sources": {
            "mesh": {
                "path": str(mesh_source),
                "sha256": observed_mesh_sha256,
                "bytes": mesh_source.stat().st_size,
            },
            "transform_receipt": {
                "path": str(transform_receipt),
                "sha256": sha256_file(transform_receipt),
                "matrix_key": matrix_key,
            },
        },
        "runtime_transform": {
            "matrix_row_major": matrix.tolist(),
            "asset_coordinates_baked": False,
            "base_matrix_row_major": base_matrix.tolist(),
            "refinement": {
                "scale_factors": factors.tolist(),
                "canonical_offset": offset.tolist(),
                "allow_nonuniform_scale": allow_nonuniform_scale,
                "maximum_scale_anisotropy": maximum_scale_anisotropy,
            },
            **transform_audit,
        },
        "logical_entity": {
            "root_node": object_id,
            "root_nodes": root_nodes,
            "component_nodes": component_nodes,
            "selection_owner": object_id,
            "motion_owner": object_id,
            "collision_owner": object_id,
        },
        "world_bounds": [vertices_world.min(axis=0).tolist(), vertices_world.max(axis=0).tolist()],
        "mesh": {
            "vertices": len(merged.vertices),
            "faces": len(merged.faces),
            "watertight": bool(merged.is_watertight),
            "winding_consistent": bool(merged.is_winding_consistent),
            "materials": materials,
        },
        "collision_semantics": {
            "asset_mode": "unified_glb",
            "topology": collision_topology,
            "max_faces": max_collision_faces,
            "fallback_proxy_allowed": False,
        },
        "acceptance_gates": gates,
        "all_acceptance_gates_passed": all_gates_passed,
        "promotion_blockers": [
            "source_camera_silhouette_review",
            "scene_interpenetration_gate",
            "support_contact_review",
            "accepted_clean_plate_or_static_scene_carve",
        ],
    }
    write_json(output_dir / "scene_fit_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--mesh-source", type=Path, required=True)
    parser.add_argument("--transform-receipt", type=Path, required=True)
    parser.add_argument(
        "--matrix-key",
        default="sources.anchor.canonical_to_anchor_row_major",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--collision-topology",
        choices=("closed_volume", "surface_bvh"),
        default="closed_volume",
    )
    parser.add_argument("--max-collision-faces", type=int, default=100_000)
    parser.add_argument("--scale-factors", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    parser.add_argument("--canonical-offset", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--allow-nonuniform-scale", action="store_true")
    parser.add_argument("--maximum-scale-anisotropy", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = prepare_affine_scene_fit(
        object_id=args.object_id,
        mesh_source=args.mesh_source.expanduser().resolve(),
        transform_receipt=args.transform_receipt.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        matrix_key=args.matrix_key,
        collision_topology=args.collision_topology,
        max_collision_faces=args.max_collision_faces,
        scale_factors=tuple(args.scale_factors),
        canonical_offset=tuple(args.canonical_offset),
        allow_nonuniform_scale=args.allow_nonuniform_scale,
        maximum_scale_anisotropy=args.maximum_scale_anisotropy,
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True))
    return 0 if report["all_acceptance_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
