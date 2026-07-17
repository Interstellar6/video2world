#!/usr/bin/env python3
"""Bake a verified OBB fit into a unified mesh or Gaussian-plus-mesh object."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from plyfile import PlyData
from trimesh.exchange.obj import export_obj


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
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


def determinant_3x3(matrix: np.ndarray) -> float:
    """Compute a small affine determinant without LAPACK underflow warnings."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (3, 3):
        raise ValueError("determinant_3x3 requires a 3x3 matrix")
    (a, b, c), (d, e, f), (g, h, i) = values.tolist()
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def load_target_obb(report_path: Path, object_id: str) -> dict[str, Any]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    instances = payload.get("instances")
    if not isinstance(instances, list):
        raise ValueError("instance report has no instances array")
    matches = [item for item in instances if item.get("object_id") == object_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {object_id!r} instance, found {len(matches)}")
    geometry = matches[0].get("geometry")
    obb = geometry.get("obb") if isinstance(geometry, dict) else None
    if not isinstance(obb, dict):
        raise ValueError(f"{object_id!r} has no verified OBB")
    center = np.asarray(obb.get("center"), dtype=np.float64)
    extents = np.asarray(obb.get("extents"), dtype=np.float64)
    axes = np.asarray(obb.get("axes_columns"), dtype=np.float64)
    if center.shape != (3,) or extents.shape != (3,) or axes.shape != (3, 3):
        raise ValueError("target OBB center/extents/axes have invalid dimensions")
    finite = np.isfinite(center).all() and np.isfinite(extents).all() and np.isfinite(axes).all()
    if not finite:
        raise ValueError("target OBB contains non-finite values")
    if np.any(extents <= 0):
        raise ValueError("target OBB extents must be positive")
    if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-5):
        raise ValueError("target OBB axes are not orthonormal")
    if not math.isclose(determinant_3x3(axes), 1.0, abs_tol=1e-5):
        raise ValueError("target OBB axes must form a right-handed rotation")
    return {
        "center": center,
        "extents": extents,
        "axes": axes,
        "raw": obb,
        "instance": matches[0],
    }


def quaternion_to_matrices(quaternions: np.ndarray) -> np.ndarray:
    values = quaternions.astype(np.float64)
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    w, x, y, z = values.T
    matrices = np.empty((len(values), 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = 1 - 2 * (y * y + z * z)
    matrices[:, 0, 1] = 2 * (x * y - z * w)
    matrices[:, 0, 2] = 2 * (x * z + y * w)
    matrices[:, 1, 0] = 2 * (x * y + z * w)
    matrices[:, 1, 1] = 1 - 2 * (x * x + z * z)
    matrices[:, 1, 2] = 2 * (y * z - x * w)
    matrices[:, 2, 0] = 2 * (x * z - y * w)
    matrices[:, 2, 1] = 2 * (y * z + x * w)
    matrices[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return matrices


def matrices_to_quaternion(matrices: np.ndarray) -> np.ndarray:
    m = matrices
    quaternions = np.column_stack(
        (
            np.sqrt(np.maximum(0.0, 1 + m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2])) / 2,
            np.copysign(
                np.sqrt(np.maximum(0.0, 1 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2])) / 2,
                m[:, 2, 1] - m[:, 1, 2],
            ),
            np.copysign(
                np.sqrt(np.maximum(0.0, 1 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2])) / 2,
                m[:, 0, 2] - m[:, 2, 0],
            ),
            np.copysign(
                np.sqrt(np.maximum(0.0, 1 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2])) / 2,
                m[:, 1, 0] - m[:, 0, 1],
            ),
        )
    )
    quaternions /= np.maximum(np.linalg.norm(quaternions, axis=1, keepdims=True), 1e-12)
    return quaternions.astype(np.float32)


def transform_gaussian_data(data: np.ndarray, linear: np.ndarray) -> np.ndarray:
    required = {
        "x",
        "y",
        "z",
        "nx",
        "ny",
        "nz",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    }
    names = set(data.dtype.names or ())
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"Gaussian PLY is missing placement fields: {missing}")
    transformed = data.copy()
    points = np.column_stack((data["x"], data["y"], data["z"])).astype(np.float64)
    points = np.einsum("ni,ji->nj", points, linear)
    for axis, name in enumerate(("x", "y", "z")):
        transformed[name] = points[:, axis]

    normals = np.column_stack((data["nx"], data["ny"], data["nz"])).astype(np.float64)
    normals = np.einsum("ni,ij->nj", normals, np.linalg.inv(linear))
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    for axis, name in enumerate(("nx", "ny", "nz")):
        transformed[name] = normals[:, axis]

    quaternions = np.column_stack((data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]))
    rotations = quaternion_to_matrices(quaternions)
    scales = np.exp(np.column_stack((data["scale_0"], data["scale_1"], data["scale_2"])))
    covariance = np.einsum(
        "nij,nj,nkj->nik",
        rotations,
        scales**2,
        rotations,
    )
    transformed_covariance = np.einsum(
        "ai,nij,bj->nab",
        linear,
        covariance,
        linear,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(transformed_covariance)
    eigenvalues = np.maximum(eigenvalues, 1e-18)
    negative = np.linalg.det(eigenvectors) < 0
    eigenvectors[negative, :, 0] *= -1
    transformed_scales = np.sqrt(eigenvalues)
    transformed_quaternions = matrices_to_quaternion(eigenvectors)
    for axis, name in enumerate(("scale_0", "scale_1", "scale_2")):
        transformed[name] = np.log(transformed_scales[:, axis])
    for axis, name in enumerate(("rot_0", "rot_1", "rot_2", "rot_3")):
        transformed[name] = transformed_quaternions[:, axis]
    numeric_fields = [
        "x",
        "y",
        "z",
        "nx",
        "ny",
        "nz",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ]
    if not np.isfinite(np.column_stack([transformed[name] for name in numeric_fields])).all():
        raise ValueError("Gaussian placement produced non-finite values")
    return transformed


def bake_gaussian(source: Path, destination: Path, linear: np.ndarray) -> dict[str, Any]:
    ply = PlyData.read(source)
    if "vertex" not in ply:
        raise ValueError("Gaussian PLY has no vertex element")
    source_data = ply["vertex"].data
    transformed = transform_gaussian_data(source_data, linear)
    ply["vertex"].data = transformed
    destination.parent.mkdir(parents=True, exist_ok=True)
    ply.write(destination)
    points = np.column_stack((transformed["x"], transformed["y"], transformed["z"]))
    return {
        "vertex_count": len(points),
        "bounds": [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
        "sha256": sha256_file(destination),
        "bytes": destination.stat().st_size,
    }


def bake_mesh(
    source: Path,
    glb_destination: Path,
    obj_destination: Path,
    linear: np.ndarray,
    *,
    material_profile: str = "preserve",
) -> dict[str, Any]:
    loaded = trimesh.load(source, force="scene")
    scene = loaded if isinstance(loaded, trimesh.Scene) else trimesh.Scene(loaded)
    if material_profile not in {"preserve", "nonmetal_fabric"}:
        raise ValueError(f"unsupported material_profile: {material_profile!r}")
    materials: list[dict[str, Any]] = []
    seen_materials: set[int] = set()
    for geometry in scene.geometry.values():
        material = getattr(getattr(geometry, "visual", None), "material", None)
        if material is None or id(material) in seen_materials:
            continue
        seen_materials.add(id(material))
        metallic_before = getattr(material, "metallicFactor", None)
        roughness_before = getattr(material, "roughnessFactor", None)
        if material_profile == "nonmetal_fabric":
            if metallic_before is None or roughness_before is None:
                raise ValueError("nonmetal_fabric requires a PBR material")
            material.metallicFactor = 0.0
            material.roughnessFactor = 1.0
        materials.append(
            {
                "type": type(material).__name__,
                "metallic_factor_before": (
                    float(metallic_before) if metallic_before is not None else None
                ),
                "roughness_factor_before": (
                    float(roughness_before) if roughness_before is not None else None
                ),
                "metallic_factor_after": (
                    float(material.metallicFactor) if metallic_before is not None else None
                ),
                "roughness_factor_after": (
                    float(material.roughnessFactor) if roughness_before is not None else None
                ),
            }
        )
    if material_profile == "nonmetal_fabric" and not materials:
        raise ValueError("nonmetal_fabric requires at least one PBR material")
    transform = np.eye(4)
    transform[:3, :3] = linear
    for geometry in scene.geometry.values():
        geometry.apply_transform(transform)
    glb_destination.parent.mkdir(parents=True, exist_ok=True)
    scene.export(glb_destination)
    merged = scene.to_geometry()
    obj_text = export_obj(merged, include_color=True)
    obj_destination.write_text(obj_text, encoding="utf-8")
    vertices = np.asarray(merged.vertices)
    faces = np.asarray(merged.faces)
    face_areas = np.asarray(merged.area_faces)
    return {
        "vertices": len(vertices),
        "faces": len(faces),
        "bounds": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
        "extents": np.asarray(merged.extents).tolist(),
        "finite_vertices": bool(np.isfinite(vertices).all()),
        "valid_face_indices": bool(
            np.issubdtype(faces.dtype, np.integer)
            and np.all(faces >= 0)
            and np.all(faces < len(vertices))
        ),
        "degenerate_face_count": int(np.count_nonzero(face_areas <= 1e-12)),
        "watertight": bool(merged.is_watertight),
        "winding_consistent": bool(merged.is_winding_consistent),
        "glb_sha256": sha256_file(glb_destination),
        "glb_bytes": glb_destination.stat().st_size,
        "obj_sha256": sha256_file(obj_destination),
        "obj_bytes": obj_destination.stat().st_size,
        "material_profile": material_profile,
        "materials": materials,
    }


def projected_extents(points: np.ndarray, axes: np.ndarray) -> np.ndarray:
    local = np.einsum("ni,ij->nj", points, axes)
    return np.ptp(local, axis=0)


def bake_scene_fit(
    *,
    object_id: str,
    gaussian_source: Path | None,
    mesh_source: Path,
    instance_report: Path,
    output_dir: Path,
    placement_mode: str = "volume_center",
    view_direction: tuple[float, float, float] | None = None,
    thickness_axis: int = 2,
    max_thickness_ratio: float | None = None,
    target_extent_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    collision_topology: str = "closed_volume",
    max_collision_faces: int = 100_000,
    material_profile: str = "preserve",
) -> dict[str, Any]:
    target = load_target_obb(instance_report, object_id)
    if placement_mode not in {"volume_center", "observed_front_surface"}:
        raise ValueError(f"unsupported placement_mode: {placement_mode!r}")
    if thickness_axis not in {0, 1, 2}:
        raise ValueError("thickness_axis must be 0, 1, or 2")
    if max_thickness_ratio is not None and not 0 < max_thickness_ratio <= 1:
        raise ValueError("max_thickness_ratio must be in (0, 1]")
    if collision_topology not in {"closed_volume", "surface_bvh"}:
        raise ValueError("collision_topology must be closed_volume or surface_bvh")
    if type(max_collision_faces) is not int or max_collision_faces < 1:
        raise ValueError("max_collision_faces must be a positive integer")
    extent_scale = np.asarray(target_extent_scale, dtype=np.float64)
    if (
        extent_scale.shape != (3,)
        or not np.isfinite(extent_scale).all()
        or np.any(extent_scale <= 0)
    ):
        raise ValueError("target_extent_scale must contain three positive finite values")
    target_extents = np.asarray(target["extents"], dtype=np.float64) * extent_scale
    non_thickness_axes = [axis for axis in range(3) if axis != thickness_axis]
    thickness_reference = float(np.max(target_extents[non_thickness_axes]))
    observed_thickness = float(target_extents[thickness_axis])
    if max_thickness_ratio is not None:
        target_extents[thickness_axis] = min(
            observed_thickness,
            max_thickness_ratio * thickness_reference,
        )

    completed_center = np.asarray(target["center"], dtype=np.float64).copy()
    normalized_view: np.ndarray | None = None
    completion_direction_sign: float | None = None
    if placement_mode == "observed_front_surface":
        if view_direction is None:
            raise ValueError("observed_front_surface placement requires view_direction")
        normalized_view = np.asarray(view_direction, dtype=np.float64)
        if normalized_view.shape != (3,) or not np.isfinite(normalized_view).all():
            raise ValueError("view_direction must contain three finite values")
        view_norm = float(np.linalg.norm(normalized_view))
        if view_norm <= 1e-9:
            raise ValueError("view_direction must be non-zero")
        normalized_view /= view_norm
        thickness_direction = target["axes"][:, thickness_axis]
        completion_direction_sign = (
            1.0 if float(np.dot(thickness_direction, normalized_view)) >= 0 else -1.0
        )
        completed_center += (
            completion_direction_sign * thickness_direction * target_extents[thickness_axis] / 2.0
        )

    mesh_loaded = trimesh.load(mesh_source, force="scene")
    mesh_scene = (
        mesh_loaded if isinstance(mesh_loaded, trimesh.Scene) else trimesh.Scene(mesh_loaded)
    )
    source_mesh = mesh_scene.to_geometry()
    source_extents = np.asarray(source_mesh.extents, dtype=np.float64)
    if source_extents.shape != (3,) or np.any(source_extents <= 0):
        raise ValueError("source mesh has invalid extents")
    scales = target_extents / source_extents
    linear = target["axes"] @ np.diag(scales)
    gaussian_destination = output_dir / f"{object_id}_gaussian_scene_fit.ply"
    glb_destination = output_dir / f"{object_id}.scene-fit.glb"
    obj_destination = output_dir / f"{object_id}.scene-fit.obj"
    gaussian = (
        bake_gaussian(gaussian_source, gaussian_destination, linear)
        if gaussian_source is not None
        else None
    )
    mesh = bake_mesh(
        mesh_source,
        glb_destination,
        obj_destination,
        linear,
        material_profile=material_profile,
    )
    fitted_mesh = trimesh.load(glb_destination, force="scene").to_geometry()
    fitted_extents = projected_extents(np.asarray(fitted_mesh.vertices), target["axes"])
    extent_relative_error = np.abs(fitted_extents - target_extents) / target_extents
    bounds = np.asarray(mesh["bounds"], dtype=np.float64)
    aabb_center = bounds.mean(axis=0)
    aabb_extents = bounds[1] - bounds[0]
    front_anchor_error = 0.0
    if placement_mode == "observed_front_surface":
        assert completion_direction_sign is not None
        completed_front_center = completed_center - (
            completion_direction_sign
            * target["axes"][:, thickness_axis]
            * target_extents[thickness_axis]
            / 2.0
        )
        front_anchor_error = float(np.linalg.norm(completed_front_center - target["center"]))
    topology_ready = (
        mesh["watertight"] is True
        if collision_topology == "closed_volume"
        else mesh["winding_consistent"] is True
    )
    gates: dict[str, bool] = {
        "target_obb_extent_fit": bool(np.max(extent_relative_error) <= 0.01),
        "mesh_nonempty": mesh["vertices"] > 0 and mesh["faces"] > 0,
        "mesh_finite": mesh["finite_vertices"] is True,
        "mesh_face_indices_valid": mesh["valid_face_indices"] is True,
        "mesh_no_degenerate_faces": mesh["degenerate_face_count"] == 0,
        "mesh_winding_consistent": mesh["winding_consistent"] is True,
        "collision_face_budget": mesh["faces"] <= max_collision_faces,
        "collision_topology_ready": topology_ready,
        "finite_linear_transform": bool(np.isfinite(linear).all()),
        "positive_transform_determinant": determinant_3x3(linear) > 0,
        "front_surface_anchor_alignment": front_anchor_error <= 1e-6,
        "material_profile_applied": (
            material_profile == "preserve"
            or all(
                item["metallic_factor_after"] == 0.0 and item["roughness_factor_after"] == 1.0
                for item in mesh["materials"]
            )
        ),
    }
    if gaussian is not None:
        gates.update(
            {
                "gaussian_nonempty": gaussian["vertex_count"] > 0,
                "gaussian_finite": bool(
                    np.isfinite(np.asarray(gaussian["bounds"], dtype=np.float64)).all()
                ),
            }
        )
    report = {
        "schema_version": 1,
        "kind": "video2world.object_scene_fit",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "technical_gates_passed" if all(gates.values()) else "rejected",
        "promotion_status": "held_pending_scene_browser_review",
        "object_id": object_id,
        "representation_mode": (
            "gaussian_visual_plus_mesh_collision"
            if gaussian is not None
            else "unified_mesh_visual_logic_collision"
        ),
        "material_profile": material_profile,
        "sources": {
            "gaussian": (
                {
                    "path": str(gaussian_source),
                    "sha256": sha256_file(gaussian_source),
                }
                if gaussian_source is not None
                else None
            ),
            "mesh": {"path": str(mesh_source), "sha256": sha256_file(mesh_source)},
            "instance_report": {
                "path": str(instance_report),
                "sha256": sha256_file(instance_report),
            },
        },
        "target_obb": {
            "center": target["center"].tolist(),
            "extents": target["extents"].tolist(),
            "axes_columns": target["axes"].tolist(),
        },
        "placement_evidence": {
            "mode": placement_mode,
            "view_direction": normalized_view.tolist() if normalized_view is not None else None,
            "thickness_axis": thickness_axis,
            "observed_thickness": observed_thickness,
            "target_extent_scale": extent_scale.tolist(),
            "max_thickness_ratio": max_thickness_ratio,
            "thickness_reference": thickness_reference,
            "completion_direction_sign": completion_direction_sign,
            "front_surface_anchor_error": front_anchor_error,
        },
        "completed_obb": {
            "center": completed_center.tolist(),
            "extents": target_extents.tolist(),
            "axes_columns": target["axes"].tolist(),
        },
        "baked_relative_transform": {
            "linear_row_major": linear.tolist(),
            "scale_before_rotation": scales.tolist(),
            "runtime_pivot": completed_center.tolist(),
            "runtime_scale": [1.0, 1.0, 1.0],
            "runtime_rotation_euler_deg": [0.0, 0.0, 0.0],
        },
        "fit": {
            "source_extents": source_extents.tolist(),
            "projected_fitted_extents": fitted_extents.tolist(),
            "relative_extent_error": extent_relative_error.tolist(),
            "aabb_center_relative_to_pivot": aabb_center.tolist(),
            "aabb_extents": aabb_extents.tolist(),
        },
        "collision_semantics": {
            "asset_mode": ("separate_mesh_collision" if gaussian is not None else "unified_glb"),
            "topology": collision_topology,
            "max_faces": max_collision_faces,
            "surface_blocking": True,
            "closed_volume_claim": (
                collision_topology == "closed_volume" and mesh["watertight"] is True
            ),
            "inside_outside_queries_allowed": (
                collision_topology == "closed_volume" and mesh["watertight"] is True
            ),
            "fallback_proxy_allowed": False,
        },
        "gaussian": gaussian,
        "mesh": mesh,
        "acceptance_gates": gates,
        "all_acceptance_gates_passed": all(gates.values()),
        "promotion_blockers": [
            "scene_visual_overlay_review",
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
    parser.add_argument("--gaussian-source", type=Path)
    parser.add_argument("--mesh-source", type=Path, required=True)
    parser.add_argument("--instance-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--placement-mode",
        choices=("volume_center", "observed_front_surface"),
        default="volume_center",
    )
    parser.add_argument("--view-direction", type=float, nargs=3)
    parser.add_argument("--thickness-axis", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--max-thickness-ratio", type=float)
    parser.add_argument("--target-extent-scale", type=float, nargs=3, default=(1, 1, 1))
    parser.add_argument(
        "--collision-topology",
        choices=("closed_volume", "surface_bvh"),
        default="closed_volume",
    )
    parser.add_argument("--max-collision-faces", type=int, default=100_000)
    parser.add_argument(
        "--material-profile",
        choices=("preserve", "nonmetal_fabric"),
        default="preserve",
        help="Apply an audited PBR material profile while preserving source textures.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = bake_scene_fit(
        object_id=args.object_id,
        gaussian_source=(
            args.gaussian_source.expanduser().resolve() if args.gaussian_source else None
        ),
        mesh_source=args.mesh_source.expanduser().resolve(),
        instance_report=args.instance_report.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        placement_mode=args.placement_mode,
        view_direction=tuple(args.view_direction) if args.view_direction else None,
        thickness_axis=args.thickness_axis,
        max_thickness_ratio=args.max_thickness_ratio,
        target_extent_scale=tuple(args.target_extent_scale),
        collision_topology=args.collision_topology,
        max_collision_faces=args.max_collision_faces,
        material_profile=args.material_profile,
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if report["all_acceptance_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
