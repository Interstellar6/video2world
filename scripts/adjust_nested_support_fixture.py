#!/usr/bin/env python3
"""Adjust a parent fixture support surface against fitted child assets."""

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


def resolve_path(value: str, *, relative_to: Path) -> Path:
    path = Path(value).expanduser()
    return (relative_to / path).resolve() if not path.is_absolute() else path.resolve()


def load_child_support(
    item: dict[str, Any],
    *,
    manifest_path: Path,
    parent_inverse: np.ndarray,
    support_axis_scale: float,
    lower_quantile: float,
    upper_quantile: float,
    maximum_penetration_world: float,
    maximum_penetration_fraction: float,
) -> dict[str, Any]:
    object_id = item.get("object_id")
    if not isinstance(object_id, str) or not object_id:
        raise ValueError("child support entry requires object_id")
    mesh_path = resolve_path(str(item.get("mesh")), relative_to=manifest_path.parent)
    report_path = resolve_path(
        str(item.get("scene_fit_report")), relative_to=manifest_path.parent
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("object_id") != object_id:
        raise ValueError(f"scene-fit object mismatch for {object_id}")
    expected_mesh_sha = report.get("mesh", {}).get("glb_sha256")
    observed_mesh_sha = sha256_file(mesh_path)
    if expected_mesh_sha != observed_mesh_sha:
        raise ValueError(f"scene-fit mesh hash mismatch for {object_id}")
    pivot = np.asarray(
        report.get("baked_relative_transform", {}).get("runtime_pivot"),
        dtype=np.float64,
    )
    if pivot.shape != (3,) or not np.isfinite(pivot).all():
        raise ValueError(f"scene-fit runtime pivot is invalid for {object_id}")
    loaded = trimesh.load(mesh_path, force="scene")
    mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded
    world = np.asarray(mesh.vertices, dtype=np.float64) + pivot
    parent = np.einsum(
        "ni,ji->nj",
        world,
        parent_inverse[:3, :3],
        optimize=False,
    ) + parent_inverse[:3, 3]
    support_values = parent[:, 1]
    lower, upper = np.quantile(support_values, [lower_quantile, upper_quantile])
    thickness_parent = float(upper - lower)
    thickness_world = thickness_parent * support_axis_scale
    allowed_world = min(
        maximum_penetration_world,
        maximum_penetration_fraction * thickness_world,
    )
    return {
        "object_id": object_id,
        "mesh": {"path": str(mesh_path), "sha256": observed_mesh_sha},
        "scene_fit_report": {
            "path": str(report_path),
            "sha256": sha256_file(report_path),
        },
        "support_lower_parent": float(lower),
        "support_upper_parent": float(upper),
        "support_thickness_parent": thickness_parent,
        "support_thickness_world": thickness_world,
        "allowed_penetration_world": allowed_world,
        "allowed_penetration_parent": allowed_world / support_axis_scale,
    }


def find_geometry(scene: trimesh.Scene, suffix: str) -> tuple[str, trimesh.Trimesh]:
    matches = [
        (name, geometry)
        for name, geometry in scene.geometry.items()
        if name.endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one geometry ending in {suffix!r}, found {len(matches)}")
    return matches[0]


def audit_scene(scene: trimesh.Scene, object_id: str) -> dict[str, Any]:
    parents = dict(scene.graph.transforms.parents)
    base_frame = scene.graph.base_frame
    root_nodes = sorted(
        node for node, parent in parents.items() if parent == base_frame and node != base_frame
    )
    component_nodes = sorted(node for node, parent in parents.items() if parent == object_id)
    component_reports = []
    for name, geometry in scene.geometry.items():
        material = getattr(getattr(geometry, "visual", None), "material", None)
        component_reports.append(
            {
                "name": name,
                "vertices": len(geometry.vertices),
                "faces": len(geometry.faces),
                "watertight": bool(geometry.is_watertight),
                "winding_consistent": bool(geometry.is_winding_consistent),
                "finite": bool(np.isfinite(geometry.vertices).all()),
                "material_type": type(material).__name__ if material is not None else None,
                "bounds": np.asarray(geometry.bounds).tolist(),
            }
        )
    return {
        "root_nodes": root_nodes,
        "component_nodes": component_nodes,
        "components": component_reports,
        "single_logical_root": root_nodes == [object_id],
        "component_parenting": bool(component_nodes),
        "components_closed": all(item["watertight"] for item in component_reports),
        "components_winding_consistent": all(
            item["winding_consistent"] for item in component_reports
        ),
        "components_finite": all(item["finite"] for item in component_reports),
        "components_pbr": all(
            item["material_type"] == "PBRMaterial" for item in component_reports
        ),
    }


def adjust_nested_support_fixture(
    *,
    object_id: str,
    parent_mesh: Path,
    parent_fixture_receipt: Path,
    parent_scene_fit_report: Path,
    child_manifest: Path,
    output_dir: Path,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
    maximum_penetration_world: float = 0.2,
    maximum_penetration_fraction: float = 0.1,
    maximum_support_gap_world: float = 0.2,
    minimum_mattress_height_fraction: float = 0.5,
) -> dict[str, Any]:
    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError("support quantiles are invalid")
    if maximum_penetration_world < 0 or maximum_penetration_fraction < 0:
        raise ValueError("penetration limits must be non-negative")
    if maximum_support_gap_world < 0:
        raise ValueError("maximum_support_gap_world must be non-negative")
    if not 0 < minimum_mattress_height_fraction <= 1:
        raise ValueError("minimum_mattress_height_fraction must be in (0, 1]")

    fixture = json.loads(parent_fixture_receipt.read_text(encoding="utf-8"))
    scene_fit = json.loads(parent_scene_fit_report.read_text(encoding="utf-8"))
    manifest = json.loads(child_manifest.read_text(encoding="utf-8"))
    if any(payload.get("object_id") != object_id for payload in (fixture, scene_fit, manifest)):
        raise ValueError("parent object_id mismatch")
    expected_parent_sha = fixture.get("output", {}).get("unified_pbr_glb", {}).get("sha256")
    observed_parent_sha = sha256_file(parent_mesh)
    if expected_parent_sha != observed_parent_sha:
        raise ValueError("parent mesh hash does not match fixture receipt")
    matrix = np.asarray(
        scene_fit.get("runtime_transform", {}).get("matrix_row_major"), dtype=np.float64
    )
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("parent scene-fit report has no finite 4x4 transform")
    parent_inverse = np.linalg.inv(matrix)
    support_axis_scale = float(np.linalg.norm(matrix[:3, 1]))
    if support_axis_scale <= 1e-9:
        raise ValueError("parent support axis scale is zero")

    child_values = manifest.get("children")
    if not isinstance(child_values, list) or not child_values:
        raise ValueError("child manifest requires at least one child")
    children = [
        load_child_support(
            item,
            manifest_path=child_manifest,
            parent_inverse=parent_inverse,
            support_axis_scale=support_axis_scale,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
            maximum_penetration_world=maximum_penetration_world,
            maximum_penetration_fraction=maximum_penetration_fraction,
        )
        for item in child_values
    ]

    loaded = trimesh.load(parent_mesh, force="scene")
    scene = loaded if isinstance(loaded, trimesh.Scene) else trimesh.Scene(loaded)
    mattress_name, mattress = find_geometry(scene, ".mattress")
    cover_name, cover = find_geometry(scene, ".foot_cover")
    layout = fixture.get("component_layout", {})
    mattress_bottom = float(layout.get("mattress_bottom_y"))
    original_mattress_top = float(layout.get("mattress_top_y"))
    original_mattress_height = original_mattress_top - mattress_bottom
    if original_mattress_height <= 0:
        raise ValueError("fixture receipt has invalid mattress bounds")
    target_ceiling = min(
        child["support_lower_parent"] + child["allowed_penetration_parent"]
        for child in children
    )
    minimum_top = mattress_bottom + original_mattress_height * minimum_mattress_height_fraction
    adjusted_mattress_top = min(original_mattress_top, max(minimum_top, target_ceiling))
    if adjusted_mattress_top > target_ceiling + 1e-9:
        raise ValueError("mattress minimum-height policy cannot satisfy child penetration gates")
    adjusted_height = adjusted_mattress_top - mattress_bottom
    height_scale = adjusted_height / original_mattress_height
    mattress.vertices[:, 1] = mattress_bottom + (
        mattress.vertices[:, 1] - mattress_bottom
    ) * height_scale
    surface_delta = adjusted_mattress_top - original_mattress_top
    cover.apply_translation([0.0, surface_delta, 0.0])

    destination = output_dir / f"{object_id}.support-adjusted.glb"
    destination.parent.mkdir(parents=True, exist_ok=True)
    scene.export(destination)
    reloaded = trimesh.load(destination, force="scene")
    output_scene = reloaded if isinstance(reloaded, trimesh.Scene) else trimesh.Scene(reloaded)
    audit = audit_scene(output_scene, object_id)
    adjusted_children = []
    for child in children:
        penetration_world = max(
            0.0,
            (adjusted_mattress_top - child["support_lower_parent"])
            * support_axis_scale,
        )
        support_gap_world = max(
            0.0,
            (child["support_lower_parent"] - adjusted_mattress_top)
            * support_axis_scale,
        )
        adjusted_children.append(
            {
                **child,
                "predicted_penetration_before_world": max(
                    0.0,
                    (original_mattress_top - child["support_lower_parent"])
                    * support_axis_scale,
                ),
                "predicted_penetration_after_world": penetration_world,
                "predicted_support_gap_after_world": support_gap_world,
                "penetration_gate": penetration_world
                <= child["allowed_penetration_world"] + 1e-6,
                "support_gap_gate": support_gap_world <= maximum_support_gap_world + 1e-6,
            }
        )
    gates = {
        "single_logical_root": audit["single_logical_root"],
        "component_parenting": audit["component_parenting"],
        "components_closed": audit["components_closed"],
        "components_winding_consistent": audit["components_winding_consistent"],
        "components_finite": audit["components_finite"],
        "components_pbr": audit["components_pbr"],
        "mattress_height_positive": adjusted_height > 0,
        "child_penetration_limits": all(
            child["penetration_gate"] for child in adjusted_children
        ),
        "child_support_gap_limits": all(
            child["support_gap_gate"] for child in adjusted_children
        ),
    }
    all_gates_passed = all(gates.values())
    report = {
        "schema_version": 1,
        "kind": "video2world.nested_support_fixture_adjustment",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "technical_gates_passed" if all_gates_passed else "rejected",
        "promotion_allowed": False,
        "object_id": object_id,
        "sources": {
            "parent_mesh": {"path": str(parent_mesh), "sha256": observed_parent_sha},
            "parent_fixture_receipt": {
                "path": str(parent_fixture_receipt),
                "sha256": sha256_file(parent_fixture_receipt),
            },
            "parent_scene_fit_report": {
                "path": str(parent_scene_fit_report),
                "sha256": sha256_file(parent_scene_fit_report),
            },
            "child_manifest": {
                "path": str(child_manifest),
                "sha256": sha256_file(child_manifest),
            },
        },
        "scene_fit_transform": {"matrix_row_major": matrix.tolist()},
        "policy": {
            "lower_quantile": lower_quantile,
            "upper_quantile": upper_quantile,
            "maximum_penetration_world": maximum_penetration_world,
            "maximum_penetration_fraction": maximum_penetration_fraction,
            "maximum_support_gap_world": maximum_support_gap_world,
            "minimum_mattress_height_fraction": minimum_mattress_height_fraction,
        },
        "adjustment": {
            "mattress_geometry": mattress_name,
            "foot_cover_geometry": cover_name,
            "mattress_bottom_parent": mattress_bottom,
            "mattress_top_before_parent": original_mattress_top,
            "mattress_top_after_parent": adjusted_mattress_top,
            "mattress_height_scale": height_scale,
            "foot_cover_translation_parent": [0.0, surface_delta, 0.0],
            "support_axis_scale_world_per_parent": support_axis_scale,
        },
        "children": adjusted_children,
        "output": {
            "unified_pbr_glb": {
                "path": str(destination),
                "sha256": sha256_file(destination),
                "size_bytes": destination.stat().st_size,
                "logical_root": object_id,
            }
        },
        "component_audit": audit,
        "acceptance_gates": gates,
        "all_acceptance_gates_passed": all_gates_passed,
        "promotion_blockers": [
            "source_camera_silhouette_recheck",
            "joint_parent_child_scene_review",
            "accepted_clean_plate_or_static_scene_carve",
        ],
    }
    write_json(output_dir / "support_adjustment_receipt.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--parent-mesh", type=Path, required=True)
    parser.add_argument("--parent-fixture-receipt", type=Path, required=True)
    parser.add_argument("--parent-scene-fit-report", type=Path, required=True)
    parser.add_argument("--child-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lower-quantile", type=float, default=0.05)
    parser.add_argument("--upper-quantile", type=float, default=0.95)
    parser.add_argument("--maximum-penetration-world", type=float, default=0.2)
    parser.add_argument("--maximum-penetration-fraction", type=float, default=0.1)
    parser.add_argument("--maximum-support-gap-world", type=float, default=0.2)
    parser.add_argument("--minimum-mattress-height-fraction", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = adjust_nested_support_fixture(
        object_id=args.object_id,
        parent_mesh=args.parent_mesh.expanduser().resolve(),
        parent_fixture_receipt=args.parent_fixture_receipt.expanduser().resolve(),
        parent_scene_fit_report=args.parent_scene_fit_report.expanduser().resolve(),
        child_manifest=args.child_manifest.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        lower_quantile=args.lower_quantile,
        upper_quantile=args.upper_quantile,
        maximum_penetration_world=args.maximum_penetration_world,
        maximum_penetration_fraction=args.maximum_penetration_fraction,
        maximum_support_gap_world=args.maximum_support_gap_world,
        minimum_mattress_height_fraction=args.minimum_mattress_height_fraction,
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True))
    return 0 if report["all_acceptance_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
