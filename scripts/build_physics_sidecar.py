#!/usr/bin/env python3
"""Export a visual OBJ and preserve Qwen physical estimates as an explicit prior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--description", type=Path, required=True)
    parser.add_argument("--visual-mesh", type=Path, required=True)
    parser.add_argument("--collision-mesh", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import trimesh

    description = json.loads(args.description.read_text(encoding="utf-8"))
    structured = description.get("structured")
    if not isinstance(structured, dict):
        raise ValueError("Qwen description has no parsed structured response")
    estimate = structured.get("physical_estimate")
    if not isinstance(estimate, dict):
        raise ValueError("Qwen response has no physical_estimate")

    scene = trimesh.load(args.visual_mesh, force="scene")
    meshes = list(scene.geometry.values())
    if not meshes:
        raise ValueError(f"visual mesh has no geometry: {args.visual_mesh}")
    visual = trimesh.util.concatenate(meshes)
    collision = trimesh.load(args.collision_mesh, force="scene")
    collision_meshes = list(collision.geometry.values())
    declared_hulls = sum(
        1
        for line in args.collision_mesh.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.startswith("o ")
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    visual_obj = args.output_dir / "physical_object.obj"
    visual.export(visual_obj)
    write_json(args.output_dir / "physics_properties.json", {
        "kind": "video2world_modeling.physics_properties",
        "object_id": structured.get("object_id"),
        "estimate": estimate,
        "estimate_status": "single_view_visual_prior_not_measurement",
        "source_description": str(args.description),
        "caveats": structured.get("geometry_caveats", []),
    })
    write_json(args.output_dir / "physics_report.json", {
        "kind": "video2world_modeling.physics_report",
        "visual_obj": str(visual_obj),
        "visual_mesh_source": str(args.visual_mesh),
        "collision_mesh_source": str(args.collision_mesh),
        "visual_vertices": len(visual.vertices),
        "visual_faces": len(visual.faces),
        "coacd_declared_hulls": declared_hulls,
        "collision_loader_geometry_count": len(collision_meshes),
        "collision_watertight": all(mesh.is_watertight for mesh in collision_meshes),
        "texture_status": "not_baked_by_this_adapter",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
