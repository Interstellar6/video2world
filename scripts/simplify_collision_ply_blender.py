#!/usr/bin/env python3
"""Create a browser collision LOD from a dense triangular PLY in Blender."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import bpy


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--target-faces", type=int, default=500_000)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def mesh_bounds(mesh: bpy.types.Mesh) -> dict[str, list[float]]:
    if not mesh.vertices:
        raise RuntimeError("mesh has no vertices")
    minimum = [float("inf")] * 3
    maximum = [float("-inf")] * 3
    for vertex in mesh.vertices:
        for axis, value in enumerate(vertex.co):
            minimum[axis] = min(minimum[axis], float(value))
            maximum[axis] = max(maximum[axis], float(value))
    return {
        "min": minimum,
        "max": maximum,
        "size": [maximum[axis] - minimum[axis] for axis in range(3)],
    }


def main() -> int:
    args = parse_args()
    source = args.input.expanduser().resolve()
    destination = args.output.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    if args.target_faces < 10_000:
        raise ValueError("target-faces must be at least 10000")
    if source.suffix.lower() != ".ply" or destination.suffix.lower() != ".ply":
        raise ValueError("input and output must be PLY files")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.wm.ply_import(
        filepath=str(source),
        forward_axis="Y",
        up_axis="Z",
        merge_verts=False,
    )
    meshes = [item for item in bpy.context.selected_objects if item.type == "MESH"]
    if len(meshes) != 1:
        raise RuntimeError(f"expected one imported mesh, found {len(meshes)}")
    obj = meshes[0]
    bpy.context.view_layer.objects.active = obj
    source_faces = len(obj.data.polygons)
    source_vertices = len(obj.data.vertices)
    source_bounds = mesh_bounds(obj.data)
    if source_faces <= args.target_faces:
        raise RuntimeError("source is already below the requested face budget")

    modifier = obj.modifiers.new(name="Video2World browser collision LOD", type="DECIMATE")
    modifier.decimate_type = "COLLAPSE"
    modifier.ratio = max(0.001, min(1.0, args.target_faces / source_faces))
    modifier.use_collapse_triangulate = True
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    output_faces = len(obj.data.polygons)
    output_vertices = len(obj.data.vertices)
    output_bounds = mesh_bounds(obj.data)
    if output_faces > int(args.target_faces * 1.02):
        raise RuntimeError(f"decimator missed face budget: {output_faces} > {args.target_faces}")
    if output_faces < int(args.target_faces * 0.75):
        raise RuntimeError(f"decimator removed too many faces: {output_faces}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.wm.ply_export(
        filepath=str(destination),
        check_existing=False,
        forward_axis="Y",
        up_axis="Z",
        apply_modifiers=True,
        export_selected_objects=True,
        export_triangulated_mesh=True,
        ascii_format=False,
    )
    maximum_bounds_delta = max(
        abs(output_bounds[field][axis] - source_bounds[field][axis])
        for field in ("min", "max")
        for axis in range(3)
    )
    bounds_tolerance = max(source_bounds["size"]) * 0.001
    report = {
        "schemaVersion": 1,
        "kind": "video2world.browser_collision_ply_lod",
        "createdAt": datetime.now(UTC).isoformat(),
        "status": "passed",
        "method": "Blender collapse decimation with triangulated PLY export",
        "source": {
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
            "vertices": source_vertices,
            "faces": source_faces,
            "bounds": source_bounds,
        },
        "output": {
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
            "vertices": output_vertices,
            "faces": output_faces,
            "bounds": output_bounds,
        },
        "targetFaces": args.target_faces,
        "faceRetentionRatio": output_faces / source_faces,
        "maximumBoundsDelta": maximum_bounds_delta,
        "boundsTolerance": bounds_tolerance,
        "gates": {
            "nonempty": output_vertices > 0 and output_faces > 0,
            "faceBudget": output_faces <= int(args.target_faces * 1.02),
            "boundsPreserved": maximum_bounds_delta <= bounds_tolerance,
        },
    }
    if not all(report["gates"].values()):
        report["status"] = "failed"
    write_json(report_path, report)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
