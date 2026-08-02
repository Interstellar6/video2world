#!/usr/bin/env python3
"""Render six hash-bound object-axis review views for a GLB with Blender."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glb", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    return parser.parse_args(argv)


def file_record(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def world_bounds() -> tuple[Vector, Vector]:
    points = [
        obj.matrix_world @ Vector(corner)
        for obj in bpy.context.scene.objects
        if obj.type == "MESH"
        for corner in obj.bound_box
    ]
    if not points:
        raise RuntimeError("the GLB did not import any mesh objects")
    return (
        Vector(tuple(min(point[index] for point in points) for index in range(3))),
        Vector(tuple(max(point[index] for point in points) for index in range(3))),
    )


def aim_at(camera: bpy.types.Object, target: Vector) -> None:
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()


def add_area_light(
    name: str,
    location: Vector,
    energy: float,
    size: float,
    target: Vector,
) -> None:
    data = bpy.data.lights.new(name=name, type="AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    light = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(light)
    light.location = location
    aim_at(light, target)


def configure_scene(size: int, center: Vector, extent: Vector) -> bpy.types.Object:
    scene = bpy.context.scene
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = size
    scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.world.color = (0.035, 0.045, 0.055)
    scene.view_settings.look = "AgX - Medium High Contrast"

    radius = max(float(extent.length) * 0.5, 1e-4)
    camera_data = bpy.data.cameras.new("ReviewCamera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = max(float(max(extent)) * 1.5, radius * 1.2)
    camera = bpy.data.objects.new("ReviewCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera

    add_area_light(
        "Key", center + Vector((1.4, -1.8, 2.4)) * radius, 900.0, radius * 3.0, center
    )
    add_area_light(
        "Fill", center + Vector((-2.0, -0.8, 1.0)) * radius, 500.0, radius * 2.5, center
    )
    add_area_light(
        "Rim", center + Vector((0.4, 2.2, 1.6)) * radius, 700.0, radius * 2.0, center
    )
    return camera


def main() -> int:
    args = parse_args()
    if not 64 <= args.size <= 4096:
        raise ValueError("--size must be inside [64, 4096]")
    glb = args.glb.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not glb.is_file():
        raise FileNotFoundError(glb)
    output_root.mkdir(parents=True, exist_ok=True)

    clear_scene()
    bpy.ops.import_scene.gltf(filepath=str(glb))
    lower, upper = world_bounds()
    center = (lower + upper) * 0.5
    extent = upper - lower
    camera = configure_scene(args.size, center, extent)
    radius = max(float(extent.length) * 0.5, 1e-4)
    distance = radius * 3.0
    directions = (
        ("x_positive", Vector((1.0, 0.0, 0.0))),
        ("x_negative", Vector((-1.0, 0.0, 0.0))),
        ("y_positive", Vector((0.0, 1.0, 0.0))),
        ("y_negative", Vector((0.0, -1.0, 0.0))),
        ("z_positive", Vector((0.0, 0.0, 1.0))),
        ("z_negative", Vector((0.0, 0.0, -1.0))),
    )
    views = []
    for view_id, direction in directions:
        output = output_root / f"{view_id}.png"
        camera.location = center + direction * distance
        aim_at(camera, center)
        bpy.context.scene.render.filepath = str(output)
        bpy.ops.render.render(write_still=True)
        views.append({"view_id": view_id, **file_record(output)})

    receipt = {
        "schema_version": 1,
        "kind": "video2world.glb_axis_review",
        "status": "rendered_pending_manual_review",
        "source_glb": file_record(glb),
        "bounds": {"min": list(lower), "max": list(upper), "center": list(center)},
        "semantic_front_back": "unresolved_coordinate_axis_only",
        "promotion_allowed": False,
        "views": views,
    }
    receipt_path = output_root / "render_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"receipt": str(receipt_path), "views": len(views)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
