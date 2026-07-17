#!/usr/bin/env python3
"""Render the bed and independently placed pillows from one calibrated camera.

Run with Blender, passing script arguments after ``--``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

RAW_TO_BLENDER = Matrix(
    (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, -1.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
)


def raw_transform_to_blender(transform: Matrix) -> Matrix:
    return RAW_TO_BLENDER @ transform @ RAW_TO_BLENDER.inverted()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bed-mesh", type=Path, required=True)
    parser.add_argument("--bed-report", type=Path, required=True)
    parser.add_argument("--pillow", action="append", nargs=3, metavar=("ID", "MESH", "REPORT"))
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--frame-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])


def load_camera(path: Path, frame_id: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    matches = [item for item in payload if str(item.get("img_name")) == frame_id]
    if len(matches) != 1:
        raise ValueError(f"expected one camera for {frame_id!r}, found {len(matches)}")
    return matches[0]


def import_glb(path: Path, transform: Matrix, label: str) -> list[bpy.types.Object]:
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    imported = [item for item in bpy.data.objects if item not in before]
    imported_set = set(imported)
    roots = [item for item in imported if item.parent not in imported_set]
    for root in roots:
        root.matrix_world = transform @ root.matrix_world
    for item in imported:
        item["video2world_entity"] = label
    return imported


def configure_camera(camera_info: dict) -> bpy.types.Object:
    data = bpy.data.cameras.new("video2world_camera")
    camera = bpy.data.objects.new("video2world_camera", data)
    bpy.context.collection.objects.link(camera)
    position = (RAW_TO_BLENDER @ Vector((*camera_info["position"], 1.0))).to_3d()
    rotation = camera_info["rotation"]
    right = RAW_TO_BLENDER.to_3x3() @ Vector((rotation[0][0], rotation[1][0], rotation[2][0]))
    down = RAW_TO_BLENDER.to_3x3() @ Vector((rotation[0][1], rotation[1][1], rotation[2][1]))
    forward = RAW_TO_BLENDER.to_3x3() @ Vector((rotation[0][2], rotation[1][2], rotation[2][2]))
    camera.matrix_world = Matrix(
        (
            (right.x, -down.x, -forward.x, position.x),
            (right.y, -down.y, -forward.y, position.y),
            (right.z, -down.z, -forward.z, position.z),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    sensor_width = 36.0
    data.sensor_fit = "HORIZONTAL"
    data.sensor_width = sensor_width
    data.lens = float(camera_info["fx"]) * sensor_width / int(camera_info["width"])
    data.clip_start = 0.01
    data.clip_end = 1000.0
    return camera


def add_point_light(name: str, location: Vector, energy: float, size: float) -> None:
    data = bpy.data.lights.new(name=name, type="POINT")
    data.energy = energy
    data.shadow_soft_size = size
    light = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(light)
    light.location = location


def main() -> int:
    args = parse_args()
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    camera_info = load_camera(args.cameras.resolve(), args.frame_id)
    bed_report = json.loads(args.bed_report.read_text(encoding="utf-8"))
    bed_transform = raw_transform_to_blender(
        Matrix(bed_report["runtime_transform"]["matrix_row_major"])
    )
    import_glb(args.bed_mesh.resolve(), bed_transform, "sam3_bed_01")
    for object_id, mesh_value, report_value in args.pillow or []:
        report = json.loads(Path(report_value).read_text(encoding="utf-8"))
        pivot = Vector(report["baked_relative_transform"]["runtime_pivot"])
        blender_pivot = RAW_TO_BLENDER.to_3x3() @ pivot
        import_glb(
            Path(mesh_value).resolve(),
            Matrix.Translation(blender_pivot),
            object_id,
        )

    scene = bpy.context.scene
    bpy.context.view_layer.update()
    scene.camera = configure_camera(camera_info)
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = int(camera_info["width"])
    scene.render.resolution_y = int(camera_info["height"])
    scene.render.resolution_percentage = 50
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.render.filepath = str(args.output.resolve())
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = float(camera_info["fx"]) / float(camera_info["fy"])
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.world.color = (0.08, 0.08, 0.08)
    position = (RAW_TO_BLENDER @ Vector((*camera_info["position"], 1.0))).to_3d()
    add_point_light(
        "camera_fill",
        Vector((position.x, position.y, position.z + 1.0)),
        1400.0,
        4.0,
    )
    add_point_light(
        "room_fill",
        (RAW_TO_BLENDER @ Vector((1.0, -4.0, 8.0, 1.0))).to_3d(),
        1000.0,
        5.0,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.render.render(write_still=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
