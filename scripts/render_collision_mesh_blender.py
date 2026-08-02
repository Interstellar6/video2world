#!/usr/bin/env python3
"""Render a loose-part, color-coded collider preview in Blender."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=900)
    return parser.parse_args(args)


def import_mesh(path: Path) -> list[bpy.types.Object]:
    before = set(bpy.data.objects)
    suffix = path.suffix.lower()
    if suffix == ".obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=str(path))
        else:
            bpy.ops.import_scene.obj(filepath=str(path))
    elif suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
    else:
        raise ValueError(f"unsupported mesh extension: {suffix}")
    meshes = [item for item in set(bpy.data.objects) - before if item.type == "MESH"]
    if not meshes:
        raise ValueError("imported asset contains no mesh objects")
    bpy.ops.object.select_all(action="DESELECT")
    for mesh in meshes:
        mesh.select_set(True)
        bpy.context.view_layer.objects.active = mesh
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.separate(type="LOOSE")
        bpy.ops.object.mode_set(mode="OBJECT")
    return [item for item in bpy.context.scene.objects if item.type == "MESH"]


def bounds(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    corners = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    minimum = Vector(tuple(min(point[index] for point in corners) for index in range(3)))
    maximum = Vector(tuple(max(point[index] for point in corners) for index in range(3)))
    return minimum, maximum


def material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    value = bpy.data.materials.new(name=name)
    value.diffuse_color = color
    value.use_nodes = True
    principled = next(node for node in value.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
    principled.inputs["Base Color"].default_value = color
    principled.inputs["Roughness"].default_value = 0.48
    principled.inputs["Metallic"].default_value = 0.05
    return value


def main() -> None:
    args = parse_args()
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if args.resolution < 256 or args.resolution > 4096:
        raise ValueError("resolution must be inside [256, 4096]")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    objects = import_mesh(source)
    palette = (
        (0.10, 0.55, 0.82, 1.0),
        (0.92, 0.40, 0.18, 1.0),
        (0.15, 0.68, 0.42, 1.0),
        (0.90, 0.67, 0.12, 1.0),
        (0.55, 0.32, 0.78, 1.0),
    )
    for index, obj in enumerate(sorted(objects, key=lambda item: item.name)):
        obj.data.materials.clear()
        obj.data.materials.append(material(f"hull_{index:03d}", palette[index % len(palette)]))

    minimum, maximum = bounds(objects)
    center = (minimum + maximum) * 0.5
    extents = maximum - minimum
    radius = max(extents) * 1.8
    camera_data = bpy.data.cameras.new("Camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = max(extents) * 1.45
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = center + Vector((1.35, -1.65, 1.05)).normalized() * radius
    camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = camera

    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (
        0.035,
        0.04,
        0.045,
        1.0,
    )
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.55
    for index, direction in enumerate(((4, -5, 7), (-4, 1, 3))):
        light_data = bpy.data.lights.new(f"Area{index}", type="AREA")
        light_data.energy = 750.0 if index == 0 else 400.0
        light_data.shape = "DISK"
        light_data.size = radius * 1.4
        light = bpy.data.objects.new(f"Area{index}", light_data)
        bpy.context.collection.objects.link(light)
        light.location = center + Vector(direction).normalized() * radius * 2.0
        light.rotation_euler = (center - light.location).to_track_quat("-Z", "Y").to_euler()

    scene = bpy.context.scene
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.render.filepath = str(output)
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.camera.data.lens = 52
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
