#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const DEFAULT_TARGET_FACES = 90_000;
const DEFAULT_BOUNDS_TOLERANCE_FRACTION = 0.01;

const USAGE = `Usage:
  node scripts/simplify_pbr_glb.mjs \\
    --input <source.glb> \\
    --output <browser-lod.glb> \\
    --report <qa.json> \\
    [--target-faces <count>] \\
    [--bounds-tolerance-fraction <fraction>] \\
    [--blender <path>]

The face budget defaults to 90000. The bounds tolerance defaults to 0.01
(1% of the largest source extent). The Blender path defaults to BLENDER_BIN,
then "blender" on PATH.`;

const BLENDER_SCRIPT = String.raw`
import bpy
import hashlib
import json
import math
import os
import struct
import traceback

input_path = os.environ["V2W_PBR_INPUT"]
output_path = os.environ["V2W_PBR_OUTPUT"]
report_path = os.environ["V2W_PBR_REPORT_TMP"]
target_faces = int(os.environ.get("V2W_PBR_TARGET_FACES", "90000"))
bounds_tolerance_fraction = float(
    os.environ.get("V2W_PBR_BOUNDS_TOLERANCE_FRACTION", "0.01")
)


def sha256_file(file_path):
    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def triangle_count(mesh):
    mesh.calc_loop_triangles()
    return len(mesh.loop_triangles)


def mesh_winding_stats(mesh):
    mesh.calc_loop_triangles()
    edge_directions = {}
    for triangle in mesh.loop_triangles:
        indices = list(triangle.vertices)
        for offset in range(3):
            start = indices[offset]
            end = indices[(offset + 1) % 3]
            key = (min(start, end), max(start, end))
            direction = 1 if start < end else -1
            edge_directions.setdefault(key, []).append(direction)

    shared_edges = 0
    conflicts = 0
    non_manifold_edges = 0
    for directions in edge_directions.values():
        if len(directions) == 2:
            shared_edges += 1
            if directions[0] == directions[1]:
                conflicts += 1
        elif len(directions) > 2:
            non_manifold_edges += 1
    return {
        "sharedEdges": shared_edges,
        "conflicts": conflicts,
        "consistent": conflicts == 0,
        "nonManifoldEdges": non_manifold_edges,
    }


def glb_contract(file_path):
    with open(file_path, "rb") as handle:
        header = handle.read(12)
        if len(header) != 12:
            raise RuntimeError("GLB header is truncated")
        magic, version, declared_length = struct.unpack("<4sII", header)
        if magic != b"glTF" or version != 2:
            raise RuntimeError("Expected a GLB 2.0 asset")
        chunk_header = handle.read(8)
        if len(chunk_header) != 8:
            raise RuntimeError("GLB JSON chunk header is truncated")
        chunk_length, chunk_type = struct.unpack("<II", chunk_header)
        if chunk_type != 0x4E4F534A:
            raise RuntimeError("First GLB chunk is not JSON")
        document = json.loads(handle.read(chunk_length).rstrip(b" \t\r\n\0"))

    primitives = [
        primitive
        for mesh in document.get("meshes", [])
        for primitive in mesh.get("primitives", [])
    ]
    material_names = [material.get("name") for material in document.get("materials", [])]
    return {
        "declaredBytes": declared_length,
        "primitiveCount": len(primitives),
        "materialCount": len(document.get("materials", [])),
        "materialNames": material_names,
        "imageCount": len(document.get("images", [])),
        "allPrimitivesHaveMaterial": bool(primitives) and all(
            "material" in primitive for primitive in primitives
        ),
        "allPrimitivesHaveNormals": bool(primitives) and all(
            "NORMAL" in primitive.get("attributes", {}) for primitive in primitives
        ),
        "allPrimitivesHaveTexcoord0": bool(primitives) and all(
            "TEXCOORD_0" in primitive.get("attributes", {}) for primitive in primitives
        ),
    }


def scene_mesh_objects():
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def clear_scene_data():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for datablock in list(datablocks):
            datablocks.remove(datablock)
    for collection in list(bpy.data.collections):
        if collection != bpy.context.scene.collection:
            bpy.data.collections.remove(collection)


def import_glb(file_path):
    bpy.ops.import_scene.gltf(filepath=file_path, import_pack_images=False)
    bpy.context.view_layer.update()


def material_texture_images(material):
    images = []
    if material and material.use_nodes and material.node_tree:
        for node in material.node_tree.nodes:
            if node.type == "TEX_IMAGE" and node.image is not None:
                images.append(node.image)
    return images


def material_summary(objects):
    used_materials = {}
    material_slots = 0
    for obj in objects:
        material_slots += len(obj.material_slots)
        used_indices = {polygon.material_index for polygon in obj.data.polygons}
        for index in used_indices:
            if index >= len(obj.material_slots):
                continue
            material = obj.material_slots[index].material
            if material is not None:
                used_materials[material.name] = material

    materials = []
    texture_images = {}
    pbr_material_count = 0
    for name in sorted(used_materials):
        material = used_materials[name]
        node_types = []
        if material.use_nodes and material.node_tree:
            node_types = sorted(node.type for node in material.node_tree.nodes)
        is_pbr = "BSDF_PRINCIPLED" in node_types
        if is_pbr:
            pbr_material_count += 1
        image_names = []
        for image in material_texture_images(material):
            width, height = int(image.size[0]), int(image.size[1])
            descriptor = {
                "name": image.name,
                "size": [width, height],
                "hasData": bool(image.has_data),
                "source": image.source,
                "fileFormat": image.file_format,
            }
            texture_images[image.name] = descriptor
            image_names.append(image.name)
        materials.append({
            "name": material.name,
            "useNodes": bool(material.use_nodes),
            "pbrPrincipled": is_pbr,
            "textureImages": sorted(image_names),
        })

    images = [texture_images[name] for name in sorted(texture_images)]
    return {
        "materialSlots": material_slots,
        "usedMaterialCount": len(materials),
        "pbrMaterialCount": pbr_material_count,
        "materials": materials,
        "textureImageCount": len(images),
        "textureImages": images,
        "textureImagesLoaded": all(
            image["hasData"] and image["size"][0] > 0 and image["size"][1] > 0
            for image in images
        ),
    }


def scene_stats(file_path):
    objects = scene_mesh_objects()
    if not objects:
        raise RuntimeError("Imported scene has no mesh objects")

    minimum = [math.inf, math.inf, math.inf]
    maximum = [-math.inf, -math.inf, -math.inf]
    vertices = 0
    faces = 0
    uv_layers = 0
    normal_loops = 0
    normals_finite = True
    degenerate_triangles = 0
    winding_conflicts = 0
    shared_edges = 0
    non_manifold_edges = 0
    vertices_finite = True

    for obj in objects:
        mesh = obj.data
        mesh.calc_loop_triangles()
        vertices += len(mesh.vertices)
        faces += len(mesh.loop_triangles)
        uv_layers += len(mesh.uv_layers)
        normal_loops += len(mesh.loops)
        world = obj.matrix_world

        world_vertices = []
        for vertex in mesh.vertices:
            point = world @ vertex.co
            values = [float(point[axis]) for axis in range(3)]
            world_vertices.append(values)
            for axis, value in enumerate(values):
                vertices_finite = vertices_finite and math.isfinite(value)
                minimum[axis] = min(minimum[axis], value)
                maximum[axis] = max(maximum[axis], value)

        edge_directions = {}
        for triangle in mesh.loop_triangles:
            indices = list(triangle.vertices)
            points = [world_vertices[index] for index in indices]
            ab = [points[1][axis] - points[0][axis] for axis in range(3)]
            ac = [points[2][axis] - points[0][axis] for axis in range(3)]
            cross = [
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            ]
            area_twice = math.sqrt(sum(value * value for value in cross))
            if not math.isfinite(area_twice) or area_twice <= 1e-14:
                degenerate_triangles += 1
            for offset in range(3):
                start = indices[offset]
                end = indices[(offset + 1) % 3]
                key = (min(start, end), max(start, end))
                direction = 1 if start < end else -1
                edge_directions.setdefault(key, []).append(direction)

        for directions in edge_directions.values():
            if len(directions) == 2:
                shared_edges += 1
                if directions[0] == directions[1]:
                    winding_conflicts += 1
            elif len(directions) > 2:
                non_manifold_edges += 1

        for polygon in mesh.polygons:
            normal = polygon.normal
            if not all(math.isfinite(float(normal[axis])) for axis in range(3)):
                normals_finite = False

    center = [(minimum[axis] + maximum[axis]) * 0.5 for axis in range(3)]
    size = [maximum[axis] - minimum[axis] for axis in range(3)]
    values = minimum + maximum + center + size
    finite_bounds = all(math.isfinite(value) for value in values)
    finite_vertices = finite_bounds and vertices_finite and vertices > 0

    result = {
        "path": file_path,
        "bytes": os.path.getsize(file_path),
        "sha256": sha256_file(file_path),
        "meshObjects": len(objects),
        "vertices": vertices,
        "faces": faces,
        "uvLayers": uv_layers,
        "normalLoops": normal_loops,
        "finiteVertices": finite_vertices,
        "normalsFinite": normals_finite,
        "degenerateTriangles": degenerate_triangles,
        "winding": {
            "sharedEdges": shared_edges,
            "conflicts": winding_conflicts,
            "consistent": winding_conflicts == 0,
            "nonManifoldEdges": non_manifold_edges,
        },
        "bounds": {
            "coordinateSystem": "blender_world_z_up",
            "min": minimum,
            "max": maximum,
            "center": center,
            "size": size,
        },
        "glbContract": glb_contract(file_path),
    }
    result["materials"] = material_summary(objects)
    return result


def activate_only(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.hide_set(False)
    obj.hide_viewport = False
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def prepare_render_meshes():
    objects = scene_mesh_objects()
    if not objects:
        raise RuntimeError("Source GLB has no mesh objects")

    for obj in objects:
        world = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = world
        activate_only(obj)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        if obj.data.shape_keys:
            obj.shape_key_clear()
        triangulate = obj.modifiers.new(name="pbr_lod_triangulate", type="TRIANGULATE")
        bpy.ops.object.modifier_apply(modifier=triangulate.name)
        obj.data.update()

    for obj in list(bpy.data.objects):
        if obj.type != "MESH":
            bpy.data.objects.remove(obj, do_unlink=True)
    bpy.context.view_layer.update()
    return scene_mesh_objects()


def repair_winding(objects):
    records = []
    for obj in objects:
        before = mesh_winding_stats(obj.data)
        applied = before["conflicts"] > 0
        if applied:
            activate_only(obj)
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.normals_make_consistent(inside=False)
            bpy.ops.object.mode_set(mode="OBJECT")
            obj.data.update()
        after = mesh_winding_stats(obj.data)
        records.append({
            "object": obj.name,
            "applied": applied,
            "before": before,
            "after": after,
        })
    return records


def apply_decimate(obj, ratio, label):
    before = triangle_count(obj.data)
    if before <= 4 or ratio >= 0.999999:
        return {
            "object": obj.name,
            "pass": label,
            "ratio": 1.0,
            "beforeFaces": before,
            "afterFaces": before,
        }
    activate_only(obj)
    modifier = obj.modifiers.new(name="pbr_lod_decimate_" + label, type="DECIMATE")
    modifier.decimate_type = "COLLAPSE"
    modifier.ratio = max(0.000001, min(1.0, ratio))
    modifier.use_collapse_triangulate = True
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    obj.data.update()
    after = triangle_count(obj.data)
    return {
        "object": obj.name,
        "pass": label,
        "ratio": ratio,
        "beforeFaces": before,
        "afterFaces": after,
    }


def decimate_to_target(objects, target):
    attempts = []
    initial_faces = sum(triangle_count(obj.data) for obj in objects)
    current_faces = initial_faces
    for pass_index in range(8):
        if current_faces <= target:
            break
        ratio = min(0.995, (target / current_faces) * 0.985)
        before_pass = current_faces
        for obj in sorted(objects, key=lambda item: triangle_count(item.data), reverse=True):
            attempts.append(apply_decimate(obj, ratio, str(pass_index + 1)))
        current_faces = sum(triangle_count(obj.data) for obj in objects)
        if current_faces >= before_pass:
            break
    return initial_faces, current_faces, attempts


def export_pbr(file_path, objects):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.export_scene.gltf(
        filepath=file_path,
        export_format="GLB",
        use_selection=True,
        export_materials="EXPORT",
        export_image_format="AUTO",
        export_normals=True,
        export_tangents=False,
        export_texcoords=True,
        export_attributes=True,
        export_vertex_color="MATERIAL",
        export_cameras=False,
        export_lights=False,
        export_animations=False,
        export_morph=False,
        export_skins=False,
        export_extras=True,
        export_yup=True,
        check_existing=False,
    )


def write_report(payload):
    temporary = report_path + ".write"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")
    os.replace(temporary, report_path)


report = {
    "schemaVersion": 1,
    "kind": "video2world.pbr_browser_lod_qa",
    "status": "failed",
    "source": {"path": input_path},
    "target": {"faces": target_faces},
    "output": {"path": output_path},
    "qa": {},
    "processing": {
        "tool": "blender_headless_pbr_decimate",
        "blenderVersion": bpy.app.version_string,
        "blenderVersionTuple": list(bpy.app.version),
        "strategy": "per_mesh_object_proportional_collapse_preserving_material_slots",
    },
}

try:
    clear_scene_data()
    import_glb(input_path)
    source = scene_stats(input_path)
    objects = prepare_render_meshes()
    winding_repairs = repair_winding(objects)
    prepared_faces, simplified_faces, attempts = decimate_to_target(objects, target_faces)
    export_pbr(output_path, objects)

    clear_scene_data()
    import_glb(output_path)
    output = scene_stats(output_path)

    bounds_delta = []
    for field in ("min", "max"):
        for axis in range(3):
            bounds_delta.append(abs(output["bounds"][field][axis] - source["bounds"][field][axis]))
    maximum_extent = max(source["bounds"]["size"])
    bounds_tolerance = maximum_extent * bounds_tolerance_fraction
    maximum_bounds_delta = max(bounds_delta)
    bounds_drift_fraction = maximum_bounds_delta / max(maximum_extent, 0.000000001)

    source_materials = source["materials"]
    output_materials = output["materials"]
    target_met = output["faces"] <= target_faces
    nonempty = output["vertices"] > 0 and output["faces"] > 0
    finite_geometry = output["finiteVertices"] and output["normalsFinite"]
    nondegenerate = output["degenerateTriangles"] == 0
    winding_consistent = output["winding"]["consistent"]
    winding_repair_successful = all(
        record["after"]["consistent"] for record in winding_repairs
    )
    bounds_preserved = maximum_bounds_delta <= bounds_tolerance
    pbr_materials_preserved = (
        source_materials["pbrMaterialCount"] > 0
        and output_materials["pbrMaterialCount"] >= source_materials["pbrMaterialCount"]
    )
    source_material_names = {
        material["name"] for material in source_materials["materials"]
    }
    output_material_names = {
        material["name"] for material in output_materials["materials"]
    }
    material_slots_preserved = (
        output_materials["usedMaterialCount"] >= source_materials["usedMaterialCount"]
        and output_materials["materialSlots"] >= source_materials["usedMaterialCount"]
        and source_material_names.issubset(output_material_names)
        and output["glbContract"]["allPrimitivesHaveMaterial"]
    )
    source_texture_images = {
        image["name"]: image["size"] for image in source_materials["textureImages"]
    }
    output_texture_images = {
        image["name"]: image["size"] for image in output_materials["textureImages"]
    }
    texture_images_present = (
        source_materials["textureImageCount"] > 0
        and source_materials["textureImagesLoaded"]
        and output_materials["textureImageCount"] >= source_materials["textureImageCount"]
        and output_materials["textureImagesLoaded"]
        and all(
            output_texture_images.get(name) == size
            for name, size in source_texture_images.items()
        )
        and output["glbContract"]["imageCount"] > 0
    )
    uv_layers_preserved = (
        source["uvLayers"] > 0 and output["uvLayers"] >= source["uvLayers"]
        and output["glbContract"]["allPrimitivesHaveTexcoord0"]
    )
    normals_preserved = (
        source["normalLoops"] > 0
        and output["normalLoops"] > 0
        and output["glbContract"]["allPrimitivesHaveNormals"]
    )

    report.update({
        "source": source,
        "target": {
            "faces": target_faces,
            "met": target_met,
            "reductionRequested": target_faces < source["faces"],
        },
        "output": output,
        "qa": {
            "reimportedAfterExport": True,
            "targetMet": target_met,
            "nonempty": nonempty,
            "finiteGeometry": finite_geometry,
            "nondegenerateTriangles": nondegenerate,
            "windingConsistent": winding_consistent,
            "windingRepairSuccessful": winding_repair_successful,
            "pbrMaterialsPreserved": pbr_materials_preserved,
            "materialSlotsPreserved": material_slots_preserved,
            "textureImagesPresent": texture_images_present,
            "uvLayersPreserved": uv_layers_preserved,
            "normalsPreserved": normals_preserved,
            "boundsTolerance": bounds_tolerance,
            "maxAbsBoundsDelta": maximum_bounds_delta,
            "boundsDriftFraction": bounds_drift_fraction,
            "boundsToleranceFraction": bounds_tolerance_fraction,
            "boundsPreservedWithinTolerance": bounds_preserved,
        },
        "processing": {
            **report["processing"],
            "sourceFacesAfterTriangulate": prepared_faces,
            "facesBeforeExport": simplified_faces,
            "windingRepair": winding_repairs,
            "decimationAttempts": attempts,
        },
        "assetBinding": {
            "input": {
                "path": source["path"],
                "sha256": source["sha256"],
                "bytes": source["bytes"],
                "faces": source["faces"],
                "bounds": source["bounds"],
                "pbrMaterials": source["materials"]["pbrMaterialCount"],
                "textureImages": source["materials"]["textureImageCount"],
                "uvLayers": source["uvLayers"],
                "normalLoops": source["normalLoops"],
                "glbHasNormals": source["glbContract"]["allPrimitivesHaveNormals"],
                "degenerateTriangles": source["degenerateTriangles"],
                "windingConsistent": source["winding"]["consistent"],
            },
            "output": {
                "path": output["path"],
                "sha256": output["sha256"],
                "bytes": output["bytes"],
                "faces": output["faces"],
                "bounds": output["bounds"],
                "pbrMaterials": output["materials"]["pbrMaterialCount"],
                "textureImages": output["materials"]["textureImageCount"],
                "uvLayers": output["uvLayers"],
                "normalLoops": output["normalLoops"],
                "glbHasNormals": output["glbContract"]["allPrimitivesHaveNormals"],
                "degenerateTriangles": output["degenerateTriangles"],
                "windingConsistent": output["winding"]["consistent"],
            },
        },
    })
    required_checks = {
        "reimportedAfterExport": True,
        "targetMet": target_met,
        "nonempty": nonempty,
        "finiteGeometry": finite_geometry,
        "nondegenerateTriangles": nondegenerate,
        "windingConsistent": winding_consistent,
        "windingRepairSuccessful": winding_repair_successful,
        "pbrMaterialsPreserved": pbr_materials_preserved,
        "materialSlotsPreserved": material_slots_preserved,
        "textureImagesPresent": texture_images_present,
        "uvLayersPreserved": uv_layers_preserved,
        "normalsPreserved": normals_preserved,
        "boundsPreservedWithinTolerance": bounds_preserved,
    }
    report["status"] = "passed" if all(required_checks.values()) else "failed"
    if report["status"] != "passed":
        failed_checks = [name for name, passed in required_checks.items() if not passed]
        report["error"] = "PBR GLB failed required QA checks: " + ", ".join(failed_checks)
    write_report(report)
    if report["status"] != "passed":
        raise RuntimeError(report["error"])
except Exception as error:
    report["status"] = "failed"
    report["error"] = str(error)
    report["traceback"] = traceback.format_exc()
    write_report(report)
    raise
`;

export const REQUIRED_QA_CHECKS = Object.freeze([
  "reimportedAfterExport",
  "targetMet",
  "nonempty",
  "finiteGeometry",
  "nondegenerateTriangles",
  "windingConsistent",
  "windingRepairSuccessful",
  "pbrMaterialsPreserved",
  "materialSlotsPreserved",
  "textureImagesPresent",
  "uvLayersPreserved",
  "normalsPreserved",
  "boundsPreservedWithinTolerance",
]);

export function parseArgs(argv) {
  const options = {};
  const allowed = new Set([
    "input",
    "output",
    "report",
    "target-faces",
    "bounds-tolerance-fraction",
    "blender",
  ]);
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === "--help" || token === "-h") {
      options.help = true;
      continue;
    }
    if (!token.startsWith("--")) throw new Error(`Unexpected argument: ${token}`);
    const key = token.slice(2);
    if (!allowed.has(key)) throw new Error(`Unknown option: --${key}`);
    const value = argv[index + 1];
    if (!value || value.startsWith("--")) throw new Error(`Missing value for --${key}`);
    if (Object.hasOwn(options, key)) throw new Error(`Duplicate option: --${key}`);
    options[key] = value;
    index += 1;
  }
  return options;
}

function required(options, key) {
  if (!options[key]) throw new Error(`Missing required option --${key}`);
  return options[key];
}

function writeJsonAtomic(filePath, payload) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  const temporary = `${filePath}.${process.pid}.tmp`;
  fs.writeFileSync(temporary, `${JSON.stringify(payload, null, 2)}\n`);
  fs.renameSync(temporary, filePath);
}

function sha256File(filePath) {
  const digest = crypto.createHash("sha256");
  digest.update(fs.readFileSync(filePath));
  return digest.digest("hex");
}

function commandTail(value, maximumLines = 80) {
  const lines = String(value ?? "").trim().split(/\r?\n/);
  return lines.slice(-maximumLines).join("\n");
}

function resolveBlender(requested) {
  const candidates = requested
    ? [requested]
    : [process.env.BLENDER_BIN, "blender", "/Applications/Blender.app/Contents/MacOS/Blender"].filter(Boolean);
  for (const candidate of [...new Set(candidates)]) {
    const probe = spawnSync(candidate, ["--version"], { encoding: "utf8" });
    if (!probe.error && probe.status === 0) return candidate;
  }
  throw new Error(`Unable to find Blender. Tried: ${[...new Set(candidates)].join(", ")}`);
}

export function failedRequiredQaChecks(report) {
  return REQUIRED_QA_CHECKS.filter((check) => report.qa?.[check] !== true);
}

export function reportBindingFailures(report) {
  const failures = [];
  for (const [bindingKey, statsKey] of [["input", "source"], ["output", "output"]]) {
    const binding = report.assetBinding?.[bindingKey];
    const stats = report[statsKey];
    if (!binding || !stats) {
      failures.push(`${bindingKey}:missing`);
      continue;
    }
    const expected = {
      path: stats.path,
      sha256: stats.sha256,
      bytes: stats.bytes,
      faces: stats.faces,
      bounds: stats.bounds,
      pbrMaterials: stats.materials?.pbrMaterialCount,
      textureImages: stats.materials?.textureImageCount,
      uvLayers: stats.uvLayers,
      normalLoops: stats.normalLoops,
      glbHasNormals: stats.glbContract?.allPrimitivesHaveNormals,
      degenerateTriangles: stats.degenerateTriangles,
      windingConsistent: stats.winding?.consistent,
    };
    for (const [field, value] of Object.entries(expected)) {
      if (JSON.stringify(binding[field]) !== JSON.stringify(value)) {
        failures.push(`${bindingKey}:${field}`);
      }
    }
    if (!/^[a-f0-9]{64}$/.test(String(binding.sha256 ?? ""))) {
      failures.push(`${bindingKey}:sha256Format`);
    }
  }
  return failures;
}

export function normalizeOptions(options, cwd = process.cwd()) {
  const inputPath = path.resolve(cwd, required(options, "input"));
  const outputPath = path.resolve(cwd, required(options, "output"));
  const reportPath = path.resolve(cwd, required(options, "report"));
  const targetFaces = Number(options["target-faces"] ?? DEFAULT_TARGET_FACES);
  const boundsToleranceFraction = Number(
    options["bounds-tolerance-fraction"] ?? DEFAULT_BOUNDS_TOLERANCE_FRACTION,
  );
  if (!Number.isSafeInteger(targetFaces) || targetFaces <= 0) {
    throw new Error(`--target-faces must be a positive integer; got ${options["target-faces"]}`);
  }
  if (!Number.isFinite(boundsToleranceFraction) || boundsToleranceFraction < 0) {
    throw new Error(
      `--bounds-tolerance-fraction must be a finite non-negative number; got ${options["bounds-tolerance-fraction"]}`,
    );
  }
  if (!/\.glb$/i.test(inputPath) || !/\.glb$/i.test(outputPath)) {
    throw new Error("--input and --output must both use the .glb extension");
  }
  if (!/\.json$/i.test(reportPath)) throw new Error("--report must use the .json extension");
  if (inputPath === outputPath) throw new Error("--output must differ from --input");
  return { inputPath, outputPath, reportPath, targetFaces, boundsToleranceFraction };
}

export function run(argv = process.argv.slice(2)) {
  const options = parseArgs(argv);
  if (options.help) {
    console.log(USAGE);
    return;
  }

  const normalized = normalizeOptions(options);
  const { inputPath, outputPath, reportPath, targetFaces, boundsToleranceFraction } = normalized;
  if (!fs.existsSync(inputPath) || !fs.statSync(inputPath).isFile()) {
    throw new Error(`Input GLB does not exist: ${inputPath}`);
  }

  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.mkdirSync(path.dirname(reportPath), { recursive: true });
  const blender = resolveBlender(options.blender);
  const temporaryReport = path.join(
    os.tmpdir(),
    `video2world-pbr-${process.pid}-${crypto.randomBytes(6).toString("hex")}.json`,
  );
  const startedAt = new Date().toISOString();
  const result = spawnSync(
    blender,
    ["--background", "--factory-startup", "--python-expr", BLENDER_SCRIPT],
    {
      encoding: "utf8",
      maxBuffer: 64 * 1024 * 1024,
      env: {
        ...process.env,
        V2W_PBR_INPUT: inputPath,
        V2W_PBR_OUTPUT: outputPath,
        V2W_PBR_REPORT_TMP: temporaryReport,
        V2W_PBR_TARGET_FACES: String(targetFaces),
        V2W_PBR_BOUNDS_TOLERANCE_FRACTION: String(boundsToleranceFraction),
      },
    },
  );

  let report = {
    schemaVersion: 1,
    kind: "video2world.pbr_browser_lod_qa",
    status: "failed",
    source: {
      path: inputPath,
      bytes: fs.statSync(inputPath).size,
      sha256: sha256File(inputPath),
    },
    target: { faces: targetFaces },
    output: { path: outputPath },
    error: "Blender exited without producing a QA report",
  };
  if (fs.existsSync(temporaryReport)) {
    try {
      report = JSON.parse(fs.readFileSync(temporaryReport, "utf8"));
    } finally {
      fs.rmSync(temporaryReport, { force: true });
    }
  }

  report.invocation = {
    startedAt,
    finishedAt: new Date().toISOString(),
    blender,
    exitCode: result.status,
    signal: result.signal,
  };
  if (result.error) report.invocation.spawnError = result.error.message;
  const failedQaChecks = failedRequiredQaChecks(report);
  const bindingFailures = reportBindingFailures(report);
  if (report.status === "passed" && (failedQaChecks.length || bindingFailures.length)) {
    report.status = "failed";
    report.error = [
      failedQaChecks.length ? `required QA: ${failedQaChecks.join(", ")}` : null,
      bindingFailures.length ? `asset binding: ${bindingFailures.join(", ")}` : null,
    ].filter(Boolean).join("; ");
  }
  if (result.status !== 0) {
    report.status = "failed";
    report.error ||= `Blender exited with code ${result.status}`;
    report.blenderStdoutTail = commandTail(result.stdout);
    report.blenderStderrTail = commandTail(result.stderr);
  }
  writeJsonAtomic(reportPath, report);

  if (report.status !== "passed" || result.status !== 0) {
    throw new Error(`${report.error ?? "PBR GLB QA failed"}. Report: ${reportPath}`);
  }
  console.log(JSON.stringify({
    status: report.status,
    source: report.source.path,
    output: report.output.path,
    report: reportPath,
    sourceFaces: report.source.faces,
    outputFaces: report.output.faces,
    targetFaces: report.target.faces,
    materials: report.output.materials.usedMaterialCount,
    textureImages: report.output.materials.textureImageCount,
    boundsDriftFraction: report.qa.boundsDriftFraction,
  }, null, 2));
}

const isMain = process.argv[1]
  && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url));
if (isMain) {
  try {
    run();
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    console.error(USAGE);
    process.exitCode = 1;
  }
}
