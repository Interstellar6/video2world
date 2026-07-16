#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { spawnSync } from "node:child_process";

const USAGE = `Usage:
  node scripts/simplify_collision_glb.mjs \\
    --input <source.glb> \\
    --output <collision.glb> \\
    --report <qa.json> \\
    --target-faces <count> \\
    [--bounds-tolerance-fraction <fraction>] \\
    [--blender <path>]

The bounds tolerance defaults to 0.01 (1% of the largest source extent).
The Blender path defaults to BLENDER_BIN, then "blender" on PATH.`;

const BLENDER_SCRIPT = String.raw`
import bpy
import hashlib
import json
import math
import os
import traceback

input_path = os.environ["V2W_COLLISION_INPUT"]
output_path = os.environ["V2W_COLLISION_OUTPUT"]
report_path = os.environ["V2W_COLLISION_REPORT_TMP"]
target_faces = int(os.environ["V2W_COLLISION_TARGET_FACES"])
bounds_tolerance_fraction = float(os.environ.get("V2W_COLLISION_BOUNDS_TOLERANCE_FRACTION", "0.01"))


def sha256_file(file_path):
    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def triangle_count(mesh):
    return sum(max(0, len(polygon.vertices) - 2) for polygon in mesh.polygons)


def scene_mesh_objects():
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def scene_stats(file_path):
    objects = scene_mesh_objects()
    if not objects:
        raise RuntimeError("Imported scene has no mesh objects")

    minimum = [math.inf, math.inf, math.inf]
    maximum = [-math.inf, -math.inf, -math.inf]
    vertices = 0
    faces = 0
    material_slots = 0
    for obj in objects:
        mesh = obj.data
        vertices += len(mesh.vertices)
        faces += triangle_count(mesh)
        material_slots += len(obj.material_slots)
        world = obj.matrix_world
        for vertex in mesh.vertices:
            point = world @ vertex.co
            for axis in range(3):
                minimum[axis] = min(minimum[axis], float(point[axis]))
                maximum[axis] = max(maximum[axis], float(point[axis]))

    center = [(minimum[axis] + maximum[axis]) * 0.5 for axis in range(3)]
    size = [maximum[axis] - minimum[axis] for axis in range(3)]
    values = minimum + maximum + center + size
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("Imported scene has non-finite bounds")

    return {
        "path": file_path,
        "bytes": os.path.getsize(file_path),
        "sha256": sha256_file(file_path),
        "meshObjects": len(objects),
        "vertices": vertices,
        "faces": faces,
        "materialSlots": material_slots,
        "bounds": {
            "coordinateSystem": "blender_world_z_up",
            "min": minimum,
            "max": maximum,
            "center": center,
            "size": size,
        },
    }


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in list(bpy.data.collections):
        if collection.users == 0:
            bpy.data.collections.remove(collection)


def import_glb(file_path):
    bpy.ops.import_scene.gltf(filepath=file_path, import_pack_images=False)
    bpy.context.view_layer.update()


def prepare_collision_mesh():
    mesh_objects = scene_mesh_objects()
    if not mesh_objects:
        raise RuntimeError("Source GLB has no mesh objects")

    bpy.ops.object.select_all(action="DESELECT")
    for obj in mesh_objects:
        obj.hide_set(False)
        obj.hide_viewport = False
        world = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = world
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objects[0]
    bpy.ops.object.join()
    collision = bpy.context.view_layer.objects.active
    collision.name = "collision"
    collision.data.name = "collision"

    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    if collision.data.shape_keys:
        collision.shape_key_clear()
    collision.data.materials.clear()
    while collision.data.uv_layers:
        collision.data.uv_layers.remove(collision.data.uv_layers[0])
    color_attributes = getattr(collision.data, "color_attributes", None)
    if color_attributes is not None:
        while color_attributes:
            color_attributes.remove(color_attributes[0])

    triangulate = collision.modifiers.new(name="collision_triangulate", type="TRIANGULATE")
    bpy.ops.object.modifier_apply(modifier=triangulate.name)

    for obj in list(bpy.data.objects):
        if obj != collision:
            bpy.data.objects.remove(obj, do_unlink=True)
    bpy.context.view_layer.update()
    return collision


def decimate_to_target(collision, target):
    initial_faces = triangle_count(collision.data)
    attempts = []
    current_faces = initial_faces
    for attempt in range(4):
        if current_faces <= target:
            break
        ratio = max(0.000001, min(1.0, (target / current_faces) * 0.995))
        modifier = collision.modifiers.new(
            name="collision_decimate_" + str(attempt + 1),
            type="DECIMATE",
        )
        modifier.decimate_type = "COLLAPSE"
        modifier.ratio = ratio
        modifier.use_collapse_triangulate = True
        bpy.context.view_layer.objects.active = collision
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        updated_faces = triangle_count(collision.data)
        attempts.append({
            "attempt": attempt + 1,
            "ratio": ratio,
            "beforeFaces": current_faces,
            "afterFaces": updated_faces,
        })
        if updated_faces >= current_faces:
            break
        current_faces = updated_faces
    return initial_faces, current_faces, attempts


def export_collision(file_path, collision):
    bpy.ops.object.select_all(action="DESELECT")
    collision.select_set(True)
    bpy.context.view_layer.objects.active = collision
    bpy.ops.export_scene.gltf(
        filepath=file_path,
        export_format="GLB",
        use_selection=True,
        export_materials="NONE",
        export_normals=False,
        export_tangents=False,
        export_texcoords=False,
        export_attributes=False,
        export_vertex_color="NONE",
        export_cameras=False,
        export_lights=False,
        export_animations=False,
        export_morph=False,
        export_skins=False,
        export_extras=False,
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
    "status": "failed",
    "source": {"path": input_path},
    "target": {"faces": target_faces},
    "output": {"path": output_path},
    "qa": {},
    "processing": {
        "tool": "blender_headless_decimate",
        "blenderVersion": bpy.app.version_string,
        "blenderVersionTuple": list(bpy.app.version),
    },
}

try:
    clear_scene()
    import_glb(input_path)
    source = scene_stats(input_path)
    collision = prepare_collision_mesh()
    joined_faces, simplified_faces, attempts = decimate_to_target(collision, target_faces)
    export_collision(output_path, collision)

    clear_scene()
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
    target_met = output["faces"] <= target_faces or source["faces"] <= target_faces
    material_free = output["materialSlots"] == 0
    nonempty = output["vertices"] > 0 and output["faces"] > 0
    bounds_preserved = maximum_bounds_delta <= bounds_tolerance

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
            "materialFree": material_free,
            "nonempty": nonempty,
            "boundsTolerance": bounds_tolerance,
            "maxAbsBoundsDelta": maximum_bounds_delta,
            "boundsDriftFraction": bounds_drift_fraction,
            "boundsToleranceFraction": bounds_tolerance_fraction,
            "boundsPreservedWithinTolerance": bounds_preserved,
        },
        "processing": {
            **report["processing"],
            "sourceFacesAfterJoinAndTriangulate": joined_faces,
            "facesBeforeExport": simplified_faces,
            "decimationAttempts": attempts,
        },
    })
    required_checks = {
        "targetMet": target_met,
        "materialFree": material_free,
        "nonempty": nonempty,
        "boundsPreservedWithinTolerance": bounds_preserved,
    }
    report["status"] = "passed" if all(required_checks.values()) else "failed"
    if report["status"] != "passed":
        failed_checks = [name for name, passed in required_checks.items() if not passed]
        report["error"] = "Collision GLB failed required QA checks: " + ", ".join(failed_checks)
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

function parseArgs(argv) {
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

function run() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    console.log(USAGE);
    return;
  }

  const inputPath = path.resolve(required(options, "input"));
  const outputPath = path.resolve(required(options, "output"));
  const reportPath = path.resolve(required(options, "report"));
  const targetFaces = Number(required(options, "target-faces"));
  const boundsToleranceFraction = Number(options["bounds-tolerance-fraction"] ?? 0.01);
  if (!Number.isSafeInteger(targetFaces) || targetFaces <= 0) {
    throw new Error(`--target-faces must be a positive integer; got ${options["target-faces"]}`);
  }
  if (!Number.isFinite(boundsToleranceFraction) || boundsToleranceFraction < 0) {
    throw new Error(
      `--bounds-tolerance-fraction must be a finite non-negative number; got ${options["bounds-tolerance-fraction"]}`,
    );
  }
  if (!fs.existsSync(inputPath) || !fs.statSync(inputPath).isFile()) {
    throw new Error(`Input GLB does not exist: ${inputPath}`);
  }
  if (!/\.glb$/i.test(inputPath) || !/\.glb$/i.test(outputPath)) {
    throw new Error("--input and --output must both use the .glb extension");
  }
  if (inputPath === outputPath) throw new Error("--output must differ from --input");

  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.mkdirSync(path.dirname(reportPath), { recursive: true });
  const blender = resolveBlender(options.blender);
  const temporaryReport = path.join(
    os.tmpdir(),
    `video2world-collision-${process.pid}-${crypto.randomBytes(6).toString("hex")}.json`,
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
        V2W_COLLISION_INPUT: inputPath,
        V2W_COLLISION_OUTPUT: outputPath,
        V2W_COLLISION_REPORT_TMP: temporaryReport,
        V2W_COLLISION_TARGET_FACES: String(targetFaces),
        V2W_COLLISION_BOUNDS_TOLERANCE_FRACTION: String(boundsToleranceFraction),
      },
    },
  );

  let report = {
    schemaVersion: 1,
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
  const requiredQaChecks = [
    "targetMet",
    "materialFree",
    "nonempty",
    "boundsPreservedWithinTolerance",
  ];
  const failedQaChecks = requiredQaChecks.filter((check) => report.qa?.[check] !== true);
  if (report.status === "passed" && failedQaChecks.length) {
    report.status = "failed";
    report.error = `Collision GLB failed required QA checks: ${failedQaChecks.join(", ")}`;
  }
  if (result.status !== 0) {
    report.status = "failed";
    report.error ||= `Blender exited with code ${result.status}`;
    report.blenderStdoutTail = commandTail(result.stdout);
    report.blenderStderrTail = commandTail(result.stderr);
  }
  writeJsonAtomic(reportPath, report);

  if (report.status !== "passed" || result.status !== 0) {
    throw new Error(`${report.error ?? "Collision GLB QA failed"}. Report: ${reportPath}`);
  }
  console.log(
    JSON.stringify(
      {
        status: report.status,
        source: report.source.path,
        output: report.output.path,
        report: reportPath,
        sourceFaces: report.source.faces,
        outputFaces: report.output.faces,
        targetFaces: report.target.faces,
      },
      null,
      2,
    ),
  );
}

try {
  run();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  console.error(USAGE);
  process.exitCode = 1;
}
