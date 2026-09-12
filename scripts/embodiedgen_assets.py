#!/usr/bin/env python3
"""Textured object assets and collision proxies from EmbodiedGen v2.

The completion provider emits, per object, a mesh and a Gaussian splat. Those
two are what a delivery actually needs to look like an asset rather than a
point dump, and EmbodiedGen v2 is the toolchain that turns them into one:

1. render the object's own splat from five elevations to get appearance views;
2. ``MeshFixer`` simplifies the generated mesh and fills its holes;
3. ``TextureBaker`` unwraps the surface and bakes those views into a texture;
4. ``save_mesh_with_mtl`` writes the OBJ/MTL/texture triple and the GLB;
5. ``decompose_convex_mesh`` (CoACD) writes the collision proxy.

Everything is recorded with hashes, versions and parameters. The output role
names match ``mesh_postprocess``, so this runs as an alternative provider for
the same stage: bind it in a profile instead of the builtin geometry tools.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import re
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from world_modeling.engine import (  # noqa: E402
    PipelineError, Task, artifact_path, read_json, sha256, verify_artifact, write_json,
)

MIN_TEXTURE_SIZE = 256
MAX_TEXTURE_SIZE = 8192
ELEVATIONS = [60.0, 30.0, 0.0, -30.0, -60.0]


class AssetError(RuntimeError):
    pass


def stub_optional_model(module_name: str, class_name: str) -> None:
    """Keep EmbodiedGen's optional learned models out of the import path.

    Texturing consumes renders of the object's own splat, so the delighting and
    super-resolution checkpoints are never called and must not be required.
    """
    module = types.ModuleType(module_name)
    setattr(module, class_name, type(class_name, (), {}))
    sys.modules[module_name] = module


def load_embodiedgen(root: Path):
    stub_optional_model("embodied_gen.models.delight_model", "DelightingModel")
    stub_optional_model("embodied_gen.models.sr_model", "ImageRealESRGAN")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from embodied_gen.data.backproject_v3 import TextureBaker
        from embodied_gen.data.convex_decomposer import decompose_convex_mesh
        from embodied_gen.data.mesh_operator import MeshFixer
        from embodied_gen.data.utils import (
            CameraSetting, init_kal_camera, normalize_vertices_array, post_process_texture, save_mesh_with_mtl,
        )
        from embodied_gen.models.gs_model import load_gs_model
    except ImportError as error:
        raise AssetError(f"EmbodiedGen v2 is not importable from {root}: {error}") from error
    return types.SimpleNamespace(TextureBaker=TextureBaker, decompose_convex_mesh=decompose_convex_mesh,
                                 MeshFixer=MeshFixer, CameraSetting=CameraSetting, init_kal_camera=init_kal_camera,
                                 normalize_vertices_array=normalize_vertices_array,
                                 post_process_texture=post_process_texture, save_mesh_with_mtl=save_mesh_with_mtl,
                                 load_gs_model=load_gs_model)


def task_path(task_dir: Path, value, *, exists: bool = True) -> Path:
    candidate = (task_dir / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    if candidate != task_dir and task_dir not in candidate.parents:
        raise AssetError(f"path escapes task: {value}")
    if exists and not candidate.exists():
        raise AssetError(f"missing input: {value}")
    return candidate


def relative(task_dir: Path, path: Path) -> str:
    return str(path.resolve().relative_to(task_dir))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def object_records(task_dir: Path, artifact: dict) -> tuple[Path, list[dict]]:
    path = task_path(task_dir, artifact["path"])
    if artifact.get("sha256") != sha256(path):
        raise AssetError(f"artifact hash mismatch: {artifact['path']}")
    document = read_json(path)
    objects = document.get("objects")
    if not isinstance(objects, list) or not objects:
        raise AssetError(f"{artifact['path']} declares no objects")
    return path, objects


def asset_directory(root: Path, object_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", object_id or ""):
        raise AssetError(f"unusable object_id for an asset directory: {object_id!r}")
    return root / object_id


def clean_splat(source: Path, destination: Path) -> tuple[Path, int]:
    """Copy a splat without the rows EmbodiedGen's renderer cannot consume."""
    from world_modeling.gaussian_io import GaussianError, read_gaussian_rows
    from plyfile import PlyData, PlyElement

    try:
        cloud = read_gaussian_rows(source, error=AssetError)
    except GaussianError as error:
        raise AssetError(str(error)) from error
    if not cloud.dropped:
        return source, 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(cloud.rows, "vertex")], text=False).write(str(destination))
    return destination, cloud.dropped


def render_views(embodiedgen, splat: Path, *, images: int, resolution: int):
    """Render the object's own splat from the standard five-elevation orbit."""
    import cv2
    import numpy as np
    import torch
    from PIL import Image

    camera_params = embodiedgen.CameraSetting(
        num_images=images,
        elevation=ELEVATIONS,
        distance=4.5,
        resolution_hw=(resolution, resolution),
        fov=math.radians(30),
        device="cuda",
    )
    camera = embodiedgen.init_kal_camera(camera_params, flip_az=True)
    matrix_mv = camera.view_matrix()
    matrix_mv[:, :3, 3] = -matrix_mv[:, :3, 3]
    c2ws = torch.linalg.inv(matrix_mv)
    intrinsics = torch.tensor(camera_params.Ks, device="cuda")
    model = embodiedgen.load_gs_model(str(splat), pre_quat=[0.0, 0.0, 1.0, 0.0])
    frames = []
    for c2w in c2ws:
        result = model.render(c2w, Ks=intrinsics, image_width=resolution, image_height=resolution)
        rgba = cv2.cvtColor(result.rgba, cv2.COLOR_BGRA2RGBA)
        frames.append(np.asarray(Image.fromarray(rgba).convert("RGB")))
    return frames, camera_params


def load_mesh(path: Path):
    import trimesh

    loaded = trimesh.load(str(path), force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise AssetError(f"{path} contains no geometry")
        return loaded.dump(concatenate=True)
    if not isinstance(loaded, trimesh.Trimesh) or not len(loaded.faces):
        raise AssetError(f"{path} is not a triangle mesh")
    return loaded


def mesh_stats(mesh) -> dict:
    import numpy as np

    degenerate = np.asarray(mesh.area_faces) <= 1e-14
    return {"vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces)),
            "degenerate_triangles": int(np.count_nonzero(degenerate)),
            "watertight": bool(mesh.is_watertight), "winding_consistent": bool(mesh.is_winding_consistent),
            "extents": [float(value) for value in mesh.extents]}


def remove_degenerate_triangles(vertices, faces):
    import numpy as np
    import trimesh

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    degenerate = np.asarray(mesh.area_faces) <= 1e-14
    removed = int(np.count_nonzero(degenerate))
    if removed:
        mesh.update_faces(~degenerate)
        mesh.remove_unreferenced_vertices()
    if not len(mesh.faces):
        raise AssetError("MeshFixer output contains no usable triangles after sanitization")
    return np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.int32), removed


def verify_reported_paths(task_dir: Path, records: list, label: str) -> None:
    """Every published path must be task-relative and already exist.

    A path computed against the wrong root still looks plausible in JSON and
    only fails much later, in whichever consumer resolves it first, so it is
    refused here where the offending record is still in hand.
    """
    for record in records:
        values = [record.get(key) for key in ("mesh_path", "obj_path", "mtl_path", "collision_path")]
        values += [texture.get("path") for texture in record.get("textures") or []]
        values += [hull.get("path") for hull in record.get("hulls") or [] if isinstance(hull, dict)]
        for value in values:
            if value is None:
                continue
            if not isinstance(value, str) or Path(value).is_absolute() or not (task_dir / value).is_file():
                raise AssetError(f"{label}: reported path is not an existing task-relative artifact: {value!r}")


def hull_report(mesh) -> dict:
    import numpy as np
    import trimesh

    vertices = np.asarray(mesh.vertices)
    return {"finite": bool(len(vertices) and np.isfinite(vertices).all()),
            "watertight": bool(mesh.is_watertight), "winding_consistent": bool(mesh.is_winding_consistent),
            "positive_volume": bool(mesh.is_volume and mesh.volume > 0), "convex": bool(mesh.is_convex),
            "faces": int(len(mesh.faces)), "vertices": int(len(vertices))}


def decompose_collision(embodiedgen, *, task_dir: Path, object_id: str, obj_path: Path, collision_path: Path,
                        collision_dir: Path, args) -> tuple[list, dict]:
    """Decompose the textured mesh and refuse a hull set that is not all convex.

    CoACD is randomised: for an input that decomposes cleanly on one draw it can
    emit an open or slightly concave piece on another, so the decomposition is
    retried with fresh seeds and only a fully valid set is published.
    """
    import trimesh

    attempts = []
    for attempt in range(3):
        seed = args.seed + attempt
        embodiedgen.decompose_convex_mesh(str(obj_path), str(collision_path), threshold=0.05, max_convex_hull=32,
                                          preprocess_resolution=30, resolution=1000, mcts_nodes=20,
                                          mcts_iterations=100, mcts_max_depth=3, seed=seed, auto_scale=True)
        scene = trimesh.load(str(collision_path), force="scene")
        pieces, invalid, convexified = [], [], 0
        for geometry in scene.geometry.values():
            piece = trimesh.Trimesh(vertices=geometry.vertices, faces=geometry.faces, process=True, validate=True)
            trimesh.repair.fix_normals(piece, multibody=True)
            report = hull_report(piece)
            if report["finite"] and report["watertight"] and report["positive_volume"] and not report["convex"]:
                # EmbodiedGen's decomposition post-processes CoACD output and can
                # hand back a slightly concave surface. The published collider
                # must be convex, and the convex hull of the piece is convex by
                # construction, so it is taken and both shapes are recorded.
                tightened = piece.convex_hull
                tightened = trimesh.Trimesh(vertices=tightened.vertices, faces=tightened.faces, process=True,
                                            validate=True)
                trimesh.repair.fix_normals(tightened, multibody=True)
                tightened_report = hull_report(tightened)
                if all(tightened_report[key] for key in ("finite", "watertight", "winding_consistent",
                                                         "positive_volume", "convex")):
                    convexified += 1
                    report = {**tightened_report, "convexified_from": report}
                    piece = tightened
            if not all(report[key] for key in ("finite", "watertight", "winding_consistent", "positive_volume", "convex")):
                invalid.append(report)
                continue
            pieces.append(piece)
        attempts.append({"attempt": attempt, "seed": seed, "pieces": len(scene.geometry), "valid": len(pieces),
                         "convexified": convexified, "invalid": len(invalid),
                         "first_invalid": invalid[0] if invalid else None})
        if not pieces or invalid:
            continue
        hulls = []
        for index, piece in enumerate(pieces):
            path = collision_dir / f"hull_{index:03d}.obj"
            piece.export(path)
            hulls.append({"path": relative(task_dir, path), "sha256": sha256(path),
                          "faces": int(len(piece.faces)), "vertices": int(len(piece.vertices))})
        return hulls, {"method": "embodiedgen_decompose_convex_mesh", "seed": seed, "convex_parts": len(hulls),
                       "convexified_parts": convexified,
                       "vertices": int(sum(item["vertices"] for item in hulls)),
                       "faces": int(sum(item["faces"] for item in hulls)), "simplex_attempts": attempts}
    raise AssetError(f"{object_id}: convex decomposition never produced a fully valid hull set: {attempts}")


def texturize_object(embodiedgen, *, task_dir: Path, object_id: str, mesh_path: Path, splat_path: Path,
                     directory: Path, args, scratch: Path) -> dict:
    import numpy as np
    import trimesh

    started = time.perf_counter()
    directory.mkdir(parents=True, exist_ok=True)
    obj_path = directory / f"{object_id}.obj"
    glb_path = directory / f"{object_id}.glb"
    collision_path = directory / f"{object_id}_collision.obj"
    collision_dir = directory / "collision"
    collision_dir.mkdir(exist_ok=True)

    raw_mesh = load_mesh(mesh_path)
    before = mesh_stats(raw_mesh)
    usable_splat, dropped = clean_splat(splat_path, scratch / f"{object_id}_splat.ply")
    frames, camera_params = render_views(embodiedgen, usable_splat, images=args.num_images, resolution=args.resolution)

    vertices, scale, center = embodiedgen.normalize_vertices_array(raw_mesh.vertices)
    x_rot = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    z_rot = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    vertices = (vertices @ x_rot @ z_rot).astype(np.float32)
    faces = raw_mesh.faces.astype(np.int32)

    fixer = embodiedgen.MeshFixer(vertices, faces, "cuda")
    fixer.simplify(ratio=args.simplify_ratio)
    simplified = {"vertices": int(len(fixer.vertices_np)), "faces": int(len(fixer.faces_np))}
    fixer.fill_holes(max_hole_size=0.04, max_hole_nbe=int(250 * math.sqrt(1 - args.simplify_ratio)),
                     resolution=args.fixer_resolution, num_views=args.fixer_views, norm_mesh_ratio=0.5)
    vertices, faces, degenerate_removed = remove_degenerate_triangles(fixer.vertices_np, fixer.faces_np)

    vertices, faces, uvs = embodiedgen.TextureBaker.parametrize_mesh(vertices, faces)
    baker = embodiedgen.TextureBaker(vertices, faces, uvs, camera_params, device="cuda")
    texture = embodiedgen.post_process_texture(
        baker.bake_texture(images=frames, texture_size=args.texture_size, mode="fast"))

    vertices = vertices @ np.linalg.inv(z_rot) @ np.linalg.inv(x_rot)
    vertices = vertices / scale + center
    textured = embodiedgen.save_mesh_with_mtl(vertices, faces, uvs, texture, str(obj_path))
    textured.export(glb_path)
    after = mesh_stats(load_mesh(obj_path))
    if after["degenerate_triangles"] != 0:
        raise AssetError(f"{object_id}: final textured mesh contains degenerate triangles")

    hulls, collision = decompose_collision(embodiedgen, task_dir=task_dir, object_id=object_id,
                                           obj_path=obj_path, collision_path=collision_path,
                                           collision_dir=collision_dir, args=args)

    files = {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
             for path in sorted(directory.iterdir()) if path.is_file()}
    # EmbodiedGen names the sidecars itself (material.mtl / material_0.png), so
    # discover them instead of assuming the object id appears in every filename.
    textures = [name for name in files if name.lower().endswith((".png", ".jpg", ".jpeg"))]
    materials = [name for name in files if name.lower().endswith(".mtl")]
    if not textures or not materials:
        raise AssetError(f"{object_id}: textured asset is missing its material or texture sidecar")
    return {
        "object_id": object_id, "mesh_path": relative(task_dir, glb_path),
        "obj_path": relative(task_dir, obj_path),
        "texture_paths": [relative(task_dir, directory / name) for name in textures],
        "texture_sha256": {relative(task_dir, directory / name): files[name]["sha256"] for name in textures},
        "mtl_path": relative(task_dir, directory / materials[0]),
        "collision_path": relative(task_dir, collision_path),
        "mesh_sha256": sha256(glb_path), "obj_sha256": sha256(obj_path),
        "collision_sha256": sha256(collision_path), "hulls": hulls,
        "mesh_before": before, "mesh_after": after, "simplified": simplified,
        "degenerate_triangles_removed": degenerate_removed,
        "non_finite_splats_dropped": dropped, "rendered_views": len(frames),
        "collision": collision, "asset_files": files, "seconds": round(time.perf_counter() - started, 2),
    }


def run(args) -> dict:
    task_dir = args.task_dir.resolve(strict=True)
    inputs_path = task_path(task_dir, args.inputs)
    outputs_path = task_path(task_dir, args.outputs, exists=False)
    payload = read_json(inputs_path)
    inputs = payload.get("inputs", {})
    if payload.get("module", "mesh_postprocess") != "mesh_postprocess":
        raise AssetError("input envelope names a different module")
    if not isinstance(inputs, dict) or "completed_object_meshes" not in inputs:
        raise AssetError("input envelope requires completed_object_meshes")

    embodiedgen_root = args.embodiedgen_root.resolve()
    if not (embodiedgen_root / "embodied_gen/data/backproject_v3.py").is_file():
        raise AssetError(f"EmbodiedGen v2 source not found under {embodiedgen_root}")
    embodiedgen = load_embodiedgen(embodiedgen_root)

    mesh_manifest_path, objects = object_records(task_dir, inputs["completed_object_meshes"])
    candidates_path = None
    if inputs.get("completion_candidates"):
        candidates_path = task_path(task_dir, inputs["completion_candidates"]["path"])

    stage_dir = task_dir / "stages/mesh_postprocess"
    output_root = Path(args.output_dir) if args.output_dir else stage_dir
    run_root = output_root / f"embodiedgen-{time.strftime('%Y%m%dT%H%M%S')}"
    scratch = run_root / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    assets_root = run_root / "assets"

    visual_records, texture_records, collision_records, object_reports, skipped = [], [], [], [], []
    for item in objects:
        object_id = item.get("object_id")
        mesh_value = item.get("mesh_path") or item.get("mesh") or item.get("path")
        splat_value = item.get("visual_ply_path") or item.get("splat_path")
        try:
            if not mesh_value or not splat_value:
                raise AssetError("completed object needs both a mesh and a gaussian splat")
            mesh_path = task_path(task_dir, mesh_value)
            splat_path = task_path(task_dir, splat_value)
            if item.get("mesh_sha256") and item["mesh_sha256"] != sha256(mesh_path):
                raise AssetError("completed mesh changed after publication")
            if item.get("visual_ply_sha256") and item["visual_ply_sha256"] != sha256(splat_path):
                raise AssetError("completed splat changed after publication")
            directory = asset_directory(assets_root, object_id)
            report = texturize_object(embodiedgen, task_dir=task_dir, object_id=object_id, mesh_path=mesh_path,
                                      splat_path=splat_path, directory=directory, args=args, scratch=scratch)
            common = {key: item[key] for key in ("object_id", "coordinate_frame", "unit", "room_alignment",
                                                 "evidence", "object_version", "source_frame_ids")
                      if key in item}
            visual_records.append({**common, "mesh_path": relative(task_dir, directory / f"{object_id}.glb"),
                                   "sha256": report["mesh_sha256"], "evidence": "generated"})
            texture_records.append({**common, "textures": [{"path": path, "sha256": report["texture_sha256"][path]}
                                                            for path in report["texture_paths"]],
                                    "mtl_path": report["mtl_path"], "obj_path": report["obj_path"]})
            collision_records.append({**common, "mesh_path": report["collision_path"],
                                      "sha256": report["collision_sha256"], "hulls": report["hulls"],
                                      "convex_parts": report["collision"]["convex_parts"],
                                      "collision_eligible": True})
            object_reports.append({**common, **report, "source_mesh": relative(task_dir, mesh_path),
                                   "source_splat": relative(task_dir, splat_path),
                                   "asset_dir": relative(task_dir, directory),
                                   "provider": "embodiedgen_v2_texture_baker"})
        except (AssetError, ImportError, OSError, ValueError, KeyError, RuntimeError) as error:
            skipped.append({"object_id": object_id, "reason": str(error)})
            print(f"{object_id}: embodiedgen texturing failed: {error}", file=sys.stderr, flush=True)
    for records, label in ((visual_records, "repaired_visual_meshes"), (texture_records, "uv_textures"),
                          (collision_records, "coacd_collision_meshes")):
        verify_reported_paths(task_dir, records, label)
    if not object_reports:
        raise AssetError("no object produced an EmbodiedGen asset: " + "; ".join(
            f"{item['object_id']}: {item['reason']}" for item in skipped))

    versions = {}
    for name in ("kaolin", "nvdiffrast", "coacd", "pymeshfix", "trimesh", "numpy", "torch", "Pillow", "opencv-python"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    manifests = {
        "repaired_visual_meshes": {"objects": visual_records},
        "uv_textures": {"objects": texture_records},
        "coacd_collision_meshes": {"objects": collision_records},
        "mesh_qa_report": {
            "status": "passed",
            "provider": "embodiedgen_v2_texture_baker",
            "learned_3d_fixer_invoked": False,
            "objects": object_reports,
            "skipped_objects": skipped,
            "input_manifests": {relative(task_dir, path): sha256(path)
                                for path in (mesh_manifest_path, candidates_path) if path is not None},
            "embodiedgen": {"source_root": str(embodiedgen_root), "versions": versions},
            "parameters": {key: value for key, value in vars(args).items() if not isinstance(value, Path)},
            "placement_and_physics_validated": False,
        },
    }
    outputs = {}
    for role, manifest in manifests.items():
        path = run_root / f"{role}.json"
        write_json(path, {"schema_version": "1.0", "kind": f"video2world-modeling.{role}", **manifest})
        outputs[role] = {"path": relative(task_dir, path), "evidence": "derived", "media_hint": "application/json",
                         "status": "candidate", "collision_eligible": role == "coacd_collision_meshes"}
    result = {"outputs": outputs}
    write_json(outputs_path, result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--embodiedgen-root", type=Path, required=True,
                        help="checkout containing embodied_gen/ (EmbodiedGen v2)")
    parser.add_argument("--output-dir", type=Path, help="where per-object assets are written; defaults to the stage dir")
    parser.add_argument("--num-images", type=int, default=40, help="splat render views; must divide by five elevations")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--texture-size", type=int, default=1024)
    parser.add_argument("--simplify-ratio", type=float, default=0.85)
    parser.add_argument("--fixer-views", type=int, default=48)
    parser.add_argument("--fixer-resolution", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)
    if args.num_images % len(ELEVATIONS) or args.num_images < len(ELEVATIONS):
        parser.error("--num-images must be a positive multiple of the five render elevations")
    if not 0 < args.simplify_ratio < 1:
        parser.error("--simplify-ratio must be between 0 and 1")
    if not MIN_TEXTURE_SIZE <= args.texture_size <= MAX_TEXTURE_SIZE:
        parser.error(f"--texture-size must be between {MIN_TEXTURE_SIZE} and {MAX_TEXTURE_SIZE}")
    try:
        result = run(args)
    except (AssetError, ImportError, OSError, ValueError, KeyError, PipelineError) as error:
        print(f"embodiedgen_assets failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
