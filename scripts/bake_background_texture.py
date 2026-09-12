#!/usr/bin/env python3
"""Bake a real texture for the delivered background mesh from the calibrated frames.

The old delivery shipped a small textured `background.glb` rather than a dense
vertex-coloured surface. This produces that artefact from the reconstruction the
task already holds:

1. decimate the TSDF surface to a delivery budget (the same VTK path the export
   uses, so the delivery and the bake agree);
2. unwrap it with xatlas;
3. for every face, pick the calibrated frame that sees it best -- in front of the
   camera, inside the image, its depth agreeing with the reconstruction's own
   depth buffer, and its normal facing the camera;
4. rasterise the face into the atlas and sample that frame's pixels.

Frames are chosen per face rather than per texel, which keeps 50 frames x 100k
faces cheap while still sampling every texel bilinearly from a real image. The
report records the covered texel fraction, the frames used and the seconds spent.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from world_modeling.engine import PipelineError, Task, artifact_path, read_json, sha256, write_json  # noqa: E402

DEFAULT_FACE_BUDGET = 100_000
DEFAULT_TEXTURE_SIZE = 2048
DEFAULT_ABSOLUTE_TOLERANCE = 0.05
DEFAULT_RELATIVE_TOLERANCE = 0.05


def role_index(task: Task) -> dict:
    artifacts = read_json(task.directory / "artifacts.json").get("artifacts", {})
    resolved = {}
    for role, record in artifacts.items():
        try:
            resolved[role] = artifact_path(task, record)
        except (PipelineError, KeyError, OSError):
            continue
    return resolved


def load_frames(task: Task, index: dict, absolute_tolerance: float, relative_tolerance: float):
    """Camera, image and depth-buffer access for every selected frame."""
    import numpy as np
    from PIL import Image

    cameras = {str(frame["frame_id"]): frame for frame in read_json(index["cameras"]).get("frames", [])}
    depths = {str(frame["frame_id"]): frame for frame in read_json(index["scene_depth"]).get("frames", [])}
    views = []
    for frame_id, camera in sorted(cameras.items()):
        depth = depths.get(frame_id)
        if depth is None:
            continue
        with Image.open(task.directory / camera["image_path"]) as opened:
            image = np.asarray(opened.convert("RGB"), dtype=np.uint8)
        depth_map = np.load(task.directory / depth["depth_path"])
        views.append({
            "frame_id": frame_id, "image": image,
            "intrinsics": np.asarray(camera["intrinsics"], dtype=np.float64),
            "world_to_camera": np.asarray(camera["world_to_camera"], dtype=np.float64),
            "image_to_depth": np.asarray(depth.get("image_to_depth") or np.eye(3), dtype=np.float64),
            "depth": depth_map, "width": int(camera["width"]), "height": int(camera["height"]),
            "eye": -(np.asarray(camera["world_to_camera"], dtype=np.float64)[:3, :3].T
                     @ np.asarray(camera["world_to_camera"], dtype=np.float64)[:3, 3]),
            "absolute_tolerance": absolute_tolerance, "relative_tolerance": relative_tolerance,
        })
    if not views:
        raise PipelineError("no calibrated frame with both an image and a depth buffer")
    return views


def frame_score(view, points_world, normals):
    """Return (visible mask, depth error) for world points seen by one view."""
    import numpy as np

    camera = points_world @ view["world_to_camera"][:3, :3].T + view["world_to_camera"][:3, 3]
    depth = camera[:, 2]
    focal = view["intrinsics"]
    u = focal[0, 0] * camera[:, 0] / np.where(depth > 0, depth, np.nan) + focal[0, 2]
    v = focal[1, 1] * camera[:, 1] / np.where(depth > 0, depth, np.nan) + focal[1, 2]
    facing = np.einsum("ij,ij->i", normals, view["eye"] - points_world) > 0
    inside = (depth > 0) & (u >= 0) & (u < view["width"]) & (v >= 0) & (v < view["height"]) & facing
    mapped = np.stack([u, v, np.ones_like(u)], axis=1) @ view["image_to_depth"].T
    du = np.where(mapped[:, 2] != 0, mapped[:, 0] / np.where(mapped[:, 2] != 0, mapped[:, 2], 1.0), -1)
    dv = np.where(mapped[:, 2] != 0, mapped[:, 1] / np.where(mapped[:, 2] != 0, mapped[:, 2], 1.0), -1)
    height, width = view["depth"].shape
    sampleable = inside & (du >= 0) & (du < width) & (dv >= 0) & (dv < height)
    indices = np.flatnonzero(sampleable)
    error = np.full(len(points_world), np.inf)
    if len(indices):
        rows = np.clip(np.rint(dv[indices]).astype(int), 0, height - 1)
        columns = np.clip(np.rint(du[indices]).astype(int), 0, width - 1)
        buffer = view["depth"][rows, columns]
        tolerance = view["absolute_tolerance"] + view["relative_tolerance"] * np.abs(buffer)
        difference = np.abs(depth[indices] - buffer)
        agreed = np.flatnonzero(difference <= tolerance)
        error[indices[agreed]] = difference[agreed]
    return error


def sample_colours(view, points_world):
    """Bilinear colour samples for world points through one view."""
    import numpy as np

    camera = points_world @ view["world_to_camera"][:3, :3].T + view["world_to_camera"][:3, 3]
    depth = camera[:, 2]
    in_front = depth > 0
    safe_depth = np.where(in_front, depth, 1.0)
    intrinsics = view["intrinsics"]
    u = intrinsics[0, 0] * camera[:, 0] / safe_depth + intrinsics[0, 2]
    v = intrinsics[1, 1] * camera[:, 1] / safe_depth + intrinsics[1, 2]
    height, width = view["image"].shape[:2]
    u = np.clip(np.where(in_front, u, 0.0), 0, width - 1.001)
    v = np.clip(np.where(in_front, v, 0.0), 0, height - 1.001)
    x0, y0 = np.floor(u).astype(int), np.floor(v).astype(int)
    fx, fy = (u - x0)[:, None], (v - y0)[:, None]
    image = view["image"]
    top = image[y0, x0].astype(np.float32) * (1 - fx) + image[y0, x0 + 1].astype(np.float32) * fx
    bottom = image[y0 + 1, x0].astype(np.float32) * (1 - fx) + image[y0 + 1, x0 + 1].astype(np.float32) * fx
    blended = top * (1 - fy) + bottom * fy
    # Points behind the camera keep the background colour rather than sampling a
    # clamped edge pixel, which would smear the frame's border across the atlas.
    return np.where(in_front[:, None], np.rint(blended), 0.0).clip(0, 255).astype(np.uint8)


def unwrap(mesh, resolution: int):
    """xatlas parametrisation of a mesh, returning vertices, faces and uvs."""
    import numpy as np
    import xatlas

    atlas = xatlas.Atlas()
    atlas.add_mesh(np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.uint32))
    chart_options = xatlas.ChartOptions()
    pack_options = xatlas.PackOptions()
    pack_options.resolution = resolution
    pack_options.padding = 2
    atlas.generate(chart_options=chart_options, pack_options=pack_options)
    mapping, faces, uvs = atlas[0]
    vertices = np.asarray(mesh.vertices, dtype=np.float64)[mapping]
    return vertices, np.asarray(faces, dtype=np.int64), np.asarray(uvs, dtype=np.float64)


def rasterize_face(uv, resolution: int):
    """Texel centres covered by one UV triangle, with barycentric weights."""
    import numpy as np

    scaled = uv * resolution
    low = np.floor(scaled.min(axis=0)).astype(int)
    high = np.ceil(scaled.max(axis=0)).astype(int)
    low = np.maximum(low, 0)
    high = np.minimum(high, resolution - 1)
    if high[0] < low[0] or high[1] < low[1]:
        return None
    xs = np.arange(low[0], high[0] + 1) + 0.5
    ys = np.arange(low[1], high[1] + 1) + 0.5
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    a, b, c = scaled[0], scaled[1], scaled[2]
    denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    if abs(denominator) < 1e-12:
        return None
    w0 = ((b[1] - c[1]) * (grid_x - c[0]) + (c[0] - b[0]) * (grid_y - c[1])) / denominator
    w1 = ((c[1] - a[1]) * (grid_x - c[0]) + (a[0] - c[0]) * (grid_y - c[1])) / denominator
    w2 = 1.0 - w0 - w1
    inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
    if not inside.any():
        return None
    texels = np.stack([grid_x[inside], grid_y[inside]], axis=1).astype(np.int64)
    weights = np.stack([w0[inside], w1[inside], w2[inside]], axis=1)
    return texels, weights


def fill_uncovered(texture, covered, passes: int = 4):
    """Grow covered texels outward so texture filtering does not sample black."""
    import numpy as np

    filled = texture.astype(np.float32).copy()
    mask = covered.copy()
    for _ in range(passes):
        if mask.all():
            break
        weight = np.zeros_like(mask, dtype=np.float32)
        total = np.zeros(filled.shape, dtype=np.float32)
        for axis in (0, 1):
            for shift in (1, -1):
                rolled_values = np.roll(filled, shift, axis=axis)
                rolled_mask = np.roll(mask, shift, axis=axis).astype(np.float32)
                total += rolled_values * rolled_mask[:, :, None]
                weight += rolled_mask
        grow = (~mask) & (weight > 0)
        filled[grow] = (total[grow] / weight[grow][:, None])
        mask = mask | grow
    return filled.astype(np.uint8), mask


def bake(task: Task, index: dict, *, face_budget: int, texture_size: int, absolute_tolerance: float,
         relative_tolerance: float, max_frames: int) -> dict:
    import numpy as np
    import trimesh
    from PIL import Image

    started = time.perf_counter()
    views = load_frames(task, index, absolute_tolerance, relative_tolerance)
    if max_frames and len(views) > max_frames:
        step = len(views) / max_frames
        views = [views[int(index * step)] for index in range(max_frames)]

    with tempfile.TemporaryDirectory() as folder:
        intermediate = Path(folder) / "decimated.ply"
        import export_assets

        decimation = export_assets.deliverable_mesh(index["scene_tsdf_mesh"], intermediate, face_budget)
        mesh = trimesh.load(intermediate, force="mesh", process=False)
    vertices, faces, uvs = unwrap(mesh, texture_size)
    vertex_normals = np.asarray(trimesh.Trimesh(vertices=vertices, faces=faces, process=False).vertex_normals)

    texture = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)
    covered = np.zeros((texture_size, texture_size), dtype=bool)
    frame_hits: dict = {}
    skipped = 0
    for face_index, triangle in enumerate(faces):
        points = vertices[triangle]
        normal = vertex_normals[triangle].mean(axis=0)
        norm = float(np.linalg.norm(normal))
        if norm <= 0:
            skipped += 1
            continue
        normal = normal / norm
        centroid = points.mean(axis=0, keepdims=True)
        best, best_error = None, np.inf
        for view in views:
            error = frame_score(view, centroid, normal[None, :])[0]
            if error < best_error:
                best, best_error = view, error
        if best is None or not np.isfinite(best_error):
            skipped += 1
            continue
        raster = rasterize_face(uvs[triangle], texture_size)
        if raster is None:
            continue
        texels, weights = raster
        positions = weights @ points
        colours = sample_colours(best, positions)
        texture[texels[:, 0], texels[:, 1]] = colours
        covered[texels[:, 0], texels[:, 1]] = True
        frame_hits[best["frame_id"]] = frame_hits.get(best["frame_id"], 0) + 1

    if not covered.any():
        raise PipelineError("no face could be sampled from any calibrated frame")
    texture, filled_mask = fill_uncovered(texture, covered)
    return {"mesh": mesh, "vertices": vertices, "uvs": uvs, "faces": faces, "texture": texture, "covered": covered,
            "decimation": decimation, "frame_hits": frame_hits, "skipped_faces": skipped,
            "seconds": round(time.perf_counter() - started, 2), "views": len(views)}


def run(args) -> dict:
    import numpy as np
    import trimesh
    from PIL import Image

    task = Task(args.task_dir.resolve().parent.parent, args.task_dir.resolve().name)
    index = role_index(task)
    for role in ("scene_tsdf_mesh", "cameras", "scene_depth"):
        if role not in index:
            raise PipelineError(f"{role} is not published by this task")
    result = bake(task, index, face_budget=args.face_budget, texture_size=args.texture_size,
                  absolute_tolerance=args.absolute_tolerance, relative_tolerance=args.relative_tolerance,
                  max_frames=args.max_frames)
    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    texture_path = destination / "background_texture.png"
    Image.fromarray(result["texture"]).save(texture_path)
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(result["texture"]),
                                                   metallicFactor=0.0, roughnessFactor=1.0)
    baked = trimesh.Trimesh(vertices=result["vertices"], faces=result["faces"], process=False)
    baked.visual = trimesh.visual.texture.TextureVisuals(uv=result["uvs"][:, :2], material=material)
    glb_path = destination / args.name
    baked.export(glb_path)
    report = {
        "schema_version": "1.0", "kind": "video2world-modeling.baked_background",
        "status": "baked", "glb": glb_path.name, "texture": texture_path.name,
        "glb_sha256": sha256(glb_path), "texture_sha256": sha256(texture_path),
        "texture_size": args.texture_size, "faces": int(len(result["faces"])),
        "vertices": int(len(result["vertices"])), "decimation": result["decimation"],
        "calibrated_frames_used": result["views"], "faces_per_frame": dict(sorted(result["frame_hits"].items())),
        "faces_sampled": int(sum(result["frame_hits"].values())), "faces_skipped": result["skipped_faces"],
        "covered_texel_fraction": float(result["covered"].mean()),
        "parameters": {"face_budget": args.face_budget, "texture_size": args.texture_size,
                       "absolute_tolerance": args.absolute_tolerance, "relative_tolerance": args.relative_tolerance,
                       "max_frames": args.max_frames},
        "method": "per_face_best_calibrated_frame_with_depth_agreement_and_normal_facing",
        "seconds": result["seconds"],
    }
    write_json(destination / "baked_background.json", report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", default="background.glb")
    parser.add_argument("--face-budget", type=int, default=DEFAULT_FACE_BUDGET)
    parser.add_argument("--texture-size", type=int, default=DEFAULT_TEXTURE_SIZE)
    parser.add_argument("--absolute-tolerance", type=float, default=DEFAULT_ABSOLUTE_TOLERANCE)
    parser.add_argument("--relative-tolerance", type=float, default=DEFAULT_RELATIVE_TOLERANCE)
    parser.add_argument("--max-frames", type=int, default=0, help="use at most this many frames; 0 uses all")
    args = parser.parse_args(argv)
    if args.face_budget < 1000 or not 256 <= args.texture_size <= 8192:
        parser.error("face-budget must be at least 1000; texture-size between 256 and 8192")
    if args.absolute_tolerance <= 0 or args.relative_tolerance <= 0 or args.max_frames < 0:
        parser.error("tolerances must be positive and max-frames cannot be negative")
    try:
        report = run(args)
    except (PipelineError, OSError, ValueError, KeyError) as error:
        print(f"bake_background_texture failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
