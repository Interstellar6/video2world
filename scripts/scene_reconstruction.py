#!/usr/bin/env python3
"""Calibrated COLMAP -> camera-conditioned DA3, fresh PGSR, and observed TSDF."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


class ReconstructionError(RuntimeError):
    pass


def task_path(task_dir: Path, value: str | Path, *, exists: bool = True) -> Path:
    path = (task_dir / value).resolve()
    if task_dir.resolve() not in path.parents:
        raise ReconstructionError(f"path must be beneath task directory: {value}")
    if exists and not path.exists():
        raise ReconstructionError(f"missing task-local input: {value}")
    return path


def relative(task_dir: Path, path: Path) -> str:
    return path.resolve().relative_to(task_dir.resolve()).as_posix()


def read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ReconstructionError(f"expected a JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_colmap_reader(da3_source: Path):
    reader_path = da3_source / "depth_anything_3/utils/read_write_model.py"
    if not reader_path.is_file():
        raise ReconstructionError(f"missing official COLMAP reader: {reader_path}")
    spec = importlib.util.spec_from_file_location("world_modeling_colmap_io", reader_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def camera_intrinsics(camera):
    import numpy as np

    if camera.model == "PINHOLE":
        fx, fy, cx, cy = camera.params
    elif camera.model == "SIMPLE_PINHOLE":
        fx, cx, cy = camera.params
        fy = fx
    else:
        raise ReconstructionError(f"camera {camera.id}: undistorted PINHOLE or SIMPLE_PINHOLE required, got {camera.model}")
    if not np.isfinite([fx, fy, cx, cy]).all() or min(fx, fy) <= 0:
        raise ReconstructionError(f"camera {camera.id}: invalid focal length or principal point")
    if abs(cx - camera.width / 2) > 1 or abs(cy - camera.height / 2) > 1:
        raise ReconstructionError(f"camera {camera.id}: current PGSR backend requires a centered principal point")
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def select_indices(count: int, maximum: int) -> list[int]:
    if count < 3:
        raise ReconstructionError("at least three calibrated views are required for DA3 scale alignment")
    if maximum < 0 or maximum in {1, 2}:
        raise ReconstructionError("max_frames must be zero (all) or at least three")
    if not maximum or maximum >= count:
        return list(range(count))
    return [round(index * (count - 1) / (maximum - 1)) for index in range(maximum)]


def camera_batches(count: int, batch_size: int) -> list[list[int]]:
    if count < 3 or batch_size < 3:
        raise ReconstructionError("camera batches require at least three views")
    if count <= batch_size:
        return [list(range(count))]
    # Shared end views give every batch the same long-baseline camera anchors.
    interior = list(range(1, count - 1))
    return [[0, count - 1, *interior[start:start + batch_size - 2]] for start in range(0, len(interior), batch_size - 2)]


def load_calibrated_scene(task_dir: Path, source: Path, reader, max_frames: int):
    import numpy as np
    from PIL import Image

    sparse = source / "sparse/0"
    if not sparse.is_dir() or not (source / "images").is_dir():
        raise ReconstructionError("source_media must contain images/ and sparse/0/ COLMAP reconstruction")
    extension = ".bin" if (sparse / "cameras.bin").is_file() else ".txt"
    for name in ("cameras", "images", "points3D"):
        task_path(task_dir, sparse / (name + extension))
    cameras, images, points = reader.read_model(str(sparse), ext=extension)
    ordered = sorted(images.values(), key=lambda item: item.name)
    selected = [ordered[index] for index in select_indices(len(ordered), max_frames)]
    records = []
    seen_ids = set()
    for image in selected:
        path = task_path(task_dir, source / "images" / image.name)
        if (source / "images").resolve() not in path.parents:
            raise ReconstructionError(f"COLMAP image escapes images/: {image.name}")
        camera = cameras[image.camera_id]
        intrinsics = camera_intrinsics(camera)
        with Image.open(path) as rgb:
            if rgb.size != (camera.width, camera.height):
                raise ReconstructionError(f"{image.name}: RGB dimensions {rgb.size} differ from calibration {(camera.width, camera.height)}")
        frame_id = Path(image.name).stem
        if frame_id in seen_ids:
            raise ReconstructionError(f"duplicate frame_id from COLMAP filenames: {frame_id}")
        seen_ids.add(frame_id)
        rotation = reader.qvec2rotmat(image.qvec)
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5):
            raise ReconstructionError(f"{frame_id}: camera rotation is not rigid")
        extrinsic = np.eye(4, dtype=np.float64)
        extrinsic[:3, :3], extrinsic[:3, 3] = rotation, image.tvec
        if not np.isfinite(extrinsic).all():
            raise ReconstructionError(f"{frame_id}: non-finite camera transform")
        records.append({
            "frame_id": frame_id, "colmap_image_id": int(image.id), "image_path": relative(task_dir, path),
            "width": int(camera.width), "height": int(camera.height),
            "intrinsics": intrinsics.tolist(), "world_to_camera": extrinsic.tolist(),
        })
    centers = np.array([np.linalg.inv(item["world_to_camera"])[:3, 3] for item in records])
    if np.linalg.norm(centers[0] - centers[-1]) < 1e-6:
        raise ReconstructionError("first and last selected camera centers coincide; DA3 scale anchors are degenerate")
    if not points:
        raise ReconstructionError("COLMAP points3D is empty; PGSR requires observed initialization points")
    return cameras, selected, points, records


def stage_pgsr_dataset(task_dir: Path, directory: Path, reader, cameras, selected, points, records) -> None:
    from PIL import Image

    (directory / "images").mkdir(parents=True)
    (directory / "sparse").mkdir()
    staged_images = {}
    staged_cameras = {}
    for image, record in zip(selected, records):
        name = f"{record['frame_id']}.png"
        with Image.open(task_path(task_dir, record["image_path"])) as rgb:
            rgb.convert("RGB").save(directory / "images" / name)
        camera = cameras[image.camera_id]
        matrix = record["intrinsics"]
        staged_cameras[camera.id] = reader.Camera(camera.id, "PINHOLE", camera.width, camera.height, [matrix[0][0], matrix[1][1], matrix[0][2], matrix[1][2]])
        staged_images[image.id] = image._replace(name=name)
    reader.write_model(staged_cameras, staged_images, points, str(directory / "sparse"), ext=".bin")


def build_depth_prior(args, task_dir: Path, depth_records: list[dict], output: Path) -> dict:
    import numpy as np
    import open3d as o3d
    from PIL import Image

    point_batches, color_batches = [], []
    for record in depth_records:
        depth = np.load(task_path(task_dir, record["depth_path"]), allow_pickle=False)
        confidence = np.load(task_path(task_dir, record["confidence_path"]), allow_pickle=False)
        color = np.asarray(Image.open(task_path(task_dir, record["rgb_path"])).convert("RGB"))
        valid = np.isfinite(depth) & (depth > 0) & (depth < args.depth_trunc) & np.isfinite(confidence)
        if valid.any():
            valid &= confidence >= np.percentile(confidence[valid], args.confidence_percentile)
        if "sky_path" in record:
            valid &= np.load(task_path(task_dir, record["sky_path"]), allow_pickle=False) < 0.5
        sampled = np.zeros_like(valid)
        sampled[::2, ::2] = valid[::2, ::2]
        y, x = np.nonzero(sampled)
        rays = np.column_stack((x, y, np.ones(len(x)))) @ np.linalg.inv(record["intrinsics"]).T
        camera_points = rays * depth[y, x, None]
        camera_to_world = np.linalg.inv(record["world_to_camera"])
        point_batches.append(camera_points @ camera_to_world[:3, :3].T + camera_to_world[:3, 3])
        color_batches.append(color[y, x].astype(np.float64) / 255.0)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.concatenate(point_batches))
    cloud.colors = o3d.utility.Vector3dVector(np.concatenate(color_batches))
    before = len(cloud.points)
    cloud = cloud.voxel_down_sample(args.pgsr_prior_voxel_size)
    coordinates = np.asarray(cloud.points)
    if not len(coordinates):
        raise ReconstructionError("confident camera-conditioned DA3 depth produced an empty PGSR point prior")
    order = np.lexsort((coordinates[:, 2], coordinates[:, 1], coordinates[:, 0]))
    if len(order) > args.pgsr_prior_max_points:
        order = order[np.linspace(0, len(order) - 1, args.pgsr_prior_max_points, dtype=int)]
    cloud = cloud.select_by_index(order.tolist())
    cloud.normals = o3d.utility.Vector3dVector(np.zeros((len(cloud.points), 3)))
    if not o3d.io.write_point_cloud(str(output), cloud):
        raise ReconstructionError(f"could not write PGSR observed point prior: {output}")
    return {
        "method": "backproject_camera_conditioned_da3", "input_samples": before, "points": len(cloud.points),
        "voxel_size": args.pgsr_prior_voxel_size, "point_limit": args.pgsr_prior_max_points,
        "confidence_percentile": args.confidence_percentile, "depth_trunc": args.depth_trunc,
        "semantic_sky_masks_available": all("sky_path" in record for record in depth_records),
        "coordinate_frame": "colmap_world", "units": "colmap_reconstruction", "path": relative(task_dir, output),
    }


def phase_options(args) -> dict:
    names = (
        "da3_source", "da3_model", "da3_python_path", "da3_resolution", "da3_batch_size", "max_frames",
        "tsdf_voxel_size", "tsdf_sdf_trunc", "depth_trunc", "confidence_percentile",
    )
    return {name: str(getattr(args, name)) if isinstance(getattr(args, name), Path) else getattr(args, name) for name in names}


def previous_provider_arguments(command: list[str]):
    matches = [index for index, value in enumerate(command) if Path(value).name == Path(__file__).name]
    if len(matches) != 1:
        raise ReconstructionError("failed execution does not identify this reconstruction adapter")
    try:
        return parser().parse_args(command[matches[0] + 1:])
    except SystemExit as error:
        raise ReconstructionError("failed execution has invalid reconstruction arguments") from error


def phase_files(task_dir: Path, run_dir: Path, records: list[dict]) -> list[Path]:
    required = [run_dir / name for name in ("frames.json", "cameras.json", "depth.json", "scene_tsdf.glb")]
    for record in records:
        required.extend(task_path(task_dir, record[name]) for name in ("depth_path", "confidence_path", "rgb_path"))
        if "sky_path" in record:
            required.append(task_path(task_dir, record["sky_path"]))
    return required


def save_phase_cache(args, task_dir: Path, run_dir: Path, artifact: dict, records: list[dict], *, adopted: bool = False) -> None:
    binding = {"source_media": artifact, "options": phase_options(args)}
    write_json(run_dir / "phase-cache.json", {
        "schema_version": "1.0", **binding,
        "source_options_sha256": hashlib.sha256(json.dumps(binding, sort_keys=True).encode("utf-8")).hexdigest(),
        "adopted_from_failed_execution_receipt": adopted,
        "files": [{"path": relative(task_dir, path), "sha256": sha256(path), "size_bytes": path.stat().st_size} for path in phase_files(task_dir, run_dir, records)],
    })


def restore_depth_phase(args, task_dir: Path, output_dir: Path, artifact: dict, records: list[dict]):
    import numpy as np
    from PIL import Image

    cached = task_path(task_dir, args.resume_run_dir)
    if cached.parent != output_dir:
        raise ReconstructionError("resume-run-dir must belong to this scene_reconstruction stage")
    cache_path = cached / "phase-cache.json"
    depth_records = read_json(cached / "depth.json")["frames"]
    if read_json(cached / "cameras.json")["frames"] != records:
        raise ReconstructionError("cached cameras/frames differ from the current calibrated source")
    if [item["frame_id"] for item in depth_records] != [item["frame_id"] for item in records]:
        raise ReconstructionError("cached depth frame identities differ from current input")
    if cache_path.exists():
        cache = read_json(cache_path)
        if cache["source_media"] != artifact or cache["options"] != phase_options(args):
            raise ReconstructionError("phase cache source/options fingerprint mismatch")
        binding = {"source_media": artifact, "options": phase_options(args)}
        if cache.get("source_options_sha256") != hashlib.sha256(json.dumps(binding, sort_keys=True).encode("utf-8")).hexdigest():
            raise ReconstructionError("phase cache source/options digest mismatch")
        listed = {item["path"]: item for item in cache["files"]}
        for path in phase_files(task_dir, cached, depth_records):
            recorded = listed.get(relative(task_dir, path), {})
            if recorded.get("sha256") != sha256(path) or recorded.get("size_bytes") != path.stat().st_size:
                raise ReconstructionError(f"phase cache hash mismatch: {path}")
    else:
        # The first failed PGSR attempt already has engine-bound source/options and per-depth hashes.
        previous = read_json(output_dir / "stage.json")
        if previous.get("status") != "failed" or previous.get("inputs", {}).get("source_media") != artifact:
            raise ReconstructionError("cannot adopt an unbound failed-run phase cache")
        previous_args = previous_provider_arguments(previous["provider"]["command"])
        if phase_options(previous_args) != phase_options(args):
            raise ReconstructionError("previous failed execution used different depth/TSDF options")
    for record, original in zip(depth_records, records):
        for key, digest_key in (("depth_path", "depth_sha256"), ("confidence_path", "confidence_sha256")):
            if sha256(task_path(task_dir, record[key])) != record[digest_key]:
                raise ReconstructionError(f"cached {key} hash mismatch for {record['frame_id']}")
        depth = np.load(task_path(task_dir, record["depth_path"]), allow_pickle=False)
        confidence = np.load(task_path(task_dir, record["confidence_path"]), allow_pickle=False)
        with Image.open(task_path(task_dir, record["rgb_path"])) as rgb:
            if depth.shape != (record["height"], record["width"]) or confidence.shape != depth.shape or rgb.size != (record["width"], record["height"]):
                raise ReconstructionError("cached DA3 raster dimensions are inconsistent")
        if not np.allclose(record["world_to_camera"], original["world_to_camera"]) or not np.allclose(np.asarray(record["image_to_depth"]) @ original["intrinsics"], record["intrinsics"]):
            raise ReconstructionError("cached DA3 camera mapping is inconsistent")
    if not cache_path.exists():
        save_phase_cache(args, task_dir, cached, artifact, depth_records, adopted=True)
    import trimesh

    mesh = trimesh.load(cached / "scene_tsdf.glb", force="mesh", process=False)
    if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ReconstructionError("cached TSDF is not a finite nonempty mesh")
    return depth_records, cached / "scene_tsdf.glb", {
        "reused_fresh_phase": relative(task_dir, cached), "phase_cache_sha256": sha256(cache_path),
        "source_and_options_verified": True,
    }


def save_depth_batch(task_dir: Path, directory: Path, records: list[dict], prediction) -> list[dict]:
    import numpy as np
    from PIL import Image

    depth = np.asarray(prediction.depth)
    confidence = np.asarray(prediction.conf)
    intrinsics = np.asarray(prediction.intrinsics)
    extrinsics = np.asarray(prediction.extrinsics)
    colors = np.asarray(prediction.processed_images)
    if depth.ndim != 3 or len(depth) != len(records) or confidence.shape != depth.shape:
        raise ReconstructionError("DA3 must return one depth raster and confidence raster per input image")
    if intrinsics.shape != (len(records), 3, 3) or extrinsics.shape not in {(len(records), 3, 4), (len(records), 4, 4)}:
        raise ReconstructionError("DA3 returned invalid aligned camera dimensions")
    if colors.shape != (*depth.shape, 3):
        raise ReconstructionError("DA3 processed RGB dimensions do not match the depth raster")
    outputs = []
    directory.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(records):
        expected_ext = np.asarray(record["world_to_camera"])
        if not np.allclose(extrinsics[index, :3], expected_ext[:3], rtol=1e-5, atol=1e-5):
            raise ReconstructionError(f"{record['frame_id']}: DA3 output is not aligned to supplied COLMAP extrinsics")
        if not np.isfinite(intrinsics[index]).all() or min(intrinsics[index, 0, 0], intrinsics[index, 1, 1]) <= 0:
            raise ReconstructionError(f"{record['frame_id']}: DA3 returned invalid raster intrinsics")
        valid = np.isfinite(depth[index]) & (depth[index] > 0) & np.isfinite(confidence[index])
        if not valid.any():
            raise ReconstructionError(f"{record['frame_id']}: DA3 returned no valid positive depth")
        stem = record["frame_id"]
        paths = {"depth_path": directory / f"{stem}.depth.npy", "confidence_path": directory / f"{stem}.confidence.npy", "rgb_path": directory / f"{stem}.rgb.png"}
        np.save(paths["depth_path"], np.asarray(depth[index], dtype=np.float32), allow_pickle=False)
        np.save(paths["confidence_path"], np.asarray(confidence[index], dtype=np.float32), allow_pickle=False)
        Image.fromarray(colors[index].astype(np.uint8)).save(paths["rgb_path"])
        height, width = depth[index].shape
        outputs.append({
            "frame_id": stem, **{name: relative(task_dir, path) for name, path in paths.items()},
            "width": width, "height": height, "intrinsics": intrinsics[index].tolist(),
            "world_to_camera": expected_ext.tolist(),
            "image_to_depth": (intrinsics[index] @ np.linalg.inv(record["intrinsics"])).tolist(),
            "valid_depth_fraction": float(valid.mean()),
            "depth_sha256": sha256(paths["depth_path"]), "confidence_sha256": sha256(paths["confidence_path"]),
        })
    return outputs


def infer_depth(args, task_dir: Path, run_dir: Path, records: list[dict]) -> tuple[list[dict], dict]:
    import numpy as np
    import torch
    from depth_anything_3.api import DepthAnything3

    if not torch.cuda.is_available():
        raise ReconstructionError("DA3 reconstruction requires an available CUDA GPU")
    model = DepthAnything3.from_pretrained(str(args.da3_model), local_files_only=True).to("cuda").eval()
    outputs = {}
    batches = camera_batches(len(records), args.da3_batch_size)
    for number, indices in enumerate(batches):
        batch = [records[index] for index in indices]
        print(f"DA3 batch {number + 1}/{len(batches)}: {[item['frame_id'] for item in batch]}", flush=True)
        prediction = model.inference(
            image=[str(task_path(task_dir, item["image_path"])) for item in batch],
            extrinsics=np.array([item["world_to_camera"] for item in batch], dtype=np.float32),
            intrinsics=np.array([item["intrinsics"] for item in batch], dtype=np.float32),
            align_to_input_ext_scale=True, infer_gs=False, process_res=args.da3_resolution,
            process_res_method="upper_bound_resize", export_dir=None,
        )
        new_indices = [index for index, record in enumerate(batch) if record["frame_id"] not in outputs]
        from types import SimpleNamespace

        selected = SimpleNamespace(**{name: np.asarray(getattr(prediction, name))[new_indices] for name in (
            "depth", "conf", "intrinsics", "extrinsics", "processed_images",
        )})
        written = save_depth_batch(task_dir, run_dir / "depth", [batch[index] for index in new_indices], selected)
        outputs.update({item["frame_id"]: item for item in written})
        del prediction, selected
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return [outputs[item["frame_id"]] for item in records], {"batches": batches, "camera_conditioned": True, "align_to_input_ext_scale": True}


def fuse_tsdf(args, task_dir: Path, run_dir: Path, depth_records: list[dict]) -> tuple[Path, dict]:
    import numpy as np
    import open3d as o3d
    from PIL import Image
    import trimesh

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.tsdf_voxel_size, sdf_trunc=args.tsdf_sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    fractions = []
    for record in depth_records:
        depth = np.load(task_path(task_dir, record["depth_path"]), allow_pickle=False).astype(np.float32)
        confidence = np.load(task_path(task_dir, record["confidence_path"]), allow_pickle=False)
        valid = np.isfinite(depth) & (depth > 0) & (depth < args.depth_trunc) & np.isfinite(confidence)
        if valid.any():
            valid &= confidence >= np.percentile(confidence[valid], args.confidence_percentile)
        depth[~valid] = 0
        fractions.append(float(valid.mean()))
        color = np.array(Image.open(task_path(task_dir, record["rgb_path"])).convert("RGB"), dtype=np.uint8)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color), o3d.geometry.Image(np.ascontiguousarray(depth)),
            depth_scale=1.0, depth_trunc=args.depth_trunc, convert_rgb_to_intensity=False,
        )
        matrix = np.asarray(record["intrinsics"])
        camera = o3d.camera.PinholeCameraIntrinsic(record["width"], record["height"], matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2])
        volume.integrate(rgbd, camera, np.asarray(record["world_to_camera"]))
    mesh = volume.extract_triangle_mesh()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    if not len(faces) or not np.isfinite(vertices).all():
        raise ReconstructionError("TSDF produced no finite triangle mesh; inspect depth scale, confidence and camera calibration")
    exported = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=(np.asarray(mesh.vertex_colors) * 255).astype(np.uint8), process=False)
    output = run_dir / "scene_tsdf.glb"
    exported.export(output)
    return output, {
        "vertices": len(vertices), "faces": len(faces), "bounds": exported.bounds.tolist(),
        "watertight": bool(exported.is_watertight), "per_frame_integrated_fraction": fractions,
        "voxel_size": args.tsdf_voxel_size, "sdf_trunc": args.tsdf_sdf_trunc, "depth_trunc": args.depth_trunc,
        "units": "colmap_reconstruction", "observation_only": True,
    }


def pgsr_environment(args, run_dir: Path) -> dict[str, str]:
    source = args.pgsr_source.resolve()
    entries = [str(source), str(source / "submodules/diff-plane-rasterization"), str(source / "submodules/simple-knn"), *args.pgsr_python_path]
    environment = os.environ.copy()
    environment.update(PYTHONPATH=os.pathsep.join(entries), PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(run_dir / "tmp"))
    (run_dir / "tmp").mkdir(exist_ok=True)
    return environment


def run_logged(argv: list[str], cwd: Path, log: Path, environment: dict[str, str]) -> None:
    print(f"Running {Path(argv[0]).name}: {' '.join(argv[1:])}", flush=True)
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(argv, cwd=cwd, env=environment, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise ReconstructionError(f"process exited {result.returncode}; inspect {log}")


def pgsr_command(args, dataset: Path, model: Path) -> list[str]:
    iterations = args.pgsr_iterations
    return [
        str(args.pgsr_python), str(args.pgsr_source / "train.py"), "-s", str(dataset), "-m", str(model),
        "--iterations", str(iterations), "--save_iterations", str(iterations), "--test_iterations", str(iterations),
        "--resolution", str(args.pgsr_resolution), "--data_device", "cpu",
        "--position_lr_max_steps", str(iterations),
        "--single_view_weight_from_iter", str(min(7000, iterations // 4)),
        "--multi_view_weight_from_iter", str(min(7000, iterations // 4)),
        "--densify_from_iter", str(min(500, iterations // 10)),
        "--densify_until_iter", str(min(15000, iterations // 2)),
    ]


def validate_pgsr_ply(path: Path) -> int:
    if not path.is_file():
        raise ReconstructionError(f"PGSR did not produce final Gaussian PLY: {path}")
    with path.open("rb") as handle:
        header = []
        for _ in range(256):
            line = handle.readline().decode("ascii").strip()
            header.append(line)
            if line == "end_header":
                break
        else:
            raise ReconstructionError("PGSR Gaussian PLY has no valid header")
    properties = {line.split()[-1] for line in header if line.startswith("property ")}
    if not {"x", "y", "z", "f_dc_0", "opacity", "scale_0", "rot_0"} <= properties:
        raise ReconstructionError("PGSR output does not have Gaussian attributes")
    count = next((int(line.split()[-1]) for line in header if line.startswith("element vertex ")), 0)
    if count <= 0:
        raise ReconstructionError("PGSR Gaussian PLY is empty")
    return count


def run(args) -> dict:
    task_dir = args.task_dir.resolve()
    inputs = task_path(task_dir, args.inputs)
    outputs = task_path(task_dir, args.outputs, exists=False)
    if args.pgsr_iterations < 4 or args.pgsr_resolution <= 0:
        raise ReconstructionError("PGSR requires at least four iterations and positive resolution")
    if args.pgsr_prior_max_points <= 0 or args.pgsr_prior_voxel_size <= 0:
        raise ReconstructionError("PGSR depth-prior point count and voxel size must be positive")
    if min(args.tsdf_voxel_size, args.tsdf_sdf_trunc, args.depth_trunc) <= 0 or not 0 <= args.confidence_percentile < 100:
        raise ReconstructionError("TSDF lengths must be positive and confidence percentile must be in [0,100)")
    if not args.da3_model.is_dir() or not (args.pgsr_source / "train.py").is_file():
        raise ReconstructionError("DA3 local model directory and PGSR source train.py are required")
    da3_source = args.da3_source.resolve()
    if (da3_source / "src/depth_anything_3").is_dir():
        da3_source /= "src"
    sys.path[:0] = [str(da3_source), *args.da3_python_path]
    try:
        import numpy
        import torch
        import open3d
        import trimesh
        from depth_anything_3.api import DepthAnything3
    except ImportError as error:
        raise ReconstructionError(f"missing scene reconstruction dependency: {error}") from error
    payload = read_json(inputs)
    artifact = payload.get("inputs", {}).get("source_media", {})
    if artifact.get("evidence") != "observed":
        raise ReconstructionError("source_media must be observed calibrated imagery")
    source = task_path(task_dir, artifact.get("path", ""))
    reader = load_colmap_reader(da3_source)
    cameras, selected, points, records = load_calibrated_scene(task_dir, source, reader, args.max_frames)
    outputs.parent.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=outputs.parent))
    environment = pgsr_environment(args, run_dir)
    run_logged([str(args.pgsr_python), "-c", "import torch; from simple_knn._C import distCUDA2; import diff_plane_rasterization; import gaussian_renderer; from scene import Scene; print(torch.__version__, torch.cuda.is_available())"], args.pgsr_source, run_dir / "pgsr_preflight.log", environment)
    coordinate = {"coordinate_frame": "colmap_world", "camera_convention": "world_to_camera_opencv", "units": "colmap_reconstruction", "metric_scale_known": False}
    frame_path, camera_path, depth_path = (run_dir / name for name in ("frames.json", "cameras.json", "depth.json"))
    write_json(frame_path, {"schema_version": "1.0", "frames": [{key: item[key] for key in ("frame_id", "image_path", "width", "height")} for item in records], **coordinate})
    write_json(camera_path, {"schema_version": "1.0", "frames": records, **coordinate})
    if args.resume_run_dir:
        depth_records, tsdf, phase_receipt = restore_depth_phase(args, task_dir, outputs.parent, artifact, records)
        depth_receipt = {"camera_conditioned": True, "align_to_input_ext_scale": True, **phase_receipt}
        tsdf_receipt = {"observation_only": True, "units": "colmap_reconstruction", **phase_receipt}
    else:
        depth_records, depth_receipt = infer_depth(args, task_dir, run_dir, records)
        tsdf, tsdf_receipt = fuse_tsdf(args, task_dir, run_dir, depth_records)
    write_json(depth_path, {"schema_version": "1.0", "frames": depth_records, "depth_kind": "camera_z", "method": "DepthAnything3", **coordinate})
    if not args.resume_run_dir:
        save_phase_cache(args, task_dir, run_dir, artifact, depth_records)
    dataset, model_dir = run_dir / "pgsr_dataset", run_dir / "pgsr_model"
    stage_pgsr_dataset(task_dir, dataset, reader, cameras, selected, points, records)
    prior_receipt = build_depth_prior(args, task_dir, depth_records, dataset / "sparse/points3D.ply")
    command = pgsr_command(args, dataset, model_dir)
    run_logged(command, args.pgsr_source, run_dir / "pgsr_training.log", environment)
    gaussian_ply = model_dir / f"point_cloud/iteration_{args.pgsr_iterations}/point_cloud.ply"
    gaussian_count = validate_pgsr_ply(gaussian_ply)
    write_json(run_dir / "reconstruction_receipt.json", {
        "schema_version": "1.0", "source_media": artifact, "frame_count": len(records),
        "source_frame_count": len(reader.read_model(str(source / "sparse/0"), ext=".bin" if (source / "sparse/0/cameras.bin").exists() else ".txt")[1]),
        "da3": {"model": str(args.da3_model), "source": str(da3_source), "resolution": args.da3_resolution, **depth_receipt},
        "tsdf": tsdf_receipt,
        "pgsr": {"source": str(args.pgsr_source), "train_sha256": sha256(args.pgsr_source / "train.py"), "command": command, "iterations": args.pgsr_iterations, "gaussians": gaussian_count, "depth_prior": prior_receipt},
        "promotion_allowed": False, **coordinate,
    })
    roles = {"frames_manifest": frame_path, "cameras": camera_path, "scene_depth": depth_path, "scene_gaussian_ply": gaussian_ply, "scene_tsdf_mesh": tsdf}
    result = {"outputs": {name: {"path": relative(task_dir, path), "evidence": "observed", "status": "candidate", "collision_eligible": False} for name, path in roles.items()}}
    write_json(outputs, result)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--task-dir", required=True, type=Path)
    result.add_argument("--inputs", required=True, type=Path)
    result.add_argument("--outputs", required=True, type=Path)
    result.add_argument("--da3-source", required=True, type=Path)
    result.add_argument("--da3-model", required=True, type=Path)
    result.add_argument("--da3-python-path", action="append", default=[])
    result.add_argument("--da3-resolution", type=int, default=504)
    result.add_argument("--da3-batch-size", type=int, default=8)
    result.add_argument("--pgsr-source", required=True, type=Path)
    result.add_argument("--pgsr-python", type=Path, default=Path(sys.executable))
    result.add_argument("--pgsr-python-path", action="append", default=[])
    result.add_argument("--pgsr-iterations", type=int, default=30000)
    result.add_argument("--pgsr-resolution", type=int, default=2)
    result.add_argument("--pgsr-prior-voxel-size", type=float, default=0.04)
    result.add_argument("--pgsr-prior-max-points", type=int, default=300000)
    result.add_argument("--resume-run-dir", type=Path)
    result.add_argument("--max-frames", type=int, default=0)
    result.add_argument("--tsdf-voxel-size", type=float, default=0.02)
    result.add_argument("--tsdf-sdf-trunc", type=float, default=0.08)
    result.add_argument("--depth-trunc", type=float, default=20.0)
    result.add_argument("--confidence-percentile", type=float, default=20.0)
    return result


if __name__ == "__main__":
    try:
        print(json.dumps(run(parser().parse_args()), indent=2))
    except (ReconstructionError, OSError, ValueError, KeyError) as error:
        print(f"scene_reconstruction failed: {error}", file=sys.stderr)
        raise SystemExit(2)
