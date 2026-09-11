#!/usr/bin/env python3
"""Reconstruct generated clean plates in their fixed source-camera coordinate frame."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile


_SPEC = importlib.util.spec_from_file_location("world_modeling_scene_helpers", Path(__file__).with_name("scene_reconstruction.py"))
scene = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(scene)


class BackgroundError(RuntimeError):
    pass


def load_inputs(task_dir: Path, inputs_path: Path) -> tuple[dict, dict, dict, dict]:
    payload = scene.read_json(scene.task_path(task_dir, inputs_path))
    inputs = payload.get("inputs", {})
    documents = {}
    for name in ("clean_plate_frames", "clean_plate_report", "cameras"):
        artifact = inputs.get(name)
        expected = "observed" if name == "cameras" else "generated"
        if not isinstance(artifact, dict) or artifact.get("evidence") != expected:
            raise BackgroundError(f"{name} must have {expected} evidence")
        if artifact.get("status") in {"contract_only", "failed", "blocked", "stale"}:
            raise BackgroundError(f"{name} is not an available reconstruction input")
        documents[name] = scene.read_json(scene.task_path(task_dir, artifact["path"]))
    clean_plates = documents["clean_plate_frames"]
    if clean_plates.get("evidence") != "generated" or clean_plates.get("camera_policy") != "fixed_source_camera":
        raise BackgroundError("clean plate manifest must declare generated evidence and fixed_source_camera policy")
    report = documents["clean_plate_report"]
    if report.get("status") in {"contract_only", "failed", "blocked", "rejected"}:
        raise BackgroundError("clean plate report rejects reconstruction input")
    cameras = documents["cameras"]
    if cameras.get("camera_convention") != "world_to_camera_opencv" or cameras.get("units") != "colmap_reconstruction":
        raise BackgroundError("source cameras must declare world_to_camera_opencv and colmap_reconstruction units")
    return inputs, clean_plates, report, cameras


def validate_clean_frames(task_dir: Path, clean_plates: dict, camera_manifest: dict, max_frames: int, outside_mask_mae: float) -> tuple[list[dict], list[dict]]:
    import numpy as np
    from PIL import Image

    source_frames = camera_manifest.get("frames")
    clean_frames = clean_plates.get("frames")
    if not isinstance(source_frames, list) or not source_frames or not isinstance(clean_frames, list) or not clean_frames:
        raise BackgroundError("clean plates and source cameras require nonempty frames lists")
    source_index = {item["frame_id"]: item for item in source_frames}
    if len(source_index) != len(source_frames):
        raise BackgroundError("source camera manifest has duplicate frame identifiers")
    if len({item.get("frame_id") for item in clean_frames}) != len(clean_frames):
        raise BackgroundError("clean plate manifest has duplicate frame identifiers")
    indices = scene.select_indices(len(clean_frames), max_frames)
    selected, checks = [], []
    for index in indices:
        clean = clean_frames[index]
        frame_id = clean.get("frame_id")
        if not isinstance(frame_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", frame_id):
            raise BackgroundError(f"unsafe frame identifier: {frame_id!r}")
        if frame_id not in source_index:
            raise BackgroundError(f"clean plate {frame_id} has no matching source camera")
        original = source_index[frame_id]
        intrinsics = np.asarray(original["intrinsics"], dtype=np.float64)
        extrinsic = np.asarray(original["world_to_camera"], dtype=np.float64)
        if intrinsics.shape != (3, 3) or extrinsic.shape != (4, 4) or not np.isfinite(intrinsics).all() or not np.isfinite(extrinsic).all():
            raise BackgroundError(f"{frame_id}: invalid source camera shape or finite values")
        if not np.allclose(extrinsic[3], [0, 0, 0, 1]) or not np.allclose(extrinsic[:3, :3].T @ extrinsic[:3, :3], np.eye(3), atol=1e-5):
            raise BackgroundError(f"{frame_id}: source world_to_camera must be rigid homogeneous")
        for name, expected in (("intrinsics", intrinsics), ("world_to_camera", extrinsic)):
            if name in clean and (np.asarray(clean[name]).shape != expected.shape or not np.allclose(clean[name], expected, atol=1e-6)):
                raise BackgroundError(f"{frame_id}: clean plate changed source {name}")
        if "image_to_source" in clean and not np.allclose(clean["image_to_source"], np.eye(3), atol=1e-6):
            raise BackgroundError(f"{frame_id}: clean plate resampling is incompatible with fixed source cameras")
        image_path = scene.task_path(task_dir, clean["image_path"])
        original_path = scene.task_path(task_dir, original["image_path"])
        mask_path = scene.task_path(task_dir, clean["mask_path"])
        if image_path == original_path:
            raise BackgroundError(f"{frame_id}: clean plate must not alias the original source image")
        expected_size = (original["width"], original["height"])
        with Image.open(image_path) as image, Image.open(original_path) as source, Image.open(mask_path) as mask_image:
            if image.size != expected_size or source.size != expected_size or mask_image.size != expected_size:
                raise BackgroundError(f"{frame_id}: generated RGB, source RGB and repair mask must retain calibrated raster dimensions")
            if (clean.get("width"), clean.get("height")) != expected_size:
                raise BackgroundError(f"{frame_id}: clean plate declared dimensions differ from source calibration")
            difference = np.abs(np.asarray(image.convert("RGB"), dtype=np.float32) - np.asarray(source.convert("RGB"), dtype=np.float32))
            mask = np.asarray(mask_image.convert("L")) > 0
        outside_error = float(difference[~mask].mean()) if (~mask).any() else 0.0
        if outside_error > outside_mask_mae:
            raise BackgroundError(f"{frame_id}: clean plate changes unmasked source pixels (MAE {outside_error:.3f} > {outside_mask_mae})")
        inside_error = float(difference[mask].mean()) if mask.any() else 0.0
        if mask.any() and inside_error == 0:
            raise BackgroundError(f"{frame_id}: repair mask is nonempty but clean plate copies unchanged source pixels")
        selected.append({
            "frame_id": frame_id, "image_path": scene.relative(task_dir, image_path),
            "source_image_path": scene.relative(task_dir, original_path), "mask_path": scene.relative(task_dir, mask_path),
            "width": expected_size[0], "height": expected_size[1],
            "intrinsics": intrinsics.tolist(), "world_to_camera": extrinsic.tolist(),
        })
        checks.append({
            "frame_id": frame_id, "masked_pixel_count": int(mask.sum()), "outside_mask_mae": outside_error,
            "inside_mask_mae": inside_error, "image_sha256": scene.sha256(image_path), "mask_sha256": scene.sha256(mask_path),
            "source_camera_unchanged": True, "raster_unchanged": True,
        })
    if not any(item["masked_pixel_count"] for item in checks):
        raise BackgroundError("clean plate set has no repair pixels; cannot claim a newly completed background")
    centers = np.array([np.linalg.inv(item["world_to_camera"])[:3, 3] for item in selected])
    if np.linalg.norm(centers[0] - centers[-1]) < 1e-6:
        raise BackgroundError("clean plate camera anchors have no baseline for scale alignment")
    return selected, checks


def generated_colmap(reader, records: list[dict]):
    import numpy as np

    cameras, images = {}, []
    for index, record in enumerate(records, start=1):
        intrinsics = np.asarray(record["intrinsics"])
        extrinsic = np.asarray(record["world_to_camera"])
        camera = reader.Camera(index, "PINHOLE", record["width"], record["height"], [intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]])
        scene.camera_intrinsics(camera)
        cameras[index] = camera
        images.append(reader.Image(
            id=index, qvec=reader.rotmat2qvec(extrinsic[:3, :3]), tvec=extrinsic[:3, 3], camera_id=index,
            name=f"{record['frame_id']}.png", xys=np.empty((0, 2)), point3D_ids=np.empty(0, dtype=np.int64),
        ))
    return cameras, images


def output_records(task_dir: Path, gaussian_ply: Path, mesh: Path, report: Path) -> dict:
    roles = {"generated_background_gaussian_ply": gaussian_ply, "generated_background_mesh": mesh, "background_reconstruction_report": report}
    return {"outputs": {name: {
        "path": scene.relative(task_dir, path), "evidence": "generated", "status": "candidate", "collision_eligible": False,
    } for name, path in roles.items()}}


def run(args) -> dict:
    task_dir = args.task_dir.resolve()
    outputs = scene.task_path(task_dir, args.outputs, exists=False)
    if args.resume_run_dir:
        raise BackgroundError("background reconstruction must not reuse an observed-scene phase cache")
    if min(args.tsdf_voxel_size, args.tsdf_sdf_trunc, args.depth_trunc, args.pgsr_prior_voxel_size) <= 0 or args.pgsr_prior_max_points <= 0:
        raise BackgroundError("reconstruction lengths and PGSR point limits must be positive")
    if args.pgsr_iterations < 4 or args.pgsr_resolution <= 0 or args.max_outside_mask_mae < 0:
        raise BackgroundError("invalid PGSR training or fixed-raster validation parameters")
    if not 0 <= args.confidence_percentile < 100:
        raise BackgroundError("confidence percentile must be in [0,100)")
    inputs, clean_plates, clean_report, source_cameras = load_inputs(task_dir, args.inputs)
    records, frame_checks = validate_clean_frames(task_dir, clean_plates, source_cameras, args.max_frames, args.max_outside_mask_mae)
    da3_source = args.da3_source.resolve()
    if (da3_source / "src/depth_anything_3").is_dir():
        da3_source /= "src"
    sys.path[:0] = [str(da3_source), *args.da3_python_path]
    if not args.da3_model.is_dir() or not (args.pgsr_source / "train.py").is_file():
        raise BackgroundError("DA3 local weights and PGSR source are required")
    try:
        import torch
        import open3d
        import trimesh
        from depth_anything_3.api import DepthAnything3
    except ImportError as error:
        raise BackgroundError(f"missing background reconstruction dependency: {error}") from error
    reader = scene.load_colmap_reader(da3_source)
    cameras, images = generated_colmap(reader, records)
    outputs.parent.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=outputs.parent))
    environment = scene.pgsr_environment(args, run_dir)
    scene.run_logged([str(args.pgsr_python), "-c", "import torch; from simple_knn._C import distCUDA2; import diff_plane_rasterization; import gaussian_renderer; from scene import Scene; print(torch.__version__, torch.cuda.is_available())"], args.pgsr_source, run_dir / "pgsr_preflight.log", environment)
    coordinate = {
        "evidence": "generated", "coordinate_frame": "colmap_world", "camera_convention": "world_to_camera_opencv",
        "units": "colmap_reconstruction", "metric_scale_known": False, "camera_policy": "fixed_source_camera",
    }
    scene.write_json(run_dir / "frames.json", {"frames": records, **coordinate})
    depth_records, da3_report = scene.infer_depth(args, task_dir, run_dir, records)
    scene.write_json(run_dir / "depth.json", {"frames": depth_records, "depth_kind": "camera_z", **coordinate})
    mesh_path, mesh_report = scene.fuse_tsdf(args, task_dir, run_dir, depth_records)
    dataset, model_dir = run_dir / "pgsr_dataset", run_dir / "pgsr_model"
    scene.stage_pgsr_dataset(task_dir, dataset, reader, cameras, images, {}, records)
    prior_report = scene.build_depth_prior(args, task_dir, depth_records, dataset / "sparse/points3D.ply")
    command = scene.pgsr_command(args, dataset, model_dir)
    scene.run_logged(command, args.pgsr_source, run_dir / "pgsr_training.log", environment)
    gaussian_ply = model_dir / f"point_cloud/iteration_{args.pgsr_iterations}/point_cloud.ply"
    gaussian_count = scene.validate_pgsr_ply(gaussian_ply)
    report_path = run_dir / "background_reconstruction_report.json"
    scene.write_json(report_path, {
        "schema_version": "1.0", "status": "candidate", **coordinate,
        "source_artifacts": inputs, "clean_plate_report_status": clean_report.get("status", "unspecified"),
        "clean_plate_count": len(clean_plates["frames"]), "used_frame_count": len(records),
        "source_camera_count": len(source_cameras["frames"]), "frame_checks": frame_checks,
        "da3": {"model": str(args.da3_model), "source": str(da3_source), "resolution": args.da3_resolution, **da3_report},
        "tsdf": {**mesh_report, "observation_only": False, "evidence": "generated"},
        "pgsr": {"iterations": args.pgsr_iterations, "gaussians": gaussian_count, "command": command, "train_sha256": scene.sha256(args.pgsr_source / "train.py"), "depth_prior": {**prior_report, "evidence": "generated"}},
        "promotion_allowed": False, "collision_eligible": False,
        "promotion_blockers": ["Generated clean plates require independent cross-view consistency and novel-view inspection.", "Generated mesh is not an observed collision surface."],
    })
    result = output_records(task_dir, gaussian_ply, mesh_path, report_path)
    scene.write_json(outputs, result)
    return result


def parser():
    result = scene.parser()
    result.description = __doc__
    result.add_argument("--max-outside-mask-mae", type=float, default=1.0)
    return result


if __name__ == "__main__":
    try:
        print(json.dumps(run(parser().parse_args()), indent=2))
    except (BackgroundError, scene.ReconstructionError, OSError, ValueError, KeyError) as error:
        print(f"background_reconstruction failed: {error}", file=sys.stderr)
        raise SystemExit(2)
