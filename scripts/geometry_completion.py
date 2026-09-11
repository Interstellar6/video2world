#!/usr/bin/env python3
"""Fresh generated-video SAM3/DA3 preprocessing and official Stream3D + SAM3D."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from world_modeling.provider_io import input_artifact, local_path, publish, read_json, write_json
from world_modeling.object_lineage import lineage_metadata, validate_object_lineages


class CompletionError(RuntimeError):
    pass


def relative(task: Path, path: Path) -> str:
    return path.resolve().relative_to(task.resolve()).as_posix()


def identifier(value) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", value):
        raise CompletionError(f"invalid object identifier: {value!r}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_indices(frame_count: int, views: int) -> list[int]:
    if type(frame_count) is not int or type(views) is not int or frame_count < 9 or views < 4 or views >= frame_count:
        raise CompletionError("orbit sampling needs >=9 frames and >=4 distinct sampled views for heldout registration")
    # The final orbit frame repeats azimuth zero; sample the open interval.
    return [(index * (frame_count - 1)) // views for index in range(views)]


def chunk_starts(frame_count: int, size: int = 8, overlap: int = 2) -> list[int]:
    if frame_count < size or not 0 <= overlap < size:
        raise CompletionError("Stream3D needs at least one complete chunk")
    starts = list(range(0, frame_count - size + 1, size - overlap))
    if starts[-1] != frame_count - size:
        starts.append(frame_count - size)
    return starts


def environment(task: Path, python_paths=()) -> dict:
    result = os.environ.copy()
    # The official runner uses an existing local DINO repository and checkpoint.
    # Keep that model cache address stable when relocating writable XDG caches.
    result.setdefault("TORCH_HOME", str(Path.home() / ".cache/torch"))
    cache = local_path(task, "stages/geometry_completion/cache", exists=False)
    for name, suffix in (("TMPDIR", "tmp"), ("XDG_CACHE_HOME", "xdg")):
        directory = cache / suffix
        directory.mkdir(parents=True, exist_ok=True)
        result[name] = str(directory)
    result.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HYDRA_FULL_ERROR": "1", "PYTHONDONTWRITEBYTECODE": "1", "ATTN_BACKEND": "sdpa", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    paths = [str(path) for path in python_paths]
    if result.get("PYTHONPATH"):
        paths.append(result["PYTHONPATH"])
    result["PYTHONPATH"] = os.pathsep.join(paths)
    return result


def execute(command: list[str], cwd: Path, env: dict, log_path: Path) -> dict:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=False, text=True)
    if process.returncode:
        raise CompletionError(f"command failed with exit {process.returncode}; see {log_path}")
    return {"argv": command, "returncode": process.returncode}


def sam3_builder_path(source: Path, backend: str) -> Path:
    return source / ("sam3/sam3/model_builder.py" if backend == "sam3i" else "sam3/model_builder.py")


def sam3_command(args, task: Path, inputs: Path, outputs: Path) -> list[str]:
    return [str(args.sam3_python), str(Path(__file__).with_name("sam3_components.py")),
            "--task-dir", str(task), "--inputs", str(inputs), "--outputs", str(outputs),
            "--source-root", str(args.sam3_source), "--checkpoint", str(args.sam3_checkpoint),
            "--backend", args.sam3_backend, "--instruction-stage", args.sam3_instruction_stage,
            "--confidence-threshold", str(args.sam3_confidence)]


def prepare_pipeline_config(source: Path, destination: Path) -> dict:
    import yaml

    configuration = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(configuration, dict) or configuration.get("_target_") != "sam3d_objects.pipeline.inference_pipeline_pointmap.InferencePipelinePointMap":
        raise CompletionError("pipeline config must target the official SAM3D pointmap pipeline")
    references = []
    def resolve(value):
        if isinstance(value, list):
            return [resolve(item) for item in value]
        if not isinstance(value, dict):
            return value
        output = {}
        for key, item in value.items():
            if isinstance(item, str) and (key.endswith("_config_path") or key.endswith("_ckpt_path") or key == "pretrained_model_name_or_path"):
                path = (source.parent / item).resolve()
                if not path.exists():
                    raise CompletionError(f"SAM3D config requires missing local model resource: {path}")
                output[key] = str(path)
                references.append({"key": key, "source_file": str(path), "size_bytes": path.stat().st_size if path.is_file() else None})
            else:
                output[key] = resolve(item)
        return output
    configuration = resolve(configuration)
    configuration["compile_model"] = False
    destination.write_text(yaml.safe_dump(configuration, sort_keys=False), encoding="utf-8")
    return {"source_config": str(source), "source_sha256": sha256(source), "model_resources": references, "compile_model": False}


def checked_hash(task: Path, value, expected, label: str) -> Path:
    path = local_path(task, value)
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected) or not path.is_file() or sha256(path) != expected:
        raise CompletionError(f"{label}: missing or changed SHA256 binding")
    return path


def finite_number(value, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise CompletionError(f"{label} must be a finite number")
    return float(value)


def unique_objects(value, label: str) -> dict:
    if not isinstance(value, list) or not value:
        raise CompletionError(f"{label} requires nonempty object records")
    result = {}
    for record in value:
        if not isinstance(record, dict):
            raise CompletionError(f"{label} object records must be dictionaries")
        object_id = identifier(record.get("object_id"))
        if object_id in result:
            raise CompletionError(f"{label} contains duplicate object IDs")
        result[object_id] = record
    return result


def validate_orbit_inputs(task: Path, obj: dict) -> list[dict]:
    import numpy as np

    clips = obj.get("orbits")
    if not isinstance(clips, list) or len(clips) != 3:
        raise CompletionError("geometry completion requires all three distinct orbit clips")
    seen_ids, seen_hashes, seen_bases, validated = set(), set(), set(), []
    common, common_calibration = None, None
    for clip in clips:
        if not isinstance(clip, dict):
            raise CompletionError("orbit records must be objects")
        orbit_id = identifier(clip["orbit_id"])
        checked_hash(task, clip["video_path"], clip.get("video_sha256"), "generated orbit video")
        path = checked_hash(task, clip["cameras_path"], clip.get("conditioning_cameras_sha256"), "conditioning cameras")
        camera = read_json(path)
        elevation = finite_number(clip.get("elevation_degrees"), "elevation")
        up = np.asarray(camera.get("basis", {}).get("up_vector"), dtype=float)
        reference = np.asarray(camera.get("basis", {}).get("reference_direction"), dtype=float)
        if any(value.shape != (3,) or not np.isfinite(value).all() for value in (up, reference)) or not np.isclose(np.linalg.norm(up), 1) or not np.isclose(np.linalg.norm(reference), 1) or not np.isclose(up @ reference, 0, atol=1e-6):
            raise CompletionError("conditioning orbit basis must be finite orthonormal scene vectors")
        # Orbits are distinguished by elevation, by rotation basis, or both: the
        # deployed mode uses one basis at three elevations, the three-axis mode
        # uses three orthonormal bases at one elevation. Repeating an identical
        # orbit is still refused.
        basis_key = (round(elevation, 6), tuple(np.round(up, 6)), tuple(np.round(reference, 6)))
        if orbit_id in seen_ids or clip["video_sha256"] in seen_hashes or basis_key in seen_bases or not -80 < elevation < 80:
            raise CompletionError("three distinct orbit IDs, video hashes and rotation bases with in-range elevations are required")
        seen_ids.add(orbit_id)
        seen_hashes.add(clip["video_sha256"])
        seen_bases.add(basis_key)
        if clip.get("evidence") == "conditioned_synthesis":
            if clip.get("camera_pose_status") != "conditioning_trajectory_not_verified_for_generated_pixels":
                raise CompletionError("generated pixels require explicit unverified conditioning-camera provenance")
        elif clip.get("evidence") == "direct_asset_render":
            # Explicit operator choice: the orbit video is the conditioning render
            # itself, so its cameras are the trajectory that rendered the asset.
            # This is not a synthesis hypothesis, but the mesh stays generated.
            if clip.get("camera_pose_status") != "conditioning_trajectory_matches_source_render_asset":
                raise CompletionError("direct asset render requires its own explicit camera provenance")
        else:
            raise CompletionError("orbit video must declare conditioned_synthesis or direct_asset_render evidence")
        if any(type(clip.get(key)) is not int or clip[key] <= 0 for key in ("frame_count", "width", "height", "fps")):
            raise CompletionError("orbit frame count, raster and FPS must be positive integers")
        if clip["frame_count"] != 61 or finite_number(clip.get("azimuth_span_degrees"), "azimuth span") != 360:
            raise CompletionError("current orbit boundary requires 61 frames covering a closed 360-degree trajectory")
        if camera.get("camera_convention") != "world_to_camera_opencv" or camera.get("object_id") != obj["object_id"] or camera.get("orbit_id") != orbit_id:
            raise CompletionError("conditioning camera identity or convention differs from the orbit")
        for key in ("frame_count", "width", "height", "elevation_degrees", "azimuth_span_degrees"):
            if camera.get(key) != clip[key]:
                raise CompletionError(f"conditioning camera {key} differs from the video manifest")
        coordinate, units = camera.get("coordinate_frame"), camera.get("units")
        if not isinstance(coordinate, str) or not coordinate or not isinstance(units, str) or units in ("", "unspecified"):
            raise CompletionError("conditioning cameras require an explicit coordinate frame and units")
        center = np.asarray(camera.get("object_center_world"), dtype=float)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise CompletionError("conditioning orbit centre must be a finite scene vector")
        distance = finite_number(camera.get("distance"), "orbit distance")
        radius = finite_number(camera.get("object_support_radius"), "object support radius")
        if not distance > radius > 0:
            raise CompletionError("conditioning orbit distance must exceed positive object support radius")
        signature = (coordinate, units, clip["width"], clip["height"], clip["fps"])
        # Each orbit may carry its own orthonormal basis so one object can be
        # conditioned on three orthogonal great circles (horizontal plus two
        # vertical). They must still be the same object: shared centre, support
        # radius, orbit distance, raster, FPS and calibration.
        shared = np.r_[center, distance, radius]
        if common is not None and (signature != common[0] or not np.allclose(shared, common[1], atol=1e-6)):
            raise CompletionError("three orbits must share one object centre, support radius, distance, raster and FPS")
        common = (signature, shared)
        records = camera.get("frames")
        if not isinstance(records, list) or len(records) != 61:
            raise CompletionError("conditioning trajectory must contain all 61 camera frames")
        calibrations = []
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise CompletionError("conditioning camera frames must be objects")
            azimuth = finite_number(record.get("azimuth_degrees"), "camera azimuth")
            if record.get("frame_id") != f"{index:06d}" or not np.isclose(azimuth, index * 6, atol=1e-6):
                raise CompletionError("conditioning frames must follow ordered uniform 0..360 azimuths")
            pose = rigid_pose(record["world_to_camera"])
            theta, angle = np.radians([azimuth, elevation])
            radial = np.cos(theta) * reference + np.sin(theta) * np.cross(up, reference)
            expected_eye = center + distance * (np.cos(angle) * radial + np.sin(angle) * up)
            eye = -pose[:3, :3].T @ pose[:3, 3]
            declared_eye = np.asarray(record.get("camera_center_world"), dtype=float)
            if declared_eye.shape != (3,) or not np.isfinite(declared_eye).all() or not np.allclose(eye, expected_eye, atol=1e-5) or not np.allclose(declared_eye, eye, atol=1e-5) or not np.allclose(pose[:3, :3] @ (center - eye), [0, 0, distance], atol=1e-5):
                raise CompletionError("conditioning pose does not realize its declared azimuth/elevation/look-at target")
            calibration = np.asarray(record["intrinsics"], dtype=float)
            if calibration.shape != (3, 3) or not np.isfinite(calibration).all() or min(calibration[0, 0], calibration[1, 1]) <= 0 or not np.allclose(calibration[2], [0, 0, 1]) or not 0 <= calibration[0, 2] <= clip["width"] or not 0 <= calibration[1, 2] <= clip["height"]:
                raise CompletionError("conditioning camera raster calibration is invalid")
            calibrations.append(calibration)
        if not np.allclose(calibrations, calibrations[0], atol=1e-6):
            raise CompletionError("conditioning orbit intrinsics must remain fixed")
        if common_calibration is not None and not np.allclose(calibrations[0], common_calibration, atol=1e-6):
            raise CompletionError("three conditioning orbits must share one raster calibration")
        common_calibration = calibrations[0]
        validated.append({"clip": clip, "camera": camera})
    return validated


def decode_orbits(args, obj: dict, object_dir: Path, category: str) -> tuple[Path, Path, list[dict]]:
    from PIL import Image

    validated = validate_orbit_inputs(args.task_dir, obj)
    frames, provenance = [], []
    for item in validated:
        clip = item["clip"]
        orbit_id = identifier(clip["orbit_id"])
        video = local_path(args.task_dir, clip["video_path"])
        digest = sha256(video)
        probe = subprocess.run([args.ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=width,height,nb_read_frames,avg_frame_rate", "-of", "json", str(video)], capture_output=True, text=True, check=False)
        if probe.returncode:
            raise CompletionError(f"cannot decode generated orbit: {probe.stderr}")
        streams = json.loads(probe.stdout).get("streams", [])
        if len(streams) != 1 or any(int(streams[0].get(key, 0)) != clip[target] for key, target in (("nb_read_frames", "frame_count"), ("width", "width"), ("height", "height"))):
            raise CompletionError("generated clip decoded count or raster differs from orbit manifest")
        rate = streams[0].get("avg_frame_rate", "0/1").split("/")
        if len(rate) != 2 or float(rate[1]) <= 0 or not math.isclose(float(rate[0]) / float(rate[1]), clip["fps"], abs_tol=.01):
            raise CompletionError("generated clip FPS differs from orbit manifest")
        indices = sample_indices(clip["frame_count"], args.views_per_orbit)
        directory = object_dir / "decoded" / orbit_id
        directory.mkdir(parents=True)
        selection = "+".join(f"eq(n\\,{index})" for index in indices)
        command = [args.ffmpeg, "-v", "error", "-i", str(video), "-vf", f"select={selection}", "-vsync", "0", "-frames:v", str(len(indices)), "-start_number", "0", str(directory / "%06d.png")]
        execute(command, object_dir, environment(args.task_dir), object_dir / f"decode_{orbit_id}.log")
        extracted = sorted(directory.glob("*.png"))
        if len(extracted) != len(indices):
            raise CompletionError("ffmpeg did not materialize the selected generated frames")
        pixel_hashes = set()
        for source_index, path in zip(indices, extracted):
            with Image.open(path) as image:
                width, height = image.size
                pixel_hashes.add(hashlib.sha256(image.convert("RGB").tobytes()).hexdigest())
            frame_id = f"{len(frames):06d}"
            frames.append({"frame_id": frame_id, "image_path": relative(args.task_dir, path), "image_sha256": sha256(path), "width": width, "height": height})
            provenance.append({"frame_id": frame_id, "orbit_id": orbit_id, "generated_video_path": relative(args.task_dir, video), "generated_video_sha256": digest, "generated_frame_index": source_index, "elevation_degrees": clip["elevation_degrees"], "conditioning_cameras_path": clip["cameras_path"], "conditioning_cameras_sha256": clip["conditioning_cameras_sha256"], "conditioning_cameras_used_as_geometry": False})
        if len(pixel_hashes) < 2:
            raise CompletionError("generated orbit sampled pixels repeat a static image")
        checked_hash(args.task_dir, clip["video_path"], digest, "video changed during decoding")
        checked_hash(args.task_dir, clip["cameras_path"], clip["conditioning_cameras_sha256"], "cameras changed during decoding")
    frames_path = object_dir / "generated_frames.json"
    proposals_path = object_dir / "generated_proposals.json"
    write_json(frames_path, {"schema_version": "1.0", "frames": frames, "evidence": "generated_rgb"})
    write_json(proposals_path, {"schema_version": "1.0", "frames": [{**frame, "objects": [{"object_id": obj["object_id"], "category": category, "bbox_xyxy": [0, 0, frame["width"], frame["height"]], "components": []}]} for frame in frames]})
    write_json(object_dir / "sampled_frame_provenance.json", {"frames": provenance, "source": "actual_decoded_FixAnything_output", "sampling": "uniform_conditioning_azimuth_indices_from_three_elevations", "conditioning_trajectory_validated": True, "generated_full_orbit_motion_validated": False})
    return frames_path, proposals_path, provenance


def predict_generated_depth(model, image_paths: list[str], resolution: int):
    return model.inference(image=image_paths, extrinsics=None, intrinsics=None, align_to_input_ext_scale=False, infer_gs=False, process_res=resolution, process_res_method="upper_bound_resize", export_dir=None)


def rigid_pose(value):
    import numpy as np

    pose = np.asarray(value, dtype=np.float64)
    if pose.shape == (3, 4):
        pose = np.vstack((pose, [0, 0, 0, 1]))
    if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1]):
        raise CompletionError("DA3 generated-frame pose is not a finite 4x4 transform")
    if not np.allclose(pose[:3, :3] @ pose[:3, :3].T, np.eye(3), atol=1e-3) or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-3):
        raise CompletionError("DA3 generated-frame pose has a non-rigid rotation")
    return pose


def generated_mask_inputs(task: Path, request: dict) -> tuple[list[dict], dict]:
    from PIL import Image

    frames = read_json(checked_hash(task, request["frames_manifest_path"], request.get("frames_manifest_sha256"), "generated frames manifest"))["frames"]
    masks = read_json(checked_hash(task, request["masks_manifest_path"], request.get("masks_manifest_sha256"), "generated masks manifest"))["masks"]
    if not isinstance(frames, list) or not frames or not isinstance(masks, list):
        raise CompletionError("generated frames and masks must be nonempty frame records")
    frame_ids, parents = set(), {}
    for frame in frames:
        frame_id = identifier(frame["frame_id"])
        if frame_id in frame_ids:
            raise CompletionError("generated frame IDs must be unique")
        frame_ids.add(frame_id)
        path = checked_hash(task, frame["image_path"], frame.get("image_sha256"), "generated source RGB")
        with Image.open(path) as image:
            if image.size != (frame["width"], frame["height"]):
                raise CompletionError("generated source RGB raster differs from frame manifest")
    for mask in masks:
        if mask.get("component_id") != "__object__":
            continue
        frame_id = str(mask["frame_id"])
        if mask.get("object_id") != request["object_id"] or frame_id not in frame_ids or frame_id in parents:
            raise CompletionError("generated masks require exactly one unambiguous parent per source frame")
        path = checked_hash(task, mask["mask_path"], mask.get("mask_sha256"), "generated parent mask")
        frame = next(item for item in frames if item["frame_id"] == frame_id)
        if local_path(task, mask.get("image_path", "")) != local_path(task, frame["image_path"]):
            raise CompletionError("generated parent mask source image differs from its frame binding")
        with Image.open(path) as image:
            if image.size != (frame["width"], frame["height"]):
                raise CompletionError("generated parent mask must use the original generated RGB raster")
        parents[frame_id] = mask
    if set(parents) != frame_ids:
        raise CompletionError("no accepted generated-object mask for one or more frames")
    return frames, parents


def save_generated_geometry(task: Path, request: dict, prediction, mappings) -> dict:
    import cv2
    import numpy as np
    from PIL import Image

    frames, parent_masks = generated_mask_inputs(task, request)
    depth, confidence = np.asarray(prediction.depth), np.asarray(prediction.conf)
    intrinsics, extrinsics = np.asarray(prediction.intrinsics), np.asarray(prediction.extrinsics)
    colors = np.asarray(prediction.processed_images)
    mappings = np.asarray(mappings, dtype=float)
    if depth.ndim != 3 or len(depth) != len(frames) or confidence.shape != depth.shape or colors.shape != (*depth.shape, 3) or intrinsics.shape != (len(frames), 3, 3) or extrinsics.shape not in ((len(frames), 3, 4), (len(frames), 4, 4)) or mappings.shape != (len(frames), 3, 3):
        raise CompletionError("DA3 must return depth, confidence, RGB, and K for every generated frame")
    if not np.isfinite(colors).all() or (colors < 0).any() or (colors > 255).any():
        raise CompletionError("DA3 processed RGB must contain finite byte-range colors")
    dataset = local_path(task, request["dataset_root"], exists=False)
    split = dataset / "render_spiral_100"
    image_dir, mask_dir, da3_dir = split / "images", split / "masks", split / "da3"
    for path in (image_dir, mask_dir, da3_dir / "results_output"):
        path.mkdir(parents=True, exist_ok=False)
    records, poses = [], []
    for index, frame in enumerate(frames):
        frame_id = str(frame["frame_id"])
        if frame_id not in parent_masks:
            raise CompletionError(f"SAM3 produced no accepted generated-object mask for frame {frame_id}")
        original_mask = np.asarray(Image.open(local_path(task, parent_masks[frame_id]["mask_path"])).convert("L")) > 127
        mapping = np.asarray(mappings[index], dtype=np.float64)
        if not np.isfinite(mapping).all() or not np.allclose(mapping[2], [0, 0, 1]) or min(mapping[0, 0], mapping[1, 1]) <= 0 or not np.allclose([mapping[0, 1], mapping[1, 0]], [0, 0]):
            raise CompletionError("DA3 raster mapping is invalid")
        height, width = depth[index].shape
        mask = cv2.warpAffine(original_mask.astype(np.uint8), mapping[:2], (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
        finite = np.isfinite(depth[index]) & (depth[index] > 0) & np.isfinite(confidence[index])
        selected = finite & mask
        if not selected.any():
            raise CompletionError("generated foreground has no finite confident positive depth")
        threshold = float(np.percentile(confidence[index][selected], request["confidence_percentile"]))
        valid = selected & (confidence[index] >= threshold)
        if int(valid.sum()) < 32 or valid.sum() / max(1, mask.sum()) < 0.2:
            raise CompletionError("generated object has insufficient depth-supported foreground")
        calibration = intrinsics[index]
        if not np.isfinite(calibration).all() or min(calibration[0, 0], calibration[1, 1]) <= 0 or not np.allclose(calibration[2], [0, 0, 1]):
            raise CompletionError("DA3 generated-frame intrinsic matrix is invalid")
        pose = rigid_pose(extrinsics[index])
        stem = f"frame_{index:06d}"
        image_path, mask_path, depth_path = image_dir / f"{stem}.png", mask_dir / f"{stem}.png", da3_dir / "results_output" / f"{stem}.npz"
        Image.fromarray(colors[index].astype(np.uint8)).save(image_path)
        Image.fromarray(valid.astype(np.uint8) * 255).save(mask_path)
        cleaned_depth = np.where(finite, depth[index], 0).astype(np.float32)
        np.savez_compressed(depth_path, depth=cleaned_depth, intrinsics=np.asarray(calibration, dtype=np.float32), confidence=np.asarray(confidence[index], dtype=np.float32), world_to_camera=np.asarray(pose, dtype=np.float32))
        poses.append(pose)
        records.append({"frame_id": frame_id, "stream3d_frame_name": stem, "image_path": relative(task, image_path), "mask_path": relative(task, mask_path), "depth_npz_path": relative(task, depth_path), "image_sha256": sha256(image_path), "mask_sha256": sha256(mask_path), "depth_sha256": sha256(depth_path), "intrinsics": calibration.tolist(), "world_to_camera": pose.tolist(), "source_image_to_processed": mapping.tolist(), "confidence_threshold": threshold, "valid_foreground_pixels": int(valid.sum())})
    centers = np.array([-pose[:3, :3].T @ pose[:3, 3] for pose in poses])
    span = float(np.linalg.norm(centers.max(axis=0) - centers.min(axis=0)))
    if span < 1e-5:
        raise CompletionError("DA3 generated-frame camera estimates collapsed to one location")
    pose_path = da3_dir / "camera_poses.txt"
    np.savetxt(pose_path, np.array(poses).reshape(len(poses), 16), fmt="%.10g")
    manifest = {"schema_version": "1.0", "object_id": request["object_id"], "frames": records, "camera_poses_path": relative(task, pose_path), "camera_poses_convention": "world_to_camera_opencv", "camera_estimation": "fresh_DA3_joint_generated_RGB", "camera_conditioned": False, "conditioning_poses_used": False, "coordinate_frame": "generated_DA3_world", "unit": "estimated_depth_units", "estimated_camera_center_span": span, "room_alignment": "unknown", "cross_view_geometry_acceptance": "numeric_depth_and_pose_checks_only; synthesis_consistency_requires_downstream_QA"}
    manifest_path = dataset / "input_manifest.json"
    write_json(manifest_path, manifest)
    return manifest


def da3_worker(task: Path, request_path: Path) -> None:
    import numpy as np
    import torch

    request = read_json(local_path(task, request_path))
    sys.path.insert(0, request["da3_source"])
    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.utils.io.input_processor import InputProcessor

    if not torch.cuda.is_available():
        raise CompletionError("fresh generated-frame DA3 inference requires CUDA")
    frames, _ = generated_mask_inputs(task, request)
    image_paths = [str(local_path(task, frame["image_path"])) for frame in frames]
    model = DepthAnything3.from_pretrained(request["da3_model"], local_files_only=True).to("cuda").eval()
    prediction = predict_generated_depth(model, image_paths, request["resolution"])
    # Identity intrinsics measure only the official resize/crop mapping. They
    # are never passed to model inference as a camera hypothesis.
    _, _, transformed_identity = InputProcessor()(image=image_paths, intrinsics=np.repeat(np.eye(3)[None], len(frames), axis=0), process_res=request["resolution"], process_res_method="upper_bound_resize", sequential=True)
    save_generated_geometry(task, request, prediction, transformed_identity.numpy())


def validate_generated_camera_chain(task: Path, geometry_path: Path, provenance_path: Path, lifting_path: Path, seed: int) -> dict:
    # Reuse the final registration camera gate before expensive mesh inference;
    # this only validates camera initialization, never observed-object alignment.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from object_registration import RegistrationError, camera_chain

    geometry, source = read_json(geometry_path), read_json(provenance_path)
    references = {item["frame_id"]: item for item in source["frames"]}
    frames = geometry.get("frames", [])
    if len(references) != len(source["frames"]) or len(frames) != len(references) or {item["frame_id"] for item in frames} != set(references) or geometry.get("conditioning_poses_used") is not False or geometry.get("coordinate_frame") != "generated_DA3_world":
        raise CompletionError("generated camera geometry must map exactly to independently decoded source frames")
    estimated, conditioned, heldout, counts = [], [], [], {}
    for frame in frames:
        for key in ("image", "mask"):
            checked_hash(task, frame[key + "_path"], frame[key + "_sha256"], "processed generated " + key)
        checked_hash(task, frame["depth_npz_path"], frame["depth_sha256"], "processed generated depth")
        reference = references[frame["frame_id"]]
        checked_hash(task, reference["generated_video_path"], reference["generated_video_sha256"], "source generated video")
        cameras_path = checked_hash(task, reference["conditioning_cameras_path"], reference.get("conditioning_cameras_sha256"), "conditioning cameras before Stream3D")
        cameras = read_json(cameras_path)
        orbit = reference["orbit_id"]
        counts[orbit] = counts.get(orbit, 0) + 1
        estimated.append(frame["world_to_camera"])
        conditioned.append(cameras["frames"][reference["generated_frame_index"]]["world_to_camera"])
        heldout.append(counts[orbit] % 4 == 0)
    if len(counts) != 3 or min(counts.values()) < 4:
        raise CompletionError("generated camera validation requires >=4 frames from each of three orbits")
    lifting = read_json(lifting_path)
    voxel = lifting.get("parameters", {}).get("voxel_size", lifting.get("voxel_size", lifting.get("geometry_parameters", {}).get("voxel_size")))
    voxel = finite_number(voxel, "observed voxel size")
    if voxel <= 0:
        raise CompletionError("observed voxel size must be positive")
    try:
        _, report = camera_chain(estimated, conditioned, heldout, 6 * voxel, 12, seed)
    except RegistrationError as error:
        raise CompletionError(f"generated camera alignment failed before Stream3D: {error}") from error
    return {**report, "acceptance_scope": "generated_camera_initialization_only_not_observed_registration"}


def validate_observed_input(task: Path, object_id: str, orbit: dict, assembly: dict, observed: dict, lifting: dict) -> None:
    if object_id not in lifting.get("accepted_object_ids", []) or observed.get("association_status") != "geometrically_verified":
        raise CompletionError("completion requires a currently accepted lifted object")
    path = checked_hash(task, observed["ply_path"], observed.get("sha256"), "observed object geometry")
    for label, record in (("orbit", orbit.get("observed_geometry", {})), ("assembly", assembly.get("observed_geometry", {}))):
        if local_path(task, record.get("ply_path", "")) != path or record.get("sha256") != observed["sha256"] or any(record.get(key) != observed.get(key) for key in ("coordinate_frame", "units", "observed_anchor")):
            raise CompletionError(f"{label} and isolated object geometry bindings disagree")
    for clip in orbit["orbits"]:
        camera = read_json(local_path(task, clip["cameras_path"]))
        if camera["coordinate_frame"] != observed.get("coordinate_frame") or camera["units"] != observed.get("units"):
            raise CompletionError("conditioning cameras are not in the observed object's declared scene frame")


def stream3d_command(args, dataset: Path, output_root: Path, model_config: Path, hydra_dir: Path) -> list[str]:
    return [str(args.stream3d_python), "-m", "streaming.runner", "backend=sam3d", "data.roots=" + json.dumps([str(dataset)]), "data.da3_dir_name=da3", "camera_pose_source=da3", "streaming.topk=8", "streaming.stage2_selection.topk=8", "pipeline.ss_weight_source=mass_relative", "pipeline.stage2_weighting.weight_source=mass_relative", "pipeline.use_stage1_distillation=true", f"pipeline.stage1_inference_steps={args.stage1_steps}", "pipeline.use_stage2_distillation=false", f"pipeline.stage2_inference_steps={args.stage2_steps}", "pipeline.decode_formats=[mesh]", "pipeline.with_mesh_postprocess=false", "pipeline.with_texture_baking=false", "pipeline.use_vertex_color=true", "output_root=" + json.dumps(str(output_root)), "model_config_path=" + json.dumps(str(model_config)), "chunk_size=8", "chunk_overlap=2", "chunk_indices=[-1]", f"seed={args.seed}", "hydra.run.dir=" + json.dumps(str(hydra_dir)), "hydra.output_subdir=.hydra", "hydra.job.chdir=false"]


def sample_generated_mesh(mesh_path: Path, ply_path: Path, count: int, seed: int) -> dict:
    import numpy as np
    import trimesh

    scene = trimesh.load(mesh_path, force="scene", process=False)
    parts = []
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        part = scene.geometry[name].copy()
        if not isinstance(part, trimesh.Trimesh) or not len(part.faces) or not np.isfinite(part.vertices).all():
            raise CompletionError("SAM3D output contains empty or non-finite mesh geometry")
        part.apply_transform(transform)
        if part.visual.kind == "texture":
            part.visual = part.visual.to_color()
        if part.visual.kind not in ("face", "vertex"):
            raise CompletionError("SAM3D output lacks generated appearance")
        parts.append(part)
    if not parts:
        raise CompletionError("SAM3D output GLB has no mesh")
    mesh = trimesh.util.concatenate(parts)
    if mesh.area <= 0:
        raise CompletionError("SAM3D output has zero surface area")
    points, _, colors = trimesh.sample.sample_surface(mesh, count, sample_color=True, seed=seed)
    trimesh.PointCloud(points, colors=np.asarray(colors, dtype=np.uint8)).export(ply_path)
    loaded = trimesh.load(ply_path, process=False)
    if len(loaded.vertices) != count or not np.isfinite(loaded.vertices).all():
        raise CompletionError("generated visual point-cloud export failed validation")
    return {"vertices": len(mesh.vertices), "faces": len(mesh.faces), "watertight": bool(mesh.is_watertight), "bounds": mesh.bounds.tolist(), "visual_point_count": count, "visual_ply_kind": "generated_surface_points_not_gaussians", "color_sampling": "trimesh_face_colors; input_textures_converted_to_vertex_colors_when_present", "mesh_sha256": sha256(mesh_path), "visual_ply_sha256": sha256(ply_path)}


def run(args) -> dict:
    task = args.task_dir.resolve(strict=True)
    args.task_dir = task
    inputs = {name: input_artifact(args, name) for name in ("object_orbit_videos", "assembled_object_views", "isolated_object_ply", "object_descriptions", "cameras", "scene_gaussian_ply", "lifting_report")}
    envelope = read_json(local_path(task, args.inputs))["inputs"]
    input_hashes = {name: sha256(path) for name, path in inputs.items()}
    for name, path in inputs.items():
        if "sha256" in envelope[name]:
            checked_hash(task, path, envelope[name]["sha256"], "input " + name)
    orbits = list(unique_objects(read_json(inputs["object_orbit_videos"])["objects"], "orbits").values())
    assembled = unique_objects(read_json(inputs["assembled_object_views"])["objects"], "assembly")
    observed = unique_objects(read_json(inputs["isolated_object_ply"])["objects"], "isolated geometry")
    descriptions_by_object = validate_object_lineages(task, observed.values(), inputs["object_descriptions"])
    lifting = read_json(inputs["lifting_report"])
    if not orbits:
        raise CompletionError("no generated object orbit inputs")
    if args.views_per_orbit < 4:
        raise CompletionError("at least four views per orbit are needed for heldout registration")
    for obj in orbits:
        object_id = identifier(obj["object_id"])
        if object_id not in assembled or object_id not in observed:
            raise CompletionError("orbit object is missing assembled or observed geometry")
        validate_orbit_inputs(task, obj)
        validate_observed_input(task, object_id, obj, assembled[object_id], observed[object_id], lifting)
    for path in (args.stream3d_source / "streaming/runner.py", args.pipeline_config, args.stream3d_python, args.da3_python, args.sam3_python, args.sam3_checkpoint, args.da3_source / "depth_anything_3/api.py", args.da3_model, sam3_builder_path(args.sam3_source, args.sam3_backend)):
        if not path.exists():
            raise CompletionError(f"missing configured model/runtime resource: {path}")
    stage = local_path(task, "stages/geometry_completion", exists=False)
    stage.mkdir(parents=True, exist_ok=True)
    output_dir = Path(tempfile.mkdtemp(prefix="run-", dir=stage))
    config_path = output_dir / "sam3d.pipeline.yaml"
    config_report = prepare_pipeline_config(args.pipeline_config.resolve(), config_path)
    report_path = output_dir / "completion_candidates.json"
    commit = subprocess.run(["git", "-c", f"safe.directory={args.stream3d_source}", "-C", str(args.stream3d_source), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    report = {"schema_version": "1.0", "kind": "video2world-modeling.completion_candidates", "status": "running", "backend": "stream3d_sam3d", "configuration": config_report, "objects": [], "room_alignment_validated": False, "input_sha256": input_hashes, "runtime": {"source_root": str(args.stream3d_source), "source_commit": commit.stdout.strip() if commit.returncode == 0 else None, "runner_sha256": sha256(args.stream3d_source / "streaming/runner.py"), "interpreter": str(args.stream3d_python)}}
    outputs = []
    seen = set()
    try:
        for obj in orbits:
            object_id = identifier(obj["object_id"])
            if object_id in seen or object_id not in assembled or object_id not in observed:
                raise CompletionError("generated object IDs must map uniquely to assembled and lifted inputs")
            seen.add(object_id)
            categories = [item["category"] for item in descriptions_by_object[object_id] if isinstance(item.get("category"), str)]
            if not categories:
                raise CompletionError(f"no object category available to segment generated views: {object_id}")
            category = Counter(categories).most_common(1)[0][0]
            object_dir = output_dir / object_id
            object_dir.mkdir()
            candidate = {"object_id": object_id, "category": category, "status": "preprocessing", "backend": "stream3d_sam3d", "input_observed_geometry": observed[object_id]}
            report["objects"].append(candidate)
            frames_path, proposals_path, provenance = decode_orbits(args, obj, object_dir, category)
            sam_inputs = object_dir / "generated_sam_inputs.json"
            sam_outputs = object_dir / "generated_sam" / "provider-artifacts.json"
            write_json(sam_inputs, {"module": "component_segmentation", "inputs": {"frames_manifest": {"path": relative(task, frames_path), "evidence": "generated"}, "object_proposals": {"path": relative(task, proposals_path), "evidence": "generated"}, "object_descriptions": {"path": relative(task, inputs["object_descriptions"]), "evidence": "derived"}}})
            candidate["sam3_execution"] = execute(sam3_command(args, task, sam_inputs, sam_outputs), object_dir, environment(task), object_dir / "generated_sam.log")
            mask_record = read_json(sam_outputs)["outputs"]["component_mask_candidates"]
            masks_path = local_path(task, mask_record["path"])
            dataset = object_dir / "stream3d_inputs" / object_id
            request_path = object_dir / "generated_da3_request.json"
            write_json(request_path, {"object_id": object_id, "frames_manifest_path": relative(task, frames_path), "frames_manifest_sha256": sha256(frames_path), "masks_manifest_path": relative(task, masks_path), "masks_manifest_sha256": sha256(masks_path), "dataset_root": relative(task, dataset), "da3_source": str(args.da3_source), "da3_model": str(args.da3_model), "resolution": args.da3_resolution, "confidence_percentile": args.confidence_percentile})
            da3_command = [str(args.da3_python), str(Path(__file__).resolve()), "--task-dir", str(task), "--da3-request", str(request_path)]
            candidate["da3_execution"] = execute(da3_command, object_dir, environment(task, [args.da3_source, *args.da3_python_path]), object_dir / "generated_da3.log")
            generated_geometry_path = dataset / "input_manifest.json"
            generated_geometry = read_json(generated_geometry_path)
            if generated_geometry.get("conditioning_poses_used") is not False or len(generated_geometry.get("frames", [])) != len(provenance):
                raise CompletionError("generated preprocessing did not establish fresh complete camera/depth inputs")
            candidate["generated_camera_preflight"] = validate_generated_camera_chain(task, generated_geometry_path, object_dir / "sampled_frame_provenance.json", inputs["lifting_report"], args.seed)
            write_json(report_path, report)
            backend_output = object_dir / "stream3d_outputs"
            command = stream3d_command(args, dataset, backend_output, config_path, object_dir / "hydra")
            candidate["stream3d_execution"] = execute(command, object_dir, environment(task, [args.stream3d_source / "_compat", args.stream3d_source, *args.stream3d_python_path]), object_dir / "stream3d.log")
            final_chunk = len(chunk_starts(len(provenance))) - 1
            result_dir = backend_output / object_id / f"chunk_{final_chunk:04d}"
            mesh_path = result_dir / "result.glb"
            metadata_path = result_dir / "result_metadata.json"
            for path in (mesh_path, metadata_path, result_dir / "result_pose.npz", result_dir / "params.npz"):
                local_path(task, path)
            visual_path = object_dir / "generated_surface_points.ply"
            qa = sample_generated_mesh(mesh_path, visual_path, args.visual_points, args.seed)
            candidate.update({"status": "candidate", "mesh_path": relative(task, mesh_path), "visual_ply_path": relative(task, visual_path), "generated_frame_geometry_path": relative(task, generated_geometry_path), "backend_result_dir": relative(task, result_dir), "normalization_metadata_path": relative(task, metadata_path), "sampled_frame_provenance_path": relative(task, object_dir / "sampled_frame_provenance.json"), "qa": qa, "coordinate_frame": "object_local", "unit": "asset_units", "room_alignment": "not_estimated"})
            outputs.append({"object_id": object_id, "mesh_path": relative(task, mesh_path), "visual_ply_path": relative(task, visual_path), "coordinate_frame": "object_local", "unit": "asset_units", "room_alignment": "not_estimated", "evidence": "generated", "backend": "stream3d_sam3d", "mesh_sha256": qa["mesh_sha256"], "visual_ply_sha256": qa["visual_ply_sha256"], "visual_ply_kind": qa["visual_ply_kind"], "normalization_metadata_path": relative(task, metadata_path), "backend_pose_path": relative(task, result_dir / "result_pose.npz"), "backend_parameters_path": relative(task, result_dir / "params.npz"), "observed_anchor": observed[object_id].get("observed_anchor"), "observed_world_bounds": observed[object_id].get("world_bounds"), "observed_anchor_applied": False, **lineage_metadata(observed[object_id])})
            write_json(report_path, report)
        report["status"] = "generated_candidates_available"
        write_json(report_path, report)
        meshes_path = output_dir / "completed_object_meshes.json"
        write_json(meshes_path, {"schema_version": "1.0", "kind": "video2world-modeling.completed_object_meshes", "objects": outputs})
        registration_inputs = output_dir / "registration_inputs.json"
        registration_outputs = output_dir / "registration_outputs.json"
        registration_paths = {name: inputs[name] for name in ("isolated_object_ply", "cameras", "scene_gaussian_ply", "lifting_report")}
        registration_paths.update({"completed_object_meshes": meshes_path, "completion_candidates": report_path})
        write_json(registration_inputs, {"module": "object_registration", "inputs": {name: {"path": relative(task, path), "evidence": "generated" if name in {"completed_object_meshes", "completion_candidates"} else "derived"} for name, path in registration_paths.items()}})
        command = [str(args.registration_python or args.stream3d_python), str(Path(__file__).with_name("object_registration.py")), "--task-dir", str(task), "--inputs", str(registration_inputs), "--outputs", str(registration_outputs), "--seed", str(args.seed)]
        report["registration_execution"] = execute(command, output_dir, environment(task), output_dir / "registration.log")
        registered = read_json(registration_outputs)["outputs"]
        paths = {name: local_path(task, registered[name]["path"]) for name in ("completion_candidates", "completed_object_meshes")}
        for name, path in inputs.items():
            checked_hash(task, path, input_hashes[name], "source changed during completion: " + name)
        for obj in orbits:
            validate_orbit_inputs(task, obj)
            validate_observed_input(task, obj["object_id"], obj, assembled[obj["object_id"]], observed[obj["object_id"]], lifting)
        publish(args, paths)
        return read_json(paths["completion_candidates"])
    except (CompletionError, ImportError, OSError, ValueError, KeyError) as error:
        report.update({"status": "failed", "error": str(error)})
        write_json(report_path, report)
        raise


def main(argv=None) -> int:
    app = argparse.ArgumentParser(description=__doc__)
    app.add_argument("--task-dir", type=Path, required=True)
    app.add_argument("--inputs", type=Path)
    app.add_argument("--outputs", type=Path)
    for name in ("stream3d-source", "stream3d-python", "pipeline-config", "da3-python", "da3-source", "da3-model", "sam3-python", "sam3-source", "sam3-checkpoint"):
        app.add_argument("--" + name, type=Path)
    app.add_argument("--sam3-backend", choices=("sam3i", "sam3"), default="sam3i")
    app.add_argument("--sam3-instruction-stage", choices=("complex", "simple"), default="complex")
    app.add_argument("--stream3d-python-path", type=Path, action="append", default=[])
    app.add_argument("--da3-python-path", type=Path, action="append", default=[])
    app.add_argument("--registration-python", type=Path, help="CPU Open3D environment; defaults to the Stream3D interpreter")
    app.add_argument("--views-per-orbit", type=int, default=4)
    app.add_argument("--da3-resolution", type=int, default=504)
    app.add_argument("--confidence-percentile", type=float, default=20)
    app.add_argument("--sam3-confidence", type=float, default=.25)
    app.add_argument("--stage1-steps", type=int, default=4)
    app.add_argument("--stage2-steps", type=int, default=25)
    app.add_argument("--visual-points", type=int, default=100000)
    app.add_argument("--seed", type=int, default=0)
    app.add_argument("--ffprobe", default="ffprobe")
    app.add_argument("--ffmpeg", default="ffmpeg")
    app.add_argument("--da3-request", type=Path, help=argparse.SUPPRESS)
    args = app.parse_args(argv)
    if args.da3_request is None and any(getattr(args, name) is None for name in ("inputs", "outputs", "stream3d_source", "stream3d_python", "pipeline_config", "da3_python", "da3_source", "da3_model", "sam3_python", "sam3_source", "sam3_checkpoint")):
        app.error("all runtime/model roots and the unified --inputs/--outputs arguments are required")
    if not 4 <= args.views_per_orbit <= 20 or args.da3_resolution < 112 or not 0 <= args.confidence_percentile < 100 or not 0 <= args.sam3_confidence < 1 or min(args.stage1_steps, args.stage2_steps, args.visual_points) < 1:
        app.error("invalid view count, resolution, threshold, or sample count")
    try:
        if args.da3_request is not None:
            da3_worker(args.task_dir.resolve(strict=True), args.da3_request)
        else:
            run(args)
    except (CompletionError, ImportError, OSError, ValueError, KeyError) as error:
        print(f"geometry_completion failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
