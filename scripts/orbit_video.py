#!/usr/bin/env python3
"""Render three calibrated Gaussian orbits and refine each with official FixAnything."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from world_modeling.provider_io import input_artifact, local_path, publish, read_json, write_json


MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-480P"
FRAME_COUNT = 61


class OrbitError(RuntimeError):
    pass


def relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized(value, name: str):
    import numpy as np

    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all() or np.linalg.norm(vector) < 1e-10:
        raise OrbitError(f"{name} must be a finite nonzero 3-vector")
    return vector / np.linalg.norm(vector)


def rigid_pose(value):
    import numpy as np

    pose = np.asarray(value, dtype=np.float64)
    if pose.shape == (3, 4):
        pose = np.vstack((pose, [0, 0, 0, 1]))
    if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1]):
        raise OrbitError("camera world_to_camera must be finite 4x4")
    if not np.allclose(pose[:3, :3] @ pose[:3, :3].T, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-5):
        raise OrbitError("camera rotation must be rigid and right-handed")
    return pose


def gaussian_arrays(path: Path) -> dict:
    import numpy as np
    from plyfile import PlyData
    from scipy.special import expit

    source = PlyData.read(str(path))
    rows = source["vertex"].data
    names = set(rows.dtype.names or ())
    required = {"x", "y", "z", "opacity", *(f"scale_{i}" for i in range(3)), *(f"rot_{i}" for i in range(4)), *(f"f_dc_{i}" for i in range(3))}
    if not required <= names or not len(rows):
        raise OrbitError("observed_geometry must be a nonempty trained PGSR PLY with SH, scale, rotation, and opacity attributes")
    means = np.column_stack([rows[name] for name in ("x", "y", "z")]).astype(np.float32)
    log_scales = np.column_stack([rows[f"scale_{i}"] for i in range(3)]).astype(np.float64)
    scales = np.exp(log_scales).astype(np.float32)
    quaternions = np.column_stack([rows[f"rot_{i}"] for i in range(4)]).astype(np.float64)
    quaternion_norms = np.linalg.norm(quaternions, axis=1)
    if (quaternion_norms <= 1e-10).any():
        raise OrbitError("PGSR PLY contains zero-length rotation quaternion")
    quaternions = (quaternions / quaternion_norms[:, None]).astype(np.float32)
    opacity = expit(np.asarray(rows["opacity"], dtype=np.float64)).astype(np.float32)
    dc = np.column_stack([rows[f"f_dc_{i}"] for i in range(3)]).astype(np.float32)[:, None, :]
    extra_names = sorted((name for name in names if name.startswith("f_rest_")), key=lambda name: int(name.rsplit("_", 1)[1]))
    if extra_names != [f"f_rest_{index}" for index in range(len(extra_names))] or len(extra_names) % 3:
        raise OrbitError("PGSR SH attribute ordering is incomplete")
    basis_count = len(extra_names) // 3 + 1
    degree = round(basis_count ** 0.5) - 1
    if (degree + 1) ** 2 != basis_count or degree > 4:
        raise OrbitError("PGSR spherical harmonics do not form supported complete bands")
    if extra_names:
        rest = np.column_stack([rows[name] for name in extra_names]).astype(np.float32)
        rest = rest.reshape(len(rows), 3, basis_count - 1).transpose(0, 2, 1)
        colors = np.concatenate((dc, rest), axis=1)
    else:
        colors = dc
    arrays = {"means": means, "scales": scales, "quats": quaternions, "opacities": opacity, "colors": colors}
    if any(not np.isfinite(value).all() for value in arrays.values()) or (scales <= 0).any():
        raise OrbitError("PGSR parameters contain non-finite or nonpositive scales")
    center = (means.min(axis=0).astype(float) + means.max(axis=0).astype(float)) / 2
    radius = float(np.max(np.linalg.norm(means - center, axis=1) + 3 * scales.max(axis=1)))
    if not radius > 0:
        raise OrbitError("observed Gaussian geometry has zero support radius")
    return {**arrays, "sh_degree": degree, "center": center, "radius": radius, "source_attributes": list(rows.dtype.names)}


def orbit_basis(obj: dict, cameras: dict, center, up_override=None, direction_override=None) -> tuple:
    import numpy as np

    available = {str(item["frame_id"]): item for item in cameras.get("frames", [])}
    selected = [available[str(view["frame_id"])] for view in obj.get("views", []) if str(view["frame_id"]) in available]
    basis = obj.get("orbit_basis", {})
    if not selected and "reference_world_to_camera" in basis:
        selected = [{"frame_id": "manifest_reference", "world_to_camera": basis["reference_world_to_camera"]}]
    poses = [rigid_pose(record["world_to_camera"]) for record in selected]
    up_value = up_override if up_override is not None else basis.get("up_vector")
    if up_value is not None:
        up = normalized(up_value, "up_vector")
        up_method = "explicit_scene_vector"
    elif poses:
        camera_ups = np.array([-pose[1, :3] for pose in poses])
        mean = camera_ups.mean(axis=0)
        if np.linalg.norm(mean) < 0.5:
            raise OrbitError("observed camera up directions disagree; supply a scene up vector")
        up = normalized(mean, "camera_up")
        up_method = "mean_observed_camera_up_not_gravity_measurement"
    else:
        raise OrbitError("orbit up requires observed cameras or an explicit up_vector")
    direction_value = direction_override if direction_override is not None else basis.get("reference_direction")
    if direction_value is not None:
        direction = np.asarray(direction_value, dtype=np.float64)
    elif poses:
        first = poses[0]
        direction = -first[:3, :3].T @ first[:3, 3] - center
    else:
        raise OrbitError("orbit reference azimuth requires an observed camera or explicit reference_direction")
    reference = normalized(direction - up * np.dot(direction, up), "reference_direction projected onto orbit plane")
    return up, reference, {"up_vector": up.tolist(), "reference_direction": reference.tolist(), "up_method": up_method, "reference_frame_ids": [str(record["frame_id"]) for record in selected]}


def look_at(eye, target, up):
    import numpy as np

    forward = normalized(target - eye, "camera forward")
    right = normalized(np.cross(forward, up), "camera right")
    down = np.cross(forward, right)
    rotation = np.vstack((right, down, forward))
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = -rotation @ eye
    return result


def trajectory(center, radius: float, up, reference, elevation: float, width: int, height: int, vertical_fov: float, margin: float) -> dict:
    import numpy as np

    focal = height / (2 * np.tan(np.radians(vertical_fov) / 2))
    calibration = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    minimum_half_fov = min(np.arctan(width / (2 * focal)), np.arctan(height / (2 * focal)))
    distance = radius / np.sin(minimum_half_fov) * margin
    side = normalized(np.cross(up, reference), "orbit tangent")
    angle = np.radians(elevation)
    frames = []
    for index, azimuth in enumerate(np.linspace(0, 2 * np.pi, FRAME_COUNT)):
        radial = np.cos(azimuth) * reference + np.sin(azimuth) * side
        eye = center + distance * (np.cos(angle) * radial + np.sin(angle) * up)
        pose = look_at(eye, center, up)
        normalized_pose = pose.copy()
        normalized_pose[:3, 3] = (pose[:3, :3] @ center + pose[:3, 3]) / radius
        frames.append({"frame_id": f"{index:06d}", "azimuth_degrees": float(np.degrees(azimuth)), "camera_center_world": eye.tolist(), "intrinsics": calibration.tolist(), "world_to_camera": pose.tolist(), "normalized_object_to_camera": normalized_pose.tolist()})
    world_to_object = np.eye(4)
    world_to_object[:3, :3] /= radius
    world_to_object[:3, 3] = -center / radius
    return {
        "schema_version": "1.0", "frames": frames, "frame_count": FRAME_COUNT, "elevation_degrees": elevation, "azimuth_span_degrees": 360,
        "endpoint_policy": "frame_60_closes_frame_0_after_60_distinct_azimuths", "width": width, "height": height,
        "distance": distance, "near_plane": max(radius * 0.001, distance - radius * 1.1), "far_plane": distance + radius * 1.1,
        "camera_convention": "world_to_camera_opencv", "world_to_object": world_to_object.tolist(), "object_to_world": np.linalg.inv(world_to_object).tolist(),
        "object_center_world": center.tolist(), "object_support_radius": radius,
        "normalization": "translation_and_uniform_scale_only; original Gaussian rotations and SH basis are unchanged",
    }


THREE_AXIS_IDS = ("horizontal_front_back_left_right", "vertical_up_down_front_back", "vertical_left_right_up_down")


def three_axis_bases(up, reference):
    """Three orthonormal great-circle bases: horizontal, vertical front-back, vertical left-right.

    Each basis traces the equator of its own ``up`` vector, so ``trajectory`` is
    reused unchanged and every orbit remains a closed 360-degree circle of the
    same radius about the same object centre. This mirrors the trajectory
    definitions of the historical Holi-Spatial three-axis scene orbit
    (``horizontal_front_back_left_right`` / ``vertical_up_down_front_back`` /
    ``vertical_left_right_up_down``).
    """
    import numpy as np

    plane_up = normalized(up, "orbit up")
    front = normalized(reference - plane_up * float(np.dot(reference, plane_up)), "orbit reference projected onto orbit plane")
    side = normalized(np.cross(plane_up, front), "orbit tangent")
    return (
        (THREE_AXIS_IDS[0], plane_up, front),
        (THREE_AXIS_IDS[1], normalized(np.cross(front, plane_up), "vertical front-back axis"), front),
        (THREE_AXIS_IDS[2], front, side),
    )


def check_model_files(source_root: Path, model_dir: Path, lora_path: Path) -> dict:
    script = source_root / "scripts/run_inference.py"
    missing = []
    if not script.is_file():
        missing.append(str(script))
    if not lora_path.is_file() or not lora_path.stat().st_size:
        missing.append(str(lora_path))
    base = model_dir / MODEL_ID
    required = [base / name for name in ("models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")]
    shards = sorted(base.glob("diffusion_pytorch_model*.safetensors"))
    if not shards:
        missing.append(str(base / "diffusion_pytorch_model*.safetensors"))
    index_path = base / "diffusion_pytorch_model.safetensors.index.json"
    if index_path.is_file():
        index = read_json(index_path)
        shard_names = set(index.get("weight_map", {}).values())
        if not shard_names:
            missing.append("valid diffusion shard index")
        for name in shard_names:
            if not isinstance(name, str) or Path(name).name != name:
                raise OrbitError("invalid WAN shard reference")
            required.append(base / name)
    else:
        numbered = [re.fullmatch(r"diffusion_pytorch_model-(\d+)-of-(\d+)\.safetensors", shard.name) for shard in shards]
        numbered = [match for match in numbered if match]
        if numbered:
            totals = {int(match[2]) for match in numbered}
            if len(totals) != 1 or {int(match[1]) for match in numbered} != set(range(1, next(iter(totals)) + 1)):
                missing.append("complete numbered diffusion checkpoint shards")
    tokenizer = base / "google"
    tokenizer_files = list(tokenizer.rglob("*")) if tokenizer.is_dir() else []
    if not any(path.is_file() and path.suffix == ".model" for path in tokenizer_files):
        missing.append(str(tokenizer / "<tokenizer>/spiece.model"))
    for path in required + shards:
        if not path.is_file() or not path.stat().st_size:
            missing.append(str(path))
    if missing:
        raise OrbitError("FixAnything preflight missing local resources: " + "; ".join(missing))
    return {"source_root": str(source_root), "inference_script_sha256": sha256(script), "model_directory": str(model_dir), "lora_file": str(lora_path), "lora_sha256": sha256(lora_path), "base_model_files": [{"name": str(path.relative_to(base)), "size_bytes": path.stat().st_size} for path in sorted(set(required + shards + [path for path in tokenizer_files if path.is_file()]))], "downloads_disabled": True}


def fixanything_command(args, frames_dir: Path, output_dir: Path, seed: int) -> list[str]:
    clean_frames = getattr(args, "clean_frame_indices", "")
    if clean_frames:
        try:
            indices = [int(value) for value in clean_frames.split(",")]
        except ValueError as error:
            raise OrbitError("clean frame indices must be comma-separated integers") from error
        if len(set(indices)) != len(indices) or not all(0 <= index < FRAME_COUNT for index in indices):
            raise OrbitError("clean frame indices must be unique and within 0..60")
        clean_frames = " ".join(str(index) for index in indices)
    return [str(args.fix_python), str(args.source_root / "scripts/run_inference.py"), "--input", str(frames_dir), "--output_dir", str(output_dir), "--lora_path", str(args.lora_path), "--model_dir", str(args.model_dir), "--clean_frame_indices", clean_frames, "--num_frames", str(FRAME_COUNT), "--num_repeat_last", "4", "--num_inference_steps", str(args.num_inference_steps), "--height", str(args.height), "--width", str(args.width), "--fps", str(args.fps), "--seed", str(seed)]


def command_environment(task: Path, source_root: Path | None = None, extra_python_path: list[str] | None = None) -> dict:
    environment = os.environ.copy()
    cache = local_path(task, "stages/orbit_video/cache", exists=False)
    for name, subdirectory in (("TMPDIR", "tmp"), ("HF_HOME", "huggingface"), ("TORCH_HOME", "torch")):
        destination = cache / subdirectory
        destination.mkdir(parents=True, exist_ok=True)
        environment[name] = str(destination)
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    paths = ([str(source_root)] if source_root is not None else []) + list(extra_python_path or [])
    if environment.get("PYTHONPATH"):
        paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    return environment


def run_command(command: list[str], directory: Path, environment: dict, log_path: Path) -> dict:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=directory, env=environment, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
    if process.returncode:
        raise OrbitError(f"command failed with exit {process.returncode}; see {log_path}")
    return {"argv": command, "returncode": process.returncode}


def bound_file(task: Path, record: dict, label: str) -> Path:
    if not isinstance(record, dict) or not isinstance(record.get("sha256"), str):
        raise OrbitError(f"{label}: missing file hash binding")
    path = local_path(task, record.get("path", ""))
    if not path.is_file() or sha256(path) != record["sha256"]:
        raise OrbitError(f"{label}: file hash mismatch")
    return path


def validate_assembly(task: Path, assembled: dict, report: dict, cameras_path: Path) -> dict:
    if report.get("status") != "assembled_observed_views" or report.get("hidden_pixels_generated") is not False:
        raise OrbitError("orbit conditioning requires completed observed-view assembly")
    bindings = report.get("source_artifacts", {})
    sources = {role: bound_file(task, bindings.get(role), role) for role in ("cameras", "isolated_object_ply", "lifting_report")}
    if sources["cameras"] != cameras_path or local_path(task, assembled.get("cameras_path", "")) != cameras_path:
        raise OrbitError("assembly camera binding disagrees with orbit cameras")
    isolated = read_json(sources["isolated_object_ply"])
    lifting = read_json(sources["lifting_report"])
    if lifting.get("carve_performed") is not True or lifting.get("generated_geometry_used") is not False:
        raise OrbitError("orbit conditioning requires observed-only accepted lifting")
    groups = (assembled.get("objects"), isolated.get("objects"), report.get("objects"))
    indices = []
    for records in groups:
        if not isinstance(records, list) or not records:
            raise OrbitError("assembly and source manifests require nonempty object records")
        index = {}
        for record in records:
            name = record.get("object_id") if isinstance(record, dict) else None
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name in index:
                raise OrbitError("assembly has duplicate or invalid object IDs")
            index[name] = record
        indices.append(index)
    if not all(set(index) == set(indices[0]) for index in indices[1:]) or set(indices[0]) != set(lifting.get("accepted_object_ids", [])):
        raise OrbitError("assembly and lifting accepted objects disagree")
    files = {role: {"path": relative(task, path), "sha256": sha256(path)} for role, path in sources.items()}
    for object_id, obj in indices[0].items():
        observed = obj.get("observed_geometry", {})
        if observed != indices[1][object_id] or observed.get("association_status") != "geometrically_verified":
            raise OrbitError(f"{object_id}: assembly geometry differs from accepted lifting")
        record = {"path": observed.get("ply_path"), "sha256": observed.get("sha256")}
        path = bound_file(task, record, f"{object_id} Gaussian PLY")
        files[f"{object_id}:gaussians"] = {"path": relative(task, path), "sha256": record["sha256"]}
        frame_ids = [view.get("frame_id") for view in obj.get("views", [])]
        if not frame_ids or len(frame_ids) != len(set(frame_ids)) or frame_ids != indices[2][object_id].get("selected_frame_ids"):
            raise OrbitError(f"{object_id}: selected assembly views disagree")
        if not set(frame_ids) <= {item.get("frame_id") for item in observed.get("observations", [])}:
            raise OrbitError(f"{object_id}: orbit basis contains unverified observation frames")
    return files


def validate_video(path: Path, args) -> dict:
    command = [args.ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=width,height,nb_read_frames,avg_frame_rate", "-of", "json", str(path)]
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    if process.returncode:
        raise OrbitError(f"generated video cannot be decoded: {process.stderr}")
    streams = json.loads(process.stdout).get("streams", [])
    if len(streams) != 1:
        raise OrbitError("generated.mp4 must have one video stream")
    stream = streams[0]
    if (int(stream.get("width", 0)), int(stream.get("height", 0)), int(stream.get("nb_read_frames", 0))) != (args.width, args.height, FRAME_COUNT):
        raise OrbitError(f"generated.mp4 does not match requested raster and 61 decoded frames: {stream}")
    rate = stream.get("avg_frame_rate", "0/1").split("/")
    if len(rate) != 2 or float(rate[1]) == 0 or abs(float(rate[0]) / float(rate[1]) - args.fps) > 0.01:
        raise OrbitError("generated.mp4 frame rate differs from requested FPS")
    decoded = subprocess.run([args.ffmpeg, "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "framemd5", "-"], capture_output=True, text=True, check=False)
    if decoded.returncode:
        raise OrbitError("generated.mp4 failed full frame decode")
    checksums = [line.rsplit(",", 1)[-1].strip() for line in decoded.stdout.splitlines() if line and not line.startswith("#")]
    if len(checksums) != FRAME_COUNT or len(set(checksums)) < 2:
        raise OrbitError("generated.mp4 is incomplete or repeats one static frame")
    return {"decoded_frames": len(checksums), "unique_decoded_frames": len(set(checksums)), "width": args.width, "height": args.height, "fps": args.fps, "sha256": sha256(path)}


def fit_conditioning_framing(task: Path, request: dict, *, margin: float = 1.1, alpha_threshold: float = 0.1) -> tuple[dict, dict]:
    import copy
    import numpy as np
    from PIL import Image

    if not np.isfinite(margin) or margin <= 1:
        raise OrbitError("conditioning framing margin must be finite and greater than one")
    if not 0 < alpha_threshold < 1:
        raise OrbitError("conditioning framing alpha threshold must be in (0, 1)")
    maximum_ratio, samples = 0., []
    for orbit in request["orbits"]:
        path = bound_file(task, {"path": orbit["cameras_path"], "sha256": orbit["conditioning_cameras_sha256"]}, "conditioning cameras")
        cameras = read_json(path)
        width, height = cameras["width"], cameras["height"]
        for frame in cameras["frames"]:
            alpha_path = local_path(task, orbit["alpha_dir"]) / f"{frame['frame_id']}.png"
            with Image.open(alpha_path) as source:
                if source.size != (width, height):
                    raise OrbitError("conditioning alpha raster disagrees with cameras")
                alpha = np.asarray(source.convert("L")) / 255.
            # One common focal change preserves relative sizes and poses across
            # all elevations. The crop extent must come from the object body, not
            # from faint outlier tails: at alpha 0.1 a few far-flung faint splats
            # set the extent and the object ends up occupying ~10% of the raster.
            support = alpha > alpha_threshold
            ys, xs = np.nonzero(support)
            if len(xs) < 32:
                support = alpha > 0.1
                ys, xs = np.nonzero(support)
            if len(xs) < 32:
                raise OrbitError("conditioning has insufficient visible support for framing")
            ratio = max((np.abs(xs + .5 - width / 2)).max() / (width / 2),
                        (np.abs(ys + .5 - height / 2)).max() / (height / 2))
            maximum_ratio = max(maximum_ratio, float(ratio))
            samples.append({"orbit_id": orbit["orbit_id"], "frame_id": frame["frame_id"],
                            "alpha_path": relative(task, alpha_path), "alpha_sha256": sha256(alpha_path),
                            "visible_pixels": len(xs), "framing_ratio": float(ratio)})
    if not maximum_ratio > 0:
        raise OrbitError("conditioning framing support is empty")
    zoom = min(4., 1. / (maximum_ratio * margin))
    qa = {"method": "shared_calibrated_intrinsic_zoom_from_rendered_alpha", "alpha_threshold": alpha_threshold,
          "margin": margin, "zoom": zoom, "maximum_zoom": 4., "gaussians_modified": False,
          "poses_modified": False, "samples": samples, "faint_tails_may_leave_raster": True}
    fitted = copy.deepcopy(request)
    fitted["report_path"] = relative(task, local_path(task, request["report_path"]).with_name("framed_render_report.json"))
    for orbit in fitted["orbits"]:
        original_path = local_path(task, orbit["cameras_path"])
        cameras = read_json(original_path)
        width, height = cameras["width"], cameras["height"]
        transform = np.array([[zoom, 0, width / 2 * (1 - zoom)], [0, zoom, height / 2 * (1 - zoom)], [0, 0, 1]])
        for frame in cameras["frames"]:
            frame["intrinsics"] = (transform @ np.asarray(frame["intrinsics"], dtype=float)).tolist()
        cameras["framing"] = {key: value for key, value in qa.items() if key != "samples"}
        cameras["framing"].update({"source_cameras_path": relative(task, original_path),
                                    "source_cameras_sha256": orbit["conditioning_cameras_sha256"],
                                    "source_image_to_framed": transform.tolist(), "images_rerendered_not_resampled": True})
        directory = original_path.parent / "framed"
        directory.mkdir()
        camera_path = directory / "conditioning_cameras.json"
        write_json(camera_path, cameras)
        orbit.update({"cameras_path": relative(task, camera_path), "conditioning_cameras_sha256": sha256(camera_path),
                      "frames_dir": relative(task, directory / "conditioning_frames"),
                      "alpha_dir": relative(task, directory / "conditioning_alpha")})
    return fitted, qa


def validate_rendered_orbit(task: Path, orbit: dict, rendered: dict) -> None:
    bound_file(task, {"path": orbit["cameras_path"], "sha256": orbit["conditioning_cameras_sha256"]}, "conditioning cameras")
    matching = [item for item in rendered.get("orbits", []) if item.get("orbit_id") == orbit["orbit_id"]]
    if len(matching) != 1:
        raise OrbitError("render report requires exactly one entry for each orbit")
    frames = matching[0].get("frames", [])
    if [frame.get("frame_id") for frame in frames] != [f"{index:06d}" for index in range(FRAME_COUNT)]:
        raise OrbitError("render report frame IDs must cover the ordered 61-frame orbit")
    directory = local_path(task, orbit["frames_dir"])
    actual = {path.name for path in directory.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}}
    if actual != {f"{index:06d}.png" for index in range(FRAME_COUNT)}:
        raise OrbitError("FixAnything conditioning directory differs from the exact 61 reported frames")
    for frame in frames:
        for kind, directory_key in (("image", "frames_dir"), ("alpha", "alpha_dir")):
            path = bound_file(task, {"path": frame.get(f"{kind}_path"), "sha256": frame.get(f"{kind}_sha256")}, f"conditioning {kind}")
            if path != local_path(task, orbit[directory_key]) / f"{frame['frame_id']}.png":
                raise OrbitError("render report frame path differs from requested conditioning directory")


def render_worker(task: Path, request_path: Path) -> dict:
    import numpy as np
    from PIL import Image
    import torch
    import gsplat

    if not torch.cuda.is_available():
        raise OrbitError("gsplat conditioning renderer requires the configured CUDA environment")
    request = read_json(local_path(task, request_path))
    gaussian_path = local_path(task, request["gaussian_ply"])
    if request.get("gaussian_sha256") is not None:
        bound_file(task, {"path": request["gaussian_ply"], "sha256": request["gaussian_sha256"]}, "render Gaussian input")
    arrays = gaussian_arrays(gaussian_path)
    radius_clip = float(request.get("radius_clip", 0.0))
    dc_only = bool(request.get("dc_only", False))
    background = str(request.get("background", "white"))
    framing_alpha = 0.1
    if background not in {"white", "black"}:
        raise OrbitError("conditioning background must be white or black")
    if not (radius_clip >= 0) or not (radius_clip == radius_clip):
        raise OrbitError("conditioning radius_clip must be a finite nonnegative number")
    colors = arrays["colors"][:, :1, :] if dc_only else arrays["colors"]
    sh_degree = 0 if dc_only else arrays["sh_degree"]
    tensors = {name: torch.as_tensor(arrays[name], dtype=torch.float32, device="cuda") for name in ("means", "quats", "scales", "opacities")}
    tensors["colors"] = torch.as_tensor(colors, dtype=torch.float32, device="cuda")
    results = []
    with torch.inference_mode():
        for orbit in request["orbits"]:
            camera_binding = {"path": orbit["cameras_path"], "sha256": orbit["conditioning_cameras_sha256"]}
            calibration = read_json(bound_file(task, camera_binding, "renderer cameras"))
            # The edge guard must use the same alpha threshold that defined the
            # crop: faint Gaussian tails are documented to leave the raster.
            framing_alpha = float(calibration.get("framing", {}).get("alpha_threshold", 0.1))
            frames_dir = local_path(task, orbit["frames_dir"], exists=False)
            alpha_dir = local_path(task, orbit["alpha_dir"], exists=False)
            frames_dir.mkdir(parents=True)
            alpha_dir.mkdir(parents=True)
            frames = []
            for frame in calibration["frames"]:
                # The deployed packed CUDA wrapper loses the camera axis for
                # backgrounds. Dense projection keeps the single-view ABI.
                rendered, alphas, _ = gsplat.rasterization(
                    **tensors,
                    viewmats=torch.as_tensor([frame["world_to_camera"]], dtype=torch.float32, device="cuda"),
                    Ks=torch.as_tensor([frame["intrinsics"]], dtype=torch.float32, device="cuda"),
                    width=calibration["width"], height=calibration["height"],
                    near_plane=calibration["near_plane"], far_plane=calibration["far_plane"],
                    backgrounds=(torch.ones((1, 3), dtype=torch.float32, device="cuda") if background == "white"
                                 else torch.zeros((1, 3), dtype=torch.float32, device="cuda")),
                    sh_degree=sh_degree, render_mode="RGB", rasterize_mode="classic", packed=False,
                    radius_clip=radius_clip,
                )
                rgb = rendered[0].detach().cpu().numpy()
                alpha = alphas[0, :, :, 0].detach().cpu().numpy()
                if not np.isfinite(rgb).all() or not np.isfinite(alpha).all():
                    raise OrbitError("gsplat returned non-finite rendered pixels")
                foreground = int((alpha > 0.01).sum())
                if foreground < 32 or foreground > alpha.size * 0.95:
                    raise OrbitError(f"{orbit['orbit_id']} frame {frame['frame_id']}: empty or clipped object framing")
                if "framing" in calibration and any((edge > framing_alpha).any() for edge in (alpha[0], alpha[-1], alpha[:, 0], alpha[:, -1])):
                    raise OrbitError(f"{orbit['orbit_id']} frame {frame['frame_id']}: visible object reaches fitted raster edge")
                frame_path, alpha_path = frames_dir / f"{frame['frame_id']}.png", alpha_dir / f"{frame['frame_id']}.png"
                Image.fromarray(np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(frame_path)
                Image.fromarray(np.rint(np.clip(alpha, 0, 1) * 255).astype(np.uint8), mode="L").save(alpha_path)
                frames.append({"frame_id": frame["frame_id"], "image_path": relative(task, frame_path), "image_sha256": sha256(frame_path), "alpha_path": relative(task, alpha_path), "alpha_sha256": sha256(alpha_path), "foreground_pixels": foreground})
            bound_file(task, camera_binding, "renderer cameras")
            results.append({"orbit_id": orbit["orbit_id"], "frames": frames})
    result = {"renderer": "gsplat.rasterization", "source_gaussian_sha256": sha256(gaussian_path), "gaussian_count": len(arrays["means"]), "source_attributes": arrays["source_attributes"], "sh_degree": arrays["sh_degree"], "gaussian_rotation_transform": "none", "orbits": results}
    write_json(local_path(task, request["report_path"], exists=False), result)
    return result


def run(args) -> dict:
    task = args.task_dir.resolve(strict=True)
    args.task_dir = task
    assembled_path = input_artifact(args, "assembled_object_views")
    assembly_report_path = input_artifact(args, "assembly_report")
    assembled = read_json(assembled_path)
    assembly_report = read_json(assembly_report_path)
    objects = assembled.get("objects")
    if not isinstance(objects, list) or not objects:
        raise OrbitError("assembled_object_views requires nonempty objects")
    stage = local_path(task, "stages/orbit_video", exists=False)
    stage.mkdir(parents=True, exist_ok=True)
    output_dir = Path(tempfile.mkdtemp(prefix="run-", dir=stage))
    report_path = output_dir / "orbit_video_report.json"
    report = {"schema_version": "1.0", "kind": "video2world-modeling.orbit_video_report", "status": "preflight", "fixanything_invoked": False, "objects": []}
    try:
        resources = check_model_files(args.source_root, args.model_dir, args.lora_path)
    except OrbitError as error:
        report.update({"status": "blocked_model_preflight", "error": str(error)})
        write_json(report_path, report)
        raise
    report["resources"] = resources
    cameras_value = args.cameras or assembled.get("cameras_path")
    if cameras_value is None:
        envelope = read_json(local_path(task, args.inputs))["inputs"]
        if "cameras" in envelope:
            cameras_value = envelope["cameras"]["path"]
    ids, output_objects = set(), []
    try:
        if cameras_value is None:
            raise OrbitError("assembly must bind source cameras")
        cameras_path = local_path(task, cameras_value)
        bound_inputs = validate_assembly(task, assembled, assembly_report, cameras_path)
        bound_inputs.update({"assembled_object_views": {"path": relative(task, assembled_path), "sha256": sha256(assembled_path)},
                             "assembly_report": {"path": relative(task, assembly_report_path), "sha256": sha256(assembly_report_path)}})
        report["source_artifacts"] = bound_inputs
        cameras = read_json(cameras_path)
        for object_index, obj in enumerate(objects):
            object_id = obj.get("object_id")
            if not isinstance(object_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", object_id) or object_id in ids:
                raise OrbitError("assembled objects must have unique safe object_id values")
            ids.add(object_id)
            observed = obj.get("observed_geometry", {})
            if observed.get("association_status") != "geometrically_verified":
                raise OrbitError(f"{object_id}: observed_geometry lacks accepted geometric instance association")
            override = getattr(args, "conditioning_ply", None)
            if override is not None:
                # Explicit override: the recipe that reached Stream3D was
                # conditioned on a complete object asset, not on this task's
                # incomplete lifting. The asset is copied into the task so every
                # input stays task-local, and it is recorded as its own provenance
                # role -- never as this task's observed geometry.
                import shutil

                requested = Path(override).expanduser().resolve(strict=True)
                if not requested.is_file():
                    raise OrbitError(f"conditioning override is not a file: {requested}")
                destination = task / "inputs" / "conditioning" / f"{sha256(requested)[:16]}_{requested.name}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.is_file() or sha256(destination) != sha256(requested):
                    shutil.copy2(requested, destination)
                gaussian_path = destination
                arrays = gaussian_arrays(gaussian_path)
                conditioning = {"role": "explicit_override_not_this_task_observation",
                                "requested_path": str(requested), "path": relative(task, gaussian_path),
                                "sha256": sha256(gaussian_path), "gaussian_count": len(arrays["means"]),
                                "observed_gaussian_count": observed.get("gaussian_count"),
                                "observed_geometry_sha256": observed.get("sha256")}
            else:
                gaussian_path = local_path(task, observed["ply_path"])
                arrays = gaussian_arrays(gaussian_path)
                if len(arrays["means"]) != observed.get("gaussian_count"):
                    raise OrbitError(f"{object_id}: Gaussian count disagrees with accepted lifting")
                conditioning = {"role": "this_task_observed_object", "path": relative(task, gaussian_path),
                                "sha256": observed["sha256"], "gaussian_count": len(arrays["means"])}
            center, radius = arrays["center"], arrays["radius"]
            up, reference, basis_report = orbit_basis(obj, cameras, center, args.up_vector, args.reference_direction)
            object_dir = output_dir / object_id
            object_dir.mkdir()
            render_report_path = object_dir / "render_report.json"
            request = {"gaussian_ply": relative(task, gaussian_path), "gaussian_sha256": conditioning["sha256"], "report_path": relative(task, render_report_path), "radius_clip": getattr(args, "conditioning_radius_clip", 0.0), "dc_only": getattr(args, "conditioning_dc_only", False), "background": getattr(args, "background", "white"), "trajectory_mode": getattr(args, "trajectory_mode", "elevations"), "orbits": []}
            if request["trajectory_mode"] == "three_axis":
                orbit_specs = [(name, axis_up, axis_reference, 0.0) for name, axis_up, axis_reference in three_axis_bases(up, reference)]
            else:
                orbit_specs = [(f"elevation_{value:+g}_degrees", up, reference, float(value)) for value in args.elevations]
            for index, (trajectory_id, axis_up, axis_reference, elevation) in enumerate(orbit_specs):
                orbit_id = f"orbit_{index:02d}"
                orbit_dir = object_dir / orbit_id
                orbit_dir.mkdir()
                cameras_path = orbit_dir / "conditioning_cameras.json"
                camera_manifest = trajectory(center, radius, axis_up, axis_reference, elevation, args.width, args.height, args.vertical_fov, args.framing_margin)
                orbit_basis_record = {**basis_report, "trajectory_id": trajectory_id,
                                      "up_vector": list(map(float, axis_up)),
                                      "reference_direction": list(map(float, axis_reference))}
                camera_manifest.update({"orbit_id": orbit_id, "object_id": object_id, "trajectory_id": trajectory_id,
                                        "coordinate_frame": observed.get("coordinate_frame", "scene_world"), "units": observed.get("units", "unspecified"), "basis": orbit_basis_record})
                write_json(cameras_path, camera_manifest)
                request["orbits"].append({"orbit_id": orbit_id, "elevation_degrees": elevation, "cameras_path": relative(task, cameras_path), "conditioning_cameras_sha256": sha256(cameras_path), "frames_dir": relative(task, orbit_dir / "conditioning_frames"), "alpha_dir": relative(task, orbit_dir / "conditioning_alpha")})
            request_path = object_dir / "render_request.json"
            write_json(request_path, request)
            render_log = object_dir / "render.log"
            renderer_argv = [str(args.render_python), str(Path(__file__).resolve()), "--task-dir", str(task), "--render-request", str(request_path)]
            render_execution = run_command(renderer_argv, object_dir, command_environment(task, extra_python_path=args.render_python_path), render_log)
            rendered = read_json(render_report_path)
            if rendered.get("source_gaussian_sha256") != conditioning["sha256"]:
                raise OrbitError("Gaussian input changed during rendering")
            if len(rendered.get("orbits", [])) != 3 or any(len(item.get("frames", [])) != FRAME_COUNT for item in rendered["orbits"]):
                raise OrbitError("conditioning renderer did not produce three complete 61-frame clips")
            for orbit in request["orbits"]:
                validate_rendered_orbit(task, orbit, rendered)
            request, framing = fit_conditioning_framing(task, request, margin=args.framing_margin, alpha_threshold=getattr(args, "framing_alpha", 0.1))
            write_json(object_dir / "framing_report.json", framing)
            framed_request_path = object_dir / "framed_render_request.json"
            write_json(framed_request_path, request)
            framed_log = object_dir / "framed_render.log"
            framed_argv = [str(args.render_python), str(Path(__file__).resolve()), "--task-dir", str(task), "--render-request", str(framed_request_path)]
            framed_execution = run_command(framed_argv, object_dir, command_environment(task, extra_python_path=args.render_python_path), framed_log)
            initial_render_report_path = render_report_path
            render_report_path = local_path(task, request["report_path"])
            rendered = read_json(render_report_path)
            if rendered.get("source_gaussian_sha256") != conditioning["sha256"] or len(rendered.get("orbits", [])) != 3 or any(len(item.get("frames", [])) != FRAME_COUNT for item in rendered["orbits"]):
                raise OrbitError("framed conditioning did not preserve the three complete observed-geometry orbits")
            object_report = {"object_id": object_id, "basis": basis_report, "conditioning_geometry": conditioning, "render_execution": render_execution, "render_log_path": relative(task, render_log),
                             "initial_render_report_path": relative(task, initial_render_report_path), "framing_report_path": relative(task, object_dir / "framing_report.json"),
                             "framed_render_execution": framed_execution, "framed_render_log_path": relative(task, framed_log),
                             "render_report_path": relative(task, render_report_path), "orbits": []}
            report["objects"].append(object_report)
            clips, generated_hashes = [], set()
            for index, orbit in enumerate(request["orbits"]):
                validate_rendered_orbit(task, orbit, rendered)
                orbit_dir = local_path(task, orbit["cameras_path"]).parent
                generated_dir = orbit_dir / "fixanything"
                generated_dir.mkdir()
                log_path = orbit_dir / "fixanything.log"
                command = fixanything_command(args, local_path(task, orbit["frames_dir"]), generated_dir, args.seed + object_index * 3 + index)
                report["fixanything_invoked"] = True
                execution = run_command(command, object_dir, command_environment(task, args.source_root), log_path)
                validate_rendered_orbit(task, orbit, rendered)
                if getattr(args, "orbit_source", "fixanything") == "conditioning":
                    # Explicit operator choice: publish the conditioning render itself
                    # as the orbit video and skip FixAnything. That render is a direct
                    # image of the conditioning Gaussian asset, so its camera geometry
                    # is the conditioning trajectory rather than a synthesis hypothesis.
                    video_path = orbit_dir / "conditioning.mp4"
                    run_command([str(args.ffmpeg), "-nostdin", "-y", "-loglevel", "error", "-framerate", str(args.fps),
                                 "-i", str(local_path(task, orbit["frames_dir"]) / "%06d.png"),
                                 "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video_path)],
                                orbit_dir, command_environment(task), orbit_dir / "conditioning_encode.log")
                    if not video_path.is_file():
                        raise OrbitError("could not encode the conditioning frames into an orbit video")
                else:
                    video_path = generated_dir / "generated.mp4"
                    if not video_path.is_file():
                        raise OrbitError("official FixAnything did not write generated.mp4")
                video_report = validate_video(video_path, args)
                if video_report["sha256"] in generated_hashes:
                    raise OrbitError("separate elevations produced identical output files; repeated orbit videos are not accepted")
                generated_hashes.add(video_report["sha256"])
                object_report["orbits"].append({"orbit_id": orbit["orbit_id"], "execution": execution, "log_path": relative(task, log_path), "video_validation": video_report})
                clips.append({**orbit, "video_path": relative(task, video_path), "video_sha256": video_report["sha256"], "frame_count": FRAME_COUNT, "width": args.width, "height": args.height, "fps": args.fps, "azimuth_span_degrees": 360, "evidence": ("direct_asset_render" if getattr(args, "orbit_source", "fixanything") == "conditioning" else "conditioned_synthesis"), "camera_pose_status": ("conditioning_trajectory_matches_source_render_asset" if getattr(args, "orbit_source", "fixanything") == "conditioning" else "conditioning_trajectory_not_verified_for_generated_pixels")})
                write_json(report_path, report)
            output_objects.append({"object_id": object_id, "orbits": clips, "render_report_path": relative(task, render_report_path), "observed_geometry": observed, "conditioning_geometry": conditioning, "basis": basis_report, "geometry_completion_required": True})
        for role, binding in bound_inputs.items():
            bound_file(task, binding, role)
        report["status"] = "passed_generation_and_decode"
        report["multiview_geometry_validated"] = False
        manifest_path = output_dir / "object_orbit_videos.json"
        manifest = {"schema_version": "1.0", "kind": "video2world-modeling.object_orbit_videos", "objects": output_objects, "source_assembly_path": relative(task, assembled_path)}
        write_json(manifest_path, manifest)
        write_json(report_path, report)
        publish(args, {"object_orbit_videos": manifest_path, "orbit_video_report": report_path})
        return manifest
    except (OrbitError, OSError, ValueError, KeyError) as error:
        report.update({"status": "failed", "error": str(error)})
        write_json(report_path, report)
        raise


def main(argv=None) -> int:
    app = argparse.ArgumentParser(description=__doc__)
    app.add_argument("--task-dir", type=Path, required=True)
    app.add_argument("--inputs", type=Path)
    app.add_argument("--outputs", type=Path)
    app.add_argument("--source-root", type=Path)
    app.add_argument("--model-dir", type=Path)
    app.add_argument("--lora-path", type=Path)
    app.add_argument("--fix-python", type=Path, default=Path(sys.executable))
    app.add_argument("--render-python", type=Path, default=Path(sys.executable))
    app.add_argument("--render-python-path", action="append", default=[])
    app.add_argument("--cameras", type=Path)
    app.add_argument("--up-vector", type=float, nargs=3)
    app.add_argument("--reference-direction", type=float, nargs=3)
    app.add_argument("--elevations", type=float, nargs=3, default=[10., 25., 40.])
    app.add_argument("--trajectory-mode", choices=("elevations", "three_axis"), default="elevations",
                     help="elevations: one up axis at three elevations (deployed). three_axis: horizontal plus two orthogonal vertical great circles.")
    app.add_argument("--background", choices=("white", "black"), default="white",
                     help="conditioning raster background; the historical Holi three-axis recipe rendered on black")
    app.add_argument("--orbit-source", choices=("fixanything", "conditioning"), default="fixanything",
                     help="conditioning publishes the conditioning render itself as the orbit video and skips FixAnything")
    app.add_argument("--conditioning-ply", type=Path,
                     help="Explicit conditioning Gaussian PLY override (recorded as an override, never as this task's observation)")
    app.add_argument("--framing-alpha", type=float, default=0.1,
                     help="Alpha threshold that defines the framing crop; raising it excludes faint Gaussian tails")
    app.add_argument("--width", type=int, default=832)
    app.add_argument("--height", type=int, default=480)
    app.add_argument("--vertical-fov", type=float, default=40)
    app.add_argument("--framing-margin", type=float, default=1.1)
    app.add_argument("--conditioning-radius-clip", type=float, default=0.0,
                     help="gsplat radius_clip in pixels when rendering conditioning frames; 0 keeps the deployed behaviour")
    app.add_argument("--conditioning-dc-only", action="store_true",
                     help="render conditioning frames from DC colour only instead of full SH")
    app.add_argument("--num-inference-steps", type=int, default=10)
    app.add_argument("--fps", type=int, default=15)
    app.add_argument("--seed", type=int, default=1)
    app.add_argument("--clean-frame-indices", default="", help="Trusted conditioning frames preserved by FixAnything; historical bed recipe used 0,60")
    app.add_argument("--ffprobe", default="ffprobe")
    app.add_argument("--ffmpeg", default="ffmpeg")
    app.add_argument("--render-request", type=Path, help=argparse.SUPPRESS)
    args = app.parse_args(argv)
    if args.render_request is None and any(getattr(args, name) is None for name in ("inputs", "outputs", "source_root", "model_dir", "lora_path")):
        app.error("--inputs, --outputs, --source-root, --model-dir, and --lora-path are required")
    if args.trajectory_mode == "elevations" and (len(set(args.elevations)) != 3 or any(abs(value) >= 80 for value in args.elevations)):
        app.error("three distinct elevations strictly between -80 and 80 degrees are required")
    if not 0 < args.framing_alpha < 1:
        app.error("framing alpha threshold must be in (0, 1)")
    if min(args.width, args.height) < 64 or args.width % 16 or args.height % 16 or not 10 <= args.vertical_fov <= 90 or not args.framing_margin > 1:
        app.error("raster must be at least 64 and divisible by 16; FOV 10..90; framing margin >1")
    if args.num_inference_steps < 1 or args.fps < 1:
        app.error("inference steps and FPS must be positive")
    try:
        if args.render_request is not None:
            render_worker(args.task_dir.resolve(strict=True), args.render_request)
        else:
            run(args)
    except (OrbitError, ImportError, OSError, ValueError, KeyError) as error:
        print(f"orbit_video failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
