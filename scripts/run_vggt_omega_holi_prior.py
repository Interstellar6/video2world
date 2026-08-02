#!/usr/bin/env python3
"""Run official VGGT-Omega and materialize a PGSR-compatible scene prior."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

VERIFIED_SOURCE_REPOSITORY = "https://github.com/facebookresearch/vggt-omega.git"
VERIFIED_SOURCE_COMMIT = "39a0cb8af88554f15ddcb5354cd52bde588fa014"
VERIFIED_MODEL_REPOSITORY = "facebook/VGGT-Omega"
VERIFIED_MODEL_REVISION = "05654241adc2f218dfb089c373a011f8a7040576"
VERIFIED_CHECKPOINT_SHA256 = "c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934"
KIND = "video2world.vggt_omega_holi_prior_receipt"


class VGGTOmegaPriorError(RuntimeError):
    """Raised when VGGT-Omega cannot produce an auditable Holi prior."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise VGGTOmegaPriorError(f"artifact is missing or empty: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def git_source_provenance(source_root: Path, expected_commit: str) -> dict[str, Any]:
    root = source_root.expanduser().resolve()
    if not root.is_dir():
        raise VGGTOmegaPriorError(f"VGGT-Omega source does not exist: {root}")
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    if commit != expected_commit:
        raise VGGTOmegaPriorError(
            f"VGGT-Omega source commit mismatch: expected {expected_commit}, got {commit}"
        )
    remote = subprocess.run(
        ["git", "-C", str(root), "config", "--get", "remote.origin.url"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    return {
        "repository": remote or VERIFIED_SOURCE_REPOSITORY,
        "path": str(root),
        "commit": commit,
        "expected_commit": expected_commit,
        "verified": True,
    }


def resize_intrinsics(
    intrinsics: np.ndarray,
    *,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> np.ndarray:
    """Scale pinhole intrinsics from source (height, width) to target size."""

    source_h, source_w = source_size
    target_h, target_w = target_size
    if min(source_h, source_w, target_h, target_w) <= 0:
        raise VGGTOmegaPriorError("image dimensions must be positive")
    value = np.asarray(intrinsics, dtype=np.float64).copy()
    if value.ndim != 3 or value.shape[1:] != (3, 3):
        raise VGGTOmegaPriorError("intrinsics must have shape (N, 3, 3)")
    value[:, 0, :] *= target_w / source_w
    value[:, 1, :] *= target_h / source_h
    value[:, 2, :] = np.array([0.0, 0.0, 1.0])
    return value.astype(np.float32)


def common_intrinsics(
    intrinsics: np.ndarray, *, max_relative_spread: float
) -> tuple[np.ndarray, float]:
    """Collapse predicted per-frame focal lengths for the legacy ScanNet loader."""

    value = np.asarray(intrinsics, dtype=np.float64)
    if value.ndim != 3 or value.shape[1:] != (3, 3) or not np.isfinite(value).all():
        raise VGGTOmegaPriorError("predicted intrinsics are invalid")
    focal = value[:, (0, 1), (0, 1)]
    median = np.median(focal, axis=0)
    if np.any(median <= 0):
        raise VGGTOmegaPriorError("predicted focal lengths must be positive")
    spread = float(np.max(np.abs(focal - median) / median))
    if spread > max_relative_spread:
        raise VGGTOmegaPriorError(
            "VGGT-Omega focal-length spread exceeds the single-intrinsic ScanNet adapter "
            f"limit: {spread:.6f} > {max_relative_spread:.6f}"
        )
    height = float(np.median(value[:, 1, 2]) * 2.0)
    width = float(np.median(value[:, 0, 2]) * 2.0)
    common = np.array(
        [
            [median[0], 0.0, width / 2.0],
            [0.0, median[1], height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return common, spread


def depth_edge_mask(depth: np.ndarray, *, relative_threshold: float) -> np.ndarray:
    value = np.asarray(depth, dtype=np.float32)
    if value.ndim != 2:
        raise VGGTOmegaPriorError("depth must be two-dimensional")
    padded = np.pad(value, ((1, 1), (1, 1)), mode="edge")
    local_min = np.full_like(value, np.inf)
    local_max = np.full_like(value, -np.inf)
    for y in range(3):
        for x in range(3):
            window = padded[y : y + value.shape[0], x : x + value.shape[1]]
            local_min = np.minimum(local_min, window)
            local_max = np.maximum(local_max, window)
    jump = (local_max - local_min) / np.maximum(np.abs(value), 1e-6)
    return jump > relative_threshold


def unproject_frame(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    world_to_camera: np.ndarray,
    pixel_indices: np.ndarray,
) -> np.ndarray:
    """Unproject selected flattened pixels into the predicted world frame."""

    value = np.asarray(depth, dtype=np.float64)
    indices = np.asarray(pixel_indices, dtype=np.int64)
    width = value.shape[1]
    y = indices // width
    x = indices % width
    z = value.reshape(-1)[indices]
    pixels = np.stack([x, y, np.ones_like(x)], axis=0).astype(np.float64)
    camera = np.linalg.inv(np.asarray(intrinsics, dtype=np.float64)) @ pixels
    camera *= z[None, :]
    w2c = np.asarray(world_to_camera, dtype=np.float64)
    rotation = w2c[:3, :3]
    translation = w2c[:3, 3]
    world = rotation.T @ (camera - translation[:, None])
    return world.T.astype(np.float32)


def write_binary_point_cloud(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    vertices = np.asarray(points, dtype=np.float32)
    rgb = np.asarray(colors, dtype=np.uint8)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) == 0:
        raise VGGTOmegaPriorError("point cloud must contain Nx3 vertices")
    if rgb.shape != vertices.shape or not np.isfinite(vertices).all():
        raise VGGTOmegaPriorError("point cloud colors or coordinates are invalid")
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    )
    packed = np.empty(len(vertices), dtype=dtype)
    packed["x"], packed["y"], packed["z"] = vertices.T
    packed["red"], packed["green"], packed["blue"] = rgb.T
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        packed.tofile(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-scene", type=Path, required=True)
    parser.add_argument("--output-scene", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--source-commit", default=VERIFIED_SOURCE_COMMIT)
    parser.add_argument("--model-revision", default=VERIFIED_MODEL_REVISION)
    parser.add_argument("--checkpoint-sha256", default=VERIFIED_CHECKPOINT_SHA256)
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--confidence-percentile", type=float, default=20.0)
    parser.add_argument("--depth-edge-threshold", type=float, default=0.03)
    parser.add_argument("--max-relative-focal-spread", type=float, default=0.05)
    parser.add_argument("--max-points", type=int, default=4_000_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.image_resolution <= 0 or args.image_resolution % 16:
        raise VGGTOmegaPriorError("image resolution must be a positive multiple of 16")
    if not 0 <= args.confidence_percentile < 100:
        raise VGGTOmegaPriorError("confidence percentile must be in [0, 100)")
    if not 0 < args.depth_edge_threshold <= 1:
        raise VGGTOmegaPriorError("depth edge threshold must be in (0, 1]")
    if not 0 <= args.max_relative_focal_spread <= 1:
        raise VGGTOmegaPriorError("max relative focal spread must be in [0, 1]")
    if args.max_points <= 0:
        raise VGGTOmegaPriorError("max points must be positive")
    if args.seed < 0:
        raise VGGTOmegaPriorError("seed must be non-negative")


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    started = time.monotonic()
    source_scene = args.source_scene.expanduser().resolve()
    output_scene = args.output_scene.expanduser().resolve()
    source_root = args.source_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    color_root = source_scene / "color"
    image_paths = sorted(
        path for path in color_root.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not image_paths:
        raise VGGTOmegaPriorError(f"source scene has no color images: {color_root}")
    sizes: set[tuple[int, int]] = set()
    for path in image_paths:
        with Image.open(path) as image:
            sizes.add(image.size)
    if len(sizes) != 1:
        raise VGGTOmegaPriorError(f"source images must share one size: {sorted(sizes)}")
    original_w, original_h = next(iter(sizes))
    if not checkpoint.is_file():
        raise VGGTOmegaPriorError(f"checkpoint does not exist: {checkpoint}")
    checkpoint_record = artifact_record(checkpoint)
    if checkpoint_record["sha256"] != args.checkpoint_sha256:
        raise VGGTOmegaPriorError(
            "VGGT-Omega checkpoint hash mismatch: expected "
            f"{args.checkpoint_sha256}, got {checkpoint_record['sha256']}"
        )
    source_provenance = git_source_provenance(source_root, args.source_commit)

    sys.path.insert(0, str(source_root))
    import torch
    import torch.nn.functional as functional
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    if not torch.cuda.is_available():
        raise VGGTOmegaPriorError("VGGT-Omega requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model = VGGTOmega().eval()
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model = model.to("cuda")
    images = load_and_preprocess_images(
        [str(path) for path in image_paths],
        mode="balanced",
        image_resolution=args.image_resolution,
    ).to("cuda")
    with torch.inference_mode():
        predictions = model(images)
        extrinsics_t, intrinsics_t = encoding_to_camera(
            predictions["pose_enc"], predictions["images"].shape[-2:]
        )
    depth_t = predictions["depth"][0, ..., 0]
    confidence_t = predictions["depth_conf"][0, ..., 0]
    processed_h, processed_w = depth_t.shape[-2:]
    depth_t = functional.interpolate(
        depth_t[:, None], size=(original_h, original_w), mode="bilinear", align_corners=False
    )[:, 0]
    confidence_t = functional.interpolate(
        confidence_t[:, None],
        size=(original_h, original_w),
        mode="bilinear",
        align_corners=False,
    )[:, 0]
    depth = depth_t.float().cpu().numpy()
    confidence = confidence_t.float().cpu().numpy()
    extrinsics = extrinsics_t[0].float().cpu().numpy()
    intrinsics = resize_intrinsics(
        intrinsics_t[0].float().cpu().numpy(),
        source_size=(processed_h, processed_w),
        target_size=(original_h, original_w),
    )
    common_k, focal_spread = common_intrinsics(
        intrinsics, max_relative_spread=args.max_relative_focal_spread
    )

    output_color = output_scene / "color"
    output_depth = output_scene / "depth_da3"
    output_pose = output_scene / "pose"
    output_intrinsic = output_scene / "intrinsic"
    for root in (output_color, output_depth, output_pose, output_intrinsic):
        root.mkdir(parents=True, exist_ok=True)

    quota = max(1, int(np.ceil(args.max_points / len(image_paths))))
    point_parts: list[np.ndarray] = []
    color_parts: list[np.ndarray] = []
    frame_records: list[dict[str, Any]] = []
    for index, image_path in enumerate(image_paths):
        frame_id = image_path.stem
        output_image = output_color / f"{frame_id}{image_path.suffix.lower()}"
        shutil.copy2(image_path, output_image)
        frame_depth = depth[index].astype(np.float32)
        depth_path = output_depth / f"{frame_id}.npy"
        np.save(depth_path, frame_depth)
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :4] = extrinsics[index]
        c2w = np.linalg.inv(w2c)
        pose_path = output_pose / f"{frame_id}.txt"
        np.savetxt(pose_path, c2w, fmt="%.10f")

        valid = np.isfinite(frame_depth) & (frame_depth > 0)
        valid &= np.isfinite(confidence[index])
        threshold = float(np.percentile(confidence[index][valid], args.confidence_percentile))
        valid &= confidence[index] >= threshold
        valid &= ~depth_edge_mask(frame_depth, relative_threshold=args.depth_edge_threshold)
        indices = np.flatnonzero(valid.reshape(-1))
        if len(indices) > quota:
            indices = indices[np.linspace(0, len(indices) - 1, quota).astype(np.int64)]
        points = unproject_frame(frame_depth, intrinsics[index], w2c, indices)
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).reshape(-1, 3)[indices]
        point_parts.append(points)
        color_parts.append(rgb)
        frame_records.append(
            {
                "frame_id": frame_id,
                "depth": artifact_record(depth_path),
                "pose": artifact_record(pose_path),
                "confidence_threshold": threshold,
                "selected_points": len(indices),
                "depth_min": float(frame_depth.min()),
                "depth_max": float(frame_depth.max()),
                "depth_median": float(np.median(frame_depth)),
            }
        )

    points = np.concatenate(point_parts, axis=0)
    colors = np.concatenate(color_parts, axis=0)
    if len(points) > args.max_points:
        selection = np.linspace(0, len(points) - 1, args.max_points).astype(np.int64)
        points, colors = points[selection], colors[selection]
    point_path = output_scene / "pointcloud_da3.ply"
    write_binary_point_cloud(point_path, points, colors)
    intrinsic4 = np.eye(4, dtype=np.float32)
    intrinsic4[:3, :3] = common_k
    intrinsic_path = output_intrinsic / "intrinsic_color.txt"
    np.savetxt(intrinsic_path, intrinsic4, fmt="%.10f")

    depth_manifest_path = output_scene / "vggt_omega_depth_manifest.json"
    depth_manifest = {
        "schema_version": 1,
        "kind": "video2world.vggt_omega_depth_manifest",
        "backend": "facebook/VGGT-Omega-1B-512",
        "scene_id": source_scene.name,
        "frame_count": len(frame_records),
        "compatibility_directory": "depth_da3",
        "compatibility_boundary": (
            "The directory name is required by the legacy PGSR ScanNet loader; these depths "
            "were predicted by VGGT-Omega, not Depth Anything 3."
        ),
        "frames": frame_records,
    }
    depth_manifest_path.write_text(json.dumps(depth_manifest, indent=2) + "\n", encoding="utf-8")

    receipt = {
        "schema_version": 1,
        "kind": KIND,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "passed_technical_prior",
        "promotion_allowed": False,
        "scene_id": source_scene.name,
        "source_scene": str(source_scene),
        "source": source_provenance,
        "model": {
            "repository": VERIFIED_MODEL_REPOSITORY,
            "revision": args.model_revision,
            "checkpoint": checkpoint_record,
            "expected_checkpoint_sha256": args.checkpoint_sha256,
        },
        "parameters": {
            "image_resolution": args.image_resolution,
            "confidence_percentile": args.confidence_percentile,
            "depth_edge_threshold": args.depth_edge_threshold,
            "max_relative_focal_spread": args.max_relative_focal_spread,
            "max_points": args.max_points,
            "seed": args.seed,
        },
        "runtime": {
            "elapsed_seconds": time.monotonic() - started,
            "cuda_peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "processed_size": [processed_w, processed_h],
            "original_size": [original_w, original_h],
        },
        "camera": {
            "convention": "world_to_camera OpenCV decoded from official pose encoding",
            "per_frame_intrinsics_collapsed_for_scannet": True,
            "maximum_relative_focal_spread": focal_spread,
            "common_intrinsic": common_k.tolist(),
        },
        "outputs": {
            "depth_manifest": artifact_record(depth_manifest_path),
            "point_prior": {
                **artifact_record(point_path),
                "vertex_count": len(points),
                "compatibility_filename": "pointcloud_da3.ply",
            },
            "intrinsic": artifact_record(intrinsic_path),
            "pose_count": len(frame_records),
        },
        "claim_boundary": (
            "This is a fresh VGGT-Omega camera/depth/point prior in a legacy PGSR-compatible "
            "ScanNet directory. It is not DA3 output, calibrated metric geometry, a PGSR result, "
            "a TSDF result, or evidence that VGGT-Omega is better than DA3."
        ),
    }
    args.receipt.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args.receipt.expanduser().resolve().write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> int:
    args = parse_args()
    try:
        receipt = run(args)
    except (
        OSError,
        RuntimeError,
        ValueError,
        subprocess.SubprocessError,
        VGGTOmegaPriorError,
    ) as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}))
        return 2
    print(json.dumps({"status": receipt["status"], "outputs": receipt["outputs"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
