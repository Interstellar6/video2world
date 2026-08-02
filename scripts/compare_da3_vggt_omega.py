#!/usr/bin/env python3
"""Compare DA3 and VGGT-Omega priors plus their downstream PGSR/TSDF runs."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from video2world.hashing import atomic_write_json, sha256_file

KIND = "video2world.da3_vggt_omega_ab_receipt"
PGSR_METRIC_PATTERN = re.compile(
    r"\[ITER\s+(?P<iteration>\d+)\].*?L1\s+(?P<l1>[0-9.eE+-]+)\s+"
    r"PSNR\s+(?P<psnr>[0-9.eE+-]+)"
)


class PriorComparisonError(RuntimeError):
    """Raised when two reconstruction priors cannot be compared safely."""


def load_camera_sequence(scene: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    pose_root = scene / "pose"
    frame_ids = sorted(path.stem for path in pose_root.glob("*.txt"))
    if not frame_ids:
        raise PriorComparisonError(f"scene has no poses: {scene}")
    cameras = np.stack([np.loadtxt(pose_root / f"{frame_id}.txt") for frame_id in frame_ids])
    if cameras.shape != (len(frame_ids), 4, 4) or not np.isfinite(cameras).all():
        raise PriorComparisonError(f"scene poses are invalid: {scene}")
    intrinsic = np.loadtxt(scene / "intrinsic" / "intrinsic_color.txt")[:3, :3]
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise PriorComparisonError(f"scene intrinsic is invalid: {scene}")
    return frame_ids, cameras, intrinsic


def umeyama_similarity(
    source: np.ndarray, target: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit target = scale * rotation @ source + translation without reflection."""

    source_points = np.asarray(source, dtype=np.float64)
    target_points = np.asarray(target, dtype=np.float64)
    if source_points.shape != target_points.shape or source_points.ndim != 2:
        raise PriorComparisonError("similarity point arrays must share shape (N, D)")
    if len(source_points) < source_points.shape[1] + 1:
        raise PriorComparisonError("similarity fit has too few points")
    source_mean = source_points.mean(axis=0)
    target_mean = target_points.mean(axis=0)
    source_centered = source_points - source_mean
    target_centered = target_points - target_mean
    variance = float(np.mean(np.sum(source_centered**2, axis=1)))
    if variance <= 1e-12:
        raise PriorComparisonError("similarity source points have no spatial extent")
    covariance = target_centered.T @ source_centered / len(source_points)
    left, singular, right_t = np.linalg.svd(covariance)
    sign = np.ones(source_points.shape[1])
    if np.linalg.det(left @ right_t) < 0:
        sign[-1] = -1.0
    rotation = left @ np.diag(sign) @ right_t
    scale = float(np.sum(singular * sign) / variance)
    if not math.isfinite(scale) or scale <= 0:
        raise PriorComparisonError("similarity fit produced invalid scale")
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def rotation_error_degrees(reference: np.ndarray, candidate: np.ndarray) -> float:
    delta = np.asarray(reference).T @ np.asarray(candidate)
    cosine = float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def parse_pgsr_metrics(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    matches = list(PGSR_METRIC_PATTERN.finditer(text))
    if not matches:
        raise PriorComparisonError(f"PGSR log has no evaluation metrics: {path}")
    final = max(matches, key=lambda item: int(item.group("iteration")))
    digest, size = sha256_file(path)
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "size_bytes": size,
        "iteration": int(final.group("iteration")),
        "l1": float(final.group("l1")),
        "psnr_db": float(final.group("psnr")),
    }


def parse_ply_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        header = handle.read(65536)
    marker = header.find(b"end_header\n")
    if marker < 0:
        raise PriorComparisonError(f"PLY header is incomplete: {path}")
    try:
        lines = header[: marker + len(b"end_header\n")].decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise PriorComparisonError(f"PLY header is not ASCII: {path}") from exc
    elements: dict[str, int] = {}
    properties: list[str] = []
    active_element: str | None = None
    for line in lines:
        parts = line.split()
        if parts[:1] == ["element"] and len(parts) == 3:
            active_element = parts[1]
            elements[active_element] = int(parts[2])
        elif parts[:1] == ["property"] and active_element == "vertex":
            properties.append(parts[-1])
    digest, size = sha256_file(path)
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "size_bytes": size,
        "elements": elements,
        "vertex_properties": properties,
    }


def compare_depths(
    da3_scene: Path,
    vggt_scene: Path,
    frame_ids: list[str],
    *,
    vggt_scale: float,
    stride: int,
) -> dict[str, Any]:
    log_errors: list[np.ndarray] = []
    relative_errors: list[np.ndarray] = []
    valid_pixels = 0
    for frame_id in frame_ids:
        da3 = np.load(da3_scene / "depth_da3" / f"{frame_id}.npy")[::stride, ::stride]
        vggt = np.load(vggt_scene / "depth_da3" / f"{frame_id}.npy")[::stride, ::stride]
        if da3.shape != vggt.shape:
            raise PriorComparisonError(f"depth shape mismatch for {frame_id}")
        vggt = vggt * vggt_scale
        valid = np.isfinite(da3) & np.isfinite(vggt) & (da3 > 0) & (vggt > 0)
        if not np.any(valid):
            continue
        da3_valid = da3[valid]
        vggt_valid = vggt[valid]
        log_errors.append(np.abs(np.log(vggt_valid) - np.log(da3_valid)))
        relative_errors.append(np.abs(vggt_valid - da3_valid) / da3_valid)
        valid_pixels += int(valid.sum())
    if not log_errors:
        raise PriorComparisonError("depth comparison has no overlapping valid pixels")
    log_value = np.concatenate(log_errors)
    relative_value = np.concatenate(relative_errors)
    return {
        "sample_stride": stride,
        "valid_pixels": valid_pixels,
        "median_absolute_log_error": float(np.median(log_value)),
        "p90_absolute_log_error": float(np.percentile(log_value, 90)),
        "median_absolute_relative_error": float(np.median(relative_value)),
        "p90_absolute_relative_error": float(np.percentile(relative_value, 90)),
        "interpretation": (
            "Indicative prior disagreement after trajectory-scale alignment; neither depth is "
            "ground truth, and predicted-camera differences remain a confounder."
        ),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    da3_scene = args.da3_scene.expanduser().resolve()
    vggt_scene = args.vggt_scene.expanduser().resolve()
    da3_ids, da3_c2w, da3_k = load_camera_sequence(da3_scene)
    vggt_ids, vggt_c2w, vggt_k = load_camera_sequence(vggt_scene)
    if da3_ids != vggt_ids:
        raise PriorComparisonError("DA3 and VGGT-Omega frame IDs differ")
    da3_centers = da3_c2w[:, :3, 3]
    vggt_centers = vggt_c2w[:, :3, 3]
    scale, rotation, translation = umeyama_similarity(vggt_centers, da3_centers)
    aligned_centers = scale * (vggt_centers @ rotation.T) + translation
    residual = np.linalg.norm(aligned_centers - da3_centers, axis=1)
    extent = float(np.linalg.norm(np.ptp(da3_centers, axis=0)))
    rotation_errors = [
        rotation_error_degrees(
            da3_c2w[index, :3, :3], rotation @ vggt_c2w[index, :3, :3]
        )
        for index in range(len(da3_ids))
    ]
    da3_metrics = parse_pgsr_metrics(args.da3_pgsr_log.expanduser().resolve())
    vggt_metrics = parse_pgsr_metrics(args.vggt_pgsr_log.expanduser().resolve())
    psnr_delta = vggt_metrics["psnr_db"] - da3_metrics["psnr_db"]
    l1_delta = vggt_metrics["l1"] - da3_metrics["l1"]
    if psnr_delta > 0 and l1_delta < 0:
        numeric_indication = "vggt_omega_better_pgsr_training_fit"
    elif psnr_delta < 0 and l1_delta > 0:
        numeric_indication = "da3_better_pgsr_training_fit"
    else:
        numeric_indication = "mixed_pgsr_training_fit"
    return {
        "schema_version": 1,
        "kind": KIND,
        "created_at": datetime.now(UTC).isoformat(),
        "scene_id": da3_scene.name,
        "status": "numeric_ab_complete_visual_review_required",
        "promotion_allowed": False,
        "frame_count": len(da3_ids),
        "camera_alignment": {
            "mapping": "da3_world = scale * rotation @ vggt_world + translation",
            "scale": scale,
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
            "center_rmse": float(np.sqrt(np.mean(residual**2))),
            "center_rmse_over_da3_extent": float(
                np.sqrt(np.mean(residual**2)) / max(extent, 1e-12)
            ),
            "center_p90": float(np.percentile(residual, 90)),
            "rotation_median_deg": float(np.median(rotation_errors)),
            "rotation_p90_deg": float(np.percentile(rotation_errors, 90)),
            "focal_relative_delta": {
                "fx": float((vggt_k[0, 0] - da3_k[0, 0]) / da3_k[0, 0]),
                "fy": float((vggt_k[1, 1] - da3_k[1, 1]) / da3_k[1, 1]),
            },
        },
        "depth_disagreement": compare_depths(
            da3_scene, vggt_scene, da3_ids, vggt_scale=scale, stride=args.depth_stride
        ),
        "pgsr": {
            "da3": da3_metrics,
            "vggt_omega": vggt_metrics,
            "vggt_minus_da3": {"psnr_db": psnr_delta, "l1": l1_delta},
            "numeric_indication": numeric_indication,
        },
        "assets": {
            "da3": {
                "scene_gaussian": parse_ply_header(args.da3_scene_gaussian),
                "scene_mesh": parse_ply_header(args.da3_scene_mesh),
            },
            "vggt_omega": {
                "scene_gaussian": parse_ply_header(args.vggt_scene_gaussian),
                "scene_mesh": parse_ply_header(args.vggt_scene_mesh),
            },
        },
        "claim_boundary": (
            "This compares two fresh priors on the same ordered RGB frames. PGSR training fit, "
            "trajectory agreement, depth disagreement, and PLY topology do not by themselves "
            "prove visual quality, metric accuracy, semantic lifting quality, or world promotion."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--da3-scene", type=Path, required=True)
    parser.add_argument("--vggt-scene", type=Path, required=True)
    parser.add_argument("--da3-pgsr-log", type=Path, required=True)
    parser.add_argument("--vggt-pgsr-log", type=Path, required=True)
    parser.add_argument("--da3-scene-gaussian", type=Path, required=True)
    parser.add_argument("--vggt-scene-gaussian", type=Path, required=True)
    parser.add_argument("--da3-scene-mesh", type=Path, required=True)
    parser.add_argument("--vggt-scene-mesh", type=Path, required=True)
    parser.add_argument("--depth-stride", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.depth_stride <= 0:
        print(json.dumps({"status": "failed", "error": "depth stride must be positive"}))
        return 2
    try:
        report = build_report(args)
        atomic_write_json(args.output.expanduser().resolve(), report)
    except (OSError, ValueError, PriorComparisonError) as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}))
        return 2
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
