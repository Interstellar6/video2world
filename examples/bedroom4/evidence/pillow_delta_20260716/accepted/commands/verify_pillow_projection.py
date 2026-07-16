#!/usr/bin/env python3
"""Verify pillow instance points by projecting them back into source frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def project(points: np.ndarray, w2c: np.ndarray, intrinsic: dict[str, Any]) -> tuple[np.ndarray, ...]:
    camera = points @ w2c[:3, :3].T + w2c[:3, 3]
    depth = camera[:, 2]
    valid_depth = depth > 1e-6
    u = np.rint(float(intrinsic["fx"]) * camera[:, 0] / depth + float(intrinsic["cx"])).astype(np.int64)
    v = np.rint(float(intrinsic["fy"]) * camera[:, 1] / depth + float(intrinsic["cy"])).astype(np.int64)
    inside = (
        valid_depth
        & (u >= 0)
        & (u < int(intrinsic["w"]))
        & (v >= 0)
        & (v < int(intrinsic["h"]))
    )
    return inside, u, v, depth


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta-root", type=Path, required=True)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--frames", default="000020,000040,000064")
    parser.add_argument("--min-point-hit-ratio", type=float, default=0.95)
    args = parser.parse_args()

    delta = args.delta_root.resolve()
    base = args.base_root.resolve()
    camera_info = read_json(delta / "video2mesh" / "scene" / "cameras" / "camera_info.json")
    intrinsic = camera_info["intrinsic"]
    point_cloud_path = base / "scannetppv2" / "data" / "bedroom_4" / "pointcloud_da3.ply"
    cloud = o3d.io.read_point_cloud(str(point_cloud_path))
    all_points = np.asarray(cloud.points)
    all_colors = np.asarray(cloud.colors)
    indices_path = delta / "video2mesh" / "masks" / "3d_instances" / "sam3_pillow_01" / "point_indices.npy"
    indices = np.load(indices_path)
    points = all_points[indices]
    colors = all_colors[indices]

    output_dir = delta / "evidence" / "projection_verification"
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for frame_id in [value.strip() for value in args.frames.split(",") if value.strip()]:
        image_path = delta / "video2mesh" / "scene" / "frames" / f"{frame_id}.png"
        mask_path = delta / "video2mesh" / "masks" / "2d" / "sam3_class_pillow" / f"{frame_id}.png"
        image = np.asarray(Image.open(image_path).convert("RGB"))
        mask = np.asarray(Image.open(mask_path).convert("L")) >= 128
        inside, u, v, depth = project(points, np.asarray(camera_info["extrinsic"][frame_id]), intrinsic)
        visible_indices = np.flatnonzero(inside)
        hit = mask[v[visible_indices], u[visible_indices]]
        hit_ratio = float(hit.mean()) if hit.size else 0.0

        zbuffer = np.full(mask.shape, np.inf, dtype=np.float32)
        np.minimum.at(zbuffer, (v[visible_indices], u[visible_indices]), depth[visible_indices].astype(np.float32))
        occupancy = np.isfinite(zbuffer)
        occupancy_dilated = cv2.dilate(occupancy.astype(np.uint8), np.ones((5, 5), dtype=np.uint8)) > 0
        overlap = occupancy_dilated & mask
        precision = float(overlap.sum() / max(1, occupancy_dilated.sum()))
        recall = float(overlap.sum() / max(1, mask.sum()))

        overlay = image.copy()
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (235, 65, 52), 3)
        draw_indices = visible_indices[:: max(1, len(visible_indices) // 50000)]
        overlay[v[draw_indices], u[draw_indices]] = np.array([30, 205, 235], dtype=np.uint8)
        output_path = output_dir / f"pillow_projection_{frame_id}.jpg"
        rendered = Image.fromarray(overlay)
        draw = ImageDraw.Draw(rendered)
        draw.rectangle((12, 12, 610, 62), fill=(0, 0, 0))
        draw.text(
            (22, 22),
            f"cyan: 3D points | red: SAM3 union | point hit {hit_ratio:.3f}",
            fill=(255, 255, 255),
        )
        rendered.save(output_path, quality=94)
        records.append(
            {
                "frame_id": frame_id,
                "projected_points": len(visible_indices),
                "projected_point_mask_hit_ratio": hit_ratio,
                "dilated_projection_precision": precision,
                "dilated_projection_mask_recall": recall,
                "image": str(output_path),
            }
        )

    centered = points - points.mean(axis=0)
    _, _, basis = np.linalg.svd(centered[:: max(1, len(centered) // 40000)], full_matrices=False)
    local = centered @ basis.T
    sample_stride = max(1, len(local) // 80000)
    sampled = local[::sample_stride]
    sampled_colors = np.clip(colors[::sample_stride], 0.0, 1.0)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 9), dpi=150)
    axes = [
        fig.add_subplot(2, 2, 1, projection="3d"),
        fig.add_subplot(2, 2, 2),
        fig.add_subplot(2, 2, 3),
        fig.add_subplot(2, 2, 4),
    ]
    axes[0].scatter(sampled[:, 0], sampled[:, 1], sampled[:, 2], s=0.12, c=sampled_colors)
    axes[0].set_title("PCA isometric")
    for axis, columns, title in zip(
        axes[1:],
        [(0, 1), (0, 2), (1, 2)],
        ["PCA axes 1-2", "PCA axes 1-3", "PCA axes 2-3"],
        strict=True,
    ):
        axis.scatter(sampled[:, columns[0]], sampled[:, columns[1]], s=0.12, c=sampled_colors)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(title)
    fig.suptitle(f"sam3_pillow_01 | {len(points):,} DA3 points | touching pillow ensemble")
    fig.tight_layout()
    scatter_path = output_dir / "pillow_point_cloud_pca.png"
    fig.savefig(scatter_path, bbox_inches="tight")
    plt.close(fig)

    minimum_hit_ratio = min(record["projected_point_mask_hit_ratio"] for record in records)
    quality_gate_passed = minimum_hit_ratio >= args.min_point_hit_ratio
    result = {
        "schema_version": 1,
        "status": "projection_checked",
        "object_id": "sam3_pillow_01",
        "point_count": len(points),
        "interpretation": (
            "One connected ensemble of three touching pillows; not three stable tracked instances."
            if quality_gate_passed
            else "Rejected for Web focus: projected 3D points include geometry outside the pillow masks."
        ),
        "quality_gate": {
            "passed": quality_gate_passed,
            "metric": "minimum projected_point_mask_hit_ratio across checked frames",
            "threshold": args.min_point_hit_ratio,
            "value": minimum_hit_ratio,
        },
        "frames": records,
        "point_cloud_preview": str(scatter_path),
    }
    write_json(output_dir / "projection_verification.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
