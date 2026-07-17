#!/usr/bin/env python3
"""Project a scene-fit mesh into a source camera and compare its silhouette to a mask."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image, ImageDraw


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_camera(path: Path, frame_id: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("camera file must contain an array")
    matches = [item for item in payload if str(item.get("img_name")) == frame_id]
    if len(matches) != 1:
        raise ValueError(f"expected one camera for {frame_id!r}, found {len(matches)}")
    camera = matches[0]
    position = np.asarray(camera.get("position"), dtype=np.float64)
    rotation = np.asarray(camera.get("rotation"), dtype=np.float64)
    if position.shape != (3,) or rotation.shape != (3, 3):
        raise ValueError("camera position/rotation has invalid dimensions")
    if not np.isfinite(position).all() or not np.isfinite(rotation).all():
        raise ValueError("camera contains non-finite values")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("camera rotation is not orthonormal")
    width = int(camera.get("width", 0))
    height = int(camera.get("height", 0))
    fx = float(camera.get("fx", 0))
    fy = float(camera.get("fy", 0))
    if width <= 0 or height <= 0 or fx <= 0 or fy <= 0:
        raise ValueError("camera intrinsics must be positive")
    return {
        "position": position,
        "rotation": rotation,
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fy,
        "raw": camera,
    }


def project_mesh_silhouette(
    mesh: trimesh.Trimesh,
    *,
    runtime_pivot: np.ndarray | None,
    camera: dict[str, Any],
    image_y_axis: str = "down",
    scene_transform: np.ndarray | None = None,
) -> np.ndarray:
    if image_y_axis not in {"down", "up"}:
        raise ValueError("image_y_axis must be 'down' or 'up'")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if scene_transform is not None:
        transform = np.asarray(scene_transform, dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("scene_transform must be a finite 4x4 matrix")
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
            raise ValueError("scene_transform must be affine")
        vertices_world = np.einsum(
            "ni,ji->nj",
            vertices,
            transform[:3, :3],
            optimize=False,
        ) + transform[:3, 3]
    else:
        pivot = np.asarray(runtime_pivot, dtype=np.float64)
        if pivot.shape != (3,) or not np.isfinite(pivot).all():
            raise ValueError("runtime_pivot must contain three finite values")
        vertices_world = vertices + pivot
    relative = vertices_world - camera["position"]
    camera_points = np.einsum("ni,ij->nj", relative, camera["rotation"])
    depth = camera_points[:, 2]
    visible = depth > 1e-6
    projected = np.full((len(camera_points), 2), np.nan, dtype=np.float64)
    projected[visible, 0] = (
        camera["fx"] * camera_points[visible, 0] / depth[visible] + camera["width"] / 2
    )
    y_sign = 1.0 if image_y_axis == "down" else -1.0
    projected[visible, 1] = (
        camera["height"] / 2 + y_sign * camera["fy"] * camera_points[visible, 1] / depth[visible]
    )

    canvas = Image.new("1", (camera["width"], camera["height"]), 0)
    draw = ImageDraw.Draw(canvas)
    for face in np.asarray(mesh.faces, dtype=np.int64):
        if not np.all(visible[face]):
            continue
        polygon = [tuple(projected[index]) for index in face]
        if not np.isfinite(np.asarray(polygon)).all():
            continue
        draw.polygon(polygon, fill=1)
    return np.asarray(canvas, dtype=bool)


def mask_bbox(mask: np.ndarray) -> list[int] | None:
    y, x = np.nonzero(mask)
    if len(x) == 0:
        return None
    return [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1]


def bbox_iou(left: list[int], right: list[int]) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def mask_center(mask: np.ndarray) -> np.ndarray:
    y, x = np.nonzero(mask)
    if len(x) == 0:
        raise ValueError("cannot compute center of an empty mask")
    return np.asarray([x.mean(), y.mean()], dtype=np.float64)


def make_overlay(
    source: Image.Image | None,
    observed: np.ndarray,
    rendered: np.ndarray,
) -> Image.Image:
    if source is None:
        background = np.full((*observed.shape, 3), 32, dtype=np.uint8)
    else:
        background = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
    observed_only = observed & ~rendered
    rendered_only = rendered & ~observed
    overlap = observed & rendered
    overlay = background.astype(np.float32)
    colors = (
        (observed_only, np.asarray([0, 220, 120], dtype=np.float32)),
        (rendered_only, np.asarray([255, 70, 70], dtype=np.float32)),
        (overlap, np.asarray([255, 210, 0], dtype=np.float32)),
    )
    for selection, color in colors:
        overlay[selection] = overlay[selection] * 0.35 + color * 0.65
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def review_scene_silhouette(
    *,
    mesh_path: Path,
    scene_fit_report_path: Path,
    cameras_path: Path,
    frame_id: str,
    observed_mask_path: Path,
    output_dir: Path,
    source_frame_path: Path | None = None,
    minimum_mask_iou: float = 0.65,
    minimum_bbox_iou: float = 0.8,
    maximum_center_error_px: float = 20.0,
    image_y_axis: str = "down",
    occluder_mask_paths: list[Path] | None = None,
) -> dict[str, Any]:
    if not 0 <= minimum_mask_iou <= 1 or not 0 <= minimum_bbox_iou <= 1:
        raise ValueError("IoU thresholds must be in [0, 1]")
    if maximum_center_error_px < 0:
        raise ValueError("maximum_center_error_px must be non-negative")
    report = json.loads(scene_fit_report_path.read_text(encoding="utf-8"))
    runtime_transform_value = report.get("runtime_transform", {}).get("matrix_row_major")
    scene_transform: np.ndarray | None = None
    pivot: np.ndarray | None = None
    if runtime_transform_value is not None:
        scene_transform = np.asarray(runtime_transform_value, dtype=np.float64)
        if scene_transform.shape != (4, 4) or not np.isfinite(scene_transform).all():
            raise ValueError("scene-fit report has no finite 4x4 runtime transform")
        if not np.allclose(scene_transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
            raise ValueError("scene-fit runtime transform must be affine")
    else:
        pivot = np.asarray(
            report.get("baked_relative_transform", {}).get("runtime_pivot"),
            dtype=np.float64,
        )
        if pivot.shape != (3,) or not np.isfinite(pivot).all():
            raise ValueError("scene-fit report has no finite runtime pivot")
    camera = load_camera(cameras_path, frame_id)
    loaded = trimesh.load(mesh_path, force="scene")
    mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded
    rendered = project_mesh_silhouette(
        mesh,
        runtime_pivot=pivot,
        camera=camera,
        image_y_axis=image_y_axis,
        scene_transform=scene_transform,
    )

    observed_image = Image.open(observed_mask_path).convert("L")
    if observed_image.size != (camera["width"], camera["height"]):
        raise ValueError("observed mask dimensions do not match camera")
    observed = np.asarray(observed_image) > 0
    observed_raw_pixels = int(np.count_nonzero(observed))
    occluder_union = np.zeros_like(observed)
    occluder_sources = []
    for path in occluder_mask_paths or []:
        image = Image.open(path).convert("L")
        if image.size != observed_image.size:
            raise ValueError("occluder mask dimensions do not match camera")
        mask = np.asarray(image) > 0
        occluder_union |= mask
        occluder_sources.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "pixels": int(np.count_nonzero(mask)),
            }
        )
        image.close()
    rendered_raw_pixels = int(np.count_nonzero(rendered))
    rendered_occluded_pixels = int(np.count_nonzero(rendered & occluder_union))
    observed_occluded_pixels = int(np.count_nonzero(observed & occluder_union))
    rendered = rendered & ~occluder_union
    observed = observed & ~occluder_union
    if not observed.any() or not rendered.any():
        raise ValueError("observed and rendered masks must both be non-empty")
    intersection = int(np.count_nonzero(observed & rendered))
    union = int(np.count_nonzero(observed | rendered))
    observed_pixels = int(np.count_nonzero(observed))
    rendered_pixels = int(np.count_nonzero(rendered))
    mask_iou = intersection / union
    precision = intersection / rendered_pixels
    recall = intersection / observed_pixels
    observed_bbox = mask_bbox(observed)
    rendered_bbox = mask_bbox(rendered)
    assert observed_bbox is not None and rendered_bbox is not None
    bounds_iou = bbox_iou(observed_bbox, rendered_bbox)
    center_error = float(np.linalg.norm(mask_center(observed) - mask_center(rendered)))
    gates = {
        "mask_iou": mask_iou >= minimum_mask_iou,
        "bbox_iou": bounds_iou >= minimum_bbox_iou,
        "center_error_px": center_error <= maximum_center_error_px,
    }

    source_frame = Image.open(source_frame_path) if source_frame_path else None
    if source_frame is not None and source_frame.size != observed_image.size:
        raise ValueError("source frame dimensions do not match observed mask")
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = output_dir / "source_camera_silhouette_overlay.png"
    rendered_path = output_dir / "rendered_silhouette.png"
    make_overlay(source_frame, observed, rendered).save(overlay_path)
    Image.fromarray(rendered.astype(np.uint8) * 255).save(rendered_path)
    receipt = {
        "schema_version": 1,
        "kind": "video2world.source_camera_silhouette_review",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "passed" if all(gates.values()) else "rejected",
        "promotion_allowed": all(gates.values()),
        "frame_id": frame_id,
        "camera_convention": {"rotation": "camera_to_world", "image_y_axis": image_y_axis},
        "scene_transform_mode": (
            "full_affine_runtime_transform"
            if scene_transform is not None
            else "baked_linear_plus_runtime_pivot"
        ),
        "sources": {
            "mesh": {"path": str(mesh_path), "sha256": sha256_file(mesh_path)},
            "scene_fit_report": {
                "path": str(scene_fit_report_path),
                "sha256": sha256_file(scene_fit_report_path),
            },
            "cameras": {"path": str(cameras_path), "sha256": sha256_file(cameras_path)},
            "observed_mask": {
                "path": str(observed_mask_path),
                "sha256": sha256_file(observed_mask_path),
            },
            "occluder_masks": occluder_sources,
        },
        "metrics": {
            "observed_pixels": observed_pixels,
            "observed_raw_pixels": observed_raw_pixels,
            "observed_occluded_pixels": observed_occluded_pixels,
            "rendered_raw_pixels": rendered_raw_pixels,
            "rendered_occluded_pixels": rendered_occluded_pixels,
            "rendered_pixels": rendered_pixels,
            "intersection_pixels": intersection,
            "union_pixels": union,
            "mask_iou": mask_iou,
            "rendered_precision": precision,
            "observed_recall": recall,
            "observed_bbox_xyxy": observed_bbox,
            "rendered_bbox_xyxy": rendered_bbox,
            "bbox_iou": bounds_iou,
            "center_error_px": center_error,
        },
        "thresholds": {
            "minimum_mask_iou": minimum_mask_iou,
            "minimum_bbox_iou": minimum_bbox_iou,
            "maximum_center_error_px": maximum_center_error_px,
        },
        "gates": gates,
        "outputs": {
            "overlay": {"path": str(overlay_path), "sha256": sha256_file(overlay_path)},
            "rendered_mask": {
                "path": str(rendered_path),
                "sha256": sha256_file(rendered_path),
            },
        },
    }
    write_json(output_dir / "source_camera_silhouette_review.json", receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--scene-fit-report", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--frame-id", required=True)
    parser.add_argument("--observed-mask", type=Path, required=True)
    parser.add_argument("--source-frame", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-mask-iou", type=float, default=0.65)
    parser.add_argument("--minimum-bbox-iou", type=float, default=0.8)
    parser.add_argument("--maximum-center-error-px", type=float, default=20.0)
    parser.add_argument("--image-y-axis", choices=("down", "up"), default="down")
    parser.add_argument("--occluder-mask", type=Path, action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = review_scene_silhouette(
        mesh_path=args.mesh.expanduser().resolve(),
        scene_fit_report_path=args.scene_fit_report.expanduser().resolve(),
        cameras_path=args.cameras.expanduser().resolve(),
        frame_id=args.frame_id,
        observed_mask_path=args.observed_mask.expanduser().resolve(),
        source_frame_path=args.source_frame.expanduser().resolve() if args.source_frame else None,
        output_dir=args.output_dir.expanduser().resolve(),
        minimum_mask_iou=args.minimum_mask_iou,
        minimum_bbox_iou=args.minimum_bbox_iou,
        maximum_center_error_px=args.maximum_center_error_px,
        image_y_axis=args.image_y_axis,
        occluder_mask_paths=[path.expanduser().resolve() for path in args.occluder_mask],
    )
    print(json.dumps(receipt, ensure_ascii=True, indent=2))
    return 0 if receipt["promotion_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
