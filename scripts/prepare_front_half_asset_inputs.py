#!/usr/bin/env python3
"""Prepare front-half object reconstruction inputs from a Video2Mesh run.

The script is intentionally pre-QA and pre-scene-graph.  It selects the best
single RGB/RGBA crop per physical object using calibrated 3D point projection,
exports a point-cloud orthographic reference image, and writes an auditable
manifest that can feed TRELLIS.2 or a later object-completion provider.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps


STRUCTURE_CATEGORIES = {"ceiling", "floor", "wall", "other_structure"}
PLANAR_CATEGORIES = {"window", "door", "picture", "mirror", "wall art"}


@dataclass(frozen=True)
class Candidate:
    frame_id: str
    image_path: str
    mask_path: str
    bbox_xyxy: list[int]
    bbox_width: int
    bbox_height: int
    bbox_area: int
    visible_points: int
    mask_support_points: int
    support_ratio: float
    instance_point_coverage: float
    foreground_fill_ratio: float
    sharpness: float
    sharpness_score: float
    border_contact_sides: int
    border_penalty: float
    selection_quality: float
    selection_quality_breakdown: dict[str, float]
    score: float


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_ascii_ply(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    header_lines = 0
    vertex_count: int | None = None
    properties: list[str] = []
    ascii_format = False
    in_vertex = False
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            header_lines += 1
            stripped = line.strip()
            if stripped == "format ascii 1.0":
                ascii_format = True
            elif stripped.startswith("element vertex "):
                vertex_count = int(stripped.split()[-1])
                in_vertex = True
            elif stripped.startswith("element ") and not stripped.startswith("element vertex "):
                in_vertex = False
            elif in_vertex and stripped.startswith("property "):
                properties.append(stripped.split()[-1])
            elif stripped == "end_header":
                break
        else:
            raise ValueError(f"PLY header has no end_header: {path}")
    if not ascii_format:
        raise ValueError(f"Only ASCII PLY is supported: {path}")
    if vertex_count is None or vertex_count <= 0:
        raise ValueError(f"PLY has no vertices: {path}")
    columns = {name: index for index, name in enumerate(properties)}
    if not {"x", "y", "z"}.issubset(columns):
        raise ValueError(f"PLY must contain x/y/z properties: {path}")
    data = np.loadtxt(path, dtype=np.float64, skiprows=header_lines)
    data = np.atleast_2d(data)
    if len(data) != vertex_count:
        raise ValueError(f"PLY vertex count mismatch for {path}: header={vertex_count}, parsed={len(data)}")
    xyz = data[:, [columns["x"], columns["y"], columns["z"]]]
    colors = None
    if {"red", "green", "blue"}.issubset(columns):
        colors = data[:, [columns["red"], columns["green"], columns["blue"]]]
        colors = np.clip(colors, 0, 255).astype(np.uint8)
    return xyz, colors


def sample_points(points: np.ndarray, colors: np.ndarray | None, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray | None]:
    if max_points <= 0 or len(points) <= max_points:
        return points, colors
    generator = np.random.default_rng(seed)
    indices = generator.choice(len(points), size=max_points, replace=False)
    sampled_colors = colors[indices] if colors is not None else None
    return points[indices], sampled_colors


def intrinsic_for_frame(camera_info: dict[str, Any], frame_id: str) -> dict[str, float]:
    intrinsics = camera_info.get("intrinsics")
    frame_camera_ids = camera_info.get("frame_camera_ids")
    value: Any = None
    if isinstance(intrinsics, dict):
        if frame_id in intrinsics:
            value = intrinsics[frame_id]
        elif isinstance(frame_camera_ids, dict):
            value = intrinsics.get(str(frame_camera_ids.get(frame_id, "")))
        elif len(intrinsics) == 1:
            value = next(iter(intrinsics.values()))
    if value is None:
        value = camera_info.get("intrinsic")
    required = ("fx", "fy", "cx", "cy", "w", "h")
    if not isinstance(value, dict) or not all(isinstance(value.get(key), int | float) for key in required):
        raise ValueError(f"Missing numeric intrinsic for frame {frame_id}: {required}")
    return {key: float(value[key]) for key in required}


def project_points(points: np.ndarray, world_to_camera: np.ndarray, intrinsic: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera_points = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    depth = camera_points[:, 2]
    valid = depth > 1e-5
    safe_depth = np.where(valid, depth, 1.0)
    u = intrinsic["fx"] * camera_points[:, 0] / safe_depth + intrinsic["cx"]
    v = intrinsic["fy"] * camera_points[:, 1] / safe_depth + intrinsic["cy"]
    return u, v, valid


def laplacian_sharpness(image: Image.Image, box: list[int]) -> tuple[float, float]:
    crop = image.crop(tuple(box)).convert("L")
    if crop.width < 4 or crop.height < 4:
        return 0.0, 0.0
    gray = np.asarray(crop, dtype=np.float32) / 255.0
    padded = np.pad(gray, 1, mode="edge")
    lap = (
        -4.0 * padded[1:-1, 1:-1]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )
    sharpness = float(lap.var())
    score = float(np.clip(math.log1p(sharpness * 1000.0) / math.log1p(60.0), 0.0, 1.0))
    return sharpness, score


def border_contact_sides(box: list[int], width: int, height: int, tolerance: int = 2) -> int:
    x0, y0, x1, y1 = box
    return int(x0 <= tolerance) + int(y0 <= tolerance) + int(x1 >= width - tolerance) + int(y1 >= height - tolerance)


def selection_quality(
    *,
    support_ratio: float,
    instance_point_coverage: float,
    foreground_fill_ratio: float,
    sharpness_score: float,
    border_sides: int,
) -> tuple[float, float, dict[str, float]]:
    fill_score = float(np.clip(1.0 - abs(foreground_fill_ratio - 0.62) / 0.62, 0.0, 1.0))
    coverage_score = float(np.clip(math.sqrt(max(0.0, instance_point_coverage)), 0.0, 1.0))
    border_penalty = float(np.clip(0.18 * border_sides, 0.0, 0.72))
    unpenalized = (
        0.42 * float(np.clip(support_ratio, 0.0, 1.0))
        + 0.32 * coverage_score
        + 0.16 * float(np.clip(sharpness_score, 0.0, 1.0))
        + 0.10 * fill_score
    )
    quality = float(np.clip(unpenalized * (1.0 - border_penalty), 0.0, 1.0))
    return quality, border_penalty, {
        "support_ratio": float(support_ratio),
        "instance_point_coverage": float(instance_point_coverage),
        "coverage_score": coverage_score,
        "foreground_fill_ratio": float(foreground_fill_ratio),
        "fill_score": fill_score,
        "sharpness_score": float(sharpness_score),
        "border_contact_sides": float(border_sides),
        "border_penalty": border_penalty,
        "quality_without_border": float(unpenalized),
    }


def candidate_rank_score(candidate: Candidate | None = None, **values: float) -> float:
    if candidate is not None:
        quality = candidate.selection_quality
        bbox_width = float(candidate.bbox_width)
        bbox_height = float(candidate.bbox_height)
        support_points = float(candidate.mask_support_points)
        border_sides = float(candidate.border_contact_sides)
    else:
        quality = float(values["selection_quality"])
        bbox_width = float(values["bbox_width"])
        bbox_height = float(values["bbox_height"])
        support_points = float(values["mask_support_points"])
        border_sides = float(values.get("border_contact_sides", 0.0))
    short_side = max(1.0, min(bbox_width, bbox_height))
    aspect = max(bbox_width / max(1.0, bbox_height), bbox_height / max(1.0, bbox_width))
    aspect_penalty = max(0.45, 1.0 - min(max(aspect - 2.2, 0.0) / 4.0, 0.55))
    border_penalty = max(0.35, 1.0 - 0.22 * border_sides)
    support_bonus = 1.0 + min(math.log1p(max(0.0, support_points)) / 18.0, 0.35)
    return float(quality * math.sqrt(short_side) * aspect_penalty * border_penalty * support_bonus)


def find_mask_dir(masks_dir: Path, object_id: str, category: str, name: str | None) -> Path | None:
    candidates = [
        masks_dir / object_id,
        masks_dir / f"sam3_instance_{object_id}",
        masks_dir / f"sam3_class_{category}",
        masks_dir / f"gdino_object_{category}",
    ]
    if name:
        candidates.append(masks_dir / str(name).replace(" ", "_"))
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    normalized_category = category.replace(" ", "-")
    for candidate in sorted(masks_dir.iterdir()):
        if candidate.is_dir() and candidate.name.endswith(normalized_category):
            return candidate
    return None


def candidate_for_frame(
    points: np.ndarray,
    frame_id: str,
    image_path: Path,
    mask_path: Path,
    world_to_camera: np.ndarray,
    intrinsic: dict[str, float],
    probability_threshold: float,
    min_support_points: int,
) -> Candidate | None:
    with Image.open(image_path) as opened_image:
        image = opened_image.convert("RGB")
        width, height = image.size
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
    if mask.shape != (height, width):
        mask = np.asarray(Image.fromarray(mask).resize((width, height), Image.Resampling.NEAREST), dtype=np.uint8)
    foreground = mask >= int(round(probability_threshold * 255.0))
    if not foreground.any():
        return None
    u, v, in_front = project_points(points, world_to_camera, intrinsic)
    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)
    in_image = in_front & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    if int(in_image.sum()) < min_support_points:
        return None
    visible_indices = np.flatnonzero(in_image)
    support = foreground[vi[visible_indices], ui[visible_indices]]
    support_indices = visible_indices[support]
    if len(support_indices) < min_support_points:
        return None
    support_u = u[support_indices]
    support_v = v[support_indices]
    x0, x1 = np.percentile(support_u, (1.0, 99.0))
    y0, y1 = np.percentile(support_v, (1.0, 99.0))
    raw_width = max(1.0, float(x1 - x0))
    raw_height = max(1.0, float(y1 - y0))
    padding = max(12.0, 0.18 * max(raw_width, raw_height))
    box = [
        max(0, int(math.floor(x0 - padding))),
        max(0, int(math.floor(y0 - padding))),
        min(width, int(math.ceil(x1 + padding))),
        min(height, int(math.ceil(y1 + padding))),
    ]
    bbox_width = max(1, box[2] - box[0])
    bbox_height = max(1, box[3] - box[1])
    bbox_area = bbox_width * bbox_height
    support_ratio = float(len(support_indices) / max(1, len(visible_indices)))
    instance_point_coverage = float(len(support_indices) / max(1, len(points)))
    crop_foreground = foreground[box[1] : box[3], box[0] : box[2]]
    foreground_fill_ratio = float(crop_foreground.sum() / max(1, bbox_area))
    sharpness, sharpness_score = laplacian_sharpness(image, box)
    border_sides = border_contact_sides(box, width, height)
    quality, border_penalty, breakdown = selection_quality(
        support_ratio=support_ratio,
        instance_point_coverage=instance_point_coverage,
        foreground_fill_ratio=foreground_fill_ratio,
        sharpness_score=sharpness_score,
        border_sides=border_sides,
    )
    score = candidate_rank_score(
        selection_quality=quality,
        bbox_width=float(bbox_width),
        bbox_height=float(bbox_height),
        mask_support_points=float(len(support_indices)),
        border_contact_sides=float(border_sides),
    )
    return Candidate(
        frame_id=frame_id,
        image_path=str(image_path),
        mask_path=str(mask_path),
        bbox_xyxy=box,
        bbox_width=bbox_width,
        bbox_height=bbox_height,
        bbox_area=bbox_area,
        visible_points=int(len(visible_indices)),
        mask_support_points=int(len(support_indices)),
        support_ratio=support_ratio,
        instance_point_coverage=instance_point_coverage,
        foreground_fill_ratio=foreground_fill_ratio,
        sharpness=sharpness,
        sharpness_score=sharpness_score,
        border_contact_sides=border_sides,
        border_penalty=border_penalty,
        selection_quality=quality,
        selection_quality_breakdown=breakdown,
        score=score,
    )


def connected_component_from_seed(crop_mask: np.ndarray, seed: np.ndarray, min_component_support: float) -> tuple[np.ndarray, dict[str, Any]]:
    from scipy import ndimage

    labels, label_count = ndimage.label(crop_mask)
    retained_labels: list[int] = []
    retained_by = "size_fallback"
    component_seed_pixels: dict[str, int] = {}
    if label_count:
        sizes = np.bincount(labels.ravel())
        if seed.any():
            seed_counts = np.bincount(labels[seed], minlength=len(sizes))
            component_seed_pixels = {str(index): int(count) for index, count in enumerate(seed_counts) if index and count}
            retained_labels = [int(index) for index, count in enumerate(seed_counts) if index and count and sizes[index] >= 8]
            if retained_labels:
                retained_by = "projected_instance_seed"
        if not retained_labels and len(sizes) > 1:
            threshold = max(8, int(sizes[1:].max() * min_component_support))
            retained_labels = [int(index) for index, size in enumerate(sizes) if index and size >= threshold]
    cleaned = np.isin(labels, retained_labels) if retained_labels else crop_mask
    if int(cleaned.sum()) < 16:
        cleaned = crop_mask
    return cleaned, {
        "foreground_pixels_before_component_filter": int(crop_mask.sum()),
        "foreground_pixels_after_component_filter": int(cleaned.sum()),
        "component_count": int(label_count),
        "retained_component_labels": retained_labels,
        "retained_by": retained_by,
        "component_seed_pixels": component_seed_pixels,
    }


def mask_for_candidate(
    candidate: Candidate,
    points: np.ndarray,
    world_to_camera: np.ndarray,
    intrinsic: dict[str, float],
    probability_threshold: float,
    min_component_support: float,
) -> tuple[Image.Image, Image.Image, Image.Image, dict[str, Any]]:
    image = Image.open(candidate.image_path).convert("RGB")
    mask = np.asarray(Image.open(candidate.mask_path).convert("L"), dtype=np.uint8)
    if mask.shape != (image.height, image.width):
        mask = np.asarray(Image.fromarray(mask).resize(image.size, Image.Resampling.NEAREST), dtype=np.uint8)
    foreground = mask >= int(round(probability_threshold * 255.0))
    x0, y0, x1, y1 = candidate.bbox_xyxy
    crop_mask = foreground[y0:y1, x0:x1]
    u, v, in_front = project_points(points, world_to_camera, intrinsic)
    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)
    in_image = in_front & (ui >= x0) & (ui < x1) & (vi >= y0) & (vi < y1)
    seed = np.zeros_like(crop_mask, dtype=bool)
    if in_image.any():
        image_indices = np.flatnonzero(in_image)
        inside = foreground[vi[image_indices], ui[image_indices]]
        support_indices = image_indices[inside]
        if len(support_indices):
            seed[vi[support_indices] - y0, ui[support_indices] - x0] = True
    cleaned, component_report = connected_component_from_seed(crop_mask, seed, min_component_support)
    crop_rgb = image.crop((x0, y0, x1, y1))
    rgba = crop_rgb.convert("RGBA")
    rgba.putalpha(Image.fromarray((cleaned.astype(np.uint8) * 255), mode="L"))
    support_image = Image.fromarray((seed.astype(np.uint8) * 255), mode="L")
    component_report["projected_instance_seed"] = {
        "seed_pixels": int(seed.sum()),
        "projected_points_in_crop": int(in_image.sum()),
    }
    return crop_rgb, rgba, support_image, component_report


def pca_basis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = points - points.mean(axis=0, keepdims=True)
    covariance = np.cov(centered.T)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = values[order]
    basis = vectors[:, order]
    if np.linalg.det(basis) < 0:
        basis[:, 2] *= -1.0
    return basis, values


def render_point_reference(
    points: np.ndarray,
    colors: np.ndarray | None,
    category: str,
    output_path: Path,
    size: int,
    margin: int,
) -> dict[str, Any]:
    basis, eigenvalues = pca_basis(points)
    center = points.mean(axis=0)
    local = (points - center) @ basis
    axes = (0, 1)
    if category not in PLANAR_CATEGORIES:
        extents = np.percentile(local, 99, axis=0) - np.percentile(local, 1, axis=0)
        axes = tuple(int(index) for index in np.argsort(extents)[-2:][::-1])
    uv = local[:, list(axes)]
    minimum = np.percentile(uv, 1, axis=0)
    maximum = np.percentile(uv, 99, axis=0)
    span = np.maximum(maximum - minimum, 1e-6)
    scale = (size - 2 * margin) / float(max(span))
    pixel = (uv - (minimum + maximum) / 2.0) * scale + size / 2.0
    x = np.clip(np.rint(pixel[:, 0]).astype(np.int32), 0, size - 1)
    y = np.clip(np.rint(size - 1 - pixel[:, 1]).astype(np.int32), 0, size - 1)
    canvas = np.zeros((size, size, 4), dtype=np.uint8)
    draw_colors = colors if colors is not None else np.full((len(points), 3), [215, 220, 225], dtype=np.uint8)
    for xi, yi, color in zip(x, y, draw_colors):
        canvas[yi, xi, :3] = color
        canvas[yi, xi, 3] = 255
    image = Image.fromarray(canvas, mode="RGBA").filter(ImageFilter.MaxFilter(3))
    image.save(output_path)
    normal = basis[:, 2]
    front_axis = basis[:, 2]
    return {
        "path": str(output_path.resolve()),
        "size": size,
        "sha256": sha256_file(output_path),
        "projection": "pca_orthographic",
        "axes": [int(axes[0]), int(axes[1])],
        "center": center.tolist(),
        "basis": basis.tolist(),
        "eigenvalues": eigenvalues.tolist(),
        "estimated_plane_normal": normal.tolist(),
        "front_axis_scene": front_axis.tolist(),
        "front_back_audit_required": category in PLANAR_CATEGORIES,
        "note": "For planar assets the PCA normal is an orientation cue only; verify front/back visually before using as TRELLIS input.",
    }


def make_contact_sheet(items: list[dict[str, Any]], output_path: Path) -> None:
    tile_width = 360
    tile_height = 320
    columns = 3
    rows = max(1, math.ceil(len(items) / columns))
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), (28, 32, 36))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, item in enumerate(items):
        column = index % columns
        row = index // columns
        ox = column * tile_width
        oy = row * tile_height
        rgba = Image.open(item["rgba_path"]).convert("RGBA")
        ref = Image.open(item["point_reference"]["path"]).convert("RGBA")
        rgba = ImageOps.contain(rgba, (170, 210), Image.Resampling.LANCZOS)
        ref = ImageOps.contain(ref, (150, 210), Image.Resampling.LANCZOS)
        for preview, px in [(rgba, ox + 12), (ref, ox + 190)]:
            checker = Image.new("RGBA", preview.size, (74, 80, 84, 255))
            checker.alpha_composite(preview)
            sheet.paste(checker.convert("RGB"), (px, oy + 12))
        label = f"{item['object_id']}  {item['frame_id']}"
        detail = f"q={item['selection_quality']:.3f} alpha={item['alpha_pixels']} box={item['bbox_width']}x{item['bbox_height']}"
        draw.text((ox + 12, oy + tile_height - 50), label, fill=(240, 242, 243), font=font)
        draw.text((ox + 12, oy + tile_height - 30), detail, fill=(184, 190, 194), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def build_trellis2_command(input_path: Path, output_dir: Path, trellis_source_dir: str | None, weights_dir: str | None, seed: int) -> list[str] | None:
    if not trellis_source_dir or not weights_dir:
        return None
    return [
        "python",
        "-m",
        "video2world.providers.trellis2_asset",
        "--trellis-source-dir",
        trellis_source_dir,
        "--weights",
        weights_dir,
        "--input-rgba",
        str(input_path),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(seed),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-manifest", type=Path, required=True)
    parser.add_argument("--camera-info", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--masks-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--objects", nargs="*")
    parser.add_argument("--include-structure", action="store_true")
    parser.add_argument("--max-points-per-object", type=int, default=80000)
    parser.add_argument("--min-mask-probability", type=float, default=0.6)
    parser.add_argument("--min-support-points", type=int, default=40)
    parser.add_argument("--min-component-support", type=float, default=0.01)
    parser.add_argument("--reference-size", type=int, default=512)
    parser.add_argument("--reference-margin", type=int, default=36)
    parser.add_argument("--trellis-source-dir")
    parser.add_argument("--trellis-weights-dir")
    parser.add_argument("--trellis-seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.reference_size < 128:
        raise ValueError("--reference-size must be at least 128")
    object_manifest = read_json(args.object_manifest)
    camera_info = read_json(args.camera_info)
    if camera_info.get("extrinsic_type") != "world_to_camera":
        raise ValueError("camera_info.extrinsic_type must be world_to_camera")
    objects = object_manifest.get("objects")
    extrinsics = camera_info.get("extrinsic")
    if not isinstance(objects, dict) or not isinstance(extrinsics, dict):
        raise ValueError("Object manifest or camera info has an unexpected schema")
    output_root = args.output_root.resolve()
    input_dir = output_root / "input"
    reference_dir = output_root / "point_references"
    trellis_dir = output_root / "trellis2_jobs"
    input_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = {path.stem: path for path in args.frames_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}}
    wanted = set(args.objects or [])
    prepared: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}
    for ordinal, (object_id, record) in enumerate(sorted(objects.items())):
        if wanted and object_id not in wanted:
            continue
        if not isinstance(record, dict):
            skipped[object_id] = "invalid_object_record"
            continue
        category = str(record.get("category") or "")
        if category in STRUCTURE_CATEGORIES and not args.include_structure:
            skipped[object_id] = "background_structure_kept_as_scan_geometry"
            continue
        point_cloud = Path(str(record.get("path") or ""))
        if not point_cloud.is_file():
            skipped[object_id] = f"missing_point_cloud:{point_cloud}"
            continue
        mask_dir = find_mask_dir(args.masks_dir, object_id, category, record.get("name"))
        if mask_dir is None:
            skipped[object_id] = f"missing_2d_mask_dir_for_object_or_category:{object_id}/{category}"
            continue
        points_all, colors_all = read_ascii_ply(point_cloud)
        if len(points_all) <= 0:
            skipped[object_id] = "empty_point_cloud"
            continue
        points, colors = sample_points(points_all, colors_all, args.max_points_per_object, seed=20260724 + ordinal)
        candidates: list[Candidate] = []
        for mask_path in sorted(mask_dir.glob("*.png")):
            frame_id = mask_path.stem
            image_path = frame_paths.get(frame_id)
            extrinsic = extrinsics.get(frame_id)
            if image_path is None or not isinstance(extrinsic, list):
                continue
            candidate = candidate_for_frame(
                points,
                frame_id,
                image_path,
                mask_path,
                np.asarray(extrinsic, dtype=np.float64),
                intrinsic_for_frame(camera_info, frame_id),
                args.min_mask_probability,
                args.min_support_points,
            )
            if candidate is not None:
                candidates.append(candidate)
        if not candidates:
            skipped[object_id] = f"no_projected_object_support_in_masks:{mask_dir}"
            continue
        candidates.sort(key=lambda item: item.score, reverse=True)
        selected = candidates[0]
        selected_extrinsic = np.asarray(extrinsics[selected.frame_id], dtype=np.float64)
        selected_intrinsic = intrinsic_for_frame(camera_info, selected.frame_id)
        rgb, rgba, support, component_report = mask_for_candidate(
            selected,
            points,
            selected_extrinsic,
            selected_intrinsic,
            args.min_mask_probability,
            args.min_component_support,
        )
        rgb_path = input_dir / f"{object_id}_rgb.png"
        rgba_path = input_dir / f"{object_id}_rgba.png"
        support_path = input_dir / f"{object_id}_projected_support.png"
        ref_path = reference_dir / f"{object_id}_pca_reference.png"
        rgb.save(rgb_path)
        rgba.save(rgba_path)
        support.save(support_path)
        point_reference = render_point_reference(points, colors, category, ref_path, args.reference_size, args.reference_margin)
        alpha_pixels = int(np.asarray(rgba.getchannel("A"), dtype=np.uint8).astype(bool).sum())
        trellis_input_mode = "rgba_crop"
        preferred_input = rgba_path
        if category in PLANAR_CATEGORIES:
            trellis_input_mode = "rgba_crop_primary_with_point_reference_for_front_back_audit"
        command = build_trellis2_command(
            preferred_input,
            trellis_dir / object_id,
            args.trellis_source_dir,
            args.trellis_weights_dir,
            args.trellis_seed,
        )
        item = {
            "object_id": object_id,
            "name": record.get("name"),
            "category": category,
            "source_point_cloud": str(point_cloud),
            "source_point_cloud_sha256": sha256_file(point_cloud),
            "source_point_count": int(len(points_all)),
            "sampled_point_count": int(len(points)),
            "mask_dir": str(mask_dir),
            "frame_id": selected.frame_id,
            "source_image": selected.image_path,
            "source_mask": selected.mask_path,
            "bbox_xyxy": selected.bbox_xyxy,
            "bbox_width": selected.bbox_width,
            "bbox_height": selected.bbox_height,
            "bbox_area": selected.bbox_area,
            "visible_points": selected.visible_points,
            "mask_support_points": selected.mask_support_points,
            "support_ratio": selected.support_ratio,
            "selection_quality": selected.selection_quality,
            "selection_quality_breakdown": selected.selection_quality_breakdown,
            "selection_score": selected.score,
            "rgb_path": str(rgb_path),
            "rgba_path": str(rgba_path),
            "projected_support_path": str(support_path),
            "point_reference": point_reference,
            "preferred_trellis_input": str(preferred_input),
            "trellis_input_mode": trellis_input_mode,
            "trellis2_command": command,
            "alpha_pixels": alpha_pixels,
            "low_detail_input": min(selected.bbox_width, selected.bbox_height) < 96 or alpha_pixels < 2500,
            "component_filter": component_report,
            "top_candidates": [asdict(candidate) for candidate in candidates[:5]],
        }
        prepared.append(item)
        print(
            f"prepared {object_id}: frame={selected.frame_id} q={selected.selection_quality:.3f} "
            f"box={selected.bbox_width}x{selected.bbox_height} alpha={alpha_pixels}",
            flush=True,
        )
    contact_sheet = output_root / "qa" / "front_half_input_contact_sheet.png"
    make_contact_sheet(prepared, contact_sheet)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "method": "front_half_projection_ranked_single_image_and_point_reference_preparation",
        "scope": "scene_reconstruction_object_completion_background_completion_pre_qa_pre_scene_graph",
        "object_manifest": str(args.object_manifest.resolve()),
        "camera_info": str(args.camera_info.resolve()),
        "frames_dir": str(args.frames_dir.resolve()),
        "masks_dir": str(args.masks_dir.resolve()),
        "output_root": str(output_root),
        "input_contact_sheet": str(contact_sheet),
        "parameters": {
            "objects": sorted(wanted),
            "include_structure": bool(args.include_structure),
            "max_points_per_object": int(args.max_points_per_object),
            "min_mask_probability": float(args.min_mask_probability),
            "min_support_points": int(args.min_support_points),
            "min_component_support": float(args.min_component_support),
            "reference_size": int(args.reference_size),
            "reference_margin": int(args.reference_margin),
            "trellis_source_dir": args.trellis_source_dir,
            "trellis_weights_dir": args.trellis_weights_dir,
            "trellis_seed": int(args.trellis_seed),
        },
        "prepared_count": len(prepared),
        "skipped_count": len(skipped),
        "prepared": prepared,
        "skipped": skipped,
    }
    manifest_path = input_dir / "front_half_input_manifest.json"
    write_json(manifest_path, report)
    print(f"prepared {len(prepared)} inputs; skipped {len(skipped)}; manifest={manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
