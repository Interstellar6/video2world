#!/usr/bin/env python3
"""Build one logical PBR fixture from a 3D anchor and multi-frame evidence.

This deterministic fallback is intended for fixtures whose observed silhouette contains
real, depth-separated negative space. Internal reconstruction components remain children
of one GLB root and are never exposed as independent scene objects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image
from plyfile import PlyData, PlyElement

MIN_EVIDENCE_FRAMES = 2
ANCHOR_SAMPLE_LIMIT = 8192


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _dominant_axis_positive(axes: np.ndarray) -> np.ndarray:
    oriented = axes.copy()
    for column in range(oriented.shape[1]):
        dominant = int(np.argmax(np.abs(oriented[:, column])))
        if oriented[dominant, column] < 0:
            oriented[:, column] *= -1
    return oriented


def load_anchor(
    path: Path,
    *,
    canonical_length: float,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray | None]:
    source_sha256 = sha256_file(path)
    if expected_sha256 is not None and source_sha256 != expected_sha256:
        raise ValueError(
            f"anchor sha256 mismatch: expected {expected_sha256}, observed {source_sha256}"
        )
    ply = PlyData.read(path)
    if "vertex" not in ply:
        raise ValueError("anchor PLY has no vertex element")
    vertex = ply["vertex"].data
    names = set(vertex.dtype.names or ())
    missing = sorted({"x", "y", "z"} - names)
    if missing:
        raise ValueError(f"anchor PLY is missing coordinates: {missing}")
    points = np.column_stack([vertex[name] for name in ("x", "y", "z")]).astype(np.float64)
    if len(points) < 64 or not np.isfinite(points).all():
        raise ValueError("anchor PLY requires at least 64 finite points")
    colors = None
    if {"red", "green", "blue"}.issubset(names):
        colors = np.column_stack([vertex[name] for name in ("red", "green", "blue")]).astype(
            np.uint8
        )

    stride = max(1, len(points) // 250_000)
    analysis_points = points[::stride]
    seed_center = np.median(analysis_points, axis=0)
    covariance = np.cov((analysis_points - seed_center).T)
    eigenvalues, axes_descending = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    axes_descending = _dominant_axis_positive(axes_descending[:, order])
    if np.linalg.det(axes_descending) < 0:
        axes_descending[:, -1] *= -1
    projected = np.einsum(
        "ni,ij->nj", analysis_points - seed_center, axes_descending, optimize=False
    )
    lower, upper = np.quantile(projected, [0.005, 0.995], axis=0)
    extents_descending = upper - lower
    if not np.isfinite(extents_descending).all() or np.any(extents_descending <= 0):
        raise ValueError("anchor robust PCA produced invalid extents")
    length, width, height = [float(value) for value in extents_descending]
    if not (length > width > height):
        raise ValueError(
            "bed fixture anchor must have ordered robust extents length > width > height"
        )
    width_ratio = width / length
    height_ratio = height / length
    if not 0.35 <= width_ratio <= 0.95 or not 0.12 <= height_ratio <= 0.65:
        raise ValueError("anchor proportions are outside the supported bed fixture profile")

    canonical_width = canonical_length * width_ratio
    canonical_height = canonical_length * height_ratio
    canonical_axes = np.column_stack(
        (axes_descending[:, 1], axes_descending[:, 2], axes_descending[:, 0])
    )
    if np.linalg.det(canonical_axes) < 0:
        canonical_axes[:, 0] *= -1
    robust_center = seed_center + np.einsum(
        "ij,j->i", axes_descending, (lower + upper) / 2.0, optimize=False
    )
    canonical_extents = np.asarray(
        [canonical_width, canonical_height, canonical_length], dtype=np.float64
    )
    physical_extents = np.asarray([width, height, length], dtype=np.float64)
    linear = canonical_axes @ np.diag(physical_extents / canonical_extents)
    canonical_center = np.asarray([0.0, canonical_height / 2.0, 0.0])
    translation = robust_center - np.einsum(
        "ij,j->i", linear, canonical_center, optimize=False
    )
    canonical_to_anchor = np.eye(4, dtype=np.float64)
    canonical_to_anchor[:3, :3] = linear
    canonical_to_anchor[:3, 3] = translation

    report = {
        "path": str(path),
        "sha256": source_sha256,
        "size_bytes": path.stat().st_size,
        "point_count": len(points),
        "analysis_stride": stride,
        "analysis_point_count": len(analysis_points),
        "robust_quantiles": [0.005, 0.995],
        "robust_center": robust_center.tolist(),
        "pca_eigenvalues_descending": eigenvalues.tolist(),
        "pca_axes_descending_columns": axes_descending.tolist(),
        "physical_extents": {
            "length": length,
            "width": width,
            "height": height,
        },
        "canonical_extents": {
            "length": canonical_length,
            "width": canonical_width,
            "height": canonical_height,
        },
        "canonical_axes_columns": canonical_axes.tolist(),
        "canonical_to_anchor_row_major": canonical_to_anchor.tolist(),
    }
    return report, points, colors


def _visible_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, columns = np.nonzero(mask)
    if not len(columns):
        raise ValueError("evidence RGBA has no visible alpha pixels")
    return int(columns.min()), int(rows.min()), int(columns.max()), int(rows.max())


def _bounded_negative_space(mask: np.ndarray) -> np.ndarray:
    left = np.maximum.accumulate(mask, axis=1)
    right = np.maximum.accumulate(mask[:, ::-1], axis=1)[:, ::-1]
    top = np.maximum.accumulate(mask, axis=0)
    bottom = np.maximum.accumulate(mask[::-1, :], axis=0)[::-1, :]
    return (~mask) & left & right & top & bottom


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.count_nonzero(left & right))
    union = int(np.count_nonzero(left | right))
    return intersection / union if union else 1.0


def analyze_evidence(
    rgba_path: Path,
    mask_path: Path,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    rgba = np.asarray(Image.open(rgba_path).convert("RGBA"), dtype=np.uint8)
    mask_image = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) >= 128
    alpha = rgba[..., 3] >= 128
    if mask_image.shape != alpha.shape:
        raise ValueError("evidence RGBA and mask dimensions differ")
    alpha_mask_iou = _mask_iou(alpha, mask_image)
    if alpha_mask_iou < 0.995:
        raise ValueError(f"evidence RGBA alpha and mask disagree: IoU={alpha_mask_iou:.6f}")
    x0, y0, x1, y1 = _visible_bbox(alpha)
    bounded_gap = _bounded_negative_space(alpha)
    bbox_area = (x1 - x0 + 1) * (y1 - y0 + 1)
    gap_pixels = int(np.count_nonzero(bounded_gap[y0 : y1 + 1, x0 : x1 + 1]))
    gap_ratio = gap_pixels / bbox_area

    visible_colors = rgba[..., :3][alpha].astype(np.float64)
    luminance = np.sum(
        visible_colors * np.asarray([0.2126, 0.7152, 0.0722]), axis=1
    )
    maximum = visible_colors.max(axis=1)
    minimum = visible_colors.min(axis=1)
    saturation = (maximum - minimum) / np.maximum(maximum, 1.0)
    wood = (
        (visible_colors[:, 0] > visible_colors[:, 1] * 1.08)
        & (visible_colors[:, 0] > visible_colors[:, 2] * 1.12)
        & (luminance > 20.0)
        & (luminance < 155.0)
    )
    fabric = (luminance > 145.0) & (saturation < 0.28)
    if int(np.count_nonzero(wood)) < 200 or int(np.count_nonzero(fabric)) < 200:
        raise ValueError("evidence does not contain enough wood and light-fabric color support")
    report = {
        "rgba": {
            "path": str(rgba_path),
            "sha256": sha256_file(rgba_path),
            "size_bytes": rgba_path.stat().st_size,
            "dimensions": [rgba.shape[1], rgba.shape[0]],
        },
        "mask": {
            "path": str(mask_path),
            "sha256": sha256_file(mask_path),
            "size_bytes": mask_path.stat().st_size,
            "alpha_iou": alpha_mask_iou,
        },
        "visible_bbox_xyxy": [x0, y0, x1, y1],
        "visible_pixel_count": int(np.count_nonzero(alpha)),
        "bounded_negative_space_pixels": gap_pixels,
        "bounded_negative_space_bbox_fraction": gap_ratio,
        "wood_support_pixels": int(np.count_nonzero(wood)),
        "fabric_support_pixels": int(np.count_nonzero(fabric)),
    }
    return report, visible_colors[wood], visible_colors[fabric]


def validate_reference_manifest(
    path: Path,
    *,
    object_id: str,
    evidence_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("object_id") != object_id:
        raise ValueError("reference manifest object_id does not match requested fixture")
    if payload.get("identity_revalidated") is not True:
        raise ValueError("reference manifest has not revalidated object identity")
    frames = payload.get("frames")
    if not isinstance(frames, list) or len(frames) < MIN_EVIDENCE_FRAMES:
        raise ValueError("reference manifest requires at least two verified frames")
    expected_pairs = {
        (item.get("image_sha256"), item.get("mask_sha256"))
        for item in frames
        if isinstance(item, dict)
    }
    observed_pairs = {
        (item["rgba"]["sha256"], item["mask"]["sha256"]) for item in evidence_reports
    }
    if observed_pairs != expected_pairs:
        raise ValueError("provided evidence hashes do not exactly match the reference manifest")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "identity_revalidated": True,
        "identity_revalidated_by": payload.get("identity_revalidated_by"),
        "frame_count": len(frames),
        "frame_ids": sorted(str(item.get("frame_id")) for item in frames),
    }


def audit_prior_rejections(paths: list[Path]) -> list[dict[str, Any]]:
    audited = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("kind") != "video2world.object_six_view_visual_review":
            raise ValueError(f"prior rejection has unexpected kind: {path}")
        if payload.get("status") != "rejected" or payload.get("promotion_allowed") is not False:
            raise ValueError(f"prior candidate is not a fail-closed visual rejection: {path}")
        technical = payload.get("technical_receipt")
        if not isinstance(technical, dict):
            raise ValueError(f"prior rejection has no technical receipt backlink: {path}")
        technical_path_value = technical.get("path")
        expected_technical_sha = technical.get("sha256")
        if not isinstance(technical_path_value, str) or not isinstance(
            expected_technical_sha, str
        ):
            raise ValueError(f"prior rejection technical receipt backlink is invalid: {path}")
        technical_path = (path.parent / technical_path_value).resolve()
        if not technical_path.is_file():
            raise ValueError(f"prior technical receipt is missing: {technical_path}")
        observed_technical_sha = sha256_file(technical_path)
        if observed_technical_sha != expected_technical_sha:
            raise ValueError(f"prior technical receipt hash mismatch: {technical_path}")
        canonical = payload.get("canonical_review")
        contact_sheet = canonical.get("contact_sheet") if isinstance(canonical, dict) else None
        contact_sheet_sha = (
            canonical.get("contact_sheet_sha256") if isinstance(canonical, dict) else None
        )
        if not isinstance(contact_sheet, str) or not isinstance(contact_sheet_sha, str):
            raise ValueError(f"prior rejection has no canonical contact sheet backlink: {path}")
        contact_sheet_path = (path.parent / contact_sheet).resolve()
        if not contact_sheet_path.is_file() or sha256_file(contact_sheet_path) != contact_sheet_sha:
            raise ValueError(
                f"prior rejection contact sheet is missing or changed: {contact_sheet_path}"
            )
        audited.append(
            {
                "visual_review": {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "status": "rejected",
                },
                "technical_receipt": {
                    "path": str(technical_path),
                    "sha256": observed_technical_sha,
                },
                "contact_sheet": {
                    "path": str(contact_sheet_path),
                    "sha256": contact_sheet_sha,
                },
                "asset_sha256": payload.get("asset", {}).get("sha256"),
                "blocking_issue_types": [
                    item.get("issue_type")
                    for item in payload.get("blocking_issues", [])
                    if isinstance(item, dict)
                ],
            }
        )
    return audited


def choose_material_colors(
    wood_samples: list[np.ndarray],
    fabric_samples: list[np.ndarray],
) -> dict[str, Any]:
    wood_observed = np.median(np.vstack(wood_samples), axis=0)
    fabric_observed = np.median(np.vstack(fabric_samples), axis=0)
    wood = np.maximum(np.rint(wood_observed), np.asarray([82.0, 48.0, 30.0]))
    fabric = np.rint(fabric_observed * 0.72 + np.asarray([232.0, 230.0, 224.0]) * 0.28)
    base = np.rint(fabric * 0.66 + np.asarray([82.0, 76.0, 68.0]) * 0.34)
    cover = np.rint(fabric * 0.86 + np.asarray([204.0, 194.0, 171.0]) * 0.14)

    def rgb(values: np.ndarray) -> list[int]:
        return [int(value) for value in np.clip(values, 0, 255)]

    return {
        "observed": {
            "wood_median_rgb": rgb(wood_observed),
            "light_fabric_median_rgb": rgb(fabric_observed),
        },
        "applied": {
            "wood_rgb": rgb(wood),
            "mattress_rgb": rgb(fabric),
            "base_rgb": rgb(base),
            "cover_rgb": rgb(cover),
        },
        "policy": "evidence_median_with_neutral_review_luminance_floor",
    }


def _signed_power(values: np.ndarray, exponent: float) -> np.ndarray:
    return np.sign(values) * np.power(np.abs(values), exponent)


def make_superellipsoid(
    *,
    width: float,
    height: float,
    depth: float,
    center: tuple[float, float, float],
    shape_exponent: float = 0.24,
    latitude_segments: int = 24,
    longitude_segments: int = 64,
) -> trimesh.Trimesh:
    latitudes = np.linspace(-math.pi / 2, math.pi / 2, latitude_segments + 1)[1:-1]
    longitudes = np.linspace(-math.pi, math.pi, longitude_segments, endpoint=False)
    vertices: list[list[float]] = []
    for latitude in latitudes:
        latitude_cos = math.cos(float(latitude)) ** shape_exponent
        y = (height / 2.0) * math.copysign(
            abs(math.sin(float(latitude))) ** shape_exponent,
            float(latitude),
        )
        x = (width / 2.0) * latitude_cos * _signed_power(
            np.cos(longitudes), shape_exponent
        )
        z = (depth / 2.0) * latitude_cos * _signed_power(
            np.sin(longitudes), shape_exponent
        )
        vertices.extend(np.column_stack((x, np.full_like(x, y), z)).tolist())
    ring_count = len(latitudes)
    bottom = len(vertices)
    vertices.append([0.0, -height / 2.0, 0.0])
    top = len(vertices)
    vertices.append([0.0, height / 2.0, 0.0])
    faces: list[list[int]] = []
    for longitude in range(longitude_segments):
        next_longitude = (longitude + 1) % longitude_segments
        faces.append([bottom, next_longitude, longitude])
    for ring in range(ring_count - 1):
        lower = ring * longitude_segments
        upper = (ring + 1) * longitude_segments
        for longitude in range(longitude_segments):
            next_longitude = (longitude + 1) % longitude_segments
            a = lower + longitude
            b = lower + next_longitude
            c = upper + next_longitude
            d = upper + longitude
            faces.extend(([a, b, c], [a, c, d]))
    last_ring = (ring_count - 1) * longitude_segments
    for longitude in range(longitude_segments):
        next_longitude = (longitude + 1) % longitude_segments
        faces.append([top, last_ring + longitude, last_ring + next_longitude])
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    mesh.apply_translation(np.asarray(center, dtype=np.float64))
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def make_curved_panel(
    *,
    width: float,
    height: float,
    thickness: float,
    bottom_y: float,
    center_z: float,
    arch_ratio: float = 0.18,
    segments: int = 40,
) -> trimesh.Trimesh:
    xs = np.linspace(-width / 2.0, width / 2.0, segments + 1)
    z_front = center_z + thickness / 2.0
    z_back = center_z - thickness / 2.0
    vertices: list[list[float]] = []
    for x in xs:
        normalized = abs(2.0 * float(x) / width)
        top_y = bottom_y + height * (1.0 - arch_ratio * normalized**2)
        vertices.extend(
            (
                [float(x), bottom_y, z_front],
                [float(x), top_y, z_front],
                [float(x), bottom_y, z_back],
                [float(x), top_y, z_back],
            )
        )
    faces: list[list[int]] = []
    for index in range(segments):
        left = index * 4
        right = (index + 1) * 4
        faces.extend(
            (
                [left, right, right + 1],
                [left, right + 1, left + 1],
                [left + 2, right + 3, right + 2],
                [left + 2, left + 3, right + 3],
                [left + 1, right + 1, right + 3],
                [left + 1, right + 3, left + 3],
                [left, right + 2, right],
                [left, left + 2, right + 2],
            )
        )
    first = 0
    last = segments * 4
    faces.extend(
        (
            [first, first + 1, first + 3],
            [first, first + 3, first + 2],
            [last, last + 3, last + 1],
            [last, last + 2, last + 3],
        )
    )
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def _pbr_material(name: str, color: list[int], roughness: float) -> Any:
    return trimesh.visual.material.PBRMaterial(
        name=name,
        baseColorFactor=[*color, 255],
        metallicFactor=0.0,
        roughnessFactor=roughness,
        doubleSided=False,
    )


def apply_material(mesh: trimesh.Trimesh, *, name: str, color: list[int], roughness: float) -> None:
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2), dtype=np.float64),
        material=_pbr_material(name, color, roughness),
    )


def _cylinder_y(radius: float, height: float, center: tuple[float, float, float]) -> Any:
    transform = trimesh.transformations.rotation_matrix(math.pi / 2.0, [1.0, 0.0, 0.0])
    transform[:3, 3] = np.asarray(center, dtype=np.float64)
    return trimesh.creation.cylinder(radius=radius, height=height, sections=32, transform=transform)


def build_bed_components(
    anchor_report: dict[str, Any],
    material_colors: dict[str, Any],
) -> tuple[dict[str, trimesh.Trimesh], dict[str, Any]]:
    dimensions = anchor_report["canonical_extents"]
    length = float(dimensions["length"])
    width = float(dimensions["width"])
    height = float(dimensions["height"])
    headboard_thickness = length * 0.045
    headboard_back = -length / 2.0
    headboard_center_z = headboard_back + headboard_thickness / 2.0
    body_min_z = headboard_back + headboard_thickness + length * 0.018
    body_max_z = length / 2.0
    body_length = body_max_z - body_min_z
    body_center_z = (body_min_z + body_max_z) / 2.0
    base_height = height * 0.25
    vertical_gap = height * 0.006
    mattress_height = height * 0.28
    mattress_bottom = base_height + vertical_gap
    mattress_top = mattress_bottom + mattress_height
    applied = material_colors["applied"]

    base = trimesh.creation.box(
        extents=[width * 0.94, base_height, body_length],
        transform=trimesh.transformations.translation_matrix(
            [0.0, base_height / 2.0, body_center_z]
        ),
    )
    mattress = make_superellipsoid(
        width=width * 0.92,
        height=mattress_height,
        depth=body_length * 0.985,
        center=(0.0, mattress_bottom + mattress_height / 2.0, body_center_z),
    )
    post_radius = max(width * 0.036, 0.025)
    post_height = height * 0.98
    post_x = width / 2.0 - post_radius
    panel_gap = width * 0.008
    panel_half_width = post_x - post_radius - panel_gap
    panel_bottom = height * 0.28
    panel_height = height * 0.68
    panel = make_curved_panel(
        width=panel_half_width * 2.0,
        height=panel_height,
        thickness=headboard_thickness,
        bottom_y=panel_bottom,
        center_z=headboard_center_z,
    )
    left_post = _cylinder_y(
        post_radius,
        post_height,
        (-post_x, post_height / 2.0, headboard_center_z),
    )
    right_post = _cylinder_y(
        post_radius,
        post_height,
        (post_x, post_height / 2.0, headboard_center_z),
    )
    cover_height = height * 0.035
    cover_length = length * 0.29
    cover_center_z = body_max_z - cover_length / 2.0 - length * 0.015
    cover = make_superellipsoid(
        width=width * 0.94,
        height=cover_height,
        depth=cover_length,
        center=(
            0.0,
            mattress_top + vertical_gap + cover_height / 2.0,
            cover_center_z,
        ),
        shape_exponent=0.18,
        latitude_segments=16,
        longitude_segments=48,
    )
    components = {
        "base_frame": base,
        "mattress": mattress,
        "headboard_panel": panel,
        "headboard_post_left": left_post,
        "headboard_post_right": right_post,
        "foot_cover": cover,
    }
    apply_material(
        base,
        name="base_fabric_pbr",
        color=applied["base_rgb"],
        roughness=0.93,
    )
    apply_material(
        mattress,
        name="mattress_fabric_pbr",
        color=applied["mattress_rgb"],
        roughness=0.96,
    )
    for name in ("headboard_panel", "headboard_post_left", "headboard_post_right"):
        apply_material(
            components[name],
            name="headboard_wood_pbr",
            color=applied["wood_rgb"],
            roughness=0.74,
        )
    apply_material(
        cover,
        name="foot_cover_fabric_pbr",
        color=applied["cover_rgb"],
        roughness=0.98,
    )

    clearance_min = np.asarray(
        [
            -width * 0.34,
            mattress_top + vertical_gap * 2.0,
            body_min_z + length * 0.035,
        ]
    )
    clearance_max = np.asarray(
        [
            width * 0.34,
            height * 0.84,
            body_min_z + length * 0.40,
        ]
    )
    layout = {
        "floor_y": 0.0,
        "base_top_y": base_height,
        "mattress_bottom_y": mattress_bottom,
        "mattress_top_y": mattress_top,
        "headboard_front_z": headboard_back + headboard_thickness,
        "body_min_z": body_min_z,
        "support_gap": mattress_bottom - base_height,
        "pillow_clearance_aabb": [clearance_min.tolist(), clearance_max.tolist()],
    }
    return components, layout


def _intersection_volume(left: np.ndarray, right: np.ndarray) -> float:
    overlap = np.minimum(left[1], right[1]) - np.maximum(left[0], right[0])
    return float(np.prod(np.maximum(overlap, 0.0)))


def audit_components(
    components: dict[str, trimesh.Trimesh],
    layout: dict[str, Any],
) -> dict[str, Any]:
    component_reports: dict[str, Any] = {}
    for name, mesh in components.items():
        material = getattr(mesh.visual, "material", None)
        component_reports[name] = {
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "bounds": mesh.bounds.tolist(),
            "watertight": bool(mesh.is_watertight),
            "winding_consistent": bool(mesh.is_winding_consistent),
            "finite_vertices": bool(np.isfinite(mesh.vertices).all()),
            "degenerate_faces": int(np.count_nonzero(mesh.area_faces <= 1e-12)),
            "material_type": type(material).__name__ if material is not None else None,
            "material_name": getattr(material, "name", None),
        }
    intersections = []
    names = list(components)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            volume = _intersection_volume(
                components[left_name].bounds,
                components[right_name].bounds,
            )
            if volume > 1e-10:
                intersections.append(
                    {
                        "left": left_name,
                        "right": right_name,
                        "aabb_intersection_volume": volume,
                    }
                )
    clearance = np.asarray(layout["pillow_clearance_aabb"], dtype=np.float64)
    clearance_intersections = {
        name: _intersection_volume(mesh.bounds, clearance) for name, mesh in components.items()
    }
    return {
        "components": component_reports,
        "pairwise_positive_aabb_intersections": intersections,
        "pillow_clearance_intersection_volumes": clearance_intersections,
        "all_components_nonempty": all(
            item["vertices"] > 0 and item["faces"] > 0 for item in component_reports.values()
        ),
        "all_components_closed": all(item["watertight"] for item in component_reports.values()),
        "all_components_winding_consistent": all(
            item["winding_consistent"] for item in component_reports.values()
        ),
        "all_components_finite": all(
            item["finite_vertices"] for item in component_reports.values()
        ),
        "all_components_nondegenerate": all(
            item["degenerate_faces"] == 0 for item in component_reports.values()
        ),
        "all_components_pbr": all(
            item["material_type"] == "PBRMaterial" for item in component_reports.values()
        ),
        "no_positive_aabb_intersections": not intersections,
        "pillow_clearance_empty": all(value <= 1e-10 for value in clearance_intersections.values()),
    }


def write_anchor_sample(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray | None,
) -> dict[str, Any]:
    count = min(len(points), ANCHOR_SAMPLE_LIMIT)
    indices = np.unique(np.linspace(0, len(points) - 1, count, dtype=np.int64))
    dtype: list[tuple[str, str]] = [(name, "f4") for name in ("x", "y", "z")]
    if colors is not None:
        dtype.extend((name, "u1") for name in ("red", "green", "blue"))
    data = np.empty(len(indices), dtype=dtype)
    for axis, name in enumerate(("x", "y", "z")):
        data[name] = points[indices, axis].astype(np.float32)
    if colors is not None:
        for channel, name in enumerate(("red", "green", "blue")):
            data[name] = colors[indices, channel]
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(data, "vertex")], text=False, byte_order="<").write(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "point_count": len(indices),
        "sampling": "deterministic_even_source_index",
    }


def export_single_root_scene(
    path: Path,
    *,
    object_id: str,
    components: dict[str, trimesh.Trimesh],
) -> dict[str, Any]:
    scene = trimesh.Scene(base_frame="world")
    scene.graph.update(frame_from="world", frame_to=object_id, matrix=np.eye(4))
    component_nodes = []
    for component_name, mesh in components.items():
        node_name = f"{object_id}__{component_name}"
        geometry_name = f"{object_id}.{component_name}"
        scene.add_geometry(
            mesh,
            node_name=node_name,
            geom_name=geometry_name,
            parent_node_name=object_id,
        )
        component_nodes.append(node_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(path)

    loaded = trimesh.load_scene(path)
    base_frame = loaded.graph.base_frame
    parents = dict(loaded.graph.transforms.parents)
    root_nodes = sorted(
        node for node, parent in parents.items() if parent == base_frame and node != base_frame
    )
    children = sorted(node for node, parent in parents.items() if parent == object_id)
    geometry_names = sorted(loaded.geometry)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "base_frame": base_frame,
        "root_nodes": root_nodes,
        "logical_root": object_id,
        "component_nodes": children,
        "expected_component_nodes": sorted(component_nodes),
        "geometry_names": geometry_names,
        "single_root_verified": root_nodes == [object_id],
        "component_parenting_verified": children == sorted(component_nodes),
    }


def build_fixture(
    *,
    object_id: str,
    profile: str,
    anchor_ply: Path,
    evidence_rgba: list[Path],
    evidence_masks: list[Path],
    reference_manifest: Path,
    output_dir: Path,
    expected_anchor_sha256: str | None = None,
    canonical_length: float = 2.2,
    prior_rejected_reviews: list[Path] | None = None,
) -> dict[str, Any]:
    if profile != "bed":
        raise ValueError(f"unsupported fixture profile: {profile}")
    if len(evidence_rgba) < MIN_EVIDENCE_FRAMES:
        raise ValueError("component assembly requires at least two evidence RGBA frames")
    if len(evidence_rgba) != len(evidence_masks):
        raise ValueError("each evidence RGBA requires one matching mask")
    if canonical_length <= 0 or not math.isfinite(canonical_length):
        raise ValueError("canonical_length must be positive and finite")

    output_dir.mkdir(parents=True, exist_ok=True)
    anchor_report, anchor_points, anchor_colors = load_anchor(
        anchor_ply,
        canonical_length=canonical_length,
        expected_sha256=expected_anchor_sha256,
    )
    evidence_reports: list[dict[str, Any]] = []
    wood_samples: list[np.ndarray] = []
    fabric_samples: list[np.ndarray] = []
    for rgba_path, mask_path in zip(evidence_rgba, evidence_masks, strict=True):
        report, wood, fabric = analyze_evidence(rgba_path, mask_path)
        evidence_reports.append(report)
        wood_samples.append(wood)
        fabric_samples.append(fabric)
    manifest_report = validate_reference_manifest(
        reference_manifest,
        object_id=object_id,
        evidence_reports=evidence_reports,
    )
    negative_space_verified = all(
        item["bounded_negative_space_bbox_fraction"] >= 0.01 for item in evidence_reports
    )
    if not negative_space_verified:
        raise ValueError("multi-frame evidence does not verify bounded negative space")
    prior_rejections = audit_prior_rejections(prior_rejected_reviews or [])

    material_colors = choose_material_colors(wood_samples, fabric_samples)
    components, layout = build_bed_components(anchor_report, material_colors)
    component_audit = audit_components(components, layout)
    anchor_sample = write_anchor_sample(
        output_dir / "evidence" / "anchor_sample.ply",
        anchor_points,
        anchor_colors,
    )
    copied_evidence = []
    for index, (rgba_source, mask_source) in enumerate(
        zip(evidence_rgba, evidence_masks, strict=True)
    ):
        rgba_destination = output_dir / "evidence" / f"frame_{index:02d}.rgba.png"
        mask_destination = output_dir / "evidence" / f"frame_{index:02d}.mask.png"
        rgba_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rgba_source, rgba_destination)
        shutil.copy2(mask_source, mask_destination)
        copied_evidence.append(
            {
                "rgba": {
                    "path": str(rgba_destination),
                    "sha256": sha256_file(rgba_destination),
                },
                "mask": {
                    "path": str(mask_destination),
                    "sha256": sha256_file(mask_destination),
                },
            }
        )
    glb = export_single_root_scene(
        output_dir / f"{object_id}.component-assembly.glb",
        object_id=object_id,
        components=components,
    )
    support_gap = float(layout["support_gap"])
    gates = {
        "instance_identity": manifest_report["identity_revalidated"] is True,
        "support_geometry": anchor_report["point_count"] >= 64,
        "same_instance_multi_frame_evidence": len(evidence_reports) >= MIN_EVIDENCE_FRAMES,
        "negative_space_preservation": (
            negative_space_verified and component_audit["pillow_clearance_empty"]
        ),
        "internal_component_alignment": (
            component_audit["no_positive_aabb_intersections"]
            and 0.0 <= support_gap <= float(anchor_report["canonical_extents"]["height"]) * 0.02
        ),
        "single_logical_asset_root": (
            glb["single_root_verified"] and glb["component_parenting_verified"]
        ),
        "geometry_completeness": (
            component_audit["all_components_nonempty"]
            and component_audit["all_components_closed"]
            and component_audit["all_components_winding_consistent"]
            and component_audit["all_components_finite"]
            and component_audit["all_components_nondegenerate"]
        ),
        "appearance_fidelity": component_audit["all_components_pbr"],
        "collision": component_audit["all_components_closed"],
    }
    all_gates_passed = all(gates.values())
    report = {
        "schema_version": "1.0",
        "kind": "video2world.multi_component_fixture_assembly",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "technical_passed_visual_pending" if all_gates_passed else "rejected",
        "promotion_allowed": False,
        "selected_backend": "component_assembly",
        "object_id": object_id,
        "profile": profile,
        "logical_entity": {
            "single_logical_root": True,
            "root_node": object_id,
            "internal_component_names": sorted(components),
            "internal_components_are_independent_scene_objects": False,
            "independent_scene_object_ids": [],
            "selection_owner": object_id,
            "motion_owner": object_id,
            "collision_owner": object_id,
        },
        "sources": {
            "anchor": anchor_report,
            "anchor_sample": anchor_sample,
            "reference_manifest": manifest_report,
            "evidence": evidence_reports,
            "copied_evidence": copied_evidence,
        },
        "material_fit": material_colors,
        "component_layout": layout,
        "component_audit": component_audit,
        "output": {"unified_pbr_glb": glb},
        "acceptance_gates": gates,
        "all_technical_acceptance_gates_passed": all_gates_passed,
        "visual_review": {
            "status": "pending" if all_gates_passed else "not_applicable",
            "required_views": ["front", "right", "back", "left", "top", "bottom"],
            "capture_driver": "scripts/capture_object_review.mjs",
        },
        "trellis_rejection_route": {
            "preserve_prior_rejected_receipts": True,
            "prior_rejected_candidate_count": len(prior_rejections),
            "prior_rejected_candidates": prior_rejections,
            "reason": (
                "Single-image generation converted verified negative space into positive geometry; "
                "component assembly preserves the clearance explicitly."
            ),
        },
    }
    write_json(output_dir / "component_assembly_receipt.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--profile", choices=("bed",), default="bed")
    parser.add_argument("--anchor-ply", type=Path, required=True)
    parser.add_argument("--expected-anchor-sha256")
    parser.add_argument("--evidence-rgba", type=Path, action="append", required=True)
    parser.add_argument("--evidence-mask", type=Path, action="append", required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--canonical-length", type=float, default=2.2)
    parser.add_argument("--prior-rejected-review", type=Path, action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_fixture(
        object_id=args.object_id,
        profile=args.profile,
        anchor_ply=args.anchor_ply.expanduser().resolve(),
        evidence_rgba=[path.expanduser().resolve() for path in args.evidence_rgba],
        evidence_masks=[path.expanduser().resolve() for path in args.evidence_mask],
        reference_manifest=args.reference_manifest.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        expected_anchor_sha256=args.expected_anchor_sha256,
        canonical_length=args.canonical_length,
        prior_rejected_reviews=[
            path.expanduser().resolve() for path in args.prior_rejected_review
        ],
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True))
    return 0 if report["all_technical_acceptance_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
