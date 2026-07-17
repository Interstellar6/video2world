#!/usr/bin/env python3
"""Split a merged object point cloud with prompted instance masks from one frame.

The projection convention intentionally matches the accepted Video2Mesh Bedroom4
verification: camera_info extrinsics are world-to-camera matrices and points are
transformed as ``points @ R.T + t`` before applying PINHOLE intrinsics.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import PIL
from PIL import Image

PLY_TYPES: dict[str, str] = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}
OBJECT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class PlyCloud:
    source_format: str
    properties: list[tuple[str, str]]
    vertices: np.ndarray


@dataclass(frozen=True)
class InstanceMask:
    object_id: str
    metadata: dict[str, Any]
    path: Path
    sha256: str
    mask: np.ndarray
    bbox_xyxy: list[int]
    prompt_box_xyxy: list[float]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_path_remaps(values: list[str]) -> list[tuple[Path, Path]]:
    remaps: list[tuple[Path, Path]] = []
    for value in values:
        source_value, separator, target_value = value.partition("=")
        if not separator or not source_value or not target_value:
            raise ValueError(f"path remap must be ABSOLUTE_SOURCE=LOCAL_TARGET: {value!r}")
        source = Path(source_value).expanduser()
        target = Path(target_value).expanduser()
        if not source.is_absolute() or not target.is_absolute():
            raise ValueError(f"path remap endpoints must be absolute: {value!r}")
        remaps.append((source, target))
    return remaps


def resolve_input_path(
    value: str,
    *,
    relative_to: Path,
    path_remaps: list[tuple[Path, Path]],
) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        for source, target in path_remaps:
            try:
                relative = path.relative_to(source)
            except ValueError:
                continue
            return (target / relative).resolve()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def read_ply_cloud(path: Path) -> PlyCloud:
    properties: list[tuple[str, str]] = []
    elements: list[tuple[str, int]] = []
    source_format: str | None = None
    current_element: str | None = None

    with path.open("rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError(f"not a PLY file: {path}")
        while True:
            raw_line = handle.readline()
            if not raw_line:
                raise ValueError(f"PLY header has no end_header: {path}")
            line = raw_line.decode("ascii").strip()
            if not line or line.startswith("comment") or line.startswith("obj_info"):
                continue
            parts = line.split()
            if parts[0] == "format":
                if len(parts) != 3 or parts[2] != "1.0":
                    raise ValueError(f"unsupported PLY format declaration: {line}")
                source_format = parts[1]
            elif parts[0] == "element":
                if len(parts) != 3:
                    raise ValueError(f"invalid PLY element declaration: {line}")
                current_element = parts[1]
                elements.append((current_element, int(parts[2])))
            elif parts[0] == "property" and current_element == "vertex":
                if len(parts) != 3 or parts[1] == "list":
                    raise ValueError("vertex list properties are not supported")
                if parts[1] not in PLY_TYPES:
                    raise ValueError(f"unsupported PLY scalar type: {parts[1]}")
                properties.append((parts[2], parts[1]))
            elif parts[0] == "end_header":
                break

        if source_format not in {"ascii", "binary_little_endian", "binary_big_endian"}:
            raise ValueError(f"unsupported PLY format: {source_format}")
        nonempty_nonvertex = [
            (name, count) for name, count in elements if name != "vertex" and count
        ]
        if nonempty_nonvertex:
            raise ValueError(
                "input must be a point cloud; non-empty non-vertex elements found: "
                f"{nonempty_nonvertex}"
            )
        vertex_counts = [count for name, count in elements if name == "vertex"]
        if len(vertex_counts) != 1:
            raise ValueError("PLY must contain exactly one vertex element")
        if not {"x", "y", "z"}.issubset({name for name, _ in properties}):
            raise ValueError("PLY vertex properties must contain x, y, and z")
        if len({name for name, _ in properties}) != len(properties):
            raise ValueError("PLY vertex property names must be unique")

        endian = "<" if source_format == "binary_little_endian" else ">"
        if vertex_counts[0] == 0:
            dtype = np.dtype([(name, PLY_TYPES[type_name]) for name, type_name in properties])
            vertices = np.empty(0, dtype=dtype)
        elif source_format == "ascii":
            dtype = np.dtype([(name, PLY_TYPES[type_name]) for name, type_name in properties])
            vertices = np.loadtxt(handle, dtype=dtype, max_rows=vertex_counts[0], ndmin=1)
        else:
            dtype = np.dtype(
                [(name, endian + PLY_TYPES[type_name]) for name, type_name in properties]
            )
            vertices = np.fromfile(handle, dtype=dtype, count=vertex_counts[0])

    if len(vertices) != vertex_counts[0]:
        raise ValueError(
            f"PLY vertex count mismatch: header={vertex_counts[0]}, decoded={len(vertices)}"
        )
    return PlyCloud(source_format=source_format, properties=properties, vertices=vertices)


def write_ply_cloud(
    path: Path,
    cloud: PlyCloud,
    vertices: np.ndarray,
    *,
    output_format: str,
) -> None:
    if output_format == "same":
        output_format = cloud.source_format
    if output_format not in {"ascii", "binary_little_endian", "binary_big_endian"}:
        raise ValueError(f"unsupported output PLY format: {output_format}")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["ply", f"format {output_format} 1.0", "comment video2world instance split"]
    header.append(f"element vertex {len(vertices)}")
    header.extend(f"property {type_name} {name}" for name, type_name in cloud.properties)
    header.append("end_header")

    with path.open("wb") as handle:
        handle.write(("\n".join(header) + "\n").encode("ascii"))
        if output_format == "ascii":
            formatters = [
                (lambda value: f"{float(value):.9g}")
                if PLY_TYPES[type_name].startswith("f")
                else (lambda value: str(int(value)))
                for _, type_name in cloud.properties
            ]
            for start in range(0, len(vertices), 8192):
                lines: list[str] = []
                for vertex in vertices[start : start + 8192]:
                    lines.append(
                        " ".join(
                            formatter(vertex[name])
                            for (name, _), formatter in zip(
                                cloud.properties, formatters, strict=True
                            )
                        )
                    )
                handle.write(("\n".join(lines) + ("\n" if lines else "")).encode("ascii"))
        else:
            endian = "<" if output_format == "binary_little_endian" else ">"
            dtype = np.dtype(
                [(name, endian + PLY_TYPES[type_name]) for name, type_name in cloud.properties]
            )
            output = np.empty(len(vertices), dtype=dtype)
            for name, _ in cloud.properties:
                output[name] = vertices[name]
            output.tofile(handle)


def xyz_from_vertices(vertices: np.ndarray) -> np.ndarray:
    return np.column_stack([vertices[axis] for axis in ("x", "y", "z")]).astype(
        np.float64, copy=False
    )


def box_corners(minimum: np.ndarray, maximum: np.ndarray) -> list[list[float]]:
    return [
        [float(x), float(y), float(z)]
        for x, y, z in itertools.product(
            (minimum[0], maximum[0]),
            (minimum[1], maximum[1]),
            (minimum[2], maximum[2]),
        )
    ]


def point_bounds(points: np.ndarray) -> dict[str, Any]:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) / 2.0
    return {
        "minimum": minimum.tolist(),
        "maximum": maximum.tolist(),
        "center": center.tolist(),
        "extents": (maximum - minimum).tolist(),
        "corners": box_corners(minimum, maximum),
    }


def oriented_point_bounds(points: np.ndarray) -> dict[str, Any]:
    mean = points.mean(axis=0)
    centered = points - mean
    covariance = centered.T @ centered / max(1, len(points))
    eigenvalues, axes = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    axes = axes[:, order]
    for column in range(3):
        dominant = int(np.argmax(np.abs(axes[:, column])))
        if axes[dominant, column] < 0:
            axes[:, column] *= -1
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1

    local = centered @ axes
    local_minimum = local.min(axis=0)
    local_maximum = local.max(axis=0)
    local_center = (local_minimum + local_maximum) / 2.0
    world_center = mean + axes @ local_center
    local_corners = np.asarray(box_corners(local_minimum, local_maximum))
    world_corners = local_corners @ axes.T + mean
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = axes
    transform[:3, 3] = world_center
    return {
        "method": "pca",
        "center": world_center.tolist(),
        "extents": (local_maximum - local_minimum).tolist(),
        "axes_columns": axes.tolist(),
        "transform_local_to_world": transform.tolist(),
        "corners": world_corners.tolist(),
        "eigenvalues": eigenvalues.tolist(),
    }


def mask_bbox(mask: np.ndarray) -> list[int]:
    rows, columns = np.nonzero(mask)
    if not len(rows):
        raise ValueError("instance mask is empty")
    return [
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    ]


def load_masks(
    instance_manifest: dict[str, Any],
    *,
    manifest_path: Path,
    threshold: float,
    expected_shape: tuple[int, int],
    path_remaps: list[tuple[Path, Path]],
) -> list[InstanceMask]:
    raw_instances = instance_manifest.get("instances")
    if not isinstance(raw_instances, list) or len(raw_instances) < 2:
        raise ValueError("instance manifest must contain at least two instances")
    loaded: list[InstanceMask] = []
    seen_ids: set[str] = set()
    for raw in raw_instances:
        if not isinstance(raw, dict):
            raise ValueError("each instance manifest entry must be an object")
        object_id = str(raw.get("object_id", ""))
        if not OBJECT_ID_PATTERN.fullmatch(object_id) or object_id in seen_ids:
            raise ValueError(f"invalid or duplicate object_id: {object_id!r}")
        seen_ids.add(object_id)
        mask_path = resolve_input_path(
            str(raw["mask_path"]),
            relative_to=manifest_path.parent,
            path_remaps=path_remaps,
        )
        image = Image.open(mask_path).convert("L")
        values = np.asarray(image)
        if values.shape != expected_shape:
            raise ValueError(
                f"mask {object_id} shape {values.shape} does not match camera {expected_shape}"
            )
        scale = 1.0 if int(values.max()) <= 1 else 255.0
        mask = values.astype(np.float32) / scale >= threshold
        actual_bbox = mask_bbox(mask)
        prompt_box = raw.get("prompt_box_xyxy", actual_bbox)
        if not isinstance(prompt_box, list) or len(prompt_box) != 4:
            raise ValueError(f"{object_id} prompt_box_xyxy must have four values")
        prompt_box = [float(value) for value in prompt_box]
        if prompt_box[2] <= prompt_box[0] or prompt_box[3] <= prompt_box[1]:
            raise ValueError(f"{object_id} prompt box is empty or inverted")
        loaded.append(
            InstanceMask(
                object_id=object_id,
                metadata=raw,
                path=mask_path,
                sha256=sha256_file(mask_path),
                mask=mask,
                bbox_xyxy=actual_bbox,
                prompt_box_xyxy=prompt_box,
            )
        )
    return loaded


def infer_frame_id(instance_manifest: dict[str, Any], explicit: str | None) -> str:
    if explicit:
        return explicit
    source_image = instance_manifest.get("source_image")
    if not source_image:
        raise ValueError("--frame-id is required when source_image is absent")
    frame_id = Path(str(source_image)).stem
    if not frame_id:
        raise ValueError("could not infer frame id from source_image")
    return frame_id


def project_points(
    points: np.ndarray,
    world_to_camera: np.ndarray,
    intrinsic: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    camera = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    depth = camera[:, 2]
    valid_depth = depth > 1e-6
    u = np.full(len(points), -1, dtype=np.int64)
    v = np.full(len(points), -1, dtype=np.int64)
    u[valid_depth] = np.rint(
        float(intrinsic["fx"]) * camera[valid_depth, 0] / depth[valid_depth]
        + float(intrinsic["cx"])
    ).astype(np.int64)
    v[valid_depth] = np.rint(
        float(intrinsic["fy"]) * camera[valid_depth, 1] / depth[valid_depth]
        + float(intrinsic["cy"])
    ).astype(np.int64)
    inside = (
        valid_depth & (u >= 0) & (u < int(intrinsic["w"])) & (v >= 0) & (v < int(intrinsic["h"]))
    )
    return valid_depth, inside, u, v, depth


def resolve_labels(
    masks: list[InstanceMask],
    *,
    inside: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hits = np.zeros((len(masks), len(inside)), dtype=bool)
    inside_indices = np.flatnonzero(inside)
    for index, instance in enumerate(masks):
        hits[index, inside_indices] = instance.mask[v[inside_indices], u[inside_indices]]
    candidate_counts = hits.sum(axis=0)
    labels = np.full(len(inside), -1, dtype=np.int32)
    unique_indices = np.flatnonzero(candidate_counts == 1)
    labels[unique_indices] = np.argmax(hits[:, unique_indices], axis=0)

    overlap_indices = np.flatnonzero(candidate_counts > 1)
    if len(overlap_indices):
        scores = np.full((len(masks), len(overlap_indices)), np.inf, dtype=np.float64)
        overlap_u = u[overlap_indices].astype(np.float64)
        overlap_v = v[overlap_indices].astype(np.float64)
        for index, instance in enumerate(masks):
            x0, y0, x1, y1 = instance.prompt_box_xyxy
            center_x = (x0 + x1) / 2.0
            center_y = (y0 + y1) / 2.0
            half_width = max((x1 - x0) / 2.0, 1.0)
            half_height = max((y1 - y0) / 2.0, 1.0)
            eligible = hits[index, overlap_indices]
            score = ((overlap_u - center_x) / half_width) ** 2
            score += ((overlap_v - center_y) / half_height) ** 2
            score -= float(instance.metadata.get("mask_score", 0.0)) * 1e-9
            scores[index, eligible] = score[eligible]
        labels[overlap_indices] = np.argmin(scores, axis=0)
    return labels, candidate_counts, hits


def source_image_record(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    expected_shape: tuple[int, int],
    path_remaps: list[tuple[Path, Path]],
) -> dict[str, Any] | None:
    value = manifest.get("source_image")
    if not value:
        return None
    path = resolve_input_path(str(value), relative_to=manifest_path.parent, path_remaps=path_remaps)
    with Image.open(path) as image:
        dimensions = [image.width, image.height]
    if dimensions != [expected_shape[1], expected_shape[0]]:
        raise ValueError(
            f"source image dimensions {dimensions} do not match camera "
            f"{[expected_shape[1], expected_shape[0]]}"
        )
    return {
        "declared_path": str(value),
        "resolved_path": str(path),
        "sha256": sha256_file(path),
        "dimensions": dimensions,
    }


def split_cloud(args: argparse.Namespace) -> dict[str, Any]:
    cloud_path = args.cloud.expanduser().resolve()
    camera_info_path = args.camera_info.expanduser().resolve()
    instance_manifest_path = args.instances.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    path_remaps = parse_path_remaps(args.path_remap)

    cloud = read_ply_cloud(cloud_path)
    camera_info = read_json(camera_info_path)
    instance_manifest = read_json(instance_manifest_path)
    frame_id = infer_frame_id(instance_manifest, args.frame_id)
    extrinsic_type = camera_info.get("extrinsic_type")
    if extrinsic_type != args.require_extrinsic_type:
        raise ValueError(
            f"camera_info extrinsic_type={extrinsic_type!r}; expected "
            f"{args.require_extrinsic_type!r}"
        )
    intrinsic = camera_info.get("intrinsics", {}).get(frame_id, camera_info.get("intrinsic"))
    if not isinstance(intrinsic, dict):
        raise ValueError(f"camera intrinsics unavailable for frame {frame_id}")
    expected_shape = (int(intrinsic["h"]), int(intrinsic["w"]))
    extrinsic = camera_info.get("extrinsic", {}).get(frame_id)
    if extrinsic is None:
        raise ValueError(f"camera extrinsic unavailable for frame {frame_id}")
    world_to_camera = np.asarray(extrinsic, dtype=np.float64)
    if world_to_camera.shape != (4, 4) or not np.isfinite(world_to_camera).all():
        raise ValueError(f"frame {frame_id} extrinsic must be a finite 4x4 matrix")

    masks = load_masks(
        instance_manifest,
        manifest_path=instance_manifest_path,
        threshold=args.mask_threshold,
        expected_shape=expected_shape,
        path_remaps=path_remaps,
    )
    image_record = source_image_record(
        instance_manifest,
        manifest_path=instance_manifest_path,
        expected_shape=expected_shape,
        path_remaps=path_remaps,
    )
    points = xyz_from_vertices(cloud.vertices)
    if not np.isfinite(points).all():
        raise ValueError("input cloud contains non-finite XYZ coordinates")
    valid_depth, inside, u, v, depth = project_points(points, world_to_camera, intrinsic)
    labels, candidate_counts, _hits = resolve_labels(masks, inside=inside, u=u, v=v)

    assigned = labels >= 0
    overlap = candidate_counts > 1
    unassigned = ~assigned
    output_instances: list[dict[str, Any]] = []
    failure_reasons: list[str] = []
    for index, instance in enumerate(masks):
        selection = labels == index
        instance_vertices = cloud.vertices[selection]
        instance_points = points[selection]
        output_path = output_dir / "objects" / instance.object_id / f"{instance.object_id}.ply"
        write_ply_cloud(
            output_path,
            cloud,
            instance_vertices,
            output_format=args.output_format,
        )
        minimum_points_passed = len(instance_vertices) >= args.min_points_per_instance
        if not minimum_points_passed:
            failure_reasons.append(
                f"{instance.object_id}: {len(instance_vertices)} points < "
                f"{args.min_points_per_instance}"
            )
        projected_u = u[selection]
        projected_v = v[selection]
        projected_depth = depth[selection]
        geometry = None
        if len(instance_points) >= 3:
            geometry = {
                "centroid": instance_points.mean(axis=0).tolist(),
                "aabb": point_bounds(instance_points),
                "obb": oriented_point_bounds(instance_points),
            }
        chosen_from_overlap = int(np.count_nonzero(selection & overlap))
        output_instances.append(
            {
                "object_id": instance.object_id,
                "name": instance.metadata.get("name"),
                "category": instance.metadata.get("category"),
                "description": instance.metadata.get("description"),
                "mask": {
                    "declared_path": str(instance.metadata["mask_path"]),
                    "resolved_path": str(instance.path),
                    "sha256": instance.sha256,
                    "score": instance.metadata.get("mask_score"),
                    "area_pixels": int(instance.mask.sum()),
                    "bbox_xyxy": instance.bbox_xyxy,
                    "prompt_box_xyxy": instance.prompt_box_xyxy,
                },
                "point_count": len(instance_vertices),
                "fraction_of_input": float(len(instance_vertices) / max(1, len(points))),
                "chosen_from_overlap_count": chosen_from_overlap,
                "projection": {
                    "pixel_bbox_xyxy": (
                        [
                            int(projected_u.min()),
                            int(projected_v.min()),
                            int(projected_u.max()) + 1,
                            int(projected_v.max()) + 1,
                        ]
                        if len(projected_u)
                        else None
                    ),
                    "depth_min_m": float(projected_depth.min()) if len(projected_depth) else None,
                    "depth_median_m": (
                        float(np.median(projected_depth)) if len(projected_depth) else None
                    ),
                    "depth_max_m": float(projected_depth.max()) if len(projected_depth) else None,
                },
                "geometry": geometry,
                "output": {
                    "path": str(output_path),
                    "relative_path": str(output_path.relative_to(output_dir)),
                    "sha256": sha256_file(output_path),
                    "bytes": output_path.stat().st_size,
                    "ply_format": (
                        cloud.source_format if args.output_format == "same" else args.output_format
                    ),
                    "preserved_vertex_properties": [name for name, _ in cloud.properties],
                },
                "gates": {"minimum_points": minimum_points_passed},
            }
        )

    unassigned_path = output_dir / "unassigned.ply"
    write_ply_cloud(
        unassigned_path,
        cloud,
        cloud.vertices[unassigned],
        output_format=args.output_format,
    )

    assigned_ratio = float(assigned.mean()) if len(assigned) else 0.0
    image_inside_ratio = float(inside.mean()) if len(inside) else 0.0
    overlap_ratio = float(overlap.sum() / max(1, assigned.sum()))
    if assigned_ratio < args.min_assigned_ratio:
        failure_reasons.append(
            f"assigned_ratio {assigned_ratio:.6f} < {args.min_assigned_ratio:.6f}"
        )
    if image_inside_ratio < args.min_image_inside_ratio:
        failure_reasons.append(
            f"image_inside_ratio {image_inside_ratio:.6f} < {args.min_image_inside_ratio:.6f}"
        )
    if overlap_ratio > args.max_overlap_ratio:
        failure_reasons.append(f"overlap_ratio {overlap_ratio:.6f} > {args.max_overlap_ratio:.6f}")
    conservation_count = sum(item["point_count"] for item in output_instances) + int(
        unassigned.sum()
    )
    conservation_passed = conservation_count == len(points)
    if not conservation_passed:
        failure_reasons.append(
            f"point conservation failed: outputs={conservation_count}, input={len(points)}"
        )

    status = "passed" if not failure_reasons else "failed"
    common = {
        "schema_version": 1,
        "status": status,
        "run_id": args.run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "execution": {
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "argv": sys.argv,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": PIL.__version__,
        },
        "projection_contract": {
            "extrinsic_type": args.require_extrinsic_type,
            "world_to_camera_formula": "camera_xyz = world_xyz @ R.T + t",
            "pixel_formula": "round(focal * camera_xy / camera_z + principal_point)",
            "origin": "Video2Mesh accepted Bedroom4 projection verifier",
        },
        "source": {
            "merged_cloud": {
                "path": str(cloud_path),
                "declared_source_uri": args.cloud_source_uri,
                "sha256": sha256_file(cloud_path),
                "bytes": cloud_path.stat().st_size,
                "point_count": len(points),
                "ply_format": cloud.source_format,
                "vertex_properties": [
                    {"name": name, "type": type_name} for name, type_name in cloud.properties
                ],
            },
            "camera_info": {
                "path": str(camera_info_path),
                "declared_source_uri": args.camera_info_source_uri,
                "sha256": sha256_file(camera_info_path),
                "extrinsic_type": extrinsic_type,
            },
            "instance_manifest": {
                "path": str(instance_manifest_path),
                "sha256": sha256_file(instance_manifest_path),
            },
            "source_image": image_record,
            "frame_id": frame_id,
        },
        "parameters": {
            "mask_threshold": args.mask_threshold,
            "overlap_resolution": (
                "minimum normalized prompt-box-center distance; mask score tie-break"
            ),
            "min_assigned_ratio": args.min_assigned_ratio,
            "min_image_inside_ratio": args.min_image_inside_ratio,
            "min_points_per_instance": args.min_points_per_instance,
            "max_overlap_ratio": args.max_overlap_ratio,
            "output_format": args.output_format,
            "path_remaps": [
                {"declared_prefix": str(source), "resolved_prefix": str(target)}
                for source, target in path_remaps
            ],
        },
        "gates": {
            "passed": not failure_reasons,
            "failure_reasons": failure_reasons,
            "assigned_ratio": {
                "passed": assigned_ratio >= args.min_assigned_ratio,
                "value": assigned_ratio,
                "threshold_minimum": args.min_assigned_ratio,
            },
            "image_inside_ratio": {
                "passed": image_inside_ratio >= args.min_image_inside_ratio,
                "value": image_inside_ratio,
                "threshold_minimum": args.min_image_inside_ratio,
            },
            "overlap_ratio": {
                "passed": overlap_ratio <= args.max_overlap_ratio,
                "value": overlap_ratio,
                "threshold_maximum": args.max_overlap_ratio,
            },
            "point_conservation": {
                "passed": conservation_passed,
                "input_count": len(points),
                "output_plus_unassigned_count": conservation_count,
            },
        },
    }
    report = {
        **common,
        "counts": {
            "input_points": len(points),
            "positive_depth_points": int(valid_depth.sum()),
            "inside_image_points": int(inside.sum()),
            "outside_image_points": int((valid_depth & ~inside).sum()),
            "behind_or_on_camera_points": int((~valid_depth).sum()),
            "mask_hit_points": int((candidate_counts > 0).sum()),
            "assigned_points": int(assigned.sum()),
            "overlap_points": int(overlap.sum()),
            "unassigned_points": int(unassigned.sum()),
            "unassigned_inside_without_mask": int((inside & (candidate_counts == 0)).sum()),
        },
        "instances": output_instances,
        "unassigned": {
            "path": str(unassigned_path),
            "relative_path": str(unassigned_path.relative_to(output_dir)),
            "sha256": sha256_file(unassigned_path),
            "bytes": unassigned_path.stat().st_size,
            "point_count": int(unassigned.sum()),
        },
    }
    report_path = output_dir / "instance_cloud_split_report.json"
    manifest_path = output_dir / "instance_cloud_manifest.json"
    write_json(report_path, report)
    manifest = {
        **common,
        "report": {
            "path": str(report_path),
            "relative_path": str(report_path.relative_to(output_dir)),
            "sha256": sha256_file(report_path),
        },
        "objects": [
            {
                "object_id": item["object_id"],
                "category": item["category"],
                "point_count": item["point_count"],
                "geometry": item["geometry"],
                "asset": item["output"],
                "source_mask": item["mask"],
            }
            for item in output_instances
        ],
        "unassigned": report["unassigned"],
    }
    write_json(manifest_path, manifest)
    return {
        "status": status,
        "report_path": str(report_path),
        "manifest_path": str(manifest_path),
        "failure_reasons": failure_reasons,
        "counts": report["counts"],
        "objects": [
            {"object_id": item["object_id"], "point_count": item["point_count"]}
            for item in output_instances
        ],
    }


def run_self_test(output_root: Path | None) -> dict[str, Any]:
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if output_root is None:
        temporary = tempfile.TemporaryDirectory(prefix="video2world_split_self_test_")
        root = Path(temporary.name)
    else:
        root = output_root.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)

    try:
        dtype = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1")])
        vertices = np.empty(201, dtype=dtype)
        left_x = np.linspace(-0.40, 0.10, 100)
        right_x = np.linspace(-0.10, 0.40, 100)
        vertices["x"] = np.concatenate([left_x, right_x, [0.90]])
        vertices["y"] = [*np.tile(np.linspace(-0.20, 0.20, 10), 20), 0.0]
        vertices["z"] = 2.0
        vertices["red"] = np.arange(201, dtype=np.uint8)
        vertices["green"] = 123
        cloud = PlyCloud(
            source_format="ascii",
            properties=[
                ("x", "float"),
                ("y", "float"),
                ("z", "float"),
                ("red", "uchar"),
                ("green", "uchar"),
            ],
            vertices=vertices,
        )
        cloud_path = root / "merged.ply"
        write_ply_cloud(cloud_path, cloud, vertices, output_format="ascii")
        camera_path = root / "camera_info.json"
        write_json(
            camera_path,
            {
                "extrinsic_type": "world_to_camera",
                "intrinsic": {"w": 100, "h": 100, "fx": 100, "fy": 100, "cx": 50, "cy": 50},
                "extrinsic": {"000001": np.eye(4).tolist()},
            },
        )
        Image.new("RGB", (100, 100), (64, 64, 64)).save(root / "000001.png")
        left_mask = np.zeros((100, 100), dtype=np.uint8)
        right_mask = np.zeros((100, 100), dtype=np.uint8)
        left_mask[35:66, 25:57] = 255
        right_mask[35:66, 43:76] = 255
        Image.fromarray(left_mask).save(root / "left.png")
        Image.fromarray(right_mask).save(root / "right.png")
        instances_path = root / "instances.json"
        write_json(
            instances_path,
            {
                "source_image": str(root / "000001.png"),
                "instances": [
                    {
                        "object_id": "left",
                        "category": "synthetic",
                        "mask_path": str(root / "left.png"),
                        "prompt_box_xyxy": [25, 35, 57, 66],
                        "mask_score": 0.9,
                    },
                    {
                        "object_id": "right",
                        "category": "synthetic",
                        "mask_path": str(root / "right.png"),
                        "prompt_box_xyxy": [43, 35, 76, 66],
                        "mask_score": 0.9,
                    },
                ],
            },
        )
        args = argparse.Namespace(
            cloud=cloud_path,
            camera_info=camera_path,
            instances=instances_path,
            output_dir=root / "output",
            frame_id="000001",
            run_id="self_test",
            mask_threshold=0.5,
            min_assigned_ratio=0.98,
            min_image_inside_ratio=1.0,
            min_points_per_instance=50,
            max_overlap_ratio=0.5,
            output_format="binary_little_endian",
            require_extrinsic_type="world_to_camera",
            path_remap=[],
            cloud_source_uri=None,
            camera_info_source_uri=None,
        )
        result = split_cloud(args)
        assert result["status"] == "passed", result
        assert result["counts"]["input_points"] == 201
        assert result["counts"]["assigned_points"] == 200
        assert result["counts"]["unassigned_points"] == 1
        assert result["counts"]["overlap_points"] > 0
        assert sum(item["point_count"] for item in result["objects"]) == 200
        for item in result["objects"]:
            assert item["point_count"] >= 50
        return {"status": "passed", "root": str(root), "split": result}
    finally:
        if temporary is not None:
            temporary.cleanup()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloud", type=Path)
    parser.add_argument("--camera-info", type=Path)
    parser.add_argument("--instances", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--frame-id")
    parser.add_argument("--run-id", default="instance_cloud_split")
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--min-assigned-ratio", type=float, default=0.90)
    parser.add_argument("--min-image-inside-ratio", type=float, default=0.90)
    parser.add_argument("--min-points-per-instance", type=int, default=1000)
    parser.add_argument("--max-overlap-ratio", type=float, default=0.20)
    parser.add_argument(
        "--output-format",
        choices=("same", "ascii", "binary_little_endian", "binary_big_endian"),
        default="binary_little_endian",
    )
    parser.add_argument("--require-extrinsic-type", default="world_to_camera")
    parser.add_argument(
        "--path-remap",
        action="append",
        default=[],
        metavar="ABSOLUTE_SOURCE=LOCAL_TARGET",
        help="Remap declared absolute image/mask paths to a local mirror; repeatable.",
    )
    parser.add_argument("--cloud-source-uri")
    parser.add_argument("--camera-info-source-uri")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--self-test-output", type=Path)
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.self_test:
        return
    missing = [
        name
        for name in ("cloud", "camera_info", "instances", "output_dir")
        if getattr(args, name) is None
    ]
    if missing:
        flags = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        parser.error("required arguments missing: " + flags)
    if not 0.0 < args.mask_threshold <= 1.0:
        parser.error("--mask-threshold must be in (0, 1]")
    for name in ("min_assigned_ratio", "min_image_inside_ratio", "max_overlap_ratio"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    if args.min_points_per_instance < 3:
        parser.error("--min-points-per-instance must be at least 3")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    result = run_self_test(args.self_test_output) if args.self_test else split_cloud(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
