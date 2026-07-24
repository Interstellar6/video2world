#!/usr/bin/env python3
"""Create a candidate-only static Gaussian carve from calibrated source evidence.

The filter removes a static Gaussian only when its center is close to an original
object anchor and the same anchor agrees with the Gaussian in multiple source
views: both must project into the original modal mask, remain close in image
space, and have consistent camera depth. It does not infer hidden content or
modify any canonical scene asset.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from plyfile import PlyData
from scipy.spatial import cKDTree

from video2world.hashing import atomic_write_json, sha256_file

KIND = "video2world.multiview_depth_static_gaussian_carve"
STATIC_REQUIRED_PROPERTIES = frozenset(
    {
        "x",
        "y",
        "z",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    }
)


@dataclass(frozen=True)
class ViewInput:
    frame_id: str
    mask_path: Path
    mask_sha256: str


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _digest(path: Path, *, expected: str, label: str) -> tuple[str, int]:
    require(isinstance(expected, str) and len(expected) == 64, f"{label} needs a SHA-256")
    observed, size = sha256_file(path)
    require(observed == expected, f"{label} SHA-256 mismatch: {observed} != {expected}")
    return observed, size


def _read_json(path: Path, *, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {label}: {path}: {exc}") from exc


def _ply_header(path: Path) -> tuple[bytes, int]:
    with path.open("rb") as handle:
        prefix = handle.read(1024 * 1024)
    marker = b"end_header"
    marker_index = prefix.find(marker)
    require(marker_index >= 0, f"static PLY has no end_header: {path}")
    offset = marker_index + len(marker)
    if prefix[offset : offset + 2] == b"\r\n":
        offset += 2
    elif prefix[offset : offset + 1] in {b"\r", b"\n"}:
        offset += 1
    else:
        raise RuntimeError(f"static PLY end_header has no line terminator: {path}")
    require(offset < len(prefix), f"static PLY has no binary payload: {path}")
    return prefix[:offset], offset


def _load_static_vertices(path: Path) -> tuple[np.ndarray, bytes, int]:
    try:
        ply = PlyData.read(path, mmap=True)
    except Exception as exc:  # pragma: no cover - parser messages vary by plyfile release
        raise RuntimeError(f"cannot inspect static PLY {path}: {exc}") from exc
    require(not ply.text and ply.byte_order == "<", "static PLY must be binary little-endian")
    require(
        len(ply.elements) == 1 and ply.elements[0].name == "vertex",
        "static PLY must be vertex-only",
    )
    vertices = ply["vertex"].data
    names = set(vertices.dtype.names or ())
    missing = sorted(STATIC_REQUIRED_PROPERTIES.difference(names))
    require(not missing, f"static PLY is missing Gaussian fields: {missing}")
    require(len(vertices) > 0, "static PLY is empty")
    require(vertices.dtype.itemsize > 0, "static PLY has an invalid vertex stride")
    positions = np.column_stack([vertices[axis] for axis in ("x", "y", "z")])
    require(np.isfinite(positions).all(), "static PLY has non-finite centers")
    header, offset = _ply_header(path)
    expected_size = offset + len(vertices) * vertices.dtype.itemsize
    require(path.stat().st_size == expected_size, "static PLY payload size is inconsistent")
    return vertices, header, offset


def _load_anchor_points(path: Path) -> np.ndarray:
    try:
        ply = PlyData.read(path, mmap=True)
    except Exception as exc:  # pragma: no cover - parser messages vary by plyfile release
        raise RuntimeError(f"cannot inspect anchor PLY {path}: {exc}") from exc
    require("vertex" in ply, "anchor PLY has no vertex element")
    vertices = ply["vertex"].data
    names = set(vertices.dtype.names or ())
    require({"x", "y", "z"}.issubset(names), "anchor PLY is missing x/y/z")
    points = np.column_stack([vertices[axis] for axis in ("x", "y", "z")]).astype(np.float32)
    require(len(points) > 0 and np.isfinite(points).all(), "anchor PLY is invalid")
    return points


def _camera_index(path: Path) -> dict[str, dict[str, Any]]:
    raw = _read_json(path, label="camera JSON")
    records = raw.get("cameras") if isinstance(raw, dict) else raw
    require(isinstance(records, list) and records, "camera JSON must contain a non-empty list")
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        require(isinstance(record, dict), "camera record must be an object")
        frame_id = record.get("img_name", record.get("frame_id"))
        require(isinstance(frame_id, str) and frame_id, "camera record has no frame identifier")
        require(frame_id not in indexed, f"duplicate camera frame: {frame_id}")
        position = np.asarray(record.get("position"), dtype=np.float64)
        rotation = np.asarray(record.get("rotation"), dtype=np.float64)
        require(
            position.shape == (3,) and np.isfinite(position).all(),
            f"invalid camera position: {frame_id}",
        )
        require(
            rotation.shape == (3, 3) and np.isfinite(rotation).all(),
            f"invalid camera rotation: {frame_id}",
        )
        require(
            np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5),
            f"camera rotation is not orthonormal: {frame_id}",
        )
        numeric = ("fx", "fy", "cx", "cy", "width", "height")
        require(
            all(isinstance(record.get(key), int | float) for key in numeric),
            f"camera has incomplete intrinsics: {frame_id}",
        )
        require(
            float(record["fx"]) > 0 and float(record["fy"]) > 0, f"invalid focal length: {frame_id}"
        )
        require(
            int(record["width"]) > 0 and int(record["height"]) > 0,
            f"invalid image size: {frame_id}",
        )
        indexed[frame_id] = {
            "position": position.astype(np.float32),
            "rotation": rotation.astype(np.float32),
            "fx": float(record["fx"]),
            "fy": float(record["fy"]),
            "cx": float(record["cx"]),
            "cy": float(record["cy"]),
            "width": int(record["width"]),
            "height": int(record["height"]),
        }
    return indexed


def _project(
    points: np.ndarray, camera: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    camera_points = (points - camera["position"]) @ camera["rotation"]
    depth = camera_points[:, 2]
    visible = depth > 1e-6
    safe_depth = np.where(visible, depth, 1.0)
    x = camera["fx"] * camera_points[:, 0] / safe_depth + camera["cx"]
    y = camera["fy"] * camera_points[:, 1] / safe_depth + camera["cy"]
    xi = np.rint(x).astype(np.int64)
    yi = np.rint(y).astype(np.int64)
    inside = visible & (xi >= 0) & (xi < camera["width"]) & (yi >= 0) & (yi < camera["height"])
    return xi, yi, depth.astype(np.float32), inside


def _mask(path: Path, *, camera: dict[str, Any], frame_id: str) -> np.ndarray:
    with Image.open(path) as image:
        value = np.asarray(image.convert("L"), dtype=np.uint8) >= 128
    require(value.shape == (camera["height"], camera["width"]), f"mask shape mismatch: {frame_id}")
    require(bool(value.any()), f"mask is empty: {frame_id}")
    return value


def _query_nearest(anchors: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(anchors)
    try:
        distances, indices = tree.query(points, k=1, workers=-1)
    except TypeError:  # pragma: no cover - old SciPy compatibility
        distances, indices = tree.query(points, k=1)
    return np.asarray(distances, dtype=np.float32), np.asarray(indices, dtype=np.int64)


def _view_vote(
    *,
    points: np.ndarray,
    anchors: np.ndarray,
    camera: dict[str, Any],
    mask: np.ndarray,
    max_pixel_delta: int,
    depth_tolerance: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    xi, yi, depth, inside = _project(points, camera)
    anchor_xi, anchor_yi, anchor_depth, anchor_inside = _project(anchors, camera)
    point_in_mask = np.zeros(len(points), dtype=bool)
    anchor_in_mask = np.zeros(len(points), dtype=bool)
    point_in_mask[inside] = mask[yi[inside], xi[inside]]
    anchor_in_mask[anchor_inside] = mask[anchor_yi[anchor_inside], anchor_xi[anchor_inside]]
    image_delta = np.maximum(np.abs(xi - anchor_xi), np.abs(yi - anchor_yi))
    depth_delta = np.abs(depth - anchor_depth)
    matched = (
        inside & anchor_inside & point_in_mask & anchor_in_mask & (image_delta <= max_pixel_delta)
    )
    accepted = matched & (depth_delta <= depth_tolerance)
    detail = {
        "candidate_centers": len(points),
        "projected_centers": int(inside.sum()),
        "projected_anchor_centers": int(anchor_inside.sum()),
        "point_mask_support": int(point_in_mask.sum()),
        "paired_modal_mask_support": int(matched.sum()),
        "depth_consistent_support": int(accepted.sum()),
        "paired_depth_delta": _stats(depth_delta[matched]),
    }
    return accepted, detail


def _stats(values: np.ndarray) -> dict[str, float | None]:
    if len(values) == 0:
        return {"minimum": None, "median": None, "p95": None, "maximum": None}
    return {
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(np.max(values)),
    }


def _write_filtered_ply(
    *,
    source: Path,
    destination: Path,
    header: bytes,
    data_offset: int,
    vertices: np.ndarray,
    keep: np.ndarray,
    batch_size: int,
) -> None:
    require(len(vertices) == len(keep), "static vertex and keep-mask lengths differ")
    require(batch_size > 0, "batch size must be positive")
    retained = int(keep.sum())
    input_count = len(vertices)
    old_declaration = f"element vertex {input_count}".encode("ascii")
    new_declaration = f"element vertex {retained}".encode("ascii")
    output_header = header.replace(old_declaration, new_declaration, 1)
    require(output_header != header, "could not update PLY vertex count")
    require(destination.resolve() != source.resolve(), "output PLY must not overwrite the source")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(output_header)
            for start in range(0, input_count, batch_size):
                stop = min(input_count, start + batch_size)
                selected = vertices[start:stop][keep[start:stop]]
                if len(selected):
                    handle.write(selected.tobytes())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    expected_size = len(output_header) + retained * vertices.dtype.itemsize
    require(destination.stat().st_size == expected_size, "filtered PLY byte count is invalid")


def carve_static_scene(
    *,
    static_ply: Path,
    static_ply_sha256: str,
    anchor_ply: Path,
    anchor_ply_sha256: str,
    camera_json: Path,
    camera_json_sha256: str,
    views: Sequence[ViewInput],
    output_ply: Path,
    receipt_path: Path,
    anchor_distance: float = 0.08,
    depth_tolerance: float = 0.08,
    max_pixel_delta: int = 3,
    min_view_votes: int = 2,
    batch_size: int = 65_536,
) -> dict[str, Any]:
    static_ply = static_ply.expanduser().resolve()
    anchor_ply = anchor_ply.expanduser().resolve()
    camera_json = camera_json.expanduser().resolve()
    output_ply = output_ply.expanduser().resolve()
    receipt_path = receipt_path.expanduser().resolve()
    require(static_ply.is_file(), f"static PLY does not exist: {static_ply}")
    require(anchor_ply.is_file(), f"anchor PLY does not exist: {anchor_ply}")
    require(camera_json.is_file(), f"camera JSON does not exist: {camera_json}")
    require(len(views) >= 2, "at least two source views are required")
    require(0 < anchor_distance <= 1.0, "anchor distance must be in (0, 1]")
    require(0 < depth_tolerance <= 1.0, "depth tolerance must be in (0, 1]")
    require(max_pixel_delta >= 0, "max pixel delta must be non-negative")
    require(1 <= min_view_votes <= len(views), "min view votes must be within the supplied views")
    require(output_ply != receipt_path, "output PLY and receipt must differ")
    protected = {static_ply, anchor_ply, camera_json}
    require(
        output_ply not in protected and receipt_path not in protected, "output overwrites an input"
    )
    require(not output_ply.exists(), f"refusing to overwrite output PLY: {output_ply}")
    require(not receipt_path.exists(), f"refusing to overwrite receipt: {receipt_path}")

    static_sha, static_size = _digest(static_ply, expected=static_ply_sha256, label="static PLY")
    anchor_sha, anchor_size = _digest(anchor_ply, expected=anchor_ply_sha256, label="anchor PLY")
    camera_sha, camera_size = _digest(camera_json, expected=camera_json_sha256, label="camera JSON")
    cameras = _camera_index(camera_json)
    seen_frames: set[str] = set()
    verified_views: list[tuple[ViewInput, dict[str, Any], np.ndarray, str, int]] = []
    for view in views:
        frame_id = str(view.frame_id)
        mask_path = view.mask_path.expanduser().resolve()
        require(
            frame_id and frame_id not in seen_frames, f"duplicate or empty view frame: {frame_id}"
        )
        require(mask_path.is_file(), f"view mask does not exist: {mask_path}")
        require(frame_id in cameras, f"view frame is absent from camera JSON: {frame_id}")
        mask_sha, mask_size = _digest(
            mask_path, expected=view.mask_sha256, label=f"mask {frame_id}"
        )
        verified_views.append(
            (
                view,
                cameras[frame_id],
                _mask(mask_path, camera=cameras[frame_id], frame_id=frame_id),
                mask_sha,
                mask_size,
            )
        )
        seen_frames.add(frame_id)

    vertices, header, data_offset = _load_static_vertices(static_ply)
    points = np.column_stack([vertices[axis] for axis in ("x", "y", "z")]).astype(np.float32)
    anchor_points = _load_anchor_points(anchor_ply)
    distances, nearest_indices = _query_nearest(anchor_points, points)
    nearby = distances <= anchor_distance
    candidate_indices = np.flatnonzero(nearby)
    require(len(candidate_indices) > 0, "no static Gaussian is close enough to an object anchor")
    candidate_points = points[candidate_indices]
    candidate_anchors = anchor_points[nearest_indices[candidate_indices]]
    votes = np.zeros(len(candidate_indices), dtype=np.uint8)
    view_records: list[dict[str, Any]] = []
    for view, camera, mask, mask_sha, mask_size in verified_views:
        accepted, detail = _view_vote(
            points=candidate_points,
            anchors=candidate_anchors,
            camera=camera,
            mask=mask,
            max_pixel_delta=max_pixel_delta,
            depth_tolerance=depth_tolerance,
        )
        votes += accepted.astype(np.uint8)
        view_records.append(
            {
                "frame_id": view.frame_id,
                "mask": {
                    "path": str(view.mask_path.expanduser().resolve()),
                    "sha256": mask_sha,
                    "bytes": mask_size,
                    "pixels": int(mask.sum()),
                },
                "camera": {
                    "frame_id": view.frame_id,
                    "width": camera["width"],
                    "height": camera["height"],
                },
                **detail,
            }
        )
    remove_local = votes >= min_view_votes
    remove = np.zeros(len(vertices), dtype=bool)
    remove[candidate_indices[remove_local]] = True
    removed_count = int(remove.sum())
    require(removed_count > 0, "multiview gate removed no static Gaussians")
    keep = ~remove
    _write_filtered_ply(
        source=static_ply,
        destination=output_ply,
        header=header,
        data_offset=data_offset,
        vertices=vertices,
        keep=keep,
        batch_size=batch_size,
    )
    output_vertices, _output_header, _output_offset = _load_static_vertices(output_ply)
    require(
        len(output_vertices) == int(keep.sum()), "filtered PLY vertex count does not match receipt"
    )
    output_sha, output_size = sha256_file(output_ply)

    # Rehash the protected source evidence after writing the candidate.
    _digest(static_ply, expected=static_sha, label="static PLY after carve")
    _digest(anchor_ply, expected=anchor_sha, label="anchor PLY after carve")
    _digest(camera_json, expected=camera_sha, label="camera JSON after carve")
    for view, _camera, _mask_value, mask_sha, _mask_size in verified_views:
        _digest(
            view.mask_path.expanduser().resolve(),
            expected=mask_sha,
            label=f"mask {view.frame_id} after carve",
        )

    vote_distribution = {
        str(vote): int(np.count_nonzero(votes == vote)) for vote in range(len(views) + 1)
    }
    receipt = {
        "schema_version": 1,
        "kind": KIND,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "candidate_materialized",
        "promotion_allowed": False,
        "method": "nearest_anchor_multiview_modal_mask_and_depth_consistency_center_carve",
        "parameters": {
            "anchor_distance": float(anchor_distance),
            "depth_tolerance": float(depth_tolerance),
            "max_pixel_delta": int(max_pixel_delta),
            "min_view_votes": int(min_view_votes),
            "center_only": True,
        },
        "input_static_scene": {
            "path": str(static_ply),
            "sha256": static_sha,
            "bytes": static_size,
            "vertex_count": len(vertices),
            "role": "original_uncarved_pgsr_full_scene",
        },
        "object_anchor": {
            "path": str(anchor_ply),
            "sha256": anchor_sha,
            "bytes": anchor_size,
            "point_count": len(anchor_points),
        },
        "camera_json": {
            "path": str(camera_json),
            "sha256": camera_sha,
            "bytes": camera_size,
        },
        "views": view_records,
        "analysis": {
            "near_anchor_static_gaussian_count": len(candidate_indices),
            "nearest_anchor_distance": _stats(distances[candidate_indices]),
            "vote_distribution": vote_distribution,
            "strict_all_view_remove_count": int(np.count_nonzero(votes == len(views))),
            "selected_remove_count": removed_count,
            "retained_count": int(keep.sum()),
        },
        "output_static_scene": {
            "path": str(output_ply),
            "sha256": output_sha,
            "bytes": output_size,
            "vertex_count": int(keep.sum()),
            "role": "candidate_multiview_depth_carved_pgsr_scene",
        },
        "limitations": [
            "Candidate only; the canonical PGSR scene and collider are unchanged.",
            "No clean plate or hidden geometry is generated by this filter.",
            "A removed center is not proof that its full Gaussian footprint belongs to the object.",
            "A browser or rendered-view review is required before any promotion.",
        ],
    }
    atomic_write_json(receipt_path, receipt)
    return receipt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-ply", type=Path, required=True)
    parser.add_argument("--static-ply-sha256", required=True)
    parser.add_argument("--anchor-ply", type=Path, required=True)
    parser.add_argument("--anchor-ply-sha256", required=True)
    parser.add_argument("--camera-json", type=Path, required=True)
    parser.add_argument("--camera-json-sha256", required=True)
    parser.add_argument(
        "--view",
        nargs=3,
        action="append",
        metavar=("FRAME_ID", "MASK_PATH", "MASK_SHA256"),
        required=True,
    )
    parser.add_argument("--output-ply", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--anchor-distance", type=float, default=0.08)
    parser.add_argument("--depth-tolerance", type=float, default=0.08)
    parser.add_argument("--max-pixel-delta", type=int, default=3)
    parser.add_argument("--min-view-votes", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=65_536)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    receipt = carve_static_scene(
        static_ply=args.static_ply,
        static_ply_sha256=args.static_ply_sha256,
        anchor_ply=args.anchor_ply,
        anchor_ply_sha256=args.anchor_ply_sha256,
        camera_json=args.camera_json,
        camera_json_sha256=args.camera_json_sha256,
        views=[
            ViewInput(frame_id=item[0], mask_path=Path(item[1]), mask_sha256=item[2])
            for item in args.view
        ],
        output_ply=args.output_ply,
        receipt_path=args.receipt,
        anchor_distance=args.anchor_distance,
        depth_tolerance=args.depth_tolerance,
        max_pixel_delta=args.max_pixel_delta,
        min_view_votes=args.min_view_votes,
        batch_size=args.batch_size,
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "removed": receipt["analysis"]["selected_remove_count"],
                "retained": receipt["analysis"]["retained_count"],
                "output": receipt["output_static_scene"]["path"],
                "receipt": str(args.receipt.expanduser().resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
