#!/usr/bin/env python3
"""Audit pairwise scene-fit separation and calibrated depth ordering."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image


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


def resolve_path(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def load_camera(path: Path, frame_id: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    matches = [item for item in payload if str(item.get("img_name")) == frame_id]
    if len(matches) != 1:
        raise ValueError(f"expected one camera for {frame_id!r}, found {len(matches)}")
    raw = matches[0]
    rotation = np.asarray(raw.get("rotation"), dtype=np.float64)
    position = np.asarray(raw.get("position"), dtype=np.float64)
    if rotation.shape != (3, 3) or position.shape != (3,):
        raise ValueError("camera extrinsics have invalid dimensions")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("camera rotation is not orthonormal")
    return {
        "width": int(raw["width"]),
        "height": int(raw["height"]),
        "fx": float(raw["fx"]),
        "fy": float(raw["fy"]),
        "rotation": rotation,
        "position": position,
    }


def depth_bins(
    world_vertices: np.ndarray,
    camera: dict[str, Any],
    bin_size_px: int,
) -> np.ndarray:
    if bin_size_px < 1:
        raise ValueError("bin_size_px must be positive")
    camera_vertices = np.einsum(
        "ni,ij->nj",
        world_vertices - camera["position"],
        camera["rotation"],
    )
    depth = camera_vertices[:, 2]
    in_front = depth > 1e-6
    u = np.zeros(len(depth), dtype=np.int64)
    v = np.zeros(len(depth), dtype=np.int64)
    u[in_front] = np.rint(
        camera["fx"] * camera_vertices[in_front, 0] / depth[in_front] + camera["width"] / 2
    ).astype(np.int64)
    v[in_front] = np.rint(
        camera["height"] / 2 + camera["fy"] * camera_vertices[in_front, 1] / depth[in_front]
    ).astype(np.int64)
    valid = in_front & (u >= 0) & (u < camera["width"]) & (v >= 0) & (v < camera["height"])
    width_bins = (camera["width"] + bin_size_px - 1) // bin_size_px
    height_bins = (camera["height"] + bin_size_px - 1) // bin_size_px
    flat = np.full(width_bins * height_bins, np.inf, dtype=np.float64)
    indices = (v[valid] // bin_size_px) * width_bins + (u[valid] // bin_size_px)
    np.minimum.at(flat, indices, depth[valid])
    return flat.reshape(height_bins, width_bins)


def mask_bins(mask: np.ndarray, bin_size_px: int) -> np.ndarray:
    height, width = mask.shape
    padded_height = ((height + bin_size_px - 1) // bin_size_px) * bin_size_px
    padded_width = ((width + bin_size_px - 1) // bin_size_px) * bin_size_px
    padded = np.zeros((padded_height, padded_width), dtype=bool)
    padded[:height, :width] = mask
    return padded.reshape(
        padded_height // bin_size_px,
        bin_size_px,
        padded_width // bin_size_px,
        bin_size_px,
    ).any(axis=(1, 3))


def evaluate_disjoint(
    left_mask: np.ndarray,
    right_mask: np.ndarray,
) -> dict[str, Any]:
    overlap_pixels = int(np.count_nonzero(left_mask & right_mask))
    return {
        "status": "passed" if overlap_pixels == 0 else "rejected",
        "metrics": {"full_projection_overlap_pixels": overlap_pixels},
        "gates": {"full_projection_disjoint": overlap_pixels == 0},
        "claim_scope": (
            "Zero overlap of two full calibrated projections excludes a common 3D point "
            "for this camera frustum."
        ),
    }


def evaluate_depth_order(
    *,
    front_depth: np.ndarray,
    rear_depth: np.ndarray,
    front_mask_bins: np.ndarray,
    rear_mask_bins: np.ndarray,
    minimum_shared_bins: int,
    minimum_depth_margin: float,
) -> dict[str, Any]:
    shared = np.isfinite(front_depth) & np.isfinite(rear_depth) & front_mask_bins & rear_mask_bins
    differences = rear_depth[shared] - front_depth[shared]
    enough = len(differences) >= minimum_shared_bins
    if len(differences):
        quantiles = np.quantile(
            differences,
            [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0],
        )
        minimum = float(quantiles[0])
        wrong_fraction = float(np.mean(differences <= 0.0))
    else:
        quantiles = None
        minimum = None
        wrong_fraction = 1.0
    margin_passed = enough and minimum is not None and minimum >= minimum_depth_margin
    gates = {
        "minimum_shared_depth_bins": enough,
        "front_is_nearer_in_every_shared_bin": margin_passed,
        "no_depth_order_inversion": enough and wrong_fraction == 0.0,
    }
    return {
        "status": "passed" if all(gates.values()) else "rejected",
        "metrics": {
            "shared_depth_bins": len(differences),
            "rear_minus_front_depth_quantiles_m": (
                {
                    key: float(value)
                    for key, value in zip(
                        ("min", "p01", "p05", "median", "p95", "p99", "max"),
                        quantiles,
                        strict=True,
                    )
                }
                if quantiles is not None
                else None
            ),
            "minimum_rear_minus_front_depth_m": minimum,
            "wrong_order_fraction": wrong_fraction,
        },
        "thresholds": {
            "minimum_shared_bins": minimum_shared_bins,
            "minimum_depth_margin_m": minimum_depth_margin,
        },
        "gates": gates,
        "claim_scope": (
            "Frontmost mesh-vertex depth in calibrated image bins; this detects obvious "
            "depth inversion but is not an exact triangle-volume boolean intersection."
        ),
    }


def load_object(
    item: dict[str, Any],
    *,
    base: Path,
    camera: dict[str, Any],
    bin_size_px: int,
) -> dict[str, Any]:
    object_id = str(item["object_id"])
    mesh_path = resolve_path(base, str(item["mesh"]))
    report_path = resolve_path(base, str(item["scene_fit_report"]))
    camera_review_path = resolve_path(base, str(item["source_camera_review"]))
    full_mask_path = resolve_path(base, str(item["full_silhouette"]))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    camera_review = json.loads(camera_review_path.read_text(encoding="utf-8"))
    geometry_review_path = (
        resolve_path(base, str(item["scene_geometry_review"]))
        if item.get("scene_geometry_review")
        else None
    )
    baseline_review_path = (
        resolve_path(base, str(item["baseline_scene_review"]))
        if item.get("baseline_scene_review")
        else None
    )
    if (geometry_review_path is None) == (baseline_review_path is None):
        raise ValueError(
            f"{object_id} must declare exactly one scene geometry or baseline scene review"
        )
    geometry_review = (
        json.loads(geometry_review_path.read_text(encoding="utf-8"))
        if geometry_review_path
        else None
    )
    baseline_review = (
        json.loads(baseline_review_path.read_text(encoding="utf-8"))
        if baseline_review_path
        else None
    )
    if report.get("object_id") != object_id:
        raise ValueError(f"object id mismatch for {object_id}")
    if report.get("mesh", {}).get("glb_sha256") != sha256_file(mesh_path):
        raise ValueError(f"mesh hash mismatch for {object_id}")
    loaded = trimesh.load(mesh_path, force="scene")
    mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded
    pivot = np.asarray(report["baked_relative_transform"]["runtime_pivot"], dtype=np.float64)
    world_vertices = np.asarray(mesh.vertices, dtype=np.float64) + pivot
    mask = np.asarray(Image.open(full_mask_path).convert("L")) > 0
    if mask.shape != (camera["height"], camera["width"]):
        raise ValueError(f"full silhouette dimensions do not match camera for {object_id}")
    bounds = np.asarray([world_vertices.min(axis=0), world_vertices.max(axis=0)])
    if geometry_review is not None:
        support_and_anchor = geometry_review.get("status") == "passed"
    else:
        assert baseline_review is not None
        checks = baseline_review.get("checks", {})
        support_and_anchor = (
            baseline_review.get("status") == "candidate_passed"
            and checks.get("frontSceneContactUnchanged") is True
            and checks.get("collisionBoundsUnchanged") is True
        )
    gates = {
        "scene_fit": report.get("all_acceptance_gates_passed") is True,
        "source_camera": camera_review.get("status") == "passed",
        "support_and_anchor": support_and_anchor,
    }
    return {
        "object_id": object_id,
        "mask": mask,
        "mask_bins": mask_bins(mask, bin_size_px),
        "depth_bins": depth_bins(world_vertices, camera, bin_size_px),
        "bounds": bounds,
        "gates": gates,
        "sources": {
            "mesh": {"path": str(mesh_path), "sha256": sha256_file(mesh_path)},
            "scene_fit_report": {
                "path": str(report_path),
                "sha256": sha256_file(report_path),
            },
            "source_camera_review": {
                "path": str(camera_review_path),
                "sha256": sha256_file(camera_review_path),
            },
            "scene_geometry_review": (
                {
                    "path": str(geometry_review_path),
                    "sha256": sha256_file(geometry_review_path),
                }
                if geometry_review_path
                else None
            ),
            "baseline_scene_review": (
                {
                    "path": str(baseline_review_path),
                    "sha256": sha256_file(baseline_review_path),
                }
                if baseline_review_path
                else None
            ),
            "full_silhouette": {
                "path": str(full_mask_path),
                "sha256": sha256_file(full_mask_path),
            },
        },
    }


def aabb_overlap(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    extents = np.maximum(0.0, np.minimum(left[1], right[1]) - np.maximum(left[0], right[0]))
    return {
        "extents": extents.tolist(),
        "volume": float(np.prod(extents)),
        "advisory_only": True,
    }


def make_projection_overlay(
    source_frame: Path,
    objects: list[dict[str, Any]],
    output_path: Path,
) -> None:
    image = np.asarray(Image.open(source_frame).convert("RGB"), dtype=np.float32).copy()
    colors = (
        np.asarray([0, 210, 255], dtype=np.float32),
        np.asarray([255, 210, 0], dtype=np.float32),
        np.asarray([255, 70, 170], dtype=np.float32),
    )
    for index, item in enumerate(objects):
        mask = item["mask"]
        color = colors[index % len(colors)]
        image[mask] = image[mask] * 0.5 + color * 0.5
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).save(output_path)


def review_relations(spec_path: Path, output_dir: Path) -> dict[str, Any]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    base = spec_path.parent
    frame_id = str(spec["frame_id"])
    cameras_path = resolve_path(base, str(spec["cameras"]))
    source_frame_path = resolve_path(base, str(spec["source_frame"]))
    bin_size_px = int(spec.get("bin_size_px", 4))
    minimum_shared_bins = int(spec.get("minimum_shared_bins", 16))
    minimum_depth_margin = float(spec.get("minimum_depth_margin_m", 0.05))
    camera = load_camera(cameras_path, frame_id)
    objects = [
        load_object(item, base=base, camera=camera, bin_size_px=bin_size_px)
        for item in spec["objects"]
    ]
    by_id = {item["object_id"]: item for item in objects}
    if len(by_id) != len(objects):
        raise ValueError("object ids must be unique")
    relations: list[dict[str, Any]] = []
    for relation in spec["relations"]:
        kind = str(relation["kind"])
        if kind == "disjoint":
            left = by_id[str(relation["left"])]
            right = by_id[str(relation["right"])]
            result = evaluate_disjoint(left["mask"], right["mask"])
            result.update(
                {
                    "kind": kind,
                    "left": left["object_id"],
                    "right": right["object_id"],
                    "aabb_overlap": aabb_overlap(left["bounds"], right["bounds"]),
                }
            )
        elif kind == "front_before_rear":
            front = by_id[str(relation["front"])]
            rear = by_id[str(relation["rear"])]
            result = evaluate_depth_order(
                front_depth=front["depth_bins"],
                rear_depth=rear["depth_bins"],
                front_mask_bins=front["mask_bins"],
                rear_mask_bins=rear["mask_bins"],
                minimum_shared_bins=minimum_shared_bins,
                minimum_depth_margin=minimum_depth_margin,
            )
            result.update(
                {
                    "kind": kind,
                    "front": front["object_id"],
                    "rear": rear["object_id"],
                    "full_projection_overlap_pixels": int(
                        np.count_nonzero(front["mask"] & rear["mask"])
                    ),
                    "aabb_overlap": aabb_overlap(front["bounds"], rear["bounds"]),
                }
            )
        else:
            raise ValueError(f"unsupported relation kind: {kind!r}")
        relations.append(result)

    object_gates = {item["object_id"]: item["gates"] for item in objects}
    passed = all(all(gates.values()) for gates in object_gates.values()) and all(
        relation["status"] == "passed" for relation in relations
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    overlay_path = output_dir / "pairwise_projection_overlay.png"
    make_projection_overlay(source_frame_path, objects, overlay_path)
    receipt = {
        "schema_version": 1,
        "kind": "video2world.scene_fit_pairwise_review",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "passed" if passed else "rejected",
        "promotion_allowed": passed,
        "frame_id": frame_id,
        "claim_scope": (
            "Calibrated full-projection separation plus vertex-bin front-depth order and "
            "per-object support/anchor gates. Non-watertight surface meshes do not permit "
            "an exact signed-volume penetration claim."
        ),
        "sources": {
            "spec": {"path": str(spec_path), "sha256": sha256_file(spec_path)},
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "cameras": {"path": str(cameras_path), "sha256": sha256_file(cameras_path)},
            "source_frame": {
                "path": str(source_frame_path),
                "sha256": sha256_file(source_frame_path),
            },
            "objects": {item["object_id"]: item["sources"] for item in objects},
        },
        "parameters": {
            "bin_size_px": bin_size_px,
            "minimum_shared_bins": minimum_shared_bins,
            "minimum_depth_margin_m": minimum_depth_margin,
        },
        "object_gates": object_gates,
        "relations": relations,
        "outputs": {
            "projection_overlay": {
                "path": str(overlay_path),
                "sha256": sha256_file(overlay_path),
            }
        },
    }
    write_json(output_dir / "pairwise_scene_fit_review.json", receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = review_relations(
        args.spec.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
    )
    print(json.dumps(receipt, ensure_ascii=True, indent=2))
    return 0 if receipt["promotion_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
