#!/usr/bin/env python3
"""Associate source-view instance masks with exact 3D anchors for scene fitting.

SAM instance suffixes are detector ordinals and can change between frames.  This
tool projects each source 3D anchor into every calibrated frame, associates it
with the same-category mask that contains the largest fraction of projected
points, and selects a deterministic maximum-baseline view set.  The resulting
manifests are inputs to ``fit_completed_object_to_scene.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from scripts.fit_completed_object_to_scene import load_ply_points
from scripts.qa_scene_camera_silhouette import load_camera, mask_bbox
from video2world.hashing import atomic_write_json, sha256_file

CONFIG_KIND = "video2world.multiview_scene_fit_evidence_config"
AUDIT_KIND = "video2world.multiview_scene_fit_evidence_audit"
VIEW_KIND = "video2world.completed_object_source_view_manifest"


@dataclass(frozen=True)
class MaskCandidate:
    frame_id: str
    category: str
    file_name: str
    path: Path
    score: float
    declared_bbox: list[float]
    sha256: str
    pixel_count: int
    pixel_bbox: list[int]
    mask: np.ndarray


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def resolve_path(value: Any, *, relative_to: Path, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    path = path.resolve()
    require(path.exists(), f"{label} does not exist: {path}")
    return path


def require_hash(path: Path, expected: Any, *, label: str) -> str:
    observed, _ = sha256_file(path)
    if expected is not None:
        require(
            isinstance(expected, str) and len(expected) == 64,
            f"{label}.sha256 must be a 64-character digest",
        )
        require(observed == expected, f"{label} SHA-256 mismatch: {observed} != {expected}")
    return observed


def _path_record(value: Any, *, relative_to: Path, label: str) -> tuple[Path, str]:
    if isinstance(value, str):
        path = resolve_path(value, relative_to=relative_to, label=label)
        expected = None
    else:
        require(isinstance(value, dict), f"{label} must be a path string or object")
        path = resolve_path(value.get("path"), relative_to=relative_to, label=f"{label}.path")
        expected = value.get("sha256")
    require(path.is_file(), f"{label} must be a file: {path}")
    return path, require_hash(path, expected, label=label)


def _load_cameras(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, list) and payload, "cameras must be a non-empty JSON array")
    result: dict[str, dict[str, Any]] = {}
    for raw in payload:
        require(isinstance(raw, dict), "camera entries must be objects")
        frame_id = str(raw.get("img_name"))
        require(frame_id not in result, f"duplicate camera frame_id: {frame_id}")
        result[frame_id] = load_camera(path, frame_id)
    return result


def _load_mask_candidates(
    *,
    mask_index_path: Path,
    mask_root: Path,
    cameras: dict[str, dict[str, Any]],
    minimum_score: float,
) -> dict[str, list[MaskCandidate]]:
    payload = read_json(mask_index_path)
    items = payload.get("items")
    require(isinstance(items, list) and items, "mask index requires a non-empty items array")
    result: dict[str, list[MaskCandidate]] = {}
    for item in items:
        require(isinstance(item, dict), "mask index items must be objects")
        image_name = item.get("image")
        category = item.get("label")
        declared_path = item.get("mask_path")
        if not all(
            isinstance(value, str) and value for value in (image_name, category, declared_path)
        ):
            continue
        frame_id = Path(image_name).stem
        if frame_id not in cameras:
            continue
        score = float(item.get("score", 0.0))
        if not math.isfinite(score) or score < minimum_score:
            continue
        file_name = Path(declared_path).name
        local_path = (mask_root / frame_id / file_name).resolve()
        require(local_path.is_file(), f"materialized mask is missing: {local_path}")
        with Image.open(local_path) as image:
            mask = np.asarray(image.convert("L")) > 0
        camera = cameras[frame_id]
        require(
            mask.shape == (camera["height"], camera["width"]),
            f"mask dimensions do not match camera: {local_path}",
        )
        bbox = mask_bbox(mask)
        require(bbox is not None, f"mask is empty: {local_path}")
        digest, _ = sha256_file(local_path)
        declared_bbox = [float(value) for value in item.get("bbox", [])]
        require(len(declared_bbox) == 4, f"mask index bbox is invalid: {local_path}")
        result.setdefault(frame_id, []).append(
            MaskCandidate(
                frame_id=frame_id,
                category=category,
                file_name=file_name,
                path=local_path,
                score=score,
                declared_bbox=declared_bbox,
                sha256=digest,
                pixel_count=int(np.count_nonzero(mask)),
                pixel_bbox=bbox,
                mask=mask,
            )
        )
    require(result, "no usable materialized masks were found")
    return result


def _project_anchor(points: np.ndarray, camera: dict[str, Any]) -> dict[str, Any]:
    relative = points - camera["position"]
    camera_points = np.einsum("ni,ij->nj", relative, camera["rotation"], optimize=False)
    depth = camera_points[:, 2]
    visible = depth > 1e-6
    x = np.full(len(points), np.nan, dtype=np.float64)
    y = np.full(len(points), np.nan, dtype=np.float64)
    x[visible] = camera["fx"] * camera_points[visible, 0] / depth[visible] + camera["width"] / 2
    y[visible] = camera["height"] / 2 + camera["fy"] * camera_points[visible, 1] / depth[visible]
    inside = visible & (x >= 0) & (x < camera["width"]) & (y >= 0) & (y < camera["height"])
    require(np.any(inside), "anchor has no projected points inside the source frame")
    projected = np.column_stack((x[inside], y[inside]))
    pixels = np.rint(projected).astype(np.int64)
    pixels[:, 0] = np.clip(pixels[:, 0], 0, camera["width"] - 1)
    pixels[:, 1] = np.clip(pixels[:, 1], 0, camera["height"] - 1)
    return {
        "pixels": pixels,
        "point_count": len(pixels),
        "bbox_xyxy": [
            float(projected[:, 0].min()),
            float(projected[:, 1].min()),
            float(projected[:, 0].max()),
            float(projected[:, 1].max()),
        ],
    }


def _associate(
    *,
    points: np.ndarray,
    camera: dict[str, Any],
    candidates: list[MaskCandidate],
    category: str,
    minimum_hit_ratio: float,
    minimum_hit_margin: float,
) -> dict[str, Any]:
    projection = _project_anchor(points, camera)
    pixels = projection["pixels"]
    matches = []
    for candidate in candidates:
        if candidate.category != category:
            continue
        hit_ratio = float(np.mean(candidate.mask[pixels[:, 1], pixels[:, 0]]))
        matches.append((hit_ratio, candidate.file_name, candidate))
    require(matches, f"no {category!r} masks exist for frame {candidates[0].frame_id}")
    matches.sort(key=lambda value: (-value[0], value[1]))
    selected_ratio, _, selected = matches[0]
    runner_up_ratio = matches[1][0] if len(matches) > 1 else 0.0
    margin = selected_ratio - runner_up_ratio
    passed = selected_ratio >= minimum_hit_ratio and margin >= minimum_hit_margin
    return {
        "passed": passed,
        "selected": selected,
        "selected_hit_ratio": selected_ratio,
        "runner_up_hit_ratio": runner_up_ratio,
        "hit_ratio_margin": margin,
        "projected_anchor_point_count_inside_image": projection["point_count"],
        "projected_anchor_bbox_xyxy": projection["bbox_xyxy"],
        "candidates": [
            {
                "file_name": candidate.file_name,
                "sha256": candidate.sha256,
                "sam3_score": candidate.score,
                "hit_ratio": ratio,
            }
            for ratio, _, candidate in matches
        ],
    }


def _ray_angle(left: np.ndarray, right: np.ndarray) -> float:
    cosine = float(np.clip(np.dot(left, right), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _select_maximum_baseline_frames(
    frame_ids: list[str],
    *,
    cameras: dict[str, dict[str, Any]],
    object_center: np.ndarray,
    count: int,
) -> tuple[list[str], dict[str, Any]]:
    require(count >= 1, "selected view count must be positive")
    require(len(frame_ids) >= count, "not enough eligible frames for requested view count")
    rays: dict[str, np.ndarray] = {}
    for frame_id in frame_ids:
        ray = object_center - cameras[frame_id]["position"]
        norm = float(np.linalg.norm(ray))
        require(norm > 1e-9, f"camera coincides with object center for {frame_id}")
        rays[frame_id] = ray / norm
    ordered = sorted(frame_ids)
    if count == 1:
        selected = [ordered[len(ordered) // 2]]
        return selected, {"policy": "middle_eligible_frame", "selected_frame_ids": selected}
    pairs = [
        (_ray_angle(rays[left], rays[right]), left, right)
        for index, left in enumerate(ordered)
        for right in ordered[index + 1 :]
    ]
    maximum_angle, left, right = max(pairs, key=lambda value: (value[0], value[1], value[2]))
    selected = [left, right]
    additions = []
    while len(selected) < count:
        remaining = [frame_id for frame_id in ordered if frame_id not in selected]
        scored = []
        for frame_id in remaining:
            angles = [_ray_angle(rays[frame_id], rays[chosen]) for chosen in selected]
            scored.append((min(angles), frame_id, angles))
        minimum_angle, frame_id, angles = max(scored, key=lambda value: (value[0], value[1]))
        selected.append(frame_id)
        additions.append(
            {
                "frame_id": frame_id,
                "minimum_angle_to_selected_deg": minimum_angle,
                "angles_to_prior_selected_deg": angles,
            }
        )
    selected.sort()
    return selected, {
        "policy": "maximum_pairwise_camera_to_object_view_ray_angle_then_maximin_additions",
        "maximum_pair": {
            "frame_ids": [left, right],
            "view_ray_angle_deg": maximum_angle,
        },
        "maximin_additions": additions,
        "selected_frame_ids": selected,
    }


def _relative_path(path: Path, *, relative_to: Path) -> str:
    return Path(os.path.relpath(path, relative_to)).as_posix()


def prepare_multiview_evidence(*, config_path: Path, output_dir: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    config = read_json(config_path)
    require(config.get("schema_version") == 1, "config schema_version must be 1")
    require(config.get("kind") == CONFIG_KIND, f"config kind must be {CONFIG_KIND}")
    config_dir = config_path.parent
    cameras_path, cameras_sha = _path_record(
        config.get("cameras"), relative_to=config_dir, label="cameras"
    )
    mask_index_path, mask_index_sha = _path_record(
        config.get("mask_index"), relative_to=config_dir, label="mask_index"
    )
    mask_root = resolve_path(config.get("mask_root"), relative_to=config_dir, label="mask_root")
    source_rgb_dir = resolve_path(
        config.get("source_rgb_dir"), relative_to=config_dir, label="source_rgb_dir"
    )
    require(mask_root.is_dir(), "mask_root must be a directory")
    require(source_rgb_dir.is_dir(), "source_rgb_dir must be a directory")
    thresholds = config.get("thresholds", {})
    require(isinstance(thresholds, dict), "thresholds must be an object")
    minimum_score = float(thresholds.get("minimum_sam3_score", 0.75))
    minimum_hit_ratio = float(thresholds.get("minimum_anchor_hit_ratio", 0.25))
    minimum_hit_margin = float(thresholds.get("minimum_anchor_hit_margin", 0.05))
    selected_view_count = int(config.get("selected_view_count", 3))
    require(0 <= minimum_score <= 1, "minimum_sam3_score must be in [0, 1]")
    require(0 <= minimum_hit_ratio <= 1, "minimum_anchor_hit_ratio must be in [0, 1]")
    require(0 <= minimum_hit_margin <= 1, "minimum_anchor_hit_margin must be in [0, 1]")
    cameras = _load_cameras(cameras_path)
    candidates = _load_mask_candidates(
        mask_index_path=mask_index_path,
        mask_root=mask_root,
        cameras=cameras,
        minimum_score=minimum_score,
    )
    configured_frames = config.get("frame_ids")
    if configured_frames is None:
        frame_ids = sorted(set(candidates).intersection(cameras))
    else:
        require(isinstance(configured_frames, list) and configured_frames, "frame_ids is invalid")
        frame_ids = [str(value) for value in configured_frames]
    require(len(set(frame_ids)) == len(frame_ids), "frame_ids must be unique")
    for frame_id in frame_ids:
        require(frame_id in candidates, f"no mask candidates for frame {frame_id}")
        require(frame_id in cameras, f"no camera for frame {frame_id}")
        require((source_rgb_dir / f"{frame_id}.png").is_file(), f"source RGB missing: {frame_id}")

    object_values = config.get("objects")
    require(isinstance(object_values, list) and object_values, "config requires objects")
    objects: dict[str, dict[str, Any]] = {}
    for value in object_values:
        require(isinstance(value, dict), "object entries must be objects")
        object_id = value.get("object_id")
        category = value.get("category")
        require(isinstance(object_id, str) and object_id, "object_id is required")
        require(isinstance(category, str) and category, f"category is required for {object_id}")
        require(object_id not in objects, f"duplicate object_id: {object_id}")
        anchor_path, anchor_sha = _path_record(
            value.get("anchor"), relative_to=config_dir, label=f"{object_id}.anchor"
        )
        points, loaded_sha = load_ply_points(anchor_path, expected_sha256=anchor_sha)
        require(loaded_sha == anchor_sha, f"anchor hash changed while loading: {object_id}")
        stride = max(1, len(points) // 250_000)
        sampled_points = points[::stride]
        associations = {}
        for frame_id in frame_ids:
            associations[frame_id] = _associate(
                points=sampled_points,
                camera=cameras[frame_id],
                candidates=candidates[frame_id],
                category=category,
                minimum_hit_ratio=minimum_hit_ratio,
                minimum_hit_margin=minimum_hit_margin,
            )
        eligible = [frame_id for frame_id in frame_ids if associations[frame_id]["passed"]]
        selected, selection = _select_maximum_baseline_frames(
            eligible,
            cameras=cameras,
            object_center=np.median(sampled_points, axis=0),
            count=selected_view_count,
        )
        occluders = value.get("occluder_object_ids", [])
        require(isinstance(occluders, list), f"occluder_object_ids is invalid for {object_id}")
        objects[object_id] = {
            "object_id": object_id,
            "category": category,
            "anchor_path": anchor_path,
            "anchor_sha256": anchor_sha,
            "anchor_point_count": len(points),
            "anchor_sample_stride": stride,
            "anchor_center": np.median(sampled_points, axis=0).tolist(),
            "occluder_object_ids": [str(item) for item in occluders],
            "associations": associations,
            "eligible_frame_ids": eligible,
            "selected_frame_ids": selected,
            "frame_selection": selection,
        }
    for value in objects.values():
        for occluder_id in value["occluder_object_ids"]:
            require(occluder_id in objects, f"unknown occluder object_id: {occluder_id}")
            for frame_id in value["selected_frame_ids"]:
                require(
                    objects[occluder_id]["associations"][frame_id]["passed"],
                    f"occluder association failed for {occluder_id} in {frame_id}",
                )

    config_sha, _ = sha256_file(config_path)
    audit_objects = []
    for object_id in sorted(objects):
        value = objects[object_id]
        selected_set = set(value["selected_frame_ids"])
        views = []
        for frame_id in frame_ids:
            association = value["associations"][frame_id]
            selected_mask: MaskCandidate = association["selected"]
            source_rgb = source_rgb_dir / f"{frame_id}.png"
            source_sha, _ = sha256_file(source_rgb)
            views.append(
                {
                    "frame_id": frame_id,
                    "selected_for_fit": frame_id in selected_set,
                    "association": {
                        "status": "passed" if association["passed"] else "rejected",
                        "method": (
                            "project_exact_source_3d_anchor_then_maximize_"
                            "same_category_mask_point_hit_ratio"
                        ),
                        "selected_mask_file_name": selected_mask.file_name,
                        "selected_hit_ratio": association["selected_hit_ratio"],
                        "runner_up_hit_ratio": association["runner_up_hit_ratio"],
                        "hit_ratio_margin": association["hit_ratio_margin"],
                        "projected_anchor_point_count_inside_image": association[
                            "projected_anchor_point_count_inside_image"
                        ],
                        "projected_anchor_bbox_xyxy": association["projected_anchor_bbox_xyxy"],
                        "all_same_category_candidates": association["candidates"],
                    },
                    "mask": {
                        "path": str(selected_mask.path),
                        "sha256": selected_mask.sha256,
                        "sam3_score": selected_mask.score,
                        "pixel_count": selected_mask.pixel_count,
                        "pixel_bbox_xyxy_exclusive": selected_mask.pixel_bbox,
                        "mask_index_bbox_xyxy": selected_mask.declared_bbox,
                    },
                    "source_rgb": {
                        "path": str(source_rgb.resolve()),
                        "sha256": source_sha,
                        "dimensions_wh": [
                            cameras[frame_id]["width"],
                            cameras[frame_id]["height"],
                        ],
                    },
                }
            )
        audit_objects.append(
            {
                "object_id": object_id,
                "category": value["category"],
                "source_anchor": {
                    "path": str(value["anchor_path"]),
                    "sha256": value["anchor_sha256"],
                    "point_count": value["anchor_point_count"],
                    "sample_stride": value["anchor_sample_stride"],
                    "robust_center": value["anchor_center"],
                },
                "occluder_object_ids": value["occluder_object_ids"],
                "eligible_frame_count": len(value["eligible_frame_ids"]),
                "frame_selection": value["frame_selection"],
                "views": views,
            }
        )
    audit = {
        "schema_version": 1,
        "kind": AUDIT_KIND,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "source_multiview_modal_masks_ready_for_scene_fit",
        "claim_scope": (
            "Source-camera placement and visible-proportion evidence only. Modal SAM masks "
            "and the selected camera arc do not prove unobserved back, top, or bottom geometry."
        ),
        "config": {"path": str(config_path), "sha256": config_sha},
        "inputs": {
            "cameras": {"path": str(cameras_path), "sha256": cameras_sha},
            "mask_index": {"path": str(mask_index_path), "sha256": mask_index_sha},
            "mask_root": str(mask_root),
            "source_rgb_dir": str(source_rgb_dir),
            "frame_ids": frame_ids,
        },
        "thresholds": {
            "minimum_sam3_score": minimum_score,
            "minimum_anchor_hit_ratio": minimum_hit_ratio,
            "minimum_anchor_hit_margin": minimum_hit_margin,
        },
        "objects": audit_objects,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "source-view-audit.json"
    atomic_write_json(audit_path, audit)
    audit_sha, _ = sha256_file(audit_path)
    views_dir = output_dir / "views"
    manifests = []
    for object_id in sorted(objects):
        value = objects[object_id]
        view_entries = []
        for frame_id in value["selected_frame_ids"]:
            selected: MaskCandidate = value["associations"][frame_id]["selected"]
            occluder_masks = []
            for occluder_id in value["occluder_object_ids"]:
                occluder: MaskCandidate = objects[occluder_id]["associations"][frame_id]["selected"]
                if occluder.sha256 == selected.sha256:
                    continue
                occluder_masks.append(
                    {
                        "object_id": occluder_id,
                        "path": _relative_path(occluder.path, relative_to=views_dir),
                        "sha256": occluder.sha256,
                    }
                )
            view_entries.append(
                {
                    "frame_id": frame_id,
                    "is_source_view": True,
                    "observed_mask": {
                        "path": _relative_path(selected.path, relative_to=views_dir),
                        "sha256": selected.sha256,
                    },
                    "occluder_masks": occluder_masks,
                }
            )
        manifest = {
            "schema_version": 1,
            "kind": VIEW_KIND,
            "evidence_scope": "multiview_source_modal_masks",
            "object_id": object_id,
            "source_audit": {
                "path": _relative_path(audit_path, relative_to=views_dir),
                "sha256": audit_sha,
            },
            "views": view_entries,
        }
        manifest_path = views_dir / f"{object_id}.json"
        atomic_write_json(manifest_path, manifest)
        manifest_sha, _ = sha256_file(manifest_path)
        manifests.append(
            {
                "object_id": object_id,
                "path": str(manifest_path),
                "sha256": manifest_sha,
                "frame_ids": value["selected_frame_ids"],
            }
        )
    return {
        "audit": {"path": str(audit_path), "sha256": audit_sha},
        "manifests": manifests,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = prepare_multiview_evidence(config_path=args.config, output_dir=args.output_dir)
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
