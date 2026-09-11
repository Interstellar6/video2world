#!/usr/bin/env python3
"""Validate multi-view instance tracks and carve observed PGSR Gaussians.

All paths in input manifests are task-relative. Depth intrinsics describe the
depth raster, unless intrinsics_resolution explicitly gives their source size.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from world_modeling.component_tracks import group_component_tracks
from world_modeling.gaussian_io import read_gaussian_rows
from world_modeling.physical_tracks import group_physical_tracks


class LiftingError(RuntimeError):
    pass


def task_path(root: Path, value, *, exists=True) -> Path:
    root = root.resolve()
    if not isinstance(value, (str, Path)):
        raise LiftingError("task path must be a string")
    path = (root / value).resolve()
    if root not in path.parents or (exists and not path.is_file()):
        raise LiftingError(f"missing or non-task-local file: {value}")
    return path


def relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def read_json(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise LiftingError(f"expected JSON object: {path}")
    return result


def write_json(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_index(document: dict, name: str) -> dict:
    frames = document.get("frames", document.get("cameras"))
    if not isinstance(frames, list) or not frames:
        raise LiftingError(f"{name} requires a nonempty frames list")
    index = {}
    for frame in frames:
        if not isinstance(frame, dict) or "frame_id" not in frame:
            raise LiftingError(f"{name}: invalid frame record")
        key = str(frame["frame_id"])
        if key in index:
            raise LiftingError(f"{name}: duplicate frame_id {key}")
        index[key] = frame
    return index


def matrix(value, shape: tuple, name: str):
    import numpy as np

    result = np.asarray(value, dtype=np.float64)
    if result.shape == (3, 4) and shape == (4, 4):
        result = np.vstack((result, [0, 0, 0, 1]))
    if result.shape != shape or not np.isfinite(result).all():
        raise LiftingError(f"{name} must be a finite {shape} matrix")
    return result


def intrinsics(value):
    import numpy as np

    result = matrix(value, (3, 3), "intrinsics")
    if result[0, 0] <= 0 or result[1, 1] <= 0 or not np.allclose(result[2], [0, 0, 1]):
        raise LiftingError("intrinsics must be a positive-focal pinhole calibration")
    return result


def world_to_camera(value):
    import numpy as np

    result = matrix(value, (4, 4), "world_to_camera")
    rotation = result[:3, :3]
    if not np.allclose(result[3], [0, 0, 0, 1]) or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-4):
        raise LiftingError("world_to_camera must be a rigid right-handed transform; encode scene scale in depth units")
    return result


def raster_size(record: dict):
    value = record.get("intrinsics_resolution", record.get("image_size", record.get("resolution")))
    if value is None and "width" in record and "height" in record:
        value = [record["width"], record["height"]]
    if value is not None:
        if not isinstance(value, (list, tuple)) or len(value) != 2 or min(value) <= 0:
            raise LiftingError("raster dimensions must be [width, height]")
        return float(value[0]), float(value[1])
    return None


def array_file(path: Path, key: str):
    import numpy as np
    from PIL import Image

    suffix = path.suffix.lower()
    if suffix == ".npy":
        result = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if key not in archive:
                raise LiftingError(f"{path.name}: missing array key {key}")
            result = archive[key]
    elif suffix in {".png", ".tif", ".tiff"}:
        result = np.asarray(Image.open(path))
    else:
        raise LiftingError(f"unsupported calibrated array format: {path.suffix}")
    result = np.asarray(result)
    if result.ndim != 2:
        raise LiftingError(f"{path.name}: calibrated arrays must be H x W")
    return result


@dataclass
class Frame:
    frame_id: str
    depth: object
    valid: object
    intrinsics: object
    world_to_camera: object
    depth_path: Path
    confidence_path: Path
    depth_scale: float
    image_to_depth: object = None
    source_size: tuple | None = None

    @property
    def center(self):
        return -self.world_to_camera[:3, :3].T @ self.world_to_camera[:3, 3]


def load_frame(root: Path, frame_id: str, record: dict, camera: dict, args) -> Frame:
    import numpy as np

    depth_path = task_path(root, record["depth_path"])
    confidence_path = task_path(root, record["confidence_path"])
    raw_depth = array_file(depth_path, record.get("depth_key", "depth"))
    if np.issubdtype(raw_depth.dtype, np.integer) and "depth_scale" not in record:
        raise LiftingError(f"{frame_id}: integer depth requires an explicit depth_scale")
    scale = float(record.get("depth_scale", 1))
    if not np.isfinite(scale) or scale <= 0:
        raise LiftingError(f"{frame_id}: depth_scale must be finite and positive")
    depth = raw_depth.astype(np.float64) * scale
    confidence = array_file(confidence_path, record.get("confidence_key", "confidence"))
    if confidence.shape != depth.shape:
        raise LiftingError(f"{frame_id}: confidence raster must match depth raster")
    height, width = depth.shape
    calibration = intrinsics(record["intrinsics"])
    source_size = record.get("intrinsics_resolution")
    if source_size is not None:
        source_width, source_height = raster_size({"intrinsics_resolution": source_size})
        calibration[0] *= width / source_width
        calibration[1] *= height / source_height
    pose = world_to_camera(record["world_to_camera"])
    camera_pose = world_to_camera(camera["world_to_camera"])
    if not np.allclose(pose, camera_pose, atol=1e-5):
        raise LiftingError(f"{frame_id}: depth and cameras world_to_camera disagree")
    camera_calibration = intrinsics(camera["intrinsics"])
    camera_size = raster_size(camera) or (width, height)
    if "image_to_depth" in record:
        image_to_depth = matrix(record["image_to_depth"], (3, 3), "image_to_depth")
        if not np.allclose(image_to_depth[2], [0, 0, 1]):
            raise LiftingError(f"{frame_id}: image_to_depth must be a raster affine transform")
    else:
        image_to_depth = np.diag([width / camera_size[0], height / camera_size[1], 1])
    if not np.allclose(calibration, image_to_depth @ camera_calibration, rtol=1e-4, atol=0.05):
        raise LiftingError(f"{frame_id}: depth and camera intrinsics disagree after image_to_depth mapping")
    valid = np.isfinite(depth) & (depth > 0) & np.isfinite(confidence) & (confidence >= args.confidence_min)
    return Frame(frame_id, depth, valid, calibration, pose, depth_path, confidence_path, scale, image_to_depth, camera_size)


def load_mask(root: Path, path, shape: tuple, frame: Frame | None = None):
    import numpy as np
    from PIL import Image

    source = task_path(root, path)
    image = Image.open(source).convert("L")
    if frame is not None and frame.image_to_depth is not None:
        original_width, original_height = frame.source_size
        mask_to_source = np.diag([original_width / image.width, original_height / image.height, 1])
        transform = frame.image_to_depth @ mask_to_source
        y, x = np.indices(shape)
        pixels = np.stack((x, y, np.ones(shape)), axis=-1) @ np.linalg.inv(transform).T
        uv = np.rint(pixels[:, :, :2] / pixels[:, :, 2, None]).astype(int)
        inside = (uv[:, :, 0] >= 0) & (uv[:, :, 0] < image.width) & (uv[:, :, 1] >= 0) & (uv[:, :, 1] < image.height)
        result = np.zeros(shape, dtype=bool)
        result[inside] = np.asarray(image)[uv[inside, 1], uv[inside, 0]] > 0
        return result
    if image.size != (shape[1], shape[0]):
        image = image.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    return np.asarray(image) > 0


def backproject(frame: Frame, mask):
    import numpy as np

    y, x = np.nonzero(mask & frame.valid)
    pixels = np.column_stack((x, y, np.ones(len(x))))
    camera_points = (pixels @ np.linalg.inv(frame.intrinsics).T) * frame.depth[y, x, None]
    world = (camera_points - frame.world_to_camera[:3, 3]) @ frame.world_to_camera[:3, :3]
    return world


def voxel_sample(points, voxel_size: float, max_points: int):
    import numpy as np

    if not len(points):
        return points
    voxels = np.floor(points / voxel_size).astype(np.int64)
    _, indices = np.unique(voxels, axis=0, return_index=True)
    if len(indices) > max_points:
        indices = indices[np.linspace(0, len(indices) - 1, max_points, dtype=int)]
    return points[indices]


def project(points, frame: Frame):
    import numpy as np

    camera = points @ frame.world_to_camera[:3, :3].T + frame.world_to_camera[:3, 3]
    depth = camera[:, 2]
    positive = depth > 1e-10
    projected = camera @ frame.intrinsics.T
    uv = np.zeros((len(points), 2), dtype=np.int64)
    uv[positive] = np.rint(projected[positive, :2] / projected[positive, 2, None]).astype(np.int64)
    height, width = frame.depth.shape
    in_frame = positive & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    return uv, depth, in_frame


def surface_votes(points, frame: Frame, mask, absolute_tolerance: float, relative_tolerance: float):
    import numpy as np

    uv, point_depth, in_frame = project(points, frame)
    indices = np.flatnonzero(in_frame)
    pixels = uv[indices]
    indices = indices[frame.valid[pixels[:, 1], pixels[:, 0]]]
    pixels = uv[indices]
    observed_depth = frame.depth[pixels[:, 1], pixels[:, 0]]
    tolerance = absolute_tolerance + relative_tolerance * observed_depth
    delta = point_depth[indices] - observed_depth
    surface = np.abs(delta) <= tolerance
    in_mask = mask[pixels[:, 1], pixels[:, 0]]
    result = {key: np.zeros(len(points), dtype=bool) for key in ("positive", "negative", "occluded", "in_front")}
    result["positive"][indices[surface & in_mask]] = True
    result["negative"][indices[surface & ~in_mask]] = True
    result["occluded"][indices[delta > tolerance]] = True
    result["in_front"][indices[delta < -tolerance]] = True
    return result


def depth_alignment(points, frame: Frame, args, absolute_tolerance: float) -> dict:
    import numpy as np

    uv, point_depth, in_frame = project(points, frame)
    height, width = frame.depth.shape
    zbuffer = np.full(height * width, np.inf)
    inside = uv[in_frame]
    np.minimum.at(zbuffer, inside[:, 1] * width + inside[:, 0], point_depth[in_frame])
    zbuffer = zbuffer.reshape(height, width)
    valid = frame.valid & np.isfinite(zbuffer)
    if not valid.any():
        return {"accepted": False, "reason": "no_scene_depth_correspondence", "pixels": 0}
    reference = zbuffer[valid]
    depth = frame.depth[valid]
    ratio = float(np.median(depth / reference))
    agreement = float(np.mean(np.abs(reference - depth) <= absolute_tolerance + args.depth_relative_tolerance * depth))
    accepted = abs(ratio - 1) <= args.depth_scale_tolerance and agreement >= args.min_depth_agreement
    return {"accepted": bool(accepted), "pixels": int(valid.sum()), "median_depth_over_scene_z": ratio, "surface_agreement_ratio": agreement, "reason": "accepted" if accepted else "depth_scale_or_scene_alignment_mismatch"}


def association_pair(first, second, frames, voxel_size, absolute_tolerance, args) -> dict:
    import numpy as np
    from scipy.spatial import cKDTree

    first_frame, second_frame = frames[first["frame_id"]], frames[second["frame_id"]]
    baseline = float(np.linalg.norm(first_frame.center - second_frame.center))
    if len(first["points"]) and len(second["points"]):
        radius = voxel_size * args.overlap_radius_voxels
        forward = float(np.mean(cKDTree(second["points"]).query(first["points"], distance_upper_bound=radius)[0] < np.inf))
        backward = float(np.mean(cKDTree(first["points"]).query(second["points"], distance_upper_bound=radius)[0] < np.inf))
    else:
        forward = backward = 0.0
    contradictions = []
    for source, target, target_frame in ((first, second, second_frame), (second, first, first_frame)):
        votes = surface_votes(source["points"], target_frame, target["mask"], absolute_tolerance, args.depth_relative_tolerance)
        positive, negative = int(votes["positive"].sum()), int(votes["negative"].sum())
        fraction = negative / max(1, positive + negative)
        conflict = negative + positive >= args.min_mask_pixels and fraction > args.max_visible_disagreement
        contradictions.append({"positive": positive, "negative": negative, "occluded": int(votes["occluded"].sum()),
                               "in_front": int(votes["in_front"].sum()), "visible_disagreement_ratio": fraction, "conflict": conflict})
    accepted = min(forward, backward) >= args.min_overlap_ratio and baseline >= voxel_size and not any(item["conflict"] for item in contradictions)
    return {"frame_ids": [first["frame_id"], second["frame_id"]], "baseline": baseline,
            "overlap_radius": voxel_size * args.overlap_radius_voxels, "forward_overlap_ratio": forward,
            "backward_overlap_ratio": backward, "reprojection": contradictions, "accepted": accepted}


def association_report(observations: list[dict], frames: dict, scene_tree, voxel_size: float, absolute_tolerance: float, args) -> dict:
    import numpy as np

    details = []
    for observation in observations:
        frame = frames[observation["frame_id"]]
        points = observation["points"]
        if len(points):
            radius = max(voxel_size * args.overlap_radius_voxels, args.depth_relative_tolerance * float(np.median(frame.depth[observation["mask"] & frame.valid])))
            support = float(np.mean(scene_tree.query(points, distance_upper_bound=radius)[0] < np.inf))
        else:
            support = 0.0
        details.append({"frame_id": frame.frame_id, "valid_mask_pixels": observation["valid_pixels"], "sampled_voxels": len(points), "scene_support_ratio": support})
    reasons = []
    if len(observations) < args.min_views:
        reasons.append("insufficient_distinct_frames")
    if any(item["valid_mask_pixels"] < args.min_mask_pixels for item in details):
        reasons.append("insufficient_confident_mask_depth")
    if any(item["scene_support_ratio"] < args.min_scene_support for item in details):
        reasons.append("lifted_depth_not_supported_by_scene_geometry")
    pairs = []
    adjacency = {i: set() for i in range(len(observations))}
    conflicting = []
    for i, first in enumerate(observations):
        for j in range(i + 1, len(observations)):
            second = observations[j]
            pair = association_pair(first, second, frames, voxel_size, absolute_tolerance, args)
            if pair["accepted"]:
                adjacency[i].add(j)
                adjacency[j].add(i)
            for item in pair["reprojection"]:
                if item["conflict"]:
                    conflicting.append({"frame_ids": pair["frame_ids"], **item})
            pairs.append(pair)
    visited, queue = set(), [0] if observations else []
    while queue:
        index = queue.pop()
        if index in visited:
            continue
        visited.add(index)
        queue.extend(adjacency[index] - visited)
    if len(visited) != len(observations) or len(observations) < 2:
        reasons.append("disconnected_geometric_identity_observations")
    total_pairs = len(pairs)
    # A pair of views separated by a wide baseline legitimately disagrees about
    # occluding a surface they see from opposite sides -- the ScanNet++ DSLR bed
    # showed exactly one such pair at the two ends of an otherwise 64/66
    # consistent, fully connected track. Identity is established by the
    # accepted-pair connectivity above, so a visibility conflict is reported and
    # never decides the track on its own.
    conflicts = {
        "policy": "recorded_not_blocking",
        "conflicting_pairs": len(conflicting),
        "total_pairs": total_pairs,
        "conflict_ratio": (len(conflicting) / total_pairs) if total_pairs else 0.0,
        "worst_visible_disagreement_ratio": max((item["visible_disagreement_ratio"] for item in conflicting), default=0.0),
        "frame_pairs": [item["frame_ids"] for item in conflicting[:20]],
    }
    return {"accepted": not reasons, "reasons": sorted(set(reasons)), "observations": details, "pairs": pairs,
            "visibility_conflicts": conflicts,
            "association_method": "world_space_nearest_voxel_overlap_and_depth_visible_reprojection"}


def composite_policy(value) -> dict:
    policy = json.loads(value) if isinstance(value, str) else value
    if not isinstance(policy, dict):
        raise LiftingError("composite policy must map parent categories to component groups and relationships")
    for category, groups in policy.items():
        if not isinstance(category, str) or not category or not isinstance(groups, dict):
            raise LiftingError("invalid composite policy category")
        for group, relation in groups.items():
            if not isinstance(group, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", group) or relation not in ("structural_part", "bedding"):
                raise LiftingError("only explicit structural_part or bedding groups may extend an asset")
            if relation == "bedding" and category != "bed":
                raise LiftingError("bedding composition requires a bed parent")
    return policy


def source_mask(root, record):
    import numpy as np
    from PIL import Image

    path = task_path(root, record["mask_path"])
    if record.get("mask_sha256") and sha256(path) != record["mask_sha256"]:
        raise LiftingError("candidate mask changed after segmentation")
    with Image.open(path) as image:
        result = np.asarray(image.convert("L"))
    if not np.isin(result, [0, 255]).all():
        raise LiftingError("candidate mask must be a binary PNG raster")
    return result > 127


def resolve_parent_hypotheses(root, tracks_document, components_document, frame_cache, camera_records,
                              depth_records, scene_sample, voxel_size, absolute_tolerance, args, alignments):
    """Build provisional physical parents without using VLM IDs as connectivity."""
    binding = tracks_document.get("source_descriptions")
    if not isinstance(binding, dict) or components_document.get("source_descriptions") != binding:
        raise LiftingError("parent association requires matching source_descriptions bindings from SAM")
    descriptions_path = task_path(root, binding.get("path"))
    if binding.get("sha256") != sha256(descriptions_path):
        raise LiftingError("parent association source descriptions hash mismatch")
    descriptions = {}
    for item in read_json(descriptions_path).get("objects", []):
        key = (str(item["frame_id"]), item["object_id"])
        if key in descriptions:
            raise LiftingError("duplicate source frame/object description")
        descriptions[key] = item
    if not descriptions:
        raise LiftingError("parent association requires source object descriptions")
    components = components_document["masks"]
    parents = {}
    for record in components:
        if record.get("component_id") != "__object__":
            continue
        key = (str(record["frame_id"]), record["object_id"])
        if key in parents:
            raise LiftingError("duplicate parent candidate for a frame/source object")
        parents[key] = record
    evidence, lifted, originals, seen = [], {}, {}, set()
    seen_tracks = set()
    for track in tracks_document["tracks"]:
        source_id, category = track["object_id"], track.get("category")
        if source_id in seen_tracks:
            raise LiftingError("duplicate input parent hypothesis")
        seen_tracks.add(source_id)
        for observation in track["observations"]:
            frame_id = str(observation["frame_id"])
            key = (frame_id, source_id)
            if key in seen:
                raise LiftingError("duplicate source parent observation")
            seen.add(key)
            if key not in parents or key not in descriptions:
                raise LiftingError("parent observation lacks its exact source candidate or description")
            parent, description = parents[key], descriptions[key]
            if description.get("category") != category or parent.get("category", category) != category:
                raise LiftingError("source parent category disagrees with its frame-local description")
            image_path = task_path(root, description.get("image_path"))
            mask_path = task_path(root, observation["mask_path"])
            if task_path(root, parent["mask_path"]) != mask_path:
                raise LiftingError("parent hypothesis and candidate masks disagree")
            for record in (observation, parent):
                if task_path(root, record.get("image_path")) != image_path:
                    raise LiftingError("source parent RGB disagrees with its description")
                if record.get("source_object_id", source_id) != source_id:
                    raise LiftingError("source parent object ID disagrees with its enclosing hypothesis")
                if record.get("mask_sha256") != sha256(mask_path):
                    raise LiftingError("source parent mask hash mismatch")
                if "image_sha256" in record and record["image_sha256"] != sha256(image_path):
                    raise LiftingError("source parent RGB hash mismatch")
            if frame_id not in camera_records or frame_id not in depth_records:
                raise LiftingError(f"missing calibrated frame for parent observation {frame_id}")
            if frame_id not in frame_cache:
                frame_cache[frame_id] = load_frame(root, frame_id, depth_records[frame_id], camera_records[frame_id], args)
                alignments[frame_id] = depth_alignment(scene_sample, frame_cache[frame_id], args, absolute_tolerance)
            source_mask(root, parent)
            frame = frame_cache[frame_id]
            mask = load_mask(root, mask_path, frame.depth.shape, frame)
            points = backproject(frame, mask)
            observation_id = "parent_obs_" + hashlib.sha256(json.dumps([frame_id, source_id, parent["mask_sha256"]]).encode()).hexdigest()[:40]
            lineage = {"observation_id": observation_id, "frame_id": frame_id, "source_object_id": source_id,
                       "category": category, "source_mask_path": relative(root, mask_path), "source_mask_sha256": sha256(mask_path),
                       "source_image_path": relative(root, image_path), "source_image_sha256": sha256(image_path)}
            evidence.append(lineage)
            originals[observation_id] = {**copy.deepcopy(observation), "frame_id": frame_id, "source_object_id": source_id,
                                         "parent_observation_id": observation_id}
            lifted[observation_id] = {"frame_id": frame_id, "mask": mask, "mask_path": relative(root, mask_path),
                                      "points": voxel_sample(points, voxel_size, args.max_observation_points), "valid_pixels": len(points)}
    pairs = []
    for index, first in enumerate(evidence):
        for second in evidence[index + 1:]:
            if first["category"] != second["category"] or first["frame_id"] == second["frame_id"]:
                continue
            pair = association_pair(lifted[first["observation_id"]], lifted[second["observation_id"]],
                                    frame_cache, voxel_size, absolute_tolerance, args)
            pairs.append({"id_a": first["observation_id"], "id_b": second["observation_id"],
                          "positive_match": pair["accepted"], "visible_conflict": any(item["conflict"] for item in pair["reprojection"]),
                          "geometry": pair})
    grouping = group_physical_tracks(evidence, pairs, min_views=args.min_views)
    rewritten, mapping = [], {}
    for group in grouping["tracks"]:
        object_id = group["object_id"]
        observations = [originals[name] for name in group["member_ids"]]
        rewritten.append({"object_id": object_id, "category": group["category"], "observations": observations,
                          "source_observations": copy.deepcopy(group["lineage"]), "source_descriptions": copy.deepcopy(binding),
                          "association_status": "provisional_parent_geometry_group", "geometry_validated": False,
                          "eligible_for_final_validation": group["eligible_for_final_validation"],
                          "provisional_parent_group": group["track_id"], "grouping_reasons": group["reasons"]})
        for item in group["lineage"]:
            mapping[item["frame_id"], item["source_object_id"]] = (object_id, item["observation_id"])
    rekeyed, unassigned = [], []
    for record in components:
        key = (str(record["frame_id"]), record["object_id"])
        if key not in mapping:
            unassigned.append(copy.deepcopy(record))
            continue
        object_id, observation_id = mapping[key]
        rekeyed.append({**copy.deepcopy(record), "object_id": object_id,
                       "source_object_id": key[1], "parent_observation_id": observation_id})
    grouping.update({"source_descriptions": copy.deepcopy(binding), "unassigned_component_records": unassigned,
                     "geometry_computed": True, "source_hypothesis_count": len(tracks_document["tracks"])})
    return rewritten, rekeyed, grouping


def resolve_components(root, track, observations, components, frames, scene_tree, voxel_size, absolute_tolerance, args, output_dir):
    import numpy as np
    from PIL import Image

    object_id = track["object_id"]
    parent_frames = {item["frame_id"]: item for item in observations}
    allowed = composite_policy(getattr(args, "composite_policy", {})).get(track.get("category"), {})
    evidence, lifted, records, source_masks, source_hashes = [], {}, {}, {}, {}
    skipped = []
    for record in components:
        if record.get("object_id") != object_id or record.get("component_id") == "__object__":
            continue
        frame_id = str(record["frame_id"])
        quality = record.get("quality")
        if quality is not None and not isinstance(quality, dict):
            raise LiftingError("candidate quality must be an object")
        if quality and quality.get("rejection_reasons"):
            skipped.append({"frame_id": frame_id, "component_id": record.get("component_id"),
                            "reason": "candidate_rejected_by_proposal_or_mask_quality", "source_record": record})
            continue
        if frame_id not in parent_frames:
            skipped.append({"frame_id": frame_id, "component_id": record.get("component_id"), "reason": "parent_observation_unavailable"})
            continue
        group = record.get("component_group_id", record.get("component_id"))
        if not isinstance(group, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", group):
            raise LiftingError("invalid component group")
        mask_source = source_mask(root, record)
        mask = load_mask(root, record["mask_path"], frames[frame_id].depth.shape, frames[frame_id])
        points = backproject(frames[frame_id], mask)
        observation_id = record.get("observation_id") or "obs_" + hashlib.sha256(
            json.dumps([frame_id, object_id, group, record["mask_path"]]).encode()).hexdigest()[:24]
        if observation_id in records:
            raise LiftingError("duplicate component observation identity")
        evidence.append({"observation_id": observation_id, "frame_id": frame_id,
                         "parent_object_id": object_id, "component_group_id": group})
        lifted[observation_id] = {"frame_id": frame_id, "mask": mask, "points": voxel_sample(points, voxel_size, args.max_observation_points),
                                  "valid_pixels": len(points), "mask_path": record["mask_path"]}
        records[observation_id], source_masks[observation_id] = record, mask_source
        source_hashes[observation_id] = sha256(task_path(root, record["mask_path"]))
    pairs = []
    for i, first in enumerate(evidence):
        for second in evidence[i + 1:]:
            if first["frame_id"] == second["frame_id"] or first["component_group_id"] != second["component_group_id"]:
                continue
            pair = association_pair(lifted[first["observation_id"]], lifted[second["observation_id"]], frames, voxel_size, absolute_tolerance, args)
            pairs.append({"id_a": first["observation_id"], "id_b": second["observation_id"], "positive_match": pair["accepted"],
                          "visible_conflict": any(item["conflict"] for item in pair["reprojection"]), "geometry": pair})
    grouping = group_component_tracks(evidence, pairs, min_views=args.min_views)
    resolved, additions = [], {frame_id: [] for frame_id in parent_frames}
    semantic_members = {}
    for item in evidence:
        semantic_members.setdefault(item["component_group_id"], {}).setdefault(item["frame_id"], []).append(item["observation_id"])
    semantic_groups = {}
    # Semantic unions validate a class region, independently of individual identity.
    for group, by_frame in sorted(semantic_members.items()):
        group_observations, frame_reports = [], []
        for index, (frame_id, keys) in enumerate(sorted(by_frame.items())):
            keys = sorted(keys)
            combined_source = np.zeros_like(source_masks[keys[0]])
            combined_depth = np.zeros_like(lifted[keys[0]]["mask"])
            for key in keys:
                if source_masks[key].shape != combined_source.shape:
                    raise LiftingError("semantic group source rasters disagree within one frame")
                combined_source |= source_masks[key]
                combined_depth |= lifted[key]["mask"]
            path = output_dir / object_id / "semantic_group_masks" / group / f"{index:06d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(combined_source.astype(np.uint8) * 255).save(path)
            points = backproject(frames[frame_id], combined_depth)
            group_observations.append({"frame_id": frame_id, "mask": combined_depth, "mask_path": relative(root, path),
                                       "points": voxel_sample(points, voxel_size, args.max_observation_points), "valid_pixels": len(points)})
            frame_reports.append({"frame_id": frame_id, "mask_path": relative(root, path), "mask_sha256": sha256(path),
                                  "source_pixels": int(combined_source.sum()), "member_ids": keys,
                                  "members": [{"observation_id": key, "mask_path": records[key]["mask_path"],
                                               "mask_sha256": source_hashes[key],
                                               "membership_relation": records[key].get("membership_relation"),
                                               "source_record": records[key]} for key in keys],
                                  "pixel_operation": "logical_or_of_candidate_masks_only", "parent_pixels_included": False})
        geometry = association_report(group_observations, frames, scene_tree, voxel_size, absolute_tolerance, args)
        keys = [key for values in by_frame.values() for key in values]
        relation = allowed.get(group)
        composition_allowed = bool(relation and all(records[key].get("membership_relation") == relation for key in keys))
        gate = {"parent_object_id": object_id, "component_group_id": group, "identity_scope": "semantic_component_group",
                "individual_identity_confirmed": False, "geometry_validated": False,
                "group_geometry_validated": bool(geometry["accepted"]), "geometry_report": geometry,
                "geometry_thresholds": {key: getattr(args, key) for key in
                                        ("min_views", "min_mask_pixels", "min_scene_support", "min_overlap_ratio",
                                         "overlap_radius_voxels", "max_visible_disagreement", "depth_relative_tolerance")},
                "required_membership_relation": relation, "composition_allowed": composition_allowed,
                "parent_union_allowed": bool(geometry["accepted"] and composition_allowed),
                "frame_ids": sorted(by_frame), "member_ids": sorted(keys), "frames": frame_reports,
                "generated_pixels_used": False, "part_ply_export_allowed": False}
        semantic_groups[group] = gate
        if gate["parent_union_allowed"]:
            for frame_id, keys in by_frame.items():
                additions[frame_id].extend(keys)
    for member in grouping["tracks"]:
        candidates = [lifted[key] for key in member["member_ids"]]
        geometry = association_report(candidates, frames, scene_tree, voxel_size, absolute_tolerance, args) if member["accepted"] else None
        member["geometry_report"] = geometry
        member["geometry_validated"] = bool(geometry and geometry["accepted"])
        member["identity_scope"] = "geometric_component_track"
        semantic_gate = semantic_groups[member["component_group_id"]]
        member["group_geometry_validated"] = semantic_gate["group_geometry_validated"]
        member["parent_union_allowed"] = semantic_gate["parent_union_allowed"]
        relation = allowed.get(member["component_group_id"])
        member["composition_allowed"] = bool(relation and all(records[key].get("membership_relation") == relation for key in member["member_ids"]))
        if not member["geometry_validated"]:
            continue
        for key in member["member_ids"]:
            record = records[key]
            resolved.append({**record, "component_id": member["track_id"], "source_component_id": record["component_id"],
                             "mask_sha256": source_hashes[key], "geometry_validated": True,
                             "identity_scope": "geometric_component_track", "composition_allowed": member["composition_allowed"],
                             "group_geometry_validated": semantic_gate["group_geometry_validated"],
                             "parent_union_allowed": semantic_gate["parent_union_allowed"]})
    for group, gate in semantic_groups.items():
        individuals = [member for member in grouping["tracks"] if member["component_group_id"] == group]
        confirmed = {key for member in individuals if member["geometry_validated"] for key in member["member_ids"]}
        gate["unconfirmed_individual_member_ids"] = sorted(set(gate["member_ids"]) - confirmed)
        gate["individual_tracks"] = [{key: member[key] for key in ("track_id", "member_ids", "accepted", "status", "reasons", "geometry_validated")}
                                     for member in individuals]
    composed = []
    parent_list = [record for record in components if record.get("object_id") == object_id and record.get("component_id") == "__object__"]
    parent_records = {str(record["frame_id"]): record for record in parent_list}
    if len(parent_list) != len(parent_records):
        raise LiftingError("duplicate parent candidate in one frame")
    for observation in observations:
        frame_id = observation["frame_id"]
        if frame_id not in parent_records:
            raise LiftingError("parent hypothesis has no matching segmentation candidate")
        original = parent_records[frame_id]
        quality = original.get("quality")
        if quality is not None and not isinstance(quality, dict):
            raise LiftingError("parent quality must be an object")
        if quality and quality.get("rejection_reasons"):
            raise LiftingError("parent mask did not pass proposal or mask quality")
        if task_path(root, original["mask_path"]) != task_path(root, observation["mask_path"]):
            raise LiftingError("parent candidate and identity hypothesis masks disagree")
        original_pixels = source_mask(root, original)
        combined = original_pixels.copy()
        sources = []
        for key in additions[frame_id]:
            if source_masks[key].shape != combined.shape:
                raise LiftingError("component and parent source rasters disagree")
            combined |= source_masks[key]
            sources.append({"observation_id": key, "mask_path": records[key]["mask_path"],
                            "sha256": source_hashes[key], "component_group_id": records[key].get("component_group_id", records[key]["component_id"]),
                            "membership_relation": records[key].get("membership_relation"),
                            "identity_scope": "semantic_component_group", "group_geometry_validated": True})
        path = task_path(root, original["mask_path"])
        if sources:
            path = output_dir / object_id / "composite_masks" / f"{frame_id}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(combined.astype(np.uint8) * 255).save(path)
        path_relative = relative(root, path)
        record = {**original, "mask_path": path_relative, "mask_sha256": sha256(path), "score_scope": "raw_parent_only",
                  "composition": {"raw_parent_path": original["mask_path"], "raw_parent_sha256": sha256(task_path(root, original["mask_path"])),
                                  "members": sources, "added_source_pixels": int((combined & ~original_pixels).sum()),
                                  "method": "same_camera_union_of_geometry_validated_policy_groups",
                                  "individual_identity_required_for_union": False,
                                  "generated_pixels_used": False}}
        resolved.append(record)
        mask = load_mask(root, path_relative, frames[frame_id].depth.shape, frames[frame_id])
        points = backproject(frames[frame_id], mask)
        composed.append({**observation, "mask_path": path_relative, "mask_sha256": record["mask_sha256"], "mask": mask,
                         "points": voxel_sample(points, voxel_size, args.max_observation_points), "valid_pixels": len(points)})
    grouping["skipped_observations"] = skipped
    grouping["semantic_groups"] = list(semantic_groups.values())
    grouping["parent_union_authority"] = "semantic_group_geometry_and_explicit_membership_policy"
    grouping["part_ply_authority"] = "geometrically_validated_individual_tracks_only"
    return composed, resolved, grouping


def write_gaussians(source, rows, destination: Path) -> None:
    from plyfile import PlyData, PlyElement

    element = PlyElement.describe(rows.copy(), "vertex", comments=source["vertex"].comments)
    result = PlyData([element], text=source.text, byte_order=source.byte_order, comments=source.comments, obj_info=source.obj_info)
    result.write(str(destination))
    checked = PlyData.read(str(destination))["vertex"].data
    if checked.dtype.names != rows.dtype.names or len(checked) != len(rows):
        raise LiftingError("exported PLY changed Gaussian attributes or vertex count")


def tsdf_geometry_report(path: Path) -> dict:
    import numpy as np
    from plyfile import PlyData

    if path.suffix.lower() == ".ply":
        mesh = PlyData.read(str(path))
        if "vertex" not in mesh or "face" not in mesh:
            raise LiftingError("scene_tsdf_mesh must contain surface triangles")
        vertices = np.column_stack([mesh["vertex"].data[name] for name in ("x", "y", "z")])
        face_count = len(mesh["face"].data)
    else:
        import trimesh

        scene = trimesh.load(path, force="scene")
        geometries = []
        for node in scene.graph.nodes_geometry:
            transform, name = scene.graph[node]
            mesh = scene.geometry[name].copy()
            if not isinstance(mesh, trimesh.Trimesh):
                raise LiftingError("scene_tsdf_mesh contains non-mesh geometry")
            mesh.apply_transform(transform)
            geometries.append(mesh)
        if not geometries:
            raise LiftingError("scene_tsdf_mesh contains no geometry")
        vertices = np.concatenate([mesh.vertices for mesh in geometries])
        face_count = sum(len(mesh.faces) for mesh in geometries)
    if not face_count or not len(vertices) or not np.isfinite(vertices).all():
        raise LiftingError("scene_tsdf_mesh must contain finite nonempty surface geometry")
    return {"vertices": len(vertices), "faces": face_count, "bounds": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()]}


def run(args) -> dict:
    import numpy as np
    from scipy.spatial import cKDTree

    root = args.task_dir.resolve(strict=True)
    input_path = task_path(root, args.inputs)
    output_path = task_path(root, args.outputs, exists=False)
    payload = read_json(input_path)
    if payload.get("module", "object_lifting") != "object_lifting":
        raise LiftingError("input envelope names a different module")
    inputs = payload.get("inputs", {})
    required = {"cameras", "scene_depth", "scene_gaussian_ply", "scene_tsdf_mesh", "component_mask_candidates", "physical_instance_hypotheses"}
    if not isinstance(inputs, dict) or not required <= inputs.keys():
        raise LiftingError(f"required input roles: {sorted(required)}")
    paths = {}
    for role in required:
        if inputs[role].get("evidence") == "contract_only":
            raise LiftingError(f"{role} cannot be contract-only")
        paths[role] = task_path(root, inputs[role]["path"])
    tsdf_report = tsdf_geometry_report(paths["scene_tsdf_mesh"])
    cameras_document, depth_document = read_json(paths["cameras"]), read_json(paths["scene_depth"])
    units = depth_document.get("units")
    coordinate_frame = depth_document.get("coordinate_frame", "scene_world")
    if not isinstance(units, str) or not units:
        raise LiftingError("scene_depth requires explicit units")
    if cameras_document.get("units", units) != units:
        raise LiftingError("camera and depth coordinate units disagree")
    if depth_document.get("depth_kind", "camera_z") != "camera_z":
        raise LiftingError("scene_depth must use camera_z, not Euclidean ray distance")
    camera_records, depth_records = frame_index(cameras_document, "cameras"), frame_index(depth_document, "scene_depth")
    tracks_document = read_json(paths["physical_instance_hypotheses"])
    tracks = tracks_document.get("tracks")
    components_document = read_json(paths["component_mask_candidates"])
    components = components_document.get("masks")
    if not isinstance(tracks, list) or not tracks or not isinstance(components, list):
        raise LiftingError("physical_instance_hypotheses needs tracks; component_mask_candidates needs masks lists")
    parent_mode = getattr(args, "parent_association", "provided")
    if tracks_document.get("identity_scope") == "frame_local_observations" and parent_mode != "geometry":
        raise LiftingError("frame-local parent observations require --parent-association geometry")
    policy = composite_policy(getattr(args, "composite_policy", {}))
    cloud = read_gaussian_rows(paths["scene_gaussian_ply"], error=LiftingError)
    rows, points, dropped_gaussians = cloud.rows, cloud.points, cloud.dropped
    robust_bounds = np.percentile(points, [0.5, 99.5], axis=0)
    diagonal = float(np.linalg.norm(robust_bounds[1] - robust_bounds[0]))
    if diagonal <= 0:
        raise LiftingError("scene Gaussian bounds have zero extent")
    voxel_size = args.voxel_size or diagonal / 512
    absolute_tolerance = args.depth_absolute_tolerance if args.depth_absolute_tolerance is not None else voxel_size * 2
    scene_sample = points[np.linspace(0, len(points) - 1, min(len(points), args.max_scene_samples), dtype=int)]
    scene_tree = cKDTree(scene_sample)
    frame_cache = {}
    alignments = {}
    stage_dir = task_path(root, "stages/object_lifting", exists=False)
    stage_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(tempfile.mkdtemp(prefix="run-", dir=stage_dir))
    report_path = output_dir / "lifting_report.json"
    report = {
        "schema_version": "1.0", "kind": "video2world-modeling.lifting_report", "status": "running",
        "units": units, "coordinate_frame": coordinate_frame, "scene_bounds": [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
        "voxel_size": voxel_size, "depth_absolute_tolerance": absolute_tolerance, "depth_relative_tolerance": args.depth_relative_tolerance,
        "source_gaussian_attributes": list(rows.dtype.names), "source_gaussian_count": len(points),
        "source_gaussian_rows_dropped_non_finite": dropped_gaussians,
        "source_files": {name: {"path": relative(root, path), "sha256": sha256(path)} for name, path in paths.items()},
        "tracks": [], "frame_depth_alignment": alignments, "generated_geometry_used": False,
        "tsdf_geometry": tsdf_report, "tsdf_usage": "validated surface input; Gaussian carving uses calibrated scene depth",
        "composite_policy": policy, "component_identity_method": "frame_local_observations_then_geometric_grouping_and_validation",
        "parent_association_mode": parent_mode,
    }
    if parent_mode == "geometry":
        tracks, components, grouping = resolve_parent_hypotheses(root, tracks_document, components_document,
            frame_cache, camera_records, depth_records, scene_sample, voxel_size, absolute_tolerance, args, alignments)
        grouping_path = output_dir / "parent_grouping.json"
        write_json(grouping_path, grouping)
        report["parent_grouping"] = {"path": relative(root, grouping_path), "sha256": sha256(grouping_path)}
    candidate_indices, observations_by_object = {}, {}
    resolved_components = [dict(record) for record in components if record.get("component_id") == "__object__"]
    resolved_tracks = []
    seen_ids = set()
    for track in tracks:
        object_id = track.get("object_id") if isinstance(track, dict) else None
        if not isinstance(object_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", object_id) or object_id in seen_ids:
            raise LiftingError(f"invalid or duplicate object_id: {object_id}")
        seen_ids.add(object_id)
        track_report = {"object_id": object_id, "input_association_status": track.get("association_status", "unspecified"), "accepted": False, "reasons": []}
        report["tracks"].append(track_report)
        resolved_track = {**track, "geometry_validated": False}
        resolved_tracks.append(resolved_track)
        observations = []
        try:
            if track.get("eligible_for_final_validation") is False:
                raise LiftingError("provisional parent cannot enter final validation: " + ", ".join(track.get("grouping_reasons", [])))
            records = track.get("observations")
            if not isinstance(records, list):
                raise LiftingError("track requires observations list")
            seen_frames = set()
            for observation in records:
                frame_id = str(observation["frame_id"])
                if frame_id in seen_frames:
                    raise LiftingError("track duplicates a frame")
                seen_frames.add(frame_id)
                if frame_id not in depth_records or frame_id not in camera_records:
                    raise LiftingError(f"missing camera or scene depth for frame {frame_id}")
                if frame_id not in frame_cache:
                    frame_cache[frame_id] = load_frame(root, frame_id, depth_records[frame_id], camera_records[frame_id], args)
                    alignments[frame_id] = depth_alignment(scene_sample, frame_cache[frame_id], args, absolute_tolerance)
                frame = frame_cache[frame_id]
                mask = load_mask(root, observation["mask_path"], frame.depth.shape, frame)
                lifted = backproject(frame, mask)
                observations.append({"frame_id": frame_id, "mask": mask, "mask_path": relative(root, task_path(root, observation["mask_path"])), "points": voxel_sample(lifted, voxel_size, args.max_observation_points), "valid_pixels": len(lifted)})
            observations, member_records, member_report = resolve_components(root, track, observations, components, frame_cache,
                scene_tree, voxel_size, absolute_tolerance, args, output_dir)
            track_report["component_association"] = member_report
            resolved_components = [record for record in resolved_components if record.get("object_id") != object_id] + member_records
            original_observations = {str(record["frame_id"]): record for record in records}
            resolved_track["observations"] = [{**original_observations[item["frame_id"]], "mask_path": item["mask_path"],
                                                "mask_sha256": item["mask_sha256"]} for item in observations]
            association = association_report(observations, frame_cache, scene_tree, voxel_size, absolute_tolerance, args)
            track_report.update(association)
            if any(not alignments[item["frame_id"]]["accepted"] for item in observations):
                track_report["accepted"] = False
                track_report["reasons"].append("depth_scale_or_scene_alignment_mismatch")
            if not track_report["accepted"]:
                continue
            positive = np.zeros(len(points), dtype=np.uint16)
            negative = np.zeros(len(points), dtype=np.uint16)
            occluded = np.zeros(len(points), dtype=np.uint16)
            in_front = np.zeros(len(points), dtype=np.uint16)
            for observation in observations:
                votes = surface_votes(points, frame_cache[observation["frame_id"]], observation["mask"], absolute_tolerance, args.depth_relative_tolerance)
                positive += votes["positive"]
                negative += votes["negative"]
                occluded += votes["occluded"]
                in_front += votes["in_front"]
            selected = (positive >= args.min_positive_views) & (negative == 0)
            track_report["gaussian_votes"] = {
                "positive_observations": int(positive.sum()), "negative_observations": int(negative.sum()), "occluded_observations": int(occluded.sum()), "in_front_observations": int(in_front.sum()),
                "selected_gaussians": int(selected.sum()), "rejected_visible_contradiction": int(((positive > 0) & (negative > 0)).sum()),
                "minimum_positive_views": args.min_positive_views, "negative_vote_policy": "any surface-visible negative vote vetoes carve", "occlusion_policy": "occluded and invalid depth are unknown, never positive or negative",
            }
            if int(selected.sum()) < args.min_object_gaussians:
                track_report["accepted"] = False
                track_report["reasons"].append("insufficient_multiview_gaussian_support")
            else:
                candidate_indices[object_id] = selected
                observations_by_object[object_id] = observations
        except (LiftingError, KeyError, ValueError) as error:
            track_report["accepted"] = False
            track_report["reasons"].append(str(error))
    ownership = np.zeros(len(points), dtype=np.uint16)
    for selected in candidate_indices.values():
        ownership += selected
    ambiguous = ownership > 1
    report["ambiguous_gaussians_excluded"] = int(ambiguous.sum())
    for track_report in report["tracks"]:
        object_id = track_report["object_id"]
        if object_id in candidate_indices:
            candidate_indices[object_id] &= ~ambiguous
            if int(candidate_indices[object_id].sum()) < args.min_object_gaussians:
                del candidate_indices[object_id]
                track_report["accepted"] = False
                track_report["reasons"].append("ambiguous_multi_object_gaussian_ownership")
    if not candidate_indices:
        report["status"] = "blocked_input_geometry_validation"
        report["carve_performed"] = False
        write_json(report_path, report)
        raise LiftingError(f"no geometrically accepted physical track; see {relative(root, report_path)}")
    objects = []
    removed = np.zeros(len(points), dtype=bool)
    for object_id, selected in candidate_indices.items():
        object_dir = output_dir / object_id
        object_dir.mkdir(exist_ok=True)
        object_path = object_dir / "observed_gaussians.ply"
        write_gaussians(cloud.payload, rows[selected], object_path)
        removed |= selected
        object_points = points[selected]
        part_candidates = {}
        for component in resolved_components:
            if component.get("object_id") != object_id:
                continue
            if component.get("geometry_validated") is not True or component.get("identity_scope") != "geometric_component_track":
                continue
            component_id = component.get("component_id")
            if component_id == "__object__":
                continue
            if not isinstance(component_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", component_id):
                raise LiftingError(f"invalid component_id for {object_id}")
            frame_id = str(component["frame_id"])
            if frame_id not in {item["frame_id"] for item in observations_by_object[object_id]}:
                continue
            frame = frame_cache[frame_id]
            mask = load_mask(root, component["mask_path"], frame.depth.shape, frame)
            votes = surface_votes(object_points, frame, mask, absolute_tolerance, args.depth_relative_tolerance)
            if component_id not in part_candidates:
                part_candidates[component_id] = {"positive": np.zeros(len(object_points), dtype=np.uint16), "negative": np.zeros(len(object_points), dtype=np.uint16), "frame_ids": []}
            part_candidates[component_id]["positive"] += votes["positive"]
            part_candidates[component_id]["negative"] += votes["negative"]
            part_candidates[component_id]["frame_ids"].append(frame_id)
        part_selection = {name: (votes["positive"] >= args.min_positive_views) & (votes["negative"] == 0) for name, votes in part_candidates.items()}
        part_ownership = np.zeros(len(object_points), dtype=np.uint16)
        for selection in part_selection.values():
            part_ownership += selection
        parts = []
        for name, selection in part_selection.items():
            selection &= part_ownership == 1
            if not selection.any():
                continue
            path = object_dir / f"part_{name}.ply"
            write_gaussians(cloud.payload, rows[selected][selection], path)
            metadata = next(item for item in resolved_components if item.get("object_id") == object_id and item.get("component_id") == name)
            parts.append({"component_id": name, "component_group_id": metadata.get("component_group_id", metadata.get("source_component_id")),
                          "ply_path": relative(root, path), "sha256": sha256(path), "gaussian_count": int(selection.sum()),
                          "frame_ids": part_candidates[name]["frame_ids"], "evidence": "derived_observed_surface",
                          "association_status": "geometrically_verified_observed_component", "minimum_positive_views": args.min_positive_views,
                          "composition_allowed": metadata.get("composition_allowed", False)})
        objects.append({
            "object_id": object_id, "ply_path": relative(root, object_path), "sha256": sha256(object_path), "gaussian_count": int(selected.sum()), "parts": parts,
            "coordinate_frame": coordinate_frame, "units": units, "world_bounds": [object_points.min(axis=0).tolist(), object_points.max(axis=0).tolist()],
            "observed_anchor": {"position": np.median(object_points, axis=0).tolist(), "method": "median_of_multiview_depth_supported_gaussian_centers", "orientation": "not_estimated"},
            "unassigned_component_gaussians": int((part_ownership != 1).sum()), "association_status": "geometrically_verified", "geometry_extent": "observed_only_incomplete",
            "observations": [{"frame_id": item["frame_id"], "mask_path": item["mask_path"]} for item in observations_by_object[object_id]],
        })
        source_track = next(track for track in tracks if track["object_id"] == object_id)
        if "source_observations" in source_track:
            objects[-1].update({"source_observations": copy.deepcopy(source_track["source_observations"]),
                               "source_descriptions": copy.deepcopy(source_track["source_descriptions"])})
    carved_path = output_dir / "carved_scene.ply"
    write_gaussians(cloud.payload, rows[~removed], carved_path)
    isolated_path = output_dir / "isolated_object_ply.json"
    write_json(isolated_path, {"schema_version": "1.0", "kind": "video2world-modeling.isolated_object_ply", "objects": objects, "coordinate_frame": coordinate_frame, "units": units, "scene_bounds": report["scene_bounds"]})
    track_reports = {item["object_id"]: item for item in report["tracks"]}
    for track in resolved_tracks:
        track["geometry_validated"] = bool(track_reports[track["object_id"]]["accepted"])
        track["association_status"] = "geometrically_verified" if track["geometry_validated"] else "rejected_geometry_hypothesis"
    resolved_masks_path, resolved_tracks_path = output_dir / "component_masks.json", output_dir / "physical_instance_tracks.json"
    write_json(resolved_masks_path, {"schema_version": "1.0", "frames": components_document.get("frames", []),
                                   "masks": resolved_components, "acceptance_authority": relative(root, report_path),
                                   "source_candidates_path": relative(root, paths["component_mask_candidates"])})
    write_json(resolved_tracks_path, {"schema_version": "1.0", "tracks": resolved_tracks,
                                    "acceptance_authority": relative(root, report_path), "complete_recall_claimed": False})
    report["resolved_files"] = {role: {"path": relative(root, path), "sha256": sha256(path)} for role, path in
                                (("component_masks", resolved_masks_path), ("physical_instance_tracks", resolved_tracks_path))}
    report.update({"status": "passed" if all(item["accepted"] for item in report["tracks"]) else "partial_tracks_rejected", "carve_performed": True, "removed_gaussian_count": int(removed.sum()), "remaining_gaussian_count": int((~removed).sum()), "accepted_object_ids": list(candidate_indices), "background_completion_required": True, "carved_scene_sha256": sha256(carved_path)})
    write_json(report_path, report)
    result = {"outputs": {name: {"path": relative(root, path), "evidence": "derived", "status": "candidate", "media_hint": "application/octet-stream" if path.suffix == ".ply" else "application/json", "collision_eligible": False} for name, path in (("isolated_object_ply", isolated_path), ("carved_scene_ply", carved_path), ("lifting_report", report_path), ("component_masks", resolved_masks_path), ("physical_instance_tracks", resolved_tracks_path))}}
    write_json(output_path, result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--confidence-min", type=float, default=0.5)
    parser.add_argument("--voxel-size", type=float, default=0)
    parser.add_argument("--overlap-radius-voxels", type=float, default=3)
    parser.add_argument("--min-overlap-ratio", type=float, default=0.15)
    parser.add_argument("--min-scene-support", type=float, default=0.2)
    parser.add_argument("--max-visible-disagreement", type=float, default=0.25)
    parser.add_argument("--min-views", type=int, default=2)
    parser.add_argument("--min-positive-views", type=int, default=2)
    parser.add_argument("--min-mask-pixels", type=int, default=64)
    parser.add_argument("--min-object-gaussians", type=int, default=20)
    parser.add_argument("--depth-relative-tolerance", type=float, default=0.05)
    parser.add_argument("--depth-absolute-tolerance", type=float)
    parser.add_argument("--depth-scale-tolerance", type=float, default=0.25)
    parser.add_argument("--min-depth-agreement", type=float, default=0.2)
    parser.add_argument("--max-observation-points", type=int, default=20000)
    parser.add_argument("--max-scene-samples", type=int, default=200000)
    parser.add_argument("--composite-policy", type=composite_policy, default={}, help="JSON parent-category -> component-group -> structural_part/bedding rules")
    parser.add_argument("--parent-association", choices=("provided", "geometry"), default="provided",
                        help="Build provisional same-category parents from calibrated frame-local observations before final validation")
    args = parser.parse_args(argv)
    if args.min_views < 2 or args.min_positive_views < 2 or args.min_mask_pixels < 1 or args.min_object_gaussians < 1 or args.max_observation_points < 1 or args.max_scene_samples < 1:
        parser.error("multi-view gates require at least two views and positive sample counts")
    if args.voxel_size < 0 or args.overlap_radius_voxels <= 0 or args.depth_relative_tolerance < 0 or (args.depth_absolute_tolerance is not None and args.depth_absolute_tolerance < 0):
        parser.error("geometric distances must be nonnegative")
    if any(not 0 <= value <= 1 for value in (args.min_overlap_ratio, args.min_scene_support, args.max_visible_disagreement, args.depth_scale_tolerance, args.min_depth_agreement)):
        parser.error("ratio thresholds must be in [0, 1]")
    try:
        result = run(args)
    except (LiftingError, ImportError, OSError, ValueError, KeyError) as error:
        print(f"object_lifting failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
