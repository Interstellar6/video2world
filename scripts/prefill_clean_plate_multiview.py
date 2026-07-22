#!/usr/bin/env python3
"""Prefill object-removal masks with calibrated multi-view donor reprojection.

The script forward-projects unmasked donor RGB-D pixels through world-to-camera
calibration, applies a per-donor z-buffer, robustly fuses depth-consistent donor
samples, and writes ProPainter-compatible residual masks. Existing scene pixels
outside each removal mask are copied byte-for-byte at the RGB array level.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

ASSOCIATED_DONOR_INDEX_KIND = "video2world.associated_object_donor_exclusion_index"
LEGACY_CUMULATIVE_DONOR_INDEX_KIND = "video2world.cumulative_donor_exclusion_index"
PATH_COMPONENT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
RAW_ASSOCIATED_INDEX_STATUS = "ready_for_cumulative_manifest_materialization"
MATERIALIZED_ASSOCIATED_INDEX_STATUS = "ready_for_measured_multiview_prefill"


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


def require_path_component(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or value in {".", ".."}
        or PATH_COMPONENT_PATTERN.fullmatch(value) is None
    ):
        raise ValueError(f"{label} must be a safe single path component")
    return value


def image_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def mask_bool(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) > 0


def binary_dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
    result = mask.astype(bool, copy=True)
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        result = (
            padded[1:-1, 1:-1]
            | padded[:-2, 1:-1]
            | padded[2:, 1:-1]
            | padded[1:-1, :-2]
            | padded[1:-1, 2:]
            | padded[:-2, :-2]
            | padded[:-2, 2:]
            | padded[2:, :-2]
            | padded[2:, 2:]
        )
    return result


def parse_frame_ids(value: str | None) -> list[str] | None:
    if value is None:
        return None
    tokens = [item.strip() for item in value.split(",") if item.strip()]
    if any(Path(item).name != item for item in tokens):
        raise ValueError("frame-id entries must be safe single path components")
    frame_ids = [Path(item).stem for item in tokens]
    if not frame_ids:
        raise ValueError("frame-id list is empty")
    for frame_id in frame_ids:
        require_path_component(frame_id, "frame-id entry")
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError("frame-id list contains duplicates")
    return frame_ids


def resolve_path(value: str, *, relative_to: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def resolve_frame_path(frames_dir: Path, frame_id: str) -> Path:
    require_path_component(frame_id, "RGB frame_id")
    matches = sorted(frames_dir.glob(f"{frame_id}.*"))
    if len(matches) != 1:
        raise ValueError(f"expected one RGB frame for {frame_id}, found {len(matches)}")
    return matches[0].resolve()


def resolve_source_frame_path(
    value: str,
    *,
    relative_to: Path,
    frames_dir: Path,
    frame_id: str,
    expected_sha256: str | None,
) -> Path:
    declared = resolve_path(value, relative_to=relative_to)
    if declared.is_file():
        return declared
    local = resolve_frame_path(frames_dir, frame_id)
    if expected_sha256 is not None and sha256_file(local) != expected_sha256:
        raise ValueError(f"local source RGB fallback SHA-256 mismatch for {frame_id}")
    return local


def resolve_depth_path(depth_dir: Path, frame_id: str) -> Path:
    require_path_component(frame_id, "depth frame_id")
    matches = sorted(depth_dir.glob(f"{frame_id}.*"))
    if len(matches) != 1:
        raise ValueError(f"expected one depth array for {frame_id}, found {len(matches)}")
    if matches[0].suffix not in {".npy", ".npz"}:
        raise ValueError(f"depth must be a .npy array or .npz depth archive: {matches[0]}")
    return matches[0].resolve()


def load_depth_array(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        depth = np.load(path, allow_pickle=False)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if archive.files != ["depth"]:
                raise ValueError(f"depth archive must contain exactly one 'depth' array: {path}")
            depth = archive["depth"]
    else:  # pragma: no cover - resolve_depth_path guards callers.
        raise ValueError(f"unsupported depth array format: {path}")
    if depth.ndim != 2 or not np.issubdtype(depth.dtype, np.number):
        raise ValueError(f"depth array must be a numeric 2D array: {path}")
    return depth.astype(np.float32, copy=False)


def load_camera_info(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError("camera info must be a JSON object")
    if value.get("extrinsic_type") != "world_to_camera":
        raise ValueError("camera_info.extrinsic_type must be 'world_to_camera'")
    if not isinstance(value.get("extrinsic"), dict):
        raise ValueError("camera_info.extrinsic must be a frame-keyed object")
    return value


def frame_intrinsic(camera_info: dict[str, Any], frame_id: str) -> dict[str, float]:
    intrinsics = camera_info.get("intrinsics")
    frame_camera_ids = camera_info.get("frame_camera_ids")
    value: Any = None
    if isinstance(intrinsics, dict):
        if frame_id in intrinsics:
            value = intrinsics[frame_id]
        elif isinstance(frame_camera_ids, dict):
            camera_id = str(frame_camera_ids.get(frame_id, ""))
            value = intrinsics.get(camera_id)
        elif len(intrinsics) == 1:
            value = next(iter(intrinsics.values()))
    if value is None:
        value = camera_info.get("intrinsic")
    if not isinstance(value, dict):
        raise ValueError(f"no intrinsic calibration for frame {frame_id}")
    required = ("fx", "fy", "cx", "cy", "w", "h")
    if not all(isinstance(value.get(key), int | float) for key in required):
        raise ValueError(f"intrinsic for {frame_id} is missing numeric {required}")
    return {key: float(value[key]) for key in required}


def world_to_camera(camera_info: dict[str, Any], frame_id: str) -> np.ndarray:
    value = camera_info["extrinsic"].get(frame_id)
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"invalid world-to-camera matrix for frame {frame_id}")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"invalid homogeneous row for frame {frame_id}")
    return matrix


def camera_center(matrix: np.ndarray) -> np.ndarray:
    return -(matrix[:3, :3].T @ matrix[:3, 3])


def load_mask_index(
    path: Path | None,
    *,
    label: str | None,
    min_score: float | None,
    donor_frames_dir: Path,
) -> tuple[dict[str, list[Path]], dict[str, Any]]:
    if path is None:
        return {}, {
            "item_count": 0,
            "frame_count": 0,
            "mask_sha256_verified": False,
            "source_rgb_sha256_bound": False,
        }
    value = read_json(path)
    if not isinstance(value, dict) or not isinstance(value.get("items"), list):
        raise ValueError("donor mask index must contain an items array")
    index_kind = value.get("kind")
    strict_physical_identity = index_kind == ASSOCIATED_DONOR_INDEX_KIND
    if strict_physical_identity:
        if value.get("status") != MATERIALIZED_ASSOCIATED_INDEX_STATUS:
            if value.get("status") == RAW_ASSOCIATED_INDEX_STATUS:
                raise ValueError(
                    "raw associated donor mask index requires cumulative materialization"
                )
            raise ValueError("associated donor mask index is not ready")
        gates = value.get("gates")
        required_gates = (
            "physical_identity_from_explicit_3d_anchors",
            "every_item_bound_to_exact_source_rgb_sha256",
            "every_mask_sha256_verified",
            "incomplete_frames_excluded_from_donors",
            "selected_object_ids_equal_cumulative_removed_object_union",
            "every_eligible_donor_has_every_removed_object_mask",
            "per_donor_multi_mask_union_materialized",
        )
        if not isinstance(gates, dict) or any(gates.get(key) is not True for key in required_gates):
            raise ValueError("associated donor mask index gates did not all pass")
    result: dict[str, list[Path]] = defaultdict(list)
    selected_items = 0
    masks_with_declared_sha = 0
    masks_with_source_binding = 0
    physical_identity_items = 0
    selected_object_ids: set[str] = set()
    source_sha_cache: dict[str, str] = {}
    detection_ids_by_frame: dict[str, set[str]] = defaultdict(set)
    mask_paths_by_frame: dict[str, set[Path]] = defaultdict(set)
    mask_hashes_by_frame: dict[str, set[str]] = defaultdict(set)
    for item in value["items"]:
        if not isinstance(item, dict):
            raise ValueError("donor mask index items must be objects")
        if label is not None and item.get("label") != label:
            if strict_physical_identity:
                raise ValueError("associated donor item label differs from its contract")
            continue
        score = item.get("score")
        if min_score is not None and (
            not isinstance(score, int | float) or float(score) < min_score
        ):
            continue
        image = item.get("image")
        mask_path = item.get("mask_path")
        if not isinstance(image, str) or not isinstance(mask_path, str):
            raise ValueError("donor mask items require image and mask_path strings")
        if strict_physical_identity and Path(image).name != image:
            raise ValueError("associated donor image must be a safe single path component")
        frame_id = Path(image).stem
        if strict_physical_identity:
            require_path_component(frame_id, "associated donor frame_id")
        declared_frame_id = item.get("frame_id")
        if strict_physical_identity and not isinstance(declared_frame_id, str):
            raise ValueError("associated donor item has no explicit frame_id")
        if declared_frame_id is not None and declared_frame_id != frame_id:
            raise ValueError("donor mask item frame_id does not match image")
        if strict_physical_identity:
            object_id_value = item.get("object_id")
            detection_id_value = item.get("detection_id")
            if not isinstance(object_id_value, str) or not object_id_value:
                raise ValueError("associated donor item is category-level and has no object_id")
            if not isinstance(detection_id_value, str) or not detection_id_value:
                raise ValueError("associated donor item has no physical detection identity")
            object_id = require_path_component(object_id_value, "associated donor object_id")
            detection_id = require_path_component(
                detection_id_value, "associated donor detection_id"
            )
            if detection_id in detection_ids_by_frame[frame_id]:
                raise ValueError(
                    f"associated donor physical detection is reused in {frame_id}: {detection_id}"
                )
            detection_ids_by_frame[frame_id].add(detection_id)
            if item.get("physical_identity_source") != "association_explicit_3d_anchor":
                raise ValueError("associated donor item has no explicit 3D-anchor identity")
            selected_object_ids.add(object_id)
            physical_identity_items += 1
        resolved = resolve_path(mask_path, relative_to=path.parent)
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        actual_mask_sha = sha256_file(resolved)
        declared_mask_sha = item.get("mask_sha256")
        if strict_physical_identity and not isinstance(declared_mask_sha, str):
            raise ValueError("associated donor item has no mask SHA-256")
        if declared_mask_sha is not None:
            if declared_mask_sha != actual_mask_sha:
                raise ValueError(f"donor mask SHA-256 mismatch for {frame_id}")
            masks_with_declared_sha += 1
        if strict_physical_identity:
            if (
                resolved in mask_paths_by_frame[frame_id]
                or actual_mask_sha in mask_hashes_by_frame[frame_id]
            ):
                raise ValueError(
                    f"associated donor physical mask path or SHA-256 is reused in {frame_id}"
                )
            mask_paths_by_frame[frame_id].add(resolved)
            mask_hashes_by_frame[frame_id].add(actual_mask_sha)
        source_sha = item.get("source_rgb_sha256")
        if strict_physical_identity and not isinstance(source_sha, str):
            raise ValueError("associated donor item has no source RGB SHA-256")
        if source_sha is not None:
            if not isinstance(source_sha, str):
                raise ValueError("source_rgb_sha256 must be a string")
            if frame_id not in source_sha_cache:
                source_sha_cache[frame_id] = sha256_file(
                    resolve_frame_path(donor_frames_dir, frame_id)
                )
            if source_sha_cache[frame_id] != source_sha:
                raise ValueError(f"donor mask source RGB SHA-256 mismatch for {frame_id}")
            masks_with_source_binding += 1
        result[frame_id].append(resolved)
        selected_items += 1
    contract = {
        "item_count": selected_items,
        "frame_count": len(result),
        "mask_sha256_verified": (selected_items > 0 and masks_with_declared_sha == selected_items),
        "source_rgb_sha256_bound": (
            selected_items > 0 and masks_with_source_binding == selected_items
        ),
    }
    if strict_physical_identity:
        contract.update(
            {
                "kind": index_kind,
                "physical_instance_identity_bound": (
                    selected_items > 0 and physical_identity_items == selected_items
                ),
                "selected_object_ids": sorted(selected_object_ids),
                "per_donor_mask_counts": {
                    frame_id: len(paths) for frame_id, paths in sorted(result.items())
                },
            }
        )
    return dict(result), contract


def union_masks(paths: list[Path], shape: tuple[int, int]) -> np.ndarray:
    union = np.zeros(shape, dtype=bool)
    for path in paths:
        mask = mask_bool(path)
        if mask.shape != shape:
            raise ValueError(f"mask {path} has shape {mask.shape}, expected {shape}")
        union |= mask
    return union


def load_cumulative_contract(
    *,
    input_manifest_path: Path,
    input_manifest: dict[str, Any],
    donor_mask_index_path: Path | None,
    donor_mask_label: str | None,
) -> dict[str, Any] | None:
    if input_manifest.get("kind") != "video2world.cumulative_removal_manifest":
        return None
    if input_manifest.get("status") != "ready_for_measured_multiview_prefill":
        raise ValueError("cumulative removal manifest is not ready for measured prefill")
    gates = input_manifest.get("gates")
    required_gates = (
        "strict_front_to_back_order",
        "input_is_immediately_previous_round",
        "all_previous_masks_are_accumulated",
        "all_removed_objects_are_excluded_from_donors",
        "donors_are_original_observed_rgb",
        "unresolved_residual_is_never_donor_evidence",
    )
    if not isinstance(gates, dict) or any(gates.get(key) is not True for key in required_gates):
        raise ValueError("cumulative removal manifest gates did not all pass")
    donor_contract = input_manifest.get("donor_contract")
    if not isinstance(donor_contract, dict):
        raise ValueError("cumulative removal manifest has no donor contract")
    if donor_contract.get("role") != "original_observed_rgb_only":
        raise ValueError("cumulative donor role must be original_observed_rgb_only")
    if donor_contract.get("generated_or_prefilled_rgb_as_donor") is not False:
        raise ValueError("generated or prefilled RGB is not allowed as cumulative donor evidence")
    if donor_contract.get("unresolved_residual_as_donor") is not False:
        raise ValueError("unresolved residual is not allowed as cumulative donor evidence")
    if donor_mask_index_path is None:
        raise ValueError("cumulative measured prefill requires its donor exclusion index")
    expected_index_path = resolve_path(
        str(donor_contract.get("exclusion_index")),
        relative_to=input_manifest_path.parent,
    )
    if expected_index_path != donor_mask_index_path:
        raise ValueError("donor exclusion index does not match cumulative manifest")
    expected_index_sha = donor_contract.get("exclusion_index_sha256")
    if (
        not isinstance(expected_index_sha, str)
        or sha256_file(expected_index_path) != expected_index_sha
    ):
        raise ValueError("cumulative donor exclusion index SHA-256 mismatch")
    exclusion_label = donor_contract.get("exclusion_label")
    if donor_mask_label != exclusion_label:
        raise ValueError("donor mask label does not match cumulative exclusion label")

    removed_object_ids = input_manifest.get("removed_object_ids")
    if (
        not isinstance(removed_object_ids, list)
        or not removed_object_ids
        or any(not isinstance(item, str) or not item for item in removed_object_ids)
        or len(set(removed_object_ids)) != len(removed_object_ids)
    ):
        raise ValueError("cumulative manifest removed_object_ids are invalid")
    for object_id in removed_object_ids:
        require_path_component(object_id, "cumulative removed object_id")

    records = input_manifest.get("frame_records")
    if not isinstance(records, list) or not records:
        raise ValueError("cumulative removal manifest has no frame records")
    expected_masks: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("frame_id"), str):
            raise ValueError("invalid cumulative frame record")
        frame_id = require_path_component(record["frame_id"], "cumulative target frame_id")
        if frame_id in expected_masks:
            raise ValueError("cumulative removal manifest repeats a target frame_id")
        mask_path = resolve_path(
            str(record.get("donor_exclusion_mask")),
            relative_to=input_manifest_path.parent,
        )
        expected_sha = record.get("donor_exclusion_mask_sha256")
        if not isinstance(expected_sha, str) or sha256_file(mask_path) != expected_sha:
            raise ValueError(f"cumulative donor exclusion mask mismatch for {frame_id}")
        if expected_sha != record.get("union_mask_sha256"):
            raise ValueError(f"donor exclusion is not the cumulative removal mask for {frame_id}")
        expected_masks[frame_id] = {"path": mask_path, "sha256": expected_sha}

    index = read_json(expected_index_path)
    if not isinstance(index, dict) or not isinstance(index.get("items"), list):
        raise ValueError("invalid cumulative donor exclusion index")
    index_kind = index.get("kind")
    assets = donor_contract.get("assets")
    if not isinstance(assets, dict):
        raise ValueError("cumulative donor contract assets are invalid")

    common = {
        "round_index": input_manifest.get("round_index"),
        "removed_object_ids": removed_object_ids,
        "manifest_sha256": sha256_file(input_manifest_path),
        "donor_assets": assets,
        "target_expected_masks": expected_masks,
        "lineage": input_manifest.get("lineage"),
        "index_path": expected_index_path,
        "index_sha256": expected_index_sha,
        "index_label": exclusion_label,
    }
    if index_kind in (None, LEGACY_CUMULATIVE_DONOR_INDEX_KIND):
        indexed_masks: dict[str, dict[str, Any]] = {}
        for item in index["items"]:
            if not isinstance(item, dict) or item.get("label") != exclusion_label:
                raise ValueError("cumulative donor index has an invalid item")
            frame_id = item.get("frame_id") or Path(str(item.get("image"))).stem
            if not isinstance(frame_id, str) or frame_id in indexed_masks:
                raise ValueError("cumulative donor index repeats or omits a frame id")
            mask_path = resolve_path(
                str(item.get("mask_path")), relative_to=expected_index_path.parent
            )
            mask_sha = sha256_file(mask_path)
            if item.get("mask_sha256") != mask_sha:
                raise ValueError(f"cumulative donor index mask SHA-256 mismatch for {frame_id}")
            indexed_masks[frame_id] = {"path": mask_path, "sha256": mask_sha}
        if set(indexed_masks) != set(expected_masks):
            raise ValueError("cumulative donor index and manifest cover different frames")
        for frame_id, expected in expected_masks.items():
            indexed = indexed_masks[frame_id]
            if indexed["path"] != expected["path"] or indexed["sha256"] != expected["sha256"]:
                raise ValueError(f"cumulative donor index does not match frame {frame_id}")
        if set(assets) != set(expected_masks):
            raise ValueError("cumulative original-donor assets do not cover the target frames")
        return {
            **common,
            "mode": "legacy_cumulative_target_mask",
            "eligible_donor_frame_ids": None,
            "expected_masks": expected_masks,
            "expected_union_masks": expected_masks,
            "expected_object_mask_paths": {
                frame_id: [record["path"]] for frame_id, record in expected_masks.items()
            },
        }

    if index_kind != ASSOCIATED_DONOR_INDEX_KIND:
        raise ValueError("cumulative donor exclusion index kind is unsupported")
    if donor_contract.get("exclusion_mode") != "associated_physical_instance_masks":
        raise ValueError("associated donor index lacks its cumulative exclusion mode")
    if donor_contract.get("per_donor_multi_mask_union") is not True:
        raise ValueError("associated donor contract does not require per-donor mask union")
    if index.get("status") != MATERIALIZED_ASSOCIATED_INDEX_STATUS:
        if index.get("status") == RAW_ASSOCIATED_INDEX_STATUS:
            raise ValueError("raw associated donor index requires cumulative materialization")
        raise ValueError("associated donor index status is not ready")
    associated_gates = index.get("gates")
    required_associated_gates = (
        "physical_identity_from_explicit_3d_anchors",
        "every_item_bound_to_exact_source_rgb_sha256",
        "every_mask_sha256_verified",
        "incomplete_frames_excluded_from_donors",
        "selected_object_ids_equal_cumulative_removed_object_union",
        "every_eligible_donor_has_every_removed_object_mask",
        "per_donor_multi_mask_union_materialized",
    )
    if not isinstance(associated_gates, dict) or any(
        associated_gates.get(key) is not True for key in required_associated_gates
    ):
        raise ValueError("associated donor index gates did not all pass")
    if index.get("selected_object_ids") != removed_object_ids:
        raise ValueError("associated donor object ids differ from cumulative removed objects")
    if index.get("target_removed_object_union") != removed_object_ids:
        raise ValueError("associated donor target object union is invalid")
    if donor_contract.get("physical_instance_object_ids") != removed_object_ids:
        raise ValueError("cumulative physical-instance object ids are invalid")
    if any(record.get("removed_object_ids") != removed_object_ids for record in records):
        raise ValueError("associated cumulative target frames differ from the removed-object union")
    eligible_donor_frame_ids = index.get("eligible_donor_frame_ids")
    if (
        not isinstance(eligible_donor_frame_ids, list)
        or len(eligible_donor_frame_ids) < 2
        or any(not isinstance(item, str) or not item for item in eligible_donor_frame_ids)
        or len(set(eligible_donor_frame_ids)) != len(eligible_donor_frame_ids)
    ):
        raise ValueError("associated eligible donor frame ids are invalid")
    for frame_id in eligible_donor_frame_ids:
        require_path_component(frame_id, "associated eligible donor frame_id")
    if donor_contract.get("eligible_donor_frame_ids") != eligible_donor_frame_ids:
        raise ValueError("cumulative and associated eligible donor ids differ")
    if set(assets) != set(eligible_donor_frame_ids):
        raise ValueError("associated original-donor assets do not cover eligible donors")

    expected_pairs = {
        (frame_id, object_id)
        for frame_id in eligible_donor_frame_ids
        for object_id in removed_object_ids
    }
    indexed_pairs: dict[tuple[str, str], dict[str, Any]] = {}
    paths_by_frame: dict[str, list[Path]] = defaultdict(list)
    detection_ids_by_frame: dict[str, set[str]] = defaultdict(set)
    mask_paths_by_frame: dict[str, set[Path]] = defaultdict(set)
    mask_hashes_by_frame: dict[str, set[str]] = defaultdict(set)
    for item in index["items"]:
        if not isinstance(item, dict) or item.get("label") != exclusion_label:
            raise ValueError("associated donor index has an invalid item")
        frame_id = item.get("frame_id")
        image = item.get("image")
        object_id = item.get("object_id")
        if (
            not isinstance(frame_id, str)
            or frame_id not in eligible_donor_frame_ids
            or not isinstance(image, str)
            or Path(image).name != image
            or Path(image).stem != frame_id
            or not isinstance(object_id, str)
        ):
            raise ValueError("associated donor item has invalid frame or object identity")
        require_path_component(frame_id, "associated donor item frame_id")
        require_path_component(object_id, "associated donor item object_id")
        pair = (frame_id, object_id)
        if pair in indexed_pairs:
            raise ValueError(f"associated donor index repeats {frame_id}/{object_id}")
        if pair not in expected_pairs:
            raise ValueError("associated donor item is outside the removed-object union")
        if not isinstance(item.get("detection_id"), str) or not item["detection_id"]:
            raise ValueError("associated donor item has no physical detection identity")
        detection_id = require_path_component(
            item["detection_id"], "associated donor item detection_id"
        )
        if detection_id in detection_ids_by_frame[frame_id]:
            raise ValueError(
                f"associated donor physical detection is reused in {frame_id}: {detection_id}"
            )
        detection_ids_by_frame[frame_id].add(detection_id)
        if item.get("physical_identity_source") != "association_explicit_3d_anchor":
            raise ValueError(
                "associated donor item is category-level rather than physical-instance"
            )
        mask_sha = item.get("mask_sha256")
        source_sha = item.get("source_rgb_sha256")
        if not isinstance(mask_sha, str):
            raise ValueError("associated donor item has no mask SHA-256")
        if not isinstance(source_sha, str):
            raise ValueError("associated donor item has no source RGB SHA-256")
        asset = assets.get(frame_id)
        if (
            not isinstance(asset, dict)
            or asset.get("role") != "original_observed_rgb_only"
            or asset.get("sha256") != source_sha
        ):
            raise ValueError(f"associated donor source contract mismatch for {frame_id}")
        mask_path = resolve_path(str(item.get("mask_path")), relative_to=expected_index_path.parent)
        if sha256_file(mask_path) != mask_sha:
            raise ValueError(f"associated donor mask SHA-256 mismatch for {frame_id}/{object_id}")
        if mask_path in mask_paths_by_frame[frame_id] or mask_sha in mask_hashes_by_frame[frame_id]:
            raise ValueError(
                f"associated donor physical mask path or SHA-256 is reused in {frame_id}"
            )
        mask_paths_by_frame[frame_id].add(mask_path)
        mask_hashes_by_frame[frame_id].add(mask_sha)
        mask_pixels = item.get("mask_pixels")
        if not isinstance(mask_pixels, int) or int(mask_bool(mask_path).sum()) != mask_pixels:
            raise ValueError(f"associated donor mask pixel mismatch for {frame_id}/{object_id}")
        if not isinstance(item.get("score"), int | float):
            raise ValueError(f"associated donor mask score is missing for {frame_id}/{object_id}")
        indexed_pairs[pair] = {"path": mask_path, "sha256": mask_sha}
        paths_by_frame[frame_id].append(mask_path)
    if set(indexed_pairs) != expected_pairs:
        raise ValueError("associated donor masks do not cover every removed physical instance")

    union_records = index.get("donor_unions")
    if not isinstance(union_records, dict) or set(union_records) != set(eligible_donor_frame_ids):
        raise ValueError("associated donor unions do not cover eligible donors")
    expected_unions: dict[str, dict[str, Any]] = {}
    expected_object_paths: dict[str, list[Path]] = {}
    for frame_id in eligible_donor_frame_ids:
        record = union_records.get(frame_id)
        if not isinstance(record, dict):
            raise ValueError(f"associated donor union is invalid for {frame_id}")
        if record.get("object_ids") != removed_object_ids:
            raise ValueError(f"associated donor union object ids differ for {frame_id}")
        if record.get("component_mask_count") != len(removed_object_ids):
            raise ValueError(f"associated donor union component count differs for {frame_id}")
        union_path = resolve_path(str(record.get("path")), relative_to=expected_index_path.parent)
        union_sha = record.get("sha256")
        if not isinstance(union_sha, str) or sha256_file(union_path) != union_sha:
            raise ValueError(f"associated donor union SHA-256 mismatch for {frame_id}")
        ordered_paths = [
            indexed_pairs[(frame_id, object_id)]["path"] for object_id in removed_object_ids
        ]
        actual_union = union_masks(ordered_paths, mask_bool(union_path).shape)
        declared_union = mask_bool(union_path)
        if not np.array_equal(actual_union, declared_union):
            raise ValueError(f"associated donor union differs from its object masks for {frame_id}")
        if record.get("pixels") != int(declared_union.sum()):
            raise ValueError(f"associated donor union pixel count mismatch for {frame_id}")
        expected_unions[frame_id] = {"path": union_path, "sha256": union_sha}
        expected_object_paths[frame_id] = ordered_paths

    return {
        **common,
        "mode": "associated_physical_instance_masks",
        "eligible_donor_frame_ids": eligible_donor_frame_ids,
        "expected_masks": expected_unions,
        "expected_union_masks": expected_unions,
        "expected_object_mask_paths": expected_object_paths,
    }


def donor_projection(
    *,
    donor_rgb: np.ndarray,
    donor_depth: np.ndarray,
    donor_exclusion: np.ndarray,
    donor_intrinsic: dict[str, float],
    donor_world_to_camera: np.ndarray,
    target_intrinsic: dict[str, float],
    target_world_to_camera: np.ndarray,
    target_width: int,
    target_height: int,
    sample_stride: int,
    splat_radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = donor_depth.shape
    y_grid, x_grid = np.mgrid[0:height:sample_stride, 0:width:sample_stride]
    x = x_grid.reshape(-1)
    y = y_grid.reshape(-1)
    depth = donor_depth[y, x].astype(np.float64, copy=False)
    valid = np.isfinite(depth) & (depth > 0) & ~donor_exclusion[y, x]
    if not np.any(valid):
        empty_int = np.empty(0, dtype=np.int64)
        return empty_int, np.empty(0, dtype=np.float64), np.empty((0, 3), dtype=np.uint8)

    x = x[valid]
    y = y[valid]
    depth = depth[valid]
    colors = donor_rgb[y, x]
    camera_points = np.stack(
        [
            (x - donor_intrinsic["cx"]) * depth / donor_intrinsic["fx"],
            (y - donor_intrinsic["cy"]) * depth / donor_intrinsic["fy"],
            depth,
        ],
        axis=1,
    )
    donor_rotation = donor_world_to_camera[:3, :3]
    donor_translation = donor_world_to_camera[:3, 3]
    world_points = (camera_points - donor_translation) @ donor_rotation
    target_camera = world_points @ target_world_to_camera[:3, :3].T + target_world_to_camera[:3, 3]
    target_depth = target_camera[:, 2]
    valid = np.isfinite(target_depth) & (target_depth > 1e-6)
    if not np.any(valid):
        empty_int = np.empty(0, dtype=np.int64)
        return empty_int, np.empty(0, dtype=np.float64), np.empty((0, 3), dtype=np.uint8)
    target_camera = target_camera[valid]
    target_depth = target_depth[valid]
    colors = colors[valid]
    projected_x = np.rint(
        target_intrinsic["fx"] * target_camera[:, 0] / target_depth + target_intrinsic["cx"]
    ).astype(np.int64)
    projected_y = np.rint(
        target_intrinsic["fy"] * target_camera[:, 1] / target_depth + target_intrinsic["cy"]
    ).astype(np.int64)

    flat_parts: list[np.ndarray] = []
    depth_parts: list[np.ndarray] = []
    color_parts: list[np.ndarray] = []
    for offset_y in range(-splat_radius, splat_radius + 1):
        for offset_x in range(-splat_radius, splat_radius + 1):
            target_x = projected_x + offset_x
            target_y = projected_y + offset_y
            inside = (
                (target_x >= 0)
                & (target_x < target_width)
                & (target_y >= 0)
                & (target_y < target_height)
            )
            if not np.any(inside):
                continue
            flat_parts.append(target_y[inside] * target_width + target_x[inside])
            depth_parts.append(target_depth[inside])
            color_parts.append(colors[inside])
    if not flat_parts:
        empty_int = np.empty(0, dtype=np.int64)
        return empty_int, np.empty(0, dtype=np.float64), np.empty((0, 3), dtype=np.uint8)

    flat = np.concatenate(flat_parts)
    projected_depth = np.concatenate(depth_parts)
    projected_color = np.concatenate(color_parts)
    order = np.lexsort((projected_depth, flat))
    flat = flat[order]
    projected_depth = projected_depth[order]
    projected_color = projected_color[order]
    first = np.empty(len(flat), dtype=bool)
    first[0] = True
    first[1:] = flat[1:] != flat[:-1]
    return flat[first], projected_depth[first], projected_color[first]


def robust_fuse(
    depths: np.ndarray,
    colors: np.ndarray,
    *,
    absolute_depth_tolerance: float,
    relative_depth_tolerance: float,
    min_support: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    median_depth = np.ma.median(np.ma.masked_invalid(depths), axis=0).filled(np.nan)
    tolerance = np.maximum(
        absolute_depth_tolerance,
        np.abs(median_depth) * relative_depth_tolerance,
    )
    inlier = np.isfinite(depths) & (np.abs(depths - median_depth[None, :]) <= tolerance[None, :])
    support = inlier.sum(axis=0).astype(np.uint16)
    accepted = support >= min_support
    fused_color = np.zeros((depths.shape[1], 3), dtype=np.uint8)
    for channel in range(3):
        values = np.ma.masked_where(~inlier, colors[:, :, channel])
        median_color = np.ma.median(values, axis=0).filled(0.0)
        fused_color[:, channel] = np.clip(np.rint(median_color), 0, 255).astype(np.uint8)
    fused_depth = (
        np.ma.median(np.ma.masked_where(~inlier, depths), axis=0).filled(np.nan).astype(np.float32)
    )
    fused_depth[~accepted] = np.nan
    return fused_color, fused_depth, support, accepted


def make_contact_sheet(records: list[dict[str, Any]], path: Path, samples: int) -> None:
    sample_count = min(samples, len(records))
    if sample_count <= 0:
        raise ValueError("contact sheet needs at least one record")
    indices = np.linspace(0, len(records) - 1, sample_count, dtype=int)
    cell_width, cell_height, label_height = 360, 203, 24
    sheet = Image.new(
        "RGB",
        (cell_width * 4, (cell_height + label_height) * sample_count),
        "#111111",
    )
    draw = ImageDraw.Draw(sheet)
    for row, index in enumerate(indices):
        record = records[int(index)]
        source = Image.open(record["source_frame"]).convert("RGB")
        output = Image.open(record["prefill_frame"]).convert("RGB")
        removal = Image.open(record["removal_mask"]).convert("L")
        residual = Image.open(record["residual_mask"]).convert("L")
        removal_overlay = source.copy()
        removal_overlay.paste(Image.new("RGB", source.size, (255, 0, 170)), mask=removal)
        residual_overlay = output.copy()
        residual_overlay.paste(Image.new("RGB", output.size, (255, 80, 0)), mask=residual)
        images = (source, removal_overlay, output, residual_overlay)
        titles = (
            f"source {record['frame_id']}",
            "removal mask",
            f"donor prefill {record['coverage_fraction']:.1%}",
            "unresolved residual",
        )
        y = row * (cell_height + label_height)
        for column, (image, title) in enumerate(zip(images, titles, strict=True)):
            image.thumbnail((cell_width, cell_height), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (cell_width, cell_height), "#202020")
            offset = (
                (cell_width - image.width) // 2,
                (cell_height - image.height) // 2,
            )
            canvas.paste(image, offset)
            x = column * cell_width
            sheet.paste(canvas, (x, y + label_height))
            draw.text((x + 8, y + 6), title, fill="#f4f4f4")
        for image in images:
            image.close()
        removal.close()
        residual.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def write_output_receipt(output_root: Path) -> dict[str, Any]:
    report_path = output_root / "multiview_prefill_report.json"
    contact_sheet_path = output_root / "multiview_prefill_contact_sheet.png"
    report = read_json(report_path)
    if not isinstance(report, dict) or not isinstance(report.get("frame_records"), list):
        raise ValueError("cannot receipt an invalid multi-view prefill report")
    provenance = report.get("pixel_provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("generated_pixels") != 0
        or provenance.get("propainter_pixels") != 0
    ):
        raise ValueError(
            "cannot receipt measured prefill without zero generated and ProPainter pixels"
        )
    receipt = {
        "schema_version": 2,
        "kind": "video2world.multiview_prefill_receipt",
        "report": report_path.name,
        "report_sha256": sha256_file(report_path),
        "contact_sheet": contact_sheet_path.name,
        "contact_sheet_sha256": sha256_file(contact_sheet_path),
        "frame_count": len(report["frame_records"]),
        "frame_artifacts_are_hashed_in_report": True,
        "fused_metric_depth_artifacts_are_hashed_in_report": True,
        "measured_provenance_excludes_generated_pixels": True,
        "measured_provenance_excludes_propainter_pixels": True,
    }
    write_json(output_root / "multiview_prefill_receipt.json", receipt)
    return receipt


def _build_prefill_in_place(args: argparse.Namespace) -> dict[str, Any]:
    input_manifest_path = args.input_manifest.expanduser().resolve()
    camera_info_path = args.camera_info.expanduser().resolve()
    donor_frames_dir = args.donor_frames_dir.expanduser().resolve()
    depth_dir = args.depth_dir.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    donor_mask_index_path = (
        args.donor_mask_index.expanduser().resolve() if args.donor_mask_index else None
    )
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_root}")
    if (
        args.sample_stride < 1
        or args.splat_radius < 0
        or args.min_support < 1
        or args.donor_mask_dilation < 0
    ):
        raise ValueError(
            "sample_stride/min_support must be positive and "
            "splat_radius/donor_mask_dilation non-negative"
        )
    if args.absolute_depth_tolerance < 0 or args.relative_depth_tolerance < 0:
        raise ValueError("depth tolerances must be non-negative")
    boundary_guard_extra_dilation = int(getattr(args, "boundary_guard_extra_dilation", 4))
    minimum_boundary_guard_retention = float(
        getattr(args, "minimum_boundary_guard_retention", 0.25)
    )
    if boundary_guard_extra_dilation < 0:
        raise ValueError("boundary guard extra dilation must be non-negative")
    if not 0.0 <= minimum_boundary_guard_retention <= 1.0:
        raise ValueError("minimum boundary guard retention must be in [0, 1]")

    input_manifest = read_json(input_manifest_path)
    if not isinstance(input_manifest, dict) or not isinstance(
        input_manifest.get("frame_records"), list
    ):
        raise ValueError("input manifest must contain frame_records")
    cumulative_contract = load_cumulative_contract(
        input_manifest_path=input_manifest_path,
        input_manifest=input_manifest,
        donor_mask_index_path=donor_mask_index_path,
        donor_mask_label=args.donor_mask_label,
    )
    camera_info = load_camera_info(camera_info_path)
    requested_targets = parse_frame_ids(args.target_frame_ids)
    target_records = [
        record
        for record in input_manifest["frame_records"]
        if requested_targets is None or str(record.get("frame_id")) in requested_targets
    ]
    selected_target_ids = {str(record.get("frame_id")) for record in target_records}
    if requested_targets is not None and selected_target_ids != set(requested_targets):
        raise ValueError("one or more requested target frames are absent from the input manifest")
    if not target_records:
        raise ValueError("no target records selected")

    donor_masks, donor_mask_index_contract = load_mask_index(
        donor_mask_index_path,
        label=args.donor_mask_label,
        min_score=args.donor_mask_min_score,
        donor_frames_dir=donor_frames_dir,
    )
    if donor_mask_index_contract.get("kind") == ASSOCIATED_DONOR_INDEX_KIND and (
        cumulative_contract is None
        or cumulative_contract["mode"] != "associated_physical_instance_masks"
    ):
        raise ValueError(
            "materialized associated donor index requires its cumulative removal contract"
        )
    requested_donor_ids = parse_frame_ids(args.donor_frame_ids)
    if (
        cumulative_contract is not None
        and cumulative_contract["mode"] == "associated_physical_instance_masks"
    ):
        if args.donor_mask_min_score is not None:
            raise ValueError(
                "associated physical-instance donor masks cannot be category-score filtered"
            )
        eligible_donor_ids = cumulative_contract["eligible_donor_frame_ids"]
        if requested_donor_ids is not None and requested_donor_ids != eligible_donor_ids:
            raise ValueError("requested donor ids must exactly equal associated eligible donor ids")
        donor_ids = list(eligible_donor_ids)
    else:
        donor_ids = requested_donor_ids
        if donor_ids is None:
            donor_ids = sorted(camera_info["extrinsic"])
    associated_mode = (
        cumulative_contract is not None
        and cumulative_contract["mode"] == "associated_physical_instance_masks"
    )
    donor_assets: dict[str, dict[str, Any]] = {}
    donor_cache: dict[
        str,
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ] = {}
    for frame_id in donor_ids:
        frame_path = resolve_frame_path(donor_frames_dir, frame_id)
        depth_path = resolve_depth_path(depth_dir, frame_id)
        rgb = image_rgb(frame_path)
        depth = load_depth_array(depth_path)
        if depth.shape != rgb.shape[:2]:
            raise ValueError(f"depth/RGB shape mismatch for donor {frame_id}")
        raw_exclusion = union_masks(donor_masks.get(frame_id, []), depth.shape)
        exclusion = binary_dilate(raw_exclusion, args.donor_mask_dilation)
        guard_exclusion = binary_dilate(
            raw_exclusion,
            args.donor_mask_dilation + (boundary_guard_extra_dilation if associated_mode else 0),
        )
        frame_sha = sha256_file(frame_path)
        if cumulative_contract is not None:
            expected_asset = cumulative_contract["donor_assets"].get(frame_id)
            if not isinstance(expected_asset, dict) or expected_asset.get("role") != (
                "original_observed_rgb_only"
            ):
                raise ValueError(f"cumulative donor {frame_id} has no original-RGB contract")
            if expected_asset.get("sha256") != frame_sha:
                raise ValueError(f"donor {frame_id} is not the contracted original observed RGB")
            expected_mask = cumulative_contract["expected_masks"][frame_id]
            actual_masks = donor_masks.get(frame_id, [])
            if cumulative_contract["mode"] == "legacy_cumulative_target_mask":
                if len(actual_masks) != 1 or actual_masks[0] != expected_mask["path"]:
                    raise ValueError(
                        f"donor {frame_id} is not paired with its cumulative exclusion mask"
                    )
            else:
                expected_paths = cumulative_contract["expected_object_mask_paths"][frame_id]
                if actual_masks != expected_paths:
                    raise ValueError(
                        f"donor {frame_id} does not carry every contracted physical-instance mask"
                    )
                expected_union = mask_bool(expected_mask["path"])
                if not np.array_equal(raw_exclusion, expected_union):
                    raise ValueError(
                        f"donor {frame_id} per-object mask union differs from its contract"
                    )
        donor_cache[frame_id] = (rgb, depth, exclusion, guard_exclusion)
        donor_assets[frame_id] = {
            "frame": str(frame_path),
            "frame_sha256": frame_sha,
            "depth": str(depth_path),
            "depth_sha256": sha256_file(depth_path),
            "depth_role": "reprojection_evidence_only",
            "exclusion_masks": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in donor_masks.get(frame_id, [])
            ],
            "excluded_pixels_after_dilation": int(exclusion.sum()),
            "boundary_guard_excluded_pixels": int(guard_exclusion.sum()),
        }

    frame_output_dir = output_root / "frames"
    residual_output_dir = output_root / "masks"
    support_output_dir = output_root / "support"
    depth_output_dir = output_root / "depth"
    frame_output_dir.mkdir(parents=True, exist_ok=True)
    residual_output_dir.mkdir(parents=True, exist_ok=True)
    support_output_dir.mkdir(parents=True, exist_ok=True)
    depth_output_dir.mkdir(parents=True, exist_ok=True)
    output_records: list[dict[str, Any]] = []

    for output_index, input_record in enumerate(target_records):
        frame_id = str(input_record["frame_id"])
        expected_source_sha = input_record.get("source_frame_sha256")
        if expected_source_sha is not None and not isinstance(expected_source_sha, str):
            raise ValueError(f"source_frame_sha256 must be a string for target {frame_id}")
        source_path = resolve_source_frame_path(
            str(input_record["source_frame"]),
            relative_to=input_manifest_path.parent,
            frames_dir=donor_frames_dir,
            frame_id=frame_id,
            expected_sha256=expected_source_sha,
        )
        mask_path = resolve_path(
            str(input_record["union_mask"]), relative_to=input_manifest_path.parent
        )
        source = image_rgb(source_path)
        removal_mask = mask_bool(mask_path)
        height, width = source.shape[:2]
        if removal_mask.shape != (height, width):
            raise ValueError(f"source/removal-mask shape mismatch for target {frame_id}")
        target_intrinsic = frame_intrinsic(camera_info, frame_id)
        if (int(target_intrinsic["w"]), int(target_intrinsic["h"])) != (width, height):
            raise ValueError(f"camera/image shape mismatch for target {frame_id}")
        target_w2c = world_to_camera(camera_info, frame_id)
        mask_flat = np.flatnonzero(removal_mask.reshape(-1))
        pixel_to_mask = np.full(height * width, -1, dtype=np.int64)
        pixel_to_mask[mask_flat] = np.arange(len(mask_flat), dtype=np.int64)
        candidate_depths = np.full((len(donor_ids), len(mask_flat)), np.nan, dtype=np.float32)
        candidate_colors = np.full((len(donor_ids), len(mask_flat), 3), np.nan, dtype=np.float32)
        guard_candidate_depths = (
            np.full((len(donor_ids), len(mask_flat)), np.nan, dtype=np.float32)
            if associated_mode
            else None
        )
        guard_candidate_colors = (
            np.full((len(donor_ids), len(mask_flat), 3), np.nan, dtype=np.float32)
            if associated_mode
            else None
        )
        target_center = camera_center(target_w2c)
        used_donors: list[dict[str, Any]] = []

        for donor_index, donor_id in enumerate(donor_ids):
            if args.exclude_same_frame and donor_id == frame_id:
                continue
            donor_rgb, donor_depth, donor_exclusion, guard_exclusion = donor_cache[donor_id]
            donor_intrinsic = frame_intrinsic(camera_info, donor_id)
            donor_w2c = world_to_camera(camera_info, donor_id)
            flat, projected_depth, projected_color = donor_projection(
                donor_rgb=donor_rgb,
                donor_depth=donor_depth,
                donor_exclusion=donor_exclusion,
                donor_intrinsic=donor_intrinsic,
                donor_world_to_camera=donor_w2c,
                target_intrinsic=target_intrinsic,
                target_world_to_camera=target_w2c,
                target_width=width,
                target_height=height,
                sample_stride=args.sample_stride,
                splat_radius=args.splat_radius,
            )
            indices = pixel_to_mask[flat]
            inside_removal = indices >= 0
            indices = indices[inside_removal]
            if len(indices):
                candidate_depths[donor_index, indices] = projected_depth[inside_removal]
                candidate_colors[donor_index, indices] = projected_color[inside_removal]
            guard_projected_pixels: int | None = None
            if associated_mode:
                assert guard_candidate_depths is not None
                assert guard_candidate_colors is not None
                guard_flat, guard_depth, guard_color = donor_projection(
                    donor_rgb=donor_rgb,
                    donor_depth=donor_depth,
                    donor_exclusion=guard_exclusion,
                    donor_intrinsic=donor_intrinsic,
                    donor_world_to_camera=donor_w2c,
                    target_intrinsic=target_intrinsic,
                    target_world_to_camera=target_w2c,
                    target_width=width,
                    target_height=height,
                    sample_stride=args.sample_stride,
                    splat_radius=args.splat_radius,
                )
                guard_indices = pixel_to_mask[guard_flat]
                guard_inside_removal = guard_indices >= 0
                guard_indices = guard_indices[guard_inside_removal]
                if len(guard_indices):
                    guard_candidate_depths[donor_index, guard_indices] = guard_depth[
                        guard_inside_removal
                    ]
                    guard_candidate_colors[donor_index, guard_indices] = guard_color[
                        guard_inside_removal
                    ]
                guard_projected_pixels = len(np.unique(guard_indices))
            used_donors.append(
                {
                    "frame_id": donor_id,
                    "camera_baseline": float(
                        np.linalg.norm(camera_center(donor_w2c) - target_center)
                    ),
                    "projected_mask_pixels": len(np.unique(indices)),
                    "boundary_guard_projected_mask_pixels": guard_projected_pixels,
                    "excluded_object_pixels": int(donor_exclusion.sum()),
                    "boundary_guard_excluded_object_pixels": int(guard_exclusion.sum()),
                }
            )

        fused_color, fused_depth, support, configured_accepted = robust_fuse(
            candidate_depths,
            candidate_colors,
            absolute_depth_tolerance=args.absolute_depth_tolerance,
            relative_depth_tolerance=args.relative_depth_tolerance,
            min_support=args.min_support,
        )
        boundary_concentration = {
            "evaluated": False,
            "configured_dilation_px": args.donor_mask_dilation,
            "guard_dilation_px": args.donor_mask_dilation,
            "configured_supported_pixels": int(configured_accepted.sum()),
            "guard_supported_pixels": int(configured_accepted.sum()),
            "guard_stable_intersection_pixels": int(configured_accepted.sum()),
            "guard_retention_fraction": 1.0,
            "minimum_guard_retention_fraction": minimum_boundary_guard_retention,
            "passed": True,
            "final_measured_pixels_are_guard_stable_only": False,
        }
        accepted = configured_accepted
        if associated_mode:
            assert guard_candidate_depths is not None
            assert guard_candidate_colors is not None
            guard_color, guard_depth, guard_support, guard_accepted = robust_fuse(
                guard_candidate_depths,
                guard_candidate_colors,
                absolute_depth_tolerance=args.absolute_depth_tolerance,
                relative_depth_tolerance=args.relative_depth_tolerance,
                min_support=args.min_support,
            )
            configured_pixels = int(configured_accepted.sum())
            guard_pixels = int(guard_accepted.sum())
            stable_accepted = guard_accepted & configured_accepted
            stable_pixels = int(stable_accepted.sum())
            retention = stable_pixels / configured_pixels if configured_pixels else None
            boundary_concentration = {
                "evaluated": configured_pixels > 0,
                "configured_dilation_px": args.donor_mask_dilation,
                "guard_dilation_px": (args.donor_mask_dilation + boundary_guard_extra_dilation),
                "configured_supported_pixels": configured_pixels,
                "guard_supported_pixels": guard_pixels,
                "guard_stable_intersection_pixels": stable_pixels,
                "guard_retention_fraction": retention,
                "minimum_guard_retention_fraction": minimum_boundary_guard_retention,
                "passed": (
                    configured_pixels > 0
                    and retention is not None
                    and retention >= minimum_boundary_guard_retention
                ),
                "failure_reason": (
                    "no_configured_support"
                    if configured_pixels == 0
                    else (
                        "boundary_guard_retention_below_minimum"
                        if retention is not None and retention < minimum_boundary_guard_retention
                        else None
                    )
                ),
                "final_measured_pixels_are_guard_stable_only": True,
            }
            fused_color = guard_color
            fused_depth = guard_depth
            support = guard_support
            accepted = stable_accepted
        output = source.copy()
        output_flat = output.reshape(-1, 3)
        output_flat[mask_flat[accepted]] = fused_color[accepted]
        residual_flat = np.zeros(height * width, dtype=bool)
        residual_flat[mask_flat[~accepted]] = True
        residual = residual_flat.reshape(height, width)
        support_image = np.zeros(height * width, dtype=np.uint16)
        support_image[mask_flat] = support
        support_image = support_image.reshape(height, width)
        measured_depth = np.full((height, width), np.nan, dtype=np.float32)
        measured_depth.reshape(-1)[mask_flat[accepted]] = fused_depth[accepted]
        valid_measured_depth = np.isfinite(measured_depth) & (measured_depth > 0)
        expected_measured = removal_mask & ~residual
        if not np.array_equal(valid_measured_depth, expected_measured):
            raise ValueError(f"fused measured depth support mismatch for target {frame_id}")

        name = f"{output_index:04d}.png"
        output_path = frame_output_dir / name
        residual_path = residual_output_dir / name
        support_path = support_output_dir / name
        depth_path = depth_output_dir / f"{output_index:04d}.npy"
        Image.fromarray(output).save(output_path)
        Image.fromarray(residual.astype(np.uint8) * 255).save(residual_path)
        max_support = max(int(support.max(initial=0)), 1)
        Image.fromarray(
            np.clip(np.rint(support_image * (255.0 / max_support)), 0, 255).astype(np.uint8),
        ).save(support_path)
        np.save(depth_path, measured_depth, allow_pickle=False)
        outside_exact = bool(np.array_equal(output[~removal_mask], source[~removal_mask]))
        coverage_fraction = float(accepted.mean()) if len(accepted) else 0.0
        output_records.append(
            {
                "sequence_index": output_index,
                "frame_id": frame_id,
                "source_frame": str(source_path),
                "source_frame_sha256": sha256_file(source_path),
                "removal_mask": str(mask_path),
                "removal_mask_sha256": sha256_file(mask_path),
                "removal_mask_pixels": int(removal_mask.sum()),
                "prefill_frame": str(output_path),
                "prefill_frame_sha256": sha256_file(output_path),
                "residual_mask": str(residual_path),
                "residual_mask_sha256": sha256_file(residual_path),
                "residual_mask_pixels": int(residual.sum()),
                "support_visualization": str(support_path),
                "support_visualization_sha256": sha256_file(support_path),
                "measured_depth": str(depth_path),
                "measured_depth_sha256": sha256_file(depth_path),
                "measured_depth_role": "fused_multiview_measured_depth",
                "measured_depth_unit": "meter",
                "measured_depth_semantics": "target_camera_positive_z_axis",
                "measured_depth_valid_pixels": int(valid_measured_depth.sum()),
                "measured_depth_validity_matches_measured_rgb": True,
                "covered_pixels": int(accepted.sum()),
                "coverage_fraction": coverage_fraction,
                "support_min_accepted": int(support[accepted].min()) if np.any(accepted) else None,
                "support_median_accepted": (
                    float(np.median(support[accepted])) if np.any(accepted) else None
                ),
                "support_max": int(support.max(initial=0)),
                "outside_removal_mask_rgb_exact": outside_exact,
                "donor_boundary_concentration": boundary_concentration,
                "donors": used_donors,
            }
        )

    outside_exact_passed = all(
        record["outside_removal_mask_rgb_exact"] for record in output_records
    )
    total_removal_pixels = sum(record["removal_mask_pixels"] for record in output_records)
    total_measured_pixels = sum(record["covered_pixels"] for record in output_records)
    total_unresolved_pixels = sum(record["residual_mask_pixels"] for record in output_records)
    boundary_concentration_passed = all(
        record["donor_boundary_concentration"]["passed"] for record in output_records
    )
    no_support_frame_ids = [
        record["frame_id"]
        for record in output_records
        if record["donor_boundary_concentration"]["configured_supported_pixels"] == 0
    ]
    donor_support_available_for_every_target = not no_support_frame_ids
    status = (
        "technical_failed_no_support"
        if no_support_frame_ids
        else (
            "technical_passed"
            if outside_exact_passed and boundary_concentration_passed
            else "technical_failed"
        )
    )
    next_action = donor_prefill_next_action(
        status=status,
        outside_exact_passed=outside_exact_passed,
        boundary_concentration_passed=boundary_concentration_passed,
        no_support_frame_ids=no_support_frame_ids,
        total_unresolved_pixels=total_unresolved_pixels,
        cumulative_contract_enforced=cumulative_contract is not None,
    )
    report = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 - mil8 uses Python 3.10
        "purpose": "calibrated multi-view donor prefill before residual video inpainting",
        "status": status,
        "promotion_approved": False,
        "promotion_blocker": (
            "one or more target frames have no configured donor support"
            if no_support_frame_ids
            else (
                "boundary-guard donor support validation failed"
                if not boundary_concentration_passed
                else "semantic texture continuity review is required"
            )
        ),
        "next_action": next_action,
        "input_manifest": str(input_manifest_path),
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "camera_info": str(camera_info_path),
        "camera_info_sha256": sha256_file(camera_info_path),
        "camera_extrinsic_type": "world_to_camera",
        "cumulative_removal_contract": (
            {
                "enforced": True,
                "round_index": cumulative_contract["round_index"],
                "removed_object_ids": cumulative_contract["removed_object_ids"],
                "manifest_sha256": cumulative_contract["manifest_sha256"],
                "lineage": cumulative_contract["lineage"],
                "donor_exclusion_mode": cumulative_contract["mode"],
                "eligible_donor_frame_ids": cumulative_contract["eligible_donor_frame_ids"],
                "donor_exclusion_index_kind": donor_mask_index_contract.get("kind"),
                "donors_are_original_observed_rgb": True,
                "all_removed_objects_excluded_from_donors": True,
                "per_donor_multi_mask_union": (
                    cumulative_contract["mode"] == "associated_physical_instance_masks"
                ),
                "unresolved_residual_as_donor": False,
            }
            if cumulative_contract is not None
            else {"enforced": False}
        ),
        "donor_frames_dir": str(donor_frames_dir),
        "depth_dir": str(depth_dir),
        "donor_mask_index": str(donor_mask_index_path) if donor_mask_index_path else None,
        "donor_mask_index_sha256": (
            sha256_file(donor_mask_index_path) if donor_mask_index_path else None
        ),
        "donor_mask_index_contract": donor_mask_index_contract,
        "config": {
            "target_frame_ids": [record["frame_id"] for record in output_records],
            "donor_frame_ids": donor_ids,
            "donor_mask_label": args.donor_mask_label,
            "donor_mask_min_score": args.donor_mask_min_score,
            "donor_mask_dilation": args.donor_mask_dilation,
            "boundary_guard_extra_dilation": boundary_guard_extra_dilation,
            "minimum_boundary_guard_retention": minimum_boundary_guard_retention,
            "exclude_same_frame": args.exclude_same_frame,
            "sample_stride": args.sample_stride,
            "splat_radius": args.splat_radius,
            "min_support": args.min_support,
            "absolute_depth_tolerance": args.absolute_depth_tolerance,
            "relative_depth_tolerance": args.relative_depth_tolerance,
            "fusion": "per_donor_nearest_z_then_cross_donor_median_depth_cluster",
        },
        "gates": {
            "outside_removal_mask_rgb_exact": outside_exact_passed,
            "all_residual_masks_subset_of_removal_masks": True,
            "cumulative_front_to_back_contract_enforced": cumulative_contract is not None,
            "semantic_texture_continuity": "pending_human_or_vlm_review",
            "new_depth_normal_estimation": "required_before_scene_reconstruction",
            "fused_measured_metric_depth_materialized": True,
            "donor_exclusion_masks_content_verified": donor_mask_index_contract[
                "mask_sha256_verified"
            ],
            "donor_exclusion_masks_bound_to_exact_source_rgb": donor_mask_index_contract[
                "source_rgb_sha256_bound"
            ],
            "physical_instance_donor_exclusion_enforced": (
                cumulative_contract is not None
                and cumulative_contract["mode"] == "associated_physical_instance_masks"
                and donor_mask_index_contract.get("physical_instance_identity_bound") is True
            ),
            "donor_support_available_for_every_target": (donor_support_available_for_every_target),
            "donor_support_not_boundary_concentrated": boundary_concentration_passed,
        },
        "pixel_provenance": {
            "denominator": "cumulative removal-mask pixels across selected frames",
            "cumulative_removal_pixels": total_removal_pixels,
            "measured_multiview_pixels": total_measured_pixels,
            "measured_multiview_fraction": (
                total_measured_pixels / total_removal_pixels if total_removal_pixels else 0.0
            ),
            "unresolved_unobserved_pixels": total_unresolved_pixels,
            "unresolved_unobserved_fraction": (
                total_unresolved_pixels / total_removal_pixels if total_removal_pixels else 0.0
            ),
            "generated_pixels": 0,
            "propainter_pixels": 0,
            "classes_are_mutually_exclusive": True,
            "unresolved_pixels_are_not_valid_donor_or_geometry_evidence": True,
        },
        "no_support_frame_ids": no_support_frame_ids,
        "donor_assets": donor_assets,
        "frame_records": output_records,
        "next_stage": {
            "residual_handling": (
                "keep masks/*.png as explicitly unresolved; recover with a cross-view-consistent "
                "background atlas or scene geometry before reconstruction"
                if cumulative_contract is not None
                else "run ProPainter only on masks/*.png residual holes for visual reinspection"
            ),
            "geometry": (
                "rerun depth and normal estimation on accepted clean plates before PGSR/TSDF; "
                "the fused donor depth is sparse measured evidence only and must not be treated "
                "as a dense final clean-plate depth map"
            ),
        },
        "limitations": [
            "Forward splatting cannot reveal a surface that is absent from every donor view.",
            "DA3 depth noise can misregister high-frequency bedding texture across views.",
            (
                "Technical coverage and exact outside-mask preservation do not certify "
                "semantic quality."
            ),
        ],
        "output_receipt": "multiview_prefill_receipt.json",
    }
    report_path = output_root / "multiview_prefill_report.json"
    contact_sheet_path = output_root / "multiview_prefill_contact_sheet.png"
    write_json(report_path, report)
    make_contact_sheet(
        output_records,
        contact_sheet_path,
        args.contact_sheet_samples,
    )
    write_output_receipt(output_root)
    return report


def rebase_output_paths(value: Any, *, source_root: Path, target_root: Path) -> Any:
    if isinstance(value, dict):
        return {
            key: rebase_output_paths(item, source_root=source_root, target_root=target_root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            rebase_output_paths(item, source_root=source_root, target_root=target_root)
            for item in value
        ]
    if isinstance(value, str):
        path = Path(value)
        if path.is_absolute():
            try:
                relative = path.relative_to(source_root)
            except ValueError:
                return value
            return str(target_root / relative)
    return value


def donor_prefill_next_action(
    *,
    status: str,
    outside_exact_passed: bool,
    boundary_concentration_passed: bool,
    no_support_frame_ids: list[str],
    total_unresolved_pixels: int,
    cumulative_contract_enforced: bool,
) -> dict[str, Any]:
    if not outside_exact_passed:
        action = "repair_prefill_compositor_before_any_residual_generation"
        reason = "The prefill changed pixels outside the cumulative removal mask."
        blocker = "outside_mask_rgb_changed"
    elif no_support_frame_ids:
        action = "add_observed_donor_or_switch_to_constrained_generation_for_residual"
        reason = (
            "At least one target frame has no guard-stable measured donor support, so the "
            "residual must stay unresolved or be filled by a separately reviewed generation path."
        )
        blocker = "no_guard_stable_measured_donor_support"
    elif not boundary_concentration_passed:
        action = "tighten_physical_donor_exclusion_or_add_nonboundary_donor_views"
        reason = (
            "Measured donor support exists only under the configured dilation and does not "
            "survive the boundary guard, indicating foreground-edge leakage."
        )
        blocker = "donor_support_boundary_concentrated"
    elif total_unresolved_pixels > 0:
        action = "run_constrained_residual_completion_then_semantic_cross_view_review"
        reason = (
            "Measured RGB-D donor prefill is valid but incomplete; unresolved pixels are not "
            "valid donor or geometry evidence."
        )
        blocker = "unresolved_residual_requires_reviewed_completion"
    else:
        action = "run_semantic_texture_and_new_depth_normal_review_before_next_round"
        reason = (
            "Measured donor prefill covered the removal masks, but semantic quality and new "
            "depth/normal evidence are still required before the next round."
        )
        blocker = "semantic_texture_and_depth_normal_pending"
    return {
        "action": action,
        "reason": reason,
        "blocker": blocker,
        "status": status,
        "no_support_frame_ids": no_support_frame_ids,
        "unresolved_unobserved_pixels": total_unresolved_pixels,
        "cumulative_contract_enforced": cumulative_contract_enforced,
        "promotion_approved": False,
    }


def build_prefill(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output.expanduser().resolve()
    if output_root.exists():
        if not output_root.is_dir() or any(output_root.iterdir()):
            raise FileExistsError(f"output directory is not empty: {output_root}")
        output_root.rmdir()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            dir=output_root.parent,
            prefix=f".{output_root.name}.staging-",
        )
    ).resolve()
    staged_args = argparse.Namespace(**vars(args))
    staged_args.output = staging
    try:
        staged_report = _build_prefill_in_place(staged_args)
        report = rebase_output_paths(
            staged_report,
            source_root=staging,
            target_root=output_root,
        )
        report_path = staging / "multiview_prefill_report.json"
        write_json(report_path, report)
        write_output_receipt(staging)
        staging.replace(output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--camera-info", type=Path, required=True)
    parser.add_argument("--donor-frames-dir", type=Path, required=True)
    parser.add_argument("--depth-dir", type=Path, required=True)
    parser.add_argument("--donor-mask-index", type=Path)
    parser.add_argument("--donor-mask-label", default="pillow")
    parser.add_argument("--donor-mask-min-score", type=float)
    parser.add_argument("--donor-mask-dilation", type=int, default=4)
    parser.add_argument("--boundary-guard-extra-dilation", type=int, default=4)
    parser.add_argument("--minimum-boundary-guard-retention", type=float, default=0.25)
    parser.add_argument("--target-frame-ids")
    parser.add_argument("--donor-frame-ids")
    parser.add_argument("--exclude-same-frame", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument("--splat-radius", type=int, default=1)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--absolute-depth-tolerance", type=float, default=0.08)
    parser.add_argument("--relative-depth-tolerance", type=float, default=0.01)
    parser.add_argument("--contact-sheet-samples", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = build_prefill(args)
    summary = {
        "status": report["status"],
        "promotion_approved": report["promotion_approved"],
        "frame_count": len(report["frame_records"]),
        "coverage": {
            record["frame_id"]: record["coverage_fraction"] for record in report["frame_records"]
        },
        "report": str(args.output / "multiview_prefill_report.json"),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
