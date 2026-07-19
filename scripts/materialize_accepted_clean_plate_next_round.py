#!/usr/bin/env python3
"""Materialize an accepted 25-frame clean plate as the next-round RGB source.

This is the only bridge from a generated candidate selection into a source
package that a deeper peel round may consume. It is intentionally narrower
than canonical or live promotion. Every upstream report is hash-bound and must
explicitly pass its own scope:

* candidate selection: source/mask lineage and pixel exactness;
* boundary QA: boundary alignment, seam, and local texture;
* temporal QA: all 48 directed pairs and 23 triplets;
* SAM/semantic QA: removed target absent and remaining semantics preserved;
* human visual review: all 25 frames accepted for next-round use.

Missing, pending, non-evaluable, rejected, or cross-batch evidence fails closed.
The output is built in a same-parent staging directory and exposed with one
atomic directory rename. This script never reads or writes a WorldManifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

INPUT_KIND = "video2world.accepted_clean_plate_next_round_materialization_input"
OUTPUT_KIND = "video2world.accepted_clean_plate_next_round_source_manifest"
OUTPUT_RECEIPT_KIND = "video2world.accepted_clean_plate_next_round_source_receipt"
SELECTION_KIND = "video2world.clean_plate_candidate_selection_receipt"
SELECTION_STATUS = "materialized_candidate_selection_pending_human_review"
BOUNDARY_KIND = "video2world.layered_clean_plate_boundary_qa_report"
TEMPORAL_KIND = "video2world.layered_clean_plate_temporal_qa_report"
SEMANTIC_KIND = "video2world.clean_plate_sam_semantic_qa_report"
VISUAL_REVIEW_KIND = "video2world.clean_plate_human_visual_review_receipt"
SAM_RUN_KIND = "video2world.clean_plate_sam_semantic_run_receipt"
SAM_RUN_STATUS = "technical_passed_semantic_evidence_materialized"
SCHEMA_VERSION = 1
EXPECTED_FRAME_COUNT = 25
EXPECTED_DIRECTED_PAIRS = 48
EXPECTED_TRIPLETS = 23
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
FRAME_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

BOUNDARY_GATE_SPECS = {
    "mask_iou_gte": ("alignment", "mask_iou", "minimum_mask_iou", True),
    "mask_precision_gte": ("alignment", "mask_precision", "minimum_mask_precision", True),
    "mask_recall_gte": ("alignment", "mask_recall", "minimum_mask_recall", True),
    "boundary_distance_p95_lte": (
        "alignment",
        "boundary_distance_p95_px",
        "maximum_boundary_distance_p95_px",
        False,
    ),
    "seam_color_p95_lte": (
        "seam",
        "seam_color_p95_abs_rgb_delta",
        "maximum_seam_color_p95_abs_rgb_delta",
        False,
    ),
    "seam_gradient_p95_lte": (
        "seam",
        "seam_gradient_p95_abs_rgb_delta",
        "maximum_seam_gradient_p95_abs_rgb_delta",
        False,
    ),
    "core_to_local_ring_laplacian_energy_ratio_gte": (
        "texture",
        "core_to_local_ring_laplacian_energy_ratio",
        "minimum_core_to_local_ring_laplacian_energy_ratio",
        True,
    ),
}
BOUNDARY_THRESHOLD_KEYS = {spec[2] for spec in BOUNDARY_GATE_SPECS.values()}

EXPECTED_RAFT_COMMIT = "e870e79321c31b733e2031af5aa2fb1fe3ac7eec"
EXPECTED_RAFT_CHECKPOINT_SHA256 = "fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1"
EXPECTED_RAFT_CHECKPOINT_BYTES = 21_108_000
TEMPORAL_PAIRING = {
    "mode": "adjacent_bidirectional",
    "frame_stride": 1,
    "undirected_pair_count": 24,
    "directed_pair_count": 48,
    "triplet_count": 23,
}
TEMPORAL_SAMPLING: dict[str, Any] = {
    "core_erosion_px": 4,
    "control_ring_inner_px": 8,
    "control_ring_outer_px": 32,
    "fb_consistency_absolute_px": 1.5,
    "fb_consistency_relative_to_flow": 0.01,
    "rgb_interpolation": "bilinear",
    "mask_interpolation": "nearest",
}
TEMPORAL_THRESHOLDS: dict[str, int | float] = {
    "minimum_core_valid_fraction_per_direction": 0.70,
    "minimum_control_valid_fraction_per_direction": 0.80,
    "minimum_region_pixels": 4096,
    "maximum_control_color_p95_abs_rgb_delta": 20.0,
    "maximum_core_color_p95_abs_rgb_delta": 24.0,
    "maximum_core_excess_over_control_p95_abs_rgb_delta": 8.0,
    "maximum_core_gradient_p95_abs_rgb_delta": 32.0,
    "bad_pixel_abs_rgb_delta": 24.0,
    "maximum_core_bad_pixel_fraction": 0.10,
    "minimum_triplet_valid_fraction": 0.50,
    "maximum_triplet_second_difference_p95_abs_rgb_delta": 20.0,
    "maximum_triplet_excess_over_control_p95_abs_rgb_delta": 6.0,
    "maximum_failed_directed_pairs": 0,
    "maximum_failed_triplets": 0,
}
TEMPORAL_PAIR_GATE_SPECS = {
    "core_valid_fraction_gte": (
        "coverage",
        "valid_core_fraction",
        "minimum_core_valid_fraction_per_direction",
        True,
    ),
    "control_valid_fraction_gte": (
        "coverage",
        "valid_control_fraction",
        "minimum_control_valid_fraction_per_direction",
        True,
    ),
    "control_color_p95_lte": (
        "metrics",
        "control_color_p95_abs_rgb_delta",
        "maximum_control_color_p95_abs_rgb_delta",
        False,
    ),
    "core_color_p95_lte": (
        "metrics",
        "core_color_p95_abs_rgb_delta",
        "maximum_core_color_p95_abs_rgb_delta",
        False,
    ),
    "core_excess_over_control_p95_lte": (
        "metrics",
        "core_excess_over_control_p95_abs_rgb_delta",
        "maximum_core_excess_over_control_p95_abs_rgb_delta",
        False,
    ),
    "core_gradient_p95_lte": (
        "metrics",
        "core_gradient_p95_abs_rgb_delta",
        "maximum_core_gradient_p95_abs_rgb_delta",
        False,
    ),
    "core_bad_pixel_fraction_lte": (
        "metrics",
        "core_bad_pixel_fraction",
        "maximum_core_bad_pixel_fraction",
        False,
    ),
}
TEMPORAL_TRIPLET_GATE_SPECS = {
    "triplet_valid_fraction_gte": (
        "coverage",
        "valid_core_fraction",
        "minimum_triplet_valid_fraction",
        True,
    ),
    "triplet_second_difference_p95_lte": (
        "metrics",
        "core_second_difference_p95_abs_rgb_delta",
        "maximum_triplet_second_difference_p95_abs_rgb_delta",
        False,
    ),
    "triplet_excess_over_control_p95_lte": (
        "metrics",
        "core_excess_over_control_p95_abs_rgb_delta",
        "maximum_triplet_excess_over_control_p95_abs_rgb_delta",
        False,
    ),
}
TEMPORAL_AGGREGATE_GATE_KEYS = {
    "all_directed_pairs_evaluable",
    "all_triplets_evaluable",
    "failed_directed_pairs_lte",
    "failed_triplets_lte",
}

EXPECTED_SAM_MODEL_FAMILY = "sam3"
EXPECTED_SAM_IMPLEMENTATION = "holi_spatial_sam3_holi_runner"
EXPECTED_SAM_RUNNER_SHA256 = "93b34a5fe50a8ff000b047479ca3ecea1f16f65b74e5cda6b20461120d1f3e44"
EXPECTED_SAM_CHECKPOINT_SHA256 = "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
EXPECTED_SAM_CHECKPOINT_BYTES = 3_450_062_241
EXPECTED_SAM_RUN_PRODUCER_SHA256 = (
    "9ca549becab2db94913909055aac84089e471901ec78e31934a1c7470276dcce"
)
EXPECTED_SEMANTIC_QA_PRODUCER_SHA256 = (
    "f69c82118d6b244e99b18516e041ca01930f0290d56c7cd173cb83cf00bbc96f"
)
EXPECTED_ASSOCIATION_PRODUCER_SHA256 = (
    "4a12fa9834aa58e6f31d3bc5ded731717d6c6c7e3aa18da5c0cb57cf09f50171"
)
EXPLICIT_ASSOCIATION_IDENTITY_SOURCE = "explicit --anchor object_id=PLY only"
FORBIDDEN_ASSOCIATION_IDENTITY_SOURCES = {
    "SAM3 mask filename",
    "SAM3 per-frame ordinal suffix",
    "cross-frame item order",
}


class AcceptedCleanPlateMaterializationError(ValueError):
    """Raised when next-round source authority is incomplete or inconsistent."""


@dataclass(frozen=True)
class Asset:
    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class SelectedFrame:
    sequence_index: int
    frame_id: str
    source_rgb: Asset
    composite_rgb: Asset
    core_mask: Asset
    editable_mask: Asset
    width: int
    height: int
    core_pixels: int
    editable_pixels: int


@dataclass(frozen=True)
class Selection:
    asset: Asset
    value: dict[str, Any]
    root: Path
    ordered_frame_ids: tuple[str, ...]
    frames: tuple[SelectedFrame, ...]


@dataclass(frozen=True)
class PreviousAcceptedFrame:
    sequence_index: int
    frame_id: str
    source_rgb: Asset
    cumulative_mask: Asset
    width: int
    height: int


@dataclass(frozen=True)
class PreviousAcceptedSource:
    asset: Asset
    frames: tuple[PreviousAcceptedFrame, ...]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AcceptedCleanPlateMaterializationError(message)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AcceptedCleanPlateMaterializationError(f"cannot read JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def required_sha256(value: Any, label: str) -> str:
    require(
        isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None,
        f"{label} must be a lowercase SHA-256 digest",
    )
    return value


def require_dict(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be an array")
    return value


def require_positive_int(value: Any, label: str) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value > 0,
        f"{label} must be a positive integer",
    )
    return value


def require_finite_number(value: Any, label: str) -> float:
    require(
        isinstance(value, int | float) and not isinstance(value, bool),
        f"{label} must be a finite number",
    )
    result = float(value)
    require(math.isfinite(result), f"{label} must be a finite number")
    return result


def require_exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    require(
        actual == expected,
        f"{label} keys differ: expected {sorted(expected)}, got {sorted(actual)}",
    )


def numbers_equal(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)


def validate_closed_gates(
    value: Any,
    *,
    expected: dict[str, tuple[float, float, bool]],
    label: str,
    observed_key: str = "observed",
) -> None:
    gates = require_dict(value, label)
    require_exact_keys(gates, set(expected), label)
    gate_keys = {observed_key, "threshold", "passed"}
    for gate_name, (expected_observed, expected_threshold, minimum) in expected.items():
        gate = require_dict(gates.get(gate_name), f"{label}.{gate_name}")
        require_exact_keys(gate, gate_keys, f"{label}.{gate_name}")
        observed = require_finite_number(
            gate.get(observed_key), f"{label}.{gate_name}.{observed_key}"
        )
        threshold = require_finite_number(gate.get("threshold"), f"{label}.{gate_name}.threshold")
        require(
            numbers_equal(observed, expected_observed),
            f"{label}.{gate_name}.{observed_key} does not close against report metrics",
        )
        require(
            numbers_equal(threshold, expected_threshold),
            f"{label}.{gate_name}.threshold does not close against report thresholds",
        )
        expected_pass = observed >= threshold if minimum else observed <= threshold
        require(
            gate.get("passed") is expected_pass,
            f"{label}.{gate_name}.passed does not match observed and threshold",
        )
        require(expected_pass, f"{label}.{gate_name} did not explicitly pass")


def require_nonempty_string(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value.strip()), f"{label} must be non-empty")
    return value


def require_timezone_timestamp(value: Any, label: str) -> str:
    raw = require_nonempty_string(value, label)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AcceptedCleanPlateMaterializationError(
            f"{label} must be an ISO-8601 timestamp"
        ) from exc
    require(
        parsed.tzinfo is not None and parsed.utcoffset() is not None,
        f"{label} must include timezone information",
    )
    return raw


def is_contained(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def resolve_asset(
    value: Any,
    *,
    relative_to: Path,
    label: str,
    containment_root: Path | None = None,
) -> Asset:
    record = require_dict(value, label)
    declared = require_nonempty_string(record.get("path"), f"{label}.path")
    path = Path(declared).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    path = path.resolve()
    require(path.is_file(), f"{label} does not exist: {path}")
    if containment_root is not None:
        require(
            is_contained(path, containment_root.resolve()),
            f"{label} escapes required containment root {containment_root}",
        )
    expected_sha = required_sha256(record.get("sha256"), f"{label}.sha256")
    actual_sha, actual_size = sha256_file(path)
    require(actual_sha == expected_sha, f"{label} SHA-256 mismatch")
    declared_sizes = [record[key] for key in ("size_bytes", "bytes") if key in record]
    for declared_size in declared_sizes:
        require(
            isinstance(declared_size, int)
            and not isinstance(declared_size, bool)
            and declared_size >= 0,
            f"{label} size must be a non-negative integer",
        )
        require(actual_size == declared_size, f"{label} size mismatch")
    return Asset(path=path, sha256=actual_sha, size_bytes=actual_size)


def parse_path_remaps(values: list[str]) -> list[tuple[Path, Path]]:
    remaps: list[tuple[Path, Path]] = []
    for raw in values:
        source, separator, destination = raw.partition("=")
        require(bool(separator and source and destination), f"invalid path remap: {raw}")
        source_path = Path(source).expanduser()
        destination_path = Path(destination).expanduser()
        require(source_path.is_absolute(), f"path remap source must be absolute: {raw}")
        require(destination_path.is_absolute(), f"path remap target must be absolute: {raw}")
        remaps.append((source_path, destination_path))
    return remaps


def resolve_remapped_path(
    value: str,
    *,
    relative_to: Path,
    remaps: list[tuple[Path, Path]],
) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        for source, destination in remaps:
            try:
                relative = path.relative_to(source)
            except ValueError:
                continue
            return (destination / relative).resolve()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def resolve_association_asset(
    value: Any,
    *,
    path_key: str,
    relative_to: Path,
    remaps: list[tuple[Path, Path]],
    label: str,
) -> Asset:
    record = require_dict(value, label)
    path = resolve_remapped_path(
        require_nonempty_string(record.get(path_key), f"{label}.{path_key}"),
        relative_to=relative_to,
        remaps=remaps,
    )
    normalized = {
        "path": str(path),
        "sha256": record.get("sha256"),
    }
    if "bytes" in record:
        normalized["bytes"] = record["bytes"]
    if "size_bytes" in record:
        normalized["size_bytes"] = record["size_bytes"]
    require(
        "bytes" in normalized or "size_bytes" in normalized,
        f"{label} must declare byte size",
    )
    return resolve_asset(normalized, relative_to=relative_to, label=label)


def asset_record(asset: Asset, *, path: str | None = None) -> dict[str, Any]:
    return {
        "path": path if path is not None else str(asset.path),
        "sha256": asset.sha256,
        "size_bytes": asset.size_bytes,
    }


def load_rgb(asset: Asset, label: str) -> np.ndarray:
    try:
        with Image.open(asset.path) as image:
            image.load()
            require(image.mode == "RGB", f"{label} must be RGB, found {image.mode!r}")
            return np.array(image, dtype=np.uint8, copy=True)
    except OSError as exc:
        raise AcceptedCleanPlateMaterializationError(
            f"cannot decode {label}: {asset.path}"
        ) from exc


def load_mask(asset: Asset, label: str) -> np.ndarray:
    try:
        with Image.open(asset.path) as image:
            image.load()
            require(image.mode in {"1", "L"}, f"{label} must use mode 1 or L")
            values = np.asarray(image)
    except OSError as exc:
        raise AcceptedCleanPlateMaterializationError(
            f"cannot decode {label}: {asset.path}"
        ) from exc
    require(values.ndim == 2, f"{label} must be single-channel")
    require(
        {int(value) for value in np.unique(values)} <= {0, 1, 255},
        f"{label} must be binary 0/1 or 0/255",
    )
    result = values.astype(bool)
    require(bool(result.any()), f"{label} must not be empty")
    return result


def load_label_map(asset: Asset, label: str) -> np.ndarray:
    try:
        with Image.open(asset.path) as image:
            image.load()
            require(
                image.mode in {"L", "I"} or image.mode.startswith("I;16"),
                f"{label} must be an integer single-channel image",
            )
            values = np.array(image, copy=True)
    except OSError as exc:
        raise AcceptedCleanPlateMaterializationError(
            f"cannot decode {label}: {asset.path}"
        ) from exc
    require(values.ndim == 2, f"{label} must be single-channel")
    require(np.issubdtype(values.dtype, np.integer), f"{label} must contain integer labels")
    labels = values.astype(np.int64, copy=False)
    require(not bool(np.any(labels < 0)), f"{label} must not contain negative labels")
    return labels


def parse_label_registry(value: Any, label: str) -> dict[str, int]:
    raw = require_dict(value, label)
    registry: dict[str, int] = {}
    seen_labels: set[int] = set()
    for object_id, raw_label in raw.items():
        require_nonempty_string(object_id, f"{label} object id")
        require(
            isinstance(raw_label, int)
            and not isinstance(raw_label, bool)
            and 0 < raw_label <= np.iinfo(np.int32).max,
            f"{label}.{object_id} must be a positive 32-bit integer",
        )
        require(raw_label not in seen_labels, f"{label} contains duplicate label {raw_label}")
        seen_labels.add(raw_label)
        registry[object_id] = raw_label
    return registry


def parse_frame_ids(value: Any, label: str) -> tuple[str, ...]:
    raw = require_list(value, label)
    require(len(raw) == EXPECTED_FRAME_COUNT, f"{label} must contain exactly 25 frame IDs")
    result: list[str] = []
    seen: set[str] = set()
    for index, frame_id in enumerate(raw):
        require(
            isinstance(frame_id, str) and FRAME_ID_PATTERN.fullmatch(frame_id) is not None,
            f"{label}[{index}] is unsafe",
        )
        require(frame_id not in seen, f"{label} contains duplicate {frame_id}")
        seen.add(frame_id)
        result.append(frame_id)
    return tuple(result)


def ordered_records(
    value: Any,
    *,
    key: str,
    expected_frame_ids: tuple[str, ...],
    label: str,
) -> tuple[dict[str, Any], ...]:
    raw = require_list(require_dict(value, label).get(key), f"{label}.{key}")
    require(len(raw) == EXPECTED_FRAME_COUNT, f"{label}.{key} must contain exactly 25 records")
    records: list[dict[str, Any]] = []
    for index, (raw_record, expected_id) in enumerate(zip(raw, expected_frame_ids, strict=True)):
        record = require_dict(raw_record, f"{label}.{key}[{index}]")
        require(record.get("sequence_index") == index, f"{label} sequence indexes are invalid")
        require(record.get("frame_id") == expected_id, f"{label} frame order mismatch at {index}")
        records.append(record)
    return tuple(records)


def require_all_gate_records_passed(value: Any, label: str) -> None:
    gates = require_dict(value, label)
    require(bool(gates), f"{label} must not be empty")
    for key, raw_gate in gates.items():
        gate = require_dict(raw_gate, f"{label}.{key}")
        require(gate.get("passed") is True, f"{label}.{key} did not explicitly pass")


def require_all_boolean_gates(value: Any, required: tuple[str, ...], label: str) -> None:
    gates = require_dict(value, label)
    for key in required:
        require(gates.get(key) is True, f"{label}.{key} did not explicitly pass")


def validate_selection(asset: Asset) -> Selection:
    receipt = read_json(asset.path)
    require(receipt.get("schema_version") == SCHEMA_VERSION, "selection schema is invalid")
    require(receipt.get("kind") == SELECTION_KIND, "selection kind is invalid")
    require(receipt.get("status") == SELECTION_STATUS, "selection status is not review-ready")
    require(receipt.get("promotion_allowed") is False, "selection exceeded its authority")
    require(receipt.get("human_review_required") is True, "selection must require review")
    ordered_ids = parse_frame_ids(receipt.get("ordered_frame_ids"), "selection.ordered_frame_ids")
    aggregate = require_dict(receipt.get("aggregate"), "selection.aggregate")
    require(aggregate.get("frame_count") == EXPECTED_FRAME_COUNT, "selection frame_count != 25")
    for key in (
        "all_source_rgb_lineages_match",
        "all_frame_sets_and_order_validated",
        "all_dimensions_match",
        "all_input_and_output_hashes_verified",
        "all_core_masks_are_subsets_of_editable_masks",
        "all_outputs_outside_editable_rgb_exact_recomputed",
    ):
        require(aggregate.get(key) is True, f"selection.aggregate.{key} is not true")

    selection_root = asset.path.parent.resolve()
    resolve_asset(
        receipt.get("input_manifest"),
        relative_to=selection_root,
        label="selection.input",
    )
    resolve_asset(
        receipt.get("source_batch_receipt"),
        relative_to=selection_root,
        label="selection.source_batch_receipt",
    )
    records = ordered_records(
        receipt,
        key="frames",
        expected_frame_ids=ordered_ids,
        label="selection",
    )
    frames: list[SelectedFrame] = []
    frame_set: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        frame_id = ordered_ids[index]
        prefix = f"selection.frames[{index}]"
        source = resolve_asset(
            record.get("source_rgb"), relative_to=selection_root, label=f"{prefix}.source_rgb"
        )
        resolve_asset(
            record.get("candidate_batch_receipt"),
            relative_to=selection_root,
            label=f"{prefix}.candidate_batch_receipt",
        )
        resolve_asset(
            record.get("candidate_frame_receipt"),
            relative_to=selection_root,
            label=f"{prefix}.candidate_frame_receipt",
        )
        selected_inputs = require_dict(record.get("selected_inputs"), f"{prefix}.selected_inputs")
        outputs = require_dict(record.get("outputs"), f"{prefix}.outputs")
        composite = resolve_asset(
            outputs.get("composite_rgb"),
            relative_to=selection_root,
            label=f"{prefix}.outputs.composite_rgb",
            containment_root=selection_root,
        )
        core = resolve_asset(
            outputs.get("core_mask"),
            relative_to=selection_root,
            label=f"{prefix}.outputs.core_mask",
            containment_root=selection_root,
        )
        editable = resolve_asset(
            outputs.get("editable_mask"),
            relative_to=selection_root,
            label=f"{prefix}.outputs.editable_mask",
            containment_root=selection_root,
        )
        for key, output in (
            ("composite_rgb", composite),
            ("core_mask", core),
            ("editable_mask", editable),
        ):
            selected = resolve_asset(
                selected_inputs.get(key),
                relative_to=selection_root,
                label=f"{prefix}.selected_inputs.{key}",
            )
            require(selected.sha256 == output.sha256, f"{prefix} input/output {key} mismatch")
        exactness = require_dict(record.get("exactness"), f"{prefix}.exactness")
        require(exactness.get("passed") is True, f"{prefix} exactness did not pass")
        for key in (
            "candidate_source_rgb_matches_source_batch",
            "selected_artifacts_match_candidate_batch_receipt",
            "core_is_subset_of_editable",
            "outside_editable_rgb_exact_recomputed",
        ):
            require(exactness.get(key) is True, f"{prefix}.exactness.{key} is not true")
        require(
            exactness.get("outside_editable_changed_pixels_recomputed") == 0,
            f"{prefix} changed pixels outside editable mask",
        )

        source_rgb = load_rgb(source, f"{prefix}.source_rgb")
        composite_rgb = load_rgb(composite, f"{prefix}.composite_rgb")
        core_mask = load_mask(core, f"{prefix}.core_mask")
        editable_mask = load_mask(editable, f"{prefix}.editable_mask")
        require(source_rgb.shape == composite_rgb.shape, f"{prefix} RGB dimensions differ")
        require(core_mask.shape == source_rgb.shape[:2], f"{prefix} core dimensions differ")
        require(editable_mask.shape == source_rgb.shape[:2], f"{prefix} editable dimensions differ")
        require(not bool(np.any(core_mask & ~editable_mask)), f"{prefix} core escapes editable")
        changed_outside = np.any(source_rgb != composite_rgb, axis=2) & ~editable_mask
        require(not bool(changed_outside.any()), f"{prefix} RGB changed outside editable")
        dimensions = require_dict(record.get("dimensions"), f"{prefix}.dimensions")
        width, height = int(source_rgb.shape[1]), int(source_rgb.shape[0])
        require(dimensions == {"width": width, "height": height}, f"{prefix} dimensions mismatch")
        mask_pixels = require_dict(record.get("mask_pixels"), f"{prefix}.mask_pixels")
        require(mask_pixels.get("core") == int(core_mask.sum()), f"{prefix} core pixels mismatch")
        require(
            mask_pixels.get("editable") == int(editable_mask.sum()),
            f"{prefix} editable pixels mismatch",
        )
        candidate_batch = require_dict(
            record.get("candidate_batch_receipt"), f"{prefix}.candidate_batch_receipt"
        )
        frame_set.append(
            {
                "sequence_index": index,
                "frame_id": frame_id,
                "source_rgb_sha256": source.sha256,
                "composite_rgb_sha256": composite.sha256,
                "core_mask_sha256": core.sha256,
                "editable_mask_sha256": editable.sha256,
                "candidate_batch_receipt_sha256": required_sha256(
                    candidate_batch.get("sha256"), f"{prefix}.candidate_batch_receipt.sha256"
                ),
            }
        )
        frames.append(
            SelectedFrame(
                sequence_index=index,
                frame_id=frame_id,
                source_rgb=source,
                composite_rgb=composite,
                core_mask=core,
                editable_mask=editable,
                width=width,
                height=height,
                core_pixels=int(core_mask.sum()),
                editable_pixels=int(editable_mask.sum()),
            )
        )
    require(
        aggregate.get("frame_set_sha256") == canonical_sha256(frame_set),
        "selection aggregate.frame_set_sha256 mismatch",
    )
    return Selection(
        asset=asset,
        value=receipt,
        root=selection_root,
        ordered_frame_ids=ordered_ids,
        frames=tuple(frames),
    )


def validate_previous_accepted_source(
    asset: Asset,
    *,
    selection: Selection,
    current_round_index: int,
) -> PreviousAcceptedSource:
    manifest = read_json(asset.path)
    require(manifest.get("schema_version") == SCHEMA_VERSION, "previous source schema is invalid")
    require(manifest.get("kind") == OUTPUT_KIND, "previous source kind is invalid")
    require(
        manifest.get("status") == "accepted_clean_plate_materialized_for_next_round_source",
        "previous source status is not accepted",
    )
    require(
        manifest.get("round_index") == current_round_index - 1,
        "previous source round_index is not the immediately preceding round",
    )
    require(
        manifest.get("next_round_index") == current_round_index,
        "previous source next_round_index does not feed the current round",
    )
    require(
        manifest.get("frame_count") == EXPECTED_FRAME_COUNT,
        "previous source frame_count != 25",
    )
    previous_ids = parse_frame_ids(
        manifest.get("ordered_frame_ids"), "previous source.ordered_frame_ids"
    )
    require(previous_ids == selection.ordered_frame_ids, "previous source frame order differs")
    require(manifest.get("next_round_source_approved") is True, "previous source is not approved")
    require(
        manifest.get("promotion_approved") is False,
        "previous source exceeded scoped authority",
    )
    require(
        manifest.get("canonical_or_live_manifest_modified") is False,
        "previous source claims a canonical/live mutation",
    )
    aggregate = require_dict(manifest.get("aggregate"), "previous source.aggregate")
    for key in (
        "one_to_one_frame_mapping_verified",
        "all_source_and_mask_hashes_verified",
        "all_cumulative_masks_and_contributor_labels_verified",
        "all_output_assets_contained",
        "all_boundary_frames_passed",
        "all_temporal_pairs_and_triplets_passed",
        "all_sam_semantic_frames_passed",
        "all_frames_human_reviewed_and_accepted",
    ):
        require(aggregate.get(key) is True, f"previous source.aggregate.{key} is not true")

    records = ordered_records(
        manifest,
        key="frame_records",
        expected_frame_ids=previous_ids,
        label="previous source",
    )
    root = asset.path.parent.resolve()
    frames: list[PreviousAcceptedFrame] = []
    for selected, record in zip(selection.frames, records, strict=True):
        prefix = f"previous source {selected.frame_id}"
        source = resolve_asset(
            record.get("source_rgb"),
            relative_to=root,
            label=f"{prefix}.source_rgb",
            containment_root=root,
        )
        cumulative = resolve_asset(
            record.get("previous_edited_region_mask"),
            relative_to=root,
            label=f"{prefix}.previous_edited_region_mask",
            containment_root=root,
        )
        require(
            record.get("segmentation_source_rgb_sha256") == source.sha256,
            f"{prefix} segmentation source binding is invalid",
        )
        require(
            selected.source_rgb.sha256 == source.sha256,
            f"{prefix} does not supply the current selection source RGB",
        )
        rgb = load_rgb(source, f"{prefix}.source_rgb")
        cumulative_values = load_mask(cumulative, f"{prefix}.previous_edited_region_mask")
        height, width = rgb.shape[:2]
        require(
            cumulative_values.shape == (height, width),
            f"{prefix} cumulative mask dimensions differ from source",
        )
        require(
            (width, height) == (selected.width, selected.height),
            f"{prefix} dimensions differ from the current selection",
        )
        frames.append(
            PreviousAcceptedFrame(
                sequence_index=selected.sequence_index,
                frame_id=selected.frame_id,
                source_rgb=source,
                cumulative_mask=cumulative,
                width=width,
                height=height,
            )
        )
    return PreviousAcceptedSource(asset=asset, frames=tuple(frames))


def resolve_matching_input(
    value: Any,
    *,
    report_root: Path,
    expected: Asset,
    label: str,
) -> Asset:
    actual = resolve_asset(value, relative_to=report_root, label=label)
    require(actual.sha256 == expected.sha256, f"{label} does not bind the selected asset")
    return actual


def validate_boundary(asset: Asset, selection: Selection) -> dict[str, Any]:
    report = read_json(asset.path)
    require(report.get("schema_version") == 1, "boundary schema is invalid")
    require(report.get("kind") == BOUNDARY_KIND, "boundary kind is invalid")
    require(report.get("status") == "technical_passed", "boundary QA did not pass")
    require(report.get("boundary_texture_gate_passed") is True, "boundary gate did not pass")
    require(report.get("promotion_approved") is False, "boundary QA exceeded scoped authority")
    require(
        report.get("promotion_scope") == "boundary_and_local_texture_only",
        "boundary promotion scope is invalid",
    )
    require(report.get("frame_count") == EXPECTED_FRAME_COUNT, "boundary frame_count != 25")
    raw_thresholds = require_dict(report.get("thresholds"), "boundary.thresholds")
    require_exact_keys(raw_thresholds, BOUNDARY_THRESHOLD_KEYS, "boundary.thresholds")
    thresholds = {
        key: require_finite_number(value, f"boundary.thresholds.{key}")
        for key, value in raw_thresholds.items()
    }
    for key in ("minimum_mask_iou", "minimum_mask_precision", "minimum_mask_recall"):
        require(0.0 <= thresholds[key] <= 1.0, f"boundary.thresholds.{key} must be in [0, 1]")
    for key in (
        "maximum_boundary_distance_p95_px",
        "maximum_seam_color_p95_abs_rgb_delta",
        "maximum_seam_gradient_p95_abs_rgb_delta",
        "minimum_core_to_local_ring_laplacian_energy_ratio",
    ):
        require(thresholds[key] >= 0.0, f"boundary.thresholds.{key} must be non-negative")
    records = ordered_records(
        report,
        key="frame_records",
        expected_frame_ids=selection.ordered_frame_ids,
        label="boundary",
    )
    observed_by_gate: dict[str, list[float]] = {gate_name: [] for gate_name in BOUNDARY_GATE_SPECS}
    for frame, record in zip(selection.frames, records, strict=True):
        prefix = f"boundary {frame.frame_id}"
        inputs = require_dict(record.get("inputs"), f"boundary {frame.frame_id}.inputs")
        for key, expected in (
            ("composite_rgb", frame.composite_rgb),
            ("source_rgb", frame.source_rgb),
            ("removal_core_mask", frame.core_mask),
            ("editable_mask", frame.editable_mask),
        ):
            resolve_matching_input(
                inputs.get(key),
                report_root=asset.path.parent,
                expected=expected,
                label=f"boundary {frame.frame_id}.{key}",
            )
        target_matte = resolve_asset(
            inputs.get("target_matte"),
            relative_to=asset.path.parent,
            label=f"{prefix}.target_matte",
        )
        target_values = load_mask(target_matte, f"{prefix}.target_matte")
        require(
            target_values.shape == (frame.height, frame.width),
            f"{prefix} target matte dimensions differ",
        )
        sections = {
            section: require_dict(record.get(section), f"{prefix}.{section}")
            for section in ("alignment", "seam", "texture")
        }
        expected_gates: dict[str, tuple[float, float, bool]] = {}
        for gate_name, (section, metric, threshold_name, minimum) in BOUNDARY_GATE_SPECS.items():
            observed = require_finite_number(
                sections[section].get(metric), f"{prefix}.{section}.{metric}"
            )
            observed_by_gate[gate_name].append(observed)
            expected_gates[gate_name] = (observed, thresholds[threshold_name], minimum)
        validate_closed_gates(
            record.get("gates"),
            expected=expected_gates,
            label=f"{prefix}.gates",
        )
        require(record.get("passed") is True, f"boundary frame {frame.frame_id} did not pass")

    aggregate_expected: dict[str, tuple[float, float, bool]] = {}
    for gate_name, (_section, _metric, threshold_name, minimum) in BOUNDARY_GATE_SPECS.items():
        observed_values = observed_by_gate[gate_name]
        observed_worst = min(observed_values) if minimum else max(observed_values)
        aggregate_expected[gate_name] = (
            observed_worst,
            thresholds[threshold_name],
            minimum,
        )
    validate_closed_gates(
        report.get("gates"),
        expected=aggregate_expected,
        label="boundary.gates",
        observed_key="observed_worst",
    )
    return report


def require_evidence_binding(
    value: Any,
    *,
    report_root: Path,
    expected: Asset,
    label: str,
) -> None:
    bound = resolve_asset(value, relative_to=report_root, label=label)
    require(bound.sha256 == expected.sha256, f"{label} binds a different receipt")


def validate_temporal(
    asset: Asset,
    selection: Selection,
    boundary_asset: Asset,
    *,
    round_index: int,
    previous_source: PreviousAcceptedSource | None,
) -> tuple[dict[str, Any], tuple[Asset, ...]]:
    report = read_json(asset.path)
    require(report.get("schema_version") == 1, "temporal schema is invalid")
    require(report.get("kind") == TEMPORAL_KIND, "temporal kind is invalid")
    require(
        report.get("status") == "technical_passed_temporal_only",
        "temporal QA did not pass",
    )
    require(report.get("review_only") is True, "temporal QA must remain review-only")
    require(report.get("promotion_approved") is False, "temporal QA exceeded scoped authority")
    forbidden_claims = require_dict(report.get("forbidden_claims"), "temporal.forbidden_claims")
    require(
        bool(forbidden_claims) and all(value is False for value in forbidden_claims.values()),
        "temporal QA contains a forbidden promotion or correctness claim",
    )
    pairing = require_dict(report.get("pairing"), "temporal.pairing")
    require(pairing == TEMPORAL_PAIRING, "temporal pairing contract is not the audited fixed set")
    sampling = require_dict(report.get("sampling"), "temporal.sampling")
    require_exact_keys(sampling, set(TEMPORAL_SAMPLING), "temporal.sampling")
    for key, expected_value in TEMPORAL_SAMPLING.items():
        observed_value = sampling[key]
        if isinstance(expected_value, float):
            require(
                numbers_equal(
                    require_finite_number(observed_value, f"temporal.sampling.{key}"),
                    expected_value,
                ),
                f"temporal.sampling.{key} differs from the audited value",
            )
        else:
            require(
                observed_value == expected_value,
                f"temporal.sampling.{key} differs from the audited value",
            )
    raw_thresholds = require_dict(report.get("thresholds"), "temporal.thresholds")
    require_exact_keys(raw_thresholds, set(TEMPORAL_THRESHOLDS), "temporal.thresholds")
    thresholds: dict[str, float] = {}
    for key, expected_value in TEMPORAL_THRESHOLDS.items():
        observed_value = require_finite_number(
            raw_thresholds.get(key), f"temporal.thresholds.{key}"
        )
        require(
            numbers_equal(observed_value, float(expected_value)),
            f"temporal.thresholds.{key} differs from the audited value",
        )
        thresholds[key] = observed_value

    flow_backend = require_dict(report.get("flow_backend"), "temporal.flow_backend")
    require_exact_keys(
        flow_backend,
        {
            "kind",
            "repository_path",
            "repository_commit",
            "checkpoint",
            "precision",
            "iterations",
            "flow_completion",
            "inference_resolution",
            "device",
        },
        "temporal.flow_backend",
    )
    require(flow_backend.get("kind") == "plain_raft", "temporal backend is not plain RAFT")
    require_nonempty_string(flow_backend.get("repository_path"), "temporal RAFT repository_path")
    require(
        flow_backend.get("repository_commit") == EXPECTED_RAFT_COMMIT,
        "temporal RAFT repository commit is not audited",
    )
    checkpoint = require_dict(flow_backend.get("checkpoint"), "temporal RAFT checkpoint")
    require_nonempty_string(checkpoint.get("path"), "temporal RAFT checkpoint.path")
    require(
        required_sha256(checkpoint.get("sha256"), "temporal RAFT checkpoint.sha256")
        == EXPECTED_RAFT_CHECKPOINT_SHA256,
        "temporal RAFT checkpoint is not audited",
    )
    checkpoint_bytes = require_positive_int(
        checkpoint.get("bytes", checkpoint.get("size_bytes")), "temporal RAFT checkpoint bytes"
    )
    require(
        checkpoint_bytes == EXPECTED_RAFT_CHECKPOINT_BYTES,
        "temporal RAFT checkpoint byte size is not audited",
    )
    require(flow_backend.get("precision") == "float32", "temporal RAFT precision is invalid")
    require(flow_backend.get("iterations") == 20, "temporal RAFT iterations are invalid")
    require(
        flow_backend.get("flow_completion") is False,
        "temporal RAFT flow completion must be disabled",
    )
    require(
        flow_backend.get("inference_resolution") == [1280, 720],
        "temporal RAFT inference resolution is invalid",
    )
    require(
        require_nonempty_string(flow_backend.get("device"), "temporal RAFT device").startswith(
            "cuda"
        ),
        "temporal RAFT device must be CUDA",
    )
    require(report.get("frame_count") == EXPECTED_FRAME_COUNT, "temporal frame_count != 25")
    require(report.get("round_index") == round_index, "temporal round_index mismatch")
    require(
        parse_frame_ids(report.get("ordered_frame_ids"), "temporal.ordered_frame_ids")
        == selection.ordered_frame_ids,
        "temporal frame order differs from selection",
    )
    dependencies = require_dict(report.get("dependencies"), "temporal.dependencies")
    require_evidence_binding(
        dependencies.get("selection_manifest"),
        report_root=asset.path.parent,
        expected=selection.asset,
        label="temporal.dependencies.selection_manifest",
    )
    require_evidence_binding(
        dependencies.get("upstream_boundary_report"),
        report_root=asset.path.parent,
        expected=boundary_asset,
        label="temporal.dependencies.upstream_boundary_report",
    )
    records = ordered_records(
        report,
        key="frame_records",
        expected_frame_ids=selection.ordered_frame_ids,
        label="temporal",
    )
    cumulative_masks: list[Asset] = []
    for frame, record in zip(selection.frames, records, strict=True):
        inputs = require_dict(record.get("inputs"), f"temporal {frame.frame_id}.inputs")
        for key, expected in (
            ("selected_composite_rgb", frame.composite_rgb),
            ("source_rgb", frame.source_rgb),
            ("current_core_mask", frame.core_mask),
            ("editable_mask", frame.editable_mask),
        ):
            resolve_matching_input(
                inputs.get(key),
                report_root=asset.path.parent,
                expected=expected,
                label=f"temporal {frame.frame_id}.{key}",
            )
        cumulative = resolve_asset(
            inputs.get("cumulative_core_mask"),
            relative_to=asset.path.parent,
            label=f"temporal {frame.frame_id}.cumulative_core_mask",
        )
        current_values = load_mask(frame.core_mask, f"temporal {frame.frame_id}.current_core_mask")
        cumulative_values = load_mask(cumulative, f"temporal {frame.frame_id}.cumulative_core_mask")
        require(
            cumulative_values.shape == (frame.height, frame.width),
            f"temporal {frame.frame_id} cumulative mask dimensions differ",
        )
        require(
            not bool(np.any(current_values & ~cumulative_values)),
            f"temporal {frame.frame_id} current core escapes cumulative mask",
        )
        if round_index == 1:
            require(
                np.array_equal(cumulative_values, current_values),
                f"temporal {frame.frame_id} round 1 cumulative mask must equal current core",
            )
        else:
            require(previous_source is not None, "round 2+ requires a previous accepted source")
            previous_frame = previous_source.frames[frame.sequence_index]
            previous_values = load_mask(
                previous_frame.cumulative_mask,
                f"previous source {frame.frame_id}.previous_edited_region_mask",
            )
            expected_cumulative = previous_values | current_values
            require(
                np.array_equal(cumulative_values, expected_cumulative),
                f"temporal {frame.frame_id} cumulative mask is not predecessor OR current core",
            )
        cumulative_masks.append(cumulative)
        require_evidence_binding(
            inputs.get("selection_receipt"),
            report_root=asset.path.parent,
            expected=selection.asset,
            label=f"temporal {frame.frame_id}.selection_receipt",
        )
    counts = require_dict(report.get("counts"), "temporal.counts")
    require(counts.get("expected_directed_pairs") == EXPECTED_DIRECTED_PAIRS, "bad pair contract")
    require(counts.get("observed_directed_pairs") == EXPECTED_DIRECTED_PAIRS, "missing pairs")
    require(counts.get("not_evaluable_directed_pairs") == 0, "non-evaluable temporal pairs")
    require(counts.get("failed_directed_pairs") == 0, "failed temporal pairs")
    require(counts.get("expected_triplets") == EXPECTED_TRIPLETS, "bad triplet contract")
    require(counts.get("observed_triplets") == EXPECTED_TRIPLETS, "missing triplets")
    require(counts.get("not_evaluable_triplets") == 0, "non-evaluable temporal triplets")
    require(counts.get("failed_triplets") == 0, "failed temporal triplets")
    pair_records = require_list(report.get("pair_records"), "temporal.pair_records")
    require(len(pair_records) == EXPECTED_DIRECTED_PAIRS, "temporal pair_records must contain 48")
    expected_pairs = {
        pair
        for left, right in pairwise(selection.ordered_frame_ids)
        for pair in ((left, right), (right, left))
    }
    frame_indexes = {frame_id: index for index, frame_id in enumerate(selection.ordered_frame_ids)}
    observed_pairs: set[tuple[str, str]] = set()
    for index, raw_record in enumerate(pair_records):
        record = require_dict(raw_record, f"temporal.pair_records[{index}]")
        pair = (record.get("source_frame_id"), record.get("target_frame_id"))
        require(pair not in observed_pairs, f"duplicate temporal directed pair {pair}")
        observed_pairs.add(pair)
        require(
            record.get("source_sequence_index") == frame_indexes.get(pair[0])
            and record.get("target_sequence_index") == frame_indexes.get(pair[1]),
            f"temporal pair {pair} sequence indexes are invalid",
        )
        require(record.get("evaluable") is True, f"temporal pair {pair} is not evaluable")
        sources = {
            section: require_dict(record.get(section), f"temporal pair {pair}.{section}")
            for section in ("coverage", "metrics")
        }
        expected_gates: dict[str, tuple[float, float, bool]] = {}
        for gate_name, (
            section,
            metric,
            threshold_name,
            minimum,
        ) in TEMPORAL_PAIR_GATE_SPECS.items():
            observed = require_finite_number(
                sources[section].get(metric), f"temporal pair {pair}.{section}.{metric}"
            )
            expected_gates[gate_name] = (observed, thresholds[threshold_name], minimum)
        validate_closed_gates(
            record.get("gates"),
            expected=expected_gates,
            label=f"temporal pair {pair}.gates",
        )
        require(record.get("passed") is True, f"temporal pair {pair} did not pass")
    require(observed_pairs == expected_pairs, "temporal directed pair mapping is incomplete")
    triplet_records = require_list(report.get("triplet_records"), "temporal.triplet_records")
    require(len(triplet_records) == EXPECTED_TRIPLETS, "temporal triplet_records must contain 23")
    expected_triplets = {
        tuple(selection.ordered_frame_ids[index : index + 3]) for index in range(EXPECTED_TRIPLETS)
    }
    observed_triplets: set[tuple[str, str, str]] = set()
    for index, raw_record in enumerate(triplet_records):
        record = require_dict(raw_record, f"temporal.triplet_records[{index}]")
        triplet = (
            record.get("previous_frame_id"),
            record.get("center_frame_id"),
            record.get("following_frame_id"),
        )
        require(triplet not in observed_triplets, f"duplicate temporal triplet {triplet}")
        observed_triplets.add(triplet)
        require(
            record.get("previous_sequence_index") == frame_indexes.get(triplet[0])
            and record.get("center_sequence_index") == frame_indexes.get(triplet[1])
            and record.get("following_sequence_index") == frame_indexes.get(triplet[2]),
            f"temporal triplet {triplet} sequence indexes are invalid",
        )
        require(record.get("evaluable") is True, f"temporal triplet {triplet} is not evaluable")
        sources = {
            section: require_dict(record.get(section), f"temporal triplet {triplet}.{section}")
            for section in ("coverage", "metrics")
        }
        expected_gates = {}
        for gate_name, (
            section,
            metric,
            threshold_name,
            minimum,
        ) in TEMPORAL_TRIPLET_GATE_SPECS.items():
            observed = require_finite_number(
                sources[section].get(metric), f"temporal triplet {triplet}.{section}.{metric}"
            )
            expected_gates[gate_name] = (observed, thresholds[threshold_name], minimum)
        validate_closed_gates(
            record.get("gates"),
            expected=expected_gates,
            label=f"temporal triplet {triplet}.gates",
        )
        require(record.get("passed") is True, f"temporal triplet {triplet} did not pass")
    require(observed_triplets == expected_triplets, "temporal triplet mapping is incomplete")

    failed_pairs = sum(record.get("passed") is not True for record in pair_records)
    failed_triplets = sum(record.get("passed") is not True for record in triplet_records)
    aggregate_expected = {
        "all_directed_pairs_evaluable": (
            float(sum(record.get("evaluable") is True for record in pair_records)),
            float(EXPECTED_DIRECTED_PAIRS),
            True,
        ),
        "all_triplets_evaluable": (
            float(sum(record.get("evaluable") is True for record in triplet_records)),
            float(EXPECTED_TRIPLETS),
            True,
        ),
        "failed_directed_pairs_lte": (
            float(failed_pairs),
            thresholds["maximum_failed_directed_pairs"],
            False,
        ),
        "failed_triplets_lte": (
            float(failed_triplets),
            thresholds["maximum_failed_triplets"],
            False,
        ),
    }
    require_exact_keys(
        require_dict(report.get("gates"), "temporal.gates"),
        TEMPORAL_AGGREGATE_GATE_KEYS,
        "temporal.gates",
    )
    validate_closed_gates(
        report.get("gates"),
        expected=aggregate_expected,
        label="temporal.gates",
    )
    return report, tuple(cumulative_masks)


SEMANTIC_FRAME_GATES = (
    "source_binding_verified",
    "target_object_absent",
    "no_unassigned_target_semantic_overlap",
    "remaining_object_semantics_preserved",
)
MAX_UNASSIGNED_TARGET_LABEL_CORE_OVERLAP_FRACTION = 0.05


def validate_semantic_association(
    asset: Asset,
    *,
    selection: Selection,
    source_role: str,
    strict_mask_index_sha256: str | None,
    strict_detection_hashes: dict[str, str] | None,
    path_remaps: list[tuple[Path, Path]],
) -> dict[str, Any]:
    report = read_json(asset.path)
    prefix = f"semantic {source_role} association"
    require(report.get("schema_version") == 1, f"{prefix} schema is invalid")
    require(report.get("status") == "passed", f"{prefix} status did not pass")
    execution = require_dict(report.get("execution"), f"{prefix}.execution")
    script = require_dict(execution.get("script"), f"{prefix}.execution.script")
    require(
        required_sha256(script.get("sha256"), f"{prefix} script.sha256")
        == EXPECTED_ASSOCIATION_PRODUCER_SHA256,
        f"{prefix} producer script SHA-256 is not audited",
    )
    script_asset = resolve_association_asset(
        script,
        path_key="path",
        relative_to=asset.path.parent,
        remaps=path_remaps,
        label=f"{prefix}.execution.script",
    )
    contract = require_dict(report.get("association_contract"), f"{prefix}.contract")
    require(
        contract.get("physical_identity_source") == EXPLICIT_ASSOCIATION_IDENTITY_SOURCE,
        f"{prefix} does not use explicit anchor identity",
    )
    forbidden = require_list(
        contract.get("forbidden_identity_sources"), f"{prefix}.forbidden_identity_sources"
    )
    require(
        set(forbidden) == FORBIDDEN_ASSOCIATION_IDENTITY_SOURCES,
        f"{prefix} forbidden identity sources differ",
    )
    gates = require_dict(report.get("gates"), f"{prefix}.gates")
    require(gates.get("passed") is True, f"{prefix} gates did not pass")
    thresholds = require_dict(report.get("thresholds"), f"{prefix}.thresholds")
    threshold_sha = canonical_sha256(thresholds)
    require(report.get("thresholds_sha256") == threshold_sha, f"{prefix} threshold digest mismatch")
    sources = require_dict(report.get("sources"), f"{prefix}.sources")
    require(
        report.get("source_set_sha256") == canonical_sha256(sources),
        f"{prefix} source digest mismatch",
    )

    anchors: dict[str, str] = {}
    anchor_sizes: dict[str, int] = {}
    for raw_anchor in require_list(sources.get("anchors"), f"{prefix}.anchors"):
        anchor = require_dict(raw_anchor, f"{prefix}.anchor")
        object_id = require_nonempty_string(anchor.get("object_id"), f"{prefix}.anchor.object_id")
        require(object_id not in anchors, f"{prefix} repeats anchor {object_id}")
        anchor_asset = resolve_association_asset(
            anchor,
            path_key="resolved_path",
            relative_to=asset.path.parent,
            remaps=path_remaps,
            label=f"{prefix}.anchor.{object_id}",
        )
        anchors[object_id] = anchor_asset.sha256
        anchor_sizes[object_id] = anchor_asset.size_bytes
    require(bool(anchors), f"{prefix} has no anchors")
    camera = require_dict(sources.get("camera_info"), f"{prefix}.camera_info")
    camera_asset = resolve_association_asset(
        camera,
        path_key="resolved_path",
        relative_to=asset.path.parent,
        remaps=path_remaps,
        label=f"{prefix}.camera_info",
    )
    camera_sha = camera_asset.sha256
    require(camera.get("extrinsic_type") == "world_to_camera", f"{prefix} camera is invalid")
    mask_index = require_dict(sources.get("sam3_mask_index"), f"{prefix}.mask_index")
    mask_index_asset = resolve_association_asset(
        mask_index,
        path_key="resolved_path",
        relative_to=asset.path.parent,
        remaps=path_remaps,
        label=f"{prefix}.mask_index",
    )
    mask_index_sha = mask_index_asset.sha256
    if strict_mask_index_sha256 is not None:
        require(
            mask_index_sha == strict_mask_index_sha256,
            "candidate association mask index differs",
        )

    expected_sources = {
        frame.frame_id: frame.composite_rgb if source_role == "candidate" else frame.source_rgb
        for frame in selection.frames
    }
    source_records = require_list(sources.get("frames"), f"{prefix}.frames")
    require(len(source_records) == EXPECTED_FRAME_COUNT, f"{prefix} must contain 25 sources")
    source_hashes: dict[str, str] = {}
    for raw_source in source_records:
        source = require_dict(raw_source, f"{prefix}.source")
        frame_id = require_nonempty_string(source.get("frame_id"), f"{prefix}.source.frame_id")
        require(frame_id in expected_sources, f"{prefix} source has unknown frame {frame_id}")
        require(frame_id not in source_hashes, f"{prefix} repeats source frame {frame_id}")
        source_asset = resolve_association_asset(
            source,
            path_key="resolved_path",
            relative_to=asset.path.parent,
            remaps=path_remaps,
            label=f"{prefix}.source.{frame_id}",
        )
        require(
            source_asset.sha256 == expected_sources[frame_id].sha256,
            f"{prefix} source RGB differs for {frame_id}",
        )
        source_hashes[frame_id] = source_asset.sha256
    require(set(source_hashes) == set(expected_sources), f"{prefix} source frame set differs")

    masks: dict[str, dict[str, Any]] = {}
    masks_by_frame: dict[str, set[str]] = {frame.frame_id: set() for frame in selection.frames}
    mask_paths_by_frame: dict[str, set[Path]] = {
        frame.frame_id: set() for frame in selection.frames
    }
    mask_sha256_by_frame: dict[str, set[str]] = {
        frame.frame_id: set() for frame in selection.frames
    }
    frame_dimensions = {frame.frame_id: (frame.height, frame.width) for frame in selection.frames}
    for raw_mask in require_list(sources.get("masks"), f"{prefix}.masks"):
        mask_record = require_dict(raw_mask, f"{prefix}.mask")
        item_index = mask_record.get("item_index")
        require(
            isinstance(item_index, int) and not isinstance(item_index, bool) and item_index >= 0,
            f"{prefix} mask item_index is invalid",
        )
        frame_id = require_nonempty_string(mask_record.get("frame_id"), f"{prefix}.mask.frame_id")
        detection_id = f"{frame_id}:mask:{item_index:06d}"
        require(
            mask_record.get("detection_id") == detection_id,
            f"{prefix} detection ID is not frame/item-index exact",
        )
        require(frame_id in frame_dimensions, f"{prefix} mask has unknown frame")
        require(detection_id not in masks, f"{prefix} repeats detection {detection_id}")
        mask_asset = resolve_association_asset(
            mask_record,
            path_key="resolved_path",
            relative_to=asset.path.parent,
            remaps=path_remaps,
            label=f"{prefix}.mask.{detection_id}",
        )
        require(
            mask_asset.path not in mask_paths_by_frame[frame_id],
            f"{prefix} reuses a mask path in frame {frame_id}",
        )
        require(
            mask_asset.sha256 not in mask_sha256_by_frame[frame_id],
            f"{prefix} reuses mask bytes in frame {frame_id}",
        )
        mask_paths_by_frame[frame_id].add(mask_asset.path)
        mask_sha256_by_frame[frame_id].add(mask_asset.sha256)
        values = load_mask(mask_asset, f"{prefix}.mask.{detection_id}")
        require(
            values.shape == frame_dimensions[frame_id],
            f"{prefix} mask dimensions differ for {detection_id}",
        )
        require(
            mask_record.get("area_pixels") == int(values.sum()),
            f"{prefix} mask area differs for {detection_id}",
        )
        label = require_nonempty_string(mask_record.get("label"), f"{prefix}.mask.label")
        masks[detection_id] = {
            "frame_id": frame_id,
            "label": label,
            "sha256": mask_asset.sha256,
            "values": values,
        }
        masks_by_frame[frame_id].add(detection_id)
    require(
        mask_index.get("loaded_detection_count") == len(masks),
        f"{prefix} mask index loaded_detection_count mismatch",
    )
    if strict_detection_hashes is not None:
        require(set(masks) == set(strict_detection_hashes), "candidate detection set differs")
        for detection_id, expected_sha in strict_detection_hashes.items():
            require(masks[detection_id]["sha256"] == expected_sha, f"{detection_id} mask differs")

    raw_frames = require_list(report.get("frames"), f"{prefix}.frame_reports")
    require(len(raw_frames) == EXPECTED_FRAME_COUNT, f"{prefix} must contain 25 frame reports")
    frame_map = {
        record.get("frame_id"): record for record in raw_frames if isinstance(record, dict)
    }
    require(set(frame_map) == set(expected_sources), f"{prefix} frame report set differs")
    frames: dict[str, dict[str, Any]] = {}
    for selected in selection.frames:
        frame_id = selected.frame_id
        frame_record = require_dict(frame_map.get(frame_id), f"{prefix}.frame.{frame_id}")
        frame_camera = require_dict(frame_record.get("camera"), f"{prefix}.frame.camera")
        require(
            frame_camera.get("extrinsic_type") == "world_to_camera",
            f"{prefix} frame camera is invalid",
        )
        assignments: dict[str, str] = {}
        assigned_ids: set[str] = set()
        assignment_labels: dict[str, str] = {}
        for raw_assignment in require_list(
            frame_record.get("assignments"), f"{prefix}.frame.assignments"
        ):
            assignment = require_dict(raw_assignment, f"{prefix}.assignment")
            object_id = require_nonempty_string(
                assignment.get("object_id"), f"{prefix}.assignment.object_id"
            )
            require(object_id in anchors, f"{prefix} assignment object has no anchor")
            require(object_id not in assignments, f"{prefix} repeats assignment {object_id}")
            detection_id = require_nonempty_string(
                assignment.get("detection_id"), f"{prefix}.assignment.detection_id"
            )
            require(detection_id in masks_by_frame[frame_id], f"{prefix} assignment mask differs")
            require(detection_id not in assigned_ids, f"{prefix} reuses a detection")
            require(assignment.get("frame_id") == frame_id, f"{prefix} assignment frame differs")
            require(
                assignment.get("frame_sha256") == source_hashes[frame_id],
                f"{prefix} assignment source hash differs",
            )
            require(
                assignment.get("mask_sha256") == masks[detection_id]["sha256"],
                f"{prefix} assignment mask hash differs",
            )
            require(
                assignment.get("label") == masks[detection_id]["label"],
                f"{prefix} assignment label differs",
            )
            metrics = require_dict(assignment.get("metrics"), f"{prefix}.assignment.metrics")
            require(metrics.get("detection_id") == detection_id, f"{prefix} metric ID differs")
            require(metrics.get("eligible") is True, f"{prefix} assignment is not eligible")
            metric_gates = require_dict(metrics.get("gates"), f"{prefix}.metric.gates")
            require(
                bool(metric_gates) and all(value is True for value in metric_gates.values()),
                f"{prefix} assignment metric gates failed",
            )
            assignments[object_id] = detection_id
            assignment_labels[object_id] = str(assignment["label"])
            assigned_ids.add(detection_id)
        unassigned = tuple(
            require_nonempty_string(item, f"{prefix}.unassigned detection")
            for item in require_list(
                frame_record.get("unassigned_detection_ids"), f"{prefix}.unassigned"
            )
        )
        require(len(set(unassigned)) == len(unassigned), f"{prefix} repeats unassigned ID")
        require(set(unassigned).isdisjoint(assigned_ids), f"{prefix} assigned/unassigned overlap")
        require(
            set(unassigned) | assigned_ids == masks_by_frame[frame_id],
            f"{prefix} detections do not close to assigned/unassigned",
        )
        frames[frame_id] = {
            "camera_sha256": canonical_sha256(frame_camera),
            "assignments": assignments,
            "assignment_labels": assignment_labels,
            "unassigned": unassigned,
        }
    return {
        "anchors": anchors,
        "anchor_size_bytes": anchor_sizes,
        "producer_script_size_bytes": script_asset.size_bytes,
        "camera_info_sha256": camera_sha,
        "camera_info_size_bytes": camera_asset.size_bytes,
        "thresholds_sha256": threshold_sha,
        "mask_index_sha256": mask_index_sha,
        "masks": masks,
        "frames": frames,
    }


def validate_sam_run_receipt(
    asset: Asset,
    *,
    selection: Selection,
    target_object_id: str,
    round_index: int,
) -> tuple[dict[str, Any], Asset, dict[str, str]]:
    receipt = read_json(asset.path)
    require(receipt.get("schema_version") == SCHEMA_VERSION, "SAM run schema is invalid")
    require(receipt.get("kind") == SAM_RUN_KIND, "SAM run kind is invalid")
    require(receipt.get("status") == SAM_RUN_STATUS, "SAM run status is not passed")
    require(receipt.get("review_only") is True, "SAM run must remain review-only")
    require(receipt.get("promotion_approved") is False, "SAM run exceeded scoped authority")
    require(receipt.get("round_index") == round_index, "SAM run round_index mismatch")
    require(
        receipt.get("target_object_id") == target_object_id,
        "SAM run target_object_id mismatch",
    )
    require(receipt.get("frame_count") == EXPECTED_FRAME_COUNT, "SAM run frame_count != 25")
    ordered_ids = parse_frame_ids(receipt.get("ordered_frame_ids"), "SAM run.ordered_frame_ids")
    require(ordered_ids == selection.ordered_frame_ids, "SAM run frame order differs")
    producer = require_dict(receipt.get("producer"), "SAM run.producer")
    producer_script = require_dict(producer.get("script"), "SAM run.producer.script")
    require(
        required_sha256(producer_script.get("sha256"), "SAM run.producer.script.sha256")
        == EXPECTED_SAM_RUN_PRODUCER_SHA256,
        "SAM run producer script SHA-256 is not audited",
    )
    producer_script_asset = resolve_asset(
        producer_script,
        relative_to=asset.path.parent,
        label="SAM run.producer.script",
        containment_root=asset.path.parent,
    )
    require(
        producer_script_asset.sha256 == EXPECTED_SAM_RUN_PRODUCER_SHA256,
        "SAM run producer script SHA-256 is not audited",
    )

    model = require_dict(receipt.get("model"), "SAM run.model")
    require_exact_keys(
        model,
        {"family", "implementation", "runner_sha256", "checkpoint"},
        "SAM run.model",
    )
    require(model.get("family") == EXPECTED_SAM_MODEL_FAMILY, "SAM model family is invalid")
    require(
        model.get("implementation") == EXPECTED_SAM_IMPLEMENTATION,
        "SAM implementation is not the audited Holi-Spatial runner",
    )
    require(
        required_sha256(model.get("runner_sha256"), "SAM run.model.runner_sha256")
        == EXPECTED_SAM_RUNNER_SHA256,
        "SAM runner hash is not audited",
    )
    checkpoint = require_dict(model.get("checkpoint"), "SAM run.model.checkpoint")
    require_nonempty_string(checkpoint.get("path"), "SAM run.model.checkpoint.path")
    require(
        required_sha256(checkpoint.get("sha256"), "SAM run.model.checkpoint.sha256")
        == EXPECTED_SAM_CHECKPOINT_SHA256,
        "SAM checkpoint hash is not audited",
    )
    checkpoint_bytes = require_positive_int(
        checkpoint.get("bytes", checkpoint.get("size_bytes")),
        "SAM run.model.checkpoint bytes",
    )
    require(
        checkpoint_bytes == EXPECTED_SAM_CHECKPOINT_BYTES,
        "SAM checkpoint byte size is not audited",
    )

    mask_index = resolve_asset(
        receipt.get("mask_index"),
        relative_to=asset.path.parent,
        label="SAM run.mask_index",
        containment_root=asset.path.parent,
    )
    mask_index_value = read_json(mask_index.path)
    require(
        mask_index_value.get("missing_images") == [],
        "SAM run mask index declares missing input images",
    )

    records = ordered_records(
        receipt,
        key="frame_records",
        expected_frame_ids=ordered_ids,
        label="SAM run",
    )
    frame_set: list[dict[str, Any]] = []
    detection_set: list[dict[str, Any]] = []
    detection_hashes: dict[str, str] = {}
    for selected, record in zip(selection.frames, records, strict=True):
        prefix = f"SAM run {selected.frame_id}"
        require(
            record.get("source_role") == "actual_sam_input_equals_accepted_clean_plate_composite",
            f"{prefix} source role is invalid",
        )
        actual_sam_input = resolve_asset(
            record.get("actual_sam_input"),
            relative_to=asset.path.parent,
            label=f"{prefix}.actual_sam_input",
            containment_root=asset.path.parent,
        )
        accepted_composite = resolve_asset(
            record.get("accepted_composite"),
            relative_to=asset.path.parent,
            label=f"{prefix}.accepted_composite",
            containment_root=asset.path.parent,
        )
        source = resolve_asset(
            record.get("source_rgb"),
            relative_to=asset.path.parent,
            label=f"{prefix}.source_rgb",
            containment_root=asset.path.parent,
        )
        require(
            actual_sam_input.sha256
            == accepted_composite.sha256
            == source.sha256
            == selected.composite_rgb.sha256,
            f"{prefix} actual SAM input does not byte-match the selected composite RGB",
        )
        require(
            record.get("actual_sam_input_sha256") == actual_sam_input.sha256,
            f"{prefix} actual_sam_input_sha256 mismatch",
        )
        require(
            record.get("accepted_composite_sha256") == accepted_composite.sha256,
            f"{prefix} accepted_composite_sha256 mismatch",
        )
        require(
            record.get("source_rgb_sha256") == source.sha256,
            f"{prefix} source_rgb_sha256 mismatch",
        )
        frame_set.append(
            {
                "sequence_index": selected.sequence_index,
                "frame_id": selected.frame_id,
                "actual_sam_input_sha256": actual_sam_input.sha256,
                "accepted_composite_sha256": accepted_composite.sha256,
            }
        )
        detections = require_list(record.get("detections"), f"{prefix}.detections")
        require(bool(detections), f"{prefix} must contain detections")
        frame_mask_paths: set[Path] = set()
        frame_mask_sha256: set[str] = set()
        for raw_detection in detections:
            detection = require_dict(raw_detection, f"{prefix}.detection")
            item_index = detection.get("item_index")
            require(
                isinstance(item_index, int)
                and not isinstance(item_index, bool)
                and item_index >= 0,
                f"{prefix} detection item_index is invalid",
            )
            detection_id = f"{selected.frame_id}:mask:{item_index:06d}"
            require(
                detection.get("detection_id") == detection_id,
                f"{prefix} detection ID is not frame/item-index exact",
            )
            require(detection_id not in detection_hashes, f"duplicate SAM detection {detection_id}")
            label = require_nonempty_string(detection.get("label"), f"{detection_id}.label")
            mask = resolve_asset(
                detection.get("mask"),
                relative_to=asset.path.parent,
                label=f"{detection_id}.mask",
                containment_root=asset.path.parent,
            )
            require(mask.path not in frame_mask_paths, f"{prefix} reuses a mask path")
            require(mask.sha256 not in frame_mask_sha256, f"{prefix} reuses mask bytes")
            frame_mask_paths.add(mask.path)
            frame_mask_sha256.add(mask.sha256)
            require(mask.sha256 == detection.get("mask_sha256"), f"{detection_id} mask mismatch")
            require(
                detection.get("source_rgb_sha256") == source.sha256,
                f"{detection_id} is not bound to the source RGB",
            )
            require(
                detection.get("actual_sam_input_sha256") == actual_sam_input.sha256,
                f"{detection_id} is not bound to the actual SAM input",
            )
            require(
                detection.get("accepted_composite_sha256") == accepted_composite.sha256,
                f"{detection_id} is not bound to the accepted composite",
            )
            mask_values = load_mask(mask, f"{detection_id}.mask")
            require(
                mask_values.shape == (selected.height, selected.width),
                f"{detection_id} dimensions differ from source",
            )
            detection_hashes[detection_id] = mask.sha256
            detection_set.append(
                {
                    "item_index": item_index,
                    "detection_id": detection_id,
                    "frame_id": selected.frame_id,
                    "label": label,
                    "source_rgb_sha256": source.sha256,
                    "actual_sam_input_sha256": actual_sam_input.sha256,
                    "accepted_composite_sha256": accepted_composite.sha256,
                    "mask_sha256": mask.sha256,
                }
            )
    aggregate = require_dict(receipt.get("aggregate"), "SAM run.aggregate")
    require(
        aggregate.get("all_source_frames_hash_bound") is True,
        "SAM run did not bind every source frame",
    )
    require(
        aggregate.get("all_actual_sam_inputs_hash_bound") is True,
        "SAM run did not bind every actual input image",
    )
    require(
        aggregate.get("all_actual_sam_inputs_match_accepted_composites") is True,
        "SAM run actual inputs do not match accepted composites",
    )
    require(
        aggregate.get("source_frame_set_sha256") == canonical_sha256(frame_set),
        "SAM run source frame set digest mismatch",
    )
    for key in (
        "all_detection_masks_hash_verified",
        "all_detection_masks_source_bound",
        "all_25_frames_have_detections",
    ):
        require(aggregate.get(key) is True, f"SAM run.aggregate.{key} is not true")
    require(
        aggregate.get("detection_set_sha256") == canonical_sha256(detection_set),
        "SAM run detection set digest mismatch",
    )
    require(
        aggregate.get("detection_count") == len(detection_hashes),
        "SAM run detection_count mismatch",
    )
    return receipt, mask_index, detection_hashes


def validate_semantic(
    asset: Asset,
    selection: Selection,
    *,
    expected_target_object_id: str,
    round_index: int,
    association_path_remaps: list[tuple[Path, Path]],
) -> tuple[dict[str, Any], tuple[Asset, ...], dict[str, int], Asset]:
    report = read_json(asset.path)
    require(report.get("schema_version") == 1, "semantic schema is invalid")
    require(report.get("kind") == SEMANTIC_KIND, "semantic kind is invalid")
    require(
        report.get("status") == "technical_passed_semantic_only",
        "SAM/semantic QA did not pass",
    )
    require(report.get("review_only") is True, "SAM/semantic QA must remain review-only")
    require(report.get("sam_semantic_gate_passed") is True, "SAM/semantic gate did not pass")
    require(report.get("promotion_approved") is False, "semantic QA exceeded scoped authority")
    require(report.get("frame_count") == EXPECTED_FRAME_COUNT, "semantic frame_count != 25")
    require(report.get("round_index") == round_index, "semantic round_index mismatch")
    producer = require_dict(report.get("producer"), "semantic.producer")
    producer_script = require_dict(producer.get("script"), "semantic.producer.script")
    require(
        required_sha256(producer_script.get("sha256"), "semantic.producer.script.sha256")
        == EXPECTED_SEMANTIC_QA_PRODUCER_SHA256,
        "semantic QA producer script SHA-256 is not audited",
    )
    producer_script_asset = resolve_asset(
        producer_script,
        relative_to=asset.path.parent,
        label="semantic.producer.script",
        containment_root=asset.path.parent,
    )
    require(
        producer_script_asset.sha256 == EXPECTED_SEMANTIC_QA_PRODUCER_SHA256,
        "semantic QA producer script SHA-256 is not audited",
    )
    require(
        parse_frame_ids(report.get("ordered_frame_ids"), "semantic.ordered_frame_ids")
        == selection.ordered_frame_ids,
        "semantic frame order differs from selection",
    )
    require_evidence_binding(
        report.get("candidate_selection_receipt"),
        report_root=asset.path.parent,
        expected=selection.asset,
        label="semantic.candidate_selection_receipt",
    )
    sam_run_asset = resolve_asset(
        report.get("sam_run_receipt"),
        relative_to=asset.path.parent,
        label="semantic.sam_run_receipt",
    )
    target_object_id = require_nonempty_string(
        report.get("target_object_id"), "semantic.target_object_id"
    )
    require(
        target_object_id == expected_target_object_id,
        "semantic target_object_id differs from the materialization input",
    )
    sam_run_value, sam_mask_index, strict_detection_hashes = validate_sam_run_receipt(
        sam_run_asset,
        selection=selection,
        target_object_id=target_object_id,
        round_index=round_index,
    )
    label_registry = parse_label_registry(report.get("label_registry"), "semantic.label_registry")
    require(
        target_object_id not in label_registry,
        "semantic.label_registry still contains the removed target object",
    )
    baseline_asset = resolve_asset(
        report.get("baseline_physical_association"),
        relative_to=asset.path.parent,
        label="semantic.baseline_physical_association",
    )
    candidate_asset = resolve_asset(
        report.get("candidate_physical_association"),
        relative_to=asset.path.parent,
        label="semantic.candidate_physical_association",
    )
    baseline = validate_semantic_association(
        baseline_asset,
        selection=selection,
        source_role="baseline",
        strict_mask_index_sha256=None,
        strict_detection_hashes=None,
        path_remaps=association_path_remaps,
    )
    candidate = validate_semantic_association(
        candidate_asset,
        selection=selection,
        source_role="candidate",
        strict_mask_index_sha256=sam_mask_index.sha256,
        strict_detection_hashes=strict_detection_hashes,
        path_remaps=association_path_remaps,
    )
    require(baseline["anchors"] == candidate["anchors"], "semantic association anchors differ")
    require(
        baseline["anchor_size_bytes"] == candidate["anchor_size_bytes"],
        "semantic association anchor sizes differ",
    )
    require(
        baseline["producer_script_size_bytes"] == candidate["producer_script_size_bytes"],
        "semantic association producer script sizes differ",
    )
    require(
        baseline["camera_info_sha256"] == candidate["camera_info_sha256"],
        "semantic association camera_info differs",
    )
    require(
        baseline["camera_info_size_bytes"] == candidate["camera_info_size_bytes"],
        "semantic association camera_info sizes differ",
    )
    require(
        baseline["thresholds_sha256"] == candidate["thresholds_sha256"],
        "semantic association thresholds differ",
    )
    require(
        [baseline["frames"][frame_id]["camera_sha256"] for frame_id in selection.ordered_frame_ids]
        == [
            candidate["frames"][frame_id]["camera_sha256"]
            for frame_id in selection.ordered_frame_ids
        ],
        "semantic per-frame association cameras differ",
    )
    require(target_object_id in baseline["anchors"], "semantic target has no explicit anchor")
    target_labels = {
        baseline["frames"][frame_id]["assignment_labels"][target_object_id]
        for frame_id in selection.ordered_frame_ids
        if target_object_id in baseline["frames"][frame_id]["assignments"]
    }
    require(bool(target_labels), "semantic target is not visible in baseline")
    require(len(target_labels) == 1, "semantic target label is inconsistent in baseline")
    target_label = next(iter(target_labels))
    require(report.get("target_label") == target_label, "semantic target_label mismatch")
    remaining_ids = sorted(
        {
            object_id
            for frame_id in selection.ordered_frame_ids
            for object_id in baseline["frames"][frame_id]["assignments"]
            if object_id != target_object_id
        }
    )
    expected_registry = {object_id: index + 1 for index, object_id in enumerate(remaining_ids)}
    require(
        label_registry == expected_registry,
        "semantic label_registry differs from associations",
    )
    bindings = require_dict(report.get("association_bindings"), "semantic.association_bindings")
    expected_bindings = {
        "producer_script_sha256": EXPECTED_ASSOCIATION_PRODUCER_SHA256,
        "producer_script_size_bytes": baseline["producer_script_size_bytes"],
        "camera_info_sha256": baseline["camera_info_sha256"],
        "camera_info_size_bytes": baseline["camera_info_size_bytes"],
        "anchors": baseline["anchors"],
        "anchor_size_bytes": baseline["anchor_size_bytes"],
        "thresholds_sha256": baseline["thresholds_sha256"],
        "baseline_mask_index_sha256": baseline["mask_index_sha256"],
        "candidate_mask_index_sha256": candidate["mask_index_sha256"],
        "strict_sam_mask_index_sha256": sam_mask_index.sha256,
        "baseline_source_role": "selection_source_rgb",
        "candidate_source_role": "selection_composite_rgb",
    }
    require(bindings == expected_bindings, "semantic association_bindings do not close")
    require_all_boolean_gates(
        report.get("aggregate"),
        (
            "all_frames_evaluated",
            "all_target_object_absent",
            "all_unassigned_target_overlap_absent",
            "all_remaining_object_semantics_preserved",
            "all_registered_remaining_labels_observed",
        ),
        "semantic.aggregate",
    )
    records = ordered_records(
        report,
        key="frame_records",
        expected_frame_ids=selection.ordered_frame_ids,
        label="semantic",
    )
    contributor_labels: list[Asset] = []
    allowed_labels = {0, *label_registry.values()}
    label_pixel_counts = {label_value: 0 for label_value in label_registry.values()}
    label_frame_counts = {label_value: 0 for label_value in label_registry.values()}
    for frame, record in zip(selection.frames, records, strict=True):
        require(record.get("passed") is True, f"semantic frame {frame.frame_id} did not pass")
        require_all_boolean_gates(
            record.get("gates"), SEMANTIC_FRAME_GATES, f"semantic {frame.frame_id}.gates"
        )
        inputs = require_dict(record.get("inputs"), f"semantic {frame.frame_id}.inputs")
        for key, expected in (
            ("composite_rgb", frame.composite_rgb),
            ("core_mask", frame.core_mask),
            ("editable_mask", frame.editable_mask),
        ):
            resolve_matching_input(
                inputs.get(key),
                report_root=asset.path.parent,
                expected=expected,
                label=f"semantic {frame.frame_id}.{key}",
            )
        baseline_frame = baseline["frames"][frame.frame_id]
        candidate_frame = candidate["frames"][frame.frame_id]
        require(
            target_object_id not in candidate_frame["assignments"],
            f"semantic frame {frame.frame_id} still assigns the target object",
        )
        required_remaining = {
            object_id
            for object_id in baseline_frame["assignments"]
            if object_id != target_object_id
        }
        require(
            required_remaining <= set(candidate_frame["assignments"]),
            f"semantic frame {frame.frame_id} lost a baseline remaining object",
        )
        require(
            set(candidate_frame["assignments"]) <= set(remaining_ids),
            f"semantic frame {frame.frame_id} has an untracked candidate object",
        )
        core_values = load_mask(frame.core_mask, f"semantic {frame.frame_id}.core_mask")
        for detection_id in candidate_frame["unassigned"]:
            mask = candidate["masks"][detection_id]
            if mask["label"] != target_label:
                continue
            overlap_fraction = float(np.count_nonzero(mask["values"] & core_values)) / float(
                core_values.sum()
            )
            require(
                overlap_fraction <= MAX_UNASSIGNED_TARGET_LABEL_CORE_OVERLAP_FRACTION,
                f"semantic frame {frame.frame_id} has target-like unassigned core overlap",
            )
        expected_labels = np.zeros((frame.height, frame.width), dtype=np.int64)
        for object_id in remaining_ids:
            detection_id = candidate_frame["assignments"].get(object_id)
            if detection_id is None:
                continue
            mask_values = candidate["masks"][detection_id]["values"]
            require(
                not bool(np.any((expected_labels > 0) & mask_values)),
                f"semantic frame {frame.frame_id} candidate masks overlap",
            )
            expected_labels[mask_values] = label_registry[object_id]
        outputs = require_dict(record.get("outputs"), f"semantic {frame.frame_id}.outputs")
        contributor_record = require_dict(
            outputs.get("contributor_labels"),
            f"semantic {frame.frame_id}.outputs.contributor_labels",
        )
        require(
            contributor_record.get("source_rgb_sha256") == frame.composite_rgb.sha256,
            f"semantic {frame.frame_id} contributor labels are not bound to the composite RGB",
        )
        contributor = resolve_asset(
            contributor_record,
            relative_to=asset.path.parent,
            label=f"semantic {frame.frame_id}.outputs.contributor_labels",
            containment_root=asset.path.parent,
        )
        labels = load_label_map(contributor, f"semantic {frame.frame_id}.contributor_labels")
        require(
            labels.shape == (frame.height, frame.width),
            f"semantic {frame.frame_id} contributor-label dimensions differ",
        )
        observed_labels = {int(value) for value in np.unique(labels)}
        require(
            observed_labels <= allowed_labels,
            f"semantic {frame.frame_id} contributor labels contain unregistered values",
        )
        require(
            np.array_equal(labels, expected_labels),
            f"semantic {frame.frame_id} contributor labels differ from candidate assignments",
        )
        for label_value in label_registry.values():
            pixels = int(np.count_nonzero(labels == label_value))
            label_pixel_counts[label_value] += pixels
            label_frame_counts[label_value] += int(pixels > 0)
        contributor_labels.append(contributor)

    preservation = require_dict(
        require_dict(report.get("aggregate"), "semantic.aggregate").get(
            "remaining_label_preservation"
        ),
        "semantic.aggregate.remaining_label_preservation",
    )
    require_exact_keys(
        preservation,
        set(label_registry),
        "semantic.aggregate.remaining_label_preservation",
    )
    for object_id, label_value in label_registry.items():
        prefix = f"semantic.aggregate.remaining_label_preservation.{object_id}"
        record = require_dict(preservation.get(object_id), prefix)
        require_exact_keys(
            record,
            {"label", "frame_occurrence_count", "pixel_count", "preserved"},
            prefix,
        )
        require(
            label_pixel_counts[label_value] > 0,
            f"registered remaining label {object_id} absent",
        )
        require(record.get("label") == label_value, f"{prefix}.label mismatch")
        require(
            record.get("frame_occurrence_count") == label_frame_counts[label_value],
            f"{prefix}.frame_occurrence_count mismatch",
        )
        require(
            record.get("pixel_count") == label_pixel_counts[label_value],
            f"{prefix}.pixel_count mismatch",
        )
        require(record.get("preserved") is True, f"{prefix}.preserved did not pass")
    require(
        sam_run_value.get("target_object_id") == target_object_id,
        "semantic strict SAM target binding changed",
    )
    return report, tuple(contributor_labels), label_registry, sam_run_asset


VISUAL_GATES = (
    "all_frames_reviewed",
    "no_visible_target_object_residual",
    "no_obvious_boundary_misalignment",
    "background_or_remaining_objects_plausible",
    "accepted_for_next_round_source",
)


def validate_visual_review(
    asset: Asset,
    *,
    selection: Selection,
    boundary: Asset,
    temporal: Asset,
    semantic: Asset,
) -> dict[str, Any]:
    review = read_json(asset.path)
    require(review.get("schema_version") == 1, "visual review schema is invalid")
    require(review.get("kind") == VISUAL_REVIEW_KIND, "visual review kind is invalid")
    require(review.get("status") == "passed", "human visual review did not pass")
    require(
        review.get("decision") == "accepted_for_next_round_source",
        "human review decision is not accepted_for_next_round_source",
    )
    require(review.get("accepted_for_next_round_source") is True, "human review did not accept")
    require(review.get("promotion_approved") is False, "human review cannot promote live assets")
    require(
        review.get("canonical_or_live_promotion_approved") is False,
        "human review cannot authorize canonical/live promotion",
    )
    require(review.get("frame_count") == EXPECTED_FRAME_COUNT, "visual frame_count != 25")
    require(
        parse_frame_ids(review.get("ordered_frame_ids"), "visual.ordered_frame_ids")
        == selection.ordered_frame_ids,
        "visual frame order differs from selection",
    )
    require(
        parse_frame_ids(review.get("reviewed_frame_ids"), "visual.reviewed_frame_ids")
        == selection.ordered_frame_ids,
        "human review did not review all frames in order",
    )
    reviewer = require_dict(review.get("reviewer"), "visual.reviewer")
    require(reviewer.get("kind") == "human", "visual reviewer must be human")
    require_nonempty_string(reviewer.get("id"), "visual.reviewer.id")
    require_timezone_timestamp(review.get("reviewed_at"), "visual.reviewed_at")
    evidence = require_dict(review.get("evidence"), "visual.evidence")
    for key, expected in (
        ("candidate_selection_receipt", selection.asset),
        ("boundary_qa_report", boundary),
        ("temporal_qa_report", temporal),
        ("sam_semantic_qa_report", semantic),
    ):
        require_evidence_binding(
            evidence.get(key),
            report_root=asset.path.parent,
            expected=expected,
            label=f"visual.evidence.{key}",
        )
    require_all_boolean_gates(review.get("gates"), VISUAL_GATES, "visual.gates")
    records = ordered_records(
        review,
        key="frame_records",
        expected_frame_ids=selection.ordered_frame_ids,
        label="visual",
    )
    for frame, record in zip(selection.frames, records, strict=True):
        require(record.get("passed") is True, f"visual frame {frame.frame_id} did not pass")
        require(
            record.get("composite_rgb_sha256") == frame.composite_rgb.sha256,
            f"visual frame {frame.frame_id} reviewed another composite",
        )
    limitations = review.get("limitations", [])
    require(isinstance(limitations, list), "visual.limitations must be an array")
    require(
        all(isinstance(item, str) and bool(item.strip()) for item in limitations),
        "visual.limitations entries must be non-empty strings",
    )
    return review


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def copy_verified(source: Asset, destination: Path) -> Asset:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source.path, destination)
    digest, size = sha256_file(destination)
    require(digest == source.sha256, f"copied asset SHA-256 mismatch: {destination}")
    require(size == source.size_bytes, f"copied asset size mismatch: {destination}")
    return Asset(path=destination, sha256=digest, size_bytes=size)


def output_record(asset: Asset, root: Path) -> dict[str, Any]:
    require(is_contained(asset.path.resolve(), root.resolve()), "output asset escaped staging root")
    return asset_record(asset, path=asset.path.relative_to(root).as_posix())


def prepare_destination(value: str | Path) -> Path:
    lexical = Path(value).expanduser()
    require(not lexical.is_symlink(), f"output_dir cannot be a symlink: {lexical}")
    destination = lexical.resolve()
    require(not destination.exists(), f"refusing to overwrite output_dir: {destination}")
    parts = destination.parts
    forbidden_pairs = {("web", "public"), ("public", "worlds")}
    require(
        not any(pair in forbidden_pairs for pair in pairwise(parts)),
        "output_dir may not target a canonical/live web asset tree",
    )
    return destination


def load_materialization_input(path: Path) -> tuple[dict[str, Any], Asset]:
    manifest_path = path.expanduser().resolve()
    require(manifest_path.is_file(), f"input manifest does not exist: {manifest_path}")
    digest, size = sha256_file(manifest_path)
    value = read_json(manifest_path)
    require(value.get("schema_version") == 1, "input schema_version must be 1")
    require(value.get("kind") == INPUT_KIND, "input kind is invalid")
    require(value.get("expected_frame_count") == 25, "input expected_frame_count must be 25")
    require(
        value.get("destination_scope") == "next_round_source_only",
        "input destination_scope must be next_round_source_only",
    )
    require(
        value.get("canonical_or_live_manifest_targeted") is False,
        "input may not target canonical/live manifests",
    )
    round_index = require_positive_int(value.get("round_index"), "input.round_index")
    require_nonempty_string(value.get("target_object_id"), "input.target_object_id")
    require(
        value.get("next_round_index") == round_index + 1,
        "input.next_round_index must equal round_index + 1",
    )
    predecessor = value.get("previous_accepted_next_round_source_manifest")
    if round_index == 1:
        require(predecessor is None, "round 1 must not declare a previous accepted source")
    else:
        require(
            isinstance(predecessor, dict),
            "round 2+ requires previous_accepted_next_round_source_manifest",
        )
    return value, Asset(manifest_path, digest, size)


def materialize_accepted_clean_plate_next_round(
    input_manifest: str | Path,
    output_dir: str | Path,
    *,
    created_at: datetime | None = None,
    association_path_remaps: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate all acceptance evidence and atomically create a next-round package."""

    input_value, input_asset = load_materialization_input(Path(input_manifest))
    parsed_association_path_remaps = parse_path_remaps(association_path_remaps or [])
    input_root = input_asset.path.parent
    selection_asset = resolve_asset(
        input_value.get("candidate_selection_receipt"),
        relative_to=input_root,
        label="input.candidate_selection_receipt",
    )
    boundary_asset = resolve_asset(
        input_value.get("boundary_qa_report"),
        relative_to=input_root,
        label="input.boundary_qa_report",
    )
    temporal_asset = resolve_asset(
        input_value.get("temporal_qa_report"),
        relative_to=input_root,
        label="input.temporal_qa_report",
    )
    semantic_asset = resolve_asset(
        input_value.get("sam_semantic_qa_report"),
        relative_to=input_root,
        label="input.sam_semantic_qa_report",
    )
    visual_asset = resolve_asset(
        input_value.get("human_visual_review_receipt"),
        relative_to=input_root,
        label="input.human_visual_review_receipt",
    )

    selection = validate_selection(selection_asset)
    round_index = int(input_value["round_index"])
    previous_asset: Asset | None = None
    previous_source: PreviousAcceptedSource | None = None
    if round_index > 1:
        previous_asset = resolve_asset(
            input_value.get("previous_accepted_next_round_source_manifest"),
            relative_to=input_root,
            label="input.previous_accepted_next_round_source_manifest",
        )
        previous_source = validate_previous_accepted_source(
            previous_asset,
            selection=selection,
            current_round_index=round_index,
        )
    validate_boundary(boundary_asset, selection)
    _, cumulative_masks = validate_temporal(
        temporal_asset,
        selection,
        boundary_asset,
        round_index=round_index,
        previous_source=previous_source,
    )
    _, contributor_labels, label_registry, sam_run_asset = validate_semantic(
        semantic_asset,
        selection,
        expected_target_object_id=input_value["target_object_id"],
        round_index=round_index,
        association_path_remaps=parsed_association_path_remaps,
    )
    visual_review = validate_visual_review(
        visual_asset,
        selection=selection,
        boundary=boundary_asset,
        temporal=temporal_asset,
        semantic=semantic_asset,
    )
    dimensions = {(frame.width, frame.height) for frame in selection.frames}
    require(len(dimensions) == 1, "selected frames do not have one common resolution")

    timestamp = created_at or datetime.now(timezone.utc)  # noqa: UP017 - remote Python 3.10.
    require(
        timestamp.tzinfo is not None and timestamp.utcoffset() is not None,
        "created_at must include timezone information",
    )
    destination = prepare_destination(output_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        evidence_sources = {
            "candidate_selection_receipt": selection_asset,
            "boundary_qa_report": boundary_asset,
            "temporal_qa_report": temporal_asset,
            "sam_semantic_qa_report": semantic_asset,
            "sam_run_receipt": sam_run_asset,
            "human_visual_review_receipt": visual_asset,
        }
        if previous_asset is not None:
            evidence_sources["previous_accepted_next_round_source_manifest"] = previous_asset
        evidence_outputs: dict[str, dict[str, Any]] = {}
        for key, source in evidence_sources.items():
            copied = copy_verified(source, staging / "evidence" / f"{key}.json")
            evidence_outputs[key] = output_record(copied, staging)

        frame_records: list[dict[str, Any]] = []
        frame_set: list[dict[str, Any]] = []
        for frame, cumulative_mask, contributor_labels_asset in zip(
            selection.frames,
            cumulative_masks,
            contributor_labels,
            strict=True,
        ):
            filename = f"{frame.sequence_index:04d}.png"
            output_rgb = copy_verified(frame.composite_rgb, staging / "frames" / filename)
            output_core = copy_verified(frame.core_mask, staging / "masks" / "core" / filename)
            output_editable = copy_verified(
                frame.editable_mask, staging / "masks" / "editable" / filename
            )
            output_cumulative = copy_verified(
                cumulative_mask, staging / "masks" / "cumulative" / filename
            )
            output_contributor_labels = copy_verified(
                contributor_labels_asset,
                staging / "contributor_labels" / filename,
            )
            rgb_record = output_record(output_rgb, staging)
            core_record = output_record(output_core, staging)
            editable_record = output_record(output_editable, staging)
            cumulative_record = output_record(output_cumulative, staging)
            contributor_record = output_record(output_contributor_labels, staging)
            contributor_record["source_rgb_sha256"] = output_rgb.sha256
            frame_records.append(
                {
                    "sequence_index": frame.sequence_index,
                    "frame_id": frame.frame_id,
                    "source_role": "immediately_previous_accepted_clean_plate",
                    "source_rgb": rgb_record,
                    "segmentation_source_rgb_sha256": output_rgb.sha256,
                    "accepted_round_core_mask": core_record,
                    "accepted_round_editable_mask": editable_record,
                    "previous_edited_region_mask": cumulative_record,
                    "previous_contributor_labels": contributor_record,
                    "previous_source_rgb": asset_record(frame.source_rgb),
                    "dimensions": {"width": frame.width, "height": frame.height},
                    "mask_pixels": {
                        "core": frame.core_pixels,
                        "editable": frame.editable_pixels,
                    },
                    "lineage": {
                        "candidate_selection_receipt_sha256": selection_asset.sha256,
                        "boundary_qa_report_sha256": boundary_asset.sha256,
                        "temporal_qa_report_sha256": temporal_asset.sha256,
                        "sam_semantic_qa_report_sha256": semantic_asset.sha256,
                        "sam_run_receipt_sha256": sam_run_asset.sha256,
                        "human_visual_review_receipt_sha256": visual_asset.sha256,
                        "previous_accepted_next_round_source_manifest_sha256": (
                            previous_asset.sha256 if previous_asset is not None else None
                        ),
                    },
                }
            )
            frame_set.append(
                {
                    "sequence_index": frame.sequence_index,
                    "frame_id": frame.frame_id,
                    "source_rgb_sha256": output_rgb.sha256,
                    "core_mask_sha256": output_core.sha256,
                    "editable_mask_sha256": output_editable.sha256,
                    "cumulative_mask_sha256": output_cumulative.sha256,
                    "contributor_labels_sha256": output_contributor_labels.sha256,
                }
            )

        output_manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": OUTPUT_KIND,
            "status": "accepted_clean_plate_materialized_for_next_round_source",
            "created_at": timestamp.isoformat(),
            "round_index": input_value["round_index"],
            "next_round_index": input_value["next_round_index"],
            "target_object_id": input_value["target_object_id"],
            "frame_count": EXPECTED_FRAME_COUNT,
            "ordered_frame_ids": list(selection.ordered_frame_ids),
            "next_round_source_approved": True,
            "promotion_approved": False,
            "canonical_or_live_manifest_modified": False,
            "evidence": evidence_outputs,
            "contributor_label_registry": label_registry,
            "frame_records": frame_records,
            "aggregate": {
                "frame_set_sha256": canonical_sha256(frame_set),
                "one_to_one_frame_mapping_verified": True,
                "all_source_and_mask_hashes_verified": True,
                "all_cumulative_masks_and_contributor_labels_verified": True,
                "all_output_assets_contained": True,
                "all_boundary_frames_passed": True,
                "all_temporal_pairs_and_triplets_passed": True,
                "all_sam_semantic_frames_passed": True,
                "all_frames_human_reviewed_and_accepted": True,
            },
            "next_stage": {
                "name": "segment_next_round_on_exact_accepted_source_rgb",
                "required_binding": (
                    "each next-round segmentation source SHA-256 must equal the corresponding "
                    "frame_records[].source_rgb.sha256"
                ),
                "required_previous_state": (
                    "frame_records[].previous_edited_region_mask and "
                    "frame_records[].previous_contributor_labels must be forwarded together"
                ),
            },
            "limitations": list(visual_review.get("limitations", [])),
        }
        output_manifest_path = staging / "next_round_source_manifest.json"
        atomic_write_json(output_manifest_path, output_manifest)
        output_manifest_sha, output_manifest_size = sha256_file(output_manifest_path)
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "kind": OUTPUT_RECEIPT_KIND,
            "status": "accepted_clean_plate_materialized_for_next_round_source",
            "created_at": timestamp.isoformat(),
            "input_manifest": asset_record(input_asset),
            "target_object_id": input_value["target_object_id"],
            "output_manifest": {
                "path": "next_round_source_manifest.json",
                "sha256": output_manifest_sha,
                "size_bytes": output_manifest_size,
            },
            "frame_count": EXPECTED_FRAME_COUNT,
            "ordered_frame_ids": list(selection.ordered_frame_ids),
            "frame_set_sha256": output_manifest["aggregate"]["frame_set_sha256"],
            "next_round_source_approved": True,
            "promotion_approved": False,
            "canonical_or_live_manifest_modified": False,
            "materialization": {
                "same_parent_staging": True,
                "atomic_directory_rename": True,
                "output_assets_hash_verified_after_copy": True,
                "output_paths_contained": True,
            },
        }
        atomic_write_json(staging / "receipt.json", receipt)

        for record in output_manifest["frame_records"]:
            for key in (
                "source_rgb",
                "accepted_round_core_mask",
                "accepted_round_editable_mask",
                "previous_edited_region_mask",
                "previous_contributor_labels",
            ):
                resolve_asset(
                    record[key],
                    relative_to=staging,
                    label=f"output.{record['frame_id']}.{key}",
                    containment_root=staging,
                )
        require(not destination.exists(), f"output_dir appeared during run: {destination}")
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_manifest, receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--association-path-remap", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest, receipt = materialize_accepted_clean_plate_next_round(
            args.manifest,
            args.output_dir,
            association_path_remaps=args.association_path_remap,
        )
    except AcceptedCleanPlateMaterializationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "frame_count": manifest["frame_count"],
                "frame_set_sha256": receipt["frame_set_sha256"],
                "output_dir": str(args.output_dir.expanduser().resolve()),
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
