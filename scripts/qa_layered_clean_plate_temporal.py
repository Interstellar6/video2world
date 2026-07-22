#!/usr/bin/env python3
"""Strict review-only temporal QA for a 25-frame layered clean plate.

The production backend is the original (plain) RAFT implementation.  Flow is
estimated in both directions for every adjacent pair.  Metrics are evaluated
only where forward/backward flow agrees, a warped cumulative repair core lands
inside the next frame's eroded repair core, and a local control ring provides a
well-conditioned robust exposure fit.

This report is deliberately insufficient for promotion: it does not use PBR or
DA3 evidence, and even a temporal pass says nothing about semantic correctness.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

import numpy as np
from PIL import Image
from scipy import ndimage

MANIFEST_KIND = "video2world.layered_clean_plate_temporal_qa_manifest"
REPORT_KIND = "video2world.layered_clean_plate_temporal_qa_report"
SELECTION_KIND = "video2world.clean_plate_candidate_selection_receipt"
BOUNDARY_REPORT_KIND = "video2world.layered_clean_plate_boundary_qa_report"

EXPECTED_FRAME_COUNT = 25
EXPECTED_RESOLUTION = (1280, 720)
EXPECTED_RAFT_COMMIT = "e870e79321c31b733e2031af5aa2fb1fe3ac7eec"
EXPECTED_RAFT_CHECKPOINT_SHA256 = "fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1"

REFERENCE_SAMPLING: dict[str, Any] = {
    "core_erosion_px": 4,
    "control_ring_inner_px": 8,
    "control_ring_outer_px": 32,
    "fb_consistency_absolute_px": 1.5,
    "fb_consistency_relative_to_flow": 0.01,
    "rgb_interpolation": "bilinear",
    "mask_interpolation": "nearest",
}
REFERENCE_THRESHOLDS: dict[str, int | float] = {
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


class InputContractError(ValueError):
    """The manifest or one of its hash-bound inputs is invalid."""


class NotEvaluableError(RuntimeError):
    """Runtime evidence is insufficient to evaluate the temporal contract."""


class FlowProvider(Protocol):
    metadata: dict[str, Any]

    def estimate(self, source_rgb: np.ndarray, target_rgb: np.ndarray) -> np.ndarray:
        """Return source-to-target flow as HxWx2 float pixels."""


FlowProviderFactory = Callable[[dict[str, Any]], FlowProvider]


@dataclass(frozen=True)
class Asset:
    path: Path
    sha256: str
    size_bytes: int
    dimensions: tuple[int, int] | None = None

    def record(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "path": str(self.path),
            "sha256": self.sha256,
            "bytes": self.size_bytes,
        }
        if self.dimensions is not None:
            result["dimensions"] = list(self.dimensions)
        return result


@dataclass(frozen=True)
class MaskAsset:
    asset: Asset
    channel: str
    threshold: int

    def record(self) -> dict[str, Any]:
        return {
            **self.asset.record(),
            "channel": self.channel,
            "threshold": self.threshold,
        }


@dataclass(frozen=True)
class FrameSpec:
    sequence_index: int
    frame_id: str
    selected_composite_rgb: Asset
    source_rgb: Asset
    current_core_mask: MaskAsset
    cumulative_core_mask: MaskAsset
    editable_mask: MaskAsset
    selection_receipt: Asset


@dataclass
class FrameData:
    spec: FrameSpec
    composite: np.ndarray
    source: np.ndarray
    current_core: np.ndarray
    cumulative_core: np.ndarray
    editable: np.ndarray
    eroded_core: np.ndarray
    control_ring: np.ndarray
    gradient: np.ndarray


@dataclass
class FlowValidity:
    target_x: np.ndarray
    target_y: np.ndarray
    in_bounds: np.ndarray
    fb_consistent: np.ndarray
    core_target_overlap: np.ndarray
    control_target_overlap: np.ndarray
    core_valid: np.ndarray
    control_valid: np.ndarray
    fb_error: np.ndarray


@dataclass
class PairRuntime:
    record: dict[str, Any]
    validity: FlowValidity
    exposure_fit: dict[str, Any] | None


def writable_c_contiguous_rgb(value: np.ndarray) -> np.ndarray:
    """Return owned writable storage before handing an image to PyTorch."""

    return np.array(value, dtype=np.uint8, order="C", copy=True)


@dataclass(frozen=True)
class ValidatedManifest:
    path: Path
    sha256: str
    round_index: int
    selection_manifest: Asset
    upstream_boundary_report: Asset
    pairing: dict[str, Any]
    flow_backend: dict[str, Any]
    sampling: dict[str, Any]
    thresholds: dict[str, int | float]
    frames: tuple[FrameSpec, ...]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InputContractError(f"{label} must be an object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise InputContractError(f"{label} must be an array")
    return value


def require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise InputContractError(f"{label} must be a non-empty string")
    return value


def require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise InputContractError(f"{label} must be a lowercase SHA-256 digest")
    return value


def require_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise InputContractError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise InputContractError(f"{label} must be at least {minimum}")
    return value


def require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InputContractError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise InputContractError(f"{label} must be a finite number")
    return result


def resolve_path(value: str, manifest_path: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def resolve_asset(value: Any, *, manifest_path: Path, label: str) -> Asset:
    record = require_dict(value, label)
    path = resolve_path(require_string(record.get("path"), f"{label}.path"), manifest_path)
    if not path.is_file():
        raise InputContractError(f"{label}.path does not exist: {path}")
    expected = require_sha256(record.get("sha256"), f"{label}.sha256")
    actual = sha256_file(path)
    if actual != expected:
        raise InputContractError(f"{label}.sha256 does not match the input file")
    return Asset(path=path, sha256=actual, size_bytes=path.stat().st_size)


def resolve_rgb_asset(value: Any, *, manifest_path: Path, label: str) -> Asset:
    asset = resolve_asset(value, manifest_path=manifest_path, label=label)
    with Image.open(asset.path) as image:
        dimensions = image.size
    if dimensions != EXPECTED_RESOLUTION:
        raise InputContractError(
            f"{label} must be full-resolution {EXPECTED_RESOLUTION[0]}x{EXPECTED_RESOLUTION[1]}"
        )
    return Asset(**{**asset.__dict__, "dimensions": dimensions})


def resolve_mask_asset(value: Any, *, manifest_path: Path, label: str) -> MaskAsset:
    record = require_dict(value, label)
    asset = resolve_asset(record, manifest_path=manifest_path, label=label)
    channel = record.get("channel")
    if channel not in {"luma", "alpha"}:
        raise InputContractError(f"{label}.channel must be 'luma' or 'alpha'")
    threshold = require_int(record.get("threshold"), f"{label}.threshold", minimum=1)
    if threshold > 255:
        raise InputContractError(f"{label}.threshold must not exceed 255")
    with Image.open(asset.path) as image:
        if channel == "alpha" and "A" not in image.getbands():
            raise InputContractError(f"{label} requests alpha but the image has no alpha")
        dimensions = image.size
    if dimensions != EXPECTED_RESOLUTION:
        raise InputContractError(
            f"{label} must be full-resolution {EXPECTED_RESOLUTION[0]}x{EXPECTED_RESOLUTION[1]}"
        )
    resolved = Asset(**{**asset.__dict__, "dimensions": dimensions})
    return MaskAsset(asset=resolved, channel=channel, threshold=threshold)


def read_mask(asset: MaskAsset) -> np.ndarray:
    with Image.open(asset.asset.path) as image:
        values = np.asarray(
            image.getchannel("A") if asset.channel == "alpha" else image.convert("L"),
            dtype=np.uint8,
        )
    return values >= asset.threshold


def read_rgb(asset: Asset) -> np.ndarray:
    with Image.open(asset.path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def validate_pairing(value: Any) -> dict[str, Any]:
    pairing = require_dict(value, "pairing")
    expected = {
        "mode": "adjacent_bidirectional",
        "frame_stride": 1,
        "undirected_pair_count": 24,
        "directed_pair_count": 48,
        "triplet_count": 23,
    }
    for key, expected_value in expected.items():
        if pairing.get(key) != expected_value:
            raise InputContractError(f"pairing.{key} must equal {expected_value!r}")
    return expected


def validate_sampling(value: Any) -> dict[str, Any]:
    sampling = require_dict(value, "sampling")
    unknown = sorted(set(sampling) - set(REFERENCE_SAMPLING))
    missing = sorted(set(REFERENCE_SAMPLING) - set(sampling))
    if missing:
        raise InputContractError(f"sampling is missing required keys: {', '.join(missing)}")
    if unknown:
        raise InputContractError(f"sampling has unknown keys: {', '.join(unknown)}")
    for key, expected in REFERENCE_SAMPLING.items():
        observed = sampling[key]
        if isinstance(expected, float):
            observed = require_number(observed, f"sampling.{key}")
            if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
                raise InputContractError(f"sampling.{key} must equal {expected}")
        elif observed != expected:
            raise InputContractError(f"sampling.{key} must equal {expected!r}")
    return dict(REFERENCE_SAMPLING)


def validate_thresholds(value: Any) -> dict[str, int | float]:
    thresholds = require_dict(value, "thresholds")
    missing = sorted(set(REFERENCE_THRESHOLDS) - set(thresholds))
    unknown = sorted(set(thresholds) - set(REFERENCE_THRESHOLDS))
    if missing:
        raise InputContractError(f"thresholds is missing required keys: {', '.join(missing)}")
    if unknown:
        raise InputContractError(f"thresholds has unknown keys: {', '.join(unknown)}")

    result: dict[str, int | float] = {}
    minimum_keys = {
        "minimum_core_valid_fraction_per_direction",
        "minimum_control_valid_fraction_per_direction",
        "minimum_region_pixels",
        "minimum_triplet_valid_fraction",
    }
    for key, reference in REFERENCE_THRESHOLDS.items():
        if isinstance(reference, int):
            observed: int | float = require_int(thresholds[key], f"thresholds.{key}", minimum=0)
        else:
            observed = require_number(thresholds[key], f"thresholds.{key}")
        if key in minimum_keys and observed < reference:
            raise InputContractError(f"thresholds.{key} must be at least {reference}")
        if key not in minimum_keys and observed > reference:
            raise InputContractError(f"thresholds.{key} must not exceed {reference}")
        result[key] = observed
    if not 0.0 <= float(result["maximum_core_bad_pixel_fraction"]) <= 1.0:
        raise InputContractError("thresholds.maximum_core_bad_pixel_fraction must be in [0, 1]")
    return result


def validate_flow_backend(
    value: Any,
    *,
    manifest_path: Path,
) -> dict[str, Any]:
    backend = require_dict(value, "flow_backend")
    if backend.get("kind") != "plain_raft":
        raise InputContractError("flow_backend.kind must equal 'plain_raft'")
    repository_path = resolve_path(
        require_string(backend.get("repository_path"), "flow_backend.repository_path"),
        manifest_path,
    )
    if not repository_path.is_dir():
        raise InputContractError(
            f"flow_backend.repository_path is not a directory: {repository_path}"
        )
    if backend.get("repository_commit") != EXPECTED_RAFT_COMMIT:
        raise InputContractError(
            f"flow_backend.repository_commit must equal {EXPECTED_RAFT_COMMIT}"
        )
    checkpoint = resolve_asset(
        backend.get("checkpoint"),
        manifest_path=manifest_path,
        label="flow_backend.checkpoint",
    )
    if checkpoint.sha256 != EXPECTED_RAFT_CHECKPOINT_SHA256:
        raise InputContractError(
            "flow_backend.checkpoint.sha256 does not match the audited RAFT checkpoint"
        )
    if backend.get("precision") != "float32":
        raise InputContractError("flow_backend.precision must equal 'float32'")
    if backend.get("iterations") != 20:
        raise InputContractError("flow_backend.iterations must equal 20")
    if backend.get("flow_completion") is not False:
        raise InputContractError("flow_backend.flow_completion must be false")
    if backend.get("inference_resolution") != list(EXPECTED_RESOLUTION):
        raise InputContractError(
            f"flow_backend.inference_resolution must equal {list(EXPECTED_RESOLUTION)}"
        )
    device = require_string(backend.get("device"), "flow_backend.device")
    if not device.startswith("cuda"):
        raise InputContractError("flow_backend.device must select a CUDA device")
    return {
        "kind": "plain_raft",
        "repository_path": str(repository_path),
        "repository_commit": EXPECTED_RAFT_COMMIT,
        "checkpoint": checkpoint.record(),
        "precision": "float32",
        "iterations": 20,
        "flow_completion": False,
        "inference_resolution": list(EXPECTED_RESOLUTION),
        "device": device,
    }


def validate_selection_receipt(
    receipt: Asset,
    *,
    frame_id: str,
    sequence_index: int,
    source_rgb: Asset,
    composite_rgb: Asset,
    current_core: MaskAsset,
    editable: MaskAsset,
) -> None:
    value = require_dict(read_json(receipt.path), f"selection receipt for {frame_id}")
    if value.get("kind") != SELECTION_KIND:
        raise InputContractError(f"selection receipt for {frame_id} has the wrong kind")
    records = require_list(value.get("frames"), f"selection receipt frames for {frame_id}")
    matches = [
        record
        for record in records
        if isinstance(record, dict) and record.get("frame_id") == frame_id
    ]
    if len(matches) != 1:
        raise InputContractError(f"selection receipt must contain frame {frame_id} exactly once")
    selected = matches[0]
    if selected.get("sequence_index") != sequence_index:
        raise InputContractError(f"selection receipt sequence index mismatch for {frame_id}")
    exactness = require_dict(selected.get("exactness"), f"selection exactness for {frame_id}")
    if exactness.get("passed") is not True:
        raise InputContractError(f"selection exactness did not pass for {frame_id}")
    outputs = require_dict(selected.get("outputs"), f"selection outputs for {frame_id}")
    bindings = (
        ("composite_rgb", composite_rgb.sha256),
        ("core_mask", current_core.asset.sha256),
        ("editable_mask", editable.asset.sha256),
    )
    for key, digest in bindings:
        output = require_dict(outputs.get(key), f"selection outputs.{key} for {frame_id}")
        if output.get("sha256") != digest:
            raise InputContractError(f"selection output {key} hash mismatch for {frame_id}")
    selected_source = require_dict(
        selected.get("source_rgb"),
        f"selection source_rgb for {frame_id}",
    )
    if selected_source.get("sha256") != source_rgb.sha256:
        raise InputContractError(f"selection source RGB hash mismatch for {frame_id}")


def upstream_boundary_report_summary(report: dict[str, Any]) -> str:
    details = []
    for key in (
        "status",
        "boundary_texture_gate_passed",
        "promotion_scope",
        "promotion_approved",
    ):
        if key in report:
            value = report[key]
            if isinstance(value, bool):
                value = str(value).lower()
            details.append(f"{key}={value}")
    next_action = report.get("next_action")
    if isinstance(next_action, dict):
        for key in (
            "action",
            "blocking_gate_groups",
            "failed_gates",
            "failed_frame_ids",
            "first_failed_frame_id",
        ):
            value = next_action.get(key)
            if isinstance(value, list):
                details.append(f"{key}=" + ",".join(str(item) for item in value))
            elif value:
                details.append(f"{key}={value}")
        failed_frame_gates = next_action.get("failed_frame_gates")
        if isinstance(failed_frame_gates, list):
            frame_summaries = []
            for item in failed_frame_gates:
                if not isinstance(item, dict):
                    continue
                frame_id = item.get("frame_id")
                failed = item.get("failed_gates")
                if frame_id and isinstance(failed, list):
                    frame_summaries.append(
                        f"{frame_id}:{','.join(str(gate) for gate in failed)}"
                    )
            if frame_summaries:
                details.append("failed_frame_gates=" + "|".join(frame_summaries))
    return "[" + "; ".join(details) + "]"


def validate_boundary_report(asset: Asset, frames: tuple[FrameSpec, ...]) -> None:
    report = require_dict(read_json(asset.path), "upstream boundary report")
    if report.get("kind") != BOUNDARY_REPORT_KIND:
        raise InputContractError("upstream boundary report has the wrong kind")
    if report.get("status") != "technical_passed":
        raise InputContractError(
            "upstream boundary report did not technically pass "
            + upstream_boundary_report_summary(report)
        )
    if report.get("boundary_texture_gate_passed") is not True:
        raise InputContractError(
            "upstream boundary and texture gate did not pass "
            + upstream_boundary_report_summary(report)
        )
    if report.get("promotion_approved") is not False:
        raise InputContractError("upstream boundary report exceeded its scoped authority")
    if report.get("promotion_scope") != "boundary_and_local_texture_only":
        raise InputContractError("upstream boundary report has an unknown promotion scope")
    if report.get("frame_count") != EXPECTED_FRAME_COUNT:
        raise InputContractError("upstream boundary report must cover exactly 25 frames")
    records = require_list(report.get("frame_records"), "upstream boundary frame_records")
    by_id = {record.get("frame_id"): record for record in records if isinstance(record, dict)}
    if set(by_id) != {frame.frame_id for frame in frames}:
        raise InputContractError("upstream boundary report frame set does not match manifest")
    for frame in frames:
        inputs = require_dict(by_id[frame.frame_id].get("inputs"), "boundary frame inputs")
        bindings = (
            ("composite_rgb", frame.selected_composite_rgb.sha256),
            ("removal_core_mask", frame.current_core_mask.asset.sha256),
            ("editable_mask", frame.editable_mask.asset.sha256),
        )
        for key, digest in bindings:
            record = require_dict(inputs.get(key), f"boundary inputs.{key}")
            if record.get("sha256") != digest:
                raise InputContractError(
                    f"upstream boundary report {key} hash mismatch for {frame.frame_id}"
                )


def validate_manifest(path: Path) -> ValidatedManifest:
    manifest_path = path.expanduser().resolve()
    if not manifest_path.is_file():
        raise InputContractError(f"input manifest does not exist: {manifest_path}")
    manifest = require_dict(read_json(manifest_path), "manifest")
    if manifest.get("schema_version") != 1:
        raise InputContractError("manifest.schema_version must equal 1")
    if manifest.get("kind") != MANIFEST_KIND:
        raise InputContractError(f"manifest.kind must equal {MANIFEST_KIND!r}")
    round_index = require_int(manifest.get("round_index"), "round_index", minimum=1)
    if manifest.get("frame_count") != EXPECTED_FRAME_COUNT:
        raise InputContractError("manifest.frame_count must equal 25")
    selection_manifest = resolve_asset(
        manifest.get("selection_manifest"),
        manifest_path=manifest_path,
        label="selection_manifest",
    )
    selection_value = require_dict(read_json(selection_manifest.path), "selection_manifest")
    if selection_value.get("kind") != SELECTION_KIND:
        raise InputContractError("selection_manifest has the wrong kind")
    upstream_boundary_report = resolve_asset(
        manifest.get("upstream_boundary_report"),
        manifest_path=manifest_path,
        label="upstream_boundary_report",
    )
    pairing = validate_pairing(manifest.get("pairing"))
    flow_backend = validate_flow_backend(
        manifest.get("flow_backend"),
        manifest_path=manifest_path,
    )
    sampling = validate_sampling(manifest.get("sampling"))
    thresholds = validate_thresholds(manifest.get("thresholds"))
    raw_records = require_list(manifest.get("frame_records"), "frame_records")
    if len(raw_records) != EXPECTED_FRAME_COUNT:
        raise InputContractError("frame_records must contain exactly 25 records")

    frames: list[FrameSpec] = []
    frame_ids: set[str] = set()
    for index, raw_record in enumerate(raw_records):
        record = require_dict(raw_record, f"frame_records[{index}]")
        sequence_index = require_int(
            record.get("sequence_index"),
            f"frame_records[{index}].sequence_index",
            minimum=0,
        )
        if sequence_index != index:
            raise InputContractError("frame sequence indexes must be contiguous and ordered from 0")
        frame_id = require_string(record.get("frame_id"), f"frame_records[{index}].frame_id")
        if frame_id in frame_ids:
            raise InputContractError(f"duplicate frame_id: {frame_id}")
        frame_ids.add(frame_id)
        prefix = f"frame_records[{index}]"
        composite = resolve_rgb_asset(
            record.get("selected_composite_rgb"),
            manifest_path=manifest_path,
            label=f"{prefix}.selected_composite_rgb",
        )
        source = resolve_rgb_asset(
            record.get("source_rgb"),
            manifest_path=manifest_path,
            label=f"{prefix}.source_rgb",
        )
        current_core = resolve_mask_asset(
            record.get("current_core_mask"),
            manifest_path=manifest_path,
            label=f"{prefix}.current_core_mask",
        )
        cumulative_core = resolve_mask_asset(
            record.get("cumulative_core_mask"),
            manifest_path=manifest_path,
            label=f"{prefix}.cumulative_core_mask",
        )
        editable = resolve_mask_asset(
            record.get("editable_mask"),
            manifest_path=manifest_path,
            label=f"{prefix}.editable_mask",
        )
        selection_receipt = resolve_asset(
            record.get("selection_receipt"),
            manifest_path=manifest_path,
            label=f"{prefix}.selection_receipt",
        )
        frame = FrameSpec(
            sequence_index=sequence_index,
            frame_id=frame_id,
            selected_composite_rgb=composite,
            source_rgb=source,
            current_core_mask=current_core,
            cumulative_core_mask=cumulative_core,
            editable_mask=editable,
            selection_receipt=selection_receipt,
        )

        current_values = read_mask(current_core)
        cumulative_values = read_mask(cumulative_core)
        editable_values = read_mask(editable)
        if not current_values.any():
            raise InputContractError(f"{prefix}.current_core_mask is empty")
        if not cumulative_values.any():
            raise InputContractError(f"{prefix}.cumulative_core_mask is empty")
        if not editable_values.any():
            raise InputContractError(f"{prefix}.editable_mask is empty")
        if np.any(current_values & ~cumulative_values):
            raise InputContractError(
                f"{prefix}.current_core_mask must be inside cumulative_core_mask"
            )
        if np.any(current_values & ~editable_values):
            raise InputContractError(f"{prefix}.current_core_mask must be inside editable_mask")
        validate_selection_receipt(
            selection_receipt,
            frame_id=frame_id,
            sequence_index=sequence_index,
            source_rgb=source,
            composite_rgb=composite,
            current_core=current_core,
            editable=editable,
        )
        frames.append(frame)

    resolved_frames = tuple(frames)
    ordered_frame_ids = selection_value.get("ordered_frame_ids")
    if ordered_frame_ids != [frame.frame_id for frame in resolved_frames]:
        raise InputContractError(
            "selection_manifest ordered_frame_ids does not match frame_records"
        )
    validate_boundary_report(upstream_boundary_report, resolved_frames)
    return ValidatedManifest(
        path=manifest_path,
        sha256=sha256_file(manifest_path),
        round_index=round_index,
        selection_manifest=selection_manifest,
        upstream_boundary_report=upstream_boundary_report,
        pairing=pairing,
        flow_backend=flow_backend,
        sampling=sampling,
        thresholds=thresholds,
        frames=resolved_frames,
    )


def rgb_gradient(rgb: np.ndarray) -> np.ndarray:
    values = rgb.astype(np.float64)
    horizontal = ndimage.sobel(values, axis=1, mode="reflect") / 8.0
    vertical = ndimage.sobel(values, axis=0, mode="reflect") / 8.0
    return np.sqrt(np.square(horizontal) + np.square(vertical))


def load_frame(spec: FrameSpec, sampling: dict[str, Any]) -> FrameData:
    composite = read_rgb(spec.selected_composite_rgb)
    source = read_rgb(spec.source_rgb)
    current_core = read_mask(spec.current_core_mask)
    cumulative_core = read_mask(spec.cumulative_core_mask)
    editable = read_mask(spec.editable_mask)
    structure = np.ones((3, 3), dtype=bool)
    eroded_core = ndimage.binary_erosion(
        cumulative_core,
        structure=structure,
        iterations=int(sampling["core_erosion_px"]),
        border_value=0,
    )
    inner = ndimage.binary_dilation(
        editable,
        structure=structure,
        iterations=int(sampling["control_ring_inner_px"]),
        border_value=0,
    )
    outer = ndimage.binary_dilation(
        editable,
        structure=structure,
        iterations=int(sampling["control_ring_outer_px"]),
        border_value=0,
    )
    control_ring = outer & ~inner
    return FrameData(
        spec=spec,
        composite=composite,
        source=source,
        current_core=current_core,
        cumulative_core=cumulative_core,
        editable=editable,
        eroded_core=eroded_core,
        control_ring=control_ring,
        gradient=rgb_gradient(composite),
    )


def bilinear_sample(values: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    height, width = values.shape[:2]
    clipped_x = np.clip(x, 0.0, width - 1.0)
    clipped_y = np.clip(y, 0.0, height - 1.0)
    x0 = np.floor(clipped_x).astype(np.int64)
    y0 = np.floor(clipped_y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = clipped_x - x0
    wy = clipped_y - y0
    if values.ndim == 3:
        wx = wx[..., None]
        wy = wy[..., None]
    top = values[y0, x0] * (1.0 - wx) + values[y0, x1] * wx
    bottom = values[y1, x0] * (1.0 - wx) + values[y1, x1] * wx
    return top * (1.0 - wy) + bottom * wy


def nearest_mask_sample(mask: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    columns = np.clip(np.rint(x).astype(np.int64), 0, width - 1)
    rows = np.clip(np.rint(y).astype(np.int64), 0, height - 1)
    return mask[rows, columns]


def validate_flow_array(flow: np.ndarray, dimensions: tuple[int, int], label: str) -> np.ndarray:
    expected_shape = (dimensions[1], dimensions[0], 2)
    result = np.asarray(flow, dtype=np.float32)
    if result.shape != expected_shape:
        raise NotEvaluableError(f"{label} has shape {result.shape}; expected {expected_shape}")
    if not np.isfinite(result).all():
        raise NotEvaluableError(f"{label} contains non-finite values")
    return result


def flow_validity(
    source: FrameData,
    target: FrameData,
    forward: np.ndarray,
    backward: np.ndarray,
    sampling: dict[str, Any],
) -> FlowValidity:
    height, width = source.eroded_core.shape
    rows, columns = np.indices((height, width), dtype=np.float32)
    target_x = columns + forward[..., 0]
    target_y = rows + forward[..., 1]
    in_bounds = (
        (target_x >= 0.0)
        & (target_x <= width - 1.0)
        & (target_y >= 0.0)
        & (target_y <= height - 1.0)
    )
    sampled_backward = bilinear_sample(backward, target_x, target_y)
    fb_error = np.linalg.norm(forward + sampled_backward, axis=2)
    flow_magnitude = np.linalg.norm(forward, axis=2)
    limit = (
        float(sampling["fb_consistency_absolute_px"])
        + float(sampling["fb_consistency_relative_to_flow"]) * flow_magnitude
    )
    fb_consistent = in_bounds & (fb_error <= limit)
    core_target_overlap = in_bounds & nearest_mask_sample(
        target.eroded_core,
        target_x,
        target_y,
    )
    control_target_overlap = in_bounds & nearest_mask_sample(
        target.control_ring,
        target_x,
        target_y,
    )
    return FlowValidity(
        target_x=target_x,
        target_y=target_y,
        in_bounds=in_bounds,
        fb_consistent=fb_consistent,
        core_target_overlap=core_target_overlap,
        control_target_overlap=control_target_overlap,
        core_valid=source.eroded_core & core_target_overlap & fb_consistent,
        control_valid=source.control_ring & control_target_overlap & fb_consistent,
        fb_error=fb_error,
    )


def robust_affine_channel(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if x.size < 2:
        raise NotEvaluableError("robust affine exposure fit has fewer than two samples")
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    if float(np.quantile(x, 0.95) - np.quantile(x, 0.05)) < 2.0:
        raise NotEvaluableError("robust affine exposure fit is ill-conditioned")
    design = np.column_stack((x, np.ones_like(x)))
    weights = np.ones_like(x)
    coefficients = np.asarray([1.0, 0.0], dtype=np.float64)
    for _ in range(8):
        weighted_design = design * np.sqrt(weights)[:, None]
        weighted_target = y * np.sqrt(weights)
        if np.linalg.cond(weighted_design) > 1e8:
            raise NotEvaluableError("robust affine exposure fit is ill-conditioned")
        coefficients = np.linalg.lstsq(weighted_design, weighted_target, rcond=None)[0]
        residual = y - (coefficients[0] * x + coefficients[1])
        median = float(np.median(residual))
        scale = 1.4826 * float(np.median(np.abs(residual - median)))
        if scale <= 1e-6:
            break
        normalized = np.abs(residual - median) / (1.345 * scale)
        weights = np.ones_like(normalized)
        large = normalized > 1.0
        weights[large] = 1.0 / normalized[large]
    gain, offset = (float(coefficients[0]), float(coefficients[1]))
    if not 0.5 <= gain <= 2.0 or not -128.0 <= offset <= 128.0:
        raise NotEvaluableError("robust affine exposure fit is outside plausible bounds")
    residual = y - (gain * x + offset)
    return {
        "gain": gain,
        "offset": offset,
        "residual_median": float(np.median(residual)),
        "residual_mad": float(np.median(np.abs(residual - np.median(residual)))),
        "residual_rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "effective_weight_fraction": float(np.mean(weights > 0.5)),
    }


def robust_exposure_fit(
    source_rgb: np.ndarray,
    warped_target_rgb: np.ndarray,
    valid: np.ndarray,
) -> dict[str, Any]:
    source_values = source_rgb[valid].astype(np.float64)
    target_values = warped_target_rgb[valid].astype(np.float64)
    channels = [
        robust_affine_channel(target_values[:, channel], source_values[:, channel])
        for channel in range(3)
    ]
    return {
        "model": "per_channel_robust_affine_target_to_source",
        "fit_region": "flow_consistent_control_ring",
        "sample_count": int(source_values.shape[0]),
        "channels": {name: channels[index] for index, name in enumerate(("red", "green", "blue"))},
    }


def apply_exposure(values: np.ndarray, fit: dict[str, Any]) -> np.ndarray:
    result = values.astype(np.float64).copy()
    for channel, name in enumerate(("red", "green", "blue")):
        parameters = fit["channels"][name]
        result[..., channel] = result[..., channel] * float(parameters["gain"]) + float(
            parameters["offset"]
        )
    return result


def apply_exposure_to_gradient(values: np.ndarray, fit: dict[str, Any]) -> np.ndarray:
    result = values.astype(np.float64).copy()
    for channel, name in enumerate(("red", "green", "blue")):
        result[..., channel] *= abs(float(fit["channels"][name]["gain"]))
    return result


def percentile(values: np.ndarray, quantile: float) -> float:
    if values.size == 0:
        raise NotEvaluableError("metric region is empty")
    return float(np.quantile(values, quantile))


def pair_coverage(source: FrameData, validity: FlowValidity) -> dict[str, Any]:
    core_pixels = int(source.eroded_core.sum())
    control_pixels = int(source.control_ring.sum())
    if core_pixels == 0 or control_pixels == 0:
        raise NotEvaluableError("eroded core or control ring is empty")
    core_overlap = int((source.eroded_core & validity.core_target_overlap).sum())
    control_overlap = int((source.control_ring & validity.control_target_overlap).sum())
    core_fb = int((source.eroded_core & validity.fb_consistent).sum())
    control_fb = int((source.control_ring & validity.fb_consistent).sum())
    core_valid = int(validity.core_valid.sum())
    control_valid = int(validity.control_valid.sum())
    core_fb_values = validity.fb_error[validity.core_valid]
    control_fb_values = validity.fb_error[validity.control_valid]
    return {
        "eroded_core_pixels": core_pixels,
        "control_ring_pixels": control_pixels,
        "target_core_overlap_pixels": core_overlap,
        "target_core_overlap_fraction": core_overlap / core_pixels,
        "target_control_overlap_pixels": control_overlap,
        "target_control_overlap_fraction": control_overlap / control_pixels,
        "fb_consistent_core_pixels": core_fb,
        "fb_consistent_core_fraction": core_fb / core_pixels,
        "fb_consistent_control_pixels": control_fb,
        "fb_consistent_control_fraction": control_fb / control_pixels,
        "valid_core_pixels": core_valid,
        "valid_core_fraction": core_valid / core_pixels,
        "valid_control_pixels": control_valid,
        "valid_control_fraction": control_valid / control_pixels,
        "core_fb_error_p95_px": (percentile(core_fb_values, 0.95) if core_fb_values.size else None),
        "control_fb_error_p95_px": (
            percentile(control_fb_values, 0.95) if control_fb_values.size else None
        ),
    }


def metric_gate(observed: float, threshold: float, *, minimum: bool) -> dict[str, Any]:
    return {
        "observed": float(observed),
        "threshold": float(threshold),
        "passed": observed >= threshold if minimum else observed <= threshold,
    }


def pair_gates(
    coverage: dict[str, float | int],
    metrics: dict[str, float | int],
    thresholds: dict[str, int | float],
) -> dict[str, dict[str, Any]]:
    return {
        "core_valid_fraction_gte": metric_gate(
            float(coverage["valid_core_fraction"]),
            float(thresholds["minimum_core_valid_fraction_per_direction"]),
            minimum=True,
        ),
        "control_valid_fraction_gte": metric_gate(
            float(coverage["valid_control_fraction"]),
            float(thresholds["minimum_control_valid_fraction_per_direction"]),
            minimum=True,
        ),
        "control_color_p95_lte": metric_gate(
            float(metrics["control_color_p95_abs_rgb_delta"]),
            float(thresholds["maximum_control_color_p95_abs_rgb_delta"]),
            minimum=False,
        ),
        "core_color_p95_lte": metric_gate(
            float(metrics["core_color_p95_abs_rgb_delta"]),
            float(thresholds["maximum_core_color_p95_abs_rgb_delta"]),
            minimum=False,
        ),
        "core_excess_over_control_p95_lte": metric_gate(
            float(metrics["core_excess_over_control_p95_abs_rgb_delta"]),
            float(thresholds["maximum_core_excess_over_control_p95_abs_rgb_delta"]),
            minimum=False,
        ),
        "core_gradient_p95_lte": metric_gate(
            float(metrics["core_gradient_p95_abs_rgb_delta"]),
            float(thresholds["maximum_core_gradient_p95_abs_rgb_delta"]),
            minimum=False,
        ),
        "core_bad_pixel_fraction_lte": metric_gate(
            float(metrics["core_bad_pixel_fraction"]),
            float(thresholds["maximum_core_bad_pixel_fraction"]),
            minimum=False,
        ),
    }


def render_error_artifact(
    source: FrameData,
    validity: FlowValidity,
    core_error: np.ndarray | None,
    destination: Path,
) -> None:
    visual = (source.composite.astype(np.float32) * 0.28).astype(np.uint8)
    visual[source.control_ring & ~validity.control_valid] = (170, 120, 20)
    visual[validity.control_valid] = (20, 120, 150)
    visual[source.eroded_core & ~validity.core_valid] = (210, 35, 35)
    if core_error is not None:
        values = np.clip(core_error / 48.0, 0.0, 1.0)
        colors = np.zeros_like(visual)
        colors[..., 0] = (255.0 * values).astype(np.uint8)
        colors[..., 1] = (220.0 * (1.0 - values)).astype(np.uint8)
        colors[..., 2] = 55
        visual[validity.core_valid] = colors[validity.core_valid]
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(visual).save(destination)


def artifact_record(staging_path: Path, final_path: Path, role: str) -> dict[str, Any]:
    return {
        "path": str(final_path),
        "sha256": sha256_file(staging_path),
        "bytes": staging_path.stat().st_size,
        "role": role,
        "review_only": True,
    }


def evaluate_directed_pair(
    source: FrameData,
    target: FrameData,
    forward: np.ndarray,
    backward: np.ndarray,
    *,
    sampling: dict[str, Any],
    thresholds: dict[str, int | float],
    artifact_staging: Path,
    artifact_final: Path,
) -> PairRuntime:
    validity = flow_validity(source, target, forward, backward, sampling)
    direction_id = f"{source.spec.sequence_index:04d}_to_{target.spec.sequence_index:04d}"
    record: dict[str, Any] = {
        "source_sequence_index": source.spec.sequence_index,
        "source_frame_id": source.spec.frame_id,
        "target_sequence_index": target.spec.sequence_index,
        "target_frame_id": target.spec.frame_id,
        "direction_id": direction_id,
        "evaluable": False,
        "passed": False,
    }
    core_error_image: np.ndarray | None = None
    exposure_fit: dict[str, Any] | None = None
    try:
        coverage = pair_coverage(source, validity)
        record["coverage"] = coverage
        minimum_pixels = int(thresholds["minimum_region_pixels"])
        if int(coverage["valid_core_pixels"]) < minimum_pixels:
            raise NotEvaluableError(
                f"valid core has {coverage['valid_core_pixels']} pixels; {minimum_pixels} required"
            )
        if int(coverage["valid_control_pixels"]) < minimum_pixels:
            raise NotEvaluableError(
                "valid control ring has "
                f"{coverage['valid_control_pixels']} pixels; {minimum_pixels} required"
            )
        warped_target = bilinear_sample(
            target.composite.astype(np.float64),
            validity.target_x,
            validity.target_y,
        )
        exposure_fit = robust_exposure_fit(
            source.composite,
            warped_target,
            validity.control_valid,
        )
        corrected_target = apply_exposure(warped_target, exposure_fit)
        per_pixel_color = np.mean(
            np.abs(source.composite.astype(np.float64) - corrected_target),
            axis=2,
        )
        warped_target_gradient = bilinear_sample(
            target.gradient,
            validity.target_x,
            validity.target_y,
        )
        corrected_gradient = apply_exposure_to_gradient(warped_target_gradient, exposure_fit)
        per_pixel_gradient = np.mean(
            np.abs(source.gradient - corrected_gradient),
            axis=2,
        )
        core_values = per_pixel_color[validity.core_valid]
        control_values = per_pixel_color[validity.control_valid]
        core_p95 = percentile(core_values, 0.95)
        control_p95 = percentile(control_values, 0.95)
        metrics = {
            "control_color_mean_abs_rgb_delta": float(control_values.mean()),
            "control_color_p95_abs_rgb_delta": control_p95,
            "core_color_mean_abs_rgb_delta": float(core_values.mean()),
            "core_color_p95_abs_rgb_delta": core_p95,
            "core_excess_over_control_p95_abs_rgb_delta": core_p95 - control_p95,
            "core_gradient_mean_abs_rgb_delta": float(
                per_pixel_gradient[validity.core_valid].mean()
            ),
            "core_gradient_p95_abs_rgb_delta": percentile(
                per_pixel_gradient[validity.core_valid],
                0.95,
            ),
            "bad_pixel_abs_rgb_delta": float(thresholds["bad_pixel_abs_rgb_delta"]),
            "core_bad_pixel_fraction": float(
                np.mean(core_values > float(thresholds["bad_pixel_abs_rgb_delta"]))
            ),
        }
        gates = pair_gates(coverage, metrics, thresholds)
        record.update(
            {
                "evaluable": True,
                "exposure_fit": exposure_fit,
                "metrics": metrics,
                "gates": gates,
                "passed": all(bool(gate["passed"]) for gate in gates.values()),
            }
        )
        core_error_image = per_pixel_color
    except NotEvaluableError as error:
        record["not_evaluable_reason"] = str(error)

    artifact_name = f"pair_{direction_id}.png"
    staging_path = artifact_staging / "pairs" / artifact_name
    final_path = artifact_final / "pairs" / artifact_name
    render_error_artifact(source, validity, core_error_image, staging_path)
    record["artifacts"] = {
        "flow_validity_and_color_error": artifact_record(
            staging_path,
            final_path,
            "review_only_pair_flow_validity_and_color_error",
        )
    }
    return PairRuntime(record=record, validity=validity, exposure_fit=exposure_fit)


def triplet_gates(
    coverage: dict[str, float | int],
    metrics: dict[str, float | int],
    thresholds: dict[str, int | float],
) -> dict[str, dict[str, Any]]:
    return {
        "triplet_valid_fraction_gte": metric_gate(
            float(coverage["valid_core_fraction"]),
            float(thresholds["minimum_triplet_valid_fraction"]),
            minimum=True,
        ),
        "triplet_second_difference_p95_lte": metric_gate(
            float(metrics["core_second_difference_p95_abs_rgb_delta"]),
            float(thresholds["maximum_triplet_second_difference_p95_abs_rgb_delta"]),
            minimum=False,
        ),
        "triplet_excess_over_control_p95_lte": metric_gate(
            float(metrics["core_excess_over_control_p95_abs_rgb_delta"]),
            float(thresholds["maximum_triplet_excess_over_control_p95_abs_rgb_delta"]),
            minimum=False,
        ),
    }


def render_triplet_artifact(
    center: FrameData,
    valid_core: np.ndarray,
    valid_control: np.ndarray,
    error: np.ndarray | None,
    destination: Path,
) -> None:
    visual = (center.composite.astype(np.float32) * 0.28).astype(np.uint8)
    visual[center.control_ring & ~valid_control] = (170, 120, 20)
    visual[valid_control] = (20, 120, 150)
    visual[center.eroded_core & ~valid_core] = (210, 35, 35)
    if error is not None:
        values = np.clip(error / 40.0, 0.0, 1.0)
        colors = np.zeros_like(visual)
        colors[..., 0] = (255.0 * values).astype(np.uint8)
        colors[..., 1] = (220.0 * (1.0 - values)).astype(np.uint8)
        colors[..., 2] = 55
        visual[valid_core] = colors[valid_core]
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(visual).save(destination)


def evaluate_triplet(
    previous: FrameData,
    center: FrameData,
    following: FrameData,
    center_to_previous: PairRuntime,
    center_to_following: PairRuntime,
    *,
    thresholds: dict[str, int | float],
    artifact_staging: Path,
    artifact_final: Path,
) -> dict[str, Any]:
    valid_core = center_to_previous.validity.core_valid & center_to_following.validity.core_valid
    valid_control = (
        center_to_previous.validity.control_valid & center_to_following.validity.control_valid
    )
    core_pixels = int(center.eroded_core.sum())
    control_pixels = int(center.control_ring.sum())
    valid_core_pixels = int(valid_core.sum())
    valid_control_pixels = int(valid_control.sum())
    coverage = {
        "eroded_center_core_pixels": core_pixels,
        "center_control_ring_pixels": control_pixels,
        "valid_core_pixels": valid_core_pixels,
        "valid_core_fraction": valid_core_pixels / core_pixels if core_pixels else 0.0,
        "valid_control_pixels": valid_control_pixels,
        "valid_control_fraction": (
            valid_control_pixels / control_pixels if control_pixels else 0.0
        ),
        "requires_both_neighbor_target_core_overlaps": True,
        "requires_both_neighbor_fb_consistency": True,
    }
    record: dict[str, Any] = {
        "previous_sequence_index": previous.spec.sequence_index,
        "previous_frame_id": previous.spec.frame_id,
        "center_sequence_index": center.spec.sequence_index,
        "center_frame_id": center.spec.frame_id,
        "following_sequence_index": following.spec.sequence_index,
        "following_frame_id": following.spec.frame_id,
        "coverage": coverage,
        "evaluable": False,
        "passed": False,
    }
    error_image: np.ndarray | None = None
    try:
        minimum_pixels = int(thresholds["minimum_region_pixels"])
        if valid_core_pixels < minimum_pixels:
            raise NotEvaluableError(
                f"triplet valid core has {valid_core_pixels} pixels; {minimum_pixels} required"
            )
        if valid_control_pixels < minimum_pixels:
            raise NotEvaluableError(
                "triplet valid control ring has "
                f"{valid_control_pixels} pixels; {minimum_pixels} required"
            )
        if center_to_previous.exposure_fit is None or center_to_following.exposure_fit is None:
            raise NotEvaluableError("triplet neighbor exposure fit is unavailable")
        warped_previous = bilinear_sample(
            previous.composite.astype(np.float64),
            center_to_previous.validity.target_x,
            center_to_previous.validity.target_y,
        )
        warped_following = bilinear_sample(
            following.composite.astype(np.float64),
            center_to_following.validity.target_x,
            center_to_following.validity.target_y,
        )
        corrected_previous = apply_exposure(
            warped_previous,
            center_to_previous.exposure_fit,
        )
        corrected_following = apply_exposure(
            warped_following,
            center_to_following.exposure_fit,
        )
        neighbor_mean = 0.5 * (corrected_previous + corrected_following)
        per_pixel = np.mean(
            np.abs(center.composite.astype(np.float64) - neighbor_mean),
            axis=2,
        )
        core_values = per_pixel[valid_core]
        control_values = per_pixel[valid_control]
        core_p95 = percentile(core_values, 0.95)
        control_p95 = percentile(control_values, 0.95)
        metrics = {
            "definition": "abs(I_t - 0.5 * (warp(I_t-1) + warp(I_t+1)))",
            "control_second_difference_mean_abs_rgb_delta": float(control_values.mean()),
            "control_second_difference_p95_abs_rgb_delta": control_p95,
            "core_second_difference_mean_abs_rgb_delta": float(core_values.mean()),
            "core_second_difference_p95_abs_rgb_delta": core_p95,
            "core_excess_over_control_p95_abs_rgb_delta": core_p95 - control_p95,
        }
        gates = triplet_gates(coverage, metrics, thresholds)
        record.update(
            {
                "evaluable": True,
                "metrics": metrics,
                "gates": gates,
                "passed": all(bool(gate["passed"]) for gate in gates.values()),
            }
        )
        error_image = per_pixel
    except NotEvaluableError as error:
        record["not_evaluable_reason"] = str(error)

    triplet_id = (
        f"{previous.spec.sequence_index:04d}_"
        f"{center.spec.sequence_index:04d}_"
        f"{following.spec.sequence_index:04d}"
    )
    artifact_name = f"triplet_{triplet_id}.png"
    staging_path = artifact_staging / "triplets" / artifact_name
    final_path = artifact_final / "triplets" / artifact_name
    render_triplet_artifact(
        center,
        valid_core,
        valid_control,
        error_image,
        staging_path,
    )
    record["artifacts"] = {
        "second_difference_error": artifact_record(
            staging_path,
            final_path,
            "review_only_triplet_second_difference_error",
        )
    }
    return record


class PlainRaftProvider:
    """Lazy one-time loader for the audited original RAFT checkpoint."""

    def __init__(self, config: dict[str, Any]):
        repository = Path(config["repository_path"])
        commit = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if commit != EXPECTED_RAFT_COMMIT:
            raise NotEvaluableError(
                f"plain RAFT checkout is {commit}; expected {EXPECTED_RAFT_COMMIT}"
            )
        checkpoint = Path(config["checkpoint"]["path"])
        if sha256_file(checkpoint) != EXPECTED_RAFT_CHECKPOINT_SHA256:
            raise NotEvaluableError("plain RAFT checkpoint changed after manifest validation")

        try:
            import torch
        except ImportError as error:
            raise NotEvaluableError("PyTorch is unavailable for plain RAFT") from error
        if not torch.cuda.is_available():
            raise NotEvaluableError("CUDA is unavailable for full-resolution plain RAFT")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False

        core_path = repository / "core"
        if not core_path.is_dir():
            raise NotEvaluableError(f"plain RAFT core directory is missing: {core_path}")
        sys.path.insert(0, str(core_path))
        try:
            raft_module = importlib.import_module("raft")
        except ImportError as error:
            raise NotEvaluableError(
                "could not import plain RAFT from its audited checkout"
            ) from error
        finally:
            if sys.path[0] == str(core_path):
                sys.path.pop(0)
        module_file = getattr(raft_module, "__file__", None)
        if module_file is None or not Path(module_file).resolve().is_relative_to(
            core_path.resolve()
        ):
            raise NotEvaluableError(
                "Python resolved 'raft' outside the audited plain RAFT checkout"
            )

        model_args = SimpleNamespace(
            small=False,
            mixed_precision=False,
            alternate_corr=False,
            dropout=0.0,
        )
        model = raft_module.RAFT(model_args)
        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(checkpoint, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise NotEvaluableError("plain RAFT checkpoint does not contain a state dictionary")
        normalized = {key.removeprefix("module."): value for key, value in state.items()}
        model.load_state_dict(normalized, strict=True)
        self._torch = torch
        self._device = torch.device(config["device"])
        self._model = model.float().to(self._device).eval()
        self._iterations = int(config["iterations"])
        self.metadata = {
            "kind": "plain_raft",
            "repository_commit_verified": commit,
            "checkpoint_sha256_verified": sha256_file(checkpoint),
            "precision": "float32",
            "tf32_enabled": False,
            "iterations": self._iterations,
            "flow_completion": False,
            "model_load_count": 1,
            "test_double": False,
        }

    def estimate(self, source_rgb: np.ndarray, target_rgb: np.ndarray) -> np.ndarray:
        torch = self._torch
        source = (
            torch.from_numpy(writable_c_contiguous_rgb(source_rgb))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=self._device, dtype=torch.float32)
        )
        target = (
            torch.from_numpy(writable_c_contiguous_rgb(target_rgb))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=self._device, dtype=torch.float32)
        )
        with torch.inference_mode():
            _flow_low, flow_up = self._model(
                source,
                target,
                iters=self._iterations,
                test_mode=True,
            )
        return flow_up[0].permute(1, 2, 0).float().cpu().numpy()


def make_forbidden_claims() -> dict[str, bool]:
    return {
        "pbr_used_as_pass_evidence": False,
        "da3_used_as_pass_evidence": False,
        "flow_completion_used": False,
        "semantic_correctness_claimed": False,
        "clean_plate_promotion_claimed": False,
        "geometry_reconstruction_readiness_claimed": False,
    }


def invalid_report(manifest_path: Path, error: Exception) -> dict[str, Any]:
    input_record: dict[str, Any] = {"path": str(manifest_path.expanduser().resolve())}
    resolved = manifest_path.expanduser().resolve()
    if resolved.is_file():
        input_record.update(
            {
                "sha256": sha256_file(resolved),
                "bytes": resolved.stat().st_size,
            }
        )
    return {
        "schema_version": 1,
        "kind": REPORT_KIND,
        "status": "invalid_input",
        "created_at": utc_now(),
        "review_only": True,
        "promotion_approved": False,
        "input_manifest": input_record,
        "error": {"type": type(error).__name__, "message": str(error)},
        "forbidden_claims": make_forbidden_claims(),
    }


def report_base(validated: ValidatedManifest) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": REPORT_KIND,
        "status": "not_evaluable",
        "created_at": utc_now(),
        "review_only": True,
        "promotion_approved": False,
        "round_index": validated.round_index,
        "input_manifest": {
            "path": str(validated.path),
            "sha256": validated.sha256,
            "bytes": validated.path.stat().st_size,
        },
        "dependencies": {
            "selection_manifest": validated.selection_manifest.record(),
            "upstream_boundary_report": validated.upstream_boundary_report.record(),
        },
        "frame_count": len(validated.frames),
        "ordered_frame_ids": [frame.frame_id for frame in validated.frames],
        "frame_records": [
            {
                "sequence_index": frame.sequence_index,
                "frame_id": frame.frame_id,
                "inputs": {
                    "selected_composite_rgb": frame.selected_composite_rgb.record(),
                    "source_rgb": frame.source_rgb.record(),
                    "current_core_mask": frame.current_core_mask.record(),
                    "cumulative_core_mask": frame.cumulative_core_mask.record(),
                    "editable_mask": frame.editable_mask.record(),
                    "selection_receipt": frame.selection_receipt.record(),
                },
            }
            for frame in validated.frames
        ],
        "pairing": validated.pairing,
        "flow_backend": validated.flow_backend,
        "sampling": validated.sampling,
        "thresholds": validated.thresholds,
        "metric_contract": {
            "core": "cumulative_core_mask eroded by four pixels",
            "target_core_overlap": (
                "source core pixel warped by forward flow must land in the target eroded core"
            ),
            "fb_consistency": "norm(F + warp(B, p + F)) <= 1.5 + 0.01 * norm(F)",
            "control_ring": "dilate(editable, 32) minus dilate(editable, 8)",
            "exposure_fit": "per-channel robust affine target-to-source fit on valid control ring",
            "triplet": "abs(I_t - 0.5 * (warp(I_t-1) + warp(I_t+1)))",
        },
        "forbidden_claims": make_forbidden_claims(),
    }


def failed_gate_names(record: dict[str, Any]) -> list[str]:
    gates = record.get("gates")
    if not isinstance(gates, dict):
        return []
    return sorted(
        str(name)
        for name, gate in gates.items()
        if isinstance(gate, dict) and gate.get("passed") is not True
    )


def temporal_next_action(report: dict[str, Any]) -> dict[str, Any]:
    pair_records = report.get("pair_records", [])
    triplet_records = report.get("triplet_records", [])
    if not isinstance(pair_records, list):
        pair_records = []
    if not isinstance(triplet_records, list):
        triplet_records = []

    failed_pairs = [
        {
            "direction_id": record["direction_id"],
            "source_frame_id": record["source_frame_id"],
            "target_frame_id": record["target_frame_id"],
            "failed_gates": failed_gate_names(record),
        }
        for record in pair_records
        if isinstance(record, dict)
        and record.get("evaluable") is True
        and record.get("passed") is not True
    ]
    failed_triplets = [
        {
            "center_frame_id": record["center_frame_id"],
            "previous_frame_id": record["previous_frame_id"],
            "following_frame_id": record["following_frame_id"],
            "failed_gates": failed_gate_names(record),
        }
        for record in triplet_records
        if isinstance(record, dict)
        and record.get("evaluable") is True
        and record.get("passed") is not True
    ]
    not_evaluable_pairs = [
        {
            "direction_id": record["direction_id"],
            "source_frame_id": record["source_frame_id"],
            "target_frame_id": record["target_frame_id"],
            "reason": record.get("not_evaluable_reason"),
        }
        for record in pair_records
        if isinstance(record, dict) and record.get("evaluable") is not True
    ]
    not_evaluable_triplets = [
        {
            "center_frame_id": record["center_frame_id"],
            "previous_frame_id": record["previous_frame_id"],
            "following_frame_id": record["following_frame_id"],
            "reason": record.get("not_evaluable_reason"),
        }
        for record in triplet_records
        if isinstance(record, dict) and record.get("evaluable") is not True
    ]

    status = report.get("status")
    if status == "technical_passed_temporal_only":
        action = "run_semantic_residual_pbr_da3_and_cross_view_review_before_any_promotion"
        reason = (
            "Review-only temporal gates passed, but this QA scope still does not prove "
            "object-free semantics, PBR consistency, DA3 geometry, or promotion readiness."
        )
    elif status == "technical_failed":
        action = "repair_temporal_flicker_or_regenerate_clean_plate_candidates"
        reason = (
            "Temporal flow evidence was evaluable, but one or more directed pairs or "
            "triplets failed strict color/gradient consistency gates."
        )
    elif status == "not_evaluable":
        action = "repair_temporal_evidence_before_retesting"
        reason = (
            "Temporal QA could not evaluate the strict review contract; fix runtime, flow "
            "validity, control-ring conditioning, or candidate selection before promotion."
        )
    else:
        action = "repair_temporal_input_contract_before_retesting"
        reason = "Temporal QA input validation failed before review-only flow evidence was run."

    return {
        "action": action,
        "reason": reason,
        "promotion_approved": False,
        "failed_pair_ids": [item["direction_id"] for item in failed_pairs],
        "failed_pairs": failed_pairs,
        "failed_triplet_center_frame_ids": [
            item["center_frame_id"] for item in failed_triplets
        ],
        "failed_triplets": failed_triplets,
        "not_evaluable_pair_ids": [
            item["direction_id"] for item in not_evaluable_pairs
        ],
        "not_evaluable_pairs": not_evaluable_pairs,
        "not_evaluable_triplet_center_frame_ids": [
            item["center_frame_id"] for item in not_evaluable_triplets
        ],
        "not_evaluable_triplets": not_evaluable_triplets,
    }


def evaluate(
    args: argparse.Namespace,
    *,
    flow_provider_factory: FlowProviderFactory | None = None,
) -> dict[str, Any]:
    output_report = Path(args.output_report).expanduser().resolve()
    manifest_path = Path(args.input_manifest)
    try:
        validated = validate_manifest(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        report = invalid_report(manifest_path, error)
        atomic_write_json(output_report, report)
        return report

    report = report_base(validated)
    artifact_final = Path(args.artifacts_dir).expanduser().resolve()
    if artifact_final.exists():
        report["runtime_error"] = f"artifacts_dir already exists: {artifact_final}"
        atomic_write_json(output_report, report)
        return report
    artifact_final.parent.mkdir(parents=True, exist_ok=True)
    artifact_staging = Path(
        tempfile.mkdtemp(
            prefix=f".{artifact_final.name}.staging-",
            dir=artifact_final.parent,
        )
    )

    pair_records: list[dict[str, Any]] = []
    triplet_records: list[dict[str, Any]] = []
    provider: FlowProvider | None = None
    try:
        factory = flow_provider_factory or PlainRaftProvider
        provider = factory(validated.flow_backend)
        report["flow_runtime"] = dict(provider.metadata)
        previous_frame: FrameData | None = None
        current_frame = load_frame(validated.frames[0], validated.sampling)
        previous_reverse_runtime: PairRuntime | None = None

        for adjacent_index in range(EXPECTED_FRAME_COUNT - 1):
            next_frame = load_frame(
                validated.frames[adjacent_index + 1],
                validated.sampling,
            )
            forward = validate_flow_array(
                provider.estimate(current_frame.composite, next_frame.composite),
                EXPECTED_RESOLUTION,
                f"flow {current_frame.spec.frame_id}->{next_frame.spec.frame_id}",
            )
            backward = validate_flow_array(
                provider.estimate(next_frame.composite, current_frame.composite),
                EXPECTED_RESOLUTION,
                f"flow {next_frame.spec.frame_id}->{current_frame.spec.frame_id}",
            )
            forward_runtime = evaluate_directed_pair(
                current_frame,
                next_frame,
                forward,
                backward,
                sampling=validated.sampling,
                thresholds=validated.thresholds,
                artifact_staging=artifact_staging,
                artifact_final=artifact_final,
            )
            reverse_runtime = evaluate_directed_pair(
                next_frame,
                current_frame,
                backward,
                forward,
                sampling=validated.sampling,
                thresholds=validated.thresholds,
                artifact_staging=artifact_staging,
                artifact_final=artifact_final,
            )
            pair_records.extend((forward_runtime.record, reverse_runtime.record))

            if previous_frame is not None and previous_reverse_runtime is not None:
                triplet_records.append(
                    evaluate_triplet(
                        previous_frame,
                        current_frame,
                        next_frame,
                        previous_reverse_runtime,
                        forward_runtime,
                        thresholds=validated.thresholds,
                        artifact_staging=artifact_staging,
                        artifact_final=artifact_final,
                    )
                )
            previous_frame = current_frame
            current_frame = next_frame
            previous_reverse_runtime = reverse_runtime

        if len(pair_records) != 48 or len(triplet_records) != 23:
            raise NotEvaluableError("runtime did not produce the required 48 pairs and 23 triplets")
    except Exception as error:  # Flow/model failures are evidence gaps, not input passes.
        report["runtime_error"] = f"{type(error).__name__}: {error}"
    finally:
        if artifact_staging.exists():
            artifact_staging.replace(artifact_final)

    report["pair_records"] = pair_records
    report["triplet_records"] = triplet_records
    pair_not_evaluable = sum(not record.get("evaluable", False) for record in pair_records)
    triplet_not_evaluable = sum(not record.get("evaluable", False) for record in triplet_records)
    failed_pairs = sum(
        bool(record.get("evaluable")) and not bool(record.get("passed")) for record in pair_records
    )
    failed_triplets = sum(
        bool(record.get("evaluable")) and not bool(record.get("passed"))
        for record in triplet_records
    )
    report["counts"] = {
        "expected_directed_pairs": 48,
        "observed_directed_pairs": len(pair_records),
        "not_evaluable_directed_pairs": pair_not_evaluable,
        "failed_directed_pairs": failed_pairs,
        "expected_triplets": 23,
        "observed_triplets": len(triplet_records),
        "not_evaluable_triplets": triplet_not_evaluable,
        "failed_triplets": failed_triplets,
    }
    report["gates"] = {
        "all_directed_pairs_evaluable": {
            "observed": len(pair_records) - pair_not_evaluable,
            "threshold": 48,
            "passed": len(pair_records) == 48 and pair_not_evaluable == 0,
        },
        "all_triplets_evaluable": {
            "observed": len(triplet_records) - triplet_not_evaluable,
            "threshold": 23,
            "passed": len(triplet_records) == 23 and triplet_not_evaluable == 0,
        },
        "failed_directed_pairs_lte": metric_gate(
            float(failed_pairs),
            float(validated.thresholds["maximum_failed_directed_pairs"]),
            minimum=False,
        ),
        "failed_triplets_lte": metric_gate(
            float(failed_triplets),
            float(validated.thresholds["maximum_failed_triplets"]),
            minimum=False,
        ),
    }
    has_runtime_error = "runtime_error" in report
    if (
        has_runtime_error
        or pair_not_evaluable
        or triplet_not_evaluable
        or len(pair_records) != 48
        or len(triplet_records) != 23
    ):
        report["status"] = "not_evaluable"
    elif failed_pairs or failed_triplets:
        report["status"] = "technical_failed"
    else:
        report["status"] = "technical_passed_temporal_only"
    report["promotion_approved"] = False
    report["next_action"] = temporal_next_action(report)
    report["artifacts"] = {
        "root": str(artifact_final),
        "review_only": True,
        "pair_artifact_count": sum(len(record.get("artifacts", {})) for record in pair_records),
        "triplet_artifact_count": sum(
            len(record.get("artifacts", {})) for record in triplet_records
        ),
    }
    report["evidence_set_sha256"] = canonical_sha256(
        {
            "input_manifest_sha256": validated.sha256,
            "pair_artifacts": [record.get("artifacts") for record in pair_records],
            "triplet_artifacts": [record.get("artifacts") for record in triplet_records],
            "status": report["status"],
        }
    )
    atomic_write_json(output_report, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate(args)
    print(json.dumps(report, ensure_ascii=True, indent=2))
    if report["status"] == "technical_passed_temporal_only":
        raise SystemExit(0)
    if report["status"] == "invalid_input":
        raise SystemExit(3)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
