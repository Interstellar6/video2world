#!/usr/bin/env python3
"""Normalize one hash-bound AOT-GAN fixed sequence into generic inpaint receipts.

The AOT experiment runner predates the model-agnostic layered clean-plate
contract.  This bridge does not rerun or reinterpret the model.  It verifies
the immutable AOT receipt, report, artifact manifest, upstream adaptive matte,
model weight, every frame input/output, and the exact alpha-compositing formula.
It then materializes the masks and masked raw candidates required by the shared
candidate-selection and boundary-QA tools.

The result remains review-only.  It cannot authorize a deeper peel round by
itself; boundary, temporal, semantic, and human review evidence are still
required by ``materialize_accepted_clean_plate_next_round.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

INPUT_RECEIPT_KIND = "video2world.aotgan_fixed25_batch_receipt"
INPUT_REPORT_KIND = "video2world.aotgan_fixed25_adaptive_alpha_batch"
INPUT_STATUS = "generated_fixed25_pending_temporal_and_3d_consistency_review"
OUTPUT_INPUT_KIND = "video2world.aotgan_clean_plate_batch_normalization_input"
OUTPUT_RECEIPT_KIND = "video2world.inpaint_clean_plate_batch_run"
OUTPUT_FRAME_RECEIPT_KIND = "video2world.inpaint_clean_plate_frame_run"
OUTPUT_STATUS = "materialized_inpaint_candidate_pending_review"
SCHEMA_VERSION = 1
EXPECTED_FRAME_COUNT = 25
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
FRAME_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class AOTBatchMaterializationError(ValueError):
    """The AOT evidence is incomplete, changed, or pixel-inconsistent."""


@dataclass(frozen=True)
class Asset:
    path: Path
    sha256: str
    size_bytes: int


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AOTBatchMaterializationError(message)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AOTBatchMaterializationError(f"cannot read JSON {path}: {exc}") from exc
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


def require_sha256(value: Any, label: str) -> str:
    require(
        isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None,
        f"{label} must be a lowercase SHA-256 digest",
    )
    return value


def require_asset_record(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an asset object")
    return value


def resolve_asset(value: Any, *, relative_to: Path, label: str) -> Asset:
    record = require_asset_record(value, label)
    raw_path = record.get("path")
    require(isinstance(raw_path, str) and bool(raw_path), f"{label}.path is missing")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    path = path.resolve()
    require(path.is_file(), f"{label} does not exist: {path}")
    expected_sha = require_sha256(record.get("sha256"), f"{label}.sha256")
    actual_sha, actual_size = sha256_file(path)
    require(actual_sha == expected_sha, f"{label} SHA-256 mismatch")
    declared_size = record.get("size_bytes", record.get("bytes"))
    if declared_size is not None:
        require(
            isinstance(declared_size, int)
            and not isinstance(declared_size, bool)
            and declared_size >= 0,
            f"{label}.bytes must be a non-negative integer",
        )
        require(actual_size == declared_size, f"{label} byte size mismatch")
    return Asset(path, actual_sha, actual_size)


def asset_record(asset: Asset, *, path: str | None = None) -> dict[str, Any]:
    return {
        "path": path if path is not None else str(asset.path),
        "sha256": asset.sha256,
        "size_bytes": asset.size_bytes,
    }


def input_asset(asset: Asset) -> dict[str, Any]:
    return asset_record(asset)


def output_asset(path: Path, *, root: Path) -> dict[str, Any]:
    digest, size = sha256_file(path)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": digest,
        "size_bytes": size,
    }


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_rgb(asset: Asset, *, label: str) -> np.ndarray:
    try:
        with Image.open(asset.path) as image:
            image.load()
            require(image.mode == "RGB", f"{label} mode must be RGB, found {image.mode!r}")
            return np.array(image, dtype=np.uint8, copy=True)
    except OSError as exc:
        raise AOTBatchMaterializationError(f"cannot decode {label}: {asset.path}") from exc


def load_luma(asset: Asset, *, label: str) -> np.ndarray:
    try:
        with Image.open(asset.path) as image:
            image.load()
            require(image.mode in {"1", "L"}, f"{label} mode must be 1 or L")
            return np.array(image.convert("L"), dtype=np.uint8, copy=True)
    except OSError as exc:
        raise AOTBatchMaterializationError(f"cannot decode {label}: {asset.path}") from exc


def save_rgb(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values.astype(np.uint8)).save(path, format="PNG", optimize=True)


def save_mask(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values.astype(np.uint8)).save(path, format="PNG", optimize=True)


def artifact_index(asset: Asset) -> dict[str, dict[str, Any]]:
    manifest = read_json(asset.path)
    require(manifest.get("schema_version", 1) == 1, "AOT artifact manifest schema is invalid")
    files = manifest.get("files")
    require(isinstance(files, list) and files, "AOT artifact manifest.files must be non-empty")
    indexed: dict[str, dict[str, Any]] = {}
    for index, raw_record in enumerate(files):
        record = require_asset_record(raw_record, f"artifact manifest.files[{index}]")
        path = record.get("path")
        require(isinstance(path, str) and bool(path), f"artifact {index} path is invalid")
        require(path not in indexed, f"duplicate AOT artifact path: {path}")
        indexed[path] = record
    return indexed


def verify_manifest_member(
    asset: Asset,
    *,
    relative_to: Path,
    indexed: dict[str, dict[str, Any]],
    label: str,
) -> None:
    try:
        relative = asset.path.relative_to(relative_to).as_posix()
    except ValueError as exc:
        raise AOTBatchMaterializationError(f"{label} escapes the AOT output package") from exc
    record = indexed.get(relative)
    require(record is not None, f"AOT artifact manifest lacks {relative}")
    require(record.get("sha256") == asset.sha256, f"AOT manifest hash differs for {relative}")
    require(record.get("bytes") == asset.size_bytes, f"AOT manifest size differs for {relative}")


def prepare_destination(output_dir: str | Path) -> Path:
    candidate = Path(output_dir).expanduser()
    require(not candidate.is_symlink(), f"output_dir cannot be a symlink: {candidate}")
    destination = candidate.resolve()
    require(not destination.exists(), f"refusing to overwrite output_dir: {destination}")
    return destination


def materialize_aotgan_clean_plate_batch(
    source_receipt_path: str | Path,
    output_dir: str | Path,
    *,
    expected_source_receipt_sha256: str | None = None,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """Verify and atomically normalize a complete AOT fixed-25 result."""

    source_path = Path(source_receipt_path).expanduser().resolve()
    require(source_path.is_file(), f"source AOT receipt does not exist: {source_path}")
    source_sha, source_size = sha256_file(source_path)
    if expected_source_receipt_sha256 is not None:
        require(
            source_sha
            == require_sha256(
                expected_source_receipt_sha256,
                "expected_source_receipt_sha256",
            ),
            "source AOT receipt SHA-256 mismatch",
        )
    source_asset = Asset(source_path, source_sha, source_size)
    source_receipt = read_json(source_path)
    require(source_receipt.get("schema_version") == 1, "AOT receipt schema is invalid")
    require(source_receipt.get("kind") == INPUT_RECEIPT_KIND, "AOT receipt kind is invalid")
    require(source_receipt.get("status") == INPUT_STATUS, "AOT receipt status is invalid")
    require(source_receipt.get("promotion_approved") is False, "AOT receipt exceeded authority")
    require(source_receipt.get("r2_allowed") is False, "AOT receipt already authorizes R2")

    source_root = source_path.parent
    report_asset = resolve_asset(
        source_receipt.get("report"), relative_to=source_root, label="AOT receipt.report"
    )
    manifest_asset = resolve_asset(
        source_receipt.get("artifact_manifest"),
        relative_to=source_root,
        label="AOT receipt.artifact_manifest",
    )
    runner_asset = resolve_asset(
        source_receipt.get("runner"), relative_to=source_root, label="AOT receipt.runner"
    )
    indexed_artifacts = artifact_index(manifest_asset)
    verify_manifest_member(
        report_asset,
        relative_to=source_root,
        indexed=indexed_artifacts,
        label="AOT report",
    )

    report = read_json(report_asset.path)
    require(report.get("schema_version") == 1, "AOT report schema is invalid")
    require(report.get("kind") == INPUT_REPORT_KIND, "AOT report kind is invalid")
    require(report.get("status") == INPUT_STATUS, "AOT report status is invalid")
    require(report.get("promotion_approved") is False, "AOT report exceeded authority")
    fixed = report.get("fixed25")
    require(isinstance(fixed, dict), "AOT report.fixed25 must be an object")
    ordered_ids = fixed.get("frame_ids")
    require(
        isinstance(ordered_ids, list) and len(ordered_ids) == EXPECTED_FRAME_COUNT,
        "AOT report must contain exactly 25 frame IDs",
    )
    require(len(set(ordered_ids)) == EXPECTED_FRAME_COUNT, "AOT frame IDs are not unique")
    require(
        all(isinstance(value, str) and FRAME_ID_PATTERN.fullmatch(value) for value in ordered_ids),
        "AOT frame IDs contain unsafe values",
    )
    require(fixed.get("frame_count") == EXPECTED_FRAME_COUNT, "AOT fixed25 frame_count mismatch")
    require(
        fixed.get("outside_alpha_zero_changed_values_total") == 0,
        "AOT report declares changes outside alpha",
    )
    raw_frames = report.get("frames")
    require(
        isinstance(raw_frames, list) and len(raw_frames) == EXPECTED_FRAME_COUNT,
        "AOT report.frames must contain exactly 25 records",
    )
    require(
        [frame.get("frame_id") if isinstance(frame, dict) else None for frame in raw_frames]
        == ordered_ids,
        "AOT report frame order differs from fixed25.frame_ids",
    )

    model = report.get("model")
    require(isinstance(model, dict), "AOT report.model must be an object")
    source_commit = model.get("source_commit")
    require(
        isinstance(source_commit, str) and COMMIT_PATTERN.fullmatch(source_commit) is not None,
        "AOT source commit is invalid",
    )
    model_weight = resolve_asset(
        model.get("weight"), relative_to=report_asset.path.parent, label="AOT model weight"
    )

    upstream = report.get("upstream_adaptive_mask")
    require(isinstance(upstream, dict), "AOT upstream_adaptive_mask must be an object")
    upstream_receipt = resolve_asset(
        upstream.get("receipt"), relative_to=report_asset.path.parent, label="adaptive receipt"
    )
    upstream_report = resolve_asset(
        upstream.get("report"), relative_to=report_asset.path.parent, label="adaptive report"
    )
    upstream_manifest = resolve_asset(
        upstream.get("artifact_manifest"),
        relative_to=report_asset.path.parent,
        label="adaptive artifact manifest",
    )
    adaptive_receipt_value = read_json(upstream_receipt.path)
    require(
        adaptive_receipt_value.get("status")
        == "technical_passed_mask_refinement_only_not_clean_plate",
        "upstream adaptive matte did not pass its technical-only gate",
    )
    bound_adaptive_report = resolve_asset(
        adaptive_receipt_value.get("report"),
        relative_to=upstream_receipt.path.parent,
        label="adaptive receipt.report",
    )
    bound_adaptive_manifest = resolve_asset(
        adaptive_receipt_value.get("artifact_manifest"),
        relative_to=upstream_receipt.path.parent,
        label="adaptive receipt.artifact_manifest",
    )
    require(
        bound_adaptive_report.sha256 == upstream_report.sha256,
        "adaptive report binding differs",
    )
    require(
        bound_adaptive_manifest.sha256 == upstream_manifest.sha256,
        "adaptive manifest binding differs",
    )

    timestamp = created_at or datetime.now(timezone.utc)  # noqa: UP017
    require(
        timestamp.tzinfo is not None and timestamp.utcoffset() is not None,
        "created_at must include timezone information",
    )
    destination = prepare_destination(output_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        producer_source = Path(__file__).resolve()
        require(producer_source.is_file(), f"normalization producer is missing: {producer_source}")
        producer_path = staging / "provenance" / "materialize_aotgan_clean_plate_batch.py"
        producer_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(producer_source, producer_path)
        producer_asset = output_asset(producer_path, root=staging)
        normalization_input = {
            "schema_version": 1,
            "kind": OUTPUT_INPUT_KIND,
            "created_at": timestamp.isoformat(),
            "source_aot_receipt": asset_record(source_asset),
            "source_aot_report": asset_record(report_asset),
            "source_aot_artifact_manifest": asset_record(manifest_asset),
            "source_aot_runner": asset_record(runner_asset),
            "normalization_producer": producer_asset,
            "upstream_adaptive_receipt": asset_record(upstream_receipt),
            "upstream_adaptive_report": asset_record(upstream_report),
            "upstream_adaptive_artifact_manifest": asset_record(upstream_manifest),
            "model": {
                "family": "AOT-GAN",
                "source_commit": source_commit,
                "weight": asset_record(model_weight),
            },
            "ordered_frame_ids": list(ordered_ids),
            "normalization_contract": {
                "editable_mask": "adaptive_alpha > 0",
                "ownership_core": "guaranteed_foreground_core > 0",
                "context_mask": "editable_mask",
                "masked_raw_candidate": "source outside context; AOT prediction inside context",
                "composite_formula": "(source*(255-alpha)+prediction*alpha+127)//255 uint16",
                "quality_acceptance_performed": False,
            },
        }
        input_path = staging / "normalization_input.json"
        atomic_write_json(input_path, normalization_input)
        input_manifest = Asset(input_path, *sha256_file(input_path))

        frame_records: list[dict[str, Any]] = []
        frame_receipt_hashes: list[str] = []
        for sequence_index, raw_frame in enumerate(raw_frames):
            require(isinstance(raw_frame, dict), f"AOT frame {sequence_index} must be an object")
            frame_id = str(ordered_ids[sequence_index])
            source = resolve_asset(
                raw_frame.get("source"),
                relative_to=report_asset.path.parent,
                label=f"{frame_id} source",
            )
            alpha_asset = resolve_asset(
                raw_frame.get("adaptive_alpha"),
                relative_to=report_asset.path.parent,
                label=f"{frame_id} adaptive alpha",
            )
            physical_mask = resolve_asset(
                raw_frame.get("physical_sam_mask"),
                relative_to=report_asset.path.parent,
                label=f"{frame_id} physical SAM mask",
            )
            core_asset = resolve_asset(
                raw_frame.get("guaranteed_core"),
                relative_to=report_asset.path.parent,
                label=f"{frame_id} guaranteed core",
            )
            collar_asset = resolve_asset(
                raw_frame.get("uncertain_collar"),
                relative_to=report_asset.path.parent,
                label=f"{frame_id} uncertain collar",
            )
            prediction_asset = resolve_asset(
                raw_frame.get("raw_prediction"),
                relative_to=report_asset.path.parent,
                label=f"{frame_id} raw prediction",
            )
            composite_asset = resolve_asset(
                raw_frame.get("composite"),
                relative_to=report_asset.path.parent,
                label=f"{frame_id} composite",
            )
            verify_manifest_member(
                prediction_asset,
                relative_to=source_root,
                indexed=indexed_artifacts,
                label=f"{frame_id} prediction",
            )
            verify_manifest_member(
                composite_asset,
                relative_to=source_root,
                indexed=indexed_artifacts,
                label=f"{frame_id} composite",
            )

            source_rgb = load_rgb(source, label=f"{frame_id} source")
            prediction_rgb = load_rgb(prediction_asset, label=f"{frame_id} prediction")
            composite_rgb = load_rgb(composite_asset, label=f"{frame_id} composite")
            alpha = load_luma(alpha_asset, label=f"{frame_id} adaptive alpha")
            core_values = load_luma(core_asset, label=f"{frame_id} core")
            physical_values = load_luma(physical_mask, label=f"{frame_id} physical mask")
            collar_values = load_luma(collar_asset, label=f"{frame_id} collar")
            require(
                source_rgb.shape == prediction_rgb.shape == composite_rgb.shape,
                f"{frame_id} RGB dimensions differ",
            )
            shape = source_rgb.shape[:2]
            mask_shapes = {
                alpha.shape,
                core_values.shape,
                physical_values.shape,
                collar_values.shape,
                shape,
            }
            require(len(mask_shapes) == 1, f"{frame_id} mask dimensions differ")
            editable = alpha > 0
            core = core_values > 0
            require(
                bool(editable.any()) and bool(core.any()),
                f"{frame_id} masks must be non-empty",
            )
            require(not bool(np.any(core & ~editable)), f"{frame_id} core escapes adaptive alpha")
            require(bool(np.all(alpha[core] == 255)), f"{frame_id} core alpha must be opaque")
            expected_composite = (
                source_rgb.astype(np.uint16) * (255 - alpha.astype(np.uint16)[..., None])
                + prediction_rgb.astype(np.uint16) * alpha.astype(np.uint16)[..., None]
                + 127
            ) // 255
            require(
                np.array_equal(composite_rgb, expected_composite.astype(np.uint8)),
                f"{frame_id} composite differs from the declared alpha formula",
            )
            require(
                not bool(np.any(composite_rgb[~editable] != source_rgb[~editable])),
                f"{frame_id} composite changed outside editable alpha",
            )
            require(
                raw_frame.get("outside_alpha_zero_changed_values") == 0,
                f"{frame_id} report declares outside-alpha changes",
            )
            require(
                raw_frame.get("adaptive_alpha_nonzero_pixels") == int(editable.sum()),
                f"{frame_id} editable pixel count differs from report",
            )

            normalized_raw = source_rgb.copy()
            normalized_raw[editable] = prediction_rgb[editable]
            context = editable
            collar = editable & ~core
            zeros = np.zeros(shape, dtype=np.uint8)
            frame_root = staging / "frames" / f"{sequence_index:04d}"
            paths = {
                "raw_candidate_rgb": frame_root / "raw_candidate.png",
                "composite_rgb": frame_root / "composite.png",
                "ownership_mask": frame_root / "masks" / "ownership.png",
                "context_mask": frame_root / "masks" / "context.png",
                "collar_mask": frame_root / "masks" / "collar.png",
                "editable_mask": frame_root / "masks" / "editable.png",
                "protected_mask": frame_root / "masks" / "protected.png",
                "protected_collar_mask": frame_root / "masks" / "protected_collar.png",
                "blend_weight": frame_root / "masks" / "blend_weight.png",
            }
            save_rgb(paths["raw_candidate_rgb"], normalized_raw)
            save_rgb(paths["composite_rgb"], composite_rgb)
            save_mask(paths["ownership_mask"], core.astype(np.uint8) * 255)
            save_mask(paths["context_mask"], context.astype(np.uint8) * 255)
            save_mask(paths["collar_mask"], collar.astype(np.uint8) * 255)
            save_mask(paths["editable_mask"], editable.astype(np.uint8) * 255)
            save_mask(paths["protected_mask"], zeros)
            save_mask(paths["protected_collar_mask"], zeros)
            save_mask(paths["blend_weight"], alpha)
            artifacts = {key: output_asset(path, root=staging) for key, path in paths.items()}
            exactness = {
                "context_includes_ownership_core": True,
                "output_ownership_mask_pixel_exact": True,
                "inference_outside_context_rgb_exact": True,
                "inference_outside_context_changed_pixels": 0,
                "guide_outside_context_changed_from_source_pixels": 0,
                "raw_candidate_outside_context_rgb_exact": True,
                "raw_candidate_outside_context_changed_pixels": 0,
                "final_outside_editable_rgb_exact": True,
                "final_outside_editable_changed_pixels": 0,
                "final_ownership_core_equals_raw_candidate": bool(
                    np.array_equal(composite_rgb[core], normalized_raw[core])
                ),
                "protected_outer_collar_rgb_exact": True,
                "provider_inside_context_changed_from_inference_pixels": int(
                    np.count_nonzero(np.any(prediction_rgb != source_rgb, axis=2) & context)
                ),
                "provider_inside_ownership_changed_from_source_pixels": int(
                    np.count_nonzero(np.any(prediction_rgb != source_rgb, axis=2) & core)
                ),
                "provider_outside_context_changed_from_source_pixels": int(
                    np.count_nonzero(np.any(prediction_rgb != source_rgb, axis=2) & ~context)
                ),
            }
            require(
                exactness["final_ownership_core_equals_raw_candidate"] is True,
                f"{frame_id} core does not equal normalized raw prediction",
            )
            inputs = {
                "source_rgb": input_asset(source),
                "ownership_mask": input_asset(core_asset),
                "ownership_mask_source_rgb_sha256": source.sha256,
                "guide_rgb": None,
                "guide_source_rgb_sha256": None,
                "protected_mask": None,
                "protected_mask_source_rgb_sha256": None,
                "adaptive_alpha": input_asset(alpha_asset),
                "physical_sam_mask": input_asset(physical_mask),
                "uncertain_collar": input_asset(collar_asset),
                "aot_raw_prediction": input_asset(prediction_asset),
                "aot_composite": input_asset(composite_asset),
            }
            masks = {
                "dilation_metric": "upstream_adaptive_matte",
                "context_dilation_pixels": None,
                "boundary_outer_collar_pixels": None,
                "frame_pixels": int(editable.size),
                "ownership_pixels": int(core.sum()),
                "context_pixels": int(context.sum()),
                "context_additional_pixels": int((context & ~core).sum()),
                "context_excluded_protected_noncore_pixels": 0,
                "collar_pixels": int(collar.sum()),
                "protected_pixels": 0,
                "protected_collar_pixels": 0,
                "editable_pixels": int(editable.sum()),
            }
            frame_receipt = {
                "schema_version": 1,
                "kind": OUTPUT_FRAME_RECEIPT_KIND,
                "status": "generated_candidate_pending_review",
                "promotion_allowed": False,
                "sequence_index": sequence_index,
                "frame_id": frame_id,
                "seed": None,
                "seed_policy": "deterministic_checkpoint_no_sampling",
                "batch_contract": {
                    "input_manifest_sha256": input_manifest.sha256,
                    "source_aot_receipt_sha256": source_asset.sha256,
                    "source_aot_report_sha256": report_asset.sha256,
                    "normalization_producer_sha256": producer_asset["sha256"],
                    "model_family": "AOT-GAN",
                    "model_source_commit": source_commit,
                    "model_weight_sha256": model_weight.sha256,
                },
                "generation": {
                    "provider": "AOT-GAN",
                    "elapsed_seconds": raw_frame.get("duration_seconds"),
                    "inference_image": "source_rgb_with_upstream_adaptive_binary_mask",
                    "normalization_only": True,
                },
                "inputs": inputs,
                "masks": masks,
                "exactness": exactness,
                "outputs": artifacts,
                "review": {
                    "automatic_accept": False,
                    "human_review_required": True,
                    "published": False,
                },
            }
            frame_receipt_path = frame_root / "frame_receipt.json"
            atomic_write_json(frame_receipt_path, frame_receipt)
            frame_receipt_asset = output_asset(frame_receipt_path, root=staging)
            frame_receipt_hashes.append(str(frame_receipt_asset["sha256"]))
            frame_records.append(
                {
                    "sequence_index": sequence_index,
                    "frame_id": frame_id,
                    "seed": None,
                    "inputs": inputs,
                    "receipt": frame_receipt_asset,
                    "outputs": artifacts,
                    "exactness": exactness,
                    "masks": masks,
                }
            )

        receipt = {
            "schema_version": 1,
            "kind": OUTPUT_RECEIPT_KIND,
            "status": OUTPUT_STATUS,
            "created_at": timestamp.isoformat(),
            "provider": "AOT-GAN",
            "promotion_allowed": False,
            "input_manifest": output_asset(input_path, root=staging),
            "producer": producer_asset,
            "source_provenance": {
                "aot_receipt": asset_record(source_asset),
                "aot_report": asset_record(report_asset),
                "aot_artifact_manifest": asset_record(manifest_asset),
                "aot_runner": asset_record(runner_asset),
                "adaptive_receipt": asset_record(upstream_receipt),
            },
            "model": {
                "family": "AOT-GAN",
                "source_commit": source_commit,
                "weight": asset_record(model_weight),
            },
            "runtime": {
                "processing_mode": "verified_normalization_of_existing_fixed_sequence",
                "model_rerun": False,
                "quality_acceptance_performed": False,
            },
            "frames": frame_records,
            "aggregate": {
                "frame_count": len(frame_records),
                "ordered_frame_ids": list(ordered_ids),
                "ordered_seeds": [None] * len(frame_records),
                "frame_receipt_set_sha256": canonical_sha256(frame_receipt_hashes),
                "all_raw_candidates_outside_context_rgb_exact": True,
                "all_inference_inputs_outside_context_rgb_exact": True,
                "all_final_outputs_outside_editable_rgb_exact": True,
                "all_ownership_cores_equal_raw_candidates": True,
                "all_protected_outer_collars_rgb_exact": True,
                "all_source_alpha_composites_recomputed_pixel_exact": True,
            },
            "review": {
                "automatic_accept": False,
                "human_review_required": True,
                "published": False,
            },
        }
        batch_receipt_path = staging / "batch_receipt.json"
        atomic_write_json(batch_receipt_path, receipt)
        batch_receipt_asset = output_asset(batch_receipt_path, root=staging)
        selection_input = {
            "schema_version": 1,
            "kind": "video2world.clean_plate_candidate_selection_input",
            "source_batch_receipt": batch_receipt_asset,
            "ordered_frame_ids": list(ordered_ids),
            "selections": [
                {
                    "sequence_index": record["sequence_index"],
                    "frame_id": record["frame_id"],
                    "selection_note": (
                        "AOT-GAN adaptive-alpha candidate; quality review remains pending"
                    ),
                    "candidate_batch_receipt": batch_receipt_asset,
                    "composite_rgb": record["outputs"]["composite_rgb"],
                    "core_mask": record["outputs"]["ownership_mask"],
                    "editable_mask": record["outputs"]["editable_mask"],
                }
                for record in frame_records
            ],
            "contact_sheet": {
                "enabled": True,
                "thumbnail_width": 256,
                "columns": 2,
            },
        }
        atomic_write_json(staging / "selection_input.json", selection_input)
        require(
            not destination.exists(),
            f"output_dir appeared during materialization: {destination}",
        )
        os.replace(staging, destination)
        return receipt
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-source-receipt-sha256")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = materialize_aotgan_clean_plate_batch(
            args.source_receipt,
            args.output_dir,
            expected_source_receipt_sha256=args.expected_source_receipt_sha256,
        )
    except AOTBatchMaterializationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
