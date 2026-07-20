#!/usr/bin/env python3
"""Build fail-closed boundary-QA input from one audited inpaint batch receipt.

The bridge does not treat receipt booleans as sufficient evidence.  It verifies
the batch and per-frame receipt hashes, reopens every bound source/core/editable
and composite asset, and recomputes the RGB/mask exactness invariants before
writing a non-overwriting QA manifest.

For the R1 bridge the target matte is intentionally the same exact ownership
core as the removal mask.  This is an alignment self-check only; it is not
semantic residual evidence and cannot establish that the peeled object is gone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

DIFFUSION_BATCH_RECEIPT_KIND = "video2world.diffusion_clean_plate_batch_run"
DIFFUSION_FRAME_RECEIPT_KIND = "video2world.diffusion_clean_plate_frame_run"
INPAINT_BATCH_RECEIPT_KIND = "video2world.inpaint_clean_plate_batch_run"
INPAINT_FRAME_RECEIPT_KIND = "video2world.inpaint_clean_plate_frame_run"
BATCH_FRAME_RECEIPT_KINDS = {
    DIFFUSION_BATCH_RECEIPT_KIND: DIFFUSION_FRAME_RECEIPT_KIND,
    INPAINT_BATCH_RECEIPT_KIND: INPAINT_FRAME_RECEIPT_KIND,
}
BATCH_RECEIPT_KIND = DIFFUSION_BATCH_RECEIPT_KIND
FRAME_RECEIPT_KIND = DIFFUSION_FRAME_RECEIPT_KIND
OUTPUT_KIND = "video2world.layered_clean_plate_boundary_qa_manifest"
SCHEMA_VERSION = 1
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

THRESHOLDS = {
    "minimum_mask_iou": 1.0,
    "minimum_mask_precision": 1.0,
    "minimum_mask_recall": 1.0,
    "maximum_boundary_distance_p95_px": 0.0,
    "maximum_seam_color_p95_abs_rgb_delta": 24.0,
    "maximum_seam_gradient_p95_abs_rgb_delta": 32.0,
    "minimum_core_to_local_ring_laplacian_energy_ratio": 0.12,
}
TEXTURE_SAMPLING = {
    "core_erosion_px": 4,
    "local_ring_inner_px": 8,
    "local_ring_outer_px": 32,
    "minimum_region_pixels": 256,
    "winsor_quantile": 0.95,
}


class BoundaryQAManifestBuildError(ValueError):
    """Raised when a batch cannot be bridged into trustworthy QA evidence."""


@dataclass(frozen=True)
class Asset:
    path: Path
    sha256: str
    size_bytes: int


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BoundaryQAManifestBuildError(message)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BoundaryQAManifestBuildError(f"cannot read JSON {path}: {exc}") from exc
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


def resolve_path(value: Any, *, relative_to: Path, label: str) -> Path:
    require(isinstance(value, str) and bool(value.strip()), f"{label}.path is missing")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    path = path.resolve()
    require(path.is_file(), f"{label} does not exist: {path}")
    return path


def resolve_asset(value: Any, *, relative_to: Path, label: str) -> Asset:
    require(isinstance(value, dict), f"{label} must be an asset object")
    path = resolve_path(value.get("path"), relative_to=relative_to, label=label)
    expected_sha256 = required_sha256(value.get("sha256"), f"{label}.sha256")
    actual_sha256, actual_size = sha256_file(path)
    require(actual_sha256 == expected_sha256, f"{label} SHA-256 mismatch")
    declared_size = value.get("size_bytes", value.get("bytes"))
    if declared_size is not None:
        require(
            isinstance(declared_size, int)
            and not isinstance(declared_size, bool)
            and declared_size >= 0,
            f"{label}.size_bytes must be a non-negative integer",
        )
        require(actual_size == declared_size, f"{label}.size_bytes mismatch")
    return Asset(path=path, sha256=actual_sha256, size_bytes=actual_size)


def asset_record(asset: Asset) -> dict[str, Any]:
    return {
        "path": str(asset.path),
        "sha256": asset.sha256,
        "size_bytes": asset.size_bytes,
    }


def mask_asset_record(asset: Asset, *, role: str | None = None) -> dict[str, Any]:
    record = {
        **asset_record(asset),
        "channel": "luma",
        "threshold": 1,
    }
    if role is not None:
        record.update(
            {
                "evidence_role": role,
                "semantic_residual_evaluated": False,
                "semantic_residual_claim_allowed": False,
            }
        )
    return record


def load_rgb(asset: Asset, *, label: str) -> np.ndarray:
    try:
        with Image.open(asset.path) as image:
            image.load()
            require(image.mode == "RGB", f"{label} mode must be RGB, found {image.mode!r}")
            return np.array(image, dtype=np.uint8, copy=True)
    except OSError as exc:
        raise BoundaryQAManifestBuildError(f"cannot decode {label}: {asset.path}") from exc


def load_binary_mask(
    asset: Asset,
    *,
    label: str,
    allow_empty: bool = False,
) -> np.ndarray:
    try:
        with Image.open(asset.path) as image:
            image.load()
            require(image.mode in {"1", "L"}, f"{label} mode must be 1 or L")
            values = np.asarray(image)
    except OSError as exc:
        raise BoundaryQAManifestBuildError(f"cannot decode {label}: {asset.path}") from exc
    require(values.ndim == 2, f"{label} must be single-channel")
    unique_values = {int(value) for value in np.unique(values)}
    require(unique_values <= {0, 1, 255}, f"{label} must be binary 0/1 or 0/255")
    mask = values.astype(bool)
    require(allow_empty or bool(mask.any()), f"{label} must contain at least one pixel")
    return mask


def output_asset(
    frame: dict[str, Any],
    key: str,
    *,
    batch_root: Path,
    label: str,
) -> Asset:
    outputs = frame.get("outputs")
    require(isinstance(outputs, dict), f"{label}.outputs must be an object")
    return resolve_asset(
        outputs.get(key),
        relative_to=batch_root,
        label=f"{label}.outputs.{key}",
    )


def verify_frame_receipt(
    batch_frame: dict[str, Any],
    *,
    sequence_index: int,
    frame_id: str,
    batch_root: Path,
    input_manifest_sha256: str,
    expected_frame_receipt_kind: str,
) -> tuple[Asset, dict[str, Any]]:
    receipt_asset = resolve_asset(
        batch_frame.get("receipt"),
        relative_to=batch_root,
        label=f"frame {frame_id} receipt",
    )
    receipt = read_json(receipt_asset.path)
    require(receipt.get("schema_version") == SCHEMA_VERSION, f"frame {frame_id} schema invalid")
    require(
        receipt.get("kind") == expected_frame_receipt_kind,
        f"frame {frame_id} receipt kind invalid for its batch kind",
    )
    require(receipt.get("sequence_index") == sequence_index, f"frame {frame_id} index mismatch")
    require(receipt.get("frame_id") == frame_id, f"frame {frame_id} receipt id mismatch")
    require(receipt.get("promotion_allowed") is False, f"frame {frame_id} is promotable")
    review = receipt.get("review")
    require(
        isinstance(review, dict) and review.get("human_review_required") is True,
        f"frame {frame_id} must require human review",
    )
    contract = receipt.get("batch_contract")
    require(isinstance(contract, dict), f"frame {frame_id}.batch_contract missing")
    require(
        contract.get("input_manifest_sha256") == input_manifest_sha256,
        f"frame {frame_id} input-manifest lineage mismatch",
    )
    for section in ("inputs", "outputs", "masks", "exactness"):
        require(
            receipt.get(section) == batch_frame.get(section),
            f"frame {frame_id} batch/frame receipt {section} mismatch",
        )
    require(receipt.get("seed") == batch_frame.get("seed"), f"frame {frame_id} seed mismatch")
    return receipt_asset, receipt


def require_declared_exactness(frame_id: str, exactness: Any) -> dict[str, Any]:
    require(isinstance(exactness, dict), f"frame {frame_id}.exactness must be an object")
    true_keys = (
        "context_includes_ownership_core",
        "output_ownership_mask_pixel_exact",
        "inference_outside_context_rgb_exact",
        "raw_candidate_outside_context_rgb_exact",
        "final_outside_editable_rgb_exact",
        "final_ownership_core_equals_raw_candidate",
        "protected_outer_collar_rgb_exact",
    )
    for key in true_keys:
        require(exactness.get(key) is True, f"frame {frame_id}.exactness.{key} must be true")
    zero_keys = (
        "inference_outside_context_changed_pixels",
        "raw_candidate_outside_context_changed_pixels",
        "final_outside_editable_changed_pixels",
    )
    for key in zero_keys:
        require(exactness.get(key) == 0, f"frame {frame_id}.exactness.{key} must be zero")
    return exactness


def verify_frame_assets(
    frame: dict[str, Any],
    receipt: dict[str, Any],
    *,
    sequence_index: int,
    frame_id: str,
    batch_root: Path,
) -> tuple[dict[str, Any], dict[str, bool]]:
    inputs = receipt.get("inputs")
    require(isinstance(inputs, dict), f"frame {frame_id}.inputs must be an object")
    source = resolve_asset(
        inputs.get("source_rgb"),
        relative_to=batch_root,
        label=f"frame {frame_id}.inputs.source_rgb",
    )
    input_ownership = resolve_asset(
        inputs.get("ownership_mask"),
        relative_to=batch_root,
        label=f"frame {frame_id}.inputs.ownership_mask",
    )
    require(
        inputs.get("ownership_mask_source_rgb_sha256") == source.sha256,
        f"frame {frame_id} ownership mask is not bound to exact source RGB",
    )
    raw_candidate = output_asset(
        frame,
        "raw_candidate_rgb",
        batch_root=batch_root,
        label=f"frame {frame_id}",
    )
    composite = output_asset(
        frame,
        "composite_rgb",
        batch_root=batch_root,
        label=f"frame {frame_id}",
    )
    core = output_asset(
        frame,
        "ownership_mask",
        batch_root=batch_root,
        label=f"frame {frame_id}",
    )
    editable = output_asset(
        frame,
        "editable_mask",
        batch_root=batch_root,
        label=f"frame {frame_id}",
    )
    context = output_asset(
        frame,
        "context_mask",
        batch_root=batch_root,
        label=f"frame {frame_id}",
    )
    protected_collar = output_asset(
        frame,
        "protected_collar_mask",
        batch_root=batch_root,
        label=f"frame {frame_id}",
    )

    source_rgb = load_rgb(source, label=f"frame {frame_id} source RGB")
    raw_rgb = load_rgb(raw_candidate, label=f"frame {frame_id} raw candidate RGB")
    composite_rgb = load_rgb(composite, label=f"frame {frame_id} composite RGB")
    input_ownership_mask = load_binary_mask(
        input_ownership,
        label=f"frame {frame_id} input ownership mask",
    )
    core_mask = load_binary_mask(core, label=f"frame {frame_id} output ownership mask")
    editable_mask = load_binary_mask(editable, label=f"frame {frame_id} editable mask")
    context_mask = load_binary_mask(context, label=f"frame {frame_id} context mask")
    protected_collar_mask = load_binary_mask(
        protected_collar,
        label=f"frame {frame_id} protected collar mask",
        allow_empty=True,
    )
    require(
        source_rgb.shape == raw_rgb.shape == composite_rgb.shape,
        f"frame {frame_id} RGB dimensions differ",
    )
    expected_mask_shape = source_rgb.shape[:2]
    for label, mask in (
        ("input ownership", input_ownership_mask),
        ("output ownership", core_mask),
        ("editable", editable_mask),
        ("context", context_mask),
        ("protected collar", protected_collar_mask),
    ):
        require(mask.shape == expected_mask_shape, f"frame {frame_id} {label} dimensions differ")
    require(
        bool(core_mask.any()) and not bool(core_mask.all()),
        f"frame {frame_id} core must have a measurable boundary",
    )
    require(
        bool(editable_mask.any()) and not bool(editable_mask.all()),
        f"frame {frame_id} editable mask must have a measurable boundary",
    )

    exactness = require_declared_exactness(frame_id, receipt.get("exactness"))
    actual = {
        "output_ownership_matches_bound_input_pixel_exact": bool(
            np.array_equal(core_mask, input_ownership_mask)
        ),
        "context_includes_ownership_core": not bool(np.any(core_mask & ~context_mask)),
        "editable_includes_ownership_core": not bool(np.any(core_mask & ~editable_mask)),
        "raw_candidate_outside_context_rgb_exact": bool(
            np.array_equal(raw_rgb[~context_mask], source_rgb[~context_mask])
        ),
        "composite_outside_editable_rgb_exact": bool(
            np.array_equal(composite_rgb[~editable_mask], source_rgb[~editable_mask])
        ),
        "composite_core_equals_raw_candidate": bool(
            np.array_equal(composite_rgb[core_mask], raw_rgb[core_mask])
        ),
        "protected_outer_collar_rgb_exact": bool(
            np.array_equal(
                composite_rgb[protected_collar_mask],
                source_rgb[protected_collar_mask],
            )
        ),
    }
    for key, passed in actual.items():
        require(passed, f"frame {frame_id} recomputed exactness failed: {key}")

    masks = receipt.get("masks")
    require(isinstance(masks, dict), f"frame {frame_id}.masks must be an object")
    expected_counts = {
        "ownership_pixels": int(core_mask.sum()),
        "context_pixels": int(context_mask.sum()),
        "editable_pixels": int(editable_mask.sum()),
        "protected_collar_pixels": int(protected_collar_mask.sum()),
    }
    for key, actual_count in expected_counts.items():
        require(masks.get(key) == actual_count, f"frame {frame_id}.masks.{key} mismatch")
    require(
        exactness.get("output_ownership_mask_pixel_exact")
        == actual["output_ownership_matches_bound_input_pixel_exact"],
        f"frame {frame_id} ownership exactness declaration differs from pixels",
    )

    qa_record = {
        "sequence_index": sequence_index,
        "frame_id": frame_id,
        "removal_core_mask": mask_asset_record(core),
        "target_matte": mask_asset_record(core, role="alignment_self_check_only"),
        "editable_mask": mask_asset_record(editable),
        "source_rgb": asset_record(source),
        "composite_rgb": asset_record(composite),
        "evidence_contract": {
            "target_matte_role": "alignment_self_check_only",
            "target_matte_is_same_exact_asset_as_removal_core_mask": True,
            "semantic_residual_evaluated": False,
            "semantic_residual_claim_allowed": False,
        },
        "recomputed_exactness": actual,
    }
    return qa_record, actual


def atomic_write_json_non_overwriting(path: Path, value: dict[str, Any]) -> None:
    require(
        not path.exists() and not path.is_symlink(),
        f"refusing to overwrite existing output: {path}",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise BoundaryQAManifestBuildError(
                f"refusing to overwrite existing output: {path}"
            ) from exc
    finally:
        if temporary.exists():
            temporary.unlink()


def build_manifest(
    batch_receipt_path: str | Path,
    output_path: str | Path,
    *,
    expected_batch_receipt_sha256: str | None = None,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """Verify one complete inpaint batch and write its boundary-QA manifest."""

    batch_path = Path(batch_receipt_path).expanduser().resolve()
    require(batch_path.is_file(), f"batch receipt does not exist: {batch_path}")
    output = Path(output_path).expanduser().resolve()
    require(
        not output.exists() and not output.is_symlink(),
        f"refusing to overwrite existing output: {output}",
    )
    batch_sha256, batch_size = sha256_file(batch_path)
    if expected_batch_receipt_sha256 is not None:
        expected = required_sha256(
            expected_batch_receipt_sha256,
            "expected_batch_receipt_sha256",
        )
        require(batch_sha256 == expected, "batch receipt SHA-256 mismatch")
    batch_asset = Asset(batch_path, batch_sha256, batch_size)
    batch_root = batch_path.parent
    batch = read_json(batch_path)
    require(batch.get("schema_version") == SCHEMA_VERSION, "batch schema_version must be 1")
    batch_kind = batch.get("kind")
    require(batch_kind in BATCH_FRAME_RECEIPT_KINDS, "batch receipt kind is invalid")
    require(batch.get("promotion_allowed") is False, "batch receipt must remain review-only")
    review = batch.get("review")
    require(
        isinstance(review, dict) and review.get("human_review_required") is True,
        "batch receipt must require human review",
    )
    input_manifest = resolve_asset(
        batch.get("input_manifest"),
        relative_to=batch_root,
        label="batch.input_manifest",
    )
    raw_frames = batch.get("frames")
    require(isinstance(raw_frames, list) and raw_frames, "batch.frames must be non-empty")
    aggregate = batch.get("aggregate")
    require(isinstance(aggregate, dict), "batch.aggregate must be an object")
    require(aggregate.get("frame_count") == len(raw_frames), "batch frame_count mismatch")

    qa_records: list[dict[str, Any]] = []
    frame_receipt_hashes: list[str] = []
    frame_ids: list[str] = []
    seen_frame_ids: set[str] = set()
    all_recomputed: list[dict[str, bool]] = []
    for sequence_index, raw_frame in enumerate(raw_frames):
        require(isinstance(raw_frame, dict), f"batch.frames[{sequence_index}] must be an object")
        require(
            raw_frame.get("sequence_index") == sequence_index,
            "batch frame sequence indexes must be contiguous and ordered from zero",
        )
        frame_id = raw_frame.get("frame_id")
        require(isinstance(frame_id, str) and bool(frame_id), f"frame {sequence_index} id invalid")
        require(frame_id not in seen_frame_ids, f"duplicate frame_id: {frame_id}")
        seen_frame_ids.add(frame_id)
        frame_ids.append(frame_id)
        frame_receipt_asset, frame_receipt = verify_frame_receipt(
            raw_frame,
            sequence_index=sequence_index,
            frame_id=frame_id,
            batch_root=batch_root,
            input_manifest_sha256=input_manifest.sha256,
            expected_frame_receipt_kind=BATCH_FRAME_RECEIPT_KINDS[str(batch_kind)],
        )
        qa_record, recomputed = verify_frame_assets(
            raw_frame,
            frame_receipt,
            sequence_index=sequence_index,
            frame_id=frame_id,
            batch_root=batch_root,
        )
        qa_record["provenance"] = {
            "batch_receipt_sha256": batch_asset.sha256,
            "frame_receipt": asset_record(frame_receipt_asset),
            "source_input_manifest_sha256": input_manifest.sha256,
        }
        qa_records.append(qa_record)
        all_recomputed.append(recomputed)
        frame_receipt_hashes.append(frame_receipt_asset.sha256)

    require(
        aggregate.get("ordered_frame_ids") == frame_ids,
        "batch aggregate ordered_frame_ids mismatch",
    )
    require(
        aggregate.get("frame_receipt_set_sha256") == canonical_sha256(frame_receipt_hashes),
        "batch aggregate frame_receipt_set_sha256 mismatch",
    )
    aggregate_true_keys = (
        "all_raw_candidates_outside_context_rgb_exact",
        "all_inference_inputs_outside_context_rgb_exact",
        "all_final_outputs_outside_editable_rgb_exact",
        "all_ownership_cores_equal_raw_candidates",
        "all_protected_outer_collars_rgb_exact",
    )
    for key in aggregate_true_keys:
        require(aggregate.get(key) is True, f"batch.aggregate.{key} must be true")

    timestamp = created_at or datetime.now(timezone.utc)  # noqa: UP017 - remote Python 3.10.
    require(
        timestamp.tzinfo is not None and timestamp.utcoffset() is not None,
        "created_at must include timezone information",
    )
    all_actual_exact = all(all(frame.values()) for frame in all_recomputed)
    require(all_actual_exact, "not all frame exactness invariants passed")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": OUTPUT_KIND,
        "created_at": timestamp.isoformat(),
        "frame_count": len(qa_records),
        "frame_ids": frame_ids,
        "source_batch_receipt": {
            **asset_record(batch_asset),
            "hash_verification": (
                "matched_caller_supplied_sha256"
                if expected_batch_receipt_sha256 is not None
                else "recomputed_from_receipt_bytes"
            ),
        },
        "source_input_manifest": asset_record(input_manifest),
        "target_matte_contract": {
            "role": "alignment_self_check_only",
            "source": "same_exact_ownership_core_asset_as_removal_core_mask",
            "semantic_residual_evaluated": False,
            "semantic_residual_claim_allowed": False,
            "limitation": (
                "Identity alignment can validate mask wiring only; semantic residual "
                "absence requires independent segmentation or association evidence."
            ),
        },
        "thresholds": dict(THRESHOLDS),
        "texture_sampling": dict(TEXTURE_SAMPLING),
        "verification_gates": {
            "batch_receipt_sha256_recomputed": True,
            "frame_receipts_match_batch_hashes": True,
            "frame_receipt_set_sha256_matches_aggregate": True,
            "frame_order_is_complete_contiguous_and_exact": True,
            "source_core_editable_composite_hashes_verified": True,
            "pixel_exactness_recomputed_from_assets": all_actual_exact,
            "target_matte_is_alignment_self_check_only": True,
            "semantic_residual_evidence_present": False,
        },
        "frame_records": qa_records,
    }
    atomic_write_json_non_overwriting(output, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-batch-receipt-sha256")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = build_manifest(
            args.batch_receipt,
            args.output,
            expected_batch_receipt_sha256=args.expected_batch_receipt_sha256,
        )
    except BoundaryQAManifestBuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
