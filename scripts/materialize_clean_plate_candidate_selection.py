#!/usr/bin/env python3
"""Materialize one review-only clean-plate candidate per ordered source frame.

The selection manifest names one complete source batch and may select frames
from any number of audited diffusion or generic inpaint batch receipts.  This
tool revalidates all hashes, source-RGB lineage, frame order, dimensions,
receipt exactness, and the outside-editable RGB invariant before copying a new
immutable candidate sequence.  Materialization is deliberately not a quality
or promotion gate.
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
from PIL import Image, ImageDraw

INPUT_KIND = "video2world.clean_plate_candidate_selection_input"
RECEIPT_KIND = "video2world.clean_plate_candidate_selection_receipt"
DIFFUSION_BATCH_RECEIPT_KIND = "video2world.diffusion_clean_plate_batch_run"
DIFFUSION_FRAME_RECEIPT_KIND = "video2world.diffusion_clean_plate_frame_run"
INPAINT_BATCH_RECEIPT_KIND = "video2world.inpaint_clean_plate_batch_run"
INPAINT_FRAME_RECEIPT_KIND = "video2world.inpaint_clean_plate_frame_run"
BATCH_FRAME_RECEIPT_KINDS = {
    DIFFUSION_BATCH_RECEIPT_KIND: DIFFUSION_FRAME_RECEIPT_KIND,
    INPAINT_BATCH_RECEIPT_KIND: INPAINT_FRAME_RECEIPT_KIND,
}
# Backward-compatible names used by existing callers and fixtures.
BATCH_RECEIPT_KIND = DIFFUSION_BATCH_RECEIPT_KIND
FRAME_RECEIPT_KIND = DIFFUSION_FRAME_RECEIPT_KIND
SCHEMA_VERSION = 1
STATUS = "materialized_candidate_selection_pending_human_review"
FRAME_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class CleanPlateCandidateSelectionError(ValueError):
    """Raised when selected candidates cannot be proven safe to materialize."""


@dataclass(frozen=True)
class Asset:
    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class BatchReceipt:
    asset: Asset
    value: dict[str, Any]
    root: Path
    frames_by_id: dict[str, dict[str, Any]]
    ordered_frame_ids: tuple[str, ...]
    frame_receipt_kind: str


@dataclass(frozen=True)
class SelectedFrame:
    sequence_index: int
    frame_id: str
    selection_note: str | None
    source_asset: Asset
    candidate_batch: BatchReceipt
    candidate_frame_receipt: Asset
    composite_asset: Asset
    core_asset: Asset
    editable_asset: Asset
    source_rgb: np.ndarray
    composite_rgb: np.ndarray
    core_mask: np.ndarray
    editable_mask: np.ndarray


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CleanPlateCandidateSelectionError(message)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CleanPlateCandidateSelectionError(f"cannot read JSON {path}: {exc}") from exc
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
    expected_sha = required_sha256(value.get("sha256"), f"{label}.sha256")
    actual_sha, actual_size = sha256_file(path)
    require(actual_sha == expected_sha, f"{label} SHA-256 mismatch")
    expected_size = value.get("size_bytes")
    if expected_size is not None:
        require(
            isinstance(expected_size, int)
            and not isinstance(expected_size, bool)
            and expected_size >= 0,
            f"{label}.size_bytes must be a non-negative integer",
        )
        require(actual_size == expected_size, f"{label}.size_bytes mismatch")
    return Asset(path=path, sha256=actual_sha, size_bytes=actual_size)


def asset_record(asset: Asset, *, path: str | None = None) -> dict[str, Any]:
    return {
        "path": path if path is not None else str(asset.path),
        "sha256": asset.sha256,
        "size_bytes": asset.size_bytes,
    }


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_rgb(path: Path, *, label: str) -> np.ndarray:
    try:
        with Image.open(path) as image:
            image.load()
            require(image.mode == "RGB", f"{label} mode must be RGB, found {image.mode!r}")
            return np.array(image, dtype=np.uint8, copy=True)
    except OSError as exc:
        raise CleanPlateCandidateSelectionError(f"cannot decode {label}: {path}") from exc


def load_binary_mask(path: Path, *, label: str, allow_empty: bool = False) -> np.ndarray:
    try:
        with Image.open(path) as image:
            image.load()
            require(image.mode in {"1", "L"}, f"{label} mode must be 1 or L")
            values = np.asarray(image)
    except OSError as exc:
        raise CleanPlateCandidateSelectionError(f"cannot decode {label}: {path}") from exc
    require(values.ndim == 2, f"{label} must be single-channel")
    unique_values = {int(value) for value in np.unique(values)}
    require(unique_values <= {0, 1, 255}, f"{label} must be binary 0/1 or 0/255")
    mask = values.astype(bool)
    require(allow_empty or bool(mask.any()), f"{label} must contain at least one pixel")
    return mask


def validate_frame_list(receipt: dict[str, Any], *, label: str) -> tuple[dict[str, Any], ...]:
    raw_frames = receipt.get("frames")
    require(isinstance(raw_frames, list) and raw_frames, f"{label}.frames must be non-empty")
    frames: list[dict[str, Any]] = []
    seen: set[str] = set()
    for sequence_index, raw_frame in enumerate(raw_frames):
        require(isinstance(raw_frame, dict), f"{label}.frames[{sequence_index}] must be an object")
        require(
            raw_frame.get("sequence_index") == sequence_index,
            f"{label} sequence indexes must be contiguous from zero",
        )
        frame_id = raw_frame.get("frame_id")
        require(
            isinstance(frame_id, str) and FRAME_ID_PATTERN.fullmatch(frame_id) is not None,
            f"{label}.frames[{sequence_index}].frame_id is unsafe",
        )
        require(frame_id not in seen, f"{label} contains duplicate frame_id {frame_id}")
        seen.add(frame_id)
        frames.append(raw_frame)
    return tuple(frames)


def load_batch_receipt(value: Any, *, manifest_path: Path, label: str) -> BatchReceipt:
    asset = resolve_asset(value, relative_to=manifest_path.parent, label=label)
    receipt = read_json(asset.path)
    require(receipt.get("schema_version") == 1, f"{label} schema_version must be 1")
    batch_kind = receipt.get("kind")
    require(batch_kind in BATCH_FRAME_RECEIPT_KINDS, f"{label} kind is invalid")
    require(receipt.get("promotion_allowed") is False, f"{label} must remain review-only")
    review = receipt.get("review")
    require(isinstance(review, dict), f"{label}.review must be an object")
    require(review.get("human_review_required") is True, f"{label} requires human review")
    frames = validate_frame_list(receipt, label=label)
    ordered = tuple(str(frame["frame_id"]) for frame in frames)
    aggregate = receipt.get("aggregate")
    require(isinstance(aggregate, dict), f"{label}.aggregate must be an object")
    require(aggregate.get("frame_count") == len(frames), f"{label} frame_count mismatch")
    require(aggregate.get("ordered_frame_ids") == list(ordered), f"{label} frame order mismatch")
    require(
        aggregate.get("all_final_outputs_outside_editable_rgb_exact") is True,
        f"{label} does not attest outside-editable exactness",
    )
    return BatchReceipt(
        asset=asset,
        value=receipt,
        root=asset.path.parent,
        frames_by_id={str(frame["frame_id"]): frame for frame in frames},
        ordered_frame_ids=ordered,
        frame_receipt_kind=BATCH_FRAME_RECEIPT_KINDS[str(batch_kind)],
    )


def receipt_output_asset(
    frame: dict[str, Any],
    key: str,
    *,
    batch: BatchReceipt,
    label: str,
) -> Asset:
    outputs = frame.get("outputs")
    require(isinstance(outputs, dict), f"{label}.outputs must be an object")
    return resolve_asset(outputs.get(key), relative_to=batch.root, label=f"{label}.outputs.{key}")


def validate_frame_receipt(
    frame: dict[str, Any],
    *,
    batch: BatchReceipt,
    frame_id: str,
) -> Asset:
    receipt_asset = resolve_asset(
        frame.get("receipt"),
        relative_to=batch.root,
        label=f"candidate frame {frame_id} receipt",
    )
    receipt = read_json(receipt_asset.path)
    require(
        receipt.get("kind") == batch.frame_receipt_kind,
        f"frame {frame_id} receipt kind is invalid for its batch kind",
    )
    require(receipt.get("frame_id") == frame_id, f"frame {frame_id} receipt frame_id mismatch")
    require(receipt.get("promotion_allowed") is False, f"frame {frame_id} receipt is promotable")
    review = receipt.get("review")
    require(
        isinstance(review, dict) and review.get("human_review_required") is True,
        f"frame {frame_id} receipt must require human review",
    )
    for section in ("inputs", "outputs", "exactness", "masks"):
        require(
            receipt.get(section) == frame.get(section),
            f"frame {frame_id} batch/frame receipt {section} mismatch",
        )
    return receipt_asset


def matching_selected_asset(
    selected_value: Any,
    receipt_asset: Asset,
    *,
    manifest_path: Path,
    label: str,
) -> Asset:
    selected = resolve_asset(selected_value, relative_to=manifest_path.parent, label=label)
    require(selected.sha256 == receipt_asset.sha256, f"{label} is not hash-bound by batch receipt")
    require(
        selected.size_bytes == receipt_asset.size_bytes,
        f"{label} size differs from batch receipt",
    )
    return selected


def source_asset_for_frame(batch: BatchReceipt, frame_id: str) -> Asset:
    frame = batch.frames_by_id[frame_id]
    inputs = frame.get("inputs")
    require(isinstance(inputs, dict), f"source frame {frame_id}.inputs must be an object")
    source = resolve_asset(
        inputs.get("source_rgb"),
        relative_to=batch.root,
        label=f"source frame {frame_id}.inputs.source_rgb",
    )
    exactness = frame.get("exactness")
    require(isinstance(exactness, dict), f"source frame {frame_id}.exactness must be an object")
    require(
        exactness.get("final_outside_editable_rgb_exact") is True
        and exactness.get("final_outside_editable_changed_pixels") == 0,
        f"source frame {frame_id} receipt lacks exact outside-editable evidence",
    )
    return source


def parse_ordered_frame_ids(value: Any) -> tuple[str, ...]:
    require(isinstance(value, list) and value, "ordered_frame_ids must be a non-empty array")
    ordered: list[str] = []
    seen: set[str] = set()
    for index, frame_id in enumerate(value):
        require(
            isinstance(frame_id, str) and FRAME_ID_PATTERN.fullmatch(frame_id) is not None,
            f"ordered_frame_ids[{index}] is unsafe",
        )
        require(frame_id not in seen, f"ordered_frame_ids contains duplicate {frame_id}")
        seen.add(frame_id)
        ordered.append(frame_id)
    return tuple(ordered)


def parse_contact_sheet(value: Any) -> dict[str, Any]:
    if value is None:
        return {"enabled": False, "thumbnail_width": 160, "columns": 4}
    require(isinstance(value, dict), "contact_sheet must be an object")
    enabled = value.get("enabled", False)
    width = value.get("thumbnail_width", 160)
    columns = value.get("columns", 4)
    require(isinstance(enabled, bool), "contact_sheet.enabled must be boolean")
    require(
        isinstance(width, int) and not isinstance(width, bool) and 64 <= width <= 512,
        "contact_sheet.thumbnail_width must be an integer inside [64, 512]",
    )
    require(
        isinstance(columns, int) and not isinstance(columns, bool) and 1 <= columns <= 8,
        "contact_sheet.columns must be an integer inside [1, 8]",
    )
    return {"enabled": enabled, "thumbnail_width": width, "columns": columns}


def validate_selected_frame(
    raw_selection: Any,
    *,
    sequence_index: int,
    expected_frame_id: str,
    source_batch: BatchReceipt,
    manifest_path: Path,
    batch_cache: dict[Path, BatchReceipt],
) -> SelectedFrame:
    label = f"selections[{sequence_index}]"
    require(isinstance(raw_selection, dict), f"{label} must be an object")
    require(raw_selection.get("sequence_index") == sequence_index, f"{label} index mismatch")
    require(raw_selection.get("frame_id") == expected_frame_id, f"{label} frame_id mismatch")
    note = raw_selection.get("selection_note")
    require(
        note is None or (isinstance(note, str) and bool(note.strip())),
        f"{label}.selection_note invalid",
    )

    candidate_receipt_value = raw_selection.get("candidate_batch_receipt")
    require(isinstance(candidate_receipt_value, dict), f"{label}.candidate_batch_receipt missing")
    candidate_receipt_path = resolve_path(
        candidate_receipt_value.get("path"),
        relative_to=manifest_path.parent,
        label=f"{label}.candidate_batch_receipt",
    )
    candidate_batch = batch_cache.get(candidate_receipt_path)
    if candidate_batch is None:
        candidate_batch = load_batch_receipt(
            candidate_receipt_value,
            manifest_path=manifest_path,
            label=f"{label}.candidate_batch_receipt",
        )
        batch_cache[candidate_receipt_path] = candidate_batch
    else:
        declared_sha = required_sha256(
            candidate_receipt_value.get("sha256"),
            f"{label}.candidate_batch_receipt.sha256",
        )
        require(
            declared_sha == candidate_batch.asset.sha256,
            f"{label}.candidate_batch_receipt conflicts with earlier declaration",
        )

    require(
        expected_frame_id in candidate_batch.frames_by_id,
        f"candidate batch does not contain frame {expected_frame_id}",
    )
    candidate_frame = candidate_batch.frames_by_id[expected_frame_id]
    frame_receipt_asset = validate_frame_receipt(
        candidate_frame,
        batch=candidate_batch,
        frame_id=expected_frame_id,
    )
    exactness = candidate_frame.get("exactness")
    require(isinstance(exactness, dict), f"candidate frame {expected_frame_id}.exactness missing")
    require(
        exactness.get("final_outside_editable_rgb_exact") is True
        and exactness.get("final_outside_editable_changed_pixels") == 0,
        f"candidate frame {expected_frame_id} lacks exact outside-editable receipt",
    )
    require(
        exactness.get("output_ownership_mask_pixel_exact") is True,
        f"candidate frame {expected_frame_id} ownership output is not exact",
    )

    source_asset = source_asset_for_frame(source_batch, expected_frame_id)
    candidate_inputs = candidate_frame.get("inputs")
    require(
        isinstance(candidate_inputs, dict),
        f"candidate frame {expected_frame_id}.inputs missing",
    )
    candidate_source = resolve_asset(
        candidate_inputs.get("source_rgb"),
        relative_to=candidate_batch.root,
        label=f"candidate frame {expected_frame_id}.inputs.source_rgb",
    )
    require(
        candidate_source.sha256 == source_asset.sha256,
        f"candidate frame {expected_frame_id} source lineage differs from source batch",
    )
    require(
        candidate_inputs.get("ownership_mask_source_rgb_sha256") == source_asset.sha256,
        f"candidate frame {expected_frame_id} ownership mask is bound to another source RGB",
    )

    receipt_composite = receipt_output_asset(
        candidate_frame,
        "composite_rgb",
        batch=candidate_batch,
        label=f"candidate frame {expected_frame_id}",
    )
    receipt_core = receipt_output_asset(
        candidate_frame,
        "ownership_mask",
        batch=candidate_batch,
        label=f"candidate frame {expected_frame_id}",
    )
    receipt_editable = receipt_output_asset(
        candidate_frame,
        "editable_mask",
        batch=candidate_batch,
        label=f"candidate frame {expected_frame_id}",
    )
    composite = matching_selected_asset(
        raw_selection.get("composite_rgb"),
        receipt_composite,
        manifest_path=manifest_path,
        label=f"{label}.composite_rgb",
    )
    core = matching_selected_asset(
        raw_selection.get("core_mask"),
        receipt_core,
        manifest_path=manifest_path,
        label=f"{label}.core_mask",
    )
    editable = matching_selected_asset(
        raw_selection.get("editable_mask"),
        receipt_editable,
        manifest_path=manifest_path,
        label=f"{label}.editable_mask",
    )

    source_rgb = load_rgb(source_asset.path, label=f"source frame {expected_frame_id}")
    composite_rgb = load_rgb(composite.path, label=f"candidate frame {expected_frame_id}")
    core_mask = load_binary_mask(core.path, label=f"frame {expected_frame_id} core mask")
    editable_mask = load_binary_mask(
        editable.path,
        label=f"frame {expected_frame_id} editable mask",
    )
    require(
        composite_rgb.shape == source_rgb.shape,
        f"frame {expected_frame_id} RGB dimensions differ",
    )
    require(
        core_mask.shape == source_rgb.shape[:2],
        f"frame {expected_frame_id} core dimensions differ",
    )
    require(
        editable_mask.shape == source_rgb.shape[:2],
        f"frame {expected_frame_id} editable dimensions differ",
    )
    require(
        not bool(np.any(core_mask & ~editable_mask)),
        f"frame {expected_frame_id} core mask escapes editable mask",
    )
    outside_changed = np.any(composite_rgb != source_rgb, axis=2) & ~editable_mask
    require(
        not bool(outside_changed.any()),
        f"frame {expected_frame_id} composite changes RGB outside editable mask",
    )
    masks = candidate_frame.get("masks")
    require(isinstance(masks, dict), f"candidate frame {expected_frame_id}.masks missing")
    require(
        masks.get("ownership_pixels") == int(core_mask.sum()),
        f"frame {expected_frame_id} core pixel count differs from receipt",
    )
    require(
        masks.get("editable_pixels") == int(editable_mask.sum()),
        f"frame {expected_frame_id} editable pixel count differs from receipt",
    )
    return SelectedFrame(
        sequence_index=sequence_index,
        frame_id=expected_frame_id,
        selection_note=note,
        source_asset=source_asset,
        candidate_batch=candidate_batch,
        candidate_frame_receipt=frame_receipt_asset,
        composite_asset=composite,
        core_asset=core,
        editable_asset=editable,
        source_rgb=source_rgb,
        composite_rgb=composite_rgb,
        core_mask=core_mask,
        editable_mask=editable_mask,
    )


def save_contact_sheet(
    path: Path,
    frames: tuple[SelectedFrame, ...],
    *,
    thumbnail_width: int,
    columns: int,
) -> None:
    header_height = 18
    first_height, first_width = frames[0].source_rgb.shape[:2]
    thumbnail_height = max(1, round(first_height * thumbnail_width / first_width))
    tile_width = thumbnail_width * 2
    tile_height = thumbnail_height + header_height
    rows = (len(frames) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for frame in frames:
        row, column = divmod(frame.sequence_index, columns)
        x = column * tile_width
        y = row * tile_height
        source = Image.fromarray(frame.source_rgb).resize(
            (thumbnail_width, thumbnail_height),
            Image.Resampling.LANCZOS,
        )
        selected_rgb = frame.composite_rgb.astype(np.float32).copy()
        selected_rgb[frame.editable_mask] = (
            selected_rgb[frame.editable_mask] * 0.72 + np.array([255, 190, 0]) * 0.28
        )
        selected_rgb[frame.core_mask] = (
            selected_rgb[frame.core_mask] * 0.65 + np.array([20, 220, 80]) * 0.35
        )
        selected = Image.fromarray(np.clip(selected_rgb, 0, 255).astype(np.uint8)).resize(
            (thumbnail_width, thumbnail_height),
            Image.Resampling.LANCZOS,
        )
        sheet.paste(source, (x, y + header_height))
        sheet.paste(selected, (x + thumbnail_width, y + header_height))
        draw.text((x + 3, y + 3), f"{frame.frame_id} | source / selected+mask", fill="white")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, format="PNG", optimize=True)


def output_asset(path: Path, *, root: Path) -> dict[str, Any]:
    digest, size = sha256_file(path)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": digest,
        "size_bytes": size,
    }


def copy_asset(source: Asset, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source.path, destination)
    digest, size = sha256_file(destination)
    require(
        digest == source.sha256 and size == source.size_bytes,
        f"copy verification failed: {source.path}",
    )


def prepare_destination(output_dir: str | Path) -> Path:
    candidate = Path(output_dir).expanduser()
    require(not candidate.is_symlink(), f"output_dir cannot be a symlink: {candidate}")
    destination = candidate.resolve()
    require(not destination.exists(), f"refusing to overwrite output_dir: {destination}")
    return destination


def materialize_clean_plate_candidate_selection(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """Validate and atomically materialize one selected candidate per source frame."""

    manifest_file = Path(manifest_path).expanduser().resolve()
    require(manifest_file.is_file(), f"manifest does not exist: {manifest_file}")
    manifest_sha, manifest_size = sha256_file(manifest_file)
    manifest = read_json(manifest_file)
    require(manifest.get("schema_version") == SCHEMA_VERSION, "manifest schema_version must be 1")
    require(manifest.get("kind") == INPUT_KIND, "manifest kind is invalid")
    ordered_frame_ids = parse_ordered_frame_ids(manifest.get("ordered_frame_ids"))
    contact_sheet = parse_contact_sheet(manifest.get("contact_sheet"))
    source_batch = load_batch_receipt(
        manifest.get("source_batch_receipt"),
        manifest_path=manifest_file,
        label="source_batch_receipt",
    )
    require(
        source_batch.ordered_frame_ids == ordered_frame_ids,
        "ordered_frame_ids must exactly match the complete source batch frame set and order",
    )
    raw_selections = manifest.get("selections")
    require(isinstance(raw_selections, list), "selections must be an array")
    require(
        len(raw_selections) == len(ordered_frame_ids),
        "selections must cover every source frame",
    )
    batch_cache = {source_batch.asset.path: source_batch}
    selected_frames = tuple(
        validate_selected_frame(
            raw_selection,
            sequence_index=sequence_index,
            expected_frame_id=frame_id,
            source_batch=source_batch,
            manifest_path=manifest_file,
            batch_cache=batch_cache,
        )
        for sequence_index, (frame_id, raw_selection) in enumerate(
            zip(ordered_frame_ids, raw_selections, strict=True)
        )
    )
    dimensions = {
        (int(frame.source_rgb.shape[1]), int(frame.source_rgb.shape[0]))
        for frame in selected_frames
    }
    require(len(dimensions) == 1, "all selected source frames must have identical dimensions")

    timestamp = created_at or datetime.now(timezone.utc)  # noqa: UP017 - remote Python 3.10.
    require(
        timestamp.tzinfo is not None and timestamp.utcoffset() is not None,
        "created_at must include timezone information",
    )
    destination = prepare_destination(output_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        frame_records: list[dict[str, Any]] = []
        for frame in selected_frames:
            output_name = f"{frame.sequence_index:04d}.png"
            output_rgb = staging / "frames" / output_name
            output_core = staging / "masks" / "core" / output_name
            output_editable = staging / "masks" / "editable" / output_name
            copy_asset(frame.composite_asset, output_rgb)
            copy_asset(frame.core_asset, output_core)
            copy_asset(frame.editable_asset, output_editable)
            frame_records.append(
                {
                    "sequence_index": frame.sequence_index,
                    "frame_id": frame.frame_id,
                    "selection_note": frame.selection_note,
                    "source_rgb": asset_record(frame.source_asset),
                    "candidate_batch_receipt": asset_record(frame.candidate_batch.asset),
                    "candidate_frame_receipt": asset_record(frame.candidate_frame_receipt),
                    "selected_inputs": {
                        "composite_rgb": asset_record(frame.composite_asset),
                        "core_mask": asset_record(frame.core_asset),
                        "editable_mask": asset_record(frame.editable_asset),
                    },
                    "outputs": {
                        "composite_rgb": output_asset(output_rgb, root=staging),
                        "core_mask": output_asset(output_core, root=staging),
                        "editable_mask": output_asset(output_editable, root=staging),
                    },
                    "dimensions": {
                        "width": int(frame.source_rgb.shape[1]),
                        "height": int(frame.source_rgb.shape[0]),
                    },
                    "mask_pixels": {
                        "core": int(frame.core_mask.sum()),
                        "editable": int(frame.editable_mask.sum()),
                    },
                    "exactness": {
                        "passed": True,
                        "candidate_source_rgb_matches_source_batch": True,
                        "selected_artifacts_match_candidate_batch_receipt": True,
                        "core_is_subset_of_editable": True,
                        "outside_editable_rgb_exact_recomputed": True,
                        "outside_editable_changed_pixels_recomputed": 0,
                    },
                }
            )

        contact_sheet_artifact = None
        if contact_sheet["enabled"]:
            contact_sheet_path = staging / "contact_sheet.png"
            save_contact_sheet(
                contact_sheet_path,
                selected_frames,
                thumbnail_width=contact_sheet["thumbnail_width"],
                columns=contact_sheet["columns"],
            )
            contact_sheet_artifact = output_asset(contact_sheet_path, root=staging)

        unique_candidate_batches = sorted(
            {frame.candidate_batch.asset for frame in selected_frames},
            key=lambda item: (str(item.path), item.sha256),
        )
        frame_set = [
            {
                "sequence_index": record["sequence_index"],
                "frame_id": record["frame_id"],
                "source_rgb_sha256": record["source_rgb"]["sha256"],
                "composite_rgb_sha256": record["outputs"]["composite_rgb"]["sha256"],
                "core_mask_sha256": record["outputs"]["core_mask"]["sha256"],
                "editable_mask_sha256": record["outputs"]["editable_mask"]["sha256"],
                "candidate_batch_receipt_sha256": record["candidate_batch_receipt"]["sha256"],
            }
            for record in frame_records
        ]
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "kind": RECEIPT_KIND,
            "status": STATUS,
            "created_at": timestamp.isoformat(),
            "promotion_allowed": False,
            "human_review_required": True,
            "input_manifest": {
                "path": str(manifest_file),
                "sha256": manifest_sha,
                "size_bytes": manifest_size,
            },
            "source_batch_receipt": asset_record(source_batch.asset),
            "candidate_batch_receipts": [asset_record(asset) for asset in unique_candidate_batches],
            "ordered_frame_ids": list(ordered_frame_ids),
            "frames": frame_records,
            "contact_sheet": contact_sheet_artifact,
            "aggregate": {
                "frame_count": len(frame_records),
                "frame_set_sha256": canonical_sha256(frame_set),
                "all_source_rgb_lineages_match": True,
                "all_frame_sets_and_order_validated": True,
                "all_dimensions_match": True,
                "all_input_and_output_hashes_verified": True,
                "all_core_masks_are_subsets_of_editable_masks": True,
                "all_outputs_outside_editable_rgb_exact_recomputed": True,
            },
            "review": {
                "automatic_accept": False,
                "human_review_required": True,
                "quality_acceptance_performed": False,
                "published": False,
            },
            "limitations": [
                (
                    "Candidate selection and materialization do not establish "
                    "semantic removal quality."
                ),
                "Candidate selection and materialization do not establish temporal consistency.",
                (
                    "A separate human or VLM review and downstream QA report are "
                    "required before promotion."
                ),
            ],
        }
        atomic_write_json(staging / "selection_receipt.json", receipt)
        require(not destination.exists(), f"output_dir appeared during run: {destination}")
        os.replace(staging, destination)
        return receipt
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = materialize_clean_plate_candidate_selection(args.manifest, args.output_dir)
    except CleanPlateCandidateSelectionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
