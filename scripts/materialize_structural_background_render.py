#!/usr/bin/env python3
"""Materialize limited planar RGB-D as verified structural-background renders."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    from scripts.compose_layered_clean_plate import validate_limited_texture_acceptance
except ModuleNotFoundError:  # Direct execution adds scripts/, not the repository root.
    from compose_layered_clean_plate import validate_limited_texture_acceptance


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
INDEX_KIND = "video2world.structural_background_render_index"
FRAME_RECEIPT_KIND = "video2world.structural_background_render_receipt"
BATCH_RECEIPT_KIND = "video2world.structural_background_render_batch_receipt"


@dataclass(frozen=True)
class FrameInput:
    sequence_index: int
    frame_id: str
    completed_frame: Path
    completed_frame_sha256: str
    synthetic_mask: Path
    synthetic_mask_sha256: str
    removal_mask: Path
    removal_mask_sha256: str
    geometry_depth: Path
    geometry_depth_sha256: str
    width: int
    height: int
    synthetic_pixels: int
    removal_pixels: int
    geometry_pixels: int


@dataclass(frozen=True)
class ValidatedInputs:
    accepted_report_path: Path
    accepted_report_sha256: str
    accepted_report: dict[str, Any]
    geometry_report_path: Path
    geometry_report_sha256: str
    geometry_report: dict[str, Any]
    acceptance: dict[str, Any]
    frames: tuple[FrameInput, ...]


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_asset(value: Any, *, relative_to: Path, label: str) -> Path:
    if isinstance(value, dict):
        value = value.get("path")
    require(isinstance(value, str) and value, f"{label} path is missing")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    path = path.resolve()
    require(path.is_file(), f"{label} does not exist: {path}")
    return path


def require_sha256(path: Path, expected: Any, label: str) -> str:
    require(
        isinstance(expected, str) and SHA256_PATTERN.fullmatch(expected) is not None,
        f"{label} expected sha256 is invalid",
    )
    actual = sha256_file(path)
    require(actual == expected, f"{label} sha256 mismatch: {actual} != {expected}")
    return actual


def load_rgb(path: Path, *, label: str) -> np.ndarray:
    with Image.open(path) as image:
        require(image.mode in {"RGB", "RGBA"}, f"{label} is not RGB-compatible")
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def load_mask(path: Path, *, label: str) -> np.ndarray:
    with Image.open(path) as image:
        value = np.asarray(image.convert("L"), dtype=np.uint8)
    require(bool(np.all((value == 0) | (value == 255))), f"{label} is not binary")
    return value == 255


def load_geometry_depth(path: Path, *, label: str) -> np.ndarray:
    require(path.suffix.lower() == ".npz", f"{label} must be an NPZ archive")
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == {"depth"}, f"{label} archive keys differ")
        depth = np.asarray(archive["depth"])
    require(
        depth.ndim == 2 and np.issubdtype(depth.dtype, np.number),
        f"{label} must contain a numeric HxW depth array",
    )
    return depth.astype(np.float32, copy=False)


def _gate_passed(value: Any) -> bool:
    return value is True or (isinstance(value, dict) and value.get("passed") is True)


def validate_frame(
    texture_record: dict[str, Any],
    geometry_record: dict[str, Any],
    *,
    accepted_report_dir: Path,
    geometry_report_dir: Path,
    expected_sequence_index: int,
) -> FrameInput:
    frame_id = texture_record.get("frame_id")
    require(isinstance(frame_id, str) and frame_id, "texture frame_id is missing")
    require(frame_id == geometry_record.get("frame_id"), f"frame_id mismatch: {frame_id}")
    sequence_index = texture_record.get("sequence_index")
    require(
        sequence_index == expected_sequence_index
        and geometry_record.get("sequence_index") == expected_sequence_index,
        f"non-contiguous or mismatched sequence index: {frame_id}",
    )
    require(
        texture_record.get("outside_synthetic_mask_rgb_exact") is True,
        f"outside-synthetic-mask RGB gate failed: {frame_id}",
    )
    require(
        texture_record.get("all_synthetic_pixels_assigned") is True,
        f"synthetic assignment gate failed: {frame_id}",
    )
    require(
        geometry_record.get("outside_removal_mask_rgb_exact") is True,
        f"outside-removal RGB gate failed: {frame_id}",
    )
    require(
        geometry_record.get("texture_partition_exact") is True,
        f"geometry texture partition gate failed: {frame_id}",
    )

    completed_frame = resolve_asset(
        texture_record.get("completed_frame"),
        relative_to=accepted_report_dir,
        label=f"{frame_id} completed frame",
    )
    synthetic_mask = resolve_asset(
        texture_record.get("synthetic_mask"),
        relative_to=accepted_report_dir,
        label=f"{frame_id} synthetic mask",
    )
    removal_mask = resolve_asset(
        geometry_record.get("removal_mask"),
        relative_to=geometry_report_dir,
        label=f"{frame_id} cumulative removal mask",
    )
    geometry_depth = resolve_asset(
        geometry_record.get("geometry_depth"),
        relative_to=geometry_report_dir,
        label=f"{frame_id} geometry depth",
    )
    completed_sha = require_sha256(
        completed_frame,
        texture_record.get("completed_frame_sha256"),
        f"{frame_id} completed frame",
    )
    synthetic_sha = require_sha256(
        synthetic_mask,
        texture_record.get("synthetic_mask_sha256"),
        f"{frame_id} synthetic mask",
    )
    removal_sha = require_sha256(
        removal_mask,
        geometry_record.get("removal_mask_sha256"),
        f"{frame_id} cumulative removal mask",
    )
    depth_sha = require_sha256(
        geometry_depth,
        geometry_record.get("geometry_depth_sha256"),
        f"{frame_id} geometry depth",
    )

    rgb = load_rgb(completed_frame, label=f"{frame_id} completed frame")
    synthetic = load_mask(synthetic_mask, label=f"{frame_id} synthetic mask")
    removal = load_mask(removal_mask, label=f"{frame_id} cumulative removal mask")
    depth = load_geometry_depth(geometry_depth, label=f"{frame_id} geometry depth")
    require(
        rgb.shape[:2] == synthetic.shape == removal.shape == depth.shape,
        f"asset dimensions differ: {frame_id}",
    )
    positive_finite = np.isfinite(depth) & (depth > 0)
    require(
        bool(np.all(np.isnan(depth[~positive_finite]))),
        f"geometry depth has a finite non-positive or infinite value: {frame_id}",
    )
    require(
        not bool(np.any(removal & ~positive_finite)),
        f"geometry depth does not cover cumulative removal mask: {frame_id}",
    )
    require(
        not bool(np.any(synthetic & ~removal)),
        f"synthetic mask is outside cumulative removal mask: {frame_id}",
    )
    require(
        not bool(np.any(synthetic & ~positive_finite)),
        f"geometry depth does not cover synthetic mask: {frame_id}",
    )
    synthetic_pixels = int(synthetic.sum())
    removal_pixels = int(removal.sum())
    geometry_pixels = int(positive_finite.sum())
    require(
        synthetic_pixels == int(texture_record.get("synthetic_pixels", -1)),
        f"synthetic pixel count mismatch: {frame_id}",
    )
    require(
        synthetic_pixels
        == int(texture_record.get("synthetic_pixels_assigned_from_shared_atlas", -2)),
        f"assigned synthetic pixel count mismatch: {frame_id}",
    )
    require(
        removal_pixels == int(geometry_record.get("removal_mask_pixels", -1)),
        f"cumulative removal pixel count mismatch: {frame_id}",
    )
    require(
        geometry_pixels == int(geometry_record.get("geometry_assigned_pixels", -1)),
        f"geometry-assigned pixel count mismatch: {frame_id}",
    )
    return FrameInput(
        sequence_index=expected_sequence_index,
        frame_id=frame_id,
        completed_frame=completed_frame,
        completed_frame_sha256=completed_sha,
        synthetic_mask=synthetic_mask,
        synthetic_mask_sha256=synthetic_sha,
        removal_mask=removal_mask,
        removal_mask_sha256=removal_sha,
        geometry_depth=geometry_depth,
        geometry_depth_sha256=depth_sha,
        width=int(rgb.shape[1]),
        height=int(rgb.shape[0]),
        synthetic_pixels=synthetic_pixels,
        removal_pixels=removal_pixels,
        geometry_pixels=geometry_pixels,
    )


def validate_inputs(
    *,
    accepted_report_path: Path,
    accepted_report_sha256: str,
    geometry_report_path: Path,
    geometry_report_sha256: str,
) -> ValidatedInputs:
    accepted_report_path = accepted_report_path.expanduser().resolve()
    geometry_report_path = geometry_report_path.expanduser().resolve()
    require(accepted_report_path.is_file(), f"accepted report is missing: {accepted_report_path}")
    require(geometry_report_path.is_file(), f"geometry report is missing: {geometry_report_path}")
    accepted_sha = require_sha256(
        accepted_report_path,
        accepted_report_sha256,
        "accepted planar texture report",
    )
    geometry_sha = require_sha256(
        geometry_report_path,
        geometry_report_sha256,
        "planar geometry report",
    )
    accepted_report = read_json(accepted_report_path)
    try:
        acceptance = validate_limited_texture_acceptance(accepted_report)
    except ValueError as exc:
        raise RuntimeError(f"limited planar texture acceptance is invalid: {exc}") from exc
    provenance = accepted_report.get("provenance")
    require(isinstance(provenance, dict), "accepted planar texture provenance is missing")
    require(
        provenance.get("texture_provenance") == "hybrid_atlas",
        "accepted planar texture is not a hybrid atlas",
    )
    require(
        provenance.get("claims_measured_donor") is False,
        "accepted planar texture incorrectly claims measured-donor provenance",
    )
    bound_geometry = resolve_asset(
        accepted_report.get("planar_geometry_report"),
        relative_to=accepted_report_path.parent,
        label="accepted report planar geometry binding",
    )
    require(
        bound_geometry == geometry_report_path,
        "explicit planar geometry report differs from accepted report binding",
    )
    require(
        accepted_report.get("planar_geometry_report_sha256") == geometry_sha,
        "accepted report planar geometry sha256 binding differs",
    )
    geometry_report = read_json(geometry_report_path)
    require(
        geometry_report.get("status") == "geometry_completed_texture_pending",
        "planar geometry report status is invalid",
    )
    geometry_gates = geometry_report.get("gates")
    require(isinstance(geometry_gates, dict), "planar geometry gates are missing")
    require(
        _gate_passed(geometry_gates.get("minimum_geometry_coverage")),
        "planar geometry coverage gate failed",
    )
    require(
        geometry_gates.get("outside_removal_mask_rgb_exact") is True,
        "planar geometry outside-removal RGB gate failed",
    )
    require(
        geometry_gates.get("texture_masks_partition_removal_mask") is True,
        "planar geometry texture partition gate failed",
    )
    texture_records = accepted_report.get("frame_records")
    geometry_records = geometry_report.get("frame_records")
    require(isinstance(texture_records, list) and texture_records, "texture frames are empty")
    require(isinstance(geometry_records, list) and geometry_records, "geometry frames are empty")
    require(
        len(texture_records) == len(geometry_records),
        "texture and geometry frame counts differ",
    )
    frames: list[FrameInput] = []
    seen: set[str] = set()
    for sequence_index, (texture_record, geometry_record) in enumerate(
        zip(texture_records, geometry_records, strict=True)
    ):
        require(isinstance(texture_record, dict), "texture frame record is not an object")
        require(isinstance(geometry_record, dict), "geometry frame record is not an object")
        frame = validate_frame(
            texture_record,
            geometry_record,
            accepted_report_dir=accepted_report_path.parent,
            geometry_report_dir=geometry_report_path.parent,
            expected_sequence_index=sequence_index,
        )
        require(frame.frame_id not in seen, f"duplicate frame_id: {frame.frame_id}")
        seen.add(frame.frame_id)
        frames.append(frame)
    return ValidatedInputs(
        accepted_report_path=accepted_report_path,
        accepted_report_sha256=accepted_sha,
        accepted_report=accepted_report,
        geometry_report_path=geometry_report_path,
        geometry_report_sha256=geometry_sha,
        geometry_report=geometry_report,
        acceptance=acceptance,
        frames=tuple(frames),
    )


def _check_output_target(output: Path) -> None:
    require(not output.is_symlink(), f"output must not be a symlink: {output}")
    if not output.exists():
        return
    require(output.is_dir(), f"output is not a directory: {output}")
    require(not any(output.iterdir()), f"output directory is not empty: {output}")


def materialize(
    *,
    accepted_report_path: Path,
    accepted_report_sha256: str,
    geometry_report_path: Path,
    geometry_report_sha256: str,
    output: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    output = output.expanduser().resolve()
    _check_output_target(output)
    inputs = validate_inputs(
        accepted_report_path=accepted_report_path,
        accepted_report_sha256=accepted_report_sha256,
        geometry_report_path=geometry_report_path,
        geometry_report_sha256=geometry_report_sha256,
    )
    temporary = output.with_name(f".{output.name}.materializing-{os.getpid()}")
    require(not temporary.exists(), f"temporary output already exists: {temporary}")
    try:
        temporary.mkdir(parents=True)
        frame_records: list[dict[str, Any]] = []
        output_frame_set: list[dict[str, Any]] = []
        for frame in inputs.frames:
            rgb = load_rgb(frame.completed_frame, label=f"{frame.frame_id} completed frame")
            removal = load_mask(
                frame.removal_mask,
                label=f"{frame.frame_id} cumulative removal mask",
            )
            synthetic = load_mask(
                frame.synthetic_mask,
                label=f"{frame.frame_id} synthetic mask",
            )
            source_depth = load_geometry_depth(
                frame.geometry_depth,
                label=f"{frame.frame_id} geometry depth",
            )
            alpha = np.isfinite(source_depth) & (source_depth > 0)
            require(
                not bool(np.any(removal & ~alpha)),
                f"removal coverage changed: {frame.frame_id}",
            )
            require(
                not bool(np.any(synthetic & ~alpha)),
                f"synthetic coverage changed: {frame.frame_id}",
            )
            rgba = np.empty((*rgb.shape[:2], 4), dtype=np.uint8)
            rgba[..., :3] = rgb
            rgba[..., 3] = alpha.astype(np.uint8) * 255
            depth = np.full(source_depth.shape, np.nan, dtype=np.float32)
            depth[alpha] = source_depth[alpha]

            basename = f"{frame.sequence_index:04d}"
            rgba_path = temporary / "rgba" / f"{basename}.png"
            depth_path = temporary / "depth" / f"{basename}.npy"
            receipt_path = temporary / "receipts" / f"{basename}.json"
            rgba_path.parent.mkdir(parents=True, exist_ok=True)
            depth_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgba).save(rgba_path)
            np.save(depth_path, depth, allow_pickle=False)
            rgba_sha = sha256_file(rgba_path)
            depth_sha = sha256_file(depth_path)
            acceptance = copy.deepcopy(inputs.acceptance)
            receipt = {
                "schema_version": 1,
                "kind": FRAME_RECEIPT_KIND,
                "role": "structural_background",
                "provenance_class": "structural_background",
                "claims_measured_donor": False,
                "texture_provenance": "hybrid_atlas",
                "sequence_index": frame.sequence_index,
                "frame_id": frame.frame_id,
                "rgba_sha256": rgba_sha,
                "depth_sha256": depth_sha,
                "accepted_texture_report": {
                    "path": str(inputs.accepted_report_path),
                    "sha256": inputs.accepted_report_sha256,
                },
                "accepted_texture_report_sha256": inputs.accepted_report_sha256,
                "planar_geometry_report": str(inputs.geometry_report_path),
                "planar_geometry_report_sha256": inputs.geometry_report_sha256,
                **acceptance,
            }
            write_json(receipt_path, receipt)
            receipt_sha = sha256_file(receipt_path)
            relative_rgba = rgba_path.relative_to(temporary).as_posix()
            relative_depth = depth_path.relative_to(temporary).as_posix()
            relative_receipt = receipt_path.relative_to(temporary).as_posix()
            frame_record = {
                "sequence_index": frame.sequence_index,
                "frame_id": frame.frame_id,
                "rgba": {"path": relative_rgba, "sha256": rgba_sha},
                "depth": {"path": relative_depth, "sha256": depth_sha},
                "receipt": {"path": relative_receipt, "sha256": receipt_sha},
                "source_completed_frame": {
                    "path": str(frame.completed_frame),
                    "sha256": frame.completed_frame_sha256,
                },
                "source_synthetic_mask": {
                    "path": str(frame.synthetic_mask),
                    "sha256": frame.synthetic_mask_sha256,
                    "pixels": frame.synthetic_pixels,
                },
                "source_cumulative_removal_mask": {
                    "path": str(frame.removal_mask),
                    "sha256": frame.removal_mask_sha256,
                    "pixels": frame.removal_pixels,
                },
                "source_geometry_depth": {
                    "path": str(frame.geometry_depth),
                    "sha256": frame.geometry_depth_sha256,
                    "positive_finite_pixels": frame.geometry_pixels,
                },
                "width": frame.width,
                "height": frame.height,
                "alpha_pixels": int(alpha.sum()),
                "gates": {
                    "rgba_rgb_equals_accepted_completed_frame": bool(
                        np.array_equal(rgba[..., :3], rgb)
                    ),
                    "alpha_equals_positive_finite_geometry_depth": bool(
                        np.array_equal(rgba[..., 3] == 255, alpha)
                    ),
                    "alpha_covers_cumulative_removal_mask": not bool(
                        np.any(removal & ~alpha)
                    ),
                    "alpha_covers_synthetic_mask": not bool(np.any(synthetic & ~alpha)),
                    "depth_nan_outside_alpha": bool(np.all(np.isnan(depth[~alpha]))),
                },
            }
            frame_records.append(frame_record)
            output_frame_set.append(
                {
                    "sequence_index": frame.sequence_index,
                    "frame_id": frame.frame_id,
                    "rgba_sha256": rgba_sha,
                    "depth_sha256": depth_sha,
                    "receipt_sha256": receipt_sha,
                }
            )

        acceptance = copy.deepcopy(inputs.acceptance)
        index = {
            "schema_version": 1,
            "kind": INDEX_KIND,
            "status": "materialized_for_current_demo_with_limitations",
            "created_at": datetime.now(UTC).isoformat(),
            "role": "structural_background",
            "provenance_class": "structural_background",
            "claims_measured_donor": False,
            "texture_provenance": "hybrid_atlas",
            "accepted_texture_report": {
                "path": str(inputs.accepted_report_path),
                "sha256": inputs.accepted_report_sha256,
            },
            "planar_geometry_report": {
                "path": str(inputs.geometry_report_path),
                "sha256": inputs.geometry_report_sha256,
            },
            **acceptance,
            "frame_count": len(frame_records),
            "frame_records": frame_records,
            "output_frame_set_sha256": digest_json(output_frame_set),
            "gates": {
                "accepted_texture_report_hash_bound": True,
                "planar_geometry_report_hash_bound": True,
                "texture_geometry_frame_order_exact": True,
                "rgba_rgb_equals_accepted_completed_frame": all(
                    record["gates"]["rgba_rgb_equals_accepted_completed_frame"]
                    for record in frame_records
                ),
                "alpha_equals_positive_finite_geometry_depth": all(
                    record["gates"]["alpha_equals_positive_finite_geometry_depth"]
                    for record in frame_records
                ),
                "alpha_covers_cumulative_removal_mask": all(
                    record["gates"]["alpha_covers_cumulative_removal_mask"]
                    for record in frame_records
                ),
                "alpha_covers_synthetic_mask": all(
                    record["gates"]["alpha_covers_synthetic_mask"]
                    for record in frame_records
                ),
                "depth_nan_outside_alpha": all(
                    record["gates"]["depth_nan_outside_alpha"]
                    for record in frame_records
                ),
            },
            "promotion_approved": False,
            "promotion_blocker": (
                "current-demo-only limited acceptance; overridden visual and boundary "
                "quality gates remain failed"
            ),
            "batch_receipt": "structural_background_render_receipt.json",
        }
        index_path = temporary / "structural_background_render_index.json"
        write_json(index_path, index)
        batch_receipt = {
            "schema_version": 1,
            "kind": BATCH_RECEIPT_KIND,
            "status": "materialized_for_current_demo_with_limitations",
            "index": index_path.name,
            "index_sha256": sha256_file(index_path),
            "output_frame_set_sha256": index["output_frame_set_sha256"],
            "frame_count": len(frame_records),
            "accepted_texture_report": {
                "path": str(inputs.accepted_report_path),
                "sha256": inputs.accepted_report_sha256,
            },
            "accepted_texture_report_sha256": inputs.accepted_report_sha256,
            "planar_geometry_report": str(inputs.geometry_report_path),
            "planar_geometry_report_sha256": inputs.geometry_report_sha256,
            "claims_measured_donor": False,
            "texture_provenance": "hybrid_atlas",
            **copy.deepcopy(inputs.acceptance),
        }
        write_json(temporary / "structural_background_render_receipt.json", batch_receipt)
        if output.exists():
            output.rmdir()
        shutil.move(str(temporary), output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return index, batch_receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-planar-texture-report", type=Path, required=True)
    parser.add_argument("--accepted-planar-texture-report-sha256", required=True)
    parser.add_argument("--planar-geometry-report", type=Path, required=True)
    parser.add_argument("--planar-geometry-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    index, receipt = materialize(
        accepted_report_path=args.accepted_planar_texture_report,
        accepted_report_sha256=args.accepted_planar_texture_report_sha256,
        geometry_report_path=args.planar_geometry_report,
        geometry_report_sha256=args.planar_geometry_report_sha256,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "status": index["status"],
                "frame_count": index["frame_count"],
                "output_frame_set_sha256": index["output_frame_set_sha256"],
                "index_sha256": receipt["index_sha256"],
                "promotion_approved": index["promotion_approved"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
