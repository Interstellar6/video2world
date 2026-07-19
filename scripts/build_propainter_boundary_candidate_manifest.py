#!/usr/bin/env python3
"""Bridge exact ProPainter inputs to the boundary-candidate compositor.

The legacy ProPainter manifest binds an original source RGB, its prepared
input copy, and one union ownership mask per frame. This builder verifies that
binding, attaches a separately generated candidate frame, and emits the
hash-bound input contract consumed by
``compose_clean_plate_boundary_candidate.py``.

Generation mask dilation is recorded as provenance only. The compositor core
always remains the original ProPainter union ownership mask.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

OUTPUT_KIND = "video2world.clean_plate_boundary_candidate_input"
SCHEMA_VERSION = 1
IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})


class PropainterBoundaryManifestError(RuntimeError):
    """Raised when the bridge cannot prove its input contract."""


@dataclass(frozen=True)
class VerifiedAsset:
    path: Path
    sha256: str
    bytes: int
    dimensions: tuple[int, int]
    pixels: int | None = None


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PropainterBoundaryManifestError(message)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PropainterBoundaryManifestError(f"cannot read JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: Any, *, label: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA-256 digest",
    )
    return value


def resolve_file(value: Any, *, relative_to: Path, label: str) -> Path:
    require(
        isinstance(value, str | Path) and bool(str(value).strip()),
        f"{label} must be a path string",
    )
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    path = path.resolve()
    require(path.is_file(), f"{label} does not exist: {path}")
    return path


def verify_declared_hash(path: Path, expected: Any, *, label: str) -> str:
    expected_sha256 = require_sha256(expected, label=f"{label} declared SHA-256")
    actual_sha256 = sha256_file(path)
    require(
        actual_sha256 == expected_sha256,
        f"{label} SHA-256 mismatch: {actual_sha256} != {expected_sha256}",
    )
    return actual_sha256


def rgb_dimensions(path: Path, *, label: str) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            image.load()
            require(image.mode == "RGB", f"{label} mode must be RGB, found {image.mode!r}")
            dimensions = image.size
    except OSError as exc:
        raise PropainterBoundaryManifestError(f"cannot decode {label}: {path}") from exc
    require(dimensions[0] > 0 and dimensions[1] > 0, f"{label} dimensions must be positive")
    return dimensions


def binary_mask_metadata(path: Path, *, label: str) -> tuple[tuple[int, int], int]:
    try:
        with Image.open(path) as image:
            image.load()
            require(image.mode in {"1", "L"}, f"{label} mode must be 1 or L")
            luminance = image.convert("L")
            histogram = luminance.histogram()
            dimensions = luminance.size
    except OSError as exc:
        raise PropainterBoundaryManifestError(f"cannot decode {label}: {path}") from exc
    used_values = {value for value, count in enumerate(histogram) if count}
    require(used_values <= {0, 255}, f"{label} must be binary 0/255")
    return dimensions, int(histogram[255])


def verify_rgb(
    path_value: Any,
    expected_sha256: Any,
    *,
    relative_to: Path,
    label: str,
) -> VerifiedAsset:
    path = resolve_file(path_value, relative_to=relative_to, label=label)
    digest = verify_declared_hash(path, expected_sha256, label=label)
    return VerifiedAsset(
        path=path,
        sha256=digest,
        bytes=path.stat().st_size,
        dimensions=rgb_dimensions(path, label=label),
    )


def verify_mask(
    path_value: Any,
    expected_sha256: Any | None,
    *,
    relative_to: Path,
    label: str,
    require_nonempty: bool,
) -> VerifiedAsset:
    path = resolve_file(path_value, relative_to=relative_to, label=label)
    digest = (
        verify_declared_hash(path, expected_sha256, label=label)
        if expected_sha256 is not None
        else sha256_file(path)
    )
    dimensions, pixels = binary_mask_metadata(path, label=label)
    if require_nonempty:
        require(pixels > 0, f"{label} must contain at least one foreground pixel")
    return VerifiedAsset(
        path=path,
        sha256=digest,
        bytes=path.stat().st_size,
        dimensions=dimensions,
        pixels=pixels,
    )


def compositor_asset(asset: VerifiedAsset) -> dict[str, Any]:
    value: dict[str, Any] = {
        "path": str(asset.path),
        "sha256": asset.sha256,
        "bytes": asset.bytes,
        "dimensions": list(asset.dimensions),
    }
    if asset.pixels is not None:
        value["pixels"] = asset.pixels
    return value


def validate_number(value: Any, *, label: str, integer: bool, minimum: float) -> int | float:
    valid_type = isinstance(value, int) if integer else isinstance(value, int | float)
    require(valid_type and not isinstance(value, bool), f"{label} has an invalid type")
    numeric = float(value)
    require(math.isfinite(numeric) and numeric >= minimum, f"{label} must be >= {minimum:g}")
    return int(value) if integer else numeric


def expected_frame_names(frame_count: int) -> list[str]:
    return [f"{sequence_index:04d}.png" for sequence_index in range(frame_count)]


def validate_frame_directory(path: Path, *, expected: list[str], label: str) -> Path:
    directory = path.expanduser().resolve()
    require(directory.is_dir(), f"{label} does not exist: {directory}")
    observed = sorted(
        item.name
        for item in directory.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    )
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    require(
        not missing and not extra,
        f"{label} frame set mismatch; missing={missing}, extra={extra}",
    )
    return directory


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    require(not path.exists(), f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    require(not temporary.exists(), f"temporary output already exists: {temporary}")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_manifest(
    args: argparse.Namespace,
    *,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    input_path = Path(args.propainter_manifest).expanduser().resolve()
    require(input_path.is_file(), f"ProPainter input manifest does not exist: {input_path}")
    output_path = Path(args.output).expanduser().resolve()
    require(not output_path.exists(), f"refusing to overwrite existing output: {output_path}")
    source_manifest = read_json(input_path)
    require(source_manifest.get("schema_version") == 1, "unsupported ProPainter schema_version")

    raw_records = source_manifest.get("frame_records")
    require(isinstance(raw_records, list) and raw_records, "frame_records must be non-empty")
    frame_count = source_manifest.get("frame_count")
    require(
        isinstance(frame_count, int)
        and not isinstance(frame_count, bool)
        and frame_count == len(raw_records),
        "frame_count does not match frame_records",
    )
    declared_frame_ids = source_manifest.get("frame_ids")
    require(isinstance(declared_frame_ids, list), "frame_ids must be an array")

    expected_names = expected_frame_names(frame_count)
    candidate_dir = validate_frame_directory(
        Path(args.candidate_frames_dir),
        expected=expected_names,
        label="candidate frames directory",
    )
    protected_dir = (
        validate_frame_directory(
            Path(args.protected_mask_dir),
            expected=expected_names,
            label="protected mask directory",
        )
        if args.protected_mask_dir is not None
        else None
    )
    dilation = validate_number(
        args.generation_mask_dilation_pixels,
        label="generation mask dilation pixels",
        integer=True,
        minimum=0,
    )
    collar = validate_number(
        args.outer_collar_pixels,
        label="outer collar pixels",
        integer=True,
        minimum=1,
    )
    color_limit = validate_number(
        args.maximum_boundary_color_p95_delta,
        label="maximum boundary color p95 delta",
        integer=False,
        minimum=0,
    )
    gradient_limit = validate_number(
        args.maximum_boundary_gradient_p95_delta,
        label="maximum boundary gradient p95 delta",
        integer=False,
        minimum=0,
    )

    records: list[dict[str, Any]] = []
    frame_ids: list[str] = []
    for position, raw_record in enumerate(raw_records):
        require(isinstance(raw_record, dict), f"frame_records[{position}] must be an object")
        sequence_index = raw_record.get("sequence_index")
        require(
            isinstance(sequence_index, int)
            and not isinstance(sequence_index, bool)
            and sequence_index == position,
            "frame_records must be ordered with contiguous sequence_index values starting at zero",
        )
        frame_id = raw_record.get("frame_id")
        require(
            isinstance(frame_id, str) and bool(frame_id.strip()),
            f"frame_records[{position}].frame_id must be non-empty",
        )
        require(frame_id not in frame_ids, f"duplicate frame_id: {frame_id}")
        frame_ids.append(frame_id)

        source = verify_rgb(
            raw_record.get("source_frame"),
            raw_record.get("source_frame_sha256"),
            relative_to=input_path.parent,
            label=f"frame {frame_id} source_frame",
        )
        prepared = verify_rgb(
            raw_record.get("input_frame"),
            raw_record.get("input_frame_sha256"),
            relative_to=input_path.parent,
            label=f"frame {frame_id} input_frame",
        )
        require(
            source.sha256 == prepared.sha256,
            f"frame {frame_id} input_frame is not byte-exact source RGB",
        )
        require(
            source.dimensions == prepared.dimensions,
            f"frame {frame_id} source and input RGB dimensions differ",
        )
        union_mask = verify_mask(
            raw_record.get("union_mask"),
            raw_record.get("union_mask_sha256"),
            relative_to=input_path.parent,
            label=f"frame {frame_id} union_mask",
            require_nonempty=True,
        )
        require(
            union_mask.dimensions == prepared.dimensions,
            f"frame {frame_id} union mask dimensions differ from source RGB",
        )
        width = raw_record.get("width")
        height = raw_record.get("height")
        require(
            isinstance(width, int)
            and not isinstance(width, bool)
            and isinstance(height, int)
            and not isinstance(height, bool)
            and (width, height) == prepared.dimensions,
            f"frame {frame_id} declared width/height differ from source RGB",
        )
        if "union_mask_pixels" in raw_record:
            require(
                isinstance(raw_record["union_mask_pixels"], int)
                and not isinstance(raw_record["union_mask_pixels"], bool)
                and raw_record["union_mask_pixels"] == union_mask.pixels,
                f"frame {frame_id} union_mask_pixels differs from decoded mask",
            )

        candidate_path = candidate_dir / expected_names[position]
        candidate = VerifiedAsset(
            path=candidate_path,
            sha256=sha256_file(candidate_path),
            bytes=candidate_path.stat().st_size,
            dimensions=rgb_dimensions(candidate_path, label=f"frame {frame_id} candidate_fill"),
        )
        require(
            candidate.dimensions == prepared.dimensions,
            f"frame {frame_id} candidate fill dimensions differ from source RGB",
        )
        protected = (
            verify_mask(
                protected_dir / expected_names[position],
                None,
                relative_to=protected_dir,
                label=f"frame {frame_id} protected_mask",
                require_nonempty=False,
            )
            if protected_dir is not None
            else None
        )
        if protected is not None:
            require(
                protected.dimensions == prepared.dimensions,
                f"frame {frame_id} protected mask dimensions differ from source RGB",
            )

        record = {
            "sequence_index": position,
            "frame_id": frame_id,
            "previous_composite": compositor_asset(prepared),
            "source_aligned_mask": compositor_asset(union_mask),
            "candidate_fill": compositor_asset(candidate),
            "source_binding": {
                "original_source_rgb": compositor_asset(source),
                "prepared_input_rgb": compositor_asset(prepared),
                "prepared_input_is_byte_exact_source_rgb": True,
                "source_aligned_mask_is_original_union_mask": True,
            },
        }
        if protected is not None:
            record["protected_mask"] = compositor_asset(protected)
        records.append(record)

    require(frame_ids == declared_frame_ids, "frame_ids does not match ordered frame_records")
    manifest_sha256 = sha256_file(input_path)
    created = created_at or datetime.now(timezone.utc)  # noqa: UP017
    require(created.tzinfo is not None, "created_at must be timezone-aware")
    output = {
        "schema_version": SCHEMA_VERSION,
        "kind": OUTPUT_KIND,
        "status": "ready_for_boundary_candidate_composition",
        "created_at": created.astimezone(timezone.utc).isoformat(),  # noqa: UP017
        "config": {
            "outer_collar_pixels": collar,
            "blend_curve": "smoothstep_euclidean_distance",
            "maximum_boundary_color_p95_delta": color_limit,
            "maximum_boundary_gradient_p95_delta": gradient_limit,
        },
        "provenance": {
            "propainter_input_manifest": {
                "path": str(input_path),
                "sha256": manifest_sha256,
                "bytes": input_path.stat().st_size,
            },
            "candidate_frames_directory": str(candidate_dir),
            "protected_mask_directory": str(protected_dir) if protected_dir is not None else None,
            "generation": {
                "mask_dilation_pixels": dilation,
                "role": "candidate_generation_context_only",
                "used_as_compositor_source_aligned_mask": False,
            },
        },
        "frame_count": len(records),
        "frame_ids": frame_ids,
        "frame_records": records,
        "gates": {
            "upstream_manifest_hash_recorded": True,
            "frame_order_and_exact_frame_set_verified": True,
            "source_and_prepared_rgb_hashes_verified": True,
            "prepared_rgb_is_byte_exact_source_in_every_frame": True,
            "union_masks_hash_dimensions_and_binary_values_verified": True,
            "union_masks_used_unchanged_as_source_aligned_core": True,
            "candidate_hashes_and_dimensions_verified": True,
            "protected_masks_present": protected_dir is not None,
            "protected_masks_hash_dimensions_and_binary_values_verified": True,
            "generation_mask_dilation_is_provenance_only": True,
        },
    }
    atomic_write_json(output_path, output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--propainter-manifest", type=Path, required=True)
    parser.add_argument("--candidate-frames-dir", type=Path, required=True)
    parser.add_argument("--protected-mask-dir", type=Path)
    parser.add_argument("--generation-mask-dilation-pixels", type=int, required=True)
    parser.add_argument("--outer-collar-pixels", type=int, default=6)
    parser.add_argument("--maximum-boundary-color-p95-delta", type=float, default=24.0)
    parser.add_argument("--maximum-boundary-gradient-p95-delta", type=float, default=32.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = build_manifest(args)
    except (OSError, PropainterBoundaryManifestError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output.resolve()), "status": manifest["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
