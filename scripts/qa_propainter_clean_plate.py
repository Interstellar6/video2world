#!/usr/bin/env python3
"""Validate ProPainter clean-plate frames and create a visual comparison sheet."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


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
        )
    return result


def masked_stats(values: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    selected = values[mask]
    if selected.size == 0:
        return {"pixel_count": 0, "mean": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "pixel_count": int(mask.sum()),
        "mean": float(selected.mean()),
        "p95": float(np.quantile(selected, 0.95)),
        "max": float(selected.max()),
    }


def image_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def mask_bool(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) > 0


def require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def require_positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def require_dimensions(value: Any, label: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(
            not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0
            for dimension in value
        )
    ):
        raise ValueError(f"{label} must be [width, height] positive integers")
    return value[0], value[1]


def resolve_portable_path(value: Any, *, manifest_path: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}.path must be a non-empty string")
    declared = Path(value).expanduser()
    if declared.is_absolute():
        raise ValueError(f"{label}.path must be relative to the input manifest parent")
    manifest_parent = manifest_path.parent.resolve()
    resolved = (manifest_parent / declared).resolve()
    try:
        resolved.relative_to(manifest_parent)
    except ValueError as error:
        raise ValueError(f"{label}.path escapes the input manifest parent") from error
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def resolve_input_record(record: Any, *, manifest_path: Path, index: int) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError(f"frame_records[{index}] must be an object")

    input_frame = record.get("input_frame")
    input_mask = record.get("input_mask")
    has_portable_field = isinstance(input_frame, dict) or isinstance(input_mask, dict)
    if has_portable_field:
        if not isinstance(input_frame, dict) or not isinstance(input_mask, dict):
            raise ValueError(
                f"frame_records[{index}] portable input_frame and input_mask must be objects"
            )
        frame_label = f"frame_records[{index}].input_frame"
        mask_label = f"frame_records[{index}].input_mask"
        source_path = resolve_portable_path(
            input_frame.get("path"), manifest_path=manifest_path, label=frame_label
        )
        union_mask_path = resolve_portable_path(
            input_mask.get("path"), manifest_path=manifest_path, label=mask_label
        )
        expected_frame_sha = require_sha256(input_frame.get("sha256"), f"{frame_label}.sha256")
        expected_mask_sha = require_sha256(input_mask.get("sha256"), f"{mask_label}.sha256")
        expected_frame_bytes = require_positive_int(
            input_frame.get("bytes"), f"{frame_label}.bytes"
        )
        expected_mask_bytes = require_positive_int(input_mask.get("bytes"), f"{mask_label}.bytes")
        if source_path.stat().st_size != expected_frame_bytes:
            raise ValueError(f"{frame_label}.bytes does not match the input file")
        if union_mask_path.stat().st_size != expected_mask_bytes:
            raise ValueError(f"{mask_label}.bytes does not match the input file")
        if sha256_file(source_path) != expected_frame_sha:
            raise ValueError(f"{frame_label}.sha256 does not match the input file")
        if sha256_file(union_mask_path) != expected_mask_sha:
            raise ValueError(f"{mask_label}.sha256 does not match the input file")
        return {
            "source_path": source_path,
            "union_mask_path": union_mask_path,
            "union_mask_pixels": require_positive_int(
                input_mask.get("area_pixels"), f"{mask_label}.area_pixels"
            ),
            "source_dimensions": require_dimensions(
                input_frame.get("dimensions"), f"{frame_label}.dimensions"
            ),
            "mask_dimensions": require_dimensions(
                input_mask.get("dimensions"), f"{mask_label}.dimensions"
            ),
            "portable": True,
        }

    try:
        source_path = Path(record["source_frame"])
        union_mask_path = Path(record["union_mask"])
        union_mask_pixels = record["union_mask_pixels"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"frame_records[{index}] must use legacy source_frame/union_mask fields "
            "or portable input_frame/input_mask objects"
        ) from error
    return {
        "source_path": source_path,
        "union_mask_path": union_mask_path,
        "union_mask_pixels": union_mask_pixels,
        "source_dimensions": None,
        "mask_dimensions": None,
        "portable": False,
    }


def make_contact_sheet(
    records: list[dict[str, Any]],
    *,
    path: Path,
    sample_count: int,
) -> None:
    sample_count = min(sample_count, len(records))
    if sample_count <= 0:
        raise ValueError("contact sheet needs at least one frame")
    sample_indices = np.linspace(0, len(records) - 1, sample_count, dtype=int).tolist()
    cell_width = 400
    cell_height = 225
    label_height = 24
    sheet_size = (cell_width * 3, (cell_height + label_height) * sample_count)
    sheet = Image.new("RGB", sheet_size, "#111111")
    draw = ImageDraw.Draw(sheet)

    for row, record_index in enumerate(sample_indices):
        record = records[record_index]
        source = Image.open(record["source_frame"]).convert("RGB")
        output = Image.open(record["output_frame"]).convert("RGB")
        mask = Image.open(record["union_mask"]).convert("L")
        overlay = source.copy()
        tint = Image.new("RGB", source.size, (255, 0, 170))
        overlay.paste(tint, mask=mask.point(lambda value: int(value * 0.55)))
        images = [source, overlay, output]
        titles = [
            f"source {record['frame_id']}",
            f"remove mask ({record['union_mask_pixels']} px)",
            "ProPainter output",
        ]
        y = row * (cell_height + label_height)
        for column, (image, title) in enumerate(zip(images, titles, strict=True)):
            image.thumbnail((cell_width, cell_height), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (cell_width, cell_height), "#202020")
            offset = ((cell_width - image.width) // 2, (cell_height - image.height) // 2)
            canvas.paste(image, offset)
            x = column * cell_width
            sheet.paste(canvas, (x, y + label_height))
            draw.text((x + 8, y + 6), title, fill="#f4f4f4")
        source.close()
        output.close()
        mask.close()

    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def validate(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.input_manifest.expanduser().resolve()
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("frame_records"), list):
        raise ValueError("input manifest must contain frame_records")

    output_frames = args.output_frames.expanduser().resolve()
    output_paths = sorted(output_frames.glob("*.png"))
    input_records = manifest["frame_records"]
    if len(output_paths) != len(input_records):
        raise ValueError(
            f"output frame count mismatch: expected {len(input_records)}, found {len(output_paths)}"
        )

    records: list[dict[str, Any]] = []
    outside_means: list[float] = []
    inside_means: list[float] = []
    dimensions_passed = True
    for record_index, (record, output_path) in enumerate(
        zip(input_records, output_paths, strict=True)
    ):
        resolved_record = resolve_input_record(
            record,
            manifest_path=manifest_path,
            index=record_index,
        )
        source_path = resolved_record["source_path"]
        union_mask_path = resolved_record["union_mask_path"]
        source = image_rgb(source_path)
        output = image_rgb(output_path)
        union_mask = mask_bool(union_mask_path)
        if output.shape != source.shape or union_mask.shape != source.shape[:2]:
            dimensions_passed = False
            raise ValueError(
                f"dimension mismatch for {output_path}: source={source.shape}, "
                f"output={output.shape}, mask={union_mask.shape}"
            )
        if resolved_record["portable"]:
            source_dimensions = (source.shape[1], source.shape[0])
            mask_dimensions = (union_mask.shape[1], union_mask.shape[0])
            if source_dimensions != resolved_record["source_dimensions"]:
                raise ValueError(
                    f"declared source dimensions do not match decoded input for {source_path}"
                )
            if mask_dimensions != resolved_record["mask_dimensions"]:
                raise ValueError(
                    f"declared mask dimensions do not match decoded input for {union_mask_path}"
                )
            if int(union_mask.sum()) != resolved_record["union_mask_pixels"]:
                raise ValueError(
                    f"declared input_mask.area_pixels does not match decoded mask for "
                    f"{union_mask_path}"
                )
        dilated_mask = binary_dilate(union_mask, args.mask_dilation)
        absolute_delta = np.abs(output.astype(np.int16) - source.astype(np.int16)).mean(axis=2)
        inside = masked_stats(absolute_delta, union_mask)
        outside = masked_stats(absolute_delta, ~dilated_mask)
        inside_means.append(float(inside["mean"]))
        outside_means.append(float(outside["mean"]))
        records.append(
            {
                "sequence_index": record["sequence_index"],
                "frame_id": record["frame_id"],
                "source_frame": str(source_path),
                "union_mask": str(union_mask_path),
                "union_mask_pixels": resolved_record["union_mask_pixels"],
                "output_frame": str(output_path),
                "output_sha256": sha256_file(output_path),
                "inside_original_mask_abs_rgb_delta": inside,
                "outside_dilated_mask_abs_rgb_delta": outside,
                "source_mask_mean_rgb": [float(value) for value in source[union_mask].mean(axis=0)],
                "output_mask_mean_rgb": [float(value) for value in output[union_mask].mean(axis=0)],
            }
        )

    gates = {
        "frame_count_matches": len(output_paths) == len(input_records),
        "dimensions_match": dimensions_passed,
        "outside_dilated_mask_mean_abs_delta_lte": {
            "threshold": args.max_outside_mae,
            "observed_max_frame_mean": max(outside_means, default=float("inf")),
            "passed": max(outside_means, default=float("inf")) <= args.max_outside_mae,
        },
        "inside_mask_mean_abs_delta_gte": {
            "threshold": args.min_inside_mae,
            "observed_min_frame_mean": min(inside_means, default=0.0),
            "passed": min(inside_means, default=0.0) >= args.min_inside_mae,
        },
    }
    technical_passed = all(
        value if isinstance(value, bool) else bool(value["passed"]) for value in gates.values()
    )
    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "technical_passed" if technical_passed else "technical_failed",
        "input_manifest": str(manifest_path),
        "input_manifest_sha256": sha256_file(manifest_path),
        "output_frames": str(output_frames),
        "frame_count": len(records),
        "mask_dilation": args.mask_dilation,
        "gates": gates,
        "frame_records": records,
        "semantic_review_required": True,
        "geometry_reconstruction_required": True,
        "limitations": [
            "Pixel-delta gates verify I/O integrity, not semantic plausibility of "
            "synthesized content.",
            "RGB output must receive multi-view review and feed depth/normal "
            "reconstruction before PGSR or TSDF rebuild.",
            "This report does not certify cross-view 3D consistency or absence of "
            "object/background interpenetration.",
        ],
    }
    write_json(args.output_report.expanduser().resolve(), report)
    make_contact_sheet(
        records,
        path=args.contact_sheet.expanduser().resolve(),
        sample_count=args.samples,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-frames", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--contact-sheet", type=Path, required=True)
    parser.add_argument("--mask-dilation", type=int, default=4)
    parser.add_argument("--max-outside-mae", type=float, default=0.5)
    parser.add_argument("--min-inside-mae", type=float, default=2.0)
    parser.add_argument("--samples", type=int, default=4)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = validate(args)
    print(json.dumps({"status": report["status"], "frame_count": report["frame_count"]}, indent=2))
    return 0 if report["status"] == "technical_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
