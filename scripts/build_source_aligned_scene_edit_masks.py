#!/usr/bin/env python3
"""Build source-bound scene-edit masks for one front-to-back peel round.

Each frame record names the exact RGB image on which its observed target mask
was produced. Round 1 emits that observed mask unchanged. Round 2 and later
recover the target inside the previously edited region from the previous
contributor-label image:

    scene_edit = (observed & ~previous_region)
                 | (previous_region & (previous_labels == target_label))

The tool validates every input hash and dimension before atomically publishing
the masks, report, and receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

INPUT_KIND = "video2world.source_aligned_scene_edit_mask_input"
REPORT_KIND = "video2world.source_aligned_scene_edit_mask_report"
RECEIPT_KIND = "video2world.source_aligned_scene_edit_mask_receipt"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON {path}: {error}") from error


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
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


def require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def require_records(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("frame_records must be a non-empty array")
    records = []
    for index, value_item in enumerate(value):
        records.append(require_dict(value_item, f"frame_records[{index}]"))
    return records


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def resolve_asset(
    value: Any,
    *,
    relative_to: Path,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    record = require_dict(value, label)
    path_value = record.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{label}.path must be a non-empty string")
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = require_sha256(record.get("sha256"), f"{label}.sha256")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {actual} != {expected}")
    return path, record


def load_rgb(path: Path, label: str) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            if image.mode not in {"RGB", "RGBA"}:
                raise ValueError(f"{label} must be RGB or RGBA, got {image.mode}")
            width, height = image.size
    except OSError as error:
        raise ValueError(f"{label} is not a decodable image: {path}") from error
    if width <= 0 or height <= 0:
        raise ValueError(f"{label} has invalid dimensions")
    return width, height


def load_binary_mask(path: Path, label: str, *, allow_empty: bool = False) -> np.ndarray:
    try:
        with Image.open(path) as image:
            value = np.asarray(image.convert("L"), dtype=np.uint8)
    except OSError as error:
        raise ValueError(f"{label} is not a decodable mask: {path}") from error
    if value.ndim != 2 or not np.all((value == 0) | (value == 255)):
        raise ValueError(f"{label} must be a binary 0/255 mask")
    mask = value == 255
    if not allow_empty and not np.any(mask):
        raise ValueError(f"{label} must not be empty")
    return mask


def load_contributor_labels(path: Path, label: str) -> np.ndarray:
    try:
        with Image.open(path) as image:
            value = np.asarray(image)
    except OSError as error:
        raise ValueError(f"{label} is not a decodable image: {path}") from error
    if value.ndim != 2 or not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"{label} must be a single-channel integer label image")
    labels = value.astype(np.int64, copy=False)
    if np.any(labels < 0):
        raise ValueError(f"{label} must not contain negative labels")
    return labels


def require_target_label(value: Any, label: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
        or value > np.iinfo(np.int32).max
    ):
        raise ValueError(f"{label} must be a positive 32-bit integer")
    return value


def optional_previous_assets(
    record: dict[str, Any],
    *,
    round_index: int,
    base: Path,
    frame_label: str,
) -> tuple[tuple[Path, dict[str, Any]] | None, tuple[Path, dict[str, Any]] | None]:
    region_value = record.get("previous_edited_region_mask")
    labels_value = record.get("previous_contributor_labels")
    region_absent = region_value is None
    labels_absent = labels_value is None
    if region_absent != labels_absent:
        raise ValueError(f"{frame_label} previous region and labels must be supplied together")
    if round_index == 1:
        if not region_absent:
            raise ValueError(f"{frame_label} round 1 must not provide previous region or labels")
        return None, None
    if region_absent:
        raise ValueError(f"{frame_label} round {round_index} requires previous region and labels")
    return (
        resolve_asset(
            region_value,
            relative_to=base,
            label=f"{frame_label}.previous_edited_region_mask",
        ),
        resolve_asset(
            labels_value,
            relative_to=base,
            label=f"{frame_label}.previous_contributor_labels",
        ),
    )


def relative_asset(path: Path, *, relative_to: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def build_scene_edit_masks(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.input_manifest.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")

    manifest = require_dict(read_json(manifest_path), "input manifest")
    if manifest.get("schema_version") != 1 or manifest.get("kind") != INPUT_KIND:
        raise ValueError("input manifest kind or schema_version is invalid")
    round_index = manifest.get("round_index")
    if not isinstance(round_index, int) or isinstance(round_index, bool) or round_index < 1:
        raise ValueError("round_index must be a positive integer")
    records = require_records(manifest.get("frame_records"))
    sequence_indices = [record.get("sequence_index") for record in records]
    if sequence_indices != list(range(len(records))):
        raise ValueError("frame sequence_index values must be contiguous and ordered")
    frame_ids = [record.get("frame_id") for record in records]
    if any(not isinstance(frame_id, str) or not frame_id for frame_id in frame_ids):
        raise ValueError("every frame record must have a non-empty frame_id")
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError("frame_id values must be unique")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_root.parent,
        prefix=f".{output_root.name}.staging-",
    ) as temporary_value:
        staging = Path(temporary_value)
        output_records: list[dict[str, Any]] = []
        output_set: list[dict[str, Any]] = []

        for record in records:
            sequence_index = int(record["sequence_index"])
            frame_id = str(record["frame_id"])
            frame_label = f"frame {frame_id}"
            target_label = require_target_label(
                record.get("target_label"),
                f"{frame_label}.target_label",
            )
            source_path, source_record = resolve_asset(
                record.get("exact_source_rgb"),
                relative_to=manifest_path.parent,
                label=f"{frame_label}.exact_source_rgb",
            )
            width, height = load_rgb(source_path, f"{frame_label} exact source RGB")
            source_sha = require_sha256(
                source_record.get("sha256"), f"{frame_label}.exact_source_rgb.sha256"
            )

            observed_path, observed_record = resolve_asset(
                record.get("observed_target_mask"),
                relative_to=manifest_path.parent,
                label=f"{frame_label}.observed_target_mask",
            )
            observed_source_sha = require_sha256(
                observed_record.get("source_rgb_sha256"),
                f"{frame_label}.observed_target_mask.source_rgb_sha256",
            )
            if observed_source_sha != source_sha:
                raise ValueError(
                    f"{frame_label} observed target mask is not bound to the exact source RGB"
                )
            observed = load_binary_mask(
                observed_path,
                f"{frame_label} observed target mask",
            )
            if observed.shape != (height, width):
                raise ValueError(
                    f"{frame_label} observed target mask dimensions differ from source"
                )

            previous_region_asset, previous_labels_asset = optional_previous_assets(
                record,
                round_index=round_index,
                base=manifest_path.parent,
                frame_label=frame_label,
            )
            if previous_region_asset is None or previous_labels_asset is None:
                previous_region = np.zeros_like(observed)
                target_contributor_inside_previous = np.zeros_like(observed)
                previous_region_record = None
                previous_labels_record = None
                previous_region_path = None
                previous_labels_path = None
            else:
                previous_region_path, previous_region_record = previous_region_asset
                previous_labels_path, previous_labels_record = previous_labels_asset
                previous_region = load_binary_mask(
                    previous_region_path,
                    f"{frame_label} previous edited-region mask",
                )
                previous_labels = load_contributor_labels(
                    previous_labels_path,
                    f"{frame_label} previous contributor labels",
                )
                if previous_region.shape != observed.shape:
                    raise ValueError(
                        f"{frame_label} previous edited-region dimensions differ from source"
                    )
                if previous_labels.shape != observed.shape:
                    raise ValueError(
                        f"{frame_label} previous contributor-label dimensions differ from source"
                    )
                target_contributor_inside_previous = previous_region & (
                    previous_labels == target_label
                )

            observed_outside_previous = observed & ~previous_region
            scene_edit = observed_outside_previous | target_contributor_inside_previous
            if not np.any(scene_edit):
                raise ValueError(f"{frame_label} derived scene-edit mask must not be empty")
            if np.any(observed_outside_previous & previous_region):
                raise ValueError(f"{frame_label} observed-outside-previous partition failed")
            if np.any(target_contributor_inside_previous & ~previous_region):
                raise ValueError(f"{frame_label} contributor-inside-previous partition failed")
            if not np.array_equal(
                scene_edit,
                observed_outside_previous | target_contributor_inside_previous,
            ):
                raise ValueError(f"{frame_label} scene-edit formula failed")

            mask_path = staging / "masks" / f"{sequence_index:04d}.png"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(scene_edit.astype(np.uint8) * 255).save(mask_path)
            mask_asset = relative_asset(mask_path, relative_to=staging)
            mask_asset["pixels"] = int(scene_edit.sum())
            output_record = {
                "sequence_index": sequence_index,
                "frame_id": frame_id,
                "target_label": target_label,
                "exact_source_rgb": {
                    "path": str(source_path),
                    "sha256": source_sha,
                    "width": width,
                    "height": height,
                },
                "observed_target_mask": {
                    "path": str(observed_path),
                    "sha256": sha256_file(observed_path),
                    "source_rgb_sha256": observed_source_sha,
                    "pixels": int(observed.sum()),
                },
                "previous_edited_region_mask": (
                    {
                        "path": str(previous_region_path),
                        "sha256": require_sha256(
                            previous_region_record.get("sha256"),
                            f"{frame_label}.previous_edited_region_mask.sha256",
                        ),
                        "pixels": int(previous_region.sum()),
                    }
                    if previous_region_record is not None
                    else None
                ),
                "previous_contributor_labels": (
                    {
                        "path": str(previous_labels_path),
                        "sha256": require_sha256(
                            previous_labels_record.get("sha256"),
                            f"{frame_label}.previous_contributor_labels.sha256",
                        ),
                    }
                    if previous_labels_record is not None
                    else None
                ),
                "counts": {
                    "observed_target_pixels": int(observed.sum()),
                    "previous_edited_region_pixels": int(previous_region.sum()),
                    "observed_target_inside_previous_pixels": int(
                        (observed & previous_region).sum()
                    ),
                    "observed_target_outside_previous_pixels": int(observed_outside_previous.sum()),
                    "target_contributor_inside_previous_pixels": int(
                        target_contributor_inside_previous.sum()
                    ),
                    "scene_edit_mask_pixels": int(scene_edit.sum()),
                },
                "scene_edit_mask": mask_asset,
                "gates": {
                    "observed_mask_bound_to_exact_source_rgb": True,
                    "input_dimensions_match_exact_source_rgb": True,
                    "previous_inputs_pair_contract_satisfied": True,
                    "scene_edit_formula_exact": True,
                    "scene_edit_mask_nonempty": True,
                },
            }
            output_records.append(output_record)
            output_set.append(
                {
                    "sequence_index": sequence_index,
                    "frame_id": frame_id,
                    "exact_source_rgb_sha256": source_sha,
                    "scene_edit_mask_sha256": mask_asset["sha256"],
                }
            )

        aggregate_counts = {
            key: sum(record["counts"][key] for record in output_records)
            for key in output_records[0]["counts"]
        }
        output_set_sha = digest_json(output_set)
        report = {
            "schema_version": 1,
            "kind": REPORT_KIND,
            "status": "technical_passed",
            "created_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
            "round_index": round_index,
            "input_manifest": str(manifest_path),
            "input_manifest_sha256": sha256_file(manifest_path),
            "frame_count": len(output_records),
            "frame_records": output_records,
            "aggregate_counts": aggregate_counts,
            "output_mask_set_sha256": output_set_sha,
            "formula": (
                "(observed_target_mask AND NOT previous_edited_region_mask) OR "
                "(previous_edited_region_mask AND previous_contributor_labels == target_label)"
            ),
            "gates": {
                "all_input_hashes_verified": True,
                "all_observed_masks_bound_to_exact_source_rgb": True,
                "all_input_dimensions_match": True,
                "all_previous_input_pairs_valid_for_round": True,
                "all_scene_edit_masks_follow_formula": True,
                "all_scene_edit_masks_nonempty": True,
            },
            "output_receipt": "receipt.json",
        }
        report_path = staging / "source_aligned_scene_edit_mask_report.json"
        write_json(report_path, report)
        receipt = {
            "schema_version": 1,
            "kind": RECEIPT_KIND,
            "status": report["status"],
            "round_index": round_index,
            "report": report_path.name,
            "report_sha256": sha256_file(report_path),
            "input_manifest_sha256": report["input_manifest_sha256"],
            "frame_count": report["frame_count"],
            "output_mask_set_sha256": output_set_sha,
            "frame_artifacts_are_hashed_in_report": True,
            "exact_source_rgb_sha256_bound_per_frame": True,
        }
        write_json(staging / "receipt.json", receipt)
        shutil.move(str(staging), output_root)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = build_scene_edit_masks(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "round_index": report["round_index"],
                "frame_count": report["frame_count"],
                "output": str(args.output.expanduser().resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
