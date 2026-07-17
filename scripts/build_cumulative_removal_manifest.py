#!/usr/bin/env python3
"""Build a strict front-to-back cumulative object-removal manifest.

Each round unions the current object's observed mask with every earlier removal
mask. Round 2 and later source RGB must come from the immediately preceding
multi-view prefill report. Donors remain original observed RGB and are paired
with the cumulative exclusion mask, so neither removed objects nor unresolved
pixels can become donor evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

KIND = "video2world.cumulative_removal_manifest"
DONOR_INDEX_KIND = "video2world.cumulative_donor_exclusion_index"


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


def require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def require_records(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty array")
    if not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{label} entries must be objects")
    return value


def resolve_path(value: str, *, relative_to: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def resolve_asset(record: dict[str, Any], key: str, *, manifest_path: Path) -> Path:
    value = record.get(key)
    if isinstance(value, str):
        path_value = value
    elif isinstance(value, dict) and isinstance(value.get("path"), str):
        path_value = value["path"]
    else:
        raise ValueError(f"frame {record.get('frame_id')} has no {key} path")
    path = resolve_path(path_value, relative_to=manifest_path.parent)
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_sha: Any = None
    if isinstance(value, dict):
        expected_sha = value.get("sha256")
    expected_sha = expected_sha or record.get(f"{key}_sha256")
    if isinstance(expected_sha, str) and sha256_file(path) != expected_sha:
        raise ValueError(f"frame {record.get('frame_id')} {key} SHA-256 mismatch")
    return path


def mask_array(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) > 0


def image_dimensions(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def frame_map(manifest: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in require_records(manifest.get("frame_records"), f"{label}.frame_records"):
        frame_id = record.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id:
            raise ValueError(f"{label} has a frame without frame_id")
        if frame_id in result:
            raise ValueError(f"{label} repeats frame {frame_id}")
        result[frame_id] = record
    return result


def resolve_raw_frame(frames_dir: Path, frame_id: str) -> Path:
    matches = sorted(frames_dir.glob(f"{frame_id}.*"))
    if len(matches) != 1:
        raise ValueError(f"expected one original donor RGB for {frame_id}, found {len(matches)}")
    return matches[0].resolve()


def current_mask(record: dict[str, Any], *, manifest_path: Path) -> Path:
    for key in ("input_mask", "union_mask", "source_mask"):
        try:
            return resolve_asset(record, key, manifest_path=manifest_path)
        except ValueError:
            continue
    raise ValueError(f"frame {record.get('frame_id')} has no current object mask")


def initial_source(record: dict[str, Any], *, manifest_path: Path) -> Path:
    for key in ("input_frame", "source_frame"):
        try:
            return resolve_asset(record, key, manifest_path=manifest_path)
        except ValueError:
            continue
    raise ValueError(f"frame {record.get('frame_id')} has no source frame")


def validate_previous_lineage(
    *,
    current_round_index: int,
    previous_manifest_path: Path,
    previous_report_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    previous_manifest = require_dict(read_json(previous_manifest_path), "previous manifest")
    if previous_manifest.get("kind") != KIND:
        raise ValueError("previous manifest is not a cumulative-removal manifest")
    if previous_manifest.get("round_index") != current_round_index - 1:
        raise ValueError("previous manifest is not the immediately preceding round")
    previous_report = require_dict(read_json(previous_report_path), "previous prefill report")
    if previous_report.get("status") != "technical_passed":
        raise ValueError("previous prefill report did not technically pass")
    gates = require_dict(previous_report.get("gates"), "previous report gates")
    if gates.get("outside_removal_mask_rgb_exact") is not True:
        raise ValueError("previous prefill did not preserve RGB outside its removal mask")
    if gates.get("all_residual_masks_subset_of_removal_masks") is not True:
        raise ValueError("previous residual masks were not subsets of removal masks")
    provenance = require_dict(
        previous_report.get("pixel_provenance"), "previous report pixel provenance"
    )
    if provenance.get("generated_pixels") != 0:
        raise ValueError("previous prefill report contains or omits generated pixel provenance")
    if provenance.get("propainter_pixels") != 0:
        raise ValueError("previous prefill report contains or omits ProPainter pixel provenance")
    if provenance.get("unresolved_pixels_are_not_valid_donor_or_geometry_evidence") is not True:
        raise ValueError("previous prefill report allows unresolved pixels as evidence")
    expected_manifest_sha = sha256_file(previous_manifest_path)
    if previous_report.get("input_manifest_sha256") != expected_manifest_sha:
        raise ValueError("previous report does not hash the previous cumulative manifest")
    report_records = frame_map(previous_report, "previous report")
    manifest_records = frame_map(previous_manifest, "previous manifest")
    if set(report_records) != set(manifest_records):
        raise ValueError("previous report and manifest cover different frames")
    return manifest_records, report_records, previous_report


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    current_manifest_path = args.current_object_manifest.expanduser().resolve()
    donor_frames_dir = args.observed_donor_frames_dir.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    if args.round_index < 1:
        raise ValueError("round-index must be positive")
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    if not donor_frames_dir.is_dir():
        raise NotADirectoryError(donor_frames_dir)

    current_manifest = require_dict(read_json(current_manifest_path), "current object manifest")
    current_records = frame_map(current_manifest, "current object manifest")
    frame_ids = sorted(current_records)
    manifest_object_id = current_manifest.get("object_id")
    object_id = args.object_id or manifest_object_id
    if not isinstance(object_id, str) or not object_id:
        raise ValueError("object-id is required when the current manifest has no object_id")
    if isinstance(manifest_object_id, str) and manifest_object_id != object_id:
        raise ValueError("object-id does not match the current object manifest")

    previous_manifest_path = (
        args.previous_cumulative_manifest.expanduser().resolve()
        if args.previous_cumulative_manifest
        else None
    )
    previous_report_path = (
        args.previous_prefill_report.expanduser().resolve()
        if args.previous_prefill_report
        else None
    )
    if args.round_index == 1 and (previous_manifest_path or previous_report_path):
        raise ValueError("round 1 cannot have previous-round lineage")
    if args.round_index > 1 and not (previous_manifest_path and previous_report_path):
        raise ValueError("round 2+ requires previous cumulative manifest and prefill report")

    previous_records: dict[str, dict[str, Any]] = {}
    previous_report_records: dict[str, dict[str, Any]] = {}
    previous_report: dict[str, Any] | None = None
    if previous_manifest_path and previous_report_path:
        previous_records, previous_report_records, previous_report = validate_previous_lineage(
            current_round_index=args.round_index,
            previous_manifest_path=previous_manifest_path,
            previous_report_path=previous_report_path,
        )
        if set(previous_records) != set(current_records):
            raise ValueError("current and previous manifests cover different frames")

    donor_assets: dict[str, dict[str, Any]] = {}
    for frame_id in frame_ids:
        donor_path = resolve_raw_frame(donor_frames_dir, frame_id)
        donor_assets[frame_id] = {
            "path": str(donor_path),
            "sha256": sha256_file(donor_path),
            "bytes": donor_path.stat().st_size,
            "dimensions": list(image_dimensions(donor_path)),
            "role": "original_observed_rgb_only",
        }

    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_root.parent,
        prefix=f".{output_root.name}.staging-",
    ) as temporary_value:
        staging = Path(temporary_value)
        masks_dir = staging / "masks"
        masks_dir.mkdir(parents=True)
        output_records: list[dict[str, Any]] = []
        donor_index_items: list[dict[str, Any]] = []

        for sequence_index, frame_id in enumerate(frame_ids):
            record = current_records[frame_id]
            object_mask_path = current_mask(record, manifest_path=current_manifest_path)
            object_mask = mask_array(object_mask_path)
            raw_asset = donor_assets[frame_id]
            raw_dimensions = tuple(raw_asset["dimensions"])
            if object_mask.shape[::-1] != raw_dimensions:
                raise ValueError(f"current mask dimensions do not match raw RGB for {frame_id}")

            components = [
                {
                    "object_id": object_id,
                    "round_index": args.round_index,
                    "role": "current_observed_object_mask",
                    "path": str(object_mask_path),
                    "sha256": sha256_file(object_mask_path),
                    "pixels": int(object_mask.sum()),
                }
            ]
            cumulative_mask = object_mask.copy()
            input_lineage: dict[str, Any]
            if args.round_index == 1:
                source_path = initial_source(record, manifest_path=current_manifest_path)
                source_sha = sha256_file(source_path)
                if source_sha != raw_asset["sha256"]:
                    raise ValueError(f"round 1 source is not original observed RGB for {frame_id}")
                prior_object_ids: list[str] = []
                input_lineage = {
                    "role": "original_observed_rgb",
                    "source_sha256": source_sha,
                }
            else:
                previous_record = previous_records[frame_id]
                previous_mask_path = resolve_asset(
                    previous_record,
                    "union_mask",
                    manifest_path=previous_manifest_path,  # type: ignore[arg-type]
                )
                previous_mask = mask_array(previous_mask_path)
                if previous_mask.shape != object_mask.shape:
                    raise ValueError(f"previous/current mask shape mismatch for {frame_id}")
                cumulative_mask |= previous_mask
                prior_object_ids = list(previous_record.get("removed_object_ids", []))
                if len(prior_object_ids) != args.round_index - 1:
                    raise ValueError("previous removed-object count does not match round index")
                components = list(previous_record.get("mask_components", [])) + components

                report_record = previous_report_records[frame_id]
                source_path = resolve_path(
                    str(report_record["prefill_frame"]),
                    relative_to=previous_report_path.parent,  # type: ignore[union-attr]
                )
                if sha256_file(source_path) != report_record.get("prefill_frame_sha256"):
                    raise ValueError(f"previous prefill frame SHA-256 mismatch for {frame_id}")
                residual_path = resolve_path(
                    str(report_record["residual_mask"]),
                    relative_to=previous_report_path.parent,  # type: ignore[union-attr]
                )
                residual = mask_array(residual_path)
                if residual.shape != previous_mask.shape or np.any(residual & ~previous_mask):
                    raise ValueError(f"previous residual escapes removal mask for {frame_id}")
                input_lineage = {
                    "role": "immediately_previous_measured_prefill_with_unresolved_mask",
                    "previous_prefill_frame_sha256": report_record["prefill_frame_sha256"],
                    "previous_residual_mask_sha256": report_record["residual_mask_sha256"],
                    "previous_residual_pixels": report_record["residual_mask_pixels"],
                    "previous_unresolved_pixels_remain_non_donor": True,
                }

            removed_object_ids = [*prior_object_ids, object_id]
            if len(set(removed_object_ids)) != len(removed_object_ids):
                raise ValueError("the same object appears in more than one peel round")
            output_name = f"{sequence_index:04d}.png"
            cumulative_path = masks_dir / output_name
            Image.fromarray(cumulative_mask.astype(np.uint8) * 255).save(cumulative_path)
            relative_mask_path = f"masks/{output_name}"
            cumulative_sha = sha256_file(cumulative_path)
            source_sha = sha256_file(source_path)
            output_records.append(
                {
                    "sequence_index": sequence_index,
                    "frame_id": frame_id,
                    "source_frame": str(source_path),
                    "source_frame_sha256": source_sha,
                    "source_role": input_lineage["role"],
                    "union_mask": relative_mask_path,
                    "union_mask_sha256": cumulative_sha,
                    "union_mask_pixels": int(cumulative_mask.sum()),
                    "donor_exclusion_mask": relative_mask_path,
                    "donor_exclusion_mask_sha256": cumulative_sha,
                    "removed_object_ids": removed_object_ids,
                    "mask_components": components,
                    "current_object_mask_sha256": sha256_file(object_mask_path),
                    "current_object_mask_pixels": int(object_mask.sum()),
                    "input_lineage": input_lineage,
                }
            )
            donor_index_items.append(
                {
                    "image": f"{frame_id}.png",
                    "frame_id": frame_id,
                    "label": "cumulative_removed",
                    "mask_path": relative_mask_path,
                    "mask_sha256": cumulative_sha,
                    "removed_object_ids": removed_object_ids,
                    "unresolved_residual_allowed_as_donor": False,
                }
            )

        donor_index = {
            "schema_version": 1,
            "kind": DONOR_INDEX_KIND,
            "round_index": args.round_index,
            "removed_object_ids": output_records[0]["removed_object_ids"],
            "label": "cumulative_removed",
            "items": donor_index_items,
        }
        donor_index_path = staging / "donor_exclusion_index.json"
        write_json(donor_index_path, donor_index)

        lineage: dict[str, Any] = {
            "previous_round_index": args.round_index - 1 if args.round_index > 1 else None,
            "previous_cumulative_manifest": (
                str(previous_manifest_path) if previous_manifest_path else None
            ),
            "previous_cumulative_manifest_sha256": (
                sha256_file(previous_manifest_path) if previous_manifest_path else None
            ),
            "previous_prefill_report": str(previous_report_path) if previous_report_path else None,
            "previous_prefill_report_sha256": (
                sha256_file(previous_report_path) if previous_report_path else None
            ),
            "previous_prefill_status": previous_report.get("status") if previous_report else None,
        }
        manifest = {
            "schema_version": 1,
            "kind": KIND,
            "status": "ready_for_measured_multiview_prefill",
            "created_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 - mil8 is 3.10
            "round_index": args.round_index,
            "current_object_id": object_id,
            "removed_object_ids": output_records[0]["removed_object_ids"],
            "current_object_manifest": str(current_manifest_path),
            "current_object_manifest_sha256": sha256_file(current_manifest_path),
            "lineage": lineage,
            "donor_contract": {
                "role": "original_observed_rgb_only",
                "frames_dir": str(donor_frames_dir),
                "assets": donor_assets,
                "exclusion_label": "cumulative_removed",
                "exclusion_index": "donor_exclusion_index.json",
                "exclusion_index_sha256": sha256_file(donor_index_path),
                "generated_or_prefilled_rgb_as_donor": False,
                "unresolved_residual_as_donor": False,
            },
            "frame_count": len(output_records),
            "frame_records": output_records,
            "gates": {
                "strict_front_to_back_order": True,
                "input_is_immediately_previous_round": True,
                "all_previous_masks_are_accumulated": True,
                "all_removed_objects_are_excluded_from_donors": True,
                "donors_are_original_observed_rgb": True,
                "unresolved_residual_is_never_donor_evidence": True,
                "previous_prefill_excludes_generated_and_propainter_pixels": True,
            },
            "pixel_provenance_classes": {
                "outside_cumulative_mask": "source scene RGB preserved exactly",
                "measured_multiview": "pending calibrated RGB-D donor reprojection",
                "unresolved_unobserved": "pending; must remain in residual mask",
                "generated": "forbidden in this measurement stage",
            },
        }
        manifest_path = staging / "cumulative_removal_manifest.json"
        write_json(manifest_path, manifest)
        receipt = {
            "schema_version": 1,
            "kind": "video2world.cumulative_removal_manifest_receipt",
            "manifest": "cumulative_removal_manifest.json",
            "manifest_sha256": sha256_file(manifest_path),
            "donor_exclusion_index": "donor_exclusion_index.json",
            "donor_exclusion_index_sha256": sha256_file(donor_index_path),
        }
        write_json(staging / "receipt.json", receipt)
        shutil.move(str(staging), output_root)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-index", type=int, required=True)
    parser.add_argument("--object-id")
    parser.add_argument("--current-object-manifest", type=Path, required=True)
    parser.add_argument("--previous-cumulative-manifest", type=Path)
    parser.add_argument("--previous-prefill-report", type=Path)
    parser.add_argument("--observed-donor-frames-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = build_manifest(args)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "round_index": manifest["round_index"],
                "removed_object_ids": manifest["removed_object_ids"],
                "frame_count": manifest["frame_count"],
                "output": str(args.output / "cumulative_removal_manifest.json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
