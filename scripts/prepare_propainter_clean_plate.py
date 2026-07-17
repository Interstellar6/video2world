#!/usr/bin/env python3
"""Build deterministic ProPainter frame/mask inputs from a filtered SAM mask index."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


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


def parse_frame_ids(args: argparse.Namespace) -> list[str]:
    if args.frame_ids:
        values = [value.strip() for value in args.frame_ids.split(",") if value.strip()]
        if not values:
            raise ValueError("--frame-ids did not contain a frame id")
        return [Path(value).stem for value in values]
    if args.frame_start is None or args.frame_end is None:
        raise ValueError("provide --frame-ids or both --frame-start and --frame-end")
    if args.frame_start < 0 or args.frame_end < args.frame_start:
        raise ValueError("frame range must satisfy 0 <= start <= end")
    return [f"{index:06d}" for index in range(args.frame_start, args.frame_end + 1)]


def resolve_mask_path(value: str, *, mask_index: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = mask_index.parent / path
    return path.resolve()


def bbox_state(item: dict[str, Any]) -> tuple[float, float, float, float]:
    bbox = item.get("bbox")
    if (
        not isinstance(bbox, list)
        or len(bbox) != 4
        or not all(isinstance(value, int | float) for value in bbox)
    ):
        raise ValueError("tracked mask items must have a four-number bbox")
    x0, y0, x1, y1 = (float(value) for value in bbox)
    width = x1 - x0
    height = y1 - y0
    if width <= 0 or height <= 0:
        raise ValueError(f"tracked mask bbox has non-positive extent: {bbox}")
    return ((x0 + x1) / 2, (y0 + y1) / 2, width, height)


def tracking_cost(previous: dict[str, Any], candidate: dict[str, Any]) -> float:
    previous_x, previous_y, previous_width, previous_height = bbox_state(previous)
    candidate_x, candidate_y, candidate_width, candidate_height = bbox_state(candidate)
    previous_diagonal = math.hypot(previous_width, previous_height)
    center_cost = math.hypot(candidate_x - previous_x, candidate_y - previous_y) / max(
        previous_diagonal, 1.0
    )
    shape_cost = abs(math.log(candidate_width / previous_width)) + abs(
        math.log(candidate_height / previous_height)
    )
    return center_cost + 0.25 * shape_cost


def track_single_instance(
    *,
    frame_ids: list[str],
    indexed_masks: dict[str, list[dict[str, Any]]],
    anchor_frame: str,
    anchor_mask_name: str,
    max_cost: float,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if anchor_frame not in frame_ids:
        raise ValueError(f"track anchor frame {anchor_frame} is outside the selected frames")
    anchor_matches = [
        item
        for item in indexed_masks.get(anchor_frame, [])
        if Path(str(item.get("mask_path", ""))).name == anchor_mask_name
    ]
    if len(anchor_matches) != 1:
        raise ValueError(
            f"expected one anchor mask named {anchor_mask_name!r} in {anchor_frame}, "
            f"found {len(anchor_matches)}"
        )

    anchor_index = frame_ids.index(anchor_frame)
    selected: dict[str, dict[str, Any]] = {anchor_frame: anchor_matches[0]}
    costs: dict[str, float] = {anchor_frame: 0.0}
    directions = [
        range(anchor_index - 1, -1, -1),
        range(anchor_index + 1, len(frame_ids)),
    ]
    for indices in directions:
        previous = anchor_matches[0]
        for index in indices:
            frame_id = frame_ids[index]
            candidates = indexed_masks.get(frame_id, [])
            if not candidates:
                raise ValueError(f"no candidate masks available for tracked frame {frame_id}")
            cost, best = min(
                ((tracking_cost(previous, item), item) for item in candidates),
                key=lambda pair: pair[0],
            )
            if cost > max_cost:
                raise ValueError(
                    f"instance tracking cost {cost:.4f} exceeds {max_cost:.4f} at frame {frame_id}"
                )
            selected[frame_id] = best
            costs[frame_id] = cost
            previous = best

    selected_lists = {frame_id: [selected[frame_id]] for frame_id in frame_ids}
    tracking = {
        "mode": "single_instance_bbox_continuity",
        "anchor_frame": anchor_frame,
        "anchor_mask_name": anchor_mask_name,
        "max_cost": max_cost,
        "per_frame": [
            {
                "frame_id": frame_id,
                "mask_path": selected[frame_id]["mask_path"],
                "bbox": selected[frame_id]["bbox"],
                "tracking_cost_from_adjacent": costs[frame_id],
            }
            for frame_id in frame_ids
        ],
    }
    return selected_lists, tracking


def build_inputs(args: argparse.Namespace) -> dict[str, Any]:
    frames_dir = args.frames_dir.expanduser().resolve()
    mask_index_path = args.mask_index.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_root}")

    frame_ids = parse_frame_ids(args)
    frame_paths: dict[str, Path] = {}
    for frame_id in frame_ids:
        candidates = sorted(frames_dir.glob(f"{frame_id}.*"))
        if len(candidates) != 1:
            raise ValueError(
                f"expected exactly one source frame for {frame_id!r}, found {len(candidates)}"
            )
        frame_paths[frame_id] = candidates[0]

    mask_index = read_json(mask_index_path)
    if not isinstance(mask_index, dict) or not isinstance(mask_index.get("items"), list):
        raise ValueError("mask index must be an object with an items array")

    indexed_masks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in mask_index["items"]:
        if not isinstance(item, dict):
            raise ValueError("every mask index item must be an object")
        image = item.get("image")
        label = item.get("label")
        score = item.get("score")
        if not isinstance(image, str):
            raise ValueError("mask index item.image must be a string")
        if args.label is not None and label != args.label:
            continue
        if args.min_score is not None and (
            not isinstance(score, int | float) or score < args.min_score
        ):
            continue
        indexed_masks[Path(image).stem].append(item)

    tracking: dict[str, Any] | None = None
    if args.track_anchor_frame is not None:
        if args.track_anchor_mask is None:
            raise ValueError("--track-anchor-mask is required with --track-anchor-frame")
        anchor_frame = Path(args.track_anchor_frame).stem
        indexed_masks, tracking = track_single_instance(
            frame_ids=frame_ids,
            indexed_masks=indexed_masks,
            anchor_frame=anchor_frame,
            anchor_mask_name=args.track_anchor_mask,
            max_cost=args.max_track_cost,
        )

    frame_output_dir = output_root / "frames"
    mask_output_dir = output_root / "masks"
    frame_output_dir.mkdir(parents=True, exist_ok=True)
    mask_output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for sequence_index, frame_id in enumerate(frame_ids):
        source_frame = frame_paths[frame_id]
        with Image.open(source_frame) as frame_image:
            frame_size = frame_image.size

        source_items = indexed_masks.get(frame_id, [])
        if not source_items and not args.allow_empty_masks:
            raise ValueError(f"no accepted masks for selected frame {frame_id}")

        union = np.zeros((frame_size[1], frame_size[0]), dtype=bool)
        mask_sources: list[dict[str, Any]] = []
        for item in source_items:
            mask_path_value = item.get("mask_path")
            if not isinstance(mask_path_value, str):
                raise ValueError(f"mask item for {frame_id} has no string mask_path")
            mask_path = resolve_mask_path(mask_path_value, mask_index=mask_index_path)
            if not mask_path.is_file():
                raise FileNotFoundError(mask_path)
            with Image.open(mask_path) as mask_image:
                mask = np.asarray(mask_image.convert("L")) > 0
            if mask.shape != union.shape:
                raise ValueError(
                    f"mask dimensions {mask.shape[::-1]} do not match frame "
                    f"{frame_size}: {mask_path}"
                )
            union |= mask
            mask_sources.append(
                {
                    "path": str(mask_path),
                    "sha256": sha256_file(mask_path),
                    "score": item.get("score"),
                    "bbox": item.get("bbox"),
                    "label": item.get("label"),
                    "pixels": int(mask.sum()),
                }
            )

        output_name = f"{sequence_index:04d}.png"
        output_frame = frame_output_dir / output_name
        output_mask = mask_output_dir / output_name
        shutil.copy2(source_frame, output_frame)
        Image.fromarray(union.astype(np.uint8) * 255, mode="L").save(output_mask)
        records.append(
            {
                "sequence_index": sequence_index,
                "frame_id": frame_id,
                "source_frame": str(source_frame),
                "source_frame_sha256": sha256_file(source_frame),
                "input_frame": str(output_frame),
                "input_frame_sha256": sha256_file(output_frame),
                "union_mask": str(output_mask),
                "union_mask_sha256": sha256_file(output_mask),
                "width": frame_size[0],
                "height": frame_size[1],
                "union_mask_pixels": int(union.sum()),
                "union_mask_fraction": float(union.mean()),
                "source_mask_count": len(mask_sources),
                "source_masks": mask_sources,
            }
        )

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "purpose": "video2world layered clean-plate input",
        "method": (
            "tracked single-instance masks from filtered SAM mask index"
            if tracking is not None
            else "union accepted frame masks from filtered SAM mask index"
        ),
        "semantic_scope": args.label,
        "tracking": tracking,
        "frames_dir": str(frames_dir),
        "mask_index": str(mask_index_path),
        "mask_index_sha256": sha256_file(mask_index_path),
        "output_root": str(output_root),
        "frame_count": len(records),
        "frame_ids": frame_ids,
        "frame_records": records,
        "limitations": [
            (
                "BBox-continuity tracking is deterministic but should be validated against "
                "multi-view 3D identity before promotion."
                if tracking is not None
                else "A class-union mask does not preserve per-instance identity."
            ),
            "RGB clean plates do not repair depth, geometry, or Gaussian splats by themselves.",
        ],
    }
    write_json(output_root / "input_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--mask-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--frame-ids", help="Comma-separated frame ids or filenames")
    selection.add_argument("--frame-start", type=int, help="First six-digit frame index")
    parser.add_argument("--frame-end", type=int, help="Last inclusive frame index")
    parser.add_argument("--label", default="pillow")
    parser.add_argument("--min-score", type=float)
    parser.add_argument("--allow-empty-masks", action="store_true")
    parser.add_argument("--track-anchor-frame", help="Frame id containing the tracked anchor")
    parser.add_argument(
        "--track-anchor-mask",
        help="Anchor mask basename, for example pillow_03.png",
    )
    parser.add_argument("--max-track-cost", type=float, default=0.5)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = build_inputs(args)
    summary = {
        "manifest": str(args.output / "input_manifest.json"),
        "frame_count": manifest["frame_count"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
