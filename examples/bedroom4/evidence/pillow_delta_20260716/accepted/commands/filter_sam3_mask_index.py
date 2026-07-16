#!/usr/bin/env python3
"""Filter SAM3 candidates per frame without changing the raw mask archive."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-score", type=float, default=0.90)
    parser.add_argument("--max-per-frame", type=int, default=3)
    args = parser.parse_args()

    source = read_json(args.input)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in source.get("items", []):
        grouped[str(item.get("image") or "")].append(item)

    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for image, items in sorted(grouped.items()):
        ranked = sorted(items, key=lambda item: float(item.get("score") or 0.0), reverse=True)
        accepted = [item for item in ranked if float(item.get("score") or 0.0) >= args.min_score][
            : args.max_per_frame
        ]
        accepted_ids = {id(item) for item in accepted}
        kept.extend(accepted)
        for rank, item in enumerate(ranked, start=1):
            if id(item) not in accepted_ids:
                dropped.append(
                    {
                        "image": image,
                        "label": item.get("label"),
                        "score": item.get("score"),
                        "bbox": item.get("bbox"),
                        "rank": rank,
                        "reason": "below_min_score_or_above_frame_cap",
                    }
                )

    frame_counts = Counter(str(item["image"]) for item in kept)
    output = {
        "scene": source.get("scene"),
        "image_root": source.get("image_root"),
        "mask_root": source.get("mask_root"),
        "items": kept,
        "missing_images": source.get("missing_images", []),
        "filter": {
            "method": "per_frame_score_then_rank",
            "source_index": str(args.input.resolve()),
            "min_score": args.min_score,
            "max_per_frame": args.max_per_frame,
            "source_item_count": sum(len(items) for items in grouped.values()),
            "kept_item_count": len(kept),
            "dropped_item_count": len(dropped),
            "frame_count": len(grouped),
            "kept_per_frame_distribution": dict(sorted(Counter(frame_counts.values()).items())),
            "dropped": dropped,
            "rationale": (
                "For bedroom_4 pillow, rank-1 scores are all >=0.906 while rank-4 scores are all <=0.871; "
                "0.90 separates the three physical pillow candidates from lower-ranked false positives."
            ),
        },
    }
    write_json(args.output, output)
    print(json.dumps(output["filter"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
