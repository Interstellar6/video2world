#!/usr/bin/env python3
"""Erode class probability masks while retaining their score intensities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2


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
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--kernel-size", type=int, default=5)
    args = parser.parse_args()

    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    if args.kernel_size < 1 or args.kernel_size % 2 == 0:
        raise ValueError("--kernel-size must be a positive odd integer")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.kernel_size, args.kernel_size))
    records = []
    for source in sorted(args.input_root.rglob("*.png")):
        relative = source.relative_to(args.input_root)
        target = args.output_root / relative
        mask = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read {source}")
        eroded = cv2.erode(mask, kernel, iterations=1)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(target), eroded):
            raise RuntimeError(f"Failed to write {target}")
        before = int((mask > 0).sum())
        after = int((eroded > 0).sum())
        records.append(
            {
                "source": str(source),
                "output": str(target),
                "pixels_before": before,
                "pixels_after": after,
                "retained_ratio": after / before if before else 0.0,
            }
        )

    source_tracking = args.input_root / "tracking_manifest.json"
    tracking = read_json(source_tracking) if source_tracking.exists() else {}
    report = {
        "schema_version": 1,
        "method": "elliptical_grayscale_erosion",
        "kernel_size": args.kernel_size,
        "source_root": str(args.input_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "mask_count": len(records),
        "mean_retained_ratio": sum(item["retained_ratio"] for item in records) / max(1, len(records)),
        "source_tracking_manifest": tracking,
        "masks": records,
    }
    write_json(args.output_root / "erosion_manifest.json", report)
    print(json.dumps({key: report[key] for key in ("method", "kernel_size", "mask_count", "mean_retained_ratio")}, indent=2))


if __name__ == "__main__":
    main()
