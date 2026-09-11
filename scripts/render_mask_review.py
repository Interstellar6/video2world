#!/usr/bin/env python3
"""Render measured parent/component masks over their original calibrated images."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from world_modeling.provider_io import local_path, read_json


def main():
    from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps

    app = argparse.ArgumentParser(description=__doc__)
    app.add_argument("--task-dir", type=Path, required=True)
    app.add_argument("--object-id", required=True)
    app.add_argument("--frames", nargs="+", required=True)
    args = app.parse_args()
    index = read_json(args.task_dir / "artifacts.json")["artifacts"]
    masks = read_json(local_path(args.task_dir, index["component_masks"]["path"]))["masks"]
    tiles = []
    for frame_id in args.frames:
        records = [item for item in masks if item["object_id"] == args.object_id and item["frame_id"] == frame_id]
        if not records:
            continue
        image = Image.open(local_path(args.task_dir, records[0]["image_path"])).convert("RGB")
        parts = []
        for item in records:
            mask = Image.open(local_path(args.task_dir, item["mask_path"])).convert("L")
            parent = item["component_id"] == "__object__"
            color = (235, 45, 70) if parent else (30, 190, 140)
            edge = ImageChops.subtract(mask.filter(ImageFilter.MaxFilter(5)), mask.filter(ImageFilter.MinFilter(5)))
            image.paste(color, mask=edge)
            if not parent:
                parts.append(item["component_id"])
        tile = Image.new("RGB", (640, 410), "white")
        tile.paste(ImageOps.contain(image, (640, 360)), (0, 0))
        ImageDraw.Draw(tile).text((8, 365), f"{args.object_id} / {frame_id} / red: parent; green: parts", fill="black")
        ImageDraw.Draw(tile).text((8, 385), ", ".join(parts), fill="black")
        tiles.append(tile)
    if not tiles:
        raise ValueError("no requested masks found")
    sheet = Image.new("RGB", (640 * min(3, len(tiles)), 410 * ((len(tiles) + 2) // 3)), "white")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, (640 * (i % 3), 410 * (i // 3)))
    destination = local_path(args.task_dir, "qa", exists=False)
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / f"{args.object_id}_mask_review.png"
    sheet.save(output)
    print(output)


if __name__ == "__main__":
    main()
