#!/usr/bin/env python3
"""Summarize and render evidence for the isolated bedroom_4 pillow delta."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def overlay_preview(image_path: Path, mask_path: Path, output_path: Path, label: str, score: float) -> None:
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if mask.size != image.size:
        mask = mask.resize(image.size, resample=Image.Resampling.NEAREST)
    red = Image.new("RGB", image.size, (220, 40, 40))
    alpha = mask.point(lambda value: 105 if value else 0)
    image = Image.composite(red, image, alpha)
    draw = ImageDraw.Draw(image)
    draw.rectangle((12, 12, 340, 52), fill=(0, 0, 0))
    draw.text((22, 22), f"{label} | SAM3 score {score:.3f}", fill=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta-root", type=Path, required=True)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()

    delta = args.delta_root.resolve()
    base = args.base_root.resolve()
    mask_index_path = delta / "sam3_masks" / "bedroom_4" / "mask_index.json"
    tracking_path = delta / "video2mesh" / "masks" / "2d" / "tracking_manifest.json"
    class_masks_path = delta / "video2mesh" / "masks" / "3d" / "object_masks.json"
    instance_masks_path = delta / "video2mesh" / "masks" / "3d_instances" / "object_masks.json"
    holi_bbox_path = delta / "output_scannetppv2_new" / "bedroom_4.json"
    cloud_index_path = (
        delta
        / "video2mesh"
        / "simulator_assets"
        / "object_masks_3d"
        / "object_mask_clouds.json"
    )

    mask_index = read_json(mask_index_path)
    tracking = read_json(tracking_path)
    class_masks = read_json(class_masks_path)
    instance_masks = read_json(instance_masks_path)
    holi_boxes = read_json(holi_bbox_path)
    cloud_index = read_json(cloud_index_path)

    items = [item for item in mask_index.get("items", []) if item.get("label") == "pillow"]
    by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_frame[Path(item["image"]).stem].append(item)
    scores = np.asarray([float(item.get("score") or 0.0) for item in items], dtype=np.float64)

    preview_dir = delta / "evidence" / "previews"
    ranked = sorted(items, key=lambda item: float(item.get("score") or 0.0), reverse=True)
    chosen: list[dict[str, Any]] = []
    used_frames: set[str] = set()
    for item in ranked:
        frame = Path(item["image"]).stem
        if frame in used_frames:
            continue
        chosen.append(item)
        used_frames.add(frame)
        if len(chosen) == 6:
            break
    preview_records = []
    image_root = Path(mask_index["image_root"])
    for index, item in enumerate(chosen, start=1):
        image_path = image_root / item["image"]
        mask_path = Path(item["mask_path"])
        output_path = preview_dir / f"pillow_{index:02d}_{Path(item['image']).stem}.jpg"
        overlay_preview(image_path, mask_path, output_path, "pillow", float(item.get("score") or 0.0))
        preview_records.append(
            {
                "frame": Path(item["image"]).stem,
                "score": float(item.get("score") or 0.0),
                "bbox": item.get("bbox"),
                "mask": str(mask_path),
                "preview": str(output_path),
            }
        )

    instances = []
    for object_id, record in sorted(instance_masks.get("objects", {}).items()):
        mask_3d = record["mask_3d"]
        indices_path = Path(mask_3d["point_indices_npy"])
        indices = np.load(indices_path, mmap_mode="r")
        cloud_record = cloud_index.get("objects", {}).get(object_id, {})
        cloud_path_raw = cloud_record.get("output") or cloud_record.get("point_cloud") or cloud_record.get("path")
        cloud_path = Path(cloud_path_raw) if cloud_path_raw else None
        instances.append(
            {
                "object_id": object_id,
                "category": record.get("category"),
                "point_count": int(indices.shape[0]),
                "bbox_3d": record.get("bbox_3d"),
                "obb_3d": record.get("obb_3d"),
                "probability_summary": mask_3d.get("probability_summary"),
                "instance_split": record.get("instance_split"),
                "point_indices": file_record(indices_path),
                "object_cloud": file_record(cloud_path) if cloud_path and cloud_path.is_file() else None,
            }
        )

    source_class = class_masks.get("objects", {}).get("sam3_class_pillow", {})
    summary = {
        "schema_version": 1,
        "status": "completed_real_sam3_pillow_delta",
        "scene_id": "bedroom_4",
        "coordinate_frame": "source Video2Mesh/DA3 world frame",
        "units": "scene_scale_not_metric",
        "provenance": {
            "base_run": str(base),
            "delta_run": str(delta),
            "base_is_read_only": True,
            "sam3_checkpoint": file_record(Path("/data/zyx/workspace/third_party/holi-spatial/checkpoints/sam3/sam3.pt")),
            "da3_point_cloud": file_record(base / "scannetppv2" / "data" / "bedroom_4" / "pointcloud_da3.ply"),
            "camera_validation": read_json(delta / "logs" / "camera_coordinate_validation.json"),
        },
        "sam3": {
            "prompt": "pillow",
            "frame_count": len(by_frame),
            "source_instance_mask_count": len(items),
            "instances_per_frame": dict(sorted(Counter(len(values) for values in by_frame.values()).items())),
            "missing_images": mask_index.get("missing_images", []),
            "score_min": float(scores.min()) if scores.size else None,
            "score_mean": float(scores.mean()) if scores.size else None,
            "score_max": float(scores.max()) if scores.size else None,
            "tracking_counts": tracking.get("counts"),
            "previews": preview_records,
        },
        "lifting": {
            "fusion": class_masks.get("fusion"),
            "class_point_count": source_class.get("point_count"),
            "class_bbox_3d": source_class.get("bbox_3d"),
            "class_probability_summary": (source_class.get("mask_3d") or {}).get("probability_summary"),
        },
        "postprocess": {
            "instance_count": len(instances),
            "instances": instances,
            "class_reports": instance_masks.get("class_reports"),
            "holi_bbox_count": len(holi_boxes),
            "limitations": [
                "SAM3 produces class-level masks without stable cross-frame instance IDs.",
                "Instances are split by Video2Mesh voxel DBSCAN and are not official Holi-Spatial instance merging.",
                "Coordinates remain scene-scale; distance values must not be labeled meters until calibrated.",
                "Detailed appearance captioning and QA generation were not run in this delta.",
            ],
        },
        "files": {
            "mask_index": file_record(mask_index_path),
            "tracking_manifest": file_record(tracking_path),
            "class_object_masks": file_record(class_masks_path),
            "instance_object_masks": file_record(instance_masks_path),
            "holi_bbox": file_record(holi_bbox_path),
            "object_cloud_index": file_record(cloud_index_path),
        },
    }
    summary_path = delta / "evidence" / "pillow_delta_summary.json"
    write_json(summary_path, summary)
    delta_manifest = {
        "schema_version": 1,
        "run_id": args.run_id or delta.name,
        "scene_id": "bedroom_4",
        "scope": "single-class pillow discovery and 2D-to-3D delta",
        "isolation": {
            "base_run": str(base),
            "base_access": "read_only",
            "delta_run": str(delta),
            "fresh_artifacts_overwritten": False,
        },
        "stage_status": {
            "data_package": "Passed",
            "DA3": "ReusedReadOnlyParent",
            "GroundingDINO": "ProvenanceOnly19OldBoxes",
            "SAM3": "PassedRealCheckpoint",
            "Video2Mesh_lifting": "PassedAdapter",
            "instance_bbox_postprocess": "PassedProxyVoxelDBSCAN",
            "object_cloud_export": "Passed",
            "PGSR": "NotRunReusedReadOnlyParent",
            "caption": "NotRun",
            "QA": "NotRun",
            "TRELLIS": "NotRunByDesign",
        },
        "counts": {
            "source_frames": len(by_frame),
            "sam3_instance_masks": len(items),
            "sam3_merged_frame_masks": tracking.get("counts", {}).get("merged_class_frame_masks"),
            "class_points": source_class.get("point_count"),
            "pillow_instances": len(instances),
            "holi_bbox_records": len(holi_boxes),
        },
        "commands": str(delta / "commands" / "run_pillow_delta.sh"),
        "log": str(delta / "logs" / "pillow_delta.log"),
        "summary": str(summary_path),
    }
    write_json(delta / "evidence" / "pillow_delta_manifest.json", delta_manifest)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
