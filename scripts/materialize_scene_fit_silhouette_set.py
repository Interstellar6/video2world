#!/usr/bin/env python3
"""Materialize analytic source-camera silhouettes for one accepted scene-fit GLB."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from qa_scene_camera_silhouette import load_camera, mask_bbox, project_mesh_silhouette
from render_source_camera_pbr_layer import (
    parse_frame_ids,
    read_json,
    scene_fit_raw_transform,
    sha256_file,
    validate_scene_fit,
    write_json,
)


def build_silhouettes(args: argparse.Namespace) -> dict:
    mesh_path = args.mesh.expanduser().resolve()
    report_path = args.scene_fit_report.expanduser().resolve()
    cameras_path = args.cameras.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    report = read_json(report_path)
    if not isinstance(report, dict):
        raise ValueError("scene-fit report must be a JSON object")
    validate_scene_fit(report, mesh_path, args.layer_id)
    frame_ids = parse_frame_ids(args.frame_id)
    loaded = trimesh.load(mesh_path, force="scene")
    mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded
    placement_mode, transform = scene_fit_raw_transform(report)
    records = []
    masks_dir = output_root / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    for frame_id in frame_ids:
        camera = load_camera(cameras_path, frame_id)
        mask = project_mesh_silhouette(
            mesh,
            runtime_pivot=(
                transform[:3, 3]
                if placement_mode == "baked_linear_plus_runtime_pivot"
                else None
            ),
            camera=camera,
            image_y_axis="down",
            scene_transform=(
                transform if placement_mode == "full_affine_runtime_transform" else None
            ),
        )
        if not np.any(mask):
            raise ValueError(f"analytic silhouette is empty for frame {frame_id}")
        path = masks_dir / f"{frame_id}.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(path)
        records.append(
            {
                "frame_id": frame_id,
                "mask": str(path),
                "mask_sha256": sha256_file(path),
                "pixels": int(mask.sum()),
                "bbox_xyxy": mask_bbox(mask),
            }
        )
    manifest = {
        "schema_version": 1,
        "kind": "video2world.scene_fit_analytic_silhouette_set",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "technical_passed",
        "layer_id": args.layer_id,
        "camera_convention": "camera_to_world_right_down_forward; image_y_down",
        "placement_mode": placement_mode,
        "sources": {
            "mesh": {"path": str(mesh_path), "sha256": sha256_file(mesh_path)},
            "scene_fit_report": {
                "path": str(report_path),
                "sha256": sha256_file(report_path),
            },
            "cameras": {"path": str(cameras_path), "sha256": sha256_file(cameras_path)},
        },
        "frame_records": records,
    }
    manifest_path = output_root / "silhouette_manifest.json"
    write_json(manifest_path, manifest)
    write_json(
        output_root / "silhouette_receipt.json",
        {
            "schema_version": 1,
            "kind": "video2world.scene_fit_analytic_silhouette_set_receipt",
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "frame_count": len(records),
            "mask_artifacts_are_hashed_in_manifest": True,
        },
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--scene-fit-report", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--layer-id", required=True)
    parser.add_argument("--frame-id", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_silhouettes(args)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "layer_id": manifest["layer_id"],
                "frame_count": len(manifest["frame_records"]),
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
