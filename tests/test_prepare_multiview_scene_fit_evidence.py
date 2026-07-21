from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

from scripts.fit_completed_object_to_scene import load_view_manifest
from scripts.prepare_multiview_scene_fit_evidence import (
    CONFIG_KIND,
    prepare_multiview_evidence,
)
from video2world.hashing import sha256_file


def _write_ply(path: Path, points: np.ndarray) -> str:
    vertices = np.empty(len(points), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    for index, name in enumerate(("x", "y", "z")):
        vertices[name] = points[:, index]
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)
    return sha256_file(path)[0]


def _project_center(camera: dict[str, object], center: np.ndarray) -> tuple[int, int]:
    position = np.asarray(camera["position"], dtype=np.float64)
    rotation = np.asarray(camera["rotation"], dtype=np.float64)
    point = (center - position) @ rotation
    return (
        round(float(camera["fx"]) * point[0] / point[2] + int(camera["width"]) / 2),
        round(float(camera["fy"]) * point[1] / point[2] + int(camera["height"]) / 2),
    )


def _write_mask(path: Path, center: tuple[int, int], *, half_size: int = 7) -> list[int]:
    image = np.zeros((120, 160), dtype=np.uint8)
    x, y = center
    image[y - half_size : y + half_size + 1, x - half_size : x + half_size + 1] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)
    return [x - half_size, y - half_size, x + half_size + 1, y + half_size + 1]


def test_prepare_multiview_evidence_associates_masks_and_declares_occluders(
    tmp_path: Path,
) -> None:
    cameras = []
    for frame, x in zip(
        ("000001", "000002", "000003", "000004"),
        (-1.2, -0.4, 0.5, 1.4),
        strict=True,
    ):
        cameras.append(
            {
                "id": int(frame),
                "img_name": frame,
                "width": 160,
                "height": 120,
                "position": [x, 0.0, 0.0],
                "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "fx": 100.0,
                "fy": 100.0,
            }
        )
    cameras_path = tmp_path / "cameras.json"
    cameras_path.write_text(json.dumps(cameras), encoding="utf-8")
    rng = np.random.default_rng(11)
    front_center = np.asarray([-0.45, 0.05, 8.0])
    rear_center = np.asarray([0.55, 0.0, 8.4])
    front = rng.normal(front_center, [0.08, 0.08, 0.03], size=(800, 3))
    rear = rng.normal(rear_center, [0.08, 0.08, 0.03], size=(800, 3))
    front_path = tmp_path / "front.ply"
    rear_path = tmp_path / "rear.ply"
    front_sha = _write_ply(front_path, front)
    rear_sha = _write_ply(rear_path, rear)
    mask_root = tmp_path / "masks"
    source_root = tmp_path / "source"
    source_root.mkdir()
    items = []
    for camera in cameras:
        frame = str(camera["img_name"])
        Image.new("RGB", (160, 120), (80, 90, 100)).save(source_root / f"{frame}.png")
        # Detector ordinals deliberately swap after the second frame.
        names = ("pillow_01.png", "pillow_02.png")
        if frame in {"000003", "000004"}:
            names = tuple(reversed(names))
        for name, center in zip(names, (front_center, rear_center), strict=True):
            bbox = _write_mask(mask_root / frame / name, _project_center(camera, center))
            items.append(
                {
                    "image": f"{frame}.png",
                    "label": "pillow",
                    "mask_path": f"/remote/{frame}/{name}",
                    "bbox": bbox,
                    "score": 0.95,
                }
            )
    mask_index_path = tmp_path / "mask-index.json"
    mask_index_path.write_text(json.dumps({"items": items}), encoding="utf-8")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": CONFIG_KIND,
                "cameras": {"path": str(cameras_path), "sha256": sha256_file(cameras_path)[0]},
                "mask_index": {
                    "path": str(mask_index_path),
                    "sha256": sha256_file(mask_index_path)[0],
                },
                "mask_root": str(mask_root),
                "source_rgb_dir": str(source_root),
                "selected_view_count": 3,
                "thresholds": {
                    "minimum_sam3_score": 0.8,
                    "minimum_anchor_hit_ratio": 0.5,
                    "minimum_anchor_hit_margin": 0.4,
                },
                "objects": [
                    {
                        "object_id": "front",
                        "category": "pillow",
                        "anchor": {"path": str(front_path), "sha256": front_sha},
                    },
                    {
                        "object_id": "rear",
                        "category": "pillow",
                        "anchor": {"path": str(rear_path), "sha256": rear_sha},
                        "occluder_object_ids": ["front"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = prepare_multiview_evidence(config_path=config_path, output_dir=tmp_path / "output")

    assert len(result["manifests"]) == 2
    audit = json.loads(Path(result["audit"]["path"]).read_text(encoding="utf-8"))
    assert audit["status"] == "source_multiview_modal_masks_ready_for_scene_fit"
    front_audit = next(item for item in audit["objects"] if item["object_id"] == "front")
    assert front_audit["eligible_frame_count"] == 4
    selected_names = {
        view["frame_id"]: view["association"]["selected_mask_file_name"]
        for view in front_audit["views"]
    }
    assert selected_names["000001"] == "pillow_01.png"
    assert selected_names["000004"] == "pillow_02.png"
    rear_manifest_path = next(
        Path(item["path"]) for item in result["manifests"] if item["object_id"] == "rear"
    )
    rear_manifest = json.loads(rear_manifest_path.read_text(encoding="utf-8"))
    assert len(rear_manifest["views"]) == 3
    assert all(len(view["occluder_masks"]) == 1 for view in rear_manifest["views"])
    loaded = load_view_manifest(
        rear_manifest_path,
        object_id="rear",
        cameras_path=cameras_path,
    )
    assert len(loaded) == 3
