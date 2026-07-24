from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.prepare_front_half_scene_fit_evidence import (
    camera_records_from_camera_info,
    prepare_front_half_scene_fit_evidence,
    project_points,
    seeded_primary_component,
)


def _write_ascii_ply(path: Path, points: np.ndarray) -> None:
    path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                f"element vertex {len(points)}",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                *[f"{x} {y} {z}" for x, y, z in points],
                "",
            ]
        ),
        encoding="utf-8",
    )


def _camera_info(frame_ids: list[str]) -> dict:
    positions = {"000001": -0.3, "000002": 0.0, "000003": 0.3}
    return {
        "extrinsic_type": "world_to_camera",
        "intrinsic": {"fx": 40, "fy": 40, "cx": 40, "cy": 30, "w": 80, "h": 60},
        "extrinsic": {
            frame_id: [
                [1, 0, 0, -positions[frame_id]],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
            for frame_id in frame_ids
        },
    }


def test_camera_conversion_matches_world_to_camera_projection() -> None:
    records, cameras = camera_records_from_camera_info(_camera_info(["000002"]))
    x, y, visible = project_points(np.asarray([[1.0, 0.5, 5.0]]), cameras["000002"])

    assert records[0]["position"] == [0.0, 0.0, 0.0]
    assert visible.tolist() == [True]
    np.testing.assert_allclose(x, [48.0])
    np.testing.assert_allclose(y, [34.0])


def test_prepares_source_anchored_multiview_evidence(tmp_path: Path) -> None:
    frame_ids = ["000001", "000002", "000003"]
    frames_dir = tmp_path / "frames"
    masks_dir = tmp_path / "masks"
    frames_dir.mkdir()
    masks_dir.mkdir()
    for frame_id in frame_ids:
        Image.new("RGB", (80, 60), (90, 100, 110)).save(frames_dir / f"{frame_id}.png")
        mask_dir = masks_dir / "object_bed"
        mask_dir.mkdir(exist_ok=True)
        mask = np.zeros((60, 80), dtype=np.uint8)
        mask[15:45, 20:60] = 255
        Image.fromarray(mask).save(mask_dir / f"{frame_id}.png")
    generator = np.random.default_rng(4)
    points = generator.uniform([-1.0, -0.5, 4.5], [1.0, 0.5, 5.5], size=(400, 3))
    cloud_path = tmp_path / "bed.ply"
    _write_ascii_ply(cloud_path, points)
    camera_path = tmp_path / "camera_info.json"
    camera_path.write_text(json.dumps(_camera_info(frame_ids)), encoding="utf-8")
    manifest_path = tmp_path / "front_half_input_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "camera_info": str(camera_path),
                "frames_dir": str(frames_dir),
                "prepared": [
                    {
                        "object_id": "object_bed",
                        "category": "bed",
                        "source_point_cloud": str(cloud_path),
                        "mask_dir": str(masks_dir / "object_bed"),
                        "frame_id": "000002",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = prepare_front_half_scene_fit_evidence(
        input_manifest_path=manifest_path,
        output_dir=tmp_path / "evidence",
        view_count=3,
        minimum_projected_points=5,
        minimum_mask_support_ratio=0.5,
    )

    object_record = result["objects"][0]
    assert object_record["status"] == "multiview_scene_fit_evidence_ready"
    assert object_record["frame_selection"]["selected_source_frame"] == "000002"
    view_manifest = json.loads(Path(object_record["view_manifest"]["path"]).read_text())
    assert [view["frame_id"] for view in view_manifest["views"]][0] == "000002"
    assert len(view_manifest["views"]) == 3
    assert (tmp_path / "evidence" / "cameras.json").is_file()
    assert (tmp_path / "evidence" / "placement_anchors" / "object_bed.ply").is_file()


def test_seeded_primary_component_drops_neighboring_instance_union() -> None:
    _, cameras = camera_records_from_camera_info(_camera_info(["000002"]))
    camera = cameras["000002"]
    mask = np.zeros((60, 80), dtype=bool)
    mask[18:42, 12:30] = True
    mask[18:42, 50:68] = True
    points = np.column_stack(
        (
            np.full(80, -2.2),
            np.linspace(-1.0, 1.0, 80),
            np.full(80, 5.0),
        )
    )

    selected, report = seeded_primary_component(mask, points=points, camera=camera)

    assert report["mode"] == "projected_anchor_primary_component"
    assert report["component_count"] == 2
    assert report["kept_pixels"] == int(mask[:, 12:30].sum())
    assert selected[:, 12:30].any()
    assert not selected[:, 50:68].any()
