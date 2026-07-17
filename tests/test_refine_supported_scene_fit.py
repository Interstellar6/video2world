from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from plyfile import PlyData, PlyElement

from scripts.qa_scene_camera_silhouette import load_camera, project_mesh_silhouette
from scripts.refine_supported_scene_fit import refine_supported_scene_fit


def _write_ply(path: Path, points: np.ndarray) -> None:
    data = np.empty(len(points), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    for index, name in enumerate(("x", "y", "z")):
        data[name] = points[:, index]
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(path)


def test_support_constrained_scene_fit_uses_one_transform_for_two_views(tmp_path: Path) -> None:
    scene = trimesh.Scene(base_frame="world")
    scene.graph.update(frame_from="world", frame_to="bed", matrix=np.eye(4))
    mesh = trimesh.creation.box(extents=(2.0, 1.0, 3.0))
    mesh.apply_translation([0.0, 0.5, 0.0])
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2)),
        material=trimesh.visual.material.PBRMaterial(
            name="bed",
            baseColorFactor=[220, 218, 210, 255],
            metallicFactor=0.0,
            roughnessFactor=0.9,
        ),
    )
    scene.add_geometry(
        mesh,
        node_name="bed__body",
        geom_name="bed.body",
        parent_node_name="bed",
    )
    mesh_path = tmp_path / "bed.glb"
    scene.export(mesh_path)

    true_matrix = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 6.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    rng = np.random.default_rng(7)
    canonical = rng.uniform([-1.0, 0.0, -1.5], [1.0, 1.0, 1.5], size=(4000, 3))
    object_points = np.einsum(
        "ni,ji->nj", canonical, true_matrix[:3, :3], optimize=False
    ) + true_matrix[:3, 3]
    floor_x, floor_z = np.meshgrid(np.linspace(-5, 5, 30), np.linspace(1, 11, 30))
    support_points = np.column_stack(
        (floor_x.reshape(-1), np.full(floor_x.size, 2.0), floor_z.reshape(-1))
    )
    object_anchor = tmp_path / "object.ply"
    support_anchor = tmp_path / "floor.ply"
    _write_ply(object_anchor, object_points)
    _write_ply(support_anchor, support_points)

    cameras_path = tmp_path / "cameras.json"
    cameras_path.write_text(
        json.dumps(
            [
                {
                    "img_name": "000001",
                    "width": 160,
                    "height": 120,
                    "position": [0.0, 0.0, 0.0],
                    "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "fx": 100,
                    "fy": 100,
                },
                {
                    "img_name": "000002",
                    "width": 160,
                    "height": 120,
                    "position": [0.8, 0.0, 0.0],
                    "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "fx": 100,
                    "fy": 100,
                },
            ]
        ),
        encoding="utf-8",
    )
    views = []
    for frame_id in ("000001", "000002"):
        camera = load_camera(cameras_path, frame_id)
        silhouette = project_mesh_silhouette(
            mesh,
            runtime_pivot=None,
            camera=camera,
            scene_transform=true_matrix,
        )
        mask_path = tmp_path / f"{frame_id}.png"
        Image.fromarray(silhouette.astype(np.uint8) * 255).save(mask_path)
        views.append({"frame_id": frame_id, "observed_mask": str(mask_path)})
    manifest_path = tmp_path / "views.json"
    manifest_path.write_text(
        json.dumps({"object_id": "bed", "views": views}), encoding="utf-8"
    )

    report = refine_supported_scene_fit(
        object_id="bed",
        mesh_source=mesh_path,
        object_anchor_ply=object_anchor,
        support_anchor_ply=support_anchor,
        cameras_path=cameras_path,
        view_manifest_path=manifest_path,
        output_dir=tmp_path / "out",
        cycles=2,
        minimum_mask_iou=0.55,
        minimum_bbox_iou=0.7,
        maximum_center_error_px=20.0,
    )

    assert report["all_acceptance_gates_passed"] is True
    assert len(report["views"]) == 2
    assert all(all(item.values()) for item in report["view_acceptance_gates"])
    assert report["runtime_transform"]["scale_anisotropy_ratio"] <= 2.0
    assert report["support_alignment"]["object_up_axis"][1] < -0.9
    assert Path(report["sources"]["normalized_view_manifest"]["path"]).is_file()
    assert report["support_contact"]["passed"] is True
