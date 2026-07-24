from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from scripts.qa_scene_camera_silhouette import (
    load_camera,
    project_mesh_silhouette,
    review_scene_silhouette,
)


def _write_camera(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "img_name": "000001",
                    "width": 200,
                    "height": 160,
                    "position": [0, 0, 0],
                    "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "fx": 120,
                    "fy": 120,
                }
            ]
        ),
        encoding="utf-8",
    )


def _write_report(path: Path, pivot: list[float]) -> None:
    path.write_text(
        json.dumps({"baked_relative_transform": {"runtime_pivot": pivot}}),
        encoding="utf-8",
    )


def _write_affine_report(path: Path, matrix: np.ndarray) -> None:
    path.write_text(
        json.dumps({"runtime_transform": {"matrix_row_major": matrix.tolist()}}),
        encoding="utf-8",
    )


def test_source_camera_silhouette_accepts_alignment_and_rejects_drift(
    tmp_path: Path,
) -> None:
    mesh = trimesh.creation.box(extents=(2.0, 1.5, 0.5))
    mesh_path = tmp_path / "box.glb"
    mesh.export(mesh_path)
    cameras_path = tmp_path / "cameras.json"
    _write_camera(cameras_path)
    camera = load_camera(cameras_path, "000001")
    with np.errstate(all="raise"):
        observed = project_mesh_silhouette(
            mesh,
            runtime_pivot=np.asarray([0.0, 0.0, 5.0]),
            camera=camera,
        )
    mask_path = tmp_path / "mask.png"
    Image.fromarray(observed.astype(np.uint8) * 255).save(mask_path)
    source_path = tmp_path / "source.png"
    Image.new("RGB", (200, 160), (100, 110, 120)).save(source_path)

    aligned_report = tmp_path / "aligned.json"
    _write_report(aligned_report, [0.0, 0.0, 5.0])
    with np.errstate(all="raise"):
        aligned = review_scene_silhouette(
            mesh_path=mesh_path,
            scene_fit_report_path=aligned_report,
            cameras_path=cameras_path,
            frame_id="000001",
            observed_mask_path=mask_path,
            source_frame_path=source_path,
            output_dir=tmp_path / "aligned",
            minimum_mask_iou=0.99,
            minimum_bbox_iou=0.99,
            maximum_center_error_px=0.1,
        )
    assert aligned["promotion_allowed"] is True
    assert aligned["metrics"]["mask_iou"] == 1.0

    drifted_report = tmp_path / "drifted.json"
    _write_report(drifted_report, [1.0, 0.0, 5.0])
    drifted = review_scene_silhouette(
        mesh_path=mesh_path,
        scene_fit_report_path=drifted_report,
        cameras_path=cameras_path,
        frame_id="000001",
        observed_mask_path=mask_path,
        output_dir=tmp_path / "drifted",
        minimum_mask_iou=0.8,
        minimum_bbox_iou=0.8,
        maximum_center_error_px=5.0,
    )
    assert drifted["promotion_allowed"] is False
    assert drifted["gates"]["center_error_px"] is False


def test_source_camera_silhouette_supports_full_affine_transform(tmp_path: Path) -> None:
    mesh = trimesh.creation.box(extents=(2.0, 1.5, 0.5))
    mesh_path = tmp_path / "box.glb"
    mesh.export(mesh_path)
    cameras_path = tmp_path / "cameras.json"
    _write_camera(cameras_path)
    camera = load_camera(cameras_path, "000001")
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(
        [[0.0, -1.4, 0.0], [1.4, 0.0, 0.0], [0.0, 0.0, 1.4]]
    )
    transform[:3, 3] = [0.25, -0.1, 6.0]
    observed = project_mesh_silhouette(
        mesh,
        runtime_pivot=None,
        camera=camera,
        scene_transform=transform,
    )
    mask_path = tmp_path / "mask.png"
    Image.fromarray(observed.astype(np.uint8) * 255).save(mask_path)
    report_path = tmp_path / "affine.json"
    _write_affine_report(report_path, transform)

    result = review_scene_silhouette(
        mesh_path=mesh_path,
        scene_fit_report_path=report_path,
        cameras_path=cameras_path,
        frame_id="000001",
        observed_mask_path=mask_path,
        output_dir=tmp_path / "affine-review",
        minimum_mask_iou=0.99,
        minimum_bbox_iou=0.99,
        maximum_center_error_px=0.1,
    )

    assert result["promotion_allowed"] is True
    assert result["scene_transform_mode"] == "full_affine_runtime_transform"
    assert result["metrics"]["mask_iou"] == 1.0


def test_source_camera_silhouette_excludes_verified_occluder(tmp_path: Path) -> None:
    mesh = trimesh.creation.box(extents=(2.0, 1.5, 0.5))
    mesh_path = tmp_path / "box.glb"
    mesh.export(mesh_path)
    cameras_path = tmp_path / "cameras.json"
    _write_camera(cameras_path)
    camera = load_camera(cameras_path, "000001")
    rendered = project_mesh_silhouette(
        mesh,
        runtime_pivot=np.asarray([0.0, 0.0, 5.0]),
        camera=camera,
    )
    occluder = np.zeros_like(rendered)
    occluder[65:95, 90:110] = True
    observed = rendered & ~occluder
    mask_path = tmp_path / "mask.png"
    occluder_path = tmp_path / "occluder.png"
    Image.fromarray(observed.astype(np.uint8) * 255).save(mask_path)
    Image.fromarray(occluder.astype(np.uint8) * 255).save(occluder_path)
    report_path = tmp_path / "report.json"
    _write_report(report_path, [0.0, 0.0, 5.0])

    result = review_scene_silhouette(
        mesh_path=mesh_path,
        scene_fit_report_path=report_path,
        cameras_path=cameras_path,
        frame_id="000001",
        observed_mask_path=mask_path,
        output_dir=tmp_path / "review",
        minimum_mask_iou=0.99,
        minimum_bbox_iou=0.99,
        maximum_center_error_px=0.1,
        occluder_mask_paths=[occluder_path],
    )

    assert result["promotion_allowed"] is True
    assert result["metrics"]["mask_iou"] == 1.0
    assert result["metrics"]["rendered_occluded_pixels"] > 0
    assert result["sources"]["occluder_masks"][0]["pixels"] > 0


def test_projection_respects_explicit_principal_point(tmp_path: Path) -> None:
    cameras_path = tmp_path / "cameras.json"
    cameras_path.write_text(
        json.dumps(
            [
                {
                    "img_name": "000001",
                    "width": 200,
                    "height": 160,
                    "position": [0, 0, 0],
                    "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "fx": 100,
                    "fy": 100,
                    "cx": 64,
                    "cy": 101,
                }
            ]
        ),
        encoding="utf-8",
    )
    camera = load_camera(cameras_path, "000001")
    rendered = project_mesh_silhouette(
        trimesh.creation.box(extents=(1.0, 1.0, 0.5)),
        runtime_pivot=np.asarray([0.0, 0.0, 5.0]),
        camera=camera,
    )
    y, x = np.nonzero(rendered)

    assert camera["cx"] == 64
    assert camera["cy"] == 101
    assert abs(float(x.mean()) - 64) < 1.0
    assert abs(float(y.mean()) - 101) < 1.0
