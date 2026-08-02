from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from scripts.run_vggt_omega_holi_prior import (
    VGGTOmegaPriorError,
    common_intrinsics,
    resize_intrinsics,
    unproject_frame,
    write_binary_point_cloud,
)


def test_resize_and_collapse_intrinsics() -> None:
    intrinsics = np.array(
        [
            [[300.0, 0.0, 160.0], [0.0, 280.0, 90.0], [0.0, 0.0, 1.0]],
            [[303.0, 0.0, 160.0], [0.0, 282.0, 90.0], [0.0, 0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    resized = resize_intrinsics(
        intrinsics, source_size=(180, 320), target_size=(720, 1280)
    )
    assert resized[0, 0, 0] == pytest.approx(1200.0)
    assert resized[0, 1, 1] == pytest.approx(1120.0)
    common, spread = common_intrinsics(resized, max_relative_spread=0.05)
    assert common[0, 2] == pytest.approx(640.0)
    assert common[1, 2] == pytest.approx(360.0)
    assert 0 < spread < 0.05


def test_common_intrinsics_rejects_camera_drift() -> None:
    intrinsics = np.array(
        [
            [[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]],
            [[200.0, 0.0, 2.0], [0.0, 200.0, 2.0], [0.0, 0.0, 1.0]],
        ]
    )
    with pytest.raises(VGGTOmegaPriorError, match="spread"):
        common_intrinsics(intrinsics, max_relative_spread=0.1)


def test_unproject_frame_uses_world_to_camera() -> None:
    depth = np.ones((2, 2), dtype=np.float32)
    intrinsic = np.eye(3, dtype=np.float32)
    world_to_camera = np.eye(4, dtype=np.float32)
    world_to_camera[0, 3] = 1.0
    points = unproject_frame(depth, intrinsic, world_to_camera, np.array([0, 3]))
    np.testing.assert_allclose(points[0], [-1.0, 0.0, 1.0])
    np.testing.assert_allclose(points[1], [0.0, 1.0, 1.0])


def test_write_binary_point_cloud_is_readable(tmp_path: Path) -> None:
    path = tmp_path / "points.ply"
    points = np.array([[0.0, 0.0, 1.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    write_binary_point_cloud(path, points, colors)
    loaded = trimesh.load(path, process=False)
    assert isinstance(loaded, trimesh.points.PointCloud)
    assert len(loaded.vertices) == 2
