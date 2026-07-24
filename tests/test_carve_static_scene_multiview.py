from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

from scripts.carve_static_scene_multiview import ViewInput, carve_static_scene
from video2world.hashing import sha256_file

GAUSSIAN_DTYPE = [
    ("x", "f4"),
    ("y", "f4"),
    ("z", "f4"),
    ("f_dc_0", "f4"),
    ("f_dc_1", "f4"),
    ("f_dc_2", "f4"),
    ("opacity", "f4"),
    ("scale_0", "f4"),
    ("scale_1", "f4"),
    ("scale_2", "f4"),
    ("rot_0", "f4"),
    ("rot_1", "f4"),
    ("rot_2", "f4"),
    ("rot_3", "f4"),
]


def _sha(path: Path) -> str:
    return sha256_file(path)[0]


def _write_static(path: Path) -> None:
    data = np.zeros(4, dtype=GAUSSIAN_DTYPE)
    data["x"] = [0.01, 0.02, 1.01, 3.0]
    data["z"] = [5.01, 5.12, 5.01, 5.0]
    data["f_dc_0"] = [10, 11, 12, 13]
    data["opacity"] = 2.0
    data["scale_0"] = -4.0
    data["scale_1"] = -4.0
    data["scale_2"] = -4.0
    data["rot_0"] = 1.0
    PlyData([PlyElement.describe(data, "vertex")], text=False, byte_order="<").write(path)


def _write_anchors(path: Path) -> None:
    points = np.empty(2, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    points["x"] = [0.0, 1.0]
    points["y"] = 0.0
    points["z"] = 5.0
    PlyData([PlyElement.describe(points, "vertex")], text=False, byte_order="<").write(path)


def _write_cameras(path: Path) -> None:
    camera = {
        "rotation": np.eye(3).tolist(),
        "fx": 10.0,
        "fy": 10.0,
        "cx": 16.0,
        "cy": 16.0,
        "width": 32,
        "height": 32,
    }
    path.write_text(
        json.dumps(
            [
                {"img_name": "000001", "position": [0.0, 0.0, 0.0], **camera},
                {"img_name": "000002", "position": [0.1, 0.0, 0.0], **camera},
            ]
        ),
        encoding="utf-8",
    )


def test_carve_requires_multiview_modal_mask_and_depth_agreement(tmp_path: Path) -> None:
    static = tmp_path / "static.ply"
    anchors = tmp_path / "anchors.ply"
    cameras = tmp_path / "cameras.json"
    mask_a = tmp_path / "000001.png"
    mask_b = tmp_path / "000002.png"
    _write_static(static)
    _write_anchors(anchors)
    _write_cameras(cameras)
    Image.fromarray(np.full((32, 32), 255, dtype=np.uint8)).save(mask_a)
    second_mask = np.zeros((32, 32), dtype=np.uint8)
    second_mask[16, 16] = 255
    Image.fromarray(second_mask).save(mask_b)
    static_sha_before = _sha(static)

    output = tmp_path / "candidate.ply"
    receipt_path = tmp_path / "receipt.json"
    receipt = carve_static_scene(
        static_ply=static,
        static_ply_sha256=static_sha_before,
        anchor_ply=anchors,
        anchor_ply_sha256=_sha(anchors),
        camera_json=cameras,
        camera_json_sha256=_sha(cameras),
        views=[
            ViewInput("000001", mask_a, _sha(mask_a)),
            ViewInput("000002", mask_b, _sha(mask_b)),
        ],
        output_ply=output,
        receipt_path=receipt_path,
        anchor_distance=0.2,
        depth_tolerance=0.05,
        max_pixel_delta=1,
        min_view_votes=2,
        batch_size=2,
    )

    assert _sha(static) == static_sha_before
    assert receipt["promotion_allowed"] is False
    assert receipt["analysis"]["near_anchor_static_gaussian_count"] == 3
    assert receipt["analysis"]["selected_remove_count"] == 1
    assert receipt["analysis"]["strict_all_view_remove_count"] == 1
    assert receipt["output_static_scene"]["vertex_count"] == 3
    assert json.loads(receipt_path.read_text(encoding="utf-8"))["kind"] == (
        "video2world.multiview_depth_static_gaussian_carve"
    )

    vertices = PlyData.read(output)["vertex"].data
    assert len(vertices) == 3
    np.testing.assert_allclose(vertices["f_dc_0"], [11.0, 12.0, 13.0])


def test_carve_rejects_static_hash_mismatch(tmp_path: Path) -> None:
    static = tmp_path / "static.ply"
    _write_static(static)
    anchors = tmp_path / "anchors.ply"
    _write_anchors(anchors)
    cameras = tmp_path / "cameras.json"
    _write_cameras(cameras)
    mask = tmp_path / "mask.png"
    Image.fromarray(np.full((32, 32), 255, dtype=np.uint8)).save(mask)

    try:
        carve_static_scene(
            static_ply=static,
            static_ply_sha256="0" * 64,
            anchor_ply=anchors,
            anchor_ply_sha256=_sha(anchors),
            camera_json=cameras,
            camera_json_sha256=_sha(cameras),
            views=[
                ViewInput("000001", mask, _sha(mask)),
                ViewInput("000002", mask, _sha(mask)),
            ],
            output_ply=tmp_path / "candidate.ply",
            receipt_path=tmp_path / "receipt.json",
        )
    except RuntimeError as exc:
        assert "static PLY SHA-256 mismatch" in str(exc)
    else:  # pragma: no cover - protects the expected failure contract
        raise AssertionError("expected the static PLY hash to be verified")
