from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.build_da3_vggt_omega_visual_ab import build_report, depth_pair_images


def write_rgb(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 16), (value, value, value)).save(path)


def test_depth_pair_uses_shared_display_range() -> None:
    da3, vggt = depth_pair_images(
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float32),
        2.0,
    )
    np.testing.assert_array_equal(np.asarray(da3), np.asarray(vggt))


def test_build_report_hashes_visual_ab_and_detects_better_render(tmp_path: Path) -> None:
    da3_scene = tmp_path / "da3" / "bedroom_4"
    vggt_scene = tmp_path / "vggt" / "bedroom_4"
    da3_renders = tmp_path / "renders_da3"
    vggt_renders = tmp_path / "renders_vggt"
    for frame_id, source_value in (("000000", 100), ("000001", 150)):
        write_rgb(da3_scene / "color" / f"{frame_id}.jpg", source_value)
        write_rgb(vggt_scene / "color" / f"{frame_id}.jpg", source_value)
        write_rgb(da3_renders / f"{frame_id}.png", source_value + 20)
        write_rgb(vggt_renders / f"{frame_id}.png", source_value + 5)
        for scene, multiplier in ((da3_scene, 1.0), (vggt_scene, 0.5)):
            depth_root = scene / "depth_da3"
            depth_root.mkdir(parents=True, exist_ok=True)
            np.save(depth_root / f"{frame_id}.npy", np.full((16, 32), 2.0 * multiplier))

    numeric_receipt = tmp_path / "numeric.json"
    numeric_receipt.write_text(
        json.dumps(
            {
                "kind": "video2world.da3_vggt_omega_ab_receipt",
                "camera_alignment": {"scale": 2.0},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    report = build_report(
        Namespace(
            da3_scene=da3_scene,
            vggt_scene=vggt_scene,
            da3_render_root=da3_renders,
            vggt_render_root=vggt_renders,
            ab_receipt=numeric_receipt,
            output_dir=output,
            frame_ids=None,
            sample_count=2,
            thumbnail_width=96,
        )
    )

    assert report["promotion_allowed"] is False
    assert report["render_metrics"]["numeric_indication"] == (
        "vggt_omega_better_selected_render_fit"
    )
    assert report["render_metrics"]["vggt_minus_da3"]["psnr_db"] > 0
    assert report["manual_review"]["status"] == "pending"
    assert Path(report["outputs"]["contact_sheet"]["path"]).is_file()
    assert (output / "visual_ab_receipt.json").is_file()
