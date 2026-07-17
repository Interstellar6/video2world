from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from scripts.project_object_anchor_supplement_masks import (
    build_parser,
    project,
    sha256_file,
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_ascii_ply(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    rows = "\n".join(f"{x:.8f} {y:.8f} {z:.8f}" for x, y, z in points)
    path.write_text(header + rows + "\n", encoding="ascii")


def point_for_pixel(
    u: int, v: int, z: float, *, fx: float, fy: float, cx: float, cy: float
) -> tuple[float, float, float]:
    return ((u - cx) * z / fx, (v - cy) * z / fy, z)


def make_fixture(root: Path, *, occluded_center: bool = False) -> dict[str, Path]:
    width, height = 16, 12
    fx = fy = 10.0
    cx, cy = 8.0, 6.0
    ring_pixels = [
        (u, v)
        for v in range(5, 8)
        for u in range(7, 10)
        if (u, v) != (8, 6)
    ]
    points = [
        point_for_pixel(u, v, 5.0, fx=fx, fy=fy, cx=cx, cy=cy)
        for u, v in ring_pixels
    ]
    points.append(point_for_pixel(12, 6, 7.0, fx=fx, fy=fy, cx=cx, cy=cy))
    anchor = root / "anchor.ply"
    write_ascii_ply(anchor, np.asarray(points, dtype=np.float64))
    camera = root / "camera_info.json"
    write_json(
        camera,
        {
            "extrinsic_type": "world_to_camera",
            "intrinsic": {
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "w": width,
                "h": height,
            },
            "extrinsic": {"000000": np.eye(4).tolist()},
        },
    )
    depth_dir = root / "depth"
    depth_dir.mkdir(parents=True)
    depth = np.full((height, width), 5.0, dtype=np.float32)
    if occluded_center:
        depth[6, 8] = 3.0
    np.save(depth_dir / "000000.npy", depth)
    frames = root / "frames"
    frames.mkdir(parents=True)
    rgb = np.full((height, width, 3), [160, 150, 140], dtype=np.uint8)
    Image.fromarray(rgb).save(frames / "000000.png")
    return {"anchor": anchor, "camera": camera, "depth": depth_dir, "frames": frames}


def make_args(
    paths: dict[str, Path],
    output: Path,
    *,
    splat_radius: int = 0,
    closing_radius: int = 0,
):
    return build_parser().parse_args(
        [
            "--object-id",
            "sam3_bed_01",
            "--round-index",
            "4",
            "--anchor",
            str(paths["anchor"]),
            "--camera-info",
            str(paths["camera"]),
            "--depth-dir",
            str(paths["depth"]),
            "--frames-dir",
            str(paths["frames"]),
            "--frame-ids",
            "000000",
            "--splat-radius",
            str(splat_radius),
            "--closing-radius",
            str(closing_radius),
            "--absolute-depth-tolerance",
            "0.05",
            "--relative-depth-tolerance",
            "0",
            "--minimum-visible-pixels",
            "1",
            "--contact-sheet-frame-ids",
            "000000",
            "--output",
            str(output),
        ]
    )


def test_projection_is_depth_visible_and_writes_compatible_hash_receipt(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    output = tmp_path / "output"

    report = project(make_args(paths, output))

    record = report["frame_records"][0]
    mask = np.asarray(Image.open(output / record["union_mask"]).convert("L")) > 0
    assert record["coverage"]["visible_raw_pixels"] == 8
    assert record["coverage"]["occluded_behind_pixels"] == 1
    assert record["coverage"]["final_mask_pixels"] == 8
    assert not mask[6, 8]
    assert not mask[6, 12]
    assert record["round_index"] == 4
    assert record["removed_object_ids"] == ["sam3_bed_01"]
    assert record["union_mask_sha256"] == sha256_file(output / record["union_mask"])
    assert report["round_index"] == 4
    assert report["removed_object_ids"] == ["sam3_bed_01"]
    assert report["gates"]["sam_mask_dilation_used"] is False
    assert report["gates"]["all_visible_raw_masks_are_subsets_of_final_masks"] is True
    assert (
        report["gates"][
            "all_final_masks_are_subsets_of_depth_consistent_anchor_envelopes"
        ]
        is True
    )
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["manifest_sha256"] == sha256_file(
        output / "object_anchor_supplement_manifest.json"
    )
    assert receipt["mask_set_sha256"] == report["mask_set_sha256"]
    assert receipt["gates"]["all_visible_raw_masks_are_subsets_of_final_masks"] is True
    assert (output / "object_anchor_projection_contact_sheet.png").is_file()
    for diagnostic in record["diagnostics"].values():
        assert (output / diagnostic["path"]).is_file()


def test_closing_fills_only_inside_a_depth_consistent_anchor_envelope(tmp_path: Path) -> None:
    visible_paths = make_fixture(tmp_path / "visible")
    occluded_paths = make_fixture(tmp_path / "occluded", occluded_center=True)

    visible_report = project(
        make_args(visible_paths, tmp_path / "visible_output", closing_radius=1)
    )
    occluded_report = project(
        make_args(occluded_paths, tmp_path / "occluded_output", closing_radius=1)
    )

    visible_record = visible_report["frame_records"][0]
    occluded_record = occluded_report["frame_records"][0]
    visible_mask = np.asarray(
        Image.open(tmp_path / "visible_output" / visible_record["union_mask"]).convert("L")
    ) > 0
    occluded_mask = np.asarray(
        Image.open(tmp_path / "occluded_output" / occluded_record["union_mask"]).convert("L")
    ) > 0
    assert visible_mask[6, 8]
    assert not occluded_mask[6, 8]
    assert visible_record["coverage"]["final_mask_pixels"] == 9
    assert occluded_record["coverage"]["final_mask_pixels"] == 8


def test_depth_aware_splat_never_removes_raw_visible_evidence(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)

    report = project(make_args(paths, tmp_path / "output", splat_radius=1))

    record = report["frame_records"][0]
    assert record["coverage"]["depth_aware_splat_pixels"] >= record["coverage"][
        "visible_raw_pixels"
    ]
    assert record["coverage"]["final_mask_pixels"] >= record["coverage"][
        "visible_raw_pixels"
    ]
    assert record["gates"]["visible_raw_is_subset_of_final_mask"] is True
    assert (
        record["gates"]["final_mask_is_subset_of_depth_consistent_anchor_envelope"]
        is True
    )


def test_invalid_camera_convention_and_invisible_anchor_fail_closed(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path / "bad_camera")
    camera = json.loads(paths["camera"].read_text(encoding="utf-8"))
    camera["extrinsic_type"] = "camera_to_world"
    write_json(paths["camera"], camera)
    with pytest.raises(ValueError, match="world_to_camera"):
        project(make_args(paths, tmp_path / "bad_camera_output"))

    paths = make_fixture(tmp_path / "invisible")
    np.save(paths["depth"] / "000000.npy", np.ones((12, 16), dtype=np.float32))
    with pytest.raises(ValueError, match="visible anchor pixels"):
        project(make_args(paths, tmp_path / "invisible_output"))
