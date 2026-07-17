from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from scripts.associate_multiframe_instance_masks import (
    anchor_depth_map,
    associate,
    binary_dilate,
    build_parser,
    maximum_weight_assignment,
    visible_anchor_projections,
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_ascii_ply(path: Path, points: np.ndarray, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    rows = "\n".join(
        f"{x:.8f} {y:.8f} {z:.8f} {color[0]} {color[1]} {color[2]}" for x, y, z in points
    )
    path.write_text(header + rows + "\n", encoding="ascii")


def grid_points(center_x: float, *, z: float = 5.0) -> np.ndarray:
    x, y = np.meshgrid(
        np.linspace(center_x - 0.35, center_x + 0.35, 31),
        np.linspace(-0.30, 0.30, 27),
    )
    return np.column_stack([x.reshape(-1), y.reshape(-1), np.full(x.size, z)])


def projected_support(
    points: np.ndarray, matrix: np.ndarray, intrinsic: dict[str, float]
) -> np.ndarray:
    depth = anchor_depth_map(points, matrix, intrinsic)
    mask = np.isfinite(depth).reshape(int(intrinsic["h"]), int(intrinsic["w"]))
    return mask


def make_fixture(root: Path, *, moving_cameras: bool = True) -> dict[str, Path]:
    frames_dir = root / "frames"
    masks_dir = root / "masks"
    frames_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)
    intrinsic = {"fx": 80.0, "fy": 80.0, "cx": 48.0, "cy": 36.0, "w": 96, "h": 72}
    anchors = {
        "physical_alpha": grid_points(-0.75),
        "physical_beta": grid_points(0.75),
    }
    anchor_paths: dict[str, Path] = {}
    for object_id, points in anchors.items():
        path = root / "anchors" / f"{object_id}.ply"
        color = (245, 245, 240) if object_id == "physical_alpha" else (40, 80, 220)
        write_ascii_ply(path, points, color)
        anchor_paths[object_id] = path

    items: list[dict[str, Any]] = []
    extrinsics: dict[str, list[list[float]]] = {}
    for frame_index, camera_x in enumerate((-0.6, 0.0, 0.6)):
        frame_id = f"{frame_index:06d}"
        if not moving_cameras:
            camera_x = 0.0
        matrix = np.eye(4, dtype=np.float64)
        matrix[0, 3] = -camera_x
        extrinsics[frame_id] = matrix.tolist()
        rgb = np.full((72, 96, 3), 100, dtype=np.uint8)
        frame_items: list[dict[str, Any]] = []
        for object_index, (object_id, points) in enumerate(reversed(list(anchors.items()))):
            support = projected_support(points, matrix, intrinsic)
            mask = binary_dilate(support, 1)
            rows, columns = np.nonzero(mask)
            # These names intentionally suggest the opposite physical object.
            misleading = "alpha" if object_id == "physical_beta" else "beta"
            mask_path = masks_dir / f"looks_like_{misleading}_{frame_id}_{object_index}.png"
            Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
            color = [40, 80, 220] if object_id == "physical_beta" else [245, 245, 240]
            rgb[mask] = color
            frame_items.append(
                {
                    "image": f"{frame_id}.png",
                    "label": "pillow",
                    "mask_path": str(mask_path),
                    "bbox": [
                        float(columns.min()),
                        float(rows.min()),
                        float(columns.max() + 1),
                        float(rows.max() + 1),
                    ],
                    "score": 0.96 - object_index * 0.01,
                }
            )
        Image.fromarray(rgb).save(frames_dir / f"{frame_id}.png")
        items.extend(frame_items)
    camera_path = root / "camera_info.json"
    write_json(
        camera_path,
        {
            "extrinsic_type": "world_to_camera",
            "intrinsic": intrinsic,
            "extrinsic": extrinsics,
        },
    )
    index_path = root / "mask_index.json"
    write_json(index_path, {"scene": "synthetic", "items": items})
    return {
        "camera": camera_path,
        "index": index_path,
        "frames": frames_dir,
        "alpha": anchor_paths["physical_alpha"],
        "beta": anchor_paths["physical_beta"],
    }


def fixture_args(paths: dict[str, Path], output: Path, *extra: str):
    parser = build_parser()
    return parser.parse_args(
        [
            "--anchor",
            f"physical_alpha={paths['alpha']}",
            "--anchor",
            f"physical_beta={paths['beta']}",
            "--camera-info",
            str(paths["camera"]),
            "--mask-index",
            str(paths["index"]),
            "--frames-dir",
            str(paths["frames"]),
            "--mask-label",
            "pillow",
            "--output",
            str(output),
            "--min-visible-projected-pixels",
            "20",
            "--min-mask-hit-ratio",
            "0.95",
            "--min-dilated-support-ratio",
            "0.95",
            "--min-bbox-iou",
            "0.50",
            "--support-dilation",
            "1",
            "--min-camera-baseline",
            "0.40",
            "--min-view-angle-degrees",
            "4.0",
            *extra,
        ]
    )


def test_geometry_assignment_ignores_mask_filenames_and_excludes_peeled_residuals(
    tmp_path: Path,
) -> None:
    paths = make_fixture(tmp_path)
    output = tmp_path / "association.json"
    args = fixture_args(paths, output, "--peeled-object", "physical_alpha")

    report = associate(args)

    assert report["status"] == "passed"
    assert report["next_layer_target_ids"] == ["physical_beta"]
    assert len(report["residual_warnings"]) == 3
    assert all(warning["object_id"] == "physical_alpha" for warning in report["residual_warnings"])
    assert all(len(frame["assignments"]) == 2 for frame in report["frames"])
    for frame in report["frames"]:
        alpha = next(item for item in frame["assignments"] if item["object_id"] == "physical_alpha")
        beta = next(item for item in frame["assignments"] if item["object_id"] == "physical_beta")
        alpha_source = next(
            item
            for item in report["sources"]["masks"]
            if item["detection_id"] == alpha["detection_id"]
        )
        beta_source = next(
            item
            for item in report["sources"]["masks"]
            if item["detection_id"] == beta["detection_id"]
        )
        assert "looks_like_beta" in alpha_source["resolved_path"]
        assert "looks_like_alpha" in beta_source["resolved_path"]
        assert alpha["metrics"]["rgb_stats"]["sam3_mask"]["mean_rgb"][0] > 240
        assert beta["metrics"]["rgb_stats"]["sam3_mask"]["mean_rgb"][2] > 210
    beta_record = next(item for item in report["objects"] if item["object_id"] == "physical_beta")
    assert beta_record["selected_evidence_frame_count"] == 2
    assert beta_record["evidence_gate_passed"] is True
    assert beta_record["selected_frame_separation"][0]["passed"] is True
    assert output.is_file()
    receipt = json.loads(output.with_suffix(".json.sha256.json").read_text(encoding="utf-8"))
    assert len(receipt["report_sha256"]) == 64


def test_insufficient_camera_separation_fails_closed(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path, moving_cameras=False)
    args = fixture_args(paths, tmp_path / "association.json")

    report = associate(args)

    assert report["status"] == "failed"
    assert report["next_layer_target_ids"] == []
    assert report["gates"]["passed"] is False
    assert len(report["gates"]["failure_reasons"]) == 2
    assert all(
        record["disposition"] == "failed_insufficient_separated_evidence"
        for record in report["objects"]
    )


def test_cross_anchor_z_buffer_hides_rear_projected_pixels() -> None:
    intrinsic = {"fx": 50.0, "fy": 50.0, "cx": 10.0, "cy": 10.0, "w": 20, "h": 20}
    matrix = np.eye(4)
    front = np.array([[0.0, 0.0, 4.0], [0.0, 0.0, 4.1]])
    rear = np.array([[0.0, 0.0, 6.0], [0.0, 0.0, 6.1]])
    maps = [
        anchor_depth_map(front, matrix, intrinsic),
        anchor_depth_map(rear, matrix, intrinsic),
    ]

    projections = visible_anchor_projections(
        maps,
        width=20,
        height=20,
        absolute_tolerance=0.01,
        relative_tolerance=0.0,
    )

    assert projections[0]["projected_unique_pixels"] == 1
    assert projections[0]["visible_unique_pixels"] == 1
    assert projections[1]["projected_unique_pixels"] == 1
    assert projections[1]["visible_unique_pixels"] == 0
    assert projections[1]["occluded_unique_pixels"] == 1


def test_one_to_one_assignment_is_globally_optimal_not_row_greedy() -> None:
    # Greedy row order would choose row0->col0 (0.90), leaving row1->col1
    # (0.10). The maximum-weight solution is row0->col1 and row1->col0.
    weights = np.array(
        [
            [0.90, 0.80, 0.0, 0.0],
            [0.85, 0.10, 0.0, 0.0],
        ]
    )

    assert maximum_weight_assignment(weights) == [1, 0]


def test_camera_convention_and_peeled_ids_fail_closed(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    camera = json.loads(paths["camera"].read_text(encoding="utf-8"))
    camera["extrinsic_type"] = "camera_to_world"
    write_json(paths["camera"], camera)
    with pytest.raises(ValueError, match="world_to_camera"):
        associate(fixture_args(paths, tmp_path / "bad_camera.json"))

    paths = make_fixture(tmp_path / "unknown")
    with pytest.raises(ValueError, match="no explicit 3D anchor"):
        associate(
            fixture_args(
                paths,
                tmp_path / "unknown" / "bad_peeled.json",
                "--peeled-object",
                "filename_invented_identity",
            )
        )
