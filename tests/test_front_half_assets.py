from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prepare = load_script("prepare_front_half_asset_inputs")
replacement = load_script("build_trellis_replacement_plan")


def test_selection_quality_penalizes_border_clipped_blurry_crop() -> None:
    good_quality, good_penalty, _ = prepare.selection_quality(
        support_ratio=0.92,
        instance_point_coverage=0.78,
        foreground_fill_ratio=0.58,
        sharpness_score=0.72,
        border_sides=0,
    )
    clipped_quality, clipped_penalty, _ = prepare.selection_quality(
        support_ratio=0.92,
        instance_point_coverage=0.78,
        foreground_fill_ratio=0.58,
        sharpness_score=0.08,
        border_sides=2,
    )
    assert good_penalty == 0.0
    assert clipped_penalty > 0.0
    assert good_quality > clipped_quality


def test_candidate_rank_prefers_less_extreme_fuller_crop_over_raw_area() -> None:
    fuller = prepare.candidate_rank_score(
        selection_quality=0.72,
        bbox_width=1056,
        bbox_height=501,
        mask_support_points=26000,
        border_contact_sides=1,
    )
    wider_clipped = prepare.candidate_rank_score(
        selection_quality=0.71,
        bbox_width=1242,
        bbox_height=485,
        mask_support_points=24000,
        border_contact_sides=1,
    )
    assert fuller > wider_clipped


def test_projected_seed_keeps_only_connected_instance_component() -> None:
    crop_mask = np.zeros((40, 80), dtype=bool)
    crop_mask[5:35, 8:32] = True
    crop_mask[5:35, 48:72] = True
    seed = np.zeros_like(crop_mask)
    seed[20, 18] = True
    cleaned, report = prepare.connected_component_from_seed(
        crop_mask,
        seed,
        min_component_support=0.01,
    )
    assert cleaned[20, 18]
    assert not cleaned[20, 60]
    assert report["retained_by"] == "projected_instance_seed"


def test_candidate_uses_object_id_mask_dir_before_category_dir(tmp_path: Path) -> None:
    masks = tmp_path / "masks"
    (masks / "gdino_object_window").mkdir(parents=True)
    (masks / "sam3_class_window").mkdir(parents=True)
    selected = prepare.find_mask_dir(masks, "gdino_object_window", "window", "window")
    assert selected == masks / "gdino_object_window"


def test_replacement_plan_uses_asset_pbr_and_original_point_cloud_bounds(tmp_path: Path) -> None:
    cloud_path = tmp_path / "window_cloud.ply"
    cloud_path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 4",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                "-1 -0.5 3",
                "1 -0.5 3",
                "-1 0.5 3",
                "1 0.5 3",
                "",
            ]
        ),
        encoding="utf-8",
    )
    asset_dir = tmp_path / "assets" / "gdino_object_window"
    asset_dir.mkdir(parents=True)
    obj_path = asset_dir / "asset_pbr.obj"
    obj_path.write_text(
        "\n".join(
            [
                "v -0.5 -0.5 0",
                "v 0.5 -0.5 0",
                "v -0.5 0.5 0",
                "v 0.5 0.5 0",
                "f 1 2 3",
                "f 2 4 3",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (asset_dir / "asset_pbr.glb").write_bytes(b"not-a-real-glb")
    assert replacement.find_asset(tmp_path / "assets", "gdino_object_window", str(obj_path)) == obj_path.resolve()
    job = {
        "object_id": "gdino_object_window",
        "category": "window",
        "source_point_cloud": str(cloud_path),
        "generated_asset": str(obj_path),
        "selection_quality": 0.8,
    }
    plan = replacement.replacement_for_job(job, tmp_path / "assets", bounds_percentile=100.0)
    assert plan["placement_mode"] == "planar_pca_preserve_scene_plane"
    assert plan["front_back_audit_required"] is True
    assert np.allclose(plan["target_center"], [0.0, 0.0, 3.0])
    assert len(plan["world_from_asset"]) == 4


def test_planar_replacement_warns_for_thick_source_cloud(tmp_path: Path) -> None:
    cloud_path = tmp_path / "polluted_window_cloud.ply"
    cloud_path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 8",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                "-1 -1 -1",
                "1 -1 -1",
                "-1 1 -1",
                "1 1 -1",
                "-1 -1 1",
                "1 -1 1",
                "-1 1 1",
                "1 1 1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    asset_dir = tmp_path / "assets" / "gdino_object_window"
    asset_dir.mkdir(parents=True)
    obj_path = asset_dir / "asset_pbr.obj"
    obj_path.write_text(
        "v -0.5 -0.5 -0.05\nv 0.5 -0.5 -0.05\nv -0.5 0.5 0.05\nf 1 2 3\n",
        encoding="utf-8",
    )
    plan = replacement.replacement_for_job(
        {
            "object_id": "gdino_object_window",
            "category": "window",
            "source_point_cloud": str(cloud_path),
            "generated_asset": str(obj_path),
        },
        tmp_path / "assets",
        bounds_percentile=100.0,
    )
    assert plan["status"] == "planned_with_warnings"
    assert plan["warnings"]


def test_planar_replacement_filters_densest_source_slab(tmp_path: Path) -> None:
    plane_points = [
        (x * 0.2, y * 0.2, 0.0)
        for x in range(-5, 5)
        for y in range(-2, 2)
    ]
    background_points = [(x * 0.2, 0.0, 4.0) for x in range(-5, 5)]
    points = plane_points + background_points
    cloud_path = tmp_path / "window_with_background_cloud.ply"
    cloud_path.write_text(
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
    asset_dir = tmp_path / "assets" / "gdino_object_window"
    asset_dir.mkdir(parents=True)
    obj_path = asset_dir / "asset_pbr.obj"
    obj_path.write_text(
        "\n".join(
            [
                "v -0.5 -0.5 -0.05",
                "v 0.5 -0.5 -0.05",
                "v -0.5 0.5 0.05",
                "v 0.5 0.5 0.05",
                "f 1 2 3",
                "f 2 4 3",
                "",
            ]
        ),
        encoding="utf-8",
    )
    plan = replacement.replacement_for_job(
        {
            "object_id": "gdino_object_window",
            "category": "window",
            "source_point_cloud": str(cloud_path),
            "generated_asset": str(obj_path),
        },
        tmp_path / "assets",
        bounds_percentile=96.0,
    )
    assert plan["placement_source_filter"]["mode"] == "planar_pca_densest_slab"
    assert plan["placement_source_filter"]["kept_points"] == len(plane_points)
    assert plan["planar_thickness_ratio"] < 0.18
