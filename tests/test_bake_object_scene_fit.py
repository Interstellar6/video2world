from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import trimesh
from plyfile import PlyData

from scripts.bake_object_scene_fit import bake_scene_fit
from scripts.build_parametric_soft_object import (
    build_superellipsoid,
    color_mesh,
    write_gaussian_ply,
)


def test_scene_fit_bakes_obb_into_mesh_and_gaussian_covariance(tmp_path: Path) -> None:
    source_mesh = build_superellipsoid(
        width=1.2,
        height=1.0,
        depth=0.3,
        latitude_segments=12,
        longitude_segments=24,
        shape_exponent=0.34,
    )
    color_mesh(
        source_mesh,
        np.full((8, 8, 3), [210, 212, 214], dtype=np.uint8),
        appearance_mode="source",
    )
    mesh_path = tmp_path / "source.glb"
    gaussian_path = tmp_path / "source.ply"
    source_mesh.export(mesh_path)
    write_gaussian_ply(gaussian_path, source_mesh, count=512, seed=42)

    angle = math.radians(27)
    axes = [
        [math.cos(angle), -math.sin(angle), 0.0],
        [math.sin(angle), math.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ]
    report_path = tmp_path / "instances.json"
    report_path.write_text(
        json.dumps(
            {
                "instances": [
                    {
                        "object_id": "pillow01",
                        "geometry": {
                            "obb": {
                                "center": [3.0, 4.0, 5.0],
                                "extents": [2.4, 1.5, 0.6],
                                "axes_columns": axes,
                            }
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with np.errstate(all="raise"):
        result = bake_scene_fit(
            object_id="pillow01",
            gaussian_source=gaussian_path,
            mesh_source=mesh_path,
            instance_report=report_path,
            output_dir=tmp_path / "fitted",
        )

    assert result["all_acceptance_gates_passed"] is True
    np.testing.assert_allclose(
        result["fit"]["projected_fitted_extents"],
        [2.4, 1.5, 0.6],
        atol=1e-5,
    )
    assert result["baked_relative_transform"]["runtime_pivot"] == [3.0, 4.0, 5.0]
    assert result["promotion_status"] == "held_pending_scene_browser_review"

    fitted_mesh = trimesh.load(
        tmp_path / "fitted" / "pillow01.scene-fit.glb",
        force="scene",
    ).to_geometry()
    assert fitted_mesh.is_watertight
    assert fitted_mesh.is_winding_consistent

    vertices = PlyData.read(tmp_path / "fitted" / "pillow01_gaussian_scene_fit.ply")["vertex"].data
    numeric = np.column_stack(
        [
            vertices[name]
            for name in (
                "x",
                "y",
                "z",
                "nx",
                "ny",
                "nz",
                "scale_0",
                "scale_1",
                "scale_2",
            )
        ]
    )
    assert np.isfinite(numeric).all()
    rotations = np.column_stack([vertices[f"rot_{channel}"] for channel in range(4)])
    np.testing.assert_allclose(np.linalg.norm(rotations, axis=1), 1.0, atol=1e-5)

    with np.errstate(all="raise"):
        anchored = bake_scene_fit(
            object_id="pillow01",
            gaussian_source=gaussian_path,
            mesh_source=mesh_path,
            instance_report=report_path,
            output_dir=tmp_path / "front-anchored",
            placement_mode="observed_front_surface",
            view_direction=(0.0, 0.0, 1.0),
            thickness_axis=2,
            max_thickness_ratio=0.2,
            target_extent_scale=(0.8, 0.8, 1.0),
        )

    assert anchored["all_acceptance_gates_passed"] is True
    np.testing.assert_allclose(anchored["completed_obb"]["extents"], [1.92, 1.2, 0.384])
    np.testing.assert_allclose(
        anchored["baked_relative_transform"]["runtime_pivot"],
        [3.0, 4.0, 5.192],
    )
    assert anchored["placement_evidence"]["mode"] == "observed_front_surface"
    assert anchored["placement_evidence"]["front_surface_anchor_error"] <= 1e-12


def test_scene_fit_accepts_unified_surface_bvh_without_gaussian(tmp_path: Path) -> None:
    mesh = build_superellipsoid(
        width=1.0,
        height=0.9,
        depth=0.3,
        latitude_segments=10,
        longitude_segments=20,
        shape_exponent=0.4,
    )
    color_mesh(
        mesh,
        np.full((8, 8, 3), [220, 220, 216], dtype=np.uint8),
        appearance_mode="source",
    )
    mesh.update_faces(np.arange(len(mesh.faces)) != 0)
    assert mesh.is_watertight is False
    assert mesh.is_winding_consistent is True
    mesh_path = tmp_path / "surface.glb"
    mesh.export(mesh_path)
    report_path = tmp_path / "instances.json"
    report_path.write_text(
        json.dumps(
            {
                "instances": [
                    {
                        "object_id": "pillow_surface",
                        "geometry": {
                            "obb": {
                                "center": [1.0, 2.0, 3.0],
                                "extents": [2.0, 1.8, 0.6],
                                "axes_columns": np.eye(3).tolist(),
                            }
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = bake_scene_fit(
        object_id="pillow_surface",
        gaussian_source=None,
        mesh_source=mesh_path,
        instance_report=report_path,
        output_dir=tmp_path / "fitted",
        collision_topology="surface_bvh",
        max_collision_faces=100_000,
    )

    assert result["all_acceptance_gates_passed"] is True
    assert result["representation_mode"] == "unified_mesh_visual_logic_collision"
    assert result["gaussian"] is None
    assert result["sources"]["gaussian"] is None
    assert result["mesh"]["watertight"] is False
    assert result["collision_semantics"] == {
        "asset_mode": "unified_glb",
        "topology": "surface_bvh",
        "max_faces": 100_000,
        "surface_blocking": True,
        "closed_volume_claim": False,
        "inside_outside_queries_allowed": False,
        "fallback_proxy_allowed": False,
    }
    assert not (tmp_path / "fitted" / "pillow_surface_gaussian_scene_fit.ply").exists()


def test_scene_fit_can_correct_fabric_pbr_without_changing_textured_geometry(
    tmp_path: Path,
) -> None:
    mesh = build_superellipsoid(
        width=1.0,
        height=0.9,
        depth=0.3,
        latitude_segments=10,
        longitude_segments=20,
        shape_exponent=0.4,
    )
    color_mesh(
        mesh,
        np.full((8, 8, 3), [226, 224, 218], dtype=np.uint8),
        appearance_mode="source",
    )
    mesh.visual = trimesh.visual.texture.TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2), dtype=np.float64),
        material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=[226, 224, 218, 255],
            metallicFactor=1.0,
            roughnessFactor=0.3,
        ),
    )
    mesh_path = tmp_path / "metallic-pillow.glb"
    mesh.export(mesh_path)
    report_path = tmp_path / "instances.json"
    report_path.write_text(
        json.dumps(
            {
                "instances": [
                    {
                        "object_id": "fabric_pillow",
                        "geometry": {
                            "obb": {
                                "center": [0.0, 0.0, 0.0],
                                "extents": [2.0, 1.8, 0.6],
                                "axes_columns": np.eye(3).tolist(),
                            }
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = bake_scene_fit(
        object_id="fabric_pillow",
        gaussian_source=None,
        mesh_source=mesh_path,
        instance_report=report_path,
        output_dir=tmp_path / "fitted",
        material_profile="nonmetal_fabric",
    )

    assert result["all_acceptance_gates_passed"] is True
    assert result["material_profile"] == "nonmetal_fabric"
    assert result["acceptance_gates"]["material_profile_applied"] is True
    assert result["mesh"]["materials"] == [
        {
            "type": "PBRMaterial",
            "metallic_factor_before": 1.0,
            "roughness_factor_before": 0.3,
            "metallic_factor_after": 0.0,
            "roughness_factor_after": 1.0,
        }
    ]
    fitted = trimesh.load(
        tmp_path / "fitted" / "fabric_pillow.scene-fit.glb",
        force="scene",
    )
    output_material = next(iter(fitted.geometry.values())).visual.material
    assert output_material.metallicFactor == 0.0
    assert output_material.roughnessFactor == 1.0
