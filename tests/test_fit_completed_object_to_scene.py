from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image
from plyfile import PlyData, PlyElement

from scripts.fit_completed_object_to_scene import (
    _build_transformed_scene,
    _candidate_matrix,
    _load_flattened_geometries,
    _scene_mesh,
    fit_completed_object_to_scene,
    right_handed_axis_permutations,
    robust_oriented_frame,
    sha256_file,
)
from scripts.qa_scene_camera_silhouette import load_camera, project_mesh_silhouette


def _write_ply(path: Path, points: np.ndarray) -> None:
    data = np.empty(len(points), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    for index, name in enumerate(("x", "y", "z")):
        data[name] = points[:, index]
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(path)


def _write_camera(path: Path) -> None:
    path.write_text(
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
                }
            ]
        ),
        encoding="utf-8",
    )


def _pbr_box(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=(1.0, 0.5, 0.25))
    mesh.apply_translation([3.0, -2.0, 1.0])
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2), dtype=np.float64),
        material=trimesh.visual.material.PBRMaterial(
            name="object",
            baseColorFactor=[230, 228, 220, 255],
            metallicFactor=0.0,
            roughnessFactor=0.9,
        ),
    )
    scene = trimesh.Scene(base_frame="world")
    node_transform = np.eye(4)
    node_transform[:3, 3] = [0.25, 0.1, -0.2]
    scene.add_geometry(
        mesh,
        node_name="source_part",
        geom_name="source.geometry",
        transform=node_transform,
    )
    scene.export(path)
    return mesh


def _box_points(
    rng: np.random.Generator,
    *,
    center: np.ndarray,
    extents: np.ndarray,
    count: int = 4000,
) -> np.ndarray:
    return rng.uniform(-0.5, 0.5, size=(count, 3)) * extents + center


def test_axis_permutations_are_the_24_proper_signed_rotations() -> None:
    permutations = right_handed_axis_permutations()

    assert len(permutations) == 24
    assert len({tuple(matrix.reshape(-1)) for matrix in permutations}) == 24
    for matrix in permutations:
        np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(matrix), 1.0, atol=1e-12)


def test_every_axis_permutation_uses_uniform_initial_scale_in_target_order() -> None:
    source_extents = np.asarray([1.0, 2.0, 3.0])
    target_extents = np.asarray([7.0, 5.0, 4.0])
    canonical_vertices = trimesh.bounds.corners(
        np.asarray([-source_extents / 2.0, source_extents / 2.0])
    )
    target = {
        "center": np.zeros(3),
        "axes_columns": np.eye(3),
        "extents": target_extents,
    }

    for permutation in right_handed_axis_permutations():
        matrix, fit = _candidate_matrix(
            permutation=permutation,
            parameters=np.zeros(9),
            source_extents=source_extents,
            target=target,
            canonical_vertices=canonical_vertices,
            maximum_scale_multiplier=1.35,
            maximum_scale_anisotropy=1.15,
            maximum_rotation_radians=np.radians(15.0),
            maximum_translation_ratio=0.2,
            support=None,
            front=None,
        )
        np.testing.assert_allclose(
            fit["scale_xyz"],
            np.full(3, fit["initial_uniform_scale"]),
            atol=1e-12,
        )
        expected_order = np.einsum("ij,j->i", np.abs(permutation), source_extents, optimize=False)
        np.testing.assert_allclose(fit["permuted_source_extents_xyz"], expected_order, atol=1e-12)
        assert np.linalg.det(matrix[:3, :3]) > 0


def test_extreme_target_extents_cannot_force_excessive_anisotropy() -> None:
    source_extents = np.asarray([1.0, 1.0, 1.0])
    target = {
        "center": np.zeros(3),
        "axes_columns": np.eye(3),
        "extents": np.asarray([20.0, 1.0, 0.1]),
    }
    canonical_vertices = trimesh.bounds.corners(
        np.asarray([-source_extents / 2.0, source_extents / 2.0])
    )
    maximum_anisotropy = 1.15

    _, fit = _candidate_matrix(
        permutation=np.eye(3),
        parameters=np.asarray([2.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        source_extents=source_extents,
        target=target,
        canonical_vertices=canonical_vertices,
        maximum_scale_multiplier=1.35,
        maximum_scale_anisotropy=maximum_anisotropy,
        maximum_rotation_radians=np.radians(15.0),
        maximum_translation_ratio=0.2,
        support=None,
        front=None,
    )

    assert fit["initial_scale_xyz"] == [1.0, 1.0, 1.0]
    assert fit["scale_anisotropy_ratio"] <= maximum_anisotropy + 1e-12


def test_target_extent_scale_initialization_preserves_planar_aspect_ratio() -> None:
    source_extents = np.asarray([1.0, 0.2, 1.0])
    target = {
        "center": np.zeros(3),
        "axes_columns": np.eye(3),
        "extents": np.asarray([12.0, 0.6, 5.0]),
    }
    canonical_vertices = trimesh.bounds.corners(
        np.asarray([-source_extents / 2.0, source_extents / 2.0])
    )

    _, fit = _candidate_matrix(
        permutation=np.eye(3),
        parameters=np.zeros(9),
        source_extents=source_extents,
        target=target,
        canonical_vertices=canonical_vertices,
        initial_scale_mode="target_extents",
        maximum_scale_multiplier=1.35,
        maximum_scale_anisotropy=32.0,
        maximum_rotation_radians=np.radians(15.0),
        maximum_translation_ratio=0.2,
        support=None,
        front=None,
    )

    assert fit["initial_scale_mode"] == "target_extents"
    np.testing.assert_allclose(fit["initial_scale_xyz"], [12.0, 3.0, 5.0], atol=1e-12)
    np.testing.assert_allclose(fit["scale_xyz"], [12.0, 3.0, 5.0], atol=1e-12)


def test_candidate_reports_semantic_up_alignment_after_axis_permutation() -> None:
    source_extents = np.asarray([1.0, 2.0, 3.0])
    target = {
        "center": np.zeros(3),
        "axes_columns": np.eye(3),
        "extents": np.asarray([1.0, 2.0, 3.0]),
    }
    canonical_vertices = trimesh.bounds.corners(
        np.asarray([-source_extents / 2.0, source_extents / 2.0])
    )
    flip_y_and_z = np.diag([1.0, -1.0, -1.0])

    _, fit = _candidate_matrix(
        permutation=flip_y_and_z,
        parameters=np.zeros(9),
        source_extents=source_extents,
        target=target,
        canonical_vertices=canonical_vertices,
        maximum_scale_multiplier=1.35,
        maximum_scale_anisotropy=1.15,
        maximum_rotation_radians=np.radians(15.0),
        maximum_translation_ratio=0.2,
        support=None,
        front=None,
        semantic_up={
            "canonical_source_up": np.asarray([0.0, 1.0, 0.0]),
            "world_up": np.asarray([0.0, -1.0, 0.0]),
            "maximum_tilt_degrees": 10.0,
        },
    )

    assert fit["semantic_up_alignment"]["tilt_degrees"] == 0.0
    np.testing.assert_allclose(
        fit["semantic_up_alignment"]["mapped_source_up_world"],
        [0.0, -1.0, 0.0],
        atol=1e-12,
    )


def test_robust_frame_ignores_sparse_far_outliers() -> None:
    rng = np.random.default_rng(9)
    inliers = _box_points(
        rng,
        center=np.asarray([4.0, -2.0, 7.0]),
        extents=np.asarray([2.0, 1.0, 0.5]),
        count=10_000,
    )
    outliers = np.asarray([[1000.0, -900.0, 700.0], [-800.0, 1200.0, -600.0]])

    frame = robust_oriented_frame(np.vstack((inliers, outliers)), robust_quantiles=(0.01, 0.99))

    assert float(np.max(frame["extents"])) < 2.2
    np.testing.assert_allclose(frame["center"], [4.0, -2.0, 7.0], atol=0.05)


def test_single_view_fit_exports_local_and_world_baked_assets(tmp_path: Path) -> None:
    mesh_path = tmp_path / "completed.glb"
    _pbr_box(mesh_path)
    rng = np.random.default_rng(17)
    target_center = np.asarray([1.8, 0.8, 7.5])
    target_extents = np.asarray([2.4, 1.1, 0.7])
    object_points = _box_points(rng, center=target_center, extents=target_extents, count=5000)
    object_anchor = tmp_path / "object.ply"
    _write_ply(object_anchor, object_points)

    cameras_path = tmp_path / "cameras.json"
    _write_camera(cameras_path)
    target_mesh = trimesh.creation.box(extents=target_extents)
    target_mesh.apply_translation(target_center)
    camera = load_camera(cameras_path, "000001")
    observed = project_mesh_silhouette(
        target_mesh,
        runtime_pivot=np.zeros(3),
        camera=camera,
    )
    mask_path = tmp_path / "000001.png"
    Image.fromarray(observed.astype(np.uint8) * 255).save(mask_path)
    manifest_path = tmp_path / "views.json"
    manifest_path.write_text(
        json.dumps(
            {
                "object_id": "pillow",
                "views": [
                    {
                        "frame_id": "000001",
                        "observed_mask": str(mask_path),
                        "is_source_view": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = fit_completed_object_to_scene(
        object_id="pillow",
        mesh_source=mesh_path,
        object_anchor_ply=object_anchor,
        cameras_path=cameras_path,
        view_manifest_path=manifest_path,
        output_dir=tmp_path / "fit",
        cycles=0,
        refine_axis_candidates=1,
        minimum_mask_iou=0.0,
        minimum_bbox_iou=0.0,
        maximum_center_error_px=1000.0,
    )

    assert report["all_acceptance_gates_passed"] is True
    assert report["promotion_allowed"] is False
    assert report["promotion_status"] == "held_pending_multiview_evidence"
    assert len(report["search"]["axis_candidates"]) == 24
    assert report["final_export_qa"]["glb_obj_bounds_match"] is True
    assert report["final_export_qa"]["canonical_to_world_round_trip"] is True
    assert report["final_export_qa"]["scene_local_pivot_round_trip"] is True
    assert report["world_baked_transform"]["asset_coordinates_baked"] is True
    assert report["exports"]["canonical_local"]["glb_role"] == ("pbr_visual_logic_authority")
    local_center = np.asarray(report["exports"]["canonical_local"]["glb"]["center"])
    world_center = np.asarray(report["exports"]["scene_space"]["glb"]["center"])
    assert np.linalg.norm(local_center) < 1e-5
    assert np.linalg.norm(world_center - target_center) < 0.1
    assert np.linalg.norm(world_center - local_center) > 1.0
    assert (tmp_path / "fit" / "pillow.canonical.glb").is_file()
    assert (tmp_path / "fit" / "pillow.canonical.obj").is_file()
    assert (tmp_path / "fit" / "pillow.scene-local.glb").is_file()
    assert (tmp_path / "fit" / "pillow.scene-local.obj").is_file()
    assert (tmp_path / "fit" / "pillow.scene-space.glb").is_file()
    assert (tmp_path / "fit" / "pillow.scene-space.obj").is_file()
    assert (tmp_path / "fit" / "scene_fit_receipt.json").is_file()


def test_support_and_visible_front_planes_are_enforced_together(tmp_path: Path) -> None:
    mesh_path = tmp_path / "completed.glb"
    _pbr_box(mesh_path)
    rng = np.random.default_rng(27)
    center = np.asarray([0.2, 1.5, 6.5])
    extents = np.asarray([2.0, 1.0, 0.8])
    object_points = _box_points(rng, center=center, extents=extents, count=5000)
    object_anchor = tmp_path / "object.ply"
    _write_ply(object_anchor, object_points)
    support_x, support_z = np.meshgrid(np.linspace(-3, 3, 30), np.linspace(3, 10, 30))
    support_points = np.column_stack(
        (support_x.reshape(-1), np.full(support_x.size, 2.0), support_z.reshape(-1))
    )
    support_anchor = tmp_path / "support.ply"
    _write_ply(support_anchor, support_points)

    cameras_path = tmp_path / "cameras.json"
    _write_camera(cameras_path)
    target_mesh = trimesh.creation.box(extents=extents)
    target_mesh.apply_translation(center)
    observed = project_mesh_silhouette(
        target_mesh,
        runtime_pivot=np.zeros(3),
        camera=load_camera(cameras_path, "000001"),
    )
    mask_path = tmp_path / "000001.png"
    Image.fromarray(observed.astype(np.uint8) * 255).save(mask_path)
    manifest_path = tmp_path / "views.json"
    manifest_path.write_text(
        json.dumps(
            {
                "object_id": "supported",
                "views": [
                    {
                        "frame_id": "000001",
                        "observed_mask": str(mask_path),
                        "is_source_view": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = fit_completed_object_to_scene(
        object_id="supported",
        mesh_source=mesh_path,
        object_anchor_ply=object_anchor,
        support_anchor_ply=support_anchor,
        cameras_path=cameras_path,
        view_manifest_path=manifest_path,
        output_dir=tmp_path / "fit",
        placement_mode="observed_front_surface",
        robust_quantiles=(0.01, 0.99),
        cycles=0,
        refine_axis_candidates=1,
        minimum_mask_iou=0.0,
        minimum_bbox_iou=0.0,
        maximum_center_error_px=1000.0,
    )

    errors = report["search"]["selected"]["fit"]["constraint_errors"]
    assert errors["support_plane_error"] < 1e-8
    assert errors["visible_front_plane_error"] < 1e-8
    assert report["acceptance_gates"]["support_contact"] is True
    assert report["acceptance_gates"]["visible_front_surface"] is True


def test_initial_scene_fit_is_refined_about_its_bottom_under_real_support(
    tmp_path: Path,
) -> None:
    source_mesh = trimesh.creation.box(extents=(1.0, 2.0, 0.8))
    source_mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(source_mesh.vertices), 2), dtype=np.float64),
        material=trimesh.visual.material.PBRMaterial(
            name="supported_object",
            baseColorFactor=[180, 210, 170, 255],
            metallicFactor=0.0,
            roughnessFactor=0.8,
        ),
    )
    source_scene = trimesh.Scene(base_frame="world")
    source_scene.add_geometry(source_mesh, node_name="source", geom_name="source.geometry")
    mesh_path = tmp_path / "supported.glb"
    source_scene.export(mesh_path)

    source_geometries, _ = _load_flattened_geometries(mesh_path)
    flattened_source = _scene_mesh(
        _build_transformed_scene(source_geometries, np.eye(4), object_id="supported")
    )
    source_frame = robust_oriented_frame(np.asarray(flattened_source.vertices))
    source_to_canonical = np.eye(4)
    source_to_canonical[:3, :3] = source_frame["axes_columns"].T
    source_to_canonical[:3, 3] = -source_frame["axes_columns"].T @ source_frame["center"]
    canonical_mesh = _scene_mesh(
        _build_transformed_scene(
            source_geometries, source_to_canonical, object_id="supported"
        )
    )
    source_to_world = np.eye(4)
    source_to_world[:3, :3] = np.diag([1.0, -1.0, -1.0])
    source_to_world[:3, 3] = [0.1, 0.2, 7.0]
    initial_world = source_to_world @ np.linalg.inv(source_to_canonical)
    canonical_vertices = np.asarray(canonical_mesh.vertices, dtype=np.float64)
    initial_vertices_world = np.einsum(
        "ni,ji->nj", canonical_vertices, initial_world[:3, :3], optimize=False
    ) + initial_world[:3, 3]

    rng = np.random.default_rng(20260720)
    canonical_anchor = rng.uniform(
        canonical_vertices.min(axis=0), canonical_vertices.max(axis=0), size=(5000, 3)
    )
    object_points = np.einsum(
        "ni,ji->nj", canonical_anchor, initial_world[:3, :3], optimize=False
    ) + initial_world[:3, 3]
    object_anchor = tmp_path / "object.ply"
    _write_ply(object_anchor, object_points)
    initial_bottom_y = float(np.max(initial_vertices_world[:, 1]))
    support_y = initial_bottom_y - 0.1
    support_x, support_z = np.meshgrid(np.linspace(-2, 2, 30), np.linspace(4, 10, 30))
    support_points = np.column_stack(
        (support_x.reshape(-1), np.full(support_x.size, support_y), support_z.reshape(-1))
    )
    support_anchor = tmp_path / "support.ply"
    _write_ply(support_anchor, support_points)

    cameras_path = tmp_path / "cameras.json"
    cameras_path.write_text(
        json.dumps(
            [
                {
                    "img_name": frame_id,
                    "width": 200,
                    "height": 160,
                    "position": [camera_x, 0.0, 0.0],
                    "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "fx": 120,
                    "fy": 120,
                }
                for frame_id, camera_x in (
                    ("000001", -0.4),
                    ("000002", 0.0),
                    ("000003", 0.4),
                )
            ]
        ),
        encoding="utf-8",
    )
    view_entries = []
    contact_entries = []
    for frame_id in ("000001", "000002", "000003"):
        silhouette = project_mesh_silhouette(
            canonical_mesh,
            runtime_pivot=None,
            camera=load_camera(cameras_path, frame_id),
            scene_transform=initial_world,
        )
        mask_path = tmp_path / f"{frame_id}.png"
        Image.fromarray(silhouette.astype(np.uint8) * 255).save(mask_path)
        mask_record = {"path": str(mask_path), "sha256": sha256_file(mask_path)}
        view_entries.append(
            {"frame_id": frame_id, "observed_mask": mask_record, "is_source_view": True}
        )
        contact_entries.append(
            {"frame_id": frame_id, "support_contact_mask": mask_record}
        )
    view_manifest = tmp_path / "views.json"
    view_manifest.write_text(
        json.dumps({"object_id": "supported", "views": view_entries}), encoding="utf-8"
    )
    contact_manifest = tmp_path / "contacts.json"
    contact_manifest.write_text(
        json.dumps({"object_id": "supported", "views": contact_entries}), encoding="utf-8"
    )
    canonical_source_up = source_to_canonical[:3, :3] @ np.asarray([0.0, 1.0, 0.0])
    canonical_source_up /= np.linalg.norm(canonical_source_up)
    initial_receipt = tmp_path / "initial-receipt.json"
    initial_receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "video2world.completed_object_scene_fit",
                "object_id": "supported",
                "all_acceptance_gates_passed": True,
                "acceptance_gates": {
                    "view_silhouettes": True,
                    "semantic_up_alignment": True,
                },
                "sources": {
                    "mesh": {"sha256": sha256_file(mesh_path)},
                    "object_anchor": {"sha256": sha256_file(object_anchor)},
                    "cameras": {"sha256": sha256_file(cameras_path)},
                    "view_manifest": {"sha256": sha256_file(view_manifest)},
                },
                "canonicalization": {
                    "source_to_canonical_matrix_row_major": source_to_canonical.tolist()
                },
                "canonical_to_world": {"matrix_row_major": initial_world.tolist()},
                "semantic_orientation": {
                    "source_up_axis": "+Y",
                    "world_up_axis": "-Y",
                    "canonical_source_up": canonical_source_up.tolist(),
                },
            }
        ),
        encoding="utf-8",
    )

    report = fit_completed_object_to_scene(
        object_id="supported",
        mesh_source=mesh_path,
        object_anchor_ply=object_anchor,
        support_anchor_ply=support_anchor,
        initial_scene_fit_receipt=initial_receipt,
        support_contact_manifest_path=contact_manifest,
        cameras_path=cameras_path,
        view_manifest_path=view_manifest,
        output_dir=tmp_path / "refined",
        cycles=3,
        maximum_scale_multiplier=1.15,
        maximum_scale_anisotropy=2.0,
        maximum_rotation_degrees=5.0,
        maximum_translation_ratio=0.1,
        maximum_vertical_compression=0.3,
        maximum_support_penetration=0.05,
        minimum_mask_iou=0.5,
        minimum_bbox_iou=0.6,
        maximum_center_error_px=15.0,
        source_up_axis="+Y",
        world_up_axis="-Y",
        maximum_up_tilt_degrees=5.0,
        support_contact_error_score_weight=0.002,
        maximum_support_contact_error_px=8.0,
    )

    assert report["all_acceptance_gates_passed"] is True
    assert all(report["acceptance_gates"].values())
    assert report["search"]["method"] == (
        "validated_initial_fit_bottom_pivot_support_refinement"
    )
    assert report["search"]["initial_refinement_geometry"]["initial_support_contact"][
        "maximum_penetration"
    ] > 0.05
    assert report["support_contact"]["maximum_penetration"] <= 0.05 + 1e-8
    assert report["support_contact"]["minimum_gap"] < 1e-8
    assert all(
        gate["support_contact_bottom_error_px"] for gate in report["view_acceptance_gates"]
    )
    assert all(
        "source_camera_support_contact" in view for view in report["views"]
    )
    assert report["semantic_orientation"]["selected_alignment"]["tilt_degrees"] <= 5.0
    assert report["sources"]["initial_scene_fit_receipt"][
        "recomputed_view_silhouettes_passed"
    ] is True

    tampered = json.loads(initial_receipt.read_text(encoding="utf-8"))
    tampered["sources"]["mesh"]["sha256"] = "0" * 64
    initial_receipt.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="mesh hash does not match"):
        fit_completed_object_to_scene(
            object_id="supported",
            mesh_source=mesh_path,
            object_anchor_ply=object_anchor,
            support_anchor_ply=support_anchor,
            initial_scene_fit_receipt=initial_receipt,
            cameras_path=cameras_path,
            view_manifest_path=view_manifest,
            output_dir=tmp_path / "rejected",
            source_up_axis="+Y",
            world_up_axis="-Y",
        )
