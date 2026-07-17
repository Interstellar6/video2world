from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "refine_scene_fit_silhouette.py"
SPEC = importlib.util.spec_from_file_location("refine_scene_fit_silhouette", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_refinement_linear_preserves_thickness_axis() -> None:
    axes = MODULE.rotation_about_axis(np.asarray([0.3, -0.2, 0.9]), 23.0)
    normal = axes[:, 2]
    linear = MODULE.refinement_linear(axes, 2, 1.2, 0.85, -31.0)

    assert np.allclose(linear @ normal, normal)
    assert MODULE.determinant_3x3(linear) == pytest.approx(1.2 * 0.85)


def test_tangent_translation_has_no_thickness_component() -> None:
    axes = MODULE.rotation_about_axis(np.asarray([0.1, 0.8, 0.4]), -17.0)
    tangent = axes[:, [0, 1]]
    coefficients = np.asarray([0.12, -0.07])
    translation = tangent @ coefficients

    assert np.dot(translation, axes[:, 2]) == pytest.approx(0.0, abs=1e-12)
    assert np.linalg.norm(translation) == pytest.approx(np.linalg.norm(coefficients))


def test_cluster_mesh_keeps_usable_surface() -> None:
    mesh = trimesh.creation.icosphere(subdivisions=3)
    clustered = MODULE.cluster_mesh(mesh, 10)

    assert 4 <= len(clustered.vertices) < len(mesh.vertices)
    assert 2 <= len(clustered.faces) < len(mesh.faces)
    assert np.isfinite(clustered.vertices).all()


def test_search_bounds_must_contain_identity() -> None:
    mesh = trimesh.creation.box()
    observed = np.zeros((64, 64), dtype=bool)
    observed[20:40, 20:40] = True
    camera = {
        "position": np.asarray([0.0, 0.0, -5.0]),
        "rotation": np.eye(3),
        "width": 64,
        "height": 64,
        "fx": 50.0,
        "fy": 50.0,
    }

    with pytest.raises(ValueError, match="contain 1"):
        MODULE.search_refinement(
            mesh=mesh,
            axes=np.eye(3),
            thickness_axis=2,
            base_pivot=np.zeros(3),
            observed=observed,
            camera=camera,
            minimum_scale=1.1,
            maximum_scale=1.5,
        )


def test_support_review_detects_front_plane_overshoot() -> None:
    report = {
        "target_obb": {
            "center": [0.0, 0.0, 0.0],
            "extents": [2.0, 2.0, 1.0],
            "axes_columns": np.eye(3).tolist(),
        },
        "placement_evidence": {"thickness_axis": 2, "completion_direction_sign": 1.0},
    }
    observed = np.zeros((32, 32), dtype=bool)
    observed[8:24, 8:24] = True
    rendered = observed.copy()
    source = np.asarray([[-0.5, -0.5, 0.0], [0.5, 0.5, 1.0]])
    fitted = source.copy()
    fitted[0, 2] = -0.2

    review = MODULE.support_and_envelope_review(
        world_vertices=fitted,
        source_world_vertices=source,
        report=report,
        observed=observed,
        rendered=rendered,
        gravity_direction=np.asarray([0.0, 1.0, 0.0]),
        maximum_support_drift=0.1,
        maximum_anchor_overflow_ratio=0.1,
        maximum_support_edge_error_px=2.0,
    )

    assert review["gates"]["no_additional_front_plane_overshoot"] is False
    assert review["status"] == "rejected"


def test_occlusion_aware_metrics_ignore_only_pixel_iou_domain() -> None:
    observed = np.zeros((16, 16), dtype=bool)
    observed[4:12, 4:12] = True
    rendered = observed.copy()
    rendered[6:10, 12:14] = True
    ignored = np.zeros_like(observed)
    ignored[6:10, 12:14] = True

    raw = MODULE.silhouette_metrics(observed, rendered)
    clipped = MODULE.silhouette_metrics(observed, rendered, ignored)

    assert clipped["mask_iou"] == pytest.approx(1.0)
    assert clipped["bbox_iou"] == raw["bbox_iou"]
    assert clipped["rendered_pixels_inside_ignored_region"] == 8
    assert clipped["pixel_evaluation_domain"] == "outside_explicit_occluder_union"


def test_camera_plane_refinement_preserves_camera_depth() -> None:
    camera_axes = MODULE.rotation_about_axis(np.asarray([0.2, 0.4, 0.8]), 19.0)
    linear = MODULE.refinement_linear(camera_axes, 2, 1.3, 0.8, -27.0)
    points = np.asarray([[0.2, 0.3, 0.5], [-0.4, 0.8, -0.7], [1.0, -0.2, 0.1]])
    transformed = np.einsum("ni,ji->nj", points, linear)

    assert np.allclose(points @ camera_axes[:, 2], transformed @ camera_axes[:, 2])


def test_support_plane_interior_normal_is_negated_for_downward_gravity(tmp_path: Path) -> None:
    report = tmp_path / "planar.json"
    report.write_text(
        json.dumps(
            {
                "support_plane": {
                    "plane_id": 4,
                    "semantic_role": "floor",
                    "interior_normal": [0.0, -1.0, 0.0],
                    "offset": -7.0,
                }
            }
        ),
        encoding="utf-8",
    )

    gravity, evidence = MODULE.load_support_plane_evidence(report)

    assert np.allclose(gravity, [0.0, 1.0, 0.0])
    assert evidence["interior_normal"] == [0.0, -1.0, 0.0]
    assert evidence["derived_downward_gravity_direction"] == [-0.0, 1.0, -0.0]
    assert "negation" in evidence["sign_semantics"]
