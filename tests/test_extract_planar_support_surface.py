from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from scripts.extract_planar_support_surface import extract_planar_support_surface


def _write_ply(path: Path, points: np.ndarray) -> None:
    vertex = np.empty(
        len(points),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    for index, name in enumerate(("x", "y", "z")):
        vertex[name] = points[:, index]
    vertex["red"] = 120
    vertex["green"] = 90
    vertex["blue"] = 60
    PlyData([PlyElement.describe(vertex, "vertex")], text=True).write(path)


def _synthetic_fixture(tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(12)
    x, z = np.meshgrid(np.linspace(-1.5, 1.5, 70), np.linspace(-1.0, 1.0, 50))
    tabletop = np.column_stack([x.ravel(), np.zeros(x.size), z.ravel()])
    tabletop[:, 1] = 1.0 + 0.04 * tabletop[:, 2]
    tabletop += rng.normal(0.0, 0.0015, tabletop.shape) * np.asarray([0.2, 1.0, 0.2])
    leg_count = 3_000
    legs = np.column_stack(
        [
            rng.choice([-1.35, 1.35], leg_count) + rng.normal(0.0, 0.03, leg_count),
            rng.uniform(1.05, 3.0, leg_count),
            rng.choice([-0.85, 0.85], leg_count) + rng.normal(0.0, 0.03, leg_count),
        ]
    )
    support = np.vstack([tabletop, legs])
    object_points = rng.normal(size=(800, 3)) * np.asarray([0.2, 0.15, 0.2])
    object_points += np.asarray([0.0, 0.45, 0.0])
    support_path = tmp_path / "support.ply"
    object_path = tmp_path / "object.ply"
    _write_ply(support_path, support)
    _write_ply(object_path, object_points)
    return support_path, object_path


def test_extracts_top_plane_and_preserves_a_deterministic_ply(tmp_path: Path) -> None:
    support_path, object_path = _synthetic_fixture(tmp_path)
    first_output = tmp_path / "first.ply"
    first_report = extract_planar_support_surface(
        support_anchor_ply=support_path,
        object_anchor_ply=object_path,
        output_ply=first_output,
        report_path=tmp_path / "first.json",
        candidate_top_fraction=0.55,
        ransac_trials=800,
        random_seed=9,
        inlier_distance=0.01,
        maximum_normal_tilt_degrees=8.0,
        refinement_iterations=8,
        connectivity_radius=0.08,
    )
    second_output = tmp_path / "second.ply"
    second_report = extract_planar_support_surface(
        support_anchor_ply=support_path,
        object_anchor_ply=object_path,
        output_ply=second_output,
        report_path=tmp_path / "second.json",
        candidate_top_fraction=0.55,
        ransac_trials=800,
        random_seed=9,
        inlier_distance=0.01,
        maximum_normal_tilt_degrees=8.0,
        refinement_iterations=8,
        connectivity_radius=0.08,
    )

    assert first_report["all_acceptance_gates_passed"] is True
    assert first_report["surface"]["point_count"] > 3_000
    assert first_report["surface"]["tilt_from_world_up_degrees"] < 3.0
    assert first_report["object_relation"]["center_projection_inside_support_hull"] is True
    assert first_report["object_relation"]["object_anchor_signed_height_quantiles"]["q005"] > 0
    assert (
        first_report["output"]["support_surface_ply"]["sha256"]
        == second_report["output"]["support_surface_ply"]["sha256"]
    )
    output_vertex = PlyData.read(first_output)["vertex"].data
    assert {"red", "green", "blue"}.issubset(output_vertex.dtype.names or ())


def test_rejects_a_changed_input_hash(tmp_path: Path) -> None:
    support_path, object_path = _synthetic_fixture(tmp_path)

    try:
        extract_planar_support_surface(
            support_anchor_ply=support_path,
            object_anchor_ply=object_path,
            output_ply=tmp_path / "output.ply",
            report_path=tmp_path / "report.json",
            expected_support_anchor_sha256="0" * 64,
        )
    except ValueError as error:
        assert "hash mismatch" in str(error)
    else:
        raise AssertionError("changed support input hash must be rejected")


@pytest.mark.parametrize("alias", ["report", "support_input", "object_input"])
def test_rejects_output_paths_that_alias_reports_or_inputs(tmp_path: Path, alias: str) -> None:
    support_path, object_path = _synthetic_fixture(tmp_path)
    output_path = tmp_path / "output.ply"
    report_path = tmp_path / "report.json"
    if alias == "report":
        report_path = output_path
    elif alias == "support_input":
        output_path = support_path
    else:
        report_path = object_path

    with pytest.raises(ValueError, match=r"paths must (?:be distinct|not overwrite inputs)"):
        extract_planar_support_surface(
            support_anchor_ply=support_path,
            object_anchor_ply=object_path,
            output_ply=output_path,
            report_path=report_path,
        )
