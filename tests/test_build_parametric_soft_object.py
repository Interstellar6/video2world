from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from plyfile import PlyData

from scripts.build_parametric_soft_object import (
    SH_C0,
    build_superellipsoid,
    connected_component_count,
    export_assets,
    sha256_file,
    write_gaussian_ply,
)


def write_dim_white_source(path: Path) -> None:
    rgba = np.zeros((48, 64, 4), dtype=np.uint8)
    rgba[4:44, 5:59, :3] = [145, 147, 150]
    rgba[18:29, 5:59, :3] = [118, 121, 125]
    rgba[4:44, 5:59, 3] = 255
    Image.fromarray(rgba).save(path)


@pytest.fixture
def completed_asset(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    source = tmp_path / "source.png"
    output = tmp_path / "asset"
    write_dim_white_source(source)
    report = export_assets(
        source_image=source,
        output_dir=output,
        object_id="test_pillow",
        appearance_mode="white_fabric",
        thickness_ratio=0.30,
        latitude_segments=16,
        longitude_segments=32,
        shape_exponent=0.34,
        gaussian_count=384,
        seed=20260717,
    )
    return output, report


def test_superellipsoid_is_closed_with_requested_dimensions() -> None:
    mesh = build_superellipsoid(
        width=1.4,
        height=1.0,
        depth=0.3,
        latitude_segments=16,
        longitude_segments=32,
        shape_exponent=0.34,
    )

    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    assert mesh.volume > 0.0
    assert connected_component_count(mesh) == 1
    np.testing.assert_allclose(mesh.extents, [1.4, 1.0, 0.3], atol=1e-12)


def test_export_is_white_on_all_sides_and_preserves_mesh_colors(
    completed_asset: tuple[Path, dict[str, object]],
) -> None:
    output, report = completed_asset
    mesh = report["mesh"]
    assert isinstance(mesh, dict)
    np.testing.assert_allclose(mesh["extents"], [1.35, 1.0, 0.3], atol=1e-12)
    assert mesh["observed_thickness_to_short_planar_ratio"] == pytest.approx(0.30)

    side_metrics = report["six_side_appearance"]
    assert isinstance(side_metrics, dict)
    assert set(side_metrics) == {"front", "back", "left", "right", "top", "bottom"}
    for metrics in side_metrics.values():
        assert metrics["count"] > 0
        assert metrics["median_luminance"] >= 190.0
        assert metrics["light_pixel_fraction"] >= 0.95
        assert min(metrics["mean_rgb"]) >= 180.0

    gates = report["acceptance_gates"]
    assert gates["white_fabric_is_white_on_all_sides"] is True
    assert gates["glb_vertex_colors_preserved"] is True
    assert gates["obj_vertex_colors_preserved"] is True
    assert report["all_acceptance_gates_passed"] is True

    material = np.asarray(Image.open(output / "material_reference.png"), dtype=np.float64)
    material_luminance = np.sum(material * np.asarray([0.2126, 0.7152, 0.0722]), axis=-1)
    assert float(np.median(material_luminance)) >= 190.0


def test_gaussian_ply_uses_standard_fields_and_a_deterministic_hash(
    completed_asset: tuple[Path, dict[str, object]],
    tmp_path: Path,
) -> None:
    output, report = completed_asset
    gaussian_path = output / "test_pillow_gaussian.ply"
    ply = PlyData.read(gaussian_path)
    vertices = ply["vertex"].data
    expected_names = (
        "x",
        "y",
        "z",
        "nx",
        "ny",
        "nz",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        *(f"f_rest_{index}" for index in range(45)),
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    )
    assert vertices.dtype.names == expected_names
    assert len(vertices) == 384
    for name in expected_names:
        assert np.isfinite(vertices[name]).all()
    for index in range(45):
        assert np.count_nonzero(vertices[f"f_rest_{index}"]) == 0

    reconstructed_rgb = np.column_stack(
        [vertices[f"f_dc_{channel}"] * SH_C0 + 0.5 for channel in range(3)]
    )
    assert float(np.median(reconstructed_rgb * 255.0)) >= 190.0
    rotations = np.column_stack([vertices[f"rot_{channel}"] for channel in range(4)])
    np.testing.assert_allclose(np.linalg.norm(rotations, axis=1), 1.0, atol=1e-5)
    assert np.all(vertices["scale_2"] < vertices["scale_0"])
    assert np.all(vertices["scale_2"] < vertices["scale_1"])
    assert np.allclose(vertices["opacity"], math.log(0.94 / 0.06))

    gaussian_report = report["gaussian"]
    assert gaussian_report["sha256"] == sha256_file(gaussian_path)
    repeated_path = tmp_path / "repeated.ply"
    mesh = build_superellipsoid(
        width=1.35,
        height=1.0,
        depth=0.3,
        latitude_segments=16,
        longitude_segments=32,
        shape_exponent=0.34,
    )
    texture = np.full((8, 8, 3), 215, dtype=np.uint8)
    from scripts.build_parametric_soft_object import color_mesh

    color_mesh(mesh, texture, appearance_mode="source")
    write_gaussian_ply(repeated_path, mesh, count=384, seed=20260717)
    first_repeat_hash = sha256_file(repeated_path)
    write_gaussian_ply(repeated_path, mesh, count=384, seed=20260717)
    assert sha256_file(repeated_path) == first_repeat_hash
