from __future__ import annotations

import numpy as np

from scripts.correct_pbr_region_material import (
    GLTF_AXIS_TO_BLENDER,
    component_passes_lower_cylinder,
    derive_neutral_metal_base_color,
    linear_to_srgb,
    srgb_to_linear,
)


def test_semantic_gltf_y_axes_map_to_blender_z_axes() -> None:
    np.testing.assert_array_equal(GLTF_AXIS_TO_BLENDER["+Y"], [0.0, 0.0, 1.0])
    np.testing.assert_array_equal(GLTF_AXIS_TO_BLENDER["-Y"], [0.0, 0.0, -1.0])


def test_real_mask_color_is_neutralized_without_increasing_luminance() -> None:
    samples = np.tile(np.asarray([[0.16, 0.10, 0.06]], dtype=np.float64), (128, 1))
    result = derive_neutral_metal_base_color(samples)

    expected_luma = 0.2126 * 0.16 + 0.7152 * 0.10 + 0.0722 * 0.06
    np.testing.assert_allclose(result["target_base_color_srgb"][:3], expected_luma)
    np.testing.assert_allclose(
        result["target_base_color_linear"][:3],
        srgb_to_linear(np.asarray([expected_luma]))[0],
    )
    assert result["target_base_color_srgb"][0] < 0.16


def test_srgb_linear_roundtrip() -> None:
    values = np.asarray([0.0, 0.01, 0.1, 0.5, 1.0])
    np.testing.assert_allclose(linear_to_srgb(srgb_to_linear(values)), values, atol=1e-12)


def test_color_target_clamps_only_outside_deep_gray_range() -> None:
    dark = np.tile(np.asarray([[0.001, 0.001, 0.001]]), (64, 1))
    bright = np.tile(np.asarray([[0.9, 0.9, 0.9]]), (64, 1))

    assert derive_neutral_metal_base_color(dark)["target_base_color_srgb"][0] == 0.06
    assert derive_neutral_metal_base_color(bright)["target_base_color_srgb"][0] == 0.18


def test_component_selector_requires_both_height_and_radius_gates() -> None:
    valid = {"maximum_height": -0.2, "maximum_radius": 0.15}
    too_tall = {"maximum_height": 0.1, "maximum_radius": 0.15}
    too_wide = {"maximum_height": -0.2, "maximum_radius": 0.3}

    assert component_passes_lower_cylinder(valid, maximum_height=-0.1, maximum_radius=0.2)
    assert not component_passes_lower_cylinder(too_tall, maximum_height=-0.1, maximum_radius=0.2)
    assert not component_passes_lower_cylinder(too_wide, maximum_height=-0.1, maximum_radius=0.2)
