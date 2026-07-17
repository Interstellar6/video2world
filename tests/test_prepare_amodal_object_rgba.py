from __future__ import annotations

import numpy as np
from scipy import ndimage

from scripts.prepare_amodal_object_rgba import (
    CompletionConfig,
    complete_rgba,
    infer_added_region,
    parser,
    resolve_shape_policy,
)


def infer(
    mask: np.ndarray,
    *,
    category: str = "bed",
    requested_policy: str = "fill_enclosed_holes",
) -> tuple[np.ndarray, dict]:
    return infer_added_region(
        mask,
        category=category,
        requested_policy=requested_policy,
        corner_radius_ratio=0.10,
        min_candidate_add_ratio=0.24,
        min_occlusion_depth_ratio=0.16,
        min_occlusion_depth_px=8,
        min_component_area_ratio=0.002,
        cavity_bridge_radius=5,
    )


def bed_mask(*, exterior_notch: bool = False) -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros((50, 64), dtype=bool)
    mask[5:45, 6:58] = True
    hole = np.zeros_like(mask)
    if exterior_notch:
        hole[5:29, 22:42] = True
    else:
        hole[17:29, 24:40] = True
    mask[hole] = False
    return mask, hole


def test_fill_enclosed_holes_accepts_only_internal_component() -> None:
    mask, hole = bed_mask()

    added, trace = infer(mask)

    assert trace["resolved_shape_policy"] == "fill_enclosed_holes"
    assert np.array_equal(added, hole)
    assert trace["accepted_hole_pixels"] == int(np.count_nonzero(hole))
    assert len(trace["accepted_enclosed_holes"]) == 1
    accepted = trace["accepted_enclosed_holes"][0]
    assert accepted["bbox_xyxy"] == [24, 17, 39, 28]
    assert accepted["boundary_connectivity"] == "enclosed"
    assert accepted["touches_crop_boundary"] is False
    exterior = next(
        component
        for component in trace["background_components"]
        if component["touches_crop_boundary"]
    )
    assert exterior["disposition"] == "rejected_connected_to_crop_boundary"


def test_fill_enclosed_holes_rejects_exterior_notch() -> None:
    mask, _ = bed_mask(exterior_notch=True)

    added, trace = infer(mask)

    assert not np.any(added)
    assert trace["completion_applied"] is False
    assert trace["completion_reason"] == "no_enclosed_alpha_holes"
    assert trace["accepted_enclosed_holes"] == []
    assert all(
        component["boundary_connectivity"] == "connected_to_crop_boundary"
        for component in trace["background_components"]
    )


def test_fill_enclosed_holes_bridges_narrow_mouth_but_not_bridge_pixels() -> None:
    mask = np.zeros((60, 72), dtype=bool)
    mask[5:55, 6:66] = True
    chamber = np.zeros_like(mask)
    chamber[21:43, 20:52] = True
    narrow_mouth = np.zeros_like(mask)
    narrow_mouth[5:22, 34:38] = True
    mask[chamber | narrow_mouth] = False

    added, trace = infer(mask)
    closed_for_topology = ndimage.binary_closing(
        mask,
        structure=np.ones((11, 11), dtype=np.uint8),
    )
    bridge_pixels = closed_for_topology & ~mask

    assert np.count_nonzero(added) > 0
    assert np.count_nonzero(added & chamber) == np.count_nonzero(added)
    assert not np.any(added & bridge_pixels)
    assert trace["cavity_bridge_radius_pixels"] == 5
    assert trace["cavity_bridge_kernel_shape"] == [11, 11]
    assert trace["bridge_is_topology_only"] is True
    assert trace["exterior_bridge_pixels_added_to_output"] == 0
    assert trace["completion_reason"] == "narrow_mouth_occlusion_cavities_completed"
    assert len(trace["accepted_bridged_cavities"]) == 1
    cavity = trace["accepted_bridged_cavities"][0]
    assert cavity["boundary_connectivity"] == "enclosed"
    assert cavity["source_was_connected_to_crop_boundary"] is True
    assert cavity["disposition"] == "accepted_narrow_mouth_cavity"


def test_fill_enclosed_holes_preserves_visible_rgba_bytes() -> None:
    mask, hole = bed_mask()
    rows, columns = np.indices(mask.shape)
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[..., 0] = (31 + columns * 3) % 255
    rgba[..., 1] = (47 + rows * 5) % 255
    rgba[..., 2] = (rows + columns * 2) % 255
    rgba[..., 3][mask] = 211
    visible_before = rgba[mask].copy()
    config = CompletionConfig(
        object_id="synthetic_bed",
        category="bed",
        attempt=1,
        shape_policy="fill_enclosed_holes",
        alpha_threshold=127,
        corner_radius_ratio=0.10,
        min_candidate_add_ratio=0.24,
        min_occlusion_depth_ratio=0.16,
        min_occlusion_depth_px=8,
        min_component_area_ratio=0.002,
        cavity_bridge_radius=5,
        seed=20260717,
        provider="test",
        retry_prompt="test",
        scene_reference_manifest="test",
        scene_reference_manifest_sha256="0" * 64,
    )

    result = complete_rgba(rgba, config)

    assert np.array_equal(result.added_mask, hole)
    assert np.array_equal(result.completed_rgba[mask], visible_before)
    assert result.color_metrics["original_visible_pixels_byte_identical"] is True
    assert result.color_metrics["added_pixels_source_only_original_foreground"] is True
    assert result.gates["original_visible_pixels_preserved"] is True


def test_fill_enclosed_holes_rejects_tiny_alpha_noise() -> None:
    mask = np.zeros((40, 48), dtype=bool)
    mask[4:36, 4:44] = True
    mask[19:21, 23:25] = False

    added, trace = infer(mask)

    assert not np.any(added)
    enclosed = next(
        component
        for component in trace["background_components"]
        if not component["touches_crop_boundary"]
    )
    assert enclosed["area_pixels"] == 4
    assert enclosed["passes_minimum_area"] is False
    assert enclosed["disposition"] == "rejected_below_minimum_area"
    assert trace["completion_reason"] == "enclosed_alpha_holes_below_minimum_area"


def test_auto_preserves_multi_depth_bed_categories_without_changing_existing_profiles() -> None:
    for category in ("bed", "bed_frame", "bed-frame", "headboard"):
        policy, reason = resolve_shape_policy(category, "auto")
        assert policy == "preserve"
        assert reason == "known_multi_depth_structure_requires_depth_aware_amodal"

    assert resolve_shape_policy("pillow", "auto") == (
        "soft_rectangle",
        "known_soft_object_category_profile",
    )
    assert resolve_shape_policy("plant", "auto") == (
        "preserve",
        "unknown_category_fail_closed",
    )

    assert parser().parse_args([]).cavity_bridge_radius == 5


def test_auto_does_not_turn_a_bed_occlusion_cavity_into_solid_geometry() -> None:
    mask, _ = bed_mask()

    added, trace = infer(mask, requested_policy="auto")

    assert not np.any(added)
    assert trace["resolved_shape_policy"] == "preserve"
    assert trace["policy_reason"] == (
        "known_multi_depth_structure_requires_depth_aware_amodal"
    )
    assert trace["completion_reason"] == "category_has_no_safe_amodal_shape_prior"
