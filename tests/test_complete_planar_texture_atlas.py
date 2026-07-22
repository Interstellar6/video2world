from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "complete_planar_texture_atlas.py"
sys.path.insert(0, str(SCRIPT_PATH.parent))
SPEC = importlib.util.spec_from_file_location("complete_planar_texture_atlas", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def make_synthetic_anchor_fixture(
    tmp_path: Path,
) -> tuple[SimpleNamespace, dict[str, object]]:
    frame_id = "000067"
    source_path = tmp_path / "source.png"
    candidate_path = tmp_path / "candidate.png"
    edit_mask_path = tmp_path / "edit_mask.png"
    removal_mask_path = tmp_path / "removal_mask.png"
    source = np.full((4, 4, 3), [120, 90, 50], dtype=np.uint8)
    candidate = source.copy()
    candidate[:2, :2] = [180, 145, 80]
    edit_mask = np.zeros((4, 4), dtype=np.uint8)
    edit_mask[:2, :2] = 255
    removal_mask = np.zeros((4, 4), dtype=np.uint8)
    removal_mask[:3, :3] = 255
    Image.fromarray(source).save(source_path)
    Image.fromarray(candidate).save(candidate_path)
    Image.fromarray(edit_mask).save(edit_mask_path)
    Image.fromarray(removal_mask).save(removal_mask_path)
    candidate_sha = MODULE.sha256_file(candidate_path)
    edit_mask_sha = MODULE.sha256_file(edit_mask_path)
    receipt_path = tmp_path / "diffusion_receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "kind": "video2world.diffusion_anchor_inpaint_run",
                "status": "generated_candidates_pending_review",
                "promotion_allowed": False,
                "inputs": {
                    "source_rgb": {"sha256": MODULE.sha256_file(source_path)},
                    "residual_mask": {
                        "sha256": edit_mask_sha,
                        "masked_pixels": 4,
                    },
                },
                "candidates": [
                    {
                        "artifact": {"sha256": candidate_sha},
                        "metrics": {
                            "outside_rgb_exact": True,
                            "outside_changed_pixels": 0,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        synthetic_anchor_frame_id=frame_id,
        synthetic_anchor_candidate=candidate_path,
        synthetic_anchor_candidate_sha256=candidate_sha,
        synthetic_anchor_edit_mask=edit_mask_path,
        synthetic_anchor_edit_mask_sha256=edit_mask_sha,
        synthetic_anchor_diffusion_receipt=receipt_path,
        synthetic_anchor_diffusion_receipt_sha256=MODULE.sha256_file(receipt_path),
    )
    geometry_report: dict[str, object] = {
        "frame_records": [
            {
                "frame_id": frame_id,
                "prefill_frame": str(source_path),
                "removal_mask": str(removal_mask_path),
            }
        ]
    }
    return args, geometry_report


def configure_synthetic_guide_fixture(
    tmp_path: Path,
    args: SimpleNamespace,
    geometry_report: dict[str, object],
) -> tuple[Path, Path]:
    source_path = Path(geometry_report["frame_records"][0]["prefill_frame"])
    edit_mask_path = Path(args.synthetic_anchor_edit_mask)
    guide_path = tmp_path / "synthetic_guide.png"
    guide = np.asarray(Image.open(source_path).convert("RGB"), dtype=np.uint8).copy()
    edit_mask = np.asarray(Image.open(edit_mask_path).convert("L"), dtype=np.uint8) > 0
    guide[edit_mask] = [155, 125, 70]
    Image.fromarray(guide).save(guide_path)
    planar_report_path = tmp_path / "planar_report.json"
    planar_report_path.write_text(json.dumps(geometry_report), encoding="utf-8")
    guide_report_path = tmp_path / "synthetic_guide_report.json"
    guide_sha = MODULE.sha256_file(guide_path)
    guide_report_path.write_text(
        json.dumps(
            {
                "status": "texture_candidate_review_pending",
                "promotion_approved": False,
                "eligible_as_round04_clean_plate": False,
                "planar_geometry_report_sha256": MODULE.sha256_file(planar_report_path),
                "provenance": {
                    "synthetic_pixels_are_measured": False,
                    "one_atlas_per_plane": True,
                    "per_frame_generation_used": False,
                },
                "gates": {
                    "measured_atlas_texels_rgb_exact": True,
                    "outside_synthetic_mask_rgb_exact": True,
                    "all_synthetic_pixels_assigned_from_shared_atlas": True,
                    "same_atlas_texel_has_identical_rgb_across_views": True,
                },
                "frame_records": [
                    {
                        "frame_id": "000067",
                        "completed_frame_sha256": guide_sha,
                        "synthetic_mask_sha256": args.synthetic_anchor_edit_mask_sha256,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    receipt = json.loads(Path(args.synthetic_anchor_diffusion_receipt).read_text())
    receipt["inputs"]["source_rgb"]["sha256"] = guide_sha
    Path(args.synthetic_anchor_diffusion_receipt).write_text(
        json.dumps(receipt),
        encoding="utf-8",
    )
    args.synthetic_anchor_diffusion_receipt_sha256 = MODULE.sha256_file(
        Path(args.synthetic_anchor_diffusion_receipt)
    )
    args.planar_report = planar_report_path
    args.synthetic_anchor_source_guide = guide_path
    args.synthetic_anchor_source_guide_sha256 = guide_sha
    args.synthetic_anchor_source_guide_report = guide_report_path
    args.synthetic_anchor_source_guide_report_sha256 = MODULE.sha256_file(
        guide_report_path
    )
    return guide_path, guide_report_path


def test_harmonic_extension_keeps_measured_texels_exact() -> None:
    color = np.zeros((9, 11, 3), dtype=np.uint8)
    observed = np.zeros((9, 11), dtype=bool)
    observed[:, 0] = True
    observed[:, -1] = True
    color[:, 0] = [120, 90, 50]
    color[:, -1] = [180, 130, 70]

    completed = MODULE.harmonic_extend(
        color,
        observed,
        iterations=80,
        relaxation=0.85,
    )

    assert np.array_equal(completed[observed], color[observed])
    assert np.all(completed[:, 5, 0] > 120)
    assert np.all(completed[:, 5, 0] < 180)


def test_synthetic_anchor_receipt_and_hashes_are_bound(tmp_path: Path) -> None:
    args, geometry_report = make_synthetic_anchor_fixture(tmp_path)

    anchor = MODULE.load_synthetic_anchor_inputs(args, geometry_report)

    assert anchor is not None
    assert anchor["frame_id"] == "000067"
    assert anchor["claims_measured_donor"] is False
    assert int(anchor["edit_mask"].sum()) == 4

    args.synthetic_anchor_candidate_sha256 = "0" * 64
    with pytest.raises(ValueError, match="candidate SHA-256 mismatch"):
        MODULE.load_synthetic_anchor_inputs(args, geometry_report)


def test_non_promotable_shared_atlas_guide_is_strictly_bound(tmp_path: Path) -> None:
    args, geometry_report = make_synthetic_anchor_fixture(tmp_path)
    configure_synthetic_guide_fixture(tmp_path, args, geometry_report)

    anchor = MODULE.load_synthetic_anchor_inputs(args, geometry_report)

    assert anchor is not None
    lineage = anchor["source_lineage"]
    assert lineage["role"] == "non_promotable_shared_atlas_synthetic_guide"
    assert lineage["outside_edit_mask_matches_planar_prefill"] is True
    assert lineage["candidate_outside_edit_mask_matches_planar_prefill"] is True
    assert lineage["claims_measured_donor"] is False


def test_synthetic_guide_must_match_planar_prefill_outside_edit_mask(
    tmp_path: Path,
) -> None:
    args, geometry_report = make_synthetic_anchor_fixture(tmp_path)
    guide_path, guide_report_path = configure_synthetic_guide_fixture(
        tmp_path,
        args,
        geometry_report,
    )
    guide = np.asarray(Image.open(guide_path).convert("RGB"), dtype=np.uint8).copy()
    candidate_path = Path(args.synthetic_anchor_candidate)
    candidate = np.asarray(Image.open(candidate_path).convert("RGB"), dtype=np.uint8).copy()
    guide[3, 3] = [1, 2, 3]
    candidate[3, 3] = [1, 2, 3]
    Image.fromarray(guide).save(guide_path)
    Image.fromarray(candidate).save(candidate_path)
    guide_sha = MODULE.sha256_file(guide_path)
    candidate_sha = MODULE.sha256_file(candidate_path)
    guide_report = json.loads(guide_report_path.read_text())
    guide_report["frame_records"][0]["completed_frame_sha256"] = guide_sha
    guide_report_path.write_text(json.dumps(guide_report), encoding="utf-8")
    receipt_path = Path(args.synthetic_anchor_diffusion_receipt)
    receipt = json.loads(receipt_path.read_text())
    receipt["inputs"]["source_rgb"]["sha256"] = guide_sha
    receipt["candidates"][0]["artifact"]["sha256"] = candidate_sha
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    args.synthetic_anchor_source_guide_sha256 = guide_sha
    args.synthetic_anchor_source_guide_report_sha256 = MODULE.sha256_file(
        guide_report_path
    )
    args.synthetic_anchor_candidate_sha256 = candidate_sha
    args.synthetic_anchor_diffusion_receipt_sha256 = MODULE.sha256_file(receipt_path)

    with pytest.raises(ValueError, match="guide differs from planar prefill"):
        MODULE.load_synthetic_anchor_inputs(args, geometry_report)


def test_synthetic_guide_report_must_remain_non_promotable(tmp_path: Path) -> None:
    args, geometry_report = make_synthetic_anchor_fixture(tmp_path)
    _, guide_report_path = configure_synthetic_guide_fixture(
        tmp_path,
        args,
        geometry_report,
    )
    guide_report = json.loads(guide_report_path.read_text())
    guide_report["promotion_approved"] = True
    guide_report_path.write_text(json.dumps(guide_report), encoding="utf-8")
    args.synthetic_anchor_source_guide_report_sha256 = MODULE.sha256_file(
        guide_report_path
    )

    with pytest.raises(ValueError, match="must remain non-promotable"):
        MODULE.load_synthetic_anchor_inputs(args, geometry_report)


def test_synthetic_anchor_edit_mask_must_be_binary(tmp_path: Path) -> None:
    path = tmp_path / "mask.png"
    mask = np.zeros((3, 3), dtype=np.uint8)
    mask[1, 1] = 128
    Image.fromarray(mask).save(path)

    with pytest.raises(ValueError, match="must contain only 0 and 255"):
        MODULE.load_binary_mask(path, "test mask")


def test_plane_target_pixel_counts_sums_frame_usage() -> None:
    counts = MODULE.plane_target_pixel_counts(
        {
            "frame_records": [
                {"plane_pixel_counts": {"1": 4, "2": 0}},
                {"plane_pixel_counts": {"1": 6, "3": 5}},
                {"plane_pixel_counts": {"1": True, "2": 2}},
                {"plane_pixel_counts": {"1": "ignored", "2": 3}},
            ]
        },
        [1, 2, 3, 4],
    )

    assert counts == {1: 10, 2: 5, 3: 5, 4: 0}


def test_measured_atlas_texels_take_priority_over_synthetic_anchor() -> None:
    measured = np.full((2, 3, 3), [40, 50, 60], dtype=np.uint8)
    observed = np.zeros((2, 3), dtype=bool)
    observed[0, 0] = True
    projected = {
        "color": np.full((2, 3, 3), [200, 180, 160], dtype=np.uint8),
        "support": np.zeros((2, 3), dtype=bool),
    }
    projected["support"][0, :2] = True

    combined, measured_category, anchor_category, interpolated = (
        MODULE.compose_atlas_provenance(measured, observed, projected)
    )

    assert np.array_equal(combined[0, 0], measured[0, 0])
    assert np.array_equal(combined[0, 1], projected["color"][0, 1])
    assert measured_category[0, 0]
    assert not anchor_category[0, 0]
    assert anchor_category[0, 1]
    assert int(measured_category.sum() + anchor_category.sum() + interpolated.sum()) == 6


def test_rectangular_footprint_core_covers_irregular_object_shape() -> None:
    footprint = np.zeros((13, 17), dtype=bool)
    footprint[5, 6] = True
    footprint[7, 10] = True
    footprint[6, 8] = True

    weight, details = MODULE.rectangular_footprint_feather_weight(
        footprint,
        padding=4,
        feather_width=2,
    )

    core = weight >= 1.0 - 1e-7
    assert np.all(core[footprint])
    assert np.all(core[3:10, 4:13])
    assert details["footprint_bounds_inclusive"] == [6, 5, 10, 7]
    assert details["core_bounds_inclusive"] == [4, 3, 12, 9]
    assert details["boundary_shape"] == "axis_aligned_plane_uv_rectangle"
    assert np.any((weight > 0.0) & (weight < 1.0))


def test_rectangular_footprint_padding_must_cover_feather_width() -> None:
    footprint = np.zeros((5, 5), dtype=bool)
    footprint[2, 2] = True

    with pytest.raises(ValueError, match="padding must be at least"):
        MODULE.rectangular_footprint_feather_weight(
            footprint,
            padding=2,
            feather_width=3,
        )


def test_footprint_neutralization_skips_planes_without_completed_atlas() -> None:
    footprints = {
        1: np.ones((2, 2), dtype=bool),
        2: np.zeros((3, 3), dtype=bool),
    }
    completed_atlases = {
        1: {
            "color": np.zeros((2, 2, 3), dtype=np.uint8),
            "footprint_weight": np.ones((2, 2), dtype=np.float32),
        }
    }
    atlas_records_by_id = {1: {"plane_id": 1}}

    eligible, skipped = MODULE.filter_footprints_for_completed_atlases(
        footprints,
        completed_atlases,
        atlas_records_by_id,
    )

    assert set(eligible) == {1}
    assert skipped == [
        {
            "plane_id": 2,
            "reason": "no_completed_atlas_for_footprint",
            "footprint_texels": 0,
            "rendering_required": False,
        }
    ]


def test_plane_footprint_compositor_preserves_protected_and_outside_pixels() -> None:
    source = np.full((3, 5, 3), [20, 30, 40], dtype=np.uint8)
    rendered = np.full((3, 5, 3), [220, 180, 120], dtype=np.uint8)
    weight = np.zeros((3, 5), dtype=np.float32)
    weight[1, 1] = 1.0
    weight[1, 2] = 0.5
    weight[1, 3] = 1.0
    original_removal = np.zeros((3, 5), dtype=bool)
    original_removal[1, 1] = True
    protected = np.zeros((3, 5), dtype=bool)
    protected[1, 1] = True
    protected[1, 3] = True

    result, provenance, details = MODULE.composite_plane_footprint_render(
        source,
        rendered,
        weight,
        original_removal,
        protected,
    )

    assert np.array_equal(result[1, 1], rendered[1, 1])
    assert np.array_equal(result[1, 3], source[1, 3])
    assert np.array_equal(result[0], source[0])
    assert result[1, 2].tolist() == [120, 105, 80]
    assert provenance["core_shared_atlas"][1, 1]
    assert provenance["screen_space_feather"][1, 2]
    assert provenance["outside_source"][1, 3]
    assert details["protected_neighbor_overlap_with_original_removal_not_overridden"] == 1
    assert details["outside_outer_mask_rgb_exact"] is True
    assert details["protected_neighbor_pixels_rgb_exact"] is True


def test_plane_footprint_compositor_rejects_removal_outside_weight_one_core() -> None:
    source = np.zeros((2, 2, 3), dtype=np.uint8)
    rendered = np.full((2, 2, 3), 255, dtype=np.uint8)
    weight = np.zeros((2, 2), dtype=np.float32)
    weight[0, 0] = 0.5
    removal = np.zeros((2, 2), dtype=bool)
    removal[0, 0] = True

    with pytest.raises(ValueError, match="escaped the weight-one core"):
        MODULE.composite_plane_footprint_render(
            source,
            rendered,
            weight,
            removal,
            np.zeros_like(removal),
        )


def test_synthetic_anchor_projects_only_valid_edit_pixels_to_shared_atlas(
    tmp_path: Path,
) -> None:
    args, geometry_report = make_synthetic_anchor_fixture(tmp_path)
    plane_ids_path = tmp_path / "plane_ids.png"
    depth_path = tmp_path / "geometry_depth.npz"
    Image.fromarray(np.ones((4, 4), dtype=np.uint16)).save(plane_ids_path)
    np.savez_compressed(depth_path, depth=np.ones((4, 4), dtype=np.float32))
    frame_record = geometry_report["frame_records"][0]
    frame_record["plane_ids"] = str(plane_ids_path)
    frame_record["geometry_depth"] = str(depth_path)
    geometry_report["texture_atlases"] = [
        {
            "plane_id": 1,
            "dimensions": [4, 4],
            "origin_uv": [0.0, 0.0],
            "texel_size_world_units": 1.0,
        }
    ]
    camera_info = {
        "intrinsic": {
            "fx": 1.0,
            "fy": 1.0,
            "cx": 0.0,
            "cy": 0.0,
            "w": 4,
            "h": 4,
        },
        "extrinsic": {"000067": np.eye(4).tolist()},
    }
    plane_records = {
        1: {
            "plane_id": 1,
            "basis_u": [1.0, 0.0, 0.0],
            "basis_v": [0.0, 1.0, 0.0],
        }
    }
    anchor = MODULE.load_synthetic_anchor_inputs(args, geometry_report)

    atlases, record = MODULE.project_synthetic_anchor_to_atlases(
        anchor,
        geometry_report,
        camera_info,
        plane_records,
    )

    assert record is not None
    assert record["eligible_geometry_pixels"] == 4
    assert record["all_eligible_pixels_inside_declared_atlases"] is True
    assert int(atlases[1]["support"].sum()) == 4
    assert np.all(atlases[1]["color"][atlases[1]["support"]] == [180, 145, 80])


def test_floor_extension_uses_low_gradient_texture_direction() -> None:
    color = np.zeros((8, 12, 3), dtype=np.uint8)
    observed = np.zeros((8, 12), dtype=bool)
    for row in range(8):
        color[row, :3] = [80 + row * 12, 45 + row * 5, 20]
        color[row, -3:] = [82 + row * 12, 47 + row * 5, 22]
    observed[:, :3] = True
    observed[:, -3:] = True

    completed, details = MODULE.directional_floor_extend(
        color,
        observed,
        fallback_iterations=0,
    )

    assert details["selected_direction"] == "horizontal_uv"
    assert np.array_equal(completed[observed], color[observed])
    assert np.all(completed[:, 5].sum(axis=1) > 0)
    assert np.ptp(completed[:, 5, 0]) > 50


def test_wall_support_excludes_bright_window_and_dark_furniture() -> None:
    color = np.zeros((10, 12, 3), dtype=np.uint8)
    observed = np.ones((10, 12), dtype=bool)
    color[:] = [145, 112, 58]
    color[:2, :3] = [245, 245, 240]
    color[-2:, -3:] = [25, 18, 10]

    support, details = MODULE.robust_low_frequency_support(
        color,
        observed,
        quantization=24,
        color_radius=72.0,
        minimum_fraction=0.08,
        luminance_quantile=0.65,
    )

    assert support[5, 5]
    assert not support[0, 0]
    assert not support[-1, -1]
    assert details["fallback_to_all_measured"] is False


def test_screened_harmonic_keeps_observed_boundary_and_biases_interior() -> None:
    color = np.zeros((9, 13, 3), dtype=np.uint8)
    observed = np.zeros((9, 13), dtype=bool)
    observed[:, 0] = True
    observed[:, -1] = True
    color[:, 0] = [20, 15, 10]
    color[:, -1] = [20, 15, 10]
    prior = np.full_like(color, [150, 110, 55])

    completed = MODULE.screened_harmonic_extend(
        color,
        observed,
        prior,
        iterations=120,
        relaxation=0.85,
        screen_weight=0.1,
    )

    assert np.array_equal(completed[observed], color[observed])
    assert completed[4, 6, 0] > completed[4, 1, 0]
    assert completed[4, 6, 0] > 80


def test_excluded_sdxl_review_is_recorded_as_non_input(tmp_path: Path) -> None:
    path = tmp_path / "visual_review.json"
    path.write_text(
        json.dumps(
            {
                "status": "rejected",
                "promotion_allowed": False,
                "blocking_findings": ["regenerated the removed object"],
                "next_use": "none",
            }
        ),
        encoding="utf-8",
    )

    records = MODULE.excluded_evidence([path])

    assert records[0]["status"] == "rejected"
    assert records[0]["role"] == "excluded_texture_input_evidence_only"
    assert records[0]["next_use"] == "none"


def test_plane_illumination_ignores_object_shaped_boundary_colors() -> None:
    height, width = 44, 58
    row, column = np.indices((height, width))
    u = (column + 0.5) / width * 2.0 - 1.0
    v = (row + 0.5) / height * 2.0 - 1.0
    truth = np.stack(
        [
            150.0 + 22.0 * u - 8.0 * v + 6.0 * u * v,
            112.0 + 13.0 * u - 5.0 * v + 4.0 * u * u,
            58.0 + 7.0 * u - 3.0 * v,
        ],
        axis=2,
    )
    color = np.clip(np.rint(truth), 0, 255).astype(np.uint8)
    synthetic = (
        (row >= 12)
        & (row <= 37)
        & (column >= 15)
        & (column <= 44)
        & (row >= 8 + np.abs(column - 30) // 3)
    )
    observed = ~synthetic
    contaminated_boundary = observed & MODULE.geometry.binary_dilate(synthetic, 2)
    color[contaminated_boundary] = [55, 35, 18]
    support, inset = MODULE.inset_support_from_synthetic_boundary(
        observed,
        observed,
        3,
        minimum_fraction=0.05,
    )

    completed, details = MODULE.robust_plane_illumination_field(
        color,
        observed,
        support,
        degree=2,
        iterations=8,
        huber_delta=1.5,
        ridge=1e-4,
    )

    prediction_error = np.abs(completed[synthetic].astype(np.float64) - truth[synthetic])
    assert inset["fallback_to_non_inset_support"] is False
    assert not np.any(support & contaminated_boundary)
    assert prediction_error.mean() < 2.0
    assert details["method"] == "robust_plane_uv_polynomial_illumination_field"
    assert np.array_equal(completed[observed], color[observed])


def test_boundary_normal_gradient_metric_accepts_continuous_low_frequency_field() -> None:
    row, column = np.indices((18, 24))
    field = np.stack(
        [80 + column * 2 + row, 60 + column + row, 40 + row],
        axis=2,
    ).astype(np.uint8)
    observed = np.ones((18, 24), dtype=bool)
    observed[5:14, 7:19] = False

    continuity = MODULE.boundary_normal_gradient_continuity(field, observed)

    assert continuity["boundary_edge_count"] > 0
    assert continuity["boundary_normal_gradient_pair_count"] > 0
    assert continuity["boundary_normal_gradient_p95_abs_rgb_delta"] == 0.0


def test_normalized_illumination_is_bounded_by_reliable_wall_colors() -> None:
    height, width = 30, 42
    row, column = np.indices((height, width))
    color = np.stack(
        [
            145 + column * 0.8,
            105 + row * 0.5,
            52 + column * 0.2,
        ],
        axis=2,
    ).astype(np.uint8)
    observed = np.ones((height, width), dtype=bool)
    observed[7:26, 10:34] = False
    support, _ = MODULE.inset_support_from_synthetic_boundary(
        observed,
        observed,
        2,
        minimum_fraction=0.05,
    )

    completed, model, details = MODULE.normalized_plane_illumination_field(
        color,
        observed,
        support,
        sigma=8.0,
        blend_weight_scale=0.02,
        color_clip_margin=0.0,
        fallback_degree=1,
        fallback_iterations=6,
        huber_delta=1.5,
        ridge=1e-4,
    )

    synthetic = ~observed
    reliable_colors = color[support]
    assert completed[synthetic].min(axis=0).min() >= reliable_colors.min(axis=0).min()
    assert completed[synthetic].max(axis=0).max() <= reliable_colors.max(axis=0).max()
    assert details["method"].startswith("robust_plane_uv_normalized_convolution")
    assert details["atlas_fraction_with_kernel_support"] > 0.99
    assert np.array_equal(completed[observed], color[observed])
    assert not np.array_equal(model[observed], color[observed])


def test_reliable_boundary_residual_collar_reduces_seam_without_changing_interior() -> None:
    height, width = 32, 46
    row, column = np.indices((height, width))
    truth = np.stack(
        [120 + column, 90 + row, 55 + column * 0.2],
        axis=2,
    ).astype(np.uint8)
    observed = np.ones((height, width), dtype=bool)
    observed[7:27, 11:37] = False
    model = np.clip(truth.astype(np.int16) - [18, 12, 8], 0, 255).astype(np.uint8)
    baseline = model.copy()
    baseline[observed] = truth[observed]

    completed, details = MODULE.apply_reliable_boundary_residual_collar(
        model,
        truth,
        observed,
        observed,
        width=6,
    )
    before = MODULE.boundary_normal_gradient_continuity(baseline, observed)
    after = MODULE.boundary_normal_gradient_continuity(completed, observed)

    assert after["boundary_color_p95_abs_rgb_delta"] < before[
        "boundary_color_p95_abs_rgb_delta"
    ]
    assert details["reliable_boundary_seed_texels"] > 0
    assert details["maximum_propagated_distance_texels"] <= 6
    assert np.array_equal(completed[16, 24], model[16, 24])
    assert np.array_equal(completed[observed], truth[observed])
