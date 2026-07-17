from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from scripts.compose_layered_clean_plate import (
    compose,
    sha256_file,
    validate_limited_texture_acceptance,
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def asset(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def save_rgb(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value.astype(np.uint8)).save(path)


def save_rgba(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value.astype(np.uint8)).save(path)


def save_mask(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value.astype(np.uint8) * 255).save(path)


def save_depth(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value.astype(np.float32))


def render_receipt(
    path: Path,
    *,
    kind: str,
    role: str,
    rgba: Path,
    depth: Path,
    layer_id: str | None = None,
    claims_measured: bool = False,
) -> None:
    value: dict[str, Any] = {
        "kind": kind,
        "role": role,
        "provenance_class": role,
        "claims_measured_donor": claims_measured,
        "rgba_sha256": sha256_file(rgba),
        "depth_sha256": sha256_file(depth),
    }
    if layer_id is not None:
        value["layer_id"] = layer_id
    if role == "structural_background":
        value["texture_provenance"] = "synthetic_atlas"
    write_json(path, value)


def make_fixture(
    root: Path,
    *,
    round_index: int = 1,
    source_override: Path | None = None,
    previous_report: Path | None = None,
    previous_receipt: Path | None = None,
    leave_unresolved: bool = True,
) -> dict[str, Path]:
    height, width = 6, 8
    frame_id = "000000"
    removed_objects = ["front"] if round_index == 1 else ["front", "left"]
    source = np.full((height, width, 3), [80, 90, 100], dtype=np.uint8)
    source_path = source_override or root / "source.png"
    if source_override is None:
        save_rgb(source_path, source)
    removal = np.zeros((height, width), dtype=bool)
    removal[2:4, 2:7] = True
    measured = np.zeros_like(removal)
    measured[2, 2] = True
    residual = removal & ~measured
    removal_path = root / "removal.png"
    measured_path = root / "measured.png"
    residual_path = root / "residual.png"
    save_mask(removal_path, removal)
    save_mask(measured_path, measured)
    save_mask(residual_path, residual)

    donor = source.copy()
    donor[measured] = [230, 20, 20]
    donor_path = root / "donor.png"
    save_rgb(donor_path, donor)
    measured_depth = np.full((height, width), np.nan, dtype=np.float32)
    measured_depth[measured] = 1.5
    measured_depth_path = root / "measured_depth.npy"
    save_depth(measured_depth_path, measured_depth)

    layer1_rgba = np.zeros((height, width, 4), dtype=np.uint8)
    layer1_rgba[2, 3] = [20, 220, 20, 255]
    layer1_rgba[2, 4] = [20, 220, 20, 255]
    layer1_depth = np.full((height, width), np.nan, dtype=np.float32)
    layer1_depth[2, 3] = 3.0
    layer1_depth[2, 4] = 3.0
    layer1_rgba_path = root / "layer1.png"
    layer1_depth_path = root / "layer1.npy"
    layer1_receipt_path = root / "layer1_receipt.json"
    save_rgba(layer1_rgba_path, layer1_rgba)
    save_depth(layer1_depth_path, layer1_depth)
    render_receipt(
        layer1_receipt_path,
        kind="video2world.pbr_layer_render_receipt",
        role="downstream_object_render",
        layer_id="rear_pillow_a",
        rgba=layer1_rgba_path,
        depth=layer1_depth_path,
    )

    layer2_rgba = np.zeros((height, width, 4), dtype=np.uint8)
    layer2_rgba[2, 4] = [20, 20, 230, 255]
    layer2_rgba[2, 5] = [20, 20, 230, 255]
    layer2_depth = np.full((height, width), np.nan, dtype=np.float32)
    layer2_depth[2, 4] = 2.0
    layer2_depth[2, 5] = 6.0
    layer2_rgba_path = root / "layer2.png"
    layer2_depth_path = root / "layer2.npy"
    layer2_receipt_path = root / "layer2_receipt.json"
    save_rgba(layer2_rgba_path, layer2_rgba)
    save_depth(layer2_depth_path, layer2_depth)
    render_receipt(
        layer2_receipt_path,
        kind="video2world.pbr_layer_render_receipt",
        role="downstream_object_render",
        layer_id="rear_pillow_b",
        rgba=layer2_rgba_path,
        depth=layer2_depth_path,
    )

    background_rgba = np.zeros((height, width, 4), dtype=np.uint8)
    background_rgba[removal] = [220, 210, 40, 255]
    if leave_unresolved:
        background_rgba[3, 6, 3] = 0
    background_depth = np.full((height, width), np.nan, dtype=np.float32)
    background_depth[removal] = 5.0
    if leave_unresolved:
        background_depth[3, 6] = np.nan
    background_rgba_path = root / "background.png"
    background_depth_path = root / "background.npy"
    background_receipt_path = root / "background_receipt.json"
    save_rgba(background_rgba_path, background_rgba)
    save_depth(background_depth_path, background_depth)
    render_receipt(
        background_receipt_path,
        kind="video2world.structural_background_render_receipt",
        role="structural_background",
        rgba=background_rgba_path,
        depth=background_depth_path,
    )

    measured_report_path = root / "measured_report.json"
    write_json(
        measured_report_path,
        {
            "status": "technical_passed",
            "cumulative_removal_contract": {
                "enforced": True,
                "removed_object_ids": removed_objects,
                "donors_are_original_observed_rgb": True,
                "unresolved_residual_as_donor": False,
            },
            "gates": {
                "outside_removal_mask_rgb_exact": True,
                "all_residual_masks_subset_of_removal_masks": True,
            },
            "pixel_provenance": {
                "generated_pixels": 0,
                "propainter_pixels": 0,
                "unresolved_pixels_are_not_valid_donor_or_geometry_evidence": True,
            },
            "frame_records": [
                {
                    "frame_id": frame_id,
                    "prefill_frame": str(donor_path),
                    "prefill_frame_sha256": sha256_file(donor_path),
                    "removal_mask": str(removal_path),
                    "removal_mask_sha256": sha256_file(removal_path),
                    "residual_mask": str(residual_path),
                    "residual_mask_sha256": sha256_file(residual_path),
                    "measured_depth": str(measured_depth_path),
                    "measured_depth_sha256": sha256_file(measured_depth_path),
                    "measured_depth_role": "fused_multiview_measured_depth",
                }
            ],
        },
    )
    measured_receipt_path = root / "measured_receipt.json"
    write_json(
        measured_receipt_path,
        {
            "kind": "video2world.multiview_prefill_receipt",
            "report_sha256": sha256_file(measured_report_path),
            "frame_artifacts_are_hashed_in_report": True,
        },
    )

    manifest_path = root / "input_manifest.json"
    previous = None
    source_contract = {
        "role": "original_observed_rgb",
        "untracked_generated_rgb": False,
        "propainter_rgb": False,
    }
    if round_index > 1:
        assert previous_report is not None and previous_receipt is not None
        source_contract["role"] = "previous_layered_composite"
        previous = {
            "round_index": round_index - 1,
            "report": asset(previous_report),
            "receipt": asset(previous_receipt),
        }
    write_json(
        manifest_path,
        {
            "kind": "video2world.layered_clean_plate_compositor_input",
            "round_index": round_index,
            "round_kind": "object_peel",
            "removed_object_ids": removed_objects,
            "newly_removed_object_id": removed_objects[-1],
            "remaining_object_ids": ["rear_pillow_a", "rear_pillow_b"],
            "source_contract": source_contract,
            "previous_round": previous,
            "measured_donor_contract": {
                "role": "calibrated_rgbd_donor_prefill",
                "generated_pixels": 0,
                "propainter_pixels": 0,
                "report": asset(measured_report_path),
                "receipt": asset(measured_receipt_path),
            },
            "config": {"alpha_threshold": 0.5, "depth_tie_epsilon": 1e-6},
            "frame_records": [
                {
                    "sequence_index": 0,
                    "frame_id": frame_id,
                    "source_rgb": asset(source_path),
                    "cumulative_removal_mask": asset(removal_path),
                    "donor_prefill_rgb": asset(donor_path),
                    "donor_measured_mask": asset(measured_path),
                    "donor_measured_depth": asset(measured_depth_path),
                    "downstream_layers": [
                        {
                            "layer_id": "rear_pillow_a",
                            "depth_order": 0,
                            "role": "downstream_object_render",
                            "rgba": asset(layer1_rgba_path),
                            "depth": asset(layer1_depth_path),
                            "receipt": asset(layer1_receipt_path),
                        },
                        {
                            "layer_id": "rear_pillow_b",
                            "depth_order": 1,
                            "role": "downstream_object_render",
                            "rgba": asset(layer2_rgba_path),
                            "depth": asset(layer2_depth_path),
                            "receipt": asset(layer2_receipt_path),
                        },
                    ],
                    "structural_background": {
                        "role": "structural_background",
                        "rgba": asset(background_rgba_path),
                        "depth": asset(background_depth_path),
                        "receipt": asset(background_receipt_path),
                    },
                }
            ],
        },
    )
    return {
        "manifest": manifest_path,
        "measured_report": measured_report_path,
        "measured_receipt": measured_receipt_path,
        "source": source_path,
        "removal": removal_path,
        "measured_mask": measured_path,
        "measured_depth": measured_depth_path,
        "background_receipt": background_receipt_path,
        "layer1_receipt": layer1_receipt_path,
    }


def run_compose(manifest: Path, output: Path) -> dict[str, Any]:
    return compose(argparse.Namespace(input_manifest=manifest, output=output))


def configure_final_background_without_measured(
    fixture: dict[str, Path],
    *,
    accepted: bool = True,
) -> Path:
    manifest = json.loads(fixture["manifest"].read_text(encoding="utf-8"))
    record = manifest["frame_records"][0]
    measured_mask = np.zeros((6, 8), dtype=bool)
    measured_depth = np.full((6, 8), np.nan, dtype=np.float32)
    save_mask(fixture["measured_mask"], measured_mask)
    save_depth(fixture["measured_depth"], measured_depth)
    record["donor_measured_mask"] = asset(fixture["measured_mask"])
    record["donor_measured_depth"] = asset(fixture["measured_depth"])
    record.pop("donor_prefill_rgb")
    record["downstream_layers"] = []
    manifest["round_kind"] = "final_background"
    manifest["remaining_object_ids"] = []
    manifest["measured_donor_contract"] = {
        "role": "no_measured_donor_evidence",
        "claims_original_observed_rgb_donor": False,
        "measured_pixels": 0,
        "generated_pixels": 0,
        "propainter_pixels": 0,
    }
    background_receipt = json.loads(
        fixture["background_receipt"].read_text(encoding="utf-8")
    )
    acceptance_report = fixture["manifest"].parent / "accepted_texture_report.json"
    write_json(
        acceptance_report,
        {
            "status": "accepted_for_round04_clean_plate",
            "eligible_as_round04_clean_plate": True,
            "acceptance_gates": {
                "visual_quality_passed": True,
                "technical_quality_passed": True,
            },
        },
    )
    background_receipt["accepted_texture_report"] = asset(acceptance_report)
    background_receipt["accepted_texture_report_sha256"] = sha256_file(
        acceptance_report
    )
    if accepted:
        background_receipt["acceptance_status"] = "accepted"
        background_receipt["promotion_approved"] = True
    write_json(fixture["background_receipt"], background_receipt)
    record["structural_background"]["receipt"] = asset(fixture["background_receipt"])
    write_json(fixture["manifest"], manifest)
    return acceptance_report


def configure_limited_structural_acceptance(
    fixture: dict[str, Path],
    acceptance_report_path: Path,
    *,
    overridden_gates: list[str] | None = None,
) -> None:
    override_keys = overridden_gates or [
        "visual_quality",
        "wall_boundary_color_continuity",
        "wall_boundary_low_frequency_gradient_continuity",
    ]
    limitations = [
        "wall boundary p95 exceeds the strict threshold",
        "visible background completion artifacts remain",
    ]
    known_overrides: dict[str, Any] = {
        "visual_quality": "pending_human_or_vlm_review",
        "wall_boundary_color_continuity": {
            "threshold_p95_abs_rgb_delta": 24.0,
            "observed_maximum": 96.0,
            "passed": False,
        },
        "wall_boundary_low_frequency_gradient_continuity": {
            "threshold_p95_abs_rgb_delta": 32.0,
            "observed_maximum": 99.0,
            "passed": False,
        },
    }
    overrides = {
        key: known_overrides.get(key, {"passed": False}) for key in override_keys
    }
    acceptance_report = json.loads(acceptance_report_path.read_text(encoding="utf-8"))
    acceptance_report.update(
        {
            "status": "accepted_for_round04_clean_plate",
            "accepted_with_limitations": True,
            "promotion_approved": False,
            "demo_use_approved": True,
            "eligible_as_round04_clean_plate": False,
            "eligible_as_current_demo_round04_clean_plate": True,
            "acceptance_scope": "current_demo_only",
        }
    )
    acceptance_report.pop("acceptance_gates", None)
    acceptance_report["acceptance"] = {
        "kind": "video2world.planar_texture_acceptance",
        "accepted_with_limitations": True,
        "promotion_approved": False,
        "demo_use_approved": True,
        "eligible_as_round04_clean_plate": False,
        "eligible_as_current_demo_round04_clean_plate": True,
        "acceptance_scope": "current_demo_only",
        "review_limitations": limitations,
        "override_gate_keys": override_keys,
        "overridden_gates": overrides,
    }
    acceptance_report["overridden_gates"] = overrides
    acceptance_report["gates"] = {
        "all_synthetic_pixels_assigned_from_shared_atlas": True,
        "measured_atlas_texels_rgb_exact": True,
        "new_depth_normal_estimation_before_pgsr": "required_after_visual_acceptance",
        "observed_synthetic_anchor_and_interpolated_partition_rendered_synthetic_pixels": True,
        "outside_synthetic_mask_rgb_exact": True,
        "protected_neighbors_outside_synthetic_mask_rgb_exact": True,
        "protected_sam_neighbor_pixels_rgb_exact": True,
        "rejected_texture_inputs_not_promotable": True,
        "same_atlas_texel_has_identical_rgb_across_views": True,
        "synthetic_anchor_atlas_texels_rgb_exact": True,
        "synthetic_anchor_claims_measured_donor": False,
        "visual_quality": "accepted_with_limitations",
        "wall_boundary_color_continuity": known_overrides[
            "wall_boundary_color_continuity"
        ],
        "wall_boundary_low_frequency_gradient_continuity": known_overrides[
            "wall_boundary_low_frequency_gradient_continuity"
        ],
        "wall_object_boundary_colors_excluded_from_synthetic_fit": True,
    }
    write_json(acceptance_report_path, acceptance_report)
    background_receipt = json.loads(
        fixture["background_receipt"].read_text(encoding="utf-8")
    )
    background_receipt.update(
        {
            "acceptance_status": "accepted_with_limitations",
            "promotion_approved": False,
            "accepted_with_limitations": True,
            "demo_use_approved": True,
            "eligible_as_round04_clean_plate": False,
            "eligible_as_current_demo_round04_clean_plate": True,
            "acceptance_scope": "current_demo_only",
            "limitations": limitations,
            "override_gate_keys": override_keys,
            "overridden_gates": overrides,
            "failed_metrics": overrides,
            "accepted_texture_report": asset(acceptance_report_path),
            "accepted_texture_report_sha256": sha256_file(
                acceptance_report_path
            ),
        }
    )
    write_json(fixture["background_receipt"], background_receipt)
    manifest = json.loads(fixture["manifest"].read_text(encoding="utf-8"))
    manifest["frame_records"][0]["structural_background"]["receipt"] = asset(
        fixture["background_receipt"]
    )
    write_json(fixture["manifest"], manifest)


def test_compositor_z_buffers_layers_and_writes_exact_provenance_partition(
    tmp_path: Path,
) -> None:
    fixture = make_fixture(tmp_path)
    output = tmp_path / "output"

    report = run_compose(fixture["manifest"], output)

    record = report["frame_records"][0]
    composite = np.asarray(Image.open(output / record["outputs"]["composite_rgb"]["path"]))
    source = np.asarray(Image.open(fixture["source"]))
    removal = np.asarray(Image.open(fixture["removal"]).convert("L")) > 0
    labels = np.asarray(Image.open(output / record["outputs"]["provenance_labels"]["path"]))
    assert np.array_equal(composite[~removal], source[~removal])
    assert composite[2, 2].tolist() == [230, 20, 20]
    assert composite[2, 3].tolist() == [20, 220, 20]
    assert composite[2, 4].tolist() == [20, 20, 230]
    assert composite[2, 5].tolist() == [220, 210, 40]
    assert composite[3, 6].tolist() == source[3, 6].tolist()
    assert labels[2, 2] == 1
    assert labels[2, 3] == 2
    assert labels[2, 4] == 2
    assert labels[2, 5] == 3
    assert labels[3, 6] == 4
    assert record["counts"] == {
        "removal_pixels": 10,
        "measured_donor_pixels": 1,
        "downstream_object_render_pixels": 2,
        "structural_background_pixels": 6,
        "unresolved_pixels": 1,
        "outside_source_pixels": 38,
    }
    assert all(record["gates"].values())
    assert report["status"] == "technical_passed_with_unresolved"
    assert report["promotion_approved"] is False
    receipt = json.loads((output / "layered_composite_receipt.json").read_text())
    assert receipt["report_sha256"] == sha256_file(
        output / "layered_composite_report.json"
    )


def test_generated_or_propainter_pixels_cannot_claim_measured_provenance(
    tmp_path: Path,
) -> None:
    fixture = make_fixture(tmp_path)
    manifest = json.loads(fixture["manifest"].read_text(encoding="utf-8"))
    manifest["measured_donor_contract"]["generated_pixels"] = 1
    write_json(fixture["manifest"], manifest)

    with pytest.raises(ValueError, match="cannot claim measured"):
        run_compose(fixture["manifest"], tmp_path / "output")


def test_round_two_binds_previous_receipt_and_source_frame_hash(tmp_path: Path) -> None:
    round1 = make_fixture(tmp_path / "round1", leave_unresolved=False)
    round1_output = tmp_path / "round1_output"
    run_compose(round1["manifest"], round1_output)
    round1_report = round1_output / "layered_composite_report.json"
    round1_receipt = round1_output / "layered_composite_receipt.json"
    round1_source = round1_output / "frames" / "0000.png"
    round2 = make_fixture(
        tmp_path / "round2",
        round_index=2,
        source_override=round1_source,
        previous_report=round1_report,
        previous_receipt=round1_receipt,
    )
    round2_output = tmp_path / "round2_output"

    report = run_compose(round2["manifest"], round2_output)

    receipt = json.loads((round2_output / "layered_composite_receipt.json").read_text())
    assert report["previous_round_binding"]["report_sha256"] == sha256_file(round1_report)
    assert receipt["previous_round_output_report_sha256"] == sha256_file(round1_report)
    assert receipt["previous_round_output_receipt_sha256"] == sha256_file(round1_receipt)

    wrong_source = tmp_path / "wrong.png"
    save_rgb(wrong_source, np.zeros((6, 8, 3), dtype=np.uint8))
    manifest = json.loads(round2["manifest"].read_text(encoding="utf-8"))
    manifest["frame_records"][0]["source_rgb"] = asset(wrong_source)
    write_json(round2["manifest"], manifest)
    with pytest.raises(ValueError, match="not the previous round output"):
        run_compose(round2["manifest"], tmp_path / "round2_wrong_output")


def test_incomplete_previous_round_cannot_feed_next_peel_round(tmp_path: Path) -> None:
    round1 = make_fixture(tmp_path / "round1")
    round1_output = tmp_path / "round1_output"
    run_compose(round1["manifest"], round1_output)
    round2 = make_fixture(
        tmp_path / "round2",
        round_index=2,
        source_override=round1_output / "frames" / "0000.png",
        previous_report=round1_output / "layered_composite_report.json",
        previous_receipt=round1_output / "layered_composite_receipt.json",
    )

    with pytest.raises(ValueError, match="incomplete"):
        run_compose(round2["manifest"], tmp_path / "round2_output")


def test_removed_object_cannot_be_rendered_back_as_remaining_layer(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    manifest = json.loads(fixture["manifest"].read_text(encoding="utf-8"))
    manifest["remaining_object_ids"] = ["front", "rear_pillow_b"]
    write_json(fixture["manifest"], manifest)

    with pytest.raises(ValueError, match="cannot remain"):
        run_compose(fixture["manifest"], tmp_path / "output")


def test_missing_fused_measured_depth_blocks_composition(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    measured_report = json.loads(fixture["measured_report"].read_text(encoding="utf-8"))
    measured_report["frame_records"][0].pop("measured_depth")
    write_json(fixture["measured_report"], measured_report)
    measured_receipt = json.loads(fixture["measured_receipt"].read_text(encoding="utf-8"))
    measured_receipt["report_sha256"] = sha256_file(fixture["measured_report"])
    write_json(fixture["measured_receipt"], measured_receipt)
    manifest = json.loads(fixture["manifest"].read_text(encoding="utf-8"))
    manifest["measured_donor_contract"]["report"] = asset(fixture["measured_report"])
    manifest["measured_donor_contract"]["receipt"] = asset(fixture["measured_receipt"])
    write_json(fixture["manifest"], manifest)

    with pytest.raises(ValueError, match="absent from the measured donor report"):
        run_compose(fixture["manifest"], tmp_path / "output")


def test_final_background_can_explicitly_use_no_measured_donor_evidence(
    tmp_path: Path,
) -> None:
    round1 = make_fixture(tmp_path / "round1", leave_unresolved=False)
    round1_output = tmp_path / "round1_output"
    run_compose(round1["manifest"], round1_output)
    round2 = make_fixture(
        tmp_path / "round2",
        round_index=2,
        source_override=round1_output / "frames" / "0000.png",
        previous_report=round1_output / "layered_composite_report.json",
        previous_receipt=round1_output / "layered_composite_receipt.json",
        leave_unresolved=False,
    )
    configure_final_background_without_measured(round2)

    report = run_compose(round2["manifest"], tmp_path / "round2_output")

    assert report["round_kind"] == "final_background"
    assert report["measured_donor_report"] is None
    assert report["measured_donor_contract"] == {
        "role": "no_measured_donor_evidence",
        "claims_original_observed_rgb_donor": False,
        "measured_pixels": 0,
        "generated_pixels": 0,
        "propainter_pixels": 0,
    }
    assert report["aggregate_counts"]["measured_donor_pixels"] == 0
    assert report["aggregate_counts"]["downstream_object_render_pixels"] == 0
    assert report["aggregate_counts"]["structural_background_pixels"] == 10
    assert report["aggregate_counts"]["unresolved_pixels"] == 0
    assert report["gates"]["final_background_entire_removal_is_structural"] is True
    assert report["promotion_approved"] is False
    receipt = json.loads(
        (tmp_path / "round2_output" / "layered_composite_receipt.json").read_text()
    )
    assert receipt["round_kind"] == "final_background"
    assert receipt["claims_original_observed_rgb_donor"] is False


def test_object_peel_cannot_use_no_measured_donor_evidence(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    manifest = json.loads(fixture["manifest"].read_text(encoding="utf-8"))
    manifest["measured_donor_contract"] = {
        "role": "no_measured_donor_evidence",
        "claims_original_observed_rgb_donor": False,
        "measured_pixels": 0,
        "generated_pixels": 0,
        "propainter_pixels": 0,
    }
    write_json(fixture["manifest"], manifest)

    with pytest.raises(ValueError, match="allowed only for final_background"):
        run_compose(fixture["manifest"], tmp_path / "output")


def test_final_background_requires_no_remaining_objects_and_a_previous_round(
    tmp_path: Path,
) -> None:
    fixture = make_fixture(tmp_path)
    manifest = json.loads(fixture["manifest"].read_text(encoding="utf-8"))
    manifest["round_kind"] = "final_background"
    write_json(fixture["manifest"], manifest)
    with pytest.raises(ValueError, match="remaining_object_ids"):
        run_compose(fixture["manifest"], tmp_path / "remaining_output")

    manifest["remaining_object_ids"] = []
    manifest["frame_records"][0]["downstream_layers"] = []
    write_json(fixture["manifest"], manifest)
    with pytest.raises(ValueError, match="complete previous round"):
        run_compose(fixture["manifest"], tmp_path / "previous_output")


@pytest.mark.parametrize(
    ("invalid_asset", "message"),
    [
        ("mask", "empty measured mask"),
        ("depth", "all NaN"),
    ],
)
def test_final_background_no_measured_assets_must_be_explicitly_empty(
    tmp_path: Path,
    invalid_asset: str,
    message: str,
) -> None:
    round1 = make_fixture(tmp_path / "round1", leave_unresolved=False)
    round1_output = tmp_path / "round1_output"
    run_compose(round1["manifest"], round1_output)
    round2 = make_fixture(
        tmp_path / "round2",
        round_index=2,
        source_override=round1_output / "frames" / "0000.png",
        previous_report=round1_output / "layered_composite_report.json",
        previous_receipt=round1_output / "layered_composite_receipt.json",
        leave_unresolved=False,
    )
    configure_final_background_without_measured(round2)
    manifest = json.loads(round2["manifest"].read_text(encoding="utf-8"))
    if invalid_asset == "mask":
        value = np.zeros((6, 8), dtype=bool)
        value[2, 2] = True
        save_mask(round2["measured_mask"], value)
        manifest["frame_records"][0]["donor_measured_mask"] = asset(
            round2["measured_mask"]
        )
    else:
        value = np.full((6, 8), np.nan, dtype=np.float32)
        value[0, 0] = 1.0
        save_depth(round2["measured_depth"], value)
        manifest["frame_records"][0]["donor_measured_depth"] = asset(
            round2["measured_depth"]
        )
    write_json(round2["manifest"], manifest)

    with pytest.raises(ValueError, match=message):
        run_compose(round2["manifest"], tmp_path / "round2_output")


def test_final_background_requires_accepted_structural_texture_report(
    tmp_path: Path,
) -> None:
    round1 = make_fixture(tmp_path / "round1", leave_unresolved=False)
    round1_output = tmp_path / "round1_output"
    run_compose(round1["manifest"], round1_output)
    round2 = make_fixture(
        tmp_path / "round2",
        round_index=2,
        source_override=round1_output / "frames" / "0000.png",
        previous_report=round1_output / "layered_composite_report.json",
        previous_receipt=round1_output / "layered_composite_receipt.json",
        leave_unresolved=False,
    )
    configure_final_background_without_measured(round2, accepted=False)

    with pytest.raises(ValueError, match="receipt is not accepted"):
        run_compose(round2["manifest"], tmp_path / "round2_output")


def test_final_background_rejects_failed_texture_acceptance_gate(tmp_path: Path) -> None:
    round1 = make_fixture(tmp_path / "round1", leave_unresolved=False)
    round1_output = tmp_path / "round1_output"
    run_compose(round1["manifest"], round1_output)
    round2 = make_fixture(
        tmp_path / "round2",
        round_index=2,
        source_override=round1_output / "frames" / "0000.png",
        previous_report=round1_output / "layered_composite_report.json",
        previous_receipt=round1_output / "layered_composite_receipt.json",
        leave_unresolved=False,
    )
    acceptance_report_path = configure_final_background_without_measured(round2)
    acceptance_report = json.loads(acceptance_report_path.read_text(encoding="utf-8"))
    acceptance_report["acceptance_gates"]["visual_quality_passed"] = False
    write_json(acceptance_report_path, acceptance_report)
    background_receipt = json.loads(
        round2["background_receipt"].read_text(encoding="utf-8")
    )
    background_receipt["accepted_texture_report"] = asset(acceptance_report_path)
    background_receipt["accepted_texture_report_sha256"] = sha256_file(
        acceptance_report_path
    )
    write_json(round2["background_receipt"], background_receipt)
    manifest = json.loads(round2["manifest"].read_text(encoding="utf-8"))
    manifest["frame_records"][0]["structural_background"]["receipt"] = asset(
        round2["background_receipt"]
    )
    write_json(round2["manifest"], manifest)

    with pytest.raises(ValueError, match="visual quality did not pass"):
        run_compose(round2["manifest"], tmp_path / "round2_output")


def test_final_background_must_cover_entire_cumulative_removal(tmp_path: Path) -> None:
    round1 = make_fixture(tmp_path / "round1", leave_unresolved=False)
    round1_output = tmp_path / "round1_output"
    run_compose(round1["manifest"], round1_output)
    round2 = make_fixture(
        tmp_path / "round2",
        round_index=2,
        source_override=round1_output / "frames" / "0000.png",
        previous_report=round1_output / "layered_composite_report.json",
        previous_receipt=round1_output / "layered_composite_receipt.json",
        leave_unresolved=True,
    )
    configure_final_background_without_measured(round2)

    with pytest.raises(ValueError, match="cover the entire cumulative removal mask"):
        run_compose(round2["manifest"], tmp_path / "round2_output")


def test_limited_final_background_propagates_demo_only_acceptance(tmp_path: Path) -> None:
    round1 = make_fixture(tmp_path / "round1", leave_unresolved=False)
    round1_output = tmp_path / "round1_output"
    run_compose(round1["manifest"], round1_output)
    round2 = make_fixture(
        tmp_path / "round2",
        round_index=2,
        source_override=round1_output / "frames" / "0000.png",
        previous_report=round1_output / "layered_composite_report.json",
        previous_receipt=round1_output / "layered_composite_receipt.json",
        leave_unresolved=False,
    )
    acceptance_report = configure_final_background_without_measured(round2)
    configure_limited_structural_acceptance(round2, acceptance_report)

    report = run_compose(round2["manifest"], tmp_path / "round2_output")

    assert report["status"] == "technical_passed_complete_partition_with_limitations"
    assert report["accepted_with_limitations"] is True
    assert report["demo_use_approved"] is True
    assert report["eligible_as_round04_clean_plate"] is False
    assert report["eligible_as_current_demo_round04_clean_plate"] is True
    assert report["acceptance_scope"] == "current_demo_only"
    assert set(report["overridden_gates"]) == {
        "visual_quality",
        "wall_boundary_color_continuity",
        "wall_boundary_low_frequency_gradient_continuity",
    }
    assert report["failed_metrics"] == report["overridden_gates"]
    assert set(report["override_gate_keys"]) == set(report["overridden_gates"])
    assert report["promotion_approved"] is False
    assert report["gates"]["limited_acceptance_scope_and_failures_propagated"] is True
    frame_acceptance = report["frame_records"][0]["inputs"]["structural_background"][
        "acceptance"
    ]
    assert frame_acceptance["acceptance_status"] == "accepted_with_limitations"
    receipt = json.loads(
        (tmp_path / "round2_output" / "layered_composite_receipt.json").read_text()
    )
    assert receipt["accepted_with_limitations"] is True
    assert receipt["demo_use_approved"] is True
    assert receipt["eligible_as_round04_clean_plate"] is False
    assert receipt["eligible_as_current_demo_round04_clean_plate"] is True
    assert receipt["acceptance_status"] == "accepted_with_limitations"
    assert receipt["acceptance_scope"] == "current_demo_only"
    assert receipt["limitations"] == report["limitations"]
    assert receipt["failed_metrics"] == report["failed_metrics"]
    assert receipt["override_gate_keys"] == report["override_gate_keys"]
    assert receipt["promotion_approved"] is False


def test_real_limited_acceptance_report_matches_compositor_contract() -> None:
    path = (
        Path(__file__).parents[1]
        / "examples/bedroom4/completion/layered-peel/round04_bed/planar-background/"
        "accepted-demo-v23/accepted_planar_texture_report.json"
    )
    report = json.loads(path.read_text(encoding="utf-8"))

    acceptance = validate_limited_texture_acceptance(report)

    assert sha256_file(path) == "1081b0f804905e73184a20651a7f72c5113ad1736352fcdcc1a7004bfd596356"
    assert acceptance["acceptance_scope"] == "current_demo_only"
    assert acceptance["promotion_approved"] is False
    assert acceptance["failed_metrics"] == report["acceptance"]["overridden_gates"]
    assert acceptance["limitations"] == report["acceptance"]["review_limitations"]
    assert set(acceptance["overridden_gates"]) == {
        "visual_quality",
        "wall_boundary_color_continuity",
        "wall_boundary_low_frequency_gradient_continuity",
    }


def test_limited_acceptance_rejects_non_whitelisted_override(tmp_path: Path) -> None:
    round1 = make_fixture(tmp_path / "round1", leave_unresolved=False)
    round1_output = tmp_path / "round1_output"
    run_compose(round1["manifest"], round1_output)
    round2 = make_fixture(
        tmp_path / "round2",
        round_index=2,
        source_override=round1_output / "frames" / "0000.png",
        previous_report=round1_output / "layered_composite_report.json",
        previous_receipt=round1_output / "layered_composite_receipt.json",
        leave_unresolved=False,
    )
    acceptance_report = configure_final_background_without_measured(round2)
    configure_limited_structural_acceptance(
        round2,
        acceptance_report,
        overridden_gates=[
            "wall_boundary_color_continuity",
            "geometry_consistency",
            "visual_quality",
        ],
    )

    with pytest.raises(ValueError, match="exactly match the approved whitelist"):
        run_compose(round2["manifest"], tmp_path / "round2_output")


def test_limited_acceptance_rejects_receipt_propagation_mismatch(tmp_path: Path) -> None:
    round1 = make_fixture(tmp_path / "round1", leave_unresolved=False)
    round1_output = tmp_path / "round1_output"
    run_compose(round1["manifest"], round1_output)
    round2 = make_fixture(
        tmp_path / "round2",
        round_index=2,
        source_override=round1_output / "frames" / "0000.png",
        previous_report=round1_output / "layered_composite_report.json",
        previous_receipt=round1_output / "layered_composite_receipt.json",
        leave_unresolved=False,
    )
    acceptance_report = configure_final_background_without_measured(round2)
    configure_limited_structural_acceptance(round2, acceptance_report)
    background_receipt = json.loads(
        round2["background_receipt"].read_text(encoding="utf-8")
    )
    background_receipt["limitations"] = ["receipt omitted the real limitations"]
    write_json(round2["background_receipt"], background_receipt)
    manifest = json.loads(round2["manifest"].read_text(encoding="utf-8"))
    manifest["frame_records"][0]["structural_background"]["receipt"] = asset(
        round2["background_receipt"]
    )
    write_json(round2["manifest"], manifest)

    with pytest.raises(ValueError, match="does not propagate limitations"):
        run_compose(round2["manifest"], tmp_path / "round2_output")


def test_limited_acceptance_requires_all_other_technical_gates(tmp_path: Path) -> None:
    round1 = make_fixture(tmp_path / "round1", leave_unresolved=False)
    round1_output = tmp_path / "round1_output"
    run_compose(round1["manifest"], round1_output)
    round2 = make_fixture(
        tmp_path / "round2",
        round_index=2,
        source_override=round1_output / "frames" / "0000.png",
        previous_report=round1_output / "layered_composite_report.json",
        previous_receipt=round1_output / "layered_composite_receipt.json",
        leave_unresolved=False,
    )
    acceptance_report_path = configure_final_background_without_measured(round2)
    configure_limited_structural_acceptance(round2, acceptance_report_path)
    acceptance_report = json.loads(acceptance_report_path.read_text(encoding="utf-8"))
    acceptance_report["gates"]["same_atlas_texel_has_identical_rgb_across_views"] = False
    write_json(acceptance_report_path, acceptance_report)
    background_receipt = json.loads(
        round2["background_receipt"].read_text(encoding="utf-8")
    )
    background_receipt["accepted_texture_report"] = asset(acceptance_report_path)
    background_receipt["accepted_texture_report_sha256"] = sha256_file(
        acceptance_report_path
    )
    write_json(round2["background_receipt"], background_receipt)
    manifest = json.loads(round2["manifest"].read_text(encoding="utf-8"))
    manifest["frame_records"][0]["structural_background"]["receipt"] = asset(
        round2["background_receipt"]
    )
    write_json(round2["manifest"], manifest)

    with pytest.raises(ValueError, match="failed non-overridden technical gate"):
        run_compose(round2["manifest"], tmp_path / "round2_output")
