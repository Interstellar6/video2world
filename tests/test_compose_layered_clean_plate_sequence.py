from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from scripts.compose_layered_clean_plate_sequence import (
    BACKGROUND_RECEIPT_KIND,
    CUMULATIVE_MANIFEST_KIND,
    OBJECT_ORDER,
    PBR_BATCH_RECEIPT_KIND,
    PBR_INDEX_KIND,
    PBR_RECEIPT_KIND,
    R4_MASK_INDEX_KIND,
    ROUND_NAMES,
    ROUND_REMAINING,
    SEQUENCE_RECEIPT_KIND,
    STRUCTURAL_BATCH_RECEIPT_KIND,
    STRUCTURAL_INDEX_KIND,
    run_sequence,
    sha256_file,
)

FRAME_IDS = ("000000", "000001")
HEIGHT = 6
WIDTH = 8


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def asset(path: Path, *, relative_to: Path | None = None) -> dict[str, str]:
    return {
        "path": (
            str(path)
            if relative_to is None
            else str(path.relative_to(relative_to))
        ),
        "sha256": sha256_file(path),
    }


def save_rgb(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((HEIGHT, WIDTH, 3), color, dtype=np.uint8)).save(path)


def save_rgba(path: Path, color: tuple[int, int, int]) -> None:
    value = np.full((HEIGHT, WIDTH, 4), (*color, 255), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value).save(path)


def save_mask(path: Path, width: int) -> np.ndarray:
    value = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    value[1:5, 1:width] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value).save(path)
    return value == 255


def save_depth(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value.astype(np.float32))


def measured_lineage(previous: dict[str, Path] | None, round_index: int) -> dict[str, Any]:
    return {
        "previous_round_index": round_index - 1 if previous else None,
        "previous_cumulative_manifest": str(previous["manifest"]) if previous else None,
        "previous_cumulative_manifest_sha256": (
            sha256_file(previous["manifest"]) if previous else None
        ),
        "previous_prefill_report": str(previous["report"]) if previous else None,
        "previous_prefill_report_sha256": (
            sha256_file(previous["report"]) if previous else None
        ),
        "previous_prefill_status": "technical_passed" if previous else None,
    }


def make_measured_round(
    root: Path,
    *,
    round_index: int,
    sources: dict[str, Path],
    previous: dict[str, Path] | None,
) -> dict[str, Path]:
    round_root = root / f"measured_round{round_index}"
    lineage = measured_lineage(previous, round_index)
    frame_records = []
    masks: dict[str, Path] = {}
    for sequence_index, frame_id in enumerate(FRAME_IDS):
        mask_path = round_root / "masks" / f"{sequence_index:04d}.png"
        save_mask(mask_path, 2 + round_index)
        masks[frame_id] = mask_path
        frame_records.append(
            {
                "sequence_index": sequence_index,
                "frame_id": frame_id,
                "source_frame": str(sources[frame_id]),
                "source_frame_sha256": sha256_file(sources[frame_id]),
                "source_role": (
                    "original_observed_rgb"
                    if round_index == 1
                    else "immediately_previous_measured_prefill_with_unresolved_mask"
                ),
                "union_mask": str(mask_path.relative_to(round_root)),
                "union_mask_sha256": sha256_file(mask_path),
                "union_mask_pixels": int(load_mask(mask_path).sum()),
                "removed_object_ids": list(OBJECT_ORDER[:round_index]),
            }
        )
    manifest_path = round_root / "cumulative_removal_manifest.json"
    manifest = {
        "schema_version": 1,
        "kind": CUMULATIVE_MANIFEST_KIND,
        "round_index": round_index,
        "removed_object_ids": list(OBJECT_ORDER[:round_index]),
        "frame_count": len(FRAME_IDS),
        "lineage": lineage,
        "gates": {
            "strict_front_to_back_order": True,
            "input_is_immediately_previous_round": True,
            "all_previous_masks_are_accumulated": True,
            "all_removed_objects_are_excluded_from_donors": True,
            "donors_are_original_observed_rgb": True,
            "unresolved_residual_is_never_donor_evidence": True,
        },
        "frame_records": frame_records,
    }
    write_json(manifest_path, manifest)

    report_records = []
    for sequence_index, frame_id in enumerate(FRAME_IDS):
        removal_path = masks[frame_id]
        residual_path = round_root / "residual" / f"{sequence_index:04d}.png"
        residual = load_mask(removal_path)
        residual_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(residual.astype(np.uint8) * 255).save(residual_path)
        depth_path = round_root / "depth" / f"{sequence_index:04d}.npy"
        save_depth(depth_path, np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32))
        report_records.append(
            {
                "sequence_index": sequence_index,
                "frame_id": frame_id,
                "source_frame": str(sources[frame_id]),
                "source_frame_sha256": sha256_file(sources[frame_id]),
                "removal_mask": str(removal_path),
                "removal_mask_sha256": sha256_file(removal_path),
                "prefill_frame": str(sources[frame_id]),
                "prefill_frame_sha256": sha256_file(sources[frame_id]),
                "residual_mask": str(residual_path),
                "residual_mask_sha256": sha256_file(residual_path),
                "residual_mask_pixels": int(residual.sum()),
                "measured_depth": str(depth_path),
                "measured_depth_sha256": sha256_file(depth_path),
                "measured_depth_role": "fused_multiview_measured_depth",
                "measured_depth_valid_pixels": 0,
            }
        )
    report_path = round_root / "multiview_prefill_report.json"
    report = {
        "schema_version": 2,
        "status": "technical_passed",
        "promotion_blocker": "semantic texture continuity review is required",
        "next_action": {
            "action": "run_semantic_texture_and_new_depth_normal_review_before_next_round",
            "blocker": "semantic_texture_and_depth_normal_pending",
            "status": "technical_passed",
            "no_support_frame_ids": [],
            "unresolved_unobserved_pixels": sum(
                record["residual_mask_pixels"] for record in report_records
            ),
            "promotion_approved": False,
        },
        "input_manifest": str(manifest_path),
        "input_manifest_sha256": sha256_file(manifest_path),
        "cumulative_removal_contract": {
            "enforced": True,
            "round_index": round_index,
            "removed_object_ids": list(OBJECT_ORDER[:round_index]),
            "manifest_sha256": sha256_file(manifest_path),
            "lineage": lineage,
            "donors_are_original_observed_rgb": True,
            "all_removed_objects_excluded_from_donors": True,
            "unresolved_residual_as_donor": False,
        },
        "pixel_provenance": {
            "measured_multiview_pixels": 0,
            "generated_pixels": 0,
            "propainter_pixels": 0,
            "unresolved_pixels_are_not_valid_donor_or_geometry_evidence": True,
        },
        "gates": {
            "outside_removal_mask_rgb_exact": True,
            "all_residual_masks_subset_of_removal_masks": True,
            "cumulative_front_to_back_contract_enforced": True,
            "fused_measured_metric_depth_materialized": True,
        },
        "frame_records": report_records,
    }
    write_json(report_path, report)
    receipt_path = round_root / "multiview_prefill_receipt.json"
    write_json(
        receipt_path,
        {
            "schema_version": 2,
            "kind": "video2world.multiview_prefill_receipt",
            "report": report_path.name,
            "report_sha256": sha256_file(report_path),
            "frame_count": len(FRAME_IDS),
            "frame_artifacts_are_hashed_in_report": True,
            "fused_metric_depth_artifacts_are_hashed_in_report": True,
        },
    )
    return {"manifest": manifest_path, "report": report_path, "receipt": receipt_path}


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 0


def make_pbr_batch(root: Path, sources: dict[str, Path]) -> dict[str, Path]:
    batch_root = root / "pbr"
    frames: dict[str, Any] = {}
    for frame_index, frame_id in enumerate(FRAME_IDS):
        layers = {}
        for layer_index, layer_id in enumerate(OBJECT_ORDER):
            layer_root = batch_root / "layers" / layer_id
            rgba_path = layer_root / "rgba" / f"{frame_id}.png"
            depth_path = layer_root / "depth" / f"{frame_id}.npy"
            receipt_path = layer_root / "receipts" / f"{frame_id}.json"
            save_rgba(
                rgba_path,
                (40 + layer_index * 40, 80 + frame_index * 10, 120),
            )
            save_depth(
                depth_path,
                np.full((HEIGHT, WIDTH), 5.0 + layer_index, dtype=np.float32),
            )
            write_json(
                receipt_path,
                {
                    "kind": PBR_RECEIPT_KIND,
                    "status": "technical_passed",
                    "role": "downstream_object_render",
                    "provenance_class": "downstream_object_render",
                    "claims_measured_donor": False,
                    "layer_id": layer_id,
                    "frame_id": frame_id,
                    "rgba_sha256": sha256_file(rgba_path),
                    "depth_sha256": sha256_file(depth_path),
                    "gates": {"synthetic_fixture_passed": True},
                },
            )
            layers[layer_id] = {
                "rgba": asset(rgba_path, relative_to=batch_root),
                "depth": asset(depth_path, relative_to=batch_root),
                "receipt": asset(receipt_path, relative_to=batch_root),
            }
        frames[frame_id] = {
            "source_frame": asset(sources[frame_id]),
            "layers": layers,
        }
    index_path = batch_root / "render_index.json"
    write_json(
        index_path,
        {
            "kind": PBR_INDEX_KIND,
            "status": "technical_passed_100_frame_asset_set_not_promoted",
            "frame_ids": list(FRAME_IDS),
            "layer_order": list(OBJECT_ORDER),
            "round_remaining_object_ids": ROUND_REMAINING,
            "contract": {
                "alpha_depth_alignment_mode": "geometry-derived-binary-matte",
                "final_alpha_equals_finite_depth_support": True,
                "claims_measured_donor": False,
            },
            "frames": frames,
        },
    )
    receipt_path = batch_root / "batch_receipt.json"
    write_json(
        receipt_path,
        {
            "kind": PBR_BATCH_RECEIPT_KIND,
            "render_index": asset(index_path, relative_to=batch_root),
            "all_required_evidence_is_hash_bound": True,
        },
    )
    return {"root": batch_root, "index": index_path, "receipt": receipt_path}


def limited_acceptance() -> tuple[dict[str, Any], dict[str, Any]]:
    limitations = [
        "wall boundary p95 exceeds the strict threshold",
        "visible background completion artifacts remain",
    ]
    overrides: dict[str, Any] = {
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
    keys = list(overrides)
    report = {
        "status": "accepted_for_round04_clean_plate",
        "accepted_with_limitations": True,
        "promotion_approved": False,
        "demo_use_approved": True,
        "eligible_as_round04_clean_plate": False,
        "eligible_as_current_demo_round04_clean_plate": True,
        "acceptance_scope": "current_demo_only",
        "overridden_gates": overrides,
        "acceptance": {
            "kind": "video2world.planar_texture_acceptance",
            "accepted_with_limitations": True,
            "promotion_approved": False,
            "demo_use_approved": True,
            "eligible_as_round04_clean_plate": False,
            "eligible_as_current_demo_round04_clean_plate": True,
            "acceptance_scope": "current_demo_only",
            "review_limitations": limitations,
            "override_gate_keys": keys,
            "overridden_gates": overrides,
        },
        "gates": {
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
            "wall_boundary_color_continuity": overrides[
                "wall_boundary_color_continuity"
            ],
            "wall_boundary_low_frequency_gradient_continuity": overrides[
                "wall_boundary_low_frequency_gradient_continuity"
            ],
            "wall_object_boundary_colors_excluded_from_synthetic_fit": True,
        },
    }
    receipt_fields = {
        "acceptance_status": "accepted_with_limitations",
        "promotion_approved": False,
        "accepted_with_limitations": True,
        "demo_use_approved": True,
        "eligible_as_round04_clean_plate": False,
        "eligible_as_current_demo_round04_clean_plate": True,
        "acceptance_scope": "current_demo_only",
        "limitations": limitations,
        "override_gate_keys": keys,
        "overridden_gates": overrides,
        "failed_metrics": overrides,
    }
    return report, receipt_fields


def make_structural_batch(
    root: Path,
    *,
    limited: bool,
    materialized_schema: bool = False,
) -> dict[str, Any]:
    batch_root = root / "structural"
    frames = {}
    frame_records = []
    source_masks = {}
    for frame_index, frame_id in enumerate(FRAME_IDS):
        rgba_path = batch_root / "rgba" / f"{frame_id}.png"
        depth_path = batch_root / "depth" / f"{frame_id}.npy"
        receipt_path = batch_root / "receipts" / f"{frame_id}.json"
        acceptance_path = batch_root / "acceptance" / f"{frame_id}.json"
        source_mask_path = batch_root / "source_masks" / f"{frame_index:04d}.png"
        source_mask = save_mask(source_mask_path, 6)
        source_masks[frame_id] = source_mask_path
        if materialized_schema:
            rgba = np.full(
                (HEIGHT, WIDTH, 4),
                (180, 170 + frame_index * 5, 150, 0),
                dtype=np.uint8,
            )
            rgba[..., 3] = source_mask.astype(np.uint8) * 255
            rgba_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgba).save(rgba_path)
            depth = np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32)
            depth[source_mask] = 20.0
            save_depth(depth_path, depth)
        else:
            save_rgba(rgba_path, (180, 170 + frame_index * 5, 150))
            save_depth(depth_path, np.full((HEIGHT, WIDTH), 20.0, dtype=np.float32))
        if limited:
            acceptance_report, receipt_fields = limited_acceptance()
        else:
            acceptance_report = {
                "status": "accepted_for_round04_clean_plate",
                "eligible_as_round04_clean_plate": True,
                "acceptance_gates": {
                    "visual_quality_passed": True,
                    "technical_quality_passed": True,
                },
            }
            receipt_fields = {
                "acceptance_status": "accepted",
                "promotion_approved": True,
            }
        write_json(acceptance_path, acceptance_report)
        write_json(
            receipt_path,
            {
                "kind": BACKGROUND_RECEIPT_KIND,
                "frame_id": frame_id,
                "sequence_index": frame_index,
                "role": "structural_background",
                "provenance_class": "structural_background",
                "claims_measured_donor": False,
                "texture_provenance": "synthetic_atlas",
                "rgba_sha256": sha256_file(rgba_path),
                "depth_sha256": sha256_file(depth_path),
                "accepted_texture_report": asset(acceptance_path),
                "accepted_texture_report_sha256": sha256_file(acceptance_path),
                **receipt_fields,
            },
        )
        frames[frame_id] = {
            "rgba": asset(rgba_path, relative_to=batch_root),
            "depth": asset(depth_path, relative_to=batch_root),
            "receipt": asset(receipt_path, relative_to=batch_root),
        }
        frame_records.append(
            {
                "sequence_index": frame_index,
                "frame_id": frame_id,
                **frames[frame_id],
                "source_cumulative_removal_mask": {
                    **asset(source_mask_path),
                    "pixels": int(source_mask.sum()),
                },
                "alpha_pixels": int(source_mask.sum()),
                "gates": {
                    "alpha_covers_cumulative_removal_mask": True,
                    "alpha_equals_positive_finite_geometry_depth": True,
                },
            }
        )
    index_path = batch_root / "render_index.json"
    if materialized_schema:
        index_value = {
            "kind": STRUCTURAL_INDEX_KIND,
            "status": "materialized_for_current_demo_with_limitations",
            "frame_count": len(FRAME_IDS),
            "frame_records": frame_records,
        }
    else:
        index_value = {
            "kind": STRUCTURAL_INDEX_KIND,
            "status": "technical_passed_2_frame_fixture",
            "frame_ids": list(FRAME_IDS),
            "frames": frames,
        }
    write_json(
        index_path,
        index_value,
    )
    receipt_path = batch_root / "batch_receipt.json"
    write_json(
        receipt_path,
        (
            {
                "kind": STRUCTURAL_BATCH_RECEIPT_KIND,
                "index": index_path.name,
                "index_sha256": sha256_file(index_path),
                "frame_count": len(FRAME_IDS),
            }
            if materialized_schema
            else {
                "kind": STRUCTURAL_BATCH_RECEIPT_KIND,
                "render_index": asset(index_path, relative_to=batch_root),
            }
        ),
    )
    return {
        "index": index_path,
        "receipt": receipt_path,
        "source_masks": source_masks,
    }


def make_r4_masks(root: Path) -> Path:
    mask_root = root / "r4_masks"
    frames = {}
    for sequence_index, frame_id in enumerate(FRAME_IDS):
        path = mask_root / "masks" / f"{sequence_index:04d}.png"
        save_mask(path, 6)
        frames[frame_id] = {"mask": asset(path, relative_to=mask_root)}
    index_path = mask_root / "index.json"
    write_json(
        index_path,
        {
            "kind": R4_MASK_INDEX_KIND,
            "round_index": 4,
            "removed_object_ids": list(OBJECT_ORDER),
            "frame_ids": list(FRAME_IDS),
            "frames": frames,
        },
    )
    return index_path


def make_fixture(tmp_path: Path, *, limited: bool = False) -> dict[str, Any]:
    source_root = tmp_path / "source"
    sources = {}
    for index, frame_id in enumerate(FRAME_IDS):
        path = source_root / f"{frame_id}.png"
        save_rgb(path, (70 + index * 5, 80, 90))
        sources[frame_id] = path
    measured = []
    previous = None
    for round_index in range(1, 4):
        current = make_measured_round(
            tmp_path,
            round_index=round_index,
            sources=sources,
            previous=previous,
        )
        measured.append(current)
        previous = current
    pbr = make_pbr_batch(tmp_path, sources)
    structural = make_structural_batch(tmp_path, limited=limited)
    r4 = make_r4_masks(tmp_path)
    args = argparse.Namespace(
        round1_cumulative_manifest=measured[0]["manifest"],
        round1_measured_report=measured[0]["report"],
        round1_measured_receipt=measured[0]["receipt"],
        round2_cumulative_manifest=measured[1]["manifest"],
        round2_measured_report=measured[1]["report"],
        round2_measured_receipt=measured[1]["receipt"],
        round3_cumulative_manifest=measured[2]["manifest"],
        round3_measured_report=measured[2]["report"],
        round3_measured_receipt=measured[2]["receipt"],
        pbr_render_index=pbr["index"],
        pbr_batch_receipt=pbr["receipt"],
        structural_background_index=structural["index"],
        structural_background_receipt=structural["receipt"],
        round4_cumulative_masks=r4,
        path_rebase=[],
        output=tmp_path / "sequence_output",
    )
    return {
        "args": args,
        "measured": measured,
        "pbr": pbr,
        "structural": structural,
        "r4": r4,
    }


def resign_index(index_path: Path, receipt_path: Path) -> None:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["render_index"] = asset(index_path, relative_to=receipt_path.parent)
    write_json(receipt_path, receipt)


def test_sequence_executes_four_rounds_and_rebases_remote_asset(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    pbr_index = json.loads(fixture["pbr"]["index"].read_text(encoding="utf-8"))
    rgba_record = pbr_index["frames"]["000000"]["layers"][OBJECT_ORDER[1]]["rgba"]
    local_path = fixture["pbr"]["root"] / rgba_record["path"]
    relative = local_path.relative_to(fixture["pbr"]["root"])
    rgba_record["path"] = str(Path("/remote/pbr") / relative)
    write_json(fixture["pbr"]["index"], pbr_index)
    resign_index(fixture["pbr"]["index"], fixture["pbr"]["receipt"])
    fixture["args"].path_rebase = [f"/remote/pbr={fixture['pbr']['root']}"]

    report = run_sequence(fixture["args"], expected_frame_count=2)

    assert report["status"] == "technical_passed_complete_sequence"
    assert [item["round_name"] for item in report["rounds"]] == list(ROUND_NAMES)
    assert [item["unresolved_pixels"] for item in report["rounds"]] == [0, 0, 0, 0]
    assert all(report["gates"].values())
    measured_inputs = report["inputs"]["measured_rounds"]
    assert (
        measured_inputs[0]["measured_action_summary"]["next_action"]["blocker"]
        == "semantic_texture_and_depth_normal_pending"
    )
    assert (
        measured_inputs[0]["measured_action_summary"]["promotion_blocker"]
        == "semantic texture continuity review is required"
    )
    for round_index, item in enumerate(report["rounds"], start=1):
        assert item["removed_object_ids"] == list(OBJECT_ORDER[:round_index])
        assert item["remaining_object_ids"] == ROUND_REMAINING[ROUND_NAMES[round_index - 1]]
    round1_report = json.loads(
        (
            fixture["args"].output
            / ROUND_NAMES[0]
            / "composite"
            / "layered_composite_report.json"
        ).read_text()
    )
    round2_manifest_path = (
        fixture["args"].output / ROUND_NAMES[1] / "compositor_input_manifest.json"
    )
    round2_manifest = json.loads(round2_manifest_path.read_text())
    round1_output = round1_report["frame_records"][0]["outputs"]["composite_rgb"]
    round1_frame = fixture["args"].output / ROUND_NAMES[0] / "composite" / round1_output["path"]
    assert round2_manifest["frame_records"][0]["source_rgb"]["sha256"] == sha256_file(
        round1_frame
    )
    round4_manifest = json.loads(
        (fixture["args"].output / ROUND_NAMES[3] / "compositor_input_manifest.json").read_text()
    )
    assert round4_manifest["round_kind"] == "final_background"
    assert round4_manifest["measured_donor_contract"]["role"] == (
        "no_measured_donor_evidence"
    )


def test_sequence_rejects_missing_pbr_layer_before_output(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    index = json.loads(fixture["pbr"]["index"].read_text())
    index["frames"]["000001"]["layers"].pop("sam3_bed_01")
    write_json(fixture["pbr"]["index"], index)
    resign_index(fixture["pbr"]["index"], fixture["pbr"]["receipt"])

    with pytest.raises(ValueError, match="missing a layer"):
        run_sequence(fixture["args"], expected_frame_count=2)
    assert not fixture["args"].output.exists()


def test_sequence_rejects_failed_measured_report_with_next_action(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    report_path = fixture["measured"][1]["report"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["status"] = "technical_failed_no_support"
    report["promotion_blocker"] = "one or more target frames have no configured donor support"
    report["next_action"] = {
        "action": "add_observed_donor_or_switch_to_constrained_generation_for_residual",
        "blocker": "no_guard_stable_measured_donor_support",
        "status": "technical_failed_no_support",
        "failed_frame_ids": ["000064"],
        "first_failed_frame_id": "000064",
        "not_evaluable_pair_ids": ["0016_to_0017"],
        "not_evaluable_triplet_center_frame_ids": ["000064"],
        "no_support_frame_ids": ["000001"],
        "unresolved_unobserved_pixels": 42,
        "promotion_approved": False,
    }
    write_json(report_path, report)

    with pytest.raises(ValueError) as error:
        run_sequence(fixture["args"], expected_frame_count=2)

    message = str(error.value)
    assert "round 2 measured report did not pass" in message
    assert "status=technical_failed_no_support" in message
    assert "add_observed_donor_or_switch_to_constrained_generation_for_residual" in message
    assert "no_guard_stable_measured_donor_support" in message
    assert "failed_frame_ids=000064" in message
    assert "first_failed_frame_id=000064" in message
    assert "not_evaluable_pair_ids=0016_to_0017" in message
    assert "not_evaluable_triplet_center_frame_ids=000064" in message
    assert "no_support_frame_ids=000001" in message
    assert not fixture["args"].output.exists()


def test_sequence_rejects_wrong_structural_frame_set(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    index = json.loads(fixture["structural"]["index"].read_text())
    index["frame_ids"] = ["000000"]
    index["frames"].pop("000001")
    write_json(fixture["structural"]["index"], index)
    resign_index(fixture["structural"]["index"], fixture["structural"]["receipt"])

    with pytest.raises(ValueError, match="wrong or missing frames"):
        run_sequence(fixture["args"], expected_frame_count=2)


def test_sequence_accepts_materialized_structural_frame_records_and_portable_r4(
    tmp_path: Path,
) -> None:
    fixture = make_fixture(tmp_path)
    structural = make_structural_batch(
        tmp_path / "real_schema",
        limited=False,
        materialized_schema=True,
    )
    fixture["args"].structural_background_index = structural["index"]
    fixture["args"].structural_background_receipt = structural["receipt"]
    fixture["args"].round4_cumulative_masks = structural["index"]

    report = run_sequence(fixture["args"], expected_frame_count=2)

    portable_index_path = (
        fixture["args"].output
        / "normalized_inputs"
        / "round04_cumulative_masks"
        / "cumulative_removal_mask_index.json"
    )
    portable = json.loads(portable_index_path.read_text(encoding="utf-8"))
    assert portable["kind"] == R4_MASK_INDEX_KIND
    assert portable["round_index"] == 4
    assert portable["removed_object_ids"] == list(OBJECT_ORDER)
    assert portable["frame_ids"] == list(FRAME_IDS)
    assert report["inputs"]["round4_cumulative_masks"]["sha256"] == sha256_file(
        portable_index_path
    )
    for sequence_index, frame_id in enumerate(FRAME_IDS):
        copied = portable_index_path.parent / portable["frames"][frame_id]["mask"][
            "path"
        ]
        assert copied.read_bytes() == structural["source_masks"][frame_id].read_bytes()
        assert copied.name == f"{sequence_index:04d}.png"


def test_sequence_rejects_materialized_structural_alpha_mask_mismatch(
    tmp_path: Path,
) -> None:
    fixture = make_fixture(tmp_path)
    structural = make_structural_batch(
        tmp_path / "real_schema",
        limited=False,
        materialized_schema=True,
    )
    index = json.loads(structural["index"].read_text(encoding="utf-8"))
    frame = index["frame_records"][0]
    source_mask_path = structural["source_masks"][FRAME_IDS[0]]
    changed_mask = save_mask(source_mask_path, 5)
    frame["source_cumulative_removal_mask"] = {
        **asset(source_mask_path),
        "pixels": int(changed_mask.sum()),
    }
    write_json(structural["index"], index)
    resign_index(structural["index"], structural["receipt"])
    fixture["args"].structural_background_index = structural["index"]
    fixture["args"].structural_background_receipt = structural["receipt"]
    fixture["args"].round4_cumulative_masks = structural["index"]

    with pytest.raises(ValueError, match="alpha differs"):
        run_sequence(fixture["args"], expected_frame_count=2)


def test_sequence_rejects_wrong_asset_hash(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    index = json.loads(fixture["pbr"]["index"].read_text())
    index["frames"]["000000"]["layers"][OBJECT_ORDER[0]]["rgba"]["sha256"] = "0" * 64
    write_json(fixture["pbr"]["index"], index)
    resign_index(fixture["pbr"]["index"], fixture["pbr"]["receipt"])

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        run_sequence(fixture["args"], expected_frame_count=2)


def test_sequence_rejects_skipped_measured_round(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    manifest_path = fixture["measured"][1]["manifest"]
    manifest = json.loads(manifest_path.read_text())
    manifest["round_index"] = 3
    write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="skips a round"):
        run_sequence(fixture["args"], expected_frame_count=2)


def test_sequence_rejects_removed_object_reintroduced_by_index(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    index = json.loads(fixture["pbr"]["index"].read_text())
    index["round_remaining_object_ids"]["round02_left_pillow"] = [
        "sam3_pillow_left",
        "sam3_pillow_right",
        "sam3_bed_01",
    ]
    write_json(fixture["pbr"]["index"], index)
    resign_index(fixture["pbr"]["index"], fixture["pbr"]["receipt"])

    with pytest.raises(ValueError, match="re-render a removed object"):
        run_sequence(fixture["args"], expected_frame_count=2)


def test_sequence_propagates_limited_demo_scope(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path, limited=True)

    report = run_sequence(fixture["args"], expected_frame_count=2)

    assert report["status"] == "technical_passed_complete_sequence_with_limitations"
    assert report["accepted_with_limitations"] is True
    assert report["acceptance_status"] == "accepted_with_limitations"
    assert report["acceptance_scope"] == "current_demo_only"
    assert report["limitations"]
    assert set(report["failed_metrics"]) == {
        "visual_quality",
        "wall_boundary_color_continuity",
        "wall_boundary_low_frequency_gradient_continuity",
    }
    receipt = json.loads(
        (
            fixture["args"].output / "layered_clean_plate_sequence_receipt.json"
        ).read_text()
    )
    assert receipt["kind"] == SEQUENCE_RECEIPT_KIND
    assert receipt["acceptance_scope"] == "current_demo_only"
    assert receipt["limitations"] == report["limitations"]
    assert receipt["failed_metrics"] == report["failed_metrics"]
    assert receipt["promotion_approved"] is False


def test_sequence_rejects_nonempty_output(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    fixture["args"].output.mkdir()
    (fixture["args"].output / "existing.txt").write_text("occupied", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        run_sequence(fixture["args"], expected_frame_count=2)
