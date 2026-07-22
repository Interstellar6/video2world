from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

import scripts.materialize_accepted_clean_plate_next_round as materializer
from scripts import build_source_aligned_scene_edit_masks as scene_edit_builder
from scripts import materialize_clean_plate_sam_semantic_run_receipt as sam_run_producer
from scripts import qa_clean_plate_sam_semantics as semantic_qa_producer
from scripts.materialize_accepted_clean_plate_next_round import (
    BOUNDARY_KIND,
    EXPECTED_DIRECTED_PAIRS,
    EXPECTED_FRAME_COUNT,
    EXPECTED_TRIPLETS,
    INPUT_KIND,
    NEXT_ROUND_SOURCE_ACCEPTANCE_SCOPE,
    NEXT_ROUND_SOURCE_LINEAGE_SCOPE,
    OUTPUT_KIND,
    OUTPUT_RECEIPT_KIND,
    SELECTION_KIND,
    SELECTION_STATUS,
    TEMPORAL_KIND,
    VISUAL_REVIEW_KIND,
    AcceptedCleanPlateMaterializationError,
    canonical_sha256,
    materialize_accepted_clean_plate_next_round,
    sha256_file,
)
from tests.test_clean_plate_sam_semantic_producers import (
    REMAINING_ID,
    TARGET_ID,
    make_association,
    make_legacy_receipt,
    make_mask_index,
)

FRAME_IDS = tuple(f"{index:06d}" for index in range(48, 73))
WIDTH = 8
HEIGHT = 6


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    digest, size = sha256_file(path)
    declared = path.relative_to(relative_to).as_posix() if relative_to else str(path)
    return {"path": declared, "sha256": digest, "size_bytes": size}


def save_rgb(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values.astype(np.uint8)).save(path)


def save_mask(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values.astype(np.uint8) * 255).save(path)


def save_labels(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values.astype(np.uint16)).save(path)


BOUNDARY_THRESHOLDS = {
    "minimum_mask_iou": 0.90,
    "minimum_mask_precision": 0.90,
    "minimum_mask_recall": 0.90,
    "maximum_boundary_distance_p95_px": 2.0,
    "maximum_seam_color_p95_abs_rgb_delta": 8.0,
    "maximum_seam_gradient_p95_abs_rgb_delta": 8.0,
    "minimum_core_to_local_ring_laplacian_energy_ratio": 0.50,
}


def metric_gate(
    observed: float,
    threshold: float,
    *,
    minimum: bool,
    observed_key: str = "observed",
) -> dict[str, Any]:
    return {
        observed_key: observed,
        "threshold": threshold,
        "passed": observed >= threshold if minimum else observed <= threshold,
    }


def boundary_sections_and_gates(
    *,
    aggregate: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    alignment = {
        "mask_iou": 1.0,
        "mask_precision": 1.0,
        "mask_recall": 1.0,
        "boundary_distance_p95_px": 0.0,
    }
    seam = {
        "seam_color_p95_abs_rgb_delta": 0.0,
        "seam_gradient_p95_abs_rgb_delta": 0.0,
    }
    texture = {"core_to_local_ring_laplacian_energy_ratio": 1.0}
    sections = {"alignment": alignment, "seam": seam, "texture": texture}
    observed_key = "observed_worst" if aggregate else "observed"
    gates = {}
    for gate_name, (
        section,
        metric,
        threshold_name,
        minimum,
    ) in materializer.BOUNDARY_GATE_SPECS.items():
        gates[gate_name] = metric_gate(
            float(sections[section][metric]),
            float(BOUNDARY_THRESHOLDS[threshold_name]),
            minimum=minimum,
            observed_key=observed_key,
        )
    return alignment, seam, texture, gates


def temporal_pair_gates(coverage: dict[str, float], metrics: dict[str, float]) -> dict[str, Any]:
    sections = {"coverage": coverage, "metrics": metrics}
    return {
        gate_name: metric_gate(
            float(sections[section][metric]),
            float(materializer.TEMPORAL_THRESHOLDS[threshold_name]),
            minimum=minimum,
        )
        for gate_name, (section, metric, threshold_name, minimum) in (
            materializer.TEMPORAL_PAIR_GATE_SPECS.items()
        )
    }


def temporal_triplet_gates(coverage: dict[str, float], metrics: dict[str, float]) -> dict[str, Any]:
    sections = {"coverage": coverage, "metrics": metrics}
    return {
        gate_name: metric_gate(
            float(sections[section][metric]),
            float(materializer.TEMPORAL_THRESHOLDS[threshold_name]),
            minimum=minimum,
        )
        for gate_name, (section, metric, threshold_name, minimum) in (
            materializer.TEMPORAL_TRIPLET_GATE_SPECS.items()
        )
    }


def make_temporal_pair_record(source: str, target: str) -> dict[str, Any]:
    coverage = {"valid_core_fraction": 1.0, "valid_control_fraction": 1.0}
    metrics = {
        "control_color_p95_abs_rgb_delta": 0.0,
        "core_color_p95_abs_rgb_delta": 0.0,
        "core_excess_over_control_p95_abs_rgb_delta": 0.0,
        "core_gradient_p95_abs_rgb_delta": 0.0,
        "core_bad_pixel_fraction": 0.0,
    }
    return {
        "source_sequence_index": FRAME_IDS.index(source),
        "source_frame_id": source,
        "target_sequence_index": FRAME_IDS.index(target),
        "target_frame_id": target,
        "coverage": coverage,
        "metrics": metrics,
        "evaluable": True,
        "passed": True,
        "gates": temporal_pair_gates(coverage, metrics),
    }


def make_temporal_triplet_record(index: int) -> dict[str, Any]:
    coverage = {"valid_core_fraction": 1.0}
    metrics = {
        "core_second_difference_p95_abs_rgb_delta": 0.0,
        "core_excess_over_control_p95_abs_rgb_delta": 0.0,
    }
    return {
        "previous_sequence_index": index,
        "previous_frame_id": FRAME_IDS[index],
        "center_sequence_index": index + 1,
        "center_frame_id": FRAME_IDS[index + 1],
        "following_sequence_index": index + 2,
        "following_frame_id": FRAME_IDS[index + 2],
        "coverage": coverage,
        "metrics": metrics,
        "evaluable": True,
        "passed": True,
        "gates": temporal_triplet_gates(coverage, metrics),
    }


def make_source(index: int) -> np.ndarray:
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    return np.stack(
        (
            20 + xx * 5 + index,
            30 + yy * 4 + index,
            np.full_like(xx, 60 + index),
        ),
        axis=2,
    ).astype(np.uint8)


def make_fixture(
    tmp_path: Path,
    *,
    round_index: int = 1,
    previous_manifest: Path | None = None,
    target_object_id: str | None = None,
) -> dict[str, Path]:
    if round_index == 1:
        assert previous_manifest is None
    else:
        assert previous_manifest is not None
    target_object_id = target_object_id or TARGET_ID
    previous_root: Path | None = None
    previous_frames: dict[str, dict[str, Any]] = {}
    if previous_manifest is not None:
        previous_root = previous_manifest.parent
        previous_value = json.loads(previous_manifest.read_text(encoding="utf-8"))
        previous_frames = {frame["frame_id"]: frame for frame in previous_value["frame_records"]}
    selection_root = tmp_path / "selection"
    source_root = tmp_path / "source"
    input_manifest_path = tmp_path / "candidate-selection-input.json"
    source_batch_path = tmp_path / "source-batch-receipt.json"
    candidate_batch_path = tmp_path / "candidate-batch-receipt.json"
    write_json(input_manifest_path, {"kind": "test.selection.input"})
    write_json(source_batch_path, {"kind": "test.source.batch"})
    write_json(candidate_batch_path, {"kind": "test.candidate.batch"})
    candidate_batch_record = record(candidate_batch_path)

    selection_frames: list[dict[str, Any]] = []
    frame_set: list[dict[str, Any]] = []
    for sequence_index, frame_id in enumerate(FRAME_IDS):
        if previous_root is None:
            source_rgb = make_source(sequence_index)
            source_path = source_root / f"{frame_id}.png"
            save_rgb(source_path, source_rgb)
        else:
            previous_record = previous_frames[frame_id]
            source_path = previous_root / previous_record["source_rgb"]["path"]
            with Image.open(source_path) as image:
                source_rgb = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        core = np.zeros((HEIGHT, WIDTH), dtype=bool)
        if round_index == 1:
            core[2:4, 3:5] = True
        else:
            core[1:3, 1:3] = True
        editable = np.zeros((HEIGHT, WIDTH), dtype=bool)
        if round_index == 1:
            editable[1:5, 2:6] = True
        else:
            editable[0:4, 0:4] = True
        composite = source_rgb.copy()
        composite[editable] = np.clip(
            composite[editable].astype(np.int16) + 20,
            0,
            255,
        ).astype(np.uint8)

        composite_path = selection_root / "frames" / f"{sequence_index:04d}.png"
        core_path = selection_root / "masks" / "core" / f"{sequence_index:04d}.png"
        editable_path = selection_root / "masks" / "editable" / f"{sequence_index:04d}.png"
        frame_receipt_path = tmp_path / "candidate-frame-receipts" / f"{frame_id}.json"
        save_rgb(composite_path, composite)
        save_mask(core_path, core)
        save_mask(editable_path, editable)
        write_json(frame_receipt_path, {"kind": "test.candidate.frame", "frame_id": frame_id})

        source_record = record(source_path)
        composite_record = record(composite_path, relative_to=selection_root)
        core_record = record(core_path, relative_to=selection_root)
        editable_record = record(editable_path, relative_to=selection_root)
        selection_frames.append(
            {
                "sequence_index": sequence_index,
                "frame_id": frame_id,
                "selection_note": "accepted fixture candidate",
                "source_rgb": source_record,
                "candidate_batch_receipt": candidate_batch_record,
                "candidate_frame_receipt": record(frame_receipt_path),
                "selected_inputs": {
                    "composite_rgb": record(composite_path),
                    "core_mask": record(core_path),
                    "editable_mask": record(editable_path),
                },
                "outputs": {
                    "composite_rgb": composite_record,
                    "core_mask": core_record,
                    "editable_mask": editable_record,
                },
                "dimensions": {"width": WIDTH, "height": HEIGHT},
                "mask_pixels": {"core": int(core.sum()), "editable": int(editable.sum())},
                "exactness": {
                    "passed": True,
                    "candidate_source_rgb_matches_source_batch": True,
                    "selected_artifacts_match_candidate_batch_receipt": True,
                    "core_is_subset_of_editable": True,
                    "outside_editable_rgb_exact_recomputed": True,
                    "outside_editable_changed_pixels_recomputed": 0,
                },
            }
        )
        frame_set.append(
            {
                "sequence_index": sequence_index,
                "frame_id": frame_id,
                "source_rgb_sha256": source_record["sha256"],
                "composite_rgb_sha256": composite_record["sha256"],
                "core_mask_sha256": core_record["sha256"],
                "editable_mask_sha256": editable_record["sha256"],
                "candidate_batch_receipt_sha256": candidate_batch_record["sha256"],
            }
        )

    selection_receipt = {
        "schema_version": 1,
        "kind": SELECTION_KIND,
        "status": SELECTION_STATUS,
        "promotion_allowed": False,
        "human_review_required": True,
        "input_manifest": record(input_manifest_path),
        "source_batch_receipt": record(source_batch_path),
        "ordered_frame_ids": list(FRAME_IDS),
        "frames": selection_frames,
        "aggregate": {
            "frame_count": EXPECTED_FRAME_COUNT,
            "frame_set_sha256": canonical_sha256(frame_set),
            "all_source_rgb_lineages_match": True,
            "all_frame_sets_and_order_validated": True,
            "all_dimensions_match": True,
            "all_input_and_output_hashes_verified": True,
            "all_core_masks_are_subsets_of_editable_masks": True,
            "all_outputs_outside_editable_rgb_exact_recomputed": True,
        },
        "review": {
            "automatic_accept": False,
            "human_review_required": True,
            "quality_acceptance_performed": False,
            "published": False,
        },
    }
    selection_path = selection_root / "selection_receipt.json"
    write_json(selection_path, selection_receipt)
    selection_record = record(selection_path)

    boundary_records: list[dict[str, Any]] = []
    for frame in selection_frames:
        alignment, seam, texture, frame_gates = boundary_sections_and_gates()
        boundary_records.append(
            {
                "sequence_index": frame["sequence_index"],
                "frame_id": frame["frame_id"],
                "inputs": {
                    "composite_rgb": frame["selected_inputs"]["composite_rgb"],
                    "source_rgb": frame["source_rgb"],
                    "removal_core_mask": frame["selected_inputs"]["core_mask"],
                    "target_matte": frame["selected_inputs"]["core_mask"],
                    "editable_mask": frame["selected_inputs"]["editable_mask"],
                },
                "alignment": alignment,
                "seam": seam,
                "texture": texture,
                "gates": frame_gates,
                "passed": True,
            }
        )
    _alignment, _seam, _texture, aggregate_boundary_gates = boundary_sections_and_gates(
        aggregate=True
    )
    boundary = {
        "schema_version": 1,
        "kind": BOUNDARY_KIND,
        "status": "technical_passed",
        "boundary_texture_gate_passed": True,
        "promotion_approved": False,
        "promotion_scope": "boundary_and_local_texture_only",
        "frame_count": EXPECTED_FRAME_COUNT,
        "thresholds": dict(BOUNDARY_THRESHOLDS),
        "gates": aggregate_boundary_gates,
        "frame_records": boundary_records,
    }
    boundary_path = tmp_path / "boundary-report.json"
    write_json(boundary_path, boundary)
    boundary_record = record(boundary_path)

    temporal_records: list[dict[str, Any]] = []
    for frame in selection_frames:
        if previous_root is None:
            cumulative_record = frame["selected_inputs"]["core_mask"]
        else:
            previous_record = previous_frames[frame["frame_id"]]
            previous_cumulative_path = (
                previous_root / previous_record["previous_edited_region_mask"]["path"]
            )
            with Image.open(previous_cumulative_path) as image:
                previous_cumulative = np.asarray(image.convert("L"), dtype=np.uint8) > 0
            current_core_path = Path(frame["selected_inputs"]["core_mask"]["path"])
            with Image.open(current_core_path) as image:
                current_core = np.asarray(image.convert("L"), dtype=np.uint8) > 0
            cumulative_path = (
                tmp_path / "temporal-cumulative" / f"{int(frame['sequence_index']):04d}.png"
            )
            save_mask(cumulative_path, previous_cumulative | current_core)
            cumulative_record = record(cumulative_path)
        temporal_records.append(
            {
                "sequence_index": frame["sequence_index"],
                "frame_id": frame["frame_id"],
                "inputs": {
                    "selected_composite_rgb": frame["selected_inputs"]["composite_rgb"],
                    "source_rgb": frame["source_rgb"],
                    "current_core_mask": frame["selected_inputs"]["core_mask"],
                    "cumulative_core_mask": cumulative_record,
                    "editable_mask": frame["selected_inputs"]["editable_mask"],
                    "selection_receipt": selection_record,
                },
            }
        )
    temporal = {
        "schema_version": 1,
        "kind": TEMPORAL_KIND,
        "status": "technical_passed_temporal_only",
        "round_index": round_index,
        "review_only": True,
        "promotion_approved": False,
        "forbidden_claims": {
            "pbr_used_as_pass_evidence": False,
            "da3_used_as_pass_evidence": False,
            "flow_completion_used": False,
            "semantic_correctness_claimed": False,
            "clean_plate_promotion_claimed": False,
            "geometry_reconstruction_readiness_claimed": False,
        },
        "dependencies": {
            "selection_manifest": selection_record,
            "upstream_boundary_report": boundary_record,
        },
        "frame_count": EXPECTED_FRAME_COUNT,
        "ordered_frame_ids": list(FRAME_IDS),
        "pairing": dict(materializer.TEMPORAL_PAIRING),
        "sampling": dict(materializer.TEMPORAL_SAMPLING),
        "thresholds": dict(materializer.TEMPORAL_THRESHOLDS),
        "flow_backend": {
            "kind": "plain_raft",
            "repository_path": "/test-only/RAFT",
            "repository_commit": materializer.EXPECTED_RAFT_COMMIT,
            "checkpoint": {
                "path": "/test-only/raft-things.pth",
                "sha256": materializer.EXPECTED_RAFT_CHECKPOINT_SHA256,
                "bytes": materializer.EXPECTED_RAFT_CHECKPOINT_BYTES,
            },
            "precision": "float32",
            "iterations": 20,
            "flow_completion": False,
            "inference_resolution": [1280, 720],
            "device": "cuda:0",
        },
        "frame_records": temporal_records,
        "counts": {
            "expected_directed_pairs": EXPECTED_DIRECTED_PAIRS,
            "observed_directed_pairs": EXPECTED_DIRECTED_PAIRS,
            "not_evaluable_directed_pairs": 0,
            "failed_directed_pairs": 0,
            "expected_triplets": EXPECTED_TRIPLETS,
            "observed_triplets": EXPECTED_TRIPLETS,
            "not_evaluable_triplets": 0,
            "failed_triplets": 0,
        },
        "gates": {
            "all_directed_pairs_evaluable": metric_gate(48.0, 48.0, minimum=True),
            "all_triplets_evaluable": metric_gate(23.0, 23.0, minimum=True),
            "failed_directed_pairs_lte": metric_gate(0.0, 0.0, minimum=False),
            "failed_triplets_lte": metric_gate(0.0, 0.0, minimum=False),
        },
        "pair_records": [
            make_temporal_pair_record(source, target)
            for left, right in pairwise(FRAME_IDS)
            for source, target in ((left, right), (right, left))
        ],
        "triplet_records": [
            make_temporal_triplet_record(index) for index in range(EXPECTED_TRIPLETS)
        ],
    }
    temporal_path = tmp_path / "temporal-report.json"
    write_json(temporal_path, temporal)
    temporal_record = record(temporal_path)

    frame_paths = {
        frame["frame_id"]: {
            "source": Path(frame["source_rgb"]["path"]),
            "composite": Path(frame["selected_inputs"]["composite_rgb"]["path"]),
        }
        for frame in selection_frames
    }
    baseline_index, baseline_items = make_mask_index(
        tmp_path,
        name="semantic-baseline",
        object_ids=(TARGET_ID, REMAINING_ID),
        image_paths={frame_id: frame_paths[frame_id]["source"] for frame_id in FRAME_IDS},
    )
    candidate_index, candidate_items = make_mask_index(
        tmp_path,
        name="semantic-candidate",
        object_ids=(REMAINING_ID,),
        image_paths={frame_id: frame_paths[frame_id]["composite"] for frame_id in FRAME_IDS},
    )
    legacy_sam = make_legacy_receipt(
        tmp_path,
        candidate_index,
        round_index=round_index,
        target_object_id=target_object_id,
    )
    baseline_association = make_association(
        tmp_path,
        name="semantic-baseline",
        index_path=baseline_index,
        index_items=baseline_items,
        frame_paths=frame_paths,
        source_key="source",
        assigned_object_ids=(TARGET_ID, REMAINING_ID),
        peeled_target=False,
    )
    candidate_association = make_association(
        tmp_path,
        name="semantic-candidate",
        index_path=candidate_index,
        index_items=candidate_items,
        frame_paths=frame_paths,
        source_key="composite",
        assigned_object_ids=(REMAINING_ID,),
        peeled_target=True,
    )
    sam_run_output = tmp_path / "strict-sam-run"
    sam_run_producer.materialize(
        selection_receipt=selection_path,
        legacy_sam_run_receipt=legacy_sam,
        mask_index_path=candidate_index,
        round_index=round_index,
        target_object_id=target_object_id,
        output_dir=sam_run_output,
    )
    sam_run_path = sam_run_output / "sam_run_receipt.json"
    semantic_output = tmp_path / "semantic-qa"
    semantic = semantic_qa_producer.produce(
        selection_receipt=selection_path,
        sam_run_receipt=sam_run_path,
        baseline_association_path=baseline_association,
        candidate_association_path=candidate_association,
        round_index=round_index,
        target_object_id=target_object_id,
        output_dir=semantic_output,
    )
    assert semantic["status"] == semantic_qa_producer.PASS_STATUS
    semantic_path = semantic_output / "semantic_qa_report.json"
    semantic_record = record(semantic_path)

    visual = {
        "schema_version": 1,
        "kind": VISUAL_REVIEW_KIND,
        "status": "passed",
        "decision": "accepted_for_next_round_source",
        "accepted_for_next_round_source": True,
        "promotion_approved": False,
        "canonical_or_live_promotion_approved": False,
        "frame_count": EXPECTED_FRAME_COUNT,
        "ordered_frame_ids": list(FRAME_IDS),
        "reviewed_frame_ids": list(FRAME_IDS),
        "reviewer": {"kind": "human", "id": "test-reviewer"},
        "reviewed_at": "2026-07-19T12:00:00+08:00",
        "evidence": {
            "candidate_selection_receipt": selection_record,
            "boundary_qa_report": boundary_record,
            "temporal_qa_report": temporal_record,
            "sam_semantic_qa_report": semantic_record,
        },
        "gates": {
            "all_frames_reviewed": True,
            "no_visible_target_object_residual": True,
            "no_obvious_boundary_misalignment": True,
            "background_or_remaining_objects_plausible": True,
            "accepted_for_next_round_source": True,
        },
        "frame_records": [
            {
                "sequence_index": frame["sequence_index"],
                "frame_id": frame["frame_id"],
                "composite_rgb_sha256": frame["selected_inputs"]["composite_rgb"]["sha256"],
                "passed": True,
            }
            for frame in selection_frames
        ],
        "limitations": ["synthetic test fixture only"],
    }
    visual_path = tmp_path / "visual-review.json"
    write_json(visual_path, visual)

    materialization_input = {
        "schema_version": 1,
        "kind": INPUT_KIND,
        "target_object_id": target_object_id,
        "round_index": round_index,
        "next_round_index": round_index + 1,
        "expected_frame_count": EXPECTED_FRAME_COUNT,
        "destination_scope": "next_round_source_only",
        "canonical_or_live_manifest_targeted": False,
        "candidate_selection_receipt": selection_record,
        "boundary_qa_report": boundary_record,
        "temporal_qa_report": temporal_record,
        "sam_semantic_qa_report": semantic_record,
        "human_visual_review_receipt": record(visual_path),
        "previous_accepted_next_round_source_manifest": (
            record(previous_manifest) if previous_manifest is not None else None
        ),
    }
    acceptance_path = tmp_path / "accepted-input.json"
    write_json(acceptance_path, materialization_input)
    return {
        "acceptance": acceptance_path,
        "selection": selection_path,
        "boundary": boundary_path,
        "temporal": temporal_path,
        "sam": sam_run_path,
        "semantic": semantic_path,
        "baseline_association": baseline_association,
        "candidate_association": candidate_association,
        "visual": visual_path,
    }


def update_acceptance_asset(paths: dict[str, Path], input_key: str, evidence_path: Path) -> None:
    acceptance = json.loads(paths["acceptance"].read_text(encoding="utf-8"))
    acceptance[input_key] = record(evidence_path)
    write_json(paths["acceptance"], acceptance)


def update_sam_and_semantic_binding(paths: dict[str, Path]) -> None:
    semantic = json.loads(paths["semantic"].read_text(encoding="utf-8"))
    semantic["sam_run_receipt"] = record(paths["sam"])
    write_json(paths["semantic"], semantic)
    update_acceptance_asset(paths, "sam_semantic_qa_report", paths["semantic"])


def update_association_and_semantic_binding(
    paths: dict[str, Path],
    *,
    semantic_key: str,
    association_path: Path,
) -> None:
    semantic = json.loads(paths["semantic"].read_text(encoding="utf-8"))
    semantic[semantic_key] = record(association_path)
    write_json(paths["semantic"], semantic)
    update_acceptance_asset(paths, "sam_semantic_qa_report", paths["semantic"])


def test_materializes_exact_25_frame_next_round_source_without_touching_canonical(
    tmp_path: Path,
) -> None:
    paths = make_fixture(tmp_path)
    canonical = tmp_path / "web" / "public" / "worlds" / "bedroom4" / "manifest.json"
    write_json(canonical, {"sentinel": "unchanged"})
    before = canonical.read_bytes()
    output = tmp_path / "accepted-r1"

    manifest, receipt = materialize_accepted_clean_plate_next_round(
        paths["acceptance"],
        output,
        created_at=datetime(2026, 7, 19, 12, 30, tzinfo=UTC),
    )

    assert manifest["kind"] == OUTPUT_KIND
    assert manifest["status"] == "accepted_clean_plate_materialized_for_next_round_source"
    assert manifest["frame_count"] == EXPECTED_FRAME_COUNT
    assert manifest["ordered_frame_ids"] == list(FRAME_IDS)
    assert manifest["next_round_source_approved"] is True
    assert manifest["acceptance_scope"] == NEXT_ROUND_SOURCE_ACCEPTANCE_SCOPE
    assert manifest["lineage_scope"] == NEXT_ROUND_SOURCE_LINEAGE_SCOPE
    assert manifest["corrected_full_pipeline"] is False
    assert manifest["promotion_approved"] is False
    assert manifest["canonical_promotion_approved"] is False
    assert manifest["canonical_or_live_manifest_modified"] is False
    assert manifest["forbidden_claims"] == {
        "canonical_or_live_promotion_claimed": False,
        "corrected_full_pipeline_terminal_report_claimed": False,
    }
    assert receipt["kind"] == OUTPUT_RECEIPT_KIND
    assert receipt["acceptance_scope"] == NEXT_ROUND_SOURCE_ACCEPTANCE_SCOPE
    assert receipt["lineage_scope"] == NEXT_ROUND_SOURCE_LINEAGE_SCOPE
    assert receipt["corrected_full_pipeline"] is False
    assert receipt["canonical_promotion_approved"] is False
    assert receipt["materialization"] == {
        "same_parent_staging": True,
        "atomic_directory_rename": True,
        "output_assets_hash_verified_after_copy": True,
        "output_paths_contained": True,
    }
    assert canonical.read_bytes() == before
    assert len(list((output / "frames").glob("*.png"))) == EXPECTED_FRAME_COUNT
    assert len(list((output / "masks" / "core").glob("*.png"))) == EXPECTED_FRAME_COUNT
    assert len(list((output / "masks" / "editable").glob("*.png"))) == EXPECTED_FRAME_COUNT
    assert len(list((output / "masks" / "cumulative").glob("*.png"))) == EXPECTED_FRAME_COUNT
    assert len(list((output / "contributor_labels").glob("*.png"))) == EXPECTED_FRAME_COUNT
    assert manifest["contributor_label_registry"] == {REMAINING_ID: 1}
    written = json.loads((output / "next_round_source_manifest.json").read_text())
    assert written == manifest
    for frame in manifest["frame_records"]:
        assert frame["segmentation_source_rgb_sha256"] == frame["source_rgb"]["sha256"]
        assert (
            frame["previous_contributor_labels"]["source_rgb_sha256"]
            == frame["source_rgb"]["sha256"]
        )
        for key in (
            "source_rgb",
            "accepted_round_core_mask",
            "accepted_round_editable_mask",
            "previous_edited_region_mask",
            "previous_contributor_labels",
        ):
            declared = output / frame[key]["path"]
            assert declared.is_file()
            assert declared.resolve().is_relative_to(output.resolve())
            assert sha256_file(declared)[0] == frame[key]["sha256"]


@pytest.mark.parametrize(
    ("evidence_key", "input_key", "mutation", "message"),
    [
        (
            "boundary",
            "boundary_qa_report",
            lambda value: value.update(status="technical_failed"),
            "boundary QA did not pass",
        ),
        (
            "temporal",
            "temporal_qa_report",
            lambda value: value.update(status="not_evaluable"),
            "temporal QA did not pass",
        ),
        (
            "semantic",
            "sam_semantic_qa_report",
            lambda value: value.update(status="evaluation_pending"),
            "SAM/semantic QA did not pass",
        ),
        (
            "visual",
            "human_visual_review_receipt",
            lambda value: value.update(status="rejected"),
            "human visual review did not pass",
        ),
    ],
)
def test_missing_pending_or_rejected_evidence_fails_closed(
    tmp_path: Path,
    evidence_key: str,
    input_key: str,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    paths = make_fixture(tmp_path)
    value = json.loads(paths[evidence_key].read_text(encoding="utf-8"))
    mutation(value)
    write_json(paths[evidence_key], value)
    update_acceptance_asset(paths, input_key, paths[evidence_key])

    with pytest.raises(AcceptedCleanPlateMaterializationError, match=message):
        materialize_accepted_clean_plate_next_round(paths["acceptance"], tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_boundary_failure_surfaces_next_action(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    boundary = json.loads(paths["boundary"].read_text(encoding="utf-8"))
    boundary["status"] = "technical_failed"
    boundary["boundary_texture_gate_passed"] = False
    boundary["promotion_blocker"] = "boundary and local texture gates failed"
    boundary["next_action"] = {
        "action": "repair_removal_mask_or_target_matte_before_regeneration",
        "reason": "The removal core and target matte disagree.",
        "blocking_gate_groups": ["mask_alignment"],
        "failed_gates": ["mask_iou_gte", "boundary_distance_p95_lte"],
        "failed_frame_ids": ["000064"],
        "first_failed_frame_id": "000064",
        "promotion_approved": False,
    }
    write_json(paths["boundary"], boundary)
    update_acceptance_asset(paths, "boundary_qa_report", paths["boundary"])

    with pytest.raises(AcceptedCleanPlateMaterializationError) as error:
        materialize_accepted_clean_plate_next_round(paths["acceptance"], tmp_path / "output")

    message = str(error.value)
    assert "boundary QA did not pass" in message
    assert "status=technical_failed" in message
    assert "promotion_blocker=boundary and local texture gates failed" in message
    assert "repair_removal_mask_or_target_matte_before_regeneration" in message
    assert "blocking_gate_groups=mask_alignment" in message
    assert "failed_gates=mask_iou_gte,boundary_distance_p95_lte" in message
    assert "failed_frame_ids=000064" in message
    assert "first_failed_frame_id=000064" in message
    assert not (tmp_path / "output").exists()


def test_temporal_failure_surfaces_next_action(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    temporal = json.loads(paths["temporal"].read_text(encoding="utf-8"))
    temporal["status"] = "not_evaluable"
    temporal["next_action"] = {
        "action": "repair_temporal_evidence_before_retesting",
        "reason": "Strict flow evidence was not evaluable.",
        "not_evaluable_pair_ids": ["0016_to_0017"],
        "not_evaluable_triplet_center_frame_ids": ["000064"],
        "promotion_approved": False,
    }
    write_json(paths["temporal"], temporal)
    update_acceptance_asset(paths, "temporal_qa_report", paths["temporal"])

    with pytest.raises(AcceptedCleanPlateMaterializationError) as error:
        materialize_accepted_clean_plate_next_round(paths["acceptance"], tmp_path / "output")

    message = str(error.value)
    assert "temporal QA did not pass" in message
    assert "status=not_evaluable" in message
    assert "next_action=repair_temporal_evidence_before_retesting" in message
    assert "not_evaluable_pair_ids=0016_to_0017" in message
    assert "not_evaluable_triplet_center_frame_ids=000064" in message
    assert not (tmp_path / "output").exists()


def test_rejects_semantic_report_with_only_24_frame_records(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    semantic = json.loads(paths["semantic"].read_text(encoding="utf-8"))
    semantic["frame_records"].pop()
    write_json(paths["semantic"], semantic)
    update_acceptance_asset(paths, "sam_semantic_qa_report", paths["semantic"])

    with pytest.raises(AcceptedCleanPlateMaterializationError, match="exactly 25 records"):
        materialize_accepted_clean_plate_next_round(paths["acceptance"], tmp_path / "output")


def test_rejects_semantic_evidence_for_another_target_object(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    acceptance = json.loads(paths["acceptance"].read_text(encoding="utf-8"))
    acceptance["target_object_id"] = "sam3_another_object"
    write_json(paths["acceptance"], acceptance)

    with pytest.raises(AcceptedCleanPlateMaterializationError, match="target_object_id differs"):
        materialize_accepted_clean_plate_next_round(paths["acceptance"], tmp_path / "output")


def test_rejects_contributor_labels_not_bound_to_accepted_composite(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    semantic = json.loads(paths["semantic"].read_text(encoding="utf-8"))
    semantic["frame_records"][0]["outputs"]["contributor_labels"]["source_rgb_sha256"] = "0" * 64
    write_json(paths["semantic"], semantic)
    update_acceptance_asset(paths, "sam_semantic_qa_report", paths["semantic"])

    with pytest.raises(AcceptedCleanPlateMaterializationError, match="not bound to the composite"):
        materialize_accepted_clean_plate_next_round(paths["acceptance"], tmp_path / "output")


def test_rejects_unregistered_contributor_label_value(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    semantic = json.loads(paths["semantic"].read_text(encoding="utf-8"))
    label_record = semantic["frame_records"][0]["outputs"]["contributor_labels"]
    label_path = paths["semantic"].parent / label_record["path"]
    with Image.open(label_path) as image:
        labels = np.array(image, dtype=np.uint16, copy=True)
    labels[0, 1] = 99
    save_labels(label_path, labels)
    label_record.update(record(label_path))
    write_json(paths["semantic"], semantic)
    update_acceptance_asset(paths, "sam_semantic_qa_report", paths["semantic"])

    with pytest.raises(AcceptedCleanPlateMaterializationError, match="unregistered values"):
        materialize_accepted_clean_plate_next_round(paths["acceptance"], tmp_path / "output")


def test_rejects_removed_target_in_contributor_label_registry(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    semantic = json.loads(paths["semantic"].read_text(encoding="utf-8"))
    semantic["label_registry"][semantic["target_object_id"]] = 9
    write_json(paths["semantic"], semantic)
    update_acceptance_asset(paths, "sam_semantic_qa_report", paths["semantic"])

    with pytest.raises(AcceptedCleanPlateMaterializationError, match="removed target object"):
        materialize_accepted_clean_plate_next_round(paths["acceptance"], tmp_path / "output")


def test_rejects_registered_remaining_label_absent_from_all_25_maps(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    semantic = json.loads(paths["semantic"].read_text(encoding="utf-8"))
    semantic["label_registry"]["physical_extra"] = 2
    semantic["aggregate"]["remaining_label_preservation"]["physical_extra"] = {
        "label": 2,
        "frame_occurrence_count": 0,
        "pixel_count": 0,
        "preserved": True,
    }
    write_json(paths["semantic"], semantic)
    update_acceptance_asset(paths, "sam_semantic_qa_report", paths["semantic"])

    with pytest.raises(AcceptedCleanPlateMaterializationError, match="label_registry differs"):
        materialize_accepted_clean_plate_next_round(paths["acceptance"], tmp_path / "output")


def test_rejects_semantic_preservation_count_not_closed_to_label_maps(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    semantic = json.loads(paths["semantic"].read_text(encoding="utf-8"))
    semantic["aggregate"]["remaining_label_preservation"][REMAINING_ID]["pixel_count"] += 1
    write_json(paths["semantic"], semantic)
    update_acceptance_asset(paths, "sam_semantic_qa_report", paths["semantic"])

    with pytest.raises(AcceptedCleanPlateMaterializationError, match="pixel_count mismatch"):
        materialize_accepted_clean_plate_next_round(paths["acceptance"], tmp_path / "output")


def test_rejects_arbitrary_json_as_sam_run_receipt(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    sam = json.loads(paths["sam"].read_text(encoding="utf-8"))
    sam["kind"] = "test.sam.run"
    write_json(paths["sam"], sam)
    update_sam_and_semantic_binding(paths)

    with pytest.raises(AcceptedCleanPlateMaterializationError, match="SAM run kind is invalid"):
        materialize_accepted_clean_plate_next_round(paths["acceptance"], tmp_path / "output")


def test_rejects_sam_run_from_unaudited_producer_script(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    sam = json.loads(paths["sam"].read_text(encoding="utf-8"))
    sam["producer"]["script"]["sha256"] = "0" * 64
    write_json(paths["sam"], sam)
    update_sam_and_semantic_binding(paths)

    with pytest.raises(AcceptedCleanPlateMaterializationError, match="producer script SHA-256"):
        materialize_accepted_clean_plate_next_round(paths["acceptance"], tmp_path / "output")


def test_rejects_semantic_report_from_unaudited_producer_script(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    semantic = json.loads(paths["semantic"].read_text(encoding="utf-8"))
    semantic["producer"]["script"]["sha256"] = "0" * 64
    write_json(paths["semantic"], semantic)
    update_acceptance_asset(paths, "sam_semantic_qa_report", paths["semantic"])

    with pytest.raises(AcceptedCleanPlateMaterializationError, match="producer script SHA-256"):
        materialize_accepted_clean_plate_next_round(paths["acceptance"], tmp_path / "output")


@pytest.mark.parametrize("producer", ["sam", "semantic"])
def test_rejects_tampered_portable_producer_script_bytes(
    tmp_path: Path,
    producer: str,
) -> None:
    paths = make_fixture(tmp_path)
    receipt_path = paths[producer]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    producer_path = receipt_path.parent / receipt["producer"]["script"]["path"]
    producer_path.write_text(f"tampered {producer} producer bytes\n", encoding="utf-8")

    with pytest.raises(AcceptedCleanPlateMaterializationError, match="SHA-256 mismatch"):
        materialize_accepted_clean_plate_next_round(paths["acceptance"], tmp_path / "output")


def test_rejects_unaudited_sam_checkpoint(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    sam = json.loads(paths["sam"].read_text(encoding="utf-8"))
    sam["model"]["checkpoint"]["sha256"] = "0" * 64
    write_json(paths["sam"], sam)
    update_sam_and_semantic_binding(paths)

    with pytest.raises(
        AcceptedCleanPlateMaterializationError,
        match="checkpoint hash is not audited",
    ):
        materialize_accepted_clean_plate_next_round(paths["acceptance"], tmp_path / "output")


def test_rejects_sam_checkpoint_with_wrong_audited_size(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    sam = json.loads(paths["sam"].read_text(encoding="utf-8"))
    sam["model"]["checkpoint"]["bytes"] -= 1
    write_json(paths["sam"], sam)
    update_sam_and_semantic_binding(paths)

    with pytest.raises(AcceptedCleanPlateMaterializationError, match="byte size is not audited"):
        materialize_accepted_clean_plate_next_round(paths["acceptance"], tmp_path / "output")


def test_rejects_sam_receipt_with_source_frame_from_another_sequence_index(
    tmp_path: Path,
) -> None:
    paths = make_fixture(tmp_path)
    sam = json.loads(paths["sam"].read_text(encoding="utf-8"))
    wrong_source = sam["frame_records"][1]["source_rgb"]
    sam["frame_records"][0]["source_rgb"] = wrong_source
    sam["frame_records"][0]["source_rgb_sha256"] = wrong_source["sha256"]
    write_json(paths["sam"], sam)
    update_sam_and_semantic_binding(paths)

    with pytest.raises(AcceptedCleanPlateMaterializationError, match="selected composite RGB"):
        materialize_accepted_clean_plate_next_round(paths["acceptance"], tmp_path / "output")


def test_rejects_tampered_portable_actual_sam_input_bytes(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    sam = json.loads(paths["sam"].read_text(encoding="utf-8"))
    actual_input = paths["sam"].parent / sam["frame_records"][0]["actual_sam_input"]["path"]
    actual_input.write_bytes(b"tampered actual SAM input\n")

    with pytest.raises(AcceptedCleanPlateMaterializationError, match="SHA-256 mismatch"):
        materialize_accepted_clean_plate_next_round(paths["acceptance"], tmp_path / "output")


def test_rejects_sam_receipt_for_another_target_object(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    sam = json.loads(paths["sam"].read_text(encoding="utf-8"))
    sam["target_object_id"] = "sam3_another_object"
    write_json(paths["sam"], sam)
    update_sam_and_semantic_binding(paths)

    with pytest.raises(AcceptedCleanPlateMaterializationError, match="SAM run target_object_id"):
        materialize_accepted_clean_plate_next_round(paths["acceptance"], tmp_path / "output")


@pytest.mark.parametrize("mutation", ["script", "source", "mask"])
def test_rejects_tampered_candidate_physical_association(
    tmp_path: Path,
    mutation: str,
) -> None:
    paths = make_fixture(tmp_path)
    association = json.loads(paths["candidate_association"].read_text(encoding="utf-8"))
    if mutation == "script":
        association["execution"]["script"]["sha256"] = "0" * 64
        expected = "producer script SHA-256"
    elif mutation == "source":
        association["sources"]["frames"][0]["sha256"] = "0" * 64
        association["source_set_sha256"] = canonical_sha256(association["sources"])
        expected = "SHA-256 mismatch"
    else:
        association["sources"]["masks"][0]["sha256"] = "0" * 64
        association["source_set_sha256"] = canonical_sha256(association["sources"])
        expected = "SHA-256 mismatch"
    write_json(paths["candidate_association"], association)
    update_association_and_semantic_binding(
        paths,
        semantic_key="candidate_physical_association",
        association_path=paths["candidate_association"],
    )

    with pytest.raises(AcceptedCleanPlateMaterializationError, match=expected):
        materialize_accepted_clean_plate_next_round(paths["acceptance"], tmp_path / "output")


@pytest.mark.parametrize("asset_kind", ["camera", "anchor"])
def test_materializer_reopens_association_camera_and_anchor_assets(
    tmp_path: Path,
    asset_kind: str,
) -> None:
    paths = make_fixture(tmp_path)
    association = json.loads(paths["candidate_association"].read_text(encoding="utf-8"))
    if asset_kind == "camera":
        asset_path = Path(association["sources"]["camera_info"]["resolved_path"])
    else:
        asset_path = Path(association["sources"]["anchors"][0]["resolved_path"])
    asset_path.write_bytes(f"tampered {asset_kind}\n".encode())

    with pytest.raises(AcceptedCleanPlateMaterializationError, match="SHA-256 mismatch"):
        materialize_accepted_clean_plate_next_round(paths["acceptance"], tmp_path / "output")


def test_materializer_rejects_reused_baseline_mask_path_and_bytes(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    association = json.loads(paths["baseline_association"].read_text(encoding="utf-8"))
    first, second = association["sources"]["masks"][:2]
    second.update(
        resolved_path=first["resolved_path"],
        sha256=first["sha256"],
        bytes=first["bytes"],
        area_pixels=first["area_pixels"],
    )
    frame = association["frames"][0]
    second_assignment = next(
        item for item in frame["assignments"] if item["detection_id"] == second["detection_id"]
    )
    second_assignment["mask_sha256"] = first["sha256"]
    association["source_set_sha256"] = canonical_sha256(association["sources"])
    write_json(paths["baseline_association"], association)
    update_association_and_semantic_binding(
        paths,
        semantic_key="baseline_physical_association",
        association_path=paths["baseline_association"],
    )

    with pytest.raises(AcceptedCleanPlateMaterializationError, match="reuses a mask path"):
        materialize_accepted_clean_plate_next_round(paths["acceptance"], tmp_path / "output")


def test_materializer_supports_explicit_association_asset_path_remaps(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    script_root = Path(semantic_qa_producer.__file__).parent
    remote_data = Path("/remote/accepted-clean-plate-data")
    remote_scripts = Path("/remote/accepted-clean-plate-scripts")
    semantic = json.loads(paths["semantic"].read_text(encoding="utf-8"))
    for association_key, semantic_key in (
        ("baseline_association", "baseline_physical_association"),
        ("candidate_association", "candidate_physical_association"),
    ):
        association_path = paths[association_key]
        association = json.loads(association_path.read_text(encoding="utf-8"))
        script_path = Path(association["execution"]["script"]["path"])
        association["execution"]["script"]["path"] = str(
            remote_scripts / script_path.relative_to(script_root)
        )
        for anchor in association["sources"]["anchors"]:
            local_path = Path(anchor["resolved_path"])
            anchor["resolved_path"] = str(remote_data / local_path.relative_to(tmp_path))
        for source_key in ("camera_info", "sam3_mask_index"):
            source = association["sources"][source_key]
            local_path = Path(source["resolved_path"])
            source["resolved_path"] = str(remote_data / local_path.relative_to(tmp_path))
        for record_value in (
            *association["sources"]["frames"],
            *association["sources"]["masks"],
        ):
            local_path = Path(record_value["resolved_path"])
            record_value["resolved_path"] = str(remote_data / local_path.relative_to(tmp_path))
        association["source_set_sha256"] = canonical_sha256(association["sources"])
        write_json(association_path, association)
        semantic[semantic_key] = record(association_path)
    write_json(paths["semantic"], semantic)
    update_acceptance_asset(paths, "sam_semantic_qa_report", paths["semantic"])
    visual = json.loads(paths["visual"].read_text(encoding="utf-8"))
    visual["evidence"]["sam_semantic_qa_report"] = record(paths["semantic"])
    write_json(paths["visual"], visual)
    update_acceptance_asset(paths, "human_visual_review_receipt", paths["visual"])

    manifest, _receipt = materialize_accepted_clean_plate_next_round(
        paths["acceptance"],
        tmp_path / "output",
        association_path_remaps=[
            f"{remote_data}={tmp_path}",
            f"{remote_scripts}={script_root}",
        ],
    )

    assert manifest["status"] == "accepted_clean_plate_materialized_for_next_round_source"


def test_rejects_cumulative_mask_that_drops_current_core(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    temporal = json.loads(paths["temporal"].read_text(encoding="utf-8"))
    incomplete = np.zeros((HEIGHT, WIDTH), dtype=bool)
    incomplete[2, 3] = True
    incomplete_path = tmp_path / "incomplete-cumulative.png"
    save_mask(incomplete_path, incomplete)
    temporal["frame_records"][0]["inputs"]["cumulative_core_mask"] = record(incomplete_path)
    write_json(paths["temporal"], temporal)
    update_acceptance_asset(paths, "temporal_qa_report", paths["temporal"])

    with pytest.raises(AcceptedCleanPlateMaterializationError, match="escapes cumulative mask"):
        materialize_accepted_clean_plate_next_round(paths["acceptance"], tmp_path / "output")


def test_rejects_round1_cumulative_superset_of_current_core(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    temporal = json.loads(paths["temporal"].read_text(encoding="utf-8"))
    core_path = Path(temporal["frame_records"][0]["inputs"]["current_core_mask"]["path"])
    with Image.open(core_path) as image:
        cumulative = np.asarray(image.convert("L"), dtype=np.uint8) > 0
    cumulative = cumulative.copy()
    cumulative[0, 0] = True
    cumulative_path = tmp_path / "round1-cumulative-superset.png"
    save_mask(cumulative_path, cumulative)
    temporal["frame_records"][0]["inputs"]["cumulative_core_mask"] = record(cumulative_path)
    write_json(paths["temporal"], temporal)
    update_acceptance_asset(paths, "temporal_qa_report", paths["temporal"])

    with pytest.raises(AcceptedCleanPlateMaterializationError, match="must equal current core"):
        materialize_accepted_clean_plate_next_round(paths["acceptance"], tmp_path / "output")


def test_round2_validates_predecessor_and_materializes_exact_cumulative_union(
    tmp_path: Path,
) -> None:
    round1_paths = make_fixture(tmp_path / "round1")
    round1_output = tmp_path / "accepted-round1"
    materialize_accepted_clean_plate_next_round(round1_paths["acceptance"], round1_output)
    previous_manifest = round1_output / "next_round_source_manifest.json"
    round2_paths = make_fixture(
        tmp_path / "round2",
        round_index=2,
        previous_manifest=previous_manifest,
    )
    round2_output = tmp_path / "accepted-round2"

    manifest, _receipt = materialize_accepted_clean_plate_next_round(
        round2_paths["acceptance"], round2_output
    )

    assert manifest["round_index"] == 2
    assert manifest["next_round_index"] == 3
    previous = json.loads(previous_manifest.read_text(encoding="utf-8"))
    for current_record, previous_record in zip(
        manifest["frame_records"], previous["frame_records"], strict=True
    ):
        with Image.open(
            round2_output / current_record["accepted_round_core_mask"]["path"]
        ) as image:
            current_core = np.asarray(image.convert("L"), dtype=np.uint8) > 0
        with Image.open(
            round1_output / previous_record["previous_edited_region_mask"]["path"]
        ) as image:
            previous_cumulative = np.asarray(image.convert("L"), dtype=np.uint8) > 0
        with Image.open(
            round2_output / current_record["previous_edited_region_mask"]["path"]
        ) as image:
            materialized_cumulative = np.asarray(image.convert("L"), dtype=np.uint8) > 0
        assert np.array_equal(materialized_cumulative, previous_cumulative | current_core)


def test_round2_without_predecessor_fails_closed(tmp_path: Path) -> None:
    round1_paths = make_fixture(tmp_path / "round1")
    round1_output = tmp_path / "accepted-round1"
    materialize_accepted_clean_plate_next_round(round1_paths["acceptance"], round1_output)
    round2_paths = make_fixture(
        tmp_path / "round2",
        round_index=2,
        previous_manifest=round1_output / "next_round_source_manifest.json",
    )
    acceptance = json.loads(round2_paths["acceptance"].read_text(encoding="utf-8"))
    acceptance["previous_accepted_next_round_source_manifest"] = None
    write_json(round2_paths["acceptance"], acceptance)

    with pytest.raises(AcceptedCleanPlateMaterializationError, match="requires previous"):
        materialize_accepted_clean_plate_next_round(
            round2_paths["acceptance"], tmp_path / "round2-output"
        )


def test_round2_rejects_predecessor_without_accepted_status(tmp_path: Path) -> None:
    round1_paths = make_fixture(tmp_path / "round1")
    round1_output = tmp_path / "accepted-round1"
    materialize_accepted_clean_plate_next_round(round1_paths["acceptance"], round1_output)
    previous_manifest = round1_output / "next_round_source_manifest.json"
    round2_paths = make_fixture(
        tmp_path / "round2",
        round_index=2,
        previous_manifest=previous_manifest,
    )
    previous = json.loads(previous_manifest.read_text(encoding="utf-8"))
    previous["status"] = "pending_human_review"
    write_json(previous_manifest, previous)
    update_acceptance_asset(
        round2_paths,
        "previous_accepted_next_round_source_manifest",
        previous_manifest,
    )

    with pytest.raises(AcceptedCleanPlateMaterializationError, match="status is not accepted"):
        materialize_accepted_clean_plate_next_round(
            round2_paths["acceptance"], tmp_path / "round2-output"
        )


def test_round2_rejects_predecessor_without_next_round_only_scope(tmp_path: Path) -> None:
    round1_paths = make_fixture(tmp_path / "round1")
    round1_output = tmp_path / "accepted-round1"
    materialize_accepted_clean_plate_next_round(round1_paths["acceptance"], round1_output)
    previous_manifest = round1_output / "next_round_source_manifest.json"
    round2_paths = make_fixture(
        tmp_path / "round2",
        round_index=2,
        previous_manifest=previous_manifest,
    )
    previous = json.loads(previous_manifest.read_text(encoding="utf-8"))
    del previous["acceptance_scope"]
    write_json(previous_manifest, previous)
    update_acceptance_asset(
        round2_paths,
        "previous_accepted_next_round_source_manifest",
        previous_manifest,
    )

    with pytest.raises(AcceptedCleanPlateMaterializationError, match="acceptance_scope"):
        materialize_accepted_clean_plate_next_round(
            round2_paths["acceptance"], tmp_path / "round2-output"
        )


def test_round2_rejects_predecessor_that_claims_corrected_full_pipeline(tmp_path: Path) -> None:
    round1_paths = make_fixture(tmp_path / "round1")
    round1_output = tmp_path / "accepted-round1"
    materialize_accepted_clean_plate_next_round(round1_paths["acceptance"], round1_output)
    previous_manifest = round1_output / "next_round_source_manifest.json"
    round2_paths = make_fixture(
        tmp_path / "round2",
        round_index=2,
        previous_manifest=previous_manifest,
    )
    previous = json.loads(previous_manifest.read_text(encoding="utf-8"))
    previous["corrected_full_pipeline"] = True
    write_json(previous_manifest, previous)
    update_acceptance_asset(
        round2_paths,
        "previous_accepted_next_round_source_manifest",
        previous_manifest,
    )

    with pytest.raises(AcceptedCleanPlateMaterializationError, match="corrected full-pipeline"):
        materialize_accepted_clean_plate_next_round(
            round2_paths["acceptance"], tmp_path / "round2-output"
        )


def test_round2_rejects_predecessor_source_sha_not_used_by_current_selection(
    tmp_path: Path,
) -> None:
    round1_paths = make_fixture(tmp_path / "round1")
    round1_output = tmp_path / "accepted-round1"
    materialize_accepted_clean_plate_next_round(round1_paths["acceptance"], round1_output)
    previous_manifest = round1_output / "next_round_source_manifest.json"
    round2_paths = make_fixture(
        tmp_path / "round2",
        round_index=2,
        previous_manifest=previous_manifest,
    )
    previous = json.loads(previous_manifest.read_text(encoding="utf-8"))
    wrong_source = previous["frame_records"][1]["source_rgb"]
    previous["frame_records"][0]["source_rgb"] = wrong_source
    previous["frame_records"][0]["segmentation_source_rgb_sha256"] = wrong_source["sha256"]
    write_json(previous_manifest, previous)
    update_acceptance_asset(
        round2_paths,
        "previous_accepted_next_round_source_manifest",
        previous_manifest,
    )

    with pytest.raises(
        AcceptedCleanPlateMaterializationError,
        match="current selection source RGB",
    ):
        materialize_accepted_clean_plate_next_round(
            round2_paths["acceptance"], tmp_path / "round2-output"
        )


def test_round2_rejects_current_only_cumulative_instead_of_predecessor_union(
    tmp_path: Path,
) -> None:
    round1_paths = make_fixture(tmp_path / "round1")
    round1_output = tmp_path / "accepted-round1"
    materialize_accepted_clean_plate_next_round(round1_paths["acceptance"], round1_output)
    round2_paths = make_fixture(
        tmp_path / "round2",
        round_index=2,
        previous_manifest=round1_output / "next_round_source_manifest.json",
    )
    temporal = json.loads(round2_paths["temporal"].read_text(encoding="utf-8"))
    temporal["frame_records"][0]["inputs"]["cumulative_core_mask"] = temporal["frame_records"][0][
        "inputs"
    ]["current_core_mask"]
    write_json(round2_paths["temporal"], temporal)
    update_acceptance_asset(round2_paths, "temporal_qa_report", round2_paths["temporal"])

    with pytest.raises(AcceptedCleanPlateMaterializationError, match="predecessor OR current core"):
        materialize_accepted_clean_plate_next_round(
            round2_paths["acceptance"], tmp_path / "round2-output"
        )


def test_rejects_boundary_asset_from_another_frame_set(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    boundary = json.loads(paths["boundary"].read_text(encoding="utf-8"))
    wrong = boundary["frame_records"][0]["inputs"]
    wrong["composite_rgb"] = wrong["removal_core_mask"]
    write_json(paths["boundary"], boundary)
    update_acceptance_asset(paths, "boundary_qa_report", paths["boundary"])

    with pytest.raises(AcceptedCleanPlateMaterializationError, match="selected asset"):
        materialize_accepted_clean_plate_next_round(paths["acceptance"], tmp_path / "output")


def test_rejects_boundary_report_with_nonofficial_gate_set(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    boundary = json.loads(paths["boundary"].read_text(encoding="utf-8"))
    boundary["frame_records"][0]["gates"] = {
        "fake_gate": {"observed": 1.0, "threshold": 1.0, "passed": True}
    }
    write_json(paths["boundary"], boundary)
    update_acceptance_asset(paths, "boundary_qa_report", paths["boundary"])

    with pytest.raises(AcceptedCleanPlateMaterializationError, match="keys differ"):
        materialize_accepted_clean_plate_next_round(paths["acceptance"], tmp_path / "output")


def test_rejects_boundary_gate_not_closed_to_report_metric(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    boundary = json.loads(paths["boundary"].read_text(encoding="utf-8"))
    boundary["frame_records"][0]["alignment"]["mask_iou"] = 0.95
    write_json(paths["boundary"], boundary)
    update_acceptance_asset(paths, "boundary_qa_report", paths["boundary"])

    with pytest.raises(AcceptedCleanPlateMaterializationError, match="does not close"):
        materialize_accepted_clean_plate_next_round(paths["acceptance"], tmp_path / "output")


def test_rejects_selection_output_that_escapes_selection_root(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    selection_root = paths["selection"].parent
    original = selection_root / selection["frames"][0]["outputs"]["composite_rgb"]["path"]
    escaped = tmp_path / "escaped-copy.png"
    escaped.write_bytes(original.read_bytes())
    selection["frames"][0]["outputs"]["composite_rgb"] = record(escaped)
    write_json(paths["selection"], selection)
    update_acceptance_asset(paths, "candidate_selection_receipt", paths["selection"])

    with pytest.raises(AcceptedCleanPlateMaterializationError, match="escapes required"):
        materialize_accepted_clean_plate_next_round(paths["acceptance"], tmp_path / "output")


def test_refuses_canonical_destination_and_existing_output(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    canonical_target = tmp_path / "web" / "public" / "worlds" / "new-candidate"
    with pytest.raises(AcceptedCleanPlateMaterializationError, match="canonical/live"):
        materialize_accepted_clean_plate_next_round(paths["acceptance"], canonical_target)

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(AcceptedCleanPlateMaterializationError, match="refusing to overwrite"):
        materialize_accepted_clean_plate_next_round(paths["acceptance"], existing)


def test_copy_failure_cleans_same_parent_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = make_fixture(tmp_path)
    output = tmp_path / "output"
    original_copy = materializer.copy_verified
    calls = 0

    def fail_during_copy(source: Any, destination: Path) -> Any:
        nonlocal calls
        calls += 1
        copied = original_copy(source, destination)
        if calls == 3:
            raise AcceptedCleanPlateMaterializationError("injected copy failure")
        return copied

    monkeypatch.setattr(materializer, "copy_verified", fail_during_copy)
    with pytest.raises(AcceptedCleanPlateMaterializationError, match="injected copy failure"):
        materialize_accepted_clean_plate_next_round(paths["acceptance"], output)
    assert not output.exists()
    assert not list(tmp_path.glob(".output.staging-*"))


def test_rejects_temporal_counts_without_complete_pair_records(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    temporal = json.loads(paths["temporal"].read_text(encoding="utf-8"))
    temporal["pair_records"].pop()
    write_json(paths["temporal"], temporal)
    update_acceptance_asset(paths, "temporal_qa_report", paths["temporal"])

    with pytest.raises(AcceptedCleanPlateMaterializationError, match="must contain 48"):
        materialize_accepted_clean_plate_next_round(paths["acceptance"], tmp_path / "output")


def test_rejects_temporal_report_with_nonofficial_pair_gate_set(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    temporal = json.loads(paths["temporal"].read_text(encoding="utf-8"))
    temporal["pair_records"][0]["gates"] = {
        "fake_gate": {"observed": 1.0, "threshold": 1.0, "passed": True}
    }
    write_json(paths["temporal"], temporal)
    update_acceptance_asset(paths, "temporal_qa_report", paths["temporal"])

    with pytest.raises(AcceptedCleanPlateMaterializationError, match="keys differ"):
        materialize_accepted_clean_plate_next_round(paths["acceptance"], tmp_path / "output")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda report: report["pairing"].update(mode="forward_only"),
            "pairing contract",
        ),
        (
            lambda report: report["flow_backend"].update(repository_commit="0" * 40),
            "repository commit is not audited",
        ),
        (
            lambda report: report["flow_backend"]["checkpoint"].update(sha256="0" * 64),
            "checkpoint is not audited",
        ),
        (
            lambda report: report["flow_backend"]["checkpoint"].update(bytes=1),
            "checkpoint byte size is not audited",
        ),
        (
            lambda report: report["sampling"].update(core_erosion_px=3),
            "sampling.core_erosion_px differs",
        ),
        (
            lambda report: report["thresholds"].update(maximum_core_color_p95_abs_rgb_delta=25.0),
            "thresholds.maximum_core_color_p95_abs_rgb_delta differs",
        ),
    ],
)
def test_rejects_temporal_report_outside_fixed_audited_contract(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    paths = make_fixture(tmp_path)
    temporal = json.loads(paths["temporal"].read_text(encoding="utf-8"))
    mutation(temporal)
    write_json(paths["temporal"], temporal)
    update_acceptance_asset(paths, "temporal_qa_report", paths["temporal"])

    with pytest.raises(AcceptedCleanPlateMaterializationError, match=message):
        materialize_accepted_clean_plate_next_round(paths["acceptance"], tmp_path / "output")


def test_rejects_temporal_gate_observed_not_closed_to_pair_metric(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    temporal = json.loads(paths["temporal"].read_text(encoding="utf-8"))
    temporal["pair_records"][0]["metrics"]["core_color_p95_abs_rgb_delta"] = 1.0
    write_json(paths["temporal"], temporal)
    update_acceptance_asset(paths, "temporal_qa_report", paths["temporal"])

    with pytest.raises(AcceptedCleanPlateMaterializationError, match="does not close"):
        materialize_accepted_clean_plate_next_round(paths["acceptance"], tmp_path / "output")


def test_rejects_temporal_pair_with_failed_nested_gate(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    temporal = json.loads(paths["temporal"].read_text(encoding="utf-8"))
    temporal["pair_records"][0]["metrics"]["core_color_p95_abs_rgb_delta"] = 100.0
    temporal["pair_records"][0]["gates"]["core_color_p95_lte"].update(
        observed=100.0,
        passed=False,
    )
    write_json(paths["temporal"], temporal)
    update_acceptance_asset(paths, "temporal_qa_report", paths["temporal"])

    with pytest.raises(AcceptedCleanPlateMaterializationError, match="did not explicitly pass"):
        materialize_accepted_clean_plate_next_round(paths["acceptance"], tmp_path / "output")


def test_rejects_aggregate_visual_pass_with_one_failed_frame(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    visual = json.loads(paths["visual"].read_text(encoding="utf-8"))
    visual["frame_records"][7]["passed"] = False
    write_json(paths["visual"], visual)
    update_acceptance_asset(paths, "human_visual_review_receipt", paths["visual"])

    with pytest.raises(
        AcceptedCleanPlateMaterializationError,
        match=r"visual frame .* did not pass",
    ):
        materialize_accepted_clean_plate_next_round(paths["acceptance"], tmp_path / "output")


def test_missing_semantic_receipt_fails_closed(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    paths["semantic"].unlink()

    with pytest.raises(AcceptedCleanPlateMaterializationError, match="does not exist"):
        materialize_accepted_clean_plate_next_round(paths["acceptance"], tmp_path / "output")


def test_materialized_state_can_feed_round2_source_aligned_mask_builder(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    accepted_output = tmp_path / "accepted-r1"
    manifest, _ = materialize_accepted_clean_plate_next_round(paths["acceptance"], accepted_output)
    target_label = manifest["contributor_label_registry"][REMAINING_ID]
    next_round_records: list[dict[str, Any]] = []
    for frame in manifest["frame_records"]:
        observed = np.zeros((HEIGHT, WIDTH), dtype=bool)
        observed[0, 1] = True
        observed_path = tmp_path / "next-round-observed" / f"{frame['frame_id']}.png"
        save_mask(observed_path, observed)
        source_path = accepted_output / frame["source_rgb"]["path"]
        previous_region_path = accepted_output / frame["previous_edited_region_mask"]["path"]
        previous_labels_path = accepted_output / frame["previous_contributor_labels"]["path"]
        next_round_records.append(
            {
                "sequence_index": frame["sequence_index"],
                "frame_id": frame["frame_id"],
                "target_label": target_label,
                "exact_source_rgb": record(source_path),
                "observed_target_mask": {
                    **record(observed_path),
                    "source_rgb_sha256": frame["source_rgb"]["sha256"],
                },
                "previous_edited_region_mask": record(previous_region_path),
                "previous_contributor_labels": record(previous_labels_path),
            }
        )
    builder_input = tmp_path / "round2-source-aligned-mask-input.json"
    write_json(
        builder_input,
        {
            "schema_version": 1,
            "kind": scene_edit_builder.INPUT_KIND,
            "round_index": 2,
            "frame_records": next_round_records,
        },
    )

    report = scene_edit_builder.build_scene_edit_masks(
        argparse.Namespace(
            input_manifest=builder_input,
            output=tmp_path / "round2-scene-edit-masks",
        )
    )

    assert report["status"] == "technical_passed"
    assert report["frame_count"] == EXPECTED_FRAME_COUNT
    assert report["gates"]["all_previous_input_pairs_valid_for_round"] is True


def test_cli_materializes_reviewed_next_round_package(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    output = tmp_path / "cli-output"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/materialize_accepted_clean_plate_next_round.py",
            "--manifest",
            str(paths["acceptance"]),
            "--output-dir",
            str(output),
        ],
        cwd=Path(__file__).parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["status"] == "accepted_clean_plate_materialized_for_next_round_source"
    assert summary["frame_count"] == EXPECTED_FRAME_COUNT
    assert (output / "receipt.json").is_file()
