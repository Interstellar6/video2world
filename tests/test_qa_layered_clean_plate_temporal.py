from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

import scripts.qa_layered_clean_plate_temporal as temporal

TEST_RESOLUTION = (192, 128)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def asset(path: Path, root: Path, *, mask: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path.relative_to(root)),
        "sha256": temporal.sha256_file(path),
    }
    if mask:
        result.update({"channel": "luma", "threshold": 128})
    return result


class FakeFlowProvider:
    def __init__(self, resolution: tuple[int, int], *, inconsistent: bool = False):
        self.resolution = resolution
        self.inconsistent = inconsistent
        self.calls = 0
        self.metadata = {
            "kind": "fake_plain_raft",
            "model_load_count": 1,
            "test_double": True,
            "precision": "float32",
            "iterations": 20,
            "flow_completion": False,
        }

    def estimate(self, source_rgb: np.ndarray, target_rgb: np.ndarray) -> np.ndarray:
        assert source_rgb.shape[:2] == (self.resolution[1], self.resolution[0])
        assert target_rgb.shape == source_rgb.shape
        flow = np.zeros((self.resolution[1], self.resolution[0], 2), dtype=np.float32)
        if self.inconsistent and self.calls % 2 == 0:
            flow[..., 0] = 20.0
        self.calls += 1
        return flow


def test_writable_c_contiguous_rgb_copies_read_only_input() -> None:
    source = np.zeros((3, 4, 3), dtype=np.uint8)
    source.setflags(write=False)

    prepared = temporal.writable_c_contiguous_rgb(source)

    assert prepared.flags.c_contiguous
    assert prepared.flags.writeable
    assert not np.shares_memory(prepared, source)
    prepared[0, 0] = 255


def make_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    flicker_core: bool = False,
    flat_control: bool = False,
    bad_hash: bool = False,
) -> dict[str, Any]:
    root = tmp_path / "fixture"
    assets = root / "assets"
    assets.mkdir(parents=True)
    monkeypatch.setattr(temporal, "EXPECTED_RESOLUTION", TEST_RESOLUTION)

    checkpoint = assets / "raft-things.pth"
    checkpoint.write_bytes(b"test-only-plain-raft-checkpoint")
    checkpoint_sha = temporal.sha256_file(checkpoint)
    monkeypatch.setattr(temporal, "EXPECTED_RAFT_CHECKPOINT_SHA256", checkpoint_sha)
    raft_repo = root / "plain-raft"
    raft_repo.mkdir()

    width, height = TEST_RESOLUTION
    rows, columns = np.indices((height, width))
    if flat_control:
        base = np.full((height, width, 3), 80, dtype=np.uint8)
    else:
        base = np.stack(
            (
                20 + (3 * columns + 5 * rows) % 180,
                25 + (7 * columns + 2 * rows) % 175,
                30 + (5 * columns + 3 * rows) % 170,
            ),
            axis=2,
        ).astype(np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[24:104, 48:144] = 255
    core_path = assets / "core.png"
    cumulative_path = assets / "cumulative.png"
    editable_path = assets / "editable.png"
    Image.fromarray(mask).save(core_path)
    Image.fromarray(mask).save(cumulative_path)
    Image.fromarray(mask).save(editable_path)

    frame_ids = [f"{48 + index:06d}" for index in range(temporal.EXPECTED_FRAME_COUNT)]
    frame_assets: list[dict[str, Any]] = []
    selection_frames: list[dict[str, Any]] = []
    for index, frame_id in enumerate(frame_ids):
        source_path = assets / f"source-{index:04d}.png"
        composite_path = assets / f"composite-{index:04d}.png"
        composite = base.copy()
        if flicker_core and index % 2 == 1:
            values = composite.astype(np.int16)
            values[mask > 0] += 60
            composite = np.clip(values, 0, 255).astype(np.uint8)
        Image.fromarray(base).save(source_path)
        Image.fromarray(composite).save(composite_path)
        source_record = asset(source_path, root)
        composite_record = asset(composite_path, root)
        core_record = asset(core_path, root, mask=True)
        cumulative_record = asset(cumulative_path, root, mask=True)
        editable_record = asset(editable_path, root, mask=True)
        frame_assets.append(
            {
                "sequence_index": index,
                "frame_id": frame_id,
                "selected_composite_rgb": composite_record,
                "source_rgb": source_record,
                "current_core_mask": core_record,
                "cumulative_core_mask": cumulative_record,
                "editable_mask": editable_record,
            }
        )
        selection_frames.append(
            {
                "sequence_index": index,
                "frame_id": frame_id,
                "source_rgb": source_record,
                "outputs": {
                    "composite_rgb": composite_record,
                    "core_mask": core_record,
                    "editable_mask": editable_record,
                },
                "exactness": {
                    "passed": True,
                    "outside_editable_rgb_exact_recomputed": True,
                    "outside_editable_changed_pixels_recomputed": 0,
                },
            }
        )

    selection_path = root / "selection.json"
    write_json(
        selection_path,
        {
            "schema_version": 1,
            "kind": temporal.SELECTION_KIND,
            "status": "review_only_candidates_selected",
            "review_only": True,
            "ordered_frame_ids": frame_ids,
            "frames": selection_frames,
            "aggregate": {"frame_set_sha256": temporal.canonical_sha256(frame_ids)},
        },
    )
    selection_asset = asset(selection_path, root)
    for record in frame_assets:
        record["selection_receipt"] = selection_asset

    boundary_path = root / "boundary-report.json"
    write_json(
        boundary_path,
        {
            "schema_version": 1,
            "kind": temporal.BOUNDARY_REPORT_KIND,
            "status": "technical_passed",
            "boundary_texture_gate_passed": True,
            "promotion_scope": "boundary_and_local_texture_only",
            "promotion_approved": False,
            "frame_count": temporal.EXPECTED_FRAME_COUNT,
            "frame_records": [
                {
                    "sequence_index": frame["sequence_index"],
                    "frame_id": frame["frame_id"],
                    "inputs": {
                        "composite_rgb": frame["selected_composite_rgb"],
                        "removal_core_mask": frame["current_core_mask"],
                        "editable_mask": frame["editable_mask"],
                    },
                }
                for frame in frame_assets
            ],
        },
    )
    manifest = {
        "schema_version": 1,
        "kind": temporal.MANIFEST_KIND,
        "round_index": 1,
        "frame_count": temporal.EXPECTED_FRAME_COUNT,
        "selection_manifest": selection_asset,
        "upstream_boundary_report": asset(boundary_path, root),
        "pairing": {
            "mode": "adjacent_bidirectional",
            "frame_stride": 1,
            "undirected_pair_count": 24,
            "directed_pair_count": 48,
            "triplet_count": 23,
        },
        "flow_backend": {
            "kind": "plain_raft",
            "repository_path": str(raft_repo.relative_to(root)),
            "repository_commit": temporal.EXPECTED_RAFT_COMMIT,
            "checkpoint": asset(checkpoint, root),
            "precision": "float32",
            "iterations": 20,
            "flow_completion": False,
            "inference_resolution": list(TEST_RESOLUTION),
            "device": "cuda:0",
        },
        "sampling": dict(temporal.REFERENCE_SAMPLING),
        "thresholds": dict(temporal.REFERENCE_THRESHOLDS),
        "frame_records": frame_assets,
    }
    if bad_hash:
        manifest["frame_records"][7]["selected_composite_rgb"]["sha256"] = "0" * 64
    manifest_path = root / "manifest.json"
    write_json(manifest_path, manifest)
    return {
        "root": root,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "report_path": root / "qa" / "temporal-report.json",
        "artifacts_dir": root / "qa" / "artifacts",
    }


def args_for(fixture: dict[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(
        input_manifest=fixture["manifest_path"],
        output_report=fixture["report_path"],
        artifacts_dir=fixture["artifacts_dir"],
    )


def run_with_fake(
    fixture: dict[str, Any],
    *,
    inconsistent: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state: dict[str, Any] = {"factory_calls": 0}

    def factory(_config: dict[str, Any]) -> FakeFlowProvider:
        state["factory_calls"] += 1
        provider = FakeFlowProvider(TEST_RESOLUTION, inconsistent=inconsistent)
        state["provider"] = provider
        return provider

    report = temporal.evaluate(args_for(fixture), flow_provider_factory=factory)
    return report, state


def test_exact_static_sequence_passes_temporal_only_with_one_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)

    report, state = run_with_fake(fixture)

    assert report["status"] == "technical_passed_temporal_only"
    assert report["promotion_approved"] is False
    assert report["review_only"] is True
    assert state["factory_calls"] == 1
    assert state["provider"].calls == 48
    assert len(report["pair_records"]) == 48
    assert len(report["triplet_records"]) == 23
    assert report["counts"]["failed_directed_pairs"] == 0
    assert report["counts"]["failed_triplets"] == 0
    assert (
        report["next_action"]["action"]
        == "run_semantic_residual_pbr_da3_and_cross_view_review_before_any_promotion"
    )
    assert report["next_action"]["failed_pair_ids"] == []
    assert report["next_action"]["not_evaluable_pair_ids"] == []
    first_pair = report["pair_records"][0]
    assert first_pair["coverage"]["target_core_overlap_fraction"] == 1.0
    assert first_pair["coverage"]["valid_core_fraction"] == 1.0
    assert first_pair["exposure_fit"]["sample_count"] >= 4096
    assert first_pair["metrics"]["core_color_p95_abs_rgb_delta"] == pytest.approx(0.0)
    assert report["triplet_records"][0]["metrics"][
        "core_second_difference_p95_abs_rgb_delta"
    ] == pytest.approx(0.0)
    assert report["artifacts"]["pair_artifact_count"] == 48
    assert report["artifacts"]["triplet_artifact_count"] == 23
    assert Path(first_pair["artifacts"]["flow_validity_and_color_error"]["path"]).is_file()
    assert report["forbidden_claims"]["pbr_used_as_pass_evidence"] is False
    assert report["forbidden_claims"]["da3_used_as_pass_evidence"] is False
    assert report["forbidden_claims"]["clean_plate_promotion_claimed"] is False


def test_core_flicker_fails_pairs_and_triplets_without_becoming_not_evaluable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch, flicker_core=True)

    report, _state = run_with_fake(fixture)

    assert report["status"] == "technical_failed"
    assert report["counts"]["not_evaluable_directed_pairs"] == 0
    assert report["counts"]["not_evaluable_triplets"] == 0
    assert report["counts"]["failed_directed_pairs"] > 0
    assert report["counts"]["failed_triplets"] > 0
    assert (
        report["next_action"]["action"]
        == "repair_temporal_flicker_or_regenerate_clean_plate_candidates"
    )
    assert len(report["next_action"]["failed_pair_ids"]) == report["counts"][
        "failed_directed_pairs"
    ]
    assert len(report["next_action"]["failed_triplet_center_frame_ids"]) == report["counts"][
        "failed_triplets"
    ]
    failing_pair = next(record for record in report["pair_records"] if not record["passed"])
    assert failing_pair["gates"]["core_color_p95_lte"]["passed"] is False
    assert failing_pair["gates"]["core_bad_pixel_fraction_lte"]["passed"] is False
    pair_action = next(
        item
        for item in report["next_action"]["failed_pairs"]
        if item["direction_id"] == failing_pair["direction_id"]
    )
    assert "core_color_p95_lte" in pair_action["failed_gates"]
    assert "core_bad_pixel_fraction_lte" in pair_action["failed_gates"]
    failing_triplet = next(record for record in report["triplet_records"] if not record["passed"])
    assert failing_triplet["gates"]["triplet_second_difference_p95_lte"]["passed"] is False
    assert failing_triplet["center_frame_id"] in report["next_action"][
        "failed_triplet_center_frame_ids"
    ]


def test_ill_conditioned_control_ring_is_not_evaluable_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch, flat_control=True)

    report, state = run_with_fake(fixture)

    assert report["status"] == "not_evaluable"
    assert state["factory_calls"] == 1
    assert report["counts"]["not_evaluable_directed_pairs"] == 48
    assert report["counts"]["not_evaluable_triplets"] == 23
    assert "ill-conditioned" in report["pair_records"][0]["not_evaluable_reason"]
    assert report["next_action"]["action"] == "repair_temporal_evidence_before_retesting"
    assert len(report["next_action"]["not_evaluable_pair_ids"]) == 48
    assert "ill-conditioned" in report["next_action"]["not_evaluable_pairs"][0]["reason"]
    assert report["promotion_approved"] is False


def test_fb_inconsistent_flow_is_not_evaluable_and_records_target_overlap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)

    report, state = run_with_fake(fixture, inconsistent=True)

    assert state["provider"].calls == 48
    assert report["status"] == "not_evaluable"
    first_pair = report["pair_records"][0]
    assert first_pair["coverage"]["target_core_overlap_fraction"] < 1.0
    assert first_pair["coverage"]["valid_core_fraction"] == 0.0
    assert "valid core has 0 pixels" in first_pair["not_evaluable_reason"]


def test_hash_mismatch_is_invalid_input_and_never_loads_flow_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch, bad_hash=True)
    state = {"factory_calls": 0}

    def factory(_config: dict[str, Any]) -> FakeFlowProvider:
        state["factory_calls"] += 1
        return FakeFlowProvider(TEST_RESOLUTION)

    report = temporal.evaluate(args_for(fixture), flow_provider_factory=factory)

    assert report["status"] == "invalid_input"
    assert "sha256 does not match" in report["error"]["message"]
    assert state["factory_calls"] == 0
    assert not fixture["artifacts_dir"].exists()
    assert fixture["report_path"].is_file()


def test_wrong_frame_order_and_weakened_thresholds_are_invalid_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    fixture["manifest"]["frame_records"][3]["sequence_index"] = 4
    write_json(fixture["manifest_path"], fixture["manifest"])

    report, state = run_with_fake(fixture)

    assert report["status"] == "invalid_input"
    assert "contiguous and ordered" in report["error"]["message"]
    assert state["factory_calls"] == 0

    fixture = make_fixture(tmp_path / "weak", monkeypatch)
    fixture["manifest"]["thresholds"]["minimum_core_valid_fraction_per_direction"] = 0.6
    write_json(fixture["manifest_path"], fixture["manifest"])
    report, state = run_with_fake(fixture)
    assert report["status"] == "invalid_input"
    assert "must be at least 0.7" in report["error"]["message"]
    assert state["factory_calls"] == 0


def test_selection_exactness_and_upstream_boundary_are_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    selection_path = fixture["root"] / fixture["manifest"]["selection_manifest"]["path"]
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["frames"][0]["exactness"]["passed"] = False
    write_json(selection_path, selection)
    new_hash = temporal.sha256_file(selection_path)
    fixture["manifest"]["selection_manifest"]["sha256"] = new_hash
    for frame in fixture["manifest"]["frame_records"]:
        frame["selection_receipt"]["sha256"] = new_hash
    write_json(fixture["manifest_path"], fixture["manifest"])

    report, state = run_with_fake(fixture)

    assert report["status"] == "invalid_input"
    assert "selection exactness did not pass" in report["error"]["message"]
    assert state["factory_calls"] == 0


def test_upstream_boundary_failure_surfaces_frame_next_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    root = fixture["root"]
    boundary_record = fixture["manifest"]["upstream_boundary_report"]
    boundary_path = root / boundary_record["path"]
    boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
    boundary["status"] = "technical_failed"
    boundary["boundary_texture_gate_passed"] = False
    boundary["next_action"] = {
        "action": "repair_removal_mask_or_target_matte_before_regeneration",
        "blocking_gate_groups": ["mask_alignment"],
        "failed_gates": ["boundary_distance_p95_lte"],
        "failed_frame_ids": ["000064"],
        "first_failed_frame_id": "000064",
        "failed_frame_gates": [
            {
                "sequence_index": 16,
                "frame_id": "000064",
                "failed_gates": ["boundary_distance_p95_lte"],
            }
        ],
    }
    write_json(boundary_path, boundary)
    boundary_record["sha256"] = temporal.sha256_file(boundary_path)
    write_json(fixture["manifest_path"], fixture["manifest"])

    report, state = run_with_fake(fixture)

    assert report["status"] == "invalid_input"
    message = report["error"]["message"]
    assert "upstream boundary report did not technically pass" in message
    assert "status=technical_failed" in message
    assert "boundary_texture_gate_passed=false" in message
    assert "action=repair_removal_mask_or_target_matte_before_regeneration" in message
    assert "blocking_gate_groups=mask_alignment" in message
    assert "failed_frame_ids=000064" in message
    assert "first_failed_frame_id=000064" in message
    assert "failed_frame_gates=000064:boundary_distance_p95_lte" in message
    assert state["factory_calls"] == 0


def test_bilinear_sampling_and_fb_formula_are_directional() -> None:
    values = np.arange(16, dtype=np.float64).reshape(4, 4)
    x = np.asarray([[0.5, 2.0]])
    y = np.asarray([[0.5, 1.0]])
    sampled = temporal.bilinear_sample(values, x, y)

    assert sampled.tolist() == [[2.5, 6.0]]
