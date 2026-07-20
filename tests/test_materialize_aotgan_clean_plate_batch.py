from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from scripts.build_layered_clean_plate_boundary_qa_manifest import build_manifest
from scripts.materialize_aotgan_clean_plate_batch import (
    OUTPUT_FRAME_RECEIPT_KIND,
    OUTPUT_RECEIPT_KIND,
    AOTBatchMaterializationError,
    materialize_aotgan_clean_plate_batch,
    sha256_file,
)
from scripts.materialize_clean_plate_candidate_selection import (
    materialize_clean_plate_candidate_selection,
)

FRAME_IDS = tuple(f"{index:06d}" for index in range(48, 73))
HEIGHT = 12
WIDTH = 16


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    digest, size = sha256_file(path)
    return {
        "path": (
            path.relative_to(relative_to).as_posix() if relative_to is not None else str(path)
        ),
        "sha256": digest,
        "bytes": size,
    }


def save_rgb(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values.astype(np.uint8)).save(path)


def save_luma(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values.astype(np.uint8)).save(path)


def make_fixture(
    tmp_path: Path,
    *,
    corrupt_composite: bool = False,
    translucent_core: bool = False,
) -> Path:
    adaptive_root = tmp_path / "adaptive"
    adaptive_report_path = adaptive_root / "output" / "report.json"
    adaptive_manifest_path = adaptive_root / "output" / "artifact_manifest.json"
    write_json(adaptive_report_path, {"kind": "test.adaptive.report"})
    write_json(adaptive_manifest_path, {"kind": "test.adaptive.manifest"})
    adaptive_receipt_path = adaptive_root / "final_receipt.json"
    write_json(
        adaptive_receipt_path,
        {
            "status": "technical_passed_mask_refinement_only_not_clean_plate",
            "report": record(adaptive_report_path, relative_to=adaptive_root),
            "artifact_manifest": record(adaptive_manifest_path, relative_to=adaptive_root),
        },
    )

    aot_root = tmp_path / "aot-output"
    model_weight = tmp_path / "models" / "G0000000.pt"
    model_weight.parent.mkdir(parents=True)
    model_weight.write_bytes(b"test-aot-weight")
    runner = tmp_path / "tools" / "run_aot.py"
    runner.parent.mkdir(parents=True)
    runner.write_text("# pinned test runner\n", encoding="utf-8")
    frames: list[dict[str, Any]] = []
    output_artifacts: list[dict[str, Any]] = []
    for sequence_index, frame_id in enumerate(FRAME_IDS):
        y, x = np.indices((HEIGHT, WIDTH))
        source = np.stack(
            [
                20 + x * 2 + sequence_index,
                40 + y * 3,
                60 + ((x + y) % 5),
            ],
            axis=2,
        ).astype(np.uint8)
        prediction = np.roll(source, shift=2, axis=1)
        alpha = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        alpha[2:10, 3:13] = 128
        alpha[4:8, 5:11] = 255
        core = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        core[4:8, 5:11] = 255
        if translucent_core:
            alpha[core > 0] = 128
        physical = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        physical[3:9, 4:12] = 255
        collar = ((alpha > 0) & (core == 0)).astype(np.uint8) * 255
        composite = (
            source.astype(np.uint16) * (255 - alpha.astype(np.uint16)[..., None])
            + prediction.astype(np.uint16) * alpha.astype(np.uint16)[..., None]
            + 127
        ) // 255
        composite = composite.astype(np.uint8)
        if corrupt_composite and sequence_index == 3:
            composite[5, 7, 0] ^= 1

        adaptive_frame = adaptive_root / "output" / "frames" / frame_id
        source_path = adaptive_frame / "bound_inputs" / "source.png"
        physical_path = adaptive_frame / "bound_inputs" / "physical_sam_mask.png"
        alpha_path = adaptive_frame / "removal_alpha.png"
        core_path = adaptive_frame / "guaranteed_foreground_core.png"
        collar_path = adaptive_frame / "uncertain_feather_collar.png"
        save_rgb(source_path, source)
        save_luma(physical_path, physical)
        save_luma(alpha_path, alpha)
        save_luma(core_path, core)
        save_luma(collar_path, collar)

        output_frame = aot_root / "frames" / frame_id
        prediction_path = output_frame / "raw_prediction.png"
        composite_path = output_frame / "adaptive_alpha_composite.png"
        save_rgb(prediction_path, prediction)
        save_rgb(composite_path, composite)
        prediction_record = record(prediction_path, relative_to=aot_root)
        composite_record = record(composite_path, relative_to=aot_root)
        output_artifacts.extend((prediction_record, composite_record))
        frames.append(
            {
                "frame_id": frame_id,
                "source": record(source_path),
                "adaptive_alpha": record(alpha_path),
                "physical_sam_mask": record(physical_path),
                "guaranteed_core": record(core_path),
                "uncertain_collar": record(collar_path),
                "raw_prediction": prediction_record,
                "composite": composite_record,
                "duration_seconds": 0.01,
                "adaptive_alpha_nonzero_pixels": int(np.count_nonzero(alpha)),
                "outside_alpha_zero_changed_values": 0,
            }
        )

    report_path = aot_root / "report.json"
    report = {
        "schema_version": 1,
        "kind": "video2world.aotgan_fixed25_adaptive_alpha_batch",
        "status": "generated_fixed25_pending_temporal_and_3d_consistency_review",
        "promotion_approved": False,
        "model": {
            "source_commit": "2" * 40,
            "weight": record(model_weight),
        },
        "upstream_adaptive_mask": {
            "receipt": record(adaptive_receipt_path),
            "report": record(adaptive_report_path),
            "artifact_manifest": record(adaptive_manifest_path),
        },
        "fixed25": {
            "frame_ids": list(FRAME_IDS),
            "frame_count": len(FRAME_IDS),
            "outside_alpha_zero_changed_values_total": 0,
        },
        "frames": frames,
    }
    write_json(report_path, report)
    output_artifacts.append(record(report_path, relative_to=aot_root))
    artifact_manifest_path = aot_root / "artifact_manifest.json"
    write_json(
        artifact_manifest_path,
        {"schema_version": 1, "files": output_artifacts},
    )
    final_receipt_path = aot_root / "final_receipt.json"
    write_json(
        final_receipt_path,
        {
            "schema_version": 1,
            "kind": "video2world.aotgan_fixed25_batch_receipt",
            "status": "generated_fixed25_pending_temporal_and_3d_consistency_review",
            "promotion_approved": False,
            "r2_allowed": False,
            "report": record(report_path, relative_to=aot_root),
            "artifact_manifest": record(artifact_manifest_path, relative_to=aot_root),
            "runner": record(runner),
        },
    )
    return final_receipt_path


def test_normalizes_real_aot_contract_and_feeds_shared_selection_and_boundary(
    tmp_path: Path,
) -> None:
    source_receipt = make_fixture(tmp_path)
    normalized = tmp_path / "normalized"

    receipt = materialize_aotgan_clean_plate_batch(
        source_receipt,
        normalized,
        expected_source_receipt_sha256=sha256_file(source_receipt)[0],
        created_at=datetime(2026, 7, 20, 10, 0, tzinfo=UTC),
    )

    assert receipt["kind"] == OUTPUT_RECEIPT_KIND
    assert receipt["promotion_allowed"] is False
    assert receipt["aggregate"]["frame_count"] == 25
    assert receipt["aggregate"]["all_source_alpha_composites_recomputed_pixel_exact"] is True
    assert all(
        frame["receipt"]["path"].endswith("frame_receipt.json") for frame in receipt["frames"]
    )
    first_frame_receipt = json.loads(
        (normalized / receipt["frames"][0]["receipt"]["path"]).read_text(encoding="utf-8")
    )
    assert first_frame_receipt["kind"] == OUTPUT_FRAME_RECEIPT_KIND
    assert first_frame_receipt["seed_policy"] == "deterministic_checkpoint_no_sampling"

    selection = materialize_clean_plate_candidate_selection(
        normalized / "selection_input.json",
        tmp_path / "selection",
    )
    assert selection["aggregate"]["frame_count"] == 25
    boundary = build_manifest(
        normalized / "batch_receipt.json",
        tmp_path / "boundary_manifest.json",
    )
    assert boundary["verification_gates"]["pixel_exactness_recomputed_from_assets"] is True


def test_rejects_hash_consistent_composite_that_violates_alpha_formula(tmp_path: Path) -> None:
    source_receipt = make_fixture(tmp_path, corrupt_composite=True)

    with pytest.raises(
        AOTBatchMaterializationError,
        match="differs from the declared alpha formula",
    ):
        materialize_aotgan_clean_plate_batch(source_receipt, tmp_path / "normalized")


def test_rejects_translucent_guaranteed_core(tmp_path: Path) -> None:
    source_receipt = make_fixture(tmp_path, translucent_core=True)

    with pytest.raises(AOTBatchMaterializationError, match="core alpha must be opaque"):
        materialize_aotgan_clean_plate_batch(source_receipt, tmp_path / "normalized")


def test_rejects_wrong_caller_supplied_receipt_hash_without_partial_output(tmp_path: Path) -> None:
    source_receipt = make_fixture(tmp_path)
    output = tmp_path / "normalized"

    with pytest.raises(AOTBatchMaterializationError, match="source AOT receipt SHA-256 mismatch"):
        materialize_aotgan_clean_plate_batch(
            source_receipt,
            output,
            expected_source_receipt_sha256="0" * 64,
        )
    assert not output.exists()
