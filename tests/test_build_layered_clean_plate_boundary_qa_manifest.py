from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from scripts.build_layered_clean_plate_boundary_qa_manifest import (
    TEXTURE_SAMPLING,
    THRESHOLDS,
    BoundaryQAManifestBuildError,
    build_manifest,
    canonical_sha256,
    sha256_file,
)
from scripts.qa_layered_clean_plate_boundaries import validate as validate_boundary_qa

HEIGHT = 96
WIDTH = 128
FRAME_IDS = ("000048", "000049")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def save_rgb(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value.astype(np.uint8)).save(path)


def save_mask(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value.astype(np.uint8) * 255).save(path)


def artifact(path: Path, *, root: Path) -> dict[str, Any]:
    digest, size = sha256_file(path)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": digest,
        "size_bytes": size,
    }


def textured_source(sequence_index: int) -> np.ndarray:
    y, x = np.indices((HEIGHT, WIDTH))
    checker = ((x // 2 + y // 2) % 2) * 7
    source = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    source[..., 0] = 48 + sequence_index + checker + (x % 9)
    source[..., 1] = 70 + checker + (y % 11)
    source[..., 2] = 92 + checker + ((x + y) % 7)
    return source


def make_fixture(tmp_path: Path, *, outside_change: bool = False) -> dict[str, Any]:
    root = tmp_path / "batch"
    root.mkdir(parents=True)
    source_manifest = root / "source-input-manifest.json"
    write_json(source_manifest, {"kind": "test.source", "frame_ids": list(FRAME_IDS)})
    source_manifest_asset = artifact(source_manifest, root=root)
    batch_frames: list[dict[str, Any]] = []
    frame_receipt_hashes: list[str] = []

    for sequence_index, frame_id in enumerate(FRAME_IDS):
        source = textured_source(sequence_index)
        core = np.zeros((HEIGHT, WIDTH), dtype=bool)
        core[30:66, 40 + sequence_index : 88 + sequence_index] = True
        editable = np.zeros_like(core)
        editable[26:70, 36 + sequence_index : 92 + sequence_index] = True
        context = np.zeros_like(core)
        context[20:76, 30 + sequence_index : 98 + sequence_index] = True
        collar = editable & ~core
        protected = np.zeros_like(core)
        protected_collar = np.zeros_like(core)

        raw = source.copy()
        shifted = np.roll(source, shift=1, axis=1)
        raw[context] = shifted[context]
        composite = source.copy()
        composite[core] = raw[core]
        if outside_change and sequence_index == 0:
            composite[2, 2] = [250, 10, 20]

        source_path = root / "inputs" / f"source-{frame_id}.png"
        input_ownership_path = root / "inputs" / f"ownership-{frame_id}.png"
        save_rgb(source_path, source)
        save_mask(input_ownership_path, core)

        frame_root = root / "frames" / f"{sequence_index:04d}"
        output_values: dict[str, tuple[Path, np.ndarray, str]] = {
            "raw_candidate_rgb": (frame_root / "raw_candidate.png", raw, "rgb"),
            "composite_rgb": (frame_root / "composite.png", composite, "rgb"),
            "ownership_mask": (frame_root / "masks" / "ownership.png", core, "mask"),
            "context_mask": (frame_root / "masks" / "context.png", context, "mask"),
            "collar_mask": (frame_root / "masks" / "collar.png", collar, "mask"),
            "editable_mask": (frame_root / "masks" / "editable.png", editable, "mask"),
            "protected_mask": (
                frame_root / "masks" / "protected.png",
                protected,
                "mask",
            ),
            "protected_collar_mask": (
                frame_root / "masks" / "protected_collar.png",
                protected_collar,
                "mask",
            ),
        }
        outputs: dict[str, dict[str, Any]] = {}
        for key, (path, value, kind) in output_values.items():
            save_rgb(path, value) if kind == "rgb" else save_mask(path, value)
            outputs[key] = artifact(path, root=root)
        weight_path = frame_root / "masks" / "blend_weight.png"
        weight_path.parent.mkdir(parents=True, exist_ok=True)
        weight = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        weight[editable] = 127
        weight[core] = 255
        Image.fromarray(weight).save(weight_path)
        outputs["blend_weight"] = artifact(weight_path, root=root)

        inputs = {
            "source_rgb": artifact(source_path, root=root),
            "ownership_mask": artifact(input_ownership_path, root=root),
            "ownership_mask_source_rgb_sha256": artifact(source_path, root=root)["sha256"],
            "guide_rgb": None,
            "guide_source_rgb_sha256": None,
            "protected_mask": None,
            "protected_mask_source_rgb_sha256": None,
        }
        masks = {
            "dilation_metric": "euclidean_distance_transform",
            "context_dilation_pixels": 12,
            "boundary_outer_collar_pixels": 4,
            "frame_pixels": HEIGHT * WIDTH,
            "ownership_pixels": int(core.sum()),
            "context_pixels": int(context.sum()),
            "context_additional_pixels": int((context & ~core).sum()),
            "context_excluded_protected_noncore_pixels": 0,
            "collar_pixels": int(collar.sum()),
            "protected_pixels": 0,
            "protected_collar_pixels": 0,
            "editable_pixels": int(editable.sum()),
        }
        exactness = {
            "context_includes_ownership_core": True,
            "output_ownership_mask_pixel_exact": True,
            "inference_outside_context_rgb_exact": True,
            "inference_outside_context_changed_pixels": 0,
            "guide_outside_context_changed_from_source_pixels": 0,
            "raw_candidate_outside_context_rgb_exact": True,
            "raw_candidate_outside_context_changed_pixels": 0,
            "final_outside_editable_rgb_exact": True,
            "final_outside_editable_changed_pixels": 0,
            "final_ownership_core_equals_raw_candidate": True,
            "protected_outer_collar_rgb_exact": True,
            "provider_inside_context_changed_from_inference_pixels": int(context.sum()),
            "provider_inside_ownership_changed_from_source_pixels": int(core.sum()),
            "provider_outside_context_changed_from_source_pixels": 0,
        }
        frame_receipt = {
            "schema_version": 1,
            "kind": "video2world.diffusion_clean_plate_frame_run",
            "status": "generated_candidate_pending_review",
            "promotion_allowed": False,
            "sequence_index": sequence_index,
            "frame_id": frame_id,
            "seed": 700,
            "seed_policy": {"kind": "base_plus_sequence_index", "base_seed": 700, "stride": 0},
            "batch_contract": {
                "input_manifest_sha256": source_manifest_asset["sha256"],
                "model_repo": "diffusers/test",
                "model_revision": "fixed",
                "model_file_manifest_sha256": "f" * 64,
            },
            "generation": {},
            "inputs": inputs,
            "masks": masks,
            "exactness": exactness,
            "outputs": outputs,
            "review": {
                "automatic_accept": False,
                "human_review_required": True,
                "published": False,
            },
        }
        frame_receipt_path = frame_root / "frame_receipt.json"
        write_json(frame_receipt_path, frame_receipt)
        frame_receipt_asset = artifact(frame_receipt_path, root=root)
        frame_receipt_hashes.append(frame_receipt_asset["sha256"])
        batch_frames.append(
            {
                "sequence_index": sequence_index,
                "frame_id": frame_id,
                "seed": 700,
                "inputs": inputs,
                "receipt": frame_receipt_asset,
                "outputs": outputs,
                "exactness": exactness,
                "masks": masks,
            }
        )

    batch_receipt = {
        "schema_version": 1,
        "kind": "video2world.diffusion_clean_plate_batch_run",
        "status": "generated_batch_pending_review",
        "promotion_allowed": False,
        "input_manifest": source_manifest_asset,
        "frames": batch_frames,
        "aggregate": {
            "frame_count": len(batch_frames),
            "ordered_frame_ids": list(FRAME_IDS),
            "ordered_seeds": [700, 700],
            "frame_receipt_set_sha256": canonical_sha256(frame_receipt_hashes),
            "all_raw_candidates_outside_context_rgb_exact": True,
            "all_inference_inputs_outside_context_rgb_exact": True,
            "all_final_outputs_outside_editable_rgb_exact": True,
            "all_ownership_cores_equal_raw_candidates": True,
            "all_protected_outer_collars_rgb_exact": True,
        },
        "review": {
            "automatic_accept": False,
            "human_review_required": True,
            "published": False,
        },
    }
    batch_path = root / "batch_receipt.json"
    write_json(batch_path, batch_receipt)
    return {
        "root": root,
        "batch_path": batch_path,
        "batch": batch_receipt,
        "output": tmp_path / "qa" / "boundary-manifest.json",
    }


def test_builds_consumer_compatible_manifest_with_explicit_self_check_contract(
    tmp_path: Path,
) -> None:
    fixture = make_fixture(tmp_path)
    batch_sha256 = sha256_file(fixture["batch_path"])[0]
    manifest = build_manifest(
        fixture["batch_path"],
        fixture["output"],
        expected_batch_receipt_sha256=batch_sha256,
        created_at=datetime(2026, 7, 18, 15, 0, tzinfo=UTC),
    )

    assert manifest["kind"] == "video2world.layered_clean_plate_boundary_qa_manifest"
    assert manifest["created_at"] == "2026-07-18T15:00:00+00:00"
    assert manifest["frame_ids"] == list(FRAME_IDS)
    assert manifest["thresholds"] == THRESHOLDS
    assert manifest["texture_sampling"] == TEXTURE_SAMPLING
    assert manifest["source_batch_receipt"]["hash_verification"] == (
        "matched_caller_supplied_sha256"
    )
    contract = manifest["target_matte_contract"]
    assert contract["role"] == "alignment_self_check_only"
    assert contract["semantic_residual_evaluated"] is False
    assert contract["semantic_residual_claim_allowed"] is False
    assert manifest["verification_gates"]["pixel_exactness_recomputed_from_assets"] is True

    for record in manifest["frame_records"]:
        assert record["target_matte"]["path"] == record["removal_core_mask"]["path"]
        assert record["target_matte"]["sha256"] == record["removal_core_mask"]["sha256"]
        assert record["target_matte"]["evidence_role"] == "alignment_self_check_only"
        assert record["evidence_contract"]["semantic_residual_evaluated"] is False
        assert all(record["recomputed_exactness"].values())
        assert Path(record["provenance"]["frame_receipt"]["path"]).is_file()

    qa_report = validate_boundary_qa(
        argparse.Namespace(
            input_manifest=fixture["output"],
            output_report=tmp_path / "qa" / "report.json",
            overlay_dir=None,
            contact_sheet=None,
            contact_sheet_samples=1,
        )
    )
    assert qa_report["status"] == "technical_passed"
    assert qa_report["boundary_texture_gate_passed"] is True
    assert qa_report["promotion_approved"] is False


def test_rejects_tampered_frame_receipt_hash_without_output(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    receipt_path = fixture["root"] / fixture["batch"]["frames"][0]["receipt"]["path"]
    receipt_path.write_text(receipt_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(BoundaryQAManifestBuildError, match="frame 000048 receipt SHA-256 mismatch"):
        build_manifest(fixture["batch_path"], fixture["output"])

    assert not fixture["output"].exists()


def test_rejects_tampered_composite_asset_hash_without_output(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    composite_path = (
        fixture["root"] / fixture["batch"]["frames"][1]["outputs"]["composite_rgb"]["path"]
    )
    value = np.asarray(Image.open(composite_path).convert("RGB"), dtype=np.uint8).copy()
    value[40, 40] = [1, 2, 3]
    save_rgb(composite_path, value)

    with pytest.raises(
        BoundaryQAManifestBuildError,
        match=r"frame 000049\.outputs\.composite_rgb SHA-256 mismatch",
    ):
        build_manifest(fixture["batch_path"], fixture["output"])

    assert not fixture["output"].exists()


def test_rejects_receipt_claim_when_composite_actually_changes_outside_editable(
    tmp_path: Path,
) -> None:
    fixture = make_fixture(tmp_path, outside_change=True)

    with pytest.raises(
        BoundaryQAManifestBuildError,
        match="recomputed exactness failed: composite_outside_editable_rgb_exact",
    ):
        build_manifest(fixture["batch_path"], fixture["output"])

    assert not fixture["output"].exists()


def test_rejects_batch_frame_order_mismatch_without_output(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    fixture["batch"]["aggregate"]["ordered_frame_ids"] = list(reversed(FRAME_IDS))
    write_json(fixture["batch_path"], fixture["batch"])

    with pytest.raises(BoundaryQAManifestBuildError, match="ordered_frame_ids mismatch"):
        build_manifest(fixture["batch_path"], fixture["output"])

    assert not fixture["output"].exists()


def test_rejects_wrong_caller_supplied_batch_hash(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)

    with pytest.raises(BoundaryQAManifestBuildError, match="batch receipt SHA-256 mismatch"):
        build_manifest(
            fixture["batch_path"],
            fixture["output"],
            expected_batch_receipt_sha256="0" * 64,
        )

    assert not fixture["output"].exists()


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    fixture["output"].parent.mkdir(parents=True)
    fixture["output"].write_text("keep", encoding="utf-8")

    with pytest.raises(BoundaryQAManifestBuildError, match="refusing to overwrite"):
        build_manifest(fixture["batch_path"], fixture["output"])

    assert fixture["output"].read_text(encoding="utf-8") == "keep"
