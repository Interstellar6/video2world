from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

import scripts.materialize_clean_plate_candidate_selection as selection_module
from scripts.materialize_clean_plate_candidate_selection import (
    BATCH_RECEIPT_KIND,
    FRAME_RECEIPT_KIND,
    INPAINT_BATCH_RECEIPT_KIND,
    INPAINT_FRAME_RECEIPT_KIND,
    INPUT_KIND,
    RECEIPT_KIND,
    CleanPlateCandidateSelectionError,
    materialize_clean_plate_candidate_selection,
    sha256_file,
)

WIDTH = 8
HEIGHT = 6
FRAME_IDS = ("000048", "000049")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def asset(path: Path, *, relative_to: Path) -> dict[str, Any]:
    digest, size = sha256_file(path)
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "sha256": digest,
        "size_bytes": size,
    }


def manifest_asset(path: Path, manifest_path: Path) -> dict[str, Any]:
    digest, size = sha256_file(path)
    return {
        "path": path.relative_to(manifest_path.parent).as_posix(),
        "sha256": digest,
        "size_bytes": size,
    }


def save_rgb(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value.astype(np.uint8)).save(path)


def save_mask(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value.astype(np.uint8) * 255).save(path)


def make_source(path: Path, index: int) -> np.ndarray:
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    rgb = np.stack(
        [
            20 + xx * 3 + index,
            30 + yy * 4 + index,
            np.full_like(xx, 50 + index),
        ],
        axis=2,
    ).astype(np.uint8)
    save_rgb(path, rgb)
    return rgb


def make_batch(
    root: Path,
    frame_ids: tuple[str, ...],
    source_paths: dict[str, Path],
    *,
    candidate_offset: int,
    outside_violation_frame: str | None = None,
    batch_kind: str = BATCH_RECEIPT_KIND,
    frame_kind: str = FRAME_RECEIPT_KIND,
) -> Path:
    frame_records: list[dict[str, Any]] = []
    for sequence_index, frame_id in enumerate(frame_ids):
        frame_root = root / "frames" / f"{sequence_index:04d}"
        source = np.asarray(Image.open(source_paths[frame_id]).convert("RGB"), dtype=np.uint8)
        core = np.zeros((HEIGHT, WIDTH), dtype=bool)
        core[2:4, 3:5] = True
        editable = np.zeros((HEIGHT, WIDTH), dtype=bool)
        editable[1:5, 2:6] = True
        composite = source.copy()
        composite[editable] = np.clip(
            composite[editable].astype(np.int16) + candidate_offset,
            0,
            255,
        ).astype(np.uint8)
        if outside_violation_frame == frame_id:
            composite[0, 0] = [255, 1, 2]
        composite_path = frame_root / "composite.png"
        core_path = frame_root / "masks" / "ownership.png"
        editable_path = frame_root / "masks" / "editable.png"
        save_rgb(composite_path, composite)
        save_mask(core_path, core)
        save_mask(editable_path, editable)
        source_sha, source_size = sha256_file(source_paths[frame_id])
        inputs = {
            "source_rgb": {
                "path": str(source_paths[frame_id]),
                "sha256": source_sha,
                "size_bytes": source_size,
            },
            "ownership_mask_source_rgb_sha256": source_sha,
        }
        outputs = {
            "composite_rgb": asset(composite_path, relative_to=root),
            "ownership_mask": asset(core_path, relative_to=root),
            "editable_mask": asset(editable_path, relative_to=root),
        }
        exactness = {
            "output_ownership_mask_pixel_exact": True,
            "final_outside_editable_rgb_exact": True,
            "final_outside_editable_changed_pixels": 0,
        }
        masks = {"ownership_pixels": int(core.sum()), "editable_pixels": int(editable.sum())}
        frame_receipt = {
            "schema_version": 1,
            "kind": frame_kind,
            "status": "generated_candidate_pending_review",
            "promotion_allowed": False,
            "sequence_index": sequence_index,
            "frame_id": frame_id,
            "inputs": inputs,
            "outputs": outputs,
            "exactness": exactness,
            "masks": masks,
            "review": {"human_review_required": True},
        }
        frame_receipt_path = frame_root / "frame_receipt.json"
        write_json(frame_receipt_path, frame_receipt)
        frame_records.append(
            {
                "sequence_index": sequence_index,
                "frame_id": frame_id,
                "inputs": inputs,
                "outputs": outputs,
                "exactness": exactness,
                "masks": masks,
                "receipt": asset(frame_receipt_path, relative_to=root),
            }
        )
    batch_receipt = {
        "schema_version": 1,
        "kind": batch_kind,
        "status": "generated_batch_pending_review",
        "promotion_allowed": False,
        "frames": frame_records,
        "aggregate": {
            "frame_count": len(frame_records),
            "ordered_frame_ids": list(frame_ids),
            "all_final_outputs_outside_editable_rgb_exact": True,
        },
        "review": {"human_review_required": True},
    }
    batch_receipt_path = root / "batch_receipt.json"
    write_json(batch_receipt_path, batch_receipt)
    return batch_receipt_path


def make_fixture(
    tmp_path: Path,
    *,
    outside_violation_frame: str | None = None,
    fallback_source_path: Path | None = None,
    contact_sheet: bool = True,
    batch_kind: str = BATCH_RECEIPT_KIND,
    frame_kind: str = FRAME_RECEIPT_KIND,
) -> tuple[Path, Path, Path, dict[str, Path]]:
    sources: dict[str, Path] = {}
    for index, frame_id in enumerate(FRAME_IDS):
        source_path = tmp_path / "sources" / f"{frame_id}.png"
        make_source(source_path, index)
        sources[frame_id] = source_path
    source_batch_path = make_batch(
        tmp_path / "batch-main",
        FRAME_IDS,
        sources,
        candidate_offset=40,
        batch_kind=batch_kind,
        frame_kind=frame_kind,
    )
    fallback_sources = dict(sources)
    if fallback_source_path is not None:
        fallback_sources[FRAME_IDS[1]] = fallback_source_path
    fallback_batch_path = make_batch(
        tmp_path / "batch-fallback",
        (FRAME_IDS[1],),
        fallback_sources,
        candidate_offset=70,
        outside_violation_frame=outside_violation_frame,
        batch_kind=batch_kind,
        frame_kind=frame_kind,
    )
    manifest_path = tmp_path / "selection_input.json"
    source_batch = json.loads(source_batch_path.read_text(encoding="utf-8"))
    fallback_batch = json.loads(fallback_batch_path.read_text(encoding="utf-8"))
    selections = []
    for sequence_index, frame_id in enumerate(FRAME_IDS):
        batch_path = source_batch_path if sequence_index == 0 else fallback_batch_path
        batch = source_batch if sequence_index == 0 else fallback_batch
        batch_frame = batch["frames"][0]
        batch_root = batch_path.parent
        selections.append(
            {
                "sequence_index": sequence_index,
                "frame_id": frame_id,
                "selection_note": "selected during test review",
                "candidate_batch_receipt": manifest_asset(batch_path, manifest_path),
                "composite_rgb": manifest_asset(
                    batch_root / batch_frame["outputs"]["composite_rgb"]["path"],
                    manifest_path,
                ),
                "core_mask": manifest_asset(
                    batch_root / batch_frame["outputs"]["ownership_mask"]["path"],
                    manifest_path,
                ),
                "editable_mask": manifest_asset(
                    batch_root / batch_frame["outputs"]["editable_mask"]["path"],
                    manifest_path,
                ),
            }
        )
    manifest = {
        "schema_version": 1,
        "kind": INPUT_KIND,
        "source_batch_receipt": manifest_asset(source_batch_path, manifest_path),
        "ordered_frame_ids": list(FRAME_IDS),
        "selections": selections,
        "contact_sheet": {
            "enabled": contact_sheet,
            "thumbnail_width": 80,
            "columns": 2,
        },
    }
    write_json(manifest_path, manifest)
    return manifest_path, source_batch_path, fallback_batch_path, sources


def test_materializes_candidates_from_multiple_batches_with_exact_receipt(tmp_path: Path) -> None:
    manifest_path, source_batch_path, fallback_batch_path, sources = make_fixture(tmp_path)
    output = tmp_path / "selected"
    receipt = materialize_clean_plate_candidate_selection(
        manifest_path,
        output,
        created_at=datetime(2026, 7, 18, 12, 0, tzinfo=UTC),
    )

    assert receipt["kind"] == RECEIPT_KIND
    assert receipt["promotion_allowed"] is False
    assert receipt["human_review_required"] is True
    assert receipt["review"]["quality_acceptance_performed"] is False
    assert receipt["ordered_frame_ids"] == list(FRAME_IDS)
    assert len(receipt["candidate_batch_receipts"]) == 2
    assert {item["sha256"] for item in receipt["candidate_batch_receipts"]} == {
        sha256_file(source_batch_path)[0],
        sha256_file(fallback_batch_path)[0],
    }
    assert receipt["aggregate"]["all_outputs_outside_editable_rgb_exact_recomputed"] is True
    assert (output / "selection_receipt.json").is_file()
    assert (output / "contact_sheet.png").is_file()

    source_batch = json.loads(source_batch_path.read_text(encoding="utf-8"))
    fallback_batch = json.loads(fallback_batch_path.read_text(encoding="utf-8"))
    expected_candidates = (
        source_batch_path.parent / source_batch["frames"][0]["outputs"]["composite_rgb"]["path"],
        fallback_batch_path.parent
        / fallback_batch["frames"][0]["outputs"]["composite_rgb"]["path"],
    )
    for sequence_index, (frame_id, expected) in enumerate(
        zip(FRAME_IDS, expected_candidates, strict=True)
    ):
        output_frame = output / "frames" / f"{sequence_index:04d}.png"
        assert output_frame.read_bytes() == expected.read_bytes()
        assert receipt["frames"][sequence_index]["frame_id"] == frame_id
        assert (
            receipt["frames"][sequence_index]["source_rgb"]["sha256"]
            == sha256_file(sources[frame_id])[0]
        )
        assert (output / "masks" / "core" / f"{sequence_index:04d}.png").is_file()
        assert (output / "masks" / "editable" / f"{sequence_index:04d}.png").is_file()


def test_materializes_generic_inpaint_batch_without_claiming_diffusion(tmp_path: Path) -> None:
    manifest_path, _, _, _ = make_fixture(
        tmp_path,
        batch_kind=INPAINT_BATCH_RECEIPT_KIND,
        frame_kind=INPAINT_FRAME_RECEIPT_KIND,
    )

    receipt = materialize_clean_plate_candidate_selection(
        manifest_path,
        tmp_path / "selected-inpaint",
    )

    assert receipt["aggregate"]["frame_count"] == len(FRAME_IDS)
    assert receipt["aggregate"]["all_outputs_outside_editable_rgb_exact_recomputed"] is True


def test_rejects_frame_kind_that_does_not_match_inpaint_batch(tmp_path: Path) -> None:
    manifest_path, _, _, _ = make_fixture(
        tmp_path,
        batch_kind=INPAINT_BATCH_RECEIPT_KIND,
        frame_kind=FRAME_RECEIPT_KIND,
    )

    with pytest.raises(CleanPlateCandidateSelectionError, match="receipt kind is invalid"):
        materialize_clean_plate_candidate_selection(manifest_path, tmp_path / "selected")


def test_rejects_candidate_with_different_source_rgb_lineage(tmp_path: Path) -> None:
    wrong_source = tmp_path / "wrong-source.png"
    make_source(wrong_source, 19)
    manifest_path, _, _, _ = make_fixture(tmp_path, fallback_source_path=wrong_source)

    with pytest.raises(CleanPlateCandidateSelectionError, match="source lineage differs"):
        materialize_clean_plate_candidate_selection(manifest_path, tmp_path / "selected")


def test_recomputes_and_rejects_outside_editable_rgb_violation(tmp_path: Path) -> None:
    manifest_path, _, _, _ = make_fixture(
        tmp_path,
        outside_violation_frame=FRAME_IDS[1],
    )

    with pytest.raises(CleanPlateCandidateSelectionError, match="outside editable mask"):
        materialize_clean_plate_candidate_selection(manifest_path, tmp_path / "selected")


def test_rejects_wrong_frame_order_and_incomplete_selection(tmp_path: Path) -> None:
    manifest_path, _, _, _ = make_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["ordered_frame_ids"] = list(reversed(FRAME_IDS))
    write_json(manifest_path, manifest)

    with pytest.raises(CleanPlateCandidateSelectionError, match="complete source batch"):
        materialize_clean_plate_candidate_selection(manifest_path, tmp_path / "wrong-order")

    manifest["ordered_frame_ids"] = list(FRAME_IDS)
    manifest["selections"] = manifest["selections"][:-1]
    write_json(manifest_path, manifest)
    with pytest.raises(CleanPlateCandidateSelectionError, match="cover every source frame"):
        materialize_clean_plate_candidate_selection(manifest_path, tmp_path / "incomplete")


def test_rejects_mutated_selected_artifact_hash(tmp_path: Path) -> None:
    manifest_path, _, fallback_batch_path, _ = make_fixture(tmp_path)
    fallback = json.loads(fallback_batch_path.read_text(encoding="utf-8"))
    candidate_path = (
        fallback_batch_path.parent / fallback["frames"][0]["outputs"]["composite_rgb"]["path"]
    )
    candidate = np.asarray(Image.open(candidate_path).convert("RGB"), dtype=np.uint8).copy()
    candidate[2, 2] = [1, 2, 3]
    save_rgb(candidate_path, candidate)

    with pytest.raises(CleanPlateCandidateSelectionError, match="SHA-256 mismatch"):
        materialize_clean_plate_candidate_selection(manifest_path, tmp_path / "selected")


def test_refuses_overwrite_and_cleans_staging_after_copy_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, _, _, _ = make_fixture(tmp_path)
    output = tmp_path / "selected"
    output.mkdir()
    with pytest.raises(CleanPlateCandidateSelectionError, match="refusing to overwrite"):
        materialize_clean_plate_candidate_selection(manifest_path, output)

    output.rmdir()
    original_copy = selection_module.copy_asset
    calls = 0

    def fail_after_first_copy(source: Any, destination: Path) -> None:
        nonlocal calls
        calls += 1
        original_copy(source, destination)
        if calls == 2:
            raise CleanPlateCandidateSelectionError("injected copy failure")

    monkeypatch.setattr(selection_module, "copy_asset", fail_after_first_copy)
    with pytest.raises(CleanPlateCandidateSelectionError, match="injected copy failure"):
        materialize_clean_plate_candidate_selection(manifest_path, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".selected.staging-*"))


def test_cli_reports_hash_failure_without_partial_output(tmp_path: Path) -> None:
    manifest_path, _, _, _ = make_fixture(tmp_path, contact_sheet=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["selections"][0]["composite_rgb"]["sha256"] = "0" * 64
    write_json(manifest_path, manifest)
    output = tmp_path / "selected"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/materialize_clean_plate_candidate_selection.py",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output),
        ],
        cwd=Path(__file__).parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "SHA-256 mismatch" in result.stderr
    assert not output.exists()
