from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "build_cumulative_removal_manifest.py"
SPEC = importlib.util.spec_from_file_location("build_cumulative_removal_manifest", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((6, 8, 3), color, dtype=np.uint8)).save(path)


def make_mask(path: Path, box: tuple[int, int, int, int]) -> np.ndarray:
    value = np.zeros((6, 8), dtype=np.uint8)
    x0, y0, x1, y1 = box
    value[y0:y1, x0:x1] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value).save(path)
    return value > 0


def object_manifest(path: Path, frames: Path, masks: Path, object_id: str) -> None:
    records = []
    for index, frame_id in enumerate(("000000", "000001")):
        frame_path = frames / f"{frame_id}.png"
        mask_path = masks / f"{frame_id}.png"
        records.append(
            {
                "sequence_index": index,
                "frame_id": frame_id,
                "input_frame": {
                    "path": str(frame_path),
                    "sha256": MODULE.sha256_file(frame_path),
                },
                "input_mask": {
                    "path": str(mask_path),
                    "sha256": MODULE.sha256_file(mask_path),
                },
            }
        )
    write_json(path, {"object_id": object_id, "frame_records": records})


def args_for(
    *,
    round_index: int,
    object_id: str,
    current_manifest: Path,
    donor_frames: Path,
    output: Path,
    previous_manifest: Path | None = None,
    previous_report: Path | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        round_index=round_index,
        object_id=object_id,
        current_object_manifest=current_manifest,
        previous_cumulative_manifest=previous_manifest,
        previous_prefill_report=previous_report,
        observed_donor_frames_dir=donor_frames,
        output=output,
    )


def make_round1(tmp_path: Path) -> tuple[dict[str, Any], Path, dict[str, np.ndarray]]:
    donors = tmp_path / "raw"
    masks = tmp_path / "round1_object_masks"
    expected: dict[str, np.ndarray] = {}
    for index, frame_id in enumerate(("000000", "000001")):
        make_image(donors / f"{frame_id}.png", (20 + index, 30, 40))
        expected[frame_id] = make_mask(masks / f"{frame_id}.png", (1, 1, 3, 3))
    current = tmp_path / "round1_object.json"
    object_manifest(current, donors, masks, "front")
    output = tmp_path / "round1"
    manifest = MODULE.build_manifest(
        args_for(
            round_index=1,
            object_id="front",
            current_manifest=current,
            donor_frames=donors,
            output=output,
        )
    )
    return manifest, output, expected


def fake_prefill_report(round_root: Path, *, wrong_input_hash: bool = False) -> Path:
    manifest_path = round_root / "cumulative_removal_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = []
    for record in manifest["frame_records"]:
        frame_id = record["frame_id"]
        output_path = round_root / "prefill" / f"{frame_id}.png"
        source = Path(record["source_frame"])
        make_image(output_path, tuple(np.asarray(Image.open(source))[0, 0].tolist()))
        mask_path = round_root / "residual" / f"{frame_id}.png"
        removal = MODULE.mask_array(round_root / record["union_mask"])
        residual = np.zeros_like(removal, dtype=np.uint8)
        residual[removal] = 255
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(residual).save(mask_path)
        records.append(
            {
                "frame_id": frame_id,
                "prefill_frame": str(output_path),
                "prefill_frame_sha256": MODULE.sha256_file(output_path),
                "residual_mask": str(mask_path),
                "residual_mask_sha256": MODULE.sha256_file(mask_path),
                "residual_mask_pixels": int(removal.sum()),
            }
        )
    report = round_root / "prefill_report.json"
    write_json(
        report,
        {
            "status": "technical_passed",
            "input_manifest_sha256": (
                "0" * 64 if wrong_input_hash else MODULE.sha256_file(manifest_path)
            ),
            "gates": {
                "outside_removal_mask_rgb_exact": True,
                "all_residual_masks_subset_of_removal_masks": True,
            },
            "pixel_provenance": {
                "generated_pixels": 0,
                "propainter_pixels": 0,
                "unresolved_pixels_are_not_valid_donor_or_geometry_evidence": True,
            },
            "frame_records": records,
        },
    )
    return report


def test_round2_unions_previous_masks_and_hashes_immediate_input(tmp_path: Path) -> None:
    _, round1, round1_masks = make_round1(tmp_path)
    report = fake_prefill_report(round1)
    donors = tmp_path / "raw"
    masks = tmp_path / "round2_object_masks"
    round2_masks = {
        frame_id: make_mask(masks / f"{frame_id}.png", (4, 2, 7, 5))
        for frame_id in ("000000", "000001")
    }
    current = tmp_path / "round2_object.json"
    object_manifest(current, donors, masks, "left")

    output = tmp_path / "round2"
    manifest = MODULE.build_manifest(
        args_for(
            round_index=2,
            object_id="left",
            current_manifest=current,
            donor_frames=donors,
            output=output,
            previous_manifest=round1 / "cumulative_removal_manifest.json",
            previous_report=report,
        )
    )

    assert manifest["removed_object_ids"] == ["front", "left"]
    assert manifest["gates"]["unresolved_residual_is_never_donor_evidence"] is True
    assert (
        manifest["gates"]["previous_prefill_excludes_generated_and_propainter_pixels"]
        is True
    )
    for record in manifest["frame_records"]:
        frame_id = record["frame_id"]
        actual = MODULE.mask_array(output / record["union_mask"])
        assert np.array_equal(actual, round1_masks[frame_id] | round2_masks[frame_id])
        assert record["input_lineage"]["previous_unresolved_pixels_remain_non_donor"] is True
        assert record["removed_object_ids"] == ["front", "left"]
    donor_index = json.loads((output / "donor_exclusion_index.json").read_text())
    assert {item["label"] for item in donor_index["items"]} == {"cumulative_removed"}
    assert all(
        item["unresolved_residual_allowed_as_donor"] is False
        for item in donor_index["items"]
    )


def test_round2_rejects_a_previous_report_with_wrong_manifest_hash(tmp_path: Path) -> None:
    _, round1, _ = make_round1(tmp_path)
    report = fake_prefill_report(round1, wrong_input_hash=True)
    donors = tmp_path / "raw"
    masks = tmp_path / "round2_object_masks"
    for frame_id in ("000000", "000001"):
        make_mask(masks / f"{frame_id}.png", (4, 2, 7, 5))
    current = tmp_path / "round2_object.json"
    object_manifest(current, donors, masks, "left")

    with pytest.raises(ValueError, match="does not hash"):
        MODULE.build_manifest(
            args_for(
                round_index=2,
                object_id="left",
                current_manifest=current,
                donor_frames=donors,
                output=tmp_path / "round2",
                previous_manifest=round1 / "cumulative_removal_manifest.json",
                previous_report=report,
            )
        )


@pytest.mark.parametrize("field", ["generated_pixels", "propainter_pixels"])
def test_round2_rejects_missing_previous_generated_provenance(
    tmp_path: Path, field: str
) -> None:
    _, round1, _ = make_round1(tmp_path)
    report = fake_prefill_report(round1)
    value = json.loads(report.read_text(encoding="utf-8"))
    del value["pixel_provenance"][field]
    write_json(report, value)
    donors = tmp_path / "raw"
    masks = tmp_path / "round2_object_masks"
    for frame_id in ("000000", "000001"):
        make_mask(masks / f"{frame_id}.png", (4, 2, 7, 5))
    current = tmp_path / "round2_object.json"
    object_manifest(current, donors, masks, "left")

    with pytest.raises(ValueError, match="contains or omits"):
        MODULE.build_manifest(
            args_for(
                round_index=2,
                object_id="left",
                current_manifest=current,
                donor_frames=donors,
                output=tmp_path / "round2",
                previous_manifest=round1 / "cumulative_removal_manifest.json",
                previous_report=report,
            )
        )


def test_round1_rejects_a_non_original_source_frame(tmp_path: Path) -> None:
    donors = tmp_path / "raw"
    source = tmp_path / "generated"
    masks = tmp_path / "masks"
    for frame_id in ("000000", "000001"):
        make_image(donors / f"{frame_id}.png", (10, 20, 30))
        make_image(source / f"{frame_id}.png", (200, 10, 10))
        make_mask(masks / f"{frame_id}.png", (1, 1, 3, 3))
    current = tmp_path / "round1_object.json"
    object_manifest(current, source, masks, "front")

    with pytest.raises(ValueError, match="not original observed RGB"):
        MODULE.build_manifest(
            args_for(
                round_index=1,
                object_id="front",
                current_manifest=current,
                donor_frames=donors,
                output=tmp_path / "round1",
            )
        )
