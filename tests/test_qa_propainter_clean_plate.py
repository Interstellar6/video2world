from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from scripts.qa_propainter_clean_plate import sha256_file, validate


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_fixture(tmp_path: Path, *, legacy: bool = False) -> dict[str, Any]:
    round_root = tmp_path / "round02"
    frame_path = round_root / "input" / "frames" / "0000.png"
    mask_path = round_root / "input" / "masks" / "0000.png"
    output_path = round_root / "propainter-output" / "0000.png"
    for path in (frame_path, mask_path, output_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    source = np.full((4, 6, 3), (20, 30, 40), dtype=np.uint8)
    mask = np.zeros((4, 6), dtype=np.uint8)
    mask[1:3, 2:5] = 255
    output = source.copy()
    output[mask > 0] += 40
    Image.fromarray(source).save(frame_path)
    Image.fromarray(mask).save(mask_path)
    Image.fromarray(output).save(output_path)

    if legacy:
        record = {
            "sequence_index": 0,
            "frame_id": "000064",
            "source_frame": str(frame_path),
            "union_mask": str(mask_path),
            "union_mask_pixels": 6,
        }
    else:
        record = {
            "sequence_index": 0,
            "frame_id": "000064",
            "input_frame": {
                "path": "input/frames/0000.png",
                "sha256": sha256_file(frame_path),
                "bytes": frame_path.stat().st_size,
                "dimensions": [6, 4],
            },
            "input_mask": {
                "path": "input/masks/0000.png",
                "sha256": sha256_file(mask_path),
                "bytes": mask_path.stat().st_size,
                "dimensions": [6, 4],
                "area_pixels": 6,
            },
        }
    manifest = {
        "schema_version": 1,
        "frame_count": 1,
        "frame_records": [record],
    }
    manifest_path = round_root / "input_manifest.json"
    write_json(manifest_path, manifest)
    return {
        "root": round_root,
        "frame": frame_path,
        "mask": mask_path,
        "output": output_path,
        "manifest": manifest,
        "manifest_path": manifest_path,
    }


def args_for(fixture: dict[str, Any]) -> argparse.Namespace:
    root = fixture["root"]
    return argparse.Namespace(
        input_manifest=fixture["manifest_path"],
        output_frames=root / "propainter-output",
        output_report=root / "qa" / "report.json",
        contact_sheet=root / "qa" / "contact-sheet.png",
        mask_dilation=0,
        max_outside_mae=0.0,
        min_inside_mae=10.0,
        samples=1,
    )


def test_validates_portable_nested_manifest_relative_paths_hashes_and_output(
    tmp_path: Path,
) -> None:
    fixture = make_fixture(tmp_path)

    report = validate(args_for(fixture))

    assert report["status"] == "technical_passed"
    assert report["input_manifest_sha256"] == sha256_file(fixture["manifest_path"])
    assert report["gates"]["outside_dilated_mask_mean_abs_delta_lte"]["passed"] is True
    assert report["gates"]["inside_mask_mean_abs_delta_gte"]["passed"] is True
    frame = report["frame_records"][0]
    assert Path(frame["source_frame"]) == fixture["frame"].resolve()
    assert Path(frame["union_mask"]) == fixture["mask"].resolve()
    assert frame["union_mask_pixels"] == 6
    assert frame["output_sha256"] == sha256_file(fixture["output"])
    assert frame["inside_original_mask_abs_rgb_delta"]["mean"] == 40.0
    assert frame["outside_dilated_mask_abs_rgb_delta"]["mean"] == 0.0
    assert (fixture["root"] / "qa" / "report.json").is_file()
    assert (fixture["root"] / "qa" / "contact-sheet.png").is_file()


def test_keeps_legacy_manifest_fields_compatible(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path, legacy=True)

    report = validate(args_for(fixture))

    assert report["status"] == "technical_passed"
    assert report["frame_records"][0]["union_mask_pixels"] == 6


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("sha256", "0" * 64, "input_frame.sha256 does not match"),
        ("area_pixels", 7, "input_mask.area_pixels does not match"),
    ],
)
def test_portable_manifest_fails_closed_on_bad_integrity_metadata(
    tmp_path: Path,
    field: str,
    replacement: str | int,
    message: str,
) -> None:
    fixture = make_fixture(tmp_path)
    record = fixture["manifest"]["frame_records"][0]
    target = record["input_frame"] if field == "sha256" else record["input_mask"]
    target[field] = replacement
    write_json(fixture["manifest_path"], fixture["manifest"])

    with pytest.raises(ValueError, match=message):
        validate(args_for(fixture))


def test_inside_change_gate_fails_when_output_is_unchanged(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    with Image.open(fixture["frame"]) as source:
        source.save(fixture["output"])

    report = validate(args_for(fixture))

    assert report["status"] == "technical_failed"
    assert report["gates"]["outside_dilated_mask_mean_abs_delta_lte"]["passed"] is True
    assert report["gates"]["inside_mask_mean_abs_delta_gte"]["passed"] is False


def test_outside_change_gate_fails_on_background_drift(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    with Image.open(fixture["output"]) as image:
        output = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    output[0, 0] += 20
    Image.fromarray(output).save(fixture["output"])

    report = validate(args_for(fixture))

    assert report["status"] == "technical_failed"
    assert report["gates"]["outside_dilated_mask_mean_abs_delta_lte"]["passed"] is False
    assert report["gates"]["inside_mask_mean_abs_delta_gte"]["passed"] is True
