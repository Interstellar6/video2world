from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from scripts.build_propainter_boundary_candidate_manifest import (
    OUTPUT_KIND,
    PropainterBoundaryManifestError,
    build_manifest,
    build_parser,
    sha256_file,
)
from scripts.compose_clean_plate_boundary_candidate import compose_manifest

HEIGHT = 11
WIDTH = 13
FRAME_IDS = ("000048", "000049")


def save_rgb(
    path: Path,
    color: tuple[int, int, int],
    *,
    shape: tuple[int, int] = (HEIGHT, WIDTH),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((shape[0], shape[1], 3), color, dtype=np.uint8)).save(path)


def save_mask(
    path: Path,
    *,
    x: int,
    value: int = 255,
    shape: tuple[int, int] = (HEIGHT, WIDTH),
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[4:7, x : x + 2] = value
    Image.fromarray(mask).save(path)
    return int(np.count_nonzero(mask))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_fixture(tmp_path: Path) -> dict[str, Any]:
    root = tmp_path / "propainter"
    records: list[dict[str, Any]] = []
    for sequence_index, frame_id in enumerate(FRAME_IDS):
        source = root / "source" / f"{frame_id}.png"
        prepared = root / "input" / "frames" / f"{sequence_index:04d}.png"
        mask = root / "input" / "masks" / f"{sequence_index:04d}.png"
        save_rgb(source, (30 + sequence_index, 50, 70))
        prepared.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, prepared)
        mask_pixels = save_mask(mask, x=3 + sequence_index)
        records.append(
            {
                "sequence_index": sequence_index,
                "frame_id": frame_id,
                "source_frame": str(source.relative_to(root)),
                "source_frame_sha256": sha256_file(source),
                "input_frame": str(prepared.relative_to(root)),
                "input_frame_sha256": sha256_file(prepared),
                "union_mask": str(mask.relative_to(root)),
                "union_mask_sha256": sha256_file(mask),
                "width": WIDTH,
                "height": HEIGHT,
                "union_mask_pixels": mask_pixels,
            }
        )
    propainter_manifest = root / "input_manifest.json"
    write_json(
        propainter_manifest,
        {
            "schema_version": 1,
            "frame_count": len(records),
            "frame_ids": list(FRAME_IDS),
            "frame_records": records,
        },
    )
    candidates = tmp_path / "candidates"
    for sequence_index in range(len(FRAME_IDS)):
        save_rgb(candidates / f"{sequence_index:04d}.png", (90, 110, 130))
    return {
        "root": root,
        "manifest": propainter_manifest,
        "records": records,
        "candidates": candidates,
        "output": tmp_path / "boundary-input.json",
    }


def build_args(fixture: dict[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(
        propainter_manifest=fixture["manifest"],
        candidate_frames_dir=fixture["candidates"],
        protected_mask_dir=None,
        generation_mask_dilation_pixels=48,
        outer_collar_pixels=6,
        maximum_boundary_color_p95_delta=24.0,
        maximum_boundary_gradient_p95_delta=32.0,
        output=fixture["output"],
    )


def test_builds_hash_bound_compositor_manifest_with_defaults(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    created_at = datetime(2026, 7, 18, 14, 0, tzinfo=UTC)

    manifest = build_manifest(build_args(fixture), created_at=created_at)

    assert manifest["kind"] == OUTPUT_KIND
    assert manifest["created_at"] == "2026-07-18T14:00:00+00:00"
    assert manifest["config"] == {
        "outer_collar_pixels": 6,
        "blend_curve": "smoothstep_euclidean_distance",
        "maximum_boundary_color_p95_delta": 24.0,
        "maximum_boundary_gradient_p95_delta": 32.0,
    }
    assert manifest["frame_ids"] == list(FRAME_IDS)
    assert manifest["gates"]["protected_masks_present"] is False
    assert manifest["gates"]["protected_masks_hash_dimensions_and_binary_values_verified"] is True
    assert manifest["provenance"]["propainter_input_manifest"]["sha256"] == sha256_file(
        fixture["manifest"]
    )
    assert manifest["provenance"]["generation"] == {
        "mask_dilation_pixels": 48,
        "role": "candidate_generation_context_only",
        "used_as_compositor_source_aligned_mask": False,
    }
    for sequence_index, record in enumerate(manifest["frame_records"]):
        source_record = fixture["records"][sequence_index]
        assert record["sequence_index"] == sequence_index
        assert record["previous_composite"]["sha256"] == source_record["input_frame_sha256"]
        assert record["source_aligned_mask"]["sha256"] == source_record["union_mask_sha256"]
        assert record["candidate_fill"]["sha256"] == sha256_file(
            fixture["candidates"] / f"{sequence_index:04d}.png"
        )
        assert "protected_mask" not in record
        assert record["source_binding"]["prepared_input_is_byte_exact_source_rgb"] is True
    assert fixture["output"].is_file()

    composed = compose_manifest(fixture["output"], tmp_path / "composed")
    assert composed["frame_count"] == len(FRAME_IDS)
    assert composed["gates"]["all_structural_compositor_invariants_passed"] is True


def test_cli_boundary_defaults_are_six_twenty_four_and_thirty_two(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "--propainter-manifest",
            str(tmp_path / "input.json"),
            "--candidate-frames-dir",
            str(tmp_path / "candidates"),
            "--generation-mask-dilation-pixels",
            "48",
            "--output",
            str(tmp_path / "output.json"),
        ]
    )

    assert args.outer_collar_pixels == 6
    assert args.maximum_boundary_color_p95_delta == 24.0
    assert args.maximum_boundary_gradient_p95_delta == 32.0


def test_attaches_exact_binary_protected_masks(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    protected_dir = tmp_path / "protected"
    for sequence_index in range(len(FRAME_IDS)):
        save_mask(protected_dir / f"{sequence_index:04d}.png", x=9)
    args = build_args(fixture)
    args.protected_mask_dir = protected_dir

    manifest = build_manifest(args)

    assert manifest["gates"]["protected_masks_hash_dimensions_and_binary_values_verified"] is True
    for sequence_index, record in enumerate(manifest["frame_records"]):
        assert record["protected_mask"]["sha256"] == sha256_file(
            protected_dir / f"{sequence_index:04d}.png"
        )


@pytest.mark.parametrize(
    ("directory_key", "mutation", "expected_message"),
    [
        ("candidates", "missing", "missing=['0001.png']"),
        ("candidates", "extra", "extra=['0002.png']"),
        ("protected", "missing", "missing=['0001.png']"),
        ("protected", "extra", "extra=['0002.png']"),
    ],
)
def test_rejects_missing_or_extra_frames(
    tmp_path: Path,
    directory_key: str,
    mutation: str,
    expected_message: str,
) -> None:
    fixture = make_fixture(tmp_path)
    protected = tmp_path / "protected"
    for sequence_index in range(len(FRAME_IDS)):
        save_mask(protected / f"{sequence_index:04d}.png", x=9)
    args = build_args(fixture)
    args.protected_mask_dir = protected
    directory = fixture["candidates"] if directory_key == "candidates" else protected
    if mutation == "missing":
        (directory / "0001.png").unlink()
    else:
        save_rgb(directory / "0002.png", (1, 2, 3)) if directory_key == "candidates" else save_mask(
            directory / "0002.png", x=9
        )

    with pytest.raises(PropainterBoundaryManifestError, match=re.escape(expected_message)):
        build_manifest(args)

    assert not fixture["output"].exists()


def test_rejects_out_of_order_records(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    source = json.loads(fixture["manifest"].read_text(encoding="utf-8"))
    source["frame_records"][0]["sequence_index"] = 1
    write_json(fixture["manifest"], source)

    with pytest.raises(PropainterBoundaryManifestError, match="must be ordered"):
        build_manifest(build_args(fixture))

    assert not fixture["output"].exists()


def test_rejects_declared_hash_mismatch(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    source = json.loads(fixture["manifest"].read_text(encoding="utf-8"))
    source["frame_records"][0]["union_mask_sha256"] = "0" * 64
    write_json(fixture["manifest"], source)

    with pytest.raises(PropainterBoundaryManifestError, match="union_mask SHA-256 mismatch"):
        build_manifest(build_args(fixture))

    assert not fixture["output"].exists()


def test_rejects_prepared_input_that_is_not_exact_source(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    prepared = fixture["root"] / fixture["records"][0]["input_frame"]
    save_rgb(prepared, (1, 2, 3))
    source = json.loads(fixture["manifest"].read_text(encoding="utf-8"))
    source["frame_records"][0]["input_frame_sha256"] = sha256_file(prepared)
    write_json(fixture["manifest"], source)

    with pytest.raises(PropainterBoundaryManifestError, match="not byte-exact source RGB"):
        build_manifest(build_args(fixture))

    assert not fixture["output"].exists()


@pytest.mark.parametrize("target", ["union", "protected"])
def test_rejects_nonbinary_masks(tmp_path: Path, target: str) -> None:
    fixture = make_fixture(tmp_path)
    args = build_args(fixture)
    if target == "union":
        mask = fixture["root"] / fixture["records"][0]["union_mask"]
        save_mask(mask, x=3, value=127)
        source = json.loads(fixture["manifest"].read_text(encoding="utf-8"))
        source["frame_records"][0]["union_mask_sha256"] = sha256_file(mask)
        write_json(fixture["manifest"], source)
    else:
        protected = tmp_path / "protected"
        for sequence_index in range(len(FRAME_IDS)):
            save_mask(protected / f"{sequence_index:04d}.png", x=9)
        save_mask(protected / "0000.png", x=9, value=127)
        args.protected_mask_dir = protected

    with pytest.raises(PropainterBoundaryManifestError, match="must be binary 0/255"):
        build_manifest(args)

    assert not fixture["output"].exists()


def test_rejects_candidate_dimension_mismatch(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    save_rgb(fixture["candidates"] / "0001.png", (1, 2, 3), shape=(HEIGHT - 1, WIDTH))

    with pytest.raises(PropainterBoundaryManifestError, match="candidate fill dimensions differ"):
        build_manifest(build_args(fixture))

    assert not fixture["output"].exists()


def test_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    fixture["output"].write_text("keep", encoding="utf-8")

    with pytest.raises(PropainterBoundaryManifestError, match="refusing to overwrite"):
        build_manifest(build_args(fixture))

    assert fixture["output"].read_text(encoding="utf-8") == "keep"
