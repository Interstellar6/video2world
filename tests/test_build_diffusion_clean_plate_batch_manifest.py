from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from scripts.build_diffusion_clean_plate_batch_manifest import (
    DiffusionBatchManifestBuildError,
    build_manifest,
    sha256_file,
)
from scripts.run_diffusion_clean_plate_batch import resolve_batch_input

HEIGHT = 9
WIDTH = 12
FRAME_IDS = ("000048", "000049")


def save_rgb(
    path: Path,
    color: tuple[int, int, int],
    *,
    shape: tuple[int, int] = (HEIGHT, WIDTH),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = np.full((shape[0], shape[1], 3), color, dtype=np.uint8)
    Image.fromarray(value).save(path)


def save_mask(path: Path, *, x: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    value[3:6, x : x + 2] = 255
    Image.fromarray(value).save(path)
    return int(np.count_nonzero(value))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_fixture(tmp_path: Path) -> dict[str, Any]:
    input_root = tmp_path / "propainter"
    records = []
    for sequence_index, frame_id in enumerate(FRAME_IDS):
        source = input_root / "source" / f"{frame_id}.png"
        prepared = input_root / "input" / "frames" / f"{sequence_index:04d}.png"
        mask = input_root / "input" / "masks" / f"{sequence_index:04d}.png"
        save_rgb(source, (30 + sequence_index, 50, 70))
        prepared.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, prepared)
        pixels = save_mask(mask, x=2 + sequence_index)
        records.append(
            {
                "sequence_index": sequence_index,
                "frame_id": frame_id,
                "source_frame": str(source),
                "source_frame_sha256": sha256_file(source)[0],
                "input_frame": str(prepared),
                "input_frame_sha256": sha256_file(prepared)[0],
                "union_mask": str(mask),
                "union_mask_sha256": sha256_file(mask)[0],
                "width": WIDTH,
                "height": HEIGHT,
                "union_mask_pixels": pixels,
            }
        )
    propainter_manifest = input_root / "input_manifest.json"
    write_json(
        propainter_manifest,
        {
            "schema_version": 1,
            "frame_count": len(records),
            "frame_ids": list(FRAME_IDS),
            "frame_records": records,
        },
    )
    pbr_dir = tmp_path / "pbr"
    fallback_dir = tmp_path / "fallback"
    pbr_dir.mkdir()
    fallback_dir.mkdir()
    save_rgb(pbr_dir / "0000.png", (100, 110, 120))
    save_rgb(pbr_dir / "0001.png", (130, 140, 150))
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model_index.json").write_text("{}", encoding="utf-8")
    return {
        "manifest": propainter_manifest,
        "records": records,
        "pbr_dir": pbr_dir,
        "fallback_dir": fallback_dir,
        "model_dir": model_dir,
        "output": tmp_path / "batch-manifest.json",
    }


def build_args(fixture: dict[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(
        propainter_manifest=fixture["manifest"],
        pbr_guide_dir=fixture["pbr_dir"],
        fallback_guide_dir=None,
        output=fixture["output"],
        model_path=fixture["model_dir"],
        model_repo="diffusers/test-sdxl-inpaint",
        model_revision="fixed-test-revision",
        device="cuda:test",
        dtype="float16",
        variant="fp16",
        prompt="Restore only the occluded background.",
        negative_prompt="foreground object, blur",
        context_dilation_pixels=4,
        boundary_outer_collar_pixels=2,
        base_seed=700,
        seed_stride=3,
        num_inference_steps=20,
        guidance_scale=6.0,
        strength=0.9,
    )


def test_builds_runner_compatible_hash_bound_ordered_manifest(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    created_at = datetime(2026, 7, 18, 13, 0, tzinfo=UTC)
    manifest = build_manifest(build_args(fixture), created_at=created_at)

    assert manifest["kind"] == "video2world.diffusion_clean_plate_batch_input"
    assert manifest["created_at"] == "2026-07-18T13:00:00+00:00"
    assert manifest["frame_ids"] == list(FRAME_IDS)
    assert manifest["gates"]["ownership_masks_bound_to_exact_source_rgb_sha256"] is True
    assert fixture["output"].is_file()

    for sequence_index, record in enumerate(manifest["frame_records"]):
        source = record["source_rgb"]
        ownership = record["ownership_mask"]
        guide = record["guide_rgb"]
        assert record["sequence_index"] == sequence_index
        assert record["ownership_mask_source_rgb_sha256"] == source["sha256"]
        assert record["guide_source_rgb_sha256"] == source["sha256"]
        assert source["sha256"] == fixture["records"][sequence_index]["source_frame_sha256"]
        assert ownership["sha256"] == fixture["records"][sequence_index]["union_mask_sha256"]
        assert guide["sha256"] == sha256_file(fixture["pbr_dir"] / f"{sequence_index:04d}.png")[0]
        assert record["guide_selection"] == {
            "source": "pbr",
            "matched_stem": f"{sequence_index:04d}",
        }
        assert source["dimensions"] == [WIDTH, HEIGHT]
        assert ownership["dimensions"] == [WIDTH, HEIGHT]
        assert guide["dimensions"] == [WIDTH, HEIGHT]

    resolved = resolve_batch_input(fixture["output"])
    assert [frame.frame_id for frame in resolved.frames] == list(FRAME_IDS)
    assert [frame.seed for frame in resolved.frames] == [700, 703]


def test_uses_fallback_only_for_missing_primary_frame(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    (fixture["pbr_dir"] / "0001.png").unlink()
    save_rgb(fixture["fallback_dir"] / "0001.png", (210, 180, 160))
    args = build_args(fixture)
    args.fallback_guide_dir = fixture["fallback_dir"]

    manifest = build_manifest(args)

    assert [record["guide_selection"]["source"] for record in manifest["frame_records"]] == [
        "pbr",
        "fallback",
    ]
    assert (
        manifest["frame_records"][1]["guide_rgb"]["sha256"]
        == sha256_file(fixture["fallback_dir"] / "0001.png")[0]
    )


def test_fixed_seed_stride_zero_is_runner_compatible(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    args = build_args(fixture)
    args.seed_stride = 0

    build_manifest(args)
    resolved = resolve_batch_input(fixture["output"])

    assert [frame.seed for frame in resolved.frames] == [700, 700]


def test_rejects_non_contiguous_frame_order_without_output(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    source = json.loads(fixture["manifest"].read_text(encoding="utf-8"))
    source["frame_records"][0]["sequence_index"] = 1
    write_json(fixture["manifest"], source)

    with pytest.raises(DiffusionBatchManifestBuildError, match="must be ordered"):
        build_manifest(build_args(fixture))

    assert not fixture["output"].exists()


def test_rejects_input_hash_mismatch_without_output(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    source = json.loads(fixture["manifest"].read_text(encoding="utf-8"))
    source["frame_records"][1]["union_mask_sha256"] = "0" * 64
    write_json(fixture["manifest"], source)

    with pytest.raises(DiffusionBatchManifestBuildError, match="union_mask SHA-256 mismatch"):
        build_manifest(build_args(fixture))

    assert not fixture["output"].exists()


def test_rejects_prepared_rgb_that_is_not_exact_source(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    prepared = Path(fixture["records"][0]["input_frame"])
    save_rgb(prepared, (1, 2, 3))
    source = json.loads(fixture["manifest"].read_text(encoding="utf-8"))
    source["frame_records"][0]["input_frame_sha256"] = sha256_file(prepared)[0]
    write_json(fixture["manifest"], source)

    with pytest.raises(DiffusionBatchManifestBuildError, match="not exact source RGB"):
        build_manifest(build_args(fixture))


def test_rejects_missing_guide_without_output(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    (fixture["pbr_dir"] / "0001.png").unlink()

    with pytest.raises(DiffusionBatchManifestBuildError, match="missing guide for frame 000049"):
        build_manifest(build_args(fixture))

    assert not fixture["output"].exists()


def test_rejects_guide_dimension_mismatch_without_output(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    save_rgb(fixture["pbr_dir"] / "0001.png", (3, 4, 5), shape=(HEIGHT - 1, WIDTH))

    with pytest.raises(DiffusionBatchManifestBuildError, match="dimensions differ"):
        build_manifest(build_args(fixture))

    assert not fixture["output"].exists()


def test_rejects_manifest_frame_id_list_mismatch(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    source = json.loads(fixture["manifest"].read_text(encoding="utf-8"))
    source["frame_ids"] = list(reversed(FRAME_IDS))
    write_json(fixture["manifest"], source)

    with pytest.raises(DiffusionBatchManifestBuildError, match="frame_ids does not match"):
        build_manifest(build_args(fixture))


def test_rejects_ambiguous_frame_id_and_sequence_guides(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    save_rgb(fixture["pbr_dir"] / "000048.png", (200, 201, 202))

    with pytest.raises(DiffusionBatchManifestBuildError, match="ambiguous guides"):
        build_manifest(build_args(fixture))

    assert not fixture["output"].exists()
