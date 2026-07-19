from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "build_source_aligned_scene_edit_masks.py"
SPEC = importlib.util.spec_from_file_location(
    "build_source_aligned_scene_edit_masks",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

HEIGHT = 6
WIDTH = 8
FRAME_IDS = ("000000", "000001")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def save_rgb(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((HEIGHT, WIDTH, 3), color, dtype=np.uint8)).save(path)


def save_mask(path: Path, points: set[tuple[int, int]]) -> np.ndarray:
    value = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for x, y in points:
        value[y, x] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value).save(path)
    return value == 255


def save_labels(
    path: Path,
    values: dict[tuple[int, int], int],
    *,
    shape: tuple[int, int] = (HEIGHT, WIDTH),
) -> np.ndarray:
    labels = np.zeros(shape, dtype=np.uint16)
    for (x, y), label in values.items():
        labels[y, x] = label
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(labels).save(path)
    return labels


def asset(path: Path, **extra: Any) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": MODULE.sha256_file(path),
        **extra,
    }


def write_manifest(
    path: Path,
    *,
    round_index: int,
    records: list[dict[str, Any]],
) -> None:
    write_json(
        path,
        {
            "schema_version": 1,
            "kind": MODULE.INPUT_KIND,
            "round_index": round_index,
            "frame_records": records,
        },
    )


def build_args(manifest: Path, output: Path) -> argparse.Namespace:
    return argparse.Namespace(input_manifest=manifest, output=output)


def make_round1_fixture(tmp_path: Path) -> tuple[Path, dict[str, np.ndarray]]:
    records = []
    expected = {}
    for sequence_index, frame_id in enumerate(FRAME_IDS):
        source = tmp_path / "source" / f"{frame_id}.png"
        observed_path = tmp_path / "observed" / f"{frame_id}.png"
        save_rgb(source, (20 + sequence_index, 30, 40))
        observed = save_mask(
            observed_path,
            {(1 + sequence_index, 1), (2 + sequence_index, 2)},
        )
        expected[frame_id] = observed
        source_sha = MODULE.sha256_file(source)
        records.append(
            {
                "sequence_index": sequence_index,
                "frame_id": frame_id,
                "target_label": 7,
                "exact_source_rgb": asset(source),
                "observed_target_mask": asset(
                    observed_path,
                    source_rgb_sha256=source_sha,
                ),
                "previous_edited_region_mask": None,
                "previous_contributor_labels": None,
            }
        )
    manifest = tmp_path / "round1.json"
    write_manifest(manifest, round_index=1, records=records)
    return manifest, expected


def make_round2_fixture(tmp_path: Path) -> tuple[Path, dict[str, np.ndarray]]:
    records = []
    expected = {}
    for sequence_index, frame_id in enumerate(FRAME_IDS):
        source = tmp_path / "source" / f"{frame_id}.png"
        observed_path = tmp_path / "observed" / f"{frame_id}.png"
        previous_path = tmp_path / "previous" / f"{frame_id}.png"
        labels_path = tmp_path / "labels" / f"{frame_id}.png"
        save_rgb(source, (50 + sequence_index, 60, 70))
        observed = save_mask(observed_path, {(1, 1), (2, 1), (4, 3)})
        previous = save_mask(previous_path, {(1, 1), (2, 1), (3, 2)})
        labels = save_labels(
            labels_path,
            {
                (1, 1): 5,
                (2, 1): 7,
                (3, 2): 7,
                (6, 4): 7,
            },
        )
        expected[frame_id] = (observed & ~previous) | (previous & (labels == 7))
        source_sha = MODULE.sha256_file(source)
        records.append(
            {
                "sequence_index": sequence_index,
                "frame_id": frame_id,
                "target_label": 7,
                "exact_source_rgb": asset(source),
                "observed_target_mask": asset(
                    observed_path,
                    source_rgb_sha256=source_sha,
                ),
                "previous_edited_region_mask": asset(previous_path),
                "previous_contributor_labels": asset(labels_path),
            }
        )
    manifest = tmp_path / "round2.json"
    write_manifest(manifest, round_index=2, records=records)
    return manifest, expected


def test_round1_emits_observed_masks_and_binds_exact_source_hashes(tmp_path: Path) -> None:
    manifest, expected = make_round1_fixture(tmp_path)
    output = tmp_path / "output"

    report = MODULE.build_scene_edit_masks(build_args(manifest, output))

    assert report["kind"] == MODULE.REPORT_KIND
    assert report["status"] == "technical_passed"
    assert report["round_index"] == 1
    assert report["frame_count"] == 2
    assert all(report["gates"].values())
    for record in report["frame_records"]:
        frame_id = record["frame_id"]
        output_mask = (
            np.asarray(Image.open(output / record["scene_edit_mask"]["path"]).convert("L")) == 255
        )
        assert np.array_equal(output_mask, expected[frame_id])
        assert record["previous_edited_region_mask"] is None
        assert record["previous_contributor_labels"] is None
        assert (
            record["observed_target_mask"]["source_rgb_sha256"]
            == record["exact_source_rgb"]["sha256"]
        )
        assert record["counts"]["scene_edit_mask_pixels"] == int(expected[frame_id].sum())

    report_path = output / "source_aligned_scene_edit_mask_report.json"
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["kind"] == MODULE.RECEIPT_KIND
    assert receipt["report_sha256"] == MODULE.sha256_file(report_path)
    assert receipt["input_manifest_sha256"] == MODULE.sha256_file(manifest)
    assert receipt["output_mask_set_sha256"] == report["output_mask_set_sha256"]
    assert receipt["exact_source_rgb_sha256_bound_per_frame"] is True


def test_round2_uses_observed_only_outside_previous_and_target_contributors_inside(
    tmp_path: Path,
) -> None:
    manifest, expected = make_round2_fixture(tmp_path)
    output = tmp_path / "output"

    report = MODULE.build_scene_edit_masks(build_args(manifest, output))

    assert report["round_index"] == 2
    assert report["aggregate_counts"] == {
        "observed_target_pixels": 6,
        "previous_edited_region_pixels": 6,
        "observed_target_inside_previous_pixels": 4,
        "observed_target_outside_previous_pixels": 2,
        "target_contributor_inside_previous_pixels": 4,
        "scene_edit_mask_pixels": 6,
    }
    for record in report["frame_records"]:
        actual = (
            np.asarray(Image.open(output / record["scene_edit_mask"]["path"]).convert("L")) == 255
        )
        assert np.array_equal(actual, expected[record["frame_id"]])
        assert not actual[1, 1]
        assert actual[1, 2]
        assert actual[2, 3]
        assert actual[3, 4]
        assert not actual[4, 6]
        assert all(record["gates"].values())


def test_rejects_observed_mask_not_bound_to_exact_source_rgb(tmp_path: Path) -> None:
    manifest, _ = make_round1_fixture(tmp_path)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["frame_records"][0]["observed_target_mask"]["source_rgb_sha256"] = "0" * 64
    write_json(manifest, value)
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="not bound to the exact source RGB"):
        MODULE.build_scene_edit_masks(build_args(manifest, output))

    assert not output.exists()


def test_rejects_any_declared_asset_hash_mismatch_without_output(tmp_path: Path) -> None:
    manifest, _ = make_round2_fixture(tmp_path)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["frame_records"][0]["previous_contributor_labels"]["sha256"] = "f" * 64
    write_json(manifest, value)
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        MODULE.build_scene_edit_masks(build_args(manifest, output))

    assert not output.exists()


def test_rejects_dimension_mismatch(tmp_path: Path) -> None:
    manifest, _ = make_round2_fixture(tmp_path)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    labels_path = tmp_path / "wrong_labels.png"
    save_labels(labels_path, {(1, 1): 7}, shape=(HEIGHT - 1, WIDTH))
    value["frame_records"][0]["previous_contributor_labels"] = asset(labels_path)
    write_json(manifest, value)

    with pytest.raises(ValueError, match="contributor-label dimensions differ from source"):
        MODULE.build_scene_edit_masks(build_args(manifest, tmp_path / "output"))


@pytest.mark.parametrize("asset_key", ["observed_target_mask", "previous_edited_region_mask"])
def test_rejects_empty_input_masks(tmp_path: Path, asset_key: str) -> None:
    manifest, _ = make_round2_fixture(tmp_path)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    empty_path = tmp_path / f"empty_{asset_key}.png"
    save_mask(empty_path, set())
    replacement = asset(empty_path)
    if asset_key == "observed_target_mask":
        replacement["source_rgb_sha256"] = value["frame_records"][0]["exact_source_rgb"]["sha256"]
    value["frame_records"][0][asset_key] = replacement
    write_json(manifest, value)

    with pytest.raises(ValueError, match="must not be empty"):
        MODULE.build_scene_edit_masks(build_args(manifest, tmp_path / "output"))


def test_rejects_empty_derived_scene_edit_mask(tmp_path: Path) -> None:
    manifest, _ = make_round2_fixture(tmp_path)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    observed_path = tmp_path / "inside_previous_only.png"
    save_mask(observed_path, {(1, 1)})
    labels_path = tmp_path / "no_target_labels.png"
    save_labels(labels_path, {(1, 1): 5})
    source_sha = value["frame_records"][0]["exact_source_rgb"]["sha256"]
    value["frame_records"][0]["observed_target_mask"] = asset(
        observed_path,
        source_rgb_sha256=source_sha,
    )
    value["frame_records"][0]["previous_contributor_labels"] = asset(labels_path)
    write_json(manifest, value)

    with pytest.raises(ValueError, match="derived scene-edit mask must not be empty"):
        MODULE.build_scene_edit_masks(build_args(manifest, tmp_path / "output"))


def test_previous_inputs_are_forbidden_in_round1_and_required_as_a_pair_later(
    tmp_path: Path,
) -> None:
    round1_manifest, _ = make_round1_fixture(tmp_path / "round1")
    round1 = json.loads(round1_manifest.read_text(encoding="utf-8"))
    previous_path = tmp_path / "round1_previous.png"
    labels_path = tmp_path / "round1_labels.png"
    save_mask(previous_path, {(1, 1)})
    save_labels(labels_path, {(1, 1): 7})
    round1["frame_records"][0]["previous_edited_region_mask"] = asset(previous_path)
    round1["frame_records"][0]["previous_contributor_labels"] = asset(labels_path)
    write_json(round1_manifest, round1)

    with pytest.raises(ValueError, match="round 1 must not provide"):
        MODULE.build_scene_edit_masks(build_args(round1_manifest, tmp_path / "round1_output"))

    round2_manifest, _ = make_round2_fixture(tmp_path / "round2")
    round2 = json.loads(round2_manifest.read_text(encoding="utf-8"))
    round2["frame_records"][0]["previous_contributor_labels"] = None
    write_json(round2_manifest, round2)

    with pytest.raises(ValueError, match="must be supplied together"):
        MODULE.build_scene_edit_masks(build_args(round2_manifest, tmp_path / "round2_output"))


def test_rejects_nonbinary_observed_mask(tmp_path: Path) -> None:
    manifest, _ = make_round1_fixture(tmp_path)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    mask_path = tmp_path / "nonbinary.png"
    pixels = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    pixels[1, 1] = 127
    Image.fromarray(pixels).save(mask_path)
    source_sha = value["frame_records"][0]["exact_source_rgb"]["sha256"]
    value["frame_records"][0]["observed_target_mask"] = asset(
        mask_path,
        source_rgb_sha256=source_sha,
    )
    write_json(manifest, value)

    with pytest.raises(ValueError, match="binary 0/255"):
        MODULE.build_scene_edit_masks(build_args(manifest, tmp_path / "output"))
