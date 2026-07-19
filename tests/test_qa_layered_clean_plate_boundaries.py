from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageFilter
from scipy import ndimage

from scripts.qa_layered_clean_plate_boundaries import (
    MANIFEST_KIND,
    mask_alignment_metrics,
    seam_continuity_metrics,
    sha256_file,
    validate,
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def asset(path: Path, root: Path, *, channel: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path.relative_to(root)),
        "sha256": sha256_file(path),
    }
    if channel is not None:
        record.update({"channel": channel, "threshold": 128})
    return record


def make_fixture(
    tmp_path: Path,
    *,
    target_shift: int = 0,
    seam_delta: int = 0,
    blur_core: bool = False,
) -> dict[str, Any]:
    root = tmp_path / "fixture"
    source_path = root / "assets" / "source.png"
    composite_path = root / "assets" / "composite.png"
    removal_path = root / "assets" / "removal.png"
    target_path = root / "assets" / "target.png"
    for path in (source_path, composite_path, removal_path, target_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    height, width = 32, 48
    removal = np.zeros((height, width), dtype=np.uint8)
    removal[8:24, 12:36] = 255
    target = np.zeros_like(removal)
    target[8:24, 12 + target_shift : 36 + target_shift] = 255
    editable = ndimage.binary_dilation(
        removal > 0,
        structure=np.ones((3, 3), dtype=bool),
        iterations=3,
        border_value=0,
    )
    y, x = np.indices((height, width))
    texture = 50 + 4 * ((x + y) % 2)
    source = np.repeat(texture[..., None], 3, axis=2).astype(np.uint8)
    composite = source.copy()
    composite[editable] += seam_delta
    if blur_core:
        blurred = np.asarray(
            Image.fromarray(source).filter(ImageFilter.GaussianBlur(radius=3.0)),
            dtype=np.uint8,
        )
        replacement_core = ndimage.binary_erosion(
            removal > 0,
            structure=np.ones((3, 3), dtype=bool),
            iterations=2,
            border_value=0,
        )
        composite[replacement_core] = blurred[replacement_core]
    Image.fromarray(source).save(source_path)
    Image.fromarray(composite).save(composite_path)
    Image.fromarray(removal).save(removal_path)
    target_rgba = np.zeros((height, width, 4), dtype=np.uint8)
    target_rgba[..., :3] = 120
    target_rgba[..., 3] = target
    Image.fromarray(target_rgba).save(target_path)
    editable_path = root / "assets" / "editable.png"
    Image.fromarray(editable.astype(np.uint8) * 255).save(editable_path)

    thresholds = {
        "minimum_mask_iou": 0.95,
        "minimum_mask_precision": 0.95,
        "minimum_mask_recall": 0.95,
        "maximum_boundary_distance_p95_px": 0.1,
        "maximum_seam_color_p95_abs_rgb_delta": 5.0,
        "maximum_seam_gradient_p95_abs_rgb_delta": 10.0,
        "minimum_core_to_local_ring_laplacian_energy_ratio": 0.25,
    }
    manifest = {
        "schema_version": 1,
        "kind": MANIFEST_KIND,
        "frame_count": 1,
        "thresholds": thresholds,
        "texture_sampling": {
            "core_erosion_px": 4,
            "local_ring_inner_px": 2,
            "local_ring_outer_px": 6,
            "minimum_region_pixels": 64,
            "winsor_quantile": 0.95,
        },
        "frame_records": [
            {
                "sequence_index": 0,
                "frame_id": "frame-0000",
                "removal_core_mask": asset(removal_path, root, channel="luma"),
                "target_matte": asset(target_path, root, channel="alpha"),
                "editable_mask": asset(editable_path, root, channel="luma"),
                "source_rgb": asset(source_path, root),
                "composite_rgb": asset(composite_path, root),
            }
        ],
    }
    manifest_path = root / "manifest.json"
    write_json(manifest_path, manifest)
    return {"root": root, "manifest": manifest, "manifest_path": manifest_path}


def args_for(fixture: dict[str, Any], *, visuals: bool = False) -> argparse.Namespace:
    root = fixture["root"]
    return argparse.Namespace(
        input_manifest=fixture["manifest_path"],
        output_report=root / "qa" / "report.json",
        overlay_dir=root / "qa" / "overlays" if visuals else None,
        contact_sheet=root / "qa" / "contact.png" if visuals else None,
        contact_sheet_samples=1,
    )


def test_exact_masks_and_continuous_composite_pass_all_fail_closed_gates(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)

    report = validate(args_for(fixture))

    assert report["status"] == "technical_passed"
    assert report["boundary_texture_gate_passed"] is True
    assert report["promotion_approved"] is False
    assert all(gate["passed"] is True for gate in report["gates"].values())
    frame = report["frame_records"][0]
    assert frame["alignment"]["mask_iou"] == 1.0
    assert frame["alignment"]["mask_precision"] == 1.0
    assert frame["alignment"]["mask_recall"] == 1.0
    assert frame["alignment"]["boundary_distance_p95_px"] == 0.0
    assert frame["seam"]["seam_color_p95_abs_rgb_delta"] == 4.0
    assert frame["texture"]["core_to_local_ring_laplacian_energy_ratio"] >= 0.25
    assert Path(report["input_manifest"]) == fixture["manifest_path"].resolve()
    assert Path(args_for(fixture).output_report).is_file()


def test_offset_target_and_hard_color_seam_fail_geometry_and_continuity(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path, target_shift=2, seam_delta=120)

    report = validate(args_for(fixture))

    assert report["status"] == "technical_failed"
    assert report["boundary_texture_gate_passed"] is False
    assert report["promotion_approved"] is False
    gates = report["gates"]
    assert gates["mask_iou_gte"]["passed"] is False
    assert gates["mask_precision_gte"]["passed"] is False
    assert gates["mask_recall_gte"]["passed"] is False
    assert gates["boundary_distance_p95_lte"]["passed"] is False
    assert gates["seam_color_p95_lte"]["passed"] is False
    assert gates["seam_gradient_p95_lte"]["passed"] is False
    frame = report["frame_records"][0]
    assert frame["alignment"]["boundary_distance_p95_px"] > 0.1
    assert frame["seam"]["seam_color_p95_abs_rgb_delta"] > 100.0
    assert frame["seam"]["seam_gradient_p95_abs_rgb_delta"] > 100.0


def test_blurred_core_fails_texture_gate_while_boundary_gates_pass(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path, blur_core=True)

    report = validate(args_for(fixture))

    assert report["status"] == "technical_failed"
    gates = report["gates"]
    assert gates["mask_iou_gte"]["passed"] is True
    assert gates["boundary_distance_p95_lte"]["passed"] is True
    assert gates["seam_color_p95_lte"]["passed"] is True
    assert gates["seam_gradient_p95_lte"]["passed"] is True
    texture_gate = gates["core_to_local_ring_laplacian_energy_ratio_gte"]
    assert texture_gate["passed"] is False
    texture = report["frame_records"][0]["texture"]
    assert texture["core_to_local_ring_laplacian_energy_ratio"] < 0.25
    assert (
        texture["core_laplacian_energy_robust_mean"]
        < texture["local_ring_laplacian_energy_robust_mean"]
    )


def test_alignment_uses_core_and_seam_uses_editable_collar(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    record = fixture["manifest"]["frame_records"][0]
    root = fixture["root"]
    core = np.asarray(Image.open(root / record["removal_core_mask"]["path"]).convert("L")) > 0
    editable = np.asarray(Image.open(root / record["editable_mask"]["path"]).convert("L")) > 0
    composite_path = root / record["composite_rgb"]["path"]
    composite = np.asarray(Image.open(composite_path).convert("RGB"), dtype=np.uint8).copy()
    composite[core] += 80
    Image.fromarray(composite).save(composite_path)
    record["composite_rgb"]["sha256"] = sha256_file(composite_path)
    write_json(fixture["manifest_path"], fixture["manifest"])

    report = validate(args_for(fixture))

    assert np.any(editable & ~core)
    assert report["frame_records"][0]["alignment"]["mask_iou"] == 1.0
    assert report["gates"]["seam_color_p95_lte"]["passed"] is True
    assert report["gates"]["seam_gradient_p95_lte"]["passed"] is True
    assert seam_continuity_metrics(composite, core)["seam_color_p95_abs_rgb_delta"] > 70
    assert seam_continuity_metrics(composite, editable)["seam_color_p95_abs_rgb_delta"] <= 5


def test_missing_threshold_fails_closed_before_writing_report(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    del fixture["manifest"]["thresholds"]["minimum_mask_iou"]
    write_json(fixture["manifest_path"], fixture["manifest"])
    args = args_for(fixture)

    with pytest.raises(ValueError, match="missing required keys: minimum_mask_iou"):
        validate(args)

    assert not args.output_report.exists()


def test_hash_mismatch_and_empty_mask_are_rejected_as_invalid_evidence(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    fixture["manifest"]["frame_records"][0]["target_matte"]["sha256"] = "0" * 64
    write_json(fixture["manifest_path"], fixture["manifest"])
    with pytest.raises(ValueError, match=r"target_matte\.sha256 does not match"):
        validate(args_for(fixture))

    fixture = make_fixture(tmp_path / "empty")
    mask_record = fixture["manifest"]["frame_records"][0]["removal_core_mask"]
    mask_path = fixture["root"] / mask_record["path"]
    Image.fromarray(np.zeros((32, 48), dtype=np.uint8)).save(mask_path)
    mask_record["sha256"] = sha256_file(mask_path)
    write_json(fixture["manifest_path"], fixture["manifest"])
    with pytest.raises(ValueError, match="removal_core_mask is empty"):
        validate(args_for(fixture))


def test_optional_overlay_and_contact_sheet_are_hashed_outputs(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    args = args_for(fixture, visuals=True)

    report = validate(args)

    overlay = report["frame_records"][0]["overlay"]
    assert Path(overlay["path"]).is_file()
    assert overlay["sha256"] == sha256_file(Path(overlay["path"]))
    contact = report["outputs"]["contact_sheet"]
    assert Path(contact["path"]).is_file()
    assert contact["sha256"] == sha256_file(Path(contact["path"]))
    assert contact["sample_count"] == 1


def test_flat_composite_is_rejected_as_unmeasurable_texture_evidence(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    record = fixture["manifest"]["frame_records"][0]["composite_rgb"]
    composite_path = fixture["root"] / record["path"]
    Image.fromarray(np.full((32, 48, 3), 50, dtype=np.uint8)).save(composite_path)
    record["sha256"] = sha256_file(composite_path)
    write_json(fixture["manifest_path"], fixture["manifest"])

    with pytest.raises(ValueError, match="no measurable Laplacian texture energy"):
        validate(args_for(fixture))


def test_metric_helpers_keep_precision_and_recall_direction_explicit() -> None:
    removal = np.zeros((8, 10), dtype=bool)
    removal[2:6, 2:7] = True
    target = np.zeros_like(removal)
    target[2:6, 4:9] = True

    alignment = mask_alignment_metrics(removal, target)

    assert alignment["intersection_pixels"] == 12
    assert alignment["removal_mask_pixels"] == 20
    assert alignment["target_matte_pixels"] == 20
    assert alignment["mask_precision"] == 0.6
    assert alignment["mask_recall"] == 0.6
    assert alignment["mask_iou"] == pytest.approx(12 / 28)

    composite = np.full((8, 10, 3), 50, dtype=np.uint8)
    composite[removal] = 80
    seam = seam_continuity_metrics(composite, removal)
    assert seam["seam_color_p95_abs_rgb_delta"] == 30.0
    assert seam["seam_gradient_p95_abs_rgb_delta"] == 30.0
