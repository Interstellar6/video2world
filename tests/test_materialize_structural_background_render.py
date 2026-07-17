from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from scripts.compose_layered_clean_plate import validate_render_receipt
from scripts.materialize_structural_background_render import (
    materialize,
    sha256_file,
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def save_rgb(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value.astype(np.uint8)).save(path)


def save_mask(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value.astype(np.uint8) * 255).save(path)


def make_fixture(root: Path, *, cover_removal: bool = True) -> dict[str, Path | str]:
    geometry_dir = root / "geometry"
    texture_dir = root / "texture"
    geometry_records: list[dict[str, Any]] = []
    texture_records: list[dict[str, Any]] = []
    for sequence_index, frame_id in enumerate(("000048", "000049")):
        rgb = np.full((3, 4, 3), 40 + sequence_index * 20, dtype=np.uint8)
        rgb[1, 1] = [120, 100, 80]
        removal = np.zeros((3, 4), dtype=bool)
        removal[1, 1:3] = True
        synthetic = np.zeros((3, 4), dtype=bool)
        synthetic[1, 1] = True
        depth = np.full((3, 4), np.nan, dtype=np.float32)
        depth[removal] = [2.0, 2.5]
        if not cover_removal and sequence_index == 0:
            depth[1, 2] = np.nan

        completed_path = texture_dir / "frames" / f"{sequence_index:04d}.png"
        synthetic_path = texture_dir / "synthetic_masks" / f"{sequence_index:04d}.png"
        removal_path = geometry_dir / "cumulative_masks" / f"{sequence_index:04d}.png"
        depth_path = geometry_dir / "geometry_depth" / f"{sequence_index:04d}.npz"
        save_rgb(completed_path, rgb)
        save_mask(synthetic_path, synthetic)
        save_mask(removal_path, removal)
        depth_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(depth_path, depth=depth)
        geometry_records.append(
            {
                "sequence_index": sequence_index,
                "frame_id": frame_id,
                "removal_mask": str(removal_path),
                "removal_mask_sha256": sha256_file(removal_path),
                "removal_mask_pixels": int(removal.sum()),
                "geometry_depth": str(depth_path),
                "geometry_depth_sha256": sha256_file(depth_path),
                "geometry_assigned_pixels": int((np.isfinite(depth) & (depth > 0)).sum()),
                "outside_removal_mask_rgb_exact": True,
                "texture_partition_exact": True,
            }
        )
        texture_records.append(
            {
                "sequence_index": sequence_index,
                "frame_id": frame_id,
                "completed_frame": str(completed_path),
                "completed_frame_sha256": sha256_file(completed_path),
                "synthetic_mask": str(synthetic_path),
                "synthetic_mask_sha256": sha256_file(synthetic_path),
                "synthetic_pixels": int(synthetic.sum()),
                "synthetic_pixels_assigned_from_shared_atlas": int(synthetic.sum()),
                "outside_synthetic_mask_rgb_exact": True,
                "all_synthetic_pixels_assigned": True,
            }
        )

    geometry_report_path = geometry_dir / "planar_background_report.json"
    write_json(
        geometry_report_path,
        {
            "status": "geometry_completed_texture_pending",
            "gates": {
                "minimum_geometry_coverage": {"passed": True},
                "outside_removal_mask_rgb_exact": True,
                "texture_masks_partition_removal_mask": True,
            },
            "frame_records": geometry_records,
        },
    )
    geometry_sha = sha256_file(geometry_report_path)
    overrides = {
        "visual_quality": "pending_human_or_vlm_review",
        "wall_boundary_color_continuity": {
            "threshold_p95_abs_rgb_delta": 24.0,
            "observed_maximum": 96.0,
            "passed": False,
        },
        "wall_boundary_low_frequency_gradient_continuity": {
            "threshold_p95_abs_rgb_delta": 32.0,
            "observed_maximum": 99.0,
            "passed": False,
        },
    }
    override_keys = list(overrides)
    limitations = ["known wall-boundary artifact", "current Bedroom4 demo only"]
    accepted_report_path = texture_dir / "accepted_planar_texture_report.json"
    write_json(
        accepted_report_path,
        {
            "status": "accepted_for_round04_clean_plate",
            "accepted_with_limitations": True,
            "demo_use_approved": True,
            "eligible_as_round04_clean_plate": False,
            "eligible_as_current_demo_round04_clean_plate": True,
            "acceptance_scope": "current_demo_only",
            "promotion_approved": False,
            "overridden_gates": overrides,
            "acceptance": {
                "kind": "video2world.planar_texture_acceptance",
                "accepted_with_limitations": True,
                "demo_use_approved": True,
                "eligible_as_round04_clean_plate": False,
                "eligible_as_current_demo_round04_clean_plate": True,
                "acceptance_scope": "current_demo_only",
                "promotion_approved": False,
                "review_limitations": limitations,
                "override_gate_keys": override_keys,
                "overridden_gates": overrides,
            },
            "gates": {
                "all_synthetic_pixels_assigned_from_shared_atlas": True,
                "measured_atlas_texels_rgb_exact": True,
                "new_depth_normal_estimation_before_pgsr": (
                    "required_after_visual_acceptance"
                ),
                (
                    "observed_synthetic_anchor_and_interpolated_partition_"
                    "rendered_synthetic_pixels"
                ): True,
                "outside_synthetic_mask_rgb_exact": True,
                "protected_neighbors_outside_synthetic_mask_rgb_exact": True,
                "protected_sam_neighbor_pixels_rgb_exact": True,
                "rejected_texture_inputs_not_promotable": True,
                "same_atlas_texel_has_identical_rgb_across_views": True,
                "synthetic_anchor_atlas_texels_rgb_exact": True,
                "synthetic_anchor_claims_measured_donor": False,
                "visual_quality": "accepted_with_limitations",
                "wall_boundary_color_continuity": overrides[
                    "wall_boundary_color_continuity"
                ],
                "wall_boundary_low_frequency_gradient_continuity": overrides[
                    "wall_boundary_low_frequency_gradient_continuity"
                ],
                "wall_object_boundary_colors_excluded_from_synthetic_fit": True,
            },
            "provenance": {
                "texture_provenance": "hybrid_atlas",
                "claims_measured_donor": False,
            },
            "planar_geometry_report": str(geometry_report_path),
            "planar_geometry_report_sha256": geometry_sha,
            "frame_records": texture_records,
        },
    )
    return {
        "accepted_report": accepted_report_path,
        "accepted_report_sha256": sha256_file(accepted_report_path),
        "geometry_report": geometry_report_path,
        "geometry_report_sha256": geometry_sha,
    }


def run_materialize(fixture: dict[str, Path | str], output: Path) -> tuple[dict, dict]:
    return materialize(
        accepted_report_path=Path(fixture["accepted_report"]),
        accepted_report_sha256=str(fixture["accepted_report_sha256"]),
        geometry_report_path=Path(fixture["geometry_report"]),
        geometry_report_sha256=str(fixture["geometry_report_sha256"]),
        output=output,
    )


def test_materializes_verified_rgba_depth_and_limited_receipts(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path / "source")
    output = tmp_path / "output"

    index, batch_receipt = run_materialize(fixture, output)

    assert index["frame_count"] == 2
    assert index["promotion_approved"] is False
    assert index["eligible_as_round04_clean_plate"] is False
    assert index["eligible_as_current_demo_round04_clean_plate"] is True
    assert batch_receipt["index_sha256"] == sha256_file(
        output / "structural_background_render_index.json"
    )
    frame = index["frame_records"][0]
    rgba = np.asarray(Image.open(output / frame["rgba"]["path"]))
    depth = np.load(output / frame["depth"]["path"], allow_pickle=False)
    source_rgb = np.asarray(Image.open(frame["source_completed_frame"]["path"]).convert("RGB"))
    removal = np.asarray(
        Image.open(frame["source_cumulative_removal_mask"]["path"]).convert("L")
    ) == 255
    assert np.array_equal(rgba[..., :3], source_rgb)
    assert np.array_equal(rgba[..., 3] == 255, removal)
    assert np.all(np.isfinite(depth[removal]) & (depth[removal] > 0))
    assert np.all(np.isnan(depth[~removal]))
    receipt = json.loads(
        (output / frame["receipt"]["path"]).read_text(encoding="utf-8")
    )
    assert receipt["kind"] == "video2world.structural_background_render_receipt"
    assert receipt["accepted_texture_report_sha256"] == fixture[
        "accepted_report_sha256"
    ]
    assert receipt["accepted_texture_report"]["sha256"] == fixture[
        "accepted_report_sha256"
    ]
    assert receipt["failed_metrics"] == receipt["overridden_gates"]
    assert receipt["promotion_approved"] is False
    validated = validate_render_receipt(
        {
            "rgba": frame["rgba"],
            "depth": frame["depth"],
            "receipt": frame["receipt"],
        },
        manifest_path=output / "structural_background_render_index.json",
        expected_kind="video2world.structural_background_render_receipt",
        expected_role="structural_background",
        layer_id=None,
        require_accepted=True,
    )
    assert validated["acceptance"]["promotion_approved"] is False


def test_rejects_geometry_depth_that_does_not_cover_removal(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path / "source", cover_removal=False)

    with pytest.raises(RuntimeError, match="does not cover cumulative removal mask"):
        run_materialize(fixture, tmp_path / "output")


def test_rejects_limited_report_claiming_general_eligibility(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path / "source")
    report_path = Path(fixture["accepted_report"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["eligible_as_round04_clean_plate"] = True
    write_json(report_path, report)
    fixture["accepted_report_sha256"] = sha256_file(report_path)

    with pytest.raises(RuntimeError, match="cannot claim general round04 eligibility"):
        run_materialize(fixture, tmp_path / "output")


def test_rejects_declared_accepted_report_hash_mismatch(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path / "source")
    fixture["accepted_report_sha256"] = "0" * 64

    with pytest.raises(RuntimeError, match="accepted planar texture report sha256 mismatch"):
        run_materialize(fixture, tmp_path / "output")
