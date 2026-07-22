from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.accept_planar_texture_candidate import (
    PlanarTextureAcceptanceError,
    sign_candidate,
)
from scripts.review_planar_texture_candidate import (
    OUTPUT_RECEIPT_NAME,
    OUTPUT_REVIEW_NAME,
    PlanarTextureReviewError,
    review_candidate,
    sha256_file,
)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_rgb_png(
    path: Path,
    value: int,
    *,
    flat_mask_region: bool = False,
    striped_mask_region: bool = False,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    row, column = np.indices((48, 64))
    texture = ((row % 7) * 3 + (column % 5) * 4).astype(np.int16)
    image = np.stack(
        [
            value + texture,
            value + texture // 2,
            value + texture // 3,
        ],
        axis=2,
    )
    if flat_mask_region:
        image[12:36, 16:48] = [value, value, value]
    if striped_mask_region:
        stripe = value + ((column[12:36, 16:48] % 4) * 28)
        image[12:36, 16:48] = np.stack(
            [
                stripe,
                stripe // 2,
                stripe // 3,
            ],
            axis=2,
        )
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).save(path)
    return sha256_file(path)


def write_mask_png(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((48, 64), dtype=np.uint8)
    mask[12:36, 16:48] = 255
    Image.fromarray(mask).save(path)
    return sha256_file(path)


def make_candidate(
    tmp_path: Path,
    *,
    flat_mask_region: bool = False,
    striped_mask_region: bool = False,
) -> Path:
    root = tmp_path / "candidate"
    geometry_path = root / "planar_background_report.json"
    geometry = {
        "schema_version": 1,
        "status": "geometry_completed_texture_pending",
        "gates": {
            "minimum_geometry_coverage": {"passed": True},
            "outside_removal_mask_rgb_exact": True,
            "texture_masks_partition_removal_mask": True,
            "new_depth_normal_estimation_before_pgsr": "required_after_final_rgb_acceptance",
        },
    }
    write_json(geometry_path, geometry)

    frame_ids = ("000064", "000067", "000072")
    frame_records = []
    full_resolution_review = []
    for index, frame_id in enumerate(frame_ids):
        completed_path = root / "frames" / f"{frame_id}.png"
        mask_path = root / "synthetic_masks" / f"{frame_id}.png"
        completed_sha = write_rgb_png(
            completed_path,
            80 + index,
            flat_mask_region=flat_mask_region,
            striped_mask_region=striped_mask_region,
        )
        mask_sha = write_mask_png(mask_path)
        color_p95 = 118.0 if index == 0 else 30.0 + index
        gradient_p95 = 119.0 if index == 0 else 40.0 + index
        frame_records.append(
            {
                "sequence_index": index,
                "frame_id": frame_id,
                "completed_frame": str(completed_path.relative_to(root)),
                "completed_frame_sha256": completed_sha,
                "synthetic_mask": str(mask_path.relative_to(root)),
                "synthetic_mask_sha256": mask_sha,
                "synthetic_pixels": 10 + index,
                "interpolated_pixels": 10 + index,
                "synthetic_boundary_continuity_full_resolution": {
                    "boundary_color_p95_abs_rgb_delta": color_p95,
                    "boundary_normal_gradient_p95_abs_rgb_delta": gradient_p95,
                },
            }
        )
        full_resolution_review.append(
            {
                "frame_id": frame_id,
                "completed_frame": str(completed_path.relative_to(root)),
                "completed_frame_sha256": completed_sha,
                "synthetic_mask": str(mask_path.relative_to(root)),
                "synthetic_mask_sha256": mask_sha,
            }
        )

    candidate = {
        "schema_version": 1,
        "status": "texture_candidate_review_pending",
        "promotion_approved": False,
        "eligible_as_round04_clean_plate": False,
        "promotion_blocker": "visual quality review is pending",
        "planar_geometry_report": geometry_path.name,
        "planar_geometry_report_sha256": sha256_file(geometry_path),
        "limitations": ["Hidden texture is synthetic."],
        "gates": {
            "measured_atlas_texels_rgb_exact": True,
            "synthetic_anchor_atlas_texels_rgb_exact": True,
            "outside_synthetic_mask_rgb_exact": True,
            "protected_neighbors_outside_synthetic_mask_rgb_exact": True,
            "all_synthetic_pixels_assigned_from_shared_atlas": True,
            "same_atlas_texel_has_identical_rgb_across_views": True,
            "synthetic_anchor_claims_measured_donor": False,
            "rejected_texture_inputs_not_promotable": True,
            "wall_boundary_color_continuity": {
                "threshold_p95_abs_rgb_delta": 24.0,
                "observed_maximum": 0.0,
                "passed": True,
            },
            "wall_boundary_low_frequency_gradient_continuity": {
                "threshold_p95_abs_rgb_delta": 32.0,
                "observed_maximum": 0.0,
                "passed": True,
            },
            "visual_quality": "pending_human_or_vlm_review",
            "new_depth_normal_estimation_before_pgsr": "required_after_visual_acceptance",
        },
        "frame_records": frame_records,
        "full_resolution_review": full_resolution_review,
        "full_resolution_boundary_continuity_summary": {
            "frame_ids": list(frame_ids),
            "maximum_boundary_color_p95_abs_rgb_delta": 118.0,
            "maximum_boundary_normal_gradient_p95_abs_rgb_delta": 119.0,
        },
    }
    candidate_path = root / "planar_texture_report.json"
    write_json(candidate_path, candidate)
    return candidate_path


def test_rejects_high_full_resolution_boundary_candidate(tmp_path: Path) -> None:
    candidate = make_candidate(tmp_path)
    candidate_sha = sha256_file(candidate)
    output = tmp_path / "review"

    review, receipt, _ = review_candidate(
        candidate_report_path=candidate,
        output_dir=output,
        reviewer_id="auto-reviewer",
        reviewed_at=datetime(2026, 7, 22, 12, 0, tzinfo=UTC),
    )

    assert sha256_file(candidate) == candidate_sha
    assert review["kind"] == "video2world.planar_texture_visual_review"
    assert review["decision"] == "rejected"
    assert receipt["status"] == "texture_candidate_rejected_or_needs_repair"
    assert receipt["promotion_approved"] is False
    assert receipt["eligible_as_round04_clean_plate"] is False
    assert {
        finding["gate"] for finding in review["blocking_findings"]
    } == {
        "full_resolution_boundary_color_continuity",
        "full_resolution_boundary_low_frequency_gradient_continuity",
    }
    assert review["gates"]["full_resolution_boundary_color_continuity"] == {
        "threshold_p95_abs_rgb_delta": 48.0,
        "observed_maximum": 118.0,
        "passed": False,
    }
    assert review["full_resolution_review"][0]["completed_frame_sha256"]
    assert (output / OUTPUT_REVIEW_NAME).is_file()
    assert read_json(output / OUTPUT_RECEIPT_NAME)["visual_review"]["sha256"] == sha256_file(
        output / OUTPUT_REVIEW_NAME
    )


def test_rejected_review_cannot_promote_candidate(tmp_path: Path) -> None:
    candidate = make_candidate(tmp_path)
    output = tmp_path / "review"
    review_candidate(
        candidate_report_path=candidate,
        output_dir=output,
        reviewer_id="auto-reviewer",
        reviewed_at=datetime(2026, 7, 22, 12, 0, tzinfo=UTC),
    )
    review_path = output / OUTPUT_REVIEW_NAME

    with pytest.raises(PlanarTextureAcceptanceError, match="decision is not accepted"):
        sign_candidate(
            candidate_report_path=candidate,
            candidate_report_sha256=sha256_file(candidate),
            visual_review_path=review_path,
            visual_review_sha256=sha256_file(review_path),
            output_dir=tmp_path / "accepted",
            accepted_at=datetime(2026, 7, 22, 12, 5, tzinfo=UTC),
        )


def test_metric_pass_candidate_still_requires_visual_review(tmp_path: Path) -> None:
    candidate_path = make_candidate(tmp_path)
    candidate = read_json(candidate_path)
    for record in candidate["frame_records"]:
        record["synthetic_boundary_continuity_full_resolution"][
            "boundary_color_p95_abs_rgb_delta"
        ] = 30.0
        record["synthetic_boundary_continuity_full_resolution"][
            "boundary_normal_gradient_p95_abs_rgb_delta"
        ] = 31.0
    write_json(candidate_path, candidate)

    review, receipt, _ = review_candidate(
        candidate_report_path=candidate_path,
        output_dir=tmp_path / "review",
        reviewer_id="auto-reviewer",
        reviewed_at=datetime(2026, 7, 22, 12, 0, tzinfo=UTC),
    )

    assert receipt["status"] == "texture_candidate_metric_pass_visual_review_required"
    assert review["decision"] == "rejected"
    assert review["blocking_findings"] == [
        {
            "gate": "visual_quality",
            "observed": "pending_human_or_vlm_review",
            "severity": "blocking",
        }
    ]
    assert review["gates"]["full_resolution_boundary_color_continuity"]["passed"] is True
    assert review["gates"]["synthetic_region_texture_energy_ratio"]["passed"] is True
    assert "run_bound_human_or_vlm_visual_review" in review["next_action"]


def test_metric_review_rejects_flat_synthetic_texture_patch(tmp_path: Path) -> None:
    candidate_path = make_candidate(tmp_path, flat_mask_region=True)
    candidate = read_json(candidate_path)
    for record in candidate["frame_records"]:
        record["synthetic_boundary_continuity_full_resolution"][
            "boundary_color_p95_abs_rgb_delta"
        ] = 30.0
        record["synthetic_boundary_continuity_full_resolution"][
            "boundary_normal_gradient_p95_abs_rgb_delta"
        ] = 31.0
    write_json(candidate_path, candidate)

    review, receipt, _ = review_candidate(
        candidate_report_path=candidate_path,
        output_dir=tmp_path / "review",
        reviewer_id="auto-reviewer",
        reviewed_at=datetime(2026, 7, 22, 12, 0, tzinfo=UTC),
    )

    assert receipt["status"] == "texture_candidate_rejected_or_needs_repair"
    assert review["gates"]["full_resolution_boundary_color_continuity"]["passed"] is True
    assert review["gates"]["synthetic_region_texture_energy_ratio"]["passed"] is False
    assert review["gates"]["synthetic_texture_orientation_anisotropy"]["passed"] is True
    assert review["blocking_findings"] == [
        {
            "gate": "synthetic_region_texture_energy_ratio",
            "threshold_minimum_ratio": 0.45,
            "observed_minimum": pytest.approx(0.0),
            "severity": "blocking",
        }
    ]
    assert "too flat" in review["next_action"]


def test_metric_review_rejects_directional_stripe_artifact(tmp_path: Path) -> None:
    candidate_path = make_candidate(tmp_path, striped_mask_region=True)
    candidate = read_json(candidate_path)
    for record in candidate["frame_records"]:
        record["synthetic_boundary_continuity_full_resolution"][
            "boundary_color_p95_abs_rgb_delta"
        ] = 30.0
        record["synthetic_boundary_continuity_full_resolution"][
            "boundary_normal_gradient_p95_abs_rgb_delta"
        ] = 31.0
    write_json(candidate_path, candidate)

    review, receipt, _ = review_candidate(
        candidate_report_path=candidate_path,
        output_dir=tmp_path / "review",
        reviewer_id="auto-reviewer",
        reviewed_at=datetime(2026, 7, 22, 12, 0, tzinfo=UTC),
    )

    assert receipt["status"] == "texture_candidate_rejected_or_needs_repair"
    assert review["gates"]["full_resolution_boundary_color_continuity"]["passed"] is True
    assert review["gates"]["synthetic_region_texture_energy_ratio"]["passed"] is True
    assert review["gates"]["synthetic_texture_orientation_anisotropy"]["passed"] is False
    observed_maximum = review["gates"]["synthetic_texture_orientation_anisotropy"][
        "observed_maximum"
    ]
    assert observed_maximum > 2.0
    assert review["blocking_findings"] == [
        {
            "gate": "synthetic_texture_orientation_anisotropy",
            "threshold_maximum": 2.0,
            "observed_maximum": pytest.approx(observed_maximum),
            "severity": "blocking",
        }
    ]
    assert "one-axis stripe energy" in review["next_action"]


def test_refuses_already_promoted_candidate(tmp_path: Path) -> None:
    candidate_path = make_candidate(tmp_path)
    candidate = read_json(candidate_path)
    candidate["promotion_approved"] = True
    write_json(candidate_path, candidate)

    with pytest.raises(PlanarTextureReviewError, match="already promoted"):
        review_candidate(
            candidate_report_path=candidate_path,
            output_dir=tmp_path / "review",
            reviewer_id="auto-reviewer",
            reviewed_at=datetime(2026, 7, 22, 12, 0, tzinfo=UTC),
        )
