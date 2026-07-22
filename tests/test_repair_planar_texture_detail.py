from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from scripts.repair_planar_texture_detail import (
    OUTPUT_REPORT_NAME,
    build_detail_repair_candidate,
    repair_texture_detail,
    sha256_file,
)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def test_repair_texture_detail_preserves_outside_mask() -> None:
    row, column = np.indices((28, 36))
    image = np.stack(
        [
            90 + (row % 5) * 8,
            70 + (column % 7) * 5,
            55 + ((row + column) % 6) * 4,
        ],
        axis=2,
    ).astype(np.uint8)
    image[8:20, 10:28] = [110, 80, 60]
    mask = np.zeros((28, 36), dtype=bool)
    mask[8:20, 10:28] = True

    repaired, details = repair_texture_detail(
        image,
        mask,
        collar_width_pixels=8,
        inner_ramp_pixels=2,
        residual_sigma_pixels=2.0,
        residual_strength=1.2,
    )

    assert details["outside_synthetic_mask_rgb_exact"] is True
    assert np.array_equal(repaired[~mask], image[~mask])
    assert not np.array_equal(repaired[mask], image[mask])
    assert details["claims_measured_donor"] is False


def test_build_detail_repair_candidate_rewrites_bound_frames(tmp_path: Path) -> None:
    root = tmp_path / "candidate"
    frames_dir = root / "frames"
    masks_dir = root / "synthetic_masks"
    frames_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)
    frame_records = []
    full_resolution_review = []
    for index, frame_id in enumerate(("000064", "000067", "000072")):
        image = np.full((32, 40, 3), [100 + index, 80, 60], dtype=np.uint8)
        image[:, :8] = [130, 120, 95]
        image[:, 32:] = [70, 50, 35]
        mask = np.zeros((32, 40), dtype=np.uint8)
        mask[8:24, 10:30] = 255
        image[mask > 0] = [105, 75, 55]
        frame_path = frames_dir / f"{frame_id}.png"
        mask_path = masks_dir / f"{frame_id}.png"
        Image.fromarray(image).save(frame_path)
        Image.fromarray(mask).save(mask_path)
        frame_records.append(
            {
                "frame_id": frame_id,
                "source_frame": str(frame_path),
                "completed_frame": str(frame_path),
                "completed_frame_sha256": sha256_file(frame_path),
                "synthetic_mask": str(mask_path),
                "synthetic_mask_sha256": sha256_file(mask_path),
                "synthetic_boundary_continuity_full_resolution": {
                    "boundary_color_p95_abs_rgb_delta": 20.0,
                    "boundary_normal_gradient_p95_abs_rgb_delta": 21.0,
                },
            }
        )
        full_resolution_review.append(
            {
                "frame_id": frame_id,
                "completed_frame": str(frame_path),
                "completed_frame_sha256": sha256_file(frame_path),
                "synthetic_mask": str(mask_path),
                "synthetic_mask_sha256": sha256_file(mask_path),
            }
        )
    candidate = {
        "schema_version": 1,
        "status": "texture_candidate_review_pending",
        "promotion_approved": False,
        "eligible_as_round04_clean_plate": False,
        "gates": {
            "outside_synthetic_mask_rgb_exact": True,
            "visual_quality": "pending_human_or_vlm_review",
        },
        "frame_records": frame_records,
        "full_resolution_review": full_resolution_review,
    }
    candidate_path = root / "planar_texture_report.json"
    write_json(candidate_path, candidate)

    output = tmp_path / "detail-repair"
    build_detail_repair_candidate(
        SimpleNamespace(
            candidate_report=candidate_path,
            output=output,
            texture_collar_width_pixels=8,
            texture_inner_ramp_pixels=2,
            texture_residual_sigma_pixels=2.0,
            texture_residual_strength=1.2,
            review_contact_sheet_samples=3,
        )
    )

    report = json.loads((output / OUTPUT_REPORT_NAME).read_text(encoding="utf-8"))
    assert report["status"] == "texture_candidate_review_pending"
    assert report["promotion_approved"] is False
    assert report["eligible_as_round04_clean_plate"] is False
    assert report["gates"]["texture_detail_repair_outside_synthetic_mask_rgb_exact"] is True
    assert report["gates"]["synthetic_texture_detail_claims_measured_donor"] is False
    assert report["full_resolution_review"][0]["completed_frame_sha256"] != (
        full_resolution_review[0]["completed_frame_sha256"]
    )
