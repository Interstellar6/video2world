from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.accept_planar_texture_candidate import (
    OUTPUT_RECEIPT_HASH_NAME,
    OUTPUT_RECEIPT_NAME,
    OUTPUT_REPORT_NAME,
    PlanarTextureAcceptanceError,
    sha256_file,
    sign_candidate,
)
from scripts.materialize_clean_scene_reconstruction_input import (
    validate_acceptance,
    validate_inputs,
)


@dataclass(frozen=True)
class AcceptanceFixture:
    candidate: Path
    review: Path
    geometry: Path
    camera_info: Path


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def acceptance_fixture(tmp_path: Path) -> AcceptanceFixture:
    root = tmp_path / "candidate"
    source_dir = root / "source"
    completed_dir = root / "frames"
    masks_dir = root / "synthetic_masks"
    depth_dir = root / "geometry_depth"
    frame_ids = ("000064", "000067", "000072")
    frame_records: list[dict] = []
    geometry_records: list[dict] = []
    full_resolution_review: list[dict] = []

    for sequence_index, frame_id in enumerate(frame_ids):
        source = np.full((3, 4, 3), 35 + sequence_index * 10, dtype=np.uint8)
        mask = np.zeros((3, 4), dtype=np.uint8)
        mask[1, sequence_index + 1] = 255
        completed = source.copy()
        completed[mask > 0] = [150 + sequence_index, 100, 60]
        source_path = source_dir / f"{frame_id}.png"
        completed_path = completed_dir / f"{frame_id}.png"
        mask_path = masks_dir / f"{frame_id}.png"
        depth_path = depth_dir / f"{frame_id}.npz"
        for path in (source_path, completed_path, mask_path, depth_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(source).save(source_path)
        Image.fromarray(completed).save(completed_path)
        Image.fromarray(mask).save(mask_path)
        np.savez(depth_path, depth=np.full((3, 4), 2.0, dtype=np.float32))
        completed_sha = sha256_file(completed_path)
        mask_sha = sha256_file(mask_path)
        frame_records.append(
            {
                "sequence_index": sequence_index,
                "frame_id": frame_id,
                "source_frame": str(source_path.relative_to(root)),
                "source_frame_sha256": sha256_file(source_path),
                "completed_frame": str(completed_path.relative_to(root)),
                "completed_frame_sha256": completed_sha,
                "synthetic_mask": str(mask_path.relative_to(root)),
                "synthetic_mask_sha256": mask_sha,
                "synthetic_pixels": 1,
                "synthetic_pixels_assigned_from_shared_atlas": 1,
                "all_synthetic_pixels_assigned": True,
                "outside_synthetic_mask_rgb_exact": True,
            }
        )
        geometry_records.append(
            {
                "sequence_index": sequence_index,
                "frame_id": frame_id,
                "geometry_depth": str(depth_path.relative_to(root)),
                "geometry_depth_sha256": sha256_file(depth_path),
                "outside_removal_mask_rgb_exact": True,
                "texture_partition_exact": True,
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

    intrinsic = {
        "camera_id": "1",
        "model": "PINHOLE",
        "w": 4,
        "h": 3,
        "fx": 5.0,
        "fy": 5.5,
        "cx": 2.0,
        "cy": 1.5,
        "params": [5.0, 5.5, 2.0, 1.5],
    }
    camera_info = {
        "schema_version": 1,
        "extrinsic_type": "world_to_camera",
        "intrinsic": intrinsic,
        "intrinsics": {"1": intrinsic},
        "extrinsic": {frame_id: np.eye(4).tolist() for frame_id in frame_ids},
        "frame_camera_ids": {frame_id: "1" for frame_id in frame_ids},
        "images": {
            frame_id: {
                "image_id": str(index + 1),
                "camera_id": "1",
                "name": f"{frame_id}.png",
            }
            for index, frame_id in enumerate(frame_ids)
        },
    }
    camera_path = root / "camera_info.json"
    write_json(camera_path, camera_info)
    geometry = {
        "schema_version": 1,
        "status": "geometry_completed_texture_pending",
        "camera_extrinsic_type": "world_to_camera",
        "camera_info": str(camera_path),
        "camera_info_sha256": sha256_file(camera_path),
        "gates": {
            "minimum_geometry_coverage": {"passed": True},
            "outside_removal_mask_rgb_exact": True,
            "texture_masks_partition_removal_mask": True,
            "generated_texture_present": False,
            "cross_view_rgb_continuity": "pending",
            "new_depth_normal_estimation_before_pgsr": ("required_after_final_rgb_acceptance"),
        },
        "frame_records": geometry_records,
    }
    geometry_path = root / "planar_background_report.json"
    write_json(geometry_path, geometry)
    candidate = {
        "schema_version": 1,
        "status": "texture_candidate_review_pending",
        "promotion_approved": False,
        "eligible_as_round04_clean_plate": False,
        "promotion_blocker": "visual quality review is pending",
        "planar_geometry_report": geometry_path.name,
        "planar_geometry_report_sha256": sha256_file(geometry_path),
        "gates": {
            "measured_atlas_texels_rgb_exact": True,
            "synthetic_anchor_atlas_texels_rgb_exact": True,
            "outside_synthetic_mask_rgb_exact": True,
            "protected_neighbors_outside_synthetic_mask_rgb_exact": True,
            "all_synthetic_pixels_assigned_from_shared_atlas": True,
            "same_atlas_texel_has_identical_rgb_across_views": True,
            "synthetic_anchor_claims_measured_donor": False,
            "rejected_texture_inputs_not_promotable": True,
            "wall_boundary_color_continuity": {"passed": True},
            "wall_boundary_low_frequency_gradient_continuity": {"passed": True},
            "visual_quality": "pending_human_or_vlm_review",
            "new_depth_normal_estimation_before_pgsr": ("required_after_visual_acceptance"),
        },
        "frame_records": frame_records,
        "full_resolution_review": full_resolution_review,
        "full_resolution_boundary_continuity_summary": {
            "frame_ids": list(frame_ids),
        },
        "limitations": ["Hidden texture is synthetic and must remain provenance-labeled."],
    }
    candidate_path = root / "planar_texture_report.json"
    write_json(candidate_path, candidate)
    candidate_sha = sha256_file(candidate_path)
    review = {
        "schema_version": 1,
        "kind": "video2world.planar_texture_visual_review",
        "status": "completed",
        "decision": "accepted",
        "candidate_report_sha256": candidate_sha,
        "review_authority": {
            "kind": "human",
            "reviewer_id": "reviewer@example.test",
            "scope": "round04_clean_plate_acceptance",
            "reviewed_at": "2026-07-17T12:00:00+08:00",
        },
        "gates": {
            "wall_visual_continuity": True,
            "floor_visual_continuity": True,
            "wall_floor_boundary_plausible": True,
        },
        "blocking_findings": [],
        "limitations": ["Acceptance does not certify fresh depth or normals."],
        "full_resolution_review": [
            {
                "frame_id": item["frame_id"],
                "completed_frame_sha256": item["completed_frame_sha256"],
                "synthetic_mask_sha256": item["synthetic_mask_sha256"],
                "decision": "accepted",
                "blocking_findings": [],
            }
            for item in full_resolution_review
        ],
    }
    review_path = tmp_path / "visual_review.json"
    write_json(review_path, review)
    return AcceptanceFixture(
        candidate=candidate_path,
        review=review_path,
        geometry=geometry_path,
        camera_info=camera_path,
    )


def rewrite_candidate_and_rebind_review(
    fixture: AcceptanceFixture, candidate: dict
) -> tuple[str, str]:
    write_json(fixture.candidate, candidate)
    candidate_sha = sha256_file(fixture.candidate)
    review = read_json(fixture.review)
    review["candidate_report_sha256"] = candidate_sha
    write_json(fixture.review, review)
    return candidate_sha, sha256_file(fixture.review)


def invoke(
    fixture: AcceptanceFixture,
    output: Path,
    *,
    candidate_sha: str | None = None,
    review_sha: str | None = None,
) -> tuple[dict, dict, str]:
    return sign_candidate(
        candidate_report_path=fixture.candidate,
        candidate_report_sha256=candidate_sha or sha256_file(fixture.candidate),
        visual_review_path=fixture.review,
        visual_review_sha256=review_sha or sha256_file(fixture.review),
        output_dir=output,
        accepted_at=datetime(2026, 7, 17, 13, 0, tzinfo=UTC),
    )


def configure_limited_review(
    fixture: AcceptanceFixture,
    *,
    failed_boundary_gate: str = "wall_boundary_color_continuity",
    real_metric_keys: bool = False,
) -> dict:
    candidate = read_json(fixture.candidate)
    if real_metric_keys:
        failed_gate = {
            "threshold_p95_abs_rgb_delta": 8.0,
            "observed_maximum": 12.5,
            "passed": False,
        }
    else:
        failed_gate = {
            "threshold": 8.0,
            "observed": 12.5,
            "passed": False,
        }
    candidate["gates"][failed_boundary_gate] = failed_gate
    write_json(fixture.candidate, candidate)

    review = read_json(fixture.review)
    review["candidate_report_sha256"] = sha256_file(fixture.candidate)
    review["decision"] = "accepted_with_limitations"
    review["acceptance_scope"] = "current_demo_only"
    review["override_gate_keys"] = ["visual_quality", failed_boundary_gate]
    review["review_authority"].update(
        {
            "kind": "user",
            "scope": "current_demo_only",
        }
    )
    for item in review["full_resolution_review"]:
        item["decision"] = "accepted_with_limitations"
    write_json(fixture.review, review)
    return failed_gate


def test_signs_new_report_without_mutating_candidate_and_matches_materializer(
    acceptance_fixture: AcceptanceFixture, tmp_path: Path
) -> None:
    candidate_sha = sha256_file(acceptance_fixture.candidate)
    review_sha = sha256_file(acceptance_fixture.review)
    output = tmp_path / "accepted"

    accepted, receipt, receipt_sha = invoke(acceptance_fixture, output)

    assert sha256_file(acceptance_fixture.candidate) == candidate_sha
    assert accepted["status"] == "accepted_for_round04_clean_plate"
    assert accepted["promotion_approved"] is True
    assert accepted["eligible_as_round04_clean_plate"] is True
    assert accepted["promotion_blocker"] is None
    assert accepted["gates"]["visual_quality"] == "accepted"
    for field in (
        "acceptance_scope",
        "accepted_with_limitations",
        "demo_use_approved",
        "eligible_as_current_demo_round04_clean_plate",
        "overridden_gates",
    ):
        assert field not in accepted
    assert accepted["acceptance"]["candidate_report"]["sha256"] == candidate_sha
    assert accepted["acceptance"]["visual_review"]["sha256"] == review_sha
    assert accepted["acceptance"]["fresh_reconstruction_requirement"] == {
        "fresh_depth_required": True,
        "fresh_normals_required": True,
        "legacy_depth_allowed_as_new_depth": False,
        "pgsr_allowed_before_fresh_depth_and_normals": False,
    }
    assert Path(accepted["planar_geometry_report"]).is_absolute()
    assert all(
        Path(record["completed_frame"]).is_absolute() for record in accepted["frame_records"]
    )
    report_path = output / OUTPUT_REPORT_NAME
    receipt_path = output / OUTPUT_RECEIPT_NAME
    assert read_json(report_path) == accepted
    assert read_json(receipt_path) == receipt
    assert receipt["accepted_report"]["sha256"] == sha256_file(report_path)
    assert receipt["promotion_approved"] is True
    assert receipt["eligible_as_round04_clean_plate"] is True
    assert "accepted_with_limitations" not in receipt
    assert receipt_sha == sha256_file(receipt_path)
    assert (output / OUTPUT_RECEIPT_HASH_NAME).read_text(encoding="ascii") == (
        f"{receipt_sha}  {OUTPUT_RECEIPT_NAME}\n"
    )

    validate_acceptance(accepted)
    validated = validate_inputs(report_path, acceptance_fixture.camera_info)
    assert len(validated.frames) == 3


def test_signs_user_limited_current_demo_acceptance_without_rewriting_failed_gate(
    acceptance_fixture: AcceptanceFixture, tmp_path: Path
) -> None:
    failed_gate = configure_limited_review(acceptance_fixture)
    candidate_sha = sha256_file(acceptance_fixture.candidate)
    output = tmp_path / "accepted-limited"

    accepted, receipt, _ = invoke(acceptance_fixture, output)

    assert sha256_file(acceptance_fixture.candidate) == candidate_sha
    assert accepted["status"] == "accepted_for_round04_clean_plate"
    assert accepted["promotion_approved"] is False
    assert accepted["eligible_as_round04_clean_plate"] is False
    assert accepted["demo_use_approved"] is True
    assert accepted["eligible_as_current_demo_round04_clean_plate"] is True
    assert accepted["acceptance_scope"] == "current_demo_only"
    assert accepted["accepted_with_limitations"] is True
    assert accepted["gates"]["visual_quality"] == "accepted_with_limitations"
    assert accepted["gates"]["wall_boundary_color_continuity"] == failed_gate
    assert accepted["overridden_gates"] == {
        "visual_quality": "pending_human_or_vlm_review",
        "wall_boundary_color_continuity": failed_gate,
    }
    assert accepted["overridden_gates"]["wall_boundary_color_continuity"]["passed"] is False
    assert accepted["acceptance"]["acceptance_scope"] == "current_demo_only"
    assert accepted["acceptance"]["review_authority"]["kind"] == "user"
    assert receipt["acceptance_scope"] == "current_demo_only"
    assert receipt["accepted_with_limitations"] is True
    assert receipt["promotion_approved"] is False
    assert receipt["eligible_as_round04_clean_plate"] is False
    assert receipt["demo_use_approved"] is True
    assert receipt["eligible_as_current_demo_round04_clean_plate"] is True
    assert receipt["overridden_gates"] == accepted["overridden_gates"]

    report_path = output / OUTPUT_REPORT_NAME
    validate_acceptance(accepted)
    validated = validate_inputs(report_path, acceptance_fixture.camera_info)
    assert len(validated.frames) == 3


def test_signs_real_planar_atlas_boundary_metric_schema(
    acceptance_fixture: AcceptanceFixture, tmp_path: Path
) -> None:
    failed_gate = configure_limited_review(acceptance_fixture, real_metric_keys=True)

    accepted, receipt, _ = invoke(acceptance_fixture, tmp_path / "accepted-real-metric-schema")

    assert failed_gate == {
        "threshold_p95_abs_rgb_delta": 8.0,
        "observed_maximum": 12.5,
        "passed": False,
    }
    assert accepted["gates"]["wall_boundary_color_continuity"] == failed_gate
    assert accepted["overridden_gates"]["wall_boundary_color_continuity"] == failed_gate
    assert receipt["overridden_gates"]["wall_boundary_color_continuity"] == failed_gate


@pytest.mark.parametrize(
    ("threshold", "observed", "expected"),
    [
        (8.0, "not-a-number", "finite numbers"),
        (8.0, 8.0, "must exceed threshold"),
        (8.0, 7.9, "must exceed threshold"),
    ],
)
def test_rejects_invalid_real_boundary_failure_metrics(
    acceptance_fixture: AcceptanceFixture,
    tmp_path: Path,
    threshold: float,
    observed: object,
    expected: str,
) -> None:
    configure_limited_review(acceptance_fixture, real_metric_keys=True)
    candidate = read_json(acceptance_fixture.candidate)
    failed_gate = candidate["gates"]["wall_boundary_color_continuity"]
    failed_gate["threshold_p95_abs_rgb_delta"] = threshold
    failed_gate["observed_maximum"] = observed
    rewrite_candidate_and_rebind_review(acceptance_fixture, candidate)
    output = tmp_path / "invalid-real-boundary-metrics"

    with pytest.raises(PlanarTextureAcceptanceError, match=expected):
        invoke(acceptance_fixture, output)

    assert not output.exists()


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("human_authority", "authority must be user"),
        ("wrong_scope", "acceptance_scope must be current_demo_only"),
        ("unknown_override", "forbidden override gate keys"),
        ("missing_failed_boundary", "must exactly match"),
    ],
)
def test_rejects_invalid_limited_acceptance_override_contract(
    acceptance_fixture: AcceptanceFixture,
    tmp_path: Path,
    mutation: str,
    expected: str,
) -> None:
    configure_limited_review(acceptance_fixture)
    review = read_json(acceptance_fixture.review)
    if mutation == "human_authority":
        review["review_authority"]["kind"] = "human"
    elif mutation == "wrong_scope":
        review["acceptance_scope"] = "all_outputs"
    elif mutation == "unknown_override":
        review["override_gate_keys"].append("measured_atlas_texels_rgb_exact")
    else:
        review["override_gate_keys"] = ["visual_quality"]
    write_json(acceptance_fixture.review, review)
    output = tmp_path / f"invalid-limited-{mutation}"

    with pytest.raises(PlanarTextureAcceptanceError, match=expected):
        invoke(acceptance_fixture, output)

    assert not output.exists()


def test_limited_acceptance_cannot_override_a_failed_technical_gate(
    acceptance_fixture: AcceptanceFixture, tmp_path: Path
) -> None:
    configure_limited_review(acceptance_fixture)
    candidate = read_json(acceptance_fixture.candidate)
    candidate["gates"]["measured_atlas_texels_rgb_exact"] = False
    rewrite_candidate_and_rebind_review(acceptance_fixture, candidate)
    output = tmp_path / "limited-failed-technical-gate"

    with pytest.raises(PlanarTextureAcceptanceError, match="candidate gate did not pass"):
        invoke(acceptance_fixture, output)

    assert not output.exists()


def test_rejects_existing_output_even_when_empty(
    acceptance_fixture: AcceptanceFixture, tmp_path: Path
) -> None:
    output = tmp_path / "already-exists"
    output.mkdir()

    with pytest.raises(PlanarTextureAcceptanceError, match="output already exists"):
        invoke(acceptance_fixture, output)

    assert list(output.iterdir()) == []


def test_rejects_broken_output_symlink(
    acceptance_fixture: AcceptanceFixture, tmp_path: Path
) -> None:
    output = tmp_path / "output-link"
    output.symlink_to(tmp_path / "absent-target", target_is_directory=True)

    with pytest.raises(PlanarTextureAcceptanceError, match="output cannot be a symlink"):
        invoke(acceptance_fixture, output)

    assert output.is_symlink()


@pytest.mark.parametrize("role", ["candidate", "review"])
def test_rejects_declared_hash_mismatch_without_output(
    acceptance_fixture: AcceptanceFixture, tmp_path: Path, role: str
) -> None:
    output = tmp_path / f"hash-mismatch-{role}"
    kwargs = {f"{role}_sha": "0" * 64}

    with pytest.raises(PlanarTextureAcceptanceError, match="SHA-256 mismatch"):
        invoke(acceptance_fixture, output, **kwargs)

    assert not output.exists()


def test_rejects_missing_reviewed_frame(
    acceptance_fixture: AcceptanceFixture, tmp_path: Path
) -> None:
    review = read_json(acceptance_fixture.review)
    review["full_resolution_review"].pop()
    write_json(acceptance_fixture.review, review)
    output = tmp_path / "missing-review-frame"

    with pytest.raises(PlanarTextureAcceptanceError, match="do not exactly match"):
        invoke(acceptance_fixture, output)

    assert not output.exists()


@pytest.mark.parametrize(
    ("gate_path", "expected"),
    [
        (("measured_atlas_texels_rgb_exact",), "candidate gate did not pass"),
        (("wall_boundary_color_continuity", "passed"), "candidate boundary gate did not pass"),
    ],
)
def test_rejects_failing_technical_or_boundary_gate(
    acceptance_fixture: AcceptanceFixture,
    tmp_path: Path,
    gate_path: tuple[str, ...],
    expected: str,
) -> None:
    candidate = read_json(acceptance_fixture.candidate)
    target = candidate["gates"]
    for key in gate_path[:-1]:
        target = target[key]
    target[gate_path[-1]] = False
    rewrite_candidate_and_rebind_review(acceptance_fixture, candidate)
    output = tmp_path / "failed-gate"

    with pytest.raises(PlanarTextureAcceptanceError, match=expected):
        invoke(acceptance_fixture, output)

    assert not output.exists()


def test_rejects_rejected_review_decision(
    acceptance_fixture: AcceptanceFixture, tmp_path: Path
) -> None:
    review = read_json(acceptance_fixture.review)
    review["decision"] = "rejected"
    write_json(acceptance_fixture.review, review)
    output = tmp_path / "rejected-review"

    with pytest.raises(PlanarTextureAcceptanceError, match="decision is not accepted"):
        invoke(acceptance_fixture, output)

    assert not output.exists()


def test_rejects_rejected_per_frame_review_decision(
    acceptance_fixture: AcceptanceFixture, tmp_path: Path
) -> None:
    review = read_json(acceptance_fixture.review)
    review["full_resolution_review"][1]["decision"] = "rejected"
    write_json(acceptance_fixture.review, review)
    output = tmp_path / "rejected-frame-review"

    with pytest.raises(PlanarTextureAcceptanceError, match="frame decision is not accepted"):
        invoke(acceptance_fixture, output)

    assert not output.exists()


@pytest.mark.parametrize("field", ["promotion_approved", "eligible_as_round04_clean_plate"])
def test_rejects_candidate_that_is_already_promoted_or_eligible(
    acceptance_fixture: AcceptanceFixture, tmp_path: Path, field: str
) -> None:
    candidate = read_json(acceptance_fixture.candidate)
    candidate[field] = True
    rewrite_candidate_and_rebind_review(acceptance_fixture, candidate)
    output = tmp_path / f"unsafe-{field}"

    with pytest.raises(PlanarTextureAcceptanceError, match="must be false"):
        invoke(acceptance_fixture, output)

    assert not output.exists()


def test_rejects_candidate_with_fewer_than_three_full_resolution_frames(
    acceptance_fixture: AcceptanceFixture, tmp_path: Path
) -> None:
    candidate = read_json(acceptance_fixture.candidate)
    candidate["full_resolution_review"] = candidate["full_resolution_review"][:2]
    candidate["full_resolution_boundary_continuity_summary"]["frame_ids"] = [
        item["frame_id"] for item in candidate["full_resolution_review"]
    ]
    rewrite_candidate_and_rebind_review(acceptance_fixture, candidate)
    output = tmp_path / "too-few-frames"

    with pytest.raises(PlanarTextureAcceptanceError, match="at least 3 frames"):
        invoke(acceptance_fixture, output)

    assert not output.exists()


def test_rejects_review_frame_hash_binding_mismatch(
    acceptance_fixture: AcceptanceFixture, tmp_path: Path
) -> None:
    review = read_json(acceptance_fixture.review)
    review["full_resolution_review"][0]["completed_frame_sha256"] = "f" * 64
    write_json(acceptance_fixture.review, review)
    output = tmp_path / "frame-hash-mismatch"

    with pytest.raises(PlanarTextureAcceptanceError, match="completed frame binding mismatch"):
        invoke(acceptance_fixture, output)

    assert not output.exists()


@pytest.mark.parametrize("mutation", ["authority", "review_limitations", "candidate_limitations"])
def test_rejects_missing_authority_or_limitations(
    acceptance_fixture: AcceptanceFixture, tmp_path: Path, mutation: str
) -> None:
    candidate = read_json(acceptance_fixture.candidate)
    review = read_json(acceptance_fixture.review)
    if mutation == "authority":
        review.pop("review_authority")
    elif mutation == "review_limitations":
        review["limitations"] = []
    else:
        candidate["limitations"] = []
        write_json(acceptance_fixture.candidate, candidate)
        review["candidate_report_sha256"] = sha256_file(acceptance_fixture.candidate)
    write_json(acceptance_fixture.review, review)
    output = tmp_path / f"missing-{mutation}"

    with pytest.raises(PlanarTextureAcceptanceError):
        invoke(acceptance_fixture, output)

    assert not output.exists()


def test_rejects_missing_fresh_depth_normal_requirement(
    acceptance_fixture: AcceptanceFixture, tmp_path: Path
) -> None:
    candidate = read_json(acceptance_fixture.candidate)
    candidate["gates"]["new_depth_normal_estimation_before_pgsr"] = "optional"
    rewrite_candidate_and_rebind_review(acceptance_fixture, candidate)
    output = tmp_path / "no-fresh-depth-normal"

    with pytest.raises(PlanarTextureAcceptanceError, match="fresh depth/normal"):
        invoke(acceptance_fixture, output)

    assert not output.exists()


@pytest.mark.parametrize("role", ["candidate", "review"])
def test_rejects_empty_json_input_without_output(
    acceptance_fixture: AcceptanceFixture, tmp_path: Path, role: str
) -> None:
    path = getattr(acceptance_fixture, role)
    path.write_text("", encoding="utf-8")
    output = tmp_path / f"empty-{role}"

    with pytest.raises(PlanarTextureAcceptanceError, match="empty"):
        invoke(acceptance_fixture, output)

    assert not output.exists()
