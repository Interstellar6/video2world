#!/usr/bin/env python3
"""Accept one immutable planar-texture candidate after a bound visual review.

This signer is intentionally fail-closed. It never edits the candidate report,
never accepts an already promoted candidate, and never infers visual approval
from technical metrics. A successful run writes a new report that matches the
entry contract of ``materialize_clean_scene_reconstruction_input.py``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ACCEPTED_STATUS = "accepted_for_round04_clean_plate"
CANDIDATE_STATUS = "texture_candidate_review_pending"
REVIEW_KIND = "video2world.planar_texture_visual_review"
REVIEW_STATUS = "completed"
REVIEW_DECISION = "accepted"
REVIEW_AUTHORITY_KIND = "human"
REVIEW_AUTHORITY_SCOPE = "round04_clean_plate_acceptance"
LIMITED_REVIEW_DECISION = "accepted_with_limitations"
LIMITED_ACCEPTANCE_SCOPE = "current_demo_only"
LIMITED_REVIEW_AUTHORITY_KIND = "user"
FRESH_DEPTH_NORMAL_REQUIREMENT = "required_after_visual_acceptance"
OUTPUT_REPORT_NAME = "accepted_planar_texture_report.json"
OUTPUT_RECEIPT_NAME = "accepted_planar_texture_receipt.json"
OUTPUT_RECEIPT_HASH_NAME = "accepted_planar_texture_receipt.sha256"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
FRAME_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# Keep these synchronized with materialize_clean_scene_reconstruction_input.py.
REQUIRED_MATERIALIZER_GATES = (
    "measured_atlas_texels_rgb_exact",
    "outside_synthetic_mask_rgb_exact",
    "protected_neighbors_outside_synthetic_mask_rgb_exact",
    "all_synthetic_pixels_assigned_from_shared_atlas",
    "same_atlas_texel_has_identical_rgb_across_views",
    "rejected_texture_inputs_not_promotable",
)
REQUIRED_BOUNDARY_GATES = (
    "wall_boundary_color_continuity",
    "wall_boundary_low_frequency_gradient_continuity",
)
ALLOWED_OVERRIDE_GATES = frozenset((*REQUIRED_BOUNDARY_GATES, "visual_quality"))
FALSE_IS_SAFE_GATES = ("synthetic_anchor_claims_measured_donor",)
REAL_BOUNDARY_METRIC_KEYS = ("threshold_p95_abs_rgb_delta", "observed_maximum")
GENERIC_BOUNDARY_METRIC_KEYS = ("threshold", "observed")


class PlanarTextureAcceptanceError(ValueError):
    """Raised when evidence is insufficient or internally inconsistent."""


@dataclass(frozen=True)
class ReviewedFrame:
    frame_id: str
    completed_frame: Path
    completed_frame_sha256: str
    synthetic_mask: Path
    synthetic_mask_sha256: str


@dataclass(frozen=True)
class ReviewOutcome:
    decision: str
    accepted_with_limitations: bool
    acceptance_scope: str | None
    authority: dict[str, Any]
    limitations: list[str]
    frame_bindings: list[dict[str, Any]]
    override_gate_keys: tuple[str, ...]


@dataclass(frozen=True)
class ValidatedEvidence:
    candidate_path: Path
    candidate_sha256: str
    candidate: dict[str, Any]
    geometry_report_path: Path
    geometry_report_sha256: str
    visual_review_path: Path
    visual_review_sha256: str
    visual_review: dict[str, Any]
    reviewed_frames: tuple[ReviewedFrame, ...]
    review_outcome: ReviewOutcome


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PlanarTextureAcceptanceError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file(), f"{label} is missing: {path}")
    require(path.stat().st_size > 0, f"{label} is empty: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PlanarTextureAcceptanceError(f"cannot read {label} JSON: {exc}") from exc
    require(isinstance(value, dict) and value, f"{label} JSON must be a non-empty object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_declared_sha256(path: Path, declared: str, label: str) -> str:
    require(
        isinstance(declared, str) and SHA256_PATTERN.fullmatch(declared) is not None,
        f"{label} declared SHA-256 is invalid",
    )
    actual = sha256_file(path)
    require(actual == declared, f"{label} SHA-256 mismatch: {actual} != {declared}")
    return actual


def require_sha256(value: Any, label: str) -> str:
    require(
        isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None,
        f"{label} SHA-256 is invalid",
    )
    return value


def resolve_path(value: Any, *, relative_to: Path, label: str) -> Path:
    if isinstance(value, dict):
        value = value.get("path")
    require(isinstance(value, str) and value.strip(), f"{label} path is missing")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    resolved = path.resolve()
    require(resolved.is_file(), f"{label} is missing: {resolved}")
    return resolved


def require_file_binding(path: Path, expected_sha256: Any, label: str) -> str:
    expected = require_sha256(expected_sha256, label)
    actual = sha256_file(path)
    require(actual == expected, f"{label} SHA-256 mismatch: {actual} != {expected}")
    return actual


def require_nonempty_strings(value: Any, label: str) -> list[str]:
    require(isinstance(value, list) and value, f"{label} must be a non-empty list")
    require(
        all(isinstance(item, str) and item.strip() for item in value),
        f"{label} must contain only non-empty strings",
    )
    return list(value)


def require_frame_id(value: Any, label: str) -> str:
    require(isinstance(value, str), f"{label} frame_id is missing")
    require(
        FRAME_ID_PATTERN.fullmatch(value) is not None and value not in {".", ".."},
        f"{label} has unsafe frame_id: {value!r}",
    )
    return value


def gate_passed(value: Any) -> bool:
    if value is True:
        return True
    return isinstance(value, dict) and value.get("passed") is True


def validate_failed_boundary_evidence(value: Any, name: str) -> None:
    require(
        isinstance(value, dict) and value.get("passed") is False,
        f"limited candidate boundary gate is not an explicit failure: {name}",
    )
    if any(key in value for key in REAL_BOUNDARY_METRIC_KEYS):
        threshold_key, observed_key = REAL_BOUNDARY_METRIC_KEYS
    else:
        threshold_key, observed_key = GENERIC_BOUNDARY_METRIC_KEYS
    require(
        threshold_key in value and observed_key in value,
        f"limited candidate boundary gate lacks known threshold/observed evidence: {name}",
    )
    threshold = value[threshold_key]
    observed = value[observed_key]
    require(
        isinstance(threshold, (int, float))
        and not isinstance(threshold, bool)
        and isinstance(observed, (int, float))
        and not isinstance(observed, bool)
        and math.isfinite(float(threshold))
        and math.isfinite(float(observed)),
        f"limited candidate boundary gate metrics must be finite numbers: {name}",
    )
    require(
        float(observed) > float(threshold),
        f"limited candidate boundary gate observed value must exceed threshold: {name}",
    )


def validate_candidate_gates(gates: Any, outcome: ReviewOutcome) -> dict[str, Any]:
    require(isinstance(gates, dict) and gates, "candidate gates are missing")
    for name in REQUIRED_MATERIALIZER_GATES:
        require(gate_passed(gates.get(name)), f"candidate gate did not pass: {name}")
    failed_boundary_gates: set[str] = set()
    for name in REQUIRED_BOUNDARY_GATES:
        value = gates.get(name)
        if gate_passed(value):
            continue
        if not outcome.accepted_with_limitations:
            raise PlanarTextureAcceptanceError(f"candidate boundary gate did not pass: {name}")
        validate_failed_boundary_evidence(value, name)
        failed_boundary_gates.add(name)

    if outcome.accepted_with_limitations:
        expected_overrides = {"visual_quality", *failed_boundary_gates}
        require(
            set(outcome.override_gate_keys) == expected_overrides,
            "limited review override_gate_keys must exactly match visual_quality and failed "
            "boundary gates",
        )
    require(
        gates.get("visual_quality") == "pending_human_or_vlm_review",
        "candidate visual quality is not pending review",
    )
    require(
        gates.get("new_depth_normal_estimation_before_pgsr") == FRESH_DEPTH_NORMAL_REQUIREMENT,
        "candidate does not require fresh depth/normal estimation before PGSR",
    )

    handled = {
        *REQUIRED_MATERIALIZER_GATES,
        *REQUIRED_BOUNDARY_GATES,
        "visual_quality",
        "new_depth_normal_estimation_before_pgsr",
    }
    for name, value in gates.items():
        if name in handled:
            continue
        if name in FALSE_IS_SAFE_GATES:
            require(value is False, f"candidate negative-claim gate is unsafe: {name}")
        elif isinstance(value, bool):
            require(value is True, f"candidate technical gate did not pass: {name}")
        elif isinstance(value, dict) and "passed" in value:
            require(value.get("passed") is True, f"candidate technical gate did not pass: {name}")
        else:
            raise PlanarTextureAcceptanceError(
                f"candidate gate has an unsupported non-pass state: {name}={value!r}"
            )
    return gates


def validate_geometry_binding(candidate: dict[str, Any], candidate_dir: Path) -> tuple[Path, str]:
    path = resolve_path(
        candidate.get("planar_geometry_report"),
        relative_to=candidate_dir,
        label="planar geometry report",
    )
    digest = require_file_binding(
        path,
        candidate.get("planar_geometry_report_sha256"),
        "planar geometry report",
    )
    report = read_json_object(path, "planar geometry report")
    require(
        report.get("status") == "geometry_completed_texture_pending",
        "planar geometry report status is invalid",
    )
    gates = report.get("gates")
    require(isinstance(gates, dict), "planar geometry gates are missing")
    require(
        gate_passed(gates.get("minimum_geometry_coverage")),
        "planar geometry minimum coverage gate did not pass",
    )
    for name in ("outside_removal_mask_rgb_exact", "texture_masks_partition_removal_mask"):
        require(gates.get(name) is True, f"planar geometry gate did not pass: {name}")
    require(
        gates.get("new_depth_normal_estimation_before_pgsr")
        == "required_after_final_rgb_acceptance",
        "planar geometry report does not require fresh depth/normal estimation",
    )
    return path, digest


def validate_candidate_frames(
    candidate: dict[str, Any], candidate_dir: Path
) -> tuple[ReviewedFrame, ...]:
    frame_records = candidate.get("frame_records")
    require(isinstance(frame_records, list) and frame_records, "candidate frame_records are empty")
    require(
        all(isinstance(record, dict) for record in frame_records),
        "candidate frame record is not an object",
    )
    records_by_id: dict[str, dict[str, Any]] = {}
    for record in frame_records:
        frame_id = require_frame_id(record.get("frame_id"), "candidate frame record")
        require(frame_id not in records_by_id, f"candidate frame_id is duplicated: {frame_id}")
        records_by_id[frame_id] = record

    full_resolution = candidate.get("full_resolution_review")
    require(
        isinstance(full_resolution, list) and len(full_resolution) >= 3,
        "candidate full_resolution_review must contain at least 3 frames",
    )
    reviewed: list[ReviewedFrame] = []
    seen: set[str] = set()
    for item in full_resolution:
        require(isinstance(item, dict), "candidate full-resolution review item is not an object")
        frame_id = require_frame_id(item.get("frame_id"), "candidate full-resolution review")
        require(frame_id not in seen, f"candidate full-resolution frame is duplicated: {frame_id}")
        seen.add(frame_id)
        require(frame_id in records_by_id, f"candidate frame record is missing: {frame_id}")
        frame_record = records_by_id[frame_id]

        completed = resolve_path(
            item.get("completed_frame"),
            relative_to=candidate_dir,
            label=f"{frame_id} completed frame",
        )
        completed_sha = require_file_binding(
            completed,
            item.get("completed_frame_sha256"),
            f"{frame_id} completed frame",
        )
        mask = resolve_path(
            item.get("synthetic_mask"),
            relative_to=candidate_dir,
            label=f"{frame_id} synthetic mask",
        )
        mask_sha = require_file_binding(
            mask,
            item.get("synthetic_mask_sha256"),
            f"{frame_id} synthetic mask",
        )
        require(
            frame_record.get("completed_frame_sha256") == completed_sha,
            f"candidate full-resolution completed frame differs from frame_records: {frame_id}",
        )
        require(
            frame_record.get("synthetic_mask_sha256") == mask_sha,
            f"candidate full-resolution mask differs from frame_records: {frame_id}",
        )
        frame_completed = resolve_path(
            frame_record.get("completed_frame"),
            relative_to=candidate_dir,
            label=f"{frame_id} frame_records completed frame",
        )
        frame_mask = resolve_path(
            frame_record.get("synthetic_mask"),
            relative_to=candidate_dir,
            label=f"{frame_id} frame_records synthetic mask",
        )
        require(frame_completed == completed, f"candidate completed frame paths differ: {frame_id}")
        require(frame_mask == mask, f"candidate synthetic mask paths differ: {frame_id}")
        reviewed.append(
            ReviewedFrame(
                frame_id=frame_id,
                completed_frame=completed,
                completed_frame_sha256=completed_sha,
                synthetic_mask=mask,
                synthetic_mask_sha256=mask_sha,
            )
        )

    summary = candidate.get("full_resolution_boundary_continuity_summary")
    if summary is not None:
        require(isinstance(summary, dict), "full-resolution boundary summary is not an object")
        summary_ids = summary.get("frame_ids")
        require(
            isinstance(summary_ids, list)
            and len(summary_ids) == len(reviewed)
            and set(summary_ids) == {item.frame_id for item in reviewed},
            "full-resolution boundary summary frame IDs differ from review frames",
        )
    return tuple(reviewed)


def validate_review_authority(value: Any, *, accepted_with_limitations: bool) -> dict[str, Any]:
    require(isinstance(value, dict), "visual review authority is missing")
    expected_kind = (
        LIMITED_REVIEW_AUTHORITY_KIND if accepted_with_limitations else REVIEW_AUTHORITY_KIND
    )
    expected_scope = (
        LIMITED_ACCEPTANCE_SCOPE if accepted_with_limitations else REVIEW_AUTHORITY_SCOPE
    )
    require(
        value.get("kind") == expected_kind,
        f"visual review authority must be {expected_kind}",
    )
    reviewer_id = value.get("reviewer_id")
    require(
        isinstance(reviewer_id, str) and reviewer_id.strip(),
        "visual review authority reviewer_id is missing",
    )
    require(
        value.get("scope") == expected_scope,
        "visual review authority scope is invalid",
    )
    reviewed_at = value.get("reviewed_at")
    require(isinstance(reviewed_at, str) and reviewed_at, "visual review reviewed_at is missing")
    try:
        parsed = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PlanarTextureAcceptanceError("visual review reviewed_at is invalid") from exc
    require(
        parsed.tzinfo is not None and parsed.utcoffset() is not None,
        "visual review reviewed_at must include a timezone",
    )
    return copy.deepcopy(value)


def validate_visual_review(
    review: dict[str, Any],
    *,
    candidate_sha256: str,
    candidate_frames: tuple[ReviewedFrame, ...],
) -> ReviewOutcome:
    require(review.get("schema_version") == 1, "visual review schema_version must equal 1")
    require(review.get("kind") == REVIEW_KIND, f"visual review kind must equal {REVIEW_KIND!r}")
    require(review.get("status") == REVIEW_STATUS, "visual review is not completed")
    decision = review.get("decision")
    require(
        decision in {REVIEW_DECISION, LIMITED_REVIEW_DECISION},
        "visual review decision is not accepted",
    )
    accepted_with_limitations = decision == LIMITED_REVIEW_DECISION
    acceptance_scope: str | None = None
    override_gate_keys: tuple[str, ...] = ()
    if accepted_with_limitations:
        require(
            review.get("acceptance_scope") == LIMITED_ACCEPTANCE_SCOPE,
            "limited visual review acceptance_scope must be current_demo_only",
        )
        acceptance_scope = LIMITED_ACCEPTANCE_SCOPE
        override_values = require_nonempty_strings(
            review.get("override_gate_keys"), "limited visual review override_gate_keys"
        )
        require(
            len(override_values) == len(set(override_values)),
            "limited visual review override_gate_keys contain duplicates",
        )
        unknown = set(override_values) - ALLOWED_OVERRIDE_GATES
        require(
            not unknown,
            f"limited visual review has forbidden override gate keys: {sorted(unknown)}",
        )
        require(
            "visual_quality" in override_values,
            "limited visual review must explicitly override visual_quality",
        )
        override_gate_keys = tuple(override_values)
    require(
        review.get("candidate_report_sha256") == candidate_sha256,
        "visual review is bound to a different candidate report",
    )
    authority = validate_review_authority(
        review.get("review_authority"),
        accepted_with_limitations=accepted_with_limitations,
    )
    limitations = require_nonempty_strings(review.get("limitations"), "visual review limitations")
    require(review.get("blocking_findings") == [], "visual review has blocking findings")
    review_gates = review.get("gates")
    if review_gates is not None:
        require(isinstance(review_gates, dict) and review_gates, "visual review gates are empty")
        for name, value in review_gates.items():
            require(gate_passed(value), f"visual review gate did not pass: {name}")

    items = review.get("full_resolution_review")
    require(isinstance(items, list), "visual review full_resolution_review is missing")
    expected = {item.frame_id: item for item in candidate_frames}
    actual: dict[str, dict[str, Any]] = {}
    for item in items:
        require(isinstance(item, dict), "visual review frame item is not an object")
        frame_id = require_frame_id(item.get("frame_id"), "visual review")
        require(frame_id not in actual, f"visual review frame_id is duplicated: {frame_id}")
        actual[frame_id] = item
    require(
        len(actual) == len(expected) and set(actual) == set(expected),
        "visual review frame IDs do not exactly match candidate full_resolution_review",
    )

    bindings: list[dict[str, Any]] = []
    for frame in candidate_frames:
        item = actual[frame.frame_id]
        require(
            item.get("decision") == decision,
            f"visual review frame decision is not {decision}: {frame.frame_id}",
        )
        require(
            item.get("completed_frame_sha256") == frame.completed_frame_sha256,
            f"visual review completed frame binding mismatch: {frame.frame_id}",
        )
        require(
            item.get("synthetic_mask_sha256") == frame.synthetic_mask_sha256,
            f"visual review synthetic mask binding mismatch: {frame.frame_id}",
        )
        require(
            item.get("blocking_findings", []) == [],
            f"visual review frame has blocking findings: {frame.frame_id}",
        )
        bindings.append(
            {
                "frame_id": frame.frame_id,
                "completed_frame_sha256": frame.completed_frame_sha256,
                "synthetic_mask_sha256": frame.synthetic_mask_sha256,
                "decision": decision,
            }
        )
    return ReviewOutcome(
        decision=decision,
        accepted_with_limitations=accepted_with_limitations,
        acceptance_scope=acceptance_scope,
        authority=authority,
        limitations=limitations,
        frame_bindings=bindings,
        override_gate_keys=override_gate_keys,
    )


def validate_evidence(
    *,
    candidate_report_path: Path,
    candidate_report_sha256: str,
    visual_review_path: Path,
    visual_review_sha256: str,
) -> ValidatedEvidence:
    candidate_path = candidate_report_path.expanduser().resolve()
    review_path = visual_review_path.expanduser().resolve()
    require(candidate_path.is_file(), f"candidate report is missing: {candidate_path}")
    require(review_path.is_file(), f"visual review is missing: {review_path}")
    candidate_sha = validate_declared_sha256(
        candidate_path, candidate_report_sha256, "candidate report"
    )
    review_sha = validate_declared_sha256(review_path, visual_review_sha256, "visual review")
    candidate = read_json_object(candidate_path, "candidate report")
    review = read_json_object(review_path, "visual review")

    require(candidate.get("schema_version") == 1, "candidate schema_version must equal 1")
    require(candidate.get("status") == CANDIDATE_STATUS, "candidate status is not review-pending")
    require(candidate.get("promotion_approved") is False, "candidate promotion must be false")
    require(
        candidate.get("eligible_as_round04_clean_plate") is False,
        "candidate eligibility must be false",
    )
    blocker = candidate.get("promotion_blocker")
    require(
        isinstance(blocker, str) and blocker.strip(),
        "review-pending candidate must retain a non-empty promotion blocker",
    )
    require_nonempty_strings(candidate.get("limitations"), "candidate limitations")
    geometry_path, geometry_sha = validate_geometry_binding(candidate, candidate_path.parent)
    frames = validate_candidate_frames(candidate, candidate_path.parent)
    review_outcome = validate_visual_review(
        review,
        candidate_sha256=candidate_sha,
        candidate_frames=frames,
    )
    validate_candidate_gates(candidate.get("gates"), review_outcome)
    return ValidatedEvidence(
        candidate_path=candidate_path,
        candidate_sha256=candidate_sha,
        candidate=candidate,
        geometry_report_path=geometry_path,
        geometry_report_sha256=geometry_sha,
        visual_review_path=review_path,
        visual_review_sha256=review_sha,
        visual_review=review,
        reviewed_frames=frames,
        review_outcome=review_outcome,
    )


def rebase_materializer_paths(accepted: dict[str, Any], evidence: ValidatedEvidence) -> None:
    candidate_dir = evidence.candidate_path.parent
    accepted["planar_geometry_report"] = str(evidence.geometry_report_path)
    for record in accepted["frame_records"]:
        frame_id = str(record["frame_id"])
        for field in ("completed_frame", "source_frame", "synthetic_mask"):
            record[field] = str(
                resolve_path(
                    record.get(field),
                    relative_to=candidate_dir,
                    label=f"{frame_id} {field}",
                )
            )
    frames_by_id = {frame.frame_id: frame for frame in evidence.reviewed_frames}
    for item in accepted["full_resolution_review"]:
        frame = frames_by_id[str(item["frame_id"])]
        item["completed_frame"] = str(frame.completed_frame)
        item["synthetic_mask"] = str(frame.synthetic_mask)


def build_accepted_report(evidence: ValidatedEvidence, *, accepted_at: datetime) -> dict[str, Any]:
    outcome = evidence.review_outcome
    accepted = copy.deepcopy(evidence.candidate)
    rebase_materializer_paths(accepted, evidence)
    accepted["accepted_at"] = accepted_at.isoformat()
    accepted["status"] = ACCEPTED_STATUS
    accepted["promotion_blocker"] = None
    accepted["gates"]["visual_quality"] = outcome.decision
    accepted["promotion_approved"] = not outcome.accepted_with_limitations
    accepted["eligible_as_round04_clean_plate"] = not outcome.accepted_with_limitations
    accepted["acceptance"] = {
        "kind": "video2world.planar_texture_acceptance",
        "candidate_report": {
            "path": str(evidence.candidate_path),
            "sha256": evidence.candidate_sha256,
            "status": CANDIDATE_STATUS,
            "promotion_approved": False,
            "eligible_as_round04_clean_plate": False,
        },
        "visual_review": {
            "path": str(evidence.visual_review_path),
            "sha256": evidence.visual_review_sha256,
            "status": REVIEW_STATUS,
            "decision": outcome.decision,
        },
        "review_authority": outcome.authority,
        "reviewed_full_resolution_frames": outcome.frame_bindings,
        "review_limitations": outcome.limitations,
        "fresh_reconstruction_requirement": {
            "fresh_depth_required": True,
            "fresh_normals_required": True,
            "legacy_depth_allowed_as_new_depth": False,
            "pgsr_allowed_before_fresh_depth_and_normals": False,
        },
    }
    if outcome.accepted_with_limitations:
        overridden_gates = {
            name: copy.deepcopy(evidence.candidate["gates"][name])
            for name in sorted(outcome.override_gate_keys)
        }
        accepted["acceptance_scope"] = LIMITED_ACCEPTANCE_SCOPE
        accepted["accepted_with_limitations"] = True
        accepted["demo_use_approved"] = True
        accepted["eligible_as_current_demo_round04_clean_plate"] = True
        accepted["overridden_gates"] = overridden_gates
        accepted["acceptance"].update(
            {
                "acceptance_scope": LIMITED_ACCEPTANCE_SCOPE,
                "accepted_with_limitations": True,
                "promotion_approved": False,
                "eligible_as_round04_clean_plate": False,
                "demo_use_approved": True,
                "eligible_as_current_demo_round04_clean_plate": True,
                "override_gate_keys": list(outcome.override_gate_keys),
                "overridden_gates": copy.deepcopy(overridden_gates),
            }
        )
    return accepted


def sign_candidate(
    *,
    candidate_report_path: Path,
    candidate_report_sha256: str,
    visual_review_path: Path,
    visual_review_sha256: str,
    output_dir: Path,
    accepted_at: datetime | None = None,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    output_candidate = output_dir.expanduser()
    require(not output_candidate.is_symlink(), f"output cannot be a symlink: {output_candidate}")
    output = output_candidate.resolve()
    require(not output.exists(), f"output already exists: {output}")
    evidence = validate_evidence(
        candidate_report_path=candidate_report_path,
        candidate_report_sha256=candidate_report_sha256,
        visual_review_path=visual_review_path,
        visual_review_sha256=visual_review_sha256,
    )
    timestamp = accepted_at or datetime.now(UTC)
    require(
        timestamp.tzinfo is not None and timestamp.utcoffset() is not None,
        "accepted_at must include a timezone",
    )
    accepted = build_accepted_report(evidence, accepted_at=timestamp)
    outcome = evidence.review_outcome

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        report_path = staging / OUTPUT_REPORT_NAME
        write_json(report_path, accepted)
        report_sha = sha256_file(report_path)
        receipt = {
            "schema_version": 1,
            "kind": "video2world.planar_texture_acceptance_receipt",
            "created_at": timestamp.isoformat(),
            "status": ACCEPTED_STATUS,
            "promotion_approved": accepted["promotion_approved"],
            "eligible_as_round04_clean_plate": accepted["eligible_as_round04_clean_plate"],
            "candidate_report": {
                "path": str(evidence.candidate_path),
                "sha256": evidence.candidate_sha256,
            },
            "visual_review": {
                "path": str(evidence.visual_review_path),
                "sha256": evidence.visual_review_sha256,
                "decision": outcome.decision,
            },
            "accepted_report": {
                "path": OUTPUT_REPORT_NAME,
                "sha256": report_sha,
                "size_bytes": report_path.stat().st_size,
            },
            "reviewed_full_resolution_frame_count": len(evidence.reviewed_frames),
            "gates": {
                "visual_quality": outcome.decision,
                "fresh_depth_required": True,
                "fresh_normals_required": True,
                "materializer_contract": "accepted_planar_texture_report",
            },
        }
        if outcome.accepted_with_limitations:
            receipt.update(
                {
                    "acceptance_scope": LIMITED_ACCEPTANCE_SCOPE,
                    "accepted_with_limitations": True,
                    "demo_use_approved": True,
                    "eligible_as_current_demo_round04_clean_plate": True,
                    "override_gate_keys": list(outcome.override_gate_keys),
                    "overridden_gates": copy.deepcopy(accepted["overridden_gates"]),
                }
            )
        receipt_path = staging / OUTPUT_RECEIPT_NAME
        write_json(receipt_path, receipt)
        receipt_sha = sha256_file(receipt_path)
        (staging / OUTPUT_RECEIPT_HASH_NAME).write_text(
            f"{receipt_sha}  {OUTPUT_RECEIPT_NAME}\n",
            encoding="ascii",
        )

        require(
            sha256_file(evidence.candidate_path) == evidence.candidate_sha256,
            "candidate report changed during signing",
        )
        require(
            sha256_file(evidence.visual_review_path) == evidence.visual_review_sha256,
            "visual review changed during signing",
        )
        require(not output.exists(), f"output appeared during signing: {output}")
        os.rename(staging, output)
        return accepted, receipt, receipt_sha
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--candidate-report-sha256", required=True)
    parser.add_argument("--visual-review", type=Path, required=True)
    parser.add_argument("--visual-review-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    accepted, receipt, receipt_sha = sign_candidate(
        candidate_report_path=args.candidate_report,
        candidate_report_sha256=args.candidate_report_sha256,
        visual_review_path=args.visual_review,
        visual_review_sha256=args.visual_review_sha256,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "status": accepted["status"],
                "promotion_approved": accepted["promotion_approved"],
                "eligible_as_round04_clean_plate": accepted["eligible_as_round04_clean_plate"],
                "reviewed_full_resolution_frame_count": receipt[
                    "reviewed_full_resolution_frame_count"
                ],
                "accepted_report": str(args.output_dir.resolve() / OUTPUT_REPORT_NAME),
                "receipt": str(args.output_dir.resolve() / OUTPUT_RECEIPT_NAME),
                "receipt_sha256": receipt_sha,
                "next_stage": "materialize_clean_scene_reconstruction_input",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
