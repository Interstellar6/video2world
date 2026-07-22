#!/usr/bin/env python3
"""Generate a fail-closed visual review for a planar texture candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REVIEW_KIND = "video2world.planar_texture_visual_review"
REVIEW_STATUS = "completed"
CANDIDATE_STATUS = "texture_candidate_review_pending"
REJECTED_DECISION = "rejected"
NEEDS_REPAIR_STATUS = "texture_candidate_rejected_or_needs_repair"
METRIC_PASS_PENDING_VISUAL_STATUS = "texture_candidate_metric_pass_visual_review_required"
OUTPUT_REVIEW_NAME = "planar_texture_visual_review.json"
OUTPUT_RECEIPT_NAME = "planar_texture_visual_review_receipt.json"
OUTPUT_RECEIPT_HASH_NAME = "planar_texture_visual_review_receipt.sha256"
DEFAULT_MAX_BOUNDARY_P95 = 48.0
DEFAULT_MAX_GRADIENT_P95 = 64.0


class PlanarTextureReviewError(ValueError):
    """Raised when the candidate cannot be reviewed safely."""


@dataclass(frozen=True)
class BoundFrame:
    frame_id: str
    completed_frame_sha256: str
    synthetic_mask_sha256: str
    boundary_color_p95: float
    boundary_gradient_p95: float


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PlanarTextureReviewError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file(), f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PlanarTextureReviewError(f"cannot read {label}: {exc}") from exc
    require(isinstance(value, dict) and value, f"{label} must be a non-empty object")
    return value


def resolve_file(value: Any, *, relative_to: Path, label: str) -> Path:
    if isinstance(value, dict):
        value = value.get("path")
    require(isinstance(value, str) and value.strip(), f"{label} path is missing")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    resolved = path.resolve()
    require(resolved.is_file(), f"{label} is missing: {resolved}")
    return resolved


def finite_number(value: Any, label: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be a number",
    )
    result = float(value)
    require(result == result and result not in {float("inf"), float("-inf")}, f"{label} invalid")
    return result


def frame_boundary_metrics(record: dict[str, Any], *, label: str) -> tuple[float, float]:
    metrics = record.get("synthetic_boundary_continuity_full_resolution")
    require(isinstance(metrics, dict), f"{label} lacks full-resolution boundary metrics")
    return (
        finite_number(metrics.get("boundary_color_p95_abs_rgb_delta"), f"{label} color p95"),
        finite_number(
            metrics.get("boundary_normal_gradient_p95_abs_rgb_delta"), f"{label} gradient p95"
        ),
    )


def validate_candidate(candidate: dict[str, Any]) -> None:
    require(candidate.get("schema_version") == 1, "candidate schema_version must equal 1")
    require(candidate.get("status") == CANDIDATE_STATUS, "candidate status is not review-pending")
    require(candidate.get("promotion_approved") is False, "candidate is already promoted")
    require(
        candidate.get("eligible_as_round04_clean_plate") is False,
        "candidate is already eligible as a clean plate",
    )
    gates = candidate.get("gates")
    require(isinstance(gates, dict), "candidate gates are missing")
    require(
        gates.get("visual_quality") == "pending_human_or_vlm_review",
        "candidate visual quality is not pending review",
    )
    require(
        gates.get("all_synthetic_pixels_assigned_from_shared_atlas") is True,
        "candidate synthetic pixels are not all bound to the shared atlas",
    )
    require(
        gates.get("outside_synthetic_mask_rgb_exact") is True,
        "candidate changes pixels outside the synthetic mask",
    )


def bind_review_frames(candidate: dict[str, Any], candidate_dir: Path) -> list[BoundFrame]:
    frame_records = candidate.get("frame_records")
    require(isinstance(frame_records, list) and frame_records, "candidate frame_records are empty")
    records_by_id: dict[str, dict[str, Any]] = {}
    for record in frame_records:
        require(isinstance(record, dict), "candidate frame record is not an object")
        frame_id = record.get("frame_id")
        require(isinstance(frame_id, str) and frame_id, "candidate frame_id is missing")
        records_by_id[frame_id] = record

    full_resolution = candidate.get("full_resolution_review")
    require(
        isinstance(full_resolution, list) and len(full_resolution) >= 3,
        "candidate full_resolution_review must contain at least 3 frames",
    )
    bound: list[BoundFrame] = []
    for item in full_resolution:
        require(isinstance(item, dict), "full-resolution review item is not an object")
        frame_id = item.get("frame_id")
        require(isinstance(frame_id, str) and frame_id, "review frame_id is missing")
        record = records_by_id.get(frame_id)
        require(record is not None, f"candidate frame record is missing: {frame_id}")
        completed_path = resolve_file(
            item.get("completed_frame"),
            relative_to=candidate_dir,
            label=f"{frame_id} completed frame",
        )
        mask_path = resolve_file(
            item.get("synthetic_mask"),
            relative_to=candidate_dir,
            label=f"{frame_id} synthetic mask",
        )
        completed_sha = str(item.get("completed_frame_sha256", ""))
        mask_sha = str(item.get("synthetic_mask_sha256", ""))
        require(
            sha256_file(completed_path) == completed_sha,
            f"{frame_id} completed frame SHA-256 mismatch",
        )
        require(sha256_file(mask_path) == mask_sha, f"{frame_id} mask SHA-256 mismatch")
        color_p95, gradient_p95 = frame_boundary_metrics(record, label=frame_id)
        bound.append(
            BoundFrame(
                frame_id=frame_id,
                completed_frame_sha256=completed_sha,
                synthetic_mask_sha256=mask_sha,
                boundary_color_p95=color_p95,
                boundary_gradient_p95=gradient_p95,
            )
        )
    return bound


def worst_frame_records(candidate: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in candidate.get("frame_records", []):
        if not isinstance(record, dict):
            continue
        frame_id = record.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id:
            continue
        color_p95, gradient_p95 = frame_boundary_metrics(record, label=frame_id)
        records.append(
            {
                "frame_id": frame_id,
                "boundary_color_p95_abs_rgb_delta": color_p95,
                "boundary_normal_gradient_p95_abs_rgb_delta": gradient_p95,
                "synthetic_pixels": record.get("synthetic_pixels"),
                "interpolated_pixels": record.get("interpolated_pixels"),
            }
        )
    records.sort(
        key=lambda item: (
            item["boundary_color_p95_abs_rgb_delta"],
            item["boundary_normal_gradient_p95_abs_rgb_delta"],
        ),
        reverse=True,
    )
    return records[:limit]


def build_review(
    *,
    candidate_path: Path,
    candidate_sha256: str,
    candidate: dict[str, Any],
    reviewed_at: datetime,
    reviewer_id: str,
    max_boundary_p95: float,
    max_gradient_p95: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_candidate(candidate)
    frames = bind_review_frames(candidate, candidate_path.parent)
    max_observed_boundary = max(frame.boundary_color_p95 for frame in frames)
    max_observed_gradient = max(frame.boundary_gradient_p95 for frame in frames)
    boundary_passed = max_observed_boundary <= max_boundary_p95
    gradient_passed = max_observed_gradient <= max_gradient_p95
    decision = REJECTED_DECISION
    blocking_findings: list[dict[str, Any]] = []
    if not boundary_passed:
        blocking_findings.append(
            {
                "gate": "full_resolution_boundary_color_continuity",
                "threshold_p95_abs_rgb_delta": max_boundary_p95,
                "observed_maximum": max_observed_boundary,
                "severity": "blocking",
            }
        )
    if not gradient_passed:
        blocking_findings.append(
            {
                "gate": "full_resolution_boundary_low_frequency_gradient_continuity",
                "threshold_p95_abs_rgb_delta": max_gradient_p95,
                "observed_maximum": max_observed_gradient,
                "severity": "blocking",
            }
        )
    if not blocking_findings:
        blocking_findings.append(
            {
                "gate": "visual_quality",
                "observed": "pending_human_or_vlm_review",
                "severity": "blocking",
            }
        )
        triage_status = METRIC_PASS_PENDING_VISUAL_STATUS
        next_action = (
            "run_bound_human_or_vlm_visual_review_before_acceptance; do not promote until "
            "photorealism, semantic plausibility, and wall-floor boundary quality are accepted"
        )
        visual_gate = "pending_human_or_vlm_review_after_metric_pass"
    else:
        triage_status = NEEDS_REPAIR_STATUS
        next_action = (
            "repair_texture_candidate_before_acceptance; suggested first attempt is "
            "complete_planar_texture_atlas --plane-footprint-neutralization"
        )
        visual_gate = "rejected_until_repaired_or_human_vlm_accepts"

    full_resolution_review = [
        {
            "frame_id": frame.frame_id,
            "completed_frame_sha256": frame.completed_frame_sha256,
            "synthetic_mask_sha256": frame.synthetic_mask_sha256,
            "decision": decision,
            "blocking_findings": [
                finding["gate"]
                for finding in blocking_findings
                if finding["gate"].startswith("full_resolution")
            ],
            "boundary_color_p95_abs_rgb_delta": frame.boundary_color_p95,
            "boundary_normal_gradient_p95_abs_rgb_delta": frame.boundary_gradient_p95,
        }
        for frame in frames
    ]
    review = {
        "schema_version": 1,
        "kind": REVIEW_KIND,
        "status": REVIEW_STATUS,
        "decision": decision,
        "candidate_report": str(candidate_path),
        "candidate_report_sha256": candidate_sha256,
        "review_authority": {
            "kind": "automated_metric_review",
            "reviewer_id": reviewer_id,
            "scope": "round04_clean_plate_texture_repair_triage",
            "reviewed_at": reviewed_at.isoformat(),
        },
        "gates": {
            "full_resolution_boundary_color_continuity": {
                "threshold_p95_abs_rgb_delta": max_boundary_p95,
                "observed_maximum": max_observed_boundary,
                "passed": boundary_passed,
            },
            "full_resolution_boundary_low_frequency_gradient_continuity": {
                "threshold_p95_abs_rgb_delta": max_gradient_p95,
                "observed_maximum": max_observed_gradient,
                "passed": gradient_passed,
            },
            "visual_quality": visual_gate,
        },
        "blocking_findings": blocking_findings,
        "limitations": [
            "This automated review only evaluates bound full-resolution boundary metrics.",
            (
                "A future human or VLM review must still inspect photorealism "
                "and semantic plausibility."
            ),
        ],
        "full_resolution_review": full_resolution_review,
        "worst_frame_records": worst_frame_records(candidate, limit=10),
        "next_action": next_action,
    }
    receipt = {
        "schema_version": 1,
        "kind": "video2world.planar_texture_visual_review_receipt",
        "created_at": reviewed_at.isoformat(),
        "status": triage_status,
        "decision": decision,
        "promotion_approved": False,
        "eligible_as_round04_clean_plate": False,
        "candidate_report": {
            "path": str(candidate_path),
            "sha256": candidate_sha256,
            "status": candidate.get("status"),
        },
        "reviewed_full_resolution_frame_count": len(frames),
        "blocking_findings": blocking_findings,
        "next_action": review["next_action"],
    }
    return review, receipt


def review_candidate(
    *,
    candidate_report_path: Path,
    output_dir: Path,
    reviewer_id: str,
    max_boundary_p95: float = DEFAULT_MAX_BOUNDARY_P95,
    max_gradient_p95: float = DEFAULT_MAX_GRADIENT_P95,
    reviewed_at: datetime | None = None,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    output_candidate = output_dir.expanduser()
    require(not output_candidate.is_symlink(), f"output cannot be a symlink: {output_candidate}")
    output = output_candidate.resolve()
    require(not output.exists(), f"output already exists: {output}")
    candidate_path = candidate_report_path.expanduser().resolve()
    candidate = read_json_object(candidate_path, "candidate report")
    candidate_sha = sha256_file(candidate_path)
    timestamp = reviewed_at or datetime.now(UTC)
    require(
        timestamp.tzinfo is not None and timestamp.utcoffset() is not None,
        "reviewed_at must include a timezone",
    )
    review, receipt = build_review(
        candidate_path=candidate_path,
        candidate_sha256=candidate_sha,
        candidate=candidate,
        reviewed_at=timestamp,
        reviewer_id=reviewer_id,
        max_boundary_p95=max_boundary_p95,
        max_gradient_p95=max_gradient_p95,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        review_path = staging / OUTPUT_REVIEW_NAME
        write_json(review_path, review)
        review_sha = sha256_file(review_path)
        receipt["visual_review"] = {
            "path": OUTPUT_REVIEW_NAME,
            "sha256": review_sha,
            "size_bytes": review_path.stat().st_size,
        }
        receipt_path = staging / OUTPUT_RECEIPT_NAME
        write_json(receipt_path, receipt)
        receipt_sha = sha256_file(receipt_path)
        (staging / OUTPUT_RECEIPT_HASH_NAME).write_text(
            f"{receipt_sha}  {OUTPUT_RECEIPT_NAME}\n",
            encoding="ascii",
        )
        require(
            sha256_file(candidate_path) == candidate_sha,
            "candidate report changed during review",
        )
        require(not output.exists(), f"output appeared during review: {output}")
        os.rename(staging, output)
        return review, receipt, receipt_sha
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reviewer-id",
        default="video2world-auto-boundary-reviewer",
        help="Stable reviewer identifier written into the review authority.",
    )
    parser.add_argument(
        "--maximum-full-resolution-boundary-p95-delta",
        type=float,
        default=DEFAULT_MAX_BOUNDARY_P95,
    )
    parser.add_argument(
        "--maximum-full-resolution-gradient-p95-delta",
        type=float,
        default=DEFAULT_MAX_GRADIENT_P95,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    review, receipt, receipt_sha = review_candidate(
        candidate_report_path=args.candidate_report,
        output_dir=args.output_dir,
        reviewer_id=args.reviewer_id,
        max_boundary_p95=args.maximum_full_resolution_boundary_p95_delta,
        max_gradient_p95=args.maximum_full_resolution_gradient_p95_delta,
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "decision": review["decision"],
                "promotion_approved": receipt["promotion_approved"],
                "eligible_as_round04_clean_plate": receipt["eligible_as_round04_clean_plate"],
                "review": str(args.output_dir.resolve() / OUTPUT_REVIEW_NAME),
                "receipt": str(args.output_dir.resolve() / OUTPUT_RECEIPT_NAME),
                "receipt_sha256": receipt_sha,
                "next_action": receipt["next_action"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
