"""Materialize auditable recovery work orders from blocked completion routes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from video2world.completion_routing import CompletionBackendRoute, RecoveryStage
from video2world.hashing import digest_json, digest_path
from video2world.models import Sha256, StrictModel


class RecoveryInputArtifact(StrictModel):
    path: str = Field(min_length=1)
    sha256: Sha256
    size_bytes: int = Field(ge=1)
    role: Literal["completion_backend_route", "clean_plate_report"]


class RecoveryFrameRecord(StrictModel):
    frame_id: str = Field(min_length=1)
    removal_mask_pixels: int | None = Field(default=None, ge=0)
    residual_mask_pixels: int | None = Field(default=None, ge=0)
    covered_pixels: int | None = Field(default=None, ge=0)
    coverage_fraction: float | None = Field(default=None, ge=0, le=1)
    support_max: int | None = Field(default=None, ge=0)


class RecoveryWorkItem(StrictModel):
    priority: int = Field(ge=1)
    stage: RecoveryStage
    action: str = Field(min_length=1)
    blocker: str | None = Field(default=None, min_length=1)
    reason: str = Field(min_length=1)
    frame_ids: list[str] = Field(default_factory=list)
    frame_records: list[RecoveryFrameRecord] = Field(default_factory=list)
    allow_deeper_rounds: Literal[False] = False
    required_verification: str = Field(min_length=1)


class CompletionRecoveryWorkOrder(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["video2world.completion_recovery_work_order"] = (
        "video2world.completion_recovery_work_order"
    )
    object_id: str = Field(min_length=1)
    route_sha256: Sha256
    work_order_sha256: Sha256
    status: Literal["blocked_pending_recovery"] = "blocked_pending_recovery"
    selected_backend: str = Field(min_length=1)
    deeper_rounds_blocked: Literal[True] = True
    output_claim: Literal["work_order_only_no_clean_plate_generated"] = (
        "work_order_only_no_clean_plate_generated"
    )
    inputs: list[RecoveryInputArtifact] = Field(min_length=2, max_length=2)
    work_items: list[RecoveryWorkItem] = Field(min_length=1)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def verify_recovery_contract(self) -> CompletionRecoveryWorkOrder:
        priorities = [item.priority for item in self.work_items]
        if priorities != sorted(set(priorities)):
            raise ValueError("work item priorities must be unique and ascending")
        if any(item.allow_deeper_rounds is not False for item in self.work_items):
            raise ValueError("all recovery work items must block deeper rounds")
        return self


def _artifact(
    path: Path,
    role: Literal["completion_backend_route", "clean_plate_report"],
) -> RecoveryInputArtifact:
    digest = digest_path(path)
    return RecoveryInputArtifact(
        path=str(digest.path),
        sha256=digest.sha256,
        size_bytes=digest.size_bytes,
        role=role,
    )


def _frame_records_by_id(report: dict[str, Any]) -> dict[str, RecoveryFrameRecord]:
    result: dict[str, RecoveryFrameRecord] = {}
    records = report.get("frame_records")
    if not isinstance(records, list):
        return result
    for record in records:
        if not isinstance(record, dict):
            continue
        frame_id = record.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id:
            continue
        result[frame_id] = RecoveryFrameRecord(
            frame_id=frame_id,
            removal_mask_pixels=record.get("removal_mask_pixels"),
            residual_mask_pixels=record.get("residual_mask_pixels"),
            covered_pixels=record.get("covered_pixels"),
            coverage_fraction=record.get("coverage_fraction"),
            support_max=record.get("support_max"),
        )
    return result


def _required_verification(stage: RecoveryStage) -> str:
    if stage == "donor_support":
        return "rerun measured donor support and strict boundary guard before any R2 input"
    if stage == "boundary_qa":
        return "rerun boundary QA with failed frame records preserved in the report"
    if stage == "temporal_qa":
        return "rerun temporal pair/triplet QA before candidate promotion"
    return "rerun clean-plate residual generation QA before promotion"


def materialize_completion_recovery_work_order(
    route_path: str | Path,
    clean_plate_report_path: str | Path,
) -> CompletionRecoveryWorkOrder:
    route_file = Path(route_path).expanduser().resolve()
    report_file = Path(clean_plate_report_path).expanduser().resolve()
    route_payload = json.loads(route_file.read_text(encoding="utf-8"))
    report_payload = json.loads(report_file.read_text(encoding="utf-8"))
    if not isinstance(route_payload, dict):
        raise ValueError("completion route root must be an object")
    if not isinstance(report_payload, dict):
        raise ValueError("clean plate report root must be an object")
    route = CompletionBackendRoute.model_validate(route_payload)
    if not route.recovery_actions:
        raise ValueError("completion route has no recovery_actions to materialize")
    frame_records = _frame_records_by_id(report_payload)
    work_items = [
        RecoveryWorkItem(
            priority=action.priority,
            stage=action.stage,
            action=action.action,
            blocker=action.blocker,
            reason=action.reason,
            frame_ids=action.frame_ids,
            frame_records=[
                frame_records[frame_id]
                for frame_id in action.frame_ids
                if frame_id in frame_records
            ],
            allow_deeper_rounds=action.allow_deeper_rounds,
            required_verification=_required_verification(action.stage),
        )
        for action in route.recovery_actions
    ]
    payload = {
        "object_id": route.object_id,
        "route_sha256": route.route_sha256,
        "status": "blocked_pending_recovery",
        "selected_backend": route.selected_backend,
        "deeper_rounds_blocked": True,
        "output_claim": "work_order_only_no_clean_plate_generated",
        "inputs": [
            _artifact(route_file, "completion_backend_route").model_dump(mode="json"),
            _artifact(report_file, "clean_plate_report").model_dump(mode="json"),
        ],
        "work_items": [item.model_dump(mode="json") for item in work_items],
        "notes": [
            (
                "This work order is a recovery plan only; it does not accept R1 or "
                "generate clean plates."
            ),
            "R2-R4 remain blocked until the listed verification steps pass.",
        ],
    }
    return CompletionRecoveryWorkOrder(
        work_order_sha256=digest_json(payload),
        **payload,
    )
