"""Atomic, idempotent submission of confirmed scene-command plans to an async queue."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from video2world.hashing import atomic_write_json, digest_json
from video2world.models import Sha256, StrictModel, WorldManifest
from video2world.scene_commands import SceneCommand, SceneCommandPlan, plan_scene_command


class QueuedSceneCommandJob(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["video2world.scene_command_job"] = "video2world.scene_command_job"
    job_id: str = Field(pattern=r"^scenejob_[0-9a-f]{20}$")
    queued_at: datetime
    status: Literal["queued"] = "queued"
    attempt: Literal[0] = 0
    idempotency_key: Sha256
    manifest_snapshot_sha256: Sha256
    command: SceneCommand
    plan: SceneCommandPlan

    @model_validator(mode="after")
    def job_matches_plan(self) -> QueuedSceneCommandJob:
        if self.queued_at.tzinfo is None or self.queued_at.utcoffset() is None:
            raise ValueError("queued_at must include timezone information")
        if self.plan.idempotency_key != self.idempotency_key:
            raise ValueError("job idempotency key does not match plan")
        if self.plan.provenance.canonical_manifest_sha256 != self.manifest_snapshot_sha256:
            raise ValueError("job manifest snapshot does not match plan")
        if self.plan.status != "ready" or not self.command.is_mutating:
            raise ValueError("queued jobs require a ready mutating command plan")
        return self


class SceneCommandSubmissionReceipt(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["video2world.scene_command_submission"] = "video2world.scene_command_submission"
    submitted_at: datetime
    status: Literal["queued", "duplicate", "completed_read", "blocked"]
    request_id: str
    plan_id: str
    idempotency_key: Sha256
    manifest_snapshot_sha256: Sha256
    job_id: str | None = None
    queue_path: str | None = None
    read_result: dict[str, object] | None = None
    blocked_reason: str | None = None

    @model_validator(mode="after")
    def receipt_shape(self) -> SceneCommandSubmissionReceipt:
        if self.submitted_at.tzinfo is None or self.submitted_at.utcoffset() is None:
            raise ValueError("submitted_at must include timezone information")
        queued = self.status in {"queued", "duplicate"}
        if queued != (self.job_id is not None and self.queue_path is not None):
            raise ValueError("queued/duplicate receipts require job_id and queue_path")
        if (self.status == "completed_read") != (self.read_result is not None):
            raise ValueError("completed_read receipts require read_result")
        if (self.status == "blocked") != (self.blocked_reason is not None):
            raise ValueError("blocked receipts require blocked_reason")
        return self


def _exclusive_write_json(path: Path, value: dict[str, object]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
        + b"\n"
    )
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return True


def submit_scene_command(
    manifest: WorldManifest,
    command: SceneCommand,
    *,
    queue_dir: str | Path,
    confirmation_phrase: str | None = None,
    submitted_at: datetime | None = None,
) -> SceneCommandSubmissionReceipt:
    """Return read results immediately or enqueue an immutable confirmed mutation job."""

    timestamp = submitted_at or datetime.now(UTC)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("submitted_at must include timezone information")
    plan = plan_scene_command(manifest, command, created_at=timestamp)
    snapshot_payload = manifest.model_dump(mode="json")
    snapshot_sha256 = digest_json(snapshot_payload)
    if plan.status != "ready":
        return SceneCommandSubmissionReceipt(
            submitted_at=timestamp,
            status="blocked",
            request_id=command.request_id,
            plan_id=plan.plan_id,
            idempotency_key=plan.idempotency_key,
            manifest_snapshot_sha256=snapshot_sha256,
            blocked_reason="; ".join(plan.warnings) or plan.status,
        )
    if not command.is_mutating:
        assert plan.read_result is not None
        return SceneCommandSubmissionReceipt(
            submitted_at=timestamp,
            status="completed_read",
            request_id=command.request_id,
            plan_id=plan.plan_id,
            idempotency_key=plan.idempotency_key,
            manifest_snapshot_sha256=snapshot_sha256,
            read_result=plan.read_result.model_dump(mode="json", exclude_none=True),
        )
    if plan.confirmation.required:
        if confirmation_phrase != plan.confirmation.required_phrase:
            raise ValueError(
                "confirmation phrase does not match this manifest-bound plan; expected "
                f"{plan.confirmation.required_phrase!r}"
            )
    elif confirmation_phrase not in {None, ""}:
        raise ValueError("this low-risk plan does not accept an unrelated confirmation phrase")

    root = Path(queue_dir).expanduser().resolve()
    snapshots = root / "manifest_snapshots"
    jobs = root / "jobs"
    receipts = root / "receipts"
    snapshot_path = snapshots / f"{snapshot_sha256}.json"
    if not snapshot_path.exists():
        atomic_write_json(snapshot_path, snapshot_payload)
    job_id = f"scenejob_{plan.idempotency_key[:20]}"
    job_path = jobs / f"{job_id}.json"
    job = QueuedSceneCommandJob(
        job_id=job_id,
        queued_at=timestamp,
        idempotency_key=plan.idempotency_key,
        manifest_snapshot_sha256=snapshot_sha256,
        command=command,
        plan=plan,
    )
    created = _exclusive_write_json(job_path, job.model_dump(mode="json", exclude_none=True))
    if not created:
        existing = QueuedSceneCommandJob.model_validate_json(job_path.read_text(encoding="utf-8"))
        if existing.idempotency_key != plan.idempotency_key:
            raise ValueError("existing queue job has a conflicting idempotency key")
    receipt = SceneCommandSubmissionReceipt(
        submitted_at=timestamp,
        status="queued" if created else "duplicate",
        request_id=command.request_id,
        plan_id=plan.plan_id,
        idempotency_key=plan.idempotency_key,
        manifest_snapshot_sha256=snapshot_sha256,
        job_id=job_id,
        queue_path=str(job_path),
    )
    receipt_path = receipts / f"{job_id}.json"
    if created or not receipt_path.exists():
        atomic_write_json(receipt_path, receipt.model_dump(mode="json", exclude_none=True))
    return receipt
