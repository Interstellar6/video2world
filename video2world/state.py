"""Durable per-stage state models."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import Field

from video2world.hashing import atomic_write_json
from video2world.models import Sha256, StrictModel


class ArtifactSnapshot(StrictModel):
    path: str = Field(min_length=1)
    kind: Literal["file", "directory"]
    sha256: Sha256
    size_bytes: int = Field(ge=1)
    file_count: int = Field(ge=1)


class AdoptionRecord(StrictModel):
    source_run_id: str = Field(min_length=1)
    source_root: str = Field(min_length=1)
    source_repository: str | None = None
    source_commit: str | None = None
    source_command: str | None = None
    note: str | None = None
    adopted_at: datetime


class StageState(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    stage_id: str
    adapter: str
    status: Literal["running", "succeeded", "failed", "adopted"]
    mode: Literal["executed", "adopted"]
    attempt: int = Field(ge=1)
    input_digest: Sha256
    config_digest: Sha256
    dependency_outputs: dict[str, dict[str, ArtifactSnapshot]] = Field(default_factory=dict)
    explicit_inputs: dict[str, ArtifactSnapshot] = Field(default_factory=dict)
    outputs: dict[str, ArtifactSnapshot] = Field(default_factory=dict)
    command: list[str] | None = None
    environment_keys: list[str] = Field(default_factory=list)
    started_at: datetime
    finished_at: datetime | None = None
    log_path: str | None = None
    adoption: AdoptionRecord | None = None
    error: str | None = None


class StagePlan(StrictModel):
    stage_id: str
    adapter: str
    action: Literal["run", "cached", "stale", "waiting", "adopt_or_configure", "disabled"]
    reason: str
    input_digest: Sha256 | None = None
    command: list[str] | None = None
    outputs: dict[str, str]


def state_path(run_dir: Path, stage_id: str) -> Path:
    return run_dir / ".video2world" / "stages" / f"{stage_id}.json"


def load_stage_state(run_dir: Path, stage_id: str) -> StageState | None:
    path = state_path(run_dir, stage_id)
    if not path.exists():
        return None
    return StageState.model_validate_json(path.read_text(encoding="utf-8"))


def write_stage_state(run_dir: Path, state: StageState) -> None:
    atomic_write_json(state_path(run_dir, state.stage_id), state.model_dump(mode="json"))
