"""Run configuration loading and DAG validation."""

from __future__ import annotations

import re
from collections import deque
from pathlib import Path
from string import Formatter
from typing import Any

import yaml
from pydantic import Field, model_validator

from video2world.errors import ConfigurationError
from video2world.hashing import atomic_write_text, digest_json
from video2world.models import StrictModel

IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
ENV_REFERENCE = re.compile(r"^\$\{[A-Za-z_][A-Za-z0-9_]*\}$")
SENSITIVE_NAME = re.compile(r"(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", re.IGNORECASE)
RESERVED_VARIABLES = {"run_dir", "run_id", "scene_id", "input_video"}


class StageConfig(StrictModel):
    adapter: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    needs: list[str] = Field(default_factory=list)
    command: list[str] | None = None
    cwd: str = "{run_dir}"
    env: dict[str, str] = Field(default_factory=dict)
    inputs: dict[str, str] = Field(default_factory=dict)
    fingerprints: dict[str, str] = Field(default_factory=dict)
    outputs: dict[str, str] = Field(min_length=1)
    enabled: bool = True
    timeout_seconds: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def command_is_argv(self) -> StageConfig:
        if self.command is not None and not self.command:
            raise ValueError("command must be null or a non-empty argv list")
        for mapping_name, mapping in (
            ("inputs", self.inputs),
            ("fingerprints", self.fingerprints),
            ("outputs", self.outputs),
            ("env", self.env),
        ):
            invalid = sorted(key for key in mapping if not IDENTIFIER.fullmatch(key))
            if invalid:
                raise ValueError(f"invalid {mapping_name} identifiers: {invalid}")
        unsafe_environment = sorted(
            key
            for key, value in self.env.items()
            if SENSITIVE_NAME.search(key) and not ENV_REFERENCE.fullmatch(value)
        )
        if unsafe_environment:
            raise ValueError(
                "sensitive environment values must use ${ENV_VAR} references: "
                f"{unsafe_environment}"
            )
        return self


class RunConfig(StrictModel):
    schema_version: str = Field(pattern=r"^1\.0$")
    run_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    scene_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    input_video: str = Field(min_length=1)
    run_dir: str = Field(min_length=1)
    variables: dict[str, str] = Field(default_factory=dict)
    stages: dict[str, StageConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_dag(self) -> RunConfig:
        invalid_stages = sorted(
            stage_id for stage_id in self.stages if not IDENTIFIER.fullmatch(stage_id)
        )
        if invalid_stages:
            raise ValueError(f"invalid stage identifiers: {invalid_stages}")
        invalid_variables = sorted(
            name
            for name in self.variables
            if not IDENTIFIER.fullmatch(name) or name in RESERVED_VARIABLES
        )
        if invalid_variables:
            raise ValueError(f"invalid or reserved variable names: {invalid_variables}")
        secret_variables = sorted(name for name in self.variables if SENSITIVE_NAME.search(name))
        if secret_variables:
            raise ValueError(
                f"secrets must be passed through stage.env, not variables: {secret_variables}"
            )
        unknown = {
            dependency
            for stage in self.stages.values()
            for dependency in stage.needs
            if dependency not in self.stages
        }
        if unknown:
            raise ValueError(f"unknown stage dependencies: {sorted(unknown)}")
        self.topological_order()
        return self

    def topological_order(self, targets: list[str] | None = None) -> list[str]:
        if targets:
            missing = sorted(set(targets) - self.stages.keys())
            if missing:
                raise ConfigurationError(f"unknown target stages: {missing}")
            selected: set[str] = set()

            def include(stage_id: str) -> None:
                if stage_id in selected:
                    return
                selected.add(stage_id)
                for dependency in self.stages[stage_id].needs:
                    include(dependency)

            for stage_id in targets:
                include(stage_id)
        else:
            selected = set(self.stages)

        indegree = {stage_id: 0 for stage_id in selected}
        children: dict[str, list[str]] = {stage_id: [] for stage_id in selected}
        for stage_id in self.stages:
            if stage_id not in selected:
                continue
            for dependency in self.stages[stage_id].needs:
                if dependency not in selected:
                    continue
                indegree[stage_id] += 1
                children[dependency].append(stage_id)

        ready = deque(
            stage_id for stage_id in self.stages if stage_id in selected and indegree[stage_id] == 0
        )
        result: list[str] = []
        while ready:
            stage_id = ready.popleft()
            result.append(stage_id)
            for child in children[stage_id]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        if len(result) != len(selected):
            cycle_nodes = sorted(stage_id for stage_id, degree in indegree.items() if degree > 0)
            raise ConfigurationError(f"stage DAG contains a cycle: {cycle_nodes}")
        return result

    def content_digest(self) -> str:
        return digest_json(self.model_dump(mode="json"))


class StrictFormatMap(dict[str, str]):
    def __missing__(self, key: str) -> str:
        raise ConfigurationError(f"unknown command template variable: {key}")


def render_template(value: str, context: dict[str, str]) -> str:
    formatter = Formatter()
    for _, field_name, format_spec, conversion in formatter.parse(value):
        if format_spec or conversion:
            raise ConfigurationError("format specs and conversions are not allowed in templates")
        if field_name and ("." in field_name or "[" in field_name):
            raise ConfigurationError(f"nested template fields are not allowed: {field_name}")
    try:
        return value.format_map(StrictFormatMap(context))
    except (KeyError, ValueError) as exc:
        raise ConfigurationError(f"cannot render template {value!r}: {exc}") from exc


def load_run_config(path: str | Path) -> RunConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigurationError(f"cannot read run config {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError(f"run config must be a mapping: {config_path}")
    return RunConfig.model_validate(raw)


def write_run_config(path: Path, config: RunConfig) -> None:
    payload: dict[str, Any] = config.model_dump(mode="json", exclude_none=False)
    text = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
    atomic_write_text(path, text)
