"""Recoverable content-addressed stage orchestration."""

from __future__ import annotations

import fcntl
import json
import os
import re
import subprocess
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from video2world.adapters import get_adapter
from video2world.config import (
    RunConfig,
    StageConfig,
    load_run_config,
    render_template,
    write_run_config,
)
from video2world.errors import ArtifactError, ConfigurationError, StageBlockedError
from video2world.hashing import PathDigest, digest_json, digest_path
from video2world.state import (
    AdoptionRecord,
    ArtifactSnapshot,
    StagePlan,
    StageState,
    load_stage_state,
    state_path,
    write_stage_state,
)

RUN_CONFIG_NAME = "run.yaml"
ENV_REFERENCE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


def utc_now() -> datetime:
    return datetime.now(UTC)


def snapshot_from_digest(value: PathDigest) -> ArtifactSnapshot:
    return ArtifactSnapshot(
        path=str(value.path),
        kind=value.kind,
        sha256=value.sha256,
        size_bytes=value.size_bytes,
        file_count=value.file_count,
    )


def snapshot_path(path: str | Path) -> ArtifactSnapshot:
    return snapshot_from_digest(digest_path(path))


def initialize_run(
    run_dir: str | Path,
    *,
    video: str | Path,
    scene_id: str,
    run_id: str | None = None,
    template_config: str | Path | None = None,
) -> Path:
    output_dir = Path(run_dir).expanduser().resolve()
    video_path = Path(video).expanduser().resolve()
    digest_path(video_path)
    config_path = output_dir / RUN_CONFIG_NAME
    if config_path.exists():
        raise ConfigurationError(f"run is already initialized: {config_path}")

    if template_config:
        template = load_run_config(template_config)
    else:
        default_resource = files("video2world").joinpath("configs/default_pipeline.yaml")
        raw = yaml.safe_load(default_resource.read_text(encoding="utf-8"))
        template = RunConfig.model_validate(raw)

    effective_run_id = run_id or f"{scene_id}-{utc_now().strftime('%Y%m%dT%H%M%SZ')}"
    config = RunConfig.model_validate(
        {
            **template.model_dump(mode="json"),
            "run_id": effective_run_id,
            "scene_id": scene_id,
            "input_video": str(video_path),
            "run_dir": str(output_dir),
        }
    )
    for stage_id, stage in config.stages.items():
        get_adapter(stage.adapter).validate(stage_id, stage)

    output_dir.mkdir(parents=True, exist_ok=True)
    for relative in (".video2world/stages", ".video2world/logs", "artifacts", "bundle"):
        (output_dir / relative).mkdir(parents=True, exist_ok=True)
    write_run_config(config_path, config)
    return config_path


class PipelineOrchestrator:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.config_path = self.run_dir / RUN_CONFIG_NAME
        self.config = load_run_config(self.config_path)
        configured_run_dir = Path(self.config.run_dir).expanduser().resolve()
        if configured_run_dir != self.run_dir:
            raise ConfigurationError(
                f"run config points to {configured_run_dir}, not requested directory {self.run_dir}"
            )
        for stage_id, stage in self.config.stages.items():
            get_adapter(stage.adapter).validate(stage_id, stage)

    def _base_context(self) -> dict[str, str]:
        context = {
            **self.config.variables,
            "run_dir": str(self.run_dir),
            "run_id": self.config.run_id,
            "scene_id": self.config.scene_id,
            "input_video": str(Path(self.config.input_video).expanduser().resolve()),
        }
        for stage_id in self.config.topological_order():
            stage = self.config.stages[stage_id]
            for role, template in stage.outputs.items():
                key = f"{stage_id}_{role}"
                context[key] = str(self._resolve_run_path(render_template(template, context)))
        return context

    def _resolve_run_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.run_dir / path
        return path.resolve()

    def _context_for_stage(
        self,
        stage_id: str,
        dependency_states: dict[str, StageState],
    ) -> dict[str, str]:
        stage = self.config.stages[stage_id]
        context = self._base_context()
        for dependency, state in dependency_states.items():
            for role, artifact in state.outputs.items():
                context[f"{dependency}_{role}"] = artifact.path
        for role, template in stage.outputs.items():
            context[f"output_{role}"] = str(
                self._resolve_run_path(render_template(template, context))
            )
        for name, template in stage.inputs.items():
            context[f"input_{name}"] = str(
                self._resolve_run_path(render_template(template, context))
            )
        return context

    def _stage_config_digest(self, stage_id: str) -> str:
        stage = self.config.stages[stage_id]
        context = self._base_context()
        fingerprints = {
            name: snapshot_path(
                self._resolve_run_path(render_template(template, context))
            ).model_dump(mode="json")
            for name, template in stage.fingerprints.items()
        }
        return digest_json(
            {
                "schema_version": self.config.schema_version,
                "run_id": self.config.run_id,
                "scene_id": self.config.scene_id,
                "run_dir": str(self.run_dir),
                "variables": self.config.variables,
                "stage": stage.model_dump(mode="json"),
                "fingerprints": fingerprints,
            }
        )

    @contextmanager
    def _execution_lock(self):  # type: ignore[no-untyped-def]
        lock_path = self.run_dir / ".video2world" / "run.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise StageBlockedError(
                    f"another Video2World process is already executing this run: {self.run_dir}"
                ) from exc
            lock.seek(0)
            lock.truncate()
            lock.write(f"pid={os.getpid()} acquired_at={utc_now().isoformat()}\n")
            lock.flush()
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _snapshots_match(snapshots: dict[str, ArtifactSnapshot]) -> tuple[bool, str]:
        for role, expected in snapshots.items():
            try:
                current = snapshot_path(expected.path)
            except ArtifactError as exc:
                return False, f"{role}: {exc}"
            if current != expected:
                return False, f"{role}: content changed at {expected.path}"
        return True, "content hashes match"

    def _current_state(
        self,
        stage_id: str,
        memo: dict[str, tuple[StageState | None, str]] | None = None,
    ) -> tuple[StageState | None, str]:
        memo = memo if memo is not None else {}
        if stage_id in memo:
            return memo[stage_id]
        state = load_stage_state(self.run_dir, stage_id)
        if state is None:
            result = (None, "no stage state")
            memo[stage_id] = result
            return result
        if state.status not in {"succeeded", "adopted"}:
            result = (None, f"last state is {state.status}")
            memo[stage_id] = result
            return result
        stage = self.config.stages[stage_id]
        if state.adapter != stage.adapter:
            result = (None, "adapter changed")
            memo[stage_id] = result
            return result
        if state.config_digest != self._stage_config_digest(stage_id):
            result = (None, "stage configuration changed")
            memo[stage_id] = result
            return result

        dependency_states: dict[str, StageState] = {}
        for dependency in stage.needs:
            dependency_state, reason = self._current_state(dependency, memo)
            if dependency_state is None:
                result = (None, f"dependency {dependency} is stale: {reason}")
                memo[stage_id] = result
                return result
            dependency_states[dependency] = dependency_state

        try:
            _, explicit_inputs, input_digest = self._input_contract(stage_id, dependency_states)
        except ArtifactError as exc:
            result = (None, f"input changed or missing: {exc}")
            memo[stage_id] = result
            return result
        if explicit_inputs != state.explicit_inputs or input_digest != state.input_digest:
            result = (None, "input content hash changed")
            memo[stage_id] = result
            return result
        outputs_match, reason = self._snapshots_match(state.outputs)
        if not outputs_match:
            result = (None, reason)
            memo[stage_id] = result
            return result
        result = (state, "state and content hashes are current")
        memo[stage_id] = result
        return result

    def _dependency_states(self, stage_id: str) -> tuple[dict[str, StageState], list[str]]:
        states: dict[str, StageState] = {}
        missing: list[str] = []
        memo: dict[str, tuple[StageState | None, str]] = {}
        for dependency in self.config.stages[stage_id].needs:
            state, reason = self._current_state(dependency, memo)
            if state is None:
                missing.append(f"{dependency} ({reason})")
            else:
                states[dependency] = state
        return states, missing

    def _input_contract(
        self,
        stage_id: str,
        dependency_states: dict[str, StageState],
    ) -> tuple[dict[str, str], dict[str, ArtifactSnapshot], str]:
        stage = self.config.stages[stage_id]
        context = self._context_for_stage(stage_id, dependency_states)
        input_paths = {
            name: str(self._resolve_run_path(render_template(template, context)))
            for name, template in stage.inputs.items()
        }
        explicit_inputs = {name: snapshot_path(path) for name, path in input_paths.items()}
        dependency_outputs = {
            dependency: state.outputs for dependency, state in dependency_states.items()
        }
        input_digest = digest_json(
            {
                "config_digest": self._stage_config_digest(stage_id),
                "explicit_inputs": {
                    name: value.model_dump(mode="json") for name, value in explicit_inputs.items()
                },
                "dependency_outputs": {
                    dependency: {
                        role: value.model_dump(mode="json") for role, value in outputs.items()
                    }
                    for dependency, outputs in dependency_outputs.items()
                },
            }
        )
        return input_paths, explicit_inputs, input_digest

    def _rendered_outputs(self, stage_id: str, context: dict[str, str]) -> dict[str, str]:
        return {
            role: str(self._resolve_run_path(render_template(template, context)))
            for role, template in self.config.stages[stage_id].outputs.items()
        }

    def _recovery_actions_from_outputs(self, outputs: dict[str, str]) -> list[dict[str, Any]]:
        for path in outputs.values():
            try:
                payload = json.loads(Path(path).read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            actions = payload.get("recovery_actions")
            if not isinstance(actions, list):
                continue
            kept = [item for item in actions if isinstance(item, dict)]
            if kept:
                return kept
        return []

    def plan(self, targets: list[str] | None = None) -> list[StagePlan]:
        result: list[StagePlan] = []
        memo: dict[str, tuple[StageState | None, str]] = {}
        for stage_id in self.config.topological_order(targets):
            stage = self.config.stages[stage_id]
            adapter = get_adapter(stage.adapter)
            current, current_reason = self._current_state(stage_id, memo)
            dependency_states, missing_dependencies = self._dependency_states(stage_id)
            context = self._context_for_stage(stage_id, dependency_states)
            outputs = self._rendered_outputs(stage_id, context)
            command = adapter.render_command(stage, context)
            previous_state = load_stage_state(self.run_dir, stage_id)

            input_digest: str | None = None
            if not missing_dependencies:
                with suppress(ArtifactError):
                    _, _, input_digest = self._input_contract(stage_id, dependency_states)

            if not stage.enabled:
                action, reason = "disabled", "stage is disabled in run.yaml"
            elif current is not None:
                action, reason = "cached", current_reason
            elif missing_dependencies:
                action = "waiting"
                reason = "dependencies are not current: " + ", ".join(missing_dependencies)
            elif command is None:
                action = "adopt_or_configure"
                reason = "no command configured; configure argv or adopt existing real outputs"
                if load_stage_state(self.run_dir, stage_id) is not None:
                    reason = f"{current_reason}; {reason}"
            elif load_stage_state(self.run_dir, stage_id) is not None:
                action, reason = "stale", current_reason
            else:
                action, reason = "run", "configured command has no current output state"
            result.append(
                StagePlan(
                    stage_id=stage_id,
                    adapter=stage.adapter,
                    action=action,
                    reason=reason,
                    input_digest=input_digest,
                    command=command,
                    outputs=outputs,
                    recovery_actions=(
                        self._recovery_actions_from_outputs(outputs)
                        if previous_state is not None and previous_state.status == "failed"
                        else []
                    ),
                )
            )
        return result

    def adopt(
        self,
        stage_id: str,
        outputs: dict[str, str | Path],
        *,
        source_run_id: str,
        source_root: str | Path | None = None,
        source_repository: str | None = None,
        source_commit: str | None = None,
        source_command: str | None = None,
        note: str | None = None,
    ) -> StageState:
        if stage_id not in self.config.stages:
            raise ConfigurationError(f"unknown stage: {stage_id}")
        stage = self.config.stages[stage_id]
        if not stage.enabled:
            raise StageBlockedError(f"stage {stage_id} is disabled")
        expected_roles = set(stage.outputs)
        provided_roles = set(outputs)
        if expected_roles != provided_roles:
            missing = sorted(expected_roles - provided_roles)
            extra = sorted(provided_roles - expected_roles)
            raise ArtifactError(f"adoption output roles differ; missing={missing}, extra={extra}")
        dependency_states, missing_dependencies = self._dependency_states(stage_id)
        if missing_dependencies:
            raise StageBlockedError(
                "cannot adopt before dependencies are current: " + ", ".join(missing_dependencies)
            )
        _, explicit_inputs, input_digest = self._input_contract(stage_id, dependency_states)
        adopted_outputs = {
            role: snapshot_path(Path(path).expanduser().resolve()) for role, path in outputs.items()
        }
        get_adapter(stage.adapter).validate_outputs(adopted_outputs)
        output_paths = [Path(item.path) for item in adopted_outputs.values()]
        inferred_root = Path(os.path.commonpath([str(path.parent) for path in output_paths]))
        effective_root = Path(source_root).expanduser().resolve() if source_root else inferred_root
        if not effective_root.is_dir():
            raise ArtifactError(f"adoption source root is not a directory: {effective_root}")
        for output_path in output_paths:
            if not output_path.is_relative_to(effective_root):
                raise ArtifactError(
                    f"adopted output {output_path} is outside source root {effective_root}"
                )
        previous = load_stage_state(self.run_dir, stage_id)
        now = utc_now()
        state = StageState(
            stage_id=stage_id,
            adapter=stage.adapter,
            status="adopted",
            mode="adopted",
            attempt=(previous.attempt + 1) if previous else 1,
            input_digest=input_digest,
            config_digest=self._stage_config_digest(stage_id),
            dependency_outputs={
                dependency: value.outputs for dependency, value in dependency_states.items()
            },
            explicit_inputs=explicit_inputs,
            outputs=adopted_outputs,
            started_at=now,
            finished_at=now,
            adoption=AdoptionRecord(
                source_run_id=source_run_id,
                source_root=str(effective_root),
                source_repository=source_repository,
                source_commit=source_commit,
                source_command=source_command,
                note=note,
                adopted_at=now,
            ),
        )
        write_stage_state(self.run_dir, state)
        return state

    def _resolve_environment(self, stage: StageConfig, context: dict[str, str]) -> dict[str, str]:
        environment = os.environ.copy()
        for key, template in stage.env.items():
            match = ENV_REFERENCE.fullmatch(template)
            if match:
                source_key = match.group(1)
                if source_key not in os.environ:
                    raise ConfigurationError(
                        f"stage environment {key} references missing variable {source_key}"
                    )
                environment[key] = os.environ[source_key]
            else:
                environment[key] = render_template(template, context)
        return environment

    def run(self, targets: list[str] | None = None) -> list[dict[str, Any]]:
        with self._execution_lock():
            return self._run_unlocked(targets)

    def _run_unlocked(self, targets: list[str] | None = None) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        for stage_id in self.config.topological_order(targets):
            stage = self.config.stages[stage_id]
            current, reason = self._current_state(stage_id)
            if current is not None:
                summaries.append({"stage_id": stage_id, "action": "cached", "reason": reason})
                continue
            if not stage.enabled:
                raise StageBlockedError(f"stage {stage_id} is disabled")
            dependency_states, missing_dependencies = self._dependency_states(stage_id)
            if missing_dependencies:
                raise StageBlockedError(
                    f"stage {stage_id} dependencies are not current: "
                    + ", ".join(missing_dependencies)
                )
            context = self._context_for_stage(stage_id, dependency_states)
            adapter = get_adapter(stage.adapter)
            command = adapter.render_command(stage, context)
            if command is None:
                raise StageBlockedError(
                    f"stage {stage_id} has no command; configure argv or use adopt-existing"
                )
            _, explicit_inputs, input_digest = self._input_contract(stage_id, dependency_states)
            output_paths = self._rendered_outputs(stage_id, context)
            for path in output_paths.values():
                Path(path).parent.mkdir(parents=True, exist_ok=True)
            cwd = self._resolve_run_path(render_template(stage.cwd, context))
            if not cwd.is_dir():
                raise ConfigurationError(f"stage {stage_id} cwd does not exist: {cwd}")
            environment = self._resolve_environment(stage, context)
            log_path = self.run_dir / ".video2world" / "logs" / f"{stage_id}.log"
            previous = load_stage_state(self.run_dir, stage_id)
            attempt = (previous.attempt + 1) if previous else 1
            started_at = utc_now()
            running = StageState(
                stage_id=stage_id,
                adapter=stage.adapter,
                status="running",
                mode="executed",
                attempt=attempt,
                input_digest=input_digest,
                config_digest=self._stage_config_digest(stage_id),
                dependency_outputs={
                    dependency: value.outputs for dependency, value in dependency_states.items()
                },
                explicit_inputs=explicit_inputs,
                command=command,
                environment_keys=sorted(stage.env),
                started_at=started_at,
                log_path=str(log_path),
            )
            write_stage_state(self.run_dir, running)
            try:
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(
                        f"\n[{started_at.isoformat()}] attempt={attempt} command={command!r}\n"
                    )
                    completed = subprocess.run(
                        command,
                        cwd=cwd,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=stage.timeout_seconds,
                        check=False,
                    )
                if completed.returncode != 0:
                    raise StageBlockedError(
                        f"stage {stage_id} exited with {completed.returncode}; see {log_path}"
                    )
                verified_outputs = {
                    role: snapshot_path(path) for role, path in output_paths.items()
                }
                adapter.validate_outputs(verified_outputs)
                succeeded = running.model_copy(
                    update={
                        "status": "succeeded",
                        "outputs": verified_outputs,
                        "finished_at": utc_now(),
                    }
                )
                write_stage_state(self.run_dir, succeeded)
                summaries.append(
                    {
                        "stage_id": stage_id,
                        "action": "executed",
                        "attempt": attempt,
                        "outputs": {
                            role: value.model_dump(mode="json")
                            for role, value in verified_outputs.items()
                        },
                    }
                )
            except Exception as exc:
                failed = running.model_copy(
                    update={"status": "failed", "finished_at": utc_now(), "error": str(exc)}
                )
                write_stage_state(self.run_dir, failed)
                raise
        return summaries

    def validate_run(self) -> dict[str, Any]:
        stages: list[dict[str, Any]] = []
        valid = True
        for stage_id in self.config.topological_order():
            state, reason = self._current_state(stage_id)
            status = "current" if state is not None else "incomplete"
            if self.config.stages[stage_id].enabled and state is None:
                valid = False
            stages.append({"stage_id": stage_id, "status": status, "reason": reason})
        return {
            "valid": valid,
            "run_id": self.config.run_id,
            "scene_id": self.config.scene_id,
            "config_digest": self.config.content_digest(),
            "stages": stages,
        }


def parse_output_assignments(values: list[str]) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ConfigurationError(f"output must use role=/absolute/path syntax: {value!r}")
        role, path = value.split("=", 1)
        if not role or not path:
            raise ConfigurationError(f"invalid output assignment: {value!r}")
        if role in outputs:
            raise ConfigurationError(f"duplicate output role: {role}")
        outputs[role] = path
    return outputs


def stage_state_file(run_dir: str | Path, stage_id: str) -> Path:
    return state_path(Path(run_dir).expanduser().resolve(), stage_id)
