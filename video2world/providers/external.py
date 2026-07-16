"""Fail-fast, shell-free execution of site-configured upstream providers.

The repository deliberately does not guess Holi-Spatial, PGSR, SAM3, or
EmbodiedGen environments.  A provider contract records the real argv and the
repositories/checkpoints it needs.  This module validates that contract and
executes it without ``shell=True``.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from string import Formatter
from typing import Annotated, Any, Literal

import yaml
from pydantic import Field, model_validator

from video2world.config import IDENTIFIER, SENSITIVE_NAME, render_template
from video2world.errors import ArtifactError, ConfigurationError, StageBlockedError
from video2world.hashing import atomic_write_json, digest_json, digest_path, sha256_file
from video2world.models import StrictModel

ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ProviderRequirement(StrictModel):
    """A repository, checkpoint, driver, or executable required by one stage."""

    path: str = Field(min_length=1)
    kind: Literal["file", "directory", "executable"]
    min_size_bytes: int = Field(default=1, ge=0)
    require_nonempty_directory: bool = True
    expected_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    description: str | None = None


class ProviderStageContract(StrictModel):
    """The exact argv and artifact roles implemented by an upstream stage."""

    provider: str = Field(min_length=1)
    command: Annotated[list[str], Field(min_length=1)]
    cwd: str = "{run_dir}"
    env: dict[str, str] = Field(default_factory=dict)
    input_roles: list[str] = Field(default_factory=list)
    output_roles: Annotated[list[str], Field(min_length=1)]
    requires: dict[str, ProviderRequirement] = Field(default_factory=dict)
    timeout_seconds: int | None = Field(default=None, gt=0)
    lineage_evidence: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_identifiers_and_environment(self) -> ProviderStageContract:
        for collection_name, values in (
            ("input_roles", self.input_roles),
            ("output_roles", self.output_roles),
            ("requirement", list(self.requires)),
        ):
            invalid = sorted(value for value in values if not IDENTIFIER.fullmatch(value))
            if invalid:
                raise ValueError(f"invalid {collection_name} identifiers: {invalid}")
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {collection_name} identifiers")
        invalid_env = sorted(key for key in self.env if not IDENTIFIER.fullmatch(key))
        if invalid_env:
            raise ValueError(f"invalid environment identifiers: {invalid_env}")
        literal_secrets = sorted(
            key
            for key, value in self.env.items()
            if SENSITIVE_NAME.search(key) and not ENV_REFERENCE.fullmatch(value)
        )
        if literal_secrets:
            raise ValueError(
                "sensitive provider environment values must be ${ENV_VAR} references: "
                f"{literal_secrets}"
            )
        if any(not argument for argument in self.command):
            raise ValueError("provider command arguments cannot be empty")
        return self


class ProviderContract(StrictModel):
    """Versioned site configuration for one or more real provider commands."""

    schema_version: Literal["1.0"]
    provider_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    scope: Literal["generic_interface", "scene_specific_verified"]
    verification_status: Literal[
        "requires_site_configuration",
        "site_preflight_passed",
        "scene_outputs_verified",
    ]
    notes: list[str] = Field(default_factory=list)
    source_repositories: dict[str, str] = Field(default_factory=dict)
    stages: dict[str, ProviderStageContract] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_stage_ids(self) -> ProviderContract:
        invalid = sorted(stage_id for stage_id in self.stages if not IDENTIFIER.fullmatch(stage_id))
        if invalid:
            raise ValueError(f"invalid provider stage identifiers: {invalid}")
        if (
            self.scope == "generic_interface"
            and self.verification_status == "scene_outputs_verified"
        ):
            raise ValueError(
                "a generic interface cannot claim scene_outputs_verified; "
                "verification belongs to a run"
            )
        return self


def load_provider_contract(path: str | Path) -> ProviderContract:
    contract_path = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigurationError(f"cannot read provider contract {contract_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError(f"provider contract must be a mapping: {contract_path}")
    return ProviderContract.model_validate(raw)


def _parse_assignments(values: list[str] | None, *, label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for assignment in values or []:
        name, separator, value = assignment.partition("=")
        if not separator or not IDENTIFIER.fullmatch(name) or not value:
            raise ConfigurationError(
                f"invalid {label} assignment {assignment!r}; expected name=value"
            )
        if name in result:
            raise ConfigurationError(f"duplicate {label} assignment: {name}")
        result[name] = value
    return result


def _expand_environment(value: str, environment: dict[str, str]) -> str:
    referenced = {match.group(1) for match in ENV_REFERENCE.finditer(value)}
    missing = sorted(referenced - environment.keys())
    if missing:
        raise ConfigurationError(f"provider contract references missing environment: {missing}")
    return ENV_REFERENCE.sub(lambda match: environment[match.group(1)], value)


def _template_fields(value: str) -> set[str]:
    fields: set[str] = set()
    for _, field_name, format_spec, conversion in Formatter().parse(value):
        if format_spec or conversion:
            raise ConfigurationError(
                "format specs and conversions are not allowed in provider templates"
            )
        if field_name:
            if "." in field_name or "[" in field_name or not IDENTIFIER.fullmatch(field_name):
                raise ConfigurationError(f"invalid provider template field: {field_name}")
            fields.add(field_name)
    return fields


def _render(value: str, context: dict[str, str], environment: dict[str, str]) -> str:
    expanded = _expand_environment(value, environment)
    _template_fields(expanded)
    return render_template(expanded, context)


def _resolve_executable(value: str, *, cwd: Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        if not candidate.is_absolute():
            candidate = cwd / candidate
        resolved = candidate.resolve()
    else:
        found = shutil.which(value)
        if found is None:
            raise ConfigurationError(f"provider executable is not on PATH: {value}")
        resolved = Path(found).resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ConfigurationError(f"provider executable is missing or not executable: {resolved}")
    return resolved


def _validate_requirement(
    name: str,
    requirement: ProviderRequirement,
    *,
    context: dict[str, str],
    environment: dict[str, str],
    cwd: Path,
) -> dict[str, Any]:
    rendered = _render(requirement.path, context, environment)
    if requirement.kind == "executable":
        path = _resolve_executable(rendered, cwd=cwd)
    else:
        path = Path(rendered).expanduser()
        if not path.is_absolute():
            path = cwd / path
        path = path.resolve()
        if requirement.kind == "file" and not path.is_file():
            raise ConfigurationError(f"provider requirement {name} is not a file: {path}")
        if requirement.kind == "directory" and not path.is_dir():
            raise ConfigurationError(f"provider requirement {name} is not a directory: {path}")

    size_bytes = path.stat().st_size if path.is_file() else None
    if size_bytes is not None and size_bytes < requirement.min_size_bytes:
        raise ConfigurationError(
            f"provider requirement {name} is too small: {path} ({size_bytes} bytes)"
        )
    if path.is_dir() and requirement.require_nonempty_directory:
        try:
            next(path.iterdir())
        except StopIteration as exc:
            raise ConfigurationError(f"provider requirement {name} is empty: {path}") from exc

    actual_sha256: str | None = None
    if requirement.expected_sha256 is not None:
        if not path.is_file():
            raise ConfigurationError(f"provider requirement {name} can hash only a file: {path}")
        actual_sha256, _ = sha256_file(path)
        if actual_sha256 != requirement.expected_sha256:
            raise ConfigurationError(
                f"provider requirement {name} sha256 mismatch: {path}; "
                f"expected {requirement.expected_sha256}, got {actual_sha256}"
            )
    return {
        "name": name,
        "kind": requirement.kind,
        "path": str(path),
        "size_bytes": size_bytes,
        "sha256_verified": actual_sha256,
        "description": requirement.description,
    }


def _contract_digest(path: Path) -> str:
    digest, _ = sha256_file(path)
    return digest


def _path_digest_payload(path: str | Path) -> dict[str, Any]:
    value = digest_path(path)
    return {
        "path": str(value.path),
        "kind": value.kind,
        "sha256": value.sha256,
        "size_bytes": value.size_bytes,
        "file_count": value.file_count,
    }


def _select_stages(contract: ProviderContract, stages: list[str] | None) -> list[str]:
    selected = stages or list(contract.stages)
    missing = sorted(set(selected) - contract.stages.keys())
    if missing:
        raise ConfigurationError(f"provider contract has no stages: {missing}")
    return selected


def preflight_contract(
    contract_path: str | Path,
    *,
    stages: list[str] | None = None,
    values: dict[str, str] | None = None,
    report_path: str | Path | None = None,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Validate static provider dependencies without launching a model command."""

    path = Path(contract_path).expanduser().resolve()
    contract = load_provider_contract(path)
    env = dict(os.environ if environment is None else environment)
    context = dict(values or {})
    selected = _select_stages(contract, stages)
    stage_reports: list[dict[str, Any]] = []

    for stage_id in selected:
        stage = contract.stages[stage_id]
        for argument in stage.command:
            expanded = _expand_environment(argument, env)
            _template_fields(expanded)
        cwd = Path(_render(stage.cwd, context, env)).expanduser().resolve()
        if not cwd.is_dir():
            raise ConfigurationError(f"provider stage {stage_id} cwd does not exist: {cwd}")
        executable_value = _render(stage.command[0], context, env)
        executable = _resolve_executable(executable_value, cwd=cwd)
        requirements = [
            _validate_requirement(
                name,
                requirement,
                context=context,
                environment=env,
                cwd=cwd,
            )
            for name, requirement in stage.requires.items()
        ]
        stage_reports.append(
            {
                "stage_id": stage_id,
                "provider": stage.provider,
                "executable": str(executable),
                "argument_count": len(stage.command),
                "input_roles": stage.input_roles,
                "output_roles": stage.output_roles,
                "requirements": requirements,
                "runtime_artifacts_validated": False,
            }
        )

    report = {
        "schema_version": 1,
        "kind": "video2world.provider_preflight",
        "status": "passed",
        "created_at": datetime.now(UTC).isoformat(),
        "contract": str(path),
        "contract_sha256": _contract_digest(path),
        "provider_id": contract.provider_id,
        "scope": contract.scope,
        "verification_status": contract.verification_status,
        "stages": stage_reports,
        "notes": [
            "Preflight validates static commands and dependencies only.",
            "Each stage validates its real input and output artifacts at execution time.",
        ],
    }
    if report_path is not None:
        atomic_write_json(Path(report_path).expanduser().resolve(), report)
    return report


def _require_exact_roles(
    configured: list[str], provided: dict[str, str], *, stage_id: str, kind: str
) -> None:
    expected = set(configured)
    actual = set(provided)
    if expected != actual:
        raise ConfigurationError(
            f"provider stage {stage_id} {kind} roles differ; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def execute_stage(
    contract_path: str | Path,
    stage_id: str,
    *,
    values: dict[str, str] | None = None,
    inputs: dict[str, str] | None = None,
    outputs: dict[str, str] | None = None,
    receipt_path: str | Path,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Execute one configured stage and verify all artifact roles."""

    path = Path(contract_path).expanduser().resolve()
    contract = load_provider_contract(path)
    if stage_id not in contract.stages:
        raise ConfigurationError(f"provider contract has no stage {stage_id!r}")
    stage = contract.stages[stage_id]
    input_paths = dict(inputs or {})
    output_paths = dict(outputs or {})
    _require_exact_roles(stage.input_roles, input_paths, stage_id=stage_id, kind="input")
    _require_exact_roles(stage.output_roles, output_paths, stage_id=stage_id, kind="output")

    context = dict(values or {})
    for role, value in input_paths.items():
        context[f"input_{role}"] = str(Path(value).expanduser().resolve())
    for role, value in output_paths.items():
        context[f"output_{role}"] = str(Path(value).expanduser().resolve())
    context["provider_stage_id"] = stage_id
    env = dict(os.environ if environment is None else environment)
    cwd = Path(_render(stage.cwd, context, env)).expanduser().resolve()
    if not cwd.is_dir():
        raise ConfigurationError(f"provider stage {stage_id} cwd does not exist: {cwd}")

    input_snapshots = {
        role: _path_digest_payload(value) for role, value in sorted(input_paths.items())
    }
    requirements = [
        _validate_requirement(
            name,
            requirement,
            context=context,
            environment=env,
            cwd=cwd,
        )
        for name, requirement in stage.requires.items()
    ]
    command = [_render(argument, context, env) for argument in stage.command]
    executable = _resolve_executable(command[0], cwd=cwd)
    command[0] = str(executable)

    provider_environment = env.copy()
    for key, value in stage.env.items():
        provider_environment[key] = _render(value, context, env)
    for value in output_paths.values():
        Path(value).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(UTC)
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=provider_environment,
        check=False,
        timeout=stage.timeout_seconds,
    )
    if completed.returncode != 0:
        raise StageBlockedError(
            f"provider stage {stage_id} exited with {completed.returncode}; "
            "inspect the enclosing Video2World stage log"
        )

    output_snapshots = {
        role: _path_digest_payload(value) for role, value in sorted(output_paths.items())
    }
    receipt = {
        "schema_version": 1,
        "kind": "video2world.provider_execution_receipt",
        "status": "completed",
        "provider_id": contract.provider_id,
        "provider_stage_id": stage_id,
        "provider": stage.provider,
        "contract": str(path),
        "contract_sha256": _contract_digest(path),
        "scope": contract.scope,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "command_sha256": digest_json(command),
        "executable": str(executable),
        "argument_count": len(command),
        "environment_keys": sorted(stage.env),
        "requirements": requirements,
        "inputs": input_snapshots,
        "outputs": output_snapshots,
        "lineage_evidence": stage.lineage_evidence,
        "security": "Full argv and environment values are intentionally omitted from this receipt.",
    }
    atomic_write_json(Path(receipt_path).expanduser().resolve(), receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="Validate static provider dependencies")
    preflight.add_argument("--contract", required=True)
    preflight.add_argument("--stage", action="append", dest="stages")
    preflight.add_argument("--value", action="append")
    preflight.add_argument("--report", required=True)

    run = subparsers.add_parser("run", help="Execute one provider stage without a shell")
    run.add_argument("--contract", required=True)
    run.add_argument("--stage", required=True)
    run.add_argument("--value", action="append")
    run.add_argument("--input", action="append")
    run.add_argument("--output", action="append")
    run.add_argument("--receipt", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        values = _parse_assignments(args.value, label="value")
        if args.command == "preflight":
            report = preflight_contract(
                args.contract,
                stages=args.stages,
                values=values,
                report_path=args.report,
            )
            print(yaml.safe_dump(report, sort_keys=False), end="")
            return 0
        receipt = execute_stage(
            args.contract,
            args.stage,
            values=values,
            inputs=_parse_assignments(args.input, label="input"),
            outputs=_parse_assignments(args.output, label="output"),
            receipt_path=args.receipt,
        )
        print(yaml.safe_dump(receipt, sort_keys=False), end="")
        return 0
    except (ArtifactError, ConfigurationError, StageBlockedError) as exc:
        print(f"provider error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
