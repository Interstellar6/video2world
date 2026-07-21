"""Bind and execute the canonical pipeline at a concrete deployment site.

The packaged pipeline describes ownership and artifact roles.  This module binds
every stage to either a real provider command or an existing artifact root.  It
never manufactures provider outputs and never treats preflight as stage success.
"""

from __future__ import annotations

import fcntl
import os
import re
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator

from video2world.adapters import get_adapter
from video2world.config import (
    IDENTIFIER,
    SENSITIVE_NAME,
    RunConfig,
    StageConfig,
    write_run_config,
)
from video2world.errors import ArtifactError, ConfigurationError, StageBlockedError
from video2world.hashing import atomic_write_json, digest_path, sha256_file
from video2world.models import Sha256, StrictModel
from video2world.orchestrator import (
    PipelineOrchestrator,
    initialize_run,
    snapshot_from_digest,
)
from video2world.providers.external import (
    ProviderContract,
    load_provider_contract,
    preflight_contract,
)
from video2world.state import load_stage_state, state_path

SITE_BINDING_NAME = "site-binding.json"
SITE_PREFLIGHT_NAME = "site-preflight.json"
SITE_RUN_RECEIPT_NAME = "site-run-receipt.json"
GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")


# These are the content-bearing inputs of the canonical twelve-stage graph.  A
# site provider must consume exactly these roles; implicit directory conventions
# are intentionally not accepted.
CANONICAL_STAGE_INPUTS: dict[str, dict[str, str]] = {
    "ingest": {"video": "{input_video}"},
    "inventory": {
        "frames_manifest": "{ingest_frames_manifest}",
        "cameras": "{ingest_cameras}",
    },
    "da3": {
        "frames_manifest": "{ingest_frames_manifest}",
        "cameras": "{ingest_cameras}",
    },
    "pgsr": {
        "frames_manifest": "{ingest_frames_manifest}",
        "cameras": "{ingest_cameras}",
        "depth_manifest": "{da3_depth_manifest}",
        "point_prior": "{da3_point_prior}",
    },
    "sam3": {
        "frames_manifest": "{ingest_frames_manifest}",
        "cameras": "{ingest_cameras}",
        "scene_inventory": "{inventory_scene_inventory}",
        "occlusion_graph": "{inventory_occlusion_graph}",
    },
    "fusion": {
        "cameras": "{ingest_cameras}",
        "scene_gaussian": "{pgsr_scene_gaussian}",
        "scene_mesh": "{pgsr_scene_mesh}",
        "masks_manifest": "{sam3_masks_manifest}",
        "captions_manifest": "{sam3_captions_manifest}",
    },
    "cognition": {
        "scene_inventory": "{inventory_scene_inventory}",
        "occlusion_graph": "{inventory_occlusion_graph}",
        "masks_manifest": "{sam3_masks_manifest}",
        "captions_manifest": "{sam3_captions_manifest}",
        "object_clouds_manifest": "{fusion_object_clouds_manifest}",
        "semantic_gaussian": "{fusion_semantic_gaussian}",
    },
    "completion_plan": {
        "scene_inventory": "{inventory_scene_inventory}",
        "occlusion_graph": "{inventory_occlusion_graph}",
    },
    "layered_completion": {
        "frames_manifest": "{ingest_frames_manifest}",
        "cameras": "{ingest_cameras}",
        "layered_completion_plan": "{completion_plan_layered_completion_plan}",
        "scene_gaussian": "{pgsr_scene_gaussian}",
        "scene_mesh": "{pgsr_scene_mesh}",
        "masks_manifest": "{sam3_masks_manifest}",
        "captions_manifest": "{sam3_captions_manifest}",
        "object_clouds_manifest": "{fusion_object_clouds_manifest}",
        "semantic_gaussian": "{fusion_semantic_gaussian}",
        "object_facts": "{cognition_object_facts}",
    },
    "placement": {
        "scene_gaussian": "{layered_completion_clean_scene_gaussian}",
        "scene_mesh": "{layered_completion_clean_scene_mesh}",
        "object_clouds_manifest": "{fusion_object_clouds_manifest}",
        "semantic_gaussian": "{fusion_semantic_gaussian}",
        "completed_object_assets_manifest": (
            "{layered_completion_completed_object_assets_manifest}"
        ),
    },
    "bundle": {
        "clean_scene_gaussian": "{layered_completion_clean_scene_gaussian}",
        "clean_scene_mesh": "{layered_completion_clean_scene_mesh}",
        "completed_object_assets_manifest": (
            "{layered_completion_completed_object_assets_manifest}"
        ),
        "object_facts": "{cognition_object_facts}",
        "aligned_objects_manifest": "{placement_aligned_objects_manifest}",
        "collision_manifest": "{placement_collision_manifest}",
    },
    "web": {"world_manifest": "{bundle_world_manifest}"},
}


def _safe_relative_path(value: str, *, label: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"{label} must be a non-escaping relative path: {value!r}")
    return path


class SitePathBinding(StrictModel):
    root_kind: Literal["checkout", "artifact"]
    root: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    relative_path: str = "."
    expected_kind: Literal["file", "directory", "executable"] = "directory"
    expected_sha256: Sha256 | None = None
    expected_git_commit: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")

    @model_validator(mode="after")
    def validate_reference(self) -> SitePathBinding:
        if self.relative_path != ".":
            _safe_relative_path(self.relative_path, label="provider binding relative_path")
        if self.expected_sha256 and self.expected_kind == "directory":
            raise ValueError("expected_sha256 is supported only for files and executables")
        if self.expected_git_commit and self.expected_kind != "directory":
            raise ValueError("expected_git_commit requires a directory checkout")
        if self.expected_git_commit and self.root_kind != "checkout":
            raise ValueError("expected_git_commit requires root_kind='checkout'")
        return self


class SiteStageProfile(StrictModel):
    mode: Literal["execute", "adopt"]
    provider_stage: str | None = Field(default=None, pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    artifact_root: str | None = Field(default=None, pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    outputs: dict[str, str] = Field(default_factory=dict)
    expected_inputs: dict[str, Sha256] = Field(default_factory=dict)
    source_run_id: str | None = None
    source_repository: str | None = None
    source_commit: str | None = None
    source_command: str | None = None
    note: str | None = None

    @model_validator(mode="after")
    def validate_mode(self) -> SiteStageProfile:
        if self.mode == "execute":
            if not self.provider_stage:
                raise ValueError("execute mode requires provider_stage")
            if any(
                (
                    self.artifact_root,
                    self.outputs,
                    self.expected_inputs,
                    self.source_run_id,
                    self.source_repository,
                    self.source_commit,
                    self.source_command,
                    self.note,
                )
            ):
                raise ValueError("execute mode cannot declare adoption fields")
        else:
            if self.provider_stage:
                raise ValueError("adopt mode cannot declare provider_stage")
            if not self.artifact_root or not self.source_run_id or not self.outputs:
                raise ValueError(
                    "adopt mode requires artifact_root, source_run_id, and role outputs"
                )
            for role, relative_path in self.outputs.items():
                if not IDENTIFIER.fullmatch(role):
                    raise ValueError(f"invalid adoption output role: {role!r}")
                _safe_relative_path(relative_path, label=f"adoption output {role}")
            invalid_inputs = sorted(
                role for role in self.expected_inputs if not IDENTIFIER.fullmatch(role)
            )
            if invalid_inputs:
                raise ValueError(f"invalid adoption input roles: {invalid_inputs}")
        return self


class SiteProfile(StrictModel):
    schema_version: Literal["1.0"]
    profile_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    description: str = Field(min_length=1)
    provider_environment: dict[str, SitePathBinding] = Field(default_factory=dict)
    stages: dict[str, SiteStageProfile] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_environment(self) -> SiteProfile:
        invalid = sorted(
            name for name in self.provider_environment if not IDENTIFIER.fullmatch(name)
        )
        if invalid:
            raise ValueError(f"invalid provider environment names: {invalid}")
        sensitive = sorted(
            name for name in self.provider_environment if SENSITIVE_NAME.search(name)
        )
        if sensitive:
            raise ValueError(
                "site root bindings cannot carry secrets; pass secrets in the process environment: "
                f"{sensitive}"
            )
        return self


class BoundRoot(StrictModel):
    kind: Literal["checkout", "artifact"]
    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    path: str = Field(min_length=1)


class BoundAdoption(StrictModel):
    source_root: str
    source_run_id: str
    outputs: dict[str, str]
    expected_inputs: dict[str, Sha256] = Field(default_factory=dict)
    source_repository: str | None = None
    source_commit: str | None = None
    source_command: str | None = None
    note: str | None = None


class BoundSiteStage(StrictModel):
    mode: Literal["execute", "adopt"]
    provider_stage: str | None = None
    adoption: BoundAdoption | None = None

    @model_validator(mode="after")
    def validate_mode(self) -> BoundSiteStage:
        if self.mode == "execute" and (not self.provider_stage or self.adoption is not None):
            raise ValueError("bound execute stage requires only provider_stage")
        if self.mode == "adopt" and (self.provider_stage is not None or self.adoption is None):
            raise ValueError("bound adopt stage requires only adoption evidence")
        return self


class BoundSiteRun(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["video2world.site_binding"] = "video2world.site_binding"
    profile_id: str
    profile_path: str
    profile_sha256: Sha256
    provider_contract: str
    provider_contract_sha256: Sha256
    driver_python: str
    roots: list[BoundRoot]
    provider_environment: dict[str, str]
    stages: dict[str, BoundSiteStage]


def load_site_profile(path: str | Path) -> SiteProfile:
    profile_path = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigurationError(f"cannot read site profile {profile_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError(f"site profile must be a mapping: {profile_path}")
    return SiteProfile.model_validate(raw)


def parse_root_assignments(values: list[str] | None, *, label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for assignment in values or []:
        name, separator, raw_path = assignment.partition("=")
        if not separator or not IDENTIFIER.fullmatch(name) or not raw_path:
            raise ConfigurationError(
                f"invalid {label} assignment {assignment!r}; expected name=path"
            )
        if name in result:
            raise ConfigurationError(f"duplicate {label} root: {name}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_dir():
            raise ConfigurationError(f"{label} root {name!r} is not a directory: {path}")
        result[name] = path
    return result


def _load_default_pipeline() -> RunConfig:
    resource = files("video2world").joinpath("configs/default_pipeline.yaml")
    return RunConfig.model_validate(yaml.safe_load(resource.read_text(encoding="utf-8")))


def _validate_profile_stage_coverage(profile: SiteProfile, template: RunConfig) -> None:
    expected = set(template.stages)
    actual = set(profile.stages)
    if actual != expected:
        raise ConfigurationError(
            "site profile must bind every canonical stage exactly once; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    if set(CANONICAL_STAGE_INPUTS) != expected:
        raise ConfigurationError("internal canonical input map differs from packaged pipeline")


def _resolve_binding(
    binding: SitePathBinding,
    *,
    checkouts: dict[str, Path],
    artifacts: dict[str, Path],
) -> Path:
    roots = checkouts if binding.root_kind == "checkout" else artifacts
    if binding.root not in roots:
        raise ConfigurationError(
            f"missing explicit {binding.root_kind} root {binding.root!r} required by site profile"
        )
    root = roots[binding.root]
    result = root if binding.relative_path == "." else (root / binding.relative_path).resolve()
    if not result.is_relative_to(root):
        raise ConfigurationError(f"site binding escapes root {root}: {result}")
    if binding.expected_kind == "directory" and not result.is_dir():
        raise ConfigurationError(f"site binding is not a directory: {result}")
    if binding.expected_kind == "file" and not result.is_file():
        raise ConfigurationError(f"site binding is not a file: {result}")
    if binding.expected_kind == "executable" and (
        not result.is_file() or not os.access(result, os.X_OK)
    ):
        raise ConfigurationError(f"site binding is not executable: {result}")
    if binding.expected_sha256:
        actual, _ = sha256_file(result)
        if actual != binding.expected_sha256:
            raise ConfigurationError(
                f"site binding SHA-256 mismatch for {result}; "
                f"expected {binding.expected_sha256}, got {actual}"
            )
    if binding.expected_git_commit:
        completed = subprocess.run(
            ["git", "-C", str(result), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        actual = completed.stdout.strip().lower()
        if completed.returncode != 0 or not GIT_COMMIT.fullmatch(actual):
            raise ConfigurationError(f"cannot read checkout commit at {result}")
        if actual != binding.expected_git_commit:
            raise ConfigurationError(
                f"checkout commit mismatch at {result}; "
                f"expected {binding.expected_git_commit}, got {actual}"
            )
    return result


def _contract_roles(
    contract: ProviderContract,
    *,
    canonical_stage_id: str,
    provider_stage_id: str,
    output_roles: set[str],
) -> None:
    if provider_stage_id not in contract.stages:
        raise ConfigurationError(
            f"provider contract has no stage {provider_stage_id!r} for {canonical_stage_id}"
        )
    provider_stage = contract.stages[provider_stage_id]
    expected_inputs = set(CANONICAL_STAGE_INPUTS[canonical_stage_id])
    if set(provider_stage.input_roles) != expected_inputs:
        raise ConfigurationError(
            f"provider input roles differ for {canonical_stage_id}; "
            f"missing={sorted(expected_inputs - set(provider_stage.input_roles))}, "
            f"extra={sorted(set(provider_stage.input_roles) - expected_inputs)}"
        )
    if set(provider_stage.output_roles) != output_roles:
        raise ConfigurationError(
            f"provider output roles differ for {canonical_stage_id}; "
            f"missing={sorted(output_roles - set(provider_stage.output_roles))}, "
            f"extra={sorted(set(provider_stage.output_roles) - output_roles)}"
        )


def _provider_command(
    stage_id: str,
    provider_stage_id: str,
    *,
    driver_python: str,
    output_roles: set[str],
) -> list[str]:
    command = [
        driver_python,
        "-m",
        "video2world.providers.external",
        "run",
        "--contract",
        "{provider_contract}",
        "--stage",
        provider_stage_id,
        "--value",
        "run_dir={run_dir}",
        "--value",
        "run_id={run_id}",
        "--value",
        "scene_id={scene_id}",
        "--value",
        "input_video={input_video}",
    ]
    for role in CANONICAL_STAGE_INPUTS[stage_id]:
        command.extend(["--input", f"{role}={{input_{role}}}"])
    for role in sorted(output_roles):
        command.extend(["--output", f"{role}={{output_{role}}}"])
    command.extend(["--receipt", "{output_provider_receipt}"])
    return command


def _adoption_guard_command(stage_id: str, *, driver_python: str) -> list[str]:
    return [
        driver_python,
        "-m",
        "video2world.site_pipeline",
        "adoption-guard",
        "--stage",
        stage_id,
    ]


def _bound_template(
    *,
    profile: SiteProfile,
    contract: ProviderContract,
    profile_path: Path,
    provider_contract_path: Path,
    run_dir: Path,
    driver_python: Path,
    provider_environment: dict[str, str],
) -> RunConfig:
    template = _load_default_pipeline()
    _validate_profile_stage_coverage(profile, template)
    stages: dict[str, StageConfig] = {}
    binding_path = run_dir / ".video2world" / SITE_BINDING_NAME
    for stage_id, base in template.stages.items():
        site_stage = profile.stages[stage_id]
        adapter = get_adapter(base.adapter)
        required_outputs = set(adapter.required_outputs)
        adoption_required_outputs = set(required_outputs)
        if base.adapter == "layered_completion":
            adoption_required_outputs.add("provider_receipt")
        outputs = {
            role: path
            for role, path in base.outputs.items()
            if role in adoption_required_outputs
        }
        inputs = CANONICAL_STAGE_INPUTS[stage_id]
        fingerprints = {"site_binding": str(binding_path)}
        if site_stage.mode == "execute":
            assert site_stage.provider_stage is not None
            _contract_roles(
                contract,
                canonical_stage_id=stage_id,
                provider_stage_id=site_stage.provider_stage,
                output_roles=required_outputs,
            )
            outputs["provider_receipt"] = f"{{run_dir}}/artifacts/{stage_id}/provider-receipt.json"
            fingerprints["provider_contract"] = str(provider_contract_path)
            provider_timeout = contract.stages[site_stage.provider_stage].timeout_seconds
            timeout = provider_timeout + 60 if provider_timeout else base.timeout_seconds
            command = _provider_command(
                stage_id,
                site_stage.provider_stage,
                driver_python=str(driver_python),
                output_roles=required_outputs,
            )
            environment = provider_environment
        else:
            provided_outputs = set(site_stage.outputs)
            if provided_outputs != adoption_required_outputs:
                raise ConfigurationError(
                    f"adoption output roles differ for {stage_id}; "
                    f"missing={sorted(adoption_required_outputs - provided_outputs)}, "
                    f"extra={sorted(provided_outputs - adoption_required_outputs)}"
                )
            timeout = base.timeout_seconds
            command = _adoption_guard_command(stage_id, driver_python=str(driver_python))
            environment = {}
        stages[stage_id] = base.model_copy(
            update={
                "command": command,
                "inputs": inputs,
                "outputs": outputs,
                "fingerprints": fingerprints,
                "env": environment,
                "timeout_seconds": timeout,
            }
        )
    return template.model_copy(
        update={
            "variables": {
                "provider_contract": str(provider_contract_path),
                "site_profile": str(profile_path),
                "site_binding": str(binding_path),
            },
            "stages": stages,
        }
    )


def create_site_run(
    run_dir: str | Path,
    *,
    video: str | Path,
    scene_id: str,
    profile_path: str | Path,
    provider_contract_path: str | Path,
    checkout_roots: dict[str, Path],
    artifact_roots: dict[str, Path],
    run_id: str | None = None,
    driver_python: str | Path | None = None,
) -> Path:
    """Create a canonical run whose every stage has an explicit site binding."""

    output_dir = Path(run_dir).expanduser().resolve()
    profile_file = Path(profile_path).expanduser().resolve()
    contract_file = Path(provider_contract_path).expanduser().resolve()
    profile = load_site_profile(profile_file)
    contract = load_provider_contract(contract_file)
    # Do not resolve a venv launcher symlink to the base interpreter; the bound
    # driver must retain the environment in which Video2World is installed.
    python = Path(driver_python or sys.executable).expanduser().absolute()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ConfigurationError(f"site driver Python is not executable: {python}")

    template = _load_default_pipeline()
    _validate_profile_stage_coverage(profile, template)
    referenced_checkouts = {
        binding.root
        for binding in profile.provider_environment.values()
        if binding.root_kind == "checkout"
    }
    referenced_artifacts = {
        binding.root
        for binding in profile.provider_environment.values()
        if binding.root_kind == "artifact"
    }
    referenced_artifacts.update(
        stage.artifact_root
        for stage in profile.stages.values()
        if stage.mode == "adopt" and stage.artifact_root
    )
    extra_checkouts = sorted(set(checkout_roots) - referenced_checkouts)
    extra_artifacts = sorted(set(artifact_roots) - referenced_artifacts)
    if extra_checkouts or extra_artifacts:
        raise ConfigurationError(
            "unreferenced site roots usually indicate a typo; "
            f"checkout={extra_checkouts}, artifact={extra_artifacts}"
        )
    missing_checkouts = sorted(referenced_checkouts - checkout_roots.keys())
    missing_artifacts = sorted(referenced_artifacts - artifact_roots.keys())
    if missing_checkouts or missing_artifacts:
        raise ConfigurationError(
            f"missing site roots; checkout={missing_checkouts}, artifact={missing_artifacts}"
        )

    provider_environment = {
        name: str(
            _resolve_binding(
                binding,
                checkouts=checkout_roots,
                artifacts=artifact_roots,
            )
        )
        for name, binding in profile.provider_environment.items()
    }
    bound_stages: dict[str, BoundSiteStage] = {}
    for stage_id, site_stage in profile.stages.items():
        if site_stage.mode == "execute":
            bound_stages[stage_id] = BoundSiteStage(
                mode="execute", provider_stage=site_stage.provider_stage
            )
            continue
        assert site_stage.artifact_root is not None
        root = artifact_roots[site_stage.artifact_root]
        outputs = {
            role: str((root / relative).resolve()) for role, relative in site_stage.outputs.items()
        }
        if any(not Path(value).is_relative_to(root) for value in outputs.values()):
            raise ConfigurationError(f"adoption paths escape artifact root for {stage_id}")
        bound_stages[stage_id] = BoundSiteStage(
            mode="adopt",
            adoption=BoundAdoption(
                source_root=str(root),
                source_run_id=site_stage.source_run_id or "",
                outputs=outputs,
                expected_inputs=site_stage.expected_inputs,
                source_repository=site_stage.source_repository,
                source_commit=site_stage.source_commit,
                source_command=site_stage.source_command,
                note=site_stage.note,
            ),
        )

    generated = _bound_template(
        profile=profile,
        contract=contract,
        profile_path=profile_file,
        provider_contract_path=contract_file,
        run_dir=output_dir,
        driver_python=python,
        provider_environment=provider_environment,
    )
    with tempfile.TemporaryDirectory(prefix="video2world-site-") as temporary:
        template_path = Path(temporary) / "bound-template.yaml"
        write_run_config(template_path, generated)
        config_path = initialize_run(
            output_dir,
            video=video,
            scene_id=scene_id,
            run_id=run_id,
            template_config=template_path,
        )

    binding = BoundSiteRun(
        profile_id=profile.profile_id,
        profile_path=str(profile_file),
        profile_sha256=sha256_file(profile_file)[0],
        provider_contract=str(contract_file),
        provider_contract_sha256=sha256_file(contract_file)[0],
        driver_python=str(python),
        roots=[
            *(
                BoundRoot(kind="checkout", name=name, path=str(path))
                for name, path in sorted(checkout_roots.items())
            ),
            *(
                BoundRoot(kind="artifact", name=name, path=str(path))
                for name, path in sorted(artifact_roots.items())
            ),
        ],
        provider_environment=provider_environment,
        stages=bound_stages,
    )
    atomic_write_json(
        output_dir / ".video2world" / SITE_BINDING_NAME,
        binding.model_dump(mode="json"),
    )
    # Force construction once all fingerprints exist.
    PipelineOrchestrator(output_dir)
    return config_path


def load_site_binding(run_dir: str | Path) -> BoundSiteRun:
    path = Path(run_dir).expanduser().resolve() / ".video2world" / SITE_BINDING_NAME
    try:
        return BoundSiteRun.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigurationError(f"cannot read site binding {path}: {exc}") from exc


def _verify_binding_identity(binding: BoundSiteRun) -> None:
    for label, path_string, expected in (
        ("site profile", binding.profile_path, binding.profile_sha256),
        ("provider contract", binding.provider_contract, binding.provider_contract_sha256),
    ):
        path = Path(path_string)
        actual, _ = sha256_file(path)
        if actual != expected:
            raise ConfigurationError(
                f"{label} changed after site initialization: {path}; reinitialize explicitly"
            )
    for root in binding.roots:
        if not Path(root.path).is_dir():
            raise ConfigurationError(f"bound {root.kind} root is unavailable: {root.path}")
    profile = load_site_profile(binding.profile_path)
    checkouts = {root.name: Path(root.path) for root in binding.roots if root.kind == "checkout"}
    artifacts = {root.name: Path(root.path) for root in binding.roots if root.kind == "artifact"}
    resolved_environment = {
        name: str(_resolve_binding(binding_item, checkouts=checkouts, artifacts=artifacts))
        for name, binding_item in profile.provider_environment.items()
    }
    if resolved_environment != binding.provider_environment:
        raise ConfigurationError("resolved provider environment differs from the site binding")


def _selected_stages(config: RunConfig, targets: list[str] | None) -> list[str]:
    return config.topological_order(targets)


def _validate_adoption_inputs(
    orchestrator: PipelineOrchestrator,
    stage_id: str,
    adoption: BoundAdoption,
    *,
    allow_deferred: bool,
) -> str:
    expected_roles = set(CANONICAL_STAGE_INPUTS[stage_id])
    provided_roles = set(adoption.expected_inputs)
    if provided_roles != expected_roles:
        raise ConfigurationError(
            f"adoption input hash roles differ for {stage_id}; "
            f"missing={sorted(expected_roles - provided_roles)}, "
            f"extra={sorted(provided_roles - expected_roles)}"
        )
    dependency_states, missing_dependencies = orchestrator._dependency_states(stage_id)
    if missing_dependencies:
        if allow_deferred:
            return "deferred_until_dependencies_are_current"
        raise StageBlockedError(
            f"cannot verify adoption inputs for {stage_id}: " + ", ".join(missing_dependencies)
        )
    _, explicit_inputs, _ = orchestrator._input_contract(stage_id, dependency_states)
    for role, snapshot in explicit_inputs.items():
        expected = adoption.expected_inputs[role]
        if snapshot.sha256 != expected:
            raise ArtifactError(
                f"adoption input {stage_id}.{role} SHA-256 mismatch; "
                f"expected {expected}, got {snapshot.sha256} at {snapshot.path}"
            )
    return "verified"


def preflight_site_run(
    run_dir: str | Path,
    *,
    targets: list[str] | None = None,
) -> dict[str, Any]:
    """Audit every selected execute/adopt binding without launching models."""

    root = Path(run_dir).expanduser().resolve()
    orchestrator = PipelineOrchestrator(root)
    binding = load_site_binding(root)
    _verify_binding_identity(binding)
    selected = _selected_stages(orchestrator.config, targets)
    reports: list[dict[str, Any]] = []
    environment = {**os.environ, **binding.provider_environment}
    values = {
        "run_dir": str(root),
        "run_id": orchestrator.config.run_id,
        "scene_id": orchestrator.config.scene_id,
        "input_video": orchestrator.config.input_video,
    }
    for stage_id in selected:
        site_stage = binding.stages[stage_id]
        try:
            if site_stage.mode == "execute":
                assert site_stage.provider_stage is not None
                provider = preflight_contract(
                    binding.provider_contract,
                    stages=[site_stage.provider_stage],
                    values=values,
                    environment=environment,
                )
                reports.append(
                    {
                        "stage_id": stage_id,
                        "mode": "execute",
                        "status": "ready",
                        "provider_stage": site_stage.provider_stage,
                        "provider_preflight": provider["stages"][0],
                    }
                )
            else:
                assert site_stage.adoption is not None
                snapshots = {
                    role: digest_path(path)
                    for role, path in sorted(site_stage.adoption.outputs.items())
                }
                get_adapter(orchestrator.config.stages[stage_id].adapter).validate_outputs(
                    {role: snapshot_from_digest(snapshot) for role, snapshot in snapshots.items()}
                )
                input_binding_status = _validate_adoption_inputs(
                    orchestrator,
                    stage_id,
                    site_stage.adoption,
                    allow_deferred=True,
                )
                reports.append(
                    {
                        "stage_id": stage_id,
                        "mode": "adopt",
                        "status": "ready",
                        "source_run_id": site_stage.adoption.source_run_id,
                        "input_hash_binding_status": input_binding_status,
                        "outputs": {
                            role: {
                                "path": str(value.path),
                                "sha256": value.sha256,
                                "size_bytes": value.size_bytes,
                                "file_count": value.file_count,
                            }
                            for role, value in snapshots.items()
                        },
                    }
                )
        except (ArtifactError, ConfigurationError, OSError, ValueError) as exc:
            reports.append(
                {
                    "stage_id": stage_id,
                    "mode": site_stage.mode,
                    "status": "blocked",
                    "error_type": type(exc).__name__,
                    "reason": str(exc),
                }
            )
    blocked = [item for item in reports if item["status"] == "blocked"]
    report = {
        "schema_version": 1,
        "kind": "video2world.site_preflight",
        "status": "blocked" if blocked else "passed",
        "created_at": datetime.now(UTC).isoformat(),
        "run_id": orchestrator.config.run_id,
        "scene_id": orchestrator.config.scene_id,
        "profile_id": binding.profile_id,
        "selected_stages": selected,
        "stages": reports,
        "runtime_artifacts_created": False,
        "models_launched": False,
    }
    atomic_write_json(root / ".video2world" / SITE_PREFLIGHT_NAME, report)
    return report


@contextmanager
def _site_lock(run_dir: Path):  # type: ignore[no-untyped-def]
    path = run_dir / ".video2world" / "site-run.lock"
    with path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise StageBlockedError(f"another site runner is active for {run_dir}") from exc
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def run_site_pipeline(
    run_dir: str | Path,
    *,
    targets: list[str] | None = None,
) -> dict[str, Any]:
    """Execute or adopt selected stages, persisting a truthful run receipt."""

    root = Path(run_dir).expanduser().resolve()
    receipt_path = root / ".video2world" / SITE_RUN_RECEIPT_NAME
    started = datetime.now(UTC)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "kind": "video2world.site_run_receipt",
        "status": "running",
        "started_at": started.isoformat(),
        "selected_stages": [],
        "stages": [],
        "complete_pipeline": False,
    }
    atomic_write_json(receipt_path, receipt)
    try:
        with _site_lock(root):
            orchestrator = PipelineOrchestrator(root)
            binding = load_site_binding(root)
            selected = _selected_stages(orchestrator.config, targets)
            receipt.update(
                {
                    "run_id": orchestrator.config.run_id,
                    "scene_id": orchestrator.config.scene_id,
                    "profile_id": binding.profile_id,
                    "selected_stages": selected,
                }
            )
            preflight = preflight_site_run(root, targets=targets)
            receipt["preflight_status"] = preflight["status"]
            if preflight["status"] != "passed":
                blocked = [
                    f"{item['stage_id']}: {item['reason']}"
                    for item in preflight["stages"]
                    if item["status"] == "blocked"
                ]
                raise StageBlockedError("site preflight failed: " + "; ".join(blocked))

            for stage_id in selected:
                plan = next(
                    item for item in orchestrator.plan([stage_id]) if item.stage_id == stage_id
                )
                if plan.action == "cached":
                    action = "cached"
                else:
                    site_stage = binding.stages[stage_id]
                    if site_stage.mode == "adopt":
                        assert site_stage.adoption is not None
                        adoption = site_stage.adoption
                        _validate_adoption_inputs(
                            orchestrator,
                            stage_id,
                            adoption,
                            allow_deferred=False,
                        )
                        orchestrator.adopt(
                            stage_id,
                            adoption.outputs,
                            source_run_id=adoption.source_run_id,
                            source_root=adoption.source_root,
                            source_repository=adoption.source_repository,
                            source_commit=adoption.source_commit,
                            source_command=adoption.source_command,
                            note=adoption.note,
                        )
                        action = "adopted"
                    else:
                        summaries = orchestrator.run([stage_id])
                        summary = next(item for item in summaries if item["stage_id"] == stage_id)
                        action = summary["action"]
                state = load_stage_state(root, stage_id)
                if state is None or state.status not in {"succeeded", "adopted"}:
                    raise StageBlockedError(
                        f"stage {stage_id} did not reach a current terminal state"
                    )
                state_digest = digest_path(state_path(root, stage_id))
                receipt["stages"].append(
                    {
                        "stage_id": stage_id,
                        "action": action,
                        "status": state.status,
                        "mode": state.mode,
                        "state_sha256": state_digest.sha256,
                        "output_roles": sorted(state.outputs),
                    }
                )
                atomic_write_json(receipt_path, receipt)

            all_stages = orchestrator.config.topological_order()
            receipt.update(
                {
                    "status": "completed",
                    "finished_at": datetime.now(UTC).isoformat(),
                    "complete_pipeline": selected == all_stages,
                }
            )
            atomic_write_json(receipt_path, receipt)
            return receipt
    except Exception as exc:
        receipt.update(
            {
                "status": "blocked",
                "finished_at": datetime.now(UTC).isoformat(),
                "error_type": type(exc).__name__,
                "reason": str(exc),
                "complete_pipeline": False,
            }
        )
        atomic_write_json(receipt_path, receipt)
        raise


def adoption_guard(stage_id: str) -> None:
    raise StageBlockedError(
        f"stage {stage_id} is bound to existing artifacts; use 'video2world site-run' "
        "so it is recorded as adopted instead of executed"
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    guard = subparsers.add_parser("adoption-guard")
    guard.add_argument("--stage", required=True)
    args = parser.parse_args(argv)
    try:
        adoption_guard(args.stage)
    except StageBlockedError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
