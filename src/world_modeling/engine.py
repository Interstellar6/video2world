from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import ModuleSpec, task_relative
from .modules import REGISTRY, SPECS
from .processes import process_identity, provider_process_state, terminate_started_process
from .source_revision import SourceRevisionError, check_command_revision, resolve_command_provider


class PipelineError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@contextmanager
def exclusive_lock(path: Path, blocking: bool = False):
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as error:
            raise PipelineError(f"another execution owns {path.name}") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_path(path: Path) -> tuple[str, int]:
    """Return a deterministic digest and size for one task-local file or directory."""
    if path.is_file():
        return sha256(path), path.stat().st_size
    if not path.is_dir():
        raise PipelineError(f"snapshot target is neither a file nor directory: {path}")
    digest = hashlib.sha256()
    total_size = 0
    children = sorted(path.rglob("*"))
    for child in children:
        if child.is_symlink():
            raise PipelineError(f"artifact directory contains a symlink: {child}")
        if not (child.is_file() or child.is_dir()):
            raise PipelineError(f"artifact directory contains an unsupported entry: {child}")
    for child in (item for item in children if item.is_file()):
        relative = child.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        file_digest = sha256(child).encode("ascii")
        digest.update(file_digest)
        total_size += child.stat().st_size
    return digest.hexdigest(), total_size


def validate_source(path: Path) -> None:
    if not path.exists() or not (path.is_file() or path.is_dir()):
        raise PipelineError(f"source must be an existing file or directory: {path}")
    if not path.is_dir():
        return
    dangling = [item.relative_to(path).as_posix() for item in path.rglob("*") if item.is_symlink() and not item.exists()]
    if dangling:
        preview = ", ".join(dangling[:5])
        suffix = " ..." if len(dangling) > 5 else ""
        raise PipelineError(f"source contains {len(dangling)} dangling symlink(s): {preview}{suffix}")


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PipelineError(f"invalid JSON: {path}: {error}") from error


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class Task:
    root: Path
    task_id: str

    def __post_init__(self):
        if not isinstance(self.task_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", self.task_id):
            raise PipelineError("task_id must be a simple directory name using letters, digits, dot, underscore or hyphen")
        root = self.root.resolve()
        store = root / "outputs"
        if store.resolve() != store or (store / self.task_id).resolve().parent != store:
            raise PipelineError("task directory escapes the project's outputs store")
        object.__setattr__(self, "root", root)

    @property
    def directory(self) -> Path:
        return self.root / "outputs" / self.task_id

    @property
    def state_path(self) -> Path:
        return self.directory / "task.json"

    @property
    def artifacts_path(self) -> Path:
        return self.directory / "artifacts.json"

    def load_state(self) -> dict[str, Any]:
        return read_json(self.state_path)

    def load_artifacts(self) -> dict[str, Any]:
        return read_json(self.artifacts_path)


def create_task(root: Path, task_id: str, source: Path | None = None) -> Task:
    if not task_id or "/" in task_id or "\\" in task_id or task_id in {".", ".."}:
        raise PipelineError("task_id must be a simple directory name")
    task = Task(root.resolve(), task_id)
    if task.directory.exists():
        raise PipelineError(f"task already exists: {task.directory}")
    source_path: Path | None = None
    if source is not None:
        source_path = source.resolve()
        validate_source(source_path)
    (task.directory / "inputs").mkdir(parents=True)
    state = {
        "schema_version": "1.0",
        "task_id": task_id,
        "created_at": now(),
        "status": "initialized",
        "promotion_allowed": False,
        "stages": {},
    }
    artifacts: dict[str, Any] = {"schema_version": "1.0", "task_id": task_id, "artifacts": {}}
    try:
        if source_path is not None:
            destination = task.directory / "inputs" / source_path.name
            if source_path.is_file():
                shutil.copy2(source_path, destination)
            else:
                shutil.copytree(source_path, destination)
            source_digest, source_size = snapshot_path(destination)
            artifacts["artifacts"]["source_media"] = {
                "role": "source_media",
                "path": task_relative(task.directory, destination),
                "sha256": source_digest,
                "size_bytes": source_size,
                "evidence": "observed",
                "media_hint": "video/*, image/*, or calibrated image-sequence directory",
                "producer": "task_input",
                "status": "available",
            }
        write_json(task.state_path, state)
        write_json(task.artifacts_path, artifacts)
    except Exception:
        shutil.rmtree(task.directory)
        raise
    return task


# The background branch is optional: clean_plate needs an image-generation
# credential, and scene_recomposition requires generated_background_gaussian_ply
# from background_reconstruction, so it cannot run when that branch is skipped.
SKIP_BG_RECON_MODULES = ("clean_plate", "background_reconstruction", "scene_recomposition")


def without_bg_recon(module_ids: list[str]) -> list[str]:
    """Drop the background branch from a pipeline declaration, preserving order."""
    kept = [module_id for module_id in module_ids if module_id not in SKIP_BG_RECON_MODULES]
    if not kept:
        raise PipelineError("--skip-bg-recon would leave an empty pipeline")
    return kept


def load_profile(path: Path) -> dict[str, Any]:
    profile = read_json(path)
    if profile.get("schema_version") != "1.0":
        raise PipelineError("profile schema_version must be 1.0")
    if not isinstance(profile.get("providers"), dict):
        raise PipelineError("profile.providers must be an object")
    unknown = sorted(set(profile["providers"]) - set(SPECS))
    if unknown:
        raise PipelineError(f"profile contains unknown modules: {unknown}")
    registry = None
    for module_id, binding in profile["providers"].items():
        if binding.get("kind") != "service":
            continue
        try:
            from .services import resolve_provider
            if "endpoint" not in binding:
                if registry is None:
                    registry = read_json(path.parent / profile["registry"])
                    if registry.get("schema_version") != "1.0":
                        raise ValueError("invalid service registry schema_version")
                binding = {**registry["services"][module_id], **binding}
            profile["providers"][module_id] = resolve_provider(module_id, binding)
        except (ValueError, KeyError, OSError) as error:
            raise PipelineError(f"{module_id}: service discovery failed: {error}") from error
    return profile


def dependency_index(task: Task) -> dict[str, dict[str, Any]]:
    raw = task.load_artifacts().get("artifacts", {})
    if not isinstance(raw, dict):
        raise PipelineError("artifacts.json.artifacts must be an object")
    return raw


def artifact_path(task: Task, record: dict[str, Any]) -> Path:
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise PipelineError("malformed artifact record: expected a path")
    path = (task.directory / record["path"]).resolve()
    if task.directory.resolve() not in path.parents or not (path.is_file() or path.is_dir()):
        raise PipelineError(f"artifact is missing or escapes task directory: {path}")
    return path


def verify_artifact(task: Task, record: dict[str, Any]) -> None:
    digest, size = snapshot_path(artifact_path(task, record))
    if digest != record.get("sha256"):
        raise PipelineError(f"sha256 mismatch: {record['path']}")
    if size != record.get("size_bytes"):
        raise PipelineError(f"size_bytes mismatch: {record['path']}")


def is_contract_artifact(record: dict[str, Any]) -> bool:
    return (
        record.get("evidence") == "contract_only"
        or record.get("status") == "contract_only"
        or str(record.get("producer", "")).startswith("contract:")
    )


def validate_available_inputs(
    task: Task, spec: ModuleSpec, *, allow_contract: bool = True,
) -> dict[str, dict[str, Any]]:
    index = dependency_index(task)
    missing = [role for role in spec.inputs if role not in index]
    if missing:
        raise PipelineError(f"{spec.id}: missing required artifact roles: {missing}")
    for role in spec.inputs:
        artifact = index[role]
        try:
            verify_artifact(task, artifact)
            if not allow_contract and is_contract_artifact(artifact):
                raise PipelineError("command providers cannot consume contract_only artifacts")
        except PipelineError as error:
            raise PipelineError(f"{spec.id}: input {role}: {error}") from error
    return {role: index[role] for role in spec.inputs}


def contract_provider(task: Task, spec: ModuleSpec, inputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Write contract artifacts only. This validates wiring without claiming model inference."""
    output_dir = task.directory / "stages" / spec.id / "contract"
    outputs: dict[str, Any] = {}
    for output in spec.outputs:
        path = output_dir / f"{output.name}.json"
        write_json(
            path,
            {
                "kind": "video2world-modeling.contract-output",
                "role": output.name,
                "expected_evidence": output.evidence,
                "expected_media_hint": output.media_hint,
                "module": spec.id,
                "message": "Contract-only artifact. No model was invoked and no geometry/image/video was produced.",
            },
        )
        outputs[output.name] = {
            "role": output.name,
            "path": task_relative(task.directory, path),
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
            "evidence": "contract_only",
            "declared_evidence": output.evidence,
            "media_hint": "application/json",
            "producer": f"contract:{spec.id}",
            "status": "contract_only",
            "collision_eligible": False,
        }
    return {"mode": "contract", "outputs": outputs, "inputs": inputs}


def _command_context(task: Task, spec: ModuleSpec) -> dict[str, str]:
    stage_dir = task.directory / "stages" / spec.id
    return {
        "task_dir": str(task.directory), "stage_dir": str(stage_dir),
        "inputs_json": str(stage_dir / "inputs.json"),
        "outputs_json": str(stage_dir / "provider-artifacts.json"), "task_id": task.task_id,
    }


def _resolve_command(task: Task, spec: ModuleSpec, provider: dict) -> dict:
    try:
        context = _command_context(task, spec)
        if "source_revision" in provider or "provider_revision" in provider:
            check_command_revision(task.root, provider, cwd=task.directory, context=context)
        return resolve_command_provider(task.root, provider, cwd=task.directory, context=context)
    except SourceRevisionError as error:
        raise PipelineError(f"{spec.id}: {error}") from error


def command_provider(
    task: Task,
    spec: ModuleSpec,
    inputs: dict[str, dict[str, Any]],
    config: dict[str, Any],
    execution_id: str | None = None,
) -> dict[str, Any]:
    stage_dir = task.directory / "stages" / spec.id
    if task.directory not in stage_dir.resolve().parents:
        raise PipelineError("provider stage escapes task directory")
    config = _resolve_command(task, spec, config)
    with exclusive_lock(stage_dir / "provider.lock"):
        _check_previous_provider(stage_dir)
        return _command_provider(task, spec, inputs, config, execution_id)


def _check_previous_provider(stage_dir: Path) -> None:
    live_path = stage_dir / "provider-live.json"
    if live_path.exists():
        previous = read_json(live_path)
        if not isinstance(previous, dict):
            raise PipelineError("invalid previous provider process receipt")
        if previous.get("status") in ("running", "initializing", "unknown"):
            state = provider_process_state(previous)
            if state != "exited":
                raise PipelineError(f"previous provider process group is {state}; refusing duplicate launch")
    execution_path = stage_dir / "provider-execution.json"
    if execution_path.exists():
        previous = read_json(execution_path)
        cleanup = previous.get("process_cleanup", {})
        if cleanup and not cleanup.get("confirmed_exited"):
            state = provider_process_state(previous.get("provider_process", {}))
            if state != "exited":
                raise PipelineError(f"previous provider cleanup is unconfirmed ({state}); refusing duplicate launch")


def _command_provider(task: Task, spec: ModuleSpec, inputs: dict, config: dict, execution_id: str | None) -> dict:
    check_command_revision(task.root, config, cwd=task.directory, context=_command_context(task, spec))
    command = config.get("command")
    if not isinstance(command, list) or not command or not all(isinstance(item, str) for item in command):
        raise PipelineError(f"{spec.id}: command provider requires a non-empty string argv list")
    stage_dir = task.directory / "stages" / spec.id
    stage_dir.mkdir(parents=True, exist_ok=True)
    inputs_path = stage_dir / "inputs.json"
    outputs_path = stage_dir / "provider-artifacts.json"
    outputs_path.unlink(missing_ok=True)
    write_json(inputs_path, {"module": spec.id, "inputs": inputs})
    context = _command_context(task, spec)
    argv = [item.format(**context) for item in command]
    stdout_path = stage_dir / "provider.stdout.log"
    stderr_path = stage_dir / "provider.stderr.log"
    live_path = stage_dir / "provider-live.json"
    process = None
    live = {"execution_id": execution_id or uuid.uuid4().hex, "hostname": socket.gethostname(),
            "started_at": now(), "status": "initializing", "argv_sha256": hashlib.sha256(json.dumps(argv).encode()).hexdigest()}
    try:
        write_json(live_path, live)
        with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
            process = subprocess.Popen(argv, cwd=task.directory, stdout=stdout, stderr=stderr,
                                       env={**os.environ, "PYTHONUNBUFFERED": "1"}, start_new_session=True)
            live.update(pid=process.pid, process_group_id=process.pid)
            live.update(process_identity(process.pid))
            if not (live.get("proc_start_ticks") or live.get("process_started_at")):
                raise PipelineError("provider process start identity could not be captured")
            live["status"] = "running"
            write_json(live_path, live)
            returncode = process.wait()
            write_json(live_path, {**live, "status": "completed" if returncode == 0 else "failed", "returncode": returncode, "ended_at": now()})
    except BaseException as error:
        failure = {"argv": argv, "returncode": None, "error": str(error), "stdout": "", "stderr": ""}
        receipt_errors = []
        if process is not None:
            try:
                cleanup = terminate_started_process(process, live)
            except Exception as cleanup_error:
                cleanup = {"confirmed_exited": False, "error": str(cleanup_error)}
            failure.update(provider_process=live, process_cleanup=cleanup, returncode=process.returncode)
            failed_live = {**live, "status": "failed" if cleanup["confirmed_exited"] else "unknown",
                           "error": str(error), "process_cleanup": cleanup, "ended_at": now()}
            try:
                write_json(live_path, failed_live)
            except Exception as receipt_error:
                receipt_errors.append(f"provider-live.json: {receipt_error}")
        else:
            try:
                write_json(live_path, {**live, "status": "failed", "spawned": False, "error": str(error), "ended_at": now()})
            except Exception as receipt_error:
                receipt_errors.append(f"provider-live.json: {receipt_error}")
        failure["receipt_errors"] = receipt_errors
        try:
            write_json(stage_dir / "provider-execution.json", failure)
        except Exception as receipt_error:
            receipt_errors.append(f"provider-execution.json: {receipt_error}")
        if not isinstance(error, Exception):
            raise
        operation = "could not start" if process is None else "initialization or execution failed"
        detail = f"; receipt errors: {receipt_errors}" if receipt_errors else ""
        raise PipelineError(f"{spec.id}: provider {operation}: {error}{detail}") from error
    execution = {
        "argv": argv, "returncode": returncode,
        "stdout": stdout_path.read_text(errors="replace")[-65536:], "stderr": stderr_path.read_text(errors="replace")[-65536:],
        "stdout_path": task_relative(task.directory, stdout_path), "stderr_path": task_relative(task.directory, stderr_path),
        "source_revision": config["source_revision"], "provider_revision": config["provider_revision"],
    }
    write_json(stage_dir / "provider-execution.json", execution)
    if returncode != 0:
        raise PipelineError(f"{spec.id}: provider failed with exit code {returncode}; see provider-execution.json")
    try:
        check_command_revision(task.root, config, cwd=task.directory, context=context)
    except SourceRevisionError as error:
        write_json(stage_dir / "provider-execution.json", {**execution, "error": str(error), "outputs_accepted": False})
        raise PipelineError(f"{spec.id}: {error}") from error
    if not outputs_path.is_file():
        raise PipelineError(f"{spec.id}: provider did not write {outputs_path.name}")
    payload = read_json(outputs_path)
    supplied = payload.get("outputs")
    if not isinstance(supplied, dict) or set(supplied) != set(spec.output_names):
        raise PipelineError(f"{spec.id}: provider output roles must be exactly {list(spec.output_names)}")
    output_roles = {role.name: role for role in spec.outputs}
    outputs: dict[str, Any] = {}
    for name, record in supplied.items():
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise PipelineError(f"{spec.id}: output {name} must have a path")
        path = artifact_path(task, record)
        declared = output_roles[name]
        evidence = record.get("evidence", declared.evidence)
        if evidence != declared.evidence:
            raise PipelineError(f"{spec.id}: output {name} evidence must be {declared.evidence}, got {evidence}")
        collision_eligible = bool(record.get("collision_eligible", declared.collision_eligible))
        if is_contract_artifact(record):
            raise PipelineError(f"{spec.id}: command output {name} cannot be contract_only")
        if collision_eligible and not declared.collision_eligible:
            raise PipelineError(f"{spec.id}: output {name} is not permitted as collision evidence")
        digest, size = snapshot_path(path)
        outputs[name] = {
            "role": name,
            "path": task_relative(task.directory, path),
            "sha256": digest,
            "size_bytes": size,
            "evidence": evidence,
            "media_hint": record.get("media_hint", declared.media_hint),
            "producer": f"command:{spec.id}",
            "status": record.get("status", "candidate"),
            "collision_eligible": collision_eligible,
        }
    return {"mode": "command", "outputs": outputs, "inputs": inputs,
            "command_source_revision": config["source_revision"], "command_provider_revision": config["provider_revision"]}


def request_fingerprint(spec: ModuleSpec, provider: dict[str, Any], inputs: dict[str, Any]) -> str:
    payload = {"module": spec.as_dict(), "provider": provider, "inputs": inputs}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def can_resume(task: Task, spec: ModuleSpec, existing: dict[str, Any], fingerprint: str, status: str) -> bool:
    if existing.get("status") != status or existing.get("request_sha256") != fingerprint:
        return False
    try:
        manifest_path = artifact_path(task, {"path": existing["manifest"]})
        manifest = read_json(manifest_path)
        if manifest.get("request_sha256") != fingerprint or manifest.get("status") != status:
            return False
        if manifest.get("module") != spec.as_dict() or request_fingerprint(spec, manifest["provider"], manifest["inputs"]) != fingerprint:
            return False
        outputs = manifest.get("outputs", {})
        if set(outputs) != set(spec.output_names):
            return False
        index = dependency_index(task)
        for name, record in outputs.items():
            if index.get(name) != record:
                return False
            verify_artifact(task, record)
        return True
    except (PipelineError, KeyError, TypeError, OSError):
        return False


def invalidate_stage(task: Task, spec: ModuleSpec, state: dict[str, Any]) -> None:
    artifacts = task.load_artifacts()
    affected = {spec.id}
    invalid_roles = set(spec.output_names)
    # Invalidation follows artifact dependencies, including branches that bypass a stage.
    while True:
        descendants = {item.id for item in REGISTRY.modules if set(item.inputs) & invalid_roles} - affected
        if not descendants:
            break
        affected.update(descendants)
        for module_id in descendants:
            invalid_roles.update(SPECS[module_id].output_names)
    for name in invalid_roles:
        artifacts["artifacts"].pop(name, None)
    for module_id in affected:
        if module_id in state["stages"]:
            state["stages"][module_id].update(status="stale", invalidated_by=spec.id)
    state["promotion_allowed"] = False
    write_json(task.artifacts_path, artifacts)


def run_stage(task: Task, spec: ModuleSpec, provider: dict[str, Any]) -> dict[str, Any]:
    with exclusive_lock(task.directory / "orchestrator.lock"):
        return _run_stage(task, spec, provider)


def _run_stage(task: Task, spec: ModuleSpec, provider: dict[str, Any]) -> dict[str, Any]:
    state = task.load_state()
    stage_dir = task.directory / "stages" / spec.id
    if task.directory.resolve() not in stage_dir.resolve().parents:
        raise PipelineError(f"{spec.id}: stage directory escapes task directory")
    # Source resolution is a read-only preflight, never a failed replacement receipt.
    if provider.get("kind") == "command":
        provider = _resolve_command(task, spec, provider)
    manifest_path = stage_dir / "stage.json"
    inputs: dict[str, Any] = {}
    fingerprint = None
    invalidated = False
    started_at = now()
    try:
        kind = provider.get("kind")
        if kind not in {"contract", "command", "service"}:
            raise PipelineError(f"{spec.id}: provider kind must be contract, command or service")
        status = "contract_only" if kind == "contract" else "executed"
        if kind == "service":
            from .services import resolve_provider
            provider = resolve_provider(spec.id, provider)
        inputs = validate_available_inputs(task, spec, allow_contract=kind == "contract")
        fingerprint = request_fingerprint(spec, provider, inputs)
        if can_resume(task, spec, state["stages"].get(spec.id, {}), fingerprint, status):
            return state["stages"][spec.id]
        invalidate_stage(task, spec, state)
        invalidated = True
        state["stages"][spec.id] = {"status": "running", "started_at": started_at}
        state["status"] = "running"
        write_json(task.state_path, state)
        if kind == "service":
            from .services import service_provider
            result = service_provider(task, spec, inputs, provider)
        elif kind == "contract":
            result = contract_provider(task, spec, inputs)
        else:
            result = command_provider(task, spec, inputs, provider)
            check_command_revision(task.root, provider, cwd=task.directory, context=_command_context(task, spec))
        if validate_available_inputs(task, spec, allow_contract=kind == "contract") != inputs:
            raise PipelineError(f"{spec.id}: provider changed its registered inputs during execution")
        stage_manifest = {
            "schema_version": "1.0", "module": spec.as_dict(), "status": status,
            "started_at": started_at, "executed_at": now(), "provider": provider,
            "request_sha256": fingerprint, **result,
        }
        write_json(manifest_path, stage_manifest)
        artifacts = task.load_artifacts()
        artifacts["artifacts"].update(result["outputs"])
        write_json(task.artifacts_path, artifacts)
        state["stages"][spec.id] = {
            "status": status, "manifest": task_relative(task.directory, manifest_path),
            "request_sha256": fingerprint,
        }
        state["status"] = "contract_only" if status == "contract_only" else "running"
        write_json(task.state_path, state)
        return state["stages"][spec.id]
    except Exception as error:
        if not invalidated:
            invalidate_stage(task, spec, state)
        write_json(manifest_path, {
            "schema_version": "1.0", "module": spec.as_dict(), "status": "failed",
            "started_at": started_at, "failed_at": now(), "provider": provider,
            "request_sha256": fingerprint, "inputs": inputs, "outputs": {},
            "error": {"type": type(error).__name__, "message": str(error)},
        })
        state["stages"][spec.id] = {
            "status": "failed", "manifest": task_relative(task.directory, manifest_path),
            "error": str(error),
        }
        state["status"] = "failed"
        write_json(task.state_path, state)
        if isinstance(error, PipelineError):
            raise
        raise PipelineError(f"{spec.id}: {error}") from error


def finalize(task: Task) -> dict[str, Any]:
    with exclusive_lock(task.directory / "orchestrator.lock"):
        return _finalize(task)


def _finalize(task: Task) -> dict[str, Any]:
    state = task.load_state()
    module_ids = state.get("pipeline", [spec.id for spec in REGISTRY.modules])
    states = [state["stages"].get(module_id, {}).get("status") for module_id in module_ids]
    if all(item == "executed" for item in states):
        state["status"] = "completed"
    elif all(item == "contract_only" for item in states):
        state["status"] = "contract_only"
    elif "failed" in states:
        state["status"] = "failed"
    elif "stale" in states:
        state["status"] = "stale"
    elif "executed" in states and "contract_only" in states:
        state["status"] = "mixed"
    state["promotion_allowed"] = False
    state["promotion_blockers"] = [
        "No visual, geometry, collision, or source-camera QA gate is implemented by the orchestration contract.",
        "Generated background outputs remain non-observed candidates.",
    ]
    write_json(task.state_path, state)
    return state


def run_pipeline(task: Task, profile: dict[str, Any], through: str | None = None) -> dict[str, Any]:
    with exclusive_lock(task.directory / "orchestrator.lock"):
        return _run_pipeline(task, profile, through)


def _run_pipeline(task: Task, profile: dict[str, Any], through: str | None) -> dict[str, Any]:
    module_ids = profile.get("pipeline", [spec.id for spec in REGISTRY.modules])
    if not isinstance(module_ids, list) or not module_ids:
        raise PipelineError("profile.pipeline must be a nonempty module ID array")
    try:
        ordered = REGISTRY.ordered(module_ids)
    except (ValueError, KeyError) as error:
        raise PipelineError(str(error)) from error
    if through is not None and through not in module_ids:
        raise PipelineError(f"unknown final module: {through}")
    providers = {}
    for spec in ordered:
        provider = profile["providers"].get(spec.id)
        if not isinstance(provider, dict):
            raise PipelineError(f"profile has no provider binding for {spec.id}")
        providers[spec.id] = _resolve_command(task, spec, provider) if provider.get("kind") == "command" else provider
        if spec.id == through:
            break
    state = task.load_state()
    removed = set(state.get("pipeline", state["stages"])) - set(module_ids)
    if removed:
        artifacts = task.load_artifacts()
        archive = state.setdefault("retired_stages", {})
        for module_id in removed:
            if module_id in state["stages"]:
                archive[module_id] = {**state["stages"].pop(module_id), "retired_at": now()}
        artifacts["artifacts"] = {name: item for name, item in artifacts["artifacts"].items()
                                   if item.get("producer", "").split(":")[-1] not in removed}
        write_json(task.artifacts_path, artifacts)
    state["pipeline"] = [spec.id for spec in ordered]
    write_json(task.state_path, state)
    for spec in ordered:
        _run_stage(task, spec, providers[spec.id])
        if spec.id == through:
            break
    return _finalize(task)


def validate_task(task: Task) -> dict[str, Any]:
    state = task.load_state()
    artifacts = dependency_index(task)
    issues: list[str] = []
    for name, record in artifacts.items():
        try:
            verify_artifact(task, record)
        except PipelineError as error:
            issues.append(f"{name}: {error}")
        except (KeyError, TypeError, OSError):
            issues.append(f"{name}: malformed artifact record")
        if isinstance(record, dict) and record.get("evidence") == "generated" and record.get("collision_eligible"):
            issues.append(f"{name}: generated output cannot be collision eligible")
    pipeline_ids = state.get("pipeline", [spec.id for spec in REGISTRY.modules])
    for module_id in pipeline_ids:
        if module_id not in SPECS:
            issues.append(f"pipeline module is not registered: {module_id}")
            continue
        spec = SPECS[module_id]
        stage = state.get("stages", {}).get(spec.id)
        if not stage:
            continue
        status = stage.get("status")
        if status not in {"executed", "contract_only"}:
            issues.append(f"{spec.id}: stage is {status}")
            continue
        try:
            inputs = validate_available_inputs(task, spec, allow_contract=status == "contract_only")
            manifest = read_json(artifact_path(task, {"path": stage["manifest"]}))
            fingerprint = request_fingerprint(spec, manifest["provider"], inputs)
            if not can_resume(task, spec, stage, fingerprint, status):
                issues.append(f"{spec.id}: stage receipt, inputs, or output records are stale")
        except (PipelineError, KeyError, TypeError, OSError) as error:
            issues.append(f"{spec.id}: {error}")
    try:
        REGISTRY.ordered(pipeline_ids)
    except (ValueError, KeyError) as error:
        issues.append(f"invalid registered pipeline: {error}")
    statuses = [state.get("stages", {}).get(module_id, {}).get("status") for module_id in pipeline_ids]
    if state.get("status") == "completed" and not all(status == "executed" for status in statuses):
        issues.append("completed status requires every module to have executed with real providers")
    return {"task_id": task.task_id, "valid": not issues, "issues": issues, "status": state.get("status")}
