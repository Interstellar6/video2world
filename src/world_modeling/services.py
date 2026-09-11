"""Versioned, asynchronous module services over a shared outputs/<task_id> store."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import uuid

from .modules import REGISTRY
from .processes import process_identity, process_state, provider_process_state
from .registry import contract_digest
from .source_revision import SourceRevisionError, resolve_command_provider

PROTOCOL = "world-modeling.service.v1"


def provider_exited(identity: dict) -> bool:
    return provider_process_state(identity, leader_state=process_state(identity)) == "exited"


def endpoint_url(endpoint: str) -> str:
    value = urlparse(endpoint)
    if value.scheme not in ("http", "https") or not value.hostname or value.username or value.password:
        raise ValueError("service endpoint must be an HTTP(S) origin without credentials")
    if value.path not in ("", "/") or value.query or value.fragment:
        raise ValueError("service endpoint must not contain a path, query or fragment")
    if value.scheme == "http" and value.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("non-loopback services require HTTPS through an authenticated reverse proxy")
    return endpoint.rstrip("/")


def request_json(endpoint: str, path: str, token_env: str | None = None, payload: dict | None = None) -> dict:
    headers = {"Accept": "application/json"}
    if token_env:
        token = os.environ.get(token_env)
        if not token:
            raise ValueError(f"missing service credential environment variable: {token_env}")
        headers["Authorization"] = f"Bearer {token}"
    data = None if payload is None else json.dumps(payload).encode()
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = Request(endpoint_url(endpoint) + path, data=data, headers=headers)
    try:
        with urlopen(request, timeout=20) as response:
            body = response.read(8 * 1024 * 1024 + 1)
        if len(body) > 8 * 1024 * 1024:
            raise ValueError("service response exceeds 8 MiB")
        value = json.loads(body)
        if not isinstance(value, dict):
            raise ValueError("service response must be a JSON object")
        return value
    except HTTPError as error:
        message = error.read(4096).decode(errors="replace")
        raise ValueError(f"service HTTP {error.code}: {message}") from None


def discover_service(endpoint: str, token_env: str | None = None) -> dict:
    response = request_json(endpoint, "/v1/modules", token_env)
    if response.get("protocol") != PROTOCOL or not isinstance(response.get("modules"), list):
        raise ValueError("incompatible module service protocol")
    for item in response["modules"]:
        spec = REGISTRY.get(item["id"])
        if item.get("contract_sha256") != contract_digest(spec):
            raise ValueError(f"service contract differs from pipeline: {spec.id}")
        if not re.fullmatch(r"[0-9a-f]{64}", item.get("provider_revision", "")):
            raise ValueError(f"service has no provider revision: {spec.id}")
    return response


def register_service(path: Path, endpoint: str, token_env: str | None = None) -> dict:
    from .engine import read_json, write_json

    discovered = discover_service(endpoint, token_env)
    registry = read_json(path) if path.exists() else {"schema_version": "1.0", "services": {}}
    if registry.get("schema_version") != "1.0" or not isinstance(registry.get("services"), dict):
        raise ValueError("invalid service registry")
    for item in discovered["modules"]:
        entry = {"endpoint": endpoint_url(endpoint)}
        if token_env:
            entry["token_env"] = token_env
        registry["services"][item["id"]] = entry
    write_json(path, registry)
    return {"registered": [item["id"] for item in discovered["modules"]], "registry": str(path.resolve())}


def resolve_provider(module_id: str, binding: dict) -> dict:
    discovery = discover_service(binding["endpoint"], binding.get("token_env"))
    available = {item["id"]: item for item in discovery["modules"]}
    if module_id not in available:
        raise ValueError(f"service does not expose requested module: {module_id}")
    module = available[module_id]
    return {**binding, "kind": "service", "provider_revision": module["provider_revision"],
            "contract_sha256": module["contract_sha256"]}


def recover_job(endpoint: str, task_id: str, module_id: str, request_id: str, token_env: str | None = None) -> dict:
    for value in (task_id, module_id):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
            raise ValueError("invalid task or module ID")
    if not re.fullmatch(r"[0-9a-f]{32}", request_id):
        raise ValueError("invalid request ID")
    return request_json(endpoint, f"/v1/jobs/{task_id}/{module_id}/{request_id}/recover", token_env,
                        {"protocol": PROTOCOL})


def service_provider(task, spec, inputs: dict, config: dict) -> dict:
    from .engine import PipelineError, read_json, verify_artifact, write_json

    endpoint = config["endpoint"]
    token_env = config.get("token_env")
    current = resolve_provider(spec.id, config)
    if current["provider_revision"] != config["provider_revision"]:
        raise PipelineError(f"{spec.id}: provider changed after discovery; reload profile")
    fingerprint = hashlib.sha256(json.dumps({"inputs": inputs, "provider": current}, sort_keys=True).encode()).hexdigest()
    receipt = task.directory / "stages" / spec.id / "service-request.json"
    previous = read_json(receipt) if receipt.exists() else {}
    if previous.get("fingerprint") == fingerprint and previous.get("status") == "interrupted":
        known = request_json(endpoint, f"/v1/jobs/{task.task_id}/{spec.id}/{previous['request_id']}", token_env)
        if known.get("status") == "failed" and known.get("recovery"):
            previous = {**previous, "status": "failed"}
    # Retain the request ID after network failure, but never reuse a terminal job
    # when the engine has decided its outputs need regeneration.
    reusable = previous.get("fingerprint") == fingerprint and previous.get("status") not in ("completed", "failed")
    request_id = previous["request_id"] if reusable else uuid.uuid4().hex
    payload = {"protocol": PROTOCOL, "module_id": spec.id, "task_id": task.task_id,
               "request_id": request_id, "provider_revision": current["provider_revision"], "inputs": inputs}
    tracking = {"request_id": request_id, "fingerprint": fingerprint, "endpoint": endpoint, "status": "submitting"}
    write_json(receipt, tracking)
    # A timed-out POST is ambiguous. Retrying this same request ID is idempotent.
    job = request_json(endpoint, "/v1/jobs", token_env, payload)
    while job.get("status") in ("queued", "running"):
        tracking.update(status=job["status"], job_id=job["job_id"])
        write_json(receipt, tracking)
        time.sleep(max(0.1, min(float(config.get("poll_seconds", 2)), 30)))
        job = request_json(endpoint, f"/v1/jobs/{task.task_id}/{spec.id}/{request_id}", token_env)
    tracking.update(status=job.get("status"), job_id=job.get("job_id"))
    write_json(receipt, tracking)
    if job.get("status") != "completed":
        raise PipelineError(f"{spec.id}: service job {job.get('status')}: {job.get('error', 'missing result')}")
    result = job.get("result", {})
    outputs = result.get("outputs", {})
    if set(outputs) != set(spec.output_names):
        raise PipelineError(f"{spec.id}: service returned incompatible artifact roles")
    for role in spec.outputs:
        record = outputs[role.name]
        if record.get("evidence") != role.evidence or (record.get("collision_eligible") and not role.collision_eligible):
            raise PipelineError(f"{spec.id}: service returned incompatible artifact evidence")
        verify_artifact(task, record)
    return {**result, "mode": "service", "service_job": tracking}


class ModuleService:
    def __init__(self, root: Path, spec, provider: dict):
        if provider.get("kind") != "command":
            raise ValueError("module services require a real local command provider")
        self.root = root.resolve()
        self.spec = spec
        self.provider = json.loads(json.dumps(provider))
        self.revision = self.source_revision()
        self.instance = uuid.uuid4().hex
        self.worker_process = process_identity(os.getpid())
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=spec.id)

    def source_revision(self) -> str:
        core = Path(__file__).resolve().parent.rglob("*.py")
        return resolve_command_provider(self.root, self.provider, extra_files=core)["provider_revision"]

    def check_revision(self) -> None:
        try:
            revision = self.source_revision()
        except SourceRevisionError as error:
            raise ValueError(f"provider source revision unavailable: {error}; restart worker after repair") from error
        if revision != self.revision:
            raise ValueError("provider source revision changed; restart worker before discovering or submitting jobs")

    def discovery(self) -> dict:
        self.check_revision()
        return {"protocol": PROTOCOL, "storage": "shared-task-directory", "modules": [
            {**self.spec.as_dict(), "contract_sha256": contract_digest(self.spec), "provider_revision": self.revision},
        ]}

    def job_path(self, task_id: str, module_id: str, request_id: str) -> Path:
        from .engine import Task

        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", task_id):
            raise ValueError("invalid task_id")
        if module_id != self.spec.id or not re.fullmatch(r"[0-9a-f]{32}", request_id):
            raise ValueError("invalid module or request ID")
        task_dir = Task(self.root, task_id).directory
        if task_dir.resolve() != task_dir or not (task_dir / "task.json").is_file():
            raise ValueError("task does not exist in the configured task store")
        result = task_dir / "stages" / module_id / "service-jobs" / f"{request_id}.json"
        if task_dir not in result.resolve().parents:
            raise ValueError("job path escapes task directory")
        return result

    def status(self, task_id: str, module_id: str, request_id: str) -> dict:
        from .engine import read_json

        result = read_json(self.job_path(task_id, module_id, request_id))
        if result["status"] in ("queued", "running") and result["worker_instance"] != self.instance:
            if process_state(result.get("worker_process")) != "alive":
                return {**result, "status": "interrupted", "error": "worker unavailable; explicitly recover this job after worker and provider exit"}
        return result

    def recover_job(self, task_id: str, module_id: str, request_id: str) -> dict:
        from .engine import exclusive_lock, read_json, write_json

        path = self.job_path(task_id, module_id, request_id)
        with exclusive_lock(path.parent / "claims.lock", blocking=True):
            result = read_json(path)
            if result.get("status") == "failed" and result.get("recovery"):
                return result
            if result.get("status") not in ("queued", "running", "interrupted"):
                raise ValueError("only an interrupted queued or running job can be recovered")
            if process_state(result.get("worker_process")) != "exited":
                raise ValueError("cannot recover: original worker is alive or its identity cannot be verified")
            stage = path.parent.parent
            with exclusive_lock(stage / "provider.lock"):
                live_path = stage / "provider-live.json"
                provider = read_json(live_path) if live_path.exists() else None
                if result.get("status") != "queued":
                    if not provider or provider.get("execution_id") != request_id:
                        raise ValueError("cannot recover: matching provider process receipt is missing")
                    if not provider_exited(provider):
                        raise ValueError("cannot recover: provider is alive or its identity cannot be verified")
                elif provider and provider.get("execution_id") == request_id:
                    if not provider_exited(provider):
                        raise ValueError("cannot recover: provider is alive or its identity cannot be verified")
                recovered = {**result, "status": "failed", "error": "explicitly recovered after original worker and provider exited",
                    "recovery": {"recovered_unix": time.time(), "worker_instance": self.instance,
                                 "worker_process": result["worker_process"], "provider_process": provider}}
                write_json(path, recovered)
                return recovered

    def submit(self, payload: dict) -> dict:
        from .engine import Task, exclusive_lock, validate_available_inputs, write_json

        self.check_revision()
        if not isinstance(payload, dict) or payload.get("protocol") != PROTOCOL or payload.get("provider_revision") != self.revision:
            raise ValueError("protocol or provider revision mismatch")
        path = self.job_path(payload["task_id"], payload["module_id"], payload["request_id"])
        task = Task(self.root, payload["task_id"])
        inputs = validate_available_inputs(task, self.spec, allow_contract=False)
        if payload.get("inputs") != inputs:
            raise ValueError("request inputs differ from registered task artifacts")
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        with exclusive_lock(path.parent / "claims.lock", blocking=True):
            if path.exists():
                existing = self.status(payload["task_id"], self.spec.id, payload["request_id"])
                if existing["request_sha256"] != digest:
                    raise ValueError("request ID was already used for different inputs")
                return existing
            result = {"protocol": PROTOCOL, "job_id": payload["request_id"], "status": "queued",
                      "worker_instance": self.instance, "worker_process": self.worker_process, "request_sha256": digest}
            write_json(path, result)
            self.executor.submit(self.execute, task, inputs, path, result)
        return result

    def execute(self, task, inputs: dict, path: Path, result: dict) -> None:
        from .engine import command_provider, validate_available_inputs, write_json

        result = {**result, "status": "running"}
        write_json(path, result)
        try:
            self.check_revision()
            if validate_available_inputs(task, self.spec, allow_contract=False) != inputs:
                raise ValueError("task inputs changed while the job was queued")
            output = command_provider(task, self.spec, inputs, self.provider, execution_id=result["job_id"])
            self.check_revision()
            if validate_available_inputs(task, self.spec, allow_contract=False) != inputs:
                raise ValueError("provider changed task inputs during execution")
            write_json(path, {**result, "status": "completed", "result": output})
        except Exception as error:
            write_json(path, {**result, "status": "failed", "error": f"{type(error).__name__}: {error}"})


def make_server(service: ModuleService, host: str = "127.0.0.1", port: int = 0, token: str | None = None):
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("bind module workers to loopback; use an authenticated HTTPS reverse proxy for remote access")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def send_json(self, code, value):
            body = json.dumps(value).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def dispatch(self, post=False):
            if token and not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
                self.send_json(401, {"error": "unauthorized"})
                return
            try:
                if not post and self.path == "/v1/modules":
                    self.send_json(200, service.discovery())
                elif not post and self.path.startswith("/v1/jobs/"):
                    pieces = self.path.split("/")
                    if len(pieces) != 6:
                        raise ValueError("expected task/module/request job path")
                    self.send_json(200, service.status(*pieces[3:]))
                elif post and self.path == "/v1/jobs":
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 8 * 1024 * 1024:
                        raise ValueError("request must contain JSON under 8 MiB")
                    payload = json.loads(self.rfile.read(length))
                    self.send_json(202, service.submit(payload))
                elif post and self.path.startswith("/v1/jobs/") and self.path.endswith("/recover"):
                    pieces = self.path.split("/")
                    if len(pieces) != 7:
                        raise ValueError("expected task/module/request recovery path")
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 4096:
                        raise ValueError("invalid recovery protocol")
                    payload = json.loads(self.rfile.read(length))
                    if not isinstance(payload, dict) or payload.get("protocol") != PROTOCOL:
                        raise ValueError("invalid recovery protocol")
                    self.send_json(200, service.recover_job(*pieces[3:6]))
                else:
                    self.send_json(404, {"error": "unknown service route"})
            except (ValueError, KeyError, TypeError, RuntimeError) as error:
                self.send_json(400, {"error": str(error)})

        def do_GET(self):
            self.dispatch()

        def do_POST(self):
            self.dispatch(post=True)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server
