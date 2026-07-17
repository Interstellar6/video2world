"""Loopback-first HTTP boundary for planning and queuing scene commands.

The browser preview is advisory. This service always validates the typed command,
loads its fixed server-side manifest, rebuilds the plan, and writes only immutable
queue jobs through :mod:`video2world.scene_command_queue`.
"""

from __future__ import annotations

import json
import socket
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pydantic import Field, JsonValue, ValidationError

from video2world.hashing import digest_json
from video2world.models import Sha256, StrictModel, WorldManifest
from video2world.scene_command_queue import submit_scene_command
from video2world.scene_commands import SceneCommand, SceneCommandPlan, plan_scene_command
from video2world.validation import load_world_manifest
from video2world.web_adoption import WEB_MANIFEST_CONTRACT, WEB_MANIFEST_SCHEMA_VERSION

PLAN_PATH = "/v1/scene-commands/plan"
SUBMIT_PATH = "/v1/scene-commands/submit"
HEALTH_PATH = "/health"
DEFAULT_MAX_JSON_BYTES = 256 * 1024


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _json_object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _blocked_reason(plan: SceneCommandPlan) -> str:
    details = [*plan.warnings]
    details.extend(
        resolution.reason
        for resolution in plan.target_resolutions
        if resolution.status != "resolved"
    )
    return "; ".join(dict.fromkeys(details)) or plan.status


class SceneCommandHttpEnvelope(StrictModel):
    """The Web transport envelope; duplicated client fields are never authoritative."""

    command: SceneCommand
    structured_intent: dict[str, JsonValue] = Field(alias="structuredIntent")
    raw_prompt: str = Field(alias="rawPrompt", min_length=1, max_length=16_384)
    expected_manifest_sha256: Sha256 | None = Field(
        default=None,
        alias="expectedManifestSha256",
    )
    client_preview: dict[str, JsonValue] = Field(alias="clientPreview")
    confirmation_phrase: str | None = Field(
        default=None,
        alias="confirmationPhrase",
        min_length=1,
        max_length=128,
    )


@dataclass(frozen=True)
class SceneCommandServiceConfig:
    manifest_path: Path
    web_manifest_path: Path | None
    queue_dir: Path
    allowed_origins: frozenset[str]
    max_json_bytes: int = DEFAULT_MAX_JSON_BYTES

    def __post_init__(self) -> None:
        if self.max_json_bytes < 1:
            raise ValueError("max_json_bytes must be positive")
        if "*" in self.allowed_origins:
            raise ValueError("wildcard CORS origins are forbidden for credentialed requests")
        for origin in self.allowed_origins:
            parsed = urlsplit(origin)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.netloc
                or parsed.username is not None
                or parsed.password is not None
            ):
                raise ValueError(f"CORS origin must be an exact http(s) origin: {origin!r}")
            if parsed.path or parsed.query or parsed.fragment:
                raise ValueError(
                    f"CORS origin must not include a path, query, or fragment: {origin!r}"
                )

    def load_manifest(self) -> tuple[WorldManifest, str]:
        manifest = load_world_manifest(self.manifest_path)
        manifest_sha256 = digest_json(manifest.model_dump(mode="json"))
        if self.web_manifest_path is not None:
            payload = json.loads(
                self.web_manifest_path.read_text(encoding="utf-8"),
                parse_constant=_reject_json_constant,
                object_pairs_hook=_json_object_without_duplicates,
            )
            if not isinstance(payload, dict):
                raise ValueError("bound Web manifest root must be an object")
            if payload.get("schemaVersion") != WEB_MANIFEST_SCHEMA_VERSION:
                raise ValueError("bound Web manifest schemaVersion must equal 1")
            if payload.get("contract") != WEB_MANIFEST_CONTRACT:
                raise ValueError(f"bound Web manifest contract must equal {WEB_MANIFEST_CONTRACT}")
            version = payload.get("version")
            if not isinstance(version, str) or not version.strip():
                raise ValueError("bound Web manifest version must be a non-empty string")
            source = payload.get("sourceWorld")
            if not isinstance(source, dict):
                raise ValueError("bound Web manifest has no sourceWorld object")
            adoption_mode = source.get("adoptionMode")
            if not isinstance(adoption_mode, str) or not adoption_mode.strip():
                raise ValueError(
                    "bound Web manifest sourceWorld.adoptionMode must be a non-empty string"
                )
            expected = {
                "worldId": manifest.world_id,
                "runId": manifest.run_id,
                "manifestSha256": manifest_sha256,
            }
            actual = {key: source.get(key) for key in expected}
            if actual != expected:
                raise ValueError(
                    "bound Web manifest sourceWorld does not match canonical manifest: "
                    f"expected {expected}, found {actual}"
                )
        return manifest, manifest_sha256


class SceneCommandHttpServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        config: SceneCommandServiceConfig,
    ) -> None:
        self.config = config
        super().__init__(server_address, SceneCommandRequestHandler)


class SceneCommandHttpServerV6(SceneCommandHttpServer):
    address_family = socket.AF_INET6


class SceneCommandRequestHandler(BaseHTTPRequestHandler):
    server: SceneCommandHttpServer
    protocol_version = "HTTP/1.1"
    server_version = "Video2WorldSceneCommand/1.0"
    sys_version = ""

    def log_message(self, format: str, *args: object) -> None:
        # Never log request bodies; retain the standard concise request line/status log.
        super().log_message(format, *args)

    def _origin(self) -> str | None:
        values = self.headers.get_all("Origin", failobj=[])
        if len(values) > 1:
            raise RequestRejected(HTTPStatus.BAD_REQUEST, "multiple_origin_headers")
        return values[0] if values else None

    def _cors_origin(self) -> str | None:
        origin = self._origin()
        if origin is None:
            return None
        if origin not in self.server.config.allowed_origins:
            raise RequestRejected(HTTPStatus.FORBIDDEN, "origin_not_allowed")
        return origin

    def _send_json(
        self,
        status_code: int,
        payload: dict[str, Any],
        *,
        origin: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        body = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if origin is not None:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Credentials", "true")
            self.send_header("Vary", "Origin")
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _reject(self, rejection: RequestRejected) -> None:
        # Rejections can happen before the request body is consumed. Do not leave unread
        # bytes on a persistent connection where they could be parsed as another request.
        self.close_connection = True
        try:
            origin = self._origin()
        except RequestRejected:
            origin = None
        if origin not in self.server.config.allowed_origins:
            origin = None
        payload: dict[str, Any] = {
            "status": "blocked",
            "reason": rejection.reason,
        }
        if rejection.details:
            payload["details"] = rejection.details
        self._send_json(
            rejection.status_code,
            payload,
            origin=origin,
            extra_headers={"Connection": "close"},
        )

    @staticmethod
    def _valid_json_content_type(value: str | None) -> bool:
        if value is None:
            return False
        parts = [part.strip() for part in value.split(";")]
        if not parts or parts[0].casefold() != "application/json":
            return False
        if len(parts) == 1:
            return True
        if len(parts) != 2 or "=" not in parts[1]:
            return False
        name, raw_value = (item.strip() for item in parts[1].split("=", 1))
        charset = raw_value.strip('"').casefold()
        return name.casefold() == "charset" and charset == "utf-8"

    def _read_envelope(self) -> SceneCommandHttpEnvelope:
        if self.headers.get("Transfer-Encoding") is not None:
            raise RequestRejected(HTTPStatus.BAD_REQUEST, "transfer_encoding_not_supported")
        content_types = self.headers.get_all("Content-Type", failobj=[])
        if len(content_types) != 1 or not self._valid_json_content_type(content_types[0]):
            raise RequestRejected(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "content_type_must_be_json")
        values = self.headers.get_all("Content-Length", failobj=[])
        if len(values) != 1:
            raise RequestRejected(HTTPStatus.LENGTH_REQUIRED, "content_length_required")
        try:
            length = int(values[0])
        except ValueError as exc:
            raise RequestRejected(HTTPStatus.BAD_REQUEST, "invalid_content_length") from exc
        if length < 1:
            raise RequestRejected(HTTPStatus.BAD_REQUEST, "empty_json_body")
        if length > self.server.config.max_json_bytes:
            raise RequestRejected(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "json_body_too_large")
        body = self.rfile.read(length)
        if len(body) != length:
            raise RequestRejected(HTTPStatus.BAD_REQUEST, "incomplete_request_body")
        try:
            payload = json.loads(
                body,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_json_object_without_duplicates,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
            raise RequestRejected(HTTPStatus.BAD_REQUEST, "invalid_json") from exc
        if not isinstance(payload, dict):
            raise RequestRejected(HTTPStatus.UNPROCESSABLE_ENTITY, "json_body_must_be_object")
        try:
            return SceneCommandHttpEnvelope.model_validate(payload)
        except ValidationError as exc:
            raise RequestRejected(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "invalid_scene_command_envelope",
                details=exc.errors(include_url=False, include_input=False),
            ) from exc

    def _authoritative_command(
        self,
        envelope: SceneCommandHttpEnvelope,
        manifest_sha256: str,
    ) -> SceneCommand:
        expected = envelope.expected_manifest_sha256
        # Ignore the command's embedded lock. The transport-level expected hash is checked
        # against the server-loaded manifest, and the authoritative command is rebuilt here.
        authoritative_hash = expected if expected is not None else manifest_sha256
        return envelope.command.model_copy(update={"expected_manifest_sha256": authoritative_hash})

    def _validate_idempotency_header(self, command: SceneCommand) -> None:
        values = self.headers.get_all("Idempotency-Key", failobj=[])
        if len(values) > 1:
            raise RequestRejected(HTTPStatus.BAD_REQUEST, "multiple_idempotency_keys")
        if values and values[0] != command.request_id:
            raise RequestRejected(HTTPStatus.CONFLICT, "idempotency_key_mismatch")

    def do_OPTIONS(self) -> None:
        try:
            origin = self._cors_origin()
            path = urlsplit(self.path).path
            if path not in {PLAN_PATH, SUBMIT_PATH}:
                raise RequestRejected(HTTPStatus.NOT_FOUND, "route_not_found")
            requested_method = self.headers.get("Access-Control-Request-Method")
            if requested_method != "POST":
                raise RequestRejected(HTTPStatus.METHOD_NOT_ALLOWED, "cors_method_not_allowed")
            requested_headers = {
                item.strip().casefold()
                for item in self.headers.get("Access-Control-Request-Headers", "").split(",")
                if item.strip()
            }
            allowed_headers = {"content-type", "idempotency-key"}
            if not requested_headers <= allowed_headers:
                raise RequestRejected(HTTPStatus.FORBIDDEN, "cors_headers_not_allowed")
            self._send_json(
                HTTPStatus.OK,
                {"status": "ok"},
                origin=origin,
                extra_headers={
                    "Access-Control-Allow-Methods": "POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type, Idempotency-Key",
                    "Access-Control-Max-Age": "600",
                },
            )
        except RequestRejected as rejection:
            self._reject(rejection)

    def do_GET(self) -> None:
        try:
            origin = self._cors_origin()
            parsed = urlsplit(self.path)
            if parsed.path != HEALTH_PATH or parsed.query:
                raise RequestRejected(HTTPStatus.NOT_FOUND, "route_not_found")
            manifest, manifest_sha256 = self.server.config.load_manifest()
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "worldId": manifest.world_id,
                    "manifestSha256": manifest_sha256,
                },
                origin=origin,
            )
        except RequestRejected as rejection:
            self._reject(rejection)
        except (OSError, ValidationError, ValueError):
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"status": "blocked", "reason": "server_manifest_unavailable"},
                origin=origin,
            )

    def do_POST(self) -> None:
        try:
            origin = self._cors_origin()
            parsed = urlsplit(self.path)
            if parsed.query or parsed.path not in {PLAN_PATH, SUBMIT_PATH}:
                raise RequestRejected(HTTPStatus.NOT_FOUND, "route_not_found")
            envelope = self._read_envelope()
            self._validate_idempotency_header(envelope.command)
            manifest, manifest_sha256 = self.server.config.load_manifest()
            command = self._authoritative_command(envelope, manifest_sha256)
            if parsed.path == PLAN_PATH:
                self._handle_plan(origin, manifest, manifest_sha256, command)
            else:
                self._handle_submit(origin, manifest, manifest_sha256, command, envelope)
        except RequestRejected as rejection:
            self._reject(rejection)
        except (OSError, ValidationError, ValueError):
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"status": "blocked", "reason": "scene_command_service_unavailable"},
                origin=origin,
            )

    def _handle_plan(
        self,
        origin: str | None,
        manifest: WorldManifest,
        manifest_sha256: str,
        command: SceneCommand,
    ) -> None:
        plan = plan_scene_command(manifest, command)
        ready = plan.status == "ready"
        self._send_json(
            HTTPStatus.OK,
            {
                "status": "ready" if ready else "blocked",
                "manifestSha256": manifest_sha256,
                "plan": plan.model_dump(mode="json", exclude_none=True),
                "confirmationRequired": plan.confirmation.required,
                "confirmationPhrase": plan.confirmation.required_phrase,
                "reason": None if ready else _blocked_reason(plan),
            },
            origin=origin,
        )

    def _handle_submit(
        self,
        origin: str | None,
        manifest: WorldManifest,
        manifest_sha256: str,
        command: SceneCommand,
        envelope: SceneCommandHttpEnvelope,
    ) -> None:
        plan = plan_scene_command(manifest, command)
        if plan.status != "ready":
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "blocked",
                    "requestId": command.request_id,
                    "manifestSha256": manifest_sha256,
                    "planId": plan.plan_id,
                    "idempotencyKey": plan.idempotency_key,
                    "reason": _blocked_reason(plan),
                },
                origin=origin,
            )
            return
        expected_phrase = plan.confirmation.required_phrase
        if plan.confirmation.required and envelope.confirmation_phrase != expected_phrase:
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "blocked",
                    "requestId": command.request_id,
                    "manifestSha256": manifest_sha256,
                    "planId": plan.plan_id,
                    "idempotencyKey": plan.idempotency_key,
                    "confirmationRequired": True,
                    "confirmationPhrase": expected_phrase,
                    "reason": "plan_specific_confirmation_phrase_required",
                },
                origin=origin,
            )
            return
        receipt = submit_scene_command(
            manifest,
            command,
            queue_dir=self.server.config.queue_dir,
            confirmation_phrase=envelope.confirmation_phrase,
        )
        payload: dict[str, Any] = {
            "status": receipt.status,
            "requestId": receipt.request_id,
            "planId": receipt.plan_id,
            "idempotencyKey": receipt.idempotency_key,
            "manifestSha256": manifest_sha256,
        }
        if receipt.job_id is not None:
            payload["jobId"] = receipt.job_id
        if receipt.read_result is not None:
            payload["readResult"] = receipt.read_result
        if receipt.blocked_reason is not None:
            payload["reason"] = receipt.blocked_reason
        self._send_json(HTTPStatus.OK, payload, origin=origin)


class RequestRejected(Exception):
    def __init__(
        self,
        status_code: int,
        reason: str,
        *,
        details: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(reason)
        self.status_code = status_code
        self.reason = reason
        self.details = details


def _fixed_path(path: str | Path, *, must_exist: bool) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ValueError(f"fixed service paths cannot be symlinks: {candidate}")
    resolved = candidate.resolve()
    if must_exist and not resolved.is_file():
        raise ValueError(f"manifest must be a regular file: {resolved}")
    return resolved


def create_scene_command_server(
    *,
    manifest_path: str | Path,
    web_manifest_path: str | Path | None = None,
    queue_dir: str | Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    allowed_origins: list[str] | tuple[str, ...] = (),
    max_json_bytes: int = DEFAULT_MAX_JSON_BYTES,
) -> SceneCommandHttpServer:
    """Create a configured server. ``port=0`` selects an ephemeral test port."""

    if not 0 <= port <= 65_535:
        raise ValueError("port must be between 0 and 65535")
    manifest = _fixed_path(manifest_path, must_exist=True)
    web_manifest = (
        _fixed_path(web_manifest_path, must_exist=True) if web_manifest_path is not None else None
    )
    queue_candidate = Path(queue_dir).expanduser()
    if queue_candidate.is_symlink():
        raise ValueError(f"fixed service paths cannot be symlinks: {queue_candidate}")
    queue = queue_candidate.resolve()
    config = SceneCommandServiceConfig(
        manifest_path=manifest,
        web_manifest_path=web_manifest,
        queue_dir=queue,
        allowed_origins=frozenset(allowed_origins),
        max_json_bytes=max_json_bytes,
    )
    # Fail before binding if the fixed manifest is invalid.
    config.load_manifest()
    queue.mkdir(parents=True, exist_ok=True, mode=0o700)
    server_type = SceneCommandHttpServerV6 if ":" in host else SceneCommandHttpServer
    return server_type((host, port), config)


def serve_scene_commands(server: SceneCommandHttpServer) -> None:
    """Serve until interrupted, closing the listening socket on exit."""

    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()


def run_server_in_thread(server: SceneCommandHttpServer) -> threading.Thread:
    """Start a daemon thread for focused tests and local embedding."""

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread
