from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from http import HTTPStatus
from pathlib import Path
from threading import Thread
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from video2world.cli import build_parser
from video2world.hashing import digest_json
from video2world.models import WorldManifest
from video2world.scene_command_server import (
    HEALTH_PATH,
    PLAN_PATH,
    SUBMIT_PATH,
    create_scene_command_server,
)
from video2world.scene_commands import (
    CommandProvenance,
    DeleteObjectIntent,
    QueryLocationIntent,
    SceneCommand,
    TargetSelector,
)

ALLOWED_ORIGIN = "http://127.0.0.1:4177"
NOW = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)


def _command(intent: Any, *, request_id: str) -> SceneCommand:
    return SceneCommand(
        request_id=request_id,
        provenance=CommandProvenance(
            requested_at=NOW,
            requester="video2world-web-demo",
            raw_prompt="服务端必须重新解析这个结构化指令",
            parser_provider="rule_based",
        ),
        intent=intent,
    )


def _envelope(
    command: SceneCommand,
    *,
    expected_manifest_sha256: str | None = None,
    confirmation_phrase: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "command": command.model_dump(mode="json", exclude_none=True),
        "structuredIntent": command.intent.model_dump(mode="json", exclude_none=True),
        "rawPrompt": command.provenance.raw_prompt,
        "expectedManifestSha256": expected_manifest_sha256,
        # Deliberately wrong: the server must not use this preview to plan targets/stages.
        "clientPreview": {
            "status": "ready",
            "previewOnly": True,
            "targetIds": ["client_lied_about_target"],
            "affectedStages": ["web"],
        },
    }
    if confirmation_phrase is not None:
        payload["confirmationPhrase"] = confirmation_phrase
    return payload


def _request_json(
    base_url: str,
    path: str,
    *,
    method: str = "POST",
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    content_type: str = "application/json; charset=utf-8",
) -> tuple[int, dict[str, Any], dict[str, str]]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request_headers = {"Accept": "application/json", **(headers or {})}
    if data is not None:
        request_headers["Content-Type"] = content_type
    request = Request(
        f"{base_url}{path}",
        data=data,
        headers=request_headers,
        method=method,
    )
    try:
        response = urlopen(request, timeout=5)
    except HTTPError as exc:
        body = json.loads(exc.read().decode("utf-8"))
        return exc.code, body, dict(exc.headers.items())
    with response:
        body = json.loads(response.read().decode("utf-8"))
        return response.status, body, dict(response.headers.items())


@contextmanager
def _running_server(
    tmp_path: Path,
    manifest: WorldManifest,
    *,
    max_json_bytes: int = 8_192,
) -> Iterator[tuple[str, Path, str]]:
    manifest_path = tmp_path / "fixed-world.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    queue_dir = tmp_path / "fixed-queue"
    server = create_scene_command_server(
        manifest_path=manifest_path,
        queue_dir=queue_dir,
        host="127.0.0.1",
        port=0,
        allowed_origins=[ALLOWED_ORIGIN],
        max_json_bytes=max_json_bytes,
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        yield f"http://{host}:{port}", queue_dir, digest_json(manifest.model_dump(mode="json"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_health_and_query_are_manifest_bound(
    tmp_path: Path,
    sample_manifest: WorldManifest,
) -> None:
    command = _command(
        QueryLocationIntent(target=TargetSelector(object_id="bed01"), language="zh"),
        request_id="query-bed-location",
    )
    with _running_server(tmp_path, sample_manifest) as (base_url, queue_dir, manifest_hash):
        status, health, _ = _request_json(base_url, HEALTH_PATH, method="GET")
        assert status == HTTPStatus.OK
        assert health == {
            "status": "ok",
            "worldId": "bedroom_4",
            "manifestSha256": manifest_hash,
        }

        status, planned, _ = _request_json(
            base_url,
            PLAN_PATH,
            payload=_envelope(command),
            headers={"Idempotency-Key": command.request_id},
        )
        assert status == HTTPStatus.OK
        assert planned["status"] == "ready"
        assert planned["manifestSha256"] == manifest_hash
        assert planned["confirmationRequired"] is False
        assert planned["plan"]["target_resolutions"][0]["resolved_object_id"] == "bed01"
        assert planned["plan"]["operations"][0]["target_object_ids"] == ["bed01"]

        status, submitted, _ = _request_json(
            base_url,
            SUBMIT_PATH,
            payload=_envelope(command),
            headers={"Idempotency-Key": command.request_id},
        )
        assert status == HTTPStatus.OK
        assert submitted["status"] == "completed_read"
        assert submitted["readResult"]["object_id"] == "bed01"
        assert not (queue_dir / "jobs").exists()


def test_ambiguous_target_is_blocked_without_queue_write(
    tmp_path: Path,
    sample_manifest: WorldManifest,
) -> None:
    command = _command(
        DeleteObjectIntent(target=TargetSelector(label="plant")),
        request_id="delete-ambiguous-plant",
    )
    with _running_server(tmp_path, sample_manifest) as (base_url, queue_dir, _):
        status, response, _ = _request_json(
            base_url,
            SUBMIT_PATH,
            payload=_envelope(command),
        )
        assert status == HTTPStatus.OK
        assert response["status"] == "blocked"
        assert "multiple instances" in response["reason"]
        assert not (queue_dir / "jobs").exists()


def test_plan_specific_confirmation_queues_once(
    tmp_path: Path,
    sample_manifest: WorldManifest,
) -> None:
    command = _command(
        DeleteObjectIntent(target=TargetSelector(object_id="table01")),
        request_id="delete-table-once",
    )
    with _running_server(tmp_path, sample_manifest) as (base_url, queue_dir, _):
        _, planned, _ = _request_json(base_url, PLAN_PATH, payload=_envelope(command))
        phrase = planned["confirmationPhrase"]
        assert planned["confirmationRequired"] is True
        assert phrase.startswith("confirm ")

        _, wrong, _ = _request_json(
            base_url,
            SUBMIT_PATH,
            payload=_envelope(command, confirmation_phrase="confirm wrongplan"),
        )
        assert wrong["status"] == "blocked"
        assert wrong["reason"] == "plan_specific_confirmation_phrase_required"
        assert wrong["confirmationPhrase"] == phrase
        assert wrong["requestId"] == command.request_id
        assert wrong["planId"] == planned["plan"]["plan_id"]
        assert wrong["idempotencyKey"] == planned["plan"]["idempotency_key"]
        assert not (queue_dir / "jobs").exists()

        _, queued, _ = _request_json(
            base_url,
            SUBMIT_PATH,
            payload=_envelope(command, confirmation_phrase=phrase),
        )
        _, duplicate, _ = _request_json(
            base_url,
            SUBMIT_PATH,
            payload=_envelope(command, confirmation_phrase=phrase),
        )
        assert queued["status"] == "queued"
        assert duplicate["status"] == "duplicate"
        assert queued["jobId"] == duplicate["jobId"]
        assert len(list((queue_dir / "jobs").glob("*.json"))) == 1
        job = json.loads(next((queue_dir / "jobs").glob("*.json")).read_text())
        assert job["status"] == "queued"
        assert job["attempt"] == 0
        assert job["plan"]["preview_only"] is True


def test_stale_expected_hash_is_blocked_by_server_manifest(
    tmp_path: Path,
    sample_manifest: WorldManifest,
) -> None:
    command = _command(
        DeleteObjectIntent(target=TargetSelector(object_id="table01")),
        request_id="delete-table-stale",
    )
    with _running_server(tmp_path, sample_manifest) as (base_url, queue_dir, actual_hash):
        _, response, _ = _request_json(
            base_url,
            PLAN_PATH,
            payload=_envelope(command, expected_manifest_sha256="f" * 64),
        )
        assert response["status"] == "blocked"
        assert response["manifestSha256"] == actual_hash
        assert response["plan"]["status"] == "blocked_stale_manifest"

        _, submitted, _ = _request_json(
            base_url,
            SUBMIT_PATH,
            payload=_envelope(command, expected_manifest_sha256="f" * 64),
        )
        assert submitted["status"] == "blocked"
        assert submitted["requestId"] == command.request_id
        assert submitted["planId"] == response["plan"]["plan_id"]
        assert submitted["idempotencyKey"] == response["plan"]["idempotency_key"]
        assert "expected_manifest_sha256" in submitted["reason"]
        assert not (queue_dir / "jobs").exists()


def test_exact_cors_content_type_and_size_limits(
    tmp_path: Path,
    sample_manifest: WorldManifest,
) -> None:
    command = _command(
        QueryLocationIntent(target=TargetSelector(object_id="bed01")),
        request_id="boundary-query",
    )
    with _running_server(
        tmp_path,
        sample_manifest,
        max_json_bytes=2_048,
    ) as (base_url, _, _):
        status, denied, _ = _request_json(
            base_url,
            PLAN_PATH,
            payload=_envelope(command),
            headers={"Origin": "https://attacker.invalid"},
        )
        assert status == HTTPStatus.FORBIDDEN
        assert denied == {"status": "blocked", "reason": "origin_not_allowed"}

        status, _, allowed_headers = _request_json(
            base_url,
            PLAN_PATH,
            payload=_envelope(command),
            headers={"Origin": ALLOWED_ORIGIN},
        )
        assert status == HTTPStatus.OK
        assert allowed_headers["Access-Control-Allow-Origin"] == ALLOWED_ORIGIN
        assert allowed_headers["Access-Control-Allow-Credentials"] == "true"

        status, rejected_type, _ = _request_json(
            base_url,
            PLAN_PATH,
            payload=_envelope(command),
            content_type="text/plain",
        )
        assert status == HTTPStatus.UNSUPPORTED_MEDIA_TYPE
        assert rejected_type["reason"] == "content_type_must_be_json"

        oversized = _envelope(command)
        oversized["clientPreview"]["padding"] = "x" * 4_096
        status, rejected_size, _ = _request_json(
            base_url,
            PLAN_PATH,
            payload=oversized,
        )
        assert status == HTTPStatus.REQUEST_ENTITY_TOO_LARGE
        assert rejected_size["reason"] == "json_body_too_large"


def test_cors_wildcard_and_non_loopback_cli_are_fail_closed(
    tmp_path: Path,
    sample_manifest: WorldManifest,
) -> None:
    manifest_path = tmp_path / "world.json"
    manifest_path.write_text(sample_manifest.model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="wildcard CORS"):
        server = create_scene_command_server(
            manifest_path=manifest_path,
            queue_dir=tmp_path / "queue",
            allowed_origins=["*"],
        )
        server.server_close()

    args = build_parser().parse_args(
        [
            "scene-command-serve",
            "--manifest",
            str(manifest_path),
            "--queue-dir",
            str(tmp_path / "queue"),
        ]
    )
    assert args.host == "127.0.0.1"
    assert args.port == 8765
    assert args.allow_non_loopback is False


def test_optional_web_manifest_binding_checks_world_run_and_canonical_hash(
    tmp_path: Path,
    sample_manifest: WorldManifest,
) -> None:
    manifest_path = tmp_path / "canonical.json"
    manifest_path.write_text(sample_manifest.model_dump_json(), encoding="utf-8")
    canonical_hash = digest_json(sample_manifest.model_dump(mode="json"))
    web_path = tmp_path / "web.json"
    web_payload = {
        "schemaVersion": 1,
        "contract": "video2world-web-manifest-1.0.0",
        "version": "bound-test-v1",
        "sourceWorld": {
            "worldId": sample_manifest.world_id,
            "runId": sample_manifest.run_id,
            "manifestSha256": canonical_hash,
            "adoptionMode": "test-bound-canonical",
        },
    }
    web_path.write_text(json.dumps(web_payload), encoding="utf-8")

    server = create_scene_command_server(
        manifest_path=manifest_path,
        web_manifest_path=web_path,
        queue_dir=tmp_path / "queue-ok",
        port=0,
    )
    server.server_close()

    web_payload["sourceWorld"]["manifestSha256"] = "f" * 64
    web_path.write_text(json.dumps(web_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match canonical manifest"):
        create_scene_command_server(
            manifest_path=manifest_path,
            web_manifest_path=web_path,
            queue_dir=tmp_path / "queue-rejected",
            port=0,
        )
    assert not (tmp_path / "queue-rejected").exists()


def test_real_bedroom_web_manifest_without_backlink_is_rejected_before_queue_creation(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    queue_dir = tmp_path / "queue"

    with pytest.raises(ValueError, match="does not match canonical manifest"):
        create_scene_command_server(
            manifest_path=root / "examples/bedroom4/world.manifest.local.json",
            web_manifest_path=root / "examples/bedroom4/manifest.production.json",
            queue_dir=queue_dir,
            port=0,
        )

    assert not queue_dir.exists()
