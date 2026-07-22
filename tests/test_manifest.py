from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from video2world.hashing import digest_path
from video2world.models import (
    Bounds3D,
    CoordinateSystem,
    GateRecord,
    InteractionPolicy,
    LocalizedText,
    QualityGates,
    SceneTransform,
    WorldManifest,
    WorldObject,
    world_manifest_json_schema,
)
from video2world.validation import validate_world_manifest

UNIFIED_GATE_REPORT_KINDS = {
    "alignment": "video2world.unified_gate.alignment",
    "collision": "video2world.unified_gate.collision",
    "visual": "video2world.unified_gate.visual",
}

REQUIRED_VISUAL_VIEWS = ("front", "right", "back", "left", "top", "bottom")


def _gate_report_payload(
    gate_name: str,
    asset_sha256: str,
    *,
    object_payload: dict[str, object] | None = None,
    kind: str | None = None,
    status: str = "passed",
    report_gate: str | None = None,
    object_id: str | None = "pillow-unified",
    target_id: str | None = None,
    scoped_id: str | None = "bedroom_4::pillow-unified-run::pillow-unified",
    asset_field: str = "unified_pbr_glb_sha256",
) -> str:
    payload: dict[str, object] = {
        "kind": kind or UNIFIED_GATE_REPORT_KINDS[gate_name],
        "gate": report_gate or gate_name,
        "status": status,
        asset_field: asset_sha256,
    }
    if scoped_id is not None:
        payload["scoped_id"] = scoped_id
    if object_id is not None:
        payload["object_id"] = object_id
    if target_id is not None:
        payload["target_id"] = target_id
    if object_payload is not None:
        quality_gates = object_payload["quality_gates"]
        assert isinstance(quality_gates, dict)
        gate = quality_gates[gate_name]
        assert isinstance(gate, dict)
        payload["metrics"] = gate.get("metrics", {})
    if object_payload is not None and gate_name == "alignment":
        payload["bbox_scene"] = object_payload["bbox_scene"]
        payload["transform_scene_from_asset"] = object_payload["transform_scene_from_asset"]
    if object_payload is not None and gate_name == "collision":
        unified_pbr_glb = object_payload["unified_pbr_glb"]
        assert isinstance(unified_pbr_glb, dict)
        provenance = unified_pbr_glb["provenance"]
        assert isinstance(provenance, dict)
        payload["collision_topology"] = object_payload["collision_topology"]
        payload["faces"] = provenance["faces"]
    if object_payload is not None and gate_name == "visual":
        payload["views"] = {
            view: {
                "uri": f"reviews/pillow-unified-{view}.png",
                "sha256": str(index) * 64,
            }
            for index, view in enumerate(REQUIRED_VISUAL_VIEWS, start=1)
        }
    return json.dumps(payload, sort_keys=True) + "\n"


def test_committed_json_schema_matches_pydantic_model() -> None:
    schema_path = Path(__file__).parents[1] / "schemas" / "world-manifest.schema.json"
    assert json.loads(schema_path.read_text(encoding="utf-8")) == world_manifest_json_schema()


def test_manifest_verifies_local_content_hashes(
    tmp_path: Path,
    sample_manifest: WorldManifest,
) -> None:
    manifest_path = tmp_path / "world.json"
    manifest_path.write_text(sample_manifest.model_dump_json(indent=2), encoding="utf-8")
    result = validate_world_manifest(manifest_path)
    assert result["valid"] is True
    assert result["checked_assets"] == 3

    Path(sample_manifest.scene.visual.uri).write_bytes(b"changed")
    invalid = validate_world_manifest(manifest_path)
    assert invalid["valid"] is False
    assert {issue["error"] for issue in invalid["issues"]} >= {
        "sha256 mismatch",
        "size_bytes mismatch",
    }


def test_manifest_verifies_local_gate_report_hashes(
    tmp_path: Path,
    sample_manifest: WorldManifest,
) -> None:
    object_payload = _unified_pbr_object_payload(sample_manifest)
    glb_path = tmp_path / "pillow-unified.glb"
    glb_path.write_bytes(b"unified pbr glb")
    glb_digest = digest_path(glb_path)
    object_payload["unified_pbr_glb"].update(
        {
            "uri": str(glb_path),
            "sha256": glb_digest.sha256,
            "size_bytes": glb_digest.size_bytes,
        }
    )
    for gate_name in ("alignment", "collision", "visual"):
        report_path = tmp_path / f"{gate_name}-report.json"
        report_path.write_text(
            _gate_report_payload(
                gate_name,
                glb_digest.sha256,
                object_payload=object_payload,
            )
        )
        report_digest = digest_path(report_path)
        object_payload["quality_gates"][gate_name].update(
            {
                "report_uri": str(report_path),
                "report_sha256": report_digest.sha256,
                "report_size_bytes": report_digest.size_bytes,
            }
        )
    payload = sample_manifest.model_dump(mode="json")
    payload["objects"].append(object_payload)
    manifest = WorldManifest.model_validate(payload)
    manifest_path = tmp_path / "world.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    result = validate_world_manifest(manifest_path)
    assert result["valid"] is True
    assert result["checked_gate_reports"] == 3

    (tmp_path / "visual-report.json").write_text('{"gate":"visual","status":"changed"}\n')
    invalid = validate_world_manifest(manifest_path)
    assert invalid["valid"] is False
    assert {issue["error"] for issue in invalid["issues"]} >= {
        "report_sha256 mismatch",
        "report_size_bytes mismatch",
    }


def test_manifest_requires_unified_gate_report_identity_bindings(
    tmp_path: Path,
    sample_manifest: WorldManifest,
) -> None:
    object_payload = _unified_pbr_object_payload(sample_manifest)
    glb_path = tmp_path / "pillow-unified.glb"
    glb_path.write_bytes(b"unified pbr glb")
    glb_digest = digest_path(glb_path)
    object_payload["unified_pbr_glb"].update(
        {
            "uri": str(glb_path),
            "sha256": glb_digest.sha256,
            "size_bytes": glb_digest.size_bytes,
        }
    )
    for gate_name in ("alignment", "collision", "visual"):
        report_path = tmp_path / f"{gate_name}-report.json"
        report_payload = {
            "gate": gate_name,
            "status": "passed",
            "unified_pbr_glb_sha256": glb_digest.sha256,
        }
        report_path.write_text(json.dumps(report_payload, sort_keys=True) + "\n")
        report_digest = digest_path(report_path)
        object_payload["quality_gates"][gate_name].update(
            {
                "report_uri": str(report_path),
                "report_sha256": report_digest.sha256,
                "report_size_bytes": report_digest.size_bytes,
            }
        )
    payload = sample_manifest.model_dump(mode="json")
    payload["objects"].append(object_payload)
    manifest = WorldManifest.model_validate(payload)
    manifest_path = tmp_path / "world.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    result = validate_world_manifest(manifest_path)

    assert result["valid"] is False
    assert any(
        "missing required bindings" in issue["error"]
        and "kind" in issue["error"]
        and "scoped_id" in issue["error"]
        for issue in result["issues"]
    )


def test_manifest_rejects_unified_gate_report_kind_mismatch(
    tmp_path: Path,
    sample_manifest: WorldManifest,
) -> None:
    object_payload = _unified_pbr_object_payload(sample_manifest)
    glb_path = tmp_path / "pillow-unified.glb"
    glb_path.write_bytes(b"unified pbr glb")
    glb_digest = digest_path(glb_path)
    object_payload["unified_pbr_glb"].update(
        {
            "uri": str(glb_path),
            "sha256": glb_digest.sha256,
            "size_bytes": glb_digest.size_bytes,
        }
    )
    for gate_name in ("alignment", "collision", "visual"):
        report_path = tmp_path / f"{gate_name}-report.json"
        report_kind = (
            UNIFIED_GATE_REPORT_KINDS["visual"]
            if gate_name == "collision"
            else UNIFIED_GATE_REPORT_KINDS[gate_name]
        )
        report_path.write_text(
            _gate_report_payload(
                gate_name,
                glb_digest.sha256,
                object_payload=object_payload,
                kind=report_kind,
            )
        )
        report_digest = digest_path(report_path)
        object_payload["quality_gates"][gate_name].update(
            {
                "report_uri": str(report_path),
                "report_sha256": report_digest.sha256,
                "report_size_bytes": report_digest.size_bytes,
            }
        )
    payload = sample_manifest.model_dump(mode="json")
    payload["objects"].append(object_payload)
    manifest = WorldManifest.model_validate(payload)
    manifest_path = tmp_path / "world.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    result = validate_world_manifest(manifest_path)

    assert result["valid"] is False
    assert any(
        "kind" in issue["error"] and "does not match expected" in issue["error"]
        for issue in result["issues"]
    )


def test_manifest_rejects_alignment_report_geometry_mismatch(
    tmp_path: Path,
    sample_manifest: WorldManifest,
) -> None:
    object_payload = _unified_pbr_object_payload(sample_manifest)
    glb_path = tmp_path / "pillow-unified.glb"
    glb_path.write_bytes(b"unified pbr glb")
    glb_digest = digest_path(glb_path)
    object_payload["unified_pbr_glb"].update(
        {
            "uri": str(glb_path),
            "sha256": glb_digest.sha256,
            "size_bytes": glb_digest.size_bytes,
        }
    )
    for gate_name in ("alignment", "collision", "visual"):
        report_path = tmp_path / f"{gate_name}-report.json"
        report_payload = json.loads(
            _gate_report_payload(
                gate_name,
                glb_digest.sha256,
                object_payload=object_payload,
            )
        )
        if gate_name == "alignment":
            report_payload["transform_scene_from_asset"]["pivot_scene"] = [9, 9, 9]
        report_path.write_text(json.dumps(report_payload, sort_keys=True) + "\n")
        report_digest = digest_path(report_path)
        object_payload["quality_gates"][gate_name].update(
            {
                "report_uri": str(report_path),
                "report_sha256": report_digest.sha256,
                "report_size_bytes": report_digest.size_bytes,
            }
        )
    payload = sample_manifest.model_dump(mode="json")
    payload["objects"].append(object_payload)
    manifest = WorldManifest.model_validate(payload)
    manifest_path = tmp_path / "world.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    result = validate_world_manifest(manifest_path)

    assert result["valid"] is False
    assert any(
        "transform_scene_from_asset does not match" in issue["error"]
        for issue in result["issues"]
    )


def test_manifest_rejects_collision_report_topology_or_face_mismatch(
    tmp_path: Path,
    sample_manifest: WorldManifest,
) -> None:
    object_payload = _unified_pbr_object_payload(sample_manifest)
    glb_path = tmp_path / "pillow-unified.glb"
    glb_path.write_bytes(b"unified pbr glb")
    glb_digest = digest_path(glb_path)
    object_payload["unified_pbr_glb"].update(
        {
            "uri": str(glb_path),
            "sha256": glb_digest.sha256,
            "size_bytes": glb_digest.size_bytes,
        }
    )
    for gate_name in ("alignment", "collision", "visual"):
        report_path = tmp_path / f"{gate_name}-report.json"
        report_payload = json.loads(
            _gate_report_payload(
                gate_name,
                glb_digest.sha256,
                object_payload=object_payload,
            )
        )
        if gate_name == "collision":
            report_payload["faces"] = 1
        report_path.write_text(json.dumps(report_payload, sort_keys=True) + "\n")
        report_digest = digest_path(report_path)
        object_payload["quality_gates"][gate_name].update(
            {
                "report_uri": str(report_path),
                "report_sha256": report_digest.sha256,
                "report_size_bytes": report_digest.size_bytes,
            }
        )
    payload = sample_manifest.model_dump(mode="json")
    payload["objects"].append(object_payload)
    manifest = WorldManifest.model_validate(payload)
    manifest_path = tmp_path / "world.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    result = validate_world_manifest(manifest_path)

    assert result["valid"] is False
    assert any(
        "face count" in issue["error"] and "does not match" in issue["error"]
        for issue in result["issues"]
    )


def test_manifest_rejects_collision_report_metric_mismatch(
    tmp_path: Path,
    sample_manifest: WorldManifest,
) -> None:
    object_payload = _unified_pbr_object_payload(sample_manifest)
    glb_path = tmp_path / "pillow-unified.glb"
    glb_path.write_bytes(b"unified pbr glb")
    glb_digest = digest_path(glb_path)
    object_payload["unified_pbr_glb"].update(
        {
            "uri": str(glb_path),
            "sha256": glb_digest.sha256,
            "size_bytes": glb_digest.size_bytes,
        }
    )
    for gate_name in ("alignment", "collision", "visual"):
        report_path = tmp_path / f"{gate_name}-report.json"
        report_payload = json.loads(
            _gate_report_payload(
                gate_name,
                glb_digest.sha256,
                object_payload=object_payload,
            )
        )
        if gate_name == "collision":
            report_payload["metrics"]["stale_static_intersection_faces"] = 99
        report_path.write_text(json.dumps(report_payload, sort_keys=True) + "\n")
        report_digest = digest_path(report_path)
        object_payload["quality_gates"][gate_name].update(
            {
                "report_uri": str(report_path),
                "report_sha256": report_digest.sha256,
                "report_size_bytes": report_digest.size_bytes,
            }
        )
    payload = sample_manifest.model_dump(mode="json")
    payload["objects"].append(object_payload)
    manifest = WorldManifest.model_validate(payload)
    manifest_path = tmp_path / "world.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    result = validate_world_manifest(manifest_path)

    assert result["valid"] is False
    assert any("metrics do not match" in issue["error"] for issue in result["issues"])


@pytest.mark.parametrize(
    ("gate_to_mutate", "metric_name", "mutated_value"),
    [
        ("alignment", "stable_support_contact", False),
        ("visual", "six_view_count", 5),
    ],
)
def test_manifest_rejects_alignment_or_visual_report_metric_mismatch(
    tmp_path: Path,
    sample_manifest: WorldManifest,
    gate_to_mutate: str,
    metric_name: str,
    mutated_value: object,
) -> None:
    object_payload = _unified_pbr_object_payload(sample_manifest)
    glb_path = tmp_path / "pillow-unified.glb"
    glb_path.write_bytes(b"unified pbr glb")
    glb_digest = digest_path(glb_path)
    object_payload["unified_pbr_glb"].update(
        {
            "uri": str(glb_path),
            "sha256": glb_digest.sha256,
            "size_bytes": glb_digest.size_bytes,
        }
    )
    for gate_name in ("alignment", "collision", "visual"):
        report_path = tmp_path / f"{gate_name}-report.json"
        report_payload = json.loads(
            _gate_report_payload(
                gate_name,
                glb_digest.sha256,
                object_payload=object_payload,
            )
        )
        if gate_name == gate_to_mutate:
            report_payload["metrics"][metric_name] = mutated_value
        report_path.write_text(json.dumps(report_payload, sort_keys=True) + "\n")
        report_digest = digest_path(report_path)
        object_payload["quality_gates"][gate_name].update(
            {
                "report_uri": str(report_path),
                "report_sha256": report_digest.sha256,
                "report_size_bytes": report_digest.size_bytes,
            }
        )
    payload = sample_manifest.model_dump(mode="json")
    payload["objects"].append(object_payload)
    manifest = WorldManifest.model_validate(payload)
    manifest_path = tmp_path / "world.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    result = validate_world_manifest(manifest_path)

    assert result["valid"] is False
    assert any(
        f"{gate_to_mutate} gate report metrics do not match" in issue["error"]
        for issue in result["issues"]
    )


@pytest.mark.parametrize(
    ("gate_to_mutate", "metric_name", "mutated_value", "expected_error"),
    [
        ("alignment", "stable_support_contact", False, "stable_support_contact"),
        ("alignment", "obvious_interpenetration", True, "cannot be true"),
        ("collision", "bvh_probe_hits", 0, "bvh_probe_hits"),
        ("collision", "obvious_interpenetration", True, "cannot be true"),
        ("visual", "six_view_count", 5, "at least 6"),
        ("visual", "browser_canvas_nonblank", False, "browser_canvas_nonblank"),
        ("visual", "severe_identity_drift", True, "cannot be true"),
    ],
)
def test_manifest_rejects_passed_gate_blocking_metrics(
    tmp_path: Path,
    sample_manifest: WorldManifest,
    gate_to_mutate: str,
    metric_name: str,
    mutated_value: object,
    expected_error: str,
) -> None:
    object_payload = _unified_pbr_object_payload(sample_manifest)
    glb_path = tmp_path / "pillow-unified.glb"
    glb_path.write_bytes(b"unified pbr glb")
    glb_digest = digest_path(glb_path)
    object_payload["unified_pbr_glb"].update(
        {
            "uri": str(glb_path),
            "sha256": glb_digest.sha256,
            "size_bytes": glb_digest.size_bytes,
        }
    )
    quality_gates = object_payload["quality_gates"]
    assert isinstance(quality_gates, dict)
    gate = quality_gates[gate_to_mutate]
    assert isinstance(gate, dict)
    metrics = gate["metrics"]
    assert isinstance(metrics, dict)
    metrics[metric_name] = mutated_value
    for gate_name in ("alignment", "collision", "visual"):
        report_path = tmp_path / f"{gate_name}-report.json"
        report_path.write_text(
            _gate_report_payload(
                gate_name,
                glb_digest.sha256,
                object_payload=object_payload,
            )
        )
        report_digest = digest_path(report_path)
        object_payload["quality_gates"][gate_name].update(
            {
                "report_uri": str(report_path),
                "report_sha256": report_digest.sha256,
                "report_size_bytes": report_digest.size_bytes,
            }
        )
    payload = sample_manifest.model_dump(mode="json")
    payload["objects"].append(object_payload)
    manifest = WorldManifest.model_validate(payload)
    manifest_path = tmp_path / "world.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    result = validate_world_manifest(manifest_path)

    assert result["valid"] is False
    assert any(expected_error in issue["error"] for issue in result["issues"])


@pytest.mark.parametrize(
    ("view_mutation", "expected_error"),
    [
        ("missing_top", "missing required six-view evidence"),
        ("empty_back", "empty six-view evidence"),
    ],
)
def test_manifest_rejects_incomplete_visual_six_view_evidence(
    tmp_path: Path,
    sample_manifest: WorldManifest,
    view_mutation: str,
    expected_error: str,
) -> None:
    object_payload = _unified_pbr_object_payload(sample_manifest)
    glb_path = tmp_path / "pillow-unified.glb"
    glb_path.write_bytes(b"unified pbr glb")
    glb_digest = digest_path(glb_path)
    object_payload["unified_pbr_glb"].update(
        {
            "uri": str(glb_path),
            "sha256": glb_digest.sha256,
            "size_bytes": glb_digest.size_bytes,
        }
    )
    for gate_name in ("alignment", "collision", "visual"):
        report_path = tmp_path / f"{gate_name}-report.json"
        report_payload = json.loads(
            _gate_report_payload(
                gate_name,
                glb_digest.sha256,
                object_payload=object_payload,
            )
        )
        if gate_name == "visual":
            views = report_payload["views"]
            assert isinstance(views, dict)
            if view_mutation == "missing_top":
                del views["top"]
            if view_mutation == "empty_back":
                views["back"] = {}
        report_path.write_text(json.dumps(report_payload, sort_keys=True) + "\n")
        report_digest = digest_path(report_path)
        object_payload["quality_gates"][gate_name].update(
            {
                "report_uri": str(report_path),
                "report_sha256": report_digest.sha256,
                "report_size_bytes": report_digest.size_bytes,
            }
        )
    payload = sample_manifest.model_dump(mode="json")
    payload["objects"].append(object_payload)
    manifest = WorldManifest.model_validate(payload)
    manifest_path = tmp_path / "world.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    result = validate_world_manifest(manifest_path)

    assert result["valid"] is False
    assert any(expected_error in issue["error"] for issue in result["issues"])


def test_manifest_rejects_non_json_gate_report_payload(
    tmp_path: Path,
    sample_manifest: WorldManifest,
) -> None:
    object_payload = _unified_pbr_object_payload(sample_manifest)
    glb_path = tmp_path / "pillow-unified.glb"
    glb_path.write_bytes(b"unified pbr glb")
    glb_digest = digest_path(glb_path)
    object_payload["unified_pbr_glb"].update(
        {
            "uri": str(glb_path),
            "sha256": glb_digest.sha256,
            "size_bytes": glb_digest.size_bytes,
        }
    )
    for gate_name in ("alignment", "collision", "visual"):
        report_path = tmp_path / f"{gate_name}-report.json"
        payload = _gate_report_payload(
            gate_name,
            glb_digest.sha256,
            object_payload=object_payload,
        )
        if gate_name == "collision":
            payload = "not json\n"
        report_path.write_text(payload)
        report_digest = digest_path(report_path)
        object_payload["quality_gates"][gate_name].update(
            {
                "report_uri": str(report_path),
                "report_sha256": report_digest.sha256,
                "report_size_bytes": report_digest.size_bytes,
            }
        )
    payload = sample_manifest.model_dump(mode="json")
    payload["objects"].append(object_payload)
    manifest = WorldManifest.model_validate(payload)
    manifest_path = tmp_path / "world.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    result = validate_world_manifest(manifest_path)

    assert result["valid"] is False
    assert any("not readable JSON" in issue["error"] for issue in result["issues"])


def test_manifest_rejects_gate_report_status_mismatch(
    tmp_path: Path,
    sample_manifest: WorldManifest,
) -> None:
    object_payload = _unified_pbr_object_payload(sample_manifest)
    glb_path = tmp_path / "pillow-unified.glb"
    glb_path.write_bytes(b"unified pbr glb")
    glb_digest = digest_path(glb_path)
    object_payload["unified_pbr_glb"].update(
        {
            "uri": str(glb_path),
            "sha256": glb_digest.sha256,
            "size_bytes": glb_digest.size_bytes,
        }
    )
    for gate_name in ("alignment", "collision", "visual"):
        report_path = tmp_path / f"{gate_name}-report.json"
        status = "failed" if gate_name == "visual" else "passed"
        report_path.write_text(
            _gate_report_payload(
                gate_name,
                glb_digest.sha256,
                object_payload=object_payload,
                status=status,
            )
        )
        report_digest = digest_path(report_path)
        object_payload["quality_gates"][gate_name].update(
            {
                "report_uri": str(report_path),
                "report_sha256": report_digest.sha256,
                "report_size_bytes": report_digest.size_bytes,
            }
        )
    payload = sample_manifest.model_dump(mode="json")
    payload["objects"].append(object_payload)
    manifest = WorldManifest.model_validate(payload)
    manifest_path = tmp_path / "world.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    result = validate_world_manifest(manifest_path)

    assert result["valid"] is False
    assert any("does not match gate status" in issue["error"] for issue in result["issues"])


def test_manifest_rejects_gate_report_role_mismatch(
    tmp_path: Path,
    sample_manifest: WorldManifest,
) -> None:
    object_payload = _unified_pbr_object_payload(sample_manifest)
    glb_path = tmp_path / "pillow-unified.glb"
    glb_path.write_bytes(b"unified pbr glb")
    glb_digest = digest_path(glb_path)
    object_payload["unified_pbr_glb"].update(
        {
            "uri": str(glb_path),
            "sha256": glb_digest.sha256,
            "size_bytes": glb_digest.size_bytes,
        }
    )
    for gate_name in ("alignment", "collision", "visual"):
        report_path = tmp_path / f"{gate_name}-report.json"
        report_gate = "visual" if gate_name == "collision" else gate_name
        report_path.write_text(
            _gate_report_payload(
                gate_name,
                glb_digest.sha256,
                object_payload=object_payload,
                report_gate=report_gate,
            )
        )
        report_digest = digest_path(report_path)
        object_payload["quality_gates"][gate_name].update(
            {
                "report_uri": str(report_path),
                "report_sha256": report_digest.sha256,
                "report_size_bytes": report_digest.size_bytes,
            }
        )
    payload = sample_manifest.model_dump(mode="json")
    payload["objects"].append(object_payload)
    manifest = WorldManifest.model_validate(payload)
    manifest_path = tmp_path / "world.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    result = validate_world_manifest(manifest_path)

    assert result["valid"] is False
    assert any("does not match manifest gate" in issue["error"] for issue in result["issues"])


def test_manifest_rejects_gate_report_object_mismatch(
    tmp_path: Path,
    sample_manifest: WorldManifest,
) -> None:
    object_payload = _unified_pbr_object_payload(sample_manifest)
    glb_path = tmp_path / "pillow-unified.glb"
    glb_path.write_bytes(b"unified pbr glb")
    glb_digest = digest_path(glb_path)
    object_payload["unified_pbr_glb"].update(
        {
            "uri": str(glb_path),
            "sha256": glb_digest.sha256,
            "size_bytes": glb_digest.size_bytes,
        }
    )
    for gate_name in ("alignment", "collision", "visual"):
        report_path = tmp_path / f"{gate_name}-report.json"
        report_object_id = "other-object" if gate_name == "collision" else "pillow-unified"
        report_path.write_text(
            _gate_report_payload(
                gate_name,
                glb_digest.sha256,
                object_payload=object_payload,
                object_id=report_object_id,
            )
        )
        report_digest = digest_path(report_path)
        object_payload["quality_gates"][gate_name].update(
            {
                "report_uri": str(report_path),
                "report_sha256": report_digest.sha256,
                "report_size_bytes": report_digest.size_bytes,
            }
        )
    payload = sample_manifest.model_dump(mode="json")
    payload["objects"].append(object_payload)
    manifest = WorldManifest.model_validate(payload)
    manifest_path = tmp_path / "world.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    result = validate_world_manifest(manifest_path)

    assert result["valid"] is False
    assert any("does not match manifest object" in issue["error"] for issue in result["issues"])


def test_manifest_rejects_gate_report_asset_mismatch(
    tmp_path: Path,
    sample_manifest: WorldManifest,
) -> None:
    object_payload = _unified_pbr_object_payload(sample_manifest)
    glb_path = tmp_path / "pillow-unified.glb"
    glb_path.write_bytes(b"unified pbr glb")
    glb_digest = digest_path(glb_path)
    object_payload["unified_pbr_glb"].update(
        {
            "uri": str(glb_path),
            "sha256": glb_digest.sha256,
            "size_bytes": glb_digest.size_bytes,
        }
    )
    for gate_name in ("alignment", "collision", "visual"):
        report_path = tmp_path / f"{gate_name}-report.json"
        asset_sha = "0" * 64 if gate_name == "visual" else glb_digest.sha256
        report_path.write_text(
            _gate_report_payload(
                gate_name,
                asset_sha,
                object_payload=object_payload,
            )
        )
        report_digest = digest_path(report_path)
        object_payload["quality_gates"][gate_name].update(
            {
                "report_uri": str(report_path),
                "report_sha256": report_digest.sha256,
                "report_size_bytes": report_digest.size_bytes,
            }
        )
    payload = sample_manifest.model_dump(mode="json")
    payload["objects"].append(object_payload)
    manifest = WorldManifest.model_validate(payload)
    manifest_path = tmp_path / "world.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    result = validate_world_manifest(manifest_path)

    assert result["valid"] is False
    assert any(
        "does not match manifest unified_pbr_glb asset" in issue["error"]
        for issue in result["issues"]
    )


def test_scoped_identity_is_checked_against_world(sample_manifest: WorldManifest) -> None:
    payload = sample_manifest.model_dump(mode="json")
    payload["objects"][0]["scoped_id"] = "other::run::bed01"
    with pytest.raises(ValidationError, match="scoped_id"):
        WorldManifest.model_validate(payload)


def test_merged_component_cannot_own_child_assets(sample_manifest: WorldManifest) -> None:
    payload = sample_manifest.model_dump(mode="json")
    payload["objects"][0].update(
        {
            "semantic_granularity": "merged_component",
            "independently_movable": False,
        }
    )
    payload["objects"][1].update(
        {
            "semantic_granularity": "independent_child_asset",
            "parent_object_id": payload["objects"][0]["id"],
            "moves_with_parent": True,
        }
    )

    with pytest.raises(ValidationError, match=r"merged component.*cannot own child assets"):
        WorldManifest.model_validate(payload)


def test_manifest_timestamp_must_be_timezone_aware(sample_manifest: WorldManifest) -> None:
    payload = sample_manifest.model_dump(mode="json")
    payload["created_at"] = datetime(2026, 7, 16).isoformat()
    with pytest.raises(ValidationError, match="explicit timezone"):
        WorldManifest.model_validate(payload)


def test_manifest_rejects_non_finite_geometry(sample_manifest: WorldManifest) -> None:
    payload = sample_manifest.model_dump(mode="json")
    payload["objects"][0]["bbox_scene"]["minimum"][0] = float("nan")
    with pytest.raises(ValidationError, match="finite number"):
        WorldManifest.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("bbox_scene", "bbox is not in the scene frame"),
        ("obb_scene", "OBB is not in the scene frame"),
        ("transform_scene_from_asset", "transform does not target the scene frame"),
    ],
)
def test_object_geometry_must_use_the_scene_frame(
    sample_manifest: WorldManifest,
    field: str,
    message: str,
) -> None:
    payload = sample_manifest.model_dump(mode="json")
    object_payload = payload["objects"][0]
    if field == "obb_scene":
        object_payload[field] = {
            "frame_id": "other",
            "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            "extents": [1, 1, 1],
        }
    elif field == "transform_scene_from_asset":
        object_payload[field] = {
            "from_frame": "asset",
            "to_frame": "other",
            "matrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            "pivot_scene": [0, 0, 0],
            "scale_xyz": [1, 1, 1],
        }
    else:
        object_payload[field]["frame_id"] = "other"
    with pytest.raises(ValidationError, match=message):
        WorldManifest.model_validate(payload)


def test_non_metric_scene_scale_is_explicit() -> None:
    coordinate_system = CoordinateSystem(
        frame_id="pgsr_native",
        up_axis="-Y",
        handedness="right",
        units="scene_scale_not_metric",
    )
    assert coordinate_system.metric_scale is None
    with pytest.raises(ValidationError, match="cannot declare a metric scale"):
        CoordinateSystem(
            frame_id="pgsr_native",
            up_axis="-Y",
            handedness="right",
            units="scene_scale_not_metric",
            metric_scale=1.0,
        )


def test_interactive_objects_require_assets_and_passed_gates() -> None:
    with pytest.raises(ValidationError, match="interactive object is missing"):
        WorldObject(
            id="plant01",
            source_run_id="run",
            scoped_id="scene::run::plant01",
            name=LocalizedText(en="plant"),
            category="plant",
            interaction=InteractionPolicy(selectable=True, double_click_action="spin_360"),
        )


def _visual_only_object(sample_manifest: WorldManifest) -> WorldObject:
    frame_id = sample_manifest.scene.coordinate_system.frame_id
    return WorldObject(
        id="pillow01",
        source_run_id="pillow-run",
        scoped_id="bedroom_4::pillow-run::pillow01",
        name=LocalizedText(zh="枕头", en="pillow"),
        category="pillow",
        bbox_scene=Bounds3D(
            frame_id=frame_id,
            minimum=(-1, 0, -1),
            maximum=(1, 1, 1),
        ),
        visual=sample_manifest.scene.visual,
        transform_scene_from_asset=SceneTransform(
            from_frame="pillow_points",
            to_frame=frame_id,
            matrix=(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1),
            pivot_scene=(0, 0.5, 0),
        ),
        quality_gates=QualityGates(
            file=GateRecord(status="passed"),
            semantic=GateRecord(status="passed"),
            alignment=GateRecord(status="passed"),
            collision=GateRecord(status="not_tested", reason="no collider exists"),
            visual=GateRecord(status="passed"),
        ),
        interaction=InteractionPolicy(
            selectable=True,
            double_click_action="spin_360",
            drag_action="rotate_yaw",
            collision_enabled=False,
            physics_mode="none",
        ),
    )


def _unified_pbr_object_payload(sample_manifest: WorldManifest) -> dict[str, object]:
    payload = _visual_only_object(sample_manifest).model_dump(mode="json")
    payload["id"] = "pillow-unified"
    payload["source_run_id"] = "pillow-unified-run"
    payload["scoped_id"] = "bedroom_4::pillow-unified-run::pillow-unified"
    payload["visual"] = None
    payload["unified_pbr_glb"] = {
        "uri": "artifact://pillow-unified.glb",
        "sha256": "d" * 64,
        "size_bytes": 4096,
        "media_type": "model/gltf-binary",
        "role": "unified_pbr_glb",
        "status": "validated",
        "provenance": {
            "faces": 97082,
            "watertight": False,
            "closed_volume_claim": False,
            "inside_outside_queries_allowed": False,
            "technical_gates": {
                "finite_vertices": True,
                "valid_triangle_indices": True,
                "no_degenerate_faces": True,
                "winding_consistent": True,
                "pbr_material_present": True,
                "positive_extents": True,
            },
        },
    }
    payload["collision_topology"] = "surface_bvh"
    payload["quality_gates"]["alignment"] = {
        "status": "passed",
        "report_uri": "artifact://pillow-unified/scene-fit-report.json",
        "report_sha256": "e" * 64,
        "report_size_bytes": 2048,
        "metrics": {
            "stable_support_contact": True,
            "max_interpenetration_ratio": 0.01,
            "placement_fit_iou": 0.92,
        },
    }
    payload["quality_gates"]["collision"] = {
        "status": "passed",
        "report_uri": "artifact://pillow-unified/collision-report.json",
        "report_sha256": "f" * 64,
        "report_size_bytes": 2048,
        "metrics": {
            "bvh_probe_hits": 8,
            "stale_static_intersection_faces": 0,
            "obvious_interpenetration": False,
        },
    }
    payload["quality_gates"]["visual"] = {
        "status": "passed",
        "report_uri": "artifact://pillow-unified/visual-review.json",
        "report_sha256": "1" * 64,
        "report_size_bytes": 2048,
        "metrics": {
            "six_view_count": 6,
            "browser_canvas_nonblank": True,
            "severe_identity_drift": False,
        },
    }
    payload["interaction"].update(
        {
            "collision_enabled": True,
            "physics_mode": "kinematic",
        }
    )
    return payload


def test_visual_only_interaction_allows_no_collider_and_rejects_collision_claims(
    sample_manifest: WorldManifest,
) -> None:
    pillow = _visual_only_object(sample_manifest)
    assert pillow.collider is None
    assert pillow.quality_gates.collision.status == "not_tested"
    assert pillow.interaction.drag_action == "rotate_yaw"

    with_collider = pillow.model_dump(mode="json")
    with_collider["collider"] = sample_manifest.scene.collider.model_dump(mode="json")
    with pytest.raises(ValidationError, match="must not declare a collider"):
        WorldObject.model_validate(with_collider)

    passed_collision = pillow.model_dump(mode="json")
    passed_collision["quality_gates"]["collision"] = {"status": "passed"}
    with pytest.raises(ValidationError, match="must not pass the collision gate"):
        WorldObject.model_validate(passed_collision)


def test_mesh_first_interaction_does_not_require_gaussian_visual(
    sample_manifest: WorldManifest,
) -> None:
    payload = _visual_only_object(sample_manifest).model_dump(mode="json")
    payload["render_mesh"] = payload.pop("visual")

    mesh_first = WorldObject.model_validate(payload)

    assert mesh_first.visual is None
    assert mesh_first.render_mesh is not None
    assert mesh_first.render_mesh.status == "validated"

    payload["render_mesh"]["status"] = "candidate"
    with pytest.raises(ValidationError, match="every interactive visual representation"):
        WorldObject.model_validate(payload)


def test_unified_surface_bvh_accepts_nonwatertight_glb_without_collider_proxy(
    sample_manifest: WorldManifest,
) -> None:
    unified = WorldObject.model_validate(_unified_pbr_object_payload(sample_manifest))

    assert unified.unified_pbr_glb is not None
    assert unified.unified_pbr_glb.provenance["watertight"] is False
    assert unified.collision_topology == "surface_bvh"
    assert unified.visual is None
    assert unified.render_mesh is None
    assert unified.collider is None


def test_unified_collision_enabled_objects_require_gate_reports(
    sample_manifest: WorldManifest,
) -> None:
    payload = _unified_pbr_object_payload(sample_manifest)
    payload["quality_gates"]["alignment"] = {"status": "passed"}

    with pytest.raises(ValidationError, match="requires report_uri"):
        WorldObject.model_validate(payload)


def test_gate_report_digest_requires_complete_identity() -> None:
    with pytest.raises(ValidationError, match="requires report_uri"):
        GateRecord(status="passed", report_sha256="a" * 64, report_size_bytes=10)
    with pytest.raises(ValidationError, match="report_sha256 and report_size_bytes"):
        GateRecord(
            status="passed",
            report_uri="artifact://gate-report.json",
            report_sha256="a" * 64,
        )


def test_unified_surface_bvh_rejects_missing_face_count(
    sample_manifest: WorldManifest,
) -> None:
    payload = _unified_pbr_object_payload(sample_manifest)
    del payload["unified_pbr_glb"]["provenance"]["faces"]

    with pytest.raises(ValidationError, match="face_count"):
        WorldObject.model_validate(payload)


def test_unified_surface_bvh_rejects_failed_technical_gate(
    sample_manifest: WorldManifest,
) -> None:
    payload = _unified_pbr_object_payload(sample_manifest)
    payload["unified_pbr_glb"]["provenance"]["technical_gates"]["winding_consistent"] = False

    with pytest.raises(ValidationError, match="winding_consistent"):
        WorldObject.model_validate(payload)


def test_unified_pbr_glb_is_iterated_once_for_visual_and_collision(
    sample_manifest: WorldManifest,
) -> None:
    unified = WorldObject.model_validate(_unified_pbr_object_payload(sample_manifest))
    payload = sample_manifest.model_dump(mode="json")
    payload["objects"].append(unified.model_dump(mode="json"))
    manifest = WorldManifest.model_validate(payload)

    object_assets = [
        (location, asset) for location, asset in manifest.iter_assets() if asset.sha256 == "d" * 64
    ]

    assert [location for location, _ in object_assets] == [
        "objects[bedroom_4::pillow-unified-run::pillow-unified].unified_pbr_glb"
    ]


def test_unified_closed_volume_rejects_nonwatertight_provenance(
    sample_manifest: WorldManifest,
) -> None:
    payload = _unified_pbr_object_payload(sample_manifest)
    payload["collision_topology"] = "closed_volume"

    with pytest.raises(ValidationError, match="closed_volume requires explicit watertight"):
        WorldObject.model_validate(payload)


def test_unified_surface_bvh_rejects_volume_claims(
    sample_manifest: WorldManifest,
) -> None:
    payload = _unified_pbr_object_payload(sample_manifest)
    payload["unified_pbr_glb"]["provenance"]["closed_volume_claim"] = True

    with pytest.raises(ValidationError, match="cannot claim volume"):
        WorldObject.model_validate(payload)


def test_unified_pbr_glb_rejects_mixed_separate_assets(
    sample_manifest: WorldManifest,
) -> None:
    payload = _unified_pbr_object_payload(sample_manifest)
    payload["render_mesh"] = sample_manifest.scene.collider.model_dump(mode="json")

    with pytest.raises(ValidationError, match="cannot be mixed with separate object assets"):
        WorldObject.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("role", "render_mesh", "asset role"),
        ("status", "candidate", "asset must be validated"),
        ("media_type", "model/obj", "media_type"),
    ],
)
def test_unified_pbr_glb_requires_validated_binary_glb_identity(
    sample_manifest: WorldManifest,
    field: str,
    value: str,
    message: str,
) -> None:
    payload = _unified_pbr_object_payload(sample_manifest)
    payload["unified_pbr_glb"][field] = value

    with pytest.raises(ValidationError, match=message):
        WorldObject.model_validate(payload)


def test_visual_only_interaction_cannot_keep_unified_collision_topology(
    sample_manifest: WorldManifest,
) -> None:
    payload = _unified_pbr_object_payload(sample_manifest)
    payload["interaction"].update(
        {
            "collision_enabled": False,
            "physics_mode": "none",
        }
    )
    payload["quality_gates"]["collision"] = {"status": "not_tested"}

    with pytest.raises(ValidationError, match="visual-only interaction cannot declare"):
        WorldObject.model_validate(payload)

    payload["collision_topology"] = None
    visual_only = WorldObject.model_validate(payload)
    assert visual_only.unified_pbr_glb is not None
    assert visual_only.collision_topology is None
    assert visual_only.interaction.collision_enabled is False


def test_collision_enabled_interaction_requires_collider_and_passed_gate(
    sample_manifest: WorldManifest,
) -> None:
    pillow = _visual_only_object(sample_manifest)
    payload = pillow.model_dump(mode="json")
    payload["interaction"].update({"collision_enabled": True, "physics_mode": "kinematic"})
    with pytest.raises(ValidationError, match="missing: collider"):
        WorldObject.model_validate(payload)

    payload["collider"] = sample_manifest.scene.collider.model_dump(mode="json")
    with pytest.raises(ValidationError, match="requires a passed collision gate"):
        WorldObject.model_validate(payload)

    payload["quality_gates"]["collision"] = {
        "status": "passed",
        "report_uri": "artifact://pillow01/collision-report.json",
    }
    collidable = WorldObject.model_validate(payload)
    assert collidable.collider is not None
    assert collidable.interaction.collision_enabled is True


def test_collision_only_policy_still_enforces_the_interactive_contract() -> None:
    with pytest.raises(ValidationError, match="interactive object is missing"):
        WorldObject(
            id="static-obstacle",
            source_run_id="run",
            scoped_id="scene::run::static-obstacle",
            name=LocalizedText(en="static obstacle"),
            category="obstacle",
            interaction=InteractionPolicy(
                collision_enabled=True,
                physics_mode="static",
            ),
        )
