"""Manifest and artifact validation."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import unquote, urlparse

from pydantic import ValidationError

from video2world.errors import ArtifactError, ValidationFailure
from video2world.hashing import digest_path
from video2world.models import WorldManifest, WorldObject


def load_world_manifest(path: str | Path) -> WorldManifest:
    manifest_path = Path(path).expanduser().resolve()
    try:
        return WorldManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValidationFailure(f"cannot read manifest {manifest_path}: {exc}") from exc
    except ValidationError as exc:
        raise ValidationFailure(f"world manifest schema validation failed: {exc}") from exc


def _local_asset_path(manifest_path: Path, uri: str) -> Path | None:
    parsed = urlparse(uri)
    if parsed.scheme in {"http", "https"}:
        return None
    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).expanduser().resolve()
    if parsed.scheme:
        raise ArtifactError(f"unsupported asset URI scheme {parsed.scheme!r}: {uri}")
    path = Path(uri).expanduser()
    return (manifest_path.parent / path).resolve() if not path.is_absolute() else path.resolve()


def _validate_gate_report_payload(
    path: Path,
    *,
    gate_name: str,
    gate_status: str,
    object_id: str,
    scoped_id: str,
    asset_sha256: str | None,
    require_unified_bindings: bool = False,
) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"gate report is not readable JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict) or not payload:
        raise ArtifactError(f"gate report must be a non-empty JSON object: {path}")
    if require_unified_bindings:
        missing_fields = [
            field
            for field in ("gate", "status", "scoped_id")
            if payload.get(field) is None
        ]
        if payload.get("object_id") is None and payload.get("target_id") is None:
            missing_fields.append("object_id or target_id")
        if asset_sha256 is not None and not any(
            payload.get(field) is not None
            for field in (
                "asset_sha256",
                "source_asset_sha256",
                "unified_pbr_glb_sha256",
                "collision_asset_sha256",
            )
        ):
            missing_fields.append("unified PBR GLB asset sha256")
        if missing_fields:
            raise ArtifactError(
                "collision-enabled unified PBR GLB gate report is missing required "
                "bindings: "
                + ", ".join(missing_fields)
            )
    report_gate = payload.get("gate")
    if report_gate is not None and report_gate != gate_name:
        raise ArtifactError(
            f"gate report gate {report_gate!r} does not match manifest gate {gate_name!r}"
        )
    report_status = payload.get("status")
    if report_status is not None and report_status != gate_status:
        raise ArtifactError(
            f"gate report status {report_status!r} does not match gate status {gate_status!r}"
        )
    report_scoped_id = payload.get("scoped_id")
    if report_scoped_id is not None and report_scoped_id != scoped_id:
        raise ArtifactError(
            f"gate report scoped_id {report_scoped_id!r} does not match manifest object "
            f"{scoped_id!r}"
        )
    for field in ("object_id", "target_id"):
        report_object_id = payload.get(field)
        if report_object_id is not None and report_object_id != object_id:
            raise ArtifactError(
                f"gate report {field} {report_object_id!r} does not match manifest object "
                f"{object_id!r}"
            )
    if asset_sha256 is not None:
        for field in (
            "asset_sha256",
            "source_asset_sha256",
            "unified_pbr_glb_sha256",
            "collision_asset_sha256",
        ):
            report_asset_sha256 = payload.get(field)
            if report_asset_sha256 is not None and report_asset_sha256 != asset_sha256:
                raise ArtifactError(
                    f"gate report {field} {report_asset_sha256!r} does not match manifest "
                    f"unified_pbr_glb asset {asset_sha256!r}"
                )
    return payload


def _expect_report_field(payload: dict[str, object], field: str, *, gate_name: str) -> object:
    if field not in payload:
        raise ArtifactError(
            f"{gate_name} gate report is missing required manifest binding field {field!r}"
        )
    return payload[field]


def _validate_unified_gate_report_manifest_bindings(
    payload: dict[str, object],
    *,
    gate_name: str,
    item: WorldObject,
) -> None:
    if gate_name == "alignment":
        expected_bbox = item.bbox_scene.model_dump(mode="json") if item.bbox_scene else None
        expected_transform = (
            item.transform_scene_from_asset.model_dump(mode="json")
            if item.transform_scene_from_asset
            else None
        )
        if _expect_report_field(payload, "bbox_scene", gate_name=gate_name) != expected_bbox:
            raise ArtifactError("alignment gate report bbox_scene does not match manifest")
        if (
            _expect_report_field(payload, "transform_scene_from_asset", gate_name=gate_name)
            != expected_transform
        ):
            raise ArtifactError(
                "alignment gate report transform_scene_from_asset does not match manifest"
            )
    if gate_name == "collision":
        report_topology = _expect_report_field(payload, "collision_topology", gate_name=gate_name)
        if report_topology != item.collision_topology:
            raise ArtifactError(
                f"collision gate report topology {report_topology!r} does not match manifest "
                f"{item.collision_topology!r}"
            )
        assert item.unified_pbr_glb is not None
        expected_faces = item.unified_pbr_glb.provenance.get("faces")
        report_faces = payload.get("faces", payload.get("face_count"))
        if report_faces is None:
            raise ArtifactError(
                "collision gate report is missing required manifest binding field "
                "'faces' or 'face_count'"
            )
        if report_faces != expected_faces:
            raise ArtifactError(
                f"collision gate report face count {report_faces!r} does not match manifest "
                f"{expected_faces!r}"
            )


def _object_identity_from_gate_location(location: str) -> tuple[str, str]:
    prefix = "objects["
    marker = "].quality_gates."
    if not location.startswith(prefix) or marker not in location:
        raise ArtifactError(f"invalid gate report location: {location}")
    scoped_id = location[len(prefix) : location.index(marker)]
    object_id = scoped_id.rsplit("::", 1)[-1]
    return object_id, scoped_id


def validate_world_manifest(
    path: str | Path,
    *,
    verify_assets: bool = True,
    allow_remote: bool = False,
) -> dict[str, object]:
    manifest_path = Path(path).expanduser().resolve()
    manifest = load_world_manifest(manifest_path)
    issues: list[dict[str, str]] = []
    checked_assets = 0
    checked_gate_reports = 0
    remote_assets = 0
    remote_gate_reports = 0
    if verify_assets:
        for location, asset in manifest.iter_assets():
            try:
                local_path = _local_asset_path(manifest_path, asset.uri)
                if local_path is None:
                    remote_assets += 1
                    if not allow_remote:
                        issues.append(
                            {
                                "location": location,
                                "error": (
                                    "remote asset was not content-verified; pass --allow-remote "
                                    "explicitly"
                                ),
                            }
                        )
                    continue
                current = digest_path(local_path)
                checked_assets += 1
                if current.sha256 != asset.sha256:
                    issues.append({"location": location, "error": "sha256 mismatch"})
                if current.size_bytes != asset.size_bytes:
                    issues.append({"location": location, "error": "size_bytes mismatch"})
            except ArtifactError as exc:
                issues.append({"location": location, "error": str(exc)})
        for location, gate, item in manifest.iter_gate_reports():
            assert gate.report_uri is not None
            assert gate.report_sha256 is not None
            assert gate.report_size_bytes is not None
            try:
                local_path = _local_asset_path(manifest_path, gate.report_uri)
                if local_path is None:
                    remote_gate_reports += 1
                    if not allow_remote:
                        issues.append(
                            {
                                "location": location,
                                "error": (
                                    "remote gate report was not content-verified; pass "
                                    "--allow-remote explicitly"
                                ),
                            }
                        )
                    continue
                current = digest_path(local_path)
                checked_gate_reports += 1
                if current.sha256 != gate.report_sha256:
                    issues.append({"location": location, "error": "report_sha256 mismatch"})
                if current.size_bytes != gate.report_size_bytes:
                    issues.append({"location": location, "error": "report_size_bytes mismatch"})
                object_id, scoped_id = _object_identity_from_gate_location(location)
                gate_name = location.rsplit(".", 1)[-1]
                require_unified_bindings = (
                    item.unified_pbr_glb is not None
                    and item.interaction.collision_enabled
                    and gate.status == "passed"
                    and gate_name in {"alignment", "collision", "visual"}
                )
                report_payload = _validate_gate_report_payload(
                    local_path,
                    gate_name=gate_name,
                    gate_status=gate.status,
                    object_id=object_id,
                    scoped_id=scoped_id,
                    asset_sha256=(
                        item.unified_pbr_glb.sha256 if item.unified_pbr_glb is not None else None
                    ),
                    require_unified_bindings=require_unified_bindings,
                )
                if require_unified_bindings:
                    _validate_unified_gate_report_manifest_bindings(
                        report_payload,
                        gate_name=gate_name,
                        item=item,
                    )
            except ArtifactError as exc:
                issues.append({"location": location, "error": str(exc)})
    return {
        "valid": not issues,
        "manifest": str(manifest_path),
        "world_id": manifest.world_id,
        "manifest_status": manifest.manifest_status,
        "objects": len(manifest.objects),
        "checked_assets": checked_assets,
        "checked_gate_reports": checked_gate_reports,
        "remote_assets": remote_assets,
        "remote_gate_reports": remote_gate_reports,
        "issues": issues,
    }
