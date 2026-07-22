"""Manifest and artifact validation."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse

from pydantic import ValidationError

from video2world.errors import ArtifactError, ValidationFailure
from video2world.hashing import digest_path
from video2world.models import WorldManifest


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
        for location, gate in manifest.iter_gate_reports():
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
