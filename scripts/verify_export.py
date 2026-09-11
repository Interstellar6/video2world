#!/usr/bin/env python3
"""Independently verify an exported asset directory against its own manifest.

This does not trust the exporter: it re-hashes every recorded file, loads every
mesh, and re-checks the geometric invariants the pipeline claims (positive
volume, convex watertight collision hulls, finite point clouds). It reports
failures instead of repairing them, and never promotes a candidate to observed
evidence.

    python3 scripts/verify_export.py --export-root /path/to/bedroom_4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(condition, message: str, issues: list[str]) -> None:
    if not condition:
        issues.append(message)


def verify_hash(export_root: Path, relative: str, expected: str | None, issues: list[str]) -> Path:
    path = export_root / relative
    check(path.is_file(), f"missing file: {relative}", issues)
    if path.is_file() and expected:
        check(sha256(path) == expected, f"sha256 mismatch: {relative}", issues)
    return path


def verify_mesh(path: Path, label: str, issues: list[str], *, convex_required: bool = False) -> None:
    import numpy as np
    import trimesh

    try:
        loaded = trimesh.load(path, force="scene", process=False)
    except Exception as error:  # noqa: BLE001 - report any loader failure
        issues.append(f"{label}: cannot load mesh: {type(error).__name__}")
        return
    geometry = list(loaded.geometry.values()) if hasattr(loaded, "geometry") else [loaded]
    check(bool(geometry), f"{label}: empty mesh", issues)
    for index, part in enumerate(geometry):
        if not isinstance(part, trimesh.Trimesh):
            issues.append(f"{label}[{index}]: not a triangle mesh")
            continue
        check(len(part.vertices) > 0 and len(part.faces) > 0, f"{label}[{index}]: no geometry", issues)
        check(bool(np.isfinite(part.vertices).all()), f"{label}[{index}]: non-finite vertices", issues)
        if convex_required:
            check(bool(part.is_convex), f"{label}[{index}]: not convex", issues)
            check(bool(part.is_watertight), f"{label}[{index}]: not watertight", issues)
            check(float(part.volume) > 0, f"{label}[{index}]: non-positive volume", issues)


def verify_point_cloud(path: Path, label: str, issues: list[str]) -> None:
    import numpy as np
    from plyfile import PlyData

    try:
        rows = PlyData.read(str(path))["vertex"].data
    except Exception as error:  # noqa: BLE001
        issues.append(f"{label}: cannot read PLY: {type(error).__name__}")
        return
    check(len(rows) > 0, f"{label}: empty point cloud", issues)
    if not len(rows):
        return
    names = set(rows.dtype.names or ())
    for key in ("x", "y", "z"):
        check(key in names, f"{label}: missing {key}", issues)
    if {"x", "y", "z"} <= names:
        xyz = np.column_stack([rows[key] for key in ("x", "y", "z")]).astype(float)
        check(bool(np.isfinite(xyz).all()), f"{label}: non-finite coordinates", issues)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--skip-geometry", action="store_true", help="hash and structure checks only")
    args = parser.parse_args(argv)
    root = args.export_root.resolve()
    issues: list[str] = []
    manifest_path = root / "export_manifest.json"
    if not manifest_path.is_file():
        print(json.dumps({"export_root": str(root), "valid": False, "issues": ["missing export_manifest.json"]}, indent=2))
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    check(manifest.get("status") == "complete", f"manifest status is {manifest.get('status')!r}", issues)
    check(not manifest.get("missing"), f"manifest reports missing roles: {manifest.get('missing')}", issues)
    for relative, record in manifest.get("scene_roles", {}).items():
        verify_hash(root, relative, record.get("sha256"), issues)
    for name, record in (manifest.get("object_version_1") or {}).items():
        for relative, entry in record.items():
            verify_hash(root, f"object/object_version_1/{name}/{relative}", entry.get("sha256"), issues)
    for name, record in (manifest.get("object_version_2") or {}).items():
        for relative, entry in record.items():
            if relative == "collision":
                for hull, hull_entry in entry.items():
                    verify_hash(root, f"object/object_version_2/{name}/collision/{hull}", hull_entry.get("sha256"), issues)
            elif isinstance(entry, dict) and entry.get("sha256"):
                verify_hash(root, f"object/object_version_2/{name}/{relative}", entry.get("sha256"), issues)
    for role, record in (manifest.get("recomposition") or {}).items():
        verify_hash(root, f"recomposition/{role}.json", record.get("sha256"), issues)
    for key in ("background", "background_generated"):
        record = manifest.get(key)
        if not isinstance(record, dict):
            continue
        relative = record.get("export_path")
        check(bool(relative), f"{key}: manifest does not record export_path", issues)
        if relative:
            verify_hash(root, relative, record.get("sha256"), issues)
    if not args.skip_geometry:
        for relative in manifest.get("scene_roles", {}):
            if relative.endswith(".ply"):
                verify_point_cloud(root / relative, relative, issues)
            elif relative.endswith((".glb", ".gltf")):
                verify_mesh(root / relative, relative, issues)
        for name, record in (manifest.get("object_version_2") or {}).items():
            mesh = record.get("asset_mesh.glb")
            if mesh:
                verify_mesh(root / f"object/object_version_2/{name}/asset_mesh.glb", f"{name}/asset_mesh.glb", issues)
            splat = record.get("asset_splat.ply")
            if splat:
                verify_point_cloud(root / f"object/object_version_2/{name}/asset_splat.ply", f"{name}/asset_splat.ply", issues)
            for hull in (record.get("collision") or {}):
                if hull.startswith("hull_"):
                    verify_mesh(root / f"object/object_version_2/{name}/collision/{hull}", f"{name}/{hull}",
                                issues, convex_required=True)
        for name, record in (manifest.get("object_version_1") or {}).items():
            for relative in record:
                if relative.endswith((".glb", ".gltf")):
                    verify_mesh(root / f"object/object_version_1/{name}/{relative}", f"{name}/{relative}", issues)
    report = {"export_root": str(root), "valid": not issues, "issues": issues,
              "checked_objects": sorted(set(manifest.get("object_version_1") or {}) | set(manifest.get("object_version_2") or {})),
              "note": "Structural and geometric integrity only; this is not visual or simulation acceptance."}
    print(json.dumps(report, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
