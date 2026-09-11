#!/usr/bin/env python3
"""Export a task's hash-verified artifacts into the video2world asset layout.

The layout mirrors `video2world-export-assets/<scene>/`:

    datasets/<source>/            copied calibrated source media
    scene/point_cloud_3dgs.ply    observed PGSR Gaussians
    scene/point_cloud_simple.ply  carved scene Gaussians
    scene/mesh.ply                observed TSDF mesh
    object/background.glb         generated background candidate when present
    object/object_version_2/<id>/asset_mesh.glb + asset_splat.ply
    object/object_version_1/<id>/<id>.obj|.mtl|.glb + textures
    export_manifest.json          role -> exported path, sha256, evidence class

Every copied artifact is re-verified against its task receipt before it is
written, and the manifest records the evidence class so generated candidates
are never presented as observed geometry. Missing roles are reported, not
silently skipped.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from world_modeling.engine import (  # noqa: E402
    PipelineError, Task, artifact_path, read_json, sha256, verify_artifact, write_json,
)
from world_modeling.gaussian_io import GaussianError, read_gaussian_rows  # noqa: E402

# role -> fixed destination inside the export root
SCENE_ROLES = {
    "scene_gaussian_ply": "scene/point_cloud_3dgs.ply",
    "carved_scene_ply": "scene/point_cloud_simple.ply",
}
MESH_TO_PLY_ROLES = {"scene_tsdf_mesh": "scene/mesh.ply"}


def glb_to_ply(source: Path, destination: Path) -> dict:
    import trimesh

    scene = trimesh.load(source, force="scene")
    mesh = scene.to_mesh() if hasattr(scene, "to_mesh") else scene
    if mesh.is_empty:
        raise PipelineError(f"mesh has no geometry: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(destination)
    return {"vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces))}


def copy_bound(task: Task, record: dict, destination: Path) -> dict:
    verify_artifact(task, record)
    source = artifact_path(task, record)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns(".DS_Store", "._*", "__pycache__"))
    else:
        shutil.copy2(source, destination)
    return {"source_path": record["path"], "sha256": record["sha256"], "evidence": record.get("evidence"),
            "collision_eligible": bool(record.get("collision_eligible"))}


def copy_gaussian_ply(task: Task, record: dict, destination: Path) -> dict:
    """Copy a scene Gaussian PLY without the rows that cannot be rendered.

    Exporting is the last place a non-finite splat can be removed: every reader
    already drops them, and a delivered asset whose coordinates are NaN fails an
    independent geometry check. The dropped count travels in the manifest so the
    degeneracy stays visible after delivery.
    """
    verify_artifact(task, record)
    source = artifact_path(task, record)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cloud = read_gaussian_rows(source, error=PipelineError)
    if cloud.dropped:
        from plyfile import PlyData, PlyElement

        PlyData([PlyElement.describe(cloud.rows, "vertex")], text=False).write(str(destination))
    else:
        shutil.copy2(source, destination)
    return {"source_path": record["path"], "source_sha256": record["sha256"], "sha256": sha256(destination),
            "evidence": record.get("evidence"), "collision_eligible": bool(record.get("collision_eligible")),
            "non_finite_rows_dropped": cloud.dropped, "exported_rows": int(len(cloud.rows))}


def records_of(document: dict, key: str) -> list[dict]:
    value = document.get(key, [])
    return value if isinstance(value, list) else []


def copy_with_sidecars(task: Task, value: str, destination: Path, manifest_entry: dict, name: str) -> None:
    """Copy one artifact file plus the sibling files a mesh references (mtl/textures)."""
    source = (task.directory / value).resolve()
    if not source.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    manifest_entry[name] = {"source_path": value, "sha256": sha256(source)}
    for sibling in sorted(source.parent.iterdir()):
        if sibling == source or not sibling.is_file():
            continue
        if sibling.suffix.lower() in {".mtl", ".png", ".jpg", ".jpeg", ".bin"}:
            shutil.copy2(sibling, destination.parent / sibling.name)
            manifest_entry[sibling.name] = {"source_path": sibling.relative_to(task.directory).as_posix(),
                                            "sha256": sha256(sibling)}


def export_object_versions(task: Task, root: Path, index: dict, manifest: dict) -> None:
    if "completed_object_meshes" in index:
        document = read_json(artifact_path(task, index["completed_object_meshes"]))
        for item in records_of(document, "objects"):
            object_id = str(item.get("object_id") or "").strip()
            if not object_id:
                continue
            directory = root / "object" / "object_version_2" / object_id
            directory.mkdir(parents=True, exist_ok=True)
            exported = {}
            for key, name in (("mesh_path", "asset_mesh.glb"), ("visual_ply_path", "asset_splat.ply")):
                value = item.get(key)
                if not isinstance(value, str):
                    continue
                source = (task.directory / value).resolve()
                if not source.is_file():
                    continue
                shutil.copy2(source, directory / name)
                exported[name] = {"source_path": value, "sha256": sha256(source)}
            if exported:
                manifest["object_version_2"][object_id] = exported
    if "coacd_collision_meshes" in index:
        document = read_json(artifact_path(task, index["coacd_collision_meshes"]))
        for item in records_of(document, "objects"):
            object_id = str(item.get("object_id") or "").strip()
            if not object_id:
                continue
            directory = root / "object" / "object_version_2" / object_id / "collision"
            entry = manifest["object_version_2"].setdefault(object_id, {})
            entry.setdefault("collision", {})
            value = item.get("mesh_path") or item.get("path")
            if isinstance(value, str):
                copy_with_sidecars(task, value, directory / "collision.obj", entry["collision"], "collision.obj")
            for index_number, hull in enumerate(item.get("hulls", []) or []):
                value = hull.get("path")
                if isinstance(value, str):
                    copy_with_sidecars(task, value, directory / f"hull_{index_number:03d}.obj",
                                       entry["collision"], f"hull_{index_number:03d}.obj")
    if "physics_properties" in index:
        document = read_json(artifact_path(task, index["physics_properties"]))
        for item in records_of(document, "objects"):
            object_id = str(item.get("object_id") or "").strip()
            if not object_id:
                continue
            directory = root / "object" / "object_version_2" / object_id
            directory.mkdir(parents=True, exist_ok=True)
            write_json(directory / "physics_properties.json", item)
            manifest["object_version_2"].setdefault(object_id, {})["physics_properties.json"] = {
                "source_path": index["physics_properties"]["path"], "sha256": sha256(directory / "physics_properties.json"),
                "evidence": index["physics_properties"].get("evidence")}
    if "repaired_visual_meshes" in index:
        document = read_json(artifact_path(task, index["repaired_visual_meshes"]))
        for item in records_of(document, "objects"):
            object_id = str(item.get("object_id") or "").strip()
            if not object_id:
                continue
            value = item.get("mesh_path") or item.get("path")
            if not isinstance(value, str):
                continue
            directory = root / "object" / "object_version_1" / object_id
            directory.mkdir(parents=True, exist_ok=True)
            entry = manifest["object_version_1"].setdefault(object_id, {})
            source = (task.directory / value).resolve()
            if source.is_file():
                target = directory / f"{object_id}{source.suffix.lower()}"
                shutil.copy2(source, target)
                entry[target.name] = {"source_path": value, "sha256": sha256(source)}
    if "physical_object_obj" in index:
        document = read_json(artifact_path(task, index["physical_object_obj"]))
        for item in records_of(document, "objects"):
            object_id = str(item.get("object_id") or "").strip()
            value = item.get("obj_path")
            if not object_id or not isinstance(value, str):
                continue
            directory = root / "object" / "object_version_1" / object_id
            directory.mkdir(parents=True, exist_ok=True)
            entry = manifest["object_version_1"].setdefault(object_id, {})
            copy_with_sidecars(task, value, directory / f"{object_id}.obj", entry, f"{object_id}.obj")
    if "uv_textures" in index:
        document = read_json(artifact_path(task, index["uv_textures"]))
        for item in records_of(document, "objects"):
            object_id = str(item.get("object_id") or "").strip()
            if not object_id:
                continue
            entry = manifest["object_version_1"].setdefault(object_id, {})
            for texture in item.get("textures", []) or []:
                value = texture.get("path")
                if not isinstance(value, str):
                    continue
                source = (task.directory / value).resolve()
                if not source.is_file():
                    continue
                destination = root / "object" / "object_version_1" / object_id / "textures" / source.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                entry[f"textures/{source.name}"] = {"source_path": value, "sha256": sha256(source)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_id")
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--scene-name")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    task = Task(root, args.task_id)
    if not task.directory.is_dir():
        raise PipelineError(f"unknown task: {task.directory}")
    export_root = args.export_root.resolve()
    export_root.mkdir(parents=True, exist_ok=True)
    index = task.load_artifacts()["artifacts"]
    manifest = {"schema_version": "1.0", "task_id": task.task_id, "export_root": str(export_root),
                "scene_roles": {}, "object_version_1": {}, "object_version_2": {}, "missing": []}
    for role, relative in SCENE_ROLES.items():
        if role not in index:
            manifest["missing"].append(role)
            continue
        try:
            manifest["scene_roles"][relative] = copy_gaussian_ply(task, index[role], export_root / relative)
        except GaussianError as error:
            raise PipelineError(f"{role}: {error}") from error
    for role, relative in MESH_TO_PLY_ROLES.items():
        if role not in index:
            manifest["missing"].append(role)
            continue
        verify_artifact(task, index[role])
        source = artifact_path(task, index[role])
        destination = export_root / relative
        if source.suffix.lower() == ".ply":
            manifest["scene_roles"][relative] = copy_bound(task, index[role], destination)
        else:
            info = glb_to_ply(source, destination)
            manifest["scene_roles"][relative] = {"source_path": index[role]["path"], "sha256": sha256(destination),
                                                 "evidence": index[role].get("evidence"), "converted_from": index[role]["path"], **info}
    for role, name in (("scene_tsdf_mesh", "background.glb"), ("generated_background_mesh", "background_generated.glb")):
        record = index.get(role)
        if record is None:
            continue
        destination = export_root / "object" / name
        key = "background" if role == "scene_tsdf_mesh" else "background_generated"
        manifest[key] = {"role": role, "export_path": f"object/{name}", **copy_bound(task, record, destination)}
    if "background" not in manifest:
        manifest["missing"].append("background")
    if "source_media" in index:
        source = artifact_path(task, index["source_media"])
        name = args.scene_name or source.name
        copy_bound(task, index["source_media"], export_root / "datasets" / name)
        manifest["datasets"] = {"name": name, "source_path": index["source_media"]["path"]}
    else:
        manifest["missing"].append("source_media")
    recomposition = {}
    for role in ("world_manifest", "visual_scene_manifest", "collision_scene_manifest", "recomposition_report"):
        if role not in index:
            continue
        record = index[role]
        destination = export_root / "recomposition" / f"{role}.json"
        recomposition[role] = copy_bound(task, record, destination)
    if recomposition:
        manifest["recomposition"] = recomposition
    export_object_versions(task, export_root, index, manifest)
    # The modeled objects are the point of the export; scene-only output is partial.
    if not manifest["object_version_2"]:
        manifest["missing"].append("object_version_2_completed_objects")
    manifest["status"] = "partial" if manifest["missing"] else "complete"
    write_json(export_root / "export_manifest.json", manifest)
    print(json.dumps({"export_root": str(export_root), "status": manifest["status"],
                      "missing": manifest["missing"], "scene_roles": sorted(manifest["scene_roles"]),
                      "object_version_1": sorted(manifest["object_version_1"]),
                      "object_version_2": sorted(manifest["object_version_2"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
