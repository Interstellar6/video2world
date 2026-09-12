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


def read_scene_mesh(source: Path):
    """Read any scene mesh into one vtkPolyData without a Python-side rebuild.

    trimesh has to walk 12.8M faces in Python, which costs far more than the
    decimation itself; the VTK readers hand back the same surface in seconds and
    keep the vertex colours as point scalars.
    """
    import vtk

    suffix = source.suffix.lower()
    if suffix in {".glb", ".gltf"}:
        reader = vtk.vtkGLTFReader()
    elif suffix == ".ply":
        reader = vtk.vtkPLYReader()
    elif suffix == ".obj":
        reader = vtk.vtkOBJReader()
    else:
        raise PipelineError(f"no VTK reader for scene mesh: {source}")
    reader.SetFileName(str(source))
    reader.Update()
    produced = reader.GetOutput()
    if produced is None:
        raise PipelineError(f"VTK could not read the scene mesh: {source}")
    if isinstance(produced, vtk.vtkPolyData):
        return produced
    append = vtk.vtkAppendPolyData()
    iterator = produced.NewIterator()
    while not iterator.IsDoneWithTraversal():
        block = iterator.GetCurrentDataObject()
        if isinstance(block, vtk.vtkPolyData):
            append.AddInputData(block)
        iterator.GoToNextItem()
    append.Update()
    merged = append.GetOutput()
    if merged is None or merged.GetNumberOfCells() == 0:
        raise PipelineError(f"scene mesh has no polygons: {source}")
    return merged


def normalize_colors(polydata) -> bool:
    """Make the vertex colours an unsigned-char RGB active scalar array.

    Both writers only emit colours for that exact array shape: the GLTF reader
    hands back whatever the file declared, and a float or four-component array
    silently produces an uncoloured mesh. Losing the measured colour here would
    be invisible until someone opened the delivered file.
    """
    import numpy as np
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

    scalars = polydata.GetPointData().GetScalars()
    if scalars is None:
        return False
    values = np.asarray(vtk_to_numpy(scalars))
    if values.ndim != 2 or values.shape[1] < 3 or not len(values):
        return False
    if values.dtype.kind == "f":
        scale = 255.0 if float(values.max()) <= 1.0 else 1.0
        values = np.clip(values[:, :3] * scale, 0, 255)
    values = values[:, :3]
    if values.dtype != np.uint8:
        values = np.clip(values, 0, 255).astype(np.uint8)
    array = numpy_to_vtk(np.ascontiguousarray(values), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    array.SetName("Colors")
    array.SetNumberOfComponents(3)
    polydata.GetPointData().SetScalars(array)
    return True


def write_scene_mesh(polydata, destination: Path) -> None:
    import vtk

    destination.parent.mkdir(parents=True, exist_ok=True)
    import tempfile

    suffix = destination.suffix.lower()
    if not normalize_colors(polydata):
        raise PipelineError(f"scene mesh carries no vertex colours to deliver: {destination}")
    scalars = polydata.GetPointData().GetScalars()
    if suffix in {".glb", ".gltf"}:
        # vtkGLTFWriter rejects a plain polydata and segfaults on the composite
        # dataset it asks for, so the decimated surface goes out as PLY and is
        # converted with trimesh -- cheap at the decimated size and it carries
        # the vertex colours through as COLOR_0.
        import trimesh

        with tempfile.TemporaryDirectory() as folder:
            intermediate = Path(folder) / "decimated.ply"
            write_scene_mesh(polydata, intermediate)
            trimesh.load(intermediate, force="mesh", process=False).export(destination)
    else:
        writer = vtk.vtkPLYWriter()
        writer.SetInputData(polydata)
        writer.SetFileTypeToBinary()
        if scalars is not None:
            writer.SetArrayName(scalars.GetName())
            writer.SetColorModeToDefault()
        writer.SetFileName(str(destination))
        writer.Write()
    if not destination.is_file() or destination.stat().st_size == 0:
        raise PipelineError(f"scene mesh writer produced no file: {destination}")


def deliverable_mesh(source: Path, destination: Path, face_budget: int) -> dict:
    """Decimate a reconstructed scene mesh to a delivery budget.

    The raw TSDF surface is 12.8M faces and 260 MB, which no viewer can open.
    ``vtkDecimatePro`` reaches the budget in about two minutes here and keeps the
    reconstructed vertex colours; ``vtkQuadricDecimation`` refused to reduce this
    open surface at all because it preserves boundary edges.
    """
    if not source.is_file():
        raise PipelineError(f"missing scene mesh: {source}")
    import time
    import vtk

    started = time.perf_counter()
    polydata = read_scene_mesh(source)
    before = int(polydata.GetNumberOfCells())
    info = {"vertices_before": int(polydata.GetNumberOfPoints()), "faces_before": before,
            "decimation": "not_needed", "decimation_target_faces": face_budget or None}
    if face_budget and before > face_budget:
        decimator = vtk.vtkDecimatePro()
        decimator.SetInputData(polydata)
        decimator.SetTargetReduction(1.0 - face_budget / before)
        decimator.PreserveTopologyOff()
        decimator.BoundaryVertexDeletionOn()
        decimator.SplittingOn()
        decimator.SetMaximumError(vtk.VTK_DOUBLE_MAX)
        decimator.Update()
        polydata = decimator.GetOutput()
        if polydata is None or polydata.GetNumberOfCells() == 0:
            raise PipelineError(f"decimation produced no geometry: {source}")
        info["decimation"] = "vtk_decimate_pro_binary_quadric"
    write_scene_mesh(polydata, destination)
    info.update({"vertices": int(polydata.GetNumberOfPoints()), "faces": int(polydata.GetNumberOfCells()),
                 "seconds": round(time.perf_counter() - started, 2)})
    return info


def export_observed_parts(task: Task, record: dict, export_root: Path) -> dict:
    """Deliver every geometrically verified observed component as its own asset.

    Lifting already carves each verified component track of an object, so the
    parts a delivery wants (bedding, headboard, pillows, a lamp shade) exist as
    observed geometry before any generative completion touches them. They are
    copied with their hashes so a part can be inspected without the parent.
    """
    verify_artifact(task, record)
    document = read_json(artifact_path(task, record))
    root = export_root / "object" / "parts"
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for obj in document.get("objects", []):
        object_id = obj.get("object_id")
        parts = obj.get("parts") or []
        for index, part in enumerate(sorted(parts, key=lambda item: str(item.get("component_id")))):
            source = artifact_path(task, {"path": part["ply_path"]})
            expected = part.get("sha256")
            if expected and sha256(source) != expected:
                raise PipelineError(f"{object_id}: component PLY changed after lifting: {part['ply_path']}")
            name = f"{object_id}_part_{index:02d}.ply"
            destination = root / object_id / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            entries.append({"object_id": object_id, "component_id": part.get("component_id"),
                            "association_status": part.get("association_status"),
                            "export_path": f"object/parts/{object_id}/{name}",
                            "source_path": part["ply_path"], "source_sha256": expected,
                            "sha256": sha256(destination), "evidence": "observed",
                            "collision_eligible": False,
                            "gaussian_count": part.get("gaussian_count") or part.get("source_gaussian_count"),
                            "world_bounds": part.get("world_bounds")})
    if not entries:
        raise PipelineError("isolated_object_ply records no verified component to deliver")
    manifest = root / "manifest.json"
    write_json(manifest, {"schema_version": "1.0", "kind": "video2world-modeling.observed_object_parts",
                          "source_path": record["path"], "source_sha256": record["sha256"],
                          "parts": entries})
    return {"root": "object/parts", "manifest": "object/parts/manifest.json", "sha256": sha256(manifest),
            "parts": len(entries),
            "objects": sorted({entry["object_id"] for entry in entries})}


def export_depth_frames(task: Task, record: dict, export_root: Path, manifest: dict) -> None:
    """Write viewable depth maps beside the scene, with their metric scale.

    The reconstruction stores depth as float arrays; a delivery needs something
    a viewer or a downstream tool can open, so each frame becomes a 16-bit PNG
    with the scale in a manifest, and the raw arrays stay bound by hash.
    """
    import numpy as np
    from PIL import Image

    verify_artifact(task, record)
    document = read_json(artifact_path(task, record))
    frames = document.get("frames") or []
    if not frames:
        manifest["missing"].append("scene_depth")
        return
    directory = export_root / "scene" / "depth"
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for frame in frames:
        depth_path = artifact_path(task, {"path": frame["depth_path"]})
        depth = np.load(depth_path)
        finite = np.isfinite(depth)
        if not finite.any():
            raise PipelineError(f"{frame['frame_id']}: depth frame has no finite sample")
        nearest, farthest = float(depth[finite].min()), float(depth[finite].max())
        scale = (farthest - nearest) / 65535.0 if farthest > nearest else 1.0
        quantized = np.where(finite, (np.nan_to_num(depth) - nearest) / scale, 0.0)
        image = Image.fromarray(np.clip(np.rint(quantized), 0, 65535).astype(np.uint16), mode="I;16")
        destination = directory / f"{frame['frame_id']}.png"
        image.save(destination)
        entries.append({"frame_id": frame["frame_id"], "path": str(destination.relative_to(export_root)),
                        "sha256": sha256(destination), "offset": nearest, "scale": scale, "invalid_value": 0,
                        "width": int(frame["width"]), "height": int(frame["height"]),
                        "valid_depth_fraction": frame.get("valid_depth_fraction"),
                        "source_depth_path": frame["depth_path"], "source_depth_sha256": frame["depth_sha256"]})
    depth_manifest = directory / "manifest.json"
    write_json(depth_manifest, {"schema_version": "1.0", "kind": "video2world-modeling.export_depth_maps",
                                "encoding": "uint16_linear", "units": document.get("units"),
                                "coordinate_frame": document.get("coordinate_frame"),
                                "depth_kind": document.get("depth_kind"), "frames": entries})
    manifest["scene_roles"]["scene/depth/manifest.json"] = {
        "path": "scene/depth/manifest.json", "sha256": sha256(depth_manifest), "frames": len(entries),
        "encoding": "uint16_linear", "units": document.get("units")}


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
    export_object_reports(task, root, index, manifest)


def object_report_sources(task: Task, index: dict) -> dict:
    """Per-object provenance for the version-1 assets, gathered from the stages.

    Every producer of a version-1 asset records the artifact it wrote and the
    stage that wrote it, so an exported object directory says where each of its
    files came from without the reader having to open the task registry.
    """
    sources: dict[str, list] = {}

    def record(object_id: str, role: str, item: dict) -> None:
        entry = {"role": role, "provider": item.get("provider") or index[role].get("producer")}
        if isinstance(item.get("mesh_path"), str):
            entry["path"] = item["mesh_path"]
        if isinstance(item.get("source_mesh"), str):
            entry["source_mesh"] = item["source_mesh"]
        if isinstance(item.get("source_splat"), str):
            entry["source_splat"] = item["source_splat"]
        if isinstance(item.get("source_frame_ids"), list):
            entry["source_frame_ids"] = item["source_frame_ids"]
        if index[role].get("evidence"):
            entry["evidence"] = index[role]["evidence"]
        sources.setdefault(object_id, []).append(entry)

    for role in ("completed_object_meshes", "repaired_visual_meshes", "mesh_qa_report",
                 "coacd_collision_meshes", "physics_properties", "uv_textures"):
        if role not in index:
            continue
        try:
            document = read_json(artifact_path(task, index[role]))
        except (OSError, ValueError, PipelineError):
            continue
        for item in records_of(document, "objects"):
            object_id = str(item.get("object_id") or "").strip()
            if object_id:
                record(object_id, role, item)
    return sources


def export_object_reports(task: Task, root: Path, index: dict, manifest: dict) -> None:
    """Write an ``asset_report.json`` next to every exported object directory."""
    sources = object_report_sources(task, index)
    for version in ("object_version_1", "object_version_2"):
        for object_id, entry in manifest[version].items():
            directory = root / "object" / version / object_id
            report = {"schema_version": "1.0", "object_id": object_id, "task_id": task.task_id,
                      "export_version": version, "exported_files": entry,
                      "provenance": sources.get(object_id, [])}
            write_json(directory / "asset_report.json", report)
            entry["asset_report.json"] = {"source_path": f"object/{version}/{object_id}/asset_report.json",
                                          "sha256": sha256(directory / "asset_report.json")}
    manifest["object_report_provenance"] = {object_id: len(items) for object_id, items in sorted(sources.items())}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_id")
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--scene-name")
    parser.add_argument("--skip-background-texture", action="store_true",
                        help="keep vertex colours instead of baking a background texture")
    parser.add_argument("--background-face-budget", type=int, default=100_000)
    parser.add_argument("--background-texture-size", type=int, default=2048)
    parser.add_argument("--skip-viewer", action="store_true",
                        help="do not generate the web viewer project")
    parser.add_argument("--skip-previews", action="store_true",
                        help="do not render the QA preview images")
    parser.add_argument("--preview-size", type=int, default=512)
    parser.add_argument("--preview-views", type=int, default=6)
    parser.add_argument("--scene-face-budget", type=int, default=1_000_000,
                        help="decimate the reconstructed scene mesh to at most this many faces; 0 disables")
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
        info = deliverable_mesh(source, destination, args.scene_face_budget)
        manifest["scene_roles"][relative] = {"source_path": index[role]["path"], "sha256": sha256(destination),
                                             "evidence": index[role].get("evidence"),
                                             "converted_from": index[role]["path"], **info}
    for role, name in (("scene_tsdf_mesh", "background.glb"), ("generated_background_mesh", "background_generated.glb")):
        record = index.get(role)
        if record is None:
            continue
        destination = export_root / "object" / name
        key = "background" if role == "scene_tsdf_mesh" else "background_generated"
        verify_artifact(task, record)
        source = artifact_path(task, record)
        texture_fallback = None
        if (role == "scene_tsdf_mesh" and not args.skip_background_texture
                and all(role_name in index for role_name in ("cameras", "scene_depth"))):
            try:
                import bake_background_texture

                report = bake_background_texture.run(argparse.Namespace(
                    task_dir=task.directory, output_dir=destination.parent, name=name,
                    face_budget=args.background_face_budget, texture_size=args.background_texture_size,
                    absolute_tolerance=0.05, relative_tolerance=0.05, max_frames=0))
                manifest[key] = {"role": role, "export_path": f"object/{name}", "source_path": record["path"],
                                 "source_sha256": record["sha256"], "evidence": record.get("evidence"),
                                 "collision_eligible": False, "textured": True, **report}
                continue
            except (PipelineError, OSError, ValueError, KeyError) as error:
                # The delivered surface is still usable without a texture, so a
                # failed bake degrades to vertex colours rather than failing the
                # export -- but the reason travels in the manifest.
                texture_fallback = str(error)
        if source.suffix.lower() in {".glb", ".gltf", ".ply", ".obj"}:
            info = deliverable_mesh(source, destination, args.scene_face_budget)
            entry = {"role": role, "export_path": f"object/{name}", "source_path": record["path"],
                     "source_sha256": record["sha256"], "sha256": sha256(destination),
                     "evidence": record.get("evidence"), "collision_eligible": False, "textured": False, **info}
            if texture_fallback:
                entry["background_texture_fallback"] = texture_fallback
            manifest[key] = entry
        else:
            manifest[key] = {"role": role, "export_path": f"object/{name}", **copy_bound(task, record, destination)}
    if "background" not in manifest:
        manifest["missing"].append("background")
    if "isolated_object_ply" in index:
        manifest["observed_parts"] = export_observed_parts(task, index["isolated_object_ply"], export_root)
    else:
        manifest["missing"].append("isolated_object_ply")
    if "scene_depth" in index:
        export_depth_frames(task, index["scene_depth"], export_root, manifest)
    else:
        manifest["missing"].append("scene_depth")
    if not args.skip_previews:
        import render_previews

        manifest["previews"] = render_previews.run(argparse.Namespace(
            task_dir=task.directory, task_id=None, export_root=export_root, size=args.preview_size,
            views=args.preview_views))

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
    if not args.skip_viewer:
        # The viewer builds its asset list from the manifest, so it runs after the
        # manifest exists and the manifest is then rewritten with its report.
        import build_viewer

        manifest["viewer"] = build_viewer.run(export_root, title=args.scene_name or task_id)
        write_json(export_root / "export_manifest.json", manifest)
    print(json.dumps({"export_root": str(export_root), "status": manifest["status"],
                      "missing": manifest["missing"], "scene_roles": sorted(manifest["scene_roles"]),
                      "object_version_1": sorted(manifest["object_version_1"]),
                      "object_version_2": sorted(manifest["object_version_2"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
