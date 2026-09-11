#!/usr/bin/env python3
"""Task-scoped CPU mesh repair, simplification, UV baking, and CoACD provider.

This adapter uses geometry utilities, not the learned 3D-Fixer model. Input and
output manifests contain an objects list whose paths are relative to task_dir.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from world_modeling.object_lineage import lineage_metadata, validate_object_lineages


class PostprocessError(RuntimeError):
    pass


# pymeshfix joins components superlinearly; above this face count its runtime is
# unbounded in practice (a 100k-face mesh ran for more than an hour), so the
# repair is refused and recorded instead of waited on.
PYMESHFIX_FACE_LIMIT = 30000


def task_path(task_dir: Path, value: str | Path, *, exists: bool = True) -> Path:
    task_dir = task_dir.resolve()
    path = (task_dir / value).resolve()
    if task_dir not in path.parents:
        raise PostprocessError(f"path must be strictly beneath task directory: {value}")
    if exists and not path.is_file():
        raise PostprocessError(f"missing task-local file: {value}")
    return path


def relative(task_dir: Path, path: Path) -> str:
    return path.resolve().relative_to(task_dir.resolve()).as_posix()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PostprocessError(f"JSON must be an object: {path}")
    return value


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_records(task_dir: Path, artifact: dict) -> tuple[Path, list[dict]]:
    if artifact.get("evidence") == "contract_only" or artifact.get("status") == "contract_only":
        raise PostprocessError("completed_object_meshes cannot be contract-only")
    path = task_path(task_dir, artifact["path"])
    manifest = read_json(path)
    objects = manifest.get("objects")
    if not isinstance(objects, list) or not objects:
        raise PostprocessError("completed_object_meshes requires a nonempty objects list")
    seen = set()
    for item in objects:
        if not isinstance(item, dict):
            raise PostprocessError("each completed object must be an object")
        object_id = item.get("object_id")
        if not isinstance(object_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", object_id):
            raise PostprocessError(f"invalid object_id: {object_id!r}")
        if object_id in seen:
            raise PostprocessError(f"duplicate object_id: {object_id}")
        seen.add(object_id)
        mesh_path = item.get("mesh_path", item.get("mesh", item.get("path")))
        if not isinstance(mesh_path, str):
            raise PostprocessError(f"{object_id}: missing mesh_path")
        task_path(task_dir, mesh_path)
        if item.get("evidence") == "contract_only":
            raise PostprocessError(f"{object_id}: contract-only geometry")
    return path, objects


def coordinate_metadata(task_dir: Path, item: dict, source_path: Path) -> dict:
    """Retain upstream placement evidence without rebinding it to repaired geometry."""
    result = {
        **lineage_metadata(item),
        "object_id": item["object_id"],
        "coordinate_frame": item.get("coordinate_frame", "object_local"),
        "unit": item.get("unit", "unspecified"),
        "room_alignment": item.get("room_alignment", "unknown"),
        "processing_source_geometry_path": relative(task_dir, source_path),
        "processing_source_geometry_sha256": sha256(source_path),
        "processing_coordinate_transform": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    }
    for key in ("observed_anchor", "observed_anchor_applied", "observed_world_bounds", "registration_sha256"):
        if key in item:
            result[key] = copy.deepcopy(item[key])
    for key in ("registration_path", "normalization_metadata_path", "backend_pose_path", "backend_parameters_path"):
        if key in item:
            result[key] = relative(task_dir, task_path(task_dir, item[key]))
    if "registration" in item:
        if not isinstance(item["registration"], dict):
            raise PostprocessError("registration must be a structured object")
        registration = copy.deepcopy(item["registration"])
        for key, value in registration.items():
            if key.endswith("_path"):
                registration[key] = relative(task_dir, task_path(task_dir, value))
        result["registration"] = registration
    if "registration" in result or "registration_path" in result:
        reference = task_path(task_dir, item.get("registration_source_geometry_path", source_path))
        reference_hash = sha256(reference)
        claimed_hash = item.get("registration_source_geometry_sha256")
        if claimed_hash is not None and claimed_hash != reference_hash:
            raise PostprocessError("registration source geometry hash does not match its retained file")
        result.update({
            "registration_source_geometry_path": relative(task_dir, reference),
            "registration_source_geometry_sha256": reference_hash,
            "registration_validation": "upstream_evidence_preserved_not_revalidated_after_mesh_postprocess",
        })
    return result


def check_external_dependencies(task_dir: Path, mesh_path: Path) -> None:
    """Validate sidecar references before the mesh loader opens them."""
    if mesh_path.suffix.lower() == ".glb":
        import struct

        with mesh_path.open("rb") as handle:
            header = handle.read(12)
            if len(header) != 12:
                raise PostprocessError("truncated GLB header")
            magic, version, total_length = struct.unpack("<4sII", header)
            if magic != b"glTF" or version != 2 or total_length != mesh_path.stat().st_size:
                raise PostprocessError("invalid GLB header")
            chunk_header = handle.read(8)
            if len(chunk_header) != 8:
                raise PostprocessError("truncated GLB JSON chunk")
            length, kind = struct.unpack("<I4s", chunk_header)
            if kind != b"JSON" or length > total_length - 20:
                raise PostprocessError("invalid GLB JSON chunk")
            document = json.loads(handle.read(length).decode("utf-8"))
    elif mesh_path.suffix.lower() == ".gltf":
        document = read_json(mesh_path)
    else:
        document = None
    if document is not None:
        for item in document.get("buffers", []) + document.get("images", []):
            uri = item.get("uri", "")
            if uri and not uri.startswith("data:"):
                from urllib.parse import unquote, urlsplit

                if urlsplit(uri).scheme or urlsplit(uri).netloc:
                    raise PostprocessError("external glTF URLs are not task-local")
                task_path(task_dir, mesh_path.parent / unquote(uri))
        return
    if mesh_path.suffix.lower() == ".obj":
        import shlex

        for line in mesh_path.read_text(encoding="utf-8").splitlines():
            fields = shlex.split(line, comments=True)
            if not fields or fields[0] != "mtllib":
                continue
            for name in fields[1:]:
                mtl = task_path(task_dir, mesh_path.parent / name)
                for material_line in mtl.read_text(encoding="utf-8").splitlines():
                    material_fields = shlex.split(material_line, comments=True)
                    if material_fields and material_fields[0].lower().startswith(("map_", "bump", "disp", "decal")):
                        task_path(task_dir, mtl.parent / material_fields[-1])
        return
    if mesh_path.suffix.lower() != ".ply":
        raise PostprocessError("supported mesh inputs are GLB, glTF, OBJ, and PLY")


def geometry_qa(mesh) -> dict:
    import numpy as np

    welded = mesh.copy()
    welded.merge_vertices(merge_tex=True, merge_norm=True)
    finite = bool(np.isfinite(welded.vertices).all())
    return {
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "finite": finite,
        "watertight": bool(welded.is_watertight),
        "winding_consistent": bool(welded.is_winding_consistent),
        "positive_volume": bool(welded.is_volume),
        "convex": bool(welded.is_convex),
        "volume": float(welded.volume),
        "bounds": welded.bounds.tolist() if finite and len(welded.vertices) else None,
    }


def vtk_polydata(mesh):
    import numpy as np
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.asarray(mesh.vertices, dtype=np.float64), deep=True))
    cells = vtk.vtkCellArray()
    packed = np.column_stack((np.full(len(mesh.faces), 3), mesh.faces)).astype(np.int64).ravel()
    cells.SetCells(len(mesh.faces), numpy_to_vtkIdTypeArray(packed, deep=True))
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(cells)
    return polydata


def meshfix_verdict(before: dict, after: dict) -> tuple[str, bool]:
    """Accept a pymeshfix result only when it closes the surface without destroying it.

    pymeshfix can collapse a large non-manifold mesh into a thin sheet; the bed
    completion lost 91% of its volume and 82% of its Y extent that way. The
    pre-repair mesh is retained instead, and the outcome is reported so no
    downstream stage mistakes a degenerate repair for a valid one.
    """
    import numpy as np

    if not after["finite"] or not after["faces"] or after["bounds"] is None or before["bounds"] is None:
        return "rejected_invalid_output", False
    span = np.asarray(before["bounds"][1], dtype=float) - np.asarray(before["bounds"][0], dtype=float)
    diagonal = float(np.linalg.norm(span))
    bounds_change = (float(np.linalg.norm(np.asarray(after["bounds"], dtype=float) - np.asarray(before["bounds"], dtype=float), axis=1).max() / diagonal)
                     if diagonal > 0 else 1.0)
    volume_ratio = float(after["volume"] / before["volume"]) if before["volume"] > 0 else 0.0
    if not after["watertight"]:
        return "rejected_not_watertight", False
    if bounds_change > 0.2:
        return f"rejected_bounds_change_{bounds_change:.4f}", False
    if not 0.5 <= volume_ratio <= 2.0:
        return f"rejected_volume_ratio_{volume_ratio:.4f}", False
    return "accepted", True


def repair_and_simplify(mesh, face_limit: int) -> tuple[object, dict]:
    import numpy as np
    import trimesh

    before = geometry_qa(mesh)
    if not before["finite"] or not before["faces"]:
        raise PostprocessError("mesh contains no triangles or non-finite coordinates")
    work = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), process=True, validate=True)
    trimesh.repair.fix_normals(work, multibody=True)
    trimesh.repair.fill_holes(work)
    meshfix_outcome = "not_needed"
    meshfix_used = False
    if not work.is_watertight:
        if len(work.faces) > PYMESHFIX_FACE_LIMIT:
            # pymeshfix joins components superlinearly: on the 100k-face 3D-Fixer
            # bed it ran for over an hour without converging. Watertightness is
            # preferred, not required -- the visual mesh only needs finite,
            # positively oriented volume and CoACD supplies collision geometry
            # independently -- so an unbounded repair is refused rather than
            # waited on, and the refusal is recorded instead of hidden.
            meshfix_outcome = f"skipped_mesh_over_{PYMESHFIX_FACE_LIMIT}_faces:{len(work.faces)}"
        else:
            import pymeshfix

            meshfix_used = True
            pre_fix = geometry_qa(work)
            fixer = pymeshfix.MeshFix(work.vertices, work.faces)
            fixer.repair(verbose=False, joincomp=False, remove_smallest_components=False)
            candidate = trimesh.Trimesh(fixer.v, fixer.f, process=True, validate=True)
            trimesh.repair.fix_normals(candidate, multibody=True)
            meshfix_outcome, accepted = meshfix_verdict(pre_fix, geometry_qa(candidate))
            if accepted:
                work = candidate
    simplification = "below_face_limit"
    if face_limit and len(work.faces) > face_limit:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy

        decimator = vtk.vtkQuadricDecimation()
        decimator.SetInputData(vtk_polydata(work))
        decimator.SetTargetReduction(1.0 - face_limit / len(work.faces))
        decimator.VolumePreservationOn()
        decimator.Update()
        output = decimator.GetOutput()
        vertices = vtk_to_numpy(output.GetPoints().GetData())
        faces = vtk_to_numpy(output.GetPolys().GetData()).reshape(-1, 4)[:, 1:]
        work = trimesh.Trimesh(vertices=vertices, faces=faces, process=True, validate=True)
        trimesh.repair.fix_normals(work, multibody=True)
        simplification = "vtk_quadric_decimation"
    after = geometry_qa(work)
    # A rejected or refused pymeshfix leaves a positive-volume but non-watertight
    # surface. Closed oriented geometry is preferred, not required: the visual
    # mesh only needs finite, positively-oriented volume, and CoACD supplies the
    # collision hulls independently. Watertightness is reported rather than assumed.
    if not after["finite"] or not after["volume"] > 0:
        raise PostprocessError(f"repair/simplification did not produce positively oriented geometry: {after}")
    if np.any(work.extents <= 0):
        raise PostprocessError("repaired geometry has zero extent")
    return work, {"before": before, "after": after, "meshfix_used": meshfix_used, "meshfix_outcome": meshfix_outcome, "simplification": simplification}


def source_colors(source, triangle_ids, barycentric):
    import numpy as np
    from PIL import Image

    visual = source.visual
    faces = source.faces[triangle_ids]
    if visual.kind == "vertex":
        return np.einsum("ij,ijk->ik", barycentric, visual.vertex_colors[faces]).clip(0, 255).astype(np.uint8)
    if visual.kind == "face":
        return visual.face_colors[triangle_ids]
    if visual.kind != "texture" or visual.uv is None:
        raise PostprocessError("source mesh has no texture or explicit vertex/face colors to bake")
    uv = np.einsum("ij,ijk->ik", barycentric, visual.uv[faces])
    assignments = visual.face_materials
    material_ids = np.zeros(len(triangle_ids), dtype=int) if assignments is None else np.asarray(assignments)[triangle_ids]
    materials = getattr(visual.material, "materials", [visual.material])
    colors = np.empty((len(uv), 4), dtype=np.uint8)
    for material_id in np.unique(material_ids):
        chosen = material_ids == material_id
        material = materials[int(material_id)]
        image = getattr(material, "baseColorTexture", None)
        if image is None:
            image = getattr(material, "image", None)
        factor = getattr(material, "baseColorFactor", None)
        if factor is None:
            factor = getattr(material, "diffuse", [255, 255, 255, 255])
        factor = np.asarray(factor, dtype=np.float64)
        if len(factor) == 3:
            factor = np.append(factor, 255)
        if factor.max() <= 1:
            factor = factor * 255
        if image is None:
            colors[chosen] = np.clip(factor, 0, 255).astype(np.uint8)
        else:
            if not isinstance(image, Image.Image):
                raise PostprocessError("source texture is not a readable image")
            pixels = np.asarray(image.convert("RGBA"), dtype=np.float64)
            sampled_uv = np.mod(uv[chosen], 1.0)
            x = np.rint(sampled_uv[:, 0] * (image.width - 1)).astype(int)
            y = np.rint((1 - sampled_uv[:, 1]) * (image.height - 1)).astype(int)
            colors[chosen] = np.clip(pixels[y, x] * factor / 255, 0, 255).astype(np.uint8)
    return colors


def bake_texture(source, target, destination: Path, resolution: int) -> tuple[object, dict]:
    import numpy as np
    import trimesh
    import vtk
    import xatlas
    from PIL import Image
    from scipy.ndimage import distance_transform_edt

    atlas = xatlas.Atlas()
    atlas.add_mesh(np.asarray(target.vertices, dtype=np.float32), np.asarray(target.faces, dtype=np.uint32))
    chart_options = xatlas.ChartOptions()
    pack_options = xatlas.PackOptions()
    pack_options.resolution = resolution
    pack_options.padding = 2
    atlas.generate(chart_options=chart_options, pack_options=pack_options)
    mapping, faces, uv = atlas[0]
    baked = trimesh.Trimesh(vertices=target.vertices[mapping], faces=faces, process=False)
    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(vtk_polydata(source))
    locator.BuildLocator()
    pixels = np.zeros((resolution, resolution, 4), dtype=np.uint8)
    covered = np.zeros((resolution, resolution), dtype=bool)
    source_triangles = np.asarray(source.triangles)
    closest, cell_id, sub_id, distance = [0.0, 0.0, 0.0], vtk.reference(0), vtk.reference(0), vtk.reference(0.0)
    max_distance = 0.0
    # Rasterize atlas triangles, then sample the nearest point on the original
    # surface. This preserves source appearance after topology has changed.
    for face in np.asarray(faces):
        triangle_uv = uv[face] * (resolution - 1)
        triangle_uv[:, 1] = resolution - 1 - triangle_uv[:, 1]
        lower = np.maximum(np.floor(triangle_uv.min(axis=0)).astype(int), 0)
        upper = np.minimum(np.ceil(triangle_uv.max(axis=0)).astype(int), resolution - 1)
        x, y = np.meshgrid(np.arange(lower[0], upper[0] + 1), np.arange(lower[1], upper[1] + 1))
        coordinates = np.column_stack((x.ravel(), y.ravel())).astype(float)
        a, b, c = triangle_uv
        matrix = np.column_stack((b - a, c - a))
        determinant = np.linalg.det(matrix)
        if abs(determinant) < 1e-8:
            continue
        weights_bc = (coordinates - a) @ np.linalg.inv(matrix).T
        weights = np.column_stack((1 - weights_bc.sum(axis=1), weights_bc))
        inside = (weights >= -1e-7).all(axis=1)
        if not inside.any():
            continue
        coordinates = coordinates[inside].astype(int)
        positions = weights[inside] @ baked.vertices[face]
        ids = np.empty(len(positions), dtype=int)
        points = np.empty_like(positions)
        for index, point in enumerate(positions):
            locator.FindClosestPoint(point, closest, cell_id, sub_id, distance)
            ids[index] = int(cell_id)
            points[index] = closest
            max_distance = max(max_distance, float(distance) ** 0.5)
        barycentric = trimesh.triangles.points_to_barycentric(source_triangles[ids], points)
        colors = source_colors(source, ids, barycentric)
        pixels[coordinates[:, 1], coordinates[:, 0]] = colors
        covered[coordinates[:, 1], coordinates[:, 0]] = True
    if not covered.any():
        raise PostprocessError("UV atlas contains no covered texture pixels")
    nearest = distance_transform_edt(~covered, return_distances=False, return_indices=True)
    pixels[~covered] = pixels[nearest[0][~covered], nearest[1][~covered]]
    image = Image.fromarray(pixels, mode="RGBA")
    image.save(destination)
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=image, metallicFactor=0, roughnessFactor=1)
    baked.visual = trimesh.visual.texture.TextureVisuals(uv=uv, material=material)
    return baked, {
        "status": "basecolor_baked_from_source_surface",
        "resolution": [resolution, resolution],
        "covered_pixels": int(covered.sum()),
        "max_surface_transfer_distance": max_distance,
        "channels": ["baseColor"],
        "material_parameters": "metallic=0 and roughness=1 are rendering defaults, not estimates",
        "normal_metallic_roughness_maps": "not_transferred",
    }


def coacd_hulls(mesh, args) -> tuple[list, dict]:
    import coacd
    import numpy as np
    import trimesh

    # CoACD is a randomised concave decomposition: it occasionally emits a
    # degenerate piece (open surface, non-convex) for an input that decomposed
    # cleanly on another draw. Retry a bounded number of times with fresh seeds
    # instead of failing the whole stage, and keep the failing reports.
    attempts = []
    for attempt in range(3):
        parameters = {
            "threshold": args.coacd_threshold,
            "max_convex_hull": -1,
            "resolution": args.coacd_resolution,
            "mcts_iterations": args.coacd_iterations,
            "seed": args.seed + attempt,
            "preprocess_mode": "off",
            "merge": True,
        }
        pieces = coacd.run_coacd(coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces)), **parameters)
        hulls, reports, invalid = [], [], []
        for vertices, faces in pieces:
            hull = trimesh.Trimesh(vertices=vertices, faces=faces, process=True, validate=True)
            trimesh.repair.fix_normals(hull, multibody=True)
            report = geometry_qa(hull)
            if not all(report[key] for key in ("finite", "watertight", "winding_consistent", "positive_volume", "convex")):
                invalid.append(report)
                continue
            hulls.append(hull)
            reports.append(report)
        attempts.append({"attempt": attempt, "seed": parameters["seed"], "pieces": len(pieces),
                         "valid": len(hulls), "invalid": len(invalid),
                         "first_invalid": invalid[0] if invalid else None})
        if hulls and not invalid:
            return hulls, {
                "method": "coacd",
                "parameters": parameters,
                "hull_count": len(hulls),
                "hulls": reports,
                "simplex_attempts": attempts,
                "concavity": "requested CoACD threshold; not an independently measured Hausdorff bound",
                "collision_eligible": True,
            }
    raise PostprocessError(f"CoACD did not return an all-valid decomposition after {len(attempts)} seeds: {attempts}")


def run(args) -> dict:
    import trimesh

    task_dir = args.task_dir.resolve(strict=True)
    inputs_path = task_path(task_dir, args.inputs)
    outputs_path = task_path(task_dir, args.outputs, exists=False)
    stage_dir = task_dir / "stages/mesh_postprocess"
    task_path(task_dir, stage_dir, exists=False)
    payload = read_json(inputs_path)
    inputs = payload.get("inputs", {})
    if payload.get("module", "mesh_postprocess") != "mesh_postprocess":
        raise PostprocessError("input envelope names a different module")
    if not isinstance(inputs, dict) or not {"completed_object_meshes", "completion_candidates"} <= inputs.keys():
        raise PostprocessError("input envelope requires completed_object_meshes and completion_candidates")
    mesh_manifest_path, objects = object_records(task_dir, inputs["completed_object_meshes"])
    validate_object_lineages(task_dir, objects)
    candidates_path = task_path(task_dir, inputs["completion_candidates"]["path"])
    candidates = read_json(candidates_path)
    if candidates.get("kind", "").endswith("contract-output") or inputs["completion_candidates"].get("evidence") == "contract_only":
        raise PostprocessError("completion_candidates cannot be contract-only")
    candidate_objects = candidates.get("objects")
    if candidate_objects is not None:
        if not isinstance(candidate_objects, list) or not {item["object_id"] for item in objects} <= {
            item.get("object_id") for item in candidate_objects if isinstance(item, dict)
        }:
            raise PostprocessError("completion_candidates does not cover every completed object")
    attempts_dir = stage_dir / "processed"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    output_root = Path(tempfile.mkdtemp(prefix="run-", dir=attempts_dir))
    visual_records, texture_records, collision_records, object_reports = [], [], [], []
    for item in objects:
        object_id = item["object_id"]
        source_path = task_path(task_dir, item.get("mesh_path", item.get("mesh", item.get("path"))))
        common = coordinate_metadata(task_dir, item, source_path)
        check_external_dependencies(task_dir, source_path)
        source_scene = trimesh.load(source_path, force="scene", process=False)
        if not source_scene.geometry:
            raise PostprocessError(f"{object_id}: input contains no geometry")
        object_dir = output_root / object_id
        texture_dir = object_dir / "textures"
        collision_dir = object_dir / "collision"
        texture_dir.mkdir(parents=True)
        collision_dir.mkdir()
        baked_scene = trimesh.Scene()
        repaired_parts = []
        part_reports = []
        textures = []
        for part_index, node in enumerate(source_scene.graph.nodes_geometry):
            transform, geometry_name = source_scene.graph[node]
            source = source_scene.geometry[geometry_name].copy()
            source.apply_transform(transform)
            if not isinstance(source, trimesh.Trimesh):
                raise PostprocessError(f"{object_id}: input geometry is not a triangle mesh")
            print(f"{object_id} part {part_index}: repairing {len(source.faces)} faces", flush=True)
            repaired, report = repair_and_simplify(source, args.face_limit)
            print(f"{object_id} part {part_index}: {report.get('meshfix_outcome', 'reported_by_provider')} "
                  f"{report.get('simplification', 'unknown')} -> {len(repaired.faces)} faces", flush=True)
            texture_path = texture_dir / f"part_{part_index:03d}_basecolor.png"
            baked, texture_report = bake_texture(source, repaired, texture_path, args.texture_resolution)
            baked_scene.add_geometry(baked, node_name=f"part_{part_index:03d}")
            repaired_parts.append(repaired)
            textures.append({"path": relative(task_dir, texture_path), "sha256": sha256(texture_path), **texture_report})
            part_reports.append({"part_index": part_index, **report, "texture": texture_report})
        visual_path = object_dir / "visual.glb"
        baked_scene.export(visual_path)
        reloaded = trimesh.load(visual_path, force="scene", process=False)
        if len(reloaded.geometry) != len(repaired_parts) or any(part.visual.kind != "texture" for part in reloaded.geometry.values()):
            raise PostprocessError(f"{object_id}: exported GLB lost geometry or texture")
        repaired_whole = trimesh.util.concatenate(repaired_parts)
        print(f"{object_id}: decomposing {len(repaired_whole.faces)} faces into collision hulls", flush=True)
        hulls, collision_report = coacd_hulls(repaired_whole, args)
        hull_paths = []
        collision_scene = trimesh.Scene()
        for index, hull in enumerate(hulls):
            path = collision_dir / f"hull_{index:03d}.obj"
            hull.export(path)
            checked = trimesh.load(path, force="mesh", process=True)
            if not checked.is_volume or not checked.is_convex:
                raise PostprocessError(f"{object_id}: exported collision hull failed validation")
            hull_paths.append({"path": relative(task_dir, path), "sha256": sha256(path)})
            collision_scene.add_geometry(hull, node_name=f"hull_{index:03d}", geom_name=f"hull_{index:03d}")
        compound_path = object_dir / "collision.obj"
        collision_scene.export(compound_path, include_texture=False)
        visual_records.append({**common, "mesh_path": relative(task_dir, visual_path), "sha256": sha256(visual_path), "evidence": "derived"})
        texture_records.append({**common, "textures": textures})
        collision_records.append({**common, "mesh_path": relative(task_dir, compound_path), "sha256": sha256(compound_path), "hulls": hull_paths, "collision_eligible": True})
        object_reports.append({**common, "source_mesh": relative(task_dir, source_path), "source_sha256": sha256(source_path), "parts": part_reports, "collision": collision_report})
    manifests = {
        "repaired_visual_meshes": {"objects": visual_records},
        "uv_textures": {"objects": texture_records},
        "coacd_collision_meshes": {"objects": collision_records},
        "mesh_qa_report": {
            "status": "passed",
            "provider": "trimesh_pymeshfix_vtk_xatlas_coacd_cpu",
            "learned_3d_fixer_invoked": False,
            "objects": object_reports,
            "input_manifests": {relative(task_dir, p): sha256(p) for p in (mesh_manifest_path, candidates_path)},
            "versions": {name: importlib.metadata.version(name) for name in ("trimesh", "vtk", "xatlas", "coacd", "pymeshfix", "numpy", "Pillow", "scipy")},
            "parameters": {key: value for key, value in vars(args).items() if not isinstance(value, Path)},
            "placement_and_physics_validated": False,
        },
    }
    outputs = {}
    for role, manifest in manifests.items():
        path = output_root / f"{role}.json"
        write_json(path, {"schema_version": "1.0", "kind": f"video2world-modeling.{role}", **manifest})
        outputs[role] = {"path": relative(task_dir, path), "evidence": "derived", "media_hint": "application/json", "status": "candidate", "collision_eligible": role == "coacd_collision_meshes"}
    result = {"outputs": outputs}
    write_json(outputs_path, result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--face-limit", type=int, default=100000, help="Maximum triangle count per source mesh part; 0 disables simplification")
    parser.add_argument("--texture-resolution", type=int, default=512)
    parser.add_argument("--coacd-threshold", type=float, default=0.05)
    parser.add_argument("--coacd-resolution", type=int, default=2000)
    parser.add_argument("--coacd-iterations", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.face_limit < 0 or 0 < args.face_limit < 4 or not 16 <= args.texture_resolution <= 8192:
        parser.error("face-limit must be 0 or >=4; texture-resolution must be between 16 and 8192")
    if not 0 < args.coacd_threshold < 1 or args.coacd_resolution < 100 or args.coacd_iterations < 1:
        parser.error("invalid CoACD threshold, resolution, or iteration count")
    try:
        result = run(args)
    except (PostprocessError, ImportError, OSError, ValueError, KeyError) as error:
        print(f"mesh_postprocess failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
