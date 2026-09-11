#!/usr/bin/env python3
"""Assemble an auditable scene bundle using verified shared object-to-world Sim(3)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from world_modeling.gaussian_io import read_gaussian_rows
from world_modeling.modules.scene_recomposition import SPEC
from world_modeling.provider_io import input_artifact, local_path, parser, publish, read_json, write_json


_POSTPROCESS_SPEC = importlib.util.spec_from_file_location("world_modeling_mesh_helpers", Path(__file__).with_name("mesh_postprocess.py"))
mesh_helpers = importlib.util.module_from_spec(_POSTPROCESS_SPEC)
_POSTPROCESS_SPEC.loader.exec_module(mesh_helpers)


class RecompositionError(RuntimeError):
    pass


def relative(task_dir: Path, path: Path) -> str:
    return path.resolve().relative_to(task_dir.resolve()).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_reference(task_dir: Path, path: Path) -> dict:
    path = local_path(task_dir, path)
    if not path.is_file():
        raise RecompositionError(f"scene bundle references require a file: {path}")
    return {"path": relative(task_dir, path), "sha256": sha256(path), "size_bytes": path.stat().st_size}


def object_index(document: dict, name: str) -> dict[str, dict]:
    import re

    objects = document.get("objects")
    if not isinstance(objects, list) or not objects:
        raise RecompositionError(f"{name} requires a nonempty objects list")
    result = {}
    for record in objects:
        object_id = record.get("object_id") if isinstance(record, dict) else None
        if not isinstance(object_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", object_id) or object_id in result:
            raise RecompositionError(f"{name}: invalid or duplicate object_id {object_id!r}")
        result[object_id] = record
    return result


def gaussian_table(path: Path, *, require_gaussians: bool = True):
    """Read a Gaussian/point PLY, dropping non-finite rows and reporting how many."""
    required = ("x", "y", "z", "f_dc_0", "opacity", "scale_0", "rot_0") if require_gaussians else ("x", "y", "z")
    cloud = read_gaussian_rows(path, required=required, error=RecompositionError)
    return cloud.rows, cloud.points, cloud.dropped


def scene_geometry(task_dir: Path, path: Path):
    import numpy as np
    import trimesh

    try:
        mesh_helpers.check_external_dependencies(task_dir, path)
    except mesh_helpers.PostprocessError as error:
        raise RecompositionError(str(error)) from error
    loaded = trimesh.load(path, force="scene", process=False)
    parts = []
    # Registration source indices explicitly refer to this stable baked-node ordering.
    for node in sorted(loaded.graph.nodes_geometry):
        transform, name = loaded.graph[node]
        part = loaded.geometry[name].copy()
        if not isinstance(part, trimesh.Trimesh) or not len(part.faces):
            raise RecompositionError(f"asset contains empty/non-triangle geometry: {path}")
        part.apply_transform(transform)
        if not np.isfinite(part.vertices).all():
            raise RecompositionError(f"asset has non-finite geometry: {path}")
        parts.append(part)
    if not parts:
        raise RecompositionError(f"mesh asset has no geometry: {path}")
    return parts, np.concatenate([part.vertices for part in parts])


def sim3(value) -> tuple[object, float]:
    import numpy as np

    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8):
        raise RecompositionError("object_to_world must be a finite homogeneous 4x4 transform")
    determinant = float(np.linalg.det(matrix[:3, :3]))
    if determinant <= 0:
        raise RecompositionError("object_to_world requires positive scale and a right-handed rotation")
    scale = determinant ** (1.0 / 3.0)
    rotation = matrix[:3, :3] / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=1e-5, atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6):
        raise RecompositionError("object_to_world must use uniform Sim(3) scale, not shear or per-axis scaling")
    return matrix, scale


def registration_record(task_dir: Path, visual: dict) -> dict:
    if "registration_path" in visual:
        path = local_path(task_dir, visual["registration_path"])
        if visual.get("registration_sha256") and visual["registration_sha256"] != sha256(path):
            raise RecompositionError("registration file hash changed")
        registration = read_json(path)
    else:
        registration = visual.get("registration")
    if not isinstance(registration, dict):
        raise RecompositionError("room alignment is unknown: a hash-bound correspondence registration is required")
    if registration.get("status") not in {"accepted", "correspondences_available"}:
        raise RecompositionError("registration is not accepted and has no validated correspondence proposal")
    if registration.get("method") not in {"observed_correspondence_sim3", "point_correspondence_sim3", "manual_correspondence_sim3", "ransac_correspondence_sim3"}:
        raise RecompositionError("registration must use observed point correspondences; bbox placement is not accepted")
    return registration


def validate_registration(task_dir: Path, object_id: str, visual: dict, observed: dict, world: dict, source_scene_sha256: str, tolerance: float, repair_bounds_ratio: float) -> tuple[dict, dict]:
    import numpy as np

    registration = registration_record(task_dir, visual)
    if registration.get("object_id", object_id) != object_id:
        raise RecompositionError("registration belongs to a different physical object")
    source_units = visual.get("unit", visual.get("units"))
    source_frame = visual.get("coordinate_frame")
    if not source_frame or not source_units or source_units in {"unknown", "unspecified"}:
        raise RecompositionError("object asset must declare its coordinate frame and units")
    expected = {
        "source_coordinate_frame": source_frame, "source_units": source_units,
        "target_coordinate_frame": world["coordinate_frame"], "target_units": world["units"],
    }
    if any(registration.get(key) != value for key, value in expected.items()):
        raise RecompositionError("registration source/target coordinate frames or units disagree with artifacts")
    repaired_path = local_path(task_dir, visual.get("mesh_path", visual.get("path")))
    source_path = local_path(task_dir, visual.get("registration_source_geometry_path", registration.get("source_geometry_path", repaired_path)))
    source_digest = sha256(source_path)
    target_path = local_path(task_dir, observed["ply_path"])
    target_digest = sha256(target_path)
    if registration.get("source_geometry_sha256") != source_digest or registration.get("target_object_ply_sha256") != target_digest or registration.get("target_scene_sha256") != source_scene_sha256:
        raise RecompositionError("registration is not hash-bound to current source geometry, observed object and source scene")
    if visual.get("registration_source_geometry_sha256", source_digest) != source_digest:
        raise RecompositionError("postprocessed registration source hash disagrees")
    if source_path != repaired_path:
        processing = np.asarray(visual.get("processing_coordinate_transform"), dtype=float)
        if processing.shape != (4, 4) or not np.allclose(processing, np.eye(4), atol=1e-8):
            raise RecompositionError("registration requires an explicit identity postprocessing coordinate transform")
    _, source_vertices = scene_geometry(task_dir, source_path)
    _, current_vertices = scene_geometry(task_dir, repaired_path)
    source_bounds = np.array([source_vertices.min(0), source_vertices.max(0)])
    current_bounds = np.array([current_vertices.min(0), current_vertices.max(0)])
    diagonal = float(np.linalg.norm(source_bounds[1] - source_bounds[0]))
    if diagonal <= 0:
        raise RecompositionError("registration source geometry has zero extent")
    bounds_change = float(np.linalg.norm(current_bounds - source_bounds, axis=1).max() / diagonal)
    if bounds_change > repair_bounds_ratio:
        raise RecompositionError("postprocessed asset bounds changed too far to reuse source-coordinate registration")
    _, target_points, _ = gaussian_table(target_path)
    proof_path = local_path(task_dir, registration["correspondences_path"])
    if registration.get("correspondences_sha256") != sha256(proof_path):
        raise RecompositionError("registration correspondence proof hash changed")
    proof = read_json(proof_path)
    if proof.get("object_id") != object_id or proof.get("source_indexing") != "trimesh_sorted_scene_nodes_baked_vertices" or proof.get("target_indexing") != "ply_vertex_row":
        raise RecompositionError("correspondence proof must identify the object and explicit source/target index conventions")
    pairs = proof.get("pairs")
    if not isinstance(pairs, list) or len(pairs) < 6:
        raise RecompositionError("registration requires at least three fit and three independent validation correspondences")
    source_indices, target_indices, splits = [], [], []
    for pair in pairs:
        source_index, target_index = pair.get("source_vertex_index"), pair.get("target_gaussian_index")
        if not isinstance(source_index, int) or isinstance(source_index, bool) or not 0 <= source_index < len(source_vertices):
            raise RecompositionError("correspondence source vertex index is invalid")
        if not isinstance(target_index, int) or isinstance(target_index, bool) or not 0 <= target_index < len(target_points):
            raise RecompositionError("correspondence observed Gaussian index is invalid")
        if pair.get("split") not in {"fit", "validation"}:
            raise RecompositionError("correspondences require explicit fit/validation splits")
        source_indices.append(source_index)
        target_indices.append(target_index)
        splits.append(pair["split"])
    if len(set(source_indices)) != len(pairs) or len(set(target_indices)) != len(pairs):
        raise RecompositionError("fit and validation correspondences must use independent source and target indices")
    source_points, targets = source_vertices[source_indices], target_points[target_indices]
    fit = np.asarray([split == "fit" for split in splits])
    for group in (fit, ~fit):
        if int(group.sum()) < 3 or np.linalg.matrix_rank(source_points[group] - source_points[group].mean(0), tol=1e-8) < 2 or np.linalg.matrix_rank(targets[group] - targets[group].mean(0), tol=1e-8) < 2:
            raise RecompositionError("fit and validation correspondences must each span non-collinear geometry")
    solved = "object_to_world" not in registration
    if solved:
        import open3d as o3d

        source_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_points[fit]))
        target_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(targets[fit]))
        identity_pairs = o3d.utility.Vector2iVector(np.column_stack((np.arange(fit.sum()), np.arange(fit.sum()))).astype(np.int32))
        matrix = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True).compute_transformation(source_cloud, target_cloud, identity_pairs)
    else:
        matrix = registration["object_to_world"]
    matrix, scale = sim3(matrix)
    residuals = np.linalg.norm(source_points @ matrix[:3, :3].T + matrix[:3, 3] - targets, axis=1)
    declared_tolerance = float(registration.get("max_error_world_units", tolerance))
    if not np.isfinite(declared_tolerance) or not 0 < declared_tolerance <= tolerance:
        raise RecompositionError("registration error allowance exceeds the scene-derived acceptance threshold")
    errors = {}
    for name, group in (("fit", fit), ("validation", ~fit)):
        values = residuals[group]
        errors[name] = {"count": int(group.sum()), "rmse_world_units": float(np.sqrt(np.mean(values ** 2))), "max_world_units": float(values.max()), "inlier_fraction": float(np.mean(values <= declared_tolerance))}
        if errors[name]["rmse_world_units"] > declared_tolerance or errors[name]["max_world_units"] > declared_tolerance * 3 or errors[name]["inlier_fraction"] < 0.9:
            raise RecompositionError(f"registration {name} residuals exceed observed-geometry acceptance threshold")
    transform = {
        "object_id": object_id, "object_to_world": matrix.tolist(), "uniform_scale": scale, **expected,
        "matrix_convention": "column_vectors_world_equals_object_to_world_times_object", "scale_estimated_from_observed_correspondences": True,
    }
    transform["sha256"] = hashlib.sha256(json.dumps(transform, sort_keys=True).encode("utf-8")).hexdigest()
    return transform, {
        "status": "numeric_correspondences_verified", "method": registration["method"], "solved_with_open3d": solved,
        "correspondences": file_reference(task_dir, proof_path), "source_geometry": file_reference(task_dir, source_path),
        "observed_object": file_reference(task_dir, target_path), "errors": errors, "threshold_world_units": declared_tolerance,
        "postprocessing_bounds_change_ratio": bounds_change, "bounds_used_for": "coordinate_consistency_check_only_not_alignment",
        "semantic_correspondence_review": registration.get("semantic_correspondence_review", "not_recorded"),
        "input_registration": registration,
    }


def sample_visual_points(task_dir: Path, mesh_path: Path, output: Path, count: int, seed: int) -> dict:
    import numpy as np
    import trimesh

    parts, _ = scene_geometry(task_dir, mesh_path)
    areas = np.asarray([part.area for part in parts])
    if not np.isfinite(areas).all() or (areas <= 0).any() or count < len(parts):
        raise RecompositionError("visual point sampling requires positive mesh areas and enough samples for every part")
    allocations = np.maximum(1, np.floor(count * areas / areas.sum()).astype(int))
    while allocations.sum() > count:
        allocations[np.argmax(allocations)] -= 1
    allocations[np.argmax(areas)] += count - allocations.sum()
    points, colors = [], []
    for index, (part, size) in enumerate(zip(parts, allocations)):
        sampled, faces = trimesh.sample.sample_surface(part, int(size), seed=seed + index)
        barycentric = trimesh.triangles.points_to_barycentric(part.triangles[faces], sampled)
        try:
            sampled_colors = mesh_helpers.source_colors(part, faces, barycentric)
        except mesh_helpers.PostprocessError as error:
            raise RecompositionError(str(error)) from error
        points.append(sampled)
        colors.append(sampled_colors)
    cloud = trimesh.PointCloud(np.concatenate(points), colors=np.concatenate(colors))
    cloud.export(output)
    checked = trimesh.load(output, process=False)
    if len(checked.vertices) != count or not np.isfinite(checked.vertices).all():
        raise RecompositionError("visual object PLY failed point-count/finite-coordinate validation")
    return {**file_reference(task_dir, output), "point_count": count, "kind": "colored_surface_points_not_gaussians", "transform_applied": False}


def validate_physical_assets(task_dir: Path, visual: dict, physical: dict) -> dict:
    import numpy as np
    import trimesh

    units = visual.get("unit", visual.get("units"))
    coordinate = visual.get("coordinate_frame")
    if physical.get("units") != units or physical.get("coordinate_frame") != coordinate or physical.get("scale_applied") is not False:
        raise RecompositionError("visual and physical assets must share unchanged object coordinates and explicit units")
    visual_path = local_path(task_dir, visual.get("mesh_path", visual.get("path")))
    obj_path = local_path(task_dir, physical["obj_path"])
    if obj_path.suffix.lower() != ".obj" or local_path(task_dir, physical["source_mesh_path"]) != visual_path:
        raise RecompositionError("physical_object_obj must reference an OBJ exported from the current repaired visual mesh")
    _, visual_vertices = scene_geometry(task_dir, visual_path)
    _, obj_vertices = scene_geometry(task_dir, obj_path)
    visual_bounds = np.array([visual_vertices.min(0), visual_vertices.max(0)])
    obj_bounds = np.array([obj_vertices.min(0), obj_vertices.max(0)])
    if not np.allclose(visual_bounds, obj_bounds, rtol=1e-5, atol=1e-6):
        raise RecompositionError("physical OBJ export changed object coordinates")
    collision = physical.get("collision", {})
    if collision.get("collision_eligible") is not True or collision.get("coordinate_frame") != coordinate or collision.get("unit", collision.get("units")) != units:
        raise RecompositionError("collision assets need independent eligibility and the same object coordinates/units")
    hulls = collision.get("hulls")
    if not isinstance(hulls, list) or not hulls:
        raise RecompositionError("CoACD must provide a nonempty list of separately validated convex OBJ hulls")
    checked_hulls = []
    for hull in hulls:
        path = local_path(task_dir, hull["path"])
        if path.suffix.lower() != ".obj" or sha256(path) != hull.get("sha256"):
            raise RecompositionError("collision hull format/hash is invalid")
        try:
            mesh_helpers.check_external_dependencies(task_dir, path)
        except mesh_helpers.PostprocessError as error:
            raise RecompositionError(str(error)) from error
        mesh = trimesh.load(path, force="mesh", process=True)
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces) or not np.isfinite(mesh.vertices).all() or not mesh.is_convex or not mesh.is_volume:
            raise RecompositionError("CoACD collision hull is not a finite closed oriented convex mesh")
        checked_hulls.append({**file_reference(task_dir, path), "vertices": len(mesh.vertices), "faces": len(mesh.faces), "convex": True, "closed_oriented": True})
    return {"visual_obj": file_reference(task_dir, obj_path), "hulls": checked_hulls, "collision_eligible": True, "world_transform_applied": False}


def validate_scene_inputs(args, paths: dict) -> tuple[dict, dict, dict, dict, dict]:
    cameras, lifting = read_json(paths["cameras"]), read_json(paths["lifting_report"])
    if not cameras.get("coordinate_frame") or not cameras.get("units") or cameras["units"] in {"unknown", "unspecified"}:
        raise RecompositionError("source cameras must declare scene coordinate frame and units")
    world = {"coordinate_frame": cameras["coordinate_frame"], "units": cameras["units"], "metric_scale_known": cameras.get("metric_scale_known") is True}
    if lifting.get("coordinate_frame") != world["coordinate_frame"] or lifting.get("units") != world["units"] or lifting.get("carve_performed") is not True:
        raise RecompositionError("lifting report must prove carving in the same source scene coordinates")
    source_digest = sha256(paths["scene_gaussian_ply"])
    if lifting.get("source_files", {}).get("scene_gaussian_ply", {}).get("sha256") != source_digest or lifting.get("carved_scene_sha256") != sha256(paths["carved_scene_ply"]):
        raise RecompositionError("lifting report does not bind the current source/carved Gaussian PLY hashes")
    source_rows, _, _ = gaussian_table(paths["scene_gaussian_ply"])
    carved_rows, _, _ = gaussian_table(paths["carved_scene_ply"])
    gaussian_table(paths["generated_background_gaussian_ply"])
    if source_rows.dtype != carved_rows.dtype or len(source_rows) != lifting.get("source_gaussian_count") or len(carved_rows) != lifting.get("remaining_gaussian_count") or len(source_rows) - len(carved_rows) != lifting.get("removed_gaussian_count"):
        raise RecompositionError("carved Gaussian attributes/counts disagree with the lifting receipt")
    observed = object_index(read_json(paths["isolated_object_ply"]), "isolated_object_ply")
    visual = object_index(read_json(paths["repaired_visual_meshes"]), "repaired_visual_meshes")
    physical = object_index(read_json(paths["physical_object_obj"]), "physical_object_obj")
    properties = object_index(read_json(paths["physics_properties"]), "physics_properties")
    if set(observed) != set(visual) or set(observed) != set(physical) or set(observed) != set(properties) or set(observed) != set(lifting.get("accepted_object_ids", [])):
        raise RecompositionError("every carved physical object must have exactly one repaired visual mesh, OBJ, and physics record")
    for object_id, record in observed.items():
        if record.get("association_status") != "geometrically_verified" or record.get("coordinate_frame") != world["coordinate_frame"] or record.get("units") != world["units"]:
            raise RecompositionError(f"{object_id}: observed target must be a geometrically verified physical instance in scene coordinates")
        if sha256(local_path(args.task_dir, record["ply_path"])) != record.get("sha256"):
            raise RecompositionError(f"{object_id}: observed isolated PLY hash changed")
    return world, lifting, observed, visual, {"physical": physical, "properties": properties, "source_scene_sha256": source_digest}


def run(args) -> dict:
    import numpy as np

    args.task_dir = args.task_dir.resolve()
    outputs_path = local_path(args.task_dir, args.outputs, exists=False)
    outputs_path.parent.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=outputs_path.parent))
    destinations = {role.name: run_dir / f"{role.name}.json" for role in SPEC.outputs}
    report = {"schema_version": "1.0", "status": "validating", "objects": [], "errors": [], "promotion_allowed": False}
    world_manifest = {"schema_version": "1.0", "status": "blocked", "path_base": "task_root", "transforms": {}, "objects": [], "promotion_allowed": False}
    visual_manifest = {"schema_version": "1.0", "path_base": "task_root", "backgrounds": [], "objects": []}
    collision_manifest = {"schema_version": "1.0", "path_base": "task_root", "objects": [], "static_background_collision": None, "simulation_ready": False}
    try:
        paths = {role: input_artifact(args, role) for role in SPEC.inputs}
        world, lifting, observed, visual, records = validate_scene_inputs(args, paths)
        tolerance = args.max_registration_error_world_units
        if tolerance is None:
            tolerance = float(lifting.get("voxel_size", 0)) * 3
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise RecompositionError("a positive scene-unit registration tolerance or lifting voxel_size is required")
        world_manifest.update(world)
        world_manifest["source_artifacts"] = {name: file_reference(args.task_dir, path) for name, path in paths.items()}
        visual_manifest.update(world)
        collision_manifest.update(world)
        visual_manifest["backgrounds"] = [
            {"id": "observed_carved", **file_reference(args.task_dir, paths["carved_scene_ply"]), "kind": "pgsr_gaussians", "evidence": "derived_observed", "enabled": True, "object_to_world": np.eye(4).tolist()},
            {"id": "generated_background_candidate", **file_reference(args.task_dir, paths["generated_background_gaussian_ply"]), "kind": "pgsr_gaussians", "evidence": "generated", "enabled": False, "reason_disabled": "Full-scene generated splats cannot be overlaid as hole-only geometry without additional visibility evidence."},
        ]
        for object_id in sorted(observed):
            object_report = {"object_id": object_id, "status": "blocked", "errors": []}
            report["objects"].append(object_report)
            try:
                visual_record = visual[object_id]
                mesh_path = local_path(args.task_dir, visual_record.get("mesh_path", visual_record.get("path")))
                if visual_record.get("sha256") and sha256(mesh_path) != visual_record["sha256"]:
                    raise RecompositionError("repaired visual mesh hash changed")
                transform, registration_report = validate_registration(args.task_dir, object_id, visual_record, observed[object_id], world, records["source_scene_sha256"], tolerance, args.max_repair_bounds_change_ratio)
                physical = validate_physical_assets(args.task_dir, visual_record, records["physical"][object_id])
                object_dir = run_dir / "objects" / object_id
                object_dir.mkdir(parents=True)
                point_cloud = sample_visual_points(args.task_dir, mesh_path, object_dir / "visual_points.ply", args.visual_points, args.seed)
                world_manifest["transforms"][object_id] = transform
                shared = {"object_id": object_id, "transform_id": object_id, "transform_sha256": transform["sha256"], "transform_manifest": relative(args.task_dir, destinations["world_manifest"])}
                visual_manifest["objects"].append({**shared, "enabled": True, "ply": point_cloud, "repaired_mesh": file_reference(args.task_dir, mesh_path), "visual_obj": physical["visual_obj"], "evidence": "derived_from_generated_completed_mesh"})
                collision_manifest["objects"].append({**shared, "hulls": physical["hulls"], "collision_eligible": True, "transform_applied": False, "physics_properties": records["properties"][object_id]})
                world_manifest["objects"].append({**shared, "placement_status": "numeric_correspondences_verified", "registration": registration_report, "source_coordinate_frame": transform["source_coordinate_frame"], "source_units": transform["source_units"]})
                object_report.update(status="assembled", registration=registration_report, visual_point_count=args.visual_points, collision_hull_count=len(physical["hulls"]))
            except (RecompositionError, ValueError, KeyError, OSError) as error:
                object_report["errors"].append(str(error))
                report["errors"].append(f"{object_id}: {error}")
                world_manifest["objects"].append({"object_id": object_id, "placement_status": "blocked", "object_to_world": None, "error": str(error), "repaired_visual_record": visual[object_id]})
                visual_manifest["objects"].append({"object_id": object_id, "enabled": False, "transform_id": None, "reason_disabled": str(error)})
        report.update({
            "status": "blocked_registration_or_assets" if report["errors"] else "assembled_candidate",
            "world_units": world["units"], "scene_metric_scale_known": world["metric_scale_known"],
            "registration_tolerance_world_units": tolerance, "automatic_semantic_registration_performed": False,
            "background_completion_promoted": False, "static_background_collision_available": False,
            "physics_calibrated": False, "simulator_validation": "not_performed",
            "lifting_status": lifting.get("status"), "background_used": "carved_observed_pgsr",
        })
        world_manifest["status"] = report["status"]
        world_manifest["visual_scene_manifest"] = relative(args.task_dir, destinations["visual_scene_manifest"])
        world_manifest["collision_scene_manifest"] = relative(args.task_dir, destinations["collision_scene_manifest"])
        world_manifest["recomposition_report"] = relative(args.task_dir, destinations["recomposition_report"])
        collision_manifest["readiness_blockers"] = ["No static background collider is provided.", "Physical properties remain uncalibrated image/category priors."]
        if not world["metric_scale_known"]:
            collision_manifest["readiness_blockers"].append("World units are not calibrated to meters.")
    except (RecompositionError, ValueError, KeyError, OSError) as error:
        report.update(status="blocked_input_validation", errors=[str(error)])
        world_manifest.update(status="blocked_input_validation", errors=[str(error)])
    for role, document in (("world_manifest", world_manifest), ("visual_scene_manifest", visual_manifest), ("collision_scene_manifest", collision_manifest), ("recomposition_report", report)):
        write_json(destinations[role], document)
    if report["errors"]:
        raise RecompositionError(f"scene assembly blocked; audit {relative(args.task_dir, destinations['recomposition_report'])}: {'; '.join(report['errors'])}")
    publish(args, destinations)
    return {"outputs": {role: relative(args.task_dir, path) for role, path in destinations.items()}, "status": report["status"]}


def main(argv=None) -> int:
    app = parser(__doc__)
    app.add_argument("--visual-points", type=int, default=50000)
    app.add_argument("--seed", type=int, default=0)
    app.add_argument("--max-registration-error-world-units", type=float)
    app.add_argument("--max-repair-bounds-change-ratio", type=float, default=0.2)
    args = app.parse_args(argv)
    if args.visual_points < 1 or not 0 <= args.max_repair_bounds_change_ratio < 1:
        app.error("visual point count must be positive and repair bounds ratio must be in [0,1)")
    try:
        print(json.dumps(run(args), indent=2))
    except (RecompositionError, ImportError, ValueError, OSError, KeyError) as error:
        print(f"scene_recomposition failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
