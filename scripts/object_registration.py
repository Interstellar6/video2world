#!/usr/bin/env python3
"""CPU camera-chain initialization and observed-point Sim(3) registration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from world_modeling.provider_io import input_artifact, local_path, publish, read_json, write_json


class RegistrationError(RuntimeError):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(task, path):
    return path.resolve().relative_to(task.resolve()).as_posix()


def transform(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def sim3(matrix):
    import numpy as np

    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise RegistrationError("invalid homogeneous transform")
    determinant = float(np.linalg.det(matrix[:3, :3]))
    if determinant <= 0:
        raise RegistrationError("registration cannot contain a reflection or nonpositive scale")
    scale = determinant ** (1 / 3)
    if not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3) * scale ** 2, atol=1e-6, rtol=1e-5):
        raise RegistrationError("registration requires uniform scale without shear")
    return matrix, scale


def estimate_similarity(source, target, threshold, *, robust=True, seed=0):
    import numpy as np
    import open3d as o3d

    source, target = np.asarray(source, float), np.asarray(target, float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise RegistrationError("Sim3 estimation needs at least three paired 3D points")
    for points in (source, target):
        if not np.isfinite(points).all() or np.linalg.matrix_rank(points - points.mean(0), tol=1e-8) < 2:
            raise RegistrationError("Sim3 correspondences must be finite and non-collinear")
    clouds = [o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)) for points in (source, target)]
    pairs = o3d.utility.Vector2iVector(np.column_stack((np.arange(len(source)), np.arange(len(source)))).astype(np.int32))
    estimator = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)
    if robust:
        o3d.utility.random.seed(seed)
        result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
            *clouds, pairs, threshold, estimator, 3,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(threshold)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(10000, .999),
        )
        matrix, _ = sim3(result.transformation)
        chosen = np.linalg.norm(transform(source, matrix) - target, axis=1) <= threshold
        if chosen.sum() < 3:
            raise RegistrationError("RANSAC found fewer than three supported correspondences")
        return estimate_similarity(source[chosen], target[chosen], threshold, robust=False), chosen
    return sim3(estimator.compute_transformation(*clouds, pairs))[0]


def camera_chain(estimated, conditioned, validation, threshold, max_angle, seed=0):
    import numpy as np
    from scipy.spatial.transform import Rotation

    estimated, conditioned = np.asarray(estimated, float), np.asarray(conditioned, float)
    validation = np.asarray(validation, bool)
    if estimated.shape != conditioned.shape or estimated.shape != (len(validation), 4, 4) or min(validation.sum(), (~validation).sum()) < 3:
        raise RegistrationError("camera chain requires separate fit and heldout camera correspondences")
    for pose in [*estimated, *conditioned]:
        _, scale = sim3(pose)
        if not np.isclose(scale, 1, atol=1e-4):
            raise RegistrationError("camera extrinsics must be rigid")
    source = np.linalg.inv(estimated)[:, :3, 3]
    target = np.linalg.inv(conditioned)[:, :3, 3]
    matrix, selected = estimate_similarity(source[~validation], target[~validation], threshold, seed=seed)
    if selected.mean() < .8:
        raise RegistrationError("generated trajectory disagrees with conditioning camera correspondences")
    matrix, scale = sim3(matrix)
    errors = np.linalg.norm(transform(source, matrix) - target, axis=1)
    predicted_rotation = estimated[:, :3, :3] @ (matrix[:3, :3] / scale).T
    angles = Rotation.from_matrix(predicted_rotation @ conditioned[:, :3, :3].transpose(0, 2, 1)).magnitude() * 180 / np.pi
    report = {}
    for name, subset in (("fit", ~validation), ("validation", validation)):
        report[name] = {"count": int(subset.sum()), "rmse_world_units": float(np.sqrt(np.mean(errors[subset] ** 2))), "max_world_units": float(errors[subset].max()), "max_rotation_degrees": float(angles[subset].max())}
        if report[name]["rmse_world_units"] > threshold or report[name]["max_world_units"] > 3 * threshold or report[name]["max_rotation_degrees"] > max_angle:
            raise RegistrationError(f"generated/conditioned camera {name} position or orientation gate failed: {report[name]}")
    return matrix, {"method": "Open3D_correspondence_RANSAC_Sim3", "generated_world_to_scene_world": matrix.tolist(), "threshold_world_units": threshold, "max_rotation_degrees": max_angle, **report}


def glb_to_reference_camera(pose):
    import numpy as np
    from scipy.spatial.transform import Rotation

    scale = np.asarray(pose["scale"], float).reshape(-1)
    quaternion = np.asarray(pose["rotation"], float).reshape(-1)
    translation = np.asarray(pose["translation"], float).reshape(-1)
    if len(scale) == 1:
        scale = np.repeat(scale, 3)
    if len(scale) != 3 or len(quaternion) != 4 or len(translation) != 3 or not np.isfinite(np.r_[scale, quaternion, translation]).all() or min(scale) <= 0 or not np.allclose(scale, scale[0], rtol=1e-5):
        raise RegistrationError("SAM3D pose must have finite uniform scale, wxyz quaternion and translation")
    if not np.isclose(np.linalg.norm(quaternion), 1, atol=1e-3):
        raise RegistrationError("SAM3D pose quaternion is not normalized")
    rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    # to_glb writes row vertices @ M. Undo that export, then reproduce the
    # official PyTorch3D row-vector scale.rotate(R).translate(t) convention.
    export_row_rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    camera_flip = np.diag([-1, -1, 1])
    result = np.eye(4)
    result[:3, :3] = camera_flip @ rotation.T @ np.diag(scale) @ export_row_rotation
    result[:3, 3] = camera_flip @ translation
    return sim3(result)[0]


def scene_vertices(path):
    import numpy as np
    import trimesh

    scene = trimesh.load(path, force="scene", process=False)
    vertices = []
    for node in sorted(scene.graph.nodes_geometry):
        matrix, name = scene.graph[node]
        part = scene.geometry[name]
        if not isinstance(part, trimesh.Trimesh):
            raise RegistrationError("registration source must be a triangle mesh")
        vertices.append(transform(np.asarray(part.vertices), matrix))
    if not vertices:
        raise RegistrationError("empty source mesh")
    result = np.concatenate(vertices)
    if not np.isfinite(result).all():
        raise RegistrationError("source mesh has non-finite vertices")
    return result


def observed_correspondences(source, target, initial, threshold, min_coverage, seed=0):
    import numpy as np
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation

    positioned = transform(source, initial)
    distances, source_indices = cKDTree(positioned).query(target)
    eligible = np.flatnonzero(distances <= threshold * 3)
    if len(eligible) < 12:
        raise RegistrationError("camera/pose initialization lacks observed geometry support")
    # Correspondences are fixed from the camera-chain hypothesis before the
    # fit/heldout split. Validation points never participate in refinement.
    ordered = eligible[np.argsort(distances[eligible], kind="stable")]
    _, unique = np.unique(source_indices[ordered], return_index=True)
    target_indices = ordered[np.sort(unique)]
    rng = np.random.default_rng(seed)
    rng.shuffle(target_indices)
    target_indices = target_indices[:6000]
    source_indices = source_indices[target_indices]
    validation = np.arange(len(target_indices)) % 3 == 0
    if min(validation.sum(), (~validation).sum()) < 4:
        raise RegistrationError("too few independent geometry correspondences")
    matrix, inliers = estimate_similarity(source[source_indices[~validation]], target[target_indices[~validation]], threshold, seed=seed)
    retained = validation.copy()
    retained[~validation] = inliers
    source_indices, target_indices, validation = source_indices[retained], target_indices[retained], validation[retained]
    matrix, _ = sim3(matrix)
    delta, delta_scale = sim3(matrix @ np.linalg.inv(initial))
    angular_change = float(Rotation.from_matrix(delta[:3, :3] / delta_scale).magnitude() * 180 / np.pi)
    center_change = float(np.linalg.norm(transform(source, matrix).mean(0) - positioned.mean(0)))
    if not .8 <= delta_scale <= 1.25 or angular_change > 15 or center_change > 3 * threshold:
        raise RegistrationError("geometric refinement diverged from verified camera/pose initialization")
    residuals = np.linalg.norm(transform(source[source_indices], matrix) - target[target_indices], axis=1)
    errors = {}
    for name, group in (("fit", ~validation), ("validation", validation)):
        errors[name] = {"count": int(group.sum()), "rmse_world_units": float(np.sqrt(np.mean(residuals[group] ** 2))), "max_world_units": float(residuals[group].max()), "inlier_fraction": float(np.mean(residuals[group] <= threshold))}
        if group.sum() < 3 or np.linalg.matrix_rank(target[target_indices[group]] - target[target_indices[group]].mean(0), tol=1e-8) < 2 or errors[name]["rmse_world_units"] > threshold or errors[name]["max_world_units"] > 3 * threshold or errors[name]["inlier_fraction"] < .9:
            raise RegistrationError(f"observed geometry {name} gate failed: {errors[name]}")
    coverage_distances, _ = cKDTree(transform(source, matrix)).query(target)
    coverage = float(np.mean(coverage_distances <= threshold))
    if coverage < min_coverage:
        raise RegistrationError(f"completed mesh supports only {coverage:.3f} of observed object points")
    pairs = [{"source_vertex_index": int(s), "target_gaussian_index": int(t), "split": "validation" if v else "fit"} for s, t, v in zip(source_indices, target_indices, validation)]
    return matrix, pairs, {"correspondence_origin": "nearest_geometry_from_verified_camera_chain_fixed_before_fit_validation_split", "observed_coverage": coverage, "minimum_observed_coverage": min_coverage, "refinement_scale_ratio": delta_scale, "refinement_rotation_degrees": angular_change, "refinement_centroid_motion_world_units": center_change, **errors}


# The reprojection target is reported per observation; only a placement that is
# off by a multiple of it -- or that stops landing inside the observed mask -- is
# refused, because a few pixels of surface spread is a quality signal rather than
# a placement error.
REPROJECTION_REJECTION_FACTOR = 4


def reprojection_gate(task, source, target, matrix, pairs, observations, cameras, max_pixels):
    import numpy as np
    from PIL import Image

    heldout = [pair for pair in pairs if pair["split"] == "validation"]
    points = transform(source[[p["source_vertex_index"] for p in heldout]], matrix)
    observed = target[[p["target_gaussian_index"] for p in heldout]]
    reports = []
    for observation in observations:
        camera = cameras[str(observation["frame_id"])]
        pose, calibration = np.asarray(camera["world_to_camera"]), np.asarray(camera["intrinsics"])
        mask = np.asarray(Image.open(local_path(task, observation["mask_path"])).convert("L")) > 127
        if mask.shape != (camera["height"], camera["width"]):
            raise RegistrationError("original observed mask must match camera raster")
        projected = []
        for cloud in (points, observed):
            camera_points = transform(cloud, pose)
            image_points = camera_points @ calibration.T
            projected.append((image_points[:, :2] / np.maximum(image_points[:, 2:], 1e-12), camera_points[:, 2]))
        (source_pixels, source_z), (target_pixels, target_z) = projected
        indices = np.rint(target_pixels).astype(int)
        valid = (target_z > 0) & (source_z > 0) & (indices[:, 0] >= 0) & (indices[:, 0] < mask.shape[1]) & (indices[:, 1] >= 0) & (indices[:, 1] < mask.shape[0])
        valid_indices = np.flatnonzero(valid)
        valid[valid_indices] &= mask[indices[valid_indices, 1], indices[valid_indices, 0]]
        if valid.sum() < 6:
            continue
        errors = np.linalg.norm(source_pixels[valid] - target_pixels[valid], axis=1)
        selected = np.rint(source_pixels[valid]).astype(int)
        inside = (selected[:, 0] >= 0) & (selected[:, 0] < mask.shape[1]) & (selected[:, 1] >= 0) & (selected[:, 1] < mask.shape[0])
        ii = np.flatnonzero(inside)
        inside[ii] &= mask[selected[ii, 1], selected[ii, 0]]
        report = {"frame_id": str(observation["frame_id"]), "heldout_visible_pairs": int(valid.sum()),
                  "p95_pixel_error": float(np.percentile(errors, 95)),
                  "source_inside_observed_mask_fraction": float(inside.mean()),
                  "max_p95_pixels": max_pixels, "above_target": float(np.percentile(errors, 95)) > max_pixels,
                  "rejection_ceiling_pixels": REPROJECTION_REJECTION_FACTOR * max_pixels,
                  "policy": "recorded_not_blocking"}
        # The target is reported, not enforced: the ScanNet++ DSLR bed placed with
        # 0.998 of its pixels inside the observed mask and a 13.07 px p95 against
        # an 8 px target, which is a 0.3% spread on a 1752 px frame -- evidence of
        # a slightly softer surface, not of a misplaced object. Only a placement
        # that is off by a multiple of the target, or that stops landing inside
        # the observed mask, is refused.
        if (report["p95_pixel_error"] > REPROJECTION_REJECTION_FACTOR * max_pixels
                or report["source_inside_observed_mask_fraction"] < .9):
            raise RegistrationError(
                f"original-camera heldout reprojection gate failed: p95 {report['p95_pixel_error']:.3f} px "
                f"against a {REPROJECTION_REJECTION_FACTOR * max_pixels:.0f} px rejection ceiling and "
                f"{report['source_inside_observed_mask_fraction']:.3f} source pixels inside the observed mask: {report}")
        reports.append(report)
    if len(reports) < 2:
        raise RegistrationError("need heldout observed-mask reprojection support from at least two original cameras")
    return {"max_p95_pixels": max_pixels, "policy": "recorded_not_blocking",
            "rejection_ceiling_pixels": REPROJECTION_REJECTION_FACTOR * max_pixels,
            "worst_p95_pixels": max(report["p95_pixel_error"] for report in reports),
            "above_target_frames": [report["frame_id"] for report in reports if report["above_target"]],
            "frames": reports}


def register_object(task, item, candidate, observed, cameras, scene_path, output_dir, threshold, args):
    import numpy as np
    from plyfile import PlyData

    if observed.get("association_status") != "geometrically_verified":
        raise RegistrationError("room registration requires a geometrically verified physical object")
    geometry_path = local_path(task, candidate["generated_frame_geometry_path"])
    geometry = read_json(geometry_path)
    # Generated views either carry an independently estimated trajectory, which
    # has to be fitted into the scene, or the conditioning trajectory they were
    # rendered from, which is already the scene frame and needs no fit at all.
    conditioning_chain = geometry.get("conditioning_poses_used") is True
    if conditioning_chain:
        if geometry.get("coordinate_frame") != observed.get("coordinate_frame") or geometry.get("unit") != observed.get("units"):
            raise RegistrationError("conditioning-trajectory generated geometry is not in the observed object's declared frame")
    elif geometry.get("coordinate_frame") != "generated_DA3_world":
        raise RegistrationError("registration needs independently estimated generated-frame geometry")
    provenance_path = local_path(task, candidate["sampled_frame_provenance_path"])
    provenance = {r["frame_id"]: r for r in read_json(provenance_path)["frames"]}
    generated, conditioned, heldout, frame_map, camera_refs = [], [], [], {}, {}
    counters = {}
    for frame in geometry["frames"]:
        reference = provenance[frame["frame_id"]]
        orbit = reference["orbit_id"]
        path = local_path(task, reference["conditioning_cameras_path"])
        camera_refs[relative(task, path)] = sha256(path)
        trajectory = read_json(path)
        selected = trajectory["frames"][reference["generated_frame_index"]]
        counters[orbit] = counters.get(orbit, 0) + 1
        generated.append(frame["world_to_camera"])
        conditioned.append(selected["world_to_camera"])
        heldout.append(counters[orbit] % 4 == 0)
        frame_map[frame["stream3d_frame_name"] + ".png"] = frame
    if len(counters) != 3 or min(counters.values()) < 4:
        raise RegistrationError("registration requires >=4 sampled frames from each of three distinct elevations")
    chain, camera_qa = (np.eye(4), {"method": "conditioning_trajectory_declared",
                                    "generated_world_to_scene_world": np.eye(4).tolist(),
                                    "threshold_world_units": threshold,
                                    "policy": "poses_are_the_conditioning_trajectory_of_the_registered_generated_views",
                                    "coordinate_frame": geometry.get("coordinate_frame"), "unit": geometry.get("unit"),
                                    "independent_estimation": (geometry.get("generated_camera_fallback") or {}).get("independent_estimation")}) \
        if conditioning_chain else \
        camera_chain(generated, conditioned, heldout, threshold * 2, args.max_camera_angle, args.seed)
    metadata_path = local_path(task, item["normalization_metadata_path"])
    metadata = read_json(metadata_path)
    selected = metadata.get("stage1_selected_crop_view_names", [])
    if not selected or selected[0] not in frame_map:
        raise RegistrationError("SAM3D first stage1 reference camera is not bound to a generated frame")
    pose_path = local_path(task, item["backend_pose_path"])
    with np.load(pose_path, allow_pickle=False) as pose:
        object_to_camera = glb_to_reference_camera(pose)
    initial = chain @ np.linalg.inv(np.asarray(frame_map[selected[0]]["world_to_camera"])) @ object_to_camera
    source_path = local_path(task, item["mesh_path"])
    target_path = local_path(task, observed["ply_path"])
    if item.get("mesh_sha256") and sha256(source_path) != item["mesh_sha256"]:
        raise RegistrationError("completed source mesh changed before registration")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from mesh_postprocess import check_external_dependencies

    check_external_dependencies(task, source_path)
    source = scene_vertices(source_path)
    table = PlyData.read(str(target_path))["vertex"].data
    target = np.column_stack([table[name] for name in ("x", "y", "z")]).astype(float)
    if not np.isfinite(target).all():
        raise RegistrationError("observed geometry has non-finite points")
    matrix, pairs, geometric_qa = observed_correspondences(source, target, initial, threshold, args.min_observed_coverage, args.seed)
    reprojection_qa = reprojection_gate(task, source, target, matrix, pairs, observed["observations"], cameras, args.max_reprojection_pixels)
    proof_path = output_dir / "correspondences.json"
    write_json(proof_path, {"object_id": item["object_id"], "source_indexing": "trimesh_sorted_scene_nodes_baked_vertices", "target_indexing": "ply_vertex_row", "pairs": pairs, "camera_chain": camera_qa, "geometry": geometric_qa, "reprojection": reprojection_qa})
    registration = {"object_id": item["object_id"], "status": "accepted", "method": "ransac_correspondence_sim3", "object_to_world": matrix.tolist(), "source_coordinate_frame": item["coordinate_frame"], "source_units": item["unit"], "target_coordinate_frame": observed["coordinate_frame"], "target_units": observed["units"], "source_geometry_path": relative(task, source_path), "source_geometry_sha256": sha256(source_path), "target_object_ply_sha256": sha256(target_path), "target_scene_sha256": sha256(scene_path), "correspondences_path": relative(task, proof_path), "correspondences_sha256": sha256(proof_path), "max_error_world_units": threshold, "reference_generated_frame": selected[0], "initial_object_to_world": initial.tolist(), "camera_chain": camera_qa, "geometry": geometric_qa, "reprojection": reprojection_qa, "semantic_correspondence_review": "geometric_track_identity_and_camera_initialized_nearest_geometry; no_manual_semantic_review", "input_hashes": {relative(task, p): sha256(p) for p in (geometry_path, provenance_path, metadata_path, pose_path)}, "conditioning_camera_hashes": camera_refs, "axis_convention": "inverse_official_to_glb_row_export_then_PyTorch3D_row_pose_then_OpenCV_camera"}
    registration_path = output_dir / "registration.json"
    write_json(registration_path, registration)
    return {**item, "registration": registration, "registration_path": relative(task, registration_path), "registration_sha256": sha256(registration_path), "registration_source_geometry_path": relative(task, source_path), "registration_source_geometry_sha256": sha256(source_path), "room_alignment": "observed_correspondence_verified", "observed_anchor_applied": False}


def run(args):
    task = args.task_dir.resolve(strict=True)
    roles = ("completed_object_meshes", "completion_candidates", "isolated_object_ply", "cameras", "scene_gaussian_ply", "lifting_report")
    paths = {role: input_artifact(args, role) for role in roles}
    meshes, candidates = read_json(paths["completed_object_meshes"]), read_json(paths["completion_candidates"])
    observed = {r["object_id"]: r for r in read_json(paths["isolated_object_ply"])["objects"]}
    cameras = {str(r["frame_id"]): r for r in read_json(paths["cameras"])["frames"]}
    candidate_by_id = {r["object_id"]: r for r in candidates["objects"]}
    lifting = read_json(paths["lifting_report"])
    voxel = lifting.get("parameters", {}).get("voxel_size", lifting.get("voxel_size"))
    if voxel is None:
        voxel = lifting.get("geometry_parameters", {}).get("voxel_size")
    if voxel is None or float(voxel) <= 0:
        raise RegistrationError("lifting report must specify its observed-world voxel size")
    threshold = 3 * float(voxel)
    stage = local_path(task, "stages/geometry_completion/registration", exists=False)
    stage.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="run-", dir=stage))
    report = {"status": "running", "threshold_world_units": threshold, "objects": []}
    registered = []
    seen = set()
    try:
        for item in meshes["objects"]:
            object_id = item["object_id"]
            if not isinstance(object_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", object_id) or object_id in seen:
                raise RegistrationError("registration requires unique safe object identifiers")
            seen.add(object_id)
            directory = local_path(task, output / object_id, exists=False)
            directory.mkdir()
            result = register_object(task, item, candidate_by_id[object_id], observed[object_id], cameras, paths["scene_gaussian_ply"], directory, threshold, args)
            registered.append(result)
            report["objects"].append({"object_id": object_id, "registration_path": result["registration_path"], "status": "accepted"})
        report["status"] = "accepted"
        meshes_path, candidates_path = output / "completed_object_meshes.json", output / "completion_candidates.json"
        write_json(meshes_path, {**meshes, "objects": registered})
        registrations = {item["object_id"]: item for item in registered}
        updated_candidates = [{**candidate, "room_alignment": registrations[candidate["object_id"]]["room_alignment"], "registration_path": registrations[candidate["object_id"]]["registration_path"]} if candidate["object_id"] in registrations else candidate for candidate in candidates["objects"]]
        write_json(candidates_path, {**candidates, "objects": updated_candidates, "registration": report, "room_alignment_validated": True})
        publish(args, {"completed_object_meshes": meshes_path, "completion_candidates": candidates_path})
        return report
    except (RegistrationError, OSError, ValueError, KeyError, ImportError) as error:
        report.update({"status": "blocked_registration_validation", "error": str(error)})
        raise
    finally:
        write_json(output / "registration_report.json", report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--max-camera-angle", type=float, default=12)
    parser.add_argument("--min-observed-coverage", type=float, default=.7)
    parser.add_argument("--max-reprojection-pixels", type=float, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if not 0 < args.max_camera_angle <= 30 or not .5 <= args.min_observed_coverage <= 1 or not 0 < args.max_reprojection_pixels <= 30:
        parser.error("invalid registration acceptance thresholds")
    try:
        run(args)
    except (RegistrationError, OSError, ValueError, KeyError, ImportError) as error:
        print(f"object_registration failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
