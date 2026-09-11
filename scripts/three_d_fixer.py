#!/usr/bin/env python3
"""Opt-in learned 3D-Fixer provider; no downloads or automatic pipeline registration.

Official API audited at HorizonRobotics/3D-Fixer f6a60328b4646389b4dd4b2ecea9df31bab9a9bd.
Use --prepare-only to audit observed inputs without importing model code. This
never publishes roles. Normal execution requires a fully local deployment.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import sys
import tempfile

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from world_modeling.provider_io import input_artifact, local_path, publish, read_json, write_json
from world_modeling.object_lineage import lineage_metadata, validate_object_lineages
from three_d_fixer_module import SPEC

UPSTREAM_COMMIT = "f6a60328b4646389b4dd4b2ecea9df31bab9a9bd"
SOURCE_HASHES = {
    "threeDFixer/pipelines/threeD_fixer.py": "0245150c1e27c7961f10bd6ca54289636ea7e57e5791601e275f262a683953de",
    "threeDFixer/models/__init__.py": "0aa3708bce86fdf86602f9c7b08f7a59fe5c1ce23dcdb982d612365bf1660dca",
    "threeDFixer/datasets/utils.py": "df26a6c07338d135d0995851ad43dc7a1a5cc5a88b7fe4c8e5c22f8686a5ae5e",
    "threeDFixer/utils/postprocessing_utils.py": "4989ce4c18c457bd142ae965277c5fcee060fdfa236325358e32dfe4b538a80c",
    "threeDFixer/moge/model/v2.py": "e2108b2cd2cc4f4f1a65ab6657655dd2c90a60c32646808fd4430cfe385408bc",
    "threeDFixer/representations/gaussian/gaussian_model.py": "d04766b65ad4045190c3db0a956b583a9911563dc43cc57d2906a43d0fb00ab7",
}
BASE_MODELS = {
    "sparse_structure_encoder": "ss_enc_conv3d_16l8_fp16",
    "sparse_structure_decoder": "ss_dec_conv3d_16l8_fp16",
    "slat_decoder_rf": "slat_dec_rf_swin8_B_64l8r16_fp16",
}
FIXER_MODELS = {
    "sparse_structure_flow_model": "ss_flow_img_dit_L_16l8_fp16",
    "scene_sparse_structure_flow_coarse_model": "scene_ss_coarse_flow_img_dit_L_16l8_fp16",
    "scene_sparse_structure_flow_fine_model": "scene_ss_fine_flow_img_dit_L_16l8_fp16",
    "slat_decoder_gs": "slat_dec_gs_swin8_B_64l8gs32_fp16",
    "slat_decoder_mesh": "slat_dec_mesh_swin8_B_64l8m256c_fp16",
    "slat_flow_model": "slat_flow_img_dit_L_64l8p2_fp16",
    "scene_slat_flow_model": "scene_slat_flow_img_dit_L_64l8p2_fp16",
}
MODEL_REVISION = "8598020130d65bddf2b8e9f5538c522e0548d0ec"
# Official Hugging Face LFS SHA256 and Git-blob IDs at MODEL_REVISION. Only
# safetensors are needed; .pt duplicates must not be downloaded as well.
FIXER_FILES = {
    "scene_slat_flow_model": (932580016, "b5f27e35390ed65414a62448e395e26c6ca0ebae42bcc1434ecbaa8ddf24024b", "b7680d3041fb450ae00d3c62d6787ba7c93b0566"),
    "scene_sparse_structure_flow_coarse_model": (862395296, "6d0401e46b0e4626ca208ea553c5bef67c90dff6e96f4f83a5b58124c7e1e43b", "ec1ab8880bbe1a1bc434df2a4b6dc3cf871a2e94"),
    "scene_sparse_structure_flow_fine_model": (862395296, "7ad6267d5f850a9f82af1298af7ced1aff874c7dc5df845882af079c4dd3413c", "ec1ab8880bbe1a1bc434df2a4b6dc3cf871a2e94"),
    "slat_decoder_gs": (171450952, "397c451f1dece04c262d794f504ae7d6146c7b2d67a5b46cd65c4e0779025da0", "051ade1e3b374738bf7a007516275f72e458f2a7"),
    "slat_decoder_mesh": (181903412, "f0b50b08edee0eee54e455cb7081ddef7553c53cfc555d941c306cc087895ff5", "28802825e9ba04bcc2cfdffb491dc9dbb72c8a38"),
    "slat_flow_model": (1203755136, "0d902601621864ca7797230b1266ab70ffa62ce3b98c62f2294843a647d09068", "22f500ab89867eb147f04d313cffac98e54ca163"),
    "sparse_structure_flow_model": (1130770840, "ac4686d81327bfb76ed24e25d05794ae938cb8c787d25e95b63a250d75ddf2ae", "d228f99cd828a54695ebd1f0fd1ad1b1f3179f42"),
}


# The selected view's DA3-vs-isolated-object agreement is measured and reported;
# SUPPORT_TARGET is the value that used to be fatal and SUPPORT_FLOOR is the
# point at which the two surfaces are genuinely different objects.
SUPPORT_TARGET = 0.9
SUPPORT_FLOOR = 0.5


class FixerError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise FixerError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(task, path):
    return local_path(task, path).relative_to(task.resolve()).as_posix()


def bound_path(task, record, path_key="path", hash_key="sha256"):
    path = local_path(task, record[path_key])
    require(record.get(hash_key) == sha256(path), f"hash binding mismatch: {path}")
    return path


def indexed(records, key):
    require(isinstance(records, list) and records, f"nonempty {key} records required")
    result = {}
    for record in records:
        value = record[key]
        require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", value)
                and value not in result, f"invalid or duplicate {key}")
        result[value] = record
    return result


def projection_inputs(camera):
    import numpy as np
    from object_lifting import intrinsics, world_to_camera

    pose = world_to_camera(camera["world_to_camera"])
    calibration = intrinsics(camera["intrinsics"])
    width, height = camera["width"], camera["height"]
    require(isinstance(width, int) and isinstance(height, int) and min(width, height) > 0,
            "camera raster dimensions must be positive integers")
    # utils3d.project_cv uses normalized image coordinates; run.project_uv
    # inverts its misleadingly named extrinsics parameter exactly once.
    return np.linalg.inv(pose), np.diag([1 / width, 1 / height, 1]) @ calibration


def canonical_to_world(coarse_translation, coarse_scale, fine_translation, fine_scale):
    import numpy as np

    coarse = np.asarray(coarse_translation, float)
    fine = np.asarray(fine_translation, float)
    require(coarse.shape == fine.shape == (3,) and np.isfinite(coarse).all() and np.isfinite(fine).all(),
            "3D-Fixer normalization translations must be finite 3-vectors")
    require(np.isfinite([coarse_scale, fine_scale]).all() and min(coarse_scale, fine_scale) > 0,
            "3D-Fixer normalization scales must be finite and positive")
    matrix = np.eye(4)
    matrix[:3, :3] *= coarse_scale * fine_scale
    matrix[:3, 3] = fine * coarse_scale + coarse
    return matrix


def verify_dedicated_checkpoint(key, prefix):
    size, digest, config_blob = FIXER_FILES[key]
    weights, config = Path(str(prefix) + ".safetensors"), Path(str(prefix) + ".json")
    require(weights.stat().st_size == size and sha256(weights) == digest,
            f"dedicated checkpoint differs from official {MODEL_REVISION}: {key}")
    data = config.read_bytes()
    blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    require(blob == config_blob, f"dedicated model architecture differs from official checkpoint: {key}")


def runtime_preflight(args):
    """Resolve only known official checkpoint roles; never accept base fallbacks."""
    require(args.source is not None and args.model_dir is not None, "3D-Fixer source and dedicated model directory required")
    source, model_dir = args.source.resolve(), args.model_dir.resolve()
    for name, digest in SOURCE_HASHES.items():
        path = source / name
        require(path.is_file() and sha256(path) == digest, f"missing or unaudited official source: {name}")
    configuration = read_json(model_dir / "pipeline.json")
    require(configuration.get("name") == "ThreeDFixerPipeline", "dedicated ThreeDFixerPipeline configuration required; TRELLIS is not a substitute")
    config = configuration["args"]
    require(set(config["models"]) == set(BASE_MODELS) | set(FIXER_MODELS), "official 3D-Fixer model role set differs")
    require(config["image_cond_model"] == "dinov2_vitl14_reg", "unsupported DINO conditioning architecture")
    require(args.trellis_base is not None and args.trellis_base.resolve() != model_dir,
            "dedicated 3D-Fixer and reusable TRELLIS base must be distinct directories")
    paths, resources = {}, {}
    for key, name in {**BASE_MODELS, **FIXER_MODELS}.items():
        expected = f"microsoft/TRELLIS-image-large/ckpts/{name}" if key in BASE_MODELS else f"ckpts/{name}"
        require(config["models"][key] == expected, f"unexpected checkpoint mapping for {key}")
        prefix = (args.trellis_base if key in BASE_MODELS else model_dir) / "ckpts" / name
        for suffix in (".json", ".safetensors"):
            path = Path(str(prefix) + suffix)
            require(path.is_file() and path.stat().st_size > 0, f"missing {key} resource: {path}")
        if key in FIXER_MODELS:
            require(model_dir in Path(str(prefix) + ".safetensors").resolve().parents,
                    f"dedicated 3D-Fixer checkpoint cannot alias a TRELLIS fallback: {key}")
            verify_dedicated_checkpoint(key, prefix)
        paths[key] = str(prefix.resolve())
        resources[key] = {suffix: {"path": str(Path(str(prefix) + suffix).resolve()),
                                   "sha256": FIXER_FILES[key][1] if key in FIXER_FILES and suffix == ".safetensors" else sha256(Path(str(prefix) + suffix))}
                          for suffix in (".json", ".safetensors")}
    for value, label in ((args.moge_checkpoint, "MoGe v2 checkpoint"), (args.dino_checkpoint, "DINO register checkpoint")):
        require(value is not None and value.is_file() and value.stat().st_size > 0, f"missing local {label}")
    require(args.dino_source is not None and (args.dino_source / "hubconf.py").is_file(), "local DINO source required; torch.hub remote loading forbidden")
    return {"source": str(source), "source_commit": UPSTREAM_COMMIT, "source_hashes": SOURCE_HASHES,
            "pipeline_config_sha256": sha256(model_dir / "pipeline.json"), "model_revision": MODEL_REVISION,
            "model_prefixes": paths, "model_resources": resources,
            "moge_checkpoint": str(args.moge_checkpoint.resolve()), "dino_checkpoint": str(args.dino_checkpoint.resolve()),
            "moge_checkpoint_sha256": sha256(args.moge_checkpoint), "dino_checkpoint_sha256": sha256(args.dino_checkpoint),
            "dino_source": str(args.dino_source.resolve()), "configuration": config}


def load_pipeline(runtime):
    """Reproduce official initialization with explicit local-only encoders."""
    import torch
    from torchvision import transforms

    sys.path.insert(0, runtime["source"])
    from threeDFixer import models
    from threeDFixer.pipelines import ThreeDFixerPipeline, samplers
    from threeDFixer.moge.model.v2 import MoGeModel

    require(torch.cuda.is_available(), "learned 3D-Fixer requires a CUDA deployment")
    pipeline = ThreeDFixerPipeline()
    pipeline.models = {key: models.from_pretrained(path).eval() for key, path in runtime["model_prefixes"].items()}
    config = runtime["configuration"]
    for key in ("sparse_structure_sampler", "coarse_sparse_structure_sampler", "slat_sampler"):
        item = config[key]
        setattr(pipeline, key, getattr(samplers, item["name"])(**item["args"]))
        setattr(pipeline, key + "_params", item["params"])
    pipeline.slat_normalization = config["slat_normalization"]
    dino = torch.hub.load(runtime["dino_source"], config["image_cond_model"], source="local", pretrained=False)
    dino.load_state_dict(torch.load(runtime["dino_checkpoint"], map_location="cpu", weights_only=True), strict=True)
    moge_state = torch.load(runtime["moge_checkpoint"], map_location="cpu", weights_only=True)
    moge = MoGeModel(**moge_state["model_config"])
    moge.load_state_dict(moge_state["model"], strict=True)
    pipeline.models.update(image_cond_model=dino.eval(), scene_cond_model=moge.eval())
    pipeline.image_cond_model_transform = transforms.Compose([transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])])
    pipeline.cuda()
    return pipeline


def validate_documents(task, paths):
    documents = {role: read_json(path) for role, path in paths.items() if role != "scene_gaussian_ply"}
    assembly, depth, lifting = (documents[name] for name in ("assembly_report", "scene_depth", "lifting_report"))
    require(assembly.get("status") == "assembled_observed_views" and
            assembly.get("method") == "same_camera_verified_parent_mask_rgba_extraction" and
            all(assembly.get(key) is False for key in ("cross_view_pixel_pasting", "hidden_pixels_generated", "complete_object_claimed")),
            "3D-Fixer requires original observed context, not generated or pasted views")
    require(depth.get("coordinate_frame") == "colmap_world" and depth.get("units") == "colmap_reconstruction" and
            depth.get("camera_convention") == "world_to_camera_opencv" and depth.get("depth_kind") == "camera_z",
            "calibrated COLMAP-world camera_z depth required; metric scale must not be guessed")
    require(lifting.get("generated_geometry_used") is False and lifting.get("carve_performed") is True,
            "geometrically verified observed lifting is required")
    require(lifting.get("coordinate_frame") == depth["coordinate_frame"] and lifting.get("units") == depth["units"], "lifting/depth coordinate units disagree")
    for role in ("component_masks", "isolated_object_ply", "cameras", "lifting_report"):
        require(bound_path(task, assembly["source_artifacts"][role]) == paths[role], f"assembly {role} binding disagrees")
    for role in ("cameras", "scene_depth", "scene_gaussian_ply"):
        require(bound_path(task, lifting["source_files"][role]) == paths[role], f"lifting {role} binding disagrees")
    objects = indexed(documents["isolated_object_ply"]["objects"], "object_id")
    validate_object_lineages(task, objects.values())
    assembled = indexed(documents["assembled_object_views"]["objects"], "object_id")
    accepted = {track["object_id"] for track in lifting["tracks"] if track.get("accepted") is True}
    require(set(objects) == set(assembled) == accepted, "assembly/observed/lifting object identities disagree")
    require(bound_path(task, lifting["resolved_files"]["component_masks"]) == paths["component_masks"],
            "resolved parent masks are not bound to the lifting receipt")
    require(local_path(task, documents["component_masks"]["acceptance_authority"]) == paths["lifting_report"],
            "resolved mask acceptance authority differs from lifting")
    documents["verified_mask_bindings"] = observation_bindings(task, objects, documents["component_masks"])
    return documents, objects, assembled


def observation_bindings(task, objects, masks):
    parents = {}
    for record in masks["masks"]:
        if record.get("component_id") == "__object__":
            key = (record["object_id"], record["frame_id"])
            require(key not in parents, "duplicate verified parent mask")
            parents[key] = record
    bindings = {}
    for object_id, obj in objects.items():
        bindings[object_id] = []
        for observation in obj["observations"]:
            parent = parents[(object_id, observation["frame_id"])]
            path = bound_path(task, parent, "mask_path", "mask_sha256")
            require(path == local_path(task, observation["mask_path"]), "heldout observation mask differs from resolved parent")
            bindings[object_id].append({"frame_id": observation["frame_id"], "path": relative(task, path), "sha256": parent["mask_sha256"]})
    return bindings


def selected_view_support(distances, threshold):
    """Measure how much of the lifted object surface lies on the isolated object.

    The measurement is evidence, not a cliff. The ScanNet++ DSLR bed came in at
    0.8971 against a 0.9 target, which no geometric argument can tell apart from
    a passing 0.90, so the shortfall is recorded against ``SUPPORT_TARGET`` and
    only a surface that mostly does not exist at the lifted location -- below
    ``SUPPORT_FLOOR`` -- is refused.
    """
    import numpy as np

    fraction = float(np.mean(np.asarray(distances) <= threshold))
    support = {"policy": "recorded_not_blocking", "support_fraction": fraction,
               "target_fraction": SUPPORT_TARGET, "floor_fraction": SUPPORT_FLOOR,
               "below_target": fraction < SUPPORT_TARGET, "registration_threshold": threshold}
    require(fraction >= SUPPORT_FLOOR,
            f"selected view DA3/isolated-object support {fraction:.6f} is below the {SUPPORT_FLOOR} floor")
    return fraction, support


def prepare_object(task, obj, assembled, cameras, depths, directory, args):
    import numpy as np
    from PIL import Image
    from plyfile import PlyData
    from scipy.spatial import cKDTree
    from object_lifting import load_frame, load_mask, backproject

    require(obj.get("association_status") == "geometrically_verified" and obj.get("coordinate_frame") == "colmap_world"
            and obj.get("units") == "colmap_reconstruction", "unverified physical object or unknown coordinate units")
    require(assembled.get("observed_geometry") == obj and assembled.get("geometry_extent") == "observed_only_incomplete", "assembly observed geometry is not bound to the current object")
    views = indexed(assembled["views"], "frame_id")
    # Fixed before inference, based only on observed mask support, never on
    # downstream completion quality. A failed selected view blocks the object.
    view = sorted(views.values(), key=lambda v: (-v["foreground_pixels"], v["frame_id"]))[0]
    frame_id = view["frame_id"]
    require(view.get("coordinate_frame") == "original_image" and view.get("evidence") == "observed_pixels"
            and view.get("alpha_source") == "verified_parent_mask", "source must be original observed RGB and verified parent mask")
    image_path = bound_path(task, view, "image_path", "image_sha256")
    mask_path = bound_path(task, view, "parent_mask_path", "parent_mask_sha256")
    observations = indexed(obj["observations"], "frame_id")
    require(len(observations) >= 3, "completion needs at least two original heldout cameras in addition to its selected view")
    for observation_frame in observations:
        require(observation_frame in cameras, "observed registration camera is missing")
        projection_inputs(cameras[observation_frame])
    require(local_path(task, observations[frame_id]["mask_path"]) == mask_path, "selected parent mask differs from lifting")
    camera, depth = cameras[frame_id], depths[frame_id]
    require(local_path(task, camera["image_path"]) == image_path, "source image is not the selected calibrated camera image")
    for key in ("depth", "confidence"):
        bound_path(task, depth, key + "_path", key + "_sha256")
    image = Image.open(image_path).convert("RGB")
    mask = np.asarray(Image.open(mask_path).convert("L"))
    require(image.size == (camera["width"], camera["height"]) and mask.shape == (image.height, image.width)
            and set(np.unique(mask)) <= {0, 255} and np.count_nonzero(mask) == view["foreground_pixels"], "original RGB/mask raster or mask-area mismatch")
    frame = load_frame(task, frame_id, depth, camera, args)
    depth_mask = load_mask(task, mask_path, frame.depth.shape, frame)
    points = backproject(frame, depth_mask)
    require(len(points) >= args.min_points, "insufficient finite confident object depth")
    target_path = bound_path(task, obj, "ply_path", "sha256")
    rows = PlyData.read(str(target_path))["vertex"].data
    target = np.column_stack([rows[key] for key in ("x", "y", "z")]).astype(float)
    require(len(target) >= args.min_points and np.isfinite(target).all(), "invalid observed Gaussian centers")
    distances = cKDTree(target).query(points)[0]
    fraction, support = selected_view_support(distances, args.registration_threshold)
    c2w, calibration = projection_inputs(camera)
    output = directory / "partial_world_points.npz"
    np.savez_compressed(output, points=points.astype(np.float32), points_mask=np.ones(len(points), np.float32))
    report = {**lineage_metadata(obj), "object_id": obj["object_id"], "selected_frame_id": frame_id,
              "selection_rule": "maximum_verified_parent_foreground_pixels_then_frame_id",
              "image_path": relative(task, image_path), "image_sha256": sha256(image_path),
              "mask_path": relative(task, mask_path), "mask_sha256": sha256(mask_path),
              "points_path": relative(task, output), "points_sha256": sha256(output), "point_count": len(points),
              "selected_view_support": support,
              "depth_path": depth["depth_path"], "depth_sha256": depth["depth_sha256"],
              "confidence_path": depth["confidence_path"], "confidence_sha256": depth["confidence_sha256"],
              "confidence_min": args.confidence_min, "image_to_depth": frame.image_to_depth.tolist(),
              "depth_intrinsics": frame.intrinsics.tolist(), "source_intrinsics": camera["intrinsics"],
              "world_to_camera": frame.world_to_camera.tolist(), "native_extrinsics_camera_to_world": c2w.tolist(),
              "native_intrinsics_normalized": calibration.tolist(), "coordinate_frame": obj["coordinate_frame"], "units": obj["units"],
              "point_axes": "colmap_world_unchanged", "pixel_center_convention": "integer_pixel_coordinates_as_DA3_lifting",
              "observed_support_fraction": fraction, "observed_support_threshold_world_units": args.registration_threshold,
              "target_object_ply_sha256": sha256(target_path), "target_object_ply_path": relative(task, target_path),
              "mask_morphology_applied": False,
              "observation_mask_bindings": args.observation_bindings[obj["object_id"]],
              "generated_orbit_used": False, "MoGe_geometry_inference_used": False,
              "distribution_note": "DA3 geometry replaces official demo MoGe geometry; MoGe scene feature encoder remains required; not yet inference-validated"}
    write_json(directory / "observed_context.json", report)
    return report


def infer_object(task, context, pipeline, directory, args, receipt=None):
    import numpy as np
    import torch
    from PIL import Image
    from threeDFixer.datasets.utils import process_instance_image_only, process_scene_image
    from threeDFixer.utils import postprocessing_utils

    image = Image.open(bound_path(task, context, "image_path", "image_sha256")).convert("RGBA")
    mask = np.asarray(Image.open(bound_path(task, context, "mask_path", "mask_sha256")).convert("L")) > 0
    with np.load(bound_path(task, context, "points_path", "points_sha256"), allow_pickle=False) as pack:
        points, points_mask = pack["points"], pack["points_mask"]
    rgb, alpha = process_instance_image_only(image, mask, 518)
    _, scene = process_scene_image(image, mask, 518)
    outputs, coarse_t, coarse_s, fine_t, fine_s = pipeline.run(
        instance_image_masked=torch.cat([rgb, alpha]).cuda(), scene_image_masked=scene.cuda(),
        extrinsics=torch.tensor(context["native_extrinsics_camera_to_world"], dtype=torch.float32, device="cuda"),
        intrinsics=torch.tensor(context["native_intrinsics_normalized"], dtype=torch.float32, device="cuda"),
        points=points, points_mask=points_mask, seed=args.seed, formats=["mesh", "gaussian"],
    )
    if receipt is not None:
        receipt["model_inference_performed"] = True
    matrix = canonical_to_world(coarse_t, float(coarse_s), fine_t, float(fine_s))
    native = outputs["mesh"][0]
    require(torch.isfinite(native.vertices).all().item() and len(native.faces) > 0, "model produced invalid or empty native mesh")
    # Identity is intentional: supplying transform_fn suppresses the official
    # default Z-up -> Y-up display rotation. Scene transform stays in metadata.
    mesh = postprocessing_utils.to_glb(outputs["gaussian"][0], native, simplify=0, fill_holes=False,
                                      texture_size=args.texture_size, transform_fn=lambda vertices: vertices)
    require(len(mesh.vertices) > 0 and len(mesh.faces) > 0 and np.isfinite(mesh.vertices).all(), "invalid exported mesh")
    refinement = None
    if getattr(args, "observed_refinement", False):
        from plyfile import PlyData

        observed_path = bound_path(task, context, "target_object_ply_path", "target_object_ply_sha256")
        rows = PlyData.read(str(observed_path))["vertex"].data
        observed = np.column_stack([rows[key] for key in ("x", "y", "z")]).astype(float)
        strength = float(getattr(args, "refinement_strength", 0.5))
        smoothing = float(getattr(args, "refinement_smoothing", 0.3))
        iterations = int(getattr(args, "refinement_iterations", 8))
        configured = float(getattr(args, "refinement_radius", 0.0))
        radius = configured if configured > 0 else 2.5 * float(args.registration_threshold)
        refined, supported = refine_observed_regions(
            mesh.vertices, mesh.faces, observed, matrix,
            radius=radius, strength=strength, smoothing=smoothing, iterations=iterations)
        mesh.vertices = refined
        refinement = {"method": "observed_point_laplacian_data_fit", "radius_world_units": radius,
                      "strength": strength, "smoothing": smoothing, "iterations": iterations,
                      "observed_points": int(len(observed)), "max_supported_vertices": supported,
                      "moves_only_toward_observed_points": True,
                      "unobserved_regions": "laplacian_coupling_only"}
    mesh_path = directory / "canonical.glb"
    mesh.export(mesh_path)
    gaussian_path = directory / "canonical_gaussian.ply"
    saturated_logits = save_canonical_gaussians(outputs["gaussian"][0], gaussian_path)
    normalization = {"coarse_translation": np.asarray(coarse_t).tolist(), "coarse_scale": float(coarse_s),
                     "fine_translation": np.asarray(fine_t).tolist(), "fine_scale": float(fine_s),
                     "canonical_to_world": matrix.tolist(), "formula": "world=(canonical*fine_scale+fine_translation)*coarse_scale+coarse_translation",
                     "default_glb_display_rotation_applied": False, "target_units": context["units"],
                     "target_coordinate_frame": context["coordinate_frame"], "source_context_path": relative(task, directory / "observed_context.json")}
    normalization_path = directory / "normalization.json"
    write_json(normalization_path, normalization)
    return {**lineage_metadata(context), "object_id": context["object_id"], "mesh_path": relative(task, mesh_path), "mesh_sha256": sha256(mesh_path),
            "visual_ply_path": relative(task, gaussian_path), "visual_ply_sha256": sha256(gaussian_path),
            "visual_ply_kind": "generated_3d_fixer_gaussians", "coordinate_frame": "object_local", "unit": "asset_units",
            "saturated_opacity_logits_clamped": saturated_logits,
            "saturated_opacity_logit_bound": OPACITY_LOGIT_CLAMP if saturated_logits else None,
            "observed_region_refinement": refinement,
            "normalization_metadata_path": relative(task, normalization_path), "normalization_sha256": sha256(normalization_path),
            "backend": "learned_three_d_fixer", "evidence": "generated", "room_alignment": "not_validated",
            "initial_object_to_world": matrix.tolist(), "observed_anchor_applied": False}


# The official exporter writes inverse_sigmoid(opacity) as the PLY opacity column.
# When an activated opacity saturates to exactly 0.0 or 1.0 in float32 that logit is
# -inf/+inf even though the Gaussian itself is valid (fully transparent/opaque). The
# logit is therefore clamped to a finite bound that still round-trips to opacity 0/1
# at float32 precision. NaN opacity remains a hard failure and every other attribute
# is still required to be finite.
OPACITY_LOGIT_CLAMP = 30.0


def refine_observed_regions(vertices, faces, observed, to_world, *, radius, strength, smoothing, iterations):
    """Pull mesh vertices toward nearby observed points with Laplacian smoothing.

    The completion model is conditioned on one observed view, so its raw surface
    drifts from the multi-view observation even where the object was observed.
    This opt-in step is a data-fitting post-process: it moves vertices only toward
    measured points inside ``radius`` and never invents geometry. Unobserved
    regions are left to the Laplacian coupling. Returns (vertices, supported).
    """
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.spatial import cKDTree

    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces)
    observed = np.asarray(observed, dtype=float)
    to_world = np.asarray(to_world, dtype=float)
    if len(vertices) == 0 or len(faces) == 0 or len(observed) == 0 or to_world.shape != (4, 4):
        raise FixerError("observed-region refinement requires a nonempty mesh, observed geometry and 4x4 transform")
    if not (radius > 0 and 0 < strength <= 1 and 0 <= smoothing < 1 and iterations >= 1):
        raise FixerError("invalid observed-region refinement parameters")
    if not (np.isfinite(vertices).all() and np.isfinite(observed).all() and np.isfinite(to_world).all()):
        raise FixerError("observed-region refinement inputs must be finite")
    world = vertices @ to_world[:3, :3].T + to_world[:3, 3]
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    adjacency = coo_matrix((np.ones(len(edges) * 2), (np.concatenate([edges[:, 0], edges[:, 1]]),
                                                       np.concatenate([edges[:, 1], edges[:, 0]]))),
                           shape=(len(vertices), len(vertices))).tocsr()
    degree = np.asarray(adjacency.sum(1)).ravel()
    if (degree == 0).any():
        raise FixerError("observed-region refinement requires a mesh without isolated vertices")
    tree = cKDTree(observed)
    supported = 0
    for _ in range(iterations):
        distances, indices = tree.query(world)
        mask = distances <= radius
        supported = max(supported, int(mask.sum()))
        world[mask] += strength * (observed[indices[mask]] - world[mask])
        world = (1 - smoothing) * world + smoothing * (adjacency @ world / degree[:, None])
    if not np.isfinite(world).all():
        raise FixerError("observed-region refinement produced non-finite vertices")
    inverse = np.linalg.inv(to_world)
    return world @ inverse[:3, :3].T + inverse[:3, 3], supported


def save_canonical_gaussians(gaussian, path):
    import numpy as np
    from plyfile import PlyData, PlyElement

    # Official save_ply otherwise applies a display rotation to XYZ and quats.
    gaussian.save_ply(str(path), transform=None)
    ply = PlyData.read(str(path))
    rows = ply["vertex"].data
    required = {"x", "y", "z", "opacity", *(f"f_dc_{i}" for i in range(3)),
                *(f"scale_{i}" for i in range(3)), *(f"rot_{i}" for i in range(4))}
    require(len(rows) > 0 and required <= set(rows.dtype.names), "empty or incomplete generated Gaussian PLY")
    opacity = np.asarray(rows["opacity"], dtype=float)
    require(not np.isnan(opacity).any(), "nonfinite generated Gaussian attributes")
    saturated = int((~np.isfinite(opacity)).sum())
    if saturated:
        # Rewrite from a copy: writing the buffer returned by PlyData.read crashes
        # with SIGBUS in the deployed plyfile build.
        clamped = rows.copy()
        clamped["opacity"] = np.clip(opacity, -OPACITY_LOGIT_CLAMP, OPACITY_LOGIT_CLAMP).astype(rows["opacity"].dtype)
        PlyData([PlyElement.describe(clamped, "vertex")]).write(str(path))
        rows = PlyData.read(str(path))["vertex"].data
    require(all(np.isfinite(rows[name]).all() for name in rows.dtype.names), "nonfinite generated Gaussian attributes")
    return saturated


def register_object(task, item, observed, cameras, scene_path, directory, args):
    import numpy as np
    from plyfile import PlyData
    from object_registration import scene_vertices, observed_correspondences, reprojection_gate

    source_path = bound_path(task, item, "mesh_path", "mesh_sha256")
    target_path = bound_path(task, observed, "ply_path", "sha256")
    source = scene_vertices(source_path)
    rows = PlyData.read(str(target_path))["vertex"].data
    target = np.column_stack([rows[key] for key in ("x", "y", "z")]).astype(float)
    matrix, pairs, geometry = observed_correspondences(source, target, np.asarray(item["initial_object_to_world"]),
                                                       args.registration_threshold, .7, args.seed)
    geometry["correspondence_origin"] = "nearest_observed_geometry_from_exact_official_coarse_fine_inverse_fixed_before_fit_validation_split"
    context = read_json(directory / "observed_context.json")
    for binding in context["observation_mask_bindings"]:
        bound_path(task, binding)
    heldout_views = [view for view in observed["observations"] if view["frame_id"] != context["selected_frame_id"]]
    reprojection = reprojection_gate(task, source, target, matrix, pairs, heldout_views, cameras, 8)
    proof_path = directory / "correspondences.json"
    write_json(proof_path, {"object_id": item["object_id"], "source_indexing": "trimesh_sorted_scene_nodes_baked_vertices",
                           "target_indexing": "ply_vertex_row", "pairs": pairs, "geometry": geometry, "reprojection": reprojection})
    registration = {"object_id": item["object_id"], "status": "accepted", "method": "ransac_correspondence_sim3",
                    "object_to_world": matrix.tolist(), "initial_object_to_world": item["initial_object_to_world"],
                    "source_coordinate_frame": "object_local", "source_units": "asset_units",
                    "target_coordinate_frame": observed["coordinate_frame"], "target_units": observed["units"],
                    "source_geometry_path": item["mesh_path"], "source_geometry_sha256": sha256(source_path),
                    "target_object_ply_sha256": sha256(target_path), "target_scene_sha256": sha256(scene_path),
                    "correspondences_path": relative(task, proof_path), "correspondences_sha256": sha256(proof_path),
                    "max_error_world_units": args.registration_threshold, "geometry": geometry, "reprojection": reprojection,
                    "observed_context_sha256": sha256(directory / "observed_context.json"),
                    "normalization_sha256": item["normalization_sha256"], "learned_completion_visual_acceptance": "not_reviewed"}
    registration["observation_mask_bindings"] = context["observation_mask_bindings"]
    path = directory / "registration.json"
    write_json(path, registration)
    return {**item, "registration": registration, "registration_path": relative(task, path), "registration_sha256": sha256(path),
            "registration_source_geometry_path": item["mesh_path"], "registration_source_geometry_sha256": sha256(source_path),
            "room_alignment": "observed_correspondence_verified", "observed_anchor": observed.get("observed_anchor"),
            "observed_world_bounds": observed.get("world_bounds")}


def prepare_objects(task, objects, assembled, cameras, depths, output, args, receipt, receipt_path):
    """Prepare completion inputs for every observed object, recording rejections.

    One unusable object must not discard the objects that are usable:
    ``object_lifting`` already reports and drops rejected tracks instead of
    failing the pipeline, and this stage follows the same rule. A run where
    nothing survived is still fatal.
    """
    for object_id, obj in objects.items():
        directory = output / object_id
        directory.mkdir()
        try:
            receipt["objects"].append(prepare_object(task, obj, assembled[object_id], cameras, depths, directory, args))
        except FixerError as error:
            receipt.setdefault("rejected_objects", []).append({"object_id": object_id, "reason": str(error)})
            write_json(receipt_path, receipt)
    require(receipt["objects"],
            "every observed object failed completion input preparation: "
            + "; ".join(f"{item['object_id']}: {item['reason']}" for item in receipt.get("rejected_objects", [])))
    return receipt


def run(args):
    task = args.task_dir.resolve(strict=True)
    paths = {role: input_artifact(args, role).resolve() for role in SPEC.inputs}
    declared_inputs = read_json(local_path(task, args.inputs))["inputs"]
    for role, path in paths.items():
        require(bound_path(task, declared_inputs[role]) == path, f"input envelope hash mismatch: {role}")
    documents, objects, assembled = validate_documents(task, paths)
    args.observation_bindings = documents["verified_mask_bindings"]
    cameras = indexed(documents["cameras"]["frames"], "frame_id")
    depths = indexed(documents["scene_depth"]["frames"], "frame_id")
    voxel = float(documents["lifting_report"]["voxel_size"])
    require(0 < voxel < float("inf"), "finite positive lifting voxel size required")
    args.registration_threshold = 3 * voxel
    require(args.confidence_min >= .5 and args.min_points >= 32, "observed depth confidence/point-count gates cannot be relaxed")
    require(64 <= args.texture_size <= 4096, "texture size must be between 64 and 4096")
    stage = local_path(task, args.work_dir or "stages/observed_context_completion", exists=False)
    stage.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="run-", dir=stage))
    receipt = {"status": "preparing", "upstream_commit": UPSTREAM_COMMIT, "model_inference_performed": False,
               "model_inference_attempted": False,
               "source_artifacts": {role: {"path": relative(task, path), "sha256": sha256(path)} for role, path in paths.items()},
               "objects": [], "rejected_objects": [], "roles_published": False}
    receipt_path = output / "provider_report.json"
    write_json(receipt_path, receipt)
    try:
        prepare_objects(task, objects, assembled, cameras, depths, output, args, receipt, receipt_path)
        if args.prepare_only:
            receipt["status"] = "observed_inputs_prepared_no_inference_no_roles"
            return receipt
        runtime = runtime_preflight(args)
        receipt["runtime"] = runtime
        # All writable runtime caches are owned by this run. Every pretrained
        # loader below receives explicit local files; no remote torch.hub path.
        for key, suffix in (("TMPDIR", "tmp"), ("XDG_CACHE_HOME", "cache"), ("TORCH_HOME", "torch-cache")):
            path = output / suffix
            path.mkdir()
            os.environ[key] = str(path)
        tempfile.tempdir = str(output / "tmp")
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
        pipeline = load_pipeline(runtime)
        completed = []
        for context in receipt["objects"]:
            directory = output / context["object_id"]
            receipt["model_inference_attempted"] = True
            write_json(receipt_path, receipt)
            item = infer_object(task, context, pipeline, directory, args, receipt)
            completed.append(register_object(task, item, objects[item["object_id"]], cameras, paths["scene_gaussian_ply"], directory, args))
        for role, bound in receipt["source_artifacts"].items():
            bound_path(task, bound)
        for context in receipt["objects"]:
            for binding in context["observation_mask_bindings"]:
                bound_path(task, binding)
        candidates_path, meshes_path = output / "completion_candidates.json", output / "completed_object_meshes.json"
        write_json(candidates_path, {"backend": "learned_three_d_fixer", "objects": completed, "room_alignment_validated": True,
                                     "generated_orbit_used": False, "visual_acceptance": "not_reviewed", "provider_report_path": relative(task, receipt_path)})
        write_json(meshes_path, {"objects": completed})
        publish(args, {"completion_candidates": candidates_path, "completed_object_meshes": meshes_path})
        receipt.update(status="generated_and_geometrically_registered", roles_published=True)
        return receipt
    except Exception as error:
        receipt.update(status="blocked_three_d_fixer", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json(receipt_path, receipt)


def argument_parser():
    app = argparse.ArgumentParser(description=__doc__)
    for key in ("task-dir", "inputs", "outputs"):
        app.add_argument("--" + key, type=Path, required=True)
    for key in ("source", "model-dir", "trellis-base", "moge-checkpoint", "dino-source", "dino-checkpoint"):
        app.add_argument("--" + key, type=Path)
    app.add_argument("--prepare-only", action="store_true")
    app.add_argument("--work-dir", type=Path, help="Optional task-local run parent, including a QA directory")
    app.add_argument("--confidence-min", type=float, default=.5)
    app.add_argument("--min-points", type=int, default=64)
    app.add_argument("--texture-size", type=int, default=1024)
    app.add_argument("--observed-refinement", action="store_true",
                     help="opt-in data-fit step pulling mesh vertices toward nearby observed points before registration")
    app.add_argument("--refinement-radius", type=float, default=0.0,
                     help="world-unit radius around observed points; 0 derives 2.5 * registration threshold")
    app.add_argument("--refinement-strength", type=float, default=0.5)
    app.add_argument("--refinement-smoothing", type=float, default=0.3)
    app.add_argument("--refinement-iterations", type=int, default=8)
    app.add_argument("--seed", type=int, default=0)
    return app


if __name__ == "__main__":
    run(argument_parser().parse_args())
