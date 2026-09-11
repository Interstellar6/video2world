#!/usr/bin/env python3
"""Qwen physical priors and separate visual OBJ / convex collision references."""

from __future__ import annotations

import json
import hashlib
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from world_modeling.provider_io import input_artifact, local_path, parser, publish, read_json, write_json
from world_modeling.object_lineage import lineage_metadata, validate_object_lineages


def parse_estimate(response: str) -> dict:
    text = response.strip()
    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.splitlines()[1:-1])
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("physical estimate must be an object")
    for name in ("length_m", "width_m", "height_m", "mass_kg"):
        bounds = value.get(name)
        if bounds is not None and (not isinstance(bounds, list) or len(bounds) != 2 or
            not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in bounds)
            or not 0 < bounds[0] <= bounds[1]):
            raise ValueError(f"{name} requires a positive [low, high] interval or null")
    for name in ("surface_smoothness", "confidence"):
        number = value.get(name)
        if not isinstance(number, (int, float)) or isinstance(number, bool) or not 0 <= number <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    if not isinstance(value.get("basis"), str) or not value["basis"].strip():
        raise ValueError("basis is required")
    return value


def checked_file(task, value, digest=None):
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError("physics input requires a file path")
    path = local_path(task, value)
    if not path.is_file():
        raise ValueError("physics input must be a regular file")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest is not None and digest != actual:
        raise ValueError("physics source/collider SHA256 mismatch")
    return path, actual


def prepare_inputs(args):
    from PIL import Image

    task = args.task_dir.resolve()
    paths = {name: input_artifact(args, name) for name in
             ("object_descriptions", "repaired_visual_meshes", "coacd_collision_meshes", "mesh_qa_report")}
    envelope = read_json(local_path(task, args.inputs))["inputs"]
    for role, path in paths.items():
        checked_file(task, path, envelope[role].get("sha256"))
    visual = read_json(paths["repaired_visual_meshes"])["objects"]
    colliders = read_json(paths["coacd_collision_meshes"])["objects"]
    qa = read_json(paths["mesh_qa_report"])
    if not visual:
        raise ValueError("empty repaired_visual_meshes")
    descriptions = validate_object_lineages(task, visual, paths["object_descriptions"])
    validate_object_lineages(task, colliders, paths["object_descriptions"])
    collision = {obj["object_id"]: obj for obj in colliders}
    plans = []
    for obj in colliders:
        checked_file(task, obj.get("mesh_path", obj.get("path")), obj.get("sha256", obj.get("mesh_sha256")))
        hulls = obj.get("hulls")
        if obj.get("collision_eligible") is not True or not isinstance(hulls, list) or not hulls:
            raise ValueError(f"{obj['object_id']}: missing independently checked collider hulls")
        for hull in hulls:
            if not isinstance(hull, dict):
                raise ValueError("invalid collider hull reference")
            checked_file(task, hull.get("path"), hull.get("sha256"))
    for obj in visual:
        object_id = obj["object_id"]
        if object_id not in collision:
            raise ValueError(f"{object_id}: missing independently checked collider")
        if lineage_metadata(obj) != lineage_metadata(collision[object_id]):
            raise ValueError(f"{object_id}: visual/collider lineage differs")
        geometry, _ = checked_file(task, obj.get("mesh_path", obj.get("path")), obj.get("sha256", obj.get("mesh_sha256")))
        observations = descriptions[object_id]
        if not observations:
            raise ValueError(f"{object_id}: no grounding description")
        description = max(observations, key=lambda value: value.get("confidence", 0))
        image, image_hash = checked_file(task, description.get("image_path"))
        with Image.open(image) as source:
            width, height = source.size
            source.verify()
        bbox = description.get("bbox_xyxy")
        if not isinstance(bbox, list) or len(bbox) != 4 or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in bbox
        ) or not (0 <= bbox[0] < bbox[2] <= width and 0 <= bbox[1] < bbox[3] <= height):
            raise ValueError(f"{object_id}: invalid grounding image bbox")
        reference = {"frame_id": description.get("frame_id"), "source_object_id": description["object_id"],
                     "descriptions_path": paths["object_descriptions"].relative_to(task).as_posix(),
                     "descriptions_sha256": hashlib.sha256(paths["object_descriptions"].read_bytes()).hexdigest(),
                     "image_path": image.relative_to(task).as_posix(), "image_sha256": image_hash,
                     "bbox_xyxy": bbox, "identity_scope": "source_observation_lineage" if lineage_metadata(obj) else "legacy_object_id_join"}
        for observation in obj.get("source_observations", []):
            if (observation["frame_id"], observation["source_object_id"]) == (description["frame_id"], description["object_id"]):
                reference.update(mask_path=observation["source_mask_path"], mask_sha256=observation["source_mask_sha256"])
        plans.append({"object": obj, "description": description, "source_description": reference,
                      "image_path": image, "geometry_path": geometry, "collision": collision[object_id]})
    return plans, qa


def run(args) -> int:
    args.task_dir = args.task_dir.resolve()
    plans, qa = prepare_inputs(args)
    import torch
    import trimesh
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.bfloat16, local_files_only=True).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True, use_fast=False)
    stage = local_path(args.task_dir, args.outputs, exists=False).parent / "physics"
    stage.mkdir(parents=True, exist_ok=True)
    objects, priors = [], []
    for index, plan in enumerate(plans):
        obj = plan["object"]
        object_id = obj["object_id"]
        description = plan["description"]
        with Image.open(plan["image_path"]) as source:
            image = source.convert("RGB").crop(tuple(description["bbox_xyxy"]))
        # The description is supplied before the instruction. Ending the prompt
        # with a JSON blob makes a small VL model continue that blob instead of
        # answering, which is what the earlier ordering produced.
        prompt = (
            "Target object description (JSON, informational only; do not repeat it): " + json.dumps(description) + "\n\n"
            "Estimate uncertain physical priors for the single target object visible in this crop. "
            "Return one JSON object only, with no prose and no repeated description, containing "
            "length_m,width_m,height_m,mass_kg each a [low,high] interval or null if unknown; "
            "surface_smoothness a number 0(rough)-1(smooth); confidence a number 0-1; and basis a short explanation. "
            "Use category priors and broad realistic intervals; image pixels do not measure dimensions or mass. "
            "Do not infer Coulomb friction or restitution from smoothness."
        )
        attempts = []
        estimate = None
        for _attempt in range(2):
            messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=700, do_sample=False)
            response = processor.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
            attempts.append({"prompt": prompt, "raw_response": response})
            try:
                estimate = parse_estimate(response)
                break
            except ValueError:
                prompt = ("The previous reply was not the requested JSON object. Reply with exactly one JSON object and nothing else, "
                          "containing length_m, width_m, height_m, mass_kg, surface_smoothness, confidence and basis.")
        write_json(stage / f"qwen_{index:04d}.json", {"object_id": object_id, "source_description": plan["source_description"],
                   "model": str(args.model), "attempts": attempts})
        if estimate is None:
            raise ValueError(f"{object_id}: model did not return a valid physical estimate after retry")
        geometry_path = plan["geometry_path"]
        scene = trimesh.load(geometry_path, force="scene")
        if not scene.geometry:
            raise ValueError(f"{object_id}: repaired mesh contains no geometry")
        directory = stage / f"object_{index:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        # The exporter writes OBJ plus material/texture files beside it.
        from trimesh.exchange.obj import export_obj
        obj_path = directory / "visual.obj"
        obj_path.write_text(export_obj(scene, include_texture=True, return_texture=False, write_texture=True,
                            resolver=trimesh.resolvers.FilePathResolver(str(directory)), digits=8), encoding="utf-8")
        reloaded = trimesh.load(obj_path, force="scene")
        if not reloaded.geometry:
            raise ValueError("OBJ round-trip has no geometry")
        objects.append({"object_id": object_id, "obj_path": str(obj_path.relative_to(args.task_dir)),
                        "source_mesh_path": str(geometry_path.relative_to(args.task_dir)),
                        "coordinate_frame": obj.get("coordinate_frame", "object_local"),
                        "units": obj.get("unit", "asset_units"), "scale_applied": False,
                        "collision": plan["collision"], "role": "visual_obj_with_separate_convex_collision",
                        **lineage_metadata(obj), "source_description": plan["source_description"]})
        priors.append({"object_id": object_id, **estimate, "evidence": "single_view_category_prior",
                       "measured": False, "dimensions_applied_to_geometry": False,
                       "friction_coefficient": None, "restitution": None,
                       **lineage_metadata(obj), "source_description": plan["source_description"]})
    paths = {role: stage / f"{role}.json" for role in ("physical_object_obj", "physics_properties", "physics_report")}
    write_json(paths["physical_object_obj"], {"schema_version": "1.0", "objects": objects})
    write_json(paths["physics_properties"], {"schema_version": "1.0", "objects": priors})
    write_json(paths["physics_report"], {"schema_version": "1.0", "object_count": len(objects), "model": str(args.model),
                                       "mesh_qa_status": qa.get("status"), "calibrated_physics": False,
                                       "simulator_validation": "not_performed"})
    publish(args, paths)
    return 0


def main(argv=None) -> int:
    app = parser(__doc__)
    app.add_argument("--model", type=Path, required=True)
    return run(app.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
