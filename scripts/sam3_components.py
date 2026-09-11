#!/usr/bin/env python3
"""Task-local native SAM3-I instruction masks with explicit identity hypotheses."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from world_modeling.provider_io import input_artifact, local_path, parser, publish, read_json, write_json


class CheckpointCompatibilityError(ValueError):
    def __init__(self, audit: dict):
        self.audit = audit
        super().__init__("incompatible SAM3 checkpoint: " + str({key: audit[key][:8] for key in
                         ("missing_keys", "shape_mismatches", "unknown_extra_keys")}))


def identifier(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}", value):
        raise ValueError(f"invalid path identifier: {value!r}")
    return value


def normalized_box(box: list, width: int, height: int) -> list[float]:
    if width < 1 or height < 1 or not isinstance(box, list) or len(box) != 4:
        raise ValueError("box requires four original-image pixel coordinates")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
        raise ValueError("box coordinates must be finite numbers")
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"box outside {width}x{height}: {box}")
    return [(x0 + x1) / (2 * width), (y0 + y1) / (2 * height), (x1 - x0) / width, (y1 - y0) / height]


def checkpoint_state(checkpoint: dict) -> dict:
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a state dictionary")
    for name in ("model", "state_dict"):
        if isinstance(checkpoint.get(name), dict):
            return checkpoint[name]
    return checkpoint


def compatible_state(expected: dict, source: dict) -> tuple[dict, dict]:
    """Do not silently discard an instruction model's learned branches."""
    state = checkpoint_state(source)
    detector = any(key.startswith("detector.") for key in state)
    candidate = {key.removeprefix("detector."): value for key, value in state.items() if key.startswith("detector.")} if detector else state
    missing = sorted(set(expected) - set(candidate))
    mismatched = sorted(key for key in set(expected) & set(candidate) if tuple(expected[key].shape) != tuple(candidate[key].shape))
    extra = sorted(set(candidate) - set(expected))
    instruction_state = all(any(token in key for key in candidate) for token in ("adapter", "simple_query", "complex_query"))
    unknown_extra = extra
    audit = {"expected_keys": len(expected), "checkpoint_keys": len(state), "missing_keys": missing,
             "shape_mismatches": mismatched, "ignored_checkpoint_keys": extra, "unknown_extra_keys": unknown_extra,
             "key_mapping": "detector_prefix" if detector else "direct", "strict_load": True,
             "checkpoint_origin": "sam3i_instruction_checkpoint_requires_native_backend" if instruction_state else "sam3_complete_state"}
    if missing or mismatched or unknown_extra:
        raise CheckpointCompatibilityError(audit)
    return {key: candidate[key] for key in expected}, audit


def native_compatible_state(expected: dict, source: dict) -> tuple[dict, dict]:
    state = checkpoint_state(source)
    missing = sorted(set(expected) - set(state))
    mismatched = sorted(k for k in set(expected) & set(state) if tuple(expected[k].shape) != tuple(state[k].shape))
    extra = sorted(set(state) - set(expected))
    unknown = extra
    counts = {name: sum(name in k for k in expected) for name in ("adapter", "simple_query", "complex_query")}
    if not all(counts.values()):
        missing.append("required_native_instruction_modules")
    audit = {"expected_keys": len(expected), "checkpoint_keys": len(state), "missing_keys": missing,
             "shape_mismatches": mismatched, "ignored_checkpoint_keys": extra, "unknown_extra_keys": unknown,
             "loaded_instruction_keys": counts, "strict_load": True, "checkpoint_origin": "sam3i_stage3_full_native_instruction_model"}
    if missing or mismatched or unknown:
        raise CheckpointCompatibilityError(audit)
    return {key: state[key] for key in expected}, audit


def instruction_prompt(obj, item):
    if item["component_id"] == "__object__":
        explicit = obj.get("segmentation_instruction")
        prefix = f"the complete {obj['category']} including all its visible constituent parts"
        description = obj.get("description", "")
    else:
        component = next(c for c in obj.get("components", []) if c["component_id"] == item["component_id"])
        explicit = component.get("segmentation_instruction")
        prefix = f"the {component['name']} belonging to the {obj['category']}"
        description = component.get("description", "")
    if explicit is not None:
        if not isinstance(explicit, str) or not explicit.strip():
            raise ValueError("segmentation_instruction must be nonempty text")
        return explicit.strip()
    if item.get("instance_mode") == "all":
        return item["text"]
    return prefix + (". " + description.strip() if isinstance(description, str) and description.strip() else "")


def native_prediction(model, image, prompt, concept, stage, threshold):
    import numpy as np
    import torch
    from sam3.eval.postprocessors import PostProcessImage
    from sam3.model.utils.misc import copy_data_to_device
    from sam3.train.data.collator import collate_fn_api
    from sam3.train.data.sam3_image_dataset import Datapoint, FindQueryLoaded, Image as SAMImage, InferenceMetadata
    from sam3.train.transforms.basic_for_api import ComposeAPI, NormalizeAPI, RandomResizeAPI, ToTensorAPI

    text_input = {"concept": [concept], "simple_query": [prompt, prompt], "complex_query": [prompt, prompt]}
    datapoint = Datapoint(find_queries=[], images=[SAMImage(data=image, objects=[], size=[image.height, image.width])])
    datapoint.find_queries.append(FindQueryLoaded(query_text=text_input, image_id=0, object_ids_output=[],
        is_exhaustive=True, query_processing_order=0, inference_metadata=InferenceMetadata(coco_image_id=0,
        original_image_id=0, original_category_id=1, original_size=[image.height, image.width], object_id=0, frame_index=0)))
    transform = ComposeAPI(transforms=[RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
        ToTensorAPI(), NormalizeAPI(mean=[.5, .5, .5], std=[.5, .5, .5])])
    batch = collate_fn_api([transform(datapoint)], dict_key="dummy")["dummy"]
    device = next(model.parameters()).device
    batch = copy_data_to_device(batch, device, non_blocking=True)
    output = model(batch, stage)
    postprocessor = PostProcessImage(max_dets_per_img=-1, iou_type="segm", use_original_sizes_box=True,
        use_original_sizes_mask=True, convert_mask_to_rle=False, detection_threshold=threshold, to_cpu=False)
    result = postprocessor.process_results(output, batch.find_metadatas).get(0)
    if result is None:
        return np.empty((0, image.height, image.width), dtype=bool), np.empty(0)
    return prediction_arrays(result)


def instruction_token_audit(model, prompt):
    encoder = model.backbone.language_backbone
    tokens = encoder.tokenizer.encode(prompt)
    capacity = encoder.context_length - 2
    return {"token_count": len(tokens) + 2, "context_length": encoder.context_length,
            "truncated": len(tokens) > capacity,
            "effective_text": encoder.tokenizer.decode(tokens[:capacity])}


def choose_mask(masks, scores, box, width, height, *, parent_mask=None, max_frame_fraction=0.90, min_pixels=32):
    import numpy as np

    normalized_box(box, width, height)
    masks = np.asarray(masks)
    scores = np.asarray(scores).reshape(-1)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3 or masks.shape[1:] != (height, width) or len(masks) != len(scores):
        raise ValueError("SAM3 mask dimensions or score count do not match source image")
    x0, y0, x1, y1 = box
    box_pixels = max(1, (math.ceil(x1) - math.floor(x0)) * (math.ceil(y1) - math.floor(y0)))
    audits = []
    selected = None
    for index, (mask, score) in enumerate(zip(masks, scores)):
        if not np.isfinite(mask).all():
            audits.append({"index": index, "rejection_reasons": ["nonfinite_mask"]})
            continue
        mask = mask > 0.5
        area = int(mask.sum())
        inside = int(mask[math.floor(y0):math.ceil(y1), math.floor(x0):math.ceil(x1)].sum())
        parent_fraction = float((mask & parent_mask).sum() / max(area, 1)) if parent_mask is not None else None
        fraction = area / (width * height)
        inside_fraction = inside / max(area, 1)
        box_iou = inside / max(area + box_pixels - inside, 1)
        reasons = []
        if not math.isfinite(float(score)) or not 0 <= score <= 1:
            reasons.append("invalid_score")
        if area < min_pixels:
            reasons.append("empty_or_tiny_mask")
        if fraction > max_frame_fraction:
            reasons.append("room_scale_mask")
        if area > box_pixels * 4 or inside_fraction < 0.50:
            reasons.append("mask_does_not_match_proposal_box")
        if parent_fraction is not None and parent_fraction < 0.50:
            reasons.append("component_outside_parent_mask")
        audit = {"index": index, "score": float(score) if math.isfinite(float(score)) else None,
                 "pixels": area, "frame_fraction": fraction, "inside_box_fraction": inside_fraction,
                 "box_iou": box_iou, "parent_overlap_fraction": parent_fraction, "rejection_reasons": reasons}
        audits.append(audit)
        utility = float(score) * (inside_fraction + box_iou) if not reasons else -1
        if not reasons and (selected is None or utility > selected[0]):
            selected = utility, mask, audit
    return (None, None, audits) if selected is None else (selected[1], selected[2], audits)


def prediction_arrays(prediction):
    return tuple(prediction[key].detach().float().cpu().numpy() for key in ("masks", "scores"))


def proposed_items(obj: dict):
    object_id = identifier(obj["object_id"])
    yield {"object_id": object_id, "component_id": "__object__", "text": obj["category"], "bbox_xyxy": obj["bbox_xyxy"]}
    seen = set()
    for component in obj.get("components", []):
        component_id = identifier(component["component_id"])
        if component_id == "__object__" or component_id in seen:
            raise ValueError(f"duplicate or reserved component ID: {component_id}")
        seen.add(component_id)
        if component.get("parent_object_id", object_id) != object_id:
            raise ValueError("component parent_object_id differs from enclosing object")
        mode = component.get("instance_mode", "single")
        if mode not in ("single", "all"):
            raise ValueError("component instance_mode must be single or all")
        yield {"object_id": object_id, "component_id": component_id,
               "text": component_id.replace("_", " ") if mode == "all" else component["name"],
               "name": component["name"],
               "bbox_xyxy": component.get("bbox_xyxy"), "visibility": component.get("visibility"),
               "instance_mode": mode, "membership_relation": component.get("relationship", "unspecified")}


def select_instances(masks, scores, box, width, height, *, parent_mask=None,
                     max_frame_fraction=0.90, confidence_threshold=0.25):
    import numpy as np

    # All-instance outputs remain candidates. A parent omission must not erase
    # the observations needed for the independent geometry/membership gate.
    _, _, audits = choose_mask(masks, scores, box, width, height, max_frame_fraction=max_frame_fraction)
    values = np.asarray(masks)
    if values.ndim == 4:
        values = values[:, 0]
    selected = []
    for audit in audits:
        if audit.get("score") is not None and audit["score"] <= confidence_threshold:
            audit["rejection_reasons"].append("below_confidence_threshold")
        mask = values[audit["index"]] > .5
        if parent_mask is not None:
            audit["parent_overlap_fraction"] = float((mask & parent_mask).sum() / max(1, mask.sum()))
        audit["parent_overlap_enforced"] = False
        if not audit["rejection_reasons"]:
            selected.append((mask, audit))
    return selected, audits


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    app = parser(__doc__)
    app.add_argument("--source-root", type=Path, required=True)
    app.add_argument("--checkpoint", type=Path, required=True)
    app.add_argument("--confidence-threshold", type=float, default=0.25)
    app.add_argument("--max-frame-fraction", type=float, default=0.90)
    app.add_argument("--checkpoint-check-only", action="store_true")
    app.add_argument("--backend", choices=("sam3i", "sam3"), default="sam3i")
    app.add_argument("--instruction-stage", choices=("complex", "simple"), default="complex")
    args = app.parse_args()
    if not 0 <= args.confidence_threshold < 1 or not 0 < args.max_frame_fraction <= 1:
        app.error("confidence threshold must be in [0,1); max frame fraction in (0,1]")
    stage_parent = local_path(args.task_dir, args.outputs, exists=False).parent
    stage_parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="sam3-", dir=stage_parent))
    for variable, directory in (("TMPDIR", "tmp"), ("XDG_CACHE_HOME", "cache"),
                                ("CUDA_CACHE_PATH", "cuda-cache"), ("TRITON_CACHE_DIR", "triton-cache")):
        target = stage / directory
        target.mkdir()
        os.environ[variable] = str(target)
    tempfile.tempdir = str(stage / "tmp")
    source_root = args.source_root.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    package_root = source_root / "sam3" if args.backend == "sam3i" else source_root
    if not (package_root / "sam3/model_builder.py").is_file():
        raise ValueError(f"{args.backend} source layout is missing: {package_root / 'sam3/model_builder.py'}")
    sys.path.insert(0, str(package_root))
    print(f"SAM3 backend={args.backend} pid={os.getpid()} stage={args.instruction_stage}", flush=True)

    import numpy as np
    import torch
    from PIL import Image
    from sam3.model_builder import build_sam3_image_model
    bpe_path = package_root / "assets/bpe_simple_vocab_16e6.txt.gz" if args.backend == "sam3i" else package_root / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    if not bpe_path.is_file():
        raise ValueError(f"SAM3 tokenizer vocabulary missing: {bpe_path}")
    build_options = {"device": "cpu", "bpe_path": str(bpe_path), "load_from_HF": False,
                     "checkpoint_path": None, "enable_inst_interactivity": False}
    stage_name = "1_2" if args.instruction_stage == "complex" else "1_1"
    if args.backend == "sam3i":
        build_options.update({"inst_stage": "3", "adapter_config": {"adapter_dim": 64, "adapter_heads": 4, "adapter_scale": 1.0},
                              "share_mlp_params": True, "use_margin_loss": True, "use_infonce_loss": True})
    model = build_sam3_image_model(**build_options)
    if args.backend == "sam3i":
        from sam3.model_builder import _replace_text_encoder

        _replace_text_encoder(model, str(bpe_path), "1_1", build_options["adapter_config"])
    raw_state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    try:
        state, checkpoint_audit = (native_compatible_state if args.backend == "sam3i" else compatible_state)(model.state_dict(), raw_state)
    except CheckpointCompatibilityError as error:
        write_json(stage / "checkpoint_audit.json", {"status": "rejected", "checkpoint": str(checkpoint), **error.audit})
        raise
    model.load_state_dict(state, strict=True)
    del state, raw_state
    commit = subprocess.run(["git", "-c", f"safe.directory={source_root}", "-C", str(source_root), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False)
    checkpoint_audit.update({"status": "passed", "checkpoint": str(checkpoint), "sha256": file_hash(checkpoint),
                             "source_root": str(source_root), "source_commit": commit.stdout.strip() if commit.returncode == 0 else None,
                             "backend": args.backend, "instruction_stage": stage_name if args.backend == "sam3i" else None,
                             "construction_stage": "3" if args.backend == "sam3i" else None,
                             "box_usage": "candidate_matching_only" if args.backend == "sam3i" else "geometric_prompt"})
    write_json(stage / "checkpoint_audit.json", checkpoint_audit)
    if args.checkpoint_check_only:
        print("SAM3 checkpoint compatibility passed; no inference executed", flush=True)
        return 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    processor = None
    if args.backend == "sam3":
        from sam3.model.sam3_image_processor import Sam3Processor

        processor = Sam3Processor(model, device=device, confidence_threshold=args.confidence_threshold)
    proposals_path = input_artifact(args, "object_proposals")
    descriptions_path = input_artifact(args, "object_descriptions")
    proposal_document = read_json(proposals_path)
    proposals = proposal_document["frames"]
    source_descriptions = {"path": descriptions_path.relative_to(args.task_dir.resolve()).as_posix(), "sha256": file_hash(descriptions_path)}
    source_proposals = {"path": proposals_path.relative_to(args.task_dir.resolve()).as_posix(), "sha256": file_hash(proposals_path)}
    frame_manifest = read_json(input_artifact(args, "frames_manifest"))["frames"]
    frames = {frame["frame_id"]: frame for frame in frame_manifest}
    if not proposals or len(frames) != len(frame_manifest):
        raise ValueError("empty proposals or duplicate source frame IDs")
    results, failures, quality, used_frames, tracks = [], [], [], [], {}
    seen_frames = set()
    for proposal_frame in proposals:
        frame_id = identifier(proposal_frame["frame_id"])
        if frame_id in seen_frames or frame_id not in frames:
            raise ValueError(f"duplicate or unknown proposal frame: {frame_id}")
        seen_frames.add(frame_id)
        frame = frames[frame_id]
        image_path = local_path(args.task_dir, frame["image_path"])
        if local_path(args.task_dir, proposal_frame["image_path"]) != image_path:
            raise ValueError(f"proposal and source RGB differ: {frame_id}")
        with Image.open(image_path) as original:
            image = original.convert("RGB")
        if image.size != (frame["width"], frame["height"]) or image.size != (proposal_frame["width"], proposal_frame["height"]):
            raise ValueError(f"source dimensions differ from proposals: {frame_id}")
        image_relative = image_path.relative_to(args.task_dir.resolve()).as_posix()
        image_sha256 = file_hash(image_path)
        used_frames.append({"frame_id": frame_id, "image_path": image_relative, "width": image.width, "height": image.height})
        with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            state = processor.set_image(image) if processor is not None else None
            seen_objects = set()
            for obj in proposal_frame["objects"]:
                if obj["object_id"] in seen_objects:
                    raise ValueError(f"duplicate object ID in frame: {obj['object_id']}")
                seen_objects.add(obj["object_id"])
                if obj["object_id"] in tracks and tracks[obj["object_id"]]["category"] != obj["category"]:
                    raise ValueError(f"object ID changed category across frames: {obj['object_id']}")
                parent_mask = None
                for item in proposed_items(obj):
                    record = {"frame_id": frame_id, "image_path": image_relative, "image_sha256": image_sha256,
                              "category": obj["category"], "source_object_id": obj["object_id"], **item}
                    try:
                        if item["component_id"] != "__object__" and item.get("visibility") not in ("visible", "partial"):
                            raise ValueError("unobserved_component")
                        all_instances = item.get("instance_mode") == "all"
                        if item["component_id"] != "__object__" and parent_mask is None and not all_instances:
                            raise ValueError("parent_mask_unavailable")
                        box = normalized_box(item["bbox_xyxy"], image.width, image.height)
                        if args.backend == "sam3i":
                            prompt = instruction_prompt(obj, item)
                            record.update({"instruction": prompt, "instruction_stage": stage_name, "box_usage": "candidate_matching_only",
                                           "instruction_tokens": instruction_token_audit(model, prompt)})
                            masks, scores = native_prediction(model, image, prompt, item["text"], stage_name, args.confidence_threshold)
                        else:
                            processor.reset_all_prompts(state)
                            processor.set_text_prompt(item["text"], state)
                            prediction = processor.add_geometric_prompt(box=box, label=True, state=state)
                            masks, scores = prediction_arrays(prediction)
                        if all_instances:
                            selections, candidates = select_instances(masks, scores, item["bbox_xyxy"], image.width, image.height,
                                parent_mask=parent_mask, max_frame_fraction=args.max_frame_fraction, confidence_threshold=args.confidence_threshold)
                        else:
                            mask, accepted, candidates = choose_mask(masks, scores, item["bbox_xyxy"], image.width, image.height,
                                parent_mask=parent_mask, max_frame_fraction=args.max_frame_fraction)
                            selections = [(mask, accepted)] if mask is not None else []
                        quality.append({**record, "candidates": candidates, "selected_indices": [audit["index"] for _, audit in selections]})
                        if not selections:
                            raise ValueError("no_candidate_passed_mask_quality")
                        for mask, accepted in selections:
                            component_id = item["component_id"] + (f"__{accepted['index']:03d}" if all_instances else "")
                            mask_path = stage / "masks" / frame_id / item["object_id"] / (component_id + ".png")
                            local_path(args.task_dir, mask_path, exists=False).parent.mkdir(parents=True, exist_ok=True)
                            Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
                            mask_relative = mask_path.relative_to(args.task_dir.resolve()).as_posix()
                            member_record = {**record, "component_id": component_id, "mask_path": mask_relative,
                                             "mask_sha256": file_hash(mask_path), "score": accepted["score"], "quality": accepted}
                            if item["component_id"] != "__object__":
                                identity = [frame_id, item["object_id"], item["component_id"], accepted["index"]]
                                member_record.update({"observation_id": "obs_" + hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:24],
                                    "component_group_id": item["component_id"], "identity_scope": "frame_local_observation",
                                    "geometry_validated": False})
                            results.append(member_record)
                            if item["component_id"] == "__object__":
                                parent_mask = mask
                                track = tracks.setdefault(item["object_id"], {"object_id": item["object_id"], "category": obj["category"],
                                    "observations": [], "association_status": "vlm_identity_hypothesis", "geometry_validated": False,
                                    "identity_scope": obj.get("identity_scope", "vlm_identity_hypothesis")})
                                track["observations"].append({"frame_id": frame_id, "image_path": image_relative,
                                    "mask_path": mask_relative, "mask_sha256": member_record["mask_sha256"],
                                    "bbox_xyxy": item["bbox_xyxy"], "score": accepted["score"],
                                    "source_object_id": item["object_id"], "image_sha256": image_sha256})
                    except ValueError as error:
                        failures.append({**record, "reason": str(error)})
        write_json(stage / "quality.json", {"frames": used_frames, "proposals": quality, "failures": failures,
                                           "checkpoint_audit": checkpoint_audit, "model_inference_executed": True})
        print(f"frame={frame_id} accepted_masks={len(results)} failures={len(failures)}", flush=True)
    masks_path, tracks_path = stage / "component_mask_candidates.json", stage / "physical_instance_hypotheses.json"
    write_json(masks_path, {"schema_version": "1.0", "frames": used_frames, "masks": results, "failures": failures,
                           "quality_path": (stage / "quality.json").relative_to(args.task_dir.resolve()).as_posix(),
                           "source_descriptions": source_descriptions, "source_proposals": source_proposals,
                           "mask_encoding": "png_uint8_0_background_255_foreground"})
    write_json(tracks_path, {"schema_version": "1.0", "tracks": list(tracks.values()), "geometry_validated": False,
                            "source_descriptions": source_descriptions, "source_proposals": source_proposals,
                            "identity_scope": proposal_document.get("association_status", "vlm_identity_hypotheses"),
                            "association_method": "Qwen persistent object IDs; depth association required before carving"})
    if not tracks:
        raise ValueError("no parent object mask passed quality; diagnostic manifests retained")
    if file_hash(descriptions_path) != source_descriptions["sha256"] or file_hash(proposals_path) != source_proposals["sha256"]:
        raise ValueError("source proposals or descriptions changed during segmentation")
    publish(args, {"component_mask_candidates": masks_path, "physical_instance_hypotheses": tracks_path})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
