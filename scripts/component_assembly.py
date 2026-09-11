#!/usr/bin/env python3
"""Extract same-camera RGBA views from geometrically verified parent masks."""

from __future__ import annotations

import hashlib
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from world_modeling.provider_io import input_artifact, local_path, parser, publish, read_json, write_json
from world_modeling.object_lineage import validate_object_lineages


METHOD = "same_camera_verified_parent_mask_rgba_extraction"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identifier(value, label):
    require(isinstance(value, (str, int)) and not isinstance(value, bool), f"invalid {label}")
    value = str(value)
    require(bool(value) and value.strip() == value, f"invalid {label}")
    return value


def indexed(records, key, label):
    require(isinstance(records, list), f"{label} must be a list")
    result = {}
    for record in records:
        require(isinstance(record, dict), f"invalid {label} record")
        name = identifier(record.get(key), label)
        require(name not in result, f"duplicate {label}: {name}")
        result[name] = record
    return result


def file_record(task, path):
    return {"path": path.relative_to(task).as_posix(), "sha256": sha256(path)}


def mask_record(task, record, size, *, require_hash):
    import numpy as np
    from PIL import Image

    path = local_path(task, record["mask_path"])
    digest = sha256(path)
    if require_hash or "mask_sha256" in record:
        require(record.get("mask_sha256") == digest, f"parent/component mask hash mismatch: {path.name}")
    with Image.open(path) as source:
        require(source.size == size, f"mask/image raster mismatch: {path.name}")
        values = np.asarray(source.convert("L"))
        require(np.isin(values, [0, 255]).all(), f"mask must be binary: {path.name}")
    return values > 0, path, digest


def load_plan(args):
    import numpy as np
    from PIL import Image

    task = args.task_dir.resolve()
    roles = ("component_masks", "isolated_object_ply", "object_descriptions", "lifting_report", "cameras")
    paths = {role: input_artifact(args, role) for role in roles}
    envelope = read_json(local_path(task, args.inputs))["inputs"]
    for role, path in paths.items():
        require(envelope[role].get("status") not in {"failed", "blocked", "stale", "contract_only"}, f"{role} is unavailable")
        if "sha256" in envelope[role]:
            require(envelope[role]["sha256"] == sha256(path), f"{role} input hash mismatch")
    documents = {role: read_json(path) for role, path in paths.items()}
    report, masks = documents["lifting_report"], documents["component_masks"]
    require(report.get("status") in {"passed", "partial_tracks_rejected"} and report.get("carve_performed") is True,
            "lifting has not accepted observed physical objects")
    require(report.get("generated_geometry_used") is False, "assembly requires observed lifting evidence")
    accepted = report.get("accepted_object_ids")
    require(isinstance(accepted, list) and accepted, "lifting must name accepted objects")
    accepted_ids = [identifier(value, "accepted object") for value in accepted]
    require(len(set(accepted_ids)) == len(accepted_ids), "duplicate accepted object")
    tracks = indexed(report.get("tracks"), "object_id", "lifting track")
    require(all(type(record.get("accepted")) is bool for record in tracks.values()), "lifting accepted flags must be boolean")
    require(set(accepted_ids) == {name for name, record in tracks.items() if record["accepted"]},
            "lifting accepted-object declarations disagree")
    objects = indexed(documents["isolated_object_ply"].get("objects"), "object_id", "isolated object")
    require(set(objects) == set(accepted_ids), "isolated objects disagree with lifting accepted objects")

    # A present resolved declaration is authoritative, even when malformed.
    binding_kind = "resolved_files" if "resolved_files" in report else "source_files"
    bindings = report.get(binding_kind)
    require(isinstance(bindings, dict), f"lifting requires {binding_kind}")
    binding = bindings.get("component_masks")
    require(isinstance(binding, dict), f"lifting does not bind current component_masks in {binding_kind}")
    require(binding.get("sha256") == sha256(paths["component_masks"]) and
            local_path(task, binding.get("path", "")) == paths["component_masks"],
            "lifting does not bind current component_masks")
    resolved = binding_kind == "resolved_files"
    if resolved or "acceptance_authority" in masks:
        require(local_path(task, masks.get("acceptance_authority", "")) == paths["lifting_report"],
                "component_masks acceptance authority disagrees with lifting report")
    camera_binding = report.get("source_files", {}).get("cameras")
    if camera_binding is not None:
        require(isinstance(camera_binding, dict) and camera_binding.get("sha256") == sha256(paths["cameras"]) and
                local_path(task, camera_binding.get("path", "")) == paths["cameras"], "lifting camera binding mismatch")
    frames = indexed(masks.get("frames"), "frame_id", "mask source frame")
    cameras = indexed(documents["cameras"].get("frames"), "frame_id", "camera frame")
    descriptions = documents["object_descriptions"].get("objects")
    require(isinstance(descriptions, list), "object_descriptions requires objects")
    descriptions_by_object = validate_object_lineages(task, objects.values(), paths["object_descriptions"])
    records = masks.get("masks")
    require(isinstance(records, list), "component_masks requires masks")
    grouped = defaultdict(list)
    for record in records:
        require(isinstance(record, dict), "invalid component mask record")
        key = (identifier(record.get("object_id"), "mask object"), identifier(record.get("frame_id"), "mask frame"))
        identifier(record.get("component_id"), "component")
        grouped[key].append(record)

    plans = []
    for object_id, obj in sorted(objects.items()):
        require(obj.get("association_status") == "geometrically_verified", f"{object_id}: isolated object is not geometrically verified")
        observations = indexed(obj.get("observations"), "frame_id", "isolated observation")
        verified = indexed(tracks[object_id].get("observations"), "frame_id", "lifting observation")
        require(observations and set(observations) <= set(verified), f"{object_id}: isolated observations are not lifting-verified frames")
        views = []
        for frame_id, observation in observations.items():
            require(frame_id in frames and frame_id in cameras, f"{object_id}: missing source camera frame {frame_id}")
            frame = frames[frame_id]
            records = grouped[object_id, frame_id]
            parents = [record for record in records if record["component_id"] == "__object__"]
            require(len(parents) == 1, f"{object_id}/{frame_id}: requires exactly one verified parent mask")
            parent = parents[0]
            require(local_path(task, observation["mask_path"]) == local_path(task, parent["mask_path"]),
                    f"{object_id}/{frame_id}: isolated observation and parent mask paths disagree")
            image_path = local_path(task, frame["image_path"])
            for record in (parent, observation, cameras[frame_id]):
                if "image_path" in record:
                    require(local_path(task, record["image_path"]) == image_path, f"{object_id}/{frame_id}: source image paths disagree")
            with Image.open(image_path) as image:
                size = image.size
            require((frame.get("width"), frame.get("height")) == size, f"{frame_id}: source dimensions disagree")
            camera = cameras[frame_id]
            if "width" in camera or "height" in camera:
                require((camera.get("width"), camera.get("height")) == size, f"{frame_id}: camera/source dimensions disagree")
            parent_mask, parent_path, digest = mask_record(task, parent, size, require_hash=resolved)
            if "mask_sha256" in observation:
                require(observation["mask_sha256"] == digest, f"{object_id}/{frame_id}: isolated observation mask hash mismatch")
            require(parent_mask.any(), f"{object_id}/{frame_id}: accepted parent mask is empty")
            parts = []
            seen_components = set()
            for record in records:
                if record["component_id"] == "__object__":
                    continue
                name = record["component_id"]
                require(name not in seen_components, f"{object_id}/{frame_id}: duplicate component track")
                seen_components.add(name)
                if "image_path" in record:
                    require(local_path(task, record["image_path"]) == image_path, f"{object_id}/{frame_id}: component/source image paths disagree")
                _, part_path, part_digest = mask_record(task, record, size, require_hash=resolved)
                group = record.get("component_group_id", record.get("source_component_id", name))
                parts.append({"component_id": name, "component_group_id": group,
                              "name": record.get("text", record.get("name", group)),
                              "observation_id": record.get("observation_id"), "identity_scope": record.get("identity_scope", "unspecified"),
                              "mask_path": part_path.relative_to(task).as_posix(), "mask_sha256": part_digest,
                              "geometry_validated": record.get("geometry_validated") is True,
                              "group_geometry_validated": record.get("group_geometry_validated"),
                              "membership_relation": record.get("membership_relation", "unspecified"),
                              "composition_allowed": record.get("composition_allowed", False), "usage": "annotation_only"})
            ys, xs = np.nonzero(parent_mask)
            views.append({"frame_id": frame_id, "image_path": image_path.relative_to(task).as_posix(), "image_sha256": sha256(image_path),
                          "parent_mask_path": parent_path.relative_to(task).as_posix(), "parent_mask_sha256": digest,
                          "parent_hash_bound_by_lifting": "mask_sha256" in parent,
                          "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                          "foreground_pixels": int(parent_mask.sum()), "components": sorted(parts, key=lambda part: part["component_id"])})
        views.sort(key=lambda view: (-view["foreground_pixels"], view["frame_id"]))
        missing = sorted({part for desc in descriptions_by_object[object_id] for part in desc.get("unobserved_components", [])})
        plans.append({"object_id": object_id, "observed_geometry": obj, "views": views, "missing_components": missing})
    return plans, {role: file_record(task, path) for role, path in paths.items()}, binding_kind


def run(args):
    import numpy as np
    from PIL import Image, ImageOps, ImageDraw

    require(type(args.max_views) is int and args.max_views > 0, "--max-views must be positive")
    task = args.task_dir.resolve()
    outputs = local_path(task, args.outputs, exists=False)
    plans, dependencies, binding_kind = load_plan(args)
    outputs.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="run-", dir=outputs.parent))
    assembled = []
    for object_index, plan in enumerate(plans):
        object_dir = stage / f"object_{object_index:04d}"
        object_dir.mkdir()
        views = []
        thumbnails = []
        for index, candidate in enumerate(plan["views"][:args.max_views]):
            source_path = local_path(task, candidate["image_path"])
            parent_path = local_path(task, candidate["parent_mask_path"])
            require(sha256(source_path) == candidate["image_sha256"] and sha256(parent_path) == candidate["parent_mask_sha256"],
                    "source image or parent mask changed during assembly")
            with Image.open(source_path) as image:
                rgba = image.convert("RGBA")
            with Image.open(parent_path) as mask:
                rgba.putalpha(mask.convert("L"))
            require(int((np.asarray(rgba)[:, :, 3] > 0).sum()) == candidate["foreground_pixels"], "parent alpha changed during assembly")
            rgba_path = object_dir / f"view_{index:03d}.png"
            rgba.save(rgba_path)
            crop = rgba.crop(candidate["bbox_xyxy"])
            crop_path = object_dir / f"crop_{index:03d}.png"
            crop.save(crop_path)
            thumb = Image.new("RGB", (320, 260), "white")
            scaled = ImageOps.contain(crop, (320, 232))
            thumb.paste(scaled, ((320 - scaled.width) // 2, 0), scaled)
            ImageDraw.Draw(thumb).text((8, 240), candidate["frame_id"], fill="black")
            thumbnails.append(thumb)
            views.append({**candidate, "rgba_path": rgba_path.relative_to(task).as_posix(), "rgba_sha256": sha256(rgba_path),
                          "crop_path": crop_path.relative_to(task).as_posix(), "crop_sha256": sha256(crop_path),
                          "component_ids": [part["component_id"] for part in candidate["components"]],
                          "coordinate_frame": "original_image", "evidence": "observed_pixels", "alpha_source": "verified_parent_mask"})
        sheet = Image.new("RGB", (320 * min(4, len(thumbnails)), 260 * ((len(thumbnails) + 3) // 4)), "white")
        for i, thumb in enumerate(thumbnails):
            sheet.paste(thumb, ((i % 4) * 320, (i // 4) * 260))
        sheet_path = object_dir / "components_contact_sheet.jpg"
        sheet.save(sheet_path)
        assembled.append({**plan, "views": views, "contact_sheet_path": sheet_path.relative_to(task).as_posix(),
                          "completion_required": True, "geometry_extent": "observed_only_incomplete"})
    for role, dependency in dependencies.items():
        require(sha256(local_path(task, dependency["path"])) == dependency["sha256"], f"{role} changed during assembly")
    manifest = stage / "assembled_object_views.json"
    qa = stage / "assembly_report.json"
    write_json(manifest, {"schema_version": "1.0", "objects": assembled,
                          "cameras_path": dependencies["cameras"]["path"]})
    write_json(qa, {"schema_version": "1.0", "status": "assembled_observed_views", "object_count": len(assembled),
                   "method": METHOD, "component_annotations_affect_alpha": False, "cross_view_pixel_pasting": False,
                   "hidden_pixels_generated": False, "complete_object_claimed": False, "source_artifacts": dependencies,
                   "component_masks_binding": binding_kind,
                   "legacy_hash_scope": "source_manifest_only; per-mask hashes recorded at assembly time" if binding_kind == "source_files" else None,
                   "lifting_report_path": dependencies["lifting_report"]["path"],
                   "objects": [{"object_id": plan["object_id"], "verified_frame_ids": [view["frame_id"] for view in plan["views"]],
                                "selected_frame_ids": [view["frame_id"] for view in result["views"]]} for plan, result in zip(plans, assembled)]})
    publish(args, {"assembled_object_views": manifest, "assembly_report": qa})
    return read_json(outputs)


def main(argv=None) -> int:
    app = parser(__doc__)
    app.add_argument("--max-views", type=int, default=8)
    args = app.parse_args(argv)
    if args.max_views < 1:
        app.error("--max-views must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
