#!/usr/bin/env python3
"""Task-local Qwen2.5-VL provider for frame grounding and object descriptions."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from world_modeling.provider_io import input_artifact, local_path, parser, publish, read_json, write_json

PROMPT_VERSION = "objects-components-v5"
INVENTORY_PROMPT_VERSION = "component-class-inventory-v4"
INVENTORY_PADDING = 0.15
SCHEMA = {
    "objects": [{
        "object_id": "bed_01", "category": "bed", "description": "Wooden bed with a white quilt and a rectangular headboard",
        "bbox_xyxy": [0, 0, 100, 100], "confidence": 0.7,
        "materials": ["wood", "fabric"],
        "components": [{"component_id": "headboard", "name": "headboard", "description": "Brown wooden rectangular panel behind the pillows",
                        "bbox_xyxy": [0, 0, 100, 50], "visibility": "visible", "confidence": 0.7}],
        "unobserved_components": ["legs hidden behind bedding"],
    }],
}


def parse_json_response(response: str):
    text = response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[-1].strip() != "```":
            raise ValueError("unterminated JSON fence")
        text = "\n".join(lines[1:-1])
    def reject_constant(value):
        raise ValueError(f"non-finite JSON number: {value}")

    return json.loads(text, parse_constant=reject_constant)


def parse_response(response: str) -> dict:
    value = parse_json_response(response)
    if isinstance(value, list):
        value = {"objects": value}
    if not isinstance(value, dict) or not isinstance(value.get("objects"), list):
        raise ValueError("response must contain an objects array")
    return value


def finite_number(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and (not isinstance(value, float) or math.isfinite(value)))


BOX_CLAMP_TOLERANCE_FRACTION = 0.1


def clamp_box(box: list, width: int, height: int, tolerance: float = BOX_CLAMP_TOLERANCE_FRACTION):
    """Bring a box that overshoots the raster back inside, or reject it.

    Vision models routinely emit a boundary-touching box a few percent wider
    than the image they were shown. Clamping that overshoot is lossless for
    every downstream consumer, whereas a box that leaves the frame by more than
    ``tolerance`` of the corresponding side -- or that lies entirely outside it
    -- is a real grounding error and stays fatal. Returns ``None`` when the box
    must be rejected.
    """
    x0, y0, x1, y1 = box
    if (x0 < -tolerance * width or y0 < -tolerance * height
            or x1 > width * (1.0 + tolerance) or y1 > height * (1.0 + tolerance)):
        return None
    clamped = [min(max(x0, 0.0), float(width)), min(max(y0, 0.0), float(height)),
               min(max(x1, 0.0), float(width)), min(max(y1, 0.0), float(height))]
    if not (clamped[0] < clamped[2] and clamped[1] < clamped[3]):
        return None
    return clamped


def validate_box(item: dict, width: int, height: int, label: str) -> list:
    raw = []
    resolved = []
    clamped_from = []
    for key in ("bbox_xyxy", "bbox_2d"):
        if key not in item:
            continue
        box = item[key]
        if not isinstance(box, list) or len(box) != 4 or not all(finite_number(v) for v in box):
            raise ValueError(f"{label}: expected four finite pixel coordinates")
        x0, y0, x1, y1 = box
        raw.append(list(box))
        if 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height:
            resolved.append([x0, y0, x1, y1])
            clamped_from.append(None)
            continue
        clamped = clamp_box([x0, y0, x1, y1], width, height)
        if clamped is None:
            raise ValueError(f"{label}: box outside {width}x{height}: {box}")
        resolved.append(clamped)
        clamped_from.append(list(box))
    if not resolved:
        raise ValueError(f"{label}: expected four finite pixel coordinates")
    if len(raw) == 2 and raw[0] != raw[1]:
        raise ValueError(f"{label}: bbox_2d conflicts with bbox_xyxy")
    if clamped_from[0] is not None:
        item["bbox_clamped_from"] = clamped_from[0]
    item["bbox_xyxy"] = list(resolved[0])
    item.pop("bbox_2d", None)
    return item["bbox_xyxy"]


def parse_categories(value: str | None) -> list[str]:
    if value is None:
        return []
    categories = [" ".join(item.casefold().split()) for item in value.split(",")]
    if any(not item for item in categories) or len(set(categories)) != len(categories):
        raise ValueError("--categories requires distinct nonempty comma-separated category names")
    return categories


DUPLICATE_IOU_THRESHOLD = 0.5


def slug_identifier(value: str, fallback: str) -> str:
    """Coerce a model-chosen label into the declared ``[a-z][a-z0-9_]{0,63}`` grammar.

    The grammar is a contract, but the label itself is cosmetic: ``"guitar case"``
    and ``guitar_case`` name the same instance. Rewriting it keeps the contract
    intact instead of discarding an otherwise valid frame, and the original
    string is preserved on the record so nothing is lost.
    """
    text = re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.casefold())).strip("_")
    if not text:
        return fallback
    if not text[0].isalpha():
        text = f"{fallback}_{text}"
    return text[:64].rstrip("_") or fallback


def box_iou(left, right) -> float:
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    intersection = (x1 - x0) * (y1 - y0)
    union = ((left[2] - left[0]) * (left[3] - left[1]) + (right[2] - right[0]) * (right[3] - right[1])
             - intersection)
    return intersection / union if union > 0 else 0.0


def unique_suffix(base: str, seen) -> str:
    index = 2
    while f"{base}_{index}" in seen:
        index += 1
    return f"{base}_{index}"


def resolve_object_identity(obj: dict, raw_id: str, seen: dict, dropped: list, rewrites: list) -> bool:
    """Give ``obj`` a unique identifier, or report that it repeats an instance.

    Generation is not stable across attempts, so one frame can carry the same
    label twice. When the two records agree on category and overlap heavily they
    are one instance and the repeat is dropped; otherwise they are separate
    instances and the later one is suffixed. Returns ``False`` when the record
    must not be kept.
    """
    candidate = slug_identifier(raw_id, "object")
    reason = None
    if candidate in seen:
        previous = seen[candidate]
        if (previous["category"] == obj["category"]
                and box_iou(previous["bbox_xyxy"], obj["bbox_xyxy"]) >= DUPLICATE_IOU_THRESHOLD):
            dropped.append({"object_id": candidate, "category": obj["category"],
                            "reason": "repeats an instance already accepted in this frame"})
            return False
        candidate = unique_suffix(candidate, seen)
        reason = "two distinct instances in one frame shared this label"
    elif candidate != raw_id:
        reason = "identifier outside [a-z][a-z0-9_]{0,63}"
    if reason:
        obj.setdefault("source_vlm_object_id", raw_id)
        rewrites.append({"kind": "object_id", "from": raw_id, "to": candidate, "reason": reason})
    obj["object_id"] = candidate
    seen[candidate] = obj
    return True


def validate_response(payload: dict, width: int, height: int, allowed_categories=None) -> dict:
    templates = {"detailed observed geometry, color and appearance", "observed appearance", "observation of mattress details"}
    seen, dropped, rewrites, kept = {}, [], [], []
    for obj in payload["objects"]:
        if not isinstance(obj, dict):
            raise ValueError("objects must contain JSON objects")
        raw_id = obj.get("object_id", "")
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise ValueError(f"invalid or duplicate object_id: {raw_id!r}")
        if not all(isinstance(obj.get(key), str) and obj[key].strip() for key in ("category", "description")):
            raise ValueError(f"{raw_id}: category and description required")
        if allowed_categories and " ".join(obj["category"].casefold().split()) not in allowed_categories:
            raise ValueError(f"{raw_id}: category {obj['category']!r} is outside requested categories {allowed_categories}; "
                             "omit unrelated objects, do not relabel them to satisfy the category list")
        if allowed_categories:
            obj["category"] = " ".join(obj["category"].casefold().split())
        if obj["description"].lower().strip() in templates:
            raise ValueError(f"{raw_id}: description is a placeholder, not visual observations")
        if not isinstance(obj.get("components"), list) or not isinstance(obj.get("materials"), list):
            raise ValueError(f"{raw_id}: components and materials must be arrays")
        validate_box(obj, width, height, raw_id)
        confidence = obj.get("confidence")
        if not finite_number(confidence) or not 0 <= confidence <= 1:
            raise ValueError(f"{raw_id}: confidence must be in [0,1]")
        if not resolve_object_identity(obj, raw_id, seen, dropped, rewrites):
            continue
        object_id = obj["object_id"]
        part_ids = set()
        for item in obj["components"]:
            if not isinstance(item, dict):
                raise ValueError(f"{object_id}: components must contain JSON objects")
            confidence = item.get("confidence")
            if not finite_number(confidence) or not 0 <= confidence <= 1:
                raise ValueError(f"{object_id}: confidence must be in [0,1]")
            validate_box(item, width, height, object_id)
            raw_component_id = item.get("component_id", "")
            component_id = (slug_identifier(raw_component_id, "component")
                            if isinstance(raw_component_id, str) and raw_component_id.strip() else "")
            if not component_id or component_id in part_ids:
                raise ValueError(f"{object_id}: invalid or duplicate component_id {raw_component_id!r}; "
                                 "use unique IDs matching [a-z][a-z0-9_]{0,63}")
            if component_id != raw_component_id:
                item.setdefault("source_vlm_component_id", raw_component_id)
                rewrites.append({"kind": "component_id", "from": raw_component_id, "to": component_id,
                                 "reason": "identifier outside [a-z][a-z0-9_]{0,63}"})
            item["component_id"] = component_id
            part_ids.add(component_id)
            if item.get("visibility") in ("partially visible", "partially_visible"):
                item["visibility"] = "partial"
            if item.get("visibility") in ("fully visible", "fully_visible", "full"):
                item["visibility"] = "visible"
            if item.get("visibility") not in ("visible", "partial"):
                raise ValueError("visibility must be 'visible' or 'partial'; hidden components belong in unobserved_components, without boxes")
            if not all(isinstance(item.get(key), str) and item[key].strip() for key in ("name", "description")):
                raise ValueError("component name and description required")
            if item["description"].lower().strip() in templates:
                raise ValueError("component description is a placeholder")
            item["parent_object_id"] = object_id
        kept.append(obj)
    payload["objects"] = kept
    if dropped:
        payload["duplicate_objects_dropped"] = dropped
    if rewrites:
        payload["identifier_rewrites"] = rewrites
    return payload


def make_prompt(width: int, height: int, categories: str | None, previous: list[dict], component_inventory=False, independent=False) -> str:
    target = f"Identify only these object categories: {categories}." if categories else "Identify movable foreground objects in the scene."
    component_instructions = (
        "For this whole-frame grounding pass, return components: [] and unobserved_components: [] for every object. "
        "A separate object-crop pass will inventory component classes. Focus on the whole physical object's box. "
        "Keep lamps, plants and nightstands as independent objects when they match the requested categories. "
    ) if component_inventory else (
        "Each component needs component_id (lowercase), name, description (actual visual appearance), "
        "bbox_xyxy (four pixel coordinates), visibility ('visible' or 'partial'), confidence (0 to 1). "
        "A component belongs physically to its parent: a bedside table, lamp, wall, or plant is NOT a bed component. "
        "Separate mattress, frame, headboard, visible legs and individual pillows. Give visible or partially visible parts boxes; "
        "list hidden parts without fabricated boxes in unobserved_components. Describe shape, material, color, pattern, damage and occlusion. "
    )
    prompt = (
        f"Ground physical object instances in this {width}x{height} image. {target}\n"
        "Return one JSON object with an objects array. Each object needs these fields: "
        "object_id (lowercase ID for this instance), category (string), description (specific observed colors, materials, shape and pattern), "
        "bbox_xyxy (four pixel coordinates), confidence (0 to 1), materials (array of material names), "
        "components (array), unobserved_components (array of hidden part names). "
        + component_instructions +
        "Never output schema descriptions or instructions as actual object descriptions. "
        "Boxes use ORIGINAL IMAGE pixel coordinates, x in [0,width], y in [0,height], xmin< xmax, ymin<ymax. "
        "Use conservative confidence values. Do not estimate numeric dimensions or mass in this stage. "
        "Find every visible instance of each requested category, including smaller and partially occluded instances. "
        "Different same-category instances need distinct IDs with spatial descriptions. "
        "A lamp placed on a table is not the table itself. Do not enclose both in one object box. "
        "Return objects: [] when no requested object is identifiable; do not invent or relabel objects. "
    )
    if independent:
        prompt += "IDs apply only to this image. Do not infer cross-view identity."
    else:
        prompt += ("Suggested IDs from previous views are hypotheses only; retain one only when visual evidence supports identity: "
                   + json.dumps(previous) + ".")
    return prompt


def ground_frame(image, categories, previous, generate, *, component_inventory, mode, frame_index):
    if mode not in ("joint", "category_scoped"):
        raise ValueError("unknown grounding mode")
    if mode == "category_scoped" and not categories:
        raise ValueError("category_scoped grounding requires explicit categories")
    scopes = [[category] for category in categories] if mode == "category_scoped" else [categories]
    objects, queries = [], []
    for scope in scopes:
        prompt = make_prompt(*image.size, ", ".join(scope) or None, [] if mode == "category_scoped" else previous,
                             component_inventory, independent=mode == "category_scoped")
        accepted, attempts = query_with_retries(image, prompt, generate,
            lambda response: validate_response(parse_response(response), *image.size,
                allowed_categories=scope if mode == "joint" else None))
        queries.append({"categories": scope, "prompt": prompt, "attempts": attempts,
                        "grounding_inference": copy.deepcopy(accepted), "excluded_objects": []})
        if accepted is None:
            return None, queries
        for obj in accepted["objects"]:
            item = copy.deepcopy(obj)
            if mode == "category_scoped":
                normalized_category = " ".join(item["category"].casefold().split())
                if normalized_category not in scope:
                    queries[-1]["excluded_objects"].append({"object": item, "reason": "outside_current_category_query"})
                    continue
                item["category"] = normalized_category
                item.setdefault("source_vlm_object_id", item["object_id"])
                item["object_id"] = f"observation_{frame_index:06d}_{len(objects):03d}"
                item["identity_scope"] = "frame_local_observation"
                for part in item["components"]:
                    part["parent_object_id"] = item["object_id"]
            objects.append(item)
    return validate_response({"objects": objects}, *image.size, allowed_categories=categories), queries


def make_inventory_prompt(category: str, width: int, height: int) -> str:
    is_bed = category.strip().casefold() == "bed"
    if is_bed:
        group_instructions = (
            "Put ALL visible pillows in one pillow group box, even if their colors differ; "
            "do not assign individual or cross-view identities. "
            "Use component_id pillow for that group, not pillow_group or numbered pillow IDs. "
        )
        relationships = "relationship (exact string 'structural_part' or 'bedding')"
        part_instructions = (
            "For a bed distinguish visible headboard, exposed frame, legs, bed skirt, quilt and pillows. "
            "Headboard, frame and legs are structural_part; pillows, quilt and skirt are bedding. "
            "Do not call a quilt a mattress. bedding is allowed only for a bed. "
            "Lamps, plants and neighboring furniture are independent nearby objects, NOT components.\n"
        )
    else:
        group_instructions = "Do not assign individual or cross-view identities. "
        relationships = "relationship (exact string 'structural_part')"
        examples = {
            "nightstand": "Visible examples can include its tabletop, drawer fronts, handles and legs. ",
            "bedside table": "Visible examples can include its tabletop, drawer fronts, handles and legs. ",
            "lamp": "Visible examples can include its shade, stem, base and switch. ",
        }.get(category.strip().casefold().replace("_", " "), "")
        part_instructions = (
            f"Include only physical structural parts of this {category}. " + examples +
            "These examples are not a checklist; report a part only when visible evidence supports it. "
            "Objects placed on or next to the target are independent nearby objects, NOT its components.\n"
        )
    return (
        f"Describe visible parts of this {category} in the {width}x{height} pixel crop.\n"
        'Return ONE JSON OBJECT with exactly these three array keys: '
        '{"components": [], "nearby_objects": [], "unobserved_components": []}.\n'
        "components: one GROUP per visible component TYPE. " + group_instructions +
        "Every component needs component_id (matching [a-z][a-z0-9_]{0,63}; use underscores, never slashes), "
        "name, description (actual observed color, material, shape, pattern, position and occlusion), "
        + relationships + ", bbox_xyxy ([left,top,right,bottom] pixel numbers), "
        "visibility (visible or partial), confidence (number 0 to 1).\n"
        + part_instructions +
        'nearby_objects: array of STRINGS naming excluded objects, e.g. ["lamp", "nightstand"], never dictionaries.\n'
        'unobserved_components: array of STRINGS naming hidden parts, without boxes. '
        "Do not invent unseen parts or use placeholder descriptions.\n"
        "If no component can be resolved in this small or occluded crop, return components: []; "
        "do not force uncertain parts or boxes.\n"
        f"All boxes use THIS crop: 0 <= left < right <= {width}, 0 <= top < bottom <= {height}."
    )


def validate_inventory(payload: dict, obj: dict, width: int, height: int) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("inventory must be a JSON object")
    result = copy.deepcopy(payload)
    for key in ("components", "nearby_objects", "unobserved_components"):
        if not isinstance(result.get(key), list):
            raise ValueError(f"{key} must be an array")
    for key in ("nearby_objects", "unobserved_components"):
        if not all(isinstance(value, str) and value.strip() for value in result[key]):
            raise ValueError(f"{key} must contain nonempty names")
    groups, types = {}, {}
    for index, raw in enumerate(payload["components"]):
        if not isinstance(raw, dict):
            raise ValueError(f"components[{index}] must be an object")
        if raw.get("relationship") not in ("structural_part", "bedding"):
            raise ValueError(f"components[{index}] {raw.get('component_id')!r}: relationship {raw.get('relationship')!r} "
                             "must be exactly 'structural_part' (e.g. headboard/frame/legs) or 'bedding' (e.g. pillow/quilt/skirt)")
        if raw["relationship"] == "bedding" and obj["category"].strip().casefold() != "bed":
            raise ValueError("bedding relationship is only valid for a bed")
        probe = {**copy.deepcopy(obj), "bbox_xyxy": [0, 0, width, height], "components": [copy.deepcopy(raw)]}
        probe.pop("bbox_2d", None)
        # The crop pass keeps the strict grammar instead of the rewrite that
        # validate_response applies: an unusable class label there means the
        # model merged two classes into one ID, and the caller can afford to
        # retry because the tolerant inventory policy keeps the parent object.
        raw_component_id = raw.get("component_id", "")
        if not isinstance(raw_component_id, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", raw_component_id):
            raise ValueError(f"components[{index}] {raw_component_id!r}: component_id must match "
                             "[a-z][a-z0-9_]{0,63}, for example 'quilt' or 'quilt_comforter'")
        part = validate_response({"objects": [probe]}, width, height)["objects"][0]["components"][0]
        name = part["name"].strip().casefold()
        category = obj["category"].strip().casefold().replace("_", " ")
        independent_names = r"\b(lamp|plant|nightstand|bedside table)\b" if category == "bed" else r"\b(lamp|plant)\b"
        if category in ("bed", "nightstand", "bedside table") and re.search(independent_names, name):
            raise ValueError("independent lamps, plants and neighboring furniture belong in nearby_objects")
        component_id = part["component_id"]
        kind = (name, part["relationship"])
        if component_id in groups and (groups[component_id]["name"].strip().casefold(), groups[component_id]["relationship"]) != kind:
            raise ValueError(f"conflicting name or relationship for component group {component_id}")
        if kind in types and types[kind] != component_id:
            raise ValueError("one component type must use one group ID, not separate instance IDs")
        types[kind] = component_id
        provenance = {"response_index": index, "component": copy.deepcopy(raw)}
        if component_id not in groups:
            part.update({"instance_mode": "all", "association_status": "frame_local_class_hypothesis",
                         "physical_identity_claimed": False,
                         "group_provenance": {"coordinate_space": "inventory_image_pixels", "entries": [provenance]}})
            groups[component_id] = part
            continue
        group = groups[component_id]
        group["bbox_xyxy"] = [min(group["bbox_xyxy"][0], part["bbox_xyxy"][0]),
                              min(group["bbox_xyxy"][1], part["bbox_xyxy"][1]),
                              max(group["bbox_xyxy"][2], part["bbox_xyxy"][2]),
                              max(group["bbox_xyxy"][3], part["bbox_xyxy"][3])]
        group["confidence"] = min(group["confidence"], part["confidence"])
        group["visibility"] = "partial" if "partial" in (group["visibility"], part["visibility"]) else "visible"
        group["group_provenance"]["entries"].append(provenance)
        group["group_provenance"]["operation"] = "same_id_name_relationship_bbox_envelope"
        descriptions = dict.fromkeys(entry["component"]["description"] for entry in group["group_provenance"]["entries"])
        group["description"] = "; ".join(descriptions)
    result["components"] = list(groups.values())
    result["association_status"] = "frame_local_class_hypotheses"
    return result


def crop_bounds(box: list, width: int, height: int) -> list[int]:
    x0, y0, x1, y1 = validate_box({"bbox_xyxy": box}, width, height, "parent crop")
    dx, dy = (x1 - x0) * INVENTORY_PADDING, (y1 - y0) * INVENTORY_PADDING
    return [max(0, math.floor(x0 - dx)), max(0, math.floor(y0 - dy)),
            min(width, math.ceil(x1 + dx)), min(height, math.ceil(y1 + dy))]


def vision_size(size) -> tuple[int, int]:
    return tuple(max(28, round(value / 28) * 28) for value in size)


def inventory_to_source(inventory: dict, crop: list, inference_size: tuple[int, int]) -> dict:
    mapped = copy.deepcopy(inventory)
    for part in mapped["components"]:
        part["bbox_xyxy"] = [value * (crop[i % 2 + 2] - crop[i % 2]) / inference_size[i % 2] + crop[i % 2]
                             for i, value in enumerate(part["bbox_xyxy"])]
    return mapped


def query_with_retries(image, prompt, generate, validate, max_attempts=2):
    attempts = []
    for _ in range(max_attempts):
        response = generate(image, prompt)
        try:
            accepted = validate(response)
            attempts.append({"prompt": prompt, "raw_response": response, "validation_error": None})
            return accepted, attempts
        except (ValueError, TypeError, KeyError) as error:
            attempts.append({"prompt": prompt, "raw_response": response, "validation_error": str(error)})
            prompt += f"\nYour previous response failed validation: {error}. Correct this and output the full JSON."
    return None, attempts


def run_component_inventory(source, obj, generate, *, stage, task_dir, frame_index, frame_id, image_path, model_parameters):
    from PIL import Image

    crop = crop_bounds(obj["bbox_xyxy"], *source.size)
    cropped = source.crop(crop)
    inference_size = vision_size(cropped.size)
    image = cropped.resize(inference_size, Image.Resampling.BICUBIC)
    stem = f"inventory_{frame_index:06d}_{obj['object_id']}"
    crop_path, inference_path = stage / f"{stem}_crop.png", stage / f"{stem}_input.png"
    cropped.save(crop_path)
    image.save(inference_path)
    prompt = make_inventory_prompt(obj["category"], *inference_size)
    accepted, attempts = query_with_retries(
        image, prompt, generate, lambda response: validate_inventory(parse_json_response(response), obj, *inference_size),
        max_attempts=3,
    )
    record = {"frame_id": frame_id, "object_id": obj["object_id"], "image_path": image_path,
              "prompt_version": INVENTORY_PROMPT_VERSION, "prompt": prompt, "attempts": attempts,
              "model_parameters": copy.deepcopy(model_parameters), "padding": INVENTORY_PADDING,
              "original_image_size": list(source.size), "crop_xyxy": crop, "crop_image_size": list(cropped.size),
              "inference_image_size": list(inference_size),
              "crop_image_path": crop_path.relative_to(task_dir).as_posix(),
              "inference_image_path": inference_path.relative_to(task_dir).as_posix(),
              "source_image_sha256": hashlib.sha256(local_path(task_dir, image_path).read_bytes()).hexdigest(),
              "crop_image_sha256": hashlib.sha256(crop_path.read_bytes()).hexdigest(),
              "inference_image_sha256": hashlib.sha256(inference_path.read_bytes()).hexdigest(),
              "inventory_to_source": {"scale_xy": [cropped.width / image.width, cropped.height / image.height],
                                      "offset_xy": crop[:2]},
              "frame_grounding_object": copy.deepcopy(obj), "accepted_structure": accepted is not None,
              "association_status": "frame_local_class_hypotheses", "physical_identity_claimed": False}
    response_path = stage / f"{stem}_response.json"
    if accepted is not None:
        record.update({"raw_inventory": copy.deepcopy(parse_json_response(attempts[-1]["raw_response"])),
                       "inventory_crop": copy.deepcopy(accepted)})
    write_json(response_path, record)
    merged = None
    if accepted is not None:
        mapped = inventory_to_source(accepted, crop, inference_size)
        merged = copy.deepcopy(obj)
        merged.update({key: copy.deepcopy(mapped[key]) for key in ("components", "nearby_objects", "unobserved_components")})
        merged["component_association_status"] = "frame_local_class_hypotheses"
        validate_response({"objects": [merged]}, *source.size)
        record["inventory_source"] = copy.deepcopy(mapped)
        write_json(response_path, record)
    if merged is None:
        raise ValueError(f"Qwen component inventory failed for {frame_id}/{obj['object_id']}; see {response_path}")
    return merged


def main() -> int:
    app = parser(__doc__)
    app.add_argument("--model", type=Path, required=True)
    app.add_argument("--categories")
    app.add_argument("--grounding-mode", choices=("joint", "category_scoped"), default="joint",
                     help="category_scoped queries each requested category independently and emits frame-local IDs")
    app.add_argument("--component-inventory-policy", choices=("required", "tolerant"), default="required",
                     help="required: a failed component inventory aborts the frame. tolerant: keep the parent object without components and record the failure.")
    app.add_argument("--component-inventory", action="store_true",
                     help="Inventory frame-local component class groups in padded source-image object crops")
    app.add_argument("--frame-stride", type=int, default=1)
    app.add_argument("--max-new-tokens", type=int, default=4096)
    args = app.parse_args()
    if args.frame_stride < 1:
        app.error("--frame-stride must be positive")
    try:
        categories = parse_categories(args.categories)
    except ValueError as error:
        app.error(str(error))
    if args.grounding_mode == "category_scoped" and not categories:
        app.error("--grounding-mode category_scoped requires --categories")
    frames = read_json(input_artifact(args, "frames_manifest"))["frames"]
    if not frames:
        raise ValueError("empty frames_manifest")
    selected = frames[::args.frame_stride]
    component_inventory_failures = []
    frame_ids = [frame["frame_id"] for frame in selected]
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError("duplicate frame IDs")

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        local_files_only=True,
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True, use_fast=False)
    model_parameters = {"model": str(args.model), "device": device,
                        "torch_dtype": "bfloat16" if device == "cuda" else "float32",
                        "max_new_tokens": args.max_new_tokens, "do_sample": False,
                        "local_files_only": True, "processor_use_fast": False,
                        "torch_version": str(torch.__version__), "component_inventory": args.component_inventory,
                        "grounding_mode": args.grounding_mode, "categories": categories}

    def generate(image, prompt):
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        return processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

    stage_parent = local_path(args.task_dir, args.outputs, exists=False).parent
    stage_parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="qwen-", dir=stage_parent))
    proposals = []
    descriptions = []
    previous = []
    for index, frame in enumerate(selected):
        image_path = local_path(args.task_dir, frame["image_path"])
        with Image.open(image_path) as opened:
            source = opened.convert("RGB")
        # Qwen's vision grid uses multiples of 28. Make that raster explicit so
        # generated pixel boxes can be mapped back to the original camera image.
        source_size = source.size
        inference_size = vision_size(source_size)
        image = source.resize(inference_size, Image.Resampling.BICUBIC)
        accepted, queries = ground_frame(image, categories, previous, generate, component_inventory=args.component_inventory,
                                        mode=args.grounding_mode, frame_index=index)
        write_json(stage / f"response_{index:06d}.json", {
            "frame_id": frame["frame_id"], "image_path": frame["image_path"], "prompt_version": PROMPT_VERSION,
            "prompt": queries[0]["prompt"], "model": str(args.model), "category_queries": queries,
            "attempts": [attempt for query in queries for attempt in query["attempts"]],
            "original_image_size": list(source_size), "inference_image_size": list(inference_size),
            "model_parameters": model_parameters, "grounding_inference": copy.deepcopy(accepted),
        })
        if accepted is None:
            raise ValueError(f"Qwen response failed validation for {frame['frame_id']}; see response_{index:06d}.json")
        for obj in accepted["objects"]:
            for item in [obj, *obj["components"]]:
                item["bbox_xyxy"] = [value * source_size[i % 2] / inference_size[i % 2]
                                     for i, value in enumerate(item["bbox_xyxy"])]
        if args.component_inventory:
            policy = getattr(args, "component_inventory_policy", "required")
            refined = []
            for obj in accepted["objects"]:
                try:
                    refined.append(run_component_inventory(
                        source, obj, generate, stage=stage, task_dir=args.task_dir.resolve(), frame_index=index,
                        frame_id=frame["frame_id"], image_path=frame["image_path"], model_parameters=model_parameters,
                    ))
                except ValueError as error:
                    # The inventory is an enrichment on top of an already grounded
                    # and described parent object. Under "tolerant" the parent is
                    # kept without components and the failure is recorded in the
                    # published report rather than hidden; "required" keeps the
                    # deployed fail-closed behaviour.
                    if policy != "tolerant":
                        raise
                    component_inventory_failures.append({"frame_id": frame["frame_id"], "object_id": obj["object_id"], "error": str(error)})
                    kept = copy.deepcopy(obj)
                    kept.setdefault("components", [])
                    kept["component_inventory_status"] = "failed_kept_parent_only"
                    refined.append(kept)
            accepted["objects"] = refined
        proposals.append({"frame_id": frame["frame_id"], "image_path": frame["image_path"], "width": source_size[0],
                          "height": source_size[1], "objects": accepted["objects"]})
        descriptions.extend({"frame_id": frame["frame_id"], "image_path": frame["image_path"], **obj} for obj in accepted["objects"])
        previous = [{"object_id": obj["object_id"], "category": obj["category"], "description": obj["description"]} for obj in accepted["objects"]]
        print(f"frame={frame['frame_id']} objects={len(accepted['objects'])}", flush=True)
    proposal_path = stage / "object_proposals.json"
    description_path = stage / "object_descriptions.json"
    write_json(proposal_path, {"schema_version": "1.0", "frames": proposals, "frame_stride": args.frame_stride,
                              "source_frame_count": len(frames),
                              "association_status": "frame_local_observations" if args.grounding_mode == "category_scoped" else "vlm_identity_hypotheses",
                              "component_inventory": args.component_inventory, "grounding_mode": args.grounding_mode})
    write_json(description_path, {"schema_version": "1.0", "objects": descriptions, "prompt_version": PROMPT_VERSION,
                                  "component_inventory": args.component_inventory, "grounding_mode": args.grounding_mode})
    if component_inventory_failures:
        prior = read_json(proposal_path)
        prior["component_inventory_policy"] = getattr(args, "component_inventory_policy", "required")
        prior["component_inventory_failures"] = component_inventory_failures
        write_json(proposal_path, prior)
    publish(args, {"object_proposals": proposal_path, "object_descriptions": description_path})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
