"""Adopt a deployed Web bundle into a canonical planning-manifest sidecar."""

from __future__ import annotations

import copy
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

from video2world.hashing import atomic_write_json, digest_json
from video2world.models import (
    AssetRef,
    Bounds3D,
    CoordinateSystem,
    DescriptionEvidence,
    LocalizedText,
    ProvenanceRecord,
    Relation,
    SceneLayer,
    WorldManifest,
    WorldObject,
)

WEB_MANIFEST_SCHEMA_VERSION = 1
WEB_MANIFEST_CONTRACT = "video2world-web-manifest-1.0.0"
SCENE_COMMAND_SERVICE_CONTRACT = "video2world-scene-command-service-1.0.0"
MAX_WEB_MANIFEST_BYTES = 8 * 1024 * 1024


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _json_object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _required_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _required_boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _validate_web_manifest_contract(web_manifest: dict[str, Any]) -> None:
    if web_manifest.get("schemaVersion") != WEB_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Web manifest schemaVersion must equal 1")
    if web_manifest.get("contract") != WEB_MANIFEST_CONTRACT:
        raise ValueError(f"Web manifest contract must equal {WEB_MANIFEST_CONTRACT}")
    _required_text(web_manifest.get("version"), "version")
    source = _required_mapping(web_manifest.get("sourceWorld"), "sourceWorld")
    _required_text(source.get("worldId"), "sourceWorld.worldId")
    _required_text(source.get("runId"), "sourceWorld.runId")
    _required_text(source.get("adoptionMode"), "sourceWorld.adoptionMode")


def _canonicalized_source_payload(web_manifest: dict[str, Any]) -> dict[str, Any]:
    """Remove canonical backlink/deployment fields from the adoption source digest."""

    payload = copy.deepcopy(web_manifest)
    payload.pop("version", None)
    payload.pop("sceneCommandService", None)
    source = payload.get("sourceWorld")
    if isinstance(source, dict):
        source.pop("manifestSha256", None)
    return payload


def _localized(value: Any, *, field: str) -> LocalizedText:
    if isinstance(value, dict):
        payload = {
            key: text.strip()
            for key in ("zh", "en")
            if isinstance((text := value.get(key)), str) and text.strip()
        }
        if payload:
            return LocalizedText.model_validate(payload)
    text = _required_text(value, field)
    if " / " in text:
        zh, en = (item.strip() for item in text.split(" / ", 1))
        if zh and en:
            return LocalizedText(zh=zh, en=en)
    if re.search(r"[\u3400-\u9fff]", text):
        return LocalizedText(zh=text)
    return LocalizedText(en=text)


def _description(value: Any) -> DescriptionEvidence:
    if not isinstance(value, dict):
        return DescriptionEvidence()
    payload: dict[str, Any] = {}
    for key in ("short", "detailed", "appearance", "location", "fidelity_caveat"):
        source_key = key
        if key == "fidelity_caveat" and source_key not in value:
            source_key = "fidelityCaveat"
        if value.get(source_key) is not None:
            payload[key] = _localized(value[source_key], field=f"description.{source_key}")
    for key in ("provider", "model", "confidence"):
        if value.get(key) is not None:
            payload[key] = value[key]
    frames = value.get("evidence_frames", value.get("evidenceFrames", []))
    if isinstance(frames, list):
        payload["evidence_frames"] = [str(item) for item in frames]
    return DescriptionEvidence.model_validate(payload)


def _asset_ref(value: Any, *, role: str) -> AssetRef:
    asset = _required_mapping(value, role)
    asset_id = _required_text(asset.get("id"), f"{role}.id")
    sha256 = _required_text(asset.get("sha256"), f"{role}.sha256")
    size = asset.get("size")
    if type(size) is not int or size < 1:
        raise ValueError(f"{role}.size must be a positive integer")
    parts = asset.get("parts", [])
    if not isinstance(parts, list):
        raise ValueError(f"{role}.parts must be an array")
    for index, raw_part in enumerate(parts):
        part = _required_mapping(raw_part, f"{role}.parts[{index}]")
        _required_text(part.get("url"), f"{role}.parts[{index}].url")
        part_size = part.get("size")
        if type(part_size) is not int or part_size < 0:
            raise ValueError(f"{role}.parts[{index}].size must be a non-negative integer")
        part_sha = _required_text(part.get("sha256"), f"{role}.parts[{index}].sha256")
        if not re.fullmatch(r"[0-9a-f]{64}", part_sha):
            raise ValueError(f"{role}.parts[{index}].sha256 must be lowercase SHA-256")
    if parts and sum(part["size"] for part in parts) != size:
        raise ValueError(f"{role}.parts sizes do not sum to {role}.size")
    deployed_url = asset.get("url")
    source_path = asset.get("sourcePath")
    if deployed_url is not None:
        _required_text(deployed_url, f"{role}.url")
        uri = f"web-bundle://{asset_id}"
    elif parts:
        # The declared sha256/size describe the reassembled deployment chunks, not sourcePath.
        uri = f"web-bundle://{asset_id}"
    elif source_path is not None:
        uri = _required_text(source_path, f"{role}.sourcePath")
    else:
        raise ValueError(f"{role} must provide url, parts, or sourcePath")
    media_type = None
    file_type = asset.get("fileType")
    if isinstance(file_type, str) and file_type:
        media_type = {
            "ply": "application/vnd.ply",
            "glb": "model/gltf-binary",
            "obj": "model/obj",
        }.get(file_type.casefold())
    return AssetRef(
        uri=str(uri),
        sha256=sha256,
        size_bytes=size,
        media_type=media_type,
        role=role,
        status="candidate",
        provenance={
            "web_asset_id": asset_id,
            "file_name": asset.get("fileName"),
            "format": asset.get("format"),
            "part_count": len(parts),
            "source_path": source_path,
            "source_sha256": asset.get("sourceSha256"),
            "deployed_url": deployed_url,
            "hash_scope": ("reassembled_deployed_web_asset" if parts else "declared_web_asset"),
            "validation_limitation": (
                "Adoption records declared Web hash/size but does not read asset bytes."
            ),
        },
    )


def _bounds(value: Any, *, frame_id: str, field: str) -> Bounds3D | None:
    if value is None:
        return None
    bounds = _required_mapping(value, field)
    declared_frame = bounds.get("coordinateFrame", bounds.get("frame_id"))
    if declared_frame is not None and declared_frame != frame_id:
        raise ValueError(f"{field} coordinate frame {declared_frame!r} does not match {frame_id!r}")
    minimum = bounds.get("min", bounds.get("minimum"))
    maximum = bounds.get("max", bounds.get("maximum"))
    if not isinstance(minimum, list) or not isinstance(maximum, list):
        raise ValueError(f"{field} requires min/max arrays")
    return Bounds3D(frame_id=frame_id, minimum=tuple(minimum), maximum=tuple(maximum))


def _axis_from_world_up(value: Any) -> str:
    axes = {
        (1, 0, 0): "+X",
        (-1, 0, 0): "-X",
        (0, 1, 0): "+Y",
        (0, -1, 0): "-Y",
        (0, 0, 1): "+Z",
        (0, 0, -1): "-Z",
    }
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("coordinateSystem.worldUp must be a signed unit axis")
    try:
        key = tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError("coordinateSystem.worldUp must be a signed unit axis") from exc
    has_fractional_axis = any(
        float(raw) != integer for raw, integer in zip(value, key, strict=True)
    )
    if key not in axes or has_fractional_axis:
        raise ValueError("coordinateSystem.worldUp must be a signed unit axis")
    return axes[key]


def _snake_predicate(value: Any) -> str:
    text = _required_text(value, "sceneKnowledge.relations[].predicate")
    text = re.sub(r"(?<!^)(?=[A-Z])", "_", text).replace("-", "_").replace(" ", "_")
    normalized = re.sub(r"_+", "_", text).strip("_").casefold()
    if not re.fullmatch(r"[a-z][a-z0-9_]*", normalized):
        raise ValueError(f"relation predicate cannot be normalized safely: {text!r}")
    return normalized


def _relation_evidence(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str) and value:
        return [value]
    return []


def _index_objects(items: list[Any], field: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(items):
        item = _required_mapping(raw, f"{field}[{index}]")
        item_id = _required_text(item.get("id"), f"{field}[{index}].id")
        if item_id in indexed:
            raise ValueError(f"{field} contains duplicate object id: {item_id}")
        indexed[item_id] = item
    return indexed


def adopt_web_manifest(web_manifest: dict[str, Any]) -> WorldManifest:
    """Build a deterministic, draft canonical manifest for planning and queries.

    The adapter intentionally does not promote Web render assets into canonical object
    assets. Those require their own alignment/collision/visual QA receipts. It preserves
    scene knowledge, bounds, relations, and hierarchy, which are the authoritative inputs
    used by scene-command planning.
    """

    _validate_web_manifest_contract(web_manifest)
    source = _required_mapping(web_manifest.get("sourceWorld"), "sourceWorld")
    world_id = _required_text(source.get("worldId"), "sourceWorld.worldId")
    run_id = _required_text(source.get("runId"), "sourceWorld.runId")
    production = _required_mapping(web_manifest.get("productionBuild"), "productionBuild")
    created_at = datetime.fromisoformat(
        _required_text(production.get("createdAt"), "productionBuild.createdAt").replace(
            "Z", "+00:00"
        )
    )
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise ValueError("productionBuild.createdAt must include a timezone")

    coordinate = _required_mapping(web_manifest.get("coordinateSystem"), "coordinateSystem")
    knowledge = _required_mapping(web_manifest.get("sceneKnowledge"), "sceneKnowledge")
    frame_id = _required_text(
        coordinate.get("frameId", knowledge.get("coordinateFrame", "visual_native")),
        "scene coordinate frame",
    )
    up_axis = _axis_from_world_up(coordinate.get("worldUp"))
    assets = _required_mapping(web_manifest.get("assets"), "assets")
    visual_payload = _required_mapping(assets.get("visual"), "assets.visual")
    collider_payload = assets.get("colliderStaticCarved") or assets.get("collider")
    visual = _asset_ref(visual_payload, role="scene_gaussian")
    collider = _asset_ref(collider_payload, role="scene_collider")
    semantic_alias = visual.model_copy(
        update={
            "role": "semantic_gaussian_unavailable_web_alias",
            "status": "candidate",
            "provenance": {
                **visual.provenance,
                "limitation": "Web bundle has no standalone semantic Gaussian asset",
            },
        }
    )

    knowledge_objects = knowledge.get("objects")
    interactive_objects = web_manifest.get("interactiveObjects")
    if not isinstance(knowledge_objects, list) or not isinstance(interactive_objects, list):
        raise ValueError("sceneKnowledge.objects and interactiveObjects must be arrays")
    knowledge_by_id = _index_objects(knowledge_objects, "sceneKnowledge.objects")
    interactive_by_id = _index_objects(interactive_objects, "interactiveObjects")
    object_ids = sorted(set(knowledge_by_id) | set(interactive_by_id))

    relations_by_subject: dict[str, list[Relation]] = {item_id: [] for item_id in object_ids}
    relation_items = knowledge.get("relations", [])
    if not isinstance(relation_items, list):
        raise ValueError("sceneKnowledge.relations must be an array")
    unrepresented_relation_target_ids: set[str] = set()
    for raw in relation_items:
        relation = _required_mapping(raw, "sceneKnowledge.relations[]")
        subject = _required_text(relation.get("subject"), "relation.subject")
        if subject not in relations_by_subject:
            raise ValueError(f"relation subject is absent from objects: {subject}")
        target_id = relation.get("object")
        target_label = relation.get("targetLabel")
        confidence = relation.get("confidence", 0.0)
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, int | float)
            or not math.isfinite(confidence)
        ):
            raise ValueError("relation.confidence must be a finite number")
        verified = relation.get("verified", False)
        if not isinstance(verified, bool):
            raise ValueError("relation.verified must be a boolean")
        relation_payload: dict[str, Any] = {
            "predicate": _snake_predicate(relation.get("predicate")),
            "confidence": confidence,
            "verified": verified,
            "evidence_ids": _relation_evidence(relation.get("evidence")),
        }
        if isinstance(target_id, str) and target_id in object_ids:
            relation_payload["target_object_id"] = f"{world_id}::{run_id}::{target_id}"
        elif target_label is not None:
            if isinstance(target_id, str) and target_id:
                unrepresented_relation_target_ids.add(target_id)
            relation_payload["target_label"] = _localized(
                target_label, field="relation.targetLabel"
            )
        elif isinstance(target_id, str) and target_id:
            unrepresented_relation_target_ids.add(target_id)
            relation_payload["target_label"] = LocalizedText(en=target_id)
        else:
            raise ValueError("relation requires an object or targetLabel")
        relations_by_subject[subject].append(Relation.model_validate(relation_payload))

    objects: list[WorldObject] = []
    for object_id in object_ids:
        metadata = knowledge_by_id.get(object_id, {})
        runtime = interactive_by_id.get(object_id, {})
        name_value = metadata.get("name", runtime.get("name", object_id))
        category = _required_text(
            metadata.get("category", runtime.get("category")),
            f"object {object_id} category",
        )
        aliases = metadata.get("aliases", runtime.get("aliases", []))
        if not isinstance(aliases, list) or any(not isinstance(item, str) for item in aliases):
            raise ValueError(f"object {object_id} aliases must be strings")
        parent_id = runtime.get("parentObjectId", metadata.get("parentObjectId"))
        if parent_id is not None:
            parent_id = _required_text(parent_id, f"object {object_id} parentObjectId")
            if parent_id not in object_ids:
                raise ValueError(f"object {object_id} has unknown parent {parent_id}")
        granularity = runtime.get("semanticGranularity", metadata.get("semanticGranularity"))
        if granularity is None:
            granularity = "independent_child_asset" if parent_id else "independent_root_asset"
        moves_with_parent = bool(parent_id)
        if runtime.get("movesWithParent") is not None:
            moves_with_parent = _required_boolean(
                runtime["movesWithParent"],
                f"object {object_id} movesWithParent",
            )
        independently_movable = runtime.get(
            "independentlyMovable", metadata.get("independentlyMovable", True)
        )
        independently_movable = _required_boolean(
            independently_movable,
            f"object {object_id} independentlyMovable",
        )
        bbox_value = metadata.get("bbox", runtime.get("bbox"))
        objects.append(
            WorldObject(
                id=object_id,
                source_run_id=run_id,
                scoped_id=f"{world_id}::{run_id}::{object_id}",
                name=_localized(name_value, field=f"object {object_id} name"),
                category=category,
                aliases=list(dict.fromkeys(item.strip() for item in aliases if item.strip())),
                semantic_granularity=granularity,
                parent_object_id=parent_id,
                moves_with_parent=moves_with_parent,
                independently_movable=independently_movable,
                description=_description(metadata.get("description")),
                bbox_scene=_bounds(
                    bbox_value,
                    frame_id=frame_id,
                    field=f"object {object_id} bbox",
                ),
                relations=relations_by_subject[object_id],
            )
        )

    return WorldManifest(
        manifest_status="draft",
        world_id=world_id,
        run_id=run_id,
        created_at=created_at,
        scene=SceneLayer(
            coordinate_system=CoordinateSystem(
                frame_id=frame_id,
                up_axis=up_axis,
                handedness="right",
                units="native",
            ),
            bounds=_bounds(visual_payload.get("bbox"), frame_id=frame_id, field="scene bbox"),
            visual=visual,
            collider=collider,
            semantic_visual=semantic_alias,
        ),
        objects=objects,
        provenance=ProvenanceRecord(
            source_runs=[run_id],
            config_sha256=digest_json(_canonicalized_source_payload(web_manifest)),
            notes=[
                "Deterministically adopted from a deployed Web manifest for planning.",
                "Scene asset hash/size values are declarations and remain candidate until "
                "their deployed bytes are independently verified.",
                "Object render/collider assets remain outside this sidecar until canonical "
                "QA adoption.",
                "The semantic scene asset is an explicit candidate alias because the "
                "Web bundle omits it.",
                "Coordinate handedness is assumed right-handed and units remain native "
                "because the Web contract does not carry complete canonical coordinates.",
                *(
                    [
                        "Relation-only target IDs were not fabricated as objects: "
                        + ", ".join(sorted(unrepresented_relation_target_ids))
                    ]
                    if unrepresented_relation_target_ids
                    else []
                ),
                "The adoption source digest excludes the Web deployment version, "
                "sourceWorld.manifestSha256, and sceneCommandService to avoid a canonical "
                "backlink cycle.",
            ],
        ),
    )


def adopt_web_manifest_file(source: str | Path, destination: str | Path) -> WorldManifest:
    source_path = Path(source).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    if source_path == destination_path:
        raise ValueError("Web manifest source and canonical sidecar destination must differ")
    payload = load_web_manifest_file(source_path)
    manifest = adopt_web_manifest(payload)
    atomic_write_json(
        destination_path,
        manifest.model_dump(mode="json", exclude_none=True),
    )
    return manifest


def load_web_manifest_file(path: str | Path) -> dict[str, Any]:
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file() or source_path.is_symlink():
        raise ValueError(f"Web manifest must be a regular non-symlink file: {source_path}")
    size = source_path.stat().st_size
    if size < 1 or size > MAX_WEB_MANIFEST_BYTES:
        raise ValueError(f"Web manifest size must be between 1 and {MAX_WEB_MANIFEST_BYTES} bytes")
    try:
        payload = json.loads(
            source_path.read_bytes(),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_json_object_without_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise ValueError(f"invalid Web manifest JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Web manifest root must be an object")
    return payload


def _validate_scene_command_endpoint(endpoint: str) -> str:
    value = _required_text(endpoint, "scene command endpoint")
    if len(value) > 2048 or value != endpoint or re.search(r"[\x00-\x20\\]", value):
        raise ValueError("scene command endpoint contains unsafe characters")
    parsed = urlsplit(urljoin("https://video2world.invalid/", value))
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("scene command endpoint must use HTTP(S)")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("scene command endpoint cannot contain credentials, query, or fragment")
    if not parsed.path.endswith("/v1/scene-commands/submit"):
        raise ValueError("scene command endpoint must target /v1/scene-commands/submit")
    if value.casefold().startswith("http:") and parsed.hostname not in {
        "localhost",
        "127.0.0.1",
        "::1",
    }:
        raise ValueError("HTTP scene command endpoints are restricted to localhost")
    return value


def bind_scene_command_web_manifest(
    *,
    web_manifest_source: str | Path,
    canonical_manifest_path: str | Path,
    destination: str | Path,
    endpoint: str,
    derived_version: str,
) -> tuple[dict[str, Any], str]:
    """Write a derived Web manifest bound to an adopted canonical planning sidecar."""

    source_path = Path(web_manifest_source).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    if source_path == destination_path:
        raise ValueError("bound Web manifest must be written to a new destination")
    version = _required_text(derived_version, "derived_version")
    service_endpoint = _validate_scene_command_endpoint(endpoint)
    source_payload = load_web_manifest_file(source_path)
    adopted = adopt_web_manifest(source_payload)
    from video2world.validation import load_world_manifest

    canonical = load_world_manifest(canonical_manifest_path)
    if adopted.model_dump(mode="json") != canonical.model_dump(mode="json"):
        raise ValueError(
            "canonical manifest is not the deterministic adoption of this Web manifest"
        )
    canonical_sha256 = digest_json(canonical.model_dump(mode="json"))
    bound = copy.deepcopy(source_payload)
    bound["version"] = version
    source_world = _required_mapping(bound.get("sourceWorld"), "sourceWorld")
    source_world["manifestSha256"] = canonical_sha256
    bound["sceneCommandService"] = {
        "contract": SCENE_COMMAND_SERVICE_CONTRACT,
        "endpoint": service_endpoint,
    }
    atomic_write_json(destination_path, bound)
    return bound, canonical_sha256
