"""Typed boundaries for the external pipeline providers."""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from video2world.config import StageConfig, render_template
from video2world.errors import ArtifactError, ConfigurationError

if TYPE_CHECKING:
    from video2world.state import ArtifactSnapshot

JSON_OUTPUT_ROLES = {
    "frames_manifest",
    "cameras",
    "scene_inventory",
    "occlusion_graph",
    "depth_manifest",
    "masks_manifest",
    "captions_manifest",
    "object_clouds_manifest",
    "object_facts",
    "object_assets_manifest",
    "aligned_objects_manifest",
    "collision_manifest",
    "layered_completion_plan",
    "layered_completion_report",
    "completed_object_assets_manifest",
    "clean_plate_manifest",
    "provider_receipt",
    "web_bundle_manifest",
}
PLY_OUTPUT_ROLES = {
    "point_prior",
    "scene_gaussian",
    "scene_mesh",
    "clean_scene_gaussian",
    "clean_scene_mesh",
    "semantic_gaussian",
}
GAUSSIAN_PROPERTIES = {
    "x",
    "y",
    "z",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
}
SEMANTIC_JSON_ROLES = {
    "scene_inventory",
    "occlusion_graph",
    "layered_completion_plan",
    "layered_completion_report",
    "completed_object_assets_manifest",
    "clean_plate_manifest",
    "provider_receipt",
}
CLEAN_SCENE_PLY_ROLES = {"clean_scene_gaussian", "clean_scene_mesh"}
_CLEAN_PLATE_COLLECTION_KEYS = {
    "frames",
    "frame_records",
    "clean_plates",
    "output_frames",
    "outputs",
}
_PLY_SCALAR_FORMATS = {
    "char": "b",
    "int8": "b",
    "uchar": "B",
    "uint8": "B",
    "short": "h",
    "int16": "h",
    "ushort": "H",
    "uint16": "H",
    "int": "i",
    "int32": "i",
    "uint": "I",
    "uint32": "I",
    "float": "f",
    "float32": "f",
    "double": "d",
    "float64": "d",
}


@dataclass(frozen=True)
class _PlyProperty:
    name: str
    scalar_type: str | None = None
    list_count_type: str | None = None
    list_item_type: str | None = None


@dataclass(frozen=True)
class _PlyElement:
    name: str
    count: int
    properties: tuple[_PlyProperty, ...]


@dataclass(frozen=True)
class StageAdapter:
    name: str
    provider: str
    required_outputs: frozenset[str]

    def validate(self, stage_id: str, config: StageConfig) -> None:
        missing = sorted(self.required_outputs - config.outputs.keys())
        if missing:
            raise ConfigurationError(
                f"stage {stage_id!r} adapter {self.name!r} is missing output roles: {missing}"
            )

    def render_command(self, config: StageConfig, context: dict[str, str]) -> list[str] | None:
        if config.command is None:
            return None
        command = [render_template(argument, context) for argument in config.command]
        if any("\x00" in argument for argument in command):
            raise ConfigurationError("command arguments cannot contain NUL bytes")
        return command

    def validate_outputs(self, outputs: dict[str, ArtifactSnapshot]) -> None:
        for role, artifact in outputs.items():
            path = Path(artifact.path)
            if role in JSON_OUTPUT_ROLES:
                _validate_json(path, role)
            elif role in PLY_OUTPUT_ROLES:
                _validate_ply(path, role)
        if "world_manifest" in outputs:
            from video2world.validation import validate_world_manifest

            result = validate_world_manifest(outputs["world_manifest"].path)
            if not result["valid"] or result["manifest_status"] != "validated":
                raise ArtifactError(
                    "world_manifest must be validated and pass local asset hash verification: "
                    f"{result['issues']}"
                )
        if "layered_completion_report" in outputs:
            if "provider_receipt" not in outputs:
                raise ArtifactError(
                    "layered_completion_report requires provider_receipt so the report can be "
                    "bound to the executed completion plan"
                )
            _validate_layered_completion_lineage(outputs)


def _validate_layered_completion_lineage(outputs: dict[str, ArtifactSnapshot]) -> None:
    from video2world.completion import (
        LayeredCompletionExecutionReport,
        load_layered_completion_plan,
    )

    report_path = Path(outputs["layered_completion_report"].path)
    receipt_path = Path(outputs["provider_receipt"].path)
    try:
        report = LayeredCompletionExecutionReport.model_validate_json(
            report_path.read_text(encoding="utf-8")
        )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(receipt, dict):
            raise ValueError("provider receipt root must be an object")
        if receipt.get("kind") != "video2world.provider_execution_receipt":
            raise ValueError("layered completion requires a provider execution receipt")
        if receipt.get("status") != "completed":
            raise ValueError("layered completion provider receipt is not completed")
        receipt_outputs = receipt.get("outputs")
        if not isinstance(receipt_outputs, dict):
            raise ValueError("provider receipt has no output snapshots")
        report_output_snapshot = receipt_outputs.get("layered_completion_report")
        if not isinstance(report_output_snapshot, dict):
            raise ValueError("provider receipt has no layered_completion_report output")
        _require_matching_receipt_snapshot(
            report_output_snapshot,
            outputs["layered_completion_report"],
            context="layered_completion_report",
        )
        inputs = receipt.get("inputs")
        if not isinstance(inputs, dict):
            raise ValueError("provider receipt has no input snapshots")
        plan_snapshot = inputs.get("layered_completion_plan")
        if not isinstance(plan_snapshot, dict):
            raise ValueError("provider receipt has no layered_completion_plan input")
        if plan_snapshot.get("sha256") != report.completion_plan_sha256:
            raise ValueError("completion report does not bind the executed plan SHA-256")
        plan_path = plan_snapshot.get("path")
        if not isinstance(plan_path, str) or not plan_path:
            raise ValueError("provider receipt has no completion plan path")
        plan = load_layered_completion_plan(plan_path)
        if plan.scene_id != report.scene_id or plan.run_id != report.run_id:
            raise ValueError("completion report scene/run differs from the executed plan")
        if len(plan.rounds) != len(report.rounds):
            raise ValueError("completion report round count differs from the executed plan")
        for planned, completed in zip(plan.rounds, report.rounds, strict=True):
            if (
                planned.index != completed.index
                or planned.kind != completed.kind
                or planned.target_ids != completed.target_ids
            ):
                raise ValueError(
                    f"completion report round {completed.index} differs from the executed plan"
                )
        planned_targets = [target for round_item in plan.rounds for target in round_item.target_ids]
        if report.planned_target_ids != planned_targets:
            raise ValueError("completion report planned targets differ from the executed plan")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ArtifactError(
            "layered completion plan/receipt lineage validation failed: "
            f"{report_path}: {exc}"
        ) from exc


def _require_matching_receipt_snapshot(
    receipt_snapshot: dict[str, object],
    artifact: ArtifactSnapshot,
    *,
    context: str,
) -> None:
    if (
        receipt_snapshot.get("kind") != artifact.kind
        or receipt_snapshot.get("sha256") != artifact.sha256
        or receipt_snapshot.get("size_bytes") != artifact.size_bytes
        or receipt_snapshot.get("file_count") != artifact.file_count
    ):
        raise ValueError(f"provider receipt {context} output snapshot does not match")
    snapshot_path = receipt_snapshot.get("path")
    if not isinstance(snapshot_path, str) or not snapshot_path:
        raise ValueError(f"provider receipt {context} output path is missing")
    if Path(snapshot_path).expanduser().resolve() != Path(artifact.path).expanduser().resolve():
        raise ValueError(f"provider receipt {context} output path does not match")


def _validate_json(path: Path, role: str) -> None:
    if not path.is_file():
        raise ArtifactError(f"{role} must be a JSON file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"{role} is not readable JSON: {path}: {exc}") from exc
    if not isinstance(value, (dict, list)):
        raise ArtifactError(f"{role} JSON root must be an object or array: {path}")
    if role in SEMANTIC_JSON_ROLES:
        _validate_semantic_json(value, path, role)


def _validate_semantic_json(value: dict[str, Any] | list[Any], path: Path, role: str) -> None:
    if not isinstance(value, dict) or not value:
        raise ArtifactError(f"{role} must be a non-empty JSON object: {path}")
    try:
        if role == "scene_inventory":
            from video2world.completion import SceneInventory

            SceneInventory.model_validate(value)
        elif role == "occlusion_graph":
            from video2world.completion import OcclusionGraph

            OcclusionGraph.model_validate(value)
        elif role == "layered_completion_plan":
            from video2world.completion import LayeredCompletionPlan

            LayeredCompletionPlan.model_validate(value)
        elif role == "completed_object_assets_manifest":
            from video2world.completion import CompletedObjectAssetsManifest

            CompletedObjectAssetsManifest.model_validate(value)
        elif role == "clean_plate_manifest":
            _require_json_collection(
                value,
                _CLEAN_PLATE_COLLECTION_KEYS,
                path=path,
                role=role,
                record_id_keys={"frame_id", "id", "path", "uri"},
            )
        elif role == "layered_completion_report":
            from video2world.completion import LayeredCompletionExecutionReport

            LayeredCompletionExecutionReport.model_validate(value)
        elif role == "provider_receipt":
            _validate_provider_receipt(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"{role} failed semantic validation: {path}: {exc}") from exc


def _validate_provider_receipt(value: dict[str, Any]) -> None:
    if value.get("kind") != "video2world.provider_execution_receipt":
        raise ValueError("provider_receipt kind must be video2world.provider_execution_receipt")
    if value.get("status") != "completed":
        raise ValueError("provider_receipt status must be completed")
    for key in ("provider_id", "provider_stage_id", "provider"):
        if not isinstance(value.get(key), str) or not value[key]:
            raise ValueError(f"provider_receipt requires non-empty {key}")
    for key in ("inputs", "outputs"):
        snapshots = value.get(key)
        if not isinstance(snapshots, dict):
            raise ValueError(f"provider_receipt {key} must be an object")
        for role, snapshot in snapshots.items():
            if not isinstance(role, str) or not role:
                raise ValueError(f"provider_receipt {key} role names must be non-empty")
            if not isinstance(snapshot, dict):
                raise ValueError(f"provider_receipt {key}.{role} must be an object")
            _validate_provider_receipt_snapshot(snapshot, context=f"{key}.{role}")


def _validate_provider_receipt_snapshot(
    snapshot: dict[str, Any],
    *,
    context: str,
) -> None:
    if not isinstance(snapshot.get("path"), str) or not snapshot["path"]:
        raise ValueError(f"provider_receipt {context}.path must be non-empty")
    if snapshot.get("kind") not in {"file", "directory"}:
        raise ValueError(f"provider_receipt {context}.kind must be file or directory")
    sha256 = snapshot.get("sha256")
    if not (
        isinstance(sha256, str)
        and len(sha256) == 64
        and all(character in "0123456789abcdef" for character in sha256)
    ):
        raise ValueError(f"provider_receipt {context}.sha256 must be lowercase SHA-256")
    if not isinstance(snapshot.get("size_bytes"), int) or snapshot["size_bytes"] <= 0:
        raise ValueError(f"provider_receipt {context}.size_bytes must be positive")
    if not isinstance(snapshot.get("file_count"), int) or snapshot["file_count"] <= 0:
        raise ValueError(f"provider_receipt {context}.file_count must be positive")


def _require_json_collection(
    value: dict[str, Any],
    allowed_keys: set[str],
    *,
    path: Path,
    role: str,
    record_id_keys: set[str],
) -> None:
    present = sorted(allowed_keys.intersection(value))
    if not present:
        raise ValueError(f"{role} requires one of these payload fields: {sorted(allowed_keys)}")
    collections = [value[key] for key in present if isinstance(value[key], (dict, list))]
    if not collections:
        raise ValueError(f"{role} payload field must be an object or array: {path}")
    if not any(collection for collection in collections):
        raise ValueError(f"{role} payload collection cannot be empty: {path}")
    for collection in collections:
        if isinstance(collection, dict):
            valid = bool(collection) and all(
                isinstance(record_id, str)
                and bool(record_id.strip())
                and isinstance(record, dict)
                and bool(record)
                for record_id, record in collection.items()
            )
        else:
            valid = bool(collection) and all(
                isinstance(record, dict)
                and bool(record)
                and any(
                    isinstance(record.get(key), str) and bool(record[key].strip())
                    for key in record_id_keys
                )
                for record in collection
            )
        if valid:
            return
    raise ValueError(f"{role} payload requires non-empty identified records: {path}")


def _parse_ply_header(
    header: str,
    path: Path,
    role: str,
) -> tuple[str, list[_PlyElement]]:
    lines = header.splitlines()
    if not lines or lines[0] != "ply":
        raise ArtifactError(f"{role} does not start with an exact PLY magic line: {path}")
    format_name: str | None = None
    elements: list[tuple[str, int, list[_PlyProperty]]] = []
    for line in lines[1:]:
        fields = line.split()
        if not fields or fields[0] in {"comment", "obj_info", "end_header"}:
            continue
        if fields[0] == "format":
            if len(fields) != 3 or fields[2] != "1.0":
                raise ArtifactError(f"{role} has an unsupported PLY format declaration: {path}")
            format_name = fields[1]
            continue
        if fields[0] == "element":
            if len(fields) != 3:
                raise ArtifactError(f"{role} has a malformed PLY element declaration: {path}")
            try:
                count = int(fields[2])
            except ValueError as exc:
                raise ArtifactError(f"{role} has a non-integer PLY element count: {path}") from exc
            if count < 0:
                raise ArtifactError(f"{role} has a negative PLY element count: {path}")
            elements.append((fields[1], count, []))
            continue
        if fields[0] != "property":
            continue
        if not elements:
            raise ArtifactError(f"{role} declares a PLY property before an element: {path}")
        if len(fields) == 3:
            scalar_type, name = fields[1:]
            if scalar_type not in _PLY_SCALAR_FORMATS:
                raise ArtifactError(
                    f"{role} uses unsupported PLY scalar type {scalar_type!r}: {path}"
                )
            elements[-1][2].append(_PlyProperty(name=name, scalar_type=scalar_type))
        elif len(fields) == 5 and fields[1] == "list":
            count_type, item_type, name = fields[2:]
            if count_type not in _PLY_SCALAR_FORMATS or item_type not in _PLY_SCALAR_FORMATS:
                raise ArtifactError(f"{role} uses unsupported PLY list types: {path}")
            elements[-1][2].append(
                _PlyProperty(
                    name=name,
                    list_count_type=count_type,
                    list_item_type=item_type,
                )
            )
        else:
            raise ArtifactError(f"{role} has a malformed PLY property declaration: {path}")
    if format_name not in {"ascii", "binary_little_endian", "binary_big_endian"}:
        raise ArtifactError(f"{role} has no supported PLY format declaration: {path}")
    parsed = [
        _PlyElement(name=name, count=count, properties=tuple(properties))
        for name, count, properties in elements
    ]
    return format_name, parsed


def _parse_ascii_scalar(token: str, scalar_type: str, path: Path, role: str) -> int | float:
    try:
        if _PLY_SCALAR_FORMATS[scalar_type].lower() in {"f", "d"}:
            value: int | float = float(token)
            if not math.isfinite(value):
                raise ValueError("non-finite value")
            return value
        return int(token)
    except ValueError as exc:
        raise ArtifactError(f"{role} has invalid ASCII PLY payload data: {path}") from exc


def _validate_ascii_ply_payload(
    path: Path,
    role: str,
    payload_offset: int,
    elements: list[_PlyElement],
) -> None:
    with path.open("rb") as stream:
        stream.seek(payload_offset)
        for element in elements:
            for _ in range(element.count):
                line = stream.readline()
                while line and not line.strip():
                    line = stream.readline()
                if not line:
                    raise ArtifactError(
                        f"{role} PLY payload is truncated before all declared records: {path}"
                    )
                try:
                    tokens = line.decode("ascii").split()
                except UnicodeDecodeError as exc:
                    raise ArtifactError(f"{role} ASCII PLY payload is not ASCII: {path}") from exc
                cursor = 0
                for property_item in element.properties:
                    if property_item.scalar_type is not None:
                        if cursor >= len(tokens):
                            raise ArtifactError(f"{role} ASCII PLY record is truncated: {path}")
                        _parse_ascii_scalar(tokens[cursor], property_item.scalar_type, path, role)
                        cursor += 1
                        continue
                    assert property_item.list_count_type is not None
                    assert property_item.list_item_type is not None
                    if cursor >= len(tokens):
                        raise ArtifactError(f"{role} ASCII PLY list record is truncated: {path}")
                    count_value = _parse_ascii_scalar(
                        tokens[cursor], property_item.list_count_type, path, role
                    )
                    if not isinstance(count_value, int) or count_value < 0:
                        raise ArtifactError(f"{role} ASCII PLY has an invalid list count: {path}")
                    cursor += 1
                    if cursor + count_value > len(tokens):
                        raise ArtifactError(f"{role} ASCII PLY list record is truncated: {path}")
                    for token in tokens[cursor : cursor + count_value]:
                        _parse_ascii_scalar(token, property_item.list_item_type, path, role)
                    if element.name == "face" and count_value < 3:
                        raise ArtifactError(f"{role} ASCII PLY declares a degenerate face: {path}")
                    cursor += count_value
                if cursor != len(tokens):
                    raise ArtifactError(f"{role} ASCII PLY record has extra fields: {path}")


def _validate_binary_ply_payload(
    path: Path,
    role: str,
    payload_offset: int,
    elements: list[_PlyElement],
    *,
    byte_order: str,
) -> None:
    file_size = path.stat().st_size
    minimum_size = payload_offset
    for element in elements:
        record_minimum = sum(
            struct.calcsize(_PLY_SCALAR_FORMATS[property_item.scalar_type])
            if property_item.scalar_type is not None
            else struct.calcsize(_PLY_SCALAR_FORMATS[property_item.list_count_type])
            for property_item in element.properties
        )
        minimum_size += element.count * record_minimum
    if file_size < minimum_size:
        raise ArtifactError(f"{role} binary PLY payload is truncated: {path}")

    with path.open("rb") as stream:
        stream.seek(payload_offset)
        for element in elements:
            for _ in range(element.count):
                for property_item in element.properties:
                    scalar_type = property_item.scalar_type or property_item.list_count_type
                    assert scalar_type is not None
                    scalar_format = byte_order + _PLY_SCALAR_FORMATS[scalar_type]
                    scalar_size = struct.calcsize(scalar_format)
                    raw_value = stream.read(scalar_size)
                    if len(raw_value) != scalar_size:
                        raise ArtifactError(f"{role} binary PLY payload is truncated: {path}")
                    value = struct.unpack(scalar_format, raw_value)[0]
                    if property_item.scalar_type is not None:
                        if isinstance(value, float) and not math.isfinite(value):
                            raise ArtifactError(
                                f"{role} binary PLY contains a non-finite scalar: {path}"
                            )
                        continue
                    if not isinstance(value, int) or value < 0:
                        raise ArtifactError(f"{role} binary PLY has an invalid list count: {path}")
                    if element.name == "face" and value < 3:
                        raise ArtifactError(f"{role} binary PLY declares a degenerate face: {path}")
                    assert property_item.list_item_type is not None
                    item_size = struct.calcsize(_PLY_SCALAR_FORMATS[property_item.list_item_type])
                    byte_count = value * item_size
                    if byte_count > file_size - stream.tell():
                        raise ArtifactError(f"{role} binary PLY list payload is truncated: {path}")
                    stream.seek(byte_count, 1)


def _validate_ply(path: Path, role: str) -> None:
    if not path.is_file():
        raise ArtifactError(f"{role} must be a PLY file: {path}")
    with path.open("rb") as stream:
        header_bytes = stream.read(1024 * 1024)
    end = header_bytes.find(b"end_header")
    if not header_bytes.startswith(b"ply") or end < 0:
        raise ArtifactError(f"{role} has no valid PLY header within the first MiB: {path}")
    marker_end = end + len(b"end_header")
    if header_bytes[marker_end : marker_end + 2] == b"\r\n":
        payload_offset = marker_end + 2
    elif header_bytes[marker_end : marker_end + 1] in {b"\n", b"\r"}:
        payload_offset = marker_end + 1
    else:
        raise ArtifactError(f"{role} PLY end_header must end with a newline: {path}")
    try:
        header = header_bytes[:marker_end].decode("ascii")
    except UnicodeDecodeError as exc:
        raise ArtifactError(f"{role} PLY header is not ASCII: {path}") from exc
    format_name, elements = _parse_ply_header(header, path, role)
    vertex_element = next((element for element in elements if element.name == "vertex"), None)
    if vertex_element is None or vertex_element.count <= 0:
        raise ArtifactError(f"{role} PLY declares no vertices: {path}")
    properties = {
        property_item.name
        for property_item in vertex_element.properties
        if property_item.scalar_type is not None
    }
    missing_xyz = {"x", "y", "z"} - properties
    if missing_xyz:
        raise ArtifactError(f"{role} PLY is missing XYZ properties {sorted(missing_xyz)}: {path}")
    if role in {"scene_gaussian", "clean_scene_gaussian", "semantic_gaussian"}:
        missing = GAUSSIAN_PROPERTIES - properties
        if missing:
            raise ArtifactError(f"{role} is missing Gaussian properties {sorted(missing)}: {path}")
    if role == "semantic_gaussian":
        missing = {"object_id", "object_probability"} - properties
        if missing:
            raise ArtifactError(f"semantic_gaussian is missing fields {sorted(missing)}: {path}")
    if role in {"scene_mesh", "clean_scene_mesh"}:
        face_element = next((element for element in elements if element.name == "face"), None)
        if face_element is None or face_element.count <= 0:
            raise ArtifactError(f"{role} PLY declares no faces: {path}")
        face_indices = next(
            (
                property_item
                for property_item in face_element.properties
                if property_item.name in {"vertex_index", "vertex_indices"}
                and property_item.list_count_type is not None
                and property_item.list_item_type is not None
            ),
            None,
        )
        if face_indices is None:
            raise ArtifactError(f"{role} PLY faces require a vertex_indices list: {path}")
    if role in CLEAN_SCENE_PLY_ROLES:
        if format_name == "ascii":
            _validate_ascii_ply_payload(path, role, payload_offset, elements)
        else:
            _validate_binary_ply_payload(
                path,
                role,
                payload_offset,
                elements,
                byte_order="<" if format_name == "binary_little_endian" else ">",
            )


ADAPTERS = {
    adapter.name: adapter
    for adapter in (
        StageAdapter("holi_ingest", "Holi-Spatial", frozenset({"frames_manifest", "cameras"})),
        StageAdapter(
            "scene_inventory",
            "Scene inventory + geometry-verified occlusion",
            frozenset({"scene_inventory", "occlusion_graph"}),
        ),
        StageAdapter("holi_da3", "Holi-Spatial/DA3", frozenset({"depth_manifest", "point_prior"})),
        StageAdapter("holi_pgsr", "Holi-Spatial/PGSR", frozenset({"scene_gaussian", "scene_mesh"})),
        StageAdapter(
            "holi_sam3",
            "Holi-Spatial/SAM3",
            frozenset({"masks_manifest", "captions_manifest"}),
        ),
        StageAdapter(
            "video2mesh_fusion",
            "Video2Mesh",
            frozenset({"object_clouds_manifest", "semantic_gaussian"}),
        ),
        StageAdapter("scene_cognition", "Video2World", frozenset({"object_facts"})),
        StageAdapter(
            "layered_completion_plan",
            "Video2World",
            frozenset({"layered_completion_plan"}),
        ),
        StageAdapter(
            "layered_completion",
            "Video2World layered completion providers",
            frozenset(
                {
                    "completed_object_assets_manifest",
                    "clean_scene_gaussian",
                    "clean_scene_mesh",
                    "clean_plate_manifest",
                    "layered_completion_report",
                }
            ),
        ),
        StageAdapter(
            "embodiedgen_trellis",
            "EmbodiedGen V2/TRELLIS",
            frozenset({"object_assets_manifest"}),
        ),
        StageAdapter(
            "object_placement",
            "Video2World",
            frozenset({"aligned_objects_manifest", "collision_manifest"}),
        ),
        StageAdapter("world_bundle", "Video2World", frozenset({"world_manifest"})),
        StageAdapter("web_bundle", "Video2World Web", frozenset({"web_bundle_manifest"})),
        StageAdapter("command", "Custom argv command", frozenset()),
    )
}


def get_adapter(name: str) -> StageAdapter:
    try:
        return ADAPTERS[name]
    except KeyError as exc:
        raise ConfigurationError(
            f"unknown adapter {name!r}; available adapters: {sorted(ADAPTERS)}"
        ) from exc
