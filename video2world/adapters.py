"""Typed boundaries for the external pipeline providers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from video2world.config import StageConfig, render_template
from video2world.errors import ArtifactError, ConfigurationError

if TYPE_CHECKING:
    from video2world.state import ArtifactSnapshot

JSON_OUTPUT_ROLES = {
    "frames_manifest",
    "cameras",
    "depth_manifest",
    "masks_manifest",
    "captions_manifest",
    "object_clouds_manifest",
    "object_facts",
    "object_assets_manifest",
    "aligned_objects_manifest",
    "collision_manifest",
    "web_bundle_manifest",
}
PLY_OUTPUT_ROLES = {
    "point_prior",
    "scene_gaussian",
    "scene_mesh",
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


def _validate_json(path: Path, role: str) -> None:
    if not path.is_file():
        raise ArtifactError(f"{role} must be a JSON file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"{role} is not readable JSON: {path}: {exc}") from exc
    if not isinstance(value, (dict, list)):
        raise ArtifactError(f"{role} JSON root must be an object or array: {path}")


def _validate_ply(path: Path, role: str) -> None:
    if not path.is_file():
        raise ArtifactError(f"{role} must be a PLY file: {path}")
    with path.open("rb") as stream:
        header_bytes = stream.read(1024 * 1024)
    end = header_bytes.find(b"end_header")
    if not header_bytes.startswith(b"ply") or end < 0:
        raise ArtifactError(f"{role} has no valid PLY header within the first MiB: {path}")
    try:
        header = header_bytes[: end + len(b"end_header")].decode("ascii")
    except UnicodeDecodeError as exc:
        raise ArtifactError(f"{role} PLY header is not ASCII: {path}") from exc
    vertex_match = next(
        (line.split() for line in header.splitlines() if line.startswith("element vertex ")),
        None,
    )
    if not vertex_match or int(vertex_match[-1]) <= 0:
        raise ArtifactError(f"{role} PLY declares no vertices: {path}")
    properties = {
        line.split()[-1]
        for line in header.splitlines()
        if line.startswith("property ") and " list " not in line
    }
    missing_xyz = {"x", "y", "z"} - properties
    if missing_xyz:
        raise ArtifactError(f"{role} PLY is missing XYZ properties {sorted(missing_xyz)}: {path}")
    if role in {"scene_gaussian", "semantic_gaussian"}:
        missing = GAUSSIAN_PROPERTIES - properties
        if missing:
            raise ArtifactError(f"{role} is missing Gaussian properties {sorted(missing)}: {path}")
    if role == "semantic_gaussian":
        missing = {"object_id", "object_probability"} - properties
        if missing:
            raise ArtifactError(f"semantic_gaussian is missing fields {sorted(missing)}: {path}")
    if role == "scene_mesh":
        face_line = next(
            (line.split() for line in header.splitlines() if line.startswith("element face ")),
            None,
        )
        if not face_line or int(face_line[-1]) <= 0:
            raise ArtifactError(f"scene_mesh PLY declares no faces: {path}")


ADAPTERS = {
    adapter.name: adapter
    for adapter in (
        StageAdapter("holi_ingest", "Holi-Spatial", frozenset({"frames_manifest", "cameras"})),
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
