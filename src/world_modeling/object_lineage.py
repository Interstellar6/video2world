"""Bind geometric object identities to unchanged, frame-local source descriptions."""

from __future__ import annotations

import copy
import hashlib
import re
from pathlib import Path

from .provider_io import local_path, read_json


class ObjectLineageError(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise ObjectLineageError(message)


def _identifier(value, label):
    _require(isinstance(value, str) and bool(value) and value.strip() == value,
             f"lineage requires a nonempty {label}")
    return value


def lineage_metadata(obj: dict) -> dict:
    """Copy validated lineage without rewriting raw source identities."""
    keys = ("source_observations", "source_descriptions")
    present = [key in obj for key in keys]
    _require(present[0] == present[1], "lineage requires both source_observations and source_descriptions")
    return {key: copy.deepcopy(obj[key]) for key in keys} if present[0] else {}


class _Resolver:
    def __init__(self, task_dir, descriptions_path):
        self.task = Path(task_dir).resolve()
        self.descriptions_path = self.path(descriptions_path) if descriptions_path is not None else None
        self.hashes = {}
        self.documents = {}

    def path(self, value):
        _require(isinstance(value, (str, Path)) and bool(str(value)), "lineage requires a file path")
        path = local_path(self.task, value)
        _require(path.is_file(), "lineage source is not a regular file")
        return path

    def bound_file(self, value, digest, label):
        _require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
                 f"lineage {label} requires SHA256")
        path = self.path(value)
        if path not in self.hashes:
            hasher = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(block)
            self.hashes[path] = hasher.hexdigest()
        _require(self.hashes[path] == digest, f"lineage {label} SHA256 mismatch")
        return path

    def descriptions(self, path):
        if path not in self.documents:
            records = read_json(path).get("objects")
            _require(isinstance(records, list) and all(isinstance(record, dict) for record in records),
                     "lineage source descriptions require an objects list")
            self.documents[path] = records
        return self.documents[path]

    def resolve(self, obj):
        object_id = _identifier(obj.get("object_id"), "object_id")
        metadata = lineage_metadata(obj)
        if not metadata:
            records = self.descriptions(self.descriptions_path) if self.descriptions_path is not None else []
            return [record for record in records if record.get("object_id") == object_id], []

        source = metadata["source_descriptions"]
        observations = metadata["source_observations"]
        _require(isinstance(source, dict), "lineage source_descriptions must be a file binding")
        _require(isinstance(observations, list) and bool(observations), "lineage source_observations must be nonempty")
        path = self.bound_file(source.get("path"), source.get("sha256"), "descriptions")
        if self.descriptions_path is not None:
            _require(path == self.descriptions_path, "lineage descriptions path differs from current input")
        records = self.descriptions(path)
        index = {}
        for record in records:
            key = (_identifier(record.get("frame_id"), "description frame_id"),
                   _identifier(record.get("object_id"), "description object_id"))
            _require(key not in index, "lineage source descriptions duplicate frame/object key")
            index[key] = record

        accepted_frames = None
        if "observations" in obj:
            accepted = obj["observations"]
            _require(isinstance(accepted, list), "lineage accepted observations must be a list")
            accepted_frames = set()
            for observation in accepted:
                _require(isinstance(observation, dict), "lineage invalid accepted observation")
                frame_id = _identifier(observation.get("frame_id"), "accepted frame_id")
                _require(frame_id not in accepted_frames, "lineage duplicate accepted frame")
                accepted_frames.add(frame_id)

        resolved, claims, seen = [], [], set()
        for observation in observations:
            _require(isinstance(observation, dict), "lineage invalid source observation")
            key = (_identifier(observation.get("frame_id"), "source frame_id"),
                   _identifier(observation.get("source_object_id"), "source_object_id"))
            _require(key not in seen, "lineage duplicate source frame/object key")
            seen.add(key)
            if accepted_frames is not None:
                _require(key[0] in accepted_frames, "lineage source frame is outside accepted observations")
            _require(key in index, "lineage source description is missing")
            image = self.bound_file(observation.get("source_image_path"), observation.get("source_image_sha256"), "image")
            self.bound_file(observation.get("source_mask_path"), observation.get("source_mask_sha256"), "mask")
            record = index[key]
            _require(self.path(record.get("image_path")) == image, "lineage description image path differs from source observation")
            if "image_sha256" in record:
                _require(record["image_sha256"] == observation["source_image_sha256"], "lineage description image SHA256 mismatch")
            resolved.append(record)
            # Neither a copied description file nor a changed mask grants a
            # second physical object ownership of the same source observation.
            claims.extend((("description", source["sha256"], *key),
                           ("image", observation["source_image_sha256"], *key)))
        return resolved, claims


def resolve_object_descriptions(task_dir: Path, obj: dict, descriptions_path: Path | None = None) -> list[dict]:
    """Return raw description records, using explicit lineage when declared."""
    _require(isinstance(obj, dict), "lineage object must be a record")
    records, _ = _Resolver(task_dir, descriptions_path).resolve(obj)
    return records


def validate_object_lineages(task_dir: Path, objects, descriptions_path: Path | None = None) -> dict[str, list[dict]]:
    """Validate one object set, including cross-object observation ownership."""
    resolver, result, owners = _Resolver(task_dir, descriptions_path), {}, {}
    for obj in objects:
        _require(isinstance(obj, dict), "lineage object must be a record")
        object_id = _identifier(obj.get("object_id"), "object_id")
        _require(object_id not in result, "lineage duplicate physical object_id")
        descriptions, claims = resolver.resolve(obj)
        for claim in claims:
            _require(claim not in owners or owners[claim] == object_id,
                     "lineage source observation claimed by multiple physical objects")
            owners[claim] = object_id
        result[object_id] = descriptions
    return result
