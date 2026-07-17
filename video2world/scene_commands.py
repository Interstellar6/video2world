"""Typed, deterministic planning for natural-language scene commands.

The module consumes structured intent JSON produced by a UI, a rule-based parser, or a
future LLM. Planning is deliberately side-effect free: mutating commands produce preview
operations with optimistic locking, confirmation, provenance, and rollback requirements.
"""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import Field, JsonValue, model_validator

from video2world.hashing import atomic_write_json, digest_json
from video2world.models import (
    Bounds3D,
    InteractionPolicy,
    LocalizedText,
    OrientedBounds3D,
    Relation,
    SceneTransform,
    Sha256,
    StrictModel,
    WorldManifest,
    WorldObject,
)
from video2world.query import query_world

CommandKind: TypeAlias = Literal[
    "query_location",
    "query_description",
    "add_object",
    "delete_object",
    "update_properties",
    "split_object",
    "merge_objects",
    "reparent_object",
]
MutatingCommandKind: TypeAlias = Literal[
    "add_object",
    "delete_object",
    "update_properties",
    "split_object",
    "merge_objects",
    "reparent_object",
]
SemanticGranularity: TypeAlias = Literal[
    "inherit",
    "independent_root_asset",
    "independent_child_asset",
    "merged_component",
]
StageName: TypeAlias = Literal[
    "inventory",
    "sam3",
    "fusion",
    "cognition",
    "completion_plan",
    "layered_completion",
    "placement",
    "bundle",
    "web",
]

_STAGE_ORDER: tuple[StageName, ...] = (
    "inventory",
    "sam3",
    "fusion",
    "cognition",
    "completion_plan",
    "layered_completion",
    "placement",
    "bundle",
    "web",
)
_RECONSTRUCTION_STAGES: tuple[StageName, ...] = _STAGE_ORDER
_HIERARCHY_STAGES: tuple[StageName, ...] = ("cognition", "placement", "bundle", "web")
_METADATA_STAGES: tuple[StageName, ...] = ("cognition", "bundle", "web")


class TargetSelector(StrictModel):
    """A stable selector; exactly one matching strategy must be supplied."""

    scoped_id: str | None = Field(default=None, min_length=1)
    object_id: str | None = Field(
        default=None,
        min_length=1,
        pattern=r"^[A-Za-z0-9_.-]+$",
    )
    label: str | None = Field(default=None, min_length=1)
    ordinal: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def one_strategy(self) -> TargetSelector:
        supplied = sum(value is not None for value in (self.scoped_id, self.object_id, self.label))
        if supplied != 1:
            raise ValueError(
                "target selector requires exactly one of scoped_id, object_id, or label"
            )
        if self.ordinal is not None and self.label is None:
            raise ValueError("ordinal is only valid with a label selector")
        return self


class ProposedObject(StrictModel):
    """Semantic object request before segmentation, reconstruction, and asset validation."""

    proposed_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    name: LocalizedText
    category: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    description_hint: LocalizedText | None = None
    semantic_granularity: SemanticGranularity = "inherit"
    parent: TargetSelector | None = None
    independently_movable: bool | None = None

    @model_validator(mode="after")
    def semantic_shape(self) -> ProposedObject:
        is_child = self.semantic_granularity == "independent_child_asset"
        if is_child != (self.parent is not None):
            raise ValueError(
                "independent_child_asset proposals require parent; other granularities forbid it"
            )
        if self.semantic_granularity == "merged_component" and self.independently_movable:
            raise ValueError("merged-component proposals cannot be independently movable")
        if self.semantic_granularity == "merged_component":
            object.__setattr__(self, "independently_movable", False)
        elif self.semantic_granularity != "inherit" and self.independently_movable is None:
            object.__setattr__(self, "independently_movable", True)
        if len(self.aliases) != len(set(self.aliases)):
            raise ValueError("proposed object aliases must be unique")
        if any(not alias.strip() for alias in self.aliases):
            raise ValueError("proposed object aliases cannot be blank")
        return self


class DescriptionPatch(StrictModel):
    short: LocalizedText | None = None
    detailed: LocalizedText | None = None
    appearance: LocalizedText | None = None
    location: LocalizedText | None = None
    fidelity_caveat: LocalizedText | None = None

    @model_validator(mode="after")
    def not_empty(self) -> DescriptionPatch:
        if not any(getattr(self, name) is not None for name in type(self).model_fields):
            raise ValueError("description patch must set at least one field")
        return self


class ObjectPropertyPatch(StrictModel):
    name: LocalizedText | None = None
    category: str | None = Field(default=None, min_length=1)
    aliases: list[str] | None = None
    description: DescriptionPatch | None = None
    relations: list[Relation] | None = None
    interaction: InteractionPolicy | None = None
    independently_movable: bool | None = None
    bbox_scene: Bounds3D | None = None
    obb_scene: OrientedBounds3D | None = None
    transform_scene_from_asset: SceneTransform | None = None

    @model_validator(mode="after")
    def not_empty_and_unique_aliases(self) -> ObjectPropertyPatch:
        if not any(getattr(self, name) is not None for name in type(self).model_fields):
            raise ValueError("property patch must set at least one field")
        if self.aliases is not None and len(self.aliases) != len(set(self.aliases)):
            raise ValueError("aliases must be unique")
        return self


class QueryLocationIntent(StrictModel):
    kind: Literal["query_location"] = "query_location"
    target: TargetSelector
    language: Literal["auto", "zh", "en"] = "auto"


class QueryDescriptionIntent(StrictModel):
    kind: Literal["query_description"] = "query_description"
    target: TargetSelector
    language: Literal["auto", "zh", "en"] = "auto"


class AddObjectIntent(StrictModel):
    kind: Literal["add_object"] = "add_object"
    object: ProposedObject
    segmentation_prompt: str = Field(min_length=1)
    reference_asset_uris: list[str] = Field(default_factory=list)
    placement_hint: LocalizedText | None = None


class DeleteObjectIntent(StrictModel):
    kind: Literal["delete_object"] = "delete_object"
    target: TargetSelector
    cascade_descendants: bool = False
    repair_exposed_background: Literal[True] = True
    retain_superseded_assets: Literal[True] = True


class UpdatePropertiesIntent(StrictModel):
    kind: Literal["update_properties"] = "update_properties"
    target: TargetSelector
    patch: ObjectPropertyPatch


class SplitObjectIntent(StrictModel):
    kind: Literal["split_object"] = "split_object"
    target: TargetSelector
    parts: list[ProposedObject] = Field(min_length=2)
    child_reassignment: dict[str, str] = Field(default_factory=dict)
    retain_superseded_assets: Literal[True] = True

    @model_validator(mode="after")
    def unique_parts_and_valid_child_destinations(self) -> SplitObjectIntent:
        part_ids = [part.proposed_id for part in self.parts]
        if len(part_ids) != len(set(part_ids)):
            raise ValueError("split part proposed_id values must be unique")
        unknown = set(self.child_reassignment.values()) - set(part_ids)
        if unknown:
            raise ValueError(
                "child_reassignment destinations must name proposed split parts: "
                + ", ".join(sorted(unknown))
            )
        return self


class MergeObjectsIntent(StrictModel):
    kind: Literal["merge_objects"] = "merge_objects"
    targets: list[TargetSelector] = Field(min_length=2)
    result: ProposedObject
    preserve_descendants: Literal[True] = True
    retain_superseded_assets: Literal[True] = True


class ReparentObjectIntent(StrictModel):
    kind: Literal["reparent_object"] = "reparent_object"
    target: TargetSelector
    new_parent: TargetSelector | None = None
    preserve_world_transform: Literal[True] = True


SceneIntent = Annotated[
    QueryLocationIntent
    | QueryDescriptionIntent
    | AddObjectIntent
    | DeleteObjectIntent
    | UpdatePropertiesIntent
    | SplitObjectIntent
    | MergeObjectsIntent
    | ReparentObjectIntent,
    Field(discriminator="kind"),
]


class CommandProvenance(StrictModel):
    requested_at: datetime
    requester: str = Field(min_length=1)
    raw_prompt: str = Field(min_length=1)
    parser_provider: Literal["structured_json", "rule_based", "external_llm"] = "structured_json"
    parser_model: str | None = None
    source_message_id: str | None = None
    context_references: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def timezone_is_explicit(self) -> CommandProvenance:
        if self.requested_at.tzinfo is None or self.requested_at.utcoffset() is None:
            raise ValueError("requested_at must include an explicit timezone")
        if self.parser_provider == "external_llm" and not self.parser_model:
            raise ValueError("external_llm provenance requires parser_model")
        return self


class SceneCommand(StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    request_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.:-]+$")
    expected_manifest_sha256: Sha256 | None = None
    provenance: CommandProvenance
    intent: SceneIntent

    @property
    def is_mutating(self) -> bool:
        return self.intent.kind not in {"query_location", "query_description"}


class TargetCandidate(StrictModel):
    object_id: str
    scoped_id: str
    name: LocalizedText
    category: str
    score: int = Field(ge=0)
    evidence: str


class TargetResolution(StrictModel):
    selector: TargetSelector
    status: Literal["resolved", "ambiguous", "not_found", "ordinal_out_of_range"]
    resolved_object_id: str | None = None
    resolved_scoped_id: str | None = None
    candidates: list[TargetCandidate] = Field(default_factory=list)
    reason: str

    @model_validator(mode="after")
    def resolution_shape(self) -> TargetResolution:
        has_resolution = self.resolved_object_id is not None or self.resolved_scoped_id is not None
        if self.status == "resolved":
            if not self.resolved_object_id or not self.resolved_scoped_id:
                raise ValueError("resolved target must include object and scoped ids")
            if len(self.candidates) != 1:
                raise ValueError("resolved target must have exactly one candidate")
        elif has_resolution:
            raise ValueError("unresolved target cannot declare resolved ids")
        if self.status == "ambiguous" and len(self.candidates) < 2:
            raise ValueError("ambiguous target requires at least two candidates")
        if self.status == "not_found" and self.candidates:
            raise ValueError("not-found target cannot include candidates")
        if self.status == "ordinal_out_of_range" and not self.candidates:
            raise ValueError("out-of-range ordinal requires the available candidates")
        return self


class AffectedStage(StrictModel):
    stage: StageName
    reason: str = Field(min_length=1)
    invalidates_previous_output: bool


OperationKind: TypeAlias = Literal[
    "read_location",
    "read_description",
    "register_object_request",
    "segment_instances",
    "fuse_instance_geometry",
    "refresh_scene_cognition",
    "build_completion_plan",
    "complete_object_and_clean_plate",
    "validate_or_update_placement",
    "tombstone_objects",
    "update_object_properties",
    "update_object_hierarchy",
    "split_semantic_object",
    "merge_semantic_objects",
    "rebuild_world_bundle",
    "run_web_qa",
]


class PlannedOperation(StrictModel):
    operation_id: str = Field(min_length=1, pattern=r"^[a-z0-9_]+$")
    kind: OperationKind
    stage: StageName | None = None
    target_object_ids: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def unique_references(self) -> PlannedOperation:
        if len(self.target_object_ids) != len(set(self.target_object_ids)):
            raise ValueError("operation target ids must be unique")
        if len(self.depends_on) != len(set(self.depends_on)):
            raise ValueError("operation dependencies must be unique")
        if self.operation_id in self.depends_on:
            raise ValueError("operation cannot depend on itself")
        return self


class ConfirmationRequirement(StrictModel):
    required: bool
    reason: str
    scope_sha256: Sha256 | None = None
    required_phrase: str | None = None

    @model_validator(mode="after")
    def required_fields(self) -> ConfirmationRequirement:
        if self.required != (self.scope_sha256 is not None):
            raise ValueError("confirmation scope_sha256 is required exactly when confirmation is")
        if self.required != (self.required_phrase is not None):
            raise ValueError("required_phrase is required exactly when confirmation is")
        return self


class RollbackPlan(StrictModel):
    reversibility: Literal["not_applicable", "fully_reversible", "conditionally_reversible"]
    manifest_snapshot_required: bool
    retain_superseded_assets: bool
    strategy: str = Field(min_length=1)
    limitations: list[str] = Field(default_factory=list)


class ReadResult(StrictModel):
    status: Literal["available", "unavailable"]
    object_id: str
    scoped_id: str
    answer: str | None = None
    focus_bbox: Bounds3D | None = None
    evidence: list[str] = Field(default_factory=list)
    reason: str | None = None


class PlanProvenance(StrictModel):
    planner: Literal["video2world.scene_commands"] = "video2world.scene_commands"
    planner_version: Literal["1.0.0"] = "1.0.0"
    command_request_id: str
    command_sha256: Sha256
    canonical_manifest_sha256: Sha256
    raw_prompt: str
    parser_provider: str
    parser_model: str | None = None
    source_message_id: str | None = None
    context_references: list[str] = Field(default_factory=list)


class SceneCommandPlan(StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    plan_id: str = Field(pattern=r"^sceneplan_[0-9a-f]{20}$")
    created_at: datetime
    command_kind: CommandKind
    status: Literal[
        "ready",
        "blocked_ambiguous",
        "blocked_not_found",
        "blocked_invariant",
        "blocked_stale_manifest",
    ]
    preview_only: Literal[True] = True
    manifest_mutated: Literal[False] = False
    risk_level: Literal["none", "low", "medium", "high", "critical"]
    idempotency_key: Sha256
    target_resolutions: list[TargetResolution] = Field(default_factory=list)
    affected_stages: list[AffectedStage] = Field(default_factory=list)
    operations: list[PlannedOperation] = Field(default_factory=list)
    confirmation: ConfirmationRequirement
    rollback: RollbackPlan
    preconditions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    read_result: ReadResult | None = None
    provenance: PlanProvenance

    @model_validator(mode="after")
    def plan_invariants(self) -> SceneCommandPlan:
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must include an explicit timezone")
        if len(self.affected_stages) != len({item.stage for item in self.affected_stages}):
            raise ValueError("affected stages must be unique")
        if self.status != "ready" and self.operations:
            raise ValueError("blocked plans cannot contain executable operations")
        if self.status == "ready" and not self.operations:
            raise ValueError("ready plans require at least one operation")
        query = self.command_kind in {"query_location", "query_description"}
        if query:
            if self.confirmation.required:
                raise ValueError("read-only query plans cannot require confirmation")
            if self.affected_stages:
                raise ValueError("read-only query plans cannot invalidate pipeline stages")
            if self.risk_level != "none":
                raise ValueError("read-only query plans must have risk_level='none'")
        elif self.status == "ready" and self.risk_level in {"high", "critical"}:
            if not self.confirmation.required:
                raise ValueError("ready high-risk plans require explicit confirmation")
        if self.confirmation.required and self.confirmation.scope_sha256 != self.idempotency_key:
            raise ValueError("confirmation scope must equal the plan idempotency key")
        operation_ids = [operation.operation_id for operation in self.operations]
        if len(operation_ids) != len(set(operation_ids)):
            raise ValueError("operation ids must be unique")
        seen: set[str] = set()
        for operation in self.operations:
            if any(dependency not in seen for dependency in operation.depends_on):
                raise ValueError("operation dependencies must reference earlier operations")
            seen.add(operation.operation_id)
        if self.status == "ready" and not query:
            affected = {item.stage for item in self.affected_stages}
            operated = {item.stage for item in self.operations if item.stage is not None}
            if affected != operated:
                raise ValueError(
                    "every affected stage must have a planned operation and vice versa"
                )
            if self.read_result is not None:
                raise ValueError("mutating plans cannot return a read result")
        if self.status == "ready" and any(
            resolution.status != "resolved" for resolution in self.target_resolutions
        ):
            raise ValueError("ready plans cannot contain unresolved targets")
        return self


def scene_command_json_schema() -> dict[str, Any]:
    schema = SceneCommand.model_json_schema(mode="validation")
    schema["$id"] = "https://relumeow.top/schemas/video2world/scene-command-1.0.0.json"
    schema["title"] = "Video2World Scene Command"
    return schema


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _instance_number(item: WorldObject) -> tuple[int, str]:
    match = re.search(r"(\d+)$", item.id)
    return (int(match.group(1)) if match else 10**9, item.scoped_id)


def _candidate(item: WorldObject, score: int, evidence: str) -> TargetCandidate:
    return TargetCandidate(
        object_id=item.id,
        scoped_id=item.scoped_id,
        name=item.name,
        category=item.category,
        score=score,
        evidence=evidence,
    )


def resolve_target(manifest: WorldManifest, selector: TargetSelector) -> TargetResolution:
    """Resolve a structured selector without guessing between equal instances."""

    if selector.scoped_id is not None:
        matches = [item for item in manifest.objects if item.scoped_id == selector.scoped_id]
        evidence = "exact scoped_id"
    elif selector.object_id is not None:
        matches = [item for item in manifest.objects if item.id == selector.object_id]
        evidence = "exact object_id"
    else:
        assert selector.label is not None
        key = _normalize(selector.label)
        scored: list[tuple[WorldObject, int, str]] = []
        for item in manifest.objects:
            terms: list[tuple[str, int, str]] = [
                (item.name.zh or "", 800, "name.zh"),
                (item.name.en or "", 800, "name.en"),
                (item.id, 700, "id"),
                *[(alias, 600, "alias") for alias in item.aliases],
                (item.category, 400, "category"),
            ]
            best: tuple[int, str] | None = None
            for value, score, source in terms:
                if _normalize(value) != key:
                    continue
                candidate = (score, f"exact {source}:{value}")
                if best is None or candidate[0] > best[0]:
                    best = candidate
            if best:
                scored.append((item, *best))
        if not scored:
            matches = []
            evidence = "no exact normalized name, alias, id, or category match"
        else:
            highest = max(score for _, score, _ in scored)
            selected = [(item, score, source) for item, score, source in scored if score == highest]
            ordered = sorted((item for item, _, _ in selected), key=_instance_number)
            if selector.ordinal is not None:
                if selector.ordinal <= len(ordered):
                    matches = [ordered[selector.ordinal - 1]]
                    evidence = f"label plus ordinal {selector.ordinal}"
                else:
                    candidates = [
                        _candidate(item, highest, source)
                        for item, _, source in sorted(
                            selected,
                            key=lambda row: _instance_number(row[0]),
                        )
                    ]
                    return TargetResolution(
                        selector=selector,
                        status="ordinal_out_of_range",
                        candidates=candidates,
                        reason=(
                            f"ordinal {selector.ordinal} is out of range for "
                            f"{len(ordered)} matching instances"
                        ),
                    )
            else:
                matches = ordered
                evidence = selected[0][2]

    if not matches:
        return TargetResolution(
            selector=selector,
            status="not_found",
            reason=evidence,
        )
    candidates = [_candidate(item, 1000, evidence) for item in matches]
    if len(matches) > 1:
        return TargetResolution(
            selector=selector,
            status="ambiguous",
            candidates=candidates,
            reason="multiple instances match; provide scoped_id, object_id, or ordinal",
        )
    item = matches[0]
    return TargetResolution(
        selector=selector,
        status="resolved",
        resolved_object_id=item.id,
        resolved_scoped_id=item.scoped_id,
        candidates=candidates,
        reason=evidence,
    )


def _resolved_object(
    manifest: WorldManifest,
    resolution: TargetResolution,
) -> WorldObject | None:
    if resolution.status != "resolved":
        return None
    return next(
        (item for item in manifest.objects if item.scoped_id == resolution.resolved_scoped_id),
        None,
    )


def _blocked_status(resolutions: list[TargetResolution]) -> str | None:
    if any(item.status == "ambiguous" for item in resolutions):
        return "blocked_ambiguous"
    if any(item.status in {"not_found", "ordinal_out_of_range"} for item in resolutions):
        return "blocked_not_found"
    return None


def _parent_map(manifest: WorldManifest) -> dict[str, str]:
    return {
        item.id: item.parent_object_id
        for item in manifest.objects
        if item.parent_object_id is not None
    }


def _direct_children(manifest: WorldManifest, object_id: str) -> list[WorldObject]:
    return sorted(
        (item for item in manifest.objects if item.parent_object_id == object_id),
        key=lambda item: item.scoped_id,
    )


def _descendant_ids(manifest: WorldManifest, object_id: str) -> list[str]:
    result: list[str] = []
    frontier = [object_id]
    while frontier:
        parent = frontier.pop(0)
        children = [item.id for item in _direct_children(manifest, parent)]
        result.extend(children)
        frontier.extend(children)
    return result


def _is_ancestor(manifest: WorldManifest, ancestor_id: str, object_id: str) -> bool:
    parents = _parent_map(manifest)
    cursor = parents.get(object_id)
    while cursor is not None:
        if cursor == ancestor_id:
            return True
        cursor = parents.get(cursor)
    return False


def _effective_language(
    command: SceneCommand,
    requested: Literal["auto", "zh", "en"],
) -> Literal["zh", "en"]:
    if requested != "auto":
        return requested
    return "zh" if re.search(r"[\u3400-\u9fff]", command.provenance.raw_prompt) else "en"


def _read_result(
    manifest: WorldManifest,
    command: SceneCommand,
    item: WorldObject,
) -> ReadResult:
    intent = command.intent
    assert isinstance(intent, QueryLocationIntent | QueryDescriptionIntent)
    language = _effective_language(command, intent.language)
    query = (
        f"where is {item.scoped_id}"
        if isinstance(intent, QueryLocationIntent)
        else f"what does {item.scoped_id} look like"
    )
    result = query_world(manifest, query, language=language)
    if result.status == "resolved":
        return ReadResult(
            status="available",
            object_id=item.id,
            scoped_id=item.scoped_id,
            answer=result.answer,
            focus_bbox=result.focus_bbox,
            evidence=result.evidence,
        )
    return ReadResult(
        status="unavailable",
        object_id=item.id,
        scoped_id=item.scoped_id,
        focus_bbox=item.bbox_scene,
        reason=result.reason or "resolved object has no reviewed evidence for this query",
    )


def _stage_records(stages: tuple[StageName, ...], reason: str) -> list[AffectedStage]:
    return [
        AffectedStage(stage=stage, reason=reason, invalidates_previous_output=True)
        for stage in _STAGE_ORDER
        if stage in stages
    ]


def _operation_chain(
    steps: list[tuple[OperationKind, StageName | None, list[str], dict[str, JsonValue]]],
) -> list[PlannedOperation]:
    operations: list[PlannedOperation] = []
    for index, (kind, stage, targets, parameters) in enumerate(steps, start=1):
        operation_id = f"op_{index:02d}_{kind}"
        operations.append(
            PlannedOperation(
                operation_id=operation_id,
                kind=kind,
                stage=stage,
                target_object_ids=targets,
                depends_on=[operations[-1].operation_id] if operations else [],
                parameters=parameters,
            )
        )
    return operations


def _full_reconstruction_steps(
    *,
    initial_kind: OperationKind,
    targets: list[str],
    initial_parameters: dict[str, JsonValue],
) -> list[PlannedOperation]:
    return _operation_chain(
        [
            (initial_kind, "inventory", targets, initial_parameters),
            ("segment_instances", "sam3", targets, {}),
            ("fuse_instance_geometry", "fusion", targets, {}),
            ("refresh_scene_cognition", "cognition", targets, {}),
            ("build_completion_plan", "completion_plan", targets, {}),
            (
                "complete_object_and_clean_plate",
                "layered_completion",
                targets,
                {
                    "required_views": ["front", "back", "left", "right", "top", "bottom"],
                    "preserve_identity_attributes": [
                        "category",
                        "color",
                        "material",
                        "shape",
                        "aspect_ratio",
                    ],
                    "vlm_review_required": True,
                    "semantic_granularity_review_required": True,
                },
            ),
            ("validate_or_update_placement", "placement", targets, {}),
            ("rebuild_world_bundle", "bundle", targets, {}),
            ("run_web_qa", "web", targets, {}),
        ]
    )


def _confirmation(
    required: bool,
    *,
    idempotency_key: str,
    reason: str,
) -> ConfirmationRequirement:
    if not required:
        return ConfirmationRequirement(required=False, reason=reason)
    return ConfirmationRequirement(
        required=True,
        reason=reason,
        scope_sha256=idempotency_key,
        required_phrase=f"confirm {idempotency_key[:12]}",
    )


def _rollback(mutating: bool) -> RollbackPlan:
    if not mutating:
        return RollbackPlan(
            reversibility="not_applicable",
            manifest_snapshot_required=False,
            retain_superseded_assets=False,
            strategy="Read-only command; no scene state changes are planned.",
        )
    return RollbackPlan(
        reversibility="conditionally_reversible",
        manifest_snapshot_required=True,
        retain_superseded_assets=True,
        strategy=(
            "Atomically restore the pre-command manifest snapshot and its content-addressed "
            "asset references; newly generated assets remain unreferenced until garbage collection."
        ),
        limitations=[
            "Rollback is guaranteed only while the manifest snapshot and superseded assets "
            "are retained."
        ],
    )


def _existing_id_conflict(
    manifest: WorldManifest,
    proposed_ids: list[str],
    *,
    replaceable_ids: set[str] | None = None,
) -> str | None:
    replaceable = replaceable_ids or set()
    existing = {item.id for item in manifest.objects} - replaceable
    conflicts = existing.intersection(proposed_ids)
    if conflicts:
        return "proposed object ids already exist: " + ", ".join(sorted(conflicts))
    return None


def _unique_ids(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _web_interaction_parent_error(parent: WorldObject) -> str | None:
    """Return why a canonical object cannot own a Web interaction-layer child."""

    if parent.semantic_granularity == "merged_component":
        return f"merged component {parent.id!r} cannot own child assets"
    interaction = parent.interaction
    interaction_enabled = (
        interaction.selectable
        or interaction.double_click_action != "none"
        or interaction.drag_action != "none"
        or interaction.collision_enabled
        or interaction.physics_mode != "none"
    )
    if not interaction_enabled:
        return f"parent {parent.id!r} is knowledge-only and has no enabled Web interaction policy"
    missing = [
        field
        for field in ("bbox_scene", "transform_scene_from_asset")
        if getattr(parent, field) is None
    ]
    visual_assets = [
        asset
        for asset in (parent.visual, parent.render_mesh, parent.unified_pbr_glb)
        if asset is not None
    ]
    if not visual_assets:
        missing.append("visual, render_mesh, or unified_pbr_glb")
    if missing:
        return (
            f"parent {parent.id!r} cannot be represented in the Web interaction layer; "
            f"missing {', '.join(missing)}"
        )
    if any(asset.status != "validated" for asset in visual_assets):
        return f"parent {parent.id!r} requires validated Web visual representations"
    if not parent.quality_gates.visual_interaction_ready():
        return f"parent {parent.id!r} requires passed file, semantic, alignment, and visual gates"
    return None


def _proposal_parent_resolutions(
    manifest: WorldManifest,
    proposals: list[ProposedObject],
) -> list[TargetResolution]:
    return [
        resolve_target(manifest, proposal.parent)
        for proposal in proposals
        if proposal.parent is not None
    ]


def _proposal_parent_error(
    manifest: WorldManifest,
    proposals: list[ProposedObject],
    resolutions: list[TargetResolution],
) -> str | None:
    resolution_index = iter(resolutions)
    for proposal in proposals:
        if proposal.parent is None:
            continue
        resolution = next(resolution_index)
        parent = _resolved_object(manifest, resolution)
        if parent and parent.id == proposal.proposed_id:
            return f"proposed object {proposal.proposed_id!r} cannot parent itself"
        if parent and (parent_error := _web_interaction_parent_error(parent)):
            return parent_error
    return None


def _build_ready_details(
    manifest: WorldManifest,
    command: SceneCommand,
    resolutions: list[TargetResolution],
) -> tuple[
    str,
    str,
    bool,
    list[AffectedStage],
    list[PlannedOperation],
    ReadResult | None,
    list[str],
    list[str],
]:
    """Return status, risk, confirmation, stages, ops, read result, warnings, preconditions."""

    intent = command.intent
    blocked = _blocked_status(resolutions)
    if blocked:
        return blocked, "none" if not command.is_mutating else "high", False, [], [], None, [], []

    warnings: list[str] = []
    preconditions = ["canonical manifest hash must still match at execution time"]

    if isinstance(intent, QueryLocationIntent | QueryDescriptionIntent):
        item = _resolved_object(manifest, resolutions[0])
        assert item is not None
        result = _read_result(manifest, command, item)
        kind: OperationKind = (
            "read_location" if isinstance(intent, QueryLocationIntent) else "read_description"
        )
        return (
            "ready",
            "none",
            False,
            [],
            _operation_chain([(kind, None, [item.id], {})]),
            result,
            warnings,
            preconditions,
        )

    if isinstance(intent, AddObjectIntent):
        if intent.object.semantic_granularity == "inherit":
            return (
                "blocked_invariant",
                "high",
                False,
                [],
                [],
                None,
                ["add_object requires explicit root, child, or merged semantic granularity"],
                preconditions,
            )
        if conflict := _existing_id_conflict(manifest, [intent.object.proposed_id]):
            return "blocked_invariant", "high", False, [], [], None, [conflict], preconditions
        if parent_error := _proposal_parent_error(manifest, [intent.object], resolutions):
            return (
                "blocked_invariant",
                "high",
                False,
                [],
                [],
                None,
                [parent_error],
                preconditions,
            )
        targets = [intent.object.proposed_id]
        operations = _full_reconstruction_steps(
            initial_kind="register_object_request",
            targets=targets,
            initial_parameters={
                "object": intent.object.model_dump(mode="json"),
                "segmentation_prompt": intent.segmentation_prompt,
                "reference_asset_uris": intent.reference_asset_uris,
                "placement_hint": (
                    intent.placement_hint.model_dump(mode="json") if intent.placement_hint else None
                ),
            },
        )
        return (
            "ready",
            "high",
            True,
            _stage_records(
                _RECONSTRUCTION_STAGES,
                "new semantic asset requires full reconstruction",
            ),
            operations,
            None,
            warnings,
            [
                *preconditions,
                "segmentation, six-view identity review, placement, and browser QA must pass",
            ],
        )

    if isinstance(intent, DeleteObjectIntent):
        item = _resolved_object(manifest, resolutions[0])
        assert item is not None
        descendants = _descendant_ids(manifest, item.id)
        if descendants and not intent.cascade_descendants:
            return (
                "blocked_invariant",
                "critical",
                False,
                [],
                [],
                None,
                [
                    f"{item.id!r} owns descendants; set cascade_descendants=true or reparent them: "
                    + ", ".join(descendants)
                ],
                preconditions,
            )
        targets = [item.id, *descendants]
        operations = _operation_chain(
            [
                (
                    "tombstone_objects",
                    "inventory",
                    targets,
                    {
                        "physical_delete": False,
                        "retain_superseded_assets": True,
                        "cascade_descendants": intent.cascade_descendants,
                    },
                ),
                ("build_completion_plan", "completion_plan", targets, {}),
                (
                    "complete_object_and_clean_plate",
                    "layered_completion",
                    targets,
                    {"repair_exposed_background": True, "vlm_review_required": True},
                ),
                ("validate_or_update_placement", "placement", targets, {}),
                ("rebuild_world_bundle", "bundle", targets, {}),
                ("run_web_qa", "web", targets, {}),
            ]
        )
        stages: tuple[StageName, ...] = (
            "inventory",
            "completion_plan",
            "layered_completion",
            "placement",
            "bundle",
            "web",
        )
        return (
            "ready",
            "critical",
            True,
            _stage_records(
                stages,
                "object removal exposes geometry and requires clean-plate repair",
            ),
            operations,
            None,
            ["source objects are tombstoned, not physically deleted"],
            [*preconditions, "clean-plate review must pass before bundle promotion"],
        )

    if isinstance(intent, UpdatePropertiesIntent):
        item = _resolved_object(manifest, resolutions[0])
        assert item is not None
        patch_fields = set(intent.patch.model_fields_set)
        geometric = {
            "bbox_scene",
            "obb_scene",
            "transform_scene_from_asset",
            "interaction",
            "independently_movable",
        }
        if item.semantic_granularity == "merged_component" and patch_fields.intersection(geometric):
            return (
                "blocked_invariant",
                "medium",
                False,
                [],
                [],
                None,
                ["merged components cannot receive independent placement or interaction updates"],
                preconditions,
            )
        scene_frame = manifest.scene.coordinate_system.frame_id
        if intent.patch.bbox_scene and intent.patch.bbox_scene.frame_id != scene_frame:
            return (
                "blocked_invariant",
                "medium",
                False,
                [],
                [],
                None,
                ["bbox_scene patch is not in the world scene frame"],
                preconditions,
            )
        if intent.patch.obb_scene and intent.patch.obb_scene.frame_id != scene_frame:
            return (
                "blocked_invariant",
                "medium",
                False,
                [],
                [],
                None,
                ["obb_scene patch is not in the world scene frame"],
                preconditions,
            )
        if (
            intent.patch.transform_scene_from_asset
            and intent.patch.transform_scene_from_asset.to_frame != scene_frame
        ):
            return (
                "blocked_invariant",
                "medium",
                False,
                [],
                [],
                None,
                ["transform patch does not target the world scene frame"],
                preconditions,
            )
        metadata_only = patch_fields <= {"name", "aliases", "description", "relations"}
        if metadata_only:
            stages = _METADATA_STAGES
            risk = "low"
            confirm = False
        else:
            stages = _HIERARCHY_STAGES
            risk = "medium"
            confirm = True
        if "category" in patch_fields:
            stages = ("inventory", *_METADATA_STAGES)
            risk = "medium"
            confirm = True
        property_parameters: dict[str, JsonValue] = {
            "patch": intent.patch.model_dump(mode="json", exclude_unset=True),
            "patch_semantics": "merge_non_null_fields",
        }
        interaction = intent.patch.interaction
        interaction_enabled = bool(
            interaction
            and (
                interaction.selectable
                or interaction.double_click_action != "none"
                or interaction.drag_action != "none"
                or interaction.collision_enabled
                or interaction.physics_mode != "none"
            )
        )
        visual_assets = [
            asset
            for asset in (item.visual, item.render_mesh, item.unified_pbr_glb)
            if asset is not None
        ]
        missing_visual_contract = interaction_enabled and (
            item.bbox_scene is None
            or not visual_assets
            or any(asset.status != "validated" for asset in visual_assets)
            or item.transform_scene_from_asset is None
            or not item.quality_gates.visual_interaction_ready()
        )
        unified_collision_ready = bool(
            item.unified_pbr_glb is not None
            and item.collision_topology is not None
            and item.unified_pbr_glb.status == "validated"
        )
        separate_collision_ready = bool(
            item.collider is not None and item.collider.status == "validated"
        )
        missing_collision_contract = bool(
            interaction
            and interaction.collision_enabled
            and (
                not (unified_collision_ready or separate_collision_ready)
                or item.quality_gates.collision.status != "passed"
            )
        )
        if missing_visual_contract or missing_collision_contract:
            stages = (
                "completion_plan",
                "layered_completion",
                "placement",
                "bundle",
                "web",
            )
            risk = "high"
            confirm = True
            required_repairs = []
            if missing_visual_contract:
                required_repairs.append("validated_visual_interaction_contract")
            if missing_collision_contract:
                required_repairs.append("validated_unified_or_legacy_collision_contract")
            property_parameters["required_asset_repairs"] = required_repairs
            warnings.append("interaction prerequisites are incomplete; reconstruction is routed")
        if interaction and not interaction_enabled:
            if item.unified_pbr_glb is not None:
                property_parameters["clear_unified_pbr_glb_reference"] = True
                property_parameters["clear_collision_topology"] = True
                property_parameters["reset_collision_gate_to_not_tested"] = True
                warnings.append(
                    "disabling interaction removes the active unified PBR GLB reference and "
                    "collision topology while retaining the content-addressed asset for rollback"
                )
            elif item.collider is not None:
                property_parameters["clear_collider_reference"] = True
                warnings.append("disabling interaction also removes the active collider reference")
        update_steps: list[
            tuple[OperationKind, StageName | None, list[str], dict[str, JsonValue]]
        ] = [
            (
                "update_object_properties",
                stages[0],
                [item.id],
                property_parameters,
            )
        ]
        if "cognition" in stages and stages[0] != "cognition":
            update_steps.append(("refresh_scene_cognition", "cognition", [item.id], {}))
        if "layered_completion" in stages:
            update_steps.append(
                (
                    "complete_object_and_clean_plate",
                    "layered_completion",
                    [item.id],
                    {
                        "unified_pbr_glb_action": (
                            "reuse_and_validate"
                            if interaction
                            and interaction.collision_enabled
                            and item.unified_pbr_glb is not None
                            else "generate_and_validate"
                            if interaction and interaction.collision_enabled
                            else "not_requested"
                        ),
                        "generate_collision_proxy": False,
                        "required_views": [
                            "front",
                            "back",
                            "left",
                            "right",
                            "top",
                            "bottom",
                        ],
                        "preserve_identity_attributes": [
                            "category",
                            "color",
                            "material",
                            "shape",
                            "aspect_ratio",
                        ],
                        "vlm_review_required": True,
                    },
                )
            )
        if "placement" in stages:
            update_steps.append(("validate_or_update_placement", "placement", [item.id], {}))
        update_steps.extend(
            [
                ("rebuild_world_bundle", "bundle", [item.id], {}),
                ("run_web_qa", "web", [item.id], {}),
            ]
        )
        operations = _operation_chain(update_steps)
        return (
            "ready",
            risk,
            confirm,
            _stage_records(stages, "object property update invalidates derived views"),
            operations,
            None,
            warnings,
            preconditions,
        )

    if isinstance(intent, SplitObjectIntent):
        item = _resolved_object(manifest, resolutions[0])
        assert item is not None
        part_ids = [part.proposed_id for part in intent.parts]
        if conflict := _existing_id_conflict(
            manifest,
            part_ids,
            replaceable_ids={item.id},
        ):
            return "blocked_invariant", "high", False, [], [], None, [conflict], preconditions
        children = _direct_children(manifest, item.id)
        child_ids = {child.id for child in children}
        assignment_ids = set(intent.child_reassignment)
        if child_ids != assignment_ids:
            missing = sorted(child_ids - assignment_ids)
            unexpected = sorted(assignment_ids - child_ids)
            details = []
            if missing:
                details.append("missing child assignments: " + ", ".join(missing))
            if unexpected:
                details.append("unknown child assignments: " + ", ".join(unexpected))
            return (
                "blocked_invariant",
                "high",
                False,
                [],
                [],
                None,
                ["split must explicitly preserve every direct child; " + "; ".join(details)],
                preconditions,
            )
        if parent_error := _proposal_parent_error(manifest, intent.parts, resolutions[1:]):
            return (
                "blocked_invariant",
                "high",
                False,
                [],
                [],
                None,
                [parent_error],
                preconditions,
            )
        split_parent_resolutions = iter(resolutions[1:])
        for part in intent.parts:
            if part.parent is None:
                continue
            parent_resolution = next(split_parent_resolutions)
            parent = _resolved_object(manifest, parent_resolution)
            if parent and parent.id == item.id:
                return (
                    "blocked_invariant",
                    "high",
                    False,
                    [],
                    [],
                    None,
                    ["split parts cannot remain children of the source object being replaced"],
                    preconditions,
                )
        operations = _full_reconstruction_steps(
            initial_kind="split_semantic_object",
            targets=_unique_ids([item.id, *part_ids]),
            initial_parameters={
                "source_object_id": item.id,
                "parts": [part.model_dump(mode="json") for part in intent.parts],
                "child_reassignment": intent.child_reassignment,
                "retain_superseded_assets": True,
            },
        )
        return (
            "ready",
            "high",
            True,
            _stage_records(_RECONSTRUCTION_STAGES, "semantic split changes masks and geometry"),
            operations,
            None,
            warnings,
            [*preconditions, "all split parts must pass identity and six-view completion review"],
        )

    if isinstance(intent, MergeObjectsIntent):
        items = [
            _resolved_object(manifest, resolution)
            for resolution in resolutions[: len(intent.targets)]
        ]
        assert all(item is not None for item in items)
        source_items = [item for item in items if item is not None]
        source_ids = [item.id for item in source_items]
        if len(source_ids) != len(set(source_ids)):
            return (
                "blocked_invariant",
                "high",
                False,
                [],
                [],
                None,
                ["merge target selectors resolve to duplicate objects"],
                preconditions,
            )
        for source in source_items:
            for other in source_items:
                if source.id != other.id and _is_ancestor(manifest, source.id, other.id):
                    return (
                        "blocked_invariant",
                        "high",
                        False,
                        [],
                        [],
                        None,
                        ["cannot merge an ancestor with its descendant; reparent first"],
                        preconditions,
                    )
        if intent.result.semantic_granularity == "merged_component":
            return (
                "blocked_invariant",
                "high",
                False,
                [],
                [],
                None,
                ["merge result must be an independent root or child asset, not a merged component"],
                preconditions,
            )
        if conflict := _existing_id_conflict(
            manifest,
            [intent.result.proposed_id],
            replaceable_ids=set(source_ids),
        ):
            return "blocked_invariant", "high", False, [], [], None, [conflict], preconditions
        if intent.result.semantic_granularity == "inherit":
            parent_ids = {item.parent_object_id for item in source_items}
            if len(parent_ids) != 1:
                return (
                    "blocked_invariant",
                    "high",
                    False,
                    [],
                    [],
                    None,
                    ["merge sources have different parents; declare an explicit result hierarchy"],
                    preconditions,
                )
        if parent_error := _proposal_parent_error(
            manifest,
            [intent.result],
            resolutions[len(intent.targets) :],
        ):
            return (
                "blocked_invariant",
                "high",
                False,
                [],
                [],
                None,
                [parent_error],
                preconditions,
            )
        if intent.result.parent is not None:
            result_parent = _resolved_object(manifest, resolutions[len(intent.targets)])
            if result_parent and result_parent.id in source_ids:
                return (
                    "blocked_invariant",
                    "high",
                    False,
                    [],
                    [],
                    None,
                    ["merge result cannot be parented to a source object being replaced"],
                    preconditions,
                )
        operations = _full_reconstruction_steps(
            initial_kind="merge_semantic_objects",
            targets=_unique_ids([*source_ids, intent.result.proposed_id]),
            initial_parameters={
                "source_object_ids": source_ids,
                "result": intent.result.model_dump(mode="json"),
                "preserve_descendants": True,
                "retain_superseded_assets": True,
            },
        )
        return (
            "ready",
            "high",
            True,
            _stage_records(_RECONSTRUCTION_STAGES, "semantic merge changes masks and geometry"),
            operations,
            None,
            warnings,
            [*preconditions, "merged identity and hierarchy must pass review"],
        )

    assert isinstance(intent, ReparentObjectIntent)
    item = _resolved_object(manifest, resolutions[0])
    assert item is not None
    parent = _resolved_object(manifest, resolutions[1]) if intent.new_parent else None
    if item.semantic_granularity == "merged_component":
        return (
            "blocked_invariant",
            "medium",
            False,
            [],
            [],
            None,
            ["merged components cannot be independently reparented; split them first"],
            preconditions,
        )
    if parent and parent.semantic_granularity == "merged_component":
        return (
            "blocked_invariant",
            "medium",
            False,
            [],
            [],
            None,
            ["merged components cannot own independently movable children"],
            preconditions,
        )
    if parent and (parent.id == item.id or _is_ancestor(manifest, item.id, parent.id)):
        return (
            "blocked_invariant",
            "medium",
            False,
            [],
            [],
            None,
            ["reparenting would create an interaction hierarchy cycle"],
            preconditions,
        )
    if item.parent_object_id == (parent.id if parent else None):
        return (
            "blocked_invariant",
            "low",
            False,
            [],
            [],
            None,
            ["requested parent is already active; no hierarchy change is needed"],
            preconditions,
        )
    if parent and (parent_error := _web_interaction_parent_error(parent)):
        return (
            "blocked_invariant",
            "medium",
            False,
            [],
            [],
            None,
            [parent_error],
            preconditions,
        )
    parameters = {
        "parent_object_id": parent.id if parent else None,
        "semantic_granularity": ("independent_child_asset" if parent else "independent_root_asset"),
        "moves_with_parent": parent is not None,
        "preserve_world_transform": True,
    }
    operations = _operation_chain(
        [
            ("update_object_hierarchy", "cognition", [item.id], parameters),
            ("validate_or_update_placement", "placement", [item.id], parameters),
            ("rebuild_world_bundle", "bundle", [item.id], {}),
            ("run_web_qa", "web", [item.id], {"verify_parent_child_motion": True}),
        ]
    )
    return (
        "ready",
        "medium",
        True,
        _stage_records(_HIERARCHY_STAGES, "parent-child transform semantics changed"),
        operations,
        None,
        warnings,
        [*preconditions, "world transform must remain unchanged when hierarchy is attached"],
    )


def plan_scene_command(
    manifest: WorldManifest,
    command: SceneCommand,
    *,
    created_at: datetime | None = None,
) -> SceneCommandPlan:
    """Build a side-effect-free plan; the supplied manifest is never modified."""

    manifest_sha256 = digest_json(manifest.model_dump(mode="json"))
    command_sha256 = digest_json(command.model_dump(mode="json"))
    idempotency_key = digest_json(
        {
            "schema_version": command.schema_version,
            "request_id": command.request_id,
            "requester": command.provenance.requester,
            "intent": command.intent.model_dump(mode="json"),
            "canonical_manifest_sha256": manifest_sha256,
        }
    )
    timestamp = created_at or datetime.now(UTC)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("created_at must include an explicit timezone")

    resolutions: list[TargetResolution] = []
    intent = command.intent
    single_target_intents = (
        QueryLocationIntent
        | QueryDescriptionIntent
        | DeleteObjectIntent
        | UpdatePropertiesIntent
        | SplitObjectIntent
        | ReparentObjectIntent
    )
    if isinstance(intent, single_target_intents):
        resolutions.append(resolve_target(manifest, intent.target))
    elif isinstance(intent, MergeObjectsIntent):
        resolutions.extend(resolve_target(manifest, target) for target in intent.targets)
    if isinstance(intent, ReparentObjectIntent) and intent.new_parent is not None:
        resolutions.append(resolve_target(manifest, intent.new_parent))

    proposals: list[ProposedObject] = []
    if isinstance(intent, AddObjectIntent):
        proposals = [intent.object]
    elif isinstance(intent, SplitObjectIntent):
        proposals = intent.parts
    elif isinstance(intent, MergeObjectsIntent):
        proposals = [intent.result]
    resolutions.extend(_proposal_parent_resolutions(manifest, proposals))

    stale = (
        command.expected_manifest_sha256 is not None
        and command.expected_manifest_sha256 != manifest_sha256
    )
    if stale:
        status = "blocked_stale_manifest"
        risk = "none" if not command.is_mutating else "high"
        require_confirmation = False
        stages: list[AffectedStage] = []
        operations: list[PlannedOperation] = []
        read_result = None
        warnings = [
            "expected_manifest_sha256 does not match the canonical manifest; re-resolve targets"
        ]
        preconditions: list[str] = []
    else:
        (
            status,
            risk,
            require_confirmation,
            stages,
            operations,
            read_result,
            warnings,
            preconditions,
        ) = _build_ready_details(manifest, command, resolutions)

    confirmation_reason = (
        "Explicit approval binds execution to this manifest, command, target set, and plan scope."
        if require_confirmation
        else (
            "Plan is blocked and cannot be confirmed."
            if status != "ready"
            else "Read-only or low-risk metadata preview does not require confirmation."
        )
    )
    return SceneCommandPlan(
        plan_id=f"sceneplan_{idempotency_key[:20]}",
        created_at=timestamp,
        command_kind=intent.kind,
        status=status,
        risk_level=risk,
        idempotency_key=idempotency_key,
        target_resolutions=resolutions,
        affected_stages=stages,
        operations=operations,
        confirmation=_confirmation(
            require_confirmation,
            idempotency_key=idempotency_key,
            reason=confirmation_reason,
        ),
        rollback=_rollback(command.is_mutating),
        preconditions=preconditions,
        warnings=warnings,
        read_result=read_result,
        provenance=PlanProvenance(
            command_request_id=command.request_id,
            command_sha256=command_sha256,
            canonical_manifest_sha256=manifest_sha256,
            raw_prompt=command.provenance.raw_prompt,
            parser_provider=command.provenance.parser_provider,
            parser_model=command.provenance.parser_model,
            source_message_id=command.provenance.source_message_id,
            context_references=command.provenance.context_references,
        ),
    )


def load_scene_command(path: str | Path) -> SceneCommand:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    return SceneCommand.model_validate(payload)


def load_scene_command_plan(path: str | Path) -> SceneCommandPlan:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    return SceneCommandPlan.model_validate(payload)


def write_scene_command_plan(path: str | Path, plan: SceneCommandPlan) -> None:
    atomic_write_json(
        Path(path).expanduser().resolve(),
        plan.model_dump(mode="json", exclude_none=True),
    )
