"""Contracts and planning for iterative front-to-back scene completion."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from video2world.hashing import atomic_write_json, digest_json
from video2world.models import (
    AssetRef,
    CollisionTopology,
    LocalizedText,
    Sha256,
    StrictModel,
    validate_unified_pbr_glb_asset,
)

CoverageStatus = Literal[
    "modeled_complete",
    "modeled_visible_surface_only",
    "segmented_only",
    "missing",
    "structure_background",
    "ignored",
]
CompletionNeed = Literal[
    "instance_split",
    "segmentation",
    "multi_view_lift",
    "backside_geometry",
    "render_asset",
    "collider",
    "clean_plate",
    "background_rebuild",
    "visual_qa",
]
MISSING_OBJECT_COMPLETION_NEEDS: frozenset[CompletionNeed] = frozenset(
    {
        "segmentation",
        "multi_view_lift",
        "backside_geometry",
        "render_asset",
        "collider",
        "visual_qa",
    }
)
ActionType = Literal[
    "split_instances",
    "segment",
    "lift_to_3d",
    "complete_object",
    "build_collider",
    "place_object",
    "clean_plate",
    "reinspect",
    "rebuild_background",
    "validate_round",
]
AcceptanceGate = Literal[
    "segmentation_evidence",
    "multi_view_projection",
    "geometry_completeness",
    "front_silhouette",
    "front_observation_fidelity",
    "backside_nonempty",
    "six_view_coverage",
    "appearance_fidelity",
    "six_side_style_consistency",
    "instance_identity",
    "placement_alignment",
    "scene_interpenetration",
    "collision",
    "mask_outside_unchanged",
    "cross_view_consistency",
    "revealed_background",
    "background_depth_normal",
    "browser_interaction",
    "vlm_geometry_review",
]
GeometryIssueType = Literal[
    "missing_back_surface",
    "unexpected_opening",
    "implausible_thickness",
    "implausible_proportions",
    "identity_drift",
    "front_fidelity_error",
    "unsupported_detail",
    "cross_view_style_inconsistency",
    "shape_mismatch",
    "color_mismatch",
    "material_mismatch",
    "part_merge",
    "part_missing",
    "disconnected_parts",
    "floating_parts",
    "scene_interpenetration",
    "self_intersection",
    "support_surface_error",
    "texture_hallucination",
    "other",
]


class InventoryFrame(StrictModel):
    frame_id: str = Field(min_length=1)
    uri: str = Field(min_length=1)
    sha256: Sha256
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class InventoryRegion(StrictModel):
    frame_id: str = Field(min_length=1)
    bbox_0_1000: tuple[int, int, int, int]
    visibility: Literal["full", "partial", "heavily_occluded"]
    visible_faces: list[Literal["front", "back", "left", "right", "top", "bottom", "unknown"]]
    occluded_by_candidate_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_normalized_box(self) -> InventoryRegion:
        x0, y0, x1, y1 = self.bbox_0_1000
        if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
            raise ValueError("bbox_0_1000 must be ordered inside [0, 1000]")
        if not self.visible_faces:
            raise ValueError("visible_faces cannot be empty")
        return self


class SceneAssetObservation(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    identity_scope: Literal["observation_not_stable_instance"] = "observation_not_stable_instance"
    category: str = Field(min_length=1)
    name: LocalizedText
    aliases: list[str] = Field(default_factory=list)
    asset_role: Literal[
        "interactive_object", "attached_object", "fixture", "decoration", "structure"
    ]
    granularity: Literal[
        "independent_root_asset",
        "independent_child_asset",
        "merge_into_parent",
        "structure_background",
        "reject_fragment",
    ]
    parent_candidate_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.-]+$")
    moves_with_parent: bool = False
    independently_movable: bool = True
    semantic_unit_rationale: str = Field(min_length=1)
    instance_count: int = Field(default=1, ge=1)
    importance: Literal["primary", "secondary", "structure"]
    structure_role: Literal["none", "wall", "floor", "ceiling", "background"] = "none"
    confidence: float = Field(ge=0, le=1)
    peel_priority: int = Field(default=50, ge=0, le=100)
    evidence_frame_ids: list[str] = Field(min_length=1)
    evidence_regions: list[InventoryRegion] = Field(default_factory=list)
    coverage_status: CoverageStatus
    matched_object_ids: list[str] = Field(default_factory=list)
    completion_needs: list[CompletionNeed] = Field(default_factory=list)
    completion_risks: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_structure_and_coverage(self) -> SceneAssetObservation:
        if self.importance == "structure" and self.structure_role == "none":
            raise ValueError("structure observations require a structure_role")
        if self.importance != "structure" and self.structure_role != "none":
            raise ValueError("only structure observations may declare a structure_role")
        if self.coverage_status == "structure_background" and self.importance != "structure":
            raise ValueError("structure_background coverage requires importance='structure'")
        if self.coverage_status == "modeled_complete" and not self.matched_object_ids:
            raise ValueError("modeled_complete observations require matched_object_ids")
        if self.asset_role == "structure" and self.importance != "structure":
            raise ValueError("asset_role='structure' requires importance='structure'")
        if self.importance == "structure" and self.asset_role != "structure":
            raise ValueError("structure importance requires asset_role='structure'")
        needs_parent = self.granularity in {
            "independent_child_asset",
            "merge_into_parent",
        }
        if needs_parent and not self.parent_candidate_id:
            raise ValueError(f"{self.granularity} requires parent_candidate_id")
        if not needs_parent and self.parent_candidate_id:
            raise ValueError(f"{self.granularity} cannot declare parent_candidate_id")
        if self.parent_candidate_id == self.id:
            raise ValueError("an observation cannot be its own parent")
        if self.moves_with_parent != needs_parent:
            raise ValueError("moves_with_parent must match whether a parent is declared")
        if self.granularity == "merge_into_parent" and self.independently_movable:
            raise ValueError("merge_into_parent observations cannot be independently movable")
        if (
            self.granularity in {"structure_background", "reject_fragment"}
            and self.independently_movable
        ):
            raise ValueError(f"{self.granularity} cannot be independently movable")
        if self.granularity == "structure_background" and self.importance != "structure":
            raise ValueError("structure_background granularity requires structure importance")
        if len(self.completion_needs) != len(set(self.completion_needs)):
            raise ValueError("completion_needs cannot contain duplicates")
        return self


class SceneInventory(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["video2world.scene_inventory"] = "video2world.scene_inventory"
    scene_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    created_at: datetime
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    evidence_frames: list[InventoryFrame] = Field(min_length=1)
    observations: list[SceneAssetObservation] = Field(min_length=1)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_identity_and_evidence(self) -> SceneInventory:
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must include timezone information")
        frame_ids = [frame.frame_id for frame in self.evidence_frames]
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("duplicate inventory frame ids")
        observation_ids = [item.id for item in self.observations]
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("duplicate scene observation ids")
        known_frames = set(frame_ids)
        known_observations = set(observation_ids)
        for item in self.observations:
            unknown = sorted(set(item.evidence_frame_ids) - known_frames)
            if unknown:
                raise ValueError(f"observation {item.id} references unknown frames: {unknown}")
            region_frames = {region.frame_id for region in item.evidence_regions}
            unknown_regions = sorted(region_frames - known_frames)
            if unknown_regions:
                raise ValueError(
                    f"observation {item.id} regions reference unknown frames: {unknown_regions}"
                )
            undeclared_regions = sorted(region_frames - set(item.evidence_frame_ids))
            if undeclared_regions:
                raise ValueError(
                    f"observation {item.id} region frames are absent from evidence_frame_ids: "
                    f"{undeclared_regions}"
                )
            if item.parent_candidate_id and item.parent_candidate_id not in known_observations:
                raise ValueError(
                    f"observation {item.id} references unknown parent: {item.parent_candidate_id}"
                )
        parents = {
            item.id: item.parent_candidate_id
            for item in self.observations
            if item.parent_candidate_id is not None
        }
        for item_id in parents:
            seen: set[str] = set()
            cursor: str | None = item_id
            while cursor is not None:
                if cursor in seen:
                    raise ValueError("interaction hierarchy contains a cycle")
                seen.add(cursor)
                cursor = parents.get(cursor)
        return self


class OcclusionEdge(StrictModel):
    foreground_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    background_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    confidence: float = Field(ge=0, le=1)
    evidence_frame_ids: list[str] = Field(min_length=1)
    evidence_uris: list[str] = Field(default_factory=list)
    ordering_source: Literal["depth", "mask_overlap", "vlm", "combined"]
    verification_status: Literal["observation_only", "geometry_verified"] = "observation_only"

    @model_validator(mode="after")
    def validate_ordering_claim(self) -> OcclusionEdge:
        if self.foreground_id == self.background_id:
            raise ValueError("an occlusion edge cannot point to itself")
        if self.verification_status == "geometry_verified" and self.ordering_source not in {
            "depth",
            "combined",
        }:
            raise ValueError(
                "geometry-verified occlusion ordering requires depth or combined evidence"
            )
        return self


class OcclusionGraph(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["video2world.occlusion_graph"] = "video2world.occlusion_graph"
    scene_id: str = Field(min_length=1)
    inventory_sha256: Sha256
    minimum_confidence: float = Field(default=0.6, ge=0, le=1)
    edges: list[OcclusionEdge] = Field(default_factory=list)
    unresolved_relations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_acyclic_confident_graph(self) -> OcclusionGraph:
        edges = [
            edge
            for edge in self.edges
            if edge.confidence >= self.minimum_confidence
            and edge.verification_status == "geometry_verified"
        ]
        adjacency: dict[str, set[str]] = defaultdict(set)
        indegree: dict[str, int] = defaultdict(int)
        nodes: set[str] = set()
        for edge in edges:
            nodes.update((edge.foreground_id, edge.background_id))
            if edge.background_id not in adjacency[edge.foreground_id]:
                adjacency[edge.foreground_id].add(edge.background_id)
                indegree[edge.background_id] += 1
            indegree.setdefault(edge.foreground_id, 0)
        queue = deque(sorted(node for node in nodes if indegree[node] == 0))
        visited = 0
        while queue:
            node = queue.popleft()
            visited += 1
            for target in sorted(adjacency[node]):
                indegree[target] -= 1
                if indegree[target] == 0:
                    queue.append(target)
        if visited != len(nodes):
            raise ValueError("confident occlusion relations contain a cycle")
        return self


class CompletionAction(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    action: ActionType
    target_ids: list[str] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    idempotency_key: Sha256
    synthetic_output: bool = False
    required_output_roles: list[str] = Field(default_factory=list)
    optional_output_roles: list[str] = Field(default_factory=list)
    acceptance_gates: list[AcceptanceGate] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_output_roles(self) -> CompletionAction:
        for label, roles in (
            ("required", self.required_output_roles),
            ("optional", self.optional_output_roles),
        ):
            if len(roles) != len(set(roles)):
                raise ValueError(f"{label} output roles cannot contain duplicates")
        overlap = sorted(set(self.required_output_roles).intersection(self.optional_output_roles))
        if overlap:
            raise ValueError(f"output roles cannot be both required and optional: {overlap}")
        return self


class CompletedObjectAsset(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    representation_mode: Literal[
        "unified_pbr_glb",
        "separate_render_and_collider",
    ] = "separate_render_and_collider"
    unified_pbr_glb: AssetRef | None = None
    collision_topology: CollisionTopology | None = None
    render_mesh: AssetRef | None = None
    collider: AssetRef | None = None
    geometry_complete_verified: Literal[True] | None = None
    closed_surface_verified: bool | None = None
    object_gaussian: AssetRef | None = None
    object_point_cloud: AssetRef | None = None
    completion_report_uri: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_validated_mesh_first_assets(self) -> CompletedObjectAsset:
        if self.representation_mode == "unified_pbr_glb":
            separate_assets = [
                name for name in ("render_mesh", "collider") if getattr(self, name) is not None
            ]
            if separate_assets:
                raise ValueError(
                    "unified_pbr_glb cannot be mixed with separate object assets: "
                    + ", ".join(separate_assets)
                )
            if self.unified_pbr_glb is None:
                raise ValueError("unified_pbr_glb representation requires unified_pbr_glb asset")
            if self.collision_topology is None:
                raise ValueError("unified_pbr_glb representation requires collision_topology")
            if self.geometry_complete_verified is not True:
                raise ValueError(
                    "unified_pbr_glb representation requires geometry_complete_verified=true"
                )
            validate_unified_pbr_glb_asset(self.unified_pbr_glb, self.collision_topology)
        else:
            if self.unified_pbr_glb is not None or self.collision_topology is not None:
                raise ValueError(
                    "separate_render_and_collider cannot declare unified_pbr_glb or "
                    "collision_topology"
                )
            missing = [name for name in ("render_mesh", "collider") if getattr(self, name) is None]
            if missing:
                raise ValueError("separate_render_and_collider is missing: " + ", ".join(missing))
            if (
                self.geometry_complete_verified is not True
                and self.closed_surface_verified is not True
            ):
                raise ValueError(
                    "separate_render_and_collider requires geometry_complete_verified=true "
                    "or legacy closed_surface_verified=true"
                )
            required = {
                "render_mesh": self.render_mesh,
                "collider": self.collider,
            }
            for name, asset in required.items():
                assert asset is not None
                if asset.role != name:
                    raise ValueError(f"mesh-first object {name} has mismatched role {asset.role!r}")
                if asset.status != "validated":
                    raise ValueError(f"mesh-first object {name} must be validated")
        optional = {
            "object_gaussian": self.object_gaussian,
            "object_point_cloud": self.object_point_cloud,
        }
        for name, asset in optional.items():
            if asset is None:
                continue
            if asset.role != name:
                raise ValueError(
                    f"optional object representation {name} has mismatched role {asset.role!r}"
                )
            if asset.status != "validated":
                raise ValueError(f"optional object representation {name} must be validated")
        return self


class CompletedObjectAssetsManifest(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["video2world.completed_object_assets_manifest"] = (
        "video2world.completed_object_assets_manifest"
    )
    scene_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    created_at: datetime
    status: Literal["passed"] = "passed"
    representation_policy: Literal[
        "unified_pbr_glb_preferred_optional_gaussian",
        "mesh_first_optional_gaussian",
    ] = "unified_pbr_glb_preferred_optional_gaussian"
    objects: list[CompletedObjectAsset] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_manifest(self) -> CompletedObjectAssetsManifest:
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must include timezone information")
        object_ids = [item.id for item in self.objects]
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("completed object ids must be unique")
        return self


class GeometryReviewIssue(StrictModel):
    issue_type: GeometryIssueType
    severity: Literal["blocking", "warning"]
    evidence_view_ids: list[str] = Field(min_length=1)
    explanation: LocalizedText
    retry_prompt_instruction: str | None = None

    @model_validator(mode="after")
    def blocking_issue_requires_remediation(self) -> GeometryReviewIssue:
        if self.severity == "blocking" and not self.retry_prompt_instruction:
            raise ValueError("blocking geometry issues require retry_prompt_instruction")
        return self


class GeometryReview(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["video2world.geometry_review"] = "video2world.geometry_review"
    object_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    attempt: int = Field(ge=1, le=16)
    created_at: datetime
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    source_asset_sha256: Sha256
    turntable_sha256: Sha256
    technical_gates: dict[str, bool] = Field(min_length=1)
    advisory_gates: dict[str, bool] = Field(default_factory=dict)
    acceptance_policy: Literal["usability_first_minor_artifacts"] = (
        "usability_first_minor_artifacts"
    )
    decision: Literal["accept", "retry", "reject"]
    issues: list[GeometryReviewIssue] = Field(default_factory=list)
    retry_prompt: str | None = None
    raw_response_sha256: Sha256

    @model_validator(mode="after")
    def enforce_fail_closed_decision(self) -> GeometryReview:
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must include timezone information")
        blocking = [issue for issue in self.issues if issue.severity == "blocking"]
        failed_technical = sorted(
            name for name, passed in self.technical_gates.items() if not passed
        )
        if self.decision == "accept":
            if blocking or failed_technical:
                raise ValueError("accept is forbidden when blocking or technical gates fail")
            if self.retry_prompt is not None:
                raise ValueError("accepted geometry cannot declare a retry_prompt")
        if self.decision == "retry":
            if not blocking and not failed_technical:
                raise ValueError("retry requires a blocking issue or failed technical gate")
            if not self.retry_prompt:
                raise ValueError("retry requires a detailed retry_prompt")
        if self.decision == "reject" and self.retry_prompt is not None:
            raise ValueError("rejected geometry cannot declare a retry_prompt")
        return self


class CompletionAttempt(StrictModel):
    attempt: int = Field(ge=1)
    seed: int = Field(ge=0)
    prompt_sha256: Sha256
    output_asset_sha256: Sha256
    review_uri: str = Field(min_length=1)
    decision: Literal["accept", "retry", "reject"]


class ObjectCompletionHistory(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["video2world.object_completion_history"] = "video2world.object_completion_history"
    object_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    max_attempts: int = Field(default=3, ge=1, le=8)
    attempts: list[CompletionAttempt] = Field(min_length=1)
    terminal_status: Literal["accepted", "exhausted", "rejected", "in_progress"]

    @model_validator(mode="after")
    def enforce_bounded_retry_history(self) -> ObjectCompletionHistory:
        expected = list(range(1, len(self.attempts) + 1))
        actual = [item.attempt for item in self.attempts]
        if actual != expected:
            raise ValueError(f"completion attempts must be sequential: {actual}")
        if len(self.attempts) > self.max_attempts:
            raise ValueError("completion history exceeds max_attempts")
        accepted = [index for index, item in enumerate(self.attempts) if item.decision == "accept"]
        rejected = [index for index, item in enumerate(self.attempts) if item.decision == "reject"]
        if accepted and accepted != [len(self.attempts) - 1]:
            raise ValueError("no attempt may follow an accepted candidate")
        if rejected and rejected != [len(self.attempts) - 1]:
            raise ValueError("no attempt may follow a rejected candidate")
        if self.terminal_status == "accepted" and not accepted:
            raise ValueError("accepted terminal status requires an accepted final attempt")
        if self.terminal_status == "rejected" and not rejected:
            raise ValueError("rejected terminal status requires a rejected final attempt")
        if self.terminal_status == "exhausted":
            if len(self.attempts) != self.max_attempts:
                raise ValueError("exhausted status requires max_attempts attempts")
            if self.attempts[-1].decision != "retry":
                raise ValueError("exhausted status requires a retry decision on the last attempt")
        if self.terminal_status == "in_progress":
            if len(self.attempts) >= self.max_attempts:
                raise ValueError("in_progress history must have retry budget remaining")
            if self.attempts[-1].decision != "retry":
                raise ValueError("in_progress history requires a retry final decision")
        return self


class CompletionArtifactEvidence(StrictModel):
    """A content-addressed artifact referenced by completion planning or execution."""

    uri: str = Field(min_length=1)
    sha256: Sha256
    size_bytes: int = Field(gt=0)
    acceptance_scope: str | None = None
    lineage_scope: str | None = None
    corrected_full_pipeline: bool | None = None
    promotion_approved: bool | None = None
    canonical_promotion_approved: bool | None = None
    canonical_or_live_manifest_modified: bool | None = None


NEXT_ROUND_CLEAN_PLATE_ACCEPTANCE_SCOPE = "corrected_clean_plate_next_round_source_only"
NEXT_ROUND_CLEAN_PLATE_LINEAGE_SCOPE = "corrected_clean_plate_round_source"
TERMINAL_CLEAN_PLATE_ACCEPTANCE_SCOPE = "corrected_full_pipeline"
TERMINAL_CLEAN_PLATE_LINEAGE_SCOPE = "corrected_full_pipeline"


def _require_scoped_next_round_clean_plate(
    evidence: CompletionArtifactEvidence,
    *,
    context: str,
) -> None:
    if evidence.acceptance_scope != NEXT_ROUND_CLEAN_PLATE_ACCEPTANCE_SCOPE:
        raise ValueError(
            f"{context} acceptance_scope must equal "
            f"{NEXT_ROUND_CLEAN_PLATE_ACCEPTANCE_SCOPE}"
        )
    if evidence.lineage_scope != NEXT_ROUND_CLEAN_PLATE_LINEAGE_SCOPE:
        raise ValueError(
            f"{context} lineage_scope must equal {NEXT_ROUND_CLEAN_PLATE_LINEAGE_SCOPE}"
        )
    if evidence.corrected_full_pipeline is not False:
        raise ValueError(f"{context} must not claim corrected full-pipeline completion")
    if evidence.promotion_approved is not False:
        raise ValueError(f"{context} must not claim promotion approval")
    if evidence.canonical_promotion_approved is not False:
        raise ValueError(f"{context} must not claim canonical promotion approval")
    if evidence.canonical_or_live_manifest_modified is not False:
        raise ValueError(f"{context} must not claim canonical/live manifest mutation")


def _require_scoped_terminal_clean_plate(
    evidence: CompletionArtifactEvidence,
    *,
    context: str,
) -> None:
    if evidence.acceptance_scope != TERMINAL_CLEAN_PLATE_ACCEPTANCE_SCOPE:
        raise ValueError(
            f"{context} acceptance_scope must equal {TERMINAL_CLEAN_PLATE_ACCEPTANCE_SCOPE}"
        )
    if evidence.lineage_scope != TERMINAL_CLEAN_PLATE_LINEAGE_SCOPE:
        raise ValueError(
            f"{context} lineage_scope must equal {TERMINAL_CLEAN_PLATE_LINEAGE_SCOPE}"
        )
    if evidence.corrected_full_pipeline is not True:
        raise ValueError(f"{context} must claim corrected full-pipeline completion")
    if evidence.promotion_approved is not True:
        raise ValueError(f"{context} must claim promotion approval")
    if evidence.canonical_promotion_approved is not True:
        raise ValueError(f"{context} must claim canonical promotion approval")
    if evidence.canonical_or_live_manifest_modified is not False:
        raise ValueError(f"{context} must not claim canonical/live manifest mutation")


class CompletionPlanEvidence(CompletionArtifactEvidence):
    """Evidence binding with an explicit, narrowly scoped planning claim."""

    role: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    claim_scope: str = Field(min_length=1)
    usage: Literal["evidence_only", "next_round_input", "display_only"] = "evidence_only"


class CompletionRound(StrictModel):
    index: int = Field(ge=1)
    kind: Literal["object_layer", "final_background"]
    target_ids: list[str]
    input_clean_plate_round: int = Field(ge=0)
    actions: list[CompletionAction] = Field(min_length=1)
    status: Literal[
        "planned",
        "running",
        "passed",
        "passed_with_limitations",
        "failed",
        "blocked",
    ] = "planned"
    input_clean_plate_uri: str | None = None
    output_clean_plate_uri: str | None = None
    quality_report_uri: str | None = None
    status_reason: str | None = None
    acceptance_scope: (
        Literal["full_round", "current_demo_only", "usable_for_reinspection_only"] | None
    ) = None
    evidence: list[CompletionPlanEvidence] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    next_layer_candidate_ids: list[str] = Field(default_factory=list)
    next_layer_candidates_evidence_uri: str | None = None

    @model_validator(mode="after")
    def validate_round_shape(self) -> CompletionRound:
        if self.kind == "object_layer" and not self.target_ids:
            raise ValueError("object_layer rounds require targets")
        if self.kind == "final_background" and self.target_ids:
            raise ValueError("final_background round cannot contain object targets")
        action_ids = [action.id for action in self.actions]
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("completion action ids must be unique within a round")
        known = set(action_ids)
        for action in self.actions:
            unknown = sorted(set(action.prerequisites) - known)
            if unknown:
                raise ValueError(f"action {action.id} has unknown prerequisites: {unknown}")
        if self.status in {"passed", "passed_with_limitations"} and not self.quality_report_uri:
            raise ValueError("successful rounds require a quality_report_uri")
        if self.status == "passed_with_limitations":
            if not self.status_reason:
                raise ValueError("passed_with_limitations rounds require a status_reason")
            if not self.evidence:
                raise ValueError(
                    "passed_with_limitations rounds require content-addressed evidence"
                )
            if not self.limitations:
                raise ValueError("passed_with_limitations rounds require explicit limitations")
            if self.acceptance_scope not in {
                "current_demo_only",
                "usable_for_reinspection_only",
            }:
                raise ValueError(
                    "passed_with_limitations rounds require a limited acceptance_scope"
                )
        if len(self.next_layer_candidate_ids) != len(set(self.next_layer_candidate_ids)):
            raise ValueError("next_layer_candidate_ids cannot contain duplicates")
        if bool(self.next_layer_candidate_ids) != bool(self.next_layer_candidates_evidence_uri):
            raise ValueError(
                "next-layer candidates and their evidence URI must either both be set or "
                "both be absent"
            )
        return self


class LayeredCompletionPlan(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["video2world.layered_completion_plan"] = "video2world.layered_completion_plan"
    scene_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    created_at: datetime
    inventory_sha256: Sha256
    occlusion_graph_sha256: Sha256
    round_status_input_uri: str | None = None
    round_status_input_sha256: Sha256 | None = None
    strategy: Literal["front_to_back_clean_plate"] = "front_to_back_clean_plate"
    ordering_policy: Literal[
        "geometry_layers_then_priority",
        "explicit_topological_sequence",
    ] = "geometry_layers_then_priority"
    audit_iteration: int = Field(default=1, ge=1)
    max_audit_iterations: int = Field(default=4, ge=1, le=32)
    max_rounds: int = Field(default=32, ge=1, le=64)
    max_parallel_targets_per_round: int = Field(default=1, ge=1, le=16)
    rounds: list[CompletionRound] = Field(min_length=1)
    terminal_status: Literal[
        "planned", "running", "passed", "passed_with_limitations", "blocked"
    ] = "planned"
    stop_conditions: list[str] = Field(min_length=1)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def enforce_sequential_rounds(self) -> LayeredCompletionPlan:
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must include timezone information")
        if (self.round_status_input_uri is None) != (self.round_status_input_sha256 is None):
            raise ValueError(
                "round status input URI and sha256 must either both be set or both be absent"
            )
        expected = list(range(1, len(self.rounds) + 1))
        actual = [item.index for item in self.rounds]
        if actual != expected:
            raise ValueError(
                f"completion rounds must be sequential: expected {expected}, got {actual}"
            )
        if len(self.rounds) > self.max_rounds:
            raise ValueError("completion plan exceeds max_rounds")
        if self.audit_iteration > self.max_audit_iterations:
            raise ValueError("completion plan exceeds max_audit_iterations")
        if self.rounds[-1].kind != "final_background":
            raise ValueError("the last round must be final_background")
        seen_targets: set[str] = set()
        for round_item in self.rounds:
            if round_item.input_clean_plate_round != round_item.index - 1:
                raise ValueError("each round must consume the immediately preceding clean plate")
            duplicates = sorted(seen_targets.intersection(round_item.target_ids))
            if duplicates:
                raise ValueError(f"objects cannot be peeled twice: {duplicates}")
            seen_targets.update(round_item.target_ids)
        for index, round_item in enumerate(self.rounds):
            if round_item.status in {
                "running",
                "passed",
                "passed_with_limitations",
                "failed",
            }:
                previous = self.rounds[:index]
                if any(
                    item.status not in {"passed", "passed_with_limitations"} for item in previous
                ):
                    raise ValueError("a round cannot advance before every previous round passes")
            if (
                round_item.status in {"passed", "passed_with_limitations"}
                and not round_item.output_clean_plate_uri
            ):
                raise ValueError("successful rounds require output_clean_plate_uri")
            if index > 0 and self.round_status_input_uri is not None:
                previous_output = self.rounds[index - 1].output_clean_plate_uri
                if round_item.input_clean_plate_uri != previous_output:
                    raise ValueError(
                        "evidence-bound rounds must reference the exact preceding output clean "
                        "plate URI"
                    )
            if index + 1 < len(self.rounds) and round_item.next_layer_candidate_ids:
                next_targets = set(self.rounds[index + 1].target_ids)
                if not next_targets.issubset(round_item.next_layer_candidate_ids):
                    raise ValueError(
                        "the next operational round must be selected from the reinspection "
                        "candidate set"
                    )
        all_rounds_passed = all(item.status == "passed" for item in self.rounds)
        all_rounds_successful = all(
            item.status in {"passed", "passed_with_limitations"} for item in self.rounds
        )
        any_limited = any(item.status == "passed_with_limitations" for item in self.rounds)
        if self.terminal_status == "passed" and not all_rounds_passed:
            raise ValueError("terminal passed status requires every completion round to pass")
        if all_rounds_passed and self.terminal_status != "passed":
            raise ValueError("all passed completion rounds require terminal_status='passed'")
        if self.terminal_status == "passed_with_limitations" and not (
            all_rounds_successful and any_limited
        ):
            raise ValueError(
                "terminal passed_with_limitations requires every round to succeed and at least "
                "one limited round"
            )
        if (
            all_rounds_successful
            and any_limited
            and self.terminal_status != ("passed_with_limitations")
        ):
            raise ValueError(
                "all successful rounds with limitations require "
                "terminal_status='passed_with_limitations'"
            )
        return self


class LayeredCompletionRoundReceipt(StrictModel):
    """Evidence that one peel round consumed and replaced a clean scene state."""

    index: int = Field(ge=1)
    kind: Literal["object_layer", "final_background"]
    target_ids: list[str]
    status: Literal["passed"] = "passed"
    input_clean_plate: CompletionArtifactEvidence
    scene_audit_receipt: CompletionArtifactEvidence
    sam3_receipt: CompletionArtifactEvidence
    output_clean_plate: CompletionArtifactEvidence
    quality_report: CompletionArtifactEvidence
    object_completion_receipts: list[CompletionArtifactEvidence] = Field(default_factory=list)
    background_rebuild_receipt: CompletionArtifactEvidence | None = None
    acceptance_gates: dict[str, bool] = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)
    synthetic_outputs: Literal[False] = False

    @model_validator(mode="after")
    def validate_round_evidence(self) -> LayeredCompletionRoundReceipt:
        if any(not passed for passed in self.acceptance_gates.values()):
            raise ValueError("passed completion rounds cannot contain failed acceptance gates")
        if self.kind == "object_layer":
            if not self.target_ids:
                raise ValueError("object_layer receipts require target_ids")
            if not self.object_completion_receipts:
                raise ValueError("object_layer receipts require object completion evidence")
            if self.background_rebuild_receipt is not None:
                raise ValueError("object_layer receipts cannot claim final background rebuild")
            _require_scoped_next_round_clean_plate(
                self.output_clean_plate,
                context=f"round {self.index} output_clean_plate",
            )
        else:
            if self.target_ids:
                raise ValueError("final_background receipts cannot contain target_ids")
            if self.object_completion_receipts:
                raise ValueError("final_background receipts cannot contain object completions")
            if self.background_rebuild_receipt is None:
                raise ValueError("final_background receipts require background rebuild evidence")
            _require_scoped_terminal_clean_plate(
                self.output_clean_plate,
                context=f"round {self.index} output_clean_plate",
            )
        return self


class LayeredCompletionExecutionReport(StrictModel):
    """Terminal evidence for a real, fully iterated front-to-back completion run."""

    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["video2world.layered_completion_execution_report"] = (
        "video2world.layered_completion_execution_report"
    )
    lineage_scope: Literal["corrected_full_pipeline"]
    scene_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    created_at: datetime
    strategy: Literal["front_to_back_clean_plate"] = "front_to_back_clean_plate"
    status: Literal["passed"] = "passed"
    terminal_status: Literal["passed"] = "passed"
    promotion_allowed: Literal[True] = True
    all_acceptance_gates_passed: Literal[True] = True
    completion_plan_sha256: Sha256
    planned_target_ids: list[str] = Field(min_length=1)
    initial_clean_plate: CompletionArtifactEvidence
    rounds: list[LayeredCompletionRoundReceipt] = Field(min_length=1)
    final_clean_plate: CompletionArtifactEvidence
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_front_to_back_chain(self) -> LayeredCompletionExecutionReport:
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must include timezone information")
        expected = list(range(1, len(self.rounds) + 1))
        actual = [item.index for item in self.rounds]
        if actual != expected:
            raise ValueError(f"completion receipt rounds must be sequential: {actual}")
        if self.rounds[-1].kind != "final_background":
            raise ValueError("the terminal completion receipt must be final_background")
        if len(self.planned_target_ids) != len(set(self.planned_target_ids)):
            raise ValueError("planned_target_ids cannot contain duplicates")
        previous = self.initial_clean_plate
        seen_targets: set[str] = set()
        clean_plate_digests: set[str] = {self.initial_clean_plate.sha256}
        execution_receipts: dict[str, str] = {}

        def register_execution_receipt(
            evidence: CompletionArtifactEvidence,
            *,
            context: str,
        ) -> None:
            if evidence.sha256 in clean_plate_digests:
                raise ValueError(f"{context} must not reuse a clean plate artifact")
            previous_context = execution_receipts.get(evidence.sha256)
            if previous_context is not None:
                raise ValueError(
                    f"{context} must be distinct from {previous_context}; "
                    "completion execution evidence cannot be reused across roles or rounds"
                )
            execution_receipts[evidence.sha256] = context

        for round_item in self.rounds:
            if round_item.index > 1:
                _require_scoped_next_round_clean_plate(
                    round_item.input_clean_plate,
                    context=f"round {round_item.index} input_clean_plate",
                )
            if (
                round_item.input_clean_plate.sha256 != previous.sha256
                or round_item.input_clean_plate.size_bytes != previous.size_bytes
            ):
                raise ValueError(
                    f"round {round_item.index} does not consume the preceding clean plate"
                )
            duplicates = sorted(seen_targets.intersection(round_item.target_ids))
            if duplicates:
                raise ValueError(f"objects cannot be peeled twice: {duplicates}")
            register_execution_receipt(
                round_item.scene_audit_receipt,
                context=f"round {round_item.index} scene_audit_receipt",
            )
            register_execution_receipt(
                round_item.sam3_receipt,
                context=f"round {round_item.index} sam3_receipt",
            )
            register_execution_receipt(
                round_item.quality_report,
                context=f"round {round_item.index} quality_report",
            )
            for receipt_index, receipt in enumerate(
                round_item.object_completion_receipts,
                start=1,
            ):
                register_execution_receipt(
                    receipt,
                    context=(
                        f"round {round_item.index} object_completion_receipts"
                        f"[{receipt_index}]"
                    ),
                )
            if round_item.background_rebuild_receipt is not None:
                register_execution_receipt(
                    round_item.background_rebuild_receipt,
                    context=f"round {round_item.index} background_rebuild_receipt",
                )
            seen_targets.update(round_item.target_ids)
            if round_item.kind == "object_layer":
                _require_scoped_next_round_clean_plate(
                    round_item.output_clean_plate,
                    context=f"round {round_item.index} output_clean_plate",
                )
            else:
                _require_scoped_terminal_clean_plate(
                    round_item.output_clean_plate,
                    context=f"round {round_item.index} output_clean_plate",
                )
            reused_context = execution_receipts.get(round_item.output_clean_plate.sha256)
            if reused_context is not None:
                raise ValueError(
                    f"round {round_item.index} output_clean_plate must be distinct from "
                    f"{reused_context}"
                )
            previous = round_item.output_clean_plate
            clean_plate_digests.add(round_item.output_clean_plate.sha256)
        if seen_targets != set(self.planned_target_ids):
            raise ValueError(
                "completion receipt targets differ from the completion plan target list"
            )
        if (
            self.final_clean_plate.sha256 != previous.sha256
            or self.final_clean_plate.size_bytes != previous.size_bytes
        ):
            raise ValueError("final_clean_plate does not match the terminal round output")
        _require_scoped_terminal_clean_plate(
            self.final_clean_plate,
            context="final_clean_plate",
        )
        return self


def _load_json(path: str | Path) -> dict[str, object]:
    value = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def load_scene_inventory(path: str | Path) -> SceneInventory:
    return SceneInventory.model_validate(_load_json(path))


def load_occlusion_graph(path: str | Path) -> OcclusionGraph:
    return OcclusionGraph.model_validate(_load_json(path))


def load_layered_completion_plan(path: str | Path) -> LayeredCompletionPlan:
    return LayeredCompletionPlan.model_validate(_load_json(path))


def _action(
    round_index: int,
    ordinal: int,
    action: ActionType,
    *,
    targets: list[str],
    prerequisites: list[str] | None = None,
    synthetic: bool = False,
    roles: list[str] | None = None,
    optional_roles: list[str] | None = None,
    gates: list[AcceptanceGate] | None = None,
    notes: list[str] | None = None,
) -> CompletionAction:
    action_id = f"r{round_index:02d}-{ordinal:02d}-{action.replace('_', '-')}"
    return CompletionAction(
        id=action_id,
        action=action,
        target_ids=targets,
        prerequisites=prerequisites or [],
        idempotency_key=digest_json(
            {
                "id": action_id,
                "action": action,
                "targets": targets,
                "prerequisites": prerequisites or [],
                "synthetic": synthetic,
                "roles": roles or [],
                "optional_roles": optional_roles or [],
                "notes": notes or [],
            }
        ),
        synthetic_output=synthetic,
        required_output_roles=roles or [],
        optional_output_roles=optional_roles or [],
        acceptance_gates=gates or [],
        notes=notes or [],
    )


def _topological_object_layers(
    inventory: SceneInventory, graph: OcclusionGraph
) -> list[list[SceneAssetObservation]]:
    candidates = {
        item.id: item
        for item in inventory.observations
        if item.importance != "structure"
        and item.granularity in {"independent_root_asset", "independent_child_asset"}
        and item.coverage_status != "ignored"
        and (item.coverage_status != "modeled_complete" or item.completion_needs)
    }
    adjacency: dict[str, set[str]] = defaultdict(set)
    indegree = {item_id: 0 for item_id in candidates}
    unverified_relations: list[str] = []
    for edge in graph.edges:
        if edge.confidence < graph.minimum_confidence:
            continue
        if edge.foreground_id not in candidates or edge.background_id not in candidates:
            continue
        if edge.verification_status != "geometry_verified":
            unverified_relations.append(f"{edge.foreground_id}->{edge.background_id}")
            continue
        if edge.background_id not in adjacency[edge.foreground_id]:
            adjacency[edge.foreground_id].add(edge.background_id)
            indegree[edge.background_id] += 1
    if unverified_relations:
        raise ValueError(
            "completion ordering requires geometry-verified depth relations; "
            f"observation-only edges remain: {sorted(unverified_relations)}"
        )

    def order(item_id: str) -> tuple[int, str]:
        return (-candidates[item_id].peel_priority, item_id)

    current = sorted((item_id for item_id, degree in indegree.items() if degree == 0), key=order)
    layers: list[list[SceneAssetObservation]] = []
    visited = 0
    while current:
        layers.append([candidates[item_id] for item_id in current])
        following: list[str] = []
        for item_id in current:
            visited += 1
            for target in sorted(adjacency[item_id]):
                indegree[target] -= 1
                if indegree[target] == 0:
                    following.append(target)
        current = sorted(following, key=order)
    if visited != len(candidates):
        raise ValueError("object occlusion graph could not be layered")
    return layers


def _round_actions(index: int, observations: list[SceneAssetObservation]) -> list[CompletionAction]:
    targets = [item.id for item in observations]
    needs = {need for item in observations for need in item.completion_needs}
    for item in observations:
        if item.coverage_status != "missing":
            continue
        needs.update(MISSING_OBJECT_COMPLETION_NEEDS)
        if item.instance_count > 1:
            needs.add("instance_split")
    actions: list[CompletionAction] = []

    def add(
        action: ActionType,
        *,
        synthetic: bool = False,
        roles: list[str] | None = None,
        optional_roles: list[str] | None = None,
        gates: list[AcceptanceGate] | None = None,
        notes: list[str] | None = None,
    ) -> str:
        prerequisite = [actions[-1].id] if actions else []
        item = _action(
            index,
            len(actions) + 1,
            action,
            targets=targets,
            prerequisites=prerequisite,
            synthetic=synthetic,
            roles=roles,
            optional_roles=optional_roles,
            gates=gates,
            notes=notes,
        )
        actions.append(item)
        return item.id

    if "instance_split" in needs:
        add(
            "split_instances",
            roles=["instance_manifest"],
            gates=["segmentation_evidence", "instance_identity"],
        )
    if "segmentation" in needs:
        add("segment", roles=["masks_manifest"], gates=["segmentation_evidence"])
    if "multi_view_lift" in needs:
        add(
            "lift_to_3d",
            roles=["object_clouds_manifest"],
            gates=["multi_view_projection"],
        )
    if {"backside_geometry", "render_asset"}.intersection(needs):
        add(
            "complete_object",
            synthetic=True,
            roles=["unified_pbr_glb", "object_completion_report"],
            optional_roles=[
                "render_mesh",
                "collider",
                "object_gaussian",
                "object_point_cloud",
            ],
            gates=[
                "geometry_completeness",
                "front_silhouette",
                "front_observation_fidelity",
                "backside_nonempty",
                "six_view_coverage",
                "appearance_fidelity",
                "six_side_style_consistency",
                "instance_identity",
                "vlm_geometry_review",
            ],
            notes=[
                "Prefer one validated PBR GLB for rendering, object logic, and collision; "
                "separate render_mesh/collider outputs are legacy compatibility only."
            ],
        )
    if "collider" in needs:
        add(
            "build_collider",
            roles=["collision_report"],
            gates=["collision"],
            notes=[
                "Validate and reuse the complete_object unified_pbr_glb collision topology; "
                "do not generate a collider proxy."
            ],
        )
    add(
        "place_object",
        roles=["aligned_objects_manifest", "interpenetration_report"],
        gates=["placement_alignment", "scene_interpenetration"],
    )
    add(
        "clean_plate",
        synthetic=True,
        roles=["clean_plate_frames", "clean_plate_manifest"],
        gates=[
            "mask_outside_unchanged",
            "cross_view_consistency",
            "revealed_background",
        ],
    )
    add("reinspect", roles=["next_scene_inventory", "next_occlusion_graph"])
    add("validate_round", roles=["round_quality_report"])
    return actions


def build_layered_completion_plan(
    inventory: SceneInventory,
    graph: OcclusionGraph,
    *,
    created_at: datetime,
    max_rounds: int = 32,
    max_parallel_targets_per_round: int = 1,
    target_order: list[str] | None = None,
) -> LayeredCompletionPlan:
    if not 1 <= max_parallel_targets_per_round <= 16:
        raise ValueError("max_parallel_targets_per_round must be between 1 and 16")
    inventory_payload = inventory.model_dump(mode="json")
    inventory_sha = digest_json(inventory_payload)
    if graph.scene_id != inventory.scene_id:
        raise ValueError("inventory and occlusion graph scene_id differ")
    if graph.inventory_sha256 != inventory_sha:
        raise ValueError("occlusion graph does not reference the supplied inventory")
    known_observations = {item.id for item in inventory.observations}
    known_frames = {item.frame_id for item in inventory.evidence_frames}
    for edge in graph.edges:
        unknown_observations = sorted({edge.foreground_id, edge.background_id} - known_observations)
        if unknown_observations:
            raise ValueError(
                f"occlusion graph references unknown observations: {unknown_observations}"
            )
        unknown_frames = sorted(set(edge.evidence_frame_ids) - known_frames)
        if unknown_frames:
            raise ValueError(
                f"occlusion edge {edge.foreground_id}->{edge.background_id} references "
                f"unknown inventory frames: {unknown_frames}"
            )
    object_layers = _topological_object_layers(inventory, graph)
    ordering_policy: Literal[
        "geometry_layers_then_priority",
        "explicit_topological_sequence",
    ] = "geometry_layers_then_priority"
    if target_order is not None:
        if max_parallel_targets_per_round != 1:
            raise ValueError("explicit target_order requires max_parallel_targets_per_round=1")
        candidates = {item.id: item for layer in object_layers for item in layer}
        if len(target_order) != len(set(target_order)):
            raise ValueError("explicit target_order cannot contain duplicates")
        unknown = sorted(set(target_order) - set(candidates))
        missing = sorted(set(candidates) - set(target_order))
        if unknown or missing:
            raise ValueError(
                "explicit target_order must contain every completion candidate exactly once; "
                f"unknown={unknown}, missing={missing}"
            )
        position = {item_id: index for index, item_id in enumerate(target_order)}
        for edge in graph.edges:
            if (
                edge.confidence < graph.minimum_confidence
                or edge.verification_status != "geometry_verified"
                or edge.foreground_id not in candidates
                or edge.background_id not in candidates
            ):
                continue
            if position[edge.foreground_id] >= position[edge.background_id]:
                raise ValueError(
                    "explicit target_order violates geometry-verified occlusion edge "
                    f"{edge.foreground_id}->{edge.background_id}"
                )
        object_layers = [[candidates[item_id]] for item_id in target_order]
        ordering_policy = "explicit_topological_sequence"
    rounds: list[CompletionRound] = []
    for observations in object_layers:
        for offset in range(0, len(observations), max_parallel_targets_per_round):
            batch = observations[offset : offset + max_parallel_targets_per_round]
            index = len(rounds) + 1
            rounds.append(
                CompletionRound(
                    index=index,
                    kind="object_layer",
                    target_ids=[item.id for item in batch],
                    input_clean_plate_round=index - 1,
                    actions=_round_actions(index, batch),
                )
            )
    background_index = len(rounds) + 1
    background_actions = [
        _action(
            background_index,
            1,
            "rebuild_background",
            targets=[],
            synthetic=True,
            roles=["clean_scene_gaussian", "clean_scene_mesh", "background_report"],
            gates=[
                "mask_outside_unchanged",
                "cross_view_consistency",
                "revealed_background",
                "background_depth_normal",
            ],
            notes=[
                "Generated visual background is not physical truth.",
                "Collision geometry remains separately validated.",
            ],
        ),
        _action(
            background_index,
            2,
            "validate_round",
            targets=[],
            prerequisites=[f"r{background_index:02d}-01-rebuild-background"],
            roles=["round_quality_report"],
            gates=["browser_interaction"],
        ),
    ]
    rounds.append(
        CompletionRound(
            index=background_index,
            kind="final_background",
            target_ids=[],
            input_clean_plate_round=background_index - 1,
            actions=background_actions,
        )
    )
    return LayeredCompletionPlan(
        scene_id=inventory.scene_id,
        run_id=inventory.run_id,
        created_at=created_at,
        inventory_sha256=inventory_sha,
        occlusion_graph_sha256=digest_json(graph.model_dump(mode="json")),
        ordering_policy=ordering_policy,
        max_rounds=max_rounds,
        max_parallel_targets_per_round=max_parallel_targets_per_round,
        rounds=rounds,
        stop_conditions=[
            "No unmodeled primary or secondary object remains after VLM reinspect.",
            "Every prior round passed mask-outside-change and cross-view quality gates.",
            "Only structural background roles remain for the final rebuild.",
        ],
        limitations=[
            "Backside geometry and clean plates are generated hypotheses, not observed surfaces.",
            "A failed round blocks every deeper occlusion layer.",
            *(
                [
                    "The explicit target sequence is an operational schedule constrained by "
                    "verified graph edges; it does not promote unresolved relations to verified "
                    "occlusions."
                ]
                if target_order is not None
                else []
            ),
        ],
    )


def write_layered_completion_plan(path: str | Path, plan: LayeredCompletionPlan) -> None:
    atomic_write_json(Path(path).expanduser().resolve(), plan.model_dump(mode="json"))
