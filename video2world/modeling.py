"""Typed artifacts for the Video2World vNext object-modeling pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from video2world.completion import CompletedObjectAsset
from video2world.models import AssetRef, Bounds2D, PhysicalProperties, Sha256, StrictModel

ProposalBackend = Literal["groundingdino", "qwen_vl_box_proposal", "manual_verified"]
ProposalFeatureEncoder = Literal["none", "dinov2", "dinov3"]
SegmentationBackend = Literal["sam3", "sam3_i"]
ObjectCompletionProvider = Literal[
    "observed_multiview",
    "stream3d",
    "trellis",
    "trellis2",
    "sam3d",
    "hunyuan3d_2_1",
    "in_place_3d_fixer",
]
CompletionInputMode = Literal[
    "observed_multiview",
    "assembled_observation",
    "generated_reference",
    "in_place_observed_point_cloud",
]


def _require_timezone(value: datetime, *, label: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must include timezone information")


def _require_unique(values: list[str], *, label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")


class ObjectProposal(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    object_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    category: str = Field(min_length=1)
    frame_id: str = Field(min_length=1)
    bbox: Bounds2D
    backend: ProposalBackend
    feature_encoder: ProposalFeatureEncoder = "none"
    confidence: float = Field(ge=0, le=1)
    prompt: str = Field(min_length=1)

    @model_validator(mode="after")
    def reject_feature_encoder_as_detector(self) -> ObjectProposal:
        if self.backend == "manual_verified" and self.confidence < 1.0:
            raise ValueError("manual_verified proposals require confidence=1")
        return self


class ObjectProposalsManifest(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["video2world.object_proposals_manifest"] = "video2world.object_proposals_manifest"
    scene_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    created_at: datetime
    proposals: list[ObjectProposal] = Field(min_length=1)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_manifest(self) -> ObjectProposalsManifest:
        _require_timezone(self.created_at, label="object proposals created_at")
        _require_unique([item.id for item in self.proposals], label="proposal ids")
        return self


class ComponentContract(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    category: str = Field(min_length=1)
    short_prompt: str = Field(min_length=1)
    required: bool = True
    evidence_scope: Literal["observed", "completion_only"]


class QwenObjectContract(StrictModel):
    object_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    category: str = Field(min_length=1)
    detailed_description: str = Field(min_length=1)
    positive_prompt: str = Field(min_length=1)
    negative_prompt: str = Field(min_length=1)
    proposal_ids: list[str] = Field(min_length=1)
    components: list[ComponentContract] = Field(min_length=1)
    excluded_concepts: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_components(self) -> QwenObjectContract:
        _require_unique([item.id for item in self.components], label="component ids")
        _require_unique(self.proposal_ids, label="proposal ids")
        if not any(item.evidence_scope == "observed" for item in self.components):
            raise ValueError("an object contract requires at least one observed component")
        return self


class QwenObjectContractsManifest(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["video2world.qwen_object_contracts_manifest"] = (
        "video2world.qwen_object_contracts_manifest"
    )
    scene_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    created_at: datetime
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    proposal_manifest_sha256: Sha256
    objects: list[QwenObjectContract] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_manifest(self) -> QwenObjectContractsManifest:
        _require_timezone(self.created_at, label="Qwen object contracts created_at")
        _require_unique([item.object_id for item in self.objects], label="object contract ids")
        return self


class ComponentMaskRecord(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    object_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    component_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    frame_id: str = Field(min_length=1)
    backend: SegmentationBackend
    prompt: str = Field(min_length=1)
    bbox: Bounds2D
    mask: AssetRef
    observation_scope: Literal["observed_pixels_only"] = "observed_pixels_only"

    @model_validator(mode="after")
    def validate_mask_role(self) -> ComponentMaskRecord:
        if self.mask.role != "component_mask":
            raise ValueError("component mask asset role must be component_mask")
        return self


class ComponentMasksManifest(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["video2world.component_masks_manifest"] = "video2world.component_masks_manifest"
    scene_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    created_at: datetime
    qwen_contracts_sha256: Sha256
    records: list[ComponentMaskRecord] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_manifest(self) -> ComponentMasksManifest:
        _require_timezone(self.created_at, label="component masks created_at")
        _require_unique([item.id for item in self.records], label="component mask ids")
        return self


class SemanticObjectLift(StrictModel):
    object_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    component_mask_ids: list[str] = Field(min_length=1)
    observed_point_cloud: AssetRef
    observed_surface: AssetRef | None = None
    coordinate_frame: str = Field(min_length=1)
    units: Literal["meters", "scene_scale_not_metric", "native"]
    depth_manifest_sha256: Sha256
    point_prior_sha256: Sha256
    scene_gaussian_sha256: Sha256
    observed_surface_complete: Literal[False] = False

    @model_validator(mode="after")
    def validate_observation_roles(self) -> SemanticObjectLift:
        if self.observed_point_cloud.role != "observed_object_point_cloud":
            raise ValueError(
                "semantic lifting point cloud role must be observed_object_point_cloud"
            )
        if self.observed_surface is not None and self.observed_surface.role != "observed_surface":
            raise ValueError("semantic lifting surface role must be observed_surface")
        _require_unique(self.component_mask_ids, label="component mask ids")
        return self


class SemanticLiftingManifest(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["video2world.semantic_lifting_manifest"] = "video2world.semantic_lifting_manifest"
    scene_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    created_at: datetime
    objects: list[SemanticObjectLift] = Field(min_length=1)
    semantic_gaussian: AssetRef

    @model_validator(mode="after")
    def validate_manifest(self) -> SemanticLiftingManifest:
        _require_timezone(self.created_at, label="semantic lifting created_at")
        _require_unique([item.object_id for item in self.objects], label="lifted object ids")
        if self.semantic_gaussian.role != "semantic_gaussian":
            raise ValueError("semantic Gaussian asset role must be semantic_gaussian")
        return self


class ComponentAssemblyRecord(StrictModel):
    object_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    observed_component_ids: list[str] = Field(min_length=1)
    completion_only_component_ids: list[str] = Field(default_factory=list)
    condition: AssetRef
    assembly_scope: Literal["observed_union_not_amodal"] = "observed_union_not_amodal"

    @model_validator(mode="after")
    def validate_assembly(self) -> ComponentAssemblyRecord:
        _require_unique(self.observed_component_ids, label="observed component ids")
        _require_unique(self.completion_only_component_ids, label="completion-only component ids")
        if set(self.observed_component_ids).intersection(self.completion_only_component_ids):
            raise ValueError("observed and completion-only components must be disjoint")
        if self.condition.role != "assembled_object_condition":
            raise ValueError("assembly condition role must be assembled_object_condition")
        return self


class ComponentAssemblyManifest(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["video2world.component_assembly_manifest"] = (
        "video2world.component_assembly_manifest"
    )
    scene_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    created_at: datetime
    records: list[ComponentAssemblyRecord] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_manifest(self) -> ComponentAssemblyManifest:
        _require_timezone(self.created_at, label="component assembly created_at")
        _require_unique([item.object_id for item in self.records], label="assembled object ids")
        return self


class ImageCompletionRecord(StrictModel):
    object_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    status: Literal["not_selected", "candidate", "accepted"]
    source_condition_sha256: Sha256
    model: str | None = Field(default=None, min_length=1)
    prompt_sha256: Sha256 | None = None
    generated_reference: AssetRef | None = None
    generated_region_mask: AssetRef | None = None
    provenance_scope: Literal["generated_reference_not_observation"] = (
        "generated_reference_not_observation"
    )

    @model_validator(mode="after")
    def validate_generation(self) -> ImageCompletionRecord:
        generated = (self.generated_reference, self.generated_region_mask)
        if self.status == "not_selected":
            if self.model is not None or self.prompt_sha256 is not None or any(generated):
                raise ValueError("not_selected image completion cannot declare generated outputs")
        elif self.model is None or self.prompt_sha256 is None or not all(generated):
            raise ValueError("selected image completion requires model, prompt, image, and mask")
        elif self.generated_reference is not None and self.generated_region_mask is not None:
            if self.generated_reference.role != "generated_object_reference":
                raise ValueError(
                    "generated reference asset role must be generated_object_reference"
                )
            if self.generated_region_mask.role != "generated_region_mask":
                raise ValueError("generated region mask role must be generated_region_mask")
        return self


class ImageCompletionManifest(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["video2world.image_completion_manifest"] = "video2world.image_completion_manifest"
    scene_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    created_at: datetime
    records: list[ImageCompletionRecord] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_manifest(self) -> ImageCompletionManifest:
        _require_timezone(self.created_at, label="image completion created_at")
        _require_unique(
            [item.object_id for item in self.records], label="image completion object ids"
        )
        return self


class ObjectCompletionCandidate(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    object_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    provider: ObjectCompletionProvider
    input_mode: CompletionInputMode
    source_sha256: Sha256
    coordinate_frame: str = Field(min_length=1)
    mesh: AssetRef | None = None
    object_gaussian: AssetRef | None = None
    object_point_cloud: AssetRef | None = None
    generation_status: Literal["generated", "failed"]
    room_alignment_status: Literal["not_tested", "passed", "failed"] = "not_tested"
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_candidate(self) -> ObjectCompletionCandidate:
        assets = (self.mesh, self.object_gaussian, self.object_point_cloud)
        if self.generation_status == "generated" and not any(assets):
            raise ValueError("generated completion candidate requires an output asset")
        if self.generation_status == "failed" and any(assets):
            raise ValueError("failed completion candidate cannot declare accepted output assets")
        if self.provider == "in_place_3d_fixer" and self.input_mode != (
            "in_place_observed_point_cloud"
        ):
            raise ValueError("in_place_3d_fixer requires in-place observed point-cloud input")
        return self


class ObjectCompletionCandidatesManifest(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["video2world.object_completion_candidates_manifest"] = (
        "video2world.object_completion_candidates_manifest"
    )
    scene_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    created_at: datetime
    candidates: list[ObjectCompletionCandidate] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_manifest(self) -> ObjectCompletionCandidatesManifest:
        _require_timezone(self.created_at, label="object completion candidates created_at")
        _require_unique([item.id for item in self.candidates], label="completion candidate ids")
        return self


class MeshPostprocessRecord(StrictModel):
    object_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    selected_candidate_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    repair_backend: str = Field(min_length=1)
    steps: list[Literal["repair", "simplify", "uv_unwrap", "texture_bake", "coacd"]]
    completed_asset: CompletedObjectAsset
    technical_gates: dict[str, bool] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_postprocess(self) -> MeshPostprocessRecord:
        _require_unique(self.steps, label="mesh postprocess steps")
        if self.completed_asset.id != self.object_id:
            raise ValueError("completed asset id must match mesh postprocess object id")
        if "coacd" in self.steps and self.completed_asset.collider_topology != (
            "convex_decomposition"
        ):
            raise ValueError("CoACD postprocess requires convex_decomposition collider topology")
        if not all(self.technical_gates.values()):
            raise ValueError("mesh postprocess manifest may contain only passed records")
        return self


class MeshPostprocessManifest(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["video2world.mesh_postprocess_manifest"] = "video2world.mesh_postprocess_manifest"
    scene_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    created_at: datetime
    records: list[MeshPostprocessRecord] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_manifest(self) -> MeshPostprocessManifest:
        _require_timezone(self.created_at, label="mesh postprocess created_at")
        _require_unique(
            [item.object_id for item in self.records], label="mesh postprocess object ids"
        )
        return self


class ObjectPhysicsRecord(StrictModel):
    object_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    properties: PhysicalProperties


class PhysicsManifest(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["video2world.physics_manifest"] = "video2world.physics_manifest"
    scene_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    created_at: datetime
    records: list[ObjectPhysicsRecord] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_manifest(self) -> PhysicsManifest:
        _require_timezone(self.created_at, label="physics manifest created_at")
        _require_unique([item.object_id for item in self.records], label="physics object ids")
        return self
