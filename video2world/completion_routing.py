"""Evidence-based routing across interchangeable object and background completion backends."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from video2world.hashing import digest_json
from video2world.models import Sha256, StrictModel

CompletionBackend = Literal[
    "multi_view_reconstruction",
    "component_assembly",
    "deformable_category_prior",
    "generative_image_to_3d",
    "retrieval_cad_prior",
    "support_surface_reconstruction",
    "hold_for_more_evidence",
]
TargetKind = Literal["object", "fixture", "structure_background"]

DEFORMABLE_CATEGORY_PROFILES = {
    "bean bag",
    "cushion",
    "pillow",
    "seat cushion",
    "throw pillow",
}
RIGID_CAD_CATEGORIES = {
    "cabinet",
    "chair",
    "door",
    "lamp",
    "nightstand",
    "stool",
    "table",
    "vase",
}
COMMON_ACCEPTANCE_GATES = [
    "instance_identity",
    "front_observation_fidelity",
    "appearance_fidelity",
    "geometry_completeness",
    "backside_nonempty",
    "six_view_coverage",
    "six_side_style_consistency",
    "placement_alignment",
    "scene_interpenetration",
    "collision",
    "vlm_geometry_review",
]


class CompletionEvidence(StrictModel):
    object_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    category: str = Field(min_length=1)
    target_kind: TargetKind = "object"
    same_instance_multi_frame_verified: bool
    appearance_contract_verified: bool
    calibrated_view_count: int = Field(ge=0)
    view_baseline_score: float = Field(ge=0, le=1)
    estimated_visible_surface_fraction: float = Field(ge=0, le=1)
    persistent_occlusion_fraction: float = Field(ge=0, le=1)
    deformable_prior_available: bool = False
    cad_retrieval_confidence: float | None = Field(default=None, ge=0, le=1)
    support_geometry_available: bool = False
    source_front_rgba_available: bool = False
    multi_depth_negative_space_verified: bool = False

    @model_validator(mode="after")
    def validate_target_evidence(self) -> CompletionEvidence:
        if self.target_kind == "structure_background" and not self.support_geometry_available:
            raise ValueError("structure background routing requires support geometry")
        return self


class BackendCandidate(StrictModel):
    backend: CompletionBackend
    eligible: bool
    score: int = Field(ge=0, le=100)
    reason: str = Field(min_length=1)
    missing_requirements: list[str] = Field(default_factory=list)


class CompletionBackendRoute(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["video2world.completion_backend_route"] = "video2world.completion_backend_route"
    object_id: str
    selected_backend: CompletionBackend
    route_sha256: Sha256
    candidates: list[BackendCandidate]
    required_acceptance_gates: list[str]
    residual_background_policy: Literal[
        "not_applicable",
        "multi_view_donor_then_constrained_generation",
    ]
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def route_is_consistent(self) -> CompletionBackendRoute:
        selected = [
            item
            for item in self.candidates
            if item.backend == self.selected_backend and item.eligible
        ]
        if len(selected) != 1:
            raise ValueError("selected backend must be exactly one eligible candidate")
        if len(self.required_acceptance_gates) != len(set(self.required_acceptance_gates)):
            raise ValueError("acceptance gates must be unique")
        return self


def _candidate(
    backend: CompletionBackend,
    eligible: bool,
    score: int,
    reason: str,
    *missing: str,
) -> BackendCandidate:
    return BackendCandidate(
        backend=backend,
        eligible=eligible,
        score=score if eligible else 0,
        reason=reason,
        missing_requirements=list(missing) if not eligible else [],
    )


def route_completion_backend(evidence: CompletionEvidence) -> CompletionBackendRoute:
    """Choose a backend from evidence; never let a category label bypass identity gates."""

    category = " ".join(evidence.category.casefold().replace("_", " ").split())
    identity_ready = (
        evidence.same_instance_multi_frame_verified and evidence.appearance_contract_verified
    )
    multi_view_ready = (
        identity_ready
        and evidence.calibrated_view_count >= 3
        and evidence.view_baseline_score >= 0.05
        and evidence.estimated_visible_surface_fraction >= 0.35
    )
    deformable_ready = (
        identity_ready
        and evidence.source_front_rgba_available
        and (evidence.deformable_prior_available or category in DEFORMABLE_CATEGORY_PROFILES)
    )
    cad_ready = (
        identity_ready
        and category in RIGID_CAD_CATEGORIES
        and evidence.cad_retrieval_confidence is not None
        and evidence.cad_retrieval_confidence >= 0.82
    )
    support_ready = (
        identity_ready
        and evidence.target_kind == "structure_background"
        and evidence.support_geometry_available
    )
    component_assembly_ready = (
        identity_ready
        and evidence.target_kind == "fixture"
        and evidence.support_geometry_available
        and evidence.multi_depth_negative_space_verified
    )
    generative_ready = identity_ready and evidence.source_front_rgba_available

    candidates = [
        _candidate(
            "support_surface_reconstruction",
            support_ready,
            98,
            "Use observed planes/depth/normals before synthesizing only residual structure holes.",
            "verified appearance",
            "support geometry",
        ),
        _candidate(
            "component_assembly",
            component_assembly_ready,
            97,
            (
                "Preserve verified negative space by reconstructing depth-separated internal "
                "components and exporting them under one logical asset root."
            ),
            "verified fixture identity and appearance",
            "support geometry",
            "verified multi-depth negative space",
        ),
        _candidate(
            "multi_view_reconstruction",
            multi_view_ready,
            96,
            "Calibrated views with usable baseline provide direct geometry evidence.",
            "at least three calibrated views",
            "view baseline >= 0.05",
            "visible surface fraction >= 0.35",
        ),
        _candidate(
            "deformable_category_prior",
            deformable_ready,
            88,
            "A deformable closed-surface prior can preserve observed silhouette and material.",
            "verified source front RGBA",
            "registered deformable category profile",
        ),
        _candidate(
            "retrieval_cad_prior",
            cad_ready,
            82,
            "A high-confidence rigid retrieval can be fitted to observed masks and depth.",
            "rigid category profile",
            "CAD retrieval confidence >= 0.82",
        ),
        _candidate(
            "generative_image_to_3d",
            generative_ready,
            70,
            "Use generation only where direct geometry or stronger priors are unavailable.",
            "verified source front RGBA",
            "same-instance appearance contract",
        ),
        _candidate(
            "hold_for_more_evidence",
            True,
            100 if not identity_ready else 1,
            (
                "Identity or appearance is not verified; generation is blocked."
                if not identity_ready
                else "Fail-closed fallback if every reconstruction candidate is rejected."
            ),
        ),
    ]
    eligible = sorted(
        (item for item in candidates if item.eligible),
        key=lambda item: (-item.score, item.backend),
    )
    selected = eligible[0].backend
    residual_policy = (
        "multi_view_donor_then_constrained_generation"
        if evidence.persistent_occlusion_fraction > 0
        else "not_applicable"
    )
    notes: list[str] = []
    if evidence.persistent_occlusion_fraction >= 0.80:
        notes.append(
            "Most hidden pixels have no observed donor; label the residual as generated, "
            "condition it on support geometry, and rerun depth/normal reconstruction."
        )
    if selected == "deformable_category_prior":
        notes.append(
            "The category prior supplies topology only; source evidence still controls aspect, "
            "material, and front appearance."
        )
    acceptance_gates = list(COMMON_ACCEPTANCE_GATES)
    if selected == "component_assembly":
        acceptance_gates.extend(
            [
                "negative_space_preservation",
                "internal_component_alignment",
                "single_logical_asset_root",
            ]
        )
        notes.append(
            "Internal components are reconstruction units, not independent scene objects; "
            "selection, motion, and collision ownership remain on one fixture entity."
        )
    payload = {
        "object_id": evidence.object_id,
        "selected_backend": selected,
        "candidates": [item.model_dump(mode="json") for item in candidates],
        "required_acceptance_gates": acceptance_gates,
        "residual_background_policy": residual_policy,
        "notes": notes,
    }
    return CompletionBackendRoute(
        route_sha256=digest_json(payload),
        **payload,
    )
