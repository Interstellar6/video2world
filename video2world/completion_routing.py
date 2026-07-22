"""Evidence-based routing across interchangeable object and background completion backends."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from video2world.hashing import digest_json
from video2world.models import Sha256, StrictModel

RecoveryStage = Literal[
    "donor_support",
    "temporal_qa",
    "boundary_qa",
    "candidate_generation",
]
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


class CleanPlateNextAction(StrictModel):
    action: str | None = Field(default=None, min_length=1)
    blocker: str | None = Field(default=None, min_length=1)
    blocking_gate_groups: list[str] = Field(default_factory=list)
    failed_gates: list[str] = Field(default_factory=list)
    failed_frame_ids: list[str] = Field(default_factory=list)
    first_failed_frame_id: str | None = Field(default=None, min_length=1)
    failed_pair_ids: list[str] = Field(default_factory=list)
    failed_triplet_center_frame_ids: list[str] = Field(default_factory=list)
    not_evaluable_pair_ids: list[str] = Field(default_factory=list)
    not_evaluable_triplet_center_frame_ids: list[str] = Field(default_factory=list)
    no_support_frame_ids: list[str] = Field(default_factory=list)
    unresolved_unobserved_pixels: int | None = Field(default=None, ge=0)
    promotion_approved: bool | None = None


class CompletionRecoveryAction(StrictModel):
    priority: int = Field(ge=1)
    stage: RecoveryStage
    action: str = Field(min_length=1)
    blocker: str | None = Field(default=None, min_length=1)
    reason: str = Field(min_length=1)
    frame_ids: list[str] = Field(default_factory=list)
    pair_ids: list[str] = Field(default_factory=list)
    triplet_center_frame_ids: list[str] = Field(default_factory=list)
    allow_deeper_rounds: Literal[False] = False


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
    clean_plate_next_action: CleanPlateNextAction | None = None

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
    recovery_actions: list[CompletionRecoveryAction] = Field(default_factory=list)
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
        priorities = [item.priority for item in self.recovery_actions]
        if priorities != sorted(set(priorities)):
            raise ValueError("recovery actions must have unique ascending priorities")
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


def _ordered_unique(values: list[str | None]) -> list[str]:
    kept: list[str] = []
    for value in values:
        if isinstance(value, str) and value and value not in kept:
            kept.append(value)
    return kept


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _derive_clean_plate_next_action(report: dict[str, object]) -> CleanPlateNextAction | None:
    pixel_provenance = report.get("pixel_provenance")
    unresolved_pixels: int | None = None
    if isinstance(pixel_provenance, dict):
        unresolved_pixels = _int_or_none(pixel_provenance.get("unresolved_unobserved_pixels"))
    frame_records = report.get("frame_records")
    failed_frame_ids: list[str] = []
    no_support_frame_ids: list[str] = []
    if isinstance(frame_records, list):
        for record in frame_records:
            if not isinstance(record, dict):
                continue
            frame_id = record.get("frame_id")
            if not isinstance(frame_id, str) or not frame_id:
                continue
            residual_pixels = _int_or_none(record.get("residual_mask_pixels")) or 0
            removal_pixels = _int_or_none(record.get("removal_mask_pixels")) or 0
            covered_pixels = _int_or_none(record.get("covered_pixels"))
            coverage_fraction = _float_or_none(record.get("coverage_fraction"))
            support_max = _int_or_none(record.get("support_max"))
            if residual_pixels > 0 or (
                removal_pixels > 0
                and coverage_fraction is not None
                and coverage_fraction < 1.0
            ):
                failed_frame_ids.append(frame_id)
            if removal_pixels > 0 and (
                support_max == 0
                or covered_pixels == 0
                or coverage_fraction == 0.0
            ):
                no_support_frame_ids.append(frame_id)
    if unresolved_pixels is None and isinstance(frame_records, list):
        unresolved_pixels = sum(
            _int_or_none(record.get("residual_mask_pixels")) or 0
            for record in frame_records
            if isinstance(record, dict)
        )
    should_derive = (
        report.get("promotion_approved") is False
        or (unresolved_pixels is not None and unresolved_pixels > 0)
        or str(report.get("status", "")).startswith("technical_failed")
    )
    if not should_derive:
        return None
    failed_frame_ids = _ordered_unique(failed_frame_ids)
    no_support_frame_ids = _ordered_unique(no_support_frame_ids)
    first_failed_frame_id = (no_support_frame_ids or failed_frame_ids or [None])[0]
    if no_support_frame_ids:
        action = "add_observed_donor_or_switch_to_constrained_generation_for_residual"
        blocker = "no_guard_stable_measured_donor_support"
        blocking_gate_groups = ["donor_support"]
    else:
        action = "regenerate_residual_with_constrained_background_prior"
        blocker = "unresolved_unobserved_residual"
        blocking_gate_groups = ["residual_generation"]
    return CleanPlateNextAction(
        action=action,
        blocker=blocker,
        blocking_gate_groups=blocking_gate_groups,
        failed_frame_ids=failed_frame_ids,
        first_failed_frame_id=first_failed_frame_id,
        no_support_frame_ids=no_support_frame_ids,
        unresolved_unobserved_pixels=unresolved_pixels,
        promotion_approved=(
            report.get("promotion_approved")
            if isinstance(report.get("promotion_approved"), bool)
            else None
        ),
    )


def clean_plate_next_action_from_report(report: dict[str, object]) -> CleanPlateNextAction:
    value = report.get("next_action")
    if not isinstance(value, dict):
        derived = _derive_clean_plate_next_action(report)
        if derived is not None:
            return derived
        raise ValueError(
            "clean plate report must contain a next_action object or failed residual evidence"
        )
    known = {
        key: item
        for key, item in value.items()
        if key in CleanPlateNextAction.model_fields
    }
    return CleanPlateNextAction.model_validate(known)


def clean_plate_recovery_actions(
    next_action: CleanPlateNextAction | None,
) -> list[CompletionRecoveryAction]:
    if next_action is None:
        return []

    frame_ids = _ordered_unique(
        [*next_action.failed_frame_ids, next_action.first_failed_frame_id]
    )
    pair_ids = _ordered_unique(
        [*next_action.failed_pair_ids, *next_action.not_evaluable_pair_ids]
    )
    triplet_ids = _ordered_unique(
        [
            *next_action.failed_triplet_center_frame_ids,
            *next_action.not_evaluable_triplet_center_frame_ids,
        ]
    )
    blocker = next_action.blocker
    actions: list[CompletionRecoveryAction] = []

    def add(
        stage: RecoveryStage,
        action: str,
        reason: str,
        *,
        frames: list[str] | None = None,
        pairs: list[str] | None = None,
        triplets: list[str] | None = None,
    ) -> None:
        actions.append(
            CompletionRecoveryAction(
                priority=len(actions) + 1,
                stage=stage,
                action=action,
                blocker=blocker,
                reason=reason,
                frame_ids=frames or [],
                pair_ids=pairs or [],
                triplet_center_frame_ids=triplets or [],
            )
        )

    if next_action.no_support_frame_ids or blocker == "no_guard_stable_measured_donor_support":
        add(
            "donor_support",
            "add_observed_donor_or_switch_to_constrained_generation_for_residual",
            "Boundary-guarded measured donor support is missing; do not advance the next layer.",
            frames=_ordered_unique([*next_action.no_support_frame_ids, *frame_ids]),
        )
    if next_action.not_evaluable_pair_ids or next_action.not_evaluable_triplet_center_frame_ids:
        add(
            "temporal_qa",
            "repair_temporal_evidence_before_retesting",
            "Temporal QA is not evaluable, so candidate quality cannot be promoted.",
            pairs=next_action.not_evaluable_pair_ids,
            triplets=next_action.not_evaluable_triplet_center_frame_ids,
        )
    if next_action.failed_pair_ids or next_action.failed_triplet_center_frame_ids:
        add(
            "temporal_qa",
            "repair_temporal_flicker_or_regenerate_clean_plate_candidates",
            "Temporal QA has failing directed pairs or triplets.",
            pairs=next_action.failed_pair_ids,
            triplets=next_action.failed_triplet_center_frame_ids,
        )
    boundary_groups = {
        "boundary",
        "core_texture",
        "donor_support",
        "mask_alignment",
    }
    if frame_ids or boundary_groups.intersection(next_action.blocking_gate_groups):
        add(
            "boundary_qa",
            "repair_removal_mask_or_target_matte_before_regeneration",
            "Boundary or local texture gates failed on specific source frames.",
            frames=frame_ids,
        )
    if (
        next_action.unresolved_unobserved_pixels is not None
        and next_action.unresolved_unobserved_pixels > 0
    ):
        add(
            "candidate_generation",
            "regenerate_residual_with_constrained_background_prior",
            "The clean plate still has unresolved unobserved pixels after measured support.",
            frames=frame_ids,
        )
    if not actions and next_action.action:
        add(
            "candidate_generation",
            next_action.action,
            "Carry forward the upstream clean-plate recovery action.",
            frames=frame_ids,
            pairs=pair_ids,
            triplets=triplet_ids,
        )
    return actions


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
    recovery_actions = clean_plate_recovery_actions(evidence.clean_plate_next_action)
    if recovery_actions:
        notes.append(
            "Clean-plate recovery is pending; deeper occlusion rounds must wait for these "
            "actions to pass."
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
        "recovery_actions": [item.model_dump(mode="json") for item in recovery_actions],
        "notes": notes,
    }
    return CompletionBackendRoute(
        route_sha256=digest_json(payload),
        **payload,
    )
