"""Materialize auditable recovery work orders from blocked completion routes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from video2world.completion_routing import CompletionBackendRoute, RecoveryStage
from video2world.hashing import digest_json, digest_path
from video2world.models import Sha256, StrictModel


class RecoveryInputArtifact(StrictModel):
    path: str = Field(min_length=1)
    sha256: Sha256
    size_bytes: int = Field(ge=1)
    role: Literal["completion_backend_route", "clean_plate_report"]


class RecoveryFrameRecord(StrictModel):
    frame_id: str = Field(min_length=1)
    removal_mask_pixels: int | None = Field(default=None, ge=0)
    residual_mask_pixels: int | None = Field(default=None, ge=0)
    covered_pixels: int | None = Field(default=None, ge=0)
    coverage_fraction: float | None = Field(default=None, ge=0, le=1)
    support_max: int | None = Field(default=None, ge=0)


class RecoveryWorkItem(StrictModel):
    priority: int = Field(ge=1)
    stage: RecoveryStage
    action: str = Field(min_length=1)
    blocker: str | None = Field(default=None, min_length=1)
    reason: str = Field(min_length=1)
    frame_ids: list[str] = Field(default_factory=list)
    frame_records: list[RecoveryFrameRecord] = Field(default_factory=list)
    allow_deeper_rounds: Literal[False] = False
    required_verification: str = Field(min_length=1)


class CompletionRecoveryWorkOrder(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["video2world.completion_recovery_work_order"] = (
        "video2world.completion_recovery_work_order"
    )
    object_id: str = Field(min_length=1)
    route_sha256: Sha256
    work_order_sha256: Sha256
    status: Literal["blocked_pending_recovery"] = "blocked_pending_recovery"
    selected_backend: str = Field(min_length=1)
    deeper_rounds_blocked: Literal[True] = True
    output_claim: Literal["work_order_only_no_clean_plate_generated"] = (
        "work_order_only_no_clean_plate_generated"
    )
    inputs: list[RecoveryInputArtifact] = Field(min_length=2, max_length=2)
    work_items: list[RecoveryWorkItem] = Field(min_length=1)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def verify_recovery_contract(self) -> CompletionRecoveryWorkOrder:
        priorities = [item.priority for item in self.work_items]
        if priorities != sorted(set(priorities)):
            raise ValueError("work item priorities must be unique and ascending")
        if any(item.allow_deeper_rounds is not False for item in self.work_items):
            raise ValueError("all recovery work items must block deeper rounds")
        return self


class RecoveryBundleInput(StrictModel):
    path: str = Field(min_length=1)
    sha256: Sha256
    size_bytes: int = Field(ge=1)
    role: Literal["completion_recovery_work_order"]


class RecoveryBundleStep(StrictModel):
    step_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.:-]+$")
    priority: int = Field(ge=1)
    stage: RecoveryStage
    depends_on: list[str] = Field(default_factory=list)
    action: str = Field(min_length=1)
    frame_ids: list[str] = Field(default_factory=list)
    input_roles: list[str] = Field(min_length=1)
    expected_output_roles: list[str] = Field(min_length=1)
    required_verification: str = Field(min_length=1)
    allow_deeper_rounds: Literal[False] = False


class CompletionRecoveryBundle(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["video2world.completion_recovery_bundle"] = (
        "video2world.completion_recovery_bundle"
    )
    object_id: str = Field(min_length=1)
    bundle_sha256: Sha256
    status: Literal["ready_for_recovery_execution"] = "ready_for_recovery_execution"
    deeper_rounds_blocked: Literal[True] = True
    input: RecoveryBundleInput
    steps: list[RecoveryBundleStep] = Field(min_length=1)
    final_gate: Literal["rerun_strict_r1_acceptance_before_r2"] = (
        "rerun_strict_r1_acceptance_before_r2"
    )
    output_claim: Literal["execution_plan_only_no_artifacts_generated"] = (
        "execution_plan_only_no_artifacts_generated"
    )
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def verify_bundle_contract(self) -> CompletionRecoveryBundle:
        priorities = [step.priority for step in self.steps]
        if priorities != sorted(set(priorities)):
            raise ValueError("bundle step priorities must be unique and ascending")
        ids = [step.step_id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("bundle step ids must be unique")
        seen: set[str] = set()
        for step in self.steps:
            if any(dependency not in seen for dependency in step.depends_on):
                raise ValueError("bundle step dependencies must reference earlier steps")
            seen.add(step.step_id)
        return self


class RecoveryPreflightBinding(StrictModel):
    role: str = Field(min_length=1)
    path: str = ""
    declared_path: str = ""
    effective_path: str = ""
    override_applied: bool = False
    expected_kind: Literal["file", "directory"]
    status: Literal["present", "missing", "kind_mismatch"]
    sha256: Sha256 | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    file_count: int | None = Field(default=None, ge=0)


class CompletionRecoveryPreflight(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["video2world.completion_recovery_preflight"] = (
        "video2world.completion_recovery_preflight"
    )
    object_id: str = Field(min_length=1)
    preflight_sha256: Sha256
    status: Literal["passed", "blocked_input_mismatch", "blocked_missing_bindings"]
    deeper_rounds_blocked: Literal[True] = True
    bundle_sha256: Sha256
    work_order_sha256: Sha256 | None = None
    clean_plate_report_sha256: Sha256 | None = None
    bundle_input_verified: bool
    work_order_input_verified: bool
    clean_plate_report_verified: bool
    bindings: list[RecoveryPreflightBinding] = Field(default_factory=list)
    missing_roles: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def _artifact(
    path: Path,
    role: Literal["completion_backend_route", "clean_plate_report"],
) -> RecoveryInputArtifact:
    digest = digest_path(path)
    return RecoveryInputArtifact(
        path=str(digest.path),
        sha256=digest.sha256,
        size_bytes=digest.size_bytes,
        role=role,
    )


def _frame_records_by_id(report: dict[str, Any]) -> dict[str, RecoveryFrameRecord]:
    result: dict[str, RecoveryFrameRecord] = {}
    records = report.get("frame_records")
    if not isinstance(records, list):
        return result
    for record in records:
        if not isinstance(record, dict):
            continue
        frame_id = record.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id:
            continue
        result[frame_id] = RecoveryFrameRecord(
            frame_id=frame_id,
            removal_mask_pixels=record.get("removal_mask_pixels"),
            residual_mask_pixels=record.get("residual_mask_pixels"),
            covered_pixels=record.get("covered_pixels"),
            coverage_fraction=record.get("coverage_fraction"),
            support_max=record.get("support_max"),
        )
    return result


def _required_verification(stage: RecoveryStage) -> str:
    if stage == "donor_support":
        return "rerun measured donor support and strict boundary guard before any R2 input"
    if stage == "boundary_qa":
        return "rerun boundary QA with failed frame records preserved in the report"
    if stage == "temporal_qa":
        return "rerun temporal pair/triplet QA before candidate promotion"
    return "rerun clean-plate residual generation QA before promotion"


def _input_roles(stage: RecoveryStage) -> list[str]:
    common = ["completion_recovery_work_order", "source_clean_plate_report"]
    if stage == "donor_support":
        return [
            *common,
            "source_rgb_frames",
            "camera_info",
            "depth_arrays",
            "physical_donor_exclusion_index",
        ]
    if stage == "boundary_qa":
        return [*common, "candidate_clean_plate_frames", "removal_masks", "source_rgb_frames"]
    if stage == "temporal_qa":
        return [*common, "candidate_clean_plate_frames", "source_rgb_frames"]
    return [
        *common,
        "residual_masks",
        "support_geometry_or_structural_prior",
        "source_rgb_frames",
    ]


def _expected_output_roles(stage: RecoveryStage) -> list[str]:
    if stage == "donor_support":
        return [
            "measured_prefill_report",
            "measured_prefill_receipt",
            "residual_masks",
            "support_visualizations",
            "measured_depth_evidence",
        ]
    if stage == "boundary_qa":
        return ["boundary_qa_manifest", "boundary_qa_report", "failed_frame_summary"]
    if stage == "temporal_qa":
        return ["temporal_qa_manifest", "temporal_qa_report", "failed_pair_summary"]
    return [
        "residual_generation_manifest",
        "generated_candidate_frames",
        "candidate_generation_receipt",
        "candidate_review_report",
    ]


_PREFLIGHT_BINDING_SPECS: dict[str, tuple[str, Literal["file", "directory"]]] = {
    "input_manifest": ("input_manifest", "file"),
    "camera_info": ("camera_info", "file"),
    "source_rgb_frames": ("donor_frames_dir", "directory"),
    "depth_arrays": ("depth_dir", "directory"),
    "physical_donor_exclusion_index": ("donor_mask_index", "file"),
}


def _normalize_binding_overrides(
    overrides: dict[str, str | Path] | None,
) -> dict[str, str]:
    if not overrides:
        return {}
    normalized: dict[str, str] = {}
    unknown = sorted(set(overrides) - set(_PREFLIGHT_BINDING_SPECS))
    if unknown:
        raise ValueError(f"unknown recovery preflight binding role(s): {', '.join(unknown)}")
    for role, value in overrides.items():
        text = str(value)
        if not text:
            raise ValueError(f"empty recovery preflight binding path for role: {role}")
        normalized[role] = text
    return normalized


def _binding(
    role: str,
    value: object,
    expected_kind: Literal["file", "directory"],
    *,
    override_path: str | None = None,
) -> RecoveryPreflightBinding:
    declared_path = str(value) if isinstance(value, str) and value else ""
    effective_value = override_path or declared_path
    path = Path(effective_value).expanduser() if effective_value else Path("")
    base = {
        "role": role,
        "path": str(path.resolve()) if effective_value else "",
        "declared_path": declared_path,
        "effective_path": str(path.resolve()) if effective_value else "",
        "override_applied": override_path is not None,
        "expected_kind": expected_kind,
    }
    if not effective_value:
        return RecoveryPreflightBinding(
            **base,
            status="missing",
        )
    resolved = path.resolve()
    if not resolved.exists():
        return RecoveryPreflightBinding(
            **(base | {"path": str(resolved), "effective_path": str(resolved)}),
            status="missing",
        )
    if (expected_kind == "file" and not resolved.is_file()) or (
        expected_kind == "directory" and not resolved.is_dir()
    ):
        return RecoveryPreflightBinding(
            **(base | {"path": str(resolved), "effective_path": str(resolved)}),
            status="kind_mismatch",
        )
    digest = digest_path(resolved)
    return RecoveryPreflightBinding(
        **(base | {"path": str(digest.path), "effective_path": str(digest.path)}),
        status="present",
        sha256=digest.sha256,
        size_bytes=digest.size_bytes,
        file_count=digest.file_count,
    )


def _find_input(work_order: CompletionRecoveryWorkOrder, role: str) -> RecoveryInputArtifact:
    matches = [item for item in work_order.inputs if item.role == role]
    if len(matches) != 1:
        raise ValueError(f"work order must contain exactly one {role} input")
    return matches[0]


def _verify_declared_file(path: str, sha256: str, size_bytes: int) -> tuple[bool, str | None]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        return False, None
    digest = digest_path(resolved)
    return digest.sha256 == sha256 and digest.size_bytes == size_bytes, digest.sha256


def materialize_completion_recovery_work_order(
    route_path: str | Path,
    clean_plate_report_path: str | Path,
) -> CompletionRecoveryWorkOrder:
    route_file = Path(route_path).expanduser().resolve()
    report_file = Path(clean_plate_report_path).expanduser().resolve()
    route_payload = json.loads(route_file.read_text(encoding="utf-8"))
    report_payload = json.loads(report_file.read_text(encoding="utf-8"))
    if not isinstance(route_payload, dict):
        raise ValueError("completion route root must be an object")
    if not isinstance(report_payload, dict):
        raise ValueError("clean plate report root must be an object")
    route = CompletionBackendRoute.model_validate(route_payload)
    if not route.recovery_actions:
        raise ValueError("completion route has no recovery_actions to materialize")
    frame_records = _frame_records_by_id(report_payload)
    work_items = [
        RecoveryWorkItem(
            priority=action.priority,
            stage=action.stage,
            action=action.action,
            blocker=action.blocker,
            reason=action.reason,
            frame_ids=action.frame_ids,
            frame_records=[
                frame_records[frame_id]
                for frame_id in action.frame_ids
                if frame_id in frame_records
            ],
            allow_deeper_rounds=action.allow_deeper_rounds,
            required_verification=_required_verification(action.stage),
        )
        for action in route.recovery_actions
    ]
    payload = {
        "object_id": route.object_id,
        "route_sha256": route.route_sha256,
        "status": "blocked_pending_recovery",
        "selected_backend": route.selected_backend,
        "deeper_rounds_blocked": True,
        "output_claim": "work_order_only_no_clean_plate_generated",
        "inputs": [
            _artifact(route_file, "completion_backend_route").model_dump(mode="json"),
            _artifact(report_file, "clean_plate_report").model_dump(mode="json"),
        ],
        "work_items": [item.model_dump(mode="json") for item in work_items],
        "notes": [
            (
                "This work order is a recovery plan only; it does not accept R1 or "
                "generate clean plates."
            ),
            "R2-R4 remain blocked until the listed verification steps pass.",
        ],
    }
    return CompletionRecoveryWorkOrder(
        work_order_sha256=digest_json(payload),
        **payload,
    )


def materialize_completion_recovery_bundle(
    work_order_path: str | Path,
) -> CompletionRecoveryBundle:
    work_order_file = Path(work_order_path).expanduser().resolve()
    payload = json.loads(work_order_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("recovery work order root must be an object")
    work_order = CompletionRecoveryWorkOrder.model_validate(payload)
    work_order_digest = digest_path(work_order_file)
    steps: list[RecoveryBundleStep] = []
    for item in work_order.work_items:
        depends_on = [steps[-1].step_id] if steps else []
        step = RecoveryBundleStep(
            step_id=f"{item.priority:02d}-{item.stage}",
            priority=item.priority,
            stage=item.stage,
            depends_on=depends_on,
            action=item.action,
            frame_ids=item.frame_ids,
            input_roles=_input_roles(item.stage),
            expected_output_roles=_expected_output_roles(item.stage),
            required_verification=item.required_verification,
            allow_deeper_rounds=item.allow_deeper_rounds,
        )
        steps.append(step)
    bundle_payload = {
        "object_id": work_order.object_id,
        "status": "ready_for_recovery_execution",
        "deeper_rounds_blocked": True,
        "input": RecoveryBundleInput(
            path=str(work_order_digest.path),
            sha256=work_order_digest.sha256,
            size_bytes=work_order_digest.size_bytes,
            role="completion_recovery_work_order",
        ).model_dump(mode="json"),
        "steps": [step.model_dump(mode="json") for step in steps],
        "final_gate": "rerun_strict_r1_acceptance_before_r2",
        "output_claim": "execution_plan_only_no_artifacts_generated",
        "notes": [
            "Execute these steps in order; each step must write its own receipt and QA report.",
            "The bundle is invalidated if the source work order SHA-256 changes.",
        ],
    }
    return CompletionRecoveryBundle(
        bundle_sha256=digest_json(bundle_payload),
        **bundle_payload,
    )


def materialize_completion_recovery_preflight(
    bundle_path: str | Path,
    *,
    binding_overrides: dict[str, str | Path] | None = None,
) -> CompletionRecoveryPreflight:
    overrides = _normalize_binding_overrides(binding_overrides)
    bundle_file = Path(bundle_path).expanduser().resolve()
    bundle_payload = json.loads(bundle_file.read_text(encoding="utf-8"))
    if not isinstance(bundle_payload, dict):
        raise ValueError("recovery bundle root must be an object")
    bundle = CompletionRecoveryBundle.model_validate(bundle_payload)
    bundle_digest = digest_path(bundle_file)
    bundle_input_verified, work_order_actual_sha = _verify_declared_file(
        bundle.input.path,
        bundle.input.sha256,
        bundle.input.size_bytes,
    )

    work_order: CompletionRecoveryWorkOrder | None = None
    report_payload: dict[str, Any] | None = None
    work_order_input_verified = False
    clean_plate_report_verified = False
    clean_plate_report_sha: str | None = None
    bindings: list[RecoveryPreflightBinding] = []

    if bundle_input_verified:
        work_order_payload = json.loads(Path(bundle.input.path).read_text(encoding="utf-8"))
        if not isinstance(work_order_payload, dict):
            raise ValueError("recovery work order root must be an object")
        work_order = CompletionRecoveryWorkOrder.model_validate(work_order_payload)
        report_input = _find_input(work_order, "clean_plate_report")
        clean_plate_report_verified, clean_plate_report_sha = _verify_declared_file(
            report_input.path,
            report_input.sha256,
            report_input.size_bytes,
        )
        work_order_input_verified = True
        if clean_plate_report_verified:
            report_payload = json.loads(Path(report_input.path).read_text(encoding="utf-8"))
            if not isinstance(report_payload, dict):
                raise ValueError("clean plate report root must be an object")

    if report_payload is not None:
        bindings = [
            _binding(
                role,
                report_payload.get(report_key),
                expected_kind,
                override_path=overrides.get(role),
            )
            for role, (report_key, expected_kind) in _PREFLIGHT_BINDING_SPECS.items()
        ]

    missing_roles = [
        item.role for item in bindings if item.status in {"missing", "kind_mismatch"}
    ]
    if not bundle_input_verified or not clean_plate_report_verified:
        status = "blocked_input_mismatch"
    elif missing_roles:
        status = "blocked_missing_bindings"
    else:
        status = "passed"
    payload = {
        "object_id": bundle.object_id,
        "status": status,
        "deeper_rounds_blocked": True,
        "bundle_sha256": bundle_digest.sha256,
        "work_order_sha256": work_order_actual_sha,
        "clean_plate_report_sha256": clean_plate_report_sha,
        "bundle_input_verified": bundle_input_verified,
        "work_order_input_verified": work_order_input_verified,
        "clean_plate_report_verified": clean_plate_report_verified,
        "bindings": [item.model_dump(mode="json") for item in bindings],
        "missing_roles": missing_roles,
        "notes": [
            "Preflight checks only local input availability and hashes; it does not run recovery.",
            (
                "Passing preflight does not accept R1; R2-R4 remain blocked until "
                "recovery execution and strict R1 acceptance pass."
            ),
            *(
                [
                    (
                        "Local binding overrides were used; declared remote paths are "
                        "preserved beside effective local mirror paths."
                    )
                ]
                if overrides
                else []
            ),
        ],
    }
    return CompletionRecoveryPreflight(
        preflight_sha256=digest_json(payload),
        **payload,
    )
