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


class RecoveryPreflightFrameAlignment(StrictModel):
    role: str = Field(min_length=1)
    status: Literal[
        "matched",
        "binding_unavailable",
        "missing_required_frames",
        "content_mismatch",
        "unverified",
    ]
    expected_frame_count: int = Field(ge=0)
    available_frame_count: int = Field(ge=0)
    matched_frame_count: int = Field(ge=0)
    missing_frame_ids: list[str] = Field(default_factory=list)
    content_mismatch_frame_ids: list[str] = Field(default_factory=list)
    extra_frame_ids: list[str] = Field(default_factory=list)
    available_frame_ids: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1)


class CompletionRecoveryPreflight(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["video2world.completion_recovery_preflight"] = (
        "video2world.completion_recovery_preflight"
    )
    object_id: str = Field(min_length=1)
    preflight_sha256: Sha256
    status: Literal[
        "passed",
        "blocked_input_mismatch",
        "blocked_missing_bindings",
        "blocked_binding_semantics",
    ]
    deeper_rounds_blocked: Literal[True] = True
    bundle_sha256: Sha256
    work_order_sha256: Sha256 | None = None
    clean_plate_report_sha256: Sha256 | None = None
    bundle_input_verified: bool
    work_order_input_verified: bool
    clean_plate_report_verified: bool
    bindings: list[RecoveryPreflightBinding] = Field(default_factory=list)
    missing_roles: list[str] = Field(default_factory=list)
    required_frame_ids: list[str] = Field(default_factory=list)
    frame_alignment: list[RecoveryPreflightFrameAlignment] = Field(default_factory=list)
    semantic_blocking_roles: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class RecoveryHandoffInput(StrictModel):
    path: str = Field(min_length=1)
    sha256: Sha256
    size_bytes: int = Field(ge=1)
    role: Literal[
        "completion_recovery_residual_handoff",
        "completion_recovery_bundle",
        "completion_recovery_preflight",
        "measured_prefill_report",
        "measured_prefill_receipt",
    ]


class RecoveryHandoffAsset(StrictModel):
    path: str = Field(min_length=1)
    sha256: Sha256
    size_bytes: int = Field(ge=1)
    role: str = Field(min_length=1)


class RecoveryResidualHandoffFrame(StrictModel):
    sequence_index: int = Field(ge=0)
    frame_id: str = Field(min_length=1)
    source_rgb: RecoveryHandoffAsset
    removal_mask: RecoveryHandoffAsset
    measured_prefill_rgb: RecoveryHandoffAsset
    residual_mask: RecoveryHandoffAsset
    support_visualization: RecoveryHandoffAsset
    measured_depth: RecoveryHandoffAsset
    removal_mask_pixels: int = Field(ge=0)
    measured_multiview_pixels: int = Field(ge=0)
    residual_mask_pixels: int = Field(ge=0)
    coverage_fraction: float = Field(ge=0, le=1)
    support_max: int = Field(ge=0)
    measured_depth_valid_pixels: int = Field(ge=0)
    donor_count: int = Field(ge=0)


class CompletionRecoveryResidualHandoff(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["video2world.completion_recovery_residual_handoff"] = (
        "video2world.completion_recovery_residual_handoff"
    )
    object_id: str = Field(min_length=1)
    handoff_sha256: Sha256
    status: Literal["ready_for_constrained_residual_completion"] = (
        "ready_for_constrained_residual_completion"
    )
    completed_step_id: Literal["01-donor_support"] = "01-donor_support"
    next_step_id: Literal["03-constrained_residual_completion"] = (
        "03-constrained_residual_completion"
    )
    deeper_rounds_blocked: Literal[True] = True
    r1_promotion_approved: Literal[False] = False
    output_claim: Literal[
        "measured_prefill_plus_residual_masks_only_no_r1_acceptance"
    ] = "measured_prefill_plus_residual_masks_only_no_r1_acceptance"
    inputs: list[RecoveryHandoffInput] = Field(min_length=4, max_length=4)
    frame_records: list[RecoveryResidualHandoffFrame] = Field(min_length=1)
    pixel_provenance: dict[str, Any] = Field(default_factory=dict)
    gates: dict[str, Any] = Field(default_factory=dict)
    next_action: dict[str, Any] = Field(default_factory=dict)
    required_next_steps: list[str] = Field(min_length=1)
    notes: list[str] = Field(default_factory=list)


class RecoveryConstrainedResidualFrame(StrictModel):
    sequence_index: int = Field(ge=0)
    frame_id: str = Field(min_length=1)
    residual_mask: RecoveryHandoffAsset
    measured_prefill_rgb: RecoveryHandoffAsset
    source_rgb: RecoveryHandoffAsset
    removal_mask: RecoveryHandoffAsset
    measured_multiview_pixels: int = Field(ge=0)
    residual_mask_pixels: int = Field(ge=0)


class CompletionRecoveryConstrainedResidualManifest(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["video2world.completion_recovery_constrained_residual_manifest"] = (
        "video2world.completion_recovery_constrained_residual_manifest"
    )
    object_id: str = Field(min_length=1)
    manifest_sha256: Sha256
    status: Literal["ready_for_structural_residual_completion"] = (
        "ready_for_structural_residual_completion"
    )
    planned_step_id: Literal["03-constrained_residual_completion"] = (
        "03-constrained_residual_completion"
    )
    execution_status: Literal["planned_not_run"] = "planned_not_run"
    deeper_rounds_blocked: Literal[True] = True
    r1_promotion_approved: Literal[False] = False
    source_handoff: RecoveryHandoffInput
    bindings: list[RecoveryHandoffAsset] = Field(min_length=5)
    frame_records: list[RecoveryConstrainedResidualFrame] = Field(min_length=1)
    depth_format_counts: dict[str, int] = Field(default_factory=dict)
    planned_output_dir: str = Field(min_length=1)
    working_directory: str = Field(min_length=1)
    command_argv: list[str] = Field(min_length=1)
    constraints: dict[str, Any] = Field(default_factory=dict)
    required_verification: list[str] = Field(min_length=1)
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


def _handoff_input(
    path: Path,
    role: Literal[
        "completion_recovery_residual_handoff",
        "completion_recovery_bundle",
        "completion_recovery_preflight",
        "measured_prefill_report",
        "measured_prefill_receipt",
    ],
) -> RecoveryHandoffInput:
    digest = digest_path(path)
    return RecoveryHandoffInput(
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


def _unique_sorted_frame_ids(values: list[object]) -> list[str]:
    frame_ids = {
        value
        for value in values
        if isinstance(value, str) and value and value.replace("_", "").isalnum()
    }
    return sorted(frame_ids)


def _required_frame_ids(bundle: CompletionRecoveryBundle) -> list[str]:
    return _unique_sorted_frame_ids(
        [frame_id for step in bundle.steps for frame_id in step.frame_ids]
    )


def _frame_ids_from_records(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    frame_ids: list[object] = []
    for item in value:
        if isinstance(item, str):
            frame_ids.append(Path(item).stem)
        elif isinstance(item, dict):
            for key in ("frame_id", "image", "name", "path", "file"):
                candidate = item.get(key)
                if isinstance(candidate, str) and candidate:
                    frame_ids.append(Path(candidate).stem)
                    break
    return _unique_sorted_frame_ids(frame_ids)


def _json_frame_ids(role: str, path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return []
    if role == "input_manifest":
        return _frame_ids_from_records(payload.get("frame_records")) or _frame_ids_from_records(
            payload.get("frames")
        )
    if role == "camera_info":
        subset = payload.get("subset_provenance")
        if isinstance(subset, dict):
            subset_ids = _frame_ids_from_records(subset.get("frame_ids"))
            if subset_ids:
                return subset_ids
        images = payload.get("images")
        if isinstance(images, dict):
            return _unique_sorted_frame_ids(list(images.keys()))
        return _frame_ids_from_records(images)
    if role == "physical_donor_exclusion_index":
        return _frame_ids_from_records(payload.get("items"))
    return []


def _directory_frame_ids(path: Path) -> list[str]:
    return _unique_sorted_frame_ids(
        [child.stem for child in path.iterdir() if child.is_file()]
    )


def _available_frame_ids(binding: RecoveryPreflightBinding) -> list[str]:
    path = Path(binding.effective_path or binding.path)
    if binding.expected_kind == "directory":
        return _directory_frame_ids(path)
    return _json_frame_ids(binding.role, path)


def _source_rgb_sha256_contract(input_manifest: dict[str, Any] | None) -> dict[str, str]:
    if input_manifest is None:
        return {}
    donor_contract = input_manifest.get("donor_contract")
    if not isinstance(donor_contract, dict):
        return {}
    assets = donor_contract.get("assets")
    if not isinstance(assets, dict):
        return {}
    result: dict[str, str] = {}
    for frame_id, asset in assets.items():
        if not isinstance(frame_id, str) or not isinstance(asset, dict):
            continue
        sha256 = asset.get("sha256")
        if isinstance(sha256, str):
            result[frame_id] = sha256
    return result


def _frame_file_for_id(directory: Path, frame_id: str) -> Path | None:
    matches = sorted(directory.glob(f"{frame_id}.*"))
    if len(matches) != 1 or not matches[0].is_file():
        return None
    return matches[0]


def _verify_source_rgb_content(
    alignment: RecoveryPreflightFrameAlignment,
    binding: RecoveryPreflightBinding,
    expected_sha256_by_frame: dict[str, str],
) -> RecoveryPreflightFrameAlignment:
    if binding.role != "source_rgb_frames" or alignment.status != "matched":
        return alignment
    expected = {
        frame_id: expected_sha256_by_frame[frame_id]
        for frame_id in alignment.available_frame_ids
        if frame_id in expected_sha256_by_frame
    }
    if not expected:
        return RecoveryPreflightFrameAlignment(
            **(alignment.model_dump(mode="json") | {
                "status": "unverified",
                "reason": "source RGB binding has no donor_contract SHA-256 contract",
            })
        )
    directory = Path(binding.effective_path or binding.path)
    mismatched: list[str] = []
    for frame_id, expected_sha in expected.items():
        frame_path = _frame_file_for_id(directory, frame_id)
        if frame_path is None or digest_path(frame_path).sha256 != expected_sha:
            mismatched.append(frame_id)
    if mismatched:
        return RecoveryPreflightFrameAlignment(
            **(
                alignment.model_dump(mode="json")
                | {
                    "status": "content_mismatch",
                    "content_mismatch_frame_ids": mismatched,
                    "reason": (
                        "source RGB files do not match the cumulative donor_contract SHA-256"
                    ),
                }
            )
        )
    return RecoveryPreflightFrameAlignment(
        **(
            alignment.model_dump(mode="json")
            | {
                "reason": (
                    "binding covers all recovery bundle frame ids and source RGB SHA-256 "
                    "matches donor_contract"
                ),
            }
        )
    )


def _frame_alignment(
    required_frame_ids: list[str],
    binding: RecoveryPreflightBinding,
) -> RecoveryPreflightFrameAlignment:
    if binding.status != "present":
        return RecoveryPreflightFrameAlignment(
            role=binding.role,
            status="binding_unavailable",
            expected_frame_count=len(required_frame_ids),
            available_frame_count=0,
            matched_frame_count=0,
            missing_frame_ids=required_frame_ids,
            reason="binding is not present",
        )
    if not required_frame_ids:
        return RecoveryPreflightFrameAlignment(
            role=binding.role,
            status="unverified",
            expected_frame_count=0,
            available_frame_count=0,
            matched_frame_count=0,
            reason="bundle does not declare recovery frame ids",
        )
    try:
        available = _available_frame_ids(binding)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return RecoveryPreflightFrameAlignment(
            role=binding.role,
            status="unverified",
            expected_frame_count=len(required_frame_ids),
            available_frame_count=0,
            matched_frame_count=0,
            missing_frame_ids=required_frame_ids,
            reason=f"failed to read frame ids: {exc}",
        )
    if not available:
        return RecoveryPreflightFrameAlignment(
            role=binding.role,
            status="unverified",
            expected_frame_count=len(required_frame_ids),
            available_frame_count=0,
            matched_frame_count=0,
            missing_frame_ids=required_frame_ids,
            reason="binding does not expose auditable frame ids",
        )
    available_set = set(available)
    required_set = set(required_frame_ids)
    missing = [frame_id for frame_id in required_frame_ids if frame_id not in available_set]
    matched = [frame_id for frame_id in required_frame_ids if frame_id in available_set]
    extra = [frame_id for frame_id in available if frame_id not in required_set]
    if missing:
        return RecoveryPreflightFrameAlignment(
            role=binding.role,
            status="missing_required_frames",
            expected_frame_count=len(required_frame_ids),
            available_frame_count=len(available),
            matched_frame_count=len(matched),
            missing_frame_ids=missing,
            extra_frame_ids=extra,
            available_frame_ids=available,
            reason="binding frame ids do not cover the recovery bundle frame ids",
        )
    return RecoveryPreflightFrameAlignment(
        role=binding.role,
        status="matched",
        expected_frame_count=len(required_frame_ids),
        available_frame_count=len(available),
        matched_frame_count=len(matched),
        extra_frame_ids=extra,
        available_frame_ids=available,
        reason="binding covers all recovery bundle frame ids",
    )


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


def _load_object(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be an object")
    return payload


def _require_gate(report: dict[str, Any], gate: str) -> None:
    gates = report.get("gates")
    if not isinstance(gates, dict) or gates.get(gate) is not True:
        raise ValueError(f"measured prefill report gate did not pass: {gate}")


def _verified_handoff_asset(
    record: dict[str, Any],
    path_key: str,
    sha_key: str,
    role: str,
) -> RecoveryHandoffAsset:
    path_value = record.get(path_key)
    sha_value = record.get(sha_key)
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"prefill frame record is missing path: {path_key}")
    if not isinstance(sha_value, str) or not sha_value:
        raise ValueError(f"prefill frame record is missing sha256: {sha_key}")
    digest = digest_path(path_value)
    if digest.sha256 != sha_value:
        raise ValueError(f"prefill frame record asset SHA-256 mismatch: {path_key}")
    return RecoveryHandoffAsset(
        path=str(digest.path),
        sha256=digest.sha256,
        size_bytes=digest.size_bytes,
        role=role,
    )


def _asset_from_path(path: str | Path, role: str) -> RecoveryHandoffAsset:
    digest = digest_path(path)
    return RecoveryHandoffAsset(
        path=str(digest.path),
        sha256=digest.sha256,
        size_bytes=digest.size_bytes,
        role=role,
    )


def _handoff_input_by_role(
    handoff: CompletionRecoveryResidualHandoff,
    role: str,
) -> RecoveryHandoffInput:
    matches = [item for item in handoff.inputs if item.role == role]
    if len(matches) != 1:
        raise ValueError(f"residual handoff must contain exactly one {role} input")
    return matches[0]


def _verify_handoff_input(input_item: RecoveryHandoffInput) -> Path:
    path = Path(input_item.path).expanduser().resolve()
    verified, actual_sha = _verify_declared_file(
        input_item.path,
        input_item.sha256,
        input_item.size_bytes,
    )
    if not verified:
        detail = f"; actual_sha256={actual_sha}" if actual_sha else ""
        raise ValueError(f"residual handoff input changed: {input_item.role}{detail}")
    return path


def _preflight_binding_by_role(
    preflight: CompletionRecoveryPreflight,
    role: str,
) -> RecoveryPreflightBinding:
    matches = [item for item in preflight.bindings if item.role == role]
    if len(matches) != 1:
        raise ValueError(f"recovery preflight must contain exactly one {role} binding")
    binding = matches[0]
    if binding.status != "present":
        raise ValueError(f"recovery preflight binding is not present: {role}")
    return binding


def _depth_path_for_frame(depth_dir: Path, frame_id: str) -> Path | None:
    for suffix in (".npy", ".npz"):
        path = depth_dir / f"{frame_id}{suffix}"
        if path.is_file():
            return path.resolve()
    return None


def _constrained_residual_frame(
    frame: RecoveryResidualHandoffFrame,
) -> RecoveryConstrainedResidualFrame:
    return RecoveryConstrainedResidualFrame(
        sequence_index=frame.sequence_index,
        frame_id=frame.frame_id,
        residual_mask=frame.residual_mask,
        measured_prefill_rgb=frame.measured_prefill_rgb,
        source_rgb=frame.source_rgb,
        removal_mask=frame.removal_mask,
        measured_multiview_pixels=frame.measured_multiview_pixels,
        residual_mask_pixels=frame.residual_mask_pixels,
    )


def _int_record_value(record: dict[str, Any], key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"prefill frame record requires non-negative integer: {key}")
    return value


def _float_record_value(record: dict[str, Any], key: str) -> float:
    value = record.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"prefill frame record requires numeric value: {key}")
    result = float(value)
    if not 0 <= result <= 1:
        raise ValueError(f"prefill frame record value must be in [0, 1]: {key}")
    return result


def _residual_handoff_frame(record: dict[str, Any]) -> RecoveryResidualHandoffFrame:
    frame_id = record.get("frame_id")
    if not isinstance(frame_id, str) or not frame_id:
        raise ValueError("prefill frame record is missing frame_id")
    donors = record.get("donors")
    if not isinstance(donors, list):
        raise ValueError("prefill frame record is missing donors")
    return RecoveryResidualHandoffFrame(
        sequence_index=_int_record_value(record, "sequence_index"),
        frame_id=frame_id,
        source_rgb=_verified_handoff_asset(
            record,
            "source_frame",
            "source_frame_sha256",
            "original_observed_rgb",
        ),
        removal_mask=_verified_handoff_asset(
            record,
            "removal_mask",
            "removal_mask_sha256",
            "cumulative_removal_mask",
        ),
        measured_prefill_rgb=_verified_handoff_asset(
            record,
            "prefill_frame",
            "prefill_frame_sha256",
            "measured_multiview_prefill_rgb",
        ),
        residual_mask=_verified_handoff_asset(
            record,
            "residual_mask",
            "residual_mask_sha256",
            "unresolved_residual_mask",
        ),
        support_visualization=_verified_handoff_asset(
            record,
            "support_visualization",
            "support_visualization_sha256",
            "measured_donor_support_visualization",
        ),
        measured_depth=_verified_handoff_asset(
            record,
            "measured_depth",
            "measured_depth_sha256",
            str(record.get("measured_depth_role") or "fused_multiview_measured_depth"),
        ),
        removal_mask_pixels=_int_record_value(record, "removal_mask_pixels"),
        measured_multiview_pixels=_int_record_value(record, "covered_pixels"),
        residual_mask_pixels=_int_record_value(record, "residual_mask_pixels"),
        coverage_fraction=_float_record_value(record, "coverage_fraction"),
        support_max=_int_record_value(record, "support_max"),
        measured_depth_valid_pixels=_int_record_value(record, "measured_depth_valid_pixels"),
        donor_count=len(donors),
    )


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
    required_frame_ids = _required_frame_ids(bundle)
    frame_alignment: list[RecoveryPreflightFrameAlignment] = []

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
        input_manifest_payload: dict[str, Any] | None = None
        input_binding = next(
            (binding for binding in bindings if binding.role == "input_manifest"),
            None,
        )
        if input_binding is not None and input_binding.status == "present":
            payload = json.loads(Path(input_binding.effective_path).read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                input_manifest_payload = payload
        expected_source_sha = _source_rgb_sha256_contract(input_manifest_payload)
        frame_alignment = [
            _verify_source_rgb_content(
                _frame_alignment(required_frame_ids, binding),
                binding,
                expected_source_sha,
            )
            for binding in bindings
        ]

    missing_roles = [
        item.role for item in bindings if item.status in {"missing", "kind_mismatch"}
    ]
    semantic_blocking_roles = [
        item.role for item in frame_alignment if item.status != "matched"
    ]
    if not bundle_input_verified or not clean_plate_report_verified:
        status = "blocked_input_mismatch"
    elif missing_roles:
        status = "blocked_missing_bindings"
    elif semantic_blocking_roles:
        status = "blocked_binding_semantics"
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
        "required_frame_ids": required_frame_ids,
        "frame_alignment": [item.model_dump(mode="json") for item in frame_alignment],
        "semantic_blocking_roles": semantic_blocking_roles,
        "notes": [
            (
                "Preflight checks local input availability, hashes, and frame-id "
                "alignment; it does not run recovery."
            ),
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


def materialize_completion_recovery_residual_handoff(
    bundle_path: str | Path,
    preflight_path: str | Path,
    prefill_report_path: str | Path,
    prefill_receipt_path: str | Path,
) -> CompletionRecoveryResidualHandoff:
    bundle_file = Path(bundle_path).expanduser().resolve()
    preflight_file = Path(preflight_path).expanduser().resolve()
    report_file = Path(prefill_report_path).expanduser().resolve()
    receipt_file = Path(prefill_receipt_path).expanduser().resolve()

    bundle = CompletionRecoveryBundle.model_validate(
        _load_object(bundle_file, "recovery bundle")
    )
    preflight = CompletionRecoveryPreflight.model_validate(
        _load_object(preflight_file, "recovery preflight")
    )
    report = _load_object(report_file, "measured prefill report")
    receipt = _load_object(receipt_file, "measured prefill receipt")

    report_digest = digest_path(report_file)
    if receipt.get("report_sha256") != report_digest.sha256:
        raise ValueError("measured prefill receipt report_sha256 does not match report file")
    if bundle.object_id != preflight.object_id:
        raise ValueError("recovery bundle and preflight object_id differ")
    if preflight.status != "passed":
        raise ValueError("recovery preflight must pass before residual handoff")
    if report.get("status") != "technical_passed":
        raise ValueError("measured prefill report must have status=technical_passed")
    if report.get("promotion_approved") is not False:
        raise ValueError("residual handoff only records non-promoted donor support output")
    _require_gate(report, "outside_removal_mask_rgb_exact")
    _require_gate(report, "all_residual_masks_subset_of_removal_masks")
    _require_gate(report, "fused_measured_metric_depth_materialized")

    pixel_provenance = report.get("pixel_provenance")
    if not isinstance(pixel_provenance, dict):
        raise ValueError("measured prefill report is missing pixel_provenance")
    unresolved = pixel_provenance.get("unresolved_unobserved_pixels")
    if not isinstance(unresolved, int) or isinstance(unresolved, bool) or unresolved <= 0:
        raise ValueError("residual handoff requires positive unresolved_unobserved_pixels")

    records = report.get("frame_records")
    if not isinstance(records, list) or not records:
        raise ValueError("measured prefill report must contain frame_records")
    frames = [
        _residual_handoff_frame(record) for record in records if isinstance(record, dict)
    ]
    if len(frames) != len(records):
        raise ValueError("measured prefill report contains non-object frame_records")
    required_ids = set(preflight.required_frame_ids)
    frame_ids = {frame.frame_id for frame in frames}
    if required_ids and frame_ids != required_ids:
        raise ValueError("residual handoff frame ids differ from recovery preflight")

    next_action = report.get("next_action")
    if not isinstance(next_action, dict):
        raise ValueError("measured prefill report is missing next_action")
    if (
        next_action.get("action")
        != "run_constrained_residual_completion_then_semantic_cross_view_review"
    ):
        raise ValueError("measured prefill report does not route to constrained residual")

    payload = {
        "object_id": bundle.object_id,
        "status": "ready_for_constrained_residual_completion",
        "completed_step_id": "01-donor_support",
        "next_step_id": "03-constrained_residual_completion",
        "deeper_rounds_blocked": True,
        "r1_promotion_approved": False,
        "output_claim": "measured_prefill_plus_residual_masks_only_no_r1_acceptance",
        "inputs": [
            _handoff_input(bundle_file, "completion_recovery_bundle").model_dump(mode="json"),
            _handoff_input(preflight_file, "completion_recovery_preflight").model_dump(
                mode="json"
            ),
            _handoff_input(report_file, "measured_prefill_report").model_dump(mode="json"),
            _handoff_input(receipt_file, "measured_prefill_receipt").model_dump(mode="json"),
        ],
        "frame_records": [frame.model_dump(mode="json") for frame in frames],
        "pixel_provenance": pixel_provenance,
        "gates": report.get("gates"),
        "next_action": next_action,
        "required_next_steps": [
            "Run constrained residual completion only inside residual_mask assets.",
            "Preserve source RGB outside removal masks and measured prefill pixels exactly.",
            "Run semantic cross-view review before any strict R1 acceptance attempt.",
            "Rerun depth/normal estimation on an accepted clean plate before PGSR or TSDF.",
        ],
        "notes": [
            (
                "Step 01 donor support completed technically, but this handoff is not "
                "an R1 acceptance artifact."
            ),
            (
                "Residual masks are unresolved unobserved pixels; they are not valid "
                "donor or geometry evidence."
            ),
            "R2-R4 remain blocked until strict R1 acceptance passes.",
        ],
    }
    return CompletionRecoveryResidualHandoff(
        handoff_sha256=digest_json(payload),
        **payload,
    )


def materialize_completion_recovery_constrained_residual_manifest(
    handoff_path: str | Path,
    *,
    mesh_path: str | Path,
    output_dir: str | Path,
    script_path: str | Path = "scripts/complete_planar_background.py",
    working_directory: str | Path | None = None,
) -> CompletionRecoveryConstrainedResidualManifest:
    handoff_file = Path(handoff_path).expanduser().resolve()
    handoff = CompletionRecoveryResidualHandoff.model_validate(
        _load_object(handoff_file, "residual handoff")
    )
    if handoff.status != "ready_for_constrained_residual_completion":
        raise ValueError("residual handoff is not ready for constrained completion")
    if handoff.r1_promotion_approved is not False:
        raise ValueError("constrained residual manifest cannot consume promoted R1 input")
    if handoff.next_step_id != "03-constrained_residual_completion":
        raise ValueError("residual handoff does not route to Step 03")

    preflight_file = _verify_handoff_input(
        _handoff_input_by_role(handoff, "completion_recovery_preflight")
    )
    prefill_report_file = _verify_handoff_input(
        _handoff_input_by_role(handoff, "measured_prefill_report")
    )
    preflight = CompletionRecoveryPreflight.model_validate(
        _load_object(preflight_file, "recovery preflight")
    )
    if preflight.status != "passed":
        raise ValueError("recovery preflight must still be passed for Step 03")

    camera_binding = _preflight_binding_by_role(preflight, "camera_info")
    depth_binding = _preflight_binding_by_role(preflight, "depth_arrays")
    camera_file = Path(camera_binding.effective_path).expanduser().resolve()
    depth_dir = Path(depth_binding.effective_path).expanduser().resolve()
    mesh_file = Path(mesh_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    workdir = (
        Path(working_directory).expanduser().resolve()
        if working_directory
        else Path.cwd().resolve()
    )
    script_file = Path(script_path).expanduser()
    if not script_file.is_absolute():
        script_file = (workdir / script_file).resolve()
    else:
        script_file = script_file.resolve()

    frame_ids = [frame.frame_id for frame in handoff.frame_records]
    missing_depth_frame_ids: list[str] = []
    depth_format_counts: dict[str, int] = {}
    for frame_id in frame_ids:
        depth_path = _depth_path_for_frame(depth_dir, frame_id)
        if depth_path is None:
            missing_depth_frame_ids.append(frame_id)
            continue
        suffix = depth_path.suffix.lstrip(".")
        depth_format_counts[suffix] = depth_format_counts.get(suffix, 0) + 1
    if missing_depth_frame_ids:
        raise ValueError(
            "depth_arrays binding does not cover Step 03 frame ids: "
            + ", ".join(missing_depth_frame_ids)
        )

    handoff_digest = digest_path(handoff_file)
    command_argv = [
        "uv",
        "run",
        "python",
        str(script_file),
        "--input-manifest",
        str(prefill_report_file),
        "--camera-info",
        str(camera_file),
        "--mesh",
        str(mesh_file),
        "--depth-dir",
        str(depth_dir),
        "--upstream-prefill-report",
        str(prefill_report_file),
        "--target-frame-ids",
        ",".join(frame_ids),
        "--output",
        str(output),
    ]
    payload = {
        "object_id": handoff.object_id,
        "status": "ready_for_structural_residual_completion",
        "planned_step_id": "03-constrained_residual_completion",
        "execution_status": "planned_not_run",
        "deeper_rounds_blocked": True,
        "r1_promotion_approved": False,
        "source_handoff": {
            "path": str(handoff_digest.path),
            "sha256": handoff_digest.sha256,
            "size_bytes": handoff_digest.size_bytes,
            "role": "completion_recovery_residual_handoff",
        },
        "bindings": [
            _asset_from_path(prefill_report_file, "step01_measured_prefill_report").model_dump(
                mode="json"
            ),
            _asset_from_path(camera_file, "camera_info").model_dump(mode="json"),
            _asset_from_path(depth_dir, "frame_id_depth_arrays").model_dump(mode="json"),
            _asset_from_path(mesh_file, "structural_tsdf_mesh_prior").model_dump(mode="json"),
            _asset_from_path(script_file, "constrained_residual_completion_script").model_dump(
                mode="json"
            ),
        ],
        "frame_records": [
            _constrained_residual_frame(frame).model_dump(mode="json")
            for frame in handoff.frame_records
        ],
        "depth_format_counts": dict(sorted(depth_format_counts.items())),
        "planned_output_dir": str(output),
        "working_directory": str(workdir),
        "command_argv": command_argv,
        "constraints": {
            "target_rgb_key": "prefill_frame",
            "target_mask_key": "residual_mask",
            "texture_rgb_key": "source_frame",
            "texture_mask_key": "removal_mask",
            "allowed_edit_region": "residual_mask",
            "outside_removal_mask_must_remain_source_rgb_exact": True,
            "measured_prefill_pixels_must_remain_exact": True,
            "residual_pixels_are_not_donor_or_geometry_evidence": True,
            "promotion_allowed": False,
        },
        "required_verification": [
            "Run complete_planar_background.py and inspect planar_background_report.json.",
            "Run semantic cross-view review before any strict R1 acceptance attempt.",
            "Verify outside-removal RGB exactness and residual-mask-only edits.",
            "Rerun fresh depth/normal estimation before PGSR or TSDF reconstruction.",
        ],
        "notes": [
            (
                "This manifest plans Step 03 only; it does not execute generation or "
                "accept R1."
            ),
            (
                "The depth_arrays binding is frame-id addressed and may contain .npy or "
                ".npz depth archives."
            ),
            "R2-R4 remain blocked until strict R1 acceptance passes.",
        ],
    }
    return CompletionRecoveryConstrainedResidualManifest(
        manifest_sha256=digest_json(payload),
        **payload,
    )
