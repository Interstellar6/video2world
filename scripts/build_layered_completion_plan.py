#!/usr/bin/env python3
"""Build a deterministic layered-completion plan from audited local evidence."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from video2world.completion import (
    CompletionPlanEvidence,
    CompletionRound,
    LayeredCompletionPlan,
    build_layered_completion_plan,
    load_occlusion_graph,
    load_scene_inventory,
    write_layered_completion_plan,
)
from video2world.hashing import sha256_file
from video2world.models import Sha256, StrictModel


class EvidenceInput(StrictModel):
    role: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    path: str = Field(min_length=1)
    expected_sha256: Sha256
    claim_scope: str = Field(min_length=1)
    usage: Literal["evidence_only", "next_round_input", "display_only"] = "evidence_only"


class RoundStatusInput(StrictModel):
    kind: Literal["object_layer", "final_background"]
    target_ids: list[str]
    status: Literal[
        "planned",
        "running",
        "passed",
        "passed_with_limitations",
        "failed",
        "blocked",
    ]
    status_reason: str | None = None
    acceptance_scope: (
        Literal["full_round", "current_demo_only", "usable_for_reinspection_only"] | None
    ) = None
    evidence: list[EvidenceInput] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    next_layer_candidate_ids: list[str] = Field(default_factory=list)
    next_layer_candidates_evidence_role: str | None = None
    next_layer_mask_index_evidence_role: str | None = None

    @model_validator(mode="after")
    def validate_status_evidence(self) -> RoundStatusInput:
        roles = [item.role for item in self.evidence]
        if len(roles) != len(set(roles)):
            raise ValueError("round evidence roles must be unique")
        successful = self.status in {"passed", "passed_with_limitations"}
        if successful:
            required = {"output_clean_plate", "round_quality_report"}
            missing = sorted(required - set(roles))
            if missing:
                raise ValueError(f"successful round is missing evidence roles: {missing}")
            output = next(item for item in self.evidence if item.role == "output_clean_plate")
            if output.usage != "next_round_input":
                raise ValueError(
                    "output_clean_plate evidence must explicitly declare usage=next_round_input"
                )
        for item in self.evidence:
            if item.role.startswith("display_candidate") and item.usage != "display_only":
                raise ValueError(
                    f"display candidate evidence must declare usage=display_only: {item.role}"
                )
            if item.role.startswith("rejected_") and item.usage != "evidence_only":
                raise ValueError(
                    "rejected evidence cannot be propagated beyond audit scope: "
                    f"{item.role}"
                )
        if self.status == "passed_with_limitations" and not self.limitations:
            raise ValueError("passed_with_limitations requires explicit limitations")
        if self.status == "passed_with_limitations" and self.acceptance_scope not in {
            "current_demo_only",
            "usable_for_reinspection_only",
        }:
            raise ValueError("passed_with_limitations requires a limited acceptance_scope")
        next_layer_fields = (
            bool(self.next_layer_candidate_ids),
            self.next_layer_candidates_evidence_role is not None,
            self.next_layer_mask_index_evidence_role is not None,
        )
        if len(set(next_layer_fields)) != 1:
            raise ValueError(
                "next-layer candidates, association role, and mask-index role must all be set "
                "or all be absent"
            )
        if len(self.next_layer_candidate_ids) != len(set(self.next_layer_candidate_ids)):
            raise ValueError("next_layer_candidate_ids cannot contain duplicates")
        for role in (
            self.next_layer_candidates_evidence_role,
            self.next_layer_mask_index_evidence_role,
        ):
            if role is not None and role not in roles:
                raise ValueError(f"next-layer evidence role is absent from evidence: {role}")
        return self


class PlanStatusInput(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["video2world.layered_completion_status_input"] = (
        "video2world.layered_completion_status_input"
    )
    created_at: str = Field(min_length=1)
    target_order: list[str] = Field(min_length=1)
    rounds: list[RoundStatusInput] = Field(min_length=1)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_target_order(self) -> PlanStatusInput:
        if len(self.target_order) != len(set(self.target_order)):
            raise ValueError("target_order cannot contain duplicates")
        keys = [(item.kind, tuple(item.target_ids)) for item in self.rounds]
        if len(keys) != len(set(keys)):
            raise ValueError("round status entries must be unique")
        return self


def read_status_input(path: Path) -> PlanStatusInput:
    value = json.loads(path.read_text(encoding="utf-8"))
    return PlanStatusInput.model_validate(value)


def repo_uri(project_root: Path, path: Path) -> str:
    relative = path.relative_to(project_root).as_posix()
    return f"repo://video2world/{relative}"


def resolve_evidence_path(project_root: Path, item: EvidenceInput) -> Path:
    declared = Path(item.path)
    if declared.is_absolute():
        raise ValueError(f"evidence path must be repository-relative: {item.path}")
    resolved = (project_root / declared).resolve()
    if not resolved.is_relative_to(project_root):
        raise ValueError(f"evidence path escapes the repository: {item.path}")
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"evidence must be a regular repository file: {item.path}")
    return resolved


def resolve_evidence(project_root: Path, item: EvidenceInput) -> CompletionPlanEvidence:
    resolved = resolve_evidence_path(project_root, item)
    actual_sha256, size_bytes = sha256_file(resolved)
    if actual_sha256 != item.expected_sha256:
        raise ValueError(
            f"evidence hash mismatch for {item.path}: "
            f"expected {item.expected_sha256}, got {actual_sha256}"
        )
    return CompletionPlanEvidence(
        role=item.role,
        uri=repo_uri(project_root, resolved),
        sha256=actual_sha256,
        size_bytes=size_bytes,
        claim_scope=item.claim_scope,
        usage=item.usage,
    )


def association_receipt_summary(association: dict[str, Any]) -> str:
    details: list[str] = []
    status = association.get("status")
    if isinstance(status, str) and status:
        details.append(f"status={status}")
    gates = association.get("gates")
    if isinstance(gates, dict):
        passed = gates.get("passed")
        if isinstance(passed, bool):
            details.append(f"gates.passed={passed}")
    actual_candidates = association.get("next_layer_target_ids")
    if isinstance(actual_candidates, list):
        candidates = [item for item in actual_candidates if isinstance(item, str) and item]
        if candidates:
            details.append(f"next_layer_target_ids={','.join(candidates)}")
    promotion_blocker = association.get("promotion_blocker")
    if isinstance(promotion_blocker, str) and promotion_blocker:
        details.append(f"promotion_blocker={promotion_blocker}")
    next_action = association.get("next_action")
    if isinstance(next_action, dict):
        action = next_action.get("action")
        blocker = next_action.get("blocker")
        if isinstance(action, str) and action:
            details.append(f"next_action={action}")
        if isinstance(blocker, str) and blocker:
            details.append(f"next_blocker={blocker}")
        missing = next_action.get("missing_target_ids")
        if isinstance(missing, list) and missing:
            ids = [item for item in missing if isinstance(item, str) and item]
            if ids:
                details.append(f"missing_target_ids={','.join(ids)}")
        failed_gates = next_action.get("failed_gates")
        if isinstance(failed_gates, list) and failed_gates:
            gates = [item for item in failed_gates if isinstance(item, str) and item]
            if gates:
                details.append(f"failed_gates={','.join(gates)}")
        failed_frame_ids = next_action.get("failed_frame_ids")
        if isinstance(failed_frame_ids, list) and failed_frame_ids:
            frames = [
                frame_id
                for frame_id in failed_frame_ids
                if isinstance(frame_id, str) and frame_id
            ]
            if frames:
                details.append(f"failed_frame_ids={','.join(frames)}")
        first_failed_frame_id = next_action.get("first_failed_frame_id")
        if isinstance(first_failed_frame_id, str) and first_failed_frame_id:
            details.append(f"first_failed_frame_id={first_failed_frame_id}")
        for key in (
            "failed_pair_ids",
            "failed_triplet_center_frame_ids",
            "not_evaluable_pair_ids",
            "not_evaluable_triplet_center_frame_ids",
        ):
            values = next_action.get(key)
            if isinstance(values, list) and values:
                kept = [value for value in values if isinstance(value, str) and value]
                if kept:
                    details.append(f"{key}={','.join(kept)}")
        no_support_frame_ids = next_action.get("no_support_frame_ids")
        if isinstance(no_support_frame_ids, list) and no_support_frame_ids:
            frames = [
                frame_id
                for frame_id in no_support_frame_ids
                if isinstance(frame_id, str) and frame_id
            ]
            if frames:
                details.append(f"no_support_frame_ids={','.join(frames)}")
        unresolved = next_action.get("unresolved_unobserved_pixels")
        if isinstance(unresolved, int):
            details.append(f"unresolved_unobserved_pixels={unresolved}")
    if not details:
        return ""
    return " [" + "; ".join(details) + "]"


def validate_next_layer_candidates(
    project_root: Path,
    status: RoundStatusInput,
    evidence_inputs: dict[str, EvidenceInput],
    evidence: dict[str, CompletionPlanEvidence],
) -> str | None:
    association_role = status.next_layer_candidates_evidence_role
    mask_index_role = status.next_layer_mask_index_evidence_role
    if association_role is None or mask_index_role is None:
        return None
    association_path = resolve_evidence_path(project_root, evidence_inputs[association_role])
    association = json.loads(association_path.read_text(encoding="utf-8"))
    if not isinstance(association, dict):
        raise ValueError("next-layer association evidence must be a JSON object")
    actual_candidates = association.get("next_layer_target_ids")
    if actual_candidates != status.next_layer_candidate_ids:
        raise ValueError(
            "next_layer_candidate_ids do not match the association receipt: "
            f"expected {actual_candidates}, got {status.next_layer_candidate_ids}"
        )
    if (
        association.get("status") != "passed"
        or association.get("gates", {}).get("passed") is not True
    ):
        raise ValueError(
            "next-layer association receipt did not pass its technical gates"
            + association_receipt_summary(association)
        )
    associated_mask_sha = association.get("sources", {}).get("sam3_mask_index", {}).get("sha256")
    if associated_mask_sha != evidence[mask_index_role].sha256:
        raise ValueError("reinspection mask-index evidence does not match the association receipt")
    return evidence[association_role].uri


def infer_terminal_status(
    rounds: list[CompletionRound],
) -> Literal["planned", "running", "passed", "passed_with_limitations", "blocked"]:
    statuses = [item.status for item in rounds]
    if any(status in {"failed", "blocked"} for status in statuses):
        return "blocked"
    if all(status == "passed" for status in statuses):
        return "passed"
    if all(status in {"passed", "passed_with_limitations"} for status in statuses):
        return "passed_with_limitations"
    if any(status != "planned" for status in statuses):
        return "running"
    return "planned"


def build_plan(
    *,
    project_root: Path,
    inventory_path: Path,
    graph_path: Path,
    status_path: Path,
) -> LayeredCompletionPlan:
    project_root = project_root.resolve()
    inventory = load_scene_inventory(inventory_path)
    graph = load_occlusion_graph(graph_path)
    status_input = read_status_input(status_path)

    created_at = datetime.fromisoformat(status_input.created_at)
    base = build_layered_completion_plan(
        inventory,
        graph,
        created_at=created_at,
        max_parallel_targets_per_round=1,
        target_order=status_input.target_order,
    )
    status_by_key = {(item.kind, tuple(item.target_ids)): item for item in status_input.rounds}
    expected_keys = {(item.kind, tuple(item.target_ids)) for item in base.rounds}
    supplied_keys = set(status_by_key)
    if supplied_keys != expected_keys:
        raise ValueError(
            "round status input must cover the generated plan exactly; "
            f"missing={sorted(expected_keys - supplied_keys)}, "
            f"unexpected={sorted(supplied_keys - expected_keys)}"
        )

    rounds: list[CompletionRound] = []
    preceding_output_clean_plate_uri: str | None = None
    for round_item in base.rounds:
        status = status_by_key[(round_item.kind, tuple(round_item.target_ids))]
        evidence = [resolve_evidence(project_root, item) for item in status.evidence]
        by_role = {item.role: item for item in evidence}
        evidence_inputs = {item.role: item for item in status.evidence}
        next_layer_evidence_uri = validate_next_layer_candidates(
            project_root,
            status,
            evidence_inputs,
            by_role,
        )
        output_clean_plate_uri = (
            by_role["output_clean_plate"].uri if "output_clean_plate" in by_role else None
        )
        payload = round_item.model_dump(mode="json")
        payload.update(
            {
                "status": status.status,
                "status_reason": status.status_reason,
                "acceptance_scope": status.acceptance_scope,
                "evidence": [item.model_dump(mode="json") for item in evidence],
                "limitations": status.limitations,
                "input_clean_plate_uri": (
                    preceding_output_clean_plate_uri if round_item.index > 1 else None
                ),
                "output_clean_plate_uri": output_clean_plate_uri,
                "quality_report_uri": (
                    by_role["round_quality_report"].uri
                    if "round_quality_report" in by_role
                    else None
                ),
                "next_layer_candidate_ids": status.next_layer_candidate_ids,
                "next_layer_candidates_evidence_uri": next_layer_evidence_uri,
            }
        )
        rounds.append(CompletionRound.model_validate(payload))
        preceding_output_clean_plate_uri = output_clean_plate_uri

    status_sha256, _ = sha256_file(status_path)
    plan_payload = base.model_dump(mode="json")
    plan_payload.update(
        {
            "rounds": [item.model_dump(mode="json") for item in rounds],
            "terminal_status": infer_terminal_status(rounds),
            "round_status_input_uri": repo_uri(project_root, status_path.resolve()),
            "round_status_input_sha256": status_sha256,
            "limitations": [*base.limitations, *status_input.limitations],
        }
    )
    return LayeredCompletionPlan.model_validate(plan_payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--occlusion-graph", type=Path, required=True)
    parser.add_argument("--round-status", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = build_plan(
        project_root=args.project_root,
        inventory_path=args.inventory,
        graph_path=args.occlusion_graph,
        status_path=args.round_status,
    )
    write_layered_completion_plan(args.output, plan)
    print(
        json.dumps(
            {
                "status": "written",
                "output": str(args.output.resolve()),
                "rounds": [item.status for item in plan.rounds],
                "terminal_status": plan.terminal_status,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
