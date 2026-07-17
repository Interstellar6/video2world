#!/usr/bin/env python3
"""Materialize a verified four-round clean-plate sequence for fresh DA3/Holi reconstruction."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

if __package__:
    from .compose_layered_clean_plate import INPUT_KIND, RECEIPT_KIND, REPORT_KIND
    from .compose_layered_clean_plate_sequence import (
        OBJECT_ORDER,
        ROUND_NAMES,
        ROUND_REMAINING,
        SEQUENCE_RECEIPT_KIND,
        SEQUENCE_REPORT_KIND,
    )
else:
    from compose_layered_clean_plate import INPUT_KIND, RECEIPT_KIND, REPORT_KIND
    from compose_layered_clean_plate_sequence import (
        OBJECT_ORDER,
        ROUND_NAMES,
        ROUND_REMAINING,
        SEQUENCE_RECEIPT_KIND,
        SEQUENCE_REPORT_KIND,
    )

EXPECTED_SEQUENCE_STATUS = "technical_passed_complete_sequence_with_limitations"
EXPECTED_ROUND_STATUSES = (
    "technical_passed_complete_partition",
    "technical_passed_complete_partition",
    "technical_passed_complete_partition",
    "technical_passed_complete_partition_with_limitations",
)
REQUIRED_SEQUENCE_GATES = (
    "all_inputs_hash_revalidated_after_path_rebase",
    "strict_four_round_order_executed",
    "fixed_removed_and_remaining_mappings_enforced",
    "every_round_unresolved_pixels_zero",
    "round2_plus_source_is_previous_composite_hash",
    "removed_objects_are_never_rendered_again",
    "round4_is_final_background_without_measured_donor",
    "round4_structural_background_covers_full_cumulative_mask",
    "limited_scope_and_failures_propagated",
)
REQUIRED_TRUE_ROUND_GATES = (
    "outside_removal_mask_rgb_exact",
    "removal_provenance_partition_exact",
    "render_composite_only_inside_measured_residual",
    "no_measured_donor_evidence_has_zero_measured_pixels",
    "final_background_entire_removal_is_structural",
    "previous_round_output_hash_bound",
    "limited_acceptance_scope_and_failures_propagated",
)
REQUIRED_FALSE_ROUND_GATES = (
    "generated_or_propainter_claimed_as_measured",
    "original_measured_donor_claimed_without_evidence",
)
FRAME_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class FinalFrame:
    sequence_index: int
    frame_id: str
    rgb_path: Path
    rgb_sha256: str
    width: int
    height: int


@dataclass(frozen=True)
class RoundEvidence:
    round_index: int
    name: str
    manifest_path: Path
    report_path: Path
    receipt_path: Path
    report: dict[str, Any]
    receipt: dict[str, Any]
    outputs_by_frame: dict[str, tuple[Path, str]]


@dataclass(frozen=True)
class ValidatedSequence:
    report_path: Path
    report_sha256: str
    report: dict[str, Any]
    receipt_path: Path
    receipt_sha256: str
    receipt: dict[str, Any]
    camera_info_path: Path
    camera_info_sha256: str
    camera_info: dict[str, Any]
    camera_lineage_sha256: str
    camera_binding_mode: str
    rounds: tuple[RoundEvidence, ...]
    frames: tuple[FinalFrame, ...]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def require_dict(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be an array")
    return value


def require_sha256(value: Any, label: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA-256 digest",
    )
    return value


def resolve_path(value: Any, *, relative_to: Path, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label} path is missing")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def asset_path(value: Any, *, relative_to: Path, label: str) -> Path:
    record = require_dict(value, label)
    path = resolve_path(record.get("path"), relative_to=relative_to, label=label)
    expected = require_sha256(record.get("sha256"), f"{label}.sha256")
    require(path.is_file(), f"{label} is missing: {path}")
    require(sha256_file(path) == expected, f"{label} SHA-256 mismatch")
    return path


def same_path(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve()


def frame_records(value: Any, *, label: str, expected_count: int) -> list[dict[str, Any]]:
    records = require_list(value, f"{label}.frame_records")
    require(len(records) == expected_count, f"{label} must contain {expected_count} frames")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for expected_index, raw in enumerate(records):
        record = require_dict(raw, f"{label} frame {expected_index}")
        require(
            record.get("sequence_index") == expected_index,
            f"{label} sequence indexes are not contiguous",
        )
        frame_id = record.get("frame_id")
        require(
            isinstance(frame_id, str) and FRAME_ID_PATTERN.fullmatch(frame_id) is not None,
            f"{label} has an unsafe frame ID",
        )
        require(frame_id not in seen, f"{label} has a duplicate frame ID: {frame_id}")
        seen.add(frame_id)
        result.append(record)
    return result


def validate_all_true_gates(value: Any, *, label: str) -> dict[str, Any]:
    gates = require_dict(value, label)
    require(bool(gates), f"{label} is empty")
    failed = sorted(key for key, gate in gates.items() if gate is not True)
    require(not failed, f"{label} contains non-passing gates: {failed}")
    return gates


def validate_round_gates(value: Any, *, round_index: int) -> dict[str, Any]:
    gates = require_dict(value, f"round {round_index} gates")
    missing = sorted(
        (set(REQUIRED_TRUE_ROUND_GATES) | set(REQUIRED_FALSE_ROUND_GATES)) - set(gates)
    )
    require(not missing, f"round {round_index} gates are missing: {missing}")
    wrong_true = sorted(key for key in REQUIRED_TRUE_ROUND_GATES if gates.get(key) is not True)
    wrong_false = sorted(key for key in REQUIRED_FALSE_ROUND_GATES if gates.get(key) is not False)
    require(not wrong_true, f"round {round_index} positive gates failed: {wrong_true}")
    require(not wrong_false, f"round {round_index} forbidden-claim gates failed: {wrong_false}")
    return gates


def validate_limited_acceptance(
    report: dict[str, Any],
    receipt: dict[str, Any],
) -> None:
    require(report.get("promotion_approved") is False, "sequence promotion must remain false")
    require(report.get("accepted_with_limitations") is True, "sequence is not limitation-bound")
    require(
        report.get("acceptance_status") == "accepted_with_limitations",
        "sequence acceptance status is invalid",
    )
    require(
        report.get("acceptance_scope") == "current_demo_only",
        "sequence acceptance scope must be current_demo_only",
    )
    limitations = require_list(report.get("limitations"), "sequence limitations")
    require(
        bool(limitations) and all(isinstance(item, str) and item for item in limitations),
        "sequence limitations must be non-empty strings",
    )
    require(
        bool(require_dict(report.get("failed_metrics"), "sequence failed_metrics")),
        "sequence must preserve failed metrics",
    )
    require(
        bool(require_dict(report.get("overridden_gates"), "sequence overridden_gates")),
        "sequence must preserve overridden gates",
    )
    for key in (
        "status",
        "promotion_approved",
        "accepted_with_limitations",
        "acceptance_status",
        "acceptance_scope",
        "limitations",
        "overridden_gates",
        "failed_metrics",
        "final_output_frame_set_sha256",
    ):
        require(receipt.get(key) == report.get(key), f"sequence receipt does not bind {key}")


def validate_sequence_receipt(
    report_path: Path,
    report: dict[str, Any],
    receipt_path: Path,
    receipt: dict[str, Any],
) -> None:
    require(report.get("kind") == SEQUENCE_REPORT_KIND, "sequence report kind is invalid")
    require(receipt.get("kind") == SEQUENCE_RECEIPT_KIND, "sequence receipt kind is invalid")
    require(
        report.get("status") == EXPECTED_SEQUENCE_STATUS,
        f"sequence status must be {EXPECTED_SEQUENCE_STATUS}",
    )
    require(receipt.get("status") == EXPECTED_SEQUENCE_STATUS, "sequence receipt status mismatch")
    require(
        receipt.get("report_sha256") == sha256_file(report_path),
        "sequence receipt does not bind the sequence report",
    )
    receipt_report_path = resolve_path(
        receipt.get("report"), relative_to=receipt_path.parent, label="sequence receipt report"
    )
    require(
        same_path(receipt_report_path, report_path), "sequence receipt points to another report"
    )
    require(
        receipt.get("all_round_manifests_reports_and_receipts_are_hash_bound") is True,
        "sequence receipt does not claim complete round hash binding",
    )
    validate_limited_acceptance(report, receipt)


def expected_round_kind(round_index: int) -> str:
    return "final_background" if round_index == 4 else "object_peel"


def validate_round_manifest(
    *,
    round_index: int,
    manifest_path: Path,
    manifest: dict[str, Any],
    report_records: list[dict[str, Any]],
    previous: RoundEvidence | None,
    expected_count: int,
) -> None:
    name = ROUND_NAMES[round_index - 1]
    removed = list(OBJECT_ORDER[:round_index])
    remaining = ROUND_REMAINING[name]
    require(manifest.get("kind") == INPUT_KIND, f"round {round_index} manifest kind is invalid")
    require(
        manifest.get("round_index") == round_index, f"round {round_index} manifest index mismatch"
    )
    require(
        manifest.get("round_kind") == expected_round_kind(round_index),
        f"round {round_index} manifest kind/order mismatch",
    )
    require(
        manifest.get("removed_object_ids") == removed,
        f"round {round_index} manifest removed-object order mismatch",
    )
    require(
        manifest.get("newly_removed_object_id") == removed[-1],
        f"round {round_index} newly removed object mismatch",
    )
    require(
        manifest.get("remaining_object_ids") == remaining,
        f"round {round_index} manifest remaining-object mapping mismatch",
    )
    source_contract = require_dict(manifest.get("source_contract"), "round source contract")
    require(source_contract.get("untracked_generated_rgb") is False, "untracked RGB is allowed")
    require(source_contract.get("propainter_rgb") is False, "ProPainter RGB is allowed")
    expected_role = "original_observed_rgb" if round_index == 1 else "previous_layered_composite"
    require(
        source_contract.get("role") == expected_role, f"round {round_index} source role mismatch"
    )

    manifest_records = frame_records(
        manifest.get("frame_records"),
        label=f"round {round_index} manifest",
        expected_count=expected_count,
    )
    require(
        [item["frame_id"] for item in manifest_records]
        == [item["frame_id"] for item in report_records],
        f"round {round_index} manifest/report frame order mismatch",
    )
    previous_value = manifest.get("previous_round")
    if previous is None:
        require(previous_value is None, "round 1 unexpectedly binds a predecessor")
    else:
        previous_record = require_dict(previous_value, f"round {round_index} predecessor")
        require(
            previous_record.get("round_index") == round_index - 1,
            f"round {round_index} predecessor index is not immediate",
        )
        bound_report = asset_path(
            previous_record.get("report"),
            relative_to=manifest_path.parent,
            label=f"round {round_index} predecessor report",
        )
        bound_receipt = asset_path(
            previous_record.get("receipt"),
            relative_to=manifest_path.parent,
            label=f"round {round_index} predecessor receipt",
        )
        require(same_path(bound_report, previous.report_path), "predecessor report path mismatch")
        require(
            same_path(bound_receipt, previous.receipt_path), "predecessor receipt path mismatch"
        )

    report_by_id = {record["frame_id"]: record for record in report_records}
    for record in manifest_records:
        frame_id = record["frame_id"]
        source_path = asset_path(
            record.get("source_rgb"),
            relative_to=manifest_path.parent,
            label=f"round {round_index} {frame_id} source RGB",
        )
        source_sha = require_sha256(
            require_dict(record.get("source_rgb"), "source RGB").get("sha256"),
            "source RGB sha256",
        )
        report_input_sha = require_dict(
            report_by_id[frame_id].get("inputs"), "round report frame inputs"
        ).get("source_rgb_sha256")
        require(report_input_sha == source_sha, f"round {round_index} {frame_id} source hash drift")
        layers = require_list(record.get("downstream_layers"), "downstream layers")
        require(
            [require_dict(layer, "downstream layer").get("layer_id") for layer in layers]
            == remaining,
            f"round {round_index} reintroduces or omits a downstream object",
        )
        if previous is not None:
            previous_path, previous_sha = previous.outputs_by_frame[frame_id]
            require(
                source_sha == previous_sha,
                f"round {round_index} {frame_id} predecessor hash mismatch",
            )
            require(
                same_path(source_path, previous_path),
                f"round {round_index} {frame_id} does not source the predecessor composite",
            )

    measured_contract = require_dict(
        manifest.get("measured_donor_contract"), f"round {round_index} measured donor contract"
    )
    if round_index == 4:
        require(
            measured_contract.get("role") == "no_measured_donor_evidence",
            "round 4 must not claim measured donor evidence",
        )
        for key in ("measured_pixels", "generated_pixels", "propainter_pixels"):
            require(measured_contract.get(key) == 0, f"round 4 {key} must be zero")
        require(measured_contract.get("report") is None, "round 4 attaches a measured report")
        require(measured_contract.get("receipt") is None, "round 4 attaches a measured receipt")


def validate_round(
    *,
    summary: dict[str, Any],
    sequence_report_path: Path,
    previous: RoundEvidence | None,
    expected_count: int,
) -> RoundEvidence:
    round_index = int(summary.get("round_index", -1))
    require(1 <= round_index <= 4, "sequence round index is invalid")
    name = ROUND_NAMES[round_index - 1]
    removed = list(OBJECT_ORDER[:round_index])
    remaining = ROUND_REMAINING[name]
    expected_status = EXPECTED_ROUND_STATUSES[round_index - 1]
    require(summary.get("round_name") == name, f"round {round_index} name/order mismatch")
    require(summary.get("round_kind") == expected_round_kind(round_index), "round kind mismatch")
    require(
        summary.get("removed_object_ids") == removed, f"round {round_index} removal order mismatch"
    )
    require(
        summary.get("remaining_object_ids") == remaining,
        f"round {round_index} remaining objects mismatch",
    )
    require(summary.get("status") == expected_status, f"round {round_index} status mismatch")
    require(summary.get("unresolved_pixels") == 0, f"round {round_index} has unresolved pixels")

    manifest_path = asset_path(
        summary.get("manifest"),
        relative_to=sequence_report_path.parent,
        label=f"round {round_index} manifest",
    )
    report_path = asset_path(
        summary.get("report"),
        relative_to=sequence_report_path.parent,
        label=f"round {round_index} report",
    )
    receipt_path = asset_path(
        summary.get("receipt"),
        relative_to=sequence_report_path.parent,
        label=f"round {round_index} receipt",
    )
    manifest = read_json(manifest_path)
    report = read_json(report_path)
    receipt = read_json(receipt_path)
    require(report.get("kind") == REPORT_KIND, f"round {round_index} report kind is invalid")
    require(receipt.get("kind") == RECEIPT_KIND, f"round {round_index} receipt kind is invalid")
    require(report.get("round_index") == round_index, f"round {round_index} report index mismatch")
    require(
        receipt.get("round_index") == round_index, f"round {round_index} receipt index mismatch"
    )
    require(
        report.get("round_kind") == expected_round_kind(round_index), "round report kind mismatch"
    )
    require(
        receipt.get("round_kind") == expected_round_kind(round_index), "round receipt kind mismatch"
    )
    require(report.get("removed_object_ids") == removed, "round report removal order mismatch")
    require(
        report.get("remaining_object_ids") == remaining, "round report remaining mapping mismatch"
    )
    require(report.get("status") == expected_status, f"round {round_index} report status mismatch")
    require(
        receipt.get("report_sha256") == sha256_file(report_path),
        "round receipt report hash mismatch",
    )
    receipt_report_path = resolve_path(
        receipt.get("report"), relative_to=receipt_path.parent, label="round receipt report"
    )
    require(same_path(receipt_report_path, report_path), "round receipt points to another report")
    require(
        receipt.get("input_manifest_sha256") == sha256_file(manifest_path),
        f"round {round_index} receipt manifest hash mismatch",
    )
    report_manifest_path = resolve_path(
        report.get("input_manifest"), relative_to=report_path.parent, label="round report manifest"
    )
    require(
        same_path(report_manifest_path, manifest_path), "round report points to another manifest"
    )
    require(
        report.get("input_manifest_sha256") == sha256_file(manifest_path),
        f"round {round_index} report manifest hash mismatch",
    )
    aggregate = require_dict(report.get("aggregate_counts"), "round aggregate counts")
    require(
        aggregate.get("unresolved_pixels") == 0, f"round {round_index} unresolved count is nonzero"
    )
    require(receipt.get("provenance_partition_exact") is True, "round provenance receipt failed")
    require(receipt.get("promotion_approved") is False, "round receipt claims promotion")
    validate_round_gates(report.get("gates"), round_index=round_index)

    records = frame_records(
        report.get("frame_records"),
        label=f"round {round_index} report",
        expected_count=expected_count,
    )
    outputs: dict[str, tuple[Path, str]] = {}
    output_set: list[dict[str, str]] = []
    for record in records:
        frame_id = record["frame_id"]
        output = require_dict(record.get("outputs"), "round frame outputs")
        rgb_record = require_dict(output.get("composite_rgb"), "round composite RGB")
        rgb_path = asset_path(
            rgb_record,
            relative_to=report_path.parent,
            label=f"round {round_index} {frame_id} composite RGB",
        )
        rgb_sha = require_sha256(rgb_record.get("sha256"), "composite RGB sha256")
        depth_sha = require_sha256(
            require_dict(output.get("composite_depth"), "composite depth").get("sha256"),
            "composite depth sha256",
        )
        labels_sha = require_sha256(
            require_dict(output.get("provenance_labels"), "provenance labels").get("sha256"),
            "provenance labels sha256",
        )
        outputs[frame_id] = (rgb_path, rgb_sha)
        output_set.append(
            {
                "frame_id": frame_id,
                "composite_rgb_sha256": rgb_sha,
                "composite_depth_sha256": depth_sha,
                "provenance_labels_sha256": labels_sha,
            }
        )
    output_set_sha = digest_json(output_set)
    require(
        report.get("output_frame_set_sha256") == output_set_sha, "round output set digest mismatch"
    )
    require(
        receipt.get("output_frame_set_sha256") == output_set_sha,
        "round receipt output set mismatch",
    )
    require(
        summary.get("output_frame_set_sha256") == output_set_sha,
        "sequence round output set mismatch",
    )

    evidence = RoundEvidence(
        round_index=round_index,
        name=name,
        manifest_path=manifest_path,
        report_path=report_path,
        receipt_path=receipt_path,
        report=report,
        receipt=receipt,
        outputs_by_frame=outputs,
    )
    validate_round_manifest(
        round_index=round_index,
        manifest_path=manifest_path,
        manifest=manifest,
        report_records=records,
        previous=previous,
        expected_count=expected_count,
    )
    previous_binding = report.get("previous_round_binding")
    if previous is None:
        require(previous_binding is None, "round 1 report unexpectedly binds a predecessor")
        require(
            receipt.get("previous_round_output_report_sha256") is None,
            "round 1 receipt binds predecessor",
        )
    else:
        binding = require_dict(previous_binding, f"round {round_index} previous report binding")
        require(binding.get("round_index") == round_index - 1, "report predecessor index mismatch")
        require(
            binding.get("report_sha256") == sha256_file(previous.report_path),
            "report predecessor hash mismatch",
        )
        require(
            binding.get("receipt_sha256") == sha256_file(previous.receipt_path),
            "receipt predecessor hash mismatch",
        )
        require(
            binding.get("output_frame_set_sha256")
            == previous.report.get("output_frame_set_sha256"),
            "predecessor output-set binding mismatch",
        )
        require(
            receipt.get("previous_round_output_report_sha256") == sha256_file(previous.report_path),
            "round receipt predecessor report mismatch",
        )
        require(
            receipt.get("previous_round_output_receipt_sha256")
            == sha256_file(previous.receipt_path),
            "round receipt predecessor receipt mismatch",
        )
    return evidence


def measured_camera_lineage(rounds: tuple[RoundEvidence, ...]) -> str:
    values: list[str] = []
    for evidence in rounds[:3]:
        binding = require_dict(
            evidence.report.get("measured_donor_report"),
            f"round {evidence.round_index} measured donor report binding",
        )
        measured_path = asset_path(
            binding,
            relative_to=evidence.report_path.parent,
            label=f"round {evidence.round_index} measured donor report",
        )
        measured = read_json(measured_path)
        values.append(require_sha256(measured.get("camera_info_sha256"), "measured camera sha256"))
    require(len(set(values)) == 1, "measured rounds use different camera calibration hashes")
    return values[0]


def validate_camera_info(
    path: Path,
    frames: tuple[FinalFrame, ...],
    measured_camera_sha256: str,
) -> tuple[dict[str, Any], str]:
    camera = read_json(path)
    actual_sha = sha256_file(path)
    subset_provenance = camera.get("subset_provenance")
    if actual_sha == measured_camera_sha256:
        binding_mode = "exact_measured_camera_info_sha256"
    else:
        subset = require_dict(subset_provenance, "camera subset_provenance")
        require(
            subset.get("source_camera_info_sha256") == measured_camera_sha256,
            "camera_info is not bound to the measured-round calibration",
        )
        require(
            subset.get("frame_ids") == [frame.frame_id for frame in frames],
            "camera subset frame IDs do not match the sequence",
        )
        require(subset.get("frame_count") == len(frames), "camera subset frame count mismatch")
        binding_mode = "validated_subset_of_measured_camera_info_sha256"

    require(
        camera.get("extrinsic_type") == "world_to_camera",
        "camera extrinsics are not world_to_camera",
    )
    extrinsics = require_dict(camera.get("extrinsic"), "camera_info.extrinsic")
    camera_ids = require_dict(camera.get("frame_camera_ids"), "camera_info.frame_camera_ids")
    intrinsics = require_dict(camera.get("intrinsics"), "camera_info.intrinsics")
    images = require_dict(camera.get("images"), "camera_info.images")
    selected_intrinsics: list[tuple[Any, ...]] = []
    frame_ids = [frame.frame_id for frame in frames]
    for frame in frames:
        frame_id = frame.frame_id
        require(frame_id in extrinsics, f"camera extrinsic is missing: {frame_id}")
        require(frame_id in images, f"camera image record is missing: {frame_id}")
        camera_id = str(camera_ids.get(frame_id, ""))
        intrinsic = require_dict(intrinsics.get(camera_id), f"camera intrinsic {camera_id}")
        require(intrinsic.get("model") == "PINHOLE", f"camera is not PINHOLE: {frame_id}")
        values = tuple(intrinsic.get(key) for key in ("fx", "fy", "cx", "cy", "w", "h"))
        require(
            all(isinstance(value, int | float) and not isinstance(value, bool) for value in values),
            f"camera intrinsic is incomplete: {frame_id}",
        )
        require(
            (int(intrinsic["w"]), int(intrinsic["h"])) == (frame.width, frame.height),
            f"camera dimensions do not match final RGB: {frame_id}",
        )
        selected_intrinsics.append(values)
        matrix = np.asarray(extrinsics[frame_id], dtype=np.float64)
        require(matrix.shape == (4, 4), f"camera matrix shape is invalid: {frame_id}")
        require(bool(np.all(np.isfinite(matrix))), f"camera matrix is non-finite: {frame_id}")
        require(
            bool(np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8)),
            f"camera homogeneous row is invalid: {frame_id}",
        )
        rotation = matrix[:3, :3]
        require(
            float(np.max(np.abs(rotation @ rotation.T - np.eye(3)))) <= 1e-4,
            f"camera rotation is not orthonormal: {frame_id}",
        )
        require(
            abs(float(np.linalg.det(rotation)) - 1.0) <= 1e-4,
            f"camera determinant failed: {frame_id}",
        )
    require(len(set(selected_intrinsics)) == 1, "selected frames do not use one PINHOLE intrinsic")
    require(set(extrinsics) >= set(frame_ids), "camera extrinsic frame set is incomplete")
    return camera, binding_mode


def final_frames(evidence: RoundEvidence, *, expected_count: int) -> tuple[FinalFrame, ...]:
    records = frame_records(
        evidence.report.get("frame_records"), label="round 4 report", expected_count=expected_count
    )
    result: list[FinalFrame] = []
    for record in records:
        frame_id = record["frame_id"]
        path, sha = evidence.outputs_by_frame[frame_id]
        try:
            with Image.open(path) as image:
                require(image.mode in {"RGB", "RGBA"}, f"final RGB mode is invalid: {frame_id}")
                width, height = image.size
                image.verify()
        except (OSError, SyntaxError) as exc:
            raise RuntimeError(f"final RGB is not a valid image: {frame_id}: {exc}") from exc
        require(width > 0 and height > 0, f"final RGB is empty: {frame_id}")
        result.append(
            FinalFrame(
                sequence_index=int(record["sequence_index"]),
                frame_id=frame_id,
                rgb_path=path,
                rgb_sha256=sha,
                width=width,
                height=height,
            )
        )
    return tuple(result)


def validate_inputs(
    sequence_report_path: Path,
    sequence_receipt_path: Path,
    camera_info_path: Path,
    *,
    expected_frame_count: int = 25,
) -> ValidatedSequence:
    require(expected_frame_count > 0, "expected frame count must be positive")
    report_path = sequence_report_path.expanduser().resolve()
    receipt_path = sequence_receipt_path.expanduser().resolve()
    camera_path = camera_info_path.expanduser().resolve()
    require(report_path.is_file(), f"sequence report is missing: {report_path}")
    require(receipt_path.is_file(), f"sequence receipt is missing: {receipt_path}")
    require(camera_path.is_file(), f"camera_info is missing: {camera_path}")
    report = read_json(report_path)
    receipt = read_json(receipt_path)
    validate_sequence_receipt(report_path, report, receipt_path, receipt)
    require(report.get("frame_count") == expected_frame_count, "sequence frame count mismatch")
    require(report.get("removed_object_order") == list(OBJECT_ORDER), "fixed object order mismatch")
    require(
        report.get("round_remaining_object_ids") == ROUND_REMAINING,
        "fixed remaining-object map mismatch",
    )
    gates = require_dict(report.get("gates"), "sequence gates")
    missing_gates = sorted(set(REQUIRED_SEQUENCE_GATES) - set(gates))
    require(not missing_gates, f"sequence gates are missing: {missing_gates}")
    validate_all_true_gates(gates, label="sequence gates")

    summaries = require_list(report.get("rounds"), "sequence rounds")
    require(len(summaries) == 4, "sequence must contain exactly four rounds")
    require(
        [require_dict(item, "sequence round").get("round_index") for item in summaries]
        == [1, 2, 3, 4],
        "sequence round order is not 1,2,3,4",
    )
    rounds: list[RoundEvidence] = []
    previous: RoundEvidence | None = None
    for raw in summaries:
        evidence = validate_round(
            summary=require_dict(raw, "sequence round"),
            sequence_report_path=report_path,
            previous=previous,
            expected_count=expected_frame_count,
        )
        rounds.append(evidence)
        previous = evidence
    round_tuple = tuple(rounds)
    r4 = round_tuple[-1]
    final_report_path = asset_path(
        receipt.get("final_round_report"),
        relative_to=receipt_path.parent,
        label="sequence final round report",
    )
    final_receipt_path = asset_path(
        receipt.get("final_round_receipt"),
        relative_to=receipt_path.parent,
        label="sequence final round receipt",
    )
    require(
        same_path(final_report_path, r4.report_path), "sequence receipt points to another R4 report"
    )
    require(
        same_path(final_receipt_path, r4.receipt_path),
        "sequence receipt points to another R4 receipt",
    )
    require(
        report.get("final_output_frame_set_sha256") == r4.report.get("output_frame_set_sha256"),
        "sequence final output set does not bind R4",
    )
    require(
        r4.report.get("accepted_with_limitations") is True
        and r4.report.get("acceptance_scope") == "current_demo_only"
        and r4.report.get("promotion_approved") is False
        and r4.report.get("demo_use_approved") is True
        and r4.report.get("eligible_as_round04_clean_plate") is False
        and r4.report.get("eligible_as_current_demo_round04_clean_plate") is True,
        "R4 is not the approved current-demo-only limited clean plate",
    )
    for key in ("limitations", "overridden_gates", "failed_metrics"):
        require(report.get(key) == r4.report.get(key), f"sequence does not preserve R4 {key}")

    frames = final_frames(r4, expected_count=expected_frame_count)
    lineage_sha = measured_camera_lineage(round_tuple)
    camera, binding_mode = validate_camera_info(camera_path, frames, lineage_sha)
    return ValidatedSequence(
        report_path=report_path,
        report_sha256=sha256_file(report_path),
        report=report,
        receipt_path=receipt_path,
        receipt_sha256=sha256_file(receipt_path),
        receipt=receipt,
        camera_info_path=camera_path,
        camera_info_sha256=sha256_file(camera_path),
        camera_info=camera,
        camera_lineage_sha256=lineage_sha,
        camera_binding_mode=binding_mode,
        rounds=round_tuple,
        frames=frames,
    )


def subset_camera_info(inputs: ValidatedSequence) -> dict[str, Any]:
    frame_ids = [frame.frame_id for frame in inputs.frames]
    camera = copy.deepcopy(inputs.camera_info)
    camera["extrinsic"] = {frame_id: camera["extrinsic"][frame_id] for frame_id in frame_ids}
    camera["frame_camera_ids"] = {
        frame_id: camera["frame_camera_ids"][frame_id] for frame_id in frame_ids
    }
    camera["images"] = {
        frame_id: {**copy.deepcopy(camera["images"][frame_id]), "name": f"{frame_id}.png"}
        for frame_id in frame_ids
    }
    used_camera_ids = {str(camera["frame_camera_ids"][frame_id]) for frame_id in frame_ids}
    camera["intrinsics"] = {
        camera_id: camera["intrinsics"][camera_id] for camera_id in sorted(used_camera_ids)
    }
    if len(used_camera_ids) == 1:
        camera["intrinsic"] = copy.deepcopy(camera["intrinsics"][next(iter(used_camera_ids))])
    camera["subset_provenance"] = {
        "source_camera_info_sha256": inputs.camera_info_sha256,
        "measured_camera_info_sha256": inputs.camera_lineage_sha256,
        "camera_binding_mode": inputs.camera_binding_mode,
        "frame_ids": frame_ids,
        "frame_count": len(frame_ids),
        "selection": "layered_clean_plate_sequence_final_round",
    }
    return camera


def build_holi_transforms(inputs: ValidatedSequence) -> tuple[dict[str, Any], dict[str, Any]]:
    camera = inputs.camera_info
    first_id = inputs.frames[0].frame_id
    camera_id = str(camera["frame_camera_ids"][first_id])
    intrinsic = camera["intrinsics"][camera_id]
    transform_frames: list[dict[str, Any]] = []
    max_roundtrip_error = 0.0
    max_rotation_error = 0.0
    determinants: list[float] = []
    for frame in inputs.frames:
        w2c = np.asarray(camera["extrinsic"][frame.frame_id], dtype=np.float64)
        c2w_colmap = np.linalg.inv(w2c)
        c2w_opengl = c2w_colmap.copy()
        c2w_opengl[:3, 1:3] *= -1.0
        recovered = c2w_opengl.copy()
        recovered[:3, 1:3] *= -1.0
        recovered_w2c = np.linalg.inv(recovered)
        max_roundtrip_error = max(max_roundtrip_error, float(np.max(np.abs(recovered_w2c - w2c))))
        rotation = w2c[:3, :3]
        max_rotation_error = max(
            max_rotation_error, float(np.max(np.abs(rotation @ rotation.T - np.eye(3))))
        )
        determinants.append(float(np.linalg.det(rotation)))
        transform_frames.append(
            {
                "frame_id": frame.frame_id,
                "file_path": f"{frame.frame_id}.png",
                "transform_matrix": c2w_opengl.tolist(),
            }
        )
    require(max_roundtrip_error <= 1e-7, "camera convention round-trip gate failed")
    require(max_rotation_error <= 1e-4, "camera rotation orthogonality gate failed")
    transforms = {
        "camera_model": "PINHOLE",
        "fl_x": float(intrinsic["fx"]),
        "fl_y": float(intrinsic["fy"]),
        "cx": float(intrinsic["cx"]),
        "cy": float(intrinsic["cy"]),
        "w": int(intrinsic["w"]),
        "h": int(intrinsic["h"]),
        "frames": transform_frames,
        "test_frames": [],
        "image_directory": "../resized_undistorted_images",
        "coordinate_convention": {
            "transform_matrix": "camera_to_world_opengl",
            "source": "camera_info extrinsic world_to_camera COLMAP",
            "conversion": "inverse(world_to_camera), then negate camera Y/Z columns",
            "holi_loader_behavior": "negates camera Y/Z columns again before inversion",
        },
    }
    validation = {
        "frame_count": len(transform_frames),
        "max_world_to_camera_roundtrip_abs_error": max_roundtrip_error,
        "max_rotation_orthogonality_abs_error": max_rotation_error,
        "rotation_determinant_min": min(determinants),
        "rotation_determinant_max": max(determinants),
        "passed": True,
    }
    return transforms, validation


def materialize_file(source: Path, destination: Path, storage_mode: str) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    require(not destination.exists(), f"destination already exists: {destination}")
    if storage_mode == "copy":
        shutil.copy2(source, destination)
    elif storage_mode == "hardlink":
        os.link(source, destination)
    else:  # pragma: no cover - argparse and public entrypoint validate this
        raise RuntimeError(f"unsupported storage mode: {storage_mode}")
    source_sha = sha256_file(source)
    destination_sha = sha256_file(destination)
    require(source_sha == destination_sha, f"materialized file hash mismatch: {destination}")
    return {
        "source": str(source),
        "destination": str(destination),
        "sha256": destination_sha,
        "bytes": destination.stat().st_size,
        "storage_mode": storage_mode,
        "hardlink_inode_match": (
            source.stat().st_ino == destination.stat().st_ino
            if storage_mode == "hardlink"
            else False
        ),
    }


def check_output_target(output: Path) -> None:
    require(not output.is_symlink(), f"output must not be a symlink: {output}")
    if not output.exists():
        return
    require(output.is_dir(), f"output is not a real directory: {output}")
    require(not any(output.iterdir()), f"output directory is not empty: {output}")


def materialize_package(
    *,
    sequence_report_path: Path,
    sequence_receipt_path: Path,
    camera_info_path: Path,
    output: Path,
    scene_id: str,
    storage_mode: str,
    expected_frame_count: int = 25,
) -> tuple[dict[str, Any], dict[str, Any]]:
    output = output.expanduser().resolve()
    require(storage_mode in {"copy", "hardlink"}, "storage mode must be copy or hardlink")
    require(FRAME_ID_PATTERN.fullmatch(scene_id) is not None, f"unsafe scene_id: {scene_id!r}")
    check_output_target(output)
    inputs = validate_inputs(
        sequence_report_path,
        sequence_receipt_path,
        camera_info_path,
        expected_frame_count=expected_frame_count,
    )
    transforms, camera_validation = build_holi_transforms(inputs)
    camera_subset = subset_camera_info(inputs)

    temporary = output.with_name(f".{output.name}.materializing-{os.getpid()}")
    require(not temporary.exists(), f"temporary output already exists: {temporary}")
    materializations: list[dict[str, Any]] = []
    try:
        temporary.mkdir(parents=True)
        images_dir = temporary / "dslr/resized_undistorted_images"
        manifest_frames: list[dict[str, Any]] = []
        for frame in inputs.frames:
            destination = images_dir / f"{frame.frame_id}.png"
            record = materialize_file(frame.rgb_path, destination, storage_mode)
            record["destination"] = destination.relative_to(temporary).as_posix()
            materializations.append(record)
            manifest_frames.append(
                {
                    "sequence_index": frame.sequence_index,
                    "frame_id": frame.frame_id,
                    "clean_rgb": {
                        "path": destination.relative_to(temporary).as_posix(),
                        "sha256": frame.rgb_sha256,
                        "source_path": str(frame.rgb_path),
                        "source_sha256": frame.rgb_sha256,
                        "source_round": 4,
                        "source_role": "final_layered_composite_rgb",
                        "width": frame.width,
                        "height": frame.height,
                    },
                }
            )

        camera_output = temporary / "camera_info.json"
        transforms_output = temporary / "dslr/nerfstudio/transforms_undistorted.json"
        write_json(camera_output, camera_subset)
        write_json(transforms_output, transforms)
        source_sequence = {
            "report": {
                "path": str(inputs.report_path),
                "sha256": inputs.report_sha256,
                "status": inputs.report["status"],
            },
            "receipt": {
                "path": str(inputs.receipt_path),
                "sha256": inputs.receipt_sha256,
            },
            "removed_object_order": list(OBJECT_ORDER),
            "round_names": list(ROUND_NAMES),
            "final_output_frame_set_sha256": inputs.report["final_output_frame_set_sha256"],
            "acceptance_scope": "current_demo_only",
            "accepted_with_limitations": True,
            "promotion_approved": False,
            "limitations": copy.deepcopy(inputs.report["limitations"]),
            "overridden_gates": copy.deepcopy(inputs.report["overridden_gates"]),
            "failed_metrics": copy.deepcopy(inputs.report["failed_metrics"]),
        }
        manifest = {
            "schema_version": 1,
            "kind": "video2world.layered_sequence_reconstruction_input",
            "scene_id": scene_id,
            "status": "ready_for_fresh_depth_and_normal_estimation_current_demo_only",
            "acceptance_scope": "current_demo_only",
            "promotion_approved": False,
            "source_layered_sequence": source_sequence,
            "frames": manifest_frames,
            "camera": {
                "subset_camera_info": {
                    "path": "camera_info.json",
                    "sha256": sha256_file(camera_output),
                    "source_sha256": inputs.camera_info_sha256,
                    "measured_camera_info_sha256": inputs.camera_lineage_sha256,
                    "binding_mode": inputs.camera_binding_mode,
                    "extrinsic_type": "world_to_camera",
                },
                "holi_pgsr_transforms": {
                    "path": "dslr/nerfstudio/transforms_undistorted.json",
                    "sha256": sha256_file(transforms_output),
                    "transform_matrix": "camera_to_world_opengl",
                    "validation": camera_validation,
                },
            },
            "storage_mode": storage_mode,
            "sequence_validation": {
                "strict_four_round_order": True,
                "fixed_removed_and_remaining_mappings": True,
                "every_round_unresolved_pixels_zero": True,
                "round2_to_round4_predecessor_paths_and_hashes_revalidated_per_frame": True,
                "round4_report_and_receipt_hash_bound": True,
                "final_composite_rgb_assets_rehashed": True,
                "sequence_gates_all_true": True,
            },
            "provenance_contract": {
                "final_rgb_source": "round04_bed final layered composite",
                "old_geometry_depth_packaged": False,
                "old_composite_depth_packaged": False,
                "legacy_da3_depth_packaged": False,
                "fresh_depth_required": True,
                "fresh_normals_required": True,
                "scope": "current_demo_only",
                "promotion_approved": False,
            },
            "stage_status": {
                "accepted_clean_rgb": "ready_current_demo_only",
                "camera_subset": "ready",
                "holi_pgsr_camera_transforms": "ready",
                "fresh_depth_estimation": "required_pending",
                "fresh_normal_estimation": "required_pending",
                "pgsr_optimization": "pending",
                "tsdf_fusion": "pending",
            },
            "forbidden_claims": [
                "the limited clean plate is approved for general promotion",
                "upstream geometry or composite depth is valid after final RGB composition",
                "legacy DA3 depth is packaged or reusable as fresh clean-scene depth",
                "PGSR or TSDF has already run on this package",
            ],
        }
        manifest_output = temporary / "manifest.json"
        write_json(manifest_output, manifest)
        receipt = {
            "schema_version": 1,
            "kind": "video2world.layered_sequence_reconstruction_input_receipt",
            "created_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 - mil8 is 3.10
            "status": "materialized_current_demo_only_fresh_depth_required",
            "scene_id": scene_id,
            "output": str(output),
            "storage_mode": storage_mode,
            "frame_count": len(inputs.frames),
            "materialized_source_file_count": len(materializations),
            "materializations": materializations,
            "source_sequence_report_sha256": inputs.report_sha256,
            "source_sequence_receipt_sha256": inputs.receipt_sha256,
            "acceptance_scope": "current_demo_only",
            "promotion_approved": False,
            "fresh_depth_required": True,
            "old_geometry_depth_packaged": False,
            "old_composite_depth_packaged": False,
            "camera_validation": camera_validation,
            "manifest": {"path": "manifest.json", "sha256": sha256_file(manifest_output)},
            "outputs": {
                "camera_info_sha256": sha256_file(camera_output),
                "transforms_undistorted_sha256": sha256_file(transforms_output),
            },
            "pending": ["fresh_depth", "fresh_normals", "PGSR", "TSDF"],
        }
        write_json(temporary / "receipt.json", receipt)
        if output.exists():
            output.rmdir()
        os.replace(temporary, output)
        return manifest, receipt
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-report", type=Path, required=True)
    parser.add_argument("--sequence-receipt", type=Path, required=True)
    parser.add_argument("--camera-info", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--storage-mode", choices=("copy", "hardlink"), default="copy")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest, receipt = materialize_package(
        sequence_report_path=args.sequence_report,
        sequence_receipt_path=args.sequence_receipt,
        camera_info_path=args.camera_info,
        output=args.output,
        scene_id=args.scene_id,
        storage_mode=args.storage_mode,
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "scene_id": manifest["scene_id"],
                "frame_count": receipt["frame_count"],
                "output": str(args.output.resolve()),
                "acceptance_scope": "current_demo_only",
                "next_stage": "fresh_depth_and_normal_estimation",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
