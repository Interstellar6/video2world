#!/usr/bin/env python3
"""Materialize an accepted clean plate as a fail-closed DA3/Holi input package."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ACCEPTED_TEXTURE_STATUS = "accepted_for_round04_clean_plate"
FRAME_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
REQUIRED_TEXTURE_GATES = (
    "measured_atlas_texels_rgb_exact",
    "outside_synthetic_mask_rgb_exact",
    "protected_neighbors_outside_synthetic_mask_rgb_exact",
    "all_synthetic_pixels_assigned_from_shared_atlas",
    "same_atlas_texel_has_identical_rgb_across_views",
    "rejected_texture_inputs_not_promotable",
)
REQUIRED_BOUNDARY_GATES = (
    "wall_boundary_color_continuity",
    "wall_boundary_low_frequency_gradient_continuity",
)
LIMITED_ACCEPTANCE_SCOPE = "current_demo_only"
LIMITED_VISUAL_QUALITY = "accepted_with_limitations"
PENDING_VISUAL_QUALITY = "pending_human_or_vlm_review"
ALLOWED_OVERRIDE_GATES = frozenset((*REQUIRED_BOUNDARY_GATES, "visual_quality"))
REAL_BOUNDARY_METRIC_KEYS = ("threshold_p95_abs_rgb_delta", "observed_maximum")
GENERIC_BOUNDARY_METRIC_KEYS = ("threshold", "observed")


@dataclass(frozen=True)
class SourceFrame:
    sequence_index: int
    frame_id: str
    completed_frame: Path
    completed_frame_sha256: str
    source_frame: Path
    source_frame_sha256: str
    synthetic_mask: Path
    synthetic_mask_sha256: str
    geometry_depth: Path
    geometry_depth_sha256: str
    width: int
    height: int
    synthetic_pixels: int


@dataclass(frozen=True)
class ValidatedInputs:
    texture_report_path: Path
    texture_report_sha256: str
    texture_report: dict[str, Any]
    geometry_report_path: Path
    geometry_report_sha256: str
    geometry_report: dict[str, Any]
    camera_info_path: Path
    camera_info_sha256: str
    camera_info: dict[str, Any]
    frames: tuple[SourceFrame, ...]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def acceptance_failure_summary(report: dict[str, Any]) -> str:
    details: list[str] = []
    status = report.get("status")
    if isinstance(status, str) and status:
        details.append(f"status={status}")
    promotion_blocker = report.get("promotion_blocker")
    if isinstance(promotion_blocker, str) and promotion_blocker:
        details.append(f"promotion_blocker={promotion_blocker}")
    for key in (
        "promotion_approved",
        "eligible_as_round04_clean_plate",
        "accepted_with_limitations",
        "eligible_as_current_demo_round04_clean_plate",
    ):
        value = report.get(key)
        if isinstance(value, bool):
            details.append(f"{key}={value}")
    next_action = report.get("next_action")
    if isinstance(next_action, dict):
        action = next_action.get("action")
        blocker = next_action.get("blocker")
        if isinstance(action, str) and action:
            details.append(f"next_action={action}")
        if isinstance(blocker, str) and blocker:
            details.append(f"next_blocker={blocker}")
        blocking_groups = next_action.get("blocking_gate_groups")
        if isinstance(blocking_groups, list) and blocking_groups:
            groups = [group for group in blocking_groups if isinstance(group, str) and group]
            if groups:
                details.append(f"blocking_gate_groups={','.join(groups)}")
        failed_gates = next_action.get("failed_gates")
        if isinstance(failed_gates, list) and failed_gates:
            gates = [gate for gate in failed_gates if isinstance(gate, str) and gate]
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
    gates = report.get("gates")
    if isinstance(gates, dict):
        failed = [
            key
            for key, value in gates.items()
            if value is False or (isinstance(value, dict) and value.get("passed") is False)
        ]
        if failed:
            details.append(f"report_failed_gates={','.join(sorted(failed))}")
        visual_quality = gates.get("visual_quality")
        if isinstance(visual_quality, str) and visual_quality:
            details.append(f"visual_quality={visual_quality}")
    if not details:
        return ""
    return " [" + "; ".join(details) + "]"


def require_acceptance(condition: bool, message: str, report: dict[str, Any]) -> None:
    if not condition:
        raise RuntimeError(message + acceptance_failure_summary(report))


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


def resolve_record_path(value: Any, *, relative_to: Path, label: str) -> Path:
    if isinstance(value, dict):
        value = value.get("path")
    require(isinstance(value, str) and value, f"{label} path is missing")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    resolved = path.resolve()
    require(resolved.is_file(), f"{label} does not exist: {resolved}")
    return resolved


def require_digest(path: Path, expected: Any, label: str) -> str:
    require(
        isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected) is not None,
        f"{label} has no valid sha256",
    )
    actual = sha256_file(path)
    require(actual == expected, f"{label} sha256 mismatch: {actual} != {expected}")
    return actual


def _gate_passed(value: Any) -> bool:
    if value is True:
        return True
    return isinstance(value, dict) and value.get("passed") is True


def _nonempty_string_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and item.strip() for item in value)
    )


def _validate_failed_boundary_evidence(value: Any, key: str) -> None:
    require(
        isinstance(value, dict) and value.get("passed") is False,
        f"limited planar texture boundary gate is not an explicit failure: {key}",
    )
    if any(metric_key in value for metric_key in REAL_BOUNDARY_METRIC_KEYS):
        threshold_key, observed_key = REAL_BOUNDARY_METRIC_KEYS
    else:
        threshold_key, observed_key = GENERIC_BOUNDARY_METRIC_KEYS
    require(
        threshold_key in value and observed_key in value,
        f"limited planar texture boundary failure lacks known metric evidence: {key}",
    )
    threshold = value[threshold_key]
    observed = value[observed_key]
    require(
        isinstance(threshold, (int, float))
        and not isinstance(threshold, bool)
        and isinstance(observed, (int, float))
        and not isinstance(observed, bool)
        and math.isfinite(float(threshold))
        and math.isfinite(float(observed)),
        f"limited planar texture boundary metrics must be finite numbers: {key}",
    )
    require(
        float(observed) > float(threshold),
        f"limited planar texture boundary observed value must exceed threshold: {key}",
    )


def _limited_acceptance_metadata(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("accepted_with_limitations") is not True:
        return {}
    overridden_gates = report["overridden_gates"]
    return {
        "acceptance_scope": LIMITED_ACCEPTANCE_SCOPE,
        "accepted_with_limitations": True,
        "demo_use_approved": True,
        "eligible_as_current_demo_round04_clean_plate": True,
        "decision": LIMITED_VISUAL_QUALITY,
        "override_gate_keys": sorted(overridden_gates),
        "overridden_gates": copy.deepcopy(overridden_gates),
    }


def _validate_limited_user_review_binding(
    acceptance: dict[str, Any], overridden_gates: dict[str, Any]
) -> None:
    candidate_binding = acceptance.get("candidate_report")
    require(
        isinstance(candidate_binding, dict),
        "limited planar texture candidate binding is missing",
    )
    require(
        candidate_binding.get("status") == "texture_candidate_review_pending"
        and candidate_binding.get("promotion_approved") is False
        and candidate_binding.get("eligible_as_round04_clean_plate") is False,
        "limited planar texture candidate binding has invalid pre-acceptance state",
    )
    candidate_path = resolve_record_path(
        candidate_binding.get("path"),
        relative_to=Path.cwd(),
        label="limited planar texture candidate report",
    )
    candidate_sha = require_digest(
        candidate_path,
        candidate_binding.get("sha256"),
        "limited planar texture candidate report",
    )

    visual_binding = acceptance.get("visual_review")
    require(
        isinstance(visual_binding, dict),
        "limited planar texture visual review binding is missing",
    )
    require(
        visual_binding.get("status") == "completed"
        and visual_binding.get("decision") == LIMITED_VISUAL_QUALITY,
        "limited planar texture acceptance decision is invalid",
    )
    visual_review_path = resolve_record_path(
        visual_binding.get("path"),
        relative_to=Path.cwd(),
        label="limited planar texture visual review",
    )
    require_digest(
        visual_review_path,
        visual_binding.get("sha256"),
        "limited planar texture visual review",
    )
    visual_review = read_json(visual_review_path)
    require(
        visual_review.get("schema_version") == 1
        and visual_review.get("kind") == "video2world.planar_texture_visual_review"
        and visual_review.get("status") == "completed"
        and visual_review.get("decision") == LIMITED_VISUAL_QUALITY
        and visual_review.get("acceptance_scope") == LIMITED_ACCEPTANCE_SCOPE,
        "limited planar texture user review contract is invalid",
    )
    require(
        visual_review.get("candidate_report_sha256") == candidate_sha,
        "limited planar texture user review is bound to a different candidate",
    )
    require(
        visual_review.get("review_authority") == acceptance.get("review_authority"),
        "limited planar texture user review authority binding changed",
    )
    require(
        visual_review.get("limitations") == acceptance.get("review_limitations"),
        "limited planar texture user review limitations binding changed",
    )
    review_override_keys = visual_review.get("override_gate_keys")
    require(
        isinstance(review_override_keys, list)
        and len(review_override_keys) == len(set(review_override_keys))
        and set(review_override_keys) == set(overridden_gates),
        "limited planar texture user review override binding changed",
    )


def validate_acceptance(report: dict[str, Any]) -> None:
    require_acceptance(
        report.get("status") == ACCEPTED_TEXTURE_STATUS,
        f"planar texture status must equal {ACCEPTED_TEXTURE_STATUS!r}",
        report,
    )
    require_acceptance(
        report.get("promotion_blocker") in {None, ""},
        "accepted planar texture report still has a promotion blocker",
        report,
    )
    gates = report.get("gates")
    require_acceptance(isinstance(gates, dict), "planar texture gates are missing", report)
    for key in REQUIRED_TEXTURE_GATES:
        require_acceptance(
            _gate_passed(gates.get(key)), f"planar texture gate did not pass: {key}", report
        )
    require_acceptance(
        gates.get("new_depth_normal_estimation_before_pgsr") == "required_after_visual_acceptance",
        "planar texture report does not require fresh depth/normal before PGSR",
        report,
    )

    limited_flag = report.get("accepted_with_limitations")
    limited = limited_flag is True
    if not limited:
        require(
            limited_flag is None or limited_flag is False,
            "planar texture limited acceptance flag is invalid",
        )
        require_acceptance(
            report.get("promotion_approved") is True,
            "planar texture promotion is not approved",
            report,
        )
        require_acceptance(
            report.get("eligible_as_round04_clean_plate") is True,
            "planar texture is not eligible as the Round 4 clean plate",
            report,
        )
        require(
            report.get("acceptance_scope") is None or report.get("acceptance_scope") == "",
            "strict planar texture report cannot carry a limited acceptance scope",
        )
        require(
            report.get("overridden_gates") is None,
            "strict planar texture report cannot carry overridden gates",
        )
        for key in REQUIRED_BOUNDARY_GATES:
            require_acceptance(
                _gate_passed(gates.get(key)),
                f"planar texture gate did not pass: {key}",
                report,
            )
        require_acceptance(
            gates.get("visual_quality") in {"passed", "accepted"},
            "planar texture visual quality was not explicitly accepted",
            report,
        )
        return

    require_acceptance(
        report.get("promotion_approved") is False,
        "limited planar texture cannot approve general promotion",
        report,
    )
    require_acceptance(
        report.get("eligible_as_round04_clean_plate") is False,
        "limited planar texture cannot claim general Round 4 eligibility",
        report,
    )
    require_acceptance(
        report.get("demo_use_approved") is True,
        "limited planar texture demo use is not approved",
        report,
    )
    require_acceptance(
        report.get("eligible_as_current_demo_round04_clean_plate") is True,
        "limited planar texture is not eligible for the current demo Round 4 clean plate",
        report,
    )
    require_acceptance(
        report.get("acceptance_scope") == LIMITED_ACCEPTANCE_SCOPE,
        "limited planar texture acceptance_scope must be current_demo_only",
        report,
    )
    require_acceptance(
        gates.get("visual_quality") == LIMITED_VISUAL_QUALITY,
        "limited planar texture visual quality must be accepted_with_limitations",
        report,
    )
    overridden_gates = report.get("overridden_gates")
    require(
        isinstance(overridden_gates, dict) and overridden_gates,
        "limited planar texture overridden_gates are missing",
    )
    unknown = set(overridden_gates) - ALLOWED_OVERRIDE_GATES
    require(
        not unknown,
        f"limited planar texture has forbidden overridden gates: {sorted(unknown)}",
    )
    require(
        overridden_gates.get("visual_quality") == PENDING_VISUAL_QUALITY,
        "limited planar texture did not preserve the original visual-quality gate",
    )
    failed_boundary_gates: set[str] = set()
    for key in REQUIRED_BOUNDARY_GATES:
        value = gates.get(key)
        if _gate_passed(value):
            require(
                key not in overridden_gates,
                f"limited planar texture cannot override a passed boundary gate: {key}",
            )
            continue
        _validate_failed_boundary_evidence(value, key)
        require(
            overridden_gates.get(key) == value,
            f"limited planar texture did not preserve failed boundary gate evidence: {key}",
        )
        failed_boundary_gates.add(key)
    require(
        set(overridden_gates) == {"visual_quality", *failed_boundary_gates},
        "limited planar texture overridden gates do not exactly match failed boundary gates",
    )

    acceptance = report.get("acceptance")
    require(isinstance(acceptance, dict), "limited planar texture acceptance record is missing")
    require(
        acceptance.get("acceptance_scope") == LIMITED_ACCEPTANCE_SCOPE,
        "limited planar texture acceptance record scope is invalid",
    )
    require(
        acceptance.get("accepted_with_limitations") is True,
        "limited planar texture acceptance record flag is invalid",
    )
    require(
        acceptance.get("promotion_approved") is False
        and acceptance.get("eligible_as_round04_clean_plate") is False,
        "limited planar texture acceptance record cannot approve general promotion",
    )
    require(
        acceptance.get("demo_use_approved") is True
        and acceptance.get("eligible_as_current_demo_round04_clean_plate") is True,
        "limited planar texture acceptance record lacks current-demo approval",
    )
    require(
        acceptance.get("overridden_gates") == overridden_gates,
        "limited planar texture acceptance record changed overridden gates",
    )
    override_gate_keys = acceptance.get("override_gate_keys")
    require(
        isinstance(override_gate_keys, list)
        and len(override_gate_keys) == len(set(override_gate_keys))
        and set(override_gate_keys) == set(overridden_gates),
        "limited planar texture acceptance record override keys are invalid",
    )
    authority = acceptance.get("review_authority")
    require(
        isinstance(authority, dict)
        and authority.get("kind") == "user"
        and authority.get("scope") == LIMITED_ACCEPTANCE_SCOPE,
        "limited planar texture acceptance authority is invalid",
    )
    require(
        _nonempty_string_list(acceptance.get("review_limitations")),
        "limited planar texture review limitations are missing",
    )
    _validate_limited_user_review_binding(acceptance, overridden_gates)


def _frame_id(value: Any) -> str:
    require(value is not None, "frame_id is missing")
    frame_id = str(value)
    require(FRAME_ID_PATTERN.fullmatch(frame_id) is not None, f"unsafe frame_id: {frame_id!r}")
    require(frame_id not in {".", ".."}, f"unsafe frame_id: {frame_id!r}")
    return frame_id


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image.convert("L"), dtype=np.uint8)
    require(bool(np.all((values == 0) | (values == 255))), f"mask is not binary: {path}")
    return values > 0


def validate_frame_record(
    texture_record: dict[str, Any],
    geometry_record: dict[str, Any],
    *,
    texture_dir: Path,
    geometry_dir: Path,
) -> SourceFrame:
    frame_id = _frame_id(texture_record.get("frame_id"))
    require(
        frame_id == _frame_id(geometry_record.get("frame_id")),
        f"texture/geometry frame_id mismatch: {frame_id}",
    )
    sequence_index = int(texture_record.get("sequence_index", -1))
    require(
        sequence_index == int(geometry_record.get("sequence_index", -2)),
        f"texture/geometry sequence mismatch: {frame_id}",
    )
    require(
        texture_record.get("outside_synthetic_mask_rgb_exact") is True,
        f"outside-mask exact flag failed: {frame_id}",
    )
    require(
        texture_record.get("all_synthetic_pixels_assigned") is True,
        f"synthetic assignment flag failed: {frame_id}",
    )
    require(
        geometry_record.get("outside_removal_mask_rgb_exact") is True,
        f"geometry outside-removal exact flag failed: {frame_id}",
    )
    require(
        geometry_record.get("texture_partition_exact") is True,
        f"geometry texture partition failed: {frame_id}",
    )

    completed = resolve_record_path(
        texture_record.get("completed_frame"),
        relative_to=texture_dir,
        label=f"{frame_id} completed frame",
    )
    source = resolve_record_path(
        texture_record.get("source_frame"),
        relative_to=texture_dir,
        label=f"{frame_id} source frame",
    )
    synthetic_mask = resolve_record_path(
        texture_record.get("synthetic_mask"),
        relative_to=texture_dir,
        label=f"{frame_id} synthetic mask",
    )
    geometry_depth = resolve_record_path(
        geometry_record.get("geometry_depth"),
        relative_to=geometry_dir,
        label=f"{frame_id} geometry depth",
    )
    completed_sha = require_digest(
        completed,
        texture_record.get("completed_frame_sha256"),
        f"{frame_id} completed frame",
    )
    source_sha = require_digest(
        source,
        texture_record.get("source_frame_sha256"),
        f"{frame_id} source frame",
    )
    mask_sha = require_digest(
        synthetic_mask,
        texture_record.get("synthetic_mask_sha256"),
        f"{frame_id} synthetic mask",
    )
    depth_sha = require_digest(
        geometry_depth,
        geometry_record.get("geometry_depth_sha256"),
        f"{frame_id} geometry depth",
    )
    require(completed.suffix.lower() == ".png", f"completed frame is not PNG: {frame_id}")
    require(synthetic_mask.suffix.lower() == ".png", f"synthetic mask is not PNG: {frame_id}")
    require(geometry_depth.suffix.lower() == ".npz", f"geometry depth is not NPZ: {frame_id}")

    completed_rgb = _load_rgb(completed)
    source_rgb = _load_rgb(source)
    mask = _load_mask(synthetic_mask)
    require(completed_rgb.shape == source_rgb.shape, f"RGB dimensions differ: {frame_id}")
    require(mask.shape == completed_rgb.shape[:2], f"mask dimensions differ: {frame_id}")
    require(
        bool(np.array_equal(completed_rgb[~mask], source_rgb[~mask])),
        f"outside-mask RGB is not byte-exact: {frame_id}",
    )
    synthetic_pixels = int(mask.sum())
    require(
        synthetic_pixels == int(texture_record.get("synthetic_pixels", -1)),
        f"synthetic pixel count mismatch: {frame_id}",
    )
    require(
        synthetic_pixels
        == int(texture_record.get("synthetic_pixels_assigned_from_shared_atlas", -2)),
        f"assigned synthetic pixel count mismatch: {frame_id}",
    )
    with np.load(geometry_depth) as archive:
        require(set(archive.files) == {"depth"}, f"geometry depth keys differ: {frame_id}")
        depth = np.asarray(archive["depth"], dtype=np.float32)
    require(depth.shape == mask.shape, f"geometry depth dimensions differ: {frame_id}")
    require(
        bool(np.all(np.isfinite(depth[mask])) and np.all(depth[mask] > 0)),
        f"synthetic pixels lack positive structural geometry depth: {frame_id}",
    )
    return SourceFrame(
        sequence_index=sequence_index,
        frame_id=frame_id,
        completed_frame=completed,
        completed_frame_sha256=completed_sha,
        source_frame=source,
        source_frame_sha256=source_sha,
        synthetic_mask=synthetic_mask,
        synthetic_mask_sha256=mask_sha,
        geometry_depth=geometry_depth,
        geometry_depth_sha256=depth_sha,
        width=int(completed_rgb.shape[1]),
        height=int(completed_rgb.shape[0]),
        synthetic_pixels=synthetic_pixels,
    )


def validate_camera_info(camera_info: dict[str, Any], frame_ids: tuple[str, ...]) -> None:
    require(
        camera_info.get("extrinsic_type") == "world_to_camera",
        "camera extrinsics are not world_to_camera",
    )
    extrinsics = camera_info.get("extrinsic")
    frame_camera_ids = camera_info.get("frame_camera_ids")
    intrinsics = camera_info.get("intrinsics")
    require(isinstance(extrinsics, dict), "camera_info.extrinsic is missing")
    require(isinstance(frame_camera_ids, dict), "camera_info.frame_camera_ids is missing")
    require(isinstance(intrinsics, dict), "camera_info.intrinsics is missing")
    selected_intrinsics: list[tuple[Any, ...]] = []
    for frame_id in frame_ids:
        require(frame_id in extrinsics, f"camera extrinsic is missing: {frame_id}")
        camera_id = str(frame_camera_ids.get(frame_id, ""))
        intrinsic = intrinsics.get(camera_id)
        require(isinstance(intrinsic, dict), f"camera intrinsic is missing: {frame_id}")
        require(intrinsic.get("model") == "PINHOLE", f"camera is not PINHOLE: {frame_id}")
        values = tuple(intrinsic.get(key) for key in ("fx", "fy", "cx", "cy", "w", "h"))
        require(
            all(isinstance(value, int | float) for value in values),
            f"intrinsic is incomplete: {frame_id}",
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
            f"camera rotation determinant failed: {frame_id}",
        )
    require(
        len(set(selected_intrinsics)) == 1,
        (
            "selected frames use multiple intrinsics; current Holi/PGSR package requires one "
            "PINHOLE camera"
        ),
    )


def validate_inputs(
    texture_report_path: Path,
    camera_info_path: Path,
) -> ValidatedInputs:
    texture_report_path = texture_report_path.expanduser().resolve()
    camera_info_path = camera_info_path.expanduser().resolve()
    require(texture_report_path.is_file(), f"texture report is missing: {texture_report_path}")
    require(camera_info_path.is_file(), f"camera_info is missing: {camera_info_path}")
    texture_report = read_json(texture_report_path)
    validate_acceptance(texture_report)
    texture_report_sha = sha256_file(texture_report_path)
    geometry_report_path = resolve_record_path(
        texture_report.get("planar_geometry_report"),
        relative_to=texture_report_path.parent,
        label="planar geometry report",
    )
    geometry_report_sha = require_digest(
        geometry_report_path,
        texture_report.get("planar_geometry_report_sha256"),
        "planar geometry report",
    )
    geometry_report = read_json(geometry_report_path)
    require(
        geometry_report.get("status") == "geometry_completed_texture_pending",
        "planar geometry report status is invalid",
    )
    require(
        geometry_report.get("camera_extrinsic_type") == "world_to_camera",
        "planar geometry camera convention is invalid",
    )
    geometry_gates = geometry_report.get("gates", {})
    require(
        _gate_passed(geometry_gates.get("minimum_geometry_coverage")),
        "planar geometry coverage gate failed",
    )
    require(
        geometry_gates.get("outside_removal_mask_rgb_exact") is True,
        "planar geometry outside-mask exact gate failed",
    )
    require(
        geometry_gates.get("texture_masks_partition_removal_mask") is True,
        "planar geometry texture partition gate failed",
    )
    camera_info_sha = sha256_file(camera_info_path)
    require(
        camera_info_sha == geometry_report.get("camera_info_sha256"),
        "explicit camera_info does not match the geometry report",
    )
    camera_info = read_json(camera_info_path)
    texture_records = texture_report.get("frame_records")
    geometry_records = geometry_report.get("frame_records")
    require(
        isinstance(texture_records, list) and texture_records, "texture frame records are empty"
    )
    require(
        isinstance(geometry_records, list) and geometry_records, "geometry frame records are empty"
    )
    require(
        all(isinstance(record, dict) for record in geometry_records),
        "geometry frame record is not an object",
    )
    geometry_by_id = {_frame_id(record.get("frame_id")): record for record in geometry_records}
    require(len(geometry_by_id) == len(geometry_records), "geometry frame IDs are duplicated")
    frames: list[SourceFrame] = []
    seen: set[str] = set()
    for expected_sequence, texture_record in enumerate(texture_records):
        require(isinstance(texture_record, dict), "texture frame record is not an object")
        frame_id = _frame_id(texture_record.get("frame_id"))
        require(frame_id not in seen, f"texture frame ID is duplicated: {frame_id}")
        seen.add(frame_id)
        require(frame_id in geometry_by_id, f"geometry record is missing: {frame_id}")
        frame = validate_frame_record(
            texture_record,
            geometry_by_id[frame_id],
            texture_dir=texture_report_path.parent,
            geometry_dir=geometry_report_path.parent,
        )
        require(
            frame.sequence_index == expected_sequence, "texture sequence indexes are not contiguous"
        )
        frames.append(frame)
    frame_ids = tuple(frame.frame_id for frame in frames)
    validate_camera_info(camera_info, frame_ids)
    for frame in frames:
        camera_id = str(camera_info["frame_camera_ids"][frame.frame_id])
        intrinsic = camera_info["intrinsics"][camera_id]
        require(
            (frame.width, frame.height) == (int(intrinsic["w"]), int(intrinsic["h"])),
            f"clean RGB dimensions do not match camera calibration: {frame.frame_id}",
        )
    return ValidatedInputs(
        texture_report_path=texture_report_path,
        texture_report_sha256=texture_report_sha,
        texture_report=texture_report,
        geometry_report_path=geometry_report_path,
        geometry_report_sha256=geometry_report_sha,
        geometry_report=geometry_report,
        camera_info_path=camera_info_path,
        camera_info_sha256=camera_info_sha,
        camera_info=camera_info,
        frames=tuple(frames),
    )


def subset_camera_info(inputs: ValidatedInputs) -> dict[str, Any]:
    frame_ids = [frame.frame_id for frame in inputs.frames]
    camera = copy.deepcopy(inputs.camera_info)
    camera["extrinsic"] = {frame_id: camera["extrinsic"][frame_id] for frame_id in frame_ids}
    camera["frame_camera_ids"] = {
        frame_id: camera["frame_camera_ids"][frame_id] for frame_id in frame_ids
    }
    images = camera.get("images")
    if isinstance(images, dict):
        require(
            all(frame_id in images for frame_id in frame_ids),
            "camera_info.images is missing one or more selected frames",
        )
        camera["images"] = {
            frame_id: {**copy.deepcopy(images[frame_id]), "name": f"{frame_id}.png"}
            for frame_id in frame_ids
            if frame_id in images
        }
    used_camera_ids = {str(camera["frame_camera_ids"][frame_id]) for frame_id in frame_ids}
    camera["intrinsics"] = {
        camera_id: camera["intrinsics"][camera_id] for camera_id in sorted(used_camera_ids)
    }
    if len(used_camera_ids) == 1:
        camera["intrinsic"] = copy.deepcopy(camera["intrinsics"][next(iter(used_camera_ids))])
    camera["subset_provenance"] = {
        "source_camera_info_sha256": inputs.camera_info_sha256,
        "frame_ids": frame_ids,
        "frame_count": len(frame_ids),
        "selection": "accepted_planar_texture_report_frame_records",
    }
    return camera


def build_holi_transforms(
    inputs: ValidatedInputs,
) -> tuple[dict[str, Any], dict[str, Any]]:
    camera = inputs.camera_info
    first_id = inputs.frames[0].frame_id
    camera_id = str(camera["frame_camera_ids"][first_id])
    intrinsic = camera["intrinsics"][camera_id]
    frames: list[dict[str, Any]] = []
    max_roundtrip_error = 0.0
    max_rotation_error = 0.0
    determinants: list[float] = []
    for source in inputs.frames:
        w2c = np.asarray(camera["extrinsic"][source.frame_id], dtype=np.float64)
        c2w_colmap = np.linalg.inv(w2c)
        c2w_opengl = c2w_colmap.copy()
        c2w_opengl[:3, 1:3] *= -1.0
        recovered = c2w_opengl.copy()
        recovered[:3, 1:3] *= -1.0
        recovered_w2c = np.linalg.inv(recovered)
        max_roundtrip_error = max(
            max_roundtrip_error,
            float(np.max(np.abs(recovered_w2c - w2c))),
        )
        rotation = w2c[:3, :3]
        max_rotation_error = max(
            max_rotation_error,
            float(np.max(np.abs(rotation @ rotation.T - np.eye(3)))),
        )
        determinants.append(float(np.linalg.det(rotation)))
        frames.append(
            {
                "frame_id": source.frame_id,
                "file_path": f"{source.frame_id}.png",
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
        "frames": frames,
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
        "frame_count": len(frames),
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
    else:  # pragma: no cover - argparse and callers validate this
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


def _check_output_target(output: Path) -> None:
    require(not output.is_symlink(), f"output must not be a symlink: {output}")
    if not output.exists():
        return
    require(output.is_dir(), f"output is not a real directory: {output}")
    require(not any(output.iterdir()), f"output directory is not empty: {output}")


def materialize_package(
    *,
    texture_report_path: Path,
    camera_info_path: Path,
    output: Path,
    scene_id: str,
    storage_mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    output = output.expanduser().resolve()
    require(storage_mode in {"copy", "hardlink"}, "storage mode must be copy or hardlink")
    require(FRAME_ID_PATTERN.fullmatch(scene_id) is not None, f"unsafe scene_id: {scene_id!r}")
    _check_output_target(output)
    inputs = validate_inputs(texture_report_path, camera_info_path)
    transforms, camera_validation = build_holi_transforms(inputs)
    camera_subset = subset_camera_info(inputs)

    temporary = output.with_name(f".{output.name}.materializing-{os.getpid()}")
    require(not temporary.exists(), f"temporary output already exists: {temporary}")
    materializations: list[dict[str, Any]] = []
    try:
        temporary.mkdir(parents=True)
        images_dir = temporary / "dslr/resized_undistorted_images"
        masks_dir = temporary / "provenance/synthetic_masks"
        depth_dir = temporary / "provenance/geometry_depth_visibility_evidence"
        frame_manifest: list[dict[str, Any]] = []
        for frame in inputs.frames:
            image_output = images_dir / f"{frame.frame_id}.png"
            mask_output = masks_dir / f"{frame.frame_id}.png"
            depth_output = depth_dir / f"{frame.frame_id}.npz"
            image_record = materialize_file(frame.completed_frame, image_output, storage_mode)
            mask_record = materialize_file(frame.synthetic_mask, mask_output, storage_mode)
            depth_record = materialize_file(frame.geometry_depth, depth_output, storage_mode)
            image_record["destination"] = image_output.relative_to(temporary).as_posix()
            mask_record["destination"] = mask_output.relative_to(temporary).as_posix()
            depth_record["destination"] = depth_output.relative_to(temporary).as_posix()
            materializations.extend((image_record, mask_record, depth_record))
            frame_manifest.append(
                {
                    "sequence_index": frame.sequence_index,
                    "frame_id": frame.frame_id,
                    "clean_rgb": {
                        "path": image_output.relative_to(temporary).as_posix(),
                        "sha256": frame.completed_frame_sha256,
                        "source_path": str(frame.completed_frame),
                        "source_sha256": frame.completed_frame_sha256,
                        "width": frame.width,
                        "height": frame.height,
                    },
                    "synthetic_provenance_mask": {
                        "path": mask_output.relative_to(temporary).as_posix(),
                        "sha256": frame.synthetic_mask_sha256,
                        "source_path": str(frame.synthetic_mask),
                        "synthetic_pixels": frame.synthetic_pixels,
                    },
                    "structural_geometry_depth": {
                        "path": depth_output.relative_to(temporary).as_posix(),
                        "sha256": frame.geometry_depth_sha256,
                        "source_path": str(frame.geometry_depth),
                        "role": "visibility_and_structural_provenance_evidence_only",
                        "allowed_as_new_da3_depth": False,
                        "allowed_as_final_pgsr_depth": False,
                    },
                    "outside_synthetic_mask_rgb_exact": True,
                    "source_prefill_rgb": {
                        "path": str(frame.source_frame),
                        "sha256": frame.source_frame_sha256,
                    },
                }
            )

        camera_output = temporary / "camera_info.json"
        transforms_output = temporary / "dslr/nerfstudio/transforms_undistorted.json"
        write_json(camera_output, camera_subset)
        write_json(transforms_output, transforms)
        source_texture_record = {
            "path": str(inputs.texture_report_path),
            "sha256": inputs.texture_report_sha256,
            "status": inputs.texture_report["status"],
            "promotion_approved": inputs.texture_report["promotion_approved"],
            "eligible_as_round04_clean_plate": inputs.texture_report[
                "eligible_as_round04_clean_plate"
            ],
        }
        acceptance_gate = {
            "status": inputs.texture_report["status"],
            "promotion_approved": inputs.texture_report["promotion_approved"],
            "eligible_as_round04_clean_plate": inputs.texture_report[
                "eligible_as_round04_clean_plate"
            ],
            "outside_synthetic_mask_rgb_exact": True,
        }
        limited_metadata = _limited_acceptance_metadata(inputs.texture_report)
        if limited_metadata:
            source_texture_record.update(copy.deepcopy(limited_metadata))
            acceptance_gate.update(copy.deepcopy(limited_metadata))

        manifest = {
            "schema_version": 1,
            "kind": "video2world.clean_scene_reconstruction_input",
            "scene_id": scene_id,
            "status": "ready_for_fresh_depth_and_normal_estimation",
            "source_planar_texture_report": source_texture_record,
            "source_planar_geometry_report": {
                "path": str(inputs.geometry_report_path),
                "sha256": inputs.geometry_report_sha256,
            },
            "frames": frame_manifest,
            "camera": {
                "subset_camera_info": {
                    "path": "camera_info.json",
                    "sha256": sha256_file(camera_output),
                    "source_sha256": inputs.camera_info_sha256,
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
            "provenance_contract": {
                "completed_rgb_outside_synthetic_mask": "source prefill RGB preserved exactly",
                "synthetic_pixels_are_measured": False,
                "geometry_depth_role": "visibility_and_structural_provenance_evidence_only",
                "legacy_da3_depth_role": "upstream_visibility_evidence_only_not_packaged",
                "fresh_depth_required": True,
                "fresh_normals_required": True,
            },
            "stage_status": {
                "accepted_clean_rgb": "ready",
                "camera_subset": "ready",
                "holi_pgsr_camera_transforms": "ready",
                "fresh_depth_estimation": "pending",
                "fresh_normal_estimation": "pending",
                "pgsr_optimization": "pending",
                "tsdf_fusion": "pending",
            },
            "forbidden_claims": [
                "geometry depth evidence is new DA3 depth",
                "legacy DA3 visibility depth is valid after clean RGB synthesis",
                "PGSR or TSDF has already run",
            ],
        }
        manifest_output = temporary / "manifest.json"
        write_json(manifest_output, manifest)
        receipt = {
            "schema_version": 1,
            "kind": "video2world.clean_scene_reconstruction_input_receipt",
            "created_at": datetime.now(UTC).isoformat(),
            "status": "materialized",
            "scene_id": scene_id,
            "output": str(output),
            "storage_mode": storage_mode,
            "frame_count": len(inputs.frames),
            "materialized_source_file_count": len(materializations),
            "materializations": materializations,
            "acceptance_gate": acceptance_gate,
            "camera_validation": camera_validation,
            "manifest": {
                "path": "manifest.json",
                "sha256": sha256_file(manifest_output),
            },
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
    parser.add_argument("--accepted-planar-texture-report", type=Path, required=True)
    parser.add_argument("--camera-info", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--storage-mode", choices=("copy", "hardlink"), default="copy")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest, receipt = materialize_package(
        texture_report_path=args.accepted_planar_texture_report,
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
                "next_stage": "fresh_depth_and_normal_estimation",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
