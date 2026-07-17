#!/usr/bin/env python3
"""Compose a provenance-partitioned clean plate from measured and rendered layers.

The compositor is fail-closed: object-peel measured pixels must be proven by a
calibrated RGB-D donor report; downstream PBR and structural-background renders
must carry their own receipt classes; generated or ProPainter RGB cannot claim
the measured class. A final-background round may explicitly carry no measured
donor evidence only after a complete previous round and with an accepted
structural render covering the entire cumulative removal mask. Round N>1 must
bind the Round N-1 output receipt and use its composite RGB byte hash as the
next source frame.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

INPUT_KIND = "video2world.layered_clean_plate_compositor_input"
REPORT_KIND = "video2world.layered_clean_plate_composite_report"
RECEIPT_KIND = "video2world.layered_clean_plate_composite_receipt"
MEASURED_RECEIPT_KIND = "video2world.multiview_prefill_receipt"
PBR_RECEIPT_KIND = "video2world.pbr_layer_render_receipt"
BACKGROUND_RECEIPT_KIND = "video2world.structural_background_render_receipt"
ROUND_KINDS = {"object_peel", "final_background"}
CALIBRATED_MEASURED_ROLE = "calibrated_rgbd_donor_prefill"
NO_MEASURED_ROLE = "no_measured_donor_evidence"
LIMITED_OVERRIDE_WHITELIST = {
    "wall_boundary_color_continuity",
    "wall_boundary_low_frequency_gradient_continuity",
    "visual_quality",
}

LABELS = {
    "outside_source": 0,
    "measured_donor": 1,
    "downstream_object_render": 2,
    "structural_background": 3,
    "unresolved": 4,
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def require_unique_string_list(value: Any, label: str) -> list[str]:
    items = require_list(value, label)
    if not items or any(not isinstance(item, str) or not item for item in items):
        raise ValueError(f"{label} must be a non-empty array of non-empty strings")
    if len(set(items)) != len(items):
        raise ValueError(f"{label} must not contain duplicates")
    return items


def resolve_path(value: str, *, relative_to: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def asset_path(asset: Any, *, relative_to: Path, label: str) -> Path:
    value = require_dict(asset, label)
    path_value = value.get("path")
    expected_sha = value.get("sha256")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{label}.path must be a non-empty string")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValueError(f"{label}.sha256 must be a SHA-256 string")
    path = resolve_path(path_value, relative_to=relative_to)
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != expected_sha:
        raise ValueError(f"{label} SHA-256 mismatch")
    return path


def binary_mask(path: Path, *, label: str) -> np.ndarray:
    with Image.open(path) as image:
        value = np.asarray(image.convert("L"), dtype=np.uint8)
    if not np.all((value == 0) | (value == 255)):
        raise ValueError(f"{label} must contain only 0 and 255")
    return value == 255


def rgb_image(path: Path, *, label: str) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode not in {"RGB", "RGBA"}:
            raise ValueError(f"{label} must be RGB-compatible")
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def rgba_image(path: Path, *, label: str) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode != "RGBA":
            raise ValueError(f"{label} must be an RGBA image")
        return np.asarray(image, dtype=np.uint8)


def depth_array(path: Path, *, label: str) -> np.ndarray:
    if path.suffix != ".npy":
        raise ValueError(f"{label} must be a .npy array")
    value = np.load(path, allow_pickle=False)
    if value.ndim != 2 or not np.issubdtype(value.dtype, np.number):
        raise ValueError(f"{label} must be a numeric HxW array")
    return value.astype(np.float32, copy=False)


def report_frame_map(report: dict[str, Any], *, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw_record in require_list(report.get("frame_records"), f"{label}.frame_records"):
        record = require_dict(raw_record, f"{label} frame record")
        frame_id = record.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id or frame_id in result:
            raise ValueError(f"{label} has an invalid or duplicate frame id")
        result[frame_id] = record
    return result


def report_asset_path(
    record: dict[str, Any],
    path_key: str,
    sha_key: str,
    *,
    report_path: Path,
    label: str,
) -> Path:
    path_value = record.get(path_key)
    expected_sha = record.get(sha_key)
    if not isinstance(path_value, str) or not isinstance(expected_sha, str):
        raise ValueError(f"{label} is absent from the measured donor report")
    path = resolve_path(path_value, relative_to=report_path.parent)
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise ValueError(f"{label} report asset SHA-256 mismatch")
    return path


def validate_measured_contract(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    round_kind: str,
    remaining_object_ids: list[str],
    previous_binding: dict[str, Any] | None,
) -> dict[str, Any]:
    contract = require_dict(manifest.get("measured_donor_contract"), "measured donor contract")
    role = contract.get("role")
    if role == NO_MEASURED_ROLE:
        if round_kind != "final_background":
            raise ValueError(
                "no_measured_donor_evidence is allowed only for final_background"
            )
        if remaining_object_ids:
            raise ValueError(
                "final_background without measured donor evidence requires no remaining objects"
            )
        if previous_binding is None:
            raise ValueError(
                "final_background without measured donor evidence requires a complete "
                "previous round"
            )
        if contract.get("measured_pixels") != 0:
            raise ValueError("no-measured contract must declare measured_pixels=0")
        if contract.get("claims_original_observed_rgb_donor") is not False:
            raise ValueError("no-measured contract must reject original donor claims")
        if contract.get("generated_pixels") != 0 or contract.get("propainter_pixels") != 0:
            raise ValueError("no-measured contract must not relabel generated pixels as measured")
        if contract.get("report") is not None or contract.get("receipt") is not None:
            raise ValueError(
                "no-measured contract cannot attach a measured donor report or receipt"
            )
        return {
            "role": NO_MEASURED_ROLE,
            "report_path": None,
            "report": None,
            "frames": {},
            "claims_original_observed_rgb_donor": False,
        }
    if role != CALIBRATED_MEASURED_ROLE:
        raise ValueError(
            "measured donor role must be calibrated_rgbd_donor_prefill or "
            "no_measured_donor_evidence"
        )
    if contract.get("generated_pixels") != 0 or contract.get("propainter_pixels") != 0:
        raise ValueError("generated or ProPainter pixels cannot claim measured provenance")
    report_path = asset_path(
        contract.get("report"),
        relative_to=manifest_path.parent,
        label="measured donor report",
    )
    receipt_path = asset_path(
        contract.get("receipt"),
        relative_to=manifest_path.parent,
        label="measured donor receipt",
    )
    report = require_dict(read_json(report_path), "measured donor report")
    receipt = require_dict(read_json(receipt_path), "measured donor receipt")
    if receipt.get("kind") != MEASURED_RECEIPT_KIND:
        raise ValueError("measured donor receipt has the wrong kind")
    if receipt.get("report_sha256") != sha256_file(report_path):
        raise ValueError("measured donor receipt does not bind the donor report")
    if receipt.get("frame_artifacts_are_hashed_in_report") is not True:
        raise ValueError("measured donor receipt does not bind frame artifacts")
    if report.get("status") != "technical_passed":
        raise ValueError("measured donor report did not technically pass")
    gates = require_dict(report.get("gates"), "measured donor gates")
    if gates.get("outside_removal_mask_rgb_exact") is not True:
        raise ValueError("measured donor report did not preserve outside-mask RGB")
    if gates.get("all_residual_masks_subset_of_removal_masks") is not True:
        raise ValueError("measured donor residuals escape the removal masks")
    cumulative = require_dict(
        report.get("cumulative_removal_contract"), "measured cumulative contract"
    )
    if cumulative.get("enforced") is not True:
        raise ValueError("measured donor report lacks the cumulative removal contract")
    if cumulative.get("donors_are_original_observed_rgb") is not True:
        raise ValueError("measured donor report does not prove original observed RGB donors")
    if cumulative.get("unresolved_residual_as_donor") is not False:
        raise ValueError("measured donor report allows unresolved residual as donor evidence")
    provenance = require_dict(report.get("pixel_provenance"), "measured pixel provenance")
    if provenance.get("generated_pixels") != 0:
        raise ValueError("measured donor report contains generated pixels")
    if provenance.get("propainter_pixels") != 0:
        raise ValueError("measured donor report contains or omits ProPainter provenance")
    if provenance.get("unresolved_pixels_are_not_valid_donor_or_geometry_evidence") is not True:
        raise ValueError("measured donor report allows unresolved pixels as evidence")
    return {
        "role": CALIBRATED_MEASURED_ROLE,
        "report_path": report_path,
        "report": report,
        "frames": report_frame_map(report, label="measured donor report"),
        "claims_original_observed_rgb_donor": True,
    }


def validate_previous_round(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
) -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]]]:
    round_index = manifest["round_index"]
    previous_value = manifest.get("previous_round")
    source_contract = require_dict(manifest.get("source_contract"), "source contract")
    if source_contract.get("untracked_generated_rgb") is not False:
        raise ValueError("source contract must explicitly reject untracked generated RGB")
    if source_contract.get("propainter_rgb") is not False:
        raise ValueError("source contract must explicitly reject ProPainter RGB")
    if round_index == 1:
        if previous_value is not None:
            raise ValueError("round 1 cannot bind a previous round")
        if source_contract.get("role") != "original_observed_rgb":
            raise ValueError("round 1 source must be original observed RGB")
        return None, {}

    if source_contract.get("role") != "previous_layered_composite":
        raise ValueError("round 2+ source must be the previous layered composite")
    previous = require_dict(previous_value, "previous round")
    if previous.get("round_index") != round_index - 1:
        raise ValueError("previous round index is not immediately preceding")
    report_path = asset_path(
        previous.get("report"), relative_to=manifest_path.parent, label="previous report"
    )
    receipt_path = asset_path(
        previous.get("receipt"), relative_to=manifest_path.parent, label="previous receipt"
    )
    report = require_dict(read_json(report_path), "previous report")
    receipt = require_dict(read_json(receipt_path), "previous receipt")
    if report.get("kind") != REPORT_KIND or report.get("round_index") != round_index - 1:
        raise ValueError("previous report kind or round index is invalid")
    if receipt.get("kind") != RECEIPT_KIND or receipt.get("round_index") != round_index - 1:
        raise ValueError("previous receipt kind or round index is invalid")
    previous_round_kind = report.get("round_kind")
    if previous_round_kind not in ROUND_KINDS:
        raise ValueError("previous report has no valid round_kind")
    if receipt.get("round_kind") != previous_round_kind:
        raise ValueError("previous receipt does not bind round_kind")
    if previous_round_kind == "final_background":
        raise ValueError("a final_background round cannot feed another round")
    if receipt.get("report_sha256") != sha256_file(report_path):
        raise ValueError("previous receipt does not bind the previous report")
    if receipt.get("output_frame_set_sha256") != report.get("output_frame_set_sha256"):
        raise ValueError("previous receipt does not bind the previous output frame set")
    if receipt.get("provenance_partition_exact") is not True:
        raise ValueError("previous receipt did not pass the provenance partition gate")
    if report.get("status") != "technical_passed_complete_partition":
        raise ValueError("previous round is incomplete and cannot feed the next peel round")
    aggregate_counts = require_dict(report.get("aggregate_counts"), "previous aggregate counts")
    if aggregate_counts.get("unresolved_pixels") != 0:
        raise ValueError("previous round still contains unresolved pixels")
    previous_gates = require_dict(report.get("gates"), "previous report gates")
    if previous_gates.get("outside_removal_mask_rgb_exact") is not True:
        raise ValueError("previous report did not preserve outside-mask RGB")
    if previous_gates.get("removal_provenance_partition_exact") is not True:
        raise ValueError("previous report provenance partition did not pass")
    previous_objects = report.get("removed_object_ids")
    current_objects = manifest.get("removed_object_ids")
    if not isinstance(previous_objects, list) or not isinstance(current_objects, list):
        raise ValueError("removed_object_ids must be arrays")
    if current_objects[:-1] != previous_objects:
        raise ValueError("current removed objects do not extend the previous round")
    return {
        "round_index": round_index - 1,
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "output_frame_set_sha256": receipt.get("output_frame_set_sha256"),
    }, report_frame_map(report, label="previous report")


def validate_limited_texture_acceptance(
    report: dict[str, Any],
) -> dict[str, Any]:
    if report.get("status") != "accepted_for_round04_clean_plate":
        raise ValueError("limited texture report is not accepted for round04")
    if report.get("accepted_with_limitations") is not True:
        raise ValueError("limited texture report must declare accepted_with_limitations")
    if report.get("promotion_approved") is not False:
        raise ValueError("limited texture report cannot claim unconditional promotion")
    if report.get("demo_use_approved") is not True:
        raise ValueError("limited texture report is not approved for demo use")
    if report.get("eligible_as_round04_clean_plate") is not False:
        raise ValueError("limited texture report cannot claim general round04 eligibility")
    if report.get("eligible_as_current_demo_round04_clean_plate") is not True:
        raise ValueError("limited texture report is not eligible for the current demo")
    if report.get("acceptance_scope") != "current_demo_only":
        raise ValueError("limited texture report scope must be current_demo_only")
    acceptance = require_dict(report.get("acceptance"), "limited texture acceptance")
    expected_acceptance_scalars = {
        "kind": "video2world.planar_texture_acceptance",
        "accepted_with_limitations": True,
        "promotion_approved": False,
        "demo_use_approved": True,
        "eligible_as_round04_clean_plate": False,
        "eligible_as_current_demo_round04_clean_plate": True,
        "acceptance_scope": "current_demo_only",
    }
    for key, expected in expected_acceptance_scalars.items():
        if acceptance.get(key) != expected:
            raise ValueError(f"limited texture acceptance has invalid {key}")
    overridden_gates = require_dict(
        acceptance.get("overridden_gates"),
        "limited texture overridden_gates",
    )
    if set(overridden_gates) != LIMITED_OVERRIDE_WHITELIST:
        raise ValueError(
            "limited texture overridden_gates must exactly match the approved whitelist"
        )
    for gate_name in (
        "wall_boundary_color_continuity",
        "wall_boundary_low_frequency_gradient_continuity",
    ):
        metrics = require_dict(
            overridden_gates.get(gate_name),
            f"limited texture override {gate_name}",
        )
        if metrics.get("passed") is not False:
            raise ValueError(f"limited texture override {gate_name} must retain passed=false")
        for metric_name in ("threshold_p95_abs_rgb_delta", "observed_maximum"):
            value = metrics.get(metric_name)
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise ValueError(
                    f"limited texture override {gate_name}.{metric_name} must be numeric"
                )
    if overridden_gates.get("visual_quality") != "pending_human_or_vlm_review":
        raise ValueError("limited texture visual override must retain the original pending state")
    override_gate_keys = require_unique_string_list(
        acceptance.get("override_gate_keys"),
        "limited texture override_gate_keys",
    )
    if set(override_gate_keys) != LIMITED_OVERRIDE_WHITELIST:
        raise ValueError("limited texture override_gate_keys do not match overridden_gates")
    if report.get("overridden_gates") != overridden_gates:
        raise ValueError("limited texture top-level overridden_gates changed after review")
    limitations = require_unique_string_list(
        acceptance.get("review_limitations"),
        "limited texture review limitations",
    )
    gates = require_dict(report.get("gates"), "limited accepted texture gates")
    if gates.get("visual_quality") != "accepted_with_limitations":
        raise ValueError("limited accepted texture report lost the review decision")
    for gate_name in (
        "wall_boundary_color_continuity",
        "wall_boundary_low_frequency_gradient_continuity",
    ):
        if gates.get(gate_name) != overridden_gates[gate_name]:
            raise ValueError(f"limited accepted texture gate {gate_name} changed after review")
    required_true_gates = {
        "all_synthetic_pixels_assigned_from_shared_atlas",
        "measured_atlas_texels_rgb_exact",
        "observed_synthetic_anchor_and_interpolated_partition_rendered_synthetic_pixels",
        "outside_synthetic_mask_rgb_exact",
        "protected_neighbors_outside_synthetic_mask_rgb_exact",
        "protected_sam_neighbor_pixels_rgb_exact",
        "rejected_texture_inputs_not_promotable",
        "same_atlas_texel_has_identical_rgb_across_views",
        "synthetic_anchor_atlas_texels_rgb_exact",
        "wall_object_boundary_colors_excluded_from_synthetic_fit",
    }
    if any(gates.get(key) is not True for key in required_true_gates):
        raise ValueError("limited texture report has a failed non-overridden technical gate")
    if gates.get("synthetic_anchor_claims_measured_donor") is not False:
        raise ValueError("limited texture report claims synthetic anchor as measured donor")
    if (
        gates.get("new_depth_normal_estimation_before_pgsr")
        != "required_after_visual_acceptance"
    ):
        raise ValueError("limited texture report lost the fresh reconstruction requirement")
    return {
        "acceptance_status": "accepted_with_limitations",
        "accepted_with_limitations": True,
        "demo_use_approved": True,
        "eligible_as_round04_clean_plate": False,
        "eligible_as_current_demo_round04_clean_plate": True,
        "acceptance_scope": "current_demo_only",
        "limitations": limitations,
        "override_gate_keys": override_gate_keys,
        "overridden_gates": overridden_gates,
        "failed_metrics": overridden_gates,
        "promotion_approved": False,
    }


def validate_render_receipt(
    layer: dict[str, Any],
    *,
    manifest_path: Path,
    expected_kind: str,
    expected_role: str,
    layer_id: str | None,
    require_accepted: bool = False,
) -> dict[str, Any]:
    receipt_path = asset_path(
        layer.get("receipt"), relative_to=manifest_path.parent, label=f"{expected_role} receipt"
    )
    receipt = require_dict(read_json(receipt_path), f"{expected_role} receipt")
    if receipt.get("kind") != expected_kind or receipt.get("role") != expected_role:
        raise ValueError(f"{expected_role} receipt has the wrong kind or role")
    if receipt.get("provenance_class") != expected_role:
        raise ValueError(f"{expected_role} cannot claim another provenance class")
    if receipt.get("claims_measured_donor") is not False:
        raise ValueError(f"{expected_role} receipt must reject measured-donor claims")
    if layer_id is not None and receipt.get("layer_id") != layer_id:
        raise ValueError("PBR render receipt layer_id mismatch")
    rgba_path = asset_path(
        layer.get("rgba"), relative_to=manifest_path.parent, label=f"{expected_role} RGBA"
    )
    depth_path = asset_path(
        layer.get("depth"), relative_to=manifest_path.parent, label=f"{expected_role} depth"
    )
    if receipt.get("rgba_sha256") != sha256_file(rgba_path):
        raise ValueError(f"{expected_role} receipt does not bind RGBA")
    if receipt.get("depth_sha256") != sha256_file(depth_path):
        raise ValueError(f"{expected_role} receipt does not bind depth")
    if expected_role == "structural_background" and receipt.get("texture_provenance") not in {
        "observed_atlas",
        "synthetic_atlas",
        "hybrid_atlas",
    }:
        raise ValueError("structural background texture provenance is missing")
    accepted_texture_report_path: Path | None = None
    accepted_texture_report: dict[str, Any] | None = None
    acceptance: dict[str, Any] | None = None
    if require_accepted:
        acceptance_status = receipt.get("acceptance_status")
        if acceptance_status not in {"accepted", "accepted_with_limitations"}:
            raise ValueError("final structural background receipt is not accepted")
        accepted_texture_report_path = asset_path(
            receipt.get("accepted_texture_report"),
            relative_to=receipt_path.parent,
            label="accepted structural texture report",
        )
        accepted_texture_report = require_dict(
            read_json(accepted_texture_report_path),
            "accepted structural texture report",
        )
        accepted_texture_report_sha256 = receipt.get("accepted_texture_report_sha256")
        if (
            not isinstance(accepted_texture_report_sha256, str)
            or len(accepted_texture_report_sha256) != 64
            or sha256_file(accepted_texture_report_path)
            != accepted_texture_report_sha256
        ):
            raise ValueError("structural receipt does not bind the accepted texture report")
        if acceptance_status == "accepted":
            if accepted_texture_report.get("status") != "accepted_for_round04_clean_plate":
                raise ValueError("structural texture report is not accepted for round04")
            if accepted_texture_report.get("eligible_as_round04_clean_plate") is not True:
                raise ValueError("strict structural texture is not eligible for round04")
            acceptance_gates = require_dict(
                accepted_texture_report.get("acceptance_gates"),
                "structural texture acceptance gates",
            )
            if acceptance_gates.get("technical_quality_passed") is not True:
                raise ValueError("structural texture technical quality did not pass")
            if receipt.get("promotion_approved") is not True:
                raise ValueError("strict structural acceptance requires promotion_approved=true")
            if acceptance_gates.get("visual_quality_passed") is not True:
                raise ValueError("structural texture visual quality did not pass")
            if accepted_texture_report.get("accepted_with_limitations") is True:
                raise ValueError("strict structural acceptance cannot carry limitations")
            acceptance = {
                "acceptance_status": "accepted",
                "accepted_with_limitations": False,
                "demo_use_approved": True,
                "eligible_as_round04_clean_plate": True,
                "eligible_as_current_demo_round04_clean_plate": True,
                "acceptance_scope": "full_round",
                "limitations": [],
                "override_gate_keys": [],
                "overridden_gates": {},
                "failed_metrics": {},
                "promotion_approved": True,
            }
        else:
            acceptance = validate_limited_texture_acceptance(accepted_texture_report)
            for key, expected in acceptance.items():
                if key == "acceptance_status":
                    continue
                if receipt.get(key) != expected:
                    raise ValueError(f"limited structural receipt does not propagate {key}")
    return {
        "receipt_path": receipt_path,
        "receipt": receipt,
        "rgba_path": rgba_path,
        "depth_path": depth_path,
        "accepted_texture_report_path": accepted_texture_report_path,
        "accepted_texture_report": accepted_texture_report,
        "acceptance": acceptance,
    }


def save_mask(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value.astype(np.uint8) * 255).save(path)


def compose_frame(
    *,
    record: dict[str, Any],
    manifest_path: Path,
    measured_contract: dict[str, Any],
    measured_record: dict[str, Any] | None,
    previous_record: dict[str, Any] | None,
    round_kind: str,
    remaining_object_ids: list[str],
    alpha_threshold: float,
    depth_tie_epsilon: float,
    staging: Path,
) -> dict[str, Any]:
    frame_id = record["frame_id"]
    sequence_index = record["sequence_index"]
    source_path = asset_path(
        record.get("source_rgb"), relative_to=manifest_path.parent, label=f"{frame_id} source RGB"
    )
    removal_path = asset_path(
        record.get("cumulative_removal_mask"),
        relative_to=manifest_path.parent,
        label=f"{frame_id} cumulative removal mask",
    )
    measured_mask_path = asset_path(
        record.get("donor_measured_mask"),
        relative_to=manifest_path.parent,
        label=f"{frame_id} donor measured mask",
    )
    measured_depth_path = asset_path(
        record.get("donor_measured_depth"),
        relative_to=manifest_path.parent,
        label=f"{frame_id} donor measured depth",
    )

    source = rgb_image(source_path, label=f"{frame_id} source RGB")
    removal = binary_mask(removal_path, label=f"{frame_id} removal mask")
    measured = binary_mask(measured_mask_path, label=f"{frame_id} measured mask")
    measured_depth = depth_array(measured_depth_path, label=f"{frame_id} measured depth")
    shape = source.shape[:2]
    if removal.shape != shape or measured.shape != shape:
        raise ValueError(f"{frame_id} RGB/mask shape mismatch")
    if measured_depth.shape != shape:
        raise ValueError(f"{frame_id} measured depth shape mismatch")
    measured_role = measured_contract["role"]
    donor_path: Path | None = None
    donor: np.ndarray | None = None
    if measured_role == CALIBRATED_MEASURED_ROLE:
        measured_report_path = measured_contract["report_path"]
        if not isinstance(measured_report_path, Path) or measured_record is None:
            raise ValueError("calibrated measured donor frame evidence is missing")
        donor_path = asset_path(
            record.get("donor_prefill_rgb"),
            relative_to=manifest_path.parent,
            label=f"{frame_id} donor prefill RGB",
        )
        donor = rgb_image(donor_path, label=f"{frame_id} donor prefill RGB")
        if donor.shape[:2] != shape:
            raise ValueError(f"{frame_id} donor RGB shape mismatch")
        report_prefill = report_asset_path(
            measured_record,
            "prefill_frame",
            "prefill_frame_sha256",
            report_path=measured_report_path,
            label=f"{frame_id} measured prefill",
        )
        report_removal = report_asset_path(
            measured_record,
            "removal_mask",
            "removal_mask_sha256",
            report_path=measured_report_path,
            label=f"{frame_id} measured removal mask",
        )
        report_residual = report_asset_path(
            measured_record,
            "residual_mask",
            "residual_mask_sha256",
            report_path=measured_report_path,
            label=f"{frame_id} measured residual mask",
        )
        report_depth = report_asset_path(
            measured_record,
            "measured_depth",
            "measured_depth_sha256",
            report_path=measured_report_path,
            label=f"{frame_id} fused measured depth",
        )
        if measured_record.get("measured_depth_role") != "fused_multiview_measured_depth":
            raise ValueError("measured depth lacks fused-multiview provenance")
        if sha256_file(donor_path) != sha256_file(report_prefill):
            raise ValueError(f"{frame_id} donor prefill does not match measured report")
        if sha256_file(removal_path) != sha256_file(report_removal):
            raise ValueError(f"{frame_id} removal mask does not match measured report")
        if sha256_file(measured_depth_path) != sha256_file(report_depth):
            raise ValueError(f"{frame_id} measured depth does not match measured report")
        residual_from_report = binary_mask(
            report_residual, label=f"{frame_id} measured report residual"
        )
        expected_measured = removal & ~residual_from_report
        if not np.array_equal(measured, expected_measured):
            raise ValueError(f"{frame_id} measured mask is not removal minus report residual")
        if np.any(measured & (~np.isfinite(measured_depth) | (measured_depth <= 0))):
            raise ValueError(f"{frame_id} measured pixels lack valid fused depth")
    else:
        if record.get("donor_prefill_rgb") is not None:
            raise ValueError("no-measured frame cannot attach donor prefill RGB")
        if np.any(measured):
            raise ValueError("no-measured frame requires an empty measured mask")
        if np.any(np.isfinite(measured_depth)):
            raise ValueError("no-measured frame requires measured depth to be all NaN")

    if previous_record is not None:
        previous_output = require_dict(previous_record.get("outputs"), "previous outputs")
        previous_rgb = require_dict(previous_output.get("composite_rgb"), "previous RGB output")
        if previous_rgb.get("sha256") != sha256_file(source_path):
            raise ValueError(f"{frame_id} source RGB is not the previous round output")

    output = source.copy()
    composite_depth = np.full(shape, np.nan, dtype=np.float32)
    provenance = np.full(shape, LABELS["outside_source"], dtype=np.uint8)
    downstream_labels = np.zeros(shape, dtype=np.uint16)
    if donor is not None:
        output[measured] = donor[measured]
    composite_depth[measured] = measured_depth[measured]
    provenance[measured] = LABELS["measured_donor"]
    residual = removal & ~measured
    best_depth = np.full(shape, np.inf, dtype=np.float32)
    best_rgb = np.zeros((*shape, 3), dtype=np.uint8)
    best_class = np.full(shape, LABELS["unresolved"], dtype=np.uint8)
    best_layer = np.zeros(shape, dtype=np.uint16)
    layer_records: list[dict[str, Any]] = []

    raw_layers = require_list(record.get("downstream_layers"), f"{frame_id} downstream layers")
    orders = [require_dict(item, "downstream layer").get("depth_order") for item in raw_layers]
    if orders != list(range(len(raw_layers))):
        raise ValueError(f"{frame_id} downstream layer depth_order must be contiguous")
    layer_ids = [require_dict(item, "downstream layer").get("layer_id") for item in raw_layers]
    if any(not isinstance(layer_id, str) or not layer_id for layer_id in layer_ids):
        raise ValueError(f"{frame_id} downstream layers require non-empty layer_id values")
    if len(set(layer_ids)) != len(layer_ids):
        raise ValueError(f"{frame_id} downstream layer_id values must be unique")
    if layer_ids != remaining_object_ids:
        raise ValueError(
            f"{frame_id} downstream layers do not match remaining_object_ids: "
            f"{layer_ids} != {remaining_object_ids}"
        )
    for layer_index, raw_layer in enumerate(raw_layers, start=1):
        layer = require_dict(raw_layer, f"{frame_id} downstream layer")
        layer_id = layer.get("layer_id")
        if not isinstance(layer_id, str) or not layer_id:
            raise ValueError(f"{frame_id} downstream layer has no layer_id")
        if layer.get("role") != "downstream_object_render":
            raise ValueError("PBR layer cannot claim measured or structural provenance")
        validated = validate_render_receipt(
            layer,
            manifest_path=manifest_path,
            expected_kind=PBR_RECEIPT_KIND,
            expected_role="downstream_object_render",
            layer_id=layer_id,
        )
        rgba = rgba_image(validated["rgba_path"], label=f"{frame_id} {layer_id} RGBA")
        depth = depth_array(validated["depth_path"], label=f"{frame_id} {layer_id} depth")
        if rgba.shape[:2] != shape or depth.shape != shape:
            raise ValueError(f"{frame_id} {layer_id} render shape mismatch")
        alpha = rgba[:, :, 3].astype(np.float32) / 255.0
        candidate = residual & (alpha >= alpha_threshold) & np.isfinite(depth) & (depth > 0)
        replace = candidate & (depth < best_depth - depth_tie_epsilon)
        best_depth[replace] = depth[replace]
        best_rgb[replace] = rgba[:, :, :3][replace]
        best_class[replace] = LABELS["downstream_object_render"]
        best_layer[replace] = layer_index
        layer_records.append(
            {
                "layer_id": layer_id,
                "depth_order": layer["depth_order"],
                "candidate_pixels": int(candidate.sum()),
                "rgba_sha256": sha256_file(validated["rgba_path"]),
                "depth_sha256": sha256_file(validated["depth_path"]),
                "receipt_sha256": sha256_file(validated["receipt_path"]),
            }
        )

    background = require_dict(record.get("structural_background"), "structural background")
    if background.get("role") != "structural_background":
        raise ValueError("background cannot claim measured or downstream-object provenance")
    validated_background = validate_render_receipt(
        background,
        manifest_path=manifest_path,
        expected_kind=BACKGROUND_RECEIPT_KIND,
        expected_role="structural_background",
        layer_id=None,
        require_accepted=(
            round_kind == "final_background" and measured_role == NO_MEASURED_ROLE
        ),
    )
    background_rgba = rgba_image(
        validated_background["rgba_path"], label=f"{frame_id} background RGBA"
    )
    background_depth = depth_array(
        validated_background["depth_path"], label=f"{frame_id} background depth"
    )
    if background_rgba.shape[:2] != shape or background_depth.shape != shape:
        raise ValueError(f"{frame_id} structural background render shape mismatch")
    background_alpha = background_rgba[:, :, 3].astype(np.float32) / 255.0
    background_candidate = (
        residual
        & (background_alpha >= alpha_threshold)
        & np.isfinite(background_depth)
        & (background_depth > 0)
    )
    replace_background = background_candidate & (
        background_depth < best_depth - depth_tie_epsilon
    )
    best_depth[replace_background] = background_depth[replace_background]
    best_rgb[replace_background] = background_rgba[:, :, :3][replace_background]
    best_class[replace_background] = LABELS["structural_background"]
    best_layer[replace_background] = 0

    resolved_render = residual & np.isfinite(best_depth)
    output[resolved_render] = best_rgb[resolved_render]
    composite_depth[resolved_render] = best_depth[resolved_render]
    provenance[resolved_render] = best_class[resolved_render]
    downstream_labels[resolved_render] = best_layer[resolved_render]
    unresolved = residual & ~resolved_render
    provenance[unresolved] = LABELS["unresolved"]

    masks = {
        name: provenance == label_value for name, label_value in LABELS.items()
    }
    removal_partition = (
        masks["measured_donor"]
        | masks["downstream_object_render"]
        | masks["structural_background"]
        | masks["unresolved"]
    )
    pair_sum = sum(mask.astype(np.uint8) for mask in masks.values())
    partition_exact = bool(np.array_equal(removal_partition, removal) and np.all(pair_sum == 1))
    outside_exact = bool(np.array_equal(output[~removal], source[~removal]))
    render_only_in_residual = bool(
        not np.any((masks["downstream_object_render"] | masks["structural_background"]) & ~residual)
    )
    final_background_only = (
        round_kind == "final_background" and measured_role == NO_MEASURED_ROLE
    )
    if final_background_only and not np.array_equal(
        masks["structural_background"],
        removal,
    ):
        raise ValueError(
            f"{frame_id} final background must cover the entire cumulative removal mask"
        )
    if not partition_exact or not outside_exact or not render_only_in_residual:
        raise ValueError(f"{frame_id} compositor invariants failed")

    output_name = f"{sequence_index:04d}.png"
    output_rgb_path = staging / "frames" / output_name
    output_depth_path = staging / "depth" / f"{sequence_index:04d}.npy"
    labels_path = staging / "provenance_labels" / output_name
    layer_labels_path = staging / "downstream_layer_labels" / output_name
    output_rgb_path.parent.mkdir(parents=True, exist_ok=True)
    output_depth_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    layer_labels_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output).save(output_rgb_path)
    np.save(output_depth_path, composite_depth)
    Image.fromarray(provenance).save(labels_path)
    Image.fromarray(downstream_labels).save(layer_labels_path)
    mask_assets: dict[str, dict[str, Any]] = {}
    for name, mask in masks.items():
        path = staging / "provenance_masks" / name / output_name
        save_mask(path, mask)
        mask_assets[name] = {
            "path": f"provenance_masks/{name}/{output_name}",
            "sha256": sha256_file(path),
            "pixels": int(mask.sum()),
        }

    return {
        "sequence_index": sequence_index,
        "frame_id": frame_id,
        "inputs": {
            "source_rgb_sha256": sha256_file(source_path),
            "cumulative_removal_mask_sha256": sha256_file(removal_path),
            "measured_donor_contract_role": measured_role,
            "claims_original_observed_rgb_donor": measured_contract[
                "claims_original_observed_rgb_donor"
            ],
            "donor_prefill_rgb_sha256": (
                sha256_file(donor_path) if donor_path is not None else None
            ),
            "donor_measured_mask_sha256": sha256_file(measured_mask_path),
            "donor_measured_depth_sha256": sha256_file(measured_depth_path),
            "downstream_layers": layer_records,
            "structural_background": {
                "rgba_sha256": sha256_file(validated_background["rgba_path"]),
                "depth_sha256": sha256_file(validated_background["depth_path"]),
                "receipt_sha256": sha256_file(validated_background["receipt_path"]),
                "accepted_texture_report_sha256": (
                    sha256_file(validated_background["accepted_texture_report_path"])
                    if validated_background["accepted_texture_report_path"] is not None
                    else None
                ),
                "texture_provenance": validated_background["receipt"]["texture_provenance"],
                "acceptance": validated_background["acceptance"],
                "candidate_pixels": int(background_candidate.sum()),
            },
        },
        "outputs": {
            "composite_rgb": {
                "path": f"frames/{output_name}",
                "sha256": sha256_file(output_rgb_path),
            },
            "composite_depth": {
                "path": f"depth/{sequence_index:04d}.npy",
                "sha256": sha256_file(output_depth_path),
                "valid_pixels": int(np.isfinite(composite_depth).sum()),
                "validity_scope": "measured_or_render_resolved_removal_pixels_only",
            },
            "provenance_labels": {
                "path": f"provenance_labels/{output_name}",
                "sha256": sha256_file(labels_path),
            },
            "downstream_layer_labels": {
                "path": f"downstream_layer_labels/{output_name}",
                "sha256": sha256_file(layer_labels_path),
                "mapping": {
                    str(index): layer["layer_id"]
                    for index, layer in enumerate(layer_records, start=1)
                },
            },
            "provenance_masks": mask_assets,
        },
        "counts": {
            "removal_pixels": int(removal.sum()),
            "measured_donor_pixels": int(masks["measured_donor"].sum()),
            "downstream_object_render_pixels": int(
                masks["downstream_object_render"].sum()
            ),
            "structural_background_pixels": int(masks["structural_background"].sum()),
            "unresolved_pixels": int(masks["unresolved"].sum()),
            "outside_source_pixels": int(masks["outside_source"].sum()),
        },
        "gates": {
            "outside_removal_mask_rgb_exact": outside_exact,
            "removal_provenance_partition_exact": partition_exact,
            "render_composite_only_inside_measured_residual": render_only_in_residual,
            **(
                {"measured_pixels_match_donor_report": True}
                if measured_role == CALIBRATED_MEASURED_ROLE
                else {"measured_pixels_are_explicitly_absent": not np.any(measured)}
            ),
            "no_measured_donor_contract_satisfied": (
                measured_role != NO_MEASURED_ROLE
                or (
                    not np.any(measured)
                    and not np.any(np.isfinite(measured_depth))
                    and np.array_equal(masks["structural_background"], removal)
                )
            ),
            "unresolved_rgb_preserves_source_but_is_not_promotable": True,
        },
    }


def compose(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.input_manifest.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    manifest = require_dict(read_json(manifest_path), "compositor input manifest")
    if manifest.get("kind") != INPUT_KIND:
        raise ValueError("input manifest has the wrong kind")
    round_index = manifest.get("round_index")
    round_kind = manifest.get("round_kind")
    removed_object_ids = manifest.get("removed_object_ids")
    remaining_object_ids = manifest.get("remaining_object_ids")
    if not isinstance(round_index, int) or round_index < 1:
        raise ValueError("round_index must be a positive integer")
    if round_kind not in ROUND_KINDS:
        raise ValueError("round_kind must be object_peel or final_background")
    if (
        not isinstance(removed_object_ids, list)
        or len(removed_object_ids) != round_index
        or any(not isinstance(object_id, str) or not object_id for object_id in removed_object_ids)
    ):
        raise ValueError("one unique removed object is required per round")
    if len(set(removed_object_ids)) != len(removed_object_ids):
        raise ValueError("removed_object_ids must be unique")
    if (
        not isinstance(remaining_object_ids, list)
        or any(
            not isinstance(object_id, str) or not object_id
            for object_id in remaining_object_ids
        )
    ):
        raise ValueError("remaining_object_ids must be an array of non-empty strings")
    if len(set(remaining_object_ids)) != len(remaining_object_ids):
        raise ValueError("remaining_object_ids must be unique")
    if round_kind == "final_background" and remaining_object_ids:
        raise ValueError("final_background requires remaining_object_ids=[]")
    overlap = set(removed_object_ids) & set(remaining_object_ids)
    if overlap:
        raise ValueError(f"removed objects cannot remain as downstream layers: {sorted(overlap)}")
    if manifest.get("newly_removed_object_id") != removed_object_ids[-1]:
        raise ValueError("newly_removed_object_id must equal the last cumulative removed object")
    config = require_dict(manifest.get("config"), "compositor config")
    alpha_threshold = config.get("alpha_threshold")
    depth_tie_epsilon = config.get("depth_tie_epsilon")
    if (
        not isinstance(alpha_threshold, int | float)
        or isinstance(alpha_threshold, bool)
        or not 0 < alpha_threshold <= 1
    ):
        raise ValueError("alpha_threshold must be in (0, 1]")
    if (
        not isinstance(depth_tie_epsilon, int | float)
        or isinstance(depth_tie_epsilon, bool)
        or depth_tie_epsilon < 0
    ):
        raise ValueError("depth_tie_epsilon must be non-negative")

    previous_binding, previous_frames = validate_previous_round(
        manifest, manifest_path=manifest_path
    )
    if round_kind == "final_background" and previous_binding is None:
        raise ValueError("final_background requires a complete previous round")
    measured_contract = validate_measured_contract(
        manifest,
        manifest_path=manifest_path,
        round_kind=round_kind,
        remaining_object_ids=remaining_object_ids,
        previous_binding=previous_binding,
    )
    measured_report = measured_contract["report"]
    measured_frames = measured_contract["frames"]
    if measured_report is not None:
        measured_objects = measured_report["cumulative_removal_contract"].get(
            "removed_object_ids"
        )
        if measured_objects != removed_object_ids:
            raise ValueError("measured donor report covers different removed objects")
    raw_records = require_list(manifest.get("frame_records"), "frame_records")
    if not raw_records:
        raise ValueError("frame_records must be non-empty")
    records = [require_dict(item, "frame record") for item in raw_records]
    frame_ids = [record.get("frame_id") for record in records]
    if any(not isinstance(frame_id, str) or not frame_id for frame_id in frame_ids):
        raise ValueError("every frame record needs a frame_id")
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError("frame_ids must be unique")
    if [record.get("sequence_index") for record in records] != list(range(len(records))):
        raise ValueError("sequence_index must be contiguous and ordered")
    if measured_contract["role"] == CALIBRATED_MEASURED_ROLE and set(frame_ids) != set(
        measured_frames
    ):
        raise ValueError("input and measured donor report cover different frames")
    if round_index > 1 and set(frame_ids) != set(previous_frames):
        raise ValueError("input and previous report cover different frames")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_root.parent, prefix=f".{output_root.name}.staging-"
    ) as temporary_value:
        staging = Path(temporary_value)
        output_records = [
            compose_frame(
                record=record,
                manifest_path=manifest_path,
                measured_contract=measured_contract,
                measured_record=measured_frames.get(record["frame_id"]),
                previous_record=previous_frames.get(record["frame_id"]),
                round_kind=round_kind,
                remaining_object_ids=remaining_object_ids,
                alpha_threshold=float(alpha_threshold),
                depth_tie_epsilon=float(depth_tie_epsilon),
                staging=staging,
            )
            for record in records
        ]
        total_counts = {
            key: sum(record["counts"][key] for record in output_records)
            for key in output_records[0]["counts"]
        }
        unresolved_pixels = total_counts["unresolved_pixels"]
        measured_report_path = measured_contract["report_path"]
        measured_report_binding = (
            {
                "path": str(measured_report_path),
                "sha256": sha256_file(measured_report_path),
                "generated_pixels": 0,
                "propainter_pixels": 0,
            }
            if isinstance(measured_report_path, Path)
            else None
        )
        structural_acceptances = [
            record["inputs"]["structural_background"]["acceptance"]
            for record in output_records
            if record["inputs"]["structural_background"]["acceptance"] is not None
        ]
        structural_acceptance = (
            structural_acceptances[0] if structural_acceptances else None
        )
        if structural_acceptance is not None and any(
            item != structural_acceptance for item in structural_acceptances[1:]
        ):
            raise ValueError("structural background acceptance differs across frames")
        accepted_with_limitations = bool(
            structural_acceptance
            and structural_acceptance["accepted_with_limitations"] is True
        )
        output_frame_set = [
            {
                "frame_id": record["frame_id"],
                "composite_rgb_sha256": record["outputs"]["composite_rgb"]["sha256"],
                "composite_depth_sha256": record["outputs"]["composite_depth"]["sha256"],
                "provenance_labels_sha256": record["outputs"]["provenance_labels"]["sha256"],
            }
            for record in output_records
        ]
        report = {
            "schema_version": 1,
            "kind": REPORT_KIND,
            "status": (
                "technical_passed_with_unresolved"
                if unresolved_pixels
                else (
                    "technical_passed_complete_partition_with_limitations"
                    if accepted_with_limitations
                    else "technical_passed_complete_partition"
                )
            ),
            "created_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 - mil8 is 3.10
            "round_index": round_index,
            "round_kind": round_kind,
            "removed_object_ids": removed_object_ids,
            "newly_removed_object_id": removed_object_ids[-1],
            "remaining_object_ids": remaining_object_ids,
            "input_manifest": str(manifest_path),
            "input_manifest_sha256": sha256_file(manifest_path),
            "measured_donor_contract": {
                "role": measured_contract["role"],
                "claims_original_observed_rgb_donor": measured_contract[
                    "claims_original_observed_rgb_donor"
                ],
                "measured_pixels": total_counts["measured_donor_pixels"],
                "generated_pixels": 0,
                "propainter_pixels": 0,
            },
            "measured_donor_report": measured_report_binding,
            "previous_round_binding": previous_binding,
            "structural_background_acceptance": structural_acceptance,
            "acceptance_status": (
                structural_acceptance["acceptance_status"]
                if structural_acceptance is not None
                else None
            ),
            "accepted_with_limitations": accepted_with_limitations,
            "demo_use_approved": (
                structural_acceptance["demo_use_approved"]
                if structural_acceptance is not None
                else None
            ),
            "eligible_as_round04_clean_plate": (
                structural_acceptance["eligible_as_round04_clean_plate"]
                if structural_acceptance is not None
                else None
            ),
            "eligible_as_current_demo_round04_clean_plate": (
                structural_acceptance["eligible_as_current_demo_round04_clean_plate"]
                if structural_acceptance is not None
                else None
            ),
            "acceptance_scope": (
                structural_acceptance["acceptance_scope"]
                if structural_acceptance is not None
                else None
            ),
            "limitations": (
                structural_acceptance["limitations"]
                if structural_acceptance is not None
                else []
            ),
            "override_gate_keys": (
                structural_acceptance["override_gate_keys"]
                if structural_acceptance is not None
                else []
            ),
            "overridden_gates": (
                structural_acceptance["overridden_gates"]
                if structural_acceptance is not None
                else []
            ),
            "failed_metrics": (
                structural_acceptance["failed_metrics"]
                if structural_acceptance is not None
                else {}
            ),
            "config": {
                "alpha_threshold": float(alpha_threshold),
                "depth_tie_epsilon": float(depth_tie_epsilon),
                "z_buffer": "nearest_positive_finite_depth; manifest depth_order breaks ties",
            },
            "provenance_labels": LABELS,
            "frame_records": output_records,
            "aggregate_counts": total_counts,
            "output_frame_set_sha256": digest_json(output_frame_set),
            "gates": {
                "outside_removal_mask_rgb_exact": all(
                    record["gates"]["outside_removal_mask_rgb_exact"]
                    for record in output_records
                ),
                "removal_provenance_partition_exact": all(
                    record["gates"]["removal_provenance_partition_exact"]
                    for record in output_records
                ),
                "render_composite_only_inside_measured_residual": all(
                    record["gates"]["render_composite_only_inside_measured_residual"]
                    for record in output_records
                ),
                "generated_or_propainter_claimed_as_measured": False,
                "original_measured_donor_claimed_without_evidence": False,
                "no_measured_donor_evidence_has_zero_measured_pixels": (
                    measured_contract["role"] != NO_MEASURED_ROLE
                    or total_counts["measured_donor_pixels"] == 0
                ),
                "final_background_entire_removal_is_structural": (
                    round_kind != "final_background"
                    or measured_contract["role"] != NO_MEASURED_ROLE
                    or total_counts["structural_background_pixels"]
                    == total_counts["removal_pixels"]
                ),
                "previous_round_output_hash_bound": round_index == 1
                or previous_binding is not None,
                "limited_acceptance_scope_and_failures_propagated": (
                    not accepted_with_limitations
                    or (
                        structural_acceptance is not None
                        and structural_acceptance["acceptance_scope"]
                        == "current_demo_only"
                        and bool(structural_acceptance["limitations"])
                        and bool(structural_acceptance["failed_metrics"])
                    )
                ),
            },
            "promotion_approved": False,
            "promotion_blockers": [
                "visual and cross-view consistency review is required",
                *(
                    ["unresolved pixels remain and preserve source RGB only as a diagnostic"]
                    if unresolved_pixels
                    else []
                ),
                *(
                    [
                        "structural background is accepted only for the current demo; "
                        "listed failed metrics remain overridden"
                    ]
                    if accepted_with_limitations
                    else []
                ),
                "depth and normal estimation must be rerun after final RGB acceptance",
            ],
            "output_receipt": "layered_composite_receipt.json",
        }
        report_path = staging / "layered_composite_report.json"
        write_json(report_path, report)
        receipt = {
            "schema_version": 1,
            "kind": RECEIPT_KIND,
            "round_index": round_index,
            "round_kind": round_kind,
            "measured_donor_contract_role": measured_contract["role"],
            "claims_original_observed_rgb_donor": measured_contract[
                "claims_original_observed_rgb_donor"
            ],
            "report": report_path.name,
            "report_sha256": sha256_file(report_path),
            "input_manifest_sha256": sha256_file(manifest_path),
            "output_frame_set_sha256": report["output_frame_set_sha256"],
            "previous_round_output_report_sha256": (
                previous_binding["report_sha256"] if previous_binding else None
            ),
            "previous_round_output_receipt_sha256": (
                previous_binding["receipt_sha256"] if previous_binding else None
            ),
            "provenance_partition_exact": report["gates"][
                "removal_provenance_partition_exact"
            ],
            "accepted_with_limitations": report["accepted_with_limitations"],
            "demo_use_approved": report["demo_use_approved"],
            "eligible_as_round04_clean_plate": report[
                "eligible_as_round04_clean_plate"
            ],
            "eligible_as_current_demo_round04_clean_plate": report[
                "eligible_as_current_demo_round04_clean_plate"
            ],
            "acceptance_status": report["acceptance_status"],
            "acceptance_scope": report["acceptance_scope"],
            "limitations": report["limitations"],
            "override_gate_keys": report["override_gate_keys"],
            "overridden_gates": report["overridden_gates"],
            "failed_metrics": report["failed_metrics"],
            "promotion_approved": False,
        }
        write_json(staging / "layered_composite_receipt.json", receipt)
        shutil.move(str(staging), output_root)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = compose(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "round_index": report["round_index"],
                "aggregate_counts": report["aggregate_counts"],
                "promotion_approved": report["promotion_approved"],
                "report": str(args.output / "layered_composite_report.json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
