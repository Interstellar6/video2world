#!/usr/bin/env python3
"""Execute the fixed four-round front-to-back clean-plate composition sequence."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from .compose_layered_clean_plate import (
        BACKGROUND_RECEIPT_KIND,
        CALIBRATED_MEASURED_ROLE,
        INPUT_KIND,
        MEASURED_RECEIPT_KIND,
        NO_MEASURED_ROLE,
        PBR_RECEIPT_KIND,
        RECEIPT_KIND,
        compose,
    )
else:
    from compose_layered_clean_plate import (
        BACKGROUND_RECEIPT_KIND,
        CALIBRATED_MEASURED_ROLE,
        INPUT_KIND,
        MEASURED_RECEIPT_KIND,
        NO_MEASURED_ROLE,
        PBR_RECEIPT_KIND,
        RECEIPT_KIND,
        compose,
    )
from PIL import Image

SEQUENCE_REPORT_KIND = "video2world.layered_clean_plate_sequence_report"
SEQUENCE_RECEIPT_KIND = "video2world.layered_clean_plate_sequence_receipt"
CUMULATIVE_MANIFEST_KIND = "video2world.cumulative_removal_manifest"
R4_MASK_INDEX_KIND = "video2world.cumulative_removal_mask_index"
PBR_INDEX_KIND = "video2world.source_camera_pbr_geometry_matte_render_index"
PBR_BATCH_RECEIPT_KIND = "video2world.source_camera_pbr_geometry_matte_batch_receipt"
STRUCTURAL_INDEX_KIND = "video2world.structural_background_render_index"
STRUCTURAL_BATCH_RECEIPT_KIND = (
    "video2world.structural_background_render_batch_receipt"
)

OBJECT_ORDER = (
    "sam3_pillow_front",
    "sam3_pillow_left",
    "sam3_pillow_right",
    "sam3_bed_01",
)
ROUND_NAMES = (
    "round01_front_pillow",
    "round02_left_pillow",
    "round03_right_pillow",
    "round04_bed",
)
ROUND_REMAINING = {
    "round01_front_pillow": [
        "sam3_pillow_left",
        "sam3_pillow_right",
        "sam3_bed_01",
    ],
    "round02_left_pillow": ["sam3_pillow_right", "sam3_bed_01"],
    "round03_right_pillow": ["sam3_bed_01"],
    "round04_bed": [],
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def frame_map(records: Any, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in require_list(records, label):
        record = require_dict(raw, f"{label} record")
        frame_id = record.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id or frame_id in result:
            raise ValueError(f"{label} has an invalid or duplicate frame ID")
        result[frame_id] = record
    return result


def parse_rebase_rules(values: list[str]) -> list[tuple[Path, Path]]:
    rules: list[tuple[Path, Path]] = []
    for value in values:
        if "=" not in value:
            raise ValueError("path rebase must use OLD=NEW")
        old_value, new_value = value.split("=", 1)
        old = Path(old_value).expanduser()
        new = Path(new_value).expanduser().resolve()
        if not old.is_absolute() or not new.is_absolute():
            raise ValueError("path rebase roots must be absolute")
        rules.append((old, new))
    rules.sort(key=lambda item: len(item[0].parts), reverse=True)
    return rules


class AssetResolver:
    def __init__(self, rules: list[tuple[Path, Path]]) -> None:
        self.rules = rules

    def _candidates(self, value: str, *, relative_to: Path) -> list[Path]:
        path = Path(value).expanduser()
        candidates: list[Path] = []
        if path.is_absolute():
            candidates.append(path)
            for old, new in self.rules:
                try:
                    relative = path.relative_to(old)
                except ValueError:
                    continue
                candidates.append(new / relative)
        else:
            candidates.append(relative_to / path)
        unique: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            resolved = candidate.resolve()
            if str(resolved) not in seen:
                unique.append(resolved)
                seen.add(str(resolved))
        return unique

    def file(
        self,
        value: str,
        *,
        relative_to: Path,
        label: str,
        expected_sha256: str | None = None,
    ) -> Path:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label}.path must be a non-empty string")
        if expected_sha256 is not None and (
            not isinstance(expected_sha256, str) or len(expected_sha256) != 64
        ):
            raise ValueError(f"{label}.sha256 must be a SHA-256 string")
        mismatches: list[str] = []
        for candidate in self._candidates(value, relative_to=relative_to):
            if not candidate.is_file():
                continue
            if expected_sha256 is None:
                return candidate
            actual = sha256_file(candidate)
            if actual == expected_sha256:
                return candidate
            mismatches.append(f"{candidate}:{actual}")
        if mismatches:
            raise ValueError(f"{label} SHA-256 mismatch: {mismatches}")
        raise FileNotFoundError(f"{label} not found after path rebase: {value}")

    def asset(self, value: Any, *, relative_to: Path, label: str) -> Path:
        record = require_dict(value, label)
        return self.file(
            record.get("path"),
            relative_to=relative_to,
            label=label,
            expected_sha256=record.get("sha256"),
        )


def portable_asset(path: Path, *, relative_to: Path) -> dict[str, str]:
    return {
        "path": os.path.relpath(path, relative_to),
        "sha256": sha256_file(path),
    }


def load_binary_mask(path: Path, label: str) -> np.ndarray:
    with Image.open(path) as image:
        value = np.asarray(image.convert("L"), dtype=np.uint8)
    if not np.all((value == 0) | (value == 255)):
        raise ValueError(f"{label} must be binary 0/255")
    return value == 255


def load_binary_alpha(path: Path, label: str) -> np.ndarray:
    with Image.open(path) as image:
        value = np.asarray(image.convert("RGBA"), dtype=np.uint8)[..., 3]
    if not np.all((value == 0) | (value == 255)):
        raise ValueError(f"{label} alpha must be binary 0/255")
    return value == 255


def resolve_record_asset(
    resolver: AssetResolver,
    record: dict[str, Any],
    path_key: str,
    sha_key: str,
    *,
    base: Path,
    label: str,
) -> Path:
    return resolver.file(
        record.get(path_key),
        relative_to=base,
        label=label,
        expected_sha256=record.get(sha_key),
    )


def measured_report_blocker_summary(report: dict[str, Any]) -> str:
    details: list[str] = []
    status = report.get("status")
    if isinstance(status, str) and status:
        details.append(f"status={status}")
    promotion_blocker = report.get("promotion_blocker")
    if isinstance(promotion_blocker, str) and promotion_blocker:
        details.append(f"promotion_blocker={promotion_blocker}")
    next_action = report.get("next_action")
    if isinstance(next_action, dict):
        action = next_action.get("action")
        blocker = next_action.get("blocker")
        if isinstance(action, str) and action:
            details.append(f"next_action={action}")
        if isinstance(blocker, str) and blocker:
            details.append(f"next_blocker={blocker}")
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


def measured_report_error(message: str, report: dict[str, Any]) -> ValueError:
    return ValueError(message + measured_report_blocker_summary(report))


def measured_report_action_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report.get("status"),
        "promotion_blocker": report.get("promotion_blocker"),
        "next_action": report.get("next_action"),
    }


def validate_measured_round(
    *,
    round_index: int,
    manifest_path: Path,
    report_path: Path,
    receipt_path: Path,
    resolver: AssetResolver,
    expected_frame_count: int,
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    report = read_json(report_path)
    receipt = read_json(receipt_path)
    expected_removed = list(OBJECT_ORDER[:round_index])
    if manifest.get("kind") != CUMULATIVE_MANIFEST_KIND:
        raise ValueError(f"round {round_index} cumulative manifest kind mismatch")
    if manifest.get("round_index") != round_index:
        raise ValueError(f"round {round_index} cumulative manifest skips a round")
    if manifest.get("removed_object_ids") != expected_removed:
        raise ValueError(f"round {round_index} removed-object mapping mismatch")
    gates = require_dict(manifest.get("gates"), "cumulative manifest gates")
    if not all(value is True for value in gates.values()):
        raise ValueError(f"round {round_index} cumulative manifest gate failed")
    if report.get("status") != "technical_passed":
        raise measured_report_error(f"round {round_index} measured report did not pass", report)
    manifest_sha = sha256_file(manifest_path)
    if report.get("input_manifest_sha256") != manifest_sha:
        raise ValueError(f"round {round_index} measured report does not bind manifest")
    cumulative = require_dict(
        report.get("cumulative_removal_contract"),
        "measured cumulative removal contract",
    )
    if (
        cumulative.get("round_index") != round_index
        or cumulative.get("removed_object_ids") != expected_removed
        or cumulative.get("manifest_sha256") != manifest_sha
    ):
        raise ValueError(f"round {round_index} measured cumulative contract mismatch")
    provenance = require_dict(report.get("pixel_provenance"), "measured provenance")
    if provenance.get("generated_pixels") != 0:
        raise ValueError("measured report contains generated pixels")
    if provenance.get("propainter_pixels") != 0:
        raise ValueError("measured report omits or contains ProPainter pixels")
    if receipt.get("kind") != MEASURED_RECEIPT_KIND:
        raise ValueError(f"round {round_index} measured receipt kind mismatch")
    if receipt.get("report_sha256") != sha256_file(report_path):
        raise ValueError(f"round {round_index} measured receipt does not bind report")
    if receipt.get("frame_artifacts_are_hashed_in_report") is not True:
        raise ValueError(f"round {round_index} measured frame artifacts are not hash-bound")
    receipt_report = receipt.get("report")
    if isinstance(receipt_report, str):
        bound_report = resolver.file(
            receipt_report,
            relative_to=receipt_path.parent,
            label=f"round {round_index} receipt report",
            expected_sha256=receipt["report_sha256"],
        )
        if sha256_file(bound_report) != sha256_file(report_path):
            raise ValueError(f"round {round_index} receipt binds a different report")

    manifest_records = frame_map(manifest.get("frame_records"), "cumulative frames")
    report_records = frame_map(report.get("frame_records"), "measured frames")
    if set(manifest_records) != set(report_records):
        raise ValueError(f"round {round_index} manifest/report frame mismatch")
    if len(manifest_records) != expected_frame_count:
        raise ValueError(f"round {round_index} frame count mismatch")
    ordered_ids = [record["frame_id"] for record in manifest["frame_records"]]
    if [record.get("sequence_index") for record in manifest["frame_records"]] != list(
        range(expected_frame_count)
    ):
        raise ValueError(f"round {round_index} frame sequence is not contiguous")

    assets: dict[str, dict[str, Path]] = {}
    for frame_id in ordered_ids:
        manifest_record = manifest_records[frame_id]
        report_record = report_records[frame_id]
        union_path = resolve_record_asset(
            resolver,
            manifest_record,
            "union_mask",
            "union_mask_sha256",
            base=manifest_path.parent,
            label=f"round {round_index}/{frame_id} union mask",
        )
        removal_path = resolve_record_asset(
            resolver,
            report_record,
            "removal_mask",
            "removal_mask_sha256",
            base=report_path.parent,
            label=f"round {round_index}/{frame_id} report removal mask",
        )
        if sha256_file(union_path) != sha256_file(removal_path):
            raise ValueError(f"round {round_index}/{frame_id} removal mask mismatch")
        prefill_path = resolve_record_asset(
            resolver,
            report_record,
            "prefill_frame",
            "prefill_frame_sha256",
            base=report_path.parent,
            label=f"round {round_index}/{frame_id} measured prefill",
        )
        residual_path = resolve_record_asset(
            resolver,
            report_record,
            "residual_mask",
            "residual_mask_sha256",
            base=report_path.parent,
            label=f"round {round_index}/{frame_id} residual mask",
        )
        depth_path = resolve_record_asset(
            resolver,
            report_record,
            "measured_depth",
            "measured_depth_sha256",
            base=report_path.parent,
            label=f"round {round_index}/{frame_id} measured depth",
        )
        if report_record.get("measured_depth_role") != "fused_multiview_measured_depth":
            raise ValueError(f"round {round_index}/{frame_id} measured depth role mismatch")
        source_path = resolve_record_asset(
            resolver,
            report_record,
            "source_frame",
            "source_frame_sha256",
            base=report_path.parent,
            label=f"round {round_index}/{frame_id} source frame",
        )
        removal = load_binary_mask(removal_path, f"round {round_index} removal")
        residual = load_binary_mask(residual_path, f"round {round_index} residual")
        if residual.shape != removal.shape or np.any(residual & ~removal):
            raise ValueError(f"round {round_index}/{frame_id} residual escapes removal")
        depth = np.load(depth_path, allow_pickle=False)
        measured = removal & ~residual
        if depth.shape != removal.shape:
            raise ValueError(f"round {round_index}/{frame_id} measured depth shape mismatch")
        if not np.array_equal(np.isfinite(depth), measured):
            raise ValueError(f"round {round_index}/{frame_id} measured depth support mismatch")
        assets[frame_id] = {
            "source": source_path,
            "removal": removal_path,
            "prefill": prefill_path,
            "residual": residual_path,
            "depth": depth_path,
        }
    return {
        "round_index": round_index,
        "manifest_path": manifest_path,
        "report_path": report_path,
        "receipt_path": receipt_path,
        "manifest": manifest,
        "report": report,
        "receipt": receipt,
        "action_summary": measured_report_action_summary(report),
        "frame_ids": ordered_ids,
        "manifest_records": manifest_records,
        "report_records": report_records,
        "assets": assets,
    }


def validate_measured_lineage(rounds: list[dict[str, Any]]) -> None:
    for index, current in enumerate(rounds):
        expected_previous = index if index else None
        manifest_lineage = require_dict(
            current["manifest"].get("lineage"),
            f"round {index + 1} manifest lineage",
        )
        if manifest_lineage.get("previous_round_index") != expected_previous:
            raise ValueError(f"round {index + 1} lineage skips an immediately previous round")
        report_lineage = require_dict(
            current["report"]["cumulative_removal_contract"].get("lineage"),
            f"round {index + 1} report lineage",
        )
        if report_lineage.get("previous_round_index") != expected_previous:
            raise ValueError(f"round {index + 1} report lineage skips a round")
        if index:
            previous = rounds[index - 1]
            previous_manifest_sha = sha256_file(previous["manifest_path"])
            previous_report_sha = sha256_file(previous["report_path"])
            if manifest_lineage.get("previous_cumulative_manifest_sha256") != (
                previous_manifest_sha
            ):
                raise ValueError(f"round {index + 1} lineage manifest hash mismatch")
            if manifest_lineage.get("previous_prefill_report_sha256") != previous_report_sha:
                raise ValueError(f"round {index + 1} lineage report hash mismatch")
            if report_lineage.get("previous_cumulative_manifest_sha256") != (
                previous_manifest_sha
            ):
                raise ValueError(f"round {index + 1} measured lineage manifest mismatch")
            if report_lineage.get("previous_prefill_report_sha256") != previous_report_sha:
                raise ValueError(f"round {index + 1} measured lineage report mismatch")


def receipt_bound_index(
    *,
    receipt: dict[str, Any],
    receipt_path: Path,
    index_path: Path,
    resolver: AssetResolver,
    keys: tuple[str, ...],
    label: str,
) -> None:
    expected_sha = sha256_file(index_path)
    for key in keys:
        value = receipt.get(key)
        if isinstance(value, dict):
            bound = resolver.asset(value, relative_to=receipt_path.parent, label=label)
            if sha256_file(bound) != expected_sha:
                raise ValueError(f"{label} binds a different index")
            return
    for key in ("render_index_sha256", "index_sha256"):
        if receipt.get(key) == expected_sha:
            return
    raise ValueError(f"{label} does not hash-bind its index")


def validate_pbr_batch(
    *,
    index_path: Path,
    receipt_path: Path,
    resolver: AssetResolver,
    expected_frame_ids: list[str],
) -> dict[str, Any]:
    index = read_json(index_path)
    receipt = read_json(receipt_path)
    if index.get("kind") != PBR_INDEX_KIND:
        raise ValueError("PBR render index kind mismatch")
    if receipt.get("kind") != PBR_BATCH_RECEIPT_KIND:
        raise ValueError("PBR batch receipt kind mismatch")
    receipt_bound_index(
        receipt=receipt,
        receipt_path=receipt_path,
        index_path=index_path,
        resolver=resolver,
        keys=("render_index",),
        label="PBR batch receipt",
    )
    if index.get("frame_ids") != expected_frame_ids:
        raise ValueError("PBR render index has wrong or missing frames")
    if index.get("layer_order") != list(OBJECT_ORDER):
        raise ValueError("PBR render index is missing an expected layer")
    if index.get("round_remaining_object_ids") != ROUND_REMAINING:
        raise ValueError("PBR render index would re-render a removed object")
    contract = require_dict(index.get("contract"), "PBR index contract")
    if (
        contract.get("alpha_depth_alignment_mode")
        != "geometry-derived-binary-matte"
        or contract.get("final_alpha_equals_finite_depth_support") is not True
        or contract.get("claims_measured_donor") is not False
    ):
        raise ValueError("PBR render index contract is not compositor-safe")
    raw_frames = require_dict(index.get("frames"), "PBR frames")
    if set(raw_frames) != set(expected_frame_ids):
        raise ValueError("PBR frame map differs from frame_ids")
    frames: dict[str, Any] = {}
    for frame_id in expected_frame_ids:
        frame = require_dict(raw_frames[frame_id], f"PBR frame {frame_id}")
        source_path = resolver.asset(
            frame.get("source_frame"),
            relative_to=index_path.parent,
            label=f"PBR source {frame_id}",
        )
        raw_layers = require_dict(frame.get("layers"), f"PBR layers {frame_id}")
        if set(raw_layers) != set(OBJECT_ORDER):
            raise ValueError(f"PBR frame {frame_id} is missing a layer")
        layers: dict[str, Any] = {}
        for layer_id in OBJECT_ORDER:
            layer = require_dict(raw_layers[layer_id], f"PBR {frame_id}/{layer_id}")
            rgba_path = resolver.asset(
                layer.get("rgba"),
                relative_to=index_path.parent,
                label=f"PBR {frame_id}/{layer_id} RGBA",
            )
            depth_path = resolver.asset(
                layer.get("depth"),
                relative_to=index_path.parent,
                label=f"PBR {frame_id}/{layer_id} depth",
            )
            frame_receipt_path = resolver.asset(
                layer.get("receipt"),
                relative_to=index_path.parent,
                label=f"PBR {frame_id}/{layer_id} receipt",
            )
            frame_receipt = read_json(frame_receipt_path)
            if (
                frame_receipt.get("kind") != PBR_RECEIPT_KIND
                or frame_receipt.get("layer_id") != layer_id
                or frame_receipt.get("frame_id") != frame_id
                or frame_receipt.get("claims_measured_donor") is not False
                or not all(require_dict(frame_receipt.get("gates"), "PBR gates").values())
            ):
                raise ValueError(f"PBR receipt failed for {frame_id}/{layer_id}")
            if frame_receipt.get("rgba_sha256") != sha256_file(rgba_path):
                raise ValueError(f"PBR receipt RGBA mismatch for {frame_id}/{layer_id}")
            if frame_receipt.get("depth_sha256") != sha256_file(depth_path):
                raise ValueError(f"PBR receipt depth mismatch for {frame_id}/{layer_id}")
            layers[layer_id] = {
                "rgba": rgba_path,
                "depth": depth_path,
                "receipt": frame_receipt_path,
            }
        frames[frame_id] = {"source": source_path, "layers": layers}
    return {
        "index_path": index_path,
        "receipt_path": receipt_path,
        "index": index,
        "receipt": receipt,
        "frames": frames,
    }


def validate_structural_batch(
    *,
    index_path: Path,
    receipt_path: Path,
    resolver: AssetResolver,
    expected_frame_ids: list[str],
) -> dict[str, Any]:
    index = read_json(index_path)
    receipt = read_json(receipt_path)
    if index.get("kind") != STRUCTURAL_INDEX_KIND:
        raise ValueError("structural background index kind mismatch")
    if receipt.get("kind") != STRUCTURAL_BATCH_RECEIPT_KIND:
        raise ValueError("structural background batch receipt kind mismatch")
    receipt_bound_index(
        receipt=receipt,
        receipt_path=receipt_path,
        index_path=index_path,
        resolver=resolver,
        keys=("render_index", "index", "structural_background_index"),
        label="structural background batch receipt",
    )
    raw_records = index.get("frame_records")
    uses_materialized_schema = isinstance(raw_records, list)
    if uses_materialized_schema:
        raw_frames = frame_map(raw_records, "structural background frame records")
        ordered_ids = [record.get("frame_id") for record in raw_records]
        sequence_indices = [record.get("sequence_index") for record in raw_records]
        if sequence_indices != list(range(len(expected_frame_ids))):
            raise ValueError("structural background sequence indices are not contiguous")
        if index.get("frame_count") != len(expected_frame_ids):
            raise ValueError("structural background index frame count mismatch")
        if receipt.get("frame_count") != len(expected_frame_ids):
            raise ValueError("structural background receipt frame count mismatch")
    else:
        ordered_ids = index.get("frame_ids")
        raw_frames = require_dict(index.get("frames"), "structural background frames")
    if ordered_ids != expected_frame_ids:
        raise ValueError("structural background index has wrong or missing frames")
    if set(raw_frames) != set(expected_frame_ids):
        raise ValueError("structural background frame map differs from ordered frames")
    frames: dict[str, Any] = {}
    for sequence_index, frame_id in enumerate(expected_frame_ids):
        frame = require_dict(raw_frames[frame_id], f"structural frame {frame_id}")
        if uses_materialized_schema and not all(
            require_dict(frame.get("gates"), f"structural {frame_id} gates").values()
        ):
            raise ValueError(f"structural frame gates failed for {frame_id}")
        rgba_path = resolver.asset(
            frame.get("rgba"),
            relative_to=index_path.parent,
            label=f"structural {frame_id} RGBA",
        )
        depth_path = resolver.asset(
            frame.get("depth"),
            relative_to=index_path.parent,
            label=f"structural {frame_id} depth",
        )
        frame_receipt_path = resolver.asset(
            frame.get("receipt"),
            relative_to=index_path.parent,
            label=f"structural {frame_id} receipt",
        )
        frame_receipt = read_json(frame_receipt_path)
        if (
            frame_receipt.get("kind") != BACKGROUND_RECEIPT_KIND
            or (
                uses_materialized_schema
                and frame_receipt.get("frame_id") != frame_id
            )
            or (
                uses_materialized_schema
                and frame_receipt.get("sequence_index") != sequence_index
            )
            or frame_receipt.get("role") != "structural_background"
            or frame_receipt.get("provenance_class") != "structural_background"
            or frame_receipt.get("claims_measured_donor") is not False
        ):
            raise ValueError(f"structural receipt failed for {frame_id}")
        if frame_receipt.get("rgba_sha256") != sha256_file(rgba_path):
            raise ValueError(f"structural receipt RGBA mismatch for {frame_id}")
        if frame_receipt.get("depth_sha256") != sha256_file(depth_path):
            raise ValueError(f"structural receipt depth mismatch for {frame_id}")
        acceptance_report_path = resolver.asset(
            frame_receipt.get("accepted_texture_report"),
            relative_to=frame_receipt_path.parent,
            label=f"structural {frame_id} accepted texture report",
        )
        acceptance_report = read_json(acceptance_report_path)
        if acceptance_report.get("status") != "accepted_for_round04_clean_plate":
            raise ValueError(f"structural texture report is not accepted for {frame_id}")
        source_mask_path: Path | None = None
        if uses_materialized_schema:
            source_mask = require_dict(
                frame.get("source_cumulative_removal_mask"),
                f"structural {frame_id} source cumulative removal mask",
            )
            source_mask_path = resolver.asset(
                source_mask,
                relative_to=index_path.parent,
                label=f"structural {frame_id} source cumulative removal mask",
            )
            mask = load_binary_mask(
                source_mask_path,
                f"structural {frame_id} source cumulative removal mask",
            )
            alpha = load_binary_alpha(rgba_path, f"structural {frame_id} RGBA")
            if not np.array_equal(alpha, mask):
                raise ValueError(
                    f"structural {frame_id} alpha differs from cumulative removal mask"
                )
            mask_pixels = int(mask.sum())
            if source_mask.get("pixels") != mask_pixels:
                raise ValueError(f"structural {frame_id} source mask pixel count mismatch")
            if frame.get("alpha_pixels") != mask_pixels:
                raise ValueError(f"structural {frame_id} alpha pixel count mismatch")
        frames[frame_id] = {
            "rgba": rgba_path,
            "depth": depth_path,
            "receipt": frame_receipt_path,
            "receipt_value": frame_receipt,
            "acceptance_report": acceptance_report_path,
            "acceptance_report_value": acceptance_report,
            "source_mask": source_mask_path,
        }
    return {
        "index_path": index_path,
        "receipt_path": receipt_path,
        "index": index,
        "receipt": receipt,
        "frames": frames,
    }


def validate_r4_masks(
    *,
    index_path: Path,
    resolver: AssetResolver,
    expected_frame_ids: list[str],
) -> dict[str, Any]:
    value = read_json(index_path)
    if value.get("kind") == CUMULATIVE_MANIFEST_KIND:
        if value.get("round_index") != 4:
            raise ValueError("R4 cumulative manifest skips round 4")
        if value.get("removed_object_ids") != list(OBJECT_ORDER):
            raise ValueError("R4 cumulative removed-object mapping mismatch")
        records = frame_map(value.get("frame_records"), "R4 cumulative frames")
        ordered_ids = [record["frame_id"] for record in value["frame_records"]]
        masks = {
            frame_id: resolve_record_asset(
                resolver,
                records[frame_id],
                "union_mask",
                "union_mask_sha256",
                base=index_path.parent,
                label=f"R4 cumulative mask {frame_id}",
            )
            for frame_id in ordered_ids
        }
    elif value.get("kind") == R4_MASK_INDEX_KIND:
        if value.get("round_index") != 4 or value.get("removed_object_ids") != list(
            OBJECT_ORDER
        ):
            raise ValueError("R4 cumulative mask index contract mismatch")
        ordered_ids = require_list(value.get("frame_ids"), "R4 frame_ids")
        raw_frames = require_dict(value.get("frames"), "R4 frames")
        masks = {}
        for frame_id in ordered_ids:
            frame = require_dict(raw_frames.get(frame_id), f"R4 frame {frame_id}")
            asset_value = frame.get("mask") or frame.get("cumulative_removal_mask")
            masks[frame_id] = resolver.asset(
                asset_value,
                relative_to=index_path.parent,
                label=f"R4 cumulative mask {frame_id}",
            )
    elif value.get("kind") == STRUCTURAL_INDEX_KIND:
        records = frame_map(
            value.get("frame_records"),
            "structural R4 source mask frames",
        )
        ordered_ids = [record.get("frame_id") for record in value["frame_records"]]
        if [record.get("sequence_index") for record in value["frame_records"]] != list(
            range(len(expected_frame_ids))
        ):
            raise ValueError("structural R4 source mask sequence is not contiguous")
        masks = {
            frame_id: resolver.asset(
                require_dict(
                    records[frame_id].get("source_cumulative_removal_mask"),
                    f"structural R4 source mask {frame_id}",
                ),
                relative_to=index_path.parent,
                label=f"structural R4 source mask {frame_id}",
            )
            for frame_id in ordered_ids
        }
        value = {
            "schema_version": 1,
            "kind": R4_MASK_INDEX_KIND,
            "round_index": 4,
            "removed_object_ids": list(OBJECT_ORDER),
            "frame_ids": ordered_ids,
            "frames": {
                frame_id: {
                    "mask": {
                        "path": str(masks[frame_id]),
                        "sha256": sha256_file(masks[frame_id]),
                    }
                }
                for frame_id in ordered_ids
            },
            "derived_from_structural_index": {
                "path": str(index_path),
                "sha256": sha256_file(index_path),
            },
        }
    else:
        raise ValueError("R4 cumulative masks have the wrong kind")
    if ordered_ids != expected_frame_ids or set(masks) != set(expected_frame_ids):
        raise ValueError("R4 cumulative masks have wrong or missing frames")
    for frame_id, path in masks.items():
        load_binary_mask(path, f"R4 cumulative mask {frame_id}")
    return {
        "index_path": index_path,
        "value": value,
        "masks": masks,
        "frame_ids": ordered_ids,
    }


def materialize_r4_masks(
    context: dict[str, Any],
    normalized_root: Path,
) -> Path:
    mask_root = normalized_root / "round04_cumulative_masks"
    frames: dict[str, Any] = {}
    for sequence_index, frame_id in enumerate(context["frame_ids"]):
        source_path = context["masks"][frame_id]
        destination = mask_root / "masks" / f"{sequence_index:04d}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination)
        if sha256_file(destination) != sha256_file(source_path):
            raise ValueError(f"materialized R4 mask hash mismatch for {frame_id}")
        frames[frame_id] = {
            "sequence_index": sequence_index,
            "mask": portable_asset(destination, relative_to=mask_root),
            "source_mask": {
                "path": str(source_path),
                "sha256": sha256_file(source_path),
            },
        }
        context["masks"][frame_id] = destination
    index_path = mask_root / "cumulative_removal_mask_index.json"
    index = {
        "schema_version": 1,
        "kind": R4_MASK_INDEX_KIND,
        "round_index": 4,
        "removed_object_ids": list(OBJECT_ORDER),
        "frame_ids": list(context["frame_ids"]),
        "frame_count": len(context["frame_ids"]),
        "copy_mode": "byte_exact_from_hash_verified_source_mask",
        "source_index": {
            "path": str(context["index_path"]),
            "sha256": sha256_file(context["index_path"]),
        },
        "frames": frames,
    }
    write_json(index_path, index)
    context["portable_index_path"] = index_path
    return index_path


def normalize_measured_report(
    context: dict[str, Any],
    normalized_root: Path,
) -> tuple[Path, Path]:
    round_index = context["round_index"]
    report = copy.deepcopy(context["report"])
    report["input_manifest"] = str(context["manifest_path"])
    report_records = frame_map(report.get("frame_records"), "normalized measured frames")
    for frame_id in context["frame_ids"]:
        record = report_records[frame_id]
        assets = context["assets"][frame_id]
        record["source_frame"] = str(assets["source"])
        record["removal_mask"] = str(assets["removal"])
        record["prefill_frame"] = str(assets["prefill"])
        record["residual_mask"] = str(assets["residual"])
        record["measured_depth"] = str(assets["depth"])
    report_path = normalized_root / f"round{round_index:02d}_measured_report.json"
    write_json(report_path, report)

    receipt = copy.deepcopy(context["receipt"])
    receipt["report"] = report_path.name
    receipt["report_sha256"] = sha256_file(report_path)
    receipt["path_rebased"] = True
    receipt["all_rebased_artifacts_sha256_revalidated"] = True
    receipt["source_report"] = {
        "path": str(context["report_path"]),
        "sha256": sha256_file(context["report_path"]),
    }
    receipt["source_receipt"] = {
        "path": str(context["receipt_path"]),
        "sha256": sha256_file(context["receipt_path"]),
    }
    receipt_path = normalized_root / f"round{round_index:02d}_measured_receipt.json"
    write_json(receipt_path, receipt)
    return report_path, receipt_path


def normalize_structural_receipts(
    context: dict[str, Any],
    normalized_root: Path,
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    reports_root = normalized_root / "structural_texture_reports"
    receipts_root = normalized_root / "structural_receipts"
    for frame_id, frame in context["frames"].items():
        acceptance_path = reports_root / f"{frame_id}.json"
        write_json(acceptance_path, frame["acceptance_report_value"])
        receipt = copy.deepcopy(frame["receipt_value"])
        receipt["accepted_texture_report"] = portable_asset(
            acceptance_path,
            relative_to=receipts_root,
        )
        receipt["accepted_texture_report_sha256"] = sha256_file(acceptance_path)
        receipt["path_rebased"] = True
        receipt["all_rebased_artifacts_sha256_revalidated"] = True
        receipt["source_receipt"] = {
            "path": str(frame["receipt"]),
            "sha256": sha256_file(frame["receipt"]),
        }
        receipt_path = receipts_root / f"{frame_id}.json"
        write_json(receipt_path, receipt)
        result[frame_id] = receipt_path
    return result


def preflight(args: argparse.Namespace, *, expected_frame_count: int) -> dict[str, Any]:
    resolver = AssetResolver(parse_rebase_rules(args.path_rebase))

    def input_file(value: Path, label: str) -> Path:
        return resolver.file(
            str(value.expanduser()),
            relative_to=Path.cwd(),
            label=label,
        )

    measured_rounds = []
    for round_index in range(1, 4):
        measured_rounds.append(
            validate_measured_round(
                round_index=round_index,
                manifest_path=input_file(
                    getattr(args, f"round{round_index}_cumulative_manifest"),
                    f"round {round_index} cumulative manifest",
                ),
                report_path=input_file(
                    getattr(args, f"round{round_index}_measured_report"),
                    f"round {round_index} measured report",
                ),
                receipt_path=input_file(
                    getattr(args, f"round{round_index}_measured_receipt"),
                    f"round {round_index} measured receipt",
                ),
                resolver=resolver,
                expected_frame_count=expected_frame_count,
            )
        )
    validate_measured_lineage(measured_rounds)
    frame_ids = measured_rounds[0]["frame_ids"]
    if any(context["frame_ids"] != frame_ids for context in measured_rounds[1:]):
        raise ValueError("measured rounds have different or reordered frames")
    pbr = validate_pbr_batch(
        index_path=input_file(args.pbr_render_index, "PBR render index"),
        receipt_path=input_file(args.pbr_batch_receipt, "PBR batch receipt"),
        resolver=resolver,
        expected_frame_ids=frame_ids,
    )
    structural = validate_structural_batch(
        index_path=input_file(args.structural_background_index, "structural index"),
        receipt_path=input_file(args.structural_background_receipt, "structural receipt"),
        resolver=resolver,
        expected_frame_ids=frame_ids,
    )
    r4 = validate_r4_masks(
        index_path=input_file(args.round4_cumulative_masks, "R4 cumulative masks"),
        resolver=resolver,
        expected_frame_ids=frame_ids,
    )
    for frame_id in frame_ids:
        if sha256_file(measured_rounds[0]["assets"][frame_id]["source"]) != sha256_file(
            pbr["frames"][frame_id]["source"]
        ):
            raise ValueError(f"R1 source and PBR source differ for {frame_id}")
        structural_mask = structural["frames"][frame_id].get("source_mask")
        if structural_mask is not None and sha256_file(structural_mask) != sha256_file(
            r4["masks"][frame_id]
        ):
            raise ValueError(
                f"structural source mask and R4 cumulative mask differ for {frame_id}"
            )
    return {
        "resolver": resolver,
        "measured_rounds": measured_rounds,
        "pbr": pbr,
        "structural": structural,
        "r4": r4,
        "frame_ids": frame_ids,
    }


def previous_frame_sources(
    report_path: Path,
    report: dict[str, Any],
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for record in require_list(report.get("frame_records"), "previous composite frames"):
        frame = require_dict(record, "previous composite frame")
        frame_id = str(frame["frame_id"])
        outputs = require_dict(frame.get("outputs"), "previous frame outputs")
        rgb = require_dict(outputs.get("composite_rgb"), "previous composite RGB")
        path = (report_path.parent / str(rgb["path"])).resolve()
        if not path.is_file() or sha256_file(path) != rgb.get("sha256"):
            raise ValueError(f"previous composite RGB hash mismatch for {frame_id}")
        result[frame_id] = path
    return result


def write_measured_mask(path: Path, removal_path: Path, residual_path: Path) -> None:
    removal = load_binary_mask(removal_path, "measured removal mask")
    residual = load_binary_mask(residual_path, "measured residual mask")
    measured = removal & ~residual
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(measured.astype(np.uint8) * 255).save(path)


def write_empty_measured_assets(
    mask_path: Path,
    depth_path: Path,
    removal_path: Path,
) -> None:
    removal = load_binary_mask(removal_path, "R4 removal mask")
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    depth_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros(removal.shape, dtype=np.uint8)).save(mask_path)
    np.save(depth_path, np.full(removal.shape, np.nan, dtype=np.float32))


def build_round_manifest(
    *,
    round_index: int,
    round_root: Path,
    context: dict[str, Any],
    normalized_measured: dict[int, tuple[Path, Path]],
    normalized_structural: dict[str, Path],
    previous: dict[str, Any] | None,
) -> Path:
    round_name = ROUND_NAMES[round_index - 1]
    manifest_path = round_root / "compositor_input_manifest.json"
    remaining = ROUND_REMAINING[round_name]
    removed = list(OBJECT_ORDER[:round_index])
    previous_sources = previous["sources"] if previous is not None else {}
    frame_records = []
    for sequence_index, frame_id in enumerate(context["frame_ids"]):
        if round_index <= 3:
            measured_context = context["measured_rounds"][round_index - 1]
            assets = measured_context["assets"][frame_id]
            source_path = assets["source"] if round_index == 1 else previous_sources[frame_id]
            removal_path = assets["removal"]
            measured_mask_path = round_root / "measured_masks" / f"{sequence_index:04d}.png"
            write_measured_mask(measured_mask_path, removal_path, assets["residual"])
            measured_depth_path = assets["depth"]
            donor_prefill: dict[str, str] | None = portable_asset(
                assets["prefill"], relative_to=manifest_path.parent
            )
        else:
            source_path = previous_sources[frame_id]
            removal_path = context["r4"]["masks"][frame_id]
            measured_mask_path = round_root / "measured_masks" / f"{sequence_index:04d}.png"
            measured_depth_path = round_root / "measured_depth" / f"{sequence_index:04d}.npy"
            write_empty_measured_assets(
                measured_mask_path,
                measured_depth_path,
                removal_path,
            )
            donor_prefill = None
        downstream_layers = []
        for depth_order, layer_id in enumerate(remaining):
            layer = context["pbr"]["frames"][frame_id]["layers"][layer_id]
            downstream_layers.append(
                {
                    "layer_id": layer_id,
                    "depth_order": depth_order,
                    "role": "downstream_object_render",
                    "rgba": portable_asset(layer["rgba"], relative_to=manifest_path.parent),
                    "depth": portable_asset(layer["depth"], relative_to=manifest_path.parent),
                    "receipt": portable_asset(
                        layer["receipt"], relative_to=manifest_path.parent
                    ),
                }
            )
        structural = context["structural"]["frames"][frame_id]
        frame_records.append(
            {
                "sequence_index": sequence_index,
                "frame_id": frame_id,
                "source_rgb": portable_asset(source_path, relative_to=manifest_path.parent),
                "cumulative_removal_mask": portable_asset(
                    removal_path, relative_to=manifest_path.parent
                ),
                "donor_prefill_rgb": donor_prefill,
                "donor_measured_mask": portable_asset(
                    measured_mask_path, relative_to=manifest_path.parent
                ),
                "donor_measured_depth": portable_asset(
                    measured_depth_path, relative_to=manifest_path.parent
                ),
                "downstream_layers": downstream_layers,
                "structural_background": {
                    "role": "structural_background",
                    "rgba": portable_asset(
                        structural["rgba"], relative_to=manifest_path.parent
                    ),
                    "depth": portable_asset(
                        structural["depth"], relative_to=manifest_path.parent
                    ),
                    "receipt": portable_asset(
                        normalized_structural[frame_id],
                        relative_to=manifest_path.parent,
                    ),
                },
            }
        )
    if round_index <= 3:
        report_path, receipt_path = normalized_measured[round_index]
        measured_report = context["measured_rounds"][round_index - 1]["report"]
        measured_contract = {
            "role": CALIBRATED_MEASURED_ROLE,
            "report": portable_asset(report_path, relative_to=manifest_path.parent),
            "receipt": portable_asset(receipt_path, relative_to=manifest_path.parent),
            "claims_original_observed_rgb_donor": True,
            "measured_pixels": measured_report["pixel_provenance"].get(
                "measured_multiview_pixels", 0
            ),
            "generated_pixels": 0,
            "propainter_pixels": 0,
        }
    else:
        measured_contract = {
            "role": NO_MEASURED_ROLE,
            "report": None,
            "receipt": None,
            "claims_original_observed_rgb_donor": False,
            "measured_pixels": 0,
            "generated_pixels": 0,
            "propainter_pixels": 0,
        }
    previous_value = None
    if previous is not None:
        previous_value = {
            "round_index": round_index - 1,
            "report": portable_asset(
                previous["report_path"], relative_to=manifest_path.parent
            ),
            "receipt": portable_asset(
                previous["receipt_path"], relative_to=manifest_path.parent
            ),
        }
    manifest = {
        "schema_version": 1,
        "kind": INPUT_KIND,
        "round_index": round_index,
        "round_kind": "final_background" if round_index == 4 else "object_peel",
        "removed_object_ids": removed,
        "newly_removed_object_id": removed[-1],
        "remaining_object_ids": remaining,
        "source_contract": {
            "role": (
                "original_observed_rgb"
                if round_index == 1
                else "previous_layered_composite"
            ),
            "untracked_generated_rgb": False,
            "propainter_rgb": False,
        },
        "previous_round": previous_value,
        "measured_donor_contract": measured_contract,
        "config": {"alpha_threshold": 0.5, "depth_tie_epsilon": 1e-6},
        "frame_records": frame_records,
    }
    write_json(manifest_path, manifest)
    return manifest_path


def execute_round(
    *,
    round_index: int,
    manifest_path: Path,
    round_root: Path,
) -> dict[str, Any]:
    composite_root = round_root / "composite"
    report = compose(
        argparse.Namespace(input_manifest=manifest_path, output=composite_root)
    )
    report_path = composite_root / "layered_composite_report.json"
    receipt_path = composite_root / "layered_composite_receipt.json"
    receipt = read_json(receipt_path)
    if receipt.get("kind") != RECEIPT_KIND:
        raise ValueError(f"round {round_index} composite receipt kind mismatch")
    if receipt.get("report_sha256") != sha256_file(report_path):
        raise ValueError(f"round {round_index} receipt does not bind report")
    counts = require_dict(report.get("aggregate_counts"), "composite aggregate counts")
    if counts.get("unresolved_pixels") != 0:
        raise ValueError(f"round {round_index} has unresolved pixels and cannot continue")
    expected_statuses = {"technical_passed_complete_partition"}
    if round_index == 4:
        expected_statuses.add("technical_passed_complete_partition_with_limitations")
    if report.get("status") not in expected_statuses:
        raise ValueError(f"round {round_index} composite did not complete")
    if round_index == 4:
        gates = require_dict(report.get("gates"), "round 4 gates")
        if gates.get("final_background_entire_removal_is_structural") is not True:
            raise ValueError("round 4 structural background does not cover the full mask")
    return {
        "report": report,
        "receipt": receipt,
        "report_path": report_path,
        "receipt_path": receipt_path,
        "sources": previous_frame_sources(report_path, report),
    }


def input_binding(path: Path, *, relative_to: Path) -> dict[str, str]:
    return portable_asset(path, relative_to=relative_to)


def run_sequence(
    args: argparse.Namespace,
    *,
    expected_frame_count: int = 25,
) -> dict[str, Any]:
    output_root = args.output.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_root}")
    context = preflight(args, expected_frame_count=expected_frame_count)
    output_root.mkdir(parents=True, exist_ok=True)
    normalized_root = output_root / "normalized_inputs"
    materialize_r4_masks(context["r4"], normalized_root)
    normalized_measured = {
        item["round_index"]: normalize_measured_report(item, normalized_root)
        for item in context["measured_rounds"]
    }
    normalized_structural = normalize_structural_receipts(
        context["structural"], normalized_root
    )

    previous: dict[str, Any] | None = None
    round_results: list[dict[str, Any]] = []
    for round_index, round_name in enumerate(ROUND_NAMES, start=1):
        round_root = output_root / round_name
        manifest_path = build_round_manifest(
            round_index=round_index,
            round_root=round_root,
            context=context,
            normalized_measured=normalized_measured,
            normalized_structural=normalized_structural,
            previous=previous,
        )
        result = execute_round(
            round_index=round_index,
            manifest_path=manifest_path,
            round_root=round_root,
        )
        result["manifest_path"] = manifest_path
        round_results.append(result)
        previous = result

    final_report = round_results[-1]["report"]
    accepted_with_limitations = final_report.get("accepted_with_limitations") is True
    sequence_status = (
        "technical_passed_complete_sequence_with_limitations"
        if accepted_with_limitations
        else "technical_passed_complete_sequence"
    )
    report_path = output_root / "layered_clean_plate_sequence_report.json"
    receipt_path = output_root / "layered_clean_plate_sequence_receipt.json"
    report = {
        "schema_version": 1,
        "kind": SEQUENCE_REPORT_KIND,
        "status": sequence_status,
        "promotion_approved": False,
        "frame_count": len(context["frame_ids"]),
        "removed_object_order": list(OBJECT_ORDER),
        "round_remaining_object_ids": ROUND_REMAINING,
        "path_rebase_rules": [
            {"old": str(old), "new": str(new)}
            for old, new in context["resolver"].rules
        ],
        "inputs": {
            "measured_rounds": [
                {
                    "round_index": item["round_index"],
                    "cumulative_manifest": input_binding(
                        item["manifest_path"], relative_to=report_path.parent
                    ),
                    "measured_report": input_binding(
                        item["report_path"], relative_to=report_path.parent
                    ),
                    "measured_receipt": input_binding(
                        item["receipt_path"], relative_to=report_path.parent
                    ),
                    "measured_action_summary": item["action_summary"],
                }
                for item in context["measured_rounds"]
            ],
            "pbr_render_index": input_binding(
                context["pbr"]["index_path"], relative_to=report_path.parent
            ),
            "pbr_batch_receipt": input_binding(
                context["pbr"]["receipt_path"], relative_to=report_path.parent
            ),
            "structural_background_index": input_binding(
                context["structural"]["index_path"], relative_to=report_path.parent
            ),
            "structural_background_receipt": input_binding(
                context["structural"]["receipt_path"], relative_to=report_path.parent
            ),
            "round4_cumulative_masks": input_binding(
                context["r4"]["portable_index_path"],
                relative_to=report_path.parent,
            ),
        },
        "rounds": [
            {
                "round_index": index,
                "round_name": ROUND_NAMES[index - 1],
                "round_kind": result["report"]["round_kind"],
                "removed_object_ids": result["report"]["removed_object_ids"],
                "remaining_object_ids": result["report"]["remaining_object_ids"],
                "status": result["report"]["status"],
                "unresolved_pixels": result["report"]["aggregate_counts"][
                    "unresolved_pixels"
                ],
                "manifest": input_binding(
                    result["manifest_path"], relative_to=report_path.parent
                ),
                "report": input_binding(
                    result["report_path"], relative_to=report_path.parent
                ),
                "receipt": input_binding(
                    result["receipt_path"], relative_to=report_path.parent
                ),
                "output_frame_set_sha256": result["report"][
                    "output_frame_set_sha256"
                ],
            }
            for index, result in enumerate(round_results, start=1)
        ],
        "accepted_with_limitations": accepted_with_limitations,
        "acceptance_status": final_report.get("acceptance_status"),
        "acceptance_scope": final_report.get("acceptance_scope"),
        "limitations": final_report.get("limitations", []),
        "overridden_gates": final_report.get("overridden_gates", []),
        "failed_metrics": final_report.get("failed_metrics", {}),
        "final_output_frame_set_sha256": final_report["output_frame_set_sha256"],
        "gates": {
            "all_inputs_hash_revalidated_after_path_rebase": True,
            "strict_four_round_order_executed": True,
            "fixed_removed_and_remaining_mappings_enforced": True,
            "every_round_unresolved_pixels_zero": True,
            "round2_plus_source_is_previous_composite_hash": True,
            "removed_objects_are_never_rendered_again": True,
            "round4_is_final_background_without_measured_donor": True,
            "round4_structural_background_covers_full_cumulative_mask": True,
            "limited_scope_and_failures_propagated": (
                not accepted_with_limitations
                or (
                    final_report.get("acceptance_scope") == "current_demo_only"
                    and bool(final_report.get("limitations"))
                    and bool(final_report.get("failed_metrics"))
                )
            ),
        },
    }
    if not all(report["gates"].values()):
        raise ValueError("sequence report gates did not all pass")
    write_json(report_path, report)
    receipt = {
        "schema_version": 1,
        "kind": SEQUENCE_RECEIPT_KIND,
        "status": sequence_status,
        "promotion_approved": False,
        "report": report_path.name,
        "report_sha256": sha256_file(report_path),
        "final_round_report": input_binding(
            round_results[-1]["report_path"], relative_to=receipt_path.parent
        ),
        "final_round_receipt": input_binding(
            round_results[-1]["receipt_path"], relative_to=receipt_path.parent
        ),
        "final_output_frame_set_sha256": final_report["output_frame_set_sha256"],
        "accepted_with_limitations": accepted_with_limitations,
        "acceptance_status": report["acceptance_status"],
        "acceptance_scope": report["acceptance_scope"],
        "limitations": report["limitations"],
        "overridden_gates": report["overridden_gates"],
        "failed_metrics": report["failed_metrics"],
        "all_round_manifests_reports_and_receipts_are_hash_bound": True,
    }
    write_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "status": sequence_status,
                "frame_count": len(context["frame_ids"]),
                "report": str(report_path),
            },
            indent=2,
        )
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for round_index in range(1, 4):
        parser.add_argument(
            f"--round{round_index}-cumulative-manifest",
            type=Path,
            required=True,
        )
        parser.add_argument(
            f"--round{round_index}-measured-report",
            type=Path,
            required=True,
        )
        parser.add_argument(
            f"--round{round_index}-measured-receipt",
            type=Path,
            required=True,
        )
    parser.add_argument("--pbr-render-index", type=Path, required=True)
    parser.add_argument("--pbr-batch-receipt", type=Path, required=True)
    parser.add_argument("--structural-background-index", type=Path, required=True)
    parser.add_argument("--structural-background-receipt", type=Path, required=True)
    parser.add_argument("--round4-cumulative-masks", type=Path, required=True)
    parser.add_argument("--path-rebase", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    run_sequence(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
