#!/usr/bin/env python3
"""Upgrade cumulative RGB-D peel reports to an explicit no-generation contract.

The upgrade is intentionally fail-closed. It verifies every hashed frame, mask,
depth, support, donor, and lineage asset before preparing any metadata writes.
It then recursively rebinds manifest/report hashes from the first peel round to
the last without changing image, mask, or depth payloads.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str, *, relative_to: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


class AssetVerifier:
    def __init__(self) -> None:
        self._hashes: dict[Path, str] = {}
        self.total_bytes = 0

    @property
    def file_count(self) -> int:
        return len(self._hashes)

    def verify(self, path: Path, expected_sha: Any, label: str) -> None:
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise ValueError(f"{label} has no valid SHA-256")
        if not path.is_file():
            raise FileNotFoundError(f"{label}: {path}")
        actual = self._hashes.get(path)
        if actual is None:
            actual = sha256_file(path)
            self._hashes[path] = actual
            self.total_bytes += path.stat().st_size
        if actual != expected_sha:
            raise ValueError(f"{label} SHA-256 mismatch: {path}")


def verify_path_field(
    verifier: AssetVerifier,
    record: dict[str, Any],
    path_key: str,
    sha_key: str,
    *,
    relative_to: Path,
    label: str,
) -> None:
    value = record.get(path_key)
    if not isinstance(value, str):
        raise ValueError(f"{label} has no {path_key}")
    verifier.verify(
        resolve_path(value, relative_to=relative_to), record.get(sha_key), label
    )


def verify_manifest_assets(
    verifier: AssetVerifier, manifest: dict[str, Any], manifest_path: Path
) -> None:
    verify_path_field(
        verifier,
        manifest,
        "current_object_manifest",
        "current_object_manifest_sha256",
        relative_to=manifest_path.parent,
        label="current object manifest",
    )
    donor_contract = require_dict(manifest.get("donor_contract"), "donor contract")
    donor_assets = require_dict(donor_contract.get("assets"), "donor assets")
    for frame_id, raw_asset in donor_assets.items():
        asset = require_dict(raw_asset, f"donor asset {frame_id}")
        verify_path_field(
            verifier,
            asset,
            "path",
            "sha256",
            relative_to=manifest_path.parent,
            label=f"original observed donor {frame_id}",
        )

    records = require_list(manifest.get("frame_records"), "manifest frame_records")
    if not records:
        raise ValueError("manifest frame_records must not be empty")
    for raw_record in records:
        record = require_dict(raw_record, "manifest frame record")
        frame_id = record.get("frame_id")
        verify_path_field(
            verifier,
            record,
            "source_frame",
            "source_frame_sha256",
            relative_to=manifest_path.parent,
            label=f"manifest source {frame_id}",
        )
        for path_key, sha_key in (
            ("union_mask", "union_mask_sha256"),
            ("donor_exclusion_mask", "donor_exclusion_mask_sha256"),
        ):
            verify_path_field(
                verifier,
                record,
                path_key,
                sha_key,
                relative_to=manifest_path.parent,
                label=f"manifest {path_key} {frame_id}",
            )
        for index, raw_component in enumerate(
            require_list(record.get("mask_components"), "mask_components")
        ):
            component = require_dict(raw_component, "mask component")
            verify_path_field(
                verifier,
                component,
                "path",
                "sha256",
                relative_to=manifest_path.parent,
                label=f"mask component {frame_id}:{index}",
            )


def verify_report_assets(
    verifier: AssetVerifier, report: dict[str, Any], report_path: Path
) -> None:
    for path_key, sha_key, label in (
        ("input_manifest", "input_manifest_sha256", "report input manifest"),
        ("camera_info", "camera_info_sha256", "camera info"),
        ("donor_mask_index", "donor_mask_index_sha256", "donor mask index"),
    ):
        verify_path_field(
            verifier,
            report,
            path_key,
            sha_key,
            relative_to=report_path.parent,
            label=label,
        )

    donor_assets = require_dict(report.get("donor_assets"), "report donor assets")
    for frame_id, raw_asset in donor_assets.items():
        asset = require_dict(raw_asset, f"report donor asset {frame_id}")
        for path_key, sha_key in (("frame", "frame_sha256"), ("depth", "depth_sha256")):
            verify_path_field(
                verifier,
                asset,
                path_key,
                sha_key,
                relative_to=report_path.parent,
                label=f"report donor {path_key} {frame_id}",
            )
        for index, raw_mask in enumerate(
            require_list(asset.get("exclusion_masks"), "donor exclusion_masks")
        ):
            mask = require_dict(raw_mask, "donor exclusion mask")
            verify_path_field(
                verifier,
                mask,
                "path",
                "sha256",
                relative_to=report_path.parent,
                label=f"report donor exclusion {frame_id}:{index}",
            )

    records = require_list(report.get("frame_records"), "report frame_records")
    if not records:
        raise ValueError("report frame_records must not be empty")
    fields = (
        ("source_frame", "source_frame_sha256"),
        ("removal_mask", "removal_mask_sha256"),
        ("prefill_frame", "prefill_frame_sha256"),
        ("residual_mask", "residual_mask_sha256"),
        ("support_visualization", "support_visualization_sha256"),
        ("measured_depth", "measured_depth_sha256"),
    )
    for raw_record in records:
        record = require_dict(raw_record, "report frame record")
        frame_id = record.get("frame_id")
        for path_key, sha_key in fields:
            verify_path_field(
                verifier,
                record,
                path_key,
                sha_key,
                relative_to=report_path.parent,
                label=f"report {path_key} {frame_id}",
            )


def round_paths(root: Path) -> list[dict[str, Path]]:
    result: list[dict[str, Path]] = []
    for manifest_path in sorted(root.glob("round*/manifest/cumulative_removal_manifest.json")):
        round_root = manifest_path.parents[1]
        result.append(
            {
                "round_root": round_root,
                "manifest": manifest_path,
                "donor_index": manifest_path.parent / "donor_exclusion_index.json",
                "manifest_receipt": manifest_path.parent / "receipt.json",
                "report": round_root / "output" / "multiview_prefill_report.json",
                "report_receipt": round_root / "output" / "multiview_prefill_receipt.json",
                "contact_sheet": round_root / "output" / "multiview_prefill_contact_sheet.png",
            }
        )
    if not result:
        raise ValueError(f"no cumulative peel rounds found under {root}")
    return result


def validate_existing(
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], AssetVerifier]:
    summary_path = root / "summary.json"
    summary = read_json(summary_path)
    summary_rounds = require_list(summary.get("rounds"), "summary rounds")
    paths = round_paths(root)
    if len(paths) != len(summary_rounds):
        raise ValueError("summary and filesystem have different round counts")

    verifier = AssetVerifier()
    states: list[dict[str, Any]] = []
    previous_manifest_sha: str | None = None
    previous_report_sha: str | None = None
    for index, (round_path, raw_summary_round) in enumerate(
        zip(paths, summary_rounds),  # noqa: B905 - lengths checked; mil8 uses Python 3.9
        start=1,
    ):
        summary_round = require_dict(raw_summary_round, f"summary round {index}")
        manifest_path = round_path["manifest"]
        report_path = round_path["report"]
        manifest = read_json(manifest_path)
        report = read_json(report_path)
        manifest_receipt = read_json(round_path["manifest_receipt"])
        report_receipt = read_json(round_path["report_receipt"])
        manifest_sha = sha256_file(manifest_path)
        report_sha = sha256_file(report_path)
        if manifest.get("round_index") != index or report.get(
            "cumulative_removal_contract", {}
        ).get("round_index") != index:
            raise ValueError(f"round {index} index is not contiguous")
        if summary_round.get("round_index") != index:
            raise ValueError(f"summary round {index} index mismatch")
        if manifest_receipt.get("manifest_sha256") != manifest_sha:
            raise ValueError(f"round {index} manifest receipt mismatch")
        if summary_round.get("manifest_sha256") != manifest_sha:
            raise ValueError(f"round {index} summary manifest mismatch")
        donor_index_sha = sha256_file(round_path["donor_index"])
        if manifest_receipt.get("donor_exclusion_index_sha256") != donor_index_sha:
            raise ValueError(f"round {index} donor index receipt mismatch")
        if report_receipt.get("report_sha256") != report_sha:
            raise ValueError(f"round {index} report receipt mismatch")
        if summary_round.get("report_sha256") != report_sha:
            raise ValueError(f"round {index} summary report mismatch")
        contact_sha = sha256_file(round_path["contact_sheet"])
        if report_receipt.get("contact_sheet_sha256") != contact_sha:
            raise ValueError(f"round {index} contact sheet receipt mismatch")
        if summary_round.get("contact_sheet_sha256") != contact_sha:
            raise ValueError(f"round {index} summary contact sheet mismatch")
        lineage = require_dict(manifest.get("lineage"), f"round {index} lineage")
        if lineage.get("previous_cumulative_manifest_sha256") != previous_manifest_sha:
            raise ValueError(f"round {index} previous manifest lineage mismatch")
        if lineage.get("previous_prefill_report_sha256") != previous_report_sha:
            raise ValueError(f"round {index} previous report lineage mismatch")
        if report.get("input_manifest_sha256") != manifest_sha:
            raise ValueError(f"round {index} report input manifest mismatch")
        cumulative = require_dict(
            report.get("cumulative_removal_contract"), f"round {index} cumulative contract"
        )
        if cumulative.get("manifest_sha256") != manifest_sha:
            raise ValueError(f"round {index} report cumulative manifest mismatch")
        if cumulative.get("lineage") != lineage:
            raise ValueError(f"round {index} report lineage copy mismatch")
        provenance = require_dict(
            report.get("pixel_provenance"), f"round {index} pixel provenance"
        )
        if provenance.get("generated_pixels") != 0:
            raise ValueError(f"round {index} generated_pixels must already be zero")
        if provenance.get("propainter_pixels") not in (None, 0):
            raise ValueError(f"round {index} propainter_pixels is non-zero")
        if (
            provenance.get("unresolved_pixels_are_not_valid_donor_or_geometry_evidence")
            is not True
        ):
            raise ValueError(f"round {index} unresolved pixels are not fail-closed")

        verify_manifest_assets(verifier, manifest, manifest_path)
        verify_report_assets(verifier, report, report_path)
        verifier.verify(
            round_path["contact_sheet"], contact_sha, f"round {index} contact sheet"
        )
        states.append(
            {
                "paths": round_path,
                "manifest": manifest,
                "manifest_receipt": manifest_receipt,
                "report": report,
                "report_receipt": report_receipt,
                "before_manifest_sha256": manifest_sha,
                "before_report_sha256": report_sha,
            }
        )
        previous_manifest_sha = manifest_sha
        previous_report_sha = report_sha
    return summary, states, verifier


def prepare_upgrade(
    root: Path, summary: dict[str, Any], states: list[dict[str, Any]], verifier: AssetVerifier
) -> tuple[dict[Path, bytes], dict[str, Any]]:
    writes: dict[Path, bytes] = {}
    output_rounds: list[dict[str, Any]] = []
    previous_manifest_sha: str | None = None
    previous_report_sha: str | None = None
    previous_manifest_path: Path | None = None
    previous_report_path: Path | None = None
    updated_summary = copy.deepcopy(summary)
    updated_summary_rounds = require_list(updated_summary.get("rounds"), "summary rounds")

    if len(states) != len(updated_summary_rounds):
        raise ValueError("validated state and summary have different round counts")
    for index, (state, raw_summary_round) in enumerate(
        zip(states, updated_summary_rounds),  # noqa: B905 - lengths checked; Python 3.9
        start=1,
    ):
        paths = state["paths"]
        manifest = copy.deepcopy(state["manifest"])
        lineage = require_dict(manifest.get("lineage"), "manifest lineage")
        lineage["previous_cumulative_manifest"] = (
            str(previous_manifest_path) if previous_manifest_path else None
        )
        lineage["previous_cumulative_manifest_sha256"] = previous_manifest_sha
        lineage["previous_prefill_report"] = (
            str(previous_report_path) if previous_report_path else None
        )
        lineage["previous_prefill_report_sha256"] = previous_report_sha
        gates = require_dict(manifest.get("gates"), "manifest gates")
        gates["previous_prefill_excludes_generated_and_propainter_pixels"] = True
        donor_contract = require_dict(manifest.get("donor_contract"), "donor contract")
        donor_contract["propainter_rgb_as_donor"] = False
        provenance_classes = require_dict(
            manifest.get("pixel_provenance_classes"), "manifest provenance classes"
        )
        provenance_classes["propainter"] = "forbidden in this measurement stage"
        manifest_payload = json_bytes(manifest)
        manifest_sha = sha256_bytes(manifest_payload)
        writes[paths["manifest"]] = manifest_payload

        manifest_receipt = copy.deepcopy(state["manifest_receipt"])
        manifest_receipt["manifest_sha256"] = manifest_sha
        manifest_receipt["metadata_only_provenance_upgrade"] = True
        writes[paths["manifest_receipt"]] = json_bytes(manifest_receipt)

        report = copy.deepcopy(state["report"])
        report["input_manifest_sha256"] = manifest_sha
        cumulative = require_dict(
            report.get("cumulative_removal_contract"), "report cumulative contract"
        )
        cumulative["manifest_sha256"] = manifest_sha
        cumulative["lineage"] = copy.deepcopy(lineage)
        provenance = require_dict(report.get("pixel_provenance"), "report provenance")
        provenance["generated_pixels"] = 0
        provenance["propainter_pixels"] = 0
        report_gates = require_dict(report.get("gates"), "report gates")
        report_gates["no_generated_or_propainter_pixels_as_measured_evidence"] = True
        report_payload = json_bytes(report)
        report_sha = sha256_bytes(report_payload)
        writes[paths["report"]] = report_payload

        report_receipt = copy.deepcopy(state["report_receipt"])
        report_receipt["report_sha256"] = report_sha
        report_receipt["measured_provenance_excludes_generated_pixels"] = True
        report_receipt["measured_provenance_excludes_propainter_pixels"] = True
        writes[paths["report_receipt"]] = json_bytes(report_receipt)

        summary_round = require_dict(raw_summary_round, "summary round")
        summary_round["manifest_sha256"] = manifest_sha
        summary_round["report_sha256"] = report_sha
        output_rounds.append(
            {
                "round_index": index,
                "before_manifest_sha256": state["before_manifest_sha256"],
                "after_manifest_sha256": manifest_sha,
                "before_report_sha256": state["before_report_sha256"],
                "after_report_sha256": report_sha,
                "pixel_payloads_modified": False,
            }
        )
        previous_manifest_sha = manifest_sha
        previous_report_sha = report_sha
        previous_manifest_path = paths["manifest"]
        previous_report_path = paths["report"]

    method = require_dict(updated_summary.get("method"), "summary method")
    method["propainter_rgb_as_donor"] = False
    summary_provenance = require_dict(
        updated_summary.get("pixel_provenance"), "summary pixel provenance"
    )
    summary_provenance["propainter"] = "zero pixels in this measurement stage"
    summary_payload = json_bytes(updated_summary)
    summary_path = root / "summary.json"
    writes[summary_path] = summary_payload
    receipt = {
        "schema_version": 1,
        "kind": "video2world.cumulative_rgbd_provenance_upgrade_receipt",
        "status": "technical_passed_metadata_rebound",
        "created_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 - mil8 Python 3.10
        "root": str(root),
        "backup_root": str(root / "metadata-before-provenance-upgrade"),
        "contract": {
            "generated_pixels": 0,
            "propainter_pixels": 0,
            "unresolved_pixels_are_not_measurement_or_geometry_evidence": True,
            "pixel_mask_depth_and_support_payloads_modified": False,
            "recursive_front_to_back_hash_lineage_rebound": True,
        },
        "verified_assets": {
            "unique_file_count": verifier.file_count,
            "unique_file_bytes": verifier.total_bytes,
            "all_preexisting_sha256_bindings_passed_before_write": True,
        },
        "rounds": output_rounds,
        "summary": {
            "path": str(summary_path),
            "sha256": sha256_bytes(summary_payload),
        },
    }
    writes[root / "provenance_upgrade_receipt.json"] = json_bytes(receipt)
    return writes, receipt


def apply_writes(root: Path, writes: dict[Path, bytes]) -> Path:
    backup_root = root / "metadata-before-provenance-upgrade"
    if backup_root.exists():
        raise FileExistsError(f"metadata backup already exists: {backup_root}")
    backup_root.mkdir()
    for path in writes:
        if path.is_file():
            relative = path.relative_to(root)
            destination = backup_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)

    staged: list[tuple[Path, Path]] = []
    try:
        for path, payload in writes.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_value = tempfile.mkstemp(
                dir=path.parent, prefix=f".{path.name}.upgrade-"
            )
            temporary = Path(temporary_value)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            staged.append((temporary, path))
        for temporary, destination in staged:
            os.replace(temporary, destination)
    finally:
        for temporary, _ in staged:
            if temporary.exists():
                temporary.unlink()
    return backup_root


def upgrade(root: Path, *, check_only: bool) -> dict[str, Any]:
    root = root.expanduser().resolve()
    summary, states, verifier = validate_existing(root)
    writes, receipt = prepare_upgrade(root, summary, states, verifier)
    if check_only:
        receipt["status"] = "technical_passed_dry_run"
        receipt["would_write_file_count"] = len(writes)
        return receipt
    apply_writes(root, writes)
    validate_existing(root)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print(json.dumps(upgrade(args.root, check_only=args.check_only), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
