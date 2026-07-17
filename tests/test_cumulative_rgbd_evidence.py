from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = (
    Path(__file__).parents[1]
    / "examples"
    / "bedroom4"
    / "completion"
    / "layered-peel"
    / "cumulative-rgbd-reprojection"
)
ROUND_DIRS = (
    "round01_front_pillow",
    "round02_left_pillow",
    "round03_right_pillow",
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_cumulative_rgbd_evidence_receipts_and_lineage_are_self_consistent() -> None:
    summary = read_json(ROOT / "summary.json")
    assert summary["status"] == "technical_passed_not_promotable"
    assert summary["promotion_approved"] is False
    assert summary["method"]["generated_or_prefilled_rgb_as_donor"] is False
    assert summary["method"]["unresolved_residual_as_donor"] is False

    previous_manifest_sha: str | None = None
    previous_report_sha: str | None = None
    for round_dir_name, summary_round in zip(ROUND_DIRS, summary["rounds"], strict=True):
        round_root = ROOT / round_dir_name
        manifest_path = round_root / "manifest" / "cumulative_removal_manifest.json"
        index_path = round_root / "manifest" / "donor_exclusion_index.json"
        manifest_receipt = read_json(round_root / "manifest" / "receipt.json")
        report_path = round_root / "output" / "multiview_prefill_report.json"
        contact_path = round_root / "output" / "multiview_prefill_contact_sheet.png"
        output_receipt = read_json(round_root / "output" / "multiview_prefill_receipt.json")

        manifest_sha = sha256_file(manifest_path)
        report_sha = sha256_file(report_path)
        assert manifest_receipt["manifest_sha256"] == manifest_sha
        assert manifest_receipt["donor_exclusion_index_sha256"] == sha256_file(index_path)
        assert output_receipt["report_sha256"] == report_sha
        assert output_receipt["contact_sheet_sha256"] == sha256_file(contact_path)
        assert output_receipt["fused_metric_depth_artifacts_are_hashed_in_report"] is True
        assert summary_round["manifest_sha256"] == manifest_sha
        assert summary_round["report_sha256"] == report_sha
        assert summary_round["contact_sheet_sha256"] == sha256_file(contact_path)

        manifest = read_json(manifest_path)
        report = read_json(report_path)
        assert manifest["round_index"] == summary_round["round_index"]
        assert manifest["removed_object_ids"] == summary_round["removed_object_ids"]
        assert report["input_manifest_sha256"] == manifest_sha
        assert report["cumulative_removal_contract"]["removed_object_ids"] == (
            summary_round["removed_object_ids"]
        )
        assert report["gates"]["outside_removal_mask_rgb_exact"] is True
        assert report["gates"]["all_residual_masks_subset_of_removal_masks"] is True
        assert report["gates"]["fused_measured_metric_depth_materialized"] is True
        assert report["promotion_approved"] is False
        assert report["pixel_provenance"]["generated_pixels"] == 0
        assert report["pixel_provenance"]["propainter_pixels"] == 0
        assert output_receipt["measured_provenance_excludes_generated_pixels"] is True
        assert output_receipt["measured_provenance_excludes_propainter_pixels"] is True
        for key in (
            "cumulative_removal_pixels",
            "measured_multiview_pixels",
            "measured_multiview_fraction",
            "unresolved_unobserved_pixels",
            "unresolved_unobserved_fraction",
        ):
            assert report["pixel_provenance"][key] == summary_round[key]

        lineage = manifest["lineage"]
        assert lineage["previous_cumulative_manifest_sha256"] == previous_manifest_sha
        assert lineage["previous_prefill_report_sha256"] == previous_report_sha
        for record in manifest["frame_records"]:
            mask_path = manifest_path.parent / record["union_mask"]
            assert sha256_file(mask_path) == record["union_mask_sha256"]
            assert record["donor_exclusion_mask_sha256"] == record["union_mask_sha256"]
            assert record["removed_object_ids"] == summary_round["removed_object_ids"]

        for record in report["frame_records"]:
            depth_path = Path(record["measured_depth"])
            assert depth_path.name == f"{record['sequence_index']:04d}.npy"
            assert record["measured_depth_role"] == "fused_multiview_measured_depth"
            assert record["measured_depth_semantics"] == "target_camera_positive_z_axis"
            assert record["measured_depth_valid_pixels"] == record["covered_pixels"]

        previous_manifest_sha = manifest_sha
        previous_report_sha = report_sha
