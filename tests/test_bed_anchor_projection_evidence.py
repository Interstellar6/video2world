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
    / "round04_bed"
    / "anchor-projection-supplement"
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


def test_v1_is_rejected_and_v2_minimal_evidence_matches_authoritative_receipt() -> None:
    rejection = read_json(ROOT / "v1-rejected" / "rejection_receipt.json")
    assert rejection["status"] == "rejected"
    assert rejection["downstream_consumption_allowed"] is False
    assert rejection["failed_gates"]["visible_raw_is_subset_of_final_mask"] is False
    assert rejection["original_run_receipt_sha256"] == sha256_file(
        ROOT / "v1-rejected" / "receipt.json"
    )
    assert rejection["contact_sheet_sha256"] == sha256_file(
        ROOT / "v1-rejected" / "object_anchor_projection_contact_sheet.png"
    )

    v2 = ROOT / "v2-authoritative"
    manifest_path = v2 / "object_anchor_supplement_manifest.json"
    manifest = read_json(manifest_path)
    receipt = read_json(v2 / "receipt.json")
    assert receipt["manifest_sha256"] == sha256_file(manifest_path)
    assert receipt["mask_set_sha256"] == manifest["mask_set_sha256"]
    assert receipt["contact_sheet_sha256"] == sha256_file(
        v2 / "object_anchor_projection_contact_sheet.png"
    )
    assert manifest["gates"]["all_visible_raw_masks_are_subsets_of_final_masks"] is True
    assert (
        manifest["gates"][
            "all_final_masks_are_subsets_of_depth_consistent_anchor_envelopes"
        ]
        is True
    )
    records = {record["frame_id"]: record for record in manifest["frame_records"]}
    for frame_id in ("000048", "000064", "000067", "000072"):
        mask_path = v2 / "masks" / f"{frame_id}.png"
        assert sha256_file(mask_path) == records[frame_id]["union_mask_sha256"]
