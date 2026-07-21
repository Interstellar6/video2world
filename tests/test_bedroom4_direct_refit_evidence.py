from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REFIT = ROOT / "examples/bedroom4/completion/direct-trellis2-refit"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_plant_part_audit_fails_closed_before_targeted_pot_masks() -> None:
    manual = _read_json(REFIT / "plant-mask-parts-audit/manual-review.json")
    report = _read_json(REFIT / "plant-mask-parts-audit/review/mask-part-audit.json")

    report_path = REFIT / "plant-mask-parts-audit" / manual["machine_report"]["path"]
    assert _sha256(report_path) == manual["machine_report"]["sha256"]
    assert manual["status"] == "rejected_missing_pot_modal_masks"
    assert manual["promotion_eligible"] is False
    assert manual["union_ready"] is False
    assert report["status"] == "rejected_missing_required_part_mask"

    objects = {item["object_id"]: item for item in report["objects"]}
    assert set(objects) == {"sam3_plant_01", "sam3_plant_02"}
    for item in objects.values():
        assert item["status"] == "rejected_missing_required_part_mask"
        assert item["union_ready"] is False
        assert item["frame_count"] == 25
        assert any("support-surface mask" in reason for reason in item["fail_closed_reasons"])


def test_targeted_potted_plant_masks_are_modal_conditioning_only() -> None:
    manual = _read_json(REFIT / "targeted-pot-segmentation/manual-review.json")
    manifest = _read_json(
        REFIT
        / "targeted-pot-segmentation/materialized/targeted-potted-plant-mask-manifest.json"
    )

    manifest_path = REFIT / "targeted-pot-segmentation" / manual["machine_manifest"]["path"]
    assert _sha256(manifest_path) == manual["machine_manifest"]["sha256"]
    assert manual["status"] == "accepted_for_modal_conditioning_and_scene_fit"
    assert manual["promotion_scope"] == "source-view modal masks only"
    assert manifest["status"] == "passed"

    objects = {item["object_id"]: item for item in manifest["objects"]}
    assert set(objects) == {"sam3_plant_01", "sam3_plant_02"}
    for item in objects.values():
        assert item["status"] == "passed"
        assert item["frame_count"] == 25
        assert len(item["frames"]) == 25

    limitations = "\n".join(manual["limitations"] + manifest["limitations"])
    assert "does not approve any reconstructed 3D asset" in limitations
    assert "do not reveal occluded backs or bottoms" in limitations
