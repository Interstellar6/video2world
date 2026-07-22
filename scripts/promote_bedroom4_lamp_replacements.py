#!/usr/bin/env python3
"""Promote the browser-verified Bedroom 4 lamp replacement candidate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORLD = ROOT / "web/public/worlds/bedroom4"
ARTIFACT_ROOT = ROOT / "examples/bedroom4/completion/lamps_trellis_seed43"
LAMP_IDS = ("sam3_lamp_01", "sam3_lamp_02")
UNIFIED_IDS = (
    "sam3_pillow_front",
    "sam3_pillow_left",
    "sam3_pillow_right",
    "sam3_bed_01",
    *LAMP_IDS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, default=WORLD / "manifest.lamp-candidate.json")
    parser.add_argument(
        "--candidate-browser-report",
        type=Path,
        default=ARTIFACT_ROOT / "browser_qa/browser_qa_report.json",
    )
    parser.add_argument("--output", type=Path, default=WORLD / "manifest.json")
    parser.add_argument(
        "--production-manifest",
        type=Path,
        default=ROOT / "examples/bedroom4/manifest.production.json",
    )
    parser.add_argument(
        "--final-browser-report",
        type=Path,
        default=ARTIFACT_ROOT / "browser_qa/browser_qa_final_report.json",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=ARTIFACT_ROOT / "promotion_receipt.json",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def canonical_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def descriptor(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "size": len(data),
        "sha256": sha256_bytes(data),
    }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def build_promoted_manifest(
    candidate: dict[str, Any],
    candidate_descriptor: dict[str, Any],
    browser_report: dict[str, Any],
    browser_report_descriptor: dict[str, Any],
) -> dict[str, Any]:
    manifest = copy.deepcopy(candidate)
    objects = manifest.get("interactiveObjects") or []
    by_id = {item.get("id"): item for item in objects}
    require(len(objects) == 10, f"Expected 10 interactive objects, found {len(objects)}")
    require(all(object_id in by_id for object_id in UNIFIED_IDS), "Unified PBR objects are incomplete")

    report_manifest = browser_report.get("manifest") or {}
    require(browser_report.get("status") == "passed", "Candidate browser QA did not pass")
    require(browser_report.get("failures") == [], "Candidate browser QA contains failures")
    require(browser_report.get("normalVisualMode") is True, "Candidate QA did not use normal visual mode")
    require(
        report_manifest.get("sha256") == candidate_descriptor["sha256"],
        "Candidate bytes do not match the browser QA report",
    )
    require(
        (browser_report.get("contract") or {}).get("objectCount") == 10,
        "Browser QA did not validate all 10 objects",
    )

    manifest["version"] = "bedroom4-production-unified-pbr-pillows-bed-lamps-20260722"
    manifest["promotion"] = {
        "status": "passed_current_demo_only",
        "promotedAt": browser_report.get("createdAt"),
        "candidateManifestSha256": candidate_descriptor["sha256"],
        "candidateBrowserQaReportSha256": browser_report_descriptor["sha256"],
        "normalVisualMode": True,
        "objectCount": 10,
        "unifiedPbrObjectIds": list(UNIFIED_IDS),
        "stableBaselineAliasUnchanged": "web-demo-baseline-stable@252a85c",
    }

    candidate_build = manifest.setdefault("candidateBuild", {})
    candidate_build["status"] = "promoted_current_demo_only_with_unified_pbr_lamps"
    candidate_build["completionClaim"] = "six_unified_pbr_objects_plus_four_legacy_gaussian_objects"
    lamp_build = candidate_build.setdefault("lampUnifiedPbrObjects", {})
    lamp_build["browserSceneQa"] = "passed"
    lamp_build["browserQaReportSha256"] = browser_report_descriptor["sha256"]
    lamp_build["status"] = "browser_qa_passed_promoted_current_demo_only"

    for object_id in UNIFIED_IDS:
        gate = by_id[object_id].setdefault("collision", {}).setdefault("gate", {})
        gate["status"] = "passed"
        gate["surfaceCollision"] = "passed"
        gate["browserSceneQa"] = "passed"
        gate["browserQaReportSha256"] = browser_report_descriptor["sha256"]
        if "candidateBrowserQa" in gate:
            gate["candidateBrowserQa"] = "passed_promoted_manifest_recheck_pending"

    collision_gate = manifest.setdefault("collisionWorld", {}).setdefault("gate", {})
    collision_gate["lampBrowserQa"] = "passed"
    collision_gate["browserQaReportSha256"] = browser_report_descriptor["sha256"]
    collider_asset = manifest.setdefault("assets", {}).get("colliderStaticCarved") or {}
    collider_asset["promotionApproved"] = True

    production_build = manifest.setdefault("productionBuild", {})
    production_build["unifiedPbrSixObjectIntegration"] = {
        "status": "browser_qa_passed_promoted_current_demo_only",
        "objectIds": list(UNIFIED_IDS),
        "representation": "one_pbr_glb_for_visual_selection_logic_and_surface_collision",
        "browserQaReportSha256": browser_report_descriptor["sha256"],
    }
    pillow_integration = production_build.get("candidatePillowIntegration")
    if isinstance(pillow_integration, dict):
        pillow_integration["rootProxyBrowserQa"] = "passed_promoted_manifest_recheck_pending"
    return manifest


def main() -> None:
    args = parse_args()
    candidate = read_json(args.candidate)
    browser_report = read_json(args.candidate_browser_report)
    candidate_descriptor = descriptor(args.candidate)
    browser_report_descriptor = descriptor(args.candidate_browser_report)
    promoted = build_promoted_manifest(
        candidate,
        candidate_descriptor,
        browser_report,
        browser_report_descriptor,
    )
    promoted_data = canonical_bytes(promoted)
    atomic_write(args.output, promoted_data)
    atomic_write(args.production_manifest, promoted_data)
    promoted_sha256 = sha256_bytes(promoted_data)

    final_report_descriptor = None
    status = "promoted_final_browser_qa_pending"
    if args.final_browser_report.exists():
        final_report = read_json(args.final_browser_report)
        final_report_descriptor = descriptor(args.final_browser_report)
        require(final_report.get("status") == "passed", "Final browser QA did not pass")
        require(final_report.get("failures") == [], "Final browser QA contains failures")
        require(
            (final_report.get("manifest") or {}).get("sha256") == promoted_sha256,
            "Final browser QA does not bind the promoted manifest bytes",
        )
        status = "passed"

    receipt = {
        "schemaVersion": 1,
        "kind": "video2world.bedroom4_lamp_replacement_promotion",
        "createdAt": (
            read_json(args.final_browser_report).get("createdAt")
            if final_report_descriptor
            else browser_report.get("createdAt")
        ),
        "status": status,
        "candidateManifest": candidate_descriptor,
        "candidateBrowserQa": browser_report_descriptor,
        "promotedManifest": {
            "path": str(args.output.resolve()),
            "size": len(promoted_data),
            "sha256": promoted_sha256,
        },
        "productionManifest": {
            "path": str(args.production_manifest.resolve()),
            "size": len(promoted_data),
            "sha256": promoted_sha256,
        },
        "finalBrowserQa": final_report_descriptor,
        "stableBaselineAliasChanged": False,
    }
    atomic_write(args.receipt, canonical_bytes(receipt))
    print(json.dumps(receipt, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
