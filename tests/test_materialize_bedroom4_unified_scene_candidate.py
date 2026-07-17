from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.materialize_bedroom4_unified_scene_candidate import (
    ADOPTED_IDS,
    BASE_MANIFEST_SHA256,
    JOINT_REVIEW_SHA256,
    PAIRWISE_REVIEW_SHA256,
    PILLOW_IDS,
    STABLE_MANIFEST_SHA256,
    SUPPORT_RECEIPT_SHA256,
    _assert_candidate_output_scope,
    build_candidate_manifest,
    load_verified_objects,
)
from video2world.hashing import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = PROJECT_ROOT / "examples/bedroom4/manifests"
BASE_MANIFEST = MANIFESTS / "bedroom4.unified-pillow.candidate.web.json"
CANDIDATE_MANIFEST = MANIFESTS / "bedroom4.unified-pbr-objects.candidate.web.json"
CANDIDATE_REPORT = MANIFESTS / "bedroom4.unified-pbr-objects.candidate.report.json"
FINALIZATION_RECEIPT = (
    PROJECT_ROOT
    / "web/public/worlds/bedroom4/qa/strict-clean-scene-finalization-receipt.json"
)


@pytest.fixture(scope="module")
def verified_objects():
    return load_verified_objects(PROJECT_ROOT)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_exact_scene_fit_and_joint_receipts_bind_all_four_glbs(verified_objects) -> None:
    objects, evidence = verified_objects
    assert tuple(objects) == ADOPTED_IDS
    assert evidence == {
        "pairwise_review_sha256": PAIRWISE_REVIEW_SHA256,
        "joint_review_sha256": JOINT_REVIEW_SHA256,
        "support_adjustment_receipt_sha256": SUPPORT_RECEIPT_SHA256,
    }
    for object_id, item in objects.items():
        assert sha256_file(item.mesh_path)[0] == item.mesh.sha256
        assert sha256_file(item.scene_fit_path)[0] == item.spec.scene_fit_sha256
        assert item.mesh.vertices == item.spec.vertices
        assert item.mesh.faces == item.spec.faces
        assert item.mesh.watertight is item.spec.watertight
        assert item.mesh.winding_consistent is True
        assert item.mesh.nondegenerate is True
        assert item.spec.topology == (
            "closed_volume" if object_id == "sam3_bed_01" else "surface_bvh"
        )


def test_baked_pillows_use_pivots_and_unbaked_bed_uses_full_matrix(
    verified_objects,
) -> None:
    objects, _ = verified_objects
    for object_id in ADOPTED_IDS[:3]:
        item = objects[object_id]
        assert item.spec.asset_coordinates_baked is True
        assert item.matrix_row_major is None
        assert len(item.pivot) == 3
    bed = objects["sam3_bed_01"]
    assert bed.spec.asset_coordinates_baked is False
    assert bed.matrix_row_major is not None
    assert bed.matrix_row_major[3] == [0.0, 0.0, 0.0, 1.0]
    assert bed.pivot == [bed.matrix_row_major[axis][3] for axis in range(3)]


def test_generated_candidate_matches_generic_builder_and_has_no_proxy_split(
    verified_objects,
) -> None:
    objects, evidence = verified_objects
    expected = build_candidate_manifest(
        read_json(BASE_MANIFEST),
        objects,
        evidence,
        base_sha256=BASE_MANIFEST_SHA256,
    )
    manifest = read_json(CANDIDATE_MANIFEST)
    assert manifest == expected
    adopted = {
        item["id"]: item for item in manifest["interactiveObjects"] if item["id"] in ADOPTED_IDS
    }
    assert tuple(adopted) == ADOPTED_IDS
    for item in adopted.values():
        assert item["collision"]["mode"] == "unified-glb"
        assert item["collision"]["characterCollision"] is True
        assert item["interaction"] == {
            "kind": "spin",
            "degrees": 360,
            "durationMs": 1250,
            "drag": "horizontal_yaw",
        }
        assert item["independentlyMovable"] is True
        assert "visual" not in item
        assert "renderAsset" not in item
        assert "colliderProxy" not in item
        assert "renderAsset" not in item["collision"]
    for object_id in PILLOW_IDS:
        pillow = adopted[object_id]
        assert pillow["semanticGranularity"] == "independent_child_asset"
        assert pillow["parentObjectId"] == "sam3_bed_01"
        assert pillow["movesWithParent"] is True
        assert pillow["childObjectIds"] == []
        assert pillow["independentlyMovable"] is True
    bed = adopted["sam3_bed_01"]
    assert bed["semanticGranularity"] == "independent_root_asset"
    assert bed["parentObjectId"] is None
    assert bed["movesWithParent"] is False
    assert bed["childObjectIds"] == list(PILLOW_IDS)
    assert manifest["initialState"]["cameraFocusObjectId"] == "sam3_bed_01"
    assert manifest["candidateBuild"]["unifiedPbrObjects"]["hierarchy"] == {
        "parent": "sam3_bed_01",
        "children": list(PILLOW_IDS),
        "parentMotionCarriesChildren": True,
        "childrenRemainIndependentlyMovable": True,
    }
    assert manifest["candidateBuild"]["cleanScene"]["status"] == "clean_scene_pending"
    assert manifest["candidateBuild"]["promotionAllowed"] is False


def test_report_and_public_manifest_boundaries_are_explicit() -> None:
    report = read_json(CANDIDATE_REPORT)
    assert report["status"] == "candidate_materialized_clean_scene_pending"
    assert report["promotion_allowed"] is False
    assert report["public_manifests"] == {
        "mutated": False,
        "promoted_manifest_sha256": BASE_MANIFEST_SHA256,
        "stable_manifest_sha256": STABLE_MANIFEST_SHA256,
    }
    assert sha256_file(CANDIDATE_MANIFEST)[0] == report["output_manifest"]["sha256"]
    if FINALIZATION_RECEIPT.is_file():
        finalization = read_json(FINALIZATION_RECEIPT)
        assert finalization["status"] == "promoted_current_demo_only"
        assert finalization["promotionAllowed"] is True
        assert finalization["manifest"]["exactQaTestedBytesMovedAtomically"] is True
        assert sha256_file(PROJECT_ROOT / "web/public/worlds/bedroom4/manifest.json")[0] == (
            finalization["manifest"]["sha256"]
        )
        assert sha256_file(
            PROJECT_ROOT
            / "web/public/worlds/bedroom4/manifest.web-demo-baseline-stable.json"
        )[0] == STABLE_MANIFEST_SHA256
        assert finalization["stableAlias"]["sha256"] == STABLE_MANIFEST_SHA256


def test_materializer_refuses_outputs_outside_candidate_manifest_directory(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="candidate manifest must stay"):
        _assert_candidate_output_scope(
            PROJECT_ROOT,
            tmp_path / "manifest.json",
            MANIFESTS / "allowed.report.json",
        )
