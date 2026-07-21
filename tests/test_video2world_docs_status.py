from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/video2world/project-docs"


def test_pipeline_docs_keep_corrected_clean_plate_status_fail_closed() -> None:
    pipeline = (DOCS / "pipeline.md").read_text(encoding="utf-8")
    completion = (DOCS / "completion.md").read_text(encoding="utf-8")

    for document in (pipeline, completion):
        assert "R1 尚未 accepted" in document
        assert "corrected R2-R4" in document
        assert "fresh DA3/PGSR/TSDF" in document
        assert "live promotion 均未运行" in document
        assert "archived current-demo-only" in document
        assert 'acceptance_scope="corrected_clean_plate_next_round_source_only"' in document
        assert 'lineage_scope="corrected_clean_plate_round_source"' in document
        assert "corrected_full_pipeline=false" in document
        assert "canonical_promotion_approved=false" in document
        assert "不能跨角色" in document
        assert "target_id" in document

    getting_started = (DOCS / "getting-started.md").read_text(encoding="utf-8")
    assert "provider receipt 本身也会被校验" in getting_started
    assert "越权 next-round source" in getting_started
    assert "provider_receipt.inputs" in getting_started
    assert "provider_receipt.outputs" in getting_started
    assert "JSON/PLY" in getting_started
    assert "object_probability" in getting_started
    assert "objects[*].id" in getting_started
    assert "clean_plate_manifest.final_clean_plate" in getting_started
    assert "不能复用原始" in getting_started
    assert "hash 或路径" in getting_started
    assert "terminal report 必须显式绑定 clean_scene_gaussian/clean_scene_mesh" in getting_started
    assert "uri/hash/size" in getting_started
    for phrase in (
        "frames_manifest",
        "layered_completion_plan",
        "semantic_gaussian",
        "object_facts",
        "completed_object_assets_manifest",
        "clean_scene_mesh",
        "clean_plate_manifest",
    ):
        assert phrase in getting_started
        assert phrase in completion or phrase in pipeline

    stale_claims = (
        "### Bedroom4 真实 R1-R4 clean plate 与 fresh DA3/PGSR/TSDF",
        "### Canonical strict clean scene Web: promoted current-demo-only",
        "新链已经把 TRELLIS2 PBR objects",
        "严格 clean scene 与四个 unified PBR objects 已进入 canonical",
    )
    for phrase in stale_claims:
        assert phrase not in pipeline
