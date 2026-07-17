from __future__ import annotations

import ast
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from video2world.cli import build_parser
from video2world.providers import qwen_scene_audit
from video2world.providers.qwen_scene_audit import (
    AuditEvidenceFrame,
    ExistingAssetSummary,
    InferenceResult,
    QwenSceneAuditError,
    SceneAudit,
    build_audit_prompt,
    compare_asset_coverage,
    load_asset_summary,
    load_scene_audit,
    merge_scene_audits,
    parse_audit_response,
    run_scene_audit_merge,
    run_scene_audit_provider,
)

REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"


def response_payload() -> dict[str, object]:
    common = {
        "evidence_frame_ids": ["000000", "000064"],
        "confidence": 0.92,
        "limitations": ["Rear surfaces are not visible in the selected frames."],
    }
    return {
        "categories": [
            {
                "category": "bed",
                "visible_instance_count": 1,
                "color_description": "Dark carved brown headboard and pale bedding.",
                "shape_description": "One large rectangular bed with an integrated frame.",
                "material_description": "Wood-like frame with textile-like bedding.",
                "support_or_parent_category": "floor",
                "importance": "primary",
                **common,
            },
            {
                "category": "pillow",
                "visible_instance_count": 3,
                "color_description": "Pale green and off-white patchwork surfaces.",
                "shape_description": "Three separate soft near-rectangular cushions.",
                "material_description": "Quilted textile-like covers with soft fill.",
                "support_or_parent_category": "bed",
                "importance": "primary",
                **common,
            },
            {
                "category": "plant",
                "visible_instance_count": 1,
                "color_description": "Green foliage above an off-white pot.",
                "shape_description": "Rounded foliage crown on upright stems.",
                "material_description": "Matte leaf-like surfaces and a smooth pot.",
                "support_or_parent_category": "nightstand",
                "importance": "secondary",
                **common,
                "evidence_frame_ids": ["000000"],
            },
            {
                "category": "nightstand",
                "visible_instance_count": 1,
                "color_description": "Warm reddish-brown cabinet surfaces.",
                "shape_description": "Compact rectangular cabinet with four legs.",
                "material_description": "Smooth wood-like furniture surface.",
                "support_or_parent_category": "floor",
                "importance": "secondary",
                **common,
                "evidence_frame_ids": ["000064"],
            },
            {
                "category": "painting",
                "visible_instance_count": 1,
                "color_description": "Muted multicolor image inside a dark frame.",
                "shape_description": "Thin rectangular framed wall decoration.",
                "material_description": "Printed or painted flat surface with a rigid frame.",
                "support_or_parent_category": "wall",
                "importance": "secondary",
                **common,
                "evidence_frame_ids": ["000000"],
            },
        ],
        "scene_limitations": [
            "The selected views do not reveal hidden backs or prove object boundaries."
        ],
    }


def asset_summary_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "kind": "video2world.existing_asset_summary",
        "scene_id": "bedroom_4",
        "assets": [
            {
                "asset_id": "legacy_pillow_ensemble",
                "category": "pillows",
                "represented_instance_count": 3,
                "instance_scope": "aggregate",
                "pipeline_state": "segmented",
            },
            {
                "asset_id": "plant_01",
                "category": "potted plant",
                "represented_instance_count": 1,
                "instance_scope": "individual",
                "pipeline_state": "segmented",
            },
            {
                "asset_id": "nightstand_01",
                "category": "bedside table",
                "represented_instance_count": 1,
                "instance_scope": "individual",
                "pipeline_state": "completed",
            },
            {
                "asset_id": "painting_01",
                "category": "wall art",
                "represented_instance_count": 1,
                "instance_scope": "individual",
                "pipeline_state": "review_passed",
            },
        ],
        "limitations": ["The legacy pillow asset merges three visible instances."],
    }


def frames() -> list[AuditEvidenceFrame]:
    return [
        AuditEvidenceFrame(frame_id="000000", uri="contact-sheet://frame/000000"),
        AuditEvidenceFrame(frame_id="000064", uri="contact-sheet://frame/000064"),
    ]


def audit_and_summary() -> tuple[SceneAudit, ExistingAssetSummary]:
    response, _, _ = parse_audit_response(
        json.dumps(response_payload(), ensure_ascii=False),
        ["000000", "000064"],
    )
    audit = SceneAudit(
        scene_id="bedroom_4",
        run_id="audit-test",
        created_at=datetime(2026, 7, 17, 10, 0, tzinfo=UTC),
        provider="test",
        model=f"test@{REVISION}",
        evidence_frames=frames(),
        categories=response.categories,
        scene_limitations=response.scene_limitations,
    )
    summary = ExistingAssetSummary.model_validate(asset_summary_payload(), strict=True)
    return audit, summary


def single_frame_audit(
    frame_id: str,
    run_id: str,
    payload: dict[str, object],
    *,
    scene_id: str = "bedroom_4",
) -> SceneAudit:
    for category in payload["categories"]:
        category["evidence_frame_ids"] = [frame_id]
    response, _, _ = parse_audit_response(json.dumps(payload), [frame_id])
    return SceneAudit(
        scene_id=scene_id,
        run_id=run_id,
        created_at=datetime(2026, 7, 17, 10, 0, tzinfo=UTC),
        provider="local_huggingface_transformers",
        model=f"Qwen/Qwen2.5-VL-7B-Instruct@{REVISION}",
        evidence_frames=[
            AuditEvidenceFrame(frame_id=frame_id, uri=f"file:///frames/{frame_id}.png")
        ],
        categories=response.categories,
        scene_limitations=response.scene_limitations,
    )


def test_prompt_is_narrow_and_hands_geometry_to_sam3_depth() -> None:
    _, summary = audit_and_summary()
    prompt = build_audit_prompt(frames(), summary)

    assert "Three distinct pillows are three instances" in prompt
    assert "visible_instance_count" in prompt
    assert "Do not output bbox, mask" in prompt
    assert "SAM3 + calibrated depth" in prompt
    assert "canonical_category" in prompt
    assert "occlusion_edges" not in prompt
    assert "peel_priority" not in prompt


def test_parser_accepts_one_fence_but_keeps_final_schema_strict() -> None:
    payload = json.dumps(response_payload(), ensure_ascii=False)
    plain, plain_envelope, plain_normalizations = parse_audit_response(
        payload, ["000000", "000064"]
    )
    fenced, fenced_envelope, fenced_normalizations = parse_audit_response(
        f"```json\n{payload}\n```", ["000000", "000064"]
    )

    assert plain_envelope == "json"
    assert fenced_envelope == "fenced_json"
    assert plain == fenced
    assert plain_normalizations == fenced_normalizations == []

    labeled = response_payload()
    labeled["categories"][0]["evidence_frame_ids"] = ["F01", "F02"]
    normalized, _, normalizations = parse_audit_response(json.dumps(labeled), ["000000", "000064"])
    assert normalized.categories[0].evidence_frame_ids == ["000000", "000064"]
    assert normalizations == [
        {"source": "F01", "frame_id": "000000"},
        {"source": "F02", "frame_id": "000064"},
    ]

    duplicate_key = payload.replace(
        '"confidence": 0.92',
        '"confidence": 0.92, "confidence": 0.92',
        1,
    )
    with pytest.raises(QwenSceneAuditError, match="repeats JSON key"):
        parse_audit_response(duplicate_key, ["000000", "000064"])

    with pytest.raises(QwenSceneAuditError, match="not one JSON object"):
        parse_audit_response(f"Here is the result:\n```json\n{payload}\n```", ["000000", "000064"])

    drifted = response_payload()
    drifted["categories"][0]["bbox_0_1000"] = [0, 0, 1000, 1000]
    with pytest.raises(QwenSceneAuditError, match="schema validation failed"):
        parse_audit_response(json.dumps(drifted), ["000000", "000064"])


def test_parser_rejects_zero_placeholder_and_unknown_frame_evidence() -> None:
    payload = response_payload()
    payload["categories"][0]["confidence"] = 0.0
    with pytest.raises(QwenSceneAuditError, match="greater than 0"):
        parse_audit_response(json.dumps(payload), ["000000", "000064"])

    payload = response_payload()
    payload["categories"][0]["material_description"] = "unknown"
    with pytest.raises(QwenSceneAuditError, match="placeholder"):
        parse_audit_response(json.dumps(payload), ["000000", "000064"])

    payload = response_payload()
    payload["categories"][0]["evidence_frame_ids"] = ["000999"]
    with pytest.raises(QwenSceneAuditError, match="unknown frames"):
        parse_audit_response(json.dumps(payload), ["000000", "000064"])

    payload = response_payload()
    payload["categories"][0]["limitations"] = ["floor"]
    with pytest.raises(QwenSceneAuditError, match="specific evidence limitation"):
        parse_audit_response(json.dumps(payload), ["000000", "000064"])

    payload = response_payload()
    for category in payload["categories"]:
        category["evidence_frame_ids"] = ["000000", "000064"]
    with pytest.raises(QwenSceneAuditError, match="not discriminative"):
        parse_audit_response(json.dumps(payload), ["000000", "000064"])

    for category in payload["categories"]:
        category["evidence_frame_ids"] = ["000000"]
    single_frame, _, _ = parse_audit_response(json.dumps(payload), ["000000"])
    assert all(item.evidence_frame_ids == ["000000"] for item in single_frame.categories)


def test_per_view_merge_is_deterministic_conservative_and_auditable(tmp_path: Path) -> None:
    source_audits: list[SceneAudit] = []
    source_paths: list[Path] = []
    for index in range(8):
        frame_id = f"{index * 5:06d}"
        payload = response_payload()
        pillow = next(item for item in payload["categories"] if item["category"] == "pillow")
        payload["categories"] = [pillow]
        pillow["visible_instance_count"] = 3 if index == 3 else 1
        pillow["confidence"] = 0.55 + index * 0.04
        audit = single_frame_audit(frame_id, f"source-{index}", payload)
        source_dir = tmp_path / f"source-{index}"
        source_dir.mkdir()
        source_path = source_dir / "scene_audit.json"
        source_path.write_text(audit.model_dump_json(indent=2), encoding="utf-8")
        (source_dir / "run_receipt.json").write_text(
            json.dumps({"status": "completed", "run_id": audit.run_id}), encoding="utf-8"
        )
        source_audits.append(audit)
        source_paths.append(source_path)

    merged = merge_scene_audits(
        source_audits,
        run_id="merged-audit",
        created_at=datetime(2026, 7, 17, 11, 0, tzinfo=UTC),
    )
    pillow = merged.categories[0]
    assert pillow.category == "pillow"
    assert pillow.visible_instance_count == 3
    assert len(pillow.evidence_frame_ids) == 6
    assert pillow.evidence_frame_ids == ["000035", "000030", "000025", "000020", "000015", "000010"]
    assert "maximum visible in one audited view" in " ".join(pillow.limitations)

    summary_path = tmp_path / "assets.json"
    summary_path.write_text(json.dumps(asset_summary_payload()), encoding="utf-8")
    output_dir = tmp_path / "merged"
    result = run_scene_audit_merge(
        audit_paths=source_paths,
        asset_summary_path=summary_path,
        run_id="merged-audit",
        output_dir=output_dir,
        created_at=datetime(2026, 7, 17, 11, 0, tzinfo=UTC),
    )
    comparison = json.loads((output_dir / "coverage_comparison.json").read_text(encoding="utf-8"))
    receipt = json.loads((output_dir / "merge_receipt.json").read_text(encoding="utf-8"))
    assert result["source_audit_count"] == 8
    assert comparison["categories"][0]["coverage_status"] == "undercovered"
    assert comparison["categories"][0]["next_action"] == "instance_split"
    assert {
        item["canonical_category"] for item in comparison["unassessed_existing_categories"]
    } == {
        "nightstand",
        "painting",
        "plant",
    }
    assert receipt["deterministic_merge_policy"]["visible_instance_count"] == (
        "maximum_per_view_never_sum"
    )
    assert receipt["deterministic_merge_policy"]["invent_missing_categories"] is False
    assert all("run_receipt" in item for item in receipt["source_audits"])
    assert len(receipt["artifacts"]["scene_audit"]["sha256"]) == 64

    mismatched = source_audits[-1].model_copy(update={"scene_id": "another_scene"})
    with pytest.raises(QwenSceneAuditError, match="scene mismatch"):
        merge_scene_audits(
            [source_audits[0], mismatched],
            run_id="bad-merge",
            created_at=datetime(2026, 7, 17, 11, 0, tzinfo=UTC),
        )


def test_deterministic_coverage_comparison_selects_each_action() -> None:
    audit, summary = audit_and_summary()
    comparison = compare_asset_coverage(audit, summary)
    by_category = {item.canonical_category: item for item in comparison.categories}

    assert (by_category["bed"].coverage_status, by_category["bed"].next_action) == (
        "missing",
        "segment",
    )
    assert (
        by_category["pillow"].coverage_status,
        by_category["pillow"].next_action,
    ) == ("undercovered", "instance_split")
    assert (by_category["plant"].coverage_status, by_category["plant"].next_action) == (
        "covered",
        "complete",
    )
    assert (
        by_category["nightstand"].coverage_status,
        by_category["nightstand"].next_action,
    ) == ("covered", "review")
    assert (
        by_category["painting"].coverage_status,
        by_category["painting"].next_action,
    ) == ("covered", "none")
    assert by_category["bed"].requires_sam3_depth
    assert by_category["pillow"].requires_sam3_depth
    assert not by_category["plant"].requires_sam3_depth
    assert "SAM3 must produce per-instance" in comparison.segmentation_geometry_handoff


def test_provider_preserves_fenced_raw_and_writes_auditable_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contact_sheet = tmp_path / "existing-contact.png"
    contact_sheet.write_bytes(b"real-existing-contact-sheet")
    summary_path = tmp_path / "assets.json"
    summary_path.write_text(
        json.dumps(asset_summary_payload(), ensure_ascii=False), encoding="utf-8"
    )
    model_dir = tmp_path / "qwen-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type":"qwen2_5_vl"}', encoding="utf-8")
    (model_dir / "preprocessor_config.json").write_text('{"processor":"test"}', encoding="utf-8")

    def forbid_pillow() -> tuple[object, object, object]:
        raise AssertionError("Pillow must stay lazy for an existing contact sheet")

    monkeypatch.setattr(qwen_scene_audit, "_load_pillow", forbid_pillow)

    def fake_inference(
        model_dir: Path,
        received_contact_sheet: Path,
        prompt: str,
        max_new_tokens: int,
        device: str,
    ) -> InferenceResult:
        assert model_dir.name == "qwen-model"
        assert received_contact_sheet.read_bytes() == b"real-existing-contact-sheet"
        assert "frame_id=000064" in prompt
        assert max_new_tokens == 2048
        assert device == "cuda:7"
        raw = json.dumps(response_payload(), ensure_ascii=False)
        return InferenceResult(
            raw_response=f"```json\n{raw}\n```",
            runtime={"device": device, "test_double": True},
        )

    output_dir = tmp_path / "output"
    result = run_scene_audit_provider(
        frame_ids=["000000", "000064"],
        scene_id="bedroom_4",
        run_id="narrow-audit-test",
        asset_summary_path=summary_path,
        model_dir=model_dir,
        model_revision=REVISION,
        output_dir=output_dir,
        contact_sheet=contact_sheet,
        max_new_tokens=2048,
        device="cuda:7",
        created_at=datetime(2026, 7, 17, 10, 0, tzinfo=UTC),
        inference_runner=fake_inference,
    )

    assert result["category_count"] == 5
    raw = (output_dir / "raw_response.txt").read_text(encoding="utf-8")
    audit = json.loads((output_dir / "scene_audit.json").read_text(encoding="utf-8"))
    comparison = json.loads((output_dir / "coverage_comparison.json").read_text(encoding="utf-8"))
    receipt = json.loads((output_dir / "run_receipt.json").read_text(encoding="utf-8"))
    assert raw.startswith("```json")
    assert load_scene_audit(output_dir / "scene_audit.json").scene_id == "bedroom_4"
    assert set(audit["categories"][0]) == {
        "category",
        "visible_instance_count",
        "evidence_frame_ids",
        "color_description",
        "shape_description",
        "material_description",
        "support_or_parent_category",
        "importance",
        "confidence",
        "limitations",
    }
    assert comparison["kind"] == "video2world.scene_coverage_comparison"
    assert receipt["status"] == "completed"
    assert receipt["response_envelope"] == "fenced_json"
    assert receipt["runtime"] == {"device": "cuda:7", "test_double": True}
    assert set(receipt["model"]["files"]) == {"config.json", "preprocessor_config.json"}
    assert receipt["model"]["files"]["config.json"]["resolved_path"] == str(
        model_dir / "config.json"
    )
    assert receipt["boundary"]["vlm_does_not_output"] == [
        "bbox",
        "mask",
        "stable_instance_id",
        "occlusion_edge",
        "depth_order",
        "hidden_geometry",
    ]
    for role in (
        "contact_sheet",
        "prompt",
        "raw_response",
        "scene_audit",
        "coverage_comparison",
    ):
        assert len(receipt["artifacts"][role]["sha256"]) == 64


def test_failed_response_still_writes_failure_receipt(tmp_path: Path) -> None:
    contact_sheet = tmp_path / "contact.png"
    contact_sheet.write_bytes(b"contact")
    summary_path = tmp_path / "assets.json"
    summary_path.write_text(json.dumps(asset_summary_payload()), encoding="utf-8")
    payload = response_payload()
    payload["categories"][0]["evidence_frame_ids"] = ["invented_frame"]

    def fake_inference(*_: object) -> InferenceResult:
        return InferenceResult(raw_response=json.dumps(payload), runtime={"test": True})

    output_dir = tmp_path / "failed"
    with pytest.raises(QwenSceneAuditError, match="unknown frames"):
        run_scene_audit_provider(
            frame_ids=["000000", "000064"],
            scene_id="bedroom_4",
            run_id="failed-audit-test",
            asset_summary_path=summary_path,
            model_dir=tmp_path / "qwen-model",
            model_revision=REVISION,
            output_dir=output_dir,
            contact_sheet=contact_sheet,
            inference_runner=fake_inference,
        )
    receipt = json.loads((output_dir / "run_receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["failure"]["error_type"] == "QwenSceneAuditError"


def test_asset_summary_is_strict_and_module_parses_as_python_310(tmp_path: Path) -> None:
    invalid = asset_summary_payload()
    invalid["assets"][0]["represented_instance_count"] = "3"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(QwenSceneAuditError, match="invalid existing asset summary"):
        load_asset_summary(path)

    source = Path(qwen_scene_audit.__file__).read_text(encoding="utf-8")
    ast.parse(source, feature_version=(3, 10))


def test_root_cli_exposes_narrow_scene_audit() -> None:
    args = build_parser().parse_args(
        [
            "qwen-scene-audit",
            "--contact-sheet",
            "/data/contact.png",
            "--frame-ids",
            "000000,000064",
            "--scene-id",
            "bedroom_4",
            "--run-id",
            "audit-run",
            "--asset-summary",
            "/data/assets.json",
            "--model-dir",
            "/models/qwen",
            "--output-dir",
            "/outputs/audit",
        ]
    )

    assert args.command == "qwen-scene-audit"
    assert args.max_new_tokens == 3072

    merge_args = build_parser().parse_args(
        [
            "qwen-scene-audit-merge",
            "--audit",
            "/outputs/view-1/scene_audit.json",
            "--audit",
            "/outputs/view-2/scene_audit.json",
            "--asset-summary",
            "/data/assets.json",
            "--run-id",
            "merged-audit",
            "--output-dir",
            "/outputs/merged",
        ]
    )
    assert merge_args.command == "qwen-scene-audit-merge"
    assert len(merge_args.audit) == 2
