from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from video2world.cli import build_parser
from video2world.completion import InventoryFrame
from video2world.hashing import digest_json, sha256_file
from video2world.providers import qwen_inventory
from video2world.providers.qwen_inventory import (
    InferenceResult,
    QwenInventoryError,
    build_inventory_prompt,
    materialize_inventory,
    parse_frame_ids,
    parse_inventory_response,
    run_inventory_provider,
    validate_scene_semantics,
)

SHA = "a" * 64
REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"


def response_payload() -> dict[str, object]:
    return {
        "observations": [
            {
                "candidate_id": "candidate_bed_01",
                "category": "bed",
                "name_en": "bed",
                "name_zh": "床",
                "aliases": ["large bed"],
                "asset_role": "interactive_object",
                "semantic_granularity": "root",
                "parent_candidate_id": None,
                "moves_with_parent": False,
                "independently_movable": True,
                "semantic_unit_rationale": "The mattress and frame form one bed asset.",
                "instance_count": 1,
                "importance": "primary",
                "structure_role": "none",
                "confidence": 0.97,
                "peel_priority": 55,
                "evidence_regions": [
                    {
                        "frame_id": "000000",
                        "bbox_0_1000": [120, 280, 850, 940],
                        "visibility": "partial",
                        "visible_faces": ["front", "top"],
                        "occluded_by_candidate_ids": ["candidate_pillow_left"],
                        "confidence": 0.96,
                    }
                ],
                "completion_needs": ["segmentation"],
                "completion_risks": ["pillows hide part of the mattress"],
                "notes": [],
            },
            {
                "candidate_id": "candidate_pillow_left",
                "category": "pillow",
                "name_en": "left pillow",
                "name_zh": "左侧枕头",
                "aliases": [],
                "asset_role": "attached_object",
                "semantic_granularity": "child",
                "parent_candidate_id": "candidate_bed_01",
                "moves_with_parent": True,
                "independently_movable": True,
                "semantic_unit_rationale": "One visibly distinct soft pillow.",
                "instance_count": 1,
                "importance": "secondary",
                "structure_role": "none",
                "confidence": 0.94,
                "peel_priority": 90,
                "evidence_regions": [
                    {
                        "frame_id": "000000",
                        "bbox_0_1000": [350, 300, 610, 570],
                        "visibility": "partial",
                        "visible_faces": ["front", "left"],
                        "occluded_by_candidate_ids": [],
                        "confidence": 0.95,
                    },
                    {
                        "frame_id": "000064",
                        "bbox_0_1000": [330, 320, 620, 590],
                        "visibility": "partial",
                        "visible_faces": ["front", "right"],
                        "occluded_by_candidate_ids": [],
                        "confidence": 0.93,
                    },
                ],
                "completion_needs": ["backside_geometry"],
                "completion_risks": ["back surface is never visible"],
                "notes": [],
            },
            {
                "candidate_id": "candidate_headboard_01",
                "category": "headboard",
                "name_en": "bed headboard",
                "name_zh": "床头板",
                "aliases": [],
                "asset_role": "fixture",
                "semantic_granularity": "merged",
                "parent_candidate_id": "candidate_bed_01",
                "moves_with_parent": True,
                "independently_movable": False,
                "semantic_unit_rationale": "The headboard is a component of the bed.",
                "instance_count": 1,
                "importance": "secondary",
                "structure_role": "none",
                "confidence": 0.91,
                "peel_priority": 40,
                "evidence_regions": [
                    {
                        "frame_id": "000064",
                        "bbox_0_1000": [100, 100, 900, 480],
                        "visibility": "partial",
                        "visible_faces": ["front"],
                        "occluded_by_candidate_ids": ["candidate_pillow_left"],
                        "confidence": 0.91,
                    }
                ],
                "completion_needs": [],
                "completion_risks": ["lower boundary is hidden by pillows"],
                "notes": [],
            },
            {
                "candidate_id": "candidate_wall_back",
                "category": "wall",
                "name_en": "back wall",
                "name_zh": "后墙",
                "aliases": [],
                "asset_role": "structure",
                "semantic_granularity": "structure",
                "parent_candidate_id": None,
                "moves_with_parent": False,
                "independently_movable": False,
                "semantic_unit_rationale": "Continuous architectural background.",
                "instance_count": 1,
                "importance": "structure",
                "structure_role": "wall",
                "confidence": 0.98,
                "peel_priority": 0,
                "evidence_regions": [
                    {
                        "frame_id": "000064",
                        "bbox_0_1000": [0, 0, 1000, 500],
                        "visibility": "partial",
                        "visible_faces": ["front"],
                        "occluded_by_candidate_ids": ["candidate_bed_01"],
                        "confidence": 0.97,
                    }
                ],
                "completion_needs": [],
                "completion_risks": ["large foreground occlusion"],
                "notes": [],
            },
        ],
        "occlusion_edges": [
            {
                "foreground_candidate_id": "candidate_pillow_left",
                "background_candidate_id": "candidate_bed_01",
                "confidence": 0.95,
                "evidence_frame_ids": ["000000", "000064"],
                "rationale": "The pillow boundary overlaps and hides the bed surface.",
            },
            {
                "foreground_candidate_id": "candidate_bed_01",
                "background_candidate_id": "candidate_wall_back",
                "confidence": 0.82,
                "evidence_frame_ids": ["000064"],
                "rationale": "The bed silhouette covers the back wall.",
            },
        ],
        "unresolved_relations": ["nightstand-to-wall depth is ambiguous"],
        "scene_limitations": ["No view proves the rear surfaces of movable objects."],
    }


def evidence_frames() -> list[InventoryFrame]:
    return [
        InventoryFrame(
            frame_id="000000",
            uri="file:///frames/000000.png",
            sha256=SHA,
            width=1280,
            height=720,
        ),
        InventoryFrame(
            frame_id="000064",
            uri="file:///frames/000064.png",
            sha256="b" * 64,
            width=1280,
            height=720,
        ),
    ]


def test_prompt_encodes_semantic_units_and_evidence_contract() -> None:
    prompt = build_inventory_prompt(evidence_frames())

    assert "F01: frame_id=000000" in prompt
    assert "one child candidate for each visually distinct pillow" in prompt
    assert "Never merge multiple pillows into one ensemble" in prompt
    assert "bed sheet, blanket, mattress, headboard" in prompt
    assert "bbox_0_1000 is normalized inside" in prompt
    assert "not a mask, stable instance identity, 3D bound" in prompt
    assert "It is not depth order" in prompt
    assert '"semantic_granularity"' in prompt
    assert '"root"' in prompt
    assert "candidate_foreground_01" not in prompt


def test_strict_parser_rejects_markdown_fence_and_extra_fields() -> None:
    payload = json.dumps(response_payload(), ensure_ascii=False)
    with pytest.raises(QwenInventoryError, match="not strict JSON"):
        parse_inventory_response(f"```json\n{payload}\n```")

    drifted = response_payload()
    drifted["unsupported"] = True
    with pytest.raises(QwenInventoryError, match="schema validation failed"):
        parse_inventory_response(json.dumps(drifted, ensure_ascii=False))

    incomplete = response_payload()
    incomplete["observations"][0].pop("completion_risks")
    with pytest.raises(QwenInventoryError, match="schema validation failed"):
        parse_inventory_response(json.dumps(incomplete, ensure_ascii=False))


def test_materialization_preserves_hierarchy_and_builds_occlusion_graph() -> None:
    response = parse_inventory_response(json.dumps(response_payload(), ensure_ascii=False))
    inventory, graph = materialize_inventory(
        response,
        evidence_frames(),
        scene_id="bedroom_4",
        run_id="inventory-test",
        created_at=datetime(2026, 7, 17, 8, 0, tzinfo=UTC),
        provider="test-provider",
        model=f"test-model@{REVISION}",
        minimum_occlusion_confidence=0.6,
    )

    pillow = next(item for item in inventory.observations if item.id == "candidate_pillow_left")
    headboard = next(item for item in inventory.observations if item.id == "candidate_headboard_01")
    wall = next(item for item in inventory.observations if item.id == "candidate_wall_back")
    assert pillow.granularity == "independent_child_asset"
    assert pillow.parent_candidate_id == "candidate_bed_01"
    assert pillow.coverage_status == "missing"
    assert pillow.identity_scope == "observation_not_stable_instance"
    assert pillow.peel_priority == 50
    assert set(pillow.completion_needs) >= {
        "segmentation",
        "multi_view_lift",
        "backside_geometry",
        "render_asset",
        "collider",
        "visual_qa",
    }
    assert headboard.granularity == "merge_into_parent"
    assert not headboard.independently_movable
    assert headboard.coverage_status == "ignored"
    assert wall.coverage_status == "structure_background"
    assert set(wall.completion_needs) >= {"clean_plate", "background_rebuild"}
    assert graph.inventory_sha256 == digest_json(inventory.model_dump(mode="json"))
    assert [(edge.foreground_id, edge.background_id) for edge in graph.edges] == [
        ("candidate_pillow_left", "candidate_bed_01"),
        ("candidate_bed_01", "candidate_wall_back"),
    ]
    assert all(edge.ordering_source == "vlm" for edge in graph.edges)
    assert all(edge.verification_status == "observation_only" for edge in graph.edges)


def test_pillow_ensemble_and_unmerged_bed_component_fail_closed() -> None:
    payload = response_payload()
    payload["observations"][1]["instance_count"] = 3
    response = parse_inventory_response(json.dumps(payload, ensure_ascii=False))
    with pytest.raises(QwenInventoryError, match="exactly one pillow"):
        validate_scene_semantics(response, evidence_frames())

    payload = response_payload()
    component = payload["observations"][2]
    component.update(
        {
            "semantic_granularity": "root",
            "parent_candidate_id": None,
            "moves_with_parent": False,
            "independently_movable": True,
        }
    )
    response = parse_inventory_response(json.dumps(payload, ensure_ascii=False))
    with pytest.raises(QwenInventoryError, match="must be merged"):
        validate_scene_semantics(response, evidence_frames())


def test_unknown_occlusion_candidate_is_rejected() -> None:
    payload = response_payload()
    payload["occlusion_edges"][0]["background_candidate_id"] = "candidate_missing"
    response = parse_inventory_response(json.dumps(payload, ensure_ascii=False))
    with pytest.raises(QwenInventoryError, match="unknown candidates"):
        validate_scene_semantics(response, evidence_frames())


def test_provider_writes_prompt_raw_revision_hashes_and_strong_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for frame_id in ("000000", "000064"):
        (frames_dir / f"{frame_id}.png").write_bytes(f"frame-{frame_id}".encode())

    def fake_contact_sheet(
        frame_ids: list[str],
        frame_paths: list[Path],
        output_path: Path,
        **_: object,
    ) -> list[InventoryFrame]:
        Path(output_path).write_bytes(b"fake-contact-sheet")
        records = []
        for frame_id, path in zip(frame_ids, frame_paths, strict=True):
            records.append(
                InventoryFrame(
                    frame_id=frame_id,
                    uri=path.as_uri(),
                    sha256=sha256_file(path)[0],
                    width=1280,
                    height=720,
                )
            )
        return records

    monkeypatch.setattr(qwen_inventory, "build_labeled_contact_sheet", fake_contact_sheet)

    def fake_inference(
        model_dir: Path,
        contact_sheet: Path,
        prompt: str,
        max_new_tokens: int,
        device: str,
    ) -> InferenceResult:
        assert model_dir.name == "qwen-model"
        assert contact_sheet.read_bytes() == b"fake-contact-sheet"
        assert "frame_id=000064" in prompt
        assert max_new_tokens == 4096
        assert device == "cuda:7"
        return InferenceResult(
            raw_response=json.dumps(response_payload(), ensure_ascii=False),
            runtime={"device": device, "test_double": True},
        )

    output_dir = tmp_path / "output"
    result = run_inventory_provider(
        frames_dir=frames_dir,
        frame_ids=["000000", "000064"],
        scene_id="bedroom_4",
        run_id="qwen-inventory-test",
        model_dir=tmp_path / "qwen-model",
        model_revision=REVISION,
        output_dir=output_dir,
        max_new_tokens=4096,
        device="cuda:7",
        created_at=datetime(2026, 7, 17, 8, 0, tzinfo=UTC),
        inference_runner=fake_inference,
    )

    assert result["observation_count"] == 4
    assert result["occlusion_edge_count"] == 2
    assert result["geometry_verified_occlusion_edge_count"] == 0
    inventory = json.loads((output_dir / "inventory.json").read_text(encoding="utf-8"))
    graph = json.loads((output_dir / "occlusion_graph.json").read_text(encoding="utf-8"))
    receipt = json.loads((output_dir / "run_receipt.json").read_text(encoding="utf-8"))
    assert inventory["kind"] == "video2world.scene_inventory"
    assert graph["kind"] == "video2world.occlusion_graph"
    assert {edge["verification_status"] for edge in graph["edges"]} == {"observation_only"}
    assert receipt["model"]["revision"] == REVISION
    assert receipt["boundary"] == {
        "vlm_bbox_scope": "visible_region_prompt_only_not_mask_or_geometry",
        "vlm_occlusion_scope": "observation_only_not_depth_order",
        "geometry_ordering_required_from": [
            "sam3_masks",
            "calibrated_cameras",
            "depth",
        ],
    }
    assert receipt["runtime"] == {"device": "cuda:7", "test_double": True}
    assert receipt["inputs"]["frames"][0]["sha256"] == sha256_file(frames_dir / "000000.png")[0]
    for role in ("contact_sheet", "prompt", "raw_response", "inventory", "occlusion_graph"):
        artifact = receipt["artifacts"][role]
        assert len(artifact["sha256"]) == 64
        assert Path(artifact["path"]).is_file()
    prompt = (output_dir / "prompt.txt").read_text(encoding="utf-8")
    raw = (output_dir / "raw_response.txt").read_text(encoding="utf-8")
    assert SYSTEM_PROMPT_MARKER in prompt
    assert raw.strip().startswith("{")


SYSTEM_PROMPT_MARKER = "conservative scene supervisor"


def test_frame_id_parser_combines_forms_and_rejects_duplicates() -> None:
    assert parse_frame_ids(["000000"], "000032, 000064") == [
        "000000",
        "000032",
        "000064",
    ]
    with pytest.raises(QwenInventoryError, match="unique"):
        parse_frame_ids(["000000"], "000000")
    with pytest.raises(QwenInventoryError, match="invalid"):
        parse_frame_ids(["../000000"], None)


def test_root_cli_exposes_qwen_inventory_provider() -> None:
    args = build_parser().parse_args(
        [
            "qwen-inventory",
            "--frames-dir",
            "/data/frames",
            "--frame-id",
            "000000",
            "--frame-ids",
            "000032,000064",
            "--scene-id",
            "bedroom_4",
            "--run-id",
            "inventory-run",
            "--model-dir",
            "/models/qwen",
            "--model-revision",
            REVISION,
            "--output-dir",
            "/outputs/inventory",
            "--device",
            "cuda:7",
        ]
    )

    assert args.command == "qwen-inventory"
    assert parse_frame_ids(args.frame_id, args.frame_ids) == ["000000", "000032", "000064"]
    assert args.device == "cuda:7"
