from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from scripts.build_layered_completion_plan import build_plan
from video2world.completion import (
    InventoryFrame,
    OcclusionEdge,
    OcclusionGraph,
    SceneAssetObservation,
    SceneInventory,
    load_layered_completion_plan,
    load_occlusion_graph,
    load_scene_inventory,
)
from video2world.hashing import digest_json, sha256_file
from video2world.models import LocalizedText

SHA = "a" * 64


def observation(item_id: str, *, category: str, priority: int) -> SceneAssetObservation:
    return SceneAssetObservation(
        id=item_id,
        category=category,
        name=LocalizedText(en=category, zh=category),
        asset_role="interactive_object",
        granularity="independent_root_asset",
        independently_movable=True,
        semantic_unit_rationale="Audited independent scene asset.",
        importance="primary",
        confidence=0.9,
        peel_priority=priority,
        evidence_frame_ids=["000064"],
        coverage_status="segmented_only",
        completion_needs=["backside_geometry", "render_asset", "clean_plate", "visual_qa"],
    )


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fixture_inputs(root: Path) -> tuple[Path, Path, Path]:
    inventory = SceneInventory(
        scene_id="bedroom_4",
        run_id="layered-test",
        created_at=datetime(2026, 7, 17, tzinfo=UTC),
        provider="test-audit",
        model="deterministic-fixture",
        evidence_frames=[
            InventoryFrame(
                frame_id="000064",
                uri="artifact://bedroom4/000064.png",
                sha256=SHA,
                width=1280,
                height=720,
            )
        ],
        observations=[
            observation("front", category="pillow", priority=100),
            observation("left", category="pillow", priority=80),
            observation("right", category="pillow", priority=79),
            observation("bed", category="bed", priority=60),
            SceneAssetObservation(
                id="background",
                category="room",
                name=LocalizedText(en="background", zh="背景"),
                asset_role="structure",
                granularity="structure_background",
                independently_movable=False,
                semantic_unit_rationale="Terminal structural background.",
                importance="structure",
                structure_role="background",
                confidence=0.9,
                evidence_frame_ids=["000064"],
                coverage_status="structure_background",
                completion_needs=["background_rebuild", "visual_qa"],
            ),
        ],
    )
    inventory_path = root / "inventory.json"
    write_json(inventory_path, inventory.model_dump(mode="json"))
    graph = OcclusionGraph(
        scene_id="bedroom_4",
        inventory_sha256=digest_json(inventory.model_dump(mode="json")),
        edges=[
            OcclusionEdge(
                foreground_id="front",
                background_id=rear,
                confidence=0.9,
                evidence_frame_ids=["000064"],
                evidence_uris=["repo://video2world/depth-evidence.json"],
                ordering_source="combined",
                verification_status="geometry_verified",
            )
            for rear in ("left", "right")
        ],
        unresolved_relations=["left/right -> bed lacks geometry-verified depth"],
    )
    graph_path = root / "graph.json"
    write_json(graph_path, graph.model_dump(mode="json"))

    output_clean_plate = root / "round1-clean.png"
    output_clean_plate.write_bytes(b"real-clean-plate-fixture")
    mask_index = root / "reinspection-mask-index.json"
    mask_index.write_text('{"items":[]}\n', encoding="utf-8")
    mask_index_sha256, _ = sha256_file(mask_index)
    quality_report = root / "round1-quality.json"
    quality_report.write_text(
        json.dumps(
            {
                "status": "passed",
                "gates": {"passed": True},
                "next_layer_target_ids": ["left", "right"],
                "sources": {"sam3_mask_index": {"sha256": mask_index_sha256}},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    depth_evidence = root / "depth-evidence.json"
    depth_evidence.write_text('{"status":"verified"}\n', encoding="utf-8")

    def evidence(
        role: str,
        path: Path,
        claim_scope: str,
        *,
        usage: str = "evidence_only",
    ) -> dict[str, str]:
        digest, _ = sha256_file(path)
        return {
            "role": role,
            "path": path.relative_to(root).as_posix(),
            "expected_sha256": digest,
            "claim_scope": claim_scope,
            "usage": usage,
        }

    status = {
        "schema_version": "1.0",
        "kind": "video2world.layered_completion_status_input",
        "created_at": "2026-07-17T08:00:00+08:00",
        "target_order": ["front", "left", "right", "bed"],
        "rounds": [
            {
                "kind": "object_layer",
                "target_ids": ["front"],
                "status": "passed_with_limitations",
                "acceptance_scope": "usable_for_reinspection_only",
                "status_reason": "Current-demo scope only.",
                "next_layer_candidate_ids": ["left", "right"],
                "next_layer_candidates_evidence_role": "round_quality_report",
                "next_layer_mask_index_evidence_role": "reinspection_mask_index",
                "evidence": [
                    evidence(
                        "output_clean_plate",
                        output_clean_plate,
                        "Selected current-demo clean plate only.",
                        usage="next_round_input",
                    ),
                    evidence(
                        "round_quality_report",
                        quality_report,
                        "Records the scope-limited review decision.",
                    ),
                    evidence(
                        "occlusion_ordering",
                        depth_evidence,
                        "Verifies only the front-to-rear ordering edges.",
                    ),
                    evidence(
                        "reinspection_mask_index",
                        mask_index,
                        "Binds the mask index used by association.",
                    ),
                ],
                "limitations": ["Hidden background geometry is not reconstructed."],
            },
            *[
                {
                    "kind": "object_layer",
                    "target_ids": [target],
                    "status": "planned",
                }
                for target in ("left", "right", "bed")
            ],
            {"kind": "final_background", "target_ids": [], "status": "planned"},
        ],
        "limitations": ["Rear-to-bed ordering remains an operational schedule."],
    }
    status_path = root / "round-status.json"
    write_json(status_path, status)
    return inventory_path, graph_path, status_path


def test_build_plan_binds_real_evidence_and_keeps_deeper_rounds_pending(
    tmp_path: Path,
) -> None:
    inventory_path, graph_path, status_path = fixture_inputs(tmp_path)

    first = build_plan(
        project_root=tmp_path,
        inventory_path=inventory_path,
        graph_path=graph_path,
        status_path=status_path,
    )
    second = build_plan(
        project_root=tmp_path,
        inventory_path=inventory_path,
        graph_path=graph_path,
        status_path=status_path,
    )

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.ordering_policy == "explicit_topological_sequence"
    assert [item.target_ids for item in first.rounds] == [
        ["front"],
        ["left"],
        ["right"],
        ["bed"],
        [],
    ]
    assert first.rounds[0].status == "passed_with_limitations"
    assert first.rounds[0].acceptance_scope == "usable_for_reinspection_only"
    assert all(item.status == "planned" for item in first.rounds[1:])
    assert first.terminal_status == "running"
    assert len(first.rounds[0].evidence) == 4
    assert first.rounds[0].next_layer_candidate_ids == ["left", "right"]
    assert first.rounds[1].input_clean_plate_uri == first.rounds[0].output_clean_plate_uri
    assert first.rounds[2].input_clean_plate_uri is None
    assert first.rounds[0].input_clean_plate_round == 0
    assert [item.input_clean_plate_round for item in first.rounds] == [0, 1, 2, 3, 4]


def test_build_plan_rejects_drifted_evidence(tmp_path: Path) -> None:
    inventory_path, graph_path, status_path = fixture_inputs(tmp_path)
    (tmp_path / "round1-quality.json").write_text('{"status":"changed"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="evidence hash mismatch"):
        build_plan(
            project_root=tmp_path,
            inventory_path=inventory_path,
            graph_path=graph_path,
            status_path=status_path,
        )


def test_explicit_order_cannot_violate_verified_occlusion(tmp_path: Path) -> None:
    inventory_path, graph_path, status_path = fixture_inputs(tmp_path)
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    payload["target_order"] = ["left", "front", "right", "bed"]
    write_json(status_path, payload)

    with pytest.raises(ValueError, match="violates geometry-verified occlusion edge"):
        build_plan(
            project_root=tmp_path,
            inventory_path=inventory_path,
            graph_path=graph_path,
            status_path=status_path,
        )


def test_next_layer_candidates_must_come_from_association(tmp_path: Path) -> None:
    inventory_path, graph_path, status_path = fixture_inputs(tmp_path)
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    payload["rounds"][0]["next_layer_candidate_ids"] = ["right", "left"]
    write_json(status_path, payload)

    with pytest.raises(ValueError, match="do not match the association receipt"):
        build_plan(
            project_root=tmp_path,
            inventory_path=inventory_path,
            graph_path=graph_path,
            status_path=status_path,
        )


def test_next_layer_association_failure_surfaces_next_action(tmp_path: Path) -> None:
    inventory_path, graph_path, status_path = fixture_inputs(tmp_path)
    association_path = tmp_path / "round1-quality.json"
    association = json.loads(association_path.read_text(encoding="utf-8"))
    association["status"] = "failed"
    association["gates"] = {"passed": False, "mask_identity_closed": False}
    association["promotion_blocker"] = "next-layer instance association is incomplete"
    association["next_action"] = {
        "action": "segment",
        "blocker": "missing_guard_stable_next_layer_instances",
        "missing_target_ids": ["left"],
        "failed_gates": ["mask_identity_closed"],
        "promotion_approved": False,
    }
    write_json(association_path, association)
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    for evidence in payload["rounds"][0]["evidence"]:
        if evidence["role"] == "round_quality_report":
            evidence["expected_sha256"] = sha256_file(association_path)[0]
    write_json(status_path, payload)

    with pytest.raises(ValueError) as error:
        build_plan(
            project_root=tmp_path,
            inventory_path=inventory_path,
            graph_path=graph_path,
            status_path=status_path,
        )

    message = str(error.value)
    assert "next-layer association receipt did not pass its technical gates" in message
    assert "status=failed" in message
    assert "gates.passed=False" in message
    assert "next_layer_target_ids=left,right" in message
    assert "next_action=segment" in message
    assert "missing_guard_stable_next_layer_instances" in message
    assert "missing_target_ids=left" in message
    assert "failed_gates=mask_identity_closed" in message


def test_display_candidate_cannot_be_declared_as_next_round_input(tmp_path: Path) -> None:
    _, _, status_path = fixture_inputs(tmp_path)
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    payload["rounds"][0]["evidence"][0]["usage"] = "display_only"
    write_json(status_path, payload)

    with pytest.raises(
        ValueError,
        match="output_clean_plate evidence must explicitly declare usage=next_round_input",
    ):
        build_plan(
            project_root=tmp_path,
            inventory_path=tmp_path / "inventory.json",
            graph_path=tmp_path / "graph.json",
            status_path=status_path,
        )


def test_rejected_candidate_cannot_be_declared_as_next_round_input(tmp_path: Path) -> None:
    _, _, status_path = fixture_inputs(tmp_path)
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    rejected = dict(payload["rounds"][0]["evidence"][2])
    rejected.update(
        {
            "role": "rejected_clean_plate_candidate",
            "usage": "next_round_input",
        }
    )
    payload["rounds"][3]["evidence"] = [rejected]
    write_json(status_path, payload)

    with pytest.raises(
        ValueError,
        match="rejected evidence cannot be propagated beyond audit scope",
    ):
        build_plan(
            project_root=tmp_path,
            inventory_path=tmp_path / "inventory.json",
            graph_path=tmp_path / "graph.json",
            status_path=status_path,
        )


def test_bedroom4_plan_is_scope_limited_content_addressed_and_not_terminal() -> None:
    root = Path(__file__).parents[1]
    layered_root = root / "examples/bedroom4/completion/layered-plan"
    inventory = load_scene_inventory(layered_root / "inventory.json")
    graph = load_occlusion_graph(layered_root / "occlusion-graph.json")
    plan = load_layered_completion_plan(layered_root / "plan.json")

    assert plan.inventory_sha256 == digest_json(inventory.model_dump(mode="json"))
    assert plan.occlusion_graph_sha256 == digest_json(graph.model_dump(mode="json"))
    assert plan.terminal_status == "running"
    assert [item.target_ids for item in plan.rounds] == [
        ["sam3_pillow_front"],
        ["sam3_pillow_left"],
        ["sam3_pillow_right"],
        ["sam3_bed_01"],
        [],
    ]
    assert plan.rounds[0].status == "passed_with_limitations"
    assert plan.rounds[0].acceptance_scope == "usable_for_reinspection_only"
    assert [item.status for item in plan.rounds] == [
        "passed_with_limitations",
        "passed_with_limitations",
        "passed_with_limitations",
        "running",
        "planned",
    ]
    assert plan.rounds[1].acceptance_scope == "usable_for_reinspection_only"
    assert plan.rounds[2].acceptance_scope == "usable_for_reinspection_only"
    assert plan.rounds[0].next_layer_candidate_ids == [
        "sam3_pillow_left",
        "sam3_pillow_right",
    ]
    assert plan.rounds[1].next_layer_candidate_ids == ["sam3_pillow_right"]
    assert plan.rounds[2].next_layer_candidate_ids == ["sam3_bed_01"]
    assert plan.rounds[1].input_clean_plate_uri == (
        "repo://video2world/examples/bedroom4/completion/clean_plate_round1/inpaint_out.mp4"
    )
    assert "sdxl" not in plan.rounds[1].input_clean_plate_uri
    assert plan.rounds[2].input_clean_plate_uri == (
        "repo://video2world/examples/bedroom4/completion/layered-peel/"
        "round02_left_pillow/propainter_output/frames/inpaint_out.mp4"
    )
    assert plan.rounds[3].input_clean_plate_uri == (
        "repo://video2world/examples/bedroom4/completion/layered-peel/"
        "round03_right_pillow/propainter_output/frames/inpaint_out.mp4"
    )
    assert plan.rounds[3].output_clean_plate_uri is None
    assert plan.rounds[4].input_clean_plate_uri is None
    assert {(edge.foreground_id, edge.background_id) for edge in graph.edges} == {
        ("sam3_pillow_front", "sam3_pillow_left"),
        ("sam3_pillow_front", "sam3_pillow_right"),
    }
    assert any("sam3_bed_01" in item for item in graph.unresolved_relations)
    evidence_by_role = {item.role: item for item in plan.rounds[0].evidence}
    assert evidence_by_role["round_quality_report"].sha256 == (
        "25ce62f3c83af0438510a2864b7daeafc0ef254815864b52a66be45e780c1ecb"
    )
    assert evidence_by_role["reinspection_mask_index"].sha256 == (
        "f93067d48886ef6a11bc4525ed8aee1d2ef42e8ca65572f2e2455566c5709b26"
    )
    assert evidence_by_role["output_clean_plate"].sha256 == (
        "6db53f516a8ded9be00f294e559a82489f4ead1175137d260e9123f44a1c0830"
    )
    assert evidence_by_role["display_candidate_sdxl"].sha256 == (
        "26a734c0296d98790ed4daaf5a5a7b45c64f25430ab95e6115e9a26eff41d7b0"
    )
    assert evidence_by_role["output_clean_plate"].usage == "next_round_input"
    assert evidence_by_role["display_candidate_sdxl"].usage == "display_only"

    round2_evidence = {item.role: item for item in plan.rounds[1].evidence}
    assert round2_evidence["round_quality_report"].sha256 == (
        "285c621fb68e526a8bd3ceb453616f088d37f49a7688b7cb957bb64c7d1fdeec"
    )
    assert round2_evidence["accepted_object_asset"].sha256 == (
        "c76e26f14cc68b17fd588984fdae289110d18dee6316f54214d39e0cdd3f4998"
    )
    assert round2_evidence["rejected_object_asset_seed42"].usage == "evidence_only"

    round3_evidence = {item.role: item for item in plan.rounds[2].evidence}
    assert round3_evidence["round_quality_report"].sha256 == (
        "4b3938de6649448677beef925099aa30d4c5e6d291bcbb91695f0a3f3592bee2"
    )
    assert round3_evidence["next_layer_association"].sha256 == (
        "bb4f620b9e33b978ebe050970ddbe37a10fc5545e04fe6c2e94f03b1d1e2bd61"
    )

    round4_evidence = {item.role: item for item in plan.rounds[3].evidence}
    assert "output_clean_plate" not in round4_evidence
    assert round4_evidence["rejected_clean_plate_candidate"].usage == "evidence_only"
    assert round4_evidence["rejected_object_asset_seed45"].usage == "evidence_only"
    assert plan.rounds[4].evidence[0].role == "blocking_round_receipt"

    status_digest, _ = sha256_file(layered_root / "round-status.json")
    assert plan.round_status_input_sha256 == status_digest
    for round_item in plan.rounds:
        for evidence in round_item.evidence:
            prefix = "repo://video2world/"
            assert evidence.uri.startswith(prefix)
            path = root / evidence.uri.removeprefix(prefix)
            assert sha256_file(path) == (evidence.sha256, evidence.size_bytes)
