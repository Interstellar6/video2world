from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from video2world.cli import main
from video2world.hashing import digest_json
from video2world.models import (
    InteractionPolicy,
    LocalizedText,
    WorldManifest,
    WorldObject,
)
from video2world.scene_command_queue import (
    QueuedSceneCommandJob,
    submit_scene_command,
)
from video2world.scene_commands import (
    AddObjectIntent,
    CommandProvenance,
    DeleteObjectIntent,
    DescriptionPatch,
    MergeObjectsIntent,
    ObjectPropertyPatch,
    ProposedObject,
    QueryDescriptionIntent,
    QueryLocationIntent,
    ReparentObjectIntent,
    SceneCommand,
    SceneCommandPlan,
    SplitObjectIntent,
    TargetSelector,
    UpdatePropertiesIntent,
    load_scene_command,
    load_scene_command_plan,
    plan_scene_command,
    resolve_target,
    write_scene_command_plan,
)

NOW = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)


def _command(intent, *, request_id: str = "request-001", expected: str | None = None):
    return SceneCommand(
        request_id=request_id,
        expected_manifest_sha256=expected,
        provenance=CommandProvenance(
            requested_at=NOW,
            requester="local-user",
            raw_prompt="请按这个结构化命令修改场景",
            parser_provider="structured_json",
            source_message_id="message-001",
        ),
        intent=intent,
    )


def _with_hierarchy(sample_manifest: WorldManifest) -> WorldManifest:
    bed = sample_manifest.objects[0]
    pillow = WorldObject(
        id="pillow01",
        source_run_id="pillow-run",
        scoped_id="bedroom_4::pillow-run::pillow01",
        name=LocalizedText(zh="枕头", en="pillow"),
        category="pillow",
        aliases=["靠枕"],
        semantic_granularity="independent_child_asset",
        parent_object_id=bed.id,
        moves_with_parent=True,
        independently_movable=True,
    )
    pillow_child = WorldObject(
        id="pillow_trim01",
        source_run_id="pillow-run",
        scoped_id="bedroom_4::pillow-run::pillow_trim01",
        name=LocalizedText(zh="枕头装饰", en="pillow trim"),
        category="trim",
        semantic_granularity="independent_child_asset",
        parent_object_id=pillow.id,
        moves_with_parent=True,
        independently_movable=True,
    )
    merged = WorldObject(
        id="bed_sheet_component",
        source_run_id="bed-run",
        scoped_id="bedroom_4::bed-run::bed_sheet_component",
        name=LocalizedText(zh="床单组件", en="bed sheet component"),
        category="bed_component",
        semantic_granularity="merged_component",
        independently_movable=False,
    )
    payload = sample_manifest.model_dump(mode="json")
    payload["objects"].extend(
        [
            pillow.model_dump(mode="json"),
            pillow_child.model_dump(mode="json"),
            merged.model_dump(mode="json"),
        ]
    )
    return WorldManifest.model_validate(payload)


def _with_web_ready_objects(manifest: WorldManifest, *object_ids: str) -> WorldManifest:
    payload = manifest.model_dump(mode="json")
    scene_frame = manifest.scene.coordinate_system.frame_id
    visual = manifest.scene.visual.model_dump(mode="json")
    for item in payload["objects"]:
        if item["id"] not in object_ids:
            continue
        bbox = item["bbox_scene"] or {
            "frame_id": scene_frame,
            "minimum": [0.0, 0.0, 0.0],
            "maximum": [1.0, 1.0, 1.0],
        }
        item["bbox_scene"] = bbox
        pivot = [
            (minimum + maximum) / 2
            for minimum, maximum in zip(bbox["minimum"], bbox["maximum"], strict=True)
        ]
        item["visual"] = visual
        item["transform_scene_from_asset"] = {
            "from_frame": f"{item['id']}_asset",
            "to_frame": scene_frame,
            "matrix": [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            "pivot_scene": pivot,
            "scale_xyz": [1.0, 1.0, 1.0],
        }
        item["quality_gates"] = {
            "file": {"status": "passed"},
            "semantic": {"status": "passed"},
            "alignment": {"status": "passed"},
            "collision": {"status": "not_tested"},
            "visual": {"status": "passed"},
        }
        item["interaction"] = {
            "selectable": True,
            "double_click_action": "spin_360",
            "drag_action": "rotate_yaw",
            "collision_enabled": False,
            "physics_mode": "none",
        }
    return WorldManifest.model_validate(payload)


def _with_unified_web_ready_object(manifest: WorldManifest, object_id: str) -> WorldManifest:
    payload = _with_web_ready_objects(manifest, object_id).model_dump(mode="json")
    item = next(candidate for candidate in payload["objects"] if candidate["id"] == object_id)
    item["visual"] = None
    item["unified_pbr_glb"] = {
        "uri": f"artifact://{object_id}.glb",
        "sha256": "d" * 64,
        "size_bytes": 4096,
        "media_type": "model/gltf-binary",
        "role": "unified_pbr_glb",
        "status": "validated",
        "provenance": {
            "faces": 97082,
            "watertight": False,
            "closed_volume_claim": False,
            "inside_outside_queries_allowed": False,
            "technical_gates": {
                "finite_vertices": True,
                "valid_triangle_indices": True,
                "no_degenerate_faces": True,
                "winding_consistent": True,
                "pbr_material_present": True,
                "positive_extents": True,
            },
        },
    }
    item["collision_topology"] = "surface_bvh"
    item["quality_gates"]["collision"] = {"status": "passed"}
    item["interaction"].update(
        {
            "collision_enabled": True,
            "physics_mode": "kinematic",
        }
    )
    return WorldManifest.model_validate(payload)


def test_target_resolution_fails_closed_for_multiple_instances(
    sample_manifest: WorldManifest,
) -> None:
    ambiguous = resolve_target(sample_manifest, TargetSelector(label="plant"))
    assert ambiguous.status == "ambiguous"
    assert [candidate.object_id for candidate in ambiguous.candidates] == [
        "plant01",
        "plant02",
    ]

    second = resolve_target(sample_manifest, TargetSelector(label="plant", ordinal=2))
    assert second.status == "resolved"
    assert second.resolved_object_id == "plant02"

    out_of_range = resolve_target(sample_manifest, TargetSelector(label="plant", ordinal=3))
    assert out_of_range.status == "ordinal_out_of_range"


def test_query_is_read_only_and_returns_reviewed_evidence(
    sample_manifest: WorldManifest,
) -> None:
    before = sample_manifest.model_dump(mode="json")
    command = _command(QueryLocationIntent(target=TargetSelector(object_id="bed01"), language="zh"))
    plan = plan_scene_command(sample_manifest, command, created_at=NOW)

    assert plan.status == "ready"
    assert plan.preview_only is True
    assert plan.manifest_mutated is False
    assert plan.risk_level == "none"
    assert plan.confirmation.required is False
    assert plan.affected_stages == []
    assert plan.read_result is not None
    assert plan.read_result.answer == "位于房间后侧靠墙处。"
    assert plan.read_result.focus_bbox == sample_manifest.objects[0].bbox_scene
    assert sample_manifest.model_dump(mode="json") == before


def test_description_query_with_ambiguous_target_has_no_operation(
    sample_manifest: WorldManifest,
) -> None:
    plan = plan_scene_command(
        sample_manifest,
        _command(QueryDescriptionIntent(target=TargetSelector(label="植物"))),
        created_at=NOW,
    )
    assert plan.status == "blocked_ambiguous"
    assert plan.operations == []
    assert plan.read_result is None


def test_add_child_routes_six_view_identity_preserving_reconstruction(
    sample_manifest: WorldManifest,
) -> None:
    intent = AddObjectIntent(
        object=ProposedObject(
            proposed_id="pillow03",
            name=LocalizedText(zh="右侧白色枕头", en="right white pillow"),
            category="pillow",
            semantic_granularity="independent_child_asset",
            parent=TargetSelector(object_id="bed01"),
            independently_movable=True,
        ),
        segmentation_prompt="床上右侧的白色长方形枕头",
    )
    blocked = plan_scene_command(sample_manifest, _command(intent), created_at=NOW)
    assert blocked.status == "blocked_invariant"
    assert "knowledge-only" in blocked.warnings[0]

    manifest = _with_web_ready_objects(sample_manifest, "bed01")
    plan = plan_scene_command(manifest, _command(intent), created_at=NOW)

    assert plan.status == "ready"
    assert plan.risk_level == "high"
    assert plan.confirmation.required is True
    assert [stage.stage for stage in plan.affected_stages] == [
        "inventory",
        "sam3",
        "fusion",
        "cognition",
        "completion_plan",
        "layered_completion",
        "placement",
        "bundle",
        "web",
    ]
    completion = next(
        operation
        for operation in plan.operations
        if operation.kind == "complete_object_and_clean_plate"
    )
    assert completion.parameters["required_views"] == [
        "front",
        "back",
        "left",
        "right",
        "top",
        "bottom",
    ]
    assert {"color", "shape", "aspect_ratio"} <= set(
        completion.parameters["preserve_identity_attributes"]
    )
    assert completion.parameters["vlm_review_required"] is True


def test_add_requires_explicit_granularity_and_unique_id(
    sample_manifest: WorldManifest,
) -> None:
    inherited = AddObjectIntent(
        object=ProposedObject(
            proposed_id="new_pillow",
            name=LocalizedText(en="new pillow"),
            category="pillow",
        ),
        segmentation_prompt="new pillow",
    )
    plan = plan_scene_command(sample_manifest, _command(inherited), created_at=NOW)
    assert plan.status == "blocked_invariant"
    assert "explicit root" in plan.warnings[0]

    duplicate = inherited.model_copy(
        update={
            "object": inherited.object.model_copy(
                update={
                    "proposed_id": "bed01",
                    "semantic_granularity": "independent_root_asset",
                }
            )
        }
    )
    duplicate_plan = plan_scene_command(
        sample_manifest,
        _command(duplicate, request_id="request-duplicate"),
        created_at=NOW,
    )
    assert duplicate_plan.status == "blocked_invariant"
    assert "already exist" in duplicate_plan.warnings[0]


def test_delete_parent_requires_explicit_cascade_and_clean_plate(
    sample_manifest: WorldManifest,
) -> None:
    manifest = _with_hierarchy(sample_manifest)
    blocked = plan_scene_command(
        manifest,
        _command(DeleteObjectIntent(target=TargetSelector(object_id="bed01"))),
        created_at=NOW,
    )
    assert blocked.status == "blocked_invariant"
    assert "cascade_descendants=true" in blocked.warnings[0]
    assert blocked.operations == []

    accepted = plan_scene_command(
        manifest,
        _command(
            DeleteObjectIntent(
                target=TargetSelector(object_id="bed01"),
                cascade_descendants=True,
            ),
            request_id="delete-with-children",
        ),
        created_at=NOW,
    )
    assert accepted.status == "ready"
    assert accepted.risk_level == "critical"
    assert accepted.confirmation.required is True
    assert accepted.rollback.reversibility == "conditionally_reversible"
    tombstone = accepted.operations[0]
    assert tombstone.kind == "tombstone_objects"
    assert tombstone.parameters["physical_delete"] is False
    assert tombstone.target_object_ids == ["bed01", "pillow01", "pillow_trim01"]
    clean_plate = next(
        operation
        for operation in accepted.operations
        if operation.kind == "complete_object_and_clean_plate"
    )
    assert clean_plate.parameters["repair_exposed_background"] is True


def test_metadata_update_is_previewed_but_does_not_need_confirmation(
    sample_manifest: WorldManifest,
) -> None:
    before = sample_manifest.model_dump(mode="json")
    plan = plan_scene_command(
        sample_manifest,
        _command(
            UpdatePropertiesIntent(
                target=TargetSelector(object_id="table01"),
                patch=ObjectPropertyPatch(
                    name=LocalizedText(zh="右侧桌子", en="right table"),
                    description=DescriptionPatch(
                        appearance=LocalizedText(zh="深色木桌", en="dark wood table")
                    ),
                ),
            )
        ),
        created_at=NOW,
    )
    assert plan.status == "ready"
    assert plan.risk_level == "low"
    assert plan.confirmation.required is False
    assert plan.preview_only is True
    assert sample_manifest.model_dump(mode="json") == before


def test_collision_update_without_collider_routes_reconstruction(
    sample_manifest: WorldManifest,
) -> None:
    plan = plan_scene_command(
        sample_manifest,
        _command(
            UpdatePropertiesIntent(
                target=TargetSelector(object_id="table01"),
                patch=ObjectPropertyPatch(
                    interaction=InteractionPolicy(
                        selectable=True,
                        double_click_action="spin_360",
                        collision_enabled=True,
                        physics_mode="kinematic",
                    )
                ),
            ),
            request_id="enable-table-collision",
        ),
        created_at=NOW,
    )
    assert plan.status == "ready"
    assert plan.risk_level == "high"
    assert plan.confirmation.required is True
    assert "layered_completion" in {stage.stage for stage in plan.affected_stages}
    completion = next(
        operation
        for operation in plan.operations
        if operation.kind == "complete_object_and_clean_plate"
    )
    assert completion.parameters["unified_pbr_glb_action"] == "generate_and_validate"
    assert completion.parameters["generate_collision_proxy"] is False


def test_interaction_update_reuses_existing_unified_pbr_glb_contract(
    sample_manifest: WorldManifest,
) -> None:
    manifest = _with_unified_web_ready_object(sample_manifest, "table01")
    plan = plan_scene_command(
        manifest,
        _command(
            UpdatePropertiesIntent(
                target=TargetSelector(object_id="table01"),
                patch=ObjectPropertyPatch(
                    interaction=InteractionPolicy(
                        selectable=True,
                        double_click_action="spin_360",
                        drag_action="rotate_yaw",
                        collision_enabled=True,
                        physics_mode="kinematic",
                    )
                ),
            ),
            request_id="reuse-unified-table",
        ),
        created_at=NOW,
    )

    assert plan.status == "ready"
    assert "layered_completion" not in {stage.stage for stage in plan.affected_stages}
    assert all(operation.kind != "complete_object_and_clean_plate" for operation in plan.operations)
    update = next(
        operation for operation in plan.operations if operation.kind == "update_object_properties"
    )
    assert "required_asset_repairs" not in update.parameters


def test_disabling_interaction_clears_unified_reference_and_topology(
    sample_manifest: WorldManifest,
) -> None:
    manifest = _with_unified_web_ready_object(sample_manifest, "table01")
    plan = plan_scene_command(
        manifest,
        _command(
            UpdatePropertiesIntent(
                target=TargetSelector(object_id="table01"),
                patch=ObjectPropertyPatch(interaction=InteractionPolicy()),
            ),
            request_id="disable-unified-table",
        ),
        created_at=NOW,
    )

    update = next(
        operation for operation in plan.operations if operation.kind == "update_object_properties"
    )
    assert update.parameters["clear_unified_pbr_glb_reference"] is True
    assert update.parameters["clear_collision_topology"] is True
    assert update.parameters["reset_collision_gate_to_not_tested"] is True
    assert "clear_collider_reference" not in update.parameters
    assert any("retaining the content-addressed asset" in warning for warning in plan.warnings)


def test_merged_component_rejects_independent_transform_or_reparent(
    sample_manifest: WorldManifest,
) -> None:
    manifest = _with_hierarchy(sample_manifest)
    update = plan_scene_command(
        manifest,
        _command(
            UpdatePropertiesIntent(
                target=TargetSelector(object_id="bed_sheet_component"),
                patch=ObjectPropertyPatch(independently_movable=True),
            )
        ),
        created_at=NOW,
    )
    assert update.status == "blocked_invariant"
    assert "merged components" in update.warnings[0]

    reparent = plan_scene_command(
        manifest,
        _command(
            ReparentObjectIntent(
                target=TargetSelector(object_id="bed_sheet_component"),
                new_parent=TargetSelector(object_id="bed01"),
            ),
            request_id="reparent-merged",
        ),
        created_at=NOW,
    )
    assert reparent.status == "blocked_invariant"
    assert "cannot be independently reparented" in reparent.warnings[0]

    merged_parent = plan_scene_command(
        _with_web_ready_objects(manifest, "pillow01"),
        _command(
            ReparentObjectIntent(
                target=TargetSelector(object_id="pillow01"),
                new_parent=TargetSelector(object_id="bed_sheet_component"),
            ),
            request_id="reparent-to-merged",
        ),
        created_at=NOW,
    )
    assert merged_parent.status == "blocked_invariant"
    assert "cannot own" in merged_parent.warnings[0]


def test_reparent_preserves_world_transform_and_rejects_cycles(
    sample_manifest: WorldManifest,
) -> None:
    manifest = _with_hierarchy(sample_manifest)
    knowledge_only_parent = plan_scene_command(
        manifest,
        _command(
            ReparentObjectIntent(
                target=TargetSelector(object_id="pillow01"),
                new_parent=TargetSelector(object_id="table01"),
            )
        ),
        created_at=NOW,
    )
    assert knowledge_only_parent.status == "blocked_invariant"
    assert "knowledge-only" in knowledge_only_parent.warnings[0]

    ready = plan_scene_command(
        _with_web_ready_objects(manifest, "pillow01", "table01"),
        _command(
            ReparentObjectIntent(
                target=TargetSelector(object_id="pillow01"),
                new_parent=TargetSelector(object_id="table01"),
            )
        ),
        created_at=NOW,
    )
    assert ready.status == "ready"
    assert ready.confirmation.required is True
    hierarchy = ready.operations[0]
    assert hierarchy.parameters == {
        "parent_object_id": "table01",
        "semantic_granularity": "independent_child_asset",
        "moves_with_parent": True,
        "preserve_world_transform": True,
    }

    mesh_parent_payload = _with_web_ready_objects(manifest, "pillow01", "table01").model_dump(
        mode="json"
    )
    table = next(item for item in mesh_parent_payload["objects"] if item["id"] == "table01")
    table["render_mesh"] = table.pop("visual")
    mesh_parent = plan_scene_command(
        WorldManifest.model_validate(mesh_parent_payload),
        _command(
            ReparentObjectIntent(
                target=TargetSelector(object_id="pillow01"),
                new_parent=TargetSelector(object_id="table01"),
            ),
            request_id="reparent-to-mesh-parent",
        ),
        created_at=NOW,
    )
    assert mesh_parent.status == "ready"

    unified_parent_manifest = _with_unified_web_ready_object(
        _with_web_ready_objects(manifest, "pillow01"),
        "table01",
    )
    unified_parent = plan_scene_command(
        unified_parent_manifest,
        _command(
            ReparentObjectIntent(
                target=TargetSelector(object_id="pillow01"),
                new_parent=TargetSelector(object_id="table01"),
            ),
            request_id="reparent-to-unified-parent",
        ),
        created_at=NOW,
    )
    assert unified_parent.status == "ready"

    cycle = plan_scene_command(
        manifest,
        _command(
            ReparentObjectIntent(
                target=TargetSelector(object_id="bed01"),
                new_parent=TargetSelector(object_id="pillow_trim01"),
            ),
            request_id="cycle-request",
        ),
        created_at=NOW,
    )
    assert cycle.status == "blocked_invariant"
    assert "cycle" in cycle.warnings[0]


def test_split_requires_complete_child_reassignment(
    sample_manifest: WorldManifest,
) -> None:
    manifest = _with_hierarchy(sample_manifest)
    parts = [
        ProposedObject(
            proposed_id="bed_frame",
            name=LocalizedText(en="bed frame"),
            category="bed",
        ),
        ProposedObject(
            proposed_id="headboard",
            name=LocalizedText(en="headboard"),
            category="headboard",
        ),
    ]
    blocked = plan_scene_command(
        manifest,
        _command(SplitObjectIntent(target=TargetSelector(object_id="bed01"), parts=parts)),
        created_at=NOW,
    )
    assert blocked.status == "blocked_invariant"
    assert "missing child assignments" in blocked.warnings[0]

    ready = plan_scene_command(
        manifest,
        _command(
            SplitObjectIntent(
                target=TargetSelector(object_id="bed01"),
                parts=parts,
                child_reassignment={"pillow01": "bed_frame"},
            ),
            request_id="split-with-child",
        ),
        created_at=NOW,
    )
    assert ready.status == "ready"
    assert ready.risk_level == "high"
    assert ready.operations[0].kind == "split_semantic_object"


def test_merge_deduplicates_targets_and_rejects_ancestor_descendant(
    sample_manifest: WorldManifest,
) -> None:
    result = ProposedObject(
        proposed_id="plants_cluster",
        name=LocalizedText(en="plant cluster"),
        category="plant_group",
    )
    duplicate = plan_scene_command(
        sample_manifest,
        _command(
            MergeObjectsIntent(
                targets=[
                    TargetSelector(object_id="plant01"),
                    TargetSelector(object_id="plant01"),
                ],
                result=result,
            )
        ),
        created_at=NOW,
    )
    assert duplicate.status == "blocked_invariant"
    assert "duplicate" in duplicate.warnings[0]

    ready = plan_scene_command(
        sample_manifest,
        _command(
            MergeObjectsIntent(
                targets=[
                    TargetSelector(object_id="plant01"),
                    TargetSelector(object_id="plant02"),
                ],
                result=result,
            ),
            request_id="merge-plants",
        ),
        created_at=NOW,
    )
    assert ready.status == "ready"
    assert ready.confirmation.required is True
    assert ready.operations[0].kind == "merge_semantic_objects"
    assert ready.operations[0].parameters["source_object_ids"] == ["plant01", "plant02"]

    manifest = _with_hierarchy(sample_manifest)
    ancestor = plan_scene_command(
        manifest,
        _command(
            MergeObjectsIntent(
                targets=[
                    TargetSelector(object_id="bed01"),
                    TargetSelector(object_id="pillow01"),
                ],
                result=ProposedObject(
                    proposed_id="bed_with_pillow",
                    name=LocalizedText(en="bed with pillow"),
                    category="bed",
                ),
            ),
            request_id="merge-ancestor",
        ),
        created_at=NOW,
    )
    assert ancestor.status == "blocked_invariant"
    assert "ancestor" in ancestor.warnings[0]


def test_idempotency_is_stable_across_plan_time_and_manifest_is_optimistically_locked(
    sample_manifest: WorldManifest,
) -> None:
    command = _command(
        DeleteObjectIntent(target=TargetSelector(object_id="table01")),
        request_id="stable-delete",
    )
    first = plan_scene_command(sample_manifest, command, created_at=NOW)
    second = plan_scene_command(sample_manifest, command, created_at=NOW + timedelta(hours=2))
    assert first.idempotency_key == second.idempotency_key
    assert first.plan_id == second.plan_id
    assert first.created_at != second.created_at

    stale_command = command.model_copy(update={"expected_manifest_sha256": "f" * 64})
    stale = plan_scene_command(sample_manifest, stale_command, created_at=NOW)
    assert stale.status == "blocked_stale_manifest"
    assert stale.operations == []

    canonical_hash = digest_json(sample_manifest.model_dump(mode="json"))
    locked = command.model_copy(update={"expected_manifest_sha256": canonical_hash})
    assert plan_scene_command(sample_manifest, locked, created_at=NOW).status == "ready"


def test_command_and_plan_json_round_trip(
    tmp_path: Path,
    sample_manifest: WorldManifest,
) -> None:
    command = _command(QueryLocationIntent(target=TargetSelector(object_id="bed01")))
    command_path = tmp_path / "command.json"
    command_path.write_text(command.model_dump_json(indent=2), encoding="utf-8")
    assert load_scene_command(command_path) == command

    plan = plan_scene_command(sample_manifest, command, created_at=NOW)
    plan_path = tmp_path / "plan.json"
    write_scene_command_plan(plan_path, plan)
    assert load_scene_command_plan(plan_path) == plan
    assert SceneCommandPlan.model_validate_json(plan_path.read_text(encoding="utf-8")) == plan


def test_structured_contract_rejects_unknown_fields_and_empty_patch() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SceneCommand.model_validate(
            {
                "request_id": "bad-command",
                "provenance": {
                    "requested_at": NOW.isoformat(),
                    "requester": "test",
                    "raw_prompt": "test",
                },
                "intent": {
                    "kind": "query_location",
                    "target": {"object_id": "bed01", "guess": True},
                },
            }
        )
    with pytest.raises(ValidationError, match="property patch must set at least one field"):
        ObjectPropertyPatch()


def test_cli_validates_and_writes_a_preview_without_mutating_manifest(
    tmp_path: Path,
    sample_manifest: WorldManifest,
    capsys,
) -> None:
    manifest_path = tmp_path / "world.json"
    manifest_path.write_text(sample_manifest.model_dump_json(indent=2), encoding="utf-8")
    command = _command(
        DeleteObjectIntent(target=TargetSelector(object_id="table01")),
        request_id="cli-delete",
    )
    command_path = tmp_path / "command.json"
    command_path.write_text(command.model_dump_json(indent=2), encoding="utf-8")

    assert main(["scene-command-validate", str(command_path)]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["valid"] is True
    assert validated["intent"] == "delete_object"
    assert validated["mutating"] is True

    manifest_before = manifest_path.read_bytes()
    output_path = tmp_path / "preview-plan.json"
    assert (
        main(
            [
                "scene-command-plan",
                "--manifest",
                str(manifest_path),
                "--command",
                str(command_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    preview = json.loads(capsys.readouterr().out)
    assert preview["status"] == "ready"
    assert preview["preview_only"] is True
    assert preview["manifest_mutated"] is False
    assert preview["confirmation"]["required"] is True
    assert load_scene_command_plan(output_path).status == "ready"
    assert manifest_path.read_bytes() == manifest_before

    queue_dir = tmp_path / "scene-command-queue"
    assert (
        main(
            [
                "scene-command-submit",
                "--manifest",
                str(manifest_path),
                "--command",
                str(command_path),
                "--queue-dir",
                str(queue_dir),
                "--confirm",
                preview["confirmation"]["required_phrase"],
            ]
        )
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "queued"
    assert Path(receipt["queue_path"]).is_file()
    assert manifest_path.read_bytes() == manifest_before


def test_cli_returns_nonzero_for_ambiguous_preview(
    tmp_path: Path,
    sample_manifest: WorldManifest,
    capsys,
) -> None:
    manifest_path = tmp_path / "world.json"
    manifest_path.write_text(sample_manifest.model_dump_json(indent=2), encoding="utf-8")
    command = _command(
        DeleteObjectIntent(target=TargetSelector(label="plant")),
        request_id="ambiguous-cli-delete",
    )
    command_path = tmp_path / "command.json"
    command_path.write_text(command.model_dump_json(indent=2), encoding="utf-8")

    assert (
        main(
            [
                "scene-command-plan",
                "--manifest",
                str(manifest_path),
                "--command",
                str(command_path),
            ]
        )
        == 3
    )
    preview = json.loads(capsys.readouterr().out)
    assert preview["status"] == "blocked_ambiguous"
    assert preview["operations"] == []


def test_query_submission_completes_without_creating_a_queue_job(
    tmp_path: Path,
    sample_manifest: WorldManifest,
) -> None:
    receipt = submit_scene_command(
        sample_manifest,
        _command(QueryLocationIntent(target=TargetSelector(object_id="bed01"))),
        queue_dir=tmp_path / "queue",
        submitted_at=NOW,
    )

    assert receipt.status == "completed_read"
    assert receipt.read_result is not None
    assert receipt.job_id is None
    assert not (tmp_path / "queue" / "jobs").exists()


def test_confirmed_mutation_is_queued_atomically_and_idempotently(
    tmp_path: Path,
    sample_manifest: WorldManifest,
) -> None:
    command = _command(
        DeleteObjectIntent(target=TargetSelector(object_id="table01")),
        request_id="queue-delete-table",
    )
    plan = plan_scene_command(sample_manifest, command, created_at=NOW)
    assert plan.confirmation.required_phrase is not None
    first = submit_scene_command(
        sample_manifest,
        command,
        queue_dir=tmp_path / "queue",
        confirmation_phrase=plan.confirmation.required_phrase,
        submitted_at=NOW,
    )
    second = submit_scene_command(
        sample_manifest,
        command,
        queue_dir=tmp_path / "queue",
        confirmation_phrase=plan.confirmation.required_phrase,
        submitted_at=NOW + timedelta(minutes=1),
    )

    assert first.status == "queued"
    assert second.status == "duplicate"
    assert first.job_id == second.job_id
    job_path = Path(first.queue_path or "")
    job = QueuedSceneCommandJob.model_validate_json(job_path.read_text(encoding="utf-8"))
    assert job.status == "queued"
    assert job.plan.command_kind == "delete_object"
    assert job.plan.preview_only is True
    assert len(list((tmp_path / "queue" / "manifest_snapshots").glob("*.json"))) == 1


def test_mutation_rejects_wrong_confirmation_without_writing_job(
    tmp_path: Path,
    sample_manifest: WorldManifest,
) -> None:
    command = _command(
        ReparentObjectIntent(
            target=TargetSelector(object_id="plant01"),
            new_parent=TargetSelector(object_id="table01"),
        ),
        request_id="queue-reparent-plant",
    )

    with pytest.raises(ValueError, match="confirmation phrase"):
        submit_scene_command(
            _with_web_ready_objects(sample_manifest, "plant01", "table01"),
            command,
            queue_dir=tmp_path / "queue",
            confirmation_phrase="confirm incorrect",
            submitted_at=NOW,
        )
    assert not (tmp_path / "queue" / "jobs").exists()


def test_blocked_command_returns_receipt_and_never_queues(
    tmp_path: Path,
    sample_manifest: WorldManifest,
) -> None:
    receipt = submit_scene_command(
        sample_manifest,
        _command(
            DeleteObjectIntent(target=TargetSelector(label="plant")),
            request_id="queue-ambiguous-delete",
        ),
        queue_dir=tmp_path / "queue",
        submitted_at=NOW,
    )

    assert receipt.status == "blocked"
    assert receipt.blocked_reason
    assert not (tmp_path / "queue" / "jobs").exists()
