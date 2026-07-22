from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from video2world.completion import (
    CompletedObjectAssetsManifest,
    CompletionAction,
    CompletionAttempt,
    CompletionRound,
    GeometryReview,
    GeometryReviewIssue,
    InventoryFrame,
    LayeredCompletionPlan,
    ObjectCompletionHistory,
    ObjectCompletionReport,
    OcclusionEdge,
    OcclusionGraph,
    SceneAssetObservation,
    SceneInventory,
    build_layered_completion_plan,
    promote_trellis2_unified_pbr_glb_completion,
)
from video2world.hashing import digest_json
from video2world.models import LocalizedText

SHA = "a" * 64


def observation(
    item_id: str,
    *,
    category: str,
    coverage: str = "missing",
    needs: list[str] | None = None,
    importance: str = "primary",
    structure_role: str = "none",
    matched: list[str] | None = None,
) -> SceneAssetObservation:
    return SceneAssetObservation.model_validate(
        {
            "id": item_id,
            "category": category,
            "name": {"en": category, "zh": category},
            "asset_role": "structure" if importance == "structure" else "interactive_object",
            "granularity": (
                "structure_background" if importance == "structure" else "independent_root_asset"
            ),
            "independently_movable": importance != "structure",
            "semantic_unit_rationale": "One coherent functional unit for interaction.",
            "importance": importance,
            "structure_role": structure_role,
            "confidence": 0.95,
            "evidence_frame_ids": ["000064"],
            "coverage_status": coverage,
            "matched_object_ids": matched or [],
            "completion_needs": (
                needs
                if needs is not None
                else [
                    "segmentation",
                    "multi_view_lift",
                    "backside_geometry",
                    "render_asset",
                    "collider",
                    "clean_plate",
                    "visual_qa",
                ]
            ),
        }
    )


def inventory() -> SceneInventory:
    return SceneInventory(
        scene_id="bedroom_4",
        run_id="audit-1",
        created_at=datetime(2026, 7, 17, tzinfo=UTC),
        provider="Qwen2.5-VL",
        model="Qwen2.5-VL-3B-Instruct",
        evidence_frames=[
            InventoryFrame(
                frame_id="000064",
                uri="artifact://bedroom4/frame_000064.png",
                sha256=SHA,
                width=1280,
                height=720,
            )
        ],
        observations=[
            observation("pillow-ensemble", category="pillow"),
            observation("bed", category="bed"),
            observation("wall-art-left", category="painting", importance="secondary"),
            observation(
                "room-background",
                category="room_structure",
                coverage="structure_background",
                needs=["background_rebuild", "visual_qa"],
                importance="structure",
                structure_role="background",
            ),
        ],
        limitations=["Single-frame inventory requires multi-view confirmation."],
    )


def graph_for(value: SceneInventory) -> OcclusionGraph:
    return OcclusionGraph(
        scene_id=value.scene_id,
        inventory_sha256=digest_json(value.model_dump(mode="json")),
        edges=[
            OcclusionEdge(
                foreground_id="pillow-ensemble",
                background_id="bed",
                confidence=0.99,
                evidence_frame_ids=["000064"],
                ordering_source="combined",
                verification_status="geometry_verified",
            ),
            OcclusionEdge(
                foreground_id="bed",
                background_id="wall-art-left",
                confidence=0.40,
                evidence_frame_ids=["000064"],
                ordering_source="vlm",
            ),
        ],
    )


def test_front_to_back_plan_peels_objects_then_rebuilds_background() -> None:
    scene_inventory = inventory()
    plan = build_layered_completion_plan(
        scene_inventory,
        graph_for(scene_inventory),
        created_at=datetime(2026, 7, 17, 1, 0, tzinfo=UTC),
    )
    assert [item.kind for item in plan.rounds] == [
        "object_layer",
        "object_layer",
        "object_layer",
        "final_background",
    ]
    assert plan.rounds[0].target_ids == ["pillow-ensemble"]
    assert plan.rounds[1].target_ids == ["wall-art-left"]
    assert plan.rounds[2].target_ids == ["bed"]
    assert plan.rounds[3].target_ids == []
    assert plan.rounds[0].actions[-2].action == "reinspect"
    assert plan.rounds[0].actions[-1].action == "validate_round"
    assert plan.rounds[-1].actions[0].action == "rebuild_background"
    assert plan.rounds[-1].actions[0].synthetic_output is True
    assert plan.max_parallel_targets_per_round == 1


def test_confident_occlusion_cycle_is_rejected() -> None:
    scene_inventory = inventory()
    digest = digest_json(scene_inventory.model_dump(mode="json"))
    with pytest.raises(ValidationError, match="cycle"):
        OcclusionGraph(
            scene_id="bedroom_4",
            inventory_sha256=digest,
            edges=[
                OcclusionEdge(
                    foreground_id="pillow-ensemble",
                    background_id="bed",
                    confidence=0.9,
                    evidence_frame_ids=["000064"],
                    ordering_source="depth",
                    verification_status="geometry_verified",
                ),
                OcclusionEdge(
                    foreground_id="bed",
                    background_id="pillow-ensemble",
                    confidence=0.9,
                    evidence_frame_ids=["000064"],
                    ordering_source="combined",
                    verification_status="geometry_verified",
                ),
            ],
        )


def test_graph_must_reference_exact_inventory_content() -> None:
    scene_inventory = inventory()
    graph = graph_for(scene_inventory).model_copy(update={"inventory_sha256": "b" * 64})
    with pytest.raises(ValueError, match="does not reference"):
        build_layered_completion_plan(
            scene_inventory,
            graph,
            created_at=datetime(2026, 7, 17, tzinfo=UTC),
        )


def test_observation_only_occlusion_cannot_schedule_completion() -> None:
    scene_inventory = inventory()
    graph = OcclusionGraph(
        scene_id=scene_inventory.scene_id,
        inventory_sha256=digest_json(scene_inventory.model_dump(mode="json")),
        edges=[
            OcclusionEdge(
                foreground_id="pillow-ensemble",
                background_id="bed",
                confidence=0.99,
                evidence_frame_ids=["000064"],
                ordering_source="vlm",
                verification_status="observation_only",
            )
        ],
    )

    with pytest.raises(ValueError, match="geometry-verified depth relations"):
        build_layered_completion_plan(
            scene_inventory,
            graph,
            created_at=datetime(2026, 7, 17, tzinfo=UTC),
        )


def test_vlm_or_mask_overlap_cannot_claim_geometry_verified_ordering() -> None:
    for source in ("vlm", "mask_overlap"):
        with pytest.raises(ValidationError, match="requires depth or combined"):
            OcclusionEdge(
                foreground_id="pillow-ensemble",
                background_id="bed",
                confidence=0.99,
                evidence_frame_ids=["000064"],
                ordering_source=source,
                verification_status="geometry_verified",
            )


def test_missing_object_with_empty_needs_gets_fail_closed_completion_actions() -> None:
    scene_inventory = inventory().model_copy(
        update={"observations": [observation("missing-pillow", category="pillow", needs=[])]}
    )
    graph = OcclusionGraph(
        scene_id=scene_inventory.scene_id,
        inventory_sha256=digest_json(scene_inventory.model_dump(mode="json")),
    )

    plan = build_layered_completion_plan(
        scene_inventory,
        graph,
        created_at=datetime(2026, 7, 17, tzinfo=UTC),
    )

    assert [action.action for action in plan.rounds[0].actions] == [
        "segment",
        "lift_to_3d",
        "complete_object",
        "build_collider",
        "place_object",
        "clean_plate",
        "reinspect",
        "validate_round",
    ]
    complete_object = next(
        action for action in plan.rounds[0].actions if action.action == "complete_object"
    )
    assert complete_object.required_output_roles == [
        "unified_pbr_glb",
        "object_completion_report",
    ]
    assert complete_object.optional_output_roles == [
        "render_mesh",
        "collider",
        "object_gaussian",
        "object_point_cloud",
    ]
    assert "one validated PBR GLB" in complete_object.notes[0]
    build_collider = next(
        action for action in plan.rounds[0].actions if action.action == "build_collider"
    )
    assert build_collider.required_output_roles == ["collision_report"]
    assert "reuse" in build_collider.notes[0]
    assert "do not generate a collider proxy" in build_collider.notes[0]


def _unified_completed_object_assets_payload() -> dict[str, object]:
    return {
        "scene_id": "bedroom_4",
        "run_id": "completion-1",
        "created_at": "2026-07-17T00:00:00Z",
        "objects": [
            {
                "id": "pillow-01",
                "representation_mode": "unified_pbr_glb",
                "unified_pbr_glb": {
                    "uri": "artifact://pillow-01.glb",
                    "sha256": "a" * 64,
                    "size_bytes": 1024,
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
                },
                "collision_topology": "surface_bvh",
                "geometry_complete_verified": True,
                "completion_report_uri": "artifact://pillow-01/completion-report.json",
            }
        ],
    }


def _legacy_completed_object_assets_payload() -> dict[str, object]:
    return {
        "scene_id": "bedroom_4",
        "run_id": "completion-legacy-1",
        "created_at": "2026-07-17T00:00:00Z",
        "representation_policy": "mesh_first_optional_gaussian",
        "objects": [
            {
                "id": "pillow-legacy-01",
                "representation_mode": "separate_render_and_collider",
                "render_mesh": {
                    "uri": "artifact://pillow-legacy-01.glb",
                    "sha256": "a" * 64,
                    "size_bytes": 1024,
                    "role": "render_mesh",
                    "status": "validated",
                },
                "collider": {
                    "uri": "artifact://pillow-legacy-01-collider.glb",
                    "sha256": "b" * 64,
                    "size_bytes": 512,
                    "role": "collider",
                    "status": "validated",
                },
                "closed_surface_verified": True,
                "completion_report_uri": "artifact://pillow-legacy-01/completion-report.json",
            }
        ],
    }


def test_completed_object_manifest_accepts_unified_surface_without_gaussian_or_proxy() -> None:
    manifest = CompletedObjectAssetsManifest.model_validate(
        _unified_completed_object_assets_payload()
    )

    assert manifest.representation_policy == "unified_pbr_glb_preferred_optional_gaussian"
    completed = manifest.objects[0]
    assert completed.unified_pbr_glb is not None
    assert completed.unified_pbr_glb.provenance["watertight"] is False
    assert completed.collision_topology == "surface_bvh"
    assert completed.render_mesh is None
    assert completed.collider is None
    assert completed.object_gaussian is None
    assert completed.object_point_cloud is None


def test_completed_object_manifest_rejects_unified_surface_without_face_evidence() -> None:
    payload = _unified_completed_object_assets_payload()
    del payload["objects"][0]["unified_pbr_glb"]["provenance"]["faces"]

    with pytest.raises(ValidationError, match="face_count"):
        CompletedObjectAssetsManifest.model_validate(payload)


def test_completed_object_manifest_rejects_unified_surface_without_technical_gate() -> None:
    payload = _unified_completed_object_assets_payload()
    gates = payload["objects"][0]["unified_pbr_glb"]["provenance"]["technical_gates"]
    gates["pbr_material_present"] = False

    with pytest.raises(ValidationError, match="pbr_material_present"):
        CompletedObjectAssetsManifest.model_validate(payload)


def test_completed_object_manifest_rejects_unvalidated_optional_representation() -> None:
    payload = _unified_completed_object_assets_payload()
    payload["objects"][0]["object_gaussian"] = {
        "uri": "artifact://pillow-01.ply",
        "sha256": "c" * 64,
        "size_bytes": 2048,
        "role": "object_gaussian",
        "status": "candidate",
    }

    with pytest.raises(ValidationError, match="optional object representation"):
        CompletedObjectAssetsManifest.model_validate(payload)


def test_completed_object_manifest_requires_verified_complete_geometry() -> None:
    payload = _unified_completed_object_assets_payload()
    del payload["objects"][0]["geometry_complete_verified"]

    with pytest.raises(ValidationError, match="geometry_complete_verified"):
        CompletedObjectAssetsManifest.model_validate(payload)


def test_completed_object_manifest_rejects_mislabeled_unified_asset() -> None:
    payload = _unified_completed_object_assets_payload()
    payload["objects"][0]["unified_pbr_glb"]["role"] = "render_mesh"

    with pytest.raises(ValidationError, match="asset role"):
        CompletedObjectAssetsManifest.model_validate(payload)


def test_completed_object_manifest_rejects_closed_volume_for_nonwatertight_glb() -> None:
    payload = _unified_completed_object_assets_payload()
    payload["objects"][0]["collision_topology"] = "closed_volume"

    with pytest.raises(ValidationError, match="closed_volume requires explicit watertight"):
        CompletedObjectAssetsManifest.model_validate(payload)


def test_completed_object_manifest_accepts_watertight_closed_volume() -> None:
    payload = _unified_completed_object_assets_payload()
    payload["objects"][0]["collision_topology"] = "closed_volume"
    payload["objects"][0]["unified_pbr_glb"]["provenance"]["watertight"] = True

    manifest = CompletedObjectAssetsManifest.model_validate(payload)

    assert manifest.objects[0].collision_topology == "closed_volume"


def test_completed_object_manifest_rejects_mixed_unified_and_separate_assets() -> None:
    payload = _unified_completed_object_assets_payload()
    payload["objects"][0]["render_mesh"] = {
        "uri": "artifact://duplicate.glb",
        "sha256": "b" * 64,
        "size_bytes": 512,
        "role": "render_mesh",
        "status": "validated",
    }

    with pytest.raises(ValidationError, match="cannot be mixed with separate object assets"):
        CompletedObjectAssetsManifest.model_validate(payload)


def test_completed_object_manifest_preserves_legacy_separate_mode() -> None:
    manifest = CompletedObjectAssetsManifest.model_validate(
        _legacy_completed_object_assets_payload()
    )

    assert manifest.representation_policy == "mesh_first_optional_gaussian"
    completed = manifest.objects[0]
    assert completed.representation_mode == "separate_render_and_collider"
    assert completed.render_mesh is not None
    assert completed.collider is not None
    assert completed.unified_pbr_glb is None


def test_completed_object_manifest_requires_legacy_policy_for_separate_mode() -> None:
    payload = _legacy_completed_object_assets_payload()
    del payload["representation_policy"]

    with pytest.raises(ValidationError, match="requires unified_pbr_glb representation"):
        CompletedObjectAssetsManifest.model_validate(payload)


def test_legacy_closed_surface_flag_does_not_gate_unified_surface_bvh() -> None:
    payload = _unified_completed_object_assets_payload()
    payload["objects"][0]["closed_surface_verified"] = False

    manifest = CompletedObjectAssetsManifest.model_validate(payload)

    assert manifest.objects[0].geometry_complete_verified is True
    assert manifest.objects[0].closed_surface_verified is False


def test_geometry_complete_flag_supersedes_legacy_closed_surface_flag() -> None:
    payload = _legacy_completed_object_assets_payload()
    payload["objects"][0]["closed_surface_verified"] = False
    payload["objects"][0]["geometry_complete_verified"] = True

    manifest = CompletedObjectAssetsManifest.model_validate(payload)

    assert manifest.objects[0].geometry_complete_verified is True


def test_deeper_round_cannot_advance_before_previous_round_passes() -> None:
    action = CompletionAction(
        id="r01-01-validate-round",
        action="validate_round",
        idempotency_key=SHA,
        required_output_roles=["round_quality_report"],
    )
    with pytest.raises(ValidationError, match="cannot advance"):
        LayeredCompletionPlan(
            scene_id="bedroom_4",
            run_id="audit-1",
            created_at=datetime(2026, 7, 17, tzinfo=UTC),
            inventory_sha256=SHA,
            occlusion_graph_sha256=SHA,
            stop_conditions=["only background remains"],
            rounds=[
                CompletionRound(
                    index=1,
                    kind="object_layer",
                    target_ids=["pillow-ensemble"],
                    input_clean_plate_round=0,
                    actions=[action],
                    status="failed",
                ),
                CompletionRound(
                    index=2,
                    kind="final_background",
                    target_ids=[],
                    input_clean_plate_round=1,
                    actions=[
                        CompletionAction(
                            id="r02-01-rebuild-background",
                            action="rebuild_background",
                            idempotency_key=SHA,
                            synthetic_output=True,
                        )
                    ],
                    status="running",
                ),
            ],
        )


def test_terminal_passed_requires_every_round_to_pass() -> None:
    action = CompletionAction(
        id="r01-01-rebuild-background",
        action="rebuild_background",
        idempotency_key=SHA,
    )
    with pytest.raises(ValidationError, match="requires every completion round to pass"):
        LayeredCompletionPlan(
            scene_id="bedroom_4",
            run_id="audit-1",
            created_at=datetime(2026, 7, 17, tzinfo=UTC),
            inventory_sha256=SHA,
            occlusion_graph_sha256=SHA,
            stop_conditions=["only background remains"],
            terminal_status="passed",
            rounds=[
                CompletionRound(
                    index=1,
                    kind="final_background",
                    target_ids=[],
                    input_clean_plate_round=0,
                    actions=[action],
                )
            ],
        )


def test_all_passed_rounds_require_terminal_passed() -> None:
    action = CompletionAction(
        id="r01-01-rebuild-background",
        action="rebuild_background",
        idempotency_key=SHA,
    )
    with pytest.raises(ValidationError, match="require terminal_status='passed'"):
        LayeredCompletionPlan(
            scene_id="bedroom_4",
            run_id="audit-1",
            created_at=datetime(2026, 7, 17, tzinfo=UTC),
            inventory_sha256=SHA,
            occlusion_graph_sha256=SHA,
            stop_conditions=["only background remains"],
            rounds=[
                CompletionRound(
                    index=1,
                    kind="final_background",
                    target_ids=[],
                    input_clean_plate_round=0,
                    actions=[action],
                    status="passed",
                    output_clean_plate_uri="artifact://clean/round-1",
                    quality_report_uri="artifact://reports/round-1.json",
                )
            ],
        )


def test_modeled_complete_observation_requires_identity_match() -> None:
    with pytest.raises(ValidationError, match="matched_object_ids"):
        observation(
            "nightstand",
            category="nightstand",
            coverage="modeled_complete",
            needs=[],
        )


def test_localized_name_requires_at_least_one_language() -> None:
    with pytest.raises(ValidationError):
        LocalizedText()


def test_geometry_review_cannot_accept_failed_technical_gate() -> None:
    with pytest.raises(ValidationError, match="accept is forbidden"):
        GeometryReview(
            object_id="sam3_pillow_front",
            attempt=1,
            created_at=datetime(2026, 7, 17, tzinfo=UTC),
            provider="Qwen2.5-VL",
            model="Qwen2.5-VL-3B-Instruct",
            source_asset_sha256=SHA,
            turntable_sha256=SHA,
            technical_gates={"backside_nonempty": False},
            decision="accept",
            raw_response_sha256=SHA,
        )


def test_retry_review_requires_detailed_remediation_prompt() -> None:
    issue = GeometryReviewIssue(
        issue_type="implausible_thickness",
        severity="blocking",
        evidence_view_ids=["right"],
        explanation={"en": "The door is too thick.", "zh": "门体过厚。"},
        retry_prompt_instruction="Generate a thin door leaf with a realistic depth-to-width ratio.",
    )
    with pytest.raises(ValidationError, match="retry requires"):
        GeometryReview(
            object_id="door-01",
            attempt=1,
            created_at=datetime(2026, 7, 17, tzinfo=UTC),
            provider="Qwen2.5-VL",
            model="Qwen2.5-VL-3B-Instruct",
            source_asset_sha256=SHA,
            turntable_sha256=SHA,
            technical_gates={"backside_nonempty": True},
            decision="retry",
            issues=[issue],
            raw_response_sha256=SHA,
        )


def _trellis2_receipt_payload(*, sha256: str = SHA) -> dict[str, object]:
    return {
        "kind": "video2world.trellis2_mesh_first_asset",
        "status": "technical_passed_visual_pending",
        "technical_audit": {"technical_status": "passed"},
        "outputs": {
            "unified_pbr_glb": {
                "path": "/artifacts/pillow-front/asset_pbr.glb",
                "sha256": sha256,
                "size_bytes": 4096,
                "media_type": "model/gltf-binary",
                "role": "unified_pbr_glb",
                "status": "candidate",
                "collision_topology": "surface_bvh",
                "provenance": {
                    "faces": 97082,
                    "face_count": 97082,
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
        },
    }


def _accepted_geometry_review(
    *,
    object_id: str = "pillow-front",
    sha256: str = SHA,
) -> GeometryReview:
    return GeometryReview(
        object_id=object_id,
        attempt=1,
        created_at=datetime(2026, 7, 17, tzinfo=UTC),
        provider="Qwen2.5-VL",
        model="Qwen2.5-VL-3B-Instruct",
        source_asset_sha256=sha256,
        turntable_sha256=SHA,
        technical_gates={
            "front_visible": True,
            "backside_nonempty": True,
            "top_bottom_nonempty": True,
            "scene_fit_plausible": True,
        },
        decision="accept",
        raw_response_sha256=SHA,
    )


def test_trellis2_completion_promotion_requires_accepted_visual_review() -> None:
    completed = promote_trellis2_unified_pbr_glb_completion(
        object_id="pillow-front",
        trellis2_receipt=_trellis2_receipt_payload(),
        geometry_review=_accepted_geometry_review(),
        completion_report_uri="artifact://pillow-front/object-completion-report.json",
    )

    assert completed.representation_mode == "unified_pbr_glb"
    assert completed.geometry_complete_verified is True
    assert completed.collision_topology == "surface_bvh"
    assert completed.unified_pbr_glb is not None
    assert completed.unified_pbr_glb.status == "validated"
    assert completed.unified_pbr_glb.provenance["faces"] == 97082


def test_trellis2_completion_promotion_rejects_pending_or_retry_review() -> None:
    issue = GeometryReviewIssue(
        issue_type="missing_back_surface",
        severity="blocking",
        evidence_view_ids=["back"],
        explanation={"en": "Back face is missing.", "zh": "背面缺失。"},
        retry_prompt_instruction="Regenerate a full pillow with nonempty back and side views.",
    )
    review = GeometryReview(
        object_id="pillow-front",
        attempt=1,
        created_at=datetime(2026, 7, 17, tzinfo=UTC),
        provider="Qwen2.5-VL",
        model="Qwen2.5-VL-3B-Instruct",
        source_asset_sha256=SHA,
        turntable_sha256=SHA,
        technical_gates={"backside_nonempty": False},
        decision="retry",
        issues=[issue],
        retry_prompt="Regenerate a complete pillow with all six views nonempty.",
        raw_response_sha256=SHA,
    )

    with pytest.raises(ValueError, match="geometry review must accept"):
        promote_trellis2_unified_pbr_glb_completion(
            object_id="pillow-front",
            trellis2_receipt=_trellis2_receipt_payload(),
            geometry_review=review,
            completion_report_uri="artifact://pillow-front/object-completion-report.json",
        )


def test_trellis2_completion_promotion_rejects_review_for_different_asset() -> None:
    with pytest.raises(ValueError, match="source_asset_sha256"):
        promote_trellis2_unified_pbr_glb_completion(
            object_id="pillow-front",
            trellis2_receipt=_trellis2_receipt_payload(sha256="b" * 64),
            geometry_review=_accepted_geometry_review(sha256="c" * 64),
            completion_report_uri="artifact://pillow-front/object-completion-report.json",
        )


def _object_completion_report_payload() -> dict[str, object]:
    completed = promote_trellis2_unified_pbr_glb_completion(
        object_id="pillow-front",
        trellis2_receipt=_trellis2_receipt_payload(),
        geometry_review=_accepted_geometry_review(),
        completion_report_uri="artifact://pillow-front/object-completion-report.json",
        asset_uri="artifact://pillow-front/asset_pbr.glb",
    )
    return {
        "kind": "video2world.object_completion_report",
        "object_id": "pillow-front",
        "created_at": "2026-07-17T00:00:00Z",
        "status": "accepted",
        "trellis2_receipt": _trellis2_receipt_payload(),
        "geometry_review": _accepted_geometry_review().model_dump(mode="json"),
        "completed_asset": completed.model_dump(mode="json"),
    }


def test_object_completion_report_binds_trellis2_receipt_review_and_asset() -> None:
    report = ObjectCompletionReport.model_validate(_object_completion_report_payload())

    assert report.object_id == "pillow-front"
    assert report.completed_asset.unified_pbr_glb is not None
    assert report.completed_asset.unified_pbr_glb.status == "validated"


def test_object_completion_report_rejects_asset_that_was_not_promoted_from_review() -> None:
    payload = _object_completion_report_payload()
    payload["completed_asset"]["unified_pbr_glb"]["sha256"] = "b" * 64

    with pytest.raises(ValidationError, match="completed_asset does not match"):
        ObjectCompletionReport.model_validate(payload)


def test_completion_history_is_bounded_and_stops_after_accept() -> None:
    attempts = [
        CompletionAttempt(
            attempt=1,
            seed=1,
            prompt_sha256=SHA,
            output_asset_sha256=SHA,
            review_uri="artifact://review-1.json",
            decision="accept",
        ),
        CompletionAttempt(
            attempt=2,
            seed=2,
            prompt_sha256=SHA,
            output_asset_sha256=SHA,
            review_uri="artifact://review-2.json",
            decision="retry",
        ),
    ]
    with pytest.raises(ValidationError, match="accepted candidate"):
        ObjectCompletionHistory(
            object_id="sam3_pillow_front",
            attempts=attempts,
            terminal_status="in_progress",
        )
