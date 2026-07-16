from __future__ import annotations

from video2world.models import (
    Bounds3D,
    DescriptionEvidence,
    GateRecord,
    InteractionPolicy,
    LocalizedText,
    QualityGates,
    Relation,
    SceneTransform,
    WorldManifest,
    WorldObject,
)
from video2world.query import query_world


def test_chinese_location_and_focus_bbox(sample_manifest: WorldManifest) -> None:
    result = query_world(sample_manifest, "床在哪里?")
    assert result.status == "resolved"
    assert result.intent == "location"
    assert result.resolved_object_id == "bed01"
    assert result.resolved_scoped_id == "bedroom_4::holi-fresh::bed01"
    assert result.answer == "位于房间后侧靠墙处。"
    assert result.focus_bbox is not None


def test_shared_alias_fails_closed_without_disambiguation(sample_manifest: WorldManifest) -> None:
    result = query_world(sample_manifest, "植物在哪里?")
    assert result.status == "ambiguous"
    assert result.resolved_object_id is None
    assert result.focus_bbox is None
    assert result.candidate_ids == [
        "bedroom_4::holi-fresh::plant01",
        "bedroom_4::holi-fresh::plant02",
    ]


def test_chinese_ordinal_resolves_verified_relation(sample_manifest: WorldManifest) -> None:
    result = query_world(sample_manifest, "第二株植物在哪里?")
    assert result.status == "resolved"
    assert result.resolved_object_id == "plant02"
    assert result.answer == "在床附近"
    assert result.evidence == ["verified_relations"]


def test_english_numbered_name_resolves_appearance(sample_manifest: WorldManifest) -> None:
    result = query_world(sample_manifest, "What does plant 1 look like?")
    assert result.status == "resolved"
    assert result.language == "en"
    assert result.intent == "appearance"
    assert result.resolved_object_id == "plant01"
    assert "broad-leaf" in result.answer


def test_missing_evidence_and_unsupported_intent_are_explicit(
    sample_manifest: WorldManifest,
) -> None:
    unavailable = query_world(sample_manifest, "桌子的外观是什么?")
    assert unavailable.status == "unavailable"
    assert unavailable.resolved_object_id == "table01"
    assert unavailable.focus_bbox is not None

    unsupported = query_world(sample_manifest, "桌子能做什么?")
    assert unsupported.status == "unsupported"
    assert unsupported.resolved_object_id == "table01"
    assert unsupported.answer is None


def test_unknown_object_never_returns_a_bbox(sample_manifest: WorldManifest) -> None:
    result = query_world(sample_manifest, "枕头在哪里?")
    assert result.status == "not_found"
    assert result.candidate_ids == []
    assert result.focus_bbox is None


def test_visual_only_interactive_pillow_resolves_without_claiming_a_collider(
    sample_manifest: WorldManifest,
) -> None:
    bed = sample_manifest.objects[0]
    pillow = WorldObject(
        id="sam3_pillow_01",
        source_run_id="holi-pillow-delta",
        scoped_id="bedroom_4::holi-pillow-delta::sam3_pillow_01",
        name=LocalizedText(zh="枕头组合", en="pillow ensemble"),
        category="pillow",
        aliases=["枕头", "pillow"],
        description=DescriptionEvidence(
            appearance=LocalizedText(
                zh="三只相接的枕头, 两只较大的浅绿色枕头在后方。",
                en="Three touching pillows with two larger pale-green pillows behind.",
            )
        ),
        bbox_scene=Bounds3D(
            frame_id=sample_manifest.scene.coordinate_system.frame_id,
            minimum=(-0.8, 0.3, -1.5),
            maximum=(0.8, 0.9, -0.5),
        ),
        visual=sample_manifest.scene.visual,
        transform_scene_from_asset=SceneTransform(
            from_frame="pillow_rgb_points",
            to_frame=sample_manifest.scene.coordinate_system.frame_id,
            matrix=(
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
            ),
            pivot_scene=(0, 0.6, -1),
        ),
        relations=[
            Relation(
                predicate="on_top_of",
                target_object_id=bed.scoped_id,
                confidence=0.97,
                verified=True,
            )
        ],
        quality_gates=QualityGates(
            file=GateRecord(status="passed"),
            semantic=GateRecord(status="passed"),
            alignment=GateRecord(status="passed"),
            collision=GateRecord(status="not_tested", reason="visual-only"),
            visual=GateRecord(status="passed"),
        ),
        interaction=InteractionPolicy(
            selectable=True,
            double_click_action="spin_360",
            drag_action="rotate_yaw",
            collision_enabled=False,
            physics_mode="none",
        ),
    )
    manifest = sample_manifest.model_copy(
        update={"objects": [*sample_manifest.objects, pillow]}, deep=True
    )

    location = query_world(manifest, "枕头在哪里?")
    assert location.status == "resolved"
    assert location.answer == "在床上方"
    assert location.focus_bbox == pillow.bbox_scene

    appearance = query_world(manifest, "枕头长什么样子?")
    assert appearance.status == "resolved"
    assert "三只相接" in appearance.answer
    assert appearance.focus_bbox == pillow.bbox_scene
    assert pillow.collider is None
    assert pillow.interaction.selectable is True
    assert pillow.interaction.drag_action == "rotate_yaw"
    assert pillow.interaction.double_click_action == "spin_360"
    assert pillow.quality_gates.collision.status == "not_tested"
