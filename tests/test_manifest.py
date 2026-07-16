from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from video2world.models import (
    Bounds3D,
    CoordinateSystem,
    GateRecord,
    InteractionPolicy,
    LocalizedText,
    QualityGates,
    SceneTransform,
    WorldManifest,
    WorldObject,
    world_manifest_json_schema,
)
from video2world.validation import validate_world_manifest


def test_committed_json_schema_matches_pydantic_model() -> None:
    schema_path = Path(__file__).parents[1] / "schemas" / "world-manifest.schema.json"
    assert json.loads(schema_path.read_text(encoding="utf-8")) == world_manifest_json_schema()


def test_manifest_verifies_local_content_hashes(
    tmp_path: Path,
    sample_manifest: WorldManifest,
) -> None:
    manifest_path = tmp_path / "world.json"
    manifest_path.write_text(sample_manifest.model_dump_json(indent=2), encoding="utf-8")
    result = validate_world_manifest(manifest_path)
    assert result["valid"] is True
    assert result["checked_assets"] == 3

    Path(sample_manifest.scene.visual.uri).write_bytes(b"changed")
    invalid = validate_world_manifest(manifest_path)
    assert invalid["valid"] is False
    assert {issue["error"] for issue in invalid["issues"]} >= {
        "sha256 mismatch",
        "size_bytes mismatch",
    }


def test_scoped_identity_is_checked_against_world(sample_manifest: WorldManifest) -> None:
    payload = sample_manifest.model_dump(mode="json")
    payload["objects"][0]["scoped_id"] = "other::run::bed01"
    with pytest.raises(ValidationError, match="scoped_id"):
        WorldManifest.model_validate(payload)


def test_manifest_timestamp_must_be_timezone_aware(sample_manifest: WorldManifest) -> None:
    payload = sample_manifest.model_dump(mode="json")
    payload["created_at"] = datetime(2026, 7, 16).isoformat()
    with pytest.raises(ValidationError, match="explicit timezone"):
        WorldManifest.model_validate(payload)


def test_manifest_rejects_non_finite_geometry(sample_manifest: WorldManifest) -> None:
    payload = sample_manifest.model_dump(mode="json")
    payload["objects"][0]["bbox_scene"]["minimum"][0] = float("nan")
    with pytest.raises(ValidationError, match="finite number"):
        WorldManifest.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("bbox_scene", "bbox is not in the scene frame"),
        ("obb_scene", "OBB is not in the scene frame"),
        ("transform_scene_from_asset", "transform does not target the scene frame"),
    ],
)
def test_object_geometry_must_use_the_scene_frame(
    sample_manifest: WorldManifest,
    field: str,
    message: str,
) -> None:
    payload = sample_manifest.model_dump(mode="json")
    object_payload = payload["objects"][0]
    if field == "obb_scene":
        object_payload[field] = {
            "frame_id": "other",
            "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            "extents": [1, 1, 1],
        }
    elif field == "transform_scene_from_asset":
        object_payload[field] = {
            "from_frame": "asset",
            "to_frame": "other",
            "matrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            "pivot_scene": [0, 0, 0],
            "scale_xyz": [1, 1, 1],
        }
    else:
        object_payload[field]["frame_id"] = "other"
    with pytest.raises(ValidationError, match=message):
        WorldManifest.model_validate(payload)


def test_non_metric_scene_scale_is_explicit() -> None:
    coordinate_system = CoordinateSystem(
        frame_id="pgsr_native",
        up_axis="-Y",
        handedness="right",
        units="scene_scale_not_metric",
    )
    assert coordinate_system.metric_scale is None
    with pytest.raises(ValidationError, match="cannot declare a metric scale"):
        CoordinateSystem(
            frame_id="pgsr_native",
            up_axis="-Y",
            handedness="right",
            units="scene_scale_not_metric",
            metric_scale=1.0,
        )


def test_interactive_objects_require_assets_and_passed_gates() -> None:
    with pytest.raises(ValidationError, match="interactive object is missing"):
        WorldObject(
            id="plant01",
            source_run_id="run",
            scoped_id="scene::run::plant01",
            name=LocalizedText(en="plant"),
            category="plant",
            interaction=InteractionPolicy(selectable=True, double_click_action="spin_360"),
        )


def _visual_only_object(sample_manifest: WorldManifest) -> WorldObject:
    frame_id = sample_manifest.scene.coordinate_system.frame_id
    return WorldObject(
        id="pillow01",
        source_run_id="pillow-run",
        scoped_id="bedroom_4::pillow-run::pillow01",
        name=LocalizedText(zh="枕头", en="pillow"),
        category="pillow",
        bbox_scene=Bounds3D(
            frame_id=frame_id,
            minimum=(-1, 0, -1),
            maximum=(1, 1, 1),
        ),
        visual=sample_manifest.scene.visual,
        transform_scene_from_asset=SceneTransform(
            from_frame="pillow_points",
            to_frame=frame_id,
            matrix=(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1),
            pivot_scene=(0, 0.5, 0),
        ),
        quality_gates=QualityGates(
            file=GateRecord(status="passed"),
            semantic=GateRecord(status="passed"),
            alignment=GateRecord(status="passed"),
            collision=GateRecord(status="not_tested", reason="no collider exists"),
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


def test_visual_only_interaction_allows_no_collider_and_rejects_collision_claims(
    sample_manifest: WorldManifest,
) -> None:
    pillow = _visual_only_object(sample_manifest)
    assert pillow.collider is None
    assert pillow.quality_gates.collision.status == "not_tested"
    assert pillow.interaction.drag_action == "rotate_yaw"

    with_collider = pillow.model_dump(mode="json")
    with_collider["collider"] = sample_manifest.scene.collider.model_dump(mode="json")
    with pytest.raises(ValidationError, match="must not declare a collider"):
        WorldObject.model_validate(with_collider)

    passed_collision = pillow.model_dump(mode="json")
    passed_collision["quality_gates"]["collision"] = {"status": "passed"}
    with pytest.raises(ValidationError, match="must not pass the collision gate"):
        WorldObject.model_validate(passed_collision)


def test_collision_enabled_interaction_requires_collider_and_passed_gate(
    sample_manifest: WorldManifest,
) -> None:
    pillow = _visual_only_object(sample_manifest)
    payload = pillow.model_dump(mode="json")
    payload["interaction"].update({"collision_enabled": True, "physics_mode": "kinematic"})
    with pytest.raises(ValidationError, match="missing: collider"):
        WorldObject.model_validate(payload)

    payload["collider"] = sample_manifest.scene.collider.model_dump(mode="json")
    with pytest.raises(ValidationError, match="requires a passed collision gate"):
        WorldObject.model_validate(payload)

    payload["quality_gates"]["collision"] = {"status": "passed"}
    collidable = WorldObject.model_validate(payload)
    assert collidable.collider is not None
    assert collidable.interaction.collision_enabled is True


def test_collision_only_policy_still_enforces_the_interactive_contract() -> None:
    with pytest.raises(ValidationError, match="interactive object is missing"):
        WorldObject(
            id="static-obstacle",
            source_run_id="run",
            scoped_id="scene::run::static-obstacle",
            name=LocalizedText(en="static obstacle"),
            category="obstacle",
            interaction=InteractionPolicy(
                collision_enabled=True,
                physics_mode="static",
            ),
        )
