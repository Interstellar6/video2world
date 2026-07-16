from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from video2world.hashing import digest_path
from video2world.models import (
    AssetRef,
    Bounds3D,
    CoordinateSystem,
    DescriptionEvidence,
    LocalizedText,
    ProvenanceRecord,
    Relation,
    SceneLayer,
    WorldManifest,
    WorldObject,
)


def asset(path: Path, role: str, *, status: str = "validated") -> AssetRef:
    digest = digest_path(path)
    return AssetRef(
        uri=str(path),
        sha256=digest.sha256,
        size_bytes=digest.size_bytes,
        role=role,
        status=status,
    )


@pytest.fixture
def sample_manifest(tmp_path: Path) -> WorldManifest:
    scene_visual = tmp_path / "scene.ply"
    scene_collider = tmp_path / "scene.glb"
    scene_semantic = tmp_path / "semantic.ply"
    scene_visual.write_bytes(b"gaussian-scene")
    scene_collider.write_bytes(b"collision-scene")
    scene_semantic.write_bytes(b"semantic-scene")
    frame_id = "pgsr_native"

    bed = WorldObject(
        id="bed01",
        source_run_id="holi-fresh",
        scoped_id="bedroom_4::holi-fresh::bed01",
        name=LocalizedText(zh="床", en="bed"),
        category="bed",
        aliases=["床铺"],
        description=DescriptionEvidence(
            appearance=LocalizedText(
                zh="一张带浅色床品的双人床。", en="A double bed with light bedding."
            ),
            location=LocalizedText(zh="位于房间后侧靠墙处。", en="Against the rear wall."),
        ),
        bbox_scene=Bounds3D(frame_id=frame_id, minimum=(-1, 0, -2), maximum=(1, 1, 0)),
    )
    plant_1 = WorldObject(
        id="plant01",
        source_run_id="holi-fresh",
        scoped_id="bedroom_4::holi-fresh::plant01",
        name=LocalizedText(zh="植物一", en="plant 1"),
        category="plant",
        aliases=["植物", "plant"],
        description=DescriptionEvidence(
            appearance=LocalizedText(zh="绿色阔叶盆栽。", en="A green broad-leaf potted plant.")
        ),
        bbox_scene=Bounds3D(frame_id=frame_id, minimum=(1, 0, 1), maximum=(2, 2, 2)),
    )
    plant_2 = WorldObject(
        id="plant02",
        source_run_id="holi-fresh",
        scoped_id="bedroom_4::holi-fresh::plant02",
        name=LocalizedText(zh="植物二", en="plant 2"),
        category="plant",
        aliases=["植物", "plant"],
        description=DescriptionEvidence(
            appearance=LocalizedText(zh="细长叶片的盆栽。", en="A potted plant with narrow leaves.")
        ),
        bbox_scene=Bounds3D(frame_id=frame_id, minimum=(-2, 0, 1), maximum=(-1, 1.5, 2)),
        relations=[
            Relation(
                predicate="near",
                target_object_id=bed.scoped_id,
                confidence=0.9,
                verified=True,
            )
        ],
    )
    table = WorldObject(
        id="table01",
        source_run_id="holi-fresh",
        scoped_id="bedroom_4::holi-fresh::table01",
        name=LocalizedText(zh="桌子", en="table"),
        category="table",
        bbox_scene=Bounds3D(frame_id=frame_id, minimum=(0, 0, 1), maximum=(1, 1, 2)),
    )
    return WorldManifest(
        manifest_status="validated",
        world_id="bedroom_4",
        run_id="world-run-01",
        created_at=datetime.now(UTC),
        scene=SceneLayer(
            coordinate_system=CoordinateSystem(
                frame_id=frame_id,
                up_axis="-Y",
                handedness="right",
                units="native",
            ),
            visual=asset(scene_visual, "scene_gaussian"),
            collider=asset(scene_collider, "scene_collider"),
            semantic_visual=asset(scene_semantic, "semantic_gaussian"),
        ),
        objects=[bed, plant_1, plant_2, table],
        provenance=ProvenanceRecord(config_sha256="0" * 64),
    )
