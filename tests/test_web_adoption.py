from __future__ import annotations

import json
from pathlib import Path

import pytest

from video2world.cli import main
from video2world.hashing import digest_json, digest_path
from video2world.validation import load_world_manifest
from video2world.web_adoption import (
    adopt_web_manifest,
    adopt_web_manifest_file,
    bind_scene_command_web_manifest,
)


def _web_manifest() -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "contract": "video2world-web-manifest-1.0.0",
        "version": "bedroom4-production-20260716",
        "sourceWorld": {
            "worldId": "bedroom_4",
            "runId": "bedroom4-production-20260716",
            "adoptionMode": "legacy_web_assets_adopted_then_canonical_manifest_validated",
        },
        "productionBuild": {"createdAt": "2026-07-16T15:16:48.625Z"},
        "coordinateSystem": {"worldUp": [0, -1, 0]},
        "assets": {
            "visual": {
                "id": "scene_visual",
                "fileType": "ply",
                "size": 100,
                "sha256": "1" * 64,
                "url": "./scene.ply",
                "bbox": {"min": [-5, -3, -2], "max": [5, 3, 8]},
            },
            "colliderStaticCarved": {
                "id": "scene_collider",
                "fileType": "glb",
                "size": 80,
                "sha256": "2" * 64,
                "url": "./scene.glb",
            },
        },
        "interactiveObjects": [
            {"id": "bed01", "category": "bed", "name": "床 / Bed"},
            {
                "id": "pillow01",
                "category": "pillow",
                "parentObjectId": "bed01",
                "semanticGranularity": "independent_child_asset",
                "movesWithParent": True,
                "independentlyMovable": True,
            },
        ],
        "sceneKnowledge": {
            "objects": [
                {
                    "id": "bed01",
                    "name": {"zh": "床", "en": "bed"},
                    "category": "bed",
                    "aliases": ["床铺"],
                    "bbox": {"min": [-2, 0, 1], "max": [2, 2, 5]},
                },
                {
                    "id": "pillow01",
                    "name": "白色枕头 / white pillow",
                    "category": "pillow",
                    "aliases": ["枕头", "pillow"],
                    "description": {
                        "appearance": {"zh": "浅白色长方形软枕。", "en": "A light white pillow."},
                        "location": {"zh": "位于床上。", "en": "On the bed."},
                    },
                    "bbox": {"min": [-1, 0, 2], "max": [1, 1, 4]},
                },
            ],
            "relations": [
                {
                    "subject": "pillow01",
                    "predicate": "OnTopOf",
                    "object": "bed01",
                    "confidence": 0.97,
                    "verified": True,
                    "evidence": "frame64",
                }
            ],
        },
    }


def test_adoption_preserves_scene_knowledge_hierarchy_and_hashes() -> None:
    source = _web_manifest()
    manifest = adopt_web_manifest(source)

    assert manifest.manifest_status == "draft"
    assert manifest.scene.coordinate_system.up_axis == "-Y"
    assert manifest.scene.visual.sha256 == "1" * 64
    assert manifest.scene.collider.sha256 == "2" * 64
    assert manifest.scene.visual.uri == "web-bundle://scene_visual"
    assert manifest.scene.visual.status == "candidate"
    assert manifest.scene.collider.status == "candidate"
    assert manifest.scene.semantic_visual.status == "candidate"
    source_without_deployment_version = {
        key: value for key, value in source.items() if key != "version"
    }
    assert manifest.provenance.config_sha256 == digest_json(source_without_deployment_version)
    assert [item.id for item in manifest.objects] == ["bed01", "pillow01"]
    pillow = manifest.objects[1]
    assert pillow.parent_object_id == "bed01"
    assert pillow.moves_with_parent is True
    assert pillow.independently_movable is True
    assert pillow.description.appearance is not None
    assert pillow.description.appearance.zh == "浅白色长方形软枕。"
    assert pillow.relations[0].predicate == "on_top_of"
    assert pillow.relations[0].target_object_id == "bedroom_4::bedroom4-production-20260716::bed01"


def test_adoption_fails_closed_for_unknown_parent() -> None:
    source = _web_manifest()
    source["interactiveObjects"][1]["parentObjectId"] = "missing-bed"  # type: ignore[index]

    with pytest.raises(ValueError, match="unknown parent"):
        adopt_web_manifest(source)


def test_adoption_rejects_wrong_contract_duplicate_ids_and_string_booleans() -> None:
    wrong_contract = _web_manifest()
    wrong_contract["contract"] = "some-other-contract"
    with pytest.raises(ValueError, match="contract must equal"):
        adopt_web_manifest(wrong_contract)

    duplicate = _web_manifest()
    knowledge = duplicate["sceneKnowledge"]  # type: ignore[assignment]
    knowledge["objects"].append(knowledge["objects"][0])  # type: ignore[index,union-attr]
    with pytest.raises(ValueError, match="duplicate object id"):
        adopt_web_manifest(duplicate)

    string_boolean = _web_manifest()
    interactive = string_boolean["interactiveObjects"]  # type: ignore[assignment]
    interactive[0]["independentlyMovable"] = "false"  # type: ignore[index]
    with pytest.raises(ValueError, match="must be a boolean"):
        adopt_web_manifest(string_boolean)


def test_chunked_deployed_asset_does_not_pair_source_path_with_deployed_hash() -> None:
    source = _web_manifest()
    assets = source["assets"]  # type: ignore[assignment]
    visual = assets["visual"]  # type: ignore[index]
    visual.pop("url")  # type: ignore[union-attr]
    visual["sourcePath"] = "artifact://original-scene.ply"  # type: ignore[index]
    visual["parts"] = [  # type: ignore[index]
        {"url": "./part0", "size": 40, "sha256": "3" * 64},
        {"url": "./part1", "size": 60, "sha256": "4" * 64},
    ]

    manifest = adopt_web_manifest(source)

    assert manifest.scene.visual.uri == "web-bundle://scene_visual"
    assert manifest.scene.visual.sha256 == "1" * 64
    assert manifest.scene.visual.status == "candidate"
    assert manifest.scene.visual.provenance["source_path"] == ("artifact://original-scene.ply")
    assert manifest.scene.visual.provenance["hash_scope"] == ("reassembled_deployed_web_asset")


def test_canonical_backlink_and_service_do_not_change_adopted_manifest() -> None:
    source = _web_manifest()
    first = adopt_web_manifest(source)
    canonical_hash = digest_json(first.model_dump(mode="json"))
    source_world = source["sourceWorld"]  # type: ignore[assignment]
    source_world["manifestSha256"] = canonical_hash  # type: ignore[index]
    source["sceneCommandService"] = {
        "contract": "video2world-scene-command-service-1.0.0",
        "endpoint": "/v1/scene-commands/submit",
    }

    second = adopt_web_manifest(source)

    assert second == first


def test_file_adoption_rejects_duplicate_json_and_source_destination_alias(
    tmp_path: Path,
) -> None:
    duplicate_json = tmp_path / "duplicate.json"
    duplicate_json.write_text(
        '{"schemaVersion":1,"schemaVersion":1}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        adopt_web_manifest_file(duplicate_json, tmp_path / "out.json")

    source_path = tmp_path / "web.json"
    source_path.write_text(json.dumps(_web_manifest()), encoding="utf-8")
    with pytest.raises(ValueError, match="source and canonical sidecar destination must differ"):
        adopt_web_manifest_file(source_path, source_path)


def test_real_bedroom4_web_manifest_adopts_without_fabricating_relation_only_bed() -> None:
    root = Path(__file__).resolve().parents[1]
    source = json.loads(
        (root / "examples/bedroom4/manifest.production.json").read_text(encoding="utf-8")
    )

    manifest = adopt_web_manifest(source)

    assert manifest.world_id == "bedroom_4"
    assert manifest.run_id == "bedroom4-production-20260716"
    assert len(manifest.objects) == 5
    assert "sam3_bed_01" not in {item.id for item in manifest.objects}
    assert any(
        "Relation-only target IDs were not fabricated as objects: sam3_bed_01" in note
        for note in manifest.provenance.notes
    )
    assert manifest.scene.visual.uri.startswith("web-bundle://")
    assert manifest.scene.collider.uri.startswith("web-bundle://")
    assert manifest.scene.visual.status == "candidate"
    assert manifest.scene.collider.status == "candidate"


def test_web_manifest_adopt_cli_writes_canonical_sidecar(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source_path = tmp_path / "web.json"
    output_path = tmp_path / "canonical.json"
    source_path.write_text(json.dumps(_web_manifest()), encoding="utf-8")

    assert (
        main(
            [
                "web-manifest-adopt",
                str(source_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["manifest_status"] == "draft"
    assert receipt["object_count"] == 2
    normalized = adopt_web_manifest(_web_manifest()).model_dump(mode="json")
    assert receipt["canonical_manifest_sha256"] == digest_json(normalized)
    assert receipt["sidecar_file_sha256"] == digest_path(output_path).sha256
    assert receipt["canonical_manifest_sha256"] == digest_json(
        load_world_manifest(output_path).model_dump(mode="json")
    )


def test_scene_command_binding_writes_new_stable_derived_manifest(tmp_path: Path) -> None:
    source_path = tmp_path / "web.json"
    canonical_path = tmp_path / "canonical.json"
    bound_path = tmp_path / "web.bound.json"
    source_path.write_text(json.dumps(_web_manifest()), encoding="utf-8")
    canonical = adopt_web_manifest_file(source_path, canonical_path)

    bound, canonical_hash = bind_scene_command_web_manifest(
        web_manifest_source=source_path,
        canonical_manifest_path=canonical_path,
        destination=bound_path,
        endpoint="http://127.0.0.1:8765/v1/scene-commands/submit",
        derived_version="bedroom4-local-command-qa",
    )

    assert canonical_hash == digest_json(canonical.model_dump(mode="json"))
    assert bound["sourceWorld"]["manifestSha256"] == canonical_hash
    assert bound["sceneCommandService"] == {
        "contract": "video2world-scene-command-service-1.0.0",
        "endpoint": "http://127.0.0.1:8765/v1/scene-commands/submit",
    }
    assert adopt_web_manifest(bound) == canonical
    with pytest.raises(ValueError, match="restricted to localhost"):
        bind_scene_command_web_manifest(
            web_manifest_source=source_path,
            canonical_manifest_path=canonical_path,
            destination=tmp_path / "unsafe.json",
            endpoint="http://example.com/v1/scene-commands/submit",
            derived_version="unsafe",
        )
    with pytest.raises(ValueError, match="new destination"):
        bind_scene_command_web_manifest(
            web_manifest_source=source_path,
            canonical_manifest_path=canonical_path,
            destination=source_path,
            endpoint="/v1/scene-commands/submit",
            derived_version="in-place",
        )
