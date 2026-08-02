from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from video2world.adapters import get_adapter
from video2world.completion import CompletedObjectAsset, CompletedObjectAssetsManifest
from video2world.config import load_run_config
from video2world.modeling import (
    ComponentAssemblyRecord,
    ComponentContract,
    ComponentMaskRecord,
    ImageCompletionRecord,
    ObjectCompletionCandidate,
    ObjectProposal,
    QwenObjectContract,
)
from video2world.models import (
    AssetRef,
    Bounds2D,
    Bounds3D,
    GateRecord,
    InteractionPolicy,
    LocalizedText,
    PhysicalProperties,
    QualityGates,
    SceneTransform,
    WorldObject,
)
from video2world.providers.external import load_provider_contract
from video2world.site_pipeline import (
    MODELING_VNEXT_STAGE_INPUTS,
    create_site_run,
    load_site_binding,
    load_site_profile,
    preflight_site_run,
)

SHA = "a" * 64


def asset(role: str, *, provenance: dict[str, object] | None = None) -> AssetRef:
    return AssetRef(
        uri=f"artifacts/{role}",
        sha256=SHA,
        size_bytes=123,
        role=role,
        status="validated",
        provenance=provenance or {},
    )


def bbox() -> Bounds2D:
    return Bounds2D(
        frame_id="000020",
        image_width=1280,
        image_height=720,
        xyxy=(100, 120, 900, 700),
    )


def test_checked_in_vnext_pipeline_and_provider_have_exact_roles() -> None:
    pipeline = load_run_config("video2world/configs/modeling_vnext_pipeline.yaml")
    contract = load_provider_contract("video2world/configs/modeling_vnext.provider.example.yaml")
    profile = load_site_profile("video2world/configs/modeling_vnext.site_profile.example.yaml")

    assert profile.pipeline_id == "modeling_vnext"
    assert pipeline.topological_order() == [
        "ingest",
        "da3",
        "pgsr",
        "object_proposals",
        "qwen_contracts",
        "component_segmentation",
        "semantic_lifting",
        "clean_plate",
        "component_assembly",
        "image_completion",
        "object_completion",
        "mesh_postprocess",
        "physics_estimation",
        "placement",
        "bundle",
        "web",
    ]
    assert set(profile.stages) == set(MODELING_VNEXT_STAGE_INPUTS)
    assert set(contract.stages) == set(MODELING_VNEXT_STAGE_INPUTS)
    for stage_id, inputs in MODELING_VNEXT_STAGE_INPUTS.items():
        assert set(pipeline.stages[stage_id].inputs) == set(inputs)
        assert set(contract.stages[stage_id].input_roles) == set(inputs)


def test_vnext_site_profile_initializes_a_bound_executable_graph(tmp_path: Path) -> None:
    pipeline = load_run_config("video2world/configs/modeling_vnext_pipeline.yaml")
    driver = tmp_path / "driver.py"
    driver.write_text("raise SystemExit('preflight only')\n", encoding="utf-8")
    contract_path = tmp_path / "provider.yaml"
    contract_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "provider_id": "vnext-test-provider",
                "scope": "generic_interface",
                "verification_status": "requires_site_configuration",
                "stages": {
                    stage_id: {
                        "provider": "test",
                        "command": [sys.executable, str(driver)],
                        "cwd": "{run_dir}",
                        "input_roles": list(stage.inputs),
                        "output_roles": sorted(get_adapter(stage.adapter).required_outputs),
                        "requires": {
                            "python": {"path": sys.executable, "kind": "executable"},
                            "driver": {"path": str(driver), "kind": "file"},
                        },
                    }
                    for stage_id, stage in pipeline.stages.items()
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "pipeline_id": "modeling_vnext",
                "profile_id": "vnext-test-profile",
                "description": "test profile",
                "stages": {
                    stage_id: {"mode": "execute", "provider_stage": stage_id}
                    for stage_id in pipeline.stages
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")
    run_dir = tmp_path / "run"

    create_site_run(
        run_dir,
        video=video,
        scene_id="scene",
        profile_path=profile_path,
        provider_contract_path=contract_path,
        checkout_roots={},
        artifact_roots={},
    )

    binding = load_site_binding(run_dir)
    bound_config = load_run_config(run_dir / "run.yaml")
    assert binding.pipeline_id == "modeling_vnext"
    assert bound_config.topological_order() == pipeline.topological_order()
    assert all(stage.command for stage in bound_config.stages.values())

    report = preflight_site_run(run_dir, targets=["ingest"])
    assert report["pipeline_id"] == "modeling_vnext"


def test_dinov3_is_feature_encoder_not_proposal_backend() -> None:
    proposal = ObjectProposal(
        id="bed-proposal-1",
        object_id="bed-1",
        category="bed",
        frame_id="000020",
        bbox=bbox(),
        backend="groundingdino",
        feature_encoder="dinov3",
        confidence=0.91,
        prompt="bed",
    )
    assert proposal.backend == "groundingdino"
    assert proposal.feature_encoder == "dinov3"

    with pytest.raises(ValidationError):
        ObjectProposal.model_validate({**proposal.model_dump(mode="json"), "backend": "dinov3"})


def test_qwen_contract_keeps_observed_and_completion_only_parts_distinct() -> None:
    contract = QwenObjectContract(
        object_id="bed-1",
        category="bed",
        detailed_description="Wooden bed with mattress, duvet, pillows, and headboard.",
        positive_prompt="complete wooden bed",
        negative_prompt="nightstand, lamp, wall, floor",
        proposal_ids=["bed-proposal-1"],
        components=[
            ComponentContract(
                id="headboard",
                category="headboard",
                short_prompt="wooden headboard",
                evidence_scope="observed",
            ),
            ComponentContract(
                id="rear_legs",
                category="bed legs",
                short_prompt="rear bed legs",
                evidence_scope="completion_only",
            ),
        ],
        excluded_concepts=["nightstand", "lamp"],
    )
    assert contract.components[1].evidence_scope == "completion_only"


def test_sam_component_mask_cannot_claim_generated_pixels() -> None:
    record = ComponentMaskRecord(
        id="bed-headboard-000020",
        object_id="bed-1",
        component_id="headboard",
        frame_id="000020",
        backend="sam3",
        prompt="wooden headboard",
        bbox=bbox(),
        mask=asset("component_mask"),
    )
    assert record.observation_scope == "observed_pixels_only"
    with pytest.raises(ValidationError):
        ComponentMaskRecord.model_validate(
            {**record.model_dump(mode="json"), "observation_scope": "amodal_generated"}
        )


def test_component_assembly_is_observed_union_not_amodal_completion() -> None:
    assembly = ComponentAssemblyRecord(
        object_id="bed-1",
        observed_component_ids=["headboard", "mattress", "duvet"],
        completion_only_component_ids=["rear_legs", "underside"],
        condition=asset("assembled_object_condition"),
    )
    assert assembly.assembly_scope == "observed_union_not_amodal"

    with pytest.raises(ValidationError, match="must be disjoint"):
        ComponentAssemblyRecord(
            object_id="bed-1",
            observed_component_ids=["headboard"],
            completion_only_component_ids=["headboard"],
            condition=asset("assembled_object_condition"),
        )


def test_3d_fixer_is_in_place_completion_not_mesh_postprocess() -> None:
    candidate = ObjectCompletionCandidate(
        id="bed-fixer-1",
        object_id="bed-1",
        provider="in_place_3d_fixer",
        input_mode="in_place_observed_point_cloud",
        source_sha256=SHA,
        coordinate_frame="moge-normalized",
        mesh=asset("candidate_mesh"),
        generation_status="generated",
    )
    assert candidate.provider == "in_place_3d_fixer"

    with pytest.raises(ValidationError, match="in-place observed point-cloud"):
        ObjectCompletionCandidate.model_validate(
            {**candidate.model_dump(mode="json"), "input_mode": "generated_reference"}
        )


def test_layered_visual_and_coacd_collider_contract() -> None:
    completed = CompletedObjectAsset(
        id="bed-1",
        representation_mode="layered_visual_and_collider",
        collider=asset("collider", provenance={"decomposition": "coacd"}),
        collider_topology="convex_decomposition",
        object_point_cloud=asset("object_point_cloud"),
        geometry_complete_verified=True,
        completion_report_uri="reports/bed-1.json",
    )
    assert completed.object_point_cloud is not None
    assert completed.collider_topology == "convex_decomposition"

    manifest = CompletedObjectAssetsManifest(
        scene_id="scene",
        run_id="run",
        created_at=datetime.now(UTC),
        representation_policy="layered_visual_with_explicit_collider",
        objects=[completed],
    )
    assert manifest.objects[0].representation_mode == "layered_visual_and_collider"


def test_generated_object_image_cannot_be_labeled_as_observation() -> None:
    record = ImageCompletionRecord(
        object_id="bed-1",
        status="accepted",
        source_condition_sha256=SHA,
        model="image-model",
        prompt_sha256=SHA,
        generated_reference=asset("generated_object_reference"),
        generated_region_mask=asset("generated_region_mask"),
    )
    assert record.provenance_scope == "generated_reference_not_observation"

    with pytest.raises(ValidationError, match="generated_object_reference"):
        ImageCompletionRecord.model_validate(
            {
                **record.model_dump(mode="json"),
                "generated_reference": asset("source_observation").model_dump(mode="json"),
            }
        )


def test_qwen_physics_stays_unvalidated_until_calibrated() -> None:
    estimate = PhysicalProperties(
        status="unvalidated_estimate",
        source="qwen_vl_prior",
        scale_basis="category_prior",
        dimensions_m=(2.0, 1.6, 0.9),
        mass_kg=80.0,
        static_friction=0.55,
        dynamic_friction=0.4,
        restitution=0.05,
        confidence=0.35,
        limitations=["No metric scene calibration."],
    )
    assert estimate.status == "unvalidated_estimate"

    with pytest.raises(ValidationError, match="unvalidated_estimate"):
        PhysicalProperties.model_validate(
            {**estimate.model_dump(mode="json"), "status": "validated"}
        )

    with pytest.raises(ValidationError, match="unvalidated_estimate"):
        PhysicalProperties.model_validate(
            {
                **estimate.model_dump(mode="json"),
                "source": "category_prior",
                "status": "validated",
            }
        )


def test_validated_physics_requires_hash_bound_evidence() -> None:
    with pytest.raises(ValidationError, match="hash-bound evidence"):
        PhysicalProperties(
            status="validated",
            source="measured",
            scale_basis="metric_calibrated",
            mass_kg=10.0,
            confidence=0.95,
        )

    value = PhysicalProperties(
        status="validated",
        source="measured",
        scale_basis="metric_calibrated",
        mass_kg=10.0,
        confidence=0.95,
        evidence_uri="reports/scale-and-mass.json",
        evidence_sha256=SHA,
        evidence_size_bytes=123,
    )
    assert value.status == "validated"

    with pytest.raises(ValidationError, match="hash-bound evidence"):
        PhysicalProperties(
            status="calibrated",
            source="simulation_calibrated",
            scale_basis="scene_scale_estimate",
            mass_kg=10.0,
            confidence=0.7,
        )


def test_dynamic_object_rejects_unvalidated_qwen_physics() -> None:
    passed = GateRecord(status="passed")
    common = {
        "id": "nightstand-1",
        "source_run_id": "run",
        "scoped_id": "scene:nightstand-1",
        "name": LocalizedText(en="nightstand"),
        "category": "nightstand",
        "bbox_scene": Bounds3D(
            frame_id="room_world",
            minimum=(0.0, 0.0, 0.0),
            maximum=(0.6, 0.6, 0.7),
        ),
        "render_mesh": asset("render_mesh"),
        "collider": asset("collider", provenance={"decomposition": "coacd"}),
        "collider_topology": "convex_decomposition",
        "transform_scene_from_asset": SceneTransform(
            from_frame="nightstand_asset",
            to_frame="room_world",
            matrix=(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1),
            pivot_scene=(0.3, 0.3, 0.35),
        ),
        "quality_gates": QualityGates(
            file=passed,
            semantic=passed,
            alignment=passed,
            collision=passed,
            visual=passed,
        ),
        "interaction": InteractionPolicy(
            selectable=True,
            collision_enabled=True,
            physics_mode="dynamic",
        ),
    }
    with pytest.raises(ValidationError, match="physics sidecar"):
        WorldObject(**common)

    qwen_prior = PhysicalProperties(
        status="unvalidated_estimate",
        source="qwen_vl_prior",
        scale_basis="category_prior",
        mass_kg=10.0,
        confidence=0.4,
    )
    with pytest.raises(ValidationError, match="unvalidated estimate"):
        WorldObject(**common, physics=qwen_prior)
