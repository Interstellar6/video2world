from __future__ import annotations

import json
from pathlib import Path

from video2world.cli import main
from video2world.completion_routing import (
    CompletionEvidence,
    route_completion_backend,
)


def evidence(**overrides):  # type: ignore[no-untyped-def]
    values = {
        "object_id": "object01",
        "category": "unknown object",
        "target_kind": "object",
        "same_instance_multi_frame_verified": True,
        "appearance_contract_verified": True,
        "calibrated_view_count": 1,
        "view_baseline_score": 0.0,
        "estimated_visible_surface_fraction": 0.24,
        "persistent_occlusion_fraction": 0.55,
        "deformable_prior_available": False,
        "cad_retrieval_confidence": None,
        "support_geometry_available": False,
        "source_front_rgba_available": True,
        "multi_depth_negative_space_verified": False,
    }
    values.update(overrides)
    return CompletionEvidence.model_validate(values)


def test_verified_pillow_uses_deformable_prior_without_hardcoding_object_id() -> None:
    route = route_completion_backend(evidence(object_id="bedroom-pillow-17", category="pillow"))
    assert route.selected_backend == "deformable_category_prior"
    assert route.residual_background_policy == "multi_view_donor_then_constrained_generation"
    assert "front_observation_fidelity" in route.required_acceptance_gates
    assert "scene_interpenetration" in route.required_acceptance_gates


def test_direct_multi_view_evidence_outranks_category_and_generation() -> None:
    route = route_completion_backend(
        evidence(
            category="pillow",
            calibrated_view_count=5,
            view_baseline_score=0.22,
            estimated_visible_surface_fraction=0.61,
        )
    )
    assert route.selected_backend == "multi_view_reconstruction"


def test_high_confidence_rigid_retrieval_outranks_generation() -> None:
    route = route_completion_backend(evidence(category="nightstand", cad_retrieval_confidence=0.91))
    assert route.selected_backend == "retrieval_cad_prior"


def test_structure_uses_support_geometry_and_records_generated_residual() -> None:
    route = route_completion_backend(
        evidence(
            category="bed support surface",
            target_kind="structure_background",
            support_geometry_available=True,
            source_front_rgba_available=False,
            persistent_occlusion_fraction=0.9938,
        )
    )
    assert route.selected_backend == "support_surface_reconstruction"
    assert route.residual_background_policy == "multi_view_donor_then_constrained_generation"
    assert any("no observed donor" in note for note in route.notes)


def test_clean_plate_no_support_blocker_routes_recovery_before_deeper_rounds() -> None:
    route = route_completion_backend(
        evidence(
            category="pillow",
            clean_plate_next_action={
                "action": "add_observed_donor_or_switch_to_constrained_generation_for_residual",
                "blocker": "no_guard_stable_measured_donor_support",
                "failed_frame_ids": ["000064"],
                "first_failed_frame_id": "000064",
                "no_support_frame_ids": ["000064"],
                "unresolved_unobserved_pixels": 23065,
                "promotion_approved": False,
            },
        )
    )

    assert route.selected_backend == "deformable_category_prior"
    assert [item.stage for item in route.recovery_actions] == [
        "donor_support",
        "boundary_qa",
        "candidate_generation",
    ]
    donor = route.recovery_actions[0]
    assert donor.action == "add_observed_donor_or_switch_to_constrained_generation_for_residual"
    assert donor.frame_ids == ["000064"]
    assert donor.allow_deeper_rounds is False
    assert all(item.allow_deeper_rounds is False for item in route.recovery_actions)
    assert any("deeper occlusion rounds must wait" in note for note in route.notes)


def test_clean_plate_temporal_not_evaluable_routes_evidence_repair() -> None:
    route = route_completion_backend(
        evidence(
            category="pillow",
            clean_plate_next_action={
                "action": "repair_temporal_evidence_before_retesting",
                "blocker": "temporal_qa_not_evaluable",
                "not_evaluable_pair_ids": ["0016_to_0017"],
                "not_evaluable_triplet_center_frame_ids": ["000064"],
                "promotion_approved": False,
            },
        )
    )

    assert len(route.recovery_actions) == 1
    recovery = route.recovery_actions[0]
    assert recovery.stage == "temporal_qa"
    assert recovery.action == "repair_temporal_evidence_before_retesting"
    assert recovery.pair_ids == ["0016_to_0017"]
    assert recovery.triplet_center_frame_ids == ["000064"]


def test_multi_depth_fixture_uses_one_logical_component_assembly() -> None:
    route = route_completion_backend(
        evidence(
            object_id="sam3_bed_01",
            category="bed",
            target_kind="fixture",
            support_geometry_available=True,
            multi_depth_negative_space_verified=True,
            calibrated_view_count=2,
        )
    )

    assert route.selected_backend == "component_assembly"
    assert "negative_space_preservation" in route.required_acceptance_gates
    assert "internal_component_alignment" in route.required_acceptance_gates
    assert "single_logical_asset_root" in route.required_acceptance_gates
    assert any("not independent scene objects" in note for note in route.notes)


def test_fixture_component_assembly_requires_explicit_negative_space_evidence() -> None:
    route = route_completion_backend(
        evidence(
            category="bed",
            target_kind="fixture",
            support_geometry_available=True,
            multi_depth_negative_space_verified=False,
        )
    )

    assert route.selected_backend == "generative_image_to_3d"
    candidate = next(item for item in route.candidates if item.backend == "component_assembly")
    assert candidate.eligible is False
    assert "verified multi-depth negative space" in candidate.missing_requirements


def test_unverified_identity_fails_closed_even_for_known_category() -> None:
    route = route_completion_backend(
        evidence(
            category="pillow",
            same_instance_multi_frame_verified=False,
            deformable_prior_available=True,
        )
    )
    assert route.selected_backend == "hold_for_more_evidence"
    assert all(
        not candidate.eligible
        for candidate in route.candidates
        if candidate.backend != "hold_for_more_evidence"
    )


def test_unknown_verified_object_falls_back_to_generation() -> None:
    route = route_completion_backend(evidence(category="unregistered decorative object"))
    assert route.selected_backend == "generative_image_to_3d"


def test_completion_route_cli_writes_auditable_route(tmp_path: Path) -> None:
    source = tmp_path / "evidence.json"
    output = tmp_path / "route.json"
    source.write_text(
        evidence(category="pillow").model_dump_json(indent=2),
        encoding="utf-8",
    )

    assert main(["completion-route", str(source), "--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["selected_backend"] == "deformable_category_prior"
    assert payload["recovery_actions"] == []
    assert len(payload["route_sha256"]) == 64


def test_completion_route_cli_can_inject_failed_clean_plate_report(tmp_path: Path) -> None:
    source = tmp_path / "evidence.json"
    report = tmp_path / "failed-clean-plate.json"
    output = tmp_path / "route.json"
    source.write_text(
        evidence(category="pillow").model_dump_json(indent=2),
        encoding="utf-8",
    )
    report.write_text(
        json.dumps(
            {
                "status": "technical_failed_no_support",
                "next_action": {
                    "action": (
                        "add_observed_donor_or_switch_to_constrained_generation_for_residual"
                    ),
                    "blocker": "no_guard_stable_measured_donor_support",
                    "failed_frame_ids": ["000064"],
                    "first_failed_frame_id": "000064",
                    "no_support_frame_ids": ["000064"],
                    "unresolved_unobserved_pixels": 23065,
                    "failed_pairs": [{"direction_id": "ignored_verbose_record"}],
                    "promotion_approved": False,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "completion-route",
                str(source),
                "--clean-plate-report",
                str(report),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["recovery_actions"][0]["stage"] == "donor_support"
    assert payload["recovery_actions"][0]["frame_ids"] == ["000064"]


def test_completion_route_cli_derives_recovery_from_legacy_prefill_report(
    tmp_path: Path,
) -> None:
    source = tmp_path / "evidence.json"
    report = tmp_path / "legacy-prefill-report.json"
    output = tmp_path / "route.json"
    source.write_text(
        evidence(category="pillow").model_dump_json(indent=2),
        encoding="utf-8",
    )
    report.write_text(
        json.dumps(
            {
                "status": "technical_passed",
                "promotion_approved": False,
                "pixel_provenance": {
                    "unresolved_unobserved_pixels": 23065,
                },
                "frame_records": [
                    {
                        "frame_id": "000063",
                        "removal_mask_pixels": 22000,
                        "residual_mask_pixels": 22000,
                        "covered_pixels": 0,
                        "coverage_fraction": 0.0,
                        "support_max": 0,
                    },
                    {
                        "frame_id": "000064",
                        "removal_mask_pixels": 23065,
                        "residual_mask_pixels": 23065,
                        "covered_pixels": 0,
                        "coverage_fraction": 0.0,
                        "support_max": 0,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "completion-route",
                str(source),
                "--clean-plate-report",
                str(report),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["recovery_actions"][0]["stage"] == "donor_support"
    assert payload["recovery_actions"][0]["frame_ids"] == ["000063", "000064"]
    assert payload["recovery_actions"][0]["allow_deeper_rounds"] is False
