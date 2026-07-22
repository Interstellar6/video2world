from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from video2world.cli import main
from video2world.completion_recovery import (
    materialize_completion_recovery_bundle,
    materialize_completion_recovery_constrained_residual_manifest,
    materialize_completion_recovery_residual_handoff,
    materialize_completion_recovery_work_order,
)
from video2world.hashing import digest_path


def _write_route(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "kind": "video2world.completion_backend_route",
                "object_id": "sam3_pillow_front",
                "selected_backend": "deformable_category_prior",
                "route_sha256": "0" * 64,
                "candidates": [
                    {
                        "backend": "deformable_category_prior",
                        "eligible": True,
                        "score": 88,
                        "reason": "A deformable closed-surface prior can preserve silhouette.",
                        "missing_requirements": [],
                    }
                ],
                "required_acceptance_gates": ["geometry_completeness"],
                "residual_background_policy": "multi_view_donor_then_constrained_generation",
                "recovery_actions": [
                    {
                        "priority": 1,
                        "stage": "donor_support",
                        "action": (
                            "add_observed_donor_or_switch_to_constrained_generation_for_residual"
                        ),
                        "blocker": "no_guard_stable_measured_donor_support",
                        "reason": "Boundary-guarded measured donor support is missing.",
                        "frame_ids": ["000064"],
                        "pair_ids": [],
                        "triplet_center_frame_ids": [],
                        "allow_deeper_rounds": False,
                    }
                ],
                "notes": ["Clean-plate recovery is pending."],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_report(path: Path, *, bindings: dict[str, str] | None = None) -> None:
    payload = {
        "status": "technical_passed",
        "promotion_approved": False,
        "frame_records": [
            {
                "frame_id": "000064",
                "removal_mask_pixels": 23065,
                "residual_mask_pixels": 23065,
                "covered_pixels": 0,
                "coverage_fraction": 0.0,
                "support_max": 0,
            }
        ],
    }
    if bindings:
        payload.update(bindings)
    path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def test_completion_recovery_work_order_preserves_blocking_contract(tmp_path: Path) -> None:
    route = tmp_path / "route.json"
    report = tmp_path / "report.json"
    _write_route(route)
    _write_report(report)

    work_order = materialize_completion_recovery_work_order(route, report)

    assert work_order.status == "blocked_pending_recovery"
    assert work_order.object_id == "sam3_pillow_front"
    assert work_order.deeper_rounds_blocked is True
    assert work_order.output_claim == "work_order_only_no_clean_plate_generated"
    assert work_order.inputs[0].sha256 == digest_path(route).sha256
    item = work_order.work_items[0]
    assert item.stage == "donor_support"
    assert item.allow_deeper_rounds is False
    assert item.frame_records[0].frame_id == "000064"
    assert item.frame_records[0].residual_mask_pixels == 23065
    assert "strict boundary guard" in item.required_verification


def test_completion_recovery_work_order_cli_writes_json(tmp_path: Path, capsys) -> None:
    route = tmp_path / "route.json"
    report = tmp_path / "report.json"
    output = tmp_path / "work-order.json"
    _write_route(route)
    _write_report(report)

    assert (
        main(
            [
                "completion-recovery-work-order",
                str(route),
                "--clean-plate-report",
                str(report),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    stdout_payload = json.loads(captured.out)
    file_payload = json.loads(output.read_text(encoding="utf-8"))
    assert stdout_payload == file_payload
    assert file_payload["work_items"][0]["frame_records"][0]["support_max"] == 0


def test_completion_recovery_bundle_orders_recovery_steps(tmp_path: Path) -> None:
    route = tmp_path / "route.json"
    report = tmp_path / "report.json"
    work_order_path = tmp_path / "work-order.json"
    _write_route(route)
    _write_report(report)
    work_order = materialize_completion_recovery_work_order(route, report)
    work_order_path.write_text(work_order.model_dump_json(indent=2), encoding="utf-8")

    bundle = materialize_completion_recovery_bundle(work_order_path)

    assert bundle.kind == "video2world.completion_recovery_bundle"
    assert bundle.status == "ready_for_recovery_execution"
    assert bundle.output_claim == "execution_plan_only_no_artifacts_generated"
    assert bundle.final_gate == "rerun_strict_r1_acceptance_before_r2"
    assert bundle.steps[0].step_id == "01-donor_support"
    assert bundle.steps[0].depends_on == []
    assert "measured_prefill_report" in bundle.steps[0].expected_output_roles
    assert "physical_donor_exclusion_index" in bundle.steps[0].input_roles


def test_completion_recovery_bundle_cli_writes_json(tmp_path: Path, capsys) -> None:
    route = tmp_path / "route.json"
    report = tmp_path / "report.json"
    work_order_path = tmp_path / "work-order.json"
    output = tmp_path / "bundle.json"
    _write_route(route)
    _write_report(report)
    work_order = materialize_completion_recovery_work_order(route, report)
    work_order_path.write_text(work_order.model_dump_json(indent=2), encoding="utf-8")

    assert (
        main(
            [
                "completion-recovery-bundle",
                str(work_order_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    stdout_payload = json.loads(captured.out)
    file_payload = json.loads(output.read_text(encoding="utf-8"))
    assert stdout_payload == file_payload
    assert file_payload["steps"][0]["allow_deeper_rounds"] is False


def test_completion_recovery_preflight_passes_with_local_bindings(tmp_path: Path) -> None:
    from video2world.completion_recovery import materialize_completion_recovery_preflight

    route = tmp_path / "route.json"
    report = tmp_path / "report.json"
    work_order_path = tmp_path / "work-order.json"
    bundle_path = tmp_path / "bundle.json"
    input_manifest = tmp_path / "manifest.json"
    camera_info = tmp_path / "camera-info.json"
    donor_index = tmp_path / "donor-index.json"
    frames_dir = tmp_path / "frames"
    depth_dir = tmp_path / "depth"
    frames_dir.mkdir()
    depth_dir.mkdir()
    source_frame = frames_dir / "000064.png"
    source_frame.write_bytes(b"frame")
    input_manifest.write_text(
        json.dumps(
            {
                "frame_records": [{"frame_id": "000064"}],
                "donor_contract": {
                    "assets": {
                        "000064": {
                            "sha256": digest_path(source_frame).sha256,
                            "role": "original_observed_rgb_only",
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    camera_info.write_text(
        '{"extrinsic_type":"world_to_camera","images":{"000064":{"name":"000064.png"}}}',
        encoding="utf-8",
    )
    donor_index.write_text(
        '{"items":[{"frame_id":"000064","image":"000064.png"}]}',
        encoding="utf-8",
    )
    (depth_dir / "000064.npy").write_bytes(b"depth")
    _write_route(route)
    _write_report(
        report,
        bindings={
            "input_manifest": str(input_manifest),
            "camera_info": str(camera_info),
            "donor_frames_dir": str(frames_dir),
            "depth_dir": str(depth_dir),
            "donor_mask_index": str(donor_index),
        },
    )
    work_order = materialize_completion_recovery_work_order(route, report)
    work_order_path.write_text(work_order.model_dump_json(indent=2), encoding="utf-8")
    bundle = materialize_completion_recovery_bundle(work_order_path)
    bundle_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")

    preflight = materialize_completion_recovery_preflight(bundle_path)

    assert preflight.status == "passed"
    assert preflight.missing_roles == []
    assert {item.role for item in preflight.bindings} == {
        "input_manifest",
        "camera_info",
        "source_rgb_frames",
        "depth_arrays",
        "physical_donor_exclusion_index",
    }
    assert all(item.status == "present" for item in preflight.bindings)
    assert all(item.declared_path == item.effective_path for item in preflight.bindings)
    assert not any(item.override_applied for item in preflight.bindings)
    assert preflight.required_frame_ids == ["000064"]
    assert all(item.status == "matched" for item in preflight.frame_alignment)
    assert preflight.semantic_blocking_roles == []


def test_completion_recovery_preflight_passes_with_binding_overrides(
    tmp_path: Path,
) -> None:
    from video2world.completion_recovery import materialize_completion_recovery_preflight

    route = tmp_path / "route.json"
    report = tmp_path / "report.json"
    work_order_path = tmp_path / "work-order.json"
    bundle_path = tmp_path / "bundle.json"
    input_manifest = tmp_path / "manifest.json"
    camera_info = tmp_path / "camera-info.json"
    donor_index = tmp_path / "donor-index.json"
    frames_dir = tmp_path / "frames"
    depth_dir = tmp_path / "depth"
    frames_dir.mkdir()
    depth_dir.mkdir()
    source_frame = frames_dir / "000064.png"
    source_frame.write_bytes(b"frame")
    input_manifest.write_text(
        json.dumps(
            {
                "frame_records": [{"frame_id": "000064"}],
                "donor_contract": {
                    "assets": {
                        "000064": {
                            "sha256": digest_path(source_frame).sha256,
                            "role": "original_observed_rgb_only",
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    camera_info.write_text(
        '{"subset_provenance":{"frame_ids":["000064"]},"images":{"000064":{}}}',
        encoding="utf-8",
    )
    donor_index.write_text(
        '{"items":[{"frame_id":"000064","image":"000064.png"}]}',
        encoding="utf-8",
    )
    (depth_dir / "000064.npy").write_bytes(b"depth")
    _write_route(route)
    _write_report(
        report,
        bindings={
            "input_manifest": "/remote/missing/manifest.json",
            "camera_info": "/remote/missing/camera_info.json",
            "donor_frames_dir": "/remote/missing/frames",
            "depth_dir": "/remote/missing/depth",
            "donor_mask_index": "/remote/missing/donor-index.json",
        },
    )
    work_order = materialize_completion_recovery_work_order(route, report)
    work_order_path.write_text(work_order.model_dump_json(indent=2), encoding="utf-8")
    bundle = materialize_completion_recovery_bundle(work_order_path)
    bundle_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")

    preflight = materialize_completion_recovery_preflight(
        bundle_path,
        binding_overrides={
            "input_manifest": input_manifest,
            "camera_info": camera_info,
            "source_rgb_frames": frames_dir,
            "depth_arrays": depth_dir,
            "physical_donor_exclusion_index": donor_index,
        },
    )

    assert preflight.status == "passed"
    assert preflight.missing_roles == []
    assert all(item.status == "present" for item in preflight.bindings)
    assert all(item.override_applied for item in preflight.bindings)
    by_role = {item.role: item for item in preflight.bindings}
    assert by_role["input_manifest"].declared_path == "/remote/missing/manifest.json"
    assert by_role["input_manifest"].effective_path == str(input_manifest.resolve())
    assert by_role["input_manifest"].path == str(input_manifest.resolve())
    assert all(item.status == "matched" for item in preflight.frame_alignment)


def test_completion_recovery_preflight_blocks_depth_frame_mismatch(
    tmp_path: Path,
) -> None:
    from video2world.completion_recovery import materialize_completion_recovery_preflight

    route = tmp_path / "route.json"
    report = tmp_path / "report.json"
    work_order_path = tmp_path / "work-order.json"
    bundle_path = tmp_path / "bundle.json"
    input_manifest = tmp_path / "manifest.json"
    camera_info = tmp_path / "camera-info.json"
    donor_index = tmp_path / "donor-index.json"
    frames_dir = tmp_path / "frames"
    depth_dir = tmp_path / "depth"
    frames_dir.mkdir()
    depth_dir.mkdir()
    source_frame = frames_dir / "000064.png"
    source_frame.write_bytes(b"frame")
    input_manifest.write_text(
        json.dumps(
            {
                "frame_records": [{"frame_id": "000064"}],
                "donor_contract": {
                    "assets": {
                        "000064": {
                            "sha256": digest_path(source_frame).sha256,
                            "role": "original_observed_rgb_only",
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    camera_info.write_text(
        '{"images":{"000064":{"name":"000064.png"}}}',
        encoding="utf-8",
    )
    donor_index.write_text(
        '{"items":[{"frame_id":"000064","image":"000064.png"}]}',
        encoding="utf-8",
    )
    (depth_dir / "000000.npy").write_bytes(b"depth")
    _write_route(route)
    _write_report(
        report,
        bindings={
            "input_manifest": str(input_manifest),
            "camera_info": str(camera_info),
            "donor_frames_dir": str(frames_dir),
            "depth_dir": str(depth_dir),
            "donor_mask_index": str(donor_index),
        },
    )
    work_order = materialize_completion_recovery_work_order(route, report)
    work_order_path.write_text(work_order.model_dump_json(indent=2), encoding="utf-8")
    bundle = materialize_completion_recovery_bundle(work_order_path)
    bundle_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")

    preflight = materialize_completion_recovery_preflight(bundle_path)

    assert preflight.status == "blocked_binding_semantics"
    assert preflight.semantic_blocking_roles == ["depth_arrays"]
    by_role = {item.role: item for item in preflight.frame_alignment}
    assert by_role["depth_arrays"].status == "missing_required_frames"
    assert by_role["depth_arrays"].missing_frame_ids == ["000064"]


def test_completion_recovery_preflight_blocks_source_rgb_sha_mismatch(
    tmp_path: Path,
) -> None:
    from video2world.completion_recovery import materialize_completion_recovery_preflight

    route = tmp_path / "route.json"
    report = tmp_path / "report.json"
    work_order_path = tmp_path / "work-order.json"
    bundle_path = tmp_path / "bundle.json"
    input_manifest = tmp_path / "manifest.json"
    camera_info = tmp_path / "camera-info.json"
    donor_index = tmp_path / "donor-index.json"
    frames_dir = tmp_path / "frames"
    depth_dir = tmp_path / "depth"
    frames_dir.mkdir()
    depth_dir.mkdir()
    (frames_dir / "000064.png").write_bytes(b"changed-frame")
    input_manifest.write_text(
        json.dumps(
            {
                "frame_records": [{"frame_id": "000064"}],
                "donor_contract": {
                    "assets": {
                        "000064": {
                            "sha256": "0" * 64,
                            "role": "original_observed_rgb_only",
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    camera_info.write_text(
        '{"images":{"000064":{"name":"000064.png"}}}',
        encoding="utf-8",
    )
    donor_index.write_text(
        '{"items":[{"frame_id":"000064","image":"000064.png"}]}',
        encoding="utf-8",
    )
    (depth_dir / "000064.npy").write_bytes(b"depth")
    _write_route(route)
    _write_report(
        report,
        bindings={
            "input_manifest": str(input_manifest),
            "camera_info": str(camera_info),
            "donor_frames_dir": str(frames_dir),
            "depth_dir": str(depth_dir),
            "donor_mask_index": str(donor_index),
        },
    )
    work_order = materialize_completion_recovery_work_order(route, report)
    work_order_path.write_text(work_order.model_dump_json(indent=2), encoding="utf-8")
    bundle = materialize_completion_recovery_bundle(work_order_path)
    bundle_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")

    preflight = materialize_completion_recovery_preflight(bundle_path)

    assert preflight.status == "blocked_binding_semantics"
    assert preflight.semantic_blocking_roles == ["source_rgb_frames"]
    by_role = {item.role: item for item in preflight.frame_alignment}
    assert by_role["source_rgb_frames"].status == "content_mismatch"
    assert by_role["source_rgb_frames"].content_mismatch_frame_ids == ["000064"]


def test_completion_recovery_preflight_cli_blocks_missing_bindings(
    tmp_path: Path,
    capsys,
) -> None:
    route = tmp_path / "route.json"
    report = tmp_path / "report.json"
    work_order_path = tmp_path / "work-order.json"
    bundle_path = tmp_path / "bundle.json"
    output = tmp_path / "preflight.json"
    _write_route(route)
    _write_report(report)
    work_order = materialize_completion_recovery_work_order(route, report)
    work_order_path.write_text(work_order.model_dump_json(indent=2), encoding="utf-8")
    bundle = materialize_completion_recovery_bundle(work_order_path)
    bundle_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")

    exit_code = main(
        [
            "completion-recovery-preflight",
            str(bundle_path),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    stdout_payload = json.loads(captured.out)
    file_payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 3
    assert stdout_payload == file_payload
    assert file_payload["status"] == "blocked_missing_bindings"
    assert "camera_info" in file_payload["missing_roles"]


def test_completion_recovery_preflight_cli_rejects_unknown_binding_role(
    tmp_path: Path,
    capsys,
) -> None:
    route = tmp_path / "route.json"
    report = tmp_path / "report.json"
    work_order_path = tmp_path / "work-order.json"
    bundle_path = tmp_path / "bundle.json"
    output = tmp_path / "preflight.json"
    _write_route(route)
    _write_report(report)
    work_order = materialize_completion_recovery_work_order(route, report)
    work_order_path.write_text(work_order.model_dump_json(indent=2), encoding="utf-8")
    bundle = materialize_completion_recovery_bundle(work_order_path)
    bundle_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")

    exit_code = main(
        [
            "completion-recovery-preflight",
            str(bundle_path),
            "--binding",
            f"unknown_role={tmp_path}",
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    stderr_payload = json.loads(captured.err)
    assert exit_code == 2
    assert stderr_payload["status"] == "error"
    assert "unknown_role" in stderr_payload["message"]
    assert not output.exists()


def _write_prefill_report_and_receipt(report: Path, receipt: Path, root: Path) -> None:
    report.parent.mkdir(parents=True, exist_ok=True)
    assets = {
        "source_frame": root / "source" / "000064.png",
        "removal_mask": root / "removal" / "000064.png",
        "prefill_frame": root / "frames" / "000064.png",
        "residual_mask": root / "masks" / "000064.png",
        "support_visualization": root / "support" / "000064.png",
        "measured_depth": root / "depth" / "000064.npy",
    }
    for key, path in assets.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{key}-bytes".encode())
    payload = {
        "schema_version": 2,
        "purpose": "measured donor prefill",
        "status": "technical_passed",
        "promotion_approved": False,
        "gates": {
            "outside_removal_mask_rgb_exact": True,
            "all_residual_masks_subset_of_removal_masks": True,
            "fused_measured_metric_depth_materialized": True,
            "physical_instance_donor_exclusion_enforced": False,
        },
        "pixel_provenance": {
            "cumulative_removal_pixels": 23065,
            "measured_multiview_pixels": 100,
            "unresolved_unobserved_pixels": 22965,
            "unresolved_pixels_are_not_valid_donor_or_geometry_evidence": True,
        },
        "next_action": {
            "action": "run_constrained_residual_completion_then_semantic_cross_view_review",
            "blocker": "unresolved_residual_requires_reviewed_completion",
            "promotion_approved": False,
        },
        "frame_records": [
            {
                "sequence_index": 0,
                "frame_id": "000064",
                "source_frame": str(assets["source_frame"]),
                "source_frame_sha256": digest_path(assets["source_frame"]).sha256,
                "removal_mask": str(assets["removal_mask"]),
                "removal_mask_sha256": digest_path(assets["removal_mask"]).sha256,
                "removal_mask_pixels": 23065,
                "prefill_frame": str(assets["prefill_frame"]),
                "prefill_frame_sha256": digest_path(assets["prefill_frame"]).sha256,
                "residual_mask": str(assets["residual_mask"]),
                "residual_mask_sha256": digest_path(assets["residual_mask"]).sha256,
                "residual_mask_pixels": 22965,
                "support_visualization": str(assets["support_visualization"]),
                "support_visualization_sha256": digest_path(
                    assets["support_visualization"]
                ).sha256,
                "measured_depth": str(assets["measured_depth"]),
                "measured_depth_sha256": digest_path(assets["measured_depth"]).sha256,
                "measured_depth_role": "fused_multiview_measured_depth",
                "measured_depth_valid_pixels": 100,
                "covered_pixels": 100,
                "coverage_fraction": 100 / 23065,
                "support_max": 4,
                "donors": [{"frame_id": "000065"}],
            }
        ],
    }
    report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    receipt.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "kind": "video2world.multiview_prefill_receipt",
                "report": report.name,
                "report_sha256": digest_path(report).sha256,
                "frame_count": 1,
                "frame_artifacts_are_hashed_in_report": True,
                "fused_metric_depth_artifacts_are_hashed_in_report": True,
                "measured_provenance_excludes_generated_pixels": True,
                "measured_provenance_excludes_propainter_pixels": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _preflight_binding(path: Path, role: str, expected_kind: str) -> dict[str, object]:
    digest = digest_path(path)
    return {
        "role": role,
        "path": str(digest.path),
        "declared_path": str(digest.path),
        "effective_path": str(digest.path),
        "override_applied": False,
        "expected_kind": expected_kind,
        "status": "present",
        "sha256": digest.sha256,
        "size_bytes": digest.size_bytes,
        "file_count": digest.file_count,
    }


def _write_passed_preflight(
    path: Path,
    bundle_path: Path,
    *,
    bindings: list[dict[str, object]] | None = None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "kind": "video2world.completion_recovery_preflight",
                "object_id": "sam3_pillow_front",
                "preflight_sha256": "0" * 64,
                "status": "passed",
                "deeper_rounds_blocked": True,
                "bundle_sha256": digest_path(bundle_path).sha256,
                "work_order_sha256": "1" * 64,
                "clean_plate_report_sha256": "2" * 64,
                "bundle_input_verified": True,
                "work_order_input_verified": True,
                "clean_plate_report_verified": True,
                "bindings": bindings or [],
                "missing_roles": [],
                "required_frame_ids": ["000064"],
                "frame_alignment": [],
                "semantic_blocking_roles": [],
                "notes": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_completion_recovery_residual_handoff_preserves_non_acceptance_contract(
    tmp_path: Path,
) -> None:
    route = tmp_path / "route.json"
    report = tmp_path / "report.json"
    work_order_path = tmp_path / "work-order.json"
    bundle_path = tmp_path / "bundle.json"
    preflight_path = tmp_path / "preflight.json"
    prefill_report = tmp_path / "prefill" / "multiview_prefill_report.json"
    prefill_receipt = tmp_path / "prefill" / "multiview_prefill_receipt.json"
    _write_route(route)
    _write_report(report)
    work_order = materialize_completion_recovery_work_order(route, report)
    work_order_path.write_text(work_order.model_dump_json(indent=2), encoding="utf-8")
    bundle = materialize_completion_recovery_bundle(work_order_path)
    bundle_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
    _write_passed_preflight(preflight_path, bundle_path)
    _write_prefill_report_and_receipt(prefill_report, prefill_receipt, tmp_path / "assets")

    handoff = materialize_completion_recovery_residual_handoff(
        bundle_path,
        preflight_path,
        prefill_report,
        prefill_receipt,
    )

    assert handoff.status == "ready_for_constrained_residual_completion"
    assert handoff.completed_step_id == "01-donor_support"
    assert handoff.next_step_id == "03-constrained_residual_completion"
    assert handoff.deeper_rounds_blocked is True
    assert handoff.r1_promotion_approved is False
    assert handoff.pixel_provenance["unresolved_unobserved_pixels"] == 22965
    assert handoff.frame_records[0].residual_mask.role == "unresolved_residual_mask"
    assert handoff.frame_records[0].donor_count == 1
    assert "no_r1_acceptance" in handoff.output_claim


def test_completion_recovery_residual_handoff_cli_writes_json(
    tmp_path: Path,
    capsys,
) -> None:
    route = tmp_path / "route.json"
    report = tmp_path / "report.json"
    work_order_path = tmp_path / "work-order.json"
    bundle_path = tmp_path / "bundle.json"
    preflight_path = tmp_path / "preflight.json"
    prefill_report = tmp_path / "prefill" / "multiview_prefill_report.json"
    prefill_receipt = tmp_path / "prefill" / "multiview_prefill_receipt.json"
    output = tmp_path / "residual_handoff.json"
    _write_route(route)
    _write_report(report)
    work_order = materialize_completion_recovery_work_order(route, report)
    work_order_path.write_text(work_order.model_dump_json(indent=2), encoding="utf-8")
    bundle = materialize_completion_recovery_bundle(work_order_path)
    bundle_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
    _write_passed_preflight(preflight_path, bundle_path)
    _write_prefill_report_and_receipt(prefill_report, prefill_receipt, tmp_path / "assets")

    assert (
        main(
            [
                "completion-recovery-residual-handoff",
                "--bundle",
                str(bundle_path),
                "--preflight",
                str(preflight_path),
                "--prefill-report",
                str(prefill_report),
                "--prefill-receipt",
                str(prefill_receipt),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    stdout_payload = json.loads(captured.out)
    file_payload = json.loads(output.read_text(encoding="utf-8"))
    assert stdout_payload == file_payload
    assert file_payload["r1_promotion_approved"] is False
    assert file_payload["frame_records"][0]["measured_multiview_pixels"] == 100


def test_completion_recovery_constrained_residual_manifest_binds_npz_depth(
    tmp_path: Path,
) -> None:
    route = tmp_path / "route.json"
    report = tmp_path / "report.json"
    work_order_path = tmp_path / "work-order.json"
    bundle_path = tmp_path / "bundle.json"
    preflight_path = tmp_path / "preflight.json"
    prefill_report = tmp_path / "prefill" / "multiview_prefill_report.json"
    prefill_receipt = tmp_path / "prefill" / "multiview_prefill_receipt.json"
    handoff_path = tmp_path / "residual_handoff.json"
    camera_info = tmp_path / "camera_info.json"
    depth_dir = tmp_path / "depth"
    mesh = tmp_path / "mesh" / "tsdf_fusion_post.ply"
    script = tmp_path / "scripts" / "complete_planar_background.py"
    output_dir = tmp_path / "03-constrained_residual_completion"
    depth_dir.mkdir()
    mesh.parent.mkdir()
    script.parent.mkdir()
    camera_info.write_text(
        '{"extrinsic_type":"world_to_camera","extrinsic":{"000064":[]}}',
        encoding="utf-8",
    )
    np.savez_compressed(depth_dir / "000064.npz", depth=np.ones((2, 2), dtype=np.float32))
    mesh.write_bytes(b"ply\n")
    script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    _write_route(route)
    _write_report(report)
    work_order = materialize_completion_recovery_work_order(route, report)
    work_order_path.write_text(work_order.model_dump_json(indent=2), encoding="utf-8")
    bundle = materialize_completion_recovery_bundle(work_order_path)
    bundle_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
    _write_passed_preflight(
        preflight_path,
        bundle_path,
        bindings=[
            _preflight_binding(camera_info, "camera_info", "file"),
            _preflight_binding(depth_dir, "depth_arrays", "directory"),
        ],
    )
    _write_prefill_report_and_receipt(prefill_report, prefill_receipt, tmp_path / "assets")
    handoff = materialize_completion_recovery_residual_handoff(
        bundle_path,
        preflight_path,
        prefill_report,
        prefill_receipt,
    )
    handoff_path.write_text(handoff.model_dump_json(indent=2), encoding="utf-8")

    manifest = materialize_completion_recovery_constrained_residual_manifest(
        handoff_path,
        mesh_path=mesh,
        output_dir=output_dir,
        script_path=script,
        working_directory=tmp_path,
    )

    assert manifest.status == "ready_for_structural_residual_completion"
    assert manifest.execution_status == "planned_not_run"
    assert manifest.r1_promotion_approved is False
    assert manifest.depth_format_counts == {"npz": 1}
    assert manifest.frame_records[0].residual_mask.role == "unresolved_residual_mask"
    assert "--upstream-prefill-report" in manifest.command_argv
    assert str(prefill_report.resolve()) in manifest.command_argv
    assert str(depth_dir.resolve()) in manifest.command_argv
    assert manifest.constraints["allowed_edit_region"] == "residual_mask"


def test_completion_recovery_constrained_residual_manifest_cli_writes_json(
    tmp_path: Path,
    capsys,
) -> None:
    route = tmp_path / "route.json"
    report = tmp_path / "report.json"
    work_order_path = tmp_path / "work-order.json"
    bundle_path = tmp_path / "bundle.json"
    preflight_path = tmp_path / "preflight.json"
    prefill_report = tmp_path / "prefill" / "multiview_prefill_report.json"
    prefill_receipt = tmp_path / "prefill" / "multiview_prefill_receipt.json"
    handoff_path = tmp_path / "residual_handoff.json"
    output = tmp_path / "step03_manifest.json"
    camera_info = tmp_path / "camera_info.json"
    depth_dir = tmp_path / "depth"
    mesh = tmp_path / "mesh.ply"
    script = tmp_path / "complete_planar_background.py"
    depth_dir.mkdir()
    camera_info.write_text(
        '{"extrinsic_type":"world_to_camera","extrinsic":{"000064":[]}}',
        encoding="utf-8",
    )
    np.savez_compressed(depth_dir / "000064.npz", depth=np.ones((2, 2), dtype=np.float32))
    mesh.write_bytes(b"ply\n")
    script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    _write_route(route)
    _write_report(report)
    work_order = materialize_completion_recovery_work_order(route, report)
    work_order_path.write_text(work_order.model_dump_json(indent=2), encoding="utf-8")
    bundle = materialize_completion_recovery_bundle(work_order_path)
    bundle_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
    _write_passed_preflight(
        preflight_path,
        bundle_path,
        bindings=[
            _preflight_binding(camera_info, "camera_info", "file"),
            _preflight_binding(depth_dir, "depth_arrays", "directory"),
        ],
    )
    _write_prefill_report_and_receipt(prefill_report, prefill_receipt, tmp_path / "assets")
    handoff = materialize_completion_recovery_residual_handoff(
        bundle_path,
        preflight_path,
        prefill_report,
        prefill_receipt,
    )
    handoff_path.write_text(handoff.model_dump_json(indent=2), encoding="utf-8")

    assert (
        main(
            [
                "completion-recovery-constrained-residual-manifest",
                "--handoff",
                str(handoff_path),
                "--mesh",
                str(mesh),
                "--output-dir",
                str(tmp_path / "planned-output"),
                "--script",
                str(script),
                "--workdir",
                str(tmp_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    stdout_payload = json.loads(captured.out)
    file_payload = json.loads(output.read_text(encoding="utf-8"))
    assert stdout_payload == file_payload
    assert file_payload["depth_format_counts"] == {"npz": 1}
    assert file_payload["execution_status"] == "planned_not_run"
