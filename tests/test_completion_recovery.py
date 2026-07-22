from __future__ import annotations

import json
from pathlib import Path

from video2world.cli import main
from video2world.completion_recovery import (
    materialize_completion_recovery_bundle,
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
    input_manifest.write_text('{"frames":[]}', encoding="utf-8")
    camera_info.write_text('{"extrinsic_type":"world_to_camera"}', encoding="utf-8")
    donor_index.write_text('{"items":[]}', encoding="utf-8")
    (frames_dir / "000064.png").write_bytes(b"frame")
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
