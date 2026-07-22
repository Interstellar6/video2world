from __future__ import annotations

import json
from pathlib import Path

from video2world.cli import main
from video2world.completion_recovery import materialize_completion_recovery_work_order
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


def _write_report(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
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
            },
            indent=2,
        ),
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
