from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest
import yaml

from video2world.adapters import get_adapter
from video2world.cli import main
from video2world.config import RunConfig, load_run_config
from video2world.errors import ArtifactError, StageBlockedError
from video2world.orchestrator import PipelineOrchestrator, initialize_run, snapshot_path
from video2world.state import load_stage_state


def write_template(
    path: Path,
    *,
    command: list[str] | None,
    fingerprints: dict[str, str] | None = None,
) -> None:
    payload = {
        "schema_version": "1.0",
        "run_id": "template",
        "scene_id": "template",
        "input_video": "/placeholder/video.mp4",
        "run_dir": "/placeholder/run",
        "variables": {},
        "stages": {
            "single": {
                "adapter": "command",
                "needs": [],
                "command": command,
                "cwd": "{run_dir}",
                "inputs": {"video": "{input_video}"},
                "fingerprints": fingerprints or {},
                "outputs": {"result": "{run_dir}/artifacts/result.txt"},
            }
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def initialized_run(
    tmp_path: Path,
    *,
    command: list[str] | None,
    fingerprints: dict[str, str] | None = None,
) -> tuple[Path, Path]:
    video = tmp_path / "input.mp4"
    video.write_bytes(b"video-v1")
    template = tmp_path / "template.yaml"
    write_template(template, command=command, fingerprints=fingerprints)
    run_dir = tmp_path / "run"
    initialize_run(
        run_dir,
        video=video,
        scene_id="test_scene",
        run_id="test_run",
        template_config=template,
    )
    return run_dir, video


def test_default_pipeline_has_named_provider_dag(tmp_path: Path) -> None:
    video = tmp_path / "input.mp4"
    video.write_bytes(b"real-input")
    run_dir = tmp_path / "default-run"
    initialize_run(run_dir, video=video, scene_id="scene")
    config = load_run_config(run_dir / "run.yaml")
    order = config.topological_order()
    assert set(order) == {
        "ingest",
        "inventory",
        "da3",
        "pgsr",
        "sam3",
        "fusion",
        "cognition",
        "completion_plan",
        "layered_completion",
        "placement",
        "bundle",
        "web",
    }
    position = {stage_id: index for index, stage_id in enumerate(order)}
    for stage_id, stage in config.stages.items():
        assert all(position[dependency] < position[stage_id] for dependency in stage.needs)

    assert position["ingest"] < position["da3"] < position["pgsr"] < position["fusion"]
    assert position["inventory"] < position["sam3"] < position["fusion"]
    assert position["fusion"] < position["completion_plan"] < position["layered_completion"]
    assert position["layered_completion"] < position["placement"] < position["bundle"]
    assert position["bundle"] < position["web"]
    plans = PipelineOrchestrator(run_dir).plan()
    assert plans[0].action == "adopt_or_configure"
    assert plans[-1].action == "waiting"


def test_cycle_is_rejected() -> None:
    payload = {
        "schema_version": "1.0",
        "run_id": "run",
        "scene_id": "scene",
        "input_video": "/input.mp4",
        "run_dir": "/run",
        "stages": {
            "a": {"adapter": "command", "needs": ["b"], "outputs": {"x": "/x"}},
            "b": {"adapter": "command", "needs": ["a"], "outputs": {"x": "/y"}},
        },
    }
    with pytest.raises(Exception, match="cycle"):
        RunConfig.model_validate(payload)


def test_reserved_variables_and_literal_secrets_are_rejected() -> None:
    base_stage = {"adapter": "command", "outputs": {"result": "result.txt"}}
    payload = {
        "schema_version": "1.0",
        "run_id": "run",
        "scene_id": "scene",
        "input_video": "/input.mp4",
        "run_dir": "/run",
        "variables": {"run_dir": "/override"},
        "stages": {"single": base_stage},
    }
    with pytest.raises(Exception, match="reserved variable"):
        RunConfig.model_validate(payload)

    payload["variables"] = {}
    payload["stages"]["single"]["env"] = {"OPENAI_API_KEY": "plaintext"}
    with pytest.raises(Exception, match="sensitive environment"):
        RunConfig.model_validate(payload)

    payload["stages"]["single"]["env"] = {"OPENAI_API_KEY": "${OPENAI_API_KEY}"}
    assert RunConfig.model_validate(payload).stages["single"].env == {
        "OPENAI_API_KEY": "${OPENAI_API_KEY}"
    }


def test_dry_run_never_writes_state_or_output(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-c",
        "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('made')",
        "{output_result}",
    ]
    run_dir, _ = initialized_run(tmp_path, command=command)
    before = set((run_dir / ".video2world" / "stages").iterdir())
    assert main(["run", str(run_dir), "--dry-run"]) == 0
    after = set((run_dir / ".video2world" / "stages").iterdir())
    assert before == after == set()
    assert not (run_dir / "artifacts" / "result.txt").exists()


def test_execute_cache_and_input_content_invalidation(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-c",
        (
            "from pathlib import Path; import sys; "
            "Path(sys.argv[1]).write_bytes(Path(sys.argv[2]).read_bytes())"
        ),
        "{output_result}",
        "{input_video}",
    ]
    run_dir, video = initialized_run(tmp_path, command=command)
    orchestrator = PipelineOrchestrator(run_dir)
    assert orchestrator.plan()[0].action == "run"
    first = orchestrator.run()
    assert first[0]["action"] == "executed"
    state = load_stage_state(run_dir, "single")
    assert state is not None and state.status == "succeeded" and state.attempt == 1
    assert (run_dir / "artifacts" / "result.txt").read_bytes() == b"video-v1"

    cached = orchestrator.run()
    assert cached == [
        {
            "stage_id": "single",
            "action": "cached",
            "reason": "state and content hashes are current",
        }
    ]

    video.write_bytes(b"video-v2")
    assert orchestrator.plan()[0].action == "stale"
    second = orchestrator.run()
    assert second[0]["attempt"] == 2
    assert (run_dir / "artifacts" / "result.txt").read_bytes() == b"video-v2"


def test_tool_fingerprint_content_invalidates_the_stage_cache(tmp_path: Path) -> None:
    run_dir, _ = initialized_run(
        tmp_path,
        command=[
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('ok')",
            "{output_result}",
        ],
        fingerprints={"tool": "{run_dir}/tool.py"},
    )
    tool = run_dir / "tool.py"
    tool.write_text("version = 1\n", encoding="utf-8")
    orchestrator = PipelineOrchestrator(run_dir)
    assert orchestrator.run()[0]["action"] == "executed"
    assert orchestrator.run()[0]["action"] == "cached"
    tool.write_text("version = 2\n", encoding="utf-8")
    assert orchestrator.plan()[0].action == "stale"


def test_run_lock_rejects_concurrent_execution(tmp_path: Path) -> None:
    run_dir, _ = initialized_run(
        tmp_path,
        command=[sys.executable, "-c", "pass"],
    )
    first = PipelineOrchestrator(run_dir)
    second = PipelineOrchestrator(run_dir)
    with first._execution_lock(), pytest.raises(StageBlockedError, match="already executing"):
        second.run()


def test_success_exit_without_required_output_is_failed(tmp_path: Path) -> None:
    run_dir, _ = initialized_run(tmp_path, command=[sys.executable, "-c", "pass"])
    with pytest.raises(ArtifactError, match="does not exist"):
        PipelineOrchestrator(run_dir).run()
    state = load_stage_state(run_dir, "single")
    assert state is not None
    assert state.status == "failed"
    assert state.mode == "executed"


def test_adopt_records_real_content_without_copying(tmp_path: Path) -> None:
    run_dir, _ = initialized_run(tmp_path, command=None)
    existing = tmp_path / "upstream" / "result.txt"
    existing.parent.mkdir()
    existing.write_text("real upstream output", encoding="utf-8")
    orchestrator = PipelineOrchestrator(run_dir)
    state = orchestrator.adopt(
        "single",
        {"result": existing},
        source_run_id="upstream-run-42",
        source_repository="/upstream/repo",
        source_commit="abc123",
    )
    assert state.status == "adopted"
    assert state.mode == "adopted"
    assert state.outputs["result"].path == str(existing.resolve())
    assert state.adoption is not None
    assert state.adoption.source_run_id == "upstream-run-42"
    assert not (run_dir / "artifacts" / "result.txt").exists()
    assert orchestrator.plan()[0].action == "cached"

    existing.write_text("mutated", encoding="utf-8")
    plan = orchestrator.plan()[0]
    assert plan.action == "adopt_or_configure"
    assert "content changed" in plan.reason


def test_adopt_requires_exact_roles_and_nonempty_files(tmp_path: Path) -> None:
    run_dir, _ = initialized_run(tmp_path, command=None)
    existing = tmp_path / "existing.txt"
    existing.write_text("real", encoding="utf-8")
    orchestrator = PipelineOrchestrator(run_dir)
    with pytest.raises(ArtifactError, match=r"missing=.*result"):
        orchestrator.adopt("single", {}, source_run_id="source")
    with pytest.raises(ArtifactError, match=r"extra=.*wrong"):
        orchestrator.adopt("single", {"wrong": existing}, source_run_id="source")


def test_explicit_adoption_root_must_contain_outputs(tmp_path: Path) -> None:
    run_dir, _ = initialized_run(tmp_path, command=None)
    existing = tmp_path / "source-a" / "existing.txt"
    existing.parent.mkdir()
    existing.write_text("real", encoding="utf-8")
    other_root = tmp_path / "source-b"
    other_root.mkdir()
    with pytest.raises(ArtifactError, match="outside source root"):
        PipelineOrchestrator(run_dir).adopt(
            "single",
            {"result": existing},
            source_run_id="source",
            source_root=other_root,
        )


def test_named_adapters_reject_mislabeled_geometry_and_json(tmp_path: Path) -> None:
    fake_ply = tmp_path / "fake.ply"
    fake_ply.write_text("not a ply", encoding="utf-8")
    with pytest.raises(ArtifactError, match="valid PLY header"):
        get_adapter("holi_pgsr").validate_outputs({"scene_gaussian": snapshot_path(fake_ply)})

    fake_json = tmp_path / "masks.json"
    fake_json.write_text("not-json", encoding="utf-8")
    with pytest.raises(ArtifactError, match="readable JSON"):
        get_adapter("holi_sam3").validate_outputs({"masks_manifest": snapshot_path(fake_json)})


def _provider_receipt_snapshot(name: str, digest: str) -> dict[str, object]:
    return {
        "path": f"/tmp/video2world/{name}",
        "kind": "file",
        "sha256": digest * 64,
        "size_bytes": 1024,
        "file_count": 1,
    }


def _artifact_snapshot_payload(path: Path) -> dict[str, object]:
    snapshot = snapshot_path(path)
    return {
        "path": snapshot.path,
        "kind": snapshot.kind,
        "sha256": snapshot.sha256,
        "size_bytes": snapshot.size_bytes,
        "file_count": snapshot.file_count,
    }


def _provider_receipt_payload(
    *,
    inputs: dict[str, object] | None = None,
    outputs: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "video2world.provider_execution_receipt",
        "status": "completed",
        "provider_id": "site-test-provider",
        "provider_stage_id": "layered_completion",
        "provider": "Video2World layered completion providers",
        "inputs": inputs or {},
        "outputs": outputs or {"result": _provider_receipt_snapshot("result.json", "a")},
    }


def test_provider_receipt_role_uses_typed_contract(tmp_path: Path) -> None:
    receipt_path = tmp_path / "provider-receipt.json"
    receipt_path.write_text(json.dumps(_provider_receipt_payload()), encoding="utf-8")
    get_adapter("holi_ingest").validate_outputs({"provider_receipt": snapshot_path(receipt_path)})

    broken = _provider_receipt_payload()
    broken["outputs"] = {"result": {"path": "/tmp/result.json", "sha256": "a" * 64}}
    receipt_path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ArtifactError, match=r"provider_receipt outputs\.result\.kind"):
        get_adapter("holi_ingest").validate_outputs(
            {"provider_receipt": snapshot_path(receipt_path)}
        )


@pytest.mark.parametrize(
    ("adapter_name", "role"),
    [
        ("layered_completion_plan", "layered_completion_plan"),
        ("layered_completion", "completed_object_assets_manifest"),
        ("layered_completion", "clean_plate_manifest"),
        ("layered_completion", "layered_completion_report"),
    ],
)
def test_completion_json_roles_reject_empty_placeholder_objects(
    tmp_path: Path,
    adapter_name: str,
    role: str,
) -> None:
    output = tmp_path / f"{role}.json"
    output.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ArtifactError, match="non-empty JSON object"):
        get_adapter(adapter_name).validate_outputs({role: snapshot_path(output)})


def test_layered_completion_plan_uses_its_typed_contract(tmp_path: Path) -> None:
    output = tmp_path / "plan.json"
    output.write_text(
        json.dumps({"kind": "video2world.layered_completion_plan"}),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError, match="failed semantic validation"):
        get_adapter("layered_completion_plan").validate_outputs(
            {"layered_completion_plan": snapshot_path(output)}
        )


def _completion_evidence(name: str, digest: str) -> dict[str, object]:
    return {
        "uri": f"artifact://completion/{name}",
        "sha256": digest * 64,
        "size_bytes": 1024,
    }


def _next_round_clean_plate_evidence(name: str, digest: str) -> dict[str, object]:
    evidence = _completion_evidence(name, digest)
    evidence.update(
        {
            "acceptance_scope": "corrected_clean_plate_next_round_source_only",
            "lineage_scope": "corrected_clean_plate_round_source",
            "corrected_full_pipeline": False,
            "promotion_approved": False,
            "canonical_promotion_approved": False,
            "canonical_or_live_manifest_modified": False,
        }
    )
    return evidence


def _terminal_clean_plate_evidence(name: str, digest: str) -> dict[str, object]:
    evidence = _completion_evidence(name, digest)
    evidence.update(
        {
            "acceptance_scope": "corrected_full_pipeline",
            "lineage_scope": "corrected_full_pipeline",
            "corrected_full_pipeline": True,
            "promotion_approved": True,
            "canonical_promotion_approved": True,
            "canonical_or_live_manifest_modified": False,
        }
    )
    return evidence


def _valid_layered_completion_report() -> dict[str, object]:
    initial = _completion_evidence("round0-clean-plate.json", "a")
    round1 = _next_round_clean_plate_evidence("round1-clean-plate.json", "b")
    final = _terminal_clean_plate_evidence("final-clean-plate.json", "c")
    return {
        "lineage_scope": "corrected_full_pipeline",
        "scene_id": "bedroom_4",
        "run_id": "full-layered-run",
        "created_at": "2026-07-17T00:00:00Z",
        "completion_plan_sha256": "6" * 64,
        "planned_target_ids": ["pillow-front"],
        "initial_clean_plate": initial,
        "rounds": [
            {
                "index": 1,
                "kind": "object_layer",
                "target_ids": ["pillow-front"],
                "input_clean_plate": initial,
                "scene_audit_receipt": _completion_evidence("r1-audit.json", "d"),
                "sam3_receipt": _completion_evidence("r1-sam3.json", "e"),
                "output_clean_plate": dict(round1),
                "quality_report": _completion_evidence("r1-quality.json", "f"),
                "object_completion_receipts": [
                    _completion_evidence("r1-pillow-trellis2.json", "1")
                ],
                "acceptance_gates": {"object_shape": True, "clean_plate": True},
            },
            {
                "index": 2,
                "kind": "final_background",
                "target_ids": [],
                "input_clean_plate": dict(round1),
                "scene_audit_receipt": _completion_evidence("r2-audit.json", "2"),
                "sam3_receipt": _completion_evidence("r2-sam3.json", "3"),
                "output_clean_plate": dict(final),
                "quality_report": _completion_evidence("r2-quality.json", "4"),
                "background_rebuild_receipt": _completion_evidence("r2-background.json", "5"),
                "acceptance_gates": {"revealed_background": True},
            },
        ],
        "final_clean_plate": dict(final),
    }


def _valid_layered_completion_plan() -> dict[str, object]:
    return {
        "scene_id": "bedroom_4",
        "run_id": "full-layered-run",
        "created_at": "2026-07-17T00:00:00Z",
        "inventory_sha256": "7" * 64,
        "occlusion_graph_sha256": "8" * 64,
        "rounds": [
            {
                "index": 1,
                "kind": "object_layer",
                "target_ids": ["pillow-front"],
                "input_clean_plate_round": 0,
                "actions": [
                    {
                        "id": "round1-complete",
                        "action": "complete_object",
                        "target_ids": ["pillow-front"],
                        "idempotency_key": "9" * 64,
                    }
                ],
            },
            {
                "index": 2,
                "kind": "final_background",
                "target_ids": [],
                "input_clean_plate_round": 1,
                "actions": [
                    {
                        "id": "round2-background",
                        "action": "rebuild_background",
                        "idempotency_key": "0" * 64,
                    }
                ],
            },
        ],
        "stop_conditions": ["All planned rounds and the final background must pass."],
    }


def test_layered_completion_report_binds_executed_plan_and_provider_receipt(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_valid_layered_completion_plan()), encoding="utf-8")
    plan_sha = snapshot_path(plan_path).sha256
    report_payload = _valid_layered_completion_report()
    report_payload["completion_plan_sha256"] = plan_sha
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")
    report_snapshot = snapshot_path(report_path)
    receipt_path = tmp_path / "provider-receipt.json"
    receipt_path.write_text(
        json.dumps(
            _provider_receipt_payload(
                inputs={
                    "layered_completion_plan": {
                        "path": str(plan_path),
                        "kind": "file",
                        "sha256": plan_sha,
                        "size_bytes": plan_path.stat().st_size,
                        "file_count": 1,
                    }
                },
                outputs={"layered_completion_report": _artifact_snapshot_payload(report_path)},
            )
        ),
        encoding="utf-8",
    )

    outputs = {
        "layered_completion_report": report_snapshot,
        "provider_receipt": snapshot_path(receipt_path),
    }
    get_adapter("layered_completion").validate_outputs(outputs)

    report_payload["completion_plan_sha256"] = "f" * 64
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")
    receipt_path.write_text(
        json.dumps(
            _provider_receipt_payload(
                inputs={
                    "layered_completion_plan": {
                        "path": str(plan_path),
                        "kind": "file",
                        "sha256": plan_sha,
                        "size_bytes": plan_path.stat().st_size,
                        "file_count": 1,
                    }
                },
                outputs={"layered_completion_report": _artifact_snapshot_payload(report_path)},
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(ArtifactError, match="does not bind the executed plan"):
        get_adapter("layered_completion").validate_outputs(
            {
                "layered_completion_report": snapshot_path(report_path),
                "provider_receipt": snapshot_path(receipt_path),
            }
        )


def test_layered_completion_report_requires_provider_receipt_output_snapshot(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_valid_layered_completion_plan()), encoding="utf-8")
    plan_sha = snapshot_path(plan_path).sha256
    report_payload = _valid_layered_completion_report()
    report_payload["completion_plan_sha256"] = plan_sha
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")
    bad_report_snapshot = _artifact_snapshot_payload(report_path)
    bad_report_snapshot["sha256"] = "f" * 64
    receipt_path = tmp_path / "provider-receipt.json"
    receipt_path.write_text(
        json.dumps(
            _provider_receipt_payload(
                inputs={
                    "layered_completion_plan": _artifact_snapshot_payload(plan_path),
                },
                outputs={"layered_completion_report": bad_report_snapshot},
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError, match="output snapshot does not match"):
        get_adapter("layered_completion").validate_outputs(
            {
                "layered_completion_report": snapshot_path(report_path),
                "provider_receipt": snapshot_path(receipt_path),
            }
        )


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda payload: payload.pop("lineage_scope"), "lineage_scope"),
        (
            lambda payload: payload.__setitem__("lineage_scope", "current_demo_only"),
            "corrected_full_pipeline",
        ),
    ],
)
def test_layered_completion_report_requires_corrected_full_pipeline_lineage(
    tmp_path: Path,
    mutation,
    expected: str,
) -> None:
    report_payload = _valid_layered_completion_report()
    mutation(report_payload)
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")

    with pytest.raises(ArtifactError, match=expected):
        get_adapter("layered_completion").validate_outputs(
            {"layered_completion_report": snapshot_path(report_path)}
        )


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (
            lambda payload: payload["rounds"][0]["output_clean_plate"].pop(
                "acceptance_scope"
            ),
            "round 1 output_clean_plate acceptance_scope",
        ),
        (
            lambda payload: payload["rounds"][0]["output_clean_plate"].__setitem__(
                "corrected_full_pipeline", True
            ),
            "round 1 output_clean_plate must not claim corrected full-pipeline",
        ),
        (
            lambda payload: payload["rounds"][1]["input_clean_plate"].__setitem__(
                "lineage_scope", "current_demo_only"
            ),
            "round 2 input_clean_plate lineage_scope",
        ),
        (
            lambda payload: payload["rounds"][1]["output_clean_plate"].__setitem__(
                "acceptance_scope", "corrected_clean_plate_next_round_source_only"
            ),
            "round 2 output_clean_plate acceptance_scope",
        ),
        (
            lambda payload: payload["final_clean_plate"].__setitem__(
                "canonical_promotion_approved", False
            ),
            "final_clean_plate must claim canonical promotion approval",
        ),
    ],
)
def test_layered_completion_report_binds_clean_plate_scope_lineage(
    tmp_path: Path,
    mutation,
    expected: str,
) -> None:
    report_payload = _valid_layered_completion_report()
    mutation(report_payload)
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")

    with pytest.raises(ArtifactError, match=expected):
        get_adapter("layered_completion").validate_outputs(
            {"layered_completion_report": snapshot_path(report_path)}
        )


def test_layered_completion_report_requires_provider_receipt_lineage(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_valid_layered_completion_report()), encoding="utf-8")

    with pytest.raises(ArtifactError, match="requires provider_receipt"):
        get_adapter("layered_completion").validate_outputs(
            {"layered_completion_report": snapshot_path(report_path)}
        )


@pytest.mark.parametrize(
    ("role", "valid_payload"),
    [
        ("clean_plate_manifest", {"frame_records": [{"frame_id": "000064"}]}),
    ],
)
def test_completion_json_roles_require_role_specific_payload_fields(
    tmp_path: Path,
    role: str,
    valid_payload: dict[str, object],
) -> None:
    output = tmp_path / f"{role}.json"
    output.write_text(json.dumps({"placeholder": True}), encoding="utf-8")
    with pytest.raises(ArtifactError, match="failed semantic validation"):
        get_adapter("layered_completion").validate_outputs({role: snapshot_path(output)})

    output.write_text(json.dumps(valid_payload), encoding="utf-8")
    get_adapter("layered_completion").validate_outputs({role: snapshot_path(output)})


def _valid_completed_object_assets_manifest() -> dict[str, object]:
    return {
        "scene_id": "bedroom_4",
        "run_id": "completion-1",
        "created_at": "2026-07-17T00:00:00Z",
        "objects": [
            {
                "id": "pillow-01",
                "render_mesh": {
                    "uri": "artifact://pillow-01.glb",
                    "sha256": "a" * 64,
                    "size_bytes": 1024,
                    "role": "render_mesh",
                    "status": "validated",
                },
                "collider": {
                    "uri": "artifact://pillow-01-collider.glb",
                    "sha256": "b" * 64,
                    "size_bytes": 512,
                    "role": "collider",
                    "status": "validated",
                },
                "closed_surface_verified": True,
                "completion_report_uri": "artifact://pillow-01/report.json",
            }
        ],
    }


@pytest.mark.parametrize("missing_field", ["render_mesh", "collider", "closed_surface_verified"])
def test_completed_object_assets_manifest_uses_mesh_first_typed_contract(
    tmp_path: Path,
    missing_field: str,
) -> None:
    output = tmp_path / "completed-object-assets.json"
    payload = _valid_completed_object_assets_manifest()
    output.write_text(json.dumps(payload), encoding="utf-8")

    get_adapter("layered_completion").validate_outputs(
        {"completed_object_assets_manifest": snapshot_path(output)}
    )

    del payload["objects"][0][missing_field]
    output.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ArtifactError, match="failed semantic validation"):
        get_adapter("layered_completion").validate_outputs(
            {"completed_object_assets_manifest": snapshot_path(output)}
        )


@pytest.mark.parametrize(
    ("role", "payload"),
    [
        ("clean_plate_manifest", {"frame_records": [{"placeholder": True}]}),
    ],
)
def test_completion_manifests_reject_unidentified_placeholder_records(
    tmp_path: Path,
    role: str,
    payload: dict[str, object],
) -> None:
    output = tmp_path / f"{role}.json"
    output.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ArtifactError, match="requires non-empty identified records"):
        get_adapter("layered_completion").validate_outputs({role: snapshot_path(output)})


@pytest.mark.parametrize("case", ["broken_chain", "missing_sam3", "first_layer_only"])
def test_layered_completion_report_rejects_incomplete_peel_runs(tmp_path: Path, case: str) -> None:
    payload = _valid_layered_completion_report()
    rounds = payload["rounds"]
    assert isinstance(rounds, list)
    if case == "broken_chain":
        rounds[1]["input_clean_plate"] = _completion_evidence("wrong.json", "9")
    elif case == "missing_sam3":
        del rounds[0]["sam3_receipt"]
    else:
        payload["rounds"] = rounds[:1]
        payload["final_clean_plate"] = rounds[0]["output_clean_plate"]
    output = tmp_path / "report.json"
    output.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ArtifactError, match="failed semantic validation"):
        get_adapter("layered_completion").validate_outputs(
            {"layered_completion_report": snapshot_path(output)}
        )


def _clean_gaussian_header(format_name: str = "ascii") -> bytes:
    properties = [
        "x",
        "y",
        "z",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ]
    lines = ["ply", f"format {format_name} 1.0", "element vertex 1"]
    lines.extend(f"property float {name}" for name in properties)
    lines.append("end_header")
    return ("\n".join(lines) + "\n").encode("ascii")


def _clean_mesh_header(format_name: str = "binary_little_endian") -> bytes:
    return (
        "ply\n"
        f"format {format_name} 1.0\n"
        "element vertex 3\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "element face 1\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")


@pytest.mark.parametrize(
    ("role", "header"),
    [
        ("clean_scene_gaussian", _clean_gaussian_header()),
        ("clean_scene_mesh", _clean_mesh_header()),
    ],
)
def test_clean_scene_ply_roles_reject_header_only_files(
    tmp_path: Path,
    role: str,
    header: bytes,
) -> None:
    output = tmp_path / f"{role}.ply"
    output.write_bytes(header)

    with pytest.raises(ArtifactError, match="payload is truncated"):
        get_adapter("layered_completion").validate_outputs({role: snapshot_path(output)})


def test_clean_scene_mesh_requires_real_face_indices_payload(tmp_path: Path) -> None:
    output = tmp_path / "mesh-with-header-only-faces.ply"
    output.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 3",
                "property float x",
                "property float y",
                "property float z",
                "element face 1",
                "end_header",
                "0 0 0",
                "1 0 0",
                "0 1 0",
            ]
        )
        + "\n",
        encoding="ascii",
    )

    with pytest.raises(ArtifactError, match="faces require a vertex_indices list"):
        get_adapter("layered_completion").validate_outputs(
            {"clean_scene_mesh": snapshot_path(output)}
        )


def test_clean_scene_ply_roles_accept_real_ascii_and_binary_payloads(tmp_path: Path) -> None:
    gaussian = tmp_path / "clean-gaussian-ascii.ply"
    gaussian.write_bytes(_clean_gaussian_header() + ("0 " * 13 + "0\n").encode("ascii"))

    mesh = tmp_path / "clean-mesh-binary.ply"
    mesh.write_bytes(
        _clean_mesh_header()
        + struct.pack(
            "<9fB3i",
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            3,
            0,
            1,
            2,
        )
    )

    get_adapter("layered_completion").validate_outputs(
        {
            "clean_scene_gaussian": snapshot_path(gaussian),
            "clean_scene_mesh": snapshot_path(mesh),
        }
    )


def test_relative_config_paths_are_rooted_in_run_directory(tmp_path: Path) -> None:
    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")
    template = tmp_path / "relative.yaml"
    write_template(
        template,
        command=[
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('ok')",
            "{output_result}",
        ],
    )
    payload = yaml.safe_load(template.read_text(encoding="utf-8"))
    payload["stages"]["single"]["outputs"]["result"] = "artifacts/relative.txt"
    template.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    run_dir = tmp_path / "run-relative"
    initialize_run(run_dir, video=video, scene_id="scene", template_config=template)
    PipelineOrchestrator(run_dir).run()
    assert (run_dir / "artifacts" / "relative.txt").read_text(encoding="utf-8") == "ok"


def test_cli_adopt_existing_is_distinct_from_dry_run(tmp_path: Path, capsys) -> None:
    run_dir, _ = initialized_run(tmp_path, command=None)
    existing = tmp_path / "actual.txt"
    existing.write_text("actual", encoding="utf-8")
    exit_code = main(
        [
            "adopt-existing",
            str(run_dir),
            "single",
            "--output",
            f"result={existing}",
            "--source-run-id",
            "real-source",
        ]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "adopt_existing"
    assert payload["executed_external_command"] is False
