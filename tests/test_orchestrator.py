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


LAYERED_COMPLETION_INPUT_ROLES = {
    "frames_manifest",
    "cameras",
    "layered_completion_plan",
    "scene_gaussian",
    "scene_mesh",
    "masks_manifest",
    "captions_manifest",
    "object_clouds_manifest",
    "semantic_gaussian",
    "object_facts",
}


LAYERED_COMPLETION_OUTPUT_ROLES = {
    "completed_object_assets_manifest",
    "clean_scene_gaussian",
    "clean_scene_mesh",
    "final_clean_plate",
    "clean_plate_manifest",
    "layered_completion_report",
}


def _layered_completion_input_snapshots(tmp_path: Path, plan_path: Path) -> dict[str, object]:
    inputs: dict[str, object] = {
        "layered_completion_plan": _artifact_snapshot_payload(plan_path)
    }
    for role in sorted(LAYERED_COMPLETION_INPUT_ROLES - {"layered_completion_plan"}):
        path = tmp_path / f"{role}.input"
        if role == "scene_gaussian":
            path.write_bytes(_clean_gaussian_header())
        elif role == "scene_mesh":
            path.write_bytes(_clean_mesh_header())
        elif role == "semantic_gaussian":
            path.write_bytes(_semantic_gaussian_header())
        else:
            path.write_text(
                json.dumps({"records": [{"id": role}]}),
                encoding="utf-8",
            )
        inputs[role] = _artifact_snapshot_payload(path)
    return inputs


def _layered_completion_output_snapshots(tmp_path: Path, report_path: Path) -> dict[str, object]:
    outputs: dict[str, object] = {}
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    for role in sorted(
        LAYERED_COMPLETION_OUTPUT_ROLES - {"clean_plate_manifest", "layered_completion_report"}
    ):
        path = tmp_path / f"{role}.output"
        if role == "completed_object_assets_manifest":
            path.write_text(json.dumps(_valid_completed_object_assets_manifest()), encoding="utf-8")
        elif role == "clean_scene_gaussian":
            path.write_bytes(_clean_gaussian_header() + ("0 " * 13 + "0\n").encode("ascii"))
        elif role == "clean_scene_mesh":
            path.write_bytes(_valid_clean_mesh_payload())
        elif role == "final_clean_plate":
            path.write_bytes(b"final clean plate frame bytes\n")
        else:
            raise AssertionError(f"unhandled layered completion output role: {role}")
        outputs[role] = _artifact_snapshot_payload(path)
    final_clean_plate = _terminal_clean_plate_evidence_from_snapshot(
        outputs["final_clean_plate"]
    )
    report_payload["rounds"][-1]["output_clean_plate"] = final_clean_plate
    report_payload["final_clean_plate"] = final_clean_plate
    report_payload["clean_scene_gaussian"] = _clean_scene_asset_ref_from_snapshot(
        outputs["clean_scene_gaussian"],
        "clean_scene_gaussian",
    )
    report_payload["clean_scene_mesh"] = _clean_scene_asset_ref_from_snapshot(
        outputs["clean_scene_mesh"],
        "clean_scene_mesh",
    )
    clean_plate_manifest_path = tmp_path / "clean_plate_manifest.output"
    clean_plate_manifest_path.write_text(
        json.dumps(
            {
                "final_clean_plate": report_payload["final_clean_plate"],
                "frame_records": [{"frame_id": "000064"}],
            }
        ),
        encoding="utf-8",
    )
    outputs["clean_plate_manifest"] = _artifact_snapshot_payload(clean_plate_manifest_path)
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")
    outputs["layered_completion_report"] = _artifact_snapshot_payload(report_path)
    return outputs


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


def _object_completion_evidence(target_id: str, name: str, digest: str) -> dict[str, object]:
    evidence = _completion_evidence(name, digest)
    evidence["target_id"] = target_id
    return evidence


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


def _terminal_clean_plate_evidence_from_snapshot(snapshot: dict[str, object]) -> dict[str, object]:
    evidence = _terminal_clean_plate_evidence(str(snapshot["path"]), str(snapshot["sha256"])[0])
    evidence["uri"] = str(snapshot["path"])
    evidence["sha256"] = snapshot["sha256"]
    evidence["size_bytes"] = snapshot["size_bytes"]
    return evidence


def _clean_scene_asset_ref(role: str, digest: str, *, size_bytes: int = 1024) -> dict[str, object]:
    return {
        "uri": f"artifact://completion/{role}.ply",
        "sha256": digest * 64,
        "size_bytes": size_bytes,
        "media_type": "application/octet-stream",
        "role": role,
        "status": "validated",
    }


def _clean_scene_asset_ref_from_snapshot(
    snapshot: dict[str, object],
    role: str,
) -> dict[str, object]:
    return {
        "uri": str(snapshot["path"]),
        "sha256": snapshot["sha256"],
        "size_bytes": snapshot["size_bytes"],
        "media_type": "application/octet-stream",
        "role": role,
        "status": "validated",
    }


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
                    _object_completion_evidence(
                        "pillow-front",
                        "r1-pillow-trellis2.json",
                        "1",
                    )
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
        "clean_scene_gaussian": _clean_scene_asset_ref("clean_scene_gaussian", "6"),
        "clean_scene_mesh": _clean_scene_asset_ref("clean_scene_mesh", "7"),
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
    receipt_outputs = _layered_completion_output_snapshots(tmp_path, report_path)
    report_snapshot = snapshot_path(report_path)
    receipt_path = tmp_path / "provider-receipt.json"
    receipt_path.write_text(
        json.dumps(
            _provider_receipt_payload(
                inputs=_layered_completion_input_snapshots(tmp_path, plan_path),
                outputs=receipt_outputs,
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
                inputs=_layered_completion_input_snapshots(tmp_path, plan_path),
                outputs=_layered_completion_output_snapshots(tmp_path, report_path),
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
    outputs = _layered_completion_output_snapshots(tmp_path, report_path)
    bad_report_snapshot = dict(outputs["layered_completion_report"])
    bad_report_snapshot["sha256"] = "f" * 64
    outputs["layered_completion_report"] = bad_report_snapshot
    receipt_path = tmp_path / "provider-receipt.json"
    receipt_path.write_text(
        json.dumps(
            _provider_receipt_payload(
                inputs=_layered_completion_input_snapshots(tmp_path, plan_path),
                outputs=outputs,
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


def test_layered_completion_report_rechecks_receipt_only_output_snapshots(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_valid_layered_completion_plan()), encoding="utf-8")
    plan_sha = snapshot_path(plan_path).sha256
    report_payload = _valid_layered_completion_report()
    report_payload["completion_plan_sha256"] = plan_sha
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")
    outputs = _layered_completion_output_snapshots(tmp_path, report_path)
    bad_assets_snapshot = dict(outputs["completed_object_assets_manifest"])
    bad_assets_snapshot["sha256"] = "f" * 64
    outputs["completed_object_assets_manifest"] = bad_assets_snapshot
    receipt_path = tmp_path / "provider-receipt.json"
    receipt_path.write_text(
        json.dumps(
            _provider_receipt_payload(
                inputs=_layered_completion_input_snapshots(tmp_path, plan_path),
                outputs=outputs,
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ArtifactError,
        match="completed_object_assets_manifest output snapshot does not match",
    ):
        get_adapter("layered_completion").validate_outputs(
            {
                "layered_completion_report": snapshot_path(report_path),
                "provider_receipt": snapshot_path(receipt_path),
            }
        )


def test_layered_completion_report_requires_layered_provider_stage_id(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_valid_layered_completion_plan()), encoding="utf-8")
    plan_sha = snapshot_path(plan_path).sha256
    report_payload = _valid_layered_completion_report()
    report_payload["completion_plan_sha256"] = plan_sha
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")
    receipt = _provider_receipt_payload(
        inputs=_layered_completion_input_snapshots(tmp_path, plan_path),
        outputs=_layered_completion_output_snapshots(tmp_path, report_path),
    )
    receipt["provider_stage_id"] = "placement"
    receipt_path = tmp_path / "provider-receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ArtifactError, match="wrong provider_stage_id"):
        get_adapter("layered_completion").validate_outputs(
            {
                "layered_completion_report": snapshot_path(report_path),
                "provider_receipt": snapshot_path(receipt_path),
            }
        )


def test_layered_completion_report_requires_provider_receipt_input_snapshot(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_valid_layered_completion_plan()), encoding="utf-8")
    plan_sha = snapshot_path(plan_path).sha256
    report_payload = _valid_layered_completion_report()
    report_payload["completion_plan_sha256"] = plan_sha
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")
    inputs = _layered_completion_input_snapshots(tmp_path, plan_path)
    bad_plan_snapshot = dict(inputs["layered_completion_plan"])
    bad_plan_snapshot["size_bytes"] = bad_plan_snapshot["size_bytes"] + 1
    inputs["layered_completion_plan"] = bad_plan_snapshot
    receipt_path = tmp_path / "provider-receipt.json"
    receipt_path.write_text(
        json.dumps(
            _provider_receipt_payload(
                inputs=inputs,
                outputs=_layered_completion_output_snapshots(tmp_path, report_path),
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError, match="layered_completion_plan input snapshot"):
        get_adapter("layered_completion").validate_outputs(
            {
                "layered_completion_report": snapshot_path(report_path),
                "provider_receipt": snapshot_path(receipt_path),
            }
        )


def test_layered_completion_report_requires_complete_provider_receipt_outputs(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_valid_layered_completion_plan()), encoding="utf-8")
    plan_sha = snapshot_path(plan_path).sha256
    report_payload = _valid_layered_completion_report()
    report_payload["completion_plan_sha256"] = plan_sha
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")
    outputs = _layered_completion_output_snapshots(tmp_path, report_path)
    del outputs["clean_scene_mesh"]
    receipt_path = tmp_path / "provider-receipt.json"
    receipt_path.write_text(
        json.dumps(
            _provider_receipt_payload(
                inputs=_layered_completion_input_snapshots(tmp_path, plan_path),
                outputs=outputs,
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError, match="missing layered completion outputs"):
        get_adapter("layered_completion").validate_outputs(
            {
                "layered_completion_report": snapshot_path(report_path),
                "provider_receipt": snapshot_path(receipt_path),
            }
        )


def test_layered_completion_report_rejects_reused_output_role_paths(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_valid_layered_completion_plan()), encoding="utf-8")
    plan_sha = snapshot_path(plan_path).sha256
    report_payload = _valid_layered_completion_report()
    report_payload["completion_plan_sha256"] = plan_sha
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")
    outputs = _layered_completion_output_snapshots(tmp_path, report_path)
    outputs["clean_scene_mesh"] = dict(outputs["clean_scene_gaussian"])
    receipt_path = tmp_path / "provider-receipt.json"
    receipt_path.write_text(
        json.dumps(
            _provider_receipt_payload(
                inputs=_layered_completion_input_snapshots(tmp_path, plan_path),
                outputs=outputs,
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ArtifactError,
        match="output clean_scene_mesh path must be distinct from clean_scene_gaussian",
    ):
        get_adapter("layered_completion").validate_outputs(
            {
                "layered_completion_report": snapshot_path(report_path),
                "provider_receipt": snapshot_path(receipt_path),
            }
        )


def test_layered_completion_report_rejects_reused_output_role_artifacts(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_valid_layered_completion_plan()), encoding="utf-8")
    plan_sha = snapshot_path(plan_path).sha256
    report_payload = _valid_layered_completion_report()
    report_payload["completion_plan_sha256"] = plan_sha
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")
    outputs = _layered_completion_output_snapshots(tmp_path, report_path)
    payload = _valid_clean_gaussian_mesh_payload()
    gaussian_path = Path(outputs["clean_scene_gaussian"]["path"])
    mesh_path = Path(outputs["clean_scene_mesh"]["path"])
    gaussian_path.write_bytes(payload)
    mesh_path.write_bytes(payload)
    outputs["clean_scene_gaussian"] = _artifact_snapshot_payload(gaussian_path)
    outputs["clean_scene_mesh"] = _artifact_snapshot_payload(mesh_path)
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    report_payload["clean_scene_gaussian"] = _clean_scene_asset_ref_from_snapshot(
        outputs["clean_scene_gaussian"],
        "clean_scene_gaussian",
    )
    report_payload["clean_scene_mesh"] = _clean_scene_asset_ref_from_snapshot(
        outputs["clean_scene_mesh"],
        "clean_scene_mesh",
    )
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")
    outputs["layered_completion_report"] = _artifact_snapshot_payload(report_path)
    receipt_path = tmp_path / "provider-receipt.json"
    receipt_path.write_text(
        json.dumps(
            _provider_receipt_payload(
                inputs=_layered_completion_input_snapshots(tmp_path, plan_path),
                outputs=outputs,
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ArtifactError,
        match="output clean_scene_mesh artifact must be distinct from clean_scene_gaussian",
    ):
        get_adapter("layered_completion").validate_outputs(
            {
                "layered_completion_report": snapshot_path(report_path),
                "provider_receipt": snapshot_path(receipt_path),
            }
        )


def test_layered_completion_report_validates_receipt_output_semantics(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_valid_layered_completion_plan()), encoding="utf-8")
    plan_sha = snapshot_path(plan_path).sha256
    report_payload = _valid_layered_completion_report()
    report_payload["completion_plan_sha256"] = plan_sha
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")
    outputs = _layered_completion_output_snapshots(tmp_path, report_path)
    mesh_path = Path(outputs["clean_scene_mesh"]["path"])
    mesh_path.write_text("not a ply\n", encoding="utf-8")
    outputs["clean_scene_mesh"] = _artifact_snapshot_payload(mesh_path)
    receipt_path = tmp_path / "provider-receipt.json"
    receipt_path.write_text(
        json.dumps(
            _provider_receipt_payload(
                inputs=_layered_completion_input_snapshots(tmp_path, plan_path),
                outputs=outputs,
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError, match="valid PLY header"):
        get_adapter("layered_completion").validate_outputs(
            {
                "layered_completion_report": snapshot_path(report_path),
                "provider_receipt": snapshot_path(receipt_path),
            }
        )


def test_layered_completion_report_binds_completed_assets_to_planned_targets(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_valid_layered_completion_plan()), encoding="utf-8")
    plan_sha = snapshot_path(plan_path).sha256
    report_payload = _valid_layered_completion_report()
    report_payload["completion_plan_sha256"] = plan_sha
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")
    outputs = _layered_completion_output_snapshots(tmp_path, report_path)
    assets_path = Path(outputs["completed_object_assets_manifest"]["path"])
    assets_payload = _valid_completed_object_assets_manifest()
    assets_payload["objects"][0]["id"] = "wrong-pillow"
    assets_path.write_text(json.dumps(assets_payload), encoding="utf-8")
    outputs["completed_object_assets_manifest"] = _artifact_snapshot_payload(assets_path)
    receipt_path = tmp_path / "provider-receipt.json"
    receipt_path.write_text(
        json.dumps(
            _provider_receipt_payload(
                inputs=_layered_completion_input_snapshots(tmp_path, plan_path),
                outputs=outputs,
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError, match="objects differ from planned target ids"):
        get_adapter("layered_completion").validate_outputs(
            {
                "layered_completion_report": snapshot_path(report_path),
                "provider_receipt": snapshot_path(receipt_path),
            }
        )


def test_layered_completion_report_binds_completed_assets_scene_run_to_report(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_valid_layered_completion_plan()), encoding="utf-8")
    plan_sha = snapshot_path(plan_path).sha256
    report_payload = _valid_layered_completion_report()
    report_payload["completion_plan_sha256"] = plan_sha
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")
    outputs = _layered_completion_output_snapshots(tmp_path, report_path)
    assets_path = Path(outputs["completed_object_assets_manifest"]["path"])
    assets_payload = _valid_completed_object_assets_manifest()
    assets_payload["run_id"] = "archived-demo-run"
    assets_path.write_text(json.dumps(assets_payload), encoding="utf-8")
    outputs["completed_object_assets_manifest"] = _artifact_snapshot_payload(assets_path)
    receipt_path = tmp_path / "provider-receipt.json"
    receipt_path.write_text(
        json.dumps(
            _provider_receipt_payload(
                inputs=_layered_completion_input_snapshots(tmp_path, plan_path),
                outputs=outputs,
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError, match="assets_manifest scene/run differs"):
        get_adapter("layered_completion").validate_outputs(
            {
                "layered_completion_report": snapshot_path(report_path),
                "provider_receipt": snapshot_path(receipt_path),
            }
        )


def test_layered_completion_report_binds_clean_plate_manifest_to_final_clean_plate(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_valid_layered_completion_plan()), encoding="utf-8")
    plan_sha = snapshot_path(plan_path).sha256
    report_payload = _valid_layered_completion_report()
    report_payload["completion_plan_sha256"] = plan_sha
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")
    outputs = _layered_completion_output_snapshots(tmp_path, report_path)
    clean_plate_path = Path(outputs["clean_plate_manifest"]["path"])
    clean_plate_payload = json.loads(clean_plate_path.read_text(encoding="utf-8"))
    clean_plate_payload["final_clean_plate"]["sha256"] = "9" * 64
    clean_plate_path.write_text(json.dumps(clean_plate_payload), encoding="utf-8")
    outputs["clean_plate_manifest"] = _artifact_snapshot_payload(clean_plate_path)
    receipt_path = tmp_path / "provider-receipt.json"
    receipt_path.write_text(
        json.dumps(
            _provider_receipt_payload(
                inputs=_layered_completion_input_snapshots(tmp_path, plan_path),
                outputs=outputs,
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError, match="final_clean_plate differs from report"):
        get_adapter("layered_completion").validate_outputs(
            {
                "layered_completion_report": snapshot_path(report_path),
                "provider_receipt": snapshot_path(receipt_path),
            }
        )


def test_layered_completion_report_binds_clean_plate_manifest_final_scope(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_valid_layered_completion_plan()), encoding="utf-8")
    plan_sha = snapshot_path(plan_path).sha256
    report_payload = _valid_layered_completion_report()
    report_payload["completion_plan_sha256"] = plan_sha
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")
    outputs = _layered_completion_output_snapshots(tmp_path, report_path)
    clean_plate_path = Path(outputs["clean_plate_manifest"]["path"])
    clean_plate_payload = json.loads(clean_plate_path.read_text(encoding="utf-8"))
    clean_plate_payload["final_clean_plate"]["acceptance_scope"] = (
        "corrected_clean_plate_next_round_source_only"
    )
    clean_plate_payload["final_clean_plate"]["promotion_approved"] = False
    clean_plate_path.write_text(json.dumps(clean_plate_payload), encoding="utf-8")
    outputs["clean_plate_manifest"] = _artifact_snapshot_payload(clean_plate_path)
    receipt_path = tmp_path / "provider-receipt.json"
    receipt_path.write_text(
        json.dumps(
            _provider_receipt_payload(
                inputs=_layered_completion_input_snapshots(tmp_path, plan_path),
                outputs=outputs,
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError, match="final_clean_plate differs from report"):
        get_adapter("layered_completion").validate_outputs(
            {
                "layered_completion_report": snapshot_path(report_path),
                "provider_receipt": snapshot_path(receipt_path),
            }
        )


def test_layered_completion_report_binds_final_clean_plate_to_receipt_output(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_valid_layered_completion_plan()), encoding="utf-8")
    plan_sha = snapshot_path(plan_path).sha256
    report_payload = _valid_layered_completion_report()
    report_payload["completion_plan_sha256"] = plan_sha
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")
    outputs = _layered_completion_output_snapshots(tmp_path, report_path)
    stale_final_clean_plate = _terminal_clean_plate_evidence("stale-final-clean-plate.png", "9")
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    report_payload["rounds"][-1]["output_clean_plate"] = stale_final_clean_plate
    report_payload["final_clean_plate"] = stale_final_clean_plate
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")
    outputs["layered_completion_report"] = _artifact_snapshot_payload(report_path)
    clean_plate_path = Path(outputs["clean_plate_manifest"]["path"])
    clean_plate_payload = json.loads(clean_plate_path.read_text(encoding="utf-8"))
    clean_plate_payload["final_clean_plate"] = stale_final_clean_plate
    clean_plate_path.write_text(json.dumps(clean_plate_payload), encoding="utf-8")
    outputs["clean_plate_manifest"] = _artifact_snapshot_payload(clean_plate_path)
    receipt_path = tmp_path / "provider-receipt.json"
    receipt_path.write_text(
        json.dumps(
            _provider_receipt_payload(
                inputs=_layered_completion_input_snapshots(tmp_path, plan_path),
                outputs=outputs,
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError, match="final_clean_plate report asset uri differs"):
        get_adapter("layered_completion").validate_outputs(
            {
                "layered_completion_report": snapshot_path(report_path),
                "provider_receipt": snapshot_path(receipt_path),
            }
        )


def test_layered_completion_report_binds_clean_scene_assets_to_report(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_valid_layered_completion_plan()), encoding="utf-8")
    plan_sha = snapshot_path(plan_path).sha256
    report_payload = _valid_layered_completion_report()
    report_payload["completion_plan_sha256"] = plan_sha
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")
    outputs = _layered_completion_output_snapshots(tmp_path, report_path)
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    report_payload["clean_scene_mesh"]["sha256"] = "9" * 64
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")
    outputs["layered_completion_report"] = _artifact_snapshot_payload(report_path)
    receipt_path = tmp_path / "provider-receipt.json"
    receipt_path.write_text(
        json.dumps(
            _provider_receipt_payload(
                inputs=_layered_completion_input_snapshots(tmp_path, plan_path),
                outputs=outputs,
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError, match="clean_scene_mesh report asset differs"):
        get_adapter("layered_completion").validate_outputs(
            {
                "layered_completion_report": snapshot_path(report_path),
                "provider_receipt": snapshot_path(receipt_path),
            }
        )


def test_layered_completion_report_binds_clean_scene_asset_uri_to_receipt_path(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_valid_layered_completion_plan()), encoding="utf-8")
    plan_sha = snapshot_path(plan_path).sha256
    report_payload = _valid_layered_completion_report()
    report_payload["completion_plan_sha256"] = plan_sha
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")
    outputs = _layered_completion_output_snapshots(tmp_path, report_path)
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    report_payload["clean_scene_gaussian"]["uri"] = "artifact://completion/stale-clean-scene.ply"
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")
    outputs["layered_completion_report"] = _artifact_snapshot_payload(report_path)
    receipt_path = tmp_path / "provider-receipt.json"
    receipt_path.write_text(
        json.dumps(
            _provider_receipt_payload(
                inputs=_layered_completion_input_snapshots(tmp_path, plan_path),
                outputs=outputs,
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError, match="clean_scene_gaussian report asset uri differs"):
        get_adapter("layered_completion").validate_outputs(
            {
                "layered_completion_report": snapshot_path(report_path),
                "provider_receipt": snapshot_path(receipt_path),
            }
        )


@pytest.mark.parametrize(
    ("input_role", "output_role", "payload_kind", "expected"),
    [
        (
            "scene_gaussian",
            "clean_scene_gaussian",
            "gaussian",
            "clean_scene_gaussian must not reuse scene_gaussian input artifact",
        ),
        (
            "semantic_gaussian",
            "clean_scene_gaussian",
            "semantic_gaussian",
            "clean_scene_gaussian must not reuse semantic_gaussian input artifact",
        ),
        (
            "scene_mesh",
            "clean_scene_mesh",
            "mesh",
            "clean_scene_mesh must not reuse scene_mesh input artifact",
        ),
    ],
)
def test_layered_completion_report_rejects_clean_scene_reusing_raw_scene_inputs(
    tmp_path: Path,
    input_role: str,
    output_role: str,
    payload_kind: str,
    expected: str,
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_valid_layered_completion_plan()), encoding="utf-8")
    plan_sha = snapshot_path(plan_path).sha256
    report_payload = _valid_layered_completion_report()
    report_payload["completion_plan_sha256"] = plan_sha
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")
    inputs = _layered_completion_input_snapshots(tmp_path, plan_path)
    outputs = _layered_completion_output_snapshots(tmp_path, report_path)
    if payload_kind == "gaussian":
        payload = _clean_gaussian_header() + ("0 " * 13 + "0\n").encode("ascii")
    elif payload_kind == "semantic_gaussian":
        payload = _semantic_gaussian_header() + ("0 " * 15 + "0\n").encode("ascii")
    else:
        payload = _valid_clean_mesh_payload()
    reused_path = Path(inputs[input_role]["path"])
    reused_path.write_bytes(payload)
    output_path = Path(outputs[output_role]["path"])
    output_path.write_bytes(payload)
    inputs[input_role] = _artifact_snapshot_payload(reused_path)
    outputs[output_role] = _artifact_snapshot_payload(output_path)
    receipt_path = tmp_path / "provider-receipt.json"
    receipt_path.write_text(
        json.dumps(
            _provider_receipt_payload(
                inputs=inputs,
                outputs=outputs,
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError, match=expected):
        get_adapter("layered_completion").validate_outputs(
            {
                "layered_completion_report": snapshot_path(report_path),
                "provider_receipt": snapshot_path(receipt_path),
            }
        )


@pytest.mark.parametrize(
    ("input_role", "output_role", "payload_kind", "expected"),
    [
        (
            "scene_gaussian",
            "clean_scene_gaussian",
            "gaussian",
            "clean_scene_gaussian path must differ from scene_gaussian input path",
        ),
        (
            "semantic_gaussian",
            "clean_scene_gaussian",
            "semantic_gaussian",
            "clean_scene_gaussian path must differ from semantic_gaussian input path",
        ),
        (
            "scene_mesh",
            "clean_scene_mesh",
            "mesh",
            "clean_scene_mesh path must differ from scene_mesh input path",
        ),
    ],
)
def test_layered_completion_report_rejects_clean_scene_reusing_raw_scene_input_paths(
    tmp_path: Path,
    input_role: str,
    output_role: str,
    payload_kind: str,
    expected: str,
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_valid_layered_completion_plan()), encoding="utf-8")
    plan_sha = snapshot_path(plan_path).sha256
    report_payload = _valid_layered_completion_report()
    report_payload["completion_plan_sha256"] = plan_sha
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")
    inputs = _layered_completion_input_snapshots(tmp_path, plan_path)
    outputs = _layered_completion_output_snapshots(tmp_path, report_path)
    if payload_kind == "gaussian":
        payload = _clean_gaussian_header() + ("0 " * 13 + "0\n").encode("ascii")
    elif payload_kind == "semantic_gaussian":
        payload = _semantic_gaussian_header() + ("0 " * 15 + "0\n").encode("ascii")
    else:
        payload = _valid_clean_mesh_payload()
    reused_path = Path(inputs[input_role]["path"])
    reused_path.write_bytes(payload)
    reused_snapshot = _artifact_snapshot_payload(reused_path)
    inputs[input_role] = reused_snapshot
    outputs[output_role] = reused_snapshot
    receipt_path = tmp_path / "provider-receipt.json"
    receipt_path.write_text(
        json.dumps(
            _provider_receipt_payload(
                inputs=inputs,
                outputs=outputs,
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError, match=expected):
        get_adapter("layered_completion").validate_outputs(
            {
                "layered_completion_report": snapshot_path(report_path),
                "provider_receipt": snapshot_path(receipt_path),
            }
        )


def test_layered_completion_report_validates_receipt_input_semantics(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_valid_layered_completion_plan()), encoding="utf-8")
    plan_sha = snapshot_path(plan_path).sha256
    report_payload = _valid_layered_completion_report()
    report_payload["completion_plan_sha256"] = plan_sha
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")
    inputs = _layered_completion_input_snapshots(tmp_path, plan_path)
    semantic_gaussian_path = Path(inputs["semantic_gaussian"]["path"])
    semantic_gaussian_path.write_bytes(_clean_gaussian_header())
    inputs["semantic_gaussian"] = _artifact_snapshot_payload(semantic_gaussian_path)
    receipt_path = tmp_path / "provider-receipt.json"
    receipt_path.write_text(
        json.dumps(
            _provider_receipt_payload(
                inputs=inputs,
                outputs=_layered_completion_output_snapshots(tmp_path, report_path),
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError, match="semantic_gaussian is missing fields"):
        get_adapter("layered_completion").validate_outputs(
            {
                "layered_completion_report": snapshot_path(report_path),
                "provider_receipt": snapshot_path(receipt_path),
            }
        )


def test_layered_completion_report_rejects_reused_input_role_paths(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_valid_layered_completion_plan()), encoding="utf-8")
    plan_sha = snapshot_path(plan_path).sha256
    report_payload = _valid_layered_completion_report()
    report_payload["completion_plan_sha256"] = plan_sha
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")
    inputs = _layered_completion_input_snapshots(tmp_path, plan_path)
    outputs = _layered_completion_output_snapshots(tmp_path, report_path)
    inputs["masks_manifest"] = dict(inputs["captions_manifest"])
    receipt_path = tmp_path / "provider-receipt.json"
    receipt_path.write_text(
        json.dumps(
            _provider_receipt_payload(
                inputs=inputs,
                outputs=outputs,
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ArtifactError,
        match="provider receipt input masks_manifest path must be distinct from captions_manifest",
    ):
        get_adapter("layered_completion").validate_outputs(
            {
                "layered_completion_report": snapshot_path(report_path),
                "provider_receipt": snapshot_path(receipt_path),
            }
        )


def test_layered_completion_report_rejects_reused_input_role_artifacts(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_valid_layered_completion_plan()), encoding="utf-8")
    plan_sha = snapshot_path(plan_path).sha256
    report_payload = _valid_layered_completion_report()
    report_payload["completion_plan_sha256"] = plan_sha
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")
    inputs = _layered_completion_input_snapshots(tmp_path, plan_path)
    outputs = _layered_completion_output_snapshots(tmp_path, report_path)
    payload = json.dumps({"records": [{"id": "same-sam3-placeholder"}]})
    captions_path = Path(inputs["captions_manifest"]["path"])
    masks_path = Path(inputs["masks_manifest"]["path"])
    captions_path.write_text(payload, encoding="utf-8")
    masks_path.write_text(payload, encoding="utf-8")
    inputs["captions_manifest"] = _artifact_snapshot_payload(captions_path)
    inputs["masks_manifest"] = _artifact_snapshot_payload(masks_path)
    receipt_path = tmp_path / "provider-receipt.json"
    receipt_path.write_text(
        json.dumps(
            _provider_receipt_payload(
                inputs=inputs,
                outputs=outputs,
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ArtifactError,
        match=(
            "provider receipt input masks_manifest artifact must be distinct from "
            "captions_manifest"
        ),
    ):
        get_adapter("layered_completion").validate_outputs(
            {
                "layered_completion_report": snapshot_path(report_path),
                "provider_receipt": snapshot_path(receipt_path),
            }
        )


def test_layered_completion_report_rejects_semantic_gaussian_reusing_scene_gaussian_input(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_valid_layered_completion_plan()), encoding="utf-8")
    plan_sha = snapshot_path(plan_path).sha256
    report_payload = _valid_layered_completion_report()
    report_payload["completion_plan_sha256"] = plan_sha
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")
    inputs = _layered_completion_input_snapshots(tmp_path, plan_path)
    outputs = _layered_completion_output_snapshots(tmp_path, report_path)
    payload = _semantic_gaussian_header() + ("0 " * 15 + "0\n").encode("ascii")
    scene_path = Path(inputs["scene_gaussian"]["path"])
    semantic_path = Path(inputs["semantic_gaussian"]["path"])
    scene_path.write_bytes(payload)
    semantic_path.write_bytes(payload)
    inputs["scene_gaussian"] = _artifact_snapshot_payload(scene_path)
    inputs["semantic_gaussian"] = _artifact_snapshot_payload(semantic_path)
    receipt_path = tmp_path / "provider-receipt.json"
    receipt_path.write_text(
        json.dumps(
            _provider_receipt_payload(
                inputs=inputs,
                outputs=outputs,
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ArtifactError,
        match=(
            "provider receipt input semantic_gaussian artifact must be distinct from "
            "scene_gaussian"
        ),
    ):
        get_adapter("layered_completion").validate_outputs(
            {
                "layered_completion_report": snapshot_path(report_path),
                "provider_receipt": snapshot_path(receipt_path),
            }
        )


def test_layered_completion_report_rejects_semantic_gaussian_reusing_scene_gaussian_path(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_valid_layered_completion_plan()), encoding="utf-8")
    plan_sha = snapshot_path(plan_path).sha256
    report_payload = _valid_layered_completion_report()
    report_payload["completion_plan_sha256"] = plan_sha
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")
    inputs = _layered_completion_input_snapshots(tmp_path, plan_path)
    outputs = _layered_completion_output_snapshots(tmp_path, report_path)
    reused_path = Path(inputs["scene_gaussian"]["path"])
    reused_path.write_bytes(_semantic_gaussian_header() + ("0 " * 15 + "0\n").encode("ascii"))
    reused_snapshot = _artifact_snapshot_payload(reused_path)
    inputs["scene_gaussian"] = reused_snapshot
    inputs["semantic_gaussian"] = reused_snapshot
    receipt_path = tmp_path / "provider-receipt.json"
    receipt_path.write_text(
        json.dumps(
            _provider_receipt_payload(
                inputs=inputs,
                outputs=outputs,
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ArtifactError,
        match="provider receipt input semantic_gaussian path must be distinct from scene_gaussian",
    ):
        get_adapter("layered_completion").validate_outputs(
            {
                "layered_completion_report": snapshot_path(report_path),
                "provider_receipt": snapshot_path(receipt_path),
            }
        )


def test_layered_completion_report_requires_complete_provider_receipt_inputs(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_valid_layered_completion_plan()), encoding="utf-8")
    plan_sha = snapshot_path(plan_path).sha256
    report_payload = _valid_layered_completion_report()
    report_payload["completion_plan_sha256"] = plan_sha
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")
    inputs = _layered_completion_input_snapshots(tmp_path, plan_path)
    del inputs["semantic_gaussian"]
    receipt_path = tmp_path / "provider-receipt.json"
    receipt_path.write_text(
        json.dumps(
            _provider_receipt_payload(
                inputs=inputs,
                outputs=_layered_completion_output_snapshots(tmp_path, report_path),
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError, match="missing layered completion inputs"):
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
            "final_clean_plate does not match the terminal round output",
        ),
        (
            lambda payload: payload["final_clean_plate"].__setitem__(
                "uri", "artifact://completion/stale-final-clean-plate.json"
            ),
            "final_clean_plate does not match the terminal round output",
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


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (
            lambda payload: payload["rounds"][0].__setitem__(
                "sam3_receipt",
                dict(payload["rounds"][0]["scene_audit_receipt"]),
            ),
            "round 1 sam3_receipt must be distinct from round 1 scene_audit_receipt",
        ),
        (
            lambda payload: payload["rounds"][1].__setitem__(
                "quality_report",
                dict(payload["rounds"][0]["quality_report"]),
            ),
            "round 2 quality_report must be distinct from round 1 quality_report",
        ),
        (
            lambda payload: payload["rounds"][0].__setitem__(
                "object_completion_receipts",
                [
                    {
                        **dict(payload["rounds"][0]["quality_report"]),
                        "target_id": "pillow-front",
                    }
                ],
            ),
            r"round 1 object_completion_receipts\[1\] must be distinct "
            "from round 1 quality_report",
        ),
        (
            lambda payload: payload["rounds"][1].__setitem__(
                "background_rebuild_receipt",
                {
                    key: value
                    for key, value in payload["rounds"][0][
                        "object_completion_receipts"
                    ][0].items()
                    if key != "target_id"
                },
            ),
            "round 2 background_rebuild_receipt must be distinct from "
            r"round 1 object_completion_receipts\[1\]",
        ),
        (
            lambda payload: payload["rounds"][0].__setitem__(
                "quality_report",
                dict(payload["initial_clean_plate"]),
            ),
            "round 1 quality_report must not reuse a clean plate artifact",
        ),
        (
            lambda payload: payload["rounds"][0].__setitem__(
                "output_clean_plate",
                {
                    **dict(payload["rounds"][0]["output_clean_plate"]),
                    "sha256": payload["rounds"][0]["quality_report"]["sha256"],
                },
            ),
            "round 1 output_clean_plate must be distinct from round 1 quality_report",
        ),
    ],
)
def test_layered_completion_report_rejects_reused_execution_evidence(
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
            lambda payload: payload["rounds"][0]["object_completion_receipts"][0].pop(
                "target_id"
            ),
            "target_id",
        ),
        (
            lambda payload: payload["rounds"][0]["object_completion_receipts"][0].__setitem__(
                "target_id",
                "wrong-pillow",
            ),
            "round 1 object completion receipts must match target_ids",
        ),
        (
            lambda payload: payload["rounds"][0].__setitem__(
                "target_ids",
                ["pillow-front", "pillow-left"],
            ),
            "round 1 object completion receipts must match target_ids",
        ),
        (
            lambda payload: payload["rounds"][0].__setitem__(
                "object_completion_receipts",
                [
                    _object_completion_evidence("pillow-front", "r1-a.json", "6"),
                    _object_completion_evidence("pillow-front", "r1-b.json", "7"),
                ],
            ),
            "round 1 object completion receipts duplicate targets",
        ),
    ],
)
def test_layered_completion_report_binds_object_completion_receipts_to_targets(
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
        "run_id": "full-layered-run",
        "created_at": "2026-07-17T00:00:00Z",
        "objects": [
            {
                "id": "pillow-front",
                "representation_mode": "unified_pbr_glb",
                "unified_pbr_glb": {
                    "uri": "artifact://pillow-front.glb",
                    "sha256": "a" * 64,
                    "size_bytes": 1024,
                    "media_type": "model/gltf-binary",
                    "role": "unified_pbr_glb",
                    "status": "validated",
                    "provenance": {
                        "faces": 97082,
                        "watertight": False,
                        "closed_volume_claim": False,
                        "inside_outside_queries_allowed": False,
                        "technical_gates": {
                            "finite_vertices": True,
                            "valid_triangle_indices": True,
                            "no_degenerate_faces": True,
                            "winding_consistent": True,
                            "pbr_material_present": True,
                            "positive_extents": True,
                        },
                    },
                },
                "collision_topology": "surface_bvh",
                "geometry_complete_verified": True,
                "completion_report_uri": "artifact://pillow-front/report.json",
            }
        ],
    }


@pytest.mark.parametrize(
    "missing_field",
    ["unified_pbr_glb", "collision_topology", "geometry_complete_verified"],
)
def test_completed_object_assets_manifest_uses_unified_pbr_typed_contract(
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


def test_completed_object_assets_manifest_requires_explicit_legacy_policy_for_separate_assets(
    tmp_path: Path,
) -> None:
    output = tmp_path / "completed-object-assets.json"
    payload = {
        "scene_id": "bedroom_4",
        "run_id": "full-layered-run",
        "created_at": "2026-07-17T00:00:00Z",
        "objects": [
            {
                "id": "pillow-front",
                "representation_mode": "separate_render_and_collider",
                "render_mesh": {
                    "uri": "artifact://pillow-front.glb",
                    "sha256": "a" * 64,
                    "size_bytes": 1024,
                    "role": "render_mesh",
                    "status": "validated",
                },
                "collider": {
                    "uri": "artifact://pillow-front-collider.glb",
                    "sha256": "b" * 64,
                    "size_bytes": 512,
                    "role": "collider",
                    "status": "validated",
                },
                "closed_surface_verified": True,
                "completion_report_uri": "artifact://pillow-front/report.json",
            }
        ],
    }
    output.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ArtifactError, match="requires unified_pbr_glb representation"):
        get_adapter("layered_completion").validate_outputs(
            {"completed_object_assets_manifest": snapshot_path(output)}
        )

    payload["representation_policy"] = "mesh_first_optional_gaussian"
    output.write_text(json.dumps(payload), encoding="utf-8")
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


def _semantic_gaussian_header(format_name: str = "ascii") -> bytes:
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
        "object_id",
        "object_probability",
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


def _valid_clean_mesh_payload() -> bytes:
    return _clean_mesh_header() + struct.pack(
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


def _valid_clean_gaussian_mesh_payload() -> bytes:
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
    lines = [
        "ply",
        "format ascii 1.0",
        "element vertex 3",
        *(f"property float {name}" for name in properties),
        "element face 1",
        "property list uchar int vertex_indices",
        "end_header",
        "0 0 0 0 0 0 1 1 1 1 1 0 0 0",
        "1 0 0 0 0 0 1 1 1 1 1 0 0 0",
        "0 1 0 0 0 0 1 1 1 1 1 0 0 0",
        "3 0 1 2",
    ]
    return ("\n".join(lines) + "\n").encode("ascii")


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
