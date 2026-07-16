from __future__ import annotations

import json
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
    assert config.topological_order() == [
        "ingest",
        "da3",
        "sam3",
        "pgsr",
        "fusion",
        "cognition",
        "trellis",
        "placement",
        "bundle",
        "web",
    ]
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
