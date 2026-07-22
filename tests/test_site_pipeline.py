from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from video2world.adapters import get_adapter
from video2world.cli import main
from video2world.config import load_run_config
from video2world.errors import ConfigurationError, StageBlockedError
from video2world.orchestrator import PipelineOrchestrator
from video2world.site_pipeline import (
    CANONICAL_STAGE_INPUTS,
    SITE_RUN_RECEIPT_NAME,
    create_site_run,
    load_site_binding,
    load_site_profile,
    preflight_site_run,
    run_site_pipeline,
)
from video2world.state import StageState, load_stage_state, write_stage_state


def _write_provider_driver(provider_root: Path) -> Path:
    (provider_root / "README.txt").write_text("site provider fixture\n", encoding="utf-8")
    driver = provider_root / "driver.py"
    driver.write_text(
        "from pathlib import Path\n"
        "import json\n"
        "import sys\n"
        "stage = sys.argv[sys.argv.index('--stage') + 1]\n"
        "Path(sys.argv[sys.argv.index('--marker') + 1]).write_text(stage)\n"
        "for index, value in enumerate(sys.argv):\n"
        "    if value != '--output':\n"
        "        continue\n"
        "    role, path = sys.argv[index + 1].split('=', 1)\n"
        "    output = Path(path)\n"
        "    output.parent.mkdir(parents=True, exist_ok=True)\n"
        "    output.write_text(json.dumps({'role': role, 'records': [{'id': 'real'}]}))\n",
        encoding="utf-8",
    )
    return driver


def _write_contract(path: Path, provider_root: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    pipeline = load_run_config(root / "video2world/configs/default_pipeline.yaml")
    stages = {}
    for stage_id, stage in pipeline.stages.items():
        output_roles = sorted(get_adapter(stage.adapter).required_outputs)
        command = [
            sys.executable,
            "${PROVIDER_ROOT}/driver.py",
            "--stage",
            stage_id,
            "--marker",
            "${PROVIDER_ROOT}/launched.txt",
        ]
        for role in output_roles:
            command.extend(["--output", f"{role}={{output_{role}}}"])
        stages[stage_id] = {
            "provider": f"test-provider-{stage_id}",
            "command": command,
            "input_roles": list(CANONICAL_STAGE_INPUTS[stage_id]),
            "output_roles": output_roles,
            "requires": {
                "python": {"path": sys.executable, "kind": "executable"},
                "provider_root": {
                    "path": "${PROVIDER_ROOT}",
                    "kind": "directory",
                },
                "driver": {
                    "path": "${PROVIDER_ROOT}/driver.py",
                    "kind": "file",
                },
            },
        }
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "provider_id": "site-test-provider",
                "scope": "generic_interface",
                "verification_status": "requires_site_configuration",
                "stages": stages,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_profile(
    path: Path,
    *,
    adopt_ingest: bool = False,
    video_sha256: str | None = None,
) -> None:
    root = Path(__file__).resolve().parents[1]
    pipeline = load_run_config(root / "video2world/configs/default_pipeline.yaml")
    stages: dict[str, object] = {
        stage_id: {"mode": "execute", "provider_stage": stage_id} for stage_id in pipeline.stages
    }
    if adopt_ingest:
        assert video_sha256 is not None
        stages["ingest"] = {
            "mode": "adopt",
            "artifact_root": "adopted",
            "source_run_id": "real-upstream-run",
            "source_repository": "https://example.invalid/audited-upstream",
            "source_commit": "a" * 40,
            "outputs": {
                "frames_manifest": "frames.json",
                "cameras": "cameras.json",
            },
            "expected_inputs": {"video": video_sha256},
            "note": "Test fixture representing pre-existing, content-hashed outputs.",
        }
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "profile_id": "site-test",
                "description": "Twelve-stage test profile",
                "provider_environment": {
                    "PROVIDER_ROOT": {
                        "root_kind": "checkout",
                        "root": "provider",
                        "expected_kind": "directory",
                    }
                },
                "stages": stages,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _site_fixture(tmp_path: Path, *, adopt_ingest: bool = False) -> dict[str, Path]:
    provider_root = tmp_path / "provider"
    provider_root.mkdir()
    _write_provider_driver(provider_root)
    contract = tmp_path / "provider.yaml"
    _write_contract(contract, provider_root)
    video = tmp_path / "input.mp4"
    video.write_bytes(b"real-video")
    profile = tmp_path / "profile.yaml"
    _write_profile(
        profile,
        adopt_ingest=adopt_ingest,
        video_sha256=hashlib.sha256(b"real-video").hexdigest(),
    )
    result = {
        "provider": provider_root,
        "contract": contract,
        "profile": profile,
        "video": video,
        "run": tmp_path / "run",
    }
    if adopt_ingest:
        adopted = tmp_path / "adopted"
        adopted.mkdir()
        (adopted / "frames.json").write_text(
            json.dumps({"frames": [{"id": "000000"}]}), encoding="utf-8"
        )
        (adopted / "cameras.json").write_text(
            json.dumps({"cameras": [{"id": "000000"}]}), encoding="utf-8"
        )
        result["adopted"] = adopted
    return result


def test_site_profile_layered_completion_adoption_requires_provider_receipt(
    tmp_path: Path,
) -> None:
    fixture = _site_fixture(tmp_path)
    profile_payload = yaml.safe_load(fixture["profile"].read_text(encoding="utf-8"))
    layered_outputs = {
        role: f"{role}.json"
        for role in get_adapter("layered_completion").required_outputs
        if role not in {"clean_scene_gaussian", "clean_scene_mesh"}
    }
    layered_outputs["clean_scene_gaussian"] = "clean-scene-gaussian.ply"
    layered_outputs["clean_scene_mesh"] = "clean-scene-mesh.ply"
    assert "provider_receipt" not in layered_outputs
    profile_payload["stages"]["layered_completion"] = {
        "mode": "adopt",
        "artifact_root": "layered",
        "source_run_id": "archived-current-demo-only",
        "outputs": layered_outputs,
    }
    profile = tmp_path / "profile-layered-adopt-missing-receipt.yaml"
    profile.write_text(yaml.safe_dump(profile_payload, sort_keys=False), encoding="utf-8")
    layered_root = tmp_path / "layered"
    layered_root.mkdir()

    with pytest.raises(ConfigurationError, match="provider_receipt"):
        create_site_run(
            fixture["run"],
            video=fixture["video"],
            scene_id="scene",
            run_id="site-run-1",
            profile_path=profile,
            provider_contract_path=fixture["contract"],
            checkout_roots={"provider": fixture["provider"]},
            artifact_roots={"layered": layered_root},
        )


def test_site_cli_binds_all_twelve_stages_and_executes_real_ingest(tmp_path: Path, capsys) -> None:
    fixture = _site_fixture(tmp_path)
    exit_code = main(
        [
            "site-init",
            str(fixture["run"]),
            "--video",
            str(fixture["video"]),
            "--scene-id",
            "scene",
            "--run-id",
            "site-run-1",
            "--profile",
            str(fixture["profile"]),
            "--provider-contract",
            str(fixture["contract"]),
            "--checkout",
            f"provider={fixture['provider']}",
        ]
    )
    assert exit_code == 0
    initialized = json.loads(capsys.readouterr().out)
    assert initialized["stage_count"] == 12
    assert initialized["all_stages_have_commands"] is True
    assert initialized["models_launched"] is False

    config = load_run_config(fixture["run"] / "run.yaml")
    assert set(config.stages) == set(CANONICAL_STAGE_INPUTS)
    assert all(stage.command for stage in config.stages.values())
    assert all("provider_receipt" in stage.outputs for stage in config.stages.values())

    preflight = preflight_site_run(fixture["run"], targets=["ingest"])
    assert preflight["status"] == "passed"
    assert preflight["models_launched"] is False
    assert not (fixture["provider"] / "launched.txt").exists()

    receipt = run_site_pipeline(fixture["run"], targets=["ingest"])
    assert receipt["status"] == "completed"
    assert receipt["complete_pipeline"] is False
    assert receipt["stages"] == [
        {
            "stage_id": "ingest",
            "action": "executed",
            "status": "succeeded",
            "mode": "executed",
            "state_sha256": receipt["stages"][0]["state_sha256"],
            "output_roles": ["cameras", "frames_manifest", "provider_receipt"],
        }
    ]
    assert (fixture["provider"] / "launched.txt").read_text() == "ingest"
    provider_receipt = json.loads(
        (fixture["run"] / "artifacts/ingest/provider-receipt.json").read_text()
    )
    assert provider_receipt["status"] == "completed"
    assert provider_receipt["provider_stage_id"] == "ingest"


def test_site_runner_records_adoption_without_launching_provider(tmp_path: Path) -> None:
    fixture = _site_fixture(tmp_path, adopt_ingest=True)
    create_site_run(
        fixture["run"],
        video=fixture["video"],
        scene_id="scene",
        run_id="site-adoption-1",
        profile_path=fixture["profile"],
        provider_contract_path=fixture["contract"],
        checkout_roots={"provider": fixture["provider"]},
        artifact_roots={"adopted": fixture["adopted"]},
    )

    config = load_run_config(fixture["run"] / "run.yaml")
    assert config.stages["ingest"].command
    assert "provider_receipt" not in config.stages["ingest"].outputs
    assert preflight_site_run(fixture["run"], targets=["ingest"])["status"] == "passed"

    receipt = run_site_pipeline(fixture["run"], targets=["ingest"])
    state = load_stage_state(fixture["run"], "ingest")
    assert state is not None
    assert state.status == "adopted"
    assert state.mode == "adopted"
    assert state.adoption is not None
    assert state.adoption.source_run_id == "real-upstream-run"
    assert receipt["stages"][0]["action"] == "adopted"
    assert not (fixture["provider"] / "launched.txt").exists()


def test_site_adoption_requires_source_bound_input_hashes(tmp_path: Path) -> None:
    fixture = _site_fixture(tmp_path, adopt_ingest=True)
    create_site_run(
        fixture["run"],
        video=fixture["video"],
        scene_id="scene",
        profile_path=fixture["profile"],
        provider_contract_path=fixture["contract"],
        checkout_roots={"provider": fixture["provider"]},
        artifact_roots={"adopted": fixture["adopted"]},
    )
    fixture["video"].write_bytes(b"different-video")

    preflight = preflight_site_run(fixture["run"], targets=["ingest"])
    assert preflight["status"] == "blocked"
    assert "ingest.video SHA-256 mismatch" in preflight["stages"][0]["reason"]
    with pytest.raises(StageBlockedError, match="site preflight failed"):
        run_site_pipeline(fixture["run"], targets=["ingest"])
    assert load_stage_state(fixture["run"], "ingest") is None


def test_site_preflight_surfaces_failed_stage_recovery_actions(tmp_path: Path) -> None:
    fixture = _site_fixture(tmp_path)
    create_site_run(
        fixture["run"],
        video=fixture["video"],
        scene_id="scene",
        run_id="site-recovery-1",
        profile_path=fixture["profile"],
        provider_contract_path=fixture["contract"],
        checkout_roots={"provider": fixture["provider"]},
        artifact_roots={},
    )
    orchestrator = PipelineOrchestrator(fixture["run"])
    plan = next(item for item in orchestrator.plan(["ingest"]) if item.stage_id == "ingest")
    provider_receipt = Path(plan.outputs["provider_receipt"])
    provider_receipt.parent.mkdir(parents=True, exist_ok=True)
    provider_receipt.write_text(
        json.dumps(
            {
                "recovery_actions": [
                    {
                        "priority": 1,
                        "stage": "donor_support",
                        "action": (
                            "add_observed_donor_or_switch_to_constrained_generation_for_residual"
                        ),
                        "frame_ids": ["000064"],
                        "allow_deeper_rounds": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    now = datetime.now(UTC)
    write_stage_state(
        fixture["run"],
        StageState(
            stage_id="ingest",
            adapter="holi_ingest",
            status="failed",
            mode="executed",
            attempt=1,
            input_digest="0" * 64,
            config_digest="0" * 64,
            command=["fixture"],
            started_at=now,
            finished_at=now,
            error="fixture failed after writing route recovery actions",
        ),
    )

    preflight = preflight_site_run(fixture["run"], targets=["ingest"])

    assert preflight["status"] == "passed"
    assert preflight["stages"][0]["recovery_actions"][0]["stage"] == "donor_support"
    assert preflight["stages"][0]["recovery_actions"][0]["frame_ids"] == ["000064"]


def test_site_run_blocked_receipt_summarizes_preflight_recovery_actions(
    tmp_path: Path,
) -> None:
    fixture = _site_fixture(tmp_path)
    create_site_run(
        fixture["run"],
        video=fixture["video"],
        scene_id="scene",
        run_id="site-recovery-blocked-1",
        profile_path=fixture["profile"],
        provider_contract_path=fixture["contract"],
        checkout_roots={"provider": fixture["provider"]},
        artifact_roots={},
    )
    orchestrator = PipelineOrchestrator(fixture["run"])
    plan = next(item for item in orchestrator.plan(["ingest"]) if item.stage_id == "ingest")
    provider_receipt = Path(plan.outputs["provider_receipt"])
    provider_receipt.parent.mkdir(parents=True, exist_ok=True)
    provider_receipt.write_text(
        json.dumps(
            {
                "recovery_actions": [
                    {
                        "priority": 1,
                        "stage": "donor_support",
                        "action": (
                            "add_observed_donor_or_switch_to_constrained_generation_for_residual"
                        ),
                        "frame_ids": ["000064"],
                        "allow_deeper_rounds": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    now = datetime.now(UTC)
    write_stage_state(
        fixture["run"],
        StageState(
            stage_id="ingest",
            adapter="holi_ingest",
            status="failed",
            mode="executed",
            attempt=1,
            input_digest="0" * 64,
            config_digest="0" * 64,
            command=["fixture"],
            started_at=now,
            finished_at=now,
            error="fixture failed after writing route recovery actions",
        ),
    )
    (fixture["provider"] / "driver.py").unlink()

    with pytest.raises(StageBlockedError, match="site preflight failed"):
        run_site_pipeline(fixture["run"], targets=["ingest"])

    receipt = json.loads((fixture["run"] / ".video2world" / SITE_RUN_RECEIPT_NAME).read_text())
    assert receipt["status"] == "blocked"
    assert receipt["recovery_actions"][0]["stage_id"] == "ingest"
    assert receipt["recovery_actions"][0]["stage"] == "donor_support"
    assert receipt["recovery_actions"][0]["frame_ids"] == ["000064"]


def test_site_run_blocked_receipt_summarizes_execution_recovery_actions(
    tmp_path: Path,
) -> None:
    fixture = _site_fixture(tmp_path)
    create_site_run(
        fixture["run"],
        video=fixture["video"],
        scene_id="scene",
        run_id="site-execution-recovery-1",
        profile_path=fixture["profile"],
        provider_contract_path=fixture["contract"],
        checkout_roots={"provider": fixture["provider"]},
        artifact_roots={},
    )
    (fixture["provider"] / "driver.py").write_text(
        "from pathlib import Path\n"
        "import json\n"
        "import sys\n"
        "stage = sys.argv[sys.argv.index('--stage') + 1]\n"
        "Path(sys.argv[sys.argv.index('--marker') + 1]).write_text(stage)\n"
        "for index, value in enumerate(sys.argv):\n"
        "    if value != '--output':\n"
        "        continue\n"
        "    role, path = sys.argv[index + 1].split('=', 1)\n"
        "    output = Path(path)\n"
        "    output.parent.mkdir(parents=True, exist_ok=True)\n"
        "    if role == 'cameras':\n"
        "        action = (\n"
        "            'add_observed_donor_or_switch_to_constrained_generation_for_residual'\n"
        "        )\n"
        "        payload = dict(\n"
        "            recovery_actions=[dict(\n"
        "                priority=1,\n"
        "                stage='donor_support',\n"
        "                action=action,\n"
        "                frame_ids=['000064'],\n"
        "                allow_deeper_rounds=False,\n"
        "            )]\n"
        "        )\n"
        "    else:\n"
        "        payload = {'role': role, 'records': [{'id': 'real'}]}\n"
        "    output.write_text(json.dumps(payload), encoding='utf-8')\n"
        "raise SystemExit(1)\n",
        encoding="utf-8",
    )

    with pytest.raises(StageBlockedError, match="exited with 2"):
        run_site_pipeline(fixture["run"], targets=["ingest"])

    receipt = json.loads((fixture["run"] / ".video2world" / SITE_RUN_RECEIPT_NAME).read_text())
    assert receipt["status"] == "blocked"
    assert receipt["recovery_actions"][0]["stage_id"] == "ingest"
    assert receipt["recovery_actions"][0]["stage"] == "donor_support"
    assert receipt["recovery_actions"][0]["frame_ids"] == ["000064"]


def test_site_run_cli_prints_blocked_execution_recovery_actions(
    tmp_path: Path,
    capsys,
) -> None:
    fixture = _site_fixture(tmp_path)
    create_site_run(
        fixture["run"],
        video=fixture["video"],
        scene_id="scene",
        run_id="site-cli-execution-recovery-1",
        profile_path=fixture["profile"],
        provider_contract_path=fixture["contract"],
        checkout_roots={"provider": fixture["provider"]},
        artifact_roots={},
    )
    (fixture["provider"] / "driver.py").write_text(
        "from pathlib import Path\n"
        "import json\n"
        "import sys\n"
        "stage = sys.argv[sys.argv.index('--stage') + 1]\n"
        "Path(sys.argv[sys.argv.index('--marker') + 1]).write_text(stage)\n"
        "for index, value in enumerate(sys.argv):\n"
        "    if value != '--output':\n"
        "        continue\n"
        "    role, path = sys.argv[index + 1].split('=', 1)\n"
        "    output = Path(path)\n"
        "    output.parent.mkdir(parents=True, exist_ok=True)\n"
        "    payload = {'role': role, 'records': [{'id': 'real'}]}\n"
        "    if role == 'cameras':\n"
        "        payload['recovery_actions'] = [dict(\n"
        "            priority=1,\n"
        "            stage='donor_support',\n"
        "            action='add_observed_donor_or_switch_to_constrained_generation',\n"
        "            frame_ids=['000064'],\n"
        "            allow_deeper_rounds=False,\n"
        "        )]\n"
        "    output.write_text(json.dumps(payload), encoding='utf-8')\n"
        "raise SystemExit(1)\n",
        encoding="utf-8",
    )

    exit_code = main(["site-run", str(fixture["run"]), "--stage", "ingest"])

    captured = capsys.readouterr()
    assert exit_code == 3
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["status"] == "blocked"
    assert payload["error_type"] == "StageBlockedError"
    assert payload["recovery_actions"][0]["stage_id"] == "ingest"
    assert payload["recovery_actions"][0]["stage"] == "donor_support"
    assert payload["recovery_actions"][0]["frame_ids"] == ["000064"]


def test_site_preflight_and_run_fail_closed_when_provider_disappears(tmp_path: Path) -> None:
    fixture = _site_fixture(tmp_path)
    create_site_run(
        fixture["run"],
        video=fixture["video"],
        scene_id="scene",
        profile_path=fixture["profile"],
        provider_contract_path=fixture["contract"],
        checkout_roots={"provider": fixture["provider"]},
        artifact_roots={},
    )
    (fixture["provider"] / "driver.py").unlink()

    preflight = preflight_site_run(fixture["run"], targets=["ingest"])
    assert preflight["status"] == "blocked"
    assert preflight["stages"][0]["status"] == "blocked"
    assert "driver" in preflight["stages"][0]["reason"]
    with pytest.raises(StageBlockedError, match="site preflight failed"):
        run_site_pipeline(fixture["run"], targets=["ingest"])
    receipt = json.loads((fixture["run"] / ".video2world" / SITE_RUN_RECEIPT_NAME).read_text())
    assert receipt["status"] == "blocked"
    assert receipt["complete_pipeline"] is False
    assert load_stage_state(fixture["run"], "ingest") is None


def test_site_init_requires_every_named_root_and_rejects_unreferenced_roots(
    tmp_path: Path,
) -> None:
    fixture = _site_fixture(tmp_path)
    with pytest.raises(ConfigurationError, match="missing site roots"):
        create_site_run(
            fixture["run"],
            video=fixture["video"],
            scene_id="scene",
            profile_path=fixture["profile"],
            provider_contract_path=fixture["contract"],
            checkout_roots={},
            artifact_roots={},
        )
    extra = tmp_path / "extra"
    extra.mkdir()
    with pytest.raises(ConfigurationError, match="unreferenced site roots"):
        create_site_run(
            fixture["run"],
            video=fixture["video"],
            scene_id="scene",
            profile_path=fixture["profile"],
            provider_contract_path=fixture["contract"],
            checkout_roots={"provider": fixture["provider"], "extra": extra},
            artifact_roots={},
        )


def test_site_init_rejects_incomplete_adoption_role_mapping(tmp_path: Path) -> None:
    fixture = _site_fixture(tmp_path, adopt_ingest=True)
    payload = yaml.safe_load(fixture["profile"].read_text(encoding="utf-8"))
    del payload["stages"]["ingest"]["outputs"]["cameras"]
    fixture["profile"].write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="adoption output roles differ"):
        create_site_run(
            fixture["run"],
            video=fixture["video"],
            scene_id="scene",
            profile_path=fixture["profile"],
            provider_contract_path=fixture["contract"],
            checkout_roots={"provider": fixture["provider"]},
            artifact_roots={"adopted": fixture["adopted"]},
        )


def test_checked_in_site_examples_cover_exact_canonical_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    profile = load_site_profile(root / "video2world/configs/site_profile.example.yaml")
    bedroom4 = load_site_profile(root / "examples/bedroom4/site-profile.partial.yaml")
    from video2world.providers.external import load_provider_contract

    contract = load_provider_contract(root / "video2world/configs/site_provider.example.yaml")
    assert set(profile.stages) == set(CANONICAL_STAGE_INPUTS)
    assert set(contract.stages) == set(CANONICAL_STAGE_INPUTS)
    assert all(stage.mode == "execute" for stage in profile.stages.values())
    assert set(bedroom4.stages) == set(CANONICAL_STAGE_INPUTS)
    assert bedroom4.stages["layered_completion"].mode == "execute"
    assert bedroom4.stages["pgsr"].mode == "adopt"
    assert set(bedroom4.stages["fusion"].expected_inputs) == set(CANONICAL_STAGE_INPUTS["fusion"])
    assert bedroom4.stages["ingest"].expected_inputs == {}
    for stage_id, inputs in CANONICAL_STAGE_INPUTS.items():
        assert set(contract.stages[stage_id].input_roles) == set(inputs)


def test_site_binding_is_content_addressed(tmp_path: Path) -> None:
    fixture = _site_fixture(tmp_path)
    create_site_run(
        fixture["run"],
        video=fixture["video"],
        scene_id="scene",
        profile_path=fixture["profile"],
        provider_contract_path=fixture["contract"],
        checkout_roots={"provider": fixture["provider"]},
        artifact_roots={},
    )
    binding = load_site_binding(fixture["run"])
    assert len(binding.profile_sha256) == 64
    assert len(binding.provider_contract_sha256) == 64
    fixture["contract"].write_text("changed\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="changed after site initialization"):
        preflight_site_run(fixture["run"], targets=["ingest"])
