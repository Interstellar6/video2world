from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from video2world.adapters import get_adapter
from video2world.config import load_run_config
from video2world.errors import ConfigurationError
from video2world.orchestrator import PipelineOrchestrator, initialize_run
from video2world.providers.external import (
    execute_stage,
    load_provider_contract,
    preflight_contract,
)


def write_contract(path: Path, *, driver: Path, requirement: Path | None = None) -> None:
    requirements = {
        "python": {"path": sys.executable, "kind": "executable"},
        "driver": {"path": str(driver), "kind": "file"},
    }
    if requirement is not None:
        requirements["checkpoint"] = {"path": str(requirement), "kind": "file"}
    payload = {
        "schema_version": "1.0",
        "provider_id": "test-provider",
        "scope": "generic_interface",
        "verification_status": "requires_site_configuration",
        "stages": {
            "ingest": {
                "provider": "test",
                "command": [
                    sys.executable,
                    str(driver),
                    "{input_video}",
                    "{output_result}",
                ],
                "cwd": "{run_dir}",
                "input_roles": ["video"],
                "output_roles": ["result"],
                "requires": requirements,
            }
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_provider_executes_real_argv_and_writes_redacted_receipt(tmp_path: Path) -> None:
    driver = tmp_path / "driver.py"
    driver.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "Path(sys.argv[2]).write_bytes(Path(sys.argv[1]).read_bytes() + b'-real')\n",
        encoding="utf-8",
    )
    contract = tmp_path / "provider.yaml"
    write_contract(contract, driver=driver)
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    output = tmp_path / "run" / "result.bin"
    output.parent.mkdir()
    receipt_path = tmp_path / "run" / "receipt.json"

    receipt = execute_stage(
        contract,
        "ingest",
        values={"run_dir": str(output.parent)},
        inputs={"video": str(source)},
        outputs={"result": str(output)},
        receipt_path=receipt_path,
    )

    assert output.read_bytes() == b"video-real"
    assert receipt["status"] == "completed"
    assert receipt["outputs"]["result"]["size_bytes"] == len(b"video-real")
    serialized = receipt_path.read_text(encoding="utf-8")
    assert "command_sha256" in serialized
    assert "-real" not in serialized


def test_provider_preflight_fails_before_launch_for_missing_checkpoint(tmp_path: Path) -> None:
    marker = tmp_path / "launched"
    driver = tmp_path / "driver.py"
    driver.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n",
        encoding="utf-8",
    )
    contract = tmp_path / "provider.yaml"
    write_contract(contract, driver=driver, requirement=tmp_path / "missing.ckpt")

    with pytest.raises(ConfigurationError, match=r"checkpoint.*not a file"):
        preflight_contract(contract, values={"run_dir": str(tmp_path)})
    assert not marker.exists()


def test_provider_rejects_artifact_role_drift(tmp_path: Path) -> None:
    driver = tmp_path / "driver.py"
    driver.write_text("raise SystemExit('must not execute')\n", encoding="utf-8")
    contract = tmp_path / "provider.yaml"
    write_contract(contract, driver=driver)
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")

    with pytest.raises(ConfigurationError, match="output roles differ"):
        execute_stage(
            contract,
            "ingest",
            values={"run_dir": str(tmp_path)},
            inputs={"video": str(source)},
            outputs={"wrong": str(tmp_path / "wrong")},
            receipt_path=tmp_path / "receipt.json",
        )


def test_generic_profile_is_configured_argv_not_verified_bedroom4_claim() -> None:
    root = Path(__file__).resolve().parents[1]
    pipeline = load_run_config(root / "video2world/configs/holi_embodiedgen_upstream.yaml")
    assert pipeline.topological_order() == [
        "provider_preflight",
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
    assert all(stage.command for stage in pipeline.stages.values())
    contract = load_provider_contract(
        root / "video2world/configs/holi_embodiedgen.provider.example.yaml"
    )
    assert contract.scope == "generic_interface"
    assert contract.verification_status == "requires_site_configuration"
    expected_stages = {
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
    }
    assert set(contract.stages) == expected_stages
    for stage_id in expected_stages:
        stage = pipeline.stages[stage_id]
        assert set(contract.stages[stage_id].input_roles) == set(stage.inputs)
        assert set(contract.stages[stage_id].output_roles) == set(
            get_adapter(stage.adapter).required_outputs
        )
    serialized = json.dumps(contract.model_dump(mode="json"), ensure_ascii=False)
    assert "scene_outputs_verified" not in serialized


def test_upstream_profile_runs_provider_preflight_node(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    driver = tmp_path / "driver.py"
    driver.write_text(
        "raise SystemExit('preflight must not launch the driver')\n",
        encoding="utf-8",
    )
    stages = {}
    for stage_id in (
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
    ):
        stages[stage_id] = {
            "provider": "integration-test",
            "command": [sys.executable, str(driver), "{input_video}", "{output_result}"],
            "cwd": "{run_dir}",
            "output_roles": ["result"],
            "requires": {
                "python": {"path": sys.executable, "kind": "executable"},
                "driver": {"path": str(driver), "kind": "file"},
            },
        }
    contract = tmp_path / "provider.yaml"
    contract.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "provider_id": "integration-provider",
                "scope": "generic_interface",
                "verification_status": "requires_site_configuration",
                "stages": stages,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    profile_payload = yaml.safe_load(
        (root / "video2world/configs/holi_embodiedgen_upstream.yaml").read_text(
            encoding="utf-8"
        )
    )
    profile_payload["variables"] = {
        "provider_python": sys.executable,
        "provider_contract": str(contract),
    }
    profile = tmp_path / "profile.yaml"
    profile.write_text(yaml.safe_dump(profile_payload, sort_keys=False), encoding="utf-8")
    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")
    run_dir = tmp_path / "run"
    initialize_run(run_dir, video=video, scene_id="scene", template_config=profile)

    result = PipelineOrchestrator(run_dir).run(["provider_preflight"])

    assert result[0]["action"] == "executed"
    report = json.loads((run_dir / "artifacts/provider/preflight.json").read_text())
    assert report["status"] == "passed"
    assert len(report["stages"]) == 10
    assert not (tmp_path / "launched").exists()
