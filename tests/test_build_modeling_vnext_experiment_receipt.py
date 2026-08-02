from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_modeling_vnext_experiment_receipt import (
    SPEC_KIND,
    STAGE_IDS,
    ExperimentReceiptError,
    build_receipt,
)


def write_spec(tmp_path: Path) -> Path:
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("fresh evidence\n", encoding="utf-8")
    spec = {
        "schema_version": 1,
        "kind": SPEC_KIND,
        "scene_id": "bedroom_4",
        "run_id": "fresh-vnext",
        "stages": [
            {
                "stage_id": stage_id,
                "status": "passed" if index < 3 else "not_tested",
                "artifacts": (
                    [{"role": "primary", "path": "evidence.txt"}] if index == 0 else []
                ),
                "notes": [],
            }
            for index, stage_id in enumerate(STAGE_IDS)
        ],
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path


def test_build_receipt_hashes_artifacts_and_never_promotes(tmp_path: Path) -> None:
    spec = write_spec(tmp_path)
    output = tmp_path / "receipt.json"
    receipt = build_receipt(spec, output)
    assert receipt["summary"] == {"not_tested": 13, "passed": 3}
    assert receipt["promotion_allowed"] is False
    artifact = receipt["stages"][0]["artifacts"][0]
    assert artifact["kind"] == "file"
    assert len(artifact["sha256"]) == 64
    assert json.loads(output.read_text()) == receipt


def test_build_receipt_rejects_missing_or_repeated_stage(tmp_path: Path) -> None:
    spec_path = write_spec(tmp_path)
    value = json.loads(spec_path.read_text())
    value["stages"][-1] = value["stages"][0]
    spec_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ExperimentReceiptError, match="repeat"):
        build_receipt(spec_path, tmp_path / "receipt.json")
