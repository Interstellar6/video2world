#!/usr/bin/env python3
"""Build a hash-bound receipt for a cross-environment modeling_vnext experiment."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from video2world.hashing import atomic_write_json, digest_path

KIND = "video2world.modeling_vnext_experiment_receipt"
SPEC_KIND = "video2world.modeling_vnext_experiment_spec"
STAGE_IDS = (
    "ingest",
    "da3",
    "pgsr",
    "object_proposals",
    "qwen_contracts",
    "component_segmentation",
    "semantic_lifting",
    "clean_plate",
    "component_assembly",
    "image_completion",
    "object_completion",
    "mesh_postprocess",
    "physics_estimation",
    "placement",
    "bundle",
    "web",
)
STATUSES = {
    "passed",
    "proxy_passed",
    "partial",
    "failed",
    "blocked",
    "not_tested",
    "reused",
}


class ExperimentReceiptError(RuntimeError):
    """Raised when an experiment spec weakens stage or artifact provenance."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ExperimentReceiptError(message)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExperimentReceiptError(f"cannot read experiment spec: {path}") from exc
    require(isinstance(value, dict), "experiment spec root must be an object")
    return value


def artifact_record(raw: Any, *, spec_dir: Path, stage_id: str) -> dict[str, Any]:
    require(isinstance(raw, dict), f"{stage_id} artifact must be an object")
    role = raw.get("role")
    raw_path = raw.get("path")
    require(isinstance(role, str) and role, f"{stage_id} artifact role is missing")
    require(isinstance(raw_path, str) and raw_path, f"{stage_id} artifact path is missing")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = spec_dir / path
    path = path.resolve()
    require(path.exists(), f"{stage_id} artifact does not exist: {path}")
    digest = digest_path(path)
    record = {
        "role": role,
        "path": str(digest.path),
        "kind": digest.kind,
        "sha256": digest.sha256,
        "size_bytes": digest.size_bytes,
    }
    note = raw.get("note")
    if note is not None:
        require(isinstance(note, str) and note, f"{stage_id} artifact note is invalid")
        record["note"] = note
    return record


def build_receipt(spec_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    source = Path(spec_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    spec = read_json(source)
    require(spec.get("schema_version") == 1, "experiment spec schema_version must be 1")
    require(spec.get("kind") == SPEC_KIND, f"experiment spec kind must be {SPEC_KIND}")
    scene_id = spec.get("scene_id")
    run_id = spec.get("run_id")
    require(isinstance(scene_id, str) and scene_id, "scene_id is missing")
    require(isinstance(run_id, str) and run_id, "run_id is missing")
    stages = spec.get("stages")
    require(isinstance(stages, list), "stages must be a list")
    stage_ids = [item.get("stage_id") for item in stages if isinstance(item, dict)]
    require(len(stage_ids) == len(stages), "every stage must be an object")
    counts = Counter(stage_ids)
    repeated = sorted(stage for stage, count in counts.items() if count > 1)
    require(not repeated, f"stage ids repeat: {repeated}")
    require(set(stage_ids) == set(STAGE_IDS), "spec must declare every canonical vNext stage")

    stage_records = []
    for stage_id in STAGE_IDS:
        raw = next(item for item in stages if item["stage_id"] == stage_id)
        status = raw.get("status")
        require(status in STATUSES, f"{stage_id} has invalid status: {status}")
        artifacts = raw.get("artifacts", [])
        require(isinstance(artifacts, list), f"{stage_id} artifacts must be a list")
        records = [
            artifact_record(item, spec_dir=source.parent, stage_id=stage_id)
            for item in artifacts
        ]
        roles = [item["role"] for item in records]
        require(len(roles) == len(set(roles)), f"{stage_id} artifact roles repeat")
        notes = raw.get("notes", [])
        require(
            isinstance(notes, list) and all(isinstance(item, str) for item in notes),
            f"{stage_id} notes must be strings",
        )
        stage_records.append(
            {"stage_id": stage_id, "status": status, "artifacts": records, "notes": notes}
        )

    summary = dict(sorted(Counter(item["status"] for item in stage_records).items()))
    receipt = {
        "schema_version": 1,
        "kind": KIND,
        "created_at": datetime.now(UTC).isoformat(),
        "scene_id": scene_id,
        "run_id": run_id,
        "pipeline_id": "modeling_vnext",
        "status": "experiment_recorded_pending_or_failed_gates",
        "promotion_allowed": False,
        "source_spec": {
            "path": str(source),
            "sha256": digest_path(source).sha256,
            "size_bytes": source.stat().st_size,
        },
        "summary": summary,
        "stages": stage_records,
        "claim_boundary": (
            "This receipt records observed execution evidence across environments. "
            "It does not convert proxy, reused, partial, failed, blocked, or not-tested "
            "stages into a canonical world promotion."
        ),
    }
    atomic_write_json(output, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = build_receipt(args.spec, args.output)
    except ExperimentReceiptError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}))
        return 2
    print(json.dumps({"output": str(args.output.resolve()), "summary": receipt["summary"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
