#!/usr/bin/env python3
"""Rebuild a task's stage records and artifact index from its stage manifests.

A stage manifest is written only after that stage's outputs exist and are
hashed, so the manifests on disk are the durable record of what a task has
actually produced. `task.json` and `artifacts.json` are derived views of them:
the index maps an output role to the record of the file that fulfils it, and a
stage record points at its manifest plus the request fingerprint that produced
it.

A run that is interrupted after it invalidated its predecessors leaves those
derived views behind -- the stage record becomes "running" with no manifest,
and the invalidated output roles are dropped from the index -- while the files
themselves are untouched. Re-running rebuilds them the expensive way. This
rebuilds the views from the manifests instead, and refuses to invent anything:
a manifest whose outputs are missing, changed, or unreadable is reported and
skipped, so the task keeps whatever record it already had for that stage.

    python3 scripts/repair_task_state.py my-task [--root .] [--dry-run]

Every artifact it restores is re-verified against its own hash before it is
written, and the command reports what it restored, role by role.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from world_modeling import modules  # noqa: E402
from world_modeling.contracts import task_relative  # noqa: E402
from world_modeling.engine import (  # noqa: E402
    PipelineError, Task, read_json, verify_artifact, write_json,
)

RESUMABLE = "executed"


def manifest_for(task: Task, module_id: str) -> Path:
    return task.directory / "stages" / module_id / "stage.json"


def adoptable(task: Task, module_id: str) -> tuple[dict | None, str]:
    """Return the stage record a manifest proves, or the reason it does not."""
    manifest_path = manifest_for(task, module_id)
    if not manifest_path.is_file():
        return None, "no manifest"
    try:
        manifest = read_json(manifest_path)
    except (OSError, ValueError) as error:
        return None, f"unreadable manifest: {error}"
    if manifest.get("status") not in (RESUMABLE, "contract_only"):
        return None, f"manifest status is {manifest.get('status')!r}"
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        return None, "manifest records no outputs"
    for name, record in outputs.items():
        try:
            verify_artifact(task, record)
        except (PipelineError, OSError, KeyError, TypeError) as error:
            return None, f"output {name} does not verify: {error}"
    if not isinstance(manifest.get("request_sha256"), str):
        return None, "manifest records no request fingerprint"
    return {"status": manifest["status"], "manifest": task_relative(task.directory, manifest_path),
            "request_sha256": manifest["request_sha256"]}, ""


def run(task: Task, dry_run: bool) -> dict:
    state = task.load_state()
    document = read_json(task.artifacts_path)
    artifacts = document.setdefault("artifacts", {})
    pipeline = list(state.get("stages", {}))
    restored, adopted_roles, skipped = [], [], []
    for module_id in pipeline:
        record, reason = adoptable(task, module_id)
        if record is None:
            skipped.append({"module": module_id, "reason": reason})
            continue
        manifest = read_json(manifest_for(task, module_id))
        changed = {name: item for name, item in manifest["outputs"].items() if artifacts.get(name) != item}
        state["stages"][module_id] = record
        artifacts.update(manifest["outputs"])
        restored.append(module_id)
        adopted_roles.extend(sorted(changed))
    if not dry_run and restored:
        write_json(task.state_path, state)
        write_json(task.artifacts_path, document)
    return {"task_id": task.task_id, "dry_run": dry_run, "restored_stages": restored,
            "restored_roles": sorted(set(adopted_roles)), "skipped": skipped}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task_id")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--dry-run", action="store_true", help="report what would be restored without writing")
    args = parser.parse_args(argv)
    task = Task(args.root.resolve(), args.task_id)
    if not task.directory.is_dir():
        print(f"unknown task: {task.directory}", file=sys.stderr)
        return 2
    known = {spec.id for spec in modules.REGISTRY.modules}
    state = task.load_state()
    unknown = sorted(set(state.get("stages", {})) - known)
    if unknown:
        print(f"task records stages this registry does not define: {', '.join(unknown)}", file=sys.stderr)
        return 2
    try:
        report = run(task, args.dry_run)
    except (PipelineError, OSError, ValueError, KeyError, TypeError) as error:
        print(f"repair_task_state failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
