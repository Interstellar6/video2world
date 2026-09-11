#!/usr/bin/env python3
"""One command: point at a calibrated dataset and an output directory, get a task.

    python scripts/run_dataset.py --source /data/scene --output-dir /data/runs

Everything else has a default: the profile is the twelve-stage SeetaCloud recipe
with the learned 3D-Fixer registered in place of ``geometry_completion``, and the
background branch is skipped because it is a separate, still-unaccepted branch.
The task id is derived from the dataset directory name unless it is given.

``--output-dir`` may be either the run root (the task lands in
``<output-dir>/outputs/<task_id>``) or the task directory itself when it already
ends in ``outputs/<task_id>``; the resolved task path is printed before anything
runs so the caller never has to guess where the artifacts went.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from world_modeling import modules  # noqa: E402

TASK_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
DEFAULT_PROFILE = REPO / "profiles/three-d-fixer.skipbg.json"


def resolve_paths(output_dir: Path, task_id: str | None, source: Path):
    """Return (root, task_id) for a user-supplied output directory."""
    combined = output_dir.expanduser().resolve()
    if combined.parent.name == "outputs" and TASK_ID.fullmatch(combined.name):
        task_id = task_id or combined.name
        if task_id != combined.name:
            raise SystemExit(f"--task-id {task_id} disagrees with the task directory {combined.name}")
        return combined.parent.parent, task_id
    derived = task_id or re.sub(r"[^A-Za-z0-9_.-]+", "-", source.resolve().name).strip("-. ")
    if not TASK_ID.fullmatch(derived or ""):
        # Two datasets must never share one task directory by accident, so an
        # unusable source name is refused instead of falling back to a default.
        raise SystemExit(f"derived task id {derived!r} is not a usable directory name; pass --task-id")
    return combined, derived


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True,
                        help="calibrated dataset directory containing images/ and sparse/0/")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="run root (task lands in <output-dir>/outputs/<task_id>) or the task directory itself")
    parser.add_argument("--task-id", help="task directory name; defaults to the dataset directory name")
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE, help=f"default: {DEFAULT_PROFILE}")
    parser.add_argument("--through", help="stop after this module")
    parser.add_argument("--keep-background", action="store_true",
                        help="keep clean_plate, background_reconstruction and scene_recomposition instead of skipping them")
    parser.add_argument("--rerun", action="store_true", help="run even when the task directory already exists")
    parser.add_argument("--export-dir", type=Path,
                        help="also write the delivery layout here and verify it independently")
    parser.add_argument("--scene-name", help="scene directory name inside the export; defaults to the task id")
    args = parser.parse_args(argv)

    source = args.source.expanduser().resolve()
    if not source.is_dir() or not (source / "images").is_dir() or not (source / "sparse").is_dir():
        raise SystemExit(f"--source must be a directory containing images/ and sparse/: {source}")
    profile = args.profile.expanduser().resolve()
    if not profile.is_file():
        raise SystemExit(f"profile not found: {profile}")

    root, task_id = resolve_paths(args.output_dir, args.task_id, source)
    task_dir = root / "outputs" / task_id
    profile_payload = __import__("json").loads(profile.read_text())
    if "observed_context_completion" in profile_payload.get("providers", {}):
        from three_d_fixer_cli import activate

        activate(modules.REGISTRY)
        modules.ORDERED_SPECS = modules.REGISTRY.modules

    from world_modeling.cli import main as unified_main

    print(f"source:  {source}", flush=True)
    print(f"task:    {task_dir}", flush=True)
    print(f"profile: {profile}", flush=True)

    common = ["--root", str(root)]
    if not task_dir.exists():
        code = unified_main([*common, "init-task", task_id, "--source", str(source)])
        if code:
            return code
    elif not args.rerun:
        print("task directory already exists; resuming it (use --rerun to force a fresh task id)", flush=True)

    command = [*common, "run", task_id, "--profile", str(profile)]
    if args.through:
        command += ["--through", args.through]
    if not args.keep_background:
        command.append("--skip-bg-recon")
    code = unified_main(command)
    print(f"PIPELINE_EXIT={code}", flush=True)
    if code or not args.export_dir:
        return code

    import export_assets
    import verify_export

    export_root = args.export_dir.expanduser().resolve()
    scene_name = args.scene_name or task_id
    print(f"export:  {export_root} (scene {scene_name})", flush=True)
    exported = export_assets.main([task_id, "--root", str(root), "--export-root", str(export_root),
                                   "--scene-name", scene_name])
    if exported:
        return exported
    verified = verify_export.main(["--export-root", str(export_root)])
    print(f"EXPORT_VERIFY_EXIT={verified}", flush=True)
    return verified


if __name__ == "__main__":
    raise SystemExit(main())
