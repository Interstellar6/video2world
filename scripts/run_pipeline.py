#!/usr/bin/env python3
"""One-command task creation and execution of explicitly bound model providers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from world_modeling.engine import (PipelineError, Task, create_task, load_profile, run_pipeline,
                                   snapshot_path, validate_source, without_bg_recon)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_id")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=REPO / "profiles" / "seetacloud.json")
    parser.add_argument("--through")
    parser.add_argument("--skip-bg-recon", action="store_true",
                        help="Skip clean_plate, background_reconstruction and scene_recomposition")
    args = parser.parse_args()
    try:
        task = Task(REPO, args.task_id)
        if not task.directory.exists():
            task = create_task(REPO, args.task_id, args.source)
        else:
            validate_source(args.source)
            source = task.load_artifacts()["artifacts"].get("source_media", {})
            if snapshot_path(args.source)[0] != source.get("sha256"):
                raise PipelineError("--source differs from this task's recorded input; use a new task_id")
        profile = load_profile(args.profile)
        if args.skip_bg_recon:
            profile["pipeline"] = without_bg_recon(profile["pipeline"])
        state = run_pipeline(task, profile, args.through)
        print(f"task_dir={task.directory}")
        print(f"status={state['status']}")
        return 0
    except PipelineError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
