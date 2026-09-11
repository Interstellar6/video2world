#!/usr/bin/env python3
"""Write a hash-bound receipt for a real but intentionally partial object run."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def record(task_dir: Path, relative_path: str, evidence: str, role: str) -> dict:
    path = (task_dir / relative_path).resolve()
    if task_dir.resolve() not in path.parents or not path.is_file():
        raise ValueError(f"invalid task-local artifact: {relative_path}")
    return {
        "role": role,
        "path": path.relative_to(task_dir.resolve()).as_posix(),
        "sha256": digest(path),
        "size_bytes": path.stat().st_size,
        "evidence": evidence,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    args = parser.parse_args()
    task_dir = args.task_dir.resolve()
    artifacts = [
        record(task_dir, "stages/scene_understanding/qwen_vl/qwen_description.json", "derived", "object_descriptions"),
        record(task_dir, "stages/geometry_completion/stream3d/outputs/bed/chunk_0000/result.glb", "derived", "completed_object_meshes"),
        record(task_dir, "stages/mesh_postprocess/coacd/coacd_collision.obj", "derived", "coacd_collision_meshes"),
        record(task_dir, "stages/physics_estimation/physical_object.obj", "derived", "physical_object_obj"),
        record(task_dir, "stages/physics_estimation/physics_properties.json", "derived", "physics_properties"),
        record(task_dir, "stages/physics_estimation/physics_report.json", "derived", "physics_report"),
    ]
    payload = {
        "kind": "video2world_modeling.partial_run_receipt",
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "scope": "real object completion and physical-proxy chain from recovered 8-view input",
        "promotion_allowed": False,
        "executed_stages": [
            "scene_understanding:qwen2.5-vl-3b",
            "geometry_completion:stream3d",
            "mesh_postprocess:coacd",
            "physics_estimation:qwen-prior-plus-obj-export",
        ],
        "artifacts": artifacts,
        "blockers": [
            "The original 795-frame source sequence referenced by historical receipts is absent on the server.",
            "SAM3-I stage-3 checkpoint was not available for a fresh component-instance run.",
            "No fresh clean-plate, background reconstruction, component assembly, FixAnything orbit, or room-frame recomposition was executed.",
            "Stream3D object-local output has no validated SceneTransform back into the original room.",
            "Qwen scale, mass, and smoothness values are single-view priors, not measured physical properties.",
            "CoACD hit the 16-hull cap and reported maximum concavity 0.11246, above its requested 0.05 threshold.",
        ],
    }
    (task_dir / "partial_run_receipt.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
