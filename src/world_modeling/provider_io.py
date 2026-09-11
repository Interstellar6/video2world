"""Small task-local file interface shared by isolated model environments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parser(description: str) -> argparse.ArgumentParser:
    app = argparse.ArgumentParser(description=description)
    app.add_argument("--task-dir", type=Path, required=True)
    app.add_argument("--inputs", type=Path, required=True)
    app.add_argument("--outputs", type=Path, required=True)
    return app


def local_path(task: Path, value: str | Path, *, exists: bool = True) -> Path:
    path = (task / value).resolve()
    if task.resolve() not in path.parents:
        raise ValueError(f"path escapes task: {value}")
    if exists and not path.exists():
        raise ValueError(f"missing task artifact: {value}")
    return path


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def input_artifact(args: argparse.Namespace, role: str) -> Path:
    inputs = read_json(local_path(args.task_dir, args.inputs))["inputs"]
    record = inputs[role]
    if record.get("evidence") == "contract_only":
        raise ValueError(f"{role}: real inference cannot consume contract-only input")
    return local_path(args.task_dir, record["path"])


def publish(args: argparse.Namespace, outputs: dict[str, Path]) -> None:
    records = {
        role: {"path": local_path(args.task_dir, path).relative_to(args.task_dir.resolve()).as_posix()}
        for role, path in outputs.items()
    }
    write_json(local_path(args.task_dir, args.outputs, exists=False), {"outputs": records})
