"""Command-line interface for Video2World orchestration."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from video2world.errors import Video2WorldError
from video2world.hashing import atomic_write_json
from video2world.models import world_manifest_json_schema
from video2world.orchestrator import (
    PipelineOrchestrator,
    initialize_run,
    parse_output_assignments,
)
from video2world.query import query_world
from video2world.validation import load_world_manifest, validate_world_manifest


def _print_json(value: Any, *, stream=None) -> None:  # type: ignore[no-untyped-def]
    print(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        file=stream or sys.stdout,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video2world",
        description="Content-addressed orchestration for layered, queryable 3D worlds.",
    )
    parser.add_argument("--version", action="version", version="video2world 0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize an auditable run directory")
    init_parser.add_argument("run_dir")
    init_parser.add_argument("--video", required=True)
    init_parser.add_argument("--scene-id", required=True)
    init_parser.add_argument("--run-id")
    init_parser.add_argument("--config", help="Optional complete pipeline template YAML")

    plan_parser = subparsers.add_parser("plan", help="Inspect the DAG without executing it")
    plan_parser.add_argument("run_dir")
    plan_parser.add_argument("--stage", action="append", dest="stages")

    run_parser = subparsers.add_parser("run", help="Execute configured argv commands")
    run_parser.add_argument("run_dir")
    run_parser.add_argument("--stage", action="append", dest="stages")
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan only; never writes stage state or output artifacts",
    )

    def add_adopt_arguments(adopt_parser: argparse.ArgumentParser) -> None:
        adopt_parser.add_argument("run_dir")
        adopt_parser.add_argument("stage")
        adopt_parser.add_argument(
            "--output",
            action="append",
            required=True,
            help="Existing artifact assignment: role=/absolute/path (repeat for every role)",
        )
        adopt_parser.add_argument("--source-run-id", required=True)
        adopt_parser.add_argument("--source-root")
        adopt_parser.add_argument("--source-repository")
        adopt_parser.add_argument("--source-commit")
        adopt_parser.add_argument("--source-command")
        adopt_parser.add_argument("--note")

    add_adopt_arguments(
        subparsers.add_parser("adopt", help="Register real existing outputs without executing")
    )
    add_adopt_arguments(
        subparsers.add_parser(
            "adopt-existing",
            help="Explicit alias for adopt; distinct from run --dry-run",
        )
    )

    validate_parser = subparsers.add_parser("validate", help="Validate a run or world manifest")
    validate_parser.add_argument("target", help="Run directory or world manifest JSON")
    validate_parser.add_argument("--no-verify-assets", action="store_true")
    validate_parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="Allow remote URIs to remain unverified (reported separately)",
    )

    query_parser = subparsers.add_parser("query", help="Run offline scene cognition QA")
    query_parser.add_argument("manifest")
    query_parser.add_argument("query")
    query_parser.add_argument("--language", choices=("auto", "zh", "en"), default="auto")

    schema_parser = subparsers.add_parser(
        "schema", help="Print or refresh the manifest JSON Schema"
    )
    schema_parser.add_argument("--output")
    return parser


def _cmd_init(args: argparse.Namespace) -> int:
    path = initialize_run(
        args.run_dir,
        video=args.video,
        scene_id=args.scene_id,
        run_id=args.run_id,
        template_config=args.config,
    )
    config = PipelineOrchestrator(args.run_dir).config
    _print_json(
        {
            "status": "initialized",
            "run_dir": str(Path(args.run_dir).expanduser().resolve()),
            "config": str(path),
            "run_id": config.run_id,
            "scene_id": config.scene_id,
            "external_outputs_created": False,
        }
    )
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    orchestrator = PipelineOrchestrator(args.run_dir)
    plans = orchestrator.plan(args.stages)
    _print_json(
        {
            "mode": "plan",
            "mutated": False,
            "run_id": orchestrator.config.run_id,
            "stages": [plan.model_dump(mode="json") for plan in plans],
        }
    )
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    orchestrator = PipelineOrchestrator(args.run_dir)
    if args.dry_run:
        plans = orchestrator.plan(args.stages)
        _print_json(
            {
                "mode": "dry_run",
                "mutated": False,
                "run_id": orchestrator.config.run_id,
                "stages": [plan.model_dump(mode="json") for plan in plans],
            }
        )
        return 0
    summaries = orchestrator.run(args.stages)
    _print_json(
        {
            "mode": "execute",
            "mutated": True,
            "run_id": orchestrator.config.run_id,
            "stages": summaries,
        }
    )
    return 0


def _cmd_adopt(args: argparse.Namespace) -> int:
    orchestrator = PipelineOrchestrator(args.run_dir)
    outputs = parse_output_assignments(args.output)
    state = orchestrator.adopt(
        args.stage,
        outputs,
        source_run_id=args.source_run_id,
        source_root=args.source_root,
        source_repository=args.source_repository,
        source_commit=args.source_commit,
        source_command=args.source_command,
        note=args.note,
    )
    _print_json(
        {
            "mode": "adopt_existing",
            "executed_external_command": False,
            "stage": state.model_dump(mode="json"),
        }
    )
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    target = Path(args.target).expanduser().resolve()
    if target.is_dir():
        result = PipelineOrchestrator(target).validate_run()
    else:
        result = validate_world_manifest(
            target,
            verify_assets=not args.no_verify_assets,
            allow_remote=args.allow_remote,
        )
    _print_json(result)
    return 0 if result["valid"] else 1


def _cmd_query(args: argparse.Namespace) -> int:
    manifest = load_world_manifest(args.manifest)
    result = query_world(manifest, args.query, language=args.language)
    _print_json(result.model_dump(mode="json"))
    return 0 if result.status == "resolved" else 3


def _cmd_schema(args: argparse.Namespace) -> int:
    schema = world_manifest_json_schema()
    if args.output:
        output = Path(args.output).expanduser().resolve()
        atomic_write_json(output, schema)
        _print_json({"status": "written", "output": str(output)})
    else:
        _print_json(schema)
    return 0


COMMANDS = {
    "init": _cmd_init,
    "plan": _cmd_plan,
    "run": _cmd_run,
    "adopt": _cmd_adopt,
    "adopt-existing": _cmd_adopt,
    "validate": _cmd_validate,
    "query": _cmd_query,
    "schema": _cmd_schema,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except (Video2WorldError, ValidationError, OSError, ValueError) as exc:
        _print_json(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
            stream=sys.stderr,
        )
        return 2
