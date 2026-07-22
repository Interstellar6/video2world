"""Command-line interface for Video2World orchestration."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from video2world.completion import (
    build_layered_completion_plan,
    load_layered_completion_plan,
    load_occlusion_graph,
    load_scene_inventory,
    write_layered_completion_plan,
)
from video2world.completion_routing import (
    CompletionEvidence,
    clean_plate_next_action_from_report,
    route_completion_backend,
)
from video2world.errors import StageBlockedError, Video2WorldError
from video2world.hashing import atomic_write_json, digest_json, digest_path
from video2world.models import world_manifest_json_schema
from video2world.orchestrator import (
    PipelineOrchestrator,
    initialize_run,
    parse_output_assignments,
)
from video2world.providers.qwen_inventory import (
    parse_frame_ids as parse_inventory_frame_ids,
)
from video2world.providers.qwen_inventory import run_inventory_provider
from video2world.query import query_world
from video2world.scene_command_queue import submit_scene_command
from video2world.scene_commands import (
    load_scene_command,
    plan_scene_command,
    write_scene_command_plan,
)
from video2world.validation import load_world_manifest, validate_world_manifest
from video2world.web_adoption import (
    adopt_web_manifest_file,
    bind_scene_command_web_manifest,
)


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

    site_init_parser = subparsers.add_parser(
        "site-init",
        help="Bind every canonical stage to an explicit provider or artifact root",
    )
    site_init_parser.add_argument("run_dir")
    site_init_parser.add_argument("--video", required=True)
    site_init_parser.add_argument("--scene-id", required=True)
    site_init_parser.add_argument("--run-id")
    site_init_parser.add_argument("--profile", required=True)
    site_init_parser.add_argument("--provider-contract", required=True)
    site_init_parser.add_argument(
        "--checkout",
        action="append",
        default=[],
        help="Named provider checkout root: name=/absolute/path",
    )
    site_init_parser.add_argument(
        "--artifact-root",
        action="append",
        default=[],
        help="Named checkpoint or adopted-output root: name=/absolute/path",
    )
    site_init_parser.add_argument("--driver-python")

    site_preflight_parser = subparsers.add_parser(
        "site-preflight",
        help="Validate all selected provider and adoption bindings without running models",
    )
    site_preflight_parser.add_argument("run_dir")
    site_preflight_parser.add_argument("--stage", action="append", dest="stages")

    site_run_parser = subparsers.add_parser(
        "site-run",
        help="Execute or adopt the bound canonical stages with durable receipts",
    )
    site_run_parser.add_argument("run_dir")
    site_run_parser.add_argument("--stage", action="append", dest="stages")

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

    inventory_parser = subparsers.add_parser(
        "qwen-inventory",
        help=(
            "Discover category observations and visible-overlap hypotheses with local "
            "Qwen2.5-VL; geometry ordering still requires SAM3 and depth"
        ),
    )
    inventory_parser.add_argument("--frames-dir", required=True)
    inventory_parser.add_argument("--frame-id", action="append", default=[])
    inventory_parser.add_argument(
        "--frame-ids", help="Comma-separated alternative to repeated --frame-id"
    )
    inventory_parser.add_argument("--scene-id", required=True)
    inventory_parser.add_argument("--run-id", required=True)
    inventory_parser.add_argument("--model-dir", required=True)
    inventory_parser.add_argument("--model-revision")
    inventory_parser.add_argument("--output-dir", required=True)
    inventory_parser.add_argument("--columns", type=int, default=4)
    inventory_parser.add_argument("--max-new-tokens", type=int, default=8192)
    inventory_parser.add_argument("--device", default="cuda:0")
    inventory_parser.add_argument("--minimum-occlusion-confidence", type=float, default=0.6)

    scene_audit_parser = subparsers.add_parser(
        "qwen-scene-audit",
        help="Audit visible scene categories, then compare deterministic asset coverage",
    )
    scene_audit_parser.add_argument("--contact-sheet")
    scene_audit_parser.add_argument("--frames-dir")
    scene_audit_parser.add_argument("--frame-id", action="append", default=[])
    scene_audit_parser.add_argument("--frame-ids")
    scene_audit_parser.add_argument("--scene-id", required=True)
    scene_audit_parser.add_argument("--run-id", required=True)
    scene_audit_parser.add_argument("--asset-summary", required=True)
    scene_audit_parser.add_argument("--model-dir", required=True)
    scene_audit_parser.add_argument("--model-revision")
    scene_audit_parser.add_argument("--model-repo")
    scene_audit_parser.add_argument("--output-dir", required=True)
    scene_audit_parser.add_argument("--columns", type=int, default=4)
    scene_audit_parser.add_argument("--max-new-tokens", type=int, default=3072)
    scene_audit_parser.add_argument("--device", default="cuda:0")

    scene_audit_merge_parser = subparsers.add_parser(
        "qwen-scene-audit-merge",
        help="Deterministically merge validated per-view Qwen scene audits",
    )
    scene_audit_merge_parser.add_argument(
        "--audit", action="append", required=True, help="Per-view scene_audit.json"
    )
    scene_audit_merge_parser.add_argument("--asset-summary", required=True)
    scene_audit_merge_parser.add_argument("--run-id", required=True)
    scene_audit_merge_parser.add_argument("--output-dir", required=True)

    completion_parser = subparsers.add_parser(
        "completion-plan",
        help="Build a fail-closed front-to-back object and clean-plate plan",
    )
    completion_parser.add_argument("--inventory", required=True)
    completion_parser.add_argument("--occlusion-graph", required=True)
    completion_parser.add_argument("--output", required=True)
    completion_parser.add_argument("--max-rounds", type=int, default=32)
    completion_parser.add_argument("--max-parallel-targets", type=int, default=1)

    completion_validate_parser = subparsers.add_parser(
        "completion-validate",
        help="Validate a layered completion plan and its round ordering",
    )
    completion_validate_parser.add_argument("plan")

    completion_route_parser = subparsers.add_parser(
        "completion-route",
        help="Choose a completion backend from verified multi-view evidence",
    )
    completion_route_parser.add_argument("evidence")
    completion_route_parser.add_argument(
        "--clean-plate-report",
        help="Optional failed clean-plate QA/materializer report; its next_action is routed.",
    )
    completion_route_parser.add_argument("--output")

    completion_recovery_parser = subparsers.add_parser(
        "completion-recovery-work-order",
        help="Materialize a blocked completion route into auditable recovery work items",
    )
    completion_recovery_parser.add_argument("route")
    completion_recovery_parser.add_argument("--clean-plate-report", required=True)
    completion_recovery_parser.add_argument("--output", required=True)

    completion_recovery_bundle_parser = subparsers.add_parser(
        "completion-recovery-bundle",
        help="Convert a recovery work order into ordered execution-plan steps",
    )
    completion_recovery_bundle_parser.add_argument("work_order")
    completion_recovery_bundle_parser.add_argument("--output", required=True)

    scene_command_validate_parser = subparsers.add_parser(
        "scene-command-validate",
        help="Validate structured natural-language scene intent JSON",
    )
    scene_command_validate_parser.add_argument("command_file")

    scene_command_plan_parser = subparsers.add_parser(
        "scene-command-plan",
        help="Resolve targets and build a non-mutating scene command preview",
    )
    scene_command_plan_parser.add_argument("--manifest", required=True)
    scene_command_plan_parser.add_argument("--command", dest="command_file", required=True)
    scene_command_plan_parser.add_argument("--output")

    scene_command_submit_parser = subparsers.add_parser(
        "scene-command-submit",
        help="Atomically enqueue a confirmed scene mutation or execute a read-only query",
    )
    scene_command_submit_parser.add_argument("--manifest", required=True)
    scene_command_submit_parser.add_argument("--command", dest="command_file", required=True)
    scene_command_submit_parser.add_argument("--queue-dir", required=True)
    scene_command_submit_parser.add_argument("--confirm")

    scene_command_serve_parser = subparsers.add_parser(
        "scene-command-serve",
        help="Serve manifest-bound scene-command planning and immutable queue submission",
    )
    scene_command_serve_parser.add_argument("--manifest", required=True)
    scene_command_serve_parser.add_argument(
        "--web-manifest",
        help=(
            "Optional deployed Web manifest whose sourceWorld world/run/hash must bind "
            "to --manifest"
        ),
    )
    scene_command_serve_parser.add_argument("--queue-dir", required=True)
    scene_command_serve_parser.add_argument("--host", default="127.0.0.1")
    scene_command_serve_parser.add_argument("--port", type=int, default=8765)
    scene_command_serve_parser.add_argument(
        "--cors-origin",
        action="append",
        default=[],
        help="Exact allowed Web origin; repeat for each origin (wildcards are forbidden)",
    )
    scene_command_serve_parser.add_argument(
        "--max-json-bytes",
        type=int,
        default=256 * 1024,
    )
    scene_command_serve_parser.add_argument(
        "--allow-non-loopback",
        action="store_true",
        help="Explicitly acknowledge binding beyond localhost (no authentication is added)",
    )

    web_adopt_parser = subparsers.add_parser(
        "web-manifest-adopt",
        help="Create a canonical planning sidecar from a deployed Web manifest",
    )
    web_adopt_parser.add_argument("web_manifest")
    web_adopt_parser.add_argument("--output", required=True)

    web_bind_parser = subparsers.add_parser(
        "web-manifest-bind",
        help="Write a derived Web manifest bound to a canonical command sidecar",
    )
    web_bind_parser.add_argument("--web-manifest", required=True)
    web_bind_parser.add_argument("--canonical-manifest", required=True)
    web_bind_parser.add_argument("--endpoint", required=True)
    web_bind_parser.add_argument("--derived-version", required=True)
    web_bind_parser.add_argument("--output", required=True)

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


def _cmd_site_init(args: argparse.Namespace) -> int:
    from video2world.site_pipeline import create_site_run, parse_root_assignments

    path = create_site_run(
        args.run_dir,
        video=args.video,
        scene_id=args.scene_id,
        run_id=args.run_id,
        profile_path=args.profile,
        provider_contract_path=args.provider_contract,
        checkout_roots=parse_root_assignments(args.checkout, label="checkout"),
        artifact_roots=parse_root_assignments(args.artifact_root, label="artifact"),
        driver_python=args.driver_python,
    )
    orchestrator = PipelineOrchestrator(args.run_dir)
    _print_json(
        {
            "status": "initialized",
            "mode": "site_bound",
            "run_dir": str(Path(args.run_dir).expanduser().resolve()),
            "config": str(path),
            "run_id": orchestrator.config.run_id,
            "scene_id": orchestrator.config.scene_id,
            "stage_count": len(orchestrator.config.stages),
            "all_stages_have_commands": all(
                stage.command is not None for stage in orchestrator.config.stages.values()
            ),
            "models_launched": False,
            "external_outputs_created": False,
        }
    )
    return 0


def _cmd_site_preflight(args: argparse.Namespace) -> int:
    from video2world.site_pipeline import preflight_site_run

    report = preflight_site_run(args.run_dir, targets=args.stages)
    _print_json(report)
    return 0 if report["status"] == "passed" else 3


def _cmd_site_run(args: argparse.Namespace) -> int:
    from video2world.site_pipeline import SITE_RUN_RECEIPT_NAME, run_site_pipeline

    try:
        receipt = run_site_pipeline(args.run_dir, targets=args.stages)
    except StageBlockedError:
        receipt_path = (
            Path(args.run_dir).expanduser().resolve()
            / ".video2world"
            / SITE_RUN_RECEIPT_NAME
        )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        _print_json(receipt)
        return 3
    _print_json(receipt)
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


def _cmd_qwen_inventory(args: argparse.Namespace) -> int:
    frame_ids = parse_inventory_frame_ids(args.frame_id, args.frame_ids)
    result = run_inventory_provider(
        frames_dir=args.frames_dir,
        frame_ids=frame_ids,
        scene_id=args.scene_id,
        run_id=args.run_id,
        model_dir=args.model_dir,
        model_revision=args.model_revision,
        output_dir=args.output_dir,
        columns=args.columns,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        minimum_occlusion_confidence=args.minimum_occlusion_confidence,
    )
    _print_json(result)
    return 0


def _cmd_qwen_scene_audit(args: argparse.Namespace) -> int:
    from video2world.providers.qwen_scene_audit import (
        parse_frame_ids,
        run_scene_audit_provider,
    )

    result = run_scene_audit_provider(
        frame_ids=parse_frame_ids(args.frame_id, args.frame_ids),
        scene_id=args.scene_id,
        run_id=args.run_id,
        asset_summary_path=args.asset_summary,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        contact_sheet=args.contact_sheet,
        frames_dir=args.frames_dir,
        model_revision=args.model_revision,
        **({"model_repo": args.model_repo} if args.model_repo else {}),
        columns=args.columns,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
    _print_json(result)
    return 0


def _cmd_qwen_scene_audit_merge(args: argparse.Namespace) -> int:
    from video2world.providers.qwen_scene_audit import run_scene_audit_merge

    result = run_scene_audit_merge(
        audit_paths=args.audit,
        asset_summary_path=args.asset_summary,
        run_id=args.run_id,
        output_dir=args.output_dir,
    )
    _print_json(result)
    return 0


def _cmd_completion_plan(args: argparse.Namespace) -> int:
    inventory = load_scene_inventory(args.inventory)
    graph = load_occlusion_graph(args.occlusion_graph)
    plan = build_layered_completion_plan(
        inventory,
        graph,
        created_at=datetime.now(UTC),
        max_rounds=args.max_rounds,
        max_parallel_targets_per_round=args.max_parallel_targets,
    )
    output = Path(args.output).expanduser().resolve()
    write_layered_completion_plan(output, plan)
    _print_json(
        {
            "status": "written",
            "output": str(output),
            "strategy": plan.strategy,
            "round_count": len(plan.rounds),
            "object_round_count": sum(item.kind == "object_layer" for item in plan.rounds),
            "final_background_round": plan.rounds[-1].index,
        }
    )
    return 0


def _cmd_completion_validate(args: argparse.Namespace) -> int:
    plan = load_layered_completion_plan(args.plan)
    _print_json(
        {
            "valid": True,
            "scene_id": plan.scene_id,
            "run_id": plan.run_id,
            "strategy": plan.strategy,
            "round_count": len(plan.rounds),
            "round_statuses": [item.status for item in plan.rounds],
        }
    )
    return 0


def _cmd_completion_route(args: argparse.Namespace) -> int:
    evidence_path = Path(args.evidence).expanduser().resolve()
    evidence_payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    if not isinstance(evidence_payload, dict):
        raise ValueError("completion evidence root must be an object")
    if args.clean_plate_report:
        report_path = Path(args.clean_plate_report).expanduser().resolve()
        report_payload = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(report_payload, dict):
            raise ValueError("clean plate report root must be an object")
        next_action = clean_plate_next_action_from_report(report_payload)
        evidence_payload["clean_plate_next_action"] = next_action.model_dump(mode="json")
    evidence = CompletionEvidence.model_validate(evidence_payload)
    route = route_completion_backend(evidence)
    payload = route.model_dump(mode="json")
    if args.output:
        atomic_write_json(Path(args.output).expanduser().resolve(), payload)
    _print_json(payload)
    return 3 if route.selected_backend == "hold_for_more_evidence" else 0


def _cmd_completion_recovery_work_order(args: argparse.Namespace) -> int:
    from video2world.completion_recovery import materialize_completion_recovery_work_order

    work_order = materialize_completion_recovery_work_order(
        args.route,
        args.clean_plate_report,
    )
    payload = work_order.model_dump(mode="json")
    atomic_write_json(Path(args.output).expanduser().resolve(), payload)
    _print_json(payload)
    return 0


def _cmd_completion_recovery_bundle(args: argparse.Namespace) -> int:
    from video2world.completion_recovery import materialize_completion_recovery_bundle

    bundle = materialize_completion_recovery_bundle(args.work_order)
    payload = bundle.model_dump(mode="json")
    atomic_write_json(Path(args.output).expanduser().resolve(), payload)
    _print_json(payload)
    return 0


def _cmd_scene_command_validate(args: argparse.Namespace) -> int:
    command = load_scene_command(args.command_file)
    _print_json(
        {
            "valid": True,
            "schema_version": command.schema_version,
            "request_id": command.request_id,
            "intent": command.intent.kind,
            "mutating": command.is_mutating,
            "command_sha256": digest_json(command.model_dump(mode="json")),
            "normalized_command": command.model_dump(mode="json", exclude_none=True),
        }
    )
    return 0


def _cmd_scene_command_plan(args: argparse.Namespace) -> int:
    manifest = load_world_manifest(args.manifest)
    command = load_scene_command(args.command_file)
    plan = plan_scene_command(manifest, command, created_at=datetime.now(UTC))
    if args.output:
        write_scene_command_plan(args.output, plan)
    _print_json(plan.model_dump(mode="json", exclude_none=True))
    return 0 if plan.status == "ready" else 3


def _cmd_scene_command_submit(args: argparse.Namespace) -> int:
    manifest = load_world_manifest(args.manifest)
    command = load_scene_command(args.command_file)
    receipt = submit_scene_command(
        manifest,
        command,
        queue_dir=args.queue_dir,
        confirmation_phrase=args.confirm,
        submitted_at=datetime.now(UTC),
    )
    _print_json(receipt.model_dump(mode="json", exclude_none=True))
    return 3 if receipt.status == "blocked" else 0


def _cmd_scene_command_serve(args: argparse.Namespace) -> int:
    from video2world.scene_command_server import (
        PLAN_PATH,
        SUBMIT_PATH,
        create_scene_command_server,
        serve_scene_commands,
    )

    if args.host not in {"127.0.0.1", "::1", "localhost"} and not args.allow_non_loopback:
        raise ValueError(
            "non-loopback bind requires --allow-non-loopback; the service has no user auth"
        )
    server = create_scene_command_server(
        manifest_path=args.manifest,
        web_manifest_path=args.web_manifest,
        queue_dir=args.queue_dir,
        host=args.host,
        port=args.port,
        allowed_origins=args.cors_origin,
        max_json_bytes=args.max_json_bytes,
    )
    bound_host, bound_port = server.server_address[:2]
    _print_json(
        {
            "status": "serving",
            "host": bound_host,
            "port": bound_port,
            "plan_path": PLAN_PATH,
            "submit_path": SUBMIT_PATH,
            "cors_origins": sorted(args.cors_origin),
        }
    )
    sys.stdout.flush()
    try:
        serve_scene_commands(server)
    except KeyboardInterrupt:
        return 0
    return 0


def _cmd_web_manifest_adopt(args: argparse.Namespace) -> int:
    manifest = adopt_web_manifest_file(args.web_manifest, args.output)
    output = Path(args.output).expanduser().resolve()
    sidecar_digest = digest_path(output)
    _print_json(
        {
            "status": "written",
            "output": str(output),
            "manifest_status": manifest.manifest_status,
            "world_id": manifest.world_id,
            "run_id": manifest.run_id,
            "object_count": len(manifest.objects),
            "canonical_manifest_sha256": digest_json(manifest.model_dump(mode="json")),
            "sidecar_file_sha256": sidecar_digest.sha256,
            "sidecar_size_bytes": sidecar_digest.size_bytes,
            "limitations": manifest.provenance.notes,
        }
    )
    return 0


def _cmd_web_manifest_bind(args: argparse.Namespace) -> int:
    output = Path(args.output).expanduser().resolve()
    bound, canonical_sha256 = bind_scene_command_web_manifest(
        web_manifest_source=args.web_manifest,
        canonical_manifest_path=args.canonical_manifest,
        destination=output,
        endpoint=args.endpoint,
        derived_version=args.derived_version,
    )
    output_digest = digest_path(output)
    _print_json(
        {
            "status": "written",
            "output": str(output),
            "version": bound["version"],
            "canonical_manifest_sha256": canonical_sha256,
            "bound_web_manifest_sha256": output_digest.sha256,
            "bound_web_manifest_size_bytes": output_digest.size_bytes,
            "scene_command_service": bound["sceneCommandService"],
        }
    )
    return 0


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
    "site-init": _cmd_site_init,
    "site-preflight": _cmd_site_preflight,
    "site-run": _cmd_site_run,
    "adopt": _cmd_adopt,
    "adopt-existing": _cmd_adopt,
    "validate": _cmd_validate,
    "query": _cmd_query,
    "qwen-inventory": _cmd_qwen_inventory,
    "qwen-scene-audit": _cmd_qwen_scene_audit,
    "qwen-scene-audit-merge": _cmd_qwen_scene_audit_merge,
    "completion-plan": _cmd_completion_plan,
    "completion-validate": _cmd_completion_validate,
    "completion-route": _cmd_completion_route,
    "completion-recovery-work-order": _cmd_completion_recovery_work_order,
    "completion-recovery-bundle": _cmd_completion_recovery_bundle,
    "scene-command-validate": _cmd_scene_command_validate,
    "scene-command-plan": _cmd_scene_command_plan,
    "scene-command-submit": _cmd_scene_command_submit,
    "scene-command-serve": _cmd_scene_command_serve,
    "web-manifest-adopt": _cmd_web_manifest_adopt,
    "web-manifest-bind": _cmd_web_manifest_bind,
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
