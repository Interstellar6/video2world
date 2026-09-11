from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .engine import (PipelineError, Task, create_task, load_profile, run_pipeline, run_stage,
                     validate_task, without_bg_recon)
from .modules import REGISTRY, SPECS


def root_from_args(value: str | None) -> Path:
    return Path(value).resolve() if value else Path.cwd().resolve()


def parser() -> argparse.ArgumentParser:
    app = argparse.ArgumentParser(prog="world-modeling")
    app.add_argument("--root", help="Repository root; defaults to current directory")
    commands = app.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init-task", help="Create outputs/<task_id> and copy source media into it")
    init.add_argument("task_id")
    init.add_argument("--source", type=Path)
    run = commands.add_parser("run", help="Run modular stages in declaration order")
    run.add_argument("task_id")
    run.add_argument("--profile", type=Path, default=Path("profiles/seetacloud.json"))
    run.add_argument("--through", choices=list(SPECS))
    run.add_argument("--skip-bg-recon", action="store_true",
                     help="Skip clean_plate, background_reconstruction and scene_recomposition")
    stage = commands.add_parser("run-stage", help="Run one module using the same task artifact interface")
    stage.add_argument("task_id")
    stage.add_argument("stage", choices=list(SPECS))
    stage.add_argument("--profile", type=Path, default=Path("profiles/seetacloud.json"))
    validate = commands.add_parser("validate", help="Verify task-scoped artifact paths and hashes")
    validate.add_argument("task_id")
    commands.add_parser("plan", help="Print all module boundaries")
    discover = commands.add_parser("discover", help="Discover module contracts or live registered services")
    discover.add_argument("--registry", type=Path)
    register = commands.add_parser("register", help="Register a live module service after contract verification")
    register.add_argument("--registry", type=Path, required=True)
    register.add_argument("--endpoint", required=True)
    register.add_argument("--token-env")
    recover = commands.add_parser("recover-job", help="Reconcile an interrupted job only after its worker and provider exited")
    recover.add_argument("task_id")
    recover.add_argument("module", choices=list(SPECS))
    recover.add_argument("request_id")
    recover.add_argument("--endpoint", required=True)
    recover.add_argument("--token-env")
    serve = commands.add_parser("serve", help="Serve one module independently using its local command provider")
    serve.add_argument("module", choices=list(SPECS))
    serve.add_argument("--profile", type=Path, default=Path("profiles/seetacloud.json"))
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, required=True)
    serve.add_argument("--token-env")
    return app


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root = root_from_args(args.root)
    try:
        if args.command == "plan":
            print(json.dumps({"modules": [spec.as_dict() for spec in REGISTRY.modules]}, indent=2))
            return 0
        if args.command in {"register", "discover", "serve", "recover-job"}:
            import os
            from .engine import read_json
            from .services import ModuleService, discover_service, make_server, register_service

            if args.command == "recover-job":
                from .services import recover_job
                print(json.dumps(recover_job(args.endpoint, args.task_id, args.module, args.request_id, args.token_env), indent=2))
            elif args.command == "register":
                print(json.dumps(register_service(args.registry, args.endpoint, args.token_env), indent=2))
            elif args.command == "discover":
                if args.registry:
                    registry = read_json(args.registry)
                    result = {name: discover_service(item["endpoint"], item.get("token_env"))
                              for name, item in registry["services"].items()}
                else:
                    result = {"modules": REGISTRY.discover()}
                print(json.dumps(result, indent=2))
            else:
                profile_path = args.profile if args.profile.is_absolute() else root / args.profile
                profile = load_profile(profile_path)
                service = ModuleService(root, REGISTRY.get(args.module), profile["providers"][args.module])
                token = os.environ.get(args.token_env) if args.token_env else None
                if args.token_env and not token:
                    raise PipelineError(f"missing service token variable: {args.token_env}")
                server = make_server(service, args.host, args.port, token)
                import signal
                stopping = False

                def stop_service(signum, frame):
                    nonlocal stopping
                    if not stopping:
                        stopping = True
                        raise KeyboardInterrupt

                signal.signal(signal.SIGINT, stop_service)
                signal.signal(signal.SIGTERM, stop_service)
                print(f"module={args.module} endpoint=http://{args.host}:{server.server_port}", flush=True)
                try:
                    server.serve_forever()
                except KeyboardInterrupt:
                    pass
                finally:
                    stopping = True
                    server.server_close()
                    service.executor.shutdown(wait=True)
            return 0
        if args.command == "init-task":
            task = create_task(root, args.task_id, args.source)
            print(task.directory)
            return 0
        task = Task(root, args.task_id)
        if not task.directory.is_dir():
            raise PipelineError(f"unknown task: {task.directory}")
        if args.command in {"run", "run-stage"}:
            profile_path = args.profile if args.profile.is_absolute() else root / args.profile
            profile = load_profile(profile_path)
            if args.command == "run-stage":
                provider = profile["providers"].get(args.stage)
                if not isinstance(provider, dict):
                    raise PipelineError(f"profile has no provider binding for {args.stage}")
                state = run_stage(task, SPECS[args.stage], provider)
            else:
                if getattr(args, "skip_bg_recon", False):
                    profile["pipeline"] = without_bg_recon(profile["pipeline"])
                state = run_pipeline(task, profile, args.through)
            print(json.dumps(state, indent=2))
            return 0
        report = validate_task(task)
        print(json.dumps(report, indent=2))
        return 0 if report["valid"] else 1
    except (PipelineError, ValueError, KeyError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
