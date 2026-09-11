#!/usr/bin/env python3
"""Start, discover and stop independent loopback module workers without model inference."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))
from world_modeling.engine import load_profile, read_json, write_json
from world_modeling.modules import REGISTRY
from world_modeling.services import discover_service, process_identity, process_state, register_service

STARTED_PROCESSES = {}


class LauncherError(RuntimeError):
    pass


def deployment_directory(root, deployment_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", deployment_id):
        raise LauncherError("invalid deployment ID")
    outputs = (root / "outputs").resolve()
    path = (outputs / deployment_id).resolve()
    if path.parent != outputs or root.resolve() not in path.parents:
        raise LauncherError("deployment path escapes repository outputs")
    return path


def selected_modules(profile, selected):
    names = selected or profile.get("pipeline") or list(profile["providers"])
    if not isinstance(names, list) or not names or len(set(names)) != len(names):
        raise LauncherError("select a nonempty, unique list of modules")
    for name in names:
        REGISTRY.get(name)
        provider = profile["providers"].get(name, {})
        command = provider.get("command")
        if provider.get("kind") != "command" or not isinstance(command, list) or not command or not all(isinstance(item, str) for item in command):
            raise LauncherError(f"{name}: worker needs a real command provider, not contract or service recursion")
    return names


def worker_environment():
    environment = os.environ.copy()
    paths = [str(CODE_ROOT / "src")]
    if environment.get("PYTHONPATH"):
        paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def free_port(host, first):
    for port in range(first, min(first + 1000, 65536)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind((host, port))
            except OSError:
                continue
            return port
    raise LauncherError("no free loopback port in the requested range")


def owned_state(record):
    if not isinstance(record.get("process_identity"), dict) or record.get("pid") != record["process_identity"].get("pid"):
        return "unknown"
    process = STARTED_PROCESSES.get(record["pid"])
    if process is not None and process.poll() is not None:
        record["returncode"] = process.returncode
        del STARTED_PROCESSES[record["pid"]]
        return "exited"
    state = process_state(record["process_identity"])
    if state == "alive":
        stat = Path(f"/proc/{record['pid']}/stat")
        if stat.is_file() and stat.read_text().rsplit(") ", 1)[1].split()[0] == "Z":
            state = "exited"
    if state == "exited":
        try:
            os.waitpid(record["pid"], os.WNOHANG)
        except (ChildProcessError, OSError):
            pass
    return state


def wait_ready(process, record, token_env, timeout):
    deadline = time.monotonic() + timeout
    ready_line = f"module={record['module']} endpoint={record['endpoint']}"
    last_error = "worker has not announced its listening socket"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise LauncherError(f"{record['module']}: worker exited {process.returncode}; see {record['log_path']}")
        # Discovery alone could hit an unrelated listener that won a port race.
        # Only the freshly started child can write this deployment's new log.
        if ready_line in Path(record["log_path"]).read_text(errors="replace"):
            try:
                discovered = discover_service(record["endpoint"], token_env)
                modules = discovered["modules"]
                if [item["id"] for item in modules] != [record["module"]]:
                    raise LauncherError("listening endpoint serves a different module")
                if process.poll() is not None:
                    raise LauncherError("worker exited during discovery")
                return discovered
            except (ValueError, OSError) as error:
                last_error = type(error).__name__
        time.sleep(.05)
    raise LauncherError(f"{record['module']}: startup timeout ({last_error}); see {record['log_path']}")


def stop_records(records, timeout):
    for record in records:
        state = owned_state(record)
        record["process_state"] = state
        if state == "alive":
            # SIGINT lets the CLI drain its provider executor. Never kill a
            # process group or escalate to SIGKILL when a model is still running.
            try:
                os.kill(record["pid"], signal.SIGINT)
                record["stop_requested"] = True
            except ProcessLookupError:
                record["process_state"] = "exited"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and any(owned_state(record) == "alive" for record in records):
        time.sleep(.05)
    for record in records:
        record["process_state"] = owned_state(record)
    return all(record["process_state"] == "exited" for record in records)


def start(args):
    root = args.root.resolve()
    directory = deployment_directory(root, args.deployment_id)
    if args.host not in {"127.0.0.1", "localhost"}:
        raise LauncherError("module workers must bind IPv4 loopback; use an authenticated reverse proxy for remote access")
    if not 1024 <= args.base_port <= 65535 or not 0 < args.startup_timeout <= 60:
        raise LauncherError("invalid base port or startup timeout")
    if args.token_env and (not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.token_env) or not os.environ.get(args.token_env)):
        raise LauncherError("service token variable is invalid or absent; only its name may be configured")
    profile_path = args.profile if args.profile.is_absolute() else root / args.profile
    profile = load_profile(profile_path)
    names = selected_modules(profile, args.module)
    if args.dry_run:
        return {"status": "dry_run", "modules": names, "deployment_directory": str(directory), "models_started": False,
                "registry_path": str(directory / "services.registry.json"), "pipeline_profile_path": str(directory / "services.pipeline.json")}
    if directory.exists():
        raise LauncherError("deployment directory already exists; inspect status or use a new deployment ID; existing processes are never replaced")
    directory.mkdir(parents=True)
    log_dir = directory / "logs"
    log_dir.mkdir()
    snapshot = directory / "provider-profile.json"
    write_json(snapshot, profile)
    registry_path = directory / "services.registry.json"
    write_json(registry_path, {"schema_version": "1.0", "services": {}})
    manifest_path = directory / "deployment.json"
    manifest = {"schema_version": "1.0", "deployment_id": args.deployment_id, "status": "starting", "root": str(root),
                "created_at": datetime.now(timezone.utc).isoformat(), "source_profile_path": str(profile_path.resolve()),
                "source_profile_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(), "profile_snapshot_path": str(snapshot),
                "registry_path": str(registry_path), "service_token_env": args.token_env, "workers": [], "models_started_by_launcher": False}
    write_json(manifest_path, manifest)
    next_port = args.base_port
    try:
        for name in names:
            port = free_port(args.host, next_port)
            next_port = port + 1
            endpoint = f"http://{args.host}:{port}"
            command = [str(args.python), "-u", "-m", "world_modeling", "--root", str(root), "serve", name,
                       "--profile", str(snapshot), "--host", args.host, "--port", str(port)]
            if args.token_env:
                command += ["--token-env", args.token_env]
            log = log_dir / (name + ".log")
            with log.open("xb") as stream:
                process = subprocess.Popen(command, cwd=root, env=worker_environment(), stdin=subprocess.DEVNULL,
                                           stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            STARTED_PROCESSES[process.pid] = process
            record = {"module": name, "pid": process.pid, "process_identity": process_identity(process.pid), "endpoint": endpoint,
                      "command": command, "log_path": str(log), "status": "starting"}
            manifest["workers"].append(record)
            write_json(manifest_path, manifest)
            discovery = wait_ready(process, record, args.token_env, args.startup_timeout)
            register_service(registry_path, endpoint, args.token_env)
            record.update({"status": "registered", "discovery": discovery})
            write_json(manifest_path, manifest)
        pipeline_path = directory / "services.pipeline.json"
        write_json(pipeline_path, {"schema_version": "1.0", "profile_id": args.deployment_id + "-services",
                                   "registry": registry_path.name, "pipeline": names,
                                   "providers": {name: {"kind": "service"} for name in names}})
        manifest.update({"status": "ready", "pipeline_profile_path": str(pipeline_path)})
        write_json(manifest_path, manifest)
        return manifest
    except BaseException as error:
        stopped = stop_records(manifest["workers"], 5)
        manifest.update({"status": "failed" if stopped else "failed_workers_draining", "failure_type": type(error).__name__})
        write_json(manifest_path, manifest)
        raise


def status(args):
    directory = deployment_directory(args.root.resolve(), args.deployment_id)
    manifest = read_json(directory / "deployment.json")
    for record in manifest["workers"]:
        record["process_state"] = owned_state(record)
    return manifest


def stop(args):
    if not 0 <= args.wait_seconds <= 60:
        raise LauncherError("stop timeout must be between 0 and 60 seconds")
    directory = deployment_directory(args.root.resolve(), args.deployment_id)
    path = directory / "deployment.json"
    manifest = read_json(path)
    stopped = stop_records(manifest["workers"], args.wait_seconds)
    manifest["status"] = "stopped" if stopped else "draining_or_identity_unknown"
    manifest["stop_requested_at"] = datetime.now(timezone.utc).isoformat()
    write_json(path, manifest)
    return manifest


def parser():
    app = argparse.ArgumentParser(description=__doc__)
    app.add_argument("--root", type=Path, default=CODE_ROOT)
    commands = app.add_subparsers(dest="action", required=True)
    start_command = commands.add_parser("start")
    start_command.add_argument("deployment_id")
    start_command.add_argument("--profile", type=Path, default=Path("profiles/seetacloud.json"))
    start_command.add_argument("--module", action="append", help="Serve selected modules; default is every module configured by the profile")
    start_command.add_argument("--python", type=Path, default=Path(sys.executable), help="CPU control interpreter; model interpreters remain in provider argv")
    start_command.add_argument("--host", default="127.0.0.1")
    start_command.add_argument("--base-port", type=int, default=8810)
    start_command.add_argument("--token-env", help="Optional module-service authentication variable name, distinct from model-provider credentials")
    start_command.add_argument("--startup-timeout", type=float, default=20)
    start_command.add_argument("--dry-run", action="store_true")
    status_command = commands.add_parser("status")
    status_command.add_argument("deployment_id")
    stop_command = commands.add_parser("stop")
    stop_command.add_argument("deployment_id")
    stop_command.add_argument("--wait-seconds", type=float, default=10)
    return app


if __name__ == "__main__":
    try:
        args = parser().parse_args()
        result = {"start": start, "status": status, "stop": stop}[args.action](args)
        print(json.dumps(result, indent=2))
    except (LauncherError, ValueError, KeyError, OSError, RuntimeError) as error:
        print(f"module launcher failed: {error}", file=sys.stderr)
        raise SystemExit(2)
