from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("module_launcher", ROOT / "scripts/serve_modules.py")
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class ProfileTests(unittest.TestCase):
    def test_all_twelve_modules_have_real_abi_bindings_and_safe_budget(self):
        profile = launcher.load_profile(ROOT / "profiles/seetacloud.json")
        self.assertEqual(len(profile["pipeline"]), 12)
        self.assertEqual(set(profile["pipeline"]), set(profile["providers"]))
        launcher.REGISTRY.ordered(profile["pipeline"])
        for module, binding in profile["providers"].items():
            self.assertEqual(binding["kind"], "command", module)
            for flag, value in (("--task-dir", "{task_dir}"), ("--inputs", "{inputs_json}"), ("--outputs", "{outputs_json}")):
                self.assertEqual(binding["command"][binding["command"].index(flag) + 1], value)
            self.assertNotIn("--resume-run-dir", binding["command"])
        clean = profile["providers"]["clean_plate"]["command"]
        self.assertEqual(clean[clean.index("--max-api-requests") + 1], "0")
        orbit = profile["providers"]["orbit_video"]["command"]
        self.assertEqual(orbit[orbit.index("--seed") + 1], "20260829")
        self.assertEqual(orbit[orbit.index("--num-inference-steps") + 1], "10")
        self.assertEqual(orbit[orbit.index("--clean-frame-indices") + 1], "")
        self.assertFalse(any("API_KEY=" in part for binding in profile["providers"].values() for part in binding["command"]))

    def test_default_dry_run_does_not_start_workers_or_create_deployment(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = launcher.parser().parse_args(["--root", temporary, "start", "dry", "--profile", str(ROOT / "profiles/seetacloud.json"), "--dry-run"])
            with patch.object(launcher.subprocess, "Popen", side_effect=AssertionError("must not start a worker")):
                result = launcher.start(args)
            self.assertEqual(len(result["modules"]), 12)
            self.assertEqual(result["status"], "dry_run")
            self.assertFalse(Path(temporary, "outputs").exists())

    def test_optional_preserved_task_recipe_keeps_stage_one_and_two(self):
        historical = ROOT / "outputs/bedroom4-fresh-modular-20260907/run-profile.json"
        if not historical.exists():
            self.skipTest("historical task output is intentionally not committed")
        original = launcher.read_json(historical)
        current = launcher.read_json(ROOT / "profiles/seetacloud.json")
        scene = copy.deepcopy(original["providers"]["scene_reconstruction"])
        index = scene["command"].index("--resume-run-dir")
        del scene["command"][index:index + 2]
        self.assertEqual(scene, current["providers"]["scene_reconstruction"])
        self.assertEqual(original["providers"]["scene_understanding"], current["providers"]["scene_understanding"])


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.profile = self.root / "commands.json"
        revision = self.root / "fixture-revision.json"
        revision.write_text('{"fixture": "discovery-only-v1"}')
        launcher.write_json(self.profile, {"schema_version": "1.0", "profile_id": "cpu-launcher-fixture", "providers": {
            "scene_understanding": {"kind": "command", "revision_files": [str(revision)],
                                    "command": [sys.executable, "-c", "raise RuntimeError('model command must never be invoked during discovery')"]}}})
        self.deployments = []

    def tearDown(self):
        for deployment in self.deployments:
            path = self.root / "outputs" / deployment / "deployment.json"
            if path.exists():
                launcher.stop(self.args("stop", deployment))
        self.temporary.cleanup()

    def args(self, action="start", deployment="cpu-worker", *extra):
        values = ["--root", str(self.root), action, deployment]
        if action == "start":
            values += ["--profile", str(self.profile), "--startup-timeout", "10"]
        return launcher.parser().parse_args(values + list(extra))

    def test_start_discover_pipeline_profile_and_graceful_owned_stop(self):
        self.deployments.append("cpu-worker")
        manifest = launcher.start(self.args())
        self.assertEqual(manifest["status"], "ready")
        self.assertFalse(manifest["models_started_by_launcher"])
        self.assertEqual(len(manifest["workers"]), 1)
        record = manifest["workers"][0]
        self.assertEqual(launcher.owned_state(record), "alive")
        self.assertTrue(Path(record["log_path"]).is_file())
        profile = launcher.load_profile(Path(manifest["pipeline_profile_path"]))
        self.assertEqual(profile["pipeline"], ["scene_understanding"])
        self.assertEqual(profile["providers"]["scene_understanding"]["kind"], "service")
        self.assertTrue(profile["providers"]["scene_understanding"]["provider_revision"])
        snapshot = launcher.status(self.args("status"))
        self.assertEqual(snapshot["workers"][0]["process_state"], "alive")
        stopped = launcher.stop(self.args("stop"))
        self.assertEqual(stopped["status"], "stopped")
        self.assertEqual(stopped["workers"][0]["process_state"], "exited")

    def test_busy_port_is_skipped_without_touching_existing_listener(self):
        self.deployments.append("busy")
        with socket.socket() as existing:
            existing.bind(("127.0.0.1", 0))
            existing.listen()
            port = existing.getsockname()[1]
            manifest = launcher.start(self.args("start", "busy", "--base-port", str(port)))
            self.assertNotEqual(manifest["workers"][0]["endpoint"], f"http://127.0.0.1:{port}")
            self.assertEqual(existing.getsockname()[1], port)

    def test_existing_deployment_directory_is_never_overwritten(self):
        directory = self.root / "outputs/existing"
        directory.mkdir(parents=True)
        marker = directory / "user-owned.txt"
        marker.write_text("preserve")
        with self.assertRaisesRegex(launcher.LauncherError, "already exists"):
            launcher.start(self.args("start", "existing"))
        self.assertEqual(marker.read_text(), "preserve")

    def test_remote_bind_and_task_escape_are_rejected(self):
        with self.assertRaisesRegex(launcher.LauncherError, "loopback"):
            launcher.start(self.args("start", "remote", "--host", "0.0.0.0"))
        with self.assertRaisesRegex(launcher.LauncherError, "invalid deployment"):
            launcher.start(self.args("start", "../escape"))

    def test_output_symlink_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as other:
            (self.root / "outputs").symlink_to(other, target_is_directory=True)
            with self.assertRaisesRegex(launcher.LauncherError, "escapes repository"):
                launcher.start(self.args("start", "outside"))

    def test_failed_startup_stops_only_workers_started_by_this_deployment(self):
        self.deployments.append("startup-failure")
        profile = launcher.read_json(self.profile)
        profile["providers"]["component_segmentation"] = {"kind": "command", "command": [sys.executable, str(self.root / "missing-adapter.py")]}
        profile["pipeline"] = ["scene_understanding", "component_segmentation"]
        launcher.write_json(self.profile, profile)
        with self.assertRaisesRegex(launcher.LauncherError, "worker exited"):
            launcher.start(self.args("start", "startup-failure"))
        manifest = launcher.read_json(self.root / "outputs/startup-failure/deployment.json")
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(len(manifest["workers"]), 2)
        self.assertTrue(all(launcher.owned_state(record) == "exited" for record in manifest["workers"]))
        self.assertFalse((self.root / "outputs/startup-failure/services.pipeline.json").exists())

    def test_mismatched_or_foreign_process_identity_is_not_signalled(self):
        record = {"pid": os.getpid(), "process_identity": launcher.process_identity(os.getpid())}
        record["process_identity"]["hostname"] = "not-the-current-host"
        with patch.object(launcher.os, "kill", side_effect=AssertionError("must not signal an unowned process")):
            self.assertFalse(launcher.stop_records([record], 0))
        record = {"pid": os.getpid(), "process_identity": launcher.process_identity(os.getpid())}
        record["pid"] += 1
        with patch.object(launcher.os, "kill", side_effect=AssertionError("must not signal inconsistent PID")):
            self.assertFalse(launcher.stop_records([record], 0))

    def test_contract_and_service_recursion_are_not_local_workers(self):
        for kind in ("contract", "service"):
            profile = {"providers": {"scene_understanding": {"kind": kind}}}
            with self.assertRaisesRegex(launcher.LauncherError, "real command"):
                launcher.selected_modules(profile, None)

    def test_service_secret_is_inherited_by_name_not_serialized(self):
        self.deployments.append("authenticated")
        with patch.dict(os.environ, {"LAUNCHER_TEST_TOKEN": "cpu-test-service-secret"}):
            manifest = launcher.start(self.args("start", "authenticated", "--token-env", "LAUNCHER_TEST_TOKEN"))
            self.assertEqual(manifest["service_token_env"], "LAUNCHER_TEST_TOKEN")
            for path in (self.root / "outputs/authenticated").rglob("*.json"):
                self.assertNotIn("cpu-test-service-secret", path.read_text())
            launcher.stop(self.args("stop", "authenticated"))


if __name__ == "__main__":
    unittest.main()
