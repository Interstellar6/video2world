from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import world_modeling.engine as engine
from world_modeling.engine import Task, create_task, exclusive_lock, finalize, load_profile, run_pipeline, run_stage, snapshot_path, validate_task, write_json
from world_modeling.modules import REGISTRY, SPECS
from world_modeling.processes import process_identity, terminate_started_process


REPO = Path(__file__).resolve().parents[1]


class PipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "scan.mp4"
        self.source.write_bytes(b"fixture scan")
        self.task = create_task(self.root, "bedroom-contract", self.source)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def command_profile(self) -> dict:
        provider = self.root / "provider.py"
        provider.write_text(
            "import json, pathlib, sys\n"
            "outputs = pathlib.Path(sys.argv[1])\n"
            "directory = outputs.parent / 'provider'\n"
            "directory.mkdir(parents=True, exist_ok=True)\n"
            "counter = directory / 'count.txt'\n"
            "count = int(counter.read_text()) + 1 if counter.exists() else 1\n"
            "counter.write_text(str(count))\n"
            "roles = ('frames_manifest', 'cameras', 'scene_depth', 'scene_gaussian_ply', 'scene_tsdf_mesh')\n"
            "outputs.write_text(json.dumps({'outputs': {role: {'path': str(counter.relative_to(outputs.parents[2])), 'evidence': 'observed'} for role in roles}}))\n",
            encoding="utf-8",
        )
        profile = load_profile(REPO / "profiles" / "contract.json")
        profile["providers"]["scene_reconstruction"] = {
            "kind": "command", "command": [sys.executable, str(provider), "{outputs_json}"],
        }
        return profile

    def execution_count(self) -> int:
        return int((self.task.directory / "stages/scene_reconstruction/provider/count.txt").read_text())

    def test_task_constructor_confines_all_entry_points(self):
        for task_id in ("../outside", "../../outside", "/tmp/elsewhere", "", ".", ".."):
            with self.subTest(task_id=task_id), self.assertRaises(RuntimeError):
                Task(self.root, task_id)
        outside = self.root / "outside"
        outside.mkdir()
        (self.root / "outputs/linked").symlink_to(outside)
        with self.assertRaisesRegex(RuntimeError, "escapes"):
            Task(self.root, "linked")

    def test_task_lock_rejects_concurrent_state_mutation(self):
        state_before = self.task.load_state()
        with exclusive_lock(self.task.directory / "orchestrator.lock"):
            with self.assertRaisesRegex(RuntimeError, "another execution"):
                run_stage(self.task, SPECS["scene_reconstruction"], {"kind": "contract"})
        self.assertEqual(self.task.load_state(), state_before)

    def test_registered_plugin_is_immediately_loadable_and_removable(self):
        from world_modeling.cli import parser
        from world_modeling.contracts import ModuleSpec, role

        spec = ModuleSpec("test_plugin", "Test plugin", ("source_media",),
                          (role("plugin_output", "derived", "application/json"),), "test", "")
        REGISTRY.register(spec)
        try:
            path = self.root / "plugin-profile.json"
            write_json(path, {"schema_version": "1.0", "pipeline": [spec.id], "providers": {spec.id: {"kind": "contract"}}})
            profile = load_profile(path)
            self.assertEqual(parser().parse_args(["run-stage", self.task.task_id, spec.id]).stage, spec.id)
            run_pipeline(self.task, profile)
            self.assertTrue(validate_task(self.task)["valid"])
        finally:
            REGISTRY.unregister(spec.id)
        self.assertNotIn(spec.id, SPECS)

    def test_removing_a_stage_retires_its_failure_without_deleting_evidence(self):
        profile = load_profile(REPO / "profiles/contract.json")
        run_pipeline(self.task, profile, through="scene_reconstruction")
        with self.assertRaises(RuntimeError):
            run_stage(self.task, SPECS["scene_understanding"], {"kind": "command", "command": ["/usr/bin/false"]})
        profile["pipeline"] = ["scene_reconstruction"]
        state = run_pipeline(self.task, profile)
        self.assertIn("scene_understanding", state["retired_stages"])
        self.assertNotIn("scene_understanding", state["stages"])
        self.assertTrue((self.task.directory / "stages/scene_understanding/stage.json").exists())
        self.assertTrue(validate_task(self.task)["valid"])

    def test_contract_pipeline_is_task_scoped_and_complete(self) -> None:
        profile = load_profile(REPO / "profiles" / "contract.json")
        state = run_pipeline(self.task, profile)
        self.assertEqual(state["status"], "contract_only")
        self.assertFalse(state["promotion_allowed"])
        self.assertEqual(len(state["stages"]), 12)
        artifacts = self.task.load_artifacts()["artifacts"]
        self.assertIn("physical_object_obj", artifacts)
        self.assertEqual(artifacts["physical_object_obj"]["evidence"], "contract_only")
        for artifact in artifacts.values():
            self.assertTrue((self.task.directory / artifact["path"]).is_file())
        self.assertTrue(validate_task(self.task)["valid"])

    def test_generated_clean_plate_is_never_collision_eligible(self) -> None:
        profile = load_profile(REPO / "profiles" / "contract.json")
        run_pipeline(self.task, profile, through="clean_plate")
        artifacts = self.task.load_artifacts()["artifacts"]
        self.assertEqual(artifacts["clean_plate_frames"]["declared_evidence"], "generated")
        self.assertFalse(artifacts["clean_plate_frames"]["collision_eligible"])

    def test_contract_stage_can_be_replaced_by_a_real_provider(self) -> None:
        profile = load_profile(REPO / "profiles" / "contract.json")
        run_pipeline(self.task, profile, through="scene_reconstruction")
        profile["providers"]["scene_reconstruction"] = {
            "kind": "command",
            "command": ["/usr/bin/false"],
        }
        with self.assertRaisesRegex(RuntimeError, "provider failed"):
            run_pipeline(self.task, profile, through="scene_reconstruction")

    def test_command_provider_hashes_directory_output(self) -> None:
        provider = self.root / "directory_provider.py"
        provider.write_text(
            "import json, pathlib, sys\n"
            "outputs = pathlib.Path(sys.argv[1])\n"
            "directory = outputs.parent / 'provider' / 'sequence'\n"
            "directory.mkdir(parents=True)\n"
            "(directory / 'frame_000.png').write_bytes(b'frame')\n"
            "roles = ('frames_manifest', 'cameras', 'scene_depth', 'scene_gaussian_ply', 'scene_tsdf_mesh')\n"
            "outputs.write_text(json.dumps({'outputs': {role: {'path': str(directory.relative_to(outputs.parents[2])), 'evidence': 'observed'} for role in roles}}))\n",
            encoding="utf-8",
        )
        profile = load_profile(REPO / "profiles" / "contract.json")
        profile["providers"]["scene_reconstruction"] = {
            "kind": "command",
            "command": [sys.executable, str(provider), "{outputs_json}"],
        }
        run_pipeline(self.task, profile, through="scene_reconstruction")
        artifact = self.task.load_artifacts()["artifacts"]["scene_depth"]
        self.assertEqual(artifact["size_bytes"], 5)
        self.assertTrue(validate_task(self.task)["valid"])

    def test_validation_detects_artifact_tampering(self) -> None:
        profile = load_profile(REPO / "profiles" / "contract.json")
        run_pipeline(self.task, profile, through="scene_reconstruction")
        artifacts = self.task.load_artifacts()["artifacts"]
        path = self.task.directory / artifacts["scene_depth"]["path"]
        path.write_text(json.dumps({"tampered": True}), encoding="utf-8")
        report = validate_task(Task(self.root, self.task.task_id))
        self.assertFalse(report["valid"])
        self.assertIn("sha256 mismatch", report["issues"][0])

    def test_invalid_source_does_not_create_a_partial_task(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "source must be an existing file or directory"):
            create_task(self.root, "bad-input", self.root / "missing.mp4")
        self.assertFalse((self.root / "outputs" / "bad-input").exists())

    def test_directory_source_is_copied_and_hash_checked(self) -> None:
        source = self.root / "calibrated"
        source.mkdir()
        (source / "frame_000.png").write_bytes(b"rgb")
        (source / "transforms.json").write_text("{}", encoding="utf-8")
        task = create_task(self.root, "directory-input", source)
        artifact = task.load_artifacts()["artifacts"]["source_media"]
        copied = task.directory / artifact["path"]
        self.assertTrue(copied.is_dir())
        self.assertEqual(artifact["size_bytes"], 5)
        self.assertTrue(validate_task(task)["valid"])

    def test_dangling_source_symlink_is_rejected_without_partial_task(self) -> None:
        source = self.root / "broken-input"
        source.mkdir()
        (source / "missing.png").symlink_to(self.root / "not-here.png")
        with self.assertRaisesRegex(RuntimeError, "dangling symlink"):
            create_task(self.root, "broken-link", source)
        self.assertFalse((self.root / "outputs" / "broken-link").exists())

    def test_resume_requires_matching_inputs_and_provider_config(self) -> None:
        profile = self.command_profile()
        run_pipeline(self.task, profile, through="scene_reconstruction")
        run_pipeline(self.task, profile, through="scene_reconstruction")
        self.assertEqual(self.execution_count(), 1)
        profile["providers"]["scene_reconstruction"]["revision"] = "new-config"
        run_pipeline(self.task, profile, through="scene_reconstruction")
        self.assertEqual(self.execution_count(), 2)
        artifacts = self.task.load_artifacts()
        source = artifacts["artifacts"]["source_media"]
        path = self.task.directory / source["path"]
        path.write_bytes(b"new registered source")
        source["sha256"], source["size_bytes"] = snapshot_path(path)
        write_json(self.task.artifacts_path, artifacts)
        run_pipeline(self.task, profile, through="scene_reconstruction")
        self.assertEqual(self.execution_count(), 3)

    def test_adapter_in_place_change_invalidates_cached_result_and_descendants(self):
        profile = self.command_profile()
        run_pipeline(self.task, profile)
        before = self.task.load_state()["stages"]["scene_reconstruction"]["request_sha256"]
        provider = Path(profile["providers"]["scene_reconstruction"]["command"][1])
        provider.write_text(provider.read_text() + "\n# implementation version two\n")
        run_pipeline(self.task, profile, through="scene_reconstruction")
        state = self.task.load_state()
        self.assertEqual(self.execution_count(), 2)
        self.assertNotEqual(before, state["stages"]["scene_reconstruction"]["request_sha256"])
        self.assertEqual(state["stages"]["scene_understanding"]["status"], "stale")
        self.assertNotIn("object_descriptions", self.task.load_artifacts()["artifacts"])
        manifest = engine.read_json(self.task.directory / "stages/scene_reconstruction/stage.json")
        self.assertEqual(manifest["provider"]["source_revision"]["files"][str(provider)], engine.sha256(provider))

    def test_imported_helper_change_invalidates_command_cache(self):
        profile = self.command_profile()
        provider = Path(profile["providers"]["scene_reconstruction"]["command"][1])
        helper = self.root / "helper.py"
        helper.write_text("VALUE = 1\n")
        provider.write_text("import helper\n" + provider.read_text())
        run_pipeline(self.task, profile, through="scene_reconstruction")
        helper.write_text("VALUE = 200\n")
        run_pipeline(self.task, profile, through="scene_reconstruction")
        self.assertEqual(self.execution_count(), 2)

    def test_unrelated_adapter_edit_does_not_invalidate_command_cache(self):
        profile = self.command_profile()
        unrelated = self.root / "orbit.py"
        unrelated.write_text("raise RuntimeError('must not import an unrelated module')\n")
        run_pipeline(self.task, profile, through="scene_reconstruction")
        unrelated.write_text(unrelated.read_text() + "# changed\n")
        run_pipeline(self.task, profile, through="scene_reconstruction")
        self.assertEqual(self.execution_count(), 1)

    def test_deleted_source_preflight_preserves_success_receipt_state_and_artifacts(self):
        profile = self.command_profile()
        run_pipeline(self.task, profile, through="scene_reconstruction")
        paths = [self.task.state_path, self.task.artifacts_path,
                 self.task.directory / "stages/scene_reconstruction/stage.json",
                 self.task.directory / "stages/scene_reconstruction/provider-execution.json"]
        before = {path: path.read_bytes() for path in paths}
        Path(profile["providers"]["scene_reconstruction"]["command"][1]).unlink()
        with patch.object(engine.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(RuntimeError, "revision file is missing"):
                run_pipeline(self.task, profile, through="scene_reconstruction")
            with self.assertRaisesRegex(RuntimeError, "revision file is missing"):
                run_stage(self.task, SPECS["scene_reconstruction"], profile["providers"]["scene_reconstruction"])
            launch.assert_not_called()
        self.assertEqual(before, {path: path.read_bytes() for path in paths})
        self.assertTrue(validate_task(self.task)["valid"])

    def test_invalid_dynamic_entrypoint_preserves_existing_receipt(self):
        profile = self.command_profile()
        run_pipeline(self.task, profile, through="scene_reconstruction")
        manifest = self.task.directory / "stages/scene_reconstruction/stage.json"
        before = manifest.read_bytes()
        with patch.object(engine.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(RuntimeError, "requires nonempty revision_files"):
                run_stage(self.task, SPECS["scene_reconstruction"],
                          {"kind": "command", "command": [sys.executable, "-c", "print('changed')"]})
            launch.assert_not_called()
        self.assertEqual(manifest.read_bytes(), before)
        self.assertEqual(self.execution_count(), 1)

    def test_source_changed_during_execution_is_not_published(self):
        profile = self.command_profile()
        provider = Path(profile["providers"]["scene_reconstruction"]["command"][1])
        provider.write_text(provider.read_text() +
                            "\nsource = pathlib.Path(__file__)\nsource.write_text(source.read_text() + '\\n# changed while running\\n')\n")
        with self.assertRaisesRegex(RuntimeError, "source revision changed"):
            run_pipeline(self.task, profile, through="scene_reconstruction")
        self.assertEqual(self.execution_count(), 1)
        self.assertEqual(self.task.load_state()["stages"]["scene_reconstruction"]["status"], "failed")
        self.assertNotIn("scene_depth", self.task.load_artifacts()["artifacts"])
        execution = engine.read_json(self.task.directory / "stages/scene_reconstruction/provider-execution.json")
        self.assertEqual(execution["returncode"], 0)
        self.assertFalse(execution["outputs_accepted"])
        self.assertIn("source revision changed", execution["error"])

    def test_legacy_receipt_validates_unchanged_but_is_not_current_source_cache(self):
        profile = self.command_profile()
        run_pipeline(self.task, profile, through="scene_reconstruction")
        path = self.task.directory / "stages/scene_reconstruction/stage.json"
        manifest = engine.read_json(path)
        # Represent a receipt produced before source identities existed, only in this fixture.
        manifest["provider"] = profile["providers"]["scene_reconstruction"]
        manifest.pop("command_source_revision")
        manifest.pop("command_provider_revision")
        fingerprint = engine.request_fingerprint(SPECS["scene_reconstruction"], manifest["provider"], manifest["inputs"])
        manifest["request_sha256"] = fingerprint
        write_json(path, manifest)
        state = self.task.load_state()
        state["stages"]["scene_reconstruction"]["request_sha256"] = fingerprint
        write_json(self.task.state_path, state)
        before = path.read_bytes()
        provider = Path(profile["providers"]["scene_reconstruction"]["command"][1])
        provider.write_text(provider.read_text() + "\n# current source is not historical evidence\n")
        self.assertTrue(validate_task(self.task)["valid"])
        self.assertEqual(path.read_bytes(), before)
        self.assertNotIn("source_revision", engine.read_json(path)["provider"])
        run_pipeline(self.task, profile, through="scene_reconstruction")
        self.assertEqual(self.execution_count(), 2)
        self.assertNotEqual(self.task.load_state()["stages"]["scene_reconstruction"]["request_sha256"], fingerprint)

    def test_tampered_input_blocks_resume_and_records_failure(self) -> None:
        profile = self.command_profile()
        run_pipeline(self.task, profile, through="scene_reconstruction")
        source = self.task.load_artifacts()["artifacts"]["source_media"]
        (self.task.directory / source["path"]).write_bytes(b"unregistered edit")
        with self.assertRaisesRegex(RuntimeError, "sha256 mismatch"):
            run_pipeline(self.task, profile, through="scene_reconstruction")
        self.assertEqual(self.execution_count(), 1)
        self.assertEqual(self.task.load_state()["status"], "failed")
        self.assertNotIn("scene_depth", self.task.load_artifacts()["artifacts"])
        manifest = json.loads((self.task.directory / "stages/scene_reconstruction/stage.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertIn("sha256 mismatch", manifest["error"]["message"])

    def test_tampered_cached_output_is_regenerated(self) -> None:
        profile = self.command_profile()
        run_pipeline(self.task, profile, through="scene_reconstruction")
        (self.task.directory / "stages/scene_reconstruction/provider/count.txt").write_text("7")
        run_pipeline(self.task, profile, through="scene_reconstruction")
        self.assertEqual(self.execution_count(), 8)
        self.assertTrue(validate_task(self.task)["valid"])

    def test_upstream_rerun_invalidates_descendants_and_their_artifacts(self) -> None:
        profile = self.command_profile()
        run_pipeline(self.task, profile)
        self.assertEqual(self.task.load_state()["status"], "mixed")
        profile["providers"]["scene_reconstruction"]["revision"] = "rerun"
        run_pipeline(self.task, profile, through="scene_reconstruction")
        state = self.task.load_state()
        self.assertEqual(state["status"], "stale")
        self.assertEqual(state["stages"]["scene_recomposition"]["status"], "stale")
        self.assertNotIn("physical_object_obj", self.task.load_artifacts()["artifacts"])
        self.assertFalse(validate_task(self.task)["valid"])

    def test_real_provider_cannot_consume_contract_inputs(self) -> None:
        profile = load_profile(REPO / "profiles" / "contract.json")
        run_pipeline(self.task, profile, through="scene_reconstruction")
        with self.assertRaisesRegex(RuntimeError, "cannot consume contract_only"):
            run_stage(self.task, SPECS["scene_understanding"], {"kind": "command", "command": ["/usr/bin/true"]})
        self.assertEqual(self.task.load_state()["stages"]["scene_understanding"]["status"], "failed")
        self.assertFalse((self.task.directory / "stages/scene_understanding/provider-execution.json").exists())

    def test_failed_stage_does_not_reuse_old_provider_manifest(self) -> None:
        profile = self.command_profile()
        run_pipeline(self.task, profile, through="scene_reconstruction")
        profile["providers"]["scene_reconstruction"]["command"] = ["/usr/bin/true"]
        with self.assertRaisesRegex(RuntimeError, "provider did not write"):
            run_pipeline(self.task, profile, through="scene_reconstruction")
        self.assertEqual(self.task.load_state()["status"], "failed")
        self.assertEqual(finalize(self.task)["status"], "failed")
        self.assertNotIn("scene_depth", self.task.load_artifacts()["artifacts"])

    def test_launch_failure_has_execution_and_stage_receipts(self) -> None:
        profile = self.command_profile()
        executable = self.root / "not-executable.py"
        executable.write_text("pass\n")
        executable.chmod(0o600)
        profile["providers"]["scene_reconstruction"]["command"] = [str(executable)]
        with self.assertRaisesRegex(RuntimeError, "could not start"):
            run_pipeline(self.task, profile, through="scene_reconstruction")
        execution = json.loads((self.task.directory / "stages/scene_reconstruction/provider-execution.json").read_text())
        self.assertIsNone(execution["returncode"])
        self.assertIn("error", execution)
        self.assertEqual(self.task.load_state()["status"], "failed")

    def test_post_spawn_identity_error_terminates_owned_process(self):
        started = []
        original_popen = subprocess.Popen

        def launch(*args, **kwargs):
            process = original_popen(*args, **kwargs)
            if kwargs.get("start_new_session"):
                started.append(process)
            return process

        provider = {"kind": "command", "command": [sys.executable, "-c", "import time; time.sleep(60)"],
                    "revision_files": [str(self.source)]}
        with patch.object(engine.subprocess, "Popen", side_effect=launch), patch.object(engine, "process_identity", side_effect=OSError("fixture identity error")):
            with self.assertRaisesRegex(RuntimeError, "fixture identity error"):
                run_stage(self.task, SPECS["scene_reconstruction"], provider)
        self.assertEqual(len(started), 1)
        self.assertIsNotNone(started[0].poll())
        stage = self.task.directory / "stages/scene_reconstruction"
        execution = json.loads((stage / "provider-execution.json").read_text())
        self.assertTrue(execution["process_cleanup"]["confirmed_exited"])
        self.assertIn("SIGTERM", execution["process_cleanup"]["signals"])
        self.assertEqual(json.loads((stage / "provider-live.json").read_text())["status"], "failed")

    def test_running_receipt_write_error_terminates_owned_process(self):
        failed_once = False

        def failing_write(path, value):
            nonlocal failed_once
            if path.name == "provider-live.json" and value.get("status") == "running" and not failed_once:
                failed_once = True
                raise OSError("fixture live receipt error")
            write_json(path, value)

        provider = {"kind": "command", "command": [sys.executable, "-c", "import time; time.sleep(60)"],
                    "revision_files": [str(self.source)]}
        with patch.object(engine, "write_json", side_effect=failing_write):
            with self.assertRaisesRegex(RuntimeError, "fixture live receipt error"):
                run_stage(self.task, SPECS["scene_reconstruction"], provider)
        execution = json.loads((self.task.directory / "stages/scene_reconstruction/provider-execution.json").read_text())
        self.assertTrue(execution["process_cleanup"]["confirmed_exited"])
        self.assertIsNotNone(execution["returncode"])

    def test_cleanup_escalates_only_owned_term_ignoring_process(self):
        ready = self.task.directory / "term-ignored.ready"
        code = f"import signal,time,pathlib; signal.signal(signal.SIGTERM,signal.SIG_IGN); pathlib.Path({str(ready)!r}).touch(); time.sleep(60)"
        provider = {"kind": "command", "command": [sys.executable, "-c", code], "revision_files": [str(self.source)]}

        def identity_error(pid):
            deadline = time.monotonic() + 5
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(.01)
            if not ready.exists():
                raise RuntimeError("fixture did not initialize signal handler")
            raise OSError("fixture identity error after handler setup")

        def bounded_cleanup(process, identity):
            return terminate_started_process(process, identity, term_timeout=.1, kill_timeout=2)

        with patch.object(engine, "process_identity", side_effect=identity_error), patch.object(engine, "terminate_started_process", side_effect=bounded_cleanup):
            with self.assertRaisesRegex(RuntimeError, "identity error after handler"):
                run_stage(self.task, SPECS["scene_reconstruction"], provider)
        execution = json.loads((self.task.directory / "stages/scene_reconstruction/provider-execution.json").read_text())
        self.assertEqual(execution["process_cleanup"]["signals"], ["SIGTERM", "SIGKILL"])
        self.assertTrue(execution["process_cleanup"]["confirmed_exited"])
        self.assertEqual(execution["returncode"], -signal.SIGKILL)

    def test_live_orphan_receipt_blocks_launch_without_touching_old_process(self):
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        stage = self.task.directory / "stages/scene_reconstruction"
        try:
            write_json(stage / "provider-live.json", {**process_identity(process.pid), "process_group_id": process.pid,
                                                     "execution_id": "old-job", "status": "running"})
            write_json(stage / "provider-artifacts.json", {"old_artifact": "preserved"})
            with patch.object(engine, "_command_provider") as launch:
                with self.assertRaisesRegex(RuntimeError, "previous provider process group is alive"):
                    run_stage(self.task, SPECS["scene_reconstruction"], {"kind": "command", "command": ["/usr/bin/true"]})
                launch.assert_not_called()
            self.assertIsNone(process.poll())
            self.assertEqual(json.loads((stage / "provider-artifacts.json").read_text()), {"old_artifact": "preserved"})
        finally:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)

    def test_unknown_old_process_identity_blocks_launch(self):
        stage = self.task.directory / "stages/scene_reconstruction"
        write_json(stage / "provider-live.json", {"pid": os.getpid(), "hostname": "another-host", "status": "running"})
        with patch.object(engine, "_command_provider") as launch:
            with self.assertRaisesRegex(RuntimeError, "process group is unknown"):
                run_stage(self.task, SPECS["scene_reconstruction"], {"kind": "command", "command": ["/usr/bin/true"]})
            launch.assert_not_called()

    def test_interrupted_initialization_intent_blocks_duplicate_launch(self):
        stage = self.task.directory / "stages/scene_reconstruction"
        write_json(stage / "provider-live.json", {"execution_id": "interrupted-before-pid", "status": "initializing"})
        with patch.object(engine, "_command_provider") as launch:
            with self.assertRaisesRegex(RuntimeError, "process group is unknown"):
                run_stage(self.task, SPECS["scene_reconstruction"], {"kind": "command", "command": ["/usr/bin/true"]})
            launch.assert_not_called()

    def test_confirmed_exited_old_process_permits_explicit_retry(self):
        process = subprocess.Popen(["/usr/bin/true"], start_new_session=True)
        identity = process_identity(process.pid)
        process.wait(timeout=5)
        stage = self.task.directory / "stages/scene_reconstruction"
        write_json(stage / "provider-live.json", {**identity, "process_group_id": process.pid, "status": "running"})
        profile = self.command_profile()
        run_pipeline(self.task, profile, through="scene_reconstruction")
        self.assertEqual(self.execution_count(), 1)

    def test_unconfirmed_cleanup_receipt_blocks_retry_without_live_file(self):
        stage = self.task.directory / "stages/scene_reconstruction"
        write_json(stage / "provider-execution.json", {"provider_process": process_identity(os.getpid()),
                                                      "process_cleanup": {"confirmed_exited": False}})
        with patch.object(engine, "_command_provider") as launch:
            with self.assertRaisesRegex(RuntimeError, "cleanup is unconfirmed"):
                run_stage(self.task, SPECS["scene_reconstruction"], {"kind": "command", "command": ["/usr/bin/true"]})
            launch.assert_not_called()

    def test_directory_artifact_rejects_nested_file_and_directory_symlinks(self) -> None:
        for directory_link in (False, True):
            with self.subTest(directory_link=directory_link):
                output = self.task.directory / f"nested-{directory_link}"
                output.mkdir()
                outside = self.root / f"outside-{directory_link}"
                if directory_link:
                    outside.mkdir()
                    (outside / "image.png").write_bytes(b"outside")
                else:
                    outside.write_bytes(b"outside")
                (output / "escape").symlink_to(outside, target_is_directory=directory_link)
                artifacts = self.task.load_artifacts()
                artifacts["artifacts"]["unsafe"] = {
                    "path": str(output.relative_to(self.task.directory)), "sha256": "untrusted", "size_bytes": 7,
                }
                write_json(self.task.artifacts_path, artifacts)
                report = validate_task(self.task)
                self.assertFalse(report["valid"])
                self.assertTrue(any("symlink" in issue for issue in report["issues"]))

    def test_false_completed_status_with_mixed_stages_is_rejected(self) -> None:
        profile = self.command_profile()
        state = run_pipeline(self.task, profile)
        self.assertEqual(state["status"], "mixed")
        state["status"] = "completed"
        write_json(self.task.state_path, state)
        report = validate_task(self.task)
        self.assertFalse(report["valid"])
        self.assertTrue(any("every module" in issue for issue in report["issues"]))
