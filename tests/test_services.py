from __future__ import annotations

import json
import hashlib
import io
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
import uuid
from unittest.mock import patch
from urllib.error import URLError

from world_modeling.contracts import ModuleSpec, role
from world_modeling.engine import PipelineError, create_task, exclusive_lock, load_profile, run_stage, validate_task, write_json
from world_modeling.modules import SPECS
from world_modeling.registry import ModuleRegistry
import world_modeling.services as services
from world_modeling.services import PROTOCOL, ModuleService, discover_service, make_server, process_identity, process_state, recover_job, register_service, request_json, resolve_provider


def submit_in_process(root, provider, payload, start, replies):
    worker = ModuleService(Path(root), SPECS["scene_reconstruction"], provider)

    def slow_write(path, value):
        if value.get("status") == "queued":
            time.sleep(.1)
        write_json(path, value)

    try:
        replies.put({"ready": True})
        if not start.wait(10):
            raise RuntimeError("fixture start timed out")
        with patch("world_modeling.engine.write_json", side_effect=slow_write):
            result = worker.submit(payload)
            worker.executor.shutdown(wait=True)
        replies.put({"job_id": result["job_id"]})
    except Exception as error:
        replies.put({"error": str(error)})
    finally:
        worker.executor.shutdown(wait=True)


class ModuleRegistrationTests(unittest.TestCase):
    def test_add_discover_remove_and_dependency_order_without_pipeline_edits(self):
        first = ModuleSpec("first", "First", ("source_media",), (role("one", "derived", "application/json"),), "cpu", "")
        second = ModuleSpec("second", "Second", ("one",), (role("two", "derived", "application/json"),), "cpu", "")
        registry = ModuleRegistry([second, first])
        self.assertEqual([item.id for item in registry.ordered(["second", "first"])], ["first", "second"])
        self.assertEqual(registry.discover()[0]["id"], "second")
        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register(first)
        registry.unregister("first")
        with self.assertRaisesRegex(ValueError, "missing producers"):
            registry.ordered(["second"])


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        source = self.root / "input.txt"
        source.write_text("fixture source")
        self.task = create_task(self.root, "service-fixture", source)
        script = self.root / "fixture_provider.py"
        script.write_text(
            "import json, pathlib, sys, time\n"
            "out=pathlib.Path(sys.argv[1]); out.parent.mkdir(parents=True,exist_ok=True)\n"
            "counter=out.parent/'count.txt'\n"
            "counter.write_text(str(int(counter.read_text())+1 if counter.exists() else 1))\n"
            "time.sleep(.05)\n"
            "roles=('frames_manifest','cameras','scene_depth','scene_gaussian_ply','scene_tsdf_mesh')\n"
            "out.write_text(json.dumps({'outputs':{r:{'path':str(counter.relative_to(out.parents[2])),'evidence':'observed'} for r in roles}}))\n"
        )
        self.provider = {"kind": "command", "command": [sys.executable, str(script), "{outputs_json}"]}
        self.service = ModuleService(self.root, SPECS["scene_reconstruction"], self.provider)
        self.server = make_server(self.service)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.endpoint = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.service.executor.shutdown(wait=True)
        self.temp.cleanup()

    def payload(self):
        return {"protocol": PROTOCOL, "module_id": "scene_reconstruction", "task_id": self.task.task_id,
                "request_id": uuid.uuid4().hex, "provider_revision": self.service.revision,
                "inputs": {"source_media": self.task.load_artifacts()["artifacts"]["source_media"]}}

    def dead_identity(self):
        identity = process_identity(os.getpid())
        field = "proc_start_ticks" if "proc_start_ticks" in identity else "process_started_at"
        identity[field] = "previous-process:" + str(identity[field])
        return identity

    def interrupted_job(self, status="queued", worker=None, provider=None):
        payload = self.payload()
        path = self.service.job_path(self.task.task_id, "scene_reconstruction", payload["request_id"])
        write_json(path, {"protocol": PROTOCOL, "job_id": payload["request_id"], "status": status,
            "worker_instance": "previous-worker", "worker_process": worker or self.dead_identity(),
            "request_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()})
        if provider is not None:
            write_json(path.parent.parent / "provider-live.json", {**provider, "execution_id": payload["request_id"], "status": "running"})
        return payload

    def test_live_discovery_registration_and_pipeline_invocation(self):
        registry = self.root / "services.json"
        self.assertEqual(register_service(registry, self.endpoint)["registered"], ["scene_reconstruction"])
        profile_path = self.root / "profile.json"
        write_json(profile_path, {"schema_version": "1.0", "registry": "services.json",
                                 "providers": {"scene_reconstruction": {"kind": "service", "poll_seconds": .1}}})
        profile = load_profile(profile_path)
        result = run_stage(self.task, SPECS["scene_reconstruction"], profile["providers"]["scene_reconstruction"])
        self.assertEqual(result["status"], "executed")
        self.assertTrue(validate_task(self.task)["valid"])
        stage = json.loads((self.task.directory / "stages/scene_reconstruction/stage.json").read_text())
        self.assertEqual(stage["mode"], "service")
        self.assertEqual(stage["service_job"]["status"], "completed")

    def test_post_retry_does_not_execute_twice(self):
        payload = self.payload()
        first = request_json(self.endpoint, "/v1/jobs", payload=payload)
        second = request_json(self.endpoint, "/v1/jobs", payload=payload)
        self.assertEqual(first["job_id"], second["job_id"])
        self.service.executor.shutdown(wait=True)
        self.assertEqual((self.task.directory / "stages/scene_reconstruction/count.txt").read_text(), "1")
        completed = self.service.status(self.task.task_id, "scene_reconstruction", payload["request_id"])
        self.assertEqual(completed["status"], "completed")

    def test_lost_post_response_reuses_persisted_request_id(self):
        config = resolve_provider("scene_reconstruction", {"endpoint": self.endpoint, "poll_seconds": .1})
        sent = []

        def disconnect(endpoint, path, token_env=None, payload=None):
            result = request_json(endpoint, path, token_env, payload)
            if path == "/v1/jobs" and payload is not None:
                sent.append(payload["request_id"])
                if len(sent) == 1:
                    raise URLError("fixture response lost after server accepted POST")
            return result

        with patch.object(services, "request_json", side_effect=disconnect):
            with self.assertRaisesRegex(PipelineError, "response lost"):
                run_stage(self.task, SPECS["scene_reconstruction"], config)
            self.assertEqual(run_stage(self.task, SPECS["scene_reconstruction"], config)["status"], "executed")
        self.assertEqual(len(sent), 2)
        self.assertEqual(sent[0], sent[1])
        self.assertEqual((self.task.directory / "stages/scene_reconstruction/count.txt").read_text(), "1")

    def test_two_workers_claim_one_request_once(self):
        other = ModuleService(self.root, SPECS["scene_reconstruction"], self.provider)
        payload = self.payload()
        start = threading.Barrier(2)

        def slow_queued_write(path, value):
            if value.get("status") == "queued":
                time.sleep(.1)
            write_json(path, value)

        def submit(worker):
            start.wait(timeout=5)
            return worker.submit(payload)

        try:
            with patch("world_modeling.engine.write_json", side_effect=slow_queued_write), ThreadPoolExecutor(max_workers=2) as pool:
                replies = list(pool.map(submit, (self.service, other)))
            self.assertEqual(replies[0]["job_id"], replies[1]["job_id"])
            self.assertNotIn("interrupted", [item["status"] for item in replies])
            self.service.executor.shutdown(wait=True)
            other.executor.shutdown(wait=True)
            self.assertEqual((self.task.directory / "stages/scene_reconstruction/count.txt").read_text(), "1")
        finally:
            other.executor.shutdown(wait=True)

    def test_two_worker_processes_share_request_claim(self):
        context = multiprocessing.get_context("spawn")
        start, replies = context.Event(), context.Queue()
        payload = self.payload()
        workers = [context.Process(target=submit_in_process, args=(str(self.root), self.provider, payload, start, replies)) for _ in range(2)]
        try:
            for worker in workers:
                worker.start()
            for _ in workers:
                self.assertTrue(replies.get(timeout=15).get("ready"))
            start.set()
            results = [replies.get(timeout=15) for _ in workers]
            self.assertEqual(results, [{"job_id": payload["request_id"]}] * 2)
            for worker in workers:
                worker.join(timeout=15)
                self.assertEqual(worker.exitcode, 0)
            self.assertEqual((self.task.directory / "stages/scene_reconstruction/count.txt").read_text(), "1")
        finally:
            start.set()
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
                worker.join(timeout=5)
            replies.close()

    def test_changed_adapter_requires_restart_before_discovery_or_submit(self):
        payload = self.payload()
        script = Path(self.provider["command"][1])
        script.write_text(script.read_text() + "\n# changed adapter\n")
        with self.assertRaisesRegex(ValueError, "restart worker"):
            self.service.discovery()
        with self.assertRaisesRegex(ValueError, "restart worker"):
            self.service.submit(payload)
        replacement = ModuleService(self.root, SPECS["scene_reconstruction"], self.provider)
        try:
            self.assertNotEqual(replacement.revision, self.service.revision)
        finally:
            replacement.executor.shutdown()

    def test_explicit_revision_files_and_provider_io_are_hashed(self):
        dependency = self.root / "model-revision.json"
        dependency.write_text("revision-one")
        worker = ModuleService(self.root, SPECS["scene_reconstruction"], {**self.provider, "revision_files": [str(dependency)]})
        try:
            dependency.write_text("revision-two")
            with self.assertRaisesRegex(ValueError, "restart worker"):
                worker.discovery()
        finally:
            worker.executor.shutdown()
        original_open = Path.open
        core = Path(services.__file__).parent / "provider_io.py"

        def changed_core(path, *args, **kwargs):
            return io.BytesIO(b"changed core dependency") if path == core else original_open(path, *args, **kwargs)

        with patch.object(Path, "open", changed_core):
            with self.assertRaisesRegex(ValueError, "restart worker"):
                self.service.discovery()

    def test_control_core_changes_still_require_worker_restart(self):
        original_open = Path.open
        for filename in ("engine.py", "services.py", "processes.py"):
            core = Path(services.__file__).parent / filename

            def changed_core(path, *args, **kwargs):
                return io.BytesIO(b"# modified control core\n") if path == core else original_open(path, *args, **kwargs)

            with self.subTest(filename=filename), patch.object(Path, "open", changed_core):
                with self.assertRaisesRegex(ValueError, "restart worker"):
                    self.service.discovery()

    def test_deleted_adapter_requires_restart_without_starting_a_job(self):
        payload = self.payload()
        Path(self.provider["command"][1]).unlink()
        with self.assertRaisesRegex(ValueError, "restart worker"):
            self.service.discovery()
        with self.assertRaisesRegex(ValueError, "restart worker"):
            self.service.submit(payload)
        self.assertFalse((self.task.directory / "stages/scene_reconstruction/service-jobs").exists())

    def test_imported_helper_changes_require_worker_restart(self):
        helper = self.root / "helper.py"
        helper.write_text("VALUE = 1\n")
        script = Path(self.provider["command"][1])
        script.write_text("import helper\n" + script.read_text())
        replacement = ModuleService(self.root, SPECS["scene_reconstruction"], self.provider)
        try:
            helper.write_text("VALUE = 2\n")
            with self.assertRaisesRegex(ValueError, "restart worker"):
                replacement.discovery()
        finally:
            replacement.executor.shutdown()

    def test_running_job_recovery_requires_exited_provider_identity(self):
        payload = self.interrupted_job("running", provider=process_identity(os.getpid()))
        self.assertEqual(self.service.status(self.task.task_id, "scene_reconstruction", payload["request_id"])["status"], "interrupted")
        with self.assertRaisesRegex(ValueError, "provider is alive"):
            recover_job(self.endpoint, self.task.task_id, "scene_reconstruction", payload["request_id"])
        live = self.task.directory / "stages/scene_reconstruction/provider-live.json"
        write_json(live, {**self.dead_identity(), "execution_id": payload["request_id"], "status": "running"})
        result = recover_job(self.endpoint, self.task.task_id, "scene_reconstruction", payload["request_id"])
        self.assertEqual(result["status"], "failed")
        self.assertIn("recovery", result)
        self.assertEqual(recover_job(self.endpoint, self.task.task_id, "scene_reconstruction", payload["request_id"]), result)

    def test_active_or_unverifiable_worker_cannot_be_recovered(self):
        for worker in (process_identity(os.getpid()), {**self.dead_identity(), "hostname": "different-worker-host"}):
            payload = self.interrupted_job(worker=worker)
            with self.assertRaisesRegex(ValueError, "original worker is alive or its identity"):
                recover_job(self.endpoint, self.task.task_id, "scene_reconstruction", payload["request_id"])

    def test_running_job_without_provider_receipt_cannot_be_recovered(self):
        payload = self.interrupted_job("running")
        with self.assertRaisesRegex(ValueError, "provider process receipt is missing"):
            recover_job(self.endpoint, self.task.task_id, "scene_reconstruction", payload["request_id"])

    def test_active_provider_lock_blocks_recovery(self):
        payload = self.interrupted_job()
        lock = self.task.directory / "stages/scene_reconstruction/provider.lock"
        with exclusive_lock(lock):
            with self.assertRaises(ValueError):
                recover_job(self.endpoint, self.task.task_id, "scene_reconstruction", payload["request_id"])

    def test_surviving_provider_process_group_blocks_recovery(self):
        payload = self.interrupted_job("running", provider=self.dead_identity())
        with patch.object(services, "process_state", return_value="exited"), patch.object(services.os, "kill", side_effect=ProcessLookupError), patch.object(services.os, "killpg", return_value=None):
            with self.assertRaisesRegex(ValueError, "provider is alive"):
                self.service.recover_job(self.task.task_id, "scene_reconstruction", payload["request_id"])

    def test_explicit_recovery_allows_new_request_without_reusing_interrupted_job(self):
        payload = self.interrupted_job()
        config = resolve_provider("scene_reconstruction", {"endpoint": self.endpoint, "poll_seconds": .1})
        fingerprint = hashlib.sha256(json.dumps({"inputs": payload["inputs"], "provider": config}, sort_keys=True).encode()).hexdigest()
        receipt = self.task.directory / "stages/scene_reconstruction/service-request.json"
        write_json(receipt, {"request_id": payload["request_id"], "fingerprint": fingerprint, "status": "interrupted"})
        with self.assertRaisesRegex(PipelineError, "interrupted"):
            run_stage(self.task, SPECS["scene_reconstruction"], config)
        recover_job(self.endpoint, self.task.task_id, "scene_reconstruction", payload["request_id"])
        self.assertEqual(run_stage(self.task, SPECS["scene_reconstruction"], config)["status"], "executed")
        self.assertNotEqual(json.loads(receipt.read_text())["request_id"], payload["request_id"])
        self.assertEqual((self.task.directory / "stages/scene_reconstruction/count.txt").read_text(), "1")

    def test_process_identity_distinguishes_live_and_reused_pid(self):
        self.assertEqual(process_state(process_identity(os.getpid())), "alive")
        self.assertEqual(process_state(self.dead_identity()), "exited")
        self.assertEqual(process_state({"pid": os.getpid(), "hostname": "another-host"}), "unknown")

    def test_request_cannot_escape_task_store_or_replace_input(self):
        for modify in (lambda p: p.update(task_id="../outside"), lambda p: p["inputs"]["source_media"].update(sha256="bad")):
            payload = self.payload()
            modify(payload)
            with self.assertRaisesRegex(ValueError, "HTTP 400"):
                request_json(self.endpoint, "/v1/jobs", payload=payload)

    def test_contract_and_provider_revision_are_checked_before_invocation(self):
        payload = self.payload()
        payload["provider_revision"] = "changed"
        with self.assertRaisesRegex(ValueError, "revision mismatch"):
            request_json(self.endpoint, "/v1/jobs", payload=payload)
        self.assertEqual(discover_service(self.endpoint)["modules"][0]["id"], "scene_reconstruction")

    def test_credentials_are_required_when_worker_has_token(self):
        secured = make_server(self.service, token="test-secret")
        worker = threading.Thread(target=secured.serve_forever, daemon=True)
        worker.start()
        endpoint = f"http://127.0.0.1:{secured.server_port}"
        try:
            with self.assertRaisesRegex(ValueError, "HTTP 401"):
                discover_service(endpoint)
            os.environ["WORLD_MODELING_TEST_TOKEN"] = "test-secret"
            self.assertEqual(discover_service(endpoint, "WORLD_MODELING_TEST_TOKEN")["protocol"], PROTOCOL)
        finally:
            os.environ.pop("WORLD_MODELING_TEST_TOKEN", None)
            secured.shutdown()
            secured.server_close()
            worker.join()


if __name__ == "__main__":
    unittest.main()
