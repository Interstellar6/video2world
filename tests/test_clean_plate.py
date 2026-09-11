from __future__ import annotations

import base64
import copy
import importlib.util
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/clean_plate.py"
SPEC = importlib.util.spec_from_file_location("clean_plate_adapter", SCRIPT)
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)
NUMERICAL_DEPS = all(importlib.util.find_spec(name) is not None for name in ("numpy", "PIL"))


class EndpointTests(unittest.TestCase):
    def test_no_credentials_or_remote_plaintext_endpoint(self):
        for endpoint in ("http://example.com/responses", "https://name:secret@example.com/responses", "https://example.com/responses?api_key=x", "https://example.com/v1"):
            args = adapter.parser().parse_args(["--task-dir", "/tmp/task", "--inputs", "in.json", "--outputs", "out.json", "--endpoint", endpoint, "--controller", "controller", "--model", "image-model"])
            with self.assertRaises(adapter.CleanPlateError):
                adapter.validate_options(args)


@unittest.skipUnless(NUMERICAL_DEPS, "requires numpy and Pillow")
class CleanPlateTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from PIL import Image

        self.temp = tempfile.TemporaryDirectory()
        self.task = Path(self.temp.name)
        frames, masks, observations = [], [], []
        for index in range(3):
            frame_id = str(index)
            source = np.full((32, 48, 3), [100, 130, 170], dtype=np.uint8)
            source[:, :, 0] = np.arange(48)[None] + 75
            mask = np.zeros((32, 48), dtype=np.uint8)
            mask[10:18, 10:18] = 255
            Image.fromarray(source).save(self.task / f"source-{index}.png")
            Image.fromarray(mask).save(self.task / f"mask-{index}.png")
            frames.append({"frame_id": frame_id, "image_path": f"source-{index}.png", "width": 48, "height": 32})
            masks.append({"frame_id": frame_id, "object_id": "accepted", "component_id": "__object__", "image_path": f"source-{index}.png", "mask_path": f"mask-{index}.png"})
            observations.append({"frame_id": frame_id, "image_path": f"source-{index}.png", "mask_path": f"mask-{index}.png"})
        self.frames = {"frames": frames}
        self.components = {"masks": masks, "frames": frames}
        self.tracks = {"tracks": [{"object_id": "accepted", "observations": observations, "association_status": "vlm_identity_hypothesis"}]}
        self.lifting = {"status": "passed", "carve_performed": True, "generated_geometry_used": False, "accepted_object_ids": ["accepted"],
                        "tracks": [{"object_id": "accepted", "accepted": True}], "source_files": {}}
        self.paths = {"frames_manifest": "frames.json", "component_masks": "components.json", "physical_instance_tracks": "tracks.json", "lifting_report": "lifting.json"}
        self.persist()
        self.args = adapter.parser().parse_args(["--task-dir", str(self.task), "--inputs", "inputs.json", "--outputs", "stages/clean_plate/provider-output.json",
                                                 "--endpoint", "https://example.com/v1/responses", "--controller", "controller", "--model", "image-model", "--dilate-pixels", "1", "--max-api-requests", "3"])
        self.env = patch.dict(os.environ, {"PLBBL_API_KEY": "cpu-test-secret-never-persist"})
        self.env.start()
        self.calls = []

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def persist(self, bind=True):
        for role, document in (("frames_manifest", self.frames), ("component_masks", self.components), ("physical_instance_tracks", self.tracks)):
            adapter.write_json(self.task / self.paths[role], document)
        if bind:
            self.lifting["source_files"] = {role: {"path": self.paths[role], "sha256": adapter.sha256(self.task / self.paths[role])} for role in ("component_masks", "physical_instance_tracks")}
        adapter.write_json(self.task / self.paths["lifting_report"], self.lifting)
        adapter.write_json(self.task / "inputs.json", {"inputs": {role: {"path": path, "evidence": "observed" if role == "frames_manifest" else "derived"} for role, path in self.paths.items()}})

    def transport(self, endpoint, token, request, timeout, request_id):
        import numpy as np
        from PIL import Image

        self.calls.append(copy.deepcopy(request))
        images = [item for item in request["input"][0]["content"] if item["type"] == "input_image"]
        with Image.open(BytesIO(base64.b64decode(images[0]["image_url"].split(",", 1)[1]))) as source:
            pixels = np.asarray(source.convert("RGB")).copy()
        with Image.open(BytesIO(base64.b64decode(images[1]["image_url"].split(",", 1)[1]))) as mask:
            pixels[np.asarray(mask) > 0] = [140, 90, 120]
        payload = BytesIO()
        Image.fromarray(pixels).save(payload, format="PNG")
        response = {"status": "completed", "output": [{"type": "image_generation_call", "result": base64.b64encode(payload.getvalue()).decode()}]}
        return 200, {"x-request-id": "cpu-fixture"}, json.dumps(response).encode()

    def report(self):
        reports = list((self.task / "stages/clean_plate/runs").glob("*/clean_plate_report.json"))
        self.assertEqual(len(reports), 1)
        return adapter.read_json(reports[0])

    def test_full_abi_crossview_anchor_raw_isolation_and_exact_outside(self):
        import numpy as np
        from PIL import Image

        result = adapter.run(self.args, self.transport)
        self.assertEqual(result["frames"], 3)
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(len(self.calls[0]["input"][0]["content"]), 3)
        self.assertEqual(len(self.calls[1]["input"][0]["content"]), 4)
        self.assertEqual(self.calls[0]["tools"][0]["action"], "edit")
        self.assertFalse(self.calls[0]["store"])
        envelope = adapter.read_json(self.task / self.args.outputs)
        self.assertEqual(set(envelope["outputs"]), {"clean_plate_frames", "clean_plate_report"})
        frames = adapter.read_json(self.task / envelope["outputs"]["clean_plate_frames"]["path"])
        self.assertEqual(frames["evidence"], "generated")
        self.assertEqual(frames["camera_policy"], "fixed_source_camera")
        for frame in frames["frames"]:
            receipt = adapter.read_json(self.task / frame["receipt_path"])
            self.assertFalse(receipt["promotion_allowed"])
            self.assertFalse(receipt["collision_eligible"])
            self.assertEqual(receipt["qa"]["visual_review"], "pending")
            self.assertIn("raw_generated", receipt["artifacts"])
            self.assertIn("raw_response", receipt["artifacts"])
            with Image.open(self.task / frame["mask_path"]) as mask:
                outside = np.asarray(mask) == 0
            old = np.asarray(adapter.rgb(self.task / frame["source_image_path"]))
            final = np.asarray(adapter.rgb(self.task / frame["image_path"]))
            self.assertTrue(np.array_equal(old[outside], final[outside]))
            self.assertGreater(receipt["qa"]["changed_mask_pixels"], 0)
        for path in (self.task / "stages/clean_plate").rglob("*.json"):
            self.assertNotIn("cpu-test-secret-never-persist", path.read_text())

    def test_successful_resume_makes_zero_new_requests_and_needs_no_token(self):
        adapter.run(self.args, self.transport)
        os.environ.pop("PLBBL_API_KEY")
        adapter.run(self.args, self.transport)
        self.assertEqual(len(self.calls), 3)

    def test_default_budget_is_zero(self):
        self.args.max_api_requests = 0
        with self.assertRaisesRegex(adapter.CleanPlateError, "budget exhausted"):
            adapter.run(self.args, self.transport)
        self.assertEqual(self.calls, [])
        self.assertFalse((self.task / self.args.outputs).exists())
        self.assertEqual(self.report()["status"], "blocked")

    def test_partial_budget_can_continue_without_charging_completed_frames(self):
        self.args.max_api_requests = 1
        with self.assertRaisesRegex(adapter.CleanPlateError, "budget exhausted"):
            adapter.run(self.args, self.transport)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.report()["completed_frame_count"], 1)
        self.assertFalse((self.task / self.args.outputs).exists())
        self.args.max_api_requests = 3
        adapter.run(self.args, self.transport)
        self.assertEqual(len(self.calls), 3)

    def test_timeout_persists_ambiguous_state_and_never_retries(self):
        attempts = []
        def timeout(*args):
            attempts.append(1)
            raise TimeoutError("deliberately ambiguous")
        with self.assertRaisesRegex(adapter.CleanPlateError, "unknown"):
            adapter.run(self.args, timeout)
        with self.assertRaisesRegex(adapter.CleanPlateError, "automatic resubmission is forbidden"):
            adapter.run(self.args, timeout)
        self.assertEqual(len(attempts), 1)
        journal = adapter.read_json(self.task / "stages/clean_plate/api_journal.json")
        self.assertEqual(journal["requests"][0]["state"], "submitted_outcome_unknown")

    def test_only_explicit_failed_request_id_can_retry_with_additional_budget(self):
        def timeout(*args):
            raise TimeoutError()
        with self.assertRaises(adapter.CleanPlateError):
            adapter.run(self.args, timeout)
        journal_path = self.task / "stages/clean_plate/api_journal.json"
        original = adapter.read_json(journal_path)["requests"][0]
        self.args.retry_request_id = [original["request_id"]]
        self.args.max_api_requests = 4
        adapter.run(self.args, self.transport)
        records = adapter.read_json(journal_path)["requests"]
        self.assertEqual(len(records), 4)
        self.assertEqual(records[0]["state"], "submitted_outcome_unknown")
        self.assertEqual(records[1]["explicit_retry_of_request_id"], original["request_id"])

    def test_real_http_transport_uses_responses_without_implicit_retries(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        adapter_test = self
        paths = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                paths.append(self.path)
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                status, headers, body = adapter_test.transport("local", "test", request, 1, "test")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            self.args.endpoint = f"http://127.0.0.1:{server.server_port}/v1/responses"
            adapter.run(self.args)
            self.assertEqual(paths, ["/v1/responses"] * 3)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)

    def test_http_failure_is_not_retried_and_credentials_are_redacted(self):
        attempts = []
        def failure(endpoint, token, *args):
            attempts.append(1)
            return 401, {"X-Request-Id": token}, json.dumps({"error": {"type": "unauthorized", "message": token}}).encode()
        with self.assertRaisesRegex(adapter.CleanPlateError, "HTTP 401"):
            adapter.run(self.args, failure)
        with self.assertRaisesRegex(adapter.CleanPlateError, "automatic resubmission"):
            adapter.run(self.args, failure)
        self.assertEqual(len(attempts), 1)
        for path in (self.task / "stages/clean_plate").rglob("*.json"):
            self.assertNotIn("cpu-test-secret-never-persist", path.read_text())

    def test_invalid_success_response_does_not_trigger_another_charge(self):
        attempts = []
        def malformed(*args):
            attempts.append(1)
            return 200, {}, b"not-json"
        with self.assertRaisesRegex(adapter.CleanPlateError, "invalid JSON"):
            adapter.run(self.args, malformed)
        with self.assertRaises(ValueError):
            adapter.run(self.args, malformed)
        self.assertEqual(len(attempts), 1)

    def test_task_lock_rejects_a_concurrent_worker_before_requests(self):
        stage = self.task / "stages/clean_plate"
        stage.mkdir(parents=True)
        with adapter.stage_lock(self.task, stage):
            with self.assertRaisesRegex(adapter.CleanPlateError, "owns the task lock"):
                adapter.run(self.args, self.transport)
        self.assertEqual(self.calls, [])

    def test_inputs_changed_during_api_processing_are_not_published(self):
        from PIL import Image

        def mutating(*args):
            response = self.transport(*args)
            if len(self.calls) == 3:
                Image.new("RGB", (48, 32), (10, 20, 30)).save(self.task / "source-0.png")
            return response
        with self.assertRaisesRegex(adapter.CleanPlateError, "hash mismatch"):
            adapter.run(self.args, mutating)
        self.assertFalse((self.task / self.args.outputs).exists())
        self.assertEqual(self.report()["status"], "blocked")

    def test_rejected_track_is_protected_and_only_accepted_masks_dilate(self):
        import numpy as np
        from PIL import Image

        protected = np.zeros((32, 48), dtype=np.uint8)
        protected[10:18, 18:23] = 255
        Image.fromarray(protected).save(self.task / "rejected.png")
        self.components["masks"].append({"frame_id": "0", "object_id": "rejected", "component_id": "__object__", "mask_path": "rejected.png"})
        self.lifting["tracks"].append({"object_id": "rejected", "accepted": False})
        self.persist()
        jobs, _, accepted = adapter.load_plan(self.args)
        self.assertEqual(accepted, ["accepted"])
        self.assertEqual(jobs[0]["protected_dilation_pixels"], 8)
        self.assertFalse(jobs[0]["mask"][protected > 0].any())
        self.assertGreater(jobs[0]["masked_pixels"], jobs[0]["parent_pixels"])

    def test_small_boundary_overlap_is_resolved_but_large_overlap_fails(self):
        import numpy as np
        from PIL import Image

        protected = np.zeros((32, 48), dtype=np.uint8)
        protected[17:18, 17:18] = 255
        Image.fromarray(protected).save(self.task / "rejected.png")
        self.components["masks"].append({"frame_id": "0", "object_id": "rejected", "component_id": "__object__", "mask_path": "rejected.png"})
        self.lifting["tracks"].append({"object_id": "rejected", "accepted": False})
        self.persist()
        self.args.max_ownership_overlap_fraction = 0.05
        jobs, _, _ = adapter.load_plan(self.args)
        self.assertEqual(jobs[0]["protected_ownership_pixels"], 1)
        self.assertFalse(jobs[0]["mask"][protected > 0].any())
        protected[:] = 0
        protected[10:18, 10:18] = 255
        Image.fromarray(protected).save(self.task / "rejected.png")
        self.persist()
        with self.assertRaisesRegex(adapter.CleanPlateError, "resolve ownership"):
            adapter.load_plan(self.args)

    def test_unsegmented_original_frames_are_not_background_training_inputs(self):
        self.tracks["tracks"][0]["observations"].pop()
        self.components["masks"].pop()
        self.persist()
        adapter.run(self.args, self.transport)
        report = self.report()
        self.assertEqual(report["excluded_without_accepted_removal_mask"], ["2"])
        self.assertEqual(report["total_source_frames"], 3)
        self.assertEqual(report["completed_frame_count"], 2)

    def test_lifting_hash_and_acceptance_are_required_before_api(self):
        self.components["masks"][0]["score"] = 0.5
        self.persist(bind=False)
        with self.assertRaisesRegex(adapter.CleanPlateError, "does not bind current component_masks"):
            adapter.run(self.args, self.transport)
        self.persist()
        self.lifting["tracks"][0]["accepted"] = False
        self.persist()
        with self.assertRaisesRegex(adapter.CleanPlateError, "declarations disagree"):
            adapter.run(self.args, self.transport)
        self.assertEqual(self.calls, [])

    def test_lifting_resolved_output_binding_is_authoritative(self):
        self.lifting["resolved_files"] = copy.deepcopy(self.lifting["source_files"])
        self.lifting["source_files"] = {"component_mask_candidates": {"path": "earlier-candidates.json", "sha256": "old-input"}}
        self.persist(bind=False)
        adapter.run(self.args, self.transport)
        self.assertEqual(len(self.calls), 3)

    def test_stale_resolved_binding_cannot_fall_back_to_valid_raw_binding(self):
        self.lifting["resolved_files"] = copy.deepcopy(self.lifting["source_files"])
        self.lifting["resolved_files"]["component_masks"]["sha256"] = "stale"
        self.persist()
        with self.assertRaisesRegex(adapter.CleanPlateError, "does not bind current component_masks"):
            adapter.run(self.args, self.transport)
        self.assertEqual(self.calls, [])

    def test_raw_aspect_ratio_changes_and_global_drift_are_rejected(self):
        import numpy as np
        from PIL import Image

        source, mask, output = self.task / "source-0.png", self.task / "mask-0.png", self.task / "composite.png"
        raw = self.task / "raw.png"
        Image.new("RGB", (48, 48), (100, 130, 170)).save(raw)
        with self.assertRaisesRegex(adapter.CleanPlateError, "aspect ratio"):
            adapter.composite(self.args, raw, source, mask, output)
        Image.fromarray(np.zeros((32, 48, 3), dtype=np.uint8)).save(raw)
        with self.assertRaisesRegex(adapter.CleanPlateError, "drifted outside"):
            adapter.composite(self.args, raw, source, mask, output)
        self.assertFalse(output.exists())

    def test_corrupt_cached_output_does_not_make_new_request(self):
        adapter.run(self.args, self.transport)
        frame = self.report()["frames"][0]
        (self.task / frame["image_path"]).write_bytes(b"corrupted")
        with self.assertRaisesRegex(adapter.CleanPlateError, "cached clean-plate artifact hash mismatch"):
            adapter.run(self.args, self.transport)
        self.assertEqual(len(self.calls), 3)

    def test_task_symlink_escape_is_rejected_before_api(self):
        with tempfile.TemporaryDirectory() as other:
            external = Path(other) / "source.png"
            external.write_bytes((self.task / "source-0.png").read_bytes())
            (self.task / "escape.png").symlink_to(external)
            self.frames["frames"][0]["image_path"] = "escape.png"
            self.persist()
            with self.assertRaisesRegex(ValueError, "escapes task"):
                adapter.run(self.args, self.transport)
        self.assertEqual(self.calls, [])

    def test_background_contract_accepts_generated_candidate_rasters(self):
        import numpy as np

        background_spec = importlib.util.spec_from_file_location("clean_plate_background_contract", SCRIPT.with_name("background_reconstruction.py"))
        background = importlib.util.module_from_spec(background_spec)
        background_spec.loader.exec_module(background)
        adapter.run(self.args, self.transport)
        outputs = adapter.read_json(self.task / self.args.outputs)["outputs"]
        frames = adapter.read_json(self.task / outputs["clean_plate_frames"]["path"])
        cameras = []
        for index, frame in enumerate(self.frames["frames"]):
            extrinsic = np.eye(4)
            extrinsic[0, 3] = index * 0.1
            cameras.append({**frame, "intrinsics": [[30, 0, 24], [0, 30, 16], [0, 0, 1]], "world_to_camera": extrinsic.tolist()})
        records, checks = background.validate_clean_frames(self.task, frames, {"frames": cameras}, 0, 0)
        self.assertEqual(len(records), 3)
        self.assertTrue(all(item["outside_mask_mae"] == 0 for item in checks))


if __name__ == "__main__":
    unittest.main()
