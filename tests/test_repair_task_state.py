from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/repair_task_state.py"
SPEC = importlib.util.spec_from_file_location("repair_task_state_adapter", SCRIPT)
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


def build_task(folder: Path) -> adapter.Task:
    """A task whose stage manifest proves an artifact the index has forgotten."""
    task = adapter.Task(folder, "scene")
    (task.directory / "stages/scene_reconstruction").mkdir(parents=True)
    (task.directory / "inputs").mkdir(parents=True)
    source = task.directory / "inputs/media"
    source.write_bytes(b"calibrated source")
    artifact = task.directory / "stages/scene_reconstruction/run-1/point_cloud.ply"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"ply\n" + b"gaussian" * 10)
    record = {"path": "stages/scene_reconstruction/run-1/point_cloud.ply",
              "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(), "size_bytes": artifact.stat().st_size,
              "role": "scene_gaussian_ply", "evidence": "observed", "producer": "command:scene_reconstruction"}
    manifest = {"schema_version": "1.0", "module": {"id": "scene_reconstruction"}, "provider": {"kind": "command"},
                "inputs": {}, "outputs": {"scene_gaussian_ply": record}, "status": "executed",
                "request_sha256": "a" * 64, "manifest": "stages/scene_reconstruction/stage.json"}
    adapter.write_json(task.directory / "stages/scene_reconstruction/stage.json", manifest)
    adapter.write_json(task.state_path, {"schema_version": "1.0", "task_id": "scene", "pipeline": ["scene_reconstruction"],
                                         "stages": {"scene_reconstruction": {"status": "running",
                                                                             "started_at": "2026-09-13T05:27:03+00:00"}},
                                         "status": "running", "promotion_allowed": False})
    adapter.write_json(task.artifacts_path, {"schema_version": "1.0", "artifacts": {}})
    return task


class RepairTests(unittest.TestCase):
    def test_a_manifest_restores_the_stage_record_and_its_artifact(self):
        with tempfile.TemporaryDirectory() as folder:
            task = build_task(Path(folder))
            report = adapter.run(task, dry_run=False)
            self.assertEqual(report["restored_stages"], ["scene_reconstruction"])
            self.assertEqual(report["restored_roles"], ["scene_gaussian_ply"])
            self.assertEqual(report["skipped"], [])
            state = json.loads(task.state_path.read_text())
            self.assertEqual(state["stages"]["scene_reconstruction"]["status"], "executed")
            self.assertEqual(state["stages"]["scene_reconstruction"]["manifest"],
                             "stages/scene_reconstruction/stage.json")
            self.assertEqual(state["stages"]["scene_reconstruction"]["request_sha256"], "a" * 64)
            artifacts = json.loads(task.artifacts_path.read_text())["artifacts"]
            self.assertEqual(artifacts["scene_gaussian_ply"]["path"],
                             "stages/scene_reconstruction/run-1/point_cloud.ply")

    def test_a_dry_run_reports_without_touching_the_task(self):
        with tempfile.TemporaryDirectory() as folder:
            task = build_task(Path(folder))
            before = task.state_path.read_text()
            report = adapter.run(task, dry_run=True)
            self.assertTrue(report["dry_run"])
            self.assertEqual(report["restored_stages"], ["scene_reconstruction"])
            self.assertEqual(task.state_path.read_text(), before)
            self.assertEqual(json.loads(task.artifacts_path.read_text())["artifacts"], {})

    def test_a_manifest_whose_output_changed_is_skipped(self):
        with tempfile.TemporaryDirectory() as folder:
            task = build_task(Path(folder))
            artifact = task.directory / "stages/scene_reconstruction/run-1/point_cloud.ply"
            artifact.write_bytes(b"ply\nsomething else entirely")
            report = adapter.run(task, dry_run=False)
            self.assertEqual(report["restored_stages"], [])
            self.assertEqual(report["restored_roles"], [])
            self.assertEqual(len(report["skipped"]), 1)
            self.assertEqual(report["skipped"][0]["module"], "scene_reconstruction")
            self.assertTrue(report["skipped"][0]["reason"].startswith("output scene_gaussian_ply does not verify:"),
                            report["skipped"][0]["reason"])
            self.assertIn("scene_gaussian_ply", report["skipped"][0]["reason"])
            self.assertEqual(json.loads(task.artifacts_path.read_text())["artifacts"], {})

    def test_a_failed_or_missing_manifest_is_not_adopted(self):
        with tempfile.TemporaryDirectory() as folder:
            task = build_task(Path(folder))
            manifest_path = task.directory / "stages/scene_reconstruction/stage.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["status"] = "failed"
            manifest_path.write_text(json.dumps(manifest))
            report = adapter.run(task, dry_run=False)
            self.assertEqual(report["skipped"], [{"module": "scene_reconstruction", "reason": "manifest status is 'failed'"}])
            manifest_path.unlink()
            report = adapter.run(task, dry_run=False)
            self.assertEqual(report["skipped"], [{"module": "scene_reconstruction", "reason": "no manifest"}])

    def test_a_manifest_without_a_fingerprint_is_not_adopted(self):
        with tempfile.TemporaryDirectory() as folder:
            task = build_task(Path(folder))
            manifest_path = task.directory / "stages/scene_reconstruction/stage.json"
            manifest = json.loads(manifest_path.read_text())
            manifest.pop("request_sha256")
            manifest_path.write_text(json.dumps(manifest))
            report = adapter.run(task, dry_run=False)
            self.assertEqual(report["skipped"], [{"module": "scene_reconstruction",
                                                  "reason": "manifest records no request fingerprint"}])

    def test_an_unknown_task_id_is_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(adapter.main(["absent", "--root", folder]), 2)


if __name__ == "__main__":
    unittest.main()
