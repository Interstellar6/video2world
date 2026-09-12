from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_dataset.py"
SPEC = importlib.util.spec_from_file_location("run_dataset_adapter", SCRIPT)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class OutputDirectoryTests(unittest.TestCase):
    def test_a_run_root_gets_the_task_in_its_outputs_store(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "runs"
            source = Path(folder) / "my-scene"
            source.mkdir()
            resolved_root, task_id = launcher.resolve_paths(root, None, source)
            self.assertEqual(resolved_root, root.resolve())
            self.assertEqual(task_id, "my-scene")
            self.assertEqual(resolved_root / "outputs" / task_id, root.resolve() / "outputs" / "my-scene")

    def test_a_task_directory_is_used_as_given(self):
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder) / "outputs" / "chosen-name"
            task.mkdir(parents=True)
            source = Path(folder) / "some-scene"
            source.mkdir()
            resolved_root, task_id = launcher.resolve_paths(task, None, source)
            self.assertEqual(task_id, "chosen-name")
            self.assertEqual(resolved_root / "outputs" / task_id, task.resolve())

    def test_an_explicit_task_id_wins_over_the_source_name(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "runs"
            source = Path(folder) / "scene"
            source.mkdir()
            _, task_id = launcher.resolve_paths(root, "custom-id", source)
            self.assertEqual(task_id, "custom-id")

    def test_a_task_id_that_contradicts_the_task_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder) / "outputs" / "chosen-name"
            task.mkdir(parents=True)
            with self.assertRaises(SystemExit):
                launcher.resolve_paths(task, "other-name", Path(folder))

    def test_a_source_name_that_cannot_be_a_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "***"
            source.mkdir()
            with self.assertRaises(SystemExit):
                launcher.resolve_paths(Path(folder) / "runs", None, source)

    def test_comparing_needs_an_export_to_compare(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "scene"
            source.mkdir()
            (source / "images").mkdir()
            (source / "sparse").mkdir()
            with self.assertRaises(SystemExit):
                launcher.main(["--source", str(source), "--output-dir", str(Path(folder) / "runs"),
                               "--compare-with", str(Path(folder) / "reference")])

    def test_the_comparison_flag_is_documented(self):
        import contextlib
        import io

        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), self.assertRaises(SystemExit):
            launcher.main(["--help"])
        self.assertIn("--compare-with", stream.getvalue())

    def test_the_default_profile_is_the_stream3d_asset_recipe(self):
        # The delivered architecture: Holi-Spatial scene, Qwen/SAM3-I objects,
        # FixAnything video, Stream3D completion and EmbodiedGen v2 assets.
        self.assertEqual(launcher.DEFAULT_PROFILE.name, "embodiedgen-stream3d.skipbg.json")
        self.assertTrue(launcher.DEFAULT_PROFILE.is_file())
        profile = json.loads(launcher.DEFAULT_PROFILE.read_text())
        self.assertIn("geometry_completion", profile["providers"])
        self.assertNotIn("observed_context_completion", profile["providers"])


if __name__ == "__main__":
    unittest.main()
