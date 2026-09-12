from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_viewer.py"
SPEC = importlib.util.spec_from_file_location("build_viewer_adapter", SCRIPT)
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


def make_export(root: Path, *, objects=("bed", "lamp"), v1=()):
    (root / "scene").mkdir(parents=True)
    for name in ("mesh.ply", "point_cloud_3dgs.ply", "point_cloud_simple.ply"):
        (root / "scene" / name).write_bytes(b"ply\n")
    (root / "scene" / "depth").mkdir()
    (root / "object").mkdir(parents=True, exist_ok=True)
    (root / "object" / "background.glb").write_bytes(b"glTF")
    for object_id in objects:
        directory = root / "object" / "object_version_2" / object_id
        directory.mkdir(parents=True)
        (directory / "asset_mesh.glb").write_bytes(b"glTF")
        (directory / "asset_splat.ply").write_bytes(b"ply\n")
    for object_id in v1:
        directory = root / "object" / "object_version_1" / object_id
        directory.mkdir(parents=True)
        (directory / f"{object_id}.glb").write_bytes(b"glTF")
    manifest = {
        "schema_version": "1.0", "status": "complete", "missing": [],
        "scene_roles": {f"scene/{name}": {"sha256": "0" * 64}
                        for name in ("mesh.ply", "point_cloud_3dgs.ply", "point_cloud_simple.ply")},
        "object_version_1": {object_id: {} for object_id in v1},
        "object_version_2": {object_id: {} for object_id in objects},
    }
    manifest["scene_roles"]["scene/depth/manifest.json"] = {"sha256": "0" * 64}
    (root / "export_manifest.json").write_text(json.dumps(manifest))


class ViewerTests(unittest.TestCase):
    def test_assets_come_from_the_manifest_with_absolute_server_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            make_export(root, objects=("bed", "lamp"), v1=("ghost",))
            payload = adapter.viewer_assets(root, "test delivery")
            self.assertEqual(payload["title"], "test delivery")
            self.assertEqual([entry["role"] for entry in payload["scene"]],
                             ["scene/mesh.ply", "scene/point_cloud_3dgs.ply", "scene/point_cloud_simple.ply"])
            self.assertEqual(payload["background"], "/object/background.glb")
            # version-2 objects first, then any object that only exists in version 1
            self.assertEqual([entry["id"] for entry in payload["objects"]], ["bed", "lamp", "ghost"])
            bed = next(entry for entry in payload["objects"] if entry["id"] == "bed")
            self.assertEqual(bed["mesh"], "/object/object_version_2/bed/asset_mesh.glb")
            self.assertEqual(bed["splat"], "/object/object_version_2/bed/asset_splat.ply")
            ghost = next(entry for entry in payload["objects"] if entry["id"] == "ghost")
            self.assertNotIn("splat", ghost)

    def test_depth_maps_are_not_offered_as_scene_layers(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            make_export(root)
            payload = adapter.viewer_assets(root, "t")
            self.assertFalse(any("depth" in entry["role"] for entry in payload["scene"]))

    def test_run_writes_a_buildable_project_and_records_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            make_export(root)
            report = adapter.run(root, title="bedroom_4")
            for name in ("index.html", "app.js", "styles.css", "package.json", "vite.config.js"):
                self.assertTrue((root / "web-demo" / name).is_file(), name)
                self.assertIn(name, report["files"])
            self.assertEqual(report["objects"], ["bed", "lamp"])
            self.assertTrue((root / "web-demo" / "public" / "assets.json").is_file())
            written = json.loads((root / "web-demo" / "public" / "assets.json").read_text())
            self.assertEqual(written["objects"][0]["mesh"], "/object/object_version_2/bed/asset_mesh.glb")
            package = json.loads((root / "web-demo" / "package.json").read_text())
            self.assertIn("three", package["dependencies"])

    def test_a_missing_manifest_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(adapter.PipelineError):
                adapter.viewer_assets(Path(folder), "t")
        self.assertEqual(adapter.main(["--export-root", str(Path(folder) / "absent")]), 2)


if __name__ == "__main__":
    unittest.main()
