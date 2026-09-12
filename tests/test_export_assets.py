from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/export_assets.py"
SPEC = importlib.util.spec_from_file_location("export_assets_adapter", SCRIPT)
export = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export)


def write_gaussian_ply(path: Path, positions):
    import numpy as np
    from plyfile import PlyData, PlyElement

    rows = np.zeros(len(positions), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("opacity", "f4")])
    for index, position in enumerate(positions):
        rows[index]["x"], rows[index]["y"], rows[index]["z"] = position
        rows[index]["opacity"] = 1.0
    PlyData([PlyElement.describe(rows, "vertex")]).write(str(path))



def _glb_json_chunk(path: Path) -> str:
    import struct

    data = path.read_bytes()
    offset = 12
    while offset < len(data):
        length, kind = struct.unpack("<II", data[offset:offset + 8])
        if kind == 0x4E4F534A:
            return data[offset + 8:offset + 8 + length].decode("utf-8")
        offset += 8 + length
    raise AssertionError("no JSON chunk in GLB")


class GaussianSceneExportTests(unittest.TestCase):
    def build(self, positions):
        directory = Path(tempfile.mkdtemp())
        task = export.Task(directory, "scene")
        (directory / "outputs/scene").mkdir(parents=True)
        source = directory / "outputs/scene/point_cloud.ply"
        write_gaussian_ply(source, positions)
        record = {"path": "point_cloud.ply", "sha256": export.sha256(source),
                  "size_bytes": source.stat().st_size}
        return task, source, record, directory

    def test_non_finite_splats_are_dropped_from_the_delivered_cloud(self):
        import numpy as np
        from plyfile import PlyData

        task, _, record, directory = self.build([[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0], [1.0, 1.0, 1.0]])
        destination = directory / "export/scene/point_cloud_3dgs.ply"
        entry = export.copy_gaussian_ply(task, record, destination)
        self.assertEqual(entry["non_finite_rows_dropped"], 1)
        self.assertEqual(entry["exported_rows"], 2)
        self.assertEqual(entry["source_sha256"], record["sha256"])
        self.assertEqual(entry["sha256"], export.sha256(destination))
        rows = PlyData.read(str(destination))["vertex"].data
        self.assertEqual(len(rows), 2)
        self.assertTrue(np.isfinite(np.column_stack([rows[key] for key in ("x", "y", "z")])).all())

    def test_a_cloud_without_non_finite_rows_is_copied_unchanged(self):
        task, _, record, directory = self.build([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        destination = directory / "export/scene/point_cloud_3dgs.ply"
        entry = export.copy_gaussian_ply(task, record, destination)
        self.assertEqual(entry["non_finite_rows_dropped"], 0)
        self.assertEqual(entry["sha256"], record["sha256"])

    def test_a_cloud_that_is_entirely_non_finite_is_still_rejected(self):
        task, _, record, directory = self.build([[float("nan")] * 3, [float("inf"), 0.0, 0.0]])
        with self.assertRaises(export.PipelineError):
            export.copy_gaussian_ply(task, record, directory / "export/scene/point_cloud_3dgs.ply")

    def test_scene_delivery_needs_a_mesh_and_a_depth_record(self):
        with tempfile.TemporaryDirectory() as folder:
            task = export.Task(Path(folder), "scene")
            (task.directory / "inputs").mkdir(parents=True)
            missing = Path(folder) / "absent.glb"
            with self.assertRaises(export.PipelineError):
                export.deliverable_mesh(missing, Path(folder) / "out.ply", 1000)

    @unittest.skipUnless(importlib.util.find_spec("vtk") and importlib.util.find_spec("trimesh"),
                         "vtk and trimesh are required for scene delivery")
    def test_scene_decimation_reaches_the_budget_and_keeps_vertex_colours(self):
        import numpy as np
        import trimesh

        with tempfile.TemporaryDirectory() as folder:
            mesh = trimesh.creation.icosphere(subdivisions=4)
            colors = np.tile(np.array([[10, 200, 30, 255]], dtype=np.uint8), (len(mesh.vertices), 1))
            mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=colors)
            source = Path(folder) / "scene.glb"
            mesh.export(source)
            destination = Path(folder) / "out.ply"
            budget = 200
            info = export.deliverable_mesh(source, destination, budget)
            self.assertEqual(info["decimation"], "vtk_decimate_pro_binary_quadric")
            self.assertLessEqual(info["faces"], budget)
            self.assertGreater(info["faces"], 0)
            reloaded = trimesh.load(destination, force="mesh", process=False)
            self.assertEqual(reloaded.visual.kind, "vertex")
            self.assertTrue((np.asarray(reloaded.visual.vertex_colors)[:, :3] == np.array([10, 200, 30])).all())

    @unittest.skipUnless(importlib.util.find_spec("vtk") and importlib.util.find_spec("trimesh"),
                         "vtk and trimesh are required for scene delivery")
    def test_scene_delivery_writes_a_glb_with_vertex_colours(self):
        import numpy as np
        import trimesh

        with tempfile.TemporaryDirectory() as folder:
            mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
            colors = np.tile(np.array([[200, 20, 20, 255]], dtype=np.uint8), (len(mesh.vertices), 1))
            mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=colors)
            source = Path(folder) / "scene.glb"
            mesh.export(source)
            destination = Path(folder) / "background.glb"
            info = export.deliverable_mesh(source, destination, 0)
            self.assertEqual(info["decimation"], "not_needed")
            attributes = json.loads(_glb_json_chunk(destination))["meshes"][0]["primitives"][0]["attributes"]
            self.assertIn("COLOR_0", attributes)

    def test_a_mesh_without_vertex_colours_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "flat.ply"
            path.write_text("ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\n"
                            "property float z\nelement face 1\nproperty list uchar int vertex_indices\n"
                            "end_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n")
            try:
                import vtk  # noqa: F401
            except ImportError:
                self.skipTest("vtk is required for scene delivery")
            polydata = export.read_scene_mesh(path)
            self.assertFalse(export.normalize_colors(polydata))
            with self.assertRaises(export.PipelineError):
                export.write_scene_mesh(polydata, Path(folder) / "out.ply")
    @unittest.skipUnless(importlib.util.find_spec("vtk") and importlib.util.find_spec("trimesh"),
                         "vtk and trimesh are required for scene delivery")
    def test_a_background_without_calibrated_frames_falls_back_to_vertex_colours(self):
        import numpy as np
        import trimesh

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            task = export.Task(root, "scene")
            mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
            mesh.visual = trimesh.visual.ColorVisuals(
                vertex_colors=np.tile(np.array([[9, 9, 9, 255]], dtype=np.uint8), (len(mesh.vertices), 1)))
            source = root / "outputs/scene/inputs/tsdf.glb"
            source.parent.mkdir(parents=True, exist_ok=True)
            mesh.export(source)
            destination = root / "delivery/object/background.glb"
            # No cameras/scene_depth role is published, so export must not try to bake.
            info = export.deliverable_mesh(source, destination, 0)
            self.assertEqual(info["decimation"], "not_needed")
            self.assertTrue(destination.is_file())

    def test_verified_observed_parts_are_delivered_one_asset_each(self):
        with tempfile.TemporaryDirectory() as folder:
            task = export.Task(Path(folder), "scene")
            (task.directory / "inputs").mkdir(parents=True)
            (task.directory / "stages/object_lifting").mkdir(parents=True)
            parts = {}
            for name in ("bed/part_a.ply", "bed/part_b.ply", "lamp/part_c.ply"):
                path = task.directory / "stages/object_lifting" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"ply\n" + name.encode())
                parts[name] = export.sha256(path)
            document = {"objects": [
                {"object_id": "bed", "parts": [
                    {"component_id": "component_track_b", "ply_path": "stages/object_lifting/bed/part_b.ply",
                     "sha256": parts["bed/part_b.ply"], "association_status": "geometrically_verified_observed_component"},
                    {"component_id": "component_track_a", "ply_path": "stages/object_lifting/bed/part_a.ply",
                     "sha256": parts["bed/part_a.ply"], "association_status": "geometrically_verified_observed_component"}]},
                {"object_id": "lamp", "parts": [
                    {"component_id": "component_track_c", "ply_path": "stages/object_lifting/lamp/part_c.ply",
                     "sha256": parts["lamp/part_c.ply"], "association_status": "geometrically_verified_observed_component"}]}]}
            manifest_path = task.directory / "isolated.json"
            manifest_path.write_text(json.dumps(document))
            record = {"path": "isolated.json", "sha256": export.sha256(manifest_path),
                      "size_bytes": manifest_path.stat().st_size}
            export_root = Path(folder) / "delivery"
            summary = export.export_observed_parts(task, record, export_root)
            self.assertEqual(summary["parts"], 3)
            self.assertEqual(summary["objects"], ["bed", "lamp"])
            # parts are ordered by component id, so part_00 is the "a" component
            self.assertEqual(sorted(p.name for p in (export_root / "object/parts/bed").iterdir()),
                             ["bed_part_00.ply", "bed_part_01.ply"])
            delivered = json.loads((export_root / "object/parts/manifest.json").read_text())
            self.assertTrue(all(entry["evidence"] == "observed" for entry in delivered["parts"]))
            self.assertEqual(delivered["parts"][0]["sha256"], parts["bed/part_a.ply"])

    def test_a_component_ply_that_changed_after_lifting_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            task = export.Task(Path(folder), "scene")
            path = task.directory / "stages/object_lifting/bed/part_a.ply"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"ply\noriginal")
            document = {"objects": [{"object_id": "bed", "parts": [
                {"component_id": "component_track_a", "ply_path": "stages/object_lifting/bed/part_a.ply",
                 "sha256": "0" * 64}]}]}
            manifest_path = task.directory / "isolated.json"
            manifest_path.write_text(json.dumps(document))
            record = {"path": "isolated.json", "sha256": export.sha256(manifest_path),
                      "size_bytes": manifest_path.stat().st_size}
            with self.assertRaises(export.PipelineError):
                export.export_observed_parts(task, record, Path(folder) / "delivery")

    def test_every_exported_object_carries_its_own_provenance_report(self):
        with tempfile.TemporaryDirectory() as folder:
            task = export.Task(Path(folder), "scene")
            documents = {
                "completed_object_meshes": {"objects": [
                    {"object_id": "bed", "mesh_path": "stages/observed_context_completion/bed.glb",
                     "source_frame_ids": [1, 2]}]},
                "repaired_visual_meshes": {"objects": [
                    {"object_id": "bed", "mesh_path": "stages/mesh_postprocess/bed.glb"}]},
                "coacd_collision_meshes": {"objects": [
                    {"object_id": "bed", "mesh_path": "stages/mesh_postprocess/bed/collision.obj"}]},
            }
            index = {}
            for role, document in documents.items():
                path = task.directory / f"{role}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(document))
                index[role] = {"path": path.name, "sha256": export.sha256(path),
                               "size_bytes": path.stat().st_size, "evidence": "generated",
                               "producer": "mesh_postprocess"}
            export_root = Path(folder) / "delivery"
            (export_root / "object/object_version_1/bed").mkdir(parents=True)
            manifest = {"object_version_1": {"bed": {"bed.glb": {"source_path": "x", "sha256": "0" * 64}}},
                        "object_version_2": {}}
            export.export_object_reports(task, export_root, index, manifest)
            report = json.loads((export_root / "object/object_version_1/bed/asset_report.json").read_text())
            self.assertEqual(report["object_id"], "bed")
            self.assertEqual([entry["role"] for entry in report["provenance"]],
                             ["completed_object_meshes", "repaired_visual_meshes", "coacd_collision_meshes"])
            self.assertEqual(report["provenance"][0]["source_frame_ids"], [1, 2])
            self.assertEqual(report["provenance"][1]["provider"], "mesh_postprocess")
            # the manifest points at the report, and the report does not hash itself
            pointer = manifest["object_version_1"]["bed"]["asset_report.json"]
            self.assertEqual(pointer["sha256"], export.sha256(export_root / "object/object_version_1/bed/asset_report.json"))
            self.assertNotIn("asset_report.json", report["exported_files"])
            self.assertEqual(manifest["object_report_provenance"], {"bed": 3})

    def test_an_object_without_recorded_sources_still_gets_an_empty_report(self):
        with tempfile.TemporaryDirectory() as folder:
            task = export.Task(Path(folder), "scene")
            export_root = Path(folder) / "delivery"
            manifest = {"object_version_1": {}, "object_version_2": {"lamp": {"asset_mesh.glb": {}}}}
            export.export_object_reports(task, export_root, {}, manifest)
            report = json.loads((export_root / "object/object_version_2/lamp/asset_report.json").read_text())
            self.assertEqual(report["provenance"], [])
            self.assertEqual(report["export_version"], "object_version_2")

    def test_the_delivery_explains_itself_in_a_readme_the_manifest_binds(self):
        with tempfile.TemporaryDirectory() as folder:
            task = export.Task(Path(folder), "scene")
            root = Path(folder) / "delivery"
            root.mkdir()
            manifest = {"status": "complete", "missing": [],
                        "scene_roles": {"scene/point_cloud_3dgs.ply": {"gaussians": 1331069, "evidence": "observed",
                                                                       "source_path": "stages/scene_reconstruction/r/point_cloud.ply"},
                                        "scene/mesh.ply": {"faces": 1000000, "colour": True, "evidence": "observed",
                                                           "converted_from": "stages/scene_reconstruction/r/scene_tsdf.glb"}},
                        "background": {"textured": True, "faces": 100000, "faces_before": 12798127, "texture_size": 2048,
                                       "calibrated_frames_used": 50, "covered_texel_fraction": 0.459, "method": "per_face_bake"},
                        "object_version_2": {"bed": {"asset_mesh.glb": {}, "collision/hull_000.obj": {}}},
                        "object_version_1": {"bed": {"bed.glb": {}, "asset_report.json": {}}},
                        "observed_parts": {"parts": 4, "objects": ["bed"]},
                        "previews": {"previews": {"scene_overview": {}, "object_layout": {}}},
                        "viewer": {"directory": "web-demo"}}
            report = export.write_delivery_readme(task, root, manifest, type("A", (), {"scene_name": "bedroom_4"})())
            text = (root / "README.md").read_text()
            self.assertEqual(report["sha256"], export.sha256(root / "README.md"))
            self.assertTrue(text.startswith("# bedroom_4"))
            self.assertIn("1,331,069 gaussians", text)
            self.assertIn("1,000,000 faces", text)
            self.assertIn("46% of texels covered", text)
            self.assertIn("across 1 object ", text)
            self.assertIn("asset_mesh.glb", text)
            self.assertIn("web-demo", text)
            self.assertIn("promotion_allowed` is false", text)

    def test_a_scene_only_export_without_a_texture_says_so(self):
        with tempfile.TemporaryDirectory() as folder:
            task = export.Task(Path(folder), "scene")
            root = Path(folder) / "delivery"
            root.mkdir()
            manifest = {"status": "partial", "missing": ["scene_depth"], "scene_roles": {},
                        "background": {"faces": 1000000, "background_texture_fallback": "no calibrated frames"}}
            export.write_delivery_readme(task, root, manifest, type("A", (), {"scene_name": None})())
            text = (root / "README.md").read_text()
            self.assertIn("# scene", text)
            self.assertIn("Missing roles: `scene_depth`", text)
            self.assertIn("no baked texture: no calibrated frames", text)

    def test_a_tampered_source_cloud_is_rejected_before_export(self):
        task, source, record, directory = self.build([[0.0, 0.0, 0.0]])
        write_gaussian_ply(source, [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
        with self.assertRaises(export.PipelineError):
            export.copy_gaussian_ply(task, record, directory / "export/scene/point_cloud_3dgs.ply")


if __name__ == "__main__":
    unittest.main()
