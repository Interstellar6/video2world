from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "mesh_postprocess.py"
SPEC = importlib.util.spec_from_file_location("mesh_postprocess_adapter", SCRIPT)
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


def add_lineage(task, obj):
    for name in ("image.png", "mask.png"):
        (task / name).write_bytes(name.encode())
    adapter.write_json(task / "descriptions.json", {"objects": [{
        "frame_id": "000000", "object_id": "original_bed_observation", "image_path": "image.png",
        "description": "Original bed description", "attributes": {"material": "fabric"},
    }]})
    obj["source_descriptions"] = {"path": "descriptions.json", "sha256": adapter.sha256(task / "descriptions.json")}
    obj["source_observations"] = [{
        "frame_id": "000000", "source_object_id": "original_bed_observation",
        "source_image_path": "image.png", "source_image_sha256": adapter.sha256(task / "image.png"),
        "source_mask_path": "mask.png", "source_mask_sha256": adapter.sha256(task / "mask.png"),
    }]
    return obj


def provider_inputs(task, objects):
    adapter.write_json(task / "meshes.json", {"objects": objects})
    adapter.write_json(task / "candidates.json", {"objects": [{"object_id": item["object_id"]} for item in objects]})
    adapter.write_json(task / "inputs.json", {"module": "mesh_postprocess", "inputs": {
        "completed_object_meshes": {"path": "meshes.json", "evidence": "generated"},
        "completion_candidates": {"path": "candidates.json", "evidence": "generated"},
    }})
    return argparse.Namespace(task_dir=task, inputs=task / "inputs.json", outputs=task / "outputs.json",
                              face_limit=40, texture_resolution=32, coacd_threshold=.05,
                              coacd_resolution=500, coacd_iterations=10, seed=0)


class ManifestBoundaryTests(unittest.TestCase):
    def test_pymeshfix_result_is_rejected_when_it_collapses_the_geometry(self):
        before = {"finite": True, "faces": 100, "watertight": False, "volume": 0.0117,
                  "bounds": [[-0.5, -0.35, -0.47], [0.48, 0.26, 0.50]]}
        collapsed = {"finite": True, "faces": 50, "watertight": True, "volume": 0.001,
                     "bounds": [[-0.49, 0.14, -0.36], [0.31, 0.25, 0.50]]}
        outcome, accepted = adapter.meshfix_verdict(before, collapsed)
        self.assertFalse(accepted)
        self.assertTrue(outcome.startswith("rejected"), outcome)
        good = {"finite": True, "faces": 120, "watertight": True, "volume": 0.012, "bounds": before["bounds"]}
        self.assertEqual(adapter.meshfix_verdict(before, good), ("accepted", True))
        open_surface = {"finite": True, "faces": 120, "watertight": False, "volume": 0.012, "bounds": before["bounds"]}
        self.assertEqual(adapter.meshfix_verdict(before, open_surface)[0], "rejected_not_watertight")

    def test_an_unbounded_pymeshfix_repair_is_refused_and_recorded(self):
        # A 100k-face 3D-Fixer bed spent over an hour inside pymeshfix without
        # converging, so an oversized mesh must be refused before the call. The
        # limit is patched down so a ten-face open box stands in for that mesh.
        import trimesh

        box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        open_mesh = trimesh.Trimesh(vertices=box.vertices, faces=box.faces[:-2], process=True)
        # fill_holes would close this box, and a watertight mesh never reaches
        # pymeshfix at all, so it is disabled to keep the mesh open.
        with patch.object(trimesh.repair, "fill_holes", lambda *args, **kwargs: None), \
                patch.object(adapter, "PYMESHFIX_FACE_LIMIT", 8):
            repaired, report = adapter.repair_and_simplify(open_mesh, 0)
        self.assertFalse(report["meshfix_used"])
        self.assertEqual(report["meshfix_outcome"], f"skipped_mesh_over_8_faces:{len(open_mesh.faces)}")
        self.assertEqual(report["simplification"], "below_face_limit")
        self.assertGreater(len(repaired.faces), 0)
        self.assertGreater(report["after"]["volume"], 0)

    def test_a_mesh_within_the_limit_still_reaches_pymeshfix(self):
        import trimesh
        import types

        class FakeMeshFix:
            def __init__(self, vertices, faces):
                self.v, self.f = vertices, faces

            def repair(self, **kwargs):
                return True

        fake = types.ModuleType("pymeshfix")
        fake.MeshFix = FakeMeshFix
        box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        open_mesh = trimesh.Trimesh(vertices=box.vertices, faces=box.faces[:-2], process=True)
        with patch.dict(sys.modules, {"pymeshfix": fake}), \
                patch.object(trimesh.repair, "fill_holes", lambda *args, **kwargs: None), \
                patch.object(adapter, "PYMESHFIX_FACE_LIMIT", 1000):
            repaired, report = adapter.repair_and_simplify(open_mesh, 0)
        self.assertTrue(report["meshfix_used"])
        self.assertEqual(report["meshfix_outcome"], "rejected_not_watertight")
        self.assertGreater(report["after"]["volume"], 0)

    def test_coordinate_metadata_deep_copies_lineage_and_keeps_legacy_absence(self):
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            source = task / "mesh.glb"
            source.write_bytes(b"geometry fixture")
            item = add_lineage(task, {"object_id": "bed", "mesh_path": source.name})
            adapter.validate_object_lineages(task, [item])
            expected = copy.deepcopy(item)
            result = adapter.coordinate_metadata(task, item, source)
            self.assertEqual(result["source_observations"], item["source_observations"])
            self.assertEqual(result["source_descriptions"], item["source_descriptions"])
            result["source_observations"][0]["source_object_id"] = "changed"
            result["source_descriptions"]["path"] = "changed"
            self.assertEqual(item, expected)
            legacy = {"object_id": "bed", "mesh_path": source.name}
            adapter.validate_object_lineages(task, [legacy])
            result = adapter.coordinate_metadata(task, legacy, source)
            self.assertNotIn("source_observations", result)
            self.assertNotIn("source_descriptions", result)

    def test_invalid_lineage_blocks_before_mesh_loading_and_external_providers(self):
        for mutation in ("mask_hash", "missing_field"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder:
                task = Path(folder)
                (task / "mesh.glb").write_bytes(b"must not be loaded")
                item = add_lineage(task, {"object_id": "bed", "mesh_path": "mesh.glb"})
                if mutation == "mask_hash":
                    item["source_observations"][0]["source_mask_sha256"] = "0" * 64
                else:
                    item.pop("source_descriptions")
                args = provider_inputs(task, [item])
                mesh_library = Mock()
                with patch.dict(sys.modules, {"trimesh": mesh_library}), \
                     patch.object(adapter, "repair_and_simplify") as repair, patch.object(adapter, "coacd_hulls") as coacd:
                    with self.assertRaisesRegex(ValueError, "lineage"):
                        adapter.run(args)
                mesh_library.load.assert_not_called()
                repair.assert_not_called()
                coacd.assert_not_called()
                self.assertFalse(args.outputs.exists())
                self.assertFalse((task / "stages/mesh_postprocess").exists())

    def test_task_escape_rejected_including_symlink(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            task = root / "task"
            task.mkdir()
            outside = root / "outside.glb"
            outside.write_bytes(b"not loaded")
            (task / "link.glb").symlink_to(outside)
            for candidate in ("../outside.glb", "link.glb"):
                with self.assertRaises(adapter.PostprocessError):
                    adapter.task_path(task, candidate)

    def test_manifest_rejects_duplicate_objects_and_contract_geometry(self):
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            (task / "mesh.glb").write_bytes(b"not loaded")
            manifest = task / "meshes.json"
            item = {"object_id": "bed", "mesh_path": "mesh.glb"}
            manifest.write_text(json.dumps({"objects": [item, item]}))
            with self.assertRaisesRegex(adapter.PostprocessError, "duplicate"):
                adapter.object_records(task, {"path": "meshes.json"})
            manifest.write_text(json.dumps({"objects": [{**item, "evidence": "contract_only"}]}))
            with self.assertRaisesRegex(adapter.PostprocessError, "contract-only"):
                adapter.object_records(task, {"path": "meshes.json"})

    def test_external_gltf_texture_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            path = task / "mesh.gltf"
            path.write_text(json.dumps({"images": [{"uri": "https://example.com/texture.png"}]}))
            with self.assertRaisesRegex(adapter.PostprocessError, "task-local"):
                adapter.check_external_dependencies(task, path)

    def test_registration_retains_original_source_binding_without_promoting_alignment(self):
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            source = task / "source.glb"
            source.write_bytes(b"original completion geometry")
            (task / "matches.json").write_text("{}")
            registration = {"status": "accepted", "source_geometry_sha256": adapter.sha256(source), "correspondences_path": "matches.json"}
            item = {"object_id": "bed", "coordinate_frame": "object_local", "unit": "asset_units", "room_alignment": "not_estimated", "registration": registration, "observed_anchor": {"position": [1, 2, 3]}, "observed_anchor_applied": False}
            metadata = adapter.coordinate_metadata(task, item, source)
            self.assertEqual(metadata["registration"], registration)
            self.assertEqual(metadata["registration_source_geometry_path"], "source.glb")
            self.assertEqual(metadata["registration_source_geometry_sha256"], registration["source_geometry_sha256"])
            self.assertEqual(metadata["room_alignment"], "not_estimated")
            self.assertFalse(metadata["observed_anchor_applied"])
            self.assertEqual(metadata["processing_coordinate_transform"][3], [0, 0, 0, 1])
            self.assertIn("not_revalidated", metadata["registration_validation"])
            self.assertNotIn("registration_source_geometry_path", item)
            item["registration"]["correspondences_path"] = "../outside.json"
            with self.assertRaisesRegex(adapter.PostprocessError, "strictly beneath"):
                adapter.coordinate_metadata(task, item, source)

    def test_registration_reference_hash_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            source = task / "source.glb"
            source.write_bytes(b"geometry")
            with self.assertRaisesRegex(adapter.PostprocessError, "hash"):
                adapter.coordinate_metadata(task, {"object_id": "bed", "registration": {}, "registration_source_geometry_sha256": "0" * 64}, source)


GEOMETRY_DEPS = all(importlib.util.find_spec(name) is not None for name in (
    "numpy", "trimesh", "vtk", "xatlas", "coacd", "pymeshfix", "PIL", "scipy",
))


@unittest.skipUnless(all(importlib.util.find_spec(name) is not None for name in ("numpy", "trimesh", "PIL")),
                     "requires CPU mesh and image IO dependencies")
class LineageProviderTests(unittest.TestCase):
    def test_all_output_manifests_preserve_lineage_and_legacy_objects(self):
        import numpy as np
        import trimesh
        from PIL import Image

        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            trimesh.creation.box().export(task / "mesh.glb")
            item = add_lineage(task, {"object_id": "bed", "mesh_path": "mesh.glb"})
            expected = adapter.lineage_metadata(item)
            args = provider_inputs(task, [item, {"object_id": "legacy", "mesh_path": "mesh.glb"}])

            def bake(source, repaired, path, resolution):
                image = Image.new("RGB", (resolution, resolution), (25, 110, 235))
                image.save(path)
                baked = repaired.copy()
                baked.visual = trimesh.visual.texture.TextureVisuals(
                    uv=np.full((len(baked.vertices), 2), .25),
                    material=trimesh.visual.material.PBRMaterial(baseColorTexture=image))
                return baked, {"status": "CPU appearance fixture"}

            with patch.object(adapter, "repair_and_simplify", side_effect=lambda mesh, limit: (mesh.copy(), {})), \
                 patch.object(adapter, "bake_texture", side_effect=bake), \
                 patch.object(adapter, "coacd_hulls", side_effect=lambda mesh, args: ([mesh.convex_hull], {})), \
                 patch.object(adapter.importlib.metadata, "version", return_value="CPU-test"):
                result = adapter.run(args)
            self.assertEqual(set(result["outputs"]), {"repaired_visual_meshes", "uv_textures", "coacd_collision_meshes", "mesh_qa_report"})
            for role, artifact in result["outputs"].items():
                with self.subTest(role=role):
                    records = adapter.read_json(task / artifact["path"])["objects"]
                    self.assertEqual(adapter.lineage_metadata(records[0]), expected)
                    self.assertEqual(records[0]["source_observations"][0]["source_object_id"], "original_bed_observation")
                    self.assertNotIn("source_observations", records[1])
                    self.assertNotIn("source_descriptions", records[1])


@unittest.skipUnless(GEOMETRY_DEPS, "requires installed CPU geometry environment")
class GeometryProviderTests(unittest.TestCase):
    def test_bakes_existing_texture_and_applies_source_scene_transform(self):
        import numpy as np
        import trimesh
        from PIL import Image

        mesh = trimesh.creation.box()
        mesh.visual = trimesh.visual.texture.TextureVisuals(
            uv=np.tile([[0.25, 0.25]], (len(mesh.vertices), 1)),
            material=trimesh.visual.material.PBRMaterial(baseColorTexture=Image.new("RGB", (8, 8), (25, 110, 235))),
        )
        mesh.apply_translation([2, 3, 4])
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "baked.png"
            baked, report = adapter.bake_texture(mesh, mesh, destination, 32)
            self.assertGreater(report["covered_pixels"], 0)
            np.testing.assert_allclose(baked.bounds, mesh.bounds)
            np.testing.assert_allclose(np.asarray(Image.open(destination))[:, :, :3].mean(axis=(0, 1)), [25, 110, 235], atol=1)

    def test_real_multi_object_repair_bake_and_coacd(self):
        import numpy as np
        import trimesh
        from PIL import Image

        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            (task / "inputs").mkdir()
            mesh = trimesh.creation.box()
            mesh.visual.vertex_colors = np.array([255, 30, 15, 255])
            mesh.export(task / "inputs/red.glb")
            second = trimesh.creation.icosphere(subdivisions=1)
            second.visual.vertex_colors = np.array([20, 210, 60, 255])
            second.export(task / "inputs/green.glb")
            objects = [{"object_id": name, "mesh_path": f"inputs/{name}.glb", "coordinate_frame": "object_local", "unit": "meter"} for name in ("red", "green")]
            adapter.write_json(task / "inputs/meshes.json", {"objects": objects})
            adapter.write_json(task / "inputs/candidates.json", {"objects": [{"object_id": name} for name in ("red", "green")]})
            adapter.write_json(task / "inputs.json", {
                "module": "mesh_postprocess",
                "inputs": {
                    "completed_object_meshes": {"path": "inputs/meshes.json", "evidence": "generated"},
                    "completion_candidates": {"path": "inputs/candidates.json", "evidence": "generated"},
                },
            })
            args = argparse.Namespace(task_dir=task, inputs=task / "inputs.json", outputs=task / "provider-artifacts.json", face_limit=40, texture_resolution=32, coacd_threshold=0.05, coacd_resolution=500, coacd_iterations=10, seed=0)
            result = adapter.run(args)
            self.assertEqual(set(result["outputs"]), {"repaired_visual_meshes", "uv_textures", "coacd_collision_meshes", "mesh_qa_report"})
            qa = adapter.read_json(task / result["outputs"]["mesh_qa_report"]["path"])
            self.assertFalse(qa["learned_3d_fixer_invoked"])
            self.assertEqual(len(qa["objects"]), 2)
            self.assertEqual(qa["objects"][1]["parts"][0]["simplification"], "vtk_quadric_decimation")
            self.assertLessEqual(qa["objects"][1]["parts"][0]["after"]["faces"], 40)
            textures = adapter.read_json(task / result["outputs"]["uv_textures"]["path"])
            for record, expected in zip(textures["objects"], ((255, 30, 15), (20, 210, 60))):
                texture = Image.open(task / record["textures"][0]["path"]).convert("RGB")
                np.testing.assert_allclose(np.asarray(texture).mean(axis=(0, 1)), expected, atol=2)
            collisions = adapter.read_json(task / result["outputs"]["coacd_collision_meshes"]["path"])
            for record in collisions["objects"]:
                self.assertTrue(record["collision_eligible"])
                self.assertEqual(record["room_alignment"], "unknown")
                self.assertEqual(record["processing_source_geometry_sha256"], adapter.sha256(task / record["processing_source_geometry_path"]))
                for hull in record["hulls"]:
                    loaded = trimesh.load(task / hull["path"], force="mesh")
                    self.assertTrue(loaded.is_volume and loaded.is_convex)

    def test_mesh_without_source_appearance_fails_instead_of_fabricating_texture(self):
        import trimesh

        mesh = trimesh.creation.box()
        with self.assertRaisesRegex(adapter.PostprocessError, "no texture"):
            adapter.source_colors(mesh, [0], [[1.0, 0.0, 0.0]])


if __name__ == "__main__":
    unittest.main()
