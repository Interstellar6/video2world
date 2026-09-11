from __future__ import annotations

import argparse
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/scene_recomposition.py"
SPEC = importlib.util.spec_from_file_location("scene_recomposition_adapter", SCRIPT)
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


class RecompositionBoundaryTests(unittest.TestCase):
    def test_contract_declares_registration_and_carving_evidence_inputs(self):
        self.assertTrue({"cameras", "lifting_report", "isolated_object_ply"} <= set(adapter.SPEC.inputs))

    def test_unknown_or_bbox_alignment_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory)
            with self.assertRaisesRegex(adapter.RecompositionError, "alignment is unknown"):
                adapter.registration_record(task, {"room_alignment": "not_estimated"})
            with self.assertRaisesRegex(adapter.RecompositionError, "bbox"):
                adapter.registration_record(task, {"registration": {"status": "accepted", "method": "bbox_alignment"}})


GEOMETRY_DEPS = all(importlib.util.find_spec(name) is not None for name in ("numpy", "PIL", "plyfile", "trimesh"))


@unittest.skipUnless(GEOMETRY_DEPS, "requires numpy, Pillow, plyfile and trimesh")
class SceneAssemblyTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import trimesh

        self.temp = tempfile.TemporaryDirectory()
        self.task = Path(self.temp.name)
        self.asset_dir = self.task / "inputs"
        self.asset_dir.mkdir()
        self.stage = self.task / "stages/scene_recomposition"
        self.stage.mkdir(parents=True)
        self.mesh = trimesh.creation.box(extents=[1, 2, 3])
        self.mesh.visual.vertex_colors = [230, 30, 20, 255]
        self.mesh.export(self.asset_dir / "visual.glb")
        self.mesh.export(self.asset_dir / "visual.obj")
        self.mesh.export(self.asset_dir / "hull.obj", include_color=False)
        _, vertices = adapter.scene_geometry(self.task, self.asset_dir / "visual.glb")
        self.transform = np.array([[0, -2, 0, 4], [2, 0, 0, 5], [0, 0, 2, -1], [0, 0, 0, 1]], dtype=float)
        observed = vertices @ self.transform[:3, :3].T + self.transform[:3, 3]
        background = np.array([[20, 20, 20], [22, 20, 20], [20, 22, 20], [20, 20, 22]], dtype=float)
        self.write_gaussians(self.asset_dir / "observed.ply", observed)
        self.write_gaussians(self.asset_dir / "carved.ply", background)
        self.write_gaussians(self.asset_dir / "source.ply", np.concatenate((observed, background)))
        self.write_gaussians(self.asset_dir / "generated_background.ply", background + 0.1)
        proof = {"object_id": "box", "source_indexing": "trimesh_sorted_scene_nodes_baked_vertices", "target_indexing": "ply_vertex_row",
                 "pairs": [{"source_vertex_index": index, "target_gaussian_index": index, "split": "fit" if index < 3 else "validation"} for index in range(len(vertices))]}
        adapter.write_json(self.asset_dir / "correspondences.json", proof)
        self.registration = {
            "object_id": "box", "status": "accepted", "method": "observed_correspondence_sim3", "object_to_world": self.transform.tolist(),
            "source_coordinate_frame": "object_local", "source_units": "asset_units", "target_coordinate_frame": "colmap_world", "target_units": "colmap_reconstruction",
            "source_geometry_sha256": adapter.sha256(self.asset_dir / "visual.glb"), "target_object_ply_sha256": adapter.sha256(self.asset_dir / "observed.ply"),
            "target_scene_sha256": adapter.sha256(self.asset_dir / "source.ply"), "correspondences_path": "inputs/correspondences.json", "correspondences_sha256": adapter.sha256(self.asset_dir / "correspondences.json"),
        }
        self.visual = {"object_id": "box", "mesh_path": "inputs/visual.glb", "sha256": adapter.sha256(self.asset_dir / "visual.glb"), "coordinate_frame": "object_local", "unit": "asset_units", "registration": self.registration}
        self.observed = {"object_id": "box", "ply_path": "inputs/observed.ply", "sha256": adapter.sha256(self.asset_dir / "observed.ply"), "coordinate_frame": "colmap_world", "units": "colmap_reconstruction", "association_status": "geometrically_verified"}
        physical = {"object_id": "box", "obj_path": "inputs/visual.obj", "source_mesh_path": "inputs/visual.glb", "coordinate_frame": "object_local", "units": "asset_units", "scale_applied": False,
                    "collision": {"coordinate_frame": "object_local", "unit": "asset_units", "collision_eligible": True, "hulls": [{"path": "inputs/hull.obj", "sha256": adapter.sha256(self.asset_dir / "hull.obj")}]}}
        self.documents = {
            "cameras": {"coordinate_frame": "colmap_world", "units": "colmap_reconstruction", "metric_scale_known": False},
            "lifting_report": {"coordinate_frame": "colmap_world", "units": "colmap_reconstruction", "status": "passed", "carve_performed": True, "voxel_size": 0.01,
                               "source_files": {"scene_gaussian_ply": {"sha256": adapter.sha256(self.asset_dir / "source.ply")}}, "carved_scene_sha256": adapter.sha256(self.asset_dir / "carved.ply"),
                               "source_gaussian_count": len(vertices) + 4, "remaining_gaussian_count": 4, "removed_gaussian_count": len(vertices), "accepted_object_ids": ["box"]},
            "isolated_object_ply": {"objects": [self.observed]}, "repaired_visual_meshes": {"objects": [self.visual]},
            "physical_object_obj": {"objects": [physical]}, "physics_properties": {"objects": [{"object_id": "box", "mass_kg": [1, 3], "measured": False}]},
        }
        self.envelope = {"module": "scene_recomposition", "inputs": {
            "scene_gaussian_ply": {"path": "inputs/source.ply", "evidence": "observed"},
            "carved_scene_ply": {"path": "inputs/carved.ply", "evidence": "derived"},
            "generated_background_gaussian_ply": {"path": "inputs/generated_background.ply", "evidence": "generated"},
        }}
        for name, document in self.documents.items():
            adapter.write_json(self.asset_dir / f"{name}.json", document)
            self.envelope["inputs"][name] = {"path": f"inputs/{name}.json", "evidence": "observed" if name == "cameras" else "derived"}
        adapter.write_json(self.stage / "inputs.json", self.envelope)
        self.args = argparse.Namespace(task_dir=self.task, inputs=self.stage / "inputs.json", outputs=self.stage / "provider-artifacts.json", visual_points=64, seed=1, max_registration_error_world_units=None, max_repair_bounds_change_ratio=0.2)

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def write_gaussians(path, positions):
        import numpy as np
        from plyfile import PlyData, PlyElement

        rows = np.zeros(len(positions), dtype=[(name, "f4") for name in ("x", "y", "z", "f_dc_0", "opacity", "scale_0", "rot_0")])
        for axis, name in enumerate(("x", "y", "z")):
            rows[name] = positions[:, axis]
        rows["rot_0"] = 1
        PlyData([PlyElement.describe(rows, "vertex")]).write(path)

    def test_visual_and_collision_share_one_verified_sim3(self):
        import numpy as np
        import trimesh

        result = adapter.run(self.args)
        self.assertEqual(result["status"], "assembled_candidate")
        published = adapter.read_json(self.args.outputs)["outputs"]
        self.assertEqual(set(published), {role.name for role in adapter.SPEC.outputs})
        world = adapter.read_json(self.task / published["world_manifest"]["path"])
        visual = adapter.read_json(self.task / published["visual_scene_manifest"]["path"])
        collision = adapter.read_json(self.task / published["collision_scene_manifest"]["path"])
        np.testing.assert_allclose(world["transforms"]["box"]["object_to_world"], self.transform)
        self.assertEqual(visual["objects"][0]["transform_sha256"], collision["objects"][0]["transform_sha256"])
        self.assertEqual(visual["objects"][0]["transform_id"], "box")
        self.assertTrue(visual["backgrounds"][0]["enabled"])
        self.assertFalse(visual["backgrounds"][1]["enabled"])
        self.assertFalse(collision["simulation_ready"])
        self.assertEqual(world["units"], "colmap_reconstruction")
        cloud = trimesh.load(self.task / visual["objects"][0]["ply"]["path"], process=False)
        self.assertEqual(len(cloud.vertices), 64)
        np.testing.assert_allclose(cloud.colors[:, :3].mean(0), [230, 30, 20], atol=1)

    def test_unknown_registration_writes_diagnostics_but_does_not_publish(self):
        del self.visual["registration"]
        adapter.write_json(self.asset_dir / "repaired_visual_meshes.json", {"objects": [self.visual]})
        with self.assertRaisesRegex(adapter.RecompositionError, "alignment is unknown"):
            adapter.run(self.args)
        self.assertFalse(self.args.outputs.exists())
        report_path = next(self.stage.glob("run-*/recomposition_report.json"))
        report = adapter.read_json(report_path)
        self.assertEqual(report["status"], "blocked_registration_or_assets")
        world = adapter.read_json(report_path.with_name("world_manifest.json"))
        self.assertIsNone(world["objects"][0]["object_to_world"])

    def test_heldout_correspondence_error_blocks_alignment(self):
        proof = adapter.read_json(self.asset_dir / "correspondences.json")
        proof["pairs"][4]["target_gaussian_index"], proof["pairs"][5]["target_gaussian_index"] = proof["pairs"][5]["target_gaussian_index"], proof["pairs"][4]["target_gaussian_index"]
        adapter.write_json(self.asset_dir / "correspondences.json", proof)
        self.registration["correspondences_sha256"] = adapter.sha256(self.asset_dir / "correspondences.json")
        with self.assertRaisesRegex(adapter.RecompositionError, "validation residuals"):
            adapter.validate_registration(self.task, "box", self.visual, self.observed, self.documents["cameras"], self.registration["target_scene_sha256"], 0.03, 0.2)

    def test_nonuniform_scale_and_units_mismatch_are_rejected(self):
        import numpy as np

        matrix = np.eye(4)
        matrix[0, 0] = 2
        with self.assertRaisesRegex(adapter.RecompositionError, "uniform"):
            adapter.sim3(matrix)
        self.registration["target_units"] = "meter"
        with self.assertRaisesRegex(adapter.RecompositionError, "units disagree"):
            adapter.validate_registration(self.task, "box", self.visual, self.observed, self.documents["cameras"], self.registration["target_scene_sha256"], 0.03, 0.2)

    def test_registration_hash_and_collider_hash_tampering_are_rejected(self):
        self.registration["source_geometry_sha256"] = "wrong"
        with self.assertRaisesRegex(adapter.RecompositionError, "hash-bound"):
            adapter.validate_registration(self.task, "box", self.visual, self.observed, self.documents["cameras"], self.registration["target_scene_sha256"], 0.03, 0.2)
        with (self.asset_dir / "hull.obj").open("a") as handle:
            handle.write("# changed\n")
        with self.assertRaisesRegex(adapter.RecompositionError, "hull format/hash"):
            adapter.validate_physical_assets(self.task, self.visual, self.documents["physical_object_obj"]["objects"][0])

    @unittest.skipUnless(importlib.util.find_spec("open3d") is not None, "requires Open3D")
    def test_open3d_can_solve_scale_from_supplied_observed_correspondences(self):
        import numpy as np

        del self.registration["object_to_world"]
        self.registration["status"] = "correspondences_available"
        transform, report = adapter.validate_registration(self.task, "box", self.visual, self.observed, self.documents["cameras"], self.registration["target_scene_sha256"], 0.03, 0.2)
        np.testing.assert_allclose(transform["object_to_world"], self.transform, atol=1e-6)
        self.assertTrue(report["solved_with_open3d"])


if __name__ == "__main__":
    unittest.main()
