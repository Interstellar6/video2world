from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/object_registration.py"
SPEC = importlib.util.spec_from_file_location("object_registration_adapter", SCRIPT)
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)
DEPS = all(importlib.util.find_spec(name) for name in ("numpy", "scipy", "open3d", "trimesh", "plyfile", "PIL"))


@unittest.skipUnless(DEPS, "requires installed CPU geometry environment")
class RegistrationTests(unittest.TestCase):
    def cameras(self):
        import numpy as np
        from scipy.spatial.transform import Rotation

        centers = np.array([[np.cos(a) * 3, np.sin(a) * 3, h] for h in (1, 2, 3) for a in (0, np.pi / 2, np.pi, np.pi * 1.5)])
        source = np.repeat(np.eye(4)[None], len(centers), axis=0)
        source[:, :3, 3] = -centers
        matrix = np.eye(4)
        rotation = Rotation.from_euler("xyz", [15, -10, 30], degrees=True).as_matrix()
        matrix[:3, :3] = 1.7 * rotation
        matrix[:3, 3] = [2, -1, 4]
        target = source.copy()
        target[:, :3, :3] = rotation.T
        target_centers = adapter.transform(centers, matrix)
        target[:, :3, 3] = -target_centers @ rotation
        return source, target, matrix, np.arange(12) % 4 == 3

    def test_camera_sim3_recovers_known_scale_rotation_and_heldout_poses(self):
        import numpy as np

        source, target, expected, heldout = self.cameras()
        solved, qa = adapter.camera_chain(source, target, heldout, .01, 2)
        np.testing.assert_allclose(solved, expected, atol=1e-6)
        self.assertEqual(qa["validation"]["count"], 3)
        self.assertLess(qa["validation"]["max_rotation_degrees"], 1e-5)

    def test_camera_heldout_drift_fails_without_refitting_to_validation(self):
        source, target, _, heldout = self.cameras()
        target[heldout, 0, 3] += .8
        with self.assertRaisesRegex(adapter.RegistrationError, "validation"):
            adapter.camera_chain(source, target, heldout, .01, 2)

    def test_camera_orientation_inconsistency_fails_even_when_centers_match(self):
        import numpy as np
        from scipy.spatial.transform import Rotation

        source, target, _, heldout = self.cameras()
        center = np.linalg.inv(target[-1])[:3, 3]
        target[-1, :3, :3] = Rotation.from_euler("z", 40, degrees=True).as_matrix() @ target[-1, :3, :3]
        target[-1, :3, 3] = -target[-1, :3, :3] @ center
        with self.assertRaisesRegex(adapter.RegistrationError, "orientation"):
            adapter.camera_chain(source, target, heldout, .01, 2)

    def test_export_axis_inverse_and_pose_roundtrip(self):
        import numpy as np
        from scipy.spatial.transform import Rotation

        q = Rotation.from_euler("xyz", [18, -27, 53], degrees=True).as_quat()
        rotation = Rotation.from_quat(q).as_matrix()
        pose = {"scale": [[2, 2, 2]], "rotation": [q[[3, 0, 1, 2]]], "translation": [[.1, .2, 5]]}
        canonical = np.array([[.2, -.4, .6], [-.3, .7, -.1]])
        export_row = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        glb = canonical @ export_row
        expected = ((canonical * 2) @ rotation + pose["translation"][0]) @ np.diag([-1, -1, 1])
        np.testing.assert_allclose(adapter.transform(glb, adapter.glb_to_reference_camera(pose)), expected, atol=1e-7)
        mirrored = np.diag([-1, 1, 1, 1])
        with self.assertRaisesRegex(adapter.RegistrationError, "reflection"):
            adapter.sim3(mirrored)

    @unittest.skipUnless(os.environ.get("PYTORCH3D_TEST_SOURCE"), "optional official CPU transform conformance")
    def test_official_pytorch3d_transform_matches_adapter(self):
        import numpy as np
        import torch

        sys.path.insert(0, os.environ["PYTORCH3D_TEST_SOURCE"])
        from pytorch3d.transforms import Transform3d, quaternion_to_matrix

        pose = {"scale": [[1.4, 1.4, 1.4]], "rotation": [[.5, .5, .5, .5]], "translation": [[.2, -.3, 4]]}
        canonical = torch.tensor([[.2, -.4, .6], [-.3, .7, -.1]])
        q = torch.tensor(pose["rotation"])
        expected = Transform3d().scale(torch.tensor(pose["scale"])).rotate(quaternion_to_matrix(q)).translate(torch.tensor(pose["translation"])).transform_points(canonical).numpy() * [-1, -1, 1]
        glb = canonical.numpy() @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        np.testing.assert_allclose(adapter.transform(glb, adapter.glb_to_reference_camera(pose)), expected, atol=1e-6)

    def geometry(self):
        import numpy as np
        import trimesh

        source = trimesh.creation.icosphere(subdivisions=2).vertices.copy()
        source *= [1.3, .8, .5]
        matrix = np.eye(4)
        matrix[:3, :3] *= 2
        matrix[:3, 3] = [.2, -.1, 5]
        target = adapter.transform(source, matrix)
        return source, target, matrix

    def test_observed_geometry_proof_has_disjoint_fit_validation_and_no_bbox_fit(self):
        import numpy as np

        source, target, matrix = self.geometry()
        initial = matrix.copy()
        initial[0, 3] += .015
        solved, pairs, qa = adapter.observed_correspondences(source, target, initial, .03, .95)
        np.testing.assert_allclose(solved, matrix, atol=1e-6)
        self.assertGreater(qa["validation"]["count"], 30)
        self.assertEqual(len(set(p["source_vertex_index"] for p in pairs)), len(pairs))
        self.assertEqual(len(set(p["target_gaussian_index"] for p in pairs)), len(pairs))
        self.assertEqual(qa["observed_coverage"], 1)

    def test_wrong_object_camera_initialization_fails_geometric_support(self):
        source, target, matrix = self.geometry()
        matrix[0, 3] += 10
        with self.assertRaisesRegex(adapter.RegistrationError, "support"):
            adapter.observed_correspondences(source, target, matrix, .03, .9)

    def test_original_camera_heldout_reprojection_and_mask_rejection(self):
        import numpy as np
        from PIL import Image

        source, target, matrix = self.geometry()
        _, pairs, _ = adapter.observed_correspondences(source, target, matrix, .03, .9)
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            Image.new("L", (200, 200), 255).save(task / "mask.png")
            pose = np.eye(4)
            pose[0, 3] = -.1
            cameras = {str(i): {"world_to_camera": p.tolist(), "intrinsics": [[100, 0, 100], [0, 100, 100], [0, 0, 1]], "width": 200, "height": 200} for i, p in enumerate((np.eye(4), pose))}
            observations = [{"frame_id": str(i), "mask_path": "mask.png"} for i in range(2)]
            qa = adapter.reprojection_gate(task, source, target, matrix, pairs, observations, cameras, 2)
            self.assertEqual(len(qa["frames"]), 2)
            displaced = matrix.copy()
            displaced[0, 3] += 1
            with self.assertRaisesRegex(adapter.RegistrationError, "reprojection"):
                adapter.reprojection_gate(task, source, target, displaced, pairs, observations, cameras, 2)
            Image.new("L", (200, 200), 0).save(task / "mask.png")
            with self.assertRaisesRegex(adapter.RegistrationError, "at least two"):
                adapter.reprojection_gate(task, source, target, matrix, pairs, observations, cameras, 2)

    def reprojection_case(self, shift, max_pixels):
        import numpy as np
        from PIL import Image

        source, target, matrix = self.geometry()
        _, pairs, _ = adapter.observed_correspondences(source, target, matrix, .03, .9)
        folder = tempfile.mkdtemp()
        task = Path(folder)
        Image.new("L", (200, 200), 255).save(task / "mask.png")
        cameras = {str(index): {"world_to_camera": np.eye(4).tolist(),
                                "intrinsics": [[100, 0, 100], [0, 100, 100], [0, 0, 1]],
                                "width": 200, "height": 200} for index in range(2)}
        observations = [{"frame_id": str(index), "mask_path": "mask.png"} for index in range(2)]
        drifted = matrix.copy()
        drifted[0, 3] += shift
        return adapter.reprojection_gate(task, source, target, drifted, pairs, observations, cameras, max_pixels)

    def test_a_reprojection_error_above_the_target_but_below_the_ceiling_is_recorded(self):
        # The fixture matrix carries a scale, so 0.35 world units lands the p95
        # between the 4 px target and the 16 px ceiling: reported, not fatal.
        qa = self.reprojection_case(0.35, 4)
        self.assertEqual(qa["policy"], "recorded_not_blocking")
        self.assertEqual(qa["max_p95_pixels"], 4)
        self.assertEqual(qa["rejection_ceiling_pixels"], 16)
        self.assertGreater(qa["worst_p95_pixels"], 4)
        self.assertLessEqual(qa["worst_p95_pixels"], 16)
        self.assertEqual(sorted(qa["above_target_frames"]), ["0", "1"])
        self.assertTrue(all(frame["above_target"] for frame in qa["frames"]))

    def test_a_reprojection_error_past_the_ceiling_is_still_refused(self):
        with self.assertRaisesRegex(adapter.RegistrationError, "rejection ceiling"):
            self.reprojection_case(1.0, 4)

    def registration_case(self, task, *, conditioning=False, geometry_overrides=None):
        """A complete solved registration task; returns the chain registration must find."""
        import numpy as np
        import trimesh
        from PIL import Image
        from plyfile import PlyData, PlyElement

        generated, conditioned, chain, _ = self.cameras()
        if conditioning:
            # the generated views carry the trajectory they were conditioned on
            generated = [pose.copy() for pose in conditioned]
            chain = np.eye(4)
        mesh = trimesh.creation.icosphere(subdivisions=2)
        mesh.vertices *= [1.3, .8, .5]
        mesh.visual.vertex_colors = [220, 80, 30, 255]
        mesh.export(task / "mesh.glb")
        source = adapter.scene_vertices(task / "mesh.glb")
        pose = {"scale": np.array([[1.4, 1.4, 1.4]]), "rotation": np.array([[1, 0, 0, 0]]), "translation": np.array([[.2, -.1, 5]])}
        np.savez(task / "pose.npz", **pose)
        expected = chain @ np.linalg.inv(generated[0]) @ adapter.glb_to_reference_camera(pose)
        target = adapter.transform(source, expected)
        table = np.zeros(len(target), dtype=[(name, "f4") for name in ("x", "y", "z", "f_dc_0", "opacity", "scale_0", "rot_0")])
        for axis, name in enumerate(("x", "y", "z")):
            table[name] = target[:, axis]
        table["rot_0"] = 1
        for name in ("observed.ply", "scene.ply"):
            PlyData([PlyElement.describe(table, "vertex")]).write(task / name)
        frames, provenance = [], []
        for orbit in range(3):
            path = task / f"conditioning_{orbit}.json"
            adapter.write_json(path, {"frames": [{"world_to_camera": p.tolist()} for p in conditioned[orbit * 4:orbit * 4 + 4]]})
            for position in range(4):
                index = orbit * 4 + position
                frames.append({"frame_id": f"{index:06d}", "stream3d_frame_name": f"frame_{index:06d}", "world_to_camera": generated[index].tolist()})
                provenance.append({"frame_id": f"{index:06d}", "orbit_id": f"orbit_{orbit}", "conditioning_cameras_path": path.name, "generated_frame_index": position})
        geometry = {"coordinate_frame": "colmap_world" if conditioning else "generated_DA3_world",
                    "unit": "colmap_reconstruction", "conditioning_poses_used": bool(conditioning), "frames": frames}
        if geometry_overrides:
            geometry.update(geometry_overrides)
        adapter.write_json(task / "geometry.json", geometry)
        adapter.write_json(task / "provenance.json", {"frames": provenance})
        adapter.write_json(task / "metadata.json", {"stage1_selected_crop_view_names": ["frame_000000.png"]})
        Image.new("L", (200, 200), 255).save(task / "mask.png")
        other = conditioned[0].copy()
        other[0, 3] -= .1
        cameras = {"coordinate_frame": "colmap_world", "units": "colmap_reconstruction", "frames": [{"frame_id": str(i), "world_to_camera": p.tolist(), "intrinsics": [[100, 0, 100], [0, 100, 100], [0, 0, 1]], "width": 200, "height": 200} for i, p in enumerate((conditioned[0], other))]}
        observations = [{"frame_id": str(i), "mask_path": "mask.png"} for i in range(2)]
        observed = {"object_id": "bed", "ply_path": "observed.ply", "coordinate_frame": "colmap_world", "units": "colmap_reconstruction", "association_status": "geometrically_verified", "observations": observations}
        item = {"object_id": "bed", "mesh_path": "mesh.glb", "mesh_sha256": adapter.sha256(task / "mesh.glb"), "coordinate_frame": "object_local", "unit": "asset_units", "normalization_metadata_path": "metadata.json", "backend_pose_path": "pose.npz"}
        documents = {"cameras": cameras, "isolated_object_ply": {"objects": [observed]}, "lifting_report": {"voxel_size": .01}, "completed_object_meshes": {"objects": [item]}, "completion_candidates": {"objects": [{"object_id": "bed", "generated_frame_geometry_path": "geometry.json", "sampled_frame_provenance_path": "provenance.json"}]}}
        envelope = {"module": "object_registration", "inputs": {"scene_gaussian_ply": {"path": "scene.ply", "evidence": "derived"}}}
        for name, document in documents.items():
            adapter.write_json(task / f"{name}.json", document)
            envelope["inputs"][name] = {"path": f"{name}.json", "evidence": "derived"}
        adapter.write_json(task / "inputs.json", envelope)
        args = argparse.Namespace(task_dir=task, inputs=task / "inputs.json", outputs=task / "outputs.json", max_camera_angle=12, min_observed_coverage=.7, max_reprojection_pixels=8, seed=0)
        return args, expected, observed, cameras

    def test_full_cpu_adapter_proof_is_accepted_by_recomposition_validator(self):
        import numpy as np

        recomposition_spec = importlib.util.spec_from_file_location("registration_recomposition_check", SCRIPT.with_name("scene_recomposition.py"))
        recomposition = importlib.util.module_from_spec(recomposition_spec)
        recomposition_spec.loader.exec_module(recomposition)
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            args, expected, observed, cameras = self.registration_case(task)
            result = adapter.run(args)
            self.assertEqual(result["status"], "accepted")
            record = adapter.read_json(task / adapter.read_json(args.outputs)["outputs"]["completed_object_meshes"]["path"])["objects"][0]
            actual, qa = recomposition.validate_registration(task, "bed", record, observed, cameras, adapter.sha256(task / "scene.ply"), .03, .2)
            np.testing.assert_allclose(actual["object_to_world"], expected, atol=2e-6)
            self.assertEqual(qa["status"], "numeric_correspondences_verified")
            self.assertEqual(record["room_alignment"], "observed_correspondence_verified")

    def test_a_declared_conditioning_trajectory_needs_no_camera_fit(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            args, expected, _, _ = self.registration_case(task, conditioning=True)
            with patch.object(adapter, "camera_chain", side_effect=AssertionError("declared poses must not be refitted")):
                result = adapter.run(args)
            self.assertEqual(result["status"], "accepted")
            record = adapter.read_json(task / adapter.read_json(args.outputs)["outputs"]["completed_object_meshes"]["path"])["objects"][0]
            chain = record["registration"]["camera_chain"]
            self.assertEqual(chain["method"], "conditioning_trajectory_declared")
            self.assertEqual(chain["policy"], "poses_are_the_conditioning_trajectory_of_the_registered_generated_views")
            self.assertEqual(chain["coordinate_frame"], "colmap_world")
            np.testing.assert_allclose(np.asarray(record["registration"]["object_to_world"]), expected, atol=2e-6)

    def test_conditioning_geometry_in_another_frame_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            args, _, _, _ = self.registration_case(task, conditioning=True,
                                                   geometry_overrides={"coordinate_frame": "generated_DA3_world"})
            with self.assertRaisesRegex(adapter.RegistrationError, "not in the observed object's declared frame"):
                adapter.run(args)


if __name__ == "__main__":
    unittest.main()
