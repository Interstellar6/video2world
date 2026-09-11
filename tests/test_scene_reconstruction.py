from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/scene_reconstruction.py"
SPEC = importlib.util.spec_from_file_location("scene_reconstruction_adapter", SCRIPT)
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


class SceneBoundaryTests(unittest.TestCase):
    def test_previous_recipe_supports_env_wrapped_provider_command(self):
        command = ["/usr/bin/env", "PYTHONDONTWRITEBYTECODE=1", "/python", str(SCRIPT),
                   "--task-dir", "{task_dir}", "--inputs", "{inputs_json}", "--outputs", "{outputs_json}",
                   "--da3-source", "/da3/src", "--da3-model", "/weights", "--pgsr-source", "/pgsr"]
        previous = adapter.previous_provider_arguments(command)
        self.assertEqual(previous.da3_source, Path("/da3/src"))
        self.assertEqual(previous.da3_resolution, 504)

    def test_camera_batches_cover_frames_with_shared_baseline(self):
        batches = adapter.camera_batches(50, 8)
        self.assertEqual(set(index for batch in batches for index in batch), set(range(50)))
        for batch in batches:
            self.assertEqual(batch[:2], [0, 49])
            self.assertLessEqual(len(batch), 8)
            self.assertGreaterEqual(len(batch), 3)
            self.assertEqual(len(batch), len(set(batch)))
        with self.assertRaises(adapter.ReconstructionError):
            adapter.camera_batches(3, 2)

    def test_view_selection_keeps_endpoints_without_duplicate_indices(self):
        self.assertEqual(adapter.select_indices(50, 0), list(range(50)))
        selected = adapter.select_indices(50, 8)
        self.assertEqual((selected[0], selected[-1]), (0, 49))
        self.assertEqual(len(set(selected)), 8)
        with self.assertRaises(adapter.ReconstructionError):
            adapter.select_indices(50, 2)

    def test_source_and_output_cannot_escape_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "task"
            task.mkdir()
            outside = root / "outside"
            outside.mkdir()
            (task / "link").symlink_to(outside, target_is_directory=True)
            for value in ("../outside", "link", "link/future"):
                with self.assertRaises(adapter.ReconstructionError):
                    adapter.task_path(task, value, exists=False)

    def test_pgsr_shorter_training_still_activates_plane_losses(self):
        args = argparse.Namespace(pgsr_iterations=6000, pgsr_resolution=2, pgsr_python=Path("/python"), pgsr_source=Path("/pgsr"))
        argv = adapter.pgsr_command(args, Path("/task/dataset"), Path("/task/model"))
        self.assertEqual(argv[argv.index("--iterations") + 1], "6000")
        self.assertLess(int(argv[argv.index("--single_view_weight_from_iter") + 1]), 6000)
        self.assertLess(int(argv[argv.index("--multi_view_weight_from_iter") + 1]), 6000)
        self.assertNotIn("--start_checkpoint", argv)

    def test_output_must_be_actual_nonempty_gaussian_ply(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "points.ply"
            path.write_text("ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n")
            with self.assertRaisesRegex(adapter.ReconstructionError, "Gaussian attributes"):
                adapter.validate_pgsr_ply(path)


NUMERICAL_DEPS = all(importlib.util.find_spec(name) is not None for name in ("numpy", "PIL"))


@unittest.skipUnless(NUMERICAL_DEPS, "requires numpy and Pillow")
class CameraDepthTests(unittest.TestCase):
    def test_phase_cache_rejects_changed_options_and_tampered_depth(self):
        import numpy as np
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory).resolve()
            stage = task / "stage"
            cached = stage / "run-original"
            cached.mkdir(parents=True)
            args = adapter.parser().parse_args([
                "--task-dir", str(task), "--inputs", str(stage / "inputs.json"), "--outputs", str(stage / "outputs.json"),
                "--da3-source", "/da3/src", "--da3-model", "/weights", "--pgsr-source", "/pgsr", "--resume-run-dir", str(cached),
            ])
            original = {"frame_id": "0", "intrinsics": np.eye(3).tolist(), "world_to_camera": np.eye(4).tolist()}
            np.save(cached / "depth.npy", np.ones((2, 2), dtype=np.float32))
            np.save(cached / "confidence.npy", np.ones((2, 2), dtype=np.float32))
            Image.new("RGB", (2, 2)).save(cached / "rgb.png")
            record = {**original, "width": 2, "height": 2, "image_to_depth": np.eye(3).tolist(),
                      "depth_path": "stage/run-original/depth.npy", "confidence_path": "stage/run-original/confidence.npy", "rgb_path": "stage/run-original/rgb.png",
                      "depth_sha256": adapter.sha256(cached / "depth.npy"), "confidence_sha256": adapter.sha256(cached / "confidence.npy")}
            adapter.write_json(cached / "frames.json", {"frames": [original]})
            adapter.write_json(cached / "cameras.json", {"frames": [original]})
            adapter.write_json(cached / "depth.json", {"frames": [record]})
            (cached / "scene_tsdf.glb").write_bytes(b"not loaded before cache validation")
            artifact = {"path": "inputs/source", "sha256": "source-digest", "evidence": "observed"}
            adapter.save_phase_cache(args, task, cached, artifact, [record])
            args.da3_resolution = 280
            with self.assertRaisesRegex(adapter.ReconstructionError, "source/options"):
                adapter.restore_depth_phase(args, task, stage, artifact, [original])
            args.da3_resolution = 504
            np.save(cached / "depth.npy", np.full((2, 2), 3.0, dtype=np.float32))
            with self.assertRaisesRegex(adapter.ReconstructionError, "hash mismatch"):
                adapter.restore_depth_phase(args, task, stage, artifact, [original])

    def test_camera_intrinsics_rejects_distortion_and_offcenter_pgsr_camera(self):
        camera = SimpleNamespace(id=1, model="PINHOLE", width=1280, height=720, params=[640, 682, 640, 360])
        self.assertEqual(adapter.camera_intrinsics(camera)[1, 1], 682)
        camera.model = "OPENCV"
        with self.assertRaisesRegex(adapter.ReconstructionError, "undistorted"):
            adapter.camera_intrinsics(camera)
        camera.model, camera.params = "PINHOLE", [640, 682, 500, 360]
        with self.assertRaisesRegex(adapter.ReconstructionError, "centered"):
            adapter.camera_intrinsics(camera)

    def test_depth_retains_processed_intrinsics_and_original_camera_pose(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory)
            extrinsic = np.eye(4)
            extrinsic[0, 3] = 1.5
            original = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=float)
            processed = np.array([[500, 0, 300], [0, 500, 170], [0, 0, 1]], dtype=float)
            record = {"frame_id": "frame_0", "intrinsics": original.tolist(), "world_to_camera": extrinsic.tolist()}
            prediction = SimpleNamespace(
                depth=np.full((1, 2, 3), 4.25), conf=np.ones((1, 2, 3)), intrinsics=processed[None],
                extrinsics=extrinsic[None, :3], processed_images=np.zeros((1, 2, 3, 3), dtype=np.uint8),
            )
            result = adapter.save_depth_batch(task, task / "depth", [record], prediction)[0]
            self.assertEqual((result["width"], result["height"]), (3, 2))
            np.testing.assert_allclose(result["intrinsics"], processed)
            np.testing.assert_allclose(result["world_to_camera"], extrinsic)
            np.testing.assert_allclose(result["image_to_depth"], [[0.5, 0, -20], [0, 0.5, -10], [0, 0, 1]])
            np.testing.assert_allclose(np.load(task / result["depth_path"]), 4.25)
            self.assertFalse(Path(result["depth_path"]).is_absolute())

    def test_unaligned_prediction_is_rejected_before_output(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory)
            record = {"frame_id": "frame_0", "intrinsics": np.eye(3).tolist(), "world_to_camera": np.eye(4).tolist()}
            predicted_pose = np.eye(4)
            predicted_pose[2, 3] = 8
            prediction = SimpleNamespace(
                depth=np.ones((1, 2, 3)), conf=np.ones((1, 2, 3)), intrinsics=np.eye(3)[None],
                extrinsics=predicted_pose[None], processed_images=np.zeros((1, 2, 3, 3), dtype=np.uint8),
            )
            with self.assertRaisesRegex(adapter.ReconstructionError, "not aligned"):
                adapter.save_depth_batch(task, task / "depth", [record], prediction)


TSDF_DEPS = all(importlib.util.find_spec(name) is not None for name in ("numpy", "PIL", "open3d", "trimesh"))


@unittest.skipUnless(TSDF_DEPS, "requires Open3D, trimesh, numpy and Pillow")
class TsdfGeometryTests(unittest.TestCase):
    def test_depth_prior_backprojects_into_source_world_and_keeps_real_rgb(self):
        import numpy as np
        import open3d as o3d
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory)
            depth = np.full((8, 8), 2.0, dtype=np.float32)
            depth[0, 0] = 1000
            np.save(task / "depth.npy", depth)
            np.save(task / "conf.npy", np.ones((8, 8), dtype=np.float32))
            Image.new("RGB", (8, 8), (210, 30, 10)).save(task / "rgb.png")
            extrinsic = np.eye(4)
            extrinsic[2, 3] = -1
            record = {"depth_path": "depth.npy", "confidence_path": "conf.npy", "rgb_path": "rgb.png", "intrinsics": [[8, 0, 4], [0, 8, 4], [0, 0, 1]], "world_to_camera": extrinsic.tolist()}
            args = argparse.Namespace(depth_trunc=20.0, confidence_percentile=20, pgsr_prior_voxel_size=0.01, pgsr_prior_max_points=10)
            receipt = adapter.build_depth_prior(args, task, [record], task / "prior.ply")
            cloud = o3d.io.read_point_cloud(str(task / "prior.ply"))
            self.assertEqual(len(cloud.points), 10)
            np.testing.assert_allclose(np.asarray(cloud.points)[:, 2], 3.0)
            np.testing.assert_allclose(np.asarray(cloud.colors), np.tile(np.array([210, 30, 10]) / 255.0, (10, 1)))
            self.assertEqual(receipt["method"], "backproject_camera_conditioned_da3")

    def test_tsdf_uses_camera_z_depth_and_supplied_world_to_camera(self):
        import numpy as np
        from PIL import Image
        import trimesh

        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory)
            records = []
            for index, camera_x in enumerate((-0.1, 0.0, 0.1)):
                np.save(task / f"depth-{index}.npy", np.full((32, 32), 2.0, dtype=np.float32))
                np.save(task / f"confidence-{index}.npy", np.ones((32, 32), dtype=np.float32))
                Image.new("RGB", (32, 32), (210, 30, 10)).save(task / f"rgb-{index}.png")
                extrinsic = np.eye(4)
                extrinsic[0, 3] = -camera_x
                extrinsic[2, 3] = -1.0
                records.append({
                    "depth_path": f"depth-{index}.npy", "confidence_path": f"confidence-{index}.npy",
                    "rgb_path": f"rgb-{index}.png", "width": 32, "height": 32,
                    "intrinsics": [[30, 0, 16], [0, 30, 16], [0, 0, 1]], "world_to_camera": extrinsic.tolist(),
                })
            args = argparse.Namespace(tsdf_voxel_size=0.05, tsdf_sdf_trunc=0.15, depth_trunc=5.0, confidence_percentile=20.0)
            path, receipt = adapter.fuse_tsdf(args, task, task, records)
            mesh = trimesh.load(path, force="mesh")
            self.assertGreater(len(mesh.faces), 0)
            np.testing.assert_allclose(mesh.vertices[:, 2], 3.0, atol=0.06)
            self.assertEqual(receipt["units"], "colmap_reconstruction")
            self.assertTrue(receipt["observation_only"])


if __name__ == "__main__":
    unittest.main()
