from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "orbit_video.py"
SPEC = importlib.util.spec_from_file_location("orbit_video_adapter", SCRIPT)
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


class OfficialInterfaceTests(unittest.TestCase):
    def assembly_fixture(self, task):
        (task / "bed.ply").write_bytes(b"hash-binding fixture, never rendered")
        observed = {"object_id": "bed", "association_status": "geometrically_verified", "ply_path": "bed.ply",
                    "sha256": adapter.sha256(task / "bed.ply"), "observations": [{"frame_id": "0"}]}
        documents = {"cameras": {"frames": []}, "isolated_object_ply": {"objects": [observed]},
                     "lifting_report": {"carve_performed": True, "generated_geometry_used": False, "accepted_object_ids": ["bed"]}}
        bindings = {}
        for role, document in documents.items():
            path = task / f"{role}.json"
            adapter.write_json(path, document)
            bindings[role] = {"path": path.name, "sha256": adapter.sha256(path)}
        assembled = {"cameras_path": "cameras.json", "objects": [{"object_id": "bed", "observed_geometry": observed,
                                                                   "views": [{"frame_id": "0"}]}]}
        report = {"status": "assembled_observed_views", "hidden_pixels_generated": False, "source_artifacts": bindings,
                  "objects": [{"object_id": "bed", "selected_frame_ids": ["0"]}]}
        return assembled, report

    def test_assembly_hash_bindings_accept_observed_geometry(self):
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder).resolve()
            assembled, report = self.assembly_fixture(task)
            bound = adapter.validate_assembly(task, assembled, report, task / "cameras.json")
            self.assertEqual(bound["bed:gaussians"]["sha256"], adapter.sha256(task / "bed.ply"))
            self.assertEqual(set(bound), {"cameras", "isolated_object_ply", "lifting_report", "bed:gaussians"})

    def test_assembly_rejects_replaced_nested_ply(self):
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder).resolve()
            assembled, report = self.assembly_fixture(task)
            (task / "bed.ply").write_bytes(b"replaced after assembly")
            with self.assertRaisesRegex(adapter.OrbitError, "Gaussian PLY: file hash mismatch"):
                adapter.validate_assembly(task, assembled, report, task / "cameras.json")

    def test_assembly_rejects_changed_cameras_or_override(self):
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder).resolve()
            assembled, report = self.assembly_fixture(task)
            with self.assertRaisesRegex(adapter.OrbitError, "camera binding"):
                adapter.validate_assembly(task, assembled, report, task / "different.json")
            adapter.write_json(task / "cameras.json", {"frames": ["modified"]})
            with self.assertRaisesRegex(adapter.OrbitError, "cameras: file hash mismatch"):
                adapter.validate_assembly(task, assembled, report, task / "cameras.json")

    def test_assembly_rejects_mixed_object_and_unverified_view(self):
        import copy

        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder).resolve()
            assembled, report = self.assembly_fixture(task)
            changed = copy.deepcopy(assembled)
            changed["objects"][0]["observed_geometry"]["units"] = "invented_metres"
            with self.assertRaisesRegex(adapter.OrbitError, "differs from accepted lifting"):
                adapter.validate_assembly(task, changed, report, task / "cameras.json")
            assembled["objects"][0]["views"][0]["frame_id"] = "new"
            report["objects"][0]["selected_frame_ids"] = ["new"]
            with self.assertRaisesRegex(adapter.OrbitError, "unverified observation"):
                adapter.validate_assembly(task, assembled, report, task / "cameras.json")

    def test_assembly_rejects_failed_report_and_missing_binding(self):
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder).resolve()
            assembled, report = self.assembly_fixture(task)
            report["status"] = "failed"
            with self.assertRaisesRegex(adapter.OrbitError, "completed observed-view"):
                adapter.validate_assembly(task, assembled, report, task / "cameras.json")
            report["status"] = "assembled_observed_views"
            del report["source_artifacts"]["lifting_report"]
            with self.assertRaisesRegex(adapter.OrbitError, "missing file hash binding"):
                adapter.validate_assembly(task, assembled, report, task / "cameras.json")

    def test_official_command_preserves_empty_clean_frame_indices(self):
        args = argparse.Namespace(fix_python=Path("/env/python"), source_root=Path("/models/fix-anything"), lora_path=Path("/models/lora.safetensors"), model_dir=Path("/models/base"), num_inference_steps=10, height=480, width=832, fps=15)
        command = adapter.fixanything_command(args, Path("/task/frames"), Path("/task/generated"), 9)
        self.assertEqual(command[:2], ["/env/python", "/models/fix-anything/scripts/run_inference.py"])
        self.assertEqual(command[command.index("--clean_frame_indices") + 1], "")
        self.assertEqual(command[command.index("--num_frames") + 1], "61")
        self.assertEqual(command[command.index("--seed") + 1], "9")
        args.clean_frame_indices = "0,60"
        command = adapter.fixanything_command(args, Path("/task/frames"), Path("/task/generated"), 20260829)
        self.assertEqual(command[command.index("--clean_frame_indices") + 1], "0 60")
        for invalid in ("0,61", "1,1", "first"):
            args.clean_frame_indices = invalid
            with self.assertRaises(adapter.OrbitError):
                adapter.fixanything_command(args, Path("/task/frames"), Path("/task/generated"), 1)

    @unittest.skipUnless(os.environ.get("FIXANYTHING_TEST_SOURCE"), "optional official FixAnything CLI conformance; no inference")
    def test_installed_official_cli_parses_clean_anchors_without_model_inference(self):
        source = Path(os.environ["FIXANYTHING_TEST_SOURCE"]) / "scripts/run_inference.py"
        tree = ast.parse(source.read_text())
        functions = [item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name in {"parse_args", "main"}]
        self.assertEqual({item.name for item in functions}, {"parse_args", "main"})
        fake_fix = MagicMock(return_value=([], []))
        namespace = {"argparse": argparse, "os": MagicMock(), "print": MagicMock(),
                     "load_frames": MagicMock(return_value=([None] * 61, None)),
                     "load_pipeline": MagicMock(), "fix_video": fake_fix,
                     "save_video": MagicMock(), "side_by_side": MagicMock()}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
        args = argparse.Namespace(fix_python=Path("/env/python"), source_root=source.parents[1],
                                  lora_path=Path("/models/lora.safetensors"), model_dir=Path("/models/base"),
                                  num_inference_steps=10, height=480, width=832, fps=15)
        for configured, expected in (("", []), ("0,60", [0, 60])):
            args.clean_frame_indices = configured
            command = adapter.fixanything_command(args, Path("/task/frames"), Path("/task/generated"), 20260829)
            with patch.object(sys, "argv", command[1:]):
                namespace["main"]()
            self.assertEqual(fake_fix.call_args.kwargs["clean_frame_indices"], expected)

    def test_missing_base_weights_blocks_before_renderer_or_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            task = root / "task"
            task.mkdir()
            adapter.write_json(task / "assembled.json", {"objects": [{"object_id": "bed"}]})
            adapter.write_json(task / "assembly_report.json", {})
            adapter.write_json(task / "inputs.json", {"inputs": {"assembled_object_views": {"path": "assembled.json"}, "assembly_report": {"path": "assembly_report.json"}}})
            source = root / "source"
            (source / "scripts").mkdir(parents=True)
            (source / "scripts/run_inference.py").write_text("# API fixture, never executed\n")
            lora = root / "lora.safetensors"
            lora.write_bytes(b"preflight fixture, not model weights")
            args = argparse.Namespace(task_dir=task, inputs=task / "inputs.json", outputs=task / "provider.json", source_root=source, model_dir=root / "missing-base", lora_path=lora)
            with patch.object(adapter, "run_command") as execute:
                with self.assertRaisesRegex(adapter.OrbitError, "preflight missing"):
                    adapter.run(args)
                execute.assert_not_called()
            report = adapter.read_json(next(task.glob("stages/orbit_video/run-*/orbit_video_report.json")))
            self.assertEqual(report["status"], "blocked_model_preflight")
            self.assertFalse(report["fixanything_invoked"])
            self.assertFalse((task / "provider.json").exists())

    def test_missing_numbered_checkpoint_shard_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "scripts").mkdir()
            (root / "scripts/run_inference.py").write_text("# not executed")
            lora = root / "lora.safetensors"
            lora.write_bytes(b"fixture")
            base = root / "models" / adapter.MODEL_ID
            (base / "google/umt5-xxl").mkdir(parents=True)
            for name in ("models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", "diffusion_pytorch_model-00001-of-00002.safetensors", "google/umt5-xxl/spiece.model"):
                (base / name).write_bytes(b"fixture")
            with self.assertRaisesRegex(adapter.OrbitError, "complete numbered"):
                adapter.check_model_files(root, root / "models", lora)

    def test_video_validation_decodes_and_rejects_static_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "generated.mp4"
            path.write_bytes(b"validation fixture, decoders mocked")
            args = argparse.Namespace(ffprobe="ffprobe", ffmpeg="ffmpeg", width=832, height=480, fps=15)
            probe = json.dumps({"streams": [{"width": 832, "height": 480, "nb_read_frames": "61", "avg_frame_rate": "15/1"}]})
            def result(stdout):
                return subprocess.CompletedProcess([], 0, stdout, "")
            with patch.object(adapter.subprocess, "run", side_effect=[result(probe), result("\n".join(f"0, {index}, 0, 1, 1024, same-checksum" for index in range(61)))]) as decoder:
                with self.assertRaisesRegex(adapter.OrbitError, "static"):
                    adapter.validate_video(path, args)
                self.assertIn("-count_frames", decoder.call_args_list[0].args[0])
            with patch.object(adapter.subprocess, "run", side_effect=[result(probe), result("\n".join(f"0, {index}, 0, 1, 1024, checksum-{index}" for index in range(61)))]):
                report = adapter.validate_video(path, args)
                self.assertEqual(report["decoded_frames"], 61)
                self.assertEqual(report["unique_decoded_frames"], 61)


GEOMETRY_DEPS = all(importlib.util.find_spec(name) is not None for name in ("numpy", "scipy", "plyfile"))


@unittest.skipUnless(GEOMETRY_DEPS, "requires numpy, scipy, and plyfile; no CUDA/model invocation")
class OrbitGeometryTests(unittest.TestCase):
    def test_three_axis_bases_are_orthonormal_distinct_great_circles(self):
        import numpy as np

        bases = adapter.three_axis_bases(np.array([0., 1., 0.]), np.array([0., 0., -1.]))
        self.assertEqual([name for name, _, _ in bases], list(adapter.THREE_AXIS_IDS))
        for name, axis_up, axis_reference in bases:
            self.assertAlmostEqual(float(np.linalg.norm(axis_up)), 1.0, places=9, msg=name)
            self.assertAlmostEqual(float(np.linalg.norm(axis_reference)), 1.0, places=9, msg=name)
            self.assertAlmostEqual(float(np.dot(axis_up, axis_reference)), 0.0, places=9, msg=name)
            side = np.cross(axis_up, axis_reference)
            points = np.array([np.cos(t) * axis_reference + np.sin(t) * side for t in np.linspace(0, 2 * np.pi, 13)[:-1]])
            self.assertTrue(np.allclose(np.linalg.norm(points, axis=1), 1.0, atol=1e-9), name)
            self.assertLess(float(np.abs(points @ axis_up).max()), 1e-9, name)
        self.assertEqual(len({(tuple(np.round(u, 6)), tuple(np.round(r, 6))) for _, u, r in bases}), 3)

    def test_real_adapter_run_connects_two_renders_to_three_fixanything_calls(self):
        import numpy as np
        from PIL import Image

        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder).resolve()
            assembled, report = OfficialInterfaceTests().assembly_fixture(task)
            observed = assembled["objects"][0]["observed_geometry"]
            observed.update(gaussian_count=2, coordinate_frame="colmap_world", units="colmap_reconstruction")
            pose = adapter.look_at(np.array([0., 0, 5.]), np.zeros(3), np.array([0., 1., 0.]))
            for role, document in (("isolated_object_ply", {"objects": [observed]}),
                                   ("cameras", {"frames": [{"frame_id": "0", "world_to_camera": pose.tolist()}]})):
                path = task / f"{role}.json"
                adapter.write_json(path, document)
                report["source_artifacts"][role]["sha256"] = adapter.sha256(path)
            adapter.write_json(task / "assembled.json", assembled)
            adapter.write_json(task / "assembly_report.json", report)
            adapter.write_json(task / "inputs.json", {"inputs": {"assembled_object_views": {"path": "assembled.json"},
                                                                    "assembly_report": {"path": "assembly_report.json"}}})
            args = argparse.Namespace(task_dir=task, inputs=task / "inputs.json", outputs=task / "outputs.json",
                source_root=task / "source", model_dir=task / "models", lora_path=task / "lora", cameras=None,
                up_vector=None, reference_direction=None, elevations=[10., 25., 40.], width=96, height=64,
                vertical_fov=40, framing_margin=1.1, render_python=Path("renderer"), render_python_path=[],
                fix_python=Path("fix-python"), seed=4, num_inference_steps=10, fps=15, clean_frame_indices="")
            commands = []
            def execute(command, directory, environment, log_path):
                commands.append(command)
                if "--render-request" in command:
                    request = adapter.read_json(Path(command[command.index("--render-request") + 1]))
                    result = {"source_gaussian_sha256": observed["sha256"], "orbits": []}
                    for orbit in request["orbits"]:
                        images, alphas = task / orbit["frames_dir"], task / orbit["alpha_dir"]
                        images.mkdir(parents=True)
                        alphas.mkdir(parents=True)
                        rows = []
                        for index in range(61):
                            image_path, alpha_path = images / f"{index:06d}.png", alphas / f"{index:06d}.png"
                            Image.new("RGB", (96, 64), (index, 0, 0)).save(image_path)
                            alpha = np.zeros((64, 96), dtype=np.uint8)
                            alpha[24:40, 40:56] = 255
                            Image.fromarray(alpha).save(alpha_path)
                            rows.append({"frame_id": f"{index:06d}", "image_path": adapter.relative(task, image_path),
                                         "image_sha256": adapter.sha256(image_path), "alpha_path": adapter.relative(task, alpha_path),
                                         "alpha_sha256": adapter.sha256(alpha_path)})
                        result["orbits"].append({"orbit_id": orbit["orbit_id"], "frames": rows})
                    adapter.write_json(task / request["report_path"], result)
                else:
                    output = Path(command[command.index("--output_dir") + 1]) / "generated.mp4"
                    output.write_text(command[command.index("--seed") + 1])
                return {"argv": command, "returncode": 0}
            def video(path, _args):
                return {"decoded_frames": 61, "unique_decoded_frames": 61, "sha256": adapter.sha256(path)}
            with patch.object(adapter, "check_model_files", return_value={}), patch.object(adapter, "gaussian_arrays",
                    return_value={"means": np.zeros((2, 3)), "center": np.zeros(3), "radius": 1.}), \
                    patch.object(adapter, "run_command", side_effect=execute), patch.object(adapter, "validate_video", side_effect=video):
                manifest = adapter.run(args)
            self.assertEqual(len(commands), 5)
            self.assertEqual(sum("--render-request" in command for command in commands), 2)
            clips = manifest["objects"][0]["orbits"]
            self.assertEqual(len(clips), 3)
            self.assertEqual({clip["elevation_degrees"] for clip in clips}, {10., 25., 40.})
            for clip in clips:
                self.assertIn("/framed/", clip["video_path"])
                self.assertEqual(adapter.sha256(task / clip["cameras_path"]), clip["conditioning_cameras_sha256"])
                self.assertGreater(adapter.read_json(task / clip["cameras_path"])["framing"]["zoom"], 1)
            self.assertEqual(set(adapter.read_json(args.outputs)["outputs"]), {"object_orbit_videos", "orbit_video_report"})
            rendered = adapter.read_json(task / manifest["objects"][0]["render_report_path"])
            bad_frame = task / rendered["orbits"][0]["frames"][0]["image_path"]
            bad_frame.write_bytes(b"changed after rendering")
            with self.assertRaisesRegex(adapter.OrbitError, "conditioning image: file hash mismatch"):
                adapter.validate_rendered_orbit(task, clips[0], rendered)

    def test_framing_uses_one_calibrated_zoom_without_changing_poses_or_geometry(self):
        import numpy as np
        from PIL import Image

        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder).resolve()
            request = {"gaussian_ply": "observed.ply", "gaussian_sha256": "retained", "report_path": "render_report.json", "orbits": []}
            (task / "render_report.json").write_text("{}")
            original = []
            calibration = [[100., 0, 80.], [0, 100., 48.], [0, 0, 1.]]
            for index in range(3):
                directory = task / f"orbit_{index}"
                (directory / "alpha").mkdir(parents=True)
                cameras = {"width": 160, "height": 96, "frames": []}
                for frame_id in ("000000", "000001"):
                    cameras["frames"].append({"frame_id": frame_id, "world_to_camera": np.eye(4).tolist(), "intrinsics": calibration})
                    alpha = np.zeros((96, 160), dtype=np.uint8)
                    alpha[35 - index:65, 60:100] = 255
                    Image.fromarray(alpha).save(directory / "alpha" / f"{frame_id}.png")
                path = directory / "cameras.json"
                adapter.write_json(path, cameras)
                digest = adapter.sha256(path)
                original.append((path, digest, cameras))
                request["orbits"].append({"orbit_id": directory.name, "cameras_path": adapter.relative(task, path),
                                          "conditioning_cameras_sha256": digest, "frames_dir": f"{directory.name}/frames",
                                          "alpha_dir": f"{directory.name}/alpha"})
            fitted, report = adapter.fit_conditioning_framing(task, request)
            self.assertGreater(report["zoom"], 2.)
            self.assertEqual(len(report["samples"]), 6)
            self.assertFalse(report["gaussians_modified"])
            self.assertFalse(report["poses_modified"])
            self.assertEqual(fitted["gaussian_sha256"], "retained")
            for orbit, (source_path, digest, source) in zip(fitted["orbits"], original):
                self.assertEqual(adapter.sha256(source_path), digest)
                path = task / orbit["cameras_path"]
                self.assertEqual(adapter.sha256(path), orbit["conditioning_cameras_sha256"])
                document = adapter.read_json(path)
                for actual, expected in zip(document["frames"], source["frames"]):
                    np.testing.assert_allclose(actual["world_to_camera"], expected["world_to_camera"])
                    self.assertAlmostEqual(actual["intrinsics"][0][0] / 100, report["zoom"])
                    self.assertEqual(actual["intrinsics"][0][2], 80.)
                    self.assertEqual(actual["intrinsics"][1][2], 48.)
                self.assertFalse((task / orbit["frames_dir"]).exists())

    def test_framing_rejects_empty_visible_support(self):
        import numpy as np
        from PIL import Image

        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder).resolve()
            (task / "alpha").mkdir()
            Image.fromarray(np.zeros((96, 160), dtype=np.uint8)).save(task / "alpha/000000.png")
            adapter.write_json(task / "cameras.json", {"width": 160, "height": 96, "frames": [{"frame_id": "000000"}]})
            request = {"orbits": [{"cameras_path": "cameras.json", "conditioning_cameras_sha256": adapter.sha256(task / "cameras.json"),
                                     "alpha_dir": "alpha", "orbit_id": "empty"}]}
            with self.assertRaisesRegex(adapter.OrbitError, "insufficient visible support"):
                adapter.fit_conditioning_framing(task, request)

    def test_render_uses_dense_single_camera_background_contract(self):
        import numpy as np

        fake_torch, fake_gsplat = MagicMock(), MagicMock()
        rendered, alphas = MagicMock(), MagicMock()
        rendered[0].detach().cpu().numpy.return_value = np.ones((16, 16, 3))
        alpha = np.zeros((16, 16))
        alpha[4:12, 4:12] = 1
        alphas[0, :, :, 0].detach().cpu().numpy.return_value = alpha
        fake_gsplat.rasterization.return_value = rendered, alphas, {}
        arrays = {name: np.zeros((2, 3)) for name in ("means", "quats", "scales", "opacities", "colors")}
        arrays.update(sh_degree=3, source_attributes=["x", "y", "z"])
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder).resolve()
            (task / "source.ply").write_bytes(b"mock geometry")
            adapter.write_json(task / "cameras.json", {
                "width": 16, "height": 16, "near_plane": .1, "far_plane": 10,
                "frames": [{"frame_id": "000000", "world_to_camera": np.eye(4).tolist(), "intrinsics": np.eye(3).tolist()}],
            })
            request = task / "request.json"
            adapter.write_json(request, {"gaussian_ply": "source.ply", "report_path": "report.json", "orbits": [
                {"orbit_id": "test", "cameras_path": "cameras.json", "conditioning_cameras_sha256": adapter.sha256(task / "cameras.json"), "frames_dir": "frames", "alpha_dir": "alpha"},
            ]})
            with patch.dict(sys.modules, {"torch": fake_torch, "gsplat": fake_gsplat}), patch.object(adapter, "gaussian_arrays", return_value=arrays):
                report = adapter.render_worker(task, request)
            self.assertIs(fake_gsplat.rasterization.call_args.kwargs["packed"], False)
            self.assertEqual(fake_torch.ones.call_args.args[0], (1, 3))
            self.assertEqual(report["orbits"][0]["frames"][0]["foreground_pixels"], 64)

    def test_three_full_camera_turns_and_normalization_roundtrip(self):
        import numpy as np

        center = np.array([2.5, -3., 1.2])
        up = adapter.normalized([.3, .9, .1], "up")
        reference = adapter.normalized(np.array([1., 0, 0]) - up * up[0], "reference")
        side = np.cross(up, reference)
        trajectories = []
        for elevation in (10, 25, 40):
            camera_manifest = adapter.trajectory(center, .4, up, reference, elevation, 832, 480, 40, 1.1)
            trajectories.append(camera_manifest)
            frames = camera_manifest["frames"]
            self.assertEqual(len(frames), 61)
            np.testing.assert_allclose(frames[0]["world_to_camera"], frames[-1]["world_to_camera"], atol=1e-10)
            centers = np.array([frame["camera_center_world"] for frame in frames])
            self.assertEqual(len(np.unique(centers[:-1].round(8), axis=0)), 60)
            directions = (centers - center) / camera_manifest["distance"]
            angles = np.unwrap(np.arctan2(directions @ side, directions @ reference))
            self.assertAlmostEqual(np.degrees(angles[-1] - angles[0]), 360)
            np.testing.assert_allclose(np.degrees(np.arcsin(directions @ up)), elevation)
            world_to_object = np.array(camera_manifest["world_to_object"])
            object_to_world = np.array(camera_manifest["object_to_world"])
            np.testing.assert_allclose(world_to_object @ object_to_world, np.eye(4), atol=1e-10)
            world_point = np.append(center + [.02, -.01, .03], 1)
            object_point = world_to_object @ world_point
            for frame in frames:
                pose = np.array(frame["world_to_camera"])
                normalized_pose = np.array(frame["normalized_object_to_camera"])
                calibration = np.array(frame["intrinsics"])
                target_camera = pose @ np.append(center, 1)
                self.assertGreater(target_camera[2], 0)
                np.testing.assert_allclose((calibration @ target_camera[:3])[:2] / target_camera[2], [416, 240])
                camera_point = pose @ world_point
                normalized_camera_point = normalized_pose @ object_point
                np.testing.assert_allclose(camera_point[:3] / camera_point[2], normalized_camera_point[:3] / normalized_camera_point[2])
        self.assertEqual(len({json.dumps(item["frames"][0]["camera_center_world"]) for item in trajectories}), 3)

    def test_up_and_azimuth_are_derived_from_observed_camera(self):
        import numpy as np

        center = np.zeros(3)
        pose = adapter.look_at(np.array([0., 0, 5.]), center, np.array([0., 1., 0.]))
        obj = {"views": [{"frame_id": "source"}]}
        cameras = {"frames": [{"frame_id": "source", "world_to_camera": pose.tolist()}]}
        up, reference, report = adapter.orbit_basis(obj, cameras, center)
        np.testing.assert_allclose(up, [0, 1, 0])
        np.testing.assert_allclose(reference, [0, 0, 1])
        self.assertIn("camera_up", report["up_method"])
        with self.assertRaisesRegex(adapter.OrbitError, "up requires"):
            adapter.orbit_basis({}, {}, center)

    def test_pgsr_quaternions_and_sh_channels_are_preserved(self):
        import numpy as np
        from plyfile import PlyData, PlyElement

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "object.ply"
            fields = ["x", "y", "z", "opacity", *(f"scale_{i}" for i in range(3)), *(f"rot_{i}" for i in range(4)), *(f"f_dc_{i}" for i in range(3)), *(f"f_rest_{i}" for i in range(45))]
            rows = np.zeros(2, dtype=[(name, "f4") for name in fields])
            rows["x"] = [5, 6]
            rows["y"] = [-2, -2]
            rows["z"] = [3, 3]
            rows["rot_0"], rows["rot_2"] = np.sqrt(.5), np.sqrt(.5)
            for index in range(3):
                rows[f"scale_{index}"] = np.log(.01 * (index + 1))
            for index in range(45):
                rows[f"f_rest_{index}"] = index
            PlyData([PlyElement.describe(rows, "vertex")]).write(str(path))
            arrays = adapter.gaussian_arrays(path)
            np.testing.assert_allclose(arrays["means"], [[5, -2, 3], [6, -2, 3]])
            np.testing.assert_allclose(arrays["quats"][0], [np.sqrt(.5), 0, np.sqrt(.5), 0])
            np.testing.assert_allclose(arrays["colors"][0, 1], [0, 15, 30])
            np.testing.assert_allclose(arrays["scales"][0], [.01, .02, .03])
            np.testing.assert_allclose(arrays["opacities"], [.5, .5])
            self.assertEqual(arrays["sh_degree"], 3)


if __name__ == "__main__":
    unittest.main()
