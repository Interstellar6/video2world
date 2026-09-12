from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "geometry_completion.py"
SPEC = importlib.util.spec_from_file_location("geometry_completion_adapter", SCRIPT)
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


def orbit_fixture(task):
    import numpy as np

    spec = importlib.util.spec_from_file_location("orbit_for_completion_test", SCRIPT.with_name("orbit_video.py"))
    orbit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(orbit)
    clips = []
    for index, elevation in enumerate((10, 25, 40)):
        orbit_id = f"orbit_{index}"
        video, path = task / f"{orbit_id}.mp4", task / f"{orbit_id}.json"
        video.write_bytes(f"not_a_video_{index}".encode())
        camera = orbit.trajectory(np.zeros(3), 1, np.array([0., 1., 0.]), np.array([0., 0., -1.]), elevation, 64, 64, 40, 1.1)
        camera.update({"object_id": "bed", "orbit_id": orbit_id, "coordinate_frame": "colmap_world", "units": "colmap_reconstruction", "basis": {"up_vector": [0, 1, 0], "reference_direction": [0, 0, -1]}})
        adapter.write_json(path, camera)
        clips.append({"orbit_id": orbit_id, "video_path": video.name, "video_sha256": adapter.sha256(video), "cameras_path": path.name, "conditioning_cameras_sha256": adapter.sha256(path), "frame_count": 61, "elevation_degrees": elevation, "azimuth_span_degrees": 360, "width": 64, "height": 64, "fps": 15, "evidence": "conditioned_synthesis", "camera_pose_status": "conditioning_trajectory_not_verified_for_generated_pixels"})
    return {"object_id": "bed", "orbits": clips}


class CompletionContractTests(unittest.TestCase):
    def test_native_sam3i_layout_and_backend_are_explicit(self):
        source = Path("/official/SAM3-I")
        self.assertEqual(adapter.sam3_builder_path(source, "sam3i"), source / "sam3/sam3/model_builder.py")
        self.assertEqual(adapter.sam3_builder_path(source, "sam3"), source / "sam3/model_builder.py")
        args = SimpleNamespace(sam3_python=Path("/env/python"), sam3_source=source, sam3_checkpoint=Path("/models/native.pt"),
            sam3_backend="sam3i", sam3_instruction_stage="complex", sam3_confidence=.25)
        command = adapter.sam3_command(args, Path("/task"), Path("/task/input.json"), Path("/task/output.json"))
        self.assertEqual(command[command.index("--backend") + 1], "sam3i")
        self.assertEqual(command[command.index("--instruction-stage") + 1], "complex")

    def test_orbit_sampling_covers_turn_without_duplicate_closing_frame(self):
        self.assertEqual(adapter.sample_indices(61, 4), [0, 15, 30, 45])
        self.assertEqual(adapter.sample_indices(61, 8), [0, 7, 15, 22, 30, 37, 45, 52])
        with self.assertRaises(adapter.CompletionError):
            adapter.sample_indices(61, 61)
        with self.assertRaisesRegex(adapter.CompletionError, "heldout"):
            adapter.sample_indices(61, 3)

    def test_stream3d_chunk_schedule_matches_upstream_final_chunk(self):
        self.assertEqual(adapter.chunk_starts(8), [0])
        self.assertEqual(adapter.chunk_starts(12), [0, 4])
        self.assertEqual(adapter.chunk_starts(24), [0, 6, 12, 16])

    def test_duplicate_objects_cannot_silently_replace_the_binding(self):
        with self.assertRaisesRegex(adapter.CompletionError, "duplicate"):
            adapter.unique_objects([{"object_id": "bed"}, {"object_id": "bed"}], "assembly")

    def test_fresh_depth_inference_does_not_receive_conditioning_cameras(self):
        model = Mock()
        prediction = object()
        model.inference.return_value = prediction
        self.assertIs(adapter.predict_generated_depth(model, ["a.png", "b.png"], 504), prediction)
        keywords = model.inference.call_args.kwargs
        self.assertIsNone(keywords["extrinsics"])
        self.assertIsNone(keywords["intrinsics"])
        self.assertFalse(keywords["align_to_input_ext_scale"])
        self.assertIsNone(keywords["export_dir"])

    def test_official_stream3d_command_owns_output_and_hydra_paths(self):
        args = argparse.Namespace(stream3d_python=Path("/env/python"), stage1_steps=4, stage2_steps=25, seed=0)
        command = adapter.stream3d_command(args, Path("/task/dataset/bed"), Path("/task/results"), Path("/task/pipeline.yaml"), Path("/task/hydra"))
        self.assertEqual(command[:4], ["/env/python", "-m", "streaming.runner", "backend=sam3d"])
        self.assertIn("chunk_indices=[-1]", command)
        self.assertIn("camera_pose_source=da3", command)
        self.assertIn('hydra.run.dir="/task/hydra"', command)
        self.assertIn('output_root="/task/results"', command)
        self.assertFalse(any("run_stream3d.sh" in item or "vnext" in item for item in command))


GEOMETRY_DEPS = all(importlib.util.find_spec(name) is not None for name in ("numpy", "PIL", "trimesh"))
RASTER_DEPS = importlib.util.find_spec("cv2") is not None


@unittest.skipUnless(GEOMETRY_DEPS, "requires CPU geometry libraries; no model inference")
class GeneratedInputTests(unittest.TestCase):
    def test_three_calibrated_elevations_are_validated_without_claiming_generated_poses(self):
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            obj = orbit_fixture(task)
            result = adapter.validate_orbit_inputs(task, obj)
            self.assertEqual(len(result), 3)
            self.assertEqual([item["camera"]["elevation_degrees"] for item in result], [10, 25, 40])

    def test_three_axis_orthogonal_bases_are_accepted_but_repeated_orbits_are_not(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            spec = importlib.util.spec_from_file_location("orbit_for_completion_test", SCRIPT.with_name("orbit_video.py"))
            orbit = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(orbit)
            clips = []
            for index, (name, axis_up, axis_reference) in enumerate(orbit.three_axis_bases(np.array([0., 1., 0.]), np.array([0., 0., -1.]))):
                orbit_id = f"orbit_{index}"
                video, path = task / f"{orbit_id}.mp4", task / f"{orbit_id}.json"
                video.write_bytes(f"axis_video_{index}".encode())
                camera = orbit.trajectory(np.zeros(3), 1, axis_up, axis_reference, 0.0, 64, 64, 40, 1.1)
                camera.update({"object_id": "bed", "orbit_id": orbit_id, "trajectory_id": name,
                               "coordinate_frame": "colmap_world", "units": "colmap_reconstruction",
                               "basis": {"up_vector": list(map(float, axis_up)), "reference_direction": list(map(float, axis_reference))}})
                adapter.write_json(path, camera)
                clips.append({"orbit_id": orbit_id, "video_path": video.name, "video_sha256": adapter.sha256(video),
                              "cameras_path": path.name, "conditioning_cameras_sha256": adapter.sha256(path),
                              "frame_count": 61, "elevation_degrees": 0, "azimuth_span_degrees": 360,
                              "width": 64, "height": 64, "fps": 15, "evidence": "conditioned_synthesis",
                              "camera_pose_status": "conditioning_trajectory_not_verified_for_generated_pixels"})
            result = adapter.validate_orbit_inputs(task, {"object_id": "bed", "orbits": clips})
            self.assertEqual(len(result), 3)
            self.assertEqual(len({tuple(item["camera"]["basis"]["up_vector"]) for item in result}), 3)
            repeated = {"object_id": "bed", "orbits": [dict(clips[0]), dict(clips[0]), dict(clips[2])]}
            with self.assertRaisesRegex(adapter.CompletionError, "distinct"):
                adapter.validate_orbit_inputs(task, repeated)

    def test_shared_framing_zoom_preserves_trajectory_but_per_orbit_intrinsics_cannot_drift(self):
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            obj = orbit_fixture(task)
            for clip in obj["orbits"]:
                path = task / clip["cameras_path"]
                camera = adapter.read_json(path)
                for frame in camera["frames"]:
                    frame["intrinsics"][0][0] *= 2.37
                    frame["intrinsics"][1][1] *= 2.37
                adapter.write_json(path, camera)
                clip["conditioning_cameras_sha256"] = adapter.sha256(path)
            self.assertEqual(len(adapter.validate_orbit_inputs(task, obj)), 3)
            clip = obj["orbits"][2]
            path = task / clip["cameras_path"]
            camera = adapter.read_json(path)
            for frame in camera["frames"]:
                frame["intrinsics"][0][0] *= 1.1
            adapter.write_json(path, camera)
            clip["conditioning_cameras_sha256"] = adapter.sha256(path)
            with self.assertRaisesRegex(adapter.CompletionError, "share one raster calibration"):
                adapter.validate_orbit_inputs(task, obj)

    def test_unbound_changed_and_escaping_conditioning_cameras_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            obj = orbit_fixture(task)
            for change in (lambda c: c.pop("conditioning_cameras_sha256"), lambda c: c.update(conditioning_cameras_sha256="0" * 64), lambda c: c.update(cameras_path="../outside.json")):
                broken = copy.deepcopy(obj)
                change(broken["orbits"][0])
                with self.assertRaises((adapter.CompletionError, ValueError)):
                    adapter.validate_orbit_inputs(task, broken)

    def test_declared_elevation_is_not_proof_of_actual_camera_motion(self):
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            obj = orbit_fixture(task)
            clip = obj["orbits"][1]
            path = task / clip["cameras_path"]
            camera = adapter.read_json(path)
            camera["frames"] = adapter.read_json(task / obj["orbits"][0]["cameras_path"])["frames"]
            adapter.write_json(path, camera)
            clip["conditioning_cameras_sha256"] = adapter.sha256(path)
            with self.assertRaisesRegex(adapter.CompletionError, "pose does not realize"):
                adapter.validate_orbit_inputs(task, obj)

    def test_duplicate_elevations_and_nonclosing_trajectory_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            obj = orbit_fixture(task)
            duplicate = copy.deepcopy(obj)
            duplicate["orbits"][1]["elevation_degrees"] = 10
            with self.assertRaisesRegex(adapter.CompletionError, "distinct"):
                adapter.validate_orbit_inputs(task, duplicate)
            clip = obj["orbits"][0]
            path = task / clip["cameras_path"]
            camera = adapter.read_json(path)
            camera["frames"][-1]["azimuth_degrees"] = 354
            adapter.write_json(path, camera)
            clip["conditioning_cameras_sha256"] = adapter.sha256(path)
            with self.assertRaisesRegex(adapter.CompletionError, "ordered uniform"):
                adapter.validate_orbit_inputs(task, obj)

    def test_orbits_cannot_silently_switch_object_units_or_scene_origin(self):
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            obj = orbit_fixture(task)
            path = task / obj["orbits"][2]["cameras_path"]
            camera = adapter.read_json(path)
            camera["units"] = "meters"
            adapter.write_json(path, camera)
            obj["orbits"][2]["conditioning_cameras_sha256"] = adapter.sha256(path)
            with self.assertRaisesRegex(adapter.CompletionError, "share one object"):
                adapter.validate_orbit_inputs(task, obj)

    def test_observed_anchor_and_geometry_must_bind_the_actual_lifted_object(self):
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            obj = orbit_fixture(task)
            ply = task / "observed.ply"
            ply.write_bytes(b"fixture observed geometry")
            observed = {"ply_path": ply.name, "sha256": adapter.sha256(ply), "association_status": "geometrically_verified", "coordinate_frame": "colmap_world", "units": "colmap_reconstruction", "observed_anchor": {"position": [1, 2, 3]}}
            obj["observed_geometry"] = copy.deepcopy(observed)
            assembled = {"observed_geometry": copy.deepcopy(observed)}
            lifting = {"accepted_object_ids": ["bed"]}
            adapter.validate_observed_input(task, "bed", obj, assembled, observed, lifting)
            assembled["observed_geometry"]["observed_anchor"]["position"][0] = 8
            with self.assertRaisesRegex(adapter.CompletionError, "bindings disagree"):
                adapter.validate_observed_input(task, "bed", obj, assembled, observed, lifting)
            with self.assertRaisesRegex(adapter.CompletionError, "accepted lifted"):
                adapter.validate_observed_input(task, "bed", obj, assembled, observed, {"accepted_object_ids": []})

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "requires CPU video codecs")
    def test_decodes_actual_three_clips_at_requested_azimuth_indices(self):
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            obj = orbit_fixture(task)
            for index, clip in enumerate(obj["orbits"]):
                path = task / clip["video_path"]
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=15", "-vf", f"hue=h={index * 90}", "-frames:v", "61", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True, capture_output=True)
                clip["video_sha256"] = adapter.sha256(path)
            output = task / "preprocess"
            output.mkdir()
            args = argparse.Namespace(task_dir=task, ffmpeg="ffmpeg", ffprobe="ffprobe", views_per_orbit=4)
            frames_path, _, provenance = adapter.decode_orbits(args, obj, output, "bed")
            frames = adapter.read_json(frames_path)["frames"]
            self.assertEqual(len(frames), 12)
            self.assertEqual([record["generated_frame_index"] for record in provenance], [0, 15, 30, 45] * 3)
            self.assertTrue(all((task / frame["image_path"]).is_file() for frame in frames))
            self.assertTrue(all(record["conditioning_cameras_used_as_geometry"] is False for record in provenance))
            self.assertTrue(all(record["conditioning_cameras_sha256"] for record in provenance))
            self.assertTrue(all(adapter.sha256(task / record["image_path"]) == record["image_sha256"] for record in frames))

    def test_decoded_raster_and_fps_must_match_calibration(self):
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            obj = orbit_fixture(task)
            args = argparse.Namespace(task_dir=task, ffmpeg="ffmpeg", ffprobe="ffprobe", views_per_orbit=4)
            for stream in ({"nb_read_frames": "61", "width": 32, "height": 64, "avg_frame_rate": "15/1"}, {"nb_read_frames": "61", "width": 64, "height": 64, "avg_frame_rate": "30/1"}):
                probe = SimpleNamespace(returncode=0, stdout=json.dumps({"streams": [stream]}), stderr="")
                with patch.object(adapter.subprocess, "run", return_value=probe), self.assertRaisesRegex(adapter.CompletionError, "raster|FPS"):
                    adapter.decode_orbits(args, obj, task, "bed")

    def test_static_sampled_pixels_and_camera_change_during_decode_are_rejected(self):
        from PIL import Image

        for failure in ("static", "changed_camera"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as folder:
                task = Path(folder)
                obj = orbit_fixture(task)
                args = argparse.Namespace(task_dir=task, ffmpeg="ffmpeg", ffprobe="ffprobe", views_per_orbit=4)
                probe = SimpleNamespace(returncode=0, stdout=json.dumps({"streams": [{"nb_read_frames": "61", "width": 64, "height": 64, "avg_frame_rate": "15/1"}]}), stderr="")
                def decode(command, *_):
                    directory = Path(command[-1]).parent
                    for index in range(4):
                        Image.new("RGB", (64, 64), (index * 20 if failure != "static" else 20, 10, 10)).save(directory / f"{index:06d}.png")
                    if failure == "changed_camera":
                        path = task / obj["orbits"][0]["cameras_path"]
                        path.write_bytes(path.read_bytes() + b" ")
                    return {"returncode": 0}
                with patch.object(adapter.subprocess, "run", return_value=probe), patch.object(adapter, "execute", side_effect=decode), self.assertRaisesRegex(adapter.CompletionError, "static image|changed during decoding"):
                    adapter.decode_orbits(args, obj, task, "bed")

    def fixture(self, task, count=3):
        import numpy as np
        from PIL import Image

        frames, masks = [], []
        for index in range(count):
            frame_id = str(index)
            image_path, mask_path = task / f"rgb_{index}.png", task / f"mask_{index}.png"
            Image.new("RGB", (16, 16), (100, 80, 40)).save(image_path)
            mask = np.zeros((16, 16), dtype=np.uint8)
            mask[2:14, 2:14] = 255
            Image.fromarray(mask).save(mask_path)
            frames.append({"frame_id": frame_id, "image_path": image_path.name, "image_sha256": adapter.sha256(image_path), "width": 16, "height": 16})
            masks.append({"frame_id": frame_id, "object_id": "bed", "component_id": "__object__", "image_path": image_path.name, "mask_path": mask_path.name, "mask_sha256": adapter.sha256(mask_path)})
        adapter.write_json(task / "frames.json", {"frames": frames})
        adapter.write_json(task / "masks.json", {"masks": masks})
        request = {"object_id": "bed", "frames_manifest_path": "frames.json", "frames_manifest_sha256": adapter.sha256(task / "frames.json"), "masks_manifest_path": "masks.json", "masks_manifest_sha256": adapter.sha256(task / "masks.json"), "dataset_root": "dataset/bed", "confidence_percentile": 20}
        poses = np.repeat(np.eye(4)[None], count, axis=0)
        poses[:, 0, 3] = np.arange(count) * -.2
        prediction = SimpleNamespace(depth=np.full((count, 8, 8), 2.), conf=np.ones((count, 8, 8)), intrinsics=np.repeat(np.array([[[4., 0, 4], [0, 4., 4], [0, 0, 1]]]), count, axis=0), extrinsics=poses, processed_images=np.full((count, 8, 8, 3), 100, dtype=np.uint8))
        mappings = np.repeat(np.diag([.5, .5, 1])[None], count, axis=0)
        return request, prediction, mappings

    @unittest.skipUnless(RASTER_DEPS, "requires OpenCV for official DA3 raster materialization")
    def test_real_stream3d_dataset_materialization_preserves_estimated_pose_and_raster(self):
        import numpy as np
        from PIL import Image

        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            request, prediction, mappings = self.fixture(task)
            result = adapter.save_generated_geometry(task, request, prediction, mappings)
            self.assertFalse(result["conditioning_poses_used"])
            self.assertFalse(result["camera_conditioned"])
            self.assertEqual(result["room_alignment"], "unknown")
            serialized_poses = np.loadtxt(task / result["camera_poses_path"]).reshape(3, 4, 4)
            np.testing.assert_allclose(serialized_poses, prediction.extrinsics)
            for frame in result["frames"]:
                with np.load(task / frame["depth_npz_path"]) as output:
                    self.assertEqual(output["depth"].shape, (8, 8))
                    np.testing.assert_allclose(output["intrinsics"], prediction.intrinsics[0])
                with Image.open(task / frame["image_path"]) as image:
                    self.assertEqual(image.size, (8, 8))
                with Image.open(task / frame["mask_path"]) as image:
                    mask = np.asarray(image)
                self.assertEqual(int((mask > 0).sum()), 36)

    @unittest.skipUnless(RASTER_DEPS and os.environ.get("STREAM3D_TEST_SOURCE"), "optional installed official Stream3D dataset conformance check")
    def test_materialized_dataset_is_accepted_by_official_stream3d_loader(self):
        import numpy as np

        sys.path.insert(0, os.environ["STREAM3D_TEST_SOURCE"])
        from streaming.data.data_gso import DataGSO, DataGSOConfig
        from streaming.utils.streaming_da3 import load_da3

        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            request, prediction, mappings = self.fixture(task, count=8)
            adapter.save_generated_geometry(task, request, prediction, mappings)
            dataset = task / "dataset/bed"
            config = DataGSOConfig(name="gso", roots=[dataset], render_split="render_spiral_100", image_dir_name="images", mask_dir_name="masks", da3_dir_name="da3", chunk_size=8, chunk_overlap=2, chunk_indices=[-1])
            example = next(iter(DataGSO(config)))
            self.assertEqual(len(example.image_files), 8)
            loaded = load_da3(example.da3_root, image_files=example.image_files, camera_pose_source="da3")
            self.assertEqual(loaded["pointmaps_sam3d"].shape, (8, 3, 8, 8))
            np.testing.assert_allclose(loaded["extrinsics"], prediction.extrinsics)
            np.testing.assert_allclose(loaded["pointmaps_sam3d"][0, :, 0, 0], [-2, -2, 2])

    def test_missing_generated_frame_mask_blocks_backend_input(self):
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            request, prediction, mappings = self.fixture(task)
            adapter.write_json(task / "masks.json", {"masks": []})
            request["masks_manifest_sha256"] = adapter.sha256(task / "masks.json")
            with self.assertRaisesRegex(adapter.CompletionError, "no accepted"):
                adapter.generated_mask_inputs(task, request)
            self.assertFalse((task / "dataset/bed/input_manifest.json").exists())

    def test_duplicate_generated_parent_masks_cannot_overwrite_each_other(self):
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            request, _, _ = self.fixture(task)
            masks = adapter.read_json(task / "masks.json")
            masks["masks"].append(copy.deepcopy(masks["masks"][0]))
            adapter.write_json(task / "masks.json", masks)
            request["masks_manifest_sha256"] = adapter.sha256(task / "masks.json")
            with self.assertRaisesRegex(adapter.CompletionError, "unambiguous parent"):
                adapter.generated_mask_inputs(task, request)

    def test_generated_mask_hash_and_original_raster_are_mandatory(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            request, _, _ = self.fixture(task)
            Image.new("L", (8, 8), 255).save(task / "mask_0.png")
            with self.assertRaisesRegex(adapter.CompletionError, "SHA256"):
                adapter.generated_mask_inputs(task, request)
            masks = adapter.read_json(task / "masks.json")
            masks["masks"][0]["mask_sha256"] = adapter.sha256(task / "mask_0.png")
            adapter.write_json(task / "masks.json", masks)
            request["masks_manifest_sha256"] = adapter.sha256(task / "masks.json")
            with self.assertRaisesRegex(adapter.CompletionError, "original generated RGB raster"):
                adapter.generated_mask_inputs(task, request)

    def test_generated_image_change_is_rejected_before_depth_inference(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            request, _, _ = self.fixture(task)
            Image.new("RGB", (16, 16), "red").save(task / "rgb_0.png")
            with self.assertRaisesRegex(adapter.CompletionError, "source RGB"):
                adapter.generated_mask_inputs(task, request)

    def test_mask_cannot_borrow_another_frame_source_image(self):
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            request, _, _ = self.fixture(task)
            masks = adapter.read_json(task / "masks.json")
            masks["masks"][0]["image_path"] = "rgb_1.png"
            adapter.write_json(task / "masks.json", masks)
            request["masks_manifest_sha256"] = adapter.sha256(task / "masks.json")
            with self.assertRaisesRegex(adapter.CompletionError, "source image differs"):
                adapter.generated_mask_inputs(task, request)

    def test_camera_gate_runs_before_backend_with_existing_registration_thresholds(self):
        import numpy as np

        sys.path.insert(0, str(SCRIPT.parent))
        import object_registration

        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            obj = orbit_fixture(task)
            payload = task / "payload.bin"
            payload.write_bytes(b"fixture hash-bound preprocessed artifact")
            digest = adapter.sha256(payload)
            frames, provenance = [], []
            for clip in obj["orbits"]:
                cameras = adapter.read_json(task / clip["cameras_path"])
                for index in (0, 15, 30, 45):
                    frame_id = f"{len(frames):06d}"
                    frames.append({"frame_id": frame_id, "image_path": payload.name, "image_sha256": digest, "mask_path": payload.name, "mask_sha256": digest, "depth_npz_path": payload.name, "depth_sha256": digest, "world_to_camera": cameras["frames"][index]["world_to_camera"]})
                    provenance.append({"frame_id": frame_id, "orbit_id": clip["orbit_id"], "generated_frame_index": index, "generated_video_path": clip["video_path"], "generated_video_sha256": clip["video_sha256"], "conditioning_cameras_path": clip["cameras_path"], "conditioning_cameras_sha256": clip["conditioning_cameras_sha256"]})
            adapter.write_json(task / "geometry.json", {"frames": frames, "coordinate_frame": "generated_DA3_world", "conditioning_poses_used": False})
            adapter.write_json(task / "provenance.json", {"frames": provenance})
            adapter.write_json(task / "lifting.json", {"voxel_size": .1})
            args = (task, task / "geometry.json", task / "provenance.json", task / "lifting.json", 7)
            with patch.object(object_registration, "camera_chain", return_value=(np.eye(4), {"passed": True})) as gate:
                result = adapter.validate_generated_camera_chain(*args)
            self.assertIn("not_observed_registration", result["acceptance_scope"])
            parameters = gate.call_args.args
            self.assertEqual(parameters[2], [False, False, False, True] * 3)
            self.assertAlmostEqual(parameters[3], .6)
            self.assertEqual(parameters[4:], (12, 7))
            with patch.object(object_registration, "camera_chain", side_effect=object_registration.RegistrationError("inconsistent camera")):
                with self.assertRaisesRegex(adapter.CompletionError, "before Stream3D"):
                    adapter.validate_generated_camera_chain(*args)
            provenance[0]["conditioning_cameras_sha256"] = "0" * 64
            adapter.write_json(task / "provenance.json", {"frames": provenance})
            with patch.object(object_registration, "camera_chain") as gate, self.assertRaisesRegex(adapter.CompletionError, "SHA256"):
                adapter.validate_generated_camera_chain(*args)
            gate.assert_not_called()

    def conditioning_cameras(self, task, count=4):
        import numpy as np

        frames = []
        for index in range(count):
            pose = np.eye(4)
            pose[:3, 3] = [.4 * index, .2, 1.5]
            frames.append({"frame_id": f"{index:06d}", "azimuth_degrees": 6.0 * index,
                           "world_to_camera": pose.tolist(),
                           "camera_center_world": (-pose[:3, :3].T @ pose[:3, 3]).tolist(),
                           "intrinsics": [[8., 0, 3], [0, 8., 3], [0, 0, 1]]})
        document = {"coordinate_frame": "scene_world", "units": "meters",
                    "camera_convention": "world_to_camera_opencv", "frames": frames}
        adapter.write_json(task / "cameras.json", document)
        return document

    def generated_dataset(self, task, count=12):
        """A materialized generated-frame dataset with its conditioning trajectory."""
        request, prediction, mappings = self.fixture(task, count=count)
        adapter.save_generated_geometry(task, request, prediction, mappings)
        geometry_path = task / "dataset/bed/input_manifest.json"
        conditioning = self.conditioning_cameras(task)
        video = task / "generated.mp4"
        video.write_bytes(b"generated orbit video fixture")
        provenance = []
        for index in range(count):
            provenance.append({"frame_id": str(index), "orbit_id": f"orbit_{index // 4:02d}",
                               "generated_video_path": video.name, "generated_video_sha256": adapter.sha256(video),
                               "generated_frame_index": index % 4,
                               "conditioning_cameras_path": "cameras.json",
                               "conditioning_cameras_sha256": adapter.sha256(task / "cameras.json")})
        adapter.write_json(task / "provenance.json", {"frames": provenance})
        adapter.write_json(task / "lifting.json", {"voxel_size": .1})
        return geometry_path, prediction, conditioning

    @unittest.skipUnless(RASTER_DEPS, "requires OpenCV for official DA3 raster materialization")
    def test_conditioning_fallback_replaces_estimated_cameras_and_keeps_the_estimate(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            geometry_path, prediction, conditioning = self.generated_dataset(task, count=12)
            args = (task, geometry_path, task / "provenance.json", task / "lifting.json", 7)
            sys.path.insert(0, str(SCRIPT.parent))
            import object_registration

            with patch.object(object_registration, "camera_chain", side_effect=object_registration.RegistrationError("fewer than three supported correspondences")):
                report = adapter.validate_generated_camera_chain(*args, fallback="conditioning")
            self.assertEqual(report["camera_source"], "conditioning_trajectory")
            self.assertEqual(report["independent_estimation"]["status"], "rejected")
            self.assertIn("fewer than three", report["independent_estimation"]["reason"])
            self.assertGreater(report["independent_estimation"]["conditioning_camera_center_span"], 0)
            geometry = adapter.read_json(geometry_path)
            self.assertTrue(geometry["conditioning_poses_used"])
            self.assertTrue(geometry["camera_conditioned"])
            self.assertEqual(geometry["coordinate_frame"], "scene_world")
            self.assertEqual(geometry["unit"], "meters")
            self.assertEqual(geometry["generated_camera_fallback"]["applied"], "conditioning_trajectory")
            mapping = np.diag([.5, .5, 1.])
            for index, frame in enumerate(geometry["frames"]):
                pose = np.asarray(conditioning["frames"][index % 4]["world_to_camera"], dtype=float)
                np.testing.assert_allclose(np.asarray(frame["world_to_camera"], dtype=float), pose, atol=1e-9)
                np.testing.assert_allclose(np.asarray(frame["da3_world_to_camera"], dtype=float), prediction.extrinsics[index], atol=1e-9)
                np.testing.assert_allclose(np.asarray(frame["intrinsics"], dtype=float),
                                           mapping @ np.asarray(conditioning["frames"][index % 4]["intrinsics"], dtype=float), atol=1e-9)
                with np.load(task / frame["depth_npz_path"]) as archive:
                    np.testing.assert_allclose(np.asarray(archive["world_to_camera"], dtype=float), pose, atol=1e-6)
                self.assertEqual(frame["depth_sha256"], adapter.sha256(task / frame["depth_npz_path"]))
                self.assertEqual(frame["camera_source"], "conditioning_trajectory")
            written = np.loadtxt(task / geometry["camera_poses_path"]).reshape(12, 4, 4)
            np.testing.assert_allclose(written, np.array([frame["world_to_camera"] for frame in geometry["frames"]]), atol=1e-6)
            provenance = adapter.read_json(task / "provenance.json")
            self.assertTrue(provenance["conditioning_cameras_used_as_geometry"])
            self.assertTrue(all(item["conditioning_cameras_used_as_geometry"] for item in provenance["frames"]))
            # a rewritten dataset is re-verified, never re-estimated
            with patch.object(object_registration, "camera_chain", side_effect=AssertionError("estimation must not run again")):
                again = adapter.validate_generated_camera_chain(*args, fallback="conditioning")
            self.assertEqual(again["camera_source"], "conditioning_trajectory")
            tampered = adapter.read_json(geometry_path)
            tampered["frames"][0]["world_to_camera"] = np.eye(4).tolist()
            adapter.write_json(geometry_path, tampered)
            with self.assertRaisesRegex(adapter.CompletionError, "not its conditioning pose"):
                adapter.verify_conditioning_trajectory(task, geometry_path, task / "provenance.json")

    @unittest.skipUnless(RASTER_DEPS, "requires OpenCV for official DA3 raster materialization")
    def test_the_conditioning_fallback_is_off_unless_the_profile_asks_for_it(self):
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            geometry_path, _, _ = self.generated_dataset(task, count=12)
            args = (task, geometry_path, task / "provenance.json", task / "lifting.json", 7)
            sys.path.insert(0, str(SCRIPT.parent))
            import object_registration

            with patch.object(object_registration, "camera_chain", side_effect=object_registration.RegistrationError("collapsed estimates")):
                with self.assertRaisesRegex(adapter.CompletionError, "before Stream3D"):
                    adapter.validate_generated_camera_chain(*args)
            geometry = adapter.read_json(geometry_path)
            self.assertFalse(geometry["conditioning_poses_used"])
            self.assertEqual(geometry["camera_estimation"], "fresh_DA3_joint_generated_RGB")
            self.assertNotIn("generated_camera_fallback", geometry)

    def test_a_conditioning_frame_outside_its_trajectory_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            self.conditioning_cameras(task)
            adapter.write_json(task / "provenance.json", {"frames": [
                {"frame_id": "000000", "conditioning_cameras_path": "cameras.json",
                 "conditioning_cameras_sha256": adapter.sha256(task / "cameras.json"), "generated_frame_index": 9}]})
            with self.assertRaisesRegex(adapter.CompletionError, "outside its conditioning trajectory"):
                adapter.conditioning_camera_records(task, task / "provenance.json")

    @unittest.skipUnless(RASTER_DEPS, "requires OpenCV for official DA3 raster materialization")
    def test_invalid_predicted_camera_cannot_be_promoted_to_stream3d_input(self):
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            request, prediction, mappings = self.fixture(task)
            prediction.extrinsics[0, 0, 0] = 2
            with self.assertRaisesRegex(adapter.CompletionError, "non-rigid"):
                adapter.save_generated_geometry(task, request, prediction, mappings)
            self.assertFalse((task / "dataset/bed/input_manifest.json").exists())

    @unittest.skipUnless(RASTER_DEPS, "requires OpenCV for official DA3 raster materialization")
    def test_invalid_mapping_or_missing_pose_does_not_publish_dataset(self):
        for failure in ("mapping", "poses", "colors"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as folder:
                task = Path(folder)
                request, prediction, mappings = self.fixture(task)
                if failure == "mapping":
                    mappings[0, 0, 0] = 0
                elif failure == "poses":
                    prediction.extrinsics = prediction.extrinsics[:-1]
                else:
                    prediction.processed_images = prediction.processed_images.astype(float)
                    prediction.processed_images[0, 0, 0, 0] = float("nan")
                with self.assertRaises(adapter.CompletionError):
                    adapter.save_generated_geometry(task, request, prediction, mappings)
                self.assertFalse((task / "dataset/bed/input_manifest.json").exists())

    def test_generated_mesh_exports_colored_surface_points_not_gaussians(self):
        import numpy as np
        import trimesh

        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            mesh = trimesh.creation.box()
            mesh.visual.vertex_colors = [180, 90, 20, 255]
            scene = trimesh.Scene()
            transform = np.eye(4)
            transform[:3, 3] = [2, 3, 4]
            scene.add_geometry(mesh, transform=transform)
            scene.export(task / "result.glb")
            report = adapter.sample_generated_mesh(task / "result.glb", task / "visual.ply", 1000, 0)
            self.assertEqual(report["visual_ply_kind"], "generated_surface_points_not_gaussians")
            pointcloud = trimesh.load(task / "visual.ply", process=False)
            self.assertEqual(len(pointcloud.vertices), 1000)
            self.assertTrue(np.all(pointcloud.vertices >= [1.5, 2.5, 3.5]))
            self.assertTrue(np.all(pointcloud.vertices <= [2.5, 3.5, 4.5]))
            np.testing.assert_allclose(pointcloud.colors[:, :3].mean(axis=0), [180, 90, 20], atol=1)


if __name__ == "__main__":
    unittest.main()
