from __future__ import annotations

import ast
import copy
import importlib.util
import os
from pathlib import Path
import sys
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import three_d_fixer as adapter
from three_d_fixer_module import SPEC
from world_modeling.registry import ModuleRegistry
from world_modeling.modules.geometry_completion import SPEC as STREAM_SPEC


def write(path, value):
    adapter.write_json(path, value)
    return {"path": path.name, "sha256": adapter.sha256(path)}


def add_lineage(task, obj):
    for name in ("image.png", "mask.png"):
        if not (task / name).exists():
            (task / name).write_bytes(name.encode())
    obj["source_descriptions"] = write(task / "descriptions.json", {"objects": [{
        "frame_id": "000000", "object_id": "original_bed_observation", "image_path": "image.png",
        "description": "Original bed description", "attributes": {"material": "fabric"},
    }]})
    obj["source_observations"] = [{
        "frame_id": "000000", "source_object_id": "original_bed_observation",
        "source_image_path": "image.png", "source_image_sha256": adapter.sha256(task / "image.png"),
        "source_mask_path": "mask.png", "source_mask_sha256": adapter.sha256(task / "mask.png"),
    }]
    return obj


def lineage_document_fixture(task, obj):
    task = task.resolve()
    paths = {role: task / (role + ".json") for role in adapter.SPEC.inputs}
    write(paths["cameras"], {})
    write(paths["scene_gaussian_ply"], {})
    coordinates = {"coordinate_frame": "colmap_world", "units": "colmap_reconstruction"}
    write(paths["scene_depth"], {**coordinates, "camera_convention": "world_to_camera_opencv", "depth_kind": "camera_z"})
    write(paths["isolated_object_ply"], {"objects": [obj]})
    write(paths["assembled_object_views"], {"objects": [{"object_id": obj["object_id"]}]})
    write(paths["component_masks"], {"acceptance_authority": paths["lifting_report"].name, "masks": [{
        "object_id": obj["object_id"], "component_id": "__object__", "frame_id": "000000",
        "mask_path": "mask.png", "mask_sha256": adapter.sha256(task / "mask.png"),
    }]})
    def binding(role):
        return {"path": paths[role].name, "sha256": adapter.sha256(paths[role])}
    write(paths["lifting_report"], {**coordinates, "generated_geometry_used": False, "carve_performed": True,
          "tracks": [{"object_id": obj["object_id"], "accepted": True}],
          "source_files": {role: binding(role) for role in ("cameras", "scene_depth", "scene_gaussian_ply")},
          "resolved_files": {"component_masks": binding("component_masks")}})
    write(paths["assembly_report"], {"status": "assembled_observed_views",
          "method": "same_camera_verified_parent_mask_rgba_extraction", "cross_view_pixel_pasting": False,
          "hidden_pixels_generated": False, "complete_object_claimed": False,
          "source_artifacts": {role: binding(role) for role in ("component_masks", "isolated_object_ply", "cameras", "lifting_report")}})
    write(task / "inputs.json", {"inputs": {role: binding(role) for role in paths}})
    return paths


def runtime_fixture(root):
    model, source, base = root / "fixer", root / "source", root / "base"
    config = {"name": "ThreeDFixerPipeline", "args": {"image_cond_model": "dinov2_vitl14_reg", "models": {}}}
    hashes = {}
    for key in adapter.SOURCE_HASHES:
        path = source / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("source fixture")
        hashes[key] = adapter.sha256(path)
    for key, name in {**adapter.BASE_MODELS, **adapter.FIXER_MODELS}.items():
        is_base = key in adapter.BASE_MODELS
        config["args"]["models"][key] = f"microsoft/TRELLIS-image-large/ckpts/{name}" if is_base else f"ckpts/{name}"
        prefix = (base if is_base else model) / "ckpts" / name
        prefix.parent.mkdir(parents=True, exist_ok=True)
        Path(str(prefix) + ".json").write_text("{}")
        Path(str(prefix) + ".safetensors").write_bytes(b"not real weights")
    write(model / "pipeline.json", config)
    dino = root / "dino"
    dino.mkdir()
    (dino / "hubconf.py").write_text("# test fixture")
    moge, dino_weights = root / "moge.pt", root / "dino.pt"
    moge.write_bytes(b"not real weights")
    dino_weights.write_bytes(b"not real weights")
    return SimpleNamespace(source=source, model_dir=model, trellis_base=base, moge_checkpoint=moge,
                           dino_source=dino, dino_checkpoint=dino_weights), hashes


class ContractTests(unittest.TestCase):
    def test_document_boundary_validates_lineage_and_keeps_legacy_objects(self):
        for legacy in (False, True):
            with self.subTest(legacy=legacy), tempfile.TemporaryDirectory() as folder:
                task = Path(folder)
                obj = add_lineage(task, {"object_id": "bed", "observations": [{"frame_id": "000000", "mask_path": "mask.png"}]})
                if legacy:
                    for key in ("source_observations", "source_descriptions"):
                        obj.pop(key)
                _, actual, _ = adapter.validate_documents(task, lineage_document_fixture(task, obj))
                self.assertEqual(actual["bed"], obj)

    def test_invalid_lineage_blocks_before_preparation_or_model_loading(self):
        for mutation in ("mask_hash", "missing_field"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder:
                task = Path(folder)
                obj = add_lineage(task, {"object_id": "bed", "observations": [{"frame_id": "000000", "mask_path": "mask.png"}]})
                if mutation == "mask_hash":
                    obj["source_observations"][0]["source_mask_sha256"] = "0" * 64
                else:
                    obj.pop("source_descriptions")
                paths = lineage_document_fixture(task, obj)
                args = adapter.argument_parser().parse_args(["--task-dir", str(task), "--inputs", "inputs.json", "--outputs", "outputs.json"])
                with patch.object(adapter, "input_artifact", side_effect=lambda args, role: paths[role]), \
                     patch.object(adapter, "prepare_object") as prepare, patch.object(adapter, "load_pipeline") as load:
                    with self.assertRaisesRegex(ValueError, "lineage"):
                        adapter.run(args)
                prepare.assert_not_called()
                load.assert_not_called()
                self.assertFalse((task / "outputs.json").exists())

    def run_fixture(self, task, prepare_only):
        paths = {}
        for role in adapter.SPEC.inputs:
            paths[role] = task / (role + ".json")
            write(paths[role], {})
        write(task / "inputs.json", {"inputs": {role: {"path": path.name, "sha256": adapter.sha256(path)} for role, path in paths.items()}})
        documents = {"cameras": {"frames": [{"frame_id": "0"}]}, "scene_depth": {"frames": [{"frame_id": "0"}]},
                     "lifting_report": {"voxel_size": .01}, "verified_mask_bindings": {"bed": []}}
        args = adapter.argument_parser().parse_args(["--task-dir", str(task), "--inputs", "inputs.json", "--outputs", "outputs.json"]
                                                    + (["--prepare-only"] if prepare_only else []))
        return args, paths, (documents, {"bed": {}}, {"bed": {}})

    def test_prepare_only_never_loads_model_or_publishes_completion_roles(self):
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            args, paths, documents = self.run_fixture(task, True)
            with patch.object(adapter, "input_artifact", side_effect=lambda args, role: paths[role]), \
                 patch.object(adapter, "validate_documents", return_value=documents), \
                 patch.object(adapter, "prepare_object", return_value={"object_id": "bed"}), \
                 patch.object(adapter, "load_pipeline") as load, patch.object(adapter, "publish") as publish:
                result = adapter.run(args)
            load.assert_not_called()
            publish.assert_not_called()
            self.assertFalse(result["model_inference_performed"])
            self.assertFalse(result["roles_published"])
            self.assertFalse((task / "outputs.json").exists())

    def test_missing_runtime_preserves_blocked_receipt_without_success_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            args, paths, documents = self.run_fixture(task, False)
            with patch.object(adapter, "input_artifact", side_effect=lambda args, role: paths[role]), \
                 patch.object(adapter, "validate_documents", return_value=documents), \
                 patch.object(adapter, "prepare_object", return_value={"object_id": "bed"}), \
                 patch.object(adapter, "load_pipeline") as load, patch.object(adapter, "publish") as publish:
                with self.assertRaisesRegex(adapter.FixerError, "source and dedicated"):
                    adapter.run(args)
            receipt_path = next((task / "stages/observed_context_completion").glob("run-*/provider_report.json"))
            receipt = adapter.read_json(receipt_path)
            self.assertEqual(receipt["status"], "blocked_three_d_fixer")
            self.assertFalse(receipt["roles_published"])
            load.assert_not_called()
            publish.assert_not_called()
            self.assertFalse((task / "outputs.json").exists())

    def test_explicit_alternative_registration_does_not_consume_generated_orbits(self):
        registry = ModuleRegistry([SPEC])
        self.assertEqual(registry.ordered([SPEC.id], initial_roles=SPEC.inputs), (SPEC,))
        self.assertNotIn("object_orbit_videos", SPEC.inputs)
        self.assertIn("scene_depth", SPEC.inputs)
        self.assertIn("assembly_report", SPEC.inputs)
        self.assertEqual(SPEC.output_names, STREAM_SPEC.output_names)

    def test_no_silent_replacement_of_default_completion_producer(self):
        registry = ModuleRegistry([STREAM_SPEC])
        with self.assertRaisesRegex(ValueError, "producer"):
            registry.register(SPEC)
        self.assertIs(registry.get("geometry_completion"), STREAM_SPEC)
        registry.unregister("geometry_completion")
        registry.register(SPEC)
        self.assertIs(registry.get(SPEC.id), SPEC)

    def test_parser_defaults_do_not_enable_prepare_or_relax_observed_gates(self):
        args = adapter.argument_parser().parse_args(["--task-dir", "/task", "--inputs", "in.json", "--outputs", "out.json"])
        self.assertFalse(args.prepare_only)
        self.assertEqual(args.confidence_min, .5)
        self.assertEqual(args.min_points, 64)

    def test_missing_core_source_blocks_before_any_model_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(source=Path(folder), model_dir=Path(folder))
            with self.assertRaisesRegex(adapter.FixerError, "official source"):
                adapter.runtime_preflight(args)

    def test_pure_trellis_configuration_is_not_three_d_fixer(self):
        with tempfile.TemporaryDirectory() as folder:
            args, hashes = runtime_fixture(Path(folder))
            config = adapter.read_json(args.model_dir / "pipeline.json")
            config["name"] = "TrellisImageTo3DPipeline"
            write(args.model_dir / "pipeline.json", config)
            with patch.dict(adapter.SOURCE_HASHES, hashes), self.assertRaisesRegex(adapter.FixerError, "not a substitute"):
                adapter.runtime_preflight(args)

    def test_dedicated_scene_model_cannot_fall_back_to_base(self):
        with tempfile.TemporaryDirectory() as folder:
            args, hashes = runtime_fixture(Path(folder))
            config = adapter.read_json(args.model_dir / "pipeline.json")
            config["args"]["models"]["scene_slat_flow_model"] = "microsoft/TRELLIS-image-large/ckpts/slat_flow_img_dit_L_64l8p2_fp16"
            write(args.model_dir / "pipeline.json", config)
            with patch.dict(adapter.SOURCE_HASHES, hashes), patch.object(adapter, "verify_dedicated_checkpoint"), self.assertRaisesRegex(adapter.FixerError, "mapping"):
                adapter.runtime_preflight(args)

    def test_dedicated_checkpoint_symlink_to_base_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            args, hashes = runtime_fixture(Path(folder))
            name = adapter.FIXER_MODELS["slat_decoder_gs"]
            path = args.model_dir / "ckpts" / (name + ".safetensors")
            path.unlink()
            path.symlink_to(args.dino_checkpoint)
            with patch.dict(adapter.SOURCE_HASHES, hashes), patch.object(adapter, "verify_dedicated_checkpoint"), self.assertRaisesRegex(adapter.FixerError, "cannot alias"):
                adapter.runtime_preflight(args)

    def test_same_name_shape_compatible_weights_do_not_bypass_official_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            args, hashes = runtime_fixture(Path(folder))
            with patch.dict(adapter.SOURCE_HASHES, hashes), self.assertRaisesRegex(adapter.FixerError, "differs from official"):
                adapter.runtime_preflight(args)

    def test_hash_bound_path_rejects_escape_and_changed_artifact(self):
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            bound = write(task / "source.json", {"x": 1})
            write(task / "source.json", {"x": 2})
            with self.assertRaisesRegex(adapter.FixerError, "hash binding"):
                adapter.bound_path(task, bound)
            with self.assertRaises(ValueError):
                adapter.bound_path(task, {"path": "../outside.json", "sha256": "0" * 64})

    def test_task_symlink_is_canonicalized_but_artifact_escape_is_not_allowed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            actual = root / "actual"
            actual.mkdir()
            alias = root / "alias"
            alias.symlink_to(actual, target_is_directory=True)
            (actual / "file.json").write_text("{}")
            self.assertEqual(adapter.relative(alias, alias / "file.json"), "file.json")
            (root / "external.json").write_text("{}")
            (actual / "escape.json").symlink_to(root / "external.json")
            with self.assertRaisesRegex(ValueError, "escapes task"):
                adapter.relative(alias, alias / "escape.json")

    def test_heldout_mask_pixels_are_hash_checked_not_only_their_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            path = task / "mask.png"
            path.write_bytes(b"original mask fixture")
            objects = {"bed": {"observations": [{"frame_id": "0", "mask_path": "mask.png"}]}}
            masks = {"masks": [{"object_id": "bed", "component_id": "__object__", "frame_id": "0",
                                "mask_path": "mask.png", "mask_sha256": adapter.sha256(path)}]}
            bindings = adapter.observation_bindings(task, objects, masks)
            path.write_bytes(b"changed mask fixture")
            with self.assertRaisesRegex(adapter.FixerError, "hash binding"):
                adapter.observation_bindings(task, objects, masks)
            with self.assertRaisesRegex(adapter.FixerError, "hash binding"):
                adapter.bound_path(task, bindings["bed"][0])


HAS_NUMPY = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(HAS_NUMPY, "requires numpy")
class CoordinateTests(unittest.TestCase):
    def test_opencv_c2w_and_normalized_intrinsics_project_original_pixels(self):
        import numpy as np

        pose = np.array([[0., -1, 0, 2], [1, 0, 0, -3], [0, 0, 1, 5], [0, 0, 0, 1]])
        camera = {"world_to_camera": pose.tolist(), "intrinsics": [[600, 0, 320], [0, 720, 240], [0, 0, 1]], "width": 640, "height": 480}
        c2w, k = adapter.projection_inputs(camera)
        pixels = np.array([[100, 50], [420, 360]])
        z = np.array([2., 4])
        cam_points = np.column_stack((pixels, np.ones(2))) @ np.linalg.inv(camera["intrinsics"]).T * z[:, None]
        world = cam_points @ c2w[:3, :3].T + c2w[:3, 3]
        native_camera = world @ np.linalg.inv(c2w)[:3, :3].T + np.linalg.inv(c2w)[:3, 3]
        uv = native_camera @ k.T
        np.testing.assert_allclose(uv[:, :2] / uv[:, 2:], pixels / [640, 480])
        np.testing.assert_allclose(np.linalg.inv(c2w), pose)

    def test_pose_reflection_and_scaled_camera_are_rejected(self):
        import numpy as np

        for diagonal in ((-1, 1, 1, 1), (2, 2, 2, 1)):
            camera = {"world_to_camera": np.diag(diagonal).tolist(), "intrinsics": np.eye(3).tolist(), "width": 64, "height": 48}
            with self.assertRaises(RuntimeError):
                adapter.projection_inputs(camera)

    def test_two_stage_inverse_preserves_original_reconstruction_units(self):
        import numpy as np

        canonical = np.array([[-.2, .1, .3], [.2, -.1, -.3]])
        matrix = adapter.canonical_to_world([10, 20, 30], 8, [.1, .2, -.3], .6)
        expected = (canonical * .6 + [.1, .2, -.3]) * 8 + [10, 20, 30]
        np.testing.assert_allclose(canonical @ matrix[:3, :3].T + matrix[:3, 3], expected)
        np.testing.assert_allclose((expected - matrix[:3, 3]) @ np.linalg.inv(matrix[:3, :3]).T, canonical)

    def test_invalid_normalization_is_not_sanitized_or_guessed(self):
        for coarse, scale, fine, scale2 in (([1, 2], 1, [1, 2, 3], 1), ([1, 2, 3], 0, [1, 2, 3], 1),
                                          ([1, 2, 3], 1, [1, 2, float("nan")], 1)):
            with self.assertRaises(adapter.FixerError):
                adapter.canonical_to_world(coarse, scale, fine, scale2)


HAS_GEOMETRY = all(importlib.util.find_spec(name) is not None for name in ("numpy", "PIL", "plyfile", "scipy"))


def observed_fixture(task):
    import numpy as np
    from PIL import Image
    from plyfile import PlyData, PlyElement
    from object_lifting import load_frame, load_mask, backproject

    width, height = 16, 12
    Image.new("RGB", (width, height), (60, 80, 100)).save(task / "image.png")
    Image.new("L", (width, height), 255).save(task / "mask.png")
    depth_path, confidence_path = task / "depth.npy", task / "confidence.npy"
    np.save(depth_path, np.full((6, 8), 3., dtype=np.float32))
    np.save(confidence_path, np.ones((6, 8), dtype=np.float32))
    camera = {"frame_id": "000000", "image_path": "image.png", "width": width, "height": height,
              "intrinsics": [[8, 0, 8], [0, 8, 6], [0, 0, 1]], "world_to_camera": np.eye(4).tolist()}
    depth = {"frame_id": "000000", "depth_path": "depth.npy", "confidence_path": "confidence.npy",
             "depth_sha256": adapter.sha256(depth_path), "confidence_sha256": adapter.sha256(confidence_path),
             "intrinsics": [[4, 0, 4], [0, 4, 3], [0, 0, 1]], "world_to_camera": np.eye(4).tolist(),
             "image_to_depth": [[.5, 0, 0], [0, .5, 0], [0, 0, 1]]}
    args = SimpleNamespace(confidence_min=.5, min_points=32, registration_threshold=.1)
    args.observation_bindings = {"bed": [{"frame_id": "000000", "path": "mask.png", "sha256": adapter.sha256(task / "mask.png")}]}
    frame = load_frame(task, "000000", depth, camera, args)
    points = backproject(frame, load_mask(task, "mask.png", frame.depth.shape, frame))
    rows = np.zeros(len(points), dtype=[(name, "f4") for name in ("x", "y", "z")])
    for i, name in enumerate(("x", "y", "z")):
        rows[name] = points[:, i]
    PlyData([PlyElement.describe(rows, "vertex")]).write(task / "observed.ply")
    obj = {"object_id": "bed", "association_status": "geometrically_verified", "coordinate_frame": "colmap_world",
           "units": "colmap_reconstruction", "ply_path": "observed.ply", "sha256": adapter.sha256(task / "observed.ply"),
           "observations": [{"frame_id": frame_id, "mask_path": "mask.png"} for frame_id in ("000000", "000001", "000002")]}
    view = {"frame_id": "000000", "foreground_pixels": width * height, "coordinate_frame": "original_image",
            "evidence": "observed_pixels", "alpha_source": "verified_parent_mask", "image_path": "image.png",
            "image_sha256": adapter.sha256(task / "image.png"), "parent_mask_path": "mask.png",
            "parent_mask_sha256": adapter.sha256(task / "mask.png")}
    assembled = {"object_id": "bed", "observed_geometry": copy.deepcopy(obj), "geometry_extent": "observed_only_incomplete", "views": [view]}
    directory = task / "prepared"
    directory.mkdir()
    cameras = {"000000": camera}
    for i, frame_id in enumerate(("000001", "000002"), start=1):
        cameras[frame_id] = {**copy.deepcopy(camera), "frame_id": frame_id}
        cameras[frame_id]["world_to_camera"][0][3] = .1 * i
    return obj, assembled, cameras, {"000000": depth}, directory, args


@unittest.skipUnless(HAS_GEOMETRY, "requires CPU numpy/Pillow/plyfile/scipy; no model inference")
class ObservedInputTests(unittest.TestCase):
    def test_observed_context_and_generated_mesh_preserve_complete_lineage(self):
        import numpy as np

        for legacy in (False, True):
            with self.subTest(legacy=legacy), tempfile.TemporaryDirectory() as folder:
                task = Path(folder)
                obj, assembled, cameras, depths, directory, args = observed_fixture(task)
                if not legacy:
                    add_lineage(task, obj)
                    assembled["observed_geometry"] = copy.deepcopy(obj)
                adapter.validate_object_lineages(task, [obj])
                expected = adapter.lineage_metadata(obj)
                context = adapter.prepare_object(task, obj, assembled, cameras, depths, directory, args)
                self.assertEqual(adapter.lineage_metadata(context), expected)
                self.assertEqual(adapter.lineage_metadata(adapter.read_json(directory / "observed_context.json")), expected)
                if not legacy:
                    self.assertIsNot(context["source_observations"], obj["source_observations"])
                args.seed, args.texture_size = 0, 64
                native = SimpleNamespace(vertices=np.zeros((3, 3)), faces=np.array([[0, 1, 2]]))
                pipeline = Mock()
                pipeline.run.return_value = ({"mesh": [native], "gaussian": [Mock()]}, np.zeros(3), 1., np.zeros(3), 1.)
                mesh = Mock(vertices=native.vertices, faces=native.faces)
                mesh.export.side_effect = lambda path: Path(path).write_bytes(b"CPU provider export fixture")
                fake_torch = Mock()
                fake_torch.isfinite.return_value.all.return_value.item.return_value = True
                image_utils = SimpleNamespace(process_instance_image_only=Mock(return_value=(Mock(), Mock())),
                                              process_scene_image=Mock(return_value=(Mock(), Mock())))
                postprocess = SimpleNamespace(to_glb=Mock(return_value=mesh))
                with patch.dict(sys.modules, {"torch": fake_torch, "threeDFixer.datasets.utils": image_utils,
                                               "threeDFixer.utils": SimpleNamespace(postprocessing_utils=postprocess)}), \
                     patch.object(adapter, "save_canonical_gaussians", side_effect=lambda gaussian, path: Path(path).write_bytes(b"CPU Gaussian fixture")):
                    result = adapter.infer_object(task, context, pipeline, directory, args)
                self.assertEqual(adapter.lineage_metadata(result), expected)
                if not legacy:
                    self.assertIsNot(result["source_observations"], context["source_observations"])
                    self.assertEqual(result["source_observations"][0]["source_object_id"], "original_bed_observation")

    def test_two_view_track_is_blocked_before_model_for_missing_heldout_views(self):
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            values = observed_fixture(task)
            values[0]["observations"] = values[0]["observations"][:2]
            values[1]["observed_geometry"] = copy.deepcopy(values[0])
            with self.assertRaisesRegex(adapter.FixerError, "two original heldout cameras"):
                adapter.prepare_object(task, *values)

    def test_gaussian_export_explicitly_disables_official_display_rotation(self):
        import numpy as np
        from plyfile import PlyData, PlyElement

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "gaussian.ply"
            names = ["x", "y", "z", "opacity"] + [f"f_dc_{i}" for i in range(3)] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
            rows = np.zeros(3, dtype=[(name, "f4") for name in names])
            gaussian = Mock()
            gaussian.save_ply.side_effect = lambda path, **kwargs: PlyData([PlyElement.describe(rows, "vertex")]).write(path)
            adapter.save_canonical_gaussians(gaussian, path)
            gaussian.save_ply.assert_called_once_with(str(path), transform=None)
            rows["x"][0] = np.nan
            with self.assertRaisesRegex(adapter.FixerError, "nonfinite"):
                adapter.save_canonical_gaussians(gaussian, path)

    def test_saturated_opacity_logits_are_clamped_but_nan_still_fails(self):
        import numpy as np
        from plyfile import PlyData, PlyElement

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "gaussian.ply"
            names = ["x", "y", "z", "opacity"] + [f"f_dc_{i}" for i in range(3)] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
            rows = np.zeros(3, dtype=[(name, "f4") for name in names])
            rows["opacity"] = np.array([np.inf, -np.inf, 0.0], dtype="f4")
            gaussian = Mock()
            gaussian.save_ply.side_effect = lambda path, **kwargs: PlyData([PlyElement.describe(rows, "vertex")]).write(path)
            clamped = adapter.save_canonical_gaussians(gaussian, path)
            self.assertEqual(clamped, 2)
            stored = np.asarray(PlyData.read(str(path))["vertex"].data["opacity"], dtype=float)
            self.assertTrue(np.isfinite(stored).all())
            self.assertAlmostEqual(float(stored[0]), adapter.OPACITY_LOGIT_CLAMP, places=4)
            self.assertAlmostEqual(float(stored[1]), -adapter.OPACITY_LOGIT_CLAMP, places=4)
            rows["opacity"] = np.array([np.nan, 0.0, 0.0], dtype="f4")
            with self.assertRaisesRegex(adapter.FixerError, "nonfinite"):
                adapter.save_canonical_gaussians(gaussian, path)

    def test_a_support_shortfall_below_the_target_is_recorded_not_fatal(self):
        import numpy as np

        # The measured ScanNet++ DSLR case: 8971 of 10000 lifted points land on
        # the isolated object, 0.3% short of the 0.9 target.
        distances = np.concatenate([np.zeros(8971), np.full(1029, 10.0)])
        fraction, support = adapter.selected_view_support(distances, 0.05)
        self.assertAlmostEqual(fraction, 0.8971, places=4)
        self.assertTrue(support["below_target"])
        self.assertEqual(support["policy"], "recorded_not_blocking")
        self.assertEqual(support["target_fraction"], adapter.SUPPORT_TARGET)
        self.assertEqual(support["floor_fraction"], adapter.SUPPORT_FLOOR)
        self.assertGreaterEqual(support["support_fraction"], adapter.SUPPORT_FLOOR)

    def test_support_at_or_above_the_target_is_not_flagged(self):
        import numpy as np

        fraction, support = adapter.selected_view_support(np.zeros(1000), 0.05)
        self.assertEqual(fraction, 1.0)
        self.assertFalse(support["below_target"])

    def test_a_surface_that_is_not_there_at_all_is_still_refused(self):
        import numpy as np

        with self.assertRaisesRegex(adapter.FixerError, "below the 0.5 floor"):
            adapter.selected_view_support(np.full(1000, 10.0), 0.05)

    def test_an_unusable_object_is_recorded_without_discarding_the_usable_ones(self):
        receipt, receipt_path = {"objects": [], "rejected_objects": []}, Path(tempfile.mkdtemp()) / "report.json"
        objects = {"bed": {"object_id": "bed"}, "table": {"object_id": "table"}}

        def fake_prepare(task, obj, assembled, cameras, depths, directory, args):
            if obj["object_id"] == "table":
                raise adapter.FixerError("selected view DA3/isolated-object support 0.238298 is below the 0.5 floor")
            return {"object_id": obj["object_id"], "model_inference_performed": False}

        with patch.object(adapter, "prepare_object", side_effect=fake_prepare):
            adapter.prepare_objects(None, objects, {"bed": {}, "table": {}}, {}, {}, Path(tempfile.mkdtemp()),
                                    None, receipt, receipt_path)
        self.assertEqual([item["object_id"] for item in receipt["objects"]], ["bed"])
        self.assertEqual(len(receipt["rejected_objects"]), 1)
        self.assertIn("below the 0.5 floor", receipt["rejected_objects"][0]["reason"])
        self.assertEqual(json.loads(receipt_path.read_text())["rejected_objects"], receipt["rejected_objects"])

    def test_a_run_where_every_object_is_unusable_is_still_fatal(self):
        receipt, receipt_path = {"objects": [], "rejected_objects": []}, Path(tempfile.mkdtemp()) / "report.json"
        with patch.object(adapter, "prepare_object", side_effect=adapter.FixerError("no usable selected view")):
            with self.assertRaisesRegex(adapter.FixerError, "every observed object failed"):
                adapter.prepare_objects(None, {"bed": {"object_id": "bed"}}, {"bed": {}}, {}, {},
                                        Path(tempfile.mkdtemp()), None, receipt, receipt_path)

    def test_observed_region_refinement_moves_vertices_toward_measured_points(self):
        import numpy as np

        vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
        observed = vertices + np.array([0, 0, 0.2])
        refined, supported = adapter.refine_observed_regions(
            vertices, faces, observed, np.eye(4), radius=0.5, strength=0.5, smoothing=0.0, iterations=4)
        self.assertEqual(supported, 4)
        before = np.linalg.norm(vertices - observed, axis=1)
        after = np.linalg.norm(refined - observed, axis=1)
        self.assertTrue((after < before).all())
        # far-away observations must not move the mesh at all
        untouched, supported_far = adapter.refine_observed_regions(
            vertices, faces, observed + 100.0, np.eye(4), radius=0.5, strength=0.5, smoothing=0.0, iterations=4)
        self.assertEqual(supported_far, 0)
        self.assertTrue(np.allclose(untouched, vertices))
        for kwargs in ({"radius": 0, "strength": 0.5, "smoothing": 0.0, "iterations": 4},
                       {"radius": 0.5, "strength": 2.0, "smoothing": 0.0, "iterations": 4},
                       {"radius": 0.5, "strength": 0.5, "smoothing": 1.0, "iterations": 4}):
            with self.assertRaisesRegex(adapter.FixerError, "invalid observed-region refinement parameters"):
                adapter.refine_observed_regions(vertices, faces, observed, np.eye(4), **kwargs)

    def test_depth_raster_and_original_image_mapping_preserve_world_scale(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            values = observed_fixture(task)
            result = adapter.prepare_object(task, *values)
            self.assertEqual(result["point_count"], 48)
            self.assertEqual(result["units"], "colmap_reconstruction")
            self.assertFalse(result["generated_orbit_used"])
            with np.load(task / result["points_path"]) as pack:
                np.testing.assert_allclose(pack["points"][:, 2], 3)

    def test_synthetic_view_cannot_be_used_as_observed_scene(self):
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            values = observed_fixture(task)
            values[1]["views"][0]["evidence"] = "conditioned_synthesis"
            with self.assertRaisesRegex(adapter.FixerError, "original observed"):
                adapter.prepare_object(task, *values)

    def test_depth_hash_mutation_is_detected_before_backprojection(self):
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            values = observed_fixture(task)
            values[3]["000000"]["depth_sha256"] = "0" * 64
            with self.assertRaisesRegex(adapter.FixerError, "hash binding"):
                adapter.prepare_object(task, *values)

    def test_calibration_mismatch_is_not_replaced_by_bbox_alignment(self):
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            values = observed_fixture(task)
            values[3]["000000"]["intrinsics"][0][0] = 30
            with self.assertRaisesRegex(RuntimeError, "intrinsics disagree"):
                adapter.prepare_object(task, *values)

    def test_camera_depth_disagreement_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            values = observed_fixture(task)
            values[3]["000000"]["world_to_camera"][0][3] = 2
            with self.assertRaisesRegex(RuntimeError, "world_to_camera disagree"):
                adapter.prepare_object(task, *values)


@unittest.skipUnless(os.environ.get("THREED_FIXER_TEST_AUDIT_DIR") and HAS_NUMPY, "requires pinned official source audit files")
class OfficialBoundaryTests(unittest.TestCase):
    def test_inverse_transform_matches_official_transform_vertices(self):
        import numpy as np

        path = Path(os.environ["THREED_FIXER_TEST_AUDIT_DIR"]) / "dataset_utils.py"
        self.assertEqual(adapter.sha256(path), adapter.SOURCE_HASHES["threeDFixer/datasets/utils.py"])
        tree = ast.parse(path.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "transform_vertices")
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        points = np.array([[.1, .2, .3], [-.2, .3, -.4]])
        expected = namespace["transform_vertices"](points, ["scale", "translation", "scale", "translation"], [.7, np.array([[.1, -.2, .3]]), 9, np.array([[11, 22, 33]])])
        matrix = adapter.canonical_to_world([11, 22, 33], 9, [.1, -.2, .3], .7)
        np.testing.assert_allclose(points @ matrix[:3, :3].T + matrix[:3, 3], expected)

    def test_native_projection_inverts_c2w_exactly_once(self):
        path = Path(os.environ["THREED_FIXER_TEST_AUDIT_DIR"]) / "threeD_fixer.py"
        self.assertEqual(adapter.sha256(path), adapter.SOURCE_HASHES["threeDFixer/pipelines/threeD_fixer.py"])
        tree = ast.parse(path.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ThreeDFixerPipeline")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "project_uv")
        inverses = [node for node in ast.walk(method) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "inverse"]
        self.assertEqual(len(inverses), 1)
        self.assertEqual(ast.unparse(inverses[0].args[0]), "extrinsics")


if __name__ == "__main__":
    unittest.main()
