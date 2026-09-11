from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/background_reconstruction.py"
SPEC = importlib.util.spec_from_file_location("background_reconstruction_adapter", SCRIPT)
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


class BackgroundBoundaryTests(unittest.TestCase):
    def test_outputs_are_always_generated_and_not_collision_eligible(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory)
            result = adapter.output_records(task, task / "background.ply", task / "background.glb", task / "report.json")
            self.assertEqual(set(result["outputs"]), {"generated_background_gaussian_ply", "generated_background_mesh", "background_reconstruction_report"})
            for record in result["outputs"].values():
                self.assertEqual(record["evidence"], "generated")
                self.assertEqual(record["status"], "candidate")
                self.assertFalse(record["collision_eligible"])

    def test_contract_clean_plate_is_rejected_before_model_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory)
            inputs = task / "inputs.json"
            adapter.scene.write_json(inputs, {"inputs": {"clean_plate_frames": {"path": "frames.json", "evidence": "contract_only"}}})
            with self.assertRaisesRegex(adapter.BackgroundError, "generated evidence"):
                adapter.load_inputs(task, inputs)


NUMERICAL_DEPS = all(importlib.util.find_spec(name) is not None for name in ("numpy", "PIL"))


@unittest.skipUnless(NUMERICAL_DEPS, "requires numpy and Pillow")
class FixedCameraCleanPlateTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from PIL import Image

        self.temp = tempfile.TemporaryDirectory()
        self.task = Path(self.temp.name)
        self.source = []
        self.clean = []
        for index in range(3):
            source = np.full((8, 12, 3), 200, dtype=np.uint8)
            mask = np.zeros((8, 12), dtype=np.uint8)
            mask[2:4, 3:5] = 255
            generated = source.copy()
            generated[mask > 0] = [30, 100, 160]
            Image.fromarray(source).save(self.task / f"source-{index}.png")
            Image.fromarray(generated).save(self.task / f"clean-{index}.png")
            Image.fromarray(mask).save(self.task / f"mask-{index}.png")
            extrinsic = np.eye(4)
            extrinsic[0, 3] = -index * 0.2
            self.source.append({"frame_id": str(index), "image_path": f"source-{index}.png", "width": 12, "height": 8,
                                "intrinsics": [[10, 0, 6], [0, 10, 4], [0, 0, 1]], "world_to_camera": extrinsic.tolist()})
            self.clean.append({"frame_id": str(index), "image_path": f"clean-{index}.png", "mask_path": f"mask-{index}.png", "width": 12, "height": 8})

    def tearDown(self):
        self.temp.cleanup()

    def validate(self, clean=None):
        return adapter.validate_clean_frames(self.task, {"frames": self.clean if clean is None else clean}, {"frames": self.source}, 0, 1.0)

    def test_generated_pixels_use_unchanged_original_camera_and_raster(self):
        records, checks = self.validate()
        self.assertEqual(len(records), 3)
        for record, source, check in zip(records, self.source, checks):
            self.assertEqual(record["intrinsics"], source["intrinsics"])
            self.assertEqual(record["world_to_camera"], source["world_to_camera"])
            self.assertNotEqual(record["image_path"], source["image_path"])
            self.assertEqual(check["outside_mask_mae"], 0)
            self.assertGreater(check["inside_mask_mae"], 0)

    def test_changed_pose_or_resampling_is_rejected(self):
        changed = copy.deepcopy(self.clean)
        changed[0]["world_to_camera"] = copy.deepcopy(self.source[0]["world_to_camera"])
        changed[0]["world_to_camera"][0][3] = 5.0
        with self.assertRaisesRegex(adapter.BackgroundError, "changed source world_to_camera"):
            self.validate(changed)
        changed = copy.deepcopy(self.clean)
        changed[0]["image_to_source"] = [[2, 0, 0], [0, 2, 0], [0, 0, 1]]
        with self.assertRaisesRegex(adapter.BackgroundError, "resampling"):
            self.validate(changed)

    def test_duplicate_and_unknown_frame_ids_are_rejected(self):
        changed = copy.deepcopy(self.clean)
        changed[1]["frame_id"] = changed[0]["frame_id"]
        with self.assertRaisesRegex(adapter.BackgroundError, "duplicate"):
            self.validate(changed)
        changed[1]["frame_id"] = "missing-camera"
        with self.assertRaisesRegex(adapter.BackgroundError, "no matching"):
            self.validate(changed)

    def test_original_copy_and_source_alias_are_rejected(self):
        from PIL import Image

        changed = copy.deepcopy(self.clean)
        changed[0]["image_path"] = self.source[0]["image_path"]
        with self.assertRaisesRegex(adapter.BackgroundError, "alias"):
            self.validate(changed)
        Image.open(self.task / "source-0.png").save(self.task / "clean-0.png")
        with self.assertRaisesRegex(adapter.BackgroundError, "copies unchanged"):
            self.validate()

    def test_unmasked_edits_and_dimension_changes_are_rejected(self):
        from PIL import Image

        original = Image.open(self.task / "clean-0.png").copy()
        Image.new("RGB", (12, 8), (0, 0, 0)).save(self.task / "clean-0.png")
        with self.assertRaisesRegex(adapter.BackgroundError, "changes unmasked"):
            self.validate()
        original.resize((6, 4)).save(self.task / "clean-0.png")
        with self.assertRaisesRegex(adapter.BackgroundError, "raster dimensions"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
