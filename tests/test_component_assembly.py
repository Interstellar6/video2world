from __future__ import annotations

import argparse
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/component_assembly.py"
SPEC = importlib.util.spec_from_file_location("component_assembly_adapter", SCRIPT)
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)
IMAGE_DEPS = all(importlib.util.find_spec(name) is not None for name in ("numpy", "PIL"))


@unittest.skipUnless(IMAGE_DEPS, "requires numpy and Pillow")
class ComponentAssemblyTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from PIL import Image

        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.task = Path(self.temporary.name)
        self.inputs = self.task / "inputs"
        self.inputs.mkdir()
        self.args = argparse.Namespace(task_dir=self.task, inputs=self.task / "inputs.json",
                                       outputs=self.task / "stages/component_assembly/provider-artifacts.json", max_views=8)
        frames, records = [], []
        for index in range(3):
            yy, xx = np.indices((10, 12))
            image = np.stack((xx * 10, yy * 15, np.full_like(xx, 40 + index * 20)), axis=-1).astype(np.uint8)
            Image.fromarray(image).save(self.inputs / f"image_{index}.png")
            parent = np.zeros((10, 12), dtype=np.uint8)
            parent[2:6, 2:6] = 255
            if index == 1:
                parent[2:6, 6:8] = 255
            elif index == 2:
                parent[:] = 255
            Image.fromarray(parent).save(self.inputs / f"parent_{index}.png")
            part = np.zeros_like(parent)
            part[0:3, 0:3] = 255
            Image.fromarray(part).save(self.inputs / f"part_{index}.png")
            frame = {"frame_id": str(index), "image_path": f"inputs/image_{index}.png", "width": 12, "height": 10}
            frames.append(frame)
            records.append({**frame, "object_id": "bed", "component_id": "__object__", "mask_path": f"inputs/parent_{index}.png",
                            "mask_sha256": adapter.sha256(self.inputs / f"parent_{index}.png")})
            records.append({**frame, "object_id": "bed", "component_id": "component_track_pillow_a", "component_group_id": "pillow",
                            "text": "white pillow", "observation_id": f"observation_{index}", "identity_scope": "geometric_component_track",
                            "geometry_validated": True, "membership_relation": "bedding", "composition_allowed": False,
                            "mask_path": f"inputs/part_{index}.png", "mask_sha256": adapter.sha256(self.inputs / f"part_{index}.png")})
        self.documents = {
            "component_masks": {"frames": frames, "masks": records, "acceptance_authority": "inputs/lifting_report.json"},
            "isolated_object_ply": {"objects": [{"object_id": "bed", "association_status": "geometrically_verified",
                "geometry_extent": "observed_only_incomplete", "parts": [{"component_id": "component_track_pillow_a", "component_group_id": "pillow"}],
                "observations": [{"frame_id": str(index), "mask_path": f"inputs/parent_{index}.png"} for index in (0, 1)]}]},
            "object_descriptions": {"objects": [{"object_id": "bed", "unobserved_components": ["legs"]}]},
            "cameras": {"frames": copy.deepcopy(frames)},
            "lifting_report": {"status": "passed", "carve_performed": True, "generated_geometry_used": False,
                "accepted_object_ids": ["bed"], "tracks": [{"object_id": "bed", "accepted": True,
                    "observations": [{"frame_id": "0"}, {"frame_id": "1"}]}], "resolved_files": {}, "source_files": {}},
        }
        self.save_documents()
        adapter.write_json(self.args.inputs, {"module": "component_assembly", "inputs": {
            role: {"path": f"inputs/{role}.json", "evidence": "derived"} for role in self.documents}})

    def save_documents(self, *, bind=True):
        for role, document in self.documents.items():
            if role != "lifting_report":
                adapter.write_json(self.inputs / f"{role}.json", document)
        report = self.documents["lifting_report"]
        if bind:
            role = "resolved_files" if "resolved_files" in report else "source_files"
            report[role]["component_masks"] = adapter.file_record(self.task, self.inputs / "component_masks.json")
            report["source_files"]["cameras"] = adapter.file_record(self.task, self.inputs / "cameras.json")
        adapter.write_json(self.inputs / "lifting_report.json", report)

    def output_documents(self, result):
        return tuple(adapter.read_json(self.task / result["outputs"][role]["path"])
                     for role in ("assembled_object_views", "assembly_report"))

    def assert_rejected(self, expression):
        with self.assertRaisesRegex(ValueError, expression):
            adapter.run(self.args)
        self.assertFalse(self.args.outputs.exists())
        self.assertFalse(list(self.task.glob("stages/component_assembly/run-*/assembly_report.json")))

    def test_rgba_uses_only_verified_parent_and_preserves_camera_pixels_and_annotations(self):
        import numpy as np
        from PIL import Image

        self.documents["isolated_object_ply"]["objects"][0]["observations"][0]["mask_sha256"] = adapter.sha256(self.inputs / "parent_0.png")
        self.save_documents()
        result = adapter.run(self.args)
        manifest, report = self.output_documents(result)
        obj, = manifest["objects"]
        self.assertEqual([view["frame_id"] for view in obj["views"]], ["1", "0"])
        self.assertEqual(obj["missing_components"], ["legs"])
        self.assertTrue(obj["completion_required"])
        for view in obj["views"]:
            with Image.open(self.task / view["rgba_path"]) as image:
                rgba = np.asarray(image)
            with Image.open(self.task / view["image_path"]) as image:
                np.testing.assert_array_equal(rgba[:, :, :3], np.asarray(image))
            with Image.open(self.task / view["parent_mask_path"]) as image:
                np.testing.assert_array_equal(rgba[:, :, 3], np.asarray(image))
            self.assertEqual(int(rgba[0, 0, 3]), 0)
            self.assertEqual(view["rgba_sha256"], adapter.sha256(self.task / view["rgba_path"]))
            self.assertEqual(view["crop_sha256"], adapter.sha256(self.task / view["crop_path"]))
            component, = view["components"]
            self.assertEqual(component["component_id"], "component_track_pillow_a")
            self.assertEqual(component["component_group_id"], "pillow")
            self.assertEqual(component["name"], "white pillow")
            self.assertEqual(component["usage"], "annotation_only")
            self.assertTrue(component["geometry_validated"])
            self.assertTrue(view["parent_hash_bound_by_lifting"])
        self.assertEqual(report["method"], "same_camera_verified_parent_mask_rgba_extraction")
        for name in ("component_annotations_affect_alpha", "cross_view_pixel_pasting", "hidden_pixels_generated", "complete_object_claimed"):
            self.assertFalse(report[name])

    def test_group_verified_annotation_does_not_claim_verified_individual_identity(self):
        for record in self.documents["component_masks"]["masks"]:
            if record["component_id"] != "__object__":
                record.update(geometry_validated=False, group_geometry_validated=True, identity_scope="frame_local_observation")
        self.save_documents()
        manifest, _ = self.output_documents(adapter.run(self.args))
        component = manifest["objects"][0]["views"][0]["components"][0]
        self.assertFalse(component["geometry_validated"])
        self.assertTrue(component["group_geometry_validated"])
        self.assertEqual(component["identity_scope"], "frame_local_observation")

    def test_observation_path_mismatch_rejects_even_identical_mask_bytes(self):
        alternate = self.inputs / "alternate.png"
        alternate.write_bytes((self.inputs / "parent_0.png").read_bytes())
        self.documents["isolated_object_ply"]["objects"][0]["observations"][0]["mask_path"] = "inputs/alternate.png"
        self.save_documents()
        self.assert_rejected("parent mask paths disagree")

    def test_observation_hash_mismatch_rejects(self):
        self.documents["isolated_object_ply"]["objects"][0]["observations"][0]["mask_sha256"] = "0" * 64
        self.save_documents()
        self.assert_rejected("observation mask hash mismatch")

    def test_parent_content_change_rejects_bound_hash(self):
        (self.inputs / "parent_0.png").write_bytes((self.inputs / "parent_1.png").read_bytes())
        self.assert_rejected("mask hash mismatch")

    def test_new_resolved_parent_requires_hash(self):
        del self.documents["component_masks"]["masks"][0]["mask_sha256"]
        self.save_documents()
        self.assert_rejected("mask hash mismatch")

    def test_changed_mask_manifest_rejects_lifting_binding(self):
        self.documents["component_masks"]["frames"][0]["width"] = 11
        self.save_documents(bind=False)
        self.assert_rejected("does not bind current component_masks")

    def test_resolved_declaration_never_falls_back_to_valid_legacy_binding(self):
        original = copy.deepcopy(self.documents["lifting_report"])
        for invalid in (None, {}, {"component_masks": {"path": "inputs/component_masks.json", "sha256": "0" * 64}},
                        {"component_masks": {"path": "inputs/cameras.json", "sha256": original["resolved_files"]["component_masks"]["sha256"]}}):
            with self.subTest(invalid=invalid):
                report = copy.deepcopy(original)
                report["source_files"]["component_masks"] = original["resolved_files"]["component_masks"]
                report["resolved_files"] = invalid
                self.documents["lifting_report"] = report
                self.save_documents(bind=False)
                self.assert_rejected("resolved_files|does not bind current component_masks")

    def test_legacy_manifest_binding_is_explicitly_weaker_not_retroactively_hashed(self):
        del self.documents["lifting_report"]["resolved_files"]
        del self.documents["component_masks"]["acceptance_authority"]
        for record in self.documents["component_masks"]["masks"]:
            del record["mask_sha256"]
        self.save_documents()
        manifest, report = self.output_documents(adapter.run(self.args))
        self.assertEqual(report["component_masks_binding"], "source_files")
        self.assertIn("source_manifest_only", report["legacy_hash_scope"])
        self.assertTrue(all(not view["parent_hash_bound_by_lifting"] for view in manifest["objects"][0]["views"]))

    def test_parts_without_parent_cannot_be_used_as_object_alpha(self):
        self.documents["component_masks"]["masks"] = [record for record in self.documents["component_masks"]["masks"]
                                                        if not (record["frame_id"] == "0" and record["component_id"] == "__object__")]
        self.save_documents()
        self.assert_rejected("exactly one verified parent mask")

    def test_duplicate_parent_is_rejected(self):
        self.documents["component_masks"]["masks"].append(copy.deepcopy(self.documents["component_masks"]["masks"][0]))
        self.save_documents()
        self.assert_rejected("exactly one verified parent mask")

    def test_only_isolated_observation_frames_are_eligible_even_if_other_masks_are_larger(self):
        self.documents["isolated_object_ply"]["objects"][0]["observations"] = [{"frame_id": "0", "mask_path": "inputs/parent_0.png"}]
        self.save_documents()
        manifest, report = self.output_documents(adapter.run(self.args))
        self.assertEqual([view["frame_id"] for view in manifest["objects"][0]["views"]], ["0"])
        self.assertEqual(report["objects"][0]["verified_frame_ids"], ["0"])

    def test_unverified_observation_cannot_be_added_to_isolated_object(self):
        self.documents["isolated_object_ply"]["objects"][0]["observations"].append({"frame_id": "2", "mask_path": "inputs/parent_2.png"})
        self.save_documents()
        self.assert_rejected("not lifting-verified frames")

    def test_acceptance_declarations_and_isolated_geometry_status_are_required(self):
        original = copy.deepcopy(self.documents)
        mutations = [lambda d: d["lifting_report"].update(status="blocked_input_geometry_validation"),
                     lambda d: d["lifting_report"].update(carve_performed=False),
                     lambda d: d["lifting_report"].update(generated_geometry_used=True),
                     lambda d: d["lifting_report"]["tracks"][0].update(accepted=False),
                     lambda d: d["lifting_report"]["tracks"][0].update(accepted=1),
                     lambda d: d["lifting_report"].update(accepted_object_ids=["bed", "bed"]),
                     lambda d: d["isolated_object_ply"]["objects"][0].update(association_status="vlm_identity_hypothesis")]
        for mutation in mutations:
            with self.subTest(mutation=mutations.index(mutation)):
                self.documents = copy.deepcopy(original)
                mutation(self.documents)
                self.save_documents()
                self.assert_rejected("lifting|duplicate|geometrically verified")

    def test_empty_objects_observations_or_parent_are_not_publishable(self):
        import numpy as np
        from PIL import Image

        original = copy.deepcopy(self.documents)
        self.documents["isolated_object_ply"]["objects"] = []
        self.save_documents()
        self.assert_rejected("isolated objects disagree")
        self.documents = copy.deepcopy(original)
        self.documents["isolated_object_ply"]["objects"][0]["observations"] = []
        self.save_documents()
        self.assert_rejected("not lifting-verified frames")
        self.documents = copy.deepcopy(original)
        Image.fromarray(np.zeros((10, 12), dtype=np.uint8)).save(self.inputs / "parent_0.png")
        self.documents["component_masks"]["masks"][0]["mask_sha256"] = adapter.sha256(self.inputs / "parent_0.png")
        self.save_documents()
        self.args.max_views = 1
        self.assert_rejected("parent mask is empty")

    def test_max_views_must_be_positive_integer_and_limits_sorted_verified_views(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value):
                self.args.max_views = value
                self.assert_rejected("max-views must be positive")
        self.args.max_views = 1
        manifest, report = self.output_documents(adapter.run(self.args))
        self.assertEqual([view["frame_id"] for view in manifest["objects"][0]["views"]], ["1"])
        self.assertEqual(report["objects"][0]["verified_frame_ids"], ["1", "0"])

    def test_camera_source_and_mask_raster_must_match_without_resampling(self):
        import numpy as np
        from PIL import Image

        self.documents["component_masks"]["masks"][0]["image_path"] = "inputs/image_1.png"
        self.save_documents()
        self.assert_rejected("source image paths disagree")
        self.documents["component_masks"]["masks"][0]["image_path"] = "inputs/image_0.png"
        Image.fromarray(np.full((5, 6), 255, dtype=np.uint8)).save(self.inputs / "parent_0.png")
        self.documents["component_masks"]["masks"][0]["mask_sha256"] = adapter.sha256(self.inputs / "parent_0.png")
        self.save_documents()
        self.assert_rejected("mask/image raster mismatch")

    def test_two_runs_preserve_old_artifact_bytes_and_legacy_directory(self):
        legacy = self.args.outputs.parent / "assembled" / "old_receipt.json"
        adapter.write_json(legacy, {"history": "preserved"})
        first = adapter.run(self.args)
        old_run = (self.task / first["outputs"]["assembly_report"]["path"]).parent
        before = {path: path.read_bytes() for path in old_run.rglob("*") if path.is_file()}
        legacy_bytes = legacy.read_bytes()
        self.args.max_views = 1
        second = adapter.run(self.args)
        new_run = (self.task / second["outputs"]["assembly_report"]["path"]).parent
        self.assertNotEqual(old_run, new_run)
        self.assertEqual(old_run.parent, self.args.outputs.parent)
        self.assertEqual(new_run.parent, self.args.outputs.parent)
        self.assertTrue(old_run.name.startswith("run-") and new_run.name.startswith("run-"))
        self.assertEqual({path: path.read_bytes() for path in old_run.rglob("*") if path.is_file()}, before)
        self.assertEqual(legacy.read_bytes(), legacy_bytes)
        self.assertEqual(adapter.read_json(self.args.outputs), second)

    def test_failed_retry_does_not_overwrite_successful_receipt_or_output_pointer(self):
        first = adapter.run(self.args)
        report_path = self.task / first["outputs"]["assembly_report"]["path"]
        previous = (report_path.read_bytes(), self.args.outputs.read_bytes())
        self.documents["lifting_report"]["resolved_files"] = {}
        self.save_documents(bind=False)
        with self.assertRaisesRegex(ValueError, "does not bind current component_masks"):
            adapter.run(self.args)
        self.assertEqual((report_path.read_bytes(), self.args.outputs.read_bytes()), previous)

    def test_common_provider_cli_publishes_exact_output_roles(self):
        code = adapter.main(["--task-dir", str(self.task), "--inputs", str(self.args.inputs),
                             "--outputs", str(self.args.outputs), "--max-views", "1"])
        self.assertEqual(code, 0)
        result = adapter.read_json(self.args.outputs)
        self.assertEqual(set(result["outputs"]), {"assembled_object_views", "assembly_report"})
        manifest, _ = self.output_documents(result)
        self.assertEqual(len(manifest["objects"][0]["views"]), 1)


if __name__ == "__main__":
    unittest.main()
