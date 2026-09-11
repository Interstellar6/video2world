from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

from world_modeling.object_lineage import (
    ObjectLineageError, lineage_metadata, resolve_object_descriptions, validate_object_lineages,
)
from world_modeling.provider_io import write_json


class ObjectLineageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.task = Path(self.temporary.name)
        self.records = []
        source_observations = []
        for frame in ("0", "1"):
            (self.task / f"image_{frame}.png").write_bytes(f"image_{frame}".encode())
            (self.task / f"mask_{frame}.png").write_bytes(f"mask_{frame}".encode())
            self.records.append({"frame_id": frame, "object_id": f"observation_{frame}",
                                 "image_path": f"image_{frame}.png", "category": "bed", "description": f"source {frame}"})
            source_observations.append({"frame_id": frame, "source_object_id": f"observation_{frame}",
                "source_mask_path": f"mask_{frame}.png", "source_mask_sha256": self.digest(f"mask_{frame}.png"),
                "source_image_path": f"image_{frame}.png", "source_image_sha256": self.digest(f"image_{frame}.png")})
        self.descriptions = self.task / "descriptions.json"
        write_json(self.descriptions, {"objects": self.records})
        self.obj = {"object_id": "physical_bed", "observations": [{"frame_id": "0"}, {"frame_id": "1"}],
                    "source_observations": source_observations,
                    "source_descriptions": {"path": "descriptions.json", "sha256": self.digest("descriptions.json")}}

    def digest(self, name):
        return hashlib.sha256((self.task / name).read_bytes()).hexdigest()

    def resolve(self, obj=None):
        return resolve_object_descriptions(self.task, self.obj if obj is None else obj, self.descriptions)

    def save_descriptions(self):
        write_json(self.descriptions, {"objects": self.records})
        self.obj["source_descriptions"]["sha256"] = self.digest("descriptions.json")

    def test_physical_id_returns_raw_records_without_mutation(self):
        self.assertEqual(self.resolve(), self.records)
        self.assertEqual(self.resolve()[0]["object_id"], "observation_0")
        self.assertEqual(resolve_object_descriptions(self.task, self.obj), self.records)

    def test_legacy_join_remains_exact_object_id_without_frame_filter(self):
        legacy = {"object_id": "observation_0", "observations": []}
        self.assertEqual(self.resolve(legacy), self.records[:1])
        self.assertEqual(resolve_object_descriptions(self.task, legacy), [])
        self.assertEqual(lineage_metadata(legacy), {})

    def test_two_same_category_objects_do_not_mix_descriptions(self):
        first, second = copy.deepcopy(self.obj), copy.deepcopy(self.obj)
        first["source_observations"] = first["source_observations"][:1]
        second.update(object_id="physical_other", source_observations=second["source_observations"][1:])
        result = validate_object_lineages(self.task, [first, second], self.descriptions)
        self.assertEqual(result, {"physical_bed": self.records[:1], "physical_other": self.records[1:]})

    def test_either_field_alone_never_falls_back_to_legacy(self):
        for key in ("source_observations", "source_descriptions"):
            with self.subTest(key=key):
                obj = copy.deepcopy(self.obj)
                del obj[key]
                with self.assertRaisesRegex(ObjectLineageError, "both"):
                    self.resolve(obj)
                with self.assertRaisesRegex(ObjectLineageError, "both"):
                    lineage_metadata(obj)

    def test_empty_or_malformed_declarations_reject(self):
        for key, values in (("source_observations", [None, {}, []]), ("source_descriptions", [None, [], {}])):
            for value in values:
                obj = copy.deepcopy(self.obj)
                obj[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.resolve(obj)

    def test_changed_descriptions_file_rejects(self):
        write_json(self.descriptions, {"objects": []})
        with self.assertRaisesRegex(ObjectLineageError, "descriptions SHA256 mismatch"):
            self.resolve()

    def test_duplicate_source_description_key_rejects(self):
        self.records.append(copy.deepcopy(self.records[0]))
        self.save_descriptions()
        with self.assertRaisesRegex(ObjectLineageError, "duplicate frame/object"):
            self.resolve()

    def test_duplicate_source_observation_key_rejects(self):
        self.obj["source_observations"].append(copy.deepcopy(self.obj["source_observations"][0]))
        with self.assertRaisesRegex(ObjectLineageError, "duplicate source frame/object"):
            self.resolve()

    def test_source_vlm_id_is_not_an_identity_fallback(self):
        self.obj["source_observations"][0]["source_object_id"] = "bed"
        self.records[0]["source_vlm_object_id"] = "bed"
        self.save_descriptions()
        with self.assertRaisesRegex(ObjectLineageError, "description is missing"):
            self.resolve()

    def test_source_frame_must_be_in_accepted_set_when_present(self):
        self.obj["observations"] = [{"frame_id": "0"}]
        with self.assertRaisesRegex(ObjectLineageError, "outside accepted"):
            self.resolve()

    def test_empty_and_duplicate_accepted_frames_reject(self):
        for observations in ([], [{"frame_id": "0"}, {"frame_id": "0"}]):
            self.obj["observations"] = observations
            with self.subTest(observations=observations), self.assertRaises(ObjectLineageError):
                self.resolve()

    def test_completed_objects_can_carry_lineage_without_observations(self):
        del self.obj["observations"]
        self.assertEqual(self.resolve(), self.records)

    def test_actual_source_image_and_mask_hashes_are_checked(self):
        for field in ("source_image_sha256", "source_mask_sha256"):
            obj = copy.deepcopy(self.obj)
            obj["source_observations"][0][field] = "0" * 64
            with self.subTest(field=field), self.assertRaisesRegex(ObjectLineageError, "SHA256 mismatch"):
                self.resolve(obj)

    def test_same_image_bytes_at_another_path_do_not_match_description(self):
        alternate = self.task / "alternate.png"
        alternate.write_bytes((self.task / "image_0.png").read_bytes())
        self.obj["source_observations"][0]["source_image_path"] = alternate.name
        with self.assertRaisesRegex(ObjectLineageError, "image path differs"):
            self.resolve()

    def test_source_binding_cannot_switch_to_another_description_input(self):
        alternate = self.task / "alternate.json"
        alternate.write_bytes(self.descriptions.read_bytes())
        self.obj["source_descriptions"]["path"] = alternate.name
        with self.assertRaisesRegex(ObjectLineageError, "differs from current input"):
            self.resolve()

    def test_escape_through_symlink_rejects(self):
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside) / "image.png"
            target.write_bytes((self.task / "image_0.png").read_bytes())
            (self.task / "escape.png").symlink_to(target)
            self.obj["source_observations"][0]["source_image_path"] = "escape.png"
            with self.assertRaisesRegex(ValueError, "escapes task"):
                self.resolve()

    def test_task_symlink_is_canonicalized_without_false_escape(self):
        alias = self.task.parent / (self.task.name + "-alias")
        alias.symlink_to(self.task, target_is_directory=True)
        self.addCleanup(alias.unlink)
        self.assertEqual(resolve_object_descriptions(alias, self.obj, alias / "descriptions.json"), self.records)

    def test_cross_object_claim_cannot_be_renamed_or_hidden_by_changed_mask(self):
        other = copy.deepcopy(self.obj)
        other["object_id"] = "physical_other"
        other["source_observations"][0].update(source_mask_path="mask_1.png", source_mask_sha256=self.digest("mask_1.png"))
        with self.assertRaisesRegex(ObjectLineageError, "multiple physical objects"):
            validate_object_lineages(self.task, [self.obj, other], self.descriptions)

    def test_copying_and_changing_description_artifact_cannot_double_claim_image(self):
        alternate = self.task / "alternate.json"
        write_json(alternate, {"objects": self.records, "extra": "different container bytes"})
        other = copy.deepcopy(self.obj)
        other.update(object_id="physical_other", source_descriptions={"path": alternate.name, "sha256": self.digest(alternate.name)})
        with self.assertRaisesRegex(ObjectLineageError, "multiple physical objects"):
            validate_object_lineages(self.task, [self.obj, other])

    def test_duplicate_physical_object_id_rejects(self):
        with self.assertRaisesRegex(ObjectLineageError, "duplicate physical"):
            validate_object_lineages(self.task, [self.obj, self.obj])

    def test_copied_metadata_is_independent_of_input(self):
        result = lineage_metadata(self.obj)
        result["source_observations"][0]["frame_id"] = "changed"
        self.assertEqual(self.obj["source_observations"][0]["frame_id"], "0")
