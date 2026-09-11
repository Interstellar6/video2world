import copy
import contextlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

try:
    from PIL import Image
except ImportError:
    Image = None

SPEC = importlib.util.spec_from_file_location("qwen_understanding", Path(__file__).resolve().parents[1] / "scripts/qwen_understanding.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class QwenContractTest(unittest.TestCase):
    def test_categories_are_parsed_before_inference(self):
        self.assertEqual(MODULE.parse_categories(None), [])
        self.assertEqual(MODULE.parse_categories(" Bed, table lamp "), ["bed", "table lamp"])
        for value in ("", "bed,", "bed,,lamp", "bed,BED"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                MODULE.parse_categories(value)

    def test_unrequested_categories_fail_instead_of_silent_filter_or_relabel(self):
        payload = copy.deepcopy(MODULE.SCHEMA)
        before = copy.deepcopy(payload)
        with self.assertRaisesRegex(ValueError, "outside requested categories"):
            MODULE.validate_response(payload, 1280, 720, ["lamp"])
        self.assertEqual(payload, before)

    def test_scoped_queries_keep_all_instances_without_previous_id_priming(self):
        image = types.SimpleNamespace(size=(1280, 720))
        responses, prompts = [], []
        for category in ("bed", "lamp"):
            obj = copy.deepcopy(MODULE.SCHEMA["objects"][0])
            obj.update(category=category, object_id="object_1")
            objects = [obj]
            if category == "lamp":
                second = copy.deepcopy(obj)
                second["object_id"] = "object_2"
                objects.append(second)
            responses.append(json.dumps({"objects": objects}))
        def generate(_image, prompt):
            prompts.append(prompt)
            return responses.pop(0)
        output, queries = MODULE.ground_frame(image, ["bed", "lamp"], [{"object_id": "previous_identity_marker"}],
            generate, component_inventory=False, mode="category_scoped", frame_index=7)
        self.assertEqual([obj["category"] for obj in output["objects"]], ["bed", "lamp", "lamp"])
        self.assertEqual([obj["object_id"] for obj in output["objects"]],
                         ["observation_000007_000", "observation_000007_001", "observation_000007_002"])
        self.assertEqual([obj["source_vlm_object_id"] for obj in output["objects"]], ["object_1", "object_1", "object_2"])
        self.assertTrue(all(obj["components"][0]["parent_object_id"] == obj["object_id"] for obj in output["objects"]))
        self.assertTrue(all("previous_identity_marker" not in prompt for prompt in prompts))
        self.assertEqual([query["categories"] for query in queries], [["bed"], ["lamp"]])
        self.assertEqual(queries[0]["grounding_inference"]["objects"][0]["object_id"], "object_1")

    def test_scoped_query_failure_preserves_raw_attempts_without_partial_objects(self):
        image = types.SimpleNamespace(size=(1280, 720))
        wrong = '{"objects": null}'
        responses = [json.dumps(MODULE.SCHEMA), wrong, wrong]
        output, queries = MODULE.ground_frame(image, ["bed", "lamp"], [], lambda *_: responses.pop(0),
            component_inventory=True, mode="category_scoped", frame_index=0)
        self.assertIsNone(output)
        self.assertEqual(len(queries), 2)
        self.assertEqual(len(queries[1]["attempts"]), 2)
        self.assertEqual(queries[1]["attempts"][0]["raw_response"], wrong)
        self.assertIn("objects array", queries[1]["attempts"][1]["validation_error"])

    def test_scoped_extra_objects_are_explicitly_audited_not_relabelled(self):
        payload = copy.deepcopy(MODULE.SCHEMA)
        lamp = copy.deepcopy(payload["objects"][0])
        lamp.update(object_id="lamp_1", category="lamp")
        payload["objects"].append(lamp)
        original = copy.deepcopy(payload)
        result, queries = MODULE.ground_frame(types.SimpleNamespace(size=(1280, 720)), ["lamp"], [],
            lambda *_: json.dumps(payload), component_inventory=True, mode="category_scoped", frame_index=0)
        self.assertEqual(payload, original)
        self.assertEqual([obj["category"] for obj in result["objects"]], ["lamp"])
        self.assertEqual(len(queries[0]["grounding_inference"]["objects"]), 2)
        self.assertEqual(queries[0]["excluded_objects"][0]["object"]["category"], "bed")
        self.assertEqual(queries[0]["excluded_objects"][0]["reason"], "outside_current_category_query")

    def test_scoped_queries_require_an_explicit_category_inventory(self):
        with self.assertRaisesRegex(ValueError, "explicit categories"):
            MODULE.ground_frame(types.SimpleNamespace(size=(1280, 720)), [], [], lambda *_: "",
                component_inventory=True, mode="category_scoped", frame_index=0)

    def test_joint_mode_retains_legacy_ids_and_does_not_claim_new_physical_identity(self):
        result, queries = MODULE.ground_frame(types.SimpleNamespace(size=(1280, 720)), [], [{"object_id": "previous_id"}],
            lambda *_: json.dumps(MODULE.SCHEMA), component_inventory=True, mode="joint", frame_index=7)
        self.assertEqual(result["objects"][0]["object_id"], "bed_01")
        self.assertNotIn("identity_scope", result["objects"][0])
        self.assertIn("previous_id", queries[0]["prompt"])

    def test_bare_array_and_partial_visibility_alias_are_normalized(self):
        import json
        objects = copy.deepcopy(MODULE.SCHEMA["objects"])
        objects[0]["components"][0]["visibility"] = "partially visible"
        output = MODULE.validate_response(MODULE.parse_response(json.dumps(objects)), 1280, 720)
        self.assertEqual(output["objects"][0]["components"][0]["visibility"], "partial")

    def test_malformed_array_items_fail_with_validation_error(self):
        for payload in ({"objects": [None]}, {"objects": [{**copy.deepcopy(MODULE.SCHEMA["objects"][0]), "components": [None]}]}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                MODULE.validate_response(payload, 1280, 720)

    def test_fenced_json_is_parsed_and_parent_is_explicit(self):
        import json
        payload = MODULE.parse_response("```json\n" + json.dumps(MODULE.SCHEMA) + "\n```")
        output = MODULE.validate_response(payload, 1280, 720)
        self.assertEqual(output["objects"][0]["components"][0]["parent_object_id"], "bed_01")

    def test_out_of_frame_box_is_not_silently_accepted(self):
        payload = copy.deepcopy(MODULE.SCHEMA)
        payload["objects"][0]["bbox_xyxy"][3] = 900
        with self.assertRaisesRegex(ValueError, "outside"):
            MODULE.validate_response(payload, 1280, 720)

    def test_a_repeated_instance_is_dropped_and_recorded(self):
        payload = copy.deepcopy(MODULE.SCHEMA)
        payload["objects"].append(copy.deepcopy(payload["objects"][0]))
        output = MODULE.validate_response(payload, 1280, 720)
        self.assertEqual(len(output["objects"]), 1)
        self.assertEqual(output["duplicate_objects_dropped"],
                         [{"object_id": "bed_01", "category": "bed",
                           "reason": "repeats an instance already accepted in this frame"}])

    def test_two_instances_sharing_one_label_get_distinct_suffixed_ids(self):
        payload = copy.deepcopy(MODULE.SCHEMA)
        second = copy.deepcopy(payload["objects"][0])
        second["bbox_xyxy"] = [500, 400, 900, 700]
        payload["objects"].append(second)
        output = MODULE.validate_response(payload, 1280, 720)
        ids = [obj["object_id"] for obj in output["objects"]]
        self.assertEqual(ids, ["bed_01", "bed_01_2"])
        self.assertEqual(output["identifier_rewrites"][0]["reason"],
                         "two distinct instances in one frame shared this label")

    def test_a_label_outside_the_identifier_grammar_is_rewritten_with_provenance(self):
        payload = copy.deepcopy(MODULE.SCHEMA)
        payload["objects"][0]["object_id"] = "guitar case"
        payload["objects"][0]["components"][0]["component_id"] = "quilt/comforter"
        output = MODULE.validate_response(payload, 1280, 720)
        self.assertEqual(output["objects"][0]["object_id"], "guitar_case")
        self.assertEqual(output["objects"][0]["source_vlm_object_id"], "guitar case")
        self.assertEqual(output["objects"][0]["components"][0]["component_id"], "quilt_comforter")
        self.assertEqual(output["objects"][0]["components"][0]["source_vlm_component_id"], "quilt/comforter")
        self.assertEqual({entry["kind"] for entry in output["identifier_rewrites"]}, {"object_id", "component_id"})

    def test_genuinely_invalid_identifiers_still_fail(self):
        for value in ("", "   ", None, 7, True, ["bed"]):
            for key in ("object_id", "component_id"):
                payload = copy.deepcopy(MODULE.SCHEMA)
                target = payload["objects"][0] if key == "object_id" else payload["objects"][0]["components"][0]
                target[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    MODULE.validate_response(payload, 1280, 720)
        payload = copy.deepcopy(MODULE.SCHEMA)
        payload["objects"][0]["components"].append(copy.deepcopy(payload["objects"][0]["components"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate component_id"):
            MODULE.validate_response(payload, 1280, 720)

    def test_slug_identifier_always_satisfies_the_declared_grammar(self):
        import re as regex

        grammar = regex.compile(r"[a-z][a-z0-9_]{0,63}")
        for raw in ("guitar case", "Quilt/Comforter", "  pet food bowl  ", "9 volt battery",
                    "a" * 200, "***", "table--lamp", "café table"):
            with self.subTest(raw=raw):
                self.assertRegex(MODULE.slug_identifier(raw, "object"), grammar)
        self.assertEqual(MODULE.slug_identifier("***", "object"), "object")

    def test_invisible_component_has_no_fabricated_box(self):
        payload = copy.deepcopy(MODULE.SCHEMA)
        payload["objects"][0]["components"][0]["visibility"] = "occluded"
        with self.assertRaisesRegex(ValueError, "hidden components"):
            MODULE.validate_response(payload, 1280, 720)

    def test_xyxy_alias_is_normalized_on_objects_and_components(self):
        payload = copy.deepcopy(MODULE.SCHEMA)
        obj = payload["objects"][0]
        for item in (obj, obj["components"][0]):
            item["bbox_2d"] = item.pop("bbox_xyxy")
        result = MODULE.validate_response(payload, 1280, 720)["objects"][0]
        self.assertEqual(result["bbox_xyxy"], [0, 0, 100, 100])
        self.assertEqual(result["components"][0]["bbox_xyxy"], [0, 0, 100, 50])
        self.assertNotIn("bbox_2d", result)

    def test_matching_aliases_are_allowed_but_conflicts_are_rejected(self):
        payload = copy.deepcopy(MODULE.SCHEMA)
        payload["objects"][0]["bbox_2d"] = [0, 0, 100.0, 100]
        MODULE.validate_response(payload, 1280, 720)
        payload["objects"][0]["bbox_2d"] = [0, 0, 99, 100]
        with self.assertRaisesRegex(ValueError, "conflicts"):
            MODULE.validate_response(payload, 1280, 720)

    def test_coordinate_types_finiteness_and_range_are_strict(self):
        boxes = [None, "0,0,1,1", [0, 0, 1], [False, 0, 1, 1], [0, 0, "1", 1],
                 [0, 0, float("nan"), 1], [0, 0, float("inf"), 1], [0, 0, 10**400, 1],
                 [-129, 0, 1, 1], [0, 0, 1409, 1], [1, 0, 1, 1], [0, 2, 1, 1]]
        for key in ("bbox_xyxy", "bbox_2d"):
            for box in boxes:
                with self.subTest(key=key, box=box), self.assertRaises(ValueError):
                    MODULE.validate_box({key: box}, 1280, 720, "test")

    def test_a_boundary_overshoot_within_tolerance_is_clamped_and_recorded(self):
        # Observed on the ScanNet++ DSLR scene: the model returned a bed box whose
        # right edge sat 84 px past the 1764 px raster it was shown. Two identical
        # retries failed, so the overshoot is a model habit, not a transient.
        item = {"bbox_xyxy": [1298, 753, 1848, 1176]}
        self.assertEqual(MODULE.validate_box(item, 1764, 1176, "bed"), [1298, 753, 1764.0, 1176])
        self.assertEqual(item["bbox_clamped_from"], [1298, 753, 1848, 1176])
        # A box that is entirely outside the raster never becomes a sliver.
        for box in ([-400, 10, -200, 200], [2000, 10, 2400, 200], [0, -400, 100, -200]):
            with self.subTest(box=box), self.assertRaises(ValueError):
                MODULE.validate_box({"bbox_xyxy": list(box)}, 1764, 1176, "bed")
        # The alias conflict check still compares the raw coordinates.
        with self.assertRaisesRegex(ValueError, "conflicts"):
            MODULE.validate_box({"bbox_xyxy": [1298, 753, 1848, 1176], "bbox_2d": [1298, 753, 1849, 1176]},
                                1764, 1176, "bed")

    def test_nonfinite_and_non_numeric_confidence_is_rejected(self):
        for value in (True, "0.5", None, float("nan"), float("inf"), -0.1, 1.1, 10**400):
            payload = copy.deepcopy(MODULE.SCHEMA)
            payload["objects"][0]["components"][0]["confidence"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                MODULE.validate_response(payload, 1280, 720)

    def test_non_string_ids_fail_without_type_errors(self):
        for value in (None, [], {}, 1, True):
            for key in ("object_id", "component_id"):
                payload = copy.deepcopy(MODULE.SCHEMA)
                target = payload["objects"][0] if key == "object_id" else payload["objects"][0]["components"][0]
                target[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    MODULE.validate_response(payload, 1280, 720)

    def test_nonfinite_json_constants_are_rejected_before_validation(self):
        for constant in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(constant=constant), self.assertRaisesRegex(ValueError, "non-finite JSON"):
                MODULE.parse_json_response('{"extra": ' + constant + '}')


def inventory_payload():
    # The crop probe emitted two valid pillow regions with the same class ID.
    return {"components": [
        {"component_id": "pillow", "name": "pillow", "description": "White pillow with lace trim on the left",
         "relationship": "bedding", "bbox_xyxy": [62, 128, 140, 188], "visibility": "visible", "confidence": 0.9},
        {"component_id": "pillow", "name": "pillow", "description": "White pillow with lace trim on the right",
         "relationship": "bedding", "bbox_xyxy": [138, 128, 172, 188], "visibility": "partial", "confidence": 0.8},
    ], "nearby_objects": ["lamp", "plant", "nightstand"], "unobserved_components": ["mattress", "frame"]}


class ComponentInventoryContractTest(unittest.TestCase):
    def setUp(self):
        self.obj = copy.deepcopy(MODULE.SCHEMA["objects"][0])

    def test_probe_duplicate_pillows_become_one_class_group_with_provenance(self):
        payload = inventory_payload()
        before = copy.deepcopy(payload)
        result = MODULE.validate_inventory(payload, self.obj, 532, 560)
        self.assertEqual(payload, before)
        self.assertEqual(len(result["components"]), 1)
        part = result["components"][0]
        self.assertEqual(part["component_id"], "pillow")
        self.assertEqual(part["bbox_xyxy"], [62, 128, 172, 188])
        self.assertEqual(part["instance_mode"], "all")
        self.assertEqual(part["visibility"], "partial")
        self.assertEqual(part["confidence"], 0.8)
        self.assertEqual(part["parent_object_id"], "bed_01")
        self.assertFalse(part["physical_identity_claimed"])
        self.assertEqual(part["association_status"], "frame_local_class_hypothesis")
        for index, item in enumerate(before["components"]):
            self.assertIn(item["description"], part["description"])
            self.assertEqual(part["group_provenance"]["entries"][index], {"response_index": index, "component": item})

    def test_duplicate_id_conflicting_name_or_membership_is_rejected(self):
        for key, value in (("name", "quilt"), ("relationship", "structural_part")):
            payload = inventory_payload()
            payload["components"][1][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "conflicting"):
                MODULE.validate_inventory(payload, self.obj, 532, 560)

    def test_same_type_with_separate_instance_ids_is_not_invented_or_accepted(self):
        payload = inventory_payload()
        payload["components"][1]["component_id"] = "pillow_02"
        with self.assertRaisesRegex(ValueError, "one group ID"):
            MODULE.validate_inventory(payload, self.obj, 532, 560)

    def test_probe_slash_id_is_rejected_with_specific_retry_feedback(self):
        payload = inventory_payload()
        payload["components"] = [{**payload["components"][0], "component_id": "quilt/comforter", "name": "quilt/comforter"}]
        with self.assertRaisesRegex(ValueError, "quilt/comforter"):
            MODULE.validate_inventory(payload, self.obj, 532, 560)
        self.assertIn("matching [a-z][a-z0-9_]{0,63}", MODULE.make_inventory_prompt("bed", 532, 560))

    def test_inventory_alias_preserves_raw_alias_in_provenance_only(self):
        payload = inventory_payload()
        for item in payload["components"]:
            item["bbox_2d"] = item.pop("bbox_xyxy")
        result = MODULE.validate_inventory(payload, self.obj, 532, 560)
        part = result["components"][0]
        self.assertNotIn("bbox_2d", part)
        self.assertEqual(part["bbox_xyxy"], [62, 128, 172, 188])
        self.assertIn("bbox_2d", part["group_provenance"]["entries"][0]["component"])
        payload["components"][0]["bbox_xyxy"] = [0, 0, 20, 20]
        with self.assertRaisesRegex(ValueError, "conflicts"):
            MODULE.validate_inventory(payload, self.obj, 532, 560)

    def test_full_visibility_alias_is_explicit_and_preserves_raw_evidence(self):
        payload = inventory_payload()
        payload["components"] = [{**payload["components"][0], "visibility": "full"}]
        result = MODULE.validate_inventory(payload, self.obj, 532, 560)
        self.assertEqual(result["components"][0]["visibility"], "visible")
        self.assertEqual(result["components"][0]["group_provenance"]["entries"][0]["component"]["visibility"], "full")

    def test_unknown_relationship_is_not_guessed_and_has_actionable_feedback(self):
        payload = inventory_payload()
        payload["components"][0].update(component_id="headboard", relationship="bed")
        with self.assertRaisesRegex(ValueError, "headboard.*'bed'.*structural_part"):
            MODULE.validate_inventory(payload, self.obj, 532, 560)

    def test_invalid_inventory_containers_and_unknown_memberships_fail(self):
        payloads = [None, [], {}, {**inventory_payload(), "nearby_objects": [{}]},
                    {**inventory_payload(), "unobserved_components": [""]},
                    {**inventory_payload(), "components": None}, {**inventory_payload(), "components": [None]}]
        for relation in (None, "nearby", True):
            payload = inventory_payload()
            payload["components"][0]["relationship"] = relation
            payloads.append(payload)
        for payload in payloads:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                MODULE.validate_inventory(payload, self.obj, 532, 560)

    def test_bedding_and_independent_objects_cannot_become_nightstand_parts(self):
        obj = {**self.obj, "category": "nightstand"}
        with self.assertRaisesRegex(ValueError, "only valid for a bed"):
            MODULE.validate_inventory(inventory_payload(), obj, 532, 560)
        for name, parent in (("lamp", obj), ("potted plant", obj), ("nightstand", self.obj), ("bedside table", self.obj)):
            payload = inventory_payload()
            payload["components"] = [{**payload["components"][0], "name": name, "relationship": "structural_part"}]
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "nearby_objects"):
                MODULE.validate_inventory(payload, parent, 532, 560)

    def test_lamp_structural_parts_remain_valid_and_nearby_objects_stay_separate(self):
        payload = inventory_payload()
        payload["components"] = [{**payload["components"][0], "component_id": "base", "name": "lamp base",
                                  "relationship": "structural_part"}]
        result = MODULE.validate_inventory(payload, {**self.obj, "category": "lamp"}, 532, 560)
        self.assertEqual([p["component_id"] for p in result["components"]], ["base"])
        self.assertEqual(result["nearby_objects"], payload["nearby_objects"])

    def test_nightstand_structural_part_can_name_its_own_parent_category(self):
        payload = inventory_payload()
        payload["components"] = [{**payload["components"][0], "component_id": "drawer", "name": "nightstand drawer",
                                  "relationship": "structural_part"}]
        result = MODULE.validate_inventory(payload, {**self.obj, "category": "nightstand"}, 532, 560)
        self.assertEqual(result["components"][0]["name"], "nightstand drawer")

    def test_crop_padding_rounds_outward_and_clamps_to_source(self):
        self.assertEqual(MODULE.crop_bounds([20, 30, 80, 90], 100, 100), [11, 21, 89, 99])
        self.assertEqual(MODULE.crop_bounds([0, 0, 100, 100], 100, 100), [0, 0, 100, 100])
        self.assertEqual(MODULE.crop_bounds([0.1, 1.2, 2.4, 3.5], 100, 100), [0, 0, 3, 4])
        self.assertEqual(MODULE.vision_size((539, 566)), (532, 560))
        self.assertEqual(MODULE.vision_size((1, 14)), (28, 28))

    def test_crop_to_source_mapping_deepcopies_all_inventory_structures(self):
        crop = MODULE.validate_inventory(inventory_payload(), self.obj, 532, 560)
        before = copy.deepcopy(crop)
        result = MODULE.inventory_to_source(crop, [530, 126, 1069, 692], (532, 560))
        expected = [62 * 539 / 532 + 530, 128 * 566 / 560 + 126,
                    172 * 539 / 532 + 530, 188 * 566 / 560 + 126]
        self.assertEqual(result["components"][0]["bbox_xyxy"], expected)
        result["components"][0]["group_provenance"]["entries"][0]["component"]["name"] = "mutated"
        result["nearby_objects"].append("extra")
        self.assertEqual(crop, before)

    def test_inventory_prompt_describes_groups_and_independent_neighbors(self):
        prompt = MODULE.make_inventory_prompt("bed", 532, 560)
        for text in ("one GROUP", "ALL visible pillows", "do not assign individual or cross-view identities",
                     '"components": []', "array of STRINGS", "nearby_objects", "structural_part", "bedding"):
            self.assertIn(text, prompt)
        self.assertIn("return components: []", MODULE.make_prompt(1288, 728, None, [], True))
        self.assertNotIn("individual pillows", MODULE.make_prompt(1288, 728, None, [], True))
        self.assertNotIn("return components: []", MODULE.make_prompt(1288, 728, None, []))

    def test_two_schema_retries_preserve_each_prompt_and_response(self):
        responses = iter(["not JSON", '{"components": []}', json.dumps(inventory_payload())])
        accepted, attempts = MODULE.query_with_retries(
            None, "inventory", lambda image, prompt: next(responses),
            lambda text: MODULE.validate_inventory(MODULE.parse_json_response(text), self.obj, 532, 560), max_attempts=3,
        )
        self.assertEqual(len(attempts), 3)
        self.assertEqual(accepted["components"][0]["instance_mode"], "all")
        self.assertEqual(attempts[0]["prompt"], "inventory")
        self.assertIn(attempts[0]["validation_error"], attempts[1]["prompt"])
        self.assertIn(attempts[1]["validation_error"], attempts[2]["prompt"])
        self.assertIsNone(attempts[2]["validation_error"])

    def test_nonbed_prompt_does_not_prime_bed_component_hallucinations(self):
        for category in ("nightstand", "bedside table", "lamp", "chair"):
            with self.subTest(category=category):
                prompt = MODULE.make_inventory_prompt(category, 84, 28)
                self.assertIn("relationship (exact string 'structural_part')", prompt)
                for unrelated in ("pillow", "bedding", "headboard", "quilt", "bed skirt"):
                    self.assertNotIn(unrelated, prompt)
                self.assertIn("return components: []", prompt)
                self.assertIn("do not force uncertain parts or boxes", prompt)
        self.assertIn("tabletop, drawer fronts", MODULE.make_inventory_prompt("nightstand", 84, 28))
        self.assertIn("shade, stem, base", MODULE.make_inventory_prompt("lamp", 84, 28))

    def test_tiny_nightstand_can_report_no_resolvable_parts_without_changing_parent(self):
        parent = {**copy.deepcopy(self.obj), "category": "nightstand"}
        before = copy.deepcopy(parent)
        empty = {"components": [], "nearby_objects": ["lamp"], "unobserved_components": ["legs"]}
        result = MODULE.validate_inventory(empty, parent, 84, 28)
        self.assertEqual(result["components"], [])
        self.assertEqual(parent, before)
        with self.assertRaisesRegex(ValueError, "bedding relationship is only valid for a bed"):
            MODULE.validate_inventory(inventory_payload(), parent, 532, 560)


@unittest.skipIf(Image is None, "Pillow is required for actual source crop checks")
class ComponentInventoryWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.task = Path(self.temp.name)
        self.stage = self.task / "stage"
        self.stage.mkdir()
        self.source = Image.new("RGB", (1280, 720), (43, 78, 122))
        self.source.putpixel((530, 126), (251, 12, 3))
        self.source.save(self.task / "source.png")
        self.obj = copy.deepcopy(MODULE.SCHEMA["objects"][0])
        self.obj["bbox_xyxy"] = [593, 192, 1006, 626]
        self.parameters = {"model": "local-test-fixture", "max_new_tokens": 4096, "do_sample": False}

    def run_inventory(self, generate):
        return MODULE.run_component_inventory(
            self.source, self.obj, generate, stage=self.stage, task_dir=self.task, frame_index=0, frame_id="000000",
            image_path="source.png", model_parameters=self.parameters,
        )

    def test_real_rgb_crop_and_both_coordinate_records_preserve_parent(self):
        original = copy.deepcopy(self.obj)
        inputs = []

        def generate(image, prompt):
            inputs.append(image.copy())
            return json.dumps(inventory_payload())

        merged = self.run_inventory(generate)
        self.assertEqual(self.obj, original)
        self.assertEqual(merged["bbox_xyxy"], original["bbox_xyxy"])
        record = json.loads((self.stage / "inventory_000000_bed_01_response.json").read_text())
        crop = MODULE.crop_bounds(original["bbox_xyxy"], *self.source.size)
        with Image.open(self.task / record["crop_image_path"]) as saved:
            self.assertEqual(saved.tobytes(), self.source.crop(crop).tobytes())
            self.assertEqual(saved.mode, "RGB")
        with Image.open(self.task / record["inference_image_path"]) as saved:
            self.assertEqual(saved.tobytes(), inputs[0].tobytes())
            self.assertEqual(saved.size[0] % 28, 0)
            self.assertEqual(saved.size[1] % 28, 0)
        self.assertEqual(record["raw_inventory"], inventory_payload())
        self.assertEqual(record["inventory_crop"]["components"][0]["bbox_xyxy"], [62, 128, 172, 188])
        self.assertEqual(record["inventory_source"]["components"], merged["components"])
        self.assertEqual(record["model_parameters"], self.parameters)
        self.assertEqual(record["frame_grounding_object"], original)
        self.assertEqual(record["padding"], 0.15)
        self.assertEqual(len(record["source_image_sha256"]), 64)
        self.assertNotEqual(record["inventory_source"]["components"][0]["bbox_xyxy"], [62, 128, 172, 188])

    def test_schema_failure_leaves_three_attempts_and_never_returns_baseline(self):
        calls = []

        def generate(image, prompt):
            calls.append(prompt)
            return '{"components": null}'

        with self.assertRaisesRegex(ValueError, "component inventory failed"):
            self.run_inventory(generate)
        record = json.loads((self.stage / "inventory_000000_bed_01_response.json").read_text())
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(record["attempts"]), 3)
        self.assertFalse(record["accepted_structure"])
        self.assertNotIn("inventory_source", record)

    def main_with_responses(self, responses, enabled):
        responses = iter(responses)
        calls = []

        class Inputs(dict):
            input_ids = types.SimpleNamespace(shape=(1, 1))

            def to(self, device):
                return self

        class Processor:
            def apply_chat_template(self, messages, **kwargs):
                return messages[0]["content"][1]["text"]

            def __call__(self, **kwargs):
                calls.append(kwargs)
                return Inputs()

            def batch_decode(self, *args, **kwargs):
                return [next(responses)]

        class Model:
            def to(self, device):
                return self

            def eval(self):
                return self

            def generate(self, **kwargs):
                class Generated:
                    def __getitem__(self, key):
                        return None
                return Generated()

        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False),
                                      float32="float32", bfloat16="bfloat16", __version__="cpu-test-fixture",
                                      inference_mode=contextlib.nullcontext)
        transformers = types.SimpleNamespace(
            AutoProcessor=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: Processor()),
            Qwen2_5_VLForConditionalGeneration=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: Model()),
        )
        MODULE.write_json(self.task / "frames.json", {"frames": [{"frame_id": "000000", "image_path": "source.png"}]})
        MODULE.write_json(self.task / "inputs.json", {"inputs": {"frames_manifest": {"path": "frames.json"}}})
        argv = ["qwen_understanding.py", "--task-dir", str(self.task), "--inputs", "inputs.json", "--outputs",
                "stage/provider-artifacts.json", "--model", str(self.task / "model")]
        if enabled:
            argv.append("--component-inventory")
        with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}), patch.object(sys, "argv", argv):
            result = MODULE.main()
        return result, calls

    def test_main_default_makes_only_whole_frame_call(self):
        result, calls = self.main_with_responses([json.dumps(MODULE.SCHEMA)], enabled=False)
        self.assertEqual(result, 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(list(self.stage.glob("qwen-*/inventory_*")), [])
        proposals = json.loads(next(self.stage.glob("qwen-*/object_proposals.json")).read_text())
        self.assertFalse(proposals["component_inventory"])

    def test_main_opt_in_inventories_every_parent_after_grounding(self):
        parents = [copy.deepcopy(self.obj), {**copy.deepcopy(self.obj), "object_id": "bed_02"}]
        for obj in parents:
            obj["components"] = []
        result, calls = self.main_with_responses(
            [json.dumps({"objects": parents}), json.dumps(inventory_payload()), json.dumps(inventory_payload())], enabled=True,
        )
        self.assertEqual(result, 0)
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(list(self.stage.glob("qwen-*/inventory_*_response.json"))), 2)
        proposals = json.loads(next(self.stage.glob("qwen-*/object_proposals.json")).read_text())
        self.assertTrue(proposals["component_inventory"])
        for raw, obj in zip(parents, proposals["frames"][0]["objects"]):
            self.assertEqual(obj["bbox_xyxy"], [v * (1280, 720)[i % 2] / (1288, 728)[i % 2]
                                               for i, v in enumerate(raw["bbox_xyxy"])])
            self.assertEqual(obj["components"][0]["instance_mode"], "all")

    def test_main_inventory_failure_does_not_publish_stage_outputs(self):
        with self.assertRaisesRegex(ValueError, "component inventory failed"):
            self.main_with_responses([json.dumps({"objects": [self.obj]}), "invalid", "invalid", "invalid"], enabled=True)
        self.assertFalse((self.stage / "provider-artifacts.json").exists())
        self.assertEqual(list(self.stage.glob("qwen-*/object_proposals.json")), [])
        receipt = json.loads(next(self.stage.glob("qwen-*/inventory_*_response.json")).read_text())
        self.assertEqual(len(receipt["attempts"]), 3)
