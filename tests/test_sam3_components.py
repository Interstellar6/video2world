import importlib.util
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location("sam3_components", Path(__file__).resolve().parents[1] / "scripts/sam3_components.py")
provider = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(provider)


class Shape:
    def __init__(self, *shape):
        self.shape = shape


class Sam3ContractTests(unittest.TestCase):
    def test_segmentation_publishes_only_unresolved_candidate_roles(self):
        from world_modeling.modules.component_segmentation import SPEC as segmentation
        from world_modeling.modules.object_lifting import SPEC as lifting

        self.assertEqual(segmentation.output_names, ("component_mask_candidates", "physical_instance_hypotheses"))
        self.assertTrue(set(segmentation.output_names) <= set(lifting.inputs))
        self.assertFalse({"component_masks", "physical_instance_tracks"} & set(segmentation.output_names))
        self.assertTrue({"component_masks", "physical_instance_tracks"} <= set(lifting.output_names))

    def test_bfloat16_predictions_convert_to_numpy_float32(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        masks, scores = provider.prediction_arrays({
            "masks": torch.ones((1, 1, 2, 2), dtype=torch.bool),
            "scores": torch.tensor([0.75], dtype=torch.bfloat16),
        })
        self.assertEqual(str(scores.dtype), "float32")
        self.assertEqual(float(scores[0]), 0.75)
        self.assertEqual(masks.shape, (1, 1, 2, 2))

    def test_boxes_convert_to_normalized_center_coordinates(self):
        self.assertEqual(provider.normalized_box([20, 10, 60, 30], 100, 50), [0.4, 0.4, 0.4, 0.4])
        for box in ([20, 10, 120, 30], [0, 0, float("nan"), 10], [False, 0, 10, 10]):
            with self.assertRaises(ValueError):
                provider.normalized_box(box, 100, 50)

    def test_base_checkpoint_rejects_discarded_instruction_branches(self):
        expected = {"backbone.weight": Shape(2, 3), "decoder.bias": Shape(4)}
        complete = {**expected, "adapter.layer": Shape(1), "simple_query.weight": Shape(1), "complex_query.weight": Shape(1),
                    "backbone.text.layer.weight": Shape(4), "concept_proj.0.weight": Shape(3, 3)}
        with self.assertRaises(provider.CheckpointCompatibilityError):
            provider.compatible_state(expected, {"model": complete})
        state, audit = provider.compatible_state(expected, {"model": expected})
        self.assertEqual(set(state), set(expected))
        self.assertEqual(audit["checkpoint_origin"], "sam3_complete_state")
        for bad in ({"backbone.weight": Shape(2, 3)}, {**complete, "decoder.bias": Shape(5)}, {**complete, "other.weight": Shape(1)}):
            with self.assertRaises(ValueError):
                provider.compatible_state(expected, bad)
        with self.assertRaises(ValueError):
            provider.compatible_state(expected, {**expected, "backbone.text.layer.weight": Shape(4)})

    def test_native_checkpoint_requires_every_instruction_branch(self):
        expected = {"backbone.weight": Shape(2, 3), "adapter.layer": Shape(1), "simple_query.weight": Shape(1), "complex_query.weight": Shape(1)}
        state, audit = provider.native_compatible_state(expected, {"model": expected})
        self.assertEqual(set(state), set(expected))
        self.assertEqual(audit["checkpoint_origin"], "sam3i_stage3_full_native_instruction_model")
        for removed in ("adapter.layer", "simple_query.weight", "complex_query.weight"):
            with self.assertRaises(provider.CheckpointCompatibilityError):
                provider.native_compatible_state(expected, {k: v for k, v in expected.items() if k != removed})
        with self.assertRaises(provider.CheckpointCompatibilityError):
            provider.native_compatible_state(expected, {**expected, "unused_adapter.weight": Shape(1)})
        with self.assertRaises(provider.CheckpointCompatibilityError):
            provider.native_compatible_state(expected, {**expected, "unused_projector.weight": Shape(1)})

    def test_instruction_token_audit_exposes_truncation(self):
        from types import SimpleNamespace
        tokenizer = SimpleNamespace(encode=lambda prompt: list(range(len(prompt.split()))), decode=lambda tokens: str(tokens))
        model = SimpleNamespace(backbone=SimpleNamespace(language_backbone=SimpleNamespace(tokenizer=tokenizer, context_length=5)))
        audit = provider.instruction_token_audit(model, "one two three four")
        self.assertTrue(audit["truncated"])
        self.assertEqual(audit["effective_text"], "[0, 1, 2]")
        self.assertFalse(provider.instruction_token_audit(model, "one two")["truncated"])

    def test_instruction_uses_description_and_explicit_object_component_scope(self):
        obj = {"object_id": "bed", "category": "bed", "description": "The white bed between the lamps.",
               "bbox_xyxy": [0, 0, 10, 10], "components": [{"component_id": "pillow", "name": "pillow", "description": "The central white pillow.", "bbox_xyxy": [1, 1, 4, 4]}]}
        items = list(provider.proposed_items(obj))
        self.assertIn("complete bed", provider.instruction_prompt(obj, items[0]))
        self.assertIn("between the lamps", provider.instruction_prompt(obj, items[0]))
        self.assertIn("pillow belonging to the bed", provider.instruction_prompt(obj, items[1]))
        obj["segmentation_instruction"] = "the bed including pillows and headboard"
        self.assertEqual(provider.instruction_prompt(obj, items[0]), obj["segmentation_instruction"])

    def test_all_mode_uses_short_concept_unless_explicit_instruction_is_present(self):
        component = {"component_id": "pillow", "name": "white and patterned pillows", "instance_mode": "all",
                     "description": "The white pillows arranged in a row on the bed."}
        obj = {"object_id": "bed", "category": "bed", "bbox_xyxy": [0, 0, 100, 100],
               "components": [component]}
        parent, pillows = list(provider.proposed_items(obj))
        self.assertEqual(provider.instruction_prompt(obj, pillows), "pillow")
        self.assertEqual(pillows["text"], "pillow")
        self.assertEqual(pillows["name"], "white and patterned pillows")
        self.assertIn("complete bed", provider.instruction_prompt(obj, parent))
        component["segmentation_instruction"] = "  the two pillows at the head of the bed  "
        self.assertEqual(provider.instruction_prompt(obj, pillows), "the two pillows at the head of the bed")
        for invalid in ("", "  ", False, 7, []):
            component["segmentation_instruction"] = invalid
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "segmentation_instruction"):
                provider.instruction_prompt(obj, pillows)

    def test_official_video_checkpoint_detector_prefix_is_explicit(self):
        expected = {"backbone.weight": Shape(2, 3)}
        state, audit = provider.compatible_state(expected, {"detector.backbone.weight": Shape(2, 3), "tracker.weight": Shape(8)})
        self.assertEqual(set(state), set(expected))
        self.assertEqual(audit["key_mapping"], "detector_prefix")

    def test_parent_ids_cannot_escape_task_or_mix_components(self):
        with self.assertRaises(ValueError):
            provider.identifier("../bed")
        obj = {"object_id": "bed_01", "category": "bed", "bbox_xyxy": [0, 0, 10, 10],
               "components": [{"component_id": "pillow_01", "name": "pillow", "parent_object_id": "bed_02"}]}
        with self.assertRaises(ValueError):
            list(provider.proposed_items(obj))

    def test_proposed_items_preserve_instance_mode_and_membership_hypothesis(self):
        obj = {"object_id": "bed", "category": "bed", "bbox_xyxy": [0, 0, 100, 100], "components": [
            {"component_id": "pillow", "name": "pillow", "bbox_xyxy": [10, 10, 80, 60],
             "visibility": "partial", "instance_mode": "all", "relationship": "bedding"},
            {"component_id": "headboard", "name": "headboard", "bbox_xyxy": [10, 10, 90, 40]},
        ]}
        parent, pillows, headboard = list(provider.proposed_items(obj))
        self.assertEqual(parent["component_id"], "__object__")
        self.assertEqual(pillows["instance_mode"], "all")
        self.assertEqual(pillows["membership_relation"], "bedding")
        self.assertEqual(pillows["component_id"], "pillow")
        self.assertEqual(pillows["visibility"], "partial")
        self.assertEqual(headboard["instance_mode"], "single")
        self.assertEqual(headboard["membership_relation"], "unspecified")

    def test_invalid_instance_modes_do_not_silently_fall_back_to_single(self):
        for mode in (None, True, False, 0, 1, "", "top_k", "ALL", [], {}):
            obj = {"object_id": "bed", "category": "bed", "bbox_xyxy": [0, 0, 100, 100],
                   "components": [{"component_id": "pillow", "name": "pillow", "instance_mode": mode}]}
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "instance_mode"):
                list(provider.proposed_items(obj))


try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, "numpy not installed")
class Sam3MaskQualityTests(unittest.TestCase):
    def test_all_instances_keeps_each_strictly_above_threshold_candidate(self):
        masks = np.zeros((5, 1, 100, 100), dtype=bool)
        for index, offset in enumerate((10, 25, 40, 55, 70)):
            masks[index, 0, offset:offset + 10, offset:offset + 10] = True
        parent = np.zeros((100, 100), dtype=bool)
        parent[:5, :5] = True
        selected, audits = provider.select_instances(masks, [0.9, 0.251, 0.25, 0.20, 1.1],
                                                     [5, 5, 85, 85], 100, 100, parent_mask=parent)
        self.assertEqual([audit["index"] for _, audit in selected], [0, 1])
        self.assertTrue(all(int(mask.sum()) == 100 for mask, _ in selected))
        for _, audit in selected:
            self.assertEqual(audit["parent_overlap_fraction"], 0)
            self.assertFalse(audit["parent_overlap_enforced"])
            self.assertNotIn("component_outside_parent_mask", audit["rejection_reasons"])
        for index in (2, 3):
            self.assertIn("below_confidence_threshold", audits[index]["rejection_reasons"])
        self.assertIn("invalid_score", audits[4]["rejection_reasons"])

    def test_all_instances_still_requires_own_box_and_non_room_scale_geometry(self):
        masks = np.zeros((3, 100, 100), dtype=bool)
        masks[0, 20:40, 20:40] = True
        masks[1, 80:95, 80:95] = True
        masks[2] = True
        selected, audits = provider.select_instances(masks, [.7, .99, .99], [10, 10, 60, 60], 100, 100)
        self.assertEqual([audit["index"] for _, audit in selected], [0])
        self.assertIn("mask_does_not_match_proposal_box", audits[1]["rejection_reasons"])
        self.assertIn("room_scale_mask", audits[2]["rejection_reasons"])
        self.assertIsNone(audits[0]["parent_overlap_fraction"])
        self.assertFalse(audits[0]["parent_overlap_enforced"])

    def test_all_mode_retains_parent_omission_while_single_keeps_old_rejection(self):
        parent = np.zeros((100, 100), dtype=bool)
        parent[10:40, 10:40] = True
        candidates = np.zeros((1, 100, 100), dtype=bool)
        candidates[0, 50:70, 50:70] = True
        selected, audits = provider.select_instances(candidates, [.8], [50, 50, 70, 70], 100, 100, parent_mask=parent)
        self.assertEqual(len(selected), 1)
        self.assertEqual(audits[0]["parent_overlap_fraction"], 0)
        self.assertFalse(audits[0]["parent_overlap_enforced"])
        mask, _, quality = provider.choose_mask(candidates, [.8], [50, 50, 70, 70], 100, 100, parent_mask=parent)
        self.assertIsNone(mask)
        self.assertIn("component_outside_parent_mask", quality[0]["rejection_reasons"])

    def test_mask_matching_target_wins_over_large_high_score_mask(self):
        masks = np.ones((2, 100, 100), dtype=bool)
        masks[1] = False
        masks[1, 20:50, 20:50] = True
        selected, accepted, quality = provider.choose_mask(masks, [0.99, 0.75], [20, 20, 50, 50], 100, 100)
        self.assertEqual(accepted["index"], 1)
        self.assertEqual(int(selected.sum()), 900)
        self.assertIn("room_scale_mask", quality[0]["rejection_reasons"])

    def test_component_cannot_cross_parent_and_empty_masks_are_rejected(self):
        parent = np.zeros((100, 100), dtype=bool)
        parent[10:50, 10:50] = True
        component = np.zeros((1, 100, 100), dtype=bool)
        component[0, 60:80, 60:80] = True
        mask, _, quality = provider.choose_mask(component, [0.8], [60, 60, 80, 80], 100, 100, parent_mask=parent)
        self.assertIsNone(mask)
        self.assertIn("component_outside_parent_mask", quality[0]["rejection_reasons"])
        self.assertIsNone(provider.choose_mask(np.empty((0, 100, 100)), [], [1, 1, 10, 10], 100, 100)[0])


if __name__ == "__main__":
    unittest.main()
