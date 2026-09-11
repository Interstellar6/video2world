from __future__ import annotations

import copy
import itertools
import unittest
from unittest.mock import patch

from world_modeling.physical_tracks import PhysicalTrackError, group_physical_tracks


def observation(identifier, frame, source="bed", category="bed", **metadata):
    return {"observation_id": identifier, "frame_id": frame, "source_object_id": source,
            "category": category, **metadata}


def pair(left, right, positive=True, conflict=False, **metadata):
    return {"id_a": left, "id_b": right, "positive_match": positive, "visible_conflict": conflict, **metadata}


class PhysicalTrackTests(unittest.TestCase):
    def test_legacy_bed_id_survives_raw_internal_conflict_without_early_veto(self):
        observations = [observation("a", "f0"), observation("b", "f1"), observation("c", "f2")]
        evidence = [pair("a", "b"), pair("b", "c"), pair("a", "c", positive=False, conflict=True)]
        result = group_physical_tracks(observations, evidence)
        track, = result["tracks"]
        self.assertEqual(track["object_id"], "bed")
        self.assertTrue(track["source_id_preserved"])
        self.assertTrue(track["eligible_for_final_validation"])
        self.assertFalse(track["accepted"])
        self.assertFalse(track["geometry_validated"])
        self.assertFalse(track["final_acceptance"])
        self.assertEqual(track["member_ids"], ["a", "b", "c"])
        self.assertEqual(len(track["internal_conflict_pair_ids"]), 1)
        self.assertEqual(len(track["positive_pair_ids"]), 2)
        self.assertEqual(track["reasons"], [])
        self.assertTrue(result["component_composition_before_final_association"])
        self.assertTrue(result["association_report_required"])

    def test_same_category_two_real_instances_are_separate_with_complete_lineage(self):
        observations = [observation("a", "f0", "left", "nightstand", mask_path="left0.png"),
                        observation("b", "f1", "left", "nightstand"),
                        observation("c", "f0", "right", "nightstand"),
                        observation("d", "f1", "right", "nightstand")]
        result = group_physical_tracks(observations, [pair("a", "b"), pair("c", "d")])
        self.assertEqual([t["object_id"] for t in result["tracks"]], ["left", "right"])
        self.assertTrue(all(t["eligible_for_final_validation"] for t in result["tracks"]))
        self.assertEqual(result["tracks"][0]["lineage"], observations[:2])
        self.assertEqual({m for t in result["tracks"] for m in t["member_ids"]}, {"a", "b", "c", "d"})

    def test_same_source_id_in_two_groups_is_not_identity(self):
        observations = [observation("a", "f0"), observation("b", "f1"),
                        observation("c", "f0"), observation("d", "f1")]
        tracks = group_physical_tracks(observations, [pair("a", "b"), pair("c", "d")])["tracks"]
        self.assertEqual(len(tracks), 2)
        self.assertEqual(len({t["object_id"] for t in tracks}), 2)
        for track in tracks:
            self.assertTrue(track["object_id"].startswith("physical_bed_"))
            self.assertFalse(track["source_id_preserved"])
            self.assertIn("source_id_reused_across_groups", track["identity_assignment_reasons"])

    def test_multiple_source_ids_in_one_group_use_member_hash_not_arbitrary_alias(self):
        observations = [observation("a", "f0", "bed_01"), observation("b", "f1", "bed_07")]
        track, = group_physical_tracks(observations, [pair("a", "b")])["tracks"]
        self.assertEqual(track["source_object_ids"], ["bed_01", "bed_07"])
        self.assertTrue(track["object_id"].startswith("physical_bed_"))
        self.assertIn("multiple_source_ids_in_group", track["identity_assignment_reasons"])

    def test_cross_category_positive_edges_are_ignored_not_discarded(self):
        observations = [observation("a", "f0", "item", "bed"), observation("b", "f1", "item", "lamp")]
        result = group_physical_tracks(observations, [pair("a", "b", conflict=True)])
        self.assertEqual(len(result["tracks"]), 2)
        edge, = result["pair_evidence"]
        self.assertFalse(edge["scope_compatible"])
        self.assertFalse(edge["used_for_connectivity"])
        self.assertEqual(edge["ignored_reasons"], ["different_category"])
        self.assertTrue(edge["input"]["visible_conflict"])
        self.assertTrue(all(t["external_conflict_pair_ids"] == [edge["pair_id"]] for t in result["tracks"]))

    def test_same_frame_transitive_multiplicity_is_not_split_to_hide_ambiguity(self):
        observations = [observation("a", "f0"), observation("b", "f1"), observation("c", "f0")]
        track, = group_physical_tracks(observations, [pair("a", "b"), pair("b", "c")])["tracks"]
        self.assertEqual(track["member_ids"], ["a", "b", "c"])
        self.assertEqual(track["same_frame_conflicts"], [{"frame_id": "f0", "observation_ids": ["a", "c"]}])
        self.assertEqual(track["status"], "ambiguous_same_frame")
        self.assertFalse(track["eligible_for_final_validation"])

    def test_positive_and_negative_same_edge_preserves_connectivity_and_audit(self):
        observations = [observation("a", "f0"), observation("b", "f1")]
        result = group_physical_tracks(observations, [pair("a", "b", conflict=True)])
        track, = result["tracks"]
        self.assertTrue(track["eligible_for_final_validation"])
        self.assertEqual(track["internal_conflict_pair_ids"], track["positive_pair_ids"])
        self.assertFalse(track["accepted"])

    def test_external_conflict_preserved_for_both_groups_prevents_legacy_alias(self):
        observations = [observation("a", "f0", "left"), observation("b", "f1", "left"),
                        observation("c", "f0", "right"), observation("d", "f1", "right")]
        evidence = [pair("a", "b"), pair("c", "d"), pair("a", "d", positive=False, conflict=True)]
        result = group_physical_tracks(observations, evidence)
        conflict, = [p for p in result["pair_evidence"] if p["input"]["visible_conflict"]]
        self.assertEqual(len(result["tracks"]), 2)
        for track in result["tracks"]:
            self.assertEqual(track["external_conflict_pair_ids"], [conflict["pair_id"]])
            self.assertTrue(track["eligible_for_final_validation"])
            self.assertFalse(track["source_id_preserved"])
            self.assertIn("cross_group_visible_conflict", track["identity_assignment_reasons"])

    def test_no_positive_edges_preserve_unaccepted_single_view_observations(self):
        observations = [observation("a", "f0"), observation("b", "f1")]
        result = group_physical_tracks(observations, [pair("a", "b", positive=False)])
        self.assertEqual(len(result["tracks"]), 2)
        self.assertEqual(len(result["pair_evidence"]), 1)
        for track in result["tracks"]:
            self.assertFalse(track["eligible_for_final_validation"])
            self.assertFalse(track["accepted"])
            self.assertEqual(track["reasons"], ["insufficient_distinct_views"])

    def test_input_order_does_not_change_ids_members_or_diagnostics(self):
        observations = [observation("c", "f2", "other"), observation("a", "f0"), observation("b", "f1")]
        evidence = [pair("a", "b"), pair("b", "c"), pair("a", "c", positive=False, conflict=True)]
        reference = group_physical_tracks(observations, evidence)
        for observed_order in itertools.permutations(observations):
            for edge_order in itertools.permutations(evidence):
                self.assertEqual(group_physical_tracks(list(observed_order), list(edge_order)), reference)

    def test_generated_id_depends_on_category_and_members_not_aliases_or_edge_orientation(self):
        observations = [observation("a", "f0", "first"), observation("b", "f1", "second")]
        first = group_physical_tracks(observations, [pair("a", "b")])["tracks"][0]
        renamed = [{**observations[0], "source_object_id": "third"}, {**observations[1], "source_object_id": "fourth"}]
        second = group_physical_tracks(renamed, [pair("b", "a", conflict=True)])["tracks"][0]
        self.assertEqual(first["object_id"], second["object_id"])
        changed_category = [{**o, "category": "lamp"} for o in observations]
        third = group_physical_tracks(changed_category, [pair("a", "b")])["tracks"][0]
        self.assertNotEqual(first["object_id"], third["object_id"])

    def test_source_alias_cannot_collide_with_another_generated_id(self):
        originals = [observation("a", "f0", "a"), observation("b", "f1", "b")]
        generated = group_physical_tracks(originals, [pair("a", "b")])["tracks"][0]["object_id"]
        observations = originals + [observation("c", "f0", generated), observation("d", "f1", generated)]
        tracks = group_physical_tracks(observations, [pair("a", "b"), pair("c", "d")])["tracks"]
        self.assertEqual(len({t["object_id"] for t in tracks}), 2)
        self.assertIn("source_id_collides_with_generated_identity", tracks[1]["identity_assignment_reasons"])

    def test_unsafe_source_output_name_is_retained_only_in_lineage(self):
        observations = [observation("a", "f0", "../bed"), observation("b", "f1", "../bed")]
        track, = group_physical_tracks(observations, [pair("a", "b")])["tracks"]
        self.assertTrue(track["object_id"].startswith("physical_bed_"))
        self.assertEqual(track["source_object_ids"], ["../bed"])
        self.assertIn("source_id_not_safe_for_output", track["identity_assignment_reasons"])

    def test_long_category_generated_ids_fit_cross_module_path_identifier_limit(self):
        category = "very_long_category_" * 30
        observations = [observation("a", "f0", "first", category), observation("b", "f1", "second", category)]
        track, = group_physical_tracks(observations, [pair("a", "b")])["tracks"]
        self.assertEqual(len(track["object_id"]), 90)
        self.assertEqual(track["category"], category)
        self.assertRegex(track["object_id"], r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
        other = [{**item, "category": category + "other"} for item in observations]
        other_track, = group_physical_tracks(other, [pair("a", "b")])["tracks"]
        self.assertNotEqual(track["object_id"], other_track["object_id"])

    def test_preserved_source_id_limit_is_exactly_96_and_longer_names_keep_lineage(self):
        for size, preserved in ((96, True), (97, False), (400, False)):
            with self.subTest(size=size):
                source = "s" * size
                observations = [observation("a", "f0", source), observation("b", "f1", source)]
                track, = group_physical_tracks(observations, [pair("a", "b")])["tracks"]
                self.assertEqual(track["source_id_preserved"], preserved)
                self.assertLessEqual(len(track["object_id"]), 96)
                self.assertLessEqual(len(track["track_id"]), 90)
                self.assertEqual(track["source_object_ids"], [source])
                self.assertEqual(track["lineage"], observations)
                if not preserved:
                    self.assertIn("source_id_not_safe_for_output", track["identity_assignment_reasons"])

    def test_truncated_digest_collision_still_fails_closed(self):
        observations = [observation("a", "f0", "first"), observation("b", "f1", "second")]
        with patch("world_modeling.physical_tracks._digest", side_effect=["a" * 48 + "b" * 16, "a" * 48 + "c" * 16]):
            with self.assertRaisesRegex(PhysicalTrackError, "generated physical identity collision"):
                group_physical_tracks(observations, [])

    def test_category_text_is_exact_not_implicitly_aliased(self):
        observations = [observation("a", "f0", "a", "bedside table"), observation("b", "f1", "b", "nightstand")]
        result = group_physical_tracks(observations, [pair("a", "b")])
        self.assertEqual(len(result["tracks"]), 2)
        self.assertEqual(result["category_matching"], "exact_input_category")

    def test_inputs_and_directional_metadata_are_preserved_without_aliasing(self):
        observations = [observation("a", "f0", metadata={"pixels": [1, 2]}), observation("b", "f1")]
        evidence = [pair("b", "a", geometry={"forward": [.3, .4]})]
        before = copy.deepcopy((observations, evidence))
        result = group_physical_tracks(observations, evidence)
        self.assertEqual(result["pair_evidence"][0]["input"], evidence[0])
        result["pair_evidence"][0]["input"]["geometry"]["forward"].append(.9)
        result["tracks"][0]["lineage"][0]["metadata"]["pixels"].append(3)
        result["observations"][0]["metadata"]["pixels"].append(4)
        self.assertEqual((observations, evidence), before)

    def test_minimum_views_cannot_be_weakened(self):
        observations = [observation("a", "f0"), observation("b", "f1")]
        track, = group_physical_tracks(observations, [pair("a", "b")], min_views=3)["tracks"]
        self.assertFalse(track["eligible_for_final_validation"])
        for value in (0, 1, -1, True, False, 2.0, "2", None):
            with self.subTest(value=value), self.assertRaises(PhysicalTrackError):
                group_physical_tracks(observations, [], min_views=value)

    def test_empty_input_never_fabricates_accepted_tracks(self):
        result = group_physical_tracks([], [])
        self.assertEqual(result["tracks"], [])
        self.assertEqual(result["observations"], [])
        self.assertFalse(result["geometry_computed"])
        self.assertFalse(result["final_acceptance"])
        self.assertFalse(result["complete_recall_claimed"])

    def test_observation_fields_and_duplicates_are_validated(self):
        valid = observation("a", "f0")
        for key in valid:
            for value in (None, "", " ", " leading", "trailing ", "a\n", "a\x00b", 0, True, [], {}):
                with self.subTest(key=key, value=value), self.assertRaises(PhysicalTrackError):
                    group_physical_tracks([{**valid, key: value}], [])
            with self.subTest(missing=key), self.assertRaises(PhysicalTrackError):
                group_physical_tracks([{k: v for k, v in valid.items() if k != key}], [])
        with self.assertRaisesRegex(PhysicalTrackError, "duplicate observation_id"):
            group_physical_tracks([valid, {**valid, "frame_id": "f1"}], [])

    def test_unknown_self_duplicate_and_reversed_duplicate_pairs_are_rejected(self):
        observations = [observation("a", "f0"), observation("b", "f1")]
        for evidence in ([pair("a", "missing")], [pair("a", "a")],
                         [pair("a", "b"), pair("a", "b")], [pair("a", "b"), pair("b", "a")]):
            with self.subTest(evidence=evidence), self.assertRaises(PhysicalTrackError):
                group_physical_tracks(observations, evidence)

    def test_array_record_and_boolean_types_are_not_coerced(self):
        observations = [observation("a", "f0"), observation("b", "f1")]
        for value in (None, {}, "value", (), 1, True):
            with self.subTest(observations=value), self.assertRaises(PhysicalTrackError):
                group_physical_tracks(value, [])
            with self.subTest(pairs=value), self.assertRaises(PhysicalTrackError):
                group_physical_tracks(observations, value)
        for value in (None, [], "record", 0, True):
            with self.subTest(observation=value), self.assertRaises(PhysicalTrackError):
                group_physical_tracks([value], [])
            with self.subTest(pair=value), self.assertRaises(PhysicalTrackError):
                group_physical_tracks(observations, [value])
        for key in ("positive_match", "visible_conflict"):
            for value in (0, 1, "true", "false", None, [], {}):
                with self.subTest(key=key, value=value), self.assertRaises(PhysicalTrackError):
                    group_physical_tracks(observations, [{**pair("a", "b"), key: value}])

    def test_non_json_nonfinite_and_circular_metadata_is_rejected(self):
        circular = {}
        circular["self"] = circular
        for value in (float("nan"), float("inf"), b"bytes", {1: "key"}, (1, 2), {"set"}, circular):
            with self.subTest(observation=value), self.assertRaises(PhysicalTrackError):
                group_physical_tracks([observation("a", "f0", metadata=value)], [])
            with self.subTest(pair=value), self.assertRaises(PhysicalTrackError):
                group_physical_tracks([observation("a", "f0"), observation("b", "f1")], [pair("a", "b", metadata=value)])
        shared = {"threshold": .25}
        result = group_physical_tracks([observation("a", "f0", metadata=[shared, shared])], [])
        self.assertEqual(result["observations"][0]["metadata"], [shared, shared])


if __name__ == "__main__":
    unittest.main()
