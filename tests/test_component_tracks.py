from __future__ import annotations

import copy
import itertools
import unittest

from world_modeling.component_tracks import ComponentTrackError, group_component_tracks


def observation(identifier, frame, parent="bed", group="pillow", **metadata):
    return {"observation_id": identifier, "frame_id": frame, "parent_object_id": parent,
            "component_group_id": group, **metadata}


def pair(left, right, positive=True, conflict=False, **metadata):
    return {"id_a": left, "id_b": right, "positive_match": positive, "visible_conflict": conflict, **metadata}


class ComponentTrackTests(unittest.TestCase):
    def test_positive_components_are_candidates_not_verified_geometry(self):
        result = group_component_tracks([observation("a", "f0"), observation("b", "f1")], [pair("a", "b")])
        track, = result["tracks"]
        self.assertTrue(track["accepted"])
        self.assertEqual(track["status"], "candidate")
        self.assertEqual(track["member_ids"], ["a", "b"])
        self.assertEqual(track["frame_ids"], ["f0", "f1"])
        self.assertEqual(track["reasons"], [])
        self.assertEqual(result["acceptance_scope"], "candidate_grouping_only")
        self.assertTrue(result["association_report_required"])
        self.assertFalse(result["geometry_computed"])
        self.assertFalse(result["complete_recall_claimed"])

    def test_transitive_same_frame_conflict_rejects_entire_track(self):
        observations = [observation("a", "f0"), observation("b", "f1"), observation("c", "f0")]
        result = group_component_tracks(observations, [pair("a", "b"), pair("b", "c")])
        track, = result["tracks"]
        self.assertEqual(track["member_ids"], ["a", "b", "c"])
        self.assertFalse(track["accepted"])
        self.assertEqual(track["status"], "ambiguous")
        self.assertEqual(track["reasons"], ["same_frame_multiple_observations"])
        self.assertEqual(track["same_frame_conflicts"], [{"frame_id": "f0", "observation_ids": ["a", "c"]}])

    def test_internal_negative_visible_conflict_rejects_positive_path(self):
        observations = [observation("a", "f0"), observation("b", "f1"), observation("c", "f2")]
        negative = pair("a", "c", positive=False, conflict=True, evidence_path="qa/a-to-c.json")
        result = group_component_tracks(observations, [pair("a", "b"), pair("b", "c"), negative])
        track, = result["tracks"]
        self.assertFalse(track["accepted"])
        self.assertEqual(track["status"], "ambiguous")
        self.assertEqual(track["reasons"], ["internal_visible_conflict"])
        self.assertEqual(len(track["positive_pair_ids"]), 2)
        self.assertEqual(len(track["pair_evidence_ids"]), 3)
        conflict, = [item for item in result["pair_evidence"] if item["pair_id"] in track["conflict_pair_ids"]]
        self.assertEqual(conflict["input"], negative)
        self.assertFalse(conflict["used_for_connectivity"])

    def test_positive_and_conflict_on_one_pair_is_ambiguous_not_silently_selected(self):
        result = group_component_tracks([observation("a", "f0"), observation("b", "f1")],
                                        [pair("a", "b", positive=True, conflict=True)])
        track, = result["tracks"]
        self.assertFalse(track["accepted"])
        self.assertEqual(track["member_ids"], ["a", "b"])
        self.assertEqual(track["reasons"], ["internal_visible_conflict"])

    def test_cross_parent_and_group_positives_never_merge_scopes(self):
        observations = [observation("a", "f0"), observation("b", "f1"),
                        observation("c", "f2", parent="chair"), observation("d", "f3", group="headboard")]
        result = group_component_tracks(observations, [pair("a", "b"), pair("b", "c"), pair("c", "d"), pair("a", "d")])
        self.assertEqual(sorted(track["member_ids"] for track in result["tracks"]), [["a", "b"], ["c"], ["d"]])
        ignored = [item for item in result["pair_evidence"] if not item["scope_compatible"]]
        self.assertEqual(len(ignored), 3)
        self.assertTrue(all(not item["used_for_connectivity"] for item in ignored))
        self.assertTrue(any(item["ignored_reasons"] == ["different_parent_object_id", "different_component_group_id"] for item in ignored))
        self.assertTrue(next(track for track in result["tracks"] if track["member_ids"] == ["a", "b"])["accepted"])

    def test_external_negative_conflict_does_not_poison_other_tracks(self):
        observations = [observation("a", "f0"), observation("b", "f1"),
                        observation("c", "f0"), observation("d", "f1")]
        result = group_component_tracks(observations, [pair("a", "b"), pair("c", "d"), pair("a", "c", positive=False, conflict=True)])
        self.assertEqual(len(result["tracks"]), 2)
        self.assertTrue(all(track["accepted"] for track in result["tracks"]))
        self.assertTrue(all(not track["conflict_pair_ids"] for track in result["tracks"]))
        self.assertEqual(len(result["pair_evidence"]), 3)

    def test_frame_local_ordinals_do_not_define_cross_frame_identity(self):
        observations = [observation("f0_pillow_01", "f0"), observation("f1_pillow_01", "f1"), observation("f1_pillow_07", "f1")]
        result = group_component_tracks(observations, [pair("f0_pillow_01", "f1_pillow_07")])
        self.assertEqual(sorted(track["member_ids"] for track in result["tracks"]),
                         [["f0_pillow_01", "f1_pillow_07"], ["f1_pillow_01"]])
        unlinked = group_component_tracks(observations[:2], [])
        self.assertEqual(len(unlinked["tracks"]), 2)
        self.assertTrue(all(not track["accepted"] for track in unlinked["tracks"]))

    def test_missing_positive_evidence_and_isolated_observations_are_retained(self):
        observations = [observation("a", "f0"), observation("b", "f1"), observation("c", "f2")]
        result = group_component_tracks(observations, [pair("a", "b", positive=False)])
        self.assertEqual(len(result["tracks"]), 3)
        self.assertEqual({member for track in result["tracks"] for member in track["member_ids"]}, {"a", "b", "c"})
        self.assertTrue(all(track["status"] == "insufficient_evidence" for track in result["tracks"]))
        self.assertTrue(all(track["reasons"] == ["insufficient_distinct_views"] for track in result["tracks"]))

    def test_min_views_can_only_strengthen_the_candidate_gate(self):
        observations = [observation("a", "f0"), observation("b", "f1")]
        result = group_component_tracks(observations, [pair("a", "b")], min_views=3)
        track, = result["tracks"]
        self.assertFalse(track["accepted"])
        self.assertEqual(track["reasons"], ["insufficient_distinct_views"])
        for invalid in (0, 1, -1, True, False, 2.0, "2", None):
            with self.subTest(invalid=invalid), self.assertRaises(ComponentTrackError):
                group_component_tracks(observations, [], min_views=invalid)

    def test_rejection_reports_all_reasons_without_dropping_members(self):
        observations = [observation("a", "f0"), observation("b", "f0")]
        track, = group_component_tracks(observations, [pair("a", "b", conflict=True)])["tracks"]
        self.assertEqual(track["member_ids"], ["a", "b"])
        self.assertEqual(track["reasons"], ["same_frame_multiple_observations", "internal_visible_conflict", "insufficient_distinct_views"])

    def test_output_sorting_and_identifiers_are_stable_under_input_permutations(self):
        observations = [observation("c", "f2"), observation("a", "f0"), observation("b", "f1")]
        pairs = [pair("a", "b"), pair("b", "c"), pair("a", "c", positive=False)]
        reference = group_component_tracks(observations, pairs)
        for observed_order in itertools.permutations(observations):
            for pair_order in itertools.permutations(pairs):
                self.assertEqual(group_component_tracks(list(observed_order), list(pair_order)), reference)

    def test_track_identity_depends_on_members_not_edge_order_or_threshold_result(self):
        observations = [observation("a", "f0"), observation("b", "f1"), observation("c", "f2")]
        first = group_component_tracks(observations, [pair("a", "b"), pair("b", "c")])["tracks"][0]
        second = group_component_tracks(observations, [pair("c", "a"), pair("a", "b", conflict=True)])["tracks"][0]
        self.assertEqual(first["track_id"], second["track_id"])
        self.assertTrue(first["accepted"])
        self.assertFalse(second["accepted"])
        smaller = group_component_tracks(observations[:2], [pair("a", "b")])["tracks"][0]
        self.assertNotEqual(first["track_id"], smaller["track_id"])

    def test_pair_orientation_metadata_and_inputs_are_preserved_without_aliasing(self):
        observations = [observation("a", "f0", proposal={"score": 0.8}), observation("b", "f1")]
        pairs = [pair("b", "a", metrics={"b_to_a_support": [0.3, 0.4]})]
        before = copy.deepcopy((observations, pairs))
        result = group_component_tracks(observations, pairs)
        self.assertEqual((observations, pairs), before)
        self.assertEqual(result["pair_evidence"][0]["input"], pairs[0])
        result["pair_evidence"][0]["input"]["metrics"]["b_to_a_support"].append(0.9)
        result["observations"][0]["proposal"]["score"] = 0.1
        self.assertEqual((observations, pairs), before)

    def test_empty_observation_set_has_no_verified_or_fabricated_tracks(self):
        result = group_component_tracks([], [])
        self.assertEqual(result["tracks"], [])
        self.assertEqual(result["observations"], [])
        self.assertEqual(result["pair_evidence"], [])
        self.assertTrue(result["association_report_required"])

    def test_observation_validation_is_strict(self):
        valid = observation("a", "f0")
        for value in (None, {}, "observations", (), 1, True):
            with self.subTest(value=value), self.assertRaises(ComponentTrackError):
                group_component_tracks(value, [])
        for value in (None, [], "a", 0, True):
            with self.subTest(record=value), self.assertRaises(ComponentTrackError):
                group_component_tracks([value], [])
        for field in valid:
            invalid_values = (None, "", " ", "a b", "a\n", "a\x00b", 0, True, [], {})
            for value in invalid_values:
                with self.subTest(field=field, value=value), self.assertRaises(ComponentTrackError):
                    group_component_tracks([{**valid, field: value}], [])
            missing = {key: value for key, value in valid.items() if key != field}
            with self.subTest(missing=field), self.assertRaises(ComponentTrackError):
                group_component_tracks([missing], [])
        with self.assertRaisesRegex(ComponentTrackError, "duplicate observation_id"):
            group_component_tracks([valid, {**valid, "frame_id": "other_frame"}], [])

    def test_pair_validation_rejects_unknown_self_and_duplicate_edges(self):
        observations = [observation("a", "f0"), observation("b", "f1")]
        invalid = [[pair("a", "missing")], [pair("a", "a")],
                   [pair("a", "b"), pair("a", "b")],
                   [pair("a", "b"), pair("b", "a", positive=False, conflict=True)]]
        for pairs in invalid:
            with self.subTest(pairs=pairs), self.assertRaises(ComponentTrackError):
                group_component_tracks(observations, pairs)

    def test_pair_types_and_boolean_fields_are_not_coerced(self):
        observations = [observation("a", "f0"), observation("b", "f1")]
        for value in (None, {}, "pairs", (), 1, True):
            with self.subTest(value=value), self.assertRaises(ComponentTrackError):
                group_component_tracks(observations, value)
        for value in (None, [], "a-b", 0, True):
            with self.subTest(record=value), self.assertRaises(ComponentTrackError):
                group_component_tracks(observations, [value])
        valid = pair("a", "b")
        for field in valid:
            missing = {key: value for key, value in valid.items() if key != field}
            with self.subTest(missing=field), self.assertRaises(ComponentTrackError):
                group_component_tracks(observations, [missing])
        for field in ("positive_match", "visible_conflict"):
            for value in (0, 1, "true", "false", None, [], {}):
                with self.subTest(field=field, value=value), self.assertRaises(ComponentTrackError):
                    group_component_tracks(observations, [{**valid, field: value}])

    def test_non_json_and_nonfinite_metadata_is_rejected(self):
        for value in (float("nan"), float("inf"), b"binary", {1: "numeric key"}, (1, 2), {"value": {"nested"}}):
            with self.subTest(value=value), self.assertRaises(ComponentTrackError):
                group_component_tracks([observation("a", "f0", metadata=value)], [])
            with self.subTest(pair_value=value), self.assertRaises(ComponentTrackError):
                group_component_tracks([observation("a", "f0"), observation("b", "f1")], [pair("a", "b", metadata=value)])

    def test_circular_metadata_is_rejected_but_repeated_values_are_valid(self):
        circular = {}
        circular["self"] = circular
        with self.assertRaisesRegex(ComponentTrackError, "circular"):
            group_component_tracks([observation("a", "f0", metadata=circular)], [])
        with self.assertRaisesRegex(ComponentTrackError, "circular"):
            group_component_tracks([observation("a", "f0"), observation("b", "f1")],
                                   [pair("a", "b", metadata=circular)])
        shared = {"threshold": 0.25}
        result = group_component_tracks([observation("a", "f0", metadata=[shared, shared])], [])
        self.assertEqual(result["observations"][0]["metadata"], [shared, shared])


if __name__ == "__main__":
    unittest.main()
