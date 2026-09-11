"""Propose parent tracks from frame-local observations and supplied geometry."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re


class PhysicalTrackError(ValueError):
    pass


def _identifier(value, label):
    if not isinstance(value, str) or not value or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in value):
        raise PhysicalTrackError(f"{label} must be a nonempty identifier without whitespace or control characters")
    return value


def _json_value(value, label, active=None):
    if value is None or type(value) in (str, int, bool):
        return
    if type(value) is float and math.isfinite(value):
        return
    if isinstance(value, list) or isinstance(value, dict) and all(isinstance(k, str) for k in value):
        active = set() if active is None else active
        if id(value) in active:
            raise PhysicalTrackError(f"{label} contains circular metadata")
        active.add(id(value))
        try:
            for item in value.values() if isinstance(value, dict) else value:
                _json_value(item, label, active)
        finally:
            active.remove(id(value))
        return
    raise PhysicalTrackError(f"{label} must contain only finite JSON values and string object keys")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def group_physical_tracks(observations: list[dict], pair_evidence: list[dict], *, min_views: int = 2) -> dict:
    """Return provisional groups; callers must compose and validate before carving.

    Categories match exactly. Source IDs never create edges. Raw visible-negative
    edges remain evidence, not an early veto: authorized component composition
    may repair an incomplete parent mask before the final geometry check.
    """
    if not isinstance(observations, list) or not isinstance(pair_evidence, list):
        raise PhysicalTrackError("observations and pair_evidence must be arrays")
    if type(min_views) is not int or min_views < 2:
        raise PhysicalTrackError("min_views must be an integer of at least two")
    by_id = {}
    for index, observation in enumerate(observations):
        if not isinstance(observation, dict):
            raise PhysicalTrackError(f"observation {index} must be an object")
        _json_value(observation, f"observation {index}")
        for key in ("observation_id", "frame_id", "source_object_id"):
            _identifier(observation.get(key), f"observation {index}.{key}")
        category = observation.get("category")
        if (not isinstance(category, str) or not category or category != category.strip()
                or any(ord(c) < 32 or ord(c) == 127 for c in category)):
            raise PhysicalTrackError(f"observation {index}.category must be nonempty text without edge whitespace or controls")
        identifier = observation["observation_id"]
        if identifier in by_id:
            raise PhysicalTrackError(f"duplicate observation_id: {identifier}")
        by_id[identifier] = copy.deepcopy(observation)

    roots = {identifier: identifier for identifier in by_id}

    def find(identifier):
        while roots[identifier] != identifier:
            roots[identifier] = roots[roots[identifier]]
            identifier = roots[identifier]
        return identifier

    pairs, seen_pairs = [], set()
    for index, pair in enumerate(pair_evidence):
        if not isinstance(pair, dict):
            raise PhysicalTrackError(f"pair {index} must be an object")
        _json_value(pair, f"pair {index}")
        left = _identifier(pair.get("id_a"), f"pair {index}.id_a")
        right = _identifier(pair.get("id_b"), f"pair {index}.id_b")
        if left not in by_id or right not in by_id:
            raise PhysicalTrackError(f"pair {index} references an unknown observation")
        if left == right:
            raise PhysicalTrackError(f"pair {index} compares an observation with itself")
        endpoints = tuple(sorted((left, right)))
        if endpoints in seen_pairs:
            raise PhysicalTrackError(f"duplicate unordered pair: {endpoints}")
        seen_pairs.add(endpoints)
        for key in ("positive_match", "visible_conflict"):
            if type(pair.get(key)) is not bool:
                raise PhysicalTrackError(f"pair {index}.{key} must be boolean")
        compatible = by_id[left]["category"] == by_id[right]["category"]
        used = compatible and pair["positive_match"]
        pairs.append({"pair_id": "pair_" + _digest(endpoints), "input": copy.deepcopy(pair),
                      "scope_compatible": compatible, "used_for_connectivity": used,
                      "ignored_reasons": [] if compatible else ["different_category"]})
        if used:
            a, b = find(left), find(right)
            roots[max(a, b)] = min(a, b)

    members_by_root = {}
    source_roots = {}
    for identifier in sorted(by_id):
        root = find(identifier)
        members_by_root.setdefault(root, []).append(identifier)
        source_roots.setdefault(by_id[identifier]["source_object_id"], set()).add(root)
    pairs.sort(key=lambda item: tuple(sorted((item["input"]["id_a"], item["input"]["id_b"]))))
    incident = {root: [] for root in members_by_root}
    for pair in pairs:
        for root in {find(pair["input"]["id_a"]), find(pair["input"]["id_b"])}:
            incident[root].append(pair)

    generated_ids = {}
    for root, members in members_by_root.items():
        category = by_id[members[0]]["category"]
        slug = re.sub(r"[^a-z0-9]+", "_", category.lower()).strip("_")[:32] or "object"
        generated_ids[root] = f"physical_{slug}_{_digest([category, members])[:48]}"
    if len(set(generated_ids.values())) != len(generated_ids):
        raise PhysicalTrackError("generated physical identity collision")
    reserved_ids = set(generated_ids.values())

    tracks = []
    for root, members in members_by_root.items():
        frames = {}
        for identifier in members:
            frames.setdefault(by_id[identifier]["frame_id"], []).append(identifier)
        same_frame = [{"frame_id": frame, "observation_ids": frames[frame]}
                      for frame in sorted(frames) if len(frames[frame]) > 1]
        internal, external = [], []
        for pair in incident[root]:
            if pair["input"]["visible_conflict"]:
                target = internal if find(pair["input"]["id_a"]) == find(pair["input"]["id_b"]) else external
                target.append(pair["pair_id"])
        source_ids = sorted({by_id[identifier]["source_object_id"] for identifier in members})
        identity_reasons = []
        if len(source_ids) != 1:
            identity_reasons.append("multiple_source_ids_in_group")
        if any(len(source_roots[source_id]) != 1 for source_id in source_ids):
            identity_reasons.append("source_id_reused_across_groups")
        if external:
            identity_reasons.append("cross_group_visible_conflict")
        if len(source_ids) == 1:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", source_ids[0]):
                identity_reasons.append("source_id_not_safe_for_output")
            if source_ids[0] in reserved_ids and source_ids[0] != generated_ids[root]:
                identity_reasons.append("source_id_collides_with_generated_identity")
        preserved = not identity_reasons
        reasons = []
        if same_frame:
            reasons.append("same_frame_multiple_observations")
        if len(frames) < min_views:
            reasons.append("insufficient_distinct_views")
        diagnostics = (["raw_internal_visible_conflict_requires_final_validation"] if internal else [])
        if external:
            diagnostics.append("raw_external_visible_conflict_preserved")
        tracks.append({
            "object_id": source_ids[0] if preserved else generated_ids[root],
            "track_id": generated_ids[root], "category": by_id[members[0]]["category"],
            "member_ids": list(members), "frame_ids": sorted(frames), "view_count": len(frames),
            "source_object_ids": source_ids, "source_id_preserved": preserved,
            "identity_assignment_reasons": identity_reasons,
            "lineage": [copy.deepcopy(by_id[identifier]) for identifier in members],
            "pair_evidence_ids": sorted(pair["pair_id"] for pair in incident[root]),
            "positive_pair_ids": sorted(pair["pair_id"] for pair in incident[root] if pair["used_for_connectivity"]),
            "internal_conflict_pair_ids": sorted(internal), "external_conflict_pair_ids": sorted(external),
            "same_frame_conflicts": same_frame, "reasons": reasons, "diagnostics": diagnostics,
            "eligible_for_final_validation": not reasons,
            "status": "ambiguous_same_frame" if same_frame else "insufficient_evidence" if reasons else "provisional",
            "identity_scope": "provisional_parent_track", "accepted": False,
            "geometry_validated": False, "final_acceptance": False,
        })
    tracks.sort(key=lambda item: (item["category"], item["member_ids"]))
    if len({track["object_id"] for track in tracks}) != len(tracks):
        raise PhysicalTrackError("duplicate output object_id")
    return {"schema_version": "1.0", "acceptance_scope": "provisional_parent_grouping_only",
            "category_matching": "exact_input_category", "association_report_required": True,
            "component_composition_before_final_association": True, "geometry_computed": False,
            "geometry_validated": False, "final_acceptance": False, "complete_recall_claimed": False,
            "min_views": min_views, "observations": [by_id[key] for key in sorted(by_id)],
            "tracks": tracks, "pair_evidence": pairs}
