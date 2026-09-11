"""Group frame-local component observations using caller-supplied geometric evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import math


class ComponentTrackError(ValueError):
    pass


def _identifier(value, label: str) -> str:
    if not isinstance(value, str) or not value or any(character.isspace() or ord(character) < 32 or ord(character) == 127 for character in value):
        raise ComponentTrackError(f"{label} must be a nonempty identifier without whitespace or control characters")
    return value


def _json_value(value, label: str, active: set[int] | None = None) -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if isinstance(value, list) or isinstance(value, dict) and all(isinstance(key, str) for key in value):
        active = set() if active is None else active
        if id(value) in active:
            raise ComponentTrackError(f"{label} contains circular metadata")
        active.add(id(value))
        try:
            for item in value.values() if isinstance(value, dict) else value:
                _json_value(item, label, active)
        finally:
            active.remove(id(value))
        return
    raise ComponentTrackError(f"{label} must contain only finite JSON values and string object keys")


def _stable_id(prefix: str, members: list[str]) -> str:
    encoded = json.dumps(sorted(members), separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()}"


def group_component_tracks(observations: list[dict], pair_evidence: list[dict], *, min_views: int = 2) -> dict:
    """Return auditable candidate components, not final geometry or recall certification.

    The caller computes positive matches and visible conflicts using its existing
    geometry thresholds. Acceptance here only permits a candidate to proceed to
    the caller's final association_report; it does not verify physical identity.
    """
    if not isinstance(observations, list) or not isinstance(pair_evidence, list):
        raise ComponentTrackError("observations and pair_evidence must be arrays")
    if type(min_views) is not int or min_views < 2:
        raise ComponentTrackError("min_views must be an integer of at least two")
    required = ("observation_id", "frame_id", "parent_object_id", "component_group_id")
    by_id = {}
    for index, observation in enumerate(observations):
        if not isinstance(observation, dict):
            raise ComponentTrackError(f"observation {index} must be a JSON object")
        _json_value(observation, f"observation {index}")
        for field in required:
            _identifier(observation.get(field), f"observation {index}.{field}")
        identifier = observation["observation_id"]
        if identifier in by_id:
            raise ComponentTrackError(f"duplicate observation_id: {identifier}")
        by_id[identifier] = copy.deepcopy(observation)

    roots = {identifier: identifier for identifier in by_id}

    def find(identifier):
        while roots[identifier] != identifier:
            roots[identifier] = roots[roots[identifier]]
            identifier = roots[identifier]
        return identifier

    pairs = []
    seen_pairs = set()
    for index, pair in enumerate(pair_evidence):
        if not isinstance(pair, dict):
            raise ComponentTrackError(f"pair {index} must be a JSON object")
        _json_value(pair, f"pair {index}")
        left = _identifier(pair.get("id_a"), f"pair {index}.id_a")
        right = _identifier(pair.get("id_b"), f"pair {index}.id_b")
        if left not in by_id or right not in by_id:
            raise ComponentTrackError(f"pair {index} references an unknown observation")
        if left == right:
            raise ComponentTrackError(f"pair {index} cannot compare an observation with itself")
        endpoints = tuple(sorted((left, right)))
        if endpoints in seen_pairs:
            raise ComponentTrackError(f"duplicate unordered pair: {endpoints}")
        seen_pairs.add(endpoints)
        for field in ("positive_match", "visible_conflict"):
            if type(pair.get(field)) is not bool:
                raise ComponentTrackError(f"pair {index}.{field} must be a boolean")
        ignored_reasons = [f"different_{field}" for field in ("parent_object_id", "component_group_id")
                           if by_id[left][field] != by_id[right][field]]
        used = not ignored_reasons and pair["positive_match"]
        pairs.append({
            "pair_id": _stable_id("pair", list(endpoints)), "input": copy.deepcopy(pair),
            "scope_compatible": not ignored_reasons, "used_for_connectivity": used,
            "ignored_reasons": ignored_reasons,
        })
        if used:
            root_left, root_right = find(left), find(right)
            roots[max(root_left, root_right)] = min(root_left, root_right)

    members_by_root = {}
    for identifier in sorted(by_id):
        members_by_root.setdefault(find(identifier), []).append(identifier)
    pairs.sort(key=lambda item: tuple(sorted((item["input"]["id_a"], item["input"]["id_b"]))))
    internal_pairs = {root: [] for root in members_by_root}
    for pair in pairs:
        root_left, root_right = find(pair["input"]["id_a"]), find(pair["input"]["id_b"])
        if root_left == root_right:
            internal_pairs[root_left].append(pair)

    tracks = []
    for root, members in members_by_root.items():
        frames = {}
        for identifier in members:
            frames.setdefault(by_id[identifier]["frame_id"], []).append(identifier)
        same_frame_conflicts = [{"frame_id": frame, "observation_ids": frames[frame]}
                                for frame in sorted(frames) if len(frames[frame]) > 1]
        evidence = internal_pairs[root]
        conflicts = sorted(item["pair_id"] for item in evidence if item["input"]["visible_conflict"])
        reasons = []
        if same_frame_conflicts:
            reasons.append("same_frame_multiple_observations")
        if conflicts:
            reasons.append("internal_visible_conflict")
        if len(frames) < min_views:
            reasons.append("insufficient_distinct_views")
        first = by_id[members[0]]
        tracks.append({
            "track_id": _stable_id("component_track", members), "member_ids": members,
            "parent_object_id": first["parent_object_id"], "component_group_id": first["component_group_id"],
            "frame_ids": sorted(frames), "view_count": len(frames), "accepted": not reasons,
            "status": "ambiguous" if same_frame_conflicts or conflicts else "insufficient_evidence" if reasons else "candidate",
            "reasons": reasons, "same_frame_conflicts": same_frame_conflicts, "conflict_pair_ids": conflicts,
            "pair_evidence_ids": sorted(item["pair_id"] for item in evidence),
            "positive_pair_ids": sorted(item["pair_id"] for item in evidence if item["used_for_connectivity"]),
        })
    tracks.sort(key=lambda item: (item["parent_object_id"], item["component_group_id"], item["member_ids"]))
    return {
        "schema_version": "1.0", "acceptance_scope": "candidate_grouping_only",
        "association_report_required": True, "geometry_computed": False, "complete_recall_claimed": False,
        "min_views": min_views, "observations": [by_id[identifier] for identifier in sorted(by_id)],
        "tracks": tracks, "pair_evidence": pairs,
    }
