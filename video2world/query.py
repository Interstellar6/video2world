"""Deterministic offline scene cognition resolver for Chinese and English queries."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from video2world.models import Bounds3D, StrictModel, WorldManifest, WorldObject

Language = Literal["zh", "en"]
Intent = Literal["location", "appearance", "unknown"]

APPEARANCE_TERMS = (
    "长什么",
    "什么样",
    "外观",
    "颜色",
    "材质",
    "看起来",
    "look like",
    "looks like",
    "appearance",
    "color",
    "colour",
    "material",
)
LOCATION_TERMS = (
    "在哪里",
    "在哪",
    "哪里",
    "位置",
    "何处",
    "where",
    "located",
    "location",
)
RELATION_LABELS: dict[str, tuple[str, str]] = {
    "on_top_of": ("在{target}上方", "on top of {target}"),
    "supported_by": ("由{target}支撑", "supported by {target}"),
    "inside": ("在{target}内部", "inside {target}"),
    "near": ("在{target}附近", "near {target}"),
    "left_of": ("在{target}左侧", "to the left of {target}"),
    "right_of": ("在{target}右侧", "to the right of {target}"),
    "in_front_of": ("在{target}前方", "in front of {target}"),
    "behind": ("在{target}后方", "behind {target}"),
}
CHINESE_NUMERALS = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}
ENGLISH_ORDINALS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}


class QueryResult(StrictModel):
    status: Literal["resolved", "ambiguous", "not_found", "unsupported", "unavailable"]
    query: str
    language: Language
    intent: Intent
    resolved_object_id: str | None = None
    resolved_scoped_id: str | None = None
    candidate_ids: list[str] = Field(default_factory=list)
    answer: str | None = None
    focus_bbox: Bounds3D | None = None
    evidence: list[str] = Field(default_factory=list)
    reason: str | None = None


@dataclass(frozen=True)
class Match:
    item: WorldObject
    score: int
    evidence: str


def detect_language(text: str) -> Language:
    return "zh" if re.search(r"[\u3400-\u9fff]", text) else "en"


def normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in normalized if character.isalnum())


def detect_intent(text: str) -> Intent:
    lowered = unicodedata.normalize("NFKC", text).casefold()
    if any(term in lowered for term in APPEARANCE_TERMS):
        return "appearance"
    if any(term in lowered for term in LOCATION_TERMS):
        return "location"
    return "unknown"


def _object_terms(item: WorldObject) -> list[tuple[str, int, str]]:
    values: list[tuple[str, int, str]] = [
        (item.scoped_id, 1000, "scoped_id"),
        (item.id, 900, "id"),
        (item.name.zh or "", 800, "name.zh"),
        (item.name.en or "", 800, "name.en"),
    ]
    values.extend((alias, 600, "alias") for alias in item.aliases)
    values.append((item.category, 400, "category"))
    unique: dict[str, tuple[str, int, str]] = {}
    for value, score, source in values:
        key = normalize(value)
        if not key:
            continue
        previous = unique.get(key)
        if previous is None or score > previous[1]:
            unique[key] = (value, score, source)
    return list(unique.values())


def _matches(manifest: WorldManifest, query: str) -> list[Match]:
    normalized_query = normalize(query)
    matches: list[Match] = []
    for item in manifest.objects:
        best: Match | None = None
        for value, base_score, source in _object_terms(item):
            key = normalize(value)
            if key not in normalized_query:
                continue
            candidate = Match(item, base_score + min(len(key), 99), f"{source}:{value}")
            if best is None or candidate.score > best.score:
                best = candidate
        if best:
            matches.append(best)
    return matches


def _requested_ordinal(query: str) -> int | None:
    lowered = unicodedata.normalize("NFKC", query).casefold()
    chinese = re.search(r"第\s*([一二两三四五六七八九十]|\d+)", lowered)
    if chinese:
        raw = chinese.group(1)
        return int(raw) if raw.isdigit() else CHINESE_NUMERALS.get(raw)
    for term, value in ENGLISH_ORDINALS.items():
        if re.search(rf"\b{term}\b", lowered):
            return value
    numbered_instance = re.search(
        r"(?:plant|植物|nightstand|床头柜|lamp|灯|window|窗|door|门|bed|床)"
        r"\s*[_#-]?\s*0*(\d+)\b",
        lowered,
    )
    return int(numbered_instance.group(1)) if numbered_instance else None


def _instance_number(item: WorldObject) -> tuple[int, str]:
    match = re.search(r"(\d+)$", item.id)
    return (int(match.group(1)) if match else 10**9, item.scoped_id)


def _resolve_object(
    manifest: WorldManifest, query: str
) -> tuple[WorldObject | None, list[str], str]:
    matches = _matches(manifest, query)
    if not matches:
        return None, [], "no manifest object id, name, category, or alias matched"
    highest = max(match.score for match in matches)
    candidates = [match.item for match in matches if match.score == highest]
    if len(candidates) == 1:
        return candidates[0], [candidates[0].scoped_id], "unique highest-specificity match"

    ordinal = _requested_ordinal(query)
    if ordinal is not None:
        ordered = sorted(candidates, key=_instance_number)
        if 1 <= ordinal <= len(ordered):
            selected = ordered[ordinal - 1]
            return selected, [item.scoped_id for item in ordered], f"resolved ordinal {ordinal}"
        return None, [item.scoped_id for item in ordered], f"ordinal {ordinal} is out of range"
    return (
        None,
        sorted(item.scoped_id for item in candidates),
        "multiple instances share the best alias",
    )


def _relation_target(manifest: WorldManifest, item: WorldObject, language: Language) -> str | None:
    by_scoped = manifest.object_index()
    by_id: dict[str, list[WorldObject]] = {}
    for candidate in manifest.objects:
        by_id.setdefault(candidate.id, []).append(candidate)
    pieces: list[str] = []
    for relation in item.relations:
        if not relation.verified or relation.confidence < 0.5:
            continue
        template = RELATION_LABELS.get(relation.predicate)
        if template is None:
            continue
        target_name = (
            relation.target_label.for_language(language) if relation.target_label else None
        )
        if not target_name and relation.target_object_id:
            target = by_scoped.get(relation.target_object_id)
            if target is None:
                same_id = by_id.get(relation.target_object_id, [])
                target = same_id[0] if len(same_id) == 1 else None
            if target:
                target_name = target.name.for_language(language)
        if target_name:
            pieces.append(template[0 if language == "zh" else 1].format(target=target_name))
    if not pieces:
        return None
    separator = ", " if language == "zh" else "; "
    return separator.join(pieces)


def _location_answer(
    manifest: WorldManifest,
    item: WorldObject,
    language: Language,
) -> tuple[str | None, list[str]]:
    direct = item.description.location.for_language(language) if item.description.location else None
    if direct:
        return direct, ["description.location"]
    relation = _relation_target(manifest, item, language)
    if relation:
        return relation, ["verified_relations"]
    if item.bbox_scene:
        center = ", ".join(f"{value:.4g}" for value in item.bbox_scene.center)
        if language == "zh":
            answer = f"其审核边界框中心为 ({center}), 坐标系是 {item.bbox_scene.frame_id}。"
        else:
            answer = (
                f"Its reviewed bounding-box center is ({center}) in {item.bbox_scene.frame_id}."
            )
        return answer, ["bbox_scene.center"]
    return None, []


def _appearance_answer(item: WorldObject, language: Language) -> tuple[str | None, list[str]]:
    for field_name in ("appearance", "detailed", "short"):
        value = getattr(item.description, field_name)
        if value:
            answer = value.for_language(language)
            if answer:
                caveat = (
                    item.description.fidelity_caveat.for_language(language)
                    if item.description.fidelity_caveat
                    else None
                )
                if caveat:
                    separator = "; "
                    answer = f"{answer}{separator}{caveat}"
                return answer, [f"description.{field_name}"]
    return None, []


def query_world(
    manifest: WorldManifest,
    query: str,
    *,
    language: Literal["auto", "zh", "en"] = "auto",
) -> QueryResult:
    effective_language: Language = detect_language(query) if language == "auto" else language
    intent = detect_intent(query)
    item, candidates, resolution_reason = _resolve_object(manifest, query)
    if item is None:
        status = "ambiguous" if candidates else "not_found"
        return QueryResult(
            status=status,
            query=query,
            language=effective_language,
            intent=intent,
            candidate_ids=candidates,
            reason=resolution_reason,
        )
    base = {
        "query": query,
        "language": effective_language,
        "intent": intent,
        "resolved_object_id": item.id,
        "resolved_scoped_id": item.scoped_id,
        "candidate_ids": candidates,
        "focus_bbox": item.bbox_scene,
    }
    if intent == "unknown":
        return QueryResult(
            status="unsupported",
            **base,
            reason="offline resolver only supports location and appearance intents",
        )
    if intent == "location":
        answer, evidence = _location_answer(manifest, item, effective_language)
    else:
        answer, evidence = _appearance_answer(item, effective_language)
    if answer is None:
        return QueryResult(
            status="unavailable",
            **base,
            reason=f"resolved object has no reviewed {intent} evidence",
        )
    return QueryResult(
        status="resolved",
        **base,
        answer=answer,
        evidence=evidence,
        reason=resolution_reason,
    )
