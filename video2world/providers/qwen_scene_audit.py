"""Narrow, auditable Qwen scene-category supervision.

Qwen is deliberately limited to category-level visual observations. Instance masks,
bounding boxes, depth ordering, occlusion edges, and 3D geometry remain downstream
SAM3 + depth responsibilities. Heavy provider dependencies, including Pillow, Torch,
Transformers, and qwen-vl-utils, are imported only inside the functions that need them.

This module intentionally stays compatible with Python 3.10 provider environments.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import re
import shutil
from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from video2world.hashing import atomic_write_json, atomic_write_text, digest_json, sha256_file
from video2world.models import Sha256, StrictModel

MODEL_REPO = "Qwen/Qwen2.5-VL-3B-Instruct"
SUPPORTED_FRAME_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
FRAME_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
FENCED_JSON_PATTERN = re.compile(r"^```(?:json)?\s*(\{.*\})\s*```$", re.DOTALL | re.IGNORECASE)

SYSTEM_PROMPT = """You are a conservative scene-category auditor for a 3D world-building
pipeline. Report only major semantic categories visibly supported by the supplied labeled
frames. Do not output bounding boxes, masks, instance IDs, occlusion edges, depth order,
hidden geometry, completion actions, or 3D claims. Never copy placeholder values. Return
one JSON object, optionally wrapped by one json code fence, and no other prose."""

PLACEHOLDER_VALUES = {
    "0",
    "n/a",
    "na",
    "none",
    "null",
    "not applicable",
    "not sure",
    "placeholder",
    "tbd",
    "unknown",
    "unspecified",
    "不确定",
    "不详",
    "占位",
    "待定",
    "无",
    "未知",
}

CATEGORY_ALIASES = {
    "bed": "bed",
    "bedframe": "bed",
    "bed frame": "bed",
    "床": "bed",
    "床架": "bed",
    "pillow": "pillow",
    "pillows": "pillow",
    "cushion": "pillow",
    "throw pillow": "pillow",
    "枕头": "pillow",
    "抱枕": "pillow",
    "nightstand": "nightstand",
    "night stand": "nightstand",
    "bedside cabinet": "nightstand",
    "bedside table": "nightstand",
    "床头柜": "nightstand",
    "plant": "plant",
    "plants": "plant",
    "potted plant": "plant",
    "indoor plant": "plant",
    "植物": "plant",
    "盆栽": "plant",
    "绿植": "plant",
    "painting": "painting",
    "paintings": "painting",
    "picture": "painting",
    "wall art": "painting",
    "wall painting": "painting",
    "挂画": "painting",
    "墙画": "painting",
    "画": "painting",
    "lamp": "lamp",
    "table lamp": "lamp",
    "台灯": "lamp",
    "chair": "chair",
    "椅子": "chair",
    "table": "table",
    "桌子": "table",
    "door": "door",
    "门": "door",
    "window": "window",
    "窗": "window",
    "curtain": "curtain",
    "curtains": "curtain",
    "窗帘": "curtain",
    "wall": "wall",
    "walls": "wall",
    "墙": "wall",
    "墙壁": "wall",
    "floor": "floor",
    "地板": "floor",
    "ceiling": "ceiling",
    "天花板": "ceiling",
}

AUDIT_BOUNDARY = (
    "This audit asserts category-level visible evidence only. It does not assert or "
    "synthesize bbox, mask, instance identity, depth order, occlusion, hidden surface, "
    "placement, or 3D geometry."
)
SAM3_DEPTH_HANDOFF = (
    "SAM3 must produce per-instance multi-frame masks; calibrated depth and cameras must "
    "then establish depth order, 2D-to-3D lift, and occlusion evidence."
)


class QwenSceneAuditError(ValueError):
    """Raised when scene-audit evidence or model output is not trustworthy."""


def _normalized_text(value: str) -> str:
    return " ".join(value.strip().casefold().replace("_", " ").replace("-", " ").split())


def _require_meaningful(value: str, field_name: str) -> None:
    normalized = _normalized_text(value).strip(" .,:;!?/\\")
    if not normalized or normalized in PLACEHOLDER_VALUES:
        raise ValueError(f"{field_name} cannot be a placeholder")
    if not any(character.isalpha() or "\u4e00" <= character <= "\u9fff" for character in value):
        raise ValueError(f"{field_name} must contain a meaningful description")


def _require_specific_limitation(value: str, field_name: str) -> None:
    _require_meaningful(value, field_name)
    compact = "".join(value.strip().split())
    if len(compact) < 8:
        raise ValueError(f"{field_name} must state a specific evidence limitation")


def canonicalize_category(value: str) -> str:
    """Map a category label to a deterministic comparison key."""

    _require_meaningful(value, "category")
    normalized = _normalized_text(value)
    if normalized in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[normalized]
    fallback = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", normalized).strip("_")
    if not fallback:
        raise QwenSceneAuditError(f"cannot canonicalize category: {value!r}")
    return fallback


class AuditEvidenceFrame(StrictModel):
    frame_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    uri: str = Field(min_length=1)
    sha256: Sha256 | None = None
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def dimensions_are_paired(self) -> AuditEvidenceFrame:
        if (self.width is None) != (self.height is None):
            raise ValueError("frame width and height must be both present or both absent")
        return self


class SceneCategoryObservation(StrictModel):
    category: str = Field(min_length=1)
    visible_instance_count: int = Field(gt=0)
    evidence_frame_ids: list[str] = Field(min_length=1, max_length=6)
    color_description: str = Field(min_length=2)
    shape_description: str = Field(min_length=2)
    material_description: str = Field(min_length=2)
    support_or_parent_category: str | None = Field(default=None, min_length=1)
    importance: Literal["primary", "secondary", "structure"]
    confidence: float = Field(gt=0, le=1)
    limitations: list[str]

    @model_validator(mode="after")
    def reject_placeholders_and_duplicates(self) -> SceneCategoryObservation:
        _require_meaningful(self.category, "category")
        for field_name in (
            "color_description",
            "shape_description",
            "material_description",
        ):
            _require_meaningful(getattr(self, field_name), field_name)
        if self.support_or_parent_category is not None:
            _require_meaningful(
                self.support_or_parent_category,
                "support_or_parent_category",
            )
        if len(self.evidence_frame_ids) != len(set(self.evidence_frame_ids)):
            raise ValueError("evidence_frame_ids must be unique")
        if len(self.limitations) != len(set(self.limitations)):
            raise ValueError("limitations must be unique")
        for limitation in self.limitations:
            _require_specific_limitation(limitation, "limitation")
        return self


class VlmSceneAuditResponse(StrictModel):
    categories: list[SceneCategoryObservation] = Field(min_length=1)
    scene_limitations: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def reject_duplicate_categories_and_placeholders(self) -> VlmSceneAuditResponse:
        canonical = [canonicalize_category(item.category) for item in self.categories]
        if len(canonical) != len(set(canonical)):
            raise ValueError("categories must be unique after canonicalization")
        if len(self.scene_limitations) != len(set(self.scene_limitations)):
            raise ValueError("scene_limitations must be unique")
        for limitation in self.scene_limitations:
            _require_specific_limitation(limitation, "scene limitation")
        return self


class SceneAuditHandoff(StrictModel):
    vlm_provides_bbox_mask_or_occlusion: Literal[False] = False
    segmentation_provider: Literal["SAM3"] = "SAM3"
    geometry_provider: Literal["calibrated_depth_and_cameras"] = "calibrated_depth_and_cameras"
    contract: str = SAM3_DEPTH_HANDOFF

    @model_validator(mode="after")
    def require_exact_handoff(self) -> SceneAuditHandoff:
        if self.contract != SAM3_DEPTH_HANDOFF:
            raise ValueError("downstream handoff contract is immutable")
        return self


class SceneAudit(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["video2world.scene_category_audit"] = "video2world.scene_category_audit"
    scene_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    created_at: datetime
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    evidence_frames: list[AuditEvidenceFrame] = Field(min_length=1)
    categories: list[SceneCategoryObservation] = Field(min_length=1)
    scene_limitations: list[str] = Field(min_length=1)
    audit_boundary: str = AUDIT_BOUNDARY
    downstream_handoff: SceneAuditHandoff = Field(default_factory=SceneAuditHandoff)

    @model_validator(mode="after")
    def validate_provenance_and_frames(self) -> SceneAudit:
        if self.audit_boundary != AUDIT_BOUNDARY:
            raise ValueError("scene audit boundary is immutable")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must include timezone information")
        frame_ids = [frame.frame_id for frame in self.evidence_frames]
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("evidence frame ids must be unique")
        validate_audit_response(
            VlmSceneAuditResponse(
                categories=self.categories,
                scene_limitations=self.scene_limitations,
            ),
            frame_ids,
        )
        return self


class ExistingAsset(StrictModel):
    asset_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    category: str = Field(min_length=1)
    represented_instance_count: int = Field(gt=0)
    instance_scope: Literal["individual", "aggregate"]
    pipeline_state: Literal[
        "candidate",
        "segmented",
        "completed",
        "review_passed",
        "review_failed",
    ]

    @model_validator(mode="after")
    def validate_scope(self) -> ExistingAsset:
        _require_meaningful(self.category, "asset category")
        if self.instance_scope == "individual" and self.represented_instance_count != 1:
            raise ValueError("individual assets must represent exactly one instance")
        if self.instance_scope == "aggregate" and self.represented_instance_count < 2:
            raise ValueError("aggregate assets must represent at least two instances")
        return self


class ExistingAssetSummary(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["video2world.existing_asset_summary"] = "video2world.existing_asset_summary"
    scene_id: str = Field(min_length=1)
    assets: list[ExistingAsset]
    limitations: list[str]

    @model_validator(mode="after")
    def validate_ids_and_limitations(self) -> ExistingAssetSummary:
        ids = [item.asset_id for item in self.assets]
        if len(ids) != len(set(ids)):
            raise ValueError("asset ids must be unique")
        for limitation in self.limitations:
            _require_meaningful(limitation, "asset summary limitation")
        return self


class CoverageCategory(StrictModel):
    canonical_category: str = Field(min_length=1)
    observed_visible_instance_count: int = Field(gt=0)
    existing_represented_instance_count: int = Field(ge=0)
    existing_individual_instance_count: int = Field(ge=0)
    matched_asset_ids: list[str]
    coverage_status: Literal["missing", "undercovered", "covered"]
    next_action: Literal["segment", "instance_split", "complete", "review", "none"]
    evidence_frame_ids: list[str] = Field(min_length=1)
    requires_sam3_depth: bool
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_status_action(self) -> CoverageCategory:
        expected_sam3 = self.next_action in {"segment", "instance_split"}
        if self.requires_sam3_depth != expected_sam3:
            raise ValueError("requires_sam3_depth must match the next action")
        if self.coverage_status == "missing" and self.next_action != "segment":
            raise ValueError("missing categories must be segmented")
        if self.coverage_status == "covered" and self.next_action == "instance_split":
            raise ValueError("covered categories cannot require instance splitting")
        return self


class UnassessedExistingCategory(StrictModel):
    canonical_category: str = Field(min_length=1)
    matched_asset_ids: list[str] = Field(min_length=1)
    audit_status: Literal["not_observed_uncertain"] = "not_observed_uncertain"
    next_action: Literal["review"] = "review"
    rationale: str = Field(min_length=1)


class CoverageComparison(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["video2world.scene_coverage_comparison"] = "video2world.scene_coverage_comparison"
    scene_id: str = Field(min_length=1)
    audit_sha256: Sha256
    asset_summary_sha256: Sha256
    categories: list[CoverageCategory] = Field(min_length=1)
    unassessed_existing_categories: list[UnassessedExistingCategory] = Field(default_factory=list)
    deterministic_policy: Literal["canonical_category_and_visible_instance_count_v1"] = (
        "canonical_category_and_visible_instance_count_v1"
    )
    segmentation_geometry_handoff: str = SAM3_DEPTH_HANDOFF

    @model_validator(mode="after")
    def require_exact_handoff(self) -> CoverageComparison:
        if self.segmentation_geometry_handoff != SAM3_DEPTH_HANDOFF:
            raise ValueError("segmentation geometry handoff is immutable")
        return self


class InferenceResult(StrictModel):
    raw_response: str = Field(min_length=1)
    runtime: dict[str, Any] = Field(default_factory=dict)


InferenceRunner = Callable[[Path, Path, str, int, str], InferenceResult]


def parse_frame_ids(repeated: Sequence[str] | None, comma_separated: str | None) -> list[str]:
    values = [value.strip() for value in repeated or () if value.strip()]
    if comma_separated:
        values.extend(value.strip() for value in comma_separated.split(",") if value.strip())
    if not values:
        raise QwenSceneAuditError("at least one frame id is required")
    if len(values) != len(set(values)):
        raise QwenSceneAuditError("frame ids must be unique")
    invalid = [value for value in values if not FRAME_ID_PATTERN.fullmatch(value)]
    if invalid:
        raise QwenSceneAuditError(f"invalid frame ids: {invalid}")
    return values


def resolve_frame_paths(frames_dir: str | Path, frame_ids: Sequence[str]) -> list[Path]:
    root = Path(frames_dir).expanduser().resolve()
    if not root.is_dir():
        raise QwenSceneAuditError(f"frames directory does not exist: {root}")
    resolved: list[Path] = []
    for frame_id in frame_ids:
        suffix = Path(frame_id).suffix.lower()
        candidates = (
            [root / frame_id]
            if suffix in SUPPORTED_FRAME_SUFFIXES
            else [root / f"{frame_id}{extension}" for extension in SUPPORTED_FRAME_SUFFIXES]
        )
        matches = [path.resolve() for path in candidates if path.is_file()]
        if not matches:
            raise QwenSceneAuditError(f"frame {frame_id!r} was not found under {root}")
        if len(matches) > 1:
            raise QwenSceneAuditError(
                f"frame {frame_id!r} is ambiguous; include the filename extension"
            )
        if matches[0].parent != root:
            raise QwenSceneAuditError(f"frame resolves outside frames directory: {frame_id!r}")
        resolved.append(matches[0])
    return resolved


def _load_pillow() -> tuple[Any, Any, Any]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:  # pragma: no cover - provider environment dependent
        raise QwenSceneAuditError(
            "Pillow is required only when building a contact sheet from source frames"
        ) from exc
    return Image, ImageDraw, ImageFont


def build_labeled_contact_sheet(
    frame_ids: Sequence[str],
    frame_paths: Sequence[Path],
    output_path: str | Path,
    *,
    columns: int = 4,
    tile_width: int = 480,
    tile_height: int = 300,
    label_height: int = 34,
) -> list[AuditEvidenceFrame]:
    if len(frame_ids) != len(frame_paths) or not frame_ids:
        raise QwenSceneAuditError("frame IDs and paths must be non-empty and aligned")
    if columns < 1:
        raise QwenSceneAuditError("contact-sheet columns must be positive")
    Image, ImageDraw, ImageFont = _load_pillow()
    rows = math.ceil(len(frame_paths) / columns)
    sheet = Image.new(
        "RGB",
        (columns * tile_width, rows * (tile_height + label_height)),
        (20, 22, 26),
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    records: list[AuditEvidenceFrame] = []
    for index, (frame_id, path) in enumerate(zip(frame_ids, frame_paths, strict=True)):
        with Image.open(path) as source:
            width, height = source.size
            frame = source.convert("RGB")
        frame.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
        column = index % columns
        row = index // columns
        x0 = column * tile_width
        y0 = row * (tile_height + label_height)
        sheet.paste(
            frame,
            (x0 + (tile_width - frame.width) // 2, y0 + (tile_height - frame.height) // 2),
        )
        draw.rectangle(
            (x0, y0 + tile_height, x0 + tile_width, y0 + tile_height + label_height),
            fill=(12, 14, 18),
        )
        draw.text(
            (x0 + 10, y0 + tile_height + 9),
            f"F{index + 1:02d} | frame_id={frame_id}",
            fill=(245, 190, 70),
            font=font,
        )
        frame_sha, _ = sha256_file(path)
        records.append(
            AuditEvidenceFrame(
                frame_id=frame_id,
                uri=path.as_uri(),
                sha256=frame_sha,
                width=width,
                height=height,
            )
        )
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    sheet.save(temporary, format="PNG", optimize=False)
    os.replace(temporary, destination)
    return records


def _frame_records_without_pillow(
    frame_ids: Sequence[str], frame_paths: Sequence[Path] | None
) -> list[AuditEvidenceFrame]:
    if frame_paths is not None and len(frame_ids) != len(frame_paths):
        raise QwenSceneAuditError("frame ids and paths must be aligned")
    records: list[AuditEvidenceFrame] = []
    for index, frame_id in enumerate(frame_ids):
        if frame_paths is None:
            records.append(
                AuditEvidenceFrame(
                    frame_id=frame_id,
                    uri=f"contact-sheet://frame/{frame_id}",
                )
            )
        else:
            path = frame_paths[index]
            frame_sha, _ = sha256_file(path)
            records.append(
                AuditEvidenceFrame(
                    frame_id=frame_id,
                    uri=path.as_uri(),
                    sha256=frame_sha,
                )
            )
    return records


def build_audit_prompt(
    frames: Sequence[AuditEvidenceFrame], asset_summary: ExistingAssetSummary
) -> str:
    if not frames:
        raise QwenSceneAuditError("at least one evidence frame is required")
    frame_map = "\n".join(
        f"- F{index + 1:02d}: frame_id={frame.frame_id}" for index, frame in enumerate(frames)
    )
    existing = [
        {
            "asset_id": asset.asset_id,
            "canonical_category": canonicalize_category(asset.category),
            "represented_instance_count": asset.represented_instance_count,
            "instance_scope": asset.instance_scope,
            "pipeline_state": asset.pipeline_state,
        }
        for asset in asset_summary.assets
    ]
    exact_frame_ids = json.dumps([frame.frame_id for frame in frames], ensure_ascii=False)
    return f"""Inspect the labeled contact sheet for scene {asset_summary.scene_id}.

Allowed evidence frames:
{frame_map}

Existing modeled asset summary (comparison is deterministic after this audit; do not
suppress visible categories or invent categories to agree with this list):
{json.dumps(existing, ensure_ascii=False, sort_keys=True)}

Return one category-level row for each major visibly supported semantic category.
Rules:
1. Aggregate by semantic category and report the conservative number of unique physical
   instances in the room. Never sum repeated appearances across frames: the same window in
   ten views is one window, not ten. Three distinct pillows are three instances.
2. Bed sheet, mattress, frame, headboard, and legs belong to the bed category unless they
   are clearly standalone furniture. Do not fragment one bed into components.
3. evidence_frame_ids must use the actual frame_id values, not the F01/F02 display labels.
   The exact allowed JSON values are {exact_frame_ids}. Never output an unknown,
   illustrative, or invented frame ID.
4. Describe visible color, shape, and material appearance concretely. Preserve identity:
   for example pale rectangular pillows cannot be summarized as black or round. If exact
   material is uncertain, describe its visible appearance and put the uncertainty in
   limitations; never write unknown, N/A, placeholder, or 0.
5. support_or_parent_category is a category-level relation only, such as pillow->bed,
   lamp->nightstand, or bed->floor. Use null when no relation is visibly supported.
6. confidence must be greater than zero. evidence_frame_ids MUST be a JSON array containing
   only the 1 to 6 strongest frames where that category is actually visible; seven or more
   entries are invalid. Do not assign all frames to every row. limitations MUST be a JSON
   array of evidence-specific sentences even when there is only one limitation, never a
   bare string and never a support category such as "floor" or "wall". Use a conservative
   count when instances touch or overlap.
7. Sweep the image systematically: room structure; large furniture; support furniture;
   pillows and other soft goods; plants and pots; lamps; paintings and other wall-mounted
   decorations; doors, windows, and curtains. Include a category only when visible.
8. Do not output bbox, mask, polygon, candidate/instance ID, occlusion edge, depth order,
   peel priority, completion action, or hidden/back geometry. SAM3 + calibrated depth and
   cameras own those downstream claims.
9. Return strict JSON matching the key contract below. You may wrap exactly that object in one
   ```json fence, but may not add prose before or after it.
10. Every category object MUST contain each of these ten keys exactly once:
    category, visible_instance_count, evidence_frame_ids, color_description,
    shape_description, material_description, support_or_parent_category, importance,
    confidence, limitations. Before answering, count the keys in every row. Do not repeat
    confidence or omit visible_instance_count.
11. The top-level object MUST contain exactly categories and scene_limitations. categories
    is a JSON array; scene_limitations is a non-empty JSON array of evidence-specific strings.
    visible_instance_count is an integer greater than zero. confidence is a number in (0,1].
    importance is exactly primary, secondary, or structure. support_or_parent_category is
    either a concrete category string or null. No other keys are allowed.
12. Your first non-fence character MUST be {{, never [. Use exactly this outer wrapper:
    {{"categories": [category objects], "scene_limitations": [specific limitations]}}
"""


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise QwenSceneAuditError(f"Qwen response repeats JSON key: {key!r}")
        result[key] = value
    return result


def _decode_response_json(raw_response: str) -> tuple[dict[str, Any], str]:
    stripped = raw_response.strip()
    if not stripped:
        raise QwenSceneAuditError("Qwen returned an empty response")
    fence = FENCED_JSON_PATTERN.fullmatch(stripped)
    envelope = "fenced_json" if fence else "json"
    payload = fence.group(1) if fence else stripped
    try:
        value = json.loads(payload, object_pairs_hook=_reject_duplicate_json_keys)
    except json.JSONDecodeError as exc:
        raise QwenSceneAuditError(f"Qwen response is not one JSON object: {exc}") from exc
    if not isinstance(value, dict):
        raise QwenSceneAuditError("Qwen response must contain one JSON object")
    return value, envelope


def _normalize_contact_sheet_frame_labels(
    value: dict[str, Any], allowed_frame_ids: Sequence[str]
) -> list[dict[str, str]]:
    aliases = {f"F{index + 1:02d}": frame_id for index, frame_id in enumerate(allowed_frame_ids)}
    normalizations: list[dict[str, str]] = []
    categories = value.get("categories")
    if not isinstance(categories, list):
        return normalizations
    for category in categories:
        if not isinstance(category, dict):
            continue
        evidence = category.get("evidence_frame_ids")
        if not isinstance(evidence, list):
            continue
        normalized: list[Any] = []
        for frame_id in evidence:
            if isinstance(frame_id, str) and frame_id in aliases:
                replacement = aliases[frame_id]
                normalizations.append({"source": frame_id, "frame_id": replacement})
                normalized.append(replacement)
            else:
                normalized.append(frame_id)
        category["evidence_frame_ids"] = normalized
    return normalizations


def parse_audit_response(
    raw_response: str, allowed_frame_ids: Sequence[str]
) -> tuple[VlmSceneAuditResponse, str, list[dict[str, str]]]:
    """Parse plain or singly fenced JSON, then enforce the evidence contract strictly."""

    value, envelope = _decode_response_json(raw_response)
    frame_label_normalizations = _normalize_contact_sheet_frame_labels(value, allowed_frame_ids)
    try:
        response = VlmSceneAuditResponse.model_validate(value, strict=True)
    except ValueError as exc:
        raise QwenSceneAuditError(f"scene audit schema validation failed: {exc}") from exc
    validate_audit_response(response, allowed_frame_ids)
    return response, envelope, frame_label_normalizations


def validate_audit_response(
    response: VlmSceneAuditResponse, allowed_frame_ids: Sequence[str]
) -> None:
    allowed = set(allowed_frame_ids)
    if not allowed:
        raise QwenSceneAuditError("allowed evidence frame ids cannot be empty")
    for category in response.categories:
        unknown = sorted(set(category.evidence_frame_ids) - allowed)
        if unknown:
            raise QwenSceneAuditError(
                f"category {category.category!r} references unknown frames: {unknown}"
            )
    evidence_signatures = {
        tuple(sorted(category.evidence_frame_ids)) for category in response.categories
    }
    if (
        len(allowed) > 1
        and len(response.categories) > 1
        and len(evidence_signatures) == 1
        and next(iter(evidence_signatures)) == tuple(sorted(allowed))
    ):
        raise QwenSceneAuditError(
            "all categories cite every frame; evidence selection is not discriminative"
        )


def load_asset_summary(path: str | Path) -> ExistingAssetSummary:
    source = Path(path).expanduser().resolve()
    try:
        return ExistingAssetSummary.model_validate_json(
            source.read_text(encoding="utf-8"), strict=True
        )
    except ValueError as exc:
        raise QwenSceneAuditError(f"invalid existing asset summary: {exc}") from exc


def load_scene_audit(path: str | Path) -> SceneAudit:
    source = Path(path).expanduser().resolve()
    try:
        return SceneAudit.model_validate_json(source.read_text(encoding="utf-8"), strict=True)
    except ValueError as exc:
        raise QwenSceneAuditError(f"invalid scene audit: {exc}") from exc


def materialize_scene_audit(
    response: VlmSceneAuditResponse,
    frames: Sequence[AuditEvidenceFrame],
    *,
    scene_id: str,
    run_id: str,
    created_at: datetime,
    provider: str,
    model: str,
) -> SceneAudit:
    return SceneAudit(
        scene_id=scene_id,
        run_id=run_id,
        created_at=created_at,
        provider=provider,
        model=model,
        evidence_frames=list(frames),
        categories=response.categories,
        scene_limitations=response.scene_limitations,
    )


def merge_scene_audits(
    audits: Sequence[SceneAudit],
    *,
    run_id: str,
    created_at: datetime,
) -> SceneAudit:
    """Merge independent view audits without inferring cross-view instance identity."""

    if not audits:
        raise QwenSceneAuditError("at least one source scene audit is required")
    if not run_id:
        raise QwenSceneAuditError("merged run_id must be non-empty")
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise QwenSceneAuditError("created_at must include timezone information")

    scene_ids = {audit.scene_id for audit in audits}
    providers = {audit.provider for audit in audits}
    models = {audit.model for audit in audits}
    source_run_ids = [audit.run_id for audit in audits]
    if len(scene_ids) != 1:
        raise QwenSceneAuditError(f"source audit scene mismatch: {sorted(scene_ids)}")
    if len(providers) != 1 or len(models) != 1:
        raise QwenSceneAuditError(
            "source audits must use exactly one provider and one model revision"
        )
    if len(source_run_ids) != len(set(source_run_ids)):
        raise QwenSceneAuditError("source audit run_ids must be unique")

    frames_by_id: dict[str, AuditEvidenceFrame] = {}
    for audit in audits:
        for frame in audit.evidence_frames:
            previous = frames_by_id.get(frame.frame_id)
            if previous is not None:
                raise QwenSceneAuditError(f"source audits repeat evidence frame {frame.frame_id!r}")
            frames_by_id[frame.frame_id] = frame

    grouped: dict[str, list[SceneCategoryObservation]] = defaultdict(list)
    for audit in audits:
        for observation in audit.categories:
            grouped[canonicalize_category(observation.category)].append(observation)

    merged_categories: list[SceneCategoryObservation] = []
    importance_rank = {"primary": 0, "structure": 1, "secondary": 2}
    for canonical_category in sorted(grouped):
        observations = grouped[canonical_category]
        maximum_count = max(item.visible_instance_count for item in observations)
        count_support = [
            item for item in observations if item.visible_instance_count == maximum_count
        ]
        representative = min(
            count_support,
            key=lambda item: (
                -item.confidence,
                tuple(sorted(item.evidence_frame_ids)),
                item.category.casefold(),
            ),
        )
        ranked_evidence: dict[str, float] = {}
        for observation in observations:
            for frame_id in observation.evidence_frame_ids:
                ranked_evidence[frame_id] = max(
                    ranked_evidence.get(frame_id, 0.0), observation.confidence
                )
        evidence_frame_ids = [
            frame_id
            for frame_id, _ in sorted(
                ranked_evidence.items(), key=lambda item: (-item[1], item[0])
            )[:6]
        ]
        importance = min(
            (item.importance for item in observations),
            key=lambda value: importance_rank[value],
        )
        limitation_set = {limitation for item in observations for limitation in item.limitations}
        if len(observations) > 1:
            limitation_set.add(
                "Cross-view identity was not inferred; the merged count is the maximum "
                "visible in one audited view, not a sum across views."
            )
        observed_counts = sorted({item.visible_instance_count for item in observations})
        if len(observed_counts) > 1:
            limitation_set.add(
                "Per-view visible counts differed ("
                + ", ".join(str(value) for value in observed_counts)
                + "); partial visibility may hide additional instances."
            )
        limitations = sorted(limitation_set)
        merged_categories.append(
            SceneCategoryObservation(
                category=canonical_category,
                visible_instance_count=maximum_count,
                evidence_frame_ids=evidence_frame_ids,
                color_description=representative.color_description,
                shape_description=representative.shape_description,
                material_description=representative.material_description,
                support_or_parent_category=representative.support_or_parent_category,
                importance=importance,
                confidence=representative.confidence,
                limitations=limitations,
            )
        )

    scene_limitation_set = {
        limitation for audit in audits for limitation in audit.scene_limitations
    }
    scene_limitation_set.add(
        "Deterministic merging does not establish cross-view instance identity; category "
        "counts are conservative per-view maxima."
    )
    scene_limitations = sorted(scene_limitation_set)
    return SceneAudit(
        scene_id=next(iter(scene_ids)),
        run_id=run_id,
        created_at=created_at,
        provider="deterministic_multi_view_merge",
        model=next(iter(models)),
        evidence_frames=[frames_by_id[key] for key in sorted(frames_by_id)],
        categories=merged_categories,
        scene_limitations=scene_limitations,
    )


def compare_asset_coverage(
    audit: SceneAudit, asset_summary: ExistingAssetSummary
) -> CoverageComparison:
    """Compare category/count coverage without asking the VLM to plan the pipeline."""

    if audit.scene_id != asset_summary.scene_id:
        raise QwenSceneAuditError(
            f"scene mismatch: audit={audit.scene_id!r}, assets={asset_summary.scene_id!r}"
        )
    grouped: dict[str, list[ExistingAsset]] = defaultdict(list)
    for asset in asset_summary.assets:
        grouped[canonicalize_category(asset.category)].append(asset)

    results: list[CoverageCategory] = []
    observed_categories: set[str] = set()
    for observation in sorted(
        audit.categories,
        key=lambda item: (
            0 if item.importance == "primary" else 1,
            canonicalize_category(item.category),
        ),
    ):
        category = canonicalize_category(observation.category)
        observed_categories.add(category)
        matched = sorted(grouped.get(category, []), key=lambda item: item.asset_id)
        represented = sum(item.represented_instance_count for item in matched)
        individual = sum(
            item.represented_instance_count
            for item in matched
            if item.instance_scope == "individual"
        )
        observed = observation.visible_instance_count

        if not matched:
            coverage_status = "missing"
            next_action = "segment"
            rationale = (
                f"No active asset represents the {observed} visibly supported {category} "
                "instance(s)."
            )
        elif represented < observed:
            coverage_status = "undercovered"
            next_action = "segment"
            rationale = (
                f"Existing assets represent {represented} of {observed} visible {category} "
                "instance(s); SAM3 + depth must discover the remainder."
            )
        elif individual < observed:
            coverage_status = "undercovered"
            next_action = "instance_split"
            rationale = (
                f"The category represents {represented} instance(s), but only {individual} "
                f"of {observed} are individual assets."
            )
        else:
            coverage_status = "covered"
            individual_assets = [item for item in matched if item.instance_scope == "individual"]
            states = {item.pipeline_state for item in individual_assets}
            if "candidate" in states:
                next_action = "segment"
            elif states & {"segmented", "review_failed"}:
                next_action = "complete"
            elif "completed" in states:
                next_action = "review"
            else:
                next_action = "none"
            rationale = (
                f"Individual assets cover all {observed} visibly supported {category} "
                f"instance(s); lifecycle states select next_action={next_action}."
            )

        results.append(
            CoverageCategory(
                canonical_category=category,
                observed_visible_instance_count=observed,
                existing_represented_instance_count=represented,
                existing_individual_instance_count=individual,
                matched_asset_ids=[item.asset_id for item in matched],
                coverage_status=coverage_status,
                next_action=next_action,
                evidence_frame_ids=observation.evidence_frame_ids,
                requires_sam3_depth=next_action in {"segment", "instance_split"},
                rationale=rationale,
            )
        )

    unassessed = [
        UnassessedExistingCategory(
            canonical_category=category,
            matched_asset_ids=[item.asset_id for item in sorted(items, key=lambda x: x.asset_id)],
            rationale=(
                "Existing assets use this category, but the VLM audit supplied no visible "
                "category/count evidence; coverage is not inferred from absence."
            ),
        )
        for category, items in sorted(grouped.items())
        if category not in observed_categories
    ]
    return CoverageComparison(
        scene_id=audit.scene_id,
        audit_sha256=digest_json(audit.model_dump(mode="json")),
        asset_summary_sha256=digest_json(asset_summary.model_dump(mode="json")),
        categories=results,
        unassessed_existing_categories=unassessed,
    )


def detect_local_model_revision(model_dir: str | Path, explicit: str | None = None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    root = Path(model_dir).expanduser().resolve()
    metadata = root / ".cache" / "huggingface" / "download" / "config.json.metadata"
    if metadata.is_file():
        revision = metadata.read_text(encoding="utf-8").splitlines()[0].strip()
        if revision:
            return revision
    config_path = root / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        revision = config.get("_commit_hash")
        if isinstance(revision, str) and revision.strip():
            return revision.strip()
    raise QwenSceneAuditError(
        "cannot establish the local model revision; pass --model-revision explicitly"
    )


def _patch_torch_pytree_for_old_provider() -> None:
    import torch.utils._pytree as pytree

    original = pytree.register_pytree_node
    if getattr(original, "_video2world_scene_audit_compat", False):
        return
    accepted = set(inspect.signature(original).parameters)

    def register_compat(
        node_type: Any,
        flatten_fn: Any,
        unflatten_fn: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        filtered = {key: value for key, value in kwargs.items() if key in accepted}
        return original(node_type, flatten_fn, unflatten_fn, *args, **filtered)

    register_compat._video2world_scene_audit_compat = True  # type: ignore[attr-defined]
    pytree.register_pytree_node = register_compat


def run_local_qwen_inference(
    model_dir: Path,
    contact_sheet: Path,
    prompt: str,
    max_new_tokens: int,
    device: str,
) -> InferenceResult:
    """Run deterministic local-only Qwen inference with lazy GPU dependencies."""

    try:
        import torch

        _patch_torch_pytree_for_old_provider()
        import transformers
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:  # pragma: no cover - provider environment dependent
        raise QwenSceneAuditError(
            "Qwen inference requires torch, transformers, qwen-vl-utils, and Pillow"
        ) from exc

    processor = AutoProcessor.from_pretrained(
        str(model_dir),
        min_pixels=65_536,
        max_pixels=2_007_040,
        local_files_only=True,
        use_fast=False,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_dir),
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        local_files_only=True,
        attn_implementation="sdpa",
    ).eval()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": str(contact_sheet),
                    "min_pixels": 512 * 28 * 28,
                    "max_pixels": 2560 * 28 * 28,
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[chat],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.03,
        )
    trimmed = [
        output[len(source) :] for source, output in zip(inputs.input_ids, generated, strict=True)
    ]
    raw_response = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return InferenceResult(
        raw_response=raw_response,
        runtime={
            "python_compatibility": ">=3.10",
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "processor_class": processor.__class__.__name__,
            "model_class": model.__class__.__name__,
            "dtype": "bfloat16",
            "attention": "sdpa",
            "device": device,
            "local_files_only": True,
            "generation": {
                "do_sample": False,
                "max_new_tokens": max_new_tokens,
                "repetition_penalty": 1.03,
            },
        },
    )


def _copy_contact_sheet(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise QwenSceneAuditError(f"contact sheet does not exist: {source}")
    _, size = sha256_file(source)
    if size == 0:
        raise QwenSceneAuditError(f"contact sheet is empty: {source}")
    if source == destination:
        return
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def _artifact_record(path: Path, *, contract: Any | None = None) -> dict[str, Any]:
    sha256, size = sha256_file(path)
    record: dict[str, Any] = {
        "path": str(path),
        "sha256": sha256,
        "size_bytes": size,
    }
    if contract is not None:
        record["contract_sha256"] = digest_json(contract)
    return record


def _model_file_records(model_path: Path) -> dict[str, dict[str, Any]]:
    candidates = [
        model_path / "config.json",
        model_path / "generation_config.json",
        model_path / "model.safetensors.index.json",
        model_path / "preprocessor_config.json",
        model_path / "chat_template.json",
        model_path / "tokenizer_config.json",
        *sorted(model_path.glob("model-*.safetensors")),
    ]
    records: dict[str, dict[str, Any]] = {}
    for path in candidates:
        if not path.is_file() or path.name in records:
            continue
        sha256, size = sha256_file(path)
        records[path.name] = {
            "path": str(path),
            "resolved_path": str(path.resolve()),
            "sha256": sha256,
            "size_bytes": size,
        }
    return records


def run_scene_audit_provider(
    *,
    frame_ids: Sequence[str],
    scene_id: str,
    run_id: str,
    asset_summary_path: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
    contact_sheet: str | Path | None = None,
    frames_dir: str | Path | None = None,
    model_revision: str | None = None,
    model_repo: str = MODEL_REPO,
    columns: int = 4,
    max_new_tokens: int = 3072,
    device: str = "cuda:0",
    created_at: datetime | None = None,
    inference_runner: InferenceRunner = run_local_qwen_inference,
) -> dict[str, Any]:
    if not scene_id or not run_id:
        raise QwenSceneAuditError("scene_id and run_id must be non-empty")
    _require_meaningful(model_repo, "model_repo")
    if max_new_tokens < 256:
        raise QwenSceneAuditError("max_new_tokens must be at least 256")
    ids = parse_frame_ids(frame_ids, None)
    asset_summary_source = Path(asset_summary_path).expanduser().resolve()
    asset_summary = load_asset_summary(asset_summary_source)
    if asset_summary.scene_id != scene_id:
        raise QwenSceneAuditError(
            f"asset summary scene_id {asset_summary.scene_id!r} does not match {scene_id!r}"
        )
    paths = resolve_frame_paths(frames_dir, ids) if frames_dir is not None else None
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    contact_path = destination / "contact_sheet.png"
    if contact_sheet is not None:
        _copy_contact_sheet(Path(contact_sheet).expanduser().resolve(), contact_path)
        frames = _frame_records_without_pillow(ids, paths)
        contact_source = "existing_contact_sheet"
    else:
        if paths is None:
            raise QwenSceneAuditError("provide --contact-sheet or --frames-dir")
        frames = build_labeled_contact_sheet(ids, paths, contact_path, columns=columns)
        contact_source = "built_from_frames"

    prompt = build_audit_prompt(frames, asset_summary)
    prompt_path = destination / "prompt.txt"
    atomic_write_text(prompt_path, SYSTEM_PROMPT.rstrip() + "\n\n" + prompt.rstrip() + "\n")
    revision = detect_local_model_revision(model_dir, model_revision)
    model_path = Path(model_dir).expanduser().resolve()
    inference = inference_runner(model_path, contact_path, prompt, max_new_tokens, device)
    raw_path = destination / "raw_response.txt"
    atomic_write_text(raw_path, inference.raw_response.rstrip() + "\n")
    timestamp = created_at or datetime.now(timezone.utc)  # noqa: UP017 - Python 3.10 provider.
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise QwenSceneAuditError("created_at must include timezone information")

    receipt_path = destination / "run_receipt.json"
    common_receipt: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": "video2world.qwen_scene_audit_run",
        "created_at": timestamp.isoformat(),
        "scene_id": scene_id,
        "run_id": run_id,
        "provider": "local_huggingface_transformers",
        "model": {
            "repo": model_repo,
            "revision": revision,
            "local_path": str(model_path),
            "local_files_only": True,
            "files": _model_file_records(model_path),
        },
        "inputs": {
            "contact_sheet_source": contact_source,
            "frames_dir": str(Path(frames_dir).expanduser().resolve())
            if frames_dir is not None
            else None,
            "frames": [frame.model_dump(mode="json") for frame in frames],
            "asset_summary": _artifact_record(
                asset_summary_source,
                contract=asset_summary.model_dump(mode="json"),
            ),
        },
        "boundary": {
            "vlm_outputs": [
                "major_semantic_category",
                "visible_instance_count",
                "evidence_frame_ids",
                "visible_color_shape_material",
                "support_or_parent_category",
                "importance",
                "confidence",
                "limitations",
            ],
            "vlm_does_not_output": [
                "bbox",
                "mask",
                "stable_instance_id",
                "occlusion_edge",
                "depth_order",
                "hidden_geometry",
            ],
            "downstream_handoff": SAM3_DEPTH_HANDOFF,
        },
        "artifacts": {
            "contact_sheet": _artifact_record(contact_path),
            "prompt": _artifact_record(prompt_path),
            "raw_response": _artifact_record(raw_path),
        },
        "runtime": inference.runtime,
    }
    try:
        response, response_envelope, frame_label_normalizations = parse_audit_response(
            inference.raw_response, ids
        )
        audit = materialize_scene_audit(
            response,
            frames,
            scene_id=scene_id,
            run_id=run_id,
            created_at=timestamp,
            provider="local_huggingface_transformers",
            model=f"{model_repo}@{revision}",
        )
        comparison = compare_asset_coverage(audit, asset_summary)
        audit_path = destination / "scene_audit.json"
        comparison_path = destination / "coverage_comparison.json"
        audit_payload = audit.model_dump(mode="json")
        comparison_payload = comparison.model_dump(mode="json")
        atomic_write_json(audit_path, audit_payload)
        atomic_write_json(comparison_path, comparison_payload)
        common_receipt.update(
            {
                "status": "completed",
                "response_envelope": response_envelope,
                "response_normalizations": {
                    "contact_sheet_frame_labels": frame_label_normalizations
                },
                "artifacts": {
                    **common_receipt["artifacts"],
                    "scene_audit": _artifact_record(audit_path, contract=audit_payload),
                    "coverage_comparison": _artifact_record(
                        comparison_path, contract=comparison_payload
                    ),
                },
                "summary": {
                    "category_count": len(audit.categories),
                    "missing_count": sum(
                        item.coverage_status == "missing" for item in comparison.categories
                    ),
                    "undercovered_count": sum(
                        item.coverage_status == "undercovered" for item in comparison.categories
                    ),
                    "covered_count": sum(
                        item.coverage_status == "covered" for item in comparison.categories
                    ),
                    "unassessed_existing_category_count": len(
                        comparison.unassessed_existing_categories
                    ),
                },
            }
        )
        atomic_write_json(receipt_path, common_receipt)
    except (QwenSceneAuditError, ValueError) as exc:
        common_receipt.update(
            {
                "status": "failed",
                "failure": {"error_type": type(exc).__name__, "message": str(exc)},
            }
        )
        atomic_write_json(receipt_path, common_receipt)
        raise

    return {
        "status": "completed",
        "scene_audit": str(audit_path),
        "coverage_comparison": str(comparison_path),
        "receipt": str(receipt_path),
        "category_count": len(audit.categories),
        "model_revision": revision,
    }


def run_scene_audit_merge(
    *,
    audit_paths: Sequence[str | Path],
    asset_summary_path: str | Path,
    run_id: str,
    output_dir: str | Path,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """Merge validated per-view audits and write deterministic coverage plus provenance."""

    if not audit_paths:
        raise QwenSceneAuditError("at least one --audit path is required")
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    timestamp = created_at or datetime.now(timezone.utc)  # noqa: UP017 - Python 3.10 provider.
    resolved_audit_paths = [Path(path).expanduser().resolve() for path in audit_paths]
    source_audits = [load_scene_audit(path) for path in resolved_audit_paths]
    summary_path = Path(asset_summary_path).expanduser().resolve()
    asset_summary = load_asset_summary(summary_path)
    merged = merge_scene_audits(source_audits, run_id=run_id, created_at=timestamp)
    comparison = compare_asset_coverage(merged, asset_summary)

    audit_path = destination / "scene_audit.json"
    comparison_path = destination / "coverage_comparison.json"
    receipt_path = destination / "merge_receipt.json"
    audit_payload = merged.model_dump(mode="json")
    comparison_payload = comparison.model_dump(mode="json")
    atomic_write_json(audit_path, audit_payload)
    atomic_write_json(comparison_path, comparison_payload)

    source_records: list[dict[str, Any]] = []
    for source_path, source_audit in zip(resolved_audit_paths, source_audits, strict=True):
        record = {
            "run_id": source_audit.run_id,
            "scene_audit": _artifact_record(
                source_path, contract=source_audit.model_dump(mode="json")
            ),
        }
        source_receipt = source_path.parent / "run_receipt.json"
        if source_receipt.is_file():
            record["run_receipt"] = _artifact_record(source_receipt)
        strict_parse_receipt = source_path.parent / "strict_parse_receipt.json"
        if strict_parse_receipt.is_file():
            record["strict_parse_receipt"] = _artifact_record(strict_parse_receipt)
        source_records.append(record)

    receipt = {
        "schema_version": "1.0",
        "kind": "video2world.scene_audit_merge_run",
        "status": "completed",
        "created_at": timestamp.isoformat(),
        "scene_id": merged.scene_id,
        "run_id": run_id,
        "model": merged.model,
        "source_audits": source_records,
        "asset_summary": _artifact_record(
            summary_path, contract=asset_summary.model_dump(mode="json")
        ),
        "deterministic_merge_policy": {
            "category_key": "canonical_category",
            "visible_instance_count": "maximum_per_view_never_sum",
            "evidence_selection": "confidence_desc_then_frame_id_asc_max_6",
            "representative_description": ("maximum_count_then_confidence_desc_then_frame_id_asc"),
            "cross_view_identity_inference": False,
            "invent_missing_categories": False,
        },
        "boundary": {
            "vlm_does_not_output": [
                "bbox",
                "mask",
                "stable_instance_id",
                "occlusion_edge",
                "depth_order",
                "hidden_geometry",
            ],
            "downstream_handoff": SAM3_DEPTH_HANDOFF,
        },
        "artifacts": {
            "scene_audit": _artifact_record(audit_path, contract=audit_payload),
            "coverage_comparison": _artifact_record(comparison_path, contract=comparison_payload),
        },
        "summary": {
            "source_audit_count": len(source_audits),
            "evidence_frame_count": len(merged.evidence_frames),
            "category_count": len(merged.categories),
            "missing_count": sum(
                item.coverage_status == "missing" for item in comparison.categories
            ),
            "undercovered_count": sum(
                item.coverage_status == "undercovered" for item in comparison.categories
            ),
            "covered_count": sum(
                item.coverage_status == "covered" for item in comparison.categories
            ),
            "unassessed_existing_category_count": len(comparison.unassessed_existing_categories),
        },
    }
    atomic_write_json(receipt_path, receipt)
    return {
        "status": "completed",
        "scene_audit": str(audit_path),
        "coverage_comparison": str(comparison_path),
        "receipt": str(receipt_path),
        **receipt["summary"],
    }


def write_coverage_comparison(
    audit_path: str | Path,
    asset_summary_path: str | Path,
    output_path: str | Path,
) -> CoverageComparison:
    audit = load_scene_audit(audit_path)
    summary = load_asset_summary(asset_summary_path)
    comparison = compare_asset_coverage(audit, summary)
    atomic_write_json(
        Path(output_path).expanduser().resolve(),
        comparison.model_dump(mode="json"),
    )
    return comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m video2world.providers.qwen_scene_audit",
        description="Run narrow Qwen category audit and deterministic coverage comparison.",
    )
    parser.add_argument("--contact-sheet")
    parser.add_argument("--frames-dir")
    parser.add_argument("--frame-id", action="append", default=[])
    parser.add_argument("--frame-ids")
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--asset-summary", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-revision")
    parser.add_argument("--model-repo", default=MODEL_REPO)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=3072)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_scene_audit_provider(
            frame_ids=parse_frame_ids(args.frame_id, args.frame_ids),
            scene_id=args.scene_id,
            run_id=args.run_id,
            asset_summary_path=args.asset_summary,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            contact_sheet=args.contact_sheet,
            frames_dir=args.frames_dir,
            model_revision=args.model_revision,
            model_repo=args.model_repo,
            columns=args.columns,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )
    except (OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
