"""Auditable multi-frame scene inventory generation with local Qwen2.5-VL.

The module keeps image preparation, strict response parsing, domain validation, and
artifact materialization testable without importing the optional GPU runtime. Torch,
Transformers, qwen-vl-utils, and Pillow are imported only when their functionality is
actually requested.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from video2world.completion import (
    CompletionNeed,
    InventoryFrame,
    InventoryRegion,
    OcclusionEdge,
    OcclusionGraph,
    SceneAssetObservation,
    SceneInventory,
)
from video2world.hashing import (
    atomic_write_json,
    atomic_write_text,
    digest_json,
    sha256_file,
)
from video2world.models import LocalizedText, StrictModel

MODEL_REPO = "Qwen/Qwen2.5-VL-3B-Instruct"
SUPPORTED_FRAME_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
FRAME_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

SYSTEM_PROMPT = """You are the conservative scene supervisor for a 3D world-building
pipeline. Inspect only visible evidence. Never invent an object, hidden geometry, a
material, an occlusion order, or a stable instance identity. Return exactly one JSON
object and no markdown, prose, or code fences. The response must obey every enum,
field, hierarchy rule, and evidence rule in the user prompt."""

GRANULARITY_MAP = {
    "root": "independent_root_asset",
    "child": "independent_child_asset",
    "merged": "merge_into_parent",
    "structure": "structure_background",
    "reject": "reject_fragment",
}


class QwenInventoryError(ValueError):
    """Raised when a Qwen inventory run is not trustworthy enough to materialize."""


class VlmInventoryRegion(StrictModel):
    frame_id: str = Field(min_length=1)
    bbox_0_1000: tuple[int, int, int, int]
    visibility: Literal["full", "partial", "heavily_occluded"]
    visible_faces: list[Literal["front", "back", "left", "right", "top", "bottom", "unknown"]] = (
        Field(min_length=1)
    )
    occluded_by_candidate_ids: list[str]
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_box(self) -> VlmInventoryRegion:
        x0, y0, x1, y1 = self.bbox_0_1000
        if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
            raise ValueError("bbox_0_1000 must be ordered inside [0, 1000]")
        return self


class VlmSceneObservation(StrictModel):
    candidate_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    category: str = Field(min_length=1)
    name_en: str = Field(min_length=1)
    name_zh: str = Field(min_length=1)
    aliases: list[str]
    asset_role: Literal[
        "interactive_object", "attached_object", "fixture", "decoration", "structure"
    ]
    semantic_granularity: Literal["root", "child", "merged", "structure", "reject"]
    parent_candidate_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.-]+$")
    moves_with_parent: bool
    independently_movable: bool
    semantic_unit_rationale: str = Field(min_length=1)
    instance_count: int = Field(ge=1)
    importance: Literal["primary", "secondary", "structure"]
    structure_role: Literal["none", "wall", "floor", "ceiling", "background"]
    confidence: float = Field(ge=0, le=1)
    peel_priority: int = Field(ge=0, le=100)
    evidence_regions: list[VlmInventoryRegion] = Field(min_length=1)
    completion_needs: list[CompletionNeed]
    completion_risks: list[str]
    notes: list[str]

    @model_validator(mode="after")
    def validate_semantic_granularity(self) -> VlmSceneObservation:
        needs_parent = self.semantic_granularity in {"child", "merged"}
        if needs_parent != (self.parent_candidate_id is not None):
            raise ValueError(
                "child/merged observations require a parent; other granularities forbid one"
            )
        if self.moves_with_parent != needs_parent:
            raise ValueError("moves_with_parent must be true exactly for child/merged")
        expected_movable = self.semantic_granularity in {"root", "child"}
        if self.independently_movable != expected_movable:
            raise ValueError("only root/child observations may be independently movable")
        is_structure = self.semantic_granularity == "structure"
        if is_structure != (self.importance == "structure"):
            raise ValueError("structure granularity and structure importance must agree")
        if is_structure != (self.asset_role == "structure"):
            raise ValueError("structure granularity and structure asset_role must agree")
        if is_structure == (self.structure_role == "none"):
            raise ValueError("only structure observations may declare structure_role")
        if len(self.completion_needs) != len(set(self.completion_needs)):
            raise ValueError("completion_needs cannot contain duplicates")
        return self


class VlmOcclusionEdge(StrictModel):
    foreground_candidate_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    background_candidate_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    confidence: float = Field(ge=0, le=1)
    evidence_frame_ids: list[str] = Field(min_length=1)
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def reject_self_edge(self) -> VlmOcclusionEdge:
        if self.foreground_candidate_id == self.background_candidate_id:
            raise ValueError("an occlusion edge cannot point to itself")
        return self


class VlmInventoryResponse(StrictModel):
    observations: list[VlmSceneObservation] = Field(min_length=1)
    occlusion_edges: list[VlmOcclusionEdge]
    unresolved_relations: list[str]
    scene_limitations: list[str]


class InferenceResult(StrictModel):
    raw_response: str = Field(min_length=1)
    runtime: dict[str, Any] = Field(default_factory=dict)


InferenceRunner = Callable[[Path, Path, str, int, str], InferenceResult]


def parse_frame_ids(repeated: Sequence[str] | None, comma_separated: str | None) -> list[str]:
    """Combine repeatable and comma-separated CLI frame identifiers."""

    values = [value.strip() for value in repeated or () if value.strip()]
    if comma_separated:
        values.extend(value.strip() for value in comma_separated.split(",") if value.strip())
    if not values:
        raise QwenInventoryError("at least one --frame-id or --frame-ids value is required")
    if len(values) != len(set(values)):
        raise QwenInventoryError("frame ids must be unique")
    invalid = [value for value in values if not FRAME_ID_PATTERN.fullmatch(value)]
    if invalid:
        raise QwenInventoryError(f"invalid frame ids: {invalid}")
    return values


def resolve_frame_paths(frames_dir: str | Path, frame_ids: Sequence[str]) -> list[Path]:
    """Resolve frame IDs without allowing path traversal or ambiguous extensions."""

    root = Path(frames_dir).expanduser().resolve()
    if not root.is_dir():
        raise QwenInventoryError(f"frames directory does not exist: {root}")
    resolved: list[Path] = []
    for frame_id in frame_ids:
        if not FRAME_ID_PATTERN.fullmatch(frame_id):
            raise QwenInventoryError(f"invalid frame id: {frame_id!r}")
        suffix = Path(frame_id).suffix.lower()
        if suffix in SUPPORTED_FRAME_SUFFIXES:
            candidates = [root / frame_id]
        else:
            candidates = [root / f"{frame_id}{extension}" for extension in SUPPORTED_FRAME_SUFFIXES]
        matches = [path.resolve() for path in candidates if path.is_file()]
        if not matches:
            raise QwenInventoryError(f"frame {frame_id!r} was not found under {root}")
        if len(matches) > 1:
            raise QwenInventoryError(
                f"frame {frame_id!r} is ambiguous; use the filename with its extension"
            )
        if matches[0].parent != root:
            raise QwenInventoryError(f"frame resolves outside frames directory: {frame_id!r}")
        resolved.append(matches[0])
    return resolved


def _load_pillow() -> tuple[Any, Any, Any]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:  # pragma: no cover - depends on provider environment
        raise QwenInventoryError(
            "Pillow is required to build the labeled contact sheet; install Pillow in "
            "the Qwen provider environment"
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
) -> list[InventoryFrame]:
    """Render one deterministically ordered, frame-labeled RGB contact sheet."""

    if len(frame_ids) != len(frame_paths) or not frame_ids:
        raise QwenInventoryError("frame IDs and paths must be non-empty and have equal length")
    if columns < 1 or tile_width < 64 or tile_height < 64 or label_height < 16:
        raise QwenInventoryError("invalid contact-sheet layout")
    Image, ImageDraw, ImageFont = _load_pillow()
    rows = math.ceil(len(frame_paths) / columns)
    sheet = Image.new(
        "RGB",
        (columns * tile_width, rows * (tile_height + label_height)),
        (20, 22, 26),
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    evidence_frames: list[InventoryFrame] = []
    for index, (frame_id, path) in enumerate(zip(frame_ids, frame_paths, strict=True)):
        with Image.open(path) as source:
            width, height = source.size
            frame = source.convert("RGB")
        frame.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
        column = index % columns
        row = index // columns
        x0 = column * tile_width
        y0 = row * (tile_height + label_height)
        image_x = x0 + (tile_width - frame.width) // 2
        image_y = y0 + (tile_height - frame.height) // 2
        sheet.paste(frame, (image_x, image_y))
        label = f"F{index + 1:02d} | frame_id={frame_id}"
        draw.rectangle(
            (x0, y0 + tile_height, x0 + tile_width, y0 + tile_height + label_height),
            fill=(12, 14, 18),
        )
        draw.text((x0 + 10, y0 + tile_height + 9), label, fill=(245, 190, 70), font=font)
        frame_sha, _ = sha256_file(path)
        evidence_frames.append(
            InventoryFrame(
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
    return evidence_frames


def build_inventory_prompt(evidence_frames: Sequence[InventoryFrame]) -> str:
    """Build the strict scene-inventory prompt used as a provenance artifact."""

    if not evidence_frames:
        raise QwenInventoryError("at least one evidence frame is required")
    frame_map = "\n".join(
        f"- F{index + 1:02d}: frame_id={frame.frame_id}, original={frame.width}x{frame.height}"
        for index, frame in enumerate(evidence_frames)
    )
    response_schema = json.dumps(
        VlmInventoryResponse.model_json_schema(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"""The supplied image is a labeled contact sheet of one scene.

Frame map:
{frame_map}

Inventory goal:
1. Discover every visually supported major semantic unit across the views, including
   furniture, movable objects, fixtures, decorations, and the wall/floor/ceiling/background.
2. IDs are candidate observation IDs, not stable instance IDs. Keep visibly distinct
   instances separate. Do not create candidates from highlights, shadows, texture patches,
   holes, splat artifacts, or uncertain fragments.
3. Use semantic_granularity exactly as follows:
   - root: an independently movable top-level asset.
   - child: an independently movable asset attached to or supported by a parent.
   - merged: a component that belongs to its parent asset and must not move independently.
   - structure: wall, floor, ceiling, or remaining architectural background.
   - reject: a non-semantic fragment or unsupported candidate.
4. Pillow rule: emit one child candidate for each visually distinct pillow, set its parent
   to the bed candidate, instance_count=1, moves_with_parent=true, and
   independently_movable=true. Never merge multiple pillows into one ensemble.
5. Bed integrity rule: prefer one bed root. A bed sheet, blanket, mattress, headboard,
   bed board/frame, and bed legs are bed components, not arbitrary independent objects.
   Omit them as separate observations unless repair needs their evidence; if listed, mark
   them merged with the bed, moves_with_parent=true, independently_movable=false.
6. Every observation needs at least one evidence region. bbox_0_1000 is normalized inside
   the original individual frame, not the contact sheet. It is only a visible-region prompt
   proposal for downstream SAM3: it is not a mask, stable instance identity, 3D bound, or
   geometry measurement. Report only visible faces and explicitly list candidate IDs that
   appear to cover the region.
7. A VLM occlusion edge foreground->background is only a visible-overlap hypothesis in the
   cited original frame. It is not depth order and cannot schedule completion until SAM3,
   calibrated cameras, and depth verify it. Emit it only when visually supported; otherwise
   record the ambiguity in unresolved_relations. Do not infer depth order from category.
8. Describe completion risks explicitly: unseen back, heavy occlusion, thin structures,
   reflective/transparent surfaces, touching instances, ambiguous boundaries, and likely
   clean-plate holes. Do not claim hidden faces are complete.
9. completion_needs values may only be: instance_split, segmentation, multi_view_lift,
   backside_geometry, render_asset, collider, clean_plate, background_rebuild, visual_qa.
   Independent root/child assets normally need segmentation, multi_view_lift,
   backside_geometry, render_asset, collider, and visual_qa. Structures normally need
   clean_plate and background_rebuild.
10. Every value must describe the actual supplied frames. Do not emit placeholder values,
    do not copy schema descriptions into data, and do not return an illustrative example.
    Use observations, occlusion_edges, unresolved_relations, and scene_limitations as the
    four top-level keys. Populate every required field from the schema.

Allowed asset_role: interactive_object, attached_object, fixture, decoration, structure.
Allowed importance: primary, secondary, structure. Allowed structure_role: none, wall,
floor, ceiling, background. Allowed visibility: full, partial, heavily_occluded. Allowed
visible_faces: front, back, left, right, top, bottom, unknown. Confidence is in [0,1].
peel_priority is a visual hypothesis in [0,100] retained only in the raw response; the
materialized inventory neutralizes it until calibrated geometry establishes ordering.

The following JSON Schema is authoritative. Every field listed in required must be present,
additionalProperties is false, and the response must validate without coercing prose:
{response_schema}
"""


def parse_inventory_response(raw_response: str) -> VlmInventoryResponse:
    """Parse exactly one JSON object and reject fences, commentary, or schema drift."""

    stripped = raw_response.strip()
    if not stripped:
        raise QwenInventoryError("Qwen returned an empty response")
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise QwenInventoryError(f"Qwen response is not strict JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise QwenInventoryError("Qwen response must be one JSON object")
    try:
        return VlmInventoryResponse.model_validate(value)
    except ValueError as exc:
        raise QwenInventoryError(f"Qwen inventory schema validation failed: {exc}") from exc


def _semantic_text(item: VlmSceneObservation) -> str:
    return " ".join((item.category, item.name_en, item.name_zh, *item.aliases)).lower()


def _is_pillow(item: VlmSceneObservation) -> bool:
    value = _semantic_text(item)
    return "pillow" in value or "枕" in value


def _is_bed(item: VlmSceneObservation) -> bool:
    if item.category.strip().lower() == "bed":
        return True
    if item.name_en.strip().lower() == "bed" or item.name_zh.strip() == "床":
        return True
    value = _semantic_text(item)
    component_terms = (
        "headboard",
        "bed sheet",
        "bedsheet",
        "blanket",
        "mattress",
        "bed leg",
        "bed frame",
        "床头",
        "床单",
        "床垫",
        "床腿",
        "床架",
    )
    return ("bed" in value or "床" in value) and not any(term in value for term in component_terms)


def _is_bed_component(item: VlmSceneObservation) -> bool:
    if _is_bed(item):
        return False
    value = _semantic_text(item)
    terms = (
        "headboard",
        "bed sheet",
        "bedsheet",
        "blanket",
        "mattress",
        "bed board",
        "bed frame",
        "bed leg",
        "床头板",
        "床单",
        "毯",
        "床垫",
        "床板",
        "床架",
        "床腿",
    )
    return any(term in value for term in terms)


def validate_scene_semantics(
    response: VlmInventoryResponse, evidence_frames: Sequence[InventoryFrame]
) -> None:
    """Apply fail-closed hierarchy/evidence rules that the VLM cannot override."""

    frame_ids = {frame.frame_id for frame in evidence_frames}
    observations = {item.candidate_id: item for item in response.observations}
    if len(observations) != len(response.observations):
        raise QwenInventoryError("Qwen returned duplicate candidate IDs")
    bed_ids = {item.candidate_id for item in response.observations if _is_bed(item)}
    if not any(item.confidence > 0 for item in response.observations):
        raise QwenInventoryError("all observation confidences are zero; likely placeholder output")
    placeholder_phrases = {
        "why this is one semantic asset",
        "visually supported risk",
        "visible overlap evidence",
    }
    for item in response.observations:
        text_values = {
            item.semantic_unit_rationale.strip().lower(),
            *(value.strip().lower() for value in item.completion_risks),
            *(value.strip().lower() for value in item.notes),
        }
        copied = sorted(text_values.intersection(placeholder_phrases))
        if copied:
            raise QwenInventoryError(
                f"observation {item.candidate_id} contains placeholder text: {copied}"
            )
        region_frame_ids = {region.frame_id for region in item.evidence_regions}
        unknown_frames = sorted(region_frame_ids - frame_ids)
        if unknown_frames:
            raise QwenInventoryError(
                f"observation {item.candidate_id} references unknown frames: {unknown_frames}"
            )
        for region in item.evidence_regions:
            unknown_occluders = sorted(set(region.occluded_by_candidate_ids) - observations.keys())
            if unknown_occluders:
                raise QwenInventoryError(
                    f"observation {item.candidate_id} references unknown occluders: "
                    f"{unknown_occluders}"
                )
        if item.parent_candidate_id and item.parent_candidate_id not in observations:
            raise QwenInventoryError(
                f"observation {item.candidate_id} references unknown parent "
                f"{item.parent_candidate_id}"
            )
        if _is_pillow(item):
            if item.semantic_granularity != "child" or item.parent_candidate_id not in bed_ids:
                raise QwenInventoryError(
                    f"pillow {item.candidate_id} must be a child of a bed candidate"
                )
            if item.instance_count != 1:
                raise QwenInventoryError(
                    f"pillow {item.candidate_id} must represent exactly one pillow instance"
                )
        if _is_bed_component(item) and (
            item.semantic_granularity != "merged" or item.parent_candidate_id not in bed_ids
        ):
            raise QwenInventoryError(
                f"bed component {item.candidate_id} must be merged into a bed candidate"
            )
    for edge in response.occlusion_edges:
        if edge.rationale.strip().lower() in placeholder_phrases:
            raise QwenInventoryError("occlusion edge contains placeholder rationale")
        edge_ids = {edge.foreground_candidate_id, edge.background_candidate_id}
        unknown = sorted(edge_ids - observations.keys())
        if unknown:
            raise QwenInventoryError(f"occlusion edge references unknown candidates: {unknown}")
        unknown_frames = sorted(set(edge.evidence_frame_ids) - frame_ids)
        if unknown_frames:
            raise QwenInventoryError(f"occlusion edge references unknown frames: {unknown_frames}")


def _completion_needs(item: VlmSceneObservation) -> list[CompletionNeed]:
    needs = list(item.completion_needs)
    required: tuple[CompletionNeed, ...]
    if item.semantic_granularity in {"root", "child"}:
        required = (
            "segmentation",
            "multi_view_lift",
            "backside_geometry",
            "render_asset",
            "collider",
            "visual_qa",
        )
    elif item.semantic_granularity == "structure":
        required = ("clean_plate", "background_rebuild")
    else:
        required = ()
    for need in required:
        if need not in needs:
            needs.append(need)
    return needs


def materialize_inventory(
    response: VlmInventoryResponse,
    evidence_frames: Sequence[InventoryFrame],
    *,
    scene_id: str,
    run_id: str,
    created_at: datetime,
    provider: str,
    model: str,
    minimum_occlusion_confidence: float,
) -> tuple[SceneInventory, OcclusionGraph]:
    """Materialize typed category observations without promoting VLM geometry claims."""

    validate_scene_semantics(response, evidence_frames)
    observations: list[SceneAssetObservation] = []
    for item in response.observations:
        evidence_frame_ids = list(
            dict.fromkeys(region.frame_id for region in item.evidence_regions)
        )
        if item.semantic_granularity == "structure":
            coverage_status = "structure_background"
        elif item.semantic_granularity in {"merged", "reject"}:
            coverage_status = "ignored"
        else:
            # A VLM discovers candidates; it cannot prove that current 3D assets cover them.
            coverage_status = "missing"
        observations.append(
            SceneAssetObservation(
                id=item.candidate_id,
                category=item.category,
                name=LocalizedText(en=item.name_en, zh=item.name_zh),
                aliases=item.aliases,
                asset_role=item.asset_role,
                granularity=GRANULARITY_MAP[item.semantic_granularity],
                parent_candidate_id=item.parent_candidate_id,
                moves_with_parent=item.moves_with_parent,
                independently_movable=item.independently_movable,
                semantic_unit_rationale=item.semantic_unit_rationale,
                instance_count=item.instance_count,
                importance=item.importance,
                structure_role=item.structure_role,
                confidence=item.confidence,
                # A VLM scheduling score is not calibrated scene depth. Geometry providers
                # may assign a priority after verifying masks, cameras, and depth.
                peel_priority=50,
                evidence_frame_ids=evidence_frame_ids,
                evidence_regions=[
                    InventoryRegion.model_validate(region.model_dump(mode="json"))
                    for region in item.evidence_regions
                ],
                coverage_status=coverage_status,
                completion_needs=_completion_needs(item),
                completion_risks=item.completion_risks,
                notes=item.notes,
            )
        )
    inventory = SceneInventory(
        scene_id=scene_id,
        run_id=run_id,
        created_at=created_at,
        provider=provider,
        model=model,
        evidence_frames=list(evidence_frames),
        observations=observations,
        limitations=response.scene_limitations,
    )
    graph = OcclusionGraph(
        scene_id=scene_id,
        inventory_sha256=digest_json(inventory.model_dump(mode="json")),
        minimum_confidence=minimum_occlusion_confidence,
        edges=[
            OcclusionEdge(
                foreground_id=edge.foreground_candidate_id,
                background_id=edge.background_candidate_id,
                confidence=edge.confidence,
                evidence_frame_ids=edge.evidence_frame_ids,
                ordering_source="vlm",
                verification_status="observation_only",
            )
            for edge in response.occlusion_edges
        ],
        unresolved_relations=response.unresolved_relations,
    )
    return inventory, graph


def detect_local_model_revision(model_dir: str | Path, explicit: str | None = None) -> str:
    """Require an exact revision instead of silently labeling an unknown checkpoint."""

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
    raise QwenInventoryError(
        "cannot establish the local model revision; pass --model-revision explicitly"
    )


def run_local_qwen_inference(
    model_dir: Path,
    contact_sheet: Path,
    prompt: str,
    max_new_tokens: int,
    device: str,
) -> InferenceResult:
    """Run deterministic local-only Qwen2.5-VL inference."""

    try:
        import torch
        import transformers
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:  # pragma: no cover - depends on provider environment
        raise QwenInventoryError(
            "local Qwen inference requires torch, transformers, qwen-vl-utils, and Pillow"
        ) from exc

    processor = AutoProcessor.from_pretrained(
        model_dir,
        local_files_only=True,
        use_fast=False,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
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
                    "max_pixels": 1600 * 28 * 28,
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
    raw = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    runtime = {
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "processor_class": processor.__class__.__name__,
        "model_class": model.__class__.__name__,
        "dtype": "bfloat16",
        "attention": "sdpa",
        "device": device,
        "device_name": torch.cuda.get_device_name(device) if device.startswith("cuda") else device,
        "local_files_only": True,
        "generation": {
            "do_sample": False,
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": 1.03,
        },
    }
    return InferenceResult(raw_response=raw, runtime=runtime)


def run_inventory_provider(
    *,
    frames_dir: str | Path,
    frame_ids: Sequence[str],
    scene_id: str,
    run_id: str,
    model_dir: str | Path,
    output_dir: str | Path,
    model_revision: str | None = None,
    columns: int = 4,
    max_new_tokens: int = 8192,
    device: str = "cuda:0",
    minimum_occlusion_confidence: float = 0.6,
    created_at: datetime | None = None,
    inference_runner: InferenceRunner = run_local_qwen_inference,
) -> dict[str, Any]:
    """Run the complete inventory provider and write all audit artifacts."""

    if not scene_id or not run_id:
        raise QwenInventoryError("scene_id and run_id must be non-empty")
    if not 0 <= minimum_occlusion_confidence <= 1:
        raise QwenInventoryError("minimum occlusion confidence must be inside [0, 1]")
    if max_new_tokens < 256:
        raise QwenInventoryError("max_new_tokens must be at least 256")
    ids = parse_frame_ids(frame_ids, None)
    paths = resolve_frame_paths(frames_dir, ids)
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    contact_sheet_path = destination / "contact_sheet.png"
    evidence_frames = build_labeled_contact_sheet(
        ids,
        paths,
        contact_sheet_path,
        columns=columns,
    )
    prompt = build_inventory_prompt(evidence_frames)
    prompt_path = destination / "prompt.txt"
    atomic_write_text(prompt_path, SYSTEM_PROMPT + "\n\n" + prompt.rstrip() + "\n")
    revision = detect_local_model_revision(model_dir, model_revision)
    model_path = Path(model_dir).expanduser().resolve()
    inference = inference_runner(model_path, contact_sheet_path, prompt, max_new_tokens, device)
    raw_path = destination / "raw_response.txt"
    atomic_write_text(raw_path, inference.raw_response.rstrip() + "\n")
    response = parse_inventory_response(inference.raw_response)
    timestamp = created_at or datetime.now(timezone.utc)  # noqa: UP017 - Python 3.10 provider.
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise QwenInventoryError("created_at must include timezone information")
    provider = "local_huggingface_transformers"
    model = f"{MODEL_REPO}@{revision}"
    inventory, graph = materialize_inventory(
        response,
        evidence_frames,
        scene_id=scene_id,
        run_id=run_id,
        created_at=timestamp,
        provider=provider,
        model=model,
        minimum_occlusion_confidence=minimum_occlusion_confidence,
    )
    inventory_path = destination / "inventory.json"
    graph_path = destination / "occlusion_graph.json"
    atomic_write_json(inventory_path, inventory.model_dump(mode="json"))
    atomic_write_json(graph_path, graph.model_dump(mode="json"))
    prompt_sha, prompt_size = sha256_file(prompt_path)
    raw_sha, raw_size = sha256_file(raw_path)
    contact_sha, contact_size = sha256_file(contact_sheet_path)
    inventory_sha, inventory_size = sha256_file(inventory_path)
    graph_sha, graph_size = sha256_file(graph_path)
    receipt = {
        "schema_version": "1.0",
        "kind": "video2world.qwen_inventory_run",
        "status": "completed",
        "created_at": timestamp.isoformat(),
        "scene_id": scene_id,
        "run_id": run_id,
        "provider": provider,
        "model": {
            "repo": MODEL_REPO,
            "revision": revision,
            "local_path": str(model_path),
            "local_files_only": True,
        },
        "boundary": {
            "vlm_bbox_scope": "visible_region_prompt_only_not_mask_or_geometry",
            "vlm_occlusion_scope": "observation_only_not_depth_order",
            "geometry_ordering_required_from": [
                "sam3_masks",
                "calibrated_cameras",
                "depth",
            ],
        },
        "inputs": {
            "frames_dir": str(Path(frames_dir).expanduser().resolve()),
            "frames": [frame.model_dump(mode="json") for frame in evidence_frames],
        },
        "artifacts": {
            "contact_sheet": {
                "path": str(contact_sheet_path),
                "sha256": contact_sha,
                "size_bytes": contact_size,
            },
            "prompt": {
                "path": str(prompt_path),
                "sha256": prompt_sha,
                "size_bytes": prompt_size,
            },
            "raw_response": {
                "path": str(raw_path),
                "sha256": raw_sha,
                "size_bytes": raw_size,
            },
            "inventory": {
                "path": str(inventory_path),
                "sha256": inventory_sha,
                "size_bytes": inventory_size,
                "contract_sha256": digest_json(inventory.model_dump(mode="json")),
            },
            "occlusion_graph": {
                "path": str(graph_path),
                "sha256": graph_sha,
                "size_bytes": graph_size,
                "contract_sha256": digest_json(graph.model_dump(mode="json")),
            },
        },
        "runtime": inference.runtime,
    }
    receipt_path = destination / "run_receipt.json"
    atomic_write_json(receipt_path, receipt)
    return {
        "status": "completed",
        "inventory": str(inventory_path),
        "occlusion_graph": str(graph_path),
        "receipt": str(receipt_path),
        "observation_count": len(inventory.observations),
        "occlusion_edge_count": len(graph.edges),
        "geometry_verified_occlusion_edge_count": sum(
            edge.verification_status == "geometry_verified" for edge in graph.edges
        ),
        "model_revision": revision,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m video2world.providers.qwen_inventory",
        description="Generate an auditable SceneInventory with local Qwen2.5-VL.",
    )
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--frame-id", action="append", default=[])
    parser.add_argument("--frame-ids", help="Comma-separated alternative to repeated --frame-id")
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-revision")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--minimum-occlusion-confidence", type=float, default=0.6)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        frame_ids = parse_frame_ids(args.frame_id, args.frame_ids)
        result = run_inventory_provider(
            frames_dir=args.frames_dir,
            frame_ids=frame_ids,
            scene_id=args.scene_id,
            run_id=args.run_id,
            model_dir=args.model_dir,
            model_revision=args.model_revision,
            output_dir=args.output_dir,
            columns=args.columns,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            minimum_occlusion_confidence=args.minimum_occlusion_confidence,
        )
    except (OSError, QwenInventoryError, ValueError) as exc:
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
