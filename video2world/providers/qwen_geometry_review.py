"""Severity-aware Qwen2.5-VL review for completed object geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from video2world.completion import GeometryReview, GeometryReviewIssue
from video2world.hashing import atomic_write_json
from video2world.models import LocalizedText

ISSUE_TYPES = {
    "missing_back_surface",
    "unexpected_opening",
    "implausible_thickness",
    "implausible_proportions",
    "identity_drift",
    "front_fidelity_error",
    "unsupported_detail",
    "cross_view_style_inconsistency",
    "shape_mismatch",
    "color_mismatch",
    "material_mismatch",
    "part_merge",
    "part_missing",
    "disconnected_parts",
    "floating_parts",
    "scene_interpenetration",
    "self_intersection",
    "support_surface_error",
    "texture_hallucination",
    "other",
}
INHERENTLY_BLOCKING_ISSUES = {
    "missing_back_surface",
    "unexpected_opening",
    "implausible_thickness",
    "implausible_proportions",
    "identity_drift",
    "front_fidelity_error",
    "shape_mismatch",
    "part_merge",
    "part_missing",
    "disconnected_parts",
    "floating_parts",
    "support_surface_error",
}
ADVISORY_GATE_NAMES = {
    "raw_surface_closed",
    "raw_parts_connected",
    "source_material_color_fidelity",
    "six_view_source_color_family",
    "six_view_color_style_consistency",
}
ALLOWED_EVIDENCE_VIEWS = {
    "source",
    "front",
    "right",
    "back",
    "left",
    "top",
    "bottom",
    "technical",
}
SIX_VIEW_IDS = ("front", "right", "back", "left", "top", "bottom")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def extract_json(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def technical_gates(category: str, report: dict[str, Any]) -> dict[str, bool]:
    raw = report.get("raw_mesh")
    processed = report.get("processed_mesh")
    raw = raw if isinstance(raw, dict) else {}
    processed = processed if isinstance(processed, dict) else {}
    extents_value = processed.get("extents") or raw.get("extents") or []
    try:
        extents = [float(value) for value in extents_value]
    except (TypeError, ValueError):
        extents = []
    faces = processed.get("faces", 0)
    raw_components = raw.get("connected_components")
    gates = {
        "finite_geometry": bool(
            raw.get("finite_vertices")
            and raw.get("finite_faces")
            and processed.get("finite_vertices")
            and processed.get("finite_faces")
        ),
        "nonempty_triangle_mesh": isinstance(faces, int) and faces > 0,
        "raw_surface_closed": raw.get("watertight") is True,
        "raw_parts_connected": isinstance(raw_components, int) and raw_components <= 3,
        "positive_3d_extents": len(extents) == 3 and all(value > 0 for value in extents),
    }
    if category.casefold() in {"pillow", "cushion", "throw pillow"} and len(extents) == 3:
        ratio = min(extents) / max(extents)
        gates["pillow_thickness_ratio"] = 0.20 <= ratio <= 0.65
    return gates


def _image_color_metrics(path: Path, *, source_mask: bool) -> dict[str, Any]:
    from PIL import Image

    image = Image.open(path).convert("RGBA")
    pixels = list(image.getdata())
    stride = max(1, math.ceil(len(pixels) / 200_000))
    channels: tuple[list[int], list[int], list[int]] = ([], [], [])
    xs: list[int] = []
    ys: list[int] = []
    width = image.width
    for flat_index in range(0, len(pixels), stride):
        red, green, blue, alpha = pixels[flat_index]
        if alpha < 128:
            continue
        if not source_mask and max(red, green, blue) < 8:
            continue
        channels[0].append(red)
        channels[1].append(green)
        channels[2].append(blue)
        if source_mask:
            xs.append(flat_index % width)
            ys.append(flat_index // width)
    if not channels[0]:
        raise ValueError(f"no usable color pixels in {path}")
    mean_rgb = [sum(channel) / len(channel) for channel in channels]
    median_rgb = [float(statistics.median(channel)) for channel in channels]
    luminance = [
        0.2126 * red + 0.7152 * green + 0.0722 * blue
        for red, green, blue in zip(*channels, strict=True)
    ]
    result: dict[str, Any] = {
        "sample_count": len(channels[0]),
        "mean_rgb": mean_rgb,
        "median_rgb": median_rgb,
        "mean_luma": 0.2126 * mean_rgb[0] + 0.7152 * mean_rgb[1] + 0.0722 * mean_rgb[2],
        "median_luma": float(statistics.median(luminance)),
        "light_pixel_fraction": sum(value >= 140 for value in luminance) / len(luminance),
        "mean_chroma_range": max(mean_rgb) - min(mean_rgb),
    }
    if source_mask:
        if not xs or not ys:
            raise ValueError(f"source alpha mask is empty: {path}")
        bbox_width = max(xs) - min(xs) + 1
        bbox_height = max(ys) - min(ys) + 1
        result["alpha_bbox_xyxy"] = [min(xs), min(ys), max(xs) + 1, max(ys) + 1]
        result["alpha_bbox_aspect"] = max(bbox_width, bbox_height) / min(bbox_width, bbox_height)
    return result


def appearance_gates(
    source_image: Path,
    front_render_image: Path,
    material_image: Path,
    report: dict[str, Any],
    *,
    view_render_dir: Path,
    render_mode: str,
    expected_tone: str,
) -> tuple[dict[str, bool], dict[str, Any]]:
    source = _image_color_metrics(source_image, source_mask=True)
    front = _image_color_metrics(front_render_image, source_mask=True)
    material = _image_color_metrics(material_image, source_mask=False)
    view_paths = {view_id: view_render_dir / f"{view_id}_object.png" for view_id in SIX_VIEW_IDS}
    missing_views = [view_id for view_id, path in view_paths.items() if not path.is_file()]
    if missing_views:
        raise ValueError("six-view render directory is incomplete: " + ", ".join(missing_views))
    views = {
        view_id: _image_color_metrics(path, source_mask=True)
        for view_id, path in view_paths.items()
    }
    if sha256_file(front_render_image) != sha256_file(view_paths["front"]):
        raise ValueError("--front-render-image must match front_object.png in --view-render-dir")
    material_mean_distance = math.dist(source["mean_rgb"], material["mean_rgb"])
    material_median_distance = math.dist(source["median_rgb"], material["median_rgb"])
    material_luma_delta = abs(source["mean_luma"] - material["mean_luma"])
    front_mean_distance = math.dist(source["mean_rgb"], front["mean_rgb"])
    front_median_distance = math.dist(source["median_rgb"], front["median_rgb"])
    front_luma_delta = abs(source["mean_luma"] - front["mean_luma"])
    view_source_distances = {
        view_id: math.dist(source["mean_rgb"], metrics["mean_rgb"])
        for view_id, metrics in views.items()
    }
    view_source_luma_deltas = {
        view_id: abs(source["mean_luma"] - metrics["mean_luma"])
        for view_id, metrics in views.items()
    }
    view_pair_distances = [
        math.dist(views[left]["mean_rgb"], views[right]["mean_rgb"])
        for index, left in enumerate(SIX_VIEW_IDS)
        for right in SIX_VIEW_IDS[index + 1 :]
    ]
    processed = report.get("processed_mesh")
    processed = processed if isinstance(processed, dict) else {}
    try:
        extents = sorted((float(value) for value in processed.get("extents", [])), reverse=True)
    except (TypeError, ValueError):
        extents = []
    generated_aspect = extents[0] / extents[1] if len(extents) == 3 and extents[1] > 0 else None
    source_aspect = float(source["alpha_bbox_aspect"])
    front_aspect = float(front["alpha_bbox_aspect"])
    aspect_log_error = (
        abs(math.log(generated_aspect / source_aspect))
        if generated_aspect is not None and source_aspect > 0
        else None
    )
    metrics = {
        "source": source,
        "front_render": front,
        "material": material,
        "front_mean_rgb_distance": front_mean_distance,
        "front_median_rgb_distance": front_median_distance,
        "front_mean_luma_delta": front_luma_delta,
        "material_mean_rgb_distance": material_mean_distance,
        "material_median_rgb_distance": material_median_distance,
        "material_mean_luma_delta": material_luma_delta,
        "render_mode": render_mode,
        "expected_tone": expected_tone,
        "six_views": views,
        "view_source_mean_rgb_distances": view_source_distances,
        "view_source_mean_luma_deltas": view_source_luma_deltas,
        "maximum_cross_view_mean_rgb_distance": max(view_pair_distances),
        "generated_planar_aspect": generated_aspect,
        "source_alpha_bbox_aspect": source_aspect,
        "front_alpha_bbox_aspect": front_aspect,
        "front_aspect_log_error": abs(math.log(front_aspect / source_aspect)),
        "aspect_log_error": aspect_log_error,
    }
    gates = {
        "front_color_fidelity": (
            front_mean_distance <= 60 and front_median_distance <= 80 and front_luma_delta <= 45
        ),
        "front_silhouette_aspect_fidelity": (metrics["front_aspect_log_error"] <= 0.15),
        "source_material_color_fidelity": (
            material_mean_distance <= 85
            and material_median_distance <= 105
            and material_luma_delta <= 60
        ),
        "source_silhouette_aspect_fidelity": (
            aspect_log_error is not None and aspect_log_error <= 0.30
        ),
        "neutral_albedo_render_evidence": render_mode == "neutral-albedo",
        "six_view_source_color_family": all(
            distance <= 95 for distance in view_source_distances.values()
        )
        and all(delta <= 65 for delta in view_source_luma_deltas.values()),
        "six_view_color_style_consistency": max(view_pair_distances) <= 78,
        "expected_white_neutral_appearance": (
            expected_tone != "white"
            or all(
                metrics["mean_luma"] >= 118
                and metrics["mean_chroma_range"] <= 48
                and metrics["light_pixel_fraction"] >= 0.22
                for metrics in views.values()
            )
        ),
    }
    return gates, metrics


def build_prompt(
    *,
    object_id: str,
    category: str,
    attempt: int,
    report: dict[str, Any],
    gates: dict[str, bool],
) -> str:
    issue_type_options = "|".join(sorted(ISSUE_TYPES))
    retry_prompt_description = (
        "A detailed image-to-3D prompt that preserves observed identity while explicitly "
        "reconstructing hidden geometry. Required for retry; null otherwise."
    )
    return f"""You are the fail-closed geometry supervisor for a video-to-world pipeline.

Image 1 is the observed source crop. It may contain occlusion and therefore is evidence
for identity, texture, visible silhouette and scale. It is not evidence that an
occlusion-shaped notch belongs to the object. Image 2 is a deterministic
FRONT/RIGHT/BACK/LEFT/TOP/BOTTOM review of the generated textured mesh.

First compare the generated FRONT view against the visible, unoccluded pixels in Image 1.
The front must preserve identity, silhouette where observed, dominant color, material,
surface pattern, part layout, and aspect ratio. Evidence-compatible refinement is allowed,
such as clearer plant leaves, but unsupported changes are not. Then inspect the other five
views for complete geometry and consistent style, material, color, and connected parts.

Object id: {object_id}
Expected semantic category: {category}
Generation attempt: {attempt}
Technical gates: {json.dumps(gates, ensure_ascii=True, sort_keys=True)}
Advisory gate names: {json.dumps(sorted(ADVISORY_GATE_NAMES))}
Mesh report: {json.dumps(report, ensure_ascii=True, sort_keys=True)}

Use a usability-first severity policy. Accept an otherwise usable object with warning issues
when its overall shape is correct, its dominant color family is basically correct, and any
interpenetration or generated detail is minor and not visually disruptive. Record minor
backside texture invention, local material drift, small surface artifacts, or subtle
interpenetration as warnings; warnings do not require regeneration. Mark an issue blocking
only when it materially changes identity or use, such as:
- back or side remains sheet-like, empty, open, or copied from an occlusion boundary;
- an implausible hole/notch was created where another foreground object hid the source;
- thickness or overall proportions are implausible for the category or conflict with the source;
- parts that should be connected are detached or floating, such as a head separated from its base;
- semantic identity, visible silhouette, support surface, shape, or dominant color family
  clearly drifts from the source. For example, a white rectangular pillow cannot become
  black or round;
- the front departs from observed evidence, adds unsupported parts, or changes part layout;
- another side has a major style/material/color discontinuity or breaks geometric continuity;
- a blocking technical gate is false. Advisory gate failures are warning evidence and may be
  accepted when the six-view result remains usable.

Return exactly one JSON object, no markdown:
{{
  "decision": "accept|retry|reject",
  "issues": [
    {{
      "issue_type": "{issue_type_options}",
      "severity": "blocking|warning",
      "evidence_view_ids": ["source|front|right|back|left|top|bottom|technical"],
      "explanation": {{"en": "specific evidence", "zh": "specific evidence"}},
      "retry_prompt_instruction": "specific repair instruction, required for blocking issues"
    }}
  ],
  "retry_prompt": "{retry_prompt_description}"
}}
"""


def patch_torch_pytree() -> None:
    import torch.utils._pytree as pytree

    original = pytree.register_pytree_node

    def register_compat(
        node_type: Any,
        flatten_fn: Any,
        unflatten_fn: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        kwargs.pop("flatten_with_keys_fn", None)
        kwargs.pop("serialized_type_name", None)
        kwargs.pop("to_dumpable_context", None)
        kwargs.pop("from_dumpable_context", None)
        return original(node_type, flatten_fn, unflatten_fn, *args, **kwargs)

    pytree.register_pytree_node = register_compat


def load_qwen(model_path: Path, attention_backend: str) -> tuple[Any, Any]:
    patch_torch_pytree()
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attention_backend,
        local_files_only=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        min_pixels=65_536,
        max_pixels=786_432,
        local_files_only=True,
    )
    return model, processor


def run_qwen(
    model: Any,
    processor: Any,
    prompt: str,
    images: list[Path],
    max_new_tokens: int,
) -> str:
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    content: list[dict[str, Any]] = [
        {"type": "image", "image": Image.open(path).convert("RGB")} for path in images
    ]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[chat],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    generated = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    trimmed = [
        output[len(source) :] for source, output in zip(inputs.input_ids, generated, strict=True)
    ]
    return processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def issue_from_value(value: Any) -> GeometryReviewIssue | None:
    if not isinstance(value, dict):
        return None
    issue_type = str(value.get("issue_type") or "other")
    if issue_type not in ISSUE_TYPES:
        issue_type = "other"
    severity = (
        "blocking"
        if issue_type in INHERENTLY_BLOCKING_ISSUES or value.get("severity") == "blocking"
        else "warning"
    )
    view_ids = value.get("evidence_view_ids")
    if not isinstance(view_ids, list) or not view_ids:
        view_ids = ["technical"]
    normalized_views: list[str] = []
    for value_id in view_ids:
        normalized_views.extend(str(value_id).split("|"))
    view_ids = [value_id for value_id in normalized_views if value_id in ALLOWED_EVIDENCE_VIEWS]
    if not view_ids:
        view_ids = ["technical"]
    explanation = value.get("explanation")
    if not isinstance(explanation, dict):
        explanation = {"en": str(explanation or issue_type), "zh": str(explanation or issue_type)}
    instruction = value.get("retry_prompt_instruction")
    if severity == "blocking" and not instruction:
        instruction = f"Repair the blocking {issue_type} issue before regeneration."
    return GeometryReviewIssue(
        issue_type=issue_type,
        severity=severity,
        evidence_view_ids=[str(item) for item in view_ids],
        explanation=LocalizedText(
            en=str(explanation.get("en") or issue_type),
            zh=str(explanation.get("zh") or explanation.get("en") or issue_type),
        ),
        retry_prompt_instruction=str(instruction) if instruction else None,
    )


def build_review(
    *,
    object_id: str,
    attempt: int,
    model_name: str,
    source_asset: Path,
    turntable: Path,
    raw_response: str,
    gates: dict[str, bool],
) -> GeometryReview:
    parsed = extract_json(raw_response)
    issues: list[GeometryReviewIssue] = []
    if parsed is not None:
        for value in parsed.get("issues", []):
            issue = issue_from_value(value)
            if issue is not None:
                issues.append(issue)
    advisory_gates = {name: passed for name, passed in gates.items() if name in ADVISORY_GATE_NAMES}
    blocking_gates = {
        name: passed for name, passed in gates.items() if name not in ADVISORY_GATE_NAMES
    }
    failed_blocking_gates = sorted(name for name, passed in blocking_gates.items() if not passed)
    failed_advisory_gates = sorted(name for name, passed in advisory_gates.items() if not passed)
    if failed_blocking_gates:
        if "pillow_thickness_ratio" in failed_blocking_gates:
            technical_issue_type = "implausible_thickness"
        elif {
            "front_color_fidelity",
            "front_silhouette_aspect_fidelity",
        }.intersection(failed_blocking_gates):
            technical_issue_type = "front_fidelity_error"
        elif {
            "expected_white_neutral_appearance",
        }.intersection(failed_blocking_gates):
            technical_issue_type = "color_mismatch"
        elif "source_silhouette_aspect_fidelity" in failed_blocking_gates:
            technical_issue_type = "shape_mismatch"
        else:
            technical_issue_type = "other"
        issues.append(
            GeometryReviewIssue(
                issue_type=technical_issue_type,
                severity="blocking",
                evidence_view_ids=["technical"],
                explanation=LocalizedText(
                    en=(
                        "Blocking deterministic technical gates failed: "
                        f"{', '.join(failed_blocking_gates)}."
                    ),
                    zh=(f"阻断性确定性技术门槛未通过: {', '.join(failed_blocking_gates)}."),
                ),
                retry_prompt_instruction=(
                    "Regenerate a closed, volumetric object with plausible category "
                    "proportions and "
                    "connected parts, then rerun all technical gates."
                ),
            )
        )
    if failed_advisory_gates:
        if "raw_surface_closed" in failed_advisory_gates:
            advisory_issue_type = "unexpected_opening"
        elif "raw_parts_connected" in failed_advisory_gates:
            advisory_issue_type = "disconnected_parts"
        elif "source_material_color_fidelity" in failed_advisory_gates:
            advisory_issue_type = "material_mismatch"
        else:
            advisory_issue_type = "cross_view_style_inconsistency"
        issues.append(
            GeometryReviewIssue(
                issue_type=advisory_issue_type,
                severity="warning",
                evidence_view_ids=["technical"],
                explanation=LocalizedText(
                    en=(
                        "Non-blocking review checks failed and were retained as limitations: "
                        f"{', '.join(failed_advisory_gates)}."
                    ),
                    zh=(
                        "以下非阻断审核项未通过, 已作为限制保留: "
                        f"{', '.join(failed_advisory_gates)}."
                    ),
                ),
            )
        )
    if parsed is None:
        issues.append(
            GeometryReviewIssue(
                issue_type="other",
                severity="blocking",
                evidence_view_ids=["technical"],
                explanation=LocalizedText(
                    en="The VLM response did not contain a valid JSON review.",
                    zh="VLM 响应不包含有效的 JSON 审核结果。",
                ),
                retry_prompt_instruction="Rerun the review before publishing this asset.",
            )
        )
    requested = parsed.get("decision") if parsed is not None else "retry"
    blocking = any(issue.severity == "blocking" for issue in issues)
    decision = requested if requested in {"accept", "retry", "reject"} else "retry"
    if blocking and decision == "accept":
        decision = "retry"
    if not blocking and decision == "retry":
        decision = "accept"
    retry_prompt = parsed.get("retry_prompt") if parsed is not None else None
    if decision == "retry" and not retry_prompt:
        instructions = [
            issue.retry_prompt_instruction
            for issue in issues
            if issue.retry_prompt_instruction is not None
        ]
        retry_prompt = " ".join(instructions)
    if decision != "retry":
        retry_prompt = None
    raw_sha = hashlib.sha256(raw_response.encode("utf-8")).hexdigest()
    return GeometryReview(
        object_id=object_id,
        attempt=attempt,
        created_at=datetime.now(timezone.utc),  # noqa: UP017 - provider runs on Python 3.10.
        provider="Qwen2.5-VL",
        model=model_name,
        source_asset_sha256=sha256_file(source_asset),
        turntable_sha256=sha256_file(turntable),
        technical_gates=blocking_gates,
        advisory_gates=advisory_gates,
        decision=decision,
        issues=issues,
        retry_prompt=str(retry_prompt) if retry_prompt else None,
        raw_response_sha256=raw_sha,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--source-image", type=Path, required=True)
    parser.add_argument("--front-render-image", type=Path, required=True)
    parser.add_argument("--view-render-dir", type=Path, required=True)
    parser.add_argument(
        "--front-render-mode",
        choices=("neutral-albedo",),
        required=True,
    )
    parser.add_argument(
        "--expected-tone",
        choices=("source", "light-neutral", "white"),
        default="source",
    )
    parser.add_argument("--turntable-image", type=Path, required=True)
    parser.add_argument("--source-asset", type=Path, required=True)
    parser.add_argument("--material-image", type=Path, required=True)
    parser.add_argument("--mesh-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--gpu-devices", default="")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.gpu_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
    report = load_json(args.mesh_report.resolve())
    gates = technical_gates(args.category, report)
    appearance_gate_values, appearance_metrics = appearance_gates(
        args.source_image.resolve(),
        args.front_render_image.resolve(),
        args.material_image.resolve(),
        report,
        view_render_dir=args.view_render_dir.resolve(),
        render_mode=args.front_render_mode,
        expected_tone=args.expected_tone,
    )
    gates.update(appearance_gate_values)
    review_report = dict(report)
    review_report["appearance_comparison"] = appearance_metrics
    prompt = build_prompt(
        object_id=args.object_id,
        category=args.category,
        attempt=args.attempt,
        report=review_report,
        gates=gates,
    )
    model, processor = load_qwen(args.model_path.resolve(), args.attention_backend)
    raw = run_qwen(
        model,
        processor,
        prompt,
        [args.source_image.resolve(), args.turntable_image.resolve()],
        args.max_new_tokens,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    (output_dir / "raw_response.txt").write_text(raw + "\n", encoding="utf-8")
    review = build_review(
        object_id=args.object_id,
        attempt=args.attempt,
        model_name=args.model_path.name,
        source_asset=args.source_asset.resolve(),
        turntable=args.turntable_image.resolve(),
        raw_response=raw,
        gates=gates,
    )
    atomic_write_json(output_dir / "geometry_review.json", review.model_dump(mode="json"))
    print(
        json.dumps(
            {
                "object_id": review.object_id,
                "decision": review.decision,
                "failed_blocking_gates": sorted(
                    name for name, passed in review.technical_gates.items() if not passed
                ),
                "failed_advisory_gates": sorted(
                    name for name, passed in review.advisory_gates.items() if not passed
                ),
                "review": str(output_dir / "geometry_review.json"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if review.decision == "accept" else 3


if __name__ == "__main__":
    raise SystemExit(main())
