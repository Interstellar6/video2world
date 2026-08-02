#!/usr/bin/env python3
"""Estimate auditable, explicitly unvalidated object physics with local Qwen-VL."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from video2world.hashing import atomic_write_json, sha256_file
from video2world.modeling import ObjectPhysicsRecord, PhysicsManifest
from video2world.models import PhysicalProperties


class PhysicsEstimationError(RuntimeError):
    """Raised when an input or model response cannot satisfy the physics contract."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=500)
    return parser.parse_args()


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    candidate = fenced.group(1) if fenced else stripped
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise PhysicsEstimationError("Qwen-VL response contains no JSON object") from None
        try:
            value = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            raise PhysicsEstimationError(f"Qwen-VL response JSON is invalid: {exc}") from exc
    if not isinstance(value, dict):
        raise PhysicsEstimationError("Qwen-VL response root must be a JSON object")
    return value


def _finite_number(value: Any, *, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise PhysicsEstimationError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PhysicsEstimationError(f"{label} must be numeric") from exc
    if not minimum <= number <= maximum:
        raise PhysicsEstimationError(
            f"{label} must be in [{minimum}, {maximum}], received {number}"
        )
    return number


def normalize_qwen_estimate(value: dict[str, Any]) -> PhysicalProperties:
    raw_dimensions = value.get("dimensions_m_width_depth_height")
    if not isinstance(raw_dimensions, list) or len(raw_dimensions) != 3:
        raise PhysicsEstimationError(
            "dimensions_m_width_depth_height must contain width, depth, and height"
        )
    dimensions = tuple(
        _finite_number(item, label=f"dimensions_m[{index}]", minimum=0.01, maximum=10.0)
        for index, item in enumerate(raw_dimensions)
    )
    mass = _finite_number(value.get("mass_kg"), label="mass_kg", minimum=0.01, maximum=5000)
    density_value = value.get("density_kg_m3")
    density = (
        None
        if density_value is None
        else _finite_number(
            density_value,
            label="density_kg_m3",
            minimum=1,
            maximum=30_000,
        )
    )
    static_friction = _finite_number(
        value.get("static_friction"), label="static_friction", minimum=0, maximum=2
    )
    dynamic_friction = _finite_number(
        value.get("dynamic_friction"), label="dynamic_friction", minimum=0, maximum=2
    )
    if dynamic_friction > static_friction:
        raise PhysicsEstimationError("dynamic_friction cannot exceed static_friction")
    restitution = _finite_number(
        value.get("restitution"), label="restitution", minimum=0, maximum=1
    )
    confidence = _finite_number(
        value.get("confidence"), label="confidence", minimum=0, maximum=1
    )
    limitations = value.get("limitations")
    if not isinstance(limitations, list) or not all(
        isinstance(item, str) and item.strip() for item in limitations
    ):
        raise PhysicsEstimationError("limitations must be a nonempty-string list")
    return PhysicalProperties(
        status="unvalidated_estimate",
        source="qwen_vl_prior",
        scale_basis="category_prior",
        dimensions_m=dimensions,
        mass_kg=mass,
        density_kg_m3=density,
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        restitution=restitution,
        confidence=min(confidence, 0.6),
        limitations=[
            *limitations,
            "The scene has no verified metric calibration for this estimate.",
            "Mass, friction, and restitution are category priors, not measurements.",
        ],
    )


def build_prompt(job: dict[str, Any]) -> str:
    return f"""Estimate a conservative physics prior for one visible indoor object.

Object id: {job['object_id']}
Category: {job['category']}
Description: {job['description']}

Use the image only to infer the broad object type, proportions, and likely materials.
There is no verified metric scene calibration. Values must therefore be plausible category
priors, not claimed measurements. Dimensions use [width, depth, height] in meters. Friction
coefficients are unitless and dynamic friction must not exceed static friction.

Return strict JSON only:
{{
  "dimensions_m_width_depth_height": [0.0, 0.0, 0.0],
  "mass_kg": 0.0,
  "density_kg_m3": 0.0,
  "static_friction": 0.0,
  "dynamic_friction": 0.0,
  "restitution": 0.0,
  "confidence": 0.0,
  "limitations": ["specific uncertainty"]
}}"""


def load_jobs(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PhysicsEstimationError(f"cannot read jobs file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PhysicsEstimationError("jobs root must be a JSON object")
    for key in ("scene_id", "run_id"):
        if not isinstance(value.get(key), str) or not value[key]:
            raise PhysicsEstimationError(f"jobs.{key} must be a nonempty string")
    jobs = value.get("objects")
    if not isinstance(jobs, list) or not jobs:
        raise PhysicsEstimationError("jobs.objects must be a nonempty list")
    ids: set[str] = set()
    for index, job in enumerate(jobs):
        if not isinstance(job, dict):
            raise PhysicsEstimationError(f"jobs.objects[{index}] must be an object")
        for key in ("object_id", "category", "description", "image"):
            if not isinstance(job.get(key), str) or not job[key]:
                raise PhysicsEstimationError(f"jobs.objects[{index}].{key} is required")
        if job["object_id"] in ids:
            raise PhysicsEstimationError(f"duplicate object_id: {job['object_id']}")
        ids.add(job["object_id"])
        image = Path(job["image"]).expanduser().resolve()
        if not image.is_file():
            raise PhysicsEstimationError(f"image does not exist: {image}")
        job["image"] = str(image)
    return value


def load_qwen(model_path: Path) -> tuple[Any, Any]:
    import torch
    from transformers import (
        AutoConfig,
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
        Qwen3VLForConditionalGeneration,
    )

    resolved = model_path.expanduser().resolve()
    if not resolved.is_dir():
        raise PhysicsEstimationError(f"model directory does not exist: {resolved}")
    processor = AutoProcessor.from_pretrained(
        str(resolved), local_files_only=True, trust_remote_code=True
    )
    config = AutoConfig.from_pretrained(
        str(resolved), local_files_only=True, trust_remote_code=True
    )
    model_class = {
        "qwen2_5_vl": Qwen2_5_VLForConditionalGeneration,
        "qwen3_vl": Qwen3VLForConditionalGeneration,
    }.get(config.model_type)
    if model_class is None:
        raise PhysicsEstimationError(
            f"unsupported Qwen-VL model_type: {config.model_type!r}"
        )
    model = model_class.from_pretrained(
        str(resolved),
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        local_files_only=True,
        trust_remote_code=True,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    return model, processor


def infer_one(
    *, model: Any, processor: Any, image_path: Path, prompt: str, max_new_tokens: int
) -> str:
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[chat],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [
        output[len(source) :]
        for source, output in zip(inputs.input_ids, generated, strict=True)
    ]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def run(args: argparse.Namespace) -> PhysicsManifest:
    if not 64 <= args.max_new_tokens <= 2000:
        raise PhysicsEstimationError("max-new-tokens must be in [64, 2000]")
    jobs_path = args.jobs.expanduser().resolve()
    jobs = load_jobs(jobs_path)
    output = args.output.expanduser().resolve()
    evidence_dir = (
        args.evidence_dir.expanduser().resolve()
        if args.evidence_dir is not None
        else output.parent / "physics_evidence"
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)
    model, processor = load_qwen(args.model)
    records: list[ObjectPhysicsRecord] = []
    for job in jobs["objects"]:
        image_path = Path(job["image"])
        prompt = build_prompt(job)
        raw = infer_one(
            model=model,
            processor=processor,
            image_path=image_path,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
        )
        properties = normalize_qwen_estimate(extract_json_object(raw))
        image_sha, image_size = sha256_file(image_path)
        evidence_path = evidence_dir / f"{job['object_id']}.qwen_vl_physics.json"
        atomic_write_json(
            evidence_path,
            {
                "schema_version": "1.0",
                "kind": "video2world.qwen_vl_physics_evidence",
                "object_id": job["object_id"],
                "model": str(args.model.expanduser().resolve()),
                "image": {
                    "path": str(image_path),
                    "sha256": image_sha,
                    "size_bytes": image_size,
                },
                "prompt": prompt,
                "raw_response": raw,
                "normalized_properties": properties.model_dump(mode="json"),
                "claim_boundary": (
                    "Qwen-VL output is an unvalidated category prior and not a physical "
                    "measurement or metric scene calibration."
                ),
            },
        )
        evidence_sha, evidence_size = sha256_file(evidence_path)
        properties = properties.model_copy(
            update={
                "evidence_uri": str(evidence_path),
                "evidence_sha256": evidence_sha,
                "evidence_size_bytes": evidence_size,
            }
        )
        records.append(ObjectPhysicsRecord(object_id=job["object_id"], properties=properties))
    manifest = PhysicsManifest(
        scene_id=jobs["scene_id"],
        run_id=jobs["run_id"],
        created_at=datetime.now(timezone.utc),  # noqa: UP017 - remote Qwen env is Python 3.10
        records=records,
    )
    atomic_write_json(output, manifest.model_dump(mode="json"))
    return manifest


def main() -> int:
    args = parse_args()
    try:
        manifest = run(args)
    except PhysicsEstimationError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "status": "passed_unvalidated_estimates",
                "records": len(manifest.records),
                "output": str(args.output.expanduser().resolve()),
                "promotion_allowed": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
