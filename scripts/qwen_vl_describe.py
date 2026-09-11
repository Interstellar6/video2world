#!/usr/bin/env python3
"""Run local Qwen2.5-VL on one task-local image and retain the raw response."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROMPT = """You are describing one physical object for a 3D reconstruction pipeline.
Return JSON only, with this exact top-level schema:
{
  "object_id": string,
  "category": string,
  "detailed_description": string,
  "components": [{"name": string, "description": string, "visibility": "visible"|"partial"|"occluded", "bbox_xyxy": [number, number, number, number] | null}],
  "materials": [string],
  "estimated_scale_hint": string,
  "physical_estimate": {
    "dimensions_m_lwh": [number, number, number] | null,
    "mass_kg": number | null,
    "surface_smoothness_0_to_1": number | null,
    "confidence_0_to_1": number,
    "basis": string
  },
  "geometry_caveats": [string]
}
Use pixel coordinates for bbox_xyxy in the supplied image. Do not claim unseen
geometry is observed; record it as a caveat. Physical estimates are priors, not
measurements: use null where the image does not support a numeric estimate and
state the uncertainty in basis. The target object id is: {object_id}."""


def parse_json_response(response: str) -> dict:
    """Accept JSON-only output and the common fenced-JSON variant."""
    candidate = response.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    return json.loads(candidate)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=700)
    args = parser.parse_args()

    from PIL import Image
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    image = Image.open(args.image).convert("RGB")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype, local_files_only=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    prompt = PROMPT.replace("{object_id}", args.object_id)
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    trimmed = [output[len(source):] for source, output in zip(inputs.input_ids, generated)]
    response = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    try:
        structured = parse_json_response(response)
        parse_error = None
    except json.JSONDecodeError as error:
        structured = None
        parse_error = str(error)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "kind": "video2world_modeling.qwen_vl_description",
        "model": str(args.model),
        "object_id": args.object_id,
        "image": str(args.image),
        "image_size": list(image.size),
        "device": device,
        "raw_response": response,
        "structured": structured,
        "json_parse_error": parse_error,
    }, indent=2) + "\n", encoding="utf-8")
    return 0 if structured is not None else 3


if __name__ == "__main__":
    raise SystemExit(main())
