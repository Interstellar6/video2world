#!/usr/bin/env python3
"""Generate review-only amodal object references through the Responses API.

The Responses controller invokes the image-generation edit tool.  GPT Image 2
does not provide native transparency, so prompts must request a flat chroma-key
background and a separate matte helper converts that background to alpha.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps

from scripts.run_responses_clean_plate_batch import (
    RETRYABLE_STATUS_CODES,
    artifact,
    atomic_write_bytes,
    atomic_write_json,
    extract_generated_png,
    image_data_url,
    require,
    response_field,
    sha256_file,
    sha256_text,
    validate_png,
)

SCHEMA_VERSION = 1
BATCH_KIND = "video2world.responses_object_completion_batch_run"
OBJECT_KIND = "video2world.responses_object_completion_run"
PROVIDER = "openai_responses_image_generation"
STATUS = "generated_candidates_pending_visual_review"


class ResponsesObjectCompletionError(RuntimeError):
    """Raised when an auditable object reference cannot be produced."""


def load_jobs(path: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResponsesObjectCompletionError(
            f"cannot read job manifest {manifest_path}: {exc}"
        ) from exc
    jobs = payload.get("objects")
    require(isinstance(jobs, list) and jobs, "job manifest requires a non-empty objects list")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in jobs:
        require(isinstance(item, dict), "each object job must be an object")
        object_id = item.get("object_id")
        prompt = item.get("prompt")
        source = Path(str(item.get("source_image", ""))).expanduser().resolve()
        require(isinstance(object_id, str) and object_id, "object_id is required")
        require(object_id not in seen, f"duplicate object_id: {object_id}")
        require(isinstance(prompt, str) and prompt.strip(), f"prompt is required for {object_id}")
        require(source.is_file(), f"source image does not exist for {object_id}: {source}")
        seen.add(object_id)
        normalized.append(
            {"object_id": object_id, "prompt": prompt.strip(), "source_image": source}
        )
    return normalized


def create_response(
    client: Any,
    *,
    job: dict[str, Any],
    controller_model: str,
    reasoning_effort: str,
    image_model: str,
    quality: str,
    requested_size: str,
    max_attempts: int,
) -> tuple[Any, int]:
    kwargs = {
        "model": controller_model,
        "reasoning": {"effort": reasoning_effort},
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": job["prompt"]},
                    {
                        "type": "input_image",
                        "image_url": image_data_url(job["source_image"]),
                        "detail": "high",
                    },
                ],
            }
        ],
        "tools": [
            {
                "type": "image_generation",
                "model": image_model,
                "action": "edit",
                "quality": quality,
                "size": requested_size,
                "output_format": "png",
            }
        ],
        "tool_choice": {"type": "image_generation"},
    }
    for attempt in range(1, max_attempts + 1):
        try:
            return client.responses.create(**kwargs), attempt
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            retryable = status_code in RETRYABLE_STATUS_CODES or status_code is None
            if attempt == max_attempts or not retryable:
                raise ResponsesObjectCompletionError(
                    f"Responses request failed for {job['object_id']} after "
                    f"{attempt} attempt(s): {exc}"
                ) from exc
            time.sleep(min(30.0, 2.0 ** (attempt - 1)))
    raise AssertionError("unreachable")


def run_matte_helper(raw_path: Path, rgba_path: Path, matte_script: Path) -> None:
    require(matte_script.is_file(), f"matte helper does not exist: {matte_script}")
    command = [
        sys.executable,
        str(matte_script),
        "--input",
        str(raw_path),
        "--out",
        str(rgba_path),
        "--auto-key",
        "border",
        "--soft-matte",
        "--spill-cleanup",
        "--edge-feather",
        "1",
        "--force",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise ResponsesObjectCompletionError(
            f"matte helper failed for {raw_path.name}: {completed.stderr.strip()}"
        )


def write_alpha_mask(rgba_path: Path, mask_path: Path) -> dict[str, float | int | bool]:
    with Image.open(rgba_path) as image:
        image.load()
        rgba = image.convert("RGBA")
        alpha = rgba.getchannel("A")
        alpha_values = list(alpha.getdata())
        total = len(alpha_values)
        transparent = sum(value == 0 for value in alpha_values)
        partial = sum(0 < value < 255 for value in alpha_values)
        visible = sum(value > 16 for value in alpha_values)
        binary = alpha.point(lambda value: 255 if value > 16 else 0)
    buffer = BytesIO()
    binary.save(buffer, format="PNG", optimize=True)
    atomic_write_bytes(mask_path, buffer.getvalue())
    visible_fraction = visible / total
    transparent_fraction = transparent / total
    return {
        "total_pixels": total,
        "transparent_pixels": transparent,
        "partially_transparent_pixels": partial,
        "visible_pixels": visible,
        "visible_fraction": visible_fraction,
        "transparent_fraction": transparent_fraction,
        "technical_gate_passed": 0.02 <= visible_fraction <= 0.90 and transparent_fraction >= 0.05,
    }


def build_contact_sheet(receipts: list[dict[str, Any]], output_path: Path) -> None:
    tile_size = (256, 256)
    label_height = 24
    sheet = Image.new(
        "RGB",
        (tile_size[0] * len(receipts), (tile_size[1] + label_height) * 2),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for column, receipt in enumerate(receipts):
        for row, output_role in enumerate(("raw", "generated_reference")):
            with Image.open(receipt["outputs"][output_role]["path"]) as image:
                image.load()
                if output_role == "generated_reference":
                    rgba = image.convert("RGBA")
                    checker = Image.new("RGB", rgba.size, (38, 38, 38))
                    checker.paste((110, 110, 110), (0, 0, rgba.width // 2, rgba.height))
                    checker.paste(rgba, mask=rgba.getchannel("A"))
                    source = checker
                else:
                    source = image.convert("RGB")
                tile = ImageOps.contain(source, tile_size, Image.Resampling.LANCZOS)
            x = column * tile_size[0] + (tile_size[0] - tile.width) // 2
            y0 = row * (tile_size[1] + label_height)
            y = y0 + label_height + (tile_size[1] - tile.height) // 2
            sheet.paste(tile, (x, y))
            label = f"{receipt['object_id']} / {output_role}"
            draw.text((column * tile_size[0] + 6, y0 + 5), label, fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="JPEG", quality=92, optimize=True)


def process_job(
    job: dict[str, Any],
    *,
    client: Any,
    output_dir: Path,
    matte_script: Path,
    controller_model: str,
    reasoning_effort: str,
    image_model: str,
    quality: str,
    requested_size: str,
    max_attempts: int,
) -> dict[str, Any]:
    object_id = job["object_id"]
    metadata_path = output_dir / "metadata" / f"{object_id}.json"
    raw_path = output_dir / "raw" / f"{object_id}.png"
    rgba_path = output_dir / "rgba" / f"{object_id}.png"
    mask_path = output_dir / "masks" / f"{object_id}.png"
    source_sha256, _ = sha256_file(job["source_image"])
    prompt_sha256 = sha256_text(job["prompt"])

    if metadata_path.is_file():
        try:
            receipt = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            receipt = None
        if (
            isinstance(receipt, dict)
            and receipt.get("source_sha256") == source_sha256
            and receipt.get("prompt_sha256") == prompt_sha256
            and receipt.get("controller_model") == controller_model
            and receipt.get("reasoning_effort") == reasoning_effort
            and receipt.get("image_model") == image_model
            and all(
                Path(value.get("path", "")).is_file()
                and sha256_file(Path(value["path"]))[0] == value.get("sha256")
                for value in receipt.get("outputs", {}).values()
            )
        ):
            return receipt

    started = time.perf_counter()
    response, attempt_count = create_response(
        client,
        job=job,
        controller_model=controller_model,
        reasoning_effort=reasoning_effort,
        image_model=image_model,
        quality=quality,
        requested_size=requested_size,
        max_attempts=max_attempts,
    )
    payload = extract_generated_png(response)
    raw_size = validate_png(payload)
    atomic_write_bytes(raw_path, payload)
    run_matte_helper(raw_path, rgba_path, matte_script)
    alpha_qa = write_alpha_mask(rgba_path, mask_path)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "kind": OBJECT_KIND,
        "status": STATUS,
        "promotion_allowed": False,
        "object_id": object_id,
        "protocol": "responses",
        "provider": PROVIDER,
        "response_id": response_field(response, "id"),
        "response_status": response_field(response, "status"),
        "attempt_count": attempt_count,
        "controller_model": controller_model,
        "reasoning_effort": reasoning_effort,
        "image_model": image_model,
        "image_action": "edit",
        "quality": quality,
        "requested_size": requested_size,
        "raw_size": list(raw_size),
        "source_sha256": source_sha256,
        "prompt_sha256": prompt_sha256,
        "input": artifact(job["source_image"]),
        "outputs": {
            "raw": artifact(raw_path),
            "generated_reference": artifact(rgba_path),
            "generated_region_mask": artifact(mask_path),
        },
        "matte": {
            "backend": "imagegen.remove_chroma_key",
            "script_sha256": sha256_file(matte_script)[0],
            "alpha_qa": alpha_qa,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "review": {
            "automatic_accept": False,
            "human_review_required": True,
            "technical_alpha_gate_passed": alpha_qa["technical_gate_passed"],
            "published": False,
        },
    }
    atomic_write_json(metadata_path, receipt)
    return receipt


def run_batch(
    job_manifest: str | Path,
    output_dir: str | Path,
    *,
    client: Any,
    matte_script: str | Path,
    controller_model: str = "gpt-5.5",
    reasoning_effort: str = "xhigh",
    image_model: str = "gpt-image-2",
    quality: str = "high",
    requested_size: str = "1024x1024",
    workers: int = 2,
    max_attempts: int = 3,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    require(1 <= workers <= 8, "workers must be inside [1, 8]")
    require(1 <= max_attempts <= 10, "max_attempts must be inside [1, 10]")
    jobs = load_jobs(job_manifest)
    destination = Path(output_dir).expanduser().resolve()
    matte = Path(matte_script).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    receipts: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_job,
                job,
                client=client,
                output_dir=destination,
                matte_script=matte,
                controller_model=controller_model,
                reasoning_effort=reasoning_effort,
                image_model=image_model,
                quality=quality,
                requested_size=requested_size,
                max_attempts=max_attempts,
            ): job["object_id"]
            for job in jobs
        }
        for future in as_completed(futures):
            receipts[futures[future]] = future.result()
    ordered = [receipts[job["object_id"]] for job in jobs]
    contact_sheet_path = destination / "qa" / "object_completion_contact_sheet.jpg"
    build_contact_sheet(ordered, contact_sheet_path)
    timestamp = created_at or datetime.now(UTC)
    require(timestamp.tzinfo is not None, "created_at must be timezone-aware")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "kind": BATCH_KIND,
        "status": STATUS,
        "created_at": timestamp.isoformat(),
        "promotion_allowed": False,
        "protocol": "responses",
        "provider": PROVIDER,
        "controller_model": controller_model,
        "reasoning_effort": reasoning_effort,
        "image_model": image_model,
        "quality": quality,
        "requested_size": requested_size,
        "aggregate": {
            "object_count": len(ordered),
            "ordered_object_ids": [item["object_id"] for item in ordered],
            "alpha_gate_passed_count": sum(
                bool(item["review"]["technical_alpha_gate_passed"]) for item in ordered
            ),
            "all_outputs_readable": True,
        },
        "objects": ordered,
        "outputs": {"contact_sheet": artifact(contact_sheet_path)},
        "review": {"human_review_required": True, "published": False},
    }
    atomic_write_json(destination / "batch_receipt.json", receipt)
    return receipt


def make_openai_client() -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ResponsesObjectCompletionError(
            "openai package is required; run with `uv run --with openai`"
        ) from exc
    return OpenAI()


def default_matte_script() -> Path:
    codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    return codex_home / "skills" / ".system" / "imagegen" / "scripts" / "remove_chroma_key.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--matte-script", type=Path, default=default_matte_script())
    parser.add_argument("--controller-model", default="gpt-5.5")
    parser.add_argument("--reasoning-effort", default="xhigh")
    parser.add_argument("--image-model", default="gpt-image-2")
    parser.add_argument("--quality", choices=("low", "medium", "high"), default="high")
    parser.add_argument("--requested-size", default="1024x1024")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = run_batch(
            args.job_manifest,
            args.output_dir,
            client=make_openai_client(),
            matte_script=args.matte_script,
            controller_model=args.controller_model,
            reasoning_effort=args.reasoning_effort,
            image_model=args.image_model,
            quality=args.quality,
            requested_size=args.requested_size,
            workers=args.workers,
            max_attempts=args.max_attempts,
        )
    except (ResponsesObjectCompletionError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt["aggregate"], ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
