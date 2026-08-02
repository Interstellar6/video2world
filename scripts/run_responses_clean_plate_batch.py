#!/usr/bin/env python3
"""Generate review-only clean plates with the Responses image-generation tool.

The controller and image model are deliberately separate: a Responses model
orchestrates the ``image_generation`` tool, while the tool performs an edit
conditioned on the target frame, adjacent source frames, and one accepted clean
anchor. API credentials are read only by the OpenAI client from the process
environment and are never serialized into receipts.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

SCHEMA_VERSION = 1
RECEIPT_KIND = "video2world.responses_clean_plate_batch_run"
FRAME_KIND = "video2world.responses_clean_plate_frame_run"
STATUS = "generated_candidates_pending_visual_review"
PROVIDER = "openai_responses_image_generation"
OUTPUT_SIZE = (1280, 720)
RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


class ResponsesCleanPlateError(RuntimeError):
    """Raised when a reviewable, provenance-bound candidate cannot be produced."""


@dataclass(frozen=True)
class FrameJob:
    frame_id: str
    target: Path
    previous: Path
    following: Path
    prompt: str


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ResponsesCleanPlateError(message)


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def artifact(path: Path) -> dict[str, Any]:
    digest, size = sha256_file(path)
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        mode = image.mode
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "size_bytes": size,
        "width": width,
        "height": height,
        "mode": mode,
    }


def image_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def response_field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def extract_generated_png(response: Any) -> bytes:
    for item in response_field(response, "output", []) or []:
        if response_field(item, "type") != "image_generation_call":
            continue
        encoded = response_field(item, "result")
        if isinstance(encoded, str) and encoded:
            try:
                return base64.b64decode(encoded, validate=True)
            except ValueError as exc:
                raise ResponsesCleanPlateError("image generation returned invalid base64") from exc
    raise ResponsesCleanPlateError("response contained no image_generation_call result")


def validate_png(payload: bytes) -> tuple[int, int]:
    try:
        with Image.open(BytesIO(payload)) as image:
            image.load()
            require(image.format == "PNG", f"image tool returned {image.format}, expected PNG")
            require(image.width > 0 and image.height > 0, "image tool returned an empty image")
            return image.size
    except (OSError, ValueError) as exc:
        raise ResponsesCleanPlateError(f"image tool returned an unreadable PNG: {exc}") from exc


def normalize_png(raw_path: Path, normalized_path: Path) -> None:
    with Image.open(raw_path) as image:
        image.load()
        normalized = image.convert("RGB").resize(OUTPUT_SIZE, Image.Resampling.LANCZOS)
    buffer = BytesIO()
    normalized.save(buffer, format="PNG", optimize=True)
    atomic_write_bytes(normalized_path, buffer.getvalue())


def list_frame_paths(input_dir: Path) -> list[Path]:
    paths = sorted(
        path.resolve()
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    require(paths, f"input directory has no images: {input_dir}")
    stems = [path.stem for path in paths]
    require(len(stems) == len(set(stems)), "input image stems must be unique")
    return paths


def build_jobs(
    frame_paths: list[Path],
    *,
    anchor_frame: str,
    prompt: str,
    anchor_prompt: str,
) -> tuple[list[FrameJob], Path]:
    by_id = {path.stem: path for path in frame_paths}
    require(anchor_frame in by_id, f"anchor frame is not present: {anchor_frame}")
    jobs = []
    for index, target in enumerate(frame_paths):
        jobs.append(
            FrameJob(
                frame_id=target.stem,
                target=target,
                previous=frame_paths[max(0, index - 1)],
                following=frame_paths[min(len(frame_paths) - 1, index + 1)],
                prompt=anchor_prompt if target.stem == anchor_frame else prompt,
            )
        )
    return jobs, by_id[anchor_frame]


def make_request_content(
    job: FrameJob,
    *,
    anchor_path: Path,
    include_anchor: bool,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "input_text", "text": job.prompt}]
    paths = [job.target, job.previous, job.following]
    if include_anchor:
        paths.append(anchor_path)
    content.extend(
        {"type": "input_image", "image_url": image_data_url(path), "detail": "high"}
        for path in paths
    )
    return content


def create_response(
    client: Any,
    *,
    job: FrameJob,
    anchor_path: Path,
    controller_model: str,
    reasoning_effort: str,
    image_model: str,
    quality: str,
    requested_size: str,
    max_attempts: int,
) -> tuple[Any, int]:
    include_anchor = job.target.resolve() != anchor_path.resolve()
    kwargs = {
        "model": controller_model,
        "reasoning": {"effort": reasoning_effort},
        "input": [
            {
                "role": "user",
                "content": make_request_content(
                    job,
                    anchor_path=anchor_path,
                    include_anchor=include_anchor,
                ),
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
                raise ResponsesCleanPlateError(
                    f"Responses request failed for {job.frame_id} after {attempt} attempt(s): {exc}"
                ) from exc
            time.sleep(min(30.0, 2.0 ** (attempt - 1)))
    raise AssertionError("unreachable")


def seed_candidate(seed_dir: Path | None, frame_id: str) -> tuple[Path, dict[str, Any]] | None:
    if seed_dir is None:
        return None
    image_path = seed_dir / f"{frame_id}.png"
    receipt_path = seed_dir / f"{frame_id}.response.json"
    if not image_path.is_file() or not receipt_path.is_file():
        return None
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResponsesCleanPlateError(f"cannot read seed receipt {receipt_path}: {exc}") from exc
    require(receipt.get("protocol") == "responses", f"seed {frame_id} is not a Responses result")
    expected = receipt.get("output", {}).get("sha256")
    actual, _ = sha256_file(image_path)
    require(expected == actual, f"seed candidate hash mismatch for {frame_id}")
    validate_png(image_path.read_bytes())
    return image_path, receipt


def existing_frame_receipt(
    metadata_path: Path,
    *,
    job: FrameJob,
    anchor_sha256: str,
    controller_model: str,
    reasoning_effort: str,
    image_model: str,
) -> dict[str, Any] | None:
    if not metadata_path.is_file():
        return None
    try:
        value = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    expected = {
        "frame_id": job.frame_id,
        "prompt_sha256": sha256_text(job.prompt),
        "anchor_sha256": anchor_sha256,
        "controller_model": controller_model,
        "reasoning_effort": reasoning_effort,
        "image_model": image_model,
    }
    if any(value.get(key) != item for key, item in expected.items()):
        return None
    for output in value.get("outputs", {}).values():
        path = Path(output.get("path", ""))
        if not path.is_file() or sha256_file(path)[0] != output.get("sha256"):
            return None
    return value


def process_frame(
    job: FrameJob,
    *,
    client: Any,
    anchor_path: Path,
    anchor_sha256: str,
    output_dir: Path,
    seed_dir: Path | None,
    controller_model: str,
    reasoning_effort: str,
    image_model: str,
    quality: str,
    requested_size: str,
    max_attempts: int,
) -> dict[str, Any]:
    raw_path = (output_dir / "raw" / f"{job.frame_id}.png").resolve()
    normalized_path = (output_dir / "frames" / f"{job.frame_id}.png").resolve()
    metadata_path = (output_dir / "metadata" / f"{job.frame_id}.json").resolve()
    existing = existing_frame_receipt(
        metadata_path,
        job=job,
        anchor_sha256=anchor_sha256,
        controller_model=controller_model,
        reasoning_effort=reasoning_effort,
        image_model=image_model,
    )
    if existing is not None:
        return existing

    started = time.perf_counter()
    seeded = seed_candidate(seed_dir, job.frame_id)
    if seeded is not None:
        source_path, source_receipt = seeded
        payload = source_path.read_bytes()
        response_id = source_receipt.get("response_id")
        response_status = source_receipt.get("response_status")
        attempt_count = 0
        origin = "verified_seed_candidate"
    else:
        response, attempt_count = create_response(
            client,
            job=job,
            anchor_path=anchor_path,
            controller_model=controller_model,
            reasoning_effort=reasoning_effort,
            image_model=image_model,
            quality=quality,
            requested_size=requested_size,
            max_attempts=max_attempts,
        )
        payload = extract_generated_png(response)
        response_id = response_field(response, "id")
        response_status = response_field(response, "status")
        origin = "fresh_response"
    raw_width, raw_height = validate_png(payload)
    atomic_write_bytes(raw_path, payload)
    normalize_png(raw_path, normalized_path)
    inputs = [artifact(job.target), artifact(job.previous), artifact(job.following)]
    if job.target.resolve() != anchor_path.resolve():
        inputs.append(artifact(anchor_path))
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "kind": FRAME_KIND,
        "status": STATUS,
        "promotion_allowed": False,
        "frame_id": job.frame_id,
        "protocol": "responses",
        "provider": PROVIDER,
        "origin": origin,
        "response_id": response_id,
        "response_status": response_status,
        "attempt_count": attempt_count,
        "controller_model": controller_model,
        "reasoning_effort": reasoning_effort,
        "image_model": image_model,
        "image_action": "edit",
        "quality": quality,
        "requested_size": requested_size,
        "raw_size": [raw_width, raw_height],
        "normalized_size": list(OUTPUT_SIZE),
        "normalization": "RGB Lanczos resize; raw provider output preserved",
        "prompt_sha256": sha256_text(job.prompt),
        "anchor_sha256": anchor_sha256,
        "inputs": inputs,
        "outputs": {
            "raw": artifact(raw_path),
            "camera_calibrated": artifact(normalized_path),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "review": {
            "automatic_accept": False,
            "human_review_required": True,
            "published": False,
        },
    }
    atomic_write_json(metadata_path, receipt)
    return receipt


def run_responses_clean_plate_batch(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    anchor_frame: str,
    prompt: str,
    anchor_prompt: str,
    client: Any,
    seed_dir: str | Path | None = None,
    controller_model: str = "gpt-5.5",
    reasoning_effort: str = "xhigh",
    image_model: str = "gpt-image-2",
    quality: str = "medium",
    requested_size: str = "1280x720",
    workers: int = 3,
    max_attempts: int = 3,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """Generate or resume a complete review-only sequence."""

    source = Path(input_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    seed = Path(seed_dir).expanduser().resolve() if seed_dir else None
    require(source.is_dir(), f"input directory does not exist: {source}")
    require(seed is None or seed.is_dir(), f"seed directory does not exist: {seed}")
    require(1 <= workers <= 16, "workers must be inside [1, 16]")
    require(1 <= max_attempts <= 10, "max_attempts must be inside [1, 10]")
    frame_paths = list_frame_paths(source)
    jobs, original_anchor_path = build_jobs(
        frame_paths,
        anchor_frame=anchor_frame,
        prompt=prompt,
        anchor_prompt=anchor_prompt,
    )
    destination.mkdir(parents=True, exist_ok=True)

    # The accepted clean anchor is either a verified prior response or this
    # batch's normalized result. Generate/adopt it before concurrent dependents.
    anchor_job = next(job for job in jobs if job.frame_id == anchor_frame)
    seeded_anchor = seed_candidate(seed, anchor_frame)
    if seeded_anchor is not None:
        clean_anchor_path = seeded_anchor[0]
        anchor_sha256, _ = sha256_file(clean_anchor_path)
        anchor_receipt = process_frame(
            anchor_job,
            client=client,
            anchor_path=clean_anchor_path,
            anchor_sha256=anchor_sha256,
            output_dir=destination,
            seed_dir=seed,
            controller_model=controller_model,
            reasoning_effort=reasoning_effort,
            image_model=image_model,
            quality=quality,
            requested_size=requested_size,
            max_attempts=max_attempts,
        )
    else:
        original_anchor_sha, _ = sha256_file(original_anchor_path)
        anchor_receipt = process_frame(
            anchor_job,
            client=client,
            anchor_path=original_anchor_path,
            anchor_sha256=original_anchor_sha,
            output_dir=destination,
            seed_dir=seed,
            controller_model=controller_model,
            reasoning_effort=reasoning_effort,
            image_model=image_model,
            quality=quality,
            requested_size=requested_size,
            max_attempts=max_attempts,
        )
        clean_anchor_path = destination / "raw" / f"{anchor_frame}.png"
    require(clean_anchor_path.is_file(), f"clean anchor was not generated: {clean_anchor_path}")
    anchor_sha256, _ = sha256_file(clean_anchor_path)

    frame_receipts: dict[str, dict[str, Any]] = {anchor_frame: anchor_receipt}
    remaining = [job for job in jobs if job.frame_id != anchor_frame]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_frame,
                job,
                client=client,
                anchor_path=clean_anchor_path,
                anchor_sha256=anchor_sha256,
                output_dir=destination,
                seed_dir=seed,
                controller_model=controller_model,
                reasoning_effort=reasoning_effort,
                image_model=image_model,
                quality=quality,
                requested_size=requested_size,
                max_attempts=max_attempts,
            ): job.frame_id
            for job in remaining
        }
        for future in as_completed(futures):
            frame_receipts[futures[future]] = future.result()

    ordered = [frame_receipts[job.frame_id] for job in jobs]
    timestamp = created_at or datetime.now(UTC)
    require(timestamp.tzinfo is not None, "created_at must be timezone-aware")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "kind": RECEIPT_KIND,
        "status": STATUS,
        "created_at": timestamp.isoformat(),
        "provider": PROVIDER,
        "promotion_allowed": False,
        "protocol": "responses",
        "controller_model": controller_model,
        "reasoning_effort": reasoning_effort,
        "image_model": image_model,
        "image_action": "edit",
        "quality": quality,
        "requested_size": requested_size,
        "normalized_size": list(OUTPUT_SIZE),
        "anchor_frame": anchor_frame,
        "anchor": artifact(clean_anchor_path),
        "prompts": {
            "anchor": {"text": anchor_prompt, "sha256": sha256_text(anchor_prompt)},
            "followup": {"text": prompt, "sha256": sha256_text(prompt)},
        },
        "aggregate": {
            "frame_count": len(ordered),
            "ordered_frame_ids": [item["frame_id"] for item in ordered],
            "fresh_response_count": sum(item["origin"] == "fresh_response" for item in ordered),
            "verified_seed_count": sum(
                item["origin"] == "verified_seed_candidate" for item in ordered
            ),
            "all_outputs_readable": True,
            "all_normalized_outputs_exact_size": all(
                item["outputs"]["camera_calibrated"]["width"] == OUTPUT_SIZE[0]
                and item["outputs"]["camera_calibrated"]["height"] == OUTPUT_SIZE[1]
                for item in ordered
            ),
        },
        "frames": ordered,
        "review": {
            "automatic_accept": False,
            "human_review_required": True,
            "temporal_review_required": True,
            "published": False,
        },
    }
    atomic_write_json(destination / "batch_receipt.json", receipt)
    return receipt


def make_openai_client() -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ResponsesCleanPlateError(
            "openai package is required; run with `uv run --with openai`"
        ) from exc
    return OpenAI()


def read_prompt(path: Path) -> str:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise ResponsesCleanPlateError(f"cannot read prompt {path}: {exc}") from exc
    require(bool(value), f"prompt is empty: {path}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anchor-frame", required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--anchor-prompt-file", type=Path, required=True)
    parser.add_argument("--seed-dir", type=Path)
    parser.add_argument("--controller-model", default="gpt-5.5")
    parser.add_argument("--reasoning-effort", default="xhigh")
    parser.add_argument("--image-model", default="gpt-image-2")
    parser.add_argument("--quality", choices=("low", "medium", "high"), default="medium")
    parser.add_argument("--requested-size", default="1280x720")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = run_responses_clean_plate_batch(
            args.input_dir,
            args.output_dir,
            anchor_frame=args.anchor_frame,
            prompt=read_prompt(args.prompt_file),
            anchor_prompt=read_prompt(args.anchor_prompt_file),
            client=make_openai_client(),
            seed_dir=args.seed_dir,
            controller_model=args.controller_model,
            reasoning_effort=args.reasoning_effort,
            image_model=args.image_model,
            quality=args.quality,
            requested_size=args.requested_size,
            workers=args.workers,
            max_attempts=args.max_attempts,
        )
    except ResponsesCleanPlateError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt["aggregate"], ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
