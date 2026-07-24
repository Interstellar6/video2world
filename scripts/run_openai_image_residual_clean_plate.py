#!/usr/bin/env python3
"""Run GPT Image only on unresolved clean-plate residual pixels.

The input is a hash-bound output from prefill_clean_plate_multiview.py.
Measured donor pixels remain part of the input image, but are never sent as an
editable region and are checked byte-for-byte after compositing. The provider
result is a review-only 2D candidate: generated pixels are not donor or
geometry evidence and cannot advance a layered completion round by themselves.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

API_URL = "https://api.openai.com/v1/images/edits"
MODEL = "gpt-image-2"
STATUS = "generated_candidate_pending_multiview_review"
SCHEMA_VERSION = 1
MAX_TIMEOUT_SECONDS = 1_200
MIN_PIXELS = 655_360
MAX_PIXELS = 8_294_400
MAX_EDGE = 3_840
ENVIRONMENT_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class OpenAIResidualCleanPlateError(ValueError):
    """Raised when a review-only GPT Image candidate cannot be proven bounded."""


@dataclass(frozen=True)
class Asset:
    path: Path
    sha256: str
    size_bytes: int

    def record(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class ResidualInput:
    prefill_report: Asset
    frame_id: str
    report_status: str
    source_rgb: Asset
    prefill_rgb: Asset
    removal_mask: Asset
    residual_mask: Asset
    source: np.ndarray
    prefill: np.ndarray
    removal: np.ndarray
    residual: np.ndarray
    measured: np.ndarray


HttpPost = Callable[[str, bytes, Mapping[str, str], float], tuple[int, Mapping[str, str], bytes]]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise OpenAIResidualCleanPlateError(message)


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


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OpenAIResidualCleanPlateError(f"cannot read JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def required_text(value: Any, label: str, *, allow_empty: bool = False) -> str:
    require(isinstance(value, str), f"{label} must be a string")
    require(allow_empty or bool(value.strip()), f"{label} must be non-empty")
    return value


def required_sha256(value: Any, label: str) -> str:
    digest = required_text(value, label).lower()
    require(SHA256.fullmatch(digest) is not None, f"{label} must be a lowercase SHA-256 digest")
    return digest


def required_nonnegative_int(value: Any, label: str) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        f"{label} must be a non-negative integer",
    )
    return value


def resolve_verified_asset(path_value: Any, expected_sha: Any, *, label: str) -> Asset:
    path = Path(required_text(path_value, f"{label}.path")).expanduser().resolve()
    require(path.is_file(), f"{label}.path is not a regular file: {path}")
    expected = required_sha256(expected_sha, f"{label}.sha256")
    actual, size = sha256_file(path)
    require(actual == expected, f"{label} SHA-256 mismatch")
    return Asset(path=path, sha256=actual, size_bytes=size)


def load_rgb(asset: Asset, *, label: str) -> np.ndarray:
    try:
        with Image.open(asset.path) as image:
            image.load()
            require(image.mode == "RGB", f"{label} must be RGB, found {image.mode!r}")
            return np.array(image, dtype=np.uint8, copy=True)
    except OSError as exc:
        raise OpenAIResidualCleanPlateError(f"cannot decode {label}: {asset.path}") from exc


def load_binary_mask(asset: Asset, *, label: str) -> np.ndarray:
    try:
        with Image.open(asset.path) as image:
            image.load()
            require(image.mode in {"1", "L"}, f"{label} must be 1 or L")
            values = np.asarray(image)
    except OSError as exc:
        raise OpenAIResidualCleanPlateError(f"cannot decode {label}: {asset.path}") from exc
    require(values.ndim == 2, f"{label} must be single-channel")
    unique = {int(value) for value in np.unique(values)}
    require(unique <= {0, 1, 255}, f"{label} must be binary 0/1 or 0/255")
    return values.astype(bool)


def select_prefill_frame(
    prefill_report_path: str | Path,
    frame_id: str,
) -> ResidualInput:
    report_path = Path(prefill_report_path).expanduser().resolve()
    require(report_path.is_file(), f"prefill report is not a regular file: {report_path}")
    report_sha, report_size = sha256_file(report_path)
    report_asset = Asset(path=report_path, sha256=report_sha, size_bytes=report_size)
    report = read_json(report_path)
    require(report.get("schema_version") == 2, "prefill report schema_version must equal 2")
    require(report.get("promotion_approved") is False, "prefill report cannot be promoted")
    report_status = required_text(report.get("status"), "prefill report.status")
    require(
        report_status in {"technical_passed", "technical_failed", "technical_failed_no_support"},
        f"unsupported prefill report status: {report_status}",
    )
    provenance = report.get("pixel_provenance")
    require(isinstance(provenance, dict), "prefill report.pixel_provenance must be an object")
    require(
        provenance.get("generated_pixels") == 0,
        "prefill report contains generated pixels and is not a measurement-only input",
    )
    require(
        provenance.get("propainter_pixels") == 0,
        "prefill report contains ProPainter pixels and is not a measurement-only input",
    )
    gates = report.get("gates")
    require(isinstance(gates, dict), "prefill report.gates must be an object")
    require(
        gates.get("outside_removal_mask_rgb_exact") is True,
        "prefill report does not attest outside-removal exactness",
    )
    require(
        gates.get("all_residual_masks_subset_of_removal_masks") is True,
        "prefill report does not attest residual-mask containment",
    )
    raw_records = report.get("frame_records")
    require(isinstance(raw_records, list) and raw_records, "prefill report has no frame records")
    matches = [
        record
        for record in raw_records
        if isinstance(record, dict) and record.get("frame_id") == frame_id
    ]
    require(len(matches) == 1, f"prefill report must contain exactly one frame {frame_id}")
    record = matches[0]
    source_asset = resolve_verified_asset(
        record.get("source_frame"),
        record.get("source_frame_sha256"),
        label=f"frame {frame_id} source_frame",
    )
    prefill_asset = resolve_verified_asset(
        record.get("prefill_frame"),
        record.get("prefill_frame_sha256"),
        label=f"frame {frame_id} prefill_frame",
    )
    removal_asset = resolve_verified_asset(
        record.get("removal_mask"),
        record.get("removal_mask_sha256"),
        label=f"frame {frame_id} removal_mask",
    )
    residual_asset = resolve_verified_asset(
        record.get("residual_mask"),
        record.get("residual_mask_sha256"),
        label=f"frame {frame_id} residual_mask",
    )
    source = load_rgb(source_asset, label=f"frame {frame_id} source_frame")
    prefill = load_rgb(prefill_asset, label=f"frame {frame_id} prefill_frame")
    removal = load_binary_mask(removal_asset, label=f"frame {frame_id} removal_mask")
    residual = load_binary_mask(residual_asset, label=f"frame {frame_id} residual_mask")
    require(source.shape == prefill.shape, f"frame {frame_id} source/prefill dimensions differ")
    require(removal.shape == source.shape[:2], f"frame {frame_id} removal dimensions differ")
    require(residual.shape == source.shape[:2], f"frame {frame_id} residual dimensions differ")
    require(bool(residual.any()), f"frame {frame_id} residual mask is empty")
    require(
        not bool(np.any(residual & ~removal)),
        f"frame {frame_id} residual mask escapes the removal mask",
    )
    measured = removal & ~residual
    removal_pixels = int(removal.sum())
    residual_pixels = int(residual.sum())
    measured_pixels = int(measured.sum())
    require(
        required_nonnegative_int(record.get("removal_mask_pixels"), "removal_mask_pixels")
        == removal_pixels,
        f"frame {frame_id} removal_mask_pixels does not match the mask",
    )
    require(
        required_nonnegative_int(record.get("residual_mask_pixels"), "residual_mask_pixels")
        == residual_pixels,
        f"frame {frame_id} residual_mask_pixels does not match the mask",
    )
    require(
        required_nonnegative_int(record.get("covered_pixels"), "covered_pixels") == measured_pixels,
        f"frame {frame_id} covered_pixels does not match the measured region",
    )
    require(
        required_nonnegative_int(
            record.get("measured_depth_valid_pixels"),
            "measured_depth_valid_pixels",
        )
        == measured_pixels,
        f"frame {frame_id} measured depth validity does not match the measured region",
    )
    require(
        record.get("outside_removal_mask_rgb_exact") is True
        and bool(np.array_equal(prefill[~removal], source[~removal])),
        f"frame {frame_id} prefill differs from source outside the removal mask",
    )
    if report_status == "technical_passed":
        require(
            gates.get("donor_support_not_boundary_concentrated") is True,
            "technical-passed prefill lacks stable donor support",
        )
    else:
        require(
            measured_pixels == 0,
            "failed prefill cannot contribute measured pixels to a generative candidate",
        )
        require(
            bool(np.array_equal(prefill, source)),
            "failed zero-measurement prefill must remain byte-exact source RGB",
        )
    return ResidualInput(
        prefill_report=report_asset,
        frame_id=frame_id,
        report_status=report_status,
        source_rgb=source_asset,
        prefill_rgb=prefill_asset,
        removal_mask=removal_asset,
        residual_mask=residual_asset,
        source=source,
        prefill=prefill,
        removal=removal,
        residual=residual,
        measured=measured,
    )


def validate_gpt_image_size(rgb: np.ndarray) -> str:
    height, width = rgb.shape[:2]
    require(width % 16 == 0 and height % 16 == 0, "GPT Image dimensions must be multiples of 16")
    require(max(width, height) <= MAX_EDGE, "GPT Image dimensions exceed the maximum edge")
    require(max(width, height) <= 3 * min(width, height), "GPT Image aspect ratio exceeds 3:1")
    pixels = width * height
    require(
        MIN_PIXELS <= pixels <= MAX_PIXELS, "GPT Image dimensions fall outside the pixel limits"
    )
    return f"{width}x{height}"


def png_bytes(rgb: np.ndarray) -> bytes:
    output = BytesIO()
    Image.fromarray(rgb).save(output, format="PNG")
    return output.getvalue()


def edit_mask_png(residual: np.ndarray) -> bytes:
    """Encode transparent editable pixels for the Images edits endpoint."""

    rgba = np.full((*residual.shape, 4), 255, dtype=np.uint8)
    rgba[..., 3] = np.where(residual, 0, 255).astype(np.uint8)
    output = BytesIO()
    Image.fromarray(rgba).save(output, format="PNG")
    return output.getvalue()


def multipart_body(
    *,
    fields: Mapping[str, str],
    image_png: bytes,
    mask_png: bytes,
) -> tuple[str, bytes]:
    boundary = f"----video2world-{uuid.uuid4().hex}"
    chunks: list[bytes] = []

    def append_text(name: str, value: str) -> None:
        chunks.extend(
            (
                f"--{boundary}\r\n".encode("ascii"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("ascii"),
                value.encode("utf-8"),
                b"\r\n",
            )
        )

    def append_file(name: str, filename: str, value: bytes) -> None:
        chunks.extend(
            (
                f"--{boundary}\r\n".encode("ascii"),
                (
                    f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                ).encode("ascii"),
                b"Content-Type: image/png\r\n\r\n",
                value,
                b"\r\n",
            )
        )

    for name, value in fields.items():
        append_text(name, value)
    append_file("image[]", "prefill.png", image_png)
    append_file("mask", "residual-transparent.png", mask_png)
    chunks.append(f"--{boundary}--\r\n".encode("ascii"))
    return boundary, b"".join(chunks)


def default_http_post(
    url: str,
    payload: bytes,
    headers: Mapping[str, str],
    timeout_seconds: float,
) -> tuple[int, Mapping[str, str], bytes]:
    request = Request(url, data=payload, headers=dict(headers), method="POST")
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return (
                int(response.getcode()),
                {key.lower(): value for key, value in response.headers.items()},
                response.read(),
            )
    except HTTPError as exc:
        raise OpenAIResidualCleanPlateError(
            f"OpenAI Images edits request failed with HTTP {exc.code}"
        ) from exc
    except URLError as exc:
        raise OpenAIResidualCleanPlateError(
            "OpenAI Images edits request failed at the network layer"
        ) from exc


def decode_provider_image(payload: bytes, *, expected_shape: tuple[int, int, int]) -> np.ndarray:
    try:
        response = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OpenAIResidualCleanPlateError("OpenAI Images edits response is not JSON") from exc
    require(isinstance(response, dict), "OpenAI Images edits response must be an object")
    data = response.get("data")
    require(isinstance(data, list) and len(data) == 1, "response must contain one image")
    item = data[0]
    require(isinstance(item, dict), "response image entry must be an object")
    encoded = item.get("b64_json")
    require(isinstance(encoded, str) and encoded, "response image has no b64_json")
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (OSError, ValueError) as exc:
        raise OpenAIResidualCleanPlateError("response b64_json is not a valid image") from exc
    require(rgb.shape == expected_shape, "provider image dimensions differ from the prefill input")
    return np.array(rgb, dtype=np.uint8, copy=True)


def output_asset(path: Path, *, root: Path) -> dict[str, Any]:
    digest, size = sha256_file(path)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": digest,
        "size_bytes": size,
    }


def save_png(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value).save(path, format="PNG")


def run_openai_image_residual_clean_plate(
    *,
    prefill_report_path: str | Path,
    frame_id: str,
    output_dir: str | Path,
    prompt: str,
    quality: str = "high",
    api_key_env: str = "OPENAI_API_KEY",
    timeout_seconds: int = 600,
    created_at: datetime | None = None,
    http_post: HttpPost | None = None,
) -> dict[str, Any]:
    """Create one bounded GPT Image candidate from a measured prefill residual."""

    frame_id = required_text(frame_id, "frame_id").strip()
    prompt = required_text(prompt, "prompt")
    require(quality in {"low", "medium", "high", "auto"}, "quality is invalid")
    require(
        isinstance(timeout_seconds, int)
        and not isinstance(timeout_seconds, bool)
        and 1 <= timeout_seconds <= MAX_TIMEOUT_SECONDS,
        f"timeout_seconds must be inside [1, {MAX_TIMEOUT_SECONDS}]",
    )
    require(
        isinstance(api_key_env, str) and ENVIRONMENT_NAME.fullmatch(api_key_env) is not None,
        "api_key_env must be an uppercase environment variable name",
    )
    bound = select_prefill_frame(prefill_report_path, frame_id)
    size = validate_gpt_image_size(bound.prefill)
    api_key = os.environ.get(api_key_env)
    require(bool(api_key), f"{api_key_env} is not set")

    destination_candidate = Path(output_dir).expanduser()
    require(not destination_candidate.is_symlink(), f"output_dir cannot be a symlink: {output_dir}")
    destination = destination_candidate.resolve()
    require(not destination.exists(), f"refusing to overwrite output_dir: {destination}")
    timestamp = created_at or datetime.now(timezone.utc)  # noqa: UP017
    require(
        timestamp.tzinfo is not None and timestamp.utcoffset() is not None,
        "created_at must include timezone information",
    )
    image_bytes = png_bytes(bound.prefill)
    mask_bytes = edit_mask_png(bound.residual)
    boundary, request_payload = multipart_body(
        fields={
            "model": MODEL,
            "prompt": prompt,
            "size": size,
            "quality": quality,
            "output_format": "png",
        },
        image_png=image_bytes,
        mask_png=mask_bytes,
    )
    request_headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    }
    post = http_post or default_http_post
    started = time.perf_counter()
    status_code, response_headers, response_payload = post(
        API_URL,
        request_payload,
        request_headers,
        float(timeout_seconds),
    )
    elapsed_seconds = time.perf_counter() - started
    require(200 <= status_code < 300, f"OpenAI Images edits returned HTTP {status_code}")
    provider_rgb = decode_provider_image(response_payload, expected_shape=bound.prefill.shape)
    changed_inside_residual = int(
        np.count_nonzero(np.any(provider_rgb != bound.prefill, axis=2) & bound.residual)
    )
    require(changed_inside_residual > 0, "provider did not change any residual pixel")
    provider_changed_outside_residual = int(
        np.count_nonzero(np.any(provider_rgb != bound.prefill, axis=2) & ~bound.residual)
    )
    candidate = bound.prefill.copy()
    candidate[bound.residual] = provider_rgb[bound.residual]
    require(
        bool(np.array_equal(candidate[~bound.residual], bound.prefill[~bound.residual])),
        "hard composite changed a non-residual pixel",
    )
    require(
        bool(np.array_equal(candidate[bound.measured], bound.prefill[bound.measured])),
        "hard composite changed a measured multiview pixel",
    )
    require(
        bool(np.array_equal(candidate[~bound.removal], bound.source[~bound.removal])),
        "candidate changed a pixel outside the original removal mask",
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        provider_input_path = staging / "provider_input.png"
        provider_path = staging / "provider_raw.png"
        candidate_path = staging / "candidate.png"
        mask_path = staging / "residual_transparent_mask.png"
        provider_input_path.write_bytes(image_bytes)
        save_png(provider_path, provider_rgb)
        save_png(candidate_path, candidate)
        mask_path.write_bytes(mask_bytes)
        request_id = response_headers.get("x-request-id") or response_headers.get("X-Request-Id")
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "kind": "video2world.openai_image_residual_clean_plate_run",
            "status": STATUS,
            "promotion_allowed": False,
            "created_at": timestamp.isoformat(),
            "provider": {
                "name": "openai_images_edits",
                "endpoint": API_URL,
                "model": MODEL,
                "api_key_environment": api_key_env,
                "response_request_id": request_id,
                "elapsed_seconds": elapsed_seconds,
            },
            "request": {
                "prompt": {"text": prompt, "sha256": sha256_text(prompt)},
                "quality": quality,
                "size": size,
                "output_format": "png",
                "mask_semantics": "transparent_residual_pixels_are_the_edit_request",
                "provider_mask_is_guidance_only": True,
            },
            "upstream_prefill": {
                "report": bound.prefill_report.record(),
                "frame_id": bound.frame_id,
                "status": bound.report_status,
                "source_rgb": bound.source_rgb.record(),
                "prefill_rgb": bound.prefill_rgb.record(),
                "removal_mask": bound.removal_mask.record(),
                "residual_mask": bound.residual_mask.record(),
            },
            "pixel_provenance": {
                "removal_pixels": int(bound.removal.sum()),
                "measured_multiview_pixels_preserved": int(bound.measured.sum()),
                "generated_residual_pixels": int(bound.residual.sum()),
                "unresolved_pixels_before_generation": int(bound.residual.sum()),
                "generated_pixels_are_not_donor_or_geometry_evidence": True,
                "classes_are_mutually_exclusive": True,
            },
            "exactness": {
                "provider_changed_residual_pixels": changed_inside_residual,
                "provider_changed_outside_residual_pixels_before_composite": (
                    provider_changed_outside_residual
                ),
                "non_residual_pixels_equal_prefill_after_composite": True,
                "measured_multiview_pixels_equal_prefill_after_composite": True,
                "outside_original_removal_pixels_equal_source_after_composite": True,
            },
            "outputs": {
                "provider_input_rgb": output_asset(provider_input_path, root=staging),
                "provider_raw_rgb": output_asset(provider_path, root=staging),
                "candidate_rgb": output_asset(candidate_path, root=staging),
                "transparent_residual_mask": output_asset(mask_path, root=staging),
            },
            "review": {
                "human_review_required": True,
                "multiview_and_temporal_review_required": True,
                "geometry_reconstruction_allowed": False,
                "next_layer_allowed": False,
                "published": False,
            },
        }
        receipt_path = staging / "run_receipt.json"
        receipt_path.write_text(
            json.dumps(receipt, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill-report", type=Path, required=True)
    parser.add_argument("--frame-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--quality", choices=("low", "medium", "high", "auto"), default="high")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        receipt = run_openai_image_residual_clean_plate(
            prefill_report_path=args.prefill_report,
            frame_id=args.frame_id,
            output_dir=args.output_dir,
            prompt=args.prompt,
            quality=args.quality,
            api_key_env=args.api_key_env,
            timeout_seconds=args.timeout_seconds,
        )
    except OpenAIResidualCleanPlateError as exc:
        print(f"error: {exc}", file=os.sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output_dir": str(args.output_dir.expanduser().resolve()),
                "promotion_allowed": receipt["promotion_allowed"],
                "generated_residual_pixels": receipt["pixel_provenance"][
                    "generated_residual_pixels"
                ],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
