#!/usr/bin/env python3
"""Run an ordered, hash-bound SDXL clean-plate batch with one model load.

The input manifest binds every source frame, ownership mask, and optional guide
RGB.  The ownership mask is dilated into a larger Euclidean context mask for
generation, while the final candidate uses the ownership mask as an exact core
plus a soft outer collar.  All outputs remain review-only and are published by
one non-overwriting atomic directory rename after the complete batch succeeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

try:
    from scripts.compose_clean_plate_boundary_candidate import (
        blend_candidate,
        build_soft_bands,
    )
except ModuleNotFoundError:  # Direct execution places scripts/ on sys.path.
    from compose_clean_plate_boundary_candidate import blend_candidate, build_soft_bands

INPUT_KIND = "video2world.diffusion_clean_plate_batch_input"
RECEIPT_KIND = "video2world.diffusion_clean_plate_batch_run"
FRAME_RECEIPT_KIND = "video2world.diffusion_clean_plate_frame_run"
SCHEMA_VERSION = 1
STATUS = "generated_batch_pending_review"
PROVIDER = "local_huggingface_diffusers"
SEED_POLICY = "base_plus_sequence_index"
PIPELINE_CLASS = "StableDiffusionXLInpaintPipeline"
DTYPES = ("float16", "bfloat16", "float32")
MAX_INFERENCE_STEPS = 1_000
MAX_MASK_RADIUS = 4_096
MAX_SEED = 2**63 - 1


class DiffusionCleanPlateBatchError(ValueError):
    """Raised when the batch cannot produce trustworthy review artifacts."""


@dataclass(frozen=True)
class LoadedPipeline:
    pipeline: Any
    runtime: dict[str, Any]


@dataclass(frozen=True)
class ResolvedAsset:
    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class FrameInput:
    sequence_index: int
    frame_id: str
    seed: int
    source_asset: ResolvedAsset
    ownership_asset: ResolvedAsset
    guide_asset: ResolvedAsset | None
    protected_asset: ResolvedAsset | None
    source_rgb: np.ndarray
    ownership_mask: np.ndarray
    guide_rgb: np.ndarray | None
    protected_mask: np.ndarray


@dataclass(frozen=True)
class BatchInput:
    manifest_path: Path
    manifest_sha256: str
    manifest_size_bytes: int
    config: dict[str, Any]
    model_path: Path
    model_files: dict[str, dict[str, Any]]
    frames: tuple[FrameInput, ...]


PipelineLoader = Callable[..., LoadedPipeline]
GeneratorFactory = Callable[[int], Any]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DiffusionCleanPlateBatchError(message)


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


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def save_png(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    Image.fromarray(value).save(temporary, format="PNG")
    os.replace(temporary, path)


def output_artifact(path: Path, *, root: Path) -> dict[str, Any]:
    digest, size = sha256_file(path)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": digest,
        "size_bytes": size,
    }


def input_artifact(asset: ResolvedAsset) -> dict[str, Any]:
    return {
        "path": str(asset.path),
        "sha256": asset.sha256,
        "size_bytes": asset.size_bytes,
    }


def required_text(value: Any, label: str, *, allow_empty: bool = False) -> str:
    require(isinstance(value, str), f"{label} must be a string")
    require(allow_empty or bool(value.strip()), f"{label} must be non-empty")
    return value


def required_sha256(value: Any, label: str) -> str:
    digest = required_text(value, label).strip().lower()
    require(
        len(digest) == 64 and all(character in "0123456789abcdef" for character in digest),
        f"{label} must be a lowercase hexadecimal SHA-256 digest",
    )
    return digest


def required_int(
    value: Any,
    label: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool) and minimum <= value <= maximum,
        f"{label} must be an integer inside [{minimum}, {maximum}]",
    )
    return value


def required_finite_number(
    value: Any,
    label: str,
    *,
    minimum: float,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
) -> float:
    require(
        isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value),
        f"{label} must be a finite number",
    )
    numeric = float(value)
    lower_ok = numeric >= minimum if minimum_inclusive else numeric > minimum
    upper_ok = maximum is None or numeric <= maximum
    interval = "[" if minimum_inclusive else "("
    upper = "inf" if maximum is None else str(maximum)
    require(lower_ok and upper_ok, f"{label} must be inside {interval}{minimum}, {upper}]")
    return numeric


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DiffusionCleanPlateBatchError(f"cannot read JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def resolve_asset(value: Any, *, manifest_path: Path, label: str) -> ResolvedAsset:
    require(isinstance(value, dict), f"{label} must be an asset object")
    path_value = value.get("path")
    require(isinstance(path_value, str) and path_value, f"{label}.path is missing")
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    require(path.is_file(), f"{label} does not exist: {path}")
    expected_sha = value.get("sha256")
    require(
        isinstance(expected_sha, str)
        and len(expected_sha) == 64
        and all(character in "0123456789abcdef" for character in expected_sha),
        f"{label}.sha256 must be a lowercase SHA-256 digest",
    )
    actual_sha, actual_size = sha256_file(path)
    require(
        actual_sha == expected_sha,
        f"{label} SHA-256 mismatch: {actual_sha} != {expected_sha}",
    )
    expected_size = value.get("size_bytes", value.get("bytes"))
    if expected_size is not None:
        require(
            isinstance(expected_size, int)
            and not isinstance(expected_size, bool)
            and expected_size >= 0,
            f"{label}.size_bytes must be a non-negative integer",
        )
        require(actual_size == expected_size, f"{label}.size_bytes mismatch")
    return ResolvedAsset(path=path, sha256=actual_sha, size_bytes=actual_size)


def load_rgb(path: Path, *, label: str) -> np.ndarray:
    try:
        with Image.open(path) as image:
            image.load()
            require(image.mode == "RGB", f"{label} mode must be RGB, found {image.mode!r}")
            return np.array(image, dtype=np.uint8, copy=True)
    except OSError as exc:
        raise DiffusionCleanPlateBatchError(f"cannot decode {label}: {path}") from exc


def load_binary_mask(path: Path, *, label: str) -> np.ndarray:
    try:
        with Image.open(path) as image:
            image.load()
            require(image.mode in {"1", "L"}, f"{label} mode must be 1 or L")
            values = np.asarray(image)
    except OSError as exc:
        raise DiffusionCleanPlateBatchError(f"cannot decode {label}: {path}") from exc
    require(values.ndim == 2, f"{label} must be single-channel")
    unique_values = {int(value) for value in np.unique(values)}
    require(unique_values <= {0, 1, 255}, f"{label} must be binary 0/1 or 0/255")
    mask = values.astype(bool)
    require(bool(mask.any()), f"{label} must contain at least one ownership pixel")
    return mask


def model_file_records(model_dir: Path) -> dict[str, dict[str, Any]]:
    require(model_dir.is_dir(), f"model.local_path must be a fixed local directory: {model_dir}")
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(model_dir.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(model_dir)
        if ".cache" in relative.parts or not path.is_file():
            continue
        resolved = path.resolve()
        require(resolved.is_file(), f"model file does not resolve locally: {path}")
        digest, size = sha256_file(resolved)
        records[relative.as_posix()] = {
            "path": str(path),
            "resolved_path": str(resolved),
            "sha256": digest,
            "size_bytes": size,
        }
    require(bool(records), f"model.local_path contains no local model files: {model_dir}")
    return records


def model_file_manifest_sha256(records: dict[str, dict[str, Any]]) -> str:
    identity = {
        relative_path: {
            "sha256": record["sha256"],
            "size_bytes": record["size_bytes"],
        }
        for relative_path, record in sorted(records.items())
    }
    return canonical_sha256(identity)


def validate_config(value: Any, *, manifest_path: Path) -> tuple[dict[str, Any], Path]:
    require(isinstance(value, dict), "config must be an object")
    prompt = required_text(value.get("prompt"), "config.prompt")
    negative_prompt = required_text(
        value.get("negative_prompt"),
        "config.negative_prompt",
        allow_empty=True,
    )
    context_radius = required_int(
        value.get("context_dilation_pixels"),
        "config.context_dilation_pixels",
        minimum=1,
        maximum=MAX_MASK_RADIUS,
    )
    collar_radius = required_int(
        value.get("boundary_outer_collar_pixels"),
        "config.boundary_outer_collar_pixels",
        minimum=1,
        maximum=MAX_MASK_RADIUS,
    )
    require(
        collar_radius <= context_radius,
        "config.boundary_outer_collar_pixels cannot exceed context_dilation_pixels",
    )

    seed_policy = value.get("seed_policy")
    require(isinstance(seed_policy, dict), "config.seed_policy must be an object")
    require(
        seed_policy.get("kind") == SEED_POLICY,
        f"config.seed_policy.kind must equal {SEED_POLICY!r}",
    )
    base_seed = required_int(
        seed_policy.get("base_seed"),
        "config.seed_policy.base_seed",
        minimum=0,
        maximum=MAX_SEED,
    )
    stride = required_int(
        seed_policy.get("stride"),
        "config.seed_policy.stride",
        minimum=0,
        maximum=MAX_SEED,
    )

    model = value.get("model")
    require(isinstance(model, dict), "config.model must be an object")
    model_repo = required_text(model.get("repo"), "config.model.repo").strip()
    model_revision = required_text(model.get("revision"), "config.model.revision").strip()
    local_value = required_text(model.get("local_path"), "config.model.local_path")
    model_path = Path(local_value).expanduser()
    if not model_path.is_absolute():
        model_path = manifest_path.parent / model_path
    model_path = model_path.resolve()
    pipeline_class = required_text(model.get("pipeline_class"), "config.model.pipeline_class")
    require(
        pipeline_class == PIPELINE_CLASS,
        f"config.model.pipeline_class must equal {PIPELINE_CLASS!r}",
    )
    require(model.get("local_files_only") is True, "config.model.local_files_only must be true")
    require(model.get("use_safetensors") is True, "config.model.use_safetensors must be true")
    dtype = required_text(model.get("dtype"), "config.model.dtype").strip()
    require(dtype in DTYPES, f"config.model.dtype must be one of: {', '.join(DTYPES)}")
    variant = required_text(model.get("variant"), "config.model.variant").strip()
    device = required_text(model.get("device"), "config.model.device").strip()
    expected_model_file_manifest_sha256 = required_sha256(
        model.get("file_manifest_sha256"),
        "config.model.file_manifest_sha256",
    )

    generation = value.get("generation")
    require(isinstance(generation, dict), "config.generation must be an object")
    steps = required_int(
        generation.get("num_inference_steps"),
        "config.generation.num_inference_steps",
        minimum=1,
        maximum=MAX_INFERENCE_STEPS,
    )
    guidance = required_finite_number(
        generation.get("guidance_scale"),
        "config.generation.guidance_scale",
        minimum=0.0,
    )
    strength = required_finite_number(
        generation.get("strength"),
        "config.generation.strength",
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=False,
    )
    normalized = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "context_dilation_pixels": context_radius,
        "boundary_outer_collar_pixels": collar_radius,
        "seed_policy": {
            "kind": SEED_POLICY,
            "base_seed": base_seed,
            "stride": stride,
        },
        "model": {
            "repo": model_repo,
            "revision": model_revision,
            "local_path": str(model_path),
            "pipeline_class": PIPELINE_CLASS,
            "local_files_only": True,
            "use_safetensors": True,
            "dtype": dtype,
            "variant": variant,
            "device": device,
            "file_manifest_sha256": expected_model_file_manifest_sha256,
        },
        "generation": {
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "strength": strength,
        },
    }
    return normalized, model_path


def euclidean_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    require(mask.ndim == 2, "ownership mask must be two-dimensional")
    require(bool(mask.any()), "ownership mask must contain at least one pixel")
    required_int(radius, "context dilation radius", minimum=1, maximum=MAX_MASK_RADIUS)
    distance = ndimage.distance_transform_edt(~mask)
    context = mask | (distance <= float(radius))
    require(bool(np.all(context[mask])), "context mask does not include ownership core")
    return context


def resolve_batch_input(manifest_path: str | Path) -> BatchInput:
    path = Path(manifest_path).expanduser().resolve()
    require(path.is_file(), f"manifest must be a regular file: {path}")
    manifest_sha, manifest_size = sha256_file(path)
    manifest = read_json(path)
    require(manifest.get("schema_version") == SCHEMA_VERSION, "unsupported schema_version")
    require(manifest.get("kind") == INPUT_KIND, f"manifest kind must equal {INPUT_KIND!r}")
    config, model_path = validate_config(manifest.get("config"), manifest_path=path)

    raw_frames = manifest.get("frame_records")
    require(isinstance(raw_frames, list) and raw_frames, "frame_records must be a non-empty list")
    frames: list[FrameInput] = []
    frame_ids: set[str] = set()
    for position, raw_frame in enumerate(raw_frames):
        require(isinstance(raw_frame, dict), f"frame_records[{position}] must be an object")
        sequence_index = required_int(
            raw_frame.get("sequence_index"),
            f"frame_records[{position}].sequence_index",
            minimum=0,
            maximum=len(raw_frames) - 1,
        )
        require(
            sequence_index == position,
            "frame_records must be ordered with contiguous sequence_index values starting at zero",
        )
        frame_id = required_text(raw_frame.get("frame_id"), f"frame_records[{position}].frame_id")
        require(frame_id not in frame_ids, f"duplicate frame_id: {frame_id}")
        frame_ids.add(frame_id)
        seed = config["seed_policy"]["base_seed"] + sequence_index * config["seed_policy"]["stride"]
        require(seed <= MAX_SEED, f"seed overflow for frame {frame_id}")
        source_asset = resolve_asset(
            raw_frame.get("source_rgb"),
            manifest_path=path,
            label=f"frame {frame_id} source_rgb",
        )
        ownership_asset = resolve_asset(
            raw_frame.get("ownership_mask"),
            manifest_path=path,
            label=f"frame {frame_id} ownership_mask",
        )
        ownership_source_sha256 = required_sha256(
            raw_frame.get("ownership_mask_source_rgb_sha256"),
            f"frame {frame_id} ownership_mask_source_rgb_sha256",
        )
        require(
            ownership_source_sha256 == source_asset.sha256,
            f"frame {frame_id} ownership mask is not bound to exact source RGB",
        )
        guide_value = raw_frame.get("guide_rgb")
        guide_asset = (
            resolve_asset(
                guide_value,
                manifest_path=path,
                label=f"frame {frame_id} guide_rgb",
            )
            if guide_value is not None
            else None
        )
        source_rgb = load_rgb(source_asset.path, label=f"frame {frame_id} source_rgb")
        ownership_mask = load_binary_mask(
            ownership_asset.path,
            label=f"frame {frame_id} ownership_mask",
        )
        require(
            ownership_mask.shape == source_rgb.shape[:2],
            f"frame {frame_id} ownership mask dimensions differ from source RGB",
        )
        guide_rgb = (
            load_rgb(guide_asset.path, label=f"frame {frame_id} guide_rgb")
            if guide_asset is not None
            else None
        )
        if guide_rgb is not None:
            require(
                guide_rgb.shape == source_rgb.shape,
                f"frame {frame_id} guide RGB dimensions differ from source RGB",
            )
            guide_source_sha256 = required_sha256(
                raw_frame.get("guide_source_rgb_sha256"),
                f"frame {frame_id} guide_source_rgb_sha256",
            )
            require(
                guide_source_sha256 == source_asset.sha256,
                f"frame {frame_id} guide RGB is not bound to exact source RGB",
            )
        else:
            require(
                raw_frame.get("guide_source_rgb_sha256") is None,
                f"frame {frame_id} guide_source_rgb_sha256 requires guide_rgb",
            )

        protected_value = raw_frame.get("protected_mask")
        protected_asset = (
            resolve_asset(
                protected_value,
                manifest_path=path,
                label=f"frame {frame_id} protected_mask",
            )
            if protected_value is not None
            else None
        )
        protected_mask = (
            load_binary_mask(protected_asset.path, label=f"frame {frame_id} protected_mask")
            if protected_asset is not None
            else np.zeros(ownership_mask.shape, dtype=bool)
        )
        require(
            protected_mask.shape == ownership_mask.shape,
            f"frame {frame_id} protected mask dimensions differ from source RGB",
        )
        if protected_asset is not None:
            protected_source_sha256 = required_sha256(
                raw_frame.get("protected_mask_source_rgb_sha256"),
                f"frame {frame_id} protected_mask_source_rgb_sha256",
            )
            require(
                protected_source_sha256 == source_asset.sha256,
                f"frame {frame_id} protected mask is not bound to exact source RGB",
            )
        else:
            require(
                raw_frame.get("protected_mask_source_rgb_sha256") is None,
                f"frame {frame_id} protected_mask_source_rgb_sha256 requires protected_mask",
            )
        frames.append(
            FrameInput(
                sequence_index=sequence_index,
                frame_id=frame_id,
                seed=seed,
                source_asset=source_asset,
                ownership_asset=ownership_asset,
                guide_asset=guide_asset,
                protected_asset=protected_asset,
                source_rgb=source_rgb,
                ownership_mask=ownership_mask,
                guide_rgb=guide_rgb,
                protected_mask=protected_mask,
            )
        )

    model_files = model_file_records(model_path)
    actual_model_file_manifest_sha256 = model_file_manifest_sha256(model_files)
    require(
        actual_model_file_manifest_sha256 == config["model"]["file_manifest_sha256"],
        "model file manifest SHA-256 differs from the pinned manifest value",
    )
    return BatchInput(
        manifest_path=path,
        manifest_sha256=manifest_sha,
        manifest_size_bytes=manifest_size,
        config=config,
        model_path=model_path,
        model_files=model_files,
        frames=tuple(frames),
    )


def load_local_sdxl_pipeline(
    *,
    model_dir: Path,
    device: str,
    dtype: str,
    variant: str,
    use_safetensors: bool,
) -> LoadedPipeline:
    """Load the fixed local SDXL inpaint model; heavy imports remain lazy."""

    try:
        import diffusers
        import torch
        from diffusers import StableDiffusionXLInpaintPipeline
    except ImportError as exc:  # pragma: no cover - optional provider runtime
        raise DiffusionCleanPlateBatchError(
            "real inference requires torch and diffusers with SDXL inpaint support"
        ) from exc
    require(dtype in DTYPES, f"dtype must be one of: {', '.join(DTYPES)}")
    require(use_safetensors is True, "use_safetensors must remain true")
    torch_dtypes = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
        str(model_dir),
        torch_dtype=torch_dtypes[dtype],
        local_files_only=True,
        variant=variant,
        use_safetensors=True,
    )
    pipeline.to(device)
    if hasattr(pipeline, "set_progress_bar_config"):
        pipeline.set_progress_bar_config(disable=False)
    return LoadedPipeline(
        pipeline=pipeline,
        runtime={
            "pipeline_class": pipeline.__class__.__name__,
            "torch_version": torch.__version__,
            "diffusers_version": diffusers.__version__,
            "device": device,
            "dtype": dtype,
            "local_files_only": True,
            "variant": variant,
            "use_safetensors": True,
        },
    )


def make_torch_generator(seed: int) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional provider runtime
        raise DiffusionCleanPlateBatchError("real inference requires torch") from exc
    return torch.Generator(device="cpu").manual_seed(seed)


def verify_bound_inputs_unchanged(batch: BatchInput) -> None:
    manifest_sha, manifest_size = sha256_file(batch.manifest_path)
    require(
        (manifest_sha, manifest_size) == (batch.manifest_sha256, batch.manifest_size_bytes),
        "input manifest changed during batch inference",
    )
    for frame in batch.frames:
        assets = [frame.source_asset, frame.ownership_asset]
        if frame.guide_asset is not None:
            assets.append(frame.guide_asset)
        if frame.protected_asset is not None:
            assets.append(frame.protected_asset)
        for asset in assets:
            digest, size = sha256_file(asset.path)
            require(
                (digest, size) == (asset.sha256, asset.size_bytes),
                f"bound input changed during batch inference: {asset.path}",
            )


def prepare_output_destination(output_dir: str | Path, *, model_path: Path) -> Path:
    candidate = Path(output_dir).expanduser()
    require(not candidate.is_symlink(), f"output_dir cannot be a symlink: {candidate}")
    destination = candidate.resolve()
    require(
        not destination.exists(),
        f"output_dir already exists; refusing overwrite: {destination}",
    )
    require(
        destination != model_path and model_path not in destination.parents,
        "output_dir cannot be inside model.local_path",
    )
    return destination


def run_frame(
    frame: FrameInput,
    *,
    batch: BatchInput,
    pipeline: Any,
    generator_factory: GeneratorFactory,
    staging: Path,
) -> dict[str, Any]:
    config = batch.config
    context_envelope = euclidean_dilate(
        frame.ownership_mask,
        config["context_dilation_pixels"],
    )
    context = frame.ownership_mask | (context_envelope & ~frame.protected_mask)
    source_image = Image.fromarray(frame.source_rgb)
    inference_rgb = frame.source_rgb.copy()
    if frame.guide_rgb is not None:
        inference_rgb[context] = frame.guide_rgb[context]
    inference_outside_context_exact = bool(
        np.array_equal(inference_rgb[~context], frame.source_rgb[~context])
    )
    require(
        inference_outside_context_exact,
        f"inference input escaped context for frame {frame.frame_id}",
    )
    inference_image = Image.fromarray(inference_rgb)
    context_image = Image.fromarray(context.astype(np.uint8) * 255)
    started = time.perf_counter()
    result = pipeline(
        prompt=config["prompt"],
        negative_prompt=config["negative_prompt"],
        image=inference_image,
        mask_image=context_image,
        generator=generator_factory(frame.seed),
        num_inference_steps=config["generation"]["num_inference_steps"],
        guidance_scale=config["generation"]["guidance_scale"],
        strength=config["generation"]["strength"],
        height=source_image.height,
        width=source_image.width,
    )
    elapsed = time.perf_counter() - started
    images = getattr(result, "images", None)
    require(
        isinstance(images, list | tuple) and len(images) == 1,
        f"pipeline must return exactly one image for frame {frame.frame_id}",
    )
    candidate_image = images[0]
    require(
        isinstance(candidate_image, Image.Image),
        f"pipeline candidate must be a Pillow image for frame {frame.frame_id}",
    )
    require(
        candidate_image.size == source_image.size,
        f"pipeline candidate dimensions differ for frame {frame.frame_id}",
    )
    provider_rgb = np.asarray(candidate_image.convert("RGB"), dtype=np.uint8)
    provider_delta_from_source = np.any(provider_rgb != frame.source_rgb, axis=2)
    provider_delta_from_inference = np.any(provider_rgb != inference_rgb, axis=2)
    changed_inside_context_from_inference = int(
        np.count_nonzero(provider_delta_from_inference & context)
    )
    require(
        changed_inside_context_from_inference > 0,
        f"pipeline candidate has no RGB delta from its input inside context for {frame.frame_id}",
    )
    changed_inside_ownership_from_source = int(
        np.count_nonzero(provider_delta_from_source & frame.ownership_mask)
    )
    require(
        changed_inside_ownership_from_source > 0,
        f"pipeline candidate has no source RGB delta inside ownership core for {frame.frame_id}",
    )

    raw_candidate = frame.source_rgb.copy()
    raw_candidate[context] = provider_rgb[context]
    raw_outside_exact = bool(np.array_equal(raw_candidate[~context], frame.source_rgb[~context]))
    require(raw_outside_exact, f"raw candidate escaped context for frame {frame.frame_id}")

    bands = build_soft_bands(
        frame.ownership_mask,
        frame.protected_mask,
        outer_collar_pixels=config["boundary_outer_collar_pixels"],
    )
    require(
        not bool(np.any(bands["editable"] & ~context)),
        f"boundary editable mask escaped generation context for frame {frame.frame_id}",
    )
    composite = blend_candidate(frame.source_rgb, raw_candidate, bands)
    final_outside_exact = bool(
        np.array_equal(composite[bands["outside"]], frame.source_rgb[bands["outside"]])
    )
    core_exact = bool(
        np.array_equal(
            composite[frame.ownership_mask],
            raw_candidate[frame.ownership_mask],
        )
    )
    require(final_outside_exact, f"final composite escaped editable mask for {frame.frame_id}")
    require(core_exact, f"final ownership core differs from raw candidate for {frame.frame_id}")

    frame_root = staging / "frames" / f"{frame.sequence_index:04d}"
    raw_path = frame_root / "raw_candidate.png"
    composite_path = frame_root / "composite.png"
    ownership_path = frame_root / "masks" / "ownership.png"
    context_path = frame_root / "masks" / "context.png"
    collar_path = frame_root / "masks" / "collar.png"
    editable_path = frame_root / "masks" / "editable.png"
    protected_path = frame_root / "masks" / "protected.png"
    protected_collar_path = frame_root / "masks" / "protected_collar.png"
    weight_path = frame_root / "masks" / "blend_weight.png"
    save_png(raw_path, raw_candidate)
    save_png(composite_path, composite)
    save_png(ownership_path, frame.ownership_mask.astype(np.uint8) * 255)
    save_png(context_path, context.astype(np.uint8) * 255)
    save_png(collar_path, bands["collar"].astype(np.uint8) * 255)
    save_png(editable_path, bands["editable"].astype(np.uint8) * 255)
    save_png(protected_path, bands["protected"].astype(np.uint8) * 255)
    save_png(protected_collar_path, bands["protected_collar"].astype(np.uint8) * 255)
    blend_weight = np.clip(np.rint(bands["weight"] * 255.0), 0, 255).astype(np.uint8)
    save_png(weight_path, blend_weight)

    artifacts = {
        "raw_candidate_rgb": output_artifact(raw_path, root=staging),
        "composite_rgb": output_artifact(composite_path, root=staging),
        "ownership_mask": output_artifact(ownership_path, root=staging),
        "context_mask": output_artifact(context_path, root=staging),
        "collar_mask": output_artifact(collar_path, root=staging),
        "editable_mask": output_artifact(editable_path, root=staging),
        "protected_mask": output_artifact(protected_path, root=staging),
        "protected_collar_mask": output_artifact(protected_collar_path, root=staging),
        "blend_weight": output_artifact(weight_path, root=staging),
    }
    frame_receipt = {
        "schema_version": SCHEMA_VERSION,
        "kind": FRAME_RECEIPT_KIND,
        "status": "generated_candidate_pending_review",
        "promotion_allowed": False,
        "sequence_index": frame.sequence_index,
        "frame_id": frame.frame_id,
        "seed": frame.seed,
        "seed_policy": config["seed_policy"],
        "batch_contract": {
            "input_manifest_sha256": batch.manifest_sha256,
            "model_repo": config["model"]["repo"],
            "model_revision": config["model"]["revision"],
            "model_file_manifest_sha256": model_file_manifest_sha256(batch.model_files),
        },
        "generation": {
            "prompt_sha256": sha256_text(config["prompt"]),
            "negative_prompt_sha256": sha256_text(config["negative_prompt"]),
            **config["generation"],
            "elapsed_seconds": elapsed,
            "inference_image": (
                "source_with_guide_inside_context" if frame.guide_rgb is not None else "source_rgb"
            ),
        },
        "inputs": {
            "source_rgb": input_artifact(frame.source_asset),
            "ownership_mask": input_artifact(frame.ownership_asset),
            "ownership_mask_source_rgb_sha256": frame.source_asset.sha256,
            "guide_rgb": input_artifact(frame.guide_asset) if frame.guide_asset else None,
            "guide_source_rgb_sha256": (
                frame.source_asset.sha256 if frame.guide_asset is not None else None
            ),
            "protected_mask": (
                input_artifact(frame.protected_asset) if frame.protected_asset else None
            ),
            "protected_mask_source_rgb_sha256": (
                frame.source_asset.sha256 if frame.protected_asset is not None else None
            ),
        },
        "masks": {
            "dilation_metric": "euclidean_distance_transform",
            "context_dilation_pixels": config["context_dilation_pixels"],
            "boundary_outer_collar_pixels": config["boundary_outer_collar_pixels"],
            "frame_pixels": int(context.size),
            "ownership_pixels": int(frame.ownership_mask.sum()),
            "context_pixels": int(context.sum()),
            "context_additional_pixels": int((context & ~frame.ownership_mask).sum()),
            "context_excluded_protected_noncore_pixels": int(
                (context_envelope & frame.protected_mask & ~frame.ownership_mask).sum()
            ),
            "collar_pixels": int(bands["collar"].sum()),
            "protected_pixels": int(frame.protected_mask.sum()),
            "protected_collar_pixels": int(bands["protected_collar"].sum()),
            "editable_pixels": int(bands["editable"].sum()),
        },
        "exactness": {
            "context_includes_ownership_core": bool(np.all(context[frame.ownership_mask])),
            "output_ownership_mask_pixel_exact": True,
            "inference_outside_context_rgb_exact": inference_outside_context_exact,
            "inference_outside_context_changed_pixels": 0,
            "guide_outside_context_changed_from_source_pixels": (
                int(
                    np.count_nonzero(np.any(frame.guide_rgb != frame.source_rgb, axis=2) & ~context)
                )
                if frame.guide_rgb is not None
                else 0
            ),
            "raw_candidate_outside_context_rgb_exact": raw_outside_exact,
            "raw_candidate_outside_context_changed_pixels": 0,
            "final_outside_editable_rgb_exact": final_outside_exact,
            "final_outside_editable_changed_pixels": 0,
            "final_ownership_core_equals_raw_candidate": core_exact,
            "protected_outer_collar_rgb_exact": bool(
                np.array_equal(
                    composite[bands["protected_collar"]],
                    frame.source_rgb[bands["protected_collar"]],
                )
            ),
            "provider_inside_context_changed_from_inference_pixels": (
                changed_inside_context_from_inference
            ),
            "provider_inside_ownership_changed_from_source_pixels": (
                changed_inside_ownership_from_source
            ),
            "provider_outside_context_changed_from_source_pixels": int(
                np.count_nonzero(provider_delta_from_source & ~context)
            ),
        },
        "outputs": artifacts,
        "review": {
            "automatic_accept": False,
            "human_review_required": True,
            "published": False,
        },
    }
    receipt_path = frame_root / "frame_receipt.json"
    atomic_write_json(receipt_path, frame_receipt)
    return {
        "sequence_index": frame.sequence_index,
        "frame_id": frame.frame_id,
        "seed": frame.seed,
        "inputs": frame_receipt["inputs"],
        "receipt": output_artifact(receipt_path, root=staging),
        "outputs": artifacts,
        "exactness": frame_receipt["exactness"],
        "masks": frame_receipt["masks"],
    }


def run_diffusion_clean_plate_batch(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    created_at: datetime | None = None,
    pipeline_loader: PipelineLoader | None = None,
    generator_factory: GeneratorFactory | None = None,
) -> dict[str, Any]:
    """Run all bound frames sequentially through one loaded SDXL pipeline."""

    batch = resolve_batch_input(manifest_path)
    destination = prepare_output_destination(output_dir, model_path=batch.model_path)
    timestamp = created_at or datetime.now(timezone.utc)  # noqa: UP017 - remote Python 3.10.
    require(
        timestamp.tzinfo is not None and timestamp.utcoffset() is not None,
        "created_at must include timezone information",
    )
    model = batch.config["model"]
    loader = pipeline_loader or load_local_sdxl_pipeline
    loaded = loader(
        model_dir=batch.model_path,
        device=model["device"],
        dtype=model["dtype"],
        variant=model["variant"],
        use_safetensors=True,
    )
    require(isinstance(loaded, LoadedPipeline), "pipeline_loader must return LoadedPipeline")
    make_generator = generator_factory or make_torch_generator

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )
    batch_started = time.perf_counter()
    try:
        frame_records = [
            run_frame(
                frame,
                batch=batch,
                pipeline=loaded.pipeline,
                generator_factory=make_generator,
                staging=staging,
            )
            for frame in batch.frames
        ]
        verify_bound_inputs_unchanged(batch)
        require(
            not destination.exists(),
            f"output_dir appeared during inference; refusing overwrite: {destination}",
        )
        frame_receipt_hashes = [record["receipt"]["sha256"] for record in frame_records]
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "kind": RECEIPT_KIND,
            "status": STATUS,
            "created_at": timestamp.isoformat(),
            "provider": PROVIDER,
            "promotion_allowed": False,
            "input_manifest": {
                "path": str(batch.manifest_path),
                "sha256": batch.manifest_sha256,
                "size_bytes": batch.manifest_size_bytes,
            },
            "config": {
                "prompt": {
                    "text": batch.config["prompt"],
                    "sha256": sha256_text(batch.config["prompt"]),
                },
                "negative_prompt": {
                    "text": batch.config["negative_prompt"],
                    "sha256": sha256_text(batch.config["negative_prompt"]),
                },
                "context_dilation_pixels": batch.config["context_dilation_pixels"],
                "boundary_outer_collar_pixels": batch.config["boundary_outer_collar_pixels"],
                "seed_policy": batch.config["seed_policy"],
                "generation": batch.config["generation"],
            },
            "model": {
                **model,
                "files": batch.model_files,
                "runtime_file_records_sha256": canonical_sha256(batch.model_files),
            },
            "runtime": {
                **loaded.runtime,
                "pipeline_load_count": 1,
                "processing_mode": "single_process_ordered_frames",
                "batch_elapsed_seconds": time.perf_counter() - batch_started,
            },
            "frames": frame_records,
            "aggregate": {
                "frame_count": len(frame_records),
                "ordered_frame_ids": [record["frame_id"] for record in frame_records],
                "ordered_seeds": [record["seed"] for record in frame_records],
                "frame_receipt_set_sha256": canonical_sha256(frame_receipt_hashes),
                "all_raw_candidates_outside_context_rgb_exact": all(
                    record["exactness"]["raw_candidate_outside_context_rgb_exact"]
                    for record in frame_records
                ),
                "all_inference_inputs_outside_context_rgb_exact": all(
                    record["exactness"]["inference_outside_context_rgb_exact"]
                    for record in frame_records
                ),
                "all_final_outputs_outside_editable_rgb_exact": all(
                    record["exactness"]["final_outside_editable_rgb_exact"]
                    for record in frame_records
                ),
                "all_ownership_cores_equal_raw_candidates": all(
                    record["exactness"]["final_ownership_core_equals_raw_candidate"]
                    for record in frame_records
                ),
                "all_protected_outer_collars_rgb_exact": all(
                    record["exactness"]["protected_outer_collar_rgb_exact"]
                    for record in frame_records
                ),
            },
            "review": {
                "automatic_accept": False,
                "human_review_required": True,
                "published": False,
            },
        }
        atomic_write_json(staging / "batch_receipt.json", receipt)
        os.replace(staging, destination)
        return receipt
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = run_diffusion_clean_plate_batch(args.manifest, args.output_dir)
    except DiffusionCleanPlateBatchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
