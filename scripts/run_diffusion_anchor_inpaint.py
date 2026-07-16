#!/usr/bin/env python3
"""Generate review-only clean-plate candidates with a local diffusion model.

The runner is deliberately narrow: it changes only a declared residual mask, records
the exact local model and inputs, and never promotes, reviews, or publishes a result.
Torch and Diffusers remain optional until real inference is requested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

STATUS = "generated_candidates_pending_review"
PROVIDER = "local_huggingface_diffusers"
MAX_SEEDS = 3
MAX_INFERENCE_STEPS = 1_000
PIPELINE_KINDS = ("auto", "sdxl")
DTYPES = ("float16", "bfloat16", "float32")


class DiffusionAnchorInpaintError(ValueError):
    """Raised when a run cannot produce trustworthy review candidates."""


@dataclass(frozen=True)
class LoadedPipeline:
    pipeline: Any
    runtime: dict[str, Any]


PipelineLoader = Callable[..., LoadedPipeline]
GeneratorFactory = Callable[[int], Any]


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
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def artifact_record(path: Path) -> dict[str, Any]:
    sha256, size = sha256_file(path)
    return {"path": str(path), "sha256": sha256, "size_bytes": size}


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    temporary.write_text(payload + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_save_png(rgb: np.ndarray, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    Image.fromarray(rgb).save(temporary, format="PNG")
    os.replace(temporary, path)


def required_text(value: str, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise DiffusionAnchorInpaintError(f"{field} must be a string")
    if not allow_empty and not value.strip():
        raise DiffusionAnchorInpaintError(f"{field} must be non-empty")
    return value


def validate_seeds(seeds: Sequence[int]) -> list[int]:
    values = list(seeds)
    if not 1 <= len(values) <= MAX_SEEDS:
        raise DiffusionAnchorInpaintError(f"provide between 1 and {MAX_SEEDS} explicit seeds")
    if any(type(value) is not int or not 0 <= value <= (2**63 - 1) for value in values):
        raise DiffusionAnchorInpaintError("seeds must be non-negative signed 64-bit integers")
    expected = list(range(values[0], values[0] + len(values)))
    if values != expected:
        raise DiffusionAnchorInpaintError(
            "seeds must be unique, ascending, and contiguous in the supplied order"
        )
    return values


def validate_generation_parameters(
    *,
    num_inference_steps: int,
    guidance_scale: float,
    strength: float,
) -> tuple[int, float, float]:
    if type(num_inference_steps) is not int or not 1 <= num_inference_steps <= MAX_INFERENCE_STEPS:
        raise DiffusionAnchorInpaintError(
            f"num_inference_steps must be an integer inside [1, {MAX_INFERENCE_STEPS}]"
        )
    numeric = {
        "guidance_scale": guidance_scale,
        "strength": strength,
    }
    for field, value in numeric.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise DiffusionAnchorInpaintError(f"{field} must be a finite number")
    if guidance_scale < 0:
        raise DiffusionAnchorInpaintError("guidance_scale must be non-negative")
    if not 0 < strength <= 1:
        raise DiffusionAnchorInpaintError("strength must be inside (0, 1]")
    return num_inference_steps, float(guidance_scale), float(strength)


def load_source_and_mask(
    source_rgb_path: str | Path,
    residual_mask_path: str | Path,
) -> tuple[Path, Path, np.ndarray, np.ndarray]:
    source_path = Path(source_rgb_path).expanduser().resolve()
    mask_path = Path(residual_mask_path).expanduser().resolve()
    if not source_path.is_file():
        raise DiffusionAnchorInpaintError(f"source RGB must be a regular file: {source_path}")
    if not mask_path.is_file():
        raise DiffusionAnchorInpaintError(f"residual mask must be a regular file: {mask_path}")

    try:
        with Image.open(source_path) as source_image:
            source_image.load()
            if source_image.mode != "RGB":
                raise DiffusionAnchorInpaintError(
                    f"source image mode must be RGB, found {source_image.mode!r}"
                )
            source_rgb = np.array(source_image, dtype=np.uint8, copy=True)
        with Image.open(mask_path) as mask_image:
            mask_image.load()
            if mask_image.mode not in {"1", "L"}:
                raise DiffusionAnchorInpaintError(
                    f"residual mask mode must be 1 or L, found {mask_image.mode!r}"
                )
            mask_values = np.asarray(mask_image)
    except OSError as exc:
        raise DiffusionAnchorInpaintError(f"cannot decode source or residual mask: {exc}") from exc

    if source_rgb.ndim != 3 or source_rgb.shape[2] != 3:
        raise DiffusionAnchorInpaintError("source RGB must have exactly three channels")
    if mask_values.ndim != 2:
        raise DiffusionAnchorInpaintError("residual mask must be single-channel")
    if mask_values.shape != source_rgb.shape[:2]:
        raise DiffusionAnchorInpaintError(
            "source RGB and residual mask dimensions must match exactly"
        )
    unique_values = {int(value) for value in np.unique(mask_values)}
    if not unique_values <= {0, 1, 255}:
        raise DiffusionAnchorInpaintError(
            "residual mask must be binary with only 0/1 or 0/255 values"
        )
    residual_mask = mask_values.astype(bool)
    if not residual_mask.any():
        raise DiffusionAnchorInpaintError("residual mask must contain at least one masked pixel")
    return source_path, mask_path, source_rgb, residual_mask


def model_file_records(model_dir: str | Path) -> tuple[Path, dict[str, dict[str, Any]]]:
    model_path = Path(model_dir).expanduser().resolve()
    if not model_path.is_dir():
        raise DiffusionAnchorInpaintError(
            f"model_dir must be a fixed local directory: {model_path}"
        )
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(model_path.rglob("*"), key=lambda item: item.as_posix()):
        relative_path = path.relative_to(model_path)
        if ".cache" in relative_path.parts or not path.is_file():
            continue
        resolved = path.resolve()
        if not resolved.is_file():
            raise DiffusionAnchorInpaintError(f"model file does not resolve locally: {path}")
        sha256, size = sha256_file(resolved)
        key = relative_path.as_posix()
        records[key] = {
            "path": str(path),
            "resolved_path": str(resolved),
            "sha256": sha256,
            "size_bytes": size,
        }
    if not records:
        raise DiffusionAnchorInpaintError(f"model_dir contains no local model files: {model_path}")
    return model_path, records


def load_local_inpaint_pipeline(
    *,
    model_dir: Path,
    pipeline_kind: str,
    device: str,
    dtype: str,
    variant: str,
    use_safetensors: bool,
) -> LoadedPipeline:
    """Load one local-only Diffusers pipeline; heavy dependencies stay lazy."""

    try:
        import diffusers
        import torch
        from diffusers import AutoPipelineForInpainting, StableDiffusionXLInpaintPipeline
    except ImportError as exc:  # pragma: no cover - optional provider runtime
        raise DiffusionAnchorInpaintError(
            "real inference requires torch and diffusers with inpainting pipeline support"
        ) from exc

    classes = {
        "auto": AutoPipelineForInpainting,
        "sdxl": StableDiffusionXLInpaintPipeline,
    }
    if pipeline_kind not in classes:
        raise DiffusionAnchorInpaintError(
            f"pipeline_kind must be one of: {', '.join(PIPELINE_KINDS)}"
        )
    torch_dtypes = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype not in torch_dtypes:
        raise DiffusionAnchorInpaintError(f"dtype must be one of: {', '.join(DTYPES)}")
    variant = required_text(variant, "variant").strip()
    if use_safetensors is not True:
        raise DiffusionAnchorInpaintError("use_safetensors must remain true")
    pipeline = classes[pipeline_kind].from_pretrained(
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
    """Build a deterministic CPU generator without importing Torch at module import."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional provider runtime
        raise DiffusionAnchorInpaintError("real inference requires torch") from exc
    return torch.Generator(device="cpu").manual_seed(seed)


def run_diffusion_anchor_inpaint(
    *,
    source_rgb_path: str | Path,
    residual_mask_path: str | Path,
    output_dir: str | Path,
    model_dir: str | Path,
    model_repo: str,
    model_revision: str,
    prompt: str,
    negative_prompt: str,
    seeds: Sequence[int],
    pipeline_kind: str = "auto",
    num_inference_steps: int = 40,
    guidance_scale: float = 7.5,
    strength: float = 1.0,
    device: str = "cuda:0",
    dtype: str = "float16",
    variant: str = "fp16",
    created_at: datetime | None = None,
    pipeline_loader: PipelineLoader | None = None,
    generator_factory: GeneratorFactory | None = None,
) -> dict[str, Any]:
    """Generate masked candidates and write an immutable review-pending receipt."""

    model_repo = required_text(model_repo, "model_repo").strip()
    model_revision = required_text(model_revision, "model_revision").strip()
    prompt = required_text(prompt, "prompt")
    negative_prompt = required_text(
        negative_prompt,
        "negative_prompt",
        allow_empty=True,
    )
    device = required_text(device, "device").strip()
    variant = required_text(variant, "variant").strip()
    if pipeline_kind not in PIPELINE_KINDS:
        raise DiffusionAnchorInpaintError(
            f"pipeline_kind must be one of: {', '.join(PIPELINE_KINDS)}"
        )
    if dtype not in DTYPES:
        raise DiffusionAnchorInpaintError(f"dtype must be one of: {', '.join(DTYPES)}")
    seed_values = validate_seeds(seeds)
    steps, guidance, validated_strength = validate_generation_parameters(
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        strength=strength,
    )
    source_path, mask_path, source_rgb, residual_mask = load_source_and_mask(
        source_rgb_path,
        residual_mask_path,
    )
    model_path, model_files = model_file_records(model_dir)

    destination_candidate = Path(output_dir).expanduser()
    if destination_candidate.is_symlink():
        raise DiffusionAnchorInpaintError(
            f"output_dir cannot be a symlink: {destination_candidate}"
        )
    destination = destination_candidate.resolve()
    if destination.exists() and not destination.is_dir():
        raise DiffusionAnchorInpaintError(f"output_dir is not a directory: {destination}")
    if destination.exists() and any(destination.iterdir()):
        raise DiffusionAnchorInpaintError(f"output_dir must be empty: {destination}")
    if destination == model_path or model_path in destination.parents:
        raise DiffusionAnchorInpaintError("output_dir cannot be inside model_dir")

    timestamp = created_at or datetime.now(UTC)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise DiffusionAnchorInpaintError("created_at must include timezone information")

    loader = pipeline_loader or load_local_inpaint_pipeline
    loaded = loader(
        model_dir=model_path,
        pipeline_kind=pipeline_kind,
        device=device,
        dtype=dtype,
        variant=variant,
        use_safetensors=True,
    )
    if not isinstance(loaded, LoadedPipeline):
        raise DiffusionAnchorInpaintError("pipeline_loader must return LoadedPipeline")
    make_generator = generator_factory or make_torch_generator
    source_image = Image.fromarray(source_rgb)
    mask_image = Image.fromarray(residual_mask.astype(np.uint8) * 255)

    generated: list[tuple[int, np.ndarray, dict[str, Any]]] = []
    for seed in seed_values:
        result = loaded.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=source_image.copy(),
            mask_image=mask_image.copy(),
            generator=make_generator(seed),
            num_inference_steps=steps,
            guidance_scale=guidance,
            strength=validated_strength,
        )
        images = getattr(result, "images", None)
        if not isinstance(images, list | tuple) or len(images) != 1:
            raise DiffusionAnchorInpaintError(
                f"pipeline must return exactly one candidate image for seed {seed}"
            )
        candidate_image = images[0]
        if not isinstance(candidate_image, Image.Image):
            raise DiffusionAnchorInpaintError(
                f"pipeline candidate for seed {seed} must be a Pillow image"
            )
        if candidate_image.size != source_image.size:
            raise DiffusionAnchorInpaintError(
                f"pipeline candidate size {candidate_image.size} does not match "
                f"source size {source_image.size} for seed {seed}"
            )
        candidate_rgb = np.asarray(candidate_image.convert("RGB"), dtype=np.uint8)
        absolute_delta = np.abs(candidate_rgb.astype(np.int16) - source_rgb.astype(np.int16))
        inside_delta = absolute_delta[residual_mask]
        changed_inside = int(np.count_nonzero(np.any(inside_delta > 0, axis=1)))
        if changed_inside == 0:
            raise DiffusionAnchorInpaintError(
                f"candidate for seed {seed} has no RGB delta inside the residual mask"
            )

        composited = source_rgb.copy()
        composited[residual_mask] = candidate_rgb[residual_mask]
        outside_exact = bool(np.array_equal(composited[~residual_mask], source_rgb[~residual_mask]))
        if not outside_exact:
            raise DiffusionAnchorInpaintError(
                f"outside-mask RGB changed after forced compositing for seed {seed}"
            )
        generated.append(
            (
                seed,
                composited,
                {
                    "masked_pixels": int(np.count_nonzero(residual_mask)),
                    "inside_changed_pixels": changed_inside,
                    "inside_mean_absolute_rgb_delta": float(inside_delta.mean()),
                    "inside_max_absolute_rgb_delta": int(inside_delta.max()),
                    "outside_changed_pixels": 0,
                    "outside_rgb_exact": True,
                },
            )
        )

    destination.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, Any]] = []
    for seed, composited, metrics in generated:
        path = destination / f"candidate_seed_{seed}.png"
        atomic_save_png(composited, path)
        outputs.append(
            {
                "seed": seed,
                "review_status": "pending_review",
                "artifact": artifact_record(path),
                "metrics": metrics,
            }
        )

    source_record = artifact_record(source_path)
    source_record.update(
        {"mode": "RGB", "width": source_rgb.shape[1], "height": source_rgb.shape[0]}
    )
    mask_record = artifact_record(mask_path)
    mask_record.update(
        {
            "interpretation": "binary_residual_mask_nonzero_is_generate",
            "masked_pixels": int(np.count_nonzero(residual_mask)),
            "total_pixels": int(residual_mask.size),
        }
    )
    receipt = {
        "schema_version": "1.0",
        "kind": "video2world.diffusion_anchor_inpaint_run",
        "status": STATUS,
        "created_at": timestamp.isoformat(),
        "provider": PROVIDER,
        "promotion_allowed": False,
        "model": {
            "repo": model_repo,
            "revision": model_revision,
            "local_path": str(model_path),
            "local_files_only": True,
            "pipeline_kind": pipeline_kind,
            "variant": variant,
            "use_safetensors": True,
            "files": model_files,
            "file_manifest_sha256": canonical_sha256(model_files),
        },
        "generation": {
            "prompt": {"text": prompt, "sha256": sha256_text(prompt)},
            "negative_prompt": {
                "text": negative_prompt,
                "sha256": sha256_text(negative_prompt),
            },
            "seeds": seed_values,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "strength": validated_strength,
            "loader": {
                "local_files_only": True,
                "variant": variant,
                "use_safetensors": True,
                "torch_dtype": dtype,
            },
        },
        "inputs": {"source_rgb": source_record, "residual_mask": mask_record},
        "candidates": outputs,
        "runtime": loaded.runtime,
        "boundary": {
            "automatic_accept": False,
            "human_review_required": True,
            "vlm_called": False,
            "published": False,
            "outside_mask_policy": "forced_exact_source_rgb_composite",
        },
    }
    receipt_path = destination / "run_receipt.json"
    atomic_write_json(receipt_path, receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-rgb", type=Path, required=True)
    parser.add_argument("--residual-mask", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-repo", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", required=True)
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        required=True,
        help="Explicit ascending contiguous seed; repeat at most three times",
    )
    parser.add_argument("--pipeline-kind", choices=PIPELINE_KINDS, default="auto")
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=DTYPES, default="float16")
    parser.add_argument(
        "--variant",
        default="fp16",
        help="Local Diffusers weight filename variant (default: fp16)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = run_diffusion_anchor_inpaint(
            source_rgb_path=args.source_rgb,
            residual_mask_path=args.residual_mask,
            output_dir=args.output_dir,
            model_dir=args.model_dir,
            model_repo=args.model_repo,
            model_revision=args.model_revision,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            seeds=args.seed,
            pipeline_kind=args.pipeline_kind,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            strength=args.strength,
            device=args.device,
            dtype=args.dtype,
            variant=args.variant,
        )
    except (DiffusionAnchorInpaintError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt": str(Path(args.output_dir).expanduser().resolve() / "run_receipt.json"),
                "candidate_count": len(receipt["candidates"]),
                "promotion_allowed": receipt["promotion_allowed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
