#!/usr/bin/env python3
"""Build a hash-bound diffusion clean-plate batch manifest from R1 inputs.

The legacy ProPainter input manifest contains an original source frame, a
prepared input copy, and one union ownership mask per frame.  This bridge
accepts only records where the prepared RGB is byte-identical to the original
source RGB, validates all declared hashes and dimensions, then attaches one
PBR guide (or an explicitly configured fallback guide) to every frame.

The result is consumed directly by ``run_diffusion_clean_plate_batch.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

OUTPUT_KIND = "video2world.diffusion_clean_plate_batch_input"
SCHEMA_VERSION = 1
MODEL_REPO = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
MODEL_REVISION = "115134f363124c53c7d878647567d04daf26e41e"
PIPELINE_CLASS = "StableDiffusionXLInpaintPipeline"
SEED_POLICY = "base_plus_sequence_index"
MAX_SEED = 2**63 - 1
GUIDE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_PROMPT = (
    "Restore only the background surface hidden behind the removed foreground object. "
    "Preserve the surrounding scene geometry, material texture, illumination, and camera view."
)
DEFAULT_NEGATIVE_PROMPT = (
    "foreground object, replacement object, duplicate object, pillow, floating object, "
    "changed camera, changed scene geometry, blur, low detail, text, watermark"
)


class DiffusionBatchManifestBuildError(ValueError):
    """Raised when an input cannot be bound into a trustworthy batch manifest."""


@dataclass(frozen=True)
class VerifiedImage:
    path: Path
    sha256: str
    size_bytes: int
    dimensions: tuple[int, int]


@dataclass(frozen=True)
class GuideMatch:
    image: VerifiedImage
    source: str
    matched_stem: str


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DiffusionBatchManifestBuildError(message)


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def model_file_records(model_dir: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(model_dir.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(model_dir)
        if ".cache" in relative.parts or not path.is_file():
            continue
        resolved = path.resolve()
        require(resolved.is_file(), f"model file does not resolve locally: {path}")
        digest, size = sha256_file(resolved)
        records[relative.as_posix()] = {
            "sha256": digest,
            "size_bytes": size,
        }
    require(bool(records), f"model path contains no local model files: {model_dir}")
    return records


def require_sha256(value: Any, label: str) -> str:
    require(
        isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None,
        f"{label} must be a lowercase SHA-256 digest",
    )
    return value


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DiffusionBatchManifestBuildError(f"cannot read JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    require(not path.exists(), f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    require(not temporary.exists(), f"temporary output already exists: {temporary}")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def resolve_path(value: Any, *, relative_to: Path, label: str) -> Path:
    require(isinstance(value, str) and bool(value.strip()), f"{label} must be a path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    path = path.resolve()
    require(path.is_file(), f"{label} does not exist: {path}")
    return path


def verified_hash(path: Path, expected: Any, *, label: str) -> tuple[str, int]:
    expected_sha = require_sha256(expected, f"{label} declared SHA-256")
    actual_sha, size = sha256_file(path)
    require(
        actual_sha == expected_sha,
        f"{label} SHA-256 mismatch: {actual_sha} != {expected_sha}",
    )
    return actual_sha, size


def load_rgb_dimensions(path: Path, *, label: str) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            image.load()
            require(image.mode == "RGB", f"{label} mode must be RGB, found {image.mode!r}")
            dimensions = image.size
    except OSError as exc:
        raise DiffusionBatchManifestBuildError(f"cannot decode {label}: {path}") from exc
    require(dimensions[0] > 0 and dimensions[1] > 0, f"{label} dimensions must be positive")
    return dimensions


def load_binary_mask_dimensions(path: Path, *, label: str) -> tuple[tuple[int, int], int]:
    try:
        with Image.open(path) as image:
            image.load()
            require(image.mode in {"1", "L"}, f"{label} mode must be 1 or L")
            luminance = image.convert("L")
            histogram = luminance.histogram()
            dimensions = luminance.size
    except OSError as exc:
        raise DiffusionBatchManifestBuildError(f"cannot decode {label}: {path}") from exc
    nonzero_values = {index for index, count in enumerate(histogram) if count}
    require(nonzero_values <= {0, 1, 255}, f"{label} must be binary 0/1 or 0/255")
    ownership_pixels = histogram[1] + histogram[255]
    require(ownership_pixels > 0, f"{label} must contain at least one ownership pixel")
    return dimensions, ownership_pixels


def asset_record(image: VerifiedImage) -> dict[str, Any]:
    return {
        "path": str(image.path),
        "sha256": image.sha256,
        "size_bytes": image.size_bytes,
        "dimensions": list(image.dimensions),
    }


def verify_rgb_asset(
    value: Any,
    expected_sha: Any,
    *,
    relative_to: Path,
    label: str,
) -> VerifiedImage:
    path = resolve_path(value, relative_to=relative_to, label=label)
    digest, size = verified_hash(path, expected_sha, label=label)
    dimensions = load_rgb_dimensions(path, label=label)
    return VerifiedImage(path=path, sha256=digest, size_bytes=size, dimensions=dimensions)


def verify_mask_asset(
    value: Any,
    expected_sha: Any,
    *,
    relative_to: Path,
    label: str,
) -> tuple[VerifiedImage, int]:
    path = resolve_path(value, relative_to=relative_to, label=label)
    digest, size = verified_hash(path, expected_sha, label=label)
    dimensions, pixels = load_binary_mask_dimensions(path, label=label)
    return (
        VerifiedImage(path=path, sha256=digest, size_bytes=size, dimensions=dimensions),
        pixels,
    )


def index_guide_directory(path: Path, *, label: str) -> dict[str, tuple[Path, ...]]:
    require(path.is_dir(), f"{label} must be a directory: {path}")
    by_stem: dict[str, list[Path]] = {}
    for candidate in sorted(path.iterdir(), key=lambda item: item.name):
        if not candidate.is_file() or candidate.suffix.lower() not in GUIDE_SUFFIXES:
            continue
        by_stem.setdefault(candidate.stem, []).append(candidate.resolve())
    return {stem: tuple(paths) for stem, paths in by_stem.items()}


def guide_stems(frame_id: str, sequence_index: int) -> tuple[str, ...]:
    values = (
        frame_id,
        f"{sequence_index:04d}",
        f"{sequence_index:06d}",
        str(sequence_index),
    )
    return tuple(dict.fromkeys(values))


def match_guide(
    index: dict[str, tuple[Path, ...]],
    *,
    frame_id: str,
    sequence_index: int,
    label: str,
) -> tuple[Path, str] | None:
    matches = [
        (candidate, stem)
        for stem in guide_stems(frame_id, sequence_index)
        for candidate in index.get(stem, ())
    ]
    unique_paths = {candidate for candidate, _ in matches}
    if not matches:
        return None
    require(
        len(unique_paths) == 1,
        f"{label} has ambiguous guides for frame {frame_id}: "
        + ", ".join(str(path) for path in sorted(unique_paths)),
    )
    path = next(iter(unique_paths))
    matched_stem = next(stem for candidate, stem in matches if candidate == path)
    return path, matched_stem


def verify_guide(
    primary_index: dict[str, tuple[Path, ...]],
    fallback_index: dict[str, tuple[Path, ...]] | None,
    *,
    frame_id: str,
    sequence_index: int,
    expected_dimensions: tuple[int, int],
) -> GuideMatch:
    match = match_guide(
        primary_index,
        frame_id=frame_id,
        sequence_index=sequence_index,
        label="PBR guide directory",
    )
    source = "pbr"
    if match is None and fallback_index is not None:
        match = match_guide(
            fallback_index,
            frame_id=frame_id,
            sequence_index=sequence_index,
            label="fallback guide directory",
        )
        source = "fallback"
    require(match is not None, f"missing guide for frame {frame_id}")
    guide_path, matched_stem = match
    dimensions = load_rgb_dimensions(guide_path, label=f"frame {frame_id} guide RGB")
    require(
        dimensions == expected_dimensions,
        f"frame {frame_id} guide RGB dimensions differ from source RGB: "
        f"{dimensions} != {expected_dimensions}",
    )
    digest, size = sha256_file(guide_path)
    return GuideMatch(
        image=VerifiedImage(
            path=guide_path,
            sha256=digest,
            size_bytes=size,
            dimensions=dimensions,
        ),
        source=source,
        matched_stem=matched_stem,
    )


def finite_number(
    value: Any,
    *,
    label: str,
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
    require(lower_ok and upper_ok, f"{label} is outside the supported range")
    return numeric


def positive_int(value: Any, *, label: str, minimum: int = 1) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
        f"{label} must be an integer >= {minimum}",
    )
    return value


def build_config(args: argparse.Namespace, *, output_path: Path) -> dict[str, Any]:
    context = positive_int(args.context_dilation_pixels, label="context dilation pixels")
    collar = positive_int(args.boundary_outer_collar_pixels, label="boundary collar pixels")
    require(collar <= context, "boundary collar pixels cannot exceed context dilation pixels")
    base_seed = positive_int(args.base_seed, label="base seed", minimum=0)
    seed_stride = positive_int(args.seed_stride, label="seed stride", minimum=0)
    require(base_seed <= MAX_SEED, f"base seed must be <= {MAX_SEED}")
    require(seed_stride <= MAX_SEED, f"seed stride must be <= {MAX_SEED}")
    steps = positive_int(args.num_inference_steps, label="num inference steps")
    guidance = finite_number(args.guidance_scale, label="guidance scale", minimum=0.0)
    strength = finite_number(
        args.strength,
        label="strength",
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=False,
    )
    require(isinstance(args.prompt, str) and args.prompt.strip(), "prompt must be non-empty")
    require(isinstance(args.negative_prompt, str), "negative prompt must be a string")
    model_path = Path(args.model_path).expanduser()
    if not model_path.is_absolute():
        model_path = output_path.parent / model_path
    model_path = model_path.resolve()
    require(model_path.is_dir(), f"model path must be a local directory: {model_path}")
    model_manifest_sha256 = canonical_sha256(model_file_records(model_path))
    require(
        isinstance(args.model_repo, str) and args.model_repo.strip(),
        "model repo must be non-empty",
    )
    require(
        isinstance(args.model_revision, str) and args.model_revision.strip(),
        "model revision must be non-empty",
    )
    require(args.dtype in {"float16", "bfloat16", "float32"}, "unsupported dtype")
    require(isinstance(args.variant, str) and args.variant.strip(), "variant must be non-empty")
    require(isinstance(args.device, str) and args.device.strip(), "device must be non-empty")
    return {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "context_dilation_pixels": context,
        "boundary_outer_collar_pixels": collar,
        "seed_policy": {
            "kind": SEED_POLICY,
            "base_seed": base_seed,
            "stride": seed_stride,
        },
        "model": {
            "repo": args.model_repo,
            "revision": args.model_revision,
            "local_path": str(model_path),
            "pipeline_class": PIPELINE_CLASS,
            "local_files_only": True,
            "use_safetensors": True,
            "dtype": args.dtype,
            "variant": args.variant,
            "device": args.device,
            "file_manifest_sha256": model_manifest_sha256,
        },
        "generation": {
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "strength": strength,
        },
    }


def build_manifest(
    args: argparse.Namespace,
    *,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    input_path = Path(args.propainter_manifest).expanduser().resolve()
    require(input_path.is_file(), f"ProPainter input manifest does not exist: {input_path}")
    output_path = Path(args.output).expanduser().resolve()
    require(not output_path.exists(), f"refusing to overwrite existing output: {output_path}")
    source_manifest = read_json(input_path)
    require(source_manifest.get("schema_version") == 1, "unsupported ProPainter schema_version")

    raw_records = source_manifest.get("frame_records")
    require(isinstance(raw_records, list) and raw_records, "frame_records must be non-empty")
    frame_count = source_manifest.get("frame_count")
    require(
        isinstance(frame_count, int)
        and not isinstance(frame_count, bool)
        and frame_count == len(raw_records),
        "frame_count does not match frame_records",
    )
    declared_frame_ids = source_manifest.get("frame_ids")
    require(isinstance(declared_frame_ids, list), "frame_ids must be an array")

    pbr_dir = Path(args.pbr_guide_dir).expanduser().resolve()
    primary_index = index_guide_directory(pbr_dir, label="PBR guide directory")
    fallback_dir = (
        Path(args.fallback_guide_dir).expanduser().resolve()
        if args.fallback_guide_dir is not None
        else None
    )
    fallback_index = (
        index_guide_directory(fallback_dir, label="fallback guide directory")
        if fallback_dir is not None
        else None
    )
    config = build_config(args, output_path=output_path)

    records: list[dict[str, Any]] = []
    observed_ids: list[str] = []
    used_guides: set[Path] = set()
    for position, raw_record in enumerate(raw_records):
        require(isinstance(raw_record, dict), f"frame_records[{position}] must be an object")
        sequence_index = raw_record.get("sequence_index")
        require(
            isinstance(sequence_index, int)
            and not isinstance(sequence_index, bool)
            and sequence_index == position,
            "frame_records must be ordered with contiguous sequence_index values starting at zero",
        )
        frame_id = raw_record.get("frame_id")
        require(
            isinstance(frame_id, str) and bool(frame_id.strip()),
            f"frame_records[{position}].frame_id must be non-empty",
        )
        require(frame_id not in observed_ids, f"duplicate frame_id: {frame_id}")
        observed_ids.append(frame_id)

        source = verify_rgb_asset(
            raw_record.get("source_frame"),
            raw_record.get("source_frame_sha256"),
            relative_to=input_path.parent,
            label=f"frame {frame_id} source_frame",
        )
        prepared = verify_rgb_asset(
            raw_record.get("input_frame"),
            raw_record.get("input_frame_sha256"),
            relative_to=input_path.parent,
            label=f"frame {frame_id} input_frame",
        )
        require(
            source.sha256 == prepared.sha256,
            f"frame {frame_id} prepared input is not exact source RGB",
        )
        require(
            source.dimensions == prepared.dimensions,
            f"frame {frame_id} source and prepared input dimensions differ",
        )
        ownership, ownership_pixels = verify_mask_asset(
            raw_record.get("union_mask"),
            raw_record.get("union_mask_sha256"),
            relative_to=input_path.parent,
            label=f"frame {frame_id} union_mask",
        )
        require(
            ownership.dimensions == prepared.dimensions,
            f"frame {frame_id} ownership mask dimensions differ from source RGB",
        )
        width = raw_record.get("width")
        height = raw_record.get("height")
        require(
            isinstance(width, int)
            and not isinstance(width, bool)
            and isinstance(height, int)
            and not isinstance(height, bool)
            and width == prepared.dimensions[0]
            and height == prepared.dimensions[1],
            f"frame {frame_id} declared width/height differ from decoded source RGB",
        )
        if "union_mask_pixels" in raw_record:
            declared_pixels = raw_record["union_mask_pixels"]
            require(
                isinstance(declared_pixels, int)
                and not isinstance(declared_pixels, bool)
                and declared_pixels == ownership_pixels,
                f"frame {frame_id} union_mask_pixels differs from decoded ownership mask",
            )

        guide = verify_guide(
            primary_index,
            fallback_index,
            frame_id=frame_id,
            sequence_index=position,
            expected_dimensions=prepared.dimensions,
        )
        require(
            guide.image.path not in used_guides,
            f"guide file is reused by multiple frames: {guide.image.path}",
        )
        used_guides.add(guide.image.path)
        records.append(
            {
                "sequence_index": position,
                "frame_id": frame_id,
                "source_rgb": asset_record(prepared),
                "ownership_mask": asset_record(ownership),
                "ownership_mask_source_rgb_sha256": prepared.sha256,
                "guide_rgb": asset_record(guide.image),
                "guide_source_rgb_sha256": prepared.sha256,
                "source_lineage": {
                    "original_source_rgb": asset_record(source),
                    "prepared_input_rgb": asset_record(prepared),
                    "prepared_input_is_byte_exact_source_rgb": True,
                },
                "guide_selection": {
                    "source": guide.source,
                    "matched_stem": guide.matched_stem,
                },
            }
        )

    require(observed_ids == declared_frame_ids, "frame_ids does not match ordered frame_records")
    maximum_seed = (
        config["seed_policy"]["base_seed"] + (len(records) - 1) * config["seed_policy"]["stride"]
    )
    require(maximum_seed <= MAX_SEED, "seed policy overflows on the final frame")
    manifest_sha, manifest_size = sha256_file(input_path)
    created = created_at or datetime.now(timezone.utc)  # noqa: UP017
    require(created.tzinfo is not None, "created_at must be timezone-aware")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": OUTPUT_KIND,
        "status": "ready_for_diffusion_clean_plate_batch",
        "created_at": created.astimezone(timezone.utc).isoformat(),  # noqa: UP017
        "config": config,
        "source": {
            "propainter_input_manifest": {
                "path": str(input_path),
                "sha256": manifest_sha,
                "size_bytes": manifest_size,
            },
            "pbr_guide_directory": str(pbr_dir),
            "fallback_guide_directory": str(fallback_dir) if fallback_dir is not None else None,
        },
        "frame_count": len(records),
        "frame_ids": observed_ids,
        "frame_records": records,
        "gates": {
            "source_manifest_frame_order_verified": True,
            "source_and_prepared_rgb_hashes_verified": True,
            "prepared_rgb_is_byte_exact_source_in_every_frame": True,
            "ownership_mask_hashes_and_dimensions_verified": True,
            "ownership_masks_bound_to_exact_source_rgb_sha256": True,
            "guides_bound_to_exact_source_rgb_sha256": True,
            "one_unique_dimension_matched_guide_per_frame": True,
            "all_expected_frames_present": True,
        },
    }
    atomic_write_json(output_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--propainter-manifest", type=Path, required=True)
    parser.add_argument("--pbr-guide-dir", type=Path, required=True)
    parser.add_argument("--fallback-guide-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-repo", default=MODEL_REPO)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--variant", default="fp16")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--context-dilation-pixels", type=int, default=48)
    parser.add_argument("--boundary-outer-collar-pixels", type=int, default=6)
    parser.add_argument("--base-seed", type=int, default=2_026_071_820)
    parser.add_argument("--seed-stride", type=int, default=1)
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--strength", type=float, default=0.9)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = build_manifest(args)
    except (DiffusionBatchManifestBuildError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "manifest": str(Path(args.output).expanduser().resolve()),
                "frame_count": manifest["frame_count"],
                "fallback_frame_count": sum(
                    record["guide_selection"]["source"] == "fallback"
                    for record in manifest["frame_records"]
                ),
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
