#!/usr/bin/env python3
"""Compose scene-agnostic clean-plate candidates with a protected soft collar.

The input manifest binds a previous composite, a source-aligned removal mask,
a full-frame candidate fill, and an optional protected mask for every frame.
The source-aligned mask is the weight-one sure-remove core. A Euclidean
distance field creates an outer collar, while protected pixels suppress edits
only in that collar. The tool always emits reviewable artifacts and records
boundary quality separately from structural compositor invariants.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

INPUT_KIND = "video2world.clean_plate_boundary_candidate_input"
REPORT_KIND = "video2world.clean_plate_boundary_candidate_report"
SCHEMA_VERSION = 1


class BoundaryCandidateError(RuntimeError):
    """Raised when a boundary-candidate input contract is invalid."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BoundaryCandidateError(message)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BoundaryCandidateError(f"cannot read JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def require_sha256(value: Any, label: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA-256 digest",
    )
    return value


def resolve_asset(value: Any, *, manifest_path: Path, label: str) -> Path:
    require(isinstance(value, dict), f"{label} must be an asset object")
    path_value = value.get("path")
    require(isinstance(path_value, str) and path_value, f"{label}.path is missing")
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    require(path.is_file(), f"{label} does not exist: {path}")
    expected = require_sha256(value.get("sha256"), f"{label}.sha256")
    actual = sha256_file(path)
    require(actual == expected, f"{label} SHA-256 mismatch: {actual} != {expected}")
    expected_bytes = value.get("bytes")
    if expected_bytes is not None:
        require(
            isinstance(expected_bytes, int)
            and not isinstance(expected_bytes, bool)
            and expected_bytes >= 0,
            f"{label}.bytes must be a non-negative integer",
        )
        require(path.stat().st_size == expected_bytes, f"{label}.bytes mismatch")
    return path


def load_rgb(path: Path, *, label: str) -> np.ndarray:
    try:
        with Image.open(path) as image:
            require(image.mode in {"RGB", "RGBA"}, f"{label} is not RGB-compatible")
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except OSError as exc:
        raise BoundaryCandidateError(f"cannot decode {label}: {path}") from exc


def load_binary_mask(path: Path, *, label: str) -> np.ndarray:
    try:
        with Image.open(path) as image:
            value = np.asarray(image.convert("L"), dtype=np.uint8)
    except OSError as exc:
        raise BoundaryCandidateError(f"cannot decode {label}: {path}") from exc
    require(bool(np.all((value == 0) | (value == 255))), f"{label} must be binary 0/255")
    return value == 255


def save_rgb(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value.astype(np.uint8, copy=False)).save(path)


def save_mask(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value.astype(np.uint8) * 255).save(path)


def save_weight(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = np.clip(np.rint(value * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(encoded).save(path)


def artifact(path: Path, *, relative_to: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def smoothstep(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def build_soft_bands(
    core: np.ndarray,
    protected: np.ndarray,
    *,
    outer_collar_pixels: int,
) -> dict[str, np.ndarray]:
    require(core.ndim == 2, "core mask must be two-dimensional")
    require(protected.shape == core.shape, "protected mask shape differs from core")
    require(bool(np.any(core)), "source-aligned mask must contain at least one core pixel")
    require(outer_collar_pixels >= 1, "outer_collar_pixels must be at least one")

    distance = ndimage.distance_transform_edt(~core).astype(np.float32)
    collar_envelope = (~core) & (distance <= float(outer_collar_pixels))
    protected_collar = collar_envelope & protected
    collar = collar_envelope & ~protected
    weight = np.zeros(core.shape, dtype=np.float32)
    weight[core] = 1.0
    normalized = (outer_collar_pixels + 1.0 - distance) / (outer_collar_pixels + 1.0)
    weight[collar] = smoothstep(normalized[collar])
    editable = core | collar
    outside = ~editable

    require(not bool(np.any(core & collar)), "core and collar overlap")
    require(not bool(np.any(collar & protected)), "collar escaped the protected mask")
    require(bool(np.all(weight[core] == 1.0)), "core weights differ from one")
    require(bool(np.all(weight[outside] == 0.0)), "outside weights differ from zero")
    require(
        not bool(np.any((weight < 0.0) | (weight > 1.0) | ~np.isfinite(weight))),
        "blend weight must be finite and in [0, 1]",
    )
    return {
        "core": core,
        "collar": collar,
        "collar_envelope": collar_envelope,
        "protected": protected,
        "protected_collar": protected_collar,
        "editable": editable,
        "outside": outside,
        "distance": distance,
        "weight": weight,
    }


def blend_candidate(
    previous: np.ndarray,
    candidate: np.ndarray,
    bands: dict[str, np.ndarray],
) -> np.ndarray:
    require(previous.shape == candidate.shape, "previous and candidate RGB shapes differ")
    require(previous.ndim == 3 and previous.shape[2] == 3, "RGB inputs must be HxWx3")
    require(bands["weight"].shape == previous.shape[:2], "weight shape differs from RGB")
    alpha = bands["weight"][..., None].astype(np.float64)
    blended = candidate.astype(np.float64) * alpha + previous.astype(np.float64) * (1.0 - alpha)
    output = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
    require(
        bool(np.array_equal(output[bands["core"]], candidate[bands["core"]])),
        "core RGB does not equal candidate fill",
    )
    require(
        bool(np.array_equal(output[bands["outside"]], previous[bands["outside"]])),
        "outside RGB does not equal previous composite",
    )
    return output


def boundary_normal_gradient_continuity(
    completed: np.ndarray,
    observed: np.ndarray,
) -> dict[str, float | int | None]:
    """Measure color and first-derivative jumps across a binary boundary."""

    color_delta: list[float] = []
    gradient_delta: list[float] = []
    height, width = observed.shape

    def add_triplet(
        observed_color: np.ndarray,
        synthetic_color: np.ndarray,
        observed_outer: np.ndarray | None,
        synthetic_inner: np.ndarray | None,
    ) -> None:
        cross_gradient = synthetic_color.astype(np.float64) - observed_color
        color_delta.append(float(np.abs(cross_gradient).mean()))
        if observed_outer is not None:
            observed_gradient = observed_color.astype(np.float64) - observed_outer
            gradient_delta.append(float(np.abs(cross_gradient - observed_gradient).mean()))
        if synthetic_inner is not None:
            synthetic_gradient = synthetic_inner.astype(np.float64) - synthetic_color
            gradient_delta.append(float(np.abs(cross_gradient - synthetic_gradient).mean()))

    for row in range(height):
        for column in np.flatnonzero(observed[row, 1:] != observed[row, :-1]):
            if observed[row, column]:
                add_triplet(
                    completed[row, column],
                    completed[row, column + 1],
                    completed[row, column - 1]
                    if column > 0 and observed[row, column - 1]
                    else None,
                    completed[row, column + 2]
                    if column + 2 < width and not observed[row, column + 2]
                    else None,
                )
            else:
                add_triplet(
                    completed[row, column + 1],
                    completed[row, column],
                    completed[row, column + 2]
                    if column + 2 < width and observed[row, column + 2]
                    else None,
                    completed[row, column - 1]
                    if column > 0 and not observed[row, column - 1]
                    else None,
                )
    for row in range(height - 1):
        for column in np.flatnonzero(observed[row + 1] != observed[row]):
            if observed[row, column]:
                add_triplet(
                    completed[row, column],
                    completed[row + 1, column],
                    completed[row - 1, column] if row > 0 and observed[row - 1, column] else None,
                    completed[row + 2, column]
                    if row + 2 < height and not observed[row + 2, column]
                    else None,
                )
            else:
                add_triplet(
                    completed[row + 1, column],
                    completed[row, column],
                    completed[row + 2, column]
                    if row + 2 < height and observed[row + 2, column]
                    else None,
                    completed[row - 1, column]
                    if row > 0 and not observed[row - 1, column]
                    else None,
                )
    return {
        "boundary_edge_count": len(color_delta),
        "boundary_color_mean_abs_rgb_delta": (float(np.mean(color_delta)) if color_delta else None),
        "boundary_color_p95_abs_rgb_delta": (
            float(np.quantile(color_delta, 0.95)) if color_delta else None
        ),
        "boundary_normal_gradient_pair_count": len(gradient_delta),
        "boundary_normal_gradient_mean_abs_rgb_delta": (
            float(np.mean(gradient_delta)) if gradient_delta else None
        ),
        "boundary_normal_gradient_p95_abs_rgb_delta": (
            float(np.quantile(gradient_delta, 0.95)) if gradient_delta else None
        ),
    }


def quality_gate(metric: float | int | None, threshold: float) -> dict[str, Any]:
    passed = metric is not None and float(metric) <= threshold
    return {"threshold": threshold, "observed": metric, "passed": passed}


def validate_config(manifest: dict[str, Any]) -> dict[str, float | int | str]:
    config = manifest.get("config")
    require(isinstance(config, dict), "manifest.config must be an object")
    radius = config.get("outer_collar_pixels")
    require(
        isinstance(radius, int) and not isinstance(radius, bool) and radius >= 1,
        "config.outer_collar_pixels must be a positive integer",
    )
    color_limit = config.get("maximum_boundary_color_p95_delta", 24.0)
    gradient_limit = config.get("maximum_boundary_gradient_p95_delta", 32.0)
    for value, label in (
        (color_limit, "maximum_boundary_color_p95_delta"),
        (gradient_limit, "maximum_boundary_gradient_p95_delta"),
    ):
        require(
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and np.isfinite(value)
            and value >= 0,
            f"config.{label} must be a non-negative finite number",
        )
    curve = config.get("blend_curve", "smoothstep_euclidean_distance")
    require(
        curve == "smoothstep_euclidean_distance",
        "config.blend_curve must be smoothstep_euclidean_distance",
    )
    return {
        "outer_collar_pixels": radius,
        "maximum_boundary_color_p95_delta": float(color_limit),
        "maximum_boundary_gradient_p95_delta": float(gradient_limit),
        "blend_curve": curve,
    }


def process_frame(
    record: dict[str, Any],
    *,
    manifest_path: Path,
    staging: Path,
    config: dict[str, float | int | str],
) -> dict[str, Any]:
    sequence_index = record.get("sequence_index")
    frame_id = record.get("frame_id")
    require(
        isinstance(sequence_index, int) and not isinstance(sequence_index, bool),
        "frame sequence_index must be an integer",
    )
    require(isinstance(frame_id, str) and frame_id, "frame_id must be a non-empty string")

    previous_path = resolve_asset(
        record.get("previous_composite"),
        manifest_path=manifest_path,
        label=f"{frame_id} previous_composite",
    )
    mask_path = resolve_asset(
        record.get("source_aligned_mask"),
        manifest_path=manifest_path,
        label=f"{frame_id} source_aligned_mask",
    )
    candidate_path = resolve_asset(
        record.get("candidate_fill"),
        manifest_path=manifest_path,
        label=f"{frame_id} candidate_fill",
    )
    protected_value = record.get("protected_mask")
    protected_path = (
        resolve_asset(
            protected_value,
            manifest_path=manifest_path,
            label=f"{frame_id} protected_mask",
        )
        if protected_value is not None
        else None
    )

    previous = load_rgb(previous_path, label=f"{frame_id} previous composite")
    candidate = load_rgb(candidate_path, label=f"{frame_id} candidate fill")
    core = load_binary_mask(mask_path, label=f"{frame_id} source-aligned mask")
    protected = (
        load_binary_mask(protected_path, label=f"{frame_id} protected mask")
        if protected_path is not None
        else np.zeros_like(core)
    )
    shape = previous.shape[:2]
    require(candidate.shape == previous.shape, f"{frame_id} RGB dimensions differ")
    require(core.shape == shape, f"{frame_id} source-aligned mask dimensions differ")
    require(protected.shape == shape, f"{frame_id} protected mask dimensions differ")

    bands = build_soft_bands(
        core,
        protected,
        outer_collar_pixels=int(config["outer_collar_pixels"]),
    )
    output = blend_candidate(previous, candidate, bands)
    outer_metrics = boundary_normal_gradient_continuity(output, bands["outside"])
    core_metrics = boundary_normal_gradient_continuity(output, ~bands["core"])
    color_gate = quality_gate(
        outer_metrics["boundary_color_p95_abs_rgb_delta"],
        float(config["maximum_boundary_color_p95_delta"]),
    )
    gradient_gate = quality_gate(
        outer_metrics["boundary_normal_gradient_p95_abs_rgb_delta"],
        float(config["maximum_boundary_gradient_p95_delta"]),
    )

    filename = f"{sequence_index:04d}.png"
    output_path = staging / "frames" / filename
    core_path = staging / "masks" / "core" / filename
    collar_path = staging / "masks" / "collar" / filename
    protected_output_path = staging / "masks" / "protected" / filename
    protected_collar_path = staging / "masks" / "protected_collar" / filename
    editable_path = staging / "masks" / "editable" / filename
    weight_path = staging / "weights" / filename
    save_rgb(output_path, output)
    save_mask(core_path, bands["core"])
    save_mask(collar_path, bands["collar"])
    save_mask(protected_output_path, bands["protected"])
    save_mask(protected_collar_path, bands["protected_collar"])
    save_mask(editable_path, bands["editable"])
    save_weight(weight_path, bands["weight"])

    structural_gates = {
        "source_aligned_mask_is_weight_one_core": bool(np.all(bands["weight"][core] == 1.0)),
        "core_and_collar_disjoint": not bool(np.any(bands["core"] & bands["collar"])),
        "collar_excludes_protected_mask": not bool(np.any(bands["collar"] & bands["protected"])),
        "outside_editable_rgb_exact": bool(
            np.array_equal(output[bands["outside"]], previous[bands["outside"]])
        ),
        "core_rgb_equals_candidate_fill": bool(
            np.array_equal(output[bands["core"]], candidate[bands["core"]])
        ),
        "protected_outer_collar_rgb_exact": bool(
            np.array_equal(
                output[bands["protected_collar"]],
                previous[bands["protected_collar"]],
            )
        ),
        "weights_finite_and_bounded": bool(
            np.isfinite(bands["weight"]).all()
            and np.all((bands["weight"] >= 0.0) & (bands["weight"] <= 1.0))
        ),
    }
    require(all(structural_gates.values()), f"{frame_id} structural compositor gate failed")
    collar_weights = bands["weight"][bands["collar"]]
    return {
        "sequence_index": sequence_index,
        "frame_id": frame_id,
        "inputs": {
            "previous_composite": {
                "path": str(previous_path),
                "sha256": sha256_file(previous_path),
            },
            "source_aligned_mask": {
                "path": str(mask_path),
                "sha256": sha256_file(mask_path),
            },
            "candidate_fill": {
                "path": str(candidate_path),
                "sha256": sha256_file(candidate_path),
            },
            "protected_mask": (
                {"path": str(protected_path), "sha256": sha256_file(protected_path)}
                if protected_path is not None
                else None
            ),
        },
        "outputs": {
            "composite_rgb": artifact(output_path, relative_to=staging),
            "core_mask": artifact(core_path, relative_to=staging),
            "collar_mask": artifact(collar_path, relative_to=staging),
            "protected_mask": artifact(protected_output_path, relative_to=staging),
            "protected_collar_mask": artifact(protected_collar_path, relative_to=staging),
            "editable_mask": artifact(editable_path, relative_to=staging),
            "blend_weight": artifact(weight_path, relative_to=staging),
        },
        "pixels": {
            "frame": int(core.size),
            "core": int(bands["core"].sum()),
            "collar": int(bands["collar"].sum()),
            "protected": int(bands["protected"].sum()),
            "protected_overlap_core": int((bands["protected"] & bands["core"]).sum()),
            "protected_outer_collar": int(bands["protected_collar"].sum()),
            "editable": int(bands["editable"].sum()),
            "outside": int(bands["outside"].sum()),
        },
        "distance_field": {
            "metric": "euclidean_pixel_center_distance_to_sure_remove_core",
            "outer_collar_pixels": int(config["outer_collar_pixels"]),
            "curve": config["blend_curve"],
            "collar_weight_minimum": (float(collar_weights.min()) if collar_weights.size else None),
            "collar_weight_maximum": (float(collar_weights.max()) if collar_weights.size else None),
        },
        "boundary_metrics": {
            "outer_editable_boundary": outer_metrics,
            "sure_remove_core_boundary": core_metrics,
        },
        "gates": {
            **structural_gates,
            "boundary_color_p95": color_gate,
            "boundary_normal_gradient_p95": gradient_gate,
        },
    }


def compose_manifest(manifest_path: Path, output_root: Path) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    require(manifest_path.is_file(), f"input manifest does not exist: {manifest_path}")
    require(not output_root.exists(), f"output directory already exists: {output_root}")
    manifest = read_json(manifest_path)
    require(manifest.get("schema_version") == SCHEMA_VERSION, "manifest schema_version mismatch")
    require(manifest.get("kind") == INPUT_KIND, "manifest kind mismatch")
    config = validate_config(manifest)
    raw_records = manifest.get("frame_records")
    require(isinstance(raw_records, list) and raw_records, "frame_records must be non-empty")
    require(
        all(isinstance(record, dict) for record in raw_records),
        "frame record must be an object",
    )
    records: list[dict[str, Any]] = raw_records
    require(
        [record.get("sequence_index") for record in records] == list(range(len(records))),
        "frame sequence_index values must be contiguous and ordered",
    )
    frame_ids = [record.get("frame_id") for record in records]
    require(
        all(isinstance(frame_id, str) and frame_id for frame_id in frame_ids),
        "every frame requires a non-empty frame_id",
    )
    require(len(set(frame_ids)) == len(frame_ids), "frame_id values must be unique")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.staging-",
            dir=output_root.parent,
        )
    )
    try:
        frame_reports = [
            process_frame(
                record,
                manifest_path=manifest_path,
                staging=staging,
                config=config,
            )
            for record in records
        ]
        color_values = [
            float(
                record["boundary_metrics"]["outer_editable_boundary"][
                    "boundary_color_p95_abs_rgb_delta"
                ]
            )
            for record in frame_reports
            if record["boundary_metrics"]["outer_editable_boundary"][
                "boundary_color_p95_abs_rgb_delta"
            ]
            is not None
        ]
        gradient_values = [
            float(
                record["boundary_metrics"]["outer_editable_boundary"][
                    "boundary_normal_gradient_p95_abs_rgb_delta"
                ]
            )
            for record in frame_reports
            if record["boundary_metrics"]["outer_editable_boundary"][
                "boundary_normal_gradient_p95_abs_rgb_delta"
            ]
            is not None
        ]
        structural_passed = all(
            all(
                value is True
                for key, value in record["gates"].items()
                if key not in {"boundary_color_p95", "boundary_normal_gradient_p95"}
            )
            for record in frame_reports
        )
        boundary_passed = all(
            record["gates"]["boundary_color_p95"]["passed"]
            and record["gates"]["boundary_normal_gradient_p95"]["passed"]
            for record in frame_reports
        )
        output_set = [
            {
                "frame_id": record["frame_id"],
                "composite_rgb_sha256": record["outputs"]["composite_rgb"]["sha256"],
                "core_mask_sha256": record["outputs"]["core_mask"]["sha256"],
                "collar_mask_sha256": record["outputs"]["collar_mask"]["sha256"],
                "protected_mask_sha256": record["outputs"]["protected_mask"]["sha256"],
                "blend_weight_sha256": record["outputs"]["blend_weight"]["sha256"],
            }
            for record in frame_reports
        ]
        report = {
            "schema_version": SCHEMA_VERSION,
            "kind": REPORT_KIND,
            "created_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
            "status": (
                "candidate_generated_boundary_passed"
                if boundary_passed
                else "candidate_generated_boundary_failed"
            ),
            "promotion_approved": False,
            "input_manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
            },
            "config": config,
            "frame_count": len(frame_reports),
            "frame_ids": frame_ids,
            "frame_records": frame_reports,
            "aggregate": {
                "maximum_boundary_color_p95_abs_rgb_delta": (
                    max(color_values) if color_values else None
                ),
                "maximum_boundary_normal_gradient_p95_abs_rgb_delta": (
                    max(gradient_values) if gradient_values else None
                ),
                "core_pixels": sum(record["pixels"]["core"] for record in frame_reports),
                "collar_pixels": sum(record["pixels"]["collar"] for record in frame_reports),
                "protected_outer_collar_pixels": sum(
                    record["pixels"]["protected_outer_collar"] for record in frame_reports
                ),
                "output_set_sha256": canonical_sha256(output_set),
            },
            "gates": {
                "all_structural_compositor_invariants_passed": structural_passed,
                "all_frames_boundary_color_p95_within_limit": all(
                    record["gates"]["boundary_color_p95"]["passed"] for record in frame_reports
                ),
                "all_frames_boundary_normal_gradient_p95_within_limit": all(
                    record["gates"]["boundary_normal_gradient_p95"]["passed"]
                    for record in frame_reports
                ),
                "boundary_quality_passed": boundary_passed,
            },
            "limitations": [
                "Boundary continuity does not prove semantic or multi-view correctness.",
                (
                    "Protected masks constrain only the outer collar; source-aligned core pixels "
                    "win overlaps."
                ),
                (
                    "Candidate RGB remains synthetic and must not be relabeled as measured donor "
                    "evidence."
                ),
            ],
        }
        require(structural_passed, "aggregate structural compositor gate failed")
        write_json(staging / "boundary_candidate_report.json", report)
        os.replace(staging, output_root)
        return report
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = compose_manifest(args.input_manifest, args.output)
    except (BoundaryCandidateError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(
                    args.output.expanduser().resolve() / "boundary_candidate_report.json"
                ),
                "frame_count": report["frame_count"],
                "boundary_quality_passed": report["gates"]["boundary_quality_passed"],
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
