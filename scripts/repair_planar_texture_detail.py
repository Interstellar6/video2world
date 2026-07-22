#!/usr/bin/env python3
"""Add deterministic local texture detail to a planar texture candidate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from complete_planar_texture_atlas import boundary_normal_gradient_continuity  # noqa: E402

OUTPUT_REPORT_NAME = "planar_texture_report.json"
DETAIL_CONTACT_SHEET_NAME = "planar_texture_detail_repair_contact_sheet.png"
DEFAULT_COLLAR_WIDTH_PIXELS = 28
DEFAULT_INNER_RAMP_PIXELS = 10
DEFAULT_RESIDUAL_SIGMA_PIXELS = 4.0
DEFAULT_RESIDUAL_STRENGTH = 1.35
DEFAULT_DONOR_REGION = "all"
DONOR_REGION_CHOICES = ("all", "lower", "upper", "left", "right", "horizontal", "vertical")
DEFAULT_FILL_MODE = "residual"
FILL_MODE_CHOICES = ("residual", "normalized_gaussian")
DEFAULT_LOW_FREQUENCY_FILL_SIGMA_PIXELS = 18.0
DEFAULT_REGION_DONOR_MODE = "single"
REGION_DONOR_MODE_CHOICES = ("single", "plane_label_auto")
DEFAULT_MINIMUM_DONOR_LUMINANCE = 0.0


class TextureDetailRepairError(ValueError):
    """Raised when a candidate cannot be repaired safely."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TextureDetailRepairError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_object(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"candidate report is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TextureDetailRepairError(f"cannot read candidate report: {exc}") from exc
    require(isinstance(value, dict) and value, "candidate report must be a non-empty object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def rewrite_path_prefix(value: Any, *, old_prefix: str, new_prefix: str) -> Any:
    if isinstance(value, str):
        return value.replace(old_prefix, new_prefix)
    if isinstance(value, list):
        return [
            rewrite_path_prefix(item, old_prefix=old_prefix, new_prefix=new_prefix)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: rewrite_path_prefix(item, old_prefix=old_prefix, new_prefix=new_prefix)
            for key, item in value.items()
        }
    return value


def rebase_report_paths(report_path: Path, *, old_root: Path, new_root: Path) -> None:
    report = read_json_object(report_path)
    rebased = rewrite_path_prefix(
        report,
        old_prefix=str(old_root),
        new_prefix=str(new_root),
    )
    write_json(report_path, rebased)


def resolve_file(value: Any, *, relative_to: Path, label: str) -> Path:
    if isinstance(value, dict):
        value = value.get("path")
    require(isinstance(value, str) and value.strip(), f"{label} path is missing")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    resolved = path.resolve()
    require(resolved.is_file(), f"{label} is missing: {resolved}")
    return resolved


def gaussian_blur_rgb(image: np.ndarray, sigma: float) -> np.ndarray:
    blurred = np.empty_like(image, dtype=np.float32)
    for channel in range(image.shape[2]):
        blurred[:, :, channel] = ndimage.gaussian_filter(
            image[:, :, channel],
            sigma=sigma,
            mode="nearest",
        )
    return blurred


def normalized_gaussian_fill_rgb(
    image: np.ndarray,
    fill_mask: np.ndarray,
    *,
    sigma: float,
    known_mask: np.ndarray | None = None,
) -> np.ndarray:
    require(image.ndim == 3 and image.shape[2] == 3, "image must be RGB")
    require(fill_mask.shape == image.shape[:2], "fill mask shape mismatch")
    require(sigma > 0.0, "low-frequency fill sigma must be positive")
    known = ~fill_mask.astype(bool) if known_mask is None else known_mask.astype(bool)
    require(known.shape == fill_mask.shape, "known mask shape mismatch")
    require(np.any(known), "normalized Gaussian fill has no known pixels")
    known_weight = known.astype(np.float32)
    blurred_weight = ndimage.gaussian_filter(known_weight, sigma=sigma, mode="nearest")
    require(
        bool(np.any(blurred_weight[fill_mask] > 1e-6)),
        "fill mask has no known pixels in normalized Gaussian support",
    )
    filled = image.astype(np.float32).copy()
    for channel in range(image.shape[2]):
        weighted = image[:, :, channel].astype(np.float32) * known_weight
        blurred = ndimage.gaussian_filter(weighted, sigma=sigma, mode="nearest")
        normalized = blurred / np.maximum(blurred_weight, 1e-6)
        filled[fill_mask, channel] = normalized[fill_mask]
    return filled


def select_texture_donor_collar(
    synthetic_mask: np.ndarray,
    collar: np.ndarray,
    *,
    donor_region: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    require(donor_region in DONOR_REGION_CHOICES, f"unsupported donor region: {donor_region}")
    mask = synthetic_mask.astype(bool)
    require(np.any(mask), "synthetic mask is empty")
    rows, columns = np.nonzero(mask)
    top = int(rows.min())
    bottom = int(rows.max())
    left = int(columns.min())
    right = int(columns.max())
    if donor_region == "all":
        donor = collar
    elif donor_region == "lower":
        row_index = np.indices(mask.shape)[0]
        donor = collar & (row_index > bottom)
    elif donor_region == "upper":
        row_index = np.indices(mask.shape)[0]
        donor = collar & (row_index < top)
    elif donor_region == "left":
        column_index = np.indices(mask.shape)[1]
        donor = collar & (column_index < left)
    elif donor_region == "right":
        column_index = np.indices(mask.shape)[1]
        donor = collar & (column_index > right)
    elif donor_region == "horizontal":
        column_index = np.indices(mask.shape)[1]
        donor = collar & ((column_index < left) | (column_index > right))
    else:
        row_index = np.indices(mask.shape)[0]
        donor = collar & ((row_index < top) | (row_index > bottom))
    if not np.any(donor):
        donor = collar
        fallback_to_all = True
    else:
        fallback_to_all = False
    return donor, {
        "donor_region": donor_region,
        "fallback_to_all_collar": fallback_to_all,
        "mask_bounds_inclusive": [left, top, right, bottom],
        "available_collar_pixels": int(collar.sum()),
        "selected_donor_collar_pixels": int(donor.sum()),
    }


def label_ids(label_image: np.ndarray) -> np.ndarray:
    require(label_image.ndim in {2, 3}, "region labels must be a 2D or RGB image")
    if label_image.ndim == 2:
        return label_image.astype(np.int64)
    require(label_image.shape[2] >= 3, "RGB region labels must have at least 3 channels")
    labels = label_image[:, :, :3].astype(np.int64)
    return (labels[:, :, 0] << 16) + (labels[:, :, 1] << 8) + labels[:, :, 2]


def partition_mask_by_label_seeds(
    synthetic_mask: np.ndarray,
    region_labels: np.ndarray,
) -> list[tuple[np.ndarray, int]]:
    mask = synthetic_mask.astype(bool)
    ids = label_ids(region_labels)
    require(ids.shape == mask.shape, "region labels shape mismatch")
    seed = mask & (ids != 0)
    if not np.any(seed):
        return [(mask, 0)]
    seed_ids = np.zeros(mask.shape, dtype=np.int64)
    seed_ids[seed] = ids[seed]
    _, nearest_indices = ndimage.distance_transform_edt(
        ~seed,
        return_distances=True,
        return_indices=True,
    )
    assigned = seed_ids[nearest_indices[0], nearest_indices[1]]
    segments: list[tuple[np.ndarray, int]] = []
    for value in sorted(int(item) for item in np.unique(assigned[mask]) if int(item) != 0):
        segment = mask & (assigned == value)
        if np.any(segment):
            segments.append((segment, value))
    return segments or [(mask, 0)]


def choose_auto_region_donor(segment: np.ndarray, whole_mask: np.ndarray) -> str:
    rows, _ = np.nonzero(segment)
    whole_rows, _ = np.nonzero(whole_mask)
    require(rows.size > 0, "segment is empty")
    require(whole_rows.size > 0, "synthetic mask is empty")
    segment_center_y = float(rows.mean())
    whole_center_y = float(whole_rows.mean())
    return "lower" if segment_center_y >= whole_center_y else "horizontal"


def transfer_texture_into_region(
    image_float: np.ndarray,
    region_mask: np.ndarray,
    whole_mask: np.ndarray,
    *,
    collar_width_pixels: int,
    ramp: np.ndarray,
    residual: np.ndarray,
    residual_strength: float,
    donor_region: str,
    fill_mode: str,
    low_frequency_fill_sigma_pixels: float,
    minimum_donor_luminance: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    collar = ndimage.binary_dilation(region_mask, iterations=collar_width_pixels) & ~whole_mask
    require(np.any(collar), "synthetic region lacks a texture donor collar")
    donor_collar, donor_details = select_texture_donor_collar(
        region_mask,
        collar,
        donor_region=donor_region,
    )
    donor_before_luminance_filter = int(donor_collar.sum())
    if minimum_donor_luminance > 0.0:
        luminance = image_float.mean(axis=2)
        filtered = donor_collar & (luminance >= minimum_donor_luminance)
        if np.any(filtered):
            donor_collar = filtered
            donor_details["fallback_after_luminance_filter"] = False
        else:
            donor_details["fallback_after_luminance_filter"] = True
    else:
        donor_details["fallback_after_luminance_filter"] = False
    donor_details["minimum_donor_luminance"] = minimum_donor_luminance
    donor_details["donor_pixels_before_luminance_filter"] = donor_before_luminance_filter
    donor_details["selected_donor_collar_pixels"] = int(donor_collar.sum())
    _, nearest_indices = ndimage.distance_transform_edt(
        ~donor_collar,
        return_distances=True,
        return_indices=True,
    )
    transferred = residual[
        nearest_indices[0],
        nearest_indices[1],
    ]
    if fill_mode == "normalized_gaussian":
        base = normalized_gaussian_fill_rgb(
            image_float,
            region_mask,
            sigma=low_frequency_fill_sigma_pixels,
            known_mask=donor_collar,
        )
        base = image_float + (base - image_float) * ramp[:, :, None]
    else:
        base = image_float
    repaired_region = (
        base[region_mask]
        + transferred[region_mask] * ramp[region_mask, None] * float(residual_strength)
    )
    return repaired_region, donor_details


def repair_texture_detail(
    image: np.ndarray,
    synthetic_mask: np.ndarray,
    *,
    collar_width_pixels: int,
    inner_ramp_pixels: int,
    residual_sigma_pixels: float,
    residual_strength: float,
    donor_region: str = DEFAULT_DONOR_REGION,
    fill_mode: str = DEFAULT_FILL_MODE,
    low_frequency_fill_sigma_pixels: float = DEFAULT_LOW_FREQUENCY_FILL_SIGMA_PIXELS,
    region_donor_mode: str = DEFAULT_REGION_DONOR_MODE,
    region_labels: np.ndarray | None = None,
    minimum_donor_luminance: float = DEFAULT_MINIMUM_DONOR_LUMINANCE,
) -> tuple[np.ndarray, dict[str, Any]]:
    require(image.ndim == 3 and image.shape[2] == 3, "image must be RGB")
    require(synthetic_mask.shape == image.shape[:2], "synthetic mask shape mismatch")
    require(collar_width_pixels > 0, "collar width must be positive")
    require(inner_ramp_pixels >= 0, "inner ramp must be non-negative")
    require(residual_sigma_pixels > 0.0, "residual sigma must be positive")
    require(residual_strength >= 0.0, "residual strength must be non-negative")
    require(fill_mode in FILL_MODE_CHOICES, f"unsupported fill mode: {fill_mode}")
    require(
        low_frequency_fill_sigma_pixels > 0.0,
        "low-frequency fill sigma must be positive",
    )
    require(minimum_donor_luminance >= 0.0, "minimum donor luminance must be non-negative")
    require(
        region_donor_mode in REGION_DONOR_MODE_CHOICES,
        f"unsupported region donor mode: {region_donor_mode}",
    )
    if region_donor_mode == "plane_label_auto":
        require(region_labels is not None, "plane-label auto mode requires region labels")

    mask = synthetic_mask.astype(bool)
    require(np.any(mask), "synthetic mask is empty")
    image_float = image.astype(np.float32)
    low_frequency = gaussian_blur_rgb(image_float, residual_sigma_pixels)
    residual = image_float - low_frequency
    distance_inside = ndimage.distance_transform_edt(mask).astype(np.float32)
    if inner_ramp_pixels:
        ramp = np.clip(distance_inside / float(inner_ramp_pixels), 0.0, 1.0)
        ramp = ramp * ramp * (3.0 - 2.0 * ramp)
    else:
        ramp = np.ones(mask.shape, dtype=np.float32)
    repaired = image_float.copy()
    if region_donor_mode == "plane_label_auto" and region_labels is not None:
        regions = partition_mask_by_label_seeds(mask, region_labels)
    else:
        regions = [(mask, 0)]
    region_records: list[dict[str, Any]] = []
    total_donor_pixels = 0
    for region_index, (region_mask, label_id) in enumerate(regions):
        selected_donor_region = (
            choose_auto_region_donor(region_mask, mask)
            if region_donor_mode == "plane_label_auto"
            else donor_region
        )
        repaired_region, donor_details = transfer_texture_into_region(
            image_float,
            region_mask,
            mask,
            collar_width_pixels=collar_width_pixels,
            ramp=ramp,
            residual=residual,
            residual_strength=residual_strength,
            donor_region=selected_donor_region,
            fill_mode=fill_mode,
            low_frequency_fill_sigma_pixels=low_frequency_fill_sigma_pixels,
            minimum_donor_luminance=minimum_donor_luminance,
        )
        repaired[region_mask] = repaired_region
        total_donor_pixels += int(donor_details["selected_donor_collar_pixels"])
        region_records.append(
            {
                "region_index": region_index,
                "label_id": label_id,
                "donor_region": selected_donor_region,
                "synthetic_pixels": int(region_mask.sum()),
                **donor_details,
            }
        )
    repaired = np.clip(np.rint(repaired), 0, 255).astype(np.uint8)
    repaired[~mask] = image[~mask]
    outside_exact = bool(np.array_equal(repaired[~mask], image[~mask]))
    return repaired, {
        "method": "nearest_collar_high_frequency_residual_transfer",
        "collar_width_pixels": collar_width_pixels,
        "inner_ramp_pixels": inner_ramp_pixels,
        "residual_sigma_pixels": residual_sigma_pixels,
        "residual_strength": residual_strength,
        "fill_mode": fill_mode,
        "low_frequency_fill_sigma_pixels": low_frequency_fill_sigma_pixels,
        "minimum_donor_luminance": minimum_donor_luminance,
        "donor_region": donor_region,
        "region_donor_mode": region_donor_mode,
        "region_count": len(region_records),
        "regions": region_records,
        "synthetic_pixels": int(mask.sum()),
        "texture_donor_collar_pixels": total_donor_pixels,
        "outside_synthetic_mask_rgb_exact": outside_exact,
        "claims_measured_donor": False,
    }


def load_candidate(candidate_report_path: Path) -> tuple[dict[str, Any], Path, str]:
    path = candidate_report_path.expanduser().resolve()
    candidate = read_json_object(path)
    require(candidate.get("schema_version") == 1, "candidate schema_version must equal 1")
    require(
        candidate.get("status") == "texture_candidate_review_pending",
        "candidate must be review-pending",
    )
    require(candidate.get("promotion_approved") is False, "candidate is already promoted")
    require(
        candidate.get("eligible_as_round04_clean_plate") is False,
        "candidate is already eligible",
    )
    return candidate, path, sha256_file(path)


def make_contact_sheet(records: list[dict[str, Any]], path: Path, samples: int) -> None:
    count = min(samples, len(records))
    indices = np.linspace(0, len(records) - 1, count, dtype=int)
    width, height, label = 288, 162, 22
    sheet = Image.new("RGB", (width * 5, (height + label) * count), "#111111")
    draw = ImageDraw.Draw(sheet)
    for output_row, record_index in enumerate(indices):
        record = records[int(record_index)]
        source = Image.open(record["source_frame"]).convert("RGB")
        upstream = Image.open(record["texture_detail_repair"]["upstream_completed_frame"]).convert(
            "RGB"
        )
        completed = Image.open(record["completed_frame"]).convert("RGB")
        synthetic = Image.open(record["synthetic_mask"]).convert("L")
        contribution = Image.new("RGB", source.size, "black")
        contribution.paste(completed, mask=synthetic)
        overlay = source.copy()
        overlay.paste(Image.new("RGB", source.size, (85, 180, 255)), mask=synthetic)
        images = (source, upstream, completed, overlay, contribution)
        titles = (
            f"source {record['frame_id']}",
            "upstream candidate",
            "detail repaired candidate",
            "synthetic mask overlay",
            "synthetic contribution only",
        )
        y = output_row * (height + label)
        for column, (image, title) in enumerate(zip(images, titles, strict=True)):
            image.thumbnail((width, height), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (width, height), "#202020")
            canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
            x = column * width
            sheet.paste(canvas, (x, y + label))
            draw.text((x + 6, y + 5), title, fill="#f4f4f4")
        for image in images:
            image.close()
        source.close()
        upstream.close()
        completed.close()
        synthetic.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def build_detail_repair_candidate(args: argparse.Namespace) -> dict[str, Any]:
    candidate, candidate_path, candidate_sha = load_candidate(args.candidate_report)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    require(
        not (output / OUTPUT_REPORT_NAME).exists(),
        f"output report already exists: {output / OUTPUT_REPORT_NAME}",
    )
    frames_dir = output / "frames"
    frames_dir.mkdir(parents=True)

    candidate_dir = candidate_path.parent
    repaired = copy.deepcopy(candidate)
    repaired["status"] = "texture_candidate_review_pending"
    repaired["promotion_approved"] = False
    repaired["eligible_as_round04_clean_plate"] = False
    repaired["promotion_blocker"] = (
        "texture detail repair candidate requires bound automated and visual review"
    )
    repaired["upstream_planar_texture_candidate"] = {
        "path": str(candidate_path),
        "sha256": candidate_sha,
        "status": candidate.get("status"),
    }
    repaired["texture_detail_repair"] = {
        "enabled": True,
        "method": "nearest_collar_high_frequency_residual_transfer",
        "collar_width_pixels": args.texture_collar_width_pixels,
        "inner_ramp_pixels": args.texture_inner_ramp_pixels,
        "residual_sigma_pixels": args.texture_residual_sigma_pixels,
        "residual_strength": args.texture_residual_strength,
        "donor_region": args.texture_donor_region,
        "fill_mode": args.texture_fill_mode,
        "low_frequency_fill_sigma_pixels": args.texture_low_frequency_fill_sigma_pixels,
        "region_donor_mode": args.texture_region_donor_mode,
        "minimum_donor_luminance": args.texture_minimum_donor_luminance,
        "claims_measured_donor": False,
        "outside_synthetic_mask_rgb_exact": True,
        "created_at": datetime.now(UTC).isoformat(),
    }

    all_outside_exact = True
    frame_records: list[dict[str, Any]] = []
    for index, record in enumerate(candidate["frame_records"]):
        frame_id = str(record["frame_id"])
        completed_path = resolve_file(
            record.get("completed_frame"),
            relative_to=candidate_dir,
            label=f"{frame_id} completed frame",
        )
        mask_path = resolve_file(
            record.get("synthetic_mask"),
            relative_to=candidate_dir,
            label=f"{frame_id} synthetic mask",
        )
        region_labels = None
        if args.texture_region_donor_mode == "plane_label_auto":
            labels_path = resolve_file(
                record.get("plane_labels"),
                relative_to=candidate_dir,
                label=f"{frame_id} plane labels",
            )
            region_labels = np.asarray(Image.open(labels_path).convert("RGB"), dtype=np.uint8)
        image = np.asarray(Image.open(completed_path).convert("RGB"), dtype=np.uint8)
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0
        repaired_image, details = repair_texture_detail(
            image,
            mask,
            collar_width_pixels=args.texture_collar_width_pixels,
            inner_ramp_pixels=args.texture_inner_ramp_pixels,
            residual_sigma_pixels=args.texture_residual_sigma_pixels,
            residual_strength=args.texture_residual_strength,
            donor_region=args.texture_donor_region,
            fill_mode=args.texture_fill_mode,
            low_frequency_fill_sigma_pixels=args.texture_low_frequency_fill_sigma_pixels,
            region_donor_mode=args.texture_region_donor_mode,
            region_labels=region_labels,
            minimum_donor_luminance=args.texture_minimum_donor_luminance,
        )
        output_frame = frames_dir / f"{index:04d}.png"
        Image.fromarray(repaired_image).save(output_frame)
        output_sha = sha256_file(output_frame)
        all_outside_exact = all_outside_exact and details["outside_synthetic_mask_rgb_exact"]
        updated = copy.deepcopy(record)
        updated["completed_frame"] = str(output_frame)
        updated["completed_frame_sha256"] = output_sha
        updated["synthetic_boundary_continuity_full_resolution"] = (
            boundary_normal_gradient_continuity(repaired_image, ~mask)
        )
        updated["texture_detail_repair"] = {
            **details,
            "upstream_completed_frame": str(completed_path),
            "upstream_completed_frame_sha256": sha256_file(completed_path),
            "repaired_frame": str(output_frame),
            "repaired_frame_sha256": output_sha,
        }
        frame_records.append(updated)

    repaired["frame_records"] = frame_records
    repaired["texture_detail_repair"]["outside_synthetic_mask_rgb_exact"] = all_outside_exact
    gates = copy.deepcopy(repaired.get("gates", {}))
    gates["outside_synthetic_mask_rgb_exact"] = bool(
        gates.get("outside_synthetic_mask_rgb_exact") is True and all_outside_exact
    )
    gates["texture_detail_repair_outside_synthetic_mask_rgb_exact"] = all_outside_exact
    gates["synthetic_texture_detail_claims_measured_donor"] = False
    gates["visual_quality"] = "pending_human_or_vlm_review"
    repaired["gates"] = gates

    full_resolution_ids = [
        str(item["frame_id"]) for item in candidate.get("full_resolution_review", [])
    ]
    records_by_id = {str(record["frame_id"]): record for record in frame_records}
    repaired["full_resolution_review"] = []
    for item in candidate.get("full_resolution_review", []):
        frame_id = str(item["frame_id"])
        record = records_by_id[frame_id]
        repaired["full_resolution_review"].append(
            {
                **copy.deepcopy(item),
                "completed_frame": record["completed_frame"],
                "completed_frame_sha256": record["completed_frame_sha256"],
            }
        )
    reviewed_records = [records_by_id[frame_id] for frame_id in full_resolution_ids]
    repaired["full_resolution_boundary_continuity_summary"] = {
        "frame_ids": full_resolution_ids,
        "maximum_boundary_color_p95_abs_rgb_delta": max(
            float(
                record["synthetic_boundary_continuity_full_resolution"][
                    "boundary_color_p95_abs_rgb_delta"
                ]
            )
            for record in reviewed_records
        ),
        "maximum_boundary_normal_gradient_p95_abs_rgb_delta": max(
            float(
                record["synthetic_boundary_continuity_full_resolution"][
                    "boundary_normal_gradient_p95_abs_rgb_delta"
                ]
            )
            for record in reviewed_records
        ),
    }
    make_contact_sheet(
        frame_records,
        output / DETAIL_CONTACT_SHEET_NAME,
        samples=args.review_contact_sheet_samples,
    )
    write_json(output / OUTPUT_REPORT_NAME, repaired)
    print(
        json.dumps(
            {
                "status": repaired["status"],
                "promotion_approved": repaired["promotion_approved"],
                "eligible_as_round04_clean_plate": repaired["eligible_as_round04_clean_plate"],
                "outside_synthetic_mask_rgb_exact": all_outside_exact,
                "report": str(output / OUTPUT_REPORT_NAME),
            },
            sort_keys=True,
        )
    )
    return repaired


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--texture-collar-width-pixels",
        type=int,
        default=DEFAULT_COLLAR_WIDTH_PIXELS,
    )
    parser.add_argument(
        "--texture-inner-ramp-pixels",
        type=int,
        default=DEFAULT_INNER_RAMP_PIXELS,
    )
    parser.add_argument(
        "--texture-residual-sigma-pixels",
        type=float,
        default=DEFAULT_RESIDUAL_SIGMA_PIXELS,
    )
    parser.add_argument(
        "--texture-residual-strength",
        type=float,
        default=DEFAULT_RESIDUAL_STRENGTH,
    )
    parser.add_argument(
        "--texture-donor-region",
        choices=DONOR_REGION_CHOICES,
        default=DEFAULT_DONOR_REGION,
    )
    parser.add_argument(
        "--texture-fill-mode",
        choices=FILL_MODE_CHOICES,
        default=DEFAULT_FILL_MODE,
    )
    parser.add_argument(
        "--texture-low-frequency-fill-sigma-pixels",
        type=float,
        default=DEFAULT_LOW_FREQUENCY_FILL_SIGMA_PIXELS,
    )
    parser.add_argument(
        "--texture-region-donor-mode",
        choices=REGION_DONOR_MODE_CHOICES,
        default=DEFAULT_REGION_DONOR_MODE,
    )
    parser.add_argument(
        "--texture-minimum-donor-luminance",
        type=float,
        default=DEFAULT_MINIMUM_DONOR_LUMINANCE,
    )
    parser.add_argument("--review-contact-sheet-samples", type=int, default=6)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_candidate = args.output.expanduser()
    if output_candidate.is_symlink():
        raise TextureDetailRepairError(f"output cannot be a symlink: {output_candidate}")
    output = output_candidate.resolve()
    if output.exists():
        raise TextureDetailRepairError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        staged_args = argparse.Namespace(**vars(args))
        staged_args.output = staging
        build_detail_repair_candidate(staged_args)
        if output.exists():
            raise TextureDetailRepairError(f"output appeared during repair: {output}")
        os.rename(staging, output)
        rebase_report_paths(
            output / OUTPUT_REPORT_NAME,
            old_root=staging,
            new_root=output,
        )
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
