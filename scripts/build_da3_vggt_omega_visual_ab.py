#!/usr/bin/env python3
"""Build hash-bound visual evidence for a DA3 versus VGGT-Omega PGSR run."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from video2world.hashing import atomic_write_json, sha256_file

KIND = "video2world.da3_vggt_omega_visual_ab"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


class VisualComparisonError(RuntimeError):
    """Raised when comparable visual evidence cannot be built."""


def artifact_record(path: Path) -> dict[str, Any]:
    digest, size = sha256_file(path)
    return {"path": str(path.resolve()), "sha256": digest, "size_bytes": size}


def resolve_image(root: Path, frame_id: str) -> Path:
    matches = [root / f"{frame_id}{suffix}" for suffix in IMAGE_SUFFIXES]
    existing = [path for path in matches if path.is_file()]
    if len(existing) != 1:
        raise VisualComparisonError(
            f"expected one image for frame {frame_id} under {root}, found {len(existing)}"
        )
    return existing[0]


def select_frame_ids(scene: Path, requested: str | None, sample_count: int) -> list[str]:
    frame_ids = sorted(
        path.stem
        for path in (scene / "color").iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not frame_ids:
        raise VisualComparisonError(f"scene has no RGB frames: {scene}")
    if requested:
        selected = [item.strip() for item in requested.split(",") if item.strip()]
        missing = sorted(set(selected) - set(frame_ids))
        if missing:
            raise VisualComparisonError(f"requested frames are missing: {missing}")
        if len(selected) != len(set(selected)):
            raise VisualComparisonError("requested frame IDs must be unique")
        return selected
    if sample_count <= 0:
        raise VisualComparisonError("sample count must be positive")
    indices = np.linspace(0, len(frame_ids) - 1, min(sample_count, len(frame_ids))).round()
    return [frame_ids[int(index)] for index in np.unique(indices.astype(np.int64))]


def image_metrics(reference: Image.Image, candidate: Image.Image) -> dict[str, float]:
    ref = np.asarray(reference.convert("RGB"), dtype=np.float32) / 255.0
    value = np.asarray(candidate.convert("RGB"), dtype=np.float32) / 255.0
    if ref.shape != value.shape:
        raise VisualComparisonError(
            f"render shape {value.shape} does not match source shape {ref.shape}"
        )
    difference = value - ref
    mse = float(np.mean(difference**2))
    return {
        "l1": float(np.mean(np.abs(difference))),
        "rmse": math.sqrt(mse),
        "psnr_db": float("inf") if mse == 0 else -10.0 * math.log10(mse),
    }


def depth_pair_images(
    da3: np.ndarray, vggt: np.ndarray, scale: float
) -> tuple[Image.Image, Image.Image]:
    if da3.shape != vggt.shape or da3.ndim != 2:
        raise VisualComparisonError("DA3 and VGGT-Omega depth arrays must share an HxW shape")
    vggt_aligned = np.asarray(vggt, dtype=np.float32) * scale
    da3_value = np.asarray(da3, dtype=np.float32)
    valid_da3 = np.isfinite(da3_value) & (da3_value > 0)
    valid_vggt = np.isfinite(vggt_aligned) & (vggt_aligned > 0)
    combined = np.concatenate([da3_value[valid_da3], vggt_aligned[valid_vggt]])
    if not len(combined):
        raise VisualComparisonError("depth pair has no positive finite values")
    lower, upper = np.percentile(combined, [2.0, 98.0])
    if not math.isfinite(float(lower)):
        raise VisualComparisonError("depth pair has no usable display range")
    if upper <= lower:
        margin = max(abs(float(lower)) * 0.01, 1e-6)
        lower, upper = lower - margin, upper + margin

    def colorize(values: np.ndarray, valid: np.ndarray) -> Image.Image:
        normalized = np.clip((upper - values) / (upper - lower), 0.0, 1.0)
        gray = np.zeros(values.shape, dtype=np.uint8)
        gray[valid] = np.round(normalized[valid] * 255.0).astype(np.uint8)
        image = Image.fromarray(gray)
        colored = ImageOps.colorize(image, black="#20133f", white="#f4d35e")
        colored_array = np.asarray(colored).copy()
        colored_array[~valid] = 0
        return Image.fromarray(colored_array)

    return colorize(da3_value, valid_da3), colorize(vggt_aligned, valid_vggt)


def labeled_tile(image: Image.Image, label: str, *, width: int, height: int) -> Image.Image:
    content = ImageOps.contain(image.convert("RGB"), (width, height), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (width, height + 28), "#111317")
    x = (width - content.width) // 2
    y = 28 + (height - content.height) // 2
    tile.paste(content, (x, y))
    ImageDraw.Draw(tile).text((8, 8), label, fill="#f4f5f7")
    return tile


def make_contact_sheet(rows: list[list[tuple[str, Image.Image]]], output: Path, width: int) -> None:
    if not rows or not rows[0]:
        raise VisualComparisonError("visual comparison has no tiles")
    source_width, source_height = rows[0][0][1].size
    height = max(1, round(width * source_height / source_width))
    rendered_rows = [
        [labeled_tile(image, label, width=width, height=height) for label, image in row]
        for row in rows
    ]
    columns = len(rendered_rows[0])
    if any(len(row) != columns for row in rendered_rows):
        raise VisualComparisonError("visual comparison rows have inconsistent columns")
    sheet = Image.new("RGB", (columns * width, len(rendered_rows) * (height + 28)), "#08090b")
    for row_index, row in enumerate(rendered_rows):
        for column_index, tile in enumerate(row):
            sheet.paste(tile, (column_index * width, row_index * (height + 28)))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def mean_metrics(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        raise VisualComparisonError("cannot aggregate empty metric records")
    return {
        key: float(np.mean([record[key] for record in records]))
        for key in ("l1", "rmse", "psnr_db")
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    da3_scene = args.da3_scene.expanduser().resolve()
    vggt_scene = args.vggt_scene.expanduser().resolve()
    da3_render_root = args.da3_render_root.expanduser().resolve()
    vggt_render_root = args.vggt_render_root.expanduser().resolve()
    ab_receipt_path = args.ab_receipt.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    receipt = json.loads(ab_receipt_path.read_text(encoding="utf-8"))
    if receipt.get("kind") != "video2world.da3_vggt_omega_ab_receipt":
        raise VisualComparisonError("numeric A/B receipt has an unexpected kind")
    scale = float(receipt["camera_alignment"]["scale"])
    if not math.isfinite(scale) or scale <= 0:
        raise VisualComparisonError("numeric A/B receipt has an invalid camera scale")
    frame_ids = select_frame_ids(da3_scene, args.frame_ids, args.sample_count)

    rows: list[list[tuple[str, Image.Image]]] = []
    frame_records: list[dict[str, Any]] = []
    da3_metrics: list[dict[str, float]] = []
    vggt_metrics: list[dict[str, float]] = []
    for frame_id in frame_ids:
        source_path = resolve_image(da3_scene / "color", frame_id)
        da3_render_path = resolve_image(da3_render_root, frame_id)
        vggt_render_path = resolve_image(vggt_render_root, frame_id)
        da3_depth_path = da3_scene / "depth_da3" / f"{frame_id}.npy"
        vggt_depth_path = vggt_scene / "depth_da3" / f"{frame_id}.npy"
        source = Image.open(source_path).convert("RGB")
        da3_render = Image.open(da3_render_path).convert("RGB")
        vggt_render = Image.open(vggt_render_path).convert("RGB")
        da3_depth, vggt_depth = depth_pair_images(
            np.load(da3_depth_path), np.load(vggt_depth_path), scale
        )
        da3_frame_metrics = image_metrics(source, da3_render)
        vggt_frame_metrics = image_metrics(source, vggt_render)
        da3_metrics.append(da3_frame_metrics)
        vggt_metrics.append(vggt_frame_metrics)
        frame_records.append(
            {
                "frame_id": frame_id,
                "source": artifact_record(source_path),
                "da3_depth": artifact_record(da3_depth_path),
                "vggt_omega_depth": artifact_record(vggt_depth_path),
                "da3_render": {**artifact_record(da3_render_path), "metrics": da3_frame_metrics},
                "vggt_omega_render": {
                    **artifact_record(vggt_render_path),
                    "metrics": vggt_frame_metrics,
                },
            }
        )
        rows.append(
            [
                (f"{frame_id} source", source),
                ("DA3 depth", da3_depth),
                ("VGGT-Omega depth (aligned)", vggt_depth),
                (f"DA3 PGSR {da3_frame_metrics['psnr_db']:.2f} dB", da3_render),
                (f"VGGT PGSR {vggt_frame_metrics['psnr_db']:.2f} dB", vggt_render),
            ]
        )

    da3_aggregate = mean_metrics(da3_metrics)
    vggt_aggregate = mean_metrics(vggt_metrics)
    delta = {
        "l1": vggt_aggregate["l1"] - da3_aggregate["l1"],
        "rmse": vggt_aggregate["rmse"] - da3_aggregate["rmse"],
        "psnr_db": vggt_aggregate["psnr_db"] - da3_aggregate["psnr_db"],
    }
    if delta["psnr_db"] > 0 and delta["l1"] < 0:
        indication = "vggt_omega_better_selected_render_fit"
    elif delta["psnr_db"] < 0 and delta["l1"] > 0:
        indication = "da3_better_selected_render_fit"
    else:
        indication = "mixed_selected_render_fit"

    contact_sheet = output_dir / "da3_vggt_omega_visual_ab.jpg"
    make_contact_sheet(rows, contact_sheet, args.thumbnail_width)
    report = {
        "schema_version": 1,
        "kind": KIND,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "visual_evidence_built_manual_review_required",
        "promotion_allowed": False,
        "scene_id": da3_scene.name,
        "frame_ids": frame_ids,
        "camera_scale_applied_to_vggt_depth": scale,
        "frames": frame_records,
        "render_metrics": {
            "da3": da3_aggregate,
            "vggt_omega": vggt_aggregate,
            "vggt_minus_da3": delta,
            "numeric_indication": indication,
            "scope": "selected training-view RGB reconstruction fit, not novel-view quality",
        },
        "inputs": {"numeric_ab_receipt": artifact_record(ab_receipt_path)},
        "outputs": {"contact_sheet": artifact_record(contact_sheet)},
        "manual_review": {
            "status": "pending",
            "required_checks": [
                "camera framing and trajectory continuity",
                "thin structures and edges",
                "depth discontinuities and foreground bleeding",
                "PGSR blur, floaters, and duplicated surfaces",
                "TSDF visual geometry in a separate render",
            ],
        },
        "claim_boundary": (
            "This sheet compares selected training views after aligning only the global VGGT-Omega "
            "trajectory scale. It does not establish metric depth accuracy, novel-view quality, "
            "semantic lifting quality, TSDF quality, or world promotion."
        ),
    }
    atomic_write_json(output_dir / "visual_ab_receipt.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--da3-scene", type=Path, required=True)
    parser.add_argument("--vggt-scene", type=Path, required=True)
    parser.add_argument("--da3-render-root", type=Path, required=True)
    parser.add_argument("--vggt-render-root", type=Path, required=True)
    parser.add_argument("--ab-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-ids")
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument("--thumbnail-width", type=int, default=320)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.thumbnail_width < 64:
        print(json.dumps({"status": "failed", "error": "thumbnail width must be at least 64"}))
        return 2
    try:
        report = build_report(args)
    except (KeyError, OSError, ValueError, VisualComparisonError) as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}))
        return 2
    print(json.dumps({"status": report["status"], "outputs": report["outputs"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
