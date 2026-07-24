#!/usr/bin/env python3
"""Fit front-half TRELLIS.2 assets with calibrated mask evidence.

The script consumes ``prepare_front_half_scene_fit_evidence.py`` output and
produces world-baked candidate GLBs plus source-view overlays. It intentionally
does not replace point clouds in a scene: planar objects remain held until their
front/back material orientation is visually audited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image

try:
    from scripts.build_trellis_replacement_plan import PLANAR_CATEGORIES, find_asset
    from scripts.fit_completed_object_to_scene import fit_completed_object_to_scene
    from scripts.qa_scene_camera_silhouette import load_camera, make_overlay, project_mesh_silhouette
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from build_trellis_replacement_plan import PLANAR_CATEGORIES, find_asset  # type: ignore[no-redef]
    from fit_completed_object_to_scene import fit_completed_object_to_scene  # type: ignore[no-redef]
    from qa_scene_camera_silhouette import (  # type: ignore[no-redef]
        load_camera,
        make_overlay,
        project_mesh_silhouette,
    )


KIND = "video2world.front_half_trellis_scene_fit"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_from(path_value: Any, *, parent: Path) -> Path:
    path = Path(str(path_value)).expanduser()
    return (path if path.is_absolute() else parent / path).resolve()


def flattened_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.to_geometry()
    else:
        mesh = loaded
    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        raise ValueError(f"candidate asset has no mesh: {path}")
    return mesh


def render_source_view_overlays(
    *,
    object_id: str,
    world_glb: Path,
    cameras_path: Path,
    view_manifest_path: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    payload = read_json(view_manifest_path)
    values = payload.get("views")
    if not isinstance(values, list):
        raise ValueError(f"view manifest has no views: {view_manifest_path}")
    mesh = flattened_mesh(world_glb)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        frame_id = str(value.get("frame_id") or "")
        observed_value = value.get("observed_mask")
        observed_path_value = observed_value.get("path") if isinstance(observed_value, dict) else observed_value
        observed_path = resolve_from(observed_path_value, parent=view_manifest_path.parent)
        if not observed_path.is_file():
            raise FileNotFoundError(observed_path)
        camera = load_camera(cameras_path, frame_id)
        with Image.open(observed_path) as source_mask:
            observed = np.asarray(source_mask.convert("L"), dtype=np.uint8) > 0
        rendered = project_mesh_silhouette(
            mesh,
            runtime_pivot=None,
            camera=camera,
            scene_transform=np.eye(4, dtype=np.float64),
        )
        source_rgb_value = value.get("source_rgb")
        source_rgb_path_value = source_rgb_value.get("path") if isinstance(source_rgb_value, dict) else None
        source_rgb_path = (
            resolve_from(source_rgb_path_value, parent=view_manifest_path.parent)
            if source_rgb_path_value
            else None
        )
        if source_rgb_path is not None and source_rgb_path.is_file():
            with Image.open(source_rgb_path) as source_image:
                overlay = make_overlay(source_image, observed, rendered)
        else:
            overlay = make_overlay(None, observed, rendered)
        output_path = output_dir / f"{object_id}_{frame_id}_overlay.png"
        overlay.save(output_path)
        intersection = int(np.count_nonzero(observed & rendered))
        union = int(np.count_nonzero(observed | rendered))
        records.append(
            {
                "frame_id": frame_id,
                "path": str(output_path),
                "sha256": sha256_file(output_path),
                "mask_iou": float(intersection / union) if union else 0.0,
                "observed_pixels": int(observed.sum()),
                "rendered_pixels": int(rendered.sum()),
                "source_rgb": str(source_rgb_path) if source_rgb_path is not None else None,
            }
        )
    return records


def fit_front_half_trellis_assets(
    *,
    evidence_path: Path,
    asset_root: Path,
    output_dir: Path,
    objects: set[str] | None = None,
    include_provisional: bool = False,
    cycles: int = 2,
    maximum_search_faces: int = 2500,
    minimum_mask_iou: float = 0.5,
    minimum_bbox_iou: float = 0.6,
    maximum_center_error_px: float = 30.0,
    maximum_scale_multiplier: float = 1.35,
    maximum_scale_anisotropy: float = 1.15,
    planar_maximum_scale_anisotropy: float = 8.0,
) -> dict[str, Any]:
    evidence_path = evidence_path.resolve()
    asset_root = asset_root.resolve()
    output_dir = output_dir.resolve()
    evidence = read_json(evidence_path)
    if evidence.get("kind") != "video2world.front_half_scene_fit_evidence":
        raise ValueError("evidence has an unexpected kind")
    camera_conversion = evidence.get("camera_conversion")
    if not isinstance(camera_conversion, dict):
        raise ValueError("evidence has no camera conversion")
    cameras_path = resolve_from(camera_conversion.get("path"), parent=evidence_path.parent)
    if not cameras_path.is_file():
        raise FileNotFoundError(cameras_path)
    values = evidence.get("objects")
    if not isinstance(values, list):
        raise ValueError("evidence has no objects")
    wanted = objects or set()
    records: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in values:
        if not isinstance(item, dict):
            continue
        object_id = str(item.get("object_id") or "")
        if not object_id or (wanted and object_id not in wanted):
            continue
        category = str(item.get("category") or "")
        evidence_status = str(item.get("status") or "")
        is_provisional = evidence_status == "single_view_scene_fit_evidence_provisional"
        record: dict[str, Any] = {
            "object_id": object_id,
            "category": category,
            "evidence_status": evidence_status,
            "front_back_audit_required": bool(item.get("front_back_audit_required")),
        }
        if evidence_status != "multiview_scene_fit_evidence_ready" and not (
            include_provisional and is_provisional
        ):
            record["status"] = "skipped_insufficient_scene_fit_evidence"
            records.append(record)
            continue
        anchor = item.get("placement_anchor")
        view_manifest = item.get("view_manifest")
        if not isinstance(anchor, dict) or not isinstance(view_manifest, dict):
            record["status"] = "skipped_missing_placement_anchor_or_view_manifest"
            records.append(record)
            continue
        anchor_path = resolve_from(anchor.get("path"), parent=evidence_path.parent)
        view_manifest_path = resolve_from(view_manifest.get("path"), parent=evidence_path.parent)
        try:
            asset_path = find_asset(asset_root, object_id, None)
        except Exception as exc:
            record["status"] = "skipped_missing_generated_asset"
            record["reason"] = f"{type(exc).__name__}: {exc}"
            records.append(record)
            continue
        planar = category.lower() in PLANAR_CATEGORIES
        fit_dir = output_dir / "fits" / object_id
        try:
            receipt = fit_completed_object_to_scene(
                object_id=object_id,
                mesh_source=asset_path,
                object_anchor_ply=anchor_path,
                cameras_path=cameras_path,
                view_manifest_path=view_manifest_path,
                output_dir=fit_dir,
                placement_mode="observed_front_surface" if planar else "volume_center",
                initial_scale_mode="target_extents" if planar else "uniform",
                cycles=cycles,
                maximum_search_faces=maximum_search_faces,
                maximum_scale_multiplier=maximum_scale_multiplier,
                maximum_scale_anisotropy=(
                    planar_maximum_scale_anisotropy if planar else maximum_scale_anisotropy
                ),
                minimum_mask_iou=minimum_mask_iou,
                minimum_bbox_iou=minimum_bbox_iou,
                maximum_center_error_px=maximum_center_error_px,
            )
        except Exception as exc:
            record["status"] = "fit_failed"
            record["reason"] = f"{type(exc).__name__}: {exc}"
            records.append(record)
            continue
        receipt_path = fit_dir / "scene_fit_receipt.json"
        world_glb = fit_dir / f"{object_id}.scene-space.glb"
        overlays = render_source_view_overlays(
            object_id=object_id,
            world_glb=world_glb,
            cameras_path=cameras_path,
            view_manifest_path=view_manifest_path,
            output_dir=output_dir / "source_view_overlays" / object_id,
        )
        if not receipt["all_acceptance_gates_passed"]:
            status = "rejected_by_scene_fit_gates"
        elif planar:
            status = "held_for_planar_front_back_audit"
        elif is_provisional:
            status = "held_for_multiview_evidence"
        else:
            status = "fitted_candidate_not_materialized_into_scene"
        record.update(
            {
                "status": status,
                "asset": {"path": str(asset_path), "sha256": sha256_file(asset_path)},
                "scene_fit_receipt": {"path": str(receipt_path), "sha256": sha256_file(receipt_path)},
                "world_baked_glb": {"path": str(world_glb), "sha256": sha256_file(world_glb)},
                "all_acceptance_gates_passed": bool(receipt["all_acceptance_gates_passed"]),
                "acceptance_gates": receipt["acceptance_gates"],
                "overlays": overlays,
            }
        )
        records.append(record)
    report = {
        "schema_version": 1,
        "kind": KIND,
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "front_half_object_completion_placement_pre_qa_pre_scene_graph",
        "claim_scope": (
            "World-baked candidate assets and source-view silhouette overlays only. This report does "
            "not remove source point clouds, accept planar front/back orientation, or mutate the scene."
        ),
        "evidence": {"path": str(evidence_path), "sha256": sha256_file(evidence_path)},
        "asset_root": str(asset_root),
        "parameters": {
            "objects": sorted(wanted),
            "include_provisional": include_provisional,
            "cycles": cycles,
            "maximum_search_faces": maximum_search_faces,
            "minimum_mask_iou": minimum_mask_iou,
            "minimum_bbox_iou": minimum_bbox_iou,
            "maximum_center_error_px": maximum_center_error_px,
            "maximum_scale_multiplier": maximum_scale_multiplier,
            "maximum_scale_anisotropy": maximum_scale_anisotropy,
            "planar_maximum_scale_anisotropy": planar_maximum_scale_anisotropy,
        },
        "objects": records,
    }
    report_path = output_dir / "front_half_scene_fit_manifest.json"
    write_json(report_path, report)
    return {"report": str(report_path), "objects": records}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--objects", nargs="*")
    parser.add_argument("--include-provisional", action="store_true")
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--maximum-search-faces", type=int, default=2500)
    parser.add_argument("--minimum-mask-iou", type=float, default=0.5)
    parser.add_argument("--minimum-bbox-iou", type=float, default=0.6)
    parser.add_argument("--maximum-center-error-px", type=float, default=30.0)
    parser.add_argument("--maximum-scale-multiplier", type=float, default=1.35)
    parser.add_argument("--maximum-scale-anisotropy", type=float, default=1.15)
    parser.add_argument("--planar-maximum-scale-anisotropy", type=float, default=8.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = fit_front_half_trellis_assets(
        evidence_path=args.evidence.expanduser(),
        asset_root=args.asset_root.expanduser(),
        output_dir=args.output_dir.expanduser(),
        objects=set(args.objects or []),
        include_provisional=args.include_provisional,
        cycles=args.cycles,
        maximum_search_faces=args.maximum_search_faces,
        minimum_mask_iou=args.minimum_mask_iou,
        minimum_bbox_iou=args.minimum_bbox_iou,
        maximum_center_error_px=args.maximum_center_error_px,
        maximum_scale_multiplier=args.maximum_scale_multiplier,
        maximum_scale_anisotropy=args.maximum_scale_anisotropy,
        planar_maximum_scale_anisotropy=args.planar_maximum_scale_anisotropy,
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
