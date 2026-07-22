#!/usr/bin/env python3
"""Audit one fitted bed and independently fitted pillows in a calibrated view."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from qa_scene_fit_relations import (
    depth_bins,
    evaluate_depth_order,
    load_camera,
    mask_bins,
    sha256_file,
    write_json,
)
from refine_scene_fit_silhouette import project_mesh_silhouette


def resolve_path(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def transform_points(vertices: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((vertices, np.ones(len(vertices), dtype=np.float64)))
    transformed = np.einsum("ni,ji->nj", homogeneous, matrix)
    return transformed[:, :3] / transformed[:, 3, None]


def transformed_scene_geometry(
    scene: trimesh.Scene,
    geometry_names: set[str],
    world_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    vertex_offset = 0
    for node_name in scene.graph.nodes_geometry:
        node_matrix, geometry_name = scene.graph.get(node_name)
        if geometry_name not in geometry_names:
            continue
        geometry = scene.geometry[geometry_name]
        item_vertices = transform_points(
            np.asarray(geometry.vertices, dtype=np.float64),
            world_matrix @ np.asarray(node_matrix, dtype=np.float64),
        )
        item_faces = np.asarray(geometry.faces, dtype=np.int64) + vertex_offset
        vertices.append(item_vertices)
        faces.append(item_faces)
        vertex_offset += len(item_vertices)
    if not vertices:
        raise ValueError(f"none of the requested geometries exist: {sorted(geometry_names)}")
    return np.vstack(vertices), np.vstack(faces)


def allowed_penetration(
    thickness: float,
    absolute_limit: float,
    thickness_fraction: float,
) -> float:
    if thickness <= 0 or absolute_limit < 0 or thickness_fraction < 0:
        raise ValueError("penetration thresholds and thickness must be positive")
    return min(absolute_limit, thickness_fraction * thickness)


def evaluate_contact_values(
    *,
    pillow_bottom_down: float,
    pillow_thickness: float,
    mattress_top_down: float,
    current_mattress_top_canonical_y: float,
    world_down_per_negative_canonical_y: float,
    absolute_limit: float,
    thickness_fraction: float,
    maximum_support_gap: float,
) -> dict[str, Any]:
    if world_down_per_negative_canonical_y <= 0:
        raise ValueError("canonical mattress lowering scale must be positive")
    penetration = max(0.0, pillow_bottom_down - mattress_top_down)
    support_gap = max(0.0, mattress_top_down - pillow_bottom_down)
    limit = allowed_penetration(pillow_thickness, absolute_limit, thickness_fraction)
    excess = max(0.0, penetration - limit)
    lowering_canonical_y = excess / world_down_per_negative_canonical_y
    penetration_passed = penetration <= limit + 1e-12
    support_gap_passed = support_gap <= maximum_support_gap + 1e-12
    passed = penetration_passed and support_gap_passed
    return {
        "status": "passed" if passed else "rejected",
        "gate": passed,
        "metrics": {
            "pillow_bottom_down_scene_units": pillow_bottom_down,
            "pillow_thickness_scene_units": pillow_thickness,
            "mattress_top_down_scene_units": mattress_top_down,
            "penetration_scene_units": penetration,
            "penetration_fraction_of_pillow_thickness": penetration / pillow_thickness,
            "excess_penetration_scene_units": excess,
            "support_gap_scene_units": support_gap,
        },
        "thresholds": {
            "absolute_maximum_penetration_scene_units": absolute_limit,
            "maximum_fraction_of_pillow_thickness": thickness_fraction,
            "allowed_penetration_scene_units": limit,
            "formula": "min(absolute_limit, thickness_fraction * pillow_thickness)",
            "maximum_support_gap_scene_units": maximum_support_gap,
        },
        "gates": {
            "penetration_within_limit": penetration_passed,
            "support_gap_within_limit": support_gap_passed,
        },
        "recommended_mattress_adjustment": {
            "required_downward_shift_scene_units": excess,
            "required_canonical_y_reduction": lowering_canonical_y,
            "maximum_mattress_top_canonical_y": (
                current_mattress_top_canonical_y - lowering_canonical_y
            ),
            "direction_semantics": (
                "Reduce bed-canonical +Y to move the mattress top downward in scene space."
            ),
        },
    }


def make_bed_floor_support_gate(
    bed_report: dict[str, Any],
    floor_support_report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": "passed",
        "support_contact": floor_support_report["support_contact"],
        "bed_scene_fit_all_acceptance_gates_passed": (bed_report["all_acceptance_gates_passed"]),
    }


def failed_gate_names(gates: Any) -> list[str]:
    if not isinstance(gates, dict):
        return []
    failed = []
    for name, value in gates.items():
        if value is False or (
            isinstance(value, dict)
            and (value.get("passed") is False or value.get("status") == "rejected")
        ):
            failed.append(str(name))
    return sorted(failed)


def review_rejection_summary(
    report: dict[str, Any],
    *,
    report_path: Path | None = None,
) -> str:
    details = []
    if report_path is not None:
        details.append(f"report={report_path}")
    for key in (
        "status",
        "decision",
        "promotion_allowed",
        "all_acceptance_gates_passed",
    ):
        if key in report:
            value = report[key]
            if isinstance(value, bool):
                value = str(value).lower()
            details.append(f"{key}={value}")
    failed_acceptance = failed_gate_names(report.get("acceptance_gates"))
    if failed_acceptance:
        details.append("failed_acceptance_gates=" + ",".join(failed_acceptance))
    failed_gates = failed_gate_names(report.get("gates"))
    if failed_gates:
        details.append("failed_gates=" + ",".join(failed_gates))
    blockers = report.get("promotion_blockers")
    if isinstance(blockers, list) and blockers:
        details.append("promotion_blockers=" + ",".join(str(item) for item in blockers))
    next_action = report.get("next_action")
    if isinstance(next_action, dict):
        action = next_action.get("action")
        blocker = next_action.get("blocker")
        if action:
            details.append(f"next_action={action}")
        if blocker:
            details.append(f"next_blocker={blocker}")
        blocking_groups = next_action.get("blocking_gate_groups")
        if isinstance(blocking_groups, list) and blocking_groups:
            groups = [group for group in blocking_groups if isinstance(group, str) and group]
            if groups:
                details.append(f"blocking_gate_groups={','.join(groups)}")
        failed_gates = next_action.get("failed_gates")
        if isinstance(failed_gates, list) and failed_gates:
            gates = [gate for gate in failed_gates if isinstance(gate, str) and gate]
            if gates:
                details.append(f"failed_gates={','.join(gates)}")
        failed_frame_ids = next_action.get("failed_frame_ids")
        if isinstance(failed_frame_ids, list) and failed_frame_ids:
            frames = [
                frame_id
                for frame_id in failed_frame_ids
                if isinstance(frame_id, str) and frame_id
            ]
            if frames:
                details.append(f"failed_frame_ids={','.join(frames)}")
        first_failed_frame_id = next_action.get("first_failed_frame_id")
        if isinstance(first_failed_frame_id, str) and first_failed_frame_id:
            details.append(f"first_failed_frame_id={first_failed_frame_id}")
        for key in (
            "failed_pair_ids",
            "failed_triplet_center_frame_ids",
            "not_evaluable_pair_ids",
            "not_evaluable_triplet_center_frame_ids",
        ):
            values = next_action.get(key)
            if isinstance(values, list) and values:
                kept = [value for value in values if isinstance(value, str) and value]
                if kept:
                    details.append(f"{key}={','.join(kept)}")
        no_support_frame_ids = next_action.get("no_support_frame_ids")
        if isinstance(no_support_frame_ids, list) and no_support_frame_ids:
            frames = [
                frame_id
                for frame_id in no_support_frame_ids
                if isinstance(frame_id, str) and frame_id
            ]
            if frames:
                details.append(f"no_support_frame_ids={','.join(frames)}")
        unresolved = next_action.get("unresolved_unobserved_pixels")
        if isinstance(unresolved, int):
            details.append(f"unresolved_unobserved_pixels={unresolved}")
    relations = report.get("relations")
    if isinstance(relations, dict):
        failed_relations = sorted(
            str(name)
            for name, relation in relations.items()
            if isinstance(relation, dict) and relation.get("status") != "passed"
        )
        if failed_relations:
            details.append("failed_relations=" + ",".join(failed_relations))
    return "[" + "; ".join(details) + "]"


def load_pillow(
    item: dict[str, Any],
    *,
    base: Path,
    camera: dict[str, Any],
) -> dict[str, Any]:
    object_id = str(item["object_id"])
    mesh_path = resolve_path(base, str(item["mesh"]))
    report_path = resolve_path(base, str(item["scene_fit_report"]))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("object_id") != object_id:
        raise ValueError(f"object id mismatch for {object_id}")
    mesh_hash = sha256_file(mesh_path)
    if report.get("mesh", {}).get("glb_sha256") != mesh_hash:
        raise ValueError(f"mesh hash mismatch for {object_id}")
    if report.get("all_acceptance_gates_passed") is not True:
        raise ValueError(
            f"scene fit is not accepted for {object_id} "
            + review_rejection_summary(report, report_path=report_path)
        )
    loaded = trimesh.load(mesh_path, force="scene")
    mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded
    pivot = np.asarray(report["baked_relative_transform"]["runtime_pivot"], dtype=np.float64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64) + pivot
    faces = np.asarray(mesh.faces, dtype=np.int64)
    silhouette = project_mesh_silhouette(
        vertices,
        faces,
        runtime_pivot=np.zeros(3, dtype=np.float64),
        camera=camera,
        image_y_axis="down",
    )
    return {
        "object_id": object_id,
        "vertices": vertices,
        "faces": faces,
        "silhouette": silhouette,
        "sources": {
            "mesh": {"path": str(mesh_path), "sha256": mesh_hash},
            "scene_fit_report": {
                "path": str(report_path),
                "sha256": sha256_file(report_path),
            },
        },
        "placement": {
            "asset_coordinates_baked": True,
            "runtime_pivot": pivot.tolist(),
            "runtime_rotation_euler_deg": [0.0, 0.0, 0.0],
            "runtime_scale": [1.0, 1.0, 1.0],
        },
    }


def make_projection_overlay(
    source: Image.Image,
    bed_mask: np.ndarray,
    headboard_mask: np.ndarray,
    pillows: list[dict[str, Any]],
) -> Image.Image:
    image = np.asarray(source.convert("RGB"), dtype=np.float32).copy()
    bed_only = bed_mask & ~headboard_mask
    image[bed_only] = image[bed_only] * 0.72 + np.asarray([35, 160, 255]) * 0.28
    image[headboard_mask] = image[headboard_mask] * 0.55 + np.asarray([255, 80, 70]) * 0.45
    colors = (
        np.asarray([0, 225, 150]),
        np.asarray([255, 205, 0]),
        np.asarray([225, 80, 255]),
    )
    for color, pillow in zip(colors, pillows, strict=True):
        mask = pillow["silhouette"]
        image[mask] = image[mask] * 0.45 + color * 0.55
    return Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))


def make_diagnostic_panel(
    size: tuple[int, int],
    contacts: dict[str, dict[str, Any]],
    headboard_relations: dict[str, dict[str, Any]],
) -> Image.Image:
    panel = Image.new("RGB", size, (24, 27, 31))
    draw = ImageDraw.Draw(panel)
    passed = all(item["gate"] for item in contacts.values()) and all(
        item["status"] == "passed" for item in headboard_relations.values()
    )
    status = "PASSED" if passed else "REJECTED"
    status_color = (105, 230, 155) if passed else (255, 105, 95)
    draw.text((24, 20), f"BED + PILLOW JOINT REVIEW: {status}", fill=status_color)
    y = 58
    for object_id, result in contacts.items():
        metrics = result["metrics"]
        limit = result["thresholds"]["allowed_penetration_scene_units"]
        recommendation = result["recommended_mattress_adjustment"]
        draw.text(
            (24, y),
            (
                f"{object_id}: penetration {metrics['penetration_scene_units']:.3f} "
                f"/ allowed {limit:.3f}; gap {metrics['support_gap_scene_units']:.3f}"
            ),
            fill=(205, 235, 210) if result["gate"] else (255, 205, 120),
        )
        draw.text(
            (40, y + 22),
            (
                "lower mattress >= "
                f"{recommendation['required_downward_shift_scene_units']:.3f} scene / "
                f"{recommendation['required_canonical_y_reduction']:.5f} canonical Y"
            ),
            fill=(220, 225, 232),
        )
        y += 56
    y += 8
    draw.text((24, y), "Headboard depth order", fill=(125, 205, 255))
    y += 28
    for object_id, result in headboard_relations.items():
        relation = result["relation"]
        if relation == "projection_disjoint":
            text = f"{object_id}: disjoint in source camera"
        else:
            minimum = result["metrics"]["minimum_rear_minus_front_depth_m"]
            if minimum is None:
                text = f"{object_id}: insufficient depth samples ({result['status']})"
            else:
                bin_size = result["sampling"]["selected_bin_size_px"]
                text = (
                    f"{object_id}: pillow ahead by >= {minimum:.3f} scene units "
                    f"at {bin_size}px bins"
                )
        draw.text((40, y), text, fill=(205, 235, 210))
        y += 24
    draw.text(
        (24, size[1] - 42),
        "No exact signed-volume claim: pillow surfaces are non-watertight.",
        fill=(165, 170, 180),
    )
    return panel


def contact_sheet(
    *,
    source: Image.Image,
    pbr_render: Image.Image | None,
    overlay: Image.Image,
    diagnostic: Image.Image,
) -> Image.Image:
    tile_size = (source.width // 2, source.height // 2)
    source_tile = source.convert("RGB").resize(tile_size, Image.Resampling.LANCZOS)
    if pbr_render is None:
        pbr_tile = Image.new("RGB", tile_size, (24, 27, 31))
        ImageDraw.Draw(pbr_tile).text((20, 20), "PBR render unavailable", fill="white")
    else:
        pbr = pbr_render.convert("RGBA").resize(tile_size, Image.Resampling.LANCZOS)
        pbr_tile = Image.alpha_composite(source_tile.convert("RGBA"), pbr).convert("RGB")
    overlay_tile = overlay.convert("RGB").resize(tile_size, Image.Resampling.LANCZOS)
    diagnostic_tile = diagnostic.convert("RGB").resize(tile_size, Image.Resampling.LANCZOS)
    sheet = Image.new("RGB", (tile_size[0] * 2, tile_size[1] * 2))
    sheet.paste(source_tile, (0, 0))
    sheet.paste(pbr_tile, (tile_size[0], 0))
    sheet.paste(overlay_tile, (0, tile_size[1]))
    sheet.paste(diagnostic_tile, (tile_size[0], tile_size[1]))
    return sheet


def evaluate_headboard_relation(
    *,
    pillow_vertices: np.ndarray,
    pillow_mask: np.ndarray,
    headboard_vertices: np.ndarray,
    headboard_mask: np.ndarray,
    camera: dict[str, Any],
    base_bin_size_px: int,
    fallback_multipliers: list[int],
    minimum_shared_bins: int,
    minimum_depth_margin: float,
) -> dict[str, Any]:
    overlap_pixels = int(np.count_nonzero(pillow_mask & headboard_mask))
    if overlap_pixels == 0:
        return {
            "status": "passed",
            "relation": "projection_disjoint",
            "metrics": {"full_projection_overlap_pixels": 0},
            "gates": {"headboard_projection_disjoint": True},
            "sampling": {
                "base_bin_size_px": base_bin_size_px,
                "selected_bin_size_px": None,
                "fallback_used": False,
                "attempts": [],
            },
        }

    attempts: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    last_result: dict[str, Any] | None = None
    for multiplier in fallback_multipliers:
        bin_size_px = base_bin_size_px * multiplier
        result = evaluate_depth_order(
            front_depth=depth_bins(pillow_vertices, camera, bin_size_px),
            rear_depth=depth_bins(headboard_vertices, camera, bin_size_px),
            front_mask_bins=mask_bins(pillow_mask, bin_size_px),
            rear_mask_bins=mask_bins(headboard_mask, bin_size_px),
            minimum_shared_bins=minimum_shared_bins,
            minimum_depth_margin=minimum_depth_margin,
        )
        last_result = result
        attempts.append(
            {
                "bin_size_px": bin_size_px,
                "shared_depth_bins": result["metrics"]["shared_depth_bins"],
                "status": result["status"],
            }
        )
        if result["metrics"]["shared_depth_bins"] >= minimum_shared_bins:
            selected = result
            break
    assert last_result is not None
    selected = last_result if selected is None else selected
    selected_bin_size = attempts[-1]["bin_size_px"]
    selected["relation"] = "pillow_before_headboard"
    selected["full_projection_overlap_pixels"] = overlap_pixels
    selected["sampling"] = {
        "base_bin_size_px": base_bin_size_px,
        "selected_bin_size_px": selected_bin_size,
        "fallback_used": selected_bin_size != base_bin_size_px,
        "attempts": attempts,
        "policy": (
            "Use the finest configured vertex-depth binning with enough shared samples; "
            "fail closed if no configured resolution reaches the minimum."
        ),
    }
    return selected


def review_joint(
    spec_path: Path,
    output_dir: Path,
    pbr_render_path: Path | None = None,
) -> dict[str, Any]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    base = spec_path.parent
    frame_id = str(spec["frame_id"])
    cameras_path = resolve_path(base, str(spec["cameras"]))
    source_frame_path = resolve_path(base, str(spec["source_frame"]))
    pairwise_path = resolve_path(base, str(spec["pairwise_review"]))
    bed = spec["bed"]
    bed_mesh_path = resolve_path(base, str(bed["mesh"]))
    bed_report_path = resolve_path(base, str(bed["scene_fit_report"]))
    floor_support_report_path = resolve_path(
        base, str(bed.get("floor_support_scene_fit_report", bed["scene_fit_report"]))
    )
    support_adjustment_path = (
        resolve_path(base, str(bed["support_adjustment_receipt"]))
        if bed.get("support_adjustment_receipt")
        else None
    )
    camera = load_camera(cameras_path, frame_id)
    bed_report = json.loads(bed_report_path.read_text(encoding="utf-8"))
    floor_support_report = json.loads(floor_support_report_path.read_text(encoding="utf-8"))
    support_adjustment = (
        json.loads(support_adjustment_path.read_text(encoding="utf-8"))
        if support_adjustment_path is not None
        else None
    )
    pairwise_review = json.loads(pairwise_path.read_text(encoding="utf-8"))
    bed_mesh_hash = sha256_file(bed_mesh_path)
    if bed_report.get("sources", {}).get("mesh", {}).get("sha256") != bed_mesh_hash:
        raise ValueError("bed mesh hash does not match its scene-fit report")
    if bed_report.get("all_acceptance_gates_passed") is not True:
        raise ValueError(
            "bed scene fit is not accepted "
            + review_rejection_summary(bed_report, report_path=bed_report_path)
        )
    if floor_support_report.get("support_contact", {}).get("passed") is not True:
        raise ValueError(
            "bed floor contact is not accepted "
            + review_rejection_summary(floor_support_report, report_path=floor_support_report_path)
        )
    if support_adjustment is not None:
        if support_adjustment.get("all_acceptance_gates_passed") is not True:
            raise ValueError(
                "bed support adjustment technical gates are not accepted "
                + review_rejection_summary(
                    support_adjustment,
                    report_path=support_adjustment_path,
                )
            )
        adjusted_mesh = support_adjustment.get("output", {}).get("unified_pbr_glb", {})
        if adjusted_mesh.get("sha256") != bed_mesh_hash:
            raise ValueError("bed support adjustment mesh hash mismatch")
    if pairwise_review.get("status") != "passed" or not pairwise_review.get("promotion_allowed"):
        raise ValueError(
            "pillow pairwise review must pass before joint review "
            + review_rejection_summary(pairwise_review, report_path=pairwise_path)
        )

    bed_matrix = np.asarray(bed_report["runtime_transform"]["matrix_row_major"], dtype=np.float64)
    if bed_matrix.shape != (4, 4):
        raise ValueError("bed runtime transform must be a full 4x4 matrix")
    inverse_bed_matrix = np.linalg.inv(bed_matrix)
    down_normal = np.asarray(
        floor_support_report["support_alignment"]["support_down_normal"],
        dtype=np.float64,
    )
    down_normal /= np.linalg.norm(down_normal)
    down_per_negative_canonical_y = -float(np.dot(bed_matrix[:3, 1], down_normal))
    if down_per_negative_canonical_y <= 0:
        raise ValueError("bed canonical +Y does not point away from support")

    loaded_bed = trimesh.load(bed_mesh_path, force="scene")
    if not isinstance(loaded_bed, trimesh.Scene):
        raise ValueError("bed component assembly must be a GLB scene")
    mattress_names = {str(value) for value in bed["mattress_geometries"]}
    headboard_names = {str(value) for value in bed["headboard_geometries"]}
    all_bed_names = set(loaded_bed.geometry)
    mattress_world, _mattress_faces = transformed_scene_geometry(
        loaded_bed, mattress_names, bed_matrix
    )
    headboard_world, headboard_faces = transformed_scene_geometry(
        loaded_bed, headboard_names, bed_matrix
    )
    bed_world, bed_faces = transformed_scene_geometry(loaded_bed, all_bed_names, bed_matrix)
    mattress_canonical = transform_points(mattress_world, inverse_bed_matrix)
    mattress_down = np.einsum("ni,i->n", mattress_world, down_normal)

    parameters = spec["parameters"]
    footprint_low, footprint_high = map(float, parameters["footprint_quantiles"])
    thickness_low, thickness_high = map(float, parameters["thickness_quantiles"])
    top_candidate_quantile = float(parameters["mattress_top_candidate_quantile"])
    top_down_quantile = float(parameters["mattress_top_down_quantile"])
    footprint_margin = float(parameters["footprint_margin_canonical"])
    pillow_bottom_quantile = float(parameters["pillow_bottom_quantile"])
    absolute_limit = float(parameters["absolute_maximum_penetration_scene_units"])
    thickness_fraction = float(parameters["maximum_penetration_fraction_of_thickness"])
    maximum_support_gap = float(parameters["maximum_support_gap_scene_units"])
    bin_size_px = int(parameters["depth_bin_size_px"])
    minimum_headboard_bins = int(parameters["minimum_headboard_shared_bins"])
    minimum_headboard_margin = float(parameters["minimum_headboard_depth_margin"])
    headboard_bin_fallback_multipliers = [
        int(value) for value in parameters["headboard_bin_fallback_multipliers"]
    ]
    current_mattress_top_canonical_y = float(
        np.quantile(mattress_canonical[:, 1], top_candidate_quantile)
    )
    top_candidates = mattress_canonical[:, 1] >= current_mattress_top_canonical_y - 1e-9

    pillows = [load_pillow(item, base=base, camera=camera) for item in spec["pillows"]]
    contacts: dict[str, dict[str, Any]] = {}
    for pillow in pillows:
        pillow_canonical = transform_points(pillow["vertices"], inverse_bed_matrix)
        low = np.quantile(pillow_canonical[:, (0, 2)], footprint_low, axis=0)
        high = np.quantile(pillow_canonical[:, (0, 2)], footprint_high, axis=0)
        local = top_candidates & np.all(
            (mattress_canonical[:, (0, 2)] >= low - footprint_margin)
            & (mattress_canonical[:, (0, 2)] <= high + footprint_margin),
            axis=1,
        )
        if int(np.count_nonzero(local)) < 3:
            raise ValueError(f"insufficient local mattress samples for {pillow['object_id']}")
        pillow_down = np.einsum("ni,i->n", pillow["vertices"], down_normal)
        bottom = float(np.quantile(pillow_down, pillow_bottom_quantile))
        thickness = float(
            np.quantile(pillow_down, thickness_high) - np.quantile(pillow_down, thickness_low)
        )
        top_down = float(np.quantile(mattress_down[local], top_down_quantile))
        contact = evaluate_contact_values(
            pillow_bottom_down=bottom,
            pillow_thickness=thickness,
            mattress_top_down=top_down,
            current_mattress_top_canonical_y=current_mattress_top_canonical_y,
            world_down_per_negative_canonical_y=down_per_negative_canonical_y,
            absolute_limit=absolute_limit,
            thickness_fraction=thickness_fraction,
            maximum_support_gap=maximum_support_gap,
        )
        contact["local_mattress_sampling"] = {
            "sample_count": int(np.count_nonzero(local)),
            "footprint_low_xz_canonical": low.tolist(),
            "footprint_high_xz_canonical": high.tolist(),
            "footprint_margin_canonical": footprint_margin,
        }
        contacts[pillow["object_id"]] = contact

    headboard_mask = project_mesh_silhouette(
        headboard_world,
        headboard_faces,
        runtime_pivot=np.zeros(3),
        camera=camera,
        image_y_axis="down",
    )
    headboard_relations: dict[str, dict[str, Any]] = {}
    for pillow in pillows:
        relation = evaluate_headboard_relation(
            pillow_vertices=pillow["vertices"],
            pillow_mask=pillow["silhouette"],
            headboard_vertices=headboard_world,
            headboard_mask=headboard_mask,
            camera=camera,
            base_bin_size_px=bin_size_px,
            fallback_multipliers=headboard_bin_fallback_multipliers,
            minimum_shared_bins=minimum_headboard_bins,
            minimum_depth_margin=minimum_headboard_margin,
        )
        headboard_relations[pillow["object_id"]] = relation

    output_dir.mkdir(parents=True, exist_ok=False)
    source_image = Image.open(source_frame_path).convert("RGB")
    bed_mask = project_mesh_silhouette(
        bed_world,
        bed_faces,
        runtime_pivot=np.zeros(3),
        camera=camera,
        image_y_axis="down",
    )
    overlay = make_projection_overlay(source_image, bed_mask, headboard_mask, pillows)
    overlay_path = output_dir / "joint_projection_overlay.png"
    overlay.save(overlay_path)
    for pillow in pillows:
        Image.fromarray(pillow["silhouette"].astype(np.uint8) * 255).save(
            output_dir / f"{pillow['object_id']}.full_silhouette.png"
        )
    Image.fromarray(bed_mask.astype(np.uint8) * 255).save(
        output_dir / "sam3_bed_01.full_silhouette.png"
    )
    Image.fromarray(headboard_mask.astype(np.uint8) * 255).save(
        output_dir / "sam3_bed_01.headboard_silhouette.png"
    )

    persisted_pbr_path: Path | None = None
    pbr_image: Image.Image | None = None
    if pbr_render_path is not None:
        persisted_pbr_path = output_dir / "joint_pbr_source_camera.png"
        shutil.copy2(pbr_render_path, persisted_pbr_path)
        pbr_image = Image.open(persisted_pbr_path).convert("RGBA")
    diagnostic = make_diagnostic_panel(source_image.size, contacts, headboard_relations)
    diagnostic_path = output_dir / "joint_contact_diagnostic.png"
    diagnostic.save(diagnostic_path)
    contact_sheet_path = output_dir / "joint_contact_sheet.png"
    contact_sheet(
        source=source_image,
        pbr_render=pbr_image,
        overlay=overlay,
        diagnostic=diagnostic,
    ).save(contact_sheet_path)

    contact_passed = all(item["gate"] for item in contacts.values())
    headboard_passed = all(item["status"] == "passed" for item in headboard_relations.values())
    passed = contact_passed and headboard_passed
    blockers = [
        (
            f"{object_id} mattress penetration "
            f"{result['metrics']['penetration_scene_units']:.9f} exceeds allowed "
            f"{result['thresholds']['allowed_penetration_scene_units']:.9f}"
        )
        for object_id, result in contacts.items()
        if not result["gates"]["penetration_within_limit"]
    ]
    blockers.extend(
        (
            f"{object_id} mattress support gap "
            f"{result['metrics']['support_gap_scene_units']:.9f} exceeds allowed "
            f"{result['thresholds']['maximum_support_gap_scene_units']:.9f}"
        )
        for object_id, result in contacts.items()
        if not result["gates"]["support_gap_within_limit"]
    )
    blockers.extend(
        f"{object_id} headboard depth-order review rejected"
        for object_id, result in headboard_relations.items()
        if result["status"] != "passed"
    )
    strictest = min(
        contacts.items(),
        key=lambda item: item[1]["recommended_mattress_adjustment"][
            "maximum_mattress_top_canonical_y"
        ],
    )
    script_path = Path(__file__).resolve()
    relation_script_path = Path(__file__).with_name("qa_scene_fit_relations.py").resolve()
    silhouette_script_path = Path(__file__).with_name("refine_scene_fit_silhouette.py").resolve()
    receipt = {
        "schema_version": 1,
        "kind": "video2world.bed_pillow_joint_review",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "passed" if passed else "rejected",
        "promotion_allowed": passed,
        "promotion_blockers": blockers,
        "frame_id": frame_id,
        "claim_scope": (
            "Bed full-4x4 replay, pillow baked-GLB-plus-pivot replay, robust local mattress "
            "support penetration, and source-camera headboard depth order. Pillow surfaces "
            "are non-watertight, so this is not an exact signed-volume boolean claim."
        ),
        "joint_asset_contract": {
            "bed": {
                "asset_coordinates_baked": False,
                "runtime_transform_matrix_row_major": bed_matrix.tolist(),
            },
            "pillows": {pillow["object_id"]: pillow["placement"] for pillow in pillows},
        },
        "bed_floor_support_gate": make_bed_floor_support_gate(bed_report, floor_support_report),
        "bed_support_adjustment_gate": (
            {
                "status": "passed",
                "all_acceptance_gates_passed": support_adjustment["all_acceptance_gates_passed"],
                "acceptance_gates": support_adjustment["acceptance_gates"],
                "policy": support_adjustment["policy"],
            }
            if support_adjustment is not None
            else None
        ),
        "pillow_pairwise_gate": {
            "status": "passed",
            "review_status": pairwise_review["status"],
            "promotion_allowed": pairwise_review["promotion_allowed"],
            "relations": pairwise_review["relations"],
            "claim_scope": pairwise_review["claim_scope"],
        },
        "mattress_contacts": contacts,
        "headboard_relations": headboard_relations,
        "joint_recommendation": {
            "strictest_object_id": strictest[0],
            "maximum_mattress_top_canonical_y_for_all_pillows": strictest[1][
                "recommended_mattress_adjustment"
            ]["maximum_mattress_top_canonical_y"],
            "required_downward_shift_scene_units_for_all_pillows": strictest[1][
                "recommended_mattress_adjustment"
            ]["required_downward_shift_scene_units"],
            "required_canonical_y_reduction_for_all_pillows": strictest[1][
                "recommended_mattress_adjustment"
            ]["required_canonical_y_reduction"],
            "bed_fit_was_modified": False,
        },
        "parameters": parameters,
        "sources": {
            "spec": {"path": str(spec_path), "sha256": sha256_file(spec_path)},
            "script": {"path": str(script_path), "sha256": sha256_file(script_path)},
            "relation_helpers": {
                "path": str(relation_script_path),
                "sha256": sha256_file(relation_script_path),
            },
            "silhouette_renderer": {
                "path": str(silhouette_script_path),
                "sha256": sha256_file(silhouette_script_path),
            },
            "cameras": {"path": str(cameras_path), "sha256": sha256_file(cameras_path)},
            "source_frame": {
                "path": str(source_frame_path),
                "sha256": sha256_file(source_frame_path),
            },
            "pairwise_review": {
                "path": str(pairwise_path),
                "sha256": sha256_file(pairwise_path),
            },
            "bed": {
                "mesh": {"path": str(bed_mesh_path), "sha256": bed_mesh_hash},
                "scene_fit_report": {
                    "path": str(bed_report_path),
                    "sha256": sha256_file(bed_report_path),
                },
                "floor_support_scene_fit_report": {
                    "path": str(floor_support_report_path),
                    "sha256": sha256_file(floor_support_report_path),
                },
                "support_adjustment_receipt": (
                    {
                        "path": str(support_adjustment_path),
                        "sha256": sha256_file(support_adjustment_path),
                    }
                    if support_adjustment_path is not None
                    else None
                ),
            },
            "pillows": {pillow["object_id"]: pillow["sources"] for pillow in pillows},
            "pbr_render_input": (
                {"path": str(pbr_render_path), "sha256": sha256_file(pbr_render_path)}
                if pbr_render_path is not None
                else None
            ),
        },
        "outputs": {
            "projection_overlay": {
                "path": str(overlay_path),
                "sha256": sha256_file(overlay_path),
            },
            "contact_diagnostic": {
                "path": str(diagnostic_path),
                "sha256": sha256_file(diagnostic_path),
            },
            "contact_sheet": {
                "path": str(contact_sheet_path),
                "sha256": sha256_file(contact_sheet_path),
            },
            "pbr_source_camera": (
                {
                    "path": str(persisted_pbr_path),
                    "sha256": sha256_file(persisted_pbr_path),
                }
                if persisted_pbr_path is not None
                else None
            ),
        },
    }
    write_json(output_dir / "bed_pillow_joint_review.json", receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pbr-render", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = review_joint(
        args.spec.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        args.pbr_render.expanduser().resolve() if args.pbr_render else None,
    )
    print(json.dumps(receipt, ensure_ascii=True, indent=2))
    return 0 if receipt["promotion_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
