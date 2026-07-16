#!/usr/bin/env python3
"""Materialize the verified Bedroom 4 production world as a local WorldManifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from video2world.hashing import atomic_write_json, digest_json, digest_path, sha256_file
from video2world.models import (
    AssetRef,
    Bounds3D,
    CoordinateSystem,
    DescriptionEvidence,
    GateRecord,
    InteractionPolicy,
    LocalizedText,
    ObjectEvidence,
    OrientedBounds3D,
    ProvenanceRecord,
    QualityGates,
    Relation,
    SceneLayer,
    SceneTransform,
    WorldManifest,
    WorldObject,
)
from video2world.query import query_world
from video2world.validation import validate_world_manifest

FRAME_ID = "pgsr_native_shared_frame_20260714"
WORLD_ID = "bedroom_4"
COLLIDABLE_IDS = (
    "sam3_nightstand_01",
    "sam3_nightstand_02",
    "sam3_plant_01",
    "sam3_plant_02",
)
VISUAL_COMPONENT_IDS = (*COLLIDABLE_IDS, "sam3_pillow_01")


@dataclass(frozen=True)
class MaterializedFile:
    path: Path
    mode: str
    source: str


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _matches(path: Path, expected_sha256: str, expected_size: int) -> bool:
    if not path.is_file() or path.stat().st_size != expected_size:
        return False
    return sha256_file(path)[0] == expected_sha256


def materialize_file(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> MaterializedFile:
    source = source.expanduser().resolve()
    require(source.is_file(), f"source asset does not exist: {source}")
    source_sha256, source_size = sha256_file(source)
    expected_sha256 = expected_sha256 or source_sha256
    expected_size = expected_size or source_size
    require(source_sha256 == expected_sha256, f"source sha256 mismatch: {source}")
    require(source_size == expected_size, f"source size mismatch: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    if _matches(destination, expected_sha256, expected_size):
        mode = "reused_hardlink" if os.path.samefile(source, destination) else "reused_verified"
        return MaterializedFile(destination, mode, str(source))

    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    mode = "hardlink"
    try:
        os.link(source, temporary)
    except OSError:
        mode = "copy2_fallback"
        shutil.copy2(source, temporary)
    require(
        _matches(temporary, expected_sha256, expected_size),
        f"materialized asset failed hash verification: {temporary}",
    )
    os.replace(temporary, destination)
    return MaterializedFile(destination, mode, str(source))


def merge_chunk_parts(
    parts: list[dict[str, Any]],
    chunk_dir: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_size: int,
) -> MaterializedFile:
    if _matches(destination, expected_sha256, expected_size):
        return MaterializedFile(destination, "reused_verified", str(chunk_dir))
    require(parts, f"chunk list is empty for {destination.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    merged_digest = hashlib.sha256()
    merged_size = 0
    try:
        with temporary.open("wb") as output:
            for index, part in enumerate(parts):
                chunk = chunk_dir / Path(str(part["url"])).name
                expected_chunk_size = int(part["size"])
                expected_chunk_sha = str(part["sha256"])
                require(chunk.is_file(), f"missing chunk {index}: {chunk}")
                chunk_sha, chunk_size = sha256_file(chunk)
                require(chunk_size == expected_chunk_size, f"chunk size mismatch: {chunk}")
                require(chunk_sha == expected_chunk_sha, f"chunk sha256 mismatch: {chunk}")
                with chunk.open("rb") as stream:
                    while block := stream.read(4 * 1024 * 1024):
                        output.write(block)
                        merged_digest.update(block)
                        merged_size += len(block)
        require(merged_size == expected_size, f"merged size mismatch: {destination}")
        require(
            merged_digest.hexdigest() == expected_sha256,
            f"merged sha256 mismatch: {destination}",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return MaterializedFile(destination, "verified_chunk_merge", str(chunk_dir))


def relative_uri(path: Path, manifest_path: Path) -> str:
    return os.path.relpath(path.resolve(), manifest_path.parent.resolve())


def asset_ref(
    materialized: MaterializedFile,
    manifest_path: Path,
    *,
    role: str,
    media_type: str,
    provenance: dict[str, Any] | None = None,
) -> AssetRef:
    digest = digest_path(materialized.path)
    details = {
        "materialization_strategy": "hardlink_preferred_copy_fallback_or_verified_chunk_merge",
        "materialized_from": materialized.source,
    }
    details.update(provenance or {})
    return AssetRef(
        uri=relative_uri(materialized.path, manifest_path),
        sha256=digest.sha256,
        size_bytes=digest.size_bytes,
        media_type=media_type,
        role=role,
        status="validated",
        provenance=details,
    )


def existing_asset_ref(
    path: Path,
    manifest_path: Path,
    *,
    role: str,
    media_type: str,
    provenance: dict[str, Any] | None = None,
) -> AssetRef:
    return asset_ref(
        MaterializedFile(path.resolve(), "project_evidence", str(path.resolve())),
        manifest_path,
        role=role,
        media_type=media_type,
        provenance=provenance,
    )


def bounds_from_record(record: dict[str, Any]) -> Bounds3D:
    return Bounds3D(
        frame_id=FRAME_ID,
        minimum=tuple(float(value) for value in record["min"]),
        maximum=tuple(float(value) for value in record["max"]),
    )


def translation_transform(object_id: str, pivot: list[float]) -> SceneTransform:
    x, y, z = (float(value) for value in pivot)
    # Matrix arrays follow Three.js/GLTF column-major storage, with translation in 12..14.
    matrix = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, x, y, z, 1.0)
    return SceneTransform(
        from_frame=f"{object_id}_web_lod_local",
        to_frame=FRAME_ID,
        matrix=matrix,
        pivot_scene=(x, y, z),
    )


def identity_transform(object_id: str, pivot: list[float]) -> SceneTransform:
    return SceneTransform(
        from_frame=f"{object_id}_pgsr_shared_frame",
        to_frame=FRAME_ID,
        matrix=(
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ),
        pivot_scene=tuple(float(value) for value in pivot),
    )


def description_from_cognition(
    cognition_by_id: dict[str, dict[str, Any]], object_id: str
) -> DescriptionEvidence:
    record = cognition_by_id.get(object_id)
    require(record is not None, f"missing cognition output for {object_id}")
    require(record.get("status") == "passed", f"cognition did not pass for {object_id}")
    return DescriptionEvidence.model_validate(record["description_evidence"])


def gate_record(
    report_uri: str,
    metrics: dict[str, float | int | str | bool | None],
) -> GateRecord:
    return GateRecord(status="passed", report_uri=report_uri, metrics=metrics)


def interactive_quality_gates(
    object_id: str,
    item: dict[str, Any],
    browser_item: dict[str, Any],
    manifest_path: Path,
) -> QualityGates:
    browser_report = relative_uri(manifest_path.parent / "browser-qa.json", manifest_path)
    production_report = relative_uri(
        manifest_path.parent / "production-bundle-report.json", manifest_path
    )
    cognition_report = relative_uri(
        manifest_path.parent / "cognition/outputs/cognition_manifest.json", manifest_path
    )
    collision_gate = item["collision"]["gate"]
    direct_ray = browser_item["directRayHit"]
    require(collision_gate["status"] == "passed", f"collision gate failed for {object_id}")
    require(direct_ray.get("matched") is True, f"direct collision ray missed {object_id}")
    return QualityGates(
        file=gate_record(
            production_report,
            {
                "visual_sha_verified": True,
                "collider_sha_verified": True,
                "render_mesh_sha_verified": True,
            },
        ),
        semantic=gate_record(
            cognition_report,
            {
                "cognition_passed": True,
                "source_anchor_points": int(item["sourceAnchor"]["pointCount"]),
            },
        ),
        alignment=gate_record(
            browser_report,
            {
                "center_error": float(browser_item["centerError"]),
                "maximum_relative_size_error": float(browser_item["maximumRelativeSizeError"]),
            },
        ),
        collision=gate_record(
            browser_report,
            {
                "faces": int(browser_item["collisionFaces"]),
                "direct_ray_matched": True,
            },
        ),
        visual=gate_record(
            browser_report,
            {
                "browser_overlay_review": collision_gate["browserOverlayReview"],
                "web_lod_gaussians": int(item["visual"]["vertexCount"]),
            },
        ),
    )


def materialize_interactive_object(
    item: dict[str, Any],
    browser_item: dict[str, Any],
    cognition_by_id: dict[str, dict[str, Any]],
    world_dir: Path,
    assets_dir: Path,
    manifest_path: Path,
    materialization_log: list[dict[str, str]],
) -> WorldObject:
    object_id = str(item["id"])
    visual_record = item["visual"]
    collision = item["collision"]
    collider_record = collision["asset"]
    render_record = collision["renderAsset"]
    object_dir = assets_dir / "objects" / object_id

    visual_file = merge_chunk_parts(
        visual_record["parts"],
        world_dir / "chunks",
        object_dir / visual_record["fileName"],
        expected_sha256=visual_record["sha256"],
        expected_size=int(visual_record["size"]),
    )
    collider_file = materialize_file(
        world_dir / "colliders" / collider_record["fileName"],
        object_dir / collider_record["fileName"],
        expected_sha256=collider_record["sha256"],
        expected_size=int(collider_record["size"]),
    )
    render_file = materialize_file(
        world_dir / "render-meshes" / render_record["fileName"],
        object_dir / render_record["fileName"],
        expected_sha256=render_record["sha256"],
        expected_size=int(render_record["size"]),
    )
    source_anchor_record = item["sourceAnchor"]
    source_anchor_file = materialize_file(
        world_dir / "evidence" / source_anchor_record["fileName"],
        assets_dir / "evidence" / f"{object_id}.source-anchor.ply",
        expected_sha256=source_anchor_record["sha256"],
    )
    for materialized in (visual_file, collider_file, render_file, source_anchor_file):
        materialization_log.append(
            {
                "path": str(materialized.path),
                "mode": materialized.mode,
                "source": materialized.source,
            }
        )

    visual = asset_ref(
        visual_file,
        manifest_path,
        role="interactive_object_gaussian_web_lod",
        media_type="application/vnd.graphdeco.gaussian-ply",
        provenance={
            "vertex_count": int(visual_record["vertexCount"]),
            "source_vertex_count": int(visual_record["sourceVertexCount"]),
            "source_sha256": visual_record["sourceSha256"],
            "selection_method": visual_record["selectionMethod"],
            "placement_baked_into_gaussian_centers_and_covariances": True,
        },
    )
    collider = asset_ref(
        collider_file,
        manifest_path,
        role="interactive_object_collision_proxy",
        media_type="model/gltf-binary",
        provenance={
            "faces": int(collider_record["faces"]),
            "vertices": int(collider_record["vertices"]),
            "object_local_matrix_column_major": collision["objectLocalMatrix"],
            "simplification": collider_record["simplification"],
        },
    )
    render_mesh = asset_ref(
        render_file,
        manifest_path,
        role="interactive_object_render_mesh",
        media_type="model/gltf-binary",
        provenance={
            "source_sha256": collision["source"]["sha256"],
            "source_faces": int(collision["source"]["faces"]),
            "object_local_matrix_column_major": collision["objectLocalMatrix"],
        },
    )
    source_point_cloud = asset_ref(
        source_anchor_file,
        manifest_path,
        role="source_3d_mask_object_point_cloud",
        media_type="model/ply",
        provenance={
            "point_count": int(source_anchor_record["pointCount"]),
            "bounds_method": "robust_quantile_0.01_0.99",
        },
    )
    bbox = bounds_from_record(item["bbox"])
    aliases = [str(alias) for alias in item.get("aliases", [])]
    zh_name = (
        f"床头柜 {object_id[-2:]}" if item["category"] == "nightstand" else f"植物 {object_id[-2:]}"
    )
    return WorldObject(
        id=object_id,
        source_run_id="holi-fresh+embodiedgen-v2",
        scoped_id=f"{WORLD_ID}::holi-fresh+embodiedgen-v2::{object_id}",
        name=LocalizedText(zh=zh_name, en=str(item["label"])),
        category=str(item["category"]),
        aliases=aliases,
        description=description_from_cognition(cognition_by_id, object_id),
        evidence=ObjectEvidence(source_point_cloud=source_point_cloud),
        bbox_scene=bbox,
        visual=visual,
        render_mesh=render_mesh,
        collider=collider,
        transform_scene_from_asset=translation_transform(object_id, item["placement"]["pivot"]),
        quality_gates=interactive_quality_gates(object_id, item, browser_item, manifest_path),
        interaction=InteractionPolicy(
            selectable=True,
            double_click_action="spin_360",
            drag_action="rotate_yaw",
            collision_enabled=True,
            physics_mode="kinematic",
        ),
    )


def build_parser(project_root: Path) -> argparse.ArgumentParser:
    example_root = project_root / "examples" / "bedroom4"
    world_dir = project_root / "web" / "public" / "worlds" / "bedroom4"
    holi_root = Path(
        os.environ.get(
            "VIDEO2WORLD_BEDROOM4_HOLI_ROOT",
            project_root / "external" / "holi-spatial-bedroom4",
        )
    )
    parser = argparse.ArgumentParser(
        description="Materialize the verified real Bedroom 4 world and validate it."
    )
    parser.add_argument(
        "--production-manifest",
        type=Path,
        default=example_root / "manifest.production.json",
    )
    parser.add_argument(
        "--production-report",
        type=Path,
        default=example_root / "production-bundle-report.json",
    )
    parser.add_argument("--browser-qa", type=Path, default=example_root / "browser-qa.json")
    parser.add_argument(
        "--cognition-manifest",
        type=Path,
        default=example_root / "cognition" / "outputs" / "cognition_manifest.json",
    )
    parser.add_argument(
        "--pillow-entity",
        type=Path,
        default=(
            example_root
            / "evidence"
            / "pillow_delta_20260716"
            / "accepted"
            / "pillow_scene_entity_candidate.json"
        ),
    )
    parser.add_argument("--world-dir", type=Path, default=world_dir)
    parser.add_argument(
        "--semantic-ply",
        type=Path,
        default=holi_root / "video2mesh/simulator_assets/semantic_pgsr_30k_projected.ply",
    )
    parser.add_argument(
        "--semantic-manifest",
        type=Path,
        default=holi_root / "video2mesh/simulator_assets/semantic_pgsr_30k_projected_manifest.json",
    )
    parser.add_argument(
        "--semantic-quality-report",
        type=Path,
        default=(
            holi_root
            / "video2mesh/simulator_assets/semantic_pgsr_30k_projected_quality_report.json"
        ),
    )
    parser.add_argument("--assets-dir", type=Path, default=example_root / "assets-local")
    parser.add_argument("--output", type=Path, default=example_root / "world.manifest.local.json")
    parser.add_argument(
        "--report", type=Path, default=example_root / "world.materialization.local.json"
    )
    return parser


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    args = build_parser(project_root).parse_args()
    production_path = args.production_manifest.resolve()
    production_report_path = args.production_report.resolve()
    browser_path = args.browser_qa.resolve()
    cognition_path = args.cognition_manifest.resolve()
    pillow_path = args.pillow_entity.resolve()
    semantic_manifest_path = args.semantic_manifest.resolve()
    semantic_quality_path = args.semantic_quality_report.resolve()
    world_dir = args.world_dir.resolve()
    assets_dir = args.assets_dir.resolve()
    manifest_path = args.output.resolve()
    report_path = args.report.resolve()

    production = read_json(production_path)
    production_report = read_json(production_report_path)
    browser_qa = read_json(browser_path)
    cognition = read_json(cognition_path)
    pillow = read_json(pillow_path)
    semantic_manifest = read_json(semantic_manifest_path)
    semantic_quality = read_json(semantic_quality_path)

    require(production.get("schemaVersion") == 1, "unsupported Web manifest schemaVersion")
    require(
        production.get("contract") == "video2world-web-manifest-1.0.0",
        "unsupported Web manifest contract",
    )
    source_world = production.get("sourceWorld", {})
    require(source_world.get("schemaVersion") == "world-manifest-1.0.0", "sourceWorld schema drift")
    require(source_world.get("worldId") == WORLD_ID, "sourceWorld worldId mismatch")
    require(source_world.get("runId") == production.get("version"), "sourceWorld runId mismatch")
    require(
        source_world.get("adoptionMode")
        == "legacy_web_assets_adopted_then_canonical_manifest_validated",
        "Bedroom4 materializer only accepts the declared legacy adoption path",
    )
    require(production_report.get("status") == "passed", "production bundle report is not passed")
    require(browser_qa.get("status") == "passed", "browser QA is not passed")
    production_sha256, _ = sha256_file(production_path)
    require(
        browser_qa.get("manifestSha256AfterPromotion") == production_sha256,
        "browser QA does not match the promoted Web manifest",
    )
    require(
        production_report.get("manifestSha256") == production_sha256,
        "production report does not match the promoted Web manifest",
    )
    for gate, status in production_report.get("gates", {}).items():
        require(status == "passed", f"production gate {gate} is not passed")
    require(cognition.get("status") == "passed", "cognition manifest is not passed")
    require(
        pillow.get("quality", {}).get("status") == "accepted_for_visual_focus_and_rotation",
        "pillow visual focus evidence is not accepted",
    )
    require(
        pillow.get("interaction", {}).get("collision_ready") is False,
        "pillow unexpectedly claims collision readiness",
    )
    require(
        semantic_quality.get("status") == "semantic_gaussian_probabilities_ready",
        "semantic Gaussian quality report is not ready",
    )
    require(
        not semantic_quality.get("required_issues"),
        "semantic Gaussian quality report contains required issues",
    )

    cognition_by_id = {str(record["object_id"]): record for record in cognition.get("objects", [])}
    interactive_by_id = {
        str(record["id"]): record for record in production.get("interactiveObjects", [])
    }
    browser_by_id = {str(record["id"]): record for record in browser_qa.get("objects", [])}
    require(
        set(interactive_by_id) == set(VISUAL_COMPONENT_IDS),
        "unexpected production visual-component set",
    )
    require(
        set(VISUAL_COMPONENT_IDS) <= set(browser_by_id),
        "browser QA visual-component coverage is incomplete",
    )

    materialization_log: list[dict[str, str]] = []
    scene_visual_record = production["assets"]["visual"]
    scene_collider_record = production["assets"]["colliderStaticCarved"]
    scene_visual_file = merge_chunk_parts(
        scene_visual_record["parts"],
        world_dir / "chunks",
        assets_dir / "scene" / scene_visual_record["fileName"],
        expected_sha256=scene_visual_record["sha256"],
        expected_size=int(scene_visual_record["size"]),
    )
    scene_collider_file = merge_chunk_parts(
        scene_collider_record["parts"],
        world_dir / "chunks",
        assets_dir / "scene" / scene_collider_record["fileName"],
        expected_sha256=scene_collider_record["sha256"],
        expected_size=int(scene_collider_record["size"]),
    )
    semantic_file = materialize_file(
        args.semantic_ply,
        assets_dir / "scene" / "semantic_pgsr_30k_projected.ply",
    )
    for materialized in (scene_visual_file, scene_collider_file, semantic_file):
        materialization_log.append(
            {
                "path": str(materialized.path),
                "mode": materialized.mode,
                "source": materialized.source,
            }
        )

    scene_visual = asset_ref(
        scene_visual_file,
        manifest_path,
        role="scene_gaussian_static_carved",
        media_type="application/vnd.graphdeco.gaussian-ply",
        provenance={
            "vertex_count": int(scene_visual_record["vertexCount"]),
            "carved_object_ids": scene_visual_record["carvedObjectIds"],
            "removed_vertex_count": int(scene_visual_record["removedVertexCount"]),
            "unrecognized_scene_gaussians_remain_static": True,
        },
    )
    scene_collider = asset_ref(
        scene_collider_file,
        manifest_path,
        role="scene_collision_mesh_static_carved",
        media_type="model/ply",
        provenance={
            "vertex_count": int(scene_collider_record["vertexCount"]),
            "face_count": int(scene_collider_record["faceCount"]),
            "removed_face_count": int(scene_collider_record["removedFaceCount"]),
            "retained_intersecting_faces": int(
                production["collisionWorld"]["gate"]["retainedIntersectingFaces"]
            ),
        },
    )
    scene_semantic = asset_ref(
        semantic_file,
        manifest_path,
        role="semantic_scene_gaussian",
        media_type="application/vnd.graphdeco.semantic-gaussian-ply",
        provenance={
            "vertex_count": int(semantic_quality["ply_contract"]["vertex_count"]),
            "includes_object_id": True,
            "includes_object_probability": True,
            "selected_gaussian_total": int(semantic_quality["summary"]["selected_gaussian_total"]),
            "method": semantic_quality["summary"]["method"],
            "quality_report": str(semantic_quality_path),
        },
    )

    objects = [
        materialize_interactive_object(
            interactive_by_id[object_id],
            browser_by_id[object_id],
            cognition_by_id,
            world_dir,
            assets_dir,
            manifest_path,
            materialization_log,
        )
        for object_id in COLLIDABLE_IDS
    ]

    bed_semantic = next(
        (
            record
            for record in semantic_manifest.get("objects", [])
            if record.get("object_id") == "sam3_class_bed"
        ),
        None,
    )
    require(bed_semantic is not None, "semantic manifest has no bed class")
    bed = WorldObject(
        id="sam3_bed_01",
        source_run_id="holi-fresh",
        scoped_id=f"{WORLD_ID}::holi-fresh::sam3_bed_01",
        name=LocalizedText(zh="床", en="bed"),
        category="bed",
        aliases=["床铺", "bed area"],
        description=DescriptionEvidence(
            short=LocalizedText(
                zh="由 SAM3 床类掩码投影到语义 3DGS 的床类区域。",
                en="The bed-class region projected from SAM3 masks into semantic 3DGS.",
            ),
            fidelity_caveat=LocalizedText(
                zh="这是类级语义边界, 不是度量级精确床体边界。",
                en="This is a class-level semantic bound, not a metric-accurate bed extent.",
            ),
        ),
        bbox_scene=bounds_from_record(bed_semantic["bbox_3d"]),
        quality_gates=QualityGates(
            file=gate_record(
                relative_uri(semantic_manifest_path, manifest_path),
                {"semantic_record_present": True},
            ),
            semantic=gate_record(
                relative_uri(semantic_quality_path, manifest_path),
                {
                    "selected_gaussians": int(bed_semantic["point_count"]),
                    "probability_mean": float(bed_semantic["probability_mean"]),
                },
            ),
            alignment=gate_record(
                relative_uri(production_path, manifest_path),
                {"identity_shared_frame": True},
            ),
            collision=GateRecord(
                status="not_tested", reason="bed is knowledge-only; no replacement collider"
            ),
            visual=GateRecord(
                status="not_tested", reason="bed remains in the static scene visual layer"
            ),
        ),
    )
    objects.append(bed)

    pillow_item = interactive_by_id["sam3_pillow_01"]
    pillow_browser = browser_by_id["sam3_pillow_01"]
    pillow_visual_record = pillow_item["visual"]
    require(pillow_item["collision"]["mode"] == "none", "pillow collision mode is not none")
    require(pillow_item["collision"].get("asset") is None, "pillow declares a collider asset")
    require(
        pillow_item["collision"]["gate"]["status"] == "not_tested",
        "pillow collision gate must remain not_tested",
    )
    require(
        pillow_browser.get("visualKind") == "rgb-points"
        and pillow_browser.get("collisionMode") == "none"
        and pillow_browser.get("colliderReady") is False,
        "browser QA does not prove a visual-only pillow component",
    )
    pillow_interactions = browser_qa.get("interactions", {}).get("pillowVisualOnly", {})
    require(
        pillow_interactions.get("pointerDrag", {}).get("passed") is True,
        "pillow drag QA failed",
    )
    require(
        pillow_interactions.get("doubleClick360", {}).get("passed") is True,
        "pillow double-click QA failed",
    )
    require(
        pillow_visual_record["sha256"] == pillow["visual"]["sha256"],
        "web pillow visual is not the accepted point cloud",
    )
    pillow_file = merge_chunk_parts(
        pillow_visual_record["parts"],
        world_dir / "chunks",
        assets_dir / "objects" / "sam3_pillow_01" / "sam3_pillow_01.ply",
        expected_sha256=pillow_visual_record["sha256"],
        expected_size=int(pillow_visual_record["size"]),
    )
    materialization_log.append(
        {"path": str(pillow_file.path), "mode": pillow_file.mode, "source": pillow_file.source}
    )
    pillow_visual = asset_ref(
        pillow_file,
        manifest_path,
        role="interactive_visual_only_rgb_point_cloud",
        media_type="model/ply",
        provenance={
            "point_count": int(pillow["visual"]["point_count"]),
            "renderer": pillow_visual_record["renderer"],
            "web_component": True,
            "selectable": True,
            "drag_action": "rotate_yaw",
            "double_click_action": "spin_360",
            "collision_ready": False,
            "instance_semantics": pillow["instance_semantics"],
        },
    )
    frame_path = manifest_path.parent / "cognition/inputs/evidence/sam3_pillow_01/frame_000064.png"
    mask_path = (
        manifest_path.parent / "cognition/inputs/evidence/sam3_pillow_01/accepted_mask_000064.png"
    )
    projection_value = float(pillow["quality"]["projection_value"])
    pillow_object = WorldObject(
        id="sam3_pillow_01",
        source_run_id="holi-pillow-delta",
        scoped_id=f"{WORLD_ID}::holi-pillow-delta::sam3_pillow_01",
        name=LocalizedText(zh="枕头组合", en="pillow ensemble"),
        category="pillow",
        aliases=[str(alias) for alias in pillow["aliases"]],
        description=description_from_cognition(cognition_by_id, "sam3_pillow_01"),
        evidence=ObjectEvidence(
            source_frame_id="000064",
            source_image=existing_asset_ref(
                frame_path,
                manifest_path,
                role="accepted_pillow_rgb_evidence",
                media_type="image/png",
            ),
            source_mask=existing_asset_ref(
                mask_path,
                manifest_path,
                role="accepted_pillow_sam3_mask",
                media_type="image/png",
            ),
            source_point_cloud=pillow_visual,
            support_ratio=projection_value,
        ),
        bbox_scene=bounds_from_record(pillow["aabb"]),
        obb_scene=OrientedBounds3D(
            frame_id=FRAME_ID,
            transform=tuple(float(value) for row in pillow["obb"]["transform"] for value in row),
            extents=tuple(float(value) for value in pillow["obb"]["extent"]),
        ),
        visual=pillow_visual,
        transform_scene_from_asset=identity_transform(
            "sam3_pillow_01", pillow_item["placement"]["pivot"]
        ),
        relations=[
            Relation(
                predicate="on_top_of",
                target_object_id=bed.scoped_id,
                target_label=LocalizedText(zh="床", en="the bed"),
                confidence=projection_value,
                verified=True,
                evidence_ids=["pillow_projection_000020_000040_000064"],
            )
        ],
        quality_gates=QualityGates(
            file=gate_record(
                relative_uri(pillow_path, manifest_path),
                {"sha256_verified": True, "point_count": int(pillow["visual"]["point_count"])},
            ),
            semantic=gate_record(
                relative_uri(cognition_path, manifest_path),
                {"qwen_cognition_passed": True, "accepted_sam3_mask": True},
            ),
            alignment=gate_record(
                relative_uri(pillow_path, manifest_path),
                {"identity_holi_da3_to_pgsr_shared_frame": True},
            ),
            collision=GateRecord(
                status="not_tested",
                reason="No pillow mesh, GLB, collider, or robot collision QA exists",
            ),
            visual=gate_record(
                relative_uri(browser_path, manifest_path),
                {
                    "projection_hit_ratio_min": projection_value,
                    "projection_threshold": float(pillow["quality"]["projection_threshold"]),
                    "browser_visual_kind": "rgb-points",
                    "pointer_drag_passed": True,
                    "double_click_360_passed": True,
                },
            ),
        ),
        interaction=InteractionPolicy(
            selectable=True,
            double_click_action="spin_360",
            drag_action="rotate_yaw",
            collision_enabled=False,
            physics_mode="none",
        ),
    )
    objects.append(pillow_object)

    created_at = datetime.fromisoformat(
        str(production["productionBuild"]["createdAt"]).replace("Z", "+00:00")
    )
    config_sha = digest_json(
        {
            "production_manifest": digest_path(production_path).sha256,
            "production_report": digest_path(production_report_path).sha256,
            "browser_qa": digest_path(browser_path).sha256,
            "cognition": digest_path(cognition_path).sha256,
            "pillow": digest_path(pillow_path).sha256,
            "semantic_manifest": digest_path(semantic_manifest_path).sha256,
            "semantic_quality": digest_path(semantic_quality_path).sha256,
        }
    )
    manifest = WorldManifest(
        manifest_status="validated",
        world_id=WORLD_ID,
        run_id=str(production["version"]),
        created_at=created_at,
        scene=SceneLayer(
            coordinate_system=CoordinateSystem(
                frame_id=FRAME_ID,
                up_axis="-Y",
                handedness="right",
                units="scene_scale_not_metric",
            ),
            bounds=bounds_from_record(scene_visual_record["bbox"]),
            visual=scene_visual,
            collider=scene_collider,
            semantic_visual=scene_semantic,
        ),
        objects=objects,
        provenance=ProvenanceRecord(
            source_repositories={
                "video2mesh_web_stable_baseline": "252a85c",
                "video2mesh_interactive_reference": "a5af3ae",
                "holi_spatial": "real_fresh_da3_sam3_pgsr_run_20260714",
                "embodiedgen_v2": "bedroom4_mesh_exports_20260716",
            },
            source_runs=[
                str(production_path),
                str(semantic_manifest_path),
                str(cognition_path),
                str(pillow_path),
            ],
            config_sha256=config_sha,
            notes=[
                "Coordinates use the shared native PGSR/DA3 frame and are not metric.",
                "Static scene Gaussian layer has all five visual components carved out; "
                "the TSDF layer only carves the four mesh-collider replacements.",
                "The four interactive visuals are verified web LOD Gaussians with "
                "simplified GLB colliders.",
                "The pillow is a selectable RGB point component with yaw drag and spin_360; "
                "it has no collider and no collision readiness is claimed.",
            ],
        ),
    )
    atomic_write_json(manifest_path, manifest.model_dump(mode="json"))

    validation = validate_world_manifest(manifest_path)
    require(validation["valid"] is True, f"generated manifest validation failed: {validation}")
    location = query_world(manifest, "枕头在哪里?")
    appearance = query_world(manifest, "枕头长什么样?")
    for result in (location, appearance):
        require(result.status == "resolved", f"pillow query failed: {result}")
        require(result.focus_bbox is not None, f"pillow query returned no focus bbox: {result}")
        require(result.resolved_object_id == "sam3_pillow_01", "pillow query resolved incorrectly")
    require(
        location.answer is not None and "床上" in location.answer,
        "pillow location lacks bed relation",
    )
    require(
        appearance.answer is not None and "三只" in appearance.answer,
        "pillow appearance lacks the reviewed three-pillow description",
    )

    report = {
        "schema_version": 1,
        "status": "passed",
        "manifest": str(manifest_path),
        "assets_dir": str(assets_dir),
        "materialized_files": materialization_log,
        "validation": validation,
        "queries": {
            "pillow_location": location.model_dump(mode="json"),
            "pillow_appearance": appearance.model_dump(mode="json"),
        },
        "truth_boundaries": {
            "units": "scene_scale_not_metric",
            "visual_component_ids": list(VISUAL_COMPONENT_IDS),
            "collidable_object_ids": list(COLLIDABLE_IDS),
            "pillow_focus_only": False,
            "pillow_visual_interactive": True,
            "pillow_drag_action": "rotate_yaw",
            "pillow_double_click_action": "spin_360",
            "pillow_collision_ready": False,
        },
    }
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
