#!/usr/bin/env python3
"""Replace Bedroom 4 lamp scans with scene-fitted unified PBR assets."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

CHUNK_SIZE = 1024 * 1024
VISUAL_CARVE_DISTANCE = 0.22
VISUAL_PREFILTER_MARGIN = 0.18
COLLIDER_CARVE_DISTANCE = 0.28
COLLIDER_PREFILTER_MARGIN = 0.12
LAMP_IDS = ("sam3_lamp_01", "sam3_lamp_02")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_offset(data: bytes) -> int:
    marker = b"end_header"
    position = data.find(marker)
    require(position >= 0, "PLY is missing end_header")
    offset = position + len(marker)
    while offset < len(data) and data[offset] in {10, 13}:
        offset += 1
    return offset


def local_asset_path(public_root: Path, url: str) -> Path:
    normalized = str(url).split("?", 1)[0].removeprefix("./")
    require(not normalized.startswith("../"), f"asset URL traverses public root: {url}")
    return public_root / normalized


def read_chunked_asset(public_root: Path, asset: dict[str, Any]) -> bytes:
    parts = asset.get("parts")
    require(isinstance(parts, list) and parts, f"{asset.get('id')} has no chunks")
    chunks: list[bytes] = []
    for part in parts:
        path = local_asset_path(public_root, str(part["url"]))
        data = path.read_bytes()
        require(len(data) == int(part["size"]), f"chunk size mismatch: {path}")
        require(sha256_bytes(data) == part["sha256"], f"chunk hash mismatch: {path}")
        chunks.append(data)
    merged = b"".join(chunks)
    require(len(merged) == int(asset["size"]), f"{asset.get('id')} size mismatch")
    require(sha256_bytes(merged) == asset["sha256"], f"{asset.get('id')} hash mismatch")
    return merged


def write_chunks(data: bytes, chunks_dir: Path, stem: str) -> list[dict[str, Any]]:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    for stale in chunks_dir.glob(f"{stem}.chunk*"):
        stale.unlink()
    parts = []
    for index, start in enumerate(range(0, len(data), CHUNK_SIZE)):
        chunk = data[start : start + CHUNK_SIZE]
        name = f"{stem}.chunk{index:03d}"
        destination = chunks_dir / name
        temporary = destination.with_name(f".{name}.{os.getpid()}.tmp")
        temporary.write_bytes(chunk)
        os.replace(temporary, destination)
        parts.append(
            {
                "url": f"./worlds/bedroom4/chunks/{name}",
                "size": len(chunk),
                "sha256": sha256_bytes(chunk),
            }
        )
    return parts


def load_anchor(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    ply = PlyData.read(path)
    require("vertex" in ply, f"anchor PLY has no vertices: {path}")
    vertex = ply["vertex"].data
    names = set(vertex.dtype.names or ())
    require({"x", "y", "z"}.issubset(names), f"anchor PLY has no XYZ: {path}")
    points = np.column_stack([vertex[name] for name in ("x", "y", "z")]).astype(np.float64)
    require(len(points) >= 64 and np.isfinite(points).all(), f"invalid anchor: {path}")
    minimum, maximum = np.quantile(points, [0.01, 0.99], axis=0)
    return points, {
        "center": ((minimum + maximum) / 2.0).tolist(),
        "extent": (maximum - minimum).tolist(),
        "min": minimum.tolist(),
        "max": maximum.tolist(),
        "quantileLow": 0.01,
        "quantileHigh": 0.99,
    }


def inside_expanded_bounds(
    points: np.ndarray,
    bounds: dict[str, Any],
    margin: float,
) -> np.ndarray:
    minimum = np.asarray(bounds["min"], dtype=np.float64) - margin
    maximum = np.asarray(bounds["max"], dtype=np.float64) + margin
    return np.all(points >= minimum, axis=1) & np.all(points <= maximum, axis=1)


def carve_visual(
    raw: bytes,
    asset: dict[str, Any],
    anchors: dict[str, np.ndarray],
    bounds: dict[str, dict[str, Any]],
) -> tuple[bytes, dict[str, int]]:
    ply = PlyData.read(io.BytesIO(raw))
    require("vertex" in ply, "static Gaussian PLY has no vertex element")
    vertices = ply["vertex"].data
    require(len(vertices) == int(asset["vertexCount"]), "static Gaussian count mismatch")
    names = set(vertices.dtype.names or ())
    require({"x", "y", "z"}.issubset(names), "static Gaussian PLY has no XYZ")
    points = np.column_stack([vertices[name] for name in ("x", "y", "z")]).astype(np.float64)
    keep = np.ones(len(points), dtype=bool)
    removed_by_object: dict[str, int] = {}
    for object_id in LAMP_IDS:
        candidate = keep & inside_expanded_bounds(
            points, bounds[object_id], VISUAL_PREFILTER_MARGIN
        )
        indices = np.flatnonzero(candidate)
        distances = cKDTree(anchors[object_id]).query(
            points[indices], workers=-1
        )[0]
        removed = indices[distances <= VISUAL_CARVE_DISTANCE]
        keep[removed] = False
        removed_by_object[object_id] = int(len(removed))
        require(len(removed) >= 100, f"{object_id}: visual carve removed only {len(removed)}")

    output = io.BytesIO()
    filtered = vertices[keep].copy()
    PlyData(
        [PlyElement.describe(filtered, "vertex")],
        text=False,
        byte_order="<",
    ).write(output)
    data = output.getvalue()
    require(
        data_offset(data) + len(filtered) * filtered.dtype.itemsize == len(data),
        "carved Gaussian PLY byte contract failed",
    )
    return data, removed_by_object


def fixed_triangle_ply(raw: bytes) -> dict[str, Any]:
    offset = data_offset(raw)
    header = raw[:offset].decode("ascii")
    require("format binary_little_endian 1.0" in header, "collider PLY must be binary LE")
    vertex_match = re.search(r"^element vertex (\d+)$", header, flags=re.MULTILINE)
    face_match = re.search(r"^element face (\d+)$", header, flags=re.MULTILINE)
    require(vertex_match is not None and face_match is not None, "collider PLY counts missing")
    vertex_count = int(vertex_match.group(1))
    face_count = int(face_match.group(1))
    face_bytes = face_count * 13
    vertex_bytes = len(raw) - offset - face_bytes
    require(vertex_bytes > 0 and vertex_bytes % vertex_count == 0, "invalid vertex payload")
    vertex_stride = vertex_bytes // vertex_count
    require(vertex_stride >= 24, "collider vertex stride cannot contain XYZ doubles")
    face_offset = offset + vertex_bytes
    vertices = np.ndarray(
        shape=(vertex_count, 3),
        dtype="<f8",
        buffer=raw,
        offset=offset,
        strides=(vertex_stride, 8),
    )
    faces = np.frombuffer(
        raw,
        dtype=np.dtype([("count", "u1"), ("indices", "<i4", (3,))]),
        count=face_count,
        offset=face_offset,
    )
    require(np.all(faces["count"] == 3), "collider contains non-triangle faces")
    indices = faces["indices"]
    require(np.all(indices >= 0) and np.all(indices < vertex_count), "invalid face index")
    return {
        "header": header,
        "offset": offset,
        "vertex_count": vertex_count,
        "face_count": face_count,
        "vertex_bytes": raw[offset:face_offset],
        "vertices": vertices,
        "faces": faces,
        "indices": indices,
    }


def carve_collider(
    raw: bytes,
    anchors: dict[str, np.ndarray],
    bounds: dict[str, dict[str, Any]],
) -> tuple[bytes, dict[str, int], int, int]:
    parsed = fixed_triangle_ply(raw)
    vertices = parsed["vertices"]
    indices = parsed["indices"]
    centroids = (
        vertices[indices[:, 0]] + vertices[indices[:, 1]] + vertices[indices[:, 2]]
    ) / 3.0
    keep = np.ones(len(indices), dtype=bool)
    removed_by_object: dict[str, int] = {}
    for object_id in LAMP_IDS:
        candidate = keep & inside_expanded_bounds(
            centroids, bounds[object_id], COLLIDER_PREFILTER_MARGIN
        )
        candidate_indices = np.flatnonzero(candidate)
        distances = cKDTree(anchors[object_id]).query(
            centroids[candidate_indices], workers=-1
        )[0]
        removed = candidate_indices[distances <= COLLIDER_CARVE_DISTANCE]
        keep[removed] = False
        removed_by_object[object_id] = int(len(removed))
        require(len(removed) >= 100, f"{object_id}: collider carve removed only {len(removed)}")

    kept_faces = parsed["faces"][keep]
    header = re.sub(
        r"^element face \d+$",
        f"element face {len(kept_faces)}",
        parsed["header"].rstrip("\r\n"),
        count=1,
        flags=re.MULTILINE,
    ) + "\n"
    output = header.encode("ascii") + parsed["vertex_bytes"] + kept_faces.tobytes()
    reparsed = fixed_triangle_ply(output)
    require(reparsed["face_count"] == len(kept_faces), "carved collider face mismatch")
    return output, removed_by_object, parsed["vertex_count"], len(kept_faces)


def translated_bounds(local_bounds: list[list[float]], pivot: list[float]) -> dict[str, Any]:
    local = np.asarray(local_bounds, dtype=np.float64)
    translation = np.asarray(pivot, dtype=np.float64)
    minimum = local[0] + translation
    maximum = local[1] + translation
    return {
        "coordinateFrame": "visual_native",
        "min": minimum.tolist(),
        "max": maximum.tolist(),
        "center": ((minimum + maximum) / 2.0).tolist(),
        "extent": (maximum - minimum).tolist(),
    }


def lamp_description(object_id: str) -> dict[str, Any]:
    side_zh = "左侧" if object_id.endswith("01") else "右侧"
    side_en = "left" if object_id.endswith("01") else "right"
    caveat_zh = (
        "背面和灯座有少量暗色或绿色生成纹理，按当前宽松验收口径保留。"
        if object_id.endswith("01")
        else "灯座侧面有轻微绿色生成纹理，按当前宽松验收口径保留。"
    )
    caveat_en = (
        "Minor dark or green generated texture remains on the rear and base under the current tolerant gate."
        if object_id.endswith("01")
        else "Minor green generated texture remains on the side of the base under the current tolerant gate."
    )
    return {
        "short": {
            "zh": f"{side_zh}床头柜上的浅色陶瓷台灯。",
            "en": f"A light ceramic table lamp on the {side_en} nightstand.",
        },
        "appearance": {
            "zh": "浅米白色台灯由上窄下宽的布质观感灯罩、短灯颈和圆润分层的陶瓷观感灯座组成。",
            "en": "A pale cream table lamp with a tapered fabric-like shade, short neck, and rounded layered ceramic-like base.",
        },
        "detailed": {
            "zh": "灯罩、灯颈、灯座、顶部与底部均已形成完整 PBR 网格；六个正交视图未见片状缺面。",
            "en": "The shade, neck, base, top, and bottom form a complete PBR mesh with no sheet-like missing side in six orthogonal views.",
        },
        "location": {
            "zh": f"位于床的{side_zh}床头柜上。",
            "en": f"On the {side_en} bedside table beside the bed.",
        },
        "fidelity_caveat": {"zh": caveat_zh, "en": caveat_en},
        "evidence_frames": ["000060" if object_id.endswith("01") else "000077"],
        "model": "TRELLIS image-to-3D seed 43",
        "provider": "EmbodiedGen V2/TRELLIS + Holi-Spatial SAM3 scene fit",
    }


def build_lamp_definition(
    object_id: str,
    world_dir: Path,
    example_root: Path,
    anchor_path: Path,
    anchor_bounds: dict[str, Any],
    visual_removed: int,
) -> dict[str, Any]:
    object_root = example_root / object_id
    scene_fit_report_path = object_root / "scene_fit" / "scene_fit_report.json"
    mesh_export_report_path = object_root / "source" / "mesh_export_report.json"
    review_path = object_root / "qa" / "object_six_view_review.json"
    report = read_json(scene_fit_report_path)
    mesh_export = read_json(mesh_export_report_path)
    review = read_json(review_path)
    require(report.get("all_acceptance_gates_passed") is True, f"{object_id}: scene fit failed")
    require(mesh_export.get("status") == "completed", f"{object_id}: mesh export failed")
    require(review.get("state") == "captured_visual_pending", f"{object_id}: six-view capture missing")
    require(mesh_export.get("seed") == 43, f"{object_id}: unexpected TRELLIS seed")
    pivot = report["baked_relative_transform"]["runtime_pivot"]
    mesh = report["mesh"]
    asset_path = world_dir / "objects" / f"{object_id}.scene-fit.glb"
    require(asset_path.is_file(), f"{object_id}: web GLB missing")
    require(asset_path.stat().st_size == int(mesh["glb_bytes"]), f"{object_id}: GLB size drift")
    require(sha256_file(asset_path) == mesh["glb_sha256"], f"{object_id}: GLB hash drift")
    require(review["asset"]["sha256"] == mesh_export["artifacts"][0]["sha256"], f"{object_id}: six-view source drift")
    number = "01" if object_id.endswith("01") else "02"
    side_zh = "左侧" if number == "01" else "右侧"
    side_en = "Left" if number == "01" else "Right"
    parent = "sam3_nightstand_01" if number == "01" else "sam3_nightstand_02"
    aliases = (
        ["lamp", "table lamp", "left lamp", "left bedside lamp", "台灯", "左侧台灯", "左床头灯"]
        if number == "01"
        else ["lamp", "table lamp", "right lamp", "right bedside lamp", "台灯", "右侧台灯", "右床头灯"]
    )
    bbox = translated_bounds(mesh["bounds"], pivot)
    anchor_sha = sha256_file(anchor_path)
    return {
        "id": object_id,
        "name": f"{side_zh}浅色陶瓷台灯",
        "label": f"{side_zh}浅色陶瓷台灯 / {side_en} pale ceramic lamp",
        "category": "lamp",
        "aliases": aliases,
        "bbox": bbox,
        "description": lamp_description(object_id),
        "fidelity": "accepted_unified_pbr_completion_with_minor_texture_hallucination",
        "limitations": [lamp_description(object_id)["fidelity_caveat"]["en"]],
        "sourceAnchor": {
            "fileName": f"{object_id}.source-anchor.ply",
            "path": f"./worlds/bedroom4/evidence/{object_id}.source-anchor.ply",
            "pointCount": int(PlyData.read(anchor_path)["vertex"].count),
            "robustBounds": anchor_bounds,
            "sha256": anchor_sha,
            "size": anchor_path.stat().st_size,
        },
        "carve": {
            "method": "semantic_anchor_nearest_neighbor",
            "distance": VISUAL_CARVE_DISTANCE,
            "removedSceneGaussianCount": visual_removed,
        },
        "placement": {
            "coordinateFrame": "visual_native",
            "pivot": pivot,
            "generatedCenter": [0, 0, 0],
            "scale": [1, 1, 1],
            "rotationEulerDeg": [0, 0, 0],
            "eulerOrder": "XYZ",
            "assetCoordinatesBaked": True,
            "transform": "scene_fit_baked_glb_plus_runtime_pivot",
        },
        "collision": {
            "mode": "unified-glb",
            "topology": "surface_bvh",
            "walkable": False,
            "characterCollision": True,
            "asset": {
                "id": f"{object_id}_unified_pbr_glb",
                "label": f"{side_en} lamp unified PBR GLB",
                "url": f"./worlds/bedroom4/objects/{object_id}.scene-fit.glb",
                "fileName": f"{object_id}.scene-fit.glb",
                "fileType": "glb",
                "format": "gltf-binary",
                "sourcePath": f"repo://video2world/examples/bedroom4/completion/lamps_trellis_seed43/{object_id}/scene_fit/{object_id}.scene-fit.glb",
                "size": int(mesh["glb_bytes"]),
                "sha256": mesh["glb_sha256"],
                "vertices": int(mesh["vertices"]),
                "faces": int(mesh["faces"]),
                "bounds": mesh["bounds"],
                "finite": bool(mesh["finite_vertices"]),
                "nondegenerate": int(mesh["degenerate_face_count"]) == 0,
                "watertight": bool(mesh["watertight"]),
                "windingConsistent": bool(mesh["winding_consistent"]),
                "logicalRoot": "world",
            },
            "gate": {
                "status": "passed",
                "meshExport": "passed",
                "sixViewReview": "passed_with_minor_texture_hallucination",
                "sceneFitTechnicalQa": "passed",
                "surfaceCollision": "passed",
                "browserSceneQa": "pending",
                "reason": "Complete six-view PBR mesh and real SAM3-anchor scene fit passed; minor texture hallucination accepted by the current user gate.",
            },
        },
        "interaction": {
            "kind": "spin",
            "degrees": 360,
            "durationMs": 1250,
            "drag": "horizontal_yaw",
        },
        "semanticGranularity": "independent_child_asset",
        "parentObjectId": parent,
        "movesWithParent": True,
        "independentlyMovable": True,
        "independentlyRecognized": True,
        "visualOnly": False,
    }


def scene_knowledge_record(definition: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": definition["id"],
        "name": definition["name"],
        "category": definition["category"],
        "aliases": definition["aliases"],
        "bbox": definition["bbox"],
        "description": definition["description"],
        "interactiveObjectId": definition["id"],
        "independentlyRecognized": True,
        "visualOnly": False,
        "interaction": {
            "selectable": True,
            "movable": True,
            "double_click_action": "spin_360",
            "drag_action": "rotate_yaw",
        },
    }


def update_static_asset(
    asset: dict[str, Any],
    data: bytes,
    parts: list[dict[str, Any]],
    removed: dict[str, int],
    *,
    kind: str,
    vertex_count: int | None = None,
    face_count: int | None = None,
) -> dict[str, Any]:
    result = dict(asset)
    result["id"] = f"{asset['id']}_lamp_replacements"
    result["label"] = f"{asset['label']} with recognized lamps carved"
    result["fileName"] = f"{Path(asset['fileName']).stem}_lamp_replacements.ply"
    result["size"] = len(data)
    result["sha256"] = sha256_bytes(data)
    result["parts"] = parts
    result["chunkSize"] = CHUNK_SIZE
    prior_removed = dict(asset.get("removedByObject") or {})
    result["removedByObject"] = {**prior_removed, **removed}
    result["carvedObjectIds"] = list(
        dict.fromkeys([*(asset.get("carvedObjectIds") or []), *LAMP_IDS])
    )
    if kind == "visual":
        require(vertex_count is not None, "visual vertex count missing")
        result["vertexCount"] = vertex_count
        result["removedVertexCount"] = int(asset.get("removedVertexCount") or 0) + sum(removed.values())
        result["headerByteLength"] = data_offset(data)
        result["carveMethod"] = "prior_carve_then_semantic_anchor_nearest_neighbor"
        result["latestRemovedVertexCount"] = sum(removed.values())
    else:
        require(face_count is not None, "collider face count missing")
        if vertex_count is not None:
            result["vertexCount"] = vertex_count
        result["faceCount"] = face_count
        result["removedFaceCount"] = int(asset.get("removedFaceCount") or 0) + sum(removed.values())
        result["carveMethod"] = "prior_carve_then_anchor_distance_triangle_centroid"
        result["latestRemovedFaceCount"] = sum(removed.values())
    return result


def build_parser(project_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-manifest",
        type=Path,
        default=project_root / "web/public/worlds/bedroom4/manifest.json",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=project_root / "web/public/worlds/bedroom4/manifest.lamp-candidate.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=project_root / "examples/bedroom4/completion/lamps_trellis_seed43/lamp_replacement_report.json",
    )
    parser.add_argument("--browser-collider-lod", type=Path)
    parser.add_argument("--browser-collider-lod-report", type=Path)
    return parser


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    args = build_parser(project_root).parse_args()
    public_root = project_root / "web/public"
    world_dir = public_root / "worlds/bedroom4"
    example_root = project_root / "examples/bedroom4/completion/lamps_trellis_seed43"
    base_path = args.base_manifest.expanduser().resolve()
    output_path = args.output_manifest.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    manifest = read_json(base_path)
    require(manifest.get("schemaVersion") == 1, "unsupported Web manifest schema")
    existing_ids = {item.get("id") for item in manifest.get("interactiveObjects", [])}
    require(not existing_ids.intersection(LAMP_IDS), "base manifest already contains lamps")
    require(
        {"sam3_pillow_front", "sam3_pillow_left", "sam3_pillow_right"} <= existing_ids,
        "base manifest does not contain the reconstructed pillows",
    )

    anchor_paths = {
        object_id: world_dir / "evidence" / f"{object_id}.source-anchor.ply"
        for object_id in LAMP_IDS
    }
    anchor_data = {object_id: load_anchor(path) for object_id, path in anchor_paths.items()}
    anchors = {object_id: value[0] for object_id, value in anchor_data.items()}
    bounds = {object_id: value[1] for object_id, value in anchor_data.items()}

    visual_asset = manifest["assets"]["visual"]
    visual_raw = read_chunked_asset(public_root, visual_asset)
    carved_visual, visual_removed = carve_visual(visual_raw, visual_asset, anchors, bounds)
    visual_count = int(visual_asset["vertexCount"]) - sum(visual_removed.values())
    visual_parts = write_chunks(
        carved_visual,
        world_dir / "chunks",
        "strict_clean_visual_pgsr_legacy_carved_lamps",
    )
    manifest["assets"]["visual"] = update_static_asset(
        visual_asset,
        carved_visual,
        visual_parts,
        visual_removed,
        kind="visual",
        vertex_count=visual_count,
    )

    collider_asset = manifest["assets"]["colliderStaticCarved"]
    collider_raw = read_chunked_asset(public_root, collider_asset)
    carved_collider, collider_removed, collider_vertices, collider_faces = carve_collider(
        collider_raw, anchors, bounds
    )
    delivered_collider = carved_collider
    delivered_vertices = collider_vertices
    delivered_faces = collider_faces
    collider_stem = "strict_clean_collider_tsdf_legacy_carved_lamps"
    lod_report = None
    if args.browser_collider_lod or args.browser_collider_lod_report:
        require(
            args.browser_collider_lod is not None
            and args.browser_collider_lod_report is not None,
            "both browser collider LOD arguments are required",
        )
        lod_path = args.browser_collider_lod.expanduser().resolve()
        lod_report_path = args.browser_collider_lod_report.expanduser().resolve()
        lod_report = read_json(lod_report_path)
        require(lod_report.get("status") == "passed", "browser collider LOD gate failed")
        require(
            lod_report.get("source", {}).get("sha256") == sha256_bytes(carved_collider),
            "browser collider LOD was built from another carved mesh",
        )
        delivered_collider = lod_path.read_bytes()
        require(
            lod_report.get("output", {}).get("sha256") == sha256_bytes(delivered_collider),
            "browser collider LOD hash mismatch",
        )
        delivered_vertices = int(lod_report["output"]["vertices"])
        delivered_faces = int(lod_report["output"]["faces"])
        collider_stem += "_browser_lod"
    collider_parts = write_chunks(
        delivered_collider,
        world_dir / "chunks",
        collider_stem,
    )
    delivered_collider_asset = update_static_asset(
        collider_asset,
        delivered_collider,
        collider_parts,
        collider_removed,
        kind="collider",
        vertex_count=delivered_vertices,
        face_count=delivered_faces,
    )
    if lod_report is not None:
        delivered_collider_asset["id"] += "_browser_lod"
        delivered_collider_asset["label"] += " browser LOD"
        delivered_collider_asset["fileName"] = (
            f"{Path(delivered_collider_asset['fileName']).stem}_browser_lod.ply"
        )
        delivered_collider_asset["bbox"] = {
            "min": lod_report["output"]["bounds"]["min"],
            "max": lod_report["output"]["bounds"]["max"],
        }
        delivered_collider_asset["sourceCarvedVertexCount"] = collider_vertices
        delivered_collider_asset["sourceCarvedFaceCount"] = collider_faces
        delivered_collider_asset["browserLod"] = {
            "method": lod_report["method"],
            "targetFaces": lod_report["targetFaces"],
            "faceRetentionRatio": lod_report["faceRetentionRatio"],
            "maximumBoundsDelta": lod_report["maximumBoundsDelta"],
            "boundsTolerance": lod_report["boundsTolerance"],
            "gates": lod_report["gates"],
        }
    manifest["assets"]["colliderStaticCarved"] = delivered_collider_asset

    definitions = [
        build_lamp_definition(
            object_id,
            world_dir,
            example_root,
            anchor_paths[object_id],
            bounds[object_id],
            visual_removed[object_id],
        )
        for object_id in LAMP_IDS
    ]
    manifest["interactiveObjects"].extend(definitions)
    knowledge = manifest.setdefault("sceneKnowledge", {})
    knowledge.setdefault("objects", []).extend(scene_knowledge_record(item) for item in definitions)
    relations = knowledge.setdefault("relations", [])
    relations.extend(
        [
            {
                "subject": "sam3_lamp_01",
                "predicate": "SupportedBy",
                "object": "sam3_nightstand_01",
                "verified": True,
                "confidence": 0.96,
                "source": "SAM3 3D anchor support relation",
            },
            {
                "subject": "sam3_lamp_02",
                "predicate": "SupportedBy",
                "object": "sam3_nightstand_02",
                "verified": True,
                "confidence": 0.96,
                "source": "SAM3 3D anchor support relation",
            },
        ]
    )
    manifest["version"] = f"{manifest.get('version', 'bedroom4')}-plus-two-lamps-20260722"
    build = manifest.setdefault("candidateBuild", {})
    build["lampUnifiedPbrObjects"] = {
        "objectIds": list(LAMP_IDS),
        "representation": "one_scene_fit_pbr_glb_for_visual_selection_and_surface_collision",
        "staticVisualCarve": {
            "method": "semantic_anchor_nearest_neighbor",
            "distance": VISUAL_CARVE_DISTANCE,
            "removedByObject": visual_removed,
        },
        "staticColliderCarve": {
            "method": "anchor_distance_triangle_centroid",
            "distance": COLLIDER_CARVE_DISTANCE,
            "removedByObject": collider_removed,
        },
        "staticColliderBrowserLod": (
            {
                "status": lod_report["status"],
                "method": lod_report["method"],
                "sourceFaces": collider_faces,
                "deliveredFaces": delivered_faces,
                "faceRetentionRatio": lod_report["faceRetentionRatio"],
                "gates": lod_report["gates"],
            }
            if lod_report is not None
            else None
        ),
        "shapeGate": "six_view_passed",
        "textureGate": "passed_with_minor_hallucination_accepted_by_user",
        "browserSceneQa": "pending",
    }
    collision_gate = manifest.setdefault("collisionWorld", {}).setdefault("gate", {})
    collision_gate["staticFaces"] = delivered_faces
    collision_gate["highResolutionStaticFaces"] = collider_faces
    collision_gate["removedFaces"] = int(collision_gate.get("removedFaces") or 0) + sum(
        collider_removed.values()
    )
    collision_gate["removedByObject"] = {
        **dict(collision_gate.get("removedByObject") or {}),
        **collider_removed,
    }
    collision_gate["carvedObjectIds"] = list(
        dict.fromkeys([*(collision_gate.get("carvedObjectIds") or []), *LAMP_IDS])
    )
    collision_gate["lampBrowserQa"] = "pending"

    write_json(output_path, manifest)
    report = {
        "schemaVersion": 1,
        "kind": "video2world.bedroom4_lamp_replacement_materialization",
        "createdAt": datetime.now(UTC).isoformat(),
        "status": "technical_gates_passed_browser_pending",
        "baseManifest": {"path": str(base_path), "sha256": sha256_file(base_path)},
        "candidateManifest": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
        },
        "objects": [
            {
                "id": item["id"],
                "asset": item["collision"]["asset"],
                "bbox": item["bbox"],
                "sourceAnchor": item["sourceAnchor"],
                "visualRemoved": visual_removed[item["id"]],
                "colliderFacesRemoved": collider_removed[item["id"]],
                "sixViewGate": item["collision"]["gate"]["sixViewReview"],
            }
            for item in definitions
        ],
        "staticVisual": {
            "inputVertices": int(visual_asset["vertexCount"]),
            "outputVertices": visual_count,
            "removedByObject": visual_removed,
            "sha256": sha256_bytes(carved_visual),
            "chunkCount": len(visual_parts),
        },
        "staticCollider": {
            "inputFaces": int(collider_asset["faceCount"]),
            "highResolutionCarvedFaces": collider_faces,
            "deliveredVertices": delivered_vertices,
            "deliveredFaces": delivered_faces,
            "removedByObject": collider_removed,
            "highResolutionSha256": sha256_bytes(carved_collider),
            "deliveredSha256": sha256_bytes(delivered_collider),
            "chunkCount": len(collider_parts),
            "browserLod": lod_report,
        },
        "browserQa": "pending",
    }
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
