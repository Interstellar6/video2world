#!/usr/bin/env python3
"""Experimentally remove residual front-pillow splats from the Bedroom 4 static layer.

The canonical manifest and chunks are never modified. Candidate bounds come from the
unified-pillow materializer, while Gaussian footprint radii follow Spark's sqrt(8)
standard-deviation support. Rear RGB point clouds are explicit protection anchors.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from plyfile import PlyData

from video2world.hashing import atomic_write_json, sha256_file

FRONT_ID = "sam3_pillow_front"
REAR_IDS = ("sam3_pillow_left", "sam3_pillow_right")
SPARK_MAX_STDDEV = math.sqrt(8.0)
BASE_DISTANCE = 0.08
MAX_SEMANTIC_DISTANCE = 0.40
REAR_PROTECTION_MARGIN = 0.02
COVARIANCE_PREFILTER_PADDING = 0.40
CHUNK_SIZE = 1024 * 1024
MODES = ("center-union", "covariance-union", "semantic-covariance")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def resolve_chunk_path(project_root: Path, url: str) -> Path:
    normalized = str(url).removeprefix("./")
    if normalized.startswith("worlds/"):
        return project_root / "web" / "public" / normalized
    return project_root / "web" / "public" / "worlds" / "bedroom4" / "chunks" / Path(url).name


def read_verified_chunked_asset(
    project_root: Path,
    asset: dict[str, Any],
) -> bytes:
    parts = asset.get("parts")
    require(isinstance(parts, list) and parts, "static visual asset has no parts")
    chunks: list[bytes] = []
    for index, part in enumerate(parts):
        require(isinstance(part, dict), f"invalid static visual chunk {index}")
        path = resolve_chunk_path(project_root, str(part["url"]))
        sha256, size = sha256_file(path)
        require(size == part.get("size"), f"chunk size mismatch: {path}")
        require(sha256 == part.get("sha256"), f"chunk hash mismatch: {path}")
        chunks.append(path.read_bytes())
    data = b"".join(chunks)
    require(len(data) == asset.get("size"), "joined static visual byte size mismatch")
    require(sha256_bytes(data) == asset.get("sha256"), "joined static visual hash mismatch")
    return data


def ply_data_offset(data: bytes) -> tuple[int, str]:
    marker = b"end_header"
    marker_index = data.find(marker)
    require(marker_index >= 0, "PLY has no end_header")
    offset = marker_index + len(marker)
    while offset < len(data) and data[offset] in {10, 13}:
        offset += 1
    header = data[:offset].decode("ascii")
    return offset, header


def parse_graphdeco_vertices(data: bytes, asset: dict[str, Any]) -> tuple[np.ndarray, int, str]:
    offset, header = ply_data_offset(data)
    ply = PlyData.read(io.BytesIO(data))
    require("vertex" in ply, "static Gaussian PLY has no vertex element")
    vertices = ply["vertex"].data
    required = {
        "x",
        "y",
        "z",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    }
    missing = sorted(required - set(vertices.dtype.names or ()))
    require(not missing, f"static Gaussian PLY is missing fields: {missing}")
    require(len(vertices) == asset.get("vertexCount"), "static Gaussian vertex count mismatch")
    stride = int(asset.get("vertexStride") or vertices.dtype.itemsize)
    require(vertices.dtype.itemsize == stride, "static Gaussian stride differs from manifest")
    require(offset + len(vertices) * stride == len(data), "static Gaussian payload size mismatch")
    return vertices, offset, header


def load_rgb_anchor(path: Path) -> np.ndarray:
    ply = PlyData.read(path)
    require("vertex" in ply, f"RGB anchor has no vertex element: {path}")
    vertices = ply["vertex"].data
    names = set(vertices.dtype.names or ())
    require({"x", "y", "z"}.issubset(names), f"RGB anchor has no positions: {path}")
    points = np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(np.float32)
    require(len(points) > 0 and np.isfinite(points).all(), f"RGB anchor is invalid: {path}")
    return points


def axis_aligned_sigma_extents(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scales = np.exp(
        np.column_stack((vertices["scale_0"], vertices["scale_1"], vertices["scale_2"])).astype(
            np.float64
        )
    )
    quaternion = np.column_stack(
        (vertices["rot_0"], vertices["rot_1"], vertices["rot_2"], vertices["rot_3"])
    ).astype(np.float64)
    quaternion /= np.maximum(np.linalg.norm(quaternion, axis=1, keepdims=True), 1e-12)
    w, x, y, z = quaternion.T
    rotation_squared = np.stack(
        (
            (1 - 2 * (y * y + z * z)) ** 2,
            (2 * (x * y - z * w)) ** 2,
            (2 * (x * z + y * w)) ** 2,
            (2 * (x * y + z * w)) ** 2,
            (1 - 2 * (x * x + z * z)) ** 2,
            (2 * (y * z - x * w)) ** 2,
            (2 * (x * z - y * w)) ** 2,
            (2 * (y * z + x * w)) ** 2,
            (1 - 2 * (x * x + y * y)) ** 2,
        ),
        axis=1,
    ).reshape(-1, 3, 3)
    variances = np.einsum("nij,nj->ni", rotation_squared, scales * scales)
    sigma_extents = np.sqrt(np.maximum(variances, 0.0)).astype(np.float32)
    spark_extents = (sigma_extents * SPARK_MAX_STDDEV).astype(np.float32)
    return sigma_extents, spark_extents


def inside_bounds(points: np.ndarray, bounds: dict[str, list[float]]) -> np.ndarray:
    minimum = np.asarray(bounds["min"], dtype=np.float32)
    maximum = np.asarray(bounds["max"], dtype=np.float32)
    return np.all(points >= minimum, axis=1) & np.all(points <= maximum, axis=1)


def covariance_intersects_bounds(
    points: np.ndarray,
    extents: np.ndarray,
    bounds: dict[str, list[float]],
) -> np.ndarray:
    minimum = np.asarray(bounds["min"], dtype=np.float32)
    maximum = np.asarray(bounds["max"], dtype=np.float32)
    prefilter = np.all(points >= minimum - COVARIANCE_PREFILTER_PADDING, axis=1) & np.all(
        points <= maximum + COVARIANCE_PREFILTER_PADDING,
        axis=1,
    )
    intersects = np.all(points + extents >= minimum, axis=1) & np.all(
        points - extents <= maximum,
        axis=1,
    )
    return prefilter & intersects


def nearest_distances(
    points: np.ndarray,
    anchors: np.ndarray,
    *,
    point_batch: int = 128,
    anchor_batch: int = 8192,
) -> np.ndarray:
    distances_squared = np.full(len(points), np.inf, dtype=np.float64)
    anchors64 = anchors.astype(np.float64, copy=False)
    for point_start in range(0, len(points), point_batch):
        point_stop = min(len(points), point_start + point_batch)
        batch = points[point_start:point_stop].astype(np.float64, copy=False)
        best = np.full(len(batch), np.inf, dtype=np.float64)
        for anchor_start in range(0, len(anchors64), anchor_batch):
            candidates = anchors64[anchor_start : anchor_start + anchor_batch]
            difference = batch[:, None, :] - candidates[None, :, :]
            best = np.minimum(
                best, np.min(np.einsum("ijk,ijk->ij", difference, difference), axis=1)
            )
        distances_squared[point_start:point_stop] = best
    return np.sqrt(distances_squared).astype(np.float32)


def rear_aabb_masks(
    points: np.ndarray,
    manifest: dict[str, Any],
) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for object_id in REAR_IDS:
        definition = next(
            item for item in manifest["interactiveObjects"] if item["id"] == object_id
        )
        masks[object_id] = inside_bounds(points, definition["bbox"])
    return masks


def calculate_masks(
    *,
    vertices: np.ndarray,
    manifest: dict[str, Any],
    front_anchor: np.ndarray,
    rear_anchor: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    points = np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(np.float32)
    require(np.isfinite(points).all(), "static Gaussian centers contain non-finite values")
    bounds_evidence = manifest["candidateBuild"]["sceneColliderCarve"]["boundsEvidence"]
    union = bounds_evidence["unionAabb"]
    center_union = inside_bounds(points, union)
    _, spark_extents = axis_aligned_sigma_extents(vertices)
    covariance_union = covariance_intersects_bounds(points, spark_extents, union)
    covariance_indices = np.flatnonzero(covariance_union)
    candidate_points = points[covariance_indices]
    front_distances = nearest_distances(candidate_points, front_anchor)
    rear_distances = nearest_distances(candidate_points, rear_anchor)
    footprint_radius = np.max(spark_extents[covariance_indices], axis=1)
    semantic_threshold = np.minimum(MAX_SEMANTIC_DISTANCE, BASE_DISTANCE + footprint_radius)
    front_semantic = front_distances <= semantic_threshold
    rear_protected = rear_distances <= front_distances + REAR_PROTECTION_MARGIN
    semantic_indices = covariance_indices[front_semantic & ~rear_protected]
    semantic_covariance = np.zeros(len(points), dtype=bool)
    semantic_covariance[semantic_indices] = True
    masks = {
        "center-union": center_union,
        "covariance-union": covariance_union,
        "semantic-covariance": semantic_covariance,
    }
    rear_masks = rear_aabb_masks(points, manifest)
    source = bounds_evidence["sourceSplitAabb"]
    completed = bounds_evidence["completedWorldAabb"]
    support_band_minimum = float(union["max"][1]) - 0.15
    analysis: dict[str, Any] = {
        "policy": {
            "sparkMaxStdDev": SPARK_MAX_STDDEV,
            "baseDistance": BASE_DISTANCE,
            "maximumSemanticDistance": MAX_SEMANTIC_DISTANCE,
            "rearProtectionMargin": REAR_PROTECTION_MARGIN,
            "covariancePrefilterPadding": COVARIANCE_PREFILTER_PADDING,
        },
        "boundsEvidence": bounds_evidence,
        "inputStaticGaussianCount": len(points),
        "frontAnchorPointCount": len(front_anchor),
        "rearProtectionAnchorPointCount": len(rear_anchor),
        "candidateDistanceStats": {
            "covarianceCandidateCount": len(covariance_indices),
            "frontDistance": stats(front_distances),
            "rearDistance": stats(rear_distances),
            "sparkFootprintRadius": stats(footprint_radius),
            "semanticThreshold": stats(semantic_threshold),
            "rearProtectedCandidateCount": int(np.count_nonzero(front_semantic & rear_protected)),
        },
        "modes": {},
    }
    for mode, mask in masks.items():
        removed = points[mask]
        analysis["modes"][mode] = {
            "removedGaussianCount": int(np.count_nonzero(mask)),
            "retainedGaussianCount": int(len(points) - np.count_nonzero(mask)),
            "removedCenterBounds": point_bounds(removed),
            "insideSourceSplitAabb": int(np.count_nonzero(mask & inside_bounds(points, source))),
            "insideCompletedWorldAabb": int(
                np.count_nonzero(mask & inside_bounds(points, completed))
            ),
            "supportBandRemovedCount": int(
                np.count_nonzero(mask & (points[:, 1] >= support_band_minimum))
            ),
            "rearAabbRemovedCounts": {
                object_id: int(np.count_nonzero(mask & rear_mask))
                for object_id, rear_mask in rear_masks.items()
            },
        }
    return masks, analysis


def stats(values: np.ndarray) -> dict[str, float | None]:
    if len(values) == 0:
        return {"minimum": None, "median": None, "p95": None, "maximum": None}
    return {
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(np.max(values)),
    }


def point_bounds(points: np.ndarray) -> dict[str, list[float]] | None:
    if len(points) == 0:
        return None
    return {"min": points.min(axis=0).tolist(), "max": points.max(axis=0).tolist()}


def chunk_bytes(
    data: bytes,
    *,
    output_dir: Path,
    public_prefix: str,
    stem: str,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    parts: list[dict[str, Any]] = []
    for index, start in enumerate(range(0, len(data), CHUNK_SIZE)):
        chunk = data[start : start + CHUNK_SIZE]
        name = f"{stem}.chunk{index:03d}"
        path = output_dir / name
        temporary = path.with_name(f".{name}.{os.getpid()}.tmp")
        temporary.write_bytes(chunk)
        os.replace(temporary, path)
        parts.append(
            {
                "url": f"{public_prefix.rstrip('/')}/{name}",
                "size": len(chunk),
                "sha256": sha256_bytes(chunk),
            }
        )
    return parts


def materialize(
    *,
    raw: bytes,
    data_offset: int,
    header: str,
    keep: np.ndarray,
    manifest: dict[str, Any],
    mode: str,
    analysis: dict[str, Any],
    source_manifest_path: Path,
    source_manifest_sha256: str,
    output_manifest: Path,
    chunk_dir: Path,
    chunk_url_prefix: str,
    report_path: Path,
) -> dict[str, Any]:
    candidate = json.loads(json.dumps(manifest))
    visual = candidate["assets"]["visual"]
    input_count = int(visual["vertexCount"])
    stride = int(visual["vertexStride"])
    retained_count = int(np.count_nonzero(keep))
    removed_count = input_count - retained_count
    require(removed_count > 0, f"{mode} removed no static Gaussians")
    output_header = header.replace(
        f"element vertex {input_count}", f"element vertex {retained_count}"
    )
    require(output_header != header, "failed to update static Gaussian PLY vertex count")
    payload = np.frombuffer(raw, dtype=np.uint8, offset=data_offset).reshape(input_count, stride)
    output = output_header.encode("ascii") + payload[keep].tobytes()
    stem = f"visual_bedroom4_static_front_recarve_{mode.replace('-', '_')}"
    parts = chunk_bytes(
        output,
        output_dir=chunk_dir,
        public_prefix=chunk_url_prefix,
        stem=stem,
    )
    prior_removed = int(visual.get("removedVertexCount") or 0)
    visual.update(
        {
            "id": f"{visual['id']}_front_residual_{mode}",
            "label": f"{visual['label']} with covariance-aware front-pillow residual carve",
            "vertexCount": retained_count,
            "inputVertexCount": input_count,
            "removedVertexCount": prior_removed + removed_count,
            "latestRemovedVertexCount": removed_count,
            "removedByObject": {
                **visual.get("removedByObject", {}),
                f"{FRONT_ID}_static_residual": removed_count,
            },
            "carveMethod": f"{visual.get('carveMethod')}+{mode}",
            "frontResidualCarve": {
                "status": "experimental_candidate",
                "mode": mode,
                **analysis["policy"],
                **analysis["modes"][mode],
            },
            "size": len(output),
            "sha256": sha256_bytes(output),
            "headerByteLength": len(output_header.encode("ascii")),
            "parts": parts,
            "chunkSize": CHUNK_SIZE,
        }
    )
    build = candidate["interactiveObjectBuild"]
    build["staticSceneGaussianCount"] = retained_count
    build["removedSceneGaussianCount"] = int(build["removedSceneGaussianCount"]) + removed_count
    build["totalVisualPrimitiveCount"] = int(build["totalVisualPrimitiveCount"]) - removed_count
    primitive_counts = candidate["candidateBuild"]["primitiveCounts"]
    primitive_counts["staticSceneGaussians"] = retained_count
    primitive_counts["totalVisualPrimitives"] = (
        int(primitive_counts["totalVisualPrimitives"]) - removed_count
    )
    candidate["candidateBuild"]["staticVisualResidualCarve"] = {
        "status": "experimental_candidate",
        "mode": mode,
        "report": report_path.as_posix(),
        **analysis["modes"][mode],
    }
    candidate["version"] = f"{candidate['version']}-static-recarve-{mode}"
    atomic_write_json(output_manifest, candidate)
    manifest_sha, manifest_size = sha256_file(output_manifest)
    report = {
        "schemaVersion": 1,
        "kind": "video2world.bedroom4_front_static_gaussian_recarve",
        "status": "experimental_candidate_materialized",
        "mode": mode,
        "inputManifest": {
            "path": str(source_manifest_path),
            "sha256": source_manifest_sha256,
        },
        "outputManifest": {
            "path": str(output_manifest),
            "sha256": manifest_sha,
            "bytes": manifest_size,
        },
        "staticVisual": {
            "inputVertexCount": input_count,
            "outputVertexCount": retained_count,
            "removedVertexCount": removed_count,
            "outputSha256": visual["sha256"],
            "outputBytes": visual["size"],
            "chunkCount": len(parts),
            "parts": parts,
        },
        "analysis": analysis,
        "limitations": [
            "This is an unpromoted candidate; canonical chunks and manifest are unchanged.",
            (
                "Rear RGB anchors protect rear-pillow evidence, but no completed hidden bed "
                "geometry is claimed."
            ),
            "Browser before/after review is required before selecting this carve mode.",
        ],
    }
    atomic_write_json(report_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project_root / "web/public/worlds/bedroom4/manifest.unified-pillow.json",
    )
    parser.add_argument("--mode", choices=MODES, default="semantic-covariance")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "web/public/worlds/bedroom4/recarve-candidates",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=(
            project_root
            / "examples/bedroom4/completion/trellis2_pillow_front_seed42/composition-review/"
            "static-recarve-report.json"
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project_root = args.project_root.resolve()
    manifest_path = args.manifest.resolve()
    manifest_sha_before, _ = sha256_file(manifest_path)
    manifest = read_json(manifest_path)
    visual = manifest["assets"]["visual"]
    raw = read_verified_chunked_asset(project_root, visual)
    vertices, data_offset, header = parse_graphdeco_vertices(raw, visual)
    completion = (
        project_root / "examples/bedroom4/completion/bedroom4_frame64_three_pillows/objects"
    )
    front_anchor = load_rgb_anchor(completion / FRONT_ID / f"{FRONT_ID}.ply")
    rear_anchor = np.concatenate(
        [load_rgb_anchor(completion / object_id / f"{object_id}.ply") for object_id in REAR_IDS],
        axis=0,
    )
    masks, analysis = calculate_masks(
        vertices=vertices,
        manifest=manifest,
        front_anchor=front_anchor,
        rear_anchor=rear_anchor,
    )
    output_root = args.output_root.resolve() / args.mode
    output_manifest = output_root / "manifest.json"
    report_path = args.report.resolve()
    if args.analyze_only:
        report = {
            "schemaVersion": 1,
            "kind": "video2world.bedroom4_front_static_gaussian_recarve_analysis",
            "status": "analysis_only",
            "manifest": {"path": str(manifest_path), "sha256": manifest_sha_before},
            "analysis": analysis,
        }
        atomic_write_json(report_path, report)
    else:
        public_prefix = f"./worlds/bedroom4/recarve-candidates/{args.mode}/chunks"
        report = materialize(
            raw=raw,
            data_offset=data_offset,
            header=header,
            keep=~masks[args.mode],
            manifest=manifest,
            mode=args.mode,
            analysis=analysis,
            source_manifest_path=manifest_path,
            source_manifest_sha256=manifest_sha_before,
            output_manifest=output_manifest,
            chunk_dir=output_root / "chunks",
            chunk_url_prefix=public_prefix,
            report_path=report_path,
        )
    manifest_sha_after, _ = sha256_file(manifest_path)
    require(manifest_sha_after == manifest_sha_before, "canonical manifest changed during recarve")
    print(
        json.dumps(
            {
                "status": report["status"],
                "mode": args.mode,
                "canonical_manifest_unchanged": True,
                "analysis": analysis["modes"],
                "output_manifest": None if args.analyze_only else str(output_manifest),
                "report": str(report_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
