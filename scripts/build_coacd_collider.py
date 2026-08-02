#!/usr/bin/env python3
"""Build and audit an explicit CoACD convex-decomposition collider."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from video2world.hashing import atomic_write_json, sha256_file


class CoACDColliderError(RuntimeError):
    """Raised when CoACD or a collider technical gate fails."""


def artifact(path: Path) -> dict[str, Any]:
    digest, size = sha256_file(path)
    return {"path": str(path.resolve()), "sha256": digest, "size_bytes": size}


def load_mesh_parts(path: Path, *, split_connected: bool = False) -> list[trimesh.Trimesh]:
    try:
        loaded = trimesh.load(path, force="scene", process=False)
    except Exception as exc:
        raise CoACDColliderError(f"cannot load mesh {path}: {exc}") from exc
    scene = loaded if isinstance(loaded, trimesh.Scene) else trimesh.Scene(loaded)
    parts: list[trimesh.Trimesh] = []
    for node in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node]
        geometry = scene.geometry[geometry_name]
        if not isinstance(geometry, trimesh.Trimesh) or not len(geometry.faces):
            continue
        mesh = geometry.copy()
        mesh.apply_transform(np.asarray(transform, dtype=np.float64))
        if split_connected:
            parts.extend(mesh.split(only_watertight=False))
        else:
            parts.append(mesh)
    if not parts:
        raise CoACDColliderError(f"mesh contains no triangle geometry: {path}")
    return parts


def inspect_parts(parts: list[trimesh.Trimesh], *, require_convex: bool) -> dict[str, Any]:
    records = []
    all_vertices = []
    for index, mesh in enumerate(parts):
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces)
        finite = bool(
            vertices.ndim == 2 and vertices.shape[1:] == (3,) and np.isfinite(vertices).all()
        )
        valid_faces = bool(
            faces.ndim == 2
            and faces.shape[1:] == (3,)
            and len(faces) > 0
            and np.issubdtype(faces.dtype, np.integer)
            and faces.min(initial=0) >= 0
            and faces.max(initial=-1) < len(vertices)
        )
        extents = np.ptp(vertices, axis=0) if finite and len(vertices) else np.zeros(3)
        positive_extents = bool(np.all(extents > 0))
        convex = bool(mesh.is_convex) if finite and valid_faces else False
        watertight = bool(mesh.is_watertight) if finite and valid_faces else False
        passed = finite and valid_faces and positive_extents and watertight
        if require_convex:
            passed = passed and convex
        records.append(
            {
                "part": index,
                "vertices": len(vertices),
                "faces": len(faces),
                "finite_vertices": finite,
                "valid_triangle_indices": valid_faces,
                "positive_extents": positive_extents,
                "extents": extents.tolist(),
                "watertight": watertight,
                "convex": convex,
                "technical_status": "passed" if passed else "failed",
            }
        )
        if finite:
            all_vertices.append(vertices)
    merged = np.concatenate(all_vertices, axis=0)
    minimum = merged.min(axis=0)
    maximum = merged.max(axis=0)
    gates = {
        "nonempty_parts": bool(records),
        "finite_vertices": all(record["finite_vertices"] for record in records),
        "valid_triangle_indices": all(record["valid_triangle_indices"] for record in records),
        "positive_extents": all(record["positive_extents"] for record in records),
        "watertight_parts": all(record["watertight"] for record in records),
        "convex_parts": (all(record["convex"] for record in records) if require_convex else True),
    }
    return {
        "part_count": len(records),
        "vertices": sum(record["vertices"] for record in records),
        "faces": sum(record["faces"] for record in records),
        "bounds": [minimum.tolist(), maximum.tolist()],
        "extents": (maximum - minimum).tolist(),
        "parts": records,
        "technical_gates": gates,
        "technical_status": "passed" if all(gates.values()) else "failed",
        "failed_gates": sorted(name for name, passed in gates.items() if not passed),
    }


def build_coacd_collider(
    input_mesh: str | Path,
    output_obj: str | Path,
    receipt_path: str | Path,
    *,
    coacd_executable: str | Path,
    threshold: float = 0.05,
    resolution: int = 2000,
    maximum_hulls: int = 32,
    maximum_vertices_per_hull: int = 64,
    seed: int = 42,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    source = Path(input_mesh).expanduser().resolve()
    output = Path(output_obj).expanduser().resolve()
    receipt_file = Path(receipt_path).expanduser().resolve()
    executable = Path(coacd_executable).expanduser().resolve()
    if not source.is_file():
        raise CoACDColliderError(f"input mesh does not exist: {source}")
    if not executable.is_file():
        raise CoACDColliderError(f"CoACD executable does not exist: {executable}")
    if not 0.01 <= threshold <= 1.0:
        raise CoACDColliderError("threshold must be inside [0.01, 1.0]")
    if resolution < 10 or maximum_hulls < 1 or maximum_vertices_per_hull < 4:
        raise CoACDColliderError("invalid CoACD resolution or hull limits")
    if seed < 0:
        raise CoACDColliderError("seed must be non-negative")
    # EmbodiedGen's audited runtime is Python 3.10, where datetime.UTC is unavailable.
    timestamp = created_at or datetime.now(timezone.utc)  # noqa: UP017
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise CoACDColliderError("created_at must include timezone information")

    input_audit = inspect_parts(load_mesh_parts(source), require_convex=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(executable),
        "-i",
        str(source),
        "-o",
        str(output),
        "-t",
        str(threshold),
        "-r",
        str(resolution),
        "-d",
        "-dt",
        str(maximum_vertices_per_hull),
        "-c",
        str(maximum_hulls),
        "--seed",
        str(seed),
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise CoACDColliderError(
            f"CoACD failed with code {completed.returncode}: {completed.stderr.strip()}"
        )
    if not output.is_file() or output.stat().st_size <= 0:
        raise CoACDColliderError("CoACD did not produce a non-empty collider")
    # CoACD's OBJ writer may serialize all hulls as one geometry. Each connected
    # component is the actual convex-decomposition part and must be audited alone.
    collider_audit = inspect_parts(
        load_mesh_parts(output, split_connected=True), require_convex=True
    )
    if collider_audit["technical_status"] != "passed":
        raise CoACDColliderError(
            "CoACD collider failed technical gates: " + ", ".join(collider_audit["failed_gates"])
        )
    receipt = {
        "schema_version": 1,
        "kind": "video2world.coacd_collider_receipt",
        "created_at": timestamp.isoformat(),
        "status": "technical_passed_visual_pending",
        "promotion_allowed": False,
        "input_visual_mesh": artifact(source),
        "input_audit": input_audit,
        "tool": {
            **artifact(executable),
            "name": "CoACD",
            "parameters": {
                "threshold": threshold,
                "resolution": resolution,
                "maximum_hulls": maximum_hulls,
                "maximum_vertices_per_hull": maximum_vertices_per_hull,
                "seed": seed,
            },
        },
        "output_collider": {
            **artifact(output),
            "role": "collision_proxy",
            "media_type": "model/obj",
            "collider_topology": "convex_decomposition",
            "provenance": {"decomposition": "coacd"},
        },
        "collider_audit": collider_audit,
        "runtime": {"elapsed_seconds": elapsed, "returncode": completed.returncode},
        "claim_boundary": (
            "The OBJ is an independent collision proxy. It is not the visual PBR mesh, "
            "does not carry texture authority, and remains pending runtime collision QA."
        ),
    }
    receipt_file.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(receipt_file, receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-mesh", type=Path, required=True)
    parser.add_argument("--output-obj", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--coacd-executable", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--resolution", type=int, default=2000)
    parser.add_argument("--maximum-hulls", type=int, default=32)
    parser.add_argument("--maximum-vertices-per-hull", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = build_coacd_collider(
            args.input_mesh,
            args.output_obj,
            args.receipt,
            coacd_executable=args.coacd_executable,
            threshold=args.threshold,
            resolution=args.resolution,
            maximum_hulls=args.maximum_hulls,
            maximum_vertices_per_hull=args.maximum_vertices_per_hull,
            seed=args.seed,
        )
    except CoACDColliderError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "collider": receipt["output_collider"]["path"],
                "parts": receipt["collider_audit"]["part_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
