#!/usr/bin/env python3
"""Build a candidate-only Bedroom 4 manifest from hash-bound unified PBR GLBs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from video2world.hashing import atomic_write_json, sha256_file

CANDIDATE_VERSION = "bedroom4-candidate-unified-pbr-four-objects-20260717"
CANDIDATE_CREATED_AT = "2026-07-17T15:30:00+08:00"
BASE_MANIFEST_SHA256 = "7e0a1dcd801225bf21bd5a77e82edcce2318e348523d69ca80b7d1e5b53dc96d"
STABLE_MANIFEST_SHA256 = "3805f0e5bab09add424b3b78f9349cd2eca6d1262777ef683e13cda07695e82b"
PAIRWISE_REVIEW_SHA256 = "b8928ffa92aaee81ef2abf76f2237da1bb893770fd6e50e4f642db8620204c74"
JOINT_REVIEW_SHA256 = "48c4ba43baa8b0767772e70fbf0165aeb6531018d4e0c172a0525d2760456d28"
SUPPORT_RECEIPT_SHA256 = "9d51f3ec09869ee6ca9ca4ffc21e77f25cf350d02665f682b241dda34778b320"
MAX_COLLISION_FACES = 100_000
PILLOW_IDS = ("sam3_pillow_front", "sam3_pillow_left", "sam3_pillow_right")
ADOPTED_IDS = (*PILLOW_IDS, "sam3_bed_01")


@dataclass(frozen=True)
class ObjectSpec:
    object_id: str
    label: str
    name: str
    category: str
    aliases: tuple[str, ...]
    mesh_relpath: str
    scene_fit_relpath: str
    mesh_sha256: str
    scene_fit_sha256: str
    size_bytes: int
    vertices: int
    faces: int
    topology: str
    watertight: bool
    asset_coordinates_baked: bool
    logical_root: str


@dataclass(frozen=True)
class MeshAudit:
    size_bytes: int
    sha256: str
    vertices: int
    faces: int
    bounds: list[list[float]]
    finite: bool
    nondegenerate: bool
    winding_consistent: bool
    watertight: bool
    geometry_count: int


@dataclass(frozen=True)
class VerifiedObject:
    spec: ObjectSpec
    mesh_path: Path
    scene_fit_path: Path
    scene_fit: dict[str, Any]
    mesh: MeshAudit
    pivot: list[float]
    matrix_row_major: list[list[float]] | None
    world_bounds: list[list[float]]


OBJECT_SPECS = (
    ObjectSpec(
        object_id="sam3_pillow_front",
        label="前排浅色枕头 / Front light pillow",
        name="前排浅色枕头",
        category="pillow",
        aliases=("pillow", "front pillow", "small pillow", "枕头", "前排枕头", "小白枕"),
        mesh_relpath=(
            "examples/bedroom4/completion/trellis2_pillow_front_seed42/"
            "scene_fit_silhouette_refined/sam3_pillow_front.scene-fit.glb"
        ),
        scene_fit_relpath=(
            "examples/bedroom4/completion/trellis2_pillow_front_seed42/"
            "scene_fit_silhouette_refined/scene_fit_report.json"
        ),
        mesh_sha256="9d84ab4561d7c017c24c07a94abe5400073b0fafb97893a8a1eafca320c391d6",
        scene_fit_sha256="da26014b1ecc29eb1dfcf8d45b74d98bd539b69945e029a6512ea6a2ab8769cd",
        size_bytes=3_979_924,
        vertices=60_237,
        faces=97_082,
        topology="surface_bvh",
        watertight=False,
        asset_coordinates_baked=True,
        logical_root="world",
    ),
    ObjectSpec(
        object_id="sam3_pillow_left",
        label="左侧后排浅色枕头 / Left rear light pillow",
        name="左侧后排浅色枕头",
        category="pillow",
        aliases=("left pillow", "left rear pillow", "左枕头", "左侧后排枕头"),
        mesh_relpath=(
            "examples/bedroom4/completion/layered-peel/round02_left_pillow/trellis2_seed44/"
            "scene_fit_silhouette_refined_v6/sam3_pillow_left.scene-fit.glb"
        ),
        scene_fit_relpath=(
            "examples/bedroom4/completion/layered-peel/round02_left_pillow/trellis2_seed44/"
            "scene_fit_silhouette_refined_v6/scene_fit_report.json"
        ),
        mesh_sha256="43f95a5756ccecb9a1ca36ba583c2ad4fd5dd1a4e66b1d591be935f2bd59b600",
        scene_fit_sha256="1f551500f5a540dfd46b9aaba33a503f36efc47c5e7169dc953ed779f40a98c7",
        size_bytes=3_440_992,
        vertices=52_981,
        faces=98_230,
        topology="surface_bvh",
        watertight=False,
        asset_coordinates_baked=True,
        logical_root="world",
    ),
    ObjectSpec(
        object_id="sam3_pillow_right",
        label="右侧后排浅色枕头 / Right rear light pillow",
        name="右侧后排浅色枕头",
        category="pillow",
        aliases=("right pillow", "right rear pillow", "右枕头", "右侧后排枕头"),
        mesh_relpath=(
            "examples/bedroom4/completion/layered-peel/round03_right_pillow/trellis2_seed43/"
            "scene_fit_silhouette_refined_v6/sam3_pillow_right.scene-fit.glb"
        ),
        scene_fit_relpath=(
            "examples/bedroom4/completion/layered-peel/round03_right_pillow/trellis2_seed43/"
            "scene_fit_silhouette_refined_v6/scene_fit_report.json"
        ),
        mesh_sha256="eb4ad2cb0398602865ff3511039e440ce9cb50ace016fffeaaf3ce3944280f1e",
        scene_fit_sha256="d9ec2811d33372cfda174b7073e574385da283bc191a09fa89cbc6175da41155",
        size_bytes=3_406_068,
        vertices=54_968,
        faces=98_226,
        topology="surface_bvh",
        watertight=False,
        asset_coordinates_baked=True,
        logical_root="world",
    ),
    ObjectSpec(
        object_id="sam3_bed_01",
        label="双人床 / Bed",
        name="双人床",
        category="bed",
        aliases=("bed", "double bed", "床", "双人床", "床架", "床垫"),
        mesh_relpath=(
            "examples/bedroom4/completion/layered-peel/round04_bed/"
            "component_assembly_v2_support_adjusted/sam3_bed_01.support-adjusted.glb"
        ),
        scene_fit_relpath=(
            "examples/bedroom4/completion/layered-peel/round04_bed/"
            "component_assembly_v2_support_adjusted/scene_fit_validated/scene_fit_report.json"
        ),
        mesh_sha256="ad9eca71a676cf8fa5cad7a38d0ff5a623149de01e61ae91ba651ea07d37790e",
        scene_fit_sha256="2f3b8b29d2270964fe8931c4826b2b126e60266cfd97d157af5b8cdd971a6adc",
        size_bytes=113_468,
        vertices=2_500,
        faces=4_976,
        topology="closed_volume",
        watertight=True,
        asset_coordinates_baked=False,
        logical_root="sam3_bed_01",
    ),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def require_sha256(path: Path, expected: str) -> tuple[str, int]:
    actual, size = sha256_file(path)
    require(actual == expected, f"sha256 mismatch for {path}: {actual} != {expected}")
    return actual, size


def _same_vector(left: Any, right: Any, *, tolerance: float = 1e-9) -> bool:
    if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
        return False
    return all(
        math.isfinite(float(a))
        and math.isfinite(float(b))
        and math.isclose(float(a), float(b), rel_tol=0, abs_tol=tolerance)
        for a, b in zip(left, right, strict=True)
    )


def _same_matrix(left: Any, right: Any) -> bool:
    return (
        isinstance(left, list)
        and isinstance(right, list)
        and len(left) == len(right) == 4
        and all(_same_vector(a, b) for a, b in zip(left, right, strict=True))
    )


def _ref_sha256(value: Any, label: str) -> str:
    require(isinstance(value, dict), f"{label} must be an object")
    digest = value.get("sha256")
    require(
        isinstance(digest, str) and len(digest) == 64,
        f"{label}.sha256 must be a digest",
    )
    return digest


def audit_glb(path: Path, spec: ObjectSpec) -> MeshAudit:
    digest, size = require_sha256(path, spec.mesh_sha256)
    require(size == spec.size_bytes, f"GLB byte count mismatch for {spec.object_id}")
    try:
        scene = trimesh.load(path, force="scene", process=False)
    except Exception as exc:  # pragma: no cover - library error detail is environment-specific
        raise RuntimeError(f"cannot inspect GLB {path}: {exc}") from exc
    require(isinstance(scene, trimesh.Scene), f"GLB did not load as a scene: {path}")
    geometries = list(scene.geometry.values())
    require(geometries, f"GLB has no mesh geometry: {path}")
    vertices = sum(len(geometry.vertices) for geometry in geometries)
    faces = sum(len(geometry.faces) for geometry in geometries)
    finite = all(np.isfinite(geometry.vertices).all() for geometry in geometries)
    nondegenerate = all(
        len(geometry.faces) > 0 and bool(np.all(geometry.area_faces > 1e-12))
        for geometry in geometries
    )
    winding_consistent = all(geometry.is_winding_consistent for geometry in geometries)
    watertight = all(geometry.is_watertight for geometry in geometries)
    require(vertices == spec.vertices, f"GLB vertex count mismatch for {spec.object_id}")
    require(faces == spec.faces, f"GLB face count mismatch for {spec.object_id}")
    require(0 < faces <= MAX_COLLISION_FACES, f"GLB collision face budget failed: {spec.object_id}")
    require(finite, f"GLB contains non-finite vertices: {spec.object_id}")
    require(nondegenerate, f"GLB contains degenerate faces: {spec.object_id}")
    require(winding_consistent, f"GLB winding is inconsistent: {spec.object_id}")
    require(watertight is spec.watertight, f"GLB watertight claim mismatch: {spec.object_id}")
    require(scene.bounds is not None, f"GLB has no bounds: {spec.object_id}")
    return MeshAudit(
        size_bytes=size,
        sha256=digest,
        vertices=vertices,
        faces=faces,
        bounds=np.asarray(scene.bounds, dtype=np.float64).tolist(),
        finite=finite,
        nondegenerate=nondegenerate,
        winding_consistent=winding_consistent,
        watertight=watertight,
        geometry_count=len(geometries),
    )


def _verify_scene_fit(spec: ObjectSpec, report: dict[str, Any], mesh: MeshAudit) -> None:
    require(
        report.get("object_id") == spec.object_id, f"scene-fit object mismatch: {spec.object_id}"
    )
    require(
        report.get("all_acceptance_gates_passed") is True,
        f"scene-fit gates failed: {spec.object_id}",
    )
    require(
        report.get("status") in {"passed", "technical_gates_passed"},
        f"scene-fit status failed: {spec.object_id}",
    )
    collision = report.get("collision_semantics")
    require(isinstance(collision, dict), f"scene-fit collision semantics missing: {spec.object_id}")
    require(collision.get("asset_mode") == "unified_glb", f"asset mode mismatch: {spec.object_id}")
    require(collision.get("topology") == spec.topology, f"topology mismatch: {spec.object_id}")
    require(
        int(collision.get("max_faces", MAX_COLLISION_FACES)) == MAX_COLLISION_FACES,
        "face budget mismatch",
    )
    report_mesh = report.get("mesh")
    require(isinstance(report_mesh, dict), f"scene-fit mesh missing: {spec.object_id}")
    require(
        int(report_mesh.get("vertices", -1)) == mesh.vertices,
        f"receipt vertices mismatch: {spec.object_id}",
    )
    require(
        int(report_mesh.get("faces", -1)) == mesh.faces, f"receipt faces mismatch: {spec.object_id}"
    )
    require(
        report_mesh.get("watertight") is mesh.watertight,
        f"receipt watertight mismatch: {spec.object_id}",
    )
    require(
        report_mesh.get("winding_consistent") is True,
        f"receipt winding mismatch: {spec.object_id}",
    )
    if spec.asset_coordinates_baked:
        require(
            report_mesh.get("glb_sha256") == mesh.sha256,
            f"receipt GLB hash mismatch: {spec.object_id}",
        )
        require(
            int(report_mesh.get("glb_bytes", -1)) == mesh.size_bytes,
            f"receipt GLB size mismatch: {spec.object_id}",
        )
    else:
        runtime = report.get("runtime_transform")
        require(isinstance(runtime, dict), "bed runtime transform is missing")
        require(runtime.get("asset_coordinates_baked") is False, "bed must remain unbaked")
        require(
            _ref_sha256(report.get("sources", {}).get("mesh"), "bed scene-fit mesh") == mesh.sha256,
            "bed scene-fit mesh hash mismatch",
        )
        logical = report.get("logical_entity")
        require(isinstance(logical, dict), "bed logical entity is missing")
        require(logical.get("root_node") == spec.logical_root, "bed logical root mismatch")


def _verify_review_chain(
    project_root: Path,
    pairwise: dict[str, Any],
    joint: dict[str, Any],
    support: dict[str, Any],
    verified: dict[str, VerifiedObject],
) -> None:
    require(pairwise.get("status") == "passed", "pillow pairwise review is not passed")
    require(pairwise.get("promotion_allowed") is True, "pillow pairwise review blocks adoption")
    require(joint.get("status") == "passed", "bed-pillow joint review is not passed")
    require(joint.get("promotion_allowed") is True, "bed-pillow joint review blocks adoption")
    require(joint.get("promotion_blockers") == [], "bed-pillow joint review has blockers")
    require(support.get("status") == "technical_gates_passed", "support adjustment is not passed")
    require(support.get("all_acceptance_gates_passed") is True, "support gates failed")
    require(
        all(support.get("acceptance_gates", {}).values()), "support receipt contains a failed gate"
    )

    pairwise_sources = pairwise.get("sources", {}).get("objects", {})
    joint_pillows = joint.get("sources", {}).get("pillows", {})
    joint_contract = joint.get("joint_asset_contract", {})
    pillow_contract = joint_contract.get("pillows", {})
    for object_id in PILLOW_IDS:
        item = verified[object_id]
        for source_set, label in ((pairwise_sources, "pairwise"), (joint_pillows, "joint")):
            refs = source_set.get(object_id, {})
            require(
                _ref_sha256(refs.get("mesh"), f"{label} {object_id} mesh") == item.mesh.sha256,
                f"{label} mesh hash mismatch: {object_id}",
            )
            require(
                _ref_sha256(refs.get("scene_fit_report"), f"{label} {object_id} scene fit")
                == item.spec.scene_fit_sha256,
                f"{label} scene-fit hash mismatch: {object_id}",
            )
        contract = pillow_contract.get(object_id, {})
        require(
            contract.get("asset_coordinates_baked") is True, f"pillow is not baked: {object_id}"
        )
        require(
            _same_vector(contract.get("runtime_pivot"), item.pivot),
            f"pillow pivot mismatch: {object_id}",
        )
        require(
            _same_vector(contract.get("runtime_scale"), [1.0, 1.0, 1.0]),
            f"pillow scale mismatch: {object_id}",
        )
        require(
            _same_vector(contract.get("runtime_rotation_euler_deg"), [0.0, 0.0, 0.0]),
            f"pillow rotation mismatch: {object_id}",
        )
        require(
            joint.get("mattress_contacts", {}).get(object_id, {}).get("status") == "passed",
            f"mattress contact failed: {object_id}",
        )
        require(
            joint.get("headboard_relations", {}).get(object_id, {}).get("status") == "passed",
            f"headboard relation failed: {object_id}",
        )

    bed = verified["sam3_bed_01"]
    bed_sources = joint.get("sources", {}).get("bed", {})
    require(
        _ref_sha256(bed_sources.get("mesh"), "joint bed mesh") == bed.mesh.sha256,
        "joint bed mesh mismatch",
    )
    require(
        _ref_sha256(bed_sources.get("scene_fit_report"), "joint bed scene fit")
        == bed.spec.scene_fit_sha256,
        "joint bed scene-fit mismatch",
    )
    require(
        _ref_sha256(bed_sources.get("support_adjustment_receipt"), "joint support receipt")
        == SUPPORT_RECEIPT_SHA256,
        "joint support receipt mismatch",
    )
    bed_contract = joint_contract.get("bed", {})
    require(
        bed_contract.get("asset_coordinates_baked") is False,
        "joint contract incorrectly marks bed baked",
    )
    require(
        _same_matrix(bed_contract.get("runtime_transform_matrix_row_major"), bed.matrix_row_major),
        "joint bed matrix mismatch",
    )
    require(
        joint.get("bed_floor_support_gate", {}).get("status") == "passed",
        "bed floor support failed",
    )
    require(
        joint.get("bed_support_adjustment_gate", {}).get("status") == "passed",
        "bed support adjustment failed",
    )
    require(
        joint.get("pillow_pairwise_gate", {}).get("status") == "passed",
        "joint pairwise gate failed",
    )

    output = support.get("output", {}).get("unified_pbr_glb", {})
    require(
        output.get("logical_root") == bed.spec.logical_root, "support receipt logical root mismatch"
    )
    require(output.get("sha256") == bed.mesh.sha256, "support receipt GLB hash mismatch")
    require(
        int(output.get("size_bytes", -1)) == bed.mesh.size_bytes,
        "support receipt GLB size mismatch",
    )
    require(
        _same_matrix(
            support.get("scene_fit_transform", {}).get("matrix_row_major"), bed.matrix_row_major
        ),
        "support receipt matrix mismatch",
    )
    support_children = {item.get("object_id"): item for item in support.get("children", [])}
    require(set(support_children) == set(PILLOW_IDS), "support receipt pillow set mismatch")
    for object_id in PILLOW_IDS:
        item = verified[object_id]
        child = support_children[object_id]
        require(child.get("penetration_gate") is True, f"support penetration failed: {object_id}")
        require(child.get("support_gap_gate") is True, f"support gap failed: {object_id}")
        require(
            _ref_sha256(child.get("mesh"), f"support {object_id} mesh") == item.mesh.sha256,
            f"support mesh mismatch: {object_id}",
        )
        require(
            _ref_sha256(child.get("scene_fit_report"), f"support {object_id} scene fit")
            == item.spec.scene_fit_sha256,
            f"support scene-fit mismatch: {object_id}",
        )

    # Paths are provenance only; all adoption decisions above are bound to content digests.
    require(project_root.is_dir(), "project root vanished during evidence verification")


def load_verified_objects(project_root: Path) -> tuple[dict[str, VerifiedObject], dict[str, str]]:
    pairwise_path = (
        project_root
        / "examples/bedroom4/completion/layered-peel"
        / "pairwise-scene-fit-review-v2/pairwise_scene_fit_review.json"
    )
    joint_path = (
        project_root
        / "examples/bedroom4/completion/layered-peel"
        / "bed-pillow-joint-review-v2/bed_pillow_joint_review.json"
    )
    support_path = (
        project_root
        / "examples/bedroom4/completion/layered-peel/round04_bed"
        / "component_assembly_v2_support_adjusted/support_adjustment_receipt.json"
    )
    require_sha256(pairwise_path, PAIRWISE_REVIEW_SHA256)
    require_sha256(joint_path, JOINT_REVIEW_SHA256)
    require_sha256(support_path, SUPPORT_RECEIPT_SHA256)
    pairwise = read_json(pairwise_path)
    joint = read_json(joint_path)
    support = read_json(support_path)

    verified: dict[str, VerifiedObject] = {}
    joint_contract = joint.get("joint_asset_contract", {})
    for spec in OBJECT_SPECS:
        mesh_path = project_root / spec.mesh_relpath
        scene_fit_path = project_root / spec.scene_fit_relpath
        require_sha256(scene_fit_path, spec.scene_fit_sha256)
        scene_fit = read_json(scene_fit_path)
        mesh = audit_glb(mesh_path, spec)
        _verify_scene_fit(spec, scene_fit, mesh)
        if spec.asset_coordinates_baked:
            transform = scene_fit.get("baked_relative_transform")
            require(isinstance(transform, dict), f"baked transform missing: {spec.object_id}")
            pivot = [float(value) for value in transform.get("runtime_pivot", [])]
            require(len(pivot) == 3, f"runtime pivot missing: {spec.object_id}")
            matrix = None
            world_bounds = [
                [mesh.bounds[0][axis] + pivot[axis] for axis in range(3)],
                [mesh.bounds[1][axis] + pivot[axis] for axis in range(3)],
            ]
        else:
            runtime = scene_fit.get("runtime_transform", {})
            matrix = copy.deepcopy(runtime.get("matrix_row_major"))
            require(
                _same_matrix(matrix, runtime.get("base_matrix_row_major")),
                "bed base/runtime matrix mismatch",
            )
            pivot = [float(matrix[axis][3]) for axis in range(3)]
            world_bounds = copy.deepcopy(scene_fit.get("world_bounds"))
            require(
                isinstance(world_bounds, list)
                and len(world_bounds) == 2
                and all(len(row) == 3 for row in world_bounds),
                "bed world bounds are missing",
            )
            require(
                _same_matrix(
                    matrix,
                    joint_contract.get("bed", {}).get("runtime_transform_matrix_row_major"),
                ),
                "bed runtime matrix is not bound to joint QA",
            )
        verified[spec.object_id] = VerifiedObject(
            spec=spec,
            mesh_path=mesh_path,
            scene_fit_path=scene_fit_path,
            scene_fit=scene_fit,
            mesh=mesh,
            pivot=pivot,
            matrix_row_major=matrix,
            world_bounds=world_bounds,
        )
    _verify_review_chain(project_root, pairwise, joint, support, verified)
    return verified, {
        "pairwise_review_sha256": PAIRWISE_REVIEW_SHA256,
        "joint_review_sha256": JOINT_REVIEW_SHA256,
        "support_adjustment_receipt_sha256": SUPPORT_RECEIPT_SHA256,
    }


def _bbox(bounds: list[list[float]]) -> dict[str, Any]:
    minimum, maximum = bounds
    center = [(minimum[axis] + maximum[axis]) / 2 for axis in range(3)]
    extent = [maximum[axis] - minimum[axis] for axis in range(3)]
    return {
        "coordinateFrame": "visual_native",
        "min": minimum,
        "max": maximum,
        "center": center,
        "extent": extent,
    }


def _description(item: VerifiedObject) -> dict[str, Any]:
    if item.spec.category == "bed":
        return {
            "short": {
                "zh": "一张带软垫、床架和床头板的双人床。",
                "en": "A double bed with mattress, frame, and headboard.",
            },
            "appearance": {
                "zh": "浅色床垫与床品配深色木质床架和床头板。",
                "en": "Light bedding over a dark wooden frame and headboard.",
            },
            "detailed": {
                "zh": "床由六个闭合 PBR 部件组成, 但在场景中由一个根节点统一选择、旋转和碰撞。",
                "en": "Six closed PBR parts act as one selectable, rotatable, collidable root.",
            },
            "location": {
                "zh": "位于卧室中央, 三只枕头在床垫上。",
                "en": "In the bedroom center, supporting the three pillows.",
            },
            "provider": "component_assembly+support_adjustment+joint_QA",
            "model": "deterministic PBR fixture assembly",
            "confidence": None,
            "evidence_frames": ["000064", "000067", "000071"],
        }
    side = {
        "sam3_pillow_front": "前排",
        "sam3_pillow_left": "左后排",
        "sam3_pillow_right": "右后排",
    }[item.spec.object_id]
    return {
        "short": {
            "zh": f"床上的{side}浅色软枕。",
            "en": f"A light-colored {side} pillow on the bed.",
        },
        "appearance": {
            "zh": "主体为灰白或米白色, 保留了柔软填充形态。",
            "en": "An off-white stuffed form with a soft fabric PBR surface.",
        },
        "detailed": {
            "zh": "可见面与背面已形成完整 PBR 网格; 轻微生成纹理差异按当前验收口径保留。",
            "en": "A completed PBR mesh with accepted minor generated texture variation.",
        },
        "location": {
            "zh": f"位于床垫上的{side}位置。",
            "en": f"At the {side} position on the mattress.",
        },
        "provider": "TRELLIS.2+scene_fit+joint_QA",
        "model": "TRELLIS.2 PBR completion",
        "confidence": None,
        "evidence_frames": ["000064"],
    }


def build_unified_object(item: VerifiedObject, evidence_hashes: dict[str, str]) -> dict[str, Any]:
    spec = item.spec
    is_pillow = spec.object_id in PILLOW_IDS
    placement: dict[str, Any] = {
        "coordinateFrame": "visual_native",
        "sourceCoordinateFrame": "scene_fit_local",
        "transform": (
            "scene_fit_baked_glb_plus_runtime_pivot"
            if spec.asset_coordinates_baked
            else "unbaked_glb_plus_runtime_matrix_row_major"
        ),
        "assetCoordinatesBaked": spec.asset_coordinates_baked,
        "pivot": item.pivot,
        "generatedCenter": [0, 0, 0],
        "scale": [1, 1, 1],
        "rotationEulerDeg": [0, 0, 0],
        "eulerOrder": "XYZ",
    }
    if item.matrix_row_major is not None:
        placement["matrixRowMajor"] = item.matrix_row_major
    file_name = item.mesh_path.name
    return {
        "id": spec.object_id,
        "label": spec.label,
        "name": spec.name,
        "category": spec.category,
        "aliases": list(spec.aliases),
        "description": _description(item),
        "fidelity": "accepted_unified_pbr_completion_candidate",
        "bbox": _bbox(item.world_bounds),
        "independentlyRecognized": True,
        "visualOnly": False,
        "semanticGranularity": (
            "independent_child_asset" if is_pillow else "independent_root_asset"
        ),
        "parentObjectId": "sam3_bed_01" if is_pillow else None,
        "movesWithParent": is_pillow,
        "childObjectIds": [] if is_pillow else list(PILLOW_IDS),
        "independentlyMovable": True,
        "placement": placement,
        "collision": {
            "mode": "unified-glb",
            "topology": spec.topology,
            "walkable": False,
            "characterCollision": True,
            "asset": {
                "id": f"{spec.object_id}_unified_pbr_glb",
                "label": f"{spec.label} unified PBR GLB",
                "url": f"./worlds/bedroom4/objects/{file_name}",
                "fileName": file_name,
                "fileType": "glb",
                "format": "gltf-binary",
                "size": item.mesh.size_bytes,
                "sha256": item.mesh.sha256,
                "vertices": item.mesh.vertices,
                "faces": item.mesh.faces,
                "bounds": item.mesh.bounds,
                "finite": item.mesh.finite,
                "nondegenerate": item.mesh.nondegenerate,
                "windingConsistent": item.mesh.winding_consistent,
                "watertight": item.mesh.watertight,
                "logicalRoot": spec.logical_root,
                "sourcePath": f"repo://video2world/{spec.mesh_relpath}",
            },
            "gate": {
                "status": "passed",
                "surfaceCollision": "passed",
                "sceneFitReceiptSha256": spec.scene_fit_sha256,
                "pairwiseQaReceiptSha256": evidence_hashes["pairwise_review_sha256"],
                "jointQaReceiptSha256": evidence_hashes["joint_review_sha256"],
                "supportAdjustmentReceiptSha256": (
                    evidence_hashes["support_adjustment_receipt_sha256"]
                    if spec.object_id == "sam3_bed_01"
                    else None
                ),
                "candidateBrowserQa": "pending",
                "reason": (
                    "The exact GLB and its scene-fit/joint QA receipts passed geometry and "
                    "surface-collision gates. Candidate transport/browser QA remains pending."
                ),
            },
        },
        "interaction": {
            "kind": "spin",
            "degrees": 360,
            "durationMs": 1250,
            "drag": "horizontal_yaw",
        },
        "limitations": [
            "Candidate asset transport and browser QA are pending.",
            (
                "The inherited static scene is not a clean plate and can still contain old "
                "object residuals."
            ),
        ],
    }


def _knowledge_object(interactive: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": interactive["id"],
        "name": interactive["name"],
        "category": interactive["category"],
        "aliases": interactive["aliases"],
        "description": interactive["description"],
        "bbox": interactive["bbox"],
        "interaction": {
            "selectable": True,
            "movable": True,
            "drag_action": "rotate_yaw",
            "double_click_action": "spin_360",
        },
    }


def _descriptor_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def build_candidate_manifest(
    base: dict[str, Any],
    verified: dict[str, VerifiedObject],
    evidence_hashes: dict[str, str],
    *,
    base_sha256: str,
) -> dict[str, Any]:
    require(base.get("contract") == "video2world-web-manifest-1.0.0", "base contract mismatch")
    require(base.get("schemaVersion") == 1, "base schema version mismatch")
    require(
        base.get("candidateBuild", {}).get("completionClaim")
        == "front_pillow_current_demo_candidate_only",
        "base is not the accepted unified-front-pillow candidate",
    )
    manifest = copy.deepcopy(base)
    unified = [
        build_unified_object(verified[object_id], evidence_hashes) for object_id in ADOPTED_IDS
    ]
    interactive = manifest.get("interactiveObjects")
    require(isinstance(interactive, list), "base interactiveObjects missing")
    manifest["interactiveObjects"] = [
        item for item in interactive if item.get("id") not in ADOPTED_IDS
    ] + unified
    knowledge = manifest.get("sceneKnowledge", {}).get("objects")
    require(isinstance(knowledge, list), "base scene knowledge missing")
    manifest["sceneKnowledge"]["objects"] = [
        item for item in knowledge if item.get("id") not in ADOPTED_IDS
    ] + [_knowledge_object(item) for item in unified]
    manifest["version"] = CANDIDATE_VERSION
    manifest.setdefault("initialState", {})["cameraFocusObjectId"] = "sam3_bed_01"
    manifest["candidateBuild"] = {
        "status": "candidate_materialized_clean_scene_pending",
        "promotionAllowed": False,
        "completionClaim": "four_unified_pbr_objects_candidate_only",
        "unifiedPbrObjects": {
            "status": "geometry_and_joint_qa_passed",
            "objectIds": list(ADOPTED_IDS),
            "representation": "one_pbr_glb_per_object_for_visual_logic_selection_and_collision",
            "forbiddenProxyFields": ["visual", "renderAsset", "colliderProxy"],
            "rootInteraction": ["selection", "spin_360", "horizontal_yaw_drag"],
            "hierarchy": {
                "parent": "sam3_bed_01",
                "children": list(PILLOW_IDS),
                "parentMotionCarriesChildren": True,
                "childrenRemainIndependentlyMovable": True,
            },
            "assetTransport": "pending_candidate_packaging",
        },
        "cleanScene": {
            "status": "clean_scene_pending",
            "methodRequired": "front_to_back_cumulative_peel_then_clean_plate_reconstruction",
            "inheritedStaticScene": "existing_front_pillow_residual_recarve_candidate",
            "knownUnremovedOrUnprovenResiduals": [
                "sam3_pillow_left",
                "sam3_pillow_right",
                "sam3_bed_01",
                "occluded_background",
            ],
            "claim": (
                "This candidate reuses the current static scene byte-for-byte and does not "
                "claim that the old bed, rear pillows, occlusion holes, or background were peeled."
            ),
        },
        "inheritedStaticScene": {
            "baseManifestSha256": base_sha256,
            "visualDescriptorSha256": _descriptor_sha256(base.get("assets", {}).get("visual")),
            "colliderDescriptorSha256": _descriptor_sha256(
                base.get("assets", {}).get(
                    "colliderStaticCarved", base.get("assets", {}).get("collider")
                )
            ),
            "modifiedByThisCandidate": False,
        },
        "evidence": {
            **evidence_hashes,
            "scene_fit_receipts": {
                object_id: verified[object_id].spec.scene_fit_sha256 for object_id in ADOPTED_IDS
            },
            "assets": {object_id: verified[object_id].mesh.sha256 for object_id in ADOPTED_IDS},
        },
        "promotionBlockers": [
            "clean_scene_pending",
            "candidate_asset_transport_pending",
            "candidate_browser_qa_pending",
        ],
        "report": "repo://video2world/examples/bedroom4/manifests/bedroom4.unified-pbr-objects.candidate.report.json",
    }
    production = manifest.setdefault("productionBuild", {})
    production["unifiedPbrFourObjectCandidate"] = {
        "status": "candidate_only_clean_scene_pending",
        "promotionAllowed": False,
        "objectIds": list(ADOPTED_IDS),
    }
    return manifest


def _repo_uri(project_root: Path, path: Path) -> str:
    return f"repo://video2world/{path.resolve().relative_to(project_root).as_posix()}"


def _assert_candidate_output_scope(project_root: Path, output: Path, report: Path) -> None:
    allowed = (project_root / "examples/bedroom4/manifests").resolve()
    require(
        output.resolve().parent == allowed,
        "candidate manifest must stay in examples/bedroom4/manifests",
    )
    require(
        report.resolve().parent == allowed,
        "candidate report must stay in examples/bedroom4/manifests",
    )
    require(output.resolve() != report.resolve(), "manifest and report paths must differ")


def materialize_candidate(
    *,
    project_root: Path,
    base_manifest: Path,
    output_manifest: Path,
    report_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    project_root = project_root.resolve()
    base_manifest = base_manifest.resolve()
    _assert_candidate_output_scope(project_root, output_manifest, report_path)
    base_sha, _ = require_sha256(base_manifest, BASE_MANIFEST_SHA256)
    public_manifest = project_root / "web/public/worlds/bedroom4/manifest.json"
    stable_manifest = (
        project_root / "web/public/worlds/bedroom4/manifest.web-demo-baseline-stable.json"
    )
    public_before = require_sha256(public_manifest, BASE_MANIFEST_SHA256)
    stable_before = require_sha256(stable_manifest, STABLE_MANIFEST_SHA256)

    verified, evidence_hashes = load_verified_objects(project_root)
    manifest = build_candidate_manifest(
        read_json(base_manifest),
        verified,
        evidence_hashes,
        base_sha256=base_sha,
    )
    atomic_write_json(output_manifest, manifest)
    manifest_sha, manifest_size = sha256_file(output_manifest)
    script_path = Path(__file__).resolve()
    script_sha, _ = sha256_file(script_path)
    report = {
        "schema_version": 1,
        "kind": "video2world.bedroom4_unified_pbr_objects_candidate_report",
        "created_at": CANDIDATE_CREATED_AT,
        "status": "candidate_materialized_clean_scene_pending",
        "candidate_version": CANDIDATE_VERSION,
        "promotion_allowed": False,
        "claim_scope": (
            "Four accepted PBR GLBs are adopted as unified visual, logical, selection, and "
            "collision roots. The inherited static scene is unchanged and is not a clean plate."
        ),
        "materializer": {"path": _repo_uri(project_root, script_path), "sha256": script_sha},
        "inputs": {
            "base_manifest": {"path": _repo_uri(project_root, base_manifest), "sha256": base_sha},
            **evidence_hashes,
        },
        "objects": {
            object_id: {
                "asset": {
                    "path": _repo_uri(project_root, item.mesh_path),
                    "sha256": item.mesh.sha256,
                    "bytes": item.mesh.size_bytes,
                    "vertices": item.mesh.vertices,
                    "faces": item.mesh.faces,
                    "finite": item.mesh.finite,
                    "nondegenerate": item.mesh.nondegenerate,
                    "winding_consistent": item.mesh.winding_consistent,
                    "watertight": item.mesh.watertight,
                    "topology": item.spec.topology,
                    "geometry_count": item.mesh.geometry_count,
                },
                "scene_fit_receipt": {
                    "path": _repo_uri(project_root, item.scene_fit_path),
                    "sha256": item.spec.scene_fit_sha256,
                },
                "asset_coordinates_baked": item.spec.asset_coordinates_baked,
                "runtime_pivot": item.pivot,
                "runtime_matrix_row_major": item.matrix_row_major,
                "collision_mode": "unified-glb",
                "proxy_fields_absent": True,
                "root_interaction": ["selection", "spin_360", "horizontal_yaw_drag"],
            }
            for object_id, item in verified.items()
        },
        "output_manifest": {
            "path": _repo_uri(project_root, output_manifest),
            "sha256": manifest_sha,
            "bytes": manifest_size,
        },
        "static_scene": manifest["candidateBuild"]["cleanScene"],
        "public_manifests": {
            "mutated": False,
            "promoted_manifest_sha256": public_before[0],
            "stable_manifest_sha256": stable_before[0],
        },
        "promotion_blockers": manifest["candidateBuild"]["promotionBlockers"],
    }
    atomic_write_json(report_path, report)
    require(sha256_file(public_manifest) == public_before, "promoted public manifest changed")
    require(sha256_file(stable_manifest) == stable_before, "stable public manifest changed")
    return manifest, report


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    manifests = project_root / "examples/bedroom4/manifests"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--base-manifest",
        type=Path,
        default=manifests / "bedroom4.unified-pillow.candidate.web.json",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=manifests / "bedroom4.unified-pbr-objects.candidate.web.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=manifests / "bedroom4.unified-pbr-objects.candidate.report.json",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest, report = materialize_candidate(
        project_root=args.project_root,
        base_manifest=args.base_manifest,
        output_manifest=args.output_manifest,
        report_path=args.report,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "version": manifest["version"],
                "manifest": str(args.output_manifest.resolve()),
                "report": str(args.report.resolve()),
                "promotion_allowed": report["promotion_allowed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
