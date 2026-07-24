from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image
from plyfile import PlyData, PlyElement

from scripts.materialize_direct_trellis_scene_candidate import audit_unified_glb
from scripts.materialize_front_half_scene_candidate import (
    CARVED_STATIC_ROLE,
    CONFIG_KIND,
    materialize_front_half_candidate,
)
from video2world.hashing import sha256_file

GAUSSIAN_DTYPE = [
    ("x", "f4"),
    ("y", "f4"),
    ("z", "f4"),
    ("f_dc_0", "f4"),
    ("f_dc_1", "f4"),
    ("f_dc_2", "f4"),
    ("opacity", "f4"),
    ("scale_0", "f4"),
    ("scale_1", "f4"),
    ("scale_2", "f4"),
    ("rot_0", "f4"),
    ("rot_1", "f4"),
    ("rot_2", "f4"),
    ("rot_3", "f4"),
]


def _sha(path: Path) -> str:
    return sha256_file(path)[0]


def _write_static(path: Path, *, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros(count, dtype=GAUSSIAN_DTYPE)
    data["x"] = np.arange(count, dtype=np.float32)
    data["opacity"] = 2.0
    data["scale_0"] = -4.0
    data["scale_1"] = -4.0
    data["scale_2"] = -4.0
    data["rot_0"] = 1.0
    PlyData([PlyElement.describe(data, "vertex")], text=False, byte_order="<").write(path)


def _write_collider(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.array(
        [
            (-1.0, -1.0, 0.0),
            (1.0, -1.0, 0.0),
            (0.0, 1.0, 0.0),
        ],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")],
    )
    faces = np.empty(1, dtype=[("vertex_index", "O")])
    faces["vertex_index"][0] = np.array([0, 1, 2], dtype=np.int32)
    PlyData(
        [
            PlyElement.describe(vertices, "vertex"),
            PlyElement.describe(
                faces,
                "face",
                len_types={"vertex_index": "u1"},
                val_types={"vertex_index": "i4"},
            ),
        ],
        text=False,
        byte_order="<",
    ).write(path)


def _write_glb(path: Path) -> None:
    mesh = trimesh.creation.box(extents=(2.0, 1.0, 3.0))
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2), dtype=np.float64),
        material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=[224, 221, 214, 255],
            baseColorTexture=Image.new("RGBA", (2, 2), (224, 221, 214, 255)),
            metallicFactor=0.0,
            roughnessFactor=0.9,
        ),
    )
    scene = trimesh.Scene(base_frame="world")
    scene.add_geometry(mesh, node_name="bed", geom_name="bed.geometry")
    scene.export(path, include_normals=True)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _fixture(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "repo"
    source = root / "inputs/source.ply"
    carved = root / "inputs/carved.ply"
    collider = root / "inputs/collider.ply"
    glb = root / "objects/bed.scene-space.glb"
    _write_static(source, count=4)
    _write_static(carved, count=3)
    _write_collider(collider)
    glb.parent.mkdir(parents=True, exist_ok=True)
    _write_glb(glb)
    glb_audit = audit_unified_glb(glb, maximum_faces=100_000)

    carve_receipt = root / "inputs/carve-receipt.json"
    _write_json(
        carve_receipt,
        {
            "schema_version": 1,
            "kind": "video2world.multiview_depth_static_gaussian_carve",
            "status": "candidate_materialized",
            "promotion_allowed": False,
            "input_static_scene": {
                "path": str(source),
                "sha256": _sha(source),
                "bytes": source.stat().st_size,
                "vertex_count": 4,
                "role": "original_uncarved_pgsr_full_scene",
            },
            "output_static_scene": {
                "path": str(carved),
                "sha256": _sha(carved),
                "bytes": carved.stat().st_size,
                "vertex_count": 3,
                "role": CARVED_STATIC_ROLE,
            },
            "analysis": {
                "selected_remove_count": 1,
                "strict_all_view_remove_count": 1,
            },
        },
    )
    scene_fit = root / "objects/bed.scene-fit-receipt.json"
    _write_json(
        scene_fit,
        {
            "schema_version": 1,
            "kind": "video2world.completed_object_scene_fit",
            "status": "technical_gates_passed",
            "all_acceptance_gates_passed": True,
            "object_id": "bed",
            "acceptance_gates": {"scene_fit": True},
            "exports": {
                "scene_space": {
                    "coordinate_space": "world_baked",
                    "glb_role": "pbr_visual_logic_authority",
                    "runtime_transform": np.eye(4).tolist(),
                    "glb": {
                        "path": str(glb),
                        "sha256": glb_audit.sha256,
                        "bytes": glb_audit.size_bytes,
                        "vertices": glb_audit.vertices,
                        "faces": glb_audit.faces,
                        "bounds": glb_audit.bounds,
                    },
                }
            },
            "final_export_qa": {
                "bounds_tolerance": 1e-5,
                "canonical_to_world_round_trip": True,
            },
        },
    )
    config = root / "inputs/composite.json"
    _write_json(
        config,
        {
            "schema_version": 1,
            "kind": CONFIG_KIND,
            "candidate_version": "test-front-half-candidate",
            "static_scene": {
                "role": CARVED_STATIC_ROLE,
                "path": str(carved),
                "sha256": _sha(carved),
                "carve_receipt": {"path": str(carve_receipt), "sha256": _sha(carve_receipt)},
            },
            "static_collider": {
                "role": "original_uncarved_scene_mesh",
                "path": str(collider),
                "sha256": _sha(collider),
            },
            "objects": [
                {
                    "object_id": "bed",
                    "category": "bed",
                    "label": "Bed",
                    "scene_fit_receipt": {"path": str(scene_fit), "sha256": _sha(scene_fit)},
                }
            ],
        },
    )
    return {
        "root": root,
        "source": source,
        "carved": carved,
        "config": config,
        "output": root / "candidate/manifest.json",
        "report": root / "candidate/report.json",
    }


def test_materializes_carved_static_scene_with_world_baked_bed(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    source_sha = _sha(paths["source"])
    config_before = paths["config"].read_bytes()

    manifest, report = materialize_front_half_candidate(
        project_root=paths["root"],
        config_path=paths["config"],
        output_manifest=paths["output"],
        report_path=paths["report"],
    )

    assert _sha(paths["source"]) == source_sha
    assert paths["config"].read_bytes() == config_before
    assert manifest["promotion_allowed"] is False
    assert manifest["static_scene"]["visual"]["vertex_count"] == 3
    assert manifest["static_scene"]["carve_receipt"]["removed_static_gaussian_count"] == 1
    assert manifest["static_collider"]["replacement_status"] == (
        "not_carved_for_this_visual_candidate"
    )
    assert manifest["objects"][0]["placement"]["runtime_transform"] == "identity"
    assert manifest["objects"][0]["visual_logic_collision"]["coordinate_space"] == "world_baked"
    assert report["promotion_allowed"] is False
    assert paths["output"].is_file() and paths["report"].is_file()


def test_rejects_uncarved_static_role(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    config["static_scene"]["role"] = "original_uncarved_pgsr_full_scene"
    _write_json(paths["config"], config)

    with pytest.raises(RuntimeError, match="static_scene role mismatch"):
        materialize_front_half_candidate(
            project_root=paths["root"],
            config_path=paths["config"],
            output_manifest=paths["output"],
            report_path=paths["report"],
        )
