from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from scripts.prepare_affine_scene_fit import prepare_affine_scene_fit


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    scene = trimesh.Scene(base_frame="world")
    scene.graph.update(frame_from="world", frame_to="bed", matrix=np.eye(4))
    material = trimesh.visual.material.PBRMaterial(
        name="fabric",
        baseColorFactor=[220, 218, 210, 255],
        metallicFactor=0.0,
        roughnessFactor=0.9,
    )
    for index, x in enumerate((-0.6, 0.6)):
        mesh = trimesh.creation.box(extents=(0.8, 0.4, 1.5))
        mesh.apply_translation([x, 0.2, 0.0])
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=np.zeros((len(mesh.vertices), 2)),
            material=material,
        )
        scene.add_geometry(
            mesh,
            node_name=f"bed__part_{index}",
            geom_name=f"bed.part_{index}",
            parent_node_name="bed",
        )
    mesh_path = tmp_path / "bed.glb"
    scene.export(mesh_path)
    angle = np.deg2rad(25.0)
    rotation = np.asarray(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    matrix = np.eye(4)
    matrix[:3, :3] = rotation * 2.5
    matrix[:3, 3] = [1.0, 2.0, 3.0]
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "object_id": "bed",
                "sources": {"anchor": {"canonical_to_anchor_row_major": matrix.tolist()}},
                "output": {"unified_pbr_glb": {"sha256": _sha256(mesh_path)}},
            }
        ),
        encoding="utf-8",
    )
    return mesh_path, receipt_path


def test_prepare_affine_scene_fit_validates_unified_asset(tmp_path: Path) -> None:
    mesh_path, receipt_path = _fixture(tmp_path)
    report = prepare_affine_scene_fit(
        object_id="bed",
        mesh_source=mesh_path,
        transform_receipt=receipt_path,
        output_dir=tmp_path / "out",
    )

    assert report["all_acceptance_gates_passed"] is True
    assert report["runtime_transform"]["uniform_scale"] == pytest.approx(2.5)
    assert report["logical_entity"]["root_nodes"] == ["bed"]
    assert len(report["logical_entity"]["component_nodes"]) == 2
    assert report["mesh"]["faces"] > 0


def test_prepare_affine_scene_fit_rejects_hash_mismatch(tmp_path: Path) -> None:
    mesh_path, receipt_path = _fixture(tmp_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["output"]["unified_pbr_glb"]["sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="mesh hash"):
        prepare_affine_scene_fit(
            object_id="bed",
            mesh_source=mesh_path,
            transform_receipt=receipt_path,
            output_dir=tmp_path / "out",
        )


def test_prepare_affine_scene_fit_fails_shear_gate(tmp_path: Path) -> None:
    mesh_path, receipt_path = _fixture(tmp_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["sources"]["anchor"]["canonical_to_anchor_row_major"][0][1] += 0.4
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    report = prepare_affine_scene_fit(
        object_id="bed",
        mesh_source=mesh_path,
        transform_receipt=receipt_path,
        output_dir=tmp_path / "out",
    )

    assert report["all_acceptance_gates_passed"] is False
    assert report["acceptance_gates"]["orthonormal_rotation"] is False


def test_prepare_affine_scene_fit_accepts_bounded_axis_refinement(tmp_path: Path) -> None:
    mesh_path, receipt_path = _fixture(tmp_path)
    report = prepare_affine_scene_fit(
        object_id="bed",
        mesh_source=mesh_path,
        transform_receipt=receipt_path,
        output_dir=tmp_path / "out",
        scale_factors=(0.72, 0.45, 0.78),
        canonical_offset=(0.0, 0.0, 0.39),
        allow_nonuniform_scale=True,
        maximum_scale_anisotropy=2.0,
    )

    assert report["all_acceptance_gates_passed"] is True
    assert report["runtime_transform"]["uniform_scale_verified"] is False
    assert report["runtime_transform"]["scale_anisotropy_ratio"] < 2.0
    assert report["runtime_transform"]["refinement"]["canonical_offset"] == [0.0, 0.0, 0.39]


def test_prepare_affine_scene_fit_rejects_excessive_axis_refinement(tmp_path: Path) -> None:
    mesh_path, receipt_path = _fixture(tmp_path)
    report = prepare_affine_scene_fit(
        object_id="bed",
        mesh_source=mesh_path,
        transform_receipt=receipt_path,
        output_dir=tmp_path / "out",
        scale_factors=(1.0, 0.2, 1.0),
        allow_nonuniform_scale=True,
        maximum_scale_anisotropy=2.0,
    )

    assert report["all_acceptance_gates_passed"] is False
    assert report["acceptance_gates"]["scale_anisotropy_within_limit"] is False
