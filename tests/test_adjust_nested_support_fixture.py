from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh

from scripts.adjust_nested_support_fixture import adjust_nested_support_fixture


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pbr(name: str) -> trimesh.visual.material.PBRMaterial:
    return trimesh.visual.material.PBRMaterial(
        name=name,
        baseColorFactor=[220, 218, 210, 255],
        metallicFactor=0.0,
        roughnessFactor=0.9,
    )


def test_adjust_nested_support_fixture_reduces_child_penetration(tmp_path: Path) -> None:
    parent = trimesh.Scene(base_frame="world")
    parent.graph.update(frame_from="world", frame_to="bed", matrix=np.eye(4))
    for name, mesh in {
        "mattress": trimesh.creation.box(extents=(2.0, 0.5, 3.0)),
        "foot_cover": trimesh.creation.box(extents=(2.0, 0.1, 0.5)),
        "headboard": trimesh.creation.box(extents=(2.0, 2.0, 0.1)),
    }.items():
        if name == "mattress":
            mesh.apply_translation([0.0, 0.75, 0.0])
        elif name == "foot_cover":
            mesh.apply_translation([0.0, 1.05, 1.0])
        else:
            mesh.apply_translation([0.0, 1.0, -1.55])
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=np.zeros((len(mesh.vertices), 2)), material=_pbr(name)
        )
        parent.add_geometry(
            mesh,
            node_name=f"bed__{name}",
            geom_name=f"bed.{name}",
            parent_node_name="bed",
        )
    parent_path = tmp_path / "bed.glb"
    parent.export(parent_path)
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(
        json.dumps(
            {
                "object_id": "bed",
                "component_layout": {"mattress_bottom_y": 0.5, "mattress_top_y": 1.0},
                "output": {"unified_pbr_glb": {"sha256": _sha(parent_path)}},
            }
        ),
        encoding="utf-8",
    )
    scene_fit_path = tmp_path / "bed-fit.json"
    scene_fit_path.write_text(
        json.dumps(
            {
                "object_id": "bed",
                "runtime_transform": {"matrix_row_major": np.eye(4).tolist()},
            }
        ),
        encoding="utf-8",
    )

    children = []
    for index, bottom in enumerate((0.82, 0.86)):
        child = trimesh.creation.box(extents=(0.7, 0.4, 0.3))
        child.visual = trimesh.visual.TextureVisuals(
            uv=np.zeros((len(child.vertices), 2)), material=_pbr(f"child{index}")
        )
        child_path = tmp_path / f"child-{index}.glb"
        child.export(child_path)
        report_path = tmp_path / f"child-{index}.json"
        report_path.write_text(
            json.dumps(
                {
                    "object_id": f"child-{index}",
                    "mesh": {"glb_sha256": _sha(child_path)},
                    "baked_relative_transform": {"runtime_pivot": [0.0, bottom + 0.2, 0.0]},
                }
            ),
            encoding="utf-8",
        )
        children.append(
            {
                "object_id": f"child-{index}",
                "mesh": str(child_path),
                "scene_fit_report": str(report_path),
            }
        )
    child_manifest = tmp_path / "children.json"
    child_manifest.write_text(
        json.dumps({"object_id": "bed", "children": children}), encoding="utf-8"
    )

    report = adjust_nested_support_fixture(
        object_id="bed",
        parent_mesh=parent_path,
        parent_fixture_receipt=fixture_path,
        parent_scene_fit_report=scene_fit_path,
        child_manifest=child_manifest,
        output_dir=tmp_path / "out",
        maximum_penetration_world=0.1,
        maximum_penetration_fraction=0.1,
        maximum_support_gap_world=0.1,
    )

    assert report["all_acceptance_gates_passed"] is True
    assert report["adjustment"]["mattress_top_after_parent"] < 1.0
    assert all(child["penetration_gate"] for child in report["children"])
    assert all(child["support_gap_gate"] for child in report["children"])
    assert Path(report["output"]["unified_pbr_glb"]["path"]).is_file()
