from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image
from plyfile import PlyData, PlyElement

from scripts.materialize_direct_trellis_scene_candidate import (
    CONFIG_KIND,
    LOD_QA_KIND,
    LOD_REQUIRED_QA_GATES,
    RECEIPT_KIND,
    audit_unified_glb,
    materialize_candidate,
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


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_gaussian_ply(path: Path) -> None:
    data = np.zeros(64, dtype=GAUSSIAN_DTYPE)
    data["x"] = np.linspace(-2.0, 2.0, len(data))
    data["y"] = np.linspace(1.0, 3.0, len(data))
    data["z"] = np.linspace(6.0, 12.0, len(data))
    data["opacity"] = 3.0
    data["scale_0"] = -3.0
    data["scale_1"] = -3.0
    data["scale_2"] = -3.0
    data["rot_0"] = 1.0
    PlyData(
        [PlyElement.describe(data, "vertex")],
        text=False,
        byte_order="<",
    ).write(path)


def _write_triangle_mesh_ply(path: Path) -> None:
    mesh = trimesh.creation.box(extents=(4.0, 2.0, 6.0))
    vertices = np.empty(
        len(mesh.vertices),
        dtype=[
            ("x", "f8"),
            ("y", "f8"),
            ("z", "f8"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    for axis, name in enumerate(("x", "y", "z")):
        vertices[name] = mesh.vertices[:, axis]
    for name in ("red", "green", "blue"):
        vertices[name] = 180
    faces = np.empty(len(mesh.faces), dtype=[("vertex_indices", "O")])
    faces["vertex_indices"] = [np.asarray(face, dtype=np.uint32) for face in mesh.faces]
    PlyData(
        [
            PlyElement.describe(vertices, "vertex"),
            PlyElement.describe(
                faces,
                "face",
                len_types={"vertex_indices": "u1"},
                val_types={"vertex_indices": "u4"},
            ),
        ],
        text=False,
        byte_order="<",
    ).write(path)


def _write_pbr_glb(
    path: Path,
    *,
    extents: tuple[float, float, float],
    degenerate_triangles: int = 0,
) -> None:
    mesh = trimesh.creation.box(extents=extents)
    if degenerate_triangles:
        mesh.faces = np.vstack(
            [mesh.faces, np.repeat([[0, 0, 0]], degenerate_triangles, axis=0)]
        )
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2), dtype=np.float64),
        material=trimesh.visual.material.PBRMaterial(
            name="trellis-pbr",
            baseColorFactor=[230, 228, 220, 255],
            baseColorTexture=Image.new("RGBA", (2, 2), (230, 228, 220, 255)),
            metallicFactor=0.0,
            roughnessFactor=0.88,
        ),
    )
    scene = trimesh.Scene(base_frame="world")
    scene.add_geometry(mesh, node_name=path.stem, geom_name=f"{path.stem}.geometry")
    scene.export(path, include_normals=True)


def _base_object(
    object_id: str,
    *,
    parent_id: str | None,
    child_ids: list[str],
) -> dict:
    is_child = parent_id is not None
    return {
        "id": object_id,
        "label": object_id,
        "name": object_id,
        "category": "pillow" if is_child else "bed",
        "aliases": [object_id],
        "bbox": {
            "coordinateFrame": "visual_native",
            "min": [-1, -1, -1],
            "max": [1, 1, 1],
            "center": [0, 0, 0],
            "extent": [2, 2, 2],
        },
        "semanticGranularity": (
            "independent_child_asset" if is_child else "independent_root_asset"
        ),
        "parentObjectId": parent_id,
        "movesWithParent": is_child,
        "childObjectIds": child_ids,
        "independentlyMovable": True,
        "placement": {"pivot": [0, 0, 0], "scale": [1, 1, 1]},
        "visual": {"url": "old-object.ply"},
        "colliderProxy": {"type": "box", "dimensions": [2, 2, 2]},
        "carve": {"method": "old-carve"},
        "collision": {
            "mode": "kinematic",
            "walkable": False,
            "renderAsset": {"url": "old-render.glb"},
        },
        "interaction": {
            "kind": "spin",
            "degrees": 360,
            "durationMs": 1250,
            "drag": "horizontal_yaw",
        },
    }


def _scene_fit_receipt(
    path: Path,
    *,
    object_id: str,
    glb_path: Path,
    pivot: list[float],
    enforce_runtime_geometry: bool = True,
) -> None:
    audit = audit_unified_glb(
        glb_path,
        maximum_faces=100_000,
        enforce_runtime_geometry=enforce_runtime_geometry,
    )
    predicted_world_bounds = (
        np.asarray(audit.bounds, dtype=np.float64) + np.asarray(pivot, dtype=np.float64)
    ).tolist()
    payload = {
        "schema_version": 1,
        "kind": RECEIPT_KIND,
        "status": "technical_gates_passed",
        "all_acceptance_gates_passed": True,
        "promotion_allowed": False,
        "object_id": object_id,
        "representation_mode": "unified_pbr_mesh_visual_logic_collision",
        "acceptance_gates": {"geometry": True, "scene_local_pivot_round_trip": True},
        "final_export_qa": {
            "bounds_tolerance": 1e-5,
            "scene_local_pivot_round_trip": True,
            "predicted_world_bounds_from_scene_local_bytes_and_pivot": (predicted_world_bounds),
        },
        "scene_local_runtime": {
            "asset_coordinates_baked": "rotation_and_scale_only",
            "placement": {
                "pivot": pivot,
                "scale": [1.0, 1.0, 1.0],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        },
        "exports": {
            "scene_local": {
                "coordinate_space": "scene_local_about_target_pivot",
                "placement": {
                    "pivot": pivot,
                    "runtime_scale": [1.0, 1.0, 1.0],
                    "runtime_rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "glb_role": "pbr_visual_logic_authority",
                "glb": {
                    "path": str(glb_path),
                    "sha256": audit.sha256,
                    "bytes": audit.size_bytes,
                    "vertices": audit.vertices,
                    "faces": audit.faces,
                    "bounds": audit.bounds,
                    "finite_vertices": audit.finite,
                    "watertight": audit.watertight,
                    "winding_consistent": audit.winding_consistent,
                    "material_types": audit.material_types,
                },
            }
        },
    }
    _write_json(path, payload)


def _qa_bounds(audit) -> dict:
    values = np.asarray(audit.bounds, dtype=np.float64)
    minimum = values[0]
    maximum = values[1]
    return {
        "coordinateSystem": "blender_world_z_up",
        "min": [minimum[0], -maximum[2], minimum[1]],
        "max": [maximum[0], -minimum[2], maximum[1]],
        "center": [
            (minimum[0] + maximum[0]) / 2.0,
            -(minimum[2] + maximum[2]) / 2.0,
            (minimum[1] + maximum[1]) / 2.0,
        ],
        "size": [
            maximum[0] - minimum[0],
            maximum[2] - minimum[2],
            maximum[1] - minimum[1],
        ],
    }


def _qa_stats(audit) -> dict:
    return {
        "path": str(audit.path),
        "bytes": audit.size_bytes,
        "sha256": audit.sha256,
        "meshObjects": 1,
        "vertices": audit.vertices,
        "faces": audit.faces,
        "uvLayers": audit.uv_layers,
        "normalLoops": audit.normal_loops,
        "finiteVertices": audit.finite,
        "normalsFinite": audit.normals_finite,
        "degenerateTriangles": audit.degenerate_triangles,
        "winding": {"consistent": audit.winding_consistent},
        "bounds": _qa_bounds(audit),
        "glbContract": {
            "allPrimitivesHaveMaterial": audit.glb_has_materials,
            "allPrimitivesHaveNormals": audit.glb_has_normals,
            "allPrimitivesHaveTexcoord0": audit.glb_has_uvs,
        },
        "materials": {
            "pbrMaterialCount": audit.pbr_materials,
            "textureImageCount": audit.texture_images,
        },
    }


def _qa_binding(audit) -> dict:
    return {
        "path": str(audit.path),
        "sha256": audit.sha256,
        "bytes": audit.size_bytes,
        "faces": audit.faces,
        "bounds": _qa_bounds(audit),
        "pbrMaterials": audit.pbr_materials,
        "textureImages": audit.texture_images,
        "uvLayers": audit.uv_layers,
        "normalLoops": audit.normal_loops,
        "glbHasNormals": audit.glb_has_normals,
        "degenerateTriangles": audit.degenerate_triangles,
        "windingConsistent": audit.winding_consistent,
    }


def _write_lod_qa(path: Path, *, source_path: Path, output_path: Path) -> None:
    source = audit_unified_glb(source_path, maximum_faces=100_000)
    output = audit_unified_glb(output_path, maximum_faces=100_000)
    bounds_delta = float(
        np.max(
            np.abs(
                np.asarray(source.bounds, dtype=np.float64)
                - np.asarray(output.bounds, dtype=np.float64)
            )
        )
    )
    payload = {
        "schemaVersion": 1,
        "kind": LOD_QA_KIND,
        "status": "passed",
        "source": _qa_stats(source),
        "target": {"faces": 90_000, "met": True},
        "output": _qa_stats(output),
        "qa": {
            **{name: True for name in LOD_REQUIRED_QA_GATES},
            "boundsTolerance": 0.01,
            "maxAbsBoundsDelta": bounds_delta,
        },
        "assetBinding": {
            "input": _qa_binding(source),
            "output": _qa_binding(output),
        },
    }
    _write_json(path, payload)


def _fixture(tmp_path: Path) -> dict[str, Path]:
    project_root = tmp_path / "repo"
    public_manifest = project_root / "web/public/worlds/bedroom4/manifest.json"
    _write_json(public_manifest, {"protected": True})

    base_path = project_root / "inputs/base.json"
    base = {
        "contract": "video2world-web-manifest-1.0.0",
        "schemaVersion": 1,
        "version": "base",
        "assets": {
            "visual": {"url": "old-carved.ply"},
            "collider": {
                "id": "bedroom4_strict_clean_tsdf_raw",
                "url": "strict-clean.ply",
                "parts": [{"url": "./chunks/strict_clean_collider.chunk000"}],
            },
            "colliderStaticCarved": {
                "id": "bedroom4_strict_clean_tsdf_static_legacy_carved",
                "parts": [{"url": "./chunks/strict_clean_legacy.chunk000"}],
            },
        },
        "collisionWorld": {
            "sceneAssetKey": "colliderStaticCarved",
            "replacementMode": "strict_clean_then_carve",
            "sceneAssetTransport": {"parts": [{"url": "./chunks/strict_clean_legacy.chunk000"}]},
        },
        "interactiveObjects": [
            _base_object("bed", parent_id=None, child_ids=["pillow"]),
            _base_object("pillow", parent_id="bed", child_ids=[]),
        ],
        "sceneKnowledge": {
            "objects": [
                {"id": "bed", "bbox": {}},
                {"id": "pillow", "bbox": {}},
            ]
        },
        "candidateBuild": {"status": "old-clean-plate-claim"},
    }
    _write_json(base_path, base)

    static_path = project_root / "inputs/original-scene.ply"
    static_path.parent.mkdir(parents=True, exist_ok=True)
    _write_gaussian_ply(static_path)
    collider_path = project_root / "inputs/tsdf_fusion_post.ply"
    _write_triangle_mesh_ply(collider_path)

    objects = []
    for object_id, extents, pivot in (
        ("bed", (2.4, 1.1, 3.2), [4.0, 1.5, 9.0]),
        ("pillow", (0.9, 0.35, 0.6), [3.6, 2.1, 9.2]),
    ):
        glb_path = project_root / "outputs" / object_id / f"{object_id}.scene-local.glb"
        glb_path.parent.mkdir(parents=True, exist_ok=True)
        _write_pbr_glb(glb_path, extents=extents)
        receipt_path = glb_path.parent / "scene_fit_receipt.json"
        _scene_fit_receipt(
            receipt_path,
            object_id=object_id,
            glb_path=glb_path,
            pivot=pivot,
        )
        objects.append(
            {
                "object_id": object_id,
                "scene_fit_receipt": {
                    "path": str(receipt_path),
                    "sha256": sha256_file(receipt_path)[0],
                },
            }
        )

    config_path = project_root / "inputs/direct.json"
    config = {
        "schema_version": 1,
        "kind": CONFIG_KIND,
        "candidate_version": "direct-trellis-local-qa",
        "unadopted_object_policy": "original_static_scene_only_with_logical_ancestors",
        "base_manifest": {
            "path": str(base_path),
            "sha256": sha256_file(base_path)[0],
        },
        "static_scene": {
            "role": "original_uncarved_pgsr_full_scene",
            "path": str(static_path),
            "sha256": sha256_file(static_path)[0],
        },
        "static_collider": {
            "role": "original_uncarved_scene_mesh",
            "path": str(collider_path),
            "sha256": sha256_file(collider_path)[0],
        },
        "maximum_collision_faces": 100_000,
        "objects": objects,
    }
    _write_json(config_path, config)
    return {
        "project_root": project_root,
        "public_manifest": public_manifest,
        "base": base_path,
        "static": static_path,
        "collider": collider_path,
        "config": config_path,
        "output": project_root / "qa/direct.manifest.json",
        "report": project_root / "qa/direct.report.json",
    }


def _add_override(paths: dict[str, Path], *, object_id: str = "bed") -> dict[str, Path]:
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    object_ref = next(item for item in config["objects"] if item["object_id"] == object_id)
    receipt_path = Path(object_ref["scene_fit_receipt"]["path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    source_path = Path(receipt["exports"]["scene_local"]["glb"]["path"])
    output_path = source_path.with_name(f"{object_id}.scene-local.browser-lod.glb")
    output_path.write_bytes(source_path.read_bytes())
    qa_path = output_path.with_suffix(".qa.json")
    _write_lod_qa(qa_path, source_path=source_path, output_path=output_path)
    config["object_overrides"] = {
        object_id: {
            "source_asset": {
                "path": str(source_path),
                "sha256": sha256_file(source_path)[0],
            },
            "browser_lod": {
                "path": str(output_path),
                "sha256": sha256_file(output_path)[0],
                "coordinate_space": "scene_local_about_target_pivot",
                "role": "pbr_visual_logic_authority",
            },
            "qa": {
                "path": str(qa_path),
                "sha256": sha256_file(qa_path)[0],
                "required_status": "passed",
            },
            "placement_policy": "inherit_scene_fit_receipt",
        }
    }
    _write_json(paths["config"], config)
    return {
        "source": source_path,
        "output": output_path,
        "qa": qa_path,
    }


def _add_blender_normalized_degenerate_override(paths: dict[str, Path]) -> dict[str, Path]:
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    object_ref = next(item for item in config["objects"] if item["object_id"] == "bed")
    receipt_path = Path(object_ref["scene_fit_receipt"]["path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    source_path = Path(receipt["exports"]["scene_local"]["glb"]["path"])
    pivot = receipt["scene_local_runtime"]["placement"]["pivot"]

    _write_pbr_glb(
        source_path,
        extents=(2.4, 1.1, 3.2),
        degenerate_triangles=1,
    )
    _scene_fit_receipt(
        receipt_path,
        object_id="bed",
        glb_path=source_path,
        pivot=pivot,
        enforce_runtime_geometry=False,
    )
    object_ref["scene_fit_receipt"]["sha256"] = sha256_file(receipt_path)[0]

    output_path = source_path.with_name("bed.scene-local.browser-lod.glb")
    _write_pbr_glb(output_path, extents=(2.4, 1.1, 3.2))
    source = audit_unified_glb(
        source_path,
        maximum_faces=100_000,
        enforce_runtime_geometry=False,
    )
    output = audit_unified_glb(output_path, maximum_faces=100_000)
    assert source.degenerate_triangles == 1
    assert output.degenerate_triangles == 0

    normalized_source = _qa_stats(source)
    normalized_source["faces"] -= source.degenerate_triangles
    normalized_source["normalLoops"] -= source.degenerate_triangles * 3
    normalized_source["degenerateTriangles"] = 0
    normalized_binding = _qa_binding(source)
    normalized_binding["faces"] -= source.degenerate_triangles
    normalized_binding["normalLoops"] -= source.degenerate_triangles * 3
    normalized_binding["degenerateTriangles"] = 0
    bounds_delta = float(
        np.max(
            np.abs(
                np.asarray(source.bounds, dtype=np.float64)
                - np.asarray(output.bounds, dtype=np.float64)
            )
        )
    )
    qa_path = output_path.with_suffix(".qa.json")
    qa = {
        "schemaVersion": 1,
        "kind": LOD_QA_KIND,
        "status": "passed",
        "source": normalized_source,
        "target": {"faces": 90_000, "met": True},
        "output": _qa_stats(output),
        "qa": {
            **{name: True for name in LOD_REQUIRED_QA_GATES},
            "boundsTolerance": 0.01,
            "maxAbsBoundsDelta": bounds_delta,
        },
        "processing": {
            "tool": "blender_headless_pbr_decimate",
            "sourceFacesAfterTriangulate": source.faces - source.degenerate_triangles,
        },
        "assetBinding": {
            "input": normalized_binding,
            "output": _qa_binding(output),
        },
    }
    _write_json(qa_path, qa)
    config["object_overrides"] = {
        "bed": {
            "source_asset": {
                "path": str(source_path),
                "sha256": source.sha256,
            },
            "browser_lod": {
                "path": str(output_path),
                "sha256": output.sha256,
                "coordinate_space": "scene_local_about_target_pivot",
                "role": "pbr_visual_logic_authority",
            },
            "qa": {
                "path": str(qa_path),
                "sha256": sha256_file(qa_path)[0],
                "required_status": "passed",
            },
            "placement_policy": "inherit_scene_fit_receipt",
        }
    }
    _write_json(paths["config"], config)
    return {
        "source": source_path,
        "output": output_path,
        "qa": qa_path,
    }


def _refresh_qa_hash(paths: dict[str, Path], qa_path: Path, *, object_id: str = "bed") -> None:
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    config["object_overrides"][object_id]["qa"]["sha256"] = sha256_file(qa_path)[0]
    _write_json(paths["config"], config)


def test_materializes_uncarved_local_scene_and_unified_glbs(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    protected_before = paths["public_manifest"].read_bytes()
    base_before = paths["base"].read_bytes()

    manifest, report = materialize_candidate(
        project_root=paths["project_root"],
        config_path=paths["config"],
        output_manifest=paths["output"],
        report_path=paths["report"],
    )

    assert paths["public_manifest"].read_bytes() == protected_before
    assert paths["base"].read_bytes() == base_before
    assert report["promotion_allowed"] is False
    assert report["clean_plate"] == {"generated": False, "consumed": False}
    assert report["carve"] == {"performed": False}
    assert report["public_manifests"]["mutated"] is False

    visual = manifest["assets"]["visual"]
    assert visual["url"].startswith("/@fs/")
    assert visual["sha256"] == sha256_file(paths["static"])[0]
    assert visual["byteForBytePreserved"] is True
    assert visual["carved"] is False
    assert visual["cleanPlateApplied"] is False
    collider = manifest["assets"]["collider"]
    assert collider["url"].startswith("/@fs/")
    assert collider["sha256"] == sha256_file(paths["collider"])[0]
    assert collider["vertexCount"] == 8
    assert collider["faceCount"] == 12
    assert collider["finite"] is True
    assert collider["faceIndicesValid"] is True
    assert collider["degenerateTriangles"] == 0
    assert collider["byteForBytePreserved"] is True
    assert collider["carved"] is False
    assert collider["cleanPlateApplied"] is False
    assert "parts" not in collider
    assert "colliderStaticCarved" not in manifest["assets"]
    assert manifest["collisionWorld"]["sceneAssetKey"] == "collider"
    assert manifest["collisionWorld"]["replacementMode"] == ("original_uncarved_scene_mesh")
    assert "parts" not in manifest["collisionWorld"]["sceneAssetTransport"]
    collider_contract = json.dumps(
        {
            "assets": manifest["assets"],
            "collisionWorld": manifest["collisionWorld"],
        },
        sort_keys=True,
    ).lower()
    assert "strict_clean" not in collider_contract
    assert "/chunks/" not in collider_contract
    assert manifest["candidateBuild"]["promotionAllowed"] is False
    static_contract = manifest["candidateBuild"]["staticScene"]
    assert static_contract["mode"] == "original_uncarved_pgsr_full_scene"
    assert static_contract["sha256"] == sha256_file(paths["static"])[0]
    assert static_contract["collider"] == {
        "role": "original_uncarved_scene_mesh",
        "sha256": sha256_file(paths["collider"])[0],
        "bytes": paths["collider"].stat().st_size,
        "vertexCount": 8,
        "faceCount": 12,
    }
    assert static_contract["byteForBytePreserved"] is True
    assert static_contract["carvePerformed"] is False
    assert static_contract["cleanPlateGenerated"] is False
    assert report["inputs"]["static_collider"]["sha256"] == sha256_file(paths["collider"])[0]
    assert report["inputs"]["static_collider"]["byte_for_byte_preserved"] is True

    objects = {item["id"]: item for item in manifest["interactiveObjects"]}
    assert objects["bed"]["parentObjectId"] is None
    assert objects["bed"]["childObjectIds"] == ["pillow"]
    assert objects["pillow"]["parentObjectId"] == "bed"
    assert objects["pillow"]["movesWithParent"] is True
    for item in objects.values():
        assert "visual" not in item
        assert "renderAsset" not in item
        assert "colliderProxy" not in item
        assert "carve" not in item
        assert item["collision"]["mode"] == "unified-glb"
        assert "renderAsset" not in item["collision"]
        assert item["collision"]["asset"]["url"].startswith("/@fs/")
        assert item["collision"]["asset"]["faces"] == 12
        assert item["placement"]["scale"] == [1, 1, 1]
        assert item["placement"]["rotationEulerDeg"] == [0, 0, 0]
    assert objects["bed"]["placement"]["pivot"] == [4.0, 1.5, 9.0]
    assert objects["pillow"]["placement"]["pivot"] == [3.6, 2.1, 9.2]
    assert objects["bed"]["bbox"]["extent"] == pytest.approx([2.4, 1.1, 3.2])


def test_three_adopted_pillows_keep_only_a_stripped_logical_bed_parent(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    base = json.loads(paths["base"].read_text(encoding="utf-8"))
    bed = next(item for item in base["interactiveObjects"] if item["id"] == "bed")
    bed["childObjectIds"] = ["pillow", "pillow_left", "pillow_right"]
    for object_id in ("pillow_left", "pillow_right"):
        base["interactiveObjects"].append(_base_object(object_id, parent_id="bed", child_ids=[]))
    for object_id in ("nightstand", "plant"):
        dropped = _base_object(object_id, parent_id=None, child_ids=[])
        dropped["visual"] = {"url": f"./chunks/{object_id}.old-gaussian.chunk000"}
        dropped["collision"]["renderAsset"] = {"url": f"./objects/{object_id}.old.glb"}
        dropped["collision"]["gate"] = {
            "status": "held",
            "candidateBrowserQa": "pending_strict_clean_scene_world",
        }
        base["interactiveObjects"].append(dropped)
        base["sceneKnowledge"]["objects"].append({"id": object_id, "bbox": {}})
    base["interactiveObjectBuild"] = {
        "legacyVisual": "./chunks/interactive_nightstand.old-gaussian.chunk000"
    }
    base["productionBuild"] = {
        "legacyObjectGlb": "./objects/nightstand.old.glb",
        "candidateBrowserQa": "pending_strict_clean_scene_world",
    }
    _write_json(paths["base"], base)

    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    pillow_ref = next(item for item in config["objects"] if item["object_id"] == "pillow")
    config["objects"] = [pillow_ref]
    for index, object_id in enumerate(("pillow_left", "pillow_right"), start=1):
        glb_path = paths["project_root"] / "outputs" / object_id / f"{object_id}.scene-local.glb"
        glb_path.parent.mkdir(parents=True, exist_ok=True)
        _write_pbr_glb(glb_path, extents=(0.8 + index * 0.05, 0.3, 0.55))
        receipt_path = glb_path.parent / "scene_fit_receipt.json"
        _scene_fit_receipt(
            receipt_path,
            object_id=object_id,
            glb_path=glb_path,
            pivot=[3.0 + index, 2.0, 9.0],
        )
        config["objects"].append(
            {
                "object_id": object_id,
                "scene_fit_receipt": {
                    "path": str(receipt_path),
                    "sha256": sha256_file(receipt_path)[0],
                },
            }
        )
    config["base_manifest"]["sha256"] = sha256_file(paths["base"])[0]
    _write_json(paths["config"], config)

    manifest, report = materialize_candidate(
        project_root=paths["project_root"],
        config_path=paths["config"],
        output_manifest=paths["output"],
        report_path=paths["report"],
    )

    runtime = {item["id"]: item for item in manifest["interactiveObjects"]}
    assert set(runtime) == {"bed", "pillow", "pillow_left", "pillow_right"}
    logical_bed = runtime["bed"]
    assert logical_bed == {
        "id": "bed",
        "label": "bed",
        "name": "bed",
        "category": "bed",
        "aliases": ["bed"],
        "semanticGranularity": "independent_root_asset",
        "parentObjectId": None,
        "movesWithParent": False,
        "childObjectIds": ["pillow", "pillow_left", "pillow_right"],
        "independentlyMovable": False,
        "logicalHierarchyOnly": True,
        "logicalRole": "unrendered_unselectable_hierarchy_ancestor",
    }
    for object_id in ("pillow", "pillow_left", "pillow_right"):
        assert runtime[object_id]["parentObjectId"] == "bed"
        assert runtime[object_id]["movesWithParent"] is True
        assert runtime[object_id]["childObjectIds"] == []
        assert runtime[object_id]["collision"]["mode"] == "unified-glb"
    serialized = json.dumps(manifest, sort_keys=True).lower()
    assert "/chunks/" not in serialized
    assert "pending_strict_clean_scene_world" not in serialized
    assert "old.glb" not in serialized
    assert "old-gaussian" not in serialized
    assert "interactiveObjectBuild" not in manifest
    assert "productionBuild" not in manifest
    assert {item["id"] for item in manifest["sceneKnowledge"]["objects"]} >= {
        "nightstand",
        "plant",
    }
    candidate = manifest["candidateBuild"]
    assert candidate["unadoptedObjectPolicy"] == (
        "original_static_scene_only_with_logical_ancestors"
    )
    assert candidate["logicalAncestorObjectIds"] == ["bed"]
    assert candidate["removedUnadoptedObjectIds"] == ["nightstand", "plant"]
    assert report["unadopted_object_policy"]["original_static_scene_is_sole_representation"] is True


def test_optional_browser_lod_override_keeps_receipt_pivot(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    override = _add_override(paths)

    manifest, report = materialize_candidate(
        project_root=paths["project_root"],
        config_path=paths["config"],
        output_manifest=paths["output"],
        report_path=paths["report"],
    )

    bed = next(item for item in manifest["interactiveObjects"] if item["id"] == "bed")
    assert bed["placement"]["pivot"] == [4.0, 1.5, 9.0]
    assert bed["placement"]["scale"] == [1, 1, 1]
    assert bed["placement"]["rotationEulerDeg"] == [0, 0, 0]
    assert bed["collision"]["asset"]["sha256"] == sha256_file(override["output"])[0]
    assert bed["collision"]["gate"]["browserLodQaSha256"] == sha256_file(override["qa"])[0]
    assert (
        report["objects"]["bed"]["scene_fit_source_glb"]["sha256"]
        == sha256_file(override["source"])[0]
    )
    assert report["objects"]["bed"]["browser_lod_override"]["placement_policy"] == (
        "inherit_scene_fit_receipt"
    )


def test_accepts_exact_blender_normalization_of_audited_source_degenerates(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    _add_blender_normalized_degenerate_override(paths)

    manifest, report = materialize_candidate(
        project_root=paths["project_root"],
        config_path=paths["config"],
        output_manifest=paths["output"],
        report_path=paths["report"],
    )

    bed = next(item for item in manifest["interactiveObjects"] if item["id"] == "bed")
    assert bed["collision"]["asset"]["nondegenerate"] is True
    normalization = report["objects"]["bed"]["browser_lod_override"][
        "source_normalization"
    ]
    assert normalization == {
        "mode": "blender_removed_audited_degenerate_triangles",
        "tool": "blender_headless_pbr_decimate",
        "audited_source_faces": 13,
        "audited_degenerate_triangles": 1,
        "normalized_source_faces": 12,
        "audited_normal_loops": 39,
        "normalized_normal_loops": 36,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("extra_face_removed", "input stats are not an exact audited-degenerate normalization"),
        ("extra_normal_loops_removed", "input stats are not an exact audited-degenerate"),
        ("wrong_processing_tool", "was not recorded by the Blender decimator"),
        ("wrong_processing_face_count", "Blender normalized source face count mismatch"),
    ],
)
def test_rejects_unverified_source_degenerate_normalization(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    paths = _fixture(tmp_path)
    override = _add_blender_normalized_degenerate_override(paths)
    qa = json.loads(override["qa"].read_text(encoding="utf-8"))
    if mutation == "extra_face_removed":
        qa["assetBinding"]["input"]["faces"] -= 1
    elif mutation == "extra_normal_loops_removed":
        qa["assetBinding"]["input"]["normalLoops"] -= 3
    elif mutation == "wrong_processing_tool":
        qa["processing"]["tool"] = "unverified_mesh_rewriter"
    elif mutation == "wrong_processing_face_count":
        qa["processing"]["sourceFacesAfterTriangulate"] += 1
    _write_json(override["qa"], qa)
    _refresh_qa_hash(paths, override["qa"])

    with pytest.raises(RuntimeError, match=message):
        materialize_candidate(
            project_root=paths["project_root"],
            config_path=paths["config"],
            output_manifest=paths["output"],
            report_path=paths["report"],
        )


def test_rejects_degenerate_override_output_after_source_normalization(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    override = _add_blender_normalized_degenerate_override(paths)
    _write_pbr_glb(
        override["output"],
        extents=(2.4, 1.1, 3.2),
        degenerate_triangles=1,
    )
    output = audit_unified_glb(
        override["output"],
        maximum_faces=100_000,
        enforce_runtime_geometry=False,
    )
    qa = json.loads(override["qa"].read_text(encoding="utf-8"))
    qa["output"] = _qa_stats(output)
    qa["assetBinding"]["output"] = _qa_binding(output)
    _write_json(override["qa"], qa)

    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    config["object_overrides"]["bed"]["browser_lod"]["sha256"] = output.sha256
    config["object_overrides"]["bed"]["qa"]["sha256"] = sha256_file(override["qa"])[0]
    _write_json(paths["config"], config)

    with pytest.raises(RuntimeError, match="GLB contains degenerate faces"):
        materialize_candidate(
            project_root=paths["project_root"],
            config_path=paths["config"],
            output_manifest=paths["output"],
            report_path=paths["report"],
        )


def test_rejects_tampered_override_qa_file(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    override = _add_override(paths)
    override["qa"].write_text(override["qa"].read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"bed\.qa sha256 mismatch"):
        materialize_candidate(
            project_root=paths["project_root"],
            config_path=paths["config"],
            output_manifest=paths["output"],
            report_path=paths["report"],
        )


def test_rejects_failed_override_required_qa_gate(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    override = _add_override(paths)
    qa = json.loads(override["qa"].read_text(encoding="utf-8"))
    qa["qa"]["normalsPreserved"] = False
    _write_json(override["qa"], qa)
    _refresh_qa_hash(paths, override["qa"])

    with pytest.raises(RuntimeError, match="required gates failed"):
        materialize_candidate(
            project_root=paths["project_root"],
            config_path=paths["config"],
            output_manifest=paths["output"],
            report_path=paths["report"],
        )


def test_rejects_override_source_sha_not_bound_to_receipt(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _add_override(paths)
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    config["object_overrides"]["bed"]["source_asset"]["sha256"] = "e" * 64
    _write_json(paths["config"], config)

    with pytest.raises(RuntimeError, match="source sha256 is not the receipt"):
        materialize_candidate(
            project_root=paths["project_root"],
            config_path=paths["config"],
            output_manifest=paths["output"],
            report_path=paths["report"],
        )


@pytest.mark.parametrize("binding_key", ["input", "output"])
def test_rejects_wrong_override_qa_asset_binding_sha(tmp_path: Path, binding_key: str) -> None:
    paths = _fixture(tmp_path)
    override = _add_override(paths)
    qa = json.loads(override["qa"].read_text(encoding="utf-8"))
    qa["assetBinding"][binding_key]["sha256"] = "f" * 64
    _write_json(override["qa"], qa)
    _refresh_qa_hash(paths, override["qa"])

    expected = "input sha256 is not" if binding_key == "input" else "output sha256 mismatch"
    with pytest.raises(RuntimeError, match=expected):
        materialize_candidate(
            project_root=paths["project_root"],
            config_path=paths["config"],
            output_manifest=paths["output"],
            report_path=paths["report"],
        )


def test_rejects_world_baked_override_masquerading_as_scene_local(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _add_override(paths)
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    config["object_overrides"]["bed"]["browser_lod"]["coordinate_space"] = "world_baked"
    _write_json(paths["config"], config)

    with pytest.raises(RuntimeError, match="coordinate space must be scene-local"):
        materialize_candidate(
            project_root=paths["project_root"],
            config_path=paths["config"],
            output_manifest=paths["output"],
            report_path=paths["report"],
        )


def test_rejects_translated_world_bake_labeled_as_scene_local(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    override = _add_override(paths)
    scene = trimesh.load(override["source"], force="scene", process=False)
    scene.apply_translation([5.0, 0.0, 0.0])
    scene.export(override["output"], include_normals=True)
    _write_lod_qa(
        override["qa"],
        source_path=override["source"],
        output_path=override["output"],
    )
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    config["object_overrides"]["bed"]["browser_lod"]["sha256"] = sha256_file(override["output"])[0]
    config["object_overrides"]["bed"]["qa"]["sha256"] = sha256_file(override["qa"])[0]
    _write_json(paths["config"], config)

    with pytest.raises(RuntimeError, match="bounds drift exceeds QA tolerance"):
        materialize_candidate(
            project_root=paths["project_root"],
            config_path=paths["config"],
            output_manifest=paths["output"],
            report_path=paths["report"],
        )


def test_rejects_override_output_bytes_changed_after_qa(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    override = _add_override(paths)
    override["output"].write_bytes(override["output"].read_bytes() + b"changed")

    with pytest.raises(RuntimeError, match=r"bed\.browser_lod sha256 mismatch"):
        materialize_candidate(
            project_root=paths["project_root"],
            config_path=paths["config"],
            output_manifest=paths["output"],
            report_path=paths["report"],
        )


def test_rejects_receipt_bounds_that_do_not_match_final_glb_bytes(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    receipt_path = Path(config["objects"][0]["scene_fit_receipt"]["path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["exports"]["scene_local"]["glb"]["bounds"][1][0] += 0.25
    _write_json(receipt_path, receipt)
    config["objects"][0]["scene_fit_receipt"]["sha256"] = sha256_file(receipt_path)[0]
    _write_json(paths["config"], config)

    with pytest.raises(RuntimeError, match="final GLB bounds mismatch: bed"):
        materialize_candidate(
            project_root=paths["project_root"],
            config_path=paths["config"],
            output_manifest=paths["output"],
            report_path=paths["report"],
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "static_collider reference is missing"),
        ("role", "static_collider.role must explicitly identify"),
        ("hash", "static_collider sha256 mismatch"),
    ],
)
def test_requires_hash_bound_original_static_collider(
    tmp_path: Path, mutation: str, message: str
) -> None:
    paths = _fixture(tmp_path)
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    if mutation == "missing":
        config.pop("static_collider")
    elif mutation == "role":
        config["static_collider"]["role"] = "strict_clean_scene_mesh"
    else:
        config["static_collider"]["sha256"] = "d" * 64
    _write_json(paths["config"], config)

    with pytest.raises(RuntimeError, match=message):
        materialize_candidate(
            project_root=paths["project_root"],
            config_path=paths["config"],
            output_manifest=paths["output"],
            report_path=paths["report"],
        )


@pytest.mark.parametrize("value", [None, "keep_all_base_interactive_objects"])
def test_requires_explicit_original_static_unadopted_object_policy(
    tmp_path: Path, value: str | None
) -> None:
    paths = _fixture(tmp_path)
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    if value is None:
        config.pop("unadopted_object_policy")
    else:
        config["unadopted_object_policy"] = value
    _write_json(paths["config"], config)

    with pytest.raises(RuntimeError, match="unadopted_object_policy must be"):
        materialize_candidate(
            project_root=paths["project_root"],
            config_path=paths["config"],
            output_manifest=paths["output"],
            report_path=paths["report"],
        )


def test_rejects_final_glb_over_configured_face_budget(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    config["maximum_collision_faces"] = 11
    _write_json(paths["config"], config)

    with pytest.raises(RuntimeError, match="GLB exceeds face budget"):
        materialize_candidate(
            project_root=paths["project_root"],
            config_path=paths["config"],
            output_manifest=paths["output"],
            report_path=paths["report"],
        )


def test_rejects_output_inside_public_manifest_tree(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    with pytest.raises(RuntimeError, match="must stay outside web/public"):
        materialize_candidate(
            project_root=paths["project_root"],
            config_path=paths["config"],
            output_manifest=paths["project_root"] / "web/public/worlds/bedroom4/direct.json",
            report_path=paths["report"],
        )


@pytest.mark.parametrize("input_key", ["config", "static", "collider", "receipt"])
def test_rejects_candidate_outputs_that_overwrite_inputs(
    tmp_path: Path, input_key: str
) -> None:
    paths = _fixture(tmp_path)
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    protected_path = (
        Path(config["objects"][0]["scene_fit_receipt"]["path"])
        if input_key == "receipt"
        else paths[input_key]
    )
    original_bytes = protected_path.read_bytes()

    with pytest.raises(RuntimeError, match="candidate outputs must not overwrite inputs"):
        materialize_candidate(
            project_root=paths["project_root"],
            config_path=paths["config"],
            output_manifest=protected_path,
            report_path=paths["report"],
        )

    assert protected_path.read_bytes() == original_bytes
