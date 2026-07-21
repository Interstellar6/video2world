from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from plyfile import PlyData, PlyElement

from scripts.materialize_targeted_potted_plant_masks import materialize
from video2world.hashing import sha256_file


def write_json(path: Path, value: object) -> str:
    path.write_text(json.dumps(value), encoding="utf-8")
    return sha256_file(path)[0]


def write_mask(path: Path, box: tuple[int, int, int, int]) -> str:
    mask = np.zeros((80, 100), dtype=np.uint8)
    x0, y0, x1, y1 = box
    mask[y0:y1, x0:x1] = 255
    Image.fromarray(mask).save(path)
    return sha256_file(path)[0]


def compressed_coco_rle(path: Path) -> dict[str, object]:
    mask = np.asarray(Image.open(path).convert("L")) > 0
    flat = mask.reshape(-1, order="F")
    counts: list[int] = []
    value = False
    run = 0
    for pixel in flat:
        if bool(pixel) == value:
            run += 1
            continue
        counts.append(run)
        run = 1
        value = not value
    counts.append(run)

    encoded = []
    for index, raw_count in enumerate(counts):
        count = raw_count - counts[index - 2] if index > 2 else raw_count
        more = True
        while more:
            char = count & 0x1F
            count >>= 5
            more = count != (-1 if char & 0x10 else 0)
            if more:
                char |= 0x20
            encoded.append(chr(char + 48))
    return {"size": list(mask.shape), "counts": "".join(encoded)}


def write_anchor(path: Path, *, center_x: float) -> str:
    values = np.zeros(64, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    grid = np.linspace(-0.4, 0.4, 8)
    values["x"] = np.repeat(center_x + grid, 8)
    values["y"] = np.tile(np.linspace(-3.0, -0.2, 8), 8)
    values["z"] = 10.0
    PlyData([PlyElement.describe(values, "vertex")], text=True).write(path)
    return sha256_file(path)[0]


def make_fixture(tmp_path: Path) -> Path:
    source = np.full((80, 100, 3), [90, 72, 58], dtype=np.uint8)
    source[20:58, 25:45] = [42, 130, 55]
    source[20:58, 55:75] = [38, 118, 48]
    source_path = tmp_path / "000001.png"
    Image.fromarray(source).save(source_path)
    source_sha = sha256_file(source_path)[0]
    original_root = tmp_path / "original" / "000001"
    original_root.mkdir(parents=True)
    foliage_a = original_root / "plant_01.png"
    foliage_b = original_root / "plant_02.png"
    foliage_a_sha = write_mask(foliage_a, (25, 20, 45, 40))
    foliage_b_sha = write_mask(foliage_b, (55, 20, 75, 40))
    write_mask(original_root / "nightstand_01.png", (0, 55, 20, 75))
    write_mask(original_root / "lamp_01.png", (80, 5, 98, 30))
    original_index_path = tmp_path / "original-index.json"
    original_index_sha = write_json(
        original_index_path,
        {
            "items": [
                {"image": "000001.png", "label": "plant", "mask_path": "plant_01.png"},
                {"image": "000001.png", "label": "plant", "mask_path": "plant_02.png"},
                {
                    "image": "000001.png",
                    "label": "nightstand",
                    "mask_path": "nightstand_01.png",
                },
                {"image": "000001.png", "label": "lamp", "mask_path": "lamp_01.png"},
            ]
        },
    )
    targeted_root = tmp_path / "targeted" / "000001"
    targeted_root.mkdir(parents=True)
    target_boxes = {
        "potted_plant_01.png": (25, 20, 45, 58),
        "potted_plant_02.png": (55, 20, 75, 58),
        "flower_pot_01.png": (31, 40, 39, 55),
        "flower_pot_02.png": (61, 40, 69, 55),
        "planter_01.png": (31, 40, 39, 55),
        "planter_02.png": (61, 40, 69, 55),
    }
    for name, box in target_boxes.items():
        write_mask(targeted_root / name, box)
    target_items = []
    for label, names in {
        "potted plant": ["potted_plant_01.png", "potted_plant_02.png"],
        "flower pot": ["flower_pot_01.png", "flower_pot_02.png"],
        "planter": ["planter_01.png", "planter_02.png"],
    }.items():
        for index, name in enumerate(names):
            target_items.append(
                {
                    "image": "000001.png",
                    "label": label,
                    "mask_path": f"remote/output/bedroom_4/000001/{name}",
                    "mask_rle": compressed_coco_rle(targeted_root / name),
                    "bbox": list(target_boxes[name]),
                    "score": 0.95 - index * 0.01,
                }
            )
    targeted_index_path = tmp_path / "targeted-index.json"
    targeted_index_sha = write_json(
        targeted_index_path,
        {"scene": "bedroom_4", "missing_images": [], "items": target_items},
    )
    anchor_a = tmp_path / "plant_a.ply"
    anchor_b = tmp_path / "plant_b.ply"
    anchor_a_sha = write_anchor(anchor_a, center_x=-3.0)
    anchor_b_sha = write_anchor(anchor_b, center_x=3.0)
    audit_path = tmp_path / "source-audit.json"
    audit_sha = write_json(
        audit_path,
        {
            "kind": "video2world.multiview_scene_fit_evidence_audit",
            "status": "source_multiview_modal_masks_ready_for_scene_fit",
            "objects": [
                {
                    "object_id": "plant_a",
                    "source_anchor": {"path": str(anchor_a), "sha256": anchor_a_sha},
                    "frame_selection": {"selected_frame_ids": ["000001"]},
                    "views": [
                        {
                            "frame_id": "000001",
                            "association": {"status": "passed"},
                            "source_rgb": {"path": str(source_path), "sha256": source_sha},
                            "mask": {"path": str(foliage_a), "sha256": foliage_a_sha},
                        }
                    ],
                },
                {
                    "object_id": "plant_b",
                    "source_anchor": {"path": str(anchor_b), "sha256": anchor_b_sha},
                    "frame_selection": {"selected_frame_ids": ["000001"]},
                    "views": [
                        {
                            "frame_id": "000001",
                            "association": {"status": "passed"},
                            "source_rgb": {"path": str(source_path), "sha256": source_sha},
                            "mask": {"path": str(foliage_b), "sha256": foliage_b_sha},
                        }
                    ],
                },
            ],
        },
    )
    cameras_path = tmp_path / "cameras.json"
    cameras_sha = write_json(
        cameras_path,
        [
            {
                "id": 1,
                "img_name": "000001",
                "width": 100,
                "height": 80,
                "position": [0.0, 0.0, 0.0],
                "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "fx": 50.0,
                "fy": 50.0,
            }
        ],
    )
    scene_path = tmp_path / "scene.json"
    runner_path = tmp_path / "runner.py"
    log_path = tmp_path / "run.log"
    scene_sha = write_json(scene_path, {"scene": "bedroom_4"})
    runner_path.write_text("# runner snapshot\n", encoding="utf-8")
    log_path.write_text("completed\n", encoding="utf-8")
    config_path = tmp_path / "config.json"
    write_json(
        config_path,
        {
            "schema_version": 1,
            "kind": "video2world.targeted_potted_plant_mask_config",
            "source_audit": {"path": str(audit_path), "sha256": audit_sha},
            "cameras": {"path": str(cameras_path), "sha256": cameras_sha},
            "original_mask_index": {
                "path": str(original_index_path),
                "sha256": original_index_sha,
            },
            "original_mask_root": str(tmp_path / "original"),
            "targeted_mask_index": {
                "path": str(targeted_index_path),
                "sha256": targeted_index_sha,
            },
            "targeted_mask_root": str(tmp_path / "targeted"),
            "object_ids": ["plant_a", "plant_b"],
            "prompt_contract": {
                "complete_instance": "potted plant",
                "container_parts": ["flower pot", "planter"],
                "contaminant_labels": ["nightstand", "lamp"],
            },
            "thresholds": {
                "minimum_complete_anchor_hit_ratio": 0.5,
                "minimum_complete_anchor_hit_margin": 0.5,
                "minimum_foliage_coverage": 0.95,
                "minimum_container_inside_complete": 0.9,
                "minimum_container_outside_foliage": 0.9,
                "maximum_contaminant_overlap": 0.01,
                "maximum_horizontal_center_offset_foliage_widths": 0.25,
                "maximum_container_top_gap_foliage_heights": 0.1,
                "minimum_complete_bottom_extension_foliage_heights": 0.5,
            },
            "inference": {
                "remote_host": "fixture",
                "remote_output_directory": "/remote/output",
                "scene_config": {"path": str(scene_path), "sha256": scene_sha},
                "runner": {"path": str(runner_path), "sha256": sha256_file(runner_path)[0]},
                "log": {"path": str(log_path), "sha256": sha256_file(log_path)[0]},
                "checkpoint": {"path": "/remote/sam3.pt", "sha256": "a" * 64, "size_bytes": 1},
                "sam3_source_files": [
                    {"path": "/remote/model_builder.py", "sha256": "b" * 64, "size_bytes": 1}
                ],
                "argv": ["python", "runner.py"],
                "environment": {"python": "test"},
            },
            "review": {
                "columns": 1,
                "panel_width": 120,
                "panel_height": 100,
                "crop_padding_ratio": 0.2,
            },
            "output_directory": str(tmp_path / "output"),
        },
    )
    return config_path


def test_materializes_anchor_associated_byte_identical_masks(tmp_path: Path) -> None:
    config_path = make_fixture(tmp_path)

    result = materialize(config_path)

    assert result["status"] == "passed"
    assert result["object_count"] == 2
    manifest = json.loads(Path(result["manifest"]["path"]).read_text(encoding="utf-8"))
    assert manifest["status"] == "passed"
    assert manifest["consumable_for_object_conditioning"] is True
    for object_record in manifest["objects"]:
        frame = object_record["frames"][0]
        assert frame["association"]["selected_anchor_hit_ratio"] == pytest.approx(1.0)
        assert frame["association"]["complete_contaminant_overlap_ratio"] == 0.0
        assert all(frame["gates"].values())
        output_mask = (
            Path(result["manifest"]["path"]).parent / frame["complete_instance_output_mask"]["path"]
        )
        assert sha256_file(output_mask)[0] == frame["complete_instance_source_mask"]["sha256"]
    for view_record in result["view_manifests"]:
        view = json.loads(Path(view_record["path"]).read_text(encoding="utf-8"))
        observed = Path(view_record["path"]).parent / view["views"][0]["observed_mask"]["path"]
        assert sha256_file(observed)[0] == view["views"][0]["observed_mask"]["sha256"]


def test_rejects_neighbor_contamination(tmp_path: Path) -> None:
    config_path = make_fixture(tmp_path)
    original_root = tmp_path / "original" / "000001"
    write_mask(original_root / "nightstand_01.png", (20, 18, 48, 60))

    with pytest.raises(RuntimeError, match="no container-part prompt passed for plant_a"):
        materialize(config_path)


def test_rejects_changed_source_rgb(tmp_path: Path) -> None:
    config_path = make_fixture(tmp_path)
    source_path = tmp_path / "000001.png"
    Image.fromarray(np.zeros((80, 100, 3), dtype=np.uint8)).save(source_path)

    with pytest.raises(RuntimeError, match="source RGB hash changed"):
        materialize(config_path)


def test_rejects_targeted_png_that_differs_from_hash_bound_index_rle(tmp_path: Path) -> None:
    config_path = make_fixture(tmp_path)
    write_mask(tmp_path / "targeted/000001/potted_plant_01.png", (24, 20, 44, 58))

    with pytest.raises(RuntimeError, match="PNG differs from hash-bound index RLE"):
        materialize(config_path)
