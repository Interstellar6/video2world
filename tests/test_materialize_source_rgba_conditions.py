from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.materialize_source_rgba_conditions import run


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def make_view(
    root: Path,
    *,
    frame_id: str,
    bbox: tuple[int, int, int, int],
    color: tuple[int, int, int],
) -> dict:
    width, height = 32, 24
    source = np.zeros((height, width, 3), dtype=np.uint8)
    rows, columns = np.indices((height, width))
    source[..., 0] = (color[0] + columns) % 255
    source[..., 1] = (color[1] + rows) % 255
    source[..., 2] = color[2]
    mask = np.zeros((height, width), dtype=np.uint8)
    x0, y0, x1, y1 = bbox
    mask[y0:y1, x0:x1] = 255
    source_path = root / f"{frame_id}.png"
    mask_path = root / f"{frame_id}-mask.png"
    Image.fromarray(source).save(source_path)
    Image.fromarray(mask).save(mask_path)
    return {
        "frame_id": frame_id,
        "association": {"status": "passed_unambiguous_for_selected_frame"},
        "source_rgb": {
            "absolute_path": str(source_path),
            "repo_uri": f"repo://fixture/{source_path.name}",
            "authoritative_upstream_path": f"/upstream/{source_path.name}",
            "authoritative_receipt_hash_match": True,
            "dimensions_wh": [width, height],
            "sha256": sha256(source_path),
        },
        "mask": {
            "absolute_path": str(mask_path),
            "file_name": mask_path.name,
            "pixel_bbox_xyxy_exclusive": list(bbox),
            "pixel_count": (x1 - x0) * (y1 - y0),
            "sha256": sha256(mask_path),
        },
    }


def fixture(
    tmp_path: Path,
    views: list[dict],
    *,
    audit_mutation: dict | None = None,
) -> tuple[Path, Path]:
    audit = {
        "kind": "video2world.direct_trellis2_source_view_audit",
        "objects": [
            {
                "object_id": "object_01",
                "category": "fixture",
                "views": views,
            }
        ],
    }
    if audit_mutation:
        audit.update(audit_mutation)
    audit_path = tmp_path / "audit.json"
    write_json(audit_path, audit)
    config_path = tmp_path / "config.json"
    write_json(
        config_path,
        {
            "kind": "video2world.source_rgba_condition_config",
            "audit": {"path": "audit.json", "sha256": sha256(audit_path)},
            "object_ids": ["object_01"],
            "selection": {
                "ranking": "foreground_pixel_count_desc_then_frame_id",
                "minimum_edge_distance_px": 1,
            },
            "crop": {
                "padding_ratio": 0.12,
                "condition_size": 1024,
                "background": "transparent",
                "emit_white_background": True,
            },
            "output_directory": "output",
        },
    )
    return config_path, tmp_path / "output"


def test_selects_largest_non_edge_view_and_preserves_raw_foreground(tmp_path: Path) -> None:
    edge_large = make_view(
        tmp_path,
        frame_id="000001",
        bbox=(0, 3, 18, 20),
        color=(20, 50, 80),
    )
    smaller = make_view(
        tmp_path,
        frame_id="000002",
        bbox=(4, 4, 14, 16),
        color=(90, 100, 110),
    )
    selected = make_view(
        tmp_path,
        frame_id="000003",
        bbox=(3, 3, 17, 18),
        color=(140, 150, 160),
    )
    config, output = fixture(tmp_path, [edge_large, smaller, selected])

    report = run(config)

    assert report["status"] == "technical_passed_source_conditions_materialized"
    object_report = report["objects"][0]
    assert object_report["selection"]["selected_frame_id"] == "000003"
    candidates = {item["frame_id"]: item for item in object_report["selection"]["candidates"]}
    assert candidates["000001"]["eligible_non_edge_view"] is False
    assert candidates["000003"]["foreground_bbox_dimensions_wh"] == [14, 15]
    raw = np.asarray(Image.open(output / "object_01" / "raw-source-crop-rgba.png"))
    source = np.asarray(Image.open(tmp_path / "000003.png").convert("RGB"))
    mask = np.asarray(Image.open(tmp_path / "000003-mask.png").convert("L")) > 0
    assert np.array_equal(raw[raw[..., 3] == 255, :3], source[mask])
    assert np.all(raw[raw[..., 3] == 0, :3] == 0)
    assert Image.open(output / "object_01" / "condition-1024-rgba.png").size == (1024, 1024)
    assert Image.open(output / "object_01" / "condition-1024-white.png").mode == "RGB"
    assert (output / "object_01" / "source-condition-four-panel.png").is_file()


def test_fails_closed_when_every_view_touches_an_edge(tmp_path: Path) -> None:
    left = make_view(tmp_path, frame_id="000001", bbox=(0, 4, 8, 14), color=(1, 2, 3))
    bottom = make_view(tmp_path, frame_id="000002", bbox=(4, 10, 18, 24), color=(4, 5, 6))
    config, output = fixture(tmp_path, [left, bottom])

    with pytest.raises(RuntimeError, match="every audited SAM mask touches"):
        run(config)

    assert not output.exists()


def test_rejects_changed_hash_bound_source_before_writing(tmp_path: Path) -> None:
    view = make_view(tmp_path, frame_id="000001", bbox=(3, 3, 12, 14), color=(1, 2, 3))
    config, output = fixture(tmp_path, [view])
    (tmp_path / "000001.png").write_bytes(b"changed after audit")

    with pytest.raises(RuntimeError, match="source_rgb SHA-256 mismatch"):
        run(config)

    assert not output.exists()


def test_rejects_audit_bbox_that_differs_from_mask_pixels(tmp_path: Path) -> None:
    view = make_view(tmp_path, frame_id="000001", bbox=(3, 3, 12, 14), color=(1, 2, 3))
    view["mask"]["pixel_bbox_xyxy_exclusive"] = [4, 3, 12, 14]
    config, output = fixture(tmp_path, [view])

    with pytest.raises(RuntimeError, match="mask bbox differs from audit"):
        run(config)

    assert not output.exists()
