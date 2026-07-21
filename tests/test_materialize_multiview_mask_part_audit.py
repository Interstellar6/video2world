from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.materialize_multiview_mask_part_audit import materialize
from video2world.hashing import sha256_file


def write_json(path: Path, value: object) -> str:
    path.write_text(json.dumps(value), encoding="utf-8")
    return sha256_file(path)[0]


def make_fixture(tmp_path: Path) -> Path:
    source = np.full((80, 100, 3), 90, dtype=np.uint8)
    source[25:55, 35:65] = [40, 150, 60]
    source_path = tmp_path / "000001.png"
    Image.fromarray(source).save(source_path)
    mask = np.zeros((80, 100), dtype=np.uint8)
    mask[25:45, 35:65] = 255
    mask_path = tmp_path / "plant_01.png"
    Image.fromarray(mask).save(mask_path)
    source_sha = sha256_file(source_path)[0]
    mask_sha = sha256_file(mask_path)[0]
    mask_index_path = tmp_path / "mask-index.json"
    mask_index_sha = write_json(
        mask_index_path,
        {
            "items": [
                {
                    "image": "000001.png",
                    "label": "plant",
                    "mask_path": str(mask_path),
                }
            ]
        },
    )
    audit_path = tmp_path / "source-audit.json"
    audit_sha = write_json(
        audit_path,
        {
            "kind": "video2world.multiview_scene_fit_evidence_audit",
            "objects": [
                {
                    "object_id": "plant_a",
                    "views": [
                        {
                            "frame_id": "000001",
                            "association": {
                                "status": "passed",
                                "selected_mask_file_name": "plant_01.png",
                                "selected_hit_ratio": 0.8,
                                "hit_ratio_margin": 0.8,
                                "projected_anchor_bbox_xyxy": [34.0, 24.0, 66.0, 56.0],
                            },
                            "source_rgb": {"path": str(source_path), "sha256": source_sha},
                            "mask": {
                                "path": str(mask_path),
                                "sha256": mask_sha,
                                "pixel_bbox_xyxy_exclusive": [35, 25, 65, 45],
                            },
                        }
                    ],
                }
            ],
        },
    )
    config_path = tmp_path / "config.json"
    write_json(
        config_path,
        {
            "schema_version": 1,
            "kind": "video2world.multiview_mask_part_audit_config",
            "source_audit": {"path": str(audit_path), "sha256": audit_sha},
            "mask_index": {"path": str(mask_index_path), "sha256": mask_index_sha},
            "object_ids": ["plant_a"],
            "part_label_aliases": {
                "primary": ["plant"],
                "required_container": ["pot", "planter"],
            },
            "review": {
                "crop_padding_ratio": 0.5,
                "columns": 2,
                "panel_width": 160,
                "panel_height": 120,
            },
            "output_directory": str(tmp_path / "output"),
        },
    )
    return config_path


def test_materializes_fail_closed_contact_sheet(tmp_path: Path) -> None:
    config_path = make_fixture(tmp_path)

    result = materialize(config_path)

    report_path = Path(result["report"]["path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert result["status"] == "rejected_missing_required_part_mask"
    assert report["part_label_contract"]["missing_required_container_aliases"] == [
        "pot",
        "planter",
    ]
    assert report["mask_index_label_inventory"]["aggregate"] == {"plant": 1}
    object_report = report["objects"][0]
    assert object_report["union_ready"] is False
    assert object_report["frame_count"] == 1
    contact_path = Path(object_report["contact_sheet"]["path"])
    assert contact_path.is_file()
    assert sha256_file(contact_path)[0] == object_report["contact_sheet"]["sha256"]


def test_rejects_changed_source_audit(tmp_path: Path) -> None:
    config_path = make_fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    Path(config["source_audit"]["path"]).write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="source_audit SHA-256 mismatch"):
        materialize(config_path)
