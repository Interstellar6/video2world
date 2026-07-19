from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = (
    Path(__file__).parents[1]
    / "examples"
    / "bedroom4"
    / "completion"
    / "layered-peel"
    / "pbr-layer-render-qa-000067"
)
LAYER_ORDER = (
    "sam3_pillow_front",
    "sam3_pillow_left",
    "sam3_pillow_right",
    "sam3_bed_01",
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_single_frame_pbr_layer_evidence_is_self_consistent() -> None:
    report = read_json(ROOT / "source_camera_pbr_layer_qa.json")
    render_index = read_json(ROOT / "render_index.json")
    index_receipt = read_json(ROOT / "render_index_receipt.json")
    assert report["status"] == "rejected"
    assert report["promotion_approved"] is False
    assert render_index["status"] == "rejected"
    assert render_index["frame_ids"] == ["000067"]
    assert render_index["layer_order"] == list(LAYER_ORDER)
    assert render_index["contract"]["this_index_is_not_a_final_composite"] is True
    assert index_receipt["render_index_sha256"] == sha256_file(ROOT / "render_index.json")
    assert render_index["round_remaining_object_ids"] == {
        "round01_front_pillow": [
            "sam3_pillow_left",
            "sam3_pillow_right",
            "sam3_bed_01",
        ],
        "round02_left_pillow": ["sam3_pillow_right", "sam3_bed_01"],
        "round03_right_pillow": ["sam3_bed_01"],
        "round04_bed": [],
    }

    report_layers = {item["layer_id"]: item for item in report["layers"]}
    assert tuple(report_layers) == LAYER_ORDER
    for layer_id in LAYER_ORDER:
        indexed = render_index["frames"]["000067"][layer_id]
        rgba_path = Path(indexed["rgba"]["path"])
        depth_path = Path(indexed["depth"]["path"])
        receipt_path = Path(indexed["receipt"]["path"])
        assert indexed["rgba"]["sha256"] == sha256_file(rgba_path)
        assert indexed["depth"]["sha256"] == sha256_file(depth_path)
        assert indexed["receipt"]["sha256"] == sha256_file(receipt_path)
        receipt = read_json(receipt_path)
        assert receipt["kind"] == "video2world.pbr_layer_render_receipt"
        assert receipt["claims_measured_donor"] is False
        assert all(receipt["gates"].values())
        assert receipt["rgba_alpha_sanitized_fraction"] <= 0.001
        assert receipt["depth_max"] < receipt["evaluated_geometry_depth_max"] + 1e-3
        rgba = np.asarray(Image.open(rgba_path).convert("RGBA"), dtype=np.uint8)
        alpha = rgba[:, :, 3] >= 128
        depth = np.load(depth_path)
        valid_depth = np.isfinite(depth) & (depth > 0)
        assert np.array_equal(alpha, valid_depth)
        assert int(alpha.sum()) == receipt["alpha_pixels"] == receipt["depth_valid_pixels"]
        assert report_layers[layer_id]["status"] == "rejected"
        assert report_layers[layer_id]["gates"]["observed_mask_iou"] is False
        assert (
            report_layers[layer_id]["metrics"]["mask_iou"]
            >= report_layers[layer_id]["thresholds"]["minimum_mask_iou"]
        )

    outputs = report["outputs"]
    assert outputs["contact_sheet"]["sha256"] == sha256_file(Path(outputs["contact_sheet"]["path"]))
    assert outputs["observed_sam_contact_sheet"]["sha256"] == sha256_file(
        Path(outputs["observed_sam_contact_sheet"]["path"])
    )
