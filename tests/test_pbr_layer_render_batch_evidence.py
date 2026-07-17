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
    / "pbr-layer-render-batch-v2-geometry-matte"
)
V1_ROOT = ROOT.parent / "pbr-layer-render-batch-v1"
LAYER_ORDER = (
    "sam3_pillow_front",
    "sam3_pillow_left",
    "sam3_pillow_right",
    "sam3_bed_01",
)
FRAME_IDS = [f"{frame:06d}" for frame in range(48, 73)]
CONTACT_FRAME_IDS = ("000048", "000064", "000067", "000072")


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


def payload_digest() -> dict[str, Any]:
    paths = sorted(
        (
            path
            for root_name in ("layers", "source")
            for path in (ROOT / root_name).rglob("*")
            if path.is_file() and not path.name.startswith(".")
        ),
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )
    digest = hashlib.sha256()
    total_bytes = 0
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
        total_bytes += path.stat().st_size
    return {
        "sha256": digest.hexdigest(),
        "file_count": len(paths),
        "bytes": total_bytes,
    }


def assert_artifact(record: dict[str, Any]) -> Path:
    path = ROOT / str(record["path"])
    assert path.is_file()
    assert record["sha256"] == sha256_file(path)
    assert record["bytes"] == path.stat().st_size
    return path


def test_geometry_matte_batch_evidence_is_hash_bound_and_complete() -> None:
    report = read_json(ROOT / "pbr_layer_batch_qa.json")
    render_index = read_json(ROOT / "render_index.json")
    batch_receipt = read_json(ROOT / "batch_receipt.json")
    old_status = read_json(V1_ROOT / "batch_status.json")

    assert report["status"] == "passed_4_layers_x_25_frames_not_promoted"
    assert report["promotion_approved"] is False
    assert report["frame_layer_asset_count"] == 100
    assert all(report["gates"].values())
    assert render_index["status"] == "technical_passed_100_frame_asset_set_not_promoted"
    assert render_index["promotion_approved"] is False
    assert render_index["frame_ids"] == FRAME_IDS
    assert tuple(render_index["layer_order"]) == LAYER_ORDER
    assert render_index["contract"]["alpha_depth_alignment_mode"] == (
        "geometry-derived-binary-matte"
    )
    assert render_index["contract"]["this_index_is_not_a_final_composite"] is True
    assert old_status["status"] == "rejected_superseded"
    assert old_status["promotion_approved"] is False

    actual_payload = payload_digest()
    for evidence in (report["payload_artifact_set"], batch_receipt["payload_artifact_set"]):
        assert evidence["sha256"] == actual_payload["sha256"]
        assert evidence["file_count"] == actual_payload["file_count"]
        assert evidence["bytes"] == actual_payload["bytes"]
    assert_artifact(batch_receipt["batch_report"])
    assert_artifact(batch_receipt["render_index"])
    contact_path = assert_artifact(batch_receipt["contact_sheet"])
    with Image.open(contact_path) as contact:
        assert contact.size == (1600, 830)

    layer_reports = {item["layer_id"]: item for item in report["layers"]}
    assert tuple(layer_reports) == LAYER_ORDER
    for layer_id in LAYER_ORDER:
        layer = layer_reports[layer_id]
        assert layer["status"] == "passed_25_frame_geometry_matte_audit"
        assert layer["frame_count"] == 25
        assert layer["minimum_initial_alpha_vs_depth_iou"] >= 0.98
        assert layer["maximum_changed_union_fraction"] < 0.02
        assert layer["minimum_analytic_mask_iou"] >= 0.85
        assert layer["minimum_analytic_bbox_iou"] >= 0.8
        assert layer["per_frame_raw_depth_exrs_retained"] == 0
        assert_artifact(layer["manifest"])
        assert_artifact(layer["manifest_receipt"])
        assert_artifact(layer["depth_probe_receipt"])

    for frame_id in FRAME_IDS:
        frame = render_index["frames"][frame_id]
        assert_artifact(frame["source_frame"])
        assert set(frame["layers"]) == set(LAYER_ORDER)
        for layer_id in LAYER_ORDER:
            indexed = frame["layers"][layer_id]
            rgba_path = assert_artifact(indexed["rgba"])
            depth_path = assert_artifact(indexed["depth"])
            receipt_path = assert_artifact(indexed["receipt"])
            receipt = read_json(receipt_path)
            assert receipt["alpha_depth_alignment_mode"] == (
                "geometry-derived-binary-matte"
            )
            assert receipt["render_configuration"]["taa_render_samples"] == 64
            assert receipt["geometry_matte_initial_alpha_vs_depth_iou"] >= 0.98
            assert receipt["geometry_matte_changed_union_fraction"] < 0.02
            assert receipt["geometry_matte_changed_pixels"] == (
                receipt["geometry_matte_cleared_pixels"]
                + receipt["geometry_matte_filled_pixels"]
            )
            assert receipt["raw_object_depth_vs_final_alpha_iou"] == 1.0
            assert all(receipt["gates"].values())
            assert indexed["alpha_pixels"] == indexed["depth_valid_pixels"]
            if frame_id in CONTACT_FRAME_IDS:
                rgba = np.asarray(Image.open(rgba_path).convert("RGBA"), dtype=np.uint8)
                alpha = rgba[:, :, 3] == 255
                assert set(np.unique(rgba[:, :, 3]).tolist()).issubset({0, 255})
                depth = np.load(depth_path, allow_pickle=False)
                assert np.array_equal(alpha, np.isfinite(depth))
                assert np.all(depth[alpha] > 0)
                assert np.all(np.isnan(depth[~alpha]))
