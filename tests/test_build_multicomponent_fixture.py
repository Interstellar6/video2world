from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image
from plyfile import PlyData, PlyElement

from scripts.build_multicomponent_fixture import (
    audit_prior_rejections,
    build_fixture,
    sha256_file,
)


def _write_anchor(path: Path, *, width: float = 2.8) -> None:
    rng = np.random.default_rng(17)
    points = rng.uniform([-2.0, -width / 2.0, -0.65], [2.0, width / 2.0, 0.65], (1200, 3))
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    data = np.empty(len(points), dtype=dtype)
    data["x"], data["y"], data["z"] = points.T
    data["red"] = 150
    data["green"] = 145
    data["blue"] = 135
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(path)


def _write_evidence(root: Path, frame_id: str, *, with_gap: bool = True) -> tuple[Path, Path]:
    rgba = np.zeros((96, 128, 4), dtype=np.uint8)
    rgba[18:74, 12:22] = [78, 43, 26, 255]
    rgba[18:74, 106:116] = [78, 43, 26, 255]
    rgba[18:28, 12:116] = [82, 47, 29, 255]
    rgba[58:86, 14:114] = [190, 189, 184, 255]
    if not with_gap:
        rgba[28:58, 22:106] = [188, 187, 182, 255]
    mask = (rgba[..., 3] > 0).astype(np.uint8) * 255
    rgba_path = root / f"{frame_id}.rgba.png"
    mask_path = root / f"{frame_id}.mask.png"
    Image.fromarray(rgba).save(rgba_path)
    Image.fromarray(mask).save(mask_path)
    return rgba_path, mask_path


def _write_manifest(
    path: Path,
    *,
    object_id: str,
    evidence: list[tuple[Path, Path]],
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "video2world.layered_object_scene_references",
                "object_id": object_id,
                "identity_revalidated": True,
                "identity_revalidated_by": "synthetic_test_anchor_association",
                "frames": [
                    {
                        "frame_id": f"{index:06d}",
                        "image_sha256": sha256_file(rgba),
                        "mask_sha256": sha256_file(mask),
                    }
                    for index, (rgba, mask) in enumerate(evidence)
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _fixture_inputs(tmp_path: Path, *, with_gap: bool = True):  # type: ignore[no-untyped-def]
    anchor = tmp_path / "anchor.ply"
    _write_anchor(anchor)
    evidence = [
        _write_evidence(tmp_path, "frame_a", with_gap=with_gap),
        _write_evidence(tmp_path, "frame_b", with_gap=with_gap),
    ]
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, object_id="bed_fixture", evidence=evidence)
    return anchor, evidence, manifest


def test_builds_one_logical_root_with_internal_pbr_components(tmp_path: Path) -> None:
    anchor, evidence, manifest = _fixture_inputs(tmp_path)
    output = tmp_path / "output"
    report = build_fixture(
        object_id="bed_fixture",
        profile="bed",
        anchor_ply=anchor,
        evidence_rgba=[item[0] for item in evidence],
        evidence_masks=[item[1] for item in evidence],
        reference_manifest=manifest,
        output_dir=output,
        expected_anchor_sha256=sha256_file(anchor),
    )

    assert report["status"] == "technical_passed_visual_pending"
    assert report["all_technical_acceptance_gates_passed"] is True
    assert all(report["acceptance_gates"].values())
    logical = report["logical_entity"]
    assert logical["single_logical_root"] is True
    assert logical["internal_components_are_independent_scene_objects"] is False
    assert logical["independent_scene_object_ids"] == []
    assert set(logical["internal_component_names"]) == {
        "base_frame",
        "foot_cover",
        "headboard_panel",
        "headboard_post_left",
        "headboard_post_right",
        "mattress",
    }
    glb = output / "bed_fixture.component-assembly.glb"
    assert glb.is_file()
    loaded = trimesh.load_scene(glb)
    assert len(loaded.geometry) == 6
    assert all(
        type(geometry.visual.material).__name__ == "PBRMaterial"
        for geometry in loaded.geometry.values()
    )


def test_preserves_empty_pillow_clearance_without_component_overlap(tmp_path: Path) -> None:
    anchor, evidence, manifest = _fixture_inputs(tmp_path)
    report = build_fixture(
        object_id="bed_fixture",
        profile="bed",
        anchor_ply=anchor,
        evidence_rgba=[item[0] for item in evidence],
        evidence_masks=[item[1] for item in evidence],
        reference_manifest=manifest,
        output_dir=tmp_path / "output",
    )

    audit = report["component_audit"]
    assert audit["pillow_clearance_empty"] is True
    assert audit["no_positive_aabb_intersections"] is True
    assert audit["pairwise_positive_aabb_intersections"] == []
    assert report["acceptance_gates"]["negative_space_preservation"] is True
    assert report["acceptance_gates"]["internal_component_alignment"] is True


def test_requires_two_matching_evidence_frames(tmp_path: Path) -> None:
    anchor, evidence, manifest = _fixture_inputs(tmp_path)
    with pytest.raises(ValueError, match="at least two evidence"):
        build_fixture(
            object_id="bed_fixture",
            profile="bed",
            anchor_ply=anchor,
            evidence_rgba=[evidence[0][0]],
            evidence_masks=[evidence[0][1]],
            reference_manifest=manifest,
            output_dir=tmp_path / "output",
        )


def test_rejects_evidence_without_verified_negative_space(tmp_path: Path) -> None:
    anchor, evidence, manifest = _fixture_inputs(tmp_path, with_gap=False)
    with pytest.raises(ValueError, match="does not verify bounded negative space"):
        build_fixture(
            object_id="bed_fixture",
            profile="bed",
            anchor_ply=anchor,
            evidence_rgba=[item[0] for item in evidence],
            evidence_masks=[item[1] for item in evidence],
            reference_manifest=manifest,
            output_dir=tmp_path / "output",
        )


def test_rejects_anchor_hash_mismatch(tmp_path: Path) -> None:
    anchor, evidence, manifest = _fixture_inputs(tmp_path)
    wrong_hash = hashlib.sha256(b"wrong anchor").hexdigest()
    with pytest.raises(ValueError, match="anchor sha256 mismatch"):
        build_fixture(
            object_id="bed_fixture",
            profile="bed",
            anchor_ply=anchor,
            evidence_rgba=[item[0] for item in evidence],
            evidence_masks=[item[1] for item in evidence],
            reference_manifest=manifest,
            output_dir=tmp_path / "output",
            expected_anchor_sha256=wrong_hash,
        )


def test_prior_route_evidence_must_be_a_real_rejection(tmp_path: Path) -> None:
    review = tmp_path / "visual_review.json"
    review.write_text(
        json.dumps(
            {
                "kind": "video2world.object_six_view_visual_review",
                "status": "accepted_for_scene_fit",
                "promotion_allowed": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not a fail-closed visual rejection"):
        audit_prior_rejections([review])
