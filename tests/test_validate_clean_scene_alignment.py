from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from scripts.validate_clean_scene_alignment import (
    CAMERA_TOLERANCE,
    IDENTITY_ROTATION,
    RECEIPT_KIND,
    RECEIPT_STATUS,
    ROUND_ORDER,
    CleanSceneAlignmentError,
    build_parser,
    canonical_sha256,
    resolve_path,
    sha256_file,
    validate_clean_scene_alignment,
)
from scripts.validate_clean_scene_alignment import (
    write_json as write_receipt,
)

SOURCE_FRAME = "bedroom4_reference_pgsr_world"
TARGET_FRAME = "source_layered_sequence_camera_world"
FRESH_NAMES = ("000002", "000001")


@dataclass(frozen=True)
class Fixture:
    reference_cameras: Path
    fresh_cameras: Path
    sequence_report: Path
    reconstruction_manifest: Path
    pgsr_receipt: Path
    tsdf_receipt: Path
    pgsr_ply: Path
    tsdf_ply: Path


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def asset(path: Path, *, relative_to: Path) -> dict[str, str]:
    return {
        "path": os.path.relpath(path, relative_to),
        "sha256": sha256_file(path),
    }


def fake_sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def camera(img_name: str, offset: float) -> dict[str, Any]:
    return {
        "id": int(img_name),
        "img_name": img_name,
        "width": 1280,
        "height": 720,
        "fx": 639.9410000000001,
        "fy": 680.253,
        "position": [offset, 1.0 + offset, -6.0 + offset],
        "rotation": [
            [1.0, 0.0, offset * 0.01],
            [0.0, 1.0, 0.0],
            [-offset * 0.01, 0.0, 1.0],
        ],
    }


def make_fixture(tmp_path: Path) -> Fixture:
    reference_cameras = tmp_path / "reference/cameras.json"
    fresh_cameras = tmp_path / "fresh/cameras.json"
    sequence_report = tmp_path / "sequence/layered_clean_plate_sequence_report.json"
    reconstruction_manifest = tmp_path / "reconstruction/manifest.json"
    pgsr_receipt = tmp_path / "receipts/pgsr_stage_receipt.json"
    tsdf_receipt = tmp_path / "receipts/tsdf_stage_receipt.json"
    pgsr_ply = tmp_path / "pgsr/point_cloud/iteration_30000/point_cloud.ply"
    tsdf_ply = tmp_path / "pgsr/mesh/tsdf_fusion_post.ply"

    reference = [
        camera("000000", 0.0),
        camera("000001", 0.1),
        camera("000002", 0.2),
    ]
    by_name = {item["img_name"]: item for item in reference}
    fresh = [copy.deepcopy(by_name[name]) for name in FRESH_NAMES]
    write_json(reference_cameras, reference)
    write_json(fresh_cameras, fresh)

    final_frame_set = fake_sha("strict-final-frame-set")
    write_json(
        sequence_report,
        {
            "schema_version": 1,
            "kind": "video2world.layered_clean_plate_sequence_report",
            "status": "technical_passed_complete_sequence_with_limitations",
            "acceptance_scope": "current_demo_only",
            "promotion_approved": False,
            "frame_count": len(FRESH_NAMES),
            "final_output_frame_set_sha256": final_frame_set,
            "rounds": [
                {
                    "round_index": index,
                    "round_name": round_name,
                    "unresolved_pixels": 0,
                    "output_frame_set_sha256": (
                        final_frame_set if index == 4 else fake_sha(f"round-{index}")
                    ),
                }
                for index, round_name in enumerate(ROUND_ORDER, start=1)
            ],
        },
    )
    write_json(
        reconstruction_manifest,
        {
            "schema_version": 1,
            "kind": "video2world.layered_sequence_reconstruction_input_manifest",
            "status": "ready_for_fresh_da3_holi_reconstruction",
            "sceneId": "bedroom4-test",
            "sourceSequence": {
                "report": asset(sequence_report, relative_to=reconstruction_manifest.parent),
                "status": "technical_passed_complete_sequence_with_limitations",
                "frameOrder": list(FRESH_NAMES),
                "roundOrder": list(ROUND_ORDER),
            },
            "frames": [
                {
                    "frameId": frame_name,
                    "rgb": {
                        "path": f"dslr/resized_undistorted_images/{frame_name}.png",
                        "sha256": fake_sha(f"rgb-{frame_name}"),
                    },
                }
                for frame_name in FRESH_NAMES
            ],
            "limitations": ["synthetic fixture"],
        },
    )

    pgsr_ply.parent.mkdir(parents=True, exist_ok=True)
    pgsr_ply.write_bytes(b"ply\nsynthetic-pgsr\n")
    tsdf_ply.parent.mkdir(parents=True, exist_ok=True)
    tsdf_ply.write_bytes(b"ply\nsynthetic-tsdf\n")
    write_json(
        pgsr_receipt,
        {
            "schema_version": 1,
            "kind": "video2world.clean_scene_pgsr_stage_receipt",
            "status": "technical_passed_current_demo_only",
            "acceptance_scope": "current_demo_only",
            "promotion_approved": False,
            "lineage": {
                "sequence_report": asset(sequence_report, relative_to=pgsr_receipt.parent),
                "reconstruction_manifest": asset(
                    reconstruction_manifest,
                    relative_to=pgsr_receipt.parent,
                ),
                "final_output_frame_set_sha256": final_frame_set,
            },
            "cameras": {
                "camera_count": len(FRESH_NAMES),
                "output_world_frame": TARGET_FRAME,
            },
            "outputs": {
                "cameras_json": asset(fresh_cameras, relative_to=pgsr_receipt.parent),
                "point_cloud_iteration_30000": {
                    **asset(pgsr_ply, relative_to=pgsr_receipt.parent),
                    "coordinate_frame": TARGET_FRAME,
                },
            },
        },
    )
    write_json(
        tsdf_receipt,
        {
            "schema_version": 1,
            "kind": "video2world.clean_scene_tsdf_stage_receipt",
            "status": "technical_passed_current_demo_only",
            "acceptance_scope": "current_demo_only",
            "promotion_approved": False,
            "lineage": {
                "pgsr_stage_receipt": asset(pgsr_receipt, relative_to=tsdf_receipt.parent),
            },
            "input": {
                "cameras_json": asset(fresh_cameras, relative_to=tsdf_receipt.parent),
                "point_cloud_iteration_30000": {
                    **asset(pgsr_ply, relative_to=tsdf_receipt.parent),
                    "coordinate_frame": TARGET_FRAME,
                },
            },
            "outputs": {
                "tsdf_fusion_post": {
                    **asset(tsdf_ply, relative_to=tsdf_receipt.parent),
                    "coordinate_frame": TARGET_FRAME,
                },
            },
        },
    )
    return Fixture(
        reference_cameras=reference_cameras,
        fresh_cameras=fresh_cameras,
        sequence_report=sequence_report,
        reconstruction_manifest=reconstruction_manifest,
        pgsr_receipt=pgsr_receipt,
        tsdf_receipt=tsdf_receipt,
        pgsr_ply=pgsr_ply,
        tsdf_ply=tsdf_ply,
    )


def validate(
    fixture: Fixture,
    *,
    path_maps: tuple[tuple[Path, Path], ...] = (),
) -> dict[str, Any]:
    return validate_clean_scene_alignment(
        reference_cameras_path=fixture.reference_cameras,
        fresh_cameras_path=fixture.fresh_cameras,
        strict_sequence_report_path=fixture.sequence_report,
        reconstruction_input_manifest_path=fixture.reconstruction_manifest,
        pgsr_stage_receipt_path=fixture.pgsr_receipt,
        tsdf_stage_receipt_path=fixture.tsdf_receipt,
        source_coordinate_frame=SOURCE_FRAME,
        target_coordinate_frame=TARGET_FRAME,
        path_maps=path_maps,
    )


def rewrite_recorded_paths_to_remote(
    value: Any,
    *,
    document_path: Path,
    local_root: Path,
    remote_root: Path,
) -> None:
    if isinstance(value, dict):
        recorded_path = value.get("path")
        if isinstance(recorded_path, str):
            local_path = Path(recorded_path)
            if not local_path.is_absolute():
                local_path = document_path.parent / local_path
            local_path = local_path.resolve()
            value["path"] = str(remote_root / local_path.relative_to(local_root.resolve()))
        for item in value.values():
            rewrite_recorded_paths_to_remote(
                item,
                document_path=document_path,
                local_root=local_root,
                remote_root=remote_root,
            )
    elif isinstance(value, list):
        for item in value:
            rewrite_recorded_paths_to_remote(
                item,
                document_path=document_path,
                local_root=local_root,
                remote_root=remote_root,
            )


def make_remote_receipt_fixture(fixture: Fixture, *, local_root: Path, remote_root: Path) -> None:
    reconstruction = read_json(fixture.reconstruction_manifest)
    rewrite_recorded_paths_to_remote(
        reconstruction,
        document_path=fixture.reconstruction_manifest,
        local_root=local_root,
        remote_root=remote_root,
    )
    write_json(fixture.reconstruction_manifest, reconstruction)

    pgsr = read_json(fixture.pgsr_receipt)
    rewrite_recorded_paths_to_remote(
        pgsr,
        document_path=fixture.pgsr_receipt,
        local_root=local_root,
        remote_root=remote_root,
    )
    pgsr["lineage"]["reconstruction_manifest"]["sha256"] = sha256_file(
        fixture.reconstruction_manifest
    )
    write_json(fixture.pgsr_receipt, pgsr)

    tsdf = read_json(fixture.tsdf_receipt)
    rewrite_recorded_paths_to_remote(
        tsdf,
        document_path=fixture.tsdf_receipt,
        local_root=local_root,
        remote_root=remote_root,
    )
    tsdf["lineage"]["pgsr_stage_receipt"]["sha256"] = sha256_file(fixture.pgsr_receipt)
    write_json(fixture.tsdf_receipt, tsdf)


def rebind_camera_and_pgsr_receipt(fixture: Fixture) -> None:
    pgsr = read_json(fixture.pgsr_receipt)
    pgsr["outputs"]["cameras_json"]["sha256"] = sha256_file(fixture.fresh_cameras)
    write_json(fixture.pgsr_receipt, pgsr)
    tsdf = read_json(fixture.tsdf_receipt)
    tsdf["input"]["cameras_json"]["sha256"] = sha256_file(fixture.fresh_cameras)
    tsdf["lineage"]["pgsr_stage_receipt"]["sha256"] = sha256_file(fixture.pgsr_receipt)
    write_json(fixture.tsdf_receipt, tsdf)


def test_writes_identity_alignment_receipt_with_all_inputs_hash_bound(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    receipt = validate(fixture)

    assert receipt["kind"] == RECEIPT_KIND
    assert receipt["status"] == RECEIPT_STATUS
    assert receipt["source_coordinate_frame"] == SOURCE_FRAME
    assert receipt["target_coordinate_frame"] == TARGET_FRAME
    assert receipt["method"] == "identity_same_camera_calibration"
    assert receipt["transform"] == {
        "scale": 1.0,
        "rotationMatrix": IDENTITY_ROTATION,
        "translation": [0.0, 0.0, 0.0],
    }
    assert receipt["promotion_approved"] is False
    assert all(receipt["gates"].values())
    assert receipt["camera_comparison"] == {
        "reference_camera_count": 3,
        "fresh_camera_count": 2,
        "fresh_img_names": sorted(FRESH_NAMES),
        "absolute_tolerance": CAMERA_TOLERANCE,
        "maximum_absolute_delta": 0.0,
    }
    reference = {item["img_name"]: item for item in read_json(fixture.reference_cameras)}
    canonical_subset = [reference[name] for name in sorted(FRESH_NAMES)]
    for item in canonical_subset:
        item.pop("id")
    assert receipt["canonical_camera_subset_sha256"] == canonical_sha256(canonical_subset)

    expected_inputs = {
        "strict_sequence_report": fixture.sequence_report,
        "reconstruction_input_manifest": fixture.reconstruction_manifest,
        "reference_cameras": fixture.reference_cameras,
        "fresh_cameras": fixture.fresh_cameras,
        "pgsr_ply": fixture.pgsr_ply,
        "tsdf_ply": fixture.tsdf_ply,
        "pgsr_stage_receipt": fixture.pgsr_receipt,
        "tsdf_stage_receipt": fixture.tsdf_receipt,
    }
    assert set(receipt["inputs"]) == set(expected_inputs)
    for key, path in expected_inputs.items():
        assert receipt["inputs"][key] == {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }

    output = tmp_path / "alignment/receipt.json"
    write_receipt(output, receipt)
    assert read_json(output) == receipt


def test_accepts_camera_delta_at_fixed_tolerance(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    fresh = read_json(fixture.fresh_cameras)
    fresh[0]["position"][0] += CAMERA_TOLERANCE / 2.0
    write_json(fixture.fresh_cameras, fresh)
    rebind_camera_and_pgsr_receipt(fixture)

    receipt = validate(fixture)

    assert 0.0 < receipt["camera_comparison"]["maximum_absolute_delta"] <= CAMERA_TOLERANCE


def test_rejects_camera_calibration_drift(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    fresh = read_json(fixture.fresh_cameras)
    fresh[0]["fx"] += CAMERA_TOLERANCE * 10.0
    write_json(fixture.fresh_cameras, fresh)
    rebind_camera_and_pgsr_receipt(fixture)

    with pytest.raises(CleanSceneAlignmentError, match="camera drift exceeds tolerance"):
        validate(fixture)


def test_rejects_tampered_tsdf_pgsr_receipt_lineage(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    tsdf = read_json(fixture.tsdf_receipt)
    tsdf["lineage"]["pgsr_stage_receipt"]["sha256"] = fake_sha("tampered-pgsr-receipt")
    write_json(fixture.tsdf_receipt, tsdf)

    with pytest.raises(
        CleanSceneAlignmentError,
        match="TSDF lineage PGSR stage receipt SHA-256 mismatch",
    ):
        validate(fixture)


def test_rejects_tampered_tsdf_output_hash(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    tsdf = read_json(fixture.tsdf_receipt)
    tsdf["outputs"]["tsdf_fusion_post"]["sha256"] = fake_sha("tampered-tsdf")
    write_json(fixture.tsdf_receipt, tsdf)

    with pytest.raises(
        CleanSceneAlignmentError,
        match="TSDF output tsdf_fusion_post SHA-256 mismatch",
    ):
        validate(fixture)


def test_remote_absolute_receipts_require_map_then_validate_local_mirror(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    remote_root = Path("/mil8/video2world/runs/bedroom4-clean")
    make_remote_receipt_fixture(
        fixture,
        local_root=tmp_path,
        remote_root=remote_root,
    )
    pgsr_before = fixture.pgsr_receipt.read_bytes()
    tsdf_before = fixture.tsdf_receipt.read_bytes()

    with pytest.raises(CleanSceneAlignmentError, match="is missing"):
        validate(fixture)

    receipt = validate(fixture, path_maps=((remote_root, tmp_path),))

    assert receipt["path_maps"] == [
        {
            "remote_root": str(remote_root),
            "local_root": str(tmp_path.resolve()),
            "resolution": "longest_absolute_prefix",
        }
    ]
    assert len(receipt["inputs"]) == 8
    for record in receipt["inputs"].values():
        local_path = Path(record["path"])
        assert local_path.is_relative_to(tmp_path.resolve())
        assert local_path.is_file()
        assert record["sha256"] == sha256_file(local_path)
    assert fixture.pgsr_receipt.read_bytes() == pgsr_before
    assert fixture.tsdf_receipt.read_bytes() == tsdf_before
    assert receipt["gates"]["source_stage_receipts_unchanged"] is True


def test_absolute_path_map_uses_longest_prefix(tmp_path: Path) -> None:
    broad_mirror = tmp_path / "broad"
    narrow_mirror = tmp_path / "narrow"
    broad_mirror.mkdir()
    narrow_mirror.mkdir()

    resolved = resolve_path(
        "/mil8/jobs/bedroom4/output.ply",
        relative_to=tmp_path,
        label="fixture asset",
        path_maps=(
            (Path("/mil8"), broad_mirror),
            (Path("/mil8/jobs"), narrow_mirror),
        ),
    )

    assert resolved == (narrow_mirror / "bedroom4/output.ply").resolve()


def test_path_maps_reject_escape_duplicate_and_relative_roots(tmp_path: Path) -> None:
    mirror = tmp_path / "mirror"
    outside = tmp_path / "outside"
    mirror.mkdir()
    outside.mkdir()
    (mirror / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(CleanSceneAlignmentError, match="escapes local root"):
        resolve_path(
            "/mil8/run/escape/asset.ply",
            relative_to=tmp_path,
            label="fixture asset",
            path_maps=((Path("/mil8/run"), mirror),),
        )
    with pytest.raises(CleanSceneAlignmentError, match="escapes mapped remote root"):
        resolve_path(
            "/mil8/run/../outside/asset.ply",
            relative_to=tmp_path,
            label="fixture asset",
            path_maps=((Path("/mil8/run"), mirror),),
        )
    with pytest.raises(CleanSceneAlignmentError, match="duplicate path-map remote root"):
        resolve_path(
            "/mil8/run/asset.ply",
            relative_to=tmp_path,
            label="fixture asset",
            path_maps=(
                (Path("/mil8/run"), mirror),
                (Path("/mil8/run"), outside),
            ),
        )
    with pytest.raises(CleanSceneAlignmentError, match="remote root must be absolute"):
        resolve_path(
            "/mil8/run/asset.ply",
            relative_to=tmp_path,
            label="fixture asset",
            path_maps=((Path("relative/remote"), mirror),),
        )
    with pytest.raises(CleanSceneAlignmentError, match="local root must be absolute"):
        resolve_path(
            "/mil8/run/asset.ply",
            relative_to=tmp_path,
            label="fixture asset",
            path_maps=((Path("/mil8/run"), Path("relative/local")),),
        )


def test_cli_accepts_repeatable_absolute_path_maps_and_rejects_relative_map() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--reference-cameras", "/local/reference.json",
            "--fresh-cameras", "/local/fresh.json",
            "--strict-sequence-report", "/local/sequence.json",
            "--reconstruction-input-manifest", "/local/reconstruction.json",
            "--pgsr-stage-receipt", "/local/pgsr.json",
            "--tsdf-stage-receipt", "/local/tsdf.json",
            "--source-coordinate-frame", SOURCE_FRAME,
            "--target-coordinate-frame", TARGET_FRAME,
            "--path-map", "/mil8/run=/local/run",
            "--path-map", "/mil8/shared=/local/shared",
            "--output", "/local/output.json",
        ]
    )
    assert args.path_map == [
        (Path("/mil8/run"), Path("/local/run")),
        (Path("/mil8/shared"), Path("/local/shared")),
    ]

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--reference-cameras", "/local/reference.json",
                "--fresh-cameras", "/local/fresh.json",
                "--strict-sequence-report", "/local/sequence.json",
                "--reconstruction-input-manifest", "/local/reconstruction.json",
                "--pgsr-stage-receipt", "/local/pgsr.json",
                "--tsdf-stage-receipt", "/local/tsdf.json",
                "--source-coordinate-frame", SOURCE_FRAME,
                "--target-coordinate-frame", TARGET_FRAME,
                "--path-map", "relative=/local/run",
                "--output", "/local/output.json",
            ]
        )
