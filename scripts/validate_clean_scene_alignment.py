#!/usr/bin/env python3
"""Validate strict clean-scene camera, PGSR, and TSDF coordinate lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CAMERA_TOLERANCE = 1e-12
RECEIPT_KIND = "video2world.clean_scene_alignment_receipt"
RECEIPT_STATUS = "passed_current_demo_only"
SEQUENCE_REPORT_KIND = "video2world.layered_clean_plate_sequence_report"
SEQUENCE_REPORT_STATUS = "technical_passed_complete_sequence_with_limitations"
RECONSTRUCTION_CONTRACTS = {
    "video2world.layered_sequence_reconstruction_input": {
        "status": "ready_for_fresh_depth_and_normal_estimation_current_demo_only",
        "source_key": "source_layered_sequence",
        "frame_key": "frame_id",
    },
    "video2world.layered_sequence_reconstruction_input_manifest": {
        "status": "ready_for_fresh_da3_holi_reconstruction",
        "source_key": "sourceSequence",
        "frame_key": "frameId",
    },
}
ROUND_ORDER = (
    "round01_front_pillow",
    "round02_left_pillow",
    "round03_right_pillow",
    "round04_bed",
)
IDENTITY_ROTATION = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
]


class CleanSceneAlignmentError(RuntimeError):
    """Raised when the clean-scene coordinate lineage is not provable."""


@dataclass(frozen=True)
class Camera:
    img_name: str
    width: int
    height: int
    fx: float
    fy: float
    position: tuple[float, float, float]
    rotation: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]

    def canonical(self) -> dict[str, Any]:
        return {
            "img_name": self.img_name,
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "position": list(self.position),
            "rotation": [list(row) for row in self.rotation],
        }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CleanSceneAlignmentError(message)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CleanSceneAlignmentError(f"cannot read JSON {path}: {exc}") from exc


def require_dict(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be an array")
    return value


def require_sha256(value: Any, label: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA-256 digest",
    )
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CleanSceneAlignmentError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


PathMap = tuple[Path, Path]
PathMaps = tuple[PathMap, ...]


def normalize_path_maps(path_maps: PathMaps) -> PathMaps:
    require(isinstance(path_maps, tuple), "path_maps must be a tuple")
    normalized: list[PathMap] = []
    remote_roots: set[Path] = set()
    for index, entry in enumerate(path_maps):
        require(
            isinstance(entry, tuple) and len(entry) == 2,
            f"path_maps[{index}] must be a (REMOTE_ROOT, LOCAL_ROOT) tuple",
        )
        remote_value, local_value = entry
        require(
            isinstance(remote_value, (str, os.PathLike))
            and isinstance(local_value, (str, os.PathLike)),
            f"path_maps[{index}] roots must be paths",
        )
        remote = Path(remote_value).expanduser()
        local = Path(local_value).expanduser()
        require(remote.is_absolute(), f"path_maps[{index}] remote root must be absolute")
        require(local.is_absolute(), f"path_maps[{index}] local root must be absolute")
        remote = Path(os.path.normpath(str(remote)))
        local = local.resolve(strict=False)
        require(remote not in remote_roots, f"duplicate path-map remote root: {remote}")
        require(local.is_dir(), f"path-map local root is not a directory: {local}")
        remote_roots.add(remote)
        normalized.append((remote, local))
    return tuple(normalized)


def parse_path_map(value: str) -> PathMap:
    if "=" not in value:
        raise argparse.ArgumentTypeError("path map must use REMOTE_ROOT=LOCAL_ROOT")
    remote_value, local_value = value.split("=", 1)
    if not remote_value or not local_value:
        raise argparse.ArgumentTypeError("path map roots must not be empty")
    remote = Path(remote_value).expanduser()
    local = Path(local_value).expanduser()
    if not remote.is_absolute() or not local.is_absolute():
        raise argparse.ArgumentTypeError("path map roots must be absolute")
    return remote, local


def resolve_path(
    value: Any,
    *,
    relative_to: Path,
    label: str,
    path_maps: PathMaps = (),
) -> Path:
    require(isinstance(value, str) and value, f"{label}.path is missing")
    path = Path(value).expanduser()
    if not path.is_absolute():
        return (relative_to / path).resolve()

    mappings = normalize_path_maps(path_maps)
    resolved_source = Path(os.path.normpath(str(path)))
    for remote_root, _local_root in mappings:
        if path.is_relative_to(remote_root) and not resolved_source.is_relative_to(remote_root):
            raise CleanSceneAlignmentError(
                f"{label} escapes mapped remote root {remote_root}: {path}"
            )
    matches = [
        (remote_root, local_root)
        for remote_root, local_root in mappings
        if resolved_source.is_relative_to(remote_root)
    ]
    if not matches:
        return resolved_source
    remote_root, local_root = max(matches, key=lambda item: len(item[0].parts))
    mapped = (local_root / resolved_source.relative_to(remote_root)).resolve(strict=False)
    require(
        mapped.is_relative_to(local_root),
        f"{label} mapped path escapes local root {local_root}: {mapped}",
    )
    return mapped


def asset_record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    require(resolved.is_file(), f"input asset is missing: {resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def validate_asset(
    value: Any,
    *,
    relative_to: Path,
    label: str,
    expected_path: Path | None = None,
    path_maps: PathMaps = (),
) -> Path:
    record = require_dict(value, label)
    path = resolve_path(
        record.get("path"),
        relative_to=relative_to,
        label=label,
        path_maps=path_maps,
    )
    expected_sha = require_sha256(record.get("sha256"), f"{label}.sha256")
    require(path.is_file(), f"{label} is missing: {path}")
    require(sha256_file(path) == expected_sha, f"{label} SHA-256 mismatch")
    if expected_path is not None:
        require(path == expected_path.resolve(), f"{label} points to another file")
    return path


def finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    require(
        not isinstance(value, bool) and isinstance(value, (int, float)),
        f"{label} is not numeric",
    )
    result = float(value)
    require(math.isfinite(result), f"{label} is not finite")
    if positive:
        require(result > 0.0, f"{label} must be positive")
    return result


def fixed_vector(value: Any, length: int, label: str) -> tuple[float, ...]:
    values = require_list(value, label)
    require(len(values) == length, f"{label} must contain {length} values")
    return tuple(finite_number(item, f"{label}[{index}]") for index, item in enumerate(values))


def parse_camera(value: Any, *, label: str) -> Camera:
    record = require_dict(value, label)
    img_name = record.get("img_name")
    require(isinstance(img_name, str) and img_name, f"{label}.img_name is invalid")
    width = record.get("width")
    height = record.get("height")
    require(
        not isinstance(width, bool) and isinstance(width, int) and width > 0,
        f"{label}.width is invalid",
    )
    require(
        not isinstance(height, bool) and isinstance(height, int) and height > 0,
        f"{label}.height is invalid",
    )
    position = fixed_vector(record.get("position"), 3, f"{label}.position")
    raw_rotation = require_list(record.get("rotation"), f"{label}.rotation")
    require(len(raw_rotation) == 3, f"{label}.rotation must contain three rows")
    rotation = tuple(
        fixed_vector(row, 3, f"{label}.rotation[{index}]")
        for index, row in enumerate(raw_rotation)
    )
    return Camera(
        img_name=img_name,
        width=width,
        height=height,
        fx=finite_number(record.get("fx"), f"{label}.fx", positive=True),
        fy=finite_number(record.get("fy"), f"{label}.fy", positive=True),
        position=(position[0], position[1], position[2]),
        rotation=(rotation[0], rotation[1], rotation[2]),
    )


def read_cameras(path: Path, *, label: str) -> dict[str, Camera]:
    values = require_list(read_json(path), label)
    require(bool(values), f"{label} must not be empty")
    result: dict[str, Camera] = {}
    for index, value in enumerate(values):
        camera = parse_camera(value, label=f"{label}[{index}]")
        require(camera.img_name not in result, f"{label} has duplicate img_name {camera.img_name}")
        result[camera.img_name] = camera
    return result


def camera_numeric_values(camera: Camera) -> tuple[float, ...]:
    return (
        camera.fx,
        camera.fy,
        *camera.position,
        *(value for row in camera.rotation for value in row),
    )


def compare_camera_subset(
    reference: dict[str, Camera],
    fresh: dict[str, Camera],
) -> tuple[str, float]:
    reference_names = set(reference)
    fresh_names = set(fresh)
    require(fresh_names <= reference_names, "fresh camera img_name set is not a reference subset")
    maximum_delta = 0.0
    for img_name in sorted(fresh_names):
        expected = reference[img_name]
        actual = fresh[img_name]
        require(actual.width == expected.width, f"camera drift for {img_name}.width")
        require(actual.height == expected.height, f"camera drift for {img_name}.height")
        for expected_value, actual_value in zip(
            camera_numeric_values(expected),
            camera_numeric_values(actual),
            strict=True,
        ):
            delta = abs(actual_value - expected_value)
            maximum_delta = max(maximum_delta, delta)
            require(delta <= CAMERA_TOLERANCE, f"camera drift exceeds tolerance for {img_name}")
    canonical_subset = [reference[name].canonical() for name in sorted(fresh_names)]
    return canonical_sha256(canonical_subset), maximum_delta


def validate_sequence_report(path: Path, *, expected_frame_count: int) -> dict[str, Any]:
    report = require_dict(read_json(path), "strict sequence report")
    require(report.get("kind") == SEQUENCE_REPORT_KIND, "strict sequence report kind is invalid")
    require(
        report.get("status") == SEQUENCE_REPORT_STATUS,
        "strict sequence report status is invalid",
    )
    require(report.get("acceptance_scope") == "current_demo_only", "sequence scope is invalid")
    require(report.get("promotion_approved") is False, "sequence promotion must remain false")
    require(report.get("frame_count") == expected_frame_count, "sequence frame count mismatch")
    final_frame_set = require_sha256(
        report.get("final_output_frame_set_sha256"),
        "sequence final_output_frame_set_sha256",
    )
    rounds = require_list(report.get("rounds"), "sequence rounds")
    require(len(rounds) == 4, "strict sequence must contain four rounds")
    for index, raw_round in enumerate(rounds, start=1):
        round_record = require_dict(raw_round, f"sequence round {index}")
        require(round_record.get("round_index") == index, "sequence round order mismatch")
        require(round_record.get("round_name") == ROUND_ORDER[index - 1], "round name mismatch")
        require(round_record.get("unresolved_pixels") == 0, "sequence contains unresolved pixels")
    require(
        rounds[-1].get("output_frame_set_sha256") == final_frame_set,
        "final round does not bind final frame set",
    )
    return report


def reconstruction_source_and_frames(
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], list[str], str]:
    kind = manifest.get("kind")
    require(kind in RECONSTRUCTION_CONTRACTS, "reconstruction input manifest kind is invalid")
    contract = RECONSTRUCTION_CONTRACTS[str(kind)]
    require(manifest.get("status") == contract["status"], "reconstruction input status is invalid")
    source = require_dict(manifest.get(contract["source_key"]), "reconstruction source sequence")
    frames = require_list(manifest.get("frames"), "reconstruction frames")
    frame_ids: list[str] = []
    for index, raw_frame in enumerate(frames):
        frame = require_dict(raw_frame, f"reconstruction frame {index}")
        frame_id = frame.get(contract["frame_key"])
        require(isinstance(frame_id, str) and frame_id, "reconstruction frame ID is invalid")
        require(frame_id not in frame_ids, f"duplicate reconstruction frame ID {frame_id}")
        frame_ids.append(frame_id)
    require(bool(frame_ids), "reconstruction frames must not be empty")
    return source, frame_ids, str(kind)


def validate_reconstruction_manifest(
    path: Path,
    *,
    sequence_report_path: Path,
    sequence_report: dict[str, Any],
    fresh_names: set[str],
    path_maps: PathMaps = (),
) -> dict[str, Any]:
    manifest = require_dict(read_json(path), "reconstruction input manifest")
    source, frame_ids, kind = reconstruction_source_and_frames(manifest)
    report_record = require_dict(source.get("report"), "reconstruction source report")
    validate_asset(
        report_record,
        relative_to=path.parent,
        label="reconstruction source report",
        expected_path=sequence_report_path,
        path_maps=path_maps,
    )
    require(set(frame_ids) == fresh_names, "reconstruction and fresh camera frame sets differ")
    if kind == "video2world.layered_sequence_reconstruction_input":
        require(
            source.get("final_output_frame_set_sha256")
            == sequence_report["final_output_frame_set_sha256"],
            "reconstruction final frame set mismatch",
        )
        validation = require_dict(manifest.get("sequence_validation"), "sequence validation")
        require(bool(validation) and all(validation.values()), "sequence validation gates failed")
        provenance = require_dict(manifest.get("provenance_contract"), "provenance contract")
        require(provenance.get("old_geometry_depth_packaged") is False, "old geometry packaged")
        require(provenance.get("old_composite_depth_packaged") is False, "old depth packaged")
        require(provenance.get("legacy_da3_depth_packaged") is False, "legacy DA3 packaged")
        require(provenance.get("fresh_depth_required") is True, "fresh depth is not required")
    else:
        require(source.get("frameOrder") == frame_ids, "reconstruction frameOrder mismatch")
        require(source.get("roundOrder") == list(ROUND_ORDER), "reconstruction roundOrder mismatch")
    return manifest


def validate_stage_header(receipt: dict[str, Any], *, label: str) -> None:
    kind = receipt.get("kind")
    status = receipt.get("status")
    require(isinstance(kind, str) and kind.startswith("video2world."), f"{label} kind is invalid")
    require(isinstance(status, str) and status, f"{label} status is invalid")
    require(receipt.get("acceptance_scope") == "current_demo_only", f"{label} scope is invalid")
    require(receipt.get("promotion_approved") is False, f"{label} promotion must remain false")


def validate_pgsr_receipt(
    path: Path,
    *,
    sequence_report_path: Path,
    sequence_report: dict[str, Any],
    reconstruction_manifest_path: Path,
    fresh_cameras_path: Path,
    target_coordinate_frame: str,
    path_maps: PathMaps = (),
) -> tuple[dict[str, Any], Path]:
    receipt = require_dict(read_json(path), "PGSR stage receipt")
    validate_stage_header(receipt, label="PGSR stage receipt")
    lineage = require_dict(receipt.get("lineage"), "PGSR lineage")
    validate_asset(
        lineage.get("sequence_report"),
        relative_to=path.parent,
        label="PGSR lineage sequence report",
        expected_path=sequence_report_path,
        path_maps=path_maps,
    )
    validate_asset(
        lineage.get("reconstruction_manifest"),
        relative_to=path.parent,
        label="PGSR lineage reconstruction manifest",
        expected_path=reconstruction_manifest_path,
        path_maps=path_maps,
    )
    require(
        lineage.get("final_output_frame_set_sha256")
        == sequence_report["final_output_frame_set_sha256"],
        "PGSR lineage final frame set mismatch",
    )
    cameras = require_dict(receipt.get("cameras"), "PGSR cameras")
    require(
        cameras.get("output_world_frame") == target_coordinate_frame,
        "PGSR output coordinate frame mismatch",
    )
    outputs = require_dict(receipt.get("outputs"), "PGSR outputs")
    validate_asset(
        outputs.get("cameras_json"),
        relative_to=path.parent,
        label="PGSR cameras.json",
        expected_path=fresh_cameras_path,
        path_maps=path_maps,
    )
    pgsr_record = require_dict(
        outputs.get("point_cloud_iteration_30000"),
        "PGSR point_cloud_iteration_30000",
    )
    pgsr_path = validate_asset(
        pgsr_record,
        relative_to=path.parent,
        label="PGSR point_cloud_iteration_30000",
        path_maps=path_maps,
    )
    require(
        pgsr_record.get("coordinate_frame") == target_coordinate_frame,
        "PGSR PLY coordinate frame mismatch",
    )
    return receipt, pgsr_path


def validate_tsdf_receipt(
    path: Path,
    *,
    pgsr_receipt_path: Path,
    pgsr_ply_path: Path,
    fresh_cameras_path: Path,
    target_coordinate_frame: str,
    path_maps: PathMaps = (),
) -> tuple[dict[str, Any], Path]:
    receipt = require_dict(read_json(path), "TSDF stage receipt")
    validate_stage_header(receipt, label="TSDF stage receipt")
    lineage = require_dict(receipt.get("lineage"), "TSDF lineage")
    validate_asset(
        lineage.get("pgsr_stage_receipt"),
        relative_to=path.parent,
        label="TSDF lineage PGSR stage receipt",
        expected_path=pgsr_receipt_path,
        path_maps=path_maps,
    )
    inputs = require_dict(receipt.get("input"), "TSDF input")
    pgsr_record = require_dict(
        inputs.get("point_cloud_iteration_30000"),
        "TSDF input point_cloud_iteration_30000",
    )
    validate_asset(
        pgsr_record,
        relative_to=path.parent,
        label="TSDF input PGSR PLY",
        expected_path=pgsr_ply_path,
        path_maps=path_maps,
    )
    require(
        pgsr_record.get("coordinate_frame") == target_coordinate_frame,
        "TSDF input PGSR coordinate frame mismatch",
    )
    validate_asset(
        inputs.get("cameras_json"),
        relative_to=path.parent,
        label="TSDF input cameras.json",
        expected_path=fresh_cameras_path,
        path_maps=path_maps,
    )
    outputs = require_dict(receipt.get("outputs"), "TSDF outputs")
    tsdf_record = require_dict(outputs.get("tsdf_fusion_post"), "TSDF output tsdf_fusion_post")
    tsdf_path = validate_asset(
        tsdf_record,
        relative_to=path.parent,
        label="TSDF output tsdf_fusion_post",
        path_maps=path_maps,
    )
    require(
        tsdf_record.get("coordinate_frame") == target_coordinate_frame,
        "TSDF output coordinate frame mismatch",
    )
    return receipt, tsdf_path


def validate_clean_scene_alignment(
    *,
    reference_cameras_path: Path,
    fresh_cameras_path: Path,
    strict_sequence_report_path: Path,
    reconstruction_input_manifest_path: Path,
    pgsr_stage_receipt_path: Path,
    tsdf_stage_receipt_path: Path,
    source_coordinate_frame: str,
    target_coordinate_frame: str,
    path_maps: PathMaps = (),
) -> dict[str, Any]:
    normalized_path_maps = normalize_path_maps(path_maps)
    paths = (
        reference_cameras_path,
        fresh_cameras_path,
        strict_sequence_report_path,
        reconstruction_input_manifest_path,
        pgsr_stage_receipt_path,
        tsdf_stage_receipt_path,
    )
    for path in paths:
        require(path.resolve().is_file(), f"required input is missing: {path}")
    require(bool(source_coordinate_frame.strip()), "source coordinate frame is empty")
    require(bool(target_coordinate_frame.strip()), "target coordinate frame is empty")
    stage_receipt_hashes_before = {
        "pgsr": sha256_file(pgsr_stage_receipt_path),
        "tsdf": sha256_file(tsdf_stage_receipt_path),
    }

    reference = read_cameras(reference_cameras_path, label="reference cameras")
    fresh = read_cameras(fresh_cameras_path, label="fresh cameras")
    canonical_subset_sha, maximum_delta = compare_camera_subset(reference, fresh)
    sequence_report = validate_sequence_report(
        strict_sequence_report_path,
        expected_frame_count=len(fresh),
    )
    validate_reconstruction_manifest(
        reconstruction_input_manifest_path,
        sequence_report_path=strict_sequence_report_path,
        sequence_report=sequence_report,
        fresh_names=set(fresh),
        path_maps=normalized_path_maps,
    )
    _pgsr_receipt, pgsr_ply_path = validate_pgsr_receipt(
        pgsr_stage_receipt_path,
        sequence_report_path=strict_sequence_report_path,
        sequence_report=sequence_report,
        reconstruction_manifest_path=reconstruction_input_manifest_path,
        fresh_cameras_path=fresh_cameras_path,
        target_coordinate_frame=target_coordinate_frame,
        path_maps=normalized_path_maps,
    )
    _tsdf_receipt, tsdf_ply_path = validate_tsdf_receipt(
        tsdf_stage_receipt_path,
        pgsr_receipt_path=pgsr_stage_receipt_path,
        pgsr_ply_path=pgsr_ply_path,
        fresh_cameras_path=fresh_cameras_path,
        target_coordinate_frame=target_coordinate_frame,
        path_maps=normalized_path_maps,
    )
    require(
        sha256_file(pgsr_stage_receipt_path) == stage_receipt_hashes_before["pgsr"],
        "PGSR source stage receipt changed during validation",
    )
    require(
        sha256_file(tsdf_stage_receipt_path) == stage_receipt_hashes_before["tsdf"],
        "TSDF source stage receipt changed during validation",
    )

    return {
        "schema_version": 1,
        "kind": RECEIPT_KIND,
        "status": RECEIPT_STATUS,
        "created_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 - mil8 uses 3.10
        "acceptance_scope": "current_demo_only",
        "source_coordinate_frame": source_coordinate_frame,
        "target_coordinate_frame": target_coordinate_frame,
        "method": "identity_same_camera_calibration",
        "path_maps": [
            {
                "remote_root": str(remote_root),
                "local_root": str(local_root),
                "resolution": "longest_absolute_prefix",
            }
            for remote_root, local_root in normalized_path_maps
        ],
        "transform": {
            "scale": 1.0,
            "rotationMatrix": IDENTITY_ROTATION,
            "translation": [0.0, 0.0, 0.0],
        },
        "canonical_camera_subset_sha256": canonical_subset_sha,
        "camera_comparison": {
            "reference_camera_count": len(reference),
            "fresh_camera_count": len(fresh),
            "fresh_img_names": sorted(fresh),
            "absolute_tolerance": CAMERA_TOLERANCE,
            "maximum_absolute_delta": maximum_delta,
        },
        "inputs": {
            "strict_sequence_report": asset_record(strict_sequence_report_path),
            "reconstruction_input_manifest": asset_record(reconstruction_input_manifest_path),
            "reference_cameras": asset_record(reference_cameras_path),
            "fresh_cameras": asset_record(fresh_cameras_path),
            "pgsr_ply": asset_record(pgsr_ply_path),
            "tsdf_ply": asset_record(tsdf_ply_path),
            "pgsr_stage_receipt": asset_record(pgsr_stage_receipt_path),
            "tsdf_stage_receipt": asset_record(tsdf_stage_receipt_path),
        },
        "gates": {
            "camera_calibration_hash_bound": True,
            "reference_subset_exact": True,
            "pgsr_and_tsdf_share_frame": True,
            "object_placements_share_target_frame": True,
            "transform_finite": True,
            "identity_or_verified_similarity": True,
            "source_stage_receipts_unchanged": True,
        },
        "promotion_approved": False,
    }


def write_json(path: Path, value: Any) -> None:
    require(not path.is_symlink(), f"output cannot be a symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-cameras", type=Path, required=True)
    parser.add_argument("--fresh-cameras", type=Path, required=True)
    parser.add_argument("--strict-sequence-report", type=Path, required=True)
    parser.add_argument("--reconstruction-input-manifest", type=Path, required=True)
    parser.add_argument("--pgsr-stage-receipt", type=Path, required=True)
    parser.add_argument("--tsdf-stage-receipt", type=Path, required=True)
    parser.add_argument("--source-coordinate-frame", required=True)
    parser.add_argument("--target-coordinate-frame", required=True)
    parser.add_argument(
        "--path-map",
        action="append",
        default=[],
        type=parse_path_map,
        metavar="REMOTE_ROOT=LOCAL_ROOT",
        help=(
            "Map absolute paths recorded on a remote host to a local mirror. "
            "Repeat for multiple roots; the longest matching remote prefix wins."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    receipt = validate_clean_scene_alignment(
        reference_cameras_path=args.reference_cameras,
        fresh_cameras_path=args.fresh_cameras,
        strict_sequence_report_path=args.strict_sequence_report,
        reconstruction_input_manifest_path=args.reconstruction_input_manifest,
        pgsr_stage_receipt_path=args.pgsr_stage_receipt,
        tsdf_stage_receipt_path=args.tsdf_stage_receipt,
        source_coordinate_frame=args.source_coordinate_frame,
        target_coordinate_frame=args.target_coordinate_frame,
        path_maps=tuple(args.path_map),
    )
    write_json(args.output, receipt)
    print(json.dumps({"status": receipt["status"], "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
