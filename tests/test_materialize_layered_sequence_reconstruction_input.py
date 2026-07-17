from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from scripts.compose_layered_clean_plate import INPUT_KIND, RECEIPT_KIND, REPORT_KIND
from scripts.compose_layered_clean_plate_sequence import (
    OBJECT_ORDER,
    ROUND_NAMES,
    ROUND_REMAINING,
    SEQUENCE_RECEIPT_KIND,
    SEQUENCE_REPORT_KIND,
)
from scripts.materialize_layered_sequence_reconstruction_input import (
    EXPECTED_SEQUENCE_STATUS,
    REQUIRED_FALSE_ROUND_GATES,
    REQUIRED_SEQUENCE_GATES,
    REQUIRED_TRUE_ROUND_GATES,
    digest_json,
    materialize_package,
    sha256_file,
)

FRAME_IDS = ("000048", "000049")
WIDTH = 4
HEIGHT = 3


@dataclass(frozen=True)
class SequenceFixture:
    report: Path
    receipt: Path
    camera_info: Path
    final_frames: dict[str, Path]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def asset(path: Path, *, relative_to: Path) -> dict[str, str]:
    return {"path": os.path.relpath(path, relative_to), "sha256": sha256_file(path)}


def fake_sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def save_rgb(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((HEIGHT, WIDTH, 3), color, dtype=np.uint8)).save(path)


def make_camera(path: Path) -> None:
    intrinsic = {
        "camera_id": "1",
        "model": "PINHOLE",
        "w": WIDTH,
        "h": HEIGHT,
        "fx": 5.0,
        "fy": 5.5,
        "cx": 2.0,
        "cy": 1.5,
        "params": [5.0, 5.5, 2.0, 1.5],
    }
    write_json(
        path,
        {
            "schema_version": 1,
            "source": "synthetic_test_fixture",
            "extrinsic_type": "world_to_camera",
            "intrinsic": intrinsic,
            "intrinsics": {"1": intrinsic},
            "extrinsic": {
                "000048": np.eye(4).tolist(),
                "000049": [
                    [1.0, 0.0, 0.0, 0.1],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            },
            "frame_camera_ids": {frame_id: "1" for frame_id in FRAME_IDS},
            "images": {
                frame_id: {
                    "image_id": str(index + 1),
                    "camera_id": "1",
                    "name": f"{frame_id}.png",
                }
                for index, frame_id in enumerate(FRAME_IDS)
            },
        },
    )


def make_fixture(tmp_path: Path) -> SequenceFixture:
    root = tmp_path / "sequence"
    camera_info = tmp_path / "camera_info.json"
    make_camera(camera_info)
    camera_sha = sha256_file(camera_info)
    measured_reports: dict[int, Path] = {}
    for round_index in range(1, 4):
        path = root / "normalized_inputs" / f"round{round_index:02d}_measured_report.json"
        write_json(
            path,
            {
                "schema_version": 2,
                "status": "technical_passed",
                "round_index": round_index,
                "camera_info_sha256": camera_sha,
            },
        )
        measured_reports[round_index] = path

    initial_sources: dict[str, Path] = {}
    for frame_index, frame_id in enumerate(FRAME_IDS):
        path = root / "source" / f"{frame_id}.png"
        save_rgb(path, (20 + frame_index, 30, 40))
        initial_sources[frame_id] = path

    round_summaries: list[dict[str, Any]] = []
    previous_report_path: Path | None = None
    previous_receipt_path: Path | None = None
    previous_report: dict[str, Any] | None = None
    previous_outputs = initial_sources
    final_frames: dict[str, Path] = {}
    for round_index, round_name in enumerate(ROUND_NAMES, start=1):
        round_root = root / round_name
        manifest_path = round_root / "compositor_input_manifest.json"
        report_path = round_root / "composite/layered_composite_report.json"
        receipt_path = round_root / "composite/layered_composite_receipt.json"
        remaining = ROUND_REMAINING[round_name]
        removed = list(OBJECT_ORDER[:round_index])
        manifest_records = []
        report_records = []
        current_outputs: dict[str, Path] = {}
        output_set = []
        for sequence_index, frame_id in enumerate(FRAME_IDS):
            source_path = previous_outputs[frame_id]
            output_path = report_path.parent / "frames" / f"{sequence_index:04d}.png"
            save_rgb(
                output_path,
                (40 * round_index + sequence_index, 60 + round_index, 80),
            )
            current_outputs[frame_id] = output_path
            rgb_sha = sha256_file(output_path)
            depth_sha = fake_sha(f"depth-{round_index}-{frame_id}")
            labels_sha = fake_sha(f"labels-{round_index}-{frame_id}")
            manifest_records.append(
                {
                    "sequence_index": sequence_index,
                    "frame_id": frame_id,
                    "source_rgb": asset(source_path, relative_to=manifest_path.parent),
                    "downstream_layers": [{"layer_id": layer_id} for layer_id in remaining],
                }
            )
            report_records.append(
                {
                    "sequence_index": sequence_index,
                    "frame_id": frame_id,
                    "inputs": {"source_rgb_sha256": sha256_file(source_path)},
                    "outputs": {
                        "composite_rgb": asset(output_path, relative_to=report_path.parent),
                        "composite_depth": {
                            "path": f"depth/{sequence_index:04d}.npy",
                            "sha256": depth_sha,
                        },
                        "provenance_labels": {
                            "path": f"provenance_labels/{sequence_index:04d}.png",
                            "sha256": labels_sha,
                        },
                    },
                }
            )
            output_set.append(
                {
                    "frame_id": frame_id,
                    "composite_rgb_sha256": rgb_sha,
                    "composite_depth_sha256": depth_sha,
                    "provenance_labels_sha256": labels_sha,
                }
            )
        previous_binding = None
        if previous_report_path is not None and previous_receipt_path is not None:
            previous_binding = {
                "round_index": round_index - 1,
                "report": asset(previous_report_path, relative_to=manifest_path.parent),
                "receipt": asset(previous_receipt_path, relative_to=manifest_path.parent),
            }
        manifest = {
            "schema_version": 1,
            "kind": INPUT_KIND,
            "round_index": round_index,
            "round_kind": "final_background" if round_index == 4 else "object_peel",
            "removed_object_ids": removed,
            "newly_removed_object_id": removed[-1],
            "remaining_object_ids": remaining,
            "source_contract": {
                "role": "original_observed_rgb"
                if round_index == 1
                else "previous_layered_composite",
                "untracked_generated_rgb": False,
                "propainter_rgb": False,
            },
            "previous_round": previous_binding,
            "measured_donor_contract": (
                {
                    "role": "no_measured_donor_evidence",
                    "report": None,
                    "receipt": None,
                    "claims_original_observed_rgb_donor": False,
                    "measured_pixels": 0,
                    "generated_pixels": 0,
                    "propainter_pixels": 0,
                }
                if round_index == 4
                else {"role": "calibrated_rgbd_donor_prefill"}
            ),
            "frame_records": manifest_records,
        }
        write_json(manifest_path, manifest)
        output_set_sha = digest_json(output_set)
        previous_round_binding = None
        if previous_report_path is not None and previous_receipt_path is not None:
            assert previous_report is not None
            previous_round_binding = {
                "round_index": round_index - 1,
                "report_path": str(previous_report_path),
                "report_sha256": sha256_file(previous_report_path),
                "receipt_path": str(previous_receipt_path),
                "receipt_sha256": sha256_file(previous_receipt_path),
                "output_frame_set_sha256": previous_report["output_frame_set_sha256"],
            }
        limited_metadata = (
            {
                "accepted_with_limitations": True,
                "acceptance_status": "accepted_with_limitations",
                "acceptance_scope": "current_demo_only",
                "demo_use_approved": True,
                "eligible_as_round04_clean_plate": False,
                "eligible_as_current_demo_round04_clean_plate": True,
                "limitations": ["Synthetic fixture: current demo only."],
                "overridden_gates": {"visual_quality": {"passed": False}},
                "failed_metrics": {"visual_quality": {"passed": False}},
            }
            if round_index == 4
            else {
                "accepted_with_limitations": False,
                "acceptance_status": None,
                "acceptance_scope": None,
                "demo_use_approved": None,
                "eligible_as_round04_clean_plate": None,
                "eligible_as_current_demo_round04_clean_plate": None,
                "limitations": [],
                "overridden_gates": {},
                "failed_metrics": {},
            }
        )
        report = {
            "schema_version": 1,
            "kind": REPORT_KIND,
            "status": (
                "technical_passed_complete_partition_with_limitations"
                if round_index == 4
                else "technical_passed_complete_partition"
            ),
            "round_index": round_index,
            "round_kind": "final_background" if round_index == 4 else "object_peel",
            "removed_object_ids": removed,
            "remaining_object_ids": remaining,
            "input_manifest": str(manifest_path),
            "input_manifest_sha256": sha256_file(manifest_path),
            "measured_donor_report": (
                None
                if round_index == 4
                else {
                    **asset(measured_reports[round_index], relative_to=report_path.parent),
                    "generated_pixels": 0,
                    "propainter_pixels": 0,
                }
            ),
            "previous_round_binding": previous_round_binding,
            "frame_records": report_records,
            "aggregate_counts": {"unresolved_pixels": 0},
            "output_frame_set_sha256": output_set_sha,
            "gates": {
                **{gate: True for gate in REQUIRED_TRUE_ROUND_GATES},
                **{gate: False for gate in REQUIRED_FALSE_ROUND_GATES},
            },
            "promotion_approved": False,
            **limited_metadata,
        }
        write_json(report_path, report)
        receipt = {
            "schema_version": 1,
            "kind": RECEIPT_KIND,
            "round_index": round_index,
            "round_kind": report["round_kind"],
            "report": report_path.name,
            "report_sha256": sha256_file(report_path),
            "input_manifest_sha256": sha256_file(manifest_path),
            "output_frame_set_sha256": output_set_sha,
            "previous_round_output_report_sha256": (
                sha256_file(previous_report_path) if previous_report_path else None
            ),
            "previous_round_output_receipt_sha256": (
                sha256_file(previous_receipt_path) if previous_receipt_path else None
            ),
            "provenance_partition_exact": True,
            "promotion_approved": False,
        }
        write_json(receipt_path, receipt)
        round_summaries.append(
            {
                "round_index": round_index,
                "round_name": round_name,
                "round_kind": report["round_kind"],
                "removed_object_ids": removed,
                "remaining_object_ids": remaining,
                "status": report["status"],
                "unresolved_pixels": 0,
                "manifest": asset(manifest_path, relative_to=root),
                "report": asset(report_path, relative_to=root),
                "receipt": asset(receipt_path, relative_to=root),
                "output_frame_set_sha256": output_set_sha,
            }
        )
        previous_report_path = report_path
        previous_receipt_path = receipt_path
        previous_report = report
        previous_outputs = current_outputs
        if round_index == 4:
            final_frames = current_outputs

    assert previous_report_path is not None and previous_receipt_path is not None
    final_report = read_json(previous_report_path)
    report_path = root / "layered_clean_plate_sequence_report.json"
    sequence_report = {
        "schema_version": 1,
        "kind": SEQUENCE_REPORT_KIND,
        "status": EXPECTED_SEQUENCE_STATUS,
        "promotion_approved": False,
        "frame_count": len(FRAME_IDS),
        "removed_object_order": list(OBJECT_ORDER),
        "round_remaining_object_ids": ROUND_REMAINING,
        "rounds": round_summaries,
        "accepted_with_limitations": True,
        "acceptance_status": "accepted_with_limitations",
        "acceptance_scope": "current_demo_only",
        "limitations": final_report["limitations"],
        "overridden_gates": final_report["overridden_gates"],
        "failed_metrics": final_report["failed_metrics"],
        "final_output_frame_set_sha256": final_report["output_frame_set_sha256"],
        "gates": {gate: True for gate in REQUIRED_SEQUENCE_GATES},
    }
    write_json(report_path, sequence_report)
    receipt_path = root / "layered_clean_plate_sequence_receipt.json"
    write_json(
        receipt_path,
        {
            "schema_version": 1,
            "kind": SEQUENCE_RECEIPT_KIND,
            "status": EXPECTED_SEQUENCE_STATUS,
            "promotion_approved": False,
            "report": report_path.name,
            "report_sha256": sha256_file(report_path),
            "final_round_report": asset(previous_report_path, relative_to=root),
            "final_round_receipt": asset(previous_receipt_path, relative_to=root),
            "final_output_frame_set_sha256": final_report["output_frame_set_sha256"],
            "accepted_with_limitations": True,
            "acceptance_status": "accepted_with_limitations",
            "acceptance_scope": "current_demo_only",
            "limitations": final_report["limitations"],
            "overridden_gates": final_report["overridden_gates"],
            "failed_metrics": final_report["failed_metrics"],
            "all_round_manifests_reports_and_receipts_are_hash_bound": True,
        },
    )
    return SequenceFixture(
        report=report_path,
        receipt=receipt_path,
        camera_info=camera_info,
        final_frames=final_frames,
    )


def resign_sequence_receipt(fixture: SequenceFixture) -> None:
    receipt = read_json(fixture.receipt)
    receipt["report_sha256"] = sha256_file(fixture.report)
    write_json(fixture.receipt, receipt)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_materializes_verified_final_sequence_without_old_depth(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    output = tmp_path / "package"

    manifest, receipt = materialize_package(
        sequence_report_path=fixture.report,
        sequence_receipt_path=fixture.receipt,
        camera_info_path=fixture.camera_info,
        output=output,
        scene_id="bedroom_4_clean_sequence",
        storage_mode="copy",
        expected_frame_count=2,
    )

    assert sorted(path.name for path in (output / "dslr/resized_undistorted_images").iterdir()) == [
        "000048.png",
        "000049.png",
    ]
    assert not (output / "depth").exists()
    assert not (output / "provenance").exists()
    assert not list(output.rglob("*.npy"))
    assert not list(output.rglob("*.npz"))
    assert manifest["acceptance_scope"] == "current_demo_only"
    assert manifest["promotion_approved"] is False
    assert manifest["provenance_contract"]["fresh_depth_required"] is True
    assert manifest["provenance_contract"]["old_geometry_depth_packaged"] is False
    assert (
        manifest["sequence_validation"][
            "round2_to_round4_predecessor_paths_and_hashes_revalidated_per_frame"
        ]
        is True
    )
    assert receipt["frame_count"] == 2
    assert receipt["materialized_source_file_count"] == 2
    assert receipt["fresh_depth_required"] is True
    assert sha256_file(output / "manifest.json") == receipt["manifest"]["sha256"]
    transforms = read_json(output / "dslr/nerfstudio/transforms_undistorted.json")
    assert [frame["file_path"] for frame in transforms["frames"]] == [
        "000048.png",
        "000049.png",
    ]


def test_rejects_tampered_final_composite_before_creating_output(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    save_rgb(fixture.final_frames["000048"], (255, 0, 0))
    output = tmp_path / "must-not-exist"

    with pytest.raises(RuntimeError, match="composite RGB SHA-256 mismatch"):
        materialize_package(
            sequence_report_path=fixture.report,
            sequence_receipt_path=fixture.receipt,
            camera_info_path=fixture.camera_info,
            output=output,
            scene_id="bedroom_4_clean_sequence",
            storage_mode="copy",
            expected_frame_count=2,
        )

    assert not output.exists()


def test_rejects_wrong_fixed_object_order_even_with_resigned_receipt(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    report = read_json(fixture.report)
    report["removed_object_order"][0:2] = reversed(report["removed_object_order"][0:2])
    write_json(fixture.report, report)
    resign_sequence_receipt(fixture)

    with pytest.raises(RuntimeError, match="fixed object order mismatch"):
        materialize_package(
            sequence_report_path=fixture.report,
            sequence_receipt_path=fixture.receipt,
            camera_info_path=fixture.camera_info,
            output=tmp_path / "must-not-exist",
            scene_id="bedroom_4_clean_sequence",
            storage_mode="copy",
            expected_frame_count=2,
        )


def test_rejects_sequence_receipt_that_does_not_bind_report(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    receipt = read_json(fixture.receipt)
    receipt["report_sha256"] = "0" * 64
    write_json(fixture.receipt, receipt)

    with pytest.raises(RuntimeError, match="does not bind the sequence report"):
        materialize_package(
            sequence_report_path=fixture.report,
            sequence_receipt_path=fixture.receipt,
            camera_info_path=fixture.camera_info,
            output=tmp_path / "must-not-exist",
            scene_id="bedroom_4_clean_sequence",
            storage_mode="copy",
            expected_frame_count=2,
        )


def test_rejects_nonempty_output_without_touching_existing_file(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    output = tmp_path / "occupied"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(RuntimeError, match="output directory is not empty"):
        materialize_package(
            sequence_report_path=fixture.report,
            sequence_receipt_path=fixture.receipt,
            camera_info_path=fixture.camera_info,
            output=output,
            scene_id="bedroom_4_clean_sequence",
            storage_mode="copy",
            expected_frame_count=2,
        )

    assert marker.read_text(encoding="utf-8") == "keep"
