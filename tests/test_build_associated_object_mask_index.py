from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "build_associated_object_mask_index.py"
SPEC = importlib.util.spec_from_file_location("build_associated_object_mask_index", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_association(root: Path) -> Path:
    source_frames = []
    source_masks = []
    frames = []
    item_index = 0
    for index in range(3):
        frame_id = f"{index:06d}"
        frame_path = root / "frames" / f"{frame_id}.png"
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((6, 8, 3), 80 + index, dtype=np.uint8)).save(frame_path)
        frame_sha = MODULE.sha256_file(frame_path)
        source_frames.append(
            {
                "frame_id": frame_id,
                "resolved_path": str(frame_path),
                "sha256": frame_sha,
                "dimensions": [8, 6],
            }
        )
        assignments = []
        for object_id, column in (("front", 2), ("rear", 5)):
            if frame_id == "000002" and object_id == "front":
                continue
            mask = np.zeros((6, 8), dtype=np.uint8)
            mask[2:5, column : column + 2] = 255
            mask_path = root / "source_masks" / frame_id / f"{object_id}.png"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask).save(mask_path)
            mask_sha = MODULE.sha256_file(mask_path)
            detection_id = f"{frame_id}:mask:{item_index:06d}"
            source_masks.append(
                {
                    "detection_id": detection_id,
                    "frame_id": frame_id,
                    "resolved_path": str(mask_path),
                    "sha256": mask_sha,
                    "area_pixels": 6,
                    "score": 0.95,
                }
            )
            assignments.append(
                {
                    "object_id": object_id,
                    "detection_id": detection_id,
                    "frame_sha256": frame_sha,
                    "mask_sha256": mask_sha,
                }
            )
            item_index += 1
        frames.append({"frame_id": frame_id, "assignments": assignments})
    report_path = root / "association.json"
    write_json(
        report_path,
        {
            "status": "passed",
            "association_contract": {
                "physical_identity_source": "explicit --anchor object_id=PLY only",
                "forbidden_identity_sources": [
                    "SAM3 mask filename",
                    "SAM3 per-frame ordinal suffix",
                    "cross-frame item order",
                ],
            },
            "sources": {
                "anchors": [{"object_id": "front"}, {"object_id": "rear"}],
                "frames": source_frames,
                "masks": source_masks,
            },
            "frames": frames,
            "gates": {"passed": True},
        },
    )
    return report_path


def test_builds_source_bound_index_and_drops_incomplete_frames(tmp_path: Path) -> None:
    association_path = make_association(tmp_path)
    output = tmp_path / "output"

    report = MODULE.build_index(
        association_path=association_path,
        object_ids=["front"],
        output_root=output,
        label="associated_removed_object",
    )

    assert report["eligible_donor_frame_count"] == 2
    assert report["dropped_frame_count"] == 1
    assert report["dropped_frames"][0]["frame_id"] == "000002"
    index = json.loads((output / "mask_index.json").read_text(encoding="utf-8"))
    assert index["selected_object_ids"] == ["front"]
    assert index["status"] == MODULE.RAW_INDEX_STATUS
    assert index["eligible_donor_frame_ids"] == ["000000", "000001"]
    assert len(index["items"]) == 2
    assert all(item["object_id"] == "front" for item in index["items"])
    assert all(item["source_rgb_sha256"] for item in index["items"])
    assert all(
        MODULE.sha256_file(output / item["mask_path"]) == item["mask_sha256"]
        for item in index["items"]
    )
    assert (output / "donor_frame_ids.txt").read_text(encoding="ascii") == ("000000\n000001\n")
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["mask_index_sha256"] == MODULE.sha256_file(output / "mask_index.json")
    assert report["status"] == "technical_passed_pending_cumulative_materialization"
    assert "build_cumulative_removal_manifest.py" in report["next_stage"]


def test_multiple_objects_require_every_selected_identity(tmp_path: Path) -> None:
    association_path = make_association(tmp_path)

    report = MODULE.build_index(
        association_path=association_path,
        object_ids=["front", "rear"],
        output_root=tmp_path / "output",
        label="cumulative_removed",
    )

    assert report["eligible_donor_frame_count"] == 2
    index = json.loads((tmp_path / "output" / "mask_index.json").read_text())
    assert len(index["items"]) == 4
    assert {item["object_id"] for item in index["items"]} == {"front", "rear"}


def test_rejects_assignment_bound_to_different_source_rgb(tmp_path: Path) -> None:
    association_path = make_association(tmp_path)
    value = json.loads(association_path.read_text(encoding="utf-8"))
    value["frames"][0]["assignments"][0]["frame_sha256"] = "0" * 64
    write_json(association_path, value)

    with pytest.raises(ValueError, match="exact source RGB"):
        MODULE.build_index(
            association_path=association_path,
            object_ids=["front"],
            output_root=tmp_path / "output",
            label="associated_removed_object",
        )
    assert not (tmp_path / "output").exists()


def test_rejects_tampered_mask_bytes(tmp_path: Path) -> None:
    association_path = make_association(tmp_path)
    value = json.loads(association_path.read_text(encoding="utf-8"))
    mask_path = Path(value["sources"]["masks"][0]["resolved_path"])
    Image.fromarray(np.zeros((6, 8), dtype=np.uint8)).save(mask_path)

    with pytest.raises(ValueError, match="mask SHA-256 mismatch"):
        MODULE.build_index(
            association_path=association_path,
            object_ids=["front"],
            output_root=tmp_path / "output",
            label="associated_removed_object",
        )


def test_rejects_unpassed_or_unanchored_association(tmp_path: Path) -> None:
    association_path = make_association(tmp_path)
    value = json.loads(association_path.read_text(encoding="utf-8"))
    value["status"] = "failed"
    write_json(association_path, value)
    with pytest.raises(ValueError, match="status"):
        MODULE.build_index(
            association_path=association_path,
            object_ids=["front"],
            output_root=tmp_path / "failed-output",
            label="associated_removed_object",
        )

    value["status"] = "passed"
    write_json(association_path, value)
    with pytest.raises(ValueError, match="no physical anchor"):
        MODULE.build_index(
            association_path=association_path,
            object_ids=["unknown"],
            output_root=tmp_path / "unanchored-output",
            label="associated_removed_object",
        )


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (
            lambda value: value["association_contract"].update(
                physical_identity_source="not explicit: category filename identity only"
            ),
            "explicit physical anchors",
        ),
        (
            lambda value: value["association_contract"].update(
                forbidden_identity_sources=["SAM3 mask filename"]
            ),
            "forbidden identity sources",
        ),
        (
            lambda value: value["gates"].update(passed=False),
            "gates.passed",
        ),
        (
            lambda value: value.update(kind="video2world.wrong_association_kind"),
            "report kind",
        ),
    ],
)
def test_rejects_inexact_physical_identity_contract(
    tmp_path: Path,
    mutation: Any,
    expected_error: str,
) -> None:
    association_path = make_association(tmp_path)
    value = json.loads(association_path.read_text(encoding="utf-8"))
    mutation(value)
    write_json(association_path, value)

    with pytest.raises(ValueError, match=expected_error):
        MODULE.build_index(
            association_path=association_path,
            object_ids=["front"],
            output_root=tmp_path / "output",
            label="associated_removed_object",
        )


def test_rejects_unsafe_frame_id_without_writing_outside_staging(tmp_path: Path) -> None:
    association_path = make_association(tmp_path)
    value = json.loads(association_path.read_text(encoding="utf-8"))
    value["sources"]["frames"][0]["frame_id"] = "../../escape"
    write_json(association_path, value)

    with pytest.raises(ValueError, match="safe single path component"):
        MODULE.build_index(
            association_path=association_path,
            object_ids=["front"],
            output_root=tmp_path / "output",
            label="associated_removed_object",
        )

    assert not (tmp_path / "escape").exists()
    assert not (tmp_path / "output").exists()


def test_rejects_unsafe_detection_id(tmp_path: Path) -> None:
    association_path = make_association(tmp_path)
    value = json.loads(association_path.read_text(encoding="utf-8"))
    value["frames"][0]["assignments"][0]["detection_id"] = "../../escape"
    write_json(association_path, value)

    with pytest.raises(ValueError, match="safe single path component"):
        MODULE.build_index(
            association_path=association_path,
            object_ids=["front"],
            output_root=tmp_path / "output",
            label="associated_removed_object",
        )


def test_rejects_detection_reused_across_physical_objects(tmp_path: Path) -> None:
    association_path = make_association(tmp_path)
    value = json.loads(association_path.read_text(encoding="utf-8"))
    for frame in value["frames"][:2]:
        front, rear = frame["assignments"]
        rear["detection_id"] = front["detection_id"]
        rear["mask_sha256"] = front["mask_sha256"]
    write_json(association_path, value)

    with pytest.raises(ValueError, match="physical detection is reused"):
        MODULE.build_index(
            association_path=association_path,
            object_ids=["front", "rear"],
            output_root=tmp_path / "output",
            label="associated_removed_object",
        )


def test_rejects_mask_path_or_hash_reused_across_physical_objects(tmp_path: Path) -> None:
    association_path = make_association(tmp_path)
    value = json.loads(association_path.read_text(encoding="utf-8"))
    frame = value["frames"][0]
    front, rear = frame["assignments"]
    front_mask = next(
        item for item in value["sources"]["masks"] if item["detection_id"] == front["detection_id"]
    )
    rear_mask = next(
        item for item in value["sources"]["masks"] if item["detection_id"] == rear["detection_id"]
    )
    rear_mask.update(
        resolved_path=front_mask["resolved_path"],
        sha256=front_mask["sha256"],
        area_pixels=front_mask["area_pixels"],
    )
    rear["mask_sha256"] = front["mask_sha256"]
    write_json(association_path, value)

    with pytest.raises(ValueError, match="mask path or SHA-256 is reused"):
        MODULE.build_index(
            association_path=association_path,
            object_ids=["front", "rear"],
            output_root=tmp_path / "output",
            label="associated_removed_object",
        )
