from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "build_cumulative_removal_manifest.py"
SPEC = importlib.util.spec_from_file_location("build_cumulative_removal_manifest", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((6, 8, 3), color, dtype=np.uint8)).save(path)


def make_mask(path: Path, box: tuple[int, int, int, int]) -> np.ndarray:
    value = np.zeros((6, 8), dtype=np.uint8)
    x0, y0, x1, y1 = box
    value[y0:y1, x0:x1] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value).save(path)
    return value > 0


def object_manifest(path: Path, frames: Path, masks: Path, object_id: str) -> None:
    records = []
    for index, frame_id in enumerate(("000000", "000001")):
        frame_path = frames / f"{frame_id}.png"
        mask_path = masks / f"{frame_id}.png"
        records.append(
            {
                "sequence_index": index,
                "frame_id": frame_id,
                "source_frame": {
                    "sha256": MODULE.sha256_file(frame_path),
                },
                "input_frame": {
                    "path": str(frame_path),
                    "sha256": MODULE.sha256_file(frame_path),
                },
                "input_mask": {
                    "path": str(mask_path),
                    "sha256": MODULE.sha256_file(mask_path),
                },
            }
        )
    write_json(path, {"object_id": object_id, "frame_records": records})


def args_for(
    *,
    round_index: int,
    object_id: str,
    current_manifest: Path,
    donor_frames: Path,
    output: Path,
    previous_manifest: Path | None = None,
    previous_report: Path | None = None,
    associated_donor_index: Path | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        round_index=round_index,
        object_id=object_id,
        current_object_manifest=current_manifest,
        previous_cumulative_manifest=previous_manifest,
        previous_prefill_report=previous_report,
        observed_donor_frames_dir=donor_frames,
        associated_donor_index=associated_donor_index,
        output=output,
    )


def make_round1(tmp_path: Path) -> tuple[dict[str, Any], Path, dict[str, np.ndarray]]:
    donors = tmp_path / "raw"
    masks = tmp_path / "round1_object_masks"
    expected: dict[str, np.ndarray] = {}
    for index, frame_id in enumerate(("000000", "000001")):
        make_image(donors / f"{frame_id}.png", (20 + index, 30, 40))
        expected[frame_id] = make_mask(masks / f"{frame_id}.png", (1, 1, 3, 3))
    current = tmp_path / "round1_object.json"
    object_manifest(current, donors, masks, "front")
    output = tmp_path / "round1"
    manifest = MODULE.build_manifest(
        args_for(
            round_index=1,
            object_id="front",
            current_manifest=current,
            donor_frames=donors,
            output=output,
        )
    )
    return manifest, output, expected


def fake_prefill_report(
    round_root: Path,
    *,
    wrong_input_hash: bool = False,
    no_support: bool = False,
    boundary_concentrated: bool = False,
    status: str = "technical_passed",
) -> Path:
    manifest_path = round_root / "cumulative_removal_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = []
    for record in manifest["frame_records"]:
        frame_id = record["frame_id"]
        output_path = round_root / "prefill" / f"{frame_id}.png"
        source = Path(record["source_frame"])
        make_image(output_path, tuple(np.asarray(Image.open(source))[0, 0].tolist()))
        mask_path = round_root / "residual" / f"{frame_id}.png"
        removal = MODULE.mask_array(round_root / record["union_mask"])
        residual = np.zeros_like(removal, dtype=np.uint8)
        residual[removal] = 255
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(residual).save(mask_path)
        records.append(
            {
                "frame_id": frame_id,
                "prefill_frame": str(output_path),
                "prefill_frame_sha256": MODULE.sha256_file(output_path),
                "residual_mask": str(mask_path),
                "residual_mask_sha256": MODULE.sha256_file(mask_path),
                "residual_mask_pixels": int(removal.sum()),
                "covered_pixels": 0 if no_support else 1,
            }
        )
    no_support_frame_ids = [record["frame_id"] for record in records] if no_support else []
    unresolved_pixels = sum(record["residual_mask_pixels"] for record in records)
    if no_support:
        promotion_blocker = "one or more target frames have no configured donor support"
        next_action = {
            "action": "add_observed_donor_or_switch_to_constrained_generation_for_residual",
            "blocker": "no_guard_stable_measured_donor_support",
            "status": status,
            "no_support_frame_ids": no_support_frame_ids,
            "unresolved_unobserved_pixels": unresolved_pixels,
            "promotion_approved": False,
        }
    elif boundary_concentrated:
        promotion_blocker = "boundary-guard donor support validation failed"
        next_action = {
            "action": "tighten_physical_donor_exclusion_or_add_nonboundary_donor_views",
            "blocker": "donor_support_boundary_concentrated",
            "status": status,
            "no_support_frame_ids": [],
            "unresolved_unobserved_pixels": unresolved_pixels,
            "promotion_approved": False,
        }
    else:
        promotion_blocker = "semantic texture continuity review is required"
        next_action = {
            "action": "run_semantic_texture_and_new_depth_normal_review_before_next_round",
            "blocker": "semantic_texture_and_depth_normal_pending",
            "status": status,
            "no_support_frame_ids": [],
            "unresolved_unobserved_pixels": unresolved_pixels,
            "promotion_approved": False,
        }
    report = round_root / "prefill_report.json"
    write_json(
        report,
        {
            "status": status,
            "promotion_approved": False,
            "promotion_blocker": promotion_blocker,
            "next_action": next_action,
            "input_manifest_sha256": (
                "0" * 64 if wrong_input_hash else MODULE.sha256_file(manifest_path)
            ),
            "gates": {
                "outside_removal_mask_rgb_exact": True,
                "all_residual_masks_subset_of_removal_masks": True,
                "donor_support_available_for_every_target": not no_support,
                "donor_support_not_boundary_concentrated": not (
                    no_support or boundary_concentrated
                ),
            },
            "pixel_provenance": {
                "generated_pixels": 0,
                "propainter_pixels": 0,
                "unresolved_pixels_are_not_valid_donor_or_geometry_evidence": True,
                "measured_multiview_pixels": 0 if no_support else len(records),
                "unresolved_unobserved_pixels": unresolved_pixels,
            },
            "frame_records": records,
        },
    )
    return report


def make_associated_index(
    root: Path,
    donor_frames: Path,
    *,
    object_ids: list[str],
    frame_ids: tuple[str, ...] = ("000000", "000001", "000002", "000003"),
) -> tuple[Path, dict[tuple[str, str], Path]]:
    items = []
    mask_paths: dict[tuple[str, str], Path] = {}
    for frame_id in frame_ids:
        source_path = donor_frames / f"{frame_id}.png"
        for object_index, object_id in enumerate(object_ids):
            mask_path = root / "associated_source_masks" / frame_id / f"{object_id}.png"
            mask = make_mask(
                mask_path,
                (1 + object_index * 3, 1, 3 + object_index * 3, 4),
            )
            mask_paths[(frame_id, object_id)] = mask_path
            items.append(
                {
                    "frame_id": frame_id,
                    "image": source_path.name,
                    "source_rgb_path": str(source_path),
                    "source_rgb_sha256": MODULE.sha256_file(source_path),
                    "label": "associated_removed_object",
                    "object_id": object_id,
                    "detection_id": f"{frame_id}:mask:{object_index:06d}",
                    "score": 0.95,
                    "mask_path": str(mask_path),
                    "mask_sha256": MODULE.sha256_file(mask_path),
                    "mask_pixels": int(mask.sum()),
                    "physical_identity_source": "association_explicit_3d_anchor",
                }
            )
    index_path = root / "associated_source_index.json"
    write_json(
        index_path,
        {
            "schema_version": 1,
            "kind": MODULE.ASSOCIATED_DONOR_INDEX_KIND,
            "status": "ready_for_measured_multiview_prefill",
            "selected_object_ids": object_ids,
            "label": "associated_removed_object",
            "eligible_donor_frame_ids": list(frame_ids),
            "items": items,
            "gates": {
                "physical_identity_from_explicit_3d_anchors": True,
                "every_item_bound_to_exact_source_rgb_sha256": True,
                "every_mask_sha256_verified": True,
                "incomplete_frames_excluded_from_donors": True,
            },
        },
    )
    return index_path, mask_paths


def make_round2_associated_fixture(
    tmp_path: Path,
    *,
    associated_object_ids: list[str] | None = None,
) -> tuple[Path, Path, Path, Path, Path, dict[tuple[str, str], Path]]:
    _, round1, _ = make_round1(tmp_path)
    report = fake_prefill_report(round1)
    donors = tmp_path / "raw"
    for index, frame_id in enumerate(("000002", "000003"), start=2):
        make_image(donors / f"{frame_id}.png", (20 + index, 30, 40))
    masks = tmp_path / "round2_object_masks"
    for frame_id in ("000000", "000001"):
        make_mask(masks / f"{frame_id}.png", (4, 2, 7, 5))
    current = tmp_path / "round2_object.json"
    object_manifest(current, round1 / "prefill", masks, "left")
    associated_index, associated_masks = make_associated_index(
        tmp_path,
        donors,
        object_ids=associated_object_ids or ["front", "left"],
    )
    return round1, report, donors, current, associated_index, associated_masks


def test_round2_unions_previous_masks_and_hashes_immediate_input(tmp_path: Path) -> None:
    _, round1, round1_masks = make_round1(tmp_path)
    report = fake_prefill_report(round1)
    donors = tmp_path / "raw"
    masks = tmp_path / "round2_object_masks"
    round2_masks = {
        frame_id: make_mask(masks / f"{frame_id}.png", (4, 2, 7, 5))
        for frame_id in ("000000", "000001")
    }
    current = tmp_path / "round2_object.json"
    object_manifest(current, round1 / "prefill", masks, "left")

    output = tmp_path / "round2"
    manifest = MODULE.build_manifest(
        args_for(
            round_index=2,
            object_id="left",
            current_manifest=current,
            donor_frames=donors,
            output=output,
            previous_manifest=round1 / "cumulative_removal_manifest.json",
            previous_report=report,
        )
    )

    assert manifest["removed_object_ids"] == ["front", "left"]
    assert manifest["gates"]["unresolved_residual_is_never_donor_evidence"] is True
    assert manifest["gates"]["previous_prefill_excludes_generated_and_propainter_pixels"] is True
    assert manifest["gates"]["current_object_mask_is_bound_to_exact_scene_source_rgb"] is True
    for record in manifest["frame_records"]:
        frame_id = record["frame_id"]
        actual = MODULE.mask_array(output / record["union_mask"])
        assert np.array_equal(actual, round1_masks[frame_id] | round2_masks[frame_id])
        assert record["input_lineage"]["previous_unresolved_pixels_remain_non_donor"] is True
        assert (
            record["current_object_mask_segmentation_source_rgb_sha256"]
            == record["input_lineage"]["previous_prefill_frame_sha256"]
        )
        assert record["removed_object_ids"] == ["front", "left"]
    donor_index = json.loads((output / "donor_exclusion_index.json").read_text())
    assert {item["label"] for item in donor_index["items"]} == {"cumulative_removed"}
    for item in donor_index["items"]:
        donor_asset = manifest["donor_contract"]["assets"][item["frame_id"]]
        assert item["source_rgb_path"] == donor_asset["path"]
        assert item["source_rgb_sha256"] == donor_asset["sha256"]
    assert all(
        item["unresolved_residual_allowed_as_donor"] is False for item in donor_index["items"]
    )


def test_round2_materializes_full_associated_physical_donor_contract(
    tmp_path: Path,
) -> None:
    round1, report, donors, current, associated_index, source_masks = (
        make_round2_associated_fixture(tmp_path)
    )
    output = tmp_path / "round2-associated"

    manifest = MODULE.build_manifest(
        args_for(
            round_index=2,
            object_id="left",
            current_manifest=current,
            donor_frames=donors,
            output=output,
            previous_manifest=round1 / "cumulative_removal_manifest.json",
            previous_report=report,
            associated_donor_index=associated_index,
        )
    )

    contract = manifest["donor_contract"]
    assert contract["exclusion_mode"] == "associated_physical_instance_masks"
    assert contract["eligible_donor_frame_ids"] == [
        "000000",
        "000001",
        "000002",
        "000003",
    ]
    assert set(contract["assets"]) == set(contract["eligible_donor_frame_ids"])
    assert contract["physical_instance_object_ids"] == ["front", "left"]
    assert contract["per_donor_multi_mask_union"] is True
    assert contract["per_donor_mask_count"] == 2
    assert manifest["gates"]["associated_physical_instance_donor_exclusion_enforced"] is True

    portable_path = output / contract["exclusion_index"]
    portable = json.loads(portable_path.read_text(encoding="utf-8"))
    assert portable["selected_object_ids"] == ["front", "left"]
    assert portable["target_removed_object_union"] == ["front", "left"]
    assert portable["gates"]["per_frame_physical_assignments_are_one_to_one"] is True
    assert len(portable["items"]) == 8
    assert MODULE.sha256_file(portable_path) == contract["exclusion_index_sha256"]
    for frame_id in portable["eligible_donor_frame_ids"]:
        frame_items = [item for item in portable["items"] if item["frame_id"] == frame_id]
        assert [item["object_id"] for item in frame_items] == ["front", "left"]
        masks = []
        for item in frame_items:
            copied = portable_path.parent / item["mask_path"]
            assert copied.is_file()
            assert MODULE.sha256_file(copied) == item["mask_sha256"]
            assert item["mask_sha256"] == MODULE.sha256_file(
                source_masks[(frame_id, item["object_id"])]
            )
            masks.append(MODULE.mask_array(copied))
        union_record = portable["donor_unions"][frame_id]
        union = MODULE.mask_array(portable_path.parent / union_record["path"])
        assert np.array_equal(union, masks[0] | masks[1])
        assert union_record["object_ids"] == ["front", "left"]
        assert union_record["component_mask_count"] == 2
    assert (output / "associated_donor_evidence/source_mask_index.json").read_bytes() == (
        associated_index.read_bytes()
    )
    assert (output / contract["legacy_target_cumulative_exclusion_index"]).is_file()


def test_round2_rejects_associated_object_ids_outside_target_union(tmp_path: Path) -> None:
    round1, report, donors, current, associated_index, _ = make_round2_associated_fixture(
        tmp_path,
        associated_object_ids=["front"],
    )

    with pytest.raises(ValueError, match="do not equal the cumulative removed-object union"):
        MODULE.build_manifest(
            args_for(
                round_index=2,
                object_id="left",
                current_manifest=current,
                donor_frames=donors,
                output=tmp_path / "round2-associated",
                previous_manifest=round1 / "cumulative_removal_manifest.json",
                previous_report=report,
                associated_donor_index=associated_index,
            )
        )


def test_round2_rejects_tampered_associated_mask_sha(tmp_path: Path) -> None:
    round1, report, donors, current, associated_index, source_masks = (
        make_round2_associated_fixture(tmp_path)
    )
    make_mask(source_masks[("000002", "left")], (0, 0, 2, 2))

    with pytest.raises(ValueError, match="associated donor mask SHA-256 mismatch"):
        MODULE.build_manifest(
            args_for(
                round_index=2,
                object_id="left",
                current_manifest=current,
                donor_frames=donors,
                output=tmp_path / "round2-associated",
                previous_manifest=round1 / "cumulative_removal_manifest.json",
                previous_report=report,
                associated_donor_index=associated_index,
            )
        )


def test_round2_accepts_new_raw_index_status_before_materialization(tmp_path: Path) -> None:
    round1, report, donors, current, associated_index, _ = make_round2_associated_fixture(tmp_path)
    value = json.loads(associated_index.read_text(encoding="utf-8"))
    value["status"] = MODULE.RAW_ASSOCIATED_INDEX_STATUS
    write_json(associated_index, value)

    manifest = MODULE.build_manifest(
        args_for(
            round_index=2,
            object_id="left",
            current_manifest=current,
            donor_frames=donors,
            output=tmp_path / "round2-associated",
            previous_manifest=round1 / "cumulative_removal_manifest.json",
            previous_report=report,
            associated_donor_index=associated_index,
        )
    )

    assert manifest["status"] == "ready_for_measured_multiview_prefill"


def test_round2_rejects_reused_physical_detection_in_external_index(
    tmp_path: Path,
) -> None:
    round1, report, donors, current, associated_index, _ = make_round2_associated_fixture(tmp_path)
    value = json.loads(associated_index.read_text(encoding="utf-8"))
    for frame_id in value["eligible_donor_frame_ids"]:
        front = next(
            item
            for item in value["items"]
            if item["frame_id"] == frame_id and item["object_id"] == "front"
        )
        left = next(
            item
            for item in value["items"]
            if item["frame_id"] == frame_id and item["object_id"] == "left"
        )
        left["detection_id"] = front["detection_id"]
    write_json(associated_index, value)

    with pytest.raises(ValueError, match="physical detection is reused"):
        MODULE.build_manifest(
            args_for(
                round_index=2,
                object_id="left",
                current_manifest=current,
                donor_frames=donors,
                output=tmp_path / "round2-associated",
                previous_manifest=round1 / "cumulative_removal_manifest.json",
                previous_report=report,
                associated_donor_index=associated_index,
            )
        )


def test_round2_rejects_reused_physical_mask_in_external_index(tmp_path: Path) -> None:
    round1, report, donors, current, associated_index, _ = make_round2_associated_fixture(tmp_path)
    value = json.loads(associated_index.read_text(encoding="utf-8"))
    for frame_id in value["eligible_donor_frame_ids"]:
        front = next(
            item
            for item in value["items"]
            if item["frame_id"] == frame_id and item["object_id"] == "front"
        )
        left = next(
            item
            for item in value["items"]
            if item["frame_id"] == frame_id and item["object_id"] == "left"
        )
        left.update(
            mask_path=front["mask_path"],
            mask_sha256=front["mask_sha256"],
            mask_pixels=front["mask_pixels"],
        )
    write_json(associated_index, value)

    with pytest.raises(ValueError, match="mask path or SHA-256 is reused"):
        MODULE.build_manifest(
            args_for(
                round_index=2,
                object_id="left",
                current_manifest=current,
                donor_frames=donors,
                output=tmp_path / "round2-associated",
                previous_manifest=round1 / "cumulative_removal_manifest.json",
                previous_report=report,
                associated_donor_index=associated_index,
            )
        )


def test_rejects_unsafe_object_id_before_materializing_paths(tmp_path: Path) -> None:
    donors = tmp_path / "raw"
    masks = tmp_path / "masks"
    for frame_id in ("000000", "000001"):
        make_image(donors / f"{frame_id}.png", (20, 30, 40))
        make_mask(masks / f"{frame_id}.png", (1, 1, 3, 3))
    current = tmp_path / "current.json"
    object_manifest(current, donors, masks, "../escape")

    with pytest.raises(ValueError, match="safe single path component"):
        MODULE.build_manifest(
            args_for(
                round_index=1,
                object_id="../escape",
                current_manifest=current,
                donor_frames=donors,
                output=tmp_path / "round1",
            )
        )

    assert not (tmp_path / "escape.png").exists()


def test_round2_rejects_previous_report_with_no_measured_support(tmp_path: Path) -> None:
    _, round1, _ = make_round1(tmp_path)
    report = fake_prefill_report(round1, no_support=True)
    donors = tmp_path / "raw"
    masks = tmp_path / "round2_object_masks"
    for frame_id in ("000000", "000001"):
        make_mask(masks / f"{frame_id}.png", (4, 2, 7, 5))
    current = tmp_path / "round2_object.json"
    object_manifest(current, round1 / "prefill", masks, "left")

    with pytest.raises(ValueError, match="no donor support"):
        MODULE.build_manifest(
            args_for(
                round_index=2,
                object_id="left",
                current_manifest=current,
                donor_frames=donors,
                output=tmp_path / "round2",
                previous_manifest=round1 / "cumulative_removal_manifest.json",
                previous_report=report,
            )
        )


def test_round2_surfaces_next_action_from_failed_previous_report(tmp_path: Path) -> None:
    _, round1, _ = make_round1(tmp_path)
    report = fake_prefill_report(
        round1,
        no_support=True,
        status="technical_failed_no_support",
    )
    report_value = json.loads(report.read_text(encoding="utf-8"))
    report_value["next_action"].update(
        {
            "failed_frame_ids": ["000064"],
            "first_failed_frame_id": "000064",
            "not_evaluable_pair_ids": ["0016_to_0017"],
            "not_evaluable_triplet_center_frame_ids": ["000064"],
        }
    )
    write_json(report, report_value)
    donors = tmp_path / "raw"
    masks = tmp_path / "round2_object_masks"
    for frame_id in ("000000", "000001"):
        make_mask(masks / f"{frame_id}.png", (4, 2, 7, 5))
    current = tmp_path / "round2_object.json"
    object_manifest(current, round1 / "prefill", masks, "left")

    with pytest.raises(ValueError) as error:
        MODULE.build_manifest(
            args_for(
                round_index=2,
                object_id="left",
                current_manifest=current,
                donor_frames=donors,
                output=tmp_path / "round2",
                previous_manifest=round1 / "cumulative_removal_manifest.json",
                previous_report=report,
            )
        )

    message = str(error.value)
    assert "previous prefill report did not technically pass" in message
    assert "status=technical_failed_no_support" in message
    assert "add_observed_donor_or_switch_to_constrained_generation_for_residual" in message
    assert "no_guard_stable_measured_donor_support" in message
    assert "failed_frame_ids=000064" in message
    assert "first_failed_frame_id=000064" in message
    assert "not_evaluable_pair_ids=0016_to_0017" in message
    assert "not_evaluable_triplet_center_frame_ids=000064" in message
    assert "no_support_frame_ids=000000,000001" in message


def test_round2_surfaces_boundary_guard_next_action(tmp_path: Path) -> None:
    _, round1, _ = make_round1(tmp_path)
    report = fake_prefill_report(round1, boundary_concentrated=True)
    donors = tmp_path / "raw"
    masks = tmp_path / "round2_object_masks"
    for frame_id in ("000000", "000001"):
        make_mask(masks / f"{frame_id}.png", (4, 2, 7, 5))
    current = tmp_path / "round2_object.json"
    object_manifest(current, round1 / "prefill", masks, "left")

    with pytest.raises(ValueError) as error:
        MODULE.build_manifest(
            args_for(
                round_index=2,
                object_id="left",
                current_manifest=current,
                donor_frames=donors,
                output=tmp_path / "round2",
                previous_manifest=round1 / "cumulative_removal_manifest.json",
                previous_report=report,
            )
        )

    message = str(error.value)
    assert "previous prefill donor support failed its boundary guard" in message
    assert "promotion_blocker=boundary-guard donor support validation failed" in message
    assert "tighten_physical_donor_exclusion_or_add_nonboundary_donor_views" in message
    assert "donor_support_boundary_concentrated" in message


def test_round2_rejects_stale_associated_source_sha(tmp_path: Path) -> None:
    round1, report, donors, current, associated_index, _ = make_round2_associated_fixture(tmp_path)
    make_image(donors / "000002.png", (250, 1, 2))

    with pytest.raises(ValueError, match="associated donor source RGB SHA-256 mismatch"):
        MODULE.build_manifest(
            args_for(
                round_index=2,
                object_id="left",
                current_manifest=current,
                donor_frames=donors,
                output=tmp_path / "round2-associated",
                previous_manifest=round1 / "cumulative_removal_manifest.json",
                previous_report=report,
                associated_donor_index=associated_index,
            )
        )


def test_round2_rejects_a_previous_report_with_wrong_manifest_hash(tmp_path: Path) -> None:
    _, round1, _ = make_round1(tmp_path)
    report = fake_prefill_report(round1, wrong_input_hash=True)
    donors = tmp_path / "raw"
    masks = tmp_path / "round2_object_masks"
    for frame_id in ("000000", "000001"):
        make_mask(masks / f"{frame_id}.png", (4, 2, 7, 5))
    current = tmp_path / "round2_object.json"
    object_manifest(current, donors, masks, "left")

    with pytest.raises(ValueError, match="does not hash"):
        MODULE.build_manifest(
            args_for(
                round_index=2,
                object_id="left",
                current_manifest=current,
                donor_frames=donors,
                output=tmp_path / "round2",
                previous_manifest=round1 / "cumulative_removal_manifest.json",
                previous_report=report,
            )
        )


def test_round2_rejects_mask_segmented_on_a_different_rgb_source(tmp_path: Path) -> None:
    _, round1, _ = make_round1(tmp_path)
    report = fake_prefill_report(round1)
    donors = tmp_path / "raw"
    wrong_source = tmp_path / "wrong_round2_source"
    masks = tmp_path / "round2_object_masks"
    for index, frame_id in enumerate(("000000", "000001")):
        make_image(wrong_source / f"{frame_id}.png", (190 + index, 10, 20))
        make_mask(masks / f"{frame_id}.png", (4, 2, 7, 5))
    current = tmp_path / "round2_object.json"
    object_manifest(current, wrong_source, masks, "left")

    with pytest.raises(ValueError, match="not segmented on the exact immediately previous"):
        MODULE.build_manifest(
            args_for(
                round_index=2,
                object_id="left",
                current_manifest=current,
                donor_frames=donors,
                output=tmp_path / "round2",
                previous_manifest=round1 / "cumulative_removal_manifest.json",
                previous_report=report,
            )
        )


@pytest.mark.parametrize("field", ["generated_pixels", "propainter_pixels"])
def test_round2_rejects_missing_previous_generated_provenance(tmp_path: Path, field: str) -> None:
    _, round1, _ = make_round1(tmp_path)
    report = fake_prefill_report(round1)
    value = json.loads(report.read_text(encoding="utf-8"))
    del value["pixel_provenance"][field]
    write_json(report, value)
    donors = tmp_path / "raw"
    masks = tmp_path / "round2_object_masks"
    for frame_id in ("000000", "000001"):
        make_mask(masks / f"{frame_id}.png", (4, 2, 7, 5))
    current = tmp_path / "round2_object.json"
    object_manifest(current, donors, masks, "left")

    with pytest.raises(ValueError, match="contains or omits"):
        MODULE.build_manifest(
            args_for(
                round_index=2,
                object_id="left",
                current_manifest=current,
                donor_frames=donors,
                output=tmp_path / "round2",
                previous_manifest=round1 / "cumulative_removal_manifest.json",
                previous_report=report,
            )
        )


def test_round1_rejects_a_non_original_source_frame(tmp_path: Path) -> None:
    donors = tmp_path / "raw"
    source = tmp_path / "generated"
    masks = tmp_path / "masks"
    for frame_id in ("000000", "000001"):
        make_image(donors / f"{frame_id}.png", (10, 20, 30))
        make_image(source / f"{frame_id}.png", (200, 10, 10))
        make_mask(masks / f"{frame_id}.png", (1, 1, 3, 3))
    current = tmp_path / "round1_object.json"
    object_manifest(current, source, masks, "front")

    with pytest.raises(ValueError, match="not original observed RGB"):
        MODULE.build_manifest(
            args_for(
                round_index=1,
                object_id="front",
                current_manifest=current,
                donor_frames=donors,
                output=tmp_path / "round1",
            )
        )
