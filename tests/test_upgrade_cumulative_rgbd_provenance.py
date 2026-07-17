from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

SCRIPT_PATH = (
    Path(__file__).parents[1] / "scripts" / "upgrade_cumulative_rgbd_provenance.py"
)
SPEC = importlib.util.spec_from_file_location("upgrade_cumulative_rgbd_provenance", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashed_path(path: Path, path_key: str, sha_key: str) -> dict[str, str]:
    return {path_key: str(path), sha_key: sha256_file(path)}


def tree_snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def make_round(
    root: Path,
    round_index: int,
    previous_manifest: Path | None,
    previous_report: Path | None,
) -> dict[str, Any]:
    round_name = ("front", "left", "right")[round_index - 1]
    round_root = root / f"round{round_index:02d}_{round_name}"
    manifest_root = round_root / "manifest"
    output_root = round_root / "output"
    asset_root = root / "assets" / f"round{round_index:02d}"

    current_object_manifest = asset_root / "current_object.json"
    camera_info = asset_root / "camera_info.json"
    source = asset_root / "source.png"
    donor = asset_root / "donor.png"
    donor_depth = asset_root / "donor_depth.npy"
    mask = asset_root / "removal_mask.png"
    prefill = asset_root / "prefill.png"
    support = asset_root / "support.png"
    measured_depth = asset_root / "measured_depth.npy"
    for path, payload in (
        (current_object_manifest, f'{{"object_id":"{round_name}"}}\n'.encode()),
        (camera_info, f'{{"round":{round_index}}}\n'.encode()),
        (source, f"source-pixels-{round_index}".encode()),
        (donor, f"donor-pixels-{round_index}".encode()),
        (donor_depth, f"donor-depth-{round_index}".encode()),
        (mask, f"binary-mask-{round_index}".encode()),
        (prefill, f"prefill-pixels-{round_index}".encode()),
        (support, f"support-pixels-{round_index}".encode()),
        (measured_depth, f"measured-depth-{round_index}".encode()),
    ):
        write_bytes(path, payload)

    donor_index = manifest_root / "donor_exclusion_index.json"
    write_json(
        donor_index,
        {
            "round_index": round_index,
            "items": [
                {
                    "frame_id": "000000",
                    "mask_path": str(mask),
                    "mask_sha256": sha256_file(mask),
                }
            ],
        },
    )

    previous_manifest_sha = sha256_file(previous_manifest) if previous_manifest else None
    previous_report_sha = sha256_file(previous_report) if previous_report else None
    lineage = {
        "previous_cumulative_manifest": str(previous_manifest) if previous_manifest else None,
        "previous_cumulative_manifest_sha256": previous_manifest_sha,
        "previous_prefill_report": str(previous_report) if previous_report else None,
        "previous_prefill_report_sha256": previous_report_sha,
    }
    manifest_path = manifest_root / "cumulative_removal_manifest.json"
    manifest = {
        "schema_version": 1,
        "round_index": round_index,
        "removed_object_ids": ["front", "left", "right"][:round_index],
        "current_object_manifest": str(current_object_manifest),
        "current_object_manifest_sha256": sha256_file(current_object_manifest),
        "lineage": lineage,
        "gates": {
            "previous_prefill_excludes_generated_and_propainter_pixels": False,
        },
        "donor_contract": {
            "propainter_rgb_as_donor": None,
            "assets": {
                "000000": {
                    "path": str(donor),
                    "sha256": sha256_file(donor),
                }
            },
        },
        "pixel_provenance_classes": {
            "measured": "original observed RGB-D",
        },
        "frame_records": [
            {
                "sequence_index": 0,
                "frame_id": "000000",
                **hashed_path(source, "source_frame", "source_frame_sha256"),
                **hashed_path(mask, "union_mask", "union_mask_sha256"),
                **hashed_path(
                    mask,
                    "donor_exclusion_mask",
                    "donor_exclusion_mask_sha256",
                ),
                "mask_components": [
                    {
                        "object_id": round_name,
                        **hashed_path(mask, "path", "sha256"),
                    }
                ],
            }
        ],
    }
    write_json(manifest_path, manifest)

    manifest_receipt = manifest_root / "receipt.json"
    write_json(
        manifest_receipt,
        {
            "manifest_sha256": sha256_file(manifest_path),
            "donor_exclusion_index_sha256": sha256_file(donor_index),
        },
    )

    report_path = output_root / "multiview_prefill_report.json"
    report = {
        "schema_version": 2,
        **hashed_path(manifest_path, "input_manifest", "input_manifest_sha256"),
        **hashed_path(camera_info, "camera_info", "camera_info_sha256"),
        **hashed_path(donor_index, "donor_mask_index", "donor_mask_index_sha256"),
        "cumulative_removal_contract": {
            "round_index": round_index,
            "manifest_sha256": sha256_file(manifest_path),
            "lineage": copy.deepcopy(lineage),
        },
        "pixel_provenance": {
            "generated_pixels": 0,
            "unresolved_pixels_are_not_valid_donor_or_geometry_evidence": True,
        },
        "gates": {
            "no_generated_or_propainter_pixels_as_measured_evidence": False,
        },
        "donor_assets": {
            "000000": {
                **hashed_path(donor, "frame", "frame_sha256"),
                **hashed_path(donor_depth, "depth", "depth_sha256"),
                "exclusion_masks": [hashed_path(mask, "path", "sha256")],
            }
        },
        "frame_records": [
            {
                "sequence_index": 0,
                "frame_id": "000000",
                **hashed_path(source, "source_frame", "source_frame_sha256"),
                **hashed_path(mask, "removal_mask", "removal_mask_sha256"),
                **hashed_path(prefill, "prefill_frame", "prefill_frame_sha256"),
                **hashed_path(mask, "residual_mask", "residual_mask_sha256"),
                **hashed_path(
                    support,
                    "support_visualization",
                    "support_visualization_sha256",
                ),
                **hashed_path(
                    measured_depth,
                    "measured_depth",
                    "measured_depth_sha256",
                ),
            }
        ],
    }
    write_json(report_path, report)

    contact_sheet = output_root / "multiview_prefill_contact_sheet.png"
    write_bytes(contact_sheet, f"contact-sheet-{round_index}".encode())
    report_receipt = output_root / "multiview_prefill_receipt.json"
    write_json(
        report_receipt,
        {
            "report_sha256": sha256_file(report_path),
            "contact_sheet_sha256": sha256_file(contact_sheet),
        },
    )
    return {
        "round_index": round_index,
        "round_root": round_root,
        "manifest": manifest_path,
        "report": report_path,
        "contact_sheet": contact_sheet,
        "summary_round": {
            "round_index": round_index,
            "manifest_sha256": sha256_file(manifest_path),
            "report_sha256": sha256_file(report_path),
            "contact_sheet_sha256": sha256_file(contact_sheet),
        },
    }


def make_fixture(root: Path) -> list[dict[str, Any]]:
    rounds: list[dict[str, Any]] = []
    previous_manifest: Path | None = None
    previous_report: Path | None = None
    for round_index in range(1, 4):
        state = make_round(root, round_index, previous_manifest, previous_report)
        rounds.append(state)
        previous_manifest = state["manifest"]
        previous_report = state["report"]
    write_json(
        root / "summary.json",
        {
            "schema_version": 1,
            "method": {"propainter_rgb_as_donor": None},
            "pixel_provenance": {"propainter": "legacy field was omitted"},
            "rounds": [state["summary_round"] for state in rounds],
        },
    )
    return rounds


def metadata_paths(root: Path, rounds: list[dict[str, Any]]) -> set[Path]:
    result = {root / "summary.json"}
    for state in rounds:
        result.update(
            {
                state["manifest"],
                state["manifest"].parent / "receipt.json",
                state["report"],
                state["report"].parent / "multiview_prefill_receipt.json",
            }
        )
    return result


def test_check_only_computes_upgrade_without_writing(tmp_path: Path) -> None:
    root = tmp_path / "cumulative"
    rounds = make_fixture(root)
    before = tree_snapshot(root)

    receipt = MODULE.upgrade(root, check_only=True)

    assert receipt["status"] == "technical_passed_dry_run"
    assert receipt["would_write_file_count"] == 14
    assert receipt["contract"]["generated_pixels"] == 0
    assert receipt["contract"]["propainter_pixels"] == 0
    assert receipt["contract"]["recursive_front_to_back_hash_lineage_rebound"] is True
    assert len(receipt["rounds"]) == 3
    assert all(item["pixel_payloads_modified"] is False for item in receipt["rounds"])
    assert tree_snapshot(root) == before
    assert not (root / "metadata-before-provenance-upgrade").exists()
    assert not (root / "provenance_upgrade_receipt.json").exists()
    for state in rounds:
        report = MODULE.read_json(state["report"])
        assert "propainter_pixels" not in report["pixel_provenance"]


def test_upgrade_recursively_rebinds_three_rounds_without_changing_payloads(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cumulative"
    rounds = make_fixture(root)
    mutable_metadata = metadata_paths(root, rounds)
    payloads_before = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and path not in mutable_metadata
    }
    metadata_before = {path.relative_to(root): path.read_bytes() for path in mutable_metadata}

    receipt = MODULE.upgrade(root, check_only=False)

    assert receipt["status"] == "technical_passed_metadata_rebound"
    assert receipt["verified_assets"]["all_preexisting_sha256_bindings_passed_before_write"]
    on_disk_receipt = MODULE.read_json(root / "provenance_upgrade_receipt.json")
    assert on_disk_receipt["contract"] == receipt["contract"]
    assert on_disk_receipt["summary"] == receipt["summary"]

    previous_manifest: Path | None = None
    previous_manifest_sha: str | None = None
    previous_report: Path | None = None
    previous_report_sha: str | None = None
    summary = MODULE.read_json(root / "summary.json")
    for state, summary_round, receipt_round in zip(
        rounds, summary["rounds"], receipt["rounds"], strict=True
    ):
        manifest = MODULE.read_json(state["manifest"])
        report = MODULE.read_json(state["report"])
        manifest_receipt = MODULE.read_json(state["manifest"].parent / "receipt.json")
        report_receipt = MODULE.read_json(
            state["report"].parent / "multiview_prefill_receipt.json"
        )
        manifest_sha = sha256_file(state["manifest"])
        report_sha = sha256_file(state["report"])
        lineage = manifest["lineage"]

        assert lineage["previous_cumulative_manifest"] == (
            str(previous_manifest) if previous_manifest else None
        )
        assert lineage["previous_cumulative_manifest_sha256"] == previous_manifest_sha
        assert lineage["previous_prefill_report"] == (
            str(previous_report) if previous_report else None
        )
        assert lineage["previous_prefill_report_sha256"] == previous_report_sha
        assert report["input_manifest_sha256"] == manifest_sha
        assert report["cumulative_removal_contract"]["manifest_sha256"] == manifest_sha
        assert report["cumulative_removal_contract"]["lineage"] == lineage
        assert report["pixel_provenance"]["generated_pixels"] == 0
        assert report["pixel_provenance"]["propainter_pixels"] == 0
        assert report["gates"]["no_generated_or_propainter_pixels_as_measured_evidence"]
        assert manifest["gates"][
            "previous_prefill_excludes_generated_and_propainter_pixels"
        ]
        assert manifest["donor_contract"]["propainter_rgb_as_donor"] is False
        assert summary_round["manifest_sha256"] == manifest_sha
        assert summary_round["report_sha256"] == report_sha
        assert receipt_round["after_manifest_sha256"] == manifest_sha
        assert receipt_round["after_report_sha256"] == report_sha
        assert receipt_round["before_manifest_sha256"] != manifest_sha
        assert receipt_round["before_report_sha256"] != report_sha
        assert manifest_receipt["manifest_sha256"] == manifest_sha
        assert manifest_receipt["metadata_only_provenance_upgrade"] is True
        assert report_receipt["report_sha256"] == report_sha
        assert report_receipt["measured_provenance_excludes_generated_pixels"] is True
        assert report_receipt["measured_provenance_excludes_propainter_pixels"] is True

        previous_manifest = state["manifest"]
        previous_manifest_sha = manifest_sha
        previous_report = state["report"]
        previous_report_sha = report_sha

    assert summary["method"]["propainter_rgb_as_donor"] is False
    assert summary["pixel_provenance"]["propainter"] == (
        "zero pixels in this measurement stage"
    )
    assert {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
        and path not in mutable_metadata
        and "metadata-before-provenance-upgrade" not in path.parts
        and path.name != "provenance_upgrade_receipt.json"
    } == payloads_before

    backup = root / "metadata-before-provenance-upgrade"
    for relative_path, payload in metadata_before.items():
        assert (backup / relative_path).read_bytes() == payload
    assert "propainter_pixels" not in MODULE.read_json(
        backup / rounds[0]["report"].relative_to(root)
    )["pixel_provenance"]

    _, validated_states, _ = MODULE.validate_existing(root)
    assert len(validated_states) == 3


@pytest.mark.parametrize(
    "tamper_target",
    (
        "assets/round01/source.png",
        "assets/round02/measured_depth.npy",
        "round03_right/output/multiview_prefill_contact_sheet.png",
        "assets/round01/current_object.json",
    ),
)
def test_upgrade_rejects_tampered_assets_before_any_write(
    tmp_path: Path, tamper_target: str
) -> None:
    root = tmp_path / "cumulative"
    make_fixture(root)
    target = root / tamper_target
    target.write_bytes(target.read_bytes() + b"-tampered")
    before = tree_snapshot(root)

    with pytest.raises(ValueError, match="mismatch"):
        MODULE.upgrade(root, check_only=False)

    assert tree_snapshot(root) == before
    assert not (root / "metadata-before-provenance-upgrade").exists()
    assert not (root / "provenance_upgrade_receipt.json").exists()


def test_upgrade_rejects_nonzero_propainter_pixels_even_with_rebound_report_hashes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cumulative"
    rounds = make_fixture(root)
    first = rounds[0]
    report = MODULE.read_json(first["report"])
    report["pixel_provenance"]["propainter_pixels"] = 1
    write_json(first["report"], report)
    report_sha = sha256_file(first["report"])
    report_receipt_path = first["report"].parent / "multiview_prefill_receipt.json"
    report_receipt = MODULE.read_json(report_receipt_path)
    report_receipt["report_sha256"] = report_sha
    write_json(report_receipt_path, report_receipt)
    summary = MODULE.read_json(root / "summary.json")
    summary["rounds"][0]["report_sha256"] = report_sha
    write_json(root / "summary.json", summary)

    with pytest.raises(ValueError, match="propainter_pixels is non-zero"):
        MODULE.upgrade(root, check_only=False)

    assert not (root / "metadata-before-provenance-upgrade").exists()
    assert not (root / "provenance_upgrade_receipt.json").exists()
