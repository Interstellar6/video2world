from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = (
    Path(__file__).parents[1]
    / "examples"
    / "bedroom4"
    / "completion"
    / "layered-peel"
    / "layered-clean-plate-sequence-v1"
)

OBJECT_ORDER = (
    "sam3_pillow_front",
    "sam3_pillow_left",
    "sam3_pillow_right",
    "sam3_bed_01",
)
FRAME_IDS = tuple(f"{frame_id:06d}" for frame_id in range(48, 73))
FINAL_FRAME_SET_SHA256 = "854bd224ab59580f6d209fbb31a50059e0075007c2f6ef0c6d98cc8fde82dfca"

SEQUENCE_FILE_SHA256 = {
    "execution_receipt.json": "ab87e8b25d91fbfa071771522b70b1443990891b7c3ee0b04accdab53d35a007",
    "layered_clean_plate_sequence_report.json": (
        "672ccc38d2d0187a4e99e85a1f54a451039deab5c6d73eb81f71108275483e88"
    ),
    "layered_clean_plate_sequence_receipt.json": (
        "cdf2ffc02dbacf184391dafd1cd35c427fd5cf868b084548d4c7e2d7496d4d25"
    ),
}

ROUND_EVIDENCE = (
    {
        "round_name": "round01_front_pillow",
        "round_kind": "object_peel",
        "manifest_sha256": "111115d47388c1053437b71064bdd7abb250181b2dee0ed3988b901bb592da5b",
        "report_sha256": "92a33bfd4027cb0f4ce39fa5f7cfaa44f059d03832f4db47d0fa71b7c1c08c63",
        "receipt_sha256": "e1e4e09791322bb8e1625040bc078cc3e06a7125dfe16efb2ca0224117f323ea",
        "output_frame_set_sha256": (
            "42435b0a933078fe0b015ec32ca0d0efb4aa2829f9f69a613c8146ee8433ddd2"
        ),
        "status": "technical_passed_complete_partition",
    },
    {
        "round_name": "round02_left_pillow",
        "round_kind": "object_peel",
        "manifest_sha256": "7c8ffca06107f7acf9691c1d81430b8049d1273a9be6e2331743839c5fda9255",
        "report_sha256": "156df1ea29e770c59bc1e744547ee273b318af22403823ec22b3bbd42f087985",
        "receipt_sha256": "f7031350b367f8d7032fca2b187d3d5f9d5760159f3d5d722ca56c8b0a42e96f",
        "output_frame_set_sha256": (
            "feac1e67c06a15054ed0ff29beb279fff912421abec145e4bde6930740be17d1"
        ),
        "status": "technical_passed_complete_partition",
    },
    {
        "round_name": "round03_right_pillow",
        "round_kind": "object_peel",
        "manifest_sha256": "8971f7008c66a3b7c041edc463084910ba62fa6c7139707ecdbdcbb78cf86bd1",
        "report_sha256": "f9f3e6981e4c29fb3188c78e559000d3f5ae3508738c00ac384c671ff81bf1b9",
        "receipt_sha256": "e3f69d13a89542b0526717d3b86826324c220cddf085fd3755a5e0e3ef0d707f",
        "output_frame_set_sha256": (
            "e2230aa445ad049246b39049a92b53145547146d9cf1bfa05b6a43c136bd0f19"
        ),
        "status": "technical_passed_complete_partition",
    },
    {
        "round_name": "round04_bed",
        "round_kind": "final_background",
        "manifest_sha256": "cdc01a2524ff0dc0b36ce07cc25c89e26e149c2408a2c454e917dc8ec8528a5c",
        "report_sha256": "2c66055adf4afb7e39e3ffdaecd9dc91af15e2098e0fc51ca1a5c2efef9a1736",
        "receipt_sha256": "ff61be109c1fb2cc0725587fcaa32f27937d786bbd48128f85c8fa77137cc0be",
        "output_frame_set_sha256": FINAL_FRAME_SET_SHA256,
        "status": "technical_passed_complete_partition_with_limitations",
    },
)

MASK_SHA256 = {
    "000048": "eff72e75df3eaa5a14d52fb637a7be0e18e0c85f5d71c21019f237faa57d95dd",
    "000049": "94dc6dbdf7ce7fd4a9c3a729783230202e621c62a97a55e8a06a730213f1539b",
    "000050": "ae4ba29871a3761e2e8429599ac5f3ba347ecc38915cf128361ef84d8b6ac1c7",
    "000051": "40e5cbca60c2eef74d2c317d012f3efd16128789d39ebb8cecba003316aa522e",
    "000052": "aa85cfbf33e18562ac551ec27ec0e5c59948b236fd47699424ce2e96a52174b2",
    "000053": "a819007142f90df52bf3de28c11527fb870d472897fc3e47d82b293cdaeb319f",
    "000054": "e621b297ddf2c4127159f23522ee1fd73cd24ddbc7de7e9cc042163070dedcc5",
    "000055": "83742eb61b3377732812baf5c4afa76379ae0f075d76cbb5d691a6710380c95a",
    "000056": "a90396e20e5a1475738a329a7f6d9358cd4e9f30a9fc194cc69950d8ed630551",
    "000057": "64f9125cd73807ba785222540212082223c5466f9fdf531fc49b231fef9c222e",
    "000058": "a9a6365e0d552af5929fa3dd0a9781946d64c4fb7fb0315d4dda3542eaeb5446",
    "000059": "cd2917be4151b09563ed28a66debffcc56e036c0c1b65c4b3c56721149a39433",
    "000060": "c6e741a0234292622c1c88cfb0c354545f558e74759c79a382624d1af2d05b70",
    "000061": "4bfe686c4a6fc67c41b0db5bd9d893853310daa0da8931515cd391ef5bc00fe0",
    "000062": "12b64ccf1d96b8f9f5d823277219e4047deb9c4020b0a09631bb56a7b0a37b19",
    "000063": "17c6f7fccd90957f45c57364aad575be749f9e55ae5bf8d18a25a22689c2cbb2",
    "000064": "bfa688313d2940b3a3d14e9f4211b90987a49f681e44b782160faf4d40c75f1c",
    "000065": "f430746b85e3eb852c5a615a749c1ef27c7d2afc4e4a3de3f47d39c48b5286ee",
    "000066": "fa8e3ceed8277ed140defc14c4ad413ff2a35175540b119ec1d302de02c3ee59",
    "000067": "3ccf88d8ac80dd1b7d7e39fedfdece9801c028efd8485e048cb00fddae8c58b0",
    "000068": "24dbbc5102a71968e9156ba170b7a9d44c72d270d9a38b0037e5e71f9b3f2101",
    "000069": "22fe4bb9852c30b136f00d6bfc4daeb8df5c8bef5c2e9f54c3bd67734bf6f465",
    "000070": "6987a8fa4cc1c0242b23b5e70d066d9f64e4bfeac9ef7f6087ad8fe19bf93051",
    "000071": "971a244f4b2653638d255e4048039957aa65f676d48f3818d6f2d118cd4bfd83",
    "000072": "0556f250f33fc4ced4acfc7dc7ad59f06a7ebd59018cd557282143825f5e6a3f",
}

CONTACT_SHEET_SHA256 = {
    "source": (
        "source_contact_sheet.png",
        "c18e229a9b1c9b04bcb044039d9dcdb67d78c9631ce965131b47df67e4869b71",
    ),
    "round01_front_pillow": (
        "round01_contact_sheet.png",
        "fdbdc172137391bf46890ceeed026f3b9a678e7271353db5e44b9133a8749066",
    ),
    "round02_left_pillow": (
        "round02_contact_sheet.png",
        "07403a3fac576710b9fdaec3956091e623051fdb6080ac1115daf01ded4b8aa3",
    ),
    "round03_right_pillow": (
        "round03_contact_sheet.png",
        "f30829672c48ca826c2fce37a6f4f9c01d7504e9f8393567a261cea01e7477a6",
    ),
    "round04_bed": (
        "round04_contact_sheet.png",
        "4118d32e18c7151dc2013036c07a7dc2229e039730c01098c5e04684767fa7a0",
    ),
}


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


def assert_local_artifact(root: Path, record: dict[str, Any]) -> Path:
    relative = Path(str(record["path"]))
    assert not relative.is_absolute()
    assert ".." not in relative.parts
    path = root / relative
    assert path.is_file()
    assert sha256_file(path) == record["sha256"]
    return path


def test_sequence_receipts_bind_the_exact_four_round_front_to_back_execution() -> None:
    for file_name, expected_sha256 in SEQUENCE_FILE_SHA256.items():
        assert sha256_file(ROOT / file_name) == expected_sha256

    report = read_json(ROOT / "layered_clean_plate_sequence_report.json")
    receipt = read_json(ROOT / "layered_clean_plate_sequence_receipt.json")
    execution = read_json(ROOT / "execution_receipt.json")

    assert report["kind"] == "video2world.layered_clean_plate_sequence_report"
    assert report["status"] == "technical_passed_complete_sequence_with_limitations"
    assert report["frame_count"] == 25
    assert tuple(report["removed_object_order"]) == OBJECT_ORDER
    assert report["final_output_frame_set_sha256"] == FINAL_FRAME_SET_SHA256
    assert report["acceptance_scope"] == "current_demo_only"
    assert report["promotion_approved"] is False
    assert report["accepted_with_limitations"] is True
    assert all(report["gates"].values())

    expected_remaining = {
        evidence["round_name"]: list(OBJECT_ORDER[index + 1 :])
        for index, evidence in enumerate(ROUND_EVIDENCE)
    }
    assert report["round_remaining_object_ids"] == expected_remaining
    assert len(report["rounds"]) == len(ROUND_EVIDENCE) == 4
    assert len(execution["rounds"]) == len(ROUND_EVIDENCE)

    previous: dict[str, Any] | None = None
    for index, (evidence, summary_round, execution_round) in enumerate(
        zip(ROUND_EVIDENCE, report["rounds"], execution["rounds"], strict=True),
        start=1,
    ):
        round_name = str(evidence["round_name"])
        removed = list(OBJECT_ORDER[:index])
        remaining = list(OBJECT_ORDER[index:])
        expected_frame_set_sha256 = evidence["output_frame_set_sha256"]

        assert summary_round["round_index"] == execution_round["round_index"] == index
        assert summary_round["round_name"] == round_name
        assert summary_round["round_kind"] == evidence["round_kind"]
        assert summary_round["status"] == evidence["status"]
        assert (
            summary_round["removed_object_ids"]
            == execution_round["removed_object_ids"]
            == removed
        )
        assert (
            summary_round["remaining_object_ids"]
            == execution_round["remaining_object_ids"]
            == remaining
        )
        assert summary_round["unresolved_pixels"] == execution_round["unresolved_pixels"] == 0
        assert summary_round["output_frame_set_sha256"] == expected_frame_set_sha256
        assert execution_round["output_frame_set_sha256"] == expected_frame_set_sha256

        manifest_path = assert_local_artifact(ROOT, summary_round["manifest"])
        round_report_path = assert_local_artifact(ROOT, summary_round["report"])
        round_receipt_path = assert_local_artifact(ROOT, summary_round["receipt"])
        assert summary_round["manifest"]["sha256"] == evidence["manifest_sha256"]
        assert summary_round["report"]["sha256"] == evidence["report_sha256"]
        assert summary_round["receipt"]["sha256"] == evidence["receipt_sha256"]
        assert execution_round["report_sha256"] == evidence["report_sha256"]
        assert execution_round["receipt_sha256"] == evidence["receipt_sha256"]

        manifest = read_json(manifest_path)
        round_report = read_json(round_report_path)
        round_receipt = read_json(round_receipt_path)
        assert manifest["kind"] == "video2world.layered_clean_plate_compositor_input"
        assert round_report["kind"] == "video2world.layered_clean_plate_composite_report"
        assert round_receipt["kind"] == "video2world.layered_clean_plate_composite_receipt"
        for document in (manifest, round_report):
            assert document["round_index"] == index
            assert document["round_kind"] == evidence["round_kind"]
            assert document["removed_object_ids"] == removed
            assert document["remaining_object_ids"] == remaining
            assert document["newly_removed_object_id"] == OBJECT_ORDER[index - 1]

        assert len(manifest["frame_records"]) == len(round_report["frame_records"]) == 25
        assert [record["frame_id"] for record in manifest["frame_records"]] == list(FRAME_IDS)
        assert [record["frame_id"] for record in round_report["frame_records"]] == list(FRAME_IDS)
        assert round_report["status"] == evidence["status"]
        assert round_report["input_manifest_sha256"] == evidence["manifest_sha256"]
        assert round_report["output_frame_set_sha256"] == expected_frame_set_sha256
        assert round_report["aggregate_counts"]["unresolved_pixels"] == 0
        assert all(
            record["counts"]["unresolved_pixels"] == 0
            for record in round_report["frame_records"]
        )
        assert round_report["promotion_approved"] is False

        assert round_receipt["round_index"] == index
        assert round_receipt["round_kind"] == evidence["round_kind"]
        assert round_receipt["report"] == "layered_composite_report.json"
        assert round_receipt["report_sha256"] == evidence["report_sha256"]
        assert round_receipt["input_manifest_sha256"] == evidence["manifest_sha256"]
        assert round_receipt["output_frame_set_sha256"] == expected_frame_set_sha256
        assert round_receipt["provenance_partition_exact"] is True
        assert round_receipt["promotion_approved"] is False

        if previous is None:
            assert manifest["previous_round"] is None
            assert round_report["previous_round_binding"] is None
            assert round_receipt["previous_round_output_report_sha256"] is None
            assert round_receipt["previous_round_output_receipt_sha256"] is None
        else:
            expected_previous = {
                "round_index": index - 1,
                "report_sha256": previous["report_sha256"],
                "receipt_sha256": previous["receipt_sha256"],
                "output_frame_set_sha256": previous["output_frame_set_sha256"],
            }
            assert manifest["source_contract"] == {
                "propainter_rgb": False,
                "role": "previous_layered_composite",
                "untracked_generated_rgb": False,
            }
            assert manifest["previous_round"]["round_index"] == index - 1
            assert manifest["previous_round"]["report"]["sha256"] == previous["report_sha256"]
            assert manifest["previous_round"]["receipt"]["sha256"] == previous["receipt_sha256"]
            binding = round_report["previous_round_binding"]
            for key, expected_value in expected_previous.items():
                assert binding[key] == expected_value
            assert round_receipt["previous_round_output_report_sha256"] == previous["report_sha256"]
            assert (
                round_receipt["previous_round_output_receipt_sha256"]
                == previous["receipt_sha256"]
            )

        previous = evidence

    assert receipt["kind"] == "video2world.layered_clean_plate_sequence_receipt"
    assert receipt["report"] == "layered_clean_plate_sequence_report.json"
    assert receipt["report_sha256"] == SEQUENCE_FILE_SHA256[receipt["report"]]
    assert receipt["final_round_report"]["sha256"] == ROUND_EVIDENCE[-1]["report_sha256"]
    assert receipt["final_round_receipt"]["sha256"] == ROUND_EVIDENCE[-1]["receipt_sha256"]
    assert receipt["final_output_frame_set_sha256"] == FINAL_FRAME_SET_SHA256
    assert receipt["all_round_manifests_reports_and_receipts_are_hash_bound"] is True
    assert receipt["acceptance_scope"] == "current_demo_only"
    assert receipt["promotion_approved"] is False

    assert execution["kind"] == "video2world.layered_clean_plate_sequence_execution_receipt"
    assert execution["status"] == report["status"] == receipt["status"]
    assert execution["acceptance_scope"] == "current_demo_only"
    assert execution["promotion_approved"] is False
    assert all(execution["gates"].values())
    assert execution["outputs"]["sequence_report"]["sha256"] == SEQUENCE_FILE_SHA256[
        "layered_clean_plate_sequence_report.json"
    ]
    assert execution["outputs"]["sequence_receipt"]["sha256"] == SEQUENCE_FILE_SHA256[
        "layered_clean_plate_sequence_receipt.json"
    ]
    assert execution["outputs"]["final_output_frame_set_sha256"] == FINAL_FRAME_SET_SHA256


def test_round4_portable_mask_index_hashes_all_25_local_masks() -> None:
    index_path = (
        ROOT
        / "normalized_inputs"
        / "round04_cumulative_masks"
        / "cumulative_removal_mask_index.json"
    )
    assert sha256_file(index_path) == (
        "4fb4abdf232c274ac0781a51e3b94c1cabc3bae545bbd47773994892d71a4f3c"
    )
    index = read_json(index_path)
    assert index["kind"] == "video2world.cumulative_removal_mask_index"
    assert index["copy_mode"] == "byte_exact_from_hash_verified_source_mask"
    assert index["round_index"] == 4
    assert index["frame_count"] == 25
    assert tuple(index["frame_ids"]) == FRAME_IDS
    assert tuple(index["removed_object_ids"]) == OBJECT_ORDER
    assert set(index["frames"]) == set(MASK_SHA256) == set(FRAME_IDS)

    mask_root = index_path.parent
    for sequence_index, frame_id in enumerate(FRAME_IDS):
        record = index["frames"][frame_id]
        expected_sha256 = MASK_SHA256[frame_id]
        assert record["sequence_index"] == sequence_index
        assert record["mask"] == {
            "path": f"masks/{sequence_index:04d}.png",
            "sha256": expected_sha256,
        }
        assert record["source_mask"]["sha256"] == expected_sha256
        assert_local_artifact(mask_root, record["mask"])

    assert sorted(path.name for path in (mask_root / "masks").glob("*.png")) == [
        f"{sequence_index:04d}.png" for sequence_index in range(25)
    ]

    report = read_json(ROOT / "layered_clean_plate_sequence_report.json")
    execution = read_json(ROOT / "execution_receipt.json")
    assert report["inputs"]["round4_cumulative_masks"]["sha256"] == sha256_file(index_path)
    assert index["source_index"]["sha256"] == execution["inputs"][
        "structural_background_index_sha256"
    ]


def test_visual_review_hashes_five_local_contact_sheets_and_forbids_promotion() -> None:
    review_path = ROOT / "qa" / "visual_review.json"
    assert sha256_file(review_path) == (
        "41decd64d22163d88c026241f6874290aaa1a52465141153065b7d0c71d9314c"
    )
    review = read_json(review_path)
    assert review["kind"] == "video2world.layered_clean_plate_sequence_visual_review"
    assert review["status"] == "accepted_for_current_demo_with_limitations"
    assert review["frame_range"] == {"first": "000048", "last": "000072", "count": 25}
    assert set(review["contact_sheets"]) == set(CONTACT_SHEET_SHA256)

    for label, (file_name, expected_sha256) in CONTACT_SHEET_SHA256.items():
        record = review["contact_sheets"][label]
        assert record == {"path": file_name, "sha256": expected_sha256}
        assert_local_artifact(review_path.parent, record)

    assert sorted(path.name for path in review_path.parent.glob("*_contact_sheet.png")) == sorted(
        file_name for file_name, _ in CONTACT_SHEET_SHA256.values()
    )
    assert review["decision"] == {
        "acceptance_scope": "current_demo_only",
        "accepted_with_limitations": True,
        "promotion_approved": False,
        "eligible_for_fresh_da3_pgsr_tsdf": True,
        "eligible_as_general_background_completion_quality_evidence": False,
    }

    for file_name in (
        "execution_receipt.json",
        "layered_clean_plate_sequence_report.json",
        "layered_clean_plate_sequence_receipt.json",
    ):
        document = read_json(ROOT / file_name)
        assert document["acceptance_scope"] == "current_demo_only"
        assert document["promotion_approved"] is False
