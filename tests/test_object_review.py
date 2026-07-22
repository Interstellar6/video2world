from __future__ import annotations

import pytest

from video2world.object_review import (
    REQUIRED_CANONICAL_OBJECT_REVIEW_VIEWS,
    visual_gate_views_from_canonical_review,
)

SHA = "a" * 64


def _canonical_receipt() -> dict[str, object]:
    return {
        "kind": "video2world.canonical_object_six_view_review",
        "canonicalViews": list(REQUIRED_CANONICAL_OBJECT_REVIEW_VIEWS),
        "horizontalOrbitAcceptedAsSixViewEvidence": False,
        "views": {
            view: {
                "uri": f"{view}_object.png",
                "file": f"{view}_object.png",
                "path": f"/tmp/review/{view}_object.png",
                "sizeBytes": 1024,
                "sha256": str(index) * 64,
            }
            for index, view in enumerate(REQUIRED_CANONICAL_OBJECT_REVIEW_VIEWS, start=1)
        },
    }


def test_extracts_manifest_ready_visual_gate_views_from_canonical_review() -> None:
    evidence = visual_gate_views_from_canonical_review(_canonical_receipt())

    assert list(evidence) == list(REQUIRED_CANONICAL_OBJECT_REVIEW_VIEWS)
    assert evidence["front"] == {"uri": "front_object.png", "sha256": "1" * 64}
    assert all(set(view_evidence) == {"uri", "sha256"} for view_evidence in evidence.values())


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("orbit", "horizontal orbit"),
        ("missing_top", "missing 'top' evidence"),
        ("missing_uri", "front' evidence requires uri"),
        ("invalid_sha", "right' evidence requires SHA-256"),
    ],
)
def test_canonical_review_visual_gate_views_fail_closed(
    mutation: str,
    expected_error: str,
) -> None:
    receipt = _canonical_receipt()
    views = receipt["views"]
    assert isinstance(views, dict)
    if mutation == "orbit":
        receipt["horizontalOrbitAcceptedAsSixViewEvidence"] = True
    if mutation == "missing_top":
        del views["top"]
    if mutation == "missing_uri":
        front = views["front"]
        assert isinstance(front, dict)
        del front["uri"]
    if mutation == "invalid_sha":
        right = views["right"]
        assert isinstance(right, dict)
        right["sha256"] = "A" * 64

    with pytest.raises(ValueError, match=expected_error):
        visual_gate_views_from_canonical_review(receipt)


def test_sha_constant_is_lowercase_hex_for_test_fixtures() -> None:
    assert visual_gate_views_from_canonical_review(
        {
            **_canonical_receipt(),
            "views": {
                view: {"uri": f"{view}.png", "sha256": SHA}
                for view in REQUIRED_CANONICAL_OBJECT_REVIEW_VIEWS
            },
        }
    )["bottom"]["sha256"] == SHA
