from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from video2world.providers import qwen_geometry_review
from video2world.providers.qwen_geometry_review import (
    SIX_VIEW_IDS,
    appearance_gates,
    build_review,
    issue_from_value,
    technical_gates,
)


def mesh_report(*, extents: list[float], components: int = 1) -> dict[str, object]:
    base = {
        "vertices": 100,
        "faces": 200,
        "finite_vertices": True,
        "finite_faces": True,
        "extents": extents,
        "connected_components": components,
        "watertight": True,
    }
    return {"raw_mesh": base, "processed_mesh": dict(base)}


def canonical_review_evidence() -> dict[str, object]:
    return {
        "receipt_uri": "artifact://object_six_view_review.json",
        "receipt_sha256": "a" * 64,
        "views": {
            view: {"uri": f"{view}_object.png", "sha256": str(index) * 64}
            for index, view in enumerate(SIX_VIEW_IDS, start=1)
        },
    }


def test_pillow_thickness_gate_rejects_sheet_geometry() -> None:
    gates = technical_gates("pillow", mesh_report(extents=[0.49, 0.84, 0.13]))
    assert gates["pillow_thickness_ratio"] is False


def test_vlm_accept_is_overridden_by_failed_technical_gate(tmp_path: Path) -> None:
    asset = tmp_path / "asset.glb"
    turntable = tmp_path / "turntable.png"
    asset.write_bytes(b"asset")
    turntable.write_bytes(b"turntable")
    review = build_review(
        object_id="pillow-right",
        attempt=1,
        model_name="qwen",
        source_asset=asset,
        turntable=turntable,
        raw_response=('{"decision":"accept","issues":[],"retry_prompt":null}'),
        gates={"finite_geometry": True, "pillow_thickness_ratio": False},
    )
    assert review.decision == "retry"
    assert any(issue.issue_type == "implausible_thickness" for issue in review.issues)
    assert review.retry_prompt


def test_invalid_vlm_response_fails_closed(tmp_path: Path) -> None:
    asset = tmp_path / "asset.glb"
    turntable = tmp_path / "turntable.png"
    asset.write_bytes(b"asset")
    turntable.write_bytes(b"turntable")
    review = build_review(
        object_id="pillow-front",
        attempt=1,
        model_name="qwen",
        source_asset=asset,
        turntable=turntable,
        raw_response="not json",
        gates={"finite_geometry": True},
    )
    assert review.decision == "retry"
    assert review.retry_prompt


def test_missing_back_surface_warning_is_upgraded_to_blocking() -> None:
    issue = issue_from_value(
        {
            "issue_type": "missing_back_surface",
            "severity": "warning",
            "evidence_view_ids": ["source|back"],
            "explanation": {"en": "Missing back.", "zh": "Missing back."},
            "retry_prompt_instruction": "Reconstruct the back.",
        }
    )
    assert issue is not None
    assert issue.severity == "blocking"
    assert issue.evidence_view_ids == ["source", "back"]


def test_minor_texture_hallucination_remains_a_warning() -> None:
    issue = issue_from_value(
        {
            "issue_type": "texture_hallucination",
            "severity": "warning",
            "evidence_view_ids": ["back"],
            "explanation": {
                "en": "Small invented pattern on the hidden back.",
                "zh": "隐藏背面有少量生成纹理。",
            },
        }
    )

    assert issue is not None
    assert issue.severity == "warning"
    assert issue.retry_prompt_instruction is None


def test_warning_only_review_is_accepted_with_advisory_limitations(tmp_path: Path) -> None:
    asset = tmp_path / "asset.glb"
    turntable = tmp_path / "turntable.png"
    asset.write_bytes(b"asset")
    turntable.write_bytes(b"turntable")
    review = build_review(
        object_id="pillow-front",
        attempt=1,
        model_name="qwen",
        source_asset=asset,
        turntable=turntable,
        raw_response=json.dumps(
            {
                "decision": "retry",
                "issues": [
                    {
                        "issue_type": "texture_hallucination",
                        "severity": "warning",
                        "evidence_view_ids": ["back"],
                        "explanation": {
                            "en": "Minor hidden-side pattern drift.",
                            "zh": "隐藏侧有轻微纹理偏差。",
                        },
                        "retry_prompt_instruction": None,
                    }
                ],
                "retry_prompt": "Regenerate the back texture.",
            }
        ),
        gates={
            "finite_geometry": True,
            "raw_surface_closed": False,
            "six_view_color_style_consistency": False,
        },
        canonical_six_view_review=canonical_review_evidence(),
    )

    assert review.decision == "accept"
    assert review.technical_gates == {"finite_geometry": True}
    assert review.advisory_gates == {
        "raw_surface_closed": False,
        "six_view_color_style_consistency": False,
    }
    assert review.retry_prompt is None
    assert all(issue.severity == "warning" for issue in review.issues)


def _write_rgba(path: Path, color: tuple[int, int, int]) -> None:
    from PIL import Image

    Image.new("RGBA", (32, 32), (*color, 255)).save(path)


def test_six_view_color_gate_rejects_pink_back_of_white_pillow(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    material = tmp_path / "material.png"
    view_dir = tmp_path / "views"
    view_dir.mkdir()
    _write_rgba(source, (210, 212, 214))
    _write_rgba(material, (205, 207, 209))
    for view_id in SIX_VIEW_IDS:
        color = (166, 91, 82) if view_id == "back" else (202, 205, 208)
        _write_rgba(view_dir / f"{view_id}_object.png", color)

    gates, metrics = appearance_gates(
        source,
        view_dir / "front_object.png",
        material,
        mesh_report(extents=[1.0, 0.94, 0.30]),
        view_render_dir=view_dir,
        render_mode="neutral-albedo",
        expected_tone="white",
    )

    assert gates["front_color_fidelity"] is True
    assert gates["neutral_albedo_render_evidence"] is True
    assert gates["expected_white_neutral_appearance"] is False
    assert gates["six_view_color_style_consistency"] is False
    assert metrics["six_views"]["back"]["mean_chroma_range"] > 48


def test_six_view_color_gate_accepts_consistent_neutral_white(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    material = tmp_path / "material.png"
    view_dir = tmp_path / "views"
    view_dir.mkdir()
    _write_rgba(source, (210, 212, 214))
    _write_rgba(material, (205, 207, 209))
    for index, view_id in enumerate(SIX_VIEW_IDS):
        _write_rgba(view_dir / f"{view_id}_object.png", (202 - index, 204 - index, 206 - index))

    gates, _ = appearance_gates(
        source,
        view_dir / "front_object.png",
        material,
        mesh_report(extents=[1.0, 0.94, 0.30]),
        view_render_dir=view_dir,
        render_mode="neutral-albedo",
        expected_tone="white",
    )

    assert all(gates.values())


@pytest.mark.parametrize(
    ("decision", "expected_exit_code"),
    [("accept", 0), ("retry", 3), ("reject", 3)],
)
def test_main_exit_code_reflects_geometry_review_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decision: str,
    expected_exit_code: int,
) -> None:
    source_asset = tmp_path / "asset.glb"
    turntable = tmp_path / "turntable.png"
    canonical_review_receipt = tmp_path / "object_six_view_review.json"
    mesh_report_path = tmp_path / "mesh-report.json"
    source_asset.write_bytes(b"asset")
    turntable.write_bytes(b"turntable")
    canonical_review_receipt.write_text(
        json.dumps(
            {
                "kind": "video2world.canonical_object_six_view_review",
                "canonicalViews": ["front", "right", "back", "left", "top", "bottom"],
                "horizontalOrbitAcceptedAsSixViewEvidence": False,
                "views": {
                    view: {"uri": f"{view}_object.png", "sha256": str(index) * 64}
                    for index, view in enumerate(
                        ["front", "right", "back", "left", "top", "bottom"],
                        start=1,
                    )
                },
            }
        ),
        encoding="utf-8",
    )
    mesh_report_path.write_text(
        json.dumps(mesh_report(extents=[1.0, 0.9, 0.3])),
        encoding="utf-8",
    )
    output_dir = tmp_path / f"output-{decision}"
    args = Namespace(
        object_id="pillow-front",
        category="pillow",
        attempt=1,
        source_image=tmp_path / "source.png",
        front_render_image=tmp_path / "front.png",
        view_render_dir=tmp_path / "views",
        front_render_mode="neutral-albedo",
        expected_tone="white",
        turntable_image=turntable,
        canonical_review_receipt=canonical_review_receipt,
        source_asset=source_asset,
        material_image=tmp_path / "material.png",
        mesh_report=mesh_report_path,
        output_dir=output_dir,
        model_path=tmp_path / "qwen-model",
        gpu_devices="",
        attention_backend="sdpa",
        max_new_tokens=128,
    )
    raw_response: dict[str, object] = {
        "decision": decision,
        "issues": [],
        "retry_prompt": None,
    }
    if decision == "retry":
        raw_response["issues"] = [
            {
                "issue_type": "missing_back_surface",
                "severity": "blocking",
                "evidence_view_ids": ["back"],
                "explanation": {"en": "Back is missing.", "zh": "背面缺失。"},
                "retry_prompt_instruction": "Reconstruct the complete back surface.",
            }
        ]
        raw_response["retry_prompt"] = "Reconstruct the complete back surface."

    monkeypatch.setattr(qwen_geometry_review, "parse_args", lambda: args)
    monkeypatch.setattr(
        qwen_geometry_review,
        "appearance_gates",
        lambda *args, **kwargs: ({"appearance_fidelity": True}, {}),
    )
    monkeypatch.setattr(qwen_geometry_review, "load_qwen", lambda *args: (object(), object()))
    monkeypatch.setattr(
        qwen_geometry_review,
        "run_qwen",
        lambda *args: json.dumps(raw_response),
    )

    assert qwen_geometry_review.main() == expected_exit_code
    saved = json.loads((output_dir / "geometry_review.json").read_text(encoding="utf-8"))
    assert saved["decision"] == decision
