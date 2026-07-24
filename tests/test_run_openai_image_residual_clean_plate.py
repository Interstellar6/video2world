from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from scripts.run_openai_image_residual_clean_plate import (
    OpenAIResidualCleanPlateError,
    run_openai_image_residual_clean_plate,
    sha256_file,
)

HEIGHT = 640
WIDTH = 1024
FRAME_ID = "000064"


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_rgb(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value).save(path)


def save_mask(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value.astype(np.uint8) * 255).save(path)


def png_response(value: np.ndarray) -> bytes:
    image = BytesIO()
    Image.fromarray(value).save(image, format="PNG")
    return json.dumps(
        {"data": [{"b64_json": base64.b64encode(image.getvalue()).decode("ascii")}]}
    ).encode("utf-8")


def make_fixture(
    tmp_path: Path,
    *,
    measured: bool = True,
    status: str = "technical_passed",
) -> Path:
    y, x = np.indices((HEIGHT, WIDTH))
    source = np.stack(
        [
            (x // 4) % 255,
            (y // 3) % 255,
            ((x + y) // 5) % 255,
        ],
        axis=2,
    ).astype(np.uint8)
    removal = np.zeros((HEIGHT, WIDTH), dtype=bool)
    removal[160:480, 256:768] = True
    residual = removal.copy()
    if measured:
        residual[220:420, 360:680] = False
    prefill = source.copy()
    prefill[removal & ~residual] = [35, 195, 75]

    source_path = tmp_path / "source.png"
    prefill_path = tmp_path / "prefill.png"
    removal_path = tmp_path / "removal.png"
    residual_path = tmp_path / "residual.png"
    save_rgb(source_path, source)
    save_rgb(prefill_path, prefill)
    save_mask(removal_path, removal)
    save_mask(residual_path, residual)
    record = {
        "sequence_index": 0,
        "frame_id": FRAME_ID,
        "source_frame": str(source_path),
        "source_frame_sha256": sha256_file(source_path)[0],
        "prefill_frame": str(prefill_path),
        "prefill_frame_sha256": sha256_file(prefill_path)[0],
        "removal_mask": str(removal_path),
        "removal_mask_sha256": sha256_file(removal_path)[0],
        "residual_mask": str(residual_path),
        "residual_mask_sha256": sha256_file(residual_path)[0],
        "removal_mask_pixels": int(removal.sum()),
        "residual_mask_pixels": int(residual.sum()),
        "covered_pixels": int((removal & ~residual).sum()),
        "measured_depth_valid_pixels": int((removal & ~residual).sum()),
        "outside_removal_mask_rgb_exact": True,
    }
    report_path = tmp_path / "multiview_prefill_report.json"
    write_json(
        report_path,
        {
            "schema_version": 2,
            "status": status,
            "promotion_approved": False,
            "pixel_provenance": {"generated_pixels": 0, "propainter_pixels": 0},
            "gates": {
                "outside_removal_mask_rgb_exact": True,
                "all_residual_masks_subset_of_removal_masks": True,
                "donor_support_not_boundary_concentrated": status == "technical_passed",
            },
            "frame_records": [record],
        },
    )
    return report_path


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) > 0


def fake_post_with_global_changes(
    *,
    expected_secret: str,
    returned: np.ndarray,
    observed: dict[str, Any],
):
    def post(
        url: str,
        payload: bytes,
        headers: dict[str, str],
        timeout_seconds: float,
    ) -> tuple[int, dict[str, str], bytes]:
        observed["url"] = url
        observed["payload"] = payload
        observed["headers"] = headers
        observed["timeout_seconds"] = timeout_seconds
        assert headers["Authorization"] == f"Bearer {expected_secret}"
        return 200, {"x-request-id": "req_fixture"}, png_response(returned)

    return post


def test_hard_composite_preserves_measurement_and_nonremoval_pixels(
    tmp_path: Path,
    monkeypatch,
) -> None:
    report = make_fixture(tmp_path)
    source = load_rgb(tmp_path / "source.png")
    prefill = load_rgb(tmp_path / "prefill.png")
    removal = load_mask(tmp_path / "removal.png")
    residual = load_mask(tmp_path / "residual.png")
    provider = np.full_like(source, [240, 20, 180])
    observed: dict[str, Any] = {}
    monkeypatch.setenv("V2W_TEST_OPENAI_KEY", "test-secret")

    receipt = run_openai_image_residual_clean_plate(
        prefill_report_path=report,
        frame_id=FRAME_ID,
        output_dir=tmp_path / "output",
        prompt="Restore only the background behind the removed object.",
        api_key_env="V2W_TEST_OPENAI_KEY",
        created_at=datetime(2026, 7, 24, 12, 0, tzinfo=UTC),
        http_post=fake_post_with_global_changes(
            expected_secret="test-secret",
            returned=provider,
            observed=observed,
        ),
    )

    candidate = load_rgb(tmp_path / "output" / "candidate.png")
    measured = removal & ~residual
    assert np.array_equal(candidate[residual], provider[residual])
    assert np.array_equal(candidate[measured], prefill[measured])
    assert np.array_equal(candidate[~removal], source[~removal])
    assert receipt["promotion_allowed"] is False
    assert receipt["review"]["geometry_reconstruction_allowed"] is False
    assert receipt["exactness"]["provider_changed_outside_residual_pixels_before_composite"] > 0
    assert receipt["pixel_provenance"]["measured_multiview_pixels_preserved"] == int(measured.sum())
    assert receipt["provider"]["response_request_id"] == "req_fixture"
    assert b'name="model"' in observed["payload"]
    assert b"gpt-image-2" in observed["payload"]
    assert b'name="image[]"' in observed["payload"]
    assert b'name="mask"' in observed["payload"]

    with Image.open(tmp_path / "output" / "residual_transparent_mask.png") as mask:
        alpha = np.asarray(mask.convert("RGBA"))[..., 3]
    assert np.all(alpha[residual] == 0)
    assert np.all(alpha[~residual] == 255)


def test_failed_prefill_can_only_continue_when_it_has_no_measured_pixels(
    tmp_path: Path,
    monkeypatch,
) -> None:
    report = make_fixture(tmp_path, measured=False, status="technical_failed")
    source = load_rgb(tmp_path / "source.png")
    monkeypatch.setenv("V2W_TEST_OPENAI_KEY", "test-secret")

    receipt = run_openai_image_residual_clean_plate(
        prefill_report_path=report,
        frame_id=FRAME_ID,
        output_dir=tmp_path / "output",
        prompt="Restore only the background behind the removed object.",
        api_key_env="V2W_TEST_OPENAI_KEY",
        http_post=fake_post_with_global_changes(
            expected_secret="test-secret",
            returned=np.full_like(source, [70, 80, 90]),
            observed={},
        ),
    )

    assert receipt["upstream_prefill"]["status"] == "technical_failed"
    assert receipt["pixel_provenance"]["measured_multiview_pixels_preserved"] == 0
    assert (
        receipt["pixel_provenance"]["generated_residual_pixels"]
        == receipt["pixel_provenance"]["removal_pixels"]
    )


def test_failed_prefill_with_measured_pixels_is_rejected_before_network(
    tmp_path: Path,
    monkeypatch,
) -> None:
    report = make_fixture(tmp_path, measured=True, status="technical_failed")
    monkeypatch.setenv("V2W_TEST_OPENAI_KEY", "test-secret")
    network_called = False

    def post(*_args: Any) -> tuple[int, dict[str, str], bytes]:
        nonlocal network_called
        network_called = True
        return 200, {}, b"{}"

    with pytest.raises(OpenAIResidualCleanPlateError, match="cannot contribute measured pixels"):
        run_openai_image_residual_clean_plate(
            prefill_report_path=report,
            frame_id=FRAME_ID,
            output_dir=tmp_path / "output",
            prompt="Restore only the background behind the removed object.",
            api_key_env="V2W_TEST_OPENAI_KEY",
            http_post=post,
        )

    assert network_called is False
    assert not (tmp_path / "output").exists()


def test_missing_api_key_fails_without_writing_output(tmp_path: Path, monkeypatch) -> None:
    report = make_fixture(tmp_path)
    monkeypatch.delenv("V2W_TEST_OPENAI_KEY", raising=False)

    with pytest.raises(OpenAIResidualCleanPlateError, match="V2W_TEST_OPENAI_KEY is not set"):
        run_openai_image_residual_clean_plate(
            prefill_report_path=report,
            frame_id=FRAME_ID,
            output_dir=tmp_path / "output",
            prompt="Restore only the background behind the removed object.",
            api_key_env="V2W_TEST_OPENAI_KEY",
        )

    assert not (tmp_path / "output").exists()
