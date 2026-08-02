from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image

from scripts.run_responses_clean_plate_batch import (
    extract_generated_png,
    run_responses_clean_plate_batch,
    sha256_file,
)


def png_bytes(color: tuple[int, int, int], size: tuple[int, int] = (17, 11)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


class FakeResponses:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        index = len(self.calls)
        payload = base64.b64encode(png_bytes((index, 20, 30))).decode("ascii")
        return SimpleNamespace(
            id=f"response-{index}",
            status="completed",
            output=[SimpleNamespace(type="image_generation_call", result=payload)],
        )


def write_frames(root: Path, count: int = 3) -> None:
    root.mkdir()
    for index in range(count):
        (root / f"{index:06d}.png").write_bytes(png_bytes((index * 20, 2, 3)))


def test_extract_generated_png_supports_dict_and_object_outputs() -> None:
    payload = png_bytes((1, 2, 3))
    encoded = base64.b64encode(payload).decode("ascii")
    assert (
        extract_generated_png(
            {"output": [{"type": "image_generation_call", "result": encoded}]}
        )
        == payload
    )
    assert extract_generated_png(
        SimpleNamespace(output=[SimpleNamespace(type="image_generation_call", result=encoded)])
    ) == payload


def test_batch_uses_responses_edit_tool_and_writes_review_only_receipts(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    write_frames(inputs)
    fake = FakeResponses()
    output = tmp_path / "output"
    receipt = run_responses_clean_plate_batch(
        inputs,
        output,
        anchor_frame="000001",
        prompt="Follow the clean anchor but preserve the target camera.",
        anchor_prompt="Create the accepted clean anchor.",
        client=SimpleNamespace(responses=fake),
        workers=1,
        max_attempts=1,
        created_at=datetime(2026, 8, 2, tzinfo=UTC),
    )

    assert receipt["protocol"] == "responses"
    assert receipt["promotion_allowed"] is False
    assert receipt["aggregate"]["frame_count"] == 3
    assert receipt["aggregate"]["fresh_response_count"] == 3
    assert receipt["aggregate"]["all_normalized_outputs_exact_size"] is True
    assert len(fake.calls) == 3
    assert all(call["model"] == "gpt-5.5" for call in fake.calls)
    assert all(call["reasoning"] == {"effort": "xhigh"} for call in fake.calls)
    assert all(call["tools"][0]["type"] == "image_generation" for call in fake.calls)
    assert all(call["tools"][0]["model"] == "gpt-image-2" for call in fake.calls)
    anchor_call = fake.calls[0]
    assert len(anchor_call["input"][0]["content"]) == 4
    assert len(fake.calls[1]["input"][0]["content"]) == 5

    for frame in receipt["frames"]:
        raw = Path(frame["outputs"]["raw"]["path"])
        normalized = Path(frame["outputs"]["camera_calibrated"]["path"])
        assert sha256_file(raw)[0] == frame["outputs"]["raw"]["sha256"]
        with Image.open(normalized) as image:
            assert image.size == (1280, 720)
            assert image.mode == "RGB"
    written = json.loads((output / "batch_receipt.json").read_text(encoding="utf-8"))
    assert written == receipt


def test_verified_seed_is_adopted_without_remote_call(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    write_frames(inputs, count=1)
    seed = tmp_path / "seed"
    seed.mkdir()
    image_path = seed / "000000.png"
    image_path.write_bytes(png_bytes((8, 9, 10), (31, 19)))
    digest, size = sha256_file(image_path)
    (seed / "000000.response.json").write_text(
        json.dumps(
            {
                "protocol": "responses",
                "response_id": "old-response",
                "response_status": "completed",
                "output": {"sha256": digest, "bytes": size},
            }
        ),
        encoding="utf-8",
    )
    fake = FakeResponses()
    receipt = run_responses_clean_plate_batch(
        inputs,
        tmp_path / "output",
        anchor_frame="000000",
        prompt="unused",
        anchor_prompt="anchor",
        client=SimpleNamespace(responses=fake),
        seed_dir=seed,
        workers=1,
    )
    assert fake.calls == []
    assert receipt["aggregate"]["verified_seed_count"] == 1
    assert receipt["frames"][0]["origin"] == "verified_seed_candidate"
