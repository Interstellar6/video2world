from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image, ImageDraw

from scripts.run_responses_object_completion_batch import run_batch


def keyed_png() -> bytes:
    image = Image.new("RGB", (128, 128), (0, 255, 0))
    ImageDraw.Draw(image).rectangle((32, 24, 96, 112), fill=(120, 80, 40))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class FakeResponses:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            id=f"response-{len(self.calls)}",
            status="completed",
            output=[
                SimpleNamespace(
                    type="image_generation_call",
                    result=base64.b64encode(keyed_png()).decode("ascii"),
                )
            ],
        )


def test_object_batch_uses_responses_and_emits_alpha_bound_receipts(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(keyed_png())
    jobs = tmp_path / "jobs.json"
    jobs.write_text(
        json.dumps(
            {
                "objects": [
                    {
                        "object_id": "nightstand",
                        "source_image": str(source),
                        "prompt": "Complete the nightstand on a flat green background.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    matte = Path.home() / ".codex/skills/.system/imagegen/scripts/remove_chroma_key.py"
    fake = FakeResponses()
    receipt = run_batch(
        jobs,
        tmp_path / "output",
        client=SimpleNamespace(responses=fake),
        matte_script=matte,
        workers=1,
        max_attempts=1,
        created_at=datetime(2026, 8, 2, tzinfo=UTC),
    )

    assert receipt["protocol"] == "responses"
    assert receipt["promotion_allowed"] is False
    assert receipt["aggregate"]["object_count"] == 1
    assert receipt["aggregate"]["alpha_gate_passed_count"] == 1
    assert fake.calls[0]["model"] == "gpt-5.5"
    assert fake.calls[0]["reasoning"] == {"effort": "xhigh"}
    assert fake.calls[0]["tools"][0] == {
        "type": "image_generation",
        "model": "gpt-image-2",
        "action": "edit",
        "quality": "high",
        "size": "1024x1024",
        "output_format": "png",
    }
    item = receipt["objects"][0]
    assert item["review"]["automatic_accept"] is False
    assert item["matte"]["backend"] == "imagegen.remove_chroma_key"
    with Image.open(item["outputs"]["generated_reference"]["path"]) as image:
        assert image.mode == "RGBA"
        assert image.getchannel("A").getextrema() == (0, 255)
    with Image.open(item["outputs"]["generated_region_mask"]["path"]) as image:
        assert image.mode == "L"
        assert image.getextrema() == (0, 255)
    with Image.open(receipt["outputs"]["contact_sheet"]["path"]) as image:
        assert image.size == (256, 560)


def test_object_batch_resumes_hash_valid_outputs_without_an_api_call(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(keyed_png())
    jobs = tmp_path / "jobs.json"
    jobs.write_text(
        json.dumps(
            {
                "objects": [
                    {
                        "object_id": "bed",
                        "source_image": str(source),
                        "prompt": "Complete the bed on a green background.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    matte = Path.home() / ".codex/skills/.system/imagegen/scripts/remove_chroma_key.py"
    output = tmp_path / "output"
    first = FakeResponses()
    run_batch(jobs, output, client=SimpleNamespace(responses=first), matte_script=matte, workers=1)
    second = FakeResponses()
    run_batch(jobs, output, client=SimpleNamespace(responses=second), matte_script=matte, workers=1)
    assert len(first.calls) == 1
    assert second.calls == []
