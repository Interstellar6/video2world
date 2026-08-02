from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.run_structured_3dfixer import configure_instances, install_bounded_preview


def test_bounded_preview_overrides_only_preview_arguments() -> None:
    calls = []

    def render(*args, **kwargs):
        calls.append((args, kwargs))
        return {"color": []}

    runner = SimpleNamespace(render_utils=SimpleNamespace(render_video=render))
    install_bounded_preview(runner, resolution=128, num_frames=4)
    result = runner.render_utils.render_video("gaussian", resolution=512, num_frames=12, r=2)
    assert result == {"color": []}
    assert calls[0][1] == {"resolution": 128, "num_frames": 4, "r": 2}


def test_bounded_preview_rejects_unsafe_values() -> None:
    runner = SimpleNamespace(render_utils=SimpleNamespace(render_video=lambda: None))
    with pytest.raises(ValueError, match="resolution"):
        install_bounded_preview(runner, resolution=32, num_frames=4)


def test_preview_skip_returns_declared_placeholder_without_calling_renderer() -> None:
    def unexpected_render(*args, **kwargs):
        raise AssertionError("upstream renderer must not run")

    runner = SimpleNamespace(render_utils=SimpleNamespace(render_video=unexpected_render))
    install_bounded_preview(
        runner,
        resolution=96,
        num_frames=4,
        skip_render=True,
    )
    result = runner.render_utils.render_video("gaussian", r=2)
    assert len(result["color"]) == 1
    assert result["color"][0].shape == (96, 96, 3)
    assert result["color"][0].min() == 255


def test_configure_instances_binds_current_mask(tmp_path) -> None:
    mask = tmp_path / "mask.png"
    mask.write_bytes(b"mask")
    result = configure_instances(
        [
            {
                "object_id": "central_white_pillow",
                "frame_id": 24,
                "mask_path": str(mask),
                "semantic_boundary": "Visible pillow pixels only.",
            }
        ]
    )
    assert result[0]["frame"] == "000024"
    assert result[0]["mask"] == mask
