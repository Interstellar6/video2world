from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from scripts.run_diffusion_anchor_inpaint import (
    DiffusionAnchorInpaintError,
    LoadedPipeline,
    build_parser,
    load_local_inpaint_pipeline,
    load_source_and_mask,
    run_diffusion_anchor_inpaint,
    sha256_file,
    validate_generation_parameters,
    validate_seeds,
)

MODEL_REPO = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
MODEL_REVISION = "115134f363124c53c7d878647567d04daf26e41e"
PROMPT = "Restore only the missing wall texture with continuous geometry."
NEGATIVE_PROMPT = "object, text, seam, blur"


class FakeInpaintPipeline:
    def __init__(self, *, preserve_inside: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self.preserve_inside = preserve_inside

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        source = np.asarray(kwargs["image"], dtype=np.uint8)
        mask = np.asarray(kwargs["mask_image"], dtype=np.uint8) > 0
        candidate = np.full_like(source, [250, 3, 177])
        if self.preserve_inside:
            candidate[mask] = source[mask]
        else:
            seed = int(kwargs["generator"])
            candidate[mask] = [20 + seed % 100, 90, 140]
        return SimpleNamespace(images=[Image.fromarray(candidate)])


def make_fixture(root: Path) -> tuple[Path, Path, Path, np.ndarray, np.ndarray]:
    source = np.zeros((8, 10, 3), dtype=np.uint8)
    source[..., 0] = np.arange(10, dtype=np.uint8)[None, :] * 10
    source[..., 1] = np.arange(8, dtype=np.uint8)[:, None] * 15
    source[..., 2] = 60
    residual = np.zeros((8, 10), dtype=np.uint8)
    residual[2:6, 3:8] = 255
    source_path = root / "source.png"
    mask_path = root / "residual.png"
    Image.fromarray(source).save(source_path)
    Image.fromarray(residual).save(mask_path)

    model_dir = root / "model"
    (model_dir / "unet").mkdir(parents=True)
    (model_dir / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusionXLInpaintPipeline"}),
        encoding="utf-8",
    )
    (model_dir / "unet" / "diffusion_pytorch_model.fp16.safetensors").write_bytes(
        b"fixed-local-test-weights"
    )
    return source_path, mask_path, model_dir, source, residual > 0


def test_fake_pipeline_preserves_outside_exactly_and_writes_auditable_receipt(
    tmp_path: Path,
) -> None:
    source_path, mask_path, model_dir, source, residual = make_fixture(tmp_path)
    pipeline = FakeInpaintPipeline()

    def fake_loader(**kwargs: Any) -> LoadedPipeline:
        assert kwargs == {
            "model_dir": model_dir.resolve(),
            "pipeline_kind": "sdxl",
            "device": "cuda:test",
            "dtype": "float16",
            "variant": "fp16",
            "use_safetensors": True,
        }
        return LoadedPipeline(
            pipeline=pipeline,
            runtime={"pipeline_class": "FakeInpaintPipeline", "test_double": True},
        )

    output_dir = tmp_path / "output"
    receipt = run_diffusion_anchor_inpaint(
        source_rgb_path=source_path,
        residual_mask_path=mask_path,
        output_dir=output_dir,
        model_dir=model_dir,
        model_repo=MODEL_REPO,
        model_revision=MODEL_REVISION,
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        seeds=[40, 41, 42],
        pipeline_kind="sdxl",
        num_inference_steps=28,
        guidance_scale=6.25,
        strength=0.95,
        device="cuda:test",
        created_at=datetime(2026, 7, 17, 12, 30, tzinfo=UTC),
        pipeline_loader=fake_loader,
        generator_factory=lambda seed: seed,
    )

    assert [call["generator"] for call in pipeline.calls] == [40, 41, 42]
    assert all(call["num_inference_steps"] == 28 for call in pipeline.calls)
    assert all(call["guidance_scale"] == 6.25 for call in pipeline.calls)
    assert all(call["height"] == 8 for call in pipeline.calls)
    assert all(call["width"] == 10 for call in pipeline.calls)
    assert all(call["prompt"] == PROMPT for call in pipeline.calls)
    assert all(call["negative_prompt"] == NEGATIVE_PROMPT for call in pipeline.calls)

    for seed in (40, 41, 42):
        candidate_path = output_dir / f"candidate_seed_{seed}.png"
        candidate = np.asarray(Image.open(candidate_path).convert("RGB"), dtype=np.uint8)
        assert np.array_equal(candidate[~residual], source[~residual])
        assert not np.array_equal(candidate[residual], source[residual])

    written = json.loads((output_dir / "run_receipt.json").read_text(encoding="utf-8"))
    assert written == receipt
    assert receipt["status"] == "generated_candidates_pending_review"
    assert receipt["promotion_allowed"] is False
    assert receipt["provider"] == "local_huggingface_diffusers"
    assert receipt["boundary"] == {
        "automatic_accept": False,
        "human_review_required": True,
        "vlm_called": False,
        "published": False,
        "outside_mask_policy": "forced_exact_source_rgb_composite",
    }
    assert receipt["model"]["repo"] == MODEL_REPO
    assert receipt["model"]["revision"] == MODEL_REVISION
    assert receipt["model"]["local_files_only"] is True
    assert receipt["model"]["variant"] == "fp16"
    assert receipt["model"]["use_safetensors"] is True
    assert set(receipt["model"]["files"]) == {
        "model_index.json",
        "unet/diffusion_pytorch_model.fp16.safetensors",
    }
    model_weight = model_dir / "unet" / "diffusion_pytorch_model.fp16.safetensors"
    assert (
        receipt["model"]["files"]["unet/diffusion_pytorch_model.fp16.safetensors"]["sha256"]
        == sha256_file(model_weight)[0]
    )
    assert len(receipt["model"]["file_manifest_sha256"]) == 64
    assert receipt["generation"]["seeds"] == [40, 41, 42]
    assert receipt["generation"]["num_inference_steps"] == 28
    assert receipt["generation"]["guidance_scale"] == 6.25
    assert receipt["generation"]["loader"] == {
        "local_files_only": True,
        "torch_dtype": "float16",
        "use_safetensors": True,
        "variant": "fp16",
    }
    assert receipt["generation"]["prompt"]["text"] == PROMPT
    assert len(receipt["generation"]["prompt"]["sha256"]) == 64
    assert receipt["inputs"]["source_rgb"]["sha256"] == sha256_file(source_path)[0]
    assert receipt["inputs"]["residual_mask"]["sha256"] == sha256_file(mask_path)[0]
    for candidate in receipt["candidates"]:
        artifact = candidate["artifact"]
        assert artifact["sha256"] == sha256_file(Path(artifact["path"]))[0]
        assert candidate["metrics"]["inside_changed_pixels"] > 0
        assert candidate["metrics"]["outside_changed_pixels"] == 0
        assert candidate["metrics"]["outside_rgb_exact"] is True
        assert candidate["review_status"] == "pending_review"


def test_seeds_must_be_contiguous_explicit_and_capped_at_three() -> None:
    assert validate_seeds([900, 901, 902]) == [900, 901, 902]
    with pytest.raises(DiffusionAnchorInpaintError, match="between 1 and 3"):
        validate_seeds([])
    with pytest.raises(DiffusionAnchorInpaintError, match="between 1 and 3"):
        validate_seeds([1, 2, 3, 4])
    with pytest.raises(DiffusionAnchorInpaintError, match="contiguous"):
        validate_seeds([10, 12])
    with pytest.raises(DiffusionAnchorInpaintError, match="contiguous"):
        validate_seeds([11, 10])
    with pytest.raises(DiffusionAnchorInpaintError, match="64-bit"):
        validate_seeds([True])  # type: ignore[list-item]


def test_input_contract_rejects_non_rgb_non_binary_empty_and_wrong_size(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.png"
    mask_path = tmp_path / "mask.png"

    Image.new("RGBA", (8, 6), (1, 2, 3, 255)).save(source_path)
    Image.new("L", (8, 6), 255).save(mask_path)
    with pytest.raises(DiffusionAnchorInpaintError, match="mode must be RGB"):
        load_source_and_mask(source_path, mask_path)

    Image.new("RGB", (8, 6), (1, 2, 3)).save(source_path)
    Image.new("L", (7, 6), 255).save(mask_path)
    with pytest.raises(DiffusionAnchorInpaintError, match="dimensions must match"):
        load_source_and_mask(source_path, mask_path)

    Image.new("L", (8, 6), 0).save(mask_path)
    with pytest.raises(DiffusionAnchorInpaintError, match="at least one masked pixel"):
        load_source_and_mask(source_path, mask_path)

    non_binary = np.zeros((6, 8), dtype=np.uint8)
    non_binary[1:3, 1:3] = 128
    Image.fromarray(non_binary).save(mask_path)
    with pytest.raises(DiffusionAnchorInpaintError, match="must be binary"):
        load_source_and_mask(source_path, mask_path)


@pytest.mark.parametrize(
    ("steps", "guidance", "strength", "message"),
    [
        (0, 7.5, 1.0, "num_inference_steps"),
        (20, float("nan"), 1.0, "guidance_scale must be a finite"),
        (20, 7.5, float("inf"), "strength must be a finite"),
        (20, 7.5, 0.0, "strength must be inside"),
    ],
)
def test_generation_parameters_fail_closed_on_invalid_or_nonfinite_values(
    steps: int,
    guidance: float,
    strength: float,
    message: str,
) -> None:
    with pytest.raises(DiffusionAnchorInpaintError, match=message):
        validate_generation_parameters(
            num_inference_steps=steps,
            guidance_scale=guidance,
            strength=strength,
        )


def test_candidate_without_inside_delta_is_rejected_before_any_output(
    tmp_path: Path,
) -> None:
    source_path, mask_path, model_dir, _, _ = make_fixture(tmp_path)
    pipeline = FakeInpaintPipeline(preserve_inside=True)

    with pytest.raises(DiffusionAnchorInpaintError, match="no RGB delta inside"):
        run_diffusion_anchor_inpaint(
            source_rgb_path=source_path,
            residual_mask_path=mask_path,
            output_dir=tmp_path / "output",
            model_dir=model_dir,
            model_repo=MODEL_REPO,
            model_revision=MODEL_REVISION,
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            seeds=[1],
            pipeline_loader=lambda **_: LoadedPipeline(
                pipeline=pipeline,
                runtime={"test_double": True},
            ),
            generator_factory=lambda seed: seed,
        )
    assert not (tmp_path / "output").exists()


def test_local_loader_selects_fp16_safetensors_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, Any] = {}
    float16_token = object()

    class FakeLoadedPipeline:
        def to(self, device: str) -> None:
            calls["device"] = device

        def set_progress_bar_config(self, *, disable: bool) -> None:
            calls["progress_disabled"] = disable

    class FakePipelineClass:
        @classmethod
        def from_pretrained(cls, model_dir: str, **kwargs: Any) -> FakeLoadedPipeline:
            calls["model_dir"] = model_dir
            calls["kwargs"] = kwargs
            return FakeLoadedPipeline()

    fake_torch = ModuleType("torch")
    fake_torch.__version__ = "test-torch"
    fake_torch.float16 = float16_token
    fake_torch.bfloat16 = object()
    fake_torch.float32 = object()
    fake_diffusers = ModuleType("diffusers")
    fake_diffusers.__version__ = "test-diffusers"
    fake_diffusers.AutoPipelineForInpainting = FakePipelineClass
    fake_diffusers.StableDiffusionXLInpaintPipeline = FakePipelineClass
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    loaded = load_local_inpaint_pipeline(
        model_dir=tmp_path,
        pipeline_kind="sdxl",
        device="cuda:7",
        dtype="float16",
        variant="fp16",
        use_safetensors=True,
    )

    assert calls == {
        "device": "cuda:7",
        "model_dir": str(tmp_path),
        "progress_disabled": False,
        "kwargs": {
            "local_files_only": True,
            "torch_dtype": float16_token,
            "use_safetensors": True,
            "variant": "fp16",
        },
    }
    assert loaded.runtime["local_files_only"] is True
    assert loaded.runtime["variant"] == "fp16"
    assert loaded.runtime["use_safetensors"] is True

    with pytest.raises(DiffusionAnchorInpaintError, match="use_safetensors must remain true"):
        load_local_inpaint_pipeline(
            model_dir=tmp_path,
            pipeline_kind="sdxl",
            device="cuda:7",
            dtype="float16",
            variant="fp16",
            use_safetensors=False,
        )


def test_module_import_keeps_torch_and_diffusers_lazy() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import scripts.run_diffusion_anchor_inpaint; "
                "print('torch' in sys.modules, 'diffusers' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "False False"


def test_cli_exposes_auto_and_sdxl_with_repeated_explicit_seeds() -> None:
    args = build_parser().parse_args(
        [
            "--source-rgb",
            "source.png",
            "--residual-mask",
            "mask.png",
            "--output-dir",
            "output",
            "--model-dir",
            "model",
            "--model-repo",
            MODEL_REPO,
            "--model-revision",
            MODEL_REVISION,
            "--prompt",
            PROMPT,
            "--negative-prompt",
            NEGATIVE_PROMPT,
            "--seed",
            "77",
            "--seed",
            "78",
            "--pipeline-kind",
            "sdxl",
        ]
    )
    assert args.seed == [77, 78]
    assert args.pipeline_kind == "sdxl"
    assert args.variant == "fp16"
