from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from scripts.run_diffusion_clean_plate_batch import (
    DiffusionCleanPlateBatchError,
    LoadedPipeline,
    euclidean_dilate,
    model_file_manifest_sha256,
    model_file_records,
    run_diffusion_clean_plate_batch,
    sha256_file,
)

MODEL_REPO = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
MODEL_REVISION = "115134f363124c53c7d878647567d04daf26e41e"


def asset(path: Path, *, root: Path) -> dict[str, Any]:
    digest, size = sha256_file(path)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": digest,
        "size_bytes": size,
    }


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) > 0


def make_manifest(root: Path, *, frame_count: int = 2) -> tuple[Path, list[dict[str, Any]]]:
    model_dir = root / "model"
    (model_dir / "unet").mkdir(parents=True)
    (model_dir / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusionXLInpaintPipeline"}),
        encoding="utf-8",
    )
    (model_dir / "unet" / "weights.fp16.safetensors").write_bytes(b"fixed-test-weights")
    records: list[dict[str, Any]] = []
    for index in range(frame_count):
        source = np.zeros((11, 13, 3), dtype=np.uint8)
        source[..., 0] = 20 + index * 30
        source[..., 1] = np.arange(13, dtype=np.uint8)[None, :] * 8
        source[..., 2] = np.arange(11, dtype=np.uint8)[:, None] * 7
        ownership = np.zeros((11, 13), dtype=np.uint8)
        ownership[5, 4 + index] = 255
        source_path = root / f"source-{index}.png"
        mask_path = root / f"ownership-{index}.png"
        Image.fromarray(source).save(source_path)
        Image.fromarray(ownership).save(mask_path)
        record: dict[str, Any] = {
            "sequence_index": index,
            "frame_id": f"frame-{index}",
            "source_rgb": asset(source_path, root=root),
            "ownership_mask": asset(mask_path, root=root),
            "ownership_mask_source_rgb_sha256": asset(source_path, root=root)["sha256"],
        }
        if index == 1:
            guide = np.full_like(source, [200, 180, 160])
            guide_path = root / "guide-1.png"
            Image.fromarray(guide).save(guide_path)
            record["guide_rgb"] = asset(guide_path, root=root)
            record["guide_source_rgb_sha256"] = record["source_rgb"]["sha256"]
        records.append(record)
    manifest = {
        "schema_version": 1,
        "kind": "video2world.diffusion_clean_plate_batch_input",
        "config": {
            "prompt": "Restore only the occluded background bed surface.",
            "negative_prompt": "new object, foreground pillow, blur",
            "context_dilation_pixels": 2,
            "boundary_outer_collar_pixels": 1,
            "seed_policy": {
                "kind": "base_plus_sequence_index",
                "base_seed": 700,
                "stride": 3,
            },
            "model": {
                "repo": MODEL_REPO,
                "revision": MODEL_REVISION,
                "local_path": "model",
                "pipeline_class": "StableDiffusionXLInpaintPipeline",
                "local_files_only": True,
                "use_safetensors": True,
                "dtype": "float16",
                "variant": "fp16",
                "device": "cuda:test",
                "file_manifest_sha256": model_file_manifest_sha256(model_file_records(model_dir)),
            },
            "generation": {
                "num_inference_steps": 28,
                "guidance_scale": 6.5,
                "strength": 0.8,
            },
        },
        "frame_records": records,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, records


class FakePipeline:
    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_on_call = fail_on_call

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self.fail_on_call == len(self.calls):
            raise DiffusionCleanPlateBatchError("injected inference failure")
        image = np.asarray(kwargs["image"].convert("RGB"), dtype=np.uint8)
        mask = np.asarray(kwargs["mask_image"].convert("L"), dtype=np.uint8) > 0
        candidate = image.copy()
        seed = int(kwargs["generator"])
        candidate[mask] = [seed % 251, 90, 140]
        candidate[~mask] = [1, 2, 3]
        return SimpleNamespace(images=[Image.fromarray(candidate)])


def test_batch_loads_once_and_processes_ordered_frames_with_distinct_masks(
    tmp_path: Path,
) -> None:
    manifest_path, records = make_manifest(tmp_path)
    pipeline = FakePipeline()
    loader_calls: list[dict[str, Any]] = []

    def loader(**kwargs: Any) -> LoadedPipeline:
        loader_calls.append(kwargs)
        return LoadedPipeline(pipeline=pipeline, runtime={"test_double": True})

    output = tmp_path / "output"
    receipt = run_diffusion_clean_plate_batch(
        manifest_path,
        output,
        created_at=datetime(2026, 7, 18, 10, 0, tzinfo=UTC),
        pipeline_loader=loader,
        generator_factory=lambda seed: seed,
    )

    assert len(loader_calls) == 1
    assert loader_calls[0] == {
        "model_dir": (tmp_path / "model").resolve(),
        "device": "cuda:test",
        "dtype": "float16",
        "variant": "fp16",
        "use_safetensors": True,
    }
    assert [call["generator"] for call in pipeline.calls] == [700, 703]
    assert receipt["aggregate"]["ordered_frame_ids"] == ["frame-0", "frame-1"]
    assert receipt["aggregate"]["ordered_seeds"] == [700, 703]
    assert receipt["runtime"]["pipeline_load_count"] == 1
    assert receipt["promotion_allowed"] is False

    for index, call in enumerate(pipeline.calls):
        ownership_path = tmp_path / records[index]["ownership_mask"]["path"]
        ownership = load_mask(ownership_path)
        expected_context = euclidean_dilate(ownership, 2)
        actual_context = np.asarray(call["mask_image"].convert("L")) > 0
        assert np.array_equal(actual_context, expected_context)
        assert np.count_nonzero(actual_context) > np.count_nonzero(ownership)
        expected_inference = load_rgb(tmp_path / records[index]["source_rgb"]["path"]).copy()
        if "guide_rgb" in records[index]:
            guide = load_rgb(tmp_path / records[index]["guide_rgb"]["path"])
            expected_inference[expected_context] = guide[expected_context]
        assert np.array_equal(np.asarray(call["image"].convert("RGB")), expected_inference)

        frame_root = output / "frames" / f"{index:04d}"
        source = load_rgb(tmp_path / records[index]["source_rgb"]["path"])
        raw = load_rgb(frame_root / "raw_candidate.png")
        composite = load_rgb(frame_root / "composite.png")
        context = load_mask(frame_root / "masks" / "context.png")
        core = load_mask(frame_root / "masks" / "ownership.png")
        editable = load_mask(frame_root / "masks" / "editable.png")
        assert np.array_equal(raw[~context], source[~context])
        assert np.array_equal(composite[core], raw[core])
        assert np.array_equal(composite[~editable], source[~editable])
        assert np.any(composite[editable & ~core] != source[editable & ~core])

        frame_receipt = json.loads((frame_root / "frame_receipt.json").read_text(encoding="utf-8"))
        assert frame_receipt["seed"] == [700, 703][index]
        assert frame_receipt["batch_contract"]["model_revision"] == MODEL_REVISION
        assert (
            frame_receipt["batch_contract"]["input_manifest_sha256"]
            == receipt["input_manifest"]["sha256"]
        )
        assert frame_receipt["masks"]["context_additional_pixels"] > 0
        assert frame_receipt["exactness"]["inference_outside_context_rgb_exact"] is True
        assert frame_receipt["exactness"]["raw_candidate_outside_context_rgb_exact"] is True
        assert frame_receipt["exactness"]["final_outside_editable_rgb_exact"] is True
        assert (
            frame_receipt["exactness"]["provider_inside_context_changed_from_inference_pixels"] > 0
        )
        for artifact_record in frame_receipt["outputs"].values():
            path = output / artifact_record["path"]
            assert sha256_file(path)[0] == artifact_record["sha256"]

    written = json.loads((output / "batch_receipt.json").read_text(encoding="utf-8"))
    assert written == receipt


def test_zero_stride_reuses_one_reviewed_seed_for_every_frame(tmp_path: Path) -> None:
    manifest_path, _ = make_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config"]["seed_policy"]["stride"] = 0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    pipeline = FakePipeline()

    receipt = run_diffusion_clean_plate_batch(
        manifest_path,
        tmp_path / "output",
        pipeline_loader=lambda **_: LoadedPipeline(pipeline=pipeline, runtime={}),
        generator_factory=lambda seed: seed,
    )

    assert [call["generator"] for call in pipeline.calls] == [700, 700]
    assert receipt["aggregate"]["ordered_seeds"] == [700, 700]


def test_hash_mismatch_fails_before_pipeline_load_and_writes_nothing(tmp_path: Path) -> None:
    manifest_path, _ = make_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["frame_records"][1]["ownership_mask"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    loader_called = False

    def loader(**_: Any) -> LoadedPipeline:
        nonlocal loader_called
        loader_called = True
        return LoadedPipeline(pipeline=FakePipeline(), runtime={})

    output = tmp_path / "output"
    with pytest.raises(DiffusionCleanPlateBatchError, match="ownership_mask SHA-256 mismatch"):
        run_diffusion_clean_plate_batch(
            manifest_path,
            output,
            pipeline_loader=loader,
            generator_factory=lambda seed: seed,
        )

    assert loader_called is False
    assert not output.exists()


def test_wrong_source_mask_lineage_fails_before_pipeline_load(tmp_path: Path) -> None:
    manifest_path, _ = make_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["frame_records"][1]["ownership_mask_source_rgb_sha256"] = manifest["frame_records"][0][
        "source_rgb"
    ]["sha256"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DiffusionCleanPlateBatchError, match="not bound to exact source RGB"):
        run_diffusion_clean_plate_batch(
            manifest_path,
            tmp_path / "output",
            pipeline_loader=lambda **_: pytest.fail("loader must not be called"),
        )


def test_model_file_manifest_mismatch_fails_before_pipeline_load(tmp_path: Path) -> None:
    manifest_path, _ = make_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config"]["model"]["file_manifest_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DiffusionCleanPlateBatchError, match="model file manifest SHA-256 differs"):
        run_diffusion_clean_plate_batch(
            manifest_path,
            tmp_path / "output",
            pipeline_loader=lambda **_: pytest.fail("loader must not be called"),
        )


def test_wrong_guide_lineage_fails_before_pipeline_load(tmp_path: Path) -> None:
    manifest_path, _ = make_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["frame_records"][1]["guide_source_rgb_sha256"] = manifest["frame_records"][0][
        "source_rgb"
    ]["sha256"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DiffusionCleanPlateBatchError, match="guide RGB is not bound"):
        run_diffusion_clean_plate_batch(
            manifest_path,
            tmp_path / "output",
            pipeline_loader=lambda **_: pytest.fail("loader must not be called"),
        )


def test_protected_collar_stays_source_exact(tmp_path: Path) -> None:
    manifest_path, _ = make_manifest(tmp_path, frame_count=1)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_sha = manifest["frame_records"][0]["source_rgb"]["sha256"]
    protected = np.zeros((11, 13), dtype=np.uint8)
    protected[5, 5] = 255
    protected_path = tmp_path / "protected-0.png"
    Image.fromarray(protected).save(protected_path)
    manifest["frame_records"][0]["protected_mask"] = asset(protected_path, root=tmp_path)
    manifest["frame_records"][0]["protected_mask_source_rgb_sha256"] = source_sha
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    pipeline = FakePipeline()

    receipt = run_diffusion_clean_plate_batch(
        manifest_path,
        tmp_path / "output",
        pipeline_loader=lambda **_: LoadedPipeline(pipeline=pipeline, runtime={}),
        generator_factory=lambda seed: seed,
    )

    call_context = np.asarray(pipeline.calls[0]["mask_image"].convert("L")) > 0
    assert not call_context[5, 5]
    composite = load_rgb(tmp_path / "output" / "frames" / "0000" / "composite.png")
    source = load_rgb(tmp_path / "source-0.png")
    assert np.array_equal(composite[5, 5], source[5, 5])
    frame_receipt = receipt["frames"][0]
    assert frame_receipt["masks"]["protected_collar_pixels"] == 1
    assert frame_receipt["exactness"]["protected_outer_collar_rgb_exact"] is True


def test_frame_order_must_be_contiguous_before_model_load(tmp_path: Path) -> None:
    manifest_path, _ = make_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["frame_records"][0]["sequence_index"] = 1
    manifest["frame_records"][1]["sequence_index"] = 0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DiffusionCleanPlateBatchError, match="must be ordered"):
        run_diffusion_clean_plate_batch(
            manifest_path,
            tmp_path / "output",
            pipeline_loader=lambda **_: pytest.fail("loader must not be called"),
        )


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    manifest_path, _ = make_manifest(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(DiffusionCleanPlateBatchError, match="refusing overwrite"):
        run_diffusion_clean_plate_batch(
            manifest_path,
            output,
            pipeline_loader=lambda **_: pytest.fail("loader must not be called"),
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_inference_failure_removes_staging_and_leaves_destination_absent(tmp_path: Path) -> None:
    manifest_path, _ = make_manifest(tmp_path)
    pipeline = FakePipeline(fail_on_call=2)
    output = tmp_path / "output"

    with pytest.raises(DiffusionCleanPlateBatchError, match="injected inference failure"):
        run_diffusion_clean_plate_batch(
            manifest_path,
            output,
            pipeline_loader=lambda **_: LoadedPipeline(pipeline=pipeline, runtime={}),
            generator_factory=lambda seed: seed,
        )

    assert not output.exists()
    assert list(tmp_path.glob(".output.staging-*")) == []


def test_euclidean_dilation_excludes_diagonal_at_radius_one() -> None:
    ownership = np.zeros((5, 5), dtype=bool)
    ownership[2, 2] = True
    context = euclidean_dilate(ownership, 1)

    assert context[2, 2]
    assert context[1, 2] and context[2, 1] and context[2, 3] and context[3, 2]
    assert not context[1, 1]


def test_import_keeps_torch_and_diffusers_lazy() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import scripts.run_diffusion_clean_plate_batch; "
                "print('torch' in sys.modules, 'diffusers' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "False False"
