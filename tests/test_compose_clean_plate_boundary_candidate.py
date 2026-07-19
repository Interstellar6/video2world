from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "compose_clean_plate_boundary_candidate.py"
SPEC = importlib.util.spec_from_file_location(
    "compose_clean_plate_boundary_candidate",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def save_rgb(path: Path, value: np.ndarray) -> None:
    Image.fromarray(value.astype(np.uint8)).save(path)


def save_mask(path: Path, value: np.ndarray) -> None:
    Image.fromarray(value.astype(np.uint8) * 255).save(path)


def asset(path: Path, *, relative_to: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "sha256": MODULE.sha256_file(path),
        "bytes": path.stat().st_size,
    }


def make_manifest(
    root: Path,
    *,
    radius: int = 3,
    protected: bool = True,
    color_limit: float = 255.0,
    gradient_limit: float = 255.0,
) -> Path:
    height = width = 11
    y, x = np.indices((height, width))
    previous = np.stack((x * 8 + 20, y * 6 + 30, x * 3 + y * 2 + 40), axis=2).astype(np.uint8)
    color_offset = np.asarray([30, 18, 9], dtype=np.int16)
    candidate = np.clip(previous.astype(np.int16) + color_offset, 0, 255).astype(np.uint8)
    core = np.zeros((height, width), dtype=bool)
    core[4:7, 4:7] = True
    protected_mask = np.zeros_like(core)
    if protected:
        protected_mask[5, 7] = True

    previous_path = root / "previous.png"
    candidate_path = root / "candidate.png"
    core_path = root / "core.png"
    protected_path = root / "protected.png"
    save_rgb(previous_path, previous)
    save_rgb(candidate_path, candidate)
    save_mask(core_path, core)
    save_mask(protected_path, protected_mask)
    manifest_path = root / "manifest.json"
    manifest = {
        "schema_version": 1,
        "kind": MODULE.INPUT_KIND,
        "config": {
            "outer_collar_pixels": radius,
            "blend_curve": "smoothstep_euclidean_distance",
            "maximum_boundary_color_p95_delta": color_limit,
            "maximum_boundary_gradient_p95_delta": gradient_limit,
        },
        "frame_records": [
            {
                "sequence_index": 0,
                "frame_id": "frame-a",
                "previous_composite": asset(previous_path, relative_to=root),
                "source_aligned_mask": asset(core_path, relative_to=root),
                "candidate_fill": asset(candidate_path, relative_to=root),
                "protected_mask": asset(protected_path, relative_to=root),
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) == 255


def test_compose_manifest_emits_protected_three_band_candidate(tmp_path: Path) -> None:
    manifest_path = make_manifest(tmp_path)
    output = tmp_path / "output"

    report = MODULE.compose_manifest(manifest_path, output)

    assert report["status"] == "candidate_generated_boundary_passed"
    assert report["gates"]["all_structural_compositor_invariants_passed"] is True
    frame = report["frame_records"][0]
    assert frame["pixels"]["core"] == 9
    assert frame["pixels"]["collar"] > 0
    assert frame["pixels"]["protected_outer_collar"] == 1
    assert frame["gates"]["collar_excludes_protected_mask"] is True
    assert (
        frame["boundary_metrics"]["outer_editable_boundary"]["boundary_color_p95_abs_rgb_delta"]
        is not None
    )
    assert (
        frame["boundary_metrics"]["outer_editable_boundary"][
            "boundary_normal_gradient_p95_abs_rgb_delta"
        ]
        is not None
    )

    previous = load_rgb(tmp_path / "previous.png")
    candidate = load_rgb(tmp_path / "candidate.png")
    composite = load_rgb(output / "frames" / "0000.png")
    core = load_mask(output / "masks" / "core" / "0000.png")
    collar = load_mask(output / "masks" / "collar" / "0000.png")
    protected = load_mask(output / "masks" / "protected" / "0000.png")
    editable = load_mask(output / "masks" / "editable" / "0000.png")

    assert np.array_equal(composite[core], candidate[core])
    assert np.array_equal(composite[~editable], previous[~editable])
    assert not np.any(core & collar)
    assert not np.any(collar & protected)
    assert np.array_equal(composite[5, 7], previous[5, 7])
    assert report["aggregate"]["output_set_sha256"]
    assert (output / "boundary_candidate_report.json").is_file()


def test_distance_weight_decreases_outward_from_core(tmp_path: Path) -> None:
    manifest_path = make_manifest(tmp_path, radius=3, protected=False)
    output = tmp_path / "output"

    MODULE.compose_manifest(manifest_path, output)

    previous = load_rgb(tmp_path / "previous.png").astype(np.int16)
    candidate = load_rgb(tmp_path / "candidate.png").astype(np.int16)
    composite = load_rgb(output / "frames" / "0000.png").astype(np.int16)
    core_delta = np.abs(candidate[5, 5] - previous[5, 5]).sum()
    near_delta = np.abs(composite[5, 7] - previous[5, 7]).sum()
    far_delta = np.abs(composite[5, 9] - previous[5, 9]).sum()

    assert np.abs(composite[5, 5] - previous[5, 5]).sum() == core_delta
    assert 0 < far_delta < near_delta < core_delta


def test_boundary_quality_failure_keeps_review_candidate(tmp_path: Path) -> None:
    manifest_path = make_manifest(
        tmp_path,
        color_limit=0.0,
        gradient_limit=0.0,
    )
    output = tmp_path / "output"

    report = MODULE.compose_manifest(manifest_path, output)

    assert report["status"] == "candidate_generated_boundary_failed"
    assert report["gates"]["boundary_quality_passed"] is False
    assert report["gates"]["all_structural_compositor_invariants_passed"] is True
    assert (output / "frames" / "0000.png").is_file()


def test_hash_mismatch_is_rejected_atomically(tmp_path: Path) -> None:
    manifest_path = make_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["frame_records"][0]["candidate_fill"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "output"

    with pytest.raises(MODULE.BoundaryCandidateError, match="candidate_fill SHA-256 mismatch"):
        MODULE.compose_manifest(manifest_path, output)

    assert not output.exists()


def test_source_aligned_mask_must_be_binary(tmp_path: Path) -> None:
    manifest_path = make_manifest(tmp_path)
    core_path = tmp_path / "core.png"
    invalid = np.zeros((11, 11), dtype=np.uint8)
    invalid[4:7, 4:7] = 127
    Image.fromarray(invalid).save(core_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["frame_records"][0]["source_aligned_mask"] = asset(
        core_path,
        relative_to=tmp_path,
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "output"

    with pytest.raises(MODULE.BoundaryCandidateError, match="must be binary 0/255"):
        MODULE.compose_manifest(manifest_path, output)

    assert not output.exists()
