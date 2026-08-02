from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.compare_da3_vggt_omega import (
    parse_pgsr_metrics,
    parse_ply_header,
    rotation_error_degrees,
    umeyama_similarity,
)


def test_umeyama_recovers_similarity() -> None:
    source = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    target = 2.5 * (source @ rotation.T) + np.array([3.0, -2.0, 1.0])
    scale, fitted_rotation, translation = umeyama_similarity(source, target)
    assert scale == pytest.approx(2.5)
    np.testing.assert_allclose(fitted_rotation, rotation, atol=1e-7)
    np.testing.assert_allclose(translation, [3.0, -2.0, 1.0], atol=1e-7)


def test_rotation_error_degrees() -> None:
    candidate = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert rotation_error_degrees(np.eye(3), candidate) == pytest.approx(90.0)


def test_parse_pgsr_metrics_selects_latest_iteration(tmp_path: Path) -> None:
    log = tmp_path / "train.log"
    log.write_text(
        "[ITER 7000] Evaluating train: L1 0.02 PSNR 26.7\n"
        "[ITER 30000] Evaluating train: L1 0.01 PSNR 29.2\n",
        encoding="utf-8",
    )
    metrics = parse_pgsr_metrics(log)
    assert metrics["iteration"] == 30000
    assert metrics["l1"] == pytest.approx(0.01)
    assert metrics["psnr_db"] == pytest.approx(29.2)


def test_parse_ply_header(tmp_path: Path) -> None:
    path = tmp_path / "mesh.ply"
    path.write_bytes(
        b"ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\n"
        b"property float y\nproperty float z\nelement face 1\n"
        b"property list uchar int vertex_indices\nend_header\n"
        b"0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n"
    )
    audit = parse_ply_header(path)
    assert audit["elements"] == {"vertex": 3, "face": 1}
    assert audit["vertex_properties"] == ["x", "y", "z"]
