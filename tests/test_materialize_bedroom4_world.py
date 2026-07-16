from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.materialize_bedroom4_world import materialize_file, merge_chunk_parts
from video2world.hashing import sha256_file


def test_chunk_merge_verifies_parts_and_final_artifact(tmp_path: Path) -> None:
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    chunks = [b"ply\nformat binary_little_endian 1.0\n", b"end_header\nreal-data"]
    parts = []
    for index, payload in enumerate(chunks):
        path = chunk_dir / f"asset.chunk{index:03d}"
        path.write_bytes(payload)
        parts.append(
            {
                "url": f"./worlds/fixture/chunks/{path.name}",
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    expected = b"".join(chunks)
    output = tmp_path / "asset.ply"
    result = merge_chunk_parts(
        parts,
        chunk_dir,
        output,
        expected_sha256=hashlib.sha256(expected).hexdigest(),
        expected_size=len(expected),
    )
    assert result.mode == "verified_chunk_merge"
    assert output.read_bytes() == expected

    reused = merge_chunk_parts(
        parts,
        chunk_dir,
        output,
        expected_sha256=sha256_file(output)[0],
        expected_size=output.stat().st_size,
    )
    assert reused.mode == "reused_verified"


def test_chunk_merge_rejects_corrupt_source_chunk(tmp_path: Path) -> None:
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    chunk = chunk_dir / "asset.chunk000"
    chunk.write_bytes(b"corrupt")
    parts = [{"url": chunk.name, "size": 7, "sha256": "0" * 64}]
    with pytest.raises(RuntimeError, match="chunk sha256 mismatch"):
        merge_chunk_parts(
            parts,
            chunk_dir,
            tmp_path / "asset.ply",
            expected_sha256="0" * 64,
            expected_size=7,
        )


def test_file_materialization_prefers_and_recognizes_hardlinks(tmp_path: Path) -> None:
    source = tmp_path / "source.glb"
    destination = tmp_path / "assets" / "object.glb"
    source.write_bytes(b"real-binary-asset")

    first = materialize_file(source, destination)
    assert first.mode == "hardlink"
    assert source.samefile(destination)

    second = materialize_file(source, destination)
    assert second.mode == "reused_hardlink"
