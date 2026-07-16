"""Deterministic content hashing for files, directories, and structured values."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from video2world.errors import ArtifactError

CHUNK_SIZE = 4 * 1024 * 1024


@dataclass(frozen=True)
class PathDigest:
    path: Path
    kind: str
    sha256: str
    size_bytes: int
    file_count: int


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_SIZE):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def digest_path(path: str | Path, *, require_nonempty: bool = True) -> PathDigest:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise ArtifactError(f"artifact does not exist: {resolved}")
    if resolved.is_symlink():
        raise ArtifactError(f"symlink artifacts are not accepted: {resolved}")
    if resolved.is_file():
        sha256, size = sha256_file(resolved)
        if require_nonempty and size == 0:
            raise ArtifactError(f"artifact is empty: {resolved}")
        return PathDigest(resolved, "file", sha256, size, 1)
    if not resolved.is_dir():
        raise ArtifactError(f"artifact must be a regular file or directory: {resolved}")

    digest = hashlib.sha256()
    total_size = 0
    file_count = 0
    for child in sorted(resolved.rglob("*"), key=lambda item: item.as_posix()):
        if child.is_symlink():
            raise ArtifactError(f"directory artifact contains a symlink: {child}")
        if not child.is_file():
            continue
        relative = child.relative_to(resolved).as_posix().encode("utf-8")
        child_sha, child_size = sha256_file(child)
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(child_sha))
        digest.update(child_size.to_bytes(8, "big"))
        total_size += child_size
        file_count += 1
    if require_nonempty and file_count == 0:
        raise ArtifactError(f"directory artifact contains no files: {resolved}")
    if require_nonempty and total_size == 0:
        raise ArtifactError(f"directory artifact contains no non-empty files: {resolved}")
    return PathDigest(resolved, "directory", digest.hexdigest(), total_size, file_count)


def digest_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
    )
