from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

from scripts.materialize_bedroom4_unified_pillow import (
    COLLIDER_CHUNK_SIZE,
    FRONT_EXPECTED,
    REAR_EXPECTED,
    STABLE_BASE_MANIFEST_SHA256,
    STATIC_VISUAL_EXPECTED,
    adopt_verified_chunks,
    build_candidate_manifest,
    carve_triangle_mesh_ply,
    inspect_rgb_ply,
    load_evidence,
    parse_binary_triangle_mesh_ply,
    preserve_stable_manifest,
    read_json,
    write_verified_chunks,
)
from video2world.hashing import sha256_file


def test_split_rear_pillow_payloads_match_geometry_reports() -> None:
    root = Path(__file__).parents[1]
    base = root / "examples/bedroom4/manifest.production.json"
    evidence = load_evidence(root, base)

    for object_id, expected in REAR_EXPECTED.items():
        stats = inspect_rgb_ply(evidence.rear_sources[object_id])
        record = evidence.rear_records[object_id]
        assert stats.point_count == expected["points"] == record["point_count"]
        assert list(stats.minimum) == record["geometry"]["aabb"]["minimum"]
        assert list(stats.maximum) == record["geometry"]["aabb"]["maximum"]
        assert sha256_file(evidence.rear_sources[object_id]) == (
            expected["sha256"],
            expected["size"],
        )


def test_curated_candidate_assets_are_present_under_tracked_example_paths() -> None:
    root = Path(__file__).parents[1]
    base = root / "examples/bedroom4/manifest.production.json"
    evidence = load_evidence(root, base)

    assert evidence.front_source.is_relative_to(root / "examples")
    assert sha256_file(evidence.front_source) == (
        FRONT_EXPECTED["sha256"],
        FRONT_EXPECTED["size"],
    )
    for object_id, source in evidence.rear_sources.items():
        assert source.is_relative_to(root / "examples")
        assert sha256_file(source) == (
            REAR_EXPECTED[object_id]["sha256"],
            REAR_EXPECTED[object_id]["size"],
        )


def test_committed_candidate_is_reproducible_from_base_and_verified_evidence() -> None:
    root = Path(__file__).parents[1]
    base_path = root / "examples/bedroom4/manifest.production.json"
    candidate_path = root / "examples/bedroom4/manifests/bedroom4.unified-pillow.candidate.web.json"
    evidence = load_evidence(root, base_path)
    candidate = read_json(candidate_path)

    expected = build_candidate_manifest(
        read_json(base_path),
        evidence,
        collider_carve=candidate["candidateBuild"]["sceneColliderCarve"],
        collider_parts=candidate["assets"]["colliderStaticCarved"]["parts"],
    )
    assert candidate == expected

    front = next(
        item for item in expected["interactiveObjects"] if item["id"] == "sam3_pillow_front"
    )
    assert front["collision"]["asset"]["sha256"] == FRONT_EXPECTED["sha256"]
    assert front["collision"]["asset"]["size"] == FRONT_EXPECTED["size"]
    assert expected["candidateBuild"]["backgroundGeometry"]["status"] == "not_proven"
    assert expected["candidateBuild"]["promotedManifestBrowserQa"]["status"] == (
        "pending_promoted_manifest_recheck"
    )
    static_visual = expected["assets"]["visual"]
    assert static_visual["sha256"] == STATIC_VISUAL_EXPECTED["sha256"]
    assert static_visual["size"] == STATIC_VISUAL_EXPECTED["size"]
    assert static_visual["vertexCount"] == STATIC_VISUAL_EXPECTED["vertices"]
    assert len(static_visual["parts"]) == STATIC_VISUAL_EXPECTED["parts"]
    assert "url" not in static_visual
    assert all("recarve-candidates" not in part["url"] for part in static_visual["parts"])
    collider = expected["assets"]["colliderStaticCarved"]
    assert "url" not in collider
    assert sum(part["size"] for part in collider["parts"]) == collider["size"]
    assert max(part["size"] for part in collider["parts"]) <= COLLIDER_CHUNK_SIZE
    assert "sam3_pillow_01" not in json.dumps(expected, ensure_ascii=False)


def _tiny_triangle_ply() -> bytes:
    vertices = [
        (-2.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (-1.5, 1.0, 0.0),
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    ]
    faces = [(0, 1, 2), (3, 4, 5)]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property double x\n"
        "property double y\n"
        "property double z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        f"element face {len(faces)}\n"
        "property list uchar uint vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    vertex_payload = b"".join(struct.pack("<dddBBB", *vertex, 255, 255, 255) for vertex in vertices)
    face_payload = b"".join(struct.pack("<BIII", 3, *face) for face in faces)
    return header + vertex_payload + face_payload


def test_triangle_collider_carve_removes_every_intersecting_face() -> None:
    source = _tiny_triangle_ply()
    bounds = {"min": [-0.1, -0.1, -0.1], "max": [1.1, 1.1, 0.1]}

    output, report = carve_triangle_mesh_ply(
        source,
        label="tiny fixture",
        bounds=bounds,
    )
    parsed = parse_binary_triangle_mesh_ply(output, "tiny carved fixture")

    assert report == {
        "inputFaceCount": 2,
        "outputFaceCount": 1,
        "removedFaceCount": 1,
        "vertexCount": 6,
        "finite": True,
        "faceIndicesValid": True,
        "retainedIntersectingFaces": 0,
        "retainedUnprotectedIntersectingFaces": 0,
        "protectedSupportFaceCount": 0,
        "supportProtection": None,
        "carveBounds": bounds,
    }
    assert parsed.face_count == 1
    assert len(output) == len(source) - 13
    assert hashlib.sha256(output).hexdigest() != hashlib.sha256(source).hexdigest()


def test_triangle_collider_carve_preserves_only_geometric_down_side_support() -> None:
    vertices = [
        (-2.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (-1.5, 0.0, 1.0),
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    ]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "element vertex 6\nproperty double x\nproperty double y\nproperty double z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "element face 2\nproperty list uchar uint vertex_indices\nend_header\n"
    ).encode("ascii")
    source = (
        header
        + b"".join(struct.pack("<dddBBB", *vertex, 255, 255, 255) for vertex in vertices)
        + b"".join(struct.pack("<BIII", 3, *face) for face in [(0, 1, 2), (3, 4, 5)])
    )
    bounds = {"min": [-2.1, -0.1, -0.1], "max": [1.1, 1.1, 1.1]}
    policy = {
        "worldUp": [0.0, -1.0, 0.0],
        "objectDownSideCoordinate": 0.0,
        "inwardBand": 0.1,
        "outwardBand": 0.1,
        "minimumNormalAlignment": 0.75,
        "classification": "test_support",
    }

    output, report = carve_triangle_mesh_ply(
        source,
        label="tiny support fixture",
        bounds=bounds,
        support_protection=policy,
    )

    assert report["protectedSupportFaceCount"] == 1
    assert report["retainedIntersectingFaces"] == 1
    assert report["retainedUnprotectedIntersectingFaces"] == 0
    assert report["removedFaceCount"] == 1
    assert parse_binary_triangle_mesh_ply(output, "protected support fixture").face_count == 1


def test_collider_publish_chunks_are_deterministic_hash_verified_and_bounded(
    tmp_path: Path,
) -> None:
    payload = bytes(range(251)) * ((COLLIDER_CHUNK_SIZE * 2 + 17) // 251 + 1)
    payload = payload[: COLLIDER_CHUNK_SIZE * 2 + 17]
    chunk_dir = tmp_path / "chunks"
    stem = "fixture_collider_ply"

    parts = write_verified_chunks(
        payload,
        output_dir=chunk_dir,
        public_prefix="./worlds/fixture/chunks",
        stem=stem,
    )

    assert [part["size"] for part in parts] == [
        COLLIDER_CHUNK_SIZE,
        COLLIDER_CHUNK_SIZE,
        17,
    ]
    assert [part["url"] for part in parts] == [
        f"./worlds/fixture/chunks/{stem}.chunk000",
        f"./worlds/fixture/chunks/{stem}.chunk001",
        f"./worlds/fixture/chunks/{stem}.chunk002",
    ]
    reconstructed = b"".join((chunk_dir / Path(part["url"]).name).read_bytes() for part in parts)
    assert reconstructed == payload
    assert hashlib.sha256(reconstructed).hexdigest() == hashlib.sha256(payload).hexdigest()
    for part in parts:
        chunk = chunk_dir / Path(part["url"]).name
        assert chunk.stat().st_size <= COLLIDER_CHUNK_SIZE
        assert sha256_file(chunk) == (part["sha256"], part["size"])

    stale = chunk_dir / f"{stem}.chunk999"
    stale.write_bytes(b"stale")
    repeated = write_verified_chunks(
        payload,
        output_dir=chunk_dir,
        public_prefix="./worlds/fixture/chunks",
        stem=stem,
    )
    assert repeated == parts
    assert not stale.exists()


def test_static_chunks_are_adopted_into_a_canonical_publish_namespace(tmp_path: Path) -> None:
    payloads = [b"static-part-zero", b"static-part-one"]
    source_dir = tmp_path / "candidate" / "chunks"
    output_dir = tmp_path / "public" / "chunks"
    source_dir.mkdir(parents=True)
    source_parts = []
    for index, payload in enumerate(payloads):
        path = source_dir / f"candidate.chunk{index:03d}"
        path.write_bytes(payload)
        source_parts.append(
            {
                "url": f"./temporary/candidate.chunk{index:03d}",
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    merged = b"".join(payloads)
    stale = output_dir / "canonical.chunk999"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")

    parts, storage_modes = adopt_verified_chunks(
        source_parts,
        source_dir=source_dir,
        output_dir=output_dir,
        public_prefix="./worlds/fixture/chunks",
        stem="canonical",
        expected_sha256=hashlib.sha256(merged).hexdigest(),
        expected_size=len(merged),
    )

    assert [part["url"] for part in parts] == [
        "./worlds/fixture/chunks/canonical.chunk000",
        "./worlds/fixture/chunks/canonical.chunk001",
    ]
    assert len(storage_modes) == len(payloads)
    assert b"".join((output_dir / Path(part["url"]).name).read_bytes() for part in parts) == merged
    assert not stale.exists()


def test_stable_manifest_is_preserved_byte_for_byte_before_promotion(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    base = root / "examples/bedroom4/manifest.production.json"
    promoted = tmp_path / "manifest.json"
    stable = tmp_path / "manifest.web-demo-baseline-stable.json"
    promoted.write_bytes(base.read_bytes())

    preserve_stable_manifest(
        base_manifest=base,
        promoted_manifest=promoted,
        stable_manifest=stable,
    )

    assert stable.read_bytes() == base.read_bytes()
    assert sha256_file(stable)[0] == STABLE_BASE_MANIFEST_SHA256
