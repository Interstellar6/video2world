from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import trimesh
from PIL import Image

from video2world.providers import trellis2_asset
from video2world.providers.external import load_provider_contract
from video2world.providers.trellis2_asset import (
    CANONICAL_SIX_VIEWS,
    MAX_DECIMATION_FACES,
    Trellis2AssetError,
    Trellis2AssetRequest,
    audit_pbr_glb,
    audit_pbr_material,
    audit_triangle_mesh,
    canonical_six_view_contract,
    model_revision_evidence,
    pretrained_config_name,
    run_trellis2_asset,
    validate_generation_parameters,
    validate_input_image,
)


def test_pretrained_config_name_keeps_local_loader_on_disk(tmp_path: Path) -> None:
    weights = tmp_path / "weights"
    config = weights / "configs" / "pipeline_512.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}", encoding="utf-8")

    assert pretrained_config_name(weights, config) == "configs/pipeline_512.json"

    external = tmp_path / "external.json"
    external.write_text("{}", encoding="utf-8")
    with pytest.raises(Trellis2AssetError, match="inside the weights directory"):
        pretrained_config_name(weights, external)


def write_rgba(path: Path) -> None:
    pixels = np.zeros((16, 16, 4), dtype=np.uint8)
    pixels[3:13, 3:13, :3] = [224, 226, 228]
    pixels[3:13, 3:13, 3] = 255
    Image.fromarray(pixels).save(path)


def write_pbr_glb(
    path: Path,
    *,
    watertight: bool,
    textured: bool = False,
    metallic_factor: float = 0.0,
    roughness_factor: float = 0.85,
    alpha_mode: str | None = None,
    alpha_cutoff: float | None = None,
    double_sided: bool = False,
) -> None:
    box = trimesh.creation.box()
    faces = np.asarray(box.faces)
    if not watertight:
        faces = faces[:-1]
    mesh = trimesh.Trimesh(
        vertices=np.asarray(box.vertices),
        faces=faces,
        process=False,
    )
    base_color_texture = Image.new("RGBA", (4, 2), (64, 128, 192, 200)) if textured else None
    material = trimesh.visual.material.PBRMaterial(
        name="test-pbr",
        baseColorFactor=[230, 231, 232, 255],
        metallicFactor=metallic_factor,
        roughnessFactor=roughness_factor,
        baseColorTexture=base_color_texture,
        alphaMode=alpha_mode,
        alphaCutoff=alpha_cutoff,
        doubleSided=double_sided,
    )
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2), dtype=np.float64),
        material=material,
    )
    path.write_bytes(trimesh.Scene(mesh).export(file_type="glb"))


def make_request(
    root: Path,
    *,
    decimation_target: int = MAX_DECIMATION_FACES,
) -> Trellis2AssetRequest:
    source_dir = root / "trellis2"
    weights_dir = root / "weights"
    output_dir = root / "output"
    source_dir.mkdir()
    weights_dir.mkdir()
    (weights_dir / "pipeline_512.json").write_text('{"name":"test"}', encoding="utf-8")
    input_path = root / "object.png"
    write_rgba(input_path)
    return Trellis2AssetRequest(
        trellis_source_dir=source_dir,
        weights_dir=weights_dir,
        config_file="pipeline_512.json",
        input_image=input_path,
        output_dir=output_dir,
        decimation_target=decimation_target,
    )


def test_rgba_is_default_and_rmbg_requires_explicit_opt_in(tmp_path: Path) -> None:
    rgba = tmp_path / "object.png"
    rgb = tmp_path / "photo.png"
    write_rgba(rgba)
    Image.new("RGB", (16, 16), (200, 201, 202)).save(rgb)

    report = validate_input_image(rgba, allow_rmbg=False)
    assert report["mode"] == "RGBA"
    assert report["alpha_range"] == [0, 255]
    assert report["background_removal"] == "disabled_rgba_required"

    with pytest.raises(Trellis2AssetError, match=r"default TRELLIS\.2 input must be RGBA"):
        validate_input_image(rgb, allow_rmbg=False)

    opted_in = validate_input_image(rgb, allow_rmbg=True)
    assert opted_in["background_removal"] == "explicit_opt_in"


def test_decimation_target_is_capped_at_one_hundred_thousand() -> None:
    validate_generation_parameters(
        seed=42,
        decimation_target=MAX_DECIMATION_FACES,
        texture_size=1024,
        debug_surface_points=200_000,
    )
    with pytest.raises(Trellis2AssetError, match="decimation_target"):
        validate_generation_parameters(
            seed=42,
            decimation_target=MAX_DECIMATION_FACES + 1,
            texture_size=1024,
            debug_surface_points=200_000,
        )


def test_triangle_audit_detects_invalid_and_degenerate_faces() -> None:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    degenerate = audit_triangle_mesh(vertices, np.asarray([[0, 0, 1]], dtype=np.int64))
    invalid = audit_triangle_mesh(vertices, np.asarray([[0, 1, 4]], dtype=np.int64))

    assert degenerate["valid_indices"] is True
    assert degenerate["degenerate_faces"] == 1
    assert invalid["valid_indices"] is False


def test_final_pbr_audit_accepts_nonwatertight_surface_bvh(tmp_path: Path) -> None:
    asset = tmp_path / "asset_pbr.glb"
    write_pbr_glb(asset, watertight=False)

    audit = audit_pbr_glb(asset)

    assert audit["technical_status"] == "passed"
    assert audit["collision_topology"] == "surface_bvh"
    assert audit["watertight"] is False
    assert audit["winding_consistent"] is True
    assert audit["pbr_material_count"] == audit["mesh_count"] == 1
    assert audit["technical_gates"]["no_degenerate_faces"] is True
    assert audit["inside_outside_queries_allowed"] is False
    material = audit["meshes"][0]["material"]
    assert material["metallic_factor"]["value"] == 0.0
    assert material["roughness_factor"]["value"] == pytest.approx(0.85)
    assert material["base_color_factor"]["rgba"] == pytest.approx(
        [230 / 255, 231 / 255, 232 / 255, 1.0]
    )
    assert material["base_color_texture"]["present"] is False
    assert material["alpha"]["mode"] == "OPAQUE"
    assert material["alpha"]["cutoff"]["value"] == 0.5
    assert material["double_sided"]["value"] is False


def test_pbr_material_receipt_records_texture_alpha_and_unit_factors(tmp_path: Path) -> None:
    asset = tmp_path / "asset_pbr.glb"
    write_pbr_glb(
        asset,
        watertight=False,
        textured=True,
        metallic_factor=1.0,
        roughness_factor=0.35,
        alpha_mode="BLEND",
        alpha_cutoff=0.25,
        double_sided=True,
    )

    audit = audit_pbr_glb(asset)
    material = audit["meshes"][0]["material"]
    texture = material["base_color_texture"]

    assert audit["technical_status"] == "passed"
    assert material["metallic_factor"]["value"] == 1.0
    assert material["metallic_factor"]["valid"] is True
    assert material["roughness_factor"]["value"] == pytest.approx(0.35)
    assert texture["present"] is True
    assert texture["valid"] is True
    assert texture["size"] == [4, 2]
    assert texture["mode"] == "RGBA"
    assert len(texture["decoded_pixel_sha256"]) == 64
    assert texture["finite"] is True
    assert texture["in_unit_range"] is True
    assert texture["alpha_channel_present"] is True
    assert material["alpha"]["mode"] == "BLEND"
    assert material["alpha"]["cutoff"]["value"] == 0.25
    assert material["double_sided"]["value"] is True
    assert material["semantic_review"]["metallic_one_is_not_a_generic_failure"] is True


def test_pbr_material_audit_fails_nonfinite_out_of_range_and_invalid_alpha() -> None:
    material = trimesh.visual.material.PBRMaterial(
        metallicFactor=float("nan"),
        roughnessFactor=1.25,
        alphaCutoff=-0.1,
    )
    material_data = vars(material)["_data"]
    material_data["alphaMode"] = "INVALID"
    material_data["doubleSided"] = "yes"

    audit = audit_pbr_material(material)

    assert audit["technical_status"] == "failed"
    assert audit["metallic_factor"]["value"] is None
    assert audit["metallic_factor"]["finite"] is False
    assert audit["roughness_factor"]["in_unit_range"] is False
    assert audit["alpha"]["mode_valid"] is False
    assert audit["alpha"]["cutoff"]["in_unit_range"] is False
    assert audit["double_sided"]["valid_boolean"] is False
    assert set(audit["failed_gates"]) == {
        "alpha_cutoff_finite_unit_range",
        "alpha_mode_valid",
        "double_sided_boolean",
        "metallic_factor_finite_unit_range",
        "roughness_factor_finite_unit_range",
    }


def test_final_pbr_audit_reports_closed_volume_only_when_watertight(tmp_path: Path) -> None:
    asset = tmp_path / "asset_pbr.glb"
    write_pbr_glb(asset, watertight=True)

    audit = audit_pbr_glb(asset)

    assert audit["technical_status"] == "passed"
    assert audit["collision_topology"] == "closed_volume"
    assert audit["closed_volume_claim"] is True


def test_final_pbr_audit_fails_closed_when_actual_faces_exceed_target(tmp_path: Path) -> None:
    asset = tmp_path / "asset_pbr.glb"
    write_pbr_glb(asset, watertight=False)

    audit = audit_pbr_glb(asset, face_limit=10)

    assert audit["faces"] > 10
    assert audit["technical_status"] == "failed"
    assert audit["failed_gates"] == ["face_budget"]


def test_model_revision_evidence_reads_huggingface_metadata(tmp_path: Path) -> None:
    weights = tmp_path / "weights"
    metadata = weights / ".cache/huggingface/download/ckpts/model.safetensors.metadata"
    metadata.parent.mkdir(parents=True)
    revision = "a" * 40
    metadata.write_text(f"{revision}\nblob\ntimestamp\n", encoding="utf-8")

    evidence = model_revision_evidence(weights, expected_revision=revision)
    assert evidence["metadata_revisions"] == [revision]
    assert evidence["verification"] == "huggingface_download_metadata"

    with pytest.raises(Trellis2AssetError, match="revision metadata mismatch"):
        model_revision_evidence(weights, expected_revision="b" * 40)


def test_provider_writes_only_required_asset_by_default_and_keeps_visual_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = make_request(tmp_path)
    monkeypatch.setattr(
        trellis2_asset,
        "git_source_provenance",
        lambda *_args, **_kwargs: {
            "repository": trellis2_asset.VERIFIED_SOURCE_REPOSITORY,
            "commit": trellis2_asset.VERIFIED_SOURCE_COMMIT,
            "verified": True,
        },
    )
    monkeypatch.setattr(
        trellis2_asset,
        "model_revision_evidence",
        lambda *_args, **_kwargs: {
            "path": str(request.weights_dir),
            "expected_revision": trellis2_asset.VERIFIED_MODEL_REVISION,
            "metadata_file_count": 1,
            "metadata_revisions": [trellis2_asset.VERIFIED_MODEL_REVISION],
            "verification": "test_double",
        },
    )

    def fake_generator(value: Trellis2AssetRequest, config_path: Path) -> dict[str, Any]:
        assert config_path.name == "pipeline_512.json"
        write_pbr_glb(value.output_dir / "asset_pbr.glb", watertight=False)
        processed = value.output_dir / "processed_input.png"
        write_rgba(processed)
        return {
            "processed_input": trellis2_asset.artifact_record(processed),
            "pbr_asset": trellis2_asset.artifact_record(value.output_dir / "asset_pbr.glb"),
            "debug_artifacts": [],
            "runtime": {"test_double": True},
        }

    receipt = run_trellis2_asset(
        request,
        generator=fake_generator,
        created_at=datetime(2026, 7, 17, 12, 0, tzinfo=UTC),
    )

    assert receipt["status"] == "technical_passed_visual_pending"
    assert receipt["promotion_allowed"] is False
    assert receipt["source"]["commit"] == trellis2_asset.VERIFIED_SOURCE_COMMIT
    assert receipt["model"]["revision"] == trellis2_asset.VERIFIED_MODEL_REVISION
    assert len(receipt["config"]["sha256"]) == 64
    assert receipt["generation"]["seed"] == 42
    assert receipt["outputs"]["unified_pbr_glb"]["collision_topology"] == "surface_bvh"
    assert receipt["outputs"]["unified_pbr_glb"]["status"] == "candidate"
    assert receipt["visual_review"]["status"] == "pending"
    assert receipt["debug_exports"]["artifacts"] == []
    assert sorted(path.name for path in request.output_dir.iterdir()) == [
        "asset_pbr.glb",
        "processed_input.png",
        "trellis2_asset_receipt.json",
    ]
    written = json.loads(
        (request.output_dir / "trellis2_asset_receipt.json").read_text(encoding="utf-8")
    )
    assert written == receipt


def test_provider_persists_failed_receipt_when_final_audit_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = make_request(tmp_path, decimation_target=10)
    monkeypatch.setattr(
        trellis2_asset,
        "git_source_provenance",
        lambda *_args, **_kwargs: {"commit": trellis2_asset.VERIFIED_SOURCE_COMMIT},
    )
    monkeypatch.setattr(
        trellis2_asset,
        "model_revision_evidence",
        lambda *_args, **_kwargs: {"verification": "test_double"},
    )

    def over_budget(value: Trellis2AssetRequest, _config_path: Path) -> dict[str, Any]:
        write_pbr_glb(value.output_dir / "asset_pbr.glb", watertight=False)
        return {"debug_artifacts": []}

    with pytest.raises(Trellis2AssetError, match="failed technical gates"):
        run_trellis2_asset(request, generator=over_budget)

    failed = json.loads(
        (request.output_dir / "trellis2_asset_receipt.json").read_text(encoding="utf-8")
    )
    assert failed["status"] == "failed"
    assert failed["promotion_allowed"] is False
    assert failed["technical_audit"]["failed_gates"] == ["face_budget"]


def test_six_view_contract_is_orthogonal_and_visual_pending() -> None:
    contract = canonical_six_view_contract()
    assert contract["required_views"] == list(CANONICAL_SIX_VIEWS)
    assert contract["orthogonal_axes_required"] is True
    assert contract["horizontal_orbit_contact_sheet_is_not_six_view_evidence"] is True
    assert contract["status"] == "pending"
    assert "shape_mismatch" in contract["severity_policy"]["blocking"]
    assert "minor_backside_texture_hallucination" in contract["severity_policy"]["warning"]


def test_generic_provider_example_pins_verified_trellis2_lineage() -> None:
    root = Path(__file__).resolve().parents[1]
    contract = load_provider_contract(
        root / "video2world/configs/holi_embodiedgen.provider.example.yaml"
    )
    stage = contract.stages["trellis"]

    assert contract.source_repositories["embodiedgen_v2"] == (
        f"{trellis2_asset.VERIFIED_SOURCE_REPOSITORY}@{trellis2_asset.VERIFIED_SOURCE_COMMIT}"
    )
    assert stage.requires["trellis_asset_provider"].path.endswith(
        "/video2world/providers/trellis2_asset.py"
    )
    assert "dino_source" not in stage.requires
    lineage = "\n".join(stage.lineage_evidence)
    assert trellis2_asset.VERIFIED_SOURCE_COMMIT in lineage
    assert trellis2_asset.VERIFIED_MODEL_REVISION in lineage
    assert "308dd782f3eaf1e30e7403ee3837f21beefa664202b0a045ca61308a42c661b3" in lineage
    assert "raw mesh, point cloud, and convex exports are debug-only" in lineage
