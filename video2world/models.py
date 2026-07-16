"""Strong contracts for the portable Video2World world bundle."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Vector3 = tuple[float, float, float]
Matrix4 = tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class LocalizedText(StrictModel):
    zh: str | None = Field(default=None, min_length=1)
    en: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def require_one_language(self) -> LocalizedText:
        if not self.zh and not self.en:
            raise ValueError("at least one localized value is required")
        return self

    def for_language(self, language: Literal["zh", "en"]) -> str | None:
        return getattr(self, language) or self.en or self.zh


class AssetRef(StrictModel):
    uri: str = Field(min_length=1)
    sha256: Sha256
    size_bytes: int = Field(ge=1)
    media_type: str | None = None
    role: str = Field(min_length=1)
    status: Literal["candidate", "validated", "rejected"] = "candidate"
    provenance: dict[str, Any] = Field(default_factory=dict)


class CoordinateSystem(StrictModel):
    frame_id: str = Field(min_length=1)
    up_axis: Literal["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
    handedness: Literal["right", "left"]
    units: Literal[
        "meters",
        "centimeters",
        "millimeters",
        "native",
        "scene_scale_not_metric",
    ] = "native"
    metric_scale: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def metric_units_require_scale(self) -> CoordinateSystem:
        if self.units in {"meters", "centimeters", "millimeters"} and self.metric_scale is None:
            object.__setattr__(self, "metric_scale", 1.0)
        if self.units == "scene_scale_not_metric" and self.metric_scale is not None:
            raise ValueError("scene_scale_not_metric cannot declare a metric scale")
        return self


class Bounds3D(StrictModel):
    frame_id: str = Field(min_length=1)
    minimum: Vector3
    maximum: Vector3

    @model_validator(mode="after")
    def ordered_bounds(self) -> Bounds3D:
        if any(high <= low for low, high in zip(self.minimum, self.maximum, strict=True)):
            raise ValueError("each bounds maximum must be greater than its minimum")
        return self

    @property
    def center(self) -> Vector3:
        return tuple(
            (low + high) / 2.0 for low, high in zip(self.minimum, self.maximum, strict=True)
        )  # type: ignore[return-value]


class Bounds2D(StrictModel):
    frame_id: str = Field(min_length=1)
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    xyxy: tuple[float, float, float, float]
    convention: Literal["pixel-inclusive", "pixel-exclusive"] = "pixel-exclusive"

    @model_validator(mode="after")
    def ordered_bounds(self) -> Bounds2D:
        x0, y0, x1, y1 = self.xyxy
        if x1 <= x0 or y1 <= y0:
            raise ValueError("2D bounds must have positive area")
        if x0 < 0 or y0 < 0 or x1 > self.image_width or y1 > self.image_height:
            raise ValueError("2D bounds exceed the declared image dimensions")
        return self


class OrientedBounds3D(StrictModel):
    frame_id: str = Field(min_length=1)
    transform: Matrix4
    extents: Vector3

    @model_validator(mode="after")
    def positive_extents(self) -> OrientedBounds3D:
        if any(value <= 0 for value in self.extents):
            raise ValueError("OBB extents must be positive")
        return self


class SceneTransform(StrictModel):
    from_frame: str = Field(min_length=1)
    to_frame: str = Field(min_length=1)
    matrix: Matrix4
    pivot_scene: Vector3
    scale_xyz: Vector3 = (1.0, 1.0, 1.0)

    @model_validator(mode="after")
    def positive_scale(self) -> SceneTransform:
        if any(value <= 0 for value in self.scale_xyz):
            raise ValueError("transform scale must be positive")
        return self


class DescriptionEvidence(StrictModel):
    short: LocalizedText | None = None
    detailed: LocalizedText | None = None
    appearance: LocalizedText | None = None
    location: LocalizedText | None = None
    provider: str | None = None
    model: str | None = None
    evidence_frames: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)
    fidelity_caveat: LocalizedText | None = None


class Relation(StrictModel):
    predicate: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    target_object_id: str | None = None
    target_label: LocalizedText | None = None
    confidence: float = Field(ge=0, le=1)
    verified: bool = False
    evidence_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_target(self) -> Relation:
        if not self.target_object_id and not self.target_label:
            raise ValueError("a relation requires target_object_id or target_label")
        return self


class GateRecord(StrictModel):
    status: Literal["passed", "failed", "not_tested"] = "not_tested"
    report_uri: str | None = None
    metrics: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    reason: str | None = None


class QualityGates(StrictModel):
    file: GateRecord = Field(default_factory=GateRecord)
    semantic: GateRecord = Field(default_factory=GateRecord)
    alignment: GateRecord = Field(default_factory=GateRecord)
    collision: GateRecord = Field(default_factory=GateRecord)
    visual: GateRecord = Field(default_factory=GateRecord)

    def interactive_ready(self) -> bool:
        return all(
            gate.status == "passed"
            for gate in (self.file, self.semantic, self.alignment, self.collision, self.visual)
        )

    def visual_interaction_ready(self) -> bool:
        return all(
            gate.status == "passed"
            for gate in (self.file, self.semantic, self.alignment, self.visual)
        )


class InteractionPolicy(StrictModel):
    selectable: bool = False
    double_click_action: Literal["none", "spin_360"] = "none"
    drag_action: Literal["none", "rotate_yaw"] = "none"
    collision_enabled: bool = False
    physics_mode: Literal["none", "static", "kinematic", "dynamic"] = "none"

    @model_validator(mode="after")
    def collision_policy_is_explicit(self) -> InteractionPolicy:
        if self.collision_enabled and self.physics_mode == "none":
            raise ValueError("collision-enabled interaction requires a physics mode")
        if not self.collision_enabled and self.physics_mode != "none":
            raise ValueError("visual-only interaction must use physics_mode='none'")
        return self


class ObjectEvidence(StrictModel):
    source_frame_id: str | None = None
    source_image: AssetRef | None = None
    source_mask: AssetRef | None = None
    source_point_cloud: AssetRef | None = None
    bbox_2d: Bounds2D | None = None
    mask_score: float | None = Field(default=None, ge=0, le=1)
    support_ratio: float | None = Field(default=None, ge=0, le=1)


class WorldObject(StrictModel):
    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    source_run_id: str = Field(min_length=1)
    scoped_id: str = Field(min_length=1)
    name: LocalizedText
    category: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    description: DescriptionEvidence = Field(default_factory=DescriptionEvidence)
    evidence: ObjectEvidence = Field(default_factory=ObjectEvidence)
    bbox_scene: Bounds3D | None = None
    obb_scene: OrientedBounds3D | None = None
    visual: AssetRef | None = None
    render_mesh: AssetRef | None = None
    collider: AssetRef | None = None
    transform_scene_from_asset: SceneTransform | None = None
    relations: list[Relation] = Field(default_factory=list)
    quality_gates: QualityGates = Field(default_factory=QualityGates)
    interaction: InteractionPolicy = Field(default_factory=InteractionPolicy)

    @model_validator(mode="after")
    def interactive_contract(self) -> WorldObject:
        enabled = (
            self.interaction.selectable
            or self.interaction.double_click_action != "none"
            or self.interaction.drag_action != "none"
            or self.interaction.collision_enabled
            or self.interaction.physics_mode != "none"
        )
        if not enabled:
            return self
        missing = [
            field
            for field in ("bbox_scene", "visual", "transform_scene_from_asset")
            if getattr(self, field) is None
        ]
        if missing:
            raise ValueError(f"interactive object is missing: {', '.join(missing)}")
        if not self.quality_gates.visual_interaction_ready():
            raise ValueError(
                "interactive visual objects require passed file/semantic/alignment/visual gates"
            )
        assert self.visual is not None
        if self.visual.status != "validated":
            raise ValueError("interactive visual asset must be validated")
        if self.interaction.collision_enabled:
            if self.collider is None:
                raise ValueError("collision-enabled interactive object is missing: collider")
            if self.quality_gates.collision.status != "passed":
                raise ValueError(
                    "collision-enabled interactive object requires a passed collision gate"
                )
            if self.collider.status != "validated":
                raise ValueError("interactive collider asset must be validated")
        else:
            if self.collider is not None:
                raise ValueError("visual-only interactive object must not declare a collider")
            if self.quality_gates.collision.status == "passed":
                raise ValueError("visual-only interactive object must not pass the collision gate")
        return self


class SceneLayer(StrictModel):
    coordinate_system: CoordinateSystem
    bounds: Bounds3D | None = None
    visual: AssetRef
    collider: AssetRef
    semantic_visual: AssetRef


class ProvenanceRecord(StrictModel):
    source_repositories: dict[str, str] = Field(default_factory=dict)
    source_runs: list[str] = Field(default_factory=list)
    config_sha256: Sha256
    notes: list[str] = Field(default_factory=list)


class WorldManifest(StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    manifest_status: Literal["draft", "validated"] = "draft"
    world_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    run_id: str = Field(min_length=1)
    created_at: datetime
    scene: SceneLayer
    objects: list[WorldObject] = Field(default_factory=list)
    provenance: ProvenanceRecord

    @model_validator(mode="after")
    def enforce_world_invariants(self) -> WorldManifest:
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must include an explicit timezone")
        scoped_ids: set[str] = set()
        scene_frame = self.scene.coordinate_system.frame_id
        if self.scene.bounds and self.scene.bounds.frame_id != scene_frame:
            raise ValueError("scene bounds are not in the scene frame")
        for item in self.objects:
            expected = f"{self.world_id}::{item.source_run_id}::{item.id}"
            if item.scoped_id != expected:
                raise ValueError(f"object scoped_id must be {expected!r}")
            if item.scoped_id in scoped_ids:
                raise ValueError(f"duplicate scoped object identity: {item.scoped_id}")
            scoped_ids.add(item.scoped_id)
            if item.bbox_scene and item.bbox_scene.frame_id != scene_frame:
                raise ValueError(f"object {item.scoped_id} bbox is not in the scene frame")
            if item.obb_scene and item.obb_scene.frame_id != scene_frame:
                raise ValueError(f"object {item.scoped_id} OBB is not in the scene frame")
            if (
                item.transform_scene_from_asset
                and item.transform_scene_from_asset.to_frame != scene_frame
            ):
                raise ValueError(
                    f"object {item.scoped_id} transform does not target the scene frame"
                )

        if self.manifest_status == "validated":
            scene_assets = (self.scene.visual, self.scene.collider, self.scene.semantic_visual)
            if any(asset.status != "validated" for asset in scene_assets):
                raise ValueError("a validated manifest requires validated scene layers")
        return self

    def object_index(self) -> dict[str, WorldObject]:
        return {item.scoped_id: item for item in self.objects}

    def iter_assets(self):  # type: ignore[no-untyped-def]
        yield "scene.visual", self.scene.visual
        yield "scene.collider", self.scene.collider
        yield "scene.semantic_visual", self.scene.semantic_visual
        for item in self.objects:
            prefix = f"objects[{item.scoped_id}]"
            for name in ("visual", "render_mesh", "collider"):
                asset = getattr(item, name)
                if asset:
                    yield f"{prefix}.{name}", asset
            for name in ("source_image", "source_mask", "source_point_cloud"):
                asset = getattr(item.evidence, name)
                if asset:
                    yield f"{prefix}.evidence.{name}", asset


def world_manifest_json_schema() -> dict[str, Any]:
    schema = WorldManifest.model_json_schema(mode="validation")
    schema["$id"] = "https://relumeow.top/schemas/video2world/world-manifest-1.0.0.json"
    schema["title"] = "Video2World World Manifest"
    return schema
