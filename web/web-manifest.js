const WEB_MANIFEST_SCHEMA_VERSION = 1;
const WEB_MANIFEST_CONTRACT = "video2world-web-manifest-1.0.0";

function isRecord(value) {
  return value != null && typeof value === "object" && !Array.isArray(value);
}

function requireCondition(condition, message) {
  if (!condition) throw new Error(`Invalid Web manifest: ${message}`);
}

function isFiniteVector(value, length = 3) {
  return Array.isArray(value)
    && value.length === length
    && value.every((item) => Number.isFinite(Number(item)));
}

function hasAssetLocation(asset) {
  return typeof asset?.url === "string"
    || (Array.isArray(asset?.parts) && asset.parts.length > 0);
}

function validateParts(asset, path) {
  if (!Array.isArray(asset.parts)) return;
  asset.parts.forEach((part, index) => {
    requireCondition(isRecord(part), `${path}.parts[${index}] must be an object`);
    requireCondition(typeof part.url === "string" && part.url.length > 0, `${path}.parts[${index}].url`);
    requireCondition(Number.isInteger(part.size) && part.size >= 0, `${path}.parts[${index}].size`);
    requireCondition(
      typeof part.sha256 === "string" && /^[a-f0-9]{64}$/i.test(part.sha256),
      `${path}.parts[${index}].sha256`,
    );
  });
}

function validateAsset(asset, path, { allowMetadataOnly = false } = {}) {
  requireCondition(isRecord(asset), `${path} must be an object`);
  validateParts(asset, path);
  const explicitMetadataOnly = allowMetadataOnly
    && (Number(asset.vertexCount) === 0 || asset.baselineMetadataOnly === true);
  requireCondition(
    hasAssetLocation(asset) || explicitMetadataOnly,
    `${path} must provide url/parts or be an explicit metadata-only record`,
  );
  if (asset.size != null) {
    requireCondition(Number.isInteger(asset.size) && asset.size >= 0, `${path}.size`);
  }
  if (asset.sha256 != null) {
    requireCondition(/^[a-f0-9]{64}$/i.test(asset.sha256), `${path}.sha256`);
  }
}

function validateInteractiveObject(object, index, ids) {
  const path = `interactiveObjects[${index}]`;
  requireCondition(isRecord(object), `${path} must be an object`);
  requireCondition(typeof object.id === "string" && object.id.length > 0, `${path}.id`);
  requireCondition(!ids.has(object.id), `${path}.id duplicates ${object.id}`);
  ids.add(object.id);
  requireCondition(isRecord(object.placement), `${path}.placement`);
  requireCondition(isFiniteVector(object.placement.pivot), `${path}.placement.pivot`);
  requireCondition(isFiniteVector(object.placement.scale), `${path}.placement.scale`);
  validateAsset(object.visual, `${path}.visual`);
  requireCondition(isRecord(object.collision), `${path}.collision`);
  requireCondition(typeof object.collision.mode === "string", `${path}.collision.mode`);
  if (object.collision.mode === "none") {
    requireCondition(object.collision.asset == null, `${path} visual-only object must not declare collision.asset`);
  } else if (object.collision.mode === "degraded-box") {
    requireCondition(object.collision.asset == null, `${path} degraded box must not declare collision.asset`);
  } else {
    validateAsset(object.collision.asset, `${path}.collision.asset`);
  }
}

export function validateWebManifest(value) {
  requireCondition(isRecord(value), "root must be an object");
  requireCondition(value.schemaVersion === WEB_MANIFEST_SCHEMA_VERSION, "schemaVersion must equal 1");
  requireCondition(value.contract === WEB_MANIFEST_CONTRACT, `contract must equal ${WEB_MANIFEST_CONTRACT}`);
  requireCondition(typeof value.version === "string" && value.version.length > 0, "version is required");
  requireCondition(isRecord(value.coordinateSystem), "coordinateSystem is required");
  requireCondition(isFiniteVector(value.coordinateSystem.worldUp), "coordinateSystem.worldUp");
  requireCondition(isRecord(value.assets), "assets is required");
  validateAsset(value.assets.visual, "assets.visual", { allowMetadataOnly: true });
  validateAsset(value.assets.collider, "assets.collider", { allowMetadataOnly: true });
  if (value.assets.colliderStaticCarved != null) {
    validateAsset(value.assets.colliderStaticCarved, "assets.colliderStaticCarved");
  }
  requireCondition(Array.isArray(value.interactiveObjects), "interactiveObjects must be an array");
  const ids = new Set();
  value.interactiveObjects.forEach((object, index) => validateInteractiveObject(object, index, ids));
  requireCondition(isRecord(value.sceneKnowledge), "sceneKnowledge is required");
  requireCondition(Array.isArray(value.sceneKnowledge.objects), "sceneKnowledge.objects must be an array");
  if (value.sourceWorld != null) {
    requireCondition(isRecord(value.sourceWorld), "sourceWorld must be an object");
    requireCondition(typeof value.sourceWorld.worldId === "string", "sourceWorld.worldId");
    requireCondition(typeof value.sourceWorld.runId === "string", "sourceWorld.runId");
    requireCondition(typeof value.sourceWorld.adoptionMode === "string", "sourceWorld.adoptionMode");
  }
  return value;
}

export { WEB_MANIFEST_CONTRACT, WEB_MANIFEST_SCHEMA_VERSION };
