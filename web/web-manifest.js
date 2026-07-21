const WEB_MANIFEST_SCHEMA_VERSION = 1;
const WEB_MANIFEST_CONTRACT = "video2world-web-manifest-1.0.0";
const SCENE_COMMAND_SERVICE_CONTRACT = "video2world-scene-command-service-1.0.0";
const LOGICAL_HIERARCHY_ROLE = "unrendered_unselectable_hierarchy_ancestor";
const SEMANTIC_GRANULARITIES = new Set([
  "independent_root_asset",
  "independent_child_asset",
  "merged_component",
]);
const COLLIDER_PROXY_TYPES = new Set(["box", "selection-box"]);
const COLLISION_GATE_STATUSES = new Set(["passed", "failed", "held", "not_tested"]);
const COLLISION_MODES = new Set(["degraded-box", "kinematic", "none", "unified-glb"]);
const UNIFIED_COLLISION_TOPOLOGIES = new Set(["closed_volume", "surface_bvh"]);
const MAX_UNIFIED_GLTF_COLLISION_FACES = 100_000;
const INTERACTION_KINDS = new Set(["spin"]);
const INTERACTION_DRAG_MODES = new Set(["horizontal_yaw"]);

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

function isStrictFiniteVector(value, length = 3) {
  return Array.isArray(value)
    && value.length === length
    && value.every((item) => typeof item === "number" && Number.isFinite(item));
}

function isStrictFiniteMatrix4Rows(value) {
  return Array.isArray(value)
    && value.length === 4
    && value.every((row) => isStrictFiniteVector(row, 4));
}

function validateDecomposableAffineMatrix(rows, path) {
  requireCondition(isStrictFiniteMatrix4Rows(rows), `${path} must be a finite 4x4 row-major matrix`);
  requireCondition(
    rows[3][0] === 0 && rows[3][1] === 0 && rows[3][2] === 0 && rows[3][3] === 1,
    `${path} must be an affine matrix with last row [0, 0, 0, 1]`,
  );
  const columns = [0, 1, 2].map((column) => [rows[0][column], rows[1][column], rows[2][column]]);
  const lengths = columns.map((column) => Math.hypot(...column));
  requireCondition(lengths.every((length) => length > 1e-8), `${path} has a zero scale axis`);
  for (let left = 0; left < 3; left += 1) {
    for (let right = left + 1; right < 3; right += 1) {
      const dot = columns[left].reduce(
        (total, value, index) => total + value * columns[right][index],
        0,
      );
      requireCondition(
        Math.abs(dot) <= lengths[left] * lengths[right] * 1e-6,
        `${path} contains shear and cannot be decomposed for root interaction`,
      );
    }
  }
  const determinant = (
    rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
    - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
    + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0])
  );
  requireCondition(determinant > 1e-8, `${path} must have a positive nonsingular linear transform`);
}

export function placementMatrixElements(placement) {
  const rows = placement?.matrixRowMajor;
  if (rows == null) return null;
  validateDecomposableAffineMatrix(rows, "placement.matrixRowMajor");
  return [
    rows[0][0], rows[1][0], rows[2][0], rows[3][0],
    rows[0][1], rows[1][1], rows[2][1], rows[3][1],
    rows[0][2], rows[1][2], rows[2][2], rows[3][2],
    rows[0][3], rows[1][3], rows[2][3], rows[3][3],
  ];
}

function validateColliderProxy(proxy, path) {
  requireCondition(isRecord(proxy), `${path} must be an object`);
  requireCondition(COLLIDER_PROXY_TYPES.has(proxy.type), `${path}.type is unsupported`);
  requireCondition(
    isStrictFiniteVector(proxy.dimensions)
      && proxy.dimensions.every((dimension) => dimension > 0),
    `${path}.dimensions must contain three positive finite numbers`,
  );
  requireCondition(
    isStrictFiniteVector(proxy.center),
    `${path}.center must contain three finite numbers`,
  );
  if (proxy.collisionEnabled != null) {
    requireCondition(
      typeof proxy.collisionEnabled === "boolean",
      `${path}.collisionEnabled must be a boolean`,
    );
  }
}

function validateInteraction(interaction, path) {
  if (interaction == null) return;
  requireCondition(isRecord(interaction), `${path} must be an object`);
  const allowedKeys = new Set(["degrees", "drag", "durationMs", "kind"]);
  const unknownKeys = Object.keys(interaction).filter((key) => !allowedKeys.has(key));
  requireCondition(unknownKeys.length === 0, `${path} has unknown fields: ${unknownKeys.join(", ")}`);
  requireCondition(INTERACTION_KINDS.has(interaction.kind), `${path}.kind is unsupported`);
  requireCondition(interaction.degrees === 360, `${path}.degrees must equal 360`);
  requireCondition(
    typeof interaction.durationMs === "number"
      && Number.isFinite(interaction.durationMs)
      && interaction.durationMs > 0,
    `${path}.durationMs must be a positive finite number`,
  );
  if (interaction.drag != null) {
    requireCondition(
      INTERACTION_DRAG_MODES.has(interaction.drag),
      `${path}.drag is unsupported`,
    );
  }
}

function validateCollisionGate(gate, path, { required = false } = {}) {
  if (gate == null) {
    requireCondition(!required, `${path} is required for collidable objects`);
    return;
  }
  requireCondition(isRecord(gate), `${path} must be an object`);
  requireCondition(
    COLLISION_GATE_STATUSES.has(gate.status),
    `${path}.status is unsupported`,
  );
}

function isSupportedWorldUp(value) {
  return isStrictFiniteVector(value)
    && value[0] === 0
    && Math.abs(value[1]) === 1
    && value[2] === 0;
}

function hasAssetLocation(asset) {
  return typeof asset?.url === "string"
    || (Array.isArray(asset?.parts) && asset.parts.length > 0);
}

function isSupportedMeshAsset(asset) {
  const descriptor = [asset?.fileName, asset?.fileType, asset?.format]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
  return /(?:^|[.\s-])glb(?:$|[.\s-])|gltf-binary|(?:^|[.\s-])obj(?:$|[.\s-])|wavefront/u
    .test(descriptor);
}

function isSupportedUnifiedGlb(asset) {
  const descriptor = [asset?.fileName, asset?.fileType, asset?.format]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
  return /(?:^|[.\s-])glb(?:$|[.\s-])|gltf-binary/u.test(descriptor);
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
  if (!Object.prototype.hasOwnProperty.call(object, "semanticGranularity")) {
    object.semanticGranularity = "independent_root_asset";
  }
  if (!Object.prototype.hasOwnProperty.call(object, "parentObjectId")) {
    object.parentObjectId = null;
  }
  if (!Object.prototype.hasOwnProperty.call(object, "movesWithParent")) {
    object.movesWithParent = false;
  }
  if (!Object.prototype.hasOwnProperty.call(object, "independentlyMovable")) {
    object.independentlyMovable = true;
  }
  requireCondition(
    SEMANTIC_GRANULARITIES.has(object.semanticGranularity),
    `${path}.semanticGranularity`,
  );
  requireCondition(
    object.parentObjectId == null
      || (typeof object.parentObjectId === "string" && object.parentObjectId.length > 0),
    `${path}.parentObjectId must be null or a non-empty object id`,
  );
  requireCondition(typeof object.movesWithParent === "boolean", `${path}.movesWithParent`);
  requireCondition(typeof object.independentlyMovable === "boolean", `${path}.independentlyMovable`);
  if (object.childObjectIds != null) {
    requireCondition(
      Array.isArray(object.childObjectIds)
        && object.childObjectIds.every((childId) => typeof childId === "string" && childId.length > 0)
        && new Set(object.childObjectIds).size === object.childObjectIds.length,
      `${path}.childObjectIds must contain unique non-empty ids`,
    );
  }
  if (object.logicalHierarchyOnly != null) {
    requireCondition(
      typeof object.logicalHierarchyOnly === "boolean",
      `${path}.logicalHierarchyOnly must be a boolean`,
    );
  }
  if (object.logicalHierarchyOnly === true) {
    requireCondition(
      object.logicalRole === LOGICAL_HIERARCHY_ROLE,
      `${path}.logicalRole must equal ${LOGICAL_HIERARCHY_ROLE}`,
    );
    requireCondition(
      Array.isArray(object.childObjectIds),
      `${path}.childObjectIds is required for logical hierarchy nodes`,
    );
    requireCondition(
      object.independentlyMovable === false,
      `${path} logical hierarchy nodes cannot be independently movable`,
    );
    for (const field of [
      "placement",
      "collision",
      "visual",
      "renderAsset",
      "colliderProxy",
      "interaction",
      "carve",
      "sourceAnchor",
    ]) {
      requireCondition(
        !Object.prototype.hasOwnProperty.call(object, field),
        `${path} logical hierarchy nodes must not declare ${field}`,
      );
    }
    return;
  }
  requireCondition(isRecord(object.placement), `${path}.placement`);
  requireCondition(isFiniteVector(object.placement.pivot), `${path}.placement.pivot`);
  requireCondition(isFiniteVector(object.placement.scale), `${path}.placement.scale`);
  requireCondition(isRecord(object.collision), `${path}.collision`);
  requireCondition(COLLISION_MODES.has(object.collision.mode), `${path}.collision.mode is unsupported`);
  const unified = object.collision.mode === "unified-glb";
  const matrixRows = object.placement.matrixRowMajor;
  if (matrixRows != null) {
    requireCondition(unified, `${path}.placement.matrixRowMajor is supported only for unified-glb`);
    validateDecomposableAffineMatrix(matrixRows, `${path}.placement.matrixRowMajor`);
    requireCondition(
      object.placement.pivot.every(
        (value, axis) => Math.abs(Number(value) - matrixRows[axis][3]) <= 1e-6,
      ),
      `${path}.placement.pivot must equal the matrix translation`,
    );
    requireCondition(
      object.placement.scale.every((value) => Number(value) === 1),
      `${path}.placement.scale must be identity when matrixRowMajor is present`,
    );
    requireCondition(
      isFiniteVector(object.placement.rotationEulerDeg)
        && object.placement.rotationEulerDeg.every((value) => Number(value) === 0),
      `${path}.placement.rotationEulerDeg must be zero when matrixRowMajor is present`,
    );
  }
  const renderAsset = object.collision.renderAsset;
  if (unified) {
    requireCondition(object.visual == null, `${path} unified-glb must not declare visual`);
    requireCondition(renderAsset == null, `${path} unified-glb must not declare renderAsset`);
    requireCondition(object.colliderProxy == null, `${path} unified-glb must not declare colliderProxy`);
    if (object.placement.generatedCenter != null) {
      requireCondition(
        isStrictFiniteVector(object.placement.generatedCenter)
          && object.placement.generatedCenter.every((value) => value === 0),
        `${path} unified-glb requires a baked local origin at generatedCenter [0, 0, 0]`,
      );
    }
    validateAsset(object.collision.asset, `${path}.collision.asset`);
    requireCondition(
      isSupportedUnifiedGlb(object.collision.asset),
      `${path}.collision.asset must be a binary GLB for unified-glb`,
    );
    requireCondition(
      UNIFIED_COLLISION_TOPOLOGIES.has(object.collision.topology),
      `${path}.collision.topology must be surface_bvh or closed_volume`,
    );
    requireCondition(
      Number.isInteger(object.collision.asset.faces)
        && object.collision.asset.faces > 0
        && object.collision.asset.faces <= MAX_UNIFIED_GLTF_COLLISION_FACES,
      `${path}.collision.asset.faces must be between 1 and ${MAX_UNIFIED_GLTF_COLLISION_FACES}`,
    );
    requireCondition(
      object.collision.asset.finite === true
        && object.collision.asset.nondegenerate === true
        && object.collision.asset.windingConsistent === true,
      `${path}.collision.asset must be finite, nondegenerate, and winding-consistent`,
    );
    requireCondition(
      typeof object.collision.asset.watertight === "boolean",
      `${path}.collision.asset.watertight must be explicit`,
    );
    if (object.collision.topology === "closed_volume") {
      requireCondition(
        object.collision.asset.watertight === true,
        `${path} closed_volume requires a watertight GLB`,
      );
    }
  } else {
    validateColliderProxy(object.colliderProxy, `${path}.colliderProxy`);
    validateAsset(object.visual, `${path}.visual`);
    requireCondition(
      !isSupportedMeshAsset(object.visual),
      `${path}.visual mesh assets require collision.mode unified-glb`,
    );
    if (renderAsset != null) {
      validateAsset(renderAsset, `${path}.collision.renderAsset`);
      requireCondition(
        isSupportedMeshAsset(renderAsset),
        `${path}.collision.renderAsset must be a GLB or OBJ mesh`,
      );
    }
  }
  validateInteraction(object.interaction, `${path}.interaction`);
  if (object.collision.walkable != null) {
    requireCondition(typeof object.collision.walkable === "boolean", `${path}.collision.walkable`);
  }
  if (object.collision.characterCollision != null) {
    requireCondition(
      typeof object.collision.characterCollision === "boolean",
      `${path}.collision.characterCollision`,
    );
  }
  validateCollisionGate(object.collision.gate, `${path}.collision.gate`, {
    required: object.collision.mode !== "none",
  });
  if (unified) {
    requireCondition(
      object.collision.gate.status === "passed"
        && object.collision.gate.surfaceCollision === "passed",
      `${path} unified-glb requires passed gate.status and gate.surfaceCollision`,
    );
    requireCondition(
      object.collision.characterCollision === true,
      `${path} unified-glb must explicitly enable characterCollision`,
    );
  } else if (object.collision.mode === "none") {
    requireCondition(object.collision.asset == null, `${path} visual-only object must not declare collision.asset`);
  } else if (object.collision.mode === "degraded-box") {
    requireCondition(object.collision.asset == null, `${path} degraded box must not declare collision.asset`);
  } else {
    validateAsset(object.collision.asset, `${path}.collision.asset`);
  }
}

function validateInteractiveObjectHierarchy(objects, ids) {
  const parents = new Map();
  const childrenByParent = new Map(Array.from(ids, (id) => [id, []]));
  objects.forEach((object, index) => {
    const path = `interactiveObjects[${index}]`;
    const isChild = object.semanticGranularity === "independent_child_asset";
    requireCondition(
      isChild === (object.parentObjectId != null),
      `${path} independent_child_asset and parentObjectId must be declared together`,
    );
    requireCondition(
      object.movesWithParent === isChild,
      `${path}.movesWithParent must be true exactly for child assets`,
    );
    requireCondition(
      object.semanticGranularity !== "merged_component" || !object.independentlyMovable,
      `${path} merged components cannot be independently movable`,
    );
    if (object.parentObjectId == null) return;
    requireCondition(object.parentObjectId !== object.id, `${path} cannot be its own parent`);
    requireCondition(
      ids.has(object.parentObjectId),
      `${path}.parentObjectId references unknown object ${object.parentObjectId}`,
    );
    parents.set(object.id, object.parentObjectId);
    childrenByParent.get(object.parentObjectId).push(object.id);
  });

  objects.forEach((object, index) => {
    if (object.logicalHierarchyOnly !== true) return;
    const declared = [...object.childObjectIds].sort();
    const derived = [...childrenByParent.get(object.id)].sort();
    requireCondition(
      declared.length === derived.length
        && declared.every((childId, childIndex) => childId === derived[childIndex]),
      `interactiveObjects[${index}].childObjectIds must exactly match parentObjectId relationships`,
    );
  });

  for (const object of objects) {
    const seen = new Set();
    let cursor = object.id;
    while (cursor != null) {
      requireCondition(!seen.has(cursor), "interactive object hierarchy contains a cycle");
      seen.add(cursor);
      cursor = parents.get(cursor) ?? null;
    }
  }
}

function validateSceneCommandService(service) {
  requireCondition(isRecord(service), "sceneCommandService must be an object");
  const allowedKeys = new Set(["endpoint", "contract"]);
  const unknownKeys = Object.keys(service).filter((key) => !allowedKeys.has(key));
  requireCondition(unknownKeys.length === 0, `sceneCommandService has unknown fields: ${unknownKeys.join(", ")}`);
  requireCondition(
    service.contract == null || service.contract === SCENE_COMMAND_SERVICE_CONTRACT,
    `sceneCommandService.contract must equal ${SCENE_COMMAND_SERVICE_CONTRACT}`,
  );
  requireCondition(
    typeof service.endpoint === "string"
      && service.endpoint.length > 0
      && service.endpoint.length <= 2048
      && service.endpoint === service.endpoint.trim(),
    "sceneCommandService.endpoint must be a trimmed non-empty string",
  );
  requireCondition(!/[\u0000-\u0020\\]/u.test(service.endpoint), "sceneCommandService.endpoint contains unsafe characters");
  let parsed;
  try {
    parsed = new URL(service.endpoint, "https://video2world.invalid/");
  } catch {
    requireCondition(false, "sceneCommandService.endpoint must be a valid URL");
  }
  requireCondition(["http:", "https:"].includes(parsed.protocol), "sceneCommandService.endpoint must use HTTP(S)");
  requireCondition(!parsed.username && !parsed.password, "sceneCommandService.endpoint must not contain credentials");
  requireCondition(!parsed.hash && !parsed.search, "sceneCommandService.endpoint must not contain a query or fragment");
  requireCondition(
    parsed.pathname.endsWith("/v1/scene-commands/submit"),
    "sceneCommandService.endpoint must target /v1/scene-commands/submit",
  );
  const explicitHttp = /^http:/iu.test(service.endpoint);
  const localHttpHosts = new Set(["localhost", "127.0.0.1", "[::1]"]);
  requireCondition(
    !explicitHttp || localHttpHosts.has(parsed.hostname),
    "sceneCommandService.endpoint must use HTTPS outside localhost",
  );
}

export function validateWebManifest(value) {
  requireCondition(isRecord(value), "root must be an object");
  requireCondition(value.schemaVersion === WEB_MANIFEST_SCHEMA_VERSION, "schemaVersion must equal 1");
  requireCondition(value.contract === WEB_MANIFEST_CONTRACT, `contract must equal ${WEB_MANIFEST_CONTRACT}`);
  requireCondition(typeof value.version === "string" && value.version.length > 0, "version is required");
  requireCondition(isRecord(value.coordinateSystem), "coordinateSystem is required");
  requireCondition(
    isSupportedWorldUp(value.coordinateSystem.worldUp),
    "coordinateSystem.worldUp must be [0, 1, 0] or [0, -1, 0]",
  );
  requireCondition(isRecord(value.assets), "assets is required");
  validateAsset(value.assets.visual, "assets.visual", { allowMetadataOnly: true });
  validateAsset(value.assets.collider, "assets.collider", { allowMetadataOnly: true });
  if (value.assets.colliderStaticCarved != null) {
    validateAsset(value.assets.colliderStaticCarved, "assets.colliderStaticCarved");
  }
  requireCondition(Array.isArray(value.interactiveObjects), "interactiveObjects must be an array");
  const ids = new Set();
  value.interactiveObjects.forEach((object, index) => validateInteractiveObject(object, index, ids));
  validateInteractiveObjectHierarchy(value.interactiveObjects, ids);
  requireCondition(isRecord(value.sceneKnowledge), "sceneKnowledge is required");
  requireCondition(Array.isArray(value.sceneKnowledge.objects), "sceneKnowledge.objects must be an array");
  if (value.sceneCommandService != null) {
    validateSceneCommandService(value.sceneCommandService);
  }
  if (value.sourceWorld != null) {
    requireCondition(isRecord(value.sourceWorld), "sourceWorld must be an object");
    requireCondition(typeof value.sourceWorld.worldId === "string", "sourceWorld.worldId");
    requireCondition(typeof value.sourceWorld.runId === "string", "sourceWorld.runId");
    requireCondition(typeof value.sourceWorld.adoptionMode === "string", "sourceWorld.adoptionMode");
    if (value.sourceWorld.manifestSha256 != null) {
      requireCondition(
        typeof value.sourceWorld.manifestSha256 === "string"
          && /^[a-f0-9]{64}$/u.test(value.sourceWorld.manifestSha256),
        "sourceWorld.manifestSha256 must be 64 lowercase hexadecimal characters",
      );
    }
  }
  return value;
}

export {
  MAX_UNIFIED_GLTF_COLLISION_FACES,
  SCENE_COMMAND_SERVICE_CONTRACT,
  WEB_MANIFEST_CONTRACT,
  WEB_MANIFEST_SCHEMA_VERSION,
};
