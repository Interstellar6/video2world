#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import { validateWebManifest } from "../web/web-manifest.js";
import {
  carveGraphdecoPly,
  carveTriangleMeshPly,
  chunkBuffer,
  describeTriangleMeshPly,
  expandedBounds,
  sha256,
  sha256File,
} from "./lib/strict-clean-scene-assets.mjs";

export const LEGACY_CARVE_OBJECT_IDS = Object.freeze([
  "sam3_nightstand_01",
  "sam3_nightstand_02",
  "sam3_plant_01",
  "sam3_plant_02",
]);

export const CLEAN_PLATE_UNIFIED_OBJECT_IDS = Object.freeze([
  "sam3_pillow_front",
  "sam3_pillow_left",
  "sam3_pillow_right",
  "sam3_bed_01",
]);

const CLEAN_PLATE_PILLOW_OBJECT_IDS = Object.freeze(
  CLEAN_PLATE_UNIFIED_OBJECT_IDS.slice(0, 3),
);

const ALIGNMENT_INPUT_KEYS = Object.freeze([
  "strict_sequence_report",
  "reconstruction_input_manifest",
  "reference_cameras",
  "fresh_cameras",
  "pgsr_ply",
  "tsdf_ply",
  "pgsr_stage_receipt",
  "tsdf_stage_receipt",
]);

const ALIGNMENT_GATE_KEYS = Object.freeze([
  "camera_calibration_hash_bound",
  "reference_subset_exact",
  "pgsr_and_tsdf_share_frame",
  "object_placements_share_target_frame",
  "transform_finite",
  "identity_or_verified_similarity",
  "source_stage_receipts_unchanged",
]);

const IDENTITY_ROTATION = Object.freeze([
  Object.freeze([1, 0, 0]),
  Object.freeze([0, 1, 0]),
  Object.freeze([0, 0, 1]),
]);

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const defaultProjectRoot = path.resolve(scriptDir, "..");

function requireCondition(condition, message) {
  if (!condition) throw new Error(message);
}

function parseArgs(argv) {
  const options = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    requireCondition(token.startsWith("--"), `Unexpected argument: ${token}`);
    const key = token.slice(2);
    const value = argv[index + 1];
    requireCondition(value != null && !value.startsWith("--"), `Missing value for --${key}`);
    requireCondition(options[key] == null, `Duplicate option --${key}`);
    options[key] = value;
    index += 1;
  }
  return options;
}

function required(options, key) {
  requireCondition(options[key] != null && options[key] !== "", `Missing required option --${key}`);
  return options[key];
}

function readJson(filePath, label = filePath) {
  let value;
  try {
    value = JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch (error) {
    throw new Error(`${label}: cannot read JSON: ${error.message}`);
  }
  requireCondition(value && typeof value === "object" && !Array.isArray(value), `${label}: root must be an object`);
  return value;
}

function writeJson(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  const temporary = `${filePath}.${process.pid}.${Date.now()}.tmp`;
  fs.writeFileSync(temporary, `${JSON.stringify(value, null, 2)}\n`);
  fs.renameSync(temporary, filePath);
}

function isSha256(value) {
  return typeof value === "string" && /^[a-f0-9]{64}$/u.test(value);
}

function requireSha256(value, label) {
  requireCondition(isSha256(value), `${label} must be a lowercase SHA-256 digest`);
  return value;
}

function sameStringSet(actual, expected) {
  return actual.length === expected.length
    && [...actual].sort().every((value, index) => value === [...expected].sort()[index]);
}

function resolveInputPath(value, relativeTo, label) {
  requireCondition(typeof value === "string" && value.length > 0, `${label}.path is missing`);
  return path.resolve(relativeTo, value);
}

function verifyFile(filePath, { size, sha: expectedSha, label }) {
  requireCondition(fs.existsSync(filePath), `${label}: missing ${filePath}`);
  const stat = fs.statSync(filePath);
  requireCondition(stat.isFile(), `${label}: not a regular file ${filePath}`);
  if (size != null) {
    requireCondition(Number.isSafeInteger(size) && size >= 0, `${label}: invalid expected size`);
    requireCondition(stat.size === size, `${label}: expected ${size} bytes, got ${stat.size}`);
  }
  const digest = sha256File(filePath);
  if (expectedSha != null) {
    requireSha256(expectedSha, `${label}.sha256`);
    requireCondition(digest === expectedSha, `${label}: SHA-256 mismatch`);
  }
  return { size: stat.size, sha256: digest };
}

function validateAlignmentReceipt({
  receiptPath,
  expectedReceiptSha,
  expectedCameraSha,
  expectedTargetFrame,
  pgsrPath,
  tsdfPath,
}) {
  const verifiedReceipt = verifyFile(receiptPath, {
    sha: expectedReceiptSha,
    label: "alignment receipt",
  });
  const receipt = readJson(receiptPath, "alignment receipt");
  requireCondition(receipt.schema_version === 1, "alignment receipt schema_version must equal 1");
  requireCondition(
    receipt.kind === "video2world.clean_scene_alignment_receipt",
    "alignment receipt kind is invalid",
  );
  requireCondition(receipt.status === "passed_current_demo_only", "alignment receipt status is invalid");
  requireCondition(receipt.acceptance_scope === "current_demo_only", "alignment scope is invalid");
  requireCondition(receipt.promotion_approved === false, "alignment promotion must remain false");
  requireCondition(
    receipt.method === "identity_same_camera_calibration",
    "alignment method must be identity_same_camera_calibration",
  );
  requireCondition(
    typeof receipt.source_coordinate_frame === "string" && receipt.source_coordinate_frame.length > 0,
    "alignment source coordinate frame is missing",
  );
  requireCondition(
    receipt.target_coordinate_frame === expectedTargetFrame,
    "alignment target coordinate frame mismatch",
  );
  requireCondition(
    receipt.canonical_camera_subset_sha256 === expectedCameraSha,
    "alignment canonical camera SHA-256 mismatch",
  );
  const transform = receipt.transform;
  requireCondition(transform && typeof transform === "object", "alignment transform is missing");
  requireCondition(transform.scale === 1, "non-identity alignment scale is forbidden; rebake/refit required");
  requireCondition(
    JSON.stringify(transform.rotationMatrix) === JSON.stringify(IDENTITY_ROTATION),
    "non-identity alignment rotation is forbidden; rebake/refit required",
  );
  requireCondition(
    JSON.stringify(transform.translation) === JSON.stringify([0, 0, 0]),
    "non-identity alignment translation is forbidden; rebake/refit required",
  );
  const gates = receipt.gates;
  requireCondition(gates && typeof gates === "object" && !Array.isArray(gates), "alignment gates are missing");
  requireCondition(
    sameStringSet(Object.keys(gates), ALIGNMENT_GATE_KEYS),
    "alignment gate key set is invalid",
  );
  for (const key of ALIGNMENT_GATE_KEYS) {
    requireCondition(gates[key] === true, `alignment gate failed: ${key}`);
  }
  const comparison = receipt.camera_comparison;
  requireCondition(comparison && typeof comparison === "object", "alignment camera comparison is missing");
  requireCondition(
    Number.isFinite(comparison.maximum_absolute_delta)
      && Number.isFinite(comparison.absolute_tolerance)
      && comparison.maximum_absolute_delta <= comparison.absolute_tolerance,
    "alignment camera comparison exceeds its tolerance",
  );
  requireCondition(
    Number.isInteger(comparison.fresh_camera_count)
      && comparison.fresh_camera_count > 0
      && Array.isArray(comparison.fresh_img_names)
      && comparison.fresh_img_names.length === comparison.fresh_camera_count,
    "alignment fresh camera set is invalid",
  );
  const inputs = receipt.inputs;
  requireCondition(inputs && typeof inputs === "object" && !Array.isArray(inputs), "alignment inputs are missing");
  requireCondition(
    sameStringSet(Object.keys(inputs), ALIGNMENT_INPUT_KEYS),
    "alignment input key set is invalid",
  );
  const resolvedInputs = {};
  for (const key of ALIGNMENT_INPUT_KEYS) {
    const record = inputs[key];
    requireCondition(record && typeof record === "object" && !Array.isArray(record), `alignment input ${key}`);
    const inputPath = resolveInputPath(record.path, path.dirname(receiptPath), `alignment input ${key}`);
    const verified = verifyFile(inputPath, {
      sha: requireSha256(record.sha256, `alignment input ${key}.sha256`),
      label: `alignment input ${key}`,
    });
    resolvedInputs[key] = { path: inputPath, ...verified };
  }
  requireCondition(
    resolvedInputs.pgsr_ply.path === path.resolve(pgsrPath),
    "alignment receipt binds another PGSR PLY",
  );
  requireCondition(
    resolvedInputs.tsdf_ply.path === path.resolve(tsdfPath),
    "alignment receipt binds another TSDF PLY",
  );
  return { receipt, receiptSha256: verifiedReceipt.sha256, resolvedInputs };
}

function validateVec3(value, label) {
  requireCondition(
    Array.isArray(value) && value.length === 3 && value.every(Number.isFinite),
    `${label} must be a finite vec3`,
  );
}

function isExactFiniteVector(value, expected) {
  return Array.isArray(value)
    && value.length === expected.length
    && value.every((item, index) => Number.isFinite(item) && item === expected[index]);
}

function validateBakedPillowPlacement(objectId, object) {
  const placement = object.placement;
  requireCondition(
    placement.assetCoordinatesBaked === true,
    `${objectId}: placement.assetCoordinatesBaked must be true`,
  );
  requireCondition(
    !Object.prototype.hasOwnProperty.call(placement, "matrixRowMajor"),
    `${objectId}: placement.matrixRowMajor is forbidden for a baked pillow asset`,
  );
  requireCondition(
    isExactFiniteVector(placement.scale, [1, 1, 1]),
    `${objectId}: placement.scale must equal [1, 1, 1]`,
  );
  requireCondition(
    isExactFiniteVector(placement.rotationEulerDeg, [0, 0, 0]),
    `${objectId}: placement.rotationEulerDeg must equal [0, 0, 0]`,
  );
}

function validateUnbakedBedPlacement(objectId, object) {
  const placement = object.placement;
  requireCondition(
    placement.assetCoordinatesBaked === false,
    `${objectId}: placement.assetCoordinatesBaked must be false`,
  );
  const rows = placement.matrixRowMajor;
  requireCondition(
    Array.isArray(rows)
      && rows.length === 4
      && rows.every(
        (row) => Array.isArray(row) && row.length === 4 && row.every(Number.isFinite),
      ),
    `${objectId}: placement.matrixRowMajor must be a finite 4x4 matrix`,
  );
  requireCondition(
    isExactFiniteVector(rows[3], [0, 0, 0, 1]),
    `${objectId}: placement.matrixRowMajor last row must equal [0, 0, 0, 1]`,
  );
  const translation = rows.slice(0, 3).map((row) => row[3]);
  requireCondition(
    isExactFiniteVector(placement.pivot, translation),
    `${objectId}: placement.pivot must equal the matrix translation`,
  );
  requireCondition(
    Array.isArray(object.childObjectIds)
      && object.childObjectIds.every((childId) => typeof childId === "string")
      && sameStringSet(object.childObjectIds, CLEAN_PLATE_PILLOW_OBJECT_IDS),
    `${objectId}: childObjectIds must contain exactly the three approved pillows`,
  );
}

function validateCandidate(candidate, publicManifestSha) {
  const objects = candidate.interactiveObjects;
  requireCondition(
    Array.isArray(objects)
      && objects.every(
        (object) => object && typeof object === "object" && typeof object.id === "string",
      ),
    "candidate interactiveObjects must be an object array with string ids",
  );
  const ids = objects.map((object) => object.id);
  const expectedIds = [...LEGACY_CARVE_OBJECT_IDS, ...CLEAN_PLATE_UNIFIED_OBJECT_IDS];
  requireCondition(sameStringSet(ids, expectedIds), "candidate must contain exactly the eight approved objects");
  const byId = new Map(objects.map((object) => [object.id, object]));
  for (const objectId of LEGACY_CARVE_OBJECT_IDS) {
    const object = byId.get(objectId);
    const robust = object.sourceAnchor?.robustBounds;
    requireCondition(robust && typeof robust === "object", `${objectId}: robust source bounds are missing`);
    validateVec3(robust.min, `${objectId}.sourceAnchor.robustBounds.min`);
    validateVec3(robust.max, `${objectId}.sourceAnchor.robustBounds.max`);
    requireCondition(
      robust.min.every((value, axis) => value < robust.max[axis]),
      `${objectId}: robust source bounds have non-positive extent`,
    );
  }
  for (const objectId of CLEAN_PLATE_UNIFIED_OBJECT_IDS) {
    const object = byId.get(objectId);
    const isPillow = objectId !== "sam3_bed_01";
    requireCondition(object.collision?.mode === "unified-glb", `${objectId}: must use unified-glb`);
    requireCondition(object.visual == null, `${objectId}: unified object must not have a visual proxy`);
    requireCondition(object.collision.renderAsset == null, `${objectId}: must not have renderAsset`);
    requireCondition(object.colliderProxy == null, `${objectId}: must not have colliderProxy`);
    requireCondition(
      typeof object.collision.asset.sourcePath === "string",
      `${objectId}: collision.asset.sourcePath is required`,
    );
    requireCondition(
      object.semanticGranularity === (isPillow ? "independent_child_asset" : "independent_root_asset"),
      `${objectId}: semantic hierarchy role is invalid`,
    );
    requireCondition(
      object.parentObjectId === (isPillow ? "sam3_bed_01" : null),
      `${objectId}: parentObjectId is invalid`,
    );
    requireCondition(object.movesWithParent === isPillow, `${objectId}: movesWithParent is invalid`);
    requireCondition(object.independentlyMovable === true, `${objectId}: must remain independently movable`);
    requireCondition(
      object.collision.topology === (isPillow ? "surface_bvh" : "closed_volume"),
      `${objectId}: collision topology is invalid`,
    );
    requireCondition(
      object.collision.asset.watertight === !isPillow,
      `${objectId}: watertight contract is invalid`,
    );
    if (isPillow) validateBakedPillowPlacement(objectId, object);
    else validateUnbakedBedPlacement(objectId, object);
  }
  const declaredUnified = candidate.candidateBuild?.unifiedPbrObjects?.objectIds;
  requireCondition(
    Array.isArray(declaredUnified) && sameStringSet(declaredUnified, CLEAN_PLATE_UNIFIED_OBJECT_IDS),
    "candidateBuild unified object set is invalid",
  );
  const hierarchy = candidate.candidateBuild?.unifiedPbrObjects?.hierarchy;
  requireCondition(
    hierarchy?.parent === "sam3_bed_01"
      && Array.isArray(hierarchy.children)
      && sameStringSet(hierarchy.children, CLEAN_PLATE_UNIFIED_OBJECT_IDS.slice(0, 3))
      && hierarchy.parentMotionCarriesChildren === true
      && hierarchy.childrenRemainIndependentlyMovable === true,
    "candidateBuild unified object hierarchy is invalid",
  );
  validateWebManifest(candidate);
  requireCondition(
    candidate.candidateBuild?.cleanScene?.methodRequired
      === "front_to_back_cumulative_peel_then_clean_plate_reconstruction",
    "candidate does not require the strict layered clean-scene method",
  );
  requireCondition(
    candidate.candidateBuild?.inheritedStaticScene?.baseManifestSha256 === publicManifestSha,
    "candidate baseline manifest SHA-256 does not match the public manifest",
  );
  return byId;
}

function normalizeUrlPrefix(value, label) {
  requireCondition(typeof value === "string" && value.length > 0, `${label} is empty`);
  requireCondition(!value.includes("\\") && !value.includes("?") && !value.includes("#"), `${label} is unsafe`);
  const output = value.endsWith("/") ? value.slice(0, -1) : value;
  requireCondition(output.startsWith("./worlds/"), `${label} must start with ./worlds/`);
  requireCondition(!output.split("/").includes(".."), `${label} cannot contain ..`);
  return output;
}

function localPathFromWorldUrl(url, urlPrefix, worldDir, label) {
  requireCondition(typeof url === "string" && url.startsWith(`${urlPrefix}/`), `${label}: URL prefix mismatch`);
  const relative = url.slice(urlPrefix.length + 1);
  requireCondition(relative.length > 0 && !relative.includes("\\"), `${label}: unsafe URL`);
  const segments = relative.split("/");
  requireCondition(!segments.includes("") && !segments.includes("..") && !segments.includes("."), `${label}: unsafe URL`);
  const resolved = path.resolve(worldDir, ...segments);
  requireCondition(resolved.startsWith(`${path.resolve(worldDir)}${path.sep}`), `${label}: URL escapes world dir`);
  return resolved;
}

function verifyWorldAsset(asset, worldDir, urlPrefix, label) {
  requireCondition(asset && typeof asset === "object", `${label}: asset is missing`);
  if (Array.isArray(asset.parts) && asset.parts.length > 0) {
    const whole = [];
    let total = 0;
    for (let index = 0; index < asset.parts.length; index += 1) {
      const part = asset.parts[index];
      const filePath = localPathFromWorldUrl(part.url, urlPrefix, worldDir, `${label}.parts[${index}]`);
      const verified = verifyFile(filePath, {
        size: part.size,
        sha: part.sha256,
        label: `${label}.parts[${index}]`,
      });
      whole.push(fs.readFileSync(filePath));
      total += verified.size;
    }
    if (asset.size != null) requireCondition(total === asset.size, `${label}: chunked size mismatch`);
    if (asset.sha256 != null) {
      requireCondition(sha256(Buffer.concat(whole, total)) === asset.sha256, `${label}: chunked SHA-256 mismatch`);
    }
    return { size: total, sha256: sha256(Buffer.concat(whole, total)) };
  }
  requireCondition(typeof asset.url === "string", `${label}: asset has no URL or parts`);
  const filePath = localPathFromWorldUrl(asset.url, urlPrefix, worldDir, label);
  return verifyFile(filePath, { size: asset.size, sha: asset.sha256, label });
}

function verifyLegacyCandidateAssets(candidateById, sourceWorldDir, sourceUrlPrefix) {
  for (const objectId of LEGACY_CARVE_OBJECT_IDS) {
    const object = candidateById.get(objectId);
    verifyWorldAsset(object.visual, sourceWorldDir, sourceUrlPrefix, `${objectId}.visual`);
    verifyWorldAsset(object.collision.asset, sourceWorldDir, sourceUrlPrefix, `${objectId}.collision.asset`);
    if (object.collision.renderAsset != null) {
      verifyWorldAsset(
        object.collision.renderAsset,
        sourceWorldDir,
        sourceUrlPrefix,
        `${objectId}.collision.renderAsset`,
      );
    }
    if (object.sourceAnchor?.path != null) {
      const anchorPath = localPathFromWorldUrl(
        object.sourceAnchor.path,
        sourceUrlPrefix,
        sourceWorldDir,
        `${objectId}.sourceAnchor`,
      );
      verifyFile(anchorPath, {
        size: object.sourceAnchor.size,
        sha: object.sourceAnchor.sha256,
        label: `${objectId}.sourceAnchor`,
      });
    }
  }
}

function recursivelyCollectWorldUrls(value, urlPrefix, output = []) {
  if (typeof value === "string") {
    if (value.startsWith(`${urlPrefix}/`)) output.push(value);
  } else if (Array.isArray(value)) {
    value.forEach((item) => recursivelyCollectWorldUrls(item, urlPrefix, output));
  } else if (value && typeof value === "object") {
    Object.values(value).forEach((item) => recursivelyCollectWorldUrls(item, urlPrefix, output));
  }
  return output;
}

function assertCandidateSourceUrls(candidate, sourceWorldDir, sourceUrlPrefix, allowedMissing) {
  const missing = [];
  for (const url of new Set(recursivelyCollectWorldUrls(candidate, sourceUrlPrefix))) {
    const filePath = localPathFromWorldUrl(url, sourceUrlPrefix, sourceWorldDir, "candidate URL");
    if (!fs.existsSync(filePath) && !allowedMissing.has(url)) missing.push(url);
  }
  requireCondition(missing.length === 0, `candidate baseline URLs are missing: ${missing.join(", ")}`);
}

function rewriteWorldPrefix(value, sourcePrefix, outputPrefix) {
  if (typeof value === "string") {
    return value.startsWith(`${sourcePrefix}/`)
      ? `${outputPrefix}/${value.slice(sourcePrefix.length + 1)}`
      : value;
  }
  if (Array.isArray(value)) return value.map((item) => rewriteWorldPrefix(item, sourcePrefix, outputPrefix));
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([key, item]) => [key, rewriteWorldPrefix(item, sourcePrefix, outputPrefix)]),
    );
  }
  return value;
}

function materializeFile(source, target, storageMode) {
  fs.mkdirSync(path.dirname(target), { recursive: true });
  if (fs.existsSync(target)) fs.unlinkSync(target);
  if (storageMode === "hardlink") fs.linkSync(source, target);
  else fs.copyFileSync(source, target);
}

function materializeTree(sourceDir, targetDir, storageMode) {
  fs.mkdirSync(targetDir, { recursive: true });
  for (const entry of fs.readdirSync(sourceDir, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) {
    const source = path.join(sourceDir, entry.name);
    const target = path.join(targetDir, entry.name);
    requireCondition(!entry.isSymbolicLink(), `source world cannot contain symlinks: ${source}`);
    if (entry.isDirectory()) materializeTree(source, target, storageMode);
    else if (entry.isFile()) materializeFile(source, target, storageMode);
    else throw new Error(`source world contains an unsupported file type: ${source}`);
  }
}

function resolveRepoSource(sourcePath, projectRoot, label) {
  const prefix = "repo://video2world/";
  requireCondition(sourcePath.startsWith(prefix), `${label}: only repo://video2world sources are accepted`);
  const relative = sourcePath.slice(prefix.length);
  requireCondition(relative.length > 0 && !relative.split("/").includes(".."), `${label}: unsafe sourcePath`);
  const resolved = path.resolve(projectRoot, ...relative.split("/"));
  requireCondition(resolved.startsWith(`${projectRoot}${path.sep}`), `${label}: sourcePath escapes project root`);
  return resolved;
}

function materializeUnifiedObjects(manifest, projectRoot, temporaryWorld, outputUrlPrefix, storageMode) {
  const outputs = {};
  const byId = new Map(manifest.interactiveObjects.map((object) => [object.id, object]));
  for (const objectId of CLEAN_PLATE_UNIFIED_OBJECT_IDS) {
    const object = byId.get(objectId);
    const asset = object.collision.asset;
    const source = resolveRepoSource(asset.sourcePath, projectRoot, `${objectId}.collision.asset`);
    const verified = verifyFile(source, {
      size: asset.size,
      sha: asset.sha256,
      label: `${objectId}.collision.asset source`,
    });
    requireCondition(
      typeof asset.fileName === "string" && asset.fileName === path.basename(asset.fileName),
      `${objectId}: asset.fileName must be a safe basename`,
    );
    const target = path.join(temporaryWorld, "objects", asset.fileName);
    materializeFile(source, target, storageMode);
    const copied = verifyFile(target, {
      size: verified.size,
      sha: verified.sha256,
      label: `${objectId}.collision.asset output`,
    });
    asset.url = `${outputUrlPrefix}/objects/${asset.fileName}`;
    asset.size = copied.size;
    asset.sha256 = copied.sha256;
    object.collision.gate = {
      ...object.collision.gate,
      candidateBrowserQa: "pending_strict_clean_scene_world",
    };
    if (Array.isArray(object.limitations)) {
      object.limitations = object.limitations
        .filter((item) => !item.includes("inherited static scene"))
        .map((item) => item.replace("Candidate asset transport", "Candidate browser"));
    }
    outputs[objectId] = {
      fileName: asset.fileName,
      size: copied.size,
      sha256: copied.sha256,
      faces: asset.faces,
      mode: object.collision.mode,
      visualProxy: false,
      colliderProxy: false,
    };
  }
  return outputs;
}

function carveBoundsFromCandidate(manifest, margin) {
  const byId = new Map(manifest.interactiveObjects.map((object) => [object.id, object]));
  return LEGACY_CARVE_OBJECT_IDS.map((objectId) => ({
    id: objectId,
    bounds: expandedBounds(byId.get(objectId).sourceAnchor.robustBounds, margin),
  }));
}

function makeVisualDescriptor(carve, parts, sourceSha, chunkSize, carveMargin) {
  return {
    id: "bedroom4_strict_clean_pgsr_static_legacy_carved",
    label: "Bedroom 4 strict layered clean-scene PGSR with legacy interactive instances carved",
    fileName: "bedroom4_strict_clean_pgsr_static_legacy_carved.ply",
    fileType: "ply",
    format: "graphdeco-gaussian-ply",
    sourcePath: "alignment-receipt://inputs/pgsr_ply",
    sourceSha256: sourceSha,
    size: carve.bytes.length,
    sha256: sha256(carve.bytes),
    headerByteLength: carve.headerByteLength,
    vertexCount: carve.vertexCount,
    inputVertexCount: carve.inputVertexCount,
    removedVertexCount: carve.removedVertexCount,
    removedByObject: carve.removedByObject,
    vertexStride: carve.vertexStride,
    vertexProperties: carve.vertexProperties,
    bbox: carve.bbox,
    carveMethod: "expanded_source_anchor_aabb_point_containment",
    carveMargin,
    carvedObjectIds: [...LEGACY_CARVE_OBJECT_IDS],
    protectedCleanPlateObjectIds: [...CLEAN_PLATE_UNIFIED_OBJECT_IDS],
    chunkSize,
    parts,
    transportMode: "verified_chunks",
    directOversizedAssetPublished: false,
    acceptanceScope: "current_demo_only",
    promotionApproved: false,
  };
}

function makeRawColliderDescriptor(mesh, parts, sourceSha, chunkSize) {
  return {
    id: "bedroom4_strict_clean_tsdf_raw",
    label: "Bedroom 4 strict layered clean-scene raw TSDF mesh",
    fileName: "bedroom4_strict_clean_tsdf_raw.ply",
    fileType: "ply",
    format: "open3d-binary-little-endian-triangle-mesh-ply",
    sourcePath: "alignment-receipt://inputs/tsdf_ply",
    sourceSha256: sourceSha,
    size: mesh.bytes.length,
    sha256: sha256(mesh.bytes),
    headerByteLength: mesh.headerByteLength,
    vertexCount: mesh.vertexCount,
    faceCount: mesh.faceCount,
    vertexStride: mesh.vertexStride,
    vertexProperties: mesh.vertexProperties,
    vertexPositionType: mesh.vertexPositionType,
    faceIndexType: mesh.faceIndexType,
    bbox: mesh.bbox,
    chunkSize,
    parts,
    transportMode: "verified_chunks",
    directOversizedAssetPublished: false,
    note: "Raw fresh TSDF bounds reference; character collision loads colliderStaticCarved.",
  };
}

function makeCarvedColliderDescriptor(carve, parts, sourceSha, chunkSize, carveMargin) {
  return {
    id: "bedroom4_strict_clean_tsdf_static_legacy_carved",
    label: "Bedroom 4 strict layered clean-scene TSDF with legacy interactive instances carved",
    fileName: "bedroom4_strict_clean_tsdf_static_legacy_carved.ply",
    fileType: "ply",
    format: "open3d-binary-little-endian-triangle-mesh-ply",
    sourcePath: "alignment-receipt://inputs/tsdf_ply",
    sourceSha256: sourceSha,
    size: carve.bytes.length,
    sha256: sha256(carve.bytes),
    headerByteLength: carve.headerByteLength,
    vertexCount: carve.vertexCount,
    originalFaceCount: carve.originalFaceCount,
    faceCount: carve.faceCount,
    removedFaceCount: carve.removedFaceCount,
    removedByObject: carve.removedByObject,
    vertexStride: carve.vertexStride,
    vertexProperties: carve.vertexProperties,
    vertexPositionType: carve.vertexPositionType,
    faceIndexType: carve.faceIndexType,
    faceIndicesValid: carve.faceIndicesValid,
    finite: carve.finite,
    bbox: carve.bbox,
    carveMethod: "expanded_source_anchor_aabb_triangle_intersection",
    carveMargin,
    carvedObjectIds: [...LEGACY_CARVE_OBJECT_IDS],
    protectedCleanPlateObjectIds: [...CLEAN_PLATE_UNIFIED_OBJECT_IDS],
    chunkSize,
    parts,
    transportMode: "verified_chunks",
    directOversizedAssetPublished: false,
    acceptanceScope: "current_demo_only",
    promotionApproved: false,
  };
}

function updateManifestForStrictCleanScene({
  manifest,
  alignment,
  visualDescriptor,
  colliderDescriptor,
  staticColliderDescriptor,
  publicManifestSha,
  candidateManifestSha,
  reportUrl,
  carveMargin,
  chunkSize,
}) {
  manifest.version = "bedroom4-candidate-strict-layered-clean-scene-unified-pbr-four-objects-20260717";
  manifest.alignment = {
    id: alignment.receipt.target_coordinate_frame,
    method: alignment.receipt.method,
    note: "Identity alignment is admitted only by the hash-bound clean-scene alignment receipt.",
    scale: 1,
    rotationMatrix: IDENTITY_ROTATION.map((row) => [...row]),
    translation: [0, 0, 0],
    canonicalCameraSubsetSha256: alignment.receipt.canonical_camera_subset_sha256,
    receiptSha256: alignment.receiptSha256,
  };
  manifest.assets.visual = visualDescriptor;
  manifest.assets.collider = colliderDescriptor;
  manifest.assets.colliderStaticCarved = staticColliderDescriptor;
  manifest.chunkSize = chunkSize;
  manifest.collisionWorld = {
    baselineBoundsAssetKey: "collider",
    sceneAssetKey: "colliderStaticCarved",
    objectFaceRemovalRequired: true,
    replacementMode: "strict_clean_tsdf_then_legacy_four_expanded_source_anchor_aabb_carve",
    gate: {
      status: "passed_current_demo_only",
      carveMethod: staticColliderDescriptor.carveMethod,
      carveMargin,
      originalFaces: staticColliderDescriptor.originalFaceCount,
      staticFaces: staticColliderDescriptor.faceCount,
      removedFaces: staticColliderDescriptor.removedFaceCount,
      removedByObject: staticColliderDescriptor.removedByObject,
      carvedObjectIds: [...LEGACY_CARVE_OBJECT_IDS],
      protectedCleanPlateObjectIds: [...CLEAN_PLATE_UNIFIED_OBJECT_IDS],
      faceIndicesValid: true,
      finite: true,
    },
    visualCarveGate: {
      status: "passed_current_demo_only",
      carveMethod: visualDescriptor.carveMethod,
      inputGaussians: visualDescriptor.inputVertexCount,
      staticGaussians: visualDescriptor.vertexCount,
      removedGaussians: visualDescriptor.removedVertexCount,
      removedByObject: visualDescriptor.removedByObject,
      carvedObjectIds: [...LEGACY_CARVE_OBJECT_IDS],
      protectedCleanPlateObjectIds: [...CLEAN_PLATE_UNIFIED_OBJECT_IDS],
    },
    sceneAssetTransport: {
      mode: "verified_chunks",
      chunkSize,
      directOversizedAssetPublished: false,
      parts: staticColliderDescriptor.parts,
    },
  };
  if (manifest.presentation && typeof manifest.presentation === "object") {
    manifest.presentation.referenceBoundsAssetKey = "collider";
  }
  const build = manifest.candidateBuild;
  build.cleanScene = {
    status: "strict_layered_clean_scene_materialized_current_demo_only",
    claim: "The static scene comes from the strict front-to-back four-round clean plate followed by fresh depth, PGSR, and TSDF reconstruction.",
    method: "front_to_back_cumulative_peel_then_clean_plate_reconstruction",
    removedObjectOrder: [...CLEAN_PLATE_UNIFIED_OBJECT_IDS],
    freshReconstruction: true,
    alignmentReceiptSha256: alignment.receiptSha256,
    alignmentStatus: alignment.receipt.status,
    canonicalCameraSubsetSha256: alignment.receipt.canonical_camera_subset_sha256,
    pgsrSha256: alignment.resolvedInputs.pgsr_ply.sha256,
    tsdfSha256: alignment.resolvedInputs.tsdf_ply.sha256,
    staticLegacyCarveObjectIds: [...LEGACY_CARVE_OBJECT_IDS],
    cleanPlateObjectIdsNotCarvedAgain: [...CLEAN_PLATE_UNIFIED_OBJECT_IDS],
    acceptanceScope: "current_demo_only",
    promotionApproved: false,
  };
  build.inheritedStaticScene = {
    baseManifestSha256: publicManifestSha,
    candidateManifestSha256: candidateManifestSha,
    baselineWorldCopiedForLegacyRuntimeAssets: true,
    sceneGeometryReplacedByStrictFreshReconstruction: true,
    modifiedByThisCandidate: true,
  };
  build.promotionAllowed = false;
  build.promotionBlockers = ["candidate_browser_qa_pending", "public_manifest_promotion_not_requested"];
  build.report = reportUrl;
  build.status = "candidate_materialized_strict_clean_scene_browser_qa_pending";
  build.unifiedPbrObjects = {
    ...build.unifiedPbrObjects,
    assetTransport: "materialized_in_candidate_world",
    objectIds: [...CLEAN_PLATE_UNIFIED_OBJECT_IDS],
    status: "geometry_joint_qa_and_candidate_transport_passed_browser_qa_pending",
  };
  manifest.sourceWorld = {
    ...(manifest.sourceWorld || {}),
    adoptionMode: "strict_layered_clean_scene_candidate_atomic_materialization",
    manifestSha256: publicManifestSha,
  };
  if (manifest.productionBuild && typeof manifest.productionBuild === "object") {
    manifest.productionBuild.strictCleanSceneCandidate = {
      status: "materialized_browser_qa_pending",
      supersedesInheritedStaticGeometryForThisCandidate: true,
      promotionApproved: false,
    };
  }
}

function assertAllWorldUrlsResolve(manifest, worldDir, urlPrefix) {
  const urls = [...new Set(recursivelyCollectWorldUrls(manifest, urlPrefix))].sort();
  const missing = [];
  for (const url of urls) {
    const filePath = localPathFromWorldUrl(url, urlPrefix, worldDir, "output manifest URL");
    if (!fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) missing.push(url);
  }
  requireCondition(missing.length === 0, `output manifest has unresolved URLs: ${missing.join(", ")}`);
  return urls;
}

function verifyOutputRuntimeAssets(manifest, worldDir, urlPrefix) {
  verifyWorldAsset(manifest.assets.visual, worldDir, urlPrefix, "assets.visual");
  verifyWorldAsset(manifest.assets.collider, worldDir, urlPrefix, "assets.collider");
  verifyWorldAsset(
    manifest.assets.colliderStaticCarved,
    worldDir,
    urlPrefix,
    "assets.colliderStaticCarved",
  );
  for (const object of manifest.interactiveObjects) {
    if (object.visual != null) verifyWorldAsset(object.visual, worldDir, urlPrefix, `${object.id}.visual`);
    if (object.collision.asset != null) {
      verifyWorldAsset(object.collision.asset, worldDir, urlPrefix, `${object.id}.collision.asset`);
    }
    if (object.collision.renderAsset != null) {
      verifyWorldAsset(
        object.collision.renderAsset,
        worldDir,
        urlPrefix,
        `${object.id}.collision.renderAsset`,
      );
    }
  }
}

function ensureOutputBoundary(outputWorld, sourceWorld) {
  const output = path.resolve(outputWorld);
  const source = path.resolve(sourceWorld);
  requireCondition(output !== source, "output world must not overwrite source-world-dir");
  requireCondition(!output.startsWith(`${source}${path.sep}`), "output world must not be inside source-world-dir");
  requireCondition(!source.startsWith(`${output}${path.sep}`), "source-world-dir must not be inside output world");
  if (fs.existsSync(output)) {
    requireCondition(fs.statSync(output).isDirectory(), "output world exists and is not a directory");
    requireCondition(fs.readdirSync(output).length === 0, "output world must be absent or empty");
  }
}

export function materializeStrictCleanScene(options) {
  const projectRoot = path.resolve(options.projectRoot || defaultProjectRoot);
  const candidateManifestPath = path.resolve(options.candidateManifest);
  const pgsrPath = path.resolve(options.pgsrPly);
  const tsdfPath = path.resolve(options.tsdfPly);
  const alignmentReceiptPath = path.resolve(options.alignmentReceipt);
  const sourceWorldDir = path.resolve(options.sourceWorldDir);
  const outputWorld = path.resolve(options.outputWorld);
  const publicManifestPath = path.resolve(options.publicManifest || path.join(sourceWorldDir, "manifest.json"));
  const sourceUrlPrefix = normalizeUrlPrefix(
    options.sourceUrlPrefix || `./worlds/${path.basename(sourceWorldDir)}`,
    "source URL prefix",
  );
  const outputUrlPrefix = normalizeUrlPrefix(
    options.urlPrefix || `./worlds/${path.basename(outputWorld)}`,
    "output URL prefix",
  );
  const storageMode = options.storageMode || "hardlink";
  requireCondition(["copy", "hardlink"].includes(storageMode), "storage mode must be copy or hardlink");
  const chunkSize = Number(options.chunkSize || 1048576);
  requireCondition(Number.isSafeInteger(chunkSize) && chunkSize > 0, "chunk size must be positive");
  const carveMargin = Number(options.carveMargin || 0.18);
  requireCondition(Number.isFinite(carveMargin) && carveMargin >= 0, "carve margin must be non-negative");
  requireSha256(options.alignmentReceiptSha256, "alignment receipt SHA-256");
  requireSha256(options.expectedCameraCanonicalSha256, "expected camera canonical SHA-256");
  requireCondition(
    typeof options.expectedTargetCoordinateFrame === "string"
      && options.expectedTargetCoordinateFrame.length > 0,
    "expected target coordinate frame is required",
  );
  for (const [inputPath, label] of [
    [projectRoot, "project root"],
    [sourceWorldDir, "source world dir"],
  ]) {
    requireCondition(fs.existsSync(inputPath) && fs.statSync(inputPath).isDirectory(), `${label} is missing`);
  }
  for (const [inputPath, label] of [
    [candidateManifestPath, "candidate manifest"],
    [pgsrPath, "PGSR PLY"],
    [tsdfPath, "TSDF PLY"],
    [alignmentReceiptPath, "alignment receipt"],
    [publicManifestPath, "public manifest"],
  ]) {
    requireCondition(fs.existsSync(inputPath) && fs.statSync(inputPath).isFile(), `${label} is missing`);
  }
  ensureOutputBoundary(outputWorld, sourceWorldDir);
  requireCondition(
    !candidateManifestPath.startsWith(`${outputWorld}${path.sep}`),
    "candidate manifest cannot be inside output world",
  );

  const publicManifestShaBefore = sha256File(publicManifestPath);
  const sourceManifestPath = path.join(sourceWorldDir, "manifest.json");
  requireCondition(fs.existsSync(sourceManifestPath), "source-world-dir/manifest.json is missing");
  requireCondition(
    sha256File(sourceManifestPath) === publicManifestShaBefore,
    "source world manifest and public manifest differ",
  );
  validateWebManifest(readJson(sourceManifestPath, "source world manifest"));
  const candidateManifestSha = sha256File(candidateManifestPath);
  const candidate = readJson(candidateManifestPath, "candidate manifest");
  const candidateById = validateCandidate(candidate, publicManifestShaBefore);
  verifyLegacyCandidateAssets(candidateById, sourceWorldDir, sourceUrlPrefix);
  const unifiedUrls = new Set(
    CLEAN_PLATE_UNIFIED_OBJECT_IDS.map((objectId) => candidateById.get(objectId).collision.asset.url),
  );
  assertCandidateSourceUrls(candidate, sourceWorldDir, sourceUrlPrefix, unifiedUrls);

  const alignment = validateAlignmentReceipt({
    receiptPath: alignmentReceiptPath,
    expectedReceiptSha: options.alignmentReceiptSha256,
    expectedCameraSha: options.expectedCameraCanonicalSha256,
    expectedTargetFrame: options.expectedTargetCoordinateFrame,
    pgsrPath,
    tsdfPath,
  });
  const pgsrVerified = verifyFile(pgsrPath, {
    sha: alignment.resolvedInputs.pgsr_ply.sha256,
    label: "fresh PGSR PLY",
  });
  const tsdfVerified = verifyFile(tsdfPath, {
    sha: alignment.resolvedInputs.tsdf_ply.sha256,
    label: "fresh TSDF PLY",
  });

  const rewritten = rewriteWorldPrefix(candidate, sourceUrlPrefix, outputUrlPrefix);
  const bounds = carveBoundsFromCandidate(rewritten, carveMargin);
  const visualCarve = carveGraphdecoPly(pgsrPath, bounds);
  const rawMesh = describeTriangleMeshPly(tsdfPath);
  const colliderCarve = carveTriangleMeshPly(tsdfPath, bounds);
  const tempWorld = path.join(
    path.dirname(outputWorld),
    `.${path.basename(outputWorld)}.${process.pid}.${Date.now()}.tmp`,
  );
  requireCondition(!fs.existsSync(tempWorld), `temporary output already exists: ${tempWorld}`);
  try {
    materializeTree(sourceWorldDir, tempWorld, storageMode);
    const chunkDir = path.join(tempWorld, "chunks");
    const visualParts = chunkBuffer(
      visualCarve.bytes,
      "strict_clean_visual_pgsr_legacy_carved",
      chunkDir,
      outputUrlPrefix,
      chunkSize,
    );
    const rawColliderParts = chunkBuffer(
      rawMesh.bytes,
      "strict_clean_collider_tsdf_raw",
      chunkDir,
      outputUrlPrefix,
      chunkSize,
    );
    const staticColliderParts = chunkBuffer(
      colliderCarve.bytes,
      "strict_clean_collider_tsdf_legacy_carved",
      chunkDir,
      outputUrlPrefix,
      chunkSize,
    );
    const unifiedOutputs = materializeUnifiedObjects(
      rewritten,
      projectRoot,
      tempWorld,
      outputUrlPrefix,
      storageMode,
    );
    const visualDescriptor = makeVisualDescriptor(
      visualCarve,
      visualParts,
      pgsrVerified.sha256,
      chunkSize,
      carveMargin,
    );
    const colliderDescriptor = makeRawColliderDescriptor(
      rawMesh,
      rawColliderParts,
      tsdfVerified.sha256,
      chunkSize,
    );
    const staticColliderDescriptor = makeCarvedColliderDescriptor(
      colliderCarve,
      staticColliderParts,
      tsdfVerified.sha256,
      chunkSize,
      carveMargin,
    );
    const reportUrl = `${outputUrlPrefix}/qa/strict-clean-scene-adoption-report.json`;
    updateManifestForStrictCleanScene({
      manifest: rewritten,
      alignment,
      visualDescriptor,
      colliderDescriptor,
      staticColliderDescriptor,
      publicManifestSha: publicManifestShaBefore,
      candidateManifestSha,
      reportUrl,
      carveMargin,
      chunkSize,
    });
    validateWebManifest(rewritten);
    const outputManifestPath = path.join(tempWorld, "manifest.json");
    writeJson(outputManifestPath, rewritten);
    const outputManifestSha = sha256File(outputManifestPath);
    const reportPath = path.join(tempWorld, "qa", "strict-clean-scene-adoption-report.json");
    const report = {
      schemaVersion: 1,
      kind: "video2world.strict_clean_scene_adoption_report",
      status: "candidate_materialized_browser_qa_pending",
      acceptanceScope: "current_demo_only",
      promotionApproved: false,
      browserQa: "pending",
      inputs: {
        candidateManifest: { sha256: candidateManifestSha },
        publicManifest: { sha256: publicManifestShaBefore },
        alignmentReceipt: {
          sha256: alignment.receiptSha256,
          status: alignment.receipt.status,
          method: alignment.receipt.method,
          canonicalCameraSubsetSha256: alignment.receipt.canonical_camera_subset_sha256,
          sourceCoordinateFrame: alignment.receipt.source_coordinate_frame,
          targetCoordinateFrame: alignment.receipt.target_coordinate_frame,
          boundInputSha256: Object.fromEntries(
            ALIGNMENT_INPUT_KEYS.map((key) => [key, alignment.resolvedInputs[key].sha256]),
          ),
        },
        pgsrPly: { size: pgsrVerified.size, sha256: pgsrVerified.sha256 },
        tsdfPly: { size: tsdfVerified.size, sha256: tsdfVerified.sha256 },
      },
      carvePolicy: {
        method: "expanded_source_anchor_aabb",
        margin: carveMargin,
        carvedObjectIds: [...LEGACY_CARVE_OBJECT_IDS],
        cleanPlateObjectIdsNotCarvedAgain: [...CLEAN_PLATE_UNIFIED_OBJECT_IDS],
      },
      visual: {
        inputVertexCount: visualCarve.inputVertexCount,
        vertexCount: visualCarve.vertexCount,
        removedVertexCount: visualCarve.removedVertexCount,
        removedByObject: visualCarve.removedByObject,
        size: visualDescriptor.size,
        sha256: visualDescriptor.sha256,
        bbox: visualDescriptor.bbox,
        chunkCount: visualDescriptor.parts.length,
      },
      collision: {
        rawFaceCount: rawMesh.faceCount,
        staticFaceCount: colliderCarve.faceCount,
        removedFaceCount: colliderCarve.removedFaceCount,
        removedByObject: colliderCarve.removedByObject,
        rawSha256: colliderDescriptor.sha256,
        staticSha256: staticColliderDescriptor.sha256,
        bbox: staticColliderDescriptor.bbox,
        rawChunkCount: colliderDescriptor.parts.length,
        staticChunkCount: staticColliderDescriptor.parts.length,
      },
      unifiedObjects: unifiedOutputs,
      baselineWorld: {
        manifestSha256: publicManifestShaBefore,
        storageMode,
        copiedBeforeSceneReplacement: true,
        sourceWorldOverwritten: false,
      },
      outputManifest: { sha256: outputManifestSha },
      runtimeAssetResolution: { status: "pending_internal_validation", checkedUrlCount: 0 },
      gates: {
        alignmentReceiptHashBound: true,
        alignmentInputsRehashed: true,
        identityAlignmentOnly: true,
        strictLayeredCleanSceneUsed: true,
        onlyLegacyFourCarved: true,
        pillowsAndBedNotCarvedAgain: true,
        fourUnifiedPbrAssetsMaterialized: true,
        unifiedPlacementAndHierarchyContractValidated: true,
        publicManifestUnchanged: true,
        browserQaPending: true,
      },
    };
    writeJson(reportPath, report);
    const resolvedUrls = assertAllWorldUrlsResolve(rewritten, tempWorld, outputUrlPrefix);
    verifyOutputRuntimeAssets(rewritten, tempWorld, outputUrlPrefix);
    report.runtimeAssetResolution = {
      status: "passed",
      checkedUrlCount: resolvedUrls.length,
      allManifestWorldUrlsResolve: true,
      runtimeAssetHashesRevalidated: true,
    };
    const publicManifestShaAfter = sha256File(publicManifestPath);
    requireCondition(
      publicManifestShaAfter === publicManifestShaBefore,
      "public manifest changed during candidate materialization",
    );
    report.publicManifest = {
      sha256Before: publicManifestShaBefore,
      sha256After: publicManifestShaAfter,
      unchanged: true,
    };
    writeJson(reportPath, report);
    assertAllWorldUrlsResolve(rewritten, tempWorld, outputUrlPrefix);
    if (fs.existsSync(outputWorld)) fs.rmdirSync(outputWorld);
    fs.renameSync(tempWorld, outputWorld);
    requireCondition(
      sha256File(publicManifestPath) === publicManifestShaBefore,
      "public manifest changed after candidate materialization",
    );
    return {
      manifest: rewritten,
      report,
      outputWorld,
      manifestSha256: outputManifestSha,
      reportSha256: sha256File(path.join(outputWorld, "qa", "strict-clean-scene-adoption-report.json")),
    };
  } catch (error) {
    if (fs.existsSync(tempWorld)) fs.rmSync(tempWorld, { recursive: true, force: true });
    throw error;
  }
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const result = materializeStrictCleanScene({
    projectRoot: args["project-root"] || defaultProjectRoot,
    candidateManifest: required(args, "candidate-manifest"),
    pgsrPly: required(args, "pgsr-ply"),
    tsdfPly: required(args, "tsdf-ply"),
    alignmentReceipt: required(args, "alignment-receipt"),
    alignmentReceiptSha256: required(args, "alignment-receipt-sha256"),
    expectedCameraCanonicalSha256: required(args, "expected-camera-canonical-sha256"),
    expectedTargetCoordinateFrame: required(args, "expected-target-coordinate-frame"),
    sourceWorldDir: required(args, "source-world-dir"),
    publicManifest: args["public-manifest"],
    outputWorld: required(args, "output-world"),
    sourceUrlPrefix: args["source-url-prefix"],
    urlPrefix: args["url-prefix"],
    storageMode: args["storage-mode"],
    chunkSize: args["chunk-size"],
    carveMargin: args["carve-margin"],
  });
  process.stdout.write(`${JSON.stringify({
    status: result.report.status,
    outputWorld: result.outputWorld,
    manifestSha256: result.manifestSha256,
    reportSha256: result.reportSha256,
    promotionApproved: false,
    browserQa: "pending",
  })}\n`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    main();
  } catch (error) {
    process.stderr.write(`${error.stack || error.message}\n`);
    process.exitCode = 1;
  }
}
