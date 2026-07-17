#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import { validateWebManifest } from "../web/web-manifest.js";
import {
  CAMERA_OVERVIEW_COVERAGE_MAX,
  CAMERA_OVERVIEW_COVERAGE_MIN,
  CAMERA_TARGET_CENTER_NORMALIZED_MAX,
  MATRIX_GRAM_EPSILON,
  deriveManifestExpectations,
} from "./qa_bedroom4_clean_unified_scene_browser.mjs";
import { sha256File } from "./lib/strict-clean-scene-assets.mjs";

export const BEDROOM4_STABLE_ALIAS_SHA256 =
  "3805f0e5bab09add424b3b78f9349cd2eca6d1262777ef683e13cda07695e82b";

const EXPECTED_OBJECT_IDS = Object.freeze([
  "sam3_nightstand_01",
  "sam3_nightstand_02",
  "sam3_plant_01",
  "sam3_plant_02",
  "sam3_pillow_front",
  "sam3_pillow_left",
  "sam3_pillow_right",
  "sam3_bed_01",
]);

const UNIFIED_OBJECT_IDS = Object.freeze([
  "sam3_pillow_front",
  "sam3_pillow_left",
  "sam3_pillow_right",
  "sam3_bed_01",
]);

const PILLOW_OBJECT_IDS = Object.freeze(UNIFIED_OBJECT_IDS.slice(0, 3));
const BED_OBJECT_ID = "sam3_bed_01";
const MATRIX_EPSILON = 1e-5;
const IMPLEMENTATION_EVIDENCE = Object.freeze({
  qaRunner: "scripts/qa_bedroom4_clean_unified_scene_browser.mjs",
  runtime: "web/visual-physics-proxy.js",
  manifestValidator: "web/web-manifest.js",
  materializer: "scripts/materialize_bedroom4_strict_clean_scene.mjs",
});
const LOCK_STALE_AFTER_MS = 24 * 60 * 60 * 1000;

function requireCondition(condition, message) {
  if (!condition) throw new Error(message);
}

function parseArgs(argv) {
  const promotionOptions = new Set([
    "candidate-world",
    "candidate-browser-qa-report",
    "canonical-source-world",
    "backup-world",
    "public-manifest",
  ]);
  if (argv[0] === "--recover-journal") {
    requireCondition(argv.length === 2, "--recover-journal must be the only option");
    return { mode: "recover", journal: argv[1] };
  }
  const options = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    requireCondition(token.startsWith("--"), `Unexpected argument: ${token}`);
    const key = token.slice(2);
    requireCondition(promotionOptions.has(key), `Unsupported option --${key}`);
    requireCondition(options[key] == null, `Duplicate option --${key}`);
    const value = argv[index + 1];
    requireCondition(value != null && !value.startsWith("--"), `Missing value for --${key}`);
    options[key] = value;
    index += 1;
  }
  for (const key of promotionOptions) {
    requireCondition(options[key] != null, `Missing required option --${key}`);
  }
  return { mode: "promote", ...options };
}

function readJson(filePath, label) {
  let value;
  try {
    value = JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch (error) {
    throw new Error(`${label}: cannot read JSON: ${error.message}`);
  }
  requireCondition(value && typeof value === "object" && !Array.isArray(value), `${label}: root must be an object`);
  return value;
}

function fsyncDirectory(directory) {
  const descriptor = fs.openSync(directory, "r");
  try {
    fs.fsyncSync(descriptor);
  } finally {
    fs.closeSync(descriptor);
  }
}

function writeBytesDurable(filePath, bytes) {
  const parent = path.dirname(filePath);
  requireCondition(fs.existsSync(parent) && fs.statSync(parent).isDirectory(), `output parent is missing: ${parent}`);
  const temporary = `${filePath}.${process.pid}.${Date.now()}.tmp`;
  let descriptor = null;
  try {
    descriptor = fs.openSync(temporary, "wx", 0o600);
    fs.writeFileSync(descriptor, bytes);
    fs.fsyncSync(descriptor);
    fs.closeSync(descriptor);
    descriptor = null;
    fs.renameSync(temporary, filePath);
    fsyncDirectory(parent);
  } finally {
    if (descriptor != null) fs.closeSync(descriptor);
    if (fs.existsSync(temporary)) fs.unlinkSync(temporary);
  }
}

function writeJsonDurable(filePath, value) {
  writeBytesDurable(filePath, Buffer.from(`${JSON.stringify(value, null, 2)}\n`));
}

function isSha256(value) {
  return typeof value === "string" && /^[a-f0-9]{64}$/u.test(value);
}

function sameSet(actual, expected) {
  return actual.length === expected.length
    && [...actual].sort().every((value, index) => value === [...expected].sort()[index]);
}

function isInside(parent, child) {
  const relative = path.relative(parent, child);
  return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative));
}

function assertNoSymlinkAncestors(filePath, label, { allowMissingLeaf = false } = {}) {
  const absolute = path.resolve(filePath);
  const root = path.parse(absolute).root;
  let current = root;
  const segments = absolute.slice(root.length).split(path.sep).filter(Boolean);
  for (let index = 0; index < segments.length; index += 1) {
    current = path.join(current, segments[index]);
    let stat;
    try {
      stat = fs.lstatSync(current);
    } catch (error) {
      if (error?.code !== "ENOENT") throw error;
      requireCondition(allowMissingLeaf && index === segments.length - 1, `${label} is missing: ${current}`);
      break;
    }
    requireCondition(!stat.isSymbolicLink(), `${label} has a symlink ancestor: ${current}`);
  }
  return absolute;
}

function realDirectory(filePath, label) {
  const absolute = assertNoSymlinkAncestors(filePath, label);
  requireCondition(fs.statSync(absolute).isDirectory(), `${label} is not a directory: ${absolute}`);
  const real = fs.realpathSync(absolute);
  requireCondition(real === absolute, `${label} does not resolve to its literal path`);
  return real;
}

function ensureRegularFile(filePath, label) {
  const absolute = assertNoSymlinkAncestors(filePath, label);
  requireCondition(fs.existsSync(absolute), `${label} is missing: ${absolute}`);
  requireCondition(fs.statSync(absolute).isFile(), `${label} is not a regular file: ${absolute}`);
  requireCondition(fs.realpathSync(absolute) === absolute, `${label} does not resolve to its literal path`);
  return absolute;
}

function localWorldPath(url, prefix, worldDir, label) {
  requireCondition(typeof url === "string" && url.startsWith(`${prefix}/`), `${label}: URL prefix mismatch`);
  const relative = url.slice(prefix.length + 1);
  const segments = relative.split("/");
  requireCondition(
    relative.length > 0
      && !relative.includes("\\")
      && !/[?#%\u0000-\u0020]/u.test(relative)
      && !segments.includes("")
      && !segments.includes(".")
      && !segments.includes(".."),
    `${label}: unsafe URL`,
  );
  const browserUrl = new URL(url, "https://video2world.invalid/runtime/");
  const expectedPathname = `/runtime/${prefix.slice(2)}/${relative}`;
  requireCondition(
    browserUrl.origin === "https://video2world.invalid"
      && browserUrl.pathname === expectedPathname
      && browserUrl.search === ""
      && browserUrl.hash === "",
    `${label}: browser URL normalization changed the path`,
  );
  const resolved = path.resolve(worldDir, ...segments);
  requireCondition(isInside(worldDir, resolved) && resolved !== worldDir, `${label}: URL escapes world`);
  return resolved;
}

function collectPrefixUrls(value, prefix, output = []) {
  if (typeof value === "string") {
    if (value.startsWith(`${prefix}/`)) output.push(value);
  } else if (Array.isArray(value)) {
    value.forEach((item) => collectPrefixUrls(item, prefix, output));
  } else if (value && typeof value === "object") {
    Object.values(value).forEach((item) => collectPrefixUrls(item, prefix, output));
  }
  return output;
}

function rewritePrefix(value, sourcePrefix, targetPrefix) {
  if (typeof value === "string") {
    return value.startsWith(`${sourcePrefix}/`)
      ? `${targetPrefix}/${value.slice(sourcePrefix.length + 1)}`
      : value;
  }
  if (Array.isArray(value)) return value.map((item) => rewritePrefix(item, sourcePrefix, targetPrefix));
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([key, item]) => [key, rewritePrefix(item, sourcePrefix, targetPrefix)]),
    );
  }
  return value;
}

function assertCandidateUrlsResolve(manifest, candidateWorld, candidatePrefix) {
  const urls = [...new Set(collectPrefixUrls(manifest, candidatePrefix))];
  requireCondition(urls.length > 0, "candidate manifest has no candidate-world URLs");
  for (const url of urls) {
    ensureRegularFile(
      localWorldPath(url, candidatePrefix, candidateWorld, "candidate manifest URL"),
      "candidate manifest URL",
    );
  }
  return urls.length;
}

function assertCanonicalUrlsResolve(manifest, canonicalWorld, canonicalPrefix) {
  const urls = [...new Set(collectPrefixUrls(manifest, canonicalPrefix))];
  requireCondition(urls.length > 0, "promoted manifest has no canonical-world URLs");
  for (const url of urls) {
    ensureRegularFile(
      localWorldPath(url, canonicalPrefix, canonicalWorld, "promoted manifest URL"),
      "promoted manifest URL",
    );
  }
  return urls.length;
}

function manifestAssetDescriptors(manifest) {
  const descriptors = [];
  for (const [key, asset] of Object.entries(manifest.assets || {})) {
    descriptors.push({ label: `assets.${key}`, asset });
  }
  for (const object of manifest.interactiveObjects || []) {
    if (object.visual != null) {
      descriptors.push({ label: `interactiveObjects.${object.id}.visual`, asset: object.visual });
    }
    if (object.collision?.asset != null) {
      descriptors.push({
        label: `interactiveObjects.${object.id}.collision.asset`,
        asset: object.collision.asset,
      });
    }
    if (object.collision?.renderAsset != null) {
      descriptors.push({
        label: `interactiveObjects.${object.id}.collision.renderAsset`,
        asset: object.collision.renderAsset,
      });
    }
  }
  requireCondition(descriptors.length > 0, "manifest contains no asset descriptors");
  return descriptors;
}

function verifyAssetDescriptor({ label, asset }, worldDir, urlPrefix) {
  requireCondition(asset && typeof asset === "object" && !Array.isArray(asset), `${label} is invalid`);
  requireCondition(Number.isSafeInteger(asset.size) && asset.size > 0, `${label}.size is invalid`);
  requireCondition(isSha256(asset.sha256), `${label}.sha256 is invalid`);
  const parts = Array.isArray(asset.parts) ? asset.parts : [];
  if (parts.length === 0 && asset.url == null) {
    requireCondition(
      asset.baselineMetadataOnly === true,
      `${label}: asset has no transported URL or parts`,
    );
    return;
  }
  if (parts.length > 0) {
    const whole = crypto.createHash("sha256");
    let total = 0;
    parts.forEach((part, index) => {
      requireCondition(part && typeof part === "object", `${label}.parts[${index}] is invalid`);
      requireCondition(Number.isSafeInteger(part.size) && part.size > 0, `${label}.parts[${index}].size`);
      requireCondition(isSha256(part.sha256), `${label}.parts[${index}].sha256`);
      const partPath = localWorldPath(
        part.url,
        urlPrefix,
        worldDir,
        `${label}.parts[${index}]`,
      );
      ensureRegularFile(partPath, `${label}.parts[${index}]`);
      requireCondition(fs.statSync(partPath).size === part.size, `${label}.parts[${index}] size mismatch`);
      requireCondition(sha256File(partPath) === part.sha256, `${label}.parts[${index}] SHA-256 mismatch`);
      const bytes = fs.readFileSync(partPath);
      whole.update(bytes);
      total += bytes.length;
    });
    requireCondition(total === asset.size, `${label} reconstructed size mismatch`);
    requireCondition(whole.digest("hex") === asset.sha256, `${label} reconstructed SHA-256 mismatch`);
  } else {
    const assetPath = localWorldPath(asset.url, urlPrefix, worldDir, label);
    ensureRegularFile(assetPath, label);
    requireCondition(fs.statSync(assetPath).size === asset.size, `${label} size mismatch`);
    requireCondition(sha256File(assetPath) === asset.sha256, `${label} SHA-256 mismatch`);
  }
  if (asset.url != null && parts.length > 0) {
    const directPath = localWorldPath(asset.url, urlPrefix, worldDir, `${label}.url`);
    ensureRegularFile(directPath, `${label}.url`);
    requireCondition(fs.statSync(directPath).size === asset.size, `${label}.url size mismatch`);
    requireCondition(sha256File(directPath) === asset.sha256, `${label}.url SHA-256 mismatch`);
  }
}

function verifyAllManifestAssets(manifest, worldDir, urlPrefix) {
  const descriptors = manifestAssetDescriptors(manifest);
  descriptors.forEach((descriptor) => verifyAssetDescriptor(descriptor, worldDir, urlPrefix));
  return descriptors.length;
}

function verifyManifestClosure(manifest, worldDir, urlPrefix, label) {
  validateWebManifest(manifest);
  const urls = [...new Set(collectPrefixUrls(manifest, urlPrefix))];
  requireCondition(urls.length > 0, `${label}: manifest has no local asset closure`);
  for (const url of urls) {
    ensureRegularFile(localWorldPath(url, urlPrefix, worldDir, `${label} URL`), `${label} URL`);
  }
  const descriptorCount = verifyAllManifestAssets(manifest, worldDir, urlPrefix);
  return { urlCount: urls.length, descriptorCount };
}

function verifyStableAliasClosure(aliasPath, worldDir, stablePrefix, expectedSha, label) {
  ensureRegularFile(aliasPath, `${label} stable alias`);
  requireCondition(sha256File(aliasPath) === expectedSha, `${label} stable alias SHA-256 mismatch`);
  return verifyManifestClosure(
    readJson(aliasPath, `${label} stable alias`),
    worldDir,
    stablePrefix,
    `${label} stable alias`,
  );
}

function validateCandidateManifest(manifest, manifestSha, candidatePrefix, adoptionReportSha) {
  validateWebManifest(manifest);
  requireCondition(
    manifest.candidateBuild?.status === "candidate_materialized_strict_clean_scene_browser_qa_pending",
    "candidate manifest is not in strict clean-scene browser-QA-pending state",
  );
  requireCondition(manifest.candidateBuild?.promotionAllowed === false, "candidate promotionAllowed must be false");
  requireCondition(
    manifest.candidateBuild?.cleanScene?.status
      === "strict_layered_clean_scene_materialized_current_demo_only",
    "candidate strict clean-scene status is invalid",
  );
  requireCondition(
    manifest.candidateBuild?.cleanScene?.acceptanceScope === "current_demo_only",
    "candidate clean-scene scope is invalid",
  );
  requireCondition(
    manifest.candidateBuild?.cleanScene?.promotionApproved === false,
    "candidate clean-scene promotion must remain false",
  );
  requireCondition(
    sameSet(manifest.interactiveObjects.map((object) => object.id), EXPECTED_OBJECT_IDS),
    "candidate object set is invalid",
  );
  const byId = new Map(manifest.interactiveObjects.map((object) => [object.id, object]));
  for (const objectId of UNIFIED_OBJECT_IDS) {
    const object = byId.get(objectId);
    requireCondition(object.collision?.mode === "unified-glb", `${objectId}: must remain unified-glb`);
    requireCondition(object.visual == null, `${objectId}: visual proxy is forbidden`);
    requireCondition(object.colliderProxy == null, `${objectId}: collider proxy is forbidden`);
    requireCondition(object.collision.renderAsset == null, `${objectId}: render proxy is forbidden`);
  }
  for (const pillowId of UNIFIED_OBJECT_IDS.slice(0, 3)) {
    const pillow = byId.get(pillowId);
    requireCondition(
      pillow.semanticGranularity === "independent_child_asset"
        && pillow.parentObjectId === "sam3_bed_01"
        && pillow.movesWithParent === true
        && pillow.independentlyMovable === true,
      `${pillowId}: must be an independently movable child of sam3_bed_01`,
    );
  }
  const bed = byId.get("sam3_bed_01");
  requireCondition(
    bed.semanticGranularity === "independent_root_asset"
      && bed.parentObjectId == null
      && bed.movesWithParent === false
      && bed.independentlyMovable === true,
    "sam3_bed_01: must remain an independently movable root",
  );
  const reportUrl = manifest.candidateBuild?.report;
  requireCondition(
    reportUrl === `${candidatePrefix}/qa/strict-clean-scene-adoption-report.json`,
    "candidate manifest adoption report URL is invalid",
  );
  requireCondition(isSha256(manifestSha) && isSha256(adoptionReportSha), "candidate hashes are invalid");
}

function validateAdoptionReport(report, candidateManifestSha, canonicalManifestSha) {
  requireCondition(report.schemaVersion === 1, "adoption report schemaVersion is invalid");
  requireCondition(
    report.kind === "video2world.strict_clean_scene_adoption_report",
    "adoption report kind is invalid",
  );
  requireCondition(
    report.status === "candidate_materialized_browser_qa_pending",
    "adoption report status is invalid",
  );
  requireCondition(report.acceptanceScope === "current_demo_only", "adoption report scope is invalid");
  requireCondition(report.promotionApproved === false, "adoption report promotion must remain false");
  requireCondition(report.browserQa === "pending", "adoption report must predate candidate browser QA");
  requireCondition(
    report.outputManifest?.sha256 === candidateManifestSha,
    "adoption report does not bind the candidate manifest SHA-256",
  );
  requireCondition(
    report.publicManifest?.unchanged === true
      && report.publicManifest.sha256Before === canonicalManifestSha
      && report.publicManifest.sha256After === canonicalManifestSha,
    "adoption report does not bind the unchanged canonical manifest",
  );
  requireCondition(
    report.runtimeAssetResolution?.status === "passed"
      && report.runtimeAssetResolution.allManifestWorldUrlsResolve === true
      && report.runtimeAssetResolution.runtimeAssetHashesRevalidated === true,
    "adoption report runtime asset resolution did not pass",
  );
  requireCondition(
    report.gates && Object.values(report.gates).every((value) => value === true),
    "adoption report contains a failed gate",
  );
}

function stableJson(value) {
  if (Array.isArray(value)) return `[${value.map(stableJson).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stableJson(value[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

function resolveQaEvidencePath(value, projectRoot, label) {
  requireCondition(typeof value === "string" && value.length > 0, `${label} path is missing`);
  let resolved;
  const repoPrefix = "repo://video2world/";
  if (value.startsWith(repoPrefix)) {
    const relative = value.slice(repoPrefix.length);
    requireCondition(
      relative.length > 0
        && !/[?#%\\\u0000-\u0020]/u.test(relative)
        && !relative.split("/").some((segment) => ["", ".", ".."].includes(segment)),
      `${label} repo path is unsafe`,
    );
    resolved = path.resolve(projectRoot, ...relative.split("/"));
    requireCondition(isInside(projectRoot, resolved), `${label} escapes project root`);
  } else {
    requireCondition(path.isAbsolute(value), `${label} must be absolute or repo://video2world`);
    resolved = path.resolve(value);
  }
  return ensureRegularFile(resolved, label);
}

function requireFinite(value, label, { positive = false } = {}) {
  requireCondition(typeof value === "number" && Number.isFinite(value), `${label} must be finite`);
  if (positive) requireCondition(value > 0, `${label} must be positive`);
  return value;
}

function validateRuntimeEvidence(runtime, expected, label) {
  requireCondition(runtime && typeof runtime === "object", `${label}.runtime is missing`);
  const state = runtime.initialState;
  const collision = runtime.collisionWorld;
  const inspections = runtime.inspections;
  requireCondition(state?.visualReady === true && state?.colliderReady === true, `${label}: static scene is not ready`);
  requireCondition(state?.sceneQaReady === true, `${label}: scene QA runtime is not ready`);
  requireCondition(
    state.cameraOverviewFocusObjectId === expected.cameraFocusObjectId,
    `${label}: camera overview focus mismatch`,
  );
  requireCondition(
    Array.isArray(state.cameraInsideInteractiveObjectIds)
      && state.cameraInsideInteractiveObjectIds.length === 0,
    `${label}: camera is inside an interactive object`,
  );
  const overviewCoverage = state.cameraOverviewCoverage;
  requireCondition(
    overviewCoverage && typeof overviewCoverage === "object" && !Array.isArray(overviewCoverage),
    `${label}: camera overview coverage is missing`,
  );
  const widthFraction = requireFinite(
    overviewCoverage.widthFraction,
    `${label}.cameraOverviewCoverage.widthFraction`,
  );
  const heightFraction = requireFinite(
    overviewCoverage.heightFraction,
    `${label}.cameraOverviewCoverage.heightFraction`,
  );
  requireCondition(
    widthFraction >= CAMERA_OVERVIEW_COVERAGE_MIN
      && widthFraction <= CAMERA_OVERVIEW_COVERAGE_MAX,
    `${label}: camera overview width coverage out of range`,
  );
  requireCondition(
    heightFraction >= CAMERA_OVERVIEW_COVERAGE_MIN
      && heightFraction <= CAMERA_OVERVIEW_COVERAGE_MAX,
    `${label}: camera overview height coverage out of range`,
  );
  requireCondition(state.staticVisualCount === expected.staticVisualCount, `${label}: static visual count mismatch`);
  requireCondition(state.colliderFaces === expected.staticColliderFaces, `${label}: static collider faces mismatch`);
  requireCondition(state.interactiveObjectCount === expected.objectCount, `${label}: object count mismatch`);
  requireCondition(state.interactiveObjectReadyCount === expected.objectCount, `${label}: ready object count mismatch`);
  requireCondition(state.objectColliderCount === expected.objectColliderCount, `${label}: collider count mismatch`);
  requireCondition(state.interactiveObjectMeshVertexCount === expected.meshVertices, `${label}: mesh vertex mismatch`);
  requireCondition(state.interactiveObjectGaussianCount === expected.gaussianVertices, `${label}: Gaussian count mismatch`);
  requireCondition(state.interactiveObjectRgbPointCount === expected.rgbPointVertices, `${label}: RGB point count mismatch`);
  requireCondition(state.visualCount === expected.totalVisualPrimitives, `${label}: total primitive mismatch`);
  requireCondition(Array.isArray(state.interactiveObjects) && state.interactiveObjects.length === expected.objectCount,
    `${label}: runtime object snapshots are incomplete`);
  for (const objectId of expected.objectIds) {
    const snapshot = state.interactiveObjects.find((item) => item.id === objectId);
    requireCondition(
      snapshot?.ready === true && snapshot.visualReady === true && snapshot.colliderReady === true,
      `${label}: ${objectId} runtime object is not ready`,
    );
    requireCondition(snapshot.collisionMode !== "degraded-box", `${label}: ${objectId} uses degraded collision`);
  }
  requireCondition(collision?.sceneReady === true, `${label}: collision scene is not ready`);
  requireCondition(collision.objectColliderReadyCount === expected.objectColliderCount,
    `${label}: ready collision count mismatch`);
  requireCondition(collision.objectColliderFaces === expected.objectColliderFaces,
    `${label}: collision face count mismatch`);
  requireCondition(collision.degradedCount === 0, `${label}: degraded colliders were loaded`);
  requireCondition(Array.isArray(collision.errors) && collision.errors.length === 0,
    `${label}: collision errors are not empty`);
  requireCondition(Array.isArray(collision.objects) && collision.objects.length === expected.objectCount,
    `${label}: collision object set is incomplete`);
  requireCondition(collision.objects.every((item) => item.ready === true), `${label}: a collider is not ready`);
  requireCondition(inspections && typeof inspections === "object", `${label}: inspections are missing`);
  for (const objectId of UNIFIED_OBJECT_IDS) {
    const inspection = inspections[objectId];
    const contract = expected.unified[objectId];
    const collider = collision.objects.find((item) => item.id === objectId);
    requireCondition(
      inspection?.visualReady === true
        && inspection.colliderReady === true
        && inspection.visualKind === "mesh"
        && inspection.collisionMode === "unified-glb"
        && inspection.unifiedVisualCollision === true,
      `${label}: ${objectId} unified runtime contract failed`,
    );
    requireCondition(inspection.collisionTopology === contract.topology, `${label}: ${objectId} topology mismatch`);
    requireCondition(inspection.collisionFaces === contract.faces, `${label}: ${objectId} runtime faces mismatch`);
    requireCondition(inspection.meshVertexCount === contract.vertices, `${label}: ${objectId} runtime vertices mismatch`);
    requireCondition(inspection.pbrMaterialCount > 0 && inspection.bvhMeshCount > 0,
      `${label}: ${objectId} lacks PBR/BVH evidence`);
    requireCondition(inspection.proxyWorldBounds == null && inspection.proxyMatrixWorld == null,
      `${label}: ${objectId} loaded a proxy`);
    requireCondition(collider?.faces === contract.faces && collider?.topology === contract.topology,
      `${label}: ${objectId} collision-world contract mismatch`);
  }
}

function validateQueryEvidence(queries, expectedCases, label) {
  requireCondition(Array.isArray(queries) && queries.length === expectedCases.length,
    `${label}: query evidence count mismatch`);
  for (const expected of expectedCases) {
    const query = queries.find((item) => item.label === expected.label);
    requireCondition(query?.expectedId === expected.expectedId, `${label}: ${expected.label} expected ID mismatch`);
    requireCondition(
      query.answer?.status === "resolved"
        && query.answer.entityId === expected.expectedId
        && query.answer.focusEntityId === expected.expectedId
        && typeof query.answer.answer === "string"
        && query.answer.answer.trim().length > 0,
      `${label}: ${expected.label} answer/focus evidence failed`,
    );
    const focusEvidence = query.focusAndBbox?.cameraTargetBoundsEvidence;
    requireCondition(
      query.focusAndBbox?.selectedSceneEntity === expected.expectedId
        && query.focusAndBbox.selectedInteractiveObject === expected.expectedId
        && query.focusAndBbox.selectionOutlineVisible === true
        && query.focusAndBbox.collisionBounds != null
        && requireFinite(query.focusAndBbox.cameraMaximumDelta, `${label}.${expected.label}.cameraDelta`) >= 0
        && focusEvidence?.passed === true
        && focusEvidence.targetInsideBounds === true
        && requireFinite(
          focusEvidence.centerDistance,
          `${label}.${expected.label}.cameraTargetCenterDistance`,
        ) >= 0
        && requireFinite(
          focusEvidence.normalizedCenterDistance,
          `${label}.${expected.label}.cameraTargetNormalizedCenterDistance`,
        ) >= 0
        && focusEvidence.normalizedCenterDistance <= CAMERA_TARGET_CENTER_NORMALIZED_MAX,
      `${label}: ${expected.label} bbox/camera focus evidence failed`,
    );
  }
}

function allTrueRecord(value, label) {
  requireCondition(value && typeof value === "object" && !Array.isArray(value), `${label} is missing`);
  requireCondition(Object.keys(value).length > 0 && Object.values(value).every((item) => item === true),
    `${label} contains a failed check`);
}

function validateImplementationEvidence(value, projectRoot) {
  requireCondition(value && typeof value === "object" && !Array.isArray(value),
    "browser QA implementation evidence is missing");
  requireCondition(sameSet(Object.keys(value), Object.keys(IMPLEMENTATION_EVIDENCE)),
    "browser QA implementation evidence set is invalid");
  for (const [key, relative] of Object.entries(IMPLEMENTATION_EVIDENCE)) {
    const descriptor = value[key];
    const expectedPath = `repo://video2world/${relative}`;
    requireCondition(descriptor?.path === expectedPath, `${key}: implementation path is invalid`);
    requireCondition(Number.isSafeInteger(descriptor.size) && descriptor.size > 0,
      `${key}: implementation size is invalid`);
    requireCondition(isSha256(descriptor.sha256), `${key}: implementation SHA-256 is invalid`);
    const filePath = resolveQaEvidencePath(descriptor.path, projectRoot, `${key} implementation`);
    requireCondition(filePath === path.join(projectRoot, ...relative.split("/")),
      `${key}: implementation path does not resolve to the expected file`);
    requireCondition(fs.statSync(filePath).size === descriptor.size,
      `${key}: implementation size mismatch`);
    requireCondition(sha256File(filePath) === descriptor.sha256,
      `${key}: implementation SHA-256 mismatch`);
  }
}

let crc32Table = null;

function pngCrc32(bytes) {
  if (crc32Table == null) {
    crc32Table = Array.from({ length: 256 }, (_, index) => {
      let value = index;
      for (let bit = 0; bit < 8; bit += 1) {
        value = (value & 1) === 1 ? (0xedb88320 ^ (value >>> 1)) : (value >>> 1);
      }
      return value >>> 0;
    });
  }
  let crc = 0xffffffff;
  for (const byte of bytes) crc = crc32Table[(crc ^ byte) & 0xff] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

function parsePng(bytes, label) {
  const signature = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  requireCondition(bytes.length >= 57 && bytes.subarray(0, 8).equals(signature),
    `${label}: invalid PNG signature`);
  let offset = 8;
  let chunkIndex = 0;
  let width = null;
  let height = null;
  let sawIdat = false;
  let sawIend = false;
  while (offset < bytes.length) {
    requireCondition(offset + 12 <= bytes.length, `${label}: truncated PNG chunk header`);
    const length = bytes.readUInt32BE(offset);
    const chunkEnd = offset + 12 + length;
    requireCondition(chunkEnd <= bytes.length, `${label}: truncated PNG chunk payload`);
    const typeBytes = bytes.subarray(offset + 4, offset + 8);
    const type = typeBytes.toString("ascii");
    requireCondition(/^[A-Za-z]{4}$/u.test(type), `${label}: invalid PNG chunk type`);
    const storedCrc = bytes.readUInt32BE(offset + 8 + length);
    const calculatedCrc = pngCrc32(bytes.subarray(offset + 4, offset + 8 + length));
    requireCondition(storedCrc === calculatedCrc, `${label}: PNG ${type} CRC mismatch`);
    if (chunkIndex === 0) {
      requireCondition(type === "IHDR" && length === 13, `${label}: PNG must start with IHDR`);
      width = bytes.readUInt32BE(offset + 8);
      height = bytes.readUInt32BE(offset + 12);
      requireCondition(width > 0 && height > 0, `${label}: PNG dimensions are invalid`);
      requireCondition(bytes[offset + 18] === 0 && bytes[offset + 19] === 0,
        `${label}: unsupported PNG compression/filter method`);
      requireCondition([0, 1].includes(bytes[offset + 20]), `${label}: invalid PNG interlace method`);
    } else {
      requireCondition(type !== "IHDR", `${label}: PNG contains multiple IHDR chunks`);
    }
    if (type === "IDAT") sawIdat = true;
    if (type === "IEND") {
      requireCondition(length === 0, `${label}: PNG IEND must be empty`);
      sawIend = true;
      offset = chunkEnd;
      requireCondition(offset === bytes.length, `${label}: PNG has trailing bytes after IEND`);
      break;
    }
    offset = chunkEnd;
    chunkIndex += 1;
  }
  requireCondition(sawIdat, `${label}: PNG has no IDAT chunk`);
  requireCondition(sawIend, `${label}: PNG has no IEND chunk`);
  return { width, height };
}

function validateScreenshot(value, viewport, candidateQaDir, projectRoot, label) {
  requireCondition(value && typeof value === "object" && !Array.isArray(value),
    `${label}: screenshot descriptor is missing`);
  requireCondition(Number.isSafeInteger(value.size) && value.size > 0,
    `${label}: screenshot size is invalid`);
  requireCondition(isSha256(value.sha256), `${label}: screenshot SHA-256 is invalid`);
  requireCondition(Number.isSafeInteger(value.width) && value.width > 0,
    `${label}: screenshot descriptor width is invalid`);
  requireCondition(Number.isSafeInteger(value.height) && value.height > 0,
    `${label}: screenshot descriptor height is invalid`);
  const screenshot = resolveQaEvidencePath(value.path, projectRoot, `${label}.screenshot`);
  requireCondition(isInside(candidateQaDir, screenshot), `${label}: screenshot must be inside candidate-world/qa`);
  const bytes = fs.readFileSync(screenshot);
  requireCondition(bytes.length === value.size, `${label}: screenshot size mismatch`);
  requireCondition(sha256File(screenshot) === value.sha256, `${label}: screenshot SHA-256 mismatch`);
  const dimensions = parsePng(bytes, `${label}.screenshot`);
  requireCondition(dimensions.width === value.width && dimensions.height === value.height,
    `${label}: screenshot descriptor dimensions mismatch`);
  requireCondition(dimensions.width === viewport[0], `${label}: screenshot width mismatch`);
  requireCondition(dimensions.height >= viewport[1], `${label}: screenshot height is smaller than viewport`);
  return screenshot;
}

function validateMotionMetric(metric, label) {
  requireCondition(metric && typeof metric === "object", `${label} is missing`);
  for (const key of ["matrixDelta", "translationDelta", "gramDelta"]) {
    requireCondition(Number.isFinite(metric[key]) && metric[key] >= 0, `${label}.${key} is invalid`);
  }
}

function validateTransformMotion(value, label) {
  requireCondition(value && typeof value === "object", `${label} is missing`);
  for (const key of ["groupMatrixWorld", "splatMatrixWorld", "collisionMatrixWorld"]) {
    validateMotionMetric(value[key], `${label}.${key}`);
  }
}

function validateHierarchyEvidence(value, label) {
  requireCondition(value && typeof value === "object" && !Array.isArray(value),
    `${label}: hierarchy evidence is missing`);
  for (const [field, ids] of [
    ["attachments", PILLOW_OBJECT_IDS],
    ["initialMotionRootGramDelta", PILLOW_OBJECT_IDS],
    ["parentMotion", UNIFIED_OBJECT_IDS],
    ["parentReturned", UNIFIED_OBJECT_IDS],
  ]) {
    requireCondition(value[field] && typeof value[field] === "object" && !Array.isArray(value[field]),
      `${label}: hierarchy.${field} is missing`);
    requireCondition(sameSet(Object.keys(value[field]), ids),
      `${label}: hierarchy.${field} object set is invalid`);
  }
  for (const pillowId of PILLOW_OBJECT_IDS) {
    requireCondition(Number.isFinite(value.attachments[pillowId])
      && value.attachments[pillowId] <= MATRIX_EPSILON,
    `${label}: ${pillowId} hierarchy attachment drifted`);
    requireCondition(Number.isFinite(value.initialMotionRootGramDelta[pillowId])
      && value.initialMotionRootGramDelta[pillowId] <= MATRIX_EPSILON,
    `${label}: ${pillowId} initial motion root is not rigid`);
  }
  for (const objectId of UNIFIED_OBJECT_IDS) {
    const motion = value.parentMotion[objectId];
    validateTransformMotion(motion, `${label}.hierarchy.parentMotion.${objectId}`);
    for (const key of ["groupMatrixWorld", "splatMatrixWorld", "collisionMatrixWorld"]) {
      requireCondition(motion[key].matrixDelta > 1e-4,
        `${label}: parent motion did not move ${objectId}.${key}`);
      requireCondition(motion[key].gramDelta <= MATRIX_GRAM_EPSILON,
        `${label}: parent motion changed ${objectId}.${key} shape`);
    }
    allTrueRecord(value.parentReturned[objectId], `${label}.hierarchy.parentReturned.${objectId}`);
  }
  requireCondition(value.parentMotion[BED_OBJECT_ID].groupMatrixWorld.translationDelta <= MATRIX_EPSILON,
    `${label}: bed pivot drifted during parent motion`);
  for (const pillowId of PILLOW_OBJECT_IDS) {
    requireCondition(value.parentMotion[pillowId].groupMatrixWorld.translationDelta > 1e-4,
      `${label}: ${pillowId} did not move with bed`);
    requireCondition(Number.isFinite(value.parentMotion[pillowId].relativeParentMatrixDelta)
      && value.parentMotion[pillowId].relativeParentMatrixDelta <= MATRIX_EPSILON,
    `${label}: ${pillowId} relative parent transform drifted`);
  }
  requireCondition(Array.isArray(value.independentChildren)
    && value.independentChildren.length === PILLOW_OBJECT_IDS.length,
  `${label}: independent child evidence count mismatch`);
  requireCondition(sameSet(value.independentChildren.map((item) => item?.objectId), PILLOW_OBJECT_IDS),
    `${label}: independent child object set is invalid`);
  for (const pillowId of PILLOW_OBJECT_IDS) {
    const evidence = value.independentChildren.find((item) => item.objectId === pillowId);
    validateTransformMotion(evidence.pillowMotion, `${label}.hierarchy.independent.${pillowId}.pillowMotion`);
    validateTransformMotion(evidence.bedMotion, `${label}.hierarchy.independent.${pillowId}.bedMotion`);
    for (const key of ["groupMatrixWorld", "splatMatrixWorld", "collisionMatrixWorld"]) {
      requireCondition(evidence.pillowMotion[key].matrixDelta > 1e-4,
        `${label}: ${pillowId} independent motion did not move ${key}`);
      requireCondition(evidence.pillowMotion[key].gramDelta <= MATRIX_GRAM_EPSILON,
        `${label}: ${pillowId} independent motion changed ${key} shape`);
      requireCondition(evidence.bedMotion[key].matrixDelta <= MATRIX_EPSILON,
        `${label}: ${pillowId} independent motion moved the bed ${key}`);
    }
    requireCondition(evidence.pillowMotion.groupMatrixWorld.translationDelta <= MATRIX_EPSILON,
      `${label}: ${pillowId} pivot drifted during independent motion`);
    allTrueRecord(evidence.returned, `${label}.hierarchy.independent.${pillowId}.returned`);
  }
}

function validateDiagnosticsEvidence(diagnostics, expected, manifestSha, runtimeUrl, label) {
  requireCondition(diagnostics?.label === label, `${label}: diagnostics label mismatch`);
  requireCondition(diagnostics.manifestRequestCount === 1, `${label}: manifest request count mismatch`);
  requireCondition(
    Array.isArray(diagnostics.manifestResponseSha256)
      && diagnostics.manifestResponseSha256.length === 1
      && diagnostics.manifestResponseSha256[0] === manifestSha,
    `${label}: served manifest SHA evidence mismatch`,
  );
  requireCondition(Array.isArray(diagnostics.pageErrors) && diagnostics.pageErrors.length === 0,
    `${label}: page errors are not empty`);
  requireCondition(Array.isArray(diagnostics.requestFailures) && diagnostics.requestFailures.length === 0,
    `${label}: request failures are not empty`);
  requireCondition(Array.isArray(diagnostics.httpErrors) && diagnostics.httpErrors.length === 0,
    `${label}: HTTP errors are not empty`);
  requireCondition(
    Array.isArray(diagnostics.consoleMessages)
      && diagnostics.consoleMessages.every((item) => item?.type !== "error"),
    `${label}: console errors are not empty`,
  );
  const requests = diagnostics.assetRequestCounts;
  requireCondition(requests && typeof requests === "object", `${label}: asset request counts are missing`);
  for (const contract of Object.values(expected.unified)) {
    const pathname = new URL(contract.url, runtimeUrl).pathname;
    requireCondition(requests[pathname] === 1, `${label}: unified asset request count mismatch for ${pathname}`);
  }
}

function validateViewportEvidence({
  value,
  label,
  viewport,
  full,
  expected,
  manifestSha,
  runtimeUrl,
  candidateQaDir,
  projectRoot,
}) {
  requireCondition(value?.label === label, `${label}: viewport label mismatch`);
  requireCondition(JSON.stringify(value.viewport) === JSON.stringify(viewport), `${label}: viewport mismatch`);
  requireCondition(value.freshPageLoad === true, `${label}: fresh page load is not proven`);
  validateRuntimeEvidence(value.runtime, expected, label);
  const expectedQueries = full
    ? expected.queryCases
    : expected.queryCases.filter((item) => ["pillow_appearance", "bed_appearance"].includes(item.label));
  validateQueryEvidence(value.queries, expectedQueries, label);
  if (full) {
    requireCondition(Array.isArray(value.interactions) && value.interactions.length === UNIFIED_OBJECT_IDS.length,
      `${label}: interaction evidence count mismatch`);
    requireCondition(Array.isArray(value.robotCollisions) && value.robotCollisions.length === UNIFIED_OBJECT_IDS.length,
      `${label}: robot collision evidence count mismatch`);
    for (const objectId of UNIFIED_OBJECT_IDS) {
      const interaction = value.interactions.find((item) => item.objectId === objectId);
      requireCondition(
        interaction?.focus?.focused === true
          && interaction.focus.selectedSceneEntity === objectId
          && interaction.focus.selectedInteractiveObject === objectId
          && interaction.focus.selectionOutlineVisible === true,
        `${label}: ${objectId} focus interaction failed`,
      );
      requireCondition(interaction.pointer?.hit?.objectId === objectId
        && interaction.pointer.hit.colliderMode === "unified-glb",
      `${label}: ${objectId} pointer hit failed`);
      requireCondition(Object.values(interaction.drag?.deltas || {}).every((item) => Number.isFinite(item) && item > 1e-4),
        `${label}: ${objectId} drag did not move visual and collision`);
      allTrueRecord(interaction.drag?.returned, `${label}.${objectId}.drag.returned`);
      requireCondition(
        interaction.doubleClick360?.turnsAfter > interaction.doubleClick360?.turnsBefore
          && Object.values(interaction.doubleClick360.inFlightDeltas || {})
            .every((item) => Number.isFinite(item) && item > 1e-4),
        `${label}: ${objectId} 360 spin evidence failed`,
      );
      allTrueRecord(interaction.doubleClick360?.returned, `${label}.${objectId}.spin.returned`);
      const collision = value.robotCollisions.find((item) => item.objectId === objectId);
      requireCondition(
        collision?.passed === true
          && collision.prepared?.prepared === true
          && collision.step?.blocked === true
          && collision.step.lastCollisionObjectId === objectId,
        `${label}: ${objectId} robot collision evidence failed`,
      );
    }
    requireCondition(
      value.interpenetrations?.automatedStatus === "passed_with_recorded_contacts"
        && Array.isArray(value.interpenetrations.classifications)
        && value.interpenetrations.classifications.every((item) => item.blocking === false),
      `${label}: interpenetration evidence contains a blocking contact`,
    );
    validateHierarchyEvidence(value.hierarchy, label);
  } else {
    requireCondition(Array.isArray(value.interactions) && value.interactions.length === 0,
      `${label}: unexpected interaction evidence`);
    requireCondition(Array.isArray(value.robotCollisions) && value.robotCollisions.length === 0,
      `${label}: unexpected robot evidence`);
    requireCondition(value.interpenetrations == null, `${label}: unexpected interpenetration evidence`);
    requireCondition(value.hierarchy == null, `${label}: mobile hierarchy evidence must be null`);
  }
  requireCondition(
    value.canvas?.nonblank === true
      && value.canvas.nonDarkPixels >= 64
      && value.canvas.quantizedColorCount >= 8
      && value.canvas.lumaRange >= 8,
    `${label}: canvas pixel evidence failed`,
  );
  allTrueRecord(value.layout?.checks, `${label}.layout.checks`);
  requireCondition(JSON.stringify(value.layout?.viewport) === JSON.stringify(viewport), `${label}: layout viewport mismatch`);
  const performance = value.performance;
  requireCondition(
    performance?.passed === true
      && Array.isArray(performance.samples)
      && performance.samples.length === 5
      && performance.samples.every(Number.isFinite)
      && requireFinite(performance.threshold, `${label}.fps.threshold`, { positive: true }) >= 18
      && requireFinite(performance.average, `${label}.fps.average`, { positive: true }) >= performance.threshold,
    `${label}: FPS evidence failed`,
  );
  validateDiagnosticsEvidence(value.diagnostics, expected, manifestSha, runtimeUrl, label);
  validateScreenshot(value.screenshot, viewport, candidateQaDir, projectRoot, label);
}

export function validateBrowserQa({
  report,
  candidateManifest,
  candidateManifestPath,
  candidateManifestSha,
  candidateWorld,
  projectRoot,
}) {
  requireCondition(report.schemaVersion === 2, "browser QA schemaVersion is invalid");
  requireCondition(
    report.kind === "video2world.bedroom4_clean_unified_scene_browser_qa",
    "browser QA kind is invalid",
  );
  requireCondition(report.status === "passed_current_demo_only", "candidate browser QA did not pass");
  requireCondition(report.automatedGate === "passed", "candidate browser QA automated gate did not pass");
  requireCondition(report.acceptanceScope === "current_demo_only", "candidate browser QA scope is invalid");
  requireCondition(report.promotionApproved === false, "candidate QA must not claim promotion");
  requireCondition(report.publishingPerformed === false, "candidate QA must not publish");
  requireCondition(report.manifestEdited === false, "candidate QA must not edit its manifest");
  const startedMs = Date.parse(report.startedAt);
  const finishedMs = Date.parse(report.finishedAt);
  requireCondition(Number.isFinite(startedMs) && Number.isFinite(finishedMs) && finishedMs >= startedMs,
    "candidate QA timestamps are invalid");
  requireCondition(finishedMs <= Date.now() + 5 * 60 * 1000,
    "candidate QA finishedAt is implausibly in the future");
  validateImplementationEvidence(report.implementation, projectRoot);
  requireCondition(
    report.manifest?.sha256Before === candidateManifestSha
      && report.manifest.sha256After === candidateManifestSha
      && report.manifest.unchanged === true,
    "candidate browser QA does not bind the exact candidate manifest before/after SHA-256",
  );
  requireCondition(Array.isArray(report.failures) && report.failures.length === 0, "candidate QA failures are not empty");
  requireCondition(
    report.blockingFailures == null
      || (Array.isArray(report.blockingFailures) && report.blockingFailures.length === 0),
    "candidate QA blocking failures are not empty",
  );
  const boundManifestPath = resolveQaEvidencePath(report.manifest.path, projectRoot, "browser QA manifest");
  requireCondition(boundManifestPath === candidateManifestPath, "candidate QA report binds another manifest path");
  const expected = deriveManifestExpectations(candidateManifest);
  requireCondition(stableJson(report.expected) === stableJson(expected), "candidate QA expected contract differs from manifest");
  requireCondition(
    report.qualityPolicy?.minorLimitationsMayPass === true
      && report.qualityPolicy.supportAndInterpenetrationAlwaysRecorded === true
      && report.qualityPolicy.obviousInterpenetrationBlocks === true
      && report.qualityPolicy.humanScreenshotReviewRequired === true,
    "candidate QA quality policy is incomplete",
  );
  let runtimeUrl;
  try {
    runtimeUrl = new URL(report.url);
  } catch {
    throw new Error("candidate QA runtime URL is invalid");
  }
  requireCondition(["http:", "https:"].includes(runtimeUrl.protocol), "candidate QA runtime URL protocol is invalid");
  requireCondition(!runtimeUrl.username && !runtimeUrl.password && !runtimeUrl.hash,
    "candidate QA runtime URL contains credentials or a fragment");
  requireCondition(runtimeUrl.searchParams.get("collisionDebug") === "on",
    "candidate QA did not enable collisionDebug");
  const manifestParameter = runtimeUrl.searchParams.get("manifest");
  requireCondition(typeof manifestParameter === "string" && !/[#%\\]/u.test(manifestParameter),
    "candidate QA runtime manifest parameter is unsafe");
  const servedManifestPath = new URL(manifestParameter, runtimeUrl).pathname;
  const publicRoot = path.join(projectRoot, "web", "public");
  requireCondition(isInside(publicRoot, candidateManifestPath),
    "candidate manifest must be inside web/public");
  const expectedServedManifestPath = `/${path.relative(publicRoot, candidateManifestPath)
    .split(path.sep).join("/")}`;
  requireCondition(
    servedManifestPath === expectedServedManifestPath,
    "candidate QA runtime URL served another manifest",
  );
  const candidateQaDir = realDirectory(path.join(candidateWorld, "qa"), "candidate qa directory");
  validateViewportEvidence({
    value: report.desktop,
    label: "desktop",
    viewport: [1440, 900],
    full: true,
    expected,
    manifestSha: candidateManifestSha,
    runtimeUrl,
    candidateQaDir,
    projectRoot,
  });
  validateViewportEvidence({
    value: report.mobile,
    label: "mobile",
    viewport: [390, 844],
    full: false,
    expected,
    manifestSha: candidateManifestSha,
    runtimeUrl,
    candidateQaDir,
    projectRoot,
  });
  return { expected, screenshotCount: 2 };
}

function directoryInventory(directory) {
  if (!fs.existsSync(directory)) return { exists: false, inode: null, fileCount: 0, totalBytes: 0 };
  requireCondition(fs.statSync(directory).isDirectory(), `${directory} must be a directory`);
  let fileCount = 0;
  let totalBytes = 0;
  const visit = (current) => {
    for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
      const entryPath = path.join(current, entry.name);
      requireCondition(!entry.isSymbolicLink(), `world inventory forbids symlinks: ${entryPath}`);
      if (entry.isDirectory()) visit(entryPath);
      else if (entry.isFile()) {
        fileCount += 1;
        totalBytes += fs.statSync(entryPath).size;
      } else throw new Error(`unsupported world entry: ${entryPath}`);
    }
  };
  visit(directory);
  return { exists: true, inode: fs.statSync(directory).ino, fileCount, totalBytes };
}

function canonicalLockPath(canonicalWorld) {
  return path.join(
    path.dirname(canonicalWorld),
    `.${path.basename(canonicalWorld)}.strict-clean-promotion.lock.json`,
  );
}

function processIsAlive(pid) {
  if (!Number.isSafeInteger(pid) || pid <= 0) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    if (error?.code === "ESRCH") return false;
    return true;
  }
}

function readLockOwner(lockPath) {
  let stat;
  try {
    stat = fs.lstatSync(lockPath);
  } catch (error) {
    if (error?.code === "ENOENT") return null;
    throw error;
  }
  requireCondition(!stat.isSymbolicLink(), `promotion lock path is a symlink: ${lockPath}`);
  requireCondition(stat.isFile(), `promotion lock is not a regular file: ${lockPath}`);
  const owner = readJson(lockPath, "promotion lock");
  const startedMs = Date.parse(owner.startedAt);
  const sameHost = owner.hostname === os.hostname();
  const deadLocalProcess = sameHost && !processIsAlive(owner.pid);
  const expired = Number.isFinite(startedMs) && Date.now() - startedMs > LOCK_STALE_AFTER_MS;
  return {
    owner,
    stale: deadLocalProcess || expired,
    reason: deadLocalProcess ? "owner_pid_not_running" : (expired ? "lock_age_exceeded" : "owner_active"),
  };
}

function writeLockExclusive(lockPath, owner) {
  const bytes = Buffer.from(`${JSON.stringify(owner, null, 2)}\n`);
  let descriptor = null;
  try {
    descriptor = fs.openSync(lockPath, "wx", 0o600);
    fs.writeFileSync(descriptor, bytes);
    fs.fsyncSync(descriptor);
    fs.closeSync(descriptor);
    descriptor = null;
    fsyncDirectory(path.dirname(lockPath));
  } finally {
    if (descriptor != null) fs.closeSync(descriptor);
  }
}

function acquireCanonicalLock(canonicalInput, {
  backupWorld = null,
  journalPath = null,
  allowStaleRecovery = false,
} = {}) {
  const canonicalAbsolute = path.resolve(canonicalInput);
  const worldsRoot = realDirectory(path.dirname(canonicalAbsolute), "canonical lock parent");
  const canonicalWorld = path.join(worldsRoot, path.basename(canonicalAbsolute));
  requireCondition(canonicalWorld === canonicalAbsolute,
    "canonical-source-world lock target does not resolve through its literal parent");
  assertNoSymlinkAncestors(canonicalWorld, "canonical-source-world lock target", {
    allowMissingLeaf: true,
  });
  if (fs.existsSync(canonicalWorld)) {
    requireCondition(realDirectory(canonicalWorld, "canonical-source-world lock target") === canonicalWorld,
      "canonical-source-world lock target is not literal");
  }
  const lockPath = canonicalLockPath(canonicalWorld);
  assertNoSymlinkAncestors(lockPath, "promotion lock", { allowMissingLeaf: true });
  const owner = {
    schemaVersion: 1,
    kind: "video2world.bedroom4_strict_clean_scene_promotion_lock",
    token: crypto.randomUUID(),
    pid: process.pid,
    hostname: os.hostname(),
    startedAt: new Date().toISOString(),
    canonicalWorld,
    backupWorld: backupWorld == null ? null : path.resolve(backupWorld),
    journalPath: journalPath == null ? null : path.resolve(journalPath),
  };
  for (;;) {
    try {
      writeLockExclusive(lockPath, owner);
      return { lockPath, owner };
    } catch (error) {
      if (error?.code !== "EEXIST") throw error;
      const existing = readLockOwner(lockPath);
      requireCondition(existing != null, `promotion lock disappeared during acquisition: ${lockPath}`);
      const sameJournal = journalPath != null
        && existing.owner.journalPath === path.resolve(journalPath);
      if (allowStaleRecovery && existing.stale && sameJournal) {
        const archive = `${lockPath}.stale-${Date.now()}-${existing.owner.token || "unknown"}`;
        fs.renameSync(lockPath, archive);
        fsyncDirectory(path.dirname(lockPath));
        continue;
      }
      throw new Error(
        `promotion lock already held: path=${lockPath} ownerPid=${existing.owner.pid ?? "unknown"}`
          + ` ownerHost=${existing.owner.hostname ?? "unknown"}`
          + ` startedAt=${existing.owner.startedAt ?? "unknown"}`
          + ` journal=${existing.owner.journalPath ?? "none"}`
          + ` stale=${existing.stale} reason=${existing.reason}`,
      );
    }
  }
}

function releaseCanonicalLock(lockHandle) {
  if (lockHandle == null) return;
  const existing = readLockOwner(lockHandle.lockPath);
  requireCondition(existing != null, `promotion lock disappeared before release: ${lockHandle.lockPath}`);
  requireCondition(existing.owner.token === lockHandle.owner.token,
    `promotion lock owner changed before release: ${lockHandle.lockPath}`);
  fs.unlinkSync(lockHandle.lockPath);
  fsyncDirectory(path.dirname(lockHandle.lockPath));
}

function validateBoundaries({
  candidateWorld: candidateInput,
  candidateQaPath: candidateQaInput,
  canonicalWorld: canonicalInput,
  backupWorld: backupInput,
  publicManifest: publicManifestInput,
}) {
  const candidateWorld = realDirectory(candidateInput, "candidate-world");
  const canonicalWorld = realDirectory(canonicalInput, "canonical-source-world");
  requireCondition(candidateWorld !== canonicalWorld, "candidate and canonical worlds must differ");
  const worldsRoot = path.dirname(canonicalWorld);
  requireCondition(
    path.basename(worldsRoot) === "worlds"
      && path.basename(path.dirname(worldsRoot)) === "public"
      && path.basename(path.dirname(path.dirname(worldsRoot))) === "web",
    "canonical-source-world must be directly inside web/public/worlds",
  );
  requireCondition(path.dirname(candidateWorld) === worldsRoot, "candidate-world must be a canonical worlds sibling");
  requireCondition(
    [path.basename(candidateWorld), path.basename(canonicalWorld)].every((name) =>
      /^[A-Za-z0-9][A-Za-z0-9._-]*$/u.test(name)),
    "candidate/canonical world names are unsafe",
  );
  const backupAbsolute = path.resolve(backupInput);
  const backupParent = realDirectory(path.dirname(backupAbsolute), "backup-world parent");
  const backupWorld = path.join(backupParent, path.basename(backupAbsolute));
  requireCondition(/^[A-Za-z0-9][A-Za-z0-9._-]*$/u.test(path.basename(backupWorld)),
    "backup-world name is unsafe");
  requireCondition(backupWorld === backupAbsolute, "backup-world does not resolve through its literal parent");
  assertNoSymlinkAncestors(backupWorld, "backup-world", { allowMissingLeaf: true });
  requireCondition(!isInside(worldsRoot, backupWorld), "backup-world must be outside web/public/worlds");
  requireCondition(
    !isInside(candidateWorld, backupWorld)
      && !isInside(backupWorld, candidateWorld)
      && !isInside(canonicalWorld, backupWorld)
      && !isInside(backupWorld, canonicalWorld),
    "backup-world overlaps a live world path",
  );
  requireCondition(!fs.existsSync(backupWorld), "backup-world must not exist; nonempty backup is rejected");
  const publicManifest = ensureRegularFile(publicManifestInput, "public manifest");
  requireCondition(
    publicManifest === path.join(canonicalWorld, "manifest.json"),
    "public-manifest must be canonical-source-world/manifest.json",
  );
  const candidateQaPath = ensureRegularFile(candidateQaInput, "candidate browser QA report");
  const candidateQaDir = realDirectory(path.join(candidateWorld, "qa"), "candidate qa directory");
  requireCondition(
    candidateQaPath !== candidateQaDir && isInside(candidateQaDir, candidateQaPath),
    "candidate browser QA report must be inside candidate-world/qa",
  );
  const device = fs.statSync(canonicalWorld).dev;
  requireCondition(fs.statSync(candidateWorld).dev === device, "candidate and canonical worlds are on different devices");
  requireCondition(fs.statSync(backupParent).dev === device, "backup-world is on another device; rename would not be atomic");
  const projectRoot = realDirectory(path.resolve(worldsRoot, "../../.."), "project root");
  return {
    candidateWorld,
    candidateQaPath,
    canonicalWorld,
    backupWorld,
    publicManifest,
    backupParent,
    worldsRoot,
    projectRoot,
  };
}

function journalPaths(backupWorld) {
  return {
    journalPath: `${backupWorld}.swap-journal.json`,
    candidateManifestSnapshotPath: `${backupWorld}.candidate-manifest-before.json`,
  };
}

function advanceJournal(journalPath, journal, phase, extra = {}) {
  const next = {
    ...journal,
    ...extra,
    phase,
    updatedAt: new Date().toISOString(),
  };
  writeJsonDurable(journalPath, next);
  return next;
}

function worldManifestSha(world) {
  if (!fs.existsSync(world)) return null;
  requireCondition(fs.statSync(world).isDirectory(), `recovery world path is not a directory: ${world}`);
  const manifest = path.join(world, "manifest.json");
  ensureRegularFile(manifest, "recovery world manifest");
  return sha256File(manifest);
}

function durableRename(source, target, renameDirectory) {
  renameDirectory(source, target);
  fsyncDirectory(path.dirname(source));
  if (path.dirname(target) !== path.dirname(source)) fsyncDirectory(path.dirname(target));
}

function validateRecoveryJournal(journalPath, journal) {
  requireCondition(journal.schemaVersion === 1, "swap journal schemaVersion is invalid");
  requireCondition(
    journal.kind === "video2world.bedroom4_strict_clean_scene_swap_journal",
    "swap journal kind is invalid",
  );
  requireCondition(journal.acceptanceScope === "current_demo_only", "swap journal scope is invalid");
  requireCondition(journal.promotionAllowed === false, "swap journal promotionAllowed must remain false");
  const paths = journal.paths;
  const hashes = journal.hashes;
  requireCondition(paths && typeof paths === "object" && hashes && typeof hashes === "object",
    "swap journal paths/hashes are missing");
  for (const key of [
    "candidateWorld",
    "canonicalWorld",
    "backupWorld",
    "publicManifest",
    "candidateManifestSnapshot",
    "receiptPath",
    "lockPath",
  ]) {
    requireCondition(path.isAbsolute(paths[key]), `swap journal ${key} must be absolute`);
  }
  for (const key of [
    "oldCanonicalManifestSha256",
    "candidateManifestSha256BeforeRewrite",
    "promotedManifestSha256",
    "candidateManifestSnapshotSha256",
    "stableAliasSha256",
  ]) requireCondition(isSha256(hashes[key]), `swap journal hash ${key} is invalid`);
  requireCondition(hashes.stableAliasSha256 === BEDROOM4_STABLE_ALIAS_SHA256,
    "swap journal stable alias pin is invalid");
  requireCondition(journalPath === `${paths.backupWorld}.swap-journal.json`,
    "swap journal path does not match backup-world");
  requireCondition(paths.candidateManifestSnapshot === `${paths.backupWorld}.candidate-manifest-before.json`,
    "swap journal snapshot path does not match backup-world");
  const worldsRoot = path.dirname(paths.canonicalWorld);
  const realWorldsRoot = realDirectory(worldsRoot, "swap journal worlds root");
  requireCondition(realWorldsRoot === worldsRoot, "swap journal worlds root is not literal");
  requireCondition(
    path.basename(worldsRoot) === "worlds"
      && path.basename(path.dirname(worldsRoot)) === "public"
      && path.basename(path.dirname(path.dirname(worldsRoot))) === "web",
    "swap journal worlds root is not web/public/worlds",
  );
  requireCondition(path.dirname(paths.candidateWorld) === worldsRoot,
    "swap journal candidate/canonical worlds are not siblings");
  requireCondition(paths.lockPath === canonicalLockPath(paths.canonicalWorld),
    "swap journal lock path does not match canonical world");
  assertNoSymlinkAncestors(paths.candidateWorld, "swap journal candidate world", {
    allowMissingLeaf: true,
  });
  assertNoSymlinkAncestors(paths.canonicalWorld, "swap journal canonical world", {
    allowMissingLeaf: true,
  });
  requireCondition(!isInside(worldsRoot, paths.backupWorld), "swap journal backup is inside worlds root");
  requireCondition(
    !isInside(paths.candidateWorld, paths.backupWorld)
      && !isInside(paths.backupWorld, paths.candidateWorld)
      && !isInside(paths.canonicalWorld, paths.backupWorld)
      && !isInside(paths.backupWorld, paths.canonicalWorld),
    "swap journal paths overlap",
  );
  requireCondition(
    paths.publicManifest === path.join(paths.canonicalWorld, "manifest.json"),
    "swap journal public manifest path is invalid",
  );
  const backupParent = realDirectory(path.dirname(paths.backupWorld), "swap journal backup parent");
  requireCondition(path.join(backupParent, path.basename(paths.backupWorld)) === paths.backupWorld,
    "swap journal backup parent is not literal");
  assertNoSymlinkAncestors(paths.backupWorld, "swap journal backup world", {
    allowMissingLeaf: true,
  });
  requireCondition(
    fs.statSync(realWorldsRoot).dev === fs.statSync(backupParent).dev,
    "swap journal paths are on different devices",
  );
  requireCondition(
    paths.receiptPath
      === path.join(paths.canonicalWorld, "qa", "strict-clean-scene-swap-receipt.json"),
    "swap journal receipt path is invalid",
  );
  requireCondition(journal.lock?.originalToken && typeof journal.lock.originalToken === "string",
    "swap journal original lock token is missing");
  const snapshotExists = fs.existsSync(paths.candidateManifestSnapshot);
  requireCondition(snapshotExists || journal.phase === "prepared",
    "candidate manifest recovery snapshot is missing after prepared phase");
  if (snapshotExists) {
    ensureRegularFile(paths.candidateManifestSnapshot, "candidate manifest recovery snapshot");
    requireCondition(
      sha256File(paths.candidateManifestSnapshot) === hashes.candidateManifestSnapshotSha256
        && hashes.candidateManifestSnapshotSha256 === hashes.candidateManifestSha256BeforeRewrite,
      "candidate manifest recovery snapshot SHA-256 mismatch",
    );
  }
  if (hashes.transactionReceiptSha256 != null) {
    requireCondition(isSha256(hashes.transactionReceiptSha256),
      "swap journal transaction receipt SHA-256 is invalid");
  }
  return { paths, hashes, worldsRoot };
}

function validateCompleteRecoveryState(paths, hashes, expectedStableAliasSha) {
  requireCondition(worldManifestSha(paths.canonicalWorld) === hashes.promotedManifestSha256,
    "complete recovery: promoted canonical manifest SHA-256 mismatch");
  requireCondition(worldManifestSha(paths.candidateWorld) == null,
    "complete recovery: candidate path unexpectedly exists");
  requireCondition(worldManifestSha(paths.backupWorld) === hashes.oldCanonicalManifestSha256,
    "complete recovery: backup manifest SHA-256 mismatch");
  const canonicalPrefix = `./worlds/${path.basename(paths.canonicalWorld)}`;
  verifyManifestClosure(
    readJson(path.join(paths.canonicalWorld, "manifest.json"), "complete promoted manifest"),
    paths.canonicalWorld,
    canonicalPrefix,
    "complete promoted manifest",
  );
  verifyManifestClosure(
    readJson(path.join(paths.backupWorld, "manifest.json"), "complete backup manifest"),
    paths.backupWorld,
    canonicalPrefix,
    "complete backup manifest",
  );
  verifyStableAliasClosure(
    path.join(paths.canonicalWorld, "manifest.web-demo-baseline-stable.json"),
    paths.canonicalWorld,
    canonicalPrefix,
    expectedStableAliasSha,
    "complete canonical",
  );
  verifyStableAliasClosure(
    path.join(paths.backupWorld, "manifest.web-demo-baseline-stable.json"),
    paths.backupWorld,
    canonicalPrefix,
    expectedStableAliasSha,
    "complete backup",
  );
  requireCondition(isSha256(hashes.transactionReceiptSha256),
    "complete recovery: transaction receipt SHA-256 is missing");
  ensureRegularFile(paths.receiptPath, "complete recovery transaction receipt");
  requireCondition(sha256File(paths.receiptPath) === hashes.transactionReceiptSha256,
    "complete recovery: transaction receipt SHA-256 mismatch");
}

function validateRolledBackRecoveryState(paths, hashes, expectedStableAliasSha) {
  requireCondition(worldManifestSha(paths.canonicalWorld) === hashes.oldCanonicalManifestSha256,
    "rolled-back recovery: canonical manifest SHA-256 mismatch");
  requireCondition(worldManifestSha(paths.candidateWorld) === hashes.candidateManifestSha256BeforeRewrite,
    "rolled-back recovery: candidate manifest SHA-256 mismatch");
  requireCondition(!fs.existsSync(paths.backupWorld), "rolled-back recovery: backup path remains");
  const canonicalPrefix = `./worlds/${path.basename(paths.canonicalWorld)}`;
  const candidatePrefix = `./worlds/${path.basename(paths.candidateWorld)}`;
  verifyManifestClosure(
    readJson(path.join(paths.canonicalWorld, "manifest.json"), "rolled-back canonical manifest"),
    paths.canonicalWorld,
    canonicalPrefix,
    "rolled-back canonical manifest",
  );
  verifyManifestClosure(
    readJson(path.join(paths.candidateWorld, "manifest.json"), "rolled-back candidate manifest"),
    paths.candidateWorld,
    candidatePrefix,
    "rolled-back candidate manifest",
  );
  verifyStableAliasClosure(
    path.join(paths.canonicalWorld, "manifest.web-demo-baseline-stable.json"),
    paths.canonicalWorld,
    canonicalPrefix,
    expectedStableAliasSha,
    "rolled-back canonical",
  );
  verifyStableAliasClosure(
    path.join(paths.candidateWorld, "manifest.web-demo-baseline-stable.json"),
    paths.candidateWorld,
    canonicalPrefix,
    expectedStableAliasSha,
    "rolled-back candidate",
  );
}

export function recoverBedroom4StrictCleanSceneSwap(journalInput, options = {}) {
  const journalPath = ensureRegularFile(journalInput, "swap journal");
  const preview = readJson(journalPath, "swap journal");
  requireCondition(path.isAbsolute(preview.paths?.canonicalWorld),
    "swap journal canonicalWorld must be absolute before lock acquisition");
  const ownsLock = options.lockHandle == null;
  const lockHandle = options.lockHandle || acquireCanonicalLock(preview.paths.canonicalWorld, {
    backupWorld: preview.paths.backupWorld,
    journalPath,
    allowStaleRecovery: true,
  });
  try {
    let journal = readJson(journalPath, "swap journal");
    const { paths, hashes } = validateRecoveryJournal(journalPath, journal);
    requireCondition(lockHandle.lockPath === paths.lockPath,
      "recovery acquired a lock for another canonical world");
    if (journal.phase === "complete") {
      validateCompleteRecoveryState(paths, hashes, BEDROOM4_STABLE_ALIAS_SHA256);
      return { status: "already_complete_verified", phase: journal.phase, journalPath };
    }
    if (journal.phase === "recovered_rolled_back") {
      validateRolledBackRecoveryState(paths, hashes, BEDROOM4_STABLE_ALIAS_SHA256);
      return { status: "already_recovered_verified", phase: journal.phase, journalPath };
    }
    const renameDirectory = options.renameDirectory || fs.renameSync;
    let canonicalSha = worldManifestSha(paths.canonicalWorld);
    let candidateSha = worldManifestSha(paths.candidateWorld);
    let backupSha = worldManifestSha(paths.backupWorld);

    if (canonicalSha === hashes.promotedManifestSha256) {
      requireCondition(candidateSha == null, "cannot recover: candidate path already exists beside promoted canonical");
      durableRename(paths.canonicalWorld, paths.candidateWorld, renameDirectory);
      journal = advanceJournal(journalPath, journal, "recovery_candidate_restored");
      canonicalSha = null;
      candidateSha = hashes.promotedManifestSha256;
    } else if (canonicalSha != null) {
      requireCondition(canonicalSha === hashes.oldCanonicalManifestSha256,
        "cannot recover: canonical manifest has an unknown SHA-256");
    }

    backupSha = worldManifestSha(paths.backupWorld);
    if (canonicalSha == null) {
      requireCondition(backupSha === hashes.oldCanonicalManifestSha256,
        "cannot recover: old canonical backup is missing or hash-mismatched");
      durableRename(paths.backupWorld, paths.canonicalWorld, renameDirectory);
      journal = advanceJournal(journalPath, journal, "recovery_canonical_restored");
      canonicalSha = hashes.oldCanonicalManifestSha256;
    } else {
      requireCondition(backupSha == null, "cannot recover: canonical and backup both exist");
    }

    candidateSha = worldManifestSha(paths.candidateWorld);
    requireCondition(
      candidateSha === hashes.promotedManifestSha256
        || candidateSha === hashes.candidateManifestSha256BeforeRewrite,
      "cannot recover: candidate manifest is missing or has an unknown SHA-256",
    );
    if (candidateSha === hashes.promotedManifestSha256) {
      ensureRegularFile(paths.candidateManifestSnapshot, "candidate manifest recovery snapshot");
      writeBytesDurable(
        path.join(paths.candidateWorld, "manifest.json"),
        fs.readFileSync(paths.candidateManifestSnapshot),
      );
    }
    requireCondition(worldManifestSha(paths.candidateWorld) === hashes.candidateManifestSha256BeforeRewrite,
      "candidate manifest rollback SHA-256 mismatch");
    const receiptRelative = path.relative(paths.canonicalWorld, paths.receiptPath);
    requireCondition(!receiptRelative.startsWith("..") && !path.isAbsolute(receiptRelative),
      "swap journal receipt path escapes canonical world");
    const candidateReceipt = path.join(paths.candidateWorld, receiptRelative);
    if (fs.existsSync(candidateReceipt)) {
      requireCondition(isSha256(hashes.transactionReceiptSha256),
        "cannot archive a receipt that is not bound to this transaction");
      requireCondition(sha256File(candidateReceipt) === hashes.transactionReceiptSha256,
        "refusing to archive a receipt not created by this transaction");
      const abortedReceipt = `${candidateReceipt}.aborted-${Date.now()}`;
      fs.renameSync(candidateReceipt, abortedReceipt);
      fsyncDirectory(path.dirname(candidateReceipt));
    }
    journal = advanceJournal(journalPath, journal, "recovered_rolled_back", {
      status: "recovered_rolled_back",
      recoveredAt: new Date().toISOString(),
    });
    validateRolledBackRecoveryState(paths, hashes, BEDROOM4_STABLE_ALIAS_SHA256);
    return { status: journal.status, phase: journal.phase, journalPath };
  } finally {
    if (ownsLock) releaseCanonicalLock(lockHandle);
  }
}

export function promoteBedroom4StrictCleanScene(options) {
  const requestedJournalPath = `${path.resolve(options.backupWorld)}.swap-journal.json`;
  const lockHandle = acquireCanonicalLock(options.canonicalSourceWorld, {
    backupWorld: options.backupWorld,
    journalPath: requestedJournalPath,
  });
  try {
    const boundaries = validateBoundaries({
    candidateWorld: options.candidateWorld,
    candidateQaPath: options.candidateBrowserQaReport,
    canonicalWorld: options.canonicalSourceWorld,
    backupWorld: options.backupWorld,
    publicManifest: options.publicManifest,
  });
    const {
    candidateWorld,
    candidateQaPath,
    canonicalWorld,
    backupWorld,
    publicManifest,
    projectRoot,
    } = boundaries;
    const expectedStableAliasSha = BEDROOM4_STABLE_ALIAS_SHA256;

    const candidatePrefix = `./worlds/${path.basename(candidateWorld)}`;
    const canonicalPrefix = `./worlds/${path.basename(canonicalWorld)}`;
  const candidateManifestPath = path.join(candidateWorld, "manifest.json");
  const adoptionReportPath = path.join(candidateWorld, "qa", "strict-clean-scene-adoption-report.json");
  const sourceStableAlias = path.join(canonicalWorld, "manifest.web-demo-baseline-stable.json");
  const candidateStableAlias = path.join(candidateWorld, "manifest.web-demo-baseline-stable.json");
  for (const [filePath, label] of [
    [candidateManifestPath, "candidate manifest"],
    [adoptionReportPath, "candidate adoption report"],
    [sourceStableAlias, "canonical stable alias"],
    [candidateStableAlias, "candidate stable alias"],
  ]) ensureRegularFile(filePath, label);
    const canonicalStableClosureBefore = verifyStableAliasClosure(
      sourceStableAlias,
      canonicalWorld,
      canonicalPrefix,
      expectedStableAliasSha,
      "canonical",
    );
    const candidateStableClosureBefore = verifyStableAliasClosure(
      candidateStableAlias,
      candidateWorld,
      canonicalPrefix,
      expectedStableAliasSha,
      "candidate",
    );

  const canonicalManifestSha = sha256File(publicManifest);
  const candidateManifestShaBefore = sha256File(candidateManifestPath);
  const adoptionReportSha = sha256File(adoptionReportPath);
  const candidateQaSha = sha256File(candidateQaPath);
  const adoptionReport = readJson(adoptionReportPath, "candidate adoption report");
  const candidateQa = readJson(candidateQaPath, "candidate browser QA report");
  const candidateManifest = readJson(candidateManifestPath, "candidate manifest");
  validateAdoptionReport(adoptionReport, candidateManifestShaBefore, canonicalManifestSha);
  validateCandidateManifest(
    candidateManifest,
    candidateManifestShaBefore,
    candidatePrefix,
    adoptionReportSha,
  );
  const qaEvidence = validateBrowserQa({
    report: candidateQa,
    candidateManifest,
    candidateManifestPath,
    candidateManifestSha: candidateManifestShaBefore,
    candidateWorld,
    projectRoot,
  });
  const candidateUrlCount = assertCandidateUrlsResolve(candidateManifest, candidateWorld, candidatePrefix);
  const assetDescriptorCount = verifyAllManifestAssets(
    candidateManifest,
    candidateWorld,
    candidatePrefix,
  );

  const candidateWorldBefore = directoryInventory(candidateWorld);
  const canonicalWorldBefore = directoryInventory(canonicalWorld);
  const oldChunksBefore = directoryInventory(path.join(canonicalWorld, "chunks"));
  const originalCandidateManifest = fs.readFileSync(candidateManifestPath);
  const promotedManifest = rewritePrefix(candidateManifest, candidatePrefix, canonicalPrefix);
  requireCondition(
    collectPrefixUrls(promotedManifest, candidatePrefix).length === 0,
    "candidate URL prefix remains after rewrite",
  );
  promotedManifest.candidateBuild.status = "materialized_pending_promoted_manifest_recheck";
  promotedManifest.candidateBuild.promotionAllowed = false;
  promotedManifest.candidateBuild.promotionBlockers = [
    "promoted_manifest_browser_qa_recheck_pending",
    "finalization_not_performed",
  ];
  promotedManifest.candidateBuild.promotionSwap = {
    status: "materialized_pending_promoted_manifest_recheck",
    candidateManifestSha256: candidateManifestShaBefore,
    candidateBrowserQaReportSha256: candidateQaSha,
    candidateAdoptionReportSha256: adoptionReportSha,
    stableAliasSha256: expectedStableAliasSha,
    finalQaClaimed: false,
    promotionAllowed: false,
  };
  if (promotedManifest.productionBuild?.strictCleanSceneCandidate) {
    promotedManifest.productionBuild.strictCleanSceneCandidate.status =
      "materialized_pending_promoted_manifest_recheck";
    promotedManifest.productionBuild.strictCleanSceneCandidate.promotionApproved = false;
  }
  promotedManifest.sourceWorld = {
    ...(promotedManifest.sourceWorld || {}),
    adoptionMode: "strict_clean_scene_swapped_pending_promoted_manifest_recheck",
  };
  validateWebManifest(promotedManifest);
  const promotedManifestBytes = Buffer.from(`${JSON.stringify(promotedManifest, null, 2)}\n`);
  const promotedManifestSha = crypto.createHash("sha256").update(promotedManifestBytes).digest("hex");
    const { journalPath, candidateManifestSnapshotPath } = journalPaths(backupWorld);
    requireCondition(journalPath === requestedJournalPath,
      "canonical lock journal path differs from validated backup journal path");
  requireCondition(!fs.existsSync(journalPath), `swap journal already exists: ${journalPath}`);
  requireCondition(!fs.existsSync(candidateManifestSnapshotPath),
    `candidate manifest recovery snapshot already exists: ${candidateManifestSnapshotPath}`);
    const receiptPath = path.join(canonicalWorld, "qa", "strict-clean-scene-swap-receipt.json");
    const candidateReceiptPath = path.join(candidateWorld, "qa", path.basename(receiptPath));
    requireCondition(!fs.existsSync(candidateReceiptPath),
      "candidate world already contains a swap receipt; refusing to overwrite it");
    let journal = {
    schemaVersion: 1,
    kind: "video2world.bedroom4_strict_clean_scene_swap_journal",
    status: "in_progress",
    phase: "prepared",
    acceptanceScope: "current_demo_only",
    promotionAllowed: false,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    paths: {
      candidateWorld,
      canonicalWorld,
      backupWorld,
      publicManifest,
      candidateManifestSnapshot: candidateManifestSnapshotPath,
      receiptPath,
      lockPath: lockHandle.lockPath,
    },
    hashes: {
      oldCanonicalManifestSha256: canonicalManifestSha,
      candidateManifestSha256BeforeRewrite: candidateManifestShaBefore,
      promotedManifestSha256: promotedManifestSha,
      candidateManifestSnapshotSha256: candidateManifestShaBefore,
      candidateBrowserQaReportSha256: candidateQaSha,
      candidateAdoptionReportSha256: adoptionReportSha,
      stableAliasSha256: expectedStableAliasSha,
    },
    lock: {
      originalToken: lockHandle.owner.token,
      ownerPid: lockHandle.owner.pid,
      ownerHost: lockHandle.owner.hostname,
      startedAt: lockHandle.owner.startedAt,
    },
  };
    writeJsonDurable(journalPath, journal);
    const writeSnapshot = options.writeSnapshot || writeBytesDurable;
    writeSnapshot(candidateManifestSnapshotPath, originalCandidateManifest);
    requireCondition(sha256File(candidateManifestSnapshotPath) === candidateManifestShaBefore,
      "candidate manifest recovery snapshot SHA-256 mismatch");
    journal = advanceJournal(journalPath, journal, "snapshot_created");
  const renameDirectory = options.renameDirectory || fs.renameSync;
  try {
    writeBytesDurable(candidateManifestPath, promotedManifestBytes);
    requireCondition(sha256File(candidateManifestPath) === promotedManifestSha,
      "candidate manifest rewrite SHA-256 mismatch");
    journal = advanceJournal(journalPath, journal, "manifest_rewritten");
    durableRename(canonicalWorld, backupWorld, renameDirectory);
    journal = advanceJournal(journalPath, journal, "canonical_moved_to_backup");
    durableRename(candidateWorld, canonicalWorld, renameDirectory);
    journal = advanceJournal(journalPath, journal, "candidate_moved_to_canonical");
    requireCondition(sha256File(publicManifest) === promotedManifestSha, "promoted public manifest SHA mismatch");
    const promotedUrlCount = assertCanonicalUrlsResolve(
      promotedManifest,
      canonicalWorld,
      canonicalPrefix,
    );
    requireCondition(promotedUrlCount === candidateUrlCount, "URL count changed during prefix rewrite");
    const promotedAssetDescriptorCount = verifyAllManifestAssets(
      promotedManifest,
      canonicalWorld,
      canonicalPrefix,
    );
    requireCondition(promotedAssetDescriptorCount === assetDescriptorCount,
      "promoted manifest asset descriptor count changed after swap");
    const canonicalStableClosureAfter = verifyStableAliasClosure(
      path.join(canonicalWorld, "manifest.web-demo-baseline-stable.json"),
      canonicalWorld,
      canonicalPrefix,
      expectedStableAliasSha,
      "promoted canonical",
    );
    const backupStableClosureAfter = verifyStableAliasClosure(
      path.join(backupWorld, "manifest.web-demo-baseline-stable.json"),
      backupWorld,
      canonicalPrefix,
      expectedStableAliasSha,
      "backup",
    );
    requireCondition(stableJson(canonicalStableClosureAfter) === stableJson(candidateStableClosureBefore),
      "promoted stable alias closure count changed during swap");
    requireCondition(stableJson(backupStableClosureAfter) === stableJson(canonicalStableClosureBefore),
      "backup stable alias closure count changed during swap");
    requireCondition(
      sha256File(path.join(backupWorld, "manifest.json")) === canonicalManifestSha,
      "backup world does not preserve the old canonical manifest",
    );
    const canonicalWorldAfter = directoryInventory(canonicalWorld);
    const backupWorldAfter = directoryInventory(backupWorld);
    const oldChunksAfter = directoryInventory(path.join(backupWorld, "chunks"));
    requireCondition(
      backupWorldAfter.inode === canonicalWorldBefore.inode
        && backupWorldAfter.fileCount === canonicalWorldBefore.fileCount
        && backupWorldAfter.totalBytes === canonicalWorldBefore.totalBytes,
      "backup world inventory does not match the old canonical world",
    );
    requireCondition(
      oldChunksAfter.inode === oldChunksBefore.inode
        && oldChunksAfter.fileCount === oldChunksBefore.fileCount
        && oldChunksAfter.totalBytes === oldChunksBefore.totalBytes,
      "backup world does not preserve the old chunks directory",
    );
    const receipt = {
      schemaVersion: 1,
      kind: "video2world.bedroom4_strict_clean_scene_swap_receipt",
      status: "materialized_pending_promoted_manifest_recheck",
      acceptanceScope: "current_demo_only",
      promotionAllowed: false,
      promotionApproved: false,
      finalQaClaimed: false,
      candidateEvidence: {
        manifestSha256Before: candidateManifestShaBefore,
        browserQaReportSha256: candidateQaSha,
        adoptionReportSha256: adoptionReportSha,
        browserQaStatus: candidateQa.status,
        blockingFailures: [],
        validatedDesktopAndMobileQa: true,
        screenshotCount: qaEvidence.screenshotCount,
        assetDescriptorCount,
      },
      promotedManifest: {
        path: publicManifest,
        sha256: promotedManifestSha,
        urlPrefix: canonicalPrefix,
        browserQaRecheck: "required_pending",
      },
      stableAlias: {
        fileName: "manifest.web-demo-baseline-stable.json",
        sha256: expectedStableAliasSha,
        preservedInCanonicalWorld: true,
        preservedInBackupWorld: true,
        closure: {
          canonicalBefore: canonicalStableClosureBefore,
          candidateBefore: candidateStableClosureBefore,
          promotedCanonical: canonicalStableClosureAfter,
          backup: backupStableClosureAfter,
        },
      },
      swap: {
        mode: "durable_journaled_two_rename_transaction",
        transactionFullyAtomic: false,
        individualDirectoryRenamesAtomicOnSameDevice: true,
        crashRecoveryUsesDurableJournal: true,
        candidateWorldOriginalPath: candidateWorld,
        canonicalWorldPath: canonicalWorld,
        oldCanonicalBackupPath: backupWorld,
        sourceWorldDeleted: false,
        candidateWorldDeleted: false,
        journalPath,
      },
      rollback: {
        automaticOnFailure: true,
        cliRecovery: `node scripts/promote_bedroom4_strict_clean_scene.mjs --recover-journal ${journalPath}`,
        canonicalRestoreFrom: backupWorld,
        candidateRestoreTo: candidateWorld,
        oldCanonicalManifestSha256: canonicalManifestSha,
        candidateManifestSha256BeforeRewrite: candidateManifestShaBefore,
        promotedManifestSha256: promotedManifestSha,
      },
      inventories: {
        candidateBefore: candidateWorldBefore,
        oldCanonicalBefore: canonicalWorldBefore,
        oldCanonicalAtBackup: backupWorldAfter,
        oldChunksBefore,
        oldChunksAtBackup: oldChunksAfter,
        promotedCanonical: canonicalWorldAfter,
      },
      gates: {
        adoptionReportHashBound: true,
        candidateBrowserQaPassedAndHashBound: true,
        candidateBlockingFailuresEmpty: true,
        desktopAndMobileRuntimeQaRevalidated: true,
        manifestAssetDescriptorsRehashed: true,
        promotedManifestAssetDescriptorsRehashedAfterSwap: true,
        localAssetUrlsBrowserNormalizedAndBounded: true,
        realpathAndSymlinkBoundariesPassed: true,
        stableAliasSha256Preserved: true,
        stableAliasAssetClosureRehashedBeforeAndAfterSwap: true,
        candidatePrefixRewrittenToCanonical: true,
        oldCanonicalWorldPreservedAtBackup: true,
        oldChunksPreservedAtBackup: true,
        promotedManifestRecheckRequired: true,
        finalQaNotClaimed: true,
      },
    };
    const receiptBytes = Buffer.from(`${JSON.stringify(receipt, null, 2)}\n`);
    const transactionReceiptSha256 = crypto.createHash("sha256").update(receiptBytes).digest("hex");
    journal = advanceJournal(journalPath, journal, "receipt_write_prepared", {
      hashes: { ...journal.hashes, transactionReceiptSha256 },
    });
    writeBytesDurable(receiptPath, receiptBytes);
    requireCondition(sha256File(receiptPath) === transactionReceiptSha256,
      "transaction receipt SHA-256 mismatch after write");
    journal = advanceJournal(journalPath, journal, "receipt_written");
    journal = advanceJournal(journalPath, journal, "complete", {
      status: "complete_pending_promoted_manifest_recheck",
      completedAt: new Date().toISOString(),
      receiptSha256: transactionReceiptSha256,
    });
    return {
      receipt,
      receiptPath,
      receiptSha256: transactionReceiptSha256,
      promotedManifestSha256: promotedManifestSha,
      journalPath,
      journalStatus: journal.status,
    };
  } catch (error) {
    let rollbackError = null;
    try {
      recoverBedroom4StrictCleanSceneSwap(journalPath, { renameDirectory, lockHandle });
    } catch (restoreError) {
      rollbackError = restoreError;
    }
    if (rollbackError) {
      throw new AggregateError([error, rollbackError], "swap failed and automatic rollback also failed");
    }
    throw error;
    }
  } finally {
    releaseCanonicalLock(lockHandle);
  }
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.mode === "recover") {
    const recovered = recoverBedroom4StrictCleanSceneSwap(args.journal);
    process.stdout.write(`${JSON.stringify(recovered)}\n`);
    return;
  }
  const result = promoteBedroom4StrictCleanScene({
    candidateWorld: args["candidate-world"],
    candidateBrowserQaReport: args["candidate-browser-qa-report"],
    canonicalSourceWorld: args["canonical-source-world"],
    backupWorld: args["backup-world"],
    publicManifest: args["public-manifest"],
  });
  process.stdout.write(`${JSON.stringify({
    status: result.receipt.status,
    receipt: result.receiptPath,
    receiptSha256: result.receiptSha256,
    promotedManifestSha256: result.promotedManifestSha256,
    journal: result.journalPath,
    journalStatus: result.journalStatus,
    promotionAllowed: false,
    promotedManifestBrowserQaRecheck: "required_pending",
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
