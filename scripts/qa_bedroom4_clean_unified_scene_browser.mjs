#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";
import { Matrix4 } from "three";

const scriptPath = fileURLToPath(import.meta.url);
const scriptDir = path.dirname(scriptPath);
const repoRoot = path.resolve(scriptDir, "..");
const publicRoot = path.join(repoRoot, "web/public");
const worldDir = path.join(publicRoot, "worlds/bedroom4");

const DEFAULT_MANIFEST = path.join(worldDir, "manifest.clean-unified-scene.json");
const DEFAULT_REPORT = path.join(worldDir, "qa/clean-unified-scene-browser-qa.json");
const DEFAULT_SCREENSHOT_DIR = path.join(worldDir, "qa/clean-unified-scene-browser");
const DEFAULT_URL = "http://127.0.0.1:4177/";
const DEFAULT_MIN_FPS = 18;
const IMPLEMENTATION_FILES = Object.freeze({
  qaRunner: scriptPath,
  runtime: path.join(repoRoot, "web/visual-physics-proxy.js"),
  manifestValidator: path.join(repoRoot, "web/web-manifest.js"),
  materializer: path.join(repoRoot, "scripts/materialize_bedroom4_strict_clean_scene.mjs"),
});

export const UNIFIED_IDS = Object.freeze([
  "sam3_pillow_front",
  "sam3_pillow_left",
  "sam3_pillow_right",
  "sam3_bed_01",
]);
export const EXPECTED_TOPOLOGY = Object.freeze({
  sam3_pillow_front: "surface_bvh",
  sam3_pillow_left: "surface_bvh",
  sam3_pillow_right: "surface_bvh",
  sam3_bed_01: "closed_volume",
});
const PILLOW_IDS = new Set(UNIFIED_IDS.slice(0, 3));
const BED_ID = "sam3_bed_01";
export const CAMERA_OVERVIEW_COVERAGE_MIN = 0.08;
export const CAMERA_OVERVIEW_COVERAGE_MAX = 0.92;
const EXPECTED_OBJECT_COUNT = 8;
const MATRIX_EPSILON = 1e-5;
export const MATRIX_GRAM_EPSILON = 2e-5;
export const CAMERA_TARGET_CENTER_NORMALIZED_MAX = 0.05;
export const APPROVED_NESTED_SUPPORT = Object.freeze({
  receiptSha256: "9d51f3ec09869ee6ca9ca4ffc21e77f25cf350d02665f682b241dda34778b320",
  parentObjectId: BED_ID,
  childObjectIds: Object.freeze([...PILLOW_IDS]),
  maximumPenetrationWorld: 0.2,
  maximumSupportGapWorld: 0.2,
});

function requireCondition(condition, message) {
  if (!condition) throw new Error(message);
}

export function cameraTargetBoundsEvidence(state, inspection) {
  const target = state?.cameraTarget;
  const bounds = inspection?.collisionBounds;
  const axes = ["x", "y", "z"];
  const finiteVector = (value) => value && axes.every((axis) => Number.isFinite(value[axis]));
  if (!finiteVector(target) || !finiteVector(bounds?.min) || !finiteVector(bounds?.max)) {
    return {
      passed: false,
      centerDistance: null,
      normalizedCenterDistance: null,
      targetInsideBounds: false,
    };
  }
  const center = finiteVector(bounds.center)
    ? bounds.center
    : Object.fromEntries(axes.map((axis) => [axis, (bounds.min[axis] + bounds.max[axis]) / 2]));
  const size = axes.map((axis) => Math.max(0, bounds.max[axis] - bounds.min[axis]));
  const diagonal = Math.hypot(...size);
  const centerDistance = Math.hypot(...axes.map((axis) => target[axis] - center[axis]));
  const normalizedCenterDistance = centerDistance / Math.max(diagonal, Number.EPSILON);
  const targetInsideBounds = axes.every((axis) => (
    target[axis] >= bounds.min[axis] - MATRIX_EPSILON
      && target[axis] <= bounds.max[axis] + MATRIX_EPSILON
  ));
  return {
    passed: targetInsideBounds
      && Number.isFinite(normalizedCenterDistance)
      && normalizedCenterDistance <= CAMERA_TARGET_CENTER_NORMALIZED_MAX,
    centerDistance: rounded(centerDistance, 6),
    normalizedCenterDistance: rounded(normalizedCenterDistance, 6),
    targetInsideBounds,
  };
}

function positiveInteger(value, label) {
  requireCondition(Number.isInteger(value) && value > 0, `${label} must be a positive integer`);
  return value;
}

function sha256Bytes(bytes) {
  return crypto.createHash("sha256").update(bytes).digest("hex");
}

function sha256File(filePath) {
  return sha256Bytes(fs.readFileSync(filePath));
}

function fileDescriptor(filePath) {
  const stats = fs.statSync(filePath);
  requireCondition(stats.isFile() && stats.size > 0, `Expected non-empty file: ${filePath}`);
  return {
    path: filePath,
    sha256: sha256File(filePath),
    size: stats.size,
  };
}

function crc32(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit += 1) {
      crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0);
    }
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function pngDescriptor(filePath, expectedWidth, expectedHeight) {
  const bytes = fs.readFileSync(filePath);
  const signature = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  requireCondition(bytes.length >= 45 && bytes.subarray(0, 8).equals(signature),
    `Invalid PNG signature: ${filePath}`);
  let offset = 8;
  let chunkIndex = 0;
  let width = null;
  let height = null;
  let sawIdat = false;
  let sawIend = false;
  while (offset < bytes.length) {
    requireCondition(offset + 12 <= bytes.length, `Truncated PNG chunk header: ${filePath}`);
    const length = bytes.readUInt32BE(offset);
    const typeStart = offset + 4;
    const dataStart = typeStart + 4;
    const dataEnd = dataStart + length;
    const crcOffset = dataEnd;
    requireCondition(crcOffset + 4 <= bytes.length, `Truncated PNG chunk: ${filePath}`);
    const type = bytes.subarray(typeStart, dataStart).toString("ascii");
    const expectedCrc = bytes.readUInt32BE(crcOffset);
    const actualCrc = crc32(bytes.subarray(typeStart, dataEnd));
    requireCondition(actualCrc === expectedCrc, `PNG CRC mismatch for ${type}: ${filePath}`);
    if (chunkIndex === 0) {
      requireCondition(type === "IHDR" && length === 13, `PNG must start with IHDR: ${filePath}`);
      width = bytes.readUInt32BE(dataStart);
      height = bytes.readUInt32BE(dataStart + 4);
    }
    if (type === "IDAT") sawIdat = true;
    offset = crcOffset + 4;
    chunkIndex += 1;
    if (type === "IEND") {
      requireCondition(length === 0, `PNG IEND must be empty: ${filePath}`);
      sawIend = true;
      break;
    }
  }
  requireCondition(sawIdat && sawIend && offset === bytes.length,
    `PNG is missing IDAT/IEND or has trailing bytes: ${filePath}`);
  requireCondition(width === expectedWidth && height === expectedHeight,
    `PNG dimensions ${width}x${height} do not match ${expectedWidth}x${expectedHeight}: ${filePath}`);
  return { ...fileDescriptor(filePath), width, height };
}

function implementationDescriptors() {
  return Object.fromEntries(
    Object.entries(IMPLEMENTATION_FILES).map(([key, filePath]) => [key, fileDescriptor(filePath)]),
  );
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function writeJson(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  const temporary = `${filePath}.${process.pid}.tmp`;
  fs.writeFileSync(temporary, `${JSON.stringify(value, null, 2)}\n`);
  fs.renameSync(temporary, filePath);
}

function checkpointPath(reportPath) {
  return reportPath.endsWith(".json")
    ? `${reportPath.slice(0, -5)}.checkpoint.json`
    : `${reportPath}.checkpoint.json`;
}

function logStage(viewport, stage, details = {}) {
  console.log(JSON.stringify({ event: "browser_qa_stage", viewport, stage, ...details }));
}

function rounded(value, digits = 4) {
  return Number(Number(value).toFixed(digits));
}

function record(condition, message, failures) {
  if (!condition) failures.push(message);
  return Boolean(condition);
}

function portablePath(value) {
  if (typeof value !== "string" || !path.isAbsolute(value)) return value;
  if (value === repoRoot || value.startsWith(`${repoRoot}${path.sep}`)) {
    return `repo://video2world/${path.relative(repoRoot, value).split(path.sep).join("/")}`;
  }
  return `artifact://external/${path.basename(value)}`;
}

function portableReport(value) {
  if (Array.isArray(value)) return value.map(portableReport);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([key, item]) => [key, portableReport(item)]),
    );
  }
  return portablePath(value);
}

function matrixMaximumDelta(left, right) {
  if (!Array.isArray(left) || !Array.isArray(right) || left.length !== right.length) {
    return Infinity;
  }
  return Math.max(...left.map((value, index) => Math.abs(value - right[index])));
}

function matricesEqual(left, right, epsilon = MATRIX_EPSILON) {
  return matrixMaximumDelta(left, right) <= epsilon;
}

function matrixGram(matrix) {
  if (!Array.isArray(matrix) || matrix.length !== 16 || !matrix.every(Number.isFinite)) return null;
  const columns = [
    [matrix[0], matrix[1], matrix[2]],
    [matrix[4], matrix[5], matrix[6]],
    [matrix[8], matrix[9], matrix[10]],
  ];
  const dot = (left, right) => left.reduce((sum, value, index) => sum + value * right[index], 0);
  return [
    dot(columns[0], columns[0]),
    dot(columns[1], columns[1]),
    dot(columns[2], columns[2]),
    dot(columns[0], columns[1]),
    dot(columns[0], columns[2]),
    dot(columns[1], columns[2]),
  ];
}

function matrixTranslationDelta(left, right) {
  if (!Array.isArray(left) || !Array.isArray(right) || left.length !== 16 || right.length !== 16) {
    return Infinity;
  }
  return Math.max(...[12, 13, 14].map((index) => Math.abs(left[index] - right[index])));
}

function transformMotionAndRigidity(before, after) {
  const keys = ["groupMatrixWorld", "splatMatrixWorld", "collisionMatrixWorld"];
  return Object.fromEntries(keys.map((key) => [key, {
    matrixDelta: matrixMaximumDelta(before[key], after[key]),
    translationDelta: matrixTranslationDelta(before[key], after[key]),
    gramDelta: matrixMaximumDelta(matrixGram(before[key]), matrixGram(after[key])),
  }]));
}

function relativeMatrix(parentMatrix, childMatrix) {
  if (
    !Array.isArray(parentMatrix) || parentMatrix.length !== 16
    || !Array.isArray(childMatrix) || childMatrix.length !== 16
  ) return null;
  return new Matrix4()
    .fromArray(parentMatrix)
    .invert()
    .multiply(new Matrix4().fromArray(childMatrix))
    .elements;
}

function visualPrimitiveKind(object) {
  if (object.collision?.mode === "unified-glb") return "mesh";
  if (
    object.visual?.renderer === "three-points"
    || ["rgb-point-cloud-ply", "rgb-point-ply"].includes(object.visual?.format)
  ) {
    return "rgb-points";
  }
  return "gaussian-splat";
}

function sceneKnowledgeObject(manifest, objectId) {
  return (manifest.sceneKnowledge?.objects || []).find((item) => item.id === objectId) || null;
}

export function deriveManifestExpectations(manifest) {
  requireCondition(manifest && typeof manifest === "object", "manifest must be an object");
  requireCondition(
    manifest.initialState?.cameraFocusObjectId === BED_ID,
    `manifest.initialState.cameraFocusObjectId must be ${BED_ID}`,
  );
  const objects = manifest.interactiveObjects;
  requireCondition(Array.isArray(objects), "manifest.interactiveObjects must be an array");
  requireCondition(
    objects.length === EXPECTED_OBJECT_COUNT,
    `manifest must contain exactly ${EXPECTED_OBJECT_COUNT} interactive objects`,
  );
  const ids = objects.map((item) => item?.id);
  requireCondition(new Set(ids).size === ids.length, "interactive object IDs must be unique");
  for (const objectId of UNIFIED_IDS) {
    requireCondition(ids.includes(objectId), `manifest lacks required unified object ${objectId}`);
  }

  const bed = objects.find((item) => item.id === BED_ID);
  requireCondition(
    bed?.collision?.gate?.supportAdjustmentReceiptSha256
      === APPROVED_NESTED_SUPPORT.receiptSha256,
    `${BED_ID}: approved nested-support receipt is not bound`,
  );
  requireCondition(
    Array.isArray(bed.childObjectIds)
      && bed.childObjectIds.length === PILLOW_IDS.size
      && [...PILLOW_IDS].every((id) => bed.childObjectIds.includes(id)),
    `${BED_ID}: childObjectIds must contain the three approved pillows`,
  );

  const staticVisualCount = positiveInteger(
    manifest.assets?.visual?.vertexCount,
    "manifest.assets.visual.vertexCount",
  );
  const sceneAssetKey = manifest.collisionWorld?.sceneAssetKey;
  requireCondition(
    typeof sceneAssetKey === "string" && sceneAssetKey,
    "manifest.collisionWorld.sceneAssetKey is missing",
  );
  const staticColliderAsset = manifest.assets?.[sceneAssetKey];
  const staticColliderFaces = positiveInteger(
    staticColliderAsset?.faceCount ?? staticColliderAsset?.faces,
    `manifest.assets.${sceneAssetKey}.faceCount`,
  );

  const unified = {};
  let objectColliderFaces = 0;
  let objectColliderCount = 0;
  let meshVertices = 0;
  let gaussianVertices = 0;
  let rgbPointVertices = 0;
  for (const object of objects) {
    requireCondition(object && typeof object === "object", "interactive object is invalid");
    const collision = object.collision;
    requireCondition(collision && typeof collision === "object", `${object.id}: collision missing`);
    requireCondition(collision.gate?.status === "passed", `${object.id}: collision gate is not passed`);
    requireCondition(
      collision.characterCollision === true,
      `${object.id}: character collision must be enabled`,
    );
    const faces = positiveInteger(collision.asset?.faces, `${object.id}: collision faces`);
    objectColliderFaces += faces;
    objectColliderCount += 1;

    const kind = visualPrimitiveKind(object);
    if (kind === "mesh") {
      meshVertices += positiveInteger(collision.asset?.vertices, `${object.id}: mesh vertices`);
    } else if (kind === "rgb-points") {
      rgbPointVertices += positiveInteger(object.visual?.vertexCount, `${object.id}: RGB points`);
    } else {
      gaussianVertices += positiveInteger(object.visual?.vertexCount, `${object.id}: Gaussians`);
    }
  }

  const assetUrls = new Set();
  for (const objectId of UNIFIED_IDS) {
    const object = objects.find((item) => item.id === objectId);
    const collision = object.collision;
    const asset = collision.asset;
    requireCondition(collision.mode === "unified-glb", `${objectId}: mode must be unified-glb`);
    requireCondition(
      collision.topology === EXPECTED_TOPOLOGY[objectId],
      `${objectId}: topology must be ${EXPECTED_TOPOLOGY[objectId]}`,
    );
    requireCondition(object.visual == null, `${objectId}: unified visual must be null`);
    requireCondition(object.renderAsset == null, `${objectId}: renderAsset must be null`);
    requireCondition(object.colliderProxy == null, `${objectId}: colliderProxy must be null`);
    requireCondition(collision.renderAsset == null, `${objectId}: collision.renderAsset must be null`);
    requireCondition(
      object.interaction?.kind === "spin"
        && object.interaction?.drag === "horizontal_yaw"
        && object.interaction?.degrees === 360,
      `${objectId}: horizontal drag and 360 spin are required`,
    );
    requireCondition(object.independentlyMovable === true, `${objectId}: must be independently movable`);
    const isPillow = PILLOW_IDS.has(objectId);
    requireCondition(
      object.semanticGranularity === (isPillow ? "independent_child_asset" : "independent_root_asset"),
      `${objectId}: semantic hierarchy role is invalid`,
    );
    requireCondition(
      object.parentObjectId === (isPillow ? BED_ID : null),
      `${objectId}: parentObjectId is invalid`,
    );
    requireCondition(object.movesWithParent === isPillow, `${objectId}: movesWithParent is invalid`);
    requireCondition(typeof asset?.url === "string" && asset.url, `${objectId}: GLB URL missing`);
    requireCondition(/^[0-9a-f]{64}$/u.test(asset.sha256), `${objectId}: GLB SHA-256 invalid`);
    let canonicalAssetUrl;
    try {
      canonicalAssetUrl = new URL(asset.url, "https://video2world.invalid/runtime/").href;
    } catch {
      throw new Error(`${objectId}: GLB URL is invalid`);
    }
    requireCondition(!assetUrls.has(canonicalAssetUrl), `${objectId}: unified GLB URL is duplicated`);
    assetUrls.add(canonicalAssetUrl);
    requireCondition(asset.finite === true, `${objectId}: mesh must be finite`);
    requireCondition(asset.nondegenerate === true, `${objectId}: mesh must be nondegenerate`);
    requireCondition(asset.windingConsistent === true, `${objectId}: winding must be consistent`);
    if (objectId === BED_ID) {
      requireCondition(asset.watertight === true, `${objectId}: closed volume must be watertight`);
    }
    const knowledge = sceneKnowledgeObject(manifest, objectId);
    requireCondition(knowledge?.bbox, `${objectId}: scene knowledge bbox missing`);
    requireCondition(knowledge?.description, `${objectId}: scene knowledge description missing`);
    unified[objectId] = {
      id: objectId,
      topology: collision.topology,
      url: asset.url,
      sha256: asset.sha256,
      vertices: positiveInteger(asset.vertices, `${objectId}: vertices`),
      faces: positiveInteger(asset.faces, `${objectId}: faces`),
      parentObjectId: object.parentObjectId,
      movesWithParent: object.movesWithParent,
      limitations: Array.isArray(object.limitations) ? object.limitations : [],
    };
  }

  return {
    cameraFocusObjectId: manifest.initialState.cameraFocusObjectId,
    objectIds: ids,
    objectCount: objects.length,
    objectColliderCount,
    objectColliderFaces,
    staticVisualCount,
    staticColliderFaces,
    sceneAssetKey,
    meshVertices,
    gaussianVertices,
    rgbPointVertices,
    totalVisualPrimitives: staticVisualCount + meshVertices + gaussianVertices + rgbPointVertices,
    unified,
    nestedSupport: APPROVED_NESTED_SUPPORT,
    queryCases: [
      { label: "pillow_location", question: "前排枕头在哪里？", expectedId: "sam3_pillow_front" },
      { label: "pillow_appearance", question: "前排枕头长什么样？", expectedId: "sam3_pillow_front" },
      { label: "bed_location", question: "双人床在哪里？", expectedId: BED_ID },
      { label: "bed_appearance", question: "双人床长什么样？", expectedId: BED_ID },
    ],
  };
}

export function parseArgs(argv) {
  const allowed = new Set(["url", "manifest", "report", "screenshot-dir", "min-fps"]);
  const options = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === "--help") return { help: true };
    if (!token.startsWith("--")) throw new Error(`Unexpected argument: ${token}`);
    const key = token.slice(2);
    if (!allowed.has(key)) throw new Error(`Unsupported option: --${key}`);
    if (Object.hasOwn(options, key)) throw new Error(`Duplicate option: --${key}`);
    const value = argv[index + 1];
    if (!value || value.startsWith("--")) throw new Error(`Missing value for --${key}`);
    options[key] = value;
    index += 1;
  }
  if (options["min-fps"] != null) {
    const minimum = Number(options["min-fps"]);
    if (!Number.isFinite(minimum) || minimum <= 0) {
      throw new Error("--min-fps must be a positive finite number");
    }
    options.minFps = minimum;
  }
  return options;
}

export function usage() {
  return [
    "Usage: node scripts/qa_bedroom4_clean_unified_scene_browser.mjs [options]",
    "",
    `  --url URL                 Runtime URL (default: ${DEFAULT_URL})`,
    `  --manifest PATH           Clean unified manifest (default: ${DEFAULT_MANIFEST})`,
    `  --report PATH             JSON report (default: ${DEFAULT_REPORT})`,
    `  --screenshot-dir PATH     Screenshot directory (default: ${DEFAULT_SCREENSHOT_DIR})`,
    `  --min-fps NUMBER          Minimum average FPS per viewport (default: ${DEFAULT_MIN_FPS})`,
    "  --help                    Show this help",
    "",
    "This QA only writes its report and screenshots. It never edits or promotes the manifest.",
  ].join("\n");
}

function manifestPublicUrl(manifestPath) {
  const relative = path.relative(publicRoot, manifestPath);
  requireCondition(
    relative && !relative.startsWith("..") && !path.isAbsolute(relative),
    `Manifest must be inside ${publicRoot}`,
  );
  return `./${relative.split(path.sep).join("/")}`;
}

export function buildRuntimeUrl(rawUrl, manifestPath) {
  const url = new URL(rawUrl || DEFAULT_URL);
  url.searchParams.set("manifest", manifestPublicUrl(manifestPath));
  url.searchParams.set("collisionDebug", "on");
  return url.toString();
}

function createPageDiagnostics(page, { label, manifestPathname, assetPathnames }) {
  const diagnostics = {
    label,
    consoleMessages: [],
    pageErrors: [],
    requestFailures: [],
    httpErrors: [],
    manifestRequestCount: 0,
    manifestResponseSha256: [],
    assetRequestCounts: Object.fromEntries([...assetPathnames].map((pathname) => [pathname, 0])),
  };
  const responseTasks = [];
  page.on("console", (message) => {
    if (["error", "warning"].includes(message.type())) {
      diagnostics.consoleMessages.push({ type: message.type(), text: message.text() });
    }
  });
  page.on("pageerror", (error) => diagnostics.pageErrors.push(error.message));
  page.on("request", (request) => {
    const pathname = new URL(request.url()).pathname;
    if (pathname === manifestPathname) diagnostics.manifestRequestCount += 1;
    if (Object.hasOwn(diagnostics.assetRequestCounts, pathname)) {
      diagnostics.assetRequestCounts[pathname] += 1;
    }
  });
  page.on("requestfailed", (request) => {
    diagnostics.requestFailures.push({
      url: request.url(),
      method: request.method(),
      error: request.failure()?.errorText || "unknown",
    });
  });
  page.on("response", (response) => {
    const pathname = new URL(response.url()).pathname;
    if (response.status() >= 400) {
      diagnostics.httpErrors.push({ url: response.url(), status: response.status() });
    }
    if (pathname === manifestPathname) {
      responseTasks.push(
        response.body()
          .then((body) => diagnostics.manifestResponseSha256.push(sha256Bytes(body)))
          .catch((error) => diagnostics.pageErrors.push(`manifest response body: ${error.message}`)),
      );
    }
  });
  diagnostics.settle = async () => {
    let timeoutId;
    const outcome = await Promise.race([
      Promise.all(responseTasks).then(() => "settled"),
      new Promise((resolve) => {
        timeoutId = setTimeout(() => resolve("timeout"), 10_000);
      }),
    ]);
    clearTimeout(timeoutId);
    if (outcome === "timeout") {
      diagnostics.pageErrors.push("diagnostic response bodies did not settle within 10 seconds");
    }
    return diagnostics;
  };
  return diagnostics;
}

async function waitForReady(page, expected) {
  await page.waitForFunction((contract) => {
    const state = window.__visualPhysicsDemoState;
    return Boolean(state?.error) || Boolean(
      state?.visualReady
      && state?.colliderReady
      && state?.sceneQaReady
      && state?.interactiveObjectReadyCount === contract.objectCount
      && state?.objectColliderReadyCount === contract.objectColliderCount,
    );
  }, expected, { timeout: 300_000 });
  const state = await page.evaluate(() => window.__visualPhysicsDemoState);
  requireCondition(!state.error, `runtime load failed: ${state.error}`);
}

async function inspectCanvasPixels(page) {
  return page.locator("#sceneCanvas").evaluate(async (canvas) => {
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    const size = 96;
    const copy = document.createElement("canvas");
    copy.width = size;
    copy.height = size;
    const context = copy.getContext("2d", { willReadFrequently: true });
    context.drawImage(canvas, 0, 0, size, size);
    const pixels = context.getImageData(0, 0, size, size).data;
    let nonDarkPixels = 0;
    let minimumLuma = 255;
    let maximumLuma = 0;
    const colors = new Set();
    for (let index = 0; index < pixels.length; index += 4) {
      const red = pixels[index];
      const green = pixels[index + 1];
      const blue = pixels[index + 2];
      const alpha = pixels[index + 3];
      const luma = 0.2126 * red + 0.7152 * green + 0.0722 * blue;
      if (luma > 8) nonDarkPixels += 1;
      minimumLuma = Math.min(minimumLuma, luma);
      maximumLuma = Math.max(maximumLuma, luma);
      colors.add(`${red >> 4}:${green >> 4}:${blue >> 4}:${alpha >> 6}`);
    }
    return {
      sampleSize: [size, size],
      backingStore: [canvas.width, canvas.height],
      cssSize: [canvas.getBoundingClientRect().width, canvas.getBoundingClientRect().height],
      nonDarkPixels,
      quantizedColorCount: colors.size,
      lumaRange: Number((maximumLuma - minimumLuma).toFixed(3)),
      nonblank: nonDarkPixels >= 64 && colors.size >= 8 && maximumLuma - minimumLuma >= 8,
    };
  });
}

async function inspectLayout(page) {
  return page.evaluate(() => {
    const box = (selector) => {
      const element = document.querySelector(selector);
      if (!element) return null;
      const bounds = element.getBoundingClientRect();
      return {
        left: bounds.left,
        top: bounds.top,
        right: bounds.right,
        bottom: bounds.bottom,
        width: bounds.width,
        height: bounds.height,
        scrollWidth: element.scrollWidth,
        clientWidth: element.clientWidth,
        scrollHeight: element.scrollHeight,
        clientHeight: element.clientHeight,
      };
    };
    const canvas = box("#sceneCanvas");
    const hud = box(".hud");
    const queryRow = box(".scene-qa-row");
    const answer = box("#sceneQaAnswer");
    const metrics = box(".metrics");
    const status = box(".status-card");
    const controls = box(".controls");
    const before = (left, right) => !left || !right || left.bottom <= right.top + 0.5;
    return {
      viewport: [window.innerWidth, window.innerHeight],
      canvas,
      hud,
      queryRow,
      answer,
      metrics,
      status,
      controls,
      checks: {
        noHorizontalOverflow: document.body.scrollWidth <= window.innerWidth
          && document.documentElement.scrollWidth <= window.innerWidth
          && (!hud || hud.scrollWidth <= hud.clientWidth),
        queryBeforeAnswer: before(queryRow, answer),
        answerBeforeMetrics: before(answer, metrics),
        statusBeforeControls: before(status, controls),
        answerNoOwnOverflow: !answer
          || (answer.scrollWidth <= answer.clientWidth && answer.scrollHeight <= answer.clientHeight),
      },
    };
  });
}

async function sampleFps(page, minimum) {
  const samples = [];
  for (let index = 0; index < 5; index += 1) {
    await page.waitForTimeout(1_050);
    samples.push(Number(await page.evaluate(() => window.__visualPhysicsDemoState.fps)));
  }
  const average = samples.reduce((sum, value) => sum + value, 0) / samples.length;
  return {
    samples,
    average: rounded(average, 2),
    minimum: Math.min(...samples),
    threshold: minimum,
    passed: samples.every(Number.isFinite) && average >= minimum,
  };
}

function inspectRuntimeContract(initialState, collisionWorld, inspections, expected, failures) {
  record(initialState.visualReady === true, "static visual is not ready", failures);
  record(initialState.colliderReady === true, "static TSDF is not ready", failures);
  record(
    initialState.cameraOverviewFocusObjectId === expected.cameraFocusObjectId,
    `camera overview focus=${initialState.cameraOverviewFocusObjectId}`,
    failures,
  );
  record(
    Array.isArray(initialState.cameraInsideInteractiveObjectIds)
      && initialState.cameraInsideInteractiveObjectIds.length === 0,
    `camera inside interactive objects=${JSON.stringify(initialState.cameraInsideInteractiveObjectIds)}`,
    failures,
  );
  const overviewCoverage = initialState.cameraOverviewCoverage;
  const widthFraction = overviewCoverage?.widthFraction;
  const heightFraction = overviewCoverage?.heightFraction;
  record(
    Number.isFinite(widthFraction)
      && widthFraction >= CAMERA_OVERVIEW_COVERAGE_MIN
      && widthFraction <= CAMERA_OVERVIEW_COVERAGE_MAX,
    `camera overview widthFraction=${widthFraction} outside [${CAMERA_OVERVIEW_COVERAGE_MIN}, ${CAMERA_OVERVIEW_COVERAGE_MAX}]`,
    failures,
  );
  record(
    Number.isFinite(heightFraction)
      && heightFraction >= CAMERA_OVERVIEW_COVERAGE_MIN
      && heightFraction <= CAMERA_OVERVIEW_COVERAGE_MAX,
    `camera overview heightFraction=${heightFraction} outside [${CAMERA_OVERVIEW_COVERAGE_MIN}, ${CAMERA_OVERVIEW_COVERAGE_MAX}]`,
    failures,
  );
  record(initialState.staticVisualCount === expected.staticVisualCount,
    `static visual count=${initialState.staticVisualCount}`, failures);
  record(initialState.colliderFaces === expected.staticColliderFaces,
    `static TSDF faces=${initialState.colliderFaces}`, failures);
  record(initialState.interactiveObjectCount === expected.objectCount,
    `interactive object count=${initialState.interactiveObjectCount}`, failures);
  record(initialState.interactiveObjectReadyCount === expected.objectCount,
    `ready object count=${initialState.interactiveObjectReadyCount}`, failures);
  record(initialState.objectColliderCount === expected.objectColliderCount,
    `object collider count=${initialState.objectColliderCount}`, failures);
  record(collisionWorld.objectColliderReadyCount === expected.objectColliderCount,
    `ready collider count=${collisionWorld.objectColliderReadyCount}`, failures);
  record(collisionWorld.objectColliderFaces === expected.objectColliderFaces,
    `object collider faces=${collisionWorld.objectColliderFaces}`, failures);
  record(initialState.interactiveObjectMeshVertexCount === expected.meshVertices,
    `mesh vertices=${initialState.interactiveObjectMeshVertexCount}`, failures);
  record(initialState.interactiveObjectGaussianCount === expected.gaussianVertices,
    `Gaussian vertices=${initialState.interactiveObjectGaussianCount}`, failures);
  record(initialState.interactiveObjectRgbPointCount === expected.rgbPointVertices,
    `RGB point vertices=${initialState.interactiveObjectRgbPointCount}`, failures);
  record(initialState.visualCount === expected.totalVisualPrimitives,
    `total visual primitives=${initialState.visualCount}`, failures);
  record(collisionWorld.sceneReady === true, "collision world scene is not ready", failures);
  record(collisionWorld.degradedCount === 0,
    `degraded collider count=${collisionWorld.degradedCount}`, failures);
  record(collisionWorld.errors.length === 0,
    `collider errors=${JSON.stringify(collisionWorld.errors)}`, failures);
  record(collisionWorld.objects.length === expected.objectCount,
    `collision world objects=${collisionWorld.objects.length}`, failures);
  record(initialState.interactiveObjects?.length === expected.objectCount,
    `runtime object snapshots=${initialState.interactiveObjects?.length}`, failures);
  for (const objectId of expected.objectIds) {
    const snapshot = initialState.interactiveObjects?.find((item) => item.id === objectId);
    record(snapshot?.ready === true, `${objectId}: runtime object not ready`, failures);
    record(snapshot?.visualReady === true, `${objectId}: runtime visual not ready`, failures);
    record(snapshot?.colliderReady === true, `${objectId}: runtime collider not ready`, failures);
    record(snapshot?.collisionMode !== "degraded-box",
      `${objectId}: degraded collision mode`, failures);
    if (UNIFIED_IDS.includes(objectId)) {
      const contract = expected.unified[objectId];
      record(snapshot?.parentObjectId === contract.parentObjectId,
        `${objectId}: runtime parent=${snapshot?.parentObjectId}`, failures);
      record(snapshot?.movesWithParent === contract.movesWithParent,
        `${objectId}: runtime movesWithParent=${snapshot?.movesWithParent}`, failures);
      record(snapshot?.independentlyMovable === true,
        `${objectId}: runtime independentlyMovable=${snapshot?.independentlyMovable}`, failures);
      if (objectId === BED_ID) {
        record(
          Array.isArray(snapshot?.childObjectIds)
            && snapshot.childObjectIds.length === PILLOW_IDS.size
            && [...PILLOW_IDS].every((id) => snapshot.childObjectIds.includes(id)),
          `${objectId}: runtime children=${JSON.stringify(snapshot?.childObjectIds)}`,
          failures,
        );
      }
    }
  }
  for (const item of collisionWorld.objects) {
    record(item.ready === true, `${item.id}: collider not ready`, failures);
  }

  for (const objectId of UNIFIED_IDS) {
    const inspection = inspections[objectId];
    const contract = expected.unified[objectId];
    const collision = collisionWorld.objects.find((item) => item.id === objectId);
    record(inspection?.visualReady === true, `${objectId}: visual not ready`, failures);
    record(inspection?.colliderReady === true, `${objectId}: collider not ready`, failures);
    record(inspection?.visualKind === "mesh", `${objectId}: visualKind=${inspection?.visualKind}`, failures);
    record(inspection?.collisionMode === "unified-glb",
      `${objectId}: collisionMode=${inspection?.collisionMode}`, failures);
    record(inspection?.collisionTopology === contract.topology,
      `${objectId}: topology=${inspection?.collisionTopology}`, failures);
    record(inspection?.unifiedVisualCollision === true,
      `${objectId}: visual/collision roots differ`, failures);
    record(inspection?.collisionFaces === contract.faces,
      `${objectId}: runtime faces=${inspection?.collisionFaces}`, failures);
    record(inspection?.meshVertexCount === contract.vertices,
      `${objectId}: runtime vertices=${inspection?.meshVertexCount}`, failures);
    record(inspection?.pbrMaterialCount > 0,
      `${objectId}: PBR material count=${inspection?.pbrMaterialCount}`, failures);
    record(inspection?.bvhMeshCount > 0,
      `${objectId}: BVH mesh count=${inspection?.bvhMeshCount}`, failures);
    record(inspection?.proxyWorldBounds == null, `${objectId}: proxy bounds are not null`, failures);
    record(inspection?.proxyMatrixWorld == null, `${objectId}: proxy matrix is not null`, failures);
    record(collision?.faces === contract.faces,
      `${objectId}: collision-world faces=${collision?.faces}`, failures);
    record(collision?.topology === contract.topology,
      `${objectId}: collision-world topology=${collision?.topology}`, failures);
  }
}

async function findVisibleObjectPoint(page, objectId) {
  const seed = await page.evaluate((id) => window.__getInteractiveObjectScreenPoint(id), objectId);
  requireCondition(
    seed && Number.isFinite(seed.x) && Number.isFinite(seed.y),
    `${objectId}: no projected screen point`,
  );
  return page.evaluate(({ id, seedPoint }) => {
    const canvas = document.querySelector("canvas");
    const rect = canvas?.getBoundingClientRect();
    if (!rect) return null;
    let samples = 0;
    const inspect = (x, y) => {
      if (x < rect.left || x > rect.right || y < rect.top || y > rect.bottom) return null;
      if (document.elementFromPoint(x, y) !== canvas) return null;
      samples += 1;
      const hit = window.__inspectInteractiveObjectHitAt(x, y);
      return hit?.objectId === id ? { x, y, hit, samples, seed: seedPoint } : null;
    };
    for (let radius = 0; radius <= 240; radius += 12) {
      const count = radius === 0 ? 1 : Math.max(8, Math.ceil((Math.PI * 2 * radius) / 12));
      for (let index = 0; index < count; index += 1) {
        const angle = count === 1 ? 0 : (Math.PI * 2 * index) / count;
        const result = inspect(
          seedPoint.x + Math.cos(angle) * radius,
          seedPoint.y + Math.sin(angle) * radius,
        );
        if (result) return result;
      }
    }
    const gridStep = 16;
    const grid = [];
    for (let y = rect.top + gridStep / 2; y < rect.bottom; y += gridStep) {
      for (let x = rect.left + gridStep / 2; x < rect.right; x += gridStep) {
        grid.push({ x, y, distance: Math.hypot(x - seedPoint.x, y - seedPoint.y) });
      }
    }
    grid.sort((left, right) => left.distance - right.distance);
    for (const point of grid) {
      const result = inspect(point.x, point.y);
      if (result) return { ...result, search: "full-canvas-grid" };
    }
    return { x: seedPoint.x, y: seedPoint.y, hit: null, samples, seed: seedPoint };
  }, { id: objectId, seedPoint: seed });
}

async function testUnifiedInteraction(page, objectId, failures) {
  const focusBefore = await page.evaluate(() => {
    window.__setCameraPreset("reference");
    return window.__visualPhysicsDemoState;
  });
  const focusResult = await page.evaluate((id) => window.__focusSceneEntity(id), objectId);
  await page.waitForTimeout(700);
  const focusAfter = await page.evaluate((id) => ({
    state: window.__visualPhysicsDemoState,
    inspection: window.__inspectInteractiveObject(id),
  }), objectId);
  const focusCameraDelta = Math.max(
    ...["x", "y", "z"].map((axis) => Math.abs(
      Number(focusBefore.cameraPosition?.[axis]) - Number(focusAfter.state.cameraPosition?.[axis]),
    )),
    ...["x", "y", "z"].map((axis) => Math.abs(
      Number(focusBefore.cameraTarget?.[axis]) - Number(focusAfter.state.cameraTarget?.[axis]),
    )),
  );
  record(focusResult?.focused === true, `${objectId}: focus returned false`, failures);
  record(focusAfter.state.selectedSceneEntity === objectId,
    `${objectId}: focus selectedSceneEntity=${focusAfter.state.selectedSceneEntity}`, failures);
  record(focusAfter.state.selectedInteractiveObject === objectId,
    `${objectId}: focus selectedInteractiveObject=${focusAfter.state.selectedInteractiveObject}`, failures);
  record(focusAfter.inspection?.selectionOutlineVisible === true,
    `${objectId}: focus selection outline is not visible`, failures);
  const focusTargetEvidence = cameraTargetBoundsEvidence(focusAfter.state, focusAfter.inspection);
  record(focusTargetEvidence.passed,
    `${objectId}: focus camera target does not frame the object`, failures);
  const visiblePoint = await findVisibleObjectPoint(page, objectId);
  const point = { x: visiblePoint.x, y: visiblePoint.y };
  const hit = visiblePoint.hit;
  record(hit?.objectId === objectId, `${objectId}: pointer hit ${hit?.objectId}`, failures);
  record(hit?.colliderMode === "unified-glb", `${objectId}: pointer mode=${hit?.colliderMode}`, failures);

  await page.evaluate((id) => window.__setInteractiveObjectYaw(id, 0), objectId);
  const before = await page.evaluate((id) => window.__inspectInteractiveObject(id), objectId);
  await page.mouse.move(point.x, point.y);
  await page.mouse.down();
  await page.mouse.move(point.x + 84, point.y, { steps: 10 });
  await page.mouse.up();
  const afterDrag = await page.evaluate((id) => ({
    inspection: window.__inspectInteractiveObject(id),
    activeDrag: window.__visualPhysicsDemoState.activeObjectDrag,
  }), objectId);
  const dragDeltas = {
    group: matrixMaximumDelta(before.groupMatrixWorld, afterDrag.inspection.groupMatrixWorld),
    visual: matrixMaximumDelta(before.splatMatrixWorld, afterDrag.inspection.splatMatrixWorld),
    collision: matrixMaximumDelta(before.collisionMatrixWorld, afterDrag.inspection.collisionMatrixWorld),
  };
  record(dragDeltas.group > 1e-4, `${objectId}: drag did not move group`, failures);
  record(dragDeltas.visual > 1e-4, `${objectId}: drag did not move visual`, failures);
  record(dragDeltas.collision > 1e-4, `${objectId}: drag did not move collision`, failures);
  record(afterDrag.activeDrag == null, `${objectId}: drag stayed active`, failures);

  await page.evaluate((id) => window.__setInteractiveObjectYaw(id, 0), objectId);
  const dragReset = await page.evaluate((id) => window.__inspectInteractiveObject(id), objectId);
  const dragReturned = {
    group: matricesEqual(before.groupMatrixWorld, dragReset.groupMatrixWorld),
    visual: matricesEqual(before.splatMatrixWorld, dragReset.splatMatrixWorld),
    collision: matricesEqual(before.collisionMatrixWorld, dragReset.collisionMatrixWorld),
  };
  record(Object.values(dragReturned).every(Boolean), `${objectId}: drag reset failed`, failures);

  const visibleSpinPoint = await findVisibleObjectPoint(page, objectId);
  const spinPoint = { x: visibleSpinPoint.x, y: visibleSpinPoint.y };
  record(visibleSpinPoint.hit?.objectId === objectId,
    `${objectId}: spin pointer hit ${visibleSpinPoint.hit?.objectId}`, failures);
  const spinBefore = await page.evaluate((id) => ({
    inspection: window.__inspectInteractiveObject(id),
    turns: window.__visualPhysicsDemoState.interactiveObjectTurns,
  }), objectId);
  await page.mouse.dblclick(spinPoint.x, spinPoint.y, { delay: 45 });
  await page.waitForFunction((id) => window.__visualPhysicsDemoState.interactiveObjects
    .find((item) => item.id === id)?.spinning === true, objectId, { timeout: 8_000 });
  await page.waitForTimeout(260);
  const duringSpin = await page.evaluate((id) => window.__inspectInteractiveObject(id), objectId);
  await page.waitForFunction(({ id, turns }) => (
    window.__visualPhysicsDemoState.interactiveObjectTurns > turns
    && window.__visualPhysicsDemoState.interactiveObjects.find((item) => item.id === id)?.spinning === false
  ), { id: objectId, turns: spinBefore.turns }, { timeout: 10_000 });
  const spinAfter = await page.evaluate((id) => ({
    inspection: window.__inspectInteractiveObject(id),
    turns: window.__visualPhysicsDemoState.interactiveObjectTurns,
  }), objectId);
  const inFlightDeltas = {
    visual: matrixMaximumDelta(spinBefore.inspection.splatMatrixWorld, duringSpin.splatMatrixWorld),
    collision: matrixMaximumDelta(
      spinBefore.inspection.collisionMatrixWorld,
      duringSpin.collisionMatrixWorld,
    ),
  };
  const spinReturned = {
    group: matricesEqual(spinBefore.inspection.groupMatrixWorld, spinAfter.inspection.groupMatrixWorld),
    visual: matricesEqual(spinBefore.inspection.splatMatrixWorld, spinAfter.inspection.splatMatrixWorld),
    collision: matricesEqual(
      spinBefore.inspection.collisionMatrixWorld,
      spinAfter.inspection.collisionMatrixWorld,
    ),
  };
  record(inFlightDeltas.visual > 1e-4, `${objectId}: spin did not move visual`, failures);
  record(inFlightDeltas.collision > 1e-4, `${objectId}: spin did not move collision`, failures);
  record(spinAfter.turns > spinBefore.turns, `${objectId}: turn counter did not increment`, failures);
  record(Object.values(spinReturned).every(Boolean), `${objectId}: spin did not return`, failures);
  return {
    objectId,
    focus: {
      focused: focusResult?.focused === true,
      selectedSceneEntity: focusAfter.state.selectedSceneEntity,
      selectedInteractiveObject: focusAfter.state.selectedInteractiveObject,
      selectionOutlineVisible: focusAfter.inspection?.selectionOutlineVisible === true,
      cameraMaximumDelta: rounded(focusCameraDelta, 6),
      cameraTargetBoundsEvidence: focusTargetEvidence,
    },
    pointer: { point, hit, visibleSurfaceSearchSamples: visiblePoint.samples, seed: visiblePoint.seed },
    drag: { deltas: dragDeltas, returned: dragReturned },
    doubleClick360: {
      turnsBefore: spinBefore.turns,
      turnsAfter: spinAfter.turns,
      inFlightDeltas,
      returned: spinReturned,
    },
  };
}

async function testBedPillowHierarchy(page, failures) {
  await page.evaluate((ids) => {
    ids.forEach((id) => window.__setInteractiveObjectYaw(id, 0));
  }, UNIFIED_IDS);
  const inspect = () => page.evaluate((ids) => Object.fromEntries(
    ids.map((id) => [id, window.__inspectInteractiveObject(id)]),
  ), UNIFIED_IDS);
  const beforeParent = await inspect();
  const attachments = {};
  const initialMotionRootGramDelta = {};
  for (const pillowId of PILLOW_IDS) {
    attachments[pillowId] = beforeParent[pillowId].hierarchyAttachmentMatrixDelta;
    record(
      attachments[pillowId] <= MATRIX_EPSILON,
      `${pillowId}: hierarchy attachment changed the world transform`,
      failures,
    );
    const groupGram = matrixGram(beforeParent[pillowId].groupMatrixWorld);
    initialMotionRootGramDelta[pillowId] = matrixMaximumDelta(
      groupGram,
      [1, 1, 1, 0, 0, 0],
    );
    record(
      initialMotionRootGramDelta[pillowId] <= MATRIX_EPSILON,
      `${pillowId}: motion root is not rigid before interaction`,
      failures,
    );
  }
  const parentUpdate = await page.evaluate((id) => window.__setInteractiveObjectYaw(id, 18), BED_ID);
  const afterParent = await inspect();
  const parentMotion = Object.fromEntries(UNIFIED_IDS.map((id) => [
    id,
    transformMotionAndRigidity(beforeParent[id], afterParent[id]),
  ]));
  record(parentUpdate?.updated === true, "bed hierarchy: parent yaw update failed", failures);
  record(parentMotion[BED_ID].groupMatrixWorld.matrixDelta > 1e-4,
    "bed hierarchy: bed world transform did not move", failures);
  record(parentMotion[BED_ID].groupMatrixWorld.translationDelta <= MATRIX_EPSILON,
    "bed hierarchy: bed pivot drifted during yaw", failures);
  record(
    [
      parentMotion[BED_ID].groupMatrixWorld,
      parentMotion[BED_ID].splatMatrixWorld,
      parentMotion[BED_ID].collisionMatrixWorld,
    ].every((item) => item.gramDelta <= MATRIX_GRAM_EPSILON),
    "bed hierarchy: bed shape changed during parent motion",
    failures,
  );
  for (const pillowId of PILLOW_IDS) {
    const motion = parentMotion[pillowId];
    record(motion.groupMatrixWorld.translationDelta > 1e-4,
      `bed hierarchy: ${pillowId} pivot did not move with bed`, failures);
    record(motion.splatMatrixWorld.matrixDelta > 1e-4,
      `bed hierarchy: ${pillowId} visual did not move with bed`, failures);
    record(motion.collisionMatrixWorld.matrixDelta > 1e-4,
      `bed hierarchy: ${pillowId} collision did not move with bed`, failures);
    record(
      [motion.groupMatrixWorld, motion.splatMatrixWorld, motion.collisionMatrixWorld]
        .every((item) => item.gramDelta <= MATRIX_GRAM_EPSILON),
      `bed hierarchy: ${pillowId} shape changed during parent motion`,
      failures,
    );
    const relativeBefore = relativeMatrix(
      beforeParent[BED_ID].groupMatrixWorld,
      beforeParent[pillowId].groupMatrixWorld,
    );
    const relativeAfter = relativeMatrix(
      afterParent[BED_ID].groupMatrixWorld,
      afterParent[pillowId].groupMatrixWorld,
    );
    record(matrixMaximumDelta(relativeBefore, relativeAfter) <= MATRIX_EPSILON,
      `bed hierarchy: ${pillowId} relative parent transform drifted`, failures);
    motion.relativeParentMatrixDelta = matrixMaximumDelta(relativeBefore, relativeAfter);
  }
  await page.evaluate((id) => window.__setInteractiveObjectYaw(id, 0), BED_ID);
  const parentReset = await inspect();
  const parentReturned = Object.fromEntries(UNIFIED_IDS.map((id) => [id, {
    group: matricesEqual(beforeParent[id].groupMatrixWorld, parentReset[id].groupMatrixWorld),
    visual: matricesEqual(beforeParent[id].splatMatrixWorld, parentReset[id].splatMatrixWorld),
    collision: matricesEqual(
      beforeParent[id].collisionMatrixWorld,
      parentReset[id].collisionMatrixWorld,
    ),
  }]));
  record(Object.values(parentReturned).every((item) => Object.values(item).every(Boolean)),
    "bed hierarchy: parent reset failed", failures);

  const independentChildren = [];
  for (const pillowId of PILLOW_IDS) {
    const beforeChild = await inspect();
    const childUpdate = await page.evaluate((id) => window.__setInteractiveObjectYaw(id, 14), pillowId);
    const afterChild = await inspect();
    const pillowMotion = transformMotionAndRigidity(beforeChild[pillowId], afterChild[pillowId]);
    const bedMotion = transformMotionAndRigidity(beforeChild[BED_ID], afterChild[BED_ID]);
    record(childUpdate?.updated === true, `${pillowId}: independent child yaw update failed`, failures);
    record(pillowMotion.groupMatrixWorld.matrixDelta > 1e-4,
      `${pillowId}: independent child transform did not move`, failures);
    record(pillowMotion.groupMatrixWorld.translationDelta <= MATRIX_EPSILON,
      `${pillowId}: independent child pivot drifted`, failures);
    record(pillowMotion.splatMatrixWorld.matrixDelta > 1e-4,
      `${pillowId}: independent child visual did not move`, failures);
    record(pillowMotion.collisionMatrixWorld.matrixDelta > 1e-4,
      `${pillowId}: independent child collision did not move`, failures);
    record(
      [pillowMotion.groupMatrixWorld, pillowMotion.splatMatrixWorld, pillowMotion.collisionMatrixWorld]
        .every((item) => item.gramDelta <= MATRIX_GRAM_EPSILON),
      `${pillowId}: independent child yaw changed shape`,
      failures,
    );
    record(
      [bedMotion.groupMatrixWorld, bedMotion.splatMatrixWorld, bedMotion.collisionMatrixWorld]
        .every((item) => item.matrixDelta <= MATRIX_EPSILON),
      `${pillowId}: child motion moved the bed`,
      failures,
    );
    await page.evaluate((id) => window.__setInteractiveObjectYaw(id, 0), pillowId);
    const reset = await inspect();
    const returned = {
      group: matricesEqual(beforeChild[pillowId].groupMatrixWorld, reset[pillowId].groupMatrixWorld),
      visual: matricesEqual(beforeChild[pillowId].splatMatrixWorld, reset[pillowId].splatMatrixWorld),
      collision: matricesEqual(
        beforeChild[pillowId].collisionMatrixWorld,
        reset[pillowId].collisionMatrixWorld,
      ),
    };
    record(Object.values(returned).every(Boolean), `${pillowId}: independent child reset failed`, failures);
    independentChildren.push({ objectId: pillowId, pillowMotion, bedMotion, returned });
  }
  return {
    attachments,
    initialMotionRootGramDelta,
    parentMotion,
    parentReturned,
    independentChildren,
  };
}

async function testRobotCollision(page, objectId, failures) {
  const prepared = await page.evaluate((id) => window.__prepareRobotCollisionTest(id), objectId);
  let step = null;
  if (prepared.prepared) {
    for (let attempt = 0; attempt < 5; attempt += 1) {
      step = await page.evaluate((id) => window.__stepRobotTowardObject(id, 0.65), objectId);
      if (step.blocked && step.lastCollisionObjectId === objectId) break;
    }
  }
  const passed = Boolean(prepared.prepared && step?.blocked && step.lastCollisionObjectId === objectId);
  record(passed, `${objectId}: robot collision probe failed`, failures);
  return {
    objectId,
    prepared: {
      prepared: prepared.prepared,
      reason: prepared.reason || null,
      supportSource: prepared.supportSource || null,
      surfacePoint: prepared.surfacePoint || null,
      supportPoint: prepared.supportPoint || null,
      robotPosition: prepared.robotPosition || null,
    },
    step: step ? {
      attempted: step.attempted,
      moved: step.moved,
      blocked: step.blocked,
      lastCollisionKind: step.lastCollisionKind,
      lastCollisionObjectId: step.lastCollisionObjectId,
      before: step.before,
      after: step.after,
      requestedDistance: step.requestedDistance,
    } : null,
    passed,
  };
}

async function testSceneQuery(page, queryCase, failures) {
  const before = await page.evaluate(() => {
    window.__setCameraPreset("reference");
    return {
      cameraPosition: window.__visualPhysicsDemoState.cameraPosition,
      cameraTarget: window.__visualPhysicsDemoState.cameraTarget,
    };
  });
  const answer = await page.evaluate((question) => window.__askScene(question), queryCase.question);
  await page.waitForTimeout(700);
  const after = await page.evaluate((id) => ({
    state: window.__visualPhysicsDemoState,
    inspection: window.__inspectInteractiveObject(id),
  }), queryCase.expectedId);
  record(answer?.status === "resolved", `${queryCase.label}: status=${answer?.status}`, failures);
  record(answer?.entityId === queryCase.expectedId,
    `${queryCase.label}: entity=${answer?.entityId}`, failures);
  record(answer?.focusEntityId === queryCase.expectedId,
    `${queryCase.label}: focus=${answer?.focusEntityId}`, failures);
  record(after.state.selectedSceneEntity === queryCase.expectedId,
    `${queryCase.label}: selectedSceneEntity=${after.state.selectedSceneEntity}`, failures);
  record(after.state.selectedInteractiveObject === queryCase.expectedId,
    `${queryCase.label}: selectedInteractiveObject=${after.state.selectedInteractiveObject}`, failures);
  record(after.inspection?.collisionBounds != null, `${queryCase.label}: bbox evidence missing`, failures);
  record(after.inspection?.selectionOutlineVisible === true,
    `${queryCase.label}: selection bbox is not visible`, failures);
  record(typeof answer?.answer === "string" && answer.answer.trim().length > 0,
    `${queryCase.label}: answer is empty`, failures);
  const cameraMaximumDelta = Math.max(
    ...["x", "y", "z"].map((axis) => Math.abs(
      Number(before.cameraPosition?.[axis]) - Number(after.state.cameraPosition?.[axis]),
    )),
    ...["x", "y", "z"].map((axis) => Math.abs(
      Number(before.cameraTarget?.[axis]) - Number(after.state.cameraTarget?.[axis]),
    )),
  );
  const cameraTargetEvidence = cameraTargetBoundsEvidence(after.state, after.inspection);
  record(cameraTargetEvidence.passed,
    `${queryCase.label}: query camera target does not frame the object`, failures);
  return {
    ...queryCase,
    answer,
    beforeCamera: before,
    afterCamera: {
      cameraPosition: after.state.cameraPosition,
      cameraTarget: after.state.cameraTarget,
    },
    focusAndBbox: {
      selectedSceneEntity: after.state.selectedSceneEntity,
      selectedInteractiveObject: after.state.selectedInteractiveObject,
      collisionBounds: after.inspection?.collisionBounds || null,
      selectionOutlineVisible: after.inspection?.selectionOutlineVisible === true,
      cameraMaximumDelta: rounded(cameraMaximumDelta, 6),
      cameraTargetBoundsEvidence: cameraTargetEvidence,
    },
  };
}

function boundsOverlapMetrics(left, right) {
  const axes = ["x", "y", "z"];
  const overlap = axes.map((axis) => (
    Math.max(0, Math.min(left.max[axis], right.max[axis]) - Math.max(left.min[axis], right.min[axis]))
  ));
  const leftSize = axes.map((axis) => Math.max(0, left.max[axis] - left.min[axis]));
  const rightSize = axes.map((axis) => Math.max(0, right.max[axis] - right.min[axis]));
  const overlapVolume = overlap.reduce((product, value) => product * value, 1);
  const smallerVolume = Math.min(
    leftSize.reduce((product, value) => product * value, 1),
    rightSize.reduce((product, value) => product * value, 1),
  );
  const normalizedDepths = overlap.map((value, index) => (
    value / Math.max(1e-8, Math.min(leftSize[index], rightSize[index]))
  ));
  return {
    overlap: { x: overlap[0], y: overlap[1], z: overlap[2] },
    overlapVolume: rounded(overlapVolume, 8),
    overlapFractionOfSmallerAabb: rounded(overlapVolume / Math.max(1e-8, smallerVolume), 6),
    minimumNormalizedAxisDepth: rounded(Math.min(...normalizedDepths), 6),
  };
}

export function classifyInterpenetrations(rawReport, nestedSupport = null) {
  const classifications = (rawReport?.intersections || []).map((intersection) => {
    const ids = intersection.colliderIds || [];
    const objectIds = ids.filter((id) => id !== "scene");
    const involvesScene = ids.includes("scene");
    const involvesUnified = objectIds.some((id) => UNIFIED_IDS.includes(id));
    const knownBedPillowSupport = ids.includes(BED_ID) && objectIds.some((id) => PILLOW_IDS.has(id));
    const validatedBedPillowSupport = knownBedPillowSupport
      && nestedSupport?.receiptSha256 === APPROVED_NESTED_SUPPORT.receiptSha256
      && nestedSupport?.parentObjectId === BED_ID
      && Array.isArray(nestedSupport?.childObjectIds)
      && objectIds.every((id) => id === BED_ID || nestedSupport.childObjectIds.includes(id));
    const overlap = boundsOverlapMetrics(intersection.leftBounds, intersection.rightBounds);
    const obvious = overlap.overlapFractionOfSmallerAabb >= 0.08
      && overlap.minimumNormalizedAxisDepth >= 0.15;
    let classification = "legacy_pair_record_only";
    let blocking = false;
    if (involvesScene) {
      classification = "scene_support_or_surface_contact_record_only";
    } else if (validatedBedPillowSupport) {
      classification = "validated_bed_pillow_support_contact_record_only";
    } else if (knownBedPillowSupport && !obvious) {
      classification = "bed_pillow_support_contact_record_only";
    } else if (involvesUnified) {
      blocking = obvious;
      classification = blocking
        ? "obvious_unified_object_interpenetration"
        : "minor_or_conservative_unified_object_contact";
    }
    return {
      ...intersection,
      overlap,
      classification,
      blocking,
      nestedSupportReceiptSha256: validatedBedPillowSupport ? nestedSupport.receiptSha256 : null,
    };
  });
  return {
    ...rawReport,
    automatedStatus: classifications.some((item) => item.blocking)
      ? "failed_obvious_interpenetration"
      : "passed_with_recorded_contacts",
    policy: {
      currentUserTolerance: "minor_contact_and_nonblocking_hallucination_allowed",
      sceneContacts: "record_only_for_support_or_human_visual_review",
      bedPillowContacts: "shallow_contact_record_only_but_obvious_interpenetration_blocks",
      validatedNestedSupport: nestedSupport?.receiptSha256 === APPROVED_NESTED_SUPPORT.receiptSha256
        ? {
          receiptSha256: nestedSupport.receiptSha256,
          maximumPenetrationWorld: nestedSupport.maximumPenetrationWorld,
          maximumSupportGapWorld: nestedSupport.maximumSupportGapWorld,
          disposition: "record_only_for_exact_receipt_bound_bed_pillow_pairs",
        }
        : null,
      obviousObjectThreshold: {
        overlapFractionOfSmallerAabb: 0.08,
        minimumNormalizedAxisDepth: 0.15,
      },
      humanScreenshotReviewStillRequired: true,
    },
    classifications,
  };
}

async function inspectViewport(page, expected, failures) {
  const initialState = await page.evaluate(() => window.__visualPhysicsDemoState);
  const collisionWorld = await page.evaluate(() => window.__inspectCollisionWorld());
  const inspections = await page.evaluate((ids) => Object.fromEntries(
    ids.map((id) => [id, window.__inspectInteractiveObject(id)]),
  ), UNIFIED_IDS);
  inspectRuntimeContract(initialState, collisionWorld, inspections, expected, failures);
  return { initialState, collisionWorld, inspections };
}

function validateDiagnostics(diagnostics, manifestSha256, expected, runtimeUrl, failures) {
  record(diagnostics.manifestRequestCount === 1,
    `${diagnostics.label}: manifest requests=${diagnostics.manifestRequestCount}`, failures);
  record(diagnostics.manifestResponseSha256.length === 1,
    `${diagnostics.label}: manifest responses=${diagnostics.manifestResponseSha256.length}`, failures);
  record(diagnostics.manifestResponseSha256[0] === manifestSha256,
    `${diagnostics.label}: served manifest SHA mismatch`, failures);
  for (const [objectId, contract] of Object.entries(expected.unified)) {
    const pathname = new URL(contract.url, runtimeUrl).pathname;
    record(diagnostics.assetRequestCounts[pathname] === 1,
      `${diagnostics.label}: ${objectId} GLB requests=${diagnostics.assetRequestCounts[pathname]}`, failures);
  }
  record(diagnostics.pageErrors.length === 0,
    `${diagnostics.label}: page errors=${JSON.stringify(diagnostics.pageErrors)}`, failures);
  record(diagnostics.requestFailures.length === 0,
    `${diagnostics.label}: request failures=${JSON.stringify(diagnostics.requestFailures)}`, failures);
  record(diagnostics.httpErrors.length === 0,
    `${diagnostics.label}: HTTP errors=${JSON.stringify(diagnostics.httpErrors)}`, failures);
  const consoleErrors = diagnostics.consoleMessages.filter((item) => item.type === "error");
  record(consoleErrors.length === 0,
    `${diagnostics.label}: console errors=${JSON.stringify(consoleErrors)}`, failures);
}

async function runViewport({
  browser,
  label,
  viewport,
  isMobile = false,
  runtimeUrl,
  expected,
  manifestSha256,
  screenshotDir,
  minFps,
  runFullQa = false,
  failures,
}) {
  const context = await browser.newContext({
    viewport,
    deviceScaleFactor: 1,
    isMobile,
    hasTouch: isMobile,
  });
  const page = await context.newPage();
  const runtime = new URL(runtimeUrl);
  const manifestPathname = new URL(runtime.searchParams.get("manifest"), runtime).pathname;
  const assetPathnames = new Set(
    Object.values(expected.unified).map((item) => new URL(item.url, runtime).pathname),
  );
  const diagnostics = createPageDiagnostics(page, { label, manifestPathname, assetPathnames });
  let result;
  try {
    logStage(label, "navigation_started", { runtimeUrl });
    await page.goto(runtimeUrl, { waitUntil: "domcontentloaded", timeout: 60_000 });
    await waitForReady(page, expected);
    logStage(label, "runtime_ready");
    await page.waitForTimeout(1_200);
    const runtimeInspection = await inspectViewport(page, expected, failures);
    const queries = [];
    const queryCases = runFullQa
      ? expected.queryCases
      : expected.queryCases.filter((item) => ["pillow_appearance", "bed_appearance"].includes(item.label));
    for (const queryCase of queryCases) {
      queries.push(await testSceneQuery(page, queryCase, failures));
    }

    const interactions = [];
    const robotCollisions = [];
    let hierarchy = null;
    let interpenetrations = null;
    if (runFullQa) {
      for (const objectId of UNIFIED_IDS) {
        interactions.push(await testUnifiedInteraction(page, objectId, failures));
      }
      hierarchy = await testBedPillowHierarchy(page, failures);
      for (const objectId of UNIFIED_IDS) {
        robotCollisions.push(await testRobotCollision(page, objectId, failures));
      }
      interpenetrations = classifyInterpenetrations(
        await page.evaluate(() => window.__inspectSceneInterpenetrations()),
        expected.nestedSupport,
      );
      record(
        !interpenetrations.classifications.some((item) => item.blocking),
        "obvious scene interpenetration detected",
        failures,
      );
      logStage(label, "full_interaction_checks_complete", {
        interactionCount: interactions.length,
        robotCollisionCount: robotCollisions.length,
      });
    }

    const canvas = await inspectCanvasPixels(page);
    const layout = await inspectLayout(page);
    const performance = await sampleFps(page, minFps);
    layout.checks.exactViewport = layout.viewport[0] === viewport.width
      && layout.viewport[1] === viewport.height;
    layout.checks.exactCanvasCssSize = Math.abs((layout.canvas?.width ?? -1) - viewport.width) <= 0.5
      && Math.abs((layout.canvas?.height ?? -1) - viewport.height) <= 0.5;
    record(canvas.nonblank, `${label}: canvas is blank`, failures);
    record(Object.values(layout.checks).every(Boolean),
      `${label}: layout checks=${JSON.stringify(layout.checks)}`, failures);
    record(performance.passed,
      `${label}: average FPS ${performance.average} < ${performance.threshold}`, failures);
    const screenshotPath = path.join(screenshotDir, `${label}-${viewport.width}x${viewport.height}.png`);
    await page.screenshot({ path: screenshotPath, fullPage: true });
    const screenshot = pngDescriptor(screenshotPath, viewport.width, viewport.height);
    logStage(label, "screenshot_complete", { screenshot: portablePath(screenshotPath) });
    await diagnostics.settle();
    logStage(label, "diagnostics_settled", {
      pageErrorCount: diagnostics.pageErrors.length,
      requestFailureCount: diagnostics.requestFailures.length,
    });
    validateDiagnostics(diagnostics, manifestSha256, expected, runtimeUrl, failures);
    result = {
      label,
      viewport: [viewport.width, viewport.height],
      freshPageLoad: true,
      runtime: runtimeInspection,
      queries,
      interactions,
      hierarchy,
      robotCollisions,
      supportObservations: robotCollisions.map((item) => ({
        objectId: item.objectId,
        prepared: item.prepared.prepared,
        supportSource: item.prepared.supportSource,
        supportPoint: item.prepared.supportPoint,
      })),
      interpenetrations,
      canvas,
      layout,
      performance,
      diagnostics: { ...diagnostics, settle: undefined },
      screenshot,
    };
  } finally {
    await context.close();
  }
  return result;
}

export async function runQa({ runtimeUrl, manifestPath, reportPath, screenshotDir, minFps }) {
  const startedAt = new Date().toISOString();
  const startedMs = Date.now();
  const failures = [];
  const manifestSha256Before = sha256File(manifestPath);
  const implementation = implementationDescriptors();
  fs.mkdirSync(screenshotDir, { recursive: true });
  let browser = null;
  let desktop = null;
  let mobile = null;
  let report;
  try {
    const manifest = readJson(manifestPath);
    const expected = deriveManifestExpectations(manifest);
    browser = await chromium.launch({ channel: "chrome", headless: true });
    desktop = await runViewport({
      browser,
      label: "desktop",
      viewport: { width: 1440, height: 900 },
      runtimeUrl,
      expected,
      manifestSha256: manifestSha256Before,
      screenshotDir,
      minFps,
      runFullQa: true,
      failures,
    });
    writeJson(checkpointPath(reportPath), portableReport({
      schemaVersion: 1,
      kind: "video2world.bedroom4_clean_unified_scene_browser_qa_checkpoint",
      phase: "desktop_complete",
      implementation,
      manifest: { path: manifestPath, sha256: manifestSha256Before },
      desktop,
      failures,
    }));
    mobile = await runViewport({
      browser,
      label: "mobile",
      viewport: { width: 390, height: 844 },
      isMobile: true,
      runtimeUrl,
      expected,
      manifestSha256: manifestSha256Before,
      screenshotDir,
      minFps,
      runFullQa: false,
      failures,
    });
    writeJson(checkpointPath(reportPath), portableReport({
      schemaVersion: 1,
      kind: "video2world.bedroom4_clean_unified_scene_browser_qa_checkpoint",
      phase: "mobile_complete",
      implementation,
      manifest: { path: manifestPath, sha256: manifestSha256Before },
      desktop,
      mobile,
      failures,
    }));
    const manifestSha256After = sha256File(manifestPath);
    record(manifestSha256After === manifestSha256Before,
      "manifest changed during browser QA", failures);
    report = {
      schemaVersion: 2,
      kind: "video2world.bedroom4_clean_unified_scene_browser_qa",
      status: failures.length ? "failed" : "passed_current_demo_only",
      automatedGate: failures.length ? "failed" : "passed",
      acceptanceScope: "current_demo_only",
      promotionApproved: false,
      publishingPerformed: false,
      manifestEdited: false,
      startedAt,
      finishedAt: new Date().toISOString(),
      durationSeconds: rounded((Date.now() - startedMs) / 1000, 2),
      implementation,
      url: runtimeUrl,
      manifest: {
        path: manifestPath,
        sha256Before: manifestSha256Before,
        sha256After: manifestSha256After,
        unchanged: manifestSha256After === manifestSha256Before,
      },
      expected,
      qualityPolicy: {
        minorLimitationsMayPass: true,
        supportAndInterpenetrationAlwaysRecorded: true,
        obviousInterpenetrationBlocks: true,
        humanScreenshotReviewRequired: true,
      },
      acceptedLimitations: Object.values(expected.unified).flatMap((item) => item.limitations),
      desktop,
      mobile,
      failures,
    };
  } catch (error) {
    report = {
      schemaVersion: 2,
      kind: "video2world.bedroom4_clean_unified_scene_browser_qa",
      status: "failed",
      automatedGate: "failed",
      acceptanceScope: "current_demo_only",
      promotionApproved: false,
      publishingPerformed: false,
      manifestEdited: false,
      startedAt,
      finishedAt: new Date().toISOString(),
      durationSeconds: rounded((Date.now() - startedMs) / 1000, 2),
      implementation,
      url: runtimeUrl,
      manifest: { path: manifestPath, sha256Before: manifestSha256Before },
      desktop,
      mobile,
      failures: [...failures, error?.stack || String(error)],
    };
  } finally {
    await browser?.close();
  }

  const portable = portableReport(report);
  writeJson(reportPath, portable);
  fs.rmSync(checkpointPath(reportPath), { force: true });
  console.log(JSON.stringify({
    status: portable.status,
    automatedGate: portable.automatedGate,
    report: portablePath(reportPath),
    failures: portable.failures,
  }, null, 2));
  if (portable.automatedGate !== "passed") process.exitCode = 1;
  return portable;
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    console.log(usage());
    return;
  }
  const manifestPath = path.resolve(options.manifest || DEFAULT_MANIFEST);
  const reportPath = path.resolve(options.report || DEFAULT_REPORT);
  const screenshotDir = path.resolve(options["screenshot-dir"] || DEFAULT_SCREENSHOT_DIR);
  const minFps = options.minFps || DEFAULT_MIN_FPS;
  requireCondition(fs.existsSync(manifestPath), `Manifest does not exist: ${manifestPath}`);
  const runtimeUrl = buildRuntimeUrl(options.url || DEFAULT_URL, manifestPath);
  await runQa({ runtimeUrl, manifestPath, reportPath, screenshotDir, minFps });
}

if (process.argv[1] && path.resolve(process.argv[1]) === scriptPath) {
  await main();
}
