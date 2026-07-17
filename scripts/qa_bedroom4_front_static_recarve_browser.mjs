#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, "..");
const publicRoot = path.join(repoRoot, "web/public");
const worldRoot = path.join(publicRoot, "worlds/bedroom4");
const reviewRoot = path.join(
  repoRoot,
  "examples/bedroom4/completion/trellis2_pillow_front_seed42/composition-review",
);
const DEFAULT_BASELINE = path.join(worldRoot, "manifest.unified-pillow.json");
const DEFAULT_CANDIDATE = path.join(
  worldRoot,
  "recarve-candidates/semantic-covariance/manifest.json",
);
const DEFAULT_REPORT = path.join(reviewRoot, "static-recarve-browser-comparison.json");
const DEFAULT_URL = "http://127.0.0.1:4177/";
const FRONT_ID = "sam3_pillow_front";
const REAR_IDS = ["sam3_pillow_left", "sam3_pillow_right"];
const SAMPLE_WIDTH = 320;
const SAMPLE_HEIGHT = 200;

function parseArgs(argv) {
  const options = {};
  const allowed = new Set(["url", "baseline", "candidate", "report", "screenshot-dir"]);
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === "--help") return { help: true };
    if (!token.startsWith("--")) throw new Error(`Unexpected argument: ${token}`);
    const key = token.slice(2);
    if (!allowed.has(key)) throw new Error(`Unsupported option: ${token}`);
    const value = argv[index + 1];
    if (!value || value.startsWith("--")) throw new Error(`Missing value for ${token}`);
    options[key] = value;
    index += 1;
  }
  return options;
}

function usage() {
  return [
    "Usage: node scripts/qa_bedroom4_front_static_recarve_browser.mjs [options]",
    "",
    `  --url URL               Runtime URL (default: ${DEFAULT_URL})`,
    `  --baseline PATH         Unmodified manifest (default: ${DEFAULT_BASELINE})`,
    `  --candidate PATH        Re-carved manifest (default: ${DEFAULT_CANDIDATE})`,
    `  --report PATH           JSON comparison report (default: ${DEFAULT_REPORT})`,
    `  --screenshot-dir PATH   Review PNG directory (default: ${reviewRoot})`,
    "",
    "This QA is candidate-only: it does not edit, promote, or publish either manifest.",
  ].join("\n");
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function sha256Bytes(bytes) {
  return crypto.createHash("sha256").update(bytes).digest("hex");
}

function sha256File(filePath) {
  return sha256Bytes(fs.readFileSync(filePath));
}

function rounded(value, digits = 4) {
  return Number(Number(value).toFixed(digits));
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
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, portableReport(item)]));
  }
  return portablePath(value);
}

function manifestPublicUrl(manifestPath) {
  const relative = path.relative(publicRoot, manifestPath);
  if (!relative || relative.startsWith("..") || path.isAbsolute(relative)) {
    throw new Error(`Manifest must be below ${publicRoot}: ${manifestPath}`);
  }
  return `/${relative.split(path.sep).join("/")}`;
}

function runtimeUrl(baseUrl, manifestPath) {
  const url = new URL(baseUrl);
  url.searchParams.set("manifest", manifestPublicUrl(manifestPath));
  return url.toString();
}

function frontDefinition(manifest) {
  const definition = manifest.interactiveObjects?.find((item) => item.id === FRONT_ID);
  if (!definition) throw new Error(`${FRONT_ID} is absent from manifest`);
  return definition;
}

function expectedCounts(manifest) {
  const primitives = manifest.candidateBuild?.primitiveCounts || {};
  const sceneAssetKey = manifest.collisionWorld?.sceneAssetKey || "collider";
  return {
    staticVisuals: manifest.assets?.visual?.vertexCount,
    totalVisuals: primitives.totalVisualPrimitives,
    visualObjects: manifest.interactiveObjects?.length,
    objectColliders: primitives.collidableObjects,
    objectColliderFaces: primitives.totalObjectColliderFaces,
    staticColliderFaces: manifest.assets?.[sceneAssetKey]?.faceCount,
  };
}

function stableJson(value) {
  if (Array.isArray(value)) return `[${value.map(stableJson).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stableJson(value[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

async function waitForReady(page) {
  await page.waitForFunction(() => {
    const state = window.__visualPhysicsDemoState;
    return Boolean(state?.error) || (state?.visualReady
      && state?.colliderReady
      && state?.sceneQaReady
      && state?.interactiveObjectReadyCount === state?.interactiveObjectCount
      && state?.objectColliderReadyCount === state?.objectColliderCount);
  }, null, { timeout: 300_000 });
  const state = await page.evaluate(() => window.__visualPhysicsDemoState);
  if (state.error) throw new Error(`Runtime load failed: ${state.error}`);
}

async function setToggle(page, selector, stateKey, wanted) {
  const current = await page.evaluate((key) => window.__visualPhysicsDemoState?.[key], stateKey);
  if (Boolean(current) !== wanted) await page.locator(selector).click({ force: true });
  await page.waitForFunction(
    ({ key, value }) => Boolean(window.__visualPhysicsDemoState?.[key]) === value,
    { key: stateKey, value: wanted },
  );
}

async function sampleCanvas(page) {
  return page.locator("#sceneCanvas").evaluate(async (canvas, size) => {
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    const copy = document.createElement("canvas");
    copy.width = size.width;
    copy.height = size.height;
    const context = copy.getContext("2d", { willReadFrequently: true });
    context.drawImage(canvas, 0, 0, size.width, size.height);
    return {
      width: size.width,
      height: size.height,
      rgba: Array.from(context.getImageData(0, 0, size.width, size.height).data),
    };
  }, { width: SAMPLE_WIDTH, height: SAMPLE_HEIGHT });
}

async function captureFrame(page, outputPath) {
  await page.waitForTimeout(900);
  await page.locator("#sceneCanvas").screenshot({ path: outputPath });
  return sampleCanvas(page);
}

function median(values) {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.floor(sorted.length / 2)];
}

function pixel(sample, index) {
  const offset = index * 4;
  return [sample.rgba[offset], sample.rgba[offset + 1], sample.rgba[offset + 2]];
}

function luma(rgb) {
  return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
}

function buildObjectMask(isolated) {
  const cornerValues = [[], [], []];
  const cornerWidth = 10;
  for (let y = 0; y < isolated.height; y += 1) {
    for (let x = 0; x < isolated.width; x += 1) {
      const corner = (x < cornerWidth || x >= isolated.width - cornerWidth)
        && (y < cornerWidth || y >= isolated.height - cornerWidth);
      if (!corner) continue;
      const rgb = pixel(isolated, y * isolated.width + x);
      for (let channel = 0; channel < 3; channel += 1) cornerValues[channel].push(rgb[channel]);
    }
  }
  const background = cornerValues.map(median);
  const raw = new Uint8Array(isolated.width * isolated.height);
  for (let index = 0; index < raw.length; index += 1) {
    const rgb = pixel(isolated, index);
    const distance = Math.max(...rgb.map((value, channel) => Math.abs(value - background[channel])));
    if (distance > 18 && luma(rgb) > 28) raw[index] = 1;
  }
  const core = new Uint8Array(raw.length);
  for (let y = 1; y < isolated.height - 1; y += 1) {
    for (let x = 1; x < isolated.width - 1; x += 1) {
      let neighbors = 0;
      for (let dy = -1; dy <= 1; dy += 1) {
        for (let dx = -1; dx <= 1; dx += 1) {
          neighbors += raw[(y + dy) * isolated.width + x + dx];
        }
      }
      if (neighbors >= 6) core[y * isolated.width + x] = 1;
    }
  }
  return { background, core, count: core.reduce((sum, value) => sum + value, 0) };
}

function frameContamination(frames) {
  const mask = buildObjectMask(frames.isolated);
  if (mask.count < 500) throw new Error(`Isolated pillow mask is unexpectedly small: ${mask.count}`);
  let absoluteDifference = 0;
  let changed20 = 0;
  let changed35 = 0;
  let changed60 = 0;
  let darkLoss35 = 0;
  let onLuma = 0;
  let offLuma = 0;
  let onChroma = 0;
  let offChroma = 0;
  let neutralOn = 0;
  let neutralOff = 0;
  const differences = [];
  for (let index = 0; index < mask.core.length; index += 1) {
    if (!mask.core[index]) continue;
    const on = pixel(frames.staticOn, index);
    const off = pixel(frames.staticOff, index);
    const channelDifference = on.map((value, channel) => Math.abs(value - off[channel]));
    const meanDifference = channelDifference.reduce((sum, value) => sum + value, 0) / 3;
    const maximumDifference = Math.max(...channelDifference);
    const currentOnLuma = luma(on);
    const currentOffLuma = luma(off);
    const currentOnChroma = Math.max(...on) - Math.min(...on);
    const currentOffChroma = Math.max(...off) - Math.min(...off);
    absoluteDifference += meanDifference;
    differences.push(meanDifference);
    if (maximumDifference > 20) changed20 += 1;
    if (maximumDifference > 35) changed35 += 1;
    if (maximumDifference > 60) changed60 += 1;
    if (currentOffLuma - currentOnLuma > 35) darkLoss35 += 1;
    if (currentOnLuma >= 75 && currentOnChroma <= 48) neutralOn += 1;
    if (currentOffLuma >= 75 && currentOffChroma <= 48) neutralOff += 1;
    onLuma += currentOnLuma;
    offLuma += currentOffLuma;
    onChroma += currentOnChroma;
    offChroma += currentOffChroma;
  }
  differences.sort((left, right) => left - right);
  const fraction = (value) => rounded(value / mask.count, 6);
  return {
    sampleSize: [frames.staticOn.width, frames.staticOn.height],
    isolatedBackgroundRgb: mask.background,
    objectCorePixelCount: mask.count,
    meanAbsoluteRgbDeltaStaticOnVsOff: rounded(absoluteDifference / mask.count, 3),
    p95MeanAbsoluteRgbDelta: rounded(differences[Math.floor(differences.length * 0.95)], 3),
    changedPixelFraction: {
      maximumChannelAbove20: fraction(changed20),
      maximumChannelAbove35: fraction(changed35),
      maximumChannelAbove60: fraction(changed60),
    },
    darkOcclusionFraction: fraction(darkLoss35),
    staticOn: {
      meanLuma: rounded(onLuma / mask.count, 3),
      meanChroma: rounded(onChroma / mask.count, 3),
      lightNeutralFraction: fraction(neutralOn),
    },
    staticOffReference: {
      meanLuma: rounded(offLuma / mask.count, 3),
      meanChroma: rounded(offChroma / mask.count, 3),
      lightNeutralFraction: fraction(neutralOff),
    },
  };
}

function dilateMask(mask, width, height, radius) {
  const output = new Uint8Array(mask.length);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let occupied = false;
      for (let dy = -radius; dy <= radius && !occupied; dy += 1) {
        const sampleY = y + dy;
        if (sampleY < 0 || sampleY >= height) continue;
        for (let dx = -radius; dx <= radius; dx += 1) {
          const sampleX = x + dx;
          if (sampleX < 0 || sampleX >= width) continue;
          if (mask[sampleY * width + sampleX]) {
            occupied = true;
            break;
          }
        }
      }
      if (occupied) output[y * width + x] = 1;
    }
  }
  return output;
}

function summarizeSceneRegion(before, after, include) {
  let count = 0;
  let absoluteDifference = 0;
  let changed20 = 0;
  let changed35 = 0;
  let darkLoss35 = 0;
  for (let index = 0; index < include.length; index += 1) {
    if (!include[index]) continue;
    const oldRgb = pixel(before, index);
    const newRgb = pixel(after, index);
    const channelDifference = oldRgb.map((value, channel) => Math.abs(value - newRgb[channel]));
    const maximumDifference = Math.max(...channelDifference);
    count += 1;
    absoluteDifference += channelDifference.reduce((sum, value) => sum + value, 0) / 3;
    if (maximumDifference > 20) changed20 += 1;
    if (maximumDifference > 35) changed35 += 1;
    if (luma(oldRgb) - luma(newRgb) > 35) darkLoss35 += 1;
  }
  const fraction = (value) => rounded(value / Math.max(1, count), 6);
  return {
    pixelCount: count,
    meanAbsoluteRgbDelta: rounded(absoluteDifference / Math.max(1, count), 3),
    changedPixelFraction: {
      maximumChannelAbove20: fraction(changed20),
      maximumChannelAbove35: fraction(changed35),
    },
    darkLossFractionAbove35Luma: fraction(darkLoss35),
  };
}

function sceneCollateral(beforeFrames, afterFrames) {
  const beforeMask = buildObjectMask(beforeFrames.isolated).core;
  const afterMask = buildObjectMask(afterFrames.isolated).core;
  const width = beforeFrames.staticOn.width;
  const height = beforeFrames.staticOn.height;
  const union = new Uint8Array(width * height);
  let minimumX = width;
  let maximumX = 0;
  let maximumY = 0;
  for (let index = 0; index < union.length; index += 1) {
    union[index] = beforeMask[index] || afterMask[index] ? 1 : 0;
    if (!union[index]) continue;
    const x = index % width;
    const y = Math.floor(index / width);
    minimumX = Math.min(minimumX, x);
    maximumX = Math.max(maximumX, x);
    maximumY = Math.max(maximumY, y);
  }
  const near = dilateMask(union, width, height, 3);
  const farBoundary = dilateMask(union, width, height, 20);
  const perimeter = new Uint8Array(union.length);
  const farScene = new Uint8Array(union.length);
  const supportBand = new Uint8Array(union.length);
  for (let index = 0; index < union.length; index += 1) {
    const x = index % width;
    const y = Math.floor(index / width);
    perimeter[index] = farBoundary[index] && !near[index] ? 1 : 0;
    farScene[index] = farBoundary[index] ? 0 : 1;
    supportBand[index] = !near[index]
      && x >= minimumX - 20
      && x <= maximumX + 20
      && y >= maximumY + 3
      && y <= maximumY + 35
      ? 1
      : 0;
  }
  return {
    policy: {
      nearObjectExclusionRadiusPx: 3,
      perimeterOuterRadiusPx: 20,
      supportBandBelowObjectPx: [3, 35],
      interpretation: "Perimeter change can include intended residual removal; dark loss records possible newly exposed holes. Far-scene change is collateral risk.",
    },
    objectSampleBounds: { minimumX, maximumX, maximumY },
    perimeter: summarizeSceneRegion(beforeFrames.staticOn, afterFrames.staticOn, perimeter),
    supportBand: summarizeSceneRegion(beforeFrames.staticOn, afterFrames.staticOn, supportBand),
    farScene: summarizeSceneRegion(beforeFrames.staticOn, afterFrames.staticOn, farScene),
  };
}

function createDiagnostics(page, manifestPathname, frontAssetPathname) {
  const diagnostics = {
    consoleErrors: [],
    pageErrors: [],
    requestFailures: [],
    httpErrors: [],
    manifestRequests: 0,
    manifestResponseSha256: [],
    frontAssetRequests: 0,
  };
  const responseTasks = [];
  page.on("console", (message) => {
    if (message.type() === "error") diagnostics.consoleErrors.push(message.text());
  });
  page.on("pageerror", (error) => diagnostics.pageErrors.push(error.message));
  page.on("requestfailed", (request) => diagnostics.requestFailures.push({
    url: request.url(),
    error: request.failure()?.errorText || "unknown",
  }));
  page.on("request", (request) => {
    const pathname = new URL(request.url()).pathname;
    if (pathname === manifestPathname) diagnostics.manifestRequests += 1;
    if (pathname === frontAssetPathname) diagnostics.frontAssetRequests += 1;
  });
  page.on("response", (response) => {
    const pathname = new URL(response.url()).pathname;
    if (response.status() >= 400) diagnostics.httpErrors.push({ url: response.url(), status: response.status() });
    if (pathname === manifestPathname) {
      responseTasks.push(response.body().then((body) => {
        diagnostics.manifestResponseSha256.push(sha256Bytes(body));
      }));
    }
  });
  diagnostics.settle = async () => Promise.all(responseTasks);
  return diagnostics;
}

async function testRobotCollision(page) {
  const prepared = await page.evaluate((id) => window.__prepareRobotCollisionTest(id), FRONT_ID);
  let step = null;
  if (prepared.prepared) {
    for (let attempt = 0; attempt < 5; attempt += 1) {
      step = await page.evaluate((id) => window.__stepRobotTowardObject(id, 0.65), FRONT_ID);
      if (step?.blocked && step.lastCollisionObjectId === FRONT_ID) break;
    }
  }
  return {
    prepared: Boolean(prepared.prepared),
    reason: prepared.reason || null,
    blockedByFront: Boolean(step?.blocked && step.lastCollisionObjectId === FRONT_ID),
    lastCollisionKind: step?.lastCollisionKind || null,
    lastCollisionObjectId: step?.lastCollisionObjectId || null,
  };
}

async function captureCandidate({ browser, label, baseUrl, manifestPath, screenshotDir }) {
  const manifest = readJson(manifestPath);
  const manifestSha256 = sha256File(manifestPath);
  const expected = expectedCounts(manifest);
  const definition = frontDefinition(manifest);
  const url = runtimeUrl(baseUrl, manifestPath);
  const manifestPathname = new URL(new URL(url).searchParams.get("manifest"), url).pathname;
  const frontAssetPathname = new URL(definition.collision.asset.url, url).pathname;
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
  const page = await context.newPage();
  const diagnostics = createDiagnostics(page, manifestPathname, frontAssetPathname);
  try {
    await page.goto(url, { waitUntil: "domcontentloaded", timeout: 60_000 });
    await waitForReady(page);
    if ((await page.evaluate(() => window.__visualPhysicsDemoState.robotEnabled)) === true) {
      await page.locator("#toggleRobot").click({ force: true });
    }
    const yaw = await page.evaluate((id) => window.__setInteractiveObjectYaw(id, 0), FRONT_ID);
    const focus = await page.evaluate((id) => window.__focusSceneEntity(id), FRONT_ID);
    await page.waitForTimeout(2_000);
    const initialState = await page.evaluate(() => window.__visualPhysicsDemoState);
    const collisionWorld = await page.evaluate(() => window.__inspectCollisionWorld());
    const front = await page.evaluate((id) => window.__inspectInteractiveObject(id), FRONT_ID);

    await setToggle(page, "#toggleObjectSolo", "interactiveObjectSolo", false);
    await setToggle(page, "#toggleStaticVisual", "showStaticVisual", true);
    const staticOn = await captureFrame(page, path.join(screenshotDir, `${label}-static-on.png`));
    await setToggle(page, "#toggleStaticVisual", "showStaticVisual", false);
    const staticOff = await captureFrame(page, path.join(screenshotDir, `${label}-static-off.png`));
    await setToggle(page, "#toggleObjectSolo", "interactiveObjectSolo", true);
    const isolated = await captureFrame(page, path.join(screenshotDir, `${label}-isolated.png`));

    await setToggle(page, "#toggleObjectSolo", "interactiveObjectSolo", false);
    const interpenetrations = await page.evaluate(() => window.__inspectSceneInterpenetrations());
    const robotCollision = await testRobotCollision(page);
    await diagnostics.settle();
    const manifestSha256After = sha256File(manifestPath);
    return {
      label,
      url,
      manifest: {
        path: manifestPath,
        sha256Before: manifestSha256,
        sha256After: manifestSha256After,
        unchanged: manifestSha256 === manifestSha256After,
        version: manifest.version,
      },
      expected,
      runtime: {
        staticVisualCount: initialState.staticVisualCount,
        visualCount: initialState.visualCount,
        interactiveObjectReadyCount: initialState.interactiveObjectReadyCount,
        objectColliderReadyCount: initialState.objectColliderReadyCount,
        staticColliderFaces: initialState.colliderFaces,
        objectColliderFaces: collisionWorld.objectColliderFaces,
        degradedColliderCount: collisionWorld.degradedCount,
        colliderErrors: collisionWorld.errors,
      },
      front: {
        yawZeroApplied: yaw.updated === true,
        focused: focus.focused === true,
        visualKind: front.visualKind,
        collisionMode: front.collisionMode,
        collisionTopology: front.collisionTopology,
        unifiedVisualCollision: front.unifiedVisualCollision,
        meshVertexCount: front.meshVertexCount,
        collisionFaces: front.collisionFaces,
        pbrMaterialCount: front.pbrMaterialCount,
        materialTypes: front.materialTypes,
        collisionBounds: front.collisionBounds,
      },
      sceneContact: {
        status: interpenetrations.status,
        frontPairs: interpenetrations.intersections.filter((item) => item.colliderIds?.includes(FRONT_ID)),
      },
      robotCollision,
      contamination: frameContamination({ staticOn, staticOff, isolated }),
      screenshots: {
        staticOn: path.join(screenshotDir, `${label}-static-on.png`),
        staticOff: path.join(screenshotDir, `${label}-static-off.png`),
        isolated: path.join(screenshotDir, `${label}-isolated.png`),
      },
      diagnostics: {
        ...diagnostics,
        settle: undefined,
      },
      _frames: { staticOn, staticOff, isolated },
    };
  } finally {
    await context.close();
  }
}

function candidateChecks(baseline, candidate, manifests, collateral) {
  const before = baseline.contamination;
  const after = candidate.contamination;
  const diagnosticsClean = (entry) => (
    entry.diagnostics.consoleErrors.length === 0
    && entry.diagnostics.pageErrors.length === 0
    && entry.diagnostics.requestFailures.length === 0
    && entry.diagnostics.httpErrors.length === 0
    && entry.diagnostics.manifestRequests === 1
    && entry.diagnostics.frontAssetRequests === 1
    && entry.diagnostics.manifestResponseSha256.length === 1
    && entry.diagnostics.manifestResponseSha256[0] === entry.manifest.sha256Before
  );
  const runtimeMatches = (entry) => (
    entry.runtime.staticVisualCount === entry.expected.staticVisuals
    && entry.runtime.visualCount === entry.expected.totalVisuals
    && entry.runtime.interactiveObjectReadyCount === entry.expected.visualObjects
    && entry.runtime.objectColliderReadyCount === entry.expected.objectColliders
    && entry.runtime.staticColliderFaces === entry.expected.staticColliderFaces
    && entry.runtime.objectColliderFaces === entry.expected.objectColliderFaces
    && entry.runtime.degradedColliderCount === 0
    && entry.runtime.colliderErrors.length === 0
  );
  const frontReady = (entry) => (
    entry.front.yawZeroApplied
    && entry.front.focused
    && entry.front.visualKind === "mesh"
    && entry.front.collisionMode === "unified-glb"
    && entry.front.collisionTopology === "surface_bvh"
    && entry.front.unifiedVisualCollision === true
    && entry.front.pbrMaterialCount > 0
    && entry.front.materialTypes.some((type) => ["MeshStandardMaterial", "MeshPhysicalMaterial"].includes(type))
  );
  const rearDefinitions = (manifest) => REAR_IDS.map((id) => (
    manifest.interactiveObjects.find((item) => item.id === id)
  ));
  return {
    bothManifestsUnchanged: baseline.manifest.unchanged && candidate.manifest.unchanged,
    bothFreshLoadsExact: runtimeMatches(baseline) && runtimeMatches(candidate),
    bothDiagnosticsClean: diagnosticsClean(baseline) && diagnosticsClean(candidate),
    frontUnifiedPbrYawZeroPreserved: frontReady(baseline) && frontReady(candidate),
    collisionBoundsUnchanged: stableJson(baseline.front.collisionBounds) === stableJson(candidate.front.collisionBounds),
    frontDefinitionUnchanged: stableJson(frontDefinition(manifests.baseline)) === stableJson(frontDefinition(manifests.candidate)),
    rearDefinitionsUnchanged: stableJson(rearDefinitions(manifests.baseline)) === stableJson(rearDefinitions(manifests.candidate)),
    sceneColliderUnchanged: stableJson({
      collisionWorld: manifests.baseline.collisionWorld,
      asset: manifests.baseline.assets[manifests.baseline.collisionWorld?.sceneAssetKey || "collider"],
    }) === stableJson({
      collisionWorld: manifests.candidate.collisionWorld,
      asset: manifests.candidate.assets[manifests.candidate.collisionWorld?.sceneAssetKey || "collider"],
    }),
    frontSceneContactUnchanged: stableJson(baseline.sceneContact.frontPairs)
      === stableJson(candidate.sceneContact.frontPairs),
    robotCollisionPreserved: baseline.robotCollision.blockedByFront && candidate.robotCollision.blockedByFront,
    objectMaskStable: Math.abs(after.objectCorePixelCount - before.objectCorePixelCount)
      / Math.max(1, before.objectCorePixelCount) <= 0.04,
    staticContaminationReduced: after.changedPixelFraction.maximumChannelAbove35
      <= before.changedPixelFraction.maximumChannelAbove35 * 0.88,
    meanStaticDeltaReduced: after.meanAbsoluteRgbDeltaStaticOnVsOff
      <= before.meanAbsoluteRgbDeltaStaticOnVsOff * 0.88,
    darkOcclusionNotWorse: after.darkOcclusionFraction <= before.darkOcclusionFraction + 0.01,
    farSceneCollateralBelowOnePercent: collateral.farScene.changedPixelFraction.maximumChannelAbove35 <= 0.01,
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    console.log(usage());
    return;
  }
  const baselinePath = path.resolve(args.baseline || DEFAULT_BASELINE);
  const candidatePath = path.resolve(args.candidate || DEFAULT_CANDIDATE);
  const reportPath = path.resolve(args.report || DEFAULT_REPORT);
  const screenshotDir = path.resolve(args["screenshot-dir"] || reviewRoot);
  const baseUrl = args.url || DEFAULT_URL;
  fs.mkdirSync(screenshotDir, { recursive: true });
  const manifests = { baseline: readJson(baselinePath), candidate: readJson(candidatePath) };
  const startedAt = new Date().toISOString();
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  try {
    const baseline = await captureCandidate({
      browser,
      label: "before",
      baseUrl,
      manifestPath: baselinePath,
      screenshotDir,
    });
    const candidate = await captureCandidate({
      browser,
      label: "after",
      baseUrl,
      manifestPath: candidatePath,
      screenshotDir,
    });
    const collateral = sceneCollateral(baseline._frames, candidate._frames);
    delete baseline._frames;
    delete candidate._frames;
    const checks = candidateChecks(baseline, candidate, manifests, collateral);
    const failedChecks = Object.entries(checks).filter(([, passed]) => !passed).map(([name]) => name);
    const removed = baseline.expected.staticVisuals - candidate.expected.staticVisuals;
    const report = {
      schemaVersion: 1,
      kind: "video2world.bedroom4_front_static_recarve_browser_comparison",
      status: failedChecks.length === 0 ? "candidate_passed" : "candidate_failed",
      promotion: "not_performed",
      publishing: "not_performed",
      startedAt,
      finishedAt: new Date().toISOString(),
      policy: {
        purpose: "Compare residual static Gaussian occlusion without changing the completed front-pillow asset.",
        camera: "Fresh deterministic focus on sam3_pillow_front at yaw=0 for each manifest.",
        staticReference: "Static Scene Off, Solo Object Off.",
        objectMask: "Eroded non-background mask from Static Scene Off, Solo Object On.",
        acceptance: "At least 12% relative reduction in >35 RGB-delta pixels and mean RGB delta, with unchanged object/collision definitions and fresh-load health.",
      },
      summary: {
        removedStaticGaussians: removed,
        retainedStaticGaussians: candidate.expected.staticVisuals,
        changedPixelFractionAbove35: {
          before: baseline.contamination.changedPixelFraction.maximumChannelAbove35,
          after: candidate.contamination.changedPixelFraction.maximumChannelAbove35,
          relativeReduction: rounded(
            1 - candidate.contamination.changedPixelFraction.maximumChannelAbove35
              / Math.max(1e-9, baseline.contamination.changedPixelFraction.maximumChannelAbove35),
            6,
          ),
        },
        meanAbsoluteRgbDelta: {
          before: baseline.contamination.meanAbsoluteRgbDeltaStaticOnVsOff,
          after: candidate.contamination.meanAbsoluteRgbDeltaStaticOnVsOff,
          relativeReduction: rounded(
            1 - candidate.contamination.meanAbsoluteRgbDeltaStaticOnVsOff
              / Math.max(1e-9, baseline.contamination.meanAbsoluteRgbDeltaStaticOnVsOff),
            6,
          ),
        },
        sceneCollateral: collateral,
      },
      decision: {
        currentDemoPromotion: failedChecks.length === 0
          ? "recommended_with_known_limitation"
          : "held",
        acceptanceBasis: [
          "completed pillow shape is credible in the current demo",
          "main color is light neutral/white in the current scene view",
          "no obvious scene interpenetration is reported",
          "unified PBR visual, selection surface, robot collision, and fresh-load checks pass",
          "residual static-Gaussian foreground contamination is materially reduced",
        ],
        backgroundAndBedCompletion: "not_proven",
        mandatoryWarning: "Near-object perimeter and support-band dark loss remain visible; this candidate removes stale foreground splats but does not reconstruct hidden bed/background geometry.",
        generalPipelineClaim: "held_until_multiview_clean_plate_depth_normal_and_scene_reconstruction_pass",
      },
      checks,
      failedChecks,
      baseline,
      candidate,
      limitations: [
        "This browser comparison validates the Bedroom 4 candidate, not a learned universal amodal completion model.",
        "Rear RGB point clouds are preservation anchors; the current artifact does not claim reconstructed hidden bed geometry.",
        "Candidate chunks remain unpromoted and are not referenced by the canonical or deployed manifest.",
      ],
    };
    fs.mkdirSync(path.dirname(reportPath), { recursive: true });
    fs.writeFileSync(reportPath, `${JSON.stringify(portableReport(report), null, 2)}\n`);
    console.log(JSON.stringify({
      status: report.status,
      failedChecks,
      summary: report.summary,
      report: reportPath,
      screenshots: screenshotDir,
    }, null, 2));
    if (failedChecks.length) process.exitCode = 1;
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error.stack || error.message || String(error));
  process.exitCode = 1;
});
