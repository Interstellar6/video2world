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
const worldDir = path.join(publicRoot, "worlds/bedroom4");
const DEFAULT_MANIFEST = path.join(worldDir, "manifest.unified-pillow.json");
const DEFAULT_REPORT = path.join(worldDir, "qa/unified-pillow-browser-qa.json");
const DEFAULT_SCREENSHOT_DIR = path.join(worldDir, "qa/unified-pillow-browser");
const DEFAULT_URL = "http://127.0.0.1:4177/";
const FRONT_ID = "sam3_pillow_front";
const REAR_IDS = ["sam3_pillow_left", "sam3_pillow_right"];
const EXPECTED = Object.freeze({
  staticVisuals: 821_785,
  existingObjectGaussians: 480_000,
  rearRgbPoints: 138_926,
  frontMeshVertices: 60_237,
  totalVisualPrimitives: 1_500_948,
  staticColliderFaces: 1_249_186,
  objectColliderFaces: 164_742,
  frontColliderFaces: 97_082,
  visualObjects: 7,
  objectColliders: 5,
});

function usage() {
  return [
    "Usage: node scripts/qa_bedroom4_unified_pillow_browser.mjs [options]",
    "",
    "Options:",
    `  --url URL                 Web runtime URL (default: ${DEFAULT_URL})`,
    `  --manifest PATH           Candidate manifest (default: ${DEFAULT_MANIFEST})`,
    `  --report PATH             JSON report (default: ${DEFAULT_REPORT})`,
    `  --screenshot-dir PATH     Screenshot directory (default: ${DEFAULT_SCREENSHOT_DIR})`,
    "  --help                    Show this help",
    "",
    "The script never edits the manifest and never publishes or promotes the candidate.",
  ].join("\n");
}

function parseArgs(argv) {
  const allowed = new Set(["url", "manifest", "report", "screenshot-dir", "screenshots-dir"]);
  const options = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === "--help") return { help: true };
    if (!token.startsWith("--")) throw new Error(`Unexpected argument: ${token}`);
    const key = token.slice(2);
    if (!allowed.has(key)) throw new Error(`Unsupported option: --${key}`);
    const value = argv[index + 1];
    if (!value || value.startsWith("--")) throw new Error(`Missing value for --${key}`);
    options[key] = value;
    index += 1;
  }
  if (options["screenshot-dir"] && options["screenshots-dir"]) {
    throw new Error("Use only one of --screenshot-dir or --screenshots-dir");
  }
  return options;
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

function sha256Bytes(bytes) {
  return crypto.createHash("sha256").update(bytes).digest("hex");
}

function sha256File(filePath) {
  return sha256Bytes(fs.readFileSync(filePath));
}

function rounded(value, digits = 4) {
  return Number(Number(value).toFixed(digits));
}

function record(condition, message, failures) {
  if (!condition) failures.push(message);
  return Boolean(condition);
}

function requireCondition(condition, message) {
  if (!condition) throw new Error(message);
}

function matrixMaximumDelta(left, right) {
  if (!Array.isArray(left) || !Array.isArray(right) || left.length !== right.length) return Infinity;
  return Math.max(...left.map((value, index) => Math.abs(value - right[index])));
}

function matricesEqual(left, right, epsilon = 1e-5) {
  return matrixMaximumDelta(left, right) <= epsilon;
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

function manifestPublicUrl(manifestPath) {
  const relative = path.relative(publicRoot, manifestPath);
  requireCondition(
    relative && !relative.startsWith("..") && !path.isAbsolute(relative),
    `Manifest must be inside ${publicRoot} so Vite can serve the exact audited file`,
  );
  return `/${relative.split(path.sep).join("/")}`;
}

function buildRuntimeUrl(rawUrl, manifestPath) {
  const url = new URL(rawUrl || DEFAULT_URL);
  url.searchParams.set("manifest", manifestPublicUrl(manifestPath));
  url.searchParams.set("collisionDebug", "on");
  return url.toString();
}

function createPageDiagnostics(page, { label, manifestPathname, frontAssetPathname }) {
  const diagnostics = {
    label,
    consoleMessages: [],
    pageErrors: [],
    requestFailures: [],
    httpErrors: [],
    manifestRequestCount: 0,
    manifestResponseSha256: [],
    frontAssetRequestCount: 0,
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
    if (pathname === frontAssetPathname) diagnostics.frontAssetRequestCount += 1;
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
    await Promise.all(responseTasks);
    return diagnostics;
  };
  return diagnostics;
}

async function waitForCandidateReady(page) {
  await page.waitForFunction((expected) => {
    const state = window.__visualPhysicsDemoState;
    return state?.visualReady
      && state?.colliderReady
      && state?.sceneQaReady
      && state?.interactiveObjectReadyCount === expected.visualObjects
      && state?.objectColliderReadyCount === expected.objectColliders;
  }, EXPECTED, { timeout: 300_000 });
}

async function inspectCanvasPixels(page) {
  return page.locator("#sceneCanvas").evaluate(async (canvas) => {
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    const sampleSize = 96;
    const copy = document.createElement("canvas");
    copy.width = sampleSize;
    copy.height = sampleSize;
    const context = copy.getContext("2d", { willReadFrequently: true });
    context.drawImage(canvas, 0, 0, sampleSize, sampleSize);
    const pixels = context.getImageData(0, 0, sampleSize, sampleSize).data;
    let opaquePixels = 0;
    let nonDarkPixels = 0;
    let minimumLuma = 255;
    let maximumLuma = 0;
    const colors = new Set();
    for (let index = 0; index < pixels.length; index += 4) {
      const red = pixels[index];
      const green = pixels[index + 1];
      const blue = pixels[index + 2];
      const alpha = pixels[index + 3];
      if (alpha > 8) opaquePixels += 1;
      const luma = 0.2126 * red + 0.7152 * green + 0.0722 * blue;
      if (luma > 8) nonDarkPixels += 1;
      minimumLuma = Math.min(minimumLuma, luma);
      maximumLuma = Math.max(maximumLuma, luma);
      colors.add(`${red >> 4}:${green >> 4}:${blue >> 4}:${alpha >> 6}`);
    }
    return {
      sampleSize: [sampleSize, sampleSize],
      backingStore: [canvas.width, canvas.height],
      cssSize: [canvas.getBoundingClientRect().width, canvas.getBoundingClientRect().height],
      opaquePixels,
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
      hud: hud ? { ...hud, rightInset: window.innerWidth - hud.right } : null,
      queryRow,
      answer,
      metrics,
      status,
      controls,
      body: {
        scrollWidth: document.body.scrollWidth,
        clientWidth: document.body.clientWidth,
        htmlScrollWidth: document.documentElement.scrollWidth,
        htmlClientWidth: document.documentElement.clientWidth,
      },
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

function boundsOverlapMetrics(left, right) {
  const overlap = ["x", "y", "z"].map((axis) => (
    Math.max(0, Math.min(left.max[axis], right.max[axis]) - Math.max(left.min[axis], right.min[axis]))
  ));
  const leftSize = ["x", "y", "z"].map((axis) => Math.max(0, left.max[axis] - left.min[axis]));
  const rightSize = ["x", "y", "z"].map((axis) => Math.max(0, right.max[axis] - right.min[axis]));
  const overlapVolume = overlap.reduce((product, value) => product * value, 1);
  const smallerVolume = Math.min(
    leftSize.reduce((product, value) => product * value, 1),
    rightSize.reduce((product, value) => product * value, 1),
  );
  const normalizedDepths = overlap.map((value, index) => (
    value / Math.max(1e-6, Math.min(leftSize[index], rightSize[index]))
  ));
  return {
    overlap: { x: overlap[0], y: overlap[1], z: overlap[2] },
    overlapVolume: rounded(overlapVolume, 8),
    overlapFractionOfSmallerAabb: rounded(overlapVolume / Math.max(1e-8, smallerVolume), 6),
    minimumNormalizedAxisDepth: rounded(Math.min(...normalizedDepths), 6),
  };
}

function classifyInterpenetrations(rawReport) {
  const classifications = (rawReport.intersections || []).map((intersection) => {
    const ids = intersection.colliderIds || [];
    const involvesFront = ids.includes(FRONT_ID);
    const involvesScene = ids.includes("scene");
    const overlap = boundsOverlapMetrics(intersection.leftBounds, intersection.rightBounds);
    let classification = "baseline_existing_pair";
    let blocking = false;
    if (involvesFront && involvesScene) {
      classification = "support_or_conservative_surface_contact";
    } else if (involvesFront) {
      blocking = overlap.overlapFractionOfSmallerAabb >= 0.08
        && overlap.minimumNormalizedAxisDepth >= 0.15;
      classification = blocking
        ? "obvious_front_object_interpenetration"
        : "minor_or_conservative_front_object_contact";
    }
    return { ...intersection, overlap, classification, blocking };
  });
  return {
    ...rawReport,
    automatedStatus: classifications.some((item) => item.blocking) ? "failed" : "passed",
    policy: {
      sceneSurfaceContact: "record_as_support_or_conservative_contact",
      existingBaselinePairs: "record_only",
      obviousFrontObjectThreshold: {
        overlapFractionOfSmallerAabb: 0.08,
        minimumNormalizedAxisDepth: 0.15,
      },
      humanReviewStillRequired: true,
    },
    classifications,
  };
}

async function inspectSupport(page, frontInspection, manifest) {
  const worldUp = manifest.coordinateSystem?.worldUp
    || manifest.coordinateFrames?.visualNative?.worldUp
    || manifest.coordinateFrame?.worldUp
    || [0, -1, 0];
  const relation = (manifest.sceneKnowledge?.relations || []).find((item) => (
    item.subject === FRONT_ID && item.predicate === "OnTopOf"
  )) || null;
  const probes = await page.evaluate(({ bounds, worldUp: upValues, objectId }) => {
    const magnitude = Math.hypot(...upValues) || 1;
    const up = upValues.map((value) => value / magnitude);
    const down = up.map((value) => -value);
    const dot = (left, right) => left.reduce((sum, value, index) => (
      sum + value * right[index]
    ), 0);
    const cross = (left, right) => [
      left[1] * right[2] - left[2] * right[1],
      left[2] * right[0] - left[0] * right[2],
      left[0] * right[1] - left[1] * right[0],
    ];
    const normalize = (value) => {
      const length = Math.hypot(...value) || 1;
      return value.map((item) => item / length);
    };
    const addScaled = (point, direction, scale) => point.map((value, index) => (
      value + direction[index] * scale
    ));
    const reference = Math.abs(up[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
    const tangentA = normalize(cross(up, reference));
    const tangentB = normalize(cross(up, tangentA));
    const size = [bounds.size.x, bounds.size.y, bounds.size.z];
    const halfVerticalExtent = 0.5 * (
      Math.abs(bounds.size.x * up[0])
      + Math.abs(bounds.size.y * up[1])
      + Math.abs(bounds.size.z * up[2])
    );
    const halfTangentA = 0.5 * size.reduce((sum, value, index) => (
      sum + Math.abs(value * tangentA[index])
    ), 0);
    const halfTangentB = 0.5 * size.reduce((sum, value, index) => (
      sum + Math.abs(value * tangentB[index])
    ), 0);
    const baseOrigin = [
      bounds.center.x + down[0] * (halfVerticalExtent + 0.08),
      bounds.center.y + down[1] * (halfVerticalExtent + 0.08),
      bounds.center.z + down[2] * (halfVerticalExtent + 0.08),
    ];
    const centerHit = window.__probeCollisionRay(baseOrigin, down, 3);
    const factors = [-0.9, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 0.9];
    const samples = [];
    for (const factorA of factors) {
      for (const factorB of factors) {
        let origin = addScaled(baseOrigin, tangentA, factorA * halfTangentA);
        origin = addScaled(origin, tangentB, factorB * halfTangentB);
        const outward = window.__probeCollisionRay(origin, down, 0.35);
        const inward = window.__probeCollisionRay(origin, up, 0.35);
        const sceneHit = [outward, inward]
          .filter((hit) => {
            if (hit?.colliderKind !== "scene" || !hit.normal) return false;
            const normal = [hit.normal.x, hit.normal.y, hit.normal.z];
            return Math.abs(dot(normal, up)) >= 0.75 && hit.distance <= 0.2;
          })
          .sort((left, right) => left.distance - right.distance)[0] || null;
        samples.push({ factorA, factorB, origin, outward, inward, sceneHit });
      }
    }
    const sceneHits = samples.filter((sample) => sample.sceneHit);
    const uniqueSceneFaces = new Set(sceneHits.map((sample) => sample.sceneHit.faceIndex));
    return {
      downwardSceneProbe: {
        objectId,
        origin: baseOrigin,
        direction: down,
        far: 3,
        hit: centerHit,
        clearsFrontSurface: Boolean(centerHit && centerHit.objectId !== objectId),
      },
      multiSampleSceneProbe: {
        objectId,
        tangentA,
        tangentB,
        factors,
        maximumSupportDistance: 0.35,
        maximumAcceptedSupportDistance: 0.2,
        minimumNormalAlignment: 0.75,
        sampleCount: samples.length,
        sceneHitCount: sceneHits.length,
        uniqueSceneFaceCount: uniqueSceneFaces.size,
        minimumRequiredSceneHits: 3,
        minimumRequiredUniqueSceneFaces: 2,
        passed: sceneHits.length >= 3 && uniqueSceneFaces.size >= 2,
        evidenceMode: "protected_down_side_support_neighborhood",
        directPhysicalContactProven: false,
        samples,
      },
    };
  }, { bounds: frontInspection.collisionBounds, worldUp, objectId: FRONT_ID });
  return { worldUp, relation, ...probes };
}

async function testFrontInteraction(page, failures) {
  await page.evaluate((id) => window.__focusSceneEntity(id), FRONT_ID);
  await page.waitForTimeout(1_400);
  const point = await page.evaluate((id) => window.__getInteractiveObjectScreenPoint(id), FRONT_ID);
  const hit = await page.evaluate(
    ({ x, y }) => window.__inspectInteractiveObjectHitAt(x, y),
    point,
  );
  record(hit?.objectId === FRONT_ID, `front pointer hit object=${hit?.objectId}`, failures);
  record(hit?.colliderMode === "unified-glb", `front pointer mode=${hit?.colliderMode}`, failures);
  record(hit?.collisionTopology === "surface_bvh", `front pointer topology=${hit?.collisionTopology}`, failures);

  const beforeDrag = await page.evaluate((id) => window.__inspectInteractiveObject(id), FRONT_ID);
  await page.mouse.move(point.x, point.y);
  await page.mouse.down();
  await page.mouse.move(point.x + 88, point.y, { steps: 10 });
  await page.mouse.up();
  const afterDrag = await page.evaluate((id) => ({
    inspection: window.__inspectInteractiveObject(id),
    activeDrag: window.__visualPhysicsDemoState.activeObjectDrag,
  }), FRONT_ID);
  const dragDeltas = {
    group: matrixMaximumDelta(beforeDrag.groupMatrixWorld, afterDrag.inspection.groupMatrixWorld),
    visual: matrixMaximumDelta(beforeDrag.splatMatrixWorld, afterDrag.inspection.splatMatrixWorld),
    collision: matrixMaximumDelta(beforeDrag.collisionMatrixWorld, afterDrag.inspection.collisionMatrixWorld),
  };
  record(dragDeltas.group > 1e-4, "front drag did not rotate group", failures);
  record(dragDeltas.visual > 1e-4, "front drag did not rotate PBR visual", failures);
  record(dragDeltas.collision > 1e-4, "front drag did not rotate collision mesh", failures);
  record(afterDrag.activeDrag == null, "front drag remained active after mouseup", failures);

  await page.evaluate((id) => window.__setInteractiveObjectYaw(id, 0), FRONT_ID);
  const afterDragReset = await page.evaluate((id) => window.__inspectInteractiveObject(id), FRONT_ID);
  const dragReturned = {
    group: matricesEqual(beforeDrag.groupMatrixWorld, afterDragReset.groupMatrixWorld),
    visual: matricesEqual(beforeDrag.splatMatrixWorld, afterDragReset.splatMatrixWorld),
    collision: matricesEqual(beforeDrag.collisionMatrixWorld, afterDragReset.collisionMatrixWorld),
  };
  record(Object.values(dragReturned).every(Boolean), "front drag reset did not restore all matrices", failures);

  const spinPoint = await page.evaluate((id) => window.__getInteractiveObjectScreenPoint(id), FRONT_ID);
  const beforeSpin = await page.evaluate((id) => ({
    inspection: window.__inspectInteractiveObject(id),
    turns: window.__visualPhysicsDemoState.interactiveObjectTurns,
  }), FRONT_ID);
  await page.mouse.dblclick(spinPoint.x, spinPoint.y, { delay: 45 });
  await page.waitForFunction((id) => window.__visualPhysicsDemoState.interactiveObjects
    .find((item) => item.id === id)?.spinning === true, FRONT_ID, { timeout: 8_000 });
  await page.waitForTimeout(260);
  const duringSpin = await page.evaluate((id) => window.__inspectInteractiveObject(id), FRONT_ID);
  await page.waitForFunction(({ id, turns }) => (
    window.__visualPhysicsDemoState.interactiveObjectTurns > turns
    && window.__visualPhysicsDemoState.interactiveObjects.find((item) => item.id === id)?.spinning === false
  ), { id: FRONT_ID, turns: beforeSpin.turns }, { timeout: 10_000 });
  const afterSpin = await page.evaluate((id) => ({
    inspection: window.__inspectInteractiveObject(id),
    turns: window.__visualPhysicsDemoState.interactiveObjectTurns,
  }), FRONT_ID);
  const spinDeltas = {
    visual: matrixMaximumDelta(beforeSpin.inspection.splatMatrixWorld, duringSpin.splatMatrixWorld),
    collision: matrixMaximumDelta(
      beforeSpin.inspection.collisionMatrixWorld,
      duringSpin.collisionMatrixWorld,
    ),
  };
  const spinReturned = {
    group: matricesEqual(beforeSpin.inspection.groupMatrixWorld, afterSpin.inspection.groupMatrixWorld),
    visual: matricesEqual(beforeSpin.inspection.splatMatrixWorld, afterSpin.inspection.splatMatrixWorld),
    collision: matricesEqual(
      beforeSpin.inspection.collisionMatrixWorld,
      afterSpin.inspection.collisionMatrixWorld,
    ),
  };
  record(spinDeltas.visual > 1e-4, "front 360 spin did not move PBR visual", failures);
  record(spinDeltas.collision > 1e-4, "front 360 spin did not move collision mesh", failures);
  record(afterSpin.turns > beforeSpin.turns, "front 360 turn counter did not increment", failures);
  record(Object.values(spinReturned).every(Boolean), "front 360 spin did not return all matrices", failures);
  return {
    pointer: { point, hit },
    drag: { deltas: dragDeltas, released: afterDrag.activeDrag == null, returned: dragReturned },
    doubleClick360: {
      turnsBefore: beforeSpin.turns,
      turnsAfter: afterSpin.turns,
      inFlightDeltas: spinDeltas,
      returned: spinReturned,
    },
  };
}

async function testRearSelectionOnly(page, objectId, failures) {
  await page.evaluate((id) => window.__focusSceneEntity(id), objectId);
  await page.waitForTimeout(1_000);
  const before = await page.evaluate((id) => ({
    inspection: window.__inspectInteractiveObject(id),
    targets: window.__inspectCollisionWorld().targetCount,
    turns: window.__visualPhysicsDemoState.interactiveObjectTurns,
  }), objectId);
  const point = await page.evaluate((id) => window.__getInteractiveObjectScreenPoint(id), objectId);
  await page.mouse.move(point.x, point.y);
  await page.mouse.down();
  await page.mouse.move(point.x + 76, point.y, { steps: 8 });
  await page.mouse.up();
  const spinPoint = await page.evaluate((id) => window.__getInteractiveObjectScreenPoint(id), objectId);
  await page.mouse.dblclick(spinPoint.x, spinPoint.y, { delay: 45 });
  await page.waitForTimeout(450);
  const after = await page.evaluate((id) => ({
    inspection: window.__inspectInteractiveObject(id),
    targets: window.__inspectCollisionWorld().targetCount,
    turns: window.__visualPhysicsDemoState.interactiveObjectTurns,
    prepared: window.__prepareRobotCollisionTest(id),
  }), objectId);
  const unchanged = {
    group: matricesEqual(before.inspection.groupMatrixWorld, after.inspection.groupMatrixWorld),
    visual: matricesEqual(before.inspection.splatMatrixWorld, after.inspection.splatMatrixWorld),
    proxy: matricesEqual(before.inspection.proxyMatrixWorld, after.inspection.proxyMatrixWorld),
  };
  record(before.inspection.visualKind === "rgb-points", `${objectId}: visual kind changed`, failures);
  record(before.inspection.interactionKind == null, `${objectId}: interaction unexpectedly enabled`, failures);
  record(before.inspection.collisionMode === "none", `${objectId}: collision mode is not none`, failures);
  record(before.inspection.colliderReady === false, `${objectId}: collider unexpectedly ready`, failures);
  record(before.inspection.collisionFaces === 0, `${objectId}: collision faces are nonzero`, failures);
  record(Object.values(unchanged).every(Boolean), `${objectId}: selection-only matrices moved`, failures);
  record(after.targets === before.targets, `${objectId}: collision target count changed`, failures);
  record(after.turns === before.turns, `${objectId}: double-click incremented turn count`, failures);
  record(after.prepared.prepared === false, `${objectId}: robot collision unexpectedly prepared`, failures);
  return {
    objectId,
    visualKind: before.inspection.visualKind,
    collisionMode: before.inspection.collisionMode,
    collisionFaces: before.inspection.collisionFaces,
    matricesUnchanged: unchanged,
    collisionTargetsUnchanged: after.targets === before.targets,
    turnCountUnchanged: after.turns === before.turns,
    robotCollision: { prepared: after.prepared.prepared, reason: after.prepared.reason },
  };
}

async function testRobotCollision(page, failures) {
  const prepared = await page.evaluate((id) => window.__prepareRobotCollisionTest(id), FRONT_ID);
  let step = null;
  if (prepared.prepared) {
    for (let attempt = 0; attempt < 5; attempt += 1) {
      step = await page.evaluate((id) => window.__stepRobotTowardObject(id, 0.65), FRONT_ID);
      if (step.blocked && step.lastCollisionObjectId === FRONT_ID) break;
    }
  }
  const passed = Boolean(
    prepared.prepared && step?.blocked && step?.lastCollisionObjectId === FRONT_ID,
  );
  record(passed, `front robot collision failed: ${prepared.reason || step?.lastCollisionObjectId || "no hit"}`, failures);
  return {
    objectId: FRONT_ID,
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

async function sampleFps(page) {
  const samples = [];
  for (let index = 0; index < 5; index += 1) {
    await page.waitForTimeout(1_050);
    samples.push(await page.evaluate(() => window.__visualPhysicsDemoState.fps));
  }
  const average = samples.reduce((sum, value) => sum + value, 0) / samples.length;
  return {
    samples,
    average: rounded(average, 2),
    minimum: Math.min(...samples),
    threshold: 18,
    passed: average >= 18,
  };
}

async function runQa({ runtimeUrl, manifestPath, reportPath, screenshotDir }) {
  const startedAt = new Date().toISOString();
  const startedMs = Date.now();
  const failures = [];
  const manifest = readJson(manifestPath);
  const manifestSha256Before = sha256File(manifestPath);
  const frontDefinition = (manifest.interactiveObjects || []).find((item) => item.id === FRONT_ID);
  const rearDefinitions = (manifest.interactiveObjects || []).filter((item) => REAR_IDS.includes(item.id));
  requireCondition(frontDefinition, `Manifest lacks ${FRONT_ID}`);
  requireCondition(rearDefinitions.length === 2, "Manifest must contain both rear selection-only pillows");
  requireCondition(frontDefinition.collision?.asset?.url, "Front unified GLB URL is missing");
  fs.mkdirSync(screenshotDir, { recursive: true });

  const runtime = new URL(runtimeUrl);
  const manifestPathname = new URL(runtime.searchParams.get("manifest"), runtime).pathname;
  const frontAssetUrl = new URL(frontDefinition.collision.asset.url, runtime);
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  let report;
  let desktopDiagnostics = null;
  let mobileDiagnostics = null;
  try {
    const desktopContext = await browser.newContext({
      viewport: { width: 1440, height: 900 },
      deviceScaleFactor: 1,
    });
    const page = await desktopContext.newPage();
    desktopDiagnostics = createPageDiagnostics(page, {
      label: "desktop",
      manifestPathname,
      frontAssetPathname: frontAssetUrl.pathname,
    });
    await page.goto(runtimeUrl, { waitUntil: "domcontentloaded", timeout: 60_000 });
    await waitForCandidateReady(page);
    await page.waitForTimeout(1_500);

    const initialState = await page.evaluate(() => window.__visualPhysicsDemoState);
    const collisionWorld = await page.evaluate(() => window.__inspectCollisionWorld());
    const front = await page.evaluate((id) => window.__inspectInteractiveObject(id), FRONT_ID);
    const frontCollision = collisionWorld.objects.find((item) => item.id === FRONT_ID);
    record(initialState.staticVisualCount === EXPECTED.staticVisuals,
      `static visual count=${initialState.staticVisualCount}`, failures);
    record(initialState.interactiveObjectGaussianCount === EXPECTED.existingObjectGaussians,
      `object Gaussian count=${initialState.interactiveObjectGaussianCount}`, failures);
    record(initialState.interactiveObjectRgbPointCount === EXPECTED.rearRgbPoints,
      `rear RGB count=${initialState.interactiveObjectRgbPointCount}`, failures);
    record(initialState.interactiveObjectMeshVertexCount === EXPECTED.frontMeshVertices,
      `front mesh vertices=${initialState.interactiveObjectMeshVertexCount}`, failures);
    record(initialState.visualCount === EXPECTED.totalVisualPrimitives,
      `total visual primitives=${initialState.visualCount}`, failures);
    record(initialState.colliderFaces === EXPECTED.staticColliderFaces,
      `static collider faces=${initialState.colliderFaces}`, failures);
    record(collisionWorld.objectColliderFaces === EXPECTED.objectColliderFaces,
      `object collider faces=${collisionWorld.objectColliderFaces}`, failures);
    record(initialState.interactiveObjectReadyCount === EXPECTED.visualObjects,
      `visual objects ready=${initialState.interactiveObjectReadyCount}`, failures);
    record(initialState.interactiveObjectCount === EXPECTED.visualObjects,
      `visual objects expected=${initialState.interactiveObjectCount}`, failures);
    record(collisionWorld.objectColliderReadyCount === EXPECTED.objectColliders,
      `object colliders ready=${collisionWorld.objectColliderReadyCount}`, failures);
    record(initialState.objectColliderCount === EXPECTED.objectColliders,
      `object colliders expected=${initialState.objectColliderCount}`, failures);
    record(collisionWorld.degradedCount === 0,
      `degraded collider count=${collisionWorld.degradedCount}`, failures);
    record(collisionWorld.errors.length === 0,
      `collider errors=${JSON.stringify(collisionWorld.errors)}`, failures);

    record(front?.visualKind === "mesh", `front visualKind=${front?.visualKind}`, failures);
    record(front?.unifiedVisualCollision === true, "front does not share one visual/collision root", failures);
    record(front?.collisionMode === "unified-glb", `front collisionMode=${front?.collisionMode}`, failures);
    record(front?.collisionTopology === "surface_bvh", `front topology=${front?.collisionTopology}`, failures);
    record(front?.collisionFaces === EXPECTED.frontColliderFaces,
      `front collision faces=${front?.collisionFaces}`, failures);
    record(front?.meshVertexCount === EXPECTED.frontMeshVertices,
      `front runtime mesh vertices=${front?.meshVertexCount}`, failures);
    record(front?.bvhMeshCount > 0, `front BVH mesh count=${front?.bvhMeshCount}`, failures);
    record(front?.pbrMaterialCount > 0, `front PBR material count=${front?.pbrMaterialCount}`, failures);
    record(front?.materialTypes?.some((type) => ["MeshStandardMaterial", "MeshPhysicalMaterial"].includes(type)),
      `front material types=${JSON.stringify(front?.materialTypes)}`, failures);
    record(front?.proxyWorldBounds == null, "front unexpectedly has proxy bounds", failures);
    record(front?.proxyMatrixWorld == null, "front unexpectedly has proxy matrix", failures);
    record(frontCollision?.volumePhysics === false,
      `front volumePhysics=${frontCollision?.volumePhysics}`, failures);
    record(frontCollision?.characterCollision === true,
      "front character collision is not enabled", failures);

    const locationResolution = await page.evaluate(() => window.__resolveSceneEntity("枕头在哪里"));
    const location = await page.evaluate(() => window.__askScene("枕头在哪里"));
    await page.waitForTimeout(1_300);
    const appearanceResolution = await page.evaluate(() => window.__resolveSceneEntity("枕头长什么样"));
    const appearance = await page.evaluate(() => window.__askScene("枕头长什么样"));
    await page.waitForTimeout(1_300);
    const queryState = await page.evaluate(() => window.__visualPhysicsDemoState);
    for (const [label, value] of [
      ["location resolver", locationResolution],
      ["location answer", location],
      ["appearance resolver", appearanceResolution],
      ["appearance answer", appearance],
    ]) {
      record(value?.status === "resolved", `${label} status=${value?.status}`, failures);
      record(value?.entityId === FRONT_ID, `${label} entity=${value?.entityId}`, failures);
    }
    record(location.focusEntityId === FRONT_ID, "location query did not focus front pillow", failures);
    record(appearance.focusEntityId === FRONT_ID, "appearance query did not focus front pillow", failures);
    record(queryState.selectedInteractiveObject === FRONT_ID,
      `selected object after query=${queryState.selectedInteractiveObject}`, failures);
    record(queryState.sceneQaResolvedObjectId === FRONT_ID,
      `scene QA resolved object=${queryState.sceneQaResolvedObjectId}`, failures);

    const desktopScreenshot = path.join(screenshotDir, "desktop-front-pillow.png");
    await page.screenshot({ path: desktopScreenshot, fullPage: true });
    const desktopCanvas = await inspectCanvasPixels(page);
    const desktopLayout = await inspectLayout(page);
    record(desktopCanvas.nonblank, `desktop canvas blank=${JSON.stringify(desktopCanvas)}`, failures);
    record(Object.values(desktopLayout.checks).every(Boolean),
      `desktop layout overlap=${JSON.stringify(desktopLayout.checks)}`, failures);

    const interactions = await testFrontInteraction(page, failures);
    const rearSelectionOnly = [];
    for (const objectId of REAR_IDS) {
      rearSelectionOnly.push(await testRearSelectionOnly(page, objectId, failures));
    }

    await page.evaluate((id) => window.__focusSceneEntity(id), FRONT_ID);
    await page.waitForTimeout(1_000);
    const freshFront = await page.evaluate((id) => window.__inspectInteractiveObject(id), FRONT_ID);
    const support = await inspectSupport(page, freshFront, manifest);
    record(support.relation?.verified === true,
      `front support relation=${JSON.stringify(support.relation)}`, failures);

    const interpenetrations = classifyInterpenetrations(
      await page.evaluate(() => window.__inspectSceneInterpenetrations()),
    );
    const conservativeSceneContact = interpenetrations.classifications.some((item) => (
      item.classification === "support_or_conservative_surface_contact"
      && item.colliderIds.includes(FRONT_ID)
      && item.colliderIds.includes("scene")
    ));
    support.evidence = {
      verifiedOnTopOfRelation: support.relation?.verified === true,
      conservativeFrontSceneSurfaceContact: conservativeSceneContact,
      downwardSceneProbeHit: Boolean(
        support.downwardSceneProbe.hit && support.downwardSceneProbe.clearsFrontSurface,
      ),
      multiSampleSceneProbeHit: support.multiSampleSceneProbe.passed,
    };
    support.status = support.evidence.verifiedOnTopOfRelation
      && (support.evidence.conservativeFrontSceneSurfaceContact
        || support.evidence.multiSampleSceneProbeHit)
      ? "recorded"
      : "insufficient";
    support.limitation = support.evidence.multiSampleSceneProbeHit
      ? null
      : "The footprint probe did not find at least three nearby scene-support hits across two distinct faces; a verified semantic relation alone is not sufficient.";
    record(support.status === "recorded", `front support status=${support.status}`, failures);
    record(interpenetrations.automatedStatus === "passed",
      "obvious front/object interpenetration crossed the documented threshold", failures);
    const robotCollision = await testRobotCollision(page, failures);
    const performance = await sampleFps(page);
    record(performance.passed, `average FPS=${performance.average} below ${performance.threshold}`, failures);
    await desktopDiagnostics.settle();
    record(desktopDiagnostics.frontAssetRequestCount === 1,
      `desktop front GLB requests=${desktopDiagnostics.frontAssetRequestCount}`, failures);
    record(desktopDiagnostics.manifestRequestCount === 1,
      `desktop manifest requests=${desktopDiagnostics.manifestRequestCount}`, failures);
    record(desktopDiagnostics.manifestResponseSha256.length === 1
      && desktopDiagnostics.manifestResponseSha256[0] === manifestSha256Before,
    `desktop served manifest hash=${desktopDiagnostics.manifestResponseSha256.join(",")}`, failures);
    await desktopContext.close();

    const mobileContext = await browser.newContext({
      viewport: { width: 390, height: 844 },
      deviceScaleFactor: 1,
      isMobile: true,
      hasTouch: true,
    });
    const mobilePage = await mobileContext.newPage();
    mobileDiagnostics = createPageDiagnostics(mobilePage, {
      label: "mobile",
      manifestPathname,
      frontAssetPathname: frontAssetUrl.pathname,
    });
    await mobilePage.goto(runtimeUrl, { waitUntil: "domcontentloaded", timeout: 60_000 });
    await waitForCandidateReady(mobilePage);
    const mobileAppearance = await mobilePage.evaluate(() => window.__askScene("枕头长什么样"));
    await mobilePage.waitForTimeout(1_300);
    const mobileScreenshot = path.join(screenshotDir, "mobile-390x844-front-pillow.png");
    await mobilePage.screenshot({ path: mobileScreenshot, fullPage: true });
    const mobileCanvas = await inspectCanvasPixels(mobilePage);
    const mobileLayout = await inspectLayout(mobilePage);
    const mobileChecks = {
      freshLoadReady: true,
      exactViewport: mobileLayout.viewport[0] === 390 && mobileLayout.viewport[1] === 844,
      exactCanvasCssSize: Math.abs(mobileLayout.canvas.width - 390) <= 0.5
        && Math.abs(mobileLayout.canvas.height - 844) <= 0.5,
      hudInsets: Math.abs(mobileLayout.hud.left - 12) <= 0.5
        && Math.abs(mobileLayout.hud.rightInset - 12) <= 0.5,
      noOverlapOrOverflow: Object.values(mobileLayout.checks).every(Boolean),
      canvasNonblank: mobileCanvas.nonblank,
      appearanceResolvedToFront: mobileAppearance.status === "resolved"
        && mobileAppearance.entityId === FRONT_ID,
    };
    record(Object.values(mobileChecks).every(Boolean),
      `mobile checks=${JSON.stringify(mobileChecks)}`, failures);
    await mobileDiagnostics.settle();
    record(mobileDiagnostics.frontAssetRequestCount === 1,
      `mobile fresh-load front GLB requests=${mobileDiagnostics.frontAssetRequestCount}`, failures);
    record(mobileDiagnostics.manifestRequestCount === 1,
      `mobile manifest requests=${mobileDiagnostics.manifestRequestCount}`, failures);
    record(mobileDiagnostics.manifestResponseSha256.length === 1
      && mobileDiagnostics.manifestResponseSha256[0] === manifestSha256Before,
    `mobile served manifest hash=${mobileDiagnostics.manifestResponseSha256.join(",")}`, failures);
    await mobileContext.close();

    const allDiagnostics = [desktopDiagnostics, mobileDiagnostics];
    const consoleErrors = allDiagnostics.flatMap((item) => (
      item.consoleMessages.filter((message) => message.type === "error")
        .map((message) => ({ viewport: item.label, ...message }))
    ));
    const pageErrors = allDiagnostics.flatMap((item) => (
      item.pageErrors.map((message) => ({ viewport: item.label, message }))
    ));
    const requestFailures = allDiagnostics.flatMap((item) => (
      item.requestFailures.map((failure) => ({ viewport: item.label, ...failure }))
    ));
    const httpErrors = allDiagnostics.flatMap((item) => (
      item.httpErrors.map((failure) => ({ viewport: item.label, ...failure }))
    ));
    record(consoleErrors.length === 0, `console errors=${JSON.stringify(consoleErrors)}`, failures);
    record(pageErrors.length === 0, `page errors=${JSON.stringify(pageErrors)}`, failures);
    record(requestFailures.length === 0,
      `request failures=${JSON.stringify(requestFailures)}`, failures);
    record(httpErrors.length === 0, `HTTP errors=${JSON.stringify(httpErrors)}`, failures);

    const manifestSha256After = sha256File(manifestPath);
    record(manifestSha256After === manifestSha256Before,
      "candidate manifest changed during browser QA", failures);
    report = {
      schemaVersion: 1,
      kind: "video2world.bedroom4_unified_pillow_browser_qa",
      status: failures.length ? "failed" : "candidate",
      automatedGate: failures.length ? "failed" : "passed",
      promotion: "not_performed",
      publishing: "not_performed",
      humanReview: "desktop_and_mobile_screenshots_required",
      startedAt,
      finishedAt: new Date().toISOString(),
      durationSeconds: rounded((Date.now() - startedMs) / 1000, 2),
      url: runtimeUrl,
      manifest: {
        path: manifestPath,
        sha256Before: manifestSha256Before,
        sha256After: manifestSha256After,
        unchanged: manifestSha256After === manifestSha256Before,
        desktopServedSha256: desktopDiagnostics.manifestResponseSha256,
        mobileServedSha256: mobileDiagnostics.manifestResponseSha256,
      },
      expected: EXPECTED,
      counts: {
        staticVisuals: initialState.staticVisualCount,
        existingObjectGaussians: initialState.interactiveObjectGaussianCount,
        rearRgbPoints: initialState.interactiveObjectRgbPointCount,
        frontMeshVertices: initialState.interactiveObjectMeshVertexCount,
        totalVisualPrimitives: initialState.visualCount,
        staticColliderFaces: initialState.colliderFaces,
        objectColliderFaces: collisionWorld.objectColliderFaces,
        visualObjectsReady: initialState.interactiveObjectReadyCount,
        visualObjectsExpected: initialState.interactiveObjectCount,
        objectCollidersReady: collisionWorld.objectColliderReadyCount,
        objectCollidersExpected: initialState.objectColliderCount,
        degradedColliders: collisionWorld.degradedCount,
      },
      front: {
        definition: {
          id: frontDefinition.id,
          asset: frontDefinition.collision.asset,
          topology: frontDefinition.collision.topology,
          limitations: frontDefinition.limitations || [],
        },
        inspection: front,
        collisionWorld: frontCollision,
        networkRequestsPerFreshPage: {
          desktop: desktopDiagnostics.frontAssetRequestCount,
          mobile: mobileDiagnostics.frontAssetRequestCount,
        },
      },
      sceneCognition: {
        location: { resolver: locationResolution, answer: location },
        appearance: { resolver: appearanceResolution, answer: appearance },
        selectedInteractiveObject: queryState.selectedInteractiveObject,
        resolvedSceneEntity: queryState.sceneQaResolvedObjectId,
      },
      interactions: {
        front: interactions,
        rearSelectionOnly,
        robotCollision,
      },
      support,
      sceneInterpenetrations: interpenetrations,
      collisionWorld,
      performance,
      responsive: {
        checks: mobileChecks,
        layout: mobileLayout,
        canvas: mobileCanvas,
        appearance: mobileAppearance,
        freshLoad: true,
      },
      desktop: { layout: desktopLayout, canvas: desktopCanvas },
      browserDiagnostics: {
        desktop: { ...desktopDiagnostics, settle: undefined },
        mobile: { ...mobileDiagnostics, settle: undefined },
        consoleErrors,
        pageErrors,
        requestFailures,
        httpErrors,
      },
      qualityPolicy: manifest.candidateBuild?.qualityPolicy || null,
      acceptedLimitations: [
        ...(frontDefinition.limitations || []),
        ...rearDefinitions.flatMap((item) => item.limitations || []),
      ],
      screenshots: { desktop: desktopScreenshot, mobile: mobileScreenshot },
      failures,
    };
  } catch (error) {
    report = {
      schemaVersion: 1,
      kind: "video2world.bedroom4_unified_pillow_browser_qa",
      status: "failed",
      automatedGate: "failed",
      promotion: "not_performed",
      publishing: "not_performed",
      startedAt,
      finishedAt: new Date().toISOString(),
      durationSeconds: rounded((Date.now() - startedMs) / 1000, 2),
      url: runtimeUrl,
      manifest: { path: manifestPath, sha256Before: manifestSha256Before },
      browserDiagnostics: {
        desktop: desktopDiagnostics ? { ...desktopDiagnostics, settle: undefined } : null,
        mobile: mobileDiagnostics ? { ...mobileDiagnostics, settle: undefined } : null,
      },
      failures: [...failures, error?.stack || String(error)],
    };
  } finally {
    await browser.close();
  }

  report = portableReport(report);
  writeJson(reportPath, report);
  console.log(JSON.stringify({
    status: report.status,
    automatedGate: report.automatedGate,
    manifestSha256: report.manifest?.sha256Before,
    report: portablePath(reportPath),
    screenshots: report.screenshots || null,
    failures: report.failures,
  }, null, 2));
  if (report.automatedGate !== "passed") process.exitCode = 1;
  return report;
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    console.log(usage());
    return;
  }
  const manifestPath = path.resolve(options.manifest || DEFAULT_MANIFEST);
  const reportPath = path.resolve(options.report || DEFAULT_REPORT);
  const screenshotDir = path.resolve(
    options["screenshot-dir"] || options["screenshots-dir"] || DEFAULT_SCREENSHOT_DIR,
  );
  requireCondition(fs.existsSync(manifestPath), `Candidate manifest does not exist: ${manifestPath}`);
  const runtimeUrl = buildRuntimeUrl(options.url || DEFAULT_URL, manifestPath);
  await runQa({ runtimeUrl, manifestPath, reportPath, screenshotDir });
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await main();
}
