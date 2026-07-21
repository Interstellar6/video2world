#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";

const scriptPath = fileURLToPath(import.meta.url);
const repoRoot = path.resolve(path.dirname(scriptPath), "..");
const publicRoot = path.join(repoRoot, "web/public");
const defaultCandidateRoot = path.join(
  repoRoot,
  "examples/bedroom4/completion/direct-trellis2-refit/candidate",
);
const DEFAULT_MANIFEST = path.join(defaultCandidateRoot, "manifest.json");
const DEFAULT_REPORT = path.join(defaultCandidateRoot, "browser-qa-report.json");
const DEFAULT_SCREENSHOT_DIR = path.join(defaultCandidateRoot, "browser-qa");
const DEFAULT_URL = "http://127.0.0.1:4181/";
const DIRECT_IDS = Object.freeze([
  "sam3_pillow_front",
  "sam3_pillow_left",
  "sam3_pillow_right",
]);
const LOGICAL_ANCESTOR_ID = "sam3_bed_01";
const EXPECTED_OBJECT_FACES = Object.freeze({
  sam3_pillow_front: 97082,
  sam3_pillow_left: 98230,
  sam3_pillow_right: 98226,
});

function parseArgs(argv) {
  const args = {};
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (!arg.startsWith("--")) throw new Error(`Unknown positional argument: ${arg}`);
    const key = arg.slice(2);
    const value = argv[index + 1];
    if (!value || value.startsWith("--")) throw new Error(`Missing value for --${key}`);
    args[key] = value;
    index += 1;
  }
  return args;
}

function sha256File(filePath) {
  return crypto.createHash("sha256").update(fs.readFileSync(filePath)).digest("hex");
}

function manifestUrlForPath(manifestPath) {
  const resolved = path.resolve(manifestPath);
  const relative = path.relative(publicRoot, resolved);
  if (!relative.startsWith("..") && !path.isAbsolute(relative)) {
    return `/${relative.split(path.sep).join("/")}`;
  }
  return `/@fs${resolved}`;
}

function runtimeUrl(baseUrl, manifestPath) {
  const url = new URL(baseUrl);
  url.searchParams.set("visual", "off");
  url.searchParams.set("collisionDebug", "on");
  url.searchParams.set("manifest", manifestUrlForPath(manifestPath));
  return url.href;
}

function record(condition, message, failures) {
  if (!condition) failures.push(message);
}

function matrixDelta(left, right) {
  if (!Array.isArray(left) || !Array.isArray(right) || left.length !== right.length) return null;
  return Math.max(...left.map((value, index) => Math.abs(Number(value) - Number(right[index]))));
}

async function captureViewport({ browser, label, url, screenshotDir }) {
  const viewport = label === "mobile" ? { width: 390, height: 844 } : { width: 1440, height: 900 };
  const page = await browser.newPage({ viewport });
  const events = [];
  page.on("pageerror", (error) => events.push({ type: "pageerror", text: error.message }));
  page.on("console", (message) => {
    if (["error", "warning"].includes(message.type())) {
      events.push({ type: message.type(), text: message.text().slice(0, 1000) });
    }
  });
  page.on("requestfailed", (request) => {
    events.push({
      type: "requestfailed",
      url: request.url(),
      text: request.failure()?.errorText || "unknown",
    });
  });

  await page.goto(url, { waitUntil: "domcontentloaded", timeout: 60_000 });
  await page.waitForFunction(
    () => window.__visualPhysicsDemoState?.colliderReady === true
      || window.__visualPhysicsDemoState?.error,
    null,
    { timeout: 120_000 },
  );
  await page.waitForFunction(
    () => window.__visualPhysicsDemoState?.objectColliderReadyCount >= 3
      || window.__visualPhysicsDemoState?.objectColliderLoadErrors?.length
      || window.__visualPhysicsDemoState?.error,
    null,
    { timeout: 180_000 },
  );
  await page.waitForTimeout(500);

  const runtime = await page.evaluate((ids) => {
    const inspections = Object.fromEntries(ids.map((id) => [
      id,
      window.__inspectInteractiveObject(id),
    ]));
    inspections.sam3_bed_01 = window.__inspectInteractiveObject("sam3_bed_01");
    return {
      state: window.__visualPhysicsDemoState,
      collisionWorld: window.__inspectCollisionWorld(),
      inspections,
      logicalAncestorScreenPoint: window.__getInteractiveObjectScreenPoint("sam3_bed_01"),
      logicalAncestorFocus: window.__focusSceneEntity("sam3_bed_01"),
    };
  }, DIRECT_IDS);

  const interaction = {};
  for (const objectId of DIRECT_IDS) {
    interaction[objectId] = await page.evaluate((id) => {
      const findVisibleObjectPoint = () => {
        const seed = window.__getInteractiveObjectScreenPoint(id);
        const canvas = document.querySelector("canvas");
        const rect = canvas?.getBoundingClientRect();
        if (!seed || !rect) return { seed, point: null, hit: null };
        const inspect = (x, y) => {
          if (x < rect.left || x > rect.right || y < rect.top || y > rect.bottom) return null;
          if (document.elementFromPoint(x, y) !== canvas) return null;
          const hit = window.__inspectInteractiveObjectHitAt(x, y);
          return hit?.objectId === id ? { point: { x, y }, hit } : null;
        };
        for (let radius = 0; radius <= 240; radius += 12) {
          const count = radius === 0 ? 1 : Math.max(8, Math.ceil((Math.PI * 2 * radius) / 12));
          for (let index = 0; index < count; index += 1) {
            const angle = count === 1 ? 0 : (Math.PI * 2 * index) / count;
            const result = inspect(seed.x + Math.cos(angle) * radius, seed.y + Math.sin(angle) * radius);
            if (result) return { seed, ...result };
          }
        }
        const grid = [];
        for (let y = rect.top + 8; y < rect.bottom; y += 16) {
          for (let x = rect.left + 8; x < rect.right; x += 16) {
            grid.push({ x, y, distance: Math.hypot(x - seed.x, y - seed.y) });
          }
        }
        grid.sort((left, right) => left.distance - right.distance);
        for (const candidate of grid) {
          const result = inspect(candidate.x, candidate.y);
          if (result) return { seed, ...result, search: "full-canvas-grid" };
        }
        return { seed, point: { x: seed.x, y: seed.y }, hit: window.__inspectInteractiveObjectHitAt(seed.x, seed.y) };
      };
      const before = window.__inspectInteractiveObject(id);
      const focus = window.__focusSceneEntity(id);
      const visiblePoint = findVisibleObjectPoint();
      const yaw = window.__setInteractiveObjectYaw(id, 37);
      const afterYaw = window.__inspectInteractiveObject(id);
      window.__setInteractiveObjectYaw(id, 0);
      const reset = window.__inspectInteractiveObject(id);
      const robotPreparation = window.__prepareRobotCollisionTest(id);
      const robotStep = robotPreparation.prepared
        ? window.__stepRobotTowardObject(id, 1.1)
        : null;
      return {
        before,
        focus,
        visiblePoint,
        yaw,
        afterYaw,
        reset,
        robotPreparation,
        robotStep,
      };
    }, objectId);
  }

  fs.mkdirSync(screenshotDir, { recursive: true });
  const screenshotPath = path.join(screenshotDir, `${label}.png`);
  await page.screenshot({ path: screenshotPath, fullPage: false });
  await page.close();
  return { label, viewport, events, runtime, interaction, screenshotPath };
}

function evaluateCapture(capture) {
  const failures = [];
  const state = capture.runtime.state;
  const collisionWorld = capture.runtime.collisionWorld;
  record(!state.error, `${capture.label}: state error=${state.error}`, failures);
  record(state.colliderReady === true, `${capture.label}: static collider not ready`, failures);
  record(state.colliderFaces === 1351454, `${capture.label}: colliderFaces=${state.colliderFaces}`, failures);
  record(state.objectColliderReadyCount === 3,
    `${capture.label}: objectColliderReadyCount=${state.objectColliderReadyCount}`, failures);
  record(state.objectColliderFaces === 293538,
    `${capture.label}: objectColliderFaces=${state.objectColliderFaces}`, failures);
  record(state.objectColliderDegradedCount === 0,
    `${capture.label}: degradedCount=${state.objectColliderDegradedCount}`, failures);
  record((state.objectColliderLoadErrors || []).length === 0,
    `${capture.label}: object collider load errors`, failures);
  record((capture.events || []).length === 0,
    `${capture.label}: browser events=${JSON.stringify(capture.events)}`, failures);
  record(collisionWorld?.targetCount === 4,
    `${capture.label}: collision target count=${collisionWorld?.targetCount}`, failures);

  for (const objectId of DIRECT_IDS) {
    const inspection = capture.runtime.inspections[objectId];
    record(inspection?.visualReady === true, `${capture.label}/${objectId}: visual not ready`, failures);
    record(inspection?.colliderReady === true, `${capture.label}/${objectId}: collider not ready`, failures);
    record(inspection?.visualKind === "mesh",
      `${capture.label}/${objectId}: visualKind=${inspection?.visualKind}`, failures);
    record(inspection?.collisionMode === "unified-glb",
      `${capture.label}/${objectId}: collisionMode=${inspection?.collisionMode}`, failures);
    record(inspection?.collisionTopology === "surface_bvh",
      `${capture.label}/${objectId}: topology=${inspection?.collisionTopology}`, failures);
    record(inspection?.collisionFaces === EXPECTED_OBJECT_FACES[objectId],
      `${capture.label}/${objectId}: faces=${inspection?.collisionFaces}`, failures);
    record(inspection?.parentObjectId === LOGICAL_ANCESTOR_ID,
      `${capture.label}/${objectId}: parent=${inspection?.parentObjectId}`, failures);
    record(inspection?.unifiedVisualCollision === true,
      `${capture.label}/${objectId}: visual/collision root mismatch`, failures);
    record(inspection?.bvhMeshCount > 0,
      `${capture.label}/${objectId}: no BVH mesh`, failures);
    record(inspection?.pbrMaterialCount > 0,
      `${capture.label}/${objectId}: no PBR material`, failures);
    record(inspection?.proxyWorldBounds == null,
      `${capture.label}/${objectId}: proxy bounds present`, failures);

    const test = capture.interaction[objectId];
    record(test?.focus?.focused === true,
      `${capture.label}/${objectId}: focus failed`, failures);
    record(test?.visiblePoint?.hit?.objectId === objectId,
      `${capture.label}/${objectId}: pointer hit=${test?.visiblePoint?.hit?.objectId}`, failures);
    record(test?.visiblePoint?.hit?.colliderMode === "unified-glb",
      `${capture.label}/${objectId}: pointer collider=${test?.visiblePoint?.hit?.colliderMode}`, failures);
    record(test?.yaw?.updated === true,
      `${capture.label}/${objectId}: yaw hook failed`, failures);
    record(test?.robotPreparation?.prepared === true,
      `${capture.label}/${objectId}: robot collision prep=${test?.robotPreparation?.reason}`, failures);
    record(test?.robotStep?.attempted === true,
      `${capture.label}/${objectId}: robot step not attempted`, failures);
    record(test?.robotStep?.blocked === true,
      `${capture.label}/${objectId}: robot was not blocked`, failures);
    record(test?.robotStep?.lastCollisionKind === "object",
      `${capture.label}/${objectId}: robot collision kind=${test?.robotStep?.lastCollisionKind}`, failures);
    record(test?.robotStep?.lastCollisionObjectId === objectId,
      `${capture.label}/${objectId}: robot collision object=${test?.robotStep?.lastCollisionObjectId}`, failures);
    for (const [key, before, after] of [
      ["group", test?.before?.groupMatrixWorld, test?.afterYaw?.groupMatrixWorld],
      ["visual", test?.before?.splatMatrixWorld, test?.afterYaw?.splatMatrixWorld],
      ["collision", test?.before?.collisionMatrixWorld, test?.afterYaw?.collisionMatrixWorld],
    ]) {
      const delta = matrixDelta(before, after);
      record(Number.isFinite(delta) && delta > 1e-4,
        `${capture.label}/${objectId}: yaw did not move ${key}`, failures);
    }
  }

  const ancestor = capture.runtime.inspections[LOGICAL_ANCESTOR_ID];
  record(ancestor?.logicalHierarchyOnly === true, `${capture.label}: bed is not logical-only`, failures);
  record(ancestor?.visualReady === false, `${capture.label}: logical bed visualReady=true`, failures);
  record(ancestor?.colliderReady === false, `${capture.label}: logical bed colliderReady=true`, failures);
  record(ancestor?.collisionMode === "none",
    `${capture.label}: logical bed collisionMode=${ancestor?.collisionMode}`, failures);
  record(capture.runtime.logicalAncestorScreenPoint == null,
    `${capture.label}: logical bed has screen point`, failures);
  record(capture.runtime.logicalAncestorFocus?.focused === false,
    `${capture.label}: logical bed accepted focus`, failures);

  return failures;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const manifestPath = path.resolve(args.manifest || DEFAULT_MANIFEST);
  const reportPath = path.resolve(args.report || DEFAULT_REPORT);
  const screenshotDir = path.resolve(args["screenshot-dir"] || DEFAULT_SCREENSHOT_DIR);
  const url = runtimeUrl(args.url || DEFAULT_URL, manifestPath);
  if (!fs.existsSync(manifestPath)) throw new Error(`Manifest does not exist: ${manifestPath}`);

  const browser = await chromium.launch({ channel: "chrome", headless: true });
  try {
    const captures = [];
    for (const label of ["desktop", "mobile"]) {
      captures.push(await captureViewport({ browser, label, url, screenshotDir }));
    }
    const failures = captures.flatMap(evaluateCapture);
    const report = {
      kind: "video2world.direct_trellis_scene_candidate_browser_qa",
      status: failures.length ? "failed" : "passed",
      createdAt: new Date().toISOString(),
      runtimeUrl: url,
      manifest: {
        path: manifestPath,
        sha256: sha256File(manifestPath),
      },
      checks: {
        staticColliderFaces: 1351454,
        unifiedObjectFaces: EXPECTED_OBJECT_FACES,
        logicalAncestorId: LOGICAL_ANCESTOR_ID,
      },
      failures,
      captures,
      claimScope: "Local browser QA for the direct TRELLIS2 three-pillow candidate. It does not promote, carve, clean, or publish the manifest.",
    };
    fs.mkdirSync(path.dirname(reportPath), { recursive: true });
    fs.writeFileSync(reportPath, `${JSON.stringify(report, null, 2)}\n`);
    console.log(JSON.stringify({
      status: report.status,
      reportPath,
      screenshotDir,
      failures,
    }, null, 2));
    if (failures.length) process.exitCode = 1;
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
